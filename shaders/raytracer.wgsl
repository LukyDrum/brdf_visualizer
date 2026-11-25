// #include "./geometry.wgsl"
// #include "./rand.wgsl"
// #include "./brdf.wgsl"

struct RtData {
    resolution: vec2<f32>,
    time: f32,
    frame: u32,
    light_position: vec3<f32>,
    stride: u32,
};

struct Camera {
    position: vec4<f32>,
    forward: vec4<f32>, // .w = half_height
    right: vec4<f32>,   // .w = aspect
    up: vec4<f32>,
};

struct BvhNode {
    aabb_min: vec3<f32>,
    left_first: u32,
    aabb_max: vec3<f32>,
    tri_count: u32,
};

@group(0) @binding(0) var<uniform> rt_data: RtData;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> faces: array<Face>;
@group(0) @binding(4) var<storage, read_write> accum_buffer: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> brdf_data: array<f32>;
@group(0) @binding(6) var<storage, read> bvh_nodes: array<BvhNode>;

override wg_x: u32 = 8u;
override wg_y: u32 = 8u;

const MAX_BOUNCES: u32 = 3u;
const SKY_COLOR: vec3<f32> = vec3<f32>(0.08, 0.10, 0.15);
const LIGHT_INTENSITY: vec3<f32> = vec3<f32>(2.0);

const STRIDE_PATTERN_2 = array<u32, 4>(1, 2, 0, 3);
const STRIDE_PATTERN_4 = array<u32, 16>(
    0, 10, 2, 8,
    5, 15, 7, 13,
    1, 11, 3, 9,
    4, 14, 6, 12,
);

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(rt_data.resolution);
    let stride = rt_data.stride;

    let tile_hash = pcg_hash(id.x + pcg_hash(id.y));
    let index_into_pattern = (rt_data.frame + tile_hash) % (stride * stride);

    var offset: vec2<u32>;
    if stride == 1 {
        offset = vec2<u32>(0, 0);
    }
    else if stride == 2 {
        let offset_index = STRIDE_PATTERN_2[index_into_pattern];
        offset = vec2<u32>(offset_index % 2, offset_index / 2);
    }
    else if stride == 4 {
        let offset_index = STRIDE_PATTERN_4[index_into_pattern];
        offset = vec2<u32>(offset_index % 4, offset_index / 4);
    }
    else {
        let tile_seed = init_rng(id.xy, rt_data.frame);
        offset = vec2<u32>(
            pcg_hash(tile_seed) % stride,
            pcg_hash(tile_seed ^ 0x9e3779b9u) % stride
        );
    }

    let pixel = id.xy * stride + offset;
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    var rng = init_rng(pixel, rt_data.frame);
    let jitter = vec2<f32>(rand_f32(&rng), rand_f32(&rng));
    let ray = generate_ray(pixel, rt_data.resolution, jitter);
    let new_sample = trace_path(ray, &rng);

    let pixel_idx = pixel.y * dims.x + pixel.x;
    let prev = accum_buffer[pixel_idx];
    let sample_count = prev.w + 1.0;

    let accumulated = mix(prev.rgb, new_sample, 1.0 / sample_count);
    accum_buffer[pixel_idx] = vec4<f32>(accumulated, sample_count);

    let display_color = aces_tonemap(accumulated);
    textureStore(output_tex, vec2<i32>(pixel), vec4<f32>(display_color, 1.0));
}

fn generate_ray(pixel: vec2<u32>, resolution: vec2<f32>, jitter: vec2<f32>) -> Ray {
    let uv = (vec2<f32>(pixel) + jitter) / resolution;

    let ndc = vec2<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0
    );

    let half_height = camera.forward.w;
    let aspect = camera.right.w;

    let dir = normalize(
        camera.forward.xyz
        + camera.right.xyz * (ndc.x * aspect * half_height)
        + camera.up.xyz    * (ndc.y * half_height)
    );

    return Ray(camera.position.xyz, dir);
}

fn trace_path(initial_ray: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = initial_ray;
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    let face_count = arrayLength(&faces);

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        var closest_face_index = 0u;
        var closest_hit = trace_bvh(ray, &closest_face_index);

        // Ray missed
        if !closest_hit.is_hit {
            radiance += throughput * SKY_COLOR;
            break;
        }

        let face = faces[closest_face_index];
        var hit_normal = interpolated_normal(face, closest_hit.weights);
        if dot(hit_normal, ray.direction) > 0.0 {
            hit_normal = -hit_normal;
        }

        // Offset ray origins to avoid self-intersection
        let geom_normal = normalize(cross(face.v1.xyz - face.v0.xyz, face.v2.xyz - face.v0.xyz));
        let face_forward_geom = select(-geom_normal, geom_normal, dot(geom_normal, hit_normal) > 0.0);
        let safe_origin = closest_hit.point + face_forward_geom * 1e-4;

        let view_dir = -ray.direction;
        let frame = make_tangent_frame(hit_normal, vec3<f32>(1.0, 0.0, 0.0));
        let sph_v = dir_to_spherical(view_dir, frame);

        // Direct light
        let to_light = rt_data.light_position - closest_hit.point;
        let light_dist = length(to_light);
        let to_light_dir = to_light / light_dist;
        let cos_theta_l = dot(to_light_dir, hit_normal);

        if cos_theta_l > 0.0 && !is_occluded(safe_origin, to_light_dir, light_dist) {
            let sph_l = dir_to_spherical(to_light_dir, frame);
            let fr_direct = lookup_brdf_val(sph_l.x, sph_l.y, sph_v.x, sph_v.y);
            let attenuation = LIGHT_INTENSITY / max(light_dist * light_dist, 0.25);

            var direct = throughput * fr_direct * cos_theta_l * attenuation;
            if bounce > 0u {
                direct = min(direct, vec3<f32>(1.5)); // Suppress indirect fireflies
            }

            radiance += direct;
        }

        // Choose random bounce in hemishpere
        let bounce_dir = sample_cosine_hemisphere(rand_f32(rng), rand_f32(rng), frame);
        if dot(face_forward_geom, bounce_dir) <= 0.0 {
            break;
        }

        let sph_i = dir_to_spherical(bounce_dir, frame);
        let fr_bounce = lookup_brdf_val(sph_i.x, sph_i.y, sph_v.x, sph_v.y);

        throughput *= min(fr_bounce * PI, vec3<f32>(1.0));

        // Russian roullete based on throughput
        if bounce >= 2u {
            let survive_prob = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if rand_f32(rng) > survive_prob {
                break;
            }

            throughput /= survive_prob;
        }

        ray = Ray(safe_origin, bounce_dir);
    }

    // Clamp extreme fireflies from sharp measured BRDF specular peaks
    return min(radiance, vec3<f32>(10.0));
}

fn is_occluded(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let shadow_ray = Ray(origin, dir);
    let face_count = arrayLength(&faces);
    for (var i = 0u; i < face_count; i++) {
        let hit = ray_face_hit(shadow_ray, faces[i]);
        if hit.is_hit && hit.distance > 1e-4 && hit.distance < max_dist {
            return true;
        }
    }
    return false;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Branchless slab test: returns distance to AABB entry, or 1e30 if missed
fn intersect_aabb(ray_origin: vec3<f32>, inv_dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, max_dist: f32) -> f32 {
    let t0 = (bmin - ray_origin) * inv_dir;
    let t1 = (bmax - ray_origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let t_exit  = min(min(tmax.x, tmax.y), min(tmax.z, max_dist));

    return select(1e30, t_enter, t_enter <= t_exit);
}

fn trace_bvh(ray: Ray, out_face_idx: ptr<function, u32>) -> RayFaceHit {
    var closest_hit = NO_RAY_HIT;
    let inv_dir = 1.0 / ray.direction;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    var node_idx = 0u;

    loop {
        let node = bvh_nodes[node_idx];

        if node.tri_count > 0u {
            // Leaf node: test the contiguous slice of faces
            let end = node.left_first + node.tri_count;
            for (var i = node.left_first; i < end; i++) {
                let hit = ray_face_hit(ray, faces[i]);
                if hit.is_hit && hit.distance > 1e-4 && hit.distance < closest_hit.distance {
                    closest_hit = hit;
                    *out_face_idx = i;
                }
            }
            if stack_ptr == 0u { break; }
            stack_ptr -= 1u;
            node_idx = stack[stack_ptr];
        } else {
            // Internal node: test both children and visit the closer one first
            let left_idx = node_idx + 1u;
            let right_idx = node.left_first;

            let dist_l = intersect_aabb(ray.origin, inv_dir, bvh_nodes[left_idx].aabb_min, bvh_nodes[left_idx].aabb_max, closest_hit.distance);
            let dist_r = intersect_aabb(ray.origin, inv_dir, bvh_nodes[right_idx].aabb_min, bvh_nodes[right_idx].aabb_max, closest_hit.distance);

            var near_idx = left_idx;
            var far_idx = right_idx;
            var dist_near = dist_l;
            var dist_far = dist_r;

            if dist_r < dist_l {
                near_idx = right_idx;
                far_idx = left_idx;
                dist_near = dist_r;
                dist_far = dist_l;
            }

            if dist_near >= closest_hit.distance {
                // Missed both boxes
                if stack_ptr == 0u { break; }
                stack_ptr -= 1u;
                node_idx = stack[stack_ptr];
            } else {
                node_idx = near_idx;
                if dist_far < closest_hit.distance {
                    stack[stack_ptr] = far_idx;
                    stack_ptr += 1u;
                }
            }
        }
    }

    return closest_hit;
}
