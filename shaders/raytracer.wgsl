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
}

@group(0) @binding(0) var<uniform> rt_data: RtData;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> faces: array<Face>;
@group(0) @binding(4) var<storage, read_write> accum_buffer: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> brdf_data: array<f32>;

override wg_x: u32 = 8u;
override wg_y: u32 = 8u;

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(rt_data.resolution);
    let stride = rt_data.stride;

    let tile_seed = init_rng(id.xy, rt_data.frame);
    let offset = vec2<u32>(
        pcg_hash(tile_seed) % stride,
        pcg_hash(tile_seed ^ 0x9e3779b9u) % stride
    );

    let pixel = id.xy * stride + offset;
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    // Perform raytracing
    var jitter_rng = init_rng(pixel, rt_data.frame);
    let jitter = vec2<f32>(rand_f32(&jitter_rng), rand_f32(&jitter_rng));
    let ray = generate_ray(pixel.xy, rt_data.resolution, jitter);
    let new_sample = raytrace(ray);

    // Incremental running average using per-pixel sample count in .w
    let pixel_idx = pixel.y * dims.x + pixel.x;
    let prev = accum_buffer[pixel_idx];
    let sample_count = prev.w + 1.0;

    let accumulated = mix(prev.rgb, new_sample, 1.0 / sample_count);
    accum_buffer[pixel_idx] = vec4<f32>(accumulated, sample_count);

    textureStore(output_tex, vec2<i32>(pixel), vec4<f32>(accumulated, 1.0));
}

fn generate_ray(pixel: vec2<u32>, resolution: vec2<f32>, jitter: vec2<f32>) -> Ray {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / resolution;

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

const NO_HIT_COLOR = vec3<f32>(0.0, 0.0, 0.0);

// Returns color at the place the ray hit.
fn raytrace(ray: Ray) -> vec3<f32> {
    var closest_hit = NO_RAY_HIT;
    var closest_face_index = 1u;
    var hit_normal = vec3<f32>(0.0, 0.0, 0.0);
    let face_count = arrayLength(&faces);

    for (var i = 0u; i < face_count; i++) {
        let face = faces[i];
        let hit = ray_face_hit(ray, face);

        if hit.distance < closest_hit.distance {
            hit_normal = interpolated_normal(face, hit.weights);
            if dot(hit_normal, ray.direction) < 0.0 {
                closest_hit = hit;
                closest_face_index = i;
            }
        }
    }

    // No hit
    if !closest_hit.is_hit {
        return NO_HIT_COLOR;
    }

    let face = faces[closest_face_index];

    let light = rt_data.light_position;
    let to_light = light - closest_hit.point;
    let light_dist = length(to_light);
    let to_light_dir = to_light / light_dist;

    // If the surface faces away from the light
    let cos_theta_i = dot(to_light_dir, hit_normal);
    if cos_theta_i <= 0.0 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // Slight offset to avoid self-intersection
    let geom_normal = normalize(cross(face.v1.xyz - face.v0.xyz, face.v2.xyz - face.v0.xyz));
    let offset_sign = select(-1.0, 1.0, dot(geom_normal, to_light_dir) > 0.0);
    let shadow_origin = closest_hit.point + geom_normal * (offset_sign * 1e-4);

    // If point is in shadow
    if is_occluded(shadow_origin, to_light_dir, light_dist) {
        return vec3<f32>(0.0);
    }

    // BRDF stuff
    let frame = make_tangent_frame(hit_normal, vec3<f32>(1.0, 0.0, 0.0));
    let sph_i = dir_to_spherical(to_light_dir, frame);
    let sph_v = dir_to_spherical(-ray.direction, frame);
    let brdf_value = lookup_brdf_val(sph_i.x, sph_i.y, sph_v.x, sph_v.y);
    let light_intensity = vec3<f32>(2.0);
    let attenuation = light_intensity / (light_dist * light_dist);

    return brdf_value * cos_theta_i * attenuation;
}

// Check if point is in shadow
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
