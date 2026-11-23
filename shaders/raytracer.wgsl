// #include "./geometry.wgsl"

struct RtData {
    resolution: vec2<f32>,
    time: f32,
    frame: u32,
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

override wg_x: u32 = 8u;
override wg_y: u32 = 8u;

@compute @workgroup_size(wg_x, wg_y, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Validate position
    let dims = vec2<u32>(rt_data.resolution);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    // Perform raytracing
    let ray = generate_ray(id.xy, rt_data.resolution);
    let color = raytrace(ray);

    textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}

fn generate_ray(pixel: vec2<u32>, resolution: vec2<f32>) -> Ray {
    // 1. Pixel center (+ 0.5) normalized to [0, 1]
    // (For anti-aliasing in path tracing, replace vec2(0.5) with a random offset in [0, 1))
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / resolution;

    let ndc = vec2<f32>(
        1.0 - uv.x * 2.0,
        1.0 - uv.y * 2.0
    );

    let half_height = camera.forward.w;
    let aspect = camera.right.w;

    // 3. Offset from the center of the image plane at distance 1.0
    let dir = normalize(
        camera.forward.xyz
        + camera.right.xyz * (ndc.x * aspect * half_height)
        + camera.up.xyz    * (ndc.y * half_height)
    );

    return Ray(camera.position.xyz, dir);
}

const NO_HIT_COLOR = vec3<f32>(0.0, 0.0, 0.0);
const BACKFACE_CULLING = true;

fn raytrace(ray: Ray) -> vec3<f32> {
    var closest_hit = NO_RAY_HIT;
    var closest_face_index = 1u;
    let face_count = arrayLength(&faces);

    for (var i = 0u; i < face_count; i++) {
        let face = faces[i];
        let hit = ray_face_hit(ray, face);

        if hit.distance < closest_hit.distance {
            let normal = interpolated_normal(face, hit.weights);
            if dot(normal, ray.direction) < 0.0 || !BACKFACE_CULLING {
                closest_hit = hit;
                closest_face_index = i;
            }
        }
    }

    // No hit
    if !closest_hit.is_hit {
        return NO_HIT_COLOR;
    }

    let depth = clamp(closest_hit.point.z / 1.0, 0.0, 1.0);

    return vec3<f32>(depth, depth, depth);
}
