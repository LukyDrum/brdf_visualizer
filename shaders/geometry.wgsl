struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct Face {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
}

struct RayFaceHit {
    is_hit: bool,
    point: vec3<f32>,
    weights: vec2<f32>,
    distance: f32,
}

const EPSILON = 0.000001;
const NO_RAY_HIT = RayFaceHit(false, vec3<f32>(0.0, 0.0, 0.0), vec2<f32>(0.0, 0.0), 100000000000.0);

fn ray_face_hit(
    ray: Ray,
    face: Face,
) -> RayFaceHit {
    // AB
    let e1 = face.v1.xyz - face.v0.xyz;
    // AC
    let e2 = face.v2.xyz - face.v0.xyz;

    let ray_cross_e2 = cross(ray.direction, e2);
    let det = dot(e1, ray_cross_e2);

    // Check if the ray is parallel to the triangle
    if abs(det) < EPSILON {
        return NO_RAY_HIT;
    }

    let inv_det = 1.0 / det;
    let origin_to_a = ray.origin - face.v0.xyz;
    let u = inv_det * dot(origin_to_a, ray_cross_e2);
    if u < 0.0 || u > 1.0 {
        return NO_RAY_HIT;
    }

    let s_cross_e1 = cross(origin_to_a, e1);
    let v = inv_det * dot(ray.direction, s_cross_e1);
    if v < 0.0 || u + v > 1.0 {
        return NO_RAY_HIT;
    }

    let t = inv_det * dot(e2, s_cross_e1);

    if t > EPSILON {
        // ray intersection
        let intersection_point = ray.origin + ray.direction * t;

        return RayFaceHit(
            true, intersection_point, vec2<f32>(u, v), abs(t),
        );
    } else {
       return NO_RAY_HIT;
    }
}
