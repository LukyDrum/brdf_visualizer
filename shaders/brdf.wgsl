const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = 1.57079632679489661923;
const RAD_TO_DEG: f32 = 57.2957795130823208768;

const BRDF_STEP_T: f32 = 15.0;
const BRDF_STEP_P: f32 = 7.5;
const BRDF_NTI: u32 = 6u;
const BRDF_NTV: u32 = 6u;
const BRDF_NPI: u32 = 48u;
const BRDF_NPV: u32 = 48u;
const BRDF_NC: u32 = 288u;
const BRDF_PLANE_STRIDE: u32 = 82944u;

struct TangentFrame {
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
};

fn make_tangent_frame(n: vec3<f32>, ref_tangent: vec3<f32>) -> TangentFrame {
    var ortho = ref_tangent - n * dot(ref_tangent, n);
    if dot(ortho, ortho) < 1e-6 {
        let fallback = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.x) > 0.9);
        ortho = fallback - n * dot(fallback, n);
    }
    let t = normalize(ortho);
    let b = cross(n, t);
    return TangentFrame(t, b, n);
}

fn dir_to_spherical(dir: vec3<f32>, frame: TangentFrame) -> vec2<f32> {
    let theta = acos(clamp(dot(dir, frame.n), 0.0, 1.0));
    let phi = atan2(dot(dir, frame.b), dot(dir, frame.t));
    return vec2<f32>(theta, phi);
}

// This assumes the existence of `brdf_data` buffer in the main file
fn lookup_brdf_val(theta_i: f32, phi_i: f32, theta_v: f32, phi_v: f32) -> vec3<f32> {
    if theta_i > HALF_PI || theta_v > HALF_PI || theta_i < 0.0 || theta_v < 0.0 {
        return vec3<f32>(0.0);
    }

    let ti_deg = theta_i * RAD_TO_DEG;
    let tv_deg = theta_v * RAD_TO_DEG;

    let pi_raw = phi_i * RAD_TO_DEG;
    let pv_raw = phi_v * RAD_TO_DEG;
    let pi_deg = pi_raw - 360.0 * floor(pi_raw / 360.0);
    let pv_deg = pv_raw - 360.0 * floor(pv_raw / 360.0);

    let iti0 = min(u32(floor(ti_deg / BRDF_STEP_T)), BRDF_NTI - 2u);
    let iti1 = iti0 + 1u;
    let t_ti = (ti_deg - f32(iti0) * BRDF_STEP_T) / BRDF_STEP_T;
    let wti = vec2<f32>(1.0 - t_ti, t_ti);

    let itv0 = min(u32(floor(tv_deg / BRDF_STEP_T)), BRDF_NTV - 2u);
    let itv1 = itv0 + 1u;
    let t_tv = (tv_deg - f32(itv0) * BRDF_STEP_T) / BRDF_STEP_T;
    let wtv = vec2<f32>(1.0 - t_tv, t_tv);

    let ipi0 = u32(floor(pi_deg / BRDF_STEP_P)) % BRDF_NPI;
    let ipi1 = (ipi0 + 1u) % BRDF_NPI;
    let t_pi = fract(pi_deg / BRDF_STEP_P);
    let wpi = vec2<f32>(1.0 - t_pi, t_pi);

    let ipv0 = u32(floor(pv_deg / BRDF_STEP_P)) % BRDF_NPV;
    let ipv1 = (ipv0 + 1u) % BRDF_NPV;
    let t_pv = fract(pv_deg / BRDF_STEP_P);
    let wpv = vec2<f32>(1.0 - t_pv, t_pv);

    let iti = array<u32, 2>(iti0, iti1);
    let itv = array<u32, 2>(itv0, itv1);
    let ipi = array<u32, 2>(ipi0, ipi1);
    let ipv = array<u32, 2>(ipv0, ipv1);

    var rgb = vec3<f32>(0.0);

    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            for (var k = 0u; k < 2u; k++) {
                for (var l = 0u; l < 2u; l++) {
                    let weight = wti[i] * wtv[j] * wpi[k] * wpv[l];
                    let idx = BRDF_NC * (BRDF_NPI * iti[i] + ipi[k])
                            + (BRDF_NPV * itv[j] + ipv[l]);

                    let sample_rgb = vec3<f32>(
                        brdf_data[idx],
                        brdf_data[idx + BRDF_PLANE_STRIDE],
                        brdf_data[idx + 2u * BRDF_PLANE_STRIDE],
                    );
                    rgb += sample_rgb * weight;
                }
            }
        }
    }

    return max(rgb, vec3<f32>(0.0));
}
