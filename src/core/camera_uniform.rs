use crate::{math::Vec4, scene::Camera};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CameraUniform {
    pub position: [f32; 4], // xyz = eye position, w = unused
    pub forward: [f32; 4],  // xyz = unit forward, w = half_height (tan(vfov/2))
    pub right: [f32; 4],    // xyz = unit right,   w = aspect_ratio
    pub up: [f32; 4],       // xyz = unit up,      w = unused
}

impl CameraUniform {
    pub fn from_camera(camera: Camera, (width, height): (u32, u32)) -> Self {
        let Camera {
            position,
            forward,
            up,
            fov,
        } = camera;

        let right: Vec4 = forward.cross(up);

        let half_height = (fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height.max(1) as f32;

        Self {
            position: position.raw_vec4(),
            forward: forward.raw_vec4_with_w(half_height),
            right: right.raw_vec4_with_w(aspect),
            up: up.raw_vec4(),
        }
    }
}
