use crate::math::Vec4;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec4,
    pub forward: Vec4,
    pub up: Vec4,
    pub fov: f32,
}
