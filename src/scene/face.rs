use crate::math::Vec4;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Face {
    pub triangle: [Vec4; 3],
    pub normals: [Vec4; 3],
}
