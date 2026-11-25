mod camera;
mod face;
mod material;
mod orbiter;
mod bvh;

pub use camera::{Camera, OrbitCamera};
pub use face::Face;
pub use material::{BrdfMaterial, Material, UNNAMED_MATERIAL};
pub use orbiter::Orbiter;
pub use bvh::{BvhNode, GpuBvhNode};

use crate::math::Vec4;

#[derive(Clone, Debug)]
pub struct Scene {
    pub faces: Vec<Face>,
    pub gpu_bvh: Vec<GpuBvhNode>,
    pub orbit_camera: OrbitCamera,
    pub light: Vec4,
    pub brdf_material: BrdfMaterial,
}
