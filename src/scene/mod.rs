mod camera;
mod face;
mod material;
mod orbiter;

pub use camera::{Camera, OrbitCamera};
pub use face::Face;
pub use material::{BrdfMaterial, Material, UNNAMED_MATERIAL};
pub use orbiter::Orbiter;

use crate::math::Vec4;

#[derive(Clone, Debug)]
pub struct Scene {
    pub faces: Vec<Face>,
    pub orbit_camera: OrbitCamera,
    pub light: Vec4,
    pub brdf_material: BrdfMaterial,
}
