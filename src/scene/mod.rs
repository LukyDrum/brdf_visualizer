mod camera;
mod face;
mod material;

pub use camera::Camera;
pub use face::Face;
pub use material::*;

#[derive(Clone, Debug)]
pub struct Scene {
    pub faces: Vec<Face>,
    pub camera: Camera,
}
