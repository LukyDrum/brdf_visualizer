mod camera;
mod face;

pub use camera::Camera;
pub use face::Face;

#[derive(Clone, Debug)]
pub struct Scene {
    pub faces: Vec<Face>,
    pub camera: Camera,
}
