use winit::event_loop::{ControlFlow, EventLoop};

use crate::{
    app::App, loader::{BrdfLoader, ObjLoader}, math::Vec4, scene::{BvhNode, OrbitCamera, Scene},
};

mod app;
mod core;
mod either;
mod loader;
mod math;
mod scene;

fn main() {
    let faces =
        ObjLoader::load_from_file("./scenes/Bunny/Bunny.obj").expect("Failed to load Scene file");
    let bvh = BvhNode::build(faces);
    let (gpu_bvh, faces) = bvh.flatten();

    let scene = Scene {
        faces,
        gpu_bvh,
        orbit_camera: OrbitCamera::new(Vec4::vec3(0.0, 0.5, 0.0), 3.0, 45.0),
        light: Vec4::vec3(2.0, 1.0, 0.5),
        brdf_material: BrdfLoader::load_from_file("../BRDF/UTIA_BRDF_PG14_bin/m072_fabric107.bin")
            .expect("Failed to load BRDF material"),
    };

    let mut app = App::new(scene);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}
