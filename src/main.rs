use winit::event_loop::{ControlFlow, EventLoop};

use crate::{
    app::App,
    loader::ObjLoader,
    math::Vec4,
    scene::{Camera, Scene},
};

mod app;
mod core;
mod either;
mod loader;
mod math;
mod scene;

fn main() {
    let faces =
        ObjLoader::load_from_file("../Scenes/Bunny/Bunny.obj").expect("Failed to load Scene file");

    let scene = Scene {
        faces,
        camera: Camera {
            position: Vec4::vec3(0.0, 1.0, -4.42),
            forward: Vec4::vec3(0.0, 0.0, 1.0),
            up: Vec4::vec3(0.0, 1.0, 0.0),
            fov: 45.0,
        },
    };

    let mut app = App::new(scene);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}
