use winit::event_loop::{ControlFlow, EventLoop};

use crate::{
    app::App,
    math::Vec4,
    scene::{Camera, Face, Scene},
};

mod app;
mod core;
mod math;
mod scene;

fn main() {
    let scene = Scene {
        faces: vec![Face {
            triangle: [
                Vec4::vec3(2.0, -2.0, 10.0),
                Vec4::vec3(-2.0, -2.0, 10.0),
                Vec4::vec3(0.0, 2.0, 10.0),
            ],
            normals: [
                Vec4::vec3(0.0, 0.0, -1.0),
                Vec4::vec3(0.0, 0.0, -1.0),
                Vec4::vec3(0.0, 0.0, -1.0),
            ],
        }],
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
