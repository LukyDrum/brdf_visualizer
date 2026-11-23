use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{core::GpuState, scene::Scene};

const ROTATION_DELTA_RAD: f32 = 45f32.to_radians();

pub struct App {
    state: Option<GpuState>,
    scene: Scene,
    last_frame: Instant,
}

impl App {
    pub fn new(scene: Scene) -> Self {
        Self {
            state: None,
            scene,
            last_frame: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Raytracer")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.state = Some(pollster::block_on(GpuState::new(window, &self.scene)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                self.last_frame = Instant::now();
                state.render();
                state.window().request_redraw();
            }
            // Handle keyboard inputs
            WindowEvent::KeyboardInput { event, .. } => {
                let mut camera_yaw = 0.0;
                let mut camera_pitch = 0.0;

                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => camera_pitch = 1.0,
                    PhysicalKey::Code(KeyCode::KeyS) => camera_pitch = -1.0,
                    PhysicalKey::Code(KeyCode::KeyA) => camera_yaw = -1.0,
                    PhysicalKey::Code(KeyCode::KeyD) => camera_yaw = 1.0,
                    _ => return,
                }

                let delta_time = self.last_frame.elapsed().as_secs_f32();

                self.scene.orbit_camera.rotate(
                    camera_yaw * ROTATION_DELTA_RAD * delta_time,
                    camera_pitch * ROTATION_DELTA_RAD * delta_time,
                );

                state.update_from_scene(&self.scene);
            }
            _ => {}
        }
    }
}
