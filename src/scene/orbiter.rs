use std::f32::consts::FRAC_PI_2;

use crate::math::Vec4;

#[derive(Clone, Copy, Debug)]
pub struct Orbiter {
    pub center: Vec4,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Orbiter {
    pub fn new(center: Vec4, radius: f32) -> Self {
        Self {
            center,
            radius: radius.max(0.05),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    pub fn position(&self) -> Vec4 {
        let cos_pitch = self.pitch.cos();
        let offset = Vec4::vec3(
            self.radius * cos_pitch * self.yaw.sin(),
            self.radius * self.pitch.sin(),
            self.radius * cos_pitch * self.yaw.cos(),
        );

        self.center + offset
    }

    /// Orbits around the target. Clamps pitch to avoid gimbal lock at the poles.
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-FRAC_PI_2 + 0.001, FRAC_PI_2 - 0.001);
    }

    /// Moves closer to or further from the target exponentially.
    pub fn zoom(&mut self, scroll_delta: f32) {
        self.radius = (self.radius * (1.0 - scroll_delta * 0.1)).max(0.05);
    }
}
