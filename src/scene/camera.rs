use std::{
    f32::consts::FRAC_PI_2,
    ops::{Deref, DerefMut},
};

use crate::{math::Vec4, scene::Orbiter};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec4,
    pub forward: Vec4,
    pub up: Vec4,
    pub fov: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    orbiter: Orbiter,
    fov: f32,
}

impl DerefMut for OrbitCamera {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.orbiter
    }
}

impl Deref for OrbitCamera {
    type Target = Orbiter;

    fn deref(&self) -> &Self::Target {
        &self.orbiter
    }
}

impl OrbitCamera {
    pub fn new(center: Vec4, radius: f32, fov: f32) -> Self {
        Self {
            orbiter: Orbiter::new(center, radius),
            fov,
        }
    }

    /// Returns orthonormal (forward, right, up) vectors looking at `target`.
    fn basis(&self) -> (Vec4, Vec4, Vec4) {
        let pos = self.position();
        let forward = (self.center - pos).normalized();
        let right = forward.cross(Vec4::vec3(0.0, 1.0, 0.0)).normalized();
        let up = right.cross(forward);

        (forward, right, up)
    }

    pub fn camera(&self) -> Camera {
        let position = self.position();
        let (forward, _, up) = self.basis();

        Camera {
            position,
            forward,
            up,
            fov: self.fov,
        }
    }
}
