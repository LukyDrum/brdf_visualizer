use crate::math::Vec4;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec4,
    pub max: Vec4,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vec4::vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec4::vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
}

impl Aabb {
    pub fn from_point(point: Vec4) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn merge(&self, other: Aabb) -> Self {
        Aabb {
            min: Vec4::vec3(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec4::vec3(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn surface_area(&self) -> f32 {
        let range = self.max - self.min;
        2.0 * (range.x * range.y + range.x * range.z + range.y * range.z)
    }
}
