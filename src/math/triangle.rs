use std::ops::Index;

use crate::math::{Aabb, Vec4, max, min};

/// A triangle described by 3 points in 3D space.
///
/// To access its vertices, use the indexing operator on the triangle itself.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Triangle(pub [Vec4; 3]);

impl Index<usize> for Triangle {
    type Output = Vec4;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Triangle {
    pub fn normal(&self) -> Vec4 {
        let [x, y, z] = self.0;
        let e1 = y - x;
        let e2 = z - x;

        e1.cross(e2).normalized()
    }

    pub fn area(&self) -> f32 {
        let [a, b, c] = self.0;
        let ab = (a - b).length();
        let bc = (b - c).length();
        let ca = (c - a).length();

        let s = (ab + bc + ca) * 0.5;

        (s * (s - ab) * (s - bc) * (s - ca)).sqrt()
    }

    pub fn aabb(&self) -> Aabb {
        let [a, b, c] = self.0;
        let min_x = min!(a.x, b.x, c.x);
        let min_y = min!(a.y, b.y, c.y);
        let min_z = min!(a.z, b.z, c.z);
        let max_x = max!(a.x, b.x, c.x);
        let max_y = max!(a.y, b.y, c.y);
        let max_z = max!(a.z, b.z, c.z);

        Aabb {
            min: Vec4::vec3(min_x, min_y, min_z),
            max: Vec4::vec3(max_x, max_y, max_z),
        }
    }

    pub fn center(&self) -> Vec4 {
        (self[0] + self[1] + self[2]) / 3.0
    }
}
