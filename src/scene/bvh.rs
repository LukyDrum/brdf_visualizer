use crate::{
    math::{Aabb, Triangle, Vec4},
    scene::Face,
};

pub enum BvhNode {
    Inner {
        left: Box<Self>,
        right: Box<Self>,
        bounds: Aabb,
    },
    Leaf {
        faces: Vec<Face>,
        bounds: Aabb,
    },
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 3],
    pub left_first: u32, // Inner: right child index | Leaf: start index in faces buffer
    pub aabb_max: [f32; 3],
    pub tri_count: u32, // Inner: 0 | Leaf: number of faces
}

impl BvhNode {
    const MAX_FACES_PER_LEAF: usize = 4;
    const BINS_COUNT: usize = 32;

    pub fn bounds(&self) -> Aabb {
        match self {
            Self::Inner { bounds, .. } | Self::Leaf { bounds, .. } => *bounds,
        }
    }

    pub fn build(faces: Vec<Face>) -> Self {
        let bounds = faces.iter().fold(Aabb::default(), |bounds, face| {
            bounds.merge(Triangle(face.triangle).aabb())
        });

        // Small enough to be a leaf
        if faces.len() <= Self::MAX_FACES_PER_LEAF {
            return Self::Leaf { faces, bounds };
        }

        // Bound over the centers of the triangles
        let centers_bounds = faces.iter().fold(Aabb::default(), |bounds, face| {
            bounds.merge(Aabb::from_point(Triangle(face.triangle).center()))
        });

        // Find in which axis is the box the biggest
        let range = centers_bounds.max - centers_bounds.min;

        let mut axis = Axis::X;
        if range.y > range.x {
            axis = Axis::Y;
        }
        if range.z > range.x && range.z > range.y {
            axis = Axis::Z;
        }

        let (axis_len, axis_start) = match axis {
            Axis::X => (range.x, centers_bounds.min.x),
            Axis::Y => (range.y, centers_bounds.min.y),
            Axis::Z => (range.z, centers_bounds.min.z),
        };

        // Collapse very small range into a leaf node
        if range.length() < 0.01 {
            return Self::Leaf { faces, bounds };
        }

        // Init bins and create a helper closure
        let mut bins = [Bin::default(); Self::BINS_COUNT];
        let bin_width = axis_len / Self::BINS_COUNT as f32;
        let center_to_bin_index = |center: Vec4| {
            let axis_position = match axis {
                Axis::X => center.x,
                Axis::Y => center.y,
                Axis::Z => center.z,
            };
            let in_range_position = axis_position - axis_start;

            // A little clamp just to handle some small overflows of the range
            ((in_range_position / bin_width).floor() as usize).clamp(0, Self::BINS_COUNT - 1)
        };

        for face in faces.iter() {
            let triangle = Triangle(face.triangle);
            let center = triangle.center();
            bins[center_to_bin_index(center)].add(triangle);
        }

        // Pairs of (count, surface area)
        let mut left = [(0, 0.0); Self::BINS_COUNT - 1];
        let mut right = [(0, 0.0); Self::BINS_COUNT - 1];

        // Compute values from left
        let mut current_left_bound = Aabb::default();
        let mut current_left_count = 0;

        for i in 0..Self::BINS_COUNT - 1 {
            current_left_count += bins[i].count;

            if bins[i].count > 0 {
                current_left_bound = current_left_bound.merge(bins[i].aabb);
            }

            left[i] = (
                current_left_count,
                if current_left_count > 0 {
                    current_left_bound.surface_area()
                } else {
                    0.0
                },
            );
        }

        // Compute values from right
        let mut current_right_bound = Aabb::default();
        let mut current_right_count = 0;

        for i in (0..Self::BINS_COUNT - 1).rev() {
            current_right_count += bins[i + 1].count;

            if bins[i + 1].count > 0 {
                current_right_bound = current_right_bound.merge(bins[i + 1].aabb);
            }

            right[i] = (
                current_right_count,
                if current_right_count > 0 {
                    current_right_bound.surface_area()
                } else {
                    0.0
                },
            );
        }

        // Find the best split index
        let (mut min_cost, mut best_index) = (f32::MAX, 0);
        for i in 0..Self::BINS_COUNT - 1 {
            let cost = left[i].0 as f32 * left[i].1 + right[i].0 as f32 * right[i].1;

            if cost < min_cost {
                min_cost = cost;
                best_index = i;
            }
        }

        let (left_faces, right_faces): (Vec<_>, Vec<_>) = faces
            .into_iter()
            .partition(|face| center_to_bin_index(Triangle(face.triangle).center()) <= best_index);

        if left_faces.is_empty() || right_faces.is_empty() {
            let mut all_faces = left_faces;
            all_faces.extend(right_faces);
            return BvhNode::Leaf {
                faces: all_faces,
                bounds,
            };
        }

        let left = Box::new(BvhNode::build(left_faces));
        let right = Box::new(BvhNode::build(right_faces));

        BvhNode::Inner {
            left,
            right,
            bounds,
        }
    }

    /// Flattens the pointer-based BVH into GPU-ready arrays
    pub fn flatten(self) -> (Vec<GpuBvhNode>, Vec<Face>) {
        let mut nodes = Vec::new();
        let mut ordered_faces = Vec::new();
        self.flatten_recursive(&mut nodes, &mut ordered_faces);

        (nodes, ordered_faces)
    }

    fn flatten_recursive(self, nodes: &mut Vec<GpuBvhNode>, ordered_faces: &mut Vec<Face>) {
        match self {
            BvhNode::Leaf { faces, bounds } => {
                let first_face = ordered_faces.len() as u32;
                let tri_count = faces.len() as u32;

                ordered_faces.extend_from_slice(&faces);

                nodes.push(GpuBvhNode {
                    aabb_min: [bounds.min.x, bounds.min.y, bounds.min.z],
                    left_first: first_face,
                    aabb_max: [bounds.max.x, bounds.max.y, bounds.max.z],
                    tri_count,
                });
            }
            BvhNode::Inner {
                left,
                right,
                bounds,
            } => {
                let node_idx = nodes.len();
                // Placeholder; `left_first` will be patched once the right child index is known
                nodes.push(GpuBvhNode {
                    aabb_min: [bounds.min.x, bounds.min.y, bounds.min.z],
                    left_first: 0,
                    aabb_max: [bounds.max.x, bounds.max.y, bounds.max.z],
                    tri_count: 0,
                });

                // Left child is placed immediately at `node_idx + 1`
                left.flatten_recursive(nodes, ordered_faces);

                // Right child starts right after the entire left subtree
                let right_idx = nodes.len() as u32;
                nodes[node_idx].left_first = right_idx;

                right.flatten_recursive(nodes, ordered_faces);
            }
        }
    }
}

enum Axis {
    X,
    Y,
    Z,
}

#[derive(Copy, Clone, Debug, Default)]
struct Bin {
    aabb: Aabb,
    count: usize,
}

impl Bin {
    pub fn add(&mut self, triangle: Triangle) {
        self.aabb = self.aabb.merge(triangle.aabb());
        self.count += 1;
    }
}
