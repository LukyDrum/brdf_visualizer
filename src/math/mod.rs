mod aabb;
mod vec4;
mod triangle;

pub use vec4::Vec4;
pub use aabb::Aabb;
pub use triangle::Triangle;

macro_rules! min {
    ($a:expr, $($b:expr),+) => {
        $a$(.min($b))+
    };
}

macro_rules! max {
    ($a:expr, $($b:expr),+) => {
        $a$(.max($b))+
    };
}

pub(crate) use {max, min};
