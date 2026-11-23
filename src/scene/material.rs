pub const UNNAMED_MATERIAL: &str = "unnamed";

#[derive(Clone, Debug)]
pub struct Material {
    /// Name of the material.
    /// It should not contain whitespace.
    pub name: String,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: UNNAMED_MATERIAL.to_owned(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BrdfMaterial(Vec<f32>);

impl BrdfMaterial {
    pub fn new(data: Vec<f32>) -> Self {
        Self(data)
    }

    pub fn raw_data(&self) -> &[f32] {
        &self.0
    }
}
