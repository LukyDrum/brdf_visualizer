pub const UNNAMED: &str = "unnamed";

#[derive(Clone, Debug)]
pub struct Material {
    /// Name of the material.
    /// It should not contain whitespace.
    pub name: String,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: UNNAMED.to_owned(),
        }
    }
}
