use std::fmt::Display;

pub type IoError = std::io::Error;

#[derive(Debug)]
pub enum LoaderError {
    FileOperation(IoError),
    EmptyInput,
    ParseError(String),
    NonExistantMaterial,
}

impl From<IoError> for LoaderError {
    fn from(err: IoError) -> Self {
        Self::FileOperation(err)
    }
}

impl Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileOperation(error) => {
                write!(f, "An error occurred when working with file: {error}")
            }
            Self::EmptyInput => write!(f, "The input for parsing was empty."),
            Self::ParseError(error) => write!(f, "An error occurred during parsing: {error}"),
            Self::NonExistantMaterial => {
                write!(f, "The .obj file referenced a non-existing material.")
            }
        }
    }
}
