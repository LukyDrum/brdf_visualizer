//! Implementation of loaders for various file formats.
//! Supported file formats have `*Loader` struct for loading from files/lines/etc.

mod error;
mod obj_loader;

use std::num::ParseFloatError;

pub use error::LoaderError;
pub use obj_loader::ObjLoader;

/// Used for parsing of text-based file formats such as OBJ or MTL.
/// It expects that each line (meant for parsing) contains a `key` and some number of `args`.
/// Note that this definition allows the number of args to be even 0.
struct KeyArgs<'a> {
    pub key: &'a str,
    pub args: Vec<&'a str>,
}

fn map_float_parse_err(err: ParseFloatError) -> LoaderError {
    LoaderError::ParseError(format!("{err}"))
}
