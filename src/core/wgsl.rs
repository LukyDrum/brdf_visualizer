use std::{
    borrow::Cow,
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use wgpu::{ShaderModuleDescriptor, ShaderSource};

const SHADERS_ROOT: &str = "shaders";

pub fn load_wgsl_shader(
    path: impl AsRef<Path>,
) -> Result<ShaderModuleDescriptor<'static>, std::io::Error> {
    let mut visited = HashSet::new();
    let mut output = String::new();

    let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(SHADERS_ROOT)
        .join(path);
    resolve(&full_path, &mut visited, &mut output)?;

    Ok(ShaderModuleDescriptor {
        label: Some("my_shader"),
        source: ShaderSource::Wgsl(Cow::Owned(output)),
    })
}

fn resolve(
    path: &Path,
    visited: &mut HashSet<PathBuf>,
    out: &mut String,
) -> Result<(), std::io::Error> {
    if !visited.insert(path.to_owned()) {
        return Ok(());
    }

    let source = fs::read_to_string(&path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("// #include \"")
            && let Some(rel_path) = rest.strip_suffix('"')
        {
            resolve(&parent.join(rel_path), visited, out)?;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    Ok(())
}
