use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use crate::scene::BrdfMaterial;

pub const UTIA_BRDF_LEN: usize = 3 * 6 * 48 * 6 * 48; // 248,832 floats (~995 KB)

pub struct BrdfLoader;

impl BrdfLoader {
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<BrdfMaterial> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::with_capacity(UTIA_BRDF_LEN);
        let mut buf = [0u8; 8];

        for _ in 0..UTIA_BRDF_LEN {
            reader.read_exact(&mut buf)?;
            data.push(f64::from_le_bytes(buf) as f32);
        }

        Ok(BrdfMaterial::new(data))
    }
}
