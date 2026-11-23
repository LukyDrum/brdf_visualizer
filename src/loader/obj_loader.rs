use std::{collections::HashMap, path::PathBuf};

use crate::{
    either::Either,
    loader::{KeyArgs, LoaderError, map_float_parse_err},
    math::Vec4,
    scene::{Face, Material, UNNAMED_MATERIAL},
};

const COMMENT_SIGN: char = '#';
const MATERIAL_LIB_KEY: &str = "mtllib";
const USE_MATERIAL_KEY: &str = "usemtl";
const GROUP_KEY: &str = "g";
const VERTEX_KEY: &str = "v";
const VERTEX_NORMAL_KEY: &str = "vn";
const VERTEX_TEXTURE_KEY: &str = "vt";
const FACE_KEY: &str = "f";
const OBJECT_KEY: &str = "o";
const SMOOTH_SHADING_KEY: &str = "s";

/// A loader for the `.obj` files.
///
/// To use it call one of its load methods.
/// It takes care of loading the materials as well.
pub struct ObjLoader;

impl ObjLoader {
    pub fn load_from_file(path: impl Into<PathBuf>) -> Result<Vec<Face>, LoaderError> {
        let path = path.into();
        let base_path = {
            let mut base = path.clone();
            base.pop();
            base
        };

        let content = std::fs::read_to_string(path)?;

        Self::load_from_lines(content.lines(), base_path)
    }

    pub fn load_from_lines<'a>(
        lines: impl Iterator<Item = &'a str>,
        base_path: impl Into<PathBuf>,
    ) -> Result<Vec<Face>, LoaderError> {
        let all_key_args = lines
            .map(|line| line.trim())
            .filter(|line| !line.starts_with(COMMENT_SIGN) && !line.is_empty())
            .map(|line| {
                let mut split = line.split_ascii_whitespace();
                let key = split.next().expect("Failed to get a key for this line!");

                let args = split
                    .take_while(|arg| !arg.starts_with(COMMENT_SIGN))
                    .collect::<Vec<_>>();

                KeyArgs { key, args }
            });

        let base_path = base_path.into();

        let mut materials = HashMap::new();
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut textures: Vec<(f32, f32)> = Vec::new();
        let mut faces = Vec::new();

        let mut used_material = String::new();
        let mut current_smoothing_group = 0;

        for KeyArgs { key, args } in all_key_args {
            match key {
                MATERIAL_LIB_KEY => {
                    let mtl_file = args[0];
                    let mtl_file = base_path.join(mtl_file);
                    materials = HashMap::from([(UNNAMED_MATERIAL.to_owned(), Material::default())]);
                }
                USE_MATERIAL_KEY => {
                    used_material = args[0].to_owned();
                }
                VERTEX_KEY => {
                    let x = parse_f32(args[0])?;
                    let y = parse_f32(args[1])?;
                    let z = parse_f32(args[2])?;

                    let vector = Vec4::vec3(x, y, z);
                    vertices.push(vector);
                }
                VERTEX_NORMAL_KEY => {
                    let x = parse_f32(args[0])?;
                    let y = parse_f32(args[1])?;
                    let z = parse_f32(args[2])?;

                    let vector = Vec4::vec3(x, y, z);
                    normals.push(vector);
                }
                VERTEX_TEXTURE_KEY => {
                    let u = parse_f32(args[0])?;
                    let v = if args.len() > 1 {
                        parse_f32(args[1])?
                    } else {
                        0.0
                    };

                    textures.push((u, v));
                }
                GROUP_KEY | OBJECT_KEY => {
                    // For now, do nothing
                }
                SMOOTH_SHADING_KEY => {
                    if let Some(&arg) = args.first() {
                        current_smoothing_group = if arg == "off" || arg == "0" {
                            0
                        } else {
                            arg.parse::<i32>().unwrap_or(1)
                        };
                    }
                }
                FACE_KEY => {
                    // Pass normals to the parser as well
                    let triangle_or_two = Self::parse_face(&args, &vertices, &textures, &normals)?;
                    match triangle_or_two {
                        Either::Left((triangle, uvs, vertex_normals)) => {
                            // If normal was provided by `vn`, use it. Otherwise calculate it.
                            let normal = triangle_normal(triangle);
                            let material = materials
                                .get(&used_material)
                                .ok_or(LoaderError::NonExistantMaterial)?
                                .clone();

                            let tangent = uvs
                                .as_ref()
                                .map(|uvs| Self::calculate_tangent(triangle, uvs));

                            let face = Face {
                                triangle,
                                normals: vertex_normals.unwrap_or([normal, normal, normal]),
                                // uvs,
                                // smooth_shading: current_smoothing_group > 0,
                                // material,
                                // tangent,
                            };

                            faces.push(face);
                        }
                        Either::Right((
                            (first_tri, uvs_1, vertex_normals_1),
                            (second_tri, uvs_2, vertex_normals_2),
                        )) => {
                            let normal_1 = triangle_normal(first_tri);
                            let normal_2 = triangle_normal(second_tri);

                            let material = materials
                                .get(&used_material)
                                .ok_or(LoaderError::NonExistantMaterial)?
                                .clone();

                            let tangent_1 = uvs_1
                                .as_ref()
                                .map(|uvs| Self::calculate_tangent(first_tri, uvs));
                            let tangent_2 = uvs_2
                                .as_ref()
                                .map(|uvs| Self::calculate_tangent(second_tri, uvs));

                            let first_face = Face {
                                triangle: first_tri,
                                normals: vertex_normals_1.unwrap_or([normal_1, normal_1, normal_1]),
                                // uvs: uvs_1,
                                // smooth_shading: current_smoothing_group > 0,
                                // material: material.clone(),
                                // tangent: tangent_1,
                            };
                            let second_face = Face {
                                triangle: second_tri,
                                normals: vertex_normals_2.unwrap_or([normal_2, normal_2, normal_2]),
                                // uvs: uvs_2,
                                // smooth_shading: current_smoothing_group > 0,
                                // material,
                                // tangent: tangent_2,
                            };

                            faces.push(first_face);
                            faces.push(second_face);
                        }
                    }
                }
                key => {
                    return Err(LoaderError::ParseError(format!("Obj: Unknown key: {key}")));
                }
            }
        }

        Ok(faces)
    }

    fn parse_face(
        args: &[&str],
        vertices: &[Vec4],
        textures: &[(f32, f32)],
        normals: &[Vec4],
    ) -> Result<
        Either<
            ([Vec4; 3], Option<[(f32, f32); 3]>, Option<[Vec4; 3]>),
            (
                ([Vec4; 3], Option<[(f32, f32); 3]>, Option<[Vec4; 3]>),
                ([Vec4; 3], Option<[(f32, f32); 3]>, Option<[Vec4; 3]>),
            ),
        >,
        LoaderError,
    > {
        // Parse the strings into vertices and optional normal vectors
        let (a, uv_a, n_a) = Self::parse_vertex_data(args[0], vertices, textures, normals)?;
        let (b, uv_b, n_b) = Self::parse_vertex_data(args[1], vertices, textures, normals)?;
        let (c, uv_c, n_c) = Self::parse_vertex_data(args[2], vertices, textures, normals)?;

        let first_tri = [a, b, c];

        if args.len() == 3 {
            return Ok(Either::Left((
                first_tri,
                Self::pack_options_3(uv_a, uv_b, uv_c),
                Self::pack_options_3(n_a, n_b, n_c),
            )));
        }
        if args.len() == 4 {
            let (d, uv_d, n_d) = Self::parse_vertex_data(args[3], vertices, textures, normals)?;
            let second_tri = [a, c, d];

            return Ok(Either::Right((
                (
                    first_tri,
                    Self::pack_options_3(uv_a, uv_b, uv_c),
                    Self::pack_options_3(n_a, n_b, n_c),
                ),
                (
                    second_tri,
                    Self::pack_options_3(uv_a, uv_c, uv_d),
                    Self::pack_options_3(n_a, n_c, n_d),
                ),
            )));
        }

        Err(LoaderError::ParseError(
            "Obj: Could not parse face.".to_owned(),
        ))
    }

    /// Helper to parse a `v/vt/vn` string into actual Vectors, while respecting 1-based indexing
    fn parse_vertex_data(
        str: &str,
        vertices: &[Vec4],
        textures: &[(f32, f32)],
        normals: &[Vec4],
    ) -> Result<(Vec4, Option<(f32, f32)>, Option<Vec4>), LoaderError> {
        let mut parts = str.split('/').map(|part| part.trim());

        let v_str = parts.next().unwrap_or("");
        let v_idx = v_str
            .parse::<i32>()
            .map_err(|e| LoaderError::ParseError(format!("{e}")))?;
        let v = vertices[Self::resolve_obj_index(v_idx, vertices.len())];

        let mut uv = None;
        if let Some(vt_str) = parts.next() {
            if !vt_str.is_empty() {
                let vt_idx = vt_str
                    .parse::<i32>()
                    .map_err(|e| LoaderError::ParseError(format!("{e}")))?;
                uv = Some(textures[Self::resolve_obj_index(vt_idx, textures.len())]);
            }
        }

        let mut normal = None;
        if let Some(vn_str) = parts.next() {
            if !vn_str.is_empty() {
                let vn_idx = vn_str
                    .parse::<i32>()
                    .map_err(|e| LoaderError::ParseError(format!("{e}")))?;
                normal = Some(normals[Self::resolve_obj_index(vn_idx, normals.len())]);
            }
        }

        Ok((v, uv, normal))
    }

    /// Converts 1-based and negative Obj indices into 0-based indices
    fn resolve_obj_index(idx: i32, len: usize) -> usize {
        if idx < 0 {
            (len as i32 + idx) as usize
        } else {
            (idx - 1) as usize
        }
    }

    fn pack_options_3<T>(a: Option<T>, b: Option<T>, c: Option<T>) -> Option<[T; 3]> {
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            return None;
        };

        Some([a, b, c])
    }

    fn calculate_tangent(triangle: [Vec4; 3], uvs: &[(f32, f32)]) -> Vec4 {
        let uv0 = uvs[0];
        let uv1 = uvs[1];
        let uv2 = uvs[2];

        let edge1 = triangle[1] - triangle[0];
        let edge2 = triangle[2] - triangle[0];

        let delta_uv1 = (uv1.0 - uv0.0, uv1.1 - uv0.1);
        let delta_uv2 = (uv2.0 - uv0.0, uv2.1 - uv0.1);

        let f = 1.0 / (delta_uv1.0 * delta_uv2.1 - delta_uv2.0 * delta_uv1.1);

        Vec4::vec3(
            f * (delta_uv2.1 * edge1.x - delta_uv1.1 * edge2.x),
            f * (delta_uv2.1 * edge1.y - delta_uv1.1 * edge2.y),
            f * (delta_uv2.1 * edge1.z - delta_uv1.1 * edge2.z),
        )
        .normalized()
    }
}

fn parse_f32(str: &str) -> Result<f32, LoaderError> {
    str.parse::<f32>().map_err(map_float_parse_err)
}

fn triangle_normal(triangle: [Vec4; 3]) -> Vec4 {
    let [x, y, z] = triangle;
    let e1 = y - x;
    let e2 = z - x;

    e1.cross(e2).normalized()
}
