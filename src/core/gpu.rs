use std::{sync::Arc, time::Instant};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    core::{CameraUniform, load_wgsl_shader},
    scene::Scene,
};

const COMPUTE_PIPELINE_OVERRIDE_CONSTANTS: [(&str, f64); 2] = [("wg_x", 8.0), ("wg_y", 8.0)];

pub struct RendererState {
    triangles_buffer: wgpu::Buffer,
    camera: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RtDataUniform {
    resolution: [f32; 2],
    time: f32,
    frame: u32,
}

pub struct GpuState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    start_time: Instant,
    frame: u32,

    renderer: RendererState,
}

impl GpuState {
    pub async fn new(window: Arc<Window>, scene: &Scene) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create GPU device");

        let surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Surface unsupported by adapter");
        surface.configure(&device, &surface_config);

        // Create uniform buffers
        let rt_data_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Data Uniform Buffer"),
            contents: bytemuck::bytes_of(&RtDataUniform {
                resolution: [surface_config.width as f32, surface_config.height as f32],
                time: 0.0,
                frame: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::from_camera(
                scene.camera,
                (surface_config.width, surface_config.height),
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Faces buffer
        let faces_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Faces Buffer"),
            contents: bytemuck::cast_slice(&scene.faces),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let (compute_bind_group, blit_bind_group) = Self::create_target_resources(
            &device,
            &compute_bgl,
            &blit_bgl,
            &rt_data_uniform_buffer,
            &camera_uniform_buffer,
            &faces_buffer,
            surface_config.width,
            surface_config.height,
        );

        let rt_shader = load_wgsl_shader("raytracer.wgsl").expect("Failed to load shader.");
        let rt_shader = device.create_shader_module(rt_shader);
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Layout"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raytracer Pipeline"),
            layout: Some(&compute_layout),
            module: &rt_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &COMPUTE_PIPELINE_OVERRIDE_CONSTANTS,
                ..Default::default()
            },
            cache: None,
        });

        let blit_shader = load_wgsl_shader("blit.wgsl").expect("Failed to load shader.");
        let blit_shader = device.create_shader_module(blit_shader);
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            window,
            device,
            queue,
            surface,
            surface_config,
            compute_pipeline,
            compute_bgl,
            compute_bind_group,
            blit_pipeline,
            blit_bgl,
            blit_bind_group,
            uniform_buffer: rt_data_uniform_buffer,
            start_time: Instant::now(),
            frame: 0,

            renderer: RendererState {
                triangles_buffer: faces_buffer,
                camera: camera_uniform_buffer,
            },
        }
    }

    pub fn create_target_resources(
        device: &wgpu::Device,
        compute_bgl: &wgpu::BindGroupLayout,
        blit_bgl: &wgpu::BindGroupLayout,
        rt_data_uniform_buffer: &wgpu::Buffer,
        camera_uniform_buffer: &wgpu::Buffer,
        triangles_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raytrace Output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());

        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG"),
            layout: compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rt_data_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triangles_buffer.as_entire_binding(),
                },
            ],
        });

        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BG"),
            layout: blit_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        (compute_bg, blit_bg)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);

        let (compute_bg, blit_bg) = Self::create_target_resources(
            &self.device,
            &self.compute_bgl,
            &self.blit_bgl,
            &self.uniform_buffer,
            &self.renderer.camera,
            &self.renderer.triangles_buffer,
            new_size.width,
            new_size.height,
        );
        self.compute_bind_group = compute_bg;
        self.blit_bind_group = blit_bg;
        self.frame = 0;
    }

    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let uniforms = RtDataUniform {
            resolution: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            time: self.start_time.elapsed().as_secs_f32(),
            frame: self.frame,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let surface_view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Raytrace Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(
                self.surface_config.width.div_ceil(8),
                self.surface_config.height.div_ceil(8),
                1,
            );
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        self.frame += 1;
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
