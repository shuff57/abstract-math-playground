use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{EventLoop, EventLoopWindowTarget},
    keyboard,
    window::{Window, WindowBuilder},
};

use nalgebra as na;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;


// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
}

impl TimeUniform {
    fn new() -> Self {
        Self { time: 0.0 }
    }

    fn update(&mut self, new_time: f32) {
        self.time = new_time;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    velocity: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[rustfmt::skip]
// This is a cube with the front face removed
const BOX_VERTICES: &[Vertex] = &[
    // Red (back)
    Vertex { position: [-1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0] },

    Vertex { position: [-1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.0, 0.0] },

    // Green (bottom)
    Vertex { position: [1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [-1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [-1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 0.0] },

    Vertex { position: [1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.0] },
    Vertex { position: [-1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.0] },

    // Blue (left)
    Vertex { position: [-1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-1.0, 1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 1.0] },

    Vertex { position: [-1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 0.8] },
    Vertex { position: [-1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 0.8] },
    Vertex { position: [-1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.0, 0.8] },

    // Yellow (right)
    Vertex { position: [1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 1.0, 0.0] },
    Vertex { position: [1.0, 1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 1.0, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [1.0, 1.0, 0.0] },

    Vertex { position: [1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.8, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.8, 0.0] },
    Vertex { position: [1.0, -1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.8, 0.8, 0.0] },

    // Cyan (top)
    Vertex { position: [-1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
    Vertex { position: [1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
    Vertex { position: [1.0, 1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },

    Vertex { position: [-1.0, 1.0, -1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.8] },
    Vertex { position: [1.0, 1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.8] },
    Vertex { position: [-1.0, 1.0, 1.0], velocity: [0.0, 0.0, 0.0], color: [0.0, 0.8, 0.8] },
];

struct Camera {
    eye: na::Point3<f32>,
    target: na::Point3<f32>,
    up: na::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> na::Matrix4<f32> {
        let view = na::Matrix4::look_at_rh(&self.eye, &self.target, &self.up);
        let proj = na::Perspective3::new(self.aspect, self.fovy, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj.as_matrix() * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    // TODO: Does the above also apply when using nalgebra???
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: na::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

struct Application<'a> {
    render_window: RenderWindow<'a>,

    time_uniform: TimeUniform,
    time_buffer: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,
    start_time: instant::Instant,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<Vertex>,
    #[allow(dead_code)] // kept with the buffers it describes
    num_vertices: u32,
    #[allow(dead_code)]
    indices: Vec<u16>,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    vertex_buffer_2: wgpu::Buffer,
    num_vertices_2: u32,
}

impl<'a> Application<'a> {
    async fn new(render_window: RenderWindow<'a>) -> Self {
        #[rustfmt::skip]
        let vertices: Vec<Vertex> = vec![
            Vertex { position: [-0.0868241, 0.49240386, 0.0], velocity: [1.0, 0.5, 0.0], color: [1.0, 0.0, 0.5] },
            Vertex { position: [-0.49513406, 0.06958647, 0.0], velocity: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [-0.21918549, -0.44939706, 0.0], velocity: [0.2, -0.8, 0.0], color: [0.0, 0.0, 1.0] },
            Vertex { position: [0.35966998, -0.3473291, 0.0], velocity: [0.5, 0.7, 0.0], color: [1.0, 1.0, 0.0] },
            Vertex { position: [0.44147372, 0.2347359, 0.0], velocity: [-1.0, -0.3, 0.0], color: [0.0, 1.0, 1.0] },
        ];

        #[rustfmt::skip]
        let indices: Vec<u16> = vec![
            0, 1, 4,
            1, 2, 4,
            2, 3, 4,
        ];

        let vertex_buffer =
            render_window
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let index_buffer =
            render_window
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

        let num_vertices = vertices.len() as u32;
        let num_indices = indices.len() as u32;

        let vertex_buffer_2 =
            render_window
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer 2"),
                    contents: bytemuck::cast_slice(BOX_VERTICES),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let num_vertices_2 = BOX_VERTICES.len() as u32;

        let mut time_uniform = TimeUniform::new();
        time_uniform.update(0.0);

        let time_buffer =
            render_window
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Time Buffer"),
                    contents: bytemuck::cast_slice(&[time_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let time_bind_group_layout =
            render_window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("time_bind_group_layout"),
                });

        let time_bind_group = render_window
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &time_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: time_buffer.as_entire_binding(),
                }],
                label: Some("time_bind_group"),
            });

        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: na::Point3::new(0.0, 2.0, 3.0),
            // have it look at the origin
            target: na::Point3::new(0.0, 0.0, 0.0),
            // which way is "up"
            up: na::Vector3::y(),
            aspect: render_window.config.width as f32 / render_window.config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer =
            render_window
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let camera_bind_group_layout =
            render_window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("camera_bind_group_layout"),
                });

        let camera_bind_group =
            render_window
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                    label: Some("camera_bind_group"),
                });

        let shader_source = include_str!("shader.wgsl");
        let shader_source = shader_source.replace("NUMBER", "3.0"); // Experimenting with how I can generate custom .wgsl shaders on the fly – there might be a better way than generating a string

        let shader = render_window
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                // source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            render_window
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&time_bind_group_layout, &camera_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline =
            render_window
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: render_window.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None, // Some(wgpu::Face::Back),
                        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                        polygon_mode: wgpu::PolygonMode::Fill,
                        // Requires Features::DEPTH_CLIP_CONTROL
                        unclipped_depth: false,
                        // Requires Features::CONSERVATIVE_RASTERIZATION
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                });

        Self {
            render_window,

            time_uniform,
            time_buffer,
            time_bind_group,
            start_time: instant::Instant::now(),

            camera_uniform,
            camera_buffer,
            camera_bind_group,

            render_pipeline,
            camera,
            vertices,
            num_vertices,
            indices,
            index_buffer,
            num_indices,

            vertex_buffer,
            vertex_buffer_2,
            num_vertices_2,
        }
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        let time = self.start_time.elapsed().as_secs_f32();

        // Update time uniform
        self.time_uniform.update(time);
        self.render_window.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );

        // Update camera uniform
        self.camera.eye = na::Point3::new(1.5 * time.cos(), 1.5 * time.sin(), 3.0);
        self.camera_uniform.update_view_proj(&self.camera);
        self.render_window.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update vertices
        let vertices = self
            .vertices
            .iter()
            .map(|v| {
                let mut position = [
                    v.position[0] + v.velocity[0] * 0.01,
                    v.position[1] + v.velocity[1] * 0.01,
                    v.position[2] + v.velocity[2] * 0.01,
                ];

                let mut velocity = v.velocity;

                // Bounce off the walls
                if position[0] < -1.0 || position[0] > 1.0 {
                    position[0] = position[0].clamp(-1.0, 1.0);
                    // Reverse the velocity
                    velocity[0] = -velocity[0];
                }
                if position[1] < -1.0 || position[1] > 1.0 {
                    position[1] = position[1].clamp(-1.0, 1.0);
                    // Reverse the velocity
                    velocity[1] = -velocity[1];
                }

                Vertex {
                    position,
                    velocity,
                    color: v.color,
                }
            })
            .collect::<Vec<Vertex>>();

        self.render_window.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );

        self.vertices = vertices;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.render_window.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.render_window
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // The {} cause the render pass to be dropped before we move on,
        // which is essential because the render pass borrows the encoder
        // mutably and it needs to be dropped before we can call encoder.finish()
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.time_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.vertex_buffer_2.slice(..));
            render_pass.draw(0..self.num_vertices_2, 0..1);

            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.render_window
            .queue
            .submit(std::iter::once(encoder.finish()));
        output.present(); // TODO: This is where the "removing swapchain" happens

        Ok(())
    }

    fn run(&mut self, window: &Window, event_loop: EventLoop<()>) {
        event_loop
            .run(|event, control_flow| {
                match event {
                    Event::WindowEvent {
                        ref event,
                        window_id,
                    } if window_id == window.id() => {
                        if !self.input(event) {
                            match event {
                                WindowEvent::CloseRequested
                                | WindowEvent::KeyboardInput {
                                    event:
                                        KeyEvent {
                                            state: ElementState::Pressed,
                                            logical_key:
                                                keyboard::Key::Named(keyboard::NamedKey::Escape),
                                            ..
                                        },
                                    ..
                                } => EventLoopWindowTarget::exit(control_flow),
                                WindowEvent::Resized(physical_size) => {
                                    log::info!("Resized: {:?}", physical_size);
                                    self.render_window.resize(Some(*physical_size));
                                }
                                // WindowEvent::ScaleFactorChanged {
                                //     scale_factor,
                                //     inner_size_writer,
                                //     ..
                                // } => {
                                //     // new_inner_size is &&mut so we have to dereference it twice
                                //     // state.resize(**new_inner_size);
                                // }
                                WindowEvent::RedrawRequested if window_id == window.id() => {
                                    self.update();

                                    match self.render() {
                                        Ok(_) => {}
                                        // Reconfigure the surface if lost
                                        Err(wgpu::SurfaceError::Lost) => {
                                            log::warn!("Lost surface, resizing");
                                            self.render_window.resize(None);
                                        }
                                        // The system is out of memory, we should probably quit
                                        Err(wgpu::SurfaceError::OutOfMemory) => {
                                            log::error!("Out of memory, exiting");
                                            EventLoopWindowTarget::exit(control_flow)
                                        }
                                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                                        Err(e) => eprintln!("{:?}", e),
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Event::AboutToWait => {
                        // RedrawRequested will only trigger once unless we manually request it
                        window.request_redraw();
                    }
                    _ => {}
                }
            })
            .unwrap();
    }
}

struct RenderWindow<'a> {
    surface: wgpu::Surface<'a>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

impl<'a> RenderWindow<'a> {
    pub async fn new(window: &'a Window, initial_size: &winit::dpi::LogicalSize<i32>) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        // let instance = wgpu::Instance::default();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // backends: wgpu::Backends::all(),
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        // # Safety
        // The surface needs to live as long as the window that created it.
        // State owns the window, so this should be safe.
        let surface: wgpu::Surface<'a> = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        log::info!("Surface capabilities: {:?}", surface_caps);

        // Choose a format that will be consistent across platforms
        // TODO: I still don't actually understand what I should
        // be doing here. This just works for now.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb()) // Prefer linear over srgb
            .unwrap_or(surface_caps.formats[0]);

        // let surface_format = surface_caps.formats[0];

        log::info!("Available surface formats: {:?}", surface_caps.formats);
        log::info!("Surface format: {:?}", surface_format);

        let size = initial_size.to_physical(window.scale_factor());
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            // present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        RenderWindow {
            window,
            surface,
            config,
            device,
            queue,
            size,
        }
    }

    pub fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        let new_size = new_size.unwrap_or(self.window.inner_size());
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub async fn run() {
    // Set up logging (console.log for wasm32, env_logger for native)
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        } else {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
        }
    }

    // First, create window and event loop
    let initial_size = winit::dpi::LogicalSize::new(640, 480);
    let (window, event_loop) = create_window(initial_size);

    let render_window = RenderWindow::new(&window, &initial_size).await;
    let mut app = Application::new(render_window).await;
    app.run(&window, event_loop);
}

fn create_window(initial_size: winit::dpi::LogicalSize<i32>) -> (Window, EventLoop<()>) {
    let event_loop = EventLoop::new().unwrap();

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::WindowBuilderExtWebSys;

            let canvas = web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| {
                    let canvas = doc.get_element_by_id("canvas")?;
                    let canvas: web_sys::HtmlCanvasElement = canvas
                        .dyn_into::<web_sys::HtmlCanvasElement>()
                        .map_err(|_| ())
                        .unwrap();
                    Some(canvas)
                })
                .expect("Couldn't find canvas element with id 'canvas'");

            let window_builder = WindowBuilder::new()
                .with_inner_size(initial_size)
                // .with_inner_size(winit::dpi::PhysicalSize::new(canvas.width(), canvas.height()))
                .with_canvas(Some(canvas));

            log::info!("WindowBuilder: {:?}", window_builder);

            let window = window_builder
                .build(&event_loop)
                .unwrap();

            log::info!("Window: {:?}", window);
            log::info!("Window size: {:?}", window.inner_size());
        } else {
            let window = WindowBuilder::new()
                .with_inner_size(initial_size)
                .with_title("Math Playground")
                .build(&event_loop)
                .unwrap();
        }
    }

    (window, event_loop)
}
