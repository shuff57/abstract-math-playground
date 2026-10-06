//! GPU renderer for a [`SceneGeometry`]: thick anti-aliased segments/dots and lit surfaces,
//! driven by the shared [`Rig`] camera. Independent of any window or surface, so it renders
//! into a swapchain frame or an offscreen texture alike.

use crate::geometry::{FieldKind, FieldSpec, MeshVertex, SceneGeometry, SegmentInstance};
use math_core::view::{CameraUniform, Mode, Rig};
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use wgpu::util::DeviceExt;

/// Maximum number of distinct field pipelines kept alive; the oldest is evicted beyond this.
/// (Slider values are uniforms, so a drag reuses one pipeline; edits to the expression itself still make new shaders, so this must be bounded.)
const FIELD_CACHE_CAP: usize = 48;

const PIPE_PENDING: u8 = 0;
const PIPE_READY: u8 = 1;
const PIPE_FAILED: u8 = 2;

/// One compiled field pipeline. `state` is set from the wgpu error scope: a shader or pipeline
/// that fails validation is marked FAILED and simply never drawn (the field stays invisible)
/// instead of poisoning the whole command buffer. On the web the scope resolves asynchronously,
/// so the pipeline stays PENDING (not drawn) for a frame or two.
struct FieldPipe {
    pipeline: wgpu::RenderPipeline,
    state: Arc<AtomicU8>,
}

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One drawable layer with an opacity multiplier. During a mode switch the renderer draws the
/// outgoing mode's geometry fading out and the incoming mode's fading in.
pub struct Layer<'a> {
    pub geometry: &'a SceneGeometry,
    pub fade: f32,
    /// The render origin the geometry was built against (the window centre at build time).
    /// May differ from the rig's current origin while a rebuild is pending.
    pub origin: [f64; 3],
    /// Scale of z about the plane z = 0 (1 = as built). A 3D scene is drawn with the rig's
    /// [`Rig::lift`] during a switch to or from 3D, so it grows out of / flattens into the plane
    /// the 2D scene lies in; see [`layer_lift`].
    pub lift: f32,
}

/// [`Layer::lift`] for a scene built in `built` mode, given the rig's current lift: only 3D
/// scenes are scaled (1D/2D geometry is flat already).
pub fn layer_lift(built: Mode, rig_lift: f64) -> f32 {
    if built == Mode::D3 && rig_lift.is_finite() {
        rig_lift.clamp(0.0, 1.0) as f32
    } else {
        1.0
    }
}

/// Opacity multipliers `(outgoing, incoming)` for the two layers of a mode switch at tween
/// `progress` in [0, 1]. Staggered so the incoming layer is nearly opaque before the outgoing
/// one is gone: surfaces write depth, so a long half-transparent phase would show back faces.
pub fn crossfade(progress: f32) -> (f32, f32) {
    let smooth = |a: f32, b: f32, x: f32| {
        let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    (1.0 - smooth(0.35, 0.95, progress), smooth(0.05, 0.55, progress))
}

/// Opacity multipliers of the two sides of a mode switch. Each side's backdrop (grid, axes) and
/// items (curves, surfaces) fade separately: see [`mode_fades`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeFades {
    pub from_backdrop: f32,
    pub from_items: f32,
    pub to_backdrop: f32,
    pub to_items: f32,
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Lift below which the 3D scene is still invisible (it is a flat sheet in the plane then, and
/// drawn over the 2D grid it would only wash it out).
pub const FADE_3D_START: f32 = 0.2;
/// Lift at which the 3D scene is fully opaque.
pub const FADE_3D_FULL: f32 = 0.6;

/// Opacities for a switch from `from` to `to` at tween `progress` with the rig's `lift`.
///
/// A switch that involves 3D is driven by the LIFT (so an interrupted switch stays consistent):
/// the flat scene's grid and axes go first, the 3D scene (box, surfaces) comes in once it has
/// some height, and the flat scene's curves stay until the 3D form has risen, so a 2D curve
/// visibly turns into its wall / surface instead of fading next to a different object.
/// Switches between the flat modes (1D <-> 2D) keep the plain staggered [`crossfade`].
pub fn mode_fades(from: Mode, to: Mode, progress: f32, lift: f32) -> ModeFades {
    if from != Mode::D3 && to != Mode::D3 {
        let (o, i) = crossfade(progress);
        return ModeFades { from_backdrop: o, from_items: o, to_backdrop: i, to_items: i };
    }
    let s = if lift.is_finite() { lift.clamp(0.0, 1.0) } else { 0.0 };
    let three = smooth(FADE_3D_START, FADE_3D_FULL, s);
    let back = 1.0 - smooth(0.0, 0.4, s);
    let items = 1.0 - smooth(0.5, 0.95, s);
    if from == Mode::D3 {
        ModeFades { from_backdrop: three, from_items: three, to_backdrop: back, to_items: items }
    } else {
        ModeFades { from_backdrop: back, from_items: items, to_backdrop: three, to_items: three }
    }
}

/// A secondary view drawn into a corner rectangle of the target (the slice inset): its own 2D
/// camera and layers, clipped to `rect`.
pub struct Inset<'a> {
    /// `[x, y, w, h]` in target pixels from the top-left corner.
    pub rect: [u32; 4],
    pub rig: &'a Rig,
    pub layers: Vec<Layer<'a>>,
}

struct Prepared {
    bind_group: wgpu::BindGroup,
    geo: Arc<GeoBufs>,
    /// (index into the layer's `fields`, params bind group)
    fields: Vec<(usize, wgpu::BindGroup)>,
}

/// The GPU buffers of one [`SceneGeometry`], uploaded once and reused every frame until the
/// geometry changes (see [`GeoKey`]); only the per-layer uniforms are rebuilt per frame.
struct GeoBufs {
    segments: Option<(wgpu::Buffer, u32)>,
    overlay: Option<(wgpu::Buffer, u32)>,
    mesh: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    /// (shared vertex buffer, flat-triangle indices, count)
    flat: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
}

/// Identity of a geometry's buffers: where its vectors live, how long they are and a sampled
/// content hash, so a rebuilt scene that happens to reuse an allocation is still told apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GeoKey {
    ptrs: [usize; 4],
    lens: [usize; 4],
    sample: u64,
}

/// FNV-1a over up to 192 evenly spaced 8-byte words of `s` (plus its length).
fn sample_hash<T: bytemuck::Pod>(h: u64, s: &[T]) -> u64 {
    let b: &[u8] = bytemuck::cast_slice(s);
    let mut h = h ^ b.len() as u64;
    h = h.wrapping_mul(0x100000001b3);
    let words = b.len() / 8;
    let step = (words / 192).max(1);
    let mut i = 0;
    while i < words {
        let w = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x100000001b3);
        i += step;
    }
    h
}

impl GeoKey {
    pub fn of(g: &SceneGeometry) -> GeoKey {
        let mut h = 0xcbf29ce484222325u64;
        h = sample_hash(h, &g.segments);
        h = sample_hash(h, &g.overlay_segments);
        h = sample_hash(h, &g.vertices);
        h = sample_hash(h, &g.indices);
        h = sample_hash(h, &g.flat_indices);
        GeoKey {
            ptrs: [
                g.segments.as_ptr() as usize,
                g.overlay_segments.as_ptr() as usize,
                g.vertices.as_ptr() as usize,
                g.indices.as_ptr() as usize,
            ],
            lens: [g.segments.len(), g.overlay_segments.len(), g.vertices.len(), g.indices.len() + g.flat_indices.len()],
            sample: h,
        }
    }
}

/// Renders (not frames) an unused cached geometry is kept for before it is dropped.
const GEO_CACHE_KEEP: u64 = 24;

pub struct Renderer {
    seg_pipeline: wgpu::RenderPipeline,
    /// Same shader as `seg_pipeline`, depth-tested but NOT depth-writing: for lines that lie ON
    /// a surface (slice curves), so overlapping segments cannot hide each other.
    overlay_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    /// Same shader as `mesh_pipeline` (zero-normal vertices are unlit) but with depth writes off,
    /// for translucent flat shapes.
    flat_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth: Option<(wgpu::TextureView, (u32, u32))>,
    field_layout: wgpu::BindGroupLayout,
    field_pipeline_layout: wgpu::PipelineLayout,
    field_cache: HashMap<String, FieldPipe>,
    field_order: VecDeque<String>,
    format: wgpu::TextureFormat,
    /// How many field pipelines have been compiled over this renderer's life (test hook: a
    /// slider drag must not increase it).
    field_compiles: usize,
    /// Uploaded geometry by identity, with the render count it was last drawn at.
    geo_cache: HashMap<GeoKey, (Arc<GeoBufs>, u64)>,
    renders: u64,
    /// How many geometries were uploaded over this renderer's life (test hook: a still scene
    /// must not be re-uploaded every frame).
    geo_uploads: usize,
    /// False re-uploads every geometry every frame (the behaviour before the cache; measurement only).
    pub cache_geometry: bool,
}

/// Camera uniform bytes in the exact layout of `Camera` in render.wgsl (112 bytes).
fn camera_bytes(c: &CameraUniform) -> [f32; 28] {
    let mut out = [0f32; 28];
    for (col, chunk) in c.view_proj.iter().enumerate() {
        out[col * 4..col * 4 + 4].copy_from_slice(chunk);
    }
    out[16..20].copy_from_slice(&c.origin);
    out[20..24].copy_from_slice(&c.scale);
    out[24] = f32::from_bits(c.mode);
    out[25] = c.ortho_t;
    out
}

/// `FieldParams` bytes in the layout of field.wgsl (112 bytes: 48 fixed + 4 `vec4` slider
/// params). Kind: 0 hue, 1 fill f > 0, 2 fill f < 0, 3 domain colouring.
fn field_params(f: &FieldSpec) -> [f32; 28] {
    let kind = match f.kind {
        FieldKind::Hue => 0.0,
        FieldKind::Fill { greater: true } => 1.0,
        FieldKind::Fill { greater: false } => 2.0,
        FieldKind::Domain => 3.0,
    };
    let mut out = [0f32; 28];
    out[..12].copy_from_slice(&[
        f.color[0], f.color[1], f.color[2], f.color[3],
        f.rect_min[0], f.rect_min[1], f.rect_max[0], f.rect_max[1],
        f.origin_xy[0] as f32, f.origin_xy[1] as f32, kind, 0.0,
    ]);
    for (slot, v) in out[12..].iter_mut().zip(f.params.iter()) {
        *slot = *v;
    }
    out
}

impl Renderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let seg_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 28, shader_location: 4 },
        ];
        let mesh_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 2 },
        ];

        let depth_state = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let target = wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let target_flat = target.clone();
        let seg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("segments"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_seg",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &seg_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_seg",
                targets: &[Some(target.clone())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay segments"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_seg",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SegmentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &seg_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_seg",
                targets: &[Some(target.clone())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState { depth_write_enabled: false, ..depth_state.clone() }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surfaces"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &mesh_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_mesh",
                targets: &[Some(target)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let flat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat shapes"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &mesh_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_mesh",
                targets: &[Some(target_flat)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Translucent: test against surfaces, never write.
            depth_stencil: Some(wgpu::DepthStencilState { depth_write_enabled: false, ..depth_state }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let field_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field params layout"),
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
        });
        let field_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field pipeline layout"),
            bind_group_layouts: &[&bind_group_layout, &field_layout],
            push_constant_ranges: &[],
        });

        Renderer {
            seg_pipeline,
            overlay_pipeline,
            mesh_pipeline,
            flat_pipeline,
            bind_group_layout,
            depth: None,
            field_layout,
            field_pipeline_layout,
            field_cache: HashMap::new(),
            field_order: VecDeque::new(),
            format,
            field_compiles: 0,
            geo_cache: HashMap::new(),
            renders: 0,
            geo_uploads: 0,
            cache_geometry: true,
        }
    }

    /// Number of field pipelines compiled so far (test hook for the pipeline cache).
    /// Geometries uploaded so far (test hook).
    pub fn geo_uploads(&self) -> usize {
        self.geo_uploads
    }

    pub fn field_compiles(&self) -> usize {
        self.field_compiles
    }

    /// Compiles the field pipeline for `wgsl` (PRELUDE + `field_fn`) inside a validation error
    /// scope. Failures are logged once and recorded as FAILED; nothing panics.
    fn create_field_pipe(&self, device: &wgpu::Device, wgsl: &str) -> FieldPipe {
        let (common, rest) = include_str!("field.wgsl").split_once("// ==== fs_field ====").unwrap_or(("", ""));
        let (scalar, domain) = rest.split_once("// ==== fs_domain ====").unwrap_or(("", ""));
        // A module that defines `field_color` is a domain-colouring field; everything else
        // defines `field_fn`. Each fragment entry point is only compiled with its own function.
        let is_domain = wgsl.contains("fn field_color(");
        let (entry, frag) = if is_domain { ("fs_domain", domain) } else { ("fs_field", scalar) };
        let source = format!("{}\n{}\n{}\n{}", include_str!("render.wgsl"), wgsl, common, frag);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field"),
            layout: Some(&self.field_pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs_field", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: entry,
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            // Test against surfaces but never write: fields are translucent.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let state = Arc::new(AtomicU8::new(PIPE_PENDING));
        let mut scope = Box::pin(device.pop_error_scope());
        let settle = |err: Option<wgpu::Error>, state: &AtomicU8| {
            if let Some(e) = err {
                log::warn!("field shader rejected, field not drawn: {e}");
                state.store(PIPE_FAILED, Ordering::Relaxed);
            } else {
                state.store(PIPE_READY, Ordering::Relaxed);
            }
        };
        // Native backends resolve the scope immediately. Poll once with a no-op waker; if it is
        // still pending (the browser), finish it on the JS event loop.
        match scope.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(err) => settle(err, &state),
            Poll::Pending => {
                #[cfg(target_arch = "wasm32")]
                {
                    let st = state.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let err = scope.await;
                        settle(err, &st);
                    });
                }
                #[cfg(not(target_arch = "wasm32"))]
                state.store(PIPE_READY, Ordering::Relaxed);
            }
        }
        FieldPipe { pipeline, state }
    }

    /// Makes sure a pipeline exists for every distinct wgsl in `wanted`, evicting the oldest
    /// entries that are not needed this frame once the cache exceeds its cap.
    fn ensure_field_pipes(&mut self, device: &wgpu::Device, wanted: &HashSet<&str>) {
        for w in wanted {
            if !self.field_cache.contains_key(*w) {
                let pipe = self.create_field_pipe(device, w);
                self.field_compiles += 1;
                self.field_cache.insert((*w).to_string(), pipe);
                self.field_order.push_back((*w).to_string());
            }
        }
        let mut i = 0;
        while self.field_cache.len() > FIELD_CACHE_CAP && i < self.field_order.len() {
            if wanted.contains(self.field_order[i].as_str()) {
                i += 1;
            } else if let Some(k) = self.field_order.remove(i) {
                self.field_cache.remove(&k);
            }
        }
    }

    /// Uploads every geometry in `layers` that is not on the GPU yet and forgets the ones unused
    /// for a while. A still scene costs no buffer uploads per frame.
    fn ensure_geometry<'a>(&mut self, device: &wgpu::Device, layers: impl Iterator<Item = &'a Layer<'a>>) {
        self.renders += 1;
        let now = self.renders;
        if !self.cache_geometry {
            self.geo_cache.clear();
        }
        for l in layers {
            let g = l.geometry;
            let key = GeoKey::of(g);
            if let Some(e) = self.geo_cache.get_mut(&key) {
                e.1 = now;
                continue;
            }
            self.geo_uploads += 1;
            let segments = (!g.segments.is_empty()).then(|| {
                    let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("segments"),
                        contents: bytemuck::cast_slice(&g.segments),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    (b, g.segments.len() as u32)
                });
                let overlay = (!g.overlay_segments.is_empty()).then(|| {
                    let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay segments"),
                        contents: bytemuck::cast_slice(&g.overlay_segments),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    (b, g.overlay_segments.len() as u32)
                });
                let mesh = (!g.indices.is_empty() && !g.vertices.is_empty()).then(|| {
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh vertices"),
                        contents: bytemuck::cast_slice(&g.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh indices"),
                        contents: bytemuck::cast_slice(&g.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    (vb, ib, g.indices.len() as u32)
                });
                let flat = (!g.flat_indices.is_empty() && !g.vertices.is_empty()).then(|| {
                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("flat vertices"),
                        contents: bytemuck::cast_slice(&g.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("flat indices"),
                        contents: bytemuck::cast_slice(&g.flat_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    (vb, ib, g.flat_indices.len() as u32)
                });
            self.geo_cache.insert(key, (Arc::new(GeoBufs { segments, overlay, mesh, flat }), now));
        }
        self.geo_cache.retain(|_, e| now - e.1 <= GEO_CACHE_KEEP);
    }

    fn prepare_layers(
        &self,
        device: &wgpu::Device,
        cam_buf: &wgpu::Buffer,
        size: (u32, u32),
        rig: &Rig,
        layers: &[Layer],
    ) -> Vec<Prepared> {
        layers
        .iter()
        .map(|l| {
            let o = rig.render_origin();
            let off = [
                (l.origin[0] - o[0]) as f32,
                (l.origin[1] - o[1]) as f32,
                (l.origin[2] - o[2]) as f32,
            ];
            // offset.w: where world z = 0 is, relative to the current origin (the lift plane).
            let frame = [size.0 as f32, size.1 as f32, l.fade, l.lift, off[0], off[1], off[2], -o[2] as f32];
            let frame_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("frame"),
                contents: bytemuck::cast_slice(&frame),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame_buf.as_entire_binding() },
                ],
            });
            let g = l.geometry;
            let geo = self.geo_cache.get(&GeoKey::of(g)).map(|e| e.0.clone()).expect("geometry uploaded by ensure_geometry");
            let fields = g
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("field params"),
                        contents: bytemuck::cast_slice(&field_params(f)),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("field params"),
                        layout: &self.field_layout,
                        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
                    });
                    (i, bg)
                })
                .collect();
            Prepared { bind_group, geo, fields }
        })
        .collect()
    }

    fn draw_prepared<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, prepared: &'p [Prepared], layers: &[Layer]) {
            for p in prepared {
                if let Some((vb, ib, n)) = &p.geo.mesh {
                    pass.set_pipeline(&self.mesh_pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                }
            }
            // Fields: after opaque surfaces (so they are depth-tested against them) but before
            // lines/axes (which draw over them). No depth writes.
            for (p, l) in prepared.iter().zip(layers) {
                for (i, bg) in &p.fields {
                    let Some(fp) = self.field_cache.get(&l.geometry.fields[*i].wgsl) else { continue };
                    if fp.state.load(Ordering::Relaxed) != PIPE_READY {
                        continue;
                    }
                    pass.set_pipeline(&fp.pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
            // Flat translucent shapes (histogram bars): above surfaces and fields, below lines.
            for p in prepared {
                if let Some((vb, ib, n)) = &p.geo.flat {
                    pass.set_pipeline(&self.flat_pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                }
            }
            for p in prepared {
                if let Some((buf, n)) = &p.geo.segments {
                    pass.set_pipeline(&self.seg_pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..6, 0..*n);
                }
            }
            // Overlay lines (slice curves) last: tested against everything, writing no depth.
            for p in prepared {
                if let Some((buf, n)) = &p.geo.overlay {
                    pass.set_pipeline(&self.overlay_pipeline);
                    pass.set_bind_group(0, &p.bind_group, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..6, 0..*n);
                }
            }
    }

    fn ensure_depth(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let stale = !matches!(&self.depth, Some((_, s)) if *s == size);
        if stale {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((tex.create_view(&wgpu::TextureViewDescriptor::default()), size));
        }
    }

    /// Draws `layers` (in order) into `target`. `background` is the clear colour.
    #[allow(clippy::too_many_arguments)] // device/queue/target/size/rig/layers/background are distinct inputs
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        size: (u32, u32),
        rig: &Rig,
        layers: &[Layer],
        background: [f32; 4],
    ) {
        self.render_with_inset(device, queue, target, size, rig, layers, background, None);
    }

    /// Like [`Renderer::render`], plus an optional secondary view (the slice inset) drawn in a
    /// corner viewport. Both passes go into one command buffer; the inset pass loads the colour
    /// the main pass wrote, clears depth, and is confined by viewport and scissor to its
    /// rectangle with its own camera, so the main pass (and its crossfade layers) is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_inset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        size: (u32, u32),
        rig: &Rig,
        layers: &[Layer],
        background: [f32; 4],
        inset: Option<&Inset>,
    ) {
        let size = (size.0.max(1), size.1.max(1));
        let aspect = size.0 as f64 / size.1 as f64;
        let cam = camera_bytes(&rig.camera_uniform(aspect));
        let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::cast_slice(&cam),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wanted: HashSet<&str> = layers
            .iter()
            .chain(inset.iter().flat_map(|i| i.layers.iter()))
            .flat_map(|l| l.geometry.fields.iter().map(|f| f.wgsl.as_str()))
            .collect();
        self.ensure_field_pipes(device, &wanted);

        self.ensure_geometry(device, layers.iter().chain(inset.iter().flat_map(|i| i.layers.iter())));
        let prepared = self.prepare_layers(device, &cam_buf, size, rig, layers);

        // Inset: clamp its rectangle into the target, build its own camera and layers.
        let inset_rect = inset.map(|i| {
            let x = i.rect[0].min(size.0 - 1);
            let y = i.rect[1].min(size.1 - 1);
            let w = i.rect[2].clamp(1, size.0 - x);
            let h = i.rect[3].clamp(1, size.1 - y);
            [x, y, w, h]
        });
        let inset_prepared = inset.zip(inset_rect).map(|(i, r)| {
            let isize = (r[2], r[3]);
            let icam = camera_bytes(&i.rig.camera_uniform(isize.0 as f64 / isize.1 as f64));
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inset camera"),
                contents: bytemuck::cast_slice(&icam),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            self.prepare_layers(device, &ibuf, isize, i.rig, &i.layers)
        });

        self.ensure_depth(device, size);
        let depth_view = &self.depth.as_ref().unwrap().0;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scene") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0] as f64,
                            g: background[1] as f64,
                            b: background[2] as f64,
                            a: background[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.draw_prepared(&mut pass, &prepared, layers);
        }
        if let (Some(i), Some(r), Some(ip)) = (inset, inset_rect, inset_prepared.as_ref()) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("inset pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32, 0.0, 1.0);
            pass.set_scissor_rect(r[0], r[1], r[2], r[3]);
            self.draw_prepared(&mut pass, ip, &i.layers);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fades_into_3d_wait_for_the_lift_and_clear_the_grid_first() {
        let f0 = mode_fades(Mode::D2, Mode::D3, 0.0, 0.0);
        assert_eq!(f0, ModeFades { from_backdrop: 1.0, from_items: 1.0, to_backdrop: 0.0, to_items: 0.0 });
        let f1 = mode_fades(Mode::D2, Mode::D3, 1.0, 1.0);
        assert_eq!(f1, ModeFades { from_backdrop: 0.0, from_items: 0.0, to_backdrop: 1.0, to_items: 1.0 });
        // A flat sheet (lift <= start threshold) is invisible, so it cannot wash out the grid.
        for l in [0.0, 0.05, FADE_3D_START] {
            assert_eq!(mode_fades(Mode::D2, Mode::D3, 0.1, l).to_items, 0.0, "lift {l}");
        }
        // The grid is mostly gone before the 3D scene is half visible.
        let m = mode_fades(Mode::D2, Mode::D3, 0.5, 0.35);
        assert!(m.from_backdrop < 0.3, "{m:?}");
        assert!(m.to_items < 0.5, "{m:?}");
        // The curves stay (nearly) opaque while the surface rises, and only then go.
        assert!(mode_fades(Mode::D2, Mode::D3, 0.5, 0.5).from_items > 0.99);
        assert!(mode_fades(Mode::D2, Mode::D3, 0.5, 0.95).from_items < 0.01);
        // Monotone in the lift, and always within [0, 1].
        let mut prev = mode_fades(Mode::D2, Mode::D3, 0.0, 0.0);
        for i in 1..=100 {
            let m = mode_fades(Mode::D2, Mode::D3, 0.0, i as f32 / 100.0);
            assert!(m.to_items >= prev.to_items && m.from_items <= prev.from_items);
            assert!(m.from_backdrop <= prev.from_backdrop);
            for v in [m.from_backdrop, m.from_items, m.to_backdrop, m.to_items] {
                assert!((0.0..=1.0).contains(&v));
            }
            prev = m;
        }
        // Garbage lift is treated as flat.
        assert_eq!(mode_fades(Mode::D2, Mode::D3, 0.5, f32::NAN).to_items, 0.0);
    }

    #[test]
    fn fades_out_of_3d_mirror_the_way_in() {
        for i in 0..=20 {
            let l = i as f32 / 20.0;
            let a = mode_fades(Mode::D2, Mode::D3, 0.3, l);
            let b = mode_fades(Mode::D3, Mode::D1, 0.3, l);
            assert_eq!((a.to_items, a.to_backdrop), (b.from_items, b.from_backdrop));
            assert_eq!((a.from_items, a.from_backdrop), (b.to_items, b.to_backdrop));
        }
    }

    #[test]
    fn flat_switches_keep_the_progress_crossfade() {
        for p in [0.0f32, 0.3, 0.7, 1.0] {
            let (o, i) = crossfade(p);
            let m = mode_fades(Mode::D1, Mode::D2, p, 0.0);
            assert_eq!((m.from_backdrop, m.from_items, m.to_backdrop, m.to_items), (o, o, i, i));
        }
    }

    #[test]
    fn only_3d_scenes_are_lifted() {
        assert_eq!(layer_lift(Mode::D3, 0.25), 0.25);
        assert_eq!(layer_lift(Mode::D3, 1.0), 1.0);
        assert_eq!(layer_lift(Mode::D3, -1.0), 0.0);
        assert_eq!(layer_lift(Mode::D3, f64::NAN), 1.0);
        assert_eq!(layer_lift(Mode::D2, 0.0), 1.0);
        assert_eq!(layer_lift(Mode::D1, 0.5), 1.0);
    }

    #[test]
    fn crossfade_ends_and_staggers() {
        assert_eq!(crossfade(0.0), (1.0, 0.0));
        assert_eq!(crossfade(1.0), (0.0, 1.0));
        // The incoming scene is opaque before the outgoing one has faded much.
        let (out, inc) = crossfade(0.55);
        assert!(inc > 0.99 && out > 0.7);
    }
}
