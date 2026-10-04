//! Native desktop shell: a winit window + wgpu surface driving the shared `App` and `Renderer`.

use crate::app::{App, Event as AppEvent, ScreenLabel};
use crate::render::Renderer;
use serde_json::json;
use std::sync::Arc;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

#[derive(Debug, Clone, Default)]
pub struct NativeArgs {
    pub exprs: Vec<String>,
    pub mode: Option<String>,
    pub dark: bool,
    pub hash: Option<String>,
    pub doc_file: Option<String>,
    pub angle: Option<String>,
}

fn report(events: &[AppEvent]) {
    for e in events {
        match e {
            AppEvent::Diagnostics { items } => {
                if items.is_empty() {
                    eprintln!("diagnostics: ok");
                }
                for d in items {
                    eprintln!("diagnostics [{}]: {}", d.id, d.message);
                }
            }
            AppEvent::Error { message } => eprintln!("error: {message}"),
            _ => {}
        }
    }
}

// ---- tick labels: a tiny embedded 5x7 bitmap font drawn as textured quads ------------------------

const GLYPHS: &str = "0123456789-.e+xyzf";
const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
#[rustfmt::skip]
const FONT: [[u8; GLYPH_H]; 18] = [
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], // 0
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 1
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111], // 2
    [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110], // 3
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], // 4
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], // 5
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110], // 6
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], // 7
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], // 8
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100], // 9
    [0, 0, 0, 0b11111, 0, 0, 0],                                     // -
    [0, 0, 0, 0, 0, 0b01100, 0b01100],                               // .
    [0, 0, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],             // e
    [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],             // +
    [0, 0, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],             // x
    [0, 0, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],             // y
    [0, 0, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],             // z
    [0b00110, 0b01000, 0b11110, 0b01000, 0b01000, 0b01000, 0b01000], // f
];

const LABEL_SHADER: &str = r#"
struct U { screen: vec2<f32>, pad: vec2<f32>, color: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) uv: vec2<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(p.x / u.screen.x * 2.0 - 1.0, 1.0 - p.y / u.screen.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    let a = textureSample(tex, samp, i.uv).r;
    return vec4<f32>(u.color.rgb, u.color.a * a);
}
"#;

struct LabelPainter {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    cap: usize, // vertices
}

impl LabelPainter {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let (tw, th) = (GLYPHS.len() * GLYPH_W, GLYPH_H);
        let mut texels = vec![0u8; tw * th];
        for (g, rows) in FONT.iter().enumerate() {
            for (y, bits) in rows.iter().enumerate() {
                for x in 0..GLYPH_W {
                    if bits & (1 << (GLYPH_W - 1 - x)) != 0 {
                        texels[y * tw + g * GLYPH_W + x] = 255;
                    }
                }
            }
        }
        let size = wgpu::Extent3d { width: tw as u32, height: th as u32, depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("label font"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &texels,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(tw as u32), rows_per_image: Some(th as u32) },
            size,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default()); // nearest, clamped
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("label uniform"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("label shader"),
            source: wgpu::ShaderSource::Wgsl(LABEL_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("label pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let cap = 1024;
        let vbuf = Self::make_vbuf(device, cap);
        LabelPainter { pipeline, bind_group, uniform, vbuf, cap }
    }

    fn make_vbuf(device: &wgpu::Device, verts: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("label verts"),
            size: (verts * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Draws `labels` over `view`. `scale` is the integer pixel size of one font texel.
    #[allow(clippy::too_many_arguments)] // wgpu handles + label params; a struct would only be used here
    fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
        scale: f32,
        background: [f32; 4],
        labels: &[ScreenLabel],
    ) {
        let (sw, sh) = (size.0 as f32, size.1 as f32);
        let adv = (GLYPH_W as f32 + 1.0) * scale;
        let (gw, gh) = (GLYPH_W as f32 * scale, GLYPH_H as f32 * scale);
        let n_glyphs = GLYPHS.len() as f32;
        let mut verts: Vec<[f32; 4]> = Vec::new();
        for l in labels.iter().filter(|l| l.visible) {
            let chars: Vec<usize> = l.text.chars().filter_map(|c| GLYPHS.find(c)).collect();
            if chars.is_empty() {
                continue;
            }
            let w = chars.len() as f32 * adv - scale;
            let (x, y) = (l.x as f32, l.y as f32);
            // x ticks sit just below their line, y ticks just left of theirs, z ticks centred.
            let (x0, y0) = match l.axis {
                0 => (x - w / 2.0, y + 5.0 * scale / 2.0),
                1 => (x - w - 3.0 * scale, y - gh / 2.0),
                _ => (x - w / 2.0, y - gh / 2.0),
            };
            // Skip labels that would be clipped by the window edge (or, for the slice inset's
            // labels, by the inset's rectangle; the glyphs here are larger than the layout
            // estimate the app uses, so this is checked again with the real size).
            let [cx0, cy0, cx1, cy1] = match l.clip {
                Some(c) => [c[0] as f32, c[1] as f32, (c[0] + c[2]) as f32, (c[1] + c[3]) as f32],
                None => [0.0, 0.0, sw, sh],
            };
            if x0 < cx0 || y0 < cy0 || x0 + w > cx1 || y0 + gh > cy1 {
                continue;
            }
            for (i, g) in chars.iter().enumerate() {
                let (px, py) = (x0.round() + i as f32 * adv, y0.round());
                let (u0, u1) = (*g as f32 / n_glyphs, (*g + 1) as f32 / n_glyphs);
                let q = [
                    [px, py, u0, 0.0],
                    [px + gw, py, u1, 0.0],
                    [px, py + gh, u0, 1.0],
                    [px + gw, py + gh, u1, 1.0],
                ];
                verts.extend_from_slice(&[q[0], q[1], q[2], q[2], q[1], q[3]]);
            }
        }
        if verts.is_empty() {
            return;
        }
        if verts.len() > self.cap {
            self.cap = verts.len().next_power_of_two();
            self.vbuf = Self::make_vbuf(device, self.cap);
        }
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&verts));
        // Muted ink that contrasts with the background.
        let lum = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2];
        let ink = if lum > 0.5 { [0.25, 0.27, 0.31, 1.0] } else { [0.72, 0.75, 0.80, 1.0] };
        let u: [f32; 8] = [sw, sh, 0.0, 0.0, ink[0], ink[1], ink[2], ink[3]];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&u));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tick labels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..(verts.len() * 16) as u64));
        pass.draw(0..verts.len() as u32, 0..1);
    }
}

struct Gfx<'w> {
    surface: wgpu::Surface<'w>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    labels: LabelPainter,
}

impl Gfx<'_> {
    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

fn dispatch(app: &mut App, cmd: serde_json::Value) {
    let out = app.dispatch(&cmd.to_string());
    if let Ok(evs) = serde_json::from_str::<Vec<serde_json::Value>>(&out) {
        for e in evs {
            match e["t"].as_str() {
                Some("diagnostics") => {
                    let items = e["items"].as_array().cloned().unwrap_or_default();
                    if items.is_empty() {
                        eprintln!("diagnostics: ok");
                    }
                    for d in items {
                        eprintln!("diagnostics [{}]: {}", d["id"].as_str().unwrap_or("?"), d["message"].as_str().unwrap_or(""));
                    }
                }
                Some("error") => eprintln!("error: {}", e["message"].as_str().unwrap_or("")),
                _ => {}
            }
        }
    }
}

pub fn run_native(args: NativeArgs) {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Math Playground")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 720))
            .with_resizable(true)
            .build(&event_loop)
            .expect("window"),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("no suitable GPU adapter");
    eprintln!("adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            label: None,
        },
        None,
    ))
    .expect("request_device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };
    surface.configure(&device, &config);
    let renderer = Renderer::new(&device, format);
    let labels = LabelPainter::new(&device, &queue, format);
    let mut gfx = Gfx { surface, device, queue, config, renderer, labels };

    let mut app = App::new((size.width.max(1), size.height.max(1)));
    if args.dark {
        dispatch(&mut app, json!({"t":"setTheme","dark":true}));
    }
    if let Some(h) = &args.hash {
        dispatch(&mut app, json!({"t":"loadHash","hash":h}));
    }
    if let Some(f) = &args.doc_file {
        match std::fs::read_to_string(f) {
            Ok(s) => dispatch(&mut app, json!({"t":"loadDoc","json":s})),
            Err(e) => eprintln!("error: {f}: {e}"),
        }
    }
    if let Some(m) = &args.mode {
        dispatch(&mut app, json!({"t":"setMode","mode":m}));
    }
    if let Some(a) = &args.angle {
        dispatch(&mut app, json!({"t":"setAngle","angle":a}));
    }
    for (i, e) in args.exprs.iter().enumerate() {
        dispatch(&mut app, json!({"t":"setExpr","id":format!("e{i}"),"latex":e}));
    }

    let start = instant::Instant::now();
    let mut cursor = (0.0f64, 0.0f64);
    let mut pressed: Option<u8> = None;
    let mut shift = false;
    let mut dark = args.dark;
    let mut ortho = false;
    let mut need_render = true;
    event_loop.set_control_flow(ControlFlow::Wait);

    event_loop
        .run(|event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(PhysicalSize { width, height }) => {
                    gfx.resize(width, height);
                    need_render = true; // reconfiguring the surface discards the old frame
                    dispatch(&mut app, json!({"t":"resize","width":width,"height":height}));
                    window.request_redraw();
                }
                WindowEvent::ModifiersChanged(m) => shift = m.state().shift_key(),
                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x, position.y);
                    if pressed.is_some() {
                        dispatch(&mut app, json!({"t":"pointer","phase":"move","x":cursor.0,"y":cursor.1}));
                        window.request_redraw();
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let b = match button {
                        MouseButton::Left => 0u8,
                        MouseButton::Middle => 1,
                        MouseButton::Right => 2,
                        _ => return,
                    };
                    match state {
                        ElementState::Pressed => {
                            pressed = Some(b);
                            dispatch(
                                &mut app,
                                json!({"t":"pointer","phase":"down","x":cursor.0,"y":cursor.1,"button":b,"shift":shift}),
                            );
                        }
                        ElementState::Released => {
                            if pressed == Some(b) {
                                pressed = None;
                                dispatch(&mut app, json!({"t":"pointer","phase":"up","x":cursor.0,"y":cursor.1}));
                            }
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -(y as f64) * 100.0,
                        MouseScrollDelta::PixelDelta(p) => -p.y,
                    };
                    dispatch(&mut app, json!({"t":"wheel","x":cursor.0,"y":cursor.1,"dy":dy}));
                    window.request_redraw();
                }
                WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !event.repeat => {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => elwt.exit(),
                        Key::Character(c) => match c.to_lowercase().as_str() {
                            "1" => dispatch(&mut app, json!({"t":"setMode","mode":"1d"})),
                            "2" => dispatch(&mut app, json!({"t":"setMode","mode":"2d"})),
                            "3" => dispatch(&mut app, json!({"t":"setMode","mode":"3d"})),
                            "d" => {
                                dark = !dark;
                                dispatch(&mut app, json!({"t":"setTheme","dark":dark}));
                            }
                            "r" => dispatch(&mut app, json!({"t":"reset"})),
                            "o" => {
                                ortho = !ortho;
                                dispatch(&mut app, json!({"t":"setOrtho","ortho":ortho}));
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                    window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    let now = start.elapsed().as_secs_f64() * 1000.0;
                    let redraw = app.frame(now);
                    report(&app.take_events());
                    if redraw || need_render {
                        need_render = false;
                        let frame = match gfx.surface.get_current_texture() {
                            Ok(f) => f,
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                gfx.surface.configure(&gfx.device, &gfx.config);
                                need_render = true;
                                window.request_redraw();
                                return;
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                eprintln!("out of memory");
                                elwt.exit();
                                return;
                            }
                            Err(e) => {
                                eprintln!("surface error: {e:?}");
                                return;
                            }
                        };
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let layers = app.layers();
                        let inset = app.inset();
                        gfx.renderer.render_with_inset(
                            &gfx.device,
                            &gfx.queue,
                            &view,
                            (gfx.config.width, gfx.config.height),
                            &app.rig,
                            &layers,
                            app.background(),
                            inset.as_ref(),
                        );
                        drop(inset);
                        drop(layers);
                        let scale = (2.0 * window.scale_factor()).round().max(2.0) as f32;
                        let mut enc = gfx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("labels") });
                        gfx.labels.draw(
                            &gfx.device,
                            &gfx.queue,
                            &mut enc,
                            &view,
                            (gfx.config.width, gfx.config.height),
                            scale,
                            app.background(),
                            &app.screen_labels(),
                        );
                        gfx.queue.submit(Some(enc.finish()));
                        frame.present();
                    }
                    if redraw || app.rig.is_animating() {
                        window.request_redraw();
                    }
                }
                _ => {}
            },
            _ => {}
        })
        .expect("event loop");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_covers_tick_text_and_axis_names() {
        assert_eq!(GLYPHS.chars().count(), FONT.len(), "one bitmap per glyph");
        // Everything `format_tick` can emit plus the axis names of the main scene and the inset.
        for c in "0123456789-.e+xyzf".chars() {
            assert!(GLYPHS.contains(c), "no glyph for {c:?}");
        }
        for (i, rows) in FONT.iter().enumerate() {
            assert!(rows.iter().any(|r| *r != 0), "glyph {i} is blank");
            assert!(rows.iter().all(|r| *r < 1 << GLYPH_W), "glyph {i} is wider than {GLYPH_W}");
        }
    }
}
