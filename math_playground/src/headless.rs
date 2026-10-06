//! Offscreen rendering to RGBA bytes / PNG files (native only). Lets tests and the
//! `render_png` tool look at real GPU output without a window.

use crate::render::{Inset, Layer, Renderer};
use math_core::view::Rig;

pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Headless {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub renderer: Renderer,
    pub adapter_name: String,
}

impl Headless {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: std::env::var("MATH_PLAYGROUND_SOFTWARE").is_ok(),
            })
            .await
            .ok_or("no suitable GPU adapter found")?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    label: None,
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;
        let renderer = Renderer::new(&device, FORMAT);
        Ok(Headless { device, queue, renderer, adapter_name })
    }

    /// Renders to tightly packed RGBA8 (`size.0 * size.1 * 4` bytes).
    pub fn render_rgba(
        &mut self,
        size: (u32, u32),
        rig: &Rig,
        layers: &[Layer],
        background: [f32; 4],
    ) -> Vec<u8> {
        self.render_rgba_with_inset(size, rig, layers, background, None)
    }

    /// Like [`Headless::render_rgba`] with an optional slice inset in a corner viewport.
    pub fn render_rgba_with_inset(
        &mut self,
        size: (u32, u32),
        rig: &Rig,
        layers: &[Layer],
        background: [f32; 4],
        inset: Option<&Inset>,
    ) -> Vec<u8> {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_with_inset(&self.device, &self.queue, &view, (w, h), rig, layers, background, inset);

        let unpadded = w * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map callback dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded * h) as usize);
        for row in 0..h {
            let start = (row * padded) as usize;
            out.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        buffer.unmap();
        out
    }
}

pub fn save_png(path: &std::path::Path, size: (u32, u32), rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size.0, size.1);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(rgba).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Theme;
    use crate::scene::build_scene;
    use math_core::doc::{Doc, Item, ItemKind, SliderCfg};
    use math_core::view::{Mode, Window3};

    fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [img[i], img[i + 1], img[i + 2], img[i + 3]]
    }

    #[test]
    fn a_still_scene_uploads_once_and_draws_the_same() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED a_still_scene_uploads_once_and_draws_the_same: no GPU adapter ({e})");
                return;
            }
        };
        let size = (320u32, 240u32);
        let theme = Theme::light();
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("a", ItemKind::Equation, "x^2+y^2+z^2=9")).unwrap();
        doc.add_item(Item::new("b", ItemKind::Equation, "y=x")).unwrap();
        let mut rig = Rig::new(Window3::new([-6.0; 3], [6.0; 3]), Mode::D3);
        rig.set_aspect(size.0 as f64 / size.1 as f64);
        let g = build_scene(&doc, Mode::D3, rig.window(), rig.render_origin(), size, &theme);
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let first = gpu.render_rgba(size, &rig, &layers, theme.background);
        let uploads = gpu.renderer.geo_uploads();
        assert_eq!(uploads, 1);
        for _ in 0..5 {
            // A different opacity / lift is only a uniform: still no upload, and it draws.
            let l2 = [Layer { geometry: &g, fade: 0.5, origin: rig.render_origin(), lift: 0.5 }];
            let _ = gpu.render_rgba(size, &rig, &l2, theme.background);
        }
        let again = gpu.render_rgba(size, &rig, &layers, theme.background);
        assert_eq!(gpu.renderer.geo_uploads(), uploads, "geometry was re-uploaded");
        assert_eq!(first, again, "a cached draw differs from the first one");
        // A rebuilt scene (new content) is uploaded and shows the new picture.
        let mut doc2 = Doc::new_default();
        doc2.add_item(Item::new("a", ItemKind::Equation, "x^2+y^2+z^2=16")).unwrap();
        let g2 = build_scene(&doc2, Mode::D3, rig.window(), rig.render_origin(), size, &theme);
        let l3 = [Layer { geometry: &g2, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let other = gpu.render_rgba(size, &rig, &l3, theme.background);
        assert_eq!(gpu.renderer.geo_uploads(), uploads + 1);
        assert_ne!(first, other);
        // Turning the cache off restores per-frame uploads (the measurement hook).
        gpu.renderer.cache_geometry = false;
        let n = gpu.renderer.geo_uploads();
        let _ = gpu.render_rgba(size, &rig, &layers, theme.background);
        let _ = gpu.render_rgba(size, &rig, &layers, theme.background);
        assert_eq!(gpu.renderer.geo_uploads(), n + 2);
    }

    #[test]
    fn inequality_fill_renders_on_gpu() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED inequality_fill_renders_on_gpu: no GPU adapter ({e})");
                return;
            }
        };
        let size = (900u32, 600u32);
        let theme = Theme::light();
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("a", ItemKind::Equation, "y<x^2")).unwrap();
        let mut rig = Rig::new(Window3::new([-10.0; 3], [10.0; 3]), Mode::D2);
        rig.set_aspect(size.0 as f64 / size.1 as f64);
        let g = build_scene(&doc, Mode::D2, rig.window(), rig.render_origin(), size, &theme);
        assert_eq!(g.fields.len(), 1);
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let img = gpu.render_rgba(size, &rig, &layers, theme.background);
        let bg = [255u8, 255, 255, 255];
        // (100, 453): left of the cup, y < x^2 (filled). (441, 200): inside the cup (empty).
        let inside = px(&img, size.0, 100, 453);
        let outside = px(&img, size.0, 441, 200);
        let diff = |a: [u8; 4], b: [u8; 4]| a.iter().zip(b).map(|(x, y)| (*x as i32 - y as i32).abs()).max().unwrap();
        assert!(diff(inside, bg) > 12, "filled pixel looks like background: {inside:?}");
        assert!(diff(outside, bg) <= 3, "unfilled pixel differs from background: {outside:?}");
    }

    #[test]
    fn bad_field_shader_does_not_panic() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED bad_field_shader_does_not_panic: no GPU adapter ({e})");
                return;
            }
        };
        let size = (64u32, 64u32);
        let theme = Theme::light();
        let mut g = crate::geometry::SceneGeometry::default();
        g.fields.push(crate::geometry::FieldSpec {
            kind: crate::geometry::FieldKind::Hue,
            wgsl: "fn field_fn(v0: f32, v1: f32) -> f32 { return nope; }".into(),
            params: Vec::new(),
            color: [1.0; 4],
            rect_min: [-1.0; 2],
            rect_max: [1.0; 2],
            origin_xy: [0.0; 2],
        });
        let mut rig = Rig::new(Window3::new([-10.0; 3], [10.0; 3]), Mode::D2);
        rig.set_aspect(1.0);
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let img = gpu.render_rgba(size, &rig, &layers, theme.background);
        assert!(img.chunks(4).all(|p| p == [255, 255, 255, 255]), "failed field must draw nothing");
    }

    #[test]
    fn domain_coloring_renders_on_gpu() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED domain_coloring_renders_on_gpu: no GPU adapter ({e})");
                return;
            }
        };
        let size = (900u32, 600u32);
        let theme = Theme::light();
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("c", ItemKind::Complex, "z")).unwrap();
        let mut rig = Rig::new(Window3::new([-10.0; 3], [10.0; 3]), Mode::D2);
        rig.set_aspect(size.0 as f64 / size.1 as f64);
        let g = build_scene(&doc, Mode::D2, rig.window(), rig.render_origin(), size, &theme);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert_eq!(g.fields.len(), 1);
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let img = gpu.render_rgba(size, &rig, &layers, theme.background);
        let rgb = |x: u32, y: u32| {
            let p = px(&img, size.0, x, y);
            [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]
        };
        let hue = |c: [f32; 3]| {
            let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
            let d = mx - mn;
            let h = if mx == c[0] {
                ((c[1] - c[2]) / d).rem_euclid(6.0)
            } else if mx == c[1] {
                (c[2] - c[0]) / d + 2.0
            } else {
                (c[0] - c[1]) / d + 4.0
            };
            h / 6.0
        };
        // The origin is at pixel (450, 300) (y down). Quadrant centres 40 px away on the
        // diagonals: arg = pi/4, 3pi/4, -3pi/4, -pi/4 -> hue 0.125, 0.375, 0.625, 0.875.
        let quads = [(490, 260, 0.125f32), (410, 260, 0.375), (410, 340, 0.625), (490, 340, 0.875)];
        let hues: Vec<f32> = quads.iter().map(|&(x, y, _)| hue(rgb(x, y))).collect();
        for (h, &(_, _, want)) in hues.iter().zip(&quads) {
            let d = (h - want).abs();
            assert!(d.min(1.0 - d) < 0.06, "hue {h} vs {want}: {hues:?}");
        }
        for i in 0..4 {
            for j in i + 1..4 {
                let d = (hues[i] - hues[j]).abs();
                assert!(d.min(1.0 - d) > 0.15, "quadrant hues too close: {hues:?}");
            }
        }
        // Next to the origin (3 px off the axis lines, which are drawn on top) it is dark.
        let near = rgb(453, 303);
        let far = rgb(490, 260);
        let lum = |c: [f32; 3]| c[0].max(c[1]).max(c[2]);
        assert!(lum(near) < 0.45 && lum(near) < lum(far) - 0.2, "near {near:?} far {far:?}");
    }

    fn diff(a: [u8; 4], b: [u8; 4]) -> i32 {
        a.iter().zip(b).map(|(x, y)| (*x as i32 - y as i32).abs()).max().unwrap()
    }

    fn rig_2d(size: (u32, u32)) -> Rig {
        let mut rig = Rig::new(Window3::new([-10.0; 3], [10.0; 3]), Mode::D2);
        rig.set_aspect(size.0 as f64 / size.1 as f64);
        rig
    }

    #[test]
    fn histogram_bars_render_translucent_on_gpu() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED histogram_bars_render_translucent_on_gpu: no GPU adapter ({e})");
                return;
            }
        };
        let size = (900u32, 600u32);
        let theme = Theme::light();
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("h", ItemKind::Equation, "histogram([1,2,2,3,3,3,4,4,5,9], 1)")).unwrap();
        let rig = rig_2d(size);
        let g = build_scene(&doc, Mode::D2, rig.window(), rig.render_origin(), size, &theme);
        assert_eq!(g.flat_indices.len(), 36);
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let img = gpu.render_rgba(size, &rig, &layers, theme.background);
        let bg = [255u8, 255, 255, 255];
        // 45 px per unit (the window fits x), origin at (450, 300). The [3,4) bar has height 3:
        // (3.5, 1.5) is inside it (px 607, 232); (-5, 5) is empty; (7, 1) is in the empty gap.
        let inside = px(&img, size.0, 607, 232);
        assert!(diff(inside, bg) > 40, "bar pixel looks like background: {inside:?}");
        assert!(inside[0] > inside[1] + 30, "bar is the first palette colour (red): {inside:?}");
        assert!(inside[1] > 60, "translucent, not opaque red: {inside:?}");
        // Empty areas (only faint grid lines): the gap between bars and the space above a bar.
        let mostly_bg = |x0: u32, y0: u32| {
            let n = (0..16).flat_map(|dy| (0..16).map(move |dx| (dx, dy)))
                .filter(|&(dx, dy)| diff(px(&img, size.0, x0 + dx, y0 + dy), bg) > 90)
                .count();
            n == 0
        };
        assert!(mostly_bg(217, 67) && mostly_bg(757, 247) && mostly_bg(599, 112));
        // A fading layer fades the bars too.
        let faded = [Layer { geometry: &g, fade: 0.3, origin: rig.render_origin(), lift: 1.0 }];
        let img2 = gpu.render_rgba(size, &rig, &faded, theme.background);
        let p2 = px(&img2, size.0, 607, 232);
        assert!(diff(p2, bg) > 5 && diff(p2, bg) < diff(inside, bg), "{p2:?} vs {inside:?}");
    }

    /// Dragging a slider must not compile a new pipeline: same shader text, new uniform.
    #[test]
    fn slider_drag_reuses_the_field_pipeline() {
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED slider_drag_reuses_the_field_pipeline: no GPU adapter ({e})");
                return;
            }
        };
        let size = (900u32, 600u32);
        let theme = Theme::light();
        let rig = rig_2d(size);
        for (latex, kind) in [("y<a*x", ItemKind::Equation), ("a*z", ItemKind::Complex)] {
            let mut doc = Doc::new_default();
            doc.add_item(Item::new("f", kind, latex)).unwrap();
            let before = gpu.renderer.field_compiles();
            let mut frames = Vec::new();
            for a in [2.0, -2.0, 0.5] {
                doc.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: a });
                let g = build_scene(&doc, Mode::D2, rig.window(), rig.render_origin(), size, &theme);
                assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
                let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
                frames.push((g.fields[0].wgsl.clone(), gpu.render_rgba(size, &rig, &layers, theme.background)));
            }
            assert!(frames.iter().all(|f| f.0 == frames[0].0), "{latex}: shader text must not change");
            assert_eq!(gpu.renderer.field_compiles() - before, 1, "{latex}: compiled once, then cached");
            let differing = |a: &[u8], b: &[u8]| a.chunks(4).zip(b.chunks(4)).filter(|(p, q)| p != q).count();
            assert!(differing(&frames[0].1, &frames[1].1) > 5000, "{latex}: a=2 vs a=-2 must look different");
            assert!(differing(&frames[0].1, &frames[2].1) > 1000, "{latex}: a=2 vs a=0.5 must look different");
        }
        // Spot check for the fill: the point right of the origin lies below y = 2x but not y = -2x.
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("f", ItemKind::Equation, "y<a*x")).unwrap();
        let mut at = |a: f64| {
            doc.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: a });
            let g = build_scene(&doc, Mode::D2, rig.window(), rig.render_origin(), size, &theme);
            let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
            px(&gpu.render_rgba(size, &rig, &layers, theme.background), size.0, 600, 330)
        };
        // Pixel (600, 330) is world (3.3, -0.7): below y = 2x yes; below y = -2x no.
        let (filled, empty) = (at(2.0), at(-2.0));
        assert!(diff(filled, [255, 255, 255, 255]) > 12, "{filled:?}");
        assert!(diff(empty, [255, 255, 255, 255]) <= 3, "{empty:?}");
    }

    #[test]
    fn slice_inset_renders_on_gpu() {
        use crate::render::Inset;
        use crate::scene::build_slice_panel;
        use math_core::slice::{parse_fixed_text, SliceCfg};
        let mut gpu = match Headless::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED slice_inset_renders_on_gpu: no GPU adapter ({e})");
                return;
            }
        };
        let size = (900u32, 600u32);
        let theme = Theme::light();
        let mut doc = Doc::new_default();
        doc.add_item(Item::new("s", ItemKind::Equation, "x^2+y^2+z^2=4")).unwrap();
        doc.slice = Some(SliceCfg::new(None, parse_fixed_text("z=1").unwrap(), Mode::D3).unwrap());
        let mut rig = Rig::new(Window3::new([-3.0; 3], [3.0; 3]), Mode::D3);
        rig.set_aspect(size.0 as f64 / size.1 as f64);
        let g = build_scene(&doc, Mode::D3, rig.window(), rig.render_origin(), size, &theme);
        let out = build_slice_panel(&doc, Mode::D3, rig.window(), size, &theme).unwrap().unwrap();
        let p = &out.panel;
        let inset = Inset {
            rect: p.rect,
            rig: &p.rig,
            layers: vec![Layer { geometry: &p.geometry, fade: 1.0, origin: p.origin, lift: 1.0 }],
        };
        let layers = [Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }];
        let with = gpu.render_rgba_with_inset(size, &rig, &layers, theme.background, Some(&inset));
        let without = gpu.render_rgba(size, &rig, &layers, theme.background);
        let [rx, ry, rw, rh] = p.rect;
        // Outside the inset rectangle the two renders are identical.
        for (x, y) in [(10u32, 10u32), (450, 100), (rx - 3, ry + rh / 2), (rx + rw / 2, ry - 3)] {
            assert_eq!(px(&with, size.0, x, y), px(&without, size.0, x, y), "({x},{y}) outside the inset");
        }
        // Inside: the inset background is the page background (not the 3D scene behind it) ...
        let corner = px(&with, size.0, rx + 3, ry + 3);
        assert!(corner.iter().take(3).all(|c| *c >= 200), "inset corner is light: {corner:?}");
        // ... and the circle of radius sqrt(3) is drawn in the item colour. The panel window is
        // 6 units tall; with the 4:3 aspect its x range is widened to 8 units over rw pixels.
        let per_unit = rw as f64 / 8.0;
        let cx = rx as f64 + rw as f64 / 2.0 + 3f64.sqrt() * per_unit;
        let cy = ry as f64 + rh as f64 / 2.0;
        let mut found = false;
        for dx in -2i32..=2 {
            let c = px(&with, size.0, (cx as i32 + dx) as u32, cy as u32);
            // theme palette 0 is a strong red
            if c[0] > 150 && c[1] < 90 && c[2] < 90 {
                found = true;
            }
        }
        assert!(found, "no red circle pixel at ({cx:.1}, {cy:.1})");
        // The inset covers what the main scene drew there.
        let differs = (ry..ry + rh).step_by(7).any(|y| (rx..rx + rw).step_by(7).any(|x| px(&with, size.0, x, y) != px(&without, size.0, x, y)));
        assert!(differs);
    }
}

