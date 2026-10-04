//! Browser shell: wgpu surface on a canvas + the shared `App` and `Renderer`. The JS side owns
//! requestAnimationFrame and calls `Calculator::frame(now_ms)` each tick.

use crate::app::App;
use crate::render::Renderer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Calculator {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    app: App,
    backend: String,
}

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub async fn create_calculator(canvas: web_sys::HtmlCanvasElement) -> Result<Calculator, JsValue> {
    console_error_panic_hook::set_once();
    let (w, h) = (canvas.width().max(1), canvas.height().max(1));

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        // wgpu 0.19's WebGPU backend reads GPUSupportedLimits.maxInterStageShaderComponents, which
        // current Chrome removed, so it traps at adapter request. Use WebGL2 until wgpu is upgraded.
        backends: wgpu::Backends::GL,
        ..Default::default()
    });
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(err)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| err("no suitable GPU adapter"))?;
    let backend = format!("{:?}", adapter.get_info().backend);
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
                label: None,
            },
            None,
        )
        .await
        .map_err(err)?;

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: w,
        height: h,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };
    surface.configure(&device, &config);
    let renderer = Renderer::new(&device, format);
    Ok(Calculator { surface, device, queue, config, renderer, app: App::new((w, h)), backend })
}

#[wasm_bindgen]
impl Calculator {
    /// JSON command in, JSON array of events out.
    pub fn dispatch(&mut self, json: &str) -> String {
        self.app.dispatch(json)
    }

    /// JSON array of events raised since the last `dispatch`/`drain` (ticker `itemEdited`,
    /// `sliderValue`, `error`, `tickerState`, throttled `slice` updates from `frame`). `dispatch`
    /// also returns anything still pending, so calling `drain` is optional but never loses events.
    pub fn drain(&mut self) -> String {
        self.app.drain_json()
    }

    /// Advances the app and renders if needed. Returns whether a frame was rendered.
    pub fn frame(&mut self, now_ms: f64) -> bool {
        if !self.app.frame(now_ms) {
            return false;
        }
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return false,
                }
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let layers = self.app.layers();
        let inset = self.app.inset();
        self.renderer.render_with_inset(
            &self.device,
            &self.queue,
            &view,
            (self.config.width, self.config.height),
            &self.app.rig,
            &layers,
            self.app.background(),
            inset.as_ref(),
        );
        drop(inset);
        drop(layers);
        frame.present();
        true
    }

    /// Physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
        self.app.dispatch(&format!(r#"{{"t":"resize","width":{width},"height":{height}}}"#));
    }

    pub fn screen_labels_json(&self) -> String {
        serde_json::to_string(&self.app.screen_labels()).unwrap_or_else(|_| "[]".into())
    }

    pub fn backend_name(&self) -> String {
        self.backend.clone()
    }
}
