//! CPU-side scene description handed to the GPU renderer. Plain data, no wgpu types, so the
//! scene builder is unit-testable without a GPU.
//!
//! All positions are RELATIVE TO THE RENDER ORIGIN (the window centre, computed in f64 and
//! subtracted before rounding to f32), so deep zoom keeps precision.

use bytemuck::{Pod, Zeroable};

/// One thick line segment (or a dot when `p0 == p1`), drawn as an instanced screen-space quad
/// with round caps. `width` is in physical pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SegmentInstance {
    pub p0: [f32; 3],
    pub width: f32,
    pub p1: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

impl SegmentInstance {
    pub fn new(p0: [f32; 3], p1: [f32; 3], width: f32, color: [f32; 4]) -> Self {
        Self { p0, width, p1, _pad: 0.0, color }
    }

    /// The same segment pulled towards the camera by `bias` of NDC depth, so it stays visible on
    /// a surface it lies on (slice curves on a mesh).
    pub fn with_depth_bias(mut self, bias: f32) -> Self {
        self._pad = bias;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub _p0: f32,
    pub normal: [f32; 3],
    pub _p1: f32,
    pub color: [f32; 4],
}

impl MeshVertex {
    pub fn new(pos: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self { pos, _p0: 0.0, normal, _p1: 0.0, color }
    }
}

/// A tick or axis label, positioned in WORLD space (f64, not origin-relative) so a text layer
/// (DOM overlay on the web, a glyph pass natively) can project it itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub pos: [f64; 3],
    pub text: String,
    pub axis: u8,
    /// Id of the item an item label (`showLabel`) belongs to.
    pub item: Option<String>,
    /// The item's `labelOffset` (CSS pixels, y down, already clamped); `[0, 0]` when none.
    pub offset: [f64; 2],
}

/// How a [`FieldSpec`] is shaded by the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// Scalar field mapped to a cyclic hue ramp (see render.wgsl for the mapping).
    Hue,
    /// Inequality region: translucent item colour where `f > 0` (`greater`) or `f < 0`.
    Fill { greater: bool },
    /// Complex domain colouring: the module defines `fn field_color(x: f32, y: f32) -> vec4<f32>`
    /// (instead of `field_fn`) and the item colour's alpha is the layer opacity.
    Domain,
}

/// Most slider parameters one field can take as uniforms (4 `vec4`s); a field using more has all
/// its sliders folded into the shader text instead.
pub const MAX_FIELD_PARAMS: usize = 16;

/// A scalar field evaluated per pixel on the GPU over a rectangle of the z=0 plane.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSpec {
    pub kind: FieldKind,
    /// WGSL module text: `PRELUDE` plus `fn field_fn(v0: f32, v1: f32) -> f32` (v0 = x, v1 = y),
    /// or for [`FieldKind::Domain`] the complex prelude plus `fn field_color(x, y) -> vec4<f32>`.
    /// Slider values are NOT folded in: used sliders are extra inputs of the generated function,
    /// read from [`FieldSpec::params`] through a wrapper, so a slider drag leaves this string
    /// (the pipeline cache key) unchanged. Other definitions are folded in as constants.
    pub wgsl: String,
    /// Current values of the slider inputs (at most [`MAX_FIELD_PARAMS`]), uploaded as a uniform
    /// array each frame. Order matches the generated wrapper (sorted by slider name).
    pub params: Vec<f32>,
    pub color: [f32; 4],
    /// Origin-relative world rectangle covered (x, y at z = 0).
    pub rect_min: [f32; 2],
    pub rect_max: [f32; 2],
    /// Render origin (x, y) the rect is relative to; the shader adds it back before `field_fn`.
    pub origin_xy: [f64; 2],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneGeometry {
    pub fields: Vec<FieldSpec>,
    pub segments: Vec<SegmentInstance>,
    /// How many of the leading `segments` are the backdrop (grid, axes, arrows): a mode switch
    /// fades those on a different schedule than the items drawn after them.
    pub backdrop_segments: usize,
    /// Lines that lie ON surfaces (slice curves). Drawn after `segments` with depth test but no
    /// depth write, optionally pulled towards the camera by `SegmentInstance::with_depth_bias`.
    pub overlay_segments: Vec<SegmentInstance>,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    /// Triangles over the same `vertices` that are drawn UNLIT and translucent (flat 2D shapes
    /// such as histogram bars; their vertices carry a zero normal). Drawn after the lit mesh
    /// and fields, depth-tested but without depth writes, so alpha blending is order-safe.
    pub flat_indices: Vec<u32>,
    pub labels: Vec<Label>,
    /// Per-item problems (parse/compile errors) as `(item id, message)`; shown inline in the UI,
    /// never logged per frame.
    pub diagnostics: Vec<(String, String)>,
    /// Per-item read-outs for the shell (derivative expression, numeric value, regression fit);
    /// see [`ItemInfo`]. Sent to the UI as the `info` event when they change.
    pub infos: Vec<ItemInfo>,
    /// Resolved colour `[r,g,b,a]` (0..1, before opacity) of every drawn item, in drawing order,
    /// so the shell's list badges always match the canvas. Sent as the `colors` event.
    pub item_colors: Vec<(String, [f32; 4])>,
}

/// One fitted regression parameter.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct InfoParam {
    pub name: String,
    pub value: f64,
    #[serde(rename = "stdError", skip_serializing_if = "Option::is_none")]
    pub std_error: Option<f64>,
}

/// A read-out attached to an item. `kind` is `"derivative"`, `"value"` or `"regression"`; the
/// other fields are present as they apply (see `calc_draw.rs`).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct ItemInfo {
    pub id: String,
    pub kind: String,
    /// LaTeX for display (`\frac{d}{dx}x^{3} = 3 x^{2}`, a fitted equation, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latex: Option<String>,
    /// Plain text version (also what a screen reader should get).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<InfoParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    /// Regressions: Pearson's correlation `r` (straight-line fits only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r: Option<f64>,
    /// Regressions: the [`math_core::reg_family::Family`] key when the text is a panel template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Regressions: `y = 1.5 x - 0.6667` (LaTeX), when the model uses one data list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equation: Option<String>,
    /// Plain-text version of `equation`.
    #[serde(rename = "equationText", skip_serializing_if = "Option::is_none")]
    pub equation_text: Option<String>,
    /// Regressions: the residuals `y - fit`, one per data point (4 significant digits).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub residuals: Vec<f64>,
}

/// Colours for one theme. The page chrome and the canvas read the same theme so they cannot
/// drift out of sync.
#[derive(Clone, Debug, PartialEq)]
pub struct Theme {
    pub dark: bool,
    pub background: [f32; 4],
    pub grid_minor: [f32; 4],
    pub grid_major: [f32; 4],
    pub axis: [f32; 4],
    pub palette: Vec<[f32; 4]>,
}

impl Theme {
    pub fn light() -> Self {
        Theme {
            dark: false,
            background: [1.0, 1.0, 1.0, 1.0],
            grid_minor: [0.0, 0.0, 0.0, 0.07],
            grid_major: [0.0, 0.0, 0.0, 0.16],
            axis: [0.1, 0.1, 0.12, 0.85],
            palette: vec![
                [0.78, 0.18, 0.18, 1.0],
                [0.17, 0.38, 0.72, 1.0],
                [0.18, 0.55, 0.30, 1.0],
                [0.45, 0.27, 0.65, 1.0],
                [0.90, 0.50, 0.10, 1.0],
                [0.10, 0.10, 0.10, 1.0],
            ],
        }
    }

    pub fn dark() -> Self {
        Theme {
            dark: true,
            background: [0.07, 0.08, 0.11, 1.0],
            grid_minor: [1.0, 1.0, 1.0, 0.07],
            grid_major: [1.0, 1.0, 1.0, 0.17],
            axis: [0.92, 0.93, 0.96, 0.85],
            palette: vec![
                [0.98, 0.42, 0.42, 1.0],
                [0.45, 0.65, 1.0, 1.0],
                [0.40, 0.82, 0.55, 1.0],
                [0.75, 0.55, 0.95, 1.0],
                [1.0, 0.70, 0.30, 1.0],
                [0.92, 0.92, 0.92, 1.0],
            ],
        }
    }

    pub fn color(&self, index: usize) -> [f32; 4] {
        self.palette[index % self.palette.len()]
    }
}
