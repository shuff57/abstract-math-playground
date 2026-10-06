//! Scene builder: turns a [`Doc`] into drawable [`SceneGeometry`] for ONE display mode.
//!
//! Pure CPU, no wgpu. The same expression is deliberately drawn differently per mode (a curve
//! in 2D, a surface in 3D, its roots on the number line in 1D); there is no automatic switching.
//!
//! Conventions and defaults:
//! * All positions are origin-relative: `(p - origin)` is computed in f64, then cast to f32.
//!   Mesh vertices from `math_core::mesh` are f32 and window-absolute, so at extreme zoom far
//!   from the origin a 3D surface keeps only f32 precision before rebasing (limitation).
//! * Undefined variables (e.g. `y=a*x` with no definition and no slider) produce a DIAGNOSTIC
//!   naming the variable; nothing is drawn for that item and other items are unaffected.
//! * Hidden items are not drawn and report no errors, but their definitions (`a=3`, `f(x)=..`)
//!   stay in scope. Empty rows are ignored silently.
//! * Parametric `t` range is `[0, 2*pi]` (`[0, 360]` in degree mode, where trig takes degrees).
//!   Polar `theta` range is `[0, 4*pi]` when theta only occurs inside trig functions, else
//!   `[0, 6*pi]` (spirals); degrees equivalents in degree mode. A trailing `{a<=t<=b}` replaces
//!   the default (`{a<=theta<=b}` for polar curves; bounds may use constants, `pi` and sliders;
//!   a single end moves only that end).
//! * `(x(u,v), y(u,v), z(u,v))` is a parametric SURFACE over `u, v` in `[0, 2*pi]` (degrees:
//!   `[0, 360]`) unless a `{a<=u<=b, c<=v<=d}` range says otherwise. It is drawn in 3D mode only;
//!   in 2D mode it draws nothing (like `z = f(x, y)`), without a diagnostic.
//! * `x=f(t)` and `y=g(t)` (and `z=h(t)`) as separate rows, or `x=f(t), y=g(t)` in one row,
//!   are one parametric curve.
//! * Line widths are physical pixels: grid 1.0/1.5, axes 2.0, curves 2.5, dots 9.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use math_core::analyze::{analyze, Kind};
use math_core::ast::{BinOp, Expr, Rel};
use math_core::compile::{compile, Angle, Program};
use math_core::complex::{compile_complex, parse_complex};
use math_core::doc::{
    AngleMode, Doc, Item, ItemKind, ItemStyle, LineStyle, PointStyle, Weight,
};
use math_core::list::{eval_value, Bindings, Value};
use math_core::mesh;
use math_core::mesh_param;
use math_core::parse::{parse_with, ParseCtx};
use math_core::resolve::Defs;
use math_core::slice::ResolvedSlice;
use math_core::special::find_roots;
use math_core::stats;
use math_core::table::{ColumnStyle, ParsedColumn};
use math_core::view::{Mode, Window3};
use math_core::wgsl::emit_module;
use math_core::wgsl_complex::{emit_complex_function, emit_domain_module, COMPLEX_PRELUDE};

mod slice_draw;
pub use slice_draw::{
    inset_rect, label_box, label_box_inside, SlicePanel, ViewReq, LABEL_CHAR_W, LABEL_GAP_X,
    LABEL_GAP_Y, LABEL_H, SLICE_COLOR,
};

use crate::axis_map::AxisMap;
use crate::geometry::{
    FieldKind, FieldSpec, Label, MeshVertex, SceneGeometry, SegmentInstance, Theme,
    MAX_FIELD_PARAMS,
};

mod calc_draw;

const MINOR_W: f32 = 1.0;
const MAJOR_W: f32 = 1.5;
const AXIS_W: f32 = 2.0;
const CURVE_W: f32 = 2.5;
const DOT_W: f32 = 9.0;
/// Width of a table column's point outline ring (pixels on each side).
const OUTLINE_W: f32 = 2.0;
const TICK_PX: f64 = 4.0;
/// Largest point size (pixels) the print weight enlarges to.
const MAX_POINT_PX: f32 = 40.0;
/// Print weight multipliers of (curve/axis/tick line widths, grid line widths, point sizes). The
/// grid scales less than the curves so that "extra bold" does not bury the plot in grid.
const WEIGHT_MUL: [(Weight, f32, f32, f32); 3] = [
    (Weight::Normal, 1.0, 1.0, 1.0),
    (Weight::Bold, 1.6, 1.3, 1.3),
    (Weight::Extra, 2.2, 1.5, 1.6),
];
/// The multipliers `(lines, grid, points)` of a print weight.
pub(crate) fn weight_mul(w: Weight) -> (f32, f32, f32) {
    let (_, l, g, p) = WEIGHT_MUL
        .into_iter()
        .find(|(x, ..)| *x == w)
        .unwrap_or(WEIGHT_MUL[0]);
    (l, g, p)
}
const MESH_ALPHA: f32 = 0.92;
/// Opacity of a domain-coloured complex plane in 3D (so surfaces/axes behind still read).
const DOMAIN_ALPHA_3D: f32 = 0.9;
/// Target distance between major grid lines in pixels.
const TARGET_MAJOR_PX: f64 = 100.0;
/// Fill opacity of histogram bars (the grid and curves read through; see `quad`).
const BAR_ALPHA: f32 = 0.55;
/// Fill opacity of a box plot's box.
const BOX_ALPHA: f32 = 0.4;
/// Outline width of histogram bars.
const BAR_OUTLINE_W: f32 = 1.5;
/// Half the height of a box plot's box, in pixels.
const BOX_HALF_PX: f64 = 18.0;
/// Most bins a histogram / dot plot may have.
const MAX_BINS: usize = 2000;
/// Most dots stacked in one dot-plot bin.
const MAX_STACK: usize = 300;
/// Number lists up to this long get a value label per dot on the 1D line.
const MAX_LABELLED: usize = 20;
/// Safety cap on grid lines per direction.
const MAX_LINES: i64 = 600;
/// Vector-field arrow shaft width in physical pixels.
const ARROW_W: f32 = 2.0;
/// Target distance between 2D/1D vector-field arrows in pixels.
const ARROW_CELL_PX: f64 = 42.0;
/// Most arrows in a 2D vector field (the grid spacing doubles until it fits).
const MAX_ARROWS_2D: usize = 4000;
/// Most lattice points per axis in a 3D vector field (so at most 12^3 arrows).
const MAX_LATTICE: usize = 12;
/// Half-angle of an arrowhead's barbs.
const BARB_ANGLE: f64 = 0.45;
/// Alpha factor the field shader (and the CPU fallback) applies to an inequality fill; an item's
/// `fillOpacity` is divided by it so that it becomes the effective opacity of the shading.
const FILL_SHADER_ALPHA: f32 = 0.22;
/// Most dash pieces one polyline may produce; the rest of a longer one is drawn solid.
/// Length in pixels of an axis arrowhead.
const ARROW_PX: f64 = 10.0;
const MAX_DASH_PIECES: usize = 50_000;
/// Most `showLabel` point labels per item.
const MAX_POINT_LABELS: usize = 100;
/// Label axis code of item labels (`showLabel`): 0/1 are x/y ticks, 2 the z ticks, 3 a title.
pub const ITEM_LABEL_AXIS: u8 = 4;
/// Segments of an open-circle point.
const RING_SIDES: usize = 20;
/// Label axis code of the angle labels of the polar grid's spokes.
pub const POLAR_LABEL_AXIS: u8 = 5;
/// Most circles of the polar grid (major and minor together).
const MAX_CIRCLES: usize = 200;
/// Message of an item that cannot be drawn on logarithmic axes.
pub const LOG_UNSUPPORTED: &str = "not available on a logarithmic axis";

// ---------------------------------------------------------------------------------------------
// Nice spacing and number formatting
// ---------------------------------------------------------------------------------------------

/// Chooses a "nice" major spacing (1, 2 or 5 times a power of ten) so that major lines are
/// about `target_px` pixels apart when `span` world units cover `px` pixels. Degenerate input
/// (non-finite, zero) falls back to a spacing for a unit span.
pub fn nice_step(span: f64, px: f64, target_px: f64) -> f64 {
    let span = if span.is_finite() && span > 0.0 {
        span.abs()
    } else {
        1.0
    };
    let px = if px.is_finite() && px >= 1.0 { px } else { 1.0 };
    let raw = span / px * target_px;
    let e = raw.log10().floor().clamp(-300.0, 300.0);
    let base = 10f64.powf(e);
    let m = raw / base;
    let mut best = 1.0;
    let mut best_d = f64::INFINITY;
    for c in [1.0, 2.0, 5.0, 10.0] {
        let d = (c / m).ln().abs();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best * base
}

/// Number of minor subdivisions per major step: 4 for a leading digit of 2, otherwise 5.
pub fn minor_divisions(step: f64) -> i64 {
    if !(step.is_finite() && step > 0.0) {
        return 5;
    }
    let m = step / 10f64.powf(step.log10().floor());
    if (m - 2.0).abs() < 0.5 {
        4
    } else {
        5
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// Scientific notation with up to `digits` mantissa decimals, trailing zeros trimmed
/// (`1e-6`, `2.5e6`).
fn fmt_sci(v: f64, digits: usize) -> String {
    let s = format!("{:.*e}", digits.min(15), v);
    match s.split_once('e') {
        Some((m, e)) => format!("{}e{}", trim_zeros(m), e),
        None => s,
    }
}

/// Formats a tick value compactly given the tick spacing: plain decimals normally, scientific
/// notation (`1e-6`, `2e7`) for tiny or huge magnitudes.
pub fn format_tick(v: f64, step: f64) -> String {
    if !v.is_finite() {
        return String::new();
    }
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    };
    if v.abs() < step * 1e-6 {
        return "0".to_string();
    }
    let step_e = step.log10().floor();
    if v.abs() >= 1e6 || v.abs() < 1e-4 {
        let digits = (v.abs().log10().floor() - step_e).clamp(0.0, 15.0) as usize;
        return fmt_sci(v, digits);
    }
    let decimals = (-step_e).clamp(0.0, 12.0) as usize;
    let s = trim_zeros(&format!("{:.*}", decimals, v));
    if s == "-0" {
        "0".to_string()
    } else {
        s
    }
}

/// Formats a free-standing value (a dot's label) with up to ~10 significant digits.
pub fn format_value(v: f64) -> String {
    if !v.is_finite() {
        return String::new();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e6 || v.abs() < 1e-4 {
        return fmt_sci(v, 6);
    }
    trim_zeros(&format!("{:.10}", v))
}

/// Multiples `k*step` of `step` inside `[lo, hi]` as `(k, position)`, empty if there would be
/// more than [`MAX_LINES`].
fn multiples(lo: f64, hi: f64, step: f64) -> Vec<(i64, f64)> {
    if !(lo.is_finite() && hi.is_finite() && step.is_finite() && step > 0.0 && hi >= lo) {
        return Vec::new();
    }
    let k0 = (lo / step).ceil();
    let k1 = (hi / step).floor();
    if !(k0.is_finite() && k1.is_finite()) || k1 - k0 > MAX_LINES as f64 {
        return Vec::new();
    }
    (k0 as i64..=k1 as i64)
        .map(|k| (k, k as f64 * step))
        .collect()
}

fn parse_hex_color(s: &str) -> Option<[f32; 4]> {
    let h = s.strip_prefix('#')?;
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let c = |i: usize| {
        u8::from_str_radix(&h[i..i + 2], 16)
            .ok()
            .map(|v| v as f32 / 255.0)
    };
    Some([c(0)?, c(2)?, c(4)?, 1.0])
}

// ---------------------------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------------------------

/// How one item is drawn.
#[derive(Clone, Copy)]
struct Style {
    color: [f32; 4],
    line_w: f32,
    /// Dash pattern `(on, off)` in pixels along the curve (`on == 0` draws dots); `None` is solid.
    dash: Option<(f64, f64)>,
    /// Marker of the item's points and its diameter in pixels.
    point: PointStyle,
    point_size: f32,
    /// Effective opacity of an inequality fill (`None`: the built-in shading).
    fill: Option<f32>,
}

impl Style {
    /// Solid lines, dot markers of the default size, default fill.
    fn new(color: [f32; 4], line_w: f32) -> Style {
        Style {
            color,
            line_w,
            dash: None,
            point: PointStyle::Dot,
            point_size: DOT_W,
            fill: None,
        }
    }

    /// The style of an item with resolved `color` (opacity already applied).
    fn for_item(s: &ItemStyle, color: [f32; 4]) -> Style {
        let line_w = s.line_width.map(|w| w as f32).unwrap_or(CURVE_W);
        let w = line_w as f64;
        // Segments have round caps of radius w/2, so the visible dash is `on + w` long and the
        // visible gap `off - w`.
        let dash = match s.line_style {
            Some(LineStyle::Dashed) => {
                let (on, off) = ((3.0 * w).max(8.0), (2.0 * w).max(5.0));
                Some(((on - w).max(1.0), off + w))
            }
            Some(LineStyle::Dotted) => Some((0.0, (2.5 * w).max(4.0))),
            _ => None,
        };
        Style {
            color,
            line_w,
            dash,
            point: s.point_style.unwrap_or(PointStyle::Dot),
            point_size: s.point_size.map(|v| v as f32).unwrap_or(DOT_W),
            fill: s.fill_opacity.map(|v| v.clamp(0.0, 1.0) as f32),
        }
    }
}

/// Per-build settings of the [`Builder`] beyond the window (view flags, point labels).
struct BuildExt {
    mode: Mode,
    grid: bool,
    axes: bool,
    axis_numbers: bool,
    minor_grid: bool,
    arrows: bool,
    /// Axis names (x, y) and fixed major steps (x, y) from the view.
    axis_names: [Option<String>; 2],
    fixed_steps: [Option<f64>; 2],
    /// Circles and spokes instead of the rectangular 2D grid.
    polar: bool,
    /// World -> display map of logarithmic 2D axes (linear outside 2D).
    map: AxisMap,
    /// `point` and `polyline` receive WORLD coordinates and map them (tables on log axes).
    world_in: bool,
    /// `showLabel` of the item being drawn: custom text (or `None` for coordinates) and how many
    /// labels it may still place.
    labels: Option<(Option<String>, usize)>,
    /// Print weight multiplier of line widths (curves, axes, ticks, arrows; dashes follow).
    line_mul: f32,
    /// Print weight multiplier of the grid lines (and the 3D box edges); gentler than `line_mul`
    /// so a heavy weight does not turn the grid into a wall.
    grid_mul: f32,
    /// Print weight multiplier of point sizes (before the clamp, see [`Builder::point_px`]).
    point_mul: f32,
    /// Transient render scale (export at a larger pixel size), already part of both multipliers.
    scale: f32,
}

impl Default for BuildExt {
    fn default() -> Self {
        BuildExt {
            mode: Mode::D2,
            grid: true,
            axes: true,
            axis_numbers: true,
            minor_grid: true,
            arrows: false,
            axis_names: [None, None],
            fixed_steps: [None, None],
            polar: false,
            map: AxisMap::LINEAR,
            world_in: false,
            labels: None,
            line_mul: 1.0,
            grid_mul: 1.0,
            point_mul: 1.0,
            scale: 1.0,
        }
    }
}

struct Prepared<'a> {
    item: &'a Item,
    kind: Kind,
    dims: [bool; 3],
    /// Set for `ItemKind::Complex` items (parsed with the complex-aware parser); `kind` is then
    /// an inert placeholder and the item is drawn by `Builder::draw_complex`.
    complex: Option<Expr>,
    /// The parsed expression as written (before definitions are resolved).
    expr: Expr,
    /// The ranges of a trailing `{a<=t<=b}` (see `math_core::param`).
    domain: Vec<math_core::param::Range>,
}

/// A table item with its cells parsed (see `math_core::table`).
struct TableData<'a> {
    item: &'a Item,
    cols: Vec<ParsedColumn>,
}

struct Builder<'a> {
    out: SceneGeometry,
    origin: [f64; 3],
    win: Window3,
    vw: f64,
    vh: f64,
    theme: &'a Theme,
    angle: Angle,
    /// Definitions with SLIDER NAMES LEFT FREE (fields compile them as uniform inputs).
    pdefs: Defs,
    /// Current slider values by name.
    sliders: BTreeMap<String, f64>,
    /// Mode, view flags and per-item label state.
    ext: BuildExt,
}

/// WGSL expression reading slider parameter `i` from the field uniform.
fn param_ref(i: usize) -> String {
    format!("fp.params[{}].{}", i / 4, ["x", "y", "z", "w"][i % 4])
}

const STAT_PLOTS: &[&str] = &["histogram", "boxplot", "dotplot"];

/// True if the (resolved) expression is list-valued or a statistical plot call, i.e. should be
/// evaluated by `list::eval_value` instead of compiled as a scalar.
fn is_listish(e: &Expr) -> bool {
    match e {
        Expr::Num(_) | Expr::Var(_) => false,
        Expr::List(_) => true,
        Expr::Call(n, args) => {
            matches!(n.as_str(), "range" | "for" | "index")
                || STAT_PLOTS.contains(&n.as_str())
                || args.iter().any(is_listish)
        }
        Expr::Neg(a) => is_listish(a),
        Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => is_listish(a) || is_listish(b),
        Expr::Tuple(v) => v.iter().any(is_listish),
    }
}

/// Default bin width: Freedman-Diaconis (`2 * IQR * n^(-1/3)`), falling back to Sturges
/// (`range / (ceil(log2 n) + 1)`) when the IQR is zero or FD would make too many bins, and to 1
/// for a constant list. `data` is non-empty and finite.
fn default_bin_width(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let (mn, mx) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| {
            (a.0.min(*v), a.1.max(*v))
        });
    let range = mx - mn;
    if range <= 0.0 {
        return 1.0;
    }
    let iqr = match (stats::quantile(data, 0.25), stats::quantile(data, 0.75)) {
        (Ok(a), Ok(b)) => b - a,
        _ => 0.0,
    };
    let fd = 2.0 * iqr * n.powf(-1.0 / 3.0);
    if fd.is_finite() && fd > 0.0 && range / fd <= MAX_BINS as f64 {
        fd
    } else {
        range / (n.log2().ceil() + 1.0)
    }
}

/// Equal-width bins aligned to multiples of `w` (left-closed, so `[k*w, (k+1)*w)`; the maximum
/// lands in the last bin). Returns `(start, w, bins)` with each bin holding its values.
fn bin_values(data: &[f64], width: Option<f64>) -> Result<(f64, f64, Vec<Vec<f64>>), String> {
    let (mn, mx) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| {
            (a.0.min(*v), a.1.max(*v))
        });
    let w = match width {
        Some(w) if w.is_finite() && w > 0.0 => w,
        Some(_) => return Err("the bin width must be a positive number".to_string()),
        None => default_bin_width(data),
    };
    let start = (mn / w).floor() * w;
    let slot = |v: f64| (((v - start) / w) + 1e-9).floor().max(0.0);
    let nb = slot(mx) + 1.0;
    if nb.is_nan() || nb > MAX_BINS as f64 {
        return Err(format!(
            "too many bins (more than {MAX_BINS}); use a larger bin width"
        ));
    }
    let nb = nb as usize;
    let mut bins = vec![Vec::new(); nb];
    for v in data {
        bins[(slot(*v) as usize).min(nb - 1)].push(*v);
    }
    Ok((start, w, bins))
}

impl<'a> Builder<'a> {
    fn rb(&self, p: [f64; 3]) -> [f32; 3] {
        [
            (p[0] - self.origin[0]) as f32,
            (p[1] - self.origin[1]) as f32,
            (p[2] - self.origin[2]) as f32,
        ]
    }

    /// A line segment of `w` nominal pixels, drawn at the print weight.
    fn seg(&mut self, a: [f64; 3], b: [f64; 3], w: f32, color: [f32; 4]) {
        self.seg_raw(a, b, w * self.ext.line_mul, color);
    }

    /// A grid line of `w` nominal pixels, drawn at the (gentler) grid print weight.
    fn gseg(&mut self, a: [f64; 3], b: [f64; 3], w: f32, color: [f32; 4]) {
        self.seg_raw(a, b, w * self.ext.grid_mul, color);
    }

    /// A point size in pixels at the print weight, at most [`MAX_POINT_PX`] (never below the
    /// requested size).
    fn point_px(&self, v: f32) -> f32 {
        let (m, s) = (self.ext.point_mul, self.ext.scale);
        if m == s {
            return v * m;
        }
        (v * m).min((MAX_POINT_PX * s).max(v * s))
    }

    /// A dot of `w` nominal pixels, drawn at the print weight.
    fn dseg(&mut self, p: [f64; 3], w: f32, color: [f32; 4]) {
        self.seg_raw(p, p, self.point_px(w), color);
    }

    /// [`Builder::seg`] with the final width (no print weight applied).
    fn seg_raw(&mut self, a: [f64; 3], b: [f64; 3], w: f32, color: [f32; 4]) {
        let (p0, p1) = (self.rb(a), self.rb(b));
        if p0.iter().chain(p1.iter()).all(|v| v.is_finite()) {
            self.out
                .segments
                .push(SegmentInstance::new(p0, p1, w, color));
        }
    }

    fn dot(&mut self, p: [f64; 3], color: [f32; 4]) {
        self.dseg(p, DOT_W, color);
    }

    /// A flat, UNLIT, translucent axis-aligned rectangle on the plane z = 0 (histogram bars,
    /// box plot box). Zero-normal `MeshVertex` quads go to `flat_indices`: the renderer draws
    /// them after the lit mesh and fields with depth TEST but no depth WRITE, so translucent
    /// bars neither hide the grid/curves nor depend on draw order.
    fn quad(&mut self, x0: f64, x1: f64, y0: f64, y1: f64, color: [f32; 4]) {
        let corners =
            [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]].map(|c| self.rb(c));
        if corners.iter().flatten().any(|v| !v.is_finite()) {
            return;
        }
        let base = self.out.vertices.len() as u32;
        for c in corners {
            self.out.vertices.push(MeshVertex::new(c, [0.0; 3], color));
        }
        self.out
            .flat_indices
            .extend([0, 1, 2, 0, 2, 3].map(|i| i + base));
    }

    /// World units per pixel along y (an approximation in 3D, where scale varies).
    fn y_per_px(&self, mode: Mode) -> f64 {
        let (vw, vh) = self.px();
        let span = self.win.max[1] - self.win.min[1];
        match mode {
            Mode::D3 => span / (0.7 * vw.min(vh)),
            _ => span / vh,
        }
    }

    fn label(&mut self, pos: [f64; 3], text: String, axis: u8) {
        if !text.is_empty() && pos.iter().all(|v| v.is_finite()) {
            self.out.labels.push(Label { pos, text, axis });
        }
    }

    fn polyline(&mut self, pts: &[[f64; 3]], st: Style) {
        if self.ext.world_in && !self.ext.map.is_linear() {
            // World points on logarithmic axes: straight in display space between mapped points;
            // a point that cannot be shown (<= 0 on a log axis) breaks the line.
            let m = self.ext.map;
            self.ext.world_in = false;
            for run in pts.split(|p| !m.fwd3(*p).iter().all(|v| v.is_finite())) {
                let mapped: Vec<[f64; 3]> = run.iter().map(|p| m.fwd3(*p)).collect();
                self.polyline(&mapped, st);
            }
            self.ext.world_in = true;
            return;
        }
        if let Some((on, off)) = st.dash {
            let m = self.ext.line_mul as f64;
            return self.dashed(pts, st, on * m, off * m);
        }
        for w in pts.windows(2) {
            self.seg(w[0], w[1], st.line_w, st.color);
        }
    }

    /// Pixels per world unit along each axis (3D: an approximation, the scale varies).
    fn px_per_unit(&self) -> [f64; 3] {
        let (vw, vh) = self.px();
        let span = |a: usize| (self.win.max[a] - self.win.min[a]).max(1e-300);
        match self.ext.mode {
            Mode::D1 => [vw / span(0); 3],
            Mode::D2 => [vw / span(0), vh / span(1), vw / span(0)],
            Mode::D3 => {
                let p = 0.7 * vw.min(vh);
                [p / span(0), p / span(1), p / span(2)]
            }
        }
    }

    /// `polyline` split into dashes of `on` pixels every `on + off` pixels (a dot every period
    /// when `on` is 0). The pattern continues across vertices.
    fn dashed(&mut self, pts: &[[f64; 3]], st: Style, on: f64, off: f64) {
        let k = self.px_per_unit();
        let period = (on + off).max(1.0);
        let mut budget = MAX_DASH_PIECES;
        let mut s = 0.0f64;
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len =
                ((d[0] * k[0]).powi(2) + (d[1] * k[1]).powi(2) + (d[2] * k[2]).powi(2)).sqrt();
            if !len.is_finite() || len <= 0.0 {
                continue;
            }
            let at = |t: f64| [a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t];
            let s1 = s + len;
            let pieces = len / period + 2.0;
            if pieces > budget as f64 {
                // Absurdly long (off-screen) runs: draw the rest solid rather than stall.
                self.seg(a, b, st.line_w, st.color);
                s = s1;
                continue;
            }
            let mut n = ((s - on) / period).floor().max(0.0);
            loop {
                let ds = n * period;
                if ds >= s1 {
                    break;
                }
                if on <= 0.0 {
                    if ds >= s {
                        let p = at((ds - s) / len);
                        self.seg(p, p, st.line_w, st.color);
                    }
                } else {
                    let (lo, hi) = (ds.max(s), (ds + on).min(s1));
                    if hi > lo {
                        self.seg(at((lo - s) / len), at((hi - s) / len), st.line_w, st.color);
                    }
                }
                n += 1.0;
                budget = budget.saturating_sub(1);
            }
            s = s1;
        }
    }

    /// One point marker in the item's style (`pointStyle`, `pointSize`), plus its `showLabel`
    /// label. Open shapes are sized in pixels; in 3D every style is a dot. `p` is in WORLD
    /// coordinates (mapped onto logarithmic axes; a point that cannot be shown there is skipped).
    fn point(&mut self, p: [f64; 3], st: Style) {
        let p = self.ext.map.fwd3(p);
        if !p.iter().all(|v| v.is_finite()) {
            return;
        }
        let size = self.point_px(st.point_size) as f64;
        let k = self.px_per_unit();
        let off = |dx: f64, dy: f64| [p[0] + dx / k[0], p[1] + dy / k[1], p[2]];
        match (st.point, self.ext.mode) {
            (PointStyle::Dot, _) | (_, Mode::D3) => self.seg_raw(p, p, size as f32, st.color),
            (PointStyle::Circle, _) => {
                let rw = (size * 0.22).max(1.5);
                let r = ((size - rw) * 0.5).max(0.5);
                let ring: Vec<[f64; 3]> = (0..=RING_SIDES)
                    .map(|i| {
                        let t = 2.0 * PI * i as f64 / RING_SIDES as f64;
                        off(r * t.cos(), r * t.sin())
                    })
                    .collect();
                for w in ring.windows(2) {
                    self.seg_raw(w[0], w[1], rw as f32, st.color);
                }
            }
            (PointStyle::Cross, _) => {
                let cw = (size * 0.22).max(1.5);
                let h = ((size - cw) * 0.5).max(0.5);
                self.seg_raw(off(-h, -h), off(h, h), cw as f32, st.color);
                self.seg_raw(off(-h, h), off(h, -h), cw as f32, st.color);
            }
            (PointStyle::Square, _) => {
                let sw = (size * 0.18).max(1.5);
                let h = ((size - sw) * 0.5).max(0.5);
                let c = [off(-h, -h), off(h, -h), off(h, h), off(-h, h), off(-h, -h)];
                for w in c.windows(2) {
                    self.seg_raw(w[0], w[1], sw as f32, st.color);
                }
            }
        }
        self.point_label(p);
    }

    /// The `showLabel` label of a drawn point (axis code [`ITEM_LABEL_AXIS`], anchored at the point): the item's
    /// label text, else its coordinates. In 1D the value is already labelled, so only custom
    /// text is added there.
    fn point_label(&mut self, p: [f64; 3]) {
        let mode = self.ext.mode;
        let Some((text, left)) = self.ext.labels.as_mut() else {
            return;
        };
        if *left == 0 || !p.iter().all(|v| v.is_finite()) {
            return;
        }
        let text = match (text, mode) {
            (Some(t), _) => t.clone(),
            (None, Mode::D1) => return,
            (None, Mode::D2) => {
                let w = self.ext.map.inv3(p);
                format!("({}, {})", format_value(w[0]), format_value(w[1]))
            }
            (None, Mode::D3) => format!(
                "({}, {}, {})",
                format_value(p[0]),
                format_value(p[1]),
                format_value(p[2])
            ),
        };
        *left -= 1;
        self.label(p, text, ITEM_LABEL_AXIS);
    }

    /// Arms `showLabel` point labels for the item about to be drawn.
    fn begin_item_labels(&mut self, s: &ItemStyle) {
        self.ext.labels = s.show_label.then(|| {
            let text = s.label.clone().filter(|t| !t.trim().is_empty());
            (text, MAX_POINT_LABELS)
        });
    }

    /// Ends the item's labels. An item with `showLabel` and label text that placed no point
    /// label (a curve) gets the text at its first drawn sample inside the window.
    fn end_item_labels(&mut self, s: &ItemStyle, seg0: usize) {
        let Some((Some(text), left)) = self.ext.labels.take() else {
            return;
        };
        if left < MAX_POINT_LABELS || !s.show_label {
            return;
        }
        let (lo, hi) = (self.win.min, self.win.max);
        let m = |a: usize| 0.05 * (hi[a] - lo[a]);
        let dims = self.ext.mode.dims() as usize;
        let at = self.out.segments[seg0.min(self.out.segments.len())..]
            .iter()
            .map(|sg| {
                [
                    sg.p0[0] as f64 + self.origin[0],
                    sg.p0[1] as f64 + self.origin[1],
                    sg.p0[2] as f64 + self.origin[2],
                ]
            })
            .find(|p| (0..dims.clamp(2, 3)).all(|a| p[a] >= lo[a] + m(a) && p[a] <= hi[a] - m(a)));
        if let Some(p) = at {
            self.label(p, text, ITEM_LABEL_AXIS);
        }
    }

    fn prog(&self, e: &Expr, vars: &[&str]) -> Result<Program, String> {
        compile(e, vars, self.angle).map_err(|e| e.to_string())
    }

    /// Pixels per axis in the viewport (at least 1).
    fn px(&self) -> (f64, f64) {
        (self.vw.max(1.0), self.vh.max(1.0))
    }

    /// Clamp the axis crossing (0) into the window range.
    fn clamp0(&self, axis: usize) -> f64 {
        0f64.clamp(self.win.min[axis], self.win.max[axis])
    }

    // ----- grid and axes ------------------------------------------------------------------

    /// Per-axis major steps for the mode.
    /// The view's fixed major step for axis `a`, unless it would draw an unreadable number of
    /// lines (then the automatic step is used).
    fn fixed_step(&self, a: usize) -> Option<f64> {
        let step = self.ext.fixed_steps[a]?;
        let (vw, vh) = self.px();
        let px = if a == 0 { vw } else { vh };
        let span = self.win.max[a] - self.win.min[a];
        (step > 0.0 && span / step <= px / 6.0).then_some(step)
    }

    fn steps(&self, mode: Mode) -> [f64; 3] {
        let (vw, vh) = self.px();
        let span = |a: usize| self.win.max[a] - self.win.min[a];
        match mode {
            Mode::D1 => {
                let s = nice_step(span(0), vw, TARGET_MAJOR_PX);
                [s, s, s]
            }
            Mode::D2 => [
                self.fixed_step(0)
                    .unwrap_or_else(|| nice_step(span(0), vw, TARGET_MAJOR_PX)),
                self.fixed_step(1)
                    .unwrap_or_else(|| nice_step(span(1), vh, TARGET_MAJOR_PX)),
                1.0,
            ],
            Mode::D3 => {
                let px = 0.7 * vw.min(vh);
                [
                    nice_step(span(0), px, TARGET_MAJOR_PX),
                    nice_step(span(1), px, TARGET_MAJOR_PX),
                    nice_step(span(2), px, TARGET_MAJOR_PX),
                ]
            }
        }
    }

    /// Grid, axes and tick numbers as the view flags allow (`grid`, `axes`, `axisNumbers`); the
    /// 3D box edges are always drawn.
    fn grid_and_axes(&mut self, mode: Mode) {
        let steps = self.steps(mode);
        let (grid, axes) = (self.ext.grid, self.ext.axes);
        match mode {
            Mode::D1 => {
                if axes {
                    self.axes_1d(steps[0]);
                }
            }
            Mode::D2 if !self.ext.map.is_linear() => self.log_grid_and_axes(steps),
            Mode::D2 if self.ext.polar => {
                // One spacing for circles and ticks (the per-axis steps do not apply).
                let (vw, _) = self.px();
                let s = nice_step(self.win.max[0] - self.win.min[0], vw, TARGET_MAJOR_PX);
                if grid {
                    self.polar_grid(s);
                }
                if axes {
                    self.axes_2d([s, s, 1.0]);
                }
            }
            Mode::D2 => {
                if grid {
                    self.plane_grid(0.0, steps);
                }
                if axes {
                    self.axes_2d(steps);
                }
            }
            Mode::D3 => {
                let zp = if self.win.min[2] <= 0.0 && 0.0 <= self.win.max[2] {
                    0.0
                } else {
                    self.win.min[2]
                };
                if grid {
                    self.plane_grid(zp, steps);
                }
                self.box_edges();
                if axes {
                    self.axes_3d(steps);
                }
            }
        }
    }

    /// Minor then major grid lines on the plane `z = zp`, over the window's x/y ranges.
    fn plane_grid(&mut self, zp: f64, steps: [f64; 3]) {
        let (minor_c, major_c) = (self.theme.grid_minor, self.theme.grid_major);
        let (lo, hi) = (self.win.min, self.win.max);
        for pass_major in [false, true] {
            for axis in 0..2usize {
                let other = 1 - axis;
                let div = minor_divisions(steps[axis]);
                let minor = steps[axis] / div as f64;
                for (k, v) in multiples(lo[axis], hi[axis], minor) {
                    let is_major = k % div == 0;
                    if is_major != pass_major || (!is_major && !self.ext.minor_grid) {
                        continue;
                    }
                    let mut a = [0.0; 3];
                    let mut b = [0.0; 3];
                    a[axis] = v;
                    b[axis] = v;
                    a[other] = lo[other];
                    b[other] = hi[other];
                    a[2] = zp;
                    b[2] = zp;
                    let (w, c) = if is_major {
                        (MAJOR_W, major_c)
                    } else {
                        (MINOR_W, minor_c)
                    };
                    self.gseg(a, b, w, c);
                }
            }
        }
    }

    /// The part of the segment `a`-`b` inside the 2D window (Liang-Barsky), if any.
    fn clip_to_window(&self, a: [f64; 2], b: [f64; 2]) -> Option<([f64; 2], [f64; 2])> {
        let (lo, hi) = (self.win.min, self.win.max);
        let d = [b[0] - a[0], b[1] - a[1]];
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for k in 0..2 {
            for (p, q) in [(-d[k], a[k] - lo[k]), (d[k], hi[k] - a[k])] {
                if p == 0.0 {
                    if q < 0.0 {
                        return None;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 {
                        t0 = t0.max(r);
                    } else {
                        t1 = t1.min(r);
                    }
                }
            }
        }
        (t0 <= t1).then(|| {
            let at = |t: f64| [a[0] + d[0] * t, a[1] + d[1] * t];
            (at(t0), at(t1))
        })
    }

    /// Polar grid: circles every `step` around the origin (minor circles between them when the
    /// view has minor gridlines) and spokes every 30 degrees labelled with their angle. Only the
    /// visible arcs of the circles are emitted.
    fn polar_grid(&mut self, step: f64) {
        let (minor_c, major_c) = (self.theme.grid_minor, self.theme.grid_major);
        let (lo, hi) = (self.win.min, self.win.max);
        let (vw, _) = self.px();
        let upp = (hi[0] - lo[0]) / vw;
        // Nearest and farthest distance of the window from the origin.
        let near = [0.0f64.clamp(lo[0], hi[0]), 0.0f64.clamp(lo[1], hi[1])];
        let r_near = near[0].hypot(near[1]);
        let r_far = [lo[0].abs().max(hi[0].abs()), lo[1].abs().max(hi[1].abs())];
        let r_far = r_far[0].hypot(r_far[1]);
        // Angular range of the window as seen from the origin (all of it when it is inside).
        let inside = r_near == 0.0;
        let (th0, th1) = if inside {
            (0.0, 2.0 * PI)
        } else {
            let mid = (0.5 * (lo[1] + hi[1])).atan2(0.5 * (lo[0] + hi[0]));
            let rel = |x: f64, y: f64| {
                let mut d = y.atan2(x) - mid;
                while d > PI {
                    d -= 2.0 * PI;
                }
                while d < -PI {
                    d += 2.0 * PI;
                }
                d
            };
            let ds = [
                rel(lo[0], lo[1]),
                rel(hi[0], lo[1]),
                rel(hi[0], hi[1]),
                rel(lo[0], hi[1]),
            ];
            let (a, b) = ds
                .iter()
                .fold((0.0f64, 0.0f64), |(a, b), d| (a.min(*d), b.max(*d)));
            (mid + a, mid + b)
        };
        let div = minor_divisions(step);
        let minor = step / div as f64;
        let ring_step = if self.ext.minor_grid { minor } else { step };
        let k0 = (r_near / ring_step).ceil().max(1.0);
        let k1 = (r_far / ring_step).floor();
        if k1 - k0 < MAX_CIRCLES as f64 {
            for pass_major in [false, true] {
                let mut k = k0;
                while k <= k1 {
                    let r = k * ring_step;
                    let is_major = !self.ext.minor_grid || (k as i64) % div == 0;
                    k += 1.0;
                    if is_major != pass_major {
                        continue;
                    }
                    let (w, c) = if is_major {
                        (MAJOR_W, major_c)
                    } else {
                        (MINOR_W, minor_c)
                    };
                    // Chords that stray at most ~0.3 px from the circle (at most 720 per circle).
                    let dth = 2.0 * (0.6 / (r / upp).max(1.0)).sqrt();
                    let n = ((th1 - th0) / dth).ceil().clamp(8.0, 720.0) as usize;
                    let pt = |i: usize| {
                        let t = th0 + (th1 - th0) * i as f64 / n as f64;
                        [r * t.cos(), r * t.sin()]
                    };
                    let mut prev = pt(0);
                    for i in 1..=n {
                        let cur = pt(i);
                        if let Some((a, b)) = self.clip_to_window(prev, cur) {
                            self.gseg([a[0], a[1], 0.0], [b[0], b[1], 0.0], w, c);
                        }
                        prev = cur;
                    }
                }
            }
        }
        // Spokes every 30 degrees, from the origin to the window edge, labelled near the edge.
        let deg = self.angle == Angle::Deg;
        let labels = self.ext.axis_numbers;
        let (pad_x, pad_y) = (22.0 * upp, 14.0 * upp);
        for j in 0..12u32 {
            let t = j as f64 * PI / 6.0;
            let far = [r_far * 1.01 * t.cos(), r_far * 1.01 * t.sin()];
            let Some((a, b)) = self.clip_to_window([0.0, 0.0], far) else {
                continue;
            };
            self.gseg([a[0], a[1], 0.0], [b[0], b[1], 0.0], MAJOR_W, major_c);
            // Every spoke but 0 (the positive x axis) is labelled with its angle.
            if !labels || j == 0 {
                continue;
            }
            let p = [
                (b[0] - t.cos() * pad_x).clamp(lo[0] + pad_x, hi[0] - pad_x),
                (b[1] - t.sin() * pad_y).clamp(lo[1] + pad_y, hi[1] - pad_y),
            ];
            if p[0] * t.cos() + p[1] * t.sin() > 0.0 {
                self.label(
                    [p[0], p[1], 0.0],
                    polar_angle_text(j, deg),
                    POLAR_LABEL_AXIS,
                );
            }
        }
    }

    /// Grid, axes and numbers when at least one 2D axis is logarithmic (see [`AxisMap`]): a
    /// logarithmic axis has major lines at powers of ten and minor lines at 2..9 times them; a
    /// linear one keeps its steps. An axis line is drawn only where the other axis has a 0.
    fn log_grid_and_axes(&mut self, steps: [f64; 3]) {
        let (vw, vh) = self.px();
        let (lo, hi) = (self.win.min, self.win.max);
        let ticks = [
            self.log_ticks(0, steps[0], vw),
            self.log_ticks(1, steps[1], vh),
        ];
        let (minor_c, major_c, axis_c) = (
            self.theme.grid_minor,
            self.theme.grid_major,
            self.theme.axis,
        );
        if self.ext.grid {
            for pass_major in [false, true] {
                for axis in 0..2usize {
                    let other = 1 - axis;
                    for t in &ticks[axis] {
                        if t.major != pass_major || (!t.major && !self.ext.minor_grid) {
                            continue;
                        }
                        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
                        a[axis] = t.pos;
                        b[axis] = t.pos;
                        a[other] = lo[other];
                        b[other] = hi[other];
                        let (w, c) = if t.major {
                            (MAJOR_W, major_c)
                        } else {
                            (MINOR_W, minor_c)
                        };
                        self.gseg(a, b, w, c);
                    }
                }
            }
        }
        if !self.ext.axes {
            return;
        }
        let map = self.ext.map;
        // Where world 0 of each axis is in display space (none on a logarithmic axis).
        let zero = |a: usize| (!map.log[a] && lo[a] <= 0.0 && 0.0 <= hi[a]).then_some(0.0);
        let upp = [(hi[0] - lo[0]) / vw, (hi[1] - lo[1]) / vh];
        for axis in 0..2usize {
            let other = 1 - axis;
            // The axis line sits at the other axis's 0; without one its numbers hug the
            // bottom / left edge.
            let at = match zero(other) {
                Some(z) => {
                    let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
                    a[axis] = lo[axis];
                    b[axis] = hi[axis];
                    a[other] = z;
                    b[other] = z;
                    self.seg(a, b, AXIS_W, axis_c);
                    for t in ticks[axis].iter().filter(|t| t.major) {
                        let h = TICK_PX * self.ext.line_mul as f64 * upp[other];
                        let (mut p, mut q) = ([0.0; 3], [0.0; 3]);
                        p[axis] = t.pos;
                        q[axis] = t.pos;
                        p[other] = z - h;
                        q[other] = z + h;
                        self.seg(p, q, MINOR_W * 1.5, axis_c);
                    }
                    z
                }
                None if map.log[other] || lo[other] > 0.0 => {
                    lo[other] + if axis == 0 { 22.0 } else { 46.0 } * upp[other]
                }
                None => self.clamp0(other),
            };
            if self.ext.axis_numbers {
                let labels: Vec<(f64, String)> = ticks[axis]
                    .iter()
                    .filter_map(|t| t.text.clone().map(|s| (t.pos, s)))
                    .collect();
                for (pos, text) in labels {
                    let mut p = [0.0; 3];
                    p[axis] = pos;
                    p[other] = at;
                    self.label(p, text, axis as u8);
                }
            }
            if let Some(n) = self.ext.axis_names[axis].clone() {
                let mut p = [0.0; 3];
                p[axis] = hi[axis] - 0.02 * (hi[axis] - lo[axis]);
                p[other] = at;
                self.label(p, n, axis as u8);
            }
        }
    }

    /// Grid lines and numbers along display axis `axis` (`px` pixels long) for the log-axes
    /// grid: decades (every n-th when they are crowded) and their 2..9 multiples on a
    /// logarithmic axis, plain multiples of `step` on a linear one.
    fn log_ticks(&self, axis: usize, step: f64, px: f64) -> Vec<Tick> {
        let (lo, hi) = (self.win.min[axis], self.win.max[axis]);
        let map = self.ext.map;
        let mut out = Vec::new();
        if !map.log[axis] {
            let div = minor_divisions(step);
            for (k, v) in multiples(lo, hi, step / div as f64) {
                let major = k % div == 0;
                out.push(Tick {
                    pos: v,
                    major,
                    text: major.then(|| format_tick(v, step)),
                });
            }
            return out;
        }
        let k = map.k[axis];
        // Pixels per decade, and the decades in view.
        let ppd = k * px / (hi - lo).max(1e-300);
        let (e0, e1) = ((lo / k).floor(), (hi / k).ceil());
        if !(e0.is_finite() && e1.is_finite()) || e1 - e0 > 2.0 * MAX_LINES as f64 {
            return out;
        }
        if ppd > 2500.0 {
            // Less than a decade across: plain steps in world units, mapped.
            let (w0, w1) = (map.inv(axis, lo), map.inv(axis, hi));
            let s = nice_step(w1 - w0, px, TARGET_MAJOR_PX);
            let div = minor_divisions(s);
            for (j, v) in multiples(w0, w1, s / div as f64) {
                let major = j % div == 0;
                let pos = map.fwd(axis, v);
                if pos.is_finite() {
                    out.push(Tick {
                        pos,
                        major,
                        text: major.then(|| format_tick(v, s)),
                    });
                }
            }
            return out;
        }
        // Label every n-th decade so numbers stay at least ~45 px apart.
        let n = [1i64, 2, 3, 5, 10, 20, 50, 100]
            .into_iter()
            .find(|n| *n as f64 * ppd >= 45.0)
            .unwrap_or(100);
        let labelled_mantissas: &[i32] = if ppd >= 700.0 {
            &[2, 3, 4, 5, 6, 7, 8, 9]
        } else if ppd >= 300.0 {
            &[2, 5]
        } else {
            &[]
        };
        for e in e0 as i64..=e1 as i64 {
            let base = 10f64.powi(e as i32);
            let pos = k * e as f64;
            if pos >= lo && pos <= hi {
                let major = e.rem_euclid(n) == 0;
                out.push(Tick {
                    pos,
                    major,
                    text: major.then(|| format_tick(base, base)),
                });
            }
            if n == 1 && ppd >= 40.0 {
                for m in 2..=9 {
                    let pos = k * (e as f64 + (m as f64).log10());
                    if pos >= lo && pos <= hi {
                        let v = m as f64 * base;
                        out.push(Tick {
                            pos,
                            major: false,
                            text: labelled_mantissas
                                .contains(&m)
                                .then(|| format_tick(v, base)),
                        });
                    }
                }
            }
        }
        out
    }
    fn axes_2d(&mut self, steps: [f64; 3]) {
        let (vw, vh) = self.px();
        let (lo, hi) = (self.win.min, self.win.max);
        let axis_c = self.theme.axis;
        let (cx, cy) = (self.clamp0(0), self.clamp0(1));
        let x_visible = lo[1] <= 0.0 && 0.0 <= hi[1];
        let y_visible = lo[0] <= 0.0 && 0.0 <= hi[0];
        if x_visible {
            self.seg([lo[0], 0.0, 0.0], [hi[0], 0.0, 0.0], AXIS_W, axis_c);
            let h = TICK_PX * self.ext.line_mul as f64 * (hi[1] - lo[1]) / vh;
            for (_, v) in multiples(lo[0], hi[0], steps[0]) {
                self.seg([v, -h, 0.0], [v, h, 0.0], MINOR_W * 1.5, axis_c);
            }
        }
        if y_visible {
            self.seg([0.0, lo[1], 0.0], [0.0, hi[1], 0.0], AXIS_W, axis_c);
            let h = TICK_PX * self.ext.line_mul as f64 * (hi[0] - lo[0]) / vw;
            for (_, v) in multiples(lo[1], hi[1], steps[1]) {
                self.seg([-h, v, 0.0], [h, v, 0.0], MINOR_W * 1.5, axis_c);
            }
        }
        self.axis_labels(0, steps[0], [cx, cy, 0.0], false);
        self.axis_labels(1, steps[1], [cx, cy, 0.0], true);
        if self.ext.arrows {
            // Open arrowheads, 10 px long, at the positive end of each visible axis.
            let ap = ARROW_PX * self.ext.line_mul as f64;
            let (ax, ay) = (ap * (hi[0] - lo[0]) / vw, ap * (hi[1] - lo[1]) / vh);
            if x_visible {
                for s in [-1.0, 1.0] {
                    self.seg([hi[0], 0.0, 0.0], [hi[0] - ax, s * ay * 0.4, 0.0], AXIS_W, axis_c);
                }
            }
            if y_visible {
                for s in [-1.0, 1.0] {
                    self.seg([0.0, hi[1], 0.0], [s * ax * 0.4, hi[1] - ay, 0.0], AXIS_W, axis_c);
                }
            }
        }
        // Axis names sit just inside the positive end of their axis.
        if let Some(n) = self.ext.axis_names[0].clone() {
            self.label([hi[0] - 0.02 * (hi[0] - lo[0]), cy, 0.0], n, 0);
        }
        if let Some(n) = self.ext.axis_names[1].clone() {
            self.label([cx, hi[1] - 0.02 * (hi[1] - lo[1]), 0.0], n, 1);
        }
    }

    fn axes_1d(&mut self, step: f64) {
        let (vw, _) = self.px();
        let (lo, hi) = (self.win.min, self.win.max);
        let axis_c = self.theme.axis;
        self.seg([lo[0], 0.0, 0.0], [hi[0], 0.0, 0.0], AXIS_W, axis_c);
        // Assumes the 1D strip has the same world-per-pixel scale on y as on x.
        let h = TICK_PX * self.ext.line_mul as f64 * (hi[0] - lo[0]) / vw;
        for (_, v) in multiples(lo[0], hi[0], step) {
            self.seg([v, -h, 0.0], [v, h, 0.0], MINOR_W * 1.5, axis_c);
        }
        self.axis_labels(0, step, [0.0, 0.0, 0.0], false);
    }

    fn axes_3d(&mut self, steps: [f64; 3]) {
        let (lo, hi) = (self.win.min, self.win.max);
        let axis_c = self.theme.axis;
        let c = [self.clamp0(0), self.clamp0(1), self.clamp0(2)];
        for axis in 0..3usize {
            let mut a = c;
            let mut b = c;
            a[axis] = lo[axis];
            b[axis] = hi[axis];
            self.seg(a, b, AXIS_W, axis_c);
            self.axis_labels(axis as u8, steps[axis], c, axis != 0);
        }
    }

    /// Major tick labels along one axis, anchored at `anchor` on the other axes (none when the
    /// view hides axis numbers).
    fn axis_labels(&mut self, axis: u8, step: f64, anchor: [f64; 3], skip_zero: bool) {
        if !self.ext.axis_numbers {
            return;
        }
        let a = axis as usize;
        for (k, v) in multiples(self.win.min[a], self.win.max[a], step) {
            if skip_zero && k == 0 {
                continue;
            }
            let mut p = anchor;
            p[a] = v;
            // The polar grid numbers distances from the origin: positive on both sides.
            let v = if self.ext.polar && self.ext.mode == Mode::D2 {
                v.abs()
            } else {
                v
            };
            self.label(p, format_tick(v, step), axis);
        }
    }

    fn box_edges(&mut self) {
        let (lo, hi) = (self.win.min, self.win.max);
        let mut c = self.theme.grid_major;
        c[3] *= 0.6;
        for i in 0..8u32 {
            let corner = |bits: u32| {
                [
                    if bits & 1 == 0 { lo[0] } else { hi[0] },
                    if bits & 2 == 0 { lo[1] } else { hi[1] },
                    if bits & 4 == 0 { lo[2] } else { hi[2] },
                ]
            };
            for bit in [1u32, 2, 4] {
                if i & bit == 0 {
                    self.gseg(corner(i), corner(i | bit), MINOR_W, c);
                }
            }
        }
    }

    // ----- helpers shared by modes --------------------------------------------------------

    fn eval_const(&self, e: &Expr, defs: &Defs) -> Result<f64, String> {
        let r = defs.resolve(e).map_err(|e| e.to_string())?;
        Ok(self.prog(&r, &[])?.eval(&[]))
    }

    fn eval_tuple(&self, comps: &[Expr], defs: &Defs) -> Result<Vec<f64>, String> {
        comps.iter().map(|c| self.eval_const(c, defs)).collect()
    }

    /// Parameter range for `t` / `theta`, taking angle mode into account.
    fn turn(&self) -> f64 {
        match self.angle {
            Angle::Rad => 2.0 * PI,
            Angle::Deg => 360.0,
        }
    }

    // ----- 2D -------------------------------------------------------------------------------

    fn contour_params(&self) -> (f64, u32) {
        let (vw, vh) = self.px();
        let sx = self.win.max[0] - self.win.min[0];
        let sy = self.win.max[1] - self.win.min[1];
        let min_cell = (sx / vw).min(sy / vh) * 2.0;
        let big = sx.max(sy);
        let depth = if min_cell > 0.0 && min_cell.is_finite() {
            ((big / min_cell).log2().ceil() + 1.0).clamp(4.0, 16.0) as u32
        } else {
            10
        };
        (min_cell, depth)
    }

    fn contour(&mut self, f: &Expr, st: Style) -> Result<(), String> {
        // `f` is in world coordinates; on logarithmic axes its zero set is traced in display ones.
        let f = &self.ext.map.display_expr(f);
        let p = self.prog(f, &["x", "y"])?;
        let (min_cell, depth) = self.contour_params();
        let w = self.win;
        let segs = mesh::contour_2d(
            &p,
            (w.min[0], w.max[0]),
            (w.min[1], w.max[1]),
            min_cell,
            depth,
            400_000,
        );
        if st.dash.is_some() {
            // Dashes need the pieces joined into curves so the pattern runs along them.
            let lines = chain_segments(&segs, min_cell * 1e-3);
            self.add_lines2(&lines, st);
            return Ok(());
        }
        for s in segs {
            self.seg(
                [s[0][0], s[0][1], 0.0],
                [s[1][0], s[1][1], 0.0],
                st.line_w,
                st.color,
            );
        }
        Ok(())
    }

    /// Resolves `raw` for a GPU field. Slider names stay free and are returned as the extra
    /// inputs (sorted), so the shader text does not depend on their values; with too many
    /// sliders (or one named like a spatial/complex variable) everything is folded instead.
    fn field_resolve(&self, raw: &Expr, defs: &Defs) -> Result<(Expr, Vec<String>), String> {
        let r = self.pdefs.resolve(raw).map_err(|e| e.to_string())?;
        let used: Vec<String> = r
            .free_vars()
            .into_iter()
            .filter(|n| self.sliders.contains_key(n))
            .collect();
        let clash = used
            .iter()
            .any(|n| matches!(n.as_str(), "x" | "y" | "z" | "i"));
        if used.len() > MAX_FIELD_PARAMS || clash {
            return Ok((defs.resolve(raw).map_err(|e| e.to_string())?, Vec::new()));
        }
        Ok((r, used))
    }

    fn param_values(&self, names: &[String]) -> Vec<f32> {
        names
            .iter()
            .map(|n| self.sliders.get(n).copied().unwrap_or(0.0) as f32)
            .collect()
    }

    /// Emits a GPU field over the window's x/y rectangle on the z=0 plane. `raw` is the
    /// UNRESOLVED expression; used sliders become uniform parameters (see `FieldSpec`).
    fn field(
        &mut self,
        raw: &Expr,
        defs: &Defs,
        kind: FieldKind,
        color: [f32; 4],
    ) -> Result<(), String> {
        let (e, used) = self.field_resolve(raw, defs)?;
        if e.contains_var("z") {
            return Err("fields may only use x and y (z is not supported)".to_string());
        }
        // The shader runs at display positions: on logarithmic axes it evaluates world ones.
        let e = self.ext.map.display_expr(&e);
        if calc_draw::uses_reduce(&e) {
            // int/sum/prod have no WGSL form: raster on the CPU instead (sliders folded in).
            let folded = defs.resolve(raw).map_err(|e| e.to_string())?;
            let folded = self.ext.map.display_expr(&folded);
            return self.field_cpu(&folded, kind, color);
        }
        let mut vars = vec!["x", "y"];
        vars.extend(used.iter().map(String::as_str));
        let p = self.prog(&e, &vars)?;
        let wgsl = if used.is_empty() {
            emit_module(&[("field_fn", &p)]).map_err(|e| e.to_string())?
        } else {
            let mut m = emit_module(&[("field_core", &p)]).map_err(|e| e.to_string())?;
            let args: Vec<String> = (0..used.len()).map(param_ref).collect();
            m.push_str(&format!(
                "fn field_fn(v0: f32, v1: f32) -> f32 {{\n    return field_core(v0, v1, {});\n}}\n",
                args.join(", ")
            ));
            m
        };
        let (lo, hi) = (self.win.min, self.win.max);
        let rb = |v: f64, a: usize| (v - self.origin[a]) as f32;
        let params = self.param_values(&used);
        self.out.fields.push(FieldSpec {
            kind,
            wgsl,
            params,
            color,
            rect_min: [rb(lo[0], 0), rb(lo[1], 1)],
            rect_max: [rb(hi[0], 0), rb(hi[1], 1)],
            origin_xy: [self.origin[0], self.origin[1]],
        });
        Ok(())
    }

    /// Inequality shading: fills where `f < 0` for `<`/`<=`, `f > 0` for `>`/`>=`.
    fn inequality_field(
        &mut self,
        rel: Rel,
        f: &Expr,
        defs: &Defs,
        st: Style,
    ) -> Result<(), String> {
        let greater = matches!(rel, Rel::Gt | Rel::Ge);
        let mut color = st.color;
        if let Some(fo) = st.fill {
            // The shader multiplies by FILL_SHADER_ALPHA; the result is `opacity * fillOpacity`.
            color[3] *= fo / FILL_SHADER_ALPHA;
        }
        self.field(f, defs, FieldKind::Fill { greater }, color)
    }

    /// Domain-colours a complex expression of `z = x + iy` over the window on the z=0 plane.
    /// 2D draws it opaque; 3D slightly translucent so surfaces and axes still read.
    fn draw_complex(&mut self, e: &Expr, defs: &Defs, mode: Mode, st: Style) -> Result<(), String> {
        if mode == Mode::D1 {
            return Ok(());
        }
        // `z` and `i` belong to the complex plane: shield them from slider/definition folding.
        let guard = |e: &Expr, from: &str, to: &str| e.subst(from, &Expr::var(to));
        let shielded = guard(&guard(e, "z", "$z"), "i", "$i");
        let unshield = |e: &Expr| guard(&guard(e, "$z", "z"), "$i", "i");
        let (r, used) = {
            let (r, used) = self.field_resolve(&shielded, defs)?;
            (unshield(&r), used)
        };
        let mut vars = vec!["z"];
        vars.extend(used.iter().map(String::as_str));
        let p = compile_complex(&r, &vars).map_err(|e| {
            let m = e.to_string();
            if ["'sum'", "'int'", "'prod'"].iter().any(|n| m.contains(n)) {
                format!(
                    "{m}: sum, int and prod are not available in complex items (use a real item)"
                )
            } else {
                m
            }
        })?;
        let wgsl = if used.is_empty() {
            emit_domain_module(&p).map_err(|e| e.to_string())?
        } else {
            let mut m = String::from(COMPLEX_PRELUDE);
            m.push_str(&emit_complex_function("cplx", &p).map_err(|e| e.to_string())?);
            let args: Vec<String> = (0..used.len())
                .map(|i| format!("vec2<f32>({}, 0.0)", param_ref(i)))
                .collect();
            m.push_str(&format!(
                "fn field_color(x: f32, y: f32) -> vec4<f32> {{\n    \
                 return vec4<f32>(am_domain_color(cplx(vec2<f32>(x, y), {})), 1.0);\n}}\n",
                args.join(", ")
            ));
            m
        };
        let alpha = if mode == Mode::D3 {
            DOMAIN_ALPHA_3D
        } else {
            1.0
        };
        let (lo, hi) = (self.win.min, self.win.max);
        let rb = |v: f64, a: usize| (v - self.origin[a]) as f32;
        let params = self.param_values(&used);
        self.out.fields.push(FieldSpec {
            kind: FieldKind::Domain,
            wgsl,
            params,
            color: [1.0, 1.0, 1.0, alpha * st.color[3]],
            rect_min: [rb(lo[0], 0), rb(lo[1], 1)],
            rect_max: [rb(hi[0], 0), rb(hi[1], 1)],
            origin_xy: [self.origin[0], self.origin[1]],
        });
        Ok(())
    }

    fn add_lines2(&mut self, lines: &[Vec<[f64; 2]>], st: Style) {
        for l in lines {
            let pts: Vec<[f64; 3]> = l.iter().map(|p| [p[0], p[1], 0.0]).collect();
            self.polyline(&pts, st);
        }
    }

    /// The curve `y = rhs(x)` over the window.
    fn explicit_y_2d(&mut self, rhs: &Expr, defs: &Defs, st: Style) -> Result<(), String> {
        let w = self.win;
        let resolved = defs.resolve(rhs).map_err(|e| e.to_string())?;
        // A constant that is not a number (`y = 1/0`) has nothing to draw: say so.
        if self.prog(&resolved, &[]).is_ok_and(|c| !c.eval(&[]).is_finite()) {
            return Err("undefined".into());
        }
        // On logarithmic axes: Y = map_y(f(world x of X)).
        let m = self.ext.map;
        let resolved = if m.is_linear() {
            resolved
        } else {
            m.display_of(1, m.display_expr(&resolved))
        };
        let p = self.prog(&resolved, &["x"])?;
        let lines = mesh::sample_explicit(
            &p,
            w.min[0],
            w.max[0],
            self.vw.max(1.0) as usize,
            (w.min[1], w.max[1]),
        );
        self.add_lines2(&lines, st);
        Ok(())
    }

    fn draw_2d(&mut self, pr: &Prepared, defs: &Defs, st: Style) -> Result<(), String> {
        match &pr.kind {
            // A bare expression of x alone (`x^2`, `d/dx x^2`, `sin(x)`) is the curve y = expr.
            Kind::Field { expr } if pr.dims[0] && !pr.dims[1] && !pr.dims[2] => {
                self.explicit_y_2d(expr, defs, st)
            }
            Kind::Field { expr } => self.field(expr, defs, FieldKind::Hue, st.color),
            Kind::VectorField { components } => {
                if pr.dims[2] {
                    Ok(())
                } else {
                    self.vector_field_2d(components, defs, st, pr.item.color.is_some())
                }
            }
            _ if pr.dims[2] => Ok(()),
            Kind::ExplicitY { rhs } => self.explicit_y_2d(rhs, defs, st),
            Kind::ExplicitX { rhs } => {
                let r = defs.resolve(rhs).map_err(|e| e.to_string())?;
                self.contour(&Expr::bin(BinOp::Sub, Expr::var("x"), r), st)
            }
            Kind::Implicit { f } => {
                let r = defs.resolve(f).map_err(|e| e.to_string())?;
                self.contour(&r, st)
            }
            Kind::Inequality { rel, f } => {
                self.inequality_field(*rel, f, defs, st)?;
                let r = defs.resolve(f).map_err(|e| e.to_string())?;
                // A strict boundary (`<`, `>`) is not part of the region: draw it dashed unless
                // the item picked a line style itself.
                let mut bst = st;
                if matches!(rel, Rel::Lt | Rel::Gt) && pr.item.style.line_style.is_none() {
                    let dashed = ItemStyle {
                        line_style: Some(LineStyle::Dashed),
                        ..pr.item.style.clone()
                    };
                    bst.dash = Style::for_item(&dashed, st.color).dash;
                }
                self.contour(&r, bst)
            }
            Kind::Polar { rhs } => {
                let r = defs.resolve(rhs).map_err(|e| e.to_string())?;
                let k = if theta_outside_trig(&r) { 3.0 } else { 2.0 };
                let th = || Expr::var("theta");
                let xe = Expr::bin(BinOp::Mul, r.clone(), Expr::call("cos", vec![th()]));
                let ye = Expr::bin(BinOp::Mul, r, Expr::call("sin", vec![th()]));
                let (xe, ye) = (self.ext.map.display_of(0, xe), self.ext.map.display_of(1, ye));
                let (px, py) = (self.prog(&xe, &["theta"])?, self.prog(&ye, &["theta"])?);
                let [t0, t1] = range_values(&pr.domain, &["theta"], "theta", defs, self.angle, [0.0, self.turn() * k])?;
                let lines = mesh::sample_parametric(&px, &py, t0, t1, curve_samples(t1 - t0, self.turn()));
                self.add_lines2(&lines, st);
                Ok(())
            }
            Kind::Parametric { components } if components.len() == 2 => {
                let mut ps = Vec::new();
                for (a, c) in components.iter().enumerate() {
                    let r = defs.resolve(c).map_err(|e| e.to_string())?;
                    let r = self.ext.map.display_of(a, r);
                    ps.push(self.prog(&r, &["t"])?);
                }
                let [t0, t1] = range_values(&pr.domain, &["t"], "t", defs, self.angle, [0.0, self.turn()])?;
                let lines = mesh::sample_parametric(&ps[0], &ps[1], t0, t1, curve_samples(t1 - t0, self.turn()));
                self.add_lines2(&lines, st);
                Ok(())
            }
            Kind::Point { components } if components.len() == 2 => {
                let v = self.eval_tuple(components, defs)?;
                self.point([v[0], v[1], 0.0], st);
                Ok(())
            }
            Kind::List { items } => {
                for it in items {
                    if let Expr::Tuple(c) = it {
                        if c.len() == 2 {
                            let v = self.eval_tuple(c, defs)?;
                            self.point([v[0], v[1], 0.0], st);
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// A regression on logarithmic axes: its curve as `y = f(x)` (mapped like any explicit
    /// curve) and its residual ticks between mapped points.
    fn draw_regression_log(
        &mut self,
        pr: &Prepared,
        fit: &math_core::regress::FitResult,
        defs: &Defs,
        st: Style,
    ) -> Result<(), String> {
        let Some(curve) = fit.curve_expr("x") else {
            return Ok(()); // several data lists: no single curve to draw
        };
        self.explicit_y_2d(&curve, defs, st)?;
        if pr.item.style.residuals {
            let xs = fit.data_vars.first().and_then(|n| {
                let body = defs.resolve(&Expr::var(n)).ok()?;
                match eval_value(&body, &Bindings::new().with_angle(self.angle)).ok()? {
                    Value::List(l) => Some(l),
                    _ => None,
                }
            });
            let m = self.ext.map;
            let col = [st.color[0], st.color[1], st.color[2], st.color[3] * 0.7];
            for ((x, f), e) in xs
                .unwrap_or_default()
                .iter()
                .zip(&fit.fitted)
                .zip(&fit.residuals)
            {
                self.seg(m.fwd3([*x, *f, 0.0]), m.fwd3([*x, *f + *e, 0.0]), 1.5, col);
            }
        }
        Ok(())
    }
    // ----- 3D -------------------------------------------------------------------------------

    /// `h = rhs(u, v)` as a height field (see `math_core::mesh_height`): `axes` are the world axes
    /// of `u`, `v` and `h`. Falls back to the implicit surface `implicit = 0` when `rhs` does not
    /// compile over just `u` and `v`.
    fn explicit_surface(&mut self, rhs: &Expr, axes: [usize; 3], implicit: &Expr, st: Style) -> Result<(), String> {
        const NAMES: [&str; 3] = ["x", "y", "z"];
        let Ok(p) = self.prog(rhs, &[NAMES[axes[0]], NAMES[axes[1]]]) else {
            return self.surface(implicit, st);
        };
        let m = self.px();
        let full = ((m.0.min(m.1) / 6.0).log2().round()).clamp(5.0, 8.0) as u32;
        let depth = match SURFACE_DEPTH_CAP.get() {
            Some(cap) if cap < full => {
                SURFACE_CAPPED.set(true);
                cap
            }
            _ => full,
        };
        // A height field is not tiled like `surface` (its steep-quad refinement looks at
        // neighbouring nodes, so tile seams would not match the one-shot mesh). It costs one
        // unsliceable step, which a progressive build can be told to postpone (see
        // `build_scene_progressive_mapped`) so it does not land in the middle of a tween.
        if SURFACE_DEADLINE.get().is_some() && SURFACE_DEFER_HEAVY.get() && depth >= HEAVY_HEIGHT_DEPTH {
            SURFACE_INCOMPLETE.set(true);
            return Ok(());
        }
        let mesh = if SURFACE_DEADLINE.get().is_some() {
            // Remembered for the next slice of this progressive build (it would otherwise be
            // meshed again at every frame while a sibling surface is still unfinished).
            let key = format!("h|{rhs:?}|{axes:?}|{:?}|{:?}|{:?}|{depth}", self.win.min, self.win.max, self.angle);
            if depth >= SLICE_HEIGHT_DEPTH && !cached_mesh_exists(&key) {
                // One unsliceable step per slice: the next heavy field waits for the next frame.
                if SURFACE_HEAVY_DONE.get() {
                    SURFACE_INCOMPLETE.set(true);
                    return Ok(());
                }
                SURFACE_HEAVY_DONE.set(true);
            }
            cached_mesh(&key, || mesh::height_surface(&p, axes, self.win.min, self.win.max, depth))
        } else {
            mesh::height_surface(&p, axes, self.win.min, self.win.max, depth)
        };
        let base = self.out.vertices.len() as u32;
        let mut col = st.color;
        col[3] = (col[3] * MESH_ALPHA / 1.0).min(1.0);
        for (pos, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            let q = self.rb([pos[0] as f64, pos[1] as f64, pos[2] as f64]);
            self.out.vertices.push(MeshVertex::new(q, *n, col));
        }
        self.out.indices.extend(mesh.indices.iter().map(|i| i + base));
        Ok(())
    }

    fn surface(&mut self, f: &Expr, st: Style) -> Result<(), String> {
        let p = self.prog(f, &["x", "y", "z"])?;
        let m = self.px();
        let full = ((m.0.min(m.1) / 6.0).log2().round()).clamp(5.0, 8.0) as u32;
        let depth = match SURFACE_DEPTH_CAP.get() {
            Some(cap) if cap < full => {
                SURFACE_CAPPED.set(true);
                cap
            }
            _ => full,
        };
        let mesh = if SURFACE_DEADLINE.get().is_some() {
            let key = format!("{f:?}|{:?}|{:?}|{:?}|{depth}", self.win.min, self.win.max, self.angle);
            match tiled_surface(&key, &p, self.win.min, self.win.max, depth) {
                Some(m) => m,
                None => return Ok(()), // not finished: the caller keeps the previous scene
            }
        } else {
            mesh::surface_3d(&p, self.win.min, self.win.max, depth, 300_000)
        };
        let base = self.out.vertices.len() as u32;
        let mut col = st.color;
        col[3] = (col[3] * MESH_ALPHA / 1.0).min(1.0);
        for (pos, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            let q = self.rb([pos[0] as f64, pos[1] as f64, pos[2] as f64]);
            self.out.vertices.push(MeshVertex::new(q, *n, col));
        }
        self.out
            .indices
            .extend(mesh.indices.iter().map(|i| i + base));
        Ok(())
    }

    /// `(x(u,v), y(u,v), z(u,v))` over its `u`, `v` ranges, clipped to the window box (see
    /// `math_core::mesh_param`). `u` and `v` stay parameters even when a slider has that name.
    fn param_surface(&mut self, comps: &[Expr], domain: &[math_core::param::Range], defs: &Defs, st: Style) -> Result<(), String> {
        let defs_uv = defs.without_vars(&["u", "v"]);
        let mut ps = Vec::new();
        for c in comps {
            let r = defs_uv.resolve(c).map_err(|e| e.to_string())?;
            ps.push(self.prog(&r, &["u", "v"])?);
        }
        let dflt = [0.0, self.turn()];
        let u = range_values(domain, &["u", "v"], "u", defs, self.angle, dflt)?;
        let v = range_values(domain, &["u", "v"], "v", defs, self.angle, dflt)?;
        let budget = match SURFACE_DEPTH_CAP.get() {
            Some(_) => {
                SURFACE_CAPPED.set(true);
                30_000
            }
            None => mesh_param::DEFAULT_MAX_VERTICES,
        };
        let mesh = mesh_param::surface_param(&ps[0], &ps[1], &ps[2], u, v, self.win.min, self.win.max, budget);
        let base = self.out.vertices.len() as u32;
        let mut col = st.color;
        col[3] = (col[3] * MESH_ALPHA).min(1.0);
        for (pos, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            let q = self.rb([pos[0] as f64, pos[1] as f64, pos[2] as f64]);
            self.out.vertices.push(MeshVertex::new(q, *n, col));
        }
        self.out.indices.extend(mesh.indices.iter().map(|i| i + base));
        Ok(())
    }

    fn curve_3d(&mut self, comps: &[Expr], domain: &[math_core::param::Range], defs: &Defs, st: Style) -> Result<(), String> {
        let mut ps = Vec::new();
        for c in comps {
            let r = defs.resolve(c).map_err(|e| e.to_string())?;
            ps.push(self.prog(&r, &["t"])?);
        }
        let [t0, t1] = range_values(domain, &["t"], "t", defs, self.angle, [0.0, self.turn()])?;
        // 2048 points per default turn, more for a long range
        let n = (2048.0 * ((t1 - t0) / self.turn()).max(1.0)).min(40_000.0) as usize;
        let mut cur: Vec<[f64; 3]> = Vec::new();
        for i in 0..=n {
            let t = t0 + (t1 - t0) * i as f64 / n as f64;
            let mut p = [0.0; 3];
            for (a, pr) in ps.iter().enumerate() {
                p[a] = pr.eval(&[t]);
            }
            if p.iter().all(|v| v.is_finite()) {
                cur.push(p);
            } else {
                self.polyline(&cur, st);
                cur.clear();
            }
        }
        self.polyline(&cur, st);
        Ok(())
    }

    fn draw_3d(&mut self, pr: &Prepared, defs: &Defs, st: Style) -> Result<(), String> {
        let sub = |a: &str, e: &Expr| Expr::bin(BinOp::Sub, Expr::var(a), e.clone());
        let res = |e: &Expr| defs.resolve(e).map_err(|e| e.to_string());
        match &pr.kind {
            Kind::ExplicitY { rhs } => self.explicit_surface(&res(rhs)?, [0, 2, 1], &sub("y", &res(rhs)?), st),
            Kind::ExplicitX { rhs } => self.explicit_surface(&res(rhs)?, [1, 2, 0], &sub("x", &res(rhs)?), st),
            Kind::ExplicitZ { rhs } => self.explicit_surface(&res(rhs)?, [0, 1, 2], &sub("z", &res(rhs)?), st),
            Kind::Implicit { f } => self.surface(&res(f)?, st),
            Kind::Inequality { rel, f } => {
                let r = res(f)?;
                // The region fill lives on the z=0 plane, so it only exists for x/y-only
                // inequalities; ones that use z just draw their boundary surface.
                if !pr.dims[2] {
                    self.inequality_field(*rel, f, defs, st)?;
                }
                self.surface(&r, st)
            }
            Kind::Field { expr } => self.field(expr, defs, FieldKind::Hue, st.color),
            Kind::VectorField { components } => {
                self.vector_field_3d(components, defs, st, pr.item.color.is_some())
            }
            Kind::Parametric { components } if pr.kind.is_param_surface() => {
                self.param_surface(components, &pr.domain, defs, st)
            }
            Kind::Parametric { components } if (2..=3).contains(&components.len()) => {
                self.curve_3d(components, &pr.domain, defs, st)
            }
            Kind::Point { components } if (2..=3).contains(&components.len()) => {
                let v = self.eval_tuple(components, defs)?;
                self.point([v[0], v[1], v.get(2).copied().unwrap_or(0.0)], st);
                Ok(())
            }
            Kind::List { items } => {
                for it in items {
                    if let Expr::Tuple(c) = it {
                        if (2..=3).contains(&c.len()) {
                            let v = self.eval_tuple(c, defs)?;
                            self.point([v[0], v[1], v.get(2).copied().unwrap_or(0.0)], st);
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // ----- vector fields ---------------------------------------------------------------------

    /// Colour for a magnitude fraction in `[0, 1]`: a three-stop ramp (blue, violet, orange)
    /// tuned for the current theme's background.
    fn magnitude_color(&self, f: f64, alpha: f32) -> [f32; 4] {
        let stops: [[f32; 3]; 3] = if self.theme.dark {
            [[0.45, 0.65, 1.0], [0.75, 0.55, 0.95], [1.0, 0.60, 0.30]]
        } else {
            [[0.17, 0.38, 0.72], [0.55, 0.25, 0.65], [0.92, 0.36, 0.12]]
        };
        let f = (f.clamp(0.0, 1.0) * 2.0) as f32;
        let i = (f.floor() as usize).min(1);
        let u = f - i as f32;
        let m = |k: usize| stops[i][k] + (stops[i + 1][k] - stops[i][k]) * u;
        [m(0), m(1), m(2), alpha]
    }

    /// Compiles each component (definitions and sliders resolved) over `vars`.
    fn field_progs(
        &self,
        comps: &[Expr],
        defs: &Defs,
        vars: &[&str],
    ) -> Result<Vec<Program>, String> {
        comps
            .iter()
            .map(|c| self.prog(&defs.resolve(c).map_err(|e| e.to_string())?, vars))
            .collect()
    }

    /// Length scale for arrows: the 90th percentile of the finite magnitudes (so one pole does
    /// not shrink every other arrow to a dot).
    fn magnitude_reference(mags: &mut [f64]) -> f64 {
        mags.sort_by(|a, b| a.total_cmp(b));
        let r = mags
            .get(((mags.len() as f64) * 0.9) as usize)
            .or(mags.last())
            .copied()
            .unwrap_or(1.0);
        if r.is_finite() && r > 0.0 {
            r
        } else {
            1.0
        }
    }

    /// One flat (screen-plane) arrow centred on `c`. `u` is the unit direction in pixel space
    /// (x right, y up), `len_px` its length; `xpp`/`ypp` are world units per pixel.
    #[allow(clippy::too_many_arguments)]
    fn arrow_px(
        &mut self,
        c: [f64; 3],
        u: (f64, f64),
        len_px: f64,
        xpp: f64,
        ypp: f64,
        color: [f32; 4],
    ) {
        let at = |s: f64| [c[0] + u.0 * s * xpp, c[1] + u.1 * s * ypp, c[2]];
        let (tail, tip) = (at(-0.5 * len_px), at(0.5 * len_px));
        self.seg(tail, tip, ARROW_W, color);
        let head = (0.38 * len_px).clamp(4.0, 10.0);
        for sgn in [-1.0f64, 1.0] {
            let (s, co) = (sgn * BARB_ANGLE).sin_cos();
            // -u rotated by +-BARB_ANGLE.
            let b = (-(u.0 * co - u.1 * s), -(u.0 * s + u.1 * co));
            self.seg(
                tip,
                [tip[0] + b.0 * head * xpp, tip[1] + b.1 * head * ypp, tip[2]],
                ARROW_W,
                color,
            );
        }
    }

    fn vector_field_2d(
        &mut self,
        comps: &[Expr],
        defs: &Defs,
        st: Style,
        fixed_color: bool,
    ) -> Result<(), String> {
        let progs = self.field_progs(&comps[..comps.len().min(2)], defs, &["x", "y"])?;
        let w = self.win;
        let (vw, vh) = self.px();
        let (sx, sy) = (w.max[0] - w.min[0], w.max[1] - w.min[1]);
        let (xpp, ypp) = (sx / vw, sy / vh);
        let mut step = nice_step(sx, vw, ARROW_CELL_PX);
        let (mut xs, mut ys);
        loop {
            xs = multiples(w.min[0] - step, w.max[0] + step, step);
            ys = multiples(w.min[1] - step, w.max[1] + step, step);
            if xs.len() * ys.len() <= MAX_ARROWS_2D || step > sx * 10.0 {
                break;
            }
            step *= 2.0;
        }
        let mut stack = Vec::with_capacity(16);
        let mut samples = Vec::with_capacity(xs.len() * ys.len());
        for (_, x) in &xs {
            for (_, y) in &ys {
                let v = [
                    progs[0].eval_with(&[*x, *y], &mut stack),
                    progs
                        .get(1)
                        .map_or(0.0, |p| p.eval_with(&[*x, *y], &mut stack)),
                ];
                if v[0].is_finite() && v[1].is_finite() {
                    samples.push(([*x, *y], v, v[0].hypot(v[1])));
                }
            }
        }
        let mut mags: Vec<f64> = samples.iter().map(|s| s.2).collect();
        let reference = Self::magnitude_reference(&mut mags);
        let cell_px = step / xpp;
        for (p, v, m) in samples {
            let f = (m / reference).min(1.0);
            let color = if fixed_color {
                st.color
            } else {
                self.magnitude_color(f, st.color[3])
            };
            if m <= 0.0 {
                self.seg([p[0], p[1], 0.0], [p[0], p[1], 0.0], 3.0, color);
                continue;
            }
            // Direction as it appears on screen (x and y may be scaled differently).
            let (dx, dy) = (v[0] / xpp, v[1] / ypp);
            let n = dx.hypot(dy);
            if !(n.is_finite() && n > 0.0) {
                continue;
            }
            let len = cell_px * 0.85 * (0.25 + 0.75 * f);
            self.arrow_px([p[0], p[1], 0.0], (dx / n, dy / n), len, xpp, ypp, color);
        }
        Ok(())
    }

    /// A row of arrows along the number line: `(f(x))` points right where f > 0, left where < 0.
    fn vector_field_1d(
        &mut self,
        comp: &Expr,
        defs: &Defs,
        st: Style,
        fixed_color: bool,
    ) -> Result<(), String> {
        let progs = self.field_progs(std::slice::from_ref(comp), defs, &["x"])?;
        let w = self.win;
        let (vw, _) = self.px();
        let sx = w.max[0] - w.min[0];
        let xpp = sx / vw;
        let ypp = self.y_per_px(Mode::D1);
        let step = nice_step(sx, vw, ARROW_CELL_PX);
        let xs = multiples(w.min[0] - step, w.max[0] + step, step);
        let mut stack = Vec::with_capacity(16);
        let samples: Vec<(f64, f64)> = xs
            .iter()
            .map(|(_, x)| (*x, progs[0].eval_with(&[*x], &mut stack)))
            .filter(|(_, v)| v.is_finite())
            .collect();
        let mut mags: Vec<f64> = samples.iter().map(|s| s.1.abs()).collect();
        let reference = Self::magnitude_reference(&mut mags);
        for (x, v) in samples {
            let f = (v.abs() / reference).min(1.0);
            let color = if fixed_color {
                st.color
            } else {
                self.magnitude_color(f, st.color[3])
            };
            if v == 0.0 {
                self.seg([x, 0.0, 0.0], [x, 0.0, 0.0], 3.0, color);
                continue;
            }
            let len = step / xpp * 0.85 * (0.25 + 0.75 * f);
            self.arrow_px([x, 0.0, 0.0], (v.signum(), 0.0), len, xpp, ypp, color);
        }
        Ok(())
    }

    /// Arrows on a coarse lattice (at most `MAX_LATTICE`^3). With only two components the field
    /// lives on the z = 0 plane.
    fn vector_field_3d(
        &mut self,
        comps: &[Expr],
        defs: &Defs,
        st: Style,
        fixed_color: bool,
    ) -> Result<(), String> {
        let progs = self.field_progs(comps, defs, &["x", "y", "z"])?;
        let w = self.win;
        let span = (0..3).map(|a| w.max[a] - w.min[a]).fold(0.0, f64::max);
        let mut step = nice_step(1.0, 1.0, span / 5.0);
        let axis = |a: usize, step: f64| -> Vec<f64> {
            if a == 2 && comps.len() < 3 {
                return vec![0.0];
            }
            multiples(w.min[a], w.max[a], step)
                .into_iter()
                .map(|(_, v)| v)
                .collect()
        };
        let mut g;
        loop {
            g = [axis(0, step), axis(1, step), axis(2, step)];
            if g.iter().all(|v| v.len() <= MAX_LATTICE) || step > span * 10.0 {
                break;
            }
            step *= 2.0;
        }
        let mut stack = Vec::with_capacity(16);
        let mut samples = Vec::new();
        for x in &g[0] {
            for y in &g[1] {
                for z in &g[2] {
                    let at = [*x, *y, *z];
                    let mut v = [0.0; 3];
                    for (k, p) in progs.iter().enumerate().take(3) {
                        v[k] = p.eval_with(&at, &mut stack);
                    }
                    if v.iter().all(|c| c.is_finite()) {
                        samples.push((at, v, (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()));
                    }
                }
            }
        }
        let mut mags: Vec<f64> = samples.iter().map(|s| s.2).collect();
        let reference = Self::magnitude_reference(&mut mags);
        for (p, v, m) in samples {
            let f = (m / reference).min(1.0);
            let color = if fixed_color {
                st.color
            } else {
                self.magnitude_color(f, st.color[3])
            };
            if m <= 0.0 {
                self.seg(p, p, 4.0, color);
                continue;
            }
            let d = [v[0] / m, v[1] / m, v[2] / m];
            let len = step * 0.85 * (0.25 + 0.75 * f);
            let tail = [
                p[0] - d[0] * len / 2.0,
                p[1] - d[1] * len / 2.0,
                p[2] - d[2] * len / 2.0,
            ];
            let tip = [
                p[0] + d[0] * len / 2.0,
                p[1] + d[1] * len / 2.0,
                p[2] + d[2] * len / 2.0,
            ];
            self.seg(tail, tip, ARROW_W, color);
            // Three barbs around the shaft: perpendicular basis from the least aligned axis.
            let k = (0..3)
                .min_by(|a, b| d[*a].abs().total_cmp(&d[*b].abs()))
                .unwrap_or(0);
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let cr = |a: [f64; 3], b: [f64; 3]| {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let norm = |a: [f64; 3]| {
                let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
                [a[0] / n, a[1] / n, a[2] / n]
            };
            let a = norm(cr(d, e));
            let b = cr(d, a);
            let head = len * 0.3;
            let (s, co) = BARB_ANGLE.sin_cos();
            for i in 0..3 {
                let phi = i as f64 * 2.0 * PI / 3.0;
                let (sp, cp) = phi.sin_cos();
                let end: Vec<f64> = (0..3)
                    .map(|c| tip[c] + head * (-d[c] * co + (a[c] * cp + b[c] * sp) * s))
                    .collect();
                self.seg(tip, [end[0], end[1], end[2]], ARROW_W, color);
            }
        }
        Ok(())
    }

    // ----- tables ----------------------------------------------------------------------------

    /// Plots one y column of a table against column 0 (x): column `yc` is y (in 1D only x is
    /// used; in 3D column `yc + 1`, when it exists, is z). Rows with a blank or non-numeric cell
    /// are skipped. The column style decides points, lines and the point outline.
    #[allow(clippy::too_many_arguments)]
    fn draw_table(
        &mut self,
        cols: &[ParsedColumn],
        yc: usize,
        cs: &ColumnStyle,
        defs: &Defs,
        mode: Mode,
        st: Style,
    ) -> Result<(), String> {
        if cs.hidden || cols.is_empty() {
            return Ok(());
        }
        let rows = cols.iter().map(|c| c.cells.len()).max().unwrap_or(0);
        let mut errors: Vec<String> = Vec::new();
        let mut value = |b: &Self, ci: usize, ri: usize| -> Option<f64> {
            let e = cols.get(ci)?.cells.get(ri)?.as_ref()?;
            match defs
                .resolve(e)
                .map_err(|e| e.to_string())
                .and_then(|r| b.prog(&r, &[]))
            {
                Ok(p) => {
                    let v = p.eval(&[]);
                    v.is_finite().then_some(v)
                }
                Err(m) => {
                    if errors.len() < 3 {
                        errors.push(format!("{} row {}: {m}", cols[ci].name, ri + 1));
                    }
                    None
                }
            }
        };
        let mut run: Vec<[f64; 3]> = Vec::new();
        let mut pts: Vec<[f64; 3]> = Vec::new();
        let mut runs: Vec<Vec<[f64; 3]>> = Vec::new();
        for ri in 0..rows {
            let x = value(self, 0, ri);
            let y = if mode == Mode::D1 {
                Some(0.0)
            } else {
                value(self, yc, ri)
            };
            let z = if mode == Mode::D3 && cols.len() > yc + 1 {
                value(self, yc + 1, ri)
            } else {
                Some(0.0)
            };
            match (x, y, z) {
                (Some(x), Some(y), Some(z)) => {
                    run.push([x, y, z]);
                    pts.push([x, y, z]);
                }
                _ => runs.push(std::mem::take(&mut run)),
            }
        }
        runs.push(run);
        if cs.lines && mode != Mode::D1 {
            for r in &runs {
                self.polyline(r, st);
            }
        }
        if cs.points {
            let bg = self.theme.background;
            for p in pts {
                if mode != Mode::D1 || (p[0] >= self.win.min[0] && p[0] <= self.win.max[0]) {
                    if cs.outline {
                        let ow = 2.0 * OUTLINE_W * self.ext.line_mul;
                        self.seg_raw(p, p, self.point_px(st.point_size) + ow, bg);
                    }
                    self.point(p, st);
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    // ----- lists and statistical plots ------------------------------------------------------

    /// Draws list-valued items and statistical plots. Returns `Ok(false)` when the item is not
    /// list-ish (or still uses x/y/z) and should take the ordinary per-kind path.
    fn draw_listish(
        &mut self,
        pr: &Prepared,
        defs: &Defs,
        mode: Mode,
        st: Style,
    ) -> Result<bool, String> {
        if !matches!(
            pr.kind,
            Kind::List { .. } | Kind::Point { .. } | Kind::Value { .. }
        ) {
            return Ok(false);
        }
        let r = defs.resolve(&pr.expr).map_err(|e| e.to_string())?;
        if !is_listish(&r) || ["x", "y", "z"].iter().any(|v| r.contains_var(v)) {
            return Ok(false);
        }
        let bind = Bindings::new().with_angle(self.angle);
        if let Expr::Call(name, args) = &r {
            if STAT_PLOTS.contains(&name.as_str()) {
                self.stat_plot(name, args, &bind, mode, st)?;
                return Ok(true);
            }
        }
        let v = eval_value(&r, &bind).map_err(|e| e.to_string())?;
        self.draw_value(&v, mode, st);
        Ok(true)
    }

    /// One point (components beyond the mode's dimension are ignored where harmless, points of
    /// the wrong dimension for the mode are skipped).
    fn draw_point(&mut self, p: &[f64], mode: Mode, st: Style) {
        match (mode, p.len()) {
            (Mode::D1, n) if n >= 1 => {
                if p[0] >= self.win.min[0] && p[0] <= self.win.max[0] {
                    self.point([p[0], 0.0, 0.0], st);
                }
            }
            (Mode::D2, 2) => self.point([p[0], p[1], 0.0], st),
            (Mode::D3, 2 | 3) => self.point([p[0], p[1], p.get(2).copied().unwrap_or(0.0)], st),
            _ => {}
        }
    }

    fn draw_value(&mut self, v: &Value, mode: Mode, st: Style) {
        match v {
            Value::Num(x) => {
                if mode == Mode::D1 {
                    self.on_line(*x, st);
                }
            }
            Value::List(l) => {
                // Number lists live on the number line; in 2D/3D they have no position.
                if mode == Mode::D1 {
                    for x in l {
                        if l.len() <= MAX_LABELLED {
                            self.on_line(*x, st);
                        } else if *x >= self.win.min[0] && *x <= self.win.max[0] {
                            self.point([*x, 0.0, 0.0], st);
                        }
                    }
                }
            }
            Value::Point(p) => self.draw_point(p, mode, st),
            Value::PointList(ps) => {
                for p in ps {
                    self.draw_point(p, mode, st);
                }
            }
        }
    }

    /// `histogram(L[, binwidth])`, `boxplot(L)`, `dotplot(L[, binwidth])`. Drawn on the z = 0
    /// plane in 2D and 3D, nothing in 1D.
    fn stat_plot(
        &mut self,
        name: &str,
        args: &[Expr],
        bind: &Bindings,
        mode: Mode,
        st: Style,
    ) -> Result<(), String> {
        let max_args = if name == "boxplot" { 1 } else { 2 };
        if args.is_empty() || args.len() > max_args {
            return Err(format!(
                "{name} takes {} argument(s), got {}",
                if max_args == 1 { "1" } else { "1 or 2" },
                args.len()
            ));
        }
        let mut data = match eval_value(&args[0], bind).map_err(|e| e.to_string())? {
            Value::List(l) => l,
            Value::Num(x) => vec![x],
            _ => return Err(format!("{name} needs a list of numbers")),
        };
        data.retain(|v| v.is_finite());
        if data.is_empty() {
            return Err(format!("{name}: the list is empty"));
        }
        let width = match args.get(1) {
            Some(a) => match eval_value(a, bind).map_err(|e| e.to_string())? {
                Value::Num(w) => Some(w),
                _ => return Err(format!("{name}: the bin width must be a number")),
            },
            None => None,
        };
        if mode == Mode::D1 {
            return Ok(());
        }
        match name {
            "histogram" => {
                let (start, w, bins) = bin_values(&data, width)?;
                self.histogram(start, w, &bins, st);
            }
            "dotplot" => {
                let (_, _, bins) = bin_values(&data, width)?;
                self.dotplot(&bins, mode, st);
            }
            _ => self.boxplot(&data, mode, st).map_err(|e| e.to_string())?,
        }
        Ok(())
    }

    /// Translucent filled bars (height = count) plus a 1.5 px outline following their silhouette.
    fn histogram(&mut self, start: f64, w: f64, bins: &[Vec<f64>], st: Style) {
        let mut fill = st.color;
        fill[3] *= BAR_ALPHA;
        let mut outline: Vec<[f64; 3]> = vec![[start, 0.0, 0.0]];
        for (i, b) in bins.iter().enumerate() {
            let (x0, x1, h) = (
                start + i as f64 * w,
                start + (i + 1) as f64 * w,
                b.len() as f64,
            );
            if !b.is_empty() {
                self.quad(x0, x1, 0.0, h, fill);
            }
            outline.push([x0, h, 0.0]);
            outline.push([x1, h, 0.0]);
        }
        outline.push([start + bins.len() as f64 * w, 0.0, 0.0]);
        // Drop zero-length steps, and the baseline runs of empty bins (the axis is there).
        let ow = BAR_OUTLINE_W;
        let c = st.color;
        for pair in outline.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a != b && !(a[1] == 0.0 && b[1] == 0.0) {
                self.seg(a, b, ow, c);
            }
        }
    }

    /// Stacked dots: one column per non-empty bin at the mean of its values, one dot per value.
    fn dotplot(&mut self, bins: &[Vec<f64>], mode: Mode, st: Style) {
        let ypp = self.y_per_px(mode);
        for b in bins.iter().filter(|b| !b.is_empty()) {
            let x = b.iter().sum::<f64>() / b.len() as f64;
            for k in 0..b.len().min(MAX_STACK) {
                let d = self.point_px(DOT_W) as f64;
                let y = (d * 0.5 + 1.0 + k as f64 * (d + 0.5)) * ypp;
                self.dot([x, y, 0.0], st.color);
            }
        }
    }

    /// Tukey box plot centred on y = 0: box Q1..Q3 (R type-7 quartiles from `stats`), median
    /// line, whiskers to the most extreme points within 1.5 IQR of the box, outliers as dots.
    fn boxplot(&mut self, data: &[f64], mode: Mode, st: Style) -> Result<(), stats::StatsError> {
        let q1 = stats::quartile(data, 1.0)?;
        let med = stats::quartile(data, 2.0)?;
        let q3 = stats::quartile(data, 3.0)?;
        let iqr = q3 - q1;
        let (lo_f, hi_f) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
        let inside = data.iter().copied().filter(|v| *v >= lo_f && *v <= hi_f);
        let lo_w = inside.clone().fold(f64::INFINITY, f64::min).min(q1);
        let hi_w = inside.fold(f64::NEG_INFINITY, f64::max).max(q3);
        let hh = BOX_HALF_PX * self.y_per_px(mode);
        let mut fill = st.color;
        fill[3] *= BOX_ALPHA;
        self.quad(q1, q3, -hh, hh, fill);
        let c = st.color;
        let rect = [[q1, -hh], [q3, -hh], [q3, hh], [q1, hh], [q1, -hh]];
        for p in rect.windows(2) {
            self.seg([p[0][0], p[0][1], 0.0], [p[1][0], p[1][1], 0.0], 2.0, c);
        }
        self.seg([med, -hh, 0.0], [med, hh, 0.0], 3.0, c);
        for (from, to) in [(q1, lo_w), (q3, hi_w)] {
            if from != to {
                self.seg([from, 0.0, 0.0], [to, 0.0, 0.0], 2.0, c);
            }
            self.seg([to, -hh * 0.5, 0.0], [to, hh * 0.5, 0.0], 2.0, c);
        }
        for v in data.iter().filter(|v| **v < lo_f || **v > hi_f) {
            self.dot([*v, 0.0, 0.0], c);
        }
        Ok(())
    }

    // ----- 1D -------------------------------------------------------------------------------

    fn on_line(&mut self, x: f64, st: Style) {
        if x.is_finite() && x >= self.win.min[0] && x <= self.win.max[0] {
            self.point([x, 0.0, 0.0], st);
            self.label([x, 0.0, 0.0], format_value(x), 0);
        }
    }

    fn roots(&mut self, e: &Expr, st: Style) -> Result<(), String> {
        let p = self.prog(e, &["x"])?;
        for r in find_roots(&p, self.win.min[0], self.win.max[0]) {
            self.on_line(r, st);
        }
        Ok(())
    }

    fn draw_1d(&mut self, pr: &Prepared, defs: &Defs, st: Style) -> Result<(), String> {
        let res = |e: &Expr| defs.resolve(e).map_err(|e| e.to_string());
        match &pr.kind {
            Kind::Value { expr } => {
                let v = self.eval_const(expr, defs)?;
                self.on_line(v, st);
                Ok(())
            }
            Kind::Point { components } if components.len() == 1 => {
                let v = self.eval_tuple(components, defs)?;
                self.on_line(v[0], st);
                Ok(())
            }
            Kind::ExplicitY { rhs } => self.roots(&res(rhs)?, st),
            Kind::ExplicitX { rhs } => {
                let v = self.eval_const(rhs, defs)?;
                self.on_line(v, st);
                Ok(())
            }
            Kind::Implicit { f } if !pr.dims[1] && !pr.dims[2] => self.roots(&res(f)?, st),
            Kind::Field { expr } if !pr.dims[1] && !pr.dims[2] => self.roots(&res(expr)?, st),
            Kind::VectorField { components } if !pr.dims[1] && !pr.dims[2] => {
                self.vector_field_1d(&components[0], defs, st, pr.item.color.is_some())
            }
            _ => Ok(()),
        }
    }
}

/// Joins unordered contour pieces into polylines by matching endpoints (snapped to a grid of
/// `tol`), so a dash pattern can run along each curve. Every piece is used exactly once.
/// One grid position of the log-axes grid (display coordinate) and its number, if labelled.
struct Tick {
    pos: f64,
    major: bool,
    text: Option<String>,
}

/// Angle label of spoke `j` (multiples of 30 degrees): `pi/6`, `2pi/3`, ... or `30°`, `120°`.
fn polar_angle_text(j: u32, degrees: bool) -> String {
    if degrees {
        return format!("{}\u{b0}", j * 30);
    }
    let g = gcd(j, 6);
    let (num, den) = (j / g, 6 / g);
    let num = if num == 1 {
        String::new()
    } else {
        num.to_string()
    };
    if den == 1 {
        format!("{num}\u{3c0}")
    } else {
        format!("{num}\u{3c0}/{den}")
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}
fn chain_segments(segs: &[[[f64; 2]; 2]], tol: f64) -> Vec<Vec<[f64; 2]>> {
    use std::collections::HashMap;
    let tol = if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        1e-9
    };
    let key = |p: [f64; 2]| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64);
    let mut at: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, s) in segs.iter().enumerate() {
        for p in s {
            at.entry(key(*p)).or_default().push(i);
        }
    }
    let mut used = vec![false; segs.len()];
    // The unused piece touching `p` and its far end.
    let mut next = |p: [f64; 2], used: &mut Vec<bool>| -> Option<[f64; 2]> {
        let list = at.get_mut(&key(p))?;
        while let Some(i) = list.pop() {
            if !used[i] {
                used[i] = true;
                let s = segs[i];
                return Some(if key(s[0]) == key(p) { s[1] } else { s[0] });
            }
        }
        None
    };
    let mut out = Vec::new();
    for i in 0..segs.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut fwd = vec![segs[i][0], segs[i][1]];
        while let Some(q) = next(*fwd.last().unwrap_or(&segs[i][1]), &mut used) {
            fwd.push(q);
        }
        let mut back = Vec::new();
        let mut tail = segs[i][0];
        while let Some(q) = next(tail, &mut used) {
            back.push(q);
            tail = q;
        }
        back.reverse();
        back.extend(fwd);
        out.push(back);
    }
    out
}

/// `e` with `var` replaced by `lo + var`, so a curve over `[lo, hi]` becomes one over `[0, hi - lo]`.
fn shift_param(e: &Expr, var: &str, lo: f64) -> Expr {
    if lo == 0.0 {
        e.clone()
    } else {
        e.subst(var, &Expr::bin(BinOp::Add, Expr::num(lo), Expr::var(var)))
    }
}

/// Sample budget for a curve over a range `span` long, where `turn` is one default turn: 4000
/// for up to a turn, proportionally more (capped) for longer ranges.
fn curve_samples(span: f64, turn: f64) -> usize {
    (4000.0 * (span / turn).max(1.0)).min(120_000.0) as usize
}

/// True if `theta` appears anywhere outside a trigonometric call.
fn theta_outside_trig(e: &Expr) -> bool {
    match e {
        Expr::Var(n) => n == "theta",
        Expr::Num(_) => false,
        Expr::Neg(a) => theta_outside_trig(a),
        Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => theta_outside_trig(a) || theta_outside_trig(b),
        Expr::Call(n, args) => {
            let trig = matches!(n.as_str(), "sin" | "cos" | "tan" | "sec" | "csc" | "cot");
            !trig && args.iter().any(theta_outside_trig)
        }
        Expr::Tuple(v) | Expr::List(v) => v.iter().any(theta_outside_trig),
    }
}

// ---------------------------------------------------------------------------------------------
// Document preparation and entry point
// ---------------------------------------------------------------------------------------------

/// Why an item cannot be drawn on logarithmic 2D axes (the start of its diagnostic), if so.
fn log_unsupported(pr: &Prepared) -> Option<&'static str> {
    if pr.complex.is_some() {
        return Some("domain colouring is");
    }
    match &pr.kind {
        Kind::VectorField { .. } => Some("a vector field is"),
        _ if matches!(&pr.expr, Expr::Call(n, _) if STAT_PLOTS.contains(&n.as_str())) => {
            Some("a statistical plot is")
        }
        _ => None,
    }
}
fn is_drawable_kind(k: ItemKind) -> bool {
    !matches!(k, ItemKind::Folder | ItemKind::Note | ItemKind::Action)
}

/// `x=f(t)`, `y=g(t)` (and `z=h(t)`) as separate rows are one parametric curve: the first visible
/// row of each axis is fused into the x row (as `(f, g[, h])`, keeping the ranges of all of
/// them) and the other rows are dropped. Rows that do not share a parameter stay as they are.
fn fuse_axis_equations(items: &mut Vec<Prepared>) {
    use math_core::param::axis_equation;
    let pick = |items: &[Prepared], axis: usize| {
        items.iter().position(|p| {
            !p.item.hidden && p.complex.is_none() && axis_equation(&p.kind).is_some_and(|(a, _)| a == axis)
        })
    };
    let (Some(ix), Some(iy)) = (pick(items, 0), pick(items, 1)) else { return };
    let iz = pick(items, 2);
    let rhs = |i: usize| axis_equation(&items[i].kind).map(|(_, r)| r.clone());
    let (Some(fx), Some(fy)) = (rhs(ix), rhs(iy)) else { return };
    let mut components = vec![fx, fy];
    if let Some(fz) = iz.and_then(rhs) {
        components.push(fz);
    }
    let kind = Kind::Parametric { components };
    let Kind::Parametric { components } = &kind else { return };
    if !(kind.is_param_surface() || components.iter().any(|c| c.contains_var("t"))) {
        return;
    }
    let dims = [0, 1, 2].map(|a| components.iter().any(|c| c.contains_var(["x", "y", "z"][a])));
    let mut domain: Vec<math_core::param::Range> = Vec::new();
    let mut gone = vec![iy];
    gone.extend(iz.filter(|_| components.len() == 3));
    for i in std::iter::once(ix).chain(gone.iter().copied()) {
        for r in &items[i].domain {
            if !domain.iter().any(|d| d.var == r.var) {
                domain.push(r.clone());
            }
        }
    }
    items[ix].kind = kind;
    items[ix].dims = dims;
    items[ix].domain = domain;
    gone.sort_unstable_by(|a, b| b.cmp(a));
    for i in gone {
        items.remove(i);
    }
}

/// The value of a range bound (constants, `pi`, sliders and definitions resolved).
fn eval_bound(e: &Expr, defs: &Defs, angle: Angle) -> Result<f64, String> {
    let r = defs.resolve(e).map_err(|e| e.to_string())?;
    Ok(compile(&r, &[], angle).map_err(|e| e.to_string())?.eval(&[]))
}

/// `[lo, hi]` of the parameter `var` of an item: its `{...}` range where given, `default`
/// otherwise (see `math_core::param::resolve_range`). `allowed` are the item's parameter names;
/// a range on any other name is an error.
fn range_values(
    domain: &[math_core::param::Range],
    allowed: &[&str],
    var: &str,
    defs: &Defs,
    angle: Angle,
    default: [f64; 2],
) -> Result<[f64; 2], String> {
    if let Some(bad) = domain.iter().find(|r| !allowed.contains(&r.var.as_str())) {
        return Err(format!(
            "a range on '{}' does not apply here (the parameter is {})",
            bad.var,
            allowed.join(" and ")
        ));
    }
    let r = domain.iter().find(|r| r.var == var);
    let get = |b: Option<&Expr>| b.map(|e| eval_bound(e, defs, angle)).transpose();
    let (lo, hi) = (get(r.and_then(|r| r.lo.as_ref()))?, get(r.and_then(|r| r.hi.as_ref()))?);
    math_core::param::resolve_range(lo, hi, default)
        .map_err(|m| if domain.is_empty() { m } else { format!("{var}: {m}") })
}

/// Parses and analyzes all relevant items; parse errors become diagnostics (for visible items).
fn prepare<'a>(
    doc: &'a Doc,
    diags: &mut Vec<(String, String)>,
) -> (Vec<Prepared<'a>>, Defs, Defs, Vec<TableData<'a>>) {
    let relevant: Vec<&Item> = doc
        .items
        .iter()
        .filter(|i| {
            is_drawable_kind(i.kind)
                && !i.latex.trim().is_empty()
                && !math_core::actions::looks_like_action(&i.latex)
        })
        .collect();
    let none = BTreeSet::new();
    let mut ctx = ParseCtx::new();
    for it in &relevant {
        if let Ok(e) = parse_with(&it.latex, &ParseCtx::new()) {
            if let Kind::Definition { name, params, .. } = analyze(&e, &none).kind {
                if !params.is_empty() {
                    ctx = ctx.with_function(&name);
                }
            }
        }
    }
    let mut out = Vec::new();
    for it in relevant {
        if it.kind == ItemKind::Complex {
            match parse_complex(&it.latex, &ctx) {
                Ok(e) => out.push(Prepared {
                    item: it,
                    kind: Kind::Value {
                        expr: Expr::Num(0.0),
                    },
                    dims: [false; 3],
                    expr: e.clone(),
                    complex: Some(e),
                    domain: Vec::new(),
                }),
                Err(e) => {
                    if !it.hidden {
                        diags.push((it.id.clone(), e.to_string()));
                    }
                }
            }
            continue;
        }
        match parse_with(&it.latex, &ctx) {
            Ok(e) => {
                let a = analyze(&e, &none);
                out.push(Prepared {
                    item: it,
                    kind: a.kind,
                    dims: a.dims,
                    complex: None,
                    expr: e,
                    domain: a.domain,
                });
            }
            Err(e) => {
                if !it.hidden {
                    diags.push((it.id.clone(), e.to_string()));
                }
            }
        }
    }
    fuse_axis_equations(&mut out);
    let mut defs = Defs::from_kinds(out.iter().map(|p| &p.kind));
    // Each table column defines a list named by its header (ordinary definitions win).
    let mut tables = Vec::new();
    for it in doc.items.iter().filter(|i| i.kind == ItemKind::Table) {
        let Some(tb) = &it.table else { continue };
        let (cols, errs) = tb.parse(&ctx);
        if !it.hidden {
            for (ci, ri, msg) in errs.iter().take(3) {
                diags.push((
                    it.id.clone(),
                    format!("{} row {}: {msg}", tb.columns[*ci].name, ri + 1),
                ));
            }
        }
        let known = defs.defined_names();
        for c in &cols {
            if !known.contains(&c.name) {
                defs.define_var(&c.name, c.list());
            }
        }
        tables.push(TableData { item: it, cols });
    }
    for (name, cfg) in &doc.sliders {
        defs.set_slider(name, cfg.value);
    }
    // `deriv` scales trig derivatives by the angle unit.
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    defs.set_angle(angle);
    // Same definitions, but slider names stay free (a slider overrides a same-named definition).
    let mut pdefs = Defs::new();
    let mut seen = BTreeSet::new();
    for p in &out {
        if let Kind::Definition { name, params, body } = &p.kind {
            if !seen.insert((name.clone(), params.is_empty())) {
                continue;
            }
            if params.is_empty() {
                if !doc.sliders.contains_key(name) {
                    pdefs.define_var(name, body.clone());
                }
            } else {
                pdefs.define_func(name, params.clone(), body.clone());
            }
        }
    }
    for t in &tables {
        for c in &t.cols {
            if !seen.contains(&(c.name.clone(), true)) && !doc.sliders.contains_key(&c.name) {
                pdefs.define_var(&c.name, c.list());
            }
        }
    }
    pdefs.set_angle(angle);
    (out, defs, pdefs, tables)
}

/// [`prepare`] plus the regression fits: fitted parameters become ordinary numbers in both
/// definition sets, so every item (and a slice constant such as `z = m`) sees them, and they are
/// refitted on every build.
fn prepare_fitted<'a>(
    doc: &'a Doc,
    diags: &mut Vec<(String, String)>,
) -> (
    Vec<Prepared<'a>>,
    Defs,
    Defs,
    Vec<TableData<'a>>,
    BTreeMap<String, math_core::regress::FitResult>,
) {
    let (items, mut defs, mut pdefs, tables) = prepare(doc, diags);
    let fits = calc_draw::fit_regressions(&items, &mut defs, &mut pdefs, diags);
    (items, defs, pdefs, tables, fits)
}

/// `doc` with every item filed in a hidden folder marked hidden (not drawn, no diagnostics, but
/// its definitions stay in scope). Borrowed unchanged when no hidden folder has items.
pub fn with_folders_applied(doc: &Doc) -> std::borrow::Cow<'_, Doc> {
    let hidden: BTreeSet<&str> = doc
        .items
        .iter()
        .filter(|i| i.kind == ItemKind::Folder && i.hidden)
        .map(|i| i.id.as_str())
        .collect();
    let affected = |i: &Item| !i.hidden && i.folder.as_deref().is_some_and(|f| hidden.contains(f));
    if !doc.items.iter().any(affected) {
        return std::borrow::Cow::Borrowed(doc);
    }
    let mut d = doc.clone();
    for i in &mut d.items {
        if affected(i) {
            i.hidden = true;
        }
    }
    std::borrow::Cow::Owned(d)
}

/// Builds the drawable geometry for `doc` in `mode` (see the module docs).
///
/// `window` is the visible box in world space, `origin` the render origin subtracted (in f64)
/// from every position, `viewport_px` the canvas size in physical pixels. Never panics; bad
/// items are reported in `SceneGeometry::diagnostics` and the rest still render. Segment order:
/// grid, axes, content. With logarithmic 2D axes, `window` is in display coordinates of the
/// map [`AxisMap::for_view`] gives for the document and the viewport (see [`build_scene_mapped`]).
pub fn build_scene(
    doc: &Doc,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
) -> SceneGeometry {
    let map = doc_map(doc, mode, viewport_px);
    build_scene_mapped(doc, map, mode, window, origin, viewport_px, theme)
}

/// The axis map [`build_scene`] uses: the document's logarithmic axes in 2D, linear otherwise.
pub fn doc_map(doc: &Doc, mode: Mode, viewport_px: (u32, u32)) -> AxisMap {
    if mode == Mode::D2 {
        AxisMap::for_view(&doc.view, viewport_px.0 as f64 / viewport_px.1.max(1) as f64)
    } else {
        AxisMap::LINEAR
    }
}

/// [`build_scene`] with an explicit world -> display map of the 2D axes (`window` and every
/// position in the result are display coordinates). Ignored outside 2D.
pub fn build_scene_mapped(
    doc: &Doc,
    map: AxisMap,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
) -> SceneGeometry {
    let doc = &*with_folders_applied(doc);
    let map = if mode == Mode::D2 { map } else { AxisMap::LINEAR };
    let scale = RENDER_SCALE.get();
    let (lm, gm, pm) = weight_mul(doc.view.weight);
    let mut b = Builder {
        out: SceneGeometry::default(),
        origin,
        win: window.sanitized(),
        vw: viewport_px.0 as f64,
        vh: viewport_px.1 as f64,
        theme,
        angle: match doc.view.angle {
            AngleMode::Rad => Angle::Rad,
            AngleMode::Deg => Angle::Deg,
        },
        pdefs: Defs::new(),
        sliders: doc
            .sliders
            .iter()
            .map(|(n, c)| (n.clone(), c.value))
            .collect(),
        ext: BuildExt {
            mode,
            grid: doc.view.grid,
            axes: doc.view.axes,
            axis_numbers: doc.view.axis_numbers,
            minor_grid: doc.view.minor_grid,
            arrows: doc.view.arrows,
            axis_names: [doc.view.x_label.clone(), doc.view.y_label.clone()],
            fixed_steps: [doc.view.x_step, doc.view.y_step],
            polar: doc.view.grid_kind == math_core::doc::GridKind::Polar,
            map,
            world_in: false,
            labels: None,
            line_mul: lm * scale,
            grid_mul: gm * scale,
            point_mul: pm * scale,
            scale,
        },
    };
    b.grid_and_axes(mode);
    b.out.backdrop_segments = b.out.segments.len();
    let mut diags = Vec::new();
    let (items, defs, pdefs, tables, fits) = prepare_fitted(doc, &mut diags);
    b.out.infos = calc_draw::collect_infos(&items, &fits, &defs, &pdefs, b.angle);
    b.pdefs = pdefs;
    let mut visible_idx = 0usize;
    for pr in items.iter().filter(|p| !p.item.hidden) {
        if matches!(pr.kind, Kind::Definition { .. }) {
            continue;
        }
        // A regression is drawn in ink (black on a light theme) unless it has a colour, as in
        // Desmos; it still takes a palette slot so the rows after it keep their colours.
        let auto = if matches!(pr.kind, Kind::Regression { .. }) {
            [theme.axis[0], theme.axis[1], theme.axis[2], 1.0]
        } else {
            theme.color(visible_idx)
        };
        let mut color = pr
            .item
            .color
            .as_deref()
            .and_then(parse_hex_color)
            .unwrap_or(auto);
        visible_idx += 1;
        b.out.item_colors.push((pr.item.id.clone(), color));
        if let Some(o) = pr.item.style.opacity {
            color[3] *= o.clamp(0.0, 1.0) as f32;
        }
        let st = Style::for_item(&pr.item.style, color);
        b.begin_item_labels(&pr.item.style);
        let seg0 = b.out.segments.len();
        if !map.is_linear() {
            if let Some(why) = log_unsupported(pr) {
                diags.push((pr.item.id.clone(), format!("{why} {LOG_UNSUPPORTED}")));
                b.end_item_labels(&pr.item.style, seg0);
                continue;
            }
        }
        if !pr.domain.is_empty() && !matches!(pr.kind, Kind::Parametric { .. } | Kind::Polar { .. }) {
            diags.push((
                pr.item.id.clone(),
                "a {range} applies to parametric curves, polar curves and parametric surfaces".into(),
            ));
        }
        if matches!(pr.kind, Kind::Regression { .. }) {
            if let Some(fit) = fits.get(&pr.item.id) {
                let r = if map.is_linear() {
                    b.draw_regression(pr, fit, &defs, mode, st)
                } else {
                    b.draw_regression_log(pr, fit, &defs, st)
                };
                if let Err(msg) = r {
                    diags.push((pr.item.id.clone(), msg));
                }
            }
            b.end_item_labels(&pr.item.style, seg0);
            continue;
        }
        let r = match mode {
            _ if pr.complex.is_some() => b.draw_complex(
                pr.complex.as_ref().unwrap_or(&Expr::Num(0.0)),
                &defs,
                mode,
                st,
            ),
            _ => match b.draw_listish(pr, &defs, mode, st) {
                Ok(true) => Ok(()),
                Err(e) => Err(e),
                Ok(false) => match mode {
                    Mode::D1 => b.draw_1d(pr, &defs, st),
                    Mode::D2 => b.draw_2d(pr, &defs, st),
                    Mode::D3 => b.draw_3d(pr, &defs, st),
                },
            },
        };
        if let Err(msg) = r {
            diags.push((pr.item.id.clone(), msg));
        }
        b.end_item_labels(&pr.item.style, seg0);
    }
    for t in tables.iter().filter(|t| !t.item.hidden) {
        // Every column after the first is its own point set against the first (Desmos). In 1D
        // only x is drawn (with the first y column's style); in 3D the column after y is z.
        let ncols = t.cols.len();
        let ys: Vec<usize> = match mode {
            _ if ncols < 2 => vec![0],
            Mode::D2 => (1..ncols).collect(),
            _ => vec![1],
        };
        let item_color = t.item.color.as_deref().and_then(parse_hex_color);
        let default_cs = ColumnStyle::default();
        let mut first = None;
        let mut errs: Vec<String> = Vec::new();
        b.begin_item_labels(&t.item.style);
        // Table rows are world coordinates: mapped onto logarithmic axes as they are drawn.
        b.ext.world_in = true;
        for (k, &yc) in ys.iter().enumerate() {
            let cs = t
                .item
                .table
                .as_ref()
                .and_then(|x| x.columns.get(yc))
                .map(|c| &c.style)
                .unwrap_or(&default_cs);
            let mut color = cs
                .color
                .as_deref()
                .and_then(parse_hex_color)
                .or(if k == 0 { item_color } else { None })
                .unwrap_or_else(|| theme.color(visible_idx));
            visible_idx += 1;
            if k == 0 {
                first = Some(color);
            }
            if k > 0 {
                b.out
                    .item_colors
                    .push((format!("{}#{yc}", t.item.id), color));
            }
            if let Some(o) = cs.opacity.or(t.item.style.opacity) {
                color[3] *= o.clamp(0.0, 1.0) as f32;
            }
            let mut st = Style::for_item(&t.item.style, color);
            if let Some(ps) = cs.point_style {
                st.point = ps;
            }
            if let Some(sz) = cs.point_size {
                st.point_size = sz as f32;
            }
            if let Err(msg) = b.draw_table(&t.cols, yc, cs, &defs, mode, st) {
                if !errs.contains(&msg) {
                    errs.push(msg);
                }
            }
        }
        b.ext.world_in = false;
        b.ext.labels = None;
        if let Some(c) = first {
            b.out.item_colors.push((t.item.id.clone(), c));
        }
        if !errs.is_empty() {
            diags.push((t.item.id.clone(), errs.join("; ")));
        }
    }
    // Slice overlay (additive; a slice that does not fit the mode is simply not drawn here, the
    // app reports why).
    if let Some(cfg) = doc.slice.as_ref().filter(|_| map.is_linear()) {
        if let Ok(rs) = ResolvedSlice::resolve(cfg, mode, &defs, b.angle) {
            let sitems = slice_draw::collect(&items, &defs, theme, b.angle, &rs, &b.win);
            let px = match mode {
                Mode::D3 => (0.7 * b.vw.min(b.vh), 0.7 * b.vw.min(b.vh)),
                _ => (b.vw, b.vh),
            };
            let geo = slice_draw::compute(&rs, &sitems, &b.win, px, b.angle);
            b.slice_overlay(&rs, &sitems, &geo);
        }
    }
    b.out.diagnostics = diags;
    b.out
}

/// Result of [`build_slice_panel`]: the inset plus how many items it cuts.
pub struct SliceOutcome {
    pub panel: SlicePanel,
    pub rs: ResolvedSlice,
}

/// Octree depth of implicit surfaces in a preview build: each level less is roughly 4x fewer
/// cells, so a mode switch to 3D can start moving at once (see [`build_scene_preview`]).
pub const PREVIEW_SURFACE_DEPTH: u32 = 5;

thread_local! {
    /// Depth cap for [`Builder::surface`] while a preview is being built.
    static SURFACE_DEPTH_CAP: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    /// Set when a surface of the current preview was meshed below full quality.
    static SURFACE_CAPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// While set, surfaces are meshed tile by tile and the build stops starting new tiles at
    /// this instant (see [`build_scene_progressive_mapped`]).
    static SURFACE_DEADLINE: std::cell::Cell<Option<instant::Instant>> = const { std::cell::Cell::new(None) };
    /// Set by a progressive build that must not start unsliceable work (see
    /// [`HEAVY_HEIGHT_DEPTH`]) right now.
    static SURFACE_DEFER_HEAVY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// An unsliceable surface was meshed in the running slice of a progressive build.
    static SURFACE_HEAVY_DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set when a tiled surface ran out of time before it was complete.
    static SURFACE_INCOMPLETE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tiles meshed so far of the surfaces of a progressive build, by surface identity.
    static SURFACE_TILES: std::cell::RefCell<std::collections::HashMap<String, TileJob>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// Transient pixel scale of the build (an export at a larger size), see [`set_render_scale`].
    static RENDER_SCALE: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

/// Sets the pixel scale the following builds on this thread draw widths at (1 = the on-screen
/// look; 2 when the canvas is rendered at twice its size for an export). Not part of the
/// document. Non-finite or non-positive values mean 1.
pub fn set_render_scale(s: f32) {
    RENDER_SCALE.set(if s.is_finite() && s > 0.0 { s.min(8.0) } else { 1.0 });
}

/// A surface being meshed tile by tile (a progressive build): the lattice-aligned sub-boxes of
/// the window meshed so far and merged. Sub-boxes of the window that keep the cell size share
/// their boundary vertices exactly with their neighbours (same lattice, same bisection, same
/// gradient normals), so the merged mesh has no seams.
struct TileJob {
    next: usize,
    total: usize,
    mesh: mesh::Mesh,
}

/// Height fields at or beyond this octree depth are one unsliceable step (about 20 ms native
/// release, far more in a debug wasm build and for steep graphs).
const HEAVY_HEIGHT_DEPTH: u32 = 6;
/// From this depth on a progressive build meshes at most one height field per slice.
const SLICE_HEIGHT_DEPTH: u32 = 5;

/// Tiles are `2^split` per axis; each tile is meshed at depth `depth - split` (3 or 4), which
/// bounds the work of one step to a few milliseconds.
fn tile_split(depth: u32) -> u32 {
    depth.saturating_sub(3).min(4)
}

/// The full mesh of `f = 0` at `depth` once all its tiles are done, else `None` (and the
/// progressive build is flagged incomplete). Works through the tiles until the thread's deadline;
/// finished tiles are kept under `key`, so the next call resumes where this one stopped. Empty
/// tiles are skipped by the mesher's interval pruning almost for free.
fn tiled_surface(key: &str, p: &Program, min: [f64; 3], max: [f64; 3], depth: u32) -> Option<mesh::Mesh> {
    let split = tile_split(depth);
    let n = 1usize << split;
    let total = n * n * n;
    let deadline = SURFACE_DEADLINE.get();
    SURFACE_TILES.with(|cell| {
        let mut jobs = cell.borrow_mut();
        // A progressive build only ever needs the surfaces of the scene being refined.
        if !jobs.contains_key(key) && jobs.len() >= 32 {
            jobs.clear();
        }
        let job = jobs
            .entry(key.to_string())
            .or_insert_with(|| TileJob { next: 0, total, mesh: mesh::Mesh::default() });
        let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let per_tile_cells = if split == 0 { 300_000 } else { 40_000 };
        let started = job.next;
        while job.next < job.total {
            if job.next > started && deadline.is_some_and(|d| instant::Instant::now() >= d) {
                SURFACE_INCOMPLETE.set(true);
                return None;
            }
            let t = job.next;
            let (i, j, k) = (t % n, (t / n) % n, t / (n * n));
            let lo = |a: usize, c: usize| min[a] + size[a] * c as f64 / n as f64;
            let hi = |a: usize, c: usize| min[a] + size[a] * (c + 1) as f64 / n as f64;
            let m = mesh::surface_3d(
                p,
                [lo(0, i), lo(1, j), lo(2, k)],
                [hi(0, i), hi(1, j), hi(2, k)],
                depth - split,
                per_tile_cells,
            );
            let base = job.mesh.positions.len() as u32;
            job.mesh.positions.extend(m.positions);
            job.mesh.normals.extend(m.normals);
            job.mesh.indices.extend(m.indices.iter().map(|i| i + base));
            job.next += 1;
        }
        Some(job.mesh.clone())
    })
}

fn cached_mesh_exists(key: &str) -> bool {
    SURFACE_TILES.with(|c| c.borrow().get(key).is_some_and(|j| j.next == j.total))
}

/// The mesh remembered under `key` for the running progressive build, made by `make` the first
/// time.
fn cached_mesh(key: &str, make: impl FnOnce() -> mesh::Mesh) -> mesh::Mesh {
    if let Some(m) = SURFACE_TILES.with(|c| c.borrow().get(key).filter(|j| j.next == j.total).map(|j| j.mesh.clone())) {
        return m;
    }
    let mesh = make();
    SURFACE_TILES.with(|c| {
        let mut jobs = c.borrow_mut();
        if jobs.len() >= 32 {
            jobs.clear();
        }
        jobs.insert(key.to_string(), TileJob { next: 1, total: 1, mesh: mesh.clone() });
    });
    mesh
}

/// Forgets the tiles of finished and abandoned progressive builds.
pub fn clear_surface_tiles() {
    SURFACE_TILES.with(|c| c.borrow_mut().clear());
}

/// [`build_scene_mapped`] with implicit surfaces meshed at most `depth_cap` deep (everything
/// else identical), meshing tile by tile until `deadline` passes. Returns the geometry and
/// `done`: false when a surface was not finished (or, with `defer_heavy`, was left for later), in which case the geometry is MISSING that
/// surface and must not be shown; call again (same arguments) to continue, the finished tiles
/// are remembered. `capped` is whether `depth_cap` made any surface coarser than a full build.
#[allow(clippy::too_many_arguments)]
pub fn build_scene_progressive_mapped(
    doc: &Doc,
    map: AxisMap,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
    depth_cap: u32,
    deadline: instant::Instant,
    defer_heavy: bool,
) -> ProgressiveBuild {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            SURFACE_DEPTH_CAP.set(None);
            SURFACE_DEADLINE.set(None);
            SURFACE_DEFER_HEAVY.set(false);
        }
    }
    SURFACE_DEPTH_CAP.set(Some(depth_cap));
    SURFACE_DEADLINE.set(Some(deadline));
    SURFACE_DEFER_HEAVY.set(defer_heavy);
    SURFACE_HEAVY_DONE.set(false);
    SURFACE_CAPPED.set(false);
    SURFACE_INCOMPLETE.set(false);
    let _reset = Reset;
    let geometry = build_scene_mapped(doc, map, mode, window, origin, viewport_px, theme);
    ProgressiveBuild { geometry, done: !SURFACE_INCOMPLETE.get(), capped: SURFACE_CAPPED.get() }
}

/// Result of [`build_scene_progressive_mapped`].
pub struct ProgressiveBuild {
    pub geometry: SceneGeometry,
    pub done: bool,
    pub capped: bool,
}

/// [`build_scene_preview_mapped`] with a chosen cap: a quick first look for a mode switch
/// ([`FIRST_PREVIEW_DEPTH`]) that a progressive build then refines.
pub fn build_scene_capped_mapped(
    doc: &Doc,
    map: AxisMap,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
    depth_cap: u32,
) -> (SceneGeometry, bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            SURFACE_DEPTH_CAP.set(None);
        }
    }
    SURFACE_DEPTH_CAP.set(Some(depth_cap));
    SURFACE_CAPPED.set(false);
    let _reset = Reset;
    let g = build_scene_mapped(doc, map, mode, window, origin, viewport_px, theme);
    (g, SURFACE_CAPPED.get())
}

/// Surface depth of the first, synchronous look at a 3D scene during a mode switch: cheap
/// enough to build between the click and the first moving frame.
pub const FIRST_PREVIEW_DEPTH: u32 = 4;

/// Surface depth of a full-quality build for a viewport of `px` pixels (what `Builder::surface`
/// picks without a cap).
pub fn full_surface_depth(px: (f64, f64)) -> u32 {
    ((px.0.min(px.1) / 6.0).log2().round()).clamp(5.0, 8.0) as u32
}

/// [`build_scene`] at preview quality: implicit surfaces are meshed at most
/// [`PREVIEW_SURFACE_DEPTH`] deep (everything else is identical: grid, labels, curves, colours,
/// diagnostics). Returns the geometry and whether anything was actually coarser than a full
/// build, i.e. whether a full rebuild should follow once there is time for it.
pub fn build_scene_preview(
    doc: &Doc,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
) -> (SceneGeometry, bool) {
    let map = doc_map(doc, mode, viewport_px);
    build_scene_preview_mapped(doc, map, mode, window, origin, viewport_px, theme)
}

/// [`build_scene_preview`] with an explicit axis map (see [`build_scene_mapped`]).
pub fn build_scene_preview_mapped(
    doc: &Doc,
    map: AxisMap,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
) -> (SceneGeometry, bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            SURFACE_DEPTH_CAP.set(None);
        }
    }
    SURFACE_DEPTH_CAP.set(Some(PREVIEW_SURFACE_DEPTH));
    SURFACE_CAPPED.set(false);
    let _reset = Reset;
    let g = build_scene_mapped(doc, map, mode, window, origin, viewport_px, theme);
    (g, SURFACE_CAPPED.get())
}

/// Builds the secondary inset for the document's slice, if it has one. `Err` carries why the
/// slice cannot be shown in this mode (shown by the app, never a panic). `window` is the visible
/// window (as given to [`build_scene`]); `main_px` the canvas size.
pub fn build_slice_panel(
    doc: &Doc,
    mode: Mode,
    window: Window3,
    main_px: (u32, u32),
    theme: &Theme,
) -> Option<Result<SliceOutcome, String>> {
    build_slice_panel_view(doc, mode, window, main_px, theme, None)
}

/// [`build_slice_panel`] with an explicit inset view (see [`ViewReq`]); `None` follows the main
/// window. Regression fits are applied first, exactly as in [`build_scene`], so a slice constant
/// may use a fitted parameter and tracks refits.
pub fn build_slice_panel_view(
    doc: &Doc,
    mode: Mode,
    window: Window3,
    main_px: (u32, u32),
    theme: &Theme,
    view: Option<&ViewReq>,
) -> Option<Result<SliceOutcome, String>> {
    let cfg = doc.slice.as_ref()?;
    let doc = &*with_folders_applied(doc);
    let mut diags = Vec::new();
    let (items, defs, pdefs, _tables, _fits) = prepare_fitted(doc, &mut diags);
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let rs = match ResolvedSlice::resolve(cfg, mode, &defs, angle) {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
    };
    let win = window.sanitized();
    let sitems = slice_draw::collect(&items, &defs, theme, angle, &rs, &win);
    let ctx = slice_draw::PanelCtx {
        defs: &defs,
        pdefs: &pdefs,
        sliders: doc
            .sliders
            .iter()
            .map(|(n, c)| (n.clone(), c.value))
            .collect(),
        view,
    };
    let panel = slice_draw::build_panel(doc, &rs, &sitems, window, main_px, theme, &ctx);
    Some(Ok(SliceOutcome { panel, rs }))
}

// ---------------------------------------------------------------------------------------------
// Draggable points
// ---------------------------------------------------------------------------------------------

/// Where one coordinate of a draggable point comes from, i.e. what to edit when it is dragged.
#[derive(Clone, Debug, PartialEq)]
pub enum CoordSrc {
    /// A number written in the tuple itself: rewrite the tuple text.
    Literal,
    /// A slider variable: change the slider value.
    Slider(String),
    /// A plain numeric definition item (`a=3`): rewrite that item.
    Def { item: String, name: String },
    /// Anything else (a formula): stays where it is.
    Fixed,
}

/// A 2D point item `(a, b)` that the pointer may drag.
#[derive(Clone, Debug, PartialEq)]
pub struct PointHandle {
    pub id: String,
    /// Current world position.
    pub pos: [f64; 2],
    pub src: [CoordSrc; 2],
    /// Text of each component as written (kept verbatim when that coordinate is not edited).
    pub text: [String; 2],
}

fn literal_number(e: &Expr) -> Option<f64> {
    match e {
        Expr::Num(v) => Some(*v),
        Expr::Neg(a) => literal_number(a).map(|v| -v),
        _ => None,
    }
}

/// A visible curve `y = f(x)` of a document, compiled for hit-testing (hover, pick).
pub struct ExplicitCurve {
    pub id: String,
    /// `f` of one variable, `x`.
    pub prog: Program,
    /// `f'` and `f''` when they have a symbolic form (otherwise analysis differences `f`).
    pub d1: Option<Program>,
    pub d2: Option<Program>,
}

/// The visible `y = f(x)` items of `doc` (an equation `y=...` or a bare expression of `x`), the
/// same ones `draw_2d` draws with `explicit_y_2d`. Items that fail to compile are skipped.
pub fn explicit_curves(doc: &Doc) -> Vec<ExplicitCurve> {
    let doc = &*with_folders_applied(doc);
    let mut diags = Vec::new();
    let (items, defs, _, _) = prepare(doc, &mut diags);
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let mut out = Vec::new();
    for p in items.iter().filter(|p| !p.item.hidden && p.complex.is_none() && !p.dims[2]) {
        let rhs = match &p.kind {
            Kind::ExplicitY { rhs } => rhs,
            Kind::Field { expr } if p.dims[0] && !p.dims[1] => expr,
            _ => continue,
        };
        let Ok(resolved) = defs.resolve(rhs) else { continue };
        let Ok(prog) = compile(&resolved, &["x"], angle) else { continue };
        let nth = |n: usize| {
            math_core::calculus::nth_deriv_angle(&resolved, "x", n, angle)
                .ok()
                .and_then(|e| compile(&e, &["x"], angle).ok())
        };
        out.push(ExplicitCurve { id: p.item.id.clone(), prog, d1: nth(1), d2: nth(2) });
    }
    out
}

/// Where a draggable curve parameter lives in the document.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamSrc {
    Slider,
    /// A numeric definition item (`a=3`).
    Def { item: String },
}

/// One parameter of a [`DragCurve`].
#[derive(Debug, Clone)]
pub struct CurveParam {
    pub name: String,
    pub src: ParamSrc,
    pub cfg: math_core::drag::DragParam,
}

/// What a [`DragCurve`] evaluates. Every program takes its curve variables first and then the
/// draggable parameters, in `params` order; values are in world coordinates.
pub enum DragShape {
    /// `y = f(x)`: variables `[x, params...]`.
    Explicit(Program),
    /// `(x(t), y(t))` over `[0, t_end]` (a polar curve is this with `t = theta`): variables
    /// `[t, params...]`.
    Parametric { px: Program, py: Program, t_end: f64 },
    /// `F(x, y) = 0`: variables `[x, y, params...]`.
    Implicit(Program),
}

/// A curve `id` compiled with its draggable parameters (sliders and numeric definitions it
/// uses) kept as inputs.
pub struct DragCurve {
    pub id: String,
    pub shape: DragShape,
    pub params: Vec<CurveParam>,
}

/// The curve item `id` (visible; `y = f(x)`, `x = g(y)`, implicit `F(x,y) = 0`, polar or 2D
/// parametric) with the parameters a drag may change, or `None` when it is not such a curve or
/// has no slider or numeric definition in its equation.
pub fn drag_curve(doc: &Doc, id: &str) -> Option<DragCurve> {
    use math_core::drag::{DragParam, MAX_PARAMS};
    let doc = &*with_folders_applied(doc);
    let mut diags = Vec::new();
    let (items, _, pdefs, _) = prepare(doc, &mut diags);
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let turn = match angle {
        Angle::Rad => 2.0 * PI,
        Angle::Deg => 360.0,
    };
    let p = items.iter().find(|p| p.item.id == id && !p.item.hidden && p.complex.is_none() && !p.dims[2])?;
    // Candidates: every slider, and every numeric definition no slider overrides.
    let mut cands: BTreeMap<String, (ParamSrc, DragParam)> = BTreeMap::new();
    for q in &items {
        if let Kind::Definition { name, params, body } = &q.kind {
            if let (true, Some(v)) = (params.is_empty(), literal_number(body)) {
                if !doc.sliders.contains_key(name) {
                    cands.entry(name.clone()).or_insert((ParamSrc::Def { item: q.item.id.clone() }, DragParam::free(v)));
                }
            }
        }
    }
    for (name, c) in &doc.sliders {
        cands.insert(name.clone(), (ParamSrc::Slider, DragParam::slider(c.value, c.min, c.max, c.step)));
    }
    let names: Vec<&String> = cands.keys().collect();
    let res = |e: &Expr| pdefs.without_vars(&names).resolve(e).ok();
    let th = || Expr::var("theta");
    // The curve variables, its expressions (already resolved) and how to build the shape.
    enum K {
        Explicit,
        Implicit,
        Param(f64),
    }
    let (kind, vars, exprs): (K, Vec<&str>, Vec<Expr>) = match &p.kind {
        Kind::ExplicitY { rhs } => (K::Explicit, vec!["x"], vec![res(rhs)?]),
        Kind::Field { expr } if p.dims[0] && !p.dims[1] => (K::Explicit, vec!["x"], vec![res(expr)?]),
        Kind::ExplicitX { rhs } => {
            (K::Implicit, vec!["x", "y"], vec![Expr::bin(BinOp::Sub, Expr::var("x"), res(rhs)?)])
        }
        Kind::Implicit { f } => (K::Implicit, vec!["x", "y"], vec![res(f)?]),
        Kind::Polar { rhs } => {
            let r = res(rhs)?;
            let k = if theta_outside_trig(&r) { 3.0 } else { 2.0 };
            let xe = Expr::bin(BinOp::Mul, r.clone(), Expr::call("cos", vec![th()]));
            let ye = Expr::bin(BinOp::Mul, r, Expr::call("sin", vec![th()]));
            // a range moves the start onto 0: t -> lo + t (bounds must be plain numbers here)
            let [t0, t1] = range_values(&p.domain, &["theta"], "theta", &pdefs, angle, [0.0, turn * k]).ok()?;
            (K::Param(t1 - t0), vec!["theta"], vec![shift_param(&xe, "theta", t0), shift_param(&ye, "theta", t0)])
        }
        Kind::Parametric { components } if components.len() == 2 => {
            let [t0, t1] = range_values(&p.domain, &["t"], "t", &pdefs, angle, [0.0, turn]).ok()?;
            (
                K::Param(t1 - t0),
                vec!["t"],
                vec![shift_param(&res(&components[0])?, "t", t0), shift_param(&res(&components[1])?, "t", t0)],
            )
        }
        _ => return None,
    };
    let mut used = BTreeSet::new();
    for e in &exprs {
        used.extend(e.free_vars());
    }
    let params: Vec<CurveParam> = cands
        .into_iter()
        .filter(|(n, _)| !vars.contains(&n.as_str()) && n != "y" && used.contains(n))
        .map(|(name, (src, cfg))| CurveParam { name, src, cfg })
        .collect();
    if params.is_empty() || params.len() > MAX_PARAMS {
        return None;
    }
    let mut all: Vec<&str> = vars.clone();
    all.extend(params.iter().map(|q| q.name.as_str()));
    let shape = match kind {
        K::Explicit => DragShape::Explicit(compile(&exprs[0], &all, angle).ok()?),
        K::Implicit => DragShape::Implicit(compile(&exprs[0], &all, angle).ok()?),
        K::Param(t_end) => DragShape::Parametric {
            px: compile(&exprs[0], &all, angle).ok()?,
            py: compile(&exprs[1], &all, angle).ok()?,
            t_end,
        },
    };
    Some(DragCurve { id: id.to_string(), shape, params })
}

/// A visible implicit, polar or parametric 2D curve compiled for hit-testing, with its programs
/// in display coordinates (see [`AxisMap`]) exactly as the renderer builds them.
pub struct ShapeCurve {
    pub id: String,
    pub kind: ShapeKind,
}

pub enum ShapeKind {
    /// `F(X, Y) = 0` in display coordinates.
    Implicit(Program),
    /// `(X(t), Y(t))` in display coordinates over `[0, t_end]`.
    Parametric { px: Program, py: Program, t_end: f64 },
}

/// The visible curves of `doc` that are not `y = f(x)`: implicit (and `x = g(y)`), polar and
/// parametric ones, for picking by distance to their drawn polyline ([`shape_lines`]).
pub fn shape_curves(doc: &Doc, map: AxisMap) -> Vec<ShapeCurve> {
    let doc = &*with_folders_applied(doc);
    let mut diags = Vec::new();
    let (items, defs, _, _) = prepare(doc, &mut diags);
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let turn = match angle {
        Angle::Rad => 2.0 * PI,
        Angle::Deg => 360.0,
    };
    let mut out = Vec::new();
    for p in items.iter().filter(|p| !p.item.hidden && p.complex.is_none() && !p.dims[2]) {
        let kind = (|| -> Option<ShapeKind> {
            match &p.kind {
                Kind::ExplicitX { rhs } => {
                    let r = defs.resolve(rhs).ok()?;
                    let f = map.display_expr(&Expr::bin(BinOp::Sub, Expr::var("x"), r));
                    Some(ShapeKind::Implicit(compile(&f, &["x", "y"], angle).ok()?))
                }
                Kind::Implicit { f } => {
                    let f = map.display_expr(&defs.resolve(f).ok()?);
                    Some(ShapeKind::Implicit(compile(&f, &["x", "y"], angle).ok()?))
                }
                Kind::Polar { rhs } => {
                    let r = defs.resolve(rhs).ok()?;
                    let k = if theta_outside_trig(&r) { 3.0 } else { 2.0 };
                    let th = || Expr::var("theta");
                    let xe = Expr::bin(BinOp::Mul, r.clone(), Expr::call("cos", vec![th()]));
                    let ye = Expr::bin(BinOp::Mul, r, Expr::call("sin", vec![th()]));
                    let [t0, t1] = range_values(&p.domain, &["theta"], "theta", &defs, angle, [0.0, turn * k]).ok()?;
                    Some(ShapeKind::Parametric {
                        px: compile(&shift_param(&map.display_of(0, xe), "theta", t0), &["theta"], angle).ok()?,
                        py: compile(&shift_param(&map.display_of(1, ye), "theta", t0), &["theta"], angle).ok()?,
                        t_end: t1 - t0,
                    })
                }
                Kind::Parametric { components } if components.len() == 2 => {
                    let [t0, t1] = range_values(&p.domain, &["t"], "t", &defs, angle, [0.0, turn]).ok()?;
                    let mut ps = Vec::new();
                    for (a, c) in components.iter().enumerate() {
                        let r = shift_param(&map.display_of(a, defs.resolve(c).ok()?), "t", t0);
                        ps.push(compile(&r, &["t"], angle).ok()?);
                    }
                    let py = ps.pop()?;
                    let px = ps.pop()?;
                    Some(ShapeKind::Parametric { px, py, t_end: t1 - t0 })
                }
                _ => None,
            }
        })();
        if let Some(kind) = kind {
            out.push(ShapeCurve { id: p.item.id.clone(), kind });
        }
    }
    out
}

/// The polylines (display coordinates) the renderer draws for `c` over the display window
/// `win` on a `vp` pixel viewport: the same `mesh::sample_parametric` and `mesh::contour_2d`
/// samplers. An implicit curve comes back as one two-point polyline per contour segment.
pub fn shape_lines(c: &ShapeCurve, win: Window3, vp: (f64, f64)) -> Vec<Vec<[f64; 2]>> {
    match &c.kind {
        ShapeKind::Parametric { px, py, t_end } => mesh::sample_parametric(px, py, 0.0, *t_end, 4000),
        ShapeKind::Implicit(p) => {
            let w = win.sanitized();
            let (vw, vh) = (vp.0.max(1.0), vp.1.max(1.0));
            let (sx, sy) = (w.max[0] - w.min[0], w.max[1] - w.min[1]);
            let min_cell = (sx / vw).min(sy / vh) * 2.0;
            let big = sx.max(sy);
            let depth = if min_cell > 0.0 && min_cell.is_finite() {
                ((big / min_cell).log2().ceil() + 1.0).clamp(4.0, 16.0) as u32
            } else {
                10
            };
            mesh::contour_2d(p, (w.min[0], w.max[0]), (w.min[1], w.max[1]), min_cell, depth, 100_000)
                .into_iter()
                .map(|s| vec![s[0], s[1]])
                .collect()
        }
    }
}

/// All draggable points of `doc`: visible 2-tuples with at least one coordinate that is a plain
/// number, a slider, or a numeric definition (`a=3`). Formulas stay fixed.
pub fn point_handles(doc: &Doc) -> Vec<PointHandle> {
    let doc = &*with_folders_applied(doc);
    let mut diags = Vec::new();
    let (items, defs, _, _) = prepare(doc, &mut diags);
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let mut numeric_defs: BTreeMap<&str, &str> = BTreeMap::new();
    for p in &items {
        if let Kind::Definition { name, params, body } = &p.kind {
            if params.is_empty() && literal_number(body).is_some() {
                numeric_defs
                    .entry(name.as_str())
                    .or_insert(p.item.id.as_str());
            }
        }
    }
    let mut out = Vec::new();
    for p in items
        .iter()
        .filter(|p| !p.item.hidden && p.complex.is_none())
    {
        let Kind::Point { components } = &p.kind else {
            continue;
        };
        if components.len() != 2 {
            continue;
        }
        let mut pos = [0.0; 2];
        let mut src = [CoordSrc::Fixed, CoordSrc::Fixed];
        let mut ok = true;
        for (k, c) in components.iter().enumerate() {
            src[k] = match c {
                _ if literal_number(c).is_some() => CoordSrc::Literal,
                Expr::Var(n) if doc.sliders.contains_key(n) => CoordSrc::Slider(n.clone()),
                Expr::Var(n) => match numeric_defs.get(n.as_str()) {
                    Some(item) => CoordSrc::Def {
                        item: item.to_string(),
                        name: n.clone(),
                    },
                    None => CoordSrc::Fixed,
                },
                _ => CoordSrc::Fixed,
            };
            let v = defs
                .resolve(c)
                .ok()
                .and_then(|r| compile(&r, &[], angle).ok())
                .map(|pr| pr.eval(&[]))
                .filter(|v| v.is_finite());
            match v {
                Some(v) => pos[k] = v,
                None => ok = false,
            }
        }
        if ok && src.iter().any(|s| *s != CoordSrc::Fixed) {
            out.push(PointHandle {
                id: p.item.id.clone(),
                pos,
                src,
                text: [
                    math_core::print::to_text(&components[0]),
                    math_core::print::to_text(&components[1]),
                ],
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use math_core::table::TableStyle;
    use math_core::doc::SliderCfg;

    fn doc_with(items: &[(&str, &str)]) -> Doc {
        let mut d = Doc::new_default();
        for (id, l) in items {
            d.items.push(Item::new(id, ItemKind::Equation, l));
        }
        d
    }

    fn build(d: &Doc, mode: Mode) -> SceneGeometry {
        build_scene(
            d,
            mode,
            Window3::default(),
            [0.0; 3],
            (800, 600),
            &Theme::light(),
        )
    }

    /// Number of segments before the content layer (grid + axes), for comparison.
    fn base_len(mode: Mode) -> usize {
        build(&Doc::new_default(), mode).segments.len()
    }

    #[test]
    fn empty_doc_only_grid_and_axes() {
        for mode in [Mode::D1, Mode::D2, Mode::D3] {
            let g = build(&Doc::new_default(), mode);
            assert!(!g.segments.is_empty());
            assert!(g.vertices.is_empty() && g.indices.is_empty());
            assert!(g.diagnostics.is_empty());
            assert!(!g.labels.is_empty());
        }
    }

    #[test]
    fn parabola_2d() {
        let d = doc_with(&[("a", "y=x^2")]);
        let origin = [1.0, 2.0, 0.0];
        let g = build_scene(
            &d,
            Mode::D2,
            Window3::default(),
            origin,
            (800, 600),
            &Theme::light(),
        );
        let content = &g.segments[base_len(Mode::D2)..];
        assert!(content.len() > 20);
        let mut checked = 0;
        for s in content {
            for p in [s.p0, s.p1] {
                let (x, y) = (p[0] as f64 + origin[0], p[1] as f64 + origin[1]);
                assert!((y - x * x).abs() < 1e-3 * (1.0 + y.abs()), "({x},{y})");
                checked += 1;
            }
        }
        assert!(checked > 40);
    }

    #[test]
    fn parabola_3d_extrudes_along_z() {
        let d = doc_with(&[("a", "y=x^2")]);
        let g = build(&d, Mode::D3);
        assert!(g.vertices.len() > 100 && g.indices.len().is_multiple_of(3));
        let (mut zmin, mut zmax) = (f32::MAX, f32::MIN);
        for v in &g.vertices {
            let (x, y) = (v.pos[0], v.pos[1]);
            assert!((y - x * x).abs() < 0.05, "({x},{y})");
            zmin = zmin.min(v.pos[2]);
            zmax = zmax.max(v.pos[2]);
        }
        assert!(zmin < -9.0 && zmax > 9.0);
        assert!(g.indices.iter().all(|&i| (i as usize) < g.vertices.len()));
    }

    #[test]
    fn preview_meshes_surfaces_coarser_and_keeps_everything_else() {
        let d = doc_with(&[("s", "z=\\sin(x)\\cos(y)"), ("p", "(1,2,3)")]);
        let full = build(&d, Mode::D3);
        let (pre, coarse) = build_scene_preview(
            &d,
            Mode::D3,
            Window3::default(),
            [0.0; 3],
            (800, 600),
            &Theme::light(),
        );
        assert!(coarse);
        assert!(!pre.vertices.is_empty() && pre.vertices.len() * 4 < full.vertices.len());
        assert_eq!(
            pre.segments, full.segments,
            "grid, axes and the point are identical"
        );
        assert_eq!(pre.labels, full.labels);
        assert_eq!(pre.item_colors, full.item_colors);
        // The cap is scoped to the preview call.
        assert_eq!(build(&d, Mode::D3).vertices.len(), full.vertices.len());
        // Nothing to coarsen: not coarse, and identical to a full build.
        let (flat, coarse) = build_scene_preview(
            &d,
            Mode::D2,
            Window3::default(),
            [0.0; 3],
            (800, 600),
            &Theme::light(),
        );
        assert!(!coarse);
        assert_eq!(flat, build(&d, Mode::D2));
    }

    #[test]
    fn sphere_3d_is_closed_with_right_radius() {
        let d = doc_with(&[("s", "x^2+y^2+z^2=4")]);
        let t = std::time::Instant::now();
        let g = build(&d, Mode::D3);
        eprintln!(
            "sphere scene built in {:?}, {} verts",
            t.elapsed(),
            g.vertices.len()
        );
        assert!(g.vertices.len() > 200);
        for v in &g.vertices {
            let r = (v.pos[0].powi(2) + v.pos[1].powi(2) + v.pos[2].powi(2)).sqrt();
            assert!((r - 2.0).abs() < 0.1, "r={r}");
        }
        // Closed: every undirected edge is shared by exactly two triangles.
        let mut edges = std::collections::HashMap::new();
        for t in g.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
            }
        }
        assert!(edges.values().all(|&c| c == 2));
    }

    #[test]
    fn bad_item_diagnostic_others_render() {
        let d = doc_with(&[("bad", "y="), ("ok", "y=x")]);
        let g = build(&d, Mode::D2);
        assert_eq!(g.diagnostics.len(), 1);
        assert_eq!(g.diagnostics[0].0, "bad");
        assert!(g.segments.len() > base_len(Mode::D2));
    }

    #[test]
    fn undefined_variable_is_diagnostic_and_slider_fixes_it() {
        let mut d = doc_with(&[("a", "y=a*x")]);
        let g = build(&d, Mode::D2);
        assert_eq!(g.diagnostics.len(), 1);
        assert!(g.diagnostics[0].1.contains('a'), "{:?}", g.diagnostics);
        assert_eq!(g.segments.len(), base_len(Mode::D2));
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty());
        let content = &g.segments[base_len(Mode::D2)..];
        assert!(!content.is_empty());
        for s in content {
            assert!((s.p0[1] - 2.0 * s.p0[0]).abs() < 1e-2);
        }
    }

    #[test]
    fn user_function_and_variable_definitions() {
        let d = doc_with(&[("f", "f(x)=x^2+1"), ("g", "y=f(x)")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let content = &g.segments[base_len(Mode::D2)..];
        assert!(!content.is_empty());
        for s in content {
            assert!((s.p0[1] - (s.p0[0] * s.p0[0] + 1.0)).abs() < 1e-2);
        }
        let d = doc_with(&[("a", "a=3"), ("g", "y=a*x")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let content = &g.segments[base_len(Mode::D2)..];
        assert!(!content.is_empty());
        for s in content {
            assert!((s.p0[1] - 3.0 * s.p0[0]).abs() < 1e-2);
        }
    }

    #[test]
    fn hidden_items_skipped() {
        let mut d = doc_with(&[("a", "y=x"), ("b", "y=")]);
        d.items[0].hidden = true;
        d.items[1].hidden = true;
        let g = build(&d, Mode::D2);
        assert_eq!(g.segments.len(), base_len(Mode::D2));
        assert!(g.diagnostics.is_empty());
    }

    #[test]
    fn nice_step_cases() {
        let s = nice_step(10.0, 800.0, 100.0);
        assert!(s == 1.0 || s == 2.0, "{s}");
        let s = nice_step(1e-6, 800.0, 100.0);
        assert!(s > 0.0 && s < 1e-6 && (s * 1e8).round() >= 1.0, "{s}");
        assert!(nice_step(-5.0, 800.0, 100.0) > 0.0);
        assert!(nice_step(0.0, 800.0, 100.0).is_finite());
        assert!(nice_step(f64::NAN, 0.0, 100.0).is_finite());
        let s = nice_step(1e20, 800.0, 100.0);
        assert!((s.log10() - s.log10().round()).abs() < 1e-9 || s > 1e18);
        // mantissa is 1, 2 or 5
        for span in [0.37, 3.3, 17.0, 460.0, 1e-3] {
            let s = nice_step(span, 800.0, 100.0);
            let m = s / 10f64.powf(s.log10().floor());
            assert!(
                [1.0, 2.0, 5.0].iter().any(|c| (m - c).abs() < 1e-9),
                "{span} {s}"
            );
        }
        assert_eq!(minor_divisions(2.0), 4);
        assert_eq!(minor_divisions(0.5), 5);
    }

    #[test]
    fn label_formatting() {
        assert_eq!(format_tick(0.5, 0.5), "0.5");
        assert_eq!(format_tick(10.0, 5.0), "10");
        assert_eq!(format_tick(-3.0, 1.0), "-3");
        assert_eq!(format_tick(0.0, 1.0), "0");
        assert_eq!(format_tick(1e-6, 1e-6), "1e-6");
        assert_eq!(format_tick(2e7, 1e7), "2e7");
        assert_eq!(format_tick(0.30000000000000004, 0.1), "0.3");
        assert_eq!(format_value(5.0), "5");
        assert_eq!(format_value(0.1 + 0.2), "0.3");
        let g = build(&Doc::new_default(), Mode::D2);
        assert!(g.labels.iter().any(|l| l.axis == 0 && l.text == "4"));
        assert!(g.labels.iter().any(|l| l.axis == 1 && l.text == "-5"));
    }

    #[test]
    fn origin_rebasing_keeps_coordinates_small() {
        let d = doc_with(&[("a", "y=x")]);
        let w = Window3::new([1e6 - 5e-4, -5e-4, -5e-4], [1e6 + 5e-4, 5e-4, 5e-4]);
        let origin = w.centre();
        let g = build_scene(&d, Mode::D2, w, origin, (800, 600), &Theme::light());
        assert!(!g.segments.is_empty());
        for s in &g.segments {
            for v in s.p0.iter().chain(s.p1.iter()) {
                assert!(v.abs() < 1e-3, "{v}");
            }
        }
    }

    #[test]
    fn one_d_value_dot() {
        let mut d = Doc::new_default();
        d.items.push(Item::new("v", ItemKind::Expression, "2+3"));
        let g = build(&d, Mode::D1);
        let dot = g
            .segments
            .iter()
            .find(|s| s.p0 == s.p1 && s.width == DOT_W)
            .expect("dot");
        assert_eq!(dot.p0, [5.0, 0.0, 0.0]);
        assert!(g
            .labels
            .iter()
            .any(|l| l.text == "5" && l.pos == [5.0, 0.0, 0.0]));
    }

    #[test]
    fn one_d_roots() {
        let d = doc_with(&[("a", "y=x^2-4")]);
        let g = build(&d, Mode::D1);
        let mut xs: Vec<f32> = g
            .segments
            .iter()
            .filter(|s| s.p0 == s.p1 && s.width == DOT_W)
            .map(|s| s.p0[0])
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(xs.len(), 2);
        assert!((xs[0] + 2.0).abs() < 1e-5 && (xs[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn degree_mode_differs() {
        let mut d = doc_with(&[("a", "y=sin(x)")]);
        let rad = build(&d, Mode::D2);
        d.view.angle = AngleMode::Deg;
        let deg = build(&d, Mode::D2);
        assert_ne!(rad.segments, deg.segments);
        // sin in degrees over [-10,10] is nearly linear: |y| <= sin(10 deg)
        for s in &deg.segments[base_len(Mode::D2)..] {
            assert!(s.p0[1].abs() < 0.18);
        }
    }

    #[test]
    fn points_polar_parametric_and_timing() {
        let d = doc_with(&[
            ("p", "(1,2)"),
            ("c", "x^2+y^2=9"),
            ("pa", "(cos(t),sin(t))"),
            ("po", "r=2*sin(3*theta)"),
            ("s", "x^2+y^2+z^2=4"),
            ("w", "z=sin(x)*cos(y)"),
        ]);
        for mode in [Mode::D1, Mode::D2, Mode::D3] {
            let t = std::time::Instant::now();
            let g = build(&d, mode);
            eprintln!(
                "{mode:?}: {:?} ({} segs, {} verts)",
                t.elapsed(),
                g.segments.len(),
                g.vertices.len()
            );
            assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        }
        let g = build(&d, Mode::D2);
        assert!(g
            .segments
            .iter()
            .any(|s| s.p0 == [1.0, 2.0, 0.0] && s.p0 == s.p1));
    }

    #[test]
    fn inequality_emits_fill_field_with_flag() {
        use crate::geometry::FieldKind;
        for (src, greater) in [
            ("y<x^2", false),
            ("y<=x^2", false),
            ("y>sin(x)", true),
            ("y>=x", true),
        ] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert_eq!(g.fields.len(), 1, "{src}");
            assert_eq!(g.fields[0].kind, FieldKind::Fill { greater }, "{src}");
            assert!(g.fields[0]
                .wgsl
                .contains("fn field_fn(v0: f32, v1: f32) -> f32"));
            // boundary contour still drawn
            assert!(g.segments.len() > base_len(Mode::D2), "{src}");
        }
        // 3D: plane patch plus the boundary surface; 1D: no field.
        assert_eq!(
            build(&doc_with(&[("a", "y<x^2")]), Mode::D3).fields.len(),
            1
        );
        assert!(build(&doc_with(&[("a", "y<x^2")]), Mode::D1)
            .fields
            .is_empty());
        // uses z: boundary surface only, no plane fill and no error
        let g = build(&doc_with(&[("a", "x^2+y^2+z^2<36")]), Mode::D3);
        assert!(g.fields.is_empty() && g.diagnostics.is_empty() && !g.vertices.is_empty());
    }

    #[test]
    fn bare_expression_of_x_draws_as_a_curve_like_y_equals() {
        let base = base_len(Mode::D2);
        for src in [
            "x^2",
            "2x",
            r"\frac{d}{dx}x^2",
            r"\frac{d}{dx}\left(x^3\right)",
            r"\frac{d^2}{dx^2}\sin x",
            "d/dx x^2",
            "sin(x)",
        ] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert!(g.fields.is_empty(), "{src} must not be a hue field");
            assert!(g.segments.len() > base + 20, "{src} draws a curve");
        }
        // d/dx x^2 is the line 2x: same samples as the explicit forms
        let a = build(&doc_with(&[("a", r"\frac{d}{dx}x^2")]), Mode::D2);
        let b = build(&doc_with(&[("a", "2x")]), Mode::D2);
        let c = build(&doc_with(&[("a", "y=2x")]), Mode::D2);
        assert_eq!(a.segments.len(), b.segments.len());
        assert_eq!(b.segments.len(), c.segments.len());
        // genuinely two-variable scalars are still hue fields
        assert_eq!(
            build(
                &doc_with(&[("a", r"\frac{d}{dx}\left(x^2y\right)")]),
                Mode::D2
            )
            .fields
            .len(),
            1
        );
    }

    #[test]
    fn mathlive_chain_fills_only_between_the_bounds() {
        use crate::geometry::FieldKind;
        use math_core::analyze::{analyze, Kind};
        use math_core::compile::{compile, Angle};
        for src in [
            r"0\le y\le x^2",
            r"0\leq y\leq x^{2}",
            r"x^2\ge y\ge 0",
            r"x^{2}\geq y\geq0",
            "0<=y<=x^2",
        ] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert_eq!(g.fields.len(), 1, "{src}");
            assert_eq!(
                g.fields[0].kind,
                FieldKind::Fill { greater: false },
                "{src}"
            );
            // the shader and the boundary contour use the one margin function: both bounds
            let e = math_core::parse::parse(src).unwrap();
            let Kind::Inequality { f, .. } = analyze(&e, &BTreeSet::new()).kind else {
                panic!("{src}")
            };
            let m = compile(&f, &["x", "y"], Angle::Rad).unwrap();
            assert!(
                m.eval(&[1.0, 0.5]) < 0.0,
                "{src}: between the bounds is filled"
            );
            assert!(m.eval(&[0.5, 5.0]) > 0.0, "{src}: above x^2 is NOT filled");
            assert!(m.eval(&[3.0, -1.0]) > 0.0, "{src}: below 0 is NOT filled");
            assert!(m.eval(&[-2.0, 3.0]) < 0.0, "{src}: x<0 side too");
        }
    }

    #[test]
    fn mathlive_integral_and_sum_items_evaluate() {
        for src in [
            r"\int_{0}^{2}x^{2}\,dx",
            r"\sum_{n=0}^{6}\frac{x^{n}}{n!}",
            r"y=\int_0^x t\,dt",
            r"\prod_{k=1}^{4}k",
        ] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
        }
        let g = build(&doc_with(&[("a", r"\int_{0}^{2}x^{2}")]), Mode::D2);
        assert_eq!(g.diagnostics.len(), 1);
        assert!(
            g.diagnostics[0].1.contains("differential"),
            "{:?}",
            g.diagnostics
        );
        let g = build(&doc_with(&[("a", r"\int x\,dx")]), Mode::D2);
        assert!(
            g.diagnostics[0].1.contains("indefinite"),
            "{:?}",
            g.diagnostics
        );
    }

    #[test]
    fn letter_times_latex_function_is_a_slider_candidate_like_k() {
        for (a, k) in [
            (r"y=a\sin\left(x\right)", r"y=k\sin\left(x\right)"),
            (r"y=a\cos x", "y=k*cos(x)"),
            (r"y=a\ln x", "y=k*ln(x)"),
        ] {
            let da = build(&doc_with(&[("i", a)]), Mode::D2).diagnostics;
            let dk = build(&doc_with(&[("i", k)]), Mode::D2).diagnostics;
            assert_eq!(da.len(), 1, "{a}: {da:?}");
            assert_eq!(da[0].1, "undefined variable 'a'", "{a}");
            assert_eq!(dk[0].1, "undefined variable 'k'");
        }
        let mut d = doc_with(&[("i", r"y=a\sin\left(x\right)")]);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(g.segments.len() > base_len(Mode::D2) + 20);
    }

    #[test]
    fn field_rect_is_origin_relative() {
        let d = doc_with(&[("a", "x+y")]);
        let origin = [3.0, -2.0, 0.0];
        let g = build_scene(
            &d,
            Mode::D2,
            Window3::default(),
            origin,
            (800, 600),
            &Theme::light(),
        );
        let f = &g.fields[0];
        assert_eq!(f.origin_xy, [3.0, -2.0]);
        let w = Window3::default();
        assert!((f.rect_min[0] as f64 + 3.0 - w.min[0]).abs() < 1e-5);
        assert!((f.rect_max[1] as f64 - 2.0 - w.max[1]).abs() < 1e-5);
    }

    #[test]
    fn scalar_expression_is_hue_field_in_2d_and_3d() {
        use crate::geometry::FieldKind;
        for e in ["x+y", "sin(x)*cos(y)"] {
            for mode in [Mode::D2, Mode::D3] {
                let g = build(&doc_with(&[("a", e)]), mode);
                assert!(g.diagnostics.is_empty(), "{e}: {:?}", g.diagnostics);
                assert_eq!(g.fields.len(), 1);
                assert_eq!(g.fields[0].kind, FieldKind::Hue);
            }
        }
    }

    #[test]
    fn z_field_is_diagnostic_not_panic() {
        for mode in [Mode::D2, Mode::D3] {
            let g = build(&doc_with(&[("a", "x+z")]), mode);
            assert!(g.fields.is_empty());
            assert_eq!(g.diagnostics.len(), 1, "{mode:?}");
            assert!(g.diagnostics[0].1.contains('z'));
        }
    }

    #[test]
    fn slider_flows_into_field() {
        let mut d = doc_with(&[("a", "a*x+y")]);
        let g = build(&d, Mode::D2);
        assert!(g.fields.is_empty() && g.diagnostics.len() == 1);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        let f2 = build(&d, Mode::D2).fields[0].clone();
        d.sliders.get_mut("a").unwrap().value = 3.0;
        let f3 = build(&d, Mode::D2).fields[0].clone();
        // Slider values are uniforms: same shader text (pipeline cache key), new parameter.
        assert_eq!(f2.wgsl, f3.wgsl, "a slider drag must not change the shader");
        assert_eq!(f2.params, vec![2.0]);
        assert_eq!(f3.params, vec![3.0]);
        assert!(
            f2.wgsl.contains("fp.params[0].x")
                && f2.wgsl.contains("fn field_core(v0: f32, v1: f32, v2: f32)")
        );
    }

    #[test]
    fn slider_field_params_are_sorted_and_definitions_stay_folded() {
        // `k` is a plain definition (folded); `b` and `a` are sliders (inputs, sorted by name).
        let mut d = doc_with(&[("k", "k=4"), ("f", "y<b*x+a*k")]);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        d.sliders.insert(
            "b".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 7.0,
            },
        );
        let f = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(f.params, vec![2.0, 7.0]);
        assert!(
            f.wgsl.contains("4.0") && f.wgsl.contains("fp.params[0].y"),
            "{}",
            f.wgsl
        );
        d.sliders.get_mut("b").unwrap().value = -1.0;
        let g = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(f.wgsl, g.wgsl);
        assert_eq!(g.params, vec![2.0, -1.0]);
        // A slider overrides a same-named definition.
        let mut d = doc_with(&[("k", "k=4"), ("f", "y<k*x")]);
        d.sliders.insert(
            "k".into(),
            SliderCfg {
                min: 0.0,
                max: 9.0,
                step: None,
                value: 5.0,
            },
        );
        assert_eq!(build(&d, Mode::D2).fields[0].params, vec![5.0]);
    }

    #[test]
    fn too_many_sliders_fall_back_to_folding() {
        let names: Vec<String> = (0..17)
            .map(|i| format!("{}", (b'a' + i as u8) as char))
            .collect();
        let latex = format!(
            "y<{}",
            names
                .iter()
                .map(|n| format!("{n}*x"))
                .collect::<Vec<_>>()
                .join("+")
        );
        let mut d = doc_with(&[("f", latex.as_str())]);
        for n in &names {
            d.sliders.insert(
                n.clone(),
                SliderCfg {
                    min: 0.0,
                    max: 9.0,
                    step: None,
                    value: 2.0,
                },
            );
        }
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(g.fields[0].params.is_empty());
    }

    fn complex_doc(latex: &str) -> Doc {
        let mut d = Doc::new_default();
        d.items.push(Item::new("c", ItemKind::Complex, latex));
        d
    }

    #[test]
    fn complex_item_emits_domain_field_in_2d_and_3d_not_1d() {
        for latex in ["z^2-1", "1/z", "e^(i*z)", "conj(z)*z", "sqrt(z)", "2i+z"] {
            for (mode, alpha) in [(Mode::D2, 1.0f32), (Mode::D3, DOMAIN_ALPHA_3D)] {
                let g = build(&complex_doc(latex), mode);
                assert!(g.diagnostics.is_empty(), "{latex}: {:?}", g.diagnostics);
                assert_eq!(g.fields.len(), 1, "{latex} {mode:?}");
                let f = &g.fields[0];
                assert_eq!(f.kind, FieldKind::Domain);
                assert!(f
                    .wgsl
                    .contains("fn field_color(x: f32, y: f32) -> vec4<f32>"));
                assert!((f.color[3] - alpha).abs() < 1e-6);
            }
            let g = build(&complex_doc(latex), Mode::D1);
            assert!(g.fields.is_empty() && g.diagnostics.is_empty(), "{latex}");
        }
    }

    #[test]
    fn complex_unbound_variable_is_diagnostic_and_i_is_not_offered() {
        let g = build(&complex_doc("w+1"), Mode::D2);
        assert!(g.fields.is_empty());
        assert_eq!(g.diagnostics.len(), 1);
        assert!(
            g.diagnostics[0].1.contains("undefined variable 'w'"),
            "{:?}",
            g.diagnostics
        );
        // `i` is the imaginary unit, never an undefined variable (so no slider is offered).
        let g = build(&complex_doc("i*z+i"), Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        // A relation or a point is a clear diagnostic, not a panic.
        for bad in ["z=1", "(z,z)", "z+", "floor(z)"] {
            let g = build(&complex_doc(bad), Mode::D2);
            assert_eq!(g.diagnostics.len(), 1, "{bad}");
            assert!(g.fields.is_empty());
        }
    }

    #[test]
    fn complex_slider_flows_through_and_z_is_not_shadowed() {
        let mut d = complex_doc("a*z");
        assert_eq!(build(&d, Mode::D2).diagnostics.len(), 1);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -5.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        let f2 = build(&d, Mode::D2).fields[0].clone();
        d.sliders.get_mut("a").unwrap().value = 3.0;
        let f3 = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(
            f2.wgsl, f3.wgsl,
            "domain colouring: slider is a uniform too"
        );
        assert_eq!(
            (f2.params.clone(), f3.params.clone()),
            (vec![2.0], vec![3.0])
        );
        let w3 = f3.wgsl;
        // A slider or definition named z / i must not replace the plane variable.
        d.sliders.insert(
            "z".into(),
            SliderCfg {
                min: 0.0,
                max: 9.0,
                step: None,
                value: 7.0,
            },
        );
        d.sliders.insert(
            "i".into(),
            SliderCfg {
                min: 0.0,
                max: 9.0,
                step: None,
                value: 7.0,
            },
        );
        let w = build(&d, Mode::D2).fields[0].wgsl.clone();
        assert_eq!(w, w3);
    }

    #[test]
    fn complex_uses_definitions_and_leaves_real_items_alone() {
        let mut d = doc_with(&[("k", "k=2"), ("f", "f(x)=x^2+k"), ("r", "y=x")]);
        d.items.push(Item::new("c", ItemKind::Complex, "f(z)"));
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert_eq!(g.fields.len(), 1);
        assert_eq!(g.fields[0].kind, FieldKind::Domain);
        assert!(
            g.segments.len() > base_len(Mode::D2),
            "real item still draws"
        );
    }

    // ----- lists and statistical plots ------------------------------------------------------

    /// World-space dots (p0 == p1, dot width) among the content segments, origin added back.
    fn dots(g: &SceneGeometry, mode: Mode, origin: [f64; 3]) -> Vec<[f64; 3]> {
        g.segments[base_len(mode)..]
            .iter()
            .filter(|s| s.p0 == s.p1 && s.width == DOT_W)
            .map(|s| {
                [
                    s.p0[0] as f64 + origin[0],
                    s.p0[1] as f64 + origin[1],
                    s.p0[2] as f64 + origin[2],
                ]
            })
            .collect()
    }

    fn build_at(d: &Doc, mode: Mode, origin: [f64; 3]) -> SceneGeometry {
        build_scene(
            d,
            mode,
            Window3::default(),
            origin,
            (800, 600),
            &Theme::light(),
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4 * (1.0 + b.abs())
    }

    /// Bars as `(x0, x1, y0, y1)` (origin added back) from the flat quads.
    fn bars(g: &SceneGeometry, origin: [f64; 3]) -> Vec<[f64; 4]> {
        g.flat_indices
            .chunks(6)
            .map(|q| {
                let vs: Vec<_> = q.iter().map(|i| g.vertices[*i as usize]).collect();
                assert!(vs
                    .iter()
                    .all(|v| v.normal == [0.0; 3] && v.pos[2] == (-origin[2]) as f32));
                let xs: Vec<f64> = vs.iter().map(|v| v.pos[0] as f64 + origin[0]).collect();
                let ys: Vec<f64> = vs.iter().map(|v| v.pos[1] as f64 + origin[1]).collect();
                let f =
                    |v: &[f64], m: fn(f64, f64) -> f64, init: f64| v.iter().copied().fold(init, m);
                [
                    f(&xs, f64::min, f64::INFINITY),
                    f(&xs, f64::max, f64::NEG_INFINITY),
                    f(&ys, f64::min, f64::INFINITY),
                    f(&ys, f64::max, f64::NEG_INFINITY),
                ]
            })
            .collect()
    }

    const SAMPLE: &str = "[1,2,2,3,3,3,4,4,5,9]";

    #[test]
    fn point_lists_are_dots_in_2d_with_origin() {
        let origin = [1.5, -2.0, 0.0];
        for src in ["[(1,2),(3,4)]", "(a, b)"] {
            let mut d = doc_with(&[("p", src)]);
            if src == "(a, b)" {
                d.items.push(Item::new("a", ItemKind::Equation, "a=[1,3]"));
                d.items.push(Item::new("b", ItemKind::Equation, "b=[2,4]"));
            }
            let g = build_at(&d, Mode::D2, origin);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            let pts = dots(&g, Mode::D2, origin);
            assert_eq!(pts.len(), 2, "{src}");
            assert!(close(pts[0][0], 1.0) && close(pts[0][1], 2.0), "{pts:?}");
            assert!(close(pts[1][0], 3.0) && close(pts[1][1], 4.0), "{pts:?}");
        }
        // Comprehension.
        let d = doc_with(&[("p", "[(n, n^2) for n=[1...10]]")]);
        let pts = dots(&build(&d, Mode::D2), Mode::D2, [0.0; 3]);
        assert_eq!(pts.len(), 10);
        for (k, p) in pts.iter().enumerate() {
            let n = (k + 1) as f64;
            assert!(close(p[0], n) && close(p[1], n * n), "{p:?}");
        }
        // Colour and opacity are honoured.
        let mut d = doc_with(&[("p", "[(1,2),(3,4)]")]);
        d.items[0].color = Some("#336699".into());
        d.items[0].style.opacity = Some(0.5);
        let g = build(&d, Mode::D2);
        let s = g.segments.last().unwrap();
        assert!(
            close(s.color[2] as f64, 0x99 as f64 / 255.0) && close(s.color[3] as f64, 0.5),
            "{:?}",
            s.color
        );
    }

    #[test]
    fn point_lists_in_3d_and_1d() {
        let mut d = doc_with(&[("p", "(a, b, c)"), ("q", "(a, b)")]);
        for (n, v) in [("a", "[1,2]"), ("b", "[3,4]"), ("c", "[5,6]")] {
            d.items
                .push(Item::new(n, ItemKind::Equation, &format!("{n}={v}")));
        }
        let g = build(&d, Mode::D3);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let pts = dots(&g, Mode::D3, [0.0; 3]);
        assert_eq!(pts.len(), 4);
        assert!(pts.contains(&[1.0, 3.0, 5.0]) && pts.contains(&[2.0, 4.0, 6.0]));
        assert!(
            pts.contains(&[1.0, 3.0, 0.0]) && pts.contains(&[2.0, 4.0, 0.0]),
            "2-comp at z=0: {pts:?}"
        );
        // 2D: 3-component points have no place on the plane (no diagnostic either).
        let g = build(&doc_with(&[("p", "[(1,2,3)]")]), Mode::D2);
        assert!(g.diagnostics.is_empty() && dots(&g, Mode::D2, [0.0; 3]).is_empty());
        // 1D: dots at the x coordinate.
        let g = build(&doc_with(&[("p", "[(1,2),(3,4)]")]), Mode::D1);
        assert!(g.diagnostics.is_empty());
        let pts = dots(&g, Mode::D1, [0.0; 3]);
        assert_eq!(
            pts.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>(),
            vec![(1.0, 0.0), (3.0, 0.0)]
        );
    }

    #[test]
    fn number_lists_are_dots_in_1d_and_nothing_in_2d_3d() {
        let d = doc_with(&[("l", "[1,2.5,4]")]);
        let g = build(&d, Mode::D1);
        assert!(g.diagnostics.is_empty());
        let xs: Vec<f64> = dots(&g, Mode::D1, [0.0; 3]).iter().map(|p| p[0]).collect();
        assert_eq!(xs, vec![1.0, 2.5, 4.0]);
        for mode in [Mode::D2, Mode::D3] {
            let g = build(&d, mode);
            assert!(g.diagnostics.is_empty(), "{mode:?}");
            assert_eq!(g.segments.len(), base_len(mode), "{mode:?} draws nothing");
            assert!(g.vertices.is_empty() && g.flat_indices.is_empty());
        }
        // A range, and a scalar aggregate of a list on the 1D line.
        let g = build(
            &doc_with(&[("l", "[1...5]"), ("m", "mean([1,2,6])")]),
            Mode::D1,
        );
        assert_eq!(dots(&g, Mode::D1, [0.0; 3]).len(), 6);
    }

    #[test]
    fn slider_flows_into_list_items() {
        let mut d = doc_with(&[("l", "[n*a for n=[1...5]]")]);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: 0.0,
                max: 5.0,
                step: None,
                value: 2.0,
            },
        );
        let xs = |d: &Doc| -> Vec<f64> {
            let g = build(d, Mode::D1);
            assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
            dots(&g, Mode::D1, [0.0; 3]).iter().map(|p| p[0]).collect()
        };
        assert_eq!(xs(&d), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        d.sliders.get_mut("a").unwrap().value = 1.0;
        assert_eq!(xs(&d), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // And into a histogram bin width.
        let mut d = doc_with(&[("h", &format!("histogram({SAMPLE}, w)"))]);
        d.sliders.insert(
            "w".into(),
            SliderCfg {
                min: 0.1,
                max: 5.0,
                step: None,
                value: 1.0,
            },
        );
        let n1 = build(&d, Mode::D2).flat_indices.len();
        d.sliders.get_mut("w").unwrap().value = 10.0;
        let g = build(&d, Mode::D2);
        assert_eq!(g.flat_indices.len(), 6, "one bar for everything");
        assert_eq!(n1, 36);
    }

    #[test]
    fn histogram_bars_match_counts_and_bin_edges() {
        let origin = [2.0, 1.0, 0.0];
        let d = doc_with(&[("h", &format!("histogram({SAMPLE}, 1)"))]);
        let g = build_at(&d, Mode::D2, origin);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(
            g.indices.is_empty(),
            "bars are flat geometry, not lit surfaces"
        );
        let b = bars(&g, origin);
        let want = [
            [1.0, 2.0, 1.0],
            [2.0, 3.0, 2.0],
            [3.0, 4.0, 3.0],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 1.0],
            [9.0, 10.0, 1.0],
        ];
        assert_eq!(b.len(), want.len(), "{b:?}");
        for (bar, w) in b.iter().zip(want) {
            assert!(
                close(bar[0], w[0])
                    && close(bar[1], w[1])
                    && close(bar[2], 0.0)
                    && close(bar[3], w[2]),
                "{bar:?} vs {w:?}"
            );
        }
        // Translucent fill; outline uses the full item colour at 1.5 px.
        assert!(g
            .vertices
            .iter()
            .all(|v| close(v.color[3] as f64, BAR_ALPHA as f64)));
        let outline: Vec<_> = g.segments[base_len(Mode::D2)..]
            .iter()
            .filter(|s| s.width == BAR_OUTLINE_W)
            .collect();
        assert!(outline.len() >= 12 && outline.iter().all(|s| s.color[3] == 1.0));
        // Default bin width (no argument): every value lands in some bar.
        let d = doc_with(&[("h", &format!("histogram({SAMPLE})"))]);
        let g = build(&d, Mode::D2);
        let total: f64 = bars(&g, [0.0; 3]).iter().map(|b| b[3] - b[2]).sum();
        assert!(
            g.diagnostics.is_empty() && g.flat_indices.len() >= 12,
            "{:?}",
            g.diagnostics
        );
        assert!(
            close(total, 10.0),
            "bar heights are counts and sum to n: {total}"
        );
    }

    #[test]
    fn default_bin_width_rules() {
        let sample = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 9.0];
        // Freedman-Diaconis: IQR = 1.75, n = 10.
        assert!(close(
            default_bin_width(&sample),
            2.0 * 1.75 * 10f64.powf(-1.0 / 3.0)
        ));
        // IQR 0 -> Sturges: range 4 / (ceil(log2 8) + 1).
        assert!(close(
            default_bin_width(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]),
            1.0
        ));
        assert_eq!(default_bin_width(&[3.0, 3.0]), 1.0);
    }

    #[test]
    fn histogram_in_3d_lies_on_z0_and_1d_draws_nothing() {
        let d = doc_with(&[("h", &format!("histogram({SAMPLE}, 1)"))]);
        let g = build(&d, Mode::D3);
        assert!(g.diagnostics.is_empty());
        assert_eq!(bars(&g, [0.0; 3]).len(), 6);
        let g = build(&d, Mode::D1);
        assert!(g.diagnostics.is_empty());
        assert_eq!(g.segments.len(), base_len(Mode::D1));
        assert!(g.vertices.is_empty() && g.flat_indices.is_empty());
    }

    #[test]
    fn boxplot_matches_quartiles() {
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 9.0];
        let (q1, med, q3) = (
            stats::quartile(&data, 1.0).unwrap(),
            stats::quartile(&data, 2.0).unwrap(),
            stats::quartile(&data, 3.0).unwrap(),
        );
        let origin = [0.5, 0.0, 0.0];
        let d = doc_with(&[("b", &format!("boxplot({SAMPLE})"))]);
        let g = build_at(&d, Mode::D2, origin);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let b = bars(&g, origin);
        assert_eq!(b.len(), 1);
        assert!(
            close(b[0][0], q1) && close(b[0][1], q3),
            "{b:?} q1={q1} q3={q3}"
        );
        assert!(
            close(b[0][2], -b[0][3]) && b[0][3] > 0.0,
            "centred on y = 0"
        );
        let segs: Vec<_> = g.segments[base_len(Mode::D2)..]
            .iter()
            .filter(|s| s.p0 != s.p1)
            .collect();
        let x = |v: f32| v as f64 + origin[0];
        // Median line.
        assert!(segs
            .iter()
            .any(|s| s.width == 3.0 && close(x(s.p0[0]), med) && close(x(s.p1[0]), med)));
        // Whiskers: lower to 1, upper to 5 (9 is beyond Q3 + 1.5 IQR = 6.625).
        let horiz: Vec<_> = segs
            .iter()
            .filter(|s| s.p0[1] == 0.0 && s.p1[1] == 0.0)
            .collect();
        assert!(
            horiz
                .iter()
                .any(|s| close(x(s.p0[0]), q1) && close(x(s.p1[0]), 1.0)),
            "{horiz:?}"
        );
        assert!(
            horiz
                .iter()
                .any(|s| close(x(s.p0[0]), q3) && close(x(s.p1[0]), 5.0)),
            "{horiz:?}"
        );
        // Outlier dot at 9.
        let pts = dots(&g, Mode::D2, origin);
        assert_eq!(pts.len(), 1);
        assert!(close(pts[0][0], 9.0) && pts[0][1] == 0.0);
    }

    #[test]
    fn dotplot_stacks_dots_per_bin() {
        let d = doc_with(&[("p", &format!("dotplot({SAMPLE}, 1)"))]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let pts = dots(&g, Mode::D2, [0.0; 3]);
        assert_eq!(pts.len(), 10);
        let at = |x: f64| {
            let mut ys: Vec<f64> = pts
                .iter()
                .filter(|p| close(p[0], x))
                .map(|p| p[1])
                .collect();
            ys.sort_by(f64::total_cmp);
            ys
        };
        for (x, n) in [(1.0, 1), (2.0, 2), (3.0, 3), (4.0, 2), (5.0, 1), (9.0, 1)] {
            let ys = at(x);
            assert_eq!(ys.len(), n, "x={x}");
            assert!(
                ys[0] > 0.0 && ys.windows(2).all(|w| w[1] > w[0]),
                "stacked upward: {ys:?}"
            );
        }
        assert!(g.flat_indices.is_empty());
    }

    #[test]
    fn list_errors_are_diagnostics_naming_the_item() {
        for bad in [
            "histogram([])",
            "[1,2]+[1,2,3]",
            "histogram([1,2], 0)",
            "histogram([1,2], -1)",
            "histogram([1,2], 0.0000001)",
            "histogram([1,2,3], [1,2])",
            "boxplot([])",
            "dotplot()",
            "boxplot([1,2],3)",
            "[1...20000]",
            "histogram((1,2))",
        ] {
            for mode in [Mode::D2, Mode::D3] {
                let g = build(&doc_with(&[("it", bad)]), mode);
                assert_eq!(
                    g.diagnostics.len(),
                    1,
                    "{bad} {mode:?}: {:?}",
                    g.diagnostics
                );
                assert_eq!(g.diagnostics[0].0, "it");
                assert!(!g.diagnostics[0].1.is_empty());
                assert!(g.vertices.is_empty() && g.flat_indices.is_empty(), "{bad}");
            }
        }
        // Other items are unaffected by a bad list.
        let g = build(
            &doc_with(&[("bad", "histogram([])"), ("ok", "[(1,2)]")]),
            Mode::D2,
        );
        assert_eq!(g.diagnostics.len(), 1);
        assert_eq!(dots(&g, Mode::D2, [0.0; 3]).len(), 1);
    }

    #[test]
    fn items_with_spatial_variables_keep_the_old_paths() {
        // Not a pure list: no panic, and the usual field / curve behaviour.
        let g = build(&doc_with(&[("f", "[1,2]*x")]), Mode::D2);
        assert!(g.fields.is_empty() || g.diagnostics.is_empty());
        let g = build(&doc_with(&[("c", "y=x^2"), ("p", "(1,2)")]), Mode::D2);
        assert!(g.diagnostics.is_empty());
        assert_eq!(dots(&g, Mode::D2, [0.0; 3]).len(), 1);
    }

    #[test]
    fn hidden_list_items_draw_nothing() {
        let mut d = doc_with(&[("h", &format!("histogram({SAMPLE}, 1)"))]);
        d.items[0].hidden = true;
        let g = build(&d, Mode::D2);
        assert!(g.flat_indices.is_empty() && g.diagnostics.is_empty());
    }
    // ----- vector fields and tables -----------------------------------------------------------

    /// Content segments, minus the dot drawn at a zero vector.
    fn content(g: &SceneGeometry, mode: Mode) -> Vec<SegmentInstance> {
        g.segments[base_len(mode)..]
            .iter()
            .filter(|s| s.p0 != s.p1)
            .copied()
            .collect()
    }

    #[test]
    fn vector_field_2d_draws_arrows_with_magnitude_colour() {
        let d = doc_with(&[("f", "(-y, x)")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let c = content(&g, Mode::D2);
        // Each arrow is a shaft plus two barbs, so segment count is a multiple of 3.
        assert!(
            c.len() >= 3 * 100 && c.len().is_multiple_of(3),
            "{}",
            c.len()
        );
        // Rotation field: longer arrows far from the origin, so more than one colour appears.
        let mut colours: Vec<[u32; 3]> = c
            .iter()
            .map(|s| {
                [
                    s.color[0].to_bits(),
                    s.color[1].to_bits(),
                    s.color[2].to_bits(),
                ]
            })
            .collect();
        colours.sort();
        colours.dedup();
        assert!(colours.len() > 3, "colour should vary with magnitude");
        // Arrows stay inside (roughly) the window.
        assert!(c
            .iter()
            .all(|s| s.p0[0].abs() < 12.5 && s.p0[1].abs() < 12.5));
        // Counter-clockwise: at (5, 0) the arrow points up (+y).
        let shaft = c
            .iter()
            .step_by(3)
            .min_by(|a, b| {
                let da = (a.p0[0] + a.p1[0]) / 2.0 - 5.0;
                let db = (b.p0[0] + b.p1[0]) / 2.0 - 5.0;
                (da.abs() + (a.p0[1] + a.p1[1]).abs())
                    .total_cmp(&(db.abs() + (b.p0[1] + b.p1[1]).abs()))
            })
            .unwrap();
        assert!(
            shaft.p1[1] > shaft.p0[1] && (shaft.p1[0] - shaft.p0[0]).abs() < 1e-3,
            "{shaft:?}"
        );
    }

    #[test]
    fn vector_field_arrow_count_is_capped_and_scales_with_zoom() {
        let d = doc_with(&[("f", "(sin(y), cos(x))")]);
        let wide = build_scene(
            &d,
            Mode::D2,
            Window3::new([-1e4; 3], [1e4; 3]),
            [0.0; 3],
            (4000, 4000),
            &Theme::light(),
        );
        let n = wide.segments.len();
        assert!(n < 3 * super::MAX_ARROWS_2D + 4000, "{n}");
    }

    #[test]
    fn vector_field_uses_sliders_and_definitions() {
        let mut d = doc_with(&[("f", "(a*y, g(x))"), ("g", "g(x)=x")]);
        d.items[1].kind = ItemKind::Expression;
        let g = build(&d, Mode::D2);
        assert!(
            g.diagnostics
                .iter()
                .any(|(id, m)| id == "f" && m.contains('a')),
            "{:?}",
            g.diagnostics
        );
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: -10.0,
                max: 10.0,
                step: None,
                value: 2.0,
            },
        );
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(content(&g, Mode::D2).len() > 100);
    }

    #[test]
    fn vector_field_in_3d_is_a_capped_lattice_and_in_1d_a_row() {
        let d = doc_with(&[("f", "(y, -x, z)")]);
        let g = build(&d, Mode::D3);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let n = content(&g, Mode::D3).len();
        // Shaft + 3 barbs per arrow, at most 12^3 arrows, and clearly more than a plane's worth.
        assert!(
            n.is_multiple_of(4) && (4 * 50..=4 * 12 * 12 * 12).contains(&n),
            "{n}"
        );
        // A planar (P,Q) field in 3D lies on z = 0.
        let g = build(&doc_with(&[("f", "(y, -x)")]), Mode::D3);
        assert!(
            content(&g, Mode::D3)
                .iter()
                .step_by(4)
                .all(|s| s.p0[2] == 0.0 && s.p1[2] == 0.0),
            "shafts lie in z = 0"
        );
        // 1D: a field mentioning y or z draws nothing; a pure x field draws a row on y = 0.
        let g = build(&doc_with(&[("f", "(x, y)")]), Mode::D1);
        assert!(content(&g, Mode::D1).is_empty());
    }

    #[test]
    fn tuples_with_parameter_or_constants_are_not_fields() {
        let d = doc_with(&[("p", "(1, 2)"), ("c", "(cos(t), sin(t))")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty());
        // One dot and one curve, no arrow soup.
        let n = content(&g, Mode::D2).len();
        assert!(
            n > 1000
                && g.segments
                    .iter()
                    .filter(|s| s.p0 == s.p1 && s.width == DOT_W)
                    .count()
                    == 1,
            "{n}"
        );
    }

    fn table_doc(cols: &[(&str, &[&str])], style: TableStyle) -> Doc {
        let mut d = Doc::new_default();
        let names: Vec<String> = cols.iter().map(|c| c.0.to_string()).collect();
        let mut t = math_core::table::Table::new(&names, 0);
        for (ci, (_, cells)) in cols.iter().enumerate() {
            for (ri, v) in cells.iter().enumerate() {
                t.set_cell(ri, ci, v).unwrap();
            }
        }
        t.apply_table_style(style);
        let mut it = Item::new("t", ItemKind::Table, "");
        it.table = Some(t);
        d.items.push(it);
        d
    }

    #[test]
    fn table_plots_first_two_columns_as_points_and_lines() {
        let cols: [(&str, &[&str]); 2] = [("x_1", &["1", "2", "3"]), ("y_1", &["2", "4", "a"])];
        let mut d = table_doc(&cols, TableStyle::Points);
        d.sliders.insert(
            "a".into(),
            SliderCfg {
                min: 0.0,
                max: 10.0,
                step: None,
                value: 9.0,
            },
        );
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let dp = dots(&g, Mode::D2, [0.0; 3]);
        assert_eq!(dp.len(), 3);
        assert!(close(dp[2][0], 3.0) && close(dp[2][1], 9.0), "{dp:?}");
        d.items[0].table.as_mut().unwrap().apply_table_style(TableStyle::Line);
        let g = build(&d, Mode::D2);
        let n_line = g.segments.len() - base_len(Mode::D2);
        assert_eq!(n_line, 3 + 2, "three dots and two joining segments");
        d.items[0].table.as_mut().unwrap().apply_table_style(TableStyle::Hidden);
        assert!(content(&build(&d, Mode::D2), Mode::D2).is_empty());
    }

    #[test]
    fn table_blank_and_bad_cells_skip_rows_and_report() {
        let cols: [(&str, &[&str]); 2] = [
            ("x_1", &["1", "", "3", "4+"]),
            ("y_1", &["1", "2", "3", "4"]),
        ];
        let g = build(&table_doc(&cols, TableStyle::Points), Mode::D2);
        assert_eq!(dots(&g, Mode::D2, [0.0; 3]).len(), 2);
        assert!(
            g.diagnostics
                .iter()
                .any(|(id, m)| id == "t" && m.contains("x_1") && m.contains("row 4")),
            "{:?}",
            g.diagnostics
        );
    }

    #[test]
    fn table_columns_are_lists_for_other_items() {
        let cols: [(&str, &[&str]); 2] = [
            ("x_1", &["1", "2", "3", "4"]),
            ("y_1", &["1", "2", "3", "4"]),
        ];
        let mut d = table_doc(&cols, TableStyle::Hidden);
        d.items
            .push(Item::new("h", ItemKind::Equation, "histogram(x_1)"));
        d.items
            .push(Item::new("m", ItemKind::Equation, "mean(y_1)"));
        let g = build(&d, Mode::D1);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        // mean(y_1) = 2.5 is a value: a dot on the number line (hidden table draws nothing).
        let dp = dots(&g, Mode::D1, [0.0; 3]);
        assert_eq!(dp.len(), 1);
        assert!(close(dp[0][0], 2.5));
        let g2 = build(&d, Mode::D2);
        assert!(g2.diagnostics.is_empty(), "{:?}", g2.diagnostics);
        assert!(
            !g2.vertices.is_empty(),
            "histogram of a table column draws bars"
        );
        // A hidden table still defines its lists; deleting the table makes the name unbound.
        d.items.remove(0);
        assert!(!build(&d, Mode::D2).diagnostics.is_empty());
    }

    #[test]
    fn table_in_3d_uses_a_third_column_as_z() {
        let cols: [(&str, &[&str]); 3] = [("x_1", &["1"]), ("y_1", &["2"]), ("w_1", &["3"])];
        let g = build(&table_doc(&cols, TableStyle::Points), Mode::D3);
        let dp = dots(&g, Mode::D3, [0.0; 3]);
        assert_eq!(dp.len(), 1);
        assert!(close(dp[0][2], 3.0), "{dp:?}");
    }

    // ----- item styles and view flags -------------------------------------------------------

    fn styled(src: &str, f: impl Fn(&mut math_core::doc::ItemStyle)) -> Doc {
        let mut d = doc_with(&[("a", src)]);
        f(&mut d.items[0].style);
        d
    }

    /// Item segments (after the grid and axes) and their total length in world units.
    fn item_segs(g: &SceneGeometry, mode: Mode) -> (Vec<SegmentInstance>, f64) {
        let v: Vec<SegmentInstance> = g.segments[base_len(mode)..].to_vec();
        let len = v
            .iter()
            .map(|s| {
                (0..3)
                    .map(|k| ((s.p1[k] - s.p0[k]) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum();
        (v, len)
    }

    fn weighted(mut d: Doc, w: Weight) -> Doc {
        d.view.weight = w;
        d
    }

    fn widths(g: &SceneGeometry) -> Vec<f32> {
        g.segments.iter().map(|s| s.width).collect()
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn print_weight_scales_lines_at_draw_time_and_normal_is_unchanged() {
        let d = doc_with(&[("a", "y=x")]);
        let normal = build(&d, Mode::D2);
        assert_eq!(widths(&normal), widths(&build(&weighted(d.clone(), Weight::Normal), Mode::D2)));
        // (weight, curve/axis/tick multiplier, grid multiplier): the grid scales less.
        for (w, m, gm) in [(Weight::Bold, 1.6f32, 1.3f32), (Weight::Extra, 2.2, 1.5)] {
            let g = build(&weighted(d.clone(), w), Mode::D2);
            assert_eq!(g.segments.len(), normal.segments.len());
            for (a, b) in normal.segments.iter().zip(&g.segments) {
                assert!(
                    near(b.width, a.width * m) || near(b.width, a.width * gm),
                    "{} -> {} at {m}/{gm}",
                    a.width,
                    b.width
                );
            }
            let ws = widths(&g);
            // Axes and curves take the full multiplier; the minor grid only the gentle one.
            for (base, k) in [(AXIS_W, m), (CURVE_W, m), (MINOR_W, gm)] {
                assert!(ws.iter().any(|x| near(*x, base * k)), "{base} x {k}");
            }
            assert!(!ws.iter().any(|x| near(*x, AXIS_W)), "no unweighted axis");
            assert!(!ws.iter().any(|x| near(*x, MINOR_W * m)), "grid not at the line weight");
            assert!(gm < m);
        }
    }

    #[test]
    fn print_weight_multiplies_an_explicit_line_width_only_when_drawing() {
        let d = styled("y=x", |s| s.line_width = Some(4.0));
        let mut b = weighted(d.clone(), Weight::Bold);
        let (segs, _) = item_segs(&build(&b, Mode::D2), Mode::D2);
        assert!(segs.iter().all(|s| near(s.width, 6.4)));
        // The stored style is untouched, and turning the weight off restores the width.
        assert_eq!(b.items[0].style.line_width, Some(4.0));
        b.view.weight = Weight::Normal;
        let (segs, _) = item_segs(&build(&b, Mode::D2), Mode::D2);
        assert!(segs.iter().all(|s| near(s.width, 4.0)));
    }

    #[test]
    fn print_weight_enlarges_points_up_to_a_clamp_and_scales_dashes() {
        let pt = |size: Option<f64>, w| {
            let d = weighted(styled("(1,2)", |s| s.point_size = size), w);
            let g = build(&d, Mode::D2);
            let dots: Vec<f32> = g.segments.iter().filter(|s| s.p0 == s.p1 && s.width > 4.0)
                .map(|s| s.width).collect();
            assert_eq!(dots.len(), 1, "{dots:?}");
            dots[0]
        };
        assert!(near(pt(None, Weight::Normal), DOT_W));
        assert!(near(pt(None, Weight::Bold), DOT_W * 1.3));
        assert!(near(pt(None, Weight::Extra), DOT_W * 1.6));
        assert!(near(pt(Some(20.0), Weight::Extra), 32.0));
        assert!(near(pt(Some(30.0), Weight::Extra), 40.0), "clamped to 40");
        assert!(near(pt(Some(60.0), Weight::Extra), 60.0), "never below the request");
        assert!(near(pt(Some(60.0), Weight::Normal), 60.0));
        // A dashed curve keeps its dash-to-width proportions: fewer, thicker dashes.
        let dashed = |w| {
            let d = weighted(styled("y=x", |s| s.line_style = Some(LineStyle::Dashed)), w);
            item_segs(&build(&d, Mode::D2), Mode::D2).0
        };
        let (n, b) = (dashed(Weight::Normal), dashed(Weight::Bold));
        assert!(b.len() < n.len() && b.len() * 2 > n.len(), "{} vs {}", b.len(), n.len());
        assert!(b.iter().all(|s| near(s.width, CURVE_W * 1.6)));
    }

    #[test]
    fn print_weight_applies_to_polar_log_and_3d_builds() {
        let mut polar = weighted(doc_with(&[("a", "r=2")]), Weight::Bold);
        polar.view.grid_kind = math_core::doc::GridKind::Polar;
        let mut log = weighted(doc_with(&[("a", "y=x")]), Weight::Bold);
        log.view.x_scale = math_core::doc::AxisScale::Log;
        log.view.y_scale = math_core::doc::AxisScale::Log;
        log.view.window.min = [0.1, 0.1, -1.0];
        log.view.window.max = [100.0, 100.0, 1.0];
        let d3 = weighted(doc_with(&[("a", "z=x+y")]), Weight::Extra);
        for (d, mode, m, gm) in [
            (polar, Mode::D2, 1.6f32, 1.3f32),
            (log, Mode::D2, 1.6, 1.3),
            (d3, Mode::D3, 2.2, 1.5),
        ] {
            let base = build(&weighted(d.clone(), Weight::Normal), mode);
            let g = build(&d, mode);
            assert_eq!(g.segments.len(), base.segments.len());
            assert!(!g.segments.is_empty());
            for (a, b) in base.segments.iter().zip(&g.segments) {
                assert!(near(b.width, a.width * m) || near(b.width, a.width * gm));
            }
            // The grid lines (width 1.0 or 1.5) never exceed the gentle multiplier.
            assert!(g.segments.iter().zip(&base.segments).any(|(b, a)| near(b.width, a.width * gm)));
        }
    }

    #[test]
    fn render_scale_multiplies_widths_for_one_build_only() {
        let d = weighted(doc_with(&[("a", "y=x")]), Weight::Bold);
        let one = build(&d, Mode::D2);
        set_render_scale(2.0);
        let two = build(&d, Mode::D2);
        set_render_scale(f32::NAN);
        let after = build(&d, Mode::D2);
        for ((a, b), c) in one.segments.iter().zip(&two.segments).zip(&after.segments) {
            assert!(near(b.width, a.width * 2.0));
            assert!(near(c.width, a.width));
        }
    }

    #[test]
    fn dashed_and_dotted_curves_leave_gaps() {
        for (src, mode) in [
            ("y=x", Mode::D2),
            ("x^2+y^2=25", Mode::D2),
            ("r=2+\\sin(3\\theta)", Mode::D2),
            ("(\\cos(t),\\sin(t),t)", Mode::D3),
        ] {
            let (solid, ls) = item_segs(&build(&styled(src, |_| {}), mode), mode);
            let dashed_doc = styled(src, |s| s.line_style = Some(LineStyle::Dashed));
            let (dashed, ld) = item_segs(&build(&dashed_doc, mode), mode);
            assert!(!solid.is_empty() && !dashed.is_empty(), "{src}");
            let r = ld / ls;
            assert!(r > 0.25 && r < 0.75, "{src}: dashed/solid length {r}");
            let dotted_doc = styled(src, |s| s.line_style = Some(LineStyle::Dotted));
            let (dotted, _) = item_segs(&build(&dotted_doc, mode), mode);
            assert!(dotted.len() > 10, "{src}");
            assert!(
                dotted.iter().all(|s| s.p0 == s.p1 && s.width == CURVE_W),
                "{src}"
            );
        }
        // A straight 2-row table line: one solid segment, many shorter dashes, phase-continuous
        // across the vertex of a 3-row line.
        let mut d = table_doc(
            &[("x_1", &["-5", "5"]), ("y_1", &["0", "0"])],
            TableStyle::Line,
        );
        let (solid, _) = item_segs(&build(&d, Mode::D2), Mode::D2);
        assert_eq!(solid.iter().filter(|s| s.p0 != s.p1).count(), 1);
        d.items[0].style.line_style = Some(LineStyle::Dashed);
        let (dashed, _) = item_segs(&build(&d, Mode::D2), Mode::D2);
        let dashes: Vec<_> = dashed.iter().filter(|s| s.p0 != s.p1).collect();
        // 10 units = 400 px; at width 2.5 the visible dash is 8 px and the visible gap 5 px, so
        // the period is 13 px: 31 dashes.
        assert!((30..=32).contains(&dashes.len()), "{}", dashes.len());
        let on = (dashes[0].p1[0] - dashes[0].p0[0]) as f64 * 40.0;
        assert!(
            (on - 5.5).abs() < 1e-3,
            "visible dash 8 px = 5.5 px + caps: {on}"
        );
        // Geometric gap between consecutive dashes: 13 - 5.5 = 7.5 px (5 px visible).
        let gap = (dashes[1].p0[0] - dashes[0].p1[0]) as f64 * 40.0;
        assert!((gap - 7.5).abs() < 1e-3, "{gap}");
    }

    #[test]
    fn strict_inequality_boundary_is_dashed() {
        let len = |src: &str, f: &dyn Fn(&mut ItemStyle)| {
            item_segs(&build(&styled(src, |s| f(s)), Mode::D2), Mode::D2).1
        };
        let solid = len("x\\ge 2", &|_| {});
        assert!(solid > 0.0);
        for src in ["x>2", "x<2"] {
            let r = len(src, &|_| {}) / solid;
            assert!(r > 0.25 && r < 0.75, "{src}: strict boundary dashed, ratio {r}");
        }
        // An explicit solid line style wins over the strict default.
        let r = len("x>2", &|s| s.line_style = Some(LineStyle::Solid)) / solid;
        assert!((r - 1.0).abs() < 0.05, "explicit solid: ratio {r}");
    }

    #[test]
    fn constant_that_is_not_a_number_is_undefined() {
        for src in ["y=1/0", "y=0/0"] {
            let g = build(&styled(src, |_| {}), Mode::D2);
            assert!(
                g.diagnostics.iter().any(|d| d.1 == "undefined"),
                "{src}: {:?}",
                g.diagnostics
            );
        }
        let g = build(&styled("y=2", |_| {}), Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
    }

    #[test]
    fn view_minor_grid_arrows_names_and_steps_shape_the_scene() {
        let seg_count = |d: &Doc| build(d, Mode::D2).segments.len();
        let base = Doc::new_default();
        let mut no_minor = base.clone();
        no_minor.view.minor_grid = false;
        assert!(seg_count(&no_minor) < seg_count(&base), "minor lines dropped");
        let mut arrows = base.clone();
        arrows.view.arrows = true;
        assert_eq!(seg_count(&arrows), seg_count(&base) + 4, "two heads of two strokes");
        let mut named = base.clone();
        named.view.x_label = Some("time".into());
        named.view.y_label = Some("h".into());
        let g = build(&named, Mode::D2);
        assert!(g.labels.iter().any(|l| l.axis == 0 && l.text == "time"));
        assert!(g.labels.iter().any(|l| l.axis == 1 && l.text == "h"));
        // A fixed x step of 1 labels every integer; the auto step for [-10, 10] is 2.
        let mut stepped = base.clone();
        stepped.view.x_step = Some(1.0);
        let has = |d: &Doc, t: &str| build(d, Mode::D2).labels.iter().any(|l| l.axis == 0 && l.text == t);
        assert!(!has(&base, "3") && has(&stepped, "3"));
        // A step that would draw hundreds of lines falls back to the automatic one.
        let mut tiny = base.clone();
        tiny.view.x_step = Some(1e-6);
        assert!(seg_count(&tiny) < 2000);
    }

    /// A 2D build of `d` as the app does it: the document's axis map and its display window.
    fn build_mapped(d: &Doc) -> (SceneGeometry, AxisMap) {
        let size = (800, 600);
        let map = doc_map(d, Mode::D2, size);
        let mut win = map.to_display(&d.view.window);
        // The camera derives y from the aspect, as `App::scene_window` does.
        let (cy, hy) = (
            0.5 * (win.min[1] + win.max[1]),
            0.5 * (win.max[0] - win.min[0]) * 600.0 / 800.0,
        );
        win.min[1] = cy - hy;
        win.max[1] = cy + hy;
        let g = build_scene_mapped(d, map, Mode::D2, win, [0.0; 3], size, &Theme::light());
        (g, map)
    }

    fn log_doc(items: &[(&str, &str)], x: Option<[f64; 2]>, y: Option<[f64; 2]>) -> Doc {
        use math_core::doc::AxisScale;
        let mut d = doc_with(items);
        for (a, r) in [(0, x), (1, y)] {
            if let Some([lo, hi]) = r {
                d.view.window.min[a] = lo;
                d.view.window.max[a] = hi;
                if a == 0 {
                    d.view.x_scale = AxisScale::Log;
                } else {
                    d.view.y_scale = AxisScale::Log;
                }
            }
        }
        d
    }

    /// The display points of the item's drawn segments (the grid and axes come first).
    fn item_points(g: &SceneGeometry, grid: &SceneGeometry) -> Vec<[f64; 2]> {
        g.segments[grid.segments.len()..]
            .iter()
            .flat_map(|s| [s.p0, s.p1])
            .map(|p| [p[0] as f64, p[1] as f64])
            .collect()
    }

    #[test]
    fn log_axes_label_decades_and_draw_minor_lines() {
        let d = log_doc(&[], Some([0.01, 1000.0]), None);
        let (g, map) = build_mapped(&d);
        assert!(!map.is_linear());
        let xs: Vec<&str> = g
            .labels
            .iter()
            .filter(|l| l.axis == 0)
            .map(|l| l.text.as_str())
            .collect();
        for t in ["0.01", "0.1", "1", "10", "100", "1000"] {
            assert!(xs.contains(&t), "{t} missing from {xs:?}");
        }
        // Decade labels sit at k * log10(v) in display space.
        let ten = g
            .labels
            .iter()
            .find(|l| l.axis == 0 && l.text == "10")
            .unwrap();
        assert!((ten.pos[0] - map.k[0]).abs() < 1e-9);
        // Minor lines at 2..9 times each power.
        let mut no_minor = d.clone();
        no_minor.view.minor_grid = false;
        assert!(build_mapped(&no_minor).0.segments.len() < g.segments.len());
        // y stays linear with its usual numbers.
        assert!(g.labels.iter().any(|l| l.axis == 1 && l.text == "5"));
        // Tiny and huge decades use the tick format (1e-5, 1e6).
        let wide = log_doc(&[], Some([1e-6, 1e7]), None);
        let w: Vec<String> = build_mapped(&wide)
            .0
            .labels
            .iter()
            .filter(|l| l.axis == 0)
            .map(|l| l.text.clone())
            .collect();
        assert!(w.iter().any(|t| t == "1e6" || t == "1e-5"), "{w:?}");
    }

    #[test]
    fn log_y_axis_makes_exponentials_straight_and_bends_lines() {
        let grid = build_mapped(&log_doc(&[], None, Some([0.01, 1000.0]))).0;
        // y = 2^x on a logarithmic y axis: Y = k log10(2) x, a straight line.
        let (g, map) = build_mapped(&log_doc(&[("e", "y=2^x")], None, Some([0.01, 1000.0])));
        let pts = item_points(&g, &grid);
        assert!(
            pts.len() > 10 && g.diagnostics.is_empty(),
            "{:?}",
            g.diagnostics
        );
        let slope = map.k[1] * 2f64.log10();
        for p in &pts {
            assert!(
                (p[1] - slope * p[0]).abs() < 1e-3 * map.k[1],
                "{p:?} off the line"
            );
        }
        // y = x is a curve there (and is not drawn for x <= 0).
        let (g, map) = build_mapped(&log_doc(&[("l", "y=x")], None, Some([0.01, 1000.0])));
        let pts = item_points(&g, &grid);
        assert!(pts.iter().all(|p| p[0] > 0.0));
        let off = pts
            .iter()
            .map(|p| (p[1] - map.fwd(1, p[0])).abs())
            .fold(0.0, f64::max);
        assert!(off < 1e-3 * map.k[1]);
        let (a, b) = (
            pts.iter().find(|p| p[0] > 0.5).unwrap(),
            pts.iter().find(|p| p[0] > 5.0).unwrap(),
        );
        let mid = pts.iter().find(|p| p[0] > 0.5 * (a[0] + b[0])).unwrap();
        let chord = a[1] + (b[1] - a[1]) * (mid[0] - a[0]) / (b[0] - a[0]);
        assert!(
            (mid[1] - chord).abs() > 0.05 * map.k[1],
            "y = x should bend on a log y axis"
        );
    }

    #[test]
    fn log_log_axes_draw_power_laws_straight_and_implicit_curves() {
        let grid = build_mapped(&log_doc(&[], Some([0.01, 1000.0]), Some([0.01, 1000.0]))).0;
        // y = x^2 on log-log axes: Y = 2 (k_y / k_x) X.
        let (g, map) = build_mapped(&log_doc(
            &[("p", "y=x^2")],
            Some([0.01, 1000.0]),
            Some([0.01, 1000.0]),
        ));
        let pts = item_points(&g, &grid);
        assert!(pts.len() > 10);
        for p in &pts {
            assert!((p[1] - 2.0 * map.k[1] / map.k[0] * p[0]).abs() < 1e-3 * map.k[1].max(1.0));
        }
        // An implicit curve (xy = 10) is traced through the map too: Y = k_y (1 - X / k_x).
        let (g, map) = build_mapped(&log_doc(
            &[("h", "xy=10")],
            Some([0.01, 1000.0]),
            Some([0.01, 1000.0]),
        ));
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let pts = item_points(&g, &grid);
        assert!(pts.len() > 4);
        for p in &pts {
            let want = map.k[1] * (1.0 - p[0] / map.k[0]);
            assert!((p[1] - want).abs() < 0.05 * map.k[1], "{p:?} vs {want}");
        }
    }

    #[test]
    fn log_axes_map_points_tables_and_report_what_they_cannot_draw() {
        let d = log_doc(
            &[("p", "(10,100)"), ("q", "(-1,5)")],
            Some([0.1, 1000.0]),
            Some([0.1, 1000.0]),
        );
        let (g, map) = build_mapped(&d);
        let dots: Vec<[f32; 3]> = g
            .segments
            .iter()
            .filter(|s| s.p0 == s.p1)
            .map(|s| s.p0)
            .collect();
        let want = map.fwd3([10.0, 100.0, 0.0]);
        assert!(dots
            .iter()
            .any(|p| (p[0] as f64 - want[0]).abs() < 1e-4 && (p[1] as f64 - want[1]).abs() < 1e-4));
        assert_eq!(dots.len(), 1, "a point at x <= 0 cannot be shown");
        // A table's points are mapped the same way.
        let mut t = table_doc(
            &[("x_1", &["1", "10", "100"]), ("y_1", &["1", "10", "100"])],
            TableStyle::Line,
        );
        t.view = log_doc(&[], Some([0.1, 1000.0]), Some([0.1, 1000.0])).view;
        let (g, map) = build_mapped(&t);
        let dots: Vec<[f32; 3]> = g
            .segments
            .iter()
            .filter(|s| s.p0 == s.p1)
            .map(|s| s.p0)
            .collect();
        for v in [1.0, 10.0, 100.0] {
            let w = map.fwd3([v, v, 0.0]);
            let near = |p: &&[f32; 3]| {
                (p[0] as f64 - w[0]).abs() < 1e-4 && (p[1] as f64 - w[1]).abs() < 1e-4
            };
            assert!(dots.iter().any(|p| near(&p)), "{v}");
        }
        // Items that have no meaning on log axes say so instead of drawing a wrong picture.
        let bad = log_doc(
            &[("v", "(-y, x)"), ("h", "histogram([1,2,2,3])")],
            Some([0.1, 1000.0]),
            Some([0.1, 1000.0]),
        );
        let (g, _) = build_mapped(&bad);
        for id in ["v", "h"] {
            let m = g
                .diagnostics
                .iter()
                .find(|(i, _)| i == id)
                .map(|(_, m)| m.as_str())
                .unwrap_or("");
            assert!(m.contains(LOG_UNSUPPORTED), "{id}: {m:?}");
        }
        // Inequalities still shade (the field evaluates world coordinates).
        let (g, _) = build_mapped(&log_doc(
            &[("i", "y>x")],
            Some([0.1, 1000.0]),
            Some([0.1, 1000.0]),
        ));
        assert_eq!(g.fields.len(), 1);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        // 1D and 3D ignore the scales.
        let mut d3 = log_doc(&[("e", "y=x")], Some([0.1, 1000.0]), None);
        d3.view.mode = math_core::doc::Mode::D3;
        assert!(build(&d3, Mode::D3).diagnostics.is_empty());
    }

    #[test]
    fn polar_grid_draws_circles_spokes_and_angle_labels() {
        let mut d = Doc::new_default();
        d.view.grid_kind = math_core::doc::GridKind::Polar;
        let g = build(&d, Mode::D2);
        let rect = build(&Doc::new_default(), Mode::D2);
        assert!(
            g.segments.len() > rect.segments.len(),
            "circles are many chords"
        );
        // Every grid chord of a major circle lies at a multiple of the step (2) from the origin.
        let major = Theme::light().grid_major;
        let radii: Vec<f64> = g
            .segments
            .iter()
            .filter(|s| s.color == major && s.p0 != s.p1)
            .map(|s| (s.p0[0] as f64).hypot(s.p0[1] as f64))
            .collect();
        for r in [2.0, 4.0, 6.0] {
            assert!(radii.iter().any(|v| (v - r).abs() < 1e-3), "circle r = {r}");
        }
        let angles: Vec<&str> = g
            .labels
            .iter()
            .filter(|l| l.axis == POLAR_LABEL_AXIS)
            .map(|l| l.text.as_str())
            .collect();
        for t in [
            "\u{3c0}/6",
            "\u{3c0}/3",
            "\u{3c0}/2",
            "\u{3c0}",
            "2\u{3c0}/3",
            "5\u{3c0}/6",
            "7\u{3c0}/6",
            "11\u{3c0}/6",
        ] {
            assert!(angles.contains(&t), "{t} missing from {angles:?}");
        }
        // Axis numbers are distances: positive on both sides.
        assert!(g
            .labels
            .iter()
            .any(|l| l.axis == 0 && l.pos[0] < 0.0 && l.text == "4"));
        assert!(!g
            .labels
            .iter()
            .any(|l| l.axis < 2 && l.text.starts_with('-')));
        // Degrees label the spokes in degrees.
        d.view.angle = AngleMode::Deg;
        let g = build(&d, Mode::D2);
        assert!(g
            .labels
            .iter()
            .any(|l| l.axis == POLAR_LABEL_AXIS && l.text == "120\u{b0}"));
        // Minor gridlines become minor circles; no minor gridlines drop them.
        let mut plain = d.clone();
        plain.view.minor_grid = false;
        assert!(build(&plain, Mode::D2).segments.len() < g.segments.len());
        // Far from the origin only the visible arcs are emitted.
        let far = build_scene(
            &d,
            Mode::D2,
            Window3::new([90.0, 90.0, -1.0], [110.0, 110.0, 1.0]),
            [100.0, 100.0, 0.0],
            (800, 600),
            &Theme::light(),
        );
        assert!(far.segments.len() < 5000 && far.segments.len() > 10);
        // 3D keeps its rectangular grid.
        assert_eq!(
            build(&d, Mode::D3).segments.len(),
            build(&Doc::new_default(), Mode::D3).segments.len()
        );
    }
    #[test]
    fn dashed_off_screen_runs_are_bounded() {
        // A parametric curve that leaves the window by far still builds quickly.
        let d = styled("(10^9 t, t)", |s| s.line_style = Some(LineStyle::Dotted));
        let g = build(&d, Mode::D2);
        assert!(g.segments.len() < 200_000, "{}", g.segments.len());
    }

    #[test]
    fn view_flags_hide_grid_axes_and_numbers() {
        let th = Theme::light();
        let grid_w = |s: &SegmentInstance| s.color == th.grid_minor || s.color == th.grid_major;
        let base = build(&Doc::new_default(), Mode::D2);
        assert!(base.segments.iter().any(grid_w));
        assert!(base.segments.iter().any(|s| s.width == AXIS_W));
        assert!(!base.labels.is_empty());
        let mut d = Doc::new_default();
        d.view.grid = false;
        let g = build(&d, Mode::D2);
        assert!(!g.segments.iter().any(grid_w), "no grid lines");
        assert!(g.segments.iter().any(|s| s.width == AXIS_W), "axes kept");
        assert_eq!(g.labels, base.labels, "numbers kept");
        let mut d = Doc::new_default();
        d.view.axes = false;
        let g = build(&d, Mode::D2);
        assert!(
            !g.segments.iter().any(|s| s.color == th.axis),
            "no axis lines or ticks"
        );
        assert!(g.labels.is_empty(), "no tick labels without axes");
        assert!(g.segments.iter().any(grid_w), "grid kept");
        let mut d = Doc::new_default();
        d.view.axis_numbers = false;
        let g = build(&d, Mode::D2);
        assert!(g.labels.is_empty());
        assert_eq!(g.segments, base.segments, "lines unchanged");
        // 1D: the number line goes with the axes; 3D keeps its box edges.
        let mut d = Doc::new_default();
        d.view.axes = false;
        assert!(build(&d, Mode::D1).segments.is_empty());
        d.view.grid = false;
        let g3 = build(&d, Mode::D3);
        assert_eq!(g3.segments.len(), 12, "only the 12 box edges");
        assert!(g3.labels.is_empty());
        let mut d = Doc::new_default();
        d.view.axis_numbers = false;
        let g3 = build(&d, Mode::D3);
        assert!(g3.labels.is_empty() && g3.segments.iter().any(|s| s.width == AXIS_W));
        // Item value labels on the number line are not tick numbers and stay.
        let mut d = doc_with(&[("v", "3")]);
        d.view.axis_numbers = false;
        assert_eq!(build(&d, Mode::D1).labels.len(), 1);
    }

    #[test]
    fn point_styles_and_sizes() {
        let px = |g: &SceneGeometry| item_segs(g, Mode::D2).0;
        // Dot: one zero-length segment as wide as the size.
        let g = build(&styled("(1,2)", |s| s.point_size = Some(20.0)), Mode::D2);
        let v = px(&g);
        assert_eq!(v.len(), 1);
        assert!(v[0].p0 == v[0].p1 && v[0].width == 20.0);
        assert_eq!(
            px(&build(&styled("(1,2)", |_| {}), Mode::D2))[0].width,
            DOT_W
        );
        // Circle: a ring of segments, every vertex the same pixel distance from the point.
        let g = build(
            &styled("(1,2)", |s| s.point_style = Some(PointStyle::Circle)),
            Mode::D2,
        );
        let v = px(&g);
        assert_eq!(v.len(), RING_SIDES);
        let dist =
            |p: [f32; 3]| (((p[0] - 1.0) * 40.0).powi(2) + ((p[1] - 2.0) * 30.0).powi(2)).sqrt();
        let r0 = dist(v[0].p0);
        assert!(r0 > 2.0 && r0 < 4.5, "{r0}");
        assert!(v
            .iter()
            .all(|s| (dist(s.p0) - r0).abs() < 1e-3 && s.p0 != s.p1));
        // A bigger size makes a bigger ring.
        let g = build(
            &styled("(1,2)", |s| {
                s.point_style = Some(PointStyle::Circle);
                s.point_size = Some(30.0);
            }),
            Mode::D2,
        );
        assert!(dist(px(&g)[0].p0) > 2.5 * r0);
        // Cross: two diagonals through the point.
        let g = build(
            &styled("(1,2)", |s| s.point_style = Some(PointStyle::Cross)),
            Mode::D2,
        );
        let v = px(&g);
        assert_eq!(v.len(), 2);
        for s in &v {
            let mid = [(s.p0[0] + s.p1[0]) / 2.0, (s.p0[1] + s.p1[1]) / 2.0];
            assert!((mid[0] - 1.0).abs() < 1e-5 && (mid[1] - 2.0).abs() < 1e-5);
            let (dx, dy) = ((s.p1[0] - s.p0[0]) * 40.0, (s.p1[1] - s.p0[1]) * 30.0);
            assert!((dx.abs() - dy.abs()).abs() < 1e-3, "45 degrees on screen");
        }
        // Square: four sides forming a closed outline, axis aligned.
        let g = build(
            &styled("(1,2)", |s| s.point_style = Some(PointStyle::Square)),
            Mode::D2,
        );
        let v = px(&g);
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|s| s.p0[0] == s.p1[0] || s.p0[1] == s.p1[1]));
        assert_eq!(v[3].p1, v[0].p0);
        // Point lists and table points use the style too; 3D falls back to dots.
        let g = build(
            &styled("[(1,2),(3,4)]", |s| s.point_style = Some(PointStyle::Cross)),
            Mode::D2,
        );
        assert_eq!(px(&g).len(), 4);
        let mut t = table_doc(
            &[("x_1", &["1", "2"]), ("y_1", &["1", "2"])],
            TableStyle::Points,
        );
        t.items[0].style.point_style = Some(PointStyle::Square);
        assert_eq!(px(&build(&t, Mode::D2)).len(), 8);
        let d = styled("(1,2,3)", |s| {
            s.point_style = Some(PointStyle::Circle);
            s.point_size = Some(14.0);
        });
        let (v, _) = item_segs(&build(&d, Mode::D3), Mode::D3);
        assert_eq!(v.len(), 1);
        assert!(v[0].p0 == v[0].p1 && v[0].width == 14.0);
        // 1D: an open ring on the number line.
        let d = styled("(2)", |s| s.point_style = Some(PointStyle::Circle));
        let (v, _) = item_segs(&build(&d, Mode::D1), Mode::D1);
        assert_eq!(v.len(), RING_SIDES);
    }

    #[test]
    fn fill_opacity_scales_the_inequality_shading() {
        let g = build(&styled("y<x", |_| {}), Mode::D2);
        assert_eq!(g.fields[0].color[3], 1.0, "default unchanged");
        let g = build(&styled("y<x", |s| s.fill_opacity = Some(0.5)), Mode::D2);
        let a = g.fields[0].color[3] * FILL_SHADER_ALPHA;
        assert!((a - 0.5).abs() < 1e-6, "effective fill opacity {a}");
        let g = build(
            &styled("y<x", |s| {
                s.fill_opacity = Some(0.5);
                s.opacity = Some(0.5);
            }),
            Mode::D2,
        );
        assert!((g.fields[0].color[3] * FILL_SHADER_ALPHA - 0.25).abs() < 1e-6);
        let g = build(&styled("y<x", |s| s.fill_opacity = Some(0.0)), Mode::D2);
        assert_eq!(g.fields[0].color[3], 0.0);
        // The CPU fallback raster follows it as well.
        let g = build(
            &styled("y<sum(k,k,1,2)", |s| s.fill_opacity = Some(0.6)),
            Mode::D2,
        );
        assert!(
            (g.vertices[0].color[3] - 0.6).abs() < 1e-5,
            "{}",
            g.vertices[0].color[3]
        );
        // 3D region fill too.
        let g = build(&styled("y<x", |s| s.fill_opacity = Some(0.5)), Mode::D3);
        assert!((g.fields[0].color[3] * FILL_SHADER_ALPHA - 0.5).abs() < 1e-6);
    }

    #[test]
    fn show_label_places_item_labels_at_points() {
        let pl = |g: &SceneGeometry| -> Vec<Label> {
            g.labels
                .iter()
                .filter(|l| l.axis == ITEM_LABEL_AXIS)
                .cloned()
                .collect()
        };
        assert!(pl(&build(&doc_with(&[("p", "[(1,2),(3,4)]")]), Mode::D2)).is_empty());
        let g = build(
            &styled("[(1,2),(3,4.5)]", |s| s.show_label = true),
            Mode::D2,
        );
        let l = pl(&g);
        assert_eq!(l.len(), 2);
        assert_eq!((l[0].text.as_str(), l[0].pos), ("(1, 2)", [1.0, 2.0, 0.0]));
        assert_eq!(l[1].text, "(3, 4.5)");
        // Custom text wins; the label alone does not show without the flag.
        let g = build(
            &styled("(1,2)", |s| {
                s.show_label = true;
                s.label = Some("A".into());
            }),
            Mode::D2,
        );
        assert_eq!(pl(&g)[0].text, "A");
        assert!(pl(&build(
            &styled("(1,2)", |s| s.label = Some("A".into())),
            Mode::D2
        ))
        .is_empty());
        // 3D coordinates, capped count.
        let g = build(&styled("(1,2,3)", |s| s.show_label = true), Mode::D3);
        assert_eq!(pl(&g)[0].text, "(1, 2, 3)");
        let many: Vec<String> = (0..150)
            .map(|i| format!("({}, 0)", i as f64 * 0.1))
            .collect();
        let g = build(
            &styled(&format!("[{}]", many.join(",")), |s| s.show_label = true),
            Mode::D2,
        );
        assert_eq!(pl(&g).len(), MAX_POINT_LABELS);
        // A curve with label text gets one label on a drawn sample inside the window.
        let g = build(
            &styled("y=x", |s| {
                s.show_label = true;
                s.label = Some("diag".into());
            }),
            Mode::D2,
        );
        let l = pl(&g);
        assert_eq!(l.len(), 1);
        assert!(
            l[0].text == "diag" && (l[0].pos[0] - l[0].pos[1]).abs() < 0.1,
            "{l:?}"
        );
        // Table points label too.
        let mut t = table_doc(&[("x_1", &["1"]), ("y_1", &["5"])], TableStyle::Points);
        t.items[0].style.show_label = true;
        assert_eq!(pl(&build(&t, Mode::D2))[0].text, "(1, 5)");
    }

    #[test]
    fn items_in_a_hidden_folder_are_not_drawn_but_define() {
        let mut d = Doc::new_default();
        d.add_item(Item::new("f", ItemKind::Folder, "")).unwrap();
        for (id, l) in [("c", "y=x^2"), ("k", "k=3")] {
            let mut it = Item::new(id, ItemKind::Equation, l);
            it.folder = Some("f".into());
            d.add_item(it).unwrap();
        }
        d.add_item(Item::new("u", ItemKind::Equation, "y=k x"))
            .unwrap();
        let shown = build(&d, Mode::D2);
        assert!(shown.diagnostics.is_empty(), "{:?}", shown.diagnostics);
        assert_eq!(shown.item_colors.len(), 2);
        d.items[0].hidden = true;
        let g = build(&d, Mode::D2);
        assert!(
            g.diagnostics.is_empty(),
            "k=3 is still defined: {:?}",
            g.diagnostics
        );
        let ids: Vec<&str> = g.item_colors.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(ids, ["u"], "the parabola is not drawn");
        assert!(g.segments.len() < shown.segments.len());
        // An error inside a hidden folder is not reported either.
        d.items[1].latex = "y=".into();
        assert!(build(&d, Mode::D2).diagnostics.is_empty());
        assert!(with_folders_applied(&Doc::new_default()).is_empty_borrow());
    }

    trait CowExt {
        fn is_empty_borrow(&self) -> bool;
    }
    impl CowExt for std::borrow::Cow<'_, Doc> {
        fn is_empty_borrow(&self) -> bool {
            matches!(self, std::borrow::Cow::Borrowed(_))
        }
    }
}
