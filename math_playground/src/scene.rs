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
//!   `[0, 6*pi]` (spirals); degrees equivalents in degree mode.
//! * Line widths are physical pixels: grid 1.0/1.5, axes 2.0, curves 2.5, dots 9.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use math_core::analyze::{analyze, Kind};
use math_core::ast::{BinOp, Expr, Rel};
use math_core::compile::{compile, Angle, Program};
use math_core::complex::{compile_complex, parse_complex};
use math_core::list::{eval_value, Bindings, Value};
use math_core::stats;
use math_core::table::{ParsedColumn, TableStyle};
use math_core::doc::{AngleMode, Doc, Item, ItemKind};
use math_core::mesh;
use math_core::parse::{parse_with, ParseCtx};
use math_core::resolve::Defs;
use math_core::slice::ResolvedSlice;
use math_core::view::{Mode, Window3};
use math_core::wgsl::emit_module;
use math_core::wgsl_complex::{emit_complex_function, emit_domain_module, COMPLEX_PRELUDE};

mod slice_draw;
pub use slice_draw::{inset_rect, label_box, label_box_inside, SlicePanel, ViewReq, LABEL_CHAR_W, LABEL_GAP_X, LABEL_GAP_Y, LABEL_H, SLICE_COLOR};

use crate::geometry::{
    FieldKind, FieldSpec, Label, MeshVertex, SceneGeometry, SegmentInstance, Theme, MAX_FIELD_PARAMS,
};

mod calc_draw;

const MINOR_W: f32 = 1.0;
const MAJOR_W: f32 = 1.5;
const AXIS_W: f32 = 2.0;
const CURVE_W: f32 = 2.5;
const DOT_W: f32 = 9.0;
const TICK_PX: f64 = 4.0;
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

// ---------------------------------------------------------------------------------------------
// Nice spacing and number formatting
// ---------------------------------------------------------------------------------------------

/// Chooses a "nice" major spacing (1, 2 or 5 times a power of ten) so that major lines are
/// about `target_px` pixels apart when `span` world units cover `px` pixels. Degenerate input
/// (non-finite, zero) falls back to a spacing for a unit span.
pub fn nice_step(span: f64, px: f64, target_px: f64) -> f64 {
    let span = if span.is_finite() && span > 0.0 { span.abs() } else { 1.0 };
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
    let step = if step.is_finite() && step > 0.0 { step } else { 1.0 };
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
    (k0 as i64..=k1 as i64).map(|k| (k, k as f64 * step)).collect()
}

fn parse_hex_color(s: &str) -> Option<[f32; 4]> {
    let h = s.strip_prefix('#')?;
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|v| v as f32 / 255.0);
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
            matches!(n.as_str(), "range" | "for" | "index") || STAT_PLOTS.contains(&n.as_str()) || args.iter().any(is_listish)
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
    let (mn, mx) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(*v), a.1.max(*v)));
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
    let (mn, mx) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(*v), a.1.max(*v)));
    let w = match width {
        Some(w) if w.is_finite() && w > 0.0 => w,
        Some(_) => return Err("the bin width must be a positive number".to_string()),
        None => default_bin_width(data),
    };
    let start = (mn / w).floor() * w;
    let slot = |v: f64| (((v - start) / w) + 1e-9).floor().max(0.0);
    let nb = slot(mx) + 1.0;
    if nb.is_nan() || nb > MAX_BINS as f64 {
        return Err(format!("too many bins (more than {MAX_BINS}); use a larger bin width"));
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

    fn seg(&mut self, a: [f64; 3], b: [f64; 3], w: f32, color: [f32; 4]) {
        let (p0, p1) = (self.rb(a), self.rb(b));
        if p0.iter().chain(p1.iter()).all(|v| v.is_finite()) {
            self.out.segments.push(SegmentInstance::new(p0, p1, w, color));
        }
    }

    fn dot(&mut self, p: [f64; 3], color: [f32; 4]) {
        self.seg(p, p, DOT_W, color);
    }

    /// A flat, UNLIT, translucent axis-aligned rectangle on the plane z = 0 (histogram bars,
    /// box plot box). Zero-normal `MeshVertex` quads go to `flat_indices`: the renderer draws
    /// them after the lit mesh and fields with depth TEST but no depth WRITE, so translucent
    /// bars neither hide the grid/curves nor depend on draw order.
    fn quad(&mut self, x0: f64, x1: f64, y0: f64, y1: f64, color: [f32; 4]) {
        let corners = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]].map(|c| self.rb(c));
        if corners.iter().flatten().any(|v| !v.is_finite()) {
            return;
        }
        let base = self.out.vertices.len() as u32;
        for c in corners {
            self.out.vertices.push(MeshVertex::new(c, [0.0; 3], color));
        }
        self.out.flat_indices.extend([0, 1, 2, 0, 2, 3].map(|i| i + base));
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
        for w in pts.windows(2) {
            self.seg(w[0], w[1], st.line_w, st.color);
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
    fn steps(&self, mode: Mode) -> [f64; 3] {
        let (vw, vh) = self.px();
        let span = |a: usize| self.win.max[a] - self.win.min[a];
        match mode {
            Mode::D1 => {
                let s = nice_step(span(0), vw, TARGET_MAJOR_PX);
                [s, s, s]
            }
            Mode::D2 => {
                [nice_step(span(0), vw, TARGET_MAJOR_PX), nice_step(span(1), vh, TARGET_MAJOR_PX), 1.0]
            }
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

    fn grid_and_axes(&mut self, mode: Mode) {
        let steps = self.steps(mode);
        match mode {
            Mode::D1 => self.axes_1d(steps[0]),
            Mode::D2 => {
                self.plane_grid(0.0, steps);
                self.axes_2d(steps);
            }
            Mode::D3 => {
                let zp = if self.win.min[2] <= 0.0 && 0.0 <= self.win.max[2] {
                    0.0
                } else {
                    self.win.min[2]
                };
                self.plane_grid(zp, steps);
                self.box_edges();
                self.axes_3d(steps);
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
                    if is_major != pass_major {
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
                    let (w, c) = if is_major { (MAJOR_W, major_c) } else { (MINOR_W, minor_c) };
                    self.seg(a, b, w, c);
                }
            }
        }
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
            let h = TICK_PX * (hi[1] - lo[1]) / vh;
            for (_, v) in multiples(lo[0], hi[0], steps[0]) {
                self.seg([v, -h, 0.0], [v, h, 0.0], MINOR_W * 1.5, axis_c);
            }
        }
        if y_visible {
            self.seg([0.0, lo[1], 0.0], [0.0, hi[1], 0.0], AXIS_W, axis_c);
            let h = TICK_PX * (hi[0] - lo[0]) / vw;
            for (_, v) in multiples(lo[1], hi[1], steps[1]) {
                self.seg([-h, v, 0.0], [h, v, 0.0], MINOR_W * 1.5, axis_c);
            }
        }
        self.axis_labels(0, steps[0], [cx, cy, 0.0], false);
        self.axis_labels(1, steps[1], [cx, cy, 0.0], true);
    }

    fn axes_1d(&mut self, step: f64) {
        let (vw, _) = self.px();
        let (lo, hi) = (self.win.min, self.win.max);
        let axis_c = self.theme.axis;
        self.seg([lo[0], 0.0, 0.0], [hi[0], 0.0, 0.0], AXIS_W, axis_c);
        // Assumes the 1D strip has the same world-per-pixel scale on y as on x.
        let h = TICK_PX * (hi[0] - lo[0]) / vw;
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

    /// Major tick labels along one axis, anchored at `anchor` on the other axes.
    fn axis_labels(&mut self, axis: u8, step: f64, anchor: [f64; 3], skip_zero: bool) {
        let a = axis as usize;
        for (k, v) in multiples(self.win.min[a], self.win.max[a], step) {
            if skip_zero && k == 0 {
                continue;
            }
            let mut p = anchor;
            p[a] = v;
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
                    self.seg(corner(i), corner(i | bit), MINOR_W, c);
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
        for s in segs {
            self.seg([s[0][0], s[0][1], 0.0], [s[1][0], s[1][1], 0.0], st.line_w, st.color);
        }
        Ok(())
    }

    /// Resolves `raw` for a GPU field. Slider names stay free and are returned as the extra
    /// inputs (sorted), so the shader text does not depend on their values; with too many
    /// sliders (or one named like a spatial/complex variable) everything is folded instead.
    fn field_resolve(&self, raw: &Expr, defs: &Defs) -> Result<(Expr, Vec<String>), String> {
        let r = self.pdefs.resolve(raw).map_err(|e| e.to_string())?;
        let used: Vec<String> = r.free_vars().into_iter().filter(|n| self.sliders.contains_key(n)).collect();
        let clash = used.iter().any(|n| matches!(n.as_str(), "x" | "y" | "z" | "i"));
        if used.len() > MAX_FIELD_PARAMS || clash {
            return Ok((defs.resolve(raw).map_err(|e| e.to_string())?, Vec::new()));
        }
        Ok((r, used))
    }

    fn param_values(&self, names: &[String]) -> Vec<f32> {
        names.iter().map(|n| self.sliders.get(n).copied().unwrap_or(0.0) as f32).collect()
    }

    /// Emits a GPU field over the window's x/y rectangle on the z=0 plane. `raw` is the
    /// UNRESOLVED expression; used sliders become uniform parameters (see `FieldSpec`).
    fn field(&mut self, raw: &Expr, defs: &Defs, kind: FieldKind, color: [f32; 4]) -> Result<(), String> {
        let (e, used) = self.field_resolve(raw, defs)?;
        if e.contains_var("z") {
            return Err("fields may only use x and y (z is not supported)".to_string());
        }
        if calc_draw::uses_reduce(&e) {
            // int/sum/prod have no WGSL form: raster on the CPU instead (sliders folded in).
            let folded = defs.resolve(raw).map_err(|e| e.to_string())?;
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
    fn inequality_field(&mut self, rel: Rel, f: &Expr, defs: &Defs, st: Style) -> Result<(), String> {
        let greater = matches!(rel, Rel::Gt | Rel::Ge);
        self.field(f, defs, FieldKind::Fill { greater }, st.color)
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
                format!("{m}: sum, int and prod are not available in complex items (use a real item)")
            } else {
                m
            }
        })?;
        let wgsl = if used.is_empty() {
            emit_domain_module(&p).map_err(|e| e.to_string())?
        } else {
            let mut m = String::from(COMPLEX_PRELUDE);
            m.push_str(&emit_complex_function("cplx", &p).map_err(|e| e.to_string())?);
            let args: Vec<String> =
                (0..used.len()).map(|i| format!("vec2<f32>({}, 0.0)", param_ref(i))).collect();
            m.push_str(&format!(
                "fn field_color(x: f32, y: f32) -> vec4<f32> {{\n    \
                 return vec4<f32>(am_domain_color(cplx(vec2<f32>(x, y), {})), 1.0);\n}}\n",
                args.join(", ")
            ));
            m
        };
        let alpha = if mode == Mode::D3 { DOMAIN_ALPHA_3D } else { 1.0 };
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
        let p = self.prog(&defs.resolve(rhs).map_err(|e| e.to_string())?, &["x"])?;
        let lines = mesh::sample_explicit(&p, w.min[0], w.max[0], self.vw.max(1.0) as usize, (w.min[1], w.max[1]));
        self.add_lines2(&lines, st);
        Ok(())
    }

    fn draw_2d(&mut self, pr: &Prepared, defs: &Defs, st: Style) -> Result<(), String> {
        match &pr.kind {
            // A bare expression of x alone (`x^2`, `d/dx x^2`, `sin(x)`) is the curve y = expr.
            Kind::Field { expr } if pr.dims[0] && !pr.dims[1] && !pr.dims[2] => self.explicit_y_2d(expr, defs, st),
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
                self.contour(&r, st)
            }
            Kind::Polar { rhs } => {
                let r = defs.resolve(rhs).map_err(|e| e.to_string())?;
                let k = if theta_outside_trig(&r) { 3.0 } else { 2.0 };
                let th = || Expr::var("theta");
                let xe = Expr::bin(BinOp::Mul, r.clone(), Expr::call("cos", vec![th()]));
                let ye = Expr::bin(BinOp::Mul, r, Expr::call("sin", vec![th()]));
                let (px, py) = (self.prog(&xe, &["theta"])?, self.prog(&ye, &["theta"])?);
                let lines = mesh::sample_parametric(&px, &py, 0.0, self.turn() * k, 4000);
                self.add_lines2(&lines, st);
                Ok(())
            }
            Kind::Parametric { components } if components.len() == 2 => {
                let mut ps = Vec::new();
                for c in components {
                    let r = defs.resolve(c).map_err(|e| e.to_string())?;
                    ps.push(self.prog(&r, &["t"])?);
                }
                let lines = mesh::sample_parametric(&ps[0], &ps[1], 0.0, self.turn(), 4000);
                self.add_lines2(&lines, st);
                Ok(())
            }
            Kind::Point { components } if components.len() == 2 => {
                let v = self.eval_tuple(components, defs)?;
                self.dot([v[0], v[1], 0.0], st.color);
                Ok(())
            }
            Kind::List { items } => {
                for it in items {
                    if let Expr::Tuple(c) = it {
                        if c.len() == 2 {
                            let v = self.eval_tuple(c, defs)?;
                            self.dot([v[0], v[1], 0.0], st.color);
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // ----- 3D -------------------------------------------------------------------------------

    fn surface(&mut self, f: &Expr, st: Style) -> Result<(), String> {
        let p = self.prog(f, &["x", "y", "z"])?;
        let m = self.px();
        let depth = ((m.0.min(m.1) / 6.0).log2().round()).clamp(5.0, 8.0) as u32;
        let mesh = mesh::surface_3d(&p, self.win.min, self.win.max, depth, 300_000);
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

    fn curve_3d(&mut self, comps: &[Expr], defs: &Defs, st: Style) -> Result<(), String> {
        let mut ps = Vec::new();
        for c in comps {
            let r = defs.resolve(c).map_err(|e| e.to_string())?;
            ps.push(self.prog(&r, &["t"])?);
        }
        const N: usize = 2048;
        let t1 = self.turn();
        let mut cur: Vec<[f64; 3]> = Vec::new();
        for i in 0..=N {
            let t = t1 * i as f64 / N as f64;
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
            Kind::ExplicitY { rhs } => self.surface(&sub("y", &res(rhs)?), st),
            Kind::ExplicitX { rhs } => self.surface(&sub("x", &res(rhs)?), st),
            Kind::ExplicitZ { rhs } => self.surface(&sub("z", &res(rhs)?), st),
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
            Kind::VectorField { components } => self.vector_field_3d(components, defs, st, pr.item.color.is_some()),
            Kind::Parametric { components } if (2..=3).contains(&components.len()) => {
                self.curve_3d(components, defs, st)
            }
            Kind::Point { components } if (2..=3).contains(&components.len()) => {
                let v = self.eval_tuple(components, defs)?;
                self.dot([v[0], v[1], v.get(2).copied().unwrap_or(0.0)], st.color);
                Ok(())
            }
            Kind::List { items } => {
                for it in items {
                    if let Expr::Tuple(c) = it {
                        if (2..=3).contains(&c.len()) {
                            let v = self.eval_tuple(c, defs)?;
                            self.dot([v[0], v[1], v.get(2).copied().unwrap_or(0.0)], st.color);
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
    fn field_progs(&self, comps: &[Expr], defs: &Defs, vars: &[&str]) -> Result<Vec<Program>, String> {
        comps
            .iter()
            .map(|c| self.prog(&defs.resolve(c).map_err(|e| e.to_string())?, vars))
            .collect()
    }

    /// Length scale for arrows: the 90th percentile of the finite magnitudes (so one pole does
    /// not shrink every other arrow to a dot).
    fn magnitude_reference(mags: &mut [f64]) -> f64 {
        mags.sort_by(|a, b| a.total_cmp(b));
        let r = mags.get(((mags.len() as f64) * 0.9) as usize).or(mags.last()).copied().unwrap_or(1.0);
        if r.is_finite() && r > 0.0 {
            r
        } else {
            1.0
        }
    }

    /// One flat (screen-plane) arrow centred on `c`. `u` is the unit direction in pixel space
    /// (x right, y up), `len_px` its length; `xpp`/`ypp` are world units per pixel.
    #[allow(clippy::too_many_arguments)]
    fn arrow_px(&mut self, c: [f64; 3], u: (f64, f64), len_px: f64, xpp: f64, ypp: f64, color: [f32; 4]) {
        let at = |s: f64| [c[0] + u.0 * s * xpp, c[1] + u.1 * s * ypp, c[2]];
        let (tail, tip) = (at(-0.5 * len_px), at(0.5 * len_px));
        self.seg(tail, tip, ARROW_W, color);
        let head = (0.38 * len_px).clamp(4.0, 10.0);
        for sgn in [-1.0f64, 1.0] {
            let (s, co) = (sgn * BARB_ANGLE).sin_cos();
            // -u rotated by +-BARB_ANGLE.
            let b = (-(u.0 * co - u.1 * s), -(u.0 * s + u.1 * co));
            self.seg(tip, [tip[0] + b.0 * head * xpp, tip[1] + b.1 * head * ypp, tip[2]], ARROW_W, color);
        }
    }

    fn vector_field_2d(&mut self, comps: &[Expr], defs: &Defs, st: Style, fixed_color: bool) -> Result<(), String> {
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
                    progs.get(1).map_or(0.0, |p| p.eval_with(&[*x, *y], &mut stack)),
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
            let color = if fixed_color { st.color } else { self.magnitude_color(f, st.color[3]) };
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
    fn vector_field_1d(&mut self, comp: &Expr, defs: &Defs, st: Style, fixed_color: bool) -> Result<(), String> {
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
            let color = if fixed_color { st.color } else { self.magnitude_color(f, st.color[3]) };
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
    fn vector_field_3d(&mut self, comps: &[Expr], defs: &Defs, st: Style, fixed_color: bool) -> Result<(), String> {
        let progs = self.field_progs(comps, defs, &["x", "y", "z"])?;
        let w = self.win;
        let span = (0..3).map(|a| w.max[a] - w.min[a]).fold(0.0, f64::max);
        let mut step = nice_step(1.0, 1.0, span / 5.0);
        let axis = |a: usize, step: f64| -> Vec<f64> {
            if a == 2 && comps.len() < 3 {
                return vec![0.0];
            }
            multiples(w.min[a], w.max[a], step).into_iter().map(|(_, v)| v).collect()
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
            let color = if fixed_color { st.color } else { self.magnitude_color(f, st.color[3]) };
            if m <= 0.0 {
                self.seg(p, p, 4.0, color);
                continue;
            }
            let d = [v[0] / m, v[1] / m, v[2] / m];
            let len = step * 0.85 * (0.25 + 0.75 * f);
            let tail = [p[0] - d[0] * len / 2.0, p[1] - d[1] * len / 2.0, p[2] - d[2] * len / 2.0];
            let tip = [p[0] + d[0] * len / 2.0, p[1] + d[1] * len / 2.0, p[2] + d[2] * len / 2.0];
            self.seg(tail, tip, ARROW_W, color);
            // Three barbs around the shaft: perpendicular basis from the least aligned axis.
            let k = (0..3).min_by(|a, b| d[*a].abs().total_cmp(&d[*b].abs())).unwrap_or(0);
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let cr = |a: [f64; 3], b: [f64; 3]| {
                [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
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

    /// Plots a table: columns 0 and 1 are x and y (a third column is z in 3D). Rows with a blank
    /// or non-numeric cell are skipped. `Line` also joins consecutive rows.
    fn draw_table(&mut self, cols: &[ParsedColumn], style: TableStyle, defs: &Defs, mode: Mode, st: Style) -> Result<(), String> {
        if style == TableStyle::Hidden || cols.is_empty() {
            return Ok(());
        }
        let rows = cols.iter().map(|c| c.cells.len()).max().unwrap_or(0);
        let mut errors: Vec<String> = Vec::new();
        let mut value = |b: &Self, ci: usize, ri: usize| -> Option<f64> {
            let e = cols.get(ci)?.cells.get(ri)?.as_ref()?;
            match defs.resolve(e).map_err(|e| e.to_string()).and_then(|r| b.prog(&r, &[])) {
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
            let y = if mode == Mode::D1 { Some(0.0) } else { value(self, 1, ri) };
            let z = if mode == Mode::D3 && cols.len() > 2 { value(self, 2, ri) } else { Some(0.0) };
            match (x, y, z) {
                (Some(x), Some(y), Some(z)) => {
                    run.push([x, y, z]);
                    pts.push([x, y, z]);
                }
                _ => runs.push(std::mem::take(&mut run)),
            }
        }
        runs.push(run);
        if style == TableStyle::Line && mode != Mode::D1 {
            for r in &runs {
                self.polyline(r, st);
            }
        }
        for p in pts {
            if mode != Mode::D1 || (p[0] >= self.win.min[0] && p[0] <= self.win.max[0]) {
                self.dot(p, st.color);
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
    fn draw_listish(&mut self, pr: &Prepared, defs: &Defs, mode: Mode, st: Style) -> Result<bool, String> {
        if !matches!(pr.kind, Kind::List { .. } | Kind::Point { .. } | Kind::Value { .. }) {
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
                    self.dot([p[0], 0.0, 0.0], st.color);
                }
            }
            (Mode::D2, 2) => self.dot([p[0], p[1], 0.0], st.color),
            (Mode::D3, 2 | 3) => self.dot([p[0], p[1], p.get(2).copied().unwrap_or(0.0)], st.color),
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
                            self.dot([*x, 0.0, 0.0], st.color);
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
    fn stat_plot(&mut self, name: &str, args: &[Expr], bind: &Bindings, mode: Mode, st: Style) -> Result<(), String> {
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
            let (x0, x1, h) = (start + i as f64 * w, start + (i + 1) as f64 * w, b.len() as f64);
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
                let y = (DOT_W as f64 * 0.5 + 1.0 + k as f64 * (DOT_W as f64 + 0.5)) * ypp;
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
            self.dot([x, 0.0, 0.0], st.color);
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

/// Roots of a single-variable program on `[x0, x1]`: sign-change scan plus bisection. Poles
/// (sign change with a growing value) are rejected. At most 64 roots.
fn find_roots(p: &Program, x0: f64, x1: f64) -> Vec<f64> {
    const N: usize = 4000;
    let mut roots = Vec::new();
    if !(x0.is_finite() && x1.is_finite() && x1 > x0) {
        return roots;
    }
    let mut st = Vec::with_capacity(16);
    let mut f = |x: f64| p.eval_with(&[x], &mut st);
    let mut xa = x0;
    let mut fa = f(xa);
    for i in 1..=N {
        let xb = x0 + (x1 - x0) * i as f64 / N as f64;
        let fb = f(xb);
        if roots.len() >= 64 {
            break;
        }
        if fa == 0.0 {
            roots.push(xa);
        } else if fa.is_finite() && fb.is_finite() && fa * fb < 0.0 {
            let (mut lo, mut hi, mut flo) = (xa, xb, fa);
            for _ in 0..100 {
                let mid = lo / 2.0 + hi / 2.0;
                let fm = f(mid);
                if fm == 0.0 || mid == lo || mid == hi {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if (fm < 0.0) == (flo < 0.0) {
                    lo = mid;
                    flo = fm;
                } else {
                    hi = mid;
                }
            }
            let r = lo / 2.0 + hi / 2.0;
            if f(r).abs() <= fa.abs().max(fb.abs()) {
                roots.push(r);
            }
        }
        xa = xb;
        fa = fb;
    }
    if fa == 0.0 && roots.len() < 64 {
        roots.push(xa);
    }
    roots
}

// ---------------------------------------------------------------------------------------------
// Document preparation and entry point
// ---------------------------------------------------------------------------------------------

fn is_drawable_kind(k: ItemKind) -> bool {
    !matches!(k, ItemKind::Folder | ItemKind::Note | ItemKind::Action)
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
            is_drawable_kind(i.kind) && !i.latex.trim().is_empty() && !math_core::actions::looks_like_action(&i.latex)
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
                    kind: Kind::Value { expr: Expr::Num(0.0) },
                    dims: [false; 3],
                    expr: e.clone(),
                    complex: Some(e),
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
                out.push(Prepared { item: it, kind: a.kind, dims: a.dims, complex: None, expr: e });
            }
            Err(e) => {
                if !it.hidden {
                    diags.push((it.id.clone(), e.to_string()));
                }
            }
        }
    }
    let mut defs = Defs::from_kinds(out.iter().map(|p| &p.kind));
    // Each table column defines a list named by its header (ordinary definitions win).
    let mut tables = Vec::new();
    for it in doc.items.iter().filter(|i| i.kind == ItemKind::Table) {
        let Some(tb) = &it.table else { continue };
        let (cols, errs) = tb.parse(&ctx);
        if !it.hidden {
            for (ci, ri, msg) in errs.iter().take(3) {
                diags.push((it.id.clone(), format!("{} row {}: {msg}", tb.columns[*ci].name, ri + 1)));
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
) -> (Vec<Prepared<'a>>, Defs, Defs, Vec<TableData<'a>>, BTreeMap<String, math_core::regress::FitResult>) {
    let (items, mut defs, mut pdefs, tables) = prepare(doc, diags);
    let fits = calc_draw::fit_regressions(&items, &mut defs, &mut pdefs, diags);
    (items, defs, pdefs, tables, fits)
}

/// Builds the drawable geometry for `doc` in `mode` (see the module docs).
///
/// `window` is the visible box in world space, `origin` the render origin subtracted (in f64)
/// from every position, `viewport_px` the canvas size in physical pixels. Never panics; bad
/// items are reported in `SceneGeometry::diagnostics` and the rest still render. Segment order:
/// grid, axes, content.
pub fn build_scene(
    doc: &Doc,
    mode: Mode,
    window: Window3,
    origin: [f64; 3],
    viewport_px: (u32, u32),
    theme: &Theme,
) -> SceneGeometry {
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
        sliders: doc.sliders.iter().map(|(n, c)| (n.clone(), c.value)).collect(),
    };
    b.grid_and_axes(mode);
    let mut diags = Vec::new();
    let (items, defs, pdefs, tables, fits) = prepare_fitted(doc, &mut diags);
    b.out.infos = calc_draw::collect_infos(&items, &fits, &defs, &pdefs, b.angle);
    b.pdefs = pdefs;
    let mut visible_idx = 0usize;
    for pr in items.iter().filter(|p| !p.item.hidden) {
        if matches!(pr.kind, Kind::Definition { .. }) {
            continue;
        }
        let mut color = pr
            .item
            .color
            .as_deref()
            .and_then(parse_hex_color)
            .unwrap_or_else(|| theme.color(visible_idx));
        visible_idx += 1;
        b.out.item_colors.push((pr.item.id.clone(), color));
        if let Some(o) = pr.item.style.opacity {
            color[3] *= o.clamp(0.0, 1.0) as f32;
        }
        let line_w = pr.item.style.line_width.map(|w| w as f32).unwrap_or(CURVE_W);
        let st = Style { color, line_w };
        if matches!(pr.kind, Kind::Regression { .. }) {
            if let Some(fit) = fits.get(&pr.item.id) {
                if let Err(msg) = b.draw_regression(pr, fit, &defs, mode, st) {
                    diags.push((pr.item.id.clone(), msg));
                }
            }
            continue;
        }
        let r = match mode {
            _ if pr.complex.is_some() => {
                b.draw_complex(pr.complex.as_ref().unwrap_or(&Expr::Num(0.0)), &defs, mode, st)
            }
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
    }
    for t in tables.iter().filter(|t| !t.item.hidden) {
        let mut color = t
            .item
            .color
            .as_deref()
            .and_then(parse_hex_color)
            .unwrap_or_else(|| theme.color(visible_idx));
        visible_idx += 1;
        b.out.item_colors.push((t.item.id.clone(), color));
        if let Some(o) = t.item.style.opacity {
            color[3] *= o.clamp(0.0, 1.0) as f32;
        }
        let line_w = t.item.style.line_width.map(|w| w as f32).unwrap_or(CURVE_W);
        let style = t.item.table.as_ref().map(|x| x.style).unwrap_or_default();
        if let Err(msg) = b.draw_table(&t.cols, style, &defs, mode, Style { color, line_w }) {
            diags.push((t.item.id.clone(), msg));
        }
    }
    // Slice overlay (additive; a slice that does not fit the mode is simply not drawn here, the
    // app reports why).
    if let Some(cfg) = &doc.slice {
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
        sliders: doc.sliders.iter().map(|(n, c)| (n.clone(), c.value)).collect(),
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

/// All draggable points of `doc`: visible 2-tuples with at least one coordinate that is a plain
/// number, a slider, or a numeric definition (`a=3`). Formulas stay fixed.
pub fn point_handles(doc: &Doc) -> Vec<PointHandle> {
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
                numeric_defs.entry(name.as_str()).or_insert(p.item.id.as_str());
            }
        }
    }
    let mut out = Vec::new();
    for p in items.iter().filter(|p| !p.item.hidden && p.complex.is_none()) {
        let Kind::Point { components } = &p.kind else { continue };
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
                    Some(item) => CoordSrc::Def { item: item.to_string(), name: n.clone() },
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
                text: [math_core::print::to_text(&components[0]), math_core::print::to_text(&components[1])],
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
    use math_core::doc::SliderCfg;

    fn doc_with(items: &[(&str, &str)]) -> Doc {
        let mut d = Doc::new_default();
        for (id, l) in items {
            d.items.push(Item::new(id, ItemKind::Equation, l));
        }
        d
    }

    fn build(d: &Doc, mode: Mode) -> SceneGeometry {
        build_scene(d, mode, Window3::default(), [0.0; 3], (800, 600), &Theme::light())
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
        let g = build_scene(&d, Mode::D2, Window3::default(), origin, (800, 600), &Theme::light());
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
    fn sphere_3d_is_closed_with_right_radius() {
        let d = doc_with(&[("s", "x^2+y^2+z^2=4")]);
        let t = std::time::Instant::now();
        let g = build(&d, Mode::D3);
        eprintln!("sphere scene built in {:?}, {} verts", t.elapsed(), g.vertices.len());
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
        d.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 2.0 });
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
            assert!([1.0, 2.0, 5.0].iter().any(|c| (m - c).abs() < 1e-9), "{span} {s}");
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
        let dot = g.segments.iter().find(|s| s.p0 == s.p1 && s.width == DOT_W).expect("dot");
        assert_eq!(dot.p0, [5.0, 0.0, 0.0]);
        assert!(g.labels.iter().any(|l| l.text == "5" && l.pos == [5.0, 0.0, 0.0]));
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
            eprintln!("{mode:?}: {:?} ({} segs, {} verts)", t.elapsed(), g.segments.len(), g.vertices.len());
            assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        }
        let g = build(&d, Mode::D2);
        assert!(g.segments.iter().any(|s| s.p0 == [1.0, 2.0, 0.0] && s.p0 == s.p1));
    }

    #[test]
    fn inequality_emits_fill_field_with_flag() {
        use crate::geometry::FieldKind;
        for (src, greater) in [("y<x^2", false), ("y<=x^2", false), ("y>sin(x)", true), ("y>=x", true)] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert_eq!(g.fields.len(), 1, "{src}");
            assert_eq!(g.fields[0].kind, FieldKind::Fill { greater }, "{src}");
            assert!(g.fields[0].wgsl.contains("fn field_fn(v0: f32, v1: f32) -> f32"));
            // boundary contour still drawn
            assert!(g.segments.len() > base_len(Mode::D2), "{src}");
        }
        // 3D: plane patch plus the boundary surface; 1D: no field.
        assert_eq!(build(&doc_with(&[("a", "y<x^2")]), Mode::D3).fields.len(), 1);
        assert!(build(&doc_with(&[("a", "y<x^2")]), Mode::D1).fields.is_empty());
        // uses z: boundary surface only, no plane fill and no error
        let g = build(&doc_with(&[("a", "x^2+y^2+z^2<36")]), Mode::D3);
        assert!(g.fields.is_empty() && g.diagnostics.is_empty() && !g.vertices.is_empty());
    }

    #[test]
    fn bare_expression_of_x_draws_as_a_curve_like_y_equals() {
        let base = base_len(Mode::D2);
        for src in ["x^2", "2x", r"\frac{d}{dx}x^2", r"\frac{d}{dx}\left(x^3\right)", r"\frac{d^2}{dx^2}\sin x", "d/dx x^2", "sin(x)"] {
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
        assert_eq!(build(&doc_with(&[("a", r"\frac{d}{dx}\left(x^2y\right)")]), Mode::D2).fields.len(), 1);
    }

    #[test]
    fn mathlive_chain_fills_only_between_the_bounds() {
        use crate::geometry::FieldKind;
        use math_core::analyze::{analyze, Kind};
        use math_core::compile::{compile, Angle};
        for src in [r"0\le y\le x^2", r"0\leq y\leq x^{2}", r"x^2\ge y\ge 0", r"x^{2}\geq y\geq0", "0<=y<=x^2"] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert_eq!(g.fields.len(), 1, "{src}");
            assert_eq!(g.fields[0].kind, FieldKind::Fill { greater: false }, "{src}");
            // the shader and the boundary contour use the one margin function: both bounds
            let e = math_core::parse::parse(src).unwrap();
            let Kind::Inequality { f, .. } = analyze(&e, &BTreeSet::new()).kind else { panic!("{src}") };
            let m = compile(&f, &["x", "y"], Angle::Rad).unwrap();
            assert!(m.eval(&[1.0, 0.5]) < 0.0, "{src}: between the bounds is filled");
            assert!(m.eval(&[0.5, 5.0]) > 0.0, "{src}: above x^2 is NOT filled");
            assert!(m.eval(&[3.0, -1.0]) > 0.0, "{src}: below 0 is NOT filled");
            assert!(m.eval(&[-2.0, 3.0]) < 0.0, "{src}: x<0 side too");
        }
    }

    #[test]
    fn mathlive_integral_and_sum_items_evaluate() {
        for src in [r"\int_{0}^{2}x^{2}\,dx", r"\sum_{n=0}^{6}\frac{x^{n}}{n!}", r"y=\int_0^x t\,dt", r"\prod_{k=1}^{4}k"] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
        }
        let g = build(&doc_with(&[("a", r"\int_{0}^{2}x^{2}")]), Mode::D2);
        assert_eq!(g.diagnostics.len(), 1);
        assert!(g.diagnostics[0].1.contains("differential"), "{:?}", g.diagnostics);
        let g = build(&doc_with(&[("a", r"\int x\,dx")]), Mode::D2);
        assert!(g.diagnostics[0].1.contains("indefinite"), "{:?}", g.diagnostics);
    }

    #[test]
    fn letter_times_latex_function_is_a_slider_candidate_like_k() {
        for (a, k) in [(r"y=a\sin\left(x\right)", r"y=k\sin\left(x\right)"), (r"y=a\cos x", "y=k*cos(x)"), (r"y=a\ln x", "y=k*ln(x)")] {
            let da = build(&doc_with(&[("i", a)]), Mode::D2).diagnostics;
            let dk = build(&doc_with(&[("i", k)]), Mode::D2).diagnostics;
            assert_eq!(da.len(), 1, "{a}: {da:?}");
            assert_eq!(da[0].1, "undefined variable 'a'", "{a}");
            assert_eq!(dk[0].1, "undefined variable 'k'");
        }
        let mut d = doc_with(&[("i", r"y=a\sin\left(x\right)")]);
        d.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 2.0 });
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(g.segments.len() > base_len(Mode::D2) + 20);
    }

    #[test]
    fn field_rect_is_origin_relative() {
        let d = doc_with(&[("a", "x+y")]);
        let origin = [3.0, -2.0, 0.0];
        let g = build_scene(&d, Mode::D2, Window3::default(), origin, (800, 600), &Theme::light());
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
        d.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 2.0 });
        let f2 = build(&d, Mode::D2).fields[0].clone();
        d.sliders.get_mut("a").unwrap().value = 3.0;
        let f3 = build(&d, Mode::D2).fields[0].clone();
        // Slider values are uniforms: same shader text (pipeline cache key), new parameter.
        assert_eq!(f2.wgsl, f3.wgsl, "a slider drag must not change the shader");
        assert_eq!(f2.params, vec![2.0]);
        assert_eq!(f3.params, vec![3.0]);
        assert!(f2.wgsl.contains("fp.params[0].x") && f2.wgsl.contains("fn field_core(v0: f32, v1: f32, v2: f32)"));
    }

    #[test]
    fn slider_field_params_are_sorted_and_definitions_stay_folded() {
        // `k` is a plain definition (folded); `b` and `a` are sliders (inputs, sorted by name).
        let mut d = doc_with(&[("k", "k=4"), ("f", "y<b*x+a*k")]);
        d.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 2.0 });
        d.sliders.insert("b".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 7.0 });
        let f = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(f.params, vec![2.0, 7.0]);
        assert!(f.wgsl.contains("4.0") && f.wgsl.contains("fp.params[0].y"), "{}", f.wgsl);
        d.sliders.get_mut("b").unwrap().value = -1.0;
        let g = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(f.wgsl, g.wgsl);
        assert_eq!(g.params, vec![2.0, -1.0]);
        // A slider overrides a same-named definition.
        let mut d = doc_with(&[("k", "k=4"), ("f", "y<k*x")]);
        d.sliders.insert("k".into(), SliderCfg { min: 0.0, max: 9.0, step: None, value: 5.0 });
        assert_eq!(build(&d, Mode::D2).fields[0].params, vec![5.0]);
    }

    #[test]
    fn too_many_sliders_fall_back_to_folding() {
        let names: Vec<String> = (0..17).map(|i| format!("{}", (b'a' + i as u8) as char)).collect();
        let latex = format!("y<{}", names.iter().map(|n| format!("{n}*x")).collect::<Vec<_>>().join("+"));
        let mut d = doc_with(&[("f", latex.as_str())]);
        for n in &names {
            d.sliders.insert(n.clone(), SliderCfg { min: 0.0, max: 9.0, step: None, value: 2.0 });
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
                assert!(f.wgsl.contains("fn field_color(x: f32, y: f32) -> vec4<f32>"));
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
        assert!(g.diagnostics[0].1.contains("undefined variable 'w'"), "{:?}", g.diagnostics);
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
        d.sliders.insert("a".into(), SliderCfg { min: -5.0, max: 5.0, step: None, value: 2.0 });
        let f2 = build(&d, Mode::D2).fields[0].clone();
        d.sliders.get_mut("a").unwrap().value = 3.0;
        let f3 = build(&d, Mode::D2).fields[0].clone();
        assert_eq!(f2.wgsl, f3.wgsl, "domain colouring: slider is a uniform too");
        assert_eq!((f2.params.clone(), f3.params.clone()), (vec![2.0], vec![3.0]));
        let w3 = f3.wgsl;
        // A slider or definition named z / i must not replace the plane variable.
        d.sliders.insert("z".into(), SliderCfg { min: 0.0, max: 9.0, step: None, value: 7.0 });
        d.sliders.insert("i".into(), SliderCfg { min: 0.0, max: 9.0, step: None, value: 7.0 });
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
        assert!(g.segments.len() > base_len(Mode::D2), "real item still draws");
    }

    // ----- lists and statistical plots ------------------------------------------------------

    /// World-space dots (p0 == p1, dot width) among the content segments, origin added back.
    fn dots(g: &SceneGeometry, mode: Mode, origin: [f64; 3]) -> Vec<[f64; 3]> {
        g.segments[base_len(mode)..]
            .iter()
            .filter(|s| s.p0 == s.p1 && s.width == DOT_W)
            .map(|s| [s.p0[0] as f64 + origin[0], s.p0[1] as f64 + origin[1], s.p0[2] as f64 + origin[2]])
            .collect()
    }

    fn build_at(d: &Doc, mode: Mode, origin: [f64; 3]) -> SceneGeometry {
        build_scene(d, mode, Window3::default(), origin, (800, 600), &Theme::light())
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
                assert!(vs.iter().all(|v| v.normal == [0.0; 3] && v.pos[2] == (-origin[2]) as f32));
                let xs: Vec<f64> = vs.iter().map(|v| v.pos[0] as f64 + origin[0]).collect();
                let ys: Vec<f64> = vs.iter().map(|v| v.pos[1] as f64 + origin[1]).collect();
                let f = |v: &[f64], m: fn(f64, f64) -> f64, init: f64| v.iter().copied().fold(init, m);
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
        assert!(close(s.color[2] as f64, 0x99 as f64 / 255.0) && close(s.color[3] as f64, 0.5), "{:?}", s.color);
    }

    #[test]
    fn point_lists_in_3d_and_1d() {
        let mut d = doc_with(&[("p", "(a, b, c)"), ("q", "(a, b)")]);
        for (n, v) in [("a", "[1,2]"), ("b", "[3,4]"), ("c", "[5,6]")] {
            d.items.push(Item::new(n, ItemKind::Equation, &format!("{n}={v}")));
        }
        let g = build(&d, Mode::D3);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let pts = dots(&g, Mode::D3, [0.0; 3]);
        assert_eq!(pts.len(), 4);
        assert!(pts.contains(&[1.0, 3.0, 5.0]) && pts.contains(&[2.0, 4.0, 6.0]));
        assert!(pts.contains(&[1.0, 3.0, 0.0]) && pts.contains(&[2.0, 4.0, 0.0]), "2-comp at z=0: {pts:?}");
        // 2D: 3-component points have no place on the plane (no diagnostic either).
        let g = build(&doc_with(&[("p", "[(1,2,3)]")]), Mode::D2);
        assert!(g.diagnostics.is_empty() && dots(&g, Mode::D2, [0.0; 3]).is_empty());
        // 1D: dots at the x coordinate.
        let g = build(&doc_with(&[("p", "[(1,2),(3,4)]")]), Mode::D1);
        assert!(g.diagnostics.is_empty());
        let pts = dots(&g, Mode::D1, [0.0; 3]);
        assert_eq!(pts.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>(), vec![(1.0, 0.0), (3.0, 0.0)]);
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
        let g = build(&doc_with(&[("l", "[1...5]"), ("m", "mean([1,2,6])")]), Mode::D1);
        assert_eq!(dots(&g, Mode::D1, [0.0; 3]).len(), 6);
    }

    #[test]
    fn slider_flows_into_list_items() {
        let mut d = doc_with(&[("l", "[n*a for n=[1...5]]")]);
        d.sliders.insert("a".into(), SliderCfg { min: 0.0, max: 5.0, step: None, value: 2.0 });
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
        d.sliders.insert("w".into(), SliderCfg { min: 0.1, max: 5.0, step: None, value: 1.0 });
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
        assert!(g.indices.is_empty(), "bars are flat geometry, not lit surfaces");
        let b = bars(&g, origin);
        let want = [[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [3.0, 4.0, 3.0], [4.0, 5.0, 2.0], [5.0, 6.0, 1.0], [9.0, 10.0, 1.0]];
        assert_eq!(b.len(), want.len(), "{b:?}");
        for (bar, w) in b.iter().zip(want) {
            assert!(close(bar[0], w[0]) && close(bar[1], w[1]) && close(bar[2], 0.0) && close(bar[3], w[2]), "{bar:?} vs {w:?}");
        }
        // Translucent fill; outline uses the full item colour at 1.5 px.
        assert!(g.vertices.iter().all(|v| close(v.color[3] as f64, BAR_ALPHA as f64)));
        let outline: Vec<_> = g.segments[base_len(Mode::D2)..].iter().filter(|s| s.width == BAR_OUTLINE_W).collect();
        assert!(outline.len() >= 12 && outline.iter().all(|s| s.color[3] == 1.0));
        // Default bin width (no argument): every value lands in some bar.
        let d = doc_with(&[("h", &format!("histogram({SAMPLE})"))]);
        let g = build(&d, Mode::D2);
        let total: f64 = bars(&g, [0.0; 3]).iter().map(|b| b[3] - b[2]).sum();
        assert!(g.diagnostics.is_empty() && g.flat_indices.len() >= 12, "{:?}", g.diagnostics);
        assert!(close(total, 10.0), "bar heights are counts and sum to n: {total}");
    }

    #[test]
    fn default_bin_width_rules() {
        let sample = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 9.0];
        // Freedman-Diaconis: IQR = 1.75, n = 10.
        assert!(close(default_bin_width(&sample), 2.0 * 1.75 * 10f64.powf(-1.0 / 3.0)));
        // IQR 0 -> Sturges: range 4 / (ceil(log2 8) + 1).
        assert!(close(default_bin_width(&[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]), 1.0));
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
        let (q1, med, q3) =
            (stats::quartile(&data, 1.0).unwrap(), stats::quartile(&data, 2.0).unwrap(), stats::quartile(&data, 3.0).unwrap());
        let origin = [0.5, 0.0, 0.0];
        let d = doc_with(&[("b", &format!("boxplot({SAMPLE})"))]);
        let g = build_at(&d, Mode::D2, origin);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let b = bars(&g, origin);
        assert_eq!(b.len(), 1);
        assert!(close(b[0][0], q1) && close(b[0][1], q3), "{b:?} q1={q1} q3={q3}");
        assert!(close(b[0][2], -b[0][3]) && b[0][3] > 0.0, "centred on y = 0");
        let segs: Vec<_> = g.segments[base_len(Mode::D2)..].iter().filter(|s| s.p0 != s.p1).collect();
        let x = |v: f32| v as f64 + origin[0];
        // Median line.
        assert!(segs.iter().any(|s| s.width == 3.0 && close(x(s.p0[0]), med) && close(x(s.p1[0]), med)));
        // Whiskers: lower to 1, upper to 5 (9 is beyond Q3 + 1.5 IQR = 6.625).
        let horiz: Vec<_> = segs.iter().filter(|s| s.p0[1] == 0.0 && s.p1[1] == 0.0).collect();
        assert!(horiz.iter().any(|s| close(x(s.p0[0]), q1) && close(x(s.p1[0]), 1.0)), "{horiz:?}");
        assert!(horiz.iter().any(|s| close(x(s.p0[0]), q3) && close(x(s.p1[0]), 5.0)), "{horiz:?}");
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
            let mut ys: Vec<f64> = pts.iter().filter(|p| close(p[0], x)).map(|p| p[1]).collect();
            ys.sort_by(f64::total_cmp);
            ys
        };
        for (x, n) in [(1.0, 1), (2.0, 2), (3.0, 3), (4.0, 2), (5.0, 1), (9.0, 1)] {
            let ys = at(x);
            assert_eq!(ys.len(), n, "x={x}");
            assert!(ys[0] > 0.0 && ys.windows(2).all(|w| w[1] > w[0]), "stacked upward: {ys:?}");
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
                assert_eq!(g.diagnostics.len(), 1, "{bad} {mode:?}: {:?}", g.diagnostics);
                assert_eq!(g.diagnostics[0].0, "it");
                assert!(!g.diagnostics[0].1.is_empty());
                assert!(g.vertices.is_empty() && g.flat_indices.is_empty(), "{bad}");
            }
        }
        // Other items are unaffected by a bad list.
        let g = build(&doc_with(&[("bad", "histogram([])"), ("ok", "[(1,2)]")]), Mode::D2);
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
        g.segments[base_len(mode)..].iter().filter(|s| s.p0 != s.p1).copied().collect()
    }

    #[test]
    fn vector_field_2d_draws_arrows_with_magnitude_colour() {
        let d = doc_with(&[("f", "(-y, x)")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let c = content(&g, Mode::D2);
        // Each arrow is a shaft plus two barbs, so segment count is a multiple of 3.
        assert!(c.len() >= 3 * 100 && c.len().is_multiple_of(3), "{}", c.len());
        // Rotation field: longer arrows far from the origin, so more than one colour appears.
        let mut colours: Vec<[u32; 3]> =
            c.iter().map(|s| [s.color[0].to_bits(), s.color[1].to_bits(), s.color[2].to_bits()]).collect();
        colours.sort();
        colours.dedup();
        assert!(colours.len() > 3, "colour should vary with magnitude");
        // Arrows stay inside (roughly) the window.
        assert!(c.iter().all(|s| s.p0[0].abs() < 12.5 && s.p0[1].abs() < 12.5));
        // Counter-clockwise: at (5, 0) the arrow points up (+y).
        let shaft = c
            .iter()
            .step_by(3)
            .min_by(|a, b| {
                let da = (a.p0[0] + a.p1[0]) / 2.0 - 5.0;
                let db = (b.p0[0] + b.p1[0]) / 2.0 - 5.0;
                (da.abs() + (a.p0[1] + a.p1[1]).abs()).total_cmp(&(db.abs() + (b.p0[1] + b.p1[1]).abs()))
            })
            .unwrap();
        assert!(shaft.p1[1] > shaft.p0[1] && (shaft.p1[0] - shaft.p0[0]).abs() < 1e-3, "{shaft:?}");
    }

    #[test]
    fn vector_field_arrow_count_is_capped_and_scales_with_zoom() {
        let d = doc_with(&[("f", "(sin(y), cos(x))")]);
        let wide = build_scene(&d, Mode::D2, Window3::new([-1e4; 3], [1e4; 3]), [0.0; 3], (4000, 4000), &Theme::light());
        let n = wide.segments.len();
        assert!(n < 3 * super::MAX_ARROWS_2D + 4000, "{n}");
    }

    #[test]
    fn vector_field_uses_sliders_and_definitions() {
        let mut d = doc_with(&[("f", "(a*y, g(x))"), ("g", "g(x)=x")]);
        d.items[1].kind = ItemKind::Expression;
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.iter().any(|(id, m)| id == "f" && m.contains('a')), "{:?}", g.diagnostics);
        d.sliders.insert("a".into(), SliderCfg { min: -10.0, max: 10.0, step: None, value: 2.0 });
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
        assert!(n.is_multiple_of(4) && (4 * 50..=4 * 12 * 12 * 12).contains(&n), "{n}");
        // A planar (P,Q) field in 3D lies on z = 0.
        let g = build(&doc_with(&[("f", "(y, -x)")]), Mode::D3);
        assert!(content(&g, Mode::D3).iter().step_by(4).all(|s| s.p0[2] == 0.0 && s.p1[2] == 0.0), "shafts lie in z = 0");
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
        assert!(n > 1000 && g.segments.iter().filter(|s| s.p0 == s.p1 && s.width == DOT_W).count() == 1, "{n}");
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
        t.style = style;
        let mut it = Item::new("t", ItemKind::Table, "");
        it.table = Some(t);
        d.items.push(it);
        d
    }

    #[test]
    fn table_plots_first_two_columns_as_points_and_lines() {
        let cols: [(&str, &[&str]); 2] = [("x_1", &["1", "2", "3"]), ("y_1", &["2", "4", "a"])];
        let mut d = table_doc(&cols, TableStyle::Points);
        d.sliders.insert("a".into(), SliderCfg { min: 0.0, max: 10.0, step: None, value: 9.0 });
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let dp = dots(&g, Mode::D2, [0.0; 3]);
        assert_eq!(dp.len(), 3);
        assert!(close(dp[2][0], 3.0) && close(dp[2][1], 9.0), "{dp:?}");
        d.items[0].table.as_mut().unwrap().style = TableStyle::Line;
        let g = build(&d, Mode::D2);
        let n_line = g.segments.len() - base_len(Mode::D2);
        assert_eq!(n_line, 3 + 2, "three dots and two joining segments");
        d.items[0].table.as_mut().unwrap().style = TableStyle::Hidden;
        assert!(content(&build(&d, Mode::D2), Mode::D2).is_empty());
    }

    #[test]
    fn table_blank_and_bad_cells_skip_rows_and_report() {
        let cols: [(&str, &[&str]); 2] = [("x_1", &["1", "", "3", "4+"]), ("y_1", &["1", "2", "3", "4"])];
        let g = build(&table_doc(&cols, TableStyle::Points), Mode::D2);
        assert_eq!(dots(&g, Mode::D2, [0.0; 3]).len(), 2);
        assert!(g.diagnostics.iter().any(|(id, m)| id == "t" && m.contains("x_1") && m.contains("row 4")), "{:?}", g.diagnostics);
    }

    #[test]
    fn table_columns_are_lists_for_other_items() {
        let cols: [(&str, &[&str]); 2] = [("x_1", &["1", "2", "3", "4"]), ("y_1", &["1", "2", "3", "4"])];
        let mut d = table_doc(&cols, TableStyle::Hidden);
        d.items.push(Item::new("h", ItemKind::Equation, "histogram(x_1)"));
        d.items.push(Item::new("m", ItemKind::Equation, "mean(y_1)"));
        let g = build(&d, Mode::D1);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        // mean(y_1) = 2.5 is a value: a dot on the number line (hidden table draws nothing).
        let dp = dots(&g, Mode::D1, [0.0; 3]);
        assert_eq!(dp.len(), 1);
        assert!(close(dp[0][0], 2.5));
        let g2 = build(&d, Mode::D2);
        assert!(g2.diagnostics.is_empty(), "{:?}", g2.diagnostics);
        assert!(!g2.vertices.is_empty(), "histogram of a table column draws bars");
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
}
