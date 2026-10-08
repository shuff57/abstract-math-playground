//! Slices (the 1D/2D/3D theme): a lower-dimensional cross-section of the same items.
//!
//! A slice fixes one or two of the axes of the current mode to constants. In 3D, fixing one axis
//! (`z = 1`) gives a 2D slice (a plane); fixing two (`y = 1, z = 0.5`) gives a 1D slice (a line).
//! In 2D, fixing one axis (`y = a`) gives a 1D slice (a line). A constant is a number or any
//! expression of slider/definition names, so dragging a slider sweeps the slice.
//!
//! This module is pure data and algebra (no drawing): the config stored in the document, text
//! parsing, resolution of the constants, and the restriction of an expression to the slice.

use crate::ast::Expr;
use crate::compile::{compile, Angle};
use crate::parse::{parse_with, ParseCtx};
use crate::resolve::Defs;
use crate::view::Mode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Longest accepted constant expression, in chars.
pub const MAX_SLICE_EXPR_CHARS: usize = 200;

/// One slice constant: a number, or an expression string (`"a"`, `"2a+1"`).
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SliceVal {
    Num(f64),
    Expr(String),
}

impl SliceVal {
    fn from_text(s: &str) -> SliceVal {
        match s.trim().parse::<f64>() {
            Ok(v) if v.is_finite() => SliceVal::Num(v),
            _ => SliceVal::Expr(s.trim().to_string()),
        }
    }

    /// Short text form (`1`, `a`).
    pub fn text(&self) -> String {
        match self {
            SliceVal::Num(v) => {
                let s = format!("{v}");
                s
            }
            SliceVal::Expr(s) => s.clone(),
        }
    }
}

/// Slice stored in the document (additive, optional field). `fixed` maps `"x"`/`"y"`/`"z"` to the
/// constant that axis is held at. `dim` is only the dimension the slice had when it was set (kept
/// for the document format); the dimension actually used is always derived from the current mode
/// and the number of fixed axes (see [`ResolvedSlice::resolve`]), so a mode switch never leaves
/// it stale.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliceCfg {
    #[serde(default = "default_dim")]
    pub dim: u8,
    pub fixed: BTreeMap<String, SliceVal>,
}

fn default_dim() -> u8 {
    1
}

/// Axis letter to index.
pub fn axis_index(name: &str) -> Option<usize> {
    match name.trim() {
        "x" | "X" => Some(0),
        "y" | "Y" => Some(1),
        "z" | "Z" => Some(2),
        _ => None,
    }
}

pub fn axis_name(i: usize) -> &'static str {
    ["x", "y", "z"][i.min(2)]
}

/// Splits `s` at top-level commas/semicolons (not inside brackets).
fn split_top(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' | ';' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out.into_iter().map(str::trim).filter(|p| !p.is_empty()).collect()
}

/// Parses `"z=1"`, `"y=1, z=0.5"` or `"y=a"` into axis constants.
pub fn parse_fixed_text(s: &str) -> Result<BTreeMap<String, SliceVal>, String> {
    let mut out = BTreeMap::new();
    for part in split_top(s) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("slice part {part:?} is not axis=value"))?;
        let i = axis_index(k).ok_or_else(|| format!("slice axis {:?} is not x, y or z", k.trim()))?;
        if v.trim().is_empty() {
            return Err(format!("slice axis {} has no value", axis_name(i)));
        }
        out.insert(axis_name(i).to_string(), SliceVal::from_text(v));
    }
    if out.is_empty() {
        return Err("slice needs at least one axis=value".into());
    }
    Ok(out)
}

/// Parses a JSON value: an object `{"z": 0.5, "y": "a"}` or a string `"y=1,z=0.5"`.
pub fn fixed_from_json(v: &serde_json::Value) -> Result<BTreeMap<String, SliceVal>, String> {
    match v {
        serde_json::Value::String(s) => parse_fixed_text(s),
        serde_json::Value::Object(m) => {
            let mut out = BTreeMap::new();
            for (k, val) in m {
                let i = axis_index(k).ok_or_else(|| format!("slice axis {k:?} is not x, y or z"))?;
                let sv = match val {
                    serde_json::Value::Number(n) => SliceVal::Num(n.as_f64().filter(|f| f.is_finite()).ok_or("bad slice number")?),
                    serde_json::Value::String(s) => SliceVal::from_text(s),
                    _ => return Err(format!("slice value for {k} must be a number or a string")),
                };
                out.insert(axis_name(i).to_string(), sv);
            }
            if out.is_empty() {
                return Err("slice needs at least one axis".into());
            }
            Ok(out)
        }
        _ => Err("slice must be an object like {\"z\":0.5} or a string like \"z=0.5\"".into()),
    }
}

impl SliceCfg {
    /// Builds a config. `dim` of `None` is derived from `mode` and the number of fixed axes.
    pub fn new(dim: Option<u8>, fixed: BTreeMap<String, SliceVal>, mode: Mode) -> Result<SliceCfg, String> {
        let ambient = mode.dims() as u8;
        let n = fixed.len() as u8;
        if n == 0 || n >= ambient {
            return Err(format!("a slice of a {ambient}D view fixes 1 to {} axes, got {n}", ambient - 1));
        }
        if let Some(k) = fixed.keys().find(|k| axis_index(k).is_some_and(|i| i >= ambient as usize)) {
            return Err(format!("axis {k} does not exist in the {ambient}D view"));
        }
        let derived = ambient - n;
        let dim = dim.unwrap_or(derived);
        if dim != derived {
            return Err(format!(
                "slice dim {dim} does not match {n} fixed axes in {ambient}D (that is a {derived}D slice)"
            ));
        }
        let cfg = SliceCfg { dim, fixed };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural checks that do not depend on the mode (used when decoding documents).
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2).contains(&self.dim) {
            return Err("slice dim must be 1 or 2".into());
        }
        if self.fixed.is_empty() || self.fixed.len() > 2 {
            return Err("slice fixes 1 or 2 axes".into());
        }
        for (k, v) in &self.fixed {
            if axis_index(k).is_none() {
                return Err(format!("slice axis {k:?} is not x, y or z"));
            }
            match v {
                SliceVal::Num(n) if !n.is_finite() => return Err("slice constant is not finite".into()),
                SliceVal::Expr(s) if s.chars().count() > MAX_SLICE_EXPR_CHARS => {
                    return Err("slice constant too long".into())
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// `"z=1"` / `"y=1,z=0.5"`, for display.
    pub fn label(&self) -> String {
        self.fixed.iter().map(|(k, v)| format!("{k}={}", v.text())).collect::<Vec<_>>().join(", ")
    }
}

/// A slice with its constants evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedSlice {
    pub mode: Mode,
    /// Dimension of the slice itself (1 or 2).
    pub dim: u8,
    /// `Some(c)` for a fixed axis. All `None` for a sloped plane.
    pub fixed: [Option<f64>; 3],
    /// A plane that is not parallel to a coordinate plane (`z = 2x + y`); then `fixed` is empty
    /// and the free coordinates are the in-plane `(u, v)` (named `x`, `y` inside the slice).
    pub plane: Option<Plane>,
}

/// A sloped plane `n . p = d` with an in-plane frame: `e1` is horizontal, `e2` runs
/// up the slope (positive z unless the plane is vertical).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub n: [f64; 3],
    pub d: f64,
    pub e1: [f64; 3],
    pub e2: [f64; 3],
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Plane {
    /// From the plane `g . p = c` (`g` need not be unit). `None` for a zero normal.
    pub fn from_normal(g: [f64; 3], c: f64) -> Option<Plane> {
        let len = dot3(g, g).sqrt();
        if !(len.is_finite() && len > 1e-12) {
            return None;
        }
        let mut n = [g[0] / len, g[1] / len, g[2] / len];
        let mut d = c / len;
        // Normal points up (or +x / +y for a vertical plane) so the frame is stable.
        let flip = if n[2].abs() > 1e-12 { n[2] < 0.0 } else if n[0].abs() > 1e-12 { n[0] < 0.0 } else { n[1] < 0.0 };
        if flip {
            n = n.map(|v| -v);
            d = -d;
        }
        let h = (n[0] * n[0] + n[1] * n[1]).sqrt();
        if h < 1e-12 {
            // Parallel to the xy plane: a frame along x and y.
            return Some(Plane { n, d, e1: [1.0, 0.0, 0.0], e2: [0.0, 1.0, 0.0] });
        }
        let e1 = [n[1] / h, -n[0] / h, 0.0];
        // e2 = e1 x n is up the slope for an upward normal.
        let e2 = [
            e1[1] * n[2] - e1[2] * n[1],
            e1[2] * n[0] - e1[0] * n[2],
            e1[0] * n[1] - e1[1] * n[0],
        ];
        Some(Plane { n, d, e1, e2 })
    }

    /// Ambient point at in-plane coordinates `(u, v)`.
    pub fn lift(&self, u: f64, v: f64) -> [f64; 3] {
        let mut p = [0.0; 3];
        for (a, slot) in p.iter_mut().enumerate() {
            *slot = self.n[a] * self.d + u * self.e1[a] + v * self.e2[a];
        }
        p
    }

    /// In-plane coordinates of the orthogonal projection of `p`.
    pub fn project(&self, p: [f64; 3]) -> [f64; 2] {
        [dot3(p, self.e1), dot3(p, self.e2)]
    }

    /// Signed distance of `p` from the plane.
    pub fn dist(&self, p: [f64; 3]) -> f64 {
        dot3(self.n, p) - self.d
    }

    /// The plane clipped to the box `lo..hi`, as ordered polygon vertices (empty when the plane
    /// misses the box).
    pub fn section(&self, lo: [f64; 3], hi: [f64; 3]) -> Vec<[f64; 3]> {
        let corner = |m: usize| [0, 1, 2].map(|a| if m >> a & 1 == 1 { hi[a] } else { lo[a] });
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for m in 0..8usize {
            for a in 0..3 {
                if m >> a & 1 == 1 {
                    continue;
                }
                let (p, q) = (corner(m), corner(m | 1 << a));
                let (dp, dq) = (self.dist(p), self.dist(q));
                if (dp <= 0.0) != (dq <= 0.0) || dp == 0.0 {
                    let t = if dp == dq { 0.0 } else { dp / (dp - dq) };
                    let r = [0, 1, 2].map(|k| p[k] + t * (q[k] - p[k]));
                    if pts.iter().all(|o| (0..3).any(|k| (o[k] - r[k]).abs() > 1e-9 * (1.0 + r[k].abs()))) {
                        pts.push(r);
                    }
                }
            }
        }
        if pts.len() < 3 {
            return Vec::new();
        }
        let uv: Vec<[f64; 2]> = pts.iter().map(|p| self.project(*p)).collect();
        let c = [
            uv.iter().map(|q| q[0]).sum::<f64>() / uv.len() as f64,
            uv.iter().map(|q| q[1]).sum::<f64>() / uv.len() as f64,
        ];
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        idx.sort_by(|a, b| {
            let (qa, qb) = (uv[*a], uv[*b]);
            (qa[1] - c[1]).atan2(qa[0] - c[0]).total_cmp(&(qb[1] - c[1]).atan2(qb[0] - c[0]))
        });
        idx.into_iter().map(|i| pts[i]).collect()
    }

    /// The `(u, v)` ranges to show for the box `lo..hi`: the section's extent, or the extent of
    /// the projected box corners when the plane misses it.
    pub fn uv_ranges(&self, lo: [f64; 3], hi: [f64; 3]) -> [(f64, f64); 2] {
        let mut pts = self.section(lo, hi);
        if pts.is_empty() {
            pts = (0..8usize)
                .map(|m| [0, 1, 2].map(|a| if m >> a & 1 == 1 { hi[a] } else { lo[a] }))
                .collect();
        }
        let mut r = [(f64::INFINITY, f64::NEG_INFINITY); 2];
        for p in pts {
            let q = self.project(p);
            for k in 0..2 {
                r[k] = (r[k].0.min(q[k]), r[k].1.max(q[k]));
            }
        }
        for k in 0..2 {
            if !(r[k].1 - r[k].0 > 1e-9) {
                r[k] = (r[k].0 - 1.0, r[k].0 + 1.0);
            }
        }
        r
    }
}

/// A slice constant that mentions x, y or z describes a sloped plane.
enum Sloped {
    /// The expression turned out to be a constant plane after all (`z = z/2 + 1`).
    Axis(usize, f64),
    Plane(Plane),
}

/// `Ok(None)` when the config is an ordinary constant slice. A single `axis = expr` whose
/// expr uses x, y or z must be linear (`z = 2x + y`, `y = 1 - x/2`) and describes a plane that
/// need not be parallel to a coordinate plane.
fn resolve_sloped(cfg: &SliceCfg, mode: Mode, defs: &Defs, angle: Angle) -> Result<Option<Sloped>, String> {
    let (k, e) = match cfg.fixed.iter().next() {
        Some((k, SliceVal::Expr(s))) if cfg.fixed.len() == 1 => (k, s),
        _ => return Ok(None),
    };
    let parsed = parse_with(e, &ParseCtx::new()).map_err(|e| format!("slice {k}: {e}"))?;
    let r = defs.resolve(&parsed).map_err(|e| format!("slice {k}: {e}"))?;
    if !["x", "y", "z"].iter().any(|v| r.contains_var(v)) {
        return Ok(None);
    }
    if mode != Mode::D3 {
        return Err("Slice paused: a sloped plane needs the 3D view (it comes back when you return to it).".into());
    }
    let ai = axis_index(k).unwrap_or(0);
    // F = axis - expr is zero on the plane.
    let f = Expr::bin(crate::ast::BinOp::Sub, Expr::Var(axis_name(ai).to_string()), r);
    let prog = compile(&f, &["x", "y", "z"], angle).map_err(|e| format!("slice {k}: {e}"))?;
    let at = |p: [f64; 3]| prog.eval(&p);
    let g0 = at([0.0; 3]);
    let g = [at([1.0, 0.0, 0.0]) - g0, at([0.0, 1.0, 0.0]) - g0, at([0.0, 0.0, 1.0]) - g0];
    let flat = |p: [f64; 3]| {
        let want = g0 + dot3(g, p);
        let got = at(p);
        got.is_finite() && (got - want).abs() <= 1e-9 * (1.0 + want.abs().max(g0.abs()))
    };
    if !(g0.is_finite() && g.iter().all(|v| v.is_finite()))
        || !flat([1.7, -2.3, 0.9])
        || !flat([-3.1, 0.4, 2.2])
        || !flat([5.5, 4.1, -6.3])
    {
        return Err(format!("slice {k}: a sloped plane must be linear in x, y and z (like z = 2x + y)"));
    }
    let Some(plane) = Plane::from_normal(g, -g0) else {
        return Err(format!("slice {k}: that does not describe a plane"));
    };
    // Parallel to a coordinate plane: use the ordinary axis form.
    for a in 0..3 {
        if plane.n[a].abs() > 1.0 - 1e-12 {
            return Ok(Some(Sloped::Axis(a, plane.d / plane.n[a])));
        }
    }
    Ok(Some(Sloped::Plane(plane)))
}

impl ResolvedSlice {
    /// Evaluates the constants of `cfg` against `defs` (which carries slider values and numeric
    /// definitions) and checks it against `mode`.
    pub fn resolve(cfg: &SliceCfg, mode: Mode, defs: &Defs, angle: Angle) -> Result<ResolvedSlice, String> {
        cfg.validate()?;
        let ambient = mode.dims() as usize;
        let name = format!("{ambient}D");
        if mode == Mode::D1 {
            return Err("Slice paused: slices need the 2D or 3D view (it comes back when you return to one).".into());
        }
        let mut fixed = [None; 3];
        let mut collapsed = false;
        if let Some(sloped) = resolve_sloped(cfg, mode, defs, angle)? {
            match sloped {
                Sloped::Axis(i, c) => {
                    fixed[i] = Some(c);
                    collapsed = true;
                }
                Sloped::Plane(plane) => {
                    return Ok(ResolvedSlice { mode, dim: 2, fixed, plane: Some(plane) });
                }
            }
        }
        for (k, v) in &cfg.fixed {
            if collapsed {
                break;
            }
            let i = axis_index(k).unwrap_or(0);
            if i >= ambient {
                return Err(format!(
                    "Slice paused: it fixes {k}, which the {name} view has no axis for (it returns in a view that does)."
                ));
            }
            let val = match v {
                SliceVal::Num(n) => *n,
                SliceVal::Expr(s) => {
                    let e = parse_with(s, &ParseCtx::new()).map_err(|e| format!("slice {k}: {e}"))?;
                    let r = defs.resolve(&e).map_err(|e| format!("slice {k}: {e}"))?;
                    let p = compile(&r, &[], angle).map_err(|e| format!("slice {k}: {e}"))?;
                    p.eval(&[])
                }
            };
            if !val.is_finite() {
                return Err(format!("slice {k} is not a finite number"));
            }
            fixed[i] = Some(val);
        }
        let n = fixed.iter().flatten().count();
        if n == 0 || n >= ambient {
            return Err(format!(
                "Slice paused: fixing {n} axes leaves nothing to show in the {name} view (it returns in a view where it fits)."
            ));
        }
        // Always derived from the mode now, never from the stored `dim`.
        let dim = (ambient - n) as u8;
        Ok(ResolvedSlice { mode, dim, fixed, plane: None })
    }

    /// Indices of the axes that stay free, ascending.
    pub fn free_axes(&self) -> Vec<usize> {
        if self.plane.is_some() {
            return vec![0, 1];
        }
        (0..self.mode.dims() as usize).filter(|i| self.fixed[*i].is_none()).collect()
    }

    /// Indices of the fixed axes, ascending.
    pub fn fixed_axes(&self) -> Vec<usize> {
        (0..3).filter(|i| self.fixed[*i].is_some()).collect()
    }

    /// `e` with every fixed axis replaced by its constant.
    pub fn restrict(&self, e: &Expr) -> Expr {
        match &self.plane {
            Some(pl) => restrict_plane(e, pl),
            None => restrict(e, &self.fixed),
        }
    }

    /// Axis names shown for the free coordinates (`u`, `v` on a sloped plane).
    pub fn free_names(&self) -> Vec<&'static str> {
        if self.plane.is_some() {
            return vec!["u", "v"];
        }
        self.free_axes().into_iter().map(axis_name).collect()
    }

    /// Coordinates of an ambient point in the slice frame (the free axes in order; `u, v` on a
    /// sloped plane).
    pub fn project(&self, p: [f64; 3]) -> Vec<f64> {
        match &self.plane {
            Some(pl) => pl.project(p).to_vec(),
            None => self.free_axes().into_iter().map(|a| p[a]).collect(),
        }
    }

    /// Ambient point from free-axis coordinates `u` (in `free_axes` order); fixed axes get their
    /// constants.
    pub fn lift(&self, u: &[f64]) -> [f64; 3] {
        if let Some(pl) = &self.plane {
            return pl.lift(u.first().copied().unwrap_or(0.0), u.get(1).copied().unwrap_or(0.0));
        }
        let mut p = [0.0; 3];
        let mut k = 0;
        for (i, slot) in p.iter_mut().enumerate() {
            match self.fixed[i] {
                Some(c) => *slot = c,
                None => {
                    *slot = u.get(k).copied().unwrap_or(0.0);
                    k += 1;
                }
            }
        }
        p
    }
}

/// Fraction of the MAIN view's span (along a fixed axis) within which a point counts as lying on
/// the slice: 1 %. A point list shows its members with `|p[a] - c| <= 0.01 * span[a]` for every
/// fixed axis `a`, so the tolerance follows zoom (a point 0.05 off the plane shows in a window
/// 10 wide, not in one 2 wide).
pub const POINT_TOL_FRAC: f64 = 0.01;

/// Name of the stand-in variable for the constant of fixed axis `axis` in GPU field expressions
/// (so a sweeping slice constant is a uniform, not a recompile).
pub fn const_var(axis: usize) -> String {
    format!("$slice_{}", axis_name(axis))
}

impl ResolvedSlice {
    /// Per-axis distance tolerances for [`ResolvedSlice::contains_point`], from the main window.
    pub fn point_tolerances(&self, lo: [f64; 3], hi: [f64; 3]) -> [f64; 3] {
        let mut t = [0.0; 3];
        for (a, slot) in t.iter_mut().enumerate() {
            let span = (hi[a] - lo[a]).abs();
            *slot = if span.is_finite() { POINT_TOL_FRAC * span } else { 0.0 };
        }
        t
    }

    /// True when `p` is within `tol` of every fixed axis constant.
    pub fn contains_point(&self, p: [f64; 3], tol: [f64; 3]) -> bool {
        if let Some(pl) = &self.plane {
            let t: f64 = (0..3).map(|a| pl.n[a].abs() * tol[a]).sum();
            return p.iter().all(|v| v.is_finite()) && pl.dist(p).abs() <= t;
        }
        p.iter().all(|v| v.is_finite())
            && self.fixed.iter().enumerate().all(|(a, c)| c.is_none_or(|c| (p[a] - c).abs() <= tol[a]))
    }

    /// `p` moved onto the slice (fixed axes set to their constants).
    pub fn snap(&self, mut p: [f64; 3]) -> [f64; 3] {
        if let Some(pl) = &self.plane {
            let k = pl.dist(p);
            return [0, 1, 2].map(|a| p[a] - k * pl.n[a]);
        }
        for (a, c) in self.fixed.iter().enumerate() {
            if let Some(c) = c {
                p[a] = *c;
            }
        }
        p
    }
}

/// `e` with each fixed axis replaced by the stand-in variable [`const_var`] and the free axes
/// renamed to `x`, `y` in ascending order, so the result is a function `f(x, y)` (or `f(x)`)
/// whatever the slice orientation. The constants' values are `rs.fixed`.
pub fn virtualize(e: &Expr, rs: &ResolvedSlice) -> Expr {
    if rs.plane.is_some() {
        // The plane's numbers are folded in (a sweeping sloped slice recompiles).
        return rs.restrict(e);
    }
    let mut out = e.clone();
    for a in rs.fixed_axes() {
        out = out.subst(axis_name(a), &Expr::Var(const_var(a)));
    }
    rename_free(&out, rs)
}

/// `e` with the free axes renamed to `x`, `y` in ascending order (a slice `y = c` of a 3D item
/// maps `z` to `y`).
pub fn rename_free(e: &Expr, rs: &ResolvedSlice) -> Expr {
    let mut out = e.clone();
    for (k, a) in rs.free_axes().into_iter().enumerate() {
        if a != k {
            out = out.subst(axis_name(a), &Expr::Var(axis_name(k).to_string()));
        }
    }
    out
}

/// Inset view window `[xmin, xmax, ymin, ymax]` helpers (pure, so they are testable).
pub fn view_valid(v: &[f64; 4]) -> bool {
    v.iter().all(|x| x.is_finite())
        && v[1] > v[0]
        && v[3] > v[2]
        && (v[1] - v[0]) >= 1e-9
        && (v[3] - v[2]) >= 1e-9
        && (v[1] - v[0]) <= 1e9
        && (v[3] - v[2]) <= 1e9
}

/// Pans by a fraction of the span: positive `fx` moves the content right, positive `fy` up.
pub fn pan_view(v: [f64; 4], fx: f64, fy: f64) -> [f64; 4] {
    let (su, sv) = (v[1] - v[0], v[3] - v[2]);
    let out = [v[0] - fx * su, v[1] - fx * su, v[2] - fy * sv, v[3] - fy * sv];
    if view_valid(&out) {
        out
    } else {
        v
    }
}

/// Zooms by `factor` (> 1 zooms in) keeping the data point under the cursor fixed; `(fx, fy)` is
/// the cursor as fractions of the view from the left / from the bottom.
pub fn zoom_view(v: [f64; 4], fx: f64, fy: f64, factor: f64) -> [f64; 4] {
    if !(factor.is_finite() && factor > 0.0) {
        return v;
    }
    let f = factor.clamp(0.1, 10.0);
    let (su, sv) = (v[1] - v[0], v[3] - v[2]);
    let (nu, nv) = (su / f, sv / f);
    let (cu, cv) = (v[0] + fx * su, v[2] + fy * sv);
    let out = [cu - fx * nu, cu + (1.0 - fx) * nu, cv - fy * nv, cv + (1.0 - fy) * nv];
    if view_valid(&out) {
        out
    } else {
        v
    }
}

/// `e` with x, y, z replaced by the point of the plane at in-plane coordinates `(x, y)`.
pub fn restrict_plane(e: &Expr, pl: &Plane) -> Expr {
    use crate::ast::BinOp;
    let o = pl.lift(0.0, 0.0);
    let mut out = e.clone();
    for (a, tmp) in ["$px", "$py", "$pz"].iter().enumerate() {
        out = out.subst(axis_name(a), &Expr::Var((*tmp).to_string()));
    }
    for (a, tmp) in ["$px", "$py", "$pz"].iter().enumerate() {
        let term = |c: f64, v: &str| Expr::bin(BinOp::Mul, Expr::Num(c), Expr::Var(v.to_string()));
        let sum = Expr::bin(
            BinOp::Add,
            Expr::bin(BinOp::Add, Expr::Num(o[a]), term(pl.e1[a], "x")),
            term(pl.e2[a], "y"),
        );
        out = out.subst(tmp, &sum);
    }
    out
}

/// `e` with each axis that has a constant replaced by that number.
pub fn restrict(e: &Expr, fixed: &[Option<f64>; 3]) -> Expr {
    let mut out = e.clone();
    for (i, c) in fixed.iter().enumerate() {
        if let Some(c) = c {
            out = out.subst(axis_name(i), &Expr::Num(*c));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_forms() {
        let f = parse_fixed_text("y=1, z=0.5").unwrap();
        assert_eq!(f["y"], SliceVal::Num(1.0));
        assert_eq!(f["z"], SliceVal::Num(0.5));
        let f = parse_fixed_text("Z = a").unwrap();
        assert_eq!(f["z"], SliceVal::Expr("a".into()));
        assert!(parse_fixed_text("w=1").is_err());
        assert!(parse_fixed_text("z").is_err());
        assert!(parse_fixed_text("").is_err());
        // Commas inside brackets do not split.
        let f = parse_fixed_text("z=max(a,1),y=2").unwrap();
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn json_forms() {
        let f = fixed_from_json(&serde_json::json!({"z": 0.5, "y": "a"})).unwrap();
        assert_eq!(f["z"], SliceVal::Num(0.5));
        assert_eq!(f["y"], SliceVal::Expr("a".into()));
        assert!(fixed_from_json(&serde_json::json!("z=1")).is_ok());
        assert!(fixed_from_json(&serde_json::json!({"q": 1})).is_err());
        assert!(fixed_from_json(&serde_json::json!(3)).is_err());
    }

    #[test]
    fn dim_checks_per_mode() {
        let one = parse_fixed_text("z=1").unwrap();
        assert_eq!(SliceCfg::new(None, one.clone(), Mode::D3).unwrap().dim, 2);
        assert!(SliceCfg::new(Some(1), one.clone(), Mode::D3).is_err());
        // In 2D one fixed axis is a 1D slice.
        assert_eq!(SliceCfg::new(Some(1), parse_fixed_text("y=a").unwrap(), Mode::D2).unwrap().dim, 1);
        assert!(SliceCfg::new(Some(2), one.clone(), Mode::D2).is_err());
        assert!(SliceCfg::new(None, parse_fixed_text("x=1,y=2").unwrap(), Mode::D2).is_err());
        assert!(SliceCfg::new(None, one.clone(), Mode::D2).is_err(), "no z axis in 2D");
        let two = parse_fixed_text("y=1,z=0.5").unwrap();
        assert_eq!(SliceCfg::new(Some(1), two, Mode::D3).unwrap().dim, 1);
    }

    #[test]
    fn resolve_uses_sliders_and_lifts() {
        let mut defs = Defs::new();
        defs.set_slider("a", 2.5);
        let cfg = SliceCfg::new(None, parse_fixed_text("z=a+1").unwrap(), Mode::D3).unwrap();
        let r = ResolvedSlice::resolve(&cfg, Mode::D3, &defs, Angle::Rad).unwrap();
        assert_eq!(r.fixed, [None, None, Some(3.5)]);
        assert_eq!(r.free_axes(), vec![0, 1]);
        assert_eq!(r.lift(&[1.0, 2.0]), [1.0, 2.0, 3.5]);
        // Mode mismatch is an error, not a panic.
        assert!(ResolvedSlice::resolve(&cfg, Mode::D2, &defs, Angle::Rad).is_err());
        let bad = SliceCfg::new(None, parse_fixed_text("z=nope").unwrap(), Mode::D3).unwrap();
        assert!(ResolvedSlice::resolve(&bad, Mode::D3, &defs, Angle::Rad).is_err());
    }

    #[test]
    fn dim_is_derived_per_mode_and_config_survives_round_trips() {
        let defs = Defs::new();
        let cfg = SliceCfg::new(None, parse_fixed_text("y=1,z=0.5").unwrap(), Mode::D3).unwrap();
        let r = |m| ResolvedSlice::resolve(&cfg, m, &defs, Angle::Rad);
        assert_eq!(r(Mode::D3).unwrap().dim, 1);
        // 2D: z does not exist, 1D: nothing slices. Calm message, config untouched.
        let e = r(Mode::D2).unwrap_err();
        assert!(e.starts_with("Slice paused") && !e.contains("does not fit"), "{e}");
        assert!(r(Mode::D1).unwrap_err().starts_with("Slice paused"));
        assert_eq!(r(Mode::D3).unwrap().dim, 1, "3D -> 2D -> 1D -> 3D re-activates");
        // A y-only slice: plane in 3D, line in 2D, paused in 1D; the stored dim (2) never matters.
        let cfg = SliceCfg::new(None, parse_fixed_text("y=1").unwrap(), Mode::D3).unwrap();
        assert_eq!(cfg.dim, 2);
        let r = |m| ResolvedSlice::resolve(&cfg, m, &defs, Angle::Rad);
        assert_eq!(r(Mode::D3).unwrap().dim, 2);
        assert_eq!(r(Mode::D2).unwrap().dim, 1);
        assert!(r(Mode::D1).is_err());
        assert_eq!(r(Mode::D3).unwrap().dim, 2);
        assert_eq!(r(Mode::D2).unwrap().free_axes(), vec![0]);
    }

    #[test]
    fn restrict_substitutes_fixed_axes() {
        let e = parse_with("x^2+y^2+z^2-4", &ParseCtx::new()).unwrap();
        let r = restrict(&e, &[None, None, Some(1.0)]);
        assert!(!r.contains_var("z") && r.contains_var("x"));
        let p = compile(&r, &["x", "y"], Angle::Rad).unwrap();
        assert!((p.eval(&[1.0, 1.0]) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn point_tolerance_scales_with_span() {
        let cfg = SliceCfg::new(None, parse_fixed_text("z=1").unwrap(), Mode::D3).unwrap();
        let rs = ResolvedSlice::resolve(&cfg, Mode::D3, &Defs::new(), Angle::Rad).unwrap();
        let wide = rs.point_tolerances([-10.0; 3], [10.0; 3]);
        let tight = rs.point_tolerances([-1.0; 3], [1.0; 3]);
        assert!((wide[2] - 0.2).abs() < 1e-12 && (tight[2] - 0.02).abs() < 1e-12);
        let p = [3.0, 4.0, 1.1];
        assert!(rs.contains_point(p, wide) && !rs.contains_point(p, tight));
        assert_eq!(rs.snap(p), [3.0, 4.0, 1.0]);
        assert!(!rs.contains_point([0.0, 0.0, f64::NAN], wide));
    }

    #[test]
    fn virtualize_renames_free_axes() {
        let cfg = SliceCfg::new(None, parse_fixed_text("y=2").unwrap(), Mode::D3).unwrap();
        let rs = ResolvedSlice::resolve(&cfg, Mode::D3, &Defs::new(), Angle::Rad).unwrap();
        let e = parse_with("x+y+z", &ParseCtx::new()).unwrap();
        let v = virtualize(&e, &rs);
        let names: Vec<String> = v.free_vars().into_iter().collect();
        assert_eq!(names, vec!["$slice_y".to_string(), "x".to_string(), "y".to_string()], "z became y, y became a constant");
    }

    #[test]
    fn view_pan_and_zoom() {
        let v = [-2.0, 2.0, -1.0, 1.0];
        assert_eq!(pan_view(v, 0.25, 0.0), [-3.0, 1.0, -1.0, 1.0]);
        // Zooming at the left edge keeps the left edge, at the centre keeps the centre.
        assert_eq!(zoom_view(v, 0.0, 0.5, 2.0), [-2.0, 0.0, -0.5, 0.5]);
        let z = zoom_view(v, 0.5, 0.5, 2.0);
        assert_eq!(z, [-1.0, 1.0, -0.5, 0.5]);
        // Absurd input leaves the view alone.
        assert_eq!(zoom_view(v, 0.5, 0.5, f64::NAN), v);
        assert!(!view_valid(&[1.0, 1.0, 0.0, 1.0]) && !view_valid(&[0.0, f64::INFINITY, 0.0, 1.0]));
        let mut w = v;
        for _ in 0..100 {
            w = zoom_view(w, 0.5, 0.5, 10.0);
        }
        assert!(view_valid(&w), "zoom stops at the smallest valid span");
    }

    #[test]
    fn cfg_json_roundtrip() {
        let cfg = SliceCfg::new(None, parse_fixed_text("z=a").unwrap(), Mode::D3).unwrap();
        let j = serde_json::to_string(&cfg).unwrap();
        assert_eq!(serde_json::from_str::<SliceCfg>(&j).unwrap(), cfg);
    }

    #[test]
    fn doc_decodes_with_and_without_slice() {
        let old = r#"{"v":1,"view":{"mode":"2d","window":{"min":[-10,-10,-10],"max":[10,10,10]},"angle":"rad"},"items":[]}"#;
        let d = crate::doc::from_json(old).unwrap();
        assert!(d.slice.is_none());
        assert!(!crate::doc::to_json(&d).contains("slice"), "absent slice is not written");
        let new = r#"{"v":1,"view":{"mode":"3d","window":{"min":[-10,-10,-10],"max":[10,10,10]},"angle":"rad"},"items":[],"slice":{"dim":2,"fixed":{"z":0.5}}}"#;
        let d = crate::doc::from_json(new).unwrap();
        assert_eq!(d.slice.as_ref().unwrap().fixed["z"], SliceVal::Num(0.5));
        let again = crate::doc::from_json(&crate::doc::to_json(&d)).unwrap();
        assert_eq!(again.slice, d.slice);
        // A bad slice is a validation error, not a panic.
        let bad = new.replace("\"z\"", "\"w\"");
        assert!(crate::doc::from_json(&bad).is_err());
    }
    fn sloped(text: &str) -> Result<ResolvedSlice, String> {
        let cfg = SliceCfg::new(None, parse_fixed_text(text).unwrap(), Mode::D3).unwrap();
        ResolvedSlice::resolve(&cfg, Mode::D3, &Defs::new(), Angle::Rad)
    }

    #[test]
    fn sloped_plane_frame_and_lift() {
        let r = sloped("z = 2x + y").unwrap();
        let pl = r.plane.expect("sloped");
        assert_eq!((r.dim, r.free_axes(), r.fixed_axes()), (2, vec![0, 1], vec![]));
        for (u, v) in [(0.0, 0.0), (1.5, -2.0), (-3.0, 4.0)] {
            let p = pl.lift(u, v);
            assert!((p[2] - (2.0 * p[0] + p[1])).abs() < 1e-9, "{p:?} lies on z = 2x + y");
            let q = pl.project(p);
            assert!((q[0] - u).abs() < 1e-9 && (q[1] - v).abs() < 1e-9);
        }
        assert!((dot3(pl.e1, pl.e2)).abs() < 1e-12 && dot3(pl.e1, pl.n).abs() < 1e-12);
        assert!(pl.e2[2] > 0.0, "v runs up the slope");
        assert_eq!(r.free_names(), vec!["u", "v"]);
    }

    #[test]
    fn sloped_plane_restricts_and_snaps() {
        let r = sloped("z = x + y").unwrap();
        // x + y - z vanishes on the plane, so its restriction is zero everywhere in (x, y).
        let e = parse_with("x + y - z", &ParseCtx::new()).unwrap();
        let p = compile(&r.restrict(&e), &["x", "y"], Angle::Rad).unwrap();
        for (u, v) in [(0.3, 0.7), (-2.0, 5.0)] {
            assert!(p.eval(&[u, v]).abs() < 1e-9);
        }
        let off = [1.0, 1.0, 3.0];
        let tol = [0.1; 3];
        assert!(!r.contains_point(off, [0.01; 3]) && r.contains_point([1.0, 1.0, 2.01], tol));
        let on = r.snap(off);
        assert!((on[2] - (on[0] + on[1])).abs() < 1e-9);
    }

    #[test]
    fn sloped_text_can_use_sliders_and_any_axis() {
        let mut defs = Defs::new();
        defs.set_slider("a", 3.0);
        let cfg = SliceCfg::new(None, parse_fixed_text("y = a x").unwrap(), Mode::D3).unwrap();
        let pl = ResolvedSlice::resolve(&cfg, Mode::D3, &defs, Angle::Rad).unwrap().plane.unwrap();
        let p = pl.lift(1.0, 1.0);
        assert!((p[1] - 3.0 * p[0]).abs() < 1e-9, "vertical plane y = 3x");
        assert!(pl.e2[2].abs() > 0.99, "a vertical plane's v axis is z");
    }

    #[test]
    fn sloped_degenerate_cases() {
        // Parallel to a coordinate plane: the ordinary axis form.
        let r = sloped("z = z/2 + 1").unwrap();
        assert!(r.plane.is_none() && r.fixed == [None, None, Some(2.0)]);
        assert!(sloped("z = x^2").unwrap_err().contains("linear"));
        assert!(sloped("z = sin(x)").is_err());
        assert!(sloped("z = z").is_err(), "not a plane");
        let cfg = SliceCfg::new(None, parse_fixed_text("z = x").unwrap(), Mode::D3).unwrap();
        let e = ResolvedSlice::resolve(&cfg, Mode::D2, &Defs::new(), Angle::Rad).unwrap_err();
        assert!(e.starts_with("Slice paused"), "{e}");
    }

    #[test]
    fn sloped_plane_section_is_a_polygon() {
        let pl = sloped("z = x").unwrap().plane.unwrap();
        let sec = pl.section([-2.0; 3], [2.0; 3]);
        assert_eq!(sec.len(), 4, "z = x cuts the cube in a rectangle");
        for p in &sec {
            assert!(pl.dist(*p).abs() < 1e-9 && p.iter().all(|v| v.abs() <= 2.0 + 1e-9));
        }
        let far = sloped("z = x + 100").unwrap().plane.unwrap();
        assert!(far.section([-2.0; 3], [2.0; 3]).is_empty());
        let r = far.uv_ranges([-2.0; 3], [2.0; 3]);
        assert!(r[0].1 > r[0].0 && r[1].1 > r[1].0);
    }

}

