//! Geometry extraction for the graphing engine: adaptive curve sampling, implicit 2D contours
//! (quadtree + interval pruning + marching squares) and implicit 3D surfaces (octree + interval
//! pruning + marching tetrahedra). Everything here is pure CPU code with no side effects.
//!
//! Surfaces use marching tetrahedra (Kuhn split of every cube into six tetrahedra around the
//! main diagonal) instead of the 256-case marching-cubes tables. The split is identical in every
//! cube, so neighbouring cells always agree on face diagonals: the mesh has no cracks and no
//! ambiguous-case holes, and no hand-typed tables can be wrong.

use crate::compile::Program;
use crate::interval::Interval;

const MAX_EXPLICIT_DEPTH: u32 = 12;
const MAX_EXPLICIT_SAMPLES: usize = 200_000;

// ---------------------------------------------------------------------------------------------
// y = f(x)
// ---------------------------------------------------------------------------------------------

struct ExplicitSampler<'a> {
    p: &'a Program,
    st: Vec<f64>,
    ist: Vec<Interval>,
    tol: f64,
    span: f64,
    ylo: f64,
    yhi: f64,
    huge: f64,
    out: Vec<(f64, f64)>,
}

impl ExplicitSampler<'_> {
    fn f(&mut self, x: f64) -> f64 {
        let y = self.p.eval_with(&[x], &mut self.st);
        if y.is_finite() && y.abs() <= self.huge {
            y
        } else {
            f64::NAN
        }
    }

    fn defined_somewhere(&mut self, xa: f64, xb: f64) -> bool {
        !self.p.eval_interval_with(&[Interval::new(xa, xb)], &mut self.ist).is_empty()
    }

    /// Pushes the samples strictly between `xa` and `xb` (the caller pushes the endpoints).
    fn refine(&mut self, xa: f64, ya: f64, xb: f64, yb: f64, depth_left: u32) {
        if depth_left == 0 || self.out.len() >= MAX_EXPLICIT_SAMPLES {
            return;
        }
        let xm = 0.5 * (xa + xb);
        if xm <= xa || xm >= xb {
            return;
        }
        let ym = self.f(xm);
        let need = match (ya.is_finite(), yb.is_finite()) {
            (true, true) if ym.is_finite() => {
                let lo = ya.min(yb).min(ym);
                let hi = ya.max(yb).max(ym);
                if lo > self.yhi + self.span || hi < self.ylo - self.span {
                    false // far off-screen on one side: not worth detail
                } else {
                    (ym - 0.5 * (ya + yb)).abs() > self.tol || (yb - ya).abs() > self.span
                }
            }
            (true, true) => true,
            (false, false) => {
                ym.is_finite()
                    || (depth_left + 4 > MAX_EXPLICIT_DEPTH && self.defined_somewhere(xa, xb))
            }
            _ => true,
        };
        if !need {
            self.out.push((xm, ym));
            return;
        }
        self.refine(xa, ya, xm, ym, depth_left - 1);
        self.out.push((xm, ym));
        self.refine(xm, ym, xb, yb, depth_left - 1);
    }
}

/// Adaptively samples `y = f(x)` on `[x0, x1]` for a program with the single variable `x`.
///
/// Starts with about one sample per two pixels and subdivides wherever the midpoint deviates
/// from the chord by more than about a quarter pixel (the vertical pixel scale is estimated
/// from `width_px` and `y_range`), up to depth 12. The result is a list of polylines: a new one
/// starts at every undefined point (NaN), infinity, absurdly large value (more than ~1000
/// viewport heights away) and at every discontinuity (a sign-flipping jump of more than three
/// viewport heights, e.g. `tan(x)` or `1/x`), so asymptotes are never connected.
pub fn sample_explicit(
    p: &Program,
    x0: f64,
    x1: f64,
    width_px: usize,
    y_range: (f64, f64),
) -> Vec<Vec<[f64; 2]>> {
    if !(x0.is_finite() && x1.is_finite() && x1 > x0) {
        return Vec::new();
    }
    let (mut ylo, mut yhi) = y_range;
    if ylo > yhi {
        std::mem::swap(&mut ylo, &mut yhi);
    }
    let mut span = yhi - ylo;
    if !(span.is_finite() && span > 0.0) {
        span = 1.0;
        ylo = -0.5;
        yhi = 0.5;
    }
    let width_px = width_px.max(16);
    let height_px = (width_px as f64 * 0.6).max(8.0);
    let mut s = ExplicitSampler {
        p,
        st: Vec::with_capacity(16),
        ist: Vec::with_capacity(16),
        tol: 0.25 * span / height_px,
        span,
        ylo,
        yhi,
        huge: 1e3 * span + ylo.abs().max(yhi.abs()),
        out: Vec::new(),
    };
    let n0 = (width_px / 2).clamp(8, 4096);
    let dx = (x1 - x0) / n0 as f64;
    let mut xa = x0;
    let mut ya = s.f(xa);
    s.out.push((xa, ya));
    for i in 1..=n0 {
        let xb = if i == n0 { x1 } else { x0 + dx * i as f64 };
        let yb = s.f(xb);
        s.refine(xa, ya, xb, yb, MAX_EXPLICIT_DEPTH);
        s.out.push((xb, yb));
        xa = xb;
        ya = yb;
    }
    let mut lines: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    let flush = |cur: &mut Vec<[f64; 2]>, lines: &mut Vec<Vec<[f64; 2]>>| {
        if cur.len() >= 2 {
            lines.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for &(x, y) in &s.out {
        if !y.is_finite() {
            flush(&mut cur, &mut lines);
            continue;
        }
        if let Some(last) = cur.last() {
            let ly = last[1];
            if ((ly < 0.0) != (y < 0.0)) && (y - ly).abs() > 3.0 * span {
                flush(&mut cur, &mut lines);
            }
        }
        cur.push([x, y]);
    }
    flush(&mut cur, &mut lines);
    lines
}

// ---------------------------------------------------------------------------------------------
// parametric (x(t), y(t))
// ---------------------------------------------------------------------------------------------

struct ParamSampler<'a> {
    px: &'a Program,
    py: &'a Program,
    st: Vec<f64>,
    tol: f64,
    budget: usize,
    out: Vec<Option<[f64; 2]>>,
}

impl ParamSampler<'_> {
    fn f(&mut self, t: f64) -> Option<[f64; 2]> {
        let x = self.px.eval_with(&[t], &mut self.st);
        let y = self.py.eval_with(&[t], &mut self.st);
        (x.is_finite() && y.is_finite() && x.abs() < 1e15 && y.abs() < 1e15).then_some([x, y])
    }

    fn refine(&mut self, ta: f64, a: Option<[f64; 2]>, tb: f64, b: Option<[f64; 2]>, depth: u32) {
        if depth == 0 || self.budget == 0 {
            return;
        }
        let tm = 0.5 * (ta + tb);
        if tm <= ta || tm >= tb {
            return;
        }
        let m = self.f(tm);
        let need = match (a, b, m) {
            (Some(a), Some(b), Some(m)) => {
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len = dx.hypot(dy);
                let (mx, my) = (m[0] - a[0], m[1] - a[1]);
                let dev = if len > 0.0 { (mx * dy - my * dx).abs() / len } else { mx.hypot(my) };
                // also catch a midpoint that overshoots along the chord (cusps, folds)
                let along = if len > 0.0 { (mx * dx + my * dy) / len } else { 0.0 };
                dev > self.tol || along < -self.tol || along > len + self.tol
            }
            (Some(_), Some(_), None) => true,
            (None, None, None) => false,
            (None, None, Some(_)) => true,
            _ => true,
        };
        if !need {
            self.budget -= 1;
            self.out.push(m);
            return;
        }
        self.refine(ta, a, tm, m, depth - 1);
        self.budget = self.budget.saturating_sub(1);
        self.out.push(m);
        self.refine(tm, m, tb, b, depth - 1);
    }
}

/// Curvature-adaptive sampling of a parametric curve `(x(t), y(t))` on `[t0, t1]`; both programs
/// take the single variable `t`.
///
/// Starts from a uniform pass and bisects every interval whose midpoint is farther from its
/// chord than a tolerance proportional to the curve's extent, until `max_points` samples are
/// used (the budget is a soft limit). A new polyline starts wherever either coordinate is
/// undefined (NaN or infinite).
pub fn sample_parametric(
    px: &Program,
    py: &Program,
    t0: f64,
    t1: f64,
    max_points: usize,
) -> Vec<Vec<[f64; 2]>> {
    if !(t0.is_finite() && t1.is_finite() && t1 > t0) {
        return Vec::new();
    }
    let max_points = max_points.max(16);
    let n0 = (max_points / 4).clamp(16, 512);
    let mut s = ParamSampler { px, py, st: Vec::new(), tol: 0.0, budget: 0, out: Vec::new() };
    let ts: Vec<f64> =
        (0..=n0).map(|i| if i == n0 { t1 } else { t0 + (t1 - t0) * i as f64 / n0 as f64 }).collect();
    let pts: Vec<Option<[f64; 2]>> = ts.iter().map(|&t| s.f(t)).collect();
    let (mut xlo, mut xhi, mut ylo, mut yhi) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for q in pts.iter().flatten() {
        xlo = xlo.min(q[0]);
        xhi = xhi.max(q[0]);
        ylo = ylo.min(q[1]);
        yhi = yhi.max(q[1]);
    }
    let diag = if xlo <= xhi { (xhi - xlo).hypot(yhi - ylo) } else { 1.0 };
    s.tol = (diag * 5e-4).max(1e-300);
    s.budget = max_points.saturating_sub(n0 + 1);
    s.out.push(pts[0]);
    for i in 0..n0 {
        s.refine(ts[i], pts[i], ts[i + 1], pts[i + 1], 12);
        s.out.push(pts[i + 1]);
    }
    let mut lines = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    for q in s.out {
        match q {
            Some(q) => cur.push(q),
            None => {
                if cur.len() >= 2 {
                    lines.push(std::mem::take(&mut cur));
                }
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        lines.push(cur);
    }
    lines
}

// ---------------------------------------------------------------------------------------------
// f(x, y) = 0
// ---------------------------------------------------------------------------------------------

/// Zero set of `f(x, y) = 0` as line segments (program variables `[x, y]`).
///
/// A quadtree over the box is refined level by level; cells whose interval enclosure of `f`
/// excludes zero are discarded. Cells that still contain zero become leaves once their larger
/// side is at most `min_cell` or after `max_depth` splits, then marching squares runs on each
/// leaf, with edge crossings located by bisection and saddles resolved by the cell-centre
/// value. `max_cells` bounds the number of interval evaluations: when it is exhausted the
/// remaining cells become coarse leaves, so the result is coarser but never empty because of
/// the budget and the call never panics.
pub fn contour_2d(
    p: &Program,
    x: (f64, f64),
    y: (f64, f64),
    min_cell: f64,
    max_depth: u32,
    max_cells: usize,
) -> Vec<[[f64; 2]; 2]> {
    contour_2d_stats(p, x, y, min_cell, max_depth, max_cells).0
}

// The quadtree + marching squares implementation lives in `mesh_contour.rs`.
pub use crate::mesh_contour::contour_2d_stats;

// ---------------------------------------------------------------------------------------------
// f(x, y, z) = 0
// ---------------------------------------------------------------------------------------------

/// Indexed triangle mesh: per-vertex positions and unit normals, and a triangle index list
/// (three indices per triangle, counter-clockwise when seen from the side the normal points to).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

// The implicit-surface polygoniser lives in `mesh_surface.rs`.
pub use crate::mesh_height::height_surface;
pub use crate::mesh_surface::surface_3d;
#[cfg(test)]
use crate::mesh_surface::{cross, dot, sub, V3};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Angle};
    use crate::parse::parse;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    fn prog(src: &str, vars: &[&str]) -> Program {
        compile(&parse(src).unwrap(), vars, Angle::Rad).unwrap()
    }

    fn seg_len(s: &[[f64; 2]; 2]) -> f64 {
        (s[0][0] - s[1][0]).hypot(s[0][1] - s[1][1])
    }

    #[test]
    fn circle_contour() {
        let p = prog("x^2+y^2-4", &["x", "y"]);
        let segs = contour_2d(&p, (-3.0, 3.0), (-3.0, 3.0), 0.02, 20, 1_000_000);
        assert!(!segs.is_empty());
        let mut total = 0.0;
        for s in &segs {
            for e in s {
                let r = e[0].hypot(e[1]);
                assert!((r - 2.0).abs() < 2e-3, "radius {r}");
            }
            total += seg_len(s);
        }
        assert!((total - 4.0 * PI).abs() < 0.02, "length {total}");
    }

    #[test]
    fn hyperbola_and_odd_shapes() {
        let p = prog("x*y-1", &["x", "y"]);
        let segs = contour_2d(&p, (-4.0, 4.0), (-4.0, 4.0), 0.02, 20, 1_000_000);
        assert!(!segs.is_empty());
        for s in &segs {
            for e in s {
                assert!((e[0] * e[1] - 1.0).abs() < 1e-3, "{e:?}");
            }
        }
        // both branches present
        assert!(segs.iter().any(|s| s[0][0] > 0.0) && segs.iter().any(|s| s[0][0] < 0.0));
        // saddle-y shape: x^2 - y^2 = 0 (two crossing lines)
        let p = prog("x^2-y^2", &["x", "y"]);
        let segs = contour_2d(&p, (-2.0, 2.0), (-2.0, 2.0), 0.02, 20, 1_000_000);
        assert!(!segs.is_empty());
        for s in &segs {
            for e in s {
                assert!((e[0].abs() - e[1].abs()).abs() < 0.03, "{e:?}");
            }
        }
        // undefined region: sqrt(x) - y has no points for x < 0
        let p = prog("sqrt(x)-y", &["x", "y"]);
        let segs = contour_2d(&p, (-2.0, 2.0), (-2.0, 2.0), 0.02, 20, 1_000_000);
        assert!(!segs.is_empty());
        for s in &segs {
            for e in s {
                assert!(e[0] > -0.03 && (e[1] - e[0].max(0.0).sqrt()).abs() < 0.05, "{e:?}");
            }
        }
    }

    #[test]
    fn thin_features_not_lost() {
        let p = prog("(x^2+y^2-1)*(x^2+y^2-1.01)", &["x", "y"]);
        let segs = contour_2d(&p, (-2.0, 2.0), (-2.0, 2.0), 0.001, 20, 5_000_000);
        let (mut inner, mut outer) = (0usize, 0usize);
        for s in &segs {
            let r = s[0][0].hypot(s[0][1]);
            if (r - 1.0).abs() < 2e-3 {
                inner += 1;
            }
            if (r - 1.005).abs() < 2e-3 {
                outer += 1;
            }
        }
        assert!(inner > 100 && outer > 100, "{inner} {outer}");
        let p = prog("y-0.0001*x", &["x", "y"]);
        let segs = contour_2d(&p, (-5.0, 5.0), (-5.0, 5.0), 0.05, 20, 1_000_000);
        assert!(segs.len() >= 100);
        let total: f64 = segs.iter().map(seg_len).sum();
        assert!((total - 10.0).abs() < 0.01, "{total}");
    }

    #[test]
    fn interval_pruning_prunes() {
        let p = prog("x^2+y^2-4", &["x", "y"]);
        let cell = 6.0 / 1024.0;
        let (segs, evals) = contour_2d_stats(&p, (-3.0, 3.0), (-3.0, 3.0), cell, 30, 10_000_000);
        assert!(!segs.is_empty());
        let full = 1024usize * 1024;
        assert!(evals * 20 < full, "evaluated {evals} cells vs {full}");
    }

    #[test]
    fn budget_respected() {
        let p = prog("x^2+y^2-4", &["x", "y"]);
        let (segs, evals) = contour_2d_stats(&p, (-3.0, 3.0), (-3.0, 3.0), 0.001, 30, 300);
        assert!(evals <= 300);
        assert!(!segs.is_empty());
        let (segs, _) = contour_2d_stats(&p, (-3.0, 3.0), (-3.0, 3.0), 0.001, 30, 0);
        let _ = segs; // must not panic
        let s = prog("x^2+y^2+z^2-4", &["x", "y", "z"]);
        let m = surface_3d(&s, [-3.0; 3], [3.0; 3], 10, 2000);
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn empty_results() {
        let p = prog("x^2+y^2+1", &["x", "y"]);
        assert!(contour_2d(&p, (-3.0, 3.0), (-3.0, 3.0), 0.01, 20, 100_000).is_empty());
        let s = prog("x^2+y^2+z^2+1", &["x", "y", "z"]);
        let m = surface_3d(&s, [-3.0; 3], [3.0; 3], 6, 100_000);
        assert!(m.positions.is_empty() && m.indices.is_empty());
    }

    fn area(m: &Mesh) -> f64 {
        let mut a = 0.0;
        for t in m.indices.chunks(3) {
            let q: Vec<V3> = t
                .iter()
                .map(|&i| {
                    let p = m.positions[i as usize];
                    [p[0] as f64, p[1] as f64, p[2] as f64]
                })
                .collect();
            let c = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            a += 0.5 * dot(c, c).sqrt();
        }
        a
    }

    fn check_sphere(m: &Mesh, closed: bool) {
        assert!(!m.indices.is_empty());
        for (p, n) in m.positions.iter().zip(&m.normals) {
            let r = (p[0] as f64).hypot(p[1] as f64).hypot(p[2] as f64);
            assert!((r - 2.0).abs() < 2e-3, "radius {r}");
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-3);
            let d = (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) / 2.0;
            assert!(d > 0.99, "normal not outward: {d}");
        }
        let a = area(m);
        assert!((a - 16.0 * PI).abs() / (16.0 * PI) < 0.04, "area {a}");
        // winding agrees with outward normals
        for t in m.indices.chunks(3) {
            let q: Vec<V3> = t
                .iter()
                .map(|&i| {
                    let p = m.positions[i as usize];
                    [p[0] as f64, p[1] as f64, p[2] as f64]
                })
                .collect();
            let c = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            assert!(dot(c, q[0]) > 0.0);
        }
        let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for t in m.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let e = edges.entry((a.min(b), a.max(b))).or_default();
                if a < b {
                    e.0 += 1
                } else {
                    e.1 += 1
                }
            }
        }
        if closed {
            for (e, c) in &edges {
                assert_eq!(*c, (1, 1), "edge {e:?} not shared by exactly two opposite faces");
            }
        }
    }

    #[test]
    fn sphere_mesh() {
        let s = prog("x^2+y^2+z^2-4", &["x", "y", "z"]);
        // box whose grid corners avoid the sphere exactly
        let m = surface_3d(&s, [-3.013, -2.987, -3.031], [2.977, 3.019, 2.969], 6, 1_000_000);
        check_sphere(&m, true);
        // symmetric box: grid corners lie exactly on the surface
        let m = surface_3d(&s, [-3.0; 3], [3.0; 3], 6, 1_000_000);
        check_sphere(&m, false);
        // the box where a naive corner sampler sees only one sign
        let m = surface_3d(&s, [-2.5; 3], [2.5; 3], 5, 1_000_000);
        check_sphere(&m, false);
    }

    #[test]
    fn plane_mesh() {
        let s = prog("z-x", &["x", "y", "z"]);
        let m = surface_3d(&s, [-1.0; 3], [1.0; 3], 4, 100_000);
        assert!(!m.indices.is_empty());
        for p in &m.positions {
            assert!((p[2] - p[0]).abs() < 1e-4);
        }
        for n in &m.normals {
            // gradient of z-x is (-1,0,1)/sqrt2
            assert!((n[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3 && (n[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        }
        // area of the plane inside the cube: 2 * 2*sqrt(2)
        let a = area(&m);
        assert!((a - 4.0 * 2f64.sqrt()).abs() < 0.02, "{a}");
    }

    #[test]
    fn explicit_tan() {
        let p = prog("tan(x)", &["x"]);
        let lines = sample_explicit(&p, -5.0, 5.0, 800, (-5.0, 5.0));
        assert!(lines.len() >= 3, "{}", lines.len());
        for l in &lines {
            for w in l.windows(2) {
                let k0 = ((w[0][0] - PI / 2.0) / PI).floor();
                let k1 = ((w[1][0] - PI / 2.0) / PI).floor();
                assert_eq!(k0, k1, "segment spans an asymptote: {w:?}");
            }
            for q in l {
                assert!((q[1] - q[0].tan()).abs() <= 1e-6 * q[1].abs().max(1.0));
            }
        }
    }

    #[test]
    fn explicit_inverse() {
        let p = prog("1/x", &["x"]);
        let lines = sample_explicit(&p, -3.0, 3.0, 600, (-4.0, 4.0));
        assert!(lines.len() >= 2);
        for l in &lines {
            let s = l[0][0] < 0.0;
            assert!(l.iter().all(|q| (q[0] < 0.0) == s));
        }
    }

    #[test]
    fn explicit_parabola_and_sqrt() {
        let p = prog("x^2", &["x"]);
        let lines = sample_explicit(&p, -3.0, 3.0, 600, (-1.0, 9.0));
        assert_eq!(lines.len(), 1);
        for q in &lines[0] {
            assert!((q[1] - q[0] * q[0]).abs() < 1e-12);
        }
        // adaptive: fewer than a brute-force per-pixel count blow-up, but at least n0 samples
        assert!(lines[0].len() >= 300);
        let p = prog("sqrt(x)", &["x"]);
        let lines = sample_explicit(&p, -1.0, 4.0, 500, (-2.0, 3.0));
        assert!(!lines.is_empty());
        for l in &lines {
            for q in l {
                assert!(q[0] >= 0.0, "{q:?}");
                assert!((q[1] - q[0].sqrt()).abs() < 1e-12);
            }
        }
        // reaches (close to) the domain edge and the right end
        let first = lines[0][0][0];
        assert!(first < 1e-3, "{first}");
        assert!((lines.last().unwrap().last().unwrap()[0] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn parametric_circle() {
        let (px, py) = (prog("cos(t)", &["t"]), prog("sin(t)", &["t"]));
        let lines = sample_parametric(&px, &py, 0.0, 2.0 * PI, 400);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].len() <= 450);
        for q in &lines[0] {
            assert!((q[0].hypot(q[1]) - 1.0).abs() < 1e-12);
        }
        // sqrt(t) breaks where undefined
        let (px, py) = (prog("sqrt(t)", &["t"]), prog("t", &["t"]));
        let lines = sample_parametric(&px, &py, -1.0, 1.0, 200);
        assert!(lines.iter().flatten().all(|q| q[1] >= -1e-9));
    }

    #[test]
    fn timing_sanity() {
        let s = prog("x^2+y^2+z^2-4", &["x", "y", "z"]);
        let t = std::time::Instant::now();
        let m = surface_3d(&s, [-3.0; 3], [3.0; 3], 7, 1_000_000);
        eprintln!("sphere depth 7: {} tris in {:?}", m.indices.len() / 3, t.elapsed());
        let c = prog("x^2+y^2-4", &["x", "y"]);
        let t = std::time::Instant::now();
        let (segs, ev) = contour_2d_stats(&c, (-3.0, 3.0), (-3.0, 3.0), 0.005, 30, 1_000_000);
        eprintln!("circle: {} segs, {ev} cells in {:?}", segs.len(), t.elapsed());
    }
}
