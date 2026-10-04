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
use std::collections::HashMap;

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

/// Like [`contour_2d`], additionally returning the number of interval-evaluated cells.
pub fn contour_2d_stats(
    p: &Program,
    x: (f64, f64),
    y: (f64, f64),
    min_cell: f64,
    max_depth: u32,
    max_cells: usize,
) -> (Vec<[[f64; 2]; 2]>, usize) {
    let mut segs = Vec::new();
    if !(x.0.is_finite() && x.1.is_finite() && y.0.is_finite() && y.1.is_finite())
        || x.1 <= x.0
        || y.1 <= y.0
    {
        return (segs, 0);
    }
    let max_depth = max_depth.min(30);
    let (w0, h0) = (x.1 - x.0, y.1 - y.0);
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let mut evals = 0usize;
    let mut leaves: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut cur = vec![(x.0, y.0, x.1, y.1)];
    let mut depth = 0u32;
    while !cur.is_empty() {
        let mut next = Vec::new();
        for &(ax, ay, bx, by) in &cur {
            if evals >= max_cells {
                leaves.push((ax, ay, bx, by));
                continue;
            }
            evals += 1;
            let r = p.eval_interval_with(&[Interval::new(ax, bx), Interval::new(ay, by)], &mut ist);
            if r.is_empty() || !r.contains_zero() {
                continue;
            }
            if (bx - ax).max(by - ay) <= min_cell || depth >= max_depth || evals >= max_cells {
                leaves.push((ax, ay, bx, by));
                continue;
            }
            let (mx, my) = (0.5 * (ax + bx), 0.5 * (ay + by));
            next.push((ax, ay, mx, my));
            next.push((mx, ay, bx, my));
            next.push((ax, my, mx, by));
            next.push((mx, my, bx, by));
        }
        cur = next;
        depth += 1;
    }
    let _ = (w0, h0);
    let mut st: Vec<f64> = Vec::with_capacity(16);
    for (ax, ay, bx, by) in leaves {
        march_square(p, &mut st, ax, ay, bx, by, &mut segs);
    }
    (segs, evals)
}

fn march_square(
    p: &Program,
    st: &mut Vec<f64>,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    out: &mut Vec<[[f64; 2]; 2]>,
) {
    // corners: c0 (x0,y0), c1 (x1,y0), c2 (x1,y1), c3 (x0,y1)
    let pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let mut v = [0.0; 4];
    for i in 0..4 {
        v[i] = p.eval_with(&pts[i], st);
        if v[i].is_nan() {
            return;
        }
    }
    let mut code = 0u8;
    for (i, &vi) in v.iter().enumerate() {
        if vi > 0.0 {
            code |= 1 << i;
        }
    }
    if code == 0 || code == 15 {
        return;
    }
    // edges: e0 c0-c1, e1 c1-c2, e2 c2-c3, e3 c3-c0
    let ends = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
    let root = |e: usize, st: &mut Vec<f64>| -> [f64; 2] {
        let (a, b) = ends[e];
        let (pa, pb) = (pts[a], pts[b]);
        let pos_a = v[a] > 0.0;
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..14 {
            let m = 0.5 * (lo + hi);
            let q = [pa[0] + (pb[0] - pa[0]) * m, pa[1] + (pb[1] - pa[1]) * m];
            let fv = p.eval_with(&q, st);
            if fv.is_nan() || (fv > 0.0) == pos_a {
                lo = m;
            } else {
                hi = m;
            }
        }
        let t = 0.5 * (lo + hi);
        [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t]
    };
    let pairs: &[(usize, usize)] = match code {
        1 | 14 => &[(3, 0)],
        2 | 13 => &[(0, 1)],
        3 | 12 => &[(3, 1)],
        4 | 11 => &[(1, 2)],
        6 | 9 => &[(0, 2)],
        7 | 8 => &[(2, 3)],
        5 | 10 => {
            let c = p.eval_with(&[0.5 * (x0 + x1), 0.5 * (y0 + y1)], st);
            let centre_pos = c > 0.0;
            if (code == 5) == centre_pos {
                &[(0, 1), (2, 3)]
            } else {
                &[(3, 0), (1, 2)]
            }
        }
        _ => return,
    };
    for &(ea, eb) in pairs {
        let a = root(ea, st);
        let b = root(eb, st);
        out.push([a, b]);
    }
}

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

type V3 = [f64; 3];

struct SurfaceBuilder<'a> {
    p: &'a Program,
    st: Vec<f64>,
    min: V3,
    size: V3,
    grid: f64, // cells per axis at the finest level
    h: f64,    // gradient step
    vals: HashMap<u64, f64>,
    edges: HashMap<(u64, u64), u32>,
    quant: HashMap<[i64; 3], u32>,
    qscale: f64,
    pos: Vec<V3>,
    nrm: Vec<V3>,
    facen: Vec<V3>,
    idx: Vec<u32>,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: V3) -> Option<V3> {
    let l = dot(a, a).sqrt();
    (l.is_finite() && l > 1e-300).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

impl SurfaceBuilder<'_> {
    fn coord(&self, ix: [u64; 3]) -> V3 {
        [
            self.min[0] + self.size[0] * ix[0] as f64 / self.grid,
            self.min[1] + self.size[1] * ix[1] as f64 / self.grid,
            self.min[2] + self.size[2] * ix[2] as f64 / self.grid,
        ]
    }

    fn key(ix: [u64; 3]) -> u64 {
        ix[0] | (ix[1] << 21) | (ix[2] << 42)
    }

    fn f(&mut self, q: V3) -> f64 {
        self.p.eval_with(&q, &mut self.st)
    }

    fn corner(&mut self, ix: [u64; 3]) -> f64 {
        let k = Self::key(ix);
        if let Some(&v) = self.vals.get(&k) {
            return v;
        }
        let v = self.f(self.coord(ix));
        self.vals.insert(k, v);
        v
    }

    fn gradient(&mut self, q: V3) -> Option<V3> {
        let h = self.h;
        let mut g = [0.0; 3];
        for a in 0..3 {
            let (mut u, mut d) = (q, q);
            u[a] += h;
            d[a] -= h;
            g[a] = (self.f(u) - self.f(d)) / (2.0 * h);
        }
        normalize(g)
    }

    /// Vertex on the edge between corners `a` and `b` (which have opposite signs).
    fn edge_vertex(&mut self, a: ([u64; 3], f64), b: ([u64; 3], f64)) -> u32 {
        let (ka, kb) = (Self::key(a.0), Self::key(b.0));
        let ek = if ka < kb { (ka, kb) } else { (kb, ka) };
        if let Some(&i) = self.edges.get(&ek) {
            return i;
        }
        let (pa, pb) = (self.coord(a.0), self.coord(b.0));
        let pos_a = a.1 > 0.0;
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..16 {
            let m = 0.5 * (lo + hi);
            let q = [
                pa[0] + (pb[0] - pa[0]) * m,
                pa[1] + (pb[1] - pa[1]) * m,
                pa[2] + (pb[2] - pa[2]) * m,
            ];
            let fv = self.f(q);
            if fv.is_nan() || (fv > 0.0) == pos_a {
                lo = m;
            } else {
                hi = m;
            }
        }
        let t = 0.5 * (lo + hi);
        let q = [
            pa[0] + (pb[0] - pa[0]) * t,
            pa[1] + (pb[1] - pa[1]) * t,
            pa[2] + (pb[2] - pa[2]) * t,
        ];
        let qk = [
            (q[0] / self.qscale).round() as i64,
            (q[1] / self.qscale).round() as i64,
            (q[2] / self.qscale).round() as i64,
        ];
        let id = if let Some(&i) = self.quant.get(&qk) {
            i
        } else {
            let i = self.pos.len() as u32;
            self.pos.push(q);
            let n = self.gradient(q).unwrap_or([0.0; 3]);
            self.nrm.push(n);
            self.facen.push([0.0; 3]);
            self.quant.insert(qk, i);
            i
        };
        self.edges.insert(ek, id);
        id
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        if a == b || b == c || a == c {
            return;
        }
        let (pa, pb, pc) = (self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]);
        let fnrm = cross(sub(pb, pa), sub(pc, pa));
        let g = [
            self.nrm[a as usize][0] + self.nrm[b as usize][0] + self.nrm[c as usize][0],
            self.nrm[a as usize][1] + self.nrm[b as usize][1] + self.nrm[c as usize][1],
            self.nrm[a as usize][2] + self.nrm[b as usize][2] + self.nrm[c as usize][2],
        ];
        let (b, c) = if dot(fnrm, g) < 0.0 { (c, b) } else { (b, c) };
        let sgn = if dot(fnrm, g) < 0.0 { -1.0 } else { 1.0 };
        for &i in &[a, b, c] {
            let f = &mut self.facen[i as usize];
            for k in 0..3 {
                f[k] += sgn * fnrm[k];
            }
        }
        self.idx.extend_from_slice(&[a, b, c]);
    }

    fn march_cell(&mut self, base: [u64; 3], step: u64) {
        let mut cs = [([0u64; 3], 0.0f64); 8];
        for (c, slot) in cs.iter_mut().enumerate() {
            let ix = [
                base[0] + (c as u64 & 1) * step,
                base[1] + ((c as u64 >> 1) & 1) * step,
                base[2] + ((c as u64 >> 2) & 1) * step,
            ];
            *slot = (ix, 0.0);
        }
        for slot in cs.iter_mut() {
            slot.1 = self.corner(slot.0);
            if slot.1.is_nan() {
                return;
            }
        }
        const AXES: [[usize; 3]; 6] =
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        for ax in AXES {
            let c0 = 0usize;
            let c1 = 1usize << ax[0];
            let c2 = c1 | (1usize << ax[1]);
            let tet = [cs[c0], cs[c1], cs[c2], cs[7]];
            let (mut pos, mut neg) = (Vec::with_capacity(4), Vec::with_capacity(4));
            for t in tet {
                if t.1 > 0.0 {
                    pos.push(t)
                } else {
                    neg.push(t)
                }
            }
            match (pos.len(), neg.len()) {
                (1, 3) | (3, 1) => {
                    let (lone, others) = if pos.len() == 1 { (pos[0], &neg) } else { (neg[0], &pos) };
                    let v: Vec<u32> = others.iter().map(|&o| self.edge_vertex(lone, o)).collect();
                    self.triangle(v[0], v[1], v[2]);
                }
                (2, 2) => {
                    let ac = self.edge_vertex(pos[0], neg[0]);
                    let ad = self.edge_vertex(pos[0], neg[1]);
                    let bd = self.edge_vertex(pos[1], neg[1]);
                    let bc = self.edge_vertex(pos[1], neg[0]);
                    self.triangle(ac, ad, bd);
                    self.triangle(ac, bd, bc);
                }
                _ => {}
            }
        }
    }
}

/// Zero set of `f(x, y, z) = 0` as an indexed triangle mesh (program variables `[x, y, z]`).
///
/// An octree over `[min, max]` is refined level by level; cells whose interval enclosure of `f`
/// excludes zero are discarded, the rest become leaves at `max_depth` (at most 20; the grid is
/// `2^max_depth` cells per axis). Leaves are polygonised with marching tetrahedra (see the module
/// docs), vertices are placed by bisection on the cell edges and shared between triangles, and
/// normals are the normalised gradient of `f` (central differences), so they point towards
/// increasing `f`; triangles are wound consistently with them. `max_cells` bounds the number of
/// interval evaluations; once it is exhausted the remaining cells become coarse leaves, so the
/// mesh degrades gracefully instead of failing.
pub fn surface_3d(p: &Program, min: [f64; 3], max: [f64; 3], max_depth: u32, max_cells: usize) -> Mesh {
    let ok = (0..3).all(|a| min[a].is_finite() && max[a].is_finite() && max[a] > min[a]);
    if !ok {
        return Mesh::default();
    }
    let depth_max = max_depth.min(20);
    let grid = (1u64 << depth_max) as f64;
    let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let cell = size[0].max(size[1]).max(size[2]) / grid;
    let mut b = SurfaceBuilder {
        p,
        st: Vec::with_capacity(16),
        min,
        size,
        grid,
        h: cell * 0.25,
        vals: HashMap::new(),
        edges: HashMap::new(),
        quant: HashMap::new(),
        qscale: cell * 1e-6,
        pos: Vec::new(),
        nrm: Vec::new(),
        facen: Vec::new(),
        idx: Vec::new(),
    };
    // octree, breadth first; cells are (ix, iy, iz) at the current depth
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let mut evals = 0usize;
    let mut leaves: Vec<([u64; 3], u32)> = Vec::new();
    let mut cur: Vec<[u64; 3]> = vec![[0, 0, 0]];
    let mut depth = 0u32;
    while !cur.is_empty() {
        let mut next = Vec::new();
        let n = (1u64 << depth) as f64;
        for &c in &cur {
            if evals >= max_cells {
                leaves.push((c, depth));
                continue;
            }
            evals += 1;
            let mut bx = [Interval::EMPTY; 3];
            for a in 0..3 {
                bx[a] = Interval::new(
                    min[a] + size[a] * c[a] as f64 / n,
                    min[a] + size[a] * (c[a] + 1) as f64 / n,
                );
            }
            let r = p.eval_interval_with(&bx, &mut ist);
            if r.is_empty() || !r.contains_zero() {
                continue;
            }
            if depth >= depth_max || evals >= max_cells {
                leaves.push((c, depth));
                continue;
            }
            for k in 0..8u64 {
                next.push([c[0] * 2 + (k & 1), c[1] * 2 + ((k >> 1) & 1), c[2] * 2 + ((k >> 2) & 1)]);
            }
        }
        cur = next;
        depth += 1;
    }
    for (c, d) in leaves {
        let step = 1u64 << (depth_max - d);
        b.march_cell([c[0] * step, c[1] * step, c[2] * step], step);
    }
    // finalise normals: fall back to the accumulated face normal where the gradient vanished
    let mut mesh = Mesh::default();
    for i in 0..b.pos.len() {
        let n = if dot(b.nrm[i], b.nrm[i]) > 0.5 {
            b.nrm[i]
        } else {
            normalize(b.facen[i]).unwrap_or([0.0, 0.0, 1.0])
        };
        mesh.positions.push([b.pos[i][0] as f32, b.pos[i][1] as f32, b.pos[i][2] as f32]);
        mesh.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
    }
    mesh.indices = b.idx;
    mesh
}

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
