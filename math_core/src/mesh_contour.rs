//! Implicit 2D contours `F(x, y) = 0`: quadtree + interval pruning + marching squares.
//!
//! Corner values are evaluated once and shared by the cells around them, edge crossings are found
//! with a bracketed secant iteration and shared between the two cells of an edge, and a sign
//! change that is really a pole or a jump (`y tan(x) = 1`-style graphs) draws nothing.
//!
//! Where `F` touches zero without changing sign (point conics `x^2 + y^2 = 0`, double lines
//! `(x - y)^2 = 0`) marching squares sees nothing. A "touch pass" over the cells with a single
//! sign draws those: the zero set of `w . grad F` (which does change sign across a double root),
//! restricted to places where `F` is zero, gives double curves, and a damped Newton search finds
//! isolated minima, drawn as a zero-length segment (a dot).

use crate::compile::Program;
use crate::interval::Interval;
use crate::mesh_touch::{newton_min, FxMap};

type Seg = [[f64; 2]; 2];

/// Direction for `w . grad F`; not parallel to an axis or a diagonal.
const W: [f64; 2] = [0.8137, 0.5812];
const NO_EDGE: [f64; 2] = [f64::NAN, f64::NAN];

#[inline]
fn key(ix: u64, iy: u64) -> u64 {
    ix | (iy << 32)
}

struct Grid<'a> {
    p: &'a Program,
    st: Vec<f64>,
    x0: f64,
    y0: f64,
    w: f64,
    h: f64,
    lat: f64,
    cell: f64,
    vals: FxMap<u64, f64>,
    gvals: FxMap<u64, f64>,
    edges: FxMap<(u64, u64), [f64; 2]>,
    gmode: bool,
    tau: f64,
}

impl Grid<'_> {
    fn coord(&self, ix: u64, iy: u64) -> [f64; 2] {
        [self.x0 + self.w * ix as f64 / self.lat, self.y0 + self.h * iy as f64 / self.lat]
    }

    #[inline]
    fn fraw(&mut self, q: [f64; 2]) -> f64 {
        self.p.eval_with(&q, &mut self.st)
    }

    fn f(&mut self, q: [f64; 2]) -> f64 {
        if !self.gmode {
            return self.fraw(q);
        }
        let e = self.cell * 1e-3;
        let a = self.fraw([q[0] + e * W[0], q[1] + e * W[1]]);
        let b = self.fraw([q[0] - e * W[0], q[1] - e * W[1]]);
        (a - b) / (2.0 * e)
    }

    fn corner(&mut self, ix: u64, iy: u64) -> f64 {
        let k = key(ix, iy);
        if let Some(&v) = self.vals.get(&k) {
            return v;
        }
        let v = self.fraw(self.coord(ix, iy));
        self.vals.insert(k, v);
        v
    }

    fn gcorner(&mut self, ix: u64, iy: u64) -> f64 {
        let k = key(ix, iy);
        if let Some(&v) = self.gvals.get(&k) {
            return v;
        }
        let v = self.f(self.coord(ix, iy));
        self.gvals.insert(k, v);
        v
    }

    /// Root of the contoured function between `pa` and `pb` (values `fa`, `fb` in different
    /// classes), or `None` for a pole / jump.
    fn root(&mut self, pa: [f64; 2], pb: [f64; 2], fa: f64, fb: f64) -> Option<[f64; 2]> {
        let lerp = |t: f64| [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t];
        if fa == 0.0 {
            return Some(pa);
        }
        if fb == 0.0 {
            return Some(pb);
        }
        let mut scale = fa.abs().max(fb.abs());
        let mut bisect = false;
        if !scale.is_finite() {
            scale = if fa.is_finite() { fa.abs() } else if fb.is_finite() { fb.abs() } else { 1.0 };
            bisect = true;
        }
        let class_a = fa > 0.0;
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let (mut f0, mut f1) = (fa, fb);
        let (mut g0, mut g1) = (fa, fb);
        let mut side = 0i8;
        let mut tm = 0.5;
        let mut fm = f64::NAN;
        let mut by_value = false;
        for _ in 0..40 {
            tm = if bisect {
                0.5 * (t0 + t1)
            } else {
                let t = t0 + (t1 - t0) * g0 / (g0 - g1);
                if t.is_finite() {
                    t.clamp(t0 + 1e-9 * (t1 - t0), t1 - 1e-9 * (t1 - t0))
                } else {
                    0.5 * (t0 + t1)
                }
            };
            fm = self.f(lerp(tm));
            if fm.is_nan() {
                bisect = true;
                t0 = tm;
                if t1 - t0 < 1e-7 {
                    break;
                }
                continue;
            }
            if fm == 0.0 || fm.abs() <= 1e-6 * scale {
                by_value = true;
                break;
            }
            if (fm > 0.0) == class_a {
                t0 = tm;
                f0 = fm;
                if side == 1 {
                    g1 *= 0.5;
                }
                g0 = fm;
                side = 1;
            } else {
                t1 = tm;
                f1 = fm;
                if side == -1 {
                    g0 *= 0.5;
                }
                g1 = fm;
                side = -1;
            }
            if t1 - t0 < 1e-7 {
                break;
            }
        }
        let fres = if by_value { fm.abs() } else { f0.abs().min(f1.abs()) };
        if !fm.is_nan() && fres > 0.05 * scale {
            return None;
        }
        if fm.is_nan() {
            tm = 0.5 * (t0 + t1);
        }
        Some(lerp(tm))
    }

    /// Crossing on the cell edge between lattice corners `a` and `b`, shared between cells.
    fn edge(&mut self, a: (u64, u64), b: (u64, u64), fa: f64, fb: f64) -> [f64; 2] {
        let (ka, kb) = (key(a.0, a.1), key(b.0, b.1));
        let ek = if ka < kb { (ka, kb) } else { (kb, ka) };
        if let Some(&r) = self.edges.get(&ek) {
            return r;
        }
        let pa = self.coord(a.0, a.1);
        let pb = self.coord(b.0, b.1);
        let r = match self.root(pa, pb, fa, fb) {
            None => NO_EDGE,
            Some(q) => {
                if self.gmode {
                    let fv = self.fraw(q);
                    if fv.abs() <= self.tau {
                        q
                    } else {
                        NO_EDGE
                    }
                } else {
                    q
                }
            }
        };
        self.edges.insert(ek, r);
        r
    }

    /// Marching squares on one cell with corner values `v` (c0 (x0,y0), c1 (x1,y0), c2 (x1,y1),
    /// c3 (x0,y1)); `centre` is evaluated lazily for the saddle cases.
    fn march(&mut self, ix: u64, iy: u64, step: u64, v: &[f64; 4], out: &mut Vec<Seg>) {
        let mut code = 0u8;
        for (i, &vi) in v.iter().enumerate() {
            if vi > 0.0 {
                code |= 1 << i;
            }
        }
        if code == 0 || code == 15 {
            return;
        }
        let c = [(ix, iy), (ix + step, iy), (ix + step, iy + step), (ix, iy + step)];
        let ends = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
        let pairs: &[(usize, usize)] = match code {
            1 | 14 => &[(3, 0)],
            2 | 13 => &[(0, 1)],
            3 | 12 => &[(3, 1)],
            4 | 11 => &[(1, 2)],
            6 | 9 => &[(0, 2)],
            7 | 8 => &[(2, 3)],
            5 | 10 => {
                let (p0, p2) = (self.coord(c[0].0, c[0].1), self.coord(c[2].0, c[2].1));
                let cf = self.f([0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1])]);
                if (code == 5) == (cf > 0.0) {
                    &[(0, 1), (2, 3)]
                } else {
                    &[(3, 0), (1, 2)]
                }
            }
            _ => return,
        };
        for &(ea, eb) in pairs {
            let (a0, a1) = ends[ea];
            let (b0, b1) = ends[eb];
            let pa = self.edge(c[a0], c[a1], v[a0], v[a1]);
            let pb = self.edge(c[b0], c[b1], v[b0], v[b1]);
            if pa[0].is_nan() || pb[0].is_nan() || pa == pb {
                continue; // a pole/jump, or both crossings are the same lattice corner
            }
            out.push([pa, pb]);
        }
    }

    // ---- touch pass ------------------------------------------------------------------------

    fn lattice_val(&mut self, i: [i64; 2]) -> Option<f64> {
        let g = self.lat as i64;
        if i[0] < 0 || i[1] < 0 || i[0] > g || i[1] > g {
            return None;
        }
        Some(self.corner(i[0] as u64, i[1] as u64))
    }

    fn has_opposite_nearby(&mut self, ix: u64, iy: u64, step: u64, pos_class: bool) -> bool {
        let s = step as i64;
        let g = self.lat as i64;
        for pass in 0..2 {
            for dy in -1..=2i64 {
                for dx in -1..=2i64 {
                    let i = [ix as i64 + dx * s, iy as i64 + dy * s];
                    if i[0] < 0 || i[1] < 0 || i[0] > g || i[1] > g {
                        continue;
                    }
                    let v = if pass == 0 {
                        match self.vals.get(&key(i[0] as u64, i[1] as u64)) {
                            Some(&v) => v,
                            None => continue,
                        }
                    } else {
                        match self.lattice_val(i) {
                            Some(v) => v,
                            None => continue,
                        }
                    };
                    if !v.is_nan() && (if pos_class { v < 0.0 } else { v > 0.0 }) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn local_min_corner(&mut self, ix: u64, iy: u64, step: u64, s: f64) -> Option<(u64, u64)> {
        let st = step as i64;
        let mut best: Option<((u64, u64), f64)> = None;
        for (cx, cy) in [(ix, iy), (ix + step, iy), (ix, iy + step), (ix + step, iy + step)] {
            let v = s * self.corner(cx, cy);
            if best.map_or(true, |(_, bv)| v < bv) {
                best = Some(((cx, cy), v));
            }
        }
        let ((bx, by), v) = best?;
        let tol = 1e-9 * v.abs();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if let Some(nv) = self.lattice_val([bx as i64 + dx * st, by as i64 + dy * st]) {
                    if !nv.is_nan() && s * nv < v - tol {
                        return None;
                    }
                }
            }
        }
        Some((bx, by))
    }

    fn touch_pass(&mut self, cands: &[(u64, u64, u64, [f64; 4])], out: &mut Vec<Seg>) {
        let mut touching: Vec<(u64, u64, u64, [f64; 4])> = Vec::new();
        for &(ix, iy, step, v) in cands {
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            if !self.has_opposite_nearby(ix, iy, step, v.iter().any(|&x| x > 0.0)) {
                touching.push((ix, iy, step, v));
            }
        }
        if touching.is_empty() {
            return;
        }
        // double curves: contour of G = W . grad F, kept where F ~ 0
        self.gmode = true;
        self.edges.clear();
        for &(ix, iy, step, v) in &touching {
            let gv = [
                self.gcorner(ix, iy),
                self.gcorner(ix + step, iy),
                self.gcorner(ix + step, iy + step),
                self.gcorner(ix, iy + step),
            ];
            if gv.iter().any(|x| x.is_nan()) {
                continue;
            }
            let cmax = v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            self.tau = 1e-4 * cmax;
            self.march(ix, iy, step, &gv, out);
        }
        self.gmode = false;
        // isolated minima: dots
        let mut tried: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut dots: Vec<[f64; 2]> = Vec::new();
        let cell = self.cell;
        for &(ix, iy, step, v) in &touching {
            let s = if v.iter().any(|&x| x > 0.0) { 1.0 } else { -1.0 };
            let Some((cx, cy)) = self.local_min_corner(ix, iy, step, s) else { continue };
            if !tried.insert(key(cx, cy)) {
                continue;
            }
            let cmax = v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let x0 = self.coord(cx, cy);
            let done = 1e-9 * cmax;
            let m = {
                let p = self.p;
                let mut st: Vec<f64> = Vec::with_capacity(16);
                let mut f = |x: &[f64; 2]| s * p.eval_with(x, &mut st);
                newton_min::<2>(&mut f, x0, 0.05 * cell, 0.7 * cell, done, 40)
            };
            let gn = m.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if !(m.val <= done) || m.rank() != 2 || gn * cell > 1e-3 * cmax {
                continue;
            }
            if (m.x[0] - x0[0]).abs() > 1.6 * cell || (m.x[1] - x0[1]).abs() > 1.6 * cell {
                continue;
            }
            let h2 = (0.5 * cell) * (0.5 * cell);
            if dots.iter().any(|d| (d[0] - m.x[0]).powi(2) + (d[1] - m.x[1]).powi(2) < h2) {
                continue;
            }
            dots.push(m.x);
            out.push([m.x, m.x]);
        }
    }
}

/// Like [`crate::mesh::contour_2d`], additionally returning the number of interval-evaluated
/// cells.
pub fn contour_2d_stats(
    p: &Program,
    x: (f64, f64),
    y: (f64, f64),
    min_cell: f64,
    max_depth: u32,
    max_cells: usize,
) -> (Vec<Seg>, usize) {
    let mut segs = Vec::new();
    if !(x.0.is_finite() && x.1.is_finite() && y.0.is_finite() && y.1.is_finite()) || x.1 <= x.0 || y.1 <= y.0 {
        return (segs, 0);
    }
    let max_depth = max_depth.min(30);
    let lat = (1u64 << max_depth) as f64;
    let (w, h) = (x.1 - x.0, y.1 - y.0);
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let mut evals = 0usize;
    // leaves: (ix, iy, depth) in cell units of that depth
    let mut leaves: Vec<(u64, u64, u32)> = Vec::new();
    let mut cur: Vec<(u64, u64)> = vec![(0, 0)];
    let mut depth = 0u32;
    while !cur.is_empty() {
        let mut next = Vec::with_capacity(cur.len() * 2);
        let n = (1u64 << depth) as f64;
        let (cw, ch) = (w / n, h / n);
        for &(cx, cy) in &cur {
            if evals >= max_cells {
                leaves.push((cx, cy, depth));
                continue;
            }
            evals += 1;
            let (ax, ay) = (x.0 + w * cx as f64 / n, y.0 + h * cy as f64 / n);
            let (bx, by) = (x.0 + w * (cx + 1) as f64 / n, y.0 + h * (cy + 1) as f64 / n);
            let r = p.eval_interval_with(&[Interval::new(ax, bx), Interval::new(ay, by)], &mut ist);
            if r.is_empty() || !r.contains_zero() {
                continue;
            }
            if (cw).max(ch) <= min_cell || depth >= max_depth || evals >= max_cells {
                leaves.push((cx, cy, depth));
                continue;
            }
            next.push((cx * 2, cy * 2));
            next.push((cx * 2 + 1, cy * 2));
            next.push((cx * 2, cy * 2 + 1));
            next.push((cx * 2 + 1, cy * 2 + 1));
        }
        cur = next;
        depth += 1;
    }
    let step_min = leaves.iter().map(|&(_, _, d)| 1u64 << (max_depth - d)).min().unwrap_or(1);
    let cell = (w / lat).max(h / lat) * step_min as f64;
    let mut g = Grid {
        p,
        st: Vec::with_capacity(16),
        x0: x.0,
        y0: y.0,
        w,
        h,
        lat,
        cell,
        vals: FxMap::default(),
        gvals: FxMap::default(),
        edges: FxMap::default(),
        gmode: false,
        tau: 0.0,
    };
    g.vals.reserve(leaves.len() * 2);
    let mut cands: Vec<(u64, u64, u64, [f64; 4])> = Vec::new();
    for (cx, cy, d) in leaves {
        let step = 1u64 << (max_depth - d);
        let (ix, iy) = (cx * step, cy * step);
        let v = [g.corner(ix, iy), g.corner(ix + step, iy), g.corner(ix + step, iy + step), g.corner(ix, iy + step)];
        if v.iter().any(|a| a.is_nan()) {
            continue;
        }
        let (npos, nneg) = (v.iter().filter(|&&a| a > 0.0).count(), v.iter().filter(|&&a| a < 0.0).count());
        if npos != 0 && npos != 4 {
            g.march(ix, iy, step, &v, &mut segs);
        }
        if (npos == 0 || nneg == 0) && npos + nneg > 0 {
            cands.push((ix, iy, step, v)); // one sign (or zero) everywhere: maybe a touch
        }
    }
    g.touch_pass(&cands, &mut segs);
    (segs, evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Angle};
    use crate::parse::parse;

    fn prog(src: &str) -> Program {
        compile(&parse(src).unwrap(), &["x", "y"], Angle::Rad).unwrap()
    }

    fn contour(src: &str, w: f64) -> Vec<Seg> {
        contour_2d_stats(&prog(src), (-w, w), (-w * 0.75, w * 0.75), 2.0 * w / 800.0, 11, 400_000).0
    }

    fn len(s: &Seg) -> f64 {
        (s[0][0] - s[1][0]).hypot(s[0][1] - s[1][1])
    }

    fn is_dot(s: &Seg) -> bool {
        len(s) == 0.0
    }

    #[test]
    fn point_conic_is_a_dot_at_the_point() {
        for (src, p) in [
            ("(x-0.37)^2+(y-0.21)^2", (0.37, 0.21)),
            ("x^2+y^2", (0.0, 0.0)),
            ("x^2+y^2-x+0.25", (0.5, 0.0)),
            ("(x-1.3)^2+4(y+0.9)^2", (1.3, -0.9)),
            ("-(x-0.37)^2-(y-0.21)^2", (0.37, 0.21)),
        ] {
            let segs = contour(src, 3.0);
            assert!(!segs.is_empty(), "{src}: nothing drawn");
            for s in &segs {
                assert!((s[0][0] - p.0).hypot(s[0][1] - p.1) < 1e-4 && (s[1][0] - p.0).hypot(s[1][1] - p.1) < 1e-4, "{src}: {s:?}");
            }
        }
    }

    #[test]
    fn double_lines_are_drawn_whole() {
        // (x-0.37)^2 = 0: the vertical line x = 0.37 across the whole window (height 4.5)
        for (src, dir) in [("(x-0.37)^2", (0.0, 1.0)), ("(x-y)^2", (1.0, 1.0)), ("(y+0.2)^2", (1.0, 0.0)), ("(2x+y-1)^2", (1.0, -2.0))] {
            let segs = contour(src, 3.0);
            let total: f64 = segs.iter().map(len).sum();
            assert!(!segs.is_empty(), "{src}");
            // length of the line inside the window x in [-3,3], y in [-2.25,2.25]
            let want = match dir {
                (0.0, _) => 4.5,
                (_, 0.0) => 6.0,
                (1.0, 1.0) => 4.5 * 2f64.sqrt(),
                _ => 4.5 * 5f64.sqrt() / 2.0 * 1.0,
            };
            assert!((total - want).abs() < 0.03 * want, "{src}: length {total} want {want}");
        }
    }

    #[test]
    fn double_curves_follow_the_curve() {
        // (x^2+y^2-4)^2 = 0: the circle of radius 2
        let segs = contour("(x^2+y^2-4)^2", 3.0);
        let total: f64 = segs.iter().map(len).sum();
        assert!((total - 4.0 * std::f64::consts::PI).abs() < 0.03 * 4.0 * std::f64::consts::PI, "{total}");
        for s in &segs {
            for q in s {
                assert!((q[0].hypot(q[1]) - 2.0).abs() < 1e-3, "{q:?}");
            }
        }
        // (y - x^2)^2 = 0: a parabola
        let segs = contour("(y-x^2)^2", 1.2);
        assert!(!segs.is_empty());
        for s in &segs {
            for q in s {
                assert!((q[1] - q[0] * q[0]).abs() < 1e-3, "{q:?}");
            }
        }
    }

    #[test]
    fn a_feature_smaller_than_a_cell_is_still_a_dot() {
        let segs = contour("x^2+y^2-0.00000001", 3.0);
        assert!(!segs.is_empty());
        assert!(segs.iter().all(|s| s[0][0].hypot(s[0][1]) < 0.02));
    }

    #[test]
    fn ordinary_curves_get_no_extra_dots() {
        for src in ["x^2+y^2-4", "x*y-1", "x^2/4-y^2-1", "sin(x)-y", "y-x^2"] {
            let segs = contour(src, 3.0);
            assert!(!segs.is_empty(), "{src}");
            assert!(!segs.iter().any(is_dot), "{src}: a zero-length segment");
        }
    }

    #[test]
    fn poles_are_not_joined() {
        // y = tan(x) as the implicit tan(x) - y: no segment spans an asymptote
        let segs = contour("tan(x)-y", 5.0);
        assert!(segs.len() > 100);
        let branch = |x: f64| ((x - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).floor();
        for s in &segs {
            assert_eq!(branch(s[0][0]), branch(s[1][0]), "{s:?}");
        }
        // y = 1/x as x*y - 1: the two branches stay apart
        let segs = contour("1/x-y", 4.0);
        assert!(segs.len() > 50);
        for s in &segs {
            assert!((s[0][0] < 0.0) == (s[1][0] < 0.0), "{s:?}");
        }
    }

    #[test]
    fn circle_at_800_px_is_traced_with_one_root_per_edge() {
        let p = prog("x^2+y^2-4");
        let (segs, cells) = contour_2d_stats(&p, (-3.0, 3.0), (-2.25, 2.25), 6.0 / 800.0, 11, 400_000);
        let total: f64 = segs.iter().map(len).sum();
        assert!((total - 4.0 * std::f64::consts::PI).abs() < 0.01 * 4.0 * std::f64::consts::PI, "{total}");
        assert!(cells > 0);
    }
}
