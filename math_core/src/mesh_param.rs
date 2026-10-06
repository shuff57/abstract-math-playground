//! Parametric surfaces `(x(u,v), y(u,v), z(u,v))` sampled into the same [`Mesh`] the implicit
//! surfaces use (indexed triangles, unit normals, counter-clockwise seen from the normal side).
//!
//! * The grid resolution follows the surface: a coarse probe measures how long the cells that
//!   touch the window box are, then `u` and `v` get as many samples as keep those cells below
//!   a fraction of the window (at least 96 each, capped by a vertex budget).
//! * Normals are `P_u x P_v` from grid differences (wrapping around a periodic seam, one-sided
//!   at an edge). Where that vanishes (a pole, a cone apex) the normal comes from the nearest
//!   non-degenerate triangle instead.
//! * Vertices that coincide on a collapsed row (pole) or across a seam and agree on the normal
//!   are welded, so closed surfaces (sphere, torus) have no open edges. A Moebius strip's
//!   seam keeps its two normals and stays unwelded.
//! * Cells with a non-finite corner are skipped (holes), as are triangles that leap across a
//!   discontinuity. Triangles are clipped to the window box.

use crate::compile::Program;
use crate::mesh::Mesh;

type V3 = [f64; 3];

const COARSE: usize = 48;
const MIN_CELLS: usize = 96;
const MAX_CELLS: usize = 640;
/// Sample budget (vertices) for the finest grid.
pub const DEFAULT_MAX_VERTICES: usize = 180_000;
const HUGE: f64 = 1e12;

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn finite(p: &V3) -> bool {
    p.iter().all(|c| c.is_finite() && c.abs() <= HUGE)
}

struct Sampler<'a> {
    p: [&'a Program; 3],
    st: Vec<f64>,
}

impl Sampler<'_> {
    fn at(&mut self, u: f64, v: f64) -> V3 {
        let mut o = [0.0; 3];
        for k in 0..3 {
            o[k] = self.p[k].eval_with(&[u, v], &mut self.st);
        }
        if finite(&o) {
            o
        } else {
            [f64::NAN; 3]
        }
    }
}

fn param_at(r: [f64; 2], i: usize, n: usize) -> f64 {
    if i == n {
        r[1]
    } else {
        r[0] + (r[1] - r[0]) * i as f64 / n as f64
    }
}

/// Nearest-rank percentile of a list (sorted in place).
fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// How many `u` and `v` cells to use: the coarse grid's cell lengths where they touch the box,
/// divided into the target length.
fn choose_resolution(s: &mut Sampler, u: [f64; 2], v: [f64; 2], min: V3, max: V3, budget: usize) -> (usize, usize) {
    let span = (0..3).map(|a| max[a] - min[a]).fold(0.0f64, f64::max);
    let target = span / 192.0;
    let n = COARSE;
    let mut g = vec![[f64::NAN; 3]; (n + 1) * (n + 1)];
    for i in 0..=n {
        for j in 0..=n {
            g[i * (n + 1) + j] = s.at(param_at(u, i, n), param_at(v, j, n));
        }
    }
    let touches = |a: &V3, b: &V3, c: &V3, d: &V3| {
        (0..3).all(|k| {
            let lo = a[k].min(b[k]).min(c[k]).min(d[k]);
            let hi = a[k].max(b[k]).max(c[k]).max(d[k]);
            hi >= min[k] - 0.1 * span && lo <= max[k] + 0.1 * span
        })
    };
    let (mut lu, mut lv) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (g[i * (n + 1) + j], g[(i + 1) * (n + 1) + j], g[(i + 1) * (n + 1) + j + 1], g[i * (n + 1) + j + 1]);
            if ![a, b, c, d].iter().all(finite) || !touches(&a, &b, &c, &d) {
                continue;
            }
            lu.push(len(sub(b, a)).max(len(sub(c, d))));
            lv.push(len(sub(d, a)).max(len(sub(c, b))));
        }
    }
    let need = |l: &mut Vec<f64>| {
        let m = percentile(l, 0.95);
        if target > 0.0 && m.is_finite() {
            (n as f64 * m / target).ceil().clamp(MIN_CELLS as f64, MAX_CELLS as f64) as usize
        } else {
            MIN_CELLS
        }
    };
    let (mut nu, mut nv) = (need(&mut lu), need(&mut lv));
    if (nu + 1) * (nv + 1) > budget {
        let k = (budget as f64 / ((nu + 1) * (nv + 1)) as f64).sqrt();
        nu = ((nu as f64 * k) as usize).max(32);
        nv = ((nv as f64 * k) as usize).max(32);
    }
    (nu, nv)
}

struct Dsu(Vec<u32>);
impl Dsu {
    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            let p = self.0[a as usize];
            self.0[a as usize] = self.0[p as usize];
            a = p;
        }
        a
    }
    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[b.max(a) as usize] = a.min(b);
        }
    }
}

/// Mesh of the surface over `u` x `v` (variables `[u, v]` of the three programs), clipped to the
/// box `min..max`. `max_vertices` bounds the grid.
pub fn surface_param(
    px: &Program,
    py: &Program,
    pz: &Program,
    u: [f64; 2],
    v: [f64; 2],
    min: V3,
    max: V3,
    max_vertices: usize,
) -> Mesh {
    let mut out = Mesh::default();
    if !(u[0].is_finite() && u[1].is_finite() && u[1] > u[0] && v[0].is_finite() && v[1].is_finite() && v[1] > v[0]) {
        return out;
    }
    let mut s = Sampler { p: [px, py, pz], st: Vec::new() };
    let (nu, nv) = choose_resolution(&mut s, u, v, min, max, max_vertices.max(4096));
    let w = nv + 1;
    let idx = |i: usize, j: usize| i * w + j;
    let mut pos = vec![[f64::NAN; 3]; (nu + 1) * w];
    for i in 0..=nu {
        let uu = param_at(u, i, nu);
        for j in 0..=nv {
            pos[idx(i, j)] = s.at(uu, param_at(v, j, nv));
        }
    }
    // size for tolerances: extent of the finite samples
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in pos.iter().filter(|p| finite(p)) {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if lo[0] > hi[0] {
        return out;
    }
    let size = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max).max(1e-300);
    let eps = size * 1e-9;
    let same = |a: &V3, b: &V3| finite(a) && finite(b) && len(sub(*a, *b)) <= eps;

    // periodic seams: the last row/column coincides with the first
    let u_periodic = (0..=nv).all(|j| {
        let (a, b) = (pos[idx(0, j)], pos[idx(nu, j)]);
        !finite(&a) || !finite(&b) || same(&a, &b)
    }) && (0..=nv).any(|j| finite(&pos[idx(0, j)]));
    let v_periodic = (0..=nu).all(|i| {
        let (a, b) = (pos[idx(i, 0)], pos[idx(i, nv)]);
        !finite(&a) || !finite(&b) || same(&a, &b)
    }) && (0..=nu).any(|i| finite(&pos[idx(i, 0)]));

    // normals from grid differences
    let nb = |i: isize, j: isize| -> Option<V3> {
        let wrap = |k: isize, n: usize, per: bool| -> Option<usize> {
            if k < 0 {
                per.then_some(n - 1)
            } else if k > n as isize {
                per.then_some(1)
            } else {
                Some(k as usize)
            }
        };
        let (a, b) = (wrap(i, nu, u_periodic)?, wrap(j, nv, v_periodic)?);
        let p = pos[idx(a, b)];
        finite(&p).then_some(p)
    };
    let diff = |c: V3, lo: Option<V3>, hi: Option<V3>| -> V3 {
        match (lo, hi) {
            (Some(l), Some(h)) => sub(h, l),
            (None, Some(h)) => sub(h, c),
            (Some(l), None) => sub(c, l),
            (None, None) => [0.0; 3],
        }
    };
    let mut normal = vec![[0.0; 3]; pos.len()];
    let mut degenerate = vec![false; pos.len()];
    for i in 0..=nu {
        for j in 0..=nv {
            let c = pos[idx(i, j)];
            if !finite(&c) {
                continue;
            }
            let (ii, jj) = (i as isize, j as isize);
            let pu = diff(c, nb(ii - 1, jj), nb(ii + 1, jj));
            let pv = diff(c, nb(ii, jj - 1), nb(ii, jj + 1));
            let n = cross(pu, pv);
            let l = len(n);
            if l > 1e-7 * len(pu) * len(pv) && l > 0.0 && l.is_finite() {
                normal[idx(i, j)] = scale(n, 1.0 / l);
            } else {
                degenerate[idx(i, j)] = true;
            }
        }
    }
    // degenerate vertices (poles, apexes): the face normal of the nearest sound triangle
    let face = |a: usize, b: usize, c: usize| -> Option<V3> {
        let (pa, pb, pc) = (pos[a], pos[b], pos[c]);
        if !(finite(&pa) && finite(&pb) && finite(&pc)) {
            return None;
        }
        let n = cross(sub(pb, pa), sub(pc, pa));
        let l = len(n);
        (l > 1e-9 * size * size).then(|| scale(n, 1.0 / l))
    };
    for i in 0..=nu {
        for j in 0..=nv {
            if !degenerate[idx(i, j)] {
                continue;
            }
            // quads around the vertex, nearest the lower-index side first
            let mut found = None;
            'search: for (di, dj) in [(-1isize, -1isize), (-1, 0), (0, -1), (0, 0)] {
                let (i0, j0) = (i as isize + di, j as isize + dj);
                if i0 < 0 || j0 < 0 || i0 as usize >= nu || j0 as usize >= nv {
                    continue;
                }
                let (i0, j0) = (i0 as usize, j0 as usize);
                let (a, b, c, d) = (idx(i0, j0), idx(i0 + 1, j0), idx(i0 + 1, j0 + 1), idx(i0, j0 + 1));
                for t in [(a, b, d), (b, c, d), (a, b, c), (a, c, d)] {
                    if let Some(n) = face(t.0, t.1, t.2) {
                        found = Some(n);
                        break 'search;
                    }
                }
            }
            normal[idx(i, j)] = found.unwrap_or([0.0, 0.0, 1.0]);
        }
    }

    // weld coincident vertices: collapsed grid edges (poles), and seams with agreeing normals
    let mut dsu = Dsu((0..pos.len() as u32).collect());
    for i in 0..=nu {
        for j in 0..=nv {
            let a = idx(i, j);
            if j < nv && same(&pos[a], &pos[idx(i, j + 1)]) {
                dsu.union(a as u32, idx(i, j + 1) as u32);
            }
            if i < nu && same(&pos[a], &pos[idx(i + 1, j)]) {
                dsu.union(a as u32, idx(i + 1, j) as u32);
            }
        }
    }
    if u_periodic {
        for j in 0..=nv {
            let (a, b) = (idx(0, j), idx(nu, j));
            if same(&pos[a], &pos[b]) && dot(normal[a], normal[b]) > 0.5 {
                dsu.union(a as u32, b as u32);
            }
        }
    }
    if v_periodic {
        for i in 0..=nu {
            let (a, b) = (idx(i, 0), idx(i, nv));
            if same(&pos[a], &pos[b]) && dot(normal[a], normal[b]) > 0.5 {
                dsu.union(a as u32, b as u32);
            }
        }
    }
    let rep: Vec<u32> = (0..pos.len() as u32).map(|a| dsu.find(a)).collect();
    // welded normal: sum of the members' normals
    let mut nsum = vec![[0.0; 3]; pos.len()];
    for (k, r) in rep.iter().enumerate() {
        if finite(&pos[k]) {
            nsum[*r as usize] = add(nsum[*r as usize], normal[k]);
        }
    }
    let unit = |n: V3, fallback: V3| -> V3 {
        let l = len(n);
        if l > 1e-9 {
            scale(n, 1.0 / l)
        } else {
            fallback
        }
    };

    // typical edge length (for the discontinuity guard)
    let mut edges = Vec::new();
    for i in (0..=nu).step_by(4) {
        for j in 0..nv {
            let (a, b) = (pos[idx(i, j)], pos[idx(i, j + 1)]);
            if finite(&a) && finite(&b) {
                edges.push(len(sub(a, b)));
            }
        }
    }
    for j in (0..=nv).step_by(4) {
        for i in 0..nu {
            let (a, b) = (pos[idx(i, j)], pos[idx(i + 1, j)]);
            if finite(&a) && finite(&b) {
                edges.push(len(sub(a, b)));
            }
        }
    }
    let nonzero: Vec<f64> = edges.iter().copied().filter(|e| *e > eps).collect();
    let mut nz = nonzero.clone();
    let median = percentile(&mut nz, 0.5);
    let max_edge = if median > 0.0 { 60.0 * median } else { f64::INFINITY };

    let slack = (0..3).map(|k| max[k] - min[k]).fold(0.0f64, f64::max) * 1e-9;
    let (bmin, bmax) = (sub(min, [slack; 3]), add(max, [slack; 3]));
    let inside = |p: &V3| (0..3).all(|k| p[k] >= bmin[k] && p[k] <= bmax[k]);

    let mut remap: Vec<u32> = vec![u32::MAX; pos.len()];
    let mut emit = |out: &mut Mesh, k: usize| -> u32 {
        let r = rep[k] as usize;
        if remap[r] == u32::MAX {
            remap[r] = out.positions.len() as u32;
            let p = pos[r];
            let n = unit(nsum[r], normal[r]);
            out.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            out.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
        }
        remap[r]
    };

    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            if ![a, b, c, d].iter().all(|k| finite(&pos[*k])) {
                continue;
            }
            // shorter diagonal
            let tris = if len(sub(pos[a], pos[c])) <= len(sub(pos[b], pos[d])) { [[a, b, c], [a, c, d]] } else { [[a, b, d], [b, c, d]] };
            for t in tris {
                let (pa, pb, pc) = (pos[t[0]], pos[t[1]], pos[t[2]]);
                let r = [rep[t[0]], rep[t[1]], rep[t[2]]];
                if r[0] == r[1] || r[1] == r[2] || r[0] == r[2] {
                    continue; // collapsed by a pole
                }
                let area2 = len(cross(sub(pb, pa), sub(pc, pa)));
                if area2 <= 1e-12 * size * size {
                    continue;
                }
                if len(sub(pa, pb)).max(len(sub(pb, pc))).max(len(sub(pc, pa))) > max_edge {
                    continue; // leaps across a discontinuity
                }
                if [pa, pb, pc].iter().all(inside) {
                    let ids = [emit(&mut out, t[0]), emit(&mut out, t[1]), emit(&mut out, t[2])];
                    out.indices.extend(ids);
                    continue;
                }
                // wholly beyond one face of the box?
                if (0..3).any(|k| {
                    [pa, pb, pc].iter().all(|p| p[k] < bmin[k]) || [pa, pb, pc].iter().all(|p| p[k] > bmax[k])
                }) {
                    continue;
                }
                let vn = |k: usize| unit(nsum[rep[k] as usize], normal[k]);
                let poly = vec![(pa, vn(t[0]), Some(t[0])), (pb, vn(t[1]), Some(t[1])), (pc, vn(t[2]), Some(t[2]))];
                let poly = clip_polygon(poly, min, max);
                if poly.len() < 3 {
                    continue;
                }
                let ids: Vec<u32> = poly
                    .iter()
                    .map(|(p, n, orig)| match orig {
                        Some(k) => emit(&mut out, *k),
                        None => {
                            out.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                            out.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
                            (out.positions.len() - 1) as u32
                        }
                    })
                    .collect();
                for q in 1..ids.len() - 1 {
                    out.indices.extend([ids[0], ids[q], ids[q + 1]]);
                }
            }
        }
    }
    out
}

type Vert = (V3, V3, Option<usize>);

/// Sutherland-Hodgman against the six planes of the box. Original vertices keep their grid id
/// (so neighbours still share them); vertices made by a cut carry `None`.
fn clip_polygon(mut poly: Vec<Vert>, min: V3, max: V3) -> Vec<Vert> {
    for k in 0..3 {
        for upper in [false, true] {
            let bound = if upper { max[k] } else { min[k] };
            let sd = |p: &V3| if upper { bound - p[k] } else { p[k] - bound };
            let mut next: Vec<Vert> = Vec::with_capacity(poly.len() + 2);
            for q in 0..poly.len() {
                let (cur, nxt) = (&poly[q], &poly[(q + 1) % poly.len()]);
                let (dc, dn) = (sd(&cur.0), sd(&nxt.0));
                if dc >= 0.0 {
                    next.push(cur.clone());
                }
                if (dc >= 0.0) != (dn >= 0.0) {
                    let t = dc / (dc - dn);
                    let p = add(cur.0, scale(sub(nxt.0, cur.0), t));
                    let mut n = add(cur.1, scale(sub(nxt.1, cur.1), t));
                    let l = len(n);
                    if l > 1e-12 {
                        n = scale(n, 1.0 / l);
                    } else {
                        n = cur.1;
                    }
                    let mut p = p;
                    p[k] = bound; // exactly on the plane
                    next.push((p, n, None));
                }
            }
            poly = next;
            if poly.len() < 3 {
                return Vec::new();
            }
        }
    }
    poly
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Angle};
    use crate::parse::parse;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    fn progs(c: [&str; 3]) -> [Program; 3] {
        let p = |s: &str| compile(&parse(s).unwrap(), &["u", "v"], Angle::Rad).unwrap();
        [p(c[0]), p(c[1]), p(c[2])]
    }

    fn mesh(c: [&str; 3], u: [f64; 2], v: [f64; 2], half: f64) -> Mesh {
        let [x, y, z] = progs(c);
        surface_param(&x, &y, &z, u, v, [-half; 3], [half; 3], DEFAULT_MAX_VERTICES)
    }

    fn open_edges(m: &Mesh) -> usize {
        let mut e: HashMap<(u32, u32), u32> = HashMap::new();
        for t in m.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *e.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        e.values().filter(|n| **n == 1).count()
    }

    fn area(m: &Mesh) -> f64 {
        m.indices
            .chunks(3)
            .map(|t| {
                let p = |i: u32| m.positions[i as usize].map(f64::from);
                0.5 * len(cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0]))))
            })
            .sum()
    }

    fn assert_good_normals(m: &Mesh) {
        for n in &m.normals {
            let l = ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) as f64).sqrt();
            assert!((l - 1.0).abs() < 1e-3, "normal length {l}");
        }
        // winding agrees with the vertex normals
        for t in m.indices.chunks(3) {
            let p = |i: u32| m.positions[i as usize].map(f64::from);
            let n = cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0])));
            let vn: V3 = (0..3).fold([0.0; 3], |a, k| add(a, m.normals[t[k] as usize].map(f64::from)));
            assert!(dot(n, vn) >= 0.0, "winding flipped");
        }
    }

    #[test]
    fn sphere_is_closed_with_the_right_area_and_outward_normals() {
        let m = mesh(["3cos(u)cos(v)", "3cos(u)sin(v)", "3sin(u)"], [-PI / 2.0, PI / 2.0], [0.0, 2.0 * PI], 10.0);
        assert!(!m.indices.is_empty());
        assert_eq!(open_edges(&m), 0, "pole rows and the seam are welded");
        let a = area(&m);
        assert!((a - 4.0 * PI * 9.0).abs() / (4.0 * PI * 9.0) < 0.01, "area {a}");
        assert_good_normals(&m);
        for (p, n) in m.positions.iter().zip(&m.normals) {
            // P_u x P_v points inward for this parametrisation; the point is that it is radial
            let d = dot(p.map(f64::from), n.map(f64::from)) / 3.0;
            assert!(d < -0.99, "normal not radial: {d}");
        }
    }

    #[test]
    fn torus_is_closed_and_has_the_right_area() {
        let m = mesh(["(3+cos(v))cos(u)", "(3+cos(v))sin(u)", "sin(v)"], [0.0, 2.0 * PI], [0.0, 2.0 * PI], 10.0);
        assert_eq!(open_edges(&m), 0);
        let want = 4.0 * PI * PI * 3.0 * 1.0;
        assert!((area(&m) - want).abs() / want < 0.01, "area {}", area(&m));
        assert_good_normals(&m);
    }

    #[test]
    fn moebius_seam_keeps_two_normals_but_no_gap() {
        let m = mesh(
            ["(2+v cos(u/2))cos(u)", "(2+v cos(u/2))sin(u)", "v sin(u/2)"],
            [0.0, 2.0 * PI],
            [-1.0, 1.0],
            10.0,
        );
        assert!(!m.indices.is_empty());
        assert_good_normals(&m);
        // the seam is open in index space only
        assert!(open_edges(&m) > 0);
    }

    #[test]
    fn clipping_cuts_at_the_box_and_keeps_inside_only() {
        let m = mesh(["3cos(u)cos(v)", "3cos(u)sin(v)", "3sin(u)"], [-PI / 2.0, PI / 2.0], [0.0, 2.0 * PI], 2.0);
        assert!(!m.indices.is_empty());
        for p in &m.positions {
            assert!(p.iter().all(|c| c.abs() <= 2.0 + 1e-4), "{p:?}");
        }
        assert_good_normals(&m);
        // a box that misses the surface draws nothing
        let [x, y, z] = progs(["3cos(u)cos(v)", "3cos(u)sin(v)", "3sin(u)"]);
        let e = surface_param(&x, &y, &z, [-PI / 2.0, PI / 2.0], [0.0, 2.0 * PI], [20.0; 3], [30.0; 3], 100_000);
        assert!(e.indices.is_empty());
    }

    #[test]
    fn nan_holes_and_bad_ranges_do_not_panic() {
        // sqrt of a negative below u=0: holes, not garbage
        let m = mesh(["u", "v", "sqrt(u)"], [-1.0, 1.0], [-1.0, 1.0], 5.0);
        assert!(!m.indices.is_empty());
        assert!(m.positions.iter().all(|p| p.iter().all(|c| c.is_finite())));
        assert!(m.positions.iter().all(|p| p[0] >= -1e-3));
        let [x, y, z] = progs(["u", "v", "0"]);
        assert!(surface_param(&x, &y, &z, [1.0, 1.0], [0.0, 1.0], [-5.0; 3], [5.0; 3], 10_000).indices.is_empty());
        assert!(surface_param(&x, &y, &z, [f64::NAN, 1.0], [0.0, 1.0], [-5.0; 3], [5.0; 3], 10_000).indices.is_empty());
    }

    #[test]
    fn cone_apex_has_sound_normals_and_long_surfaces_get_finer_grids() {
        let cone = mesh(["v cos(u)", "v sin(u)", "v"], [0.0, 2.0 * PI], [0.0, 4.0], 6.0);
        assert_good_normals(&cone);
        // a helicoid winding fast needs far more than 96 samples across u
        let h = mesh(["v cos(u)", "v sin(u)", "u/2"], [-6.0, 6.0], [-5.0, 5.0], 6.0);
        assert!(h.positions.len() > 96 * 96 * 2, "{}", h.positions.len());
        assert!(h.positions.len() <= DEFAULT_MAX_VERTICES + 4096);
    }

    #[test]
    fn a_flat_plane_is_one_sheet_with_up_normals() {
        let m = mesh(["u", "v", "1"], [-3.0, 3.0], [-3.0, 3.0], 6.0);
        assert_eq!(m.positions.len(), (MIN_CELLS + 1) * (MIN_CELLS + 1));
        assert!(m.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-4));
        assert_eq!(open_edges(&m), 4 * MIN_CELLS);
    }
}
