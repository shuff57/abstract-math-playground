//! Implicit 3D surfaces `F(x, y, z) = 0`: octree + interval pruning + marching tetrahedra.
//!
//! Every cube is split into six Kuhn tetrahedra around its main diagonal. The split is the same
//! in every cube, so neighbouring cells always agree on face diagonals: the mesh has no cracks and
//! no ambiguous-case holes, and no hand-typed tables can be wrong.
//!
//! Speed: corner values live in one memo table shared by all cells, edge crossings are found with
//! a bracketed secant (Illinois) iteration of 3-6 evaluations instead of 16 bisections, the vertex
//! normal costs three forward differences, and nothing allocates per cell.
//!
//! Robustness: a sign change that is really a pole or a jump (`tan`, `1/x`, `floor`) is detected
//! at the edge root (the function value does not shrink with the bracket) and produces no
//! triangle. Cells where `F` touches zero without changing sign (point and line quadrics, squared
//! factors) are handled by the "touch pass": the zero set of `w . grad F` (which does change sign
//! across a double root) restricted to places where `F` is zero gives the double surface, and a
//! damped Newton search finds isolated minima and double lines.

use crate::compile::Program;
use crate::interval::Interval;
use crate::mesh::Mesh;
use crate::mesh_touch::{newton_min, FxMap};

pub(crate) type V3 = [f64; 3];

pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: V3) -> Option<V3> {
    let l = dot(a, a).sqrt();
    (l.is_finite() && l > 1e-300).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

const REJECT: u32 = u32::MAX;
const UNSET: u32 = u32::MAX - 1;
/// The six Kuhn tetrahedra of a cube (corner bits: 1 = +x, 2 = +y, 4 = +z). Every tetrahedron is a
/// chain `0 < a < b < 7` in the subset order of corner bits, so each of its edges runs from a
/// corner to a corner with strictly more bits set: a lattice edge is `(lower corner, bit mask)`.
const KUHN: [[usize; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
/// Directions used for `w . grad F`; deliberately not parallel to a coordinate plane or
/// diagonal. A cell whose first direction hits a place where `w . grad F` itself vanishes tries
/// the next one.
const WS: [V3; 3] = [[0.6237, 0.5513, 0.5565], [-0.4437, 0.7791, 0.4416], [0.2371, -0.5718, 0.7864]];

struct Surf<'a> {
    p: &'a Program,
    st: Vec<f64>,
    min: V3,
    size: V3,
    grid: f64,
    cell: f64,
    gstep: f64,
    /// Lattice key -> compact vertex index (into `vval`, `evert`, `cvert`).
    vals: FxMap<u64, u32>,
    vval: Vec<f64>,
    /// Per lattice vertex and per edge direction (mask - 1): the mesh vertex on that edge.
    evert: Vec<[u32; 7]>,
    /// Per lattice vertex: the mesh vertex placed exactly on it (a root that lands on a corner).
    cvert: Vec<u32>,
    gvals: [FxMap<u64, f64>; 3],
    edges: FxMap<(u64, u64), u32>,
    gedges: [FxMap<(u64, u64), u32>; 3],
    /// Which of [`WS`] the touch pass is using, and whether a crossing was rejected under it.
    wdir: usize,
    rejected: bool,
    quant: FxMap<[i64; 3], u32>,
    qscale: f64,
    pos: Vec<V3>,
    nrm: Vec<V3>,
    facen: Vec<V3>,
    idx: Vec<u32>,
    /// `true` while polygonising `G = W . grad F` instead of `F` (touch pass).
    gmode: bool,
    /// Largest `|F|` allowed at a `G` vertex.
    tau: f64,
}

#[inline]
fn key(ix: [u64; 3]) -> u64 {
    ix[0] | (ix[1] << 21) | (ix[2] << 42)
}

impl Surf<'_> {
    fn coord(&self, ix: [u64; 3]) -> V3 {
        [
            self.min[0] + self.size[0] * ix[0] as f64 / self.grid,
            self.min[1] + self.size[1] * ix[1] as f64 / self.grid,
            self.min[2] + self.size[2] * ix[2] as f64 / self.grid,
        ]
    }

    #[inline]
    fn fraw(&mut self, q: V3) -> f64 {
        self.p.eval_with(&q, &mut self.st)
    }

    /// The function being polygonised: `F`, or `W . grad F` in the touch pass.
    fn f(&mut self, q: V3) -> f64 {
        if !self.gmode {
            return self.fraw(q);
        }
        let e = self.cell * 1e-3;
        let w = WS[self.wdir];
        let a = self.fraw([q[0] + e * w[0], q[1] + e * w[1], q[2] + e * w[2]]);
        let b = self.fraw([q[0] - e * w[0], q[1] - e * w[1], q[2] - e * w[2]]);
        (a - b) / (2.0 * e)
    }

    /// Compact index and value of lattice vertex `ix` (evaluated once).
    fn corner_id(&mut self, ix: [u64; 3]) -> (u32, f64) {
        let k = key(ix);
        if let Some(&i) = self.vals.get(&k) {
            return (i, self.vval[i as usize]);
        }
        let v = self.fraw(self.coord(ix));
        let i = self.vval.len() as u32;
        self.vval.push(v);
        self.evert.push([UNSET; 7]);
        self.cvert.push(UNSET);
        self.vals.insert(k, i);
        (i, v)
    }

    fn corner(&mut self, ix: [u64; 3]) -> f64 {
        self.corner_id(ix).1
    }

    fn gcorner(&mut self, ix: [u64; 3]) -> f64 {
        let k = key(ix);
        if let Some(&v) = self.gvals[self.wdir].get(&k) {
            return v;
        }
        let v = self.f(self.coord(ix));
        self.gvals[self.wdir].insert(k, v);
        v
    }

    /// Unit gradient of `F` at `q` (forward differences from the known value `fq`), if finite.
    fn gradient(&mut self, q: V3, fq: f64) -> Option<V3> {
        let h = self.gstep;
        let mut g = [0.0; 3];
        for a in 0..3 {
            let mut u = q;
            u[a] += h;
            let fu = self.fraw(u);
            if fu.is_finite() {
                g[a] = (fu - fq) / h;
            } else {
                let mut d = q;
                d[a] -= h;
                let fd = self.fraw(d);
                if !fd.is_finite() {
                    return None;
                }
                g[a] = (fq - fd) / h;
            }
        }
        normalize(g)
    }

    /// Root of the polygonised function on the segment `pa..pb` whose end values `fa`, `fb` lie
    /// in different classes (`> 0` or not): the point, its value and whether it is so close to an
    /// end that it is that end (`-1` = `pa`, `1` = `pb`, `0` = neither). `None` when the "root"
    /// is a pole or a jump.
    fn root(&mut self, pa: V3, pb: V3, fa: f64, fb: f64) -> Option<(V3, f64, i8)> {
        let lerp = |t: f64| [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t];
        if fa == 0.0 {
            return Some((pa, 0.0, -1));
        }
        if fb == 0.0 {
            return Some((pb, 0.0, 1));
        }
        let mut scale = fa.abs().max(fb.abs());
        let mut bisect = false;
        if !scale.is_finite() {
            // an infinite end (the function overflows right at a pole of the graph): bisect
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
                // A hole in the domain: fall back to bisection and treat it as the a-side.
                bisect = true;
                t0 = tm;
                if t1 - t0 < 1e-7 {
                    break;
                }
                continue;
            }
            if fm == 0.0 || fm.abs() <= 1e-5 * scale {
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
            return None; // the function does not go to zero with the bracket: pole or jump
        }
        if fm.is_nan() {
            tm = 0.5 * (t0 + t1);
            fm = 0.0;
        }
        let snap = if tm < 1e-6 {
            -1
        } else if tm > 1.0 - 1e-6 {
            1
        } else {
            0
        };
        Some((lerp(tm), fm, snap))
    }

    /// Adds a mesh vertex at `q`, merging with an existing one at (almost) the same place.
    fn add_vertex_q(&mut self, q: V3, fq: f64) -> u32 {
        let qk = [
            (q[0] / self.qscale).round() as i64,
            (q[1] / self.qscale).round() as i64,
            (q[2] / self.qscale).round() as i64,
        ];
        if let Some(&i) = self.quant.get(&qk) {
            return i;
        }
        let i = self.add_vertex(q, fq);
        self.quant.insert(qk, i);
        i
    }

    fn add_vertex(&mut self, q: V3, fq: f64) -> u32 {
        let i = self.pos.len() as u32;
        self.pos.push(q);
        let n = if self.gmode { None } else { self.gradient(q, fq) };
        let n = n.unwrap_or([0.0; 3]);
        self.nrm.push(n);
        self.facen.push([0.0; 3]);
        i
    }

    /// Vertex on the edge between cube corners `a` and `b` (opposite classes), or [`REJECT`].
    fn edge(&mut self, base: [u64; 3], step: u64, ids: &[u32; 8], v: &[f64; 8], a: usize, b: usize) -> u32 {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        let fast = !self.gmode && step == 1;
        let cix = |c: usize| [base[0] + (c as u64 & 1) * step, base[1] + ((c as u64 >> 1) & 1) * step, base[2] + ((c as u64 >> 2) & 1) * step];
        let (ia, ib) = (cix(a), cix(b));
        let ek = if fast {
            let s = self.evert[ids[a] as usize][(a ^ b) - 1];
            if s != UNSET {
                return s;
            }
            (0, 0)
        } else {
            let (ka, kb) = (key(ia), key(ib));
            let ek = if ka < kb { (ka, kb) } else { (kb, ka) };
            let cached = if self.gmode { self.gedges[self.wdir].get(&ek) } else { self.edges.get(&ek) };
            if let Some(&i) = cached {
                if i == REJECT {
                    self.rejected = true;
                }
                return i;
            }
            ek
        };
        let id = match self.root(self.coord(ia), self.coord(ib), v[a], v[b]) {
            None => {
                self.rejected |= self.gmode;
                REJECT
            }
            Some((q, fq, snap)) => {
                if self.gmode {
                    let fv = self.fraw(q);
                    if !(fv.abs() <= self.tau) {
                        self.rejected = true;
                        REJECT
                    } else {
                        self.add_vertex_q(q, fq)
                    }
                } else if fast {
                    match snap {
                        0 => self.add_vertex(q, fq),
                        s => {
                            let ci = ids[if s < 0 { a } else { b }] as usize;
                            if self.cvert[ci] == UNSET {
                                self.cvert[ci] = self.add_vertex(q, fq);
                            }
                            self.cvert[ci]
                        }
                    }
                } else {
                    self.add_vertex_q(q, fq)
                }
            }
        };
        if fast {
            self.evert[ids[a] as usize][(a ^ b) - 1] = id;
        } else if self.gmode {
            self.gedges[self.wdir].insert(ek, id);
        } else {
            self.edges.insert(ek, id);
        }
        id
    }

    /// Emits triangle `a b c`, wound so that its normal points from the negative towards the
    /// positive side (`dir` = positive centroid minus negative centroid of the tetrahedron).
    fn triangle(&mut self, a: u32, b: u32, c: u32, dir: V3) {
        if a == REJECT || b == REJECT || c == REJECT || a == b || b == c || a == c {
            return;
        }
        let (pa, pb, pc) = (self.pos[a as usize], self.pos[b as usize], self.pos[c as usize]);
        let fnrm = cross(sub(pb, pa), sub(pc, pa));
        if dot(fnrm, fnrm) == 0.0 {
            return;
        }
        // wind with the vertex normals (what the audit and the renderer compare against); where
        // those vanish or cancel, use the side the positive corners lie on
        let g = [
            self.nrm[a as usize][0] + self.nrm[b as usize][0] + self.nrm[c as usize][0],
            self.nrm[a as usize][1] + self.nrm[b as usize][1] + self.nrm[c as usize][1],
            self.nrm[a as usize][2] + self.nrm[b as usize][2] + self.nrm[c as usize][2],
        ];
        let want = if dot(g, g) > 1e-6 { g } else { dir };
        let (b, c, fnrm) = if dot(fnrm, want) < 0.0 { (c, b, [-fnrm[0], -fnrm[1], -fnrm[2]]) } else { (b, c, fnrm) };
        let _ = fnrm;
        self.idx.extend_from_slice(&[a, b, c]);
    }

    /// Polygonises one cube with corner values `v` (indexed by corner bits).
    fn march(&mut self, base: [u64; 3], step: u64, ids: &[u32; 8], v: &[f64; 8]) {
        let sc = [
            self.size[0] / self.grid * step as f64,
            self.size[1] / self.grid * step as f64,
            self.size[2] / self.grid * step as f64,
        ];
        for tet in KUHN {
            let (mut pos, mut neg) = ([0usize; 4], [0usize; 4]);
            let (mut np, mut nn) = (0, 0);
            for &c in &tet {
                if v[c] > 0.0 {
                    pos[np] = c;
                    np += 1;
                } else {
                    neg[nn] = c;
                    nn += 1;
                }
            }
            if np == 0 || nn == 0 {
                continue;
            }
            // positive centroid minus negative centroid, from the corner bits
            let mut dir = [0.0; 3];
            for k in 0..3 {
                let mp: f64 = pos[..np].iter().map(|&c| ((c >> k) & 1) as f64).sum::<f64>() / np as f64;
                let mn: f64 = neg[..nn].iter().map(|&c| ((c >> k) & 1) as f64).sum::<f64>() / nn as f64;
                dir[k] = (mp - mn) * sc[k];
            }
            match (np, nn) {
                (1, 3) | (3, 1) => {
                    let (lone, others) = if np == 1 { (pos[0], &neg[..3]) } else { (neg[0], &pos[..3]) };
                    let a = self.edge(base, step, ids, v, lone, others[0]);
                    let b = self.edge(base, step, ids, v, lone, others[1]);
                    let c = self.edge(base, step, ids, v, lone, others[2]);
                    self.triangle(a, b, c, dir);
                }
                (2, 2) => {
                    let ac = self.edge(base, step, ids, v, pos[0], neg[0]);
                    let ad = self.edge(base, step, ids, v, pos[0], neg[1]);
                    let bd = self.edge(base, step, ids, v, pos[1], neg[1]);
                    let bc = self.edge(base, step, ids, v, pos[1], neg[0]);
                    self.triangle(ac, ad, bd, dir);
                    self.triangle(ac, bd, bc, dir);
                }
                _ => {}
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // touch pass
    // -----------------------------------------------------------------------------------------

    fn lattice_val(&mut self, i: [i64; 3]) -> Option<f64> {
        let g = self.grid as i64;
        if (0..3).any(|a| i[a] < 0 || i[a] > g) {
            return None;
        }
        Some(self.corner([i[0] as u64, i[1] as u64, i[2] as u64]))
    }

    /// Whether some lattice vertex in the 4x4x4 block around the cell has the opposite class
    /// (then the zero set there is an ordinary sign-changing surface, not a touch).
    fn has_opposite_nearby(&mut self, base: [u64; 3], step: u64, pos_class: bool) -> bool {
        let s = step as i64;
        let b = [base[0] as i64, base[1] as i64, base[2] as i64];
        // memo-only scan first: almost every ordinary cell is decided here without evaluating
        for pass in 0..2 {
            for dz in -1..=2i64 {
                for dy in -1..=2i64 {
                    for dx in -1..=2i64 {
                        let i = [b[0] + dx * s, b[1] + dy * s, b[2] + dz * s];
                        let g = self.grid as i64;
                        if (0..3).any(|a| i[a] < 0 || i[a] > g) {
                            continue;
                        }
                        let k = key([i[0] as u64, i[1] as u64, i[2] as u64]);
                        let v = if pass == 0 {
                            match self.vals.get(&k) {
                                Some(&vi) => self.vval[vi as usize],
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
        }
        false
    }

    /// Smallest-`s*F` corner of the cell if it is also no larger than its 26 lattice neighbours.
    fn local_min_corner(&mut self, base: [u64; 3], step: u64, s: f64) -> Option<[u64; 3]> {
        let st = step as i64;
        let mut best: Option<([u64; 3], f64)> = None;
        for c in 0..8u64 {
            let ix = [base[0] + (c & 1) * step, base[1] + ((c >> 1) & 1) * step, base[2] + ((c >> 2) & 1) * step];
            let v = s * self.corner(ix);
            if best.map_or(true, |(_, bv)| v < bv) {
                best = Some((ix, v));
            }
        }
        let (ix, v) = best?;
        let tol = 1e-9 * v.abs();
        for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let i = [ix[0] as i64 + dx * st, ix[1] as i64 + dy * st, ix[2] as i64 + dz * st];
                    if let Some(nv) = self.lattice_val(i) {
                        if !nv.is_nan() && s * nv < v - tol {
                            return None;
                        }
                    }
                }
            }
        }
        Some(ix)
    }

    fn octahedron(&mut self, c: V3, r: f64) {
        // octahedron subdivided once and pushed to the sphere: 18 vertices, 32 triangles
        let mut verts: Vec<V3> = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let faces: [[usize; 3]; 8] = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let mut mids: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for f in faces {
            let mut m = [0usize; 3];
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let e = (a.min(b), a.max(b));
                m[k] = *mids.entry(e).or_insert_with(|| {
                    let s = [verts[a][0] + verts[b][0], verts[a][1] + verts[b][1], verts[a][2] + verts[b][2]];
                    verts.push(normalize(s).unwrap_or([0.0, 0.0, 1.0]));
                    verts.len() - 1
                });
            }
            tris.push([f[0], m[0], m[2]]);
            tris.push([m[0], f[1], m[1]]);
            tris.push([m[2], m[1], f[2]]);
            tris.push([m[0], m[1], m[2]]);
        }
        let base = self.pos.len() as u32;
        for v in &verts {
            self.pos.push([c[0] + r * v[0], c[1] + r * v[1], c[2] + r * v[2]]);
            self.nrm.push(*v);
            self.facen.push(*v);
        }
        for t in tris {
            let (a, b, cc) = (verts[t[0]], verts[t[1]], verts[t[2]]);
            let n = cross(sub(b, a), sub(cc, a));
            let cen = [a[0] + b[0] + cc[0], a[1] + b[1] + cc[1], a[2] + b[2] + cc[2]];
            let (j, k) = if dot(n, cen) < 0.0 { (t[2], t[1]) } else { (t[1], t[2]) };
            self.idx.extend_from_slice(&[base + t[0] as u32, base + j as u32, base + k as u32]);
        }
    }

    fn tube(&mut self, pts: &[V3], r: f64) {
        const SIDES: usize = 8;
        if pts.len() < 2 {
            return;
        }
        let base = self.pos.len() as u32;
        for i in 0..pts.len() {
            let a = pts[i.saturating_sub(1)];
            let b = pts[(i + 1).min(pts.len() - 1)];
            let t = normalize(sub(b, a)).unwrap_or([0.0, 0.0, 1.0]);
            let helper = if t[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let u = normalize(cross(t, helper)).unwrap_or([1.0, 0.0, 0.0]);
            let w = cross(t, u);
            for k in 0..SIDES {
                let ang = std::f64::consts::TAU * k as f64 / SIDES as f64;
                let n = [
                    ang.cos() * u[0] + ang.sin() * w[0],
                    ang.cos() * u[1] + ang.sin() * w[1],
                    ang.cos() * u[2] + ang.sin() * w[2],
                ];
                self.pos.push([pts[i][0] + r * n[0], pts[i][1] + r * n[1], pts[i][2] + r * n[2]]);
                self.nrm.push(n);
                self.facen.push(n);
            }
        }
        for i in 0..pts.len() - 1 {
            for k in 0..SIDES {
                let k1 = (k + 1) % SIDES;
                let a = base + (i * SIDES + k) as u32;
                let b = base + (i * SIDES + k1) as u32;
                let c = base + ((i + 1) * SIDES + k) as u32;
                let d = base + ((i + 1) * SIDES + k1) as u32;
                // orient outward (radially) using the vertex normal of `a`
                for tri in [[a, b, d], [a, d, c]] {
                    let (pa, pb, pc) = (self.pos[tri[0] as usize], self.pos[tri[1] as usize], self.pos[tri[2] as usize]);
                    let n = cross(sub(pb, pa), sub(pc, pa));
                    if dot(n, self.nrm[a as usize]) < 0.0 {
                        self.idx.extend_from_slice(&[tri[0], tri[2], tri[1]]);
                    } else {
                        self.idx.extend_from_slice(&tri);
                    }
                }
            }
        }
    }

    /// Re-winds the triangles `idx[from..]` so that neighbouring ones agree on which side is
    /// front. A double surface has no gradient to say which way is out (`G` changes sign across
    /// the great circle where `w . grad A` vanishes), so the sides are matched by flood fill.
    fn orient_consistently(&mut self, from: usize) {
        let ntri = (self.idx.len() - from) / 3;
        if ntri == 0 {
            return;
        }
        // undirected edge -> triangles using it, with the direction they traverse it
        let mut by_edge: FxMap<(u32, u32), [(u32, i8); 2]> = FxMap::default();
        let mut overfull: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for t in 0..ntri {
            let v = [self.idx[from + 3 * t], self.idx[from + 3 * t + 1], self.idx[from + 3 * t + 2]];
            for k in 0..3 {
                let (a, b) = (v[k], v[(k + 1) % 3]);
                let (e, d) = if a < b { ((a, b), 1i8) } else { ((b, a), -1i8) };
                let slot = by_edge.entry(e).or_insert([(u32::MAX, 0); 2]);
                if slot[0].0 == u32::MAX {
                    slot[0] = (t as u32, d);
                } else if slot[1].0 == u32::MAX {
                    slot[1] = (t as u32, d);
                } else {
                    overfull.insert(e);
                }
            }
        }
        let mut flip = vec![false; ntri];
        let mut seen = vec![false; ntri];
        let mut stack: Vec<u32> = Vec::new();
        for seed in 0..ntri {
            if seen[seed] {
                continue;
            }
            seen[seed] = true;
            stack.push(seed as u32);
            while let Some(t) = stack.pop() {
                let t = t as usize;
                let v = [self.idx[from + 3 * t], self.idx[from + 3 * t + 1], self.idx[from + 3 * t + 2]];
                for k in 0..3 {
                    let (a, b) = (v[k], v[(k + 1) % 3]);
                    let (e, d) = if a < b { ((a, b), 1i8) } else { ((b, a), -1i8) };
                    if overfull.contains(&e) {
                        continue;
                    }
                    let slot = by_edge[&e];
                    let other = if slot[0].0 as usize == t { slot[1] } else { slot[0] };
                    if other.0 == u32::MAX || other.0 as usize == t {
                        continue;
                    }
                    let n = other.0 as usize;
                    if seen[n] {
                        continue;
                    }
                    // effective direction of t on this edge, and the one n needs
                    let eff_t = if flip[t] { -d } else { d };
                    let need = -eff_t;
                    flip[n] = other.1 != need;
                    seen[n] = true;
                    stack.push(n as u32);
                }
            }
        }
        for t in 0..ntri {
            if flip[t] {
                self.idx.swap(from + 3 * t + 1, from + 3 * t + 2);
            }
        }
    }

    /// Cells where `F` has one sign at all corners and nowhere nearby: look for a double surface
    /// (the zero set of `W . grad F` where `F = 0`), isolated minima and double lines.
    fn touch_pass(&mut self, cands: &[([u64; 3], u64, [f64; 8])]) {
        let mut touching: Vec<([u64; 3], u64, [f64; 8])> = Vec::new();
        for &(base, step, v) in cands {
            if v.iter().any(|x| !x.is_finite()) {
                continue; // a pole of F, not a touch
            }
            let pos_class = v.iter().any(|&x| x > 0.0);
            if !self.has_opposite_nearby(base, step, pos_class) {
                touching.push((base, step, v));
            }
        }
        if touching.is_empty() {
            return;
        }
        // (1) double surfaces: polygonise G = w . grad F, keep only vertices where F ~ 0. Where a
        // cell meets the surface w . grad F = 0 (so G vanishes off the double surface and the
        // crossing is rejected) it is redone with the next direction.
        self.gmode = true;
        let g_from = self.idx.len();
        for &(base, step, v) in &touching {
            let cmax = v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            self.tau = 1e-4 * cmax;
            let mark = self.idx.len();
            for d in 0..WS.len() {
                self.wdir = d;
                let mut gv = [0.0; 8];
                for c in 0..8u64 {
                    let ix = [base[0] + (c & 1) * step, base[1] + ((c >> 1) & 1) * step, base[2] + ((c >> 2) & 1) * step];
                    gv[c as usize] = self.gcorner(ix);
                }
                if gv.iter().any(|x| x.is_nan()) {
                    continue;
                }
                self.idx.truncate(mark);
                self.rejected = false;
                self.march(base, step, &[0; 8], &gv);
                if !self.rejected {
                    break;
                }
            }
        }
        self.gmode = false;
        self.wdir = 0;
        self.orient_consistently(g_from);
        // (2) isolated minima and (3) double lines
        let mut tried: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut points: Vec<V3> = Vec::new();
        let mut visited: std::collections::HashSet<[i64; 3]> = std::collections::HashSet::new();
        let cell = self.cell;
        for &(base, step, v) in &touching {
            let s = if v.iter().any(|&x| x > 0.0) { 1.0 } else { -1.0 };
            let Some(ix) = self.local_min_corner(base, step, s) else { continue };
            if !tried.insert(key(ix)) {
                continue;
            }
            let cmax = v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let x0 = self.coord(ix);
            let near_visited = |q: V3, visited: &std::collections::HashSet<[i64; 3]>| {
                let k = [(q[0] / cell).floor() as i64, (q[1] / cell).floor() as i64, (q[2] / cell).floor() as i64];
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if visited.contains(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                                return true;
                            }
                        }
                    }
                }
                false
            };
            if near_visited(x0, &visited) {
                continue;
            }
            let done = 1e-9 * cmax;
            let m = {
                let p = self.p;
                let mut st: Vec<f64> = Vec::with_capacity(16);
                let mut f = |x: &V3| s * p.eval_with(x, &mut st);
                newton_min::<3>(&mut f, x0, 0.05 * cell, 0.7 * cell, done, 40)
            };
            // a real touch: F ~ 0 (or dipping below it inside the cell) AND flat there; a
            // sign-changing surface seen from a cell without neighbours has a large gradient
            let gn = m.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if !(m.val <= done) || gn * cell > 1e-3 * cmax {
                continue;
            }
            if sub(m.x, x0).iter().any(|d| d.abs() > 1.6 * cell) {
                continue; // wandered off: another cell owns that minimum
            }
            match m.rank() {
                3 => {
                    let dup = points.iter().any(|q| dot(sub(*q, m.x), sub(*q, m.x)) < (0.5 * cell) * (0.5 * cell));
                    if !dup {
                        points.push(m.x);
                        self.octahedron(m.x, 0.8 * cell);
                    }
                }
                2 => {
                    let line = self.trace_line(s, m.x, m.flattest(), &mut visited);
                    self.tube(&line, 0.4 * cell);
                }
                _ => {}
            }
        }
    }

    /// Follows a double line from `q0` along `t0` in both directions (damped Newton keeps the
    /// points on the line); returns the polyline inside the window.
    fn trace_line(&mut self, s: f64, q0: V3, t0: V3, visited: &mut std::collections::HashSet<[i64; 3]>) -> Vec<V3> {
        let cell = self.cell;
        let lo = self.min;
        let hi = [self.min[0] + self.size[0], self.min[1] + self.size[1], self.min[2] + self.size[2]];
        let inside = |q: V3| (0..3).all(|a| q[a] >= lo[a] && q[a] <= hi[a]);
        let mut dirs: [Vec<V3>; 2] = [Vec::new(), Vec::new()];
        for (di, sign) in [1.0f64, -1.0].iter().enumerate() {
            let mut q = q0;
            let mut t = [t0[0] * sign, t0[1] * sign, t0[2] * sign];
            let max_steps = (self.grid as usize * 4).min(20_000);
            for n in 0..max_steps {
                let guess = [q[0] + t[0] * cell, q[1] + t[1] * cell, q[2] + t[2] * cell];
                let m = {
                    let p = self.p;
                    let mut st: Vec<f64> = Vec::with_capacity(16);
                    let mut f = |x: &V3| s * p.eval_with(x, &mut st);
                    newton_min::<3>(&mut f, guess, 0.05 * cell, 0.7 * cell, 1e-18, 12)
                };
                // the line is where s*F is ~0 relative to the curvature scale
                let lam = m.eig.iter().fold(0.0f64, |a, &b| a.max(b));
                if !(lam > 0.0) || m.val > 1e-6 * lam * cell * cell || m.rank() != 2 {
                    break;
                }
                let nt = m.flattest();
                let sg = if dot(nt, t) < 0.0 { -1.0 } else { 1.0 };
                t = [nt[0] * sg, nt[1] * sg, nt[2] * sg];
                if !inside(m.x) {
                    break;
                }
                if n > 4 && dot(sub(m.x, q0), sub(m.x, q0)) < cell * cell {
                    break; // closed curve
                }
                q = m.x;
                dirs[di].push(q);
            }
        }
        let mut line: Vec<V3> = dirs[1].iter().rev().copied().collect();
        line.push(q0);
        line.extend(dirs[0].iter().copied());
        for q in &line {
            visited.insert([(q[0] / cell).floor() as i64, (q[1] / cell).floor() as i64, (q[2] / cell).floor() as i64]);
        }
        line
    }
}

/// Zero set of `f(x, y, z) = 0` as an indexed triangle mesh (program variables `[x, y, z]`).
///
/// An octree over `[min, max]` is refined level by level; cells whose interval enclosure of `f`
/// excludes zero are discarded, the rest become leaves at `max_depth` (at most 20; the grid is
/// `2^max_depth` cells per axis). Leaves are polygonised with marching tetrahedra (see the module
/// docs), vertices are placed by a bracketed secant search on the cell edges and shared between
/// triangles, and normals are the normalised gradient of `f`, so they point towards increasing
/// `f`; triangles are wound consistently with them. A sign change that is a pole or a jump
/// (`tan`, `1/x`, `floor`) draws nothing. Where `f` touches zero without changing sign (point and
/// line quadrics, `(g)^2 = 0`) the touching set is drawn: a double surface as a surface, a double
/// line as a thin tube and an isolated point as a small ball. `max_cells` bounds the number of
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
    // octree, breadth first; cells are (ix, iy, iz) at the current depth
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let mut evals = 0usize;
    let mut leaves: Vec<([u64; 3], u32)> = Vec::new();
    let mut cur: Vec<[u64; 3]> = vec![[0, 0, 0]];
    let mut depth = 0u32;
    while !cur.is_empty() {
        let mut next = Vec::with_capacity(cur.len() * 2);
        let n = (1u64 << depth) as f64;
        for &c in &cur {
            if evals >= max_cells {
                leaves.push((c, depth));
                continue;
            }
            evals += 1;
            let mut bx = [Interval::EMPTY; 3];
            for a in 0..3 {
                bx[a] = Interval::new(min[a] + size[a] * c[a] as f64 / n, min[a] + size[a] * (c[a] + 1) as f64 / n);
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
    let mut b = Surf {
        p,
        st: Vec::with_capacity(16),
        min,
        size,
        grid,
        cell,
        gstep: cell * 0.05,
        vals: FxMap::default(),
        vval: Vec::new(),
        evert: Vec::new(),
        cvert: Vec::new(),
        gvals: Default::default(),
        edges: FxMap::default(),
        gedges: Default::default(),
        wdir: 0,
        rejected: false,
        quant: FxMap::default(),
        qscale: cell * 1e-6,
        pos: Vec::new(),
        nrm: Vec::new(),
        facen: Vec::new(),
        idx: Vec::new(),
        gmode: false,
        tau: 0.0,
    };
    b.vals.reserve(leaves.len() * 2);
    b.vval.reserve(leaves.len() * 2);
    b.evert.reserve(leaves.len() * 2);
    b.cvert.reserve(leaves.len() * 2);
    b.pos.reserve(leaves.len() * 2);
    b.nrm.reserve(leaves.len() * 2);
    b.facen.reserve(leaves.len() * 2);
    b.idx.reserve(leaves.len() * 12);
    let mut cands: Vec<([u64; 3], u64, [f64; 8])> = Vec::new();
    for (c, d) in leaves {
        let step = 1u64 << (depth_max - d);
        let base = [c[0] * step, c[1] * step, c[2] * step];
        let mut v = [0.0f64; 8];
        let mut ids = [0u32; 8];
        let mut nan = false;
        for (k, slot) in v.iter_mut().enumerate() {
            let k = k as u64;
            let ix = [base[0] + (k & 1) * step, base[1] + ((k >> 1) & 1) * step, base[2] + ((k >> 2) & 1) * step];
            let (id, val) = b.corner_id(ix);
            ids[k as usize] = id;
            *slot = val;
            nan |= val.is_nan();
        }
        if nan {
            continue;
        }
        let (npos, nneg) = (v.iter().filter(|&&x| x > 0.0).count(), v.iter().filter(|&&x| x < 0.0).count());
        if npos != 0 && npos != 8 {
            b.march(base, step, &ids, &v);
        }
        if (npos == 0 || nneg == 0) && npos + nneg > 0 {
            cands.push((base, step, v)); // one sign (or zero) everywhere: maybe a touch
        }
    }
    b.touch_pass(&cands);
    // finalise normals: fall back to the accumulated face normal where the gradient vanished
    // (only vertices without a gradient normal need it)
    for t in b.idx.chunks_exact(3) {
        if t.iter().all(|&i| b.nrm[i as usize] != [0.0; 3]) {
            continue;
        }
        let (pa, pb, pc) = (b.pos[t[0] as usize], b.pos[t[1] as usize], b.pos[t[2] as usize]);
        let fnrm = cross(sub(pb, pa), sub(pc, pa));
        for &i in t {
            if b.nrm[i as usize] == [0.0; 3] {
                for k in 0..3 {
                    b.facen[i as usize][k] += fnrm[k];
                }
            }
        }
    }
    let mut mesh = Mesh::default();
    mesh.positions.reserve(b.pos.len());
    mesh.normals.reserve(b.pos.len());
    for i in 0..b.pos.len() {
        let n = if dot(b.nrm[i], b.nrm[i]) > 0.5 { b.nrm[i] } else { normalize(b.facen[i]).unwrap_or([0.0, 0.0, 1.0]) };
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

    fn prog(src: &str) -> Program {
        compile(&parse(src).unwrap(), &["x", "y", "z"], Angle::Rad).unwrap()
    }

    fn mesh(src: &str, w: f64, depth: u32) -> Mesh {
        surface_3d(&prog(src), [-w; 3], [w; 3], depth, 400_000)
    }

    fn area(m: &Mesh) -> f64 {
        m.indices
            .chunks(3)
            .map(|t| {
                let q: Vec<V3> = t.iter().map(|&i| m.positions[i as usize].map(|c| c as f64)).collect();
                let c = cross(sub(q[1], q[0]), sub(q[2], q[0]));
                0.5 * dot(c, c).sqrt()
            })
            .sum()
    }

    fn flips(m: &Mesh) -> usize {
        m.indices
            .chunks(3)
            .filter(|t| {
                let q: Vec<V3> = t.iter().map(|&i| m.positions[i as usize].map(|c| c as f64)).collect();
                let f = cross(sub(q[1], q[0]), sub(q[2], q[0]));
                let mut g = [0.0; 3];
                for &i in *t {
                    for k in 0..3 {
                        g[k] += m.normals[i as usize][k] as f64;
                    }
                }
                dot(f, g) < 0.0
            })
            .count()
    }

    fn unit_normals(m: &Mesh) -> bool {
        m.normals.iter().all(|n| ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-3)
    }

    #[test]
    fn point_quadric_is_a_small_ball_at_the_point() {
        for src in ["(x-0.37)^2+(y-0.21)^2+(z-0.11)^2", "x^2+y^2+z^2", "(x-1.1)^2+2(y+0.7)^2+(z-2)^2"] {
            let m = mesh(src, 6.0, 6);
            assert!(!m.indices.is_empty(), "{src}: nothing drawn");
            let c = m.positions.iter().fold([0.0f64; 3], |a, p| [a[0] + p[0] as f64, a[1] + p[1] as f64, a[2] + p[2] as f64]);
            let c = [c[0] / m.positions.len() as f64, c[1] / m.positions.len() as f64, c[2] / m.positions.len() as f64];
            let want = if src.starts_with("(x-1.1)") { [1.1, -0.7, 2.0] } else if src.starts_with("(x-0.37)") { [0.37, 0.21, 0.11] } else { [0.0; 3] };
            assert!(dot(sub(c, want), sub(c, want)).sqrt() < 0.02, "{src}: centre {c:?}");
            for p in &m.positions {
                let d = sub(p.map(|c| c as f64), want);
                assert!(dot(d, d).sqrt() < 0.25, "{src}: vertex far from the point");
            }
            assert!(unit_normals(&m) && flips(&m) == 0);
        }
    }

    #[test]
    fn double_shell_is_the_sphere_once() {
        for src in ["(x^2+y^2+z^2-9)^2", "(x^2+y^2+z^2-9)^2+0", "-(x^2+y^2+z^2-9)^2"] {
            let m = mesh(src, 4.0, 6);
            assert!(!m.indices.is_empty(), "{src}");
            for p in &m.positions {
                let r = (p[0] as f64).hypot(p[1] as f64).hypot(p[2] as f64);
                assert!((r - 3.0).abs() < 2e-3, "{src}: radius {r}");
            }
            let a = area(&m);
            assert!((a - 36.0 * std::f64::consts::PI).abs() < 0.06 * 36.0 * std::f64::consts::PI, "{src}: area {a}");
            assert!(unit_normals(&m) && flips(&m) == 0, "{src}");
        }
    }

    /// Open (used once) edges of a mesh after welding vertices that coincide to 1e-4.
    fn open_edges(m: &Mesh) -> usize {
        let mut weld: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
        let ids: Vec<u32> = m
            .positions
            .iter()
            .map(|p| {
                let k = [(p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64];
                let n = weld.len() as u32;
                *weld.entry(k).or_insert(n)
            })
            .collect();
        let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        for t in m.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (ids[t[k] as usize], ids[t[(k + 1) % 3] as usize]);
                if a != b {
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
        }
        edges.values().filter(|&&n| n == 1).count()
    }

    #[test]
    fn double_shell_has_no_slits() {
        // the zero set of w . grad F is a second surface crossing the shell; cells there are redone
        // with another direction, so the closed shell has (almost) no open edges and is wound
        // consistently
        for src in ["(x^2+y^2+z^2-9)^2", "(x^2/4+y^2+z^2/9-1)^2"] {
            let m = mesh(src, 4.0, 6);
            let open = open_edges(&m);
            assert!(open * 200 < m.indices.len() / 3, "{src}: {open} open edges of {} triangles", m.indices.len() / 3);
            // consistent winding: outward everywhere or inward everywhere, never a mix
            let f = prog(src);
            let mut out = 0usize;
            let mut inn = 0usize;
            for t in m.indices.chunks(3) {
                let q: Vec<V3> = t.iter().map(|&i| m.positions[i as usize].map(|c| c as f64)).collect();
                let n = cross(sub(q[1], q[0]), sub(q[2], q[0]));
                let c = [(q[0][0] + q[1][0] + q[2][0]) / 3.0, (q[0][1] + q[1][1] + q[2][1]) / 3.0, (q[0][2] + q[1][2] + q[2][2]) / 3.0];
                let _ = &f;
                if dot(n, c) > 0.0 {
                    out += 1;
                } else {
                    inn += 1;
                }
            }
            assert!(out.min(inn) * 100 < out + inn, "{src}: {out} outward, {inn} inward");
        }
    }

    #[test]
    fn line_quadric_is_a_thin_tube_along_the_line() {
        let m = mesh("(x-0.37)^2+(y-0.21)^2", 6.0, 6);
        assert!(!m.indices.is_empty());
        let (mut zmin, mut zmax) = (f64::MAX, f64::MIN);
        for p in &m.positions {
            let d = ((p[0] as f64 - 0.37).powi(2) + (p[1] as f64 - 0.21).powi(2)).sqrt();
            assert!(d < 0.1, "vertex {d} from the line");
            zmin = zmin.min(p[2] as f64);
            zmax = zmax.max(p[2] as f64);
        }
        assert!(zmin < -5.5 && zmax > 5.5, "tube covers z {zmin}..{zmax}");
        assert!(unit_normals(&m) && flips(&m) == 0);
        // an oblique double line (x-y)^2 + (y-z)^2 = 0 (the diagonal)
        let m = mesh("(x-y)^2+(y-z)^2", 6.0, 6);
        assert!(!m.indices.is_empty());
        for p in &m.positions {
            let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
            assert!((x - y).abs() < 0.12 && (y - z).abs() < 0.12, "{p:?}");
        }
    }

    #[test]
    fn a_sphere_smaller_than_a_cell_is_a_ball() {
        // radius 0.01 at (0.03, 0.02, 0.04), between lattice points 0.19 apart: every corner is positive
        let m = mesh("(x-0.03)^2+(y-0.02)^2+(z-0.04)^2-0.0001", 3.0, 5);
        assert!(!m.indices.is_empty());
        assert!(m.positions.iter().all(|p| ((p[0] as f64 - 0.03).powi(2) + (p[1] as f64 - 0.02).powi(2) + (p[2] as f64 - 0.04).powi(2)).sqrt() < 0.2));
    }

    #[test]
    fn ordinary_surfaces_get_no_extra_geometry() {
        for src in ["x^2+y^2+z^2-16", "x^2+y^2-z^2", "z-x^2/4+y^2/4", "(sqrt(x^2+y^2)-3)^2+z^2-1"] {
            let m = mesh(src, 6.0, 6);
            let f = prog(src);
            for p in &m.positions {
                let q = p.map(|c| c as f64);
                let h = 1e-3;
                let mut g2 = 0.0;
                for a in 0..3 {
                    let (mut u, mut d) = (q, q);
                    u[a] += h;
                    d[a] -= h;
                    let gv = (f.eval(&u) - f.eval(&d)) / (2.0 * h);
                    g2 += gv * gv;
                }
                let dist = f.eval(&q).abs() / g2.sqrt().max(1e-9);
                assert!(dist < 1e-3, "{src}: vertex {dist} off the surface");
            }
            assert_eq!(flips(&m), 0, "{src}");
        }
    }

    #[test]
    fn poles_and_jumps_leave_gaps_instead_of_sheets() {
        // z = tan(x) as the implicit z - tan(x): no triangle spans a pole
        let m = mesh("z-tan(x)", 6.0, 6);
        assert!(!m.indices.is_empty());
        let branch = |x: f32| ((x as f64 - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).floor();
        for t in m.indices.chunks(3) {
            let b: Vec<f64> = t.iter().map(|&i| branch(m.positions[i as usize][0])).collect();
            assert!(b.iter().all(|&x| x == b[0]), "triangle spans a pole: {b:?}");
        }
        // z = floor(x): treads only
        let m = mesh("z-floor(x)", 6.0, 6);
        assert!(!m.indices.is_empty());
        for t in m.indices.chunks(3) {
            let zs: Vec<f32> = t.iter().map(|&i| m.positions[i as usize][2]).collect();
            let r = zs.iter().cloned().fold(f32::MIN, f32::max) - zs.iter().cloned().fold(f32::MAX, f32::min);
            assert!(r < 1e-3, "a triangle spans {r} of height");
        }
        // y = 1/x as the implicit y - 1/x: the two branches are separate sheets
        let m = mesh("y-1/x", 6.0, 6);
        for t in m.indices.chunks(3) {
            let s: Vec<bool> = t.iter().map(|&i| m.positions[i as usize][0] < 0.0).collect();
            assert!(s.iter().all(|&x| x == s[0]), "triangle joins the two branches");
        }
    }
}
