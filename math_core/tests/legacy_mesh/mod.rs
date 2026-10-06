//! Legacy (pre-optimisation) marching-tetrahedra surface mesher, kept verbatim from the commit
//! before the rewrite so the benchmark can report before/after timings and compare outputs.
#![allow(dead_code)]
use math_core::compile::Program;
use math_core::interval::Interval;
use std::collections::HashMap;

/// Indexed triangle mesh: per-vertex positions and unit normals, and a triangle index list
/// (three indices per triangle, counter-clockwise when seen from the side the normal points to).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LegacyMesh {
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
pub fn legacy_surface_3d(p: &Program, min: [f64; 3], max: [f64; 3], max_depth: u32, max_cells: usize) -> LegacyMesh {
    let ok = (0..3).all(|a| min[a].is_finite() && max[a].is_finite() && max[a] > min[a]);
    if !ok {
        return LegacyMesh::default();
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
    let mut mesh = LegacyMesh::default();
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

