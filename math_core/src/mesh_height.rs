//! Explicit surfaces `h = f(u, v)` (`z = f(x, y)`, `y = f(x, z)`, `x = f(y, z)`) as a height
//! field: one function value per grid node instead of an implicit search, so these graphs cost a
//! few thousand evaluations rather than hundreds of thousands, and the mesh can be broken exactly
//! where the graph is.
//!
//! * a quad touching a point where `f` is undefined (NaN, infinite, absurdly large) is trimmed
//!   to the defined part: the edge between a defined and an undefined node gets a rim vertex at
//!   the domain boundary (found by bisection), so `sqrt(16 - x^2 - y^2)` ends in a clean rim;
//! * a quad whose interval enclosure of `f` is unbounded contains a pole (`tan(x)`, `1/(x^2+y^2)`)
//!   and is dropped, so the two branches are never joined by a vertical sheet;
//! * a jump between neighbouring samples that is large against the window height and against
//!   the neighbouring steps (`floor(x)`, `sign(x)`) removes the quads across it, so there are no
//!   walls between treads;
//! * triangles are clipped to the height range of the window;
//! * triangles are wound counter-clockwise seen from `+h`, so normals (which point to `+h`, towards
//!   increasing `h - f`) and winding agree by construction.

use crate::compile::Program;
use crate::interval::Interval;
use crate::mesh::Mesh;
use crate::mesh_surface::{cross, dot, sub, V3};
use crate::mesh_touch::FxMap;

const UNSET: u32 = u32::MAX;
/// Fine lattice steps per grid cell (the largest refinement of a steep quad).
const SUB: usize = 8;
/// A sub-quad's polygon: 4 corners plus at most 7 extra nodes on each of 4 edges.
const MAX_RING: usize = 32;

#[derive(Default)]
struct Ring {
    p: [(usize, usize); MAX_RING],
    len: usize,
}

impl Ring {
    fn push(&mut self, v: (usize, usize)) {
        self.p[self.len] = v;
        self.len += 1;
    }
}

fn parity_odd(p: [usize; 3]) -> bool {
    let mut inv = 0;
    for i in 0..3 {
        for j in i + 1..3 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 1
}

/// Height-field mesh of `h = f(u, v)` over `[min, max]` (world coordinates). `axes = [au, av,
/// ah]` says which world axis `u`, `v` and `h` are; `p` takes the variables `[u, v]`. The grid
/// has `2^depth` cells per axis. The result is in world coordinates, clipped to the box.
pub fn height_surface(p: &Program, axes: [usize; 3], min: [f64; 3], max: [f64; 3], depth: u32) -> Mesh {
    let [au, av, ah] = axes;
    let ok = (0..3).all(|a| min[a].is_finite() && max[a].is_finite() && max[a] > min[a]);
    if !ok {
        return Mesh::default();
    }
    let n = 1usize << depth.clamp(1, 11);
    let (u0, v0) = (min[au], min[av]);
    let (du, dv) = ((max[au] - min[au]) / n as f64, (max[av] - min[av]) / n as f64);
    let (hlo, hhi) = (min[ah], max[ah]);
    let hspan = hhi - hlo;
    let huge = 1e6 * hspan + hlo.abs().max(hhi.abs());
    let nn = n + 1;
    let mut st: Vec<f64> = Vec::with_capacity(16);
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let eval = |u: f64, v: f64, st: &mut Vec<f64>| -> f64 {
        let h = p.eval_with(&[u, v], st);
        if h.is_finite() && h.abs() <= huge {
            h
        } else {
            f64::NAN
        }
    };
    // samples
    let mut z = vec![f64::NAN; nn * nn];
    for j in 0..nn {
        let v = v0 + dv * j as f64;
        for i in 0..nn {
            z[j * nn + i] = eval(u0 + du * i as f64, v, &mut st);
        }
    }
    let node = |i: usize, j: usize| j * nn + i;

    // jumps between neighbouring finite samples
    let thr = 0.03 * hspan;
    let mut jump_u = vec![false; nn * nn]; // edge (i,j)-(i+1,j)
    let mut jump_v = vec![false; nn * nn]; // edge (i,j)-(i,j+1)
    let step = |a: f64, b: f64| if a.is_finite() && b.is_finite() { (a - b).abs() } else { f64::NAN };
    for j in 0..nn {
        for i in 0..nn {
            let a = z[node(i, j)];
            if i + 1 < nn {
                let d = step(a, z[node(i + 1, j)]);
                if d > thr {
                    let prev = if i > 0 { step(z[node(i - 1, j)], a) } else { f64::NAN };
                    let next = if i + 2 < nn { step(z[node(i + 1, j)], z[node(i + 2, j)]) } else { f64::NAN };
                    let m = [prev, next].iter().fold(0.0f64, |m, &x| if x.is_nan() { m } else { m.max(x) });
                    jump_u[node(i, j)] = d > 5.0 * m;
                }
            }
            if j + 1 < nn {
                let d = step(a, z[node(i, j + 1)]);
                if d > thr {
                    let prev = if j > 0 { step(z[node(i, j - 1)], a) } else { f64::NAN };
                    let next = if j + 2 < nn { step(z[node(i, j + 1)], z[node(i, j + 2)]) } else { f64::NAN };
                    let m = [prev, next].iter().fold(0.0f64, |m, &x| if x.is_nan() { m } else { m.max(x) });
                    jump_v[node(i, j)] = d > 5.0 * m;
                }
            }
        }
    }

    // poles: quads whose interval enclosure is unbounded
    let mut pole = vec![false; n * n];
    {
        fn block(
            p: &Program,
            ist: &mut Vec<Interval>,
            pole: &mut [bool],
            n: usize,
            org: (f64, f64),
            d: (f64, f64),
            (i0, j0, size): (usize, usize, usize),
        ) {
            let iu = Interval::new(org.0 + d.0 * i0 as f64, org.0 + d.0 * (i0 + size) as f64);
            let iv = Interval::new(org.1 + d.1 * j0 as f64, org.1 + d.1 * (j0 + size) as f64);
            let r = p.eval_interval_with(&[iu, iv], ist);
            if r.is_empty() || (r.lo.is_finite() && r.hi.is_finite()) {
                return;
            }
            if size == 1 {
                pole[j0 * n + i0] = true;
                return;
            }
            let h = size / 2;
            for (a, b) in [(0, 0), (h, 0), (0, h), (h, h)] {
                block(p, ist, pole, n, org, d, (i0 + a, j0 + b, h));
            }
        }
        let b0 = n.min(16);
        for j in (0..n).step_by(b0) {
            for i in (0..n).step_by(b0) {
                block(p, &mut ist, &mut pole, n, (u0, v0), (du, dv), (i, j, b0));
            }
        }
    }

    // vertices: nodes (finite ones), rim vertices, clip vertices; normals accumulated later
    let mut pos: Vec<V3> = Vec::with_capacity(nn * nn);
    for j in 0..nn {
        for i in 0..nn {
            pos.push([u0 + du * i as f64, v0 + dv * j as f64, z[node(i, j)]]);
        }
    }
    let mut rim: std::collections::HashMap<(usize, usize), u32> = std::collections::HashMap::new();
    let mut rim_vertex = |a: usize, b: usize, pos: &mut Vec<V3>, st: &mut Vec<f64>| -> u32 {
        let (ia, ib) = if pos[a][2].is_finite() { (a, b) } else { (b, a) }; // ia finite, ib undefined
        let key = (ia.min(ib), ia.max(ib));
        if let Some(&r) = rim.get(&key) {
            return r;
        }
        let (pa, pb) = (pos[ia], pos[ib]);
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let mut best = (pa[0], pa[1], pa[2]);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            let (uu, vv) = (pa[0] + (pb[0] - pa[0]) * m, pa[1] + (pb[1] - pa[1]) * m);
            let h = eval(uu, vv, st);
            if h.is_finite() {
                lo = m;
                best = (uu, vv, h);
            } else {
                hi = m;
            }
        }
        pos.push([best.0, best.1, best.2]);
        let id = (pos.len() - 1) as u32;
        rim.insert(key, id);
        id
    };

    // Steep quads are refined: a quad whose corner heights differ by more than ~1.5 cells is
    // split into k x k sub-quads (k = 2, 4 or 8) whose extra nodes are evaluated, so the 3D
    // size of the triangles stays about one cell on steep walls. Edges shared with a coarser
    // neighbour get that neighbour's missing nodes as extra polygon vertices, so no T-junction
    // cracks open.
    let cell3 = du.max(dv);
    let mut kq = vec![1usize; n * n];
    let skip = |i: usize, j: usize| -> bool {
        pole[j * n + i] || jump_u[node(i, j)] || jump_v[node(i + 1, j)] || jump_u[node(i, j + 1)] || jump_v[node(i, j)]
    };
    for j in 0..n {
        for i in 0..n {
            if skip(i, j) {
                continue;
            }
            let c = [z[node(i, j)], z[node(i + 1, j)], z[node(i + 1, j + 1)], z[node(i, j + 1)]];
            let nfin = c.iter().filter(|v| v.is_finite()).count();
            if nfin == 0 {
                continue;
            }
            if nfin < 4 {
                // the rim of the domain: usually steep (sqrt), so sample it finely
                kq[j * n + i] = SUB;
                continue;
            }
            let (mn, mx) = (c.iter().cloned().fold(f64::MAX, f64::min), c.iter().cloned().fold(f64::MIN, f64::max));
            if mx < hlo || mn > hhi {
                continue; // outside the window: clipped away anyway
            }
            let need = (mx - mn) / (1.5 * cell3);
            kq[j * n + i] = if need <= 1.0 {
                1
            } else if need <= 2.0 {
                2
            } else if need <= 4.0 {
                4
            } else {
                8
            };
        }
    }
    let k_of = |i: isize, j: isize, own: usize| -> usize {
        if i < 0 || j < 0 || i >= n as isize || j >= n as isize || skip(i as usize, j as usize) {
            own
        } else {
            kq[j as usize * n + i as usize]
        }
    };
    let mut fine: FxMap<(usize, usize), u32> = FxMap::default();
    let mut node_id = |ii: usize, jj: usize, pos: &mut Vec<V3>, st: &mut Vec<f64>| -> u32 {
        if ii % SUB == 0 && jj % SUB == 0 {
            return node(ii / SUB, jj / SUB) as u32;
        }
        if let Some(&id) = fine.get(&(ii, jj)) {
            return id;
        }
        let (uu, vv) = (u0 + du * (ii as f64 / SUB as f64), v0 + dv * (jj as f64 / SUB as f64));
        let h = eval(uu, vv, st);
        pos.push([uu, vv, h]);
        let id = (pos.len() - 1) as u32;
        fine.insert((ii, jj), id);
        id
    };
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            if skip(i, j) {
                continue;
            }
            let k = kq[j * n + i];
            let (ii, jj) = (i as isize, j as isize);
            let kb = k_of(ii, jj - 1, k).max(k); // bottom neighbour
            let kr = k_of(ii + 1, jj, k).max(k);
            let kt = k_of(ii, jj + 1, k).max(k);
            let kl = k_of(ii - 1, jj, k).max(k);
            let s = SUB / k;
            for b in 0..k {
                for a in 0..k {
                    let (i0, j0) = (i * SUB + a * s, j * SUB + b * s);
                    // polygon around the sub-quad: corners plus the finer neighbours' nodes on
                    // the edges that lie on the parent quad's boundary
                    let mut ring: Ring = Ring::default();
                    ring.push((i0, j0));
                    if b == 0 && kb > k {
                        let step = SUB / kb;
                        for t in 1..kb / k {
                            ring.push((i0 + t * step, j0));
                        }
                    }
                    ring.push((i0 + s, j0));
                    if a == k - 1 && kr > k {
                        let step = SUB / kr;
                        for t in 1..kr / k {
                            ring.push((i0 + s, j0 + t * step));
                        }
                    }
                    ring.push((i0 + s, j0 + s));
                    if b == k - 1 && kt > k {
                        let step = SUB / kt;
                        for t in (1..kt / k).rev() {
                            ring.push((i0 + t * step, j0 + s));
                        }
                    }
                    ring.push((i0, j0 + s));
                    if a == 0 && kl > k {
                        let step = SUB / kl;
                        for t in (1..kl / k).rev() {
                            ring.push((i0, j0 + t * step));
                        }
                    }
                    let mut ids = [0u32; MAX_RING];
                    let mut fin = [false; MAX_RING];
                    let m = ring.len;
                    for q in 0..m {
                        let (x, y) = ring.p[q];
                        ids[q] = node_id(x, y, &mut pos, &mut st);
                        fin[q] = pos[ids[q] as usize][2].is_finite();
                    }
                    let (ids, fin) = (&ids[..m], &fin[..m]);
                    let nfin = fin.iter().filter(|&&f| f).count();
                    if nfin == 0 {
                        continue;
                    }
                    if nfin == ids.len() {
                        if ids.len() == 4 {
                            let h = |q: usize| pos[ids[q] as usize][2];
                            if (h(0) - h(2)).abs() <= (h(1) - h(3)).abs() {
                                tris.push([ids[0], ids[1], ids[2]]);
                                tris.push([ids[0], ids[2], ids[3]]);
                            } else {
                                tris.push([ids[0], ids[1], ids[3]]);
                                tris.push([ids[1], ids[2], ids[3]]);
                            }
                        } else {
                            for t in 1..ids.len() - 1 {
                                tris.push([ids[0], ids[t], ids[t + 1]]);
                            }
                        }
                        continue;
                    }
                    let mut poly: Vec<u32> = Vec::with_capacity(ids.len() + 4);
                    for q in 0..ids.len() {
                        let q1 = (q + 1) % ids.len();
                        if fin[q] {
                            poly.push(ids[q]);
                        }
                        if fin[q] != fin[q1] {
                            poly.push(rim_vertex(ids[q] as usize, ids[q1] as usize, &mut pos, &mut st));
                        }
                    }
                    for t in 1..poly.len().saturating_sub(1) {
                        tris.push([poly[0], poly[t], poly[t + 1]]);
                    }
                }
            }
        }
    }
    // wind counter-clockwise in (u, v) (normal towards +h) and drop degenerate triangles
    tris.retain_mut(|t| {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area == 0.0 || area.is_nan() {
            return false;
        }
        if area < 0.0 {
            t.swap(1, 2);
        }
        true
    });
    // vertex normals (area weighted over the unclipped triangles), pointing to +h
    let mut nrm: Vec<V3> = vec![[0.0; 3]; pos.len()];
    for t in &tris {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let fnrm = cross(sub(b, a), sub(c, a));
        for &i in t {
            for k in 0..3 {
                nrm[i as usize][k] += fnrm[k];
            }
        }
    }
    // clip to the height range
    let mut out: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    let ph = |i: u32, pos: &Vec<V3>| pos[i as usize][2];
    let lerp_vertex = |a: u32, b: u32, h: f64, pos: &mut Vec<V3>, nrm: &mut Vec<V3>| -> u32 {
        let (pa, pb) = (pos[a as usize], pos[b as usize]);
        let t = ((h - pa[2]) / (pb[2] - pa[2])).clamp(0.0, 1.0);
        let q = [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, h];
        let un = |v: V3| {
            let l = dot(v, v).sqrt();
            if l > 0.0 {
                [v[0] / l, v[1] / l, v[2] / l]
            } else {
                [0.0, 0.0, 1.0]
            }
        };
        let (na, nb) = (un(nrm[a as usize]), un(nrm[b as usize]));
        let nl = [na[0] + (nb[0] - na[0]) * t, na[1] + (nb[1] - na[1]) * t, na[2] + (nb[2] - na[2]) * t];
        pos.push(q);
        nrm.push(un(nl));
        (pos.len() - 1) as u32
    };
    for t in &tris {
        let hs = [ph(t[0], &pos), ph(t[1], &pos), ph(t[2], &pos)];
        let (mn, mx) = (hs.iter().cloned().fold(f64::MAX, f64::min), hs.iter().cloned().fold(f64::MIN, f64::max));
        if mx < hlo || mn > hhi {
            continue;
        }
        if mn >= hlo && mx <= hhi {
            out.push(*t);
            continue;
        }
        // Sutherland-Hodgman against h <= hhi, then h >= hlo
        let mut poly: Vec<u32> = t.to_vec();
        for (plane, keep_below) in [(hhi, true), (hlo, false)] {
            let inside = |h: f64| if keep_below { h <= plane } else { h >= plane };
            let mut res: Vec<u32> = Vec::with_capacity(poly.len() + 2);
            for k in 0..poly.len() {
                let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
                let (ha, hb) = (ph(a, &pos), ph(b, &pos));
                if inside(ha) {
                    res.push(a);
                }
                if inside(ha) != inside(hb) {
                    res.push(lerp_vertex(a, b, plane, &mut pos, &mut nrm));
                }
            }
            poly = res;
            if poly.len() < 3 {
                break;
            }
        }
        for k in 1..poly.len().saturating_sub(1) {
            out.push([poly[0], poly[k], poly[k + 1]]);
        }
    }
    // emit in world coordinates, compacting the vertices that are used
    let odd = parity_odd(axes);
    let mut remap = vec![UNSET; pos.len()];
    let mut mesh = Mesh::default();
    for t in &out {
        let mut ids = [0u32; 3];
        for k in 0..3 {
            let v = t[k] as usize;
            if remap[v] == UNSET {
                remap[v] = mesh.positions.len() as u32;
                let q = pos[v];
                let mut w = [0.0f32; 3];
                w[au] = q[0] as f32;
                w[av] = q[1] as f32;
                w[ah] = q[2] as f32;
                mesh.positions.push(w);
                let nv = nrm[v];
                let l = dot(nv, nv).sqrt();
                let nv = if l > 0.0 { [nv[0] / l, nv[1] / l, nv[2] / l] } else { [0.0, 0.0, 1.0] };
                let mut m = [0.0f32; 3];
                m[au] = nv[0] as f32;
                m[av] = nv[1] as f32;
                m[ah] = nv[2] as f32;
                mesh.normals.push(m);
            }
            ids[k] = remap[v];
        }
        if odd {
            mesh.indices.extend_from_slice(&[ids[0], ids[2], ids[1]]);
        } else {
            mesh.indices.extend_from_slice(&ids);
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Angle};
    use crate::parse::parse;

    fn prog(src: &str) -> Program {
        compile(&parse(src).unwrap(), &["x", "y"], Angle::Rad).unwrap()
    }

    const BOX: ([f64; 3], [f64; 3]) = ([-6.0; 3], [6.0; 3]);

    fn tri_normal_ok(m: &Mesh) -> usize {
        // number of triangles whose winding disagrees with the vertex normals
        let mut bad = 0;
        for t in m.indices.chunks(3) {
            let q: Vec<V3> = t.iter().map(|&i| m.positions[i as usize].map(|c| c as f64)).collect();
            let f = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            let mut g = [0.0; 3];
            for &i in t {
                for k in 0..3 {
                    g[k] += m.normals[i as usize][k] as f64;
                }
            }
            if dot(f, g) < 0.0 {
                bad += 1;
            }
        }
        bad
    }

    #[test]
    fn paraboloid_is_a_clean_graph() {
        let m = height_surface(&prog("x^2/4+y^2/9"), [0, 1, 2], BOX.0, BOX.1, 6);
        assert!(!m.indices.is_empty());
        for (p, n) in m.positions.iter().zip(&m.normals) {
            let z = (p[0] as f64).powi(2) / 4.0 + (p[1] as f64).powi(2) / 9.0;
            assert!((p[2] as f64 - z).abs() < 1e-5 || p[2] as f64 > 5.999, "{p:?}");
            assert!(n[2] > 0.0);
        }
        assert_eq!(tri_normal_ok(&m), 0);
    }

    #[test]
    fn tan_has_no_sheets_across_the_poles() {
        let m = height_surface(&prog("tan(x)"), [0, 1, 2], BOX.0, BOX.1, 7);
        assert!(!m.indices.is_empty());
        for t in m.indices.chunks(3) {
            // no triangle spans a pole: all its vertices are on one branch
            let br: Vec<f64> = t.iter().map(|&i| ((m.positions[i as usize][0] as f64 - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).floor()).collect();
            assert!(br.iter().all(|&b| b == br[0]), "{br:?}");
        }
        for p in &m.positions {
            assert!(p[2].abs() <= 6.0 + 1e-5);
        }
    }

    #[test]
    fn floor_has_no_walls() {
        let m = height_surface(&prog("floor(x)"), [0, 1, 2], BOX.0, BOX.1, 7);
        assert!(!m.indices.is_empty());
        for t in m.indices.chunks(3) {
            let zs: Vec<f32> = t.iter().map(|&i| m.positions[i as usize][2]).collect();
            let r = zs.iter().cloned().fold(f32::MIN, f32::max) - zs.iter().cloned().fold(f32::MAX, f32::min);
            assert!(r < 1e-4, "a triangle spans {r} of height");
        }
    }

    #[test]
    fn sqrt_rim_is_trimmed_to_the_domain_edge() {
        let m = height_surface(&prog("sqrt(16-x^2-y^2)"), [0, 1, 2], BOX.0, BOX.1, 7);
        assert!(!m.indices.is_empty());
        let mut rim = 0;
        for p in &m.positions {
            let r2 = (p[0] as f64).powi(2) + (p[1] as f64).powi(2);
            assert!(r2 <= 16.0 + 1e-6, "outside the domain: r2 = {r2}");
            assert!(((p[2] as f64).powi(2) - (16.0 - r2)).abs() < 1e-4);
            if p[2] < 1e-6 {
                rim += 1;
            }
        }
        assert!(rim > 100, "{rim} rim vertices");
        assert_eq!(tri_normal_ok(&m), 0);
    }

    #[test]
    fn other_axes_keep_their_normals_up() {
        // y = f(x, z) and x = f(y, z): the normal points to +y / +x (increasing y - f)
        let m = height_surface(&prog("x^2+y^2"), [0, 2, 1], BOX.0, BOX.1, 5);
        assert!(m.normals.iter().all(|n| n[1] > 0.0));
        assert_eq!(tri_normal_ok(&m), 0);
        let m = height_surface(&prog("x^2+y^2"), [1, 2, 0], BOX.0, BOX.1, 5);
        assert!(m.normals.iter().all(|n| n[0] > 0.0));
        assert_eq!(tri_normal_ok(&m), 0);
    }

    #[test]
    fn reciprocal_pole_is_cut_at_the_window_top() {
        let m = height_surface(&prog("1/(x^2+y^2)"), [0, 1, 2], BOX.0, BOX.1, 7);
        assert!(!m.indices.is_empty());
        assert!(m.positions.iter().all(|p| p[2] <= 6.0 + 1e-5));
    }
}
