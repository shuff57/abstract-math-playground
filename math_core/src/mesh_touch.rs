//! Shared helpers for the meshers: a fast integer hasher, a small symmetric eigen-solver and a
//! damped Newton minimiser. The minimiser is what finds the places where a function touches zero
//! without changing sign (point conics, double lines, point quadrics): there marching squares or
//! tetrahedra see no sign change at all, so the meshers look for the minimum of `|F|` themselves.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative hasher with a murmur finaliser; the lattice keys are small integers, for which
/// the default SipHash is several times slower than the work it guards.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Fx {
    #[inline]
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

pub(crate) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations. Returns the eigenvalues
/// and the eigenvectors as the COLUMNS of the second matrix (`v[row][k]` belongs to `vals[k]`).
pub(crate) fn eig_sym<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..24 {
        let mut off = 0.0;
        for p in 0..N {
            for q in p + 1..N {
                off += a[p][q].abs();
            }
        }
        if off < 1e-300 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..N {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut vals = [0.0; N];
    for i in 0..N {
        vals[i] = a[i][i];
    }
    (vals, v)
}

/// Result of [`newton_min`]: the minimum found, its value, and the curvature there.
pub(crate) struct Min<const N: usize> {
    pub x: [f64; N],
    pub val: f64,
    /// Gradient at `x`.
    pub grad: [f64; N],
    /// Eigenvalues of the Hessian of `sign * F` (ascending is not guaranteed).
    pub eig: [f64; N],
    /// Eigenvectors as columns (`vec[row][k]` belongs to `eig[k]`).
    pub vec: [[f64; N]; N],
}

impl<const N: usize> Min<N> {
    /// Number of curvature directions (eigenvalues above 0.1% of the largest).
    pub fn rank(&self) -> usize {
        let m = self.eig.iter().fold(0.0f64, |a, &b| a.max(b));
        if m <= 0.0 {
            return 0;
        }
        self.eig.iter().filter(|&&e| e > 1e-3 * m).count()
    }

    /// Unit eigenvector of the smallest-curvature direction.
    pub fn flattest(&self) -> [f64; N] {
        let mut k = 0;
        for i in 1..N {
            if self.eig[i] < self.eig[k] {
                k = i;
            }
        }
        let mut r = [0.0; N];
        for i in 0..N {
            r[i] = self.vec[i][k];
        }
        r
    }
}

/// Damped Newton minimisation of `f` starting at `x0` (finite-difference derivatives with step
/// `hd`). Directions of (near) zero curvature are left alone, so a ridge `F = d^2` converges to
/// the nearest point of the line. `done` stops early once `|f(x)| <= done`. Steps are at most
/// `max_step` long. Returns the final point whether or not it converged; the caller checks
/// `val`.
pub(crate) fn newton_min<const N: usize>(
    f: &mut dyn FnMut(&[f64; N]) -> f64,
    x0: [f64; N],
    hd: f64,
    max_step: f64,
    done: f64,
    max_iter: usize,
) -> Min<N> {
    let mut x = x0;
    let mut fx = f(&x);
    let mut eig = [0.0; N];
    let mut vec = [[0.0; N]; N];
    let mut grad = [0.0; N];
    for it in 0..=max_iter {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        let mut fp = [0.0; N];
        let mut fm = [0.0; N];
        for i in 0..N {
            let (mut a, mut b) = (x, x);
            a[i] += hd;
            b[i] -= hd;
            fp[i] = f(&a);
            fm[i] = f(&b);
            g[i] = (fp[i] - fm[i]) / (2.0 * hd);
            h[i][i] = (fp[i] - 2.0 * fx + fm[i]) / (hd * hd);
        }
        for i in 0..N {
            for j in i + 1..N {
                let (mut a, mut b, mut c, mut d) = (x, x, x, x);
                a[i] += hd;
                a[j] += hd;
                b[i] += hd;
                b[j] -= hd;
                c[i] -= hd;
                c[j] += hd;
                d[i] -= hd;
                d[j] -= hd;
                let v = (f(&a) - f(&b) - f(&c) + f(&d)) / (4.0 * hd * hd);
                h[i][j] = v;
                h[j][i] = v;
            }
        }
        let (ev, evec) = eig_sym(h);
        eig = ev;
        vec = evec;
        grad = g;
        if fx.abs() <= done || it == max_iter {
            break;
        }
        let top = ev.iter().fold(0.0f64, |a, &b| a.max(b));
        if top <= 0.0 {
            break;
        }
        // Newton step in the eigenbasis, skipping the flat and the concave directions.
        let mut d = [0.0; N];
        for k in 0..N {
            if ev[k] > 1e-3 * top {
                let mut gk = 0.0;
                for i in 0..N {
                    gk += evec[i][k] * g[i];
                }
                let s = -gk / ev[k];
                for i in 0..N {
                    d[i] += s * evec[i][k];
                }
            }
        }
        let len = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(len.is_finite() && len > 1e-9 * hd) {
            break; // converged (or stuck)
        }
        if len > max_step {
            for v in d.iter_mut() {
                *v *= max_step / len;
            }
        }
        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..10 {
            let mut xn = x;
            for i in 0..N {
                xn[i] += d[i] * scale;
            }
            let fnv = f(&xn);
            if fnv.is_finite() && fnv <= fx {
                x = xn;
                fx = fnv;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Min { x, val: fx, grad, eig, vec }
}
