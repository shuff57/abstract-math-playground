//! Intersection points of two 2D curves inside a window.
//!
//! A curve is a predicate/function plus an optional clip, never a mesh, so a domain restriction
//! such as `{x>0}` can be honoured by passing it as [`Curve::clip`]. Supported shapes:
//!
//! * [`Shape::Explicit`]: `y = f(x)`
//! * [`Shape::Implicit`]: `F(x, y) = 0` (circles, conics, `x = g(y)` written as `x - g(y)`, ...)
//! * [`Shape::Param`]: `(x(t), y(t))` over `[t0, t1]` (polar curves are parametric in `theta`)
//!
//! How each pair is solved:
//!
//! * explicit/explicit, explicit/implicit, explicit/parametric, implicit/parametric: one
//!   variable. The pair collapses to `h(s) = 0` (`f - g`, `G(x, f(x))`, `y(t) - f(x(t))`,
//!   `F(x(t), y(t))`), which is sampled, bisected at sign changes, and searched for tangent
//!   touches (a local minimum of `|h|` that reaches zero without a sign change).
//! * implicit/implicit and parametric/parametric: two variables. The system `(P, Q) = (0, 0)` is
//!   sampled on a grid; every cell where both `P` and `Q` change sign is a candidate, which is
//!   refined with damped Gauss-Newton (which also converges on a tangent touch, where the
//!   Jacobian is singular, though only linearly).
//!
//! Results are deduplicated, sorted by x, and capped at [`MAX_HITS`]. Curves that coincide
//! (identical, or overlapping along a stretch) have no isolated intersections and give none.
//! Undefined regions (NaN, infinity, poles) are skipped, never crossed.

/// Most points returned for one pair of curves.
pub const MAX_HITS: usize = 48;
/// Samples for a one-variable scan.
const SAMPLES: usize = 4000;
/// Grid cells per side for a two-variable solve.
const GRID: usize = 160;

/// The visible window, `x0 < x1`, `y0 < y1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

impl Window {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
    fn valid(&self) -> bool {
        [self.x0, self.x1, self.y0, self.y1].iter().all(|v| v.is_finite()) && self.x1 > self.x0 && self.y1 > self.y0
    }
}

/// The geometry of a curve.
pub enum Shape<'a> {
    /// `y = f(x)`.
    Explicit(&'a dyn Fn(f64) -> f64),
    /// `F(x, y) = 0`.
    Implicit(&'a dyn Fn(f64, f64) -> f64),
    /// `(x(t), y(t))` for `t` in `[t0, t1]`.
    Param { x: &'a dyn Fn(f64) -> f64, y: &'a dyn Fn(f64) -> f64, t0: f64, t1: f64 },
}

/// A curve and the part of the plane it is restricted to.
pub struct Curve<'a> {
    pub shape: Shape<'a>,
    /// `Some(p)`: only points where `p(x, y)` is true belong to the curve (`{x>0}`).
    pub clip: Option<&'a dyn Fn(f64, f64) -> bool>,
}

impl<'a> Curve<'a> {
    pub fn new(shape: Shape<'a>) -> Self {
        Curve { shape, clip: None }
    }
    fn keeps(&self, x: f64, y: f64) -> bool {
        self.clip.is_none_or(|c| c(x, y))
    }
}

/// The intersection points of `a` and `b` inside `win`, sorted by x.
pub fn intersections(a: &Curve, b: &Curve, win: Window) -> Vec<(f64, f64)> {
    if !win.valid() {
        return Vec::new();
    }
    use Shape::*;
    let mut pts: Vec<(f64, f64)> = match (&a.shape, &b.shape) {
        (Explicit(f), Explicit(g)) => from_x(&|x| f(x) - g(x), f, win),
        (Explicit(f), Implicit(g)) | (Implicit(g), Explicit(f)) => from_x(&|x| g(x, f(x)), f, win),
        (Explicit(f), Param { x, y, t0, t1 }) | (Param { x, y, t0, t1 }, Explicit(f)) => {
            from_t(&|t| y(t) - f(x(t)), x, y, *t0, *t1)
        }
        (Implicit(f), Param { x, y, t0, t1 }) | (Param { x, y, t0, t1 }, Implicit(f)) => {
            from_t(&|t| f(x(t), y(t)), x, y, *t0, *t1)
        }
        (Implicit(f), Implicit(g)) => solve2(f, g, [win.x0, win.x1, win.y0, win.y1], &|u, v| Some((u, v))),
        (Param { x: x1, y: y1, t0: a0, t1: a1 }, Param { x: x2, y: y2, t0: b0, t1: b1 }) => solve2(
            &|t, s| x1(t) - x2(s),
            &|t, s| y1(t) - y2(s),
            [*a0, *a1, *b0, *b1],
            &|t, _| Some((x1(t), y1(t))),
        )
        .into_iter()
        .collect(),
    };
    // The two-variable param solve returns points in (t, s) already mapped to the plane by the
    // last argument; nothing more to do here.
    pts.retain(|&(x, y)| x.is_finite() && y.is_finite() && win.contains(x, y) && a.keeps(x, y) && b.keeps(x, y));
    dedupe(pts, win)
}

fn dedupe(mut pts: Vec<(f64, f64)>, win: Window) -> Vec<(f64, f64)> {
    pts.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
    let (tx, ty) = (2e-6 * (win.x1 - win.x0), 2e-6 * (win.y1 - win.y0));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for p in pts {
        // Compare with every kept point whose x is within tolerance (sorted by x).
        let dup = out.iter().rev().take_while(|q| p.0 - q.0 <= tx).any(|q| (p.1 - q.1).abs() <= ty);
        if !dup {
            out.push(p);
        }
    }
    out.truncate(MAX_HITS);
    out
}

// ---------------------------------------------------------------------------------------------
// One variable
// ---------------------------------------------------------------------------------------------

/// Roots of `h` over `[lo, hi]`: sign changes (bisected; poles and jumps rejected) and tangent
/// touches (a local minimum of `|h|` that reaches zero). Empty when `h` is zero on a large part
/// of the range (coincident curves).
fn roots_1d(h: &dyn Fn(f64) -> f64, lo: f64, hi: f64) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Vec::new();
    }
    let n = SAMPLES;
    let xs: Vec<f64> = (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect();
    let hs: Vec<f64> = xs.iter().map(|&x| h(x)).collect();
    let finite = hs.iter().filter(|v| v.is_finite()).count();
    if finite == 0 {
        return Vec::new();
    }
    let scale = hs.iter().filter(|v| v.is_finite()).map(|v| v.abs()).sum::<f64>() / finite as f64;
    // Coincident: h is (numerically) zero over a quarter of what is defined.
    let tiny = 1e-12 * scale.max(1e-300);
    if hs.iter().filter(|v| v.is_finite() && v.abs() <= tiny).count() * 4 >= finite.max(4) && scale > 0.0
        || (scale == 0.0)
    {
        return Vec::new();
    }
    let mut roots: Vec<f64> = Vec::new();
    for i in 0..=n {
        let (xa, fa) = (xs[i], hs[i]);
        if fa == 0.0 {
            roots.push(xa);
            continue;
        }
        if i == n || !fa.is_finite() {
            continue;
        }
        let (xb, fb) = (xs[i + 1], hs[i + 1]);
        if fb.is_finite() && fa * fb < 0.0 {
            let (mut l, mut r, fl) = (xa, xb, fa);
            for _ in 0..100 {
                let m = l / 2.0 + r / 2.0;
                let fm = h(m);
                if fm == 0.0 || m == l || m == r {
                    l = m;
                    r = m;
                    break;
                }
                if (fm < 0.0) == (fl < 0.0) {
                    l = m;
                } else {
                    r = m;
                }
            }
            let root = l / 2.0 + r / 2.0;
            // A pole or a jump leaves a large value behind; a root leaves ~0.
            if h(root).abs() <= 1e-6 * fa.abs().max(fb.abs()) {
                roots.push(root);
            }
        }
    }
    // Tangent touches: a sampled local minimum of |h| that never changed sign around it.
    let mut tried = 0;
    for i in 1..n {
        let (a, b, c) = (hs[i - 1], hs[i], hs[i + 1]);
        if !(a.is_finite() && b.is_finite() && c.is_finite()) || b == 0.0 {
            continue;
        }
        if a * b <= 0.0 || b * c <= 0.0 {
            continue; // a sign change, handled above
        }
        if !(b.abs() <= a.abs() && b.abs() <= c.abs()) || b.abs() > 1e-3 * scale {
            continue;
        }
        tried += 1;
        if tried > 200 {
            break;
        }
        // Golden-section search for the minimum of |h| on [x_{i-1}, x_{i+1}].
        let (mut l, mut r) = (xs[i - 1], xs[i + 1]);
        let g = 0.5 * (5f64.sqrt() - 1.0);
        let (mut c1, mut c2) = (r - g * (r - l), l + g * (r - l));
        let (mut f1, mut f2) = (h(c1).abs(), h(c2).abs());
        for _ in 0..200 {
            if f1 < f2 {
                r = c2;
                c2 = c1;
                f2 = f1;
                c1 = r - g * (r - l);
                f1 = h(c1).abs();
            } else {
                l = c1;
                c1 = c2;
                f1 = f2;
                c2 = l + g * (r - l);
                f2 = h(c2).abs();
            }
            if r - l <= 1e-15 * (1.0 + l.abs()) {
                break;
            }
        }
        let xm = 0.5 * (l + r);
        let v = h(xm);
        if v.is_finite() && v.abs() <= 1e-9 * scale {
            roots.push(xm);
        }
    }
    roots
}

/// Pairs whose difference is a function of x: points are `(x, y_of(x))`.
fn from_x(h: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, win: Window) -> Vec<(f64, f64)> {
    roots_1d(h, win.x0, win.x1).into_iter().map(|x| (x, f(x))).collect()
}

/// Pairs whose difference is a function of the parameter of the parametric curve.
fn from_t(
    h: &dyn Fn(f64) -> f64,
    x: &dyn Fn(f64) -> f64,
    y: &dyn Fn(f64) -> f64,
    t0: f64,
    t1: f64,
) -> Vec<(f64, f64)> {
    roots_1d(h, t0, t1).into_iter().map(|t| (x(t), y(t))).collect()
}

// ---------------------------------------------------------------------------------------------
// Two variables
// ---------------------------------------------------------------------------------------------

/// Solves `P(u, v) = Q(u, v) = 0` over `dom = [u0, u1, v0, v1]`; `map` turns a solution into the
/// plane point to report (`None` drops it).
fn solve2(
    p: &dyn Fn(f64, f64) -> f64,
    q: &dyn Fn(f64, f64) -> f64,
    dom: [f64; 4],
    map: &dyn Fn(f64, f64) -> Option<(f64, f64)>,
) -> Vec<(f64, f64)> {
    let [u0, u1, v0, v1] = dom;
    if !dom.iter().all(|v| v.is_finite()) || u1 <= u0 || v1 <= v0 {
        return Vec::new();
    }
    let n = GRID;
    let (du, dv) = ((u1 - u0) / n as f64, (v1 - v0) / n as f64);
    let w = n + 1;
    let mut pv = vec![f64::NAN; w * w];
    let mut qv = vec![f64::NAN; w * w];
    for j in 0..w {
        for i in 0..w {
            let (u, v) = (u0 + du * i as f64, v0 + dv * j as f64);
            let (a, b) = (p(u, v), q(u, v));
            if a.is_finite() && b.is_finite() {
                pv[j * w + i] = a;
                qv[j * w + i] = b;
            }
        }
    }
    let changes = |g: &[f64], i: usize, j: usize| -> bool {
        let c = [g[j * w + i], g[j * w + i + 1], g[(j + 1) * w + i], g[(j + 1) * w + i + 1]];
        if c.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let (mn, mx) = c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        mn <= 0.0 && mx >= 0.0
    };
    let mut cells: Vec<(usize, usize)> = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if changes(&pv, i, j) && changes(&qv, i, j) {
                cells.push((i, j));
            }
        }
    }
    let spans = (u1 - u0, v1 - v0);
    let multi = cells.len() <= 100;
    let mut found: Vec<(f64, f64)> = Vec::new();
    let mut overlapped = 0usize;
    'cells: for &(i, j) in &cells {
        let (cu, cv) = (u0 + du * (i as f64 + 0.5), v0 + dv * (j as f64 + 0.5));
        let starts: &[(f64, f64)] = if multi {
            &[(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        } else {
            &[(0.5, 0.5)]
        };
        let _ = (cu, cv);
        for &(fu, fv) in starts {
            let s = (u0 + du * (i as f64 + fu), v0 + dv * (j as f64 + fv));
            if found.iter().any(|z| (z.0 - s.0).abs() < 0.6 * du && (z.1 - s.1).abs() < 0.6 * dv) {
                continue;
            }
            if let Some(z) = newton(p, q, s, (du, dv), spans) {
                if found.iter().any(|f| (f.0 - z.0).abs() < 1e-7 * spans.0 && (f.1 - z.1).abs() < 1e-7 * spans.1) {
                    continue;
                }
                if along_overlap(p, q, z, (du, dv)) {
                    overlapped += 1;
                    if overlapped > 400 {
                        break 'cells;
                    }
                    continue;
                }
                found.push(z);
            }
        }
    }
    found.into_iter().filter_map(|(u, v)| map(u, v)).collect()
}

/// Numeric gradient of `f` at `(u, v)` with step `(hu, hv)`.
fn grad(f: &dyn Fn(f64, f64) -> f64, u: f64, v: f64, hu: f64, hv: f64) -> [f64; 2] {
    [(f(u + hu, v) - f(u - hu, v)) / (2.0 * hu), (f(u, v + hv) - f(u, v - hv)) / (2.0 * hv)]
}

/// Damped Gauss-Newton from `s`; the root, if it converges to a genuine common zero near `s`.
fn newton(
    p: &dyn Fn(f64, f64) -> f64,
    q: &dyn Fn(f64, f64) -> f64,
    s: (f64, f64),
    cell: (f64, f64),
    spans: (f64, f64),
) -> Option<(f64, f64)> {
    let (hu, hv) = (1e-6 * spans.0, 1e-6 * spans.1);
    let (mut u, mut v) = s;
    let norm2 = |a: f64, b: f64| a * a + b * b;
    let (mut a, mut b) = (p(u, v), q(u, v));
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    for _ in 0..80 {
        if a == 0.0 && b == 0.0 {
            break;
        }
        let gp = grad(p, u, v, hu, hv);
        let gq = grad(q, u, v, hu, hv);
        if ![gp[0], gp[1], gq[0], gq[1]].iter().all(|x| x.is_finite()) {
            return None;
        }
        // (JtJ + lambda I) d = -Jt r
        let m00 = gp[0] * gp[0] + gq[0] * gq[0];
        let m01 = gp[0] * gp[1] + gq[0] * gq[1];
        let m11 = gp[1] * gp[1] + gq[1] * gq[1];
        let lam = 1e-10 * (m00 + m11) + 1e-300;
        let (r0, r1) = (gp[0] * a + gq[0] * b, gp[1] * a + gq[1] * b);
        let det = (m00 + lam) * (m11 + lam) - m01 * m01;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let mut du = -((m11 + lam) * r0 - m01 * r1) / det;
        let mut dv = -(-m01 * r0 + (m00 + lam) * r1) / det;
        // No step longer than two cells.
        let big = (du.abs() / cell.0).max(dv.abs() / cell.1);
        if big > 2.0 {
            du *= 2.0 / big;
            dv *= 2.0 / big;
        }
        let before = norm2(a / (1.0 + a.abs()), b / (1.0 + b.abs()));
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..10 {
            let (nu, nv) = (u + step * du, v + step * dv);
            let (na, nb) = (p(nu, nv), q(nu, nv));
            if na.is_finite() && nb.is_finite() && norm2(na / (1.0 + na.abs()), nb / (1.0 + nb.abs())) <= before {
                u = nu;
                v = nv;
                a = na;
                b = nb;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        if (u - s.0).abs() > 3.0 * cell.0 || (v - s.1).abs() > 3.0 * cell.1 {
            return None;
        }
        if (step * du).abs() < 1e-14 * spans.0 && (step * dv).abs() < 1e-14 * spans.1 {
            break;
        }
    }
    // Genuine common zero: both functions are within a tiny distance (value over gradient) of it.
    let tol = 1e-7 * spans.0.max(spans.1);
    let near = |f: &dyn Fn(f64, f64) -> f64, val: f64| {
        let g = grad(f, u, v, hu, hv);
        let gl = g[0].hypot(g[1]);
        if gl > 1e-12 {
            (val / gl).abs() <= tol
        } else {
            val.abs() <= 1e-9
        }
    };
    (near(p, a) && near(q, b)).then_some((u, v))
}

/// True when `Q` stays zero along the `P = 0` curve for a few cells on both sides of `z`, which
/// means the two curves overlap there rather than cross.
fn along_overlap(
    p: &dyn Fn(f64, f64) -> f64,
    q: &dyn Fn(f64, f64) -> f64,
    z: (f64, f64),
    cell: (f64, f64),
) -> bool {
    let (hu, hv) = (1e-6 * cell.0 * 160.0, 1e-6 * cell.1 * 160.0);
    let g = grad(p, z.0, z.1, hu, hv);
    let gl = g[0].hypot(g[1]);
    if gl < 1e-12 {
        return false;
    }
    // Unit tangent of the P-curve in cell units.
    let (tu, tv) = (-g[1] / gl, g[0] / gl);
    let tol = 1e-6 * cell.0.min(cell.1);
    let mut hits = 0;
    for k in [-3.0, -1.5, 1.5, 3.0] {
        // Walk along the tangent, then pull back onto P = 0 with a couple of Newton steps.
        let (mut u, mut v) = (z.0 + k * cell.0 * tu, z.1 + k * cell.1 * tv);
        for _ in 0..6 {
            let pv = p(u, v);
            let gp = grad(p, u, v, hu, hv);
            let l2 = gp[0] * gp[0] + gp[1] * gp[1];
            if !pv.is_finite() || l2 < 1e-24 {
                break;
            }
            u -= pv * gp[0] / l2;
            v -= pv * gp[1] / l2;
        }
        let qv = q(u, v);
        let gq = grad(q, u, v, hu, hv);
        let gl = gq[0].hypot(gq[1]);
        if qv.is_finite() && gl > 1e-12 && (qv / gl).abs() <= tol {
            hits += 1;
        }
    }
    hits >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: Window = Window { x0: -10.0, x1: 10.0, y0: -10.0, y1: 10.0 };

    fn ex<'a>(f: &'a dyn Fn(f64) -> f64) -> Curve<'a> {
        Curve::new(Shape::Explicit(f))
    }
    fn im<'a>(f: &'a dyn Fn(f64, f64) -> f64) -> Curve<'a> {
        Curve::new(Shape::Implicit(f))
    }
    fn has(v: &[(f64, f64)], x: f64, y: f64) -> bool {
        v.iter().any(|p| (p.0 - x).abs() < 1e-6 && (p.1 - y).abs() < 1e-6)
    }

    #[test]
    fn sin_vs_half_known_roots() {
        let (f, g) = (|x: f64| x.sin(), |_x: f64| 0.5);
        let w = Window { x0: -7.0, x1: 7.0, ..W };
        let v = intersections(&ex(&f), &ex(&g), w);
        let (a, b) = (std::f64::consts::PI / 6.0, 5.0 * std::f64::consts::PI / 6.0);
        let tp = 2.0 * std::f64::consts::PI;
        let want = [a - tp, b - tp, a, b, a + tp];
        assert_eq!(v.len(), 5, "{v:?}");
        for x in want {
            assert!(has(&v, x, 0.5), "missing {x} in {v:?}");
        }
        // Order of arguments does not matter.
        assert_eq!(intersections(&ex(&g), &ex(&f), w).len(), 5);
    }

    #[test]
    fn parabola_and_line() {
        let (f, g) = (|x: f64| x * x, |x: f64| x + 2.0);
        let v = intersections(&ex(&f), &ex(&g), W);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, -1.0, 1.0) && has(&v, 2.0, 4.0));
        // Tangent: y = x^2 and y = 2x - 1 touch at (1, 1).
        let t = |x: f64| 2.0 * x - 1.0;
        let v = intersections(&ex(&f), &ex(&t), W);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(has(&v, 1.0, 1.0));
        // Miss.
        let m = |_x: f64| -1.0;
        assert!(intersections(&ex(&f), &ex(&m), W).is_empty());
    }

    #[test]
    fn line_vs_circle_two_one_zero() {
        let c = |x: f64, y: f64| x * x + y * y - 1.0;
        let line = |x: f64| 0.5 + 0.0 * x;
        let v = intersections(&ex(&line), &im(&c), W);
        assert_eq!(v.len(), 2, "{v:?}");
        let r = 0.75f64.sqrt();
        assert!(has(&v, -r, 0.5) && has(&v, r, 0.5));
        // Tangent line y = 1 touches at (0, 1).
        let t = |x: f64| 1.0 + 0.0 * x;
        let v = intersections(&ex(&t), &im(&c), W);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(has(&v, 0.0, 1.0));
        // Implicit line y = 1 (F = y - 1) tangent to the circle: the two-variable solver.
        let l = |_x: f64, y: f64| y - 1.0;
        let v = intersections(&im(&l), &im(&c), W);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!((v[0].0).abs() < 1e-5 && (v[0].1 - 1.0).abs() < 1e-8, "{v:?}");
        // Miss.
        let m = |x: f64| 2.0 + 0.0 * x;
        assert!(intersections(&ex(&m), &im(&c), W).is_empty());
        let l2 = |_x: f64, y: f64| y - 2.0;
        assert!(intersections(&im(&l2), &im(&c), W).is_empty());
    }

    #[test]
    fn implicit_line_two_points() {
        let c = |x: f64, y: f64| x * x + y * y - 1.0;
        let l = |x: f64, y: f64| x - y; // y = x
        let v = intersections(&im(&l), &im(&c), W);
        let r = 0.5f64.sqrt();
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, -r, -r) && has(&v, r, r));
    }

    #[test]
    fn circle_vs_circle() {
        let a = |x: f64, y: f64| x * x + y * y - 4.0;
        let b = |x: f64, y: f64| (x - 2.0).powi(2) + y * y - 4.0;
        let v = intersections(&im(&a), &im(&b), W);
        let h = 3f64.sqrt();
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, 1.0, h) && has(&v, 1.0, -h));
        // Externally tangent circles touch once.
        let c = |x: f64, y: f64| (x - 4.0).powi(2) + y * y - 4.0;
        let v = intersections(&im(&a), &im(&c), W);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!((v[0].0 - 2.0).abs() < 1e-4 && v[0].1.abs() < 1e-3, "{v:?}");
        // Concentric: none.
        let d = |x: f64, y: f64| x * x + y * y - 1.0;
        assert!(intersections(&im(&a), &im(&d), W).is_empty());
    }

    #[test]
    fn hyperbola_vs_ellipse() {
        // x^2/4 + y^2 = 1 and x^2 - y^2 = 1 meet in four points: x^2 = 8/5, y^2 = 3/5.
        let e = |x: f64, y: f64| x * x / 4.0 + y * y - 1.0;
        let h = |x: f64, y: f64| x * x - y * y - 1.0;
        let v = intersections(&im(&e), &im(&h), W);
        assert_eq!(v.len(), 4, "{v:?}");
        let (x, y) = ((8.0f64 / 5.0).sqrt(), (3.0f64 / 5.0).sqrt());
        for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            assert!(has(&v, sx * x, sy * y), "{v:?}");
        }
    }

    #[test]
    fn explicit_vs_implicit_and_x_equals_g_of_y() {
        // y = x^2 and x = y^2 (written F = x - y^2) meet at (0,0) and (1,1).
        let f = |x: f64| x * x;
        let g = |x: f64, y: f64| x - y * y;
        let v = intersections(&ex(&f), &im(&g), W);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, 0.0, 0.0) && has(&v, 1.0, 1.0));
        // The same pair as two implicit curves.
        let f2 = |x: f64, y: f64| y - x * x;
        let v = intersections(&im(&f2), &im(&g), W);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, 1.0, 1.0));
    }

    #[test]
    fn identical_curves_give_nothing() {
        let f = |x: f64| x.sin();
        assert!(intersections(&ex(&f), &ex(&f), W).is_empty());
        let a = |x: f64, y: f64| x * x + y * y - 1.0;
        let b = |x: f64, y: f64| 3.0 * (x * x + y * y - 1.0);
        assert!(intersections(&im(&a), &im(&b), W).is_empty());
        let c = |x: f64, y: f64| y - x.sin();
        assert!(intersections(&ex(&f), &im(&c), W).is_empty());
        assert!(intersections(&im(&a), &im(&a), W).is_empty());
    }

    #[test]
    fn partial_overlap_keeps_the_real_crossing() {
        // y = |x| as an implicit curve against y = x: they coincide for x >= 0 (no isolated
        // points) and the origin is the end of the overlap.
        let a = |x: f64, y: f64| y - x.abs();
        let b = |x: f64, y: f64| y - x;
        let v = intersections(&im(&a), &im(&b), W);
        assert!(v.len() <= 4, "{} points", v.len());
    }

    #[test]
    fn poles_and_nan() {
        // 1/x never meets y = 0, and the sign flip across the pole is not a root.
        let f = |x: f64| 1.0 / x;
        let z = |x: f64| 0.0 * x;
        assert!(intersections(&ex(&f), &ex(&z), W).is_empty());
        // tan(x) = x has crossings only at the origin in (-1.5, 1.5); poles must not appear.
        let t = |x: f64| x.tan();
        let id = |x: f64| x;
        let w = Window { x0: -1.5, x1: 1.5, ..W };
        let v = intersections(&ex(&t), &ex(&id), w);
        assert!(v.len() == 1 && has(&v, 0.0, 0.0), "{v:?}");
        // sqrt(x) is NaN for x < 0: the only crossing with y = x is 0 and 1.
        let s = |x: f64| x.sqrt();
        let v = intersections(&ex(&s), &ex(&id), W);
        assert!(has(&v, 1.0, 1.0), "{v:?}");
        assert!(v.iter().all(|p| p.0 >= 0.0));
        // Implicit with a pole: xy = 1 against y = x.
        let h = |x: f64, y: f64| x * y - 1.0;
        let l = |x: f64, y: f64| y - x;
        let v = intersections(&im(&h), &im(&l), W);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, 1.0, 1.0) && has(&v, -1.0, -1.0));
        // A circle against sqrt(1 - x^2): NaN outside [-1, 1].
        let c = |x: f64, y: f64| x * x + y * y - 1.0;
        let up = |x: f64| (1.0 - x * x).sqrt();
        let v = intersections(&ex(&up), &im(&c), W);
        assert!(v.is_empty(), "coincident on the upper half: {v:?}");
    }

    #[test]
    fn clip_is_honoured() {
        let f = |x: f64| x * x;
        let g = |x: f64| x + 2.0;
        let pos = |x: f64, _y: f64| x > 0.0;
        let a = Curve { shape: Shape::Explicit(&f), clip: Some(&pos) };
        let v = intersections(&a, &ex(&g), W);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(has(&v, 2.0, 4.0));
    }

    #[test]
    fn parametric_pairs() {
        // Unit circle (cos t, sin t) against y = 0.5 and against x - y = 0.
        let (cx, cy) = (|t: f64| t.cos(), |t: f64| t.sin());
        let tau = 2.0 * std::f64::consts::PI;
        let c = Curve::new(Shape::Param { x: &cx, y: &cy, t0: 0.0, t1: tau });
        let line = |x: f64| 0.5 + 0.0 * x;
        let v = intersections(&c, &ex(&line), W);
        assert_eq!(v.len(), 2, "{v:?}");
        let l = |x: f64, y: f64| x - y;
        let v = intersections(&c, &im(&l), W);
        assert_eq!(v.len(), 2, "{v:?}");
        // Two parametric circles.
        let (dx, dy) = (|t: f64| 1.0 + t.cos(), |t: f64| t.sin());
        let d = Curve::new(Shape::Param { x: &dx, y: &dy, t0: 0.0, t1: tau });
        let v = intersections(&c, &d, W);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(has(&v, 0.5, 0.75f64.sqrt()) && has(&v, 0.5, -(0.75f64.sqrt())));
    }

    #[test]
    fn many_oscillations_are_capped() {
        let f = |x: f64| (50.0 * x).sin();
        let g = |_x: f64| 0.3;
        let v = intersections(&ex(&f), &ex(&g), W);
        assert_eq!(v.len(), MAX_HITS);
    }

    #[test]
    fn outside_the_window_is_dropped() {
        let f = |x: f64| x * x;
        let g = |_x: f64| 400.0;
        // y = 400 is outside the y window even though x = +-20 would be..., both outside.
        assert!(intersections(&ex(&f), &ex(&g), W).is_empty());
    }
}
