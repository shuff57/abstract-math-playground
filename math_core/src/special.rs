//! Special points of a curve `y = f(x)` over an x range: roots, the y-intercept, local extrema
//! and inflection points. Pure f64 (no GPU, no window), so it tests natively.
//!
//! Everything is found by sampling then refining: a sign change between neighbouring samples is
//! bisected to a root. Extrema are the sign-changing roots of `f'` and inflection points the
//! sign-changing roots of `f''`. A pole (`1/x`, `tan x`) is not a root: its bisected value stays
//! large, which [`find_roots_fn`] rejects.

use crate::compile::Program;

/// What kind of special point it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialKind {
    Root,
    YIntercept,
    Min,
    Max,
    Inflection,
}

impl SpecialKind {
    /// The name sent in events and read aloud: `root`, `y-intercept`, `minimum`, `maximum`,
    /// `inflection`.
    pub fn name(self) -> &'static str {
        match self {
            SpecialKind::Root => "root",
            SpecialKind::YIntercept => "y-intercept",
            SpecialKind::Min => "minimum",
            SpecialKind::Max => "maximum",
            SpecialKind::Inflection => "inflection",
        }
    }
}

/// A special point `(x, y)` of a curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Special {
    pub kind: SpecialKind,
    pub x: f64,
    pub y: f64,
}

/// Samples per scan of the x range.
const SAMPLES: usize = 4000;
/// Roots kept per scan (a curve like `sin(100x)` has hundreds).
const MAX_ROOTS: usize = 64;
/// Points returned in all.
const MAX_POINTS: usize = 48;

/// The roots of `p` (one variable, `x`) in `[x0, x1]`.
pub fn find_roots(p: &Program, x0: f64, x1: f64) -> Vec<f64> {
    let mut st = Vec::with_capacity(16);
    find_roots_fn(&mut |x| p.eval_with(&[x], &mut st), x0, x1)
}

/// The roots of `f` in `[x0, x1]`: a sign change between samples is bisected, an exact zero at a
/// sample counts, and a sign change across a pole is rejected.
pub fn find_roots_fn(f: &mut dyn FnMut(f64) -> f64, x0: f64, x1: f64) -> Vec<f64> {
    let mut roots = Vec::new();
    if !(x0.is_finite() && x1.is_finite() && x1 > x0) {
        return roots;
    }
    let mut xa = x0;
    let mut fa = f(xa);
    for i in 1..=SAMPLES {
        let xb = x0 + (x1 - x0) * i as f64 / SAMPLES as f64;
        let fb = f(xb);
        if roots.len() >= MAX_ROOTS {
            break;
        }
        if fa == 0.0 {
            roots.push(xa);
        } else if fa.is_finite() && fb.is_finite() && fa * fb < 0.0 {
            let (mut lo, mut hi, mut flo) = (xa, xb, fa);
            for _ in 0..100 {
                let mid = lo / 2.0 + hi / 2.0;
                let fm = f(mid);
                if fm == 0.0 || mid == lo || mid == hi {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if (fm < 0.0) == (flo < 0.0) {
                    lo = mid;
                    flo = fm;
                } else {
                    hi = mid;
                }
            }
            let r = lo / 2.0 + hi / 2.0;
            if f(r).abs() <= fa.abs().max(fb.abs()) {
                roots.push(r);
            }
        }
        xa = xb;
        fa = fb;
    }
    if fa == 0.0 && roots.len() < MAX_ROOTS {
        roots.push(xa);
    }
    roots
}

/// Central difference of `f` at `x` with step `h`.
fn central(f: &mut dyn FnMut(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// `f'(x)`: the caller's derivative, or a central difference of `f`.
fn slope(
    f: &mut dyn FnMut(f64) -> f64,
    d1: &mut Option<&mut dyn FnMut(f64) -> f64>,
    x: f64,
    h: f64,
) -> f64 {
    match d1 {
        Some(g) => g(x),
        None => central(f, x, h),
    }
}

/// `f''(x)`: the caller's second derivative, or a second difference of `f`.
fn bend(
    f: &mut dyn FnMut(f64) -> f64,
    d2: &mut Option<&mut dyn FnMut(f64) -> f64>,
    x: f64,
    h: f64,
) -> f64 {
    match d2 {
        Some(g) => g(x),
        None => (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h),
    }
}

/// The special points of `f` over `[x0, x1]`, in order of x.
///
/// `d1` and `d2` are `f'` and `f''` when the caller has them (symbolic derivatives); `None` falls
/// back to finite differences of `f`, which is less exact but works for any `f`.
pub fn analyze(
    f: &mut dyn FnMut(f64) -> f64,
    d1: Option<&mut dyn FnMut(f64) -> f64>,
    d2: Option<&mut dyn FnMut(f64) -> f64>,
    x0: f64,
    x1: f64,
) -> Vec<Special> {
    let mut out: Vec<Special> = Vec::new();
    if !(x0.is_finite() && x1.is_finite() && x1 > x0) {
        return out;
    }
    let span = x1 - x0;
    // Step for judging which side of a stationary point is up: small against the window, large
    // against rounding.
    let side = span * 1e-4;
    let hd = span * 1e-5;

    for r in find_roots_fn(f, x0, x1) {
        out.push(Special { kind: SpecialKind::Root, x: r, y: 0.0 });
    }

    let y0 = f(0.0);
    if x0 <= 0.0 && 0.0 <= x1 && y0.is_finite() && y0 != 0.0 {
        out.push(Special { kind: SpecialKind::YIntercept, x: 0.0, y: y0 });
    }

    // Extrema: sign-changing roots of f'.
    let mut d1 = d1;
    let ext = find_roots_fn(&mut |x| slope(f, &mut d1, x, hd), x0, x1);
    for r in ext {
        let y = f(r);
        if !y.is_finite() {
            continue;
        }
        let before = slope(f, &mut d1, r - side, hd);
        let after = slope(f, &mut d1, r + side, hd);
        let kind = if before < 0.0 && after > 0.0 {
            SpecialKind::Min
        } else if before > 0.0 && after < 0.0 {
            SpecialKind::Max
        } else {
            continue; // a flat spot that does not turn (x^3 at 0)
        };
        out.push(Special { kind, x: r, y });
    }

    // Inflection: sign-changing roots of f''.
    let mut d2 = d2;
    let hb = span * 1e-3;
    let infl = find_roots_fn(&mut |x| bend(f, &mut d2, x, hb), x0, x1);
    for r in infl {
        let y = f(r);
        if !y.is_finite() {
            continue;
        }
        let before = bend(f, &mut d2, r - side, hb);
        let after = bend(f, &mut d2, r + side, hb);
        if before * after < 0.0 {
            out.push(Special { kind: SpecialKind::Inflection, x: r, y });
        }
    }

    out.sort_by(|a, b| a.x.total_cmp(&b.x));
    out.truncate(MAX_POINTS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl Fn(f64) -> f64, d1: Option<fn(f64) -> f64>, d2: Option<fn(f64) -> f64>, a: f64, b: f64) -> Vec<Special> {
        let mut g1 = d1;
        let mut g2 = d2;
        let mut ff = |x| f(x);
        match (g1.as_mut(), g2.as_mut()) {
            (Some(a1), Some(a2)) => {
                let (mut p, mut q) = (|x| a1(x), |x| a2(x));
                analyze(&mut ff, Some(&mut p), Some(&mut q), a, b)
            }
            _ => analyze(&mut ff, None, None, a, b),
        }
    }

    fn of(v: &[Special], k: SpecialKind) -> Vec<(f64, f64)> {
        v.iter().filter(|s| s.kind == k).map(|s| (s.x, s.y)).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn parabola_roots_intercept_and_minimum() {
        let v = run(|x| x * x - 4.0, None, None, -10.0, 10.0);
        let roots = of(&v, SpecialKind::Root);
        assert_eq!(roots.len(), 2);
        assert!(close(roots[0].0, -2.0) && close(roots[1].0, 2.0));
        let yi = of(&v, SpecialKind::YIntercept);
        assert_eq!(yi.len(), 1);
        assert!(close(yi[0].0, 0.0) && close(yi[0].1, -4.0));
        let mins = of(&v, SpecialKind::Min);
        assert_eq!(mins.len(), 1);
        assert!(close(mins[0].0, 0.0) && close(mins[0].1, -4.0));
        assert!(of(&v, SpecialKind::Max).is_empty() && of(&v, SpecialKind::Inflection).is_empty());
    }

    #[test]
    fn cubic_has_extrema_and_an_inflection_but_a_flat_spot_does_not_count() {
        let v = run(|x| x * x * x - 3.0 * x, None, None, -5.0, 5.0);
        let (mn, mx) = (of(&v, SpecialKind::Min), of(&v, SpecialKind::Max));
        assert!(mn.len() == 1 && close(mn[0].0, 1.0) && close(mn[0].1, -2.0));
        assert!(mx.len() == 1 && close(mx[0].0, -1.0) && close(mx[0].1, 2.0));
        let inf = of(&v, SpecialKind::Inflection);
        assert!(inf.len() == 1 && close(inf[0].0, 0.0));
        // x^3: f' = 0 at 0 but it does not turn, so no extremum there; the inflection is found
        let w = run(|x| x * x * x, None, None, -5.0, 5.0);
        assert!(of(&w, SpecialKind::Min).is_empty() && of(&w, SpecialKind::Max).is_empty());
        assert_eq!(of(&w, SpecialKind::Inflection).len(), 1);
    }

    #[test]
    fn exact_derivatives_give_the_same_points() {
        let v = run(|x| x * x * x - 3.0 * x, Some(|x| 3.0 * x * x - 3.0), Some(|x| 6.0 * x), -5.0, 5.0);
        assert_eq!(of(&v, SpecialKind::Min).len(), 1);
        assert_eq!(of(&v, SpecialKind::Max).len(), 1);
        assert_eq!(of(&v, SpecialKind::Inflection).len(), 1);
    }

    #[test]
    fn sine_over_two_periods() {
        let v = run(f64::sin, None, None, -7.0, 7.0);
        assert_eq!(of(&v, SpecialKind::Root).len(), 5, "-2pi, -pi, 0, pi, 2pi");
        assert_eq!(of(&v, SpecialKind::Max).len(), 2);
        assert_eq!(of(&v, SpecialKind::Min).len(), 2);
        // the y-intercept coincides with the root at 0, so it is not listed twice
        assert!(of(&v, SpecialKind::YIntercept).is_empty());
    }

    #[test]
    fn poles_and_gaps_are_not_roots() {
        // 1/x and tan x change sign across a pole without crossing zero
        let v = run(|x| 1.0 / x, None, None, -5.0, 5.0);
        assert!(of(&v, SpecialKind::Root).is_empty(), "{v:?}");
        let t = run(f64::tan, None, None, -1.4, 1.4);
        assert_eq!(of(&t, SpecialKind::Root).len(), 1, "only the root at 0");
        // sqrt(x) is NaN for x < 0: its root at 0 is the edge of the domain
        let s = run(|x| x.sqrt() - 1.0, None, None, -4.0, 4.0);
        let roots = of(&s, SpecialKind::Root);
        assert!(roots.len() == 1 && close(roots[0].0, 1.0));
    }

    #[test]
    fn a_bad_window_gives_nothing() {
        assert!(run(|x| x, None, None, 1.0, 1.0).is_empty());
        assert!(run(|x| x, None, None, f64::NAN, 1.0).is_empty());
    }

    #[test]
    fn caps_the_number_of_points() {
        let v = run(|x| (50.0 * x).sin(), None, None, -10.0, 10.0);
        assert!(v.len() <= MAX_POINTS);
    }
}
