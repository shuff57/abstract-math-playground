//! Dragging a curve `y = f(x; p)` by its parameters (pure numerics, no document types).
//!
//! The caller grabs a point of the curve; each pointer move asks [`solve`] for the smallest
//! parameter change that puts the curve through the point's new position and keeps its shape
//! around it (so a vertex-form parabola translates instead of flattening). The solve is a few
//! damped Gauss-Newton steps with box constraints: a slider never leaves `[min, max]`, and the
//! caller rounds the result to the slider's step with [`DragParam::quantize`].

/// One parameter the drag may change.
#[derive(Debug, Clone, PartialEq)]
pub struct DragParam {
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: Option<f64>,
    /// A typical size of the parameter (its range, or its magnitude when unbounded): changes are
    /// weighed relative to it, so a slider spanning 0..100 and one spanning 0..1 are equally cheap.
    pub scale: f64,
}

impl DragParam {
    /// A slider with the given range and step.
    pub fn slider(value: f64, min: f64, max: f64, step: Option<f64>) -> Self {
        let range = max - min;
        let scale = if range.is_finite() && range > 0.0 { range } else { value.abs().max(1.0) };
        DragParam { value, min, max, step: step.filter(|s| *s > 0.0), scale }
    }

    /// A free number (a definition such as `a=3`): no range, no step.
    pub fn free(value: f64) -> Self {
        DragParam {
            value,
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            step: None,
            scale: value.abs().max(1.0),
        }
    }

    pub fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    /// `v` inside the range and on the step grid (anchored at `min`), without float dust.
    pub fn quantize(&self, v: f64) -> f64 {
        let mut v = self.clamp(v);
        if let Some(st) = self.step {
            let base = if self.min.is_finite() { self.min } else { 0.0 };
            v = ((v - base) / st).round() * st + base;
            v = self.clamp(v);
            // 0.1 * 3 style dust: keep ~12 significant decimals of the step.
            let dec = (-st.log10().floor()).clamp(0.0, 12.0) as i32 + 3;
            let k = 10f64.powi(dec);
            v = (v * k).round() / k;
        }
        v
    }
}

/// World units per pixel, per axis (the residuals are measured in pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Units {
    pub x: f64,
    pub y: f64,
}

/// Where the grabbed curve point should be, plus a few samples of the curve's shape around it.
///
/// The first sample is the grabbed point itself (the curve must pass through the pointer); the
/// others are the original curve translated by the pointer's movement. Keeping them close to the
/// new curve makes a vertex-form parabola translate (`h`, `k`) instead of flattening (`a -> 0`),
/// which would pass through the pointer just as well.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    /// `(x, y, weight)`; `x` is where `f` is evaluated.
    pub samples: Vec<(f64, f64, f64)>,
}

/// Pixel offsets (each side) of the shape samples, and their weight against the grabbed point's 1.
pub const SHAPE_PX: [f64; 4] = [-60.0, -30.0, 30.0, 60.0];
pub const SHAPE_W: f64 = 0.4;
/// Cost (pixels) of changing a parameter by its whole `scale`: only breaks ties between
/// parameters that do not matter.
const REG_W: f64 = 1.0;
/// The residual of a shape sample the curve is undefined at.
pub const UNDEFINED_PX: f64 = 50.0;
const MAX_ITERS: usize = 12;
/// More free parameters than this is not a drag any more.
pub const MAX_PARAMS: usize = 8;

impl Target {
    /// The grabbed point at `x0` (curve `f` with parameters `p0`) moved by `(dx, dy)` in world
    /// units. `None` when `f` is not finite at `x0`.
    pub fn translated<F: FnMut(f64, &[f64]) -> f64>(f: &mut F, p0: &[f64], x0: f64, dx: f64, dy: f64, u: Units) -> Option<Target> {
        let y0 = f(x0, p0);
        if !y0.is_finite() || !dx.is_finite() || !dy.is_finite() {
            return None;
        }
        let mut samples = vec![(x0 + dx, y0 + dy, 1.0)];
        for o in SHAPE_PX {
            let x = x0 + o * u.x;
            let y = f(x, p0);
            if y.is_finite() {
                samples.push((x + dx, y + dy, SHAPE_W));
            }
        }
        Some(Target { samples })
    }
}

/// The gradient of `f` at `p` by central differences with steps `h` (one per axis).
pub fn gradient<F: FnMut(f64, f64) -> f64>(f: &mut F, p: [f64; 2], h: [f64; 2]) -> [f64; 2] {
    [
        (f(p[0] + h[0], p[1]) - f(p[0] - h[0], p[1])) / (2.0 * h[0]),
        (f(p[0], p[1] + h[1]) - f(p[0], p[1] - h[1])) / (2.0 * h[1]),
    ]
}

/// A few Newton steps from `p` onto the curve `f = 0`; `h` is the scale (a pixel) the gradient is
/// taken at and the accuracy aimed for. `None` when `f` is undefined or flat there.
pub fn project_to_zero<F: FnMut(f64, f64) -> f64>(f: &mut F, mut p: [f64; 2], h: [f64; 2]) -> Option<[f64; 2]> {
    for _ in 0..20 {
        let v = f(p[0], p[1]);
        let g = gradient(f, p, h);
        let g2 = g[0] * g[0] + g[1] * g[1];
        if !(v.is_finite() && g2.is_finite() && g2 > 0.0) {
            return None;
        }
        let step = [v * g[0] / g2, v * g[1] / g2];
        p = [p[0] - step[0], p[1] - step[1]];
        if (step[0] / h[0]).hypot(step[1] / h[1]) < 1e-3 {
            return Some(p);
        }
    }
    f(p[0], p[1]).is_finite().then_some(p)
}

/// The weighted pixel residuals of `f(.; p)` against the target samples.
fn residuals<F: FnMut(f64, &[f64]) -> f64>(f: &mut F, p: &[f64], t: &Target, u: Units, out: &mut Vec<f64>) -> Option<()> {
    out.clear();
    for (k, &(x, y, w)) in t.samples.iter().enumerate() {
        let v = f(x, p);
        if v.is_finite() {
            out.push(w * (v - y) / u.y);
        } else if k == 0 {
            return None;
        } else {
            out.push(w * UNDEFINED_PX);
        }
    }
    Some(())
}

/// The parameter values (continuous, not yet on the step grid) that move the curve through
/// `target`, starting from `start`; `None` when `f` is not finite at the start (nothing sensible
/// to do). With no parameters it returns `None` too.
pub fn solve<F: FnMut(f64, &[f64]) -> f64>(
    f: &mut F,
    params: &[DragParam],
    start: &[f64],
    target: &Target,
    units: Units,
) -> Option<Vec<f64>> {
    if target.samples.is_empty() {
        return None;
    }
    if !(units.x > 0.0 && units.y > 0.0) || target.samples.iter().any(|s| !(s.0.is_finite() && s.1.is_finite())) {
        return None;
    }
    solve_with(params, start, |p, out| residuals(f, p, target, units, out))
}

/// [`solve`] for any curve kind: `res(p, out)` fills `out` with the (pixel, weighted) residuals of
/// the curve with parameters `p` against the drag target, or returns `None` when the curve is
/// undefined at the grabbed point. The count of residuals must not depend on `p`.
pub fn solve_with<R: FnMut(&[f64], &mut Vec<f64>) -> Option<()>>(
    params: &[DragParam],
    start: &[f64],
    mut res: R,
) -> Option<Vec<f64>> {
    let n = params.len();
    if n == 0 || n > MAX_PARAMS || start.len() != n {
        return None;
    }
    let anchor: Vec<f64> = start.iter().zip(params).map(|(v, q)| q.clamp(*v)).collect();
    let mut p = anchor.clone();
    let cost_of = |r: &[f64], p: &[f64]| -> f64 {
        let reg: f64 = p
            .iter()
            .zip(&anchor)
            .zip(params)
            .map(|((a, b), q)| (REG_W * (a - b) / q.scale).powi(2))
            .sum();
        r.iter().map(|v| v * v).sum::<f64>() + reg
    };
    let mut r = Vec::new();
    let mut r2 = Vec::new();
    res(&p, &mut r)?;
    let mut cost = cost_of(&r, &p);
    let mut lambda = 1e-3;
    let mut q = p.clone();
    for _ in 0..MAX_ITERS {
        // Jacobian in scaled parameters u_i = p_i / scale_i (one column per parameter).
        let mut jac = vec![vec![0.0f64; r.len()]; n];
        for i in 0..n {
            let d = 1e-4 * params[i].scale;
            q.copy_from_slice(&p);
            q[i] += d;
            res(&q, &mut r2)?;
            for (k, j) in jac[i].iter_mut().enumerate() {
                *j = (r2[k] - r[k]) / d * params[i].scale;
            }
        }
        // Normal equations (JtJ + reg + lambda D) du = -(Jt r + reg (p - anchor) / scale).
        let mut a = vec![0.0f64; n * n];
        let mut g = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = jac[i].iter().zip(&jac[j]).map(|(x, y)| x * y).sum();
            }
            a[i * n + i] += REG_W * REG_W;
            g[i] = jac[i].iter().zip(&r).map(|(x, y)| x * y).sum::<f64>()
                + REG_W * REG_W * (p[i] - anchor[i]) / params[i].scale;
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut m = a.clone();
            for i in 0..n {
                m[i * n + i] += lambda * (1.0 + a[i * n + i]);
            }
            let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
            if let Some(du) = solve_linear(&mut m, rhs, n) {
                for i in 0..n {
                    q[i] = params[i].clamp(p[i] + du[i] * params[i].scale);
                }
                if res(&q, &mut r2).is_some() {
                    let c2 = cost_of(&r2, &q);
                    if c2 < cost {
                        improved = (0..n).any(|i| (q[i] - p[i]).abs() > 1e-7 * params[i].scale);
                        p.copy_from_slice(&q);
                        std::mem::swap(&mut r, &mut r2);
                        let gain = cost - c2;
                        cost = c2;
                        lambda = (lambda * 0.3).max(1e-9);
                        if gain < 1e-6 {
                            improved = false;
                        }
                        break;
                    }
                }
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    p.iter().all(|v| v.is_finite()).then_some(p)
}

/// The point of the step grid nearest the continuous solution `p`, by residual rather than by
/// distance: every stepped parameter tries the grid values just below and above `p`, and the
/// combination the curve fits the target best with wins (parameters without a step stay as
/// solved). A plain rounding would pick the per-parameter nearest, which is not always the best
/// pair when the parameters trade off against each other. `res` is as in [`solve_with`].
pub fn snap_to_steps<R: FnMut(&[f64], &mut Vec<f64>) -> Option<()>>(params: &[DragParam], p: &[f64], mut res: R) -> Vec<f64> {
    let plain: Vec<f64> = params.iter().zip(p).map(|(q, v)| if q.step.is_some() { q.quantize(*v) } else { *v }).collect();
    let stepped: Vec<usize> = (0..params.len()).filter(|&i| params[i].step.is_some()).collect();
    if stepped.is_empty() || stepped.len() > 6 {
        return plain;
    }
    let cands: Vec<Vec<f64>> = stepped
        .iter()
        .map(|&i| {
            let q = &params[i];
            let st = q.step.unwrap_or(1.0);
            let base = if q.min.is_finite() { q.min } else { 0.0 };
            let lo = q.quantize(((p[i] - base) / st).floor() * st + base);
            let hi = q.quantize(((p[i] - base) / st).floor() * st + base + st);
            if lo == hi { vec![lo] } else { vec![lo, hi] }
        })
        .collect();
    let mut best = plain.clone();
    let mut r = Vec::new();
    let cost = |v: &[f64], r: &mut Vec<f64>, res: &mut R| -> f64 {
        if res(v, r).is_none() {
            return f64::INFINITY;
        }
        r.iter().map(|x| x * x).sum()
    };
    let mut best_cost = cost(&best, &mut r, &mut res);
    let total: usize = cands.iter().map(Vec::len).product();
    let mut cur = plain;
    for k in 0..total {
        let mut rest = k;
        for (c, &i) in cands.iter().zip(&stepped) {
            cur[i] = c[rest % c.len()];
            rest /= c.len();
        }
        let c = cost(&cur, &mut r, &mut res);
        if c < best_cost {
            best_cost = c;
            best.copy_from_slice(&cur);
        }
    }
    best
}

/// Gaussian elimination with partial pivoting on the `n x n` row-major `m`; `None` if singular.
fn solve_linear(m: &mut [f64], mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs()))?;
        if !(m[piv * n + c].abs() > 1e-300) {
            return None;
        }
        if piv != c {
            for k in 0..n {
                m.swap(piv * n + k, c * n + k);
            }
            b.swap(piv, c);
        }
        for i in c + 1..n {
            let f = m[i * n + c] / m[c * n + c];
            for k in c..n {
                m[i * n + k] -= f * m[c * n + k];
            }
            b[i] -= f * b[c];
        }
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| m[i * n + k] * b[k]).sum();
        b[i] = (b[i] - s) / m[i * n + i];
    }
    b.iter().all(|v| v.is_finite()).then_some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: Units = Units { x: 0.02, y: 0.02 };

    fn quad(x: f64, p: &[f64]) -> f64 {
        p[0] * (x - p[1]).powi(2) + p[2]
    }

    fn quad_params() -> [DragParam; 3] {
        [
            DragParam::slider(1.0, -5.0, 5.0, None),
            DragParam::slider(0.0, -10.0, 10.0, None),
            DragParam::slider(0.0, -10.0, 10.0, None),
        ]
    }

    #[test]
    fn dragging_the_vertex_of_a_parabola_translates_it() {
        let mut f = quad;
        let start = [1.0, 0.0, 0.0];
        let t = Target::translated(&mut f, &start, 0.0, 3.0, 2.0, U).unwrap();
        let p = solve(&mut f, &quad_params(), &start, &t, U).unwrap();
        assert!((p[1] - 3.0).abs() < 0.05 && (p[2] - 2.0).abs() < 0.05 && (p[0] - 1.0).abs() < 0.05, "p = {p:?}");
    }

    #[test]
    fn dragging_a_flank_translates_the_parabola() {
        let mut f = quad;
        let start = [1.0, 0.0, 0.0];
        let t = Target::translated(&mut f, &start, 2.0, 1.0, 1.5, U).unwrap();
        let p = solve(&mut f, &quad_params(), &start, &t, U).unwrap();
        assert!((quad(3.0, &p) - (4.0 + 1.5)).abs() < 0.05, "p = {p:?}");
        assert!((p[0] - 1.0).abs() < 0.1, "p = {p:?}");
    }

    #[test]
    fn a_two_parameter_curve_follows_the_pointer() {
        // y = a x^2 + b: drag the point at x = 1 up.
        let ps = [DragParam::slider(1.0, -5.0, 5.0, None), DragParam::slider(0.0, -5.0, 5.0, None)];
        let mut f = |x: f64, p: &[f64]| p[0] * x * x + p[1];
        let start = [1.0, 0.0];
        let t = Target::translated(&mut f, &start, 1.0, 0.0, 2.0, U).unwrap();
        let p = solve(&mut f, &ps, &start, &t, U).unwrap();
        assert!((f(1.0, &p) - 3.0).abs() < 0.05, "p = {p:?}");
        assert!((p[1] - 2.0).abs() < 0.1, "b should carry a vertical drag: {p:?}");
    }

    #[test]
    fn a_slider_at_its_limit_stays_clamped() {
        let ps = [DragParam::slider(5.0, -5.0, 5.0, None), DragParam::slider(0.0, -5.0, 5.0, None)];
        let mut f = |x: f64, p: &[f64]| p[0] * x * x + p[1];
        let start = [5.0, 0.0];
        // Wants a steeper curve than `a` allows.
        let t = Target { samples: vec![(2.0, 40.0, 1.0)] };
        let p = solve(&mut f, &ps, &start, &t, U).unwrap();
        assert!((p[0] - 5.0).abs() < 1e-9 && p[1] > 0.0 && p[1] <= 5.0, "p = {p:?}");
    }

    #[test]
    fn values_land_on_the_step_grid() {
        let q = DragParam::slider(0.0, -10.0, 10.0, Some(0.1));
        assert_eq!(q.quantize(0.3000000004), 0.3);
        assert_eq!(q.quantize(0.26), 0.3);
        assert_eq!(q.quantize(99.0), 10.0);
        let q = DragParam::slider(1.0, 1.0, 9.0, Some(2.0));
        assert_eq!(q.quantize(4.2), 5.0);
        assert_eq!(DragParam::free(1.0).quantize(1.2345), 1.2345);
    }

    #[test]
    fn no_parameters_is_not_draggable() {
        let mut f = |x: f64, _: &[f64]| x * x;
        let t = Target { samples: vec![(1.0, 2.0, 1.0)] };
        assert!(solve(&mut f, &[], &[], &t, U).is_none());
    }

    #[test]
    fn degenerate_input_returns_none() {
        let ps = [DragParam::slider(1.0, -5.0, 5.0, None)];
        let mut f = |x: f64, p: &[f64]| p[0] / x;
        // The grabbed point is at the pole.
        assert!(Target::translated(&mut f, &[1.0], 0.0, 1.0, 1.0, U).is_none());
        let at_pole = Target { samples: vec![(0.0, 1.0, 1.0)] };
        assert!(solve(&mut f, &ps, &[1.0], &at_pole, U).is_none());
        let nan = Target { samples: vec![(1.0, f64::NAN, 1.0)] };
        assert!(solve(&mut f, &ps, &[1.0], &nan, U).is_none());
        let ok = Target { samples: vec![(1.0, 1.0, 1.0)] };
        assert!(solve(&mut f, &ps, &[1.0], &ok, Units { x: 0.0, y: 1.0 }).is_none());
        assert!(solve(&mut f, &ps, &[1.0, 2.0], &ok, U).is_none());
        assert!(Target::translated(&mut f, &[1.0], 1.0, f64::NAN, 0.0, U).is_none());
    }

    #[test]
    fn an_unrelated_parameter_does_not_move() {
        // The second parameter does not appear in f: it keeps its value.
        let ps = [DragParam::slider(0.0, -5.0, 5.0, None), DragParam::slider(2.0, -5.0, 5.0, None)];
        let mut f = |x: f64, p: &[f64]| x + p[0];
        let t = Target::translated(&mut f, &[0.0, 2.0], 1.0, 0.0, 2.0, U).unwrap();
        let p = solve(&mut f, &ps, &[0.0, 2.0], &t, U).unwrap();
        assert!((p[0] - 2.0).abs() < 0.05 && (p[1] - 2.0).abs() < 1e-6, "p = {p:?}");
    }

    #[test]
    fn a_sine_offset_follows_a_vertical_drag() {
        // y = sin(b x) + c
        let ps = [DragParam::slider(1.0, 0.1, 5.0, None), DragParam::slider(0.0, -3.0, 3.0, None)];
        let mut f = |x: f64, p: &[f64]| (p[0] * x).sin() + p[1];
        let start = [1.0, 0.0];
        let t = Target::translated(&mut f, &start, 1.0, 0.0, 0.5, U).unwrap();
        let p = solve(&mut f, &ps, &start, &t, U).unwrap();
        assert!((f(1.0, &p) - f(1.0, &start) - 0.5).abs() < 0.05, "p = {p:?}");
        assert!((p[1] - 0.5).abs() < 0.1 && (p[0] - 1.0).abs() < 0.1, "p = {p:?}");
    }

    #[test]
    fn snapping_picks_the_best_grid_pair_not_each_one_rounded() {
        // Fits a + b = 1: the continuous solution (0.45, 0.45) rounds to (0, 0), which misses by 1,
        // while the grid pair (1, 0) or (0, 1) is exact.
        let ps = [DragParam::slider(0.0, -5.0, 5.0, Some(1.0)), DragParam::slider(0.0, -5.0, 5.0, Some(1.0))];
        let res = |p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] + p[1] - 1.0);
            Some(())
        };
        let s = snap_to_steps(&ps, &[0.45, 0.45], res);
        assert!((s[0] + s[1] - 1.0).abs() < 1e-12 && s.iter().all(|v| v.fract() == 0.0), "{s:?}");
        // a free parameter is left as solved
        let ps = [DragParam::slider(0.0, -5.0, 5.0, Some(0.5)), DragParam::free(0.0)];
        let s = snap_to_steps(&ps, &[0.3, 0.123], |_: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(0.0);
            Some(())
        });
        assert_eq!(s[1], 0.123);
        assert!(s[0] == 0.0 || s[0] == 0.5);
    }

    #[test]
    fn solve_with_fits_a_custom_residual() {
        // A circle (x-h)^2 + (y-k)^2 = r^2 through (4, 1) with h, k free: the residual is the
        // signed distance of the point to the circle.
        let ps = [DragParam::slider(0.0, -10.0, 10.0, None), DragParam::slider(0.0, -10.0, 10.0, None)];
        let p = solve_with(&ps, &[0.0, 0.0], |p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(((4.0 - p[0]).hypot(1.0 - p[1]) - 3.0) / 0.02);
            Some(())
        })
        .unwrap();
        assert!(((4.0 - p[0]).hypot(1.0 - p[1]) - 3.0).abs() < 0.05, "{p:?}");
    }

    #[test]
    fn project_to_zero_lands_on_the_circle() {
        let mut f = |x: f64, y: f64| x * x + y * y - 9.0;
        let q = project_to_zero(&mut f, [3.4, 0.9], [0.02, 0.02]).unwrap();
        assert!((q[0].hypot(q[1]) - 3.0).abs() < 1e-3, "{q:?}");
        assert!(project_to_zero(&mut |_, _| f64::NAN, [1.0, 1.0], [0.02, 0.02]).is_none());
    }
}
