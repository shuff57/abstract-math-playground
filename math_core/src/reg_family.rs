//! Regression families: the types offered by a table's regression panel ("Linear Regression",
//! "Quadratic Regression", ...), as Desmos offers them.
//!
//! A family is only a *template* for an ordinary regression item (`y_1 \sim m x_1 + b`): the
//! fitting itself is the general least-squares / Levenberg-Marquardt code in [`crate::regress`],
//! which already handles any model. What a family adds:
//!
//! * the LaTeX the panel writes ([`Family::latex`]) and its inverse ([`Family::detect`]), so a
//!   document stores nothing but the regression text and survives undo, share links and the
//!   gallery unchanged;
//! * starting values for the nonlinear families ([`Family::starts`], from closed-form
//!   linearised fits), so exponential, power and logistic fits converge from any data scale;
//! * [`fit_detected`], the fit used by the scene: detection + seeding, falling back to the plain
//!   fit for any other regression text.

use std::collections::BTreeMap;

use crate::ast::Expr;
use crate::list::{eval_value, Bindings, Value};
use crate::regress::{self, FitOptions, FitResult, Linearisation, RegressError};
use crate::resolve::Defs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `y = m x + b`
    Linear,
    /// `y = a x^2 + b x + c`
    Quadratic,
    /// `y = a x^3 + b x^2 + c x + d`
    Cubic,
    /// `y = a x^4 + b x^3 + c x^2 + d x + f` (no `e`: that is Euler's number)
    Quartic,
    /// `y = a b^x`
    Exponential,
    /// `y = a + b ln(x)`
    Logarithmic,
    /// `y = a x^b`
    Power,
    /// `y = c / (1 + a e^(-b x))`
    Logistic,
    /// `y = a sin(b (x - h)) + k`
    Sinusoidal,
}

impl Family {
    pub const ALL: [Family; 9] = [
        Family::Linear,
        Family::Quadratic,
        Family::Cubic,
        Family::Quartic,
        Family::Exponential,
        Family::Logarithmic,
        Family::Power,
        Family::Logistic,
        Family::Sinusoidal,
    ];

    /// Stable machine name (`"linear"`, ...).
    pub fn key(self) -> &'static str {
        match self {
            Family::Linear => "linear",
            Family::Quadratic => "quadratic",
            Family::Cubic => "cubic",
            Family::Quartic => "quartic",
            Family::Exponential => "exponential",
            Family::Logarithmic => "logarithmic",
            Family::Power => "power",
            Family::Logistic => "logistic",
            Family::Sinusoidal => "sinusoidal",
        }
    }

    pub fn from_key(k: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.key() == k)
    }

    /// Menu label (`"Linear"`; the panel shows `"Linear Regression"`).
    pub fn label(self) -> &'static str {
        match self {
            Family::Linear => "Linear",
            Family::Quadratic => "Quadratic",
            Family::Cubic => "Cubic",
            Family::Quartic => "Quartic",
            Family::Exponential => "Exponential",
            Family::Logarithmic => "Logarithmic",
            Family::Power => "Power",
            Family::Logistic => "Logistic",
            Family::Sinusoidal => "Sinusoidal",
        }
    }

    /// Parameter names, in the order the model uses them.
    pub fn params(self) -> &'static [&'static str] {
        match self {
            Family::Linear => &["m", "b"],
            Family::Quadratic => &["a", "b", "c"],
            Family::Cubic => &["a", "b", "c", "d"],
            Family::Quartic => &["a", "b", "c", "d", "f"],
            Family::Exponential | Family::Logarithmic | Family::Power => &["a", "b"],
            Family::Logistic => &["a", "b", "c"],
            Family::Sinusoidal => &["a", "b", "h", "k"],
        }
    }

    /// The right-hand side in LaTeX for the data list `x` (e.g. `mx_1+b`).
    pub fn model_latex(self, x: &str) -> String {
        match self {
            Family::Linear => format!("m{x}+b"),
            Family::Quadratic => format!("a{x}^2+b{x}+c"),
            Family::Cubic => format!("a{x}^3+b{x}^2+c{x}+d"),
            Family::Quartic => format!("a{x}^4+b{x}^3+c{x}^2+d{x}+f"),
            Family::Exponential => format!("ab^{{{x}}}"),
            Family::Logarithmic => format!("a+b\\ln({x})"),
            Family::Power => format!("a{x}^{{b}}"),
            Family::Logistic => format!("\\frac{{c}}{{1+ae^{{-b{x}}}}}"),
            Family::Sinusoidal => format!("a\\sin(b({x}-h))+k"),
        }
    }

    /// The whole regression in LaTeX: `y_1\sim mx_1+b`.
    pub fn latex(self, y: &str, x: &str) -> String {
        format!("{y}\\sim {}", self.model_latex(x))
    }

    /// Recognises text written by [`Family::latex`] (spaces ignored): `(family, y, x)`.
    pub fn detect(latex: &str) -> Option<(Family, String, String)> {
        let s: String = latex.chars().filter(|c| !c.is_whitespace()).collect();
        let (lhs, rhs) = s.split_once("\\sim").or_else(|| s.split_once('~'))?;
        let y = list_name_at(lhs, 0).filter(|n| n.len() == lhs.len())?;
        Family::ALL.into_iter().find_map(|f| {
            let head = f.model_latex("\u{1}");
            let head = head.split('\u{1}').next().unwrap_or("");
            let rest = rhs.strip_prefix(head)?;
            let x = list_name_at(rest, 0)?;
            (f.model_latex(&x) == rhs).then(|| (f, y.clone(), x))
        })
    }

    /// A straight line (`r`, the correlation coefficient, is meaningful).
    pub fn is_linear(self) -> bool {
        self == Family::Linear
    }

    /// Starting values for the nonlinear families from a closed-form fit in transformed
    /// coordinates. `None` when the family is linear in its parameters (no start needed) or the
    /// data are outside the transform's domain (the default starts are used then).
    pub fn starts(self, x: &[f64], y: &[f64]) -> Option<BTreeMap<String, f64>> {
        let two = |a: f64, b: f64| {
            (a.is_finite() && b.is_finite())
                .then(|| BTreeMap::from([("a".to_string(), a), ("b".to_string(), b)]))
        };
        match self {
            Family::Exponential => {
                let r = regress::fit_linearised(Linearisation::Exponential, x, y).ok()?;
                // a e^(k x) = a (e^k)^x
                two(r.param("a")?, r.param("b")?.exp())
            }
            Family::Power => {
                let r = regress::fit_linearised(Linearisation::Power, x, y).ok()?;
                two(r.param("a")?, r.param("b")?)
            }
            Family::Logistic => {
                if x.len() != y.len() || y.len() < 3 || !y.iter().all(|v| *v > 0.0) {
                    return None;
                }
                let c = 1.1 * y.iter().cloned().fold(f64::MIN, f64::max);
                // c / y - 1 = a e^(-b x)  =>  ln(c / y - 1) = ln a - b x
                let z: Vec<f64> = y.iter().map(|v| (c / v - 1.0).ln()).collect();
                let line = regress::polyfit(x, &z, 1).ok()?;
                let (a, b) = (line.param("a_0")?.exp(), -line.param("a_1")?);
                let mut m = two(a, b)?;
                m.insert("c".into(), c);
                Some(m)
            }
            Family::Sinusoidal => sinusoid_start(x, y),
            _ => None,
        }
    }
}

/// Starting values `a, b, h, k` for `a sin(b (x - h)) + k`. The frequency is the one whose
/// linear fit `A sin(w x) + B cos(w x) + K` leaves the least residual over a grid fine enough to
/// resolve the phase across the whole x span (a plain Gauss-Newton start far from the true
/// frequency lands in a local minimum). `None` for fewer than 4 points, no spread in x, or
/// non-finite data.
fn sinusoid_start(x: &[f64], y: &[f64]) -> Option<BTreeMap<String, f64>> {
    let n = x.len();
    if n != y.len() || n < 4 || !x.iter().chain(y).all(|v| v.is_finite()) {
        return None;
    }
    let (xmin, xmax) = x.iter().fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    let span = xmax - xmin;
    if span <= 0.0 {
        return None;
    }
    let mut xs: Vec<f64> = x.to_vec();
    xs.sort_by(|a, b| a.total_cmp(b));
    let mut gaps: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(|a, b| a.total_cmp(b));
    // Up to the Nyquist frequency of the typical spacing; at least one period over the span.
    let w_lo = std::f64::consts::PI / span;
    let w_hi = (std::f64::consts::PI / gaps[gaps.len() / 2]).max(2.0 * w_lo);
    let steps = (((w_hi - w_lo) * span / (std::f64::consts::PI / 8.0)).ceil() as usize).clamp(16, 4000);
    let x0 = xmin;
    let mut best: Option<(f64, f64, [f64; 3])> = None; // (rss, w, [A, B, K])
    for i in 0..=steps {
        let w = w_lo + (w_hi - w_lo) * i as f64 / steps as f64;
        let Some((coef, rss)) = sin_cos_fit(x, y, w, x0) else { continue };
        if best.as_ref().is_none_or(|b| rss < b.0) {
            best = Some((rss, w, coef));
        }
    }
    let (_, w, [aa, bb, kk]) = best?;
    let amp = aa.hypot(bb);
    if !amp.is_finite() || amp == 0.0 {
        return None;
    }
    // A sin(w t) + B cos(w t) = amp sin(w t + phi), t = x - x0; the model wants sin(w (x - h)).
    let phi = bb.atan2(aa);
    let h = x0 - phi / w;
    Some(BTreeMap::from([
        ("a".to_string(), amp),
        ("b".to_string(), w),
        ("h".to_string(), h),
        ("k".to_string(), kk),
    ]))
}

/// Least squares of `y ~ A sin(w t) + B cos(w t) + K` (`t = x - x0`): `([A, B, K], rss)`, `None`
/// when the 3x3 normal equations are singular.
fn sin_cos_fit(x: &[f64], y: &[f64], w: f64, x0: f64) -> Option<([f64; 3], f64)> {
    let mut m = [[0.0f64; 4]; 3];
    for (xi, yi) in x.iter().zip(y) {
        let t = w * (xi - x0);
        let b = [t.sin(), t.cos(), 1.0];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += b[r] * b[c];
            }
            m[r][3] += b[r] * yi;
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..3 {
        let piv = (col..3).max_by(|a, b| m[*a][col].abs().total_cmp(&m[*b][col].abs()))?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        for r in col + 1..3 {
            let f = m[r][col] / m[col][col];
            for c in col..4 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    let mut s = [0.0; 3];
    for r in (0..3).rev() {
        let mut v = m[r][3];
        for c in r + 1..3 {
            v -= m[r][c] * s[c];
        }
        s[r] = v / m[r][r];
    }
    let rss: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| {
            let t = w * (xi - x0);
            (yi - (s[0] * t.sin() + s[1] * t.cos() + s[2])).powi(2)
        })
        .sum();
    rss.is_finite().then_some((s, rss))
}

/// A list name (`x_1`, `y_{12}`, `L`) starting at byte `i` of `s`: one ASCII letter, then
/// optionally `_` and one alphanumeric or a `{...}` group.
fn list_name_at(s: &str, i: usize) -> Option<String> {
    let b = s.as_bytes();
    if !b.get(i)?.is_ascii_alphabetic() {
        return None;
    }
    let mut j = i + 1;
    if b.get(j) == Some(&b'_') {
        match b.get(j + 1) {
            Some(b'{') => {
                let close = s[j + 2..].find('}')? + j + 2;
                if close == j + 2 || !s[j + 2..close].bytes().all(|c| c.is_ascii_alphanumeric()) {
                    return None;
                }
                j = close + 1;
            }
            Some(c) if c.is_ascii_alphanumeric() => j += 2,
            _ => return None,
        }
    }
    Some(s[i..j].to_string())
}

/// The numeric list a name has in `defs` (`None` when it is not a list of numbers).
fn list_of(defs: &Defs, name: &str) -> Option<Vec<f64>> {
    let body = defs.resolve(&Expr::var(name)).ok()?;
    match eval_value(&body, &Bindings::new().with_angle(defs.angle())).ok()? {
        Value::List(l) => Some(l),
        _ => None,
    }
}

/// Fits a regression item the way the scene does: when `latex` is a [`Family`] template the
/// nonlinear families start from [`Family::starts`] (and fall back to the default starts if that
/// fails); any other regression is the plain [`regress::fit_regression`].
pub fn fit_detected(
    reg: &Expr,
    latex: &str,
    defs: &Defs,
    params: Option<&[String]>,
) -> Result<FitResult, RegressError> {
    fit_detected_mode(reg, latex, defs, params, false)
}

/// [`fit_detected`] with Desmos' "log mode": for the exponential and power families the
/// parameters are those of the straight-line fit of `ln y` (against `x` or `ln x`), instead of
/// the minimum of the squared error in `y`. It needs positive data (an error otherwise) and
/// weighs small and large `y` alike, which suits data over several orders of magnitude. The
/// reported `r2` and `rmse` are still measured in `y`. Other families ignore `log_mode`.
pub fn fit_detected_mode(
    reg: &Expr,
    latex: &str,
    defs: &Defs,
    params: Option<&[String]>,
    log_mode: bool,
) -> Result<FitResult, RegressError> {
    if log_mode {
        if let Some((f, y, x)) = Family::detect(latex) {
            if matches!(f, Family::Exponential | Family::Power) {
                let (xs, ys) = (list_of(defs, &x), list_of(defs, &y));
                let init = match (xs, ys) {
                    (Some(xs), Some(ys)) => f.starts(&xs, &ys),
                    _ => None,
                };
                let Some(init) = init else {
                    return Err(RegressError::Domain(
                        "log mode needs positive data (x for power, y for both)".into(),
                    ));
                };
                // No iterations: the log-space solution itself, with its fit statistics.
                let opts = FitOptions { init, max_iter: 0 };
                return regress::fit_regression_with(reg, defs, params, &opts);
            }
        }
    }
    let seeded = Family::detect(latex).and_then(|(f, y, x)| {
        let (xs, ys) = (list_of(defs, &x)?, list_of(defs, &y)?);
        f.starts(&xs, &ys)
    });
    if let Some(init) = seeded {
        let opts = FitOptions {
            init,
            ..FitOptions::default()
        };
        if let Ok(r) = regress::fit_regression_with(reg, defs, params, &opts) {
            return Ok(r);
        }
    }
    regress::fit_regression(reg, defs, params)
}

/// Pearson's `r` of a straight-line fit: `sign(slope) * sqrt(R^2)` (exact for least squares
/// with an intercept). `None` for other families or a non-finite `R^2`.
pub fn linear_r(family: Family, fit: &FitResult) -> Option<f64> {
    if !family.is_linear() || !fit.r2.is_finite() {
        return None;
    }
    let m = fit.param("m")?;
    Some(m.signum() * fit.r2.clamp(0.0, 1.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::Kind;
    use crate::parse::parse;

    fn defs(x: &[f64], y: &[f64]) -> Defs {
        let list = |v: &[f64]| Expr::List(v.iter().map(|t| Expr::Num(*t)).collect());
        let mut d = Defs::new();
        d.define_var("x_1", list(x));
        d.define_var("y_1", list(y));
        d
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    fn fit_family(f: Family, x: &[f64], y: &[f64]) -> FitResult {
        let latex = f.latex("y_1", "x_1");
        let reg = parse(&latex).unwrap_or_else(|e| panic!("{latex}: {e}"));
        let params: Vec<String> = f.params().iter().map(|s| s.to_string()).collect();
        fit_detected(&reg, &latex, &defs(x, y), Some(&params))
            .unwrap_or_else(|e| panic!("{latex}: {e}"))
    }

    #[test]
    fn every_template_parses_as_a_regression_and_round_trips() {
        for f in Family::ALL {
            for (y, x) in [("y_1", "x_1"), ("y_{12}", "x_{12}"), ("L", "a_2")] {
                let latex = f.latex(y, x);
                let e = parse(&latex).unwrap_or_else(|e| panic!("{latex}: {e}"));
                assert!(
                    matches!(
                        crate::analyze::analyze(&e, &Default::default()).kind,
                        Kind::Regression { .. }
                    ),
                    "{latex}"
                );
                assert_eq!(
                    Family::detect(&latex),
                    Some((f, y.to_string(), x.to_string())),
                    "{latex}"
                );
                // Spaces (MathLive puts some in) do not matter.
                let spaced = latex.replace('+', " + ");
                assert_eq!(Family::detect(&spaced).map(|d| d.0), Some(f), "{spaced}");
            }
            assert_eq!(Family::from_key(f.key()), Some(f));
        }
        for other in [
            "y_1\\sim mx_1",
            "y_1=mx_1+b",
            "y_1\\sim mx_1+b+1",
            "",
            "y_1\\sim a\\sin(x_1)",
            "2\\sim mx_1+b",
        ] {
            assert_eq!(Family::detect(other), None, "{other}");
        }
        assert_eq!(
            Family::detect("y_1 ~ mx_1+b").map(|d| d.0),
            Some(Family::Linear)
        );
    }

    #[test]
    fn sinusoidal_fits_recover_amplitude_frequency_phase_and_offset() {
        // 3 sin(2 (x - 0.5)) + 1 over about 1.6 periods, then a fast one the default starts miss.
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
        for (a, b, h, k) in [(3.0, 2.0, 0.5, 1.0), (0.5, 5.0, 0.2, -4.0), (10.0, 0.7, 2.0, 0.0)] {
            let y: Vec<f64> = x.iter().map(|t| a * (b * (t - h)).sin() + k).collect();
            let r = fit_family(Family::Sinusoidal, &x, &y);
            assert!(close(r.r2, 1.0, 1e-9), "r2 {} for {a} {b} {h} {k}", r.r2);
            // a, b, h are determined up to the sin symmetries; compare the curve instead
            let (fa, fb, fh, fk) = (
                r.param("a").unwrap(),
                r.param("b").unwrap(),
                r.param("h").unwrap(),
                r.param("k").unwrap(),
            );
            assert!(close(fk, k, 1e-6), "k {fk} vs {k}");
            assert!(close(fa.abs(), a, 1e-6), "a {fa} vs {a}");
            assert!(close(fb.abs(), b, 1e-6), "b {fb} vs {b}");
            for t in [0.3, 1.7, 6.1] {
                let want = a * (b * (t - h)).sin() + k;
                let got = fa * (fb * (t - fh)).sin() + fk;
                assert!((want - got).abs() < 1e-6, "curve differs at {t}");
            }
        }
        // Noise and irregular spacing still land on the right frequency.
        let x: Vec<f64> = (0..60).map(|i| i as f64 * 0.17 + ((i * 7) % 5) as f64 * 0.01).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, t)| 4.0 * (1.3 * (t - 0.4)).sin() + 2.0 + 0.2 * ((i * 37 % 11) as f64 / 11.0 - 0.5))
            .collect();
        let r = fit_family(Family::Sinusoidal, &x, &y);
        assert!(close(r.param("b").unwrap().abs(), 1.3, 0.02), "{:?}", r.param("b"));
        assert!(close(r.param("k").unwrap(), 2.0, 0.05));
        assert!(r.r2 > 0.99);
        // Too little data is not seeded (the plain fit may still run).
        assert!(Family::Sinusoidal.starts(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0]).is_none());
        assert!(Family::Sinusoidal.starts(&[1.0; 6], &[1.0, 0.0, 1.0, 2.0, 1.0, 3.0]).is_none());
    }

    #[test]
    fn log_mode_fits_ln_y_instead_of_y() {
        // Data over several orders of magnitude with one outlier at the large end: the fit in y
        // is dragged to the outlier, the log-space fit follows the whole curve.
        let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|t| 2.0 * 3.0f64.powf(*t)).collect();
        y[7] *= 1.5;
        let latex = Family::Exponential.latex("y_1", "x_1");
        let reg = parse(&latex).unwrap();
        let params: Vec<String> = Family::Exponential.params().iter().map(|s| s.to_string()).collect();
        let normal = fit_detected_mode(&reg, &latex, &defs(&x, &y), Some(&params), false).unwrap();
        let logm = fit_detected_mode(&reg, &latex, &defs(&x, &y), Some(&params), true).unwrap();
        // the log-space line through ln y: slope ln b, the closed-form answer
        let ly: Vec<f64> = y.iter().map(|v| v.ln()).collect();
        let line = regress::polyfit(&x, &ly, 1).unwrap();
        assert!(close(logm.param("b").unwrap(), line.param("a_1").unwrap().exp(), 1e-9));
        assert!(close(logm.param("a").unwrap(), line.param("a_0").unwrap().exp(), 1e-9));
        // and it differs from the squared-error-in-y fit
        assert!((logm.param("b").unwrap() - normal.param("b").unwrap()).abs() > 1e-4);
        assert!(logm.r2.is_finite() && logm.r2 > 0.9, "r2 stays measured in y: {}", logm.r2);
        // the power family too
        let y: Vec<f64> = x.iter().map(|t| 5.0 * t.powf(1.5)).collect();
        let latex = Family::Power.latex("y_1", "x_1");
        let reg = parse(&latex).unwrap();
        let pw = fit_detected_mode(&reg, &latex, &defs(&x, &y), Some(&["a".into(), "b".into()]), true).unwrap();
        assert!(close(pw.param("a").unwrap(), 5.0, 1e-9) && close(pw.param("b").unwrap(), 1.5, 1e-9));
        // non-positive data is an error, other families ignore the flag
        let bad = vec![1.0, -2.0, 3.0, 4.0];
        let err = fit_detected_mode(&reg, &latex, &defs(&[1.0, 2.0, 3.0, 4.0], &bad), None, true);
        assert!(matches!(err, Err(RegressError::Domain(_))), "{err:?}");
        let lin = Family::Linear.latex("y_1", "x_1");
        let lreg = parse(&lin).unwrap();
        let a = fit_detected_mode(&lreg, &lin, &defs(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]), None, true).unwrap();
        let b = fit_detected_mode(&lreg, &lin, &defs(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]), None, false).unwrap();
        assert_eq!(a.params, b.params);
    }

    #[test]
    fn linear_fit_and_r() {
        let r = fit_family(Family::Linear, &[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]);
        assert!(close(r.param("m").unwrap(), 1.5, 1e-12));
        assert!(close(r.param("b").unwrap(), -2.0 / 3.0, 1e-12));
        let rr = linear_r(Family::Linear, &r).unwrap();
        assert!(close(rr, (r.r2).sqrt(), 1e-12) && rr > 0.98);
        let down = fit_family(Family::Linear, &[1.0, 2.0, 3.0], &[3.0, 2.0, 0.5]);
        assert!(linear_r(Family::Linear, &down).unwrap() < -0.9);
        assert_eq!(linear_r(Family::Quadratic, &r), None);
    }

    #[test]
    fn polynomial_families_recover_exact_data() {
        let x: Vec<f64> = (0..8).map(|i| i as f64 - 3.0).collect();
        type Case<'a> = (Family, &'a dyn Fn(f64) -> f64, &'a [(&'a str, f64)]);
        let cases: [Case; 3] = [
            (
                Family::Quadratic,
                &|t| 2.0 * t * t - 3.0 * t + 1.0,
                &[("a", 2.0), ("b", -3.0), ("c", 1.0)],
            ),
            (
                Family::Cubic,
                &|t| 0.5 * t.powi(3) - t + 4.0,
                &[("a", 0.5), ("b", 0.0), ("c", -1.0), ("d", 4.0)],
            ),
            (
                Family::Quartic,
                &|t| -0.25 * t.powi(4) + t.powi(2) + 2.0,
                &[("a", -0.25), ("b", 0.0), ("c", 1.0), ("d", 0.0), ("f", 2.0)],
            ),
        ];
        for (f, g, want) in cases {
            let y: Vec<f64> = x.iter().map(|t| g(*t)).collect();
            let r = fit_family(f, &x, &y);
            for (n, v) in want {
                assert!((r.param(n).unwrap() - v).abs() < 1e-9, "{f:?} {n}");
            }
            assert!(close(r.r2, 1.0, 1e-12), "{f:?}");
        }
    }

    #[test]
    fn nonlinear_families_converge_from_data_scale_starts() {
        let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        // Exponential with a large scale (default starts of 1 would be far off).
        let y: Vec<f64> = x.iter().map(|t| 300.0 * 1.4f64.powf(*t)).collect();
        let r = fit_family(Family::Exponential, &x, &y);
        assert!(
            close(r.param("a").unwrap(), 300.0, 1e-6) && close(r.param("b").unwrap(), 1.4, 1e-8)
        );
        // Power
        let y: Vec<f64> = x.iter().map(|t| 7.0 * t.powf(1.7)).collect();
        let r = fit_family(Family::Power, &x, &y);
        assert!(close(r.param("a").unwrap(), 7.0, 1e-6) && close(r.param("b").unwrap(), 1.7, 1e-8));
        // Logarithmic (linear in its parameters)
        let y: Vec<f64> = x.iter().map(|t| 2.0 + 3.0 * t.ln()).collect();
        let r = fit_family(Family::Logarithmic, &x, &y);
        assert!(close(r.param("a").unwrap(), 2.0, 1e-9) && close(r.param("b").unwrap(), 3.0, 1e-9));
        // Logistic with a carrying capacity of 1000
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|t| 1000.0 / (1.0 + 50.0 * (-0.8 * t).exp()))
            .collect();
        let r = fit_family(Family::Logistic, &x, &y);
        assert!(close(r.param("c").unwrap(), 1000.0, 1e-5), "{:?}", r.params);
        assert!(
            close(r.param("a").unwrap(), 50.0, 1e-4) && close(r.param("b").unwrap(), 0.8, 1e-5)
        );
    }

    #[test]
    fn starts_only_for_nonlinear_families_and_valid_domains() {
        let x = [1.0, 2.0, 3.0];
        assert!(Family::Linear.starts(&x, &[1.0, 2.0, 4.0]).is_none());
        assert!(Family::Exponential.starts(&x, &[1.0, -2.0, 4.0]).is_none());
        // Outside the domain the plain fit still runs (and may still succeed).
        let r = fit_family(Family::Exponential, &[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]);
        assert!(close(r.param("a").unwrap(), 0.5, 1e-8) && close(r.param("b").unwrap(), 2.0, 1e-8));
    }
}
