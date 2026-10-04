//! Regression: least-squares fitting of a model expression to lists.
//!
//! `y_1 ~ a x_1 + b` parses to a regression item (see [`crate::analyze::Kind::Regression`]).
//! The model is any expression of list variables and *parameters* (names that are not lists).
//! [`fit`] picks the method:
//!
//! * the model is **linear in its parameters** (checked symbolically: no partial derivative of
//!   the model mentions a parameter): one QR least-squares solve, which covers multi-term models
//!   (`a x_1 + b x_2 + c`), polynomials and `a sin(x) + b cos(x)`;
//! * otherwise **Levenberg-Marquardt** with the symbolic Jacobian ([`crate::calculus`]), falling
//!   back to finite differences per parameter when a derivative is not available.
//!
//! [`polyfit`] and [`fit_linearised`] (exponential, logarithmic, power) are the closed-form
//! helpers. Everything is a pure function of its inputs: nothing here touches a document, and
//! user text never reaches an evaluator except as an [`Expr`] through [`crate::list`].
//!
//! Output: fitted parameter values, `r2`, `rmse`, the residuals (`y - fit`) and, when it can be
//! computed, standard errors. [`FitResult::apply_to`] writes the fitted values into a [`Defs`] as
//! slider-style scalars so the fitted curve (`FitResult::curve_expr`) resolves and plots like any
//! other expression.

use crate::ast::{BinOp, Expr};
use crate::calculus;
use crate::list::{eval_value, Bindings, Value};
use crate::resolve::Defs;
use std::collections::{BTreeMap, BTreeSet};

/// Named data lists (the `x_1` and `y_1` of `y_1 ~ a x_1 + b`).
pub type Lists = BTreeMap<String, Vec<f64>>;

#[derive(Debug, Clone, PartialEq)]
pub enum RegressError {
    /// The model has no parameters (every name in it is a list).
    NoParams,
    /// The model or the regression item is not usable (message says why).
    BadModel(String),
    /// Two lists, or the response and the model, have different lengths.
    LengthMismatch { a: usize, b: usize },
    /// Fewer data points than parameters.
    TooFewPoints { have: usize, need: usize },
    /// The parameters are not identifiable (dependent columns, a constant data list, ...).
    Singular,
    /// The model produced `NaN` or infinity at every starting point, or the data hold such values.
    NonFinite,
    /// The data are outside the domain of a linearisation (e.g. `ln` of a non-positive value).
    Domain(String),
    /// Evaluating the model failed (message from the list evaluator).
    Eval(String),
}

impl std::fmt::Display for RegressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressError::NoParams => write!(f, "the model has no parameters to fit"),
            RegressError::BadModel(m) | RegressError::Eval(m) | RegressError::Domain(m) => write!(f, "{m}"),
            RegressError::LengthMismatch { a, b } => write!(f, "list lengths differ ({a} vs {b})"),
            RegressError::TooFewPoints { have, need } => {
                write!(f, "{have} data point(s) cannot fit {need} parameter(s)")
            }
            RegressError::Singular => write!(f, "the parameters cannot be told apart from this data"),
            RegressError::NonFinite => write!(f, "the model is not finite on this data"),
        }
    }
}

impl std::error::Error for RegressError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linearisation {
    /// `y = a e^(b x)`
    Exponential,
    /// `y = a + b ln(x)`
    Logarithmic,
    /// `y = a x^b`
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMethod {
    /// Linear in the parameters: one least-squares solve (exact minimum).
    Linear,
    /// Closed form after a change of variables ([`fit_linearised`]); minimises the error in the
    /// transformed space, not the residuals reported here.
    Linearised(Linearisation),
    /// Levenberg-Marquardt.
    NonLinear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Parameter names with their fitted values, in the order requested.
    pub params: Vec<(String, f64)>,
    /// Coefficient of determination `1 - SSE/SST` (SST about the mean of `y`). `NaN` when `y`
    /// is constant and the fit is not exact.
    pub r2: f64,
    /// `sqrt(SSE / n)`.
    pub rmse: f64,
    /// Sum of squared residuals.
    pub sse: f64,
    /// `y - fitted`, one per data point.
    pub residuals: Vec<f64>,
    /// The model evaluated at the fitted parameters.
    pub fitted: Vec<f64>,
    pub n: usize,
    /// Standard error of each parameter (needs `n > #params` and an invertible normal matrix).
    pub std_errors: Option<Vec<f64>>,
    /// Levenberg-Marquardt iterations (0 for closed-form methods).
    pub iterations: usize,
    /// `false` when Levenberg-Marquardt hit its iteration limit.
    pub converged: bool,
    pub method: FitMethod,
    /// The model with the parameters still symbolic (list names intact).
    pub model: Expr,
    /// The list names the model uses.
    pub data_vars: Vec<String>,
}

impl FitResult {
    pub fn param(&self, name: &str) -> Option<f64> {
        self.params.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// Writes every fitted parameter into `defs` as a slider-style scalar (overriding a
    /// same-named definition), so expressions that use the parameters resolve to the fit.
    pub fn apply_to(&self, defs: &mut Defs) {
        for (name, v) in &self.params {
            defs.set_slider(name, *v);
        }
    }

    /// The model with the fitted numbers substituted for the parameters.
    pub fn fitted_model(&self) -> Expr {
        let mut e = self.model.clone();
        for (name, v) in &self.params {
            e = e.subst(name, &Expr::Num(*v));
        }
        e
    }

    /// The model curve over a single independent variable: the one data list the model uses is
    /// renamed to `indep` (e.g. `"x"`) and the parameters stay symbolic, ready to resolve against
    /// a [`Defs`] that [`FitResult::apply_to`] filled. `None` unless the model uses exactly one list.
    pub fn curve_expr(&self, indep: &str) -> Option<Expr> {
        match self.data_vars.as_slice() {
            [v] => Some(self.model.subst(v, &Expr::var(indep))),
            _ => None,
        }
    }

    /// Like [`FitResult::curve_expr`] with the fitted numbers already substituted.
    pub fn fitted_curve(&self, indep: &str) -> Option<Expr> {
        match self.data_vars.as_slice() {
            [v] => Some(self.fitted_model().subst(v, &Expr::var(indep))),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Starting values for nonlinear fits (default 1.0 each).
    pub init: BTreeMap<String, f64>,
    pub max_iter: usize,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions { init: BTreeMap::new(), max_iter: 200 }
    }
}

// ---------------------------------------------------------------- small dense linear algebra

#[allow(clippy::needless_range_loop)] // dense-matrix indexing reads best with indices
/// Least squares `min |A c - b|` by Householder QR. `cols` are the columns of `A`.
fn lstsq(cols: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, RegressError> {
    let (n, p) = (b.len(), cols.len());
    if n < p {
        return Err(RegressError::TooFewPoints { have: n, need: p });
    }
    let mut a: Vec<Vec<f64>> = (0..n).map(|i| cols.iter().map(|c| c[i]).collect()).collect();
    let mut rhs = b.to_vec();
    let col_norm: Vec<f64> = cols.iter().map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    for k in 0..p {
        let norm = (k..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
        if norm.is_nan() || norm <= 1e-10 * col_norm[k] {
            return Err(RegressError::Singular);
        }
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..n).map(|i| a[i][k]).collect();
        v[0] -= alpha;
        let vn2: f64 = v.iter().map(|x| x * x).sum();
        if vn2 > 0.0 {
            for j in k..p {
                let dot: f64 = (k..n).map(|i| v[i - k] * a[i][j]).sum();
                let s = 2.0 * dot / vn2;
                for i in k..n {
                    a[i][j] -= s * v[i - k];
                }
            }
            let dot: f64 = (k..n).map(|i| v[i - k] * rhs[i]).sum();
            let s = 2.0 * dot / vn2;
            for i in k..n {
                rhs[i] -= s * v[i - k];
            }
        }
        a[k][k] = alpha;
        for row in a.iter_mut().take(n).skip(k + 1) {
            row[k] = 0.0;
        }
    }
    let mut c = vec![0.0; p];
    for k in (0..p).rev() {
        let s: f64 = (k + 1..p).map(|j| a[k][j] * c[j]).sum();
        c[k] = (rhs[k] - s) / a[k][k];
    }
    if c.iter().any(|v| !v.is_finite()) {
        return Err(RegressError::NonFinite);
    }
    Ok(c)
}

#[allow(clippy::needless_range_loop)] // dense-matrix indexing reads best with indices
/// Solves `m x = g` by Gaussian elimination with partial pivoting.
fn solve(m: &[Vec<f64>], g: &[f64]) -> Option<Vec<f64>> {
    let p = g.len();
    let mut a: Vec<Vec<f64>> = m.iter().zip(g).map(|(row, gi)| row.iter().copied().chain([*gi]).collect()).collect();
    for k in 0..p {
        let piv = (k..p).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[piv][k].is_nan() || a[piv][k].abs() <= 1e-300 {
            return None;
        }
        a.swap(k, piv);
        for i in k + 1..p {
            let f = a[i][k] / a[k][k];
            for j in k..=p {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    let mut x = vec![0.0; p];
    for k in (0..p).rev() {
        let s: f64 = (k + 1..p).map(|j| a[k][j] * x[j]).sum();
        x[k] = (a[k][p] - s) / a[k][k];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let p = m.len();
    let cols: Option<Vec<Vec<f64>>> = (0..p)
        .map(|j| {
            let e: Vec<f64> = (0..p).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
            solve(m, &e)
        })
        .collect();
    let cols = cols?;
    Some((0..p).map(|i| (0..p).map(|j| cols[j][i]).collect()).collect())
}

fn normal_matrix(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = cols.len();
    let mut m = vec![vec![0.0; p]; p];
    for a in 0..p {
        for b in a..p {
            let s: f64 = cols[a].iter().zip(&cols[b]).map(|(x, y)| x * y).sum();
            m[a][b] = s;
            m[b][a] = s;
        }
    }
    m
}

// ---------------------------------------------------------------- the fitting problem

struct Problem<'a> {
    y: &'a [f64],
    model: &'a Expr,
    params: &'a [String],
    bind: Bindings,
    /// Symbolic `d model / d param`, where one exists.
    jac: Vec<Option<Expr>>,
}

impl Problem<'_> {
    fn n(&self) -> usize {
        self.y.len()
    }

    fn eval(&mut self, e: &Expr, theta: &[f64]) -> Result<Vec<f64>, RegressError> {
        for (p, v) in self.params.iter().zip(theta) {
            self.bind.set(p, Value::Num(*v));
        }
        match eval_value(e, &self.bind).map_err(|e| RegressError::Eval(e.to_string()))? {
            Value::Num(c) => Ok(vec![c; self.n()]),
            Value::List(l) if l.len() == self.n() => Ok(l),
            Value::List(l) => Err(RegressError::LengthMismatch { a: l.len(), b: self.n() }),
            _ => Err(RegressError::BadModel("the model must evaluate to numbers".into())),
        }
    }

    fn predict(&mut self, theta: &[f64]) -> Result<Vec<f64>, RegressError> {
        let m = self.model;
        self.eval(m, theta)
    }

    /// Columns `d model / d theta_j` at `theta`.
    fn jacobian(&mut self, theta: &[f64]) -> Result<Vec<Vec<f64>>, RegressError> {
        let mut cols = Vec::with_capacity(theta.len());
        for j in 0..theta.len() {
            let sym = self.jac[j].clone();
            let col = match sym {
                Some(e) => self.eval(&e, theta)?,
                None => {
                    let h = 6e-6 * theta[j].abs().max(1.0);
                    let (mut up, mut dn) = (theta.to_vec(), theta.to_vec());
                    up[j] += h;
                    dn[j] -= h;
                    let (fu, fd) = (self.predict(&up)?, self.predict(&dn)?);
                    fu.iter().zip(&fd).map(|(a, b)| (a - b) / (2.0 * h)).collect()
                }
            };
            cols.push(col);
        }
        Ok(cols)
    }
}

fn sse_of(y: &[f64], pred: &[f64]) -> f64 {
    y.iter().zip(pred).map(|(a, b)| (a - b) * (a - b)).sum()
}

struct Lm {
    theta: Vec<f64>,
    sse: f64,
    iterations: usize,
    converged: bool,
}

fn levenberg_marquardt(pr: &mut Problem, init: Vec<f64>, max_iter: usize) -> Result<Lm, RegressError> {
    let p = init.len();
    let mut theta = init;
    let mut pred = pr.predict(&theta)?;
    let mut sse = sse_of(pr.y, &pred);
    if !sse.is_finite() {
        return Err(RegressError::NonFinite);
    }
    let mut lambda = 1e-3;
    let mut iterations = 0;
    let mut converged = false;
    while iterations < max_iter {
        if sse == 0.0 {
            converged = true;
            break;
        }
        iterations += 1;
        let cols = pr.jacobian(&theta)?;
        let r: Vec<f64> = pr.y.iter().zip(&pred).map(|(a, b)| a - b).collect();
        let jtj = normal_matrix(&cols);
        let g: Vec<f64> = cols.iter().map(|c| c.iter().zip(&r).map(|(a, b)| a * b).sum()).collect();
        if g.iter().any(|v| !v.is_finite()) {
            return Err(RegressError::NonFinite);
        }
        let mut accepted = None;
        for _ in 0..40 {
            let mut m = jtj.clone();
            for k in 0..p {
                m[k][k] += lambda * jtj[k][k].max(1e-12);
            }
            if let Some(delta) = solve(&m, &g) {
                let cand: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
                if let Ok(pc) = pr.predict(&cand) {
                    let sc = sse_of(pr.y, &pc);
                    if sc.is_finite() && sc < sse {
                        accepted = Some((cand, pc, sc, delta));
                        break;
                    }
                }
            }
            lambda *= 10.0;
            if lambda > 1e14 {
                break;
            }
        }
        match accepted {
            None => {
                // No step improves: a (numerical) minimum.
                converged = true;
                break;
            }
            Some((cand, pc, sc, delta)) => {
                let drop = (sse - sc) / sse.max(1e-300);
                let step_small = delta.iter().zip(&cand).all(|(d, t)| d.abs() <= 1e-11 * (t.abs() + 1e-11));
                theta = cand;
                pred = pc;
                sse = sc;
                lambda = (lambda / 10.0).max(1e-12);
                if drop <= 1e-13 || step_small {
                    converged = true;
                    break;
                }
            }
        }
    }
    Ok(Lm { theta, sse, iterations, converged })
}

fn std_errors(cols: &[Vec<f64>], sse: f64, n: usize) -> Option<Vec<f64>> {
    let p = cols.len();
    if n <= p {
        return None;
    }
    let inv = invert(&normal_matrix(cols))?;
    let s2 = sse / (n - p) as f64;
    let se: Vec<f64> = (0..p).map(|i| (s2 * inv[i][i]).sqrt()).collect();
    se.iter().all(|v| v.is_finite()).then_some(se)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    y: &[f64],
    fitted: Vec<f64>,
    params: &[String],
    theta: &[f64],
    model: &Expr,
    data_vars: Vec<String>,
    method: FitMethod,
    iterations: usize,
    converged: bool,
    cols: Option<&[Vec<f64>]>,
) -> Result<FitResult, RegressError> {
    let n = y.len();
    if fitted.iter().any(|v| !v.is_finite()) {
        return Err(RegressError::NonFinite);
    }
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let mean = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    let r2 = if sst > 0.0 {
        1.0 - sse / sst
    } else if sse <= 1e-20 {
        1.0
    } else {
        f64::NAN
    };
    Ok(FitResult {
        params: params.iter().cloned().zip(theta.iter().copied()).collect(),
        r2,
        rmse: (sse / n as f64).sqrt(),
        sse,
        residuals,
        fitted,
        n,
        std_errors: cols.and_then(|c| std_errors(c, sse, n)),
        iterations,
        converged,
        method,
        model: model.clone(),
        data_vars,
    })
}

// ---------------------------------------------------------------- public API

/// Fits `y ≈ model` over the data `lists`, choosing a closed-form linear solve when the model is
/// linear in `params` and Levenberg-Marquardt otherwise. See [`fit_with`] for starting values.
pub fn fit(y: &[f64], model: &Expr, params: &[String], lists: &Lists) -> Result<FitResult, RegressError> {
    fit_with(y, model, params, lists, &FitOptions::default())
}

pub fn fit_with(
    y: &[f64],
    model: &Expr,
    params: &[String],
    lists: &Lists,
    opts: &FitOptions,
) -> Result<FitResult, RegressError> {
    if params.is_empty() {
        return Err(RegressError::NoParams);
    }
    let unique: BTreeSet<&String> = params.iter().collect();
    if unique.len() != params.len() {
        return Err(RegressError::BadModel("a parameter is listed twice".into()));
    }
    for p in params {
        if !model.contains_var(p) {
            return Err(RegressError::BadModel(format!("the parameter '{p}' does not appear in the model")));
        }
        if lists.contains_key(p) {
            return Err(RegressError::BadModel(format!("'{p}' is both a list and a parameter")));
        }
    }
    let n = y.len();
    if n < params.len() {
        return Err(RegressError::TooFewPoints { have: n, need: params.len() });
    }
    if y.iter().any(|v| !v.is_finite()) || lists.values().any(|l| l.iter().any(|v| !v.is_finite())) {
        return Err(RegressError::NonFinite);
    }
    for l in lists.values() {
        if l.len() != n {
            return Err(RegressError::LengthMismatch { a: l.len(), b: n });
        }
    }
    let data_vars: Vec<String> = model.free_vars().into_iter().filter(|v| lists.contains_key(v)).collect();
    let mut bind = Bindings::new();
    for (name, l) in lists {
        bind.set(name, Value::List(l.clone()));
    }
    let jac: Vec<Option<Expr>> = params.iter().map(|p| calculus::try_deriv(model, p).ok()).collect();
    let mut pr = Problem { y, model, params, bind, jac };
    let p = params.len();

    let linear = pr
        .jac
        .iter()
        .all(|j| j.as_ref().is_some_and(|e| params.iter().all(|q| !e.contains_var(q))));
    if linear {
        let zeros = vec![0.0; p];
        let g0 = pr.predict(&zeros)?;
        let cols = pr.jacobian(&zeros)?;
        if g0.iter().chain(cols.iter().flatten()).any(|v| !v.is_finite()) {
            return Err(RegressError::NonFinite);
        }
        let target: Vec<f64> = y.iter().zip(&g0).map(|(a, b)| a - b).collect();
        let theta = lstsq(&cols, &target)?;
        let fitted = pr.predict(&theta)?;
        return finish(y, fitted, params, &theta, model, data_vars, FitMethod::Linear, 0, true, Some(&cols));
    }

    // Nonlinear: the caller's (or default) start first, then a few alternates if it fails.
    let start = |fill: Option<f64>| -> Vec<f64> {
        params
            .iter()
            .map(|n| opts.init.get(n).copied().unwrap_or(fill.unwrap_or(1.0)))
            .collect()
    };
    let mut best: Option<Lm> = None;
    let mut last_err = RegressError::NonFinite;
    let fills = [None, Some(0.1), Some(10.0), Some(-1.0), Some(0.5), Some(-0.1), Some(2.0), Some(100.0), Some(0.01)];
    for fill in fills {
        match levenberg_marquardt(&mut pr, start(fill), opts.max_iter) {
            Ok(out) => {
                let good = out.converged;
                if best.as_ref().is_none_or(|b| out.sse < b.sse) {
                    best = Some(out);
                }
                if good {
                    break;
                }
            }
            Err(e) => last_err = e,
        }
    }
    let Some(lm) = best else { return Err(last_err) };
    let fitted = pr.predict(&lm.theta)?;
    let cols = pr.jacobian(&lm.theta).ok();
    finish(
        y,
        fitted,
        params,
        &lm.theta,
        model,
        data_vars,
        FitMethod::NonLinear,
        lm.iterations,
        lm.converged,
        cols.as_deref(),
    )
}

fn v(name: &str) -> Expr {
    Expr::var(name)
}

/// Least-squares polynomial `a_0 + a_1 x + ... + a_degree x^degree` over the lists `x` and `y`.
/// The coefficients are the parameters `a_0 ..= a_degree`; the model is in terms of `x`.
pub fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Result<FitResult, RegressError> {
    if degree > 30 {
        return Err(RegressError::BadModel("polynomial degree is limited to 30".into()));
    }
    let names: Vec<String> = (0..=degree).map(|k| format!("a_{k}")).collect();
    let mut model = v("a_0");
    for (k, name) in names.iter().enumerate().skip(1) {
        let xk = if k == 1 { v("x") } else { Expr::bin(BinOp::Pow, v("x"), Expr::num(k as f64)) };
        model = Expr::bin(BinOp::Add, model, Expr::bin(BinOp::Mul, v(name), xk));
    }
    let mut lists = Lists::new();
    lists.insert("x".into(), x.to_vec());
    fit(y, &model, &names, &lists)
}

/// Exponential `y = a e^(b x)`, logarithmic `y = a + b ln x` or power `y = a x^b` through a
/// change of variables (so it needs `y > 0` and/or `x > 0` as the form requires). The reported
/// `r2`, `rmse` and residuals are on the original `y` scale. Parameters are `a` and `b`; the
/// model is in terms of `x`.
pub fn fit_linearised(kind: Linearisation, x: &[f64], y: &[f64]) -> Result<FitResult, RegressError> {
    if x.len() != y.len() {
        return Err(RegressError::LengthMismatch { a: x.len(), b: y.len() });
    }
    let n = y.len();
    if n < 2 {
        return Err(RegressError::TooFewPoints { have: n, need: 2 });
    }
    if x.iter().chain(y).any(|t| !t.is_finite()) {
        return Err(RegressError::NonFinite);
    }
    let positive = |what: &str, d: &[f64]| {
        if d.iter().all(|t| *t > 0.0) {
            Ok(())
        } else {
            Err(RegressError::Domain(format!("{what} must be positive for this fit")))
        }
    };
    let ones = vec![1.0; n];
    let ln = |d: &[f64]| d.iter().map(|t| t.ln()).collect::<Vec<f64>>();
    let (a, b, model) = match kind {
        Linearisation::Exponential => {
            positive("y", y)?;
            let c = lstsq(&[ones, x.to_vec()], &ln(y))?;
            let model = Expr::bin(
                BinOp::Mul,
                v("a"),
                Expr::bin(BinOp::Pow, v("e"), Expr::bin(BinOp::Mul, v("b"), v("x"))),
            );
            (c[0].exp(), c[1], model)
        }
        Linearisation::Logarithmic => {
            positive("x", x)?;
            let c = lstsq(&[ones, ln(x)], y)?;
            let model = Expr::bin(BinOp::Add, v("a"), Expr::bin(BinOp::Mul, v("b"), Expr::call("ln", vec![v("x")])));
            (c[0], c[1], model)
        }
        Linearisation::Power => {
            positive("x", x)?;
            positive("y", y)?;
            let c = lstsq(&[ones, ln(x)], &ln(y))?;
            let model = Expr::bin(BinOp::Mul, v("a"), Expr::bin(BinOp::Pow, v("x"), v("b")));
            (c[0].exp(), c[1], model)
        }
    };
    let names = vec!["a".to_string(), "b".to_string()];
    let mut lists = Lists::new();
    lists.insert("x".into(), x.to_vec());
    let mut bind = Bindings::new();
    bind.set("x", Value::List(x.to_vec()));
    bind.set("a", Value::Num(a));
    bind.set("b", Value::Num(b));
    let fitted = match eval_value(&model, &bind).map_err(|e| RegressError::Eval(e.to_string()))? {
        Value::List(l) => l,
        _ => return Err(RegressError::BadModel("the model must evaluate to numbers".into())),
    };
    finish(y, fitted, &names, &[a, b], &model, vec!["x".into()], FitMethod::Linearised(kind), 0, true, None)
}

/// Names a regression `lhs ~ model` treats as parameters: free names of either side that are not
/// defined in `defs` (list names and constants are defined there; an undefined name is unknown).
pub fn infer_params(lhs: &Expr, model: &Expr, defs: &Defs) -> Vec<String> {
    let mut names: BTreeSet<String> = lhs.free_vars();
    names.extend(model.free_vars());
    names.into_iter().filter(|n| !defs.has_var(n)).collect()
}

/// Fits a parsed regression item `lhs ~ model` against a document's definitions.
///
/// * `params`: the names to fit. `None` infers them with [`infer_params`]. Parameters override a
///   same-named definition (so a slider `a` the document already made, or the fitted value
///   injected by [`FitResult::apply_to`] last time, does not turn the parameter into a constant).
/// * Every other free name that `defs` defines as a list is fed to the fit as data; scalars and
///   functions are inlined.
///
/// The result's `model` is the resolved model, so `curve_expr` plots against the same `defs`.
pub fn fit_regression(reg: &Expr, defs: &Defs, params: Option<&[String]>) -> Result<FitResult, RegressError> {
    fit_regression_with(reg, defs, params, &FitOptions::default())
}

pub fn fit_regression_with(
    reg: &Expr,
    defs: &Defs,
    params: Option<&[String]>,
    opts: &FitOptions,
) -> Result<FitResult, RegressError> {
    let Expr::Call(name, args) = reg else {
        return Err(RegressError::BadModel("not a regression (expected lhs ~ model)".into()));
    };
    let (lhs, model) = match (name.as_str(), args.as_slice()) {
        ("regress", [l, m]) => (l, m),
        _ => return Err(RegressError::BadModel("not a regression (expected lhs ~ model)".into())),
    };
    let params: Vec<String> = match params {
        Some(p) => p.to_vec(),
        None => infer_params(lhs, model, defs),
    };
    let mut names: BTreeSet<String> = lhs.free_vars();
    names.extend(model.free_vars());
    let eval_bind = Bindings::new().with_angle(defs.angle());
    let mut lists = Lists::new();
    let mut hidden: Vec<String> = params.clone();
    for n in &names {
        if params.contains(n) || !defs.has_var(n) {
            continue;
        }
        let body = defs.resolve(&Expr::var(n)).map_err(|e| RegressError::BadModel(e.to_string()))?;
        if let Ok(Value::List(l)) = eval_value(&body, &eval_bind) {
            lists.insert(n.clone(), l);
            hidden.push(n.clone());
        }
    }
    let scoped = defs.without_vars(&hidden);
    let model = scoped.resolve(model).map_err(|e| RegressError::BadModel(e.to_string()))?;
    let lhs = scoped.resolve(lhs).map_err(|e| RegressError::BadModel(e.to_string()))?;
    let mut bind = eval_bind;
    for (n, l) in &lists {
        bind.set(n, Value::List(l.clone()));
    }
    let y = match eval_value(&lhs, &bind).map_err(|e| RegressError::Eval(e.to_string()))? {
        Value::List(l) => l,
        _ => return Err(RegressError::BadModel("the left side of ~ must be a list".into())),
    };
    fit_with(&y, &model, &params, &lists, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn p(s: &str) -> Expr {
        parse(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"))
    }

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    fn lists(items: &[(&str, &[f64])]) -> Lists {
        items.iter().map(|(n, l)| (n.to_string(), l.to_vec())).collect()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    const X: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    #[test]
    fn simple_linear_recovers_exact_line() {
        let y: Vec<f64> = X.iter().map(|x| 2.0 * x + 1.0).collect();
        let r = fit(&y, &p("a x_1 + b"), &names(&["a", "b"]), &lists(&[("x_1", &X)])).unwrap();
        assert_eq!(r.method, FitMethod::Linear);
        assert!(close(r.param("a").unwrap(), 2.0, 1e-12));
        assert!(close(r.param("b").unwrap(), 1.0, 1e-12));
        assert!(close(r.r2, 1.0, 1e-12));
        assert!(r.rmse < 1e-12);
        assert!(r.residuals.iter().all(|e| e.abs() < 1e-12));
        assert_eq!(r.residuals.len(), 6);
        assert_eq!(r.n, 6);
        assert!(r.converged);
    }

    #[test]
    fn noisy_line_matches_closed_form() {
        let y = [2.2, 2.8, 3.6, 4.5, 5.1, 5.9];
        let r = fit(&y, &p("a x + b"), &names(&["a", "b"]), &lists(&[("x", &X)])).unwrap();
        let n = 6.0;
        let (mx, my) = (X.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = X.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = X.iter().map(|a| (a - mx) * (a - mx)).sum();
        let slope = sxy / sxx;
        let icpt = my - slope * mx;
        assert!(close(r.param("a").unwrap(), slope, 1e-12));
        assert!(close(r.param("b").unwrap(), icpt, 1e-12));
        let sse: f64 = X.iter().zip(&y).map(|(a, b)| (b - (slope * a + icpt)).powi(2)).sum();
        let sst: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        assert!(close(r.r2, 1.0 - sse / sst, 1e-12));
        assert!(close(r.rmse, (sse / n).sqrt(), 1e-12));
        assert!(close(r.sse, sse, 1e-12));
        // residuals sum to zero for a model with an intercept
        assert!(r.residuals.iter().sum::<f64>().abs() < 1e-12);
        // standard errors: se(slope) = s / sqrt(Sxx)
        let s = (sse / (n - 2.0)).sqrt();
        let se = r.std_errors.as_ref().unwrap();
        assert!(close(se[0], s / sxx.sqrt(), 1e-9), "{se:?}");
    }

    #[test]
    fn multi_term_linear() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0];
        let y: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| 3.0 + 2.0 * a - 0.5 * b).collect();
        let r = fit(&y, &p("c + a x_1 + b x_2"), &names(&["a", "b", "c"]), &lists(&[("x_1", &x1), ("x_2", &x2)])).unwrap();
        assert!(close(r.param("a").unwrap(), 2.0, 1e-10));
        assert!(close(r.param("b").unwrap(), -0.5, 1e-10));
        assert!(close(r.param("c").unwrap(), 3.0, 1e-10));
        assert_eq!(r.data_vars, vec!["x_1", "x_2"]);
        assert!(r.curve_expr("x").is_none(), "two lists have no single curve");
    }

    #[test]
    fn linear_in_parameters_with_nonlinear_basis() {
        let y: Vec<f64> = X.iter().map(|x| 1.5 * x.sin() - 2.0 * x.cos() + 0.25 * x * x).collect();
        let r = fit(&y, &p("a sin(x) + b cos(x) + c x^2"), &names(&["a", "b", "c"]), &lists(&[("x", &X)])).unwrap();
        assert_eq!(r.method, FitMethod::Linear);
        assert!(close(r.param("a").unwrap(), 1.5, 1e-9));
        assert!(close(r.param("b").unwrap(), -2.0, 1e-9));
        assert!(close(r.param("c").unwrap(), 0.25, 1e-9));
    }

    #[test]
    fn no_intercept_and_offset_models() {
        let y: Vec<f64> = X.iter().map(|x| 3.0 * x).collect();
        let r = fit(&y, &p("a x"), &names(&["a"]), &lists(&[("x", &X)])).unwrap();
        assert!(close(r.param("a").unwrap(), 3.0, 1e-12));
        // a fixed offset in the model is respected: y - 10 = a x
        let y: Vec<f64> = X.iter().map(|x| 10.0 + 3.0 * x).collect();
        let r = fit(&y, &p("10 + a x"), &names(&["a"]), &lists(&[("x", &X)])).unwrap();
        assert!(close(r.param("a").unwrap(), 3.0, 1e-12));
        assert!(close(r.r2, 1.0, 1e-12));
    }

    #[test]
    fn polynomial_fits() {
        let x: Vec<f64> = (-5..=5).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.0 - 2.0 * x + 0.5 * x * x * x).collect();
        let r = polyfit(&x, &y, 3).unwrap();
        let want = [1.0, -2.0, 0.0, 0.5];
        for (k, w) in want.iter().enumerate() {
            let got = r.param(&format!("a_{k}")).unwrap();
            assert!((got - w).abs() < 1e-9, "a_{k} = {got}");
        }
        assert!(r.r2 > 1.0 - 1e-12);
        // a quadratic cannot fit a cubic exactly
        let q = polyfit(&x, &y, 2).unwrap();
        assert!(q.r2 < 0.5 && q.r2 >= 0.0, "{}", q.r2);
        // the curve evaluates like the data
        let curve = r.fitted_curve("x").unwrap();
        let prog = crate::compile::compile(&curve, &["x"], crate::compile::Angle::Rad).unwrap();
        assert!((prog.eval(&[1.5]) - (1.0 - 3.0 + 0.5 * 3.375)).abs() < 1e-8);
        assert!(polyfit(&x, &y[..3], 2).is_err());
        assert!(matches!(polyfit(&x[..2], &y[..2], 3), Err(RegressError::TooFewPoints { .. })));
    }

    #[test]
    fn linearised_fits_recover_noiseless_data() {
        let x = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|x| 2.5 * (0.7f64 * x).exp()).collect();
        let r = fit_linearised(Linearisation::Exponential, &x, &y).unwrap();
        assert!(close(r.param("a").unwrap(), 2.5, 1e-10) && close(r.param("b").unwrap(), 0.7, 1e-10));
        assert!(r.r2 > 1.0 - 1e-12);
        assert_eq!(r.method, FitMethod::Linearised(Linearisation::Exponential));

        let y: Vec<f64> = x.iter().map(|x| 4.0 - 1.5 * x.ln()).collect();
        let r = fit_linearised(Linearisation::Logarithmic, &x, &y).unwrap();
        assert!(close(r.param("a").unwrap(), 4.0, 1e-10) && close(r.param("b").unwrap(), -1.5, 1e-10));

        let y: Vec<f64> = x.iter().map(|x| 3.0 * x.powf(1.5)).collect();
        let r = fit_linearised(Linearisation::Power, &x, &y).unwrap();
        assert!(close(r.param("a").unwrap(), 3.0, 1e-10) && close(r.param("b").unwrap(), 1.5, 1e-10));
        // r2 is on the original scale
        assert!(r.residuals.iter().all(|e| e.abs() < 1e-9));
    }

    #[test]
    fn linearisation_domain_errors() {
        let x = [1.0, 2.0, 3.0];
        assert!(matches!(fit_linearised(Linearisation::Exponential, &x, &[1.0, -2.0, 3.0]), Err(RegressError::Domain(_))));
        assert!(matches!(fit_linearised(Linearisation::Logarithmic, &[0.0, 1.0, 2.0], &x), Err(RegressError::Domain(_))));
        assert!(matches!(fit_linearised(Linearisation::Power, &x, &[1.0, 0.0, 3.0]), Err(RegressError::Domain(_))));
        assert!(fit_linearised(Linearisation::Power, &x, &x[..2]).is_err());
        assert!(matches!(fit_linearised(Linearisation::Power, &[1.0], &[1.0]), Err(RegressError::TooFewPoints { .. })));
    }

    #[test]
    fn nonlinear_exponential_with_noise() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
        // deterministic "noise"
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, x)| 2.0 * (0.5 * x).exp() + 0.01 * ((i * 7 % 5) as f64 - 2.0))
            .collect();
        let r = fit(&y, &p("a e^(b x)"), &names(&["a", "b"]), &lists(&[("x", &x)])).unwrap();
        assert_eq!(r.method, FitMethod::NonLinear);
        assert!(r.converged);
        assert!(close(r.param("a").unwrap(), 2.0, 0.02), "{:?}", r.params);
        assert!(close(r.param("b").unwrap(), 0.5, 0.01), "{:?}", r.params);
        assert!(r.r2 > 0.999);
        // the result is a minimum: perturbing a parameter does not reduce SSE
        let model = &r.model;
        let sse_at = |a: f64, b: f64| {
            let mut e = model.subst("a", &Expr::num(a));
            e = e.subst("b", &Expr::num(b));
            let prog = crate::compile::compile(&e, &["x"], crate::compile::Angle::Rad).unwrap();
            x.iter().zip(&y).map(|(x, y)| (y - prog.eval(&[*x])).powi(2)).sum::<f64>()
        };
        let (a, b) = (r.param("a").unwrap(), r.param("b").unwrap());
        for (da, db) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-4), (0.0, -1e-4)] {
            assert!(sse_at(a + da, b + db) >= r.sse * (1.0 - 1e-9));
        }
        // refit from the closed-form linearised start gives the same answer
        let mut opts = FitOptions::default();
        opts.init.insert("a".into(), 1.0);
        opts.init.insert("b".into(), 0.1);
        let r2 = fit_with(&y, &p("a e^(b x)"), &names(&["a", "b"]), &lists(&[("x", &x)]), &opts).unwrap();
        assert!(close(r2.param("b").unwrap(), r.param("b").unwrap(), 1e-6));
    }

    #[test]
    fn nonlinear_power_and_rational() {
        let x: Vec<f64> = (1..=15).map(|i| i as f64 * 0.4).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.7 * x.powf(0.8)).collect();
        let r = fit(&y, &p("a x^b"), &names(&["a", "b"]), &lists(&[("x", &x)])).unwrap();
        assert!(close(r.param("a").unwrap(), 1.7, 1e-6) && close(r.param("b").unwrap(), 0.8, 1e-6), "{:?}", r.params);
        // saturation curve a x / (b + x)
        let y: Vec<f64> = x.iter().map(|x| 5.0 * x / (2.0 + x)).collect();
        let r = fit(&y, &p("a x/(b+x)"), &names(&["a", "b"]), &lists(&[("x", &x)])).unwrap();
        assert!(close(r.param("a").unwrap(), 5.0, 1e-6) && close(r.param("b").unwrap(), 2.0, 1e-6), "{:?}", r.params);
        assert!(r.rmse < 1e-6);
    }

    #[test]
    fn nonlinear_with_a_start_and_sinusoid() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|x| 3.0 * (1.3 * x + 0.4).sin() + 1.0).collect();
        let mut opts = FitOptions::default();
        opts.init.insert("a".into(), 2.5);
        opts.init.insert("b".into(), 1.2);
        opts.init.insert("c".into(), 0.3);
        opts.init.insert("d".into(), 0.5);
        let r = fit_with(&y, &p("a sin(b x + c) + d"), &names(&["a", "b", "c", "d"]), &lists(&[("x", &x)]), &opts).unwrap();
        assert!(close(r.param("a").unwrap(), 3.0, 1e-6), "{:?}", r.params);
        assert!(close(r.param("b").unwrap(), 1.3, 1e-6));
        assert!(close(r.param("c").unwrap(), 0.4, 1e-5));
        assert!(close(r.param("d").unwrap(), 1.0, 1e-6));
    }

    #[test]
    fn finite_difference_fallback_when_no_symbolic_derivative() {
        // d/d(mean) of normalpdf has no symbolic rule, so the Jacobian is numeric.
        let x: Vec<f64> = (-10..=14).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|x| 3.0 * crate::stats::normalpdf(*x, 2.0, 1.5)).collect();
        let mut opts = FitOptions::default();
        opts.init.insert("m".into(), 1.5);
        opts.init.insert("s".into(), 1.0);
        opts.init.insert("k".into(), 2.0);
        let r = fit_with(&y, &p("k normalpdf(x, m, s)"), &names(&["k", "m", "s"]), &lists(&[("x", &x)]), &opts).unwrap();
        assert!(close(r.param("k").unwrap(), 3.0, 1e-5), "{:?}", r.params);
        assert!(close(r.param("m").unwrap(), 2.0, 1e-5));
        assert!(close(r.param("s").unwrap(), 1.5, 1e-5));
    }

    #[test]
    fn errors_are_typed() {
        let x = lists(&[("x", &X)]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(fit(&y, &p("2x"), &[], &x), Err(RegressError::NoParams));
        assert!(matches!(fit(&y, &p("a x"), &names(&["a", "b"]), &x), Err(RegressError::BadModel(_))));
        assert!(matches!(fit(&y, &p("a x + a"), &names(&["a", "a"]), &x), Err(RegressError::BadModel(_))));
        assert!(matches!(fit(&y[..4], &p("a x"), &names(&["a"]), &x), Err(RegressError::LengthMismatch { .. })));
        assert!(matches!(
            fit(&y[..1], &p("a x + b"), &names(&["a", "b"]), &lists(&[("x", &X[..1])])),
            Err(RegressError::TooFewPoints { have: 1, need: 2 })
        ));
        // (a+b) x: the two parameters cannot be separated
        assert_eq!(fit(&y, &p("(a+b) x"), &names(&["a", "b"]), &x), Err(RegressError::Singular));
        // constant data cannot give both a slope and an intercept... with a constant x list
        let c = lists(&[("x", &[2.0; 6])]);
        assert_eq!(fit(&y, &p("a x + b"), &names(&["a", "b"]), &c), Err(RegressError::Singular));
        // non-finite data
        let mut bad = y;
        bad[2] = f64::NAN;
        assert_eq!(fit(&bad, &p("a x"), &names(&["a"]), &x), Err(RegressError::NonFinite));
        // an unbound name is an evaluation error
        assert!(matches!(fit(&y, &p("a q"), &names(&["a"]), &x), Err(RegressError::Eval(_))));
        // a list that is also a parameter
        assert!(matches!(fit(&y, &p("a x"), &names(&["a", "x"]), &x), Err(RegressError::BadModel(_))));
        // the model is not finite anywhere
        assert_eq!(fit(&y, &p("a ln(0-x)"), &names(&["a"]), &x), Err(RegressError::NonFinite));
        // text of every error
        for e in [RegressError::NoParams, RegressError::Singular, RegressError::NonFinite] {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn constant_response() {
        let y = [4.0; 6];
        let r = fit(&y, &p("a x + b"), &names(&["a", "b"]), &lists(&[("x", &X)])).unwrap();
        assert!(close(r.param("b").unwrap(), 4.0, 1e-10) && r.param("a").unwrap().abs() < 1e-10);
        assert_eq!(r.r2, 1.0);
        // a no-intercept fit to a constant is not exact: r2 is undefined
        let r = fit(&y, &p("a x"), &names(&["a"]), &lists(&[("x", &X)])).unwrap();
        assert!(r.r2.is_nan());
    }

    #[test]
    fn exactly_determined_has_no_standard_errors() {
        let r = fit(&[1.0, 3.0], &p("a x + b"), &names(&["a", "b"]), &lists(&[("x", &[1.0, 2.0])])).unwrap();
        assert!(r.std_errors.is_none());
        assert!(close(r.param("a").unwrap(), 2.0, 1e-12));
    }

    #[test]
    fn large_data_set() {
        let x: Vec<f64> = (0..5000).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = x.iter().map(|x| 0.3 * x - 7.0).collect();
        let r = fit(&y, &p("a x + b"), &names(&["a", "b"]), &lists(&[("x", &x)])).unwrap();
        assert!(close(r.param("a").unwrap(), 0.3, 1e-10) && close(r.param("b").unwrap(), -7.0, 1e-10));
    }

    // ---------------------------------------------------------------- syntax and documents

    #[test]
    fn tilde_parses_prints_and_classifies() {
        use crate::analyze::{analyze, Kind};
        let e = p("y_1 ~ a x_1 + b");
        let Expr::Call(name, args) = &e else { panic!("{e:?}") };
        assert_eq!(name, "regress");
        assert_eq!(args.len(), 2);
        assert_eq!(e.to_string(), "y_1 ~ a * x_1 + b");
        assert_eq!(p(&e.to_string()), e);
        assert_eq!(crate::print::to_latex(&e), "y_{1}\\sim a\\cdot x_{1}+b");
        assert_eq!(p("y_{1}\\sim a\\cdot x_{1}+b"), e);
        let a = analyze(&e, &BTreeSet::new());
        match a.kind {
            Kind::Regression { lhs, model } => {
                assert_eq!(lhs, Expr::var("y_1"));
                assert_eq!(model, p("a x_1 + b"));
            }
            k => panic!("{k:?}"),
        }
        // the free names (and so slider candidates) include the parameters
        assert!(a.slider_candidates.contains("a") && a.slider_candidates.contains("b"));
        assert!(parse("a ~ b ~ c").is_err());
        assert!(parse("a ~ b = c").is_err());
        assert!(parse("a ~").is_err());
        // ~ binds like a relation: below arithmetic
        assert_eq!(p("y_1 ~ x_1 + 1").to_string(), "y_1 ~ x_1 + 1");
    }

    fn doc_defs(lines: &[(&str, &str)]) -> Defs {
        let mut d = Defs::new();
        for (n, body) in lines {
            d.define_var(n, p(body));
        }
        d
    }

    #[test]
    fn fit_regression_from_document_definitions() {
        let defs = doc_defs(&[("x_1", "[1,2,3,4,5]"), ("y_1", "[3,5,7,9,11]")]);
        let reg = p("y_1 ~ a x_1 + b");
        let r = fit_regression(&reg, &defs, None).unwrap();
        assert!(close(r.param("a").unwrap(), 2.0, 1e-10) && close(r.param("b").unwrap(), 1.0, 1e-10));
        assert_eq!(r.params.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        // list comprehensions as data, and a transformed left side
        let defs = doc_defs(&[("x_1", "[1...6]"), ("y_1", "[2*e^(0.5 k) for k=[1...6]]")]);
        let r = fit_regression(&p("ln(y_1) ~ a x_1 + b"), &defs, None).unwrap();
        assert!(close(r.param("a").unwrap(), 0.5, 1e-9));
        assert!(close(r.param("b").unwrap(), 2.0f64.ln(), 1e-9));
    }

    #[test]
    fn injected_parameters_draw_the_curve() {
        let mut defs = doc_defs(&[("x_1", "[1,2,3,4]"), ("y_1", "[5,7,9,11]")]);
        // The document already has a slider `a` and a scalar `b`; explicit params override them.
        defs.set_slider("a", 99.0);
        defs.define_var("k", p("1"));
        let params = names(&["a", "b"]);
        let r = fit_regression(&p("y_1 ~ a x_1 + b + k - k"), &defs, Some(&params)).unwrap();
        assert!(close(r.param("a").unwrap(), 2.0, 1e-9) && close(r.param("b").unwrap(), 3.0, 1e-9));
        r.apply_to(&mut defs);
        // the curve in terms of x resolves against the document and passes through the data
        let curve = r.curve_expr("x").unwrap();
        let resolved = defs.resolve(&curve).unwrap();
        let prog = crate::compile::compile(&resolved, &["x"], crate::compile::Angle::Rad).unwrap();
        assert!((prog.eval(&[2.0]) - 7.0).abs() < 1e-9);
        assert!((prog.eval(&[10.0]) - 23.0).abs() < 1e-8);
        // and a refit after injection finds the same answer (the slider does not freeze it)
        let again = fit_regression(&p("y_1 ~ a x_1 + b"), &defs, Some(&params)).unwrap();
        assert!(close(again.param("a").unwrap(), 2.0, 1e-9));
        let fm = r.fitted_model();
        assert!(!fm.contains_var("a") && !fm.contains_var("b"));
    }

    #[test]
    fn fit_regression_errors() {
        let defs = doc_defs(&[("x_1", "[1,2,3]"), ("y_1", "[1,2,3]")]);
        assert!(matches!(fit_regression(&p("x_1+1"), &defs, None), Err(RegressError::BadModel(_))));
        assert!(matches!(fit_regression(&p("x_1 ~ y_1"), &defs, None), Err(RegressError::NoParams)));
        // a scalar on the left
        assert!(matches!(fit_regression(&p("2 ~ a x_1"), &defs, None), Err(RegressError::BadModel(_))));
        // function definitions and derivatives in the model are resolved
        let mut defs = doc_defs(&[("x_1", "[1,2,3,4]"), ("y_1", "[1,4,9,16]")]);
        defs.define_func("g", names(&["u"]), p("u^2"));
        let ctx = crate::parse::ParseCtx::new().with_function("g");
        let reg = crate::parse::parse_with("y_1 ~ a g(x_1)", &ctx).unwrap();
        let r = fit_regression(&reg, &defs, None).unwrap();
        assert!(close(r.param("a").unwrap(), 1.0, 1e-10));
        let r = fit_regression(&p("y_1 ~ a d/dx x^2 + b"), &doc_defs(&[("x", "[1,2,3,4]"), ("y_1", "[3,5,7,9]")]), None);
        // `x` is a list here, so d/dx would differentiate with x as the variable: the call
        // resolves to 2x and the fit is a straight line.
        let r = r.unwrap();
        assert!(close(r.param("a").unwrap(), 1.0, 1e-9) && close(r.param("b").unwrap(), 1.0, 1e-9));
    }
}
