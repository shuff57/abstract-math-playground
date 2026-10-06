//! Calculus: symbolic differentiation and numeric integration / summation helpers.
//!
//! * [`deriv`] / [`try_deriv`] differentiate an [`Expr`] symbolically with respect to a variable
//!   and simplify the result, so the derivative is plain AST that compiles, prints and (for the
//!   GPU path) becomes WGSL like any other expression.
//! * [`expand`] rewrites every `deriv(f, x)` / `deriv(f, x, at)` call in a tree into its
//!   symbolic derivative. The resolver and the compiler both call it, so a derivative never
//!   reaches an evaluator as a call.
//! * [`integrate`] is an adaptive Gauss-Kronrod (G7/K15) integrator with a tanh-sinh fallback for
//!   endpoint singularities; it returns `NaN` for divergent or undefined integrals.
//! * [`sum_range`] / [`prod_range`] iterate integer bounds, capped at [`MAX_ITERS`].
//!
//! Derivatives assume the angle unit the caller passes (radians by default): in degree mode
//! `sin(x)` is `sin(x * pi/180)`, so its derivative carries the factor `pi/180`.

use crate::ast::{BinOp, Expr};
use crate::compile::Angle;

/// Most terms a `sum` or `prod` evaluates; a larger range is `NaN`.
pub const MAX_ITERS: f64 = 1e6;

#[derive(Debug, Clone, PartialEq)]
pub enum CalcError {
    /// No differentiation rule for this function (or construct).
    Unsupported(String),
    /// A calculus call was malformed (wrong arity, bound variable is not a name, ...).
    BadArgs(String),
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalcError::Unsupported(n) => write!(f, "cannot differentiate '{n}'"),
            CalcError::BadArgs(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CalcError {}

// ---------------------------------------------------------------- smart constructors

fn num(e: &Expr) -> Option<f64> {
    if let Expr::Num(v) = e {
        Some(*v)
    } else {
        None
    }
}

fn fold(v: f64) -> Option<Expr> {
    v.is_finite().then_some(Expr::Num(v))
}

pub fn neg(a: Expr) -> Expr {
    match a {
        Expr::Num(v) => Expr::Num(-v),
        Expr::Neg(inner) => *inner,
        Expr::Bin(BinOp::Mul, l, r) => match *l {
            Expr::Num(c) => Expr::Bin(BinOp::Mul, Box::new(Expr::Num(-c)), r),
            l => Expr::Neg(Box::new(Expr::Bin(BinOp::Mul, Box::new(l), r))),
        },
        other => Expr::Neg(Box::new(other)),
    }
}

pub fn add(a: Expr, b: Expr) -> Expr {
    if let (Some(x), Some(y)) = (num(&a), num(&b)) {
        if let Some(e) = fold(x + y) {
            return e;
        }
    }
    if num(&a) == Some(0.0) {
        return b;
    }
    if num(&b) == Some(0.0) {
        return a;
    }
    match (a, b) {
        (a, Expr::Neg(i)) => sub(a, *i),
        (Expr::Neg(i), b) => sub(b, *i),
        (a, b) => Expr::bin(BinOp::Add, a, b),
    }
}

pub fn sub(a: Expr, b: Expr) -> Expr {
    if let (Some(x), Some(y)) = (num(&a), num(&b)) {
        if let Some(e) = fold(x - y) {
            return e;
        }
    }
    if num(&b) == Some(0.0) {
        return a;
    }
    if num(&a) == Some(0.0) {
        return neg(b);
    }
    if a == b {
        return Expr::Num(0.0);
    }
    match b {
        Expr::Neg(i) => add(a, *i),
        b => Expr::bin(BinOp::Sub, a, b),
    }
}

pub fn mul(a: Expr, b: Expr) -> Expr {
    if let (Some(x), Some(y)) = (num(&a), num(&b)) {
        if let Some(e) = fold(x * y) {
            return e;
        }
    }
    if num(&a) == Some(0.0) || num(&b) == Some(0.0) {
        return Expr::Num(0.0);
    }
    if num(&a) == Some(1.0) {
        return b;
    }
    if num(&b) == Some(1.0) {
        return a;
    }
    if num(&a) == Some(-1.0) {
        return neg(b);
    }
    if num(&b) == Some(-1.0) {
        return neg(a);
    }
    // Pull signs out so they can cancel.
    match (a, b) {
        (Expr::Neg(x), Expr::Neg(y)) => mul(*x, *y),
        (Expr::Neg(x), y) => neg(mul(*x, y)),
        (x, Expr::Neg(y)) => neg(mul(x, *y)),
        // Constants go first, and nested constants merge: c*(d*x) = (c*d)*x.
        (a, b) => {
            let (a, b) = if num(&b).is_some() && num(&a).is_none() { (b, a) } else { (a, b) };
            if let (Some(c), Expr::Bin(BinOp::Mul, l, r)) = (num(&a), &b) {
                if let Some(d) = num(l) {
                    if let Some(cd) = fold(c * d) {
                        return mul(cd, (**r).clone());
                    }
                }
            }
            Expr::bin(BinOp::Mul, a, b)
        }
    }
}

pub fn div(a: Expr, b: Expr) -> Expr {
    if let (Some(x), Some(y)) = (num(&a), num(&b)) {
        if y != 0.0 {
            if let Some(e) = fold(x / y) {
                return e;
            }
        }
    }
    if num(&b) == Some(1.0) {
        return a;
    }
    if num(&a) == Some(0.0) && num(&b) != Some(0.0) {
        return a;
    }
    if a == b && num(&b) != Some(0.0) {
        return Expr::Num(1.0);
    }
    match (a, b) {
        (Expr::Neg(x), Expr::Neg(y)) => div(*x, *y),
        (Expr::Neg(x), y) => neg(div(*x, y)),
        (x, Expr::Neg(y)) => neg(div(x, *y)),
        (a, b) => Expr::bin(BinOp::Div, a, b),
    }
}

pub fn pow(a: Expr, b: Expr) -> Expr {
    if num(&b) == Some(1.0) {
        return a;
    }
    if num(&b) == Some(0.0) {
        return Expr::Num(1.0);
    }
    if num(&a) == Some(1.0) {
        return a;
    }
    if let (Some(x), Some(y)) = (num(&a), num(&b)) {
        if let Some(e) = fold(x.powf(y)) {
            return e;
        }
    }
    Expr::bin(BinOp::Pow, a, b)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::call(name, args)
}

fn n(v: f64) -> Expr {
    Expr::Num(v)
}

/// Rebuilds `e` bottom-up with the smart constructors: folds constants and removes `0`, `1`
/// identities. It never changes the value of the expression (apart from `0*x` and `x/x`, which
/// follow computer-algebra convention).
pub fn simplify(e: &Expr) -> Expr {
    match e {
        Expr::Num(_) | Expr::Var(_) => e.clone(),
        Expr::Neg(a) => neg(simplify(a)),
        Expr::Bin(op, a, b) => {
            let (a, b) = (simplify(a), simplify(b));
            match op {
                BinOp::Add => add(a, b),
                BinOp::Sub => sub(a, b),
                BinOp::Mul => mul(a, b),
                BinOp::Div => div(a, b),
                BinOp::Pow => pow(a, b),
            }
        }
        Expr::Call(name, args) => Expr::Call(name.clone(), args.iter().map(simplify).collect()),
        Expr::Tuple(items) => Expr::Tuple(items.iter().map(simplify).collect()),
        Expr::List(items) => Expr::List(items.iter().map(simplify).collect()),
        Expr::Rel(r, a, b) => Expr::Rel(*r, Box::new(simplify(a)), Box::new(simplify(b))),
    }
}

// ---------------------------------------------------------------- differentiation

/// `d(expr)/d(var)`, radians. An unsupported subterm becomes a `NaN` constant; use
/// [`try_deriv`] to get the error instead.
pub fn deriv(expr: &Expr, var: &str) -> Expr {
    try_deriv(expr, var).unwrap_or(Expr::Num(f64::NAN))
}

pub fn try_deriv(expr: &Expr, var: &str) -> Result<Expr, CalcError> {
    try_deriv_angle(expr, var, Angle::Rad)
}

/// Like [`try_deriv`] for an expression whose trig functions use `angle`.
pub fn try_deriv_angle(expr: &Expr, var: &str, angle: Angle) -> Result<Expr, CalcError> {
    let d = D { var, k: angle.to_rad() }.d(expr)?;
    Ok(simplify(&d))
}

/// The `n`-th derivative (`n = 0` returns the simplified expression).
pub fn nth_deriv(expr: &Expr, var: &str, order: usize) -> Result<Expr, CalcError> {
    nth_deriv_angle(expr, var, order, Angle::Rad)
}

/// Like [`nth_deriv`] for an expression whose trig functions use `angle`.
pub fn nth_deriv_angle(expr: &Expr, var: &str, order: usize, angle: Angle) -> Result<Expr, CalcError> {
    let mut e = simplify(expr);
    for _ in 0..order {
        e = try_deriv_angle(&e, var, angle)?;
    }
    Ok(e)
}

struct D<'a> {
    var: &'a str,
    /// Radians per angle unit.
    k: f64,
}

impl D<'_> {
    fn free(&self, e: &Expr) -> bool {
        !e.contains_var(self.var)
    }

    /// `k * x`, omitting the factor in radian mode.
    fn scale(&self, e: Expr) -> Expr {
        if self.k == 1.0 {
            e
        } else {
            mul(n(self.k), e)
        }
    }

    fn d(&self, e: &Expr) -> Result<Expr, CalcError> {
        if self.free(e) {
            return Ok(n(0.0));
        }
        match e {
            Expr::Num(_) => Ok(n(0.0)),
            Expr::Var(name) => Ok(n(if name == self.var { 1.0 } else { 0.0 })),
            Expr::Neg(a) => Ok(neg(self.d(a)?)),
            Expr::Bin(op, a, b) => self.bin(*op, a, b),
            Expr::Call(name, args) => self.call(name, args),
            Expr::Tuple(items) => Ok(Expr::Tuple(items.iter().map(|i| self.d(i)).collect::<Result<_, _>>()?)),
            Expr::List(items) => Ok(Expr::List(items.iter().map(|i| self.d(i)).collect::<Result<_, _>>()?)),
            Expr::Rel(..) => Err(CalcError::Unsupported("a comparison".into())),
        }
    }

    fn bin(&self, op: BinOp, a: &Expr, b: &Expr) -> Result<Expr, CalcError> {
        match op {
            BinOp::Add => Ok(add(self.d(a)?, self.d(b)?)),
            BinOp::Sub => Ok(sub(self.d(a)?, self.d(b)?)),
            BinOp::Mul => {
                let (da, db) = (self.d(a)?, self.d(b)?);
                Ok(add(mul(da, b.clone()), mul(a.clone(), db)))
            }
            BinOp::Div => {
                let da = self.d(a)?;
                if self.free(b) {
                    return Ok(div(da, b.clone()));
                }
                let db = self.d(b)?;
                let top = sub(mul(da, b.clone()), mul(a.clone(), db));
                Ok(div(top, pow(b.clone(), n(2.0))))
            }
            BinOp::Pow => {
                if self.free(b) {
                    // d(u^c) = c * u^(c-1) * u'
                    let du = self.d(a)?;
                    let rest = pow(a.clone(), sub(b.clone(), n(1.0)));
                    return Ok(mul(mul(b.clone(), rest), du));
                }
                let db = self.d(b)?;
                let whole = Expr::bin(BinOp::Pow, a.clone(), b.clone());
                if self.free(a) {
                    // d(c^w) = c^w * ln(c) * w'
                    let ln_c = if *a == Expr::var("e") { n(1.0) } else { call("ln", vec![a.clone()]) };
                    return Ok(mul(mul(whole, ln_c), db));
                }
                // d(u^w) = u^w * (w' ln u + w u'/u)
                let du = self.d(a)?;
                let t1 = mul(db, call("ln", vec![a.clone()]));
                let t2 = div(mul(b.clone(), du), a.clone());
                Ok(mul(whole, add(t1, t2)))
            }
        }
    }

    fn call(&self, name: &str, args: &[Expr]) -> Result<Expr, CalcError> {
        let unsupported = || CalcError::Unsupported(name.to_string());
        // Multi-argument and structural functions first.
        match (name, args.len()) {
            ("deriv", 2) | ("deriv", 3) => {
                let expanded = expand(&Expr::Call(name.to_string(), args.to_vec()), self.angle())?;
                return self.d(&expanded);
            }
            ("atan2", 2) => {
                let (y, x) = (&args[0], &args[1]);
                let (dy, dx) = (self.d(y)?, self.d(x)?);
                let top = sub(mul(x.clone(), dy), mul(y.clone(), dx));
                let bot = add(pow(x.clone(), n(2.0)), pow(y.clone(), n(2.0)));
                return Ok(div(top, bot));
            }
            ("max", 2) | ("min", 2) => {
                // max = (a+b)/2 + |a-b|/2, min = (a+b)/2 - |a-b|/2
                let (a, b) = (&args[0], &args[1]);
                let (da, db) = (self.d(a)?, self.d(b)?);
                let mean = div(add(da.clone(), db.clone()), n(2.0));
                let s = call("sign", vec![sub(a.clone(), b.clone())]);
                let half = mul(div(sub(da, db), n(2.0)), s);
                return Ok(if name == "max" { add(mean, half) } else { sub(mean, half) });
            }
            ("mod", 2) => {
                let (a, b) = (&args[0], &args[1]);
                let da = self.d(a)?;
                if self.free(b) {
                    return Ok(da);
                }
                let db = self.d(b)?;
                return Ok(sub(da, mul(db, call("floor", vec![div(a.clone(), b.clone())]))));
            }
            ("normalpdf", 3) => {
                let (u, m, s) = (&args[0], &args[1], &args[2]);
                if !(self.free(m) && self.free(s)) {
                    return Err(unsupported());
                }
                let du = self.d(u)?;
                let slope = neg(div(sub(u.clone(), m.clone()), pow(s.clone(), n(2.0))));
                return Ok(mul(du, mul(slope, Expr::Call(name.to_string(), args.to_vec()))));
            }
            ("int", 4) => return self.integral(args),
            ("sum", 4) => return self.sum(args),
            _ => {}
        }
        if args.len() != 1 {
            return Err(unsupported());
        }
        let u = &args[0];
        let du = self.d(u)?;
        let f = |e: Expr| mul(du.clone(), e);
        let sq = |e: &Expr| pow(e.clone(), n(2.0));
        let me = Expr::call(name, vec![u.clone()]);
        Ok(match name {
            "sin" => f(self.scale(call("cos", vec![u.clone()]))),
            "cos" => f(neg(self.scale(call("sin", vec![u.clone()])))),
            "tan" => f(self.scale(div(n(1.0), sq(&call("cos", vec![u.clone()]))))),
            "sec" => f(self.scale(mul(me, call("tan", vec![u.clone()])))),
            "csc" => f(neg(self.scale(mul(me, call("cot", vec![u.clone()]))))),
            "cot" => f(neg(self.scale(div(n(1.0), sq(&call("sin", vec![u.clone()])))))),
            "asin" | "arcsin" | "acos" | "arccos" => {
                let root = call("sqrt", vec![sub(n(1.0), sq(u))]);
                let inv = div(n(1.0), self.unscale(root));
                if name.starts_with("asin") || name == "arcsin" {
                    f(inv)
                } else {
                    f(neg(inv))
                }
            }
            "atan" | "arctan" => f(div(n(1.0), self.unscale(add(n(1.0), sq(u))))),
            "sinh" => f(call("cosh", vec![u.clone()])),
            "cosh" => f(call("sinh", vec![u.clone()])),
            "tanh" => f(div(n(1.0), sq(&call("cosh", vec![u.clone()])))),
            "exp" => f(me),
            "ln" => div(du, u.clone()),
            "log" => div(du, mul(u.clone(), n(std::f64::consts::LN_10))),
            "sqrt" => div(du, mul(n(2.0), me)),
            "cbrt" => div(du, mul(n(3.0), sq(&me))),
            "abs" => f(call("sign", vec![u.clone()])),
            "floor" | "ceil" | "round" | "sign" | "sgn" => n(0.0),
            _ => return Err(unsupported()),
        })
    }

    fn angle(&self) -> Angle {
        if self.k == 1.0 {
            Angle::Rad
        } else {
            Angle::Deg
        }
    }

    /// Divides by the radians-per-unit factor in degree mode (inverse trig derivatives).
    fn unscale(&self, e: Expr) -> Expr {
        self.scale(e)
    }

    /// Leibniz rule: d/dv of the integral of f(t) from a(v) to b(v).
    fn integral(&self, args: &[Expr]) -> Result<Expr, CalcError> {
        let (f, bound, a, b) = binder_parts(args)?;
        let (da, db) = (self.d(a)?, self.d(b)?);
        let inner = if bound != self.var && f.contains_var(self.var) {
            Expr::call("int", vec![self.d(f)?, Expr::var(bound), a.clone(), b.clone()])
        } else {
            n(0.0)
        };
        let upper = mul(db, f.subst(bound, b));
        let lower = mul(da, f.subst(bound, a));
        Ok(add(sub(upper, lower), inner))
    }

    fn sum(&self, args: &[Expr]) -> Result<Expr, CalcError> {
        let (f, bound, a, b) = binder_parts(args)?;
        if !(self.free(a) && self.free(b)) {
            return Err(CalcError::Unsupported("sum with variable bounds".into()));
        }
        if bound == self.var {
            return Ok(n(0.0));
        }
        Ok(Expr::call("sum", vec![self.d(f)?, Expr::var(bound), a.clone(), b.clone()]))
    }
}

/// `(body, bound variable, lo, hi)` of an `int` / `sum` / `prod` call's arguments.
fn binder_parts(args: &[Expr]) -> Result<(&Expr, &str, &Expr, &Expr), CalcError> {
    match args {
        [f, Expr::Var(v), a, b] => Ok((f, v, a, b)),
        _ => Err(CalcError::BadArgs("expected (expression, variable, from, to)".into())),
    }
}

/// Rewrites every `deriv(f, x)` and `deriv(f, x, at)` call into the symbolic derivative.
pub fn expand(e: &Expr, angle: Angle) -> Result<Expr, CalcError> {
    Ok(match e {
        Expr::Num(_) | Expr::Var(_) => e.clone(),
        Expr::Neg(a) => Expr::Neg(Box::new(expand(a, angle)?)),
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(expand(a, angle)?), Box::new(expand(b, angle)?)),
        Expr::Rel(r, a, b) => Expr::Rel(*r, Box::new(expand(a, angle)?), Box::new(expand(b, angle)?)),
        Expr::Tuple(items) => Expr::Tuple(items.iter().map(|i| expand(i, angle)).collect::<Result<_, _>>()?),
        Expr::List(items) => Expr::List(items.iter().map(|i| expand(i, angle)).collect::<Result<_, _>>()?),
        Expr::Call(name, args) => {
            let args: Vec<Expr> = args.iter().map(|a| expand(a, angle)).collect::<Result<_, _>>()?;
            if name == "deriv" {
                expand_deriv(&args, angle)?
            } else {
                Expr::Call(name.clone(), args)
            }
        }
    })
}

/// The derivative a `deriv(f, x[, at])` call stands for; `args` must already be expanded.
pub fn expand_deriv(args: &[Expr], angle: Angle) -> Result<Expr, CalcError> {
    let var = match args {
        [_, Expr::Var(v)] | [_, Expr::Var(v), _] => v,
        _ => {
            return Err(CalcError::BadArgs(
                "deriv takes (expression, variable) or (expression, variable, point)".into(),
            ))
        }
    };
    let d = try_deriv_angle(&args[0], var, angle)?;
    Ok(match args.get(2) {
        Some(at) => simplify(&d.subst(var, at)),
        None => d,
    })
}

// ---------------------------------------------------------------- sum / prod

/// Integer bounds `(lo, hi)` for a `sum`/`prod`, or `None` (result is `NaN`) when a bound is not
/// finite or the range has more than [`MAX_ITERS`] terms. An empty range has `hi < lo`.
pub fn int_bounds(lo: f64, hi: f64) -> Option<(i64, i64)> {
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let snap = |x: f64, up: bool| {
        let r = x.round();
        if (x - r).abs() < 1e-9 * (1.0 + x.abs()) {
            r
        } else if up {
            x.ceil()
        } else {
            x.floor()
        }
    };
    let (l, h) = (snap(lo, true), snap(hi, false));
    if h - l + 1.0 > MAX_ITERS {
        return None;
    }
    // Bounds beyond the cap are rejected above, but a huge empty range still must not overflow.
    if l.abs() > 1e15 || h.abs() > 1e15 {
        return None;
    }
    Some((l as i64, h as i64))
}

pub fn sum_range(lo: f64, hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    match int_bounds(lo, hi) {
        None => f64::NAN,
        Some((l, h)) => (l..=h).map(|k| f(k as f64)).sum(),
    }
}

pub fn prod_range(lo: f64, hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    match int_bounds(lo, hi) {
        None => f64::NAN,
        Some((l, h)) => (l..=h).fold(1.0, |acc, k| acc * f(k as f64)),
    }
}

// ---------------------------------------------------------------- integration

const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_47,
    0.0,
];
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];
/// 7-point Gauss weights for the nodes `XGK[1]`, `XGK[3]`, `XGK[5]` and the centre.
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

const REL_TOL: f64 = 1e-10;
const ABS_TOL: f64 = 1e-13;
const MAX_SEGMENTS: usize = 120;

type Integrand<'a> = &'a mut dyn FnMut(f64) -> f64;

/// One G7/K15 panel: `(integral, error estimate)`, or `None` if the integrand was not finite.
fn gk15(f: Integrand, a: f64, b: f64) -> Option<(f64, f64)> {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    if !fc.is_finite() {
        return None;
    }
    let mut resg = fc * WG[3];
    let mut resk = fc * WGK[7];
    let mut resabs = resk.abs();
    let mut vals = [(0.0, 0.0); 7];
    for j in 0..7 {
        let x = h * XGK[j];
        let (f1, f2) = (f(c - x), f(c + x));
        if !f1.is_finite() || !f2.is_finite() {
            return None;
        }
        vals[j] = (f1, f2);
        resk += WGK[j] * (f1 + f2);
        resabs += WGK[j] * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            resg += WG[j / 2] * (f1 + f2);
        }
    }
    let reskh = 0.5 * resk;
    let mut resasc = WGK[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((vals[j].0 - reskh).abs() + (vals[j].1 - reskh).abs());
    }
    let result = resk * h;
    let (resabs, resasc) = (resabs * h.abs(), resasc * h.abs());
    let mut err = ((resk - resg) * h).abs();
    if resasc != 0.0 && err != 0.0 {
        err = resasc * (200.0 * err / resasc).powf(1.5).min(1.0);
    }
    if resabs > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
        err = err.max(50.0 * f64::EPSILON * resabs);
    }
    Some((result, err))
}

#[derive(Clone, Copy)]
struct Seg {
    a: f64,
    b: f64,
    i: f64,
    e: f64,
}

/// Adaptive G7/K15 on a finite interval. `Err(true)` means a non-finite integrand value,
/// `Err(false)` means it did not converge within the panel budget.
fn adaptive(f: Integrand, a: f64, b: f64) -> Result<f64, bool> {
    const INIT: usize = 4;
    let mut segs: Vec<Seg> = Vec::with_capacity(MAX_SEGMENTS + 2);
    for k in 0..INIT {
        let lo = a + (b - a) * k as f64 / INIT as f64;
        let hi = if k + 1 == INIT { b } else { a + (b - a) * (k + 1) as f64 / INIT as f64 };
        let (i, e) = gk15(f, lo, hi).ok_or(true)?;
        segs.push(Seg { a: lo, b: hi, i, e });
    }
    loop {
        let total: f64 = segs.iter().map(|s| s.i).sum();
        let err: f64 = segs.iter().map(|s| s.e).sum();
        if err <= ABS_TOL.max(REL_TOL * total.abs()) {
            return Ok(total);
        }
        if segs.len() >= MAX_SEGMENTS {
            return Err(false);
        }
        let (worst, _) = segs
            .iter()
            .enumerate()
            .fold((0, -1.0), |best, (k, s)| if s.e > best.1 { (k, s.e) } else { best });
        let s = segs[worst];
        let mid = 0.5 * (s.a + s.b);
        if mid <= s.a || mid >= s.b {
            return Err(false);
        }
        // A non-finite value on a tiny panel is an endpoint singularity being sampled right at
        // the end, not a hole in the integrand: hand over to the fallback instead.
        let tiny = (s.b - s.a).abs() < 1e-9 * (b - a).abs();
        let (i1, e1) = gk15(f, s.a, mid).ok_or(!tiny)?;
        let (i2, e2) = gk15(f, mid, s.b).ok_or(!tiny)?;
        segs[worst] = Seg { a: s.a, b: mid, i: i1, e: e1 };
        segs.push(Seg { a: mid, b: s.b, i: i2, e: e2 });
    }
}

/// Tanh-sinh (double exponential) quadrature: robust against integrable endpoint singularities.
/// Nodes closer to an end than one ulp are dropped. Returns `None` unless successive
/// refinements agree *and* the outermost nodes contribute nothing, which is what tells a
/// convergent singular integral from a divergent one (a divergent integrand keeps contributing
/// at every depth).
fn tanh_sinh(f: Integrand, a: f64, b: f64) -> Option<f64> {
    use std::f64::consts::FRAC_PI_2;
    const TMAX: f64 = 6.0;
    let width = b - a;
    // Weighted value at abscissa `t`, plus whether it lies in the outer tail.
    let node = |t: f64, f: Integrand| -> Option<(f64, bool)> {
        let u = FRAC_PI_2 * t.sinh();
        let w = FRAC_PI_2 * t.cosh() / u.cosh().powi(2);
        let d = 2.0 / ((2.0 * u.abs()).exp() + 1.0);
        let off = width * d * 0.5;
        let x = if t < 0.0 {
            a + off
        } else if t > 0.0 {
            b - off
        } else {
            0.5 * (a + b)
        };
        let outside = x <= a || x >= b;
        let v = if outside { 0.0 } else { f(x) };
        let tail = t.abs() >= TMAX - 1.0;
        if !v.is_finite() {
            return if outside || w < 1e-14 { Some((0.0, tail)) } else { None };
        }
        Some((w * v, tail))
    };
    let mut sum = 0.0;
    let mut tail = 0.0;
    let mut prev = f64::NAN;
    let mut h = 1.0;
    for level in 0..=9 {
        // Level 0 takes every integer abscissa; later levels add only the odd multiples of h.
        let steps = (TMAX / h) as i64;
        let mut j = -steps;
        if level > 0 && j % 2 == 0 {
            j += 1;
        }
        while j <= steps {
            let (v, in_tail) = node(j as f64 * h, f)?;
            sum += v;
            if in_tail {
                tail += v.abs();
            }
            j += if level == 0 { 1 } else { 2 };
        }
        let cur = sum * h * width * 0.5;
        let tail_part = tail * h * width * 0.5;
        if level >= 3 && prev.is_finite() {
            let tol = 1e-10 * cur.abs() + 1e-13;
            if (cur - prev).abs() <= tol && tail_part <= 1e-9 * cur.abs() + 1e-12 {
                return Some(cur);
            }
        }
        prev = cur;
        h *= 0.5;
    }
    None
}

fn finite_integral(f: Integrand, a: f64, b: f64) -> f64 {
    match adaptive(f, a, b) {
        Ok(v) => v,
        Err(true) => f64::NAN,
        Err(false) => tanh_sinh(f, a, b).unwrap_or(f64::NAN),
    }
}

/// Definite integral of `f` over `[a, b]` (either bound may be infinite). Returns `NaN` when the
/// integral diverges, is undefined (the integrand is `NaN` or infinite inside the interval) or
/// does not converge. Integrable endpoint singularities such as `1/sqrt(x)` or `ln(x)` are fine.
pub fn integrate(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return 0.0;
    }
    if a > b {
        return -integrate(f, b, a);
    }
    match (a.is_infinite(), b.is_infinite()) {
        (false, false) => finite_integral(f, a, b),
        (false, true) => {
            let mut g = |t: f64| {
                let s = 1.0 - t;
                f(a + t / s) / (s * s)
            };
            finite_integral(&mut g, 0.0, 1.0)
        }
        (true, false) => {
            let mut g = |t: f64| {
                let s = 1.0 - t;
                f(b - t / s) / (s * s)
            };
            finite_integral(&mut g, 0.0, 1.0)
        }
        (true, true) => {
            let mut g = |t: f64| {
                let s = 1.0 - t * t;
                f(t / s) * (1.0 + t * t) / (s * s)
            };
            finite_integral(&mut g, -1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile;
    use crate::parse::parse;

    fn p(s: &str) -> Expr {
        parse(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"))
    }

    fn d(s: &str, v: &str) -> Expr {
        try_deriv(&p(s), v).unwrap_or_else(|e| panic!("deriv {s:?}: {e}"))
    }

    fn at(e: &Expr, x: f64) -> f64 {
        compile(e, &["x"], Angle::Rad).unwrap().eval(&[x])
    }

    /// The symbolic derivative matches a central difference at several points.
    fn check(src: &str, points: &[f64]) {
        let f = p(src);
        let df = d(src, "x");
        let prog = compile(&f, &["x"], Angle::Rad).unwrap();
        let dprog = compile(&df, &["x"], Angle::Rad).unwrap();
        for &x in points {
            let h = 1e-6;
            let num = (prog.eval(&[x + h]) - prog.eval(&[x - h])) / (2.0 * h);
            let sym = dprog.eval(&[x]);
            assert!(
                (num - sym).abs() <= 1e-5 * (1.0 + num.abs()),
                "{src} at {x}: numeric {num} vs symbolic {sym} ({df})"
            );
        }
    }

    #[test]
    fn simple_forms_are_tidy() {
        assert_eq!(d("x^2", "x").to_string(), "2 * x");
        assert_eq!(d("3x^2+2x+1", "x").to_string(), "6 * x + 2");
        assert_eq!(d("5", "x"), Expr::Num(0.0));
        assert_eq!(d("y", "x"), Expr::Num(0.0));
        assert_eq!(d("x", "x"), Expr::Num(1.0));
        assert_eq!(d("sin(2x)", "x").to_string(), "2 * cos(2 * x)");
        assert_eq!(d("e^x", "x").to_string(), "e^x");
        assert_eq!(d("ln(x)", "x").to_string(), "1 / x");
        assert_eq!(d("1/x", "x").to_string(), "-1 / x^2");
        assert_eq!(d("a*x", "x").to_string(), "a");
    }

    #[test]
    fn power_chain_product_quotient() {
        check("x^5-3x^3+x", &[-2.0, 0.5, 3.0]);
        check("sin(x)^2", &[0.3, 1.2, -2.0]);
        check("(x^2+1)^3", &[-1.0, 0.7, 2.0]);
        check("x*sin(x)", &[0.5, 2.0]);
        check("x*sin(x)*cos(x)*exp(x)", &[0.5, 1.5]);
        check("(x+1)/(x^2+1)", &[-3.0, 0.2, 4.0]);
        check("sqrt(x^2+1)", &[-2.0, 0.0, 3.0]);
        check("x^0.5", &[0.4, 2.0]);
        check("1/(1+x^2)", &[-1.0, 2.0]);
        check("-x^3", &[1.5]);
    }

    #[test]
    fn variable_exponents() {
        check("x^x", &[0.5, 1.5, 3.0]);
        check("2^x", &[-1.0, 0.5, 3.0]);
        check("e^(x^2)", &[0.3, -1.1]);
        check("x^sin(x)", &[0.7, 2.5]);
        check("(x^2+1)^x", &[0.5, 1.0]);
        check("3^(2x+1)", &[0.1, 1.0]);
        // constant to the x with `e` gives no ln factor
        assert_eq!(d("e^(2x)", "x").to_string(), "2 * e^(2 * x)");
    }

    #[test]
    fn every_elementary_function() {
        let xs = [0.3, 0.7];
        for f in [
            "sin(x)", "cos(x)", "tan(x)", "sec(x)", "csc(x)", "cot(x)", "asin(x)", "acos(x)",
            "atan(x)", "sinh(x)", "cosh(x)", "tanh(x)", "exp(x)", "ln(x)", "log(x)", "sqrt(x)",
            "cbrt(x)", "abs(x)", "atan(x^2)", "sin(cos(x))", "ln(sin(x)+2)", "tanh(2x)",
            "atan2(x, 2)", "atan2(x^2, x+1)", "max(x, x^2)", "min(x, x^2)", "mod(x^2, 5)",
            "cbrt(x^2+1)", "log(x+1)", "acos(x^2)", "sec(2x)^2", "cosh(x)*sinh(x)",
        ] {
            check(f, &xs);
        }
        check("|x|", &[-2.0, 3.0]);
        // step-like functions have zero slope away from the jumps
        for f in ["floor(x)", "ceil(x)", "round(x)", "sign(x)"] {
            check(f, &[0.3, 2.7]);
        }
        check("normalpdf(x, 1, 2)", &[0.3, 1.5]);
        check("normalpdf(2x, 1, 2)", &[0.3, 1.5]);
    }

    #[test]
    fn partial_derivatives_and_symbolic_parameters() {
        let e = d("x^2*y+sin(y)", "y");
        assert_eq!(e.to_string(), "x^2 + cos(y)");
        let e = d("a*x^2+b*x+c", "a");
        assert_eq!(e.to_string(), "x^2");
        // derivative w.r.t. a variable that is absent is zero
        assert_eq!(d("sin(x)", "q"), Expr::Num(0.0));
    }

    #[test]
    fn higher_order() {
        let e = nth_deriv(&p("x^4"), "x", 2).unwrap();
        assert!((at(&e, 2.0) - 48.0).abs() < 1e-12);
        let e = nth_deriv(&p("sin(x)"), "x", 4).unwrap();
        assert!((at(&e, 0.7) - 0.7f64.sin()).abs() < 1e-12);
        assert_eq!(nth_deriv(&p("x*1"), "x", 0).unwrap(), Expr::var("x"));
    }

    #[test]
    fn degree_mode_derivatives() {
        let k = std::f64::consts::PI / 180.0;
        for (src, x) in [("sin(x)", 30.0), ("cos(x)", 40.0), ("tan(x)", 20.0), ("atan(x)", 0.5), ("asin(x)", 0.4)] {
            let f = p(src);
            let df = try_deriv_angle(&f, "x", Angle::Deg).unwrap();
            let prog = compile(&f, &["x"], Angle::Deg).unwrap();
            let dprog = compile(&df, &["x"], Angle::Deg).unwrap();
            let h = 1e-6;
            let num = (prog.eval(&[x + h]) - prog.eval(&[x - h])) / (2.0 * h);
            let sym = dprog.eval(&[x]);
            assert!((num - sym).abs() < 1e-6 * (1.0 + num.abs()), "{src}: {num} vs {sym} (k={k})");
        }
    }

    #[test]
    fn unsupported_is_a_typed_error_or_nan() {
        assert_eq!(try_deriv(&p("mean(x)"), "x"), Err(CalcError::Unsupported("mean".into())));
        assert!(matches!(try_deriv(&p("x<2"), "x"), Err(CalcError::Unsupported(_))));
        assert!(matches!(try_deriv(&p("normalpdf(1, x, 2)"), "x"), Err(CalcError::Unsupported(_))));
        let nan = deriv(&p("mean(x)"), "x");
        assert!(matches!(nan, Expr::Num(v) if v.is_nan()));
        // a function that does not involve the variable is constant, whatever it is
        assert_eq!(d("mean(y)", "x"), Expr::Num(0.0));
    }

    #[test]
    fn integral_and_sum_derivatives() {
        // fundamental theorem: d/dx int(sin(t), t, 0, x) = sin(x)
        let e = Expr::call("int", vec![p("sin(t)"), Expr::var("t"), Expr::num(0.0), Expr::var("x")]);
        let de = try_deriv(&e, "x").unwrap();
        assert!((at(&de, 1.1) - 1.1f64.sin()).abs() < 1e-12);
        // chain rule on the upper bound and a parameter inside
        let e = Expr::call("int", vec![p("t*x"), Expr::var("t"), Expr::num(0.0), p("x^2")]);
        check_expr(&e, &[0.5, 1.3]);
        // sum of terms
        let e = Expr::call("sum", vec![p("x^n"), Expr::var("n"), Expr::num(1.0), Expr::num(4.0)]);
        check_expr(&e, &[0.5, 1.3]);
        let bad = Expr::call("sum", vec![p("x"), Expr::var("n"), Expr::num(1.0), p("x")]);
        assert!(try_deriv(&bad, "x").is_err());
    }

    fn check_expr(e: &Expr, points: &[f64]) {
        let de = try_deriv(e, "x").unwrap();
        let prog = compile(e, &["x"], Angle::Rad).unwrap();
        let dprog = compile(&de, &["x"], Angle::Rad).unwrap();
        for &x in points {
            let h = 1e-5;
            let num = (prog.eval(&[x + h]) - prog.eval(&[x - h])) / (2.0 * h);
            let sym = dprog.eval(&[x]);
            assert!((num - sym).abs() < 1e-5 * (1.0 + num.abs()), "{e} at {x}: {num} vs {sym}");
        }
    }

    #[test]
    fn expand_rewrites_deriv_calls() {
        let e = expand(&p("deriv(x^3, x) + 1"), Angle::Rad).unwrap();
        assert_eq!(e.to_string(), "3 * x^2 + 1");
        let e = expand(&p("deriv(x^3, x, 2)"), Angle::Rad).unwrap();
        assert_eq!(e, Expr::Num(12.0));
        // nested
        let e = expand(&p("deriv(deriv(x^3, x), x)"), Angle::Rad).unwrap();
        assert_eq!(e.to_string(), "6 * x");
        // inside other calls
        let e = expand(&p("sin(deriv(x^2, x))"), Angle::Rad).unwrap();
        assert_eq!(e.to_string(), "sin(2 * x)");
        assert!(matches!(expand(&p("deriv(x, 2)"), Angle::Rad), Err(CalcError::BadArgs(_))));
        assert!(matches!(expand(&p("deriv(x)"), Angle::Rad), Err(CalcError::BadArgs(_))));
        // a deriv call met while differentiating is expanded first
        let e = try_deriv(&p("deriv(x^3, x)"), "x").unwrap();
        assert_eq!(e.to_string(), "6 * x");
    }

    #[test]
    fn simplifier_identities() {
        assert_eq!(simplify(&p("0*x+1*y")), Expr::var("y"));
        assert_eq!(simplify(&p("x^1+0")), Expr::var("x"));
        assert_eq!(simplify(&p("2*3+4")), Expr::Num(10.0));
        assert_eq!(simplify(&p("x-x")), Expr::Num(0.0));
        assert_eq!(simplify(&p("x/x")), Expr::Num(1.0));
        assert_eq!(simplify(&p("x*2")).to_string(), "2 * x");
        assert_eq!(simplify(&p("2*(3*x)")).to_string(), "6 * x");
        assert_eq!(simplify(&p("-(-x)")), Expr::var("x"));
        assert_eq!(simplify(&p("x+(-y)")).to_string(), "x - y");
        // does not fold a non-finite result
        assert!(matches!(simplify(&p("1/0")), Expr::Bin(BinOp::Div, ..)));
        assert!(matches!(simplify(&p("(-8)^0.5")), Expr::Bin(BinOp::Pow, ..)));
    }

    // ---------------------------------------------------------------- integration

    fn int(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        integrate(&mut |x| f(x), a, b)
    }

    #[test]
    fn gauss_kronrod_is_exact_on_polynomials() {
        let v = int(|x| x.powi(14), 0.0, 1.0);
        assert!((v - 1.0 / 15.0).abs() < 1e-14, "{v}");
        let v = int(|x| 3.0 * x * x + 2.0 * x + 1.0, -1.0, 2.0);
        assert!((v - 15.0).abs() < 1e-12, "{v}");
    }

    #[test]
    fn smooth_integrals() {
        use std::f64::consts::PI;
        assert!((int(f64::sin, 0.0, PI) - 2.0).abs() < 1e-11);
        assert!(int(f64::sin, 0.0, 2.0 * PI).abs() < 1e-11);
        assert!((int(f64::exp, 0.0, 1.0) - (std::f64::consts::E - 1.0)).abs() < 1e-11);
        assert!((int(|x| 1.0 / (1.0 + x * x), -1.0, 1.0) - PI / 2.0).abs() < 1e-10);
        // reversed bounds flip the sign; equal bounds give zero
        assert!((int(f64::sin, PI, 0.0) + 2.0).abs() < 1e-11);
        assert_eq!(int(f64::sin, 1.0, 1.0), 0.0);
        // oscillatory
        let v = int(|x| (20.0 * x).sin(), 0.0, PI);
        assert!(v.abs() < 1e-9, "{v}");
        // kinks and jumps converge
        assert!((int(f64::abs, -1.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((int(|x| x.floor(), 0.0, 3.0) - 3.0).abs() < 1e-8);
    }

    #[test]
    fn endpoint_singularities_converge() {
        let v = int(|x| 1.0 / x.sqrt(), 0.0, 1.0);
        assert!((v - 2.0).abs() < 1e-8, "{v}");
        let v = int(f64::ln, 0.0, 1.0);
        assert!((v + 1.0).abs() < 1e-8, "{v}");
        let v = int(|x| 1.0 / (1.0 - x * x).sqrt(), -1.0, 1.0);
        assert!((v - std::f64::consts::PI).abs() < 1e-7, "{v}");
        let v = int(|x| x.powf(-0.9), 0.0, 1.0);
        assert!((v - 10.0).abs() < 1e-5, "{v}");
        let v = int(|x| (1.0 / x).ln().sqrt(), 0.0, 1.0);
        assert!((v - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-8, "{v}");
    }

    #[test]
    fn divergence_and_undefined_give_nan() {
        assert!(int(|x| 1.0 / x, 0.0, 1.0).is_nan());
        assert!(int(|x| 1.0 / (x * x), 0.0, 1.0).is_nan());
        assert!(int(|x| 1.0 / x, -1.0, 1.0).is_nan());
        assert!(int(|x| 1.0 / (x * x), -1.0, 1.0).is_nan());
        assert!(int(f64::sqrt, -1.0, 1.0).is_nan());
        assert!(int(|x| x, f64::NAN, 1.0).is_nan());
        assert!(int(|x| x, 0.0, f64::NAN).is_nan());
        assert!(int(|_| 1.0, 0.0, f64::INFINITY).is_nan());
    }

    #[test]
    fn infinite_bounds() {
        let v = int(|x| (-x * x).exp(), f64::NEG_INFINITY, f64::INFINITY);
        assert!((v - std::f64::consts::PI.sqrt()).abs() < 1e-8, "{v}");
        let v = int(|x| (-x).exp(), 0.0, f64::INFINITY);
        assert!((v - 1.0).abs() < 1e-8, "{v}");
        let v = int(|x| 1.0 / (1.0 + x * x), f64::NEG_INFINITY, 0.0);
        assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-8, "{v}");
        // 1/x never converges at infinity
        assert!(int(|x| 1.0 / x, 1.0, f64::INFINITY).is_nan());
    }

    #[test]
    fn sum_and_prod_ranges() {
        assert_eq!(sum_range(1.0, 100.0, |k| k), 5050.0);
        assert_eq!(prod_range(1.0, 5.0, |k| k), 120.0);
        // empty ranges give the identity
        assert_eq!(sum_range(5.0, 1.0, |k| k), 0.0);
        assert_eq!(prod_range(5.0, 1.0, |k| k), 1.0);
        // float fuzz on integer bounds is snapped
        assert_eq!(sum_range(0.1 + 0.2 + 0.7, 3.0, |k| k), 6.0);
        // non-integer bounds round inward
        assert_eq!(sum_range(0.5, 3.5, |k| k), 6.0);
        // iteration cap, non-finite bounds
        assert!(sum_range(1.0, 2e6, |k| k).is_nan());
        assert_eq!(sum_range(1.0, 1e6, |_| 1.0), 1e6);
        assert!(sum_range(f64::NEG_INFINITY, 1.0, |k| k).is_nan());
        assert!(prod_range(1.0, f64::NAN, |k| k).is_nan());
        assert!(sum_range(-1e300, 1e300, |k| k).is_nan());
    }
}

/// End-to-end behaviour across the parser, resolver, compiler and list evaluator.
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analyze::{analyze, Kind};
    use crate::compile::compile;
    use crate::interval::Interval;
    use crate::list::{eval_value, Bindings, ListError, Value};
    use crate::parse::{parse, parse_with, ParseCtx};
    use crate::print::{to_latex, to_text};
    use crate::resolve::{Defs, ResolveError};
    use crate::wgsl::emit_function;
    use std::collections::BTreeSet;

    fn p(s: &str) -> Expr {
        parse(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"))
    }

    fn eval1(src: &str, x: f64) -> f64 {
        compile(&p(src), &["x"], Angle::Rad).unwrap_or_else(|e| panic!("{src}: {e}")).eval(&[x])
    }

    fn num(src: &str) -> f64 {
        compile(&p(src), &[], Angle::Rad).unwrap_or_else(|e| panic!("{src}: {e}")).eval(&[])
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-8 * (1.0 + b.abs())
    }

    fn resolved(defs: &Defs, src: &str, ctx: &ParseCtx) -> Expr {
        defs.resolve(&parse_with(src, ctx).unwrap()).unwrap()
    }

    #[test]
    fn leibniz_syntax() {
        let want = Expr::call("deriv", vec![p("x^2"), Expr::var("x")]);
        assert_eq!(p("d/dx x^2"), want);
        assert_eq!(p("d/dx(x^2)"), want);
        assert_eq!(p("\\frac{d}{dx}\\left(x^2\\right)"), want);
        assert_eq!(p("\\frac{d}{dx} x^2"), want);
        assert_eq!(p("deriv(x^2, x)"), want);
        // the operand is a product term, then ordinary arithmetic continues
        assert_eq!(p("d/dx 3x^2 + 1"), Expr::bin(BinOp::Add, Expr::call("deriv", vec![p("3x^2"), Expr::var("x")]), Expr::num(1.0)));
        assert_eq!(p("d/dx sin(x) cos(x)"), Expr::call("deriv", vec![p("sin(x) cos(x)"), Expr::var("x")]));
        // other variables, negative operand, scaling by a coefficient
        assert_eq!(p("d/dt -t^2"), Expr::call("deriv", vec![p("-t^2"), Expr::var("t")]));
        assert_eq!(p("2 d/dx x^2"), Expr::bin(BinOp::Mul, Expr::num(2.0), want.clone()));
        // second order
        let second = Expr::call("deriv", vec![want.clone(), Expr::var("x")]);
        assert_eq!(p("d^2/dx^2 x^2"), second);
        assert_eq!(p("\\frac{d^{2}}{dx^{2}} x^2"), second);
        assert_eq!(p("\\frac{d^2}{dx^2}\\left(x^2\\right)"), second);
        // not a derivative: stays an ordinary quotient
        assert_eq!(p("d/dx"), Expr::bin(BinOp::Mul, Expr::bin(BinOp::Div, Expr::var("d"), Expr::var("d")), Expr::var("x")));
        assert_eq!(p("\\frac{dy}{dx}"), p("((dy)/(dx))"));
        assert_eq!(p("d/dx+1"), p("d/d*x+1"));
        // mismatched orders are ordinary algebra, not an error
        assert!(parse("d^2/dx x").is_ok());
        assert!(parse("d^2/dx^3 x").is_ok());
    }

    #[test]
    fn derivative_printing_round_trips() {
        let e = p("d/dx x^2");
        assert_eq!(to_text(&e), "deriv(x^2, x)");
        assert_eq!(p(&to_text(&e)), e);
        assert_eq!(to_latex(&e), "\\frac{d}{dx}\\left(x^{2}\\right)");
        assert_eq!(p(&to_latex(&e)), e);
        let at = p("deriv(x^2, x, 3)");
        assert_eq!(p(&to_text(&at)), at);
        // int/sum/prod print as calls and re-parse
        for s in ["int(sin(t), t, 0, x)", "sum(k^2, k, 1, 10)", "prod(k, k, 1, 5)"] {
            let e = p(s);
            assert_eq!(p(&to_text(&e)), e, "{s}");
            assert_eq!(p(&to_latex(&e)), e, "{s}");
        }
    }

    #[test]
    fn prime_notation() {
        let ctx = ParseCtx::new().with_function("f");
        let e = parse_with("f'(2)", &ctx).unwrap();
        assert_eq!(e, Expr::call("deriv", vec![Expr::call("f", vec![Expr::var("x")]), Expr::var("x"), Expr::num(2.0)]));
        let mut defs = Defs::new();
        defs.define_func("f", vec!["x".into()], p("x^3"));
        let r = |s: &str| compile(&defs.resolve(&parse_with(s, &ctx).unwrap()).unwrap(), &["x"], Angle::Rad).unwrap();
        assert!(close(r("f'(2)").eval(&[0.0]), 12.0));
        assert!(close(r("f''(2)").eval(&[0.0]), 12.0));
        assert!(close(r("f'''(2)").eval(&[0.0]), 6.0));
        // the argument is an expression, evaluated after differentiating (no chain rule)
        assert!(close(r("f'(2x)").eval(&[1.5]), 27.0));
        assert!(close(r("f'(x)+f(x)").eval(&[2.0]), 12.0 + 8.0));
        // builtins too
        assert!(close(compile(&parse("sin'(x)").unwrap(), &["x"], Angle::Rad).unwrap().eval(&[0.0]), 1.0));
        // a prime without a call is an error, and so is a prime on a non-function
        assert!(parse_with("f'", &ctx).is_err());
        assert!(parse_with("f'(1,2)", &ctx).is_err());
        assert!(parse("q'(1)").is_err());
        assert_eq!(to_text(&e), "deriv(f(x), x, 2)");
    }

    #[test]
    fn calculus_names_are_builtin_functions() {
        for n in ["deriv", "int", "sum", "prod"] {
            assert!(crate::ast::is_builtin_func(n));
        }
        // single letters and ordinary products are unaffected
        assert_eq!(p("sx"), p("s*x"));
        assert_eq!(p("pu"), p("p*u"));
        assert_eq!(p("2i"), p("2*i"));
        assert_eq!(p("n*t"), p("n t"));
    }

    #[test]
    fn resolver_expands_derivatives() {
        let ctx = ParseCtx::new().with_function("f").with_function("g");
        let mut defs = Defs::new();
        defs.define_func("f", vec!["x".into()], p("x^3+a x"));
        defs.define_var("a", Expr::num(2.0));
        let r = resolved(&defs, "d/dx f(x)", &ctx);
        assert_eq!(to_text(&r), "3 * x^2 + 2");
        assert!(r.called_functions().is_empty());
        // `deriv(f, x)` with the bare function name
        let r = resolved(&defs, "deriv(f, x)", &ctx);
        assert_eq!(to_text(&r), "3 * x^2 + 2");
        let r = resolved(&defs, "deriv(f, t, 2)", &ctx);
        assert_eq!(r, Expr::num(14.0));
        // a function defined with a derivative inlines correctly: h(x) = d/dx f(x)
        defs.define_func("h", vec!["x".into()], Expr::call("deriv", vec![p("x^3"), Expr::var("x")]));
        let ctx = ctx.with_function("h");
        let r = resolved(&defs, "h(2)", &ctx);
        assert_eq!(compile(&r, &[], Angle::Rad).unwrap().eval(&[]), 12.0);
        let r = resolved(&defs, "h(y+1)", &ctx);
        assert_eq!(compile(&r, &["y"], Angle::Rad).unwrap().eval(&[1.0]), 12.0);
        // differentiate with respect to a parameter that is defined: the definition is shadowed
        let r = resolved(&defs, "deriv(a x^2, a)", &ctx);
        assert_eq!(to_text(&r), "x^2");
        // nested derivative and a derivative of a derivative
        let r = resolved(&defs, "d^2/dx^2 f(x)", &ctx);
        assert_eq!(to_text(&r), "6 * x");
        // errors are typed
        assert!(matches!(defs.resolve(&p("deriv(x, 2)")), Err(ResolveError::Calculus(_))));
        assert!(matches!(defs.resolve(&p("deriv(mean(x), x)")), Err(ResolveError::Calculus(_))));
        assert!(matches!(defs.resolve(&p("deriv(x)")), Err(ResolveError::Calculus(_))));
        assert!(ResolveError::Calculus("m".into()).to_string() == "m");
    }

    #[test]
    fn resolver_respects_degree_mode() {
        let mut defs = Defs::new();
        defs.set_angle(Angle::Deg);
        let r = defs.resolve(&p("deriv(sin(x), x)")).unwrap();
        let v = compile(&r, &["x"], Angle::Deg).unwrap().eval(&[0.0]);
        assert!(close(v, std::f64::consts::PI / 180.0));
        let mut rad = Defs::new();
        rad.set_angle(Angle::Rad);
        let r = rad.resolve(&p("deriv(sin(x), x)")).unwrap();
        assert!(close(compile(&r, &["x"], Angle::Rad).unwrap().eval(&[0.0]), 1.0));
    }

    #[test]
    fn resolver_keeps_bound_variables() {
        let mut defs = Defs::new();
        defs.define_var("n", Expr::num(3.0));
        defs.define_var("k", Expr::num(2.0));
        // n is bound by the sum, so the definition n=3 is not applied inside it
        let r = defs.resolve(&p("sum(k*n, n, 1, n)")).unwrap();
        assert_eq!(num(&to_text(&r)), 2.0 * 6.0);
        assert_eq!(r.free_vars().len(), 0);
        // a function whose body sums over its own parameter name
        defs.define_func("g", vec!["m".into()], p("sum(n, n, 1, m)"));
        let ctx = ParseCtx::new().with_function("g");
        // the argument mentions n: the bound n must not capture it
        let r = resolved(&defs, "g(n+1)", &ctx);
        assert_eq!(to_text(&r), "sum(n, n, 1, 3 + 1)");
        assert_eq!(num(&to_text(&r)), 10.0);
    }

    #[test]
    fn free_variables_exclude_bound_ones() {
        let e = p("sum(k*n, n, 1, m)");
        assert_eq!(e.free_vars().into_iter().collect::<Vec<_>>(), vec!["k", "m"]);
        assert!(!e.contains_var("n"));
        assert!(e.contains_var("m") && e.contains_var("k"));
        let e = p("int(sin(t)*x, t, 0, x)");
        assert_eq!(e.free_vars().into_iter().collect::<Vec<_>>(), vec!["x"]);
        // the bound name is free in the bounds
        let e = p("sum(i, i, 1, i)");
        assert!(e.contains_var("i"));
        assert_eq!(p("sum(x, x, 1, 3)").free_vars().len(), 0);
        // classification: x is bound, so this is a plain value, not a field
        let a = analyze(&p("sum(x, x, 1, 3)"), &BTreeSet::new());
        assert!(matches!(a.kind, Kind::Value { .. }));
        let a = analyze(&p("y=int(sin(t), t, 0, x)"), &BTreeSet::new());
        assert!(matches!(a.kind, Kind::ExplicitY { .. }));
        assert!(a.slider_candidates.is_empty(), "{:?}", a.slider_candidates);
        assert_eq!(a.dims, [true, true, false]);
        let a = analyze(&p("sum(k*n, n, 1, 5)"), &BTreeSet::new());
        assert_eq!(a.slider_candidates.iter().collect::<Vec<_>>(), vec!["k"]);
    }

    #[test]
    fn substitution_avoids_capture() {
        let e = p("sum(n*m, n, 1, 3)");
        // m -> n+1 must not be captured by the bound n
        let s = e.subst("m", &p("n+1"));
        let at = |n: f64| compile(&s, &["n"], Angle::Rad).unwrap().eval(&[n]);
        assert_eq!(at(1.0), 2.0 + 4.0 + 6.0);
        assert_eq!(at(0.0), 1.0 + 2.0 + 3.0);
        // substituting the bound name itself only touches the bounds
        let e = p("sum(n, n, 1, n)");
        let s = e.subst("n", &Expr::num(4.0));
        assert_eq!(num(&to_text(&s)), 10.0);
    }

    #[test]
    fn compiles_sums_and_products() {
        assert_eq!(num("sum(k, k, 1, 100)"), 5050.0);
        assert_eq!(num("prod(k, k, 1, 6)"), 720.0);
        assert!(close(num("sum(1/k^2, k, 1, 100000)"), std::f64::consts::PI.powi(2) / 6.0 - 1e-5));
        assert_eq!(num("sum(k, k, 5, 1)"), 0.0);
        assert_eq!(num("prod(k, k, 5, 1)"), 1.0);
        // outer variables, expression bounds, nesting
        assert_eq!(eval1("sum(k*x, k, 1, 4)", 2.0), 20.0);
        assert_eq!(eval1("sum(k, k, 1, x)", 10.0), 55.0);
        assert_eq!(num("sum(sum(i*j, j, 1, 3), i, 1, 3)"), 36.0);
        assert_eq!(num("sum(prod(j, j, 1, i), i, 1, 4)"), 1.0 + 2.0 + 6.0 + 24.0);
        // a bound variable shadows an outer variable of the same name
        assert_eq!(eval1("sum(x, x, 1, 3)+x", 10.0), 16.0);
        // Taylor series for e
        assert!(close(num("sum(1/prod(j, j, 1, k), k, 1, 15)+1"), std::f64::consts::E));
        // the iteration cap: more than a million terms is NaN
        assert!(num("sum(1, k, 1, 2000000)").is_nan());
        assert_eq!(num("sum(1, k, 1, 1000000)"), 1e6);
        assert!(num("sum(k, k, 1, 1/0)").is_nan());
        // arity and bound-variable checks are typed errors
        assert!(matches!(compile(&p("sum(k, k, 1)"), &[], Angle::Rad), Err(crate::compile::CompileError::Arity { .. })));
        assert!(matches!(compile(&p("sum(k, 2, 1, 3)"), &[], Angle::Rad), Err(crate::compile::CompileError::Calculus(_))));
        assert!(matches!(compile(&p("sum(k, k, 1, 3)+q"), &[], Angle::Rad), Err(crate::compile::CompileError::UnboundVariable(_))));
        assert!(matches!(compile(&p("sum(q, k, 1, 3)"), &[], Angle::Rad), Err(crate::compile::CompileError::UnboundVariable(_))));
    }

    #[test]
    fn compiles_integrals() {
        let pi = std::f64::consts::PI;
        assert!(close(num("int(sin(t), t, 0, pi)"), 2.0));
        assert!(close(num("int(x^2, x, 0, 3)"), 9.0));
        // an integral as a function of x plots as a curve: y = int(sin(t), t, 0, x) = 1 - cos(x)
        for x in [0.0, 0.5, 1.0, 2.5, 6.0, -1.0] {
            assert!(close(eval1("int(sin(t), t, 0, x)", x), 1.0 - x.cos()), "x={x}");
        }
        // parameter inside the integrand, nested integrals, integrand of the outer variable
        assert!(close(eval1("int(x*t, t, 0, 2)", 3.0), 6.0));
        assert!(close(eval1("int(int(s, s, 0, t), t, 0, x)", 3.0), 4.5));
        // d/dx of an integral is the integrand (resolver expands, the integral differentiates)
        let defs = Defs::new();
        let e = defs.resolve(&p("d/dx int(sin(t), t, 0, x)")).unwrap();
        assert!(close(compile(&e, &["x"], Angle::Rad).unwrap().eval(&[1.2]), 1.2f64.sin()));
        // integral of a derivative recovers the function (FTC the other way)
        assert!(close(num("int(deriv(t^3, t), t, 0, 2)"), 8.0));
        // singular and divergent integrals
        assert!(close(num("int(1/sqrt(t), t, 0, 1)"), 2.0));
        assert!(num("int(1/t, t, 0, 1)").is_nan());
        assert!(num("int(sqrt(t), t, -1, 1)").is_nan());
        // improper integral with an infinite bound
        assert!(close(num("int(e^(-t^2), t, -1e999, 1e999)"), pi.sqrt()));
        // degree mode reaches the integrand
        let deg = compile(&p("int(cos(t), t, 0, 90)"), &[], Angle::Deg).unwrap().eval(&[]);
        assert!(close(deg, 180.0 / pi));
    }

    #[test]
    fn interval_enclosures_are_sound() {
        let cases = [
            ("sum(k*x, k, 1, 4)", -2.0, 3.0),
            ("sum(x^k, k, 0, 5)", -1.0, 2.0),
            ("prod(x+k, k, 1, 3)", 0.0, 1.0),
            ("int(sin(t), t, 0, x)", 0.0, 3.0),
            ("int(t^2, t, 0, x)", -2.0, 2.0),
            ("sum(k, k, 1, x)", 1.0, 5.0),
            ("int(1/t, t, 0, x)", 1.0, 2.0),
        ];
        for (src, lo, hi) in cases {
            let prog = compile(&p(src), &["x"], Angle::Rad).unwrap();
            let enc = prog.eval_interval(&[Interval::new(lo, hi)]);
            assert!(enc.lo.is_finite() && enc.hi.is_finite(), "{src}: {enc:?}");
            for i in 0..=20 {
                let x = lo + (hi - lo) * i as f64 / 20.0;
                let v = prog.eval(&[x]);
                if v.is_finite() {
                    assert!(enc.lo <= v + 1e-9 && v <= enc.hi + 1e-9, "{src} at {x}: {v} not in {enc:?}");
                }
            }
        }
    }

    #[test]
    fn gpu_path_rejects_loops_but_takes_expanded_derivatives() {
        // deriv expands to plain operations, so WGSL emission works
        let d = compile(&p("d/dx x^3"), &["x"], Angle::Rad).unwrap();
        assert!(d.subs.is_empty());
        let wgsl = emit_function("dfn", &d).unwrap();
        assert!(wgsl.contains("fn dfn"));
        // sums, products and integrals are a typed error, never code
        for s in ["sum(k, k, 1, 3)", "prod(k, k, 1, 3)", "int(t, t, 0, 1)", "x+int(t, t, 0, 1)"] {
            let prog = compile(&p(s), &["x"], Angle::Rad).unwrap();
            let err = emit_function("f", &prog).unwrap_err();
            assert!(err.to_string().contains("GPU"), "{s}: {err}");
        }
    }

    fn list(src: &str, b: &Bindings) -> Result<Value, ListError> {
        eval_value(&p(src), b)
    }

    #[test]
    fn list_evaluator_sums_and_integrals() {
        let b = Bindings::new().with("L", Value::List(vec![5.0, 1.0, 4.0, 2.0, 3.0]));
        assert_eq!(list("sum(k, k, 1, 10)", &b).unwrap(), Value::Num(55.0));
        // sum inside a list comprehension, using the comprehension variable
        assert_eq!(
            list("[sum(k, k, 1, m) for m=[1...5]]", &b).unwrap(),
            Value::List(vec![1.0, 3.0, 6.0, 10.0, 15.0])
        );
        assert_eq!(
            list("[int(t, t, 0, m) for m=[1...3]]", &b).unwrap(),
            Value::List(vec![0.5, 2.0, 4.5])
        );
        assert_eq!(list("[prod(j, j, 1, m) for m=[1...5]]", &b).unwrap(), Value::List(vec![1.0, 2.0, 6.0, 24.0, 120.0]));
        // a body that indexes a list takes the term-by-term path
        assert_eq!(list("sum(L[i], i, 1, length(L))", &b).unwrap(), Value::Num(15.0));
        assert_eq!(list("sum(L[i]^2, i, 1, 3)", &b).unwrap(), Value::Num(25.0 + 1.0 + 16.0));
        assert_eq!(list("prod(L[i], i, 1, 2)", &b).unwrap(), Value::Num(5.0));
        let v = list("int(L[1]*t, t, 0, 2)", &b).unwrap();
        assert!(close(v.as_num().unwrap(), 10.0));
        // a list-valued body is a type error, not a panic
        assert!(matches!(list("sum(L, k, 1, 3)", &b), Err(ListError::Type(_))));
        // unbound names are reported
        assert!(matches!(list("sum(q, k, 1, 3)", &b), Err(ListError::UnboundVariable(_))));
        // bound variable shadows a binding of the same name
        let shadow = b.clone().with("k", Value::Num(100.0));
        assert_eq!(list("sum(k, k, 1, 3)+k", &shadow).unwrap(), Value::Num(106.0));
        // bounds must be numbers; arity is checked
        assert!(matches!(list("sum(k, k, L, 3)", &b), Err(ListError::Type(_))));
        assert!(matches!(list("sum(k, k, 1)", &b), Err(ListError::Arity { .. })));
        assert!(matches!(list("sum(k, 2, 1, 3)", &b), Err(ListError::Type(_))));
    }

    #[test]
    fn list_evaluator_budget_and_derivs() {
        let b = Bindings::new();
        // the work budget stops a loop that would otherwise run away
        assert_eq!(list("sum(1, k, 1, 1000000)", &b).unwrap(), Value::Num(1e6));
        assert!(matches!(
            list("[sum(1, k, 1, 1000000) for m=[1...10]]", &b),
            Err(ListError::TooMuchWork)
        ));
        assert!(list("sum(1, k, 1, 2000000)", &b).unwrap().as_num().unwrap().is_nan());
        // unresolved derivative calls evaluate too
        assert!(close(list("deriv(x^3, x, 2)", &b).unwrap().as_num().unwrap(), 12.0));
        assert!(matches!(list("deriv(x, 2)", &b), Err(ListError::Type(_))));
        // element-wise use through a comprehension
        assert_eq!(
            list("[deriv(x^2, x, m) for m=[1...3]]", &b).unwrap(),
            Value::List(vec![2.0, 4.0, 6.0])
        );
    }
}
