//! Compiles a scalar [`Expr`] into postfix bytecode with f64 and interval evaluators.
//!
//! The compiler only ever emits opcodes from a closed set and constants from the AST, so no
//! user text can reach an evaluator or a generated shader: unknown names are errors, not code.

use crate::ast::{constant_value, BinOp, Expr, Rel};
use crate::calculus;
use crate::interval::Interval;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Angle {
    #[default]
    Rad,
    Deg,
}

impl Angle {
    /// Multiplier that converts an angle in this unit to radians.
    pub fn to_rad(self) -> f64 {
        match self {
            Angle::Rad => 1.0,
            Angle::Deg => PI / 180.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F1 {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    /// The error function.
    Erf,
    Exp,
    Ln,
    Log10,
    Sqrt,
    Cbrt,
    Abs,
    Floor,
    Ceil,
    Round,
    Sign,
    /// `x!` = Gamma(x + 1).
    Factorial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F2 {
    Atan2,
    Min,
    Max,
    Mod,
    /// `nCr(n, k)`.
    Choose,
    /// `nPr(n, k)`.
    Perm,
    /// `gcd(a, b)`: undefined unless both are integers.
    Gcd,
    /// `lcm(a, b)`: undefined unless both are integers.
    Lcm,
}

/// Scalar-argument distribution functions (fixed arity, arguments on the stack in order).
/// Evaluated in f64 by [`crate::stats`]; not available on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatFn {
    NormalPdf,
    NormalCdf,
    InvNorm,
    BinomPdf,
    BinomCdf,
    PoissonPdf,
    PoissonCdf,
    UniformPdf,
    UniformCdf,
    TPdf,
    TCdf,
    InvT,
}

impl StatFn {
    pub fn from_name(name: &str) -> Option<StatFn> {
        Some(match name {
            "normalpdf" => StatFn::NormalPdf,
            "normalcdf" => StatFn::NormalCdf,
            "invnorm" => StatFn::InvNorm,
            "binompdf" => StatFn::BinomPdf,
            "binomcdf" => StatFn::BinomCdf,
            "poissonpdf" => StatFn::PoissonPdf,
            "poissoncdf" => StatFn::PoissonCdf,
            "uniformpdf" => StatFn::UniformPdf,
            "uniformcdf" => StatFn::UniformCdf,
            "tpdf" => StatFn::TPdf,
            "tcdf" => StatFn::TCdf,
            "invt" => StatFn::InvT,
            _ => return None,
        })
    }

    pub fn arity(self) -> usize {
        match self {
            StatFn::NormalCdf | StatFn::UniformCdf => 4,
            StatFn::NormalPdf
            | StatFn::InvNorm
            | StatFn::BinomPdf
            | StatFn::BinomCdf
            | StatFn::UniformPdf
            | StatFn::TCdf => 3,
            StatFn::PoissonPdf | StatFn::PoissonCdf | StatFn::TPdf | StatFn::InvT => 2,
        }
    }

    /// `a` holds exactly `arity()` values.
    pub fn apply(self, a: &[f64]) -> f64 {
        use crate::stats as s;
        match self {
            StatFn::NormalPdf => s::normalpdf(a[0], a[1], a[2]),
            StatFn::NormalCdf => s::normalcdf(a[0], a[1], a[2], a[3]),
            StatFn::InvNorm => s::invnorm(a[0], a[1], a[2]),
            StatFn::BinomPdf => s::binompdf(a[0], a[1], a[2]),
            StatFn::BinomCdf => s::binomcdf(a[0], a[1], a[2]),
            StatFn::PoissonPdf => s::poissonpdf(a[0], a[1]),
            StatFn::PoissonCdf => s::poissoncdf(a[0], a[1]),
            StatFn::UniformPdf => s::uniformpdf(a[0], a[1], a[2]),
            StatFn::UniformCdf => s::uniformcdf(a[0], a[1], a[2], a[3]),
            StatFn::TPdf => s::tpdf(a[0], a[1]),
            StatFn::TCdf => s::tcdf(a[0], a[1], a[2]),
            StatFn::InvT => s::invt(a[0], a[1]),
        }
    }

    /// Finite, sound, deliberately loose enclosure. Never infinite or ENTIRE (the mesh treats
    /// infinite enclosures as poles). `a` holds exactly `arity()` intervals.
    pub fn enclosure(self, a: &[Interval]) -> Interval {
        const BIG: f64 = 1e300;
        if a.iter().any(|i| i.is_empty()) {
            return Interval::EMPTY;
        }
        match self {
            StatFn::NormalCdf
            | StatFn::BinomCdf
            | StatFn::PoissonCdf
            | StatFn::UniformCdf
            | StatFn::TCdf => Interval::new(0.0, 1.0),
            StatFn::BinomPdf | StatFn::PoissonPdf => Interval::new(0.0, 1.0),
            StatFn::TPdf => Interval::new(0.0, 0.4),
            StatFn::NormalPdf => {
                // peak 1/(sigma sqrt(2 pi)) when sigma is provably positive
                let sig = a[2].lo;
                let hi = if sig > 0.0 { (1.0 / (sig * 2.5066282746310002) * 1.000001).min(BIG) } else { BIG };
                Interval::new(0.0, hi)
            }
            StatFn::UniformPdf => {
                let w = a[2].lo - a[1].hi;
                let hi = if w > 0.0 { (1.0 / w * 1.000001).min(BIG) } else { BIG };
                Interval::new(0.0, hi)
            }
            StatFn::InvNorm | StatFn::InvT => Interval::new(-BIG, BIG),
        }
    }
}

/// The iterated / integrated calls: `int(f, t, a, b)`, `sum(f, n, a, b)`, `prod(f, n, a, b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    Int,
    Sum,
    Prod,
}

impl ReduceKind {
    pub fn from_name(name: &str) -> Option<ReduceKind> {
        Some(match name {
            "int" => ReduceKind::Int,
            "sum" => ReduceKind::Sum,
            "prod" => ReduceKind::Prod,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Const(f64),
    Load(usize),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    F1(F1),
    F2(F2),
    Stat(StatFn),
    /// Pops `hi` then `lo` and reduces `Program::subs[index]` over them. The sub-program's
    /// variables are this program's followed by the bound variable (the last slot).
    Reduce(ReduceKind, usize),
    /// Pops `b` then `a`, pushes 1 when `a rel b` holds and 0 otherwise (also when either is NaN).
    Cmp(Rel),
    /// Pops two truth values, pushes 1 when both are true. A truth value is non-zero and not NaN.
    And,
    /// Piecewise: the stack holds `pairs` condition/value pairs (condition first, deepest first)
    /// and, when `default`, one more value on top. Pushes the value of the first true condition,
    /// else the default, else NaN. Every branch is evaluated (no side effects), then selected.
    Piece { pairs: usize, default: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    UnboundVariable(String),
    UnknownFunction(String),
    Arity { name: String, expected: usize, got: usize },
    NotScalar(&'static str),
    /// A `deriv` that cannot be expanded, or a malformed `int`/`sum`/`prod`/`deriv` call.
    Calculus(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnboundVariable(n) => write!(f, "undefined variable '{n}'"),
            CompileError::UnknownFunction(n) => write!(f, "unknown function '{n}'"),
            CompileError::Arity { name, expected, got } => {
                write!(f, "{name} takes {expected} argument(s), got {got}")
            }
            CompileError::NotScalar(what) => write!(f, "{what} is not a scalar expression"),
            CompileError::Calculus(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub ops: Vec<Op>,
    pub vars: Vec<String>,
    pub angle: Angle,
    /// Bodies of `int`/`sum`/`prod` calls, referenced by [`Op::Reduce`].
    pub subs: Vec<Program>,
}

fn f1_of(name: &str) -> Option<F1> {
    Some(match name {
        "sin" => F1::Sin,
        "cos" => F1::Cos,
        "tan" => F1::Tan,
        "asin" | "arcsin" => F1::Asin,
        "acos" | "arccos" => F1::Acos,
        "atan" | "arctan" => F1::Atan,
        "sinh" => F1::Sinh,
        "cosh" => F1::Cosh,
        "tanh" => F1::Tanh,
        "asinh" | "arcsinh" => F1::Asinh,
        "acosh" | "arccosh" => F1::Acosh,
        "atanh" | "arctanh" => F1::Atanh,
        "erf" => F1::Erf,
        "exp" => F1::Exp,
        "ln" => F1::Ln,
        "log" => F1::Log10,
        "sqrt" => F1::Sqrt,
        "cbrt" => F1::Cbrt,
        "abs" => F1::Abs,
        "floor" => F1::Floor,
        "ceil" => F1::Ceil,
        "round" => F1::Round,
        "sign" | "sgn" => F1::Sign,
        "factorial" => F1::Factorial,
        _ => return None,
    })
}

fn f2_of(name: &str) -> Option<F2> {
    Some(match name {
        "atan2" => F2::Atan2,
        "min" => F2::Min,
        "max" => F2::Max,
        "mod" => F2::Mod,
        "nCr" => F2::Choose,
        "nPr" => F2::Perm,
        "gcd" | "gcf" | "mcd" => F2::Gcd,
        "lcm" | "mcm" => F2::Lcm,
        _ => return None,
    })
}

fn arity(name: &str, expected: usize, args: &[Expr]) -> Result<(), CompileError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(CompileError::Arity { name: name.to_string(), expected, got: args.len() })
    }
}

struct Cx {
    out: Vec<Op>,
    subs: Vec<Program>,
    angle: Angle,
}

fn emit(e: &Expr, vars: &[String], cx: &mut Cx) -> Result<(), CompileError> {
    match e {
        Expr::Num(v) => cx.out.push(Op::Const(*v)),
        Expr::Var(n) => {
            if let Some(i) = vars.iter().rposition(|v| v == n) {
                cx.out.push(Op::Load(i));
            } else if let Some(c) = constant_value(n) {
                cx.out.push(Op::Const(c));
            } else {
                return Err(CompileError::UnboundVariable(n.clone()));
            }
        }
        Expr::Neg(a) => {
            emit(a, vars, cx)?;
            cx.out.push(Op::Neg);
        }
        Expr::Bin(op, a, b) => {
            emit(a, vars, cx)?;
            emit(b, vars, cx)?;
            cx.out.push(match op {
                BinOp::Add => Op::Add,
                BinOp::Sub => Op::Sub,
                BinOp::Mul => Op::Mul,
                BinOp::Div => Op::Div,
                BinOp::Pow => Op::Pow,
            });
        }
        Expr::Call(name, args) if name == crate::ast::PIECE_FN && !args.is_empty() => {
            let pairs = args.len() / 2;
            for c in args.chunks_exact(2) {
                emit_cond(&c[0], vars, cx)?;
                emit(&c[1], vars, cx)?;
            }
            let default = args.len() % 2 == 1;
            if default {
                emit(&args[args.len() - 1], vars, cx)?;
            }
            cx.out.push(Op::Piece { pairs, default });
        }
        Expr::Call(name, args) => {
            let n = name.as_str();
            if n == "deriv" {
                let args: Vec<Expr> = args
                    .iter()
                    .map(|a| calculus::expand(a, cx.angle))
                    .collect::<Result<_, _>>()
                    .map_err(|e| CompileError::Calculus(e.to_string()))?;
                let d = calculus::expand_deriv(&args, cx.angle)
                    .map_err(|e| CompileError::Calculus(e.to_string()))?;
                emit(&d, vars, cx)?;
            } else if let Some(kind) = ReduceKind::from_name(n) {
                arity(n, 4, args)?;
                let Expr::Var(bound) = &args[1] else {
                    return Err(CompileError::Calculus(format!("{n} needs a variable name as its second argument")));
                };
                emit(&args[2], vars, cx)?;
                emit(&args[3], vars, cx)?;
                // The bound variable takes the last slot, so it shadows an outer variable of
                // the same name (`Load` resolves names from the end).
                let mut sub_vars = vars.to_vec();
                sub_vars.push(bound.clone());
                let mut inner = Cx { out: Vec::new(), subs: Vec::new(), angle: cx.angle };
                emit(&args[0], &sub_vars, &mut inner)?;
                cx.subs.push(Program { ops: inner.out, vars: sub_vars, angle: cx.angle, subs: inner.subs });
                cx.out.push(Op::Reduce(kind, cx.subs.len() - 1));
            } else if let Some(f) = f1_of(n) {
                arity(n, 1, args)?;
                emit(&args[0], vars, cx)?;
                cx.out.push(Op::F1(f));
            } else if let Some(f) = f2_of(n) {
                arity(n, 2, args)?;
                emit(&args[0], vars, cx)?;
                emit(&args[1], vars, cx)?;
                cx.out.push(Op::F2(f));
            } else if let Some(f) = StatFn::from_name(n) {
                arity(n, f.arity(), args)?;
                for a in args {
                    emit(a, vars, cx)?;
                }
                cx.out.push(Op::Stat(f));
            } else {
                match n {
                    "sec" | "csc" => {
                        arity(n, 1, args)?;
                        cx.out.push(Op::Const(1.0));
                        emit(&args[0], vars, cx)?;
                        cx.out.push(Op::F1(if n == "sec" { F1::Cos } else { F1::Sin }));
                        cx.out.push(Op::Div);
                    }
                    "cot" => {
                        arity(n, 1, args)?;
                        emit(&args[0], vars, cx)?;
                        cx.out.push(Op::F1(F1::Cos));
                        emit(&args[0], vars, cx)?;
                        cx.out.push(Op::F1(F1::Sin));
                        cx.out.push(Op::Div);
                    }
                    _ => return Err(CompileError::UnknownFunction(name.clone())),
                }
            }
        }
        Expr::Tuple(_) => return Err(CompileError::NotScalar("a point")),
        Expr::List(_) => return Err(CompileError::NotScalar("a list")),
        Expr::Rel(Rel::Ne, ..) => {
            return Err(CompileError::NotScalar("a '!=' comparison (use it as a condition inside a piecewise {..})"))
        }
        Expr::Rel(..) => return Err(CompileError::NotScalar("a relation")),
    }
    Ok(())
}

/// A condition of a piecewise branch: a comparison, a chained comparison (`and`), or any scalar
/// (non-zero is true).
fn emit_cond(e: &Expr, vars: &[String], cx: &mut Cx) -> Result<(), CompileError> {
    match e {
        Expr::Rel(r, a, b) => {
            emit(a, vars, cx)?;
            emit(b, vars, cx)?;
            cx.out.push(Op::Cmp(*r));
        }
        _ if crate::ast::rel_chain(e).is_some() => {
            for (i, part) in crate::ast::rel_chain(e).unwrap().iter().enumerate() {
                emit_cond(part, vars, cx)?;
                if i > 0 {
                    cx.out.push(Op::And);
                }
            }
        }
        _ => emit(e, vars, cx)?,
    }
    Ok(())
}

/// Compiles `expr` with `vars` as the input slots (in order).
pub fn compile(expr: &Expr, vars: &[&str], angle: Angle) -> Result<Program, CompileError> {
    let vars: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
    let mut cx = Cx { out: Vec::new(), subs: Vec::new(), angle };
    emit(expr, &vars, &mut cx)?;
    Ok(Program { ops: cx.out, vars, angle, subs: cx.subs })
}

fn sign0(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

fn truth(c: f64) -> bool {
    c != 0.0 && !c.is_nan()
}

fn cmp_f64(r: Rel, a: f64, b: f64) -> f64 {
    use Rel;
    let t = match r {
        Rel::Eq => a == b,
        Rel::Lt => a < b,
        Rel::Le => a <= b,
        Rel::Gt => a > b,
        Rel::Ge => a >= b,
        Rel::Ne => a != b,
    };
    // an undefined operand makes every comparison false, `!=` included
    (t && !a.is_nan() && !b.is_nan()) as u8 as f64
}

/// Interval of a comparison: exactly 0 or 1 when the boxes decide it, else [0, 1]. An empty
/// (undefined) operand makes the comparison false, as in the f64 evaluator.
fn cmp_i(r: Rel, a: Interval, b: Interval) -> Interval {
    use Rel;
    if a.is_empty() || b.is_empty() {
        return Interval::point(0.0);
    }
    let (yes, no) = match r {
        Rel::Lt => (a.hi < b.lo, a.lo >= b.hi),
        Rel::Le => (a.hi <= b.lo, a.lo > b.hi),
        Rel::Gt => (a.lo > b.hi, a.hi <= b.lo),
        Rel::Ge => (a.lo >= b.hi, a.hi < b.lo),
        Rel::Eq => (a.lo == a.hi && b.lo == b.hi && a.lo == b.lo, a.hi < b.lo || a.lo > b.hi),
        Rel::Ne => (a.hi < b.lo || a.lo > b.hi, a.lo == a.hi && b.lo == b.hi && a.lo == b.lo),
    };
    match (yes, no) {
        (true, _) => Interval::point(1.0),
        (_, true) => Interval::point(0.0),
        _ => Interval::new(0.0, 1.0),
    }
}

fn hull(a: Interval, b: Interval) -> Interval {
    if a.is_empty() {
        b
    } else if b.is_empty() {
        a
    } else {
        Interval::new(a.lo.min(b.lo), a.hi.max(b.hi))
    }
}

/// Enclosure of a piecewise value: the hull of every branch that may be taken, up to and
/// including the first one that surely is. `st` holds the pairs (and the default) in order.
fn piece_i(st: &[Interval], pairs: usize, default: bool) -> Interval {
    let mut acc = Interval::EMPTY;
    for k in 0..pairs {
        let (c, v) = (st[2 * k], st[2 * k + 1]);
        if c.is_empty() || (c.lo == 0.0 && c.hi == 0.0) {
            continue; // surely false
        }
        acc = hull(acc, v);
        if c.lo > 0.0 {
            return acc; // surely true: later branches are never reached
        }
    }
    if default {
        acc = hull(acc, st[2 * pairs]);
    }
    acc
}

fn mod_f64(a: f64, b: f64) -> f64 {
    a - b * (a / b).floor()
}

fn apply1(f: F1, x: f64, angle: Angle) -> f64 {
    let k = angle.to_rad();
    match f {
        F1::Sin => (x * k).sin(),
        F1::Cos => (x * k).cos(),
        F1::Tan => (x * k).tan(),
        F1::Asin => x.asin() / k,
        F1::Acos => x.acos() / k,
        F1::Atan => x.atan() / k,
        F1::Sinh => x.sinh(),
        F1::Cosh => x.cosh(),
        F1::Tanh => x.tanh(),
        F1::Asinh => x.asinh(),
        F1::Acosh => x.acosh(),
        F1::Atanh => x.atanh(),
        F1::Erf => crate::stats::erf(x),
        F1::Exp => x.exp(),
        F1::Ln => x.ln(),
        F1::Log10 => x.log10(),
        F1::Sqrt => x.sqrt(),
        F1::Cbrt => x.cbrt(),
        F1::Abs => x.abs(),
        F1::Floor => x.floor(),
        F1::Ceil => x.ceil(),
        F1::Round => x.round(),
        F1::Sign => sign0(x),
        F1::Factorial => crate::stats::factorial(x),
    }
}

fn apply2(f: F2, a: f64, b: f64) -> f64 {
    match f {
        F2::Atan2 => a.atan2(b),
        F2::Min => a.min(b),
        F2::Max => a.max(b),
        F2::Mod => mod_f64(a, b),
        F2::Choose => crate::stats::choose(a, b),
        F2::Perm => crate::stats::permute(a, b),
        F2::Gcd => crate::stats::gcd(a, b),
        F2::Lcm => crate::stats::lcm(a, b),
    }
}

fn apply1_i(f: F1, x: Interval, angle: Angle) -> Interval {
    let k = Interval::point(angle.to_rad());
    let scaled = |x: Interval| if angle == Angle::Rad { x } else { x.mul(k) };
    let unscale = |r: Interval| if angle == Angle::Rad { r } else { r.div(k) };
    match f {
        F1::Sin => scaled(x).sin(),
        F1::Cos => scaled(x).cos(),
        F1::Tan => scaled(x).tan(),
        F1::Asin => unscale(x.asin()),
        F1::Acos => unscale(x.acos()),
        F1::Atan => unscale(x.atan()),
        F1::Sinh => x.sinh(),
        F1::Cosh => x.cosh(),
        F1::Tanh => x.tanh(),
        F1::Asinh => x.asinh(),
        F1::Acosh => x.acosh(),
        F1::Atanh => x.atanh(),
        F1::Erf => x.erf(),
        F1::Exp => x.exp(),
        F1::Ln => x.ln(),
        F1::Log10 => x.log10(),
        F1::Sqrt => x.sqrt(),
        F1::Cbrt => x.cbrt(),
        F1::Abs => x.abs(),
        F1::Floor => x.floor(),
        F1::Ceil => x.ceil(),
        F1::Round => x.round(),
        F1::Sign => x.sign(),
        F1::Factorial => x.factorial(),
    }
}

fn apply2_i(f: F2, a: Interval, b: Interval) -> Interval {
    match f {
        F2::Atan2 => a.atan2(b),
        F2::Min => a.min(b),
        F2::Max => a.max(b),
        F2::Mod => a.modulo(b),
        F2::Choose | F2::Perm => a.comb_enclosure(b, f == F2::Perm),
        F2::Gcd | F2::Lcm => a.number_theory(b, f == F2::Lcm),
    }
}

impl Program {
    pub fn eval(&self, vars: &[f64]) -> f64 {
        let mut stack = Vec::with_capacity(16);
        self.eval_with(vars, &mut stack)
    }

    /// Evaluates reusing `stack` to avoid allocating in hot loops.
    pub fn eval_with(&self, vars: &[f64], st: &mut Vec<f64>) -> f64 {
        st.clear();
        for op in &self.ops {
            match *op {
                Op::Const(c) => st.push(c),
                Op::Load(i) => st.push(vars[i]),
                Op::Neg => {
                    let a = st.pop().unwrap();
                    st.push(-a);
                }
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(match op {
                        Op::Add => a + b,
                        Op::Sub => a - b,
                        Op::Mul => a * b,
                        Op::Div => a / b,
                        _ if b == 2.0 => a * a,
                        _ => a.powf(b),
                    });
                }
                Op::F1(f) => {
                    let a = st.pop().unwrap();
                    st.push(apply1(f, a, self.angle));
                }
                Op::F2(f) => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(apply2(f, a, b));
                }
                Op::Stat(f) => {
                    let n = f.arity();
                    let at = st.len() - n;
                    let r = f.apply(&st[at..]);
                    st.truncate(at);
                    st.push(r);
                }
                Op::Reduce(kind, si) => {
                    let hi = st.pop().unwrap();
                    let lo = st.pop().unwrap();
                    st.push(self.reduce(kind, &self.subs[si], vars, lo, hi));
                }
                Op::Cmp(r) => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(cmp_f64(r, a, b));
                }
                Op::And => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push((truth(a) && truth(b)) as u8 as f64);
                }
                Op::Piece { pairs, default } => {
                    let at = st.len() - 2 * pairs - default as usize;
                    let mut v = if default { st[at + 2 * pairs] } else { f64::NAN };
                    for k in 0..pairs {
                        if truth(st[at + 2 * k]) {
                            v = st[at + 2 * k + 1];
                            break;
                        }
                    }
                    st.truncate(at);
                    st.push(v);
                }
            }
        }
        st.pop().unwrap_or(f64::NAN)
    }

    /// `int`/`sum`/`prod` of `sub` (whose last variable is the bound one) over `[lo, hi]`.
    fn reduce(&self, kind: ReduceKind, sub: &Program, vars: &[f64], lo: f64, hi: f64) -> f64 {
        let mut sv: Vec<f64> = vars.iter().copied().take(self.vars.len()).collect();
        sv.push(0.0);
        let slot = sv.len() - 1;
        let mut stack = Vec::with_capacity(16);
        let mut f = |t: f64| {
            sv[slot] = t;
            sub.eval_with(&sv, &mut stack)
        };
        match kind {
            ReduceKind::Int => calculus::integrate(&mut f, lo, hi),
            ReduceKind::Sum => calculus::sum_range(lo, hi, f),
            ReduceKind::Prod => calculus::prod_range(lo, hi, f),
        }
    }

    pub fn eval_interval(&self, vars: &[Interval]) -> Interval {
        let mut stack = Vec::with_capacity(16);
        self.eval_interval_with(vars, &mut stack)
    }

    pub fn eval_interval_with(&self, vars: &[Interval], st: &mut Vec<Interval>) -> Interval {
        st.clear();
        for op in &self.ops {
            match *op {
                Op::Const(c) => st.push(Interval::point(c)),
                Op::Load(i) => st.push(vars[i]),
                Op::Neg => {
                    let a = st.pop().unwrap();
                    st.push(a.neg());
                }
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(match op {
                        Op::Add => a.add(b),
                        Op::Sub => a.sub(b),
                        Op::Mul => a.mul(b),
                        Op::Div => a.div(b),
                        _ => a.pow(b),
                    });
                }
                Op::F1(f) => {
                    let a = st.pop().unwrap();
                    st.push(apply1_i(f, a, self.angle));
                }
                Op::F2(f) => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(apply2_i(f, a, b));
                }
                Op::Stat(f) => {
                    let n = f.arity();
                    let at = st.len() - n;
                    let r = f.enclosure(&st[at..]);
                    st.truncate(at);
                    st.push(r);
                }
                Op::Reduce(kind, si) => {
                    let hi = st.pop().unwrap();
                    let lo = st.pop().unwrap();
                    st.push(reduce_enclosure(kind, &self.subs[si], &vars[..self.vars.len()], lo, hi));
                }
                Op::Cmp(r) => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(cmp_i(r, a, b));
                }
                Op::And => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    let (fa, fb) = (a.is_empty() || (a.lo == 0.0 && a.hi == 0.0), b.is_empty() || (b.lo == 0.0 && b.hi == 0.0));
                    st.push(if fa || fb {
                        Interval::point(0.0)
                    } else if a.lo > 0.0 && b.lo > 0.0 {
                        Interval::point(1.0)
                    } else {
                        Interval::new(0.0, 1.0)
                    });
                }
                Op::Piece { pairs, default } => {
                    let at = st.len() - 2 * pairs - default as usize;
                    let r = piece_i(&st[at..], pairs, default);
                    st.truncate(at);
                    st.push(r);
                }
            }
        }
        st.pop().unwrap_or(Interval::EMPTY)
    }
}

/// Sound enclosure of an `int`/`sum`/`prod`. Always finite (the mesh reads infinite enclosures as
/// poles); loose unless the bounds are exact and the work is small.
fn reduce_enclosure(kind: ReduceKind, sub: &Program, vars: &[Interval], lo: Interval, hi: Interval) -> Interval {
    const BIG: f64 = 1e300;
    const MAX_TERMS: i64 = 256;
    let loose = Interval::new(-BIG, BIG);
    if lo.is_empty() || hi.is_empty() {
        return Interval::EMPTY;
    }
    let mut sv: Vec<Interval> = vars.to_vec();
    sv.push(Interval::ENTIRE);
    let slot = sv.len() - 1;
    let mut stack = Vec::with_capacity(16);
    let finite = |i: Interval| i.lo.is_finite() && i.hi.is_finite();
    match kind {
        ReduceKind::Int => {
            sv[slot] = Interval::new(lo.lo.min(hi.lo), lo.hi.max(hi.hi));
            let body = sub.eval_interval_with(&sv, &mut stack);
            if !finite(body) || !lo.lo.is_finite() || !lo.hi.is_finite() || !hi.lo.is_finite() || !hi.hi.is_finite() {
                return loose;
            }
            let m = body.lo.abs().max(body.hi.abs());
            let w = (hi.hi - lo.lo).abs().max((lo.hi - hi.lo).abs());
            let r = m * w;
            if r.is_finite() {
                Interval::new(-r, r)
            } else {
                loose
            }
        }
        ReduceKind::Sum | ReduceKind::Prod => {
            if lo.lo != lo.hi || hi.lo != hi.hi {
                return loose;
            }
            let Some((l, h)) = calculus::int_bounds(lo.lo, hi.lo) else { return loose };
            if h - l + 1 > MAX_TERMS {
                return loose;
            }
            let prod = kind == ReduceKind::Prod;
            let mut acc = Interval::point(if prod { 1.0 } else { 0.0 });
            for k in l..=h {
                sv[slot] = Interval::point(k as f64);
                let t = sub.eval_interval_with(&sv, &mut stack);
                acc = if prod { acc.mul(t) } else { acc.add(t) };
            }
            if finite(acc) {
                acc
            } else {
                loose
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn prog(src: &str, vars: &[&str], angle: Angle) -> Program {
        compile(&parse(src).unwrap(), vars, angle).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn evaluates_known_values() {
        assert!(close(prog("x^2+1", &["x"], Angle::Rad).eval(&[3.0]), 10.0));
        assert!(close(prog("2pi", &[], Angle::Rad).eval(&[]), std::f64::consts::TAU));
        assert!(close(prog("sqrt(x^2+y^2)", &["x", "y"], Angle::Rad).eval(&[3.0, 4.0]), 5.0));
        assert!(close(prog("|x|-1", &["x"], Angle::Rad).eval(&[-3.0]), 2.0));
        assert!(close(prog("log(1000)", &[], Angle::Rad).eval(&[]), 3.0));
        assert!(close(prog("ln(e)", &[], Angle::Rad).eval(&[]), 1.0));
        assert!(close(prog("max(1,2)+min(3,4)", &[], Angle::Rad).eval(&[]), 5.0));
        assert!(close(prog("sec(0)", &[], Angle::Rad).eval(&[]), 1.0));
        assert!(close(prog("cot(pi/4)", &[], Angle::Rad).eval(&[]), 1.0));
        assert!(close(prog("mod(-1,3)", &[], Angle::Rad).eval(&[]), 2.0));
        assert!(close(prog("sign(-5)+sign(0)", &[], Angle::Rad).eval(&[]), -1.0));
    }

    #[test]
    fn inverse_hyperbolic_reciprocal_erf_gcd_lcm_evaluate() {
        let f = |src: &str, x: f64| prog(src, &["x"], Angle::Rad).eval(&[x]);
        assert!(close(f("asinh(x)", 2.0), 1.4436354751788103));
        assert!(close(f("arcsinh(x)", -2.0), -1.4436354751788103));
        assert!(close(f("acosh(x)", 2.0), 1.3169578969248166));
        assert!(f("arccosh(x)", 0.5).is_nan());
        assert!(close(f("atanh(x)", 0.5), 0.5493061443340549));
        assert!(close(f("arctanh(x)", 0.5), 0.5493061443340549));
        assert!(close(f("sech(x)", 1.0), 1.0 / 1f64.cosh()));
        assert!(close(f("csch(x)", 1.0), 1.0 / 1f64.sinh()));
        assert!(close(f("sech^2 x", 1.0), 1.0 / 1f64.cosh().powi(2)));
        assert!(close(f("arcsech(x)", 0.5), 1.3169578969248166));
        assert!(close(f("arccsch(x)", 2.0), 0.48121182505960347));
        assert!(close(f("arccoth(x)", 2.0), 0.5493061443340549));
        assert!(close(f("arcsec(x)", 2.0), std::f64::consts::FRAC_PI_3));
        assert!(close(f("arccsc(x)", 2.0), std::f64::consts::FRAC_PI_6));
        // arccot has range (0, pi): continuous through 0 and above pi/2 for negative x
        assert!(close(f("arccot(x)", 0.0), std::f64::consts::FRAC_PI_2));
        assert!(close(f("arccot(x)", 1.0), std::f64::consts::FRAC_PI_4));
        assert!(close(f("arccot(x)", -1.0), 3.0 * std::f64::consts::FRAC_PI_4));
        let d = prog("arccot(x)", &["x"], Angle::Deg);
        assert!(close(d.eval(&[1.0]), 45.0) && close(d.eval(&[-1.0]), 135.0));
        assert!(close(f("erf(x)", 0.5), 0.5204998778130465));
        assert!(close(f("erf(x)", -0.5), -0.5204998778130465));
        assert_eq!(f("gcd(x,18)", 12.0), 6.0);
        assert_eq!(f("gcd(x,18)", -12.0), 6.0);
        assert_eq!(f("gcf(x,18)", 12.0), 6.0);
        assert_eq!(f("mcd(x,18)", 12.0), 6.0);
        assert_eq!(f("gcd(x,0)", 7.0), 7.0);
        assert_eq!(f("gcd(0,0)+0*x", 0.0), 0.0);
        assert_eq!(f("lcm(x,18)", 12.0), 36.0);
        assert_eq!(f("mcm(x,18)", 12.0), 36.0);
        assert_eq!(f("lcm(x,0)", 7.0), 0.0);
        assert!(f("gcd(x,18)", 2.5).is_nan() && f("lcm(x,18)", 2.5).is_nan());
        assert_eq!(f("signum(x)", -4.0), -1.0);
        // wrong argument counts keep the function's own name in the error
        let e = parse("gcd(1)").unwrap();
        assert!(matches!(compile(&e, &[], Angle::Rad), Err(CompileError::Arity { expected: 2, got: 1, .. })));
        // enclosures: monotone functions, domain clipping, exact on points only
        let p = prog("acosh(x)", &["x"], Angle::Rad);
        assert!(p.eval_interval(&[Interval::new(0.0, 0.5)]).is_empty());
        let e = p.eval_interval(&[Interval::new(0.0, 2.0)]);
        assert!(e.lo <= 0.0 && e.hi >= 1.3169578969248166, "{e:?}");
        let p = prog("atanh(x)", &["x"], Angle::Rad);
        let e = p.eval_interval(&[Interval::new(-2.0, 0.5)]);
        assert!(e.lo == f64::NEG_INFINITY && e.hi >= 0.5493, "{e:?}");
        let p = prog("gcd(x,18)", &["x"], Angle::Rad);
        let e = p.eval_interval(&[Interval::point(12.0)]);
        assert!(e.lo <= 6.0 && e.hi >= 6.0 && e.hi < 6.001, "{e:?}");
        let e = p.eval_interval(&[Interval::new(1.0, 20.0)]);
        assert!(e.lo == f64::NEG_INFINITY && e.hi == f64::INFINITY, "{e:?}");
    }

    #[test]
    fn factorial_and_combinatorics_evaluate() {
        let f = |src: &str, x: f64| prog(src, &["x"], Angle::Rad).eval(&[x]);
        assert_eq!(f("x!", 0.0), 1.0);
        assert_eq!(f("x!", 5.0), 120.0);
        assert_eq!(f("x!", 20.0), 2432902008176640000.0);
        assert!(close(f("x!", 170.0), 7.257415615307994e306));
        assert_eq!(f("x!", 171.0), f64::INFINITY);
        assert_eq!(f("x!", 1000.0), f64::INFINITY);
        // gamma for non-integers: 0.5! = sqrt(pi)/2, (-0.5)! = sqrt(pi)
        assert!((f("x!", 0.5) - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-13);
        assert!((f("x!", -0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-13);
        assert!((f("x!", 4.5) - 52.34277778455352).abs() < 1e-11);
        assert!(f("x!", -1.0).is_nan() && f("x!", -3.0).is_nan());
        assert!(f("x!", f64::NAN).is_nan());
        assert!(close(f("(x+1)!", 2.0), 6.0));
        assert!(close(f("2^x!", 3.0), 64.0));
        assert!(close(f("x^3!", 2.0), 64.0));
        assert!(close(f("factorial(x)", 4.0), 24.0));
        assert_eq!(f("nCr(x,2)", 5.0), 10.0);
        assert_eq!(f("nCr(x,0)", 5.0), 1.0);
        assert_eq!(f("nCr(x,7)", 5.0), 0.0);
        assert_eq!(f("nCr(52,5)+0*x", 0.0), 2598960.0);
        assert!(f("nCr(x,2)", 2.5).is_nan() && f("nCr(x,2)", -1.0).is_nan());
        assert_eq!(f("nPr(x,2)", 5.0), 20.0);
        assert_eq!(f("nPr(x,5)", 5.0), 120.0);
        assert_eq!(f("nPr(x,6)", 5.0), 0.0);
        let e = parse("nCr(1)").unwrap();
        assert!(matches!(compile(&e, &[], Angle::Rad), Err(CompileError::Arity { expected: 2, got: 1, .. })));
        // enclosures: monotone branch, interior minimum of gamma, poles and negatives
        let p = prog("x!", &["x"], Angle::Rad);
        let e = p.eval_interval(&[Interval::new(0.0, 4.0)]);
        assert!(e.lo <= 0.8857 && e.lo >= 0.8856 - 1e-6 && e.hi >= 24.0 && e.hi < 24.001, "{e:?}");
        let e = p.eval_interval(&[Interval::new(1.0, 3.0)]);
        assert!(e.lo <= 1.0 && e.hi >= 6.0);
        assert_eq!(p.eval_interval(&[Interval::new(-3.0, 2.0)]), Interval::ENTIRE);
        assert_eq!(p.eval_interval(&[Interval::point(3.0)]), Interval::point(6.0));
    }

    #[test]
    fn angle_mode_changes_trig_semantics() {
        assert!(close(prog("sin(90)", &[], Angle::Deg).eval(&[]), 1.0));
        assert!(close(prog("sin(pi/2)", &[], Angle::Rad).eval(&[]), 1.0));
        assert!(close(prog("asin(1)", &[], Angle::Deg).eval(&[]), 90.0));
        assert!(close(prog("atan(1)", &[], Angle::Deg).eval(&[]), 45.0));
        assert!(close(prog("cos(60)", &[], Angle::Deg).eval(&[]), 0.5));
    }

    #[test]
    fn rejects_unknown_names_instead_of_emitting_them() {
        assert_eq!(
            compile(&parse("x+1").unwrap(), &[], Angle::Rad),
            Err(CompileError::UnboundVariable("x".into()))
        );
        assert!(matches!(
            compile(&parse("f(x)=1").unwrap(), &["x"], Angle::Rad),
            Err(CompileError::NotScalar(_))
        ));
        let ctx = crate::parse::ParseCtx::new().with_function("g");
        let e = crate::parse::parse_with("g(1)", &ctx).unwrap();
        assert_eq!(compile(&e, &[], Angle::Rad), Err(CompileError::UnknownFunction("g".into())));
        assert!(matches!(
            compile(&parse("sin(1,2)").unwrap(), &[], Angle::Rad),
            Err(CompileError::Arity { .. })
        ));
    }

    /// Deterministic generator so the soundness test needs no dependencies.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn range(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }
    }

    const CORPUS: &[&str] = &[
        "x^2+y",
        "x^3-2x+y^2",
        "sin(x)*cos(y)",
        "sin(x)+sin(y)^2",
        "1/x",
        "x/(y+3)",
        "sqrt(x)+y",
        "sqrt(x^2+y^2)-2",
        "abs(x)-y",
        "exp(x)/(1+y^2)",
        "tan(x)",
        "ln(x)+y",
        "x^y",
        "x^2.5",
        "atan(x*y)",
        "floor(x)+y",
        "ceil(x*y)",
        "(x+1)(x-1)",
        "sin(1/x)",
        "min(x,y)*max(x,y)",
        "mod(x,3)+y",
        "sinh(x)-cosh(y)",
        "tanh(x*y)",
        "asin(x/5)+acos(y/5)",
        "x^-2",
        "round(x)+sign(y)",
        "cbrt(x)*y",
        "log(x)+y",
        "sec(x)+csc(y)",
        "x*y/(x^2+y^2+1)",
        "sin(x^2+y^2)",
        "2^x",
        "cos(x)^2+sin(x)^2-1",
        "normalpdf(x,0,1)+y",
        "normalcdf(-1,x,y,2)",
        "normalpdf(x,y,1)",
        "invnorm(x/12+0.5,y,2)",
        "tcdf(x,y,5)",
        "tpdf(x,abs(y)+1)",
        "invt(x/12+0.5,3)",
        "binompdf(10,0.3,x)",
        "binomcdf(10,0.3,x)",
        "poissonpdf(3,x)",
        "poissoncdf(abs(y),x)",
        "uniformpdf(x,-2,y)",
        "uniformcdf(-2,3,x,y)",
        "x!+y",
        "(x+y)!",
        "(abs(x)+0.5)!",
        "x!/(y!+1)",
        "factorial(x/2)",
        "nCr(x,y)",
        "nPr(x+6,y+6)",
        "nCr(5,2)+x",
        "arcsinh(x)+arccosh(abs(x)+1)",
        "atanh(x/10)+erf(y)",
        "sech(x)+csch(x+0.5)",
        "arcsec(abs(x)+1)+arccsc(abs(y)+1)+arccot(x)",
        "gcd(x+6,y+9)+lcm(x+6,y+9)",
    ];

    #[test]
    fn stat_functions_compile_and_evaluate() {
        let p = prog("normalpdf(x,0,1)", &["x"], Angle::Rad);
        assert!(close(p.eval(&[0.0]), 0.3989422804014327));
        let p = prog("normalcdf(-1.96,x,0,1)+1", &["x"], Angle::Rad);
        assert!(close(p.eval(&[1.96]), 1.950004209703559));
        let e = parse("normalpdf(x,0)").unwrap();
        assert!(matches!(compile(&e, &["x"], Angle::Rad), Err(CompileError::Arity { expected: 3, got: 2, .. })));
        // enclosures are finite (the mesh treats infinite ones as poles)
        let p = prog("invnorm(x,0,1)", &["x"], Angle::Rad);
        let e = p.eval_interval(&[Interval::new(0.1, 0.9)]);
        assert!(e.lo.is_finite() && e.hi.is_finite());
        let p = prog("normalpdf(x,0,1)", &["x"], Angle::Rad);
        let e = p.eval_interval(&[Interval::new(-1e308, 1e308)]);
        assert!(e.lo == 0.0 && e.hi.is_finite() && e.hi >= 0.3989);
    }

    #[test]
    fn interval_enclosure_is_sound() {
        let mut rng = Lcg(0x5eed);
        for angle in [Angle::Rad, Angle::Deg] {
            for src in CORPUS {
                let p = prog(src, &["x", "y"], angle);
                for _ in 0..300 {
                    let xl = rng.range(-6.0, 6.0);
                    let yl = rng.range(-6.0, 6.0);
                    // Occasionally a degenerate or large box.
                    let xw = if rng.next() < 0.15 { 0.0 } else { rng.range(0.0, 4.0) * rng.next() * 3.0 };
                    let yw = if rng.next() < 0.15 { 0.0 } else { rng.range(0.0, 4.0) * rng.next() * 3.0 };
                    let bx = Interval::new(xl, xl + xw);
                    let by = Interval::new(yl, yl + yw);
                    let enclosure = p.eval_interval(&[bx, by]);
                    for _ in 0..20 {
                        let x = rng.range(bx.lo, bx.hi);
                        let y = rng.range(by.lo, by.hi);
                        let v = p.eval(&[x, y]);
                        if v.is_nan() {
                            continue;
                        }
                        assert!(
                            enclosure.contains(v),
                            "{src} [{angle:?}] at ({x},{y}) = {v} not in {:?} for box {bx:?} x {by:?}",
                            enclosure
                        );
                    }
                }
            }
        }
    }
}
