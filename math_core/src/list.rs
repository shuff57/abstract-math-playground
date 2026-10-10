//! Value evaluator for lists, points and statistics.
//!
//! [`eval_value`] evaluates an [`Expr`] to a [`Value`] (number, list, point, list of points)
//! with elementwise broadcasting. Lists are plain AST (see the encodings below), so the
//! parser, printer, resolver and analyzer need no list-specific variants.
//!
//! Encodings (all `Expr::Call`, produced by the parser):
//! * `range(a, b)` / `range(a, next, b)` for `[a...b]` / `[a, next...b]`
//! * `index(L, i)` for `L[i]`; `i` is a number, a list/range, or a comparison (`Expr::Rel`,
//!   a boolean mask evaluated against the same list)
//! * `for(body, Var(n), source)` for `[body for n=source]`
//!
//! Semantics worth knowing
//! * Indexing is 1-based. A single out-of-range or non-integer index yields `NaN` (a point of
//!   NaNs for lists of points); out-of-range or non-integer entries of a list index are dropped.
//! * Scalar functions broadcast elementwise by compiling `name(a0, a1, ..)` once per distinct
//!   name/arity with [`crate::compile`], so the math is not duplicated here.
//! * `min`/`max` with one argument are list aggregates; with two they are the scalar functions.
//! * A scalar passed where a list is expected counts as a one-element list.
//! * A comparison only evaluates inside an index; anywhere else it is an error.
//! * Hostile input is an error, never a panic: recursion depth, list length (10 000) and a
//!   total work budget are capped.

use crate::ast::{constant_value, BinOp, Expr, Rel};
use crate::calculus;
use crate::compile::{compile, Angle, CompileError, Program, ReduceKind};
use crate::stats::{self, StatsError};
use std::collections::HashMap;
use std::fmt;

/// Longest list a range, comprehension source or comprehension result may have.
pub const MAX_LIST_LEN: usize = 10_000;
/// Deepest expression nesting evaluated.
pub const MAX_DEPTH: usize = 200;
/// Total evaluation work budget (nodes visited plus elements produced).
pub const MAX_STEPS: u64 = 5_000_000;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    List(Vec<f64>),
    Point(Vec<f64>),
    PointList(Vec<Vec<f64>>),
}

impl Value {
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[f64]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Bindings {
    vars: HashMap<String, Value>,
    pub angle: Angle,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_angle(mut self, angle: Angle) -> Self {
        self.angle = angle;
        self
    }

    pub fn set(&mut self, name: &str, v: Value) {
        self.vars.insert(name.to_string(), v);
    }

    pub fn with(mut self, name: &str, v: Value) -> Self {
        self.set(name, v);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ListError {
    UnboundVariable(String),
    UnknownFunction(String),
    Arity { name: String, expected: String, got: usize },
    LengthMismatch { a: usize, b: usize },
    /// Wrong kind of value for an operation (message says what).
    Type(String),
    BadRange(String),
    TooLong { limit: usize },
    TooDeep,
    TooMuchWork,
    /// A comparison outside an index brackets.
    RelOutsideIndex,
    Stats(StatsError),
}

impl fmt::Display for ListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ListError::UnboundVariable(n) => write!(f, "undefined variable '{n}'"),
            ListError::UnknownFunction(n) => write!(f, "unknown function '{n}'"),
            ListError::Arity { name, expected, got } => {
                write!(f, "{name} takes {expected} argument(s), got {got}")
            }
            ListError::LengthMismatch { a, b } => write!(f, "list lengths differ ({a} vs {b})"),
            ListError::Type(m) => write!(f, "{m}"),
            ListError::BadRange(m) => write!(f, "bad range: {m}"),
            ListError::TooLong { limit } => write!(f, "list is longer than {limit} elements"),
            ListError::TooDeep => write!(f, "expression is nested too deeply"),
            ListError::TooMuchWork => write!(f, "expression is too expensive to evaluate"),
            ListError::RelOutsideIndex => write!(f, "a comparison is only allowed inside list index brackets"),
            ListError::Stats(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ListError {}

impl From<StatsError> for ListError {
    fn from(e: StatsError) -> Self {
        ListError::Stats(e)
    }
}

type R<T> = Result<T, ListError>;

pub fn eval_value(expr: &Expr, b: &Bindings) -> Result<Value, ListError> {
    let mut ev = Evaluator {
        base: b,
        locals: Vec::new(),
        steps: 0,
        cache: HashMap::new(),
        rng: RANDOM_SEED,
    };
    ev.ev(expr, 0)
}

/// Where `random()` starts in every evaluation: the numbers are the same each time the same
/// expression is evaluated (a redraw must not reshuffle a graph), and `random(n, seed)` gives a
/// different stream per seed, e.g. a slider.
const RANDOM_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// One step of splitmix64: returns a uniform number in `[0, 1)` and advances `state`.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

struct Evaluator<'a> {
    base: &'a Bindings,
    locals: Vec<(String, Value)>,
    steps: u64,
    cache: HashMap<(String, usize), Program>,
    /// State of the `random()` stream of this evaluation.
    rng: u64,
}

fn apply_bin(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Pow => a.powf(b),
    }
}

fn compare(rel: Rel, a: f64, b: f64) -> bool {
    match rel {
        Rel::Eq => a == b,
        Rel::Lt => a < b,
        Rel::Le => a <= b,
        Rel::Gt => a > b,
        Rel::Ge => a >= b,
        Rel::Ne => a != b,
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "a number",
        Value::List(_) => "a list",
        Value::Point(_) => "a point",
        Value::PointList(_) => "a list of points",
    }
}

fn slice_of(v: &Value) -> R<&[f64]> {
    match v {
        Value::List(l) => Ok(l),
        Value::Num(x) => Ok(std::slice::from_ref(x)),
        other => Err(ListError::Type(format!("expected a list, got {}", kind(other)))),
    }
}

fn check_arity(name: &str, expected: &str, ok: bool, got: usize) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(ListError::Arity { name: name.to_string(), expected: expected.to_string(), got })
    }
}

/// Elementwise binary map over numbers and lists (scalars repeat, lists must match).
fn zip_map(a: &Value, b: &Value, f: impl Fn(f64, f64) -> f64) -> R<Option<Value>> {
    Ok(match (a, b) {
        (Value::Num(x), Value::Num(y)) => Some(Value::Num(f(*x, *y))),
        (Value::Num(x), Value::List(l)) => Some(Value::List(l.iter().map(|y| f(*x, *y)).collect())),
        (Value::List(l), Value::Num(y)) => Some(Value::List(l.iter().map(|x| f(*x, *y)).collect())),
        (Value::List(l), Value::List(m)) => {
            if l.len() != m.len() {
                return Err(ListError::LengthMismatch { a: l.len(), b: m.len() });
            }
            Some(Value::List(l.iter().zip(m).map(|(x, y)| f(*x, *y)).collect()))
        }
        _ => None,
    })
}

fn point_op(op: BinOp, a: &Value, b: &Value) -> R<Value> {
    use Value::*;
    let additive = matches!(op, BinOp::Add | BinOp::Sub);
    let comp = |p: &[f64], q: &[f64]| -> R<Vec<f64>> {
        if p.len() != q.len() {
            return Err(ListError::LengthMismatch { a: p.len(), b: q.len() });
        }
        Ok(p.iter().zip(q).map(|(x, y)| apply_bin(op, *x, *y)).collect())
    };
    let scale = |p: &[f64], s: f64, left: bool| -> Vec<f64> {
        p.iter().map(|x| if left { apply_bin(op, s, *x) } else { apply_bin(op, *x, s) }).collect()
    };
    let mul = op == BinOp::Mul;
    let muldiv = matches!(op, BinOp::Mul | BinOp::Div);
    match (a, b) {
        (Point(p), Point(q)) if additive => Ok(Point(comp(p, q)?)),
        (PointList(ps), Point(q)) if additive => {
            Ok(PointList(ps.iter().map(|p| comp(p, q)).collect::<R<_>>()?))
        }
        (Point(p), PointList(qs)) if additive => {
            Ok(PointList(qs.iter().map(|q| comp(p, q)).collect::<R<_>>()?))
        }
        (PointList(ps), PointList(qs)) if additive => {
            if ps.len() != qs.len() {
                return Err(ListError::LengthMismatch { a: ps.len(), b: qs.len() });
            }
            Ok(PointList(ps.iter().zip(qs).map(|(p, q)| comp(p, q)).collect::<R<_>>()?))
        }
        (Num(s), Point(p)) if mul => Ok(Point(scale(p, *s, true))),
        (Point(p), Num(s)) if muldiv => Ok(Point(scale(p, *s, false))),
        (Num(s), PointList(ps)) if mul => Ok(PointList(ps.iter().map(|p| scale(p, *s, true)).collect())),
        (PointList(ps), Num(s)) if muldiv => Ok(PointList(ps.iter().map(|p| scale(p, *s, false)).collect())),
        _ => Err(ListError::Type(format!(
            "cannot apply this operator to {} and {}",
            kind(a),
            kind(b)
        ))),
    }
}

impl<'a> Evaluator<'a> {
    fn tick(&mut self, n: usize) -> R<()> {
        self.steps += 1 + n as u64;
        if self.steps > MAX_STEPS {
            Err(ListError::TooMuchWork)
        } else {
            Ok(())
        }
    }

    fn lookup_ref(&self, name: &str) -> Option<&Value> {
        self.locals
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
            .or_else(|| self.base.get(name))
    }

    fn ev(&mut self, e: &Expr, depth: usize) -> R<Value> {
        if depth > MAX_DEPTH {
            return Err(ListError::TooDeep);
        }
        self.tick(0)?;
        match e {
            Expr::Num(v) => Ok(Value::Num(*v)),
            Expr::Var(n) => {
                if let Some(v) = self.lookup_ref(n) {
                    let v = v.clone();
                    let len = match &v {
                        Value::List(l) => l.len(),
                        Value::PointList(l) => l.len(),
                        _ => 0,
                    };
                    self.tick(len)?;
                    Ok(v)
                } else if let Some(c) = constant_value(n) {
                    Ok(Value::Num(c))
                } else {
                    Err(ListError::UnboundVariable(n.clone()))
                }
            }
            Expr::Neg(a) => match self.ev(a, depth + 1)? {
                Value::Num(x) => Ok(Value::Num(-x)),
                Value::List(l) => Ok(Value::List(l.into_iter().map(|x| -x).collect())),
                Value::Point(p) => Ok(Value::Point(p.into_iter().map(|x| -x).collect())),
                Value::PointList(ps) => Ok(Value::PointList(
                    ps.into_iter().map(|p| p.into_iter().map(|x| -x).collect()).collect(),
                )),
            },
            Expr::Bin(op, a, b) => {
                let x = self.ev(a, depth + 1)?;
                let y = self.ev(b, depth + 1)?;
                let op = *op;
                if let Some(v) = zip_map(&x, &y, |p, q| apply_bin(op, p, q))? {
                    if let Value::List(l) = &v {
                        self.tick(l.len())?;
                    }
                    return Ok(v);
                }
                point_op(op, &x, &y)
            }
            Expr::Rel(..) => Err(ListError::RelOutsideIndex),
            Expr::Tuple(items) => {
                let vals = items.iter().map(|i| self.ev(i, depth + 1)).collect::<R<Vec<_>>>()?;
                self.make_point(vals)
            }
            Expr::List(items) => {
                let vals = items.iter().map(|i| self.ev(i, depth + 1)).collect::<R<Vec<_>>>()?;
                collect_elements(vals)
            }
            Expr::Call(name, args) => self.call(name, args, depth),
        }
    }

    /// A tuple whose components are numbers is a point; list components broadcast into a list
    /// of points (scalar components repeat, list lengths must agree).
    fn make_point(&mut self, vals: Vec<Value>) -> R<Value> {
        let mut len: Option<usize> = None;
        for v in &vals {
            match v {
                Value::Num(_) => {}
                Value::List(l) => match len {
                    None => len = Some(l.len()),
                    Some(n) if n != l.len() => {
                        return Err(ListError::LengthMismatch { a: n, b: l.len() })
                    }
                    _ => {}
                },
                other => {
                    return Err(ListError::Type(format!("a point component cannot be {}", kind(other))))
                }
            }
        }
        match len {
            None => Ok(Value::Point(vals.iter().map(|v| v.as_num().unwrap_or(f64::NAN)).collect())),
            Some(n) => {
                self.tick(n * vals.len())?;
                Ok(Value::PointList(
                    (0..n)
                        .map(|i| {
                            vals.iter()
                                .map(|v| match v {
                                    Value::Num(x) => *x,
                                    Value::List(l) => l[i],
                                    _ => f64::NAN,
                                })
                                .collect()
                        })
                        .collect(),
                ))
            }
        }
    }

    fn call(&mut self, name: &str, args: &[Expr], depth: usize) -> R<Value> {
        match name {
            "range" => {
                check_arity(name, "2 or 3", args.len() == 2 || args.len() == 3, args.len())?;
                let mut nums = Vec::with_capacity(3);
                for a in args {
                    match self.ev(a, depth + 1)? {
                        Value::Num(x) => nums.push(x),
                        other => {
                            return Err(ListError::BadRange(format!("endpoints must be numbers, got {}", kind(&other))))
                        }
                    }
                }
                self.range(&nums)
            }
            "index" => {
                check_arity(name, "2", args.len() == 2, args.len())?;
                self.index(&args[0], &args[1], depth)
            }
            "for" => {
                check_arity(name, "3", args.len() == 3, args.len())?;
                let Expr::Var(var) = &args[1] else {
                    return Err(ListError::Type("a comprehension needs a loop variable".into()));
                };
                self.comprehension(&args[0], var, &args[2], depth)
            }
            "deriv" => {
                let call = Expr::Call(name.to_string(), args.to_vec());
                let e = calculus::expand(&call, self.base.angle).map_err(|e| ListError::Type(e.to_string()))?;
                self.ev(&e, depth + 1)
            }
            "int" | "sum" | "prod" => self.reduce(name, args, depth),
            _ => {
                let vals = args.iter().map(|a| self.ev(a, depth + 1)).collect::<R<Vec<_>>>()?;
                self.apply(name, vals)
            }
        }
    }

    /// `int(f, t, a, b)`, `sum(f, n, a, b)`, `prod(f, n, a, b)`. A body that only needs scalars
    /// goes through the f64 compiler (fast); one that indexes or builds lists is evaluated here
    /// term by term under the work budget.
    fn reduce(&mut self, name: &str, args: &[Expr], depth: usize) -> R<Value> {
        check_arity(name, "4", args.len() == 4, args.len())?;
        let Expr::Var(bound) = &args[1] else {
            return Err(ListError::Type(format!("{name} needs a variable name as its second argument")));
        };
        let rk = ReduceKind::from_name(name).expect("reduce is only called for int/sum/prod");
        let mut ends = [0.0; 2];
        for (slot, a) in ends.iter_mut().zip(&args[2..]) {
            match self.ev(a, depth + 1)? {
                Value::Num(x) => *slot = x,
                other => {
                    return Err(ListError::Type(format!("{name} bounds must be numbers, got {}", kind(&other))))
                }
            }
        }
        let [lo, hi] = ends;
        // Charge the budget up front: a sum of a million terms must not be free.
        let terms = match rk {
            ReduceKind::Int => 2_000,
            _ => match calculus::int_bounds(lo, hi) {
                Some((l, h)) => (h - l + 1).max(0) as usize,
                None => 0,
            },
        };
        self.tick(terms)?;

        let whole = Expr::Call(name.to_string(), args.to_vec());
        let free: Vec<String> = whole.free_vars().into_iter().collect();
        let scalars: Option<Vec<f64>> = free
            .iter()
            .map(|n| match self.lookup_ref(n) {
                Some(Value::Num(x)) => Some(*x),
                _ => None,
            })
            .collect();
        if let Some(vals) = scalars {
            let names: Vec<&str> = free.iter().map(String::as_str).collect();
            if let Ok(p) = compile(&whole, &names, self.base.angle) {
                return Ok(Value::Num(p.eval(&vals)));
            }
        }

        let body = &args[0];
        self.locals.push((bound.clone(), Value::Num(0.0)));
        let slot = self.locals.len() - 1;
        let mut failure: Option<ListError> = None;
        let result = {
            let mut at = |t: f64| -> f64 {
                if failure.is_some() {
                    return f64::NAN;
                }
                self.locals[slot].1 = Value::Num(t);
                match self.ev(body, depth + 1) {
                    Ok(Value::Num(x)) => x,
                    Ok(other) => {
                        failure = Some(ListError::Type(format!(
                            "the body of {name} must be a number, got {}",
                            kind(&other)
                        )));
                        f64::NAN
                    }
                    Err(e) => {
                        failure = Some(e);
                        f64::NAN
                    }
                }
            };
            match rk {
                ReduceKind::Int => calculus::integrate(&mut at, lo, hi),
                ReduceKind::Sum => calculus::sum_range(lo, hi, &mut at),
                ReduceKind::Prod => calculus::prod_range(lo, hi, &mut at),
            }
        };
        self.locals.truncate(slot);
        match failure {
            Some(e) => Err(e),
            None => Ok(Value::Num(result)),
        }
    }

    fn range(&mut self, n: &[f64]) -> R<Value> {
        if n.iter().any(|x| !x.is_finite()) {
            return Err(ListError::BadRange("endpoints must be finite".into()));
        }
        let (start, end) = (n[0], *n.last().unwrap());
        let step = if n.len() == 3 {
            n[1] - n[0]
        } else if end >= start {
            1.0
        } else {
            -1.0
        };
        if step == 0.0 || !step.is_finite() {
            return Err(ListError::BadRange("the step must be non-zero".into()));
        }
        let span = (end - start) / step;
        if span < 0.0 {
            return Ok(Value::List(Vec::new()));
        }
        let count = (span + 1e-9).floor() + 1.0;
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must be rejected too
        let too_long = !(count <= MAX_LIST_LEN as f64);
        if too_long {
            return Err(ListError::TooLong { limit: MAX_LIST_LEN });
        }
        let count = count as usize;
        self.tick(count)?;
        Ok(Value::List((0..count).map(|i| start + step * i as f64).collect()))
    }

    fn mask(&mut self, rel: &Expr, depth: usize) -> R<Vec<bool>> {
        // `L[0<L<5]`: every part of the chain must hold.
        if let Some(parts) = crate::ast::rel_chain(rel) {
            let mut all: Option<Vec<bool>> = None;
            for p in parts {
                let m = self.mask(p, depth)?;
                all = Some(match all {
                    None => m,
                    Some(prev) => prev.iter().zip(&m).map(|(a, b)| *a && *b).collect(),
                });
            }
            return Ok(all.unwrap_or_default());
        }
        let Expr::Rel(r, a, b) = rel else { unreachable!("mask is only called with Rel") };
        let x = self.ev(a, depth + 1)?;
        let y = self.ev(b, depth + 1)?;
        let r = *r;
        match zip_map(&x, &y, |p, q| if compare(r, p, q) { 1.0 } else { 0.0 })? {
            Some(Value::List(l)) => Ok(l.into_iter().map(|v| v != 0.0).collect()),
            Some(_) => Err(ListError::Type("a comparison inside an index must involve a list".into())),
            None => Err(ListError::Type(format!(
                "cannot compare {} with {}",
                kind(&x),
                kind(&y)
            ))),
        }
    }

    fn index(&mut self, l: &Expr, i: &Expr, depth: usize) -> R<Value> {
        enum Idx {
            One(f64),
            Many(Vec<f64>),
            Mask(Vec<bool>),
        }
        let idx = if matches!(i, Expr::Rel(..)) || crate::ast::rel_chain(i).is_some() {
            Idx::Mask(self.mask(i, depth)?)
        } else {
            match self.ev(i, depth + 1)? {
                Value::Num(x) => Idx::One(x),
                Value::List(v) => Idx::Many(v),
                other => {
                    return Err(ListError::Type(format!(
                        "an index must be a number, a list or a comparison, got {}",
                        kind(&other)
                    )))
                }
            }
        };
        // Avoid cloning a big list held in a variable (comprehensions index it repeatedly).
        let owned;
        let base: &Value = match l {
            Expr::Var(n) if self.lookup_ref(n).is_some() => self.lookup_ref(n).unwrap(),
            _ => {
                owned = self.ev(l, depth + 1)?;
                &owned
            }
        };
        let len = match base {
            Value::List(v) => v.len(),
            Value::PointList(v) => v.len(),
            other => {
                return Err(ListError::Type(format!("only lists can be indexed, got {}", kind(other))))
            }
        };
        let pick = |k: usize| -> Value {
            match base {
                Value::List(v) => Value::Num(v[k]),
                Value::PointList(v) => Value::Point(v[k].clone()),
                _ => unreachable!(),
            }
        };
        let slot = |x: f64| -> Option<usize> {
            if x.is_finite() && x == x.floor() && x >= 1.0 && x <= len as f64 {
                Some(x as usize - 1)
            } else {
                None
            }
        };
        let keep = |ks: Vec<usize>| -> Value {
            match base {
                Value::List(v) => Value::List(ks.into_iter().map(|k| v[k]).collect()),
                Value::PointList(v) => Value::PointList(ks.into_iter().map(|k| v[k].clone()).collect()),
                _ => unreachable!(),
            }
        };
        let out = match idx {
            Idx::One(x) => match slot(x) {
                Some(k) => pick(k),
                None => match base {
                    Value::PointList(v) if !v.is_empty() => Value::Point(vec![f64::NAN; v[0].len()]),
                    _ => Value::Num(f64::NAN),
                },
            },
            Idx::Many(xs) => keep(xs.into_iter().filter_map(slot).collect()),
            Idx::Mask(m) => {
                if m.len() != len {
                    return Err(ListError::LengthMismatch { a: len, b: m.len() });
                }
                keep(m.iter().enumerate().filter(|(_, b)| **b).map(|(k, _)| k).collect())
            }
        };
        let n = match &out {
            Value::List(v) => v.len(),
            Value::PointList(v) => v.len(),
            _ => 0,
        };
        self.tick(n)?;
        Ok(out)
    }

    fn comprehension(&mut self, body: &Expr, var: &str, source: &Expr, depth: usize) -> R<Value> {
        let items: Vec<Value> = match self.ev(source, depth + 1)? {
            Value::List(l) => l.into_iter().map(Value::Num).collect(),
            Value::PointList(l) => l.into_iter().map(Value::Point).collect(),
            other => {
                return Err(ListError::Type(format!("a comprehension iterates over a list, got {}", kind(&other))))
            }
        };
        if items.len() > MAX_LIST_LEN {
            return Err(ListError::TooLong { limit: MAX_LIST_LEN });
        }
        self.locals.push((var.to_string(), Value::Num(0.0)));
        let slot = self.locals.len() - 1;
        let mut out = Vec::with_capacity(items.len());
        let mut failure = None;
        for it in items {
            self.locals[slot].1 = it;
            match self.ev(body, depth + 1) {
                Ok(v) => out.push(v),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        self.locals.truncate(slot);
        if let Some(e) = failure {
            return Err(e);
        }
        collect_elements(out)
    }

    /// Scalar function (compiled once per name/arity) broadcast over its arguments.
    fn scalar(&mut self, name: &str, args: &[Value]) -> R<Value> {
        let key = (name.to_string(), args.len());
        if !self.cache.contains_key(&key) {
            let vars: Vec<String> = (0..args.len()).map(|i| format!("a{i}")).collect();
            let call = Expr::Call(name.to_string(), vars.iter().map(|v| Expr::Var(v.clone())).collect());
            let names: Vec<&str> = vars.iter().map(String::as_str).collect();
            let prog = compile(&call, &names, self.base.angle).map_err(|e| match e {
                CompileError::UnknownFunction(n) => ListError::UnknownFunction(n),
                CompileError::Arity { name, expected, got } => {
                    ListError::Arity { name, expected: expected.to_string(), got }
                }
                CompileError::UnboundVariable(n) => ListError::UnboundVariable(n),
                CompileError::NotScalar(w) => ListError::Type(format!("{w} is not a scalar")),
                CompileError::Calculus(m) => ListError::Type(m),
            })?;
            self.cache.insert(key.clone(), prog);
        }
        let prog = &self.cache[&key];
        let mut len: Option<usize> = None;
        for a in args {
            match a {
                Value::Num(_) => {}
                Value::List(l) => match len {
                    None => len = Some(l.len()),
                    Some(n) if n != l.len() => return Err(ListError::LengthMismatch { a: n, b: l.len() }),
                    _ => {}
                },
                other => {
                    return Err(ListError::Type(format!("{name} does not accept {}", kind(other))))
                }
            }
        }
        let mut slots = vec![0.0; args.len()];
        let mut stack = Vec::with_capacity(16);
        let at = |i: usize, slots: &mut Vec<f64>| {
            for (s, a) in slots.iter_mut().zip(args) {
                *s = match a {
                    Value::Num(x) => *x,
                    Value::List(l) => l[i],
                    _ => f64::NAN,
                };
            }
        };
        match len {
            None => {
                at(0, &mut slots);
                Ok(Value::Num(prog.eval_with(&slots, &mut stack)))
            }
            Some(n) => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    at(i, &mut slots);
                    out.push(prog.eval_with(&slots, &mut stack));
                }
                self.tick(n)?;
                Ok(Value::List(out))
            }
        }
    }

    /// Aggregates, list utilities, then scalar functions.
    fn apply(&mut self, name: &str, a: Vec<Value>) -> R<Value> {
        let n = a.len();
        let agg1 = |f: fn(&[f64]) -> Result<f64, StatsError>| -> R<Value> {
            check_arity(name, "1", n == 1, n)?;
            Ok(Value::Num(f(slice_of(&a[0])?)?))
        };
        match name {
            "length" | "count" => {
                check_arity(name, "1", n == 1, n)?;
                Ok(Value::Num(match &a[0] {
                    Value::PointList(p) => p.len() as f64,
                    other => slice_of(other)?.len() as f64,
                }))
            }
            "total" => {
                check_arity(name, "1", n == 1, n)?;
                Ok(Value::Num(stats::total(slice_of(&a[0])?)))
            }
            "mean" => agg1(stats::mean),
            "median" => agg1(stats::median),
            "var" => agg1(stats::var),
            "varp" => agg1(stats::varp),
            "stdev" => agg1(stats::stdev),
            "stdevp" => agg1(stats::stdevp),
            "mad" => agg1(stats::mad),
            "min" | "max" if n == 1 => agg1(if name == "min" { stats::min } else { stats::max }),
            "quartile" | "quantile" => {
                check_arity(name, "2", n == 2, n)?;
                let data = slice_of(&a[0])?;
                let f = if name == "quartile" { stats::quartile } else { stats::quantile };
                match &a[1] {
                    Value::Num(q) => Ok(Value::Num(f(data, *q)?)),
                    Value::List(qs) => {
                        self.tick(qs.len() * data.len())?;
                        Ok(Value::List(qs.iter().map(|q| f(data, *q)).collect::<Result<_, _>>()?))
                    }
                    other => Err(ListError::Type(format!("{name} needs a number or list, got {}", kind(other)))),
                }
            }
            "sort" => {
                check_arity(name, "1 or 2", n == 1 || n == 2, n)?;
                if n == 1 {
                    Ok(Value::List(stats::sort(slice_of(&a[0])?)))
                } else {
                    Ok(Value::List(stats::sort_by_keys(slice_of(&a[0])?, slice_of(&a[1])?)?))
                }
            }
            "reverse" => {
                check_arity(name, "1", n == 1, n)?;
                match &a[0] {
                    Value::PointList(p) => Ok(Value::PointList(p.iter().rev().cloned().collect())),
                    other => Ok(Value::List(stats::reverse(slice_of(other)?))),
                }
            }
            "unique" => {
                check_arity(name, "1", n == 1, n)?;
                Ok(Value::List(stats::unique(slice_of(&a[0])?)))
            }
            "join" => {
                check_arity(name, "at least 1", n >= 1, n)?;
                if a.iter().any(|v| matches!(v, Value::Point(_) | Value::PointList(_))) {
                    let mut pts: Vec<Vec<f64>> = Vec::new();
                    for v in &a {
                        match v {
                            Value::Point(p) => pts.push(p.clone()),
                            Value::PointList(l) => pts.extend(l.iter().cloned()),
                            other => {
                                return Err(ListError::Type(format!("cannot join {} with points", kind(other))))
                            }
                        }
                    }
                    if pts.windows(2).any(|w| w[0].len() != w[1].len()) {
                        return Err(ListError::Type("joined points must have the same dimension".into()));
                    }
                    self.tick(pts.len())?;
                    return Ok(Value::PointList(pts));
                }
                let parts = a.iter().map(slice_of).collect::<R<Vec<_>>>()?;
                let out = stats::join(&parts);
                self.tick(out.len())?;
                Ok(Value::List(out))
            }
            "gcd" | "lcm" if n == 1 => {
                // Over a list: the gcd / lcm of all its elements (NaN for an empty list).
                let l = slice_of(&a[0])?;
                let f: fn(f64, f64) -> f64 = if name == "gcd" { stats::gcd } else { stats::lcm };
                Ok(Value::Num(match l.split_first() {
                    Some((first, rest)) => rest.iter().fold(f(*first, *first), |acc, v| f(acc, *v)),
                    None => f64::NAN,
                }))
            }
            "distance" | "midpoint" => {
                check_arity(name, "2", n == 2, n)?;
                let pairs = point_pairs(name, &a[0], &a[1])?;
                self.tick(pairs.0.len())?;
                let is_list = pairs.1;
                if name == "distance" {
                    let d: Vec<f64> = pairs
                        .0
                        .iter()
                        .map(|(p, q)| p.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt())
                        .collect();
                    Ok(if is_list { Value::List(d) } else { Value::Num(d[0]) })
                } else {
                    let mut m: Vec<Vec<f64>> = pairs
                        .0
                        .iter()
                        .map(|(p, q)| p.iter().zip(q).map(|(x, y)| (x + y) / 2.0).collect())
                        .collect();
                    Ok(if is_list { Value::PointList(m) } else { Value::Point(m.remove(0)) })
                }
            }
            "random" => {
                check_arity(name, "0 to 2", n <= 2, n)?;
                if n == 0 {
                    return Ok(Value::Num(next_unit(&mut self.rng)));
                }
                let count = match &a[0] {
                    Value::Num(k) if k.is_finite() && *k >= 0.0 && *k <= MAX_LIST_LEN as f64 => *k as usize,
                    Value::Num(k) => {
                        return Err(ListError::TooLong { limit: if *k < 0.0 { 0 } else { MAX_LIST_LEN } })
                    }
                    other => {
                        return Err(ListError::Type(format!(
                            "random needs a count, got {}",
                            kind(other)
                        )))
                    }
                };
                let mut own;
                let state = if n == 2 {
                    let seed = match &a[1] {
                        Value::Num(s) if s.is_finite() => *s,
                        other => {
                            return Err(ListError::Type(format!(
                                "random needs a number seed, got {}",
                                kind(other)
                            )))
                        }
                    };
                    own = RANDOM_SEED ^ seed.to_bits().wrapping_mul(0xD6E8_FEB8_6659_FD93);
                    &mut own
                } else {
                    &mut self.rng
                };
                let out: Vec<f64> = (0..count).map(|_| next_unit(state)).collect();
                self.tick(count)?;
                Ok(Value::List(out))
            }
            "corr" | "cov" => {
                check_arity(name, "2", n == 2, n)?;
                let (x, y) = (slice_of(&a[0])?, slice_of(&a[1])?);
                Ok(Value::Num(if name == "corr" { stats::corr(x, y)? } else { stats::cov(x, y)? }))
            }
            _ => self.scalar(name, &a),
        }
    }
}

/// The `(p, q)` pairs of a two-point function over a point or a list of points on each side
/// (a single point repeats against a list), and whether the result is a list.
#[allow(clippy::type_complexity)]
fn point_pairs(name: &str, a: &Value, b: &Value) -> R<(Vec<(Vec<f64>, Vec<f64>)>, bool)> {
    let pts = |v: &Value| -> R<(Vec<Vec<f64>>, bool)> {
        match v {
            Value::Point(p) => Ok((vec![p.clone()], false)),
            Value::PointList(l) => Ok((l.clone(), true)),
            other => Err(ListError::Type(format!("{name} needs points, got {}", kind(other)))),
        }
    };
    let ((pa, la), (pb, lb)) = (pts(a)?, pts(b)?);
    let len = match (la, lb) {
        (false, false) => 1,
        (true, false) => pa.len(),
        (false, true) => pb.len(),
        (true, true) if pa.len() == pb.len() => pa.len(),
        (true, true) => return Err(ListError::LengthMismatch { a: pa.len(), b: pb.len() }),
    };
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let p = &pa[if la { i } else { 0 }];
        let q = &pb[if lb { i } else { 0 }];
        if p.len() != q.len() {
            return Err(ListError::Type(format!("{name} needs points of the same dimension")));
        }
        out.push((p.clone(), q.clone()));
    }
    Ok((out, la || lb))
}

/// Numbers make a list; points of equal dimension make a list of points.
fn collect_elements(vals: Vec<Value>) -> R<Value> {
    if vals.iter().all(|v| matches!(v, Value::Num(_))) {
        return Ok(Value::List(vals.iter().map(|v| v.as_num().unwrap()).collect()));
    }
    let mut pts = Vec::with_capacity(vals.len());
    for v in vals {
        match v {
            Value::Point(p) => pts.push(p),
            other => {
                return Err(ListError::Type(format!(
                    "a list holds numbers or points, not {} (lists cannot be nested)",
                    kind(&other)
                )))
            }
        }
    }
    if pts.windows(2).any(|w| w[0].len() != w[1].len()) {
        return Err(ListError::Type("points in a list must have the same dimension".into()));
    }
    Ok(Value::PointList(pts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn ev_with(src: &str, b: &Bindings) -> Result<Value, ListError> {
        eval_value(&parse(src).unwrap_or_else(|e| panic!("parse {src:?}: {e}")), b)
    }

    fn ev(src: &str) -> Value {
        let b = Bindings::new().with("L", Value::List(vec![5.0, 1.0, 4.0, 2.0, 3.0]));
        ev_with(src, &b).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    fn err(src: &str) -> ListError {
        let b = Bindings::new().with("L", Value::List(vec![5.0, 1.0, 4.0, 2.0, 3.0]));
        ev_with(src, &b).unwrap_err()
    }

    fn list(v: &[f64]) -> Value {
        Value::List(v.to_vec())
    }

    fn num(src: &str) -> f64 {
        ev(src).as_num().unwrap_or_else(|| panic!("{src} is not a number"))
    }

    #[test]
    fn literals_and_ranges() {
        assert_eq!(ev("[1,2,3]"), list(&[1.0, 2.0, 3.0]));
        assert_eq!(ev("[]"), list(&[]));
        assert_eq!(ev("[1...5]"), list(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(ev("[5...1]"), list(&[5.0, 4.0, 3.0, 2.0, 1.0]));
        assert_eq!(ev("[1,3...11]"), list(&[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]));
        assert_eq!(ev("[10,7...0]"), list(&[10.0, 7.0, 4.0, 1.0]));
        assert_eq!(ev("[1.5...4]"), list(&[1.5, 2.5, 3.5]));
        assert_eq!(ev("[0,0.1...0.3]").as_list().unwrap().len(), 4);
        assert_eq!(ev("[3...3]"), list(&[3.0]));
        // direction mismatch gives an empty list
        assert_eq!(ev("[1,2...-5]"), list(&[]));
        assert_eq!(err("[1,1...5]"), ListError::BadRange("the step must be non-zero".into()));
        assert!(matches!(err("[1...100000]"), ListError::TooLong { .. }));
        assert!(matches!(err("[1...1e300]"), ListError::TooLong { .. }));
        assert!(matches!(err("[1...L]"), ListError::BadRange(_)));
        assert_eq!(ev("[1...10000]").as_list().unwrap().len(), 10_000);
    }

    #[test]
    fn gcd_and_lcm_over_a_list() {
        assert_eq!(num("gcd([12, 18, 30])"), 6.0);
        assert_eq!(num("lcm([4, 6, 10])"), 60.0);
        assert_eq!(num("gcd([7])"), 7.0);
        assert_eq!(num("gcd(12, 18)"), 6.0);
        assert!(num("gcd([])").is_nan());
        assert!(num("gcd([4, 2.5])").is_nan());
        // two arguments still broadcast
        assert_eq!(ev("gcd([12, 20], 8)"), list(&[4.0, 4.0]));
        assert_eq!(ev("lcm([2, 3], [4, 6])"), list(&[4.0, 6.0]));
    }

    #[test]
    fn distance_and_midpoint() {
        assert_eq!(num("distance((0,0),(3,4))"), 5.0);
        assert_eq!(num("distance((1,2,3),(1,2,3))"), 0.0);
        assert!((num("distance((0,0,0),(1,2,2))") - 3.0).abs() < 1e-12);
        assert_eq!(ev("midpoint((0,0),(4,6))"), Value::Point(vec![2.0, 3.0]));
        // a point against a list repeats; two lists go pairwise
        assert_eq!(ev("distance((0,0),[(3,4),(6,8)])"), list(&[5.0, 10.0]));
        assert_eq!(
            ev("midpoint([(0,0),(2,2)],(2,0))"),
            Value::PointList(vec![vec![1.0, 0.0], vec![2.0, 1.0]])
        );
        assert_eq!(ev("distance([(0,0),(1,1)],[(0,3),(1,5)])"), list(&[3.0, 4.0]));
        assert!(matches!(err("distance([(0,0),(1,1)],[(0,3)])"), ListError::LengthMismatch { a: 2, b: 1 }));
        assert!(matches!(err("distance((0,0),(1,2,3))"), ListError::Type(_)));
        assert!(matches!(err("midpoint(1,(1,2))"), ListError::Type(_)));
        assert!(matches!(err("distance((0,0))"), ListError::Arity { .. }));
    }

    #[test]
    fn random_is_stable_per_evaluation_and_seedable() {
        let r = num("random()");
        assert!((0.0..1.0).contains(&r));
        assert_eq!(num("random()"), r, "the same expression gives the same number");
        // two calls in one expression draw successive numbers
        let two = ev("[random(), random()]");
        let l = two.as_list().unwrap();
        assert!(l[0] != l[1] && l[0] == r);
        let a = ev("random(5)");
        assert_eq!(a.as_list().unwrap().len(), 5);
        assert!(a.as_list().unwrap().iter().all(|v| (0.0..1.0).contains(v)));
        assert_eq!(ev("random(5, 1)"), ev("random(5, 1)"));
        assert!(ev("random(5, 1)") != ev("random(5, 2)"));
        assert_eq!(ev("random(0)"), list(&[]));
        // roughly uniform
        let big = ev("random(2000, 7)");
        let m: f64 = big.as_list().unwrap().iter().sum::<f64>() / 2000.0;
        assert!((m - 0.5).abs() < 0.05, "{m}");
        assert!(matches!(err("random(-1)"), ListError::TooLong { .. }));
        assert!(matches!(err("random(100000)"), ListError::TooLong { .. }));
        assert!(matches!(err("random(1,2,3)"), ListError::Arity { .. }));
        assert!(matches!(err("random(L)"), ListError::Type(_)));
        // usable in comprehension: a list of rolls 1..6
        let rolls = ev("[floor(6*random(1, k)[1])+1 for k=[1...50]]");
        assert!(rolls.as_list().unwrap().iter().all(|v| (1.0..=6.0).contains(v)));
    }

    #[test]
    fn broadcasting_arithmetic() {
        assert_eq!(ev("L+1"), list(&[6.0, 2.0, 5.0, 3.0, 4.0]));
        assert_eq!(ev("2L"), list(&[10.0, 2.0, 8.0, 4.0, 6.0]));
        assert_eq!(ev("-L"), list(&[-5.0, -1.0, -4.0, -2.0, -3.0]));
        assert_eq!(ev("L^2"), list(&[25.0, 1.0, 16.0, 4.0, 9.0]));
        assert_eq!(ev("[1,2,3]+[10,20,30]"), list(&[11.0, 22.0, 33.0]));
        assert_eq!(ev("[1,2,3]*[1,2,3]"), list(&[1.0, 4.0, 9.0]));
        assert_eq!(ev("1/[1,2,4]"), list(&[1.0, 0.5, 0.25]));
        assert_eq!(ev("2^[1,2,3]"), list(&[2.0, 4.0, 8.0]));
        assert_eq!(err("[1,2,3]+[1,2]"), ListError::LengthMismatch { a: 3, b: 2 });
        assert_eq!(num("2+3*4"), 14.0);
        assert_eq!(num("pi"), std::f64::consts::PI);
    }

    #[test]
    fn scalar_functions_broadcast() {
        assert_eq!(ev("sqrt([1,4,9])"), list(&[1.0, 2.0, 3.0]));
        assert_eq!(ev("abs([-1,2,-3])"), list(&[1.0, 2.0, 3.0]));
        assert_eq!(ev("mod([5,6,7],3)"), list(&[2.0, 0.0, 1.0]));
        assert_eq!(ev("max([1,5,3],2)"), list(&[2.0, 5.0, 3.0]));
        assert_eq!(ev("min([1,5,3],[2,2,2])"), list(&[1.0, 2.0, 2.0]));
        assert_eq!(ev("atan2([1,0],[0,1])").as_list().unwrap().len(), 2);
        assert_eq!(num("min(3,2)"), 2.0);
        assert_eq!(num("max(3,2)"), 3.0);
        assert!((num("sin(pi/2)") - 1.0).abs() < 1e-15);
        let b = Bindings::new().with_angle(Angle::Deg);
        let v = ev_with("sin(90)", &b).unwrap();
        assert!((v.as_num().unwrap() - 1.0).abs() < 1e-15);
        assert_eq!(err("foo([1,2])"), err("foo([1,2])"));
        assert!(matches!(err("sin(L,L)"), ListError::Arity { .. }));
        assert!(matches!(err("normalpdf(1,2)"), ListError::Arity { .. }));
        assert!(matches!(err("sqrt((1,2))"), ListError::Type(_)));
    }

    #[test]
    fn unbound_and_unknown_names() {
        assert_eq!(err("q+1"), ListError::UnboundVariable("q".into()));
        assert_eq!(err("[1,2,Z]"), ListError::UnboundVariable("Z".into()));
        let b = Bindings::new();
        let e = ev_with("L[1]", &b).unwrap_err();
        assert_eq!(e, ListError::UnboundVariable("L".into()));
        // Display is human readable
        assert_eq!(e.to_string(), "undefined variable 'L'");
        assert_eq!(ListError::LengthMismatch { a: 3, b: 2 }.to_string(), "list lengths differ (3 vs 2)");
    }

    #[test]
    fn aggregates() {
        assert_eq!(num("length(L)"), 5.0);
        assert_eq!(num("count(L)"), 5.0);
        assert_eq!(num("total(L)"), 15.0);
        assert_eq!(num("mean(L)"), 3.0);
        assert_eq!(num("median(L)"), 3.0);
        assert_eq!(num("min(L)"), 1.0);
        assert_eq!(num("max(L)"), 5.0);
        assert_eq!(num("var(L)"), 2.5);
        assert_eq!(num("varp(L)"), 2.0);
        assert!((num("stdev(L)") - 2.5f64.sqrt()).abs() < 1e-15);
        assert!((num("stdevp(L)") - 2.0f64.sqrt()).abs() < 1e-15);
        assert_eq!(num("mad(L)"), 1.2);
        assert_eq!(num("quartile(L,1)"), 2.0);
        assert_eq!(num("quartile(L,3)"), 4.0);
        assert_eq!(num("quantile(L,0.5)"), 3.0);
        assert_eq!(ev("quantile(L,[0,1])"), list(&[1.0, 5.0]));
        assert_eq!(num("mean([1...10])"), 5.5);
        assert_eq!(num("total([n^2 for n=[1...4]])"), 30.0);
        assert_eq!(num("mean(L)+1"), 4.0);
        assert_eq!(num("total(5)"), 5.0);
        assert_eq!(num("corr([1,2,3],[2,4,6])"), 1.0);
        assert_eq!(num("cov([1,2,3],[2,4,6])"), 2.0);
        assert_eq!(num("length([])"), 0.0);
        assert_eq!(num("total([])"), 0.0);
        assert!(matches!(err("mean([])"), ListError::Stats(StatsError::Empty)));
        assert!(matches!(err("var([1])"), ListError::Stats(StatsError::TooFew { .. })));
        assert!(matches!(err("corr([1,2],[1,2,3])"), ListError::Stats(StatsError::LengthMismatch { .. })));
        assert!(matches!(err("mean(1,2)"), ListError::Arity { .. }));
        assert!(matches!(err("quartile(L,7)"), ListError::Stats(StatsError::BadArg(_))));
        assert!(matches!(err("mean((1,2))"), ListError::Type(_)));
    }

    #[test]
    fn list_utilities() {
        assert_eq!(ev("sort(L)"), list(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(ev("sort([1,2,3],[3,1,2])"), list(&[2.0, 3.0, 1.0]));
        assert_eq!(ev("reverse(L)"), list(&[3.0, 2.0, 4.0, 1.0, 5.0]));
        assert_eq!(ev("unique([3,1,3,2,1])"), list(&[1.0, 2.0, 3.0]));
        assert_eq!(ev("join([1,2],[3],4)"), list(&[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(ev("join(1,2)"), list(&[1.0, 2.0]));
        assert!(matches!(err("sort([1,2],[1])"), ListError::Stats(_)));
        assert_eq!(ev("join((1,2),[(3,4)])"), Value::PointList(vec![vec![1.0, 2.0], vec![3.0, 4.0]]));
    }

    #[test]
    fn factorial_broadcasts_over_lists() {
        assert_eq!(ev("[0...5]!"), list(&[1.0, 1.0, 2.0, 6.0, 24.0, 120.0]));
        assert_eq!(ev("[n! for n=[1...4]]"), list(&[1.0, 2.0, 6.0, 24.0]));
        assert!((num("total([1/n! for n=[0...15]])") - std::f64::consts::E).abs() < 1e-12);
        assert_eq!(num("5!"), 120.0);
        assert_eq!(ev("nCr(5,[0...5])"), list(&[1.0, 5.0, 10.0, 10.0, 5.0, 1.0]));
    }

    #[test]
    fn indexing() {
        assert_eq!(num("L[1]"), 5.0);
        assert_eq!(num("L[5]"), 3.0);
        assert!(num("L[0]").is_nan());
        assert!(num("L[6]").is_nan());
        assert!(num("L[-1]").is_nan());
        assert!(num("L[1.5]").is_nan());
        assert_eq!(ev("L[2...4]"), list(&[1.0, 4.0, 2.0]));
        assert_eq!(ev("L[1,3...5]"), list(&[5.0, 4.0, 3.0]));
        assert_eq!(ev("L[[5,1,1]]"), list(&[3.0, 5.0, 5.0]));
        assert_eq!(ev("L[[1,9,0,2]]"), list(&[5.0, 1.0]));
        assert_eq!(ev("L[3...9]"), list(&[4.0, 2.0, 3.0]));
        assert_eq!(ev("L[L>3]"), list(&[5.0, 4.0]));
        assert_eq!(ev("L[L<=2]"), list(&[1.0, 2.0]));
        assert_eq!(ev("L[L=4]"), list(&[4.0]));
        assert_eq!(ev("L[L^2<10]"), list(&[1.0, 2.0, 3.0]));
        assert_eq!(ev("L[2L>7]"), list(&[5.0, 4.0]));
        assert_eq!(ev("L[L>10]"), list(&[]));
        assert_eq!(ev("L[L>3][1]"), Value::Num(5.0));
        assert_eq!(ev("L[2<L<=4]"), list(&[4.0, 3.0]));
        assert_eq!(ev("L[1<L<5]"), list(&[4.0, 2.0, 3.0]));
        assert_eq!(ev("sort(L)[2]"), Value::Num(2.0));
        assert_eq!(ev("[10,20,30][2]"), Value::Num(20.0));
        assert_eq!(ev("L[length(L)]"), Value::Num(3.0));
        // mask taken from a different list of the same length
        let b = Bindings::new()
            .with("L", list(&[1.0, 2.0, 3.0]))
            .with("M", list(&[9.0, 0.0, 9.0]));
        assert_eq!(ev_with("L[M>5]", &b).unwrap(), list(&[1.0, 3.0]));
        let b = b.with("K", list(&[1.0, 2.0]));
        assert_eq!(ev_with("L[K>1]", &b).unwrap_err(), ListError::LengthMismatch { a: 3, b: 2 });
        // after a number `[` is implicit multiplication, not an index
        assert_eq!(ev("5[1]"), list(&[5.0]));
        assert!(matches!(err("L[3>2]"), ListError::Type(_)));
        assert!(matches!(err("L[(1,2)]"), ListError::Type(_)));
        assert!(matches!(err("(1,2)[1]"), ListError::Type(_)));
        assert_eq!(err("L>3"), ListError::RelOutsideIndex);
        assert_eq!(err("y=3"), ListError::RelOutsideIndex);
    }

    #[test]
    fn comprehensions() {
        assert_eq!(ev("[n^2 for n=[1...5]]"), list(&[1.0, 4.0, 9.0, 16.0, 25.0]));
        assert_eq!(ev("[2k for k=L]"), list(&[10.0, 2.0, 8.0, 4.0, 6.0]));
        assert_eq!(ev("[x for x=L[L>3]]"), list(&[5.0, 4.0]));
        assert_eq!(ev("[L[i]-L[i+1] for i=[1...4]]"), list(&[4.0, -3.0, 2.0, -1.0]));
        assert_eq!(ev("[n for n=[]]"), list(&[]));
        // nested: inner variable shadows nothing, outer is visible
        assert_eq!(ev("total([total([i*j for j=[1...3]]) for i=[1...3]])"), Value::Num(36.0));
        // shadowing
        assert_eq!(ev("[n for n=[n for n=[1...3]]]"), list(&[1.0, 2.0, 3.0]));
        // points
        assert_eq!(
            ev("[(n,n^2) for n=[1...3]]"),
            Value::PointList(vec![vec![1.0, 1.0], vec![2.0, 4.0], vec![3.0, 9.0]])
        );
        // loop variable does not leak
        assert_eq!(err("total([n for n=[1...3]])+n"), ListError::UnboundVariable("n".into()));
        assert!(matches!(err("[n for n=5]"), ListError::Type(_)));
        assert!(matches!(err("[[1,2] for n=[1...3]]"), ListError::Type(_)));
        assert!(matches!(err("[q for n=[1...3]]"), ListError::UnboundVariable(_)));
        // source longer than the cap
        let big = Bindings::new().with("B", Value::List(vec![0.0; MAX_LIST_LEN + 1]));
        assert!(matches!(ev_with("[x for x=B]", &big), Err(ListError::TooLong { .. })));
        let ok = Bindings::new().with("B", Value::List(vec![1.0; MAX_LIST_LEN]));
        assert_eq!(ev_with("total([x for x=B])", &ok).unwrap(), Value::Num(10_000.0));
        // indexing a big list inside a comprehension must not clone it each time
        assert_eq!(ev_with("total([B[i] for i=[1...10000]])", &ok).unwrap(), Value::Num(10_000.0));
    }

    #[test]
    fn points() {
        assert_eq!(ev("(1,2)"), Value::Point(vec![1.0, 2.0]));
        assert_eq!(ev("(1,2)+(3,4)"), Value::Point(vec![4.0, 6.0]));
        assert_eq!(ev("2*(1,2)"), Value::Point(vec![2.0, 4.0]));
        assert_eq!(ev("(1,2)/2"), Value::Point(vec![0.5, 1.0]));
        assert_eq!(ev("-(1,2)"), Value::Point(vec![-1.0, -2.0]));
        assert_eq!(ev("([1,2,3],0)"), Value::PointList(vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]]));
        assert_eq!(
            ev("([1,2],[3,4])"),
            Value::PointList(vec![vec![1.0, 3.0], vec![2.0, 4.0]])
        );
        assert_eq!(
            ev("([1,2],[3,4])+(1,1)"),
            Value::PointList(vec![vec![2.0, 4.0], vec![3.0, 5.0]])
        );
        assert_eq!(ev("[(1,2),(3,4)][2]"), Value::Point(vec![3.0, 4.0]));
        match ev("[(1,2),(3,4)][3]") {
            Value::Point(p) => assert!(p.len() == 2 && p.iter().all(|x| x.is_nan())),
            other => panic!("{other:?}"),
        }
        assert_eq!(ev("length([(1,2),(3,4)])"), Value::Num(2.0));
        assert_eq!(err("([1,2],[1,2,3])"), ListError::LengthMismatch { a: 2, b: 3 });
        assert!(matches!(err("(1,2)+(1,2,3)"), ListError::LengthMismatch { .. }));
        assert!(matches!(err("(1,2)*(1,2)"), ListError::Type(_)));
        assert!(matches!(err("[(1,2),(1,2,3)]"), ListError::Type(_)));
        assert!(matches!(err("[(1,2),3]"), ListError::Type(_)));
        assert!(matches!(err("((1,2),3)"), ListError::Type(_)));
    }

    #[test]
    fn distributions_broadcast() {
        assert!((num("normalcdf(-1.96,1.96,0,1)") - 0.950004209703559).abs() < 1e-13);
        assert!((num("normalpdf(0,0,1)") - 0.3989422804014327).abs() < 1e-15);
        let v = ev("normalpdf([-1,0,1],0,1)");
        let l = v.as_list().unwrap();
        assert_eq!(l.len(), 3);
        assert_eq!(l[0], l[2]);
        assert_eq!(ev("binompdf(2,0.5,[0,1,2])"), list(&[0.25, 0.5, 0.25]));
        assert_eq!(ev("binompdf(4,0.5,[0,2])"), list(&[0.0625, 0.375]));
        assert!((num("total(binompdf(10,0.3,[0...10]))") - 1.0).abs() < 1e-12);
        assert!((num("invnorm(0.975,0,1)") - 1.959963984540054).abs() < 1e-13);
        assert!(num("normalpdf(0,0,-1)").is_nan());
        assert!((num("tcdf(-1,1,1)") - 0.5).abs() < 1e-14);
        assert!(matches!(err("normalpdf([1,2],[1,2,3],1)"), ListError::LengthMismatch { .. }));
    }

    #[test]
    fn hostile_input_is_an_error() {
        // deep nesting built by hand (the parser has its own cap)
        let mut e = Expr::num(1.0);
        for _ in 0..5000 {
            e = Expr::neg(e);
        }
        assert_eq!(eval_value(&e, &Bindings::new()), Err(ListError::TooDeep));
        let mut e = Expr::List(vec![Expr::num(1.0)]);
        for _ in 0..5000 {
            e = Expr::call("total", vec![e]);
        }
        assert_eq!(eval_value(&e, &Bindings::new()), Err(ListError::TooDeep));
        // quadratic/cubic blow-ups hit the work budget instead of hanging
        let r = ev_with("total([total([total([a*b*c for c=[1...300]]) for b=[1...300]]) for a=[1...300]])", &Bindings::new());
        assert_eq!(r, Err(ListError::TooMuchWork));
        let r = ev_with("[[1...10000] for n=[1...10000]]", &Bindings::new());
        assert!(r.is_err());
        // huge join chains
        let r = ev_with("join(join(L,L),join(L,L))", &Bindings::new().with("L", Value::List(vec![1.0; 100])));
        assert_eq!(r.unwrap().as_list().unwrap().len(), 400);
        assert!(matches!(ev_with("[1...1e9]", &Bindings::new()), Err(ListError::TooLong { .. })));
        assert!(ev_with("[1...nan]", &Bindings::new()).is_err());
        // NaN and infinities are values, not panics
        assert!(num("L[0/0]").is_nan());
        assert!(num("mean([1/0])").is_infinite());
    }
}
