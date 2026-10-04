//! The expression tree shared by every stage (parser, printer, compiler, analyzer).

use std::collections::BTreeSet;
use std::f64::consts::{E, PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rel {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Rel(Rel, Box<Expr>, Box<Expr>),
}

/// Names the lexer treats as functions (longest match wins over single letters).
pub const BUILTIN_FUNCS: &[&str] = &[
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "arcsin", "arccos",
    "arctan", "sinh", "cosh", "tanh", "exp", "ln", "log", "sqrt", "cbrt", "abs", "floor",
    "ceil", "round", "sign", "sgn", "min", "max", "mod", "atan2",
    // lists and statistics
    "length", "count", "total", "mean", "median", "var", "varp", "stdev", "stdevp", "mad",
    "quartile", "quantile", "sort", "reverse", "join", "unique", "corr", "cov",
    // statistical plots (drawn by the scene, not evaluated as numbers)
    "histogram", "boxplot", "dotplot",
    // distributions
    "normalpdf", "normalcdf", "invnorm", "binompdf", "binomcdf", "poissonpdf", "poissoncdf",
    "uniformpdf", "uniformcdf", "tpdf", "tcdf", "invt",
    // calculus: `deriv(f, x)`, `deriv(f, x, at)`, `int(f, t, a, b)`, `sum(f, n, a, b)`,
    // `prod(f, n, a, b)` (see `calculus`)
    "deriv", "int", "sum", "prod",
    // combinatorics: `factorial(n)` (the same node as the postfix `n!`), `nCr(n, k)`, `nPr(n, k)`
    "factorial", "nCr", "nPr",
];

/// The name of the call a chained comparison (`0<=y<=f(x)`) desugars to: `and(a<b, b<c)` with
/// two or three [`Expr::Rel`] arguments, each pair sharing its middle operand. Not a builtin
/// function name, so the lexer never produces it from user text.
pub const CHAIN_FN: &str = "and";

/// The comparisons of a chained-comparison node, if `e` is one.
pub fn rel_chain(e: &Expr) -> Option<&[Expr]> {
    match e {
        Expr::Call(n, args)
            if n == CHAIN_FN && (2..=3).contains(&args.len()) && args.iter().all(|a| matches!(a, Expr::Rel(r, ..) if *r != Rel::Eq)) =>
        {
            Some(args)
        }
        _ => None,
    }
}

/// Multi-letter names that are not functions but must not be split into letters.
pub const NAMED_SYMBOLS: &[&str] = &["pi", "tau", "theta"];

pub const CONSTANTS: &[(&str, f64)] = &[("pi", PI), ("e", E), ("tau", TAU)];

pub fn constant_value(name: &str) -> Option<f64> {
    CONSTANTS.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

pub fn is_builtin_func(name: &str) -> bool {
    BUILTIN_FUNCS.contains(&name)
}

/// For `int(f, t, a, b)`, `sum(f, n, a, b)` and `prod(f, n, a, b)`: the bound variable of the
/// body `f`. The bounds `a` and `b` are outside its scope.
pub fn binder_var<'a>(name: &str, args: &'a [Expr]) -> Option<&'a str> {
    match (name, args) {
        ("int" | "sum" | "prod", [_, Expr::Var(v), _, _]) => Some(v),
        _ => None,
    }
}

impl Expr {
    pub fn num(v: f64) -> Expr {
        Expr::Num(v)
    }

    pub fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    pub fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Bin(op, Box::new(l), Box::new(r))
    }

    pub fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call(name.to_string(), args)
    }

    #[allow(clippy::should_implement_trait)] // a constructor, not unary negation of `self`
    pub fn neg(e: Expr) -> Expr {
        Expr::Neg(Box::new(e))
    }

    /// Variable names used in the expression, excluding the built-in constants.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(n) => {
                if constant_value(n).is_none() {
                    out.insert(n.clone());
                }
            }
            Expr::Neg(a) => a.collect_vars(out),
            Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Call(name, args) => {
                if let Some(bound) = binder_var(name, args) {
                    let mut body = BTreeSet::new();
                    args[0].collect_vars(&mut body);
                    body.remove(bound);
                    out.extend(body);
                    args[2].collect_vars(out);
                    args[3].collect_vars(out);
                } else {
                    args.iter().for_each(|a| a.collect_vars(out))
                }
            }
            Expr::Tuple(args) | Expr::List(args) => args.iter().for_each(|a| a.collect_vars(out)),
        }
    }

    /// Names of functions called anywhere in the expression.
    pub fn called_functions(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_calls(&mut out);
        out
    }

    fn collect_calls(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Num(_) | Expr::Var(_) => {}
            Expr::Neg(a) => a.collect_calls(out),
            Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => {
                a.collect_calls(out);
                b.collect_calls(out);
            }
            Expr::Call(n, args) => {
                out.insert(n.clone());
                args.iter().for_each(|a| a.collect_calls(out));
            }
            Expr::Tuple(args) | Expr::List(args) => args.iter().for_each(|a| a.collect_calls(out)),
        }
    }

    pub fn contains_var(&self, name: &str) -> bool {
        match self {
            Expr::Num(_) => false,
            Expr::Var(n) => n == name,
            Expr::Neg(a) => a.contains_var(name),
            Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => a.contains_var(name) || b.contains_var(name),
            Expr::Call(f, args) => match binder_var(f, args) {
                Some(bound) => {
                    (bound != name && args[0].contains_var(name))
                        || args[2].contains_var(name)
                        || args[3].contains_var(name)
                }
                None => args.iter().any(|a| a.contains_var(name)),
            },
            Expr::Tuple(args) | Expr::List(args) => args.iter().any(|a| a.contains_var(name)),
        }
    }

    /// Replace every occurrence of variable `name` with `with`.
    pub fn subst(&self, name: &str, with: &Expr) -> Expr {
        match self {
            Expr::Num(_) => self.clone(),
            Expr::Var(n) => {
                if n == name {
                    with.clone()
                } else {
                    self.clone()
                }
            }
            Expr::Neg(a) => Expr::Neg(Box::new(a.subst(name, with))),
            Expr::Bin(op, a, b) => {
                Expr::Bin(*op, Box::new(a.subst(name, with)), Box::new(b.subst(name, with)))
            }
            Expr::Rel(r, a, b) => {
                Expr::Rel(*r, Box::new(a.subst(name, with)), Box::new(b.subst(name, with)))
            }
            Expr::Call(n, args) => {
                if let Some(bound) = binder_var(n, args) {
                    return self.subst_binder(n, args, bound, name, with);
                }
                Expr::Call(n.clone(), args.iter().map(|a| a.subst(name, with)).collect())
            }
            Expr::Tuple(args) => Expr::Tuple(args.iter().map(|a| a.subst(name, with)).collect()),
            Expr::List(args) => Expr::List(args.iter().map(|a| a.subst(name, with)).collect()),
        }
    }

    /// `subst` through `int`/`sum`/`prod`: the bound variable shadows `name` inside the body and
    /// is renamed if it would capture a variable of `with`.
    fn subst_binder(&self, call: &str, args: &[Expr], bound: &str, name: &str, with: &Expr) -> Expr {
        let (lo, hi) = (args[2].subst(name, with), args[3].subst(name, with));
        let mut body = args[0].clone();
        let mut bound = bound.to_string();
        if bound != name && body.contains_var(name) {
            if with.contains_var(&bound) {
                let mut fresh = format!("{bound}_");
                while body.contains_var(&fresh) || with.contains_var(&fresh) {
                    fresh.push('_');
                }
                body = body.subst(&bound, &Expr::Var(fresh.clone()));
                bound = fresh;
            }
            body = body.subst(name, with);
        }
        Expr::Call(call.to_string(), vec![body, Expr::Var(bound), lo, hi])
    }
}
