//! Inlines user definitions (`a=3`, `f(x)=x^2+a`) and slider values into an expression so it
//! can be compiled with only spatial variables free.

use crate::analyze::Kind;
use crate::ast::{binder_var, Expr};
use crate::calculus;
use crate::compile::Angle;
use std::borrow::Cow;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolveError {
    Cycle(String),
    TooDeep,
    Arity { name: String, expected: usize, got: usize },
    /// A `deriv`/`int`/`sum`/`prod` call that is malformed or cannot be differentiated.
    Calculus(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Cycle(n) => write!(f, "'{n}' is defined in terms of itself"),
            ResolveError::TooDeep => write!(f, "definitions are nested too deeply"),
            ResolveError::Arity { name, expected, got } => {
                write!(f, "{name} takes {expected} argument(s), got {got}")
            }
            ResolveError::Calculus(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ResolveError {}

const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Default)]
pub struct Defs {
    vars: BTreeMap<String, Expr>,
    funcs: BTreeMap<String, (Vec<String>, Expr)>,
    angle: Angle,
}

impl Defs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define_var(&mut self, name: &str, body: Expr) {
        self.vars.insert(name.to_string(), body);
    }

    pub fn define_func(&mut self, name: &str, params: Vec<String>, body: Expr) {
        self.funcs.insert(name.to_string(), (params, body));
    }

    /// Slider values are plain numeric variables; they override a same-named definition.
    pub fn set_slider(&mut self, name: &str, value: f64) {
        self.vars.insert(name.to_string(), Expr::Num(value));
    }

    /// The angle unit trig functions use; `deriv` needs it to scale derivatives in degree mode.
    pub fn set_angle(&mut self, angle: Angle) {
        self.angle = angle;
    }

    pub fn angle(&self) -> Angle {
        self.angle
    }

    /// Forgets the scalar definitions of `names` (so they stay free variables when resolving,
    /// e.g. the parameters of a regression model).
    pub fn remove_vars<S: AsRef<str>>(&mut self, names: &[S]) {
        for n in names {
            self.vars.remove(n.as_ref());
        }
    }

    /// A copy without the scalar definitions of `names`.
    pub fn without_vars<S: AsRef<str>>(&self, names: &[S]) -> Defs {
        let mut d = self.clone();
        d.remove_vars(names);
        d
    }

    /// `self`, or a copy lacking the definitions of `names` when any of them is defined.
    fn scoped<S: AsRef<str>>(&self, names: &[S]) -> Cow<'_, Defs> {
        if names.iter().any(|n| self.vars.contains_key(n.as_ref())) {
            Cow::Owned(self.without_vars(names))
        } else {
            Cow::Borrowed(self)
        }
    }

    /// Whether `name` is defined as a scalar variable (a list, a constant or a slider value).
    pub fn has_var(&self, name: &str) -> bool {
        self.vars.contains_key(name)
    }

    /// Collects every `Definition` among the analyzed kinds. Later definitions of the same
    /// name are ignored (the first wins), matching top-to-bottom document order.
    pub fn from_kinds<'a>(kinds: impl IntoIterator<Item = &'a Kind>) -> Self {
        let mut d = Defs::new();
        for k in kinds {
            if let Kind::Definition { name, params, body } = k {
                if params.is_empty() {
                    d.vars.entry(name.clone()).or_insert_with(|| body.clone());
                } else {
                    d.funcs.entry(name.clone()).or_insert_with(|| (params.clone(), body.clone()));
                }
            }
        }
        d
    }

    pub fn defined_names(&self) -> std::collections::BTreeSet<String> {
        self.vars.keys().chain(self.funcs.keys()).cloned().collect()
    }

    pub fn resolve(&self, e: &Expr) -> Result<Expr, ResolveError> {
        self.go(e, &mut Vec::new())
    }

    fn go(&self, e: &Expr, stack: &mut Vec<String>) -> Result<Expr, ResolveError> {
        if stack.len() > MAX_DEPTH {
            return Err(ResolveError::TooDeep);
        }
        Ok(match e {
            Expr::Num(_) => e.clone(),
            Expr::Var(n) => match self.vars.get(n) {
                Some(body) => {
                    if stack.contains(n) {
                        return Err(ResolveError::Cycle(n.clone()));
                    }
                    stack.push(n.clone());
                    let r = self.go(body, stack);
                    stack.pop();
                    r?
                }
                None => e.clone(),
            },
            Expr::Neg(a) => Expr::Neg(Box::new(self.go(a, stack)?)),
            Expr::Bin(op, a, b) => {
                Expr::Bin(*op, Box::new(self.go(a, stack)?), Box::new(self.go(b, stack)?))
            }
            Expr::Rel(r, a, b) => {
                Expr::Rel(*r, Box::new(self.go(a, stack)?), Box::new(self.go(b, stack)?))
            }
            Expr::Tuple(items) => Expr::Tuple(self.go_all(items, stack)?),
            Expr::List(items) => Expr::List(self.go_all(items, stack)?),
            // `[body for n=source]` binds `n`: it shadows any definition of the same name inside
            // the body, and the header must stay a plain variable.
            Expr::Call(name, args) if name == "for" && args.len() == 3 => {
                let source = self.go(&args[2], stack)?;
                let body = match &args[1] {
                    Expr::Var(n) if self.vars.contains_key(n) => {
                        let mut inner = self.clone();
                        inner.vars.remove(n);
                        inner.go(&args[0], stack)?
                    }
                    _ => self.go(&args[0], stack)?,
                };
                Expr::Call(name.clone(), vec![body, args[1].clone(), source])
            }
            Expr::Call(name, args) if name == "deriv" => self.deriv(args, stack)?,
            // `int(f, t, a, b)`, `sum`, `prod` bind `t`: it shadows a definition of the same name
            // inside `f`, and the header must stay a plain variable.
            Expr::Call(name, args) if binder_var(name, args).is_some() => {
                let bound = binder_var(name, args).unwrap();
                let body = self.scoped(&[bound]).go(&args[0], stack)?;
                let (lo, hi) = (self.go(&args[2], stack)?, self.go(&args[3], stack)?);
                Expr::Call(name.clone(), vec![body, args[1].clone(), lo, hi])
            }
            Expr::Call(name, args) => {
                let args = self.go_all(args, stack)?;
                match self.funcs.get(name) {
                    Some((params, body)) => {
                        if params.len() != args.len() {
                            return Err(ResolveError::Arity {
                                name: name.clone(),
                                expected: params.len(),
                                got: args.len(),
                            });
                        }
                        if stack.contains(name) {
                            return Err(ResolveError::Cycle(name.clone()));
                        }
                        // Resolve the body first with its parameters left free (so a `deriv` in
                        // it sees the parameter as its variable), then substitute the arguments
                        // simultaneously: park the parameters under unique names so f(y, x) for
                        // f(x, y) cannot capture.
                        stack.push(name.clone());
                        let r = self.scoped(params).go(body, stack);
                        stack.pop();
                        let mut b = r?;
                        for (i, p) in params.iter().enumerate() {
                            b = b.subst(p, &Expr::Var(format!("${i}")));
                        }
                        for (i, a) in args.iter().enumerate() {
                            b = b.subst(&format!("${i}"), a);
                        }
                        b
                    }
                    None => Expr::Call(name.clone(), args),
                }
            }
        })
    }

    /// `deriv(f, x)` and `deriv(f, x, at)`: resolve `f` (with `x` left free), differentiate, and
    /// substitute `at`. `f` may be the bare name of a one-parameter user function.
    fn deriv(&self, args: &[Expr], stack: &mut Vec<String>) -> Result<Expr, ResolveError> {
        let err = |m: &str| ResolveError::Calculus(m.to_string());
        let var = match args {
            [_, Expr::Var(v)] | [_, Expr::Var(v), _] => v,
            _ => return Err(err("deriv takes (expression, variable) or (expression, variable, point)")),
        };
        let body = match &args[0] {
            Expr::Var(f) if !self.vars.contains_key(f) && self.funcs.get(f).is_some_and(|(p, _)| p.len() == 1) => {
                Expr::Call(f.clone(), vec![Expr::Var(var.clone())])
            }
            other => other.clone(),
        };
        let body = self.scoped(&[var.as_str()]).go(&body, stack)?;
        let mut resolved = vec![body, Expr::Var(var.clone())];
        if let Some(at) = args.get(2) {
            resolved.push(self.go(at, stack)?);
        }
        calculus::expand_deriv(&resolved, self.angle).map_err(|e| ResolveError::Calculus(e.to_string()))
    }

    fn go_all(&self, items: &[Expr], stack: &mut Vec<String>) -> Result<Vec<Expr>, ResolveError> {
        items.iter().map(|i| self.go(i, stack)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::analyze;
    use crate::compile::{compile, Angle};
    use crate::parse::{parse, parse_with, ParseCtx};
    use std::collections::BTreeSet;

    fn defs_from(lines: &[&str]) -> Defs {
        let mut ctx = ParseCtx::new();
        for l in lines {
            if let Ok(crate::ast::Expr::Rel(_, lhs, _)) = parse(l) {
                if let Expr::Call(n, _) = *lhs {
                    ctx = ctx.with_function(&n);
                }
            }
        }
        let kinds: Vec<Kind> = lines
            .iter()
            .map(|l| analyze(&parse_with(l, &ctx).unwrap(), &BTreeSet::new()).kind)
            .collect();
        Defs::from_kinds(&kinds)
    }

    fn eval(d: &Defs, src: &str, ctx: &ParseCtx, vars: &[&str], at: &[f64]) -> f64 {
        let e = d.resolve(&parse_with(src, ctx).unwrap()).unwrap();
        compile(&e, vars, Angle::Rad).unwrap().eval(at)
    }

    #[test]
    fn inlines_constants_and_functions() {
        let d = defs_from(&["a=3", "f(x)=x^2+a"]);
        let ctx = ParseCtx::new().with_function("f");
        assert_eq!(eval(&d, "f(2)+a", &ctx, &[], &[]), 10.0);
        assert_eq!(eval(&d, "f(y)", &ctx, &["y"], &[4.0]), 19.0);
    }

    #[test]
    fn substitution_is_simultaneous() {
        let d = defs_from(&["g(x,y)=x-y"]);
        let ctx = ParseCtx::new().with_function("g");
        // g(y, x) must be y - x, not x - x.
        assert_eq!(eval(&d, "g(y,x)", &ctx, &["x", "y"], &[1.0, 5.0]), 4.0);
    }

    #[test]
    fn nested_calls_resolve() {
        let d = defs_from(&["f(x)=2x", "g(x)=f(x)+1"]);
        let ctx = ParseCtx::new().with_function("f").with_function("g");
        assert_eq!(eval(&d, "g(f(3))", &ctx, &[], &[]), 13.0);
    }

    #[test]
    fn sliders_override_definitions() {
        let mut d = defs_from(&["a=3"]);
        d.set_slider("a", 7.0);
        assert_eq!(eval(&d, "a*2", &ParseCtx::new(), &[], &[]), 14.0);
    }

    #[test]
    fn cycles_arity_and_depth_are_errors() {
        let d = defs_from(&["a=b", "b=a"]);
        assert!(matches!(d.resolve(&parse("a").unwrap()), Err(ResolveError::Cycle(_))));
        let d = defs_from(&["a=a+1"]);
        // `a=a+1` is classified Implicit by the analyzer, so nothing is defined: a stays free.
        assert_eq!(d.resolve(&parse("a").unwrap()).unwrap(), Expr::var("a"));
        let d = defs_from(&["f(x)=x"]);
        let ctx = ParseCtx::new().with_function("f");
        assert!(matches!(
            d.resolve(&parse_with("f(1,2)", &ctx).unwrap()),
            Err(ResolveError::Arity { .. })
        ));
        let d = defs_from(&["f(x)=f(x)"]);
        assert!(matches!(
            d.resolve(&parse_with("f(1)", &ParseCtx::new().with_function("f")).unwrap()),
            Err(ResolveError::Cycle(_))
        ));
    }

    #[test]
    fn comprehension_variable_shadows_definitions() {
        use crate::list::{eval_value, Bindings, Value};
        let d = defs_from(&["n=3", "k=2"]);
        let e = d.resolve(&parse("[k*n for n=[1...3]]").unwrap()).unwrap();
        // k folds in; the loop variable n is NOT replaced by the definition n=3.
        let v = eval_value(&e, &Bindings::new()).unwrap();
        assert_eq!(v, Value::List(vec![2.0, 4.0, 6.0]));
        // Outside a comprehension n still resolves.
        let e = d.resolve(&parse("n+1").unwrap()).unwrap();
        assert_eq!(eval_value(&e, &Bindings::new()).unwrap(), Value::Num(4.0));
        // A definition used in the SOURCE of the comprehension still resolves.
        let e = d.resolve(&parse("[m for m=[1...n]]").unwrap()).unwrap();
        assert_eq!(eval_value(&e, &Bindings::new()).unwrap(), Value::List(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn spatial_variables_stay_free() {
        let d = defs_from(&["a=2"]);
        let e = d.resolve(&parse("y=a*x").unwrap()).unwrap();
        assert_eq!(e.free_vars().into_iter().collect::<Vec<_>>(), vec!["x", "y"]);
    }
}
