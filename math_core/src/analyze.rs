//! Classifies a parsed expression and orders a document by dependency.

use crate::ast::{rel_chain, Expr, Rel};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// `f(x) = ...` (params non-empty) or `a = 3` (params empty).
    Definition { name: String, params: Vec<String>, body: Expr },
    /// `y = f(x)` (the right side does not mention `y`).
    ExplicitY { rhs: Expr },
    /// `x = f(y)`.
    ExplicitX { rhs: Expr },
    /// `z = f(x, y)`.
    ExplicitZ { rhs: Expr },
    /// `r = f(theta)`.
    Polar { rhs: Expr },
    /// Any other equation, stored as `lhs - rhs = 0`.
    Implicit { f: Expr },
    /// `lhs rel rhs` with `<`, `<=`, `>` or `>=`, stored as `lhs - rhs`. A chained comparison
    /// (`0<=y<=f(x)`, `1<x^2+y^2<=4`) is the same kind with `rel: Lt` and `f` the largest of
    /// the parts' margins (`lhs - rhs` for `<`/`<=`, `rhs - lhs` for `>`/`>=`): the region is
    /// where every part holds, i.e. where `f < 0`, and `f = 0` traces exactly the region's
    /// edge, so the existing fill and boundary pipeline draws it unchanged.
    Inequality { rel: Rel, f: Expr },
    /// A tuple that depends on `t`, such as `(cos(t), sin(t))`: a parametric curve. A tuple of
    /// three that uses both `u` and `v` (and no `t`), such as `(cos(u), sin(u), v)`, is a
    /// parametric SURFACE: see [`Kind::is_param_surface`].
    Parametric { components: Vec<Expr> },
    /// A constant tuple such as `(1, 2)` or `(1, 2, 3)`.
    Point { components: Vec<Expr> },
    /// A tuple of expressions of x, y(, z) with no parametric variable: `(-y, x)` is a vector
    /// field drawn as arrows. Length 1..=3.
    VectorField { components: Vec<Expr> },
    List { items: Vec<Expr> },
    /// `lhs ~ model`: a regression of the list expression `lhs` on `model`, whose free names
    /// that are not lists are the parameters (see [`crate::regress`]).
    Regression { lhs: Expr, model: Expr },
    /// A scalar expression of x, y, z (a field to colour).
    Field { expr: Expr },
    /// A scalar with no spatial variables, e.g. `2+3`.
    Value { expr: Expr },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub kind: Kind,
    /// Every non-constant variable used.
    pub free: BTreeSet<String>,
    /// Names that should become sliders: free, not spatial, not defined elsewhere.
    pub slider_candidates: BTreeSet<String>,
    /// Which of x, y, z appear.
    pub dims: [bool; 3],
    /// The ranges of a trailing `{a<=t<=b, ...}` (see [`crate::param`]); empty without one.
    pub domain: Vec<crate::param::Range>,
}

impl Kind {
    /// A parametric tuple of three in `u` and `v` (no `t`): `(x(u,v), y(u,v), z(u,v))`.
    pub fn is_param_surface(&self) -> bool {
        match self {
            Kind::Parametric { components } => is_surface_tuple(components),
            _ => false,
        }
    }
}

fn is_surface_tuple(items: &[Expr]) -> bool {
    let uses = |v: &str| items.iter().any(|i| i.contains_var(v));
    items.len() == 3 && !uses("t") && uses("u") && uses("v")
}

pub const SPATIAL: [&str; 3] = ["x", "y", "z"];
const RESERVED: &[&str] = &["x", "y", "z", "t", "r", "theta"];

fn sub(a: &Expr, b: &Expr) -> Expr {
    Expr::bin(crate::ast::BinOp::Sub, a.clone(), b.clone())
}

fn single_var(e: &Expr) -> Option<&str> {
    if let Expr::Var(n) = e {
        Some(n)
    } else {
        None
    }
}

/// `defined` are names (variables or functions) defined elsewhere in the document.
pub fn analyze(e: &Expr, defined: &BTreeSet<String>) -> Analysis {
    let free = e.free_vars();
    let (body, domain) = crate::param::unwrap_domain(e);
    let dims = [body.contains_var("x"), body.contains_var("y"), body.contains_var("z")];
    let kind = classify(body);
    let own: BTreeSet<String> = match &kind {
        Kind::Definition { name, params, .. } => {
            let mut s: BTreeSet<String> = params.iter().cloned().collect();
            s.insert(name.clone());
            s
        }
        _ => BTreeSet::new(),
    };
    // `u` and `v` are the surface parameters of a parametric surface, never sliders.
    let surface = kind.is_param_surface();
    let slider_candidates = free
        .iter()
        .filter(|n| !(surface && (*n == "u" || *n == "v")))
        .filter(|n| !RESERVED.contains(&n.as_str()) && !defined.contains(*n) && !own.contains(*n))
        .cloned()
        .collect();
    Analysis { kind, free, slider_candidates, dims, domain }
}

fn classify(e: &Expr) -> Kind {
    match e {
        Expr::Rel(Rel::Eq, lhs, rhs) => {
            if let Expr::Call(name, args) = &**lhs {
                let params: Option<Vec<String>> =
                    args.iter().map(|a| single_var(a).map(str::to_string)).collect();
                if let Some(params) = params {
                    return Kind::Definition { name: name.clone(), params, body: (**rhs).clone() };
                }
            }
            match single_var(lhs) {
                Some("y") if !rhs.contains_var("y") => Kind::ExplicitY { rhs: (**rhs).clone() },
                Some("x") if !rhs.contains_var("x") => Kind::ExplicitX { rhs: (**rhs).clone() },
                Some("z") if !rhs.contains_var("z") => Kind::ExplicitZ { rhs: (**rhs).clone() },
                Some("r") if !rhs.contains_var("r") => Kind::Polar { rhs: (**rhs).clone() },
                Some(name) if !RESERVED.contains(&name) && !rhs.contains_var(name) => {
                    Kind::Definition { name: name.to_string(), params: vec![], body: (**rhs).clone() }
                }
                _ => Kind::Implicit { f: sub(lhs, rhs) },
            }
        }
        Expr::Rel(rel, lhs, rhs) if *rel != Rel::Ne => Kind::Inequality { rel: *rel, f: sub(lhs, rhs) },
        Expr::Call(..) if rel_chain(e).is_some() => {
            let margins = rel_chain(e).unwrap().iter().filter_map(|p| match p {
                Expr::Rel(Rel::Gt | Rel::Ge, l, r) => Some(sub(r, l)),
                Expr::Rel(_, l, r) => Some(sub(l, r)),
                _ => None,
            });
            let f = margins.reduce(|a, b| Expr::call("max", vec![a, b])).expect("a chain has parts");
            Kind::Inequality { rel: Rel::Lt, f }
        }
        Expr::Tuple(items) => {
            let uses = |v: &str| items.iter().any(|i| i.contains_var(v));
            if uses("t") || is_surface_tuple(items) {
                Kind::Parametric { components: items.clone() }
            } else if (1..=3).contains(&items.len())
                && SPATIAL.iter().any(|v| uses(v))
                && !uses("u")
                && !uses("v")
            {
                Kind::VectorField { components: items.clone() }
            } else {
                Kind::Point { components: items.clone() }
            }
        }
        Expr::List(items) => Kind::List { items: items.clone() },
        Expr::Call(name, args) if name == "regress" && args.len() == 2 => {
            Kind::Regression { lhs: args[0].clone(), model: args[1].clone() }
        }
        // `[1...10]` and `[f(n) for n=...]` are list-valued: the single item is the whole
        // list-valued expression (evaluate it with `list::eval_value`).
        Expr::Call(name, _)
            if (name == "range" || name == "for") && !SPATIAL.iter().any(|v| e.contains_var(v)) =>
        {
            Kind::List { items: vec![e.clone()] }
        }
        other => {
            if SPATIAL.iter().any(|v| other.contains_var(v)) {
                Kind::Field { expr: other.clone() }
            } else {
                Kind::Value { expr: other.clone() }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderError {
    Cycle(Vec<usize>),
}

/// One document row for dependency ordering.
#[derive(Debug, Clone)]
pub struct Node {
    /// The name this row defines, if any.
    pub defines: Option<String>,
    /// Names it uses (free variables and called user functions).
    pub uses: BTreeSet<String>,
}

/// Returns row indices so every definition precedes its users. Rows that use only
/// undefined names are unconstrained. A cycle returns the rows involved.
pub fn dependency_order(nodes: &[Node]) -> Result<Vec<usize>, OrderError> {
    let mut definer: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, n) in nodes.iter().enumerate() {
        if let Some(d) = &n.defines {
            definer.entry(d.as_str()).or_insert(i);
        }
    }
    let deps: Vec<BTreeSet<usize>> = nodes
        .iter()
        .map(|n| n.uses.iter().filter_map(|u| definer.get(u.as_str()).copied()).collect())
        .collect();

    // 0 = unvisited, 1 = in progress, 2 = done
    let mut state = vec![0u8; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    let mut stack_path: Vec<usize> = Vec::new();

    fn visit(
        i: usize,
        deps: &[BTreeSet<usize>],
        state: &mut [u8],
        order: &mut Vec<usize>,
        path: &mut Vec<usize>,
    ) -> Result<(), OrderError> {
        match state[i] {
            2 => return Ok(()),
            1 => {
                let start = path.iter().position(|p| *p == i).unwrap_or(0);
                return Err(OrderError::Cycle(path[start..].to_vec()));
            }
            _ => {}
        }
        state[i] = 1;
        path.push(i);
        for &d in &deps[i] {
            visit(d, deps, state, order, path)?;
        }
        path.pop();
        state[i] = 2;
        order.push(i);
        Ok(())
    }

    for i in 0..nodes.len() {
        visit(i, &deps, &mut state, &mut order, &mut stack_path)?;
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn kind(s: &str) -> Kind {
        analyze(&parse(s).unwrap(), &BTreeSet::new()).kind
    }

    #[test]
    fn classifies_equations() {
        assert!(matches!(kind("y=x^2"), Kind::ExplicitY { .. }));
        assert!(matches!(kind("x=y^2"), Kind::ExplicitX { .. }));
        assert!(matches!(kind("z=sin(x)cos(y)"), Kind::ExplicitZ { .. }));
        assert!(matches!(kind("r=2sin(theta)"), Kind::Polar { .. }));
        assert!(matches!(kind("x^2+y^2=4"), Kind::Implicit { .. }));
        assert!(matches!(kind("y=y+x"), Kind::Implicit { .. }));
        assert!(matches!(kind("y<=x"), Kind::Inequality { rel: Rel::Le, .. }));
        assert!(matches!(kind("0<=y<=x^2"), Kind::Inequality { rel: Rel::Lt, .. }));
        match kind("f(x)=x^2") {
            Kind::Definition { name, params, .. } => {
                assert_eq!(name, "f");
                assert_eq!(params, vec!["x"]);
            }
            k => panic!("{k:?}"),
        }
        match kind("a=3") {
            Kind::Definition { name, params, .. } => {
                assert_eq!(name, "a");
                assert!(params.is_empty());
            }
            k => panic!("{k:?}"),
        }
    }

    #[test]
    fn chained_comparison_is_one_margin_function() {
        use crate::compile::{compile, Angle};
        let f = |src: &str| match kind(src) {
            Kind::Inequality { rel: Rel::Lt, f } => compile(&f, &["x", "y"], Angle::Rad).unwrap(),
            k => panic!("{k:?}"),
        };
        // region where the margin is negative; its zero set is the region's edge
        let band = f("0<=y<=x^2");
        assert!(band.eval(&[1.0, 0.5]) < 0.0);
        assert!(band.eval(&[1.0, 2.0]) > 0.0);
        assert!(band.eval(&[1.0, -1.0]) > 0.0);
        assert_eq!(band.eval(&[2.0, 4.0]), 0.0);
        let ring = f("1<x^2+y^2<=4");
        assert!(ring.eval(&[1.5, 0.0]) < 0.0);
        assert!(ring.eval(&[0.5, 0.0]) > 0.0 && ring.eval(&[2.5, 0.0]) > 0.0);
        // `>` parts are flipped: 4 > r2 > 1 is the same ring
        let flipped = f("4>=x^2+y^2>1");
        for p in [[1.5, 0.0], [0.5, 0.0], [2.5, 0.0], [0.0, 1.0]] {
            assert_eq!(ring.eval(&p), flipped.eval(&p));
        }
        let three = f("0<x<y<=3");
        assert!(three.eval(&[1.0, 2.0]) < 0.0 && three.eval(&[2.0, 1.0]) > 0.0 && three.eval(&[1.0, 4.0]) > 0.0);
    }

    #[test]
    fn classifies_values_fields_points() {
        assert!(matches!(kind("2+3"), Kind::Value { .. }));
        assert!(matches!(kind("x+y"), Kind::Field { .. }));
        assert!(matches!(kind("(1,2)"), Kind::Point { .. }));
        assert!(matches!(kind("(cos(t),sin(t))"), Kind::Parametric { .. }));
        assert!(matches!(kind("[1,2]"), Kind::List { .. }));
        // x/y(/z) tuples without t are vector fields; a parameter or a constant keeps the old kinds
        assert!(matches!(kind("(-y,x)"), Kind::VectorField { .. }));
        assert!(matches!(kind("(y,z,x)"), Kind::VectorField { .. }));
        assert!(matches!(kind("(cos(t),x)"), Kind::Parametric { .. }));
        assert!(matches!(kind("(a,b)"), Kind::Point { .. }));
        // list syntax: ranges and comprehensions are lists; aggregates and indexing are values
        assert!(matches!(kind("[1...5]"), Kind::List { .. }));
        assert!(matches!(kind("[n^2 for n=[1...5]]"), Kind::List { .. }));
        assert!(matches!(kind("mean([1,2,3])"), Kind::Value { .. }));
        assert!(matches!(kind("L[2]"), Kind::Value { .. }));
        assert!(matches!(kind("L[L>3]"), Kind::Value { .. }));
        assert!(matches!(kind("total([1...10])"), Kind::Value { .. }));
        // distributions of x still plot as ordinary curves
        assert!(matches!(kind("y=normalpdf(x,0,1)"), Kind::ExplicitY { .. }));
        assert!(matches!(kind("y=tcdf(-10,x,5)"), Kind::ExplicitY { .. }));
        assert!(matches!(kind("normalpdf(x,0,1)"), Kind::Field { .. }));
    }

    #[test]
    fn classifies_ranges_pairs_and_surfaces() {
        let a = analyze(&parse("(t^2, 2t) {-3<=t<=3}").unwrap(), &BTreeSet::new());
        assert!(matches!(a.kind, Kind::Parametric { ref components } if components.len() == 2));
        assert_eq!(a.domain.len(), 1);
        assert_eq!(a.domain[0].var, "t");
        assert!(a.slider_candidates.is_empty());
        // a slider in a bound is a slider candidate
        let a = analyze(&parse("(cos(t), sin(t)) {0<=t<=a}").unwrap(), &BTreeSet::new());
        assert_eq!(a.slider_candidates.iter().collect::<Vec<_>>(), vec!["a"]);
        // polar with a range
        let a = analyze(&parse("r=theta {0<=theta<=6pi}").unwrap(), &BTreeSet::new());
        assert!(matches!(a.kind, Kind::Polar { .. }) && a.domain[0].var == "theta");
        // pair in one item is the tuple of the right sides, any order, 2D or 3D
        assert_eq!(parse("x=3cos(t), y=2sin(t)").unwrap(), parse("(3cos(t), 2sin(t))").unwrap());
        assert_eq!(parse("y=2sin(t); x=3cos(t)").unwrap(), parse("(3cos(t), 2sin(t))").unwrap());
        assert_eq!(parse("x=cos(t), y=sin(t), z=t/4").unwrap(), parse("(cos(t), sin(t), t/4)").unwrap());
        assert!(matches!(kind("x=cos(t), y=sin(t) {0<=t<=pi}"), Kind::Parametric { .. }));
        // not a pair: a repeated axis, an axis on the right, or just a stray comma
        assert!(parse("x=1, x=2").is_err());
        assert!(parse("x=y, y=2").is_err());
        assert!(parse("y=2x, 3").is_err());
        // surfaces
        let a = analyze(&parse("(3cos(u)cos(v), 3cos(u)sin(v), 3sin(u))").unwrap(), &BTreeSet::new());
        assert!(a.kind.is_param_surface() && a.slider_candidates.is_empty());
        assert!(!kind("(cos(t), sin(t), t)").is_param_surface());
        assert!(!kind("(u, 2)").is_param_surface() && matches!(kind("(u, 2)"), Kind::Point { .. }));
        let a = analyze(&parse("(a u, v, 0) {0<=u<=1, 0<=v<=1}").unwrap(), &BTreeSet::new());
        assert!(a.kind.is_param_surface() && a.domain.len() == 2);
        assert_eq!(a.slider_candidates.iter().collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn slider_candidates_exclude_spatial_and_defined() {
        let defined: BTreeSet<String> = ["b".to_string()].into_iter().collect();
        let a = analyze(&parse("y=a*x+b+c").unwrap(), &defined);
        let names: Vec<&str> = a.slider_candidates.iter().map(String::as_str).collect();
        assert_eq!(names, vec!["a", "c"]);
        assert_eq!(a.dims, [true, true, false]);
        // A definition does not make its own name a slider.
        let a = analyze(&parse("k=3").unwrap(), &BTreeSet::new());
        assert!(a.slider_candidates.is_empty());
    }

    fn node(def: Option<&str>, uses: &[&str]) -> Node {
        Node { defines: def.map(str::to_string), uses: uses.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn orders_definitions_before_uses() {
        let nodes = vec![node(None, &["a", "f"]), node(Some("f"), &["a"]), node(Some("a"), &[])];
        let order = dependency_order(&nodes).unwrap();
        let pos = |i| order.iter().position(|x| *x == i).unwrap();
        assert!(pos(2) < pos(1) && pos(1) < pos(0));
    }

    #[test]
    fn detects_cycles() {
        let nodes = vec![node(Some("a"), &["b"]), node(Some("b"), &["a"])];
        assert!(matches!(dependency_order(&nodes), Err(OrderError::Cycle(c)) if c.len() == 2));
        let self_ref = vec![node(Some("a"), &["a"])];
        assert!(dependency_order(&self_ref).is_err());
    }

    #[test]
    fn mathlive_chain_is_one_inequality_and_a_free_letter_is_a_slider() {
        for src in [r"0\le y\le x^2", r"0\leq y\leq x^{2}", r"x^2\ge y\ge 0", r"1<x^2+y^2\le4"] {
            assert!(matches!(kind(src), Kind::Inequality { rel: Rel::Lt, .. }), "{src}");
        }
        // `a\sin x` is a times sin: `a` is a slider candidate exactly like `k`
        for src in [r"y=a\sin\left(x\right)", r"y=a\cos x", r"y=a\ln x", r"y=b\sin x"] {
            let a = analyze(&parse(src).unwrap(), &BTreeSet::new());
            let names: Vec<&str> = a.slider_candidates.iter().map(String::as_str).collect();
            assert_eq!(names.len(), 1, "{src}: {names:?}");
            assert!(names[0] == "a" || names[0] == "b");
        }
        // bare x-expressions: a derivative is a one-variable scalar like `2x`
        let (d, two_x) = (kind(r"\frac{d}{dx}x^2"), kind("2x"));
        assert_eq!(std::mem::discriminant(&d), std::mem::discriminant(&two_x));
    }
}
