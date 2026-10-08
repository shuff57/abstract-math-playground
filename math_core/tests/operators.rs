//! `!=` / `≠` as a relation, `%` as modulo, and how `!` still means factorial.
use math_core::analyze::{analyze, Kind};
use math_core::ast::{Expr, Rel};
use math_core::compile::{compile, Angle};
use math_core::list::{eval_value, Bindings, Value};
use math_core::parse::parse;
use math_core::print::{to_latex, to_text};
use std::collections::BTreeSet;

fn num(src: &str, vars: &[(&str, f64)]) -> f64 {
    let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let names: Vec<&str> = vars.iter().map(|v| v.0).collect();
    let vals: Vec<f64> = vars.iter().map(|v| v.1).collect();
    compile(&e, &names, Angle::Rad).unwrap_or_else(|e| panic!("{src}: {e}")).eval(&vals)
}

#[test]
fn not_equal_is_a_relation() {
    for src in ["x!=3", "x≠3", r"x\ne 3", r"x\neq 3", "x != 3"] {
        let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert!(matches!(e, Expr::Rel(Rel::Ne, ..)), "{src}: {e:?}");
    }
    // n!=3 is n != 3, not (n!) = 3
    assert_eq!(parse("n!=3").unwrap(), parse("n≠3").unwrap());
}

#[test]
fn factorial_still_works() {
    assert_eq!(num("5!", &[]), 120.0);
    assert_eq!(num("n!", &[("n", 4.0)]), 24.0);
    assert_eq!(num("(n+1)!", &[("n", 3.0)]), 24.0);
    // the documented rule: `!` directly followed by `=` is the relation; a space or brackets keep the factorial
    assert!(matches!(parse("n! = 120").unwrap(), Expr::Rel(Rel::Eq, ..)));
    assert!(matches!(parse("(n!)=120").unwrap(), Expr::Rel(Rel::Eq, ..)));
    assert!(matches!(parse("factorial(n)=120").unwrap(), Expr::Rel(Rel::Eq, ..)));
    assert!(matches!(parse("5!=120").unwrap(), Expr::Rel(Rel::Ne, ..)));
    // double factorial and factorial before other operators are unchanged
    assert_eq!(num("3!!", &[]), 720.0);
    assert_eq!(num("2^3!", &[]), 64.0);
    assert_eq!(num("x^n/n!", &[("x", 2.0), ("n", 3.0)]), 8.0 / 6.0);
}

#[test]
fn not_equal_conditions_in_piecewise() {
    let f = |x: f64| num("{x!=2: 1, 0}", &[("x", x)]);
    assert_eq!(f(1.0), 1.0);
    assert_eq!(f(2.0), 0.0);
    // a hole in the graph: undefined at exactly 2
    let g = |x: f64| num("{x≠2: x}", &[("x", x)]);
    assert_eq!(g(3.0), 3.0);
    assert!(g(2.0).is_nan());
    // a NaN operand makes any comparison false, including !=
    assert_eq!(num("{sqrt(x)!=1: 1, 0}", &[("x", -4.0)]), 0.0);
}

#[test]
fn not_equal_in_a_list_mask() {
    let e = parse("L[L!=2]").unwrap();
    let mut b = Bindings::new();
    b.set("L", Value::List(vec![1.0, 2.0, 3.0, 2.0]));
    match eval_value(&e, &b).unwrap() {
        Value::List(l) => assert_eq!(l, vec![1.0, 3.0]),
        v => panic!("{v:?}"),
    }
}

#[test]
fn not_equal_is_not_a_region_or_a_chain_or_a_range() {
    let a = analyze(&parse("x!=3").unwrap(), &BTreeSet::new());
    assert!(!matches!(a.kind, Kind::Inequality { .. }));
    // as an item it is a clear compile error, not a silent nothing
    let e = parse("x!=3").unwrap();
    let err = compile(&e, &["x"], Angle::Rad).unwrap_err().to_string();
    assert!(err.contains("!=") && err.contains("piecewise"), "{err}");
    assert!(parse("1<x!=3").is_err());
    // a range cannot use it, but `{x!=0}` after a body is a piecewise condition: a hole at 0
    let hole = parse("y=x {x!=0}").unwrap();
    let Expr::Rel(_, _, rhs) = &hole else { panic!("{hole:?}") };
    let p = compile(rhs, &["x"], Angle::Rad).unwrap();
    assert_eq!(p.eval(&[2.0]), 2.0);
    assert!(p.eval(&[0.0]).is_nan());
}

#[test]
fn percent_is_mod() {
    for (a, b) in [(7.0, 3.0), (-7.0, 3.0), (7.0, -3.0), (5.5, 2.0), (0.0, 4.0), (3.0, 0.0)] {
        let vars = [("a", a), ("b", b)];
        let (p, m) = (num("a%b", &vars), num("mod(a,b)", &vars));
        assert!(p == m || (p.is_nan() && m.is_nan()), "{a} % {b}: {p} vs {m}");
    }
    assert_eq!(num("7%3", &[]), 1.0);
    assert_eq!(num("-7%3", &[]), 2.0); // (-7) % 3, like mod(-7, 3)
    // same level as * and /, left to right
    assert_eq!(num("2*7%4", &[]), 2.0); // (2*7) % 4
    assert_eq!(num("20%7%4", &[]), 2.0); // (20%7)%4 = 6%4
    assert_eq!(num("1+7%4", &[]), 4.0);
    assert_eq!(num("10/4%2", &[]), 0.5);
    assert_eq!(num(r"7\%3", &[]), 1.0);
    // as the condition of a piecewise: parity
    let parity = |x: f64| num("{x%2=0: 1, 0}", &[("x", x)]);
    assert_eq!((parity(4.0), parity(7.0)), (1.0, 0.0));
}

#[test]
fn printing_round_trips() {
    for src in ["x != 3", "{x != 2: 1, 0}", "a % b", "y = x % 3"] {
        let e = parse(src).unwrap();
        assert_eq!(parse(&to_text(&e)).unwrap(), e, "{src}: {}", to_text(&e));
        assert_eq!(parse(&to_latex(&e)).unwrap(), e, "{src}: {}", to_latex(&e));
    }
}
