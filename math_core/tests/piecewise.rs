//! Piecewise values `{x<0: -x, x}`: parsing (text and LaTeX), evaluation, interval enclosures,
//! printing and meshing.
use math_core::analyze::{analyze, Kind};
use math_core::ast::{piece_parts, Expr, PIECE_FN};
use math_core::compile::{compile, Angle, Program};
use math_core::interval::Interval;
use math_core::mesh::sample_explicit;
use math_core::parse::parse;
use math_core::print::{to_latex, to_text};
use std::collections::BTreeSet;

fn prog(src: &str) -> Program {
    let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
    compile(&e, &["x"], Angle::Rad).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn ev(src: &str, x: f64) -> f64 {
    prog(src).eval(&[x])
}

#[test]
fn parses_branches_and_default() {
    let e = parse("{x<0: -x, x}").unwrap();
    let (pairs, default) = piece_parts(&e).unwrap();
    assert_eq!(pairs.len(), 1);
    assert!(default.is_some());
    let e = parse("{x<0: -x, x>=0: x}").unwrap();
    let (pairs, default) = piece_parts(&e).unwrap();
    assert_eq!(pairs.len(), 2);
    assert!(default.is_none());
    // a bare condition is "1 where it holds"
    let e = parse("{x>0}").unwrap();
    let (pairs, _) = piece_parts(&e).unwrap();
    assert_eq!(pairs[0].1, &Expr::Num(1.0));
    // chained condition
    let e = parse("{0<x<1: 5, 7}").unwrap();
    assert!(matches!(piece_parts(&e).unwrap().0[0].0, Expr::Call(..)));
}

#[test]
fn evaluates_like_desmos() {
    assert_eq!(ev("{x<0: -x, x}", -3.0), 3.0);
    assert_eq!(ev("{x<0: -x, x}", 2.0), 2.0);
    assert_eq!(ev("{x<0: -x, x>=0: x}", -1.5), 1.5);
    assert_eq!(ev("{x<0: -x, x>=0: x}", 0.0), 0.0);
    // no branch matches and no default: undefined
    assert!(ev("{x<0: 1}", 1.0).is_nan());
    assert!(ev("{x<0: 1, x>2: 2}", 1.0).is_nan());
    // first true branch wins
    assert_eq!(ev("{x<5: 1, x<10: 2, 3}", 1.0), 1.0);
    assert_eq!(ev("{x<5: 1, x<10: 2, 3}", 7.0), 2.0);
    assert_eq!(ev("{x<5: 1, x<10: 2, 3}", 70.0), 3.0);
    // chains and bare conditions
    assert_eq!(ev("{0<x<1: 5, 7}", 0.5), 5.0);
    assert_eq!(ev("{0<x<1: 5, 7}", 1.5), 7.0);
    assert_eq!(ev("{x>0}", 2.0), 1.0);
    assert!(ev("{x>0}", -2.0).is_nan());
    // equality
    assert_eq!(ev("{x=2: 10, 0}", 2.0), 10.0);
    // in an expression; an undefined branch value that is not taken does not matter
    assert_eq!(ev("2{x<0: -x, x}+1", -4.0), 9.0);
    assert_eq!(ev("{x>0: sqrt(x), 0}", -4.0), 0.0);
    // a NaN condition is false
    assert_eq!(ev("{sqrt(x)<1: 1, 2}", -1.0), 2.0);
}

#[test]
fn latex_forms_parse() {
    let a = parse(r"\left\{x<0:-x,x\right\}").unwrap();
    assert_eq!(a, parse("{x<0: -x, x}").unwrap());
    let b = parse(r"y=\left\{x<0:-x,x\ge 0:x\right\}").unwrap();
    assert!(matches!(b, Expr::Rel(..)));
    let c = parse(r"\lbrace x\le 0:1,2\rbrace").unwrap();
    assert_eq!(c, parse("{x<=0: 1, 2}").unwrap());
    let d = parse(r"\{x<0:-x,x\}").unwrap();
    assert_eq!(d, a);
    // a piece group nested in a fraction
    let e = parse(r"\frac{\left\{x<0:1,2\right\}}{3}").unwrap();
    assert_eq!(compile(&e, &["x"], Angle::Rad).unwrap().eval(&[-1.0]), 1.0 / 3.0);
}

#[test]
fn ranges_and_pieces_are_told_apart() {
    // a trailing {..} without a colon after a body is a range
    let a = analyze(&parse("y=x^2 {x>0}").unwrap(), &BTreeSet::new());
    assert_eq!(a.domain.len(), 1);
    // with a colon it is a piecewise value: no domain, an explicit curve
    let a = analyze(&parse("y={x<0: -x, x}").unwrap(), &BTreeSet::new());
    assert!(a.domain.is_empty());
    assert!(matches!(a.kind, Kind::ExplicitY { ref rhs } if matches!(rhs, Expr::Call(n, _) if n == PIECE_FN)));
    // both together: piecewise restricted by a range
    let a = analyze(&parse("y={x<0: -x, x} {x>-3}").unwrap(), &BTreeSet::new());
    assert_eq!(a.domain.len(), 1);
    assert!(matches!(a.kind, Kind::ExplicitY { .. }));
    // plain braces that are neither stay parentheses
    assert_eq!(parse("2{x}").unwrap(), parse("2(x)").unwrap());
    // a script is never a piece group
    assert_eq!(parse("x^{a<b}").is_ok(), parse("x^(a<b)").is_ok());
}

#[test]
fn bad_piecewise_is_a_parse_error() {
    for bad in ["{}", "{x<0: 1, 2, 3}", "{x<0: }", "{1, x<0: 2}"] {
        assert!(parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn prints_and_reparses() {
    for src in ["{x<0: -x, x}", "{x<0: -x, x>=0: x}", "{0<x<1: 5, 7}", "y = {x<0: 1}"] {
        let e = parse(src).unwrap();
        assert_eq!(parse(&to_text(&e)).unwrap(), e, "text {src}: {}", to_text(&e));
        assert_eq!(parse(&to_latex(&e)).unwrap(), e, "latex {src}: {}", to_latex(&e));
    }
}

#[test]
fn interval_enclosures_are_sound() {
    let p = prog("{x<0: -x, x}");
    // surely in the first branch
    let r = p.eval_interval(&[Interval::new(-3.0, -1.0)]);
    assert!(r.lo <= 1.0 && r.hi >= 3.0 && r.hi < 3.1);
    // straddles 0: contains both branches' values
    let r = p.eval_interval(&[Interval::new(-2.0, 1.0)]);
    for x in [-2.0, -1.0, -0.1, 0.0, 0.5, 1.0] {
        assert!(r.contains(p.eval(&[x])), "{x}");
    }
    // surely undefined: empty
    let q = prog("{x<0: 1}");
    assert!(q.eval_interval(&[Interval::new(1.0, 2.0)]).is_empty());
    assert!(!q.eval_interval(&[Interval::new(-1.0, 2.0)]).is_empty());
    // random-ish soundness sweep
    let p = prog("{x<-1: x^2, x<1: 0, x^3}");
    for k in 0..60 {
        let a = -4.0 + 0.13 * k as f64;
        let iv = Interval::new(a, a + 0.5);
        let r = p.eval_interval(&[iv]);
        for j in 0..=10 {
            let x = a + 0.05 * j as f64;
            assert!(r.contains(p.eval(&[x])), "{x} in {iv:?} -> {r:?}");
        }
    }
}

#[test]
fn explicit_samples_leave_a_gap_where_no_branch_matches() {
    let p = prog("{x<-1: x, x>1: x}");
    let lines = sample_explicit(&p, -3.0, 3.0, 400, (-5.0, 5.0));
    assert_eq!(lines.len(), 2, "{lines:?}");
    for l in &lines {
        for pt in l {
            assert!(pt[0] < -1.0 + 1e-6 || pt[0] > 1.0 - 1e-6, "{pt:?}");
        }
    }
    // absolute value as a piecewise: one continuous V
    let p = prog("{x<0: -x, x}");
    let lines = sample_explicit(&p, -3.0, 3.0, 400, (-5.0, 5.0));
    assert_eq!(lines.len(), 1);
}

#[test]
fn the_gpu_refuses_piecewise_with_a_clear_message() {
    let p = prog("{x<0: -x, x}");
    let err = math_core::wgsl::emit_module(&[("f", &p)]).unwrap_err();
    assert!(err.to_string().contains("piecewise"), "{err}");
}
