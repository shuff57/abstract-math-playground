//! Expression printers: plain text (re-parseable) and LaTeX (for math fields and drag writeback).

use crate::ast::{rel_chain, BinOp, Expr, Rel};
use std::fmt;

fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Rel(..) => 10,
        Expr::Call(..) if is_chain(e) => 10,
        Expr::Bin(BinOp::Add | BinOp::Sub, ..) => 20,
        Expr::Bin(BinOp::Mul | BinOp::Div, ..) => 30,
        Expr::Neg(_) => 35,
        Expr::Num(v) if *v < 0.0 => 35,
        Expr::Bin(BinOp::Pow, ..) => 40,
        _ => 100,
    }
}

/// A chained comparison whose parts share their middle operands, so it prints as `a < b <= c`.
fn is_chain(e: &Expr) -> bool {
    chain_operands(e).is_some()
}

/// `[a, b, c]` and `[<, <=]` for `and(a<b, b<=c)`.
fn chain_operands(e: &Expr) -> Option<(Vec<&Expr>, Vec<Rel>)> {
    let parts = rel_chain(e)?;
    let mut ops: Vec<&Expr> = Vec::new();
    let mut rels = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        let Expr::Rel(r, a, b) = p else { return None };
        if i == 0 {
            ops.push(a);
        } else if **a != **ops.last()? {
            return None;
        }
        ops.push(b);
        rels.push(*r);
    }
    Some((ops, rels))
}

/// Whether `e` prints as a single token or a bracketed form, so a postfix `!` can follow it.
fn factorial_atom(e: &Expr) -> bool {
    match e {
        Expr::Num(v) => *v >= 0.0,
        Expr::Var(_) | Expr::List(_) | Expr::Tuple(_) => true,
        Expr::Call(name, args) => {
            !is_chain(e)
                && !matches!(name.as_str(), "regress" | "deriv")
                && !(name == "factorial" && args.len() != 1)
        }
        _ => false,
    }
}

fn num(v: f64) -> String {
    format!("{v}")
}

fn rel_text(r: Rel) -> &'static str {
    match r {
        Rel::Eq => "=",
        Rel::Lt => "<",
        Rel::Le => "<=",
        Rel::Gt => ">",
        Rel::Ge => ">=",
    }
}

/// Whether the left/right operand of `op` needs parentheses to keep the tree shape.
fn left_parens(op: BinOp, l: &Expr) -> bool {
    match op {
        BinOp::Pow => prec(l) <= 40,
        BinOp::Add | BinOp::Sub => prec(l) < 20,
        BinOp::Mul | BinOp::Div => prec(l) < 30,
    }
}

fn right_parens(op: BinOp, r: &Expr) -> bool {
    match op {
        BinOp::Pow => prec(r) < 40,
        BinOp::Add | BinOp::Sub => prec(r) <= 20,
        BinOp::Mul | BinOp::Div => prec(r) <= 30,
    }
}

/// Whether `e` can be followed directly by `[idx]` without parentheses.
fn indexable(e: &Expr) -> bool {
    matches!(e, Expr::Var(_) | Expr::List(_) | Expr::Call(..))
}

/// The list-syntax call forms, recognised by shape so other calls print normally.
enum ListForm<'a> {
    Range(&'a [Expr]),
    Index(&'a Expr, &'a Expr),
    For(&'a Expr, &'a str, &'a Expr),
}

fn list_form<'a>(name: &str, args: &'a [Expr]) -> Option<ListForm<'a>> {
    match (name, args) {
        ("range", [_, _]) | ("range", [_, _, _]) => Some(ListForm::Range(args)),
        ("index", [l, i]) => Some(ListForm::Index(l, i)),
        ("for", [body, Expr::Var(v), src]) => Some(ListForm::For(body, v, src)),
        _ => None,
    }
}

fn range_inner(args: &[Expr], f: fn(&Expr) -> String, dots: &str) -> String {
    match args {
        [a, b] => format!("{}{dots}{}", f(a), f(b)),
        [a, b, c] => format!("{}, {}{dots}{}", f(a), f(b), f(c)),
        _ => unreachable!(),
    }
}

/// `regress(lhs, rhs)` is the parser's encoding of `lhs ~ rhs`.
fn regress_form<'a>(name: &str, args: &'a [Expr]) -> Option<(&'a Expr, &'a Expr)> {
    match (name, args) {
        ("regress", [l, r]) => Some((l, r)),
        _ => None,
    }
}

/// `deriv(f, x)` with a one-letter (or theta) variable prints as `\frac{d}{dx}`.
fn deriv_form<'a>(name: &str, args: &'a [Expr]) -> Option<(&'a Expr, &'a str)> {
    match (name, args) {
        ("deriv", [f, Expr::Var(v)]) if v == "theta" || (v.chars().count() == 1 && v.chars().all(|c| c.is_ascii_alphabetic())) => {
            Some((f, v))
        }
        _ => None,
    }
}

/// `{c1: v1, c2: v2, d}` for a piecewise node, through `f` for the parts.
fn piece_body(args: &[Expr], f: impl Fn(&Expr) -> String, colon: &str, sep: &str) -> String {
    let mut parts: Vec<String> = args.chunks_exact(2).map(|c| format!("{}{colon}{}", f(&c[0]), f(&c[1]))).collect();
    if args.len() % 2 == 1 {
        parts.push(f(&args[args.len() - 1]));
    }
    parts.join(sep)
}

pub fn to_text(e: &Expr) -> String {
    match e {
        Expr::Call(name, args) if name == crate::ast::PIECE_FN && !args.is_empty() => {
            format!("{{{}}}", piece_body(args, to_text, ": ", ", "))
        }
        Expr::Call(name, args) if name == crate::ast::DOMAIN_FN && args.len() >= 2 => {
            let clauses: Vec<String> = args[1..].iter().map(to_text).collect();
            format!("{} {{{}}}", to_text(&args[0]), clauses.join(", "))
        }
        Expr::Call(name, args) if name == "factorial" && args.len() == 1 => {
            let inner = to_text(&args[0]);
            if factorial_atom(&args[0]) {
                format!("{inner}!")
            } else {
                format!("({inner})!")
            }
        }
        Expr::Call(..) if is_chain(e) => {
            let (ops, rels) = chain_operands(e).unwrap();
            let mut out = to_text(ops[0]);
            for (r, o) in rels.iter().zip(&ops[1..]) {
                out.push_str(&format!(" {} {}", rel_text(*r), to_text(o)));
            }
            out
        }
        Expr::Call(name, args) if regress_form(name, args).is_some() => {
            let (l, r) = regress_form(name, args).unwrap();
            format!("{} ~ {}", to_text(l), to_text(r))
        }
        Expr::Call(name, args) if list_form(name, args).is_some() => match list_form(name, args).unwrap() {
            ListForm::Range(a) => format!("[{}]", range_inner(a, to_text, "...")),
            ListForm::Index(l, i) => {
                let ls = if indexable(l) { to_text(l) } else { format!("({})", to_text(l)) };
                let is = match i {
                    Expr::Call(n, a) if n == "range" && (a.len() == 2 || a.len() == 3) => {
                        range_inner(a, to_text, "...")
                    }
                    other => to_text(other),
                };
                format!("{ls}[{is}]")
            }
            ListForm::For(body, v, src) => format!("[{} for {v}={}]", to_text(body), to_text(src)),
        },
        Expr::Num(v) => num(*v),
        Expr::Var(n) => n.clone(),
        Expr::Neg(a) => {
            let inner = to_text(a);
            if prec(a) < 35 {
                format!("-({inner})")
            } else {
                format!("-{inner}")
            }
        }
        Expr::Bin(op, l, r) => {
            let ls = to_text(l);
            let rs = to_text(r);
            let ls = if left_parens(*op, l) { format!("({ls})") } else { ls };
            let rs = if right_parens(*op, r) { format!("({rs})") } else { rs };
            match op {
                BinOp::Add => format!("{ls} + {rs}"),
                BinOp::Sub => format!("{ls} - {rs}"),
                BinOp::Mul => format!("{ls} * {rs}"),
                BinOp::Div => format!("{ls} / {rs}"),
                BinOp::Pow => format!("{ls}^{rs}"),
            }
        }
        Expr::Call(name, args) => {
            format!("{name}({})", args.iter().map(to_text).collect::<Vec<_>>().join(", "))
        }
        Expr::Tuple(items) => format!("({})", items.iter().map(to_text).collect::<Vec<_>>().join(", ")),
        Expr::List(items) => format!("[{}]", items.iter().map(to_text).collect::<Vec<_>>().join(", ")),
        Expr::Rel(r, a, b) => format!("{} {} {}", to_text(a), rel_text(*r), to_text(b)),
    }
}

fn latex_name(name: &str) -> String {
    match name {
        "pi" => "\\pi".into(),
        "tau" => "\\tau".into(),
        "theta" => "\\theta".into(),
        n if n.contains('_') => {
            let (base, sub) = n.split_once('_').unwrap();
            format!("{base}_{{{sub}}}")
        }
        n => n.to_string(),
    }
}

fn latex_func(name: &str) -> String {
    match name {
        "sin" | "cos" | "tan" | "sec" | "csc" | "cot" | "sinh" | "cosh" | "tanh" | "exp" | "ln"
        | "log" | "min" | "max" => format!("\\{name}"),
        "asin" => "\\arcsin".into(),
        "acos" => "\\arccos".into(),
        "atan" => "\\arctan".into(),
        other => format!("\\operatorname{{{other}}}"),
    }
}

fn rel_latex(r: Rel) -> &'static str {
    match r {
        Rel::Eq => "=",
        Rel::Lt => "<",
        Rel::Le => "\\le ",
        Rel::Gt => ">",
        Rel::Ge => "\\ge ",
    }
}

/// `int(f,t,a,b)`, `sum(f,n,a,b)`, `prod(f,n,a,b)` with a plain variable name.
fn big_op_form<'a>(name: &str, args: &'a [Expr]) -> Option<(&'a str, &'a [Expr])> {
    match (name, args) {
        ("int" | "sum" | "prod", [_, Expr::Var(v), _, _]) if !crate::ast::is_builtin_func(v) => {
            Some((name_static(name), args))
        }
        _ => None,
    }
}

fn name_static(n: &str) -> &'static str {
    match n {
        "int" => "int",
        "sum" => "sum",
        _ => "prod",
    }
}

fn is_big_op(e: &Expr) -> bool {
    matches!(e, Expr::Call(n, a) if big_op_form(n, a).is_some())
}

pub fn to_latex(e: &Expr) -> String {
    match e {
        Expr::Call(name, args) if name == crate::ast::PIECE_FN && !args.is_empty() => {
            format!("\\left\\{{{}\\right\\}}", piece_body(args, to_latex, ":", ",\\,"))
        }
        Expr::Call(name, args) if name == crate::ast::DOMAIN_FN && args.len() >= 2 => {
            let clauses: Vec<String> = args[1..].iter().map(to_latex).collect();
            format!("{}\\left\\{{{}\\right\\}}", to_latex(&args[0]), clauses.join(",\\,"))
        }
        Expr::Call(name, args) if big_op_form(name, args).is_some() => {
            let (kind, a) = big_op_form(name, args).unwrap();
            let Expr::Var(v) = &a[1] else { unreachable!() };
            let (lo, hi) = (to_latex(&a[2]), to_latex(&a[3]));
            if kind == "int" {
                format!("\\int_{{{lo}}}^{{{hi}}} {}\\,d{}", to_latex(&a[0]), latex_name(v))
            } else {
                let body = if prec(&a[0]) < 30 { format!("\\left({}\\right)", to_latex(&a[0])) } else { to_latex(&a[0]) };
                format!("\\{kind}_{{{}={lo}}}^{{{hi}}} {body}", latex_name(v))
            }
        }
        Expr::Call(name, args) if name == "factorial" && args.len() == 1 => {
            let inner = to_latex(&args[0]);
            if factorial_atom(&args[0]) {
                format!("{inner}!")
            } else {
                format!("\\left({inner}\\right)!")
            }
        }
        Expr::Call(name, args) if name == "nCr" && args.len() == 2 => {
            format!("\\binom{{{}}}{{{}}}", to_latex(&args[0]), to_latex(&args[1]))
        }
        Expr::Call(..) if is_chain(e) => {
            let (ops, rels) = chain_operands(e).unwrap();
            let mut out = to_latex(ops[0]);
            for (r, o) in rels.iter().zip(&ops[1..]) {
                out.push_str(rel_latex(*r));
                out.push_str(&to_latex(o));
            }
            out
        }
        Expr::Call(name, args) if regress_form(name, args).is_some() => {
            let (l, r) = regress_form(name, args).unwrap();
            format!("{}\\sim {}", to_latex(l), to_latex(r))
        }
        Expr::Call(name, args) if deriv_form(name, args).is_some() => {
            let (f, v) = deriv_form(name, args).unwrap();
            let v = if v == "theta" { "\\theta".to_string() } else { v.to_string() };
            format!("\\frac{{d}}{{d{v}}}\\left({}\\right)", to_latex(f))
        }
        Expr::Call(name, args) if list_form(name, args).is_some() => match list_form(name, args).unwrap() {
            ListForm::Range(a) => format!("\\left[{}\\right]", range_inner(a, to_latex, "\\ldots ")),
            ListForm::Index(l, i) => {
                let ls = if indexable(l) { to_latex(l) } else { format!("\\left({}\\right)", to_latex(l)) };
                let is = match i {
                    Expr::Call(n, a) if n == "range" && (a.len() == 2 || a.len() == 3) => {
                        range_inner(a, to_latex, "\\ldots ")
                    }
                    other => to_latex(other),
                };
                format!("{ls}\\left[{is}\\right]")
            }
            ListForm::For(body, v, src) => format!(
                "\\left[{} \\operatorname{{for}} {}={}\\right]",
                to_latex(body),
                latex_name(v),
                to_latex(src)
            ),
        },
        Expr::Num(v) => num(*v),
        Expr::Var(n) => latex_name(n),
        Expr::Neg(a) => {
            let inner = to_latex(a);
            if prec(a) < 35 {
                format!("-\\left({inner}\\right)")
            } else {
                format!("-{inner}")
            }
        }
        Expr::Bin(BinOp::Div, l, r) => format!("\\frac{{{}}}{{{}}}", to_latex(l), to_latex(r)),
        Expr::Bin(op, l, r) => {
            let ls = to_latex(l);
            let rs = to_latex(r);
            // A sum/product body would swallow what follows; `\int ... dx^2` is ambiguous too.
            let wrap = left_parens(*op, l)
                || (is_big_op(l) && matches!(op, BinOp::Pow | BinOp::Mul | BinOp::Div));
            let ls = if wrap { format!("\\left({ls}\\right)") } else { ls };
            match op {
                BinOp::Pow => format!("{ls}^{{{rs}}}"),
                _ => {
                    let rs = if right_parens(*op, r) { format!("\\left({rs}\\right)") } else { rs };
                    match op {
                        BinOp::Add => format!("{ls}+{rs}"),
                        BinOp::Sub => format!("{ls}-{rs}"),
                        _ => format!("{ls}\\cdot {rs}"),
                    }
                }
            }
        }
        Expr::Call(name, args) if name == "sqrt" && args.len() == 1 => {
            format!("\\sqrt{{{}}}", to_latex(&args[0]))
        }
        Expr::Call(name, args) if name == "abs" && args.len() == 1 => {
            format!("\\left|{}\\right|", to_latex(&args[0]))
        }
        Expr::Call(name, args) => format!(
            "{}\\left({}\\right)",
            latex_func(name),
            args.iter().map(to_latex).collect::<Vec<_>>().join(",")
        ),
        Expr::Tuple(items) => {
            format!("\\left({}\\right)", items.iter().map(to_latex).collect::<Vec<_>>().join(","))
        }
        Expr::List(items) => {
            format!("\\left[{}\\right]", items.iter().map(to_latex).collect::<Vec<_>>().join(","))
        }
        Expr::Rel(r, a, b) => format!("{}{}{}", to_latex(a), rel_latex(*r), to_latex(b)),
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&to_text(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    const SAMPLES: &[&str] = &[
        "1+2*3",
        "(1+2)*3",
        "a-(b-c)",
        "a-b-c",
        "2^3^2",
        "(2^3)^2",
        "-x^2",
        "(-x)^2",
        "2^-x",
        "x/(y+1)",
        "1/2x",
        "sin(x)^2+cos(x)^2",
        "sqrt(x^2+y^2)",
        "|x-1|+2",
        "y=x^2-3x+1",
        "x<=sin(y)",
        "(1,2)",
        "[1,2,3]",
        "max(a,b)+atan(x)",
        "2pi r",
        "x_1+x_2",
        "-(a+b)*c",
        "a*-b",
        "[1...10]",
        "[1,3...11]",
        "[10,8...0]",
        "[-2.5...x]",
        "L[3]",
        "L[2...4]",
        "L[1,3...9]",
        "L[[1,3]]",
        "L[L>3]",
        "L[L^2<10]",
        "L[i+1]",
        "(a+b)[1]",
        "sort(L)[1]",
        "M[1][2]",
        "[n^2 for n=[1...5]]",
        "[2k+1 for k=L[L>0]]",
        "[[1...3][j] for j=[1...3]]",
        "mean([1...10])+stdev(L)",
        "normalcdf(-1,1,0,1)",
        "tcdf(-10,x,5)",
        "sort(L,K)",
        "2[1,2,3]",
        "n!",
        "(n+1)!",
        "5!",
        "n!!",
        "(x^2)!",
        "(-3)!",
        "sin(x)!",
        "x^n!",
        "n!^2",
        "-n!",
        "2^3!",
        "(a+b)!/(a!b!)",
        "sum(x^n/n!,n,0,6)",
        "[n! for n=[1...5]]",
        "nCr(n,2)+nPr(5,k)",
        "0<=y<=x^2",
        "a<x<b",
        "1<x^2+y^2<=4",
        "1>x<=2",
        "0<x<y<=3",
        "(0<x<1)+1",
        "int(x^2, x, 0, 2)",
        "sum(x^n/n!, n, 0, 6)",
        "prod(k, k, 1, 4)",
        "sum(n, n, 1, 3)+1",
        "sum(n+1, n, 1, 3)*x",
        "sum(n, n, 1, 3)^2",
        "2*int(sin(x), x, 0, pi)+3",
        "int(int(x*y, x, 0, 2), y, 0, 1)",
        "int(sin(t), t, 0, x)",
        "sum(int(x, x, 0, n), n, 1, 3)",
        "a*sin(x)",
        "(t^2, 2t) {-3<=t<=3}",
        "(cos(t), sin(t), t/4) {0<=t<=12pi}",
        "r=theta {0<=theta<=6pi}",
        "(u, v, u*v) {0<=u<=1, 0<=v<=a}",
        "(2sin(t), 1) {t>=0}",
    ];

    #[test]
    fn text_roundtrip_is_a_fixpoint() {
        for s in SAMPLES {
            let e = parse(s).unwrap();
            let printed = to_text(&e);
            let reparsed = parse(&printed).unwrap_or_else(|er| panic!("{s} -> {printed}: {er}"));
            assert_eq!(e, reparsed, "{s} -> {printed}");
        }
    }

    #[test]
    fn latex_roundtrip_is_a_fixpoint() {
        for s in SAMPLES {
            let e = parse(s).unwrap();
            let latex = to_latex(&e);
            let reparsed = parse(&latex).unwrap_or_else(|er| panic!("{s} -> {latex}: {er}"));
            assert_eq!(to_text(&e), to_text(&reparsed), "{s} -> {latex}");
        }
    }

    #[test]
    fn list_forms_print_to_surface_syntax() {
        let t = |s: &str| to_text(&parse(s).unwrap());
        assert_eq!(t("[1...10]"), "[1...10]");
        assert_eq!(t("[1,3...11]"), "[1, 3...11]");
        assert_eq!(t("L[2...4]"), "L[2...4]");
        assert_eq!(t("L[L>3]"), "L[L > 3]");
        assert_eq!(t("[n^2 for n=L]"), "[n^2 for n=L]");
        assert_eq!(to_latex(&parse("[1...10]").unwrap()), "\\left[1\\ldots 10\\right]");
        assert_eq!(to_latex(&parse("L[L>3]").unwrap()), "L\\left[L>3\\right]");
        assert_eq!(
            to_latex(&parse("[n for n=L]").unwrap()),
            "\\left[n \\operatorname{for} n=L\\right]"
        );
        // malformed shapes fall back to ordinary call printing instead of panicking
        let weird = Expr::call("range", vec![Expr::num(1.0)]);
        assert_eq!(to_text(&weird), "range(1)");
        let weird = Expr::call("for", vec![Expr::num(1.0), Expr::num(2.0), Expr::num(3.0)]);
        assert_eq!(to_text(&weird), "for(1, 2, 3)");
        let _ = to_latex(&weird);
    }

    #[test]
    fn factorial_and_chain_shapes() {
        let t = |s: &str| to_text(&parse(s).unwrap());
        let l = |s: &str| to_latex(&parse(s).unwrap());
        assert_eq!(t("n!"), "n!");
        assert_eq!(t("factorial(n+1)"), "(n + 1)!");
        assert_eq!(t("2^3!"), "2^3!");
        assert_eq!(t("(2^3)!"), "(2^3)!");
        assert_eq!(l("n!"), "n!");
        assert_eq!(l("(n+1)!"), "\\left(n+1\\right)!");
        assert_eq!(l("nCr(n,k)"), "\\binom{n}{k}");
        assert_eq!(t("0<=y<=x^2"), "0 <= y <= x^2");
        assert_eq!(l("0<=y<=x^2"), "0\\le y\\le x^{2}");
        assert_eq!(t("1 < x^2+y^2 <= 4"), "1 < x^2 + y^2 <= 4");
        // a hand-built `and` whose parts do not chain falls back to an ordinary call
        let odd = Expr::call(
            "and",
            vec![parse("x<1").unwrap(), parse("y<2").unwrap()],
        );
        assert_eq!(to_text(&odd), "and(x < 1, y < 2)");
    }

    #[test]
    fn latex_shapes() {
        assert_eq!(to_latex(&parse("1/2").unwrap()), "\\frac{1}{2}");
        assert_eq!(to_latex(&parse("x^2").unwrap()), "x^{2}");
        assert_eq!(to_latex(&parse("sqrt(x)").unwrap()), "\\sqrt{x}");
        assert_eq!(to_latex(&parse("pi").unwrap()), "\\pi");
    }

    #[test]
    fn big_operators_print_as_mathlive_forms() {
        let l = |s: &str| to_latex(&parse(s).unwrap());
        assert_eq!(l("int(x^2, x, 0, 2)"), r"\int_{0}^{2} x^{2}\,dx");
        assert_eq!(l("sum(x^n/n!, n, 0, 6)"), r"\sum_{n=0}^{6} \frac{x^{n}}{n!}");
        assert_eq!(l("prod(k, k, 1, 4)"), r"\prod_{k=1}^{4} k");
        assert_eq!(l("sum(n+1, n, 1, 3)"), r"\sum_{n=1}^{3} \left(n+1\right)");
        assert_eq!(l("int(sin(t), t, 0, theta)"), r"\int_{0}^{\theta} \sin\left(t\right)\,dt");
        // every printed form parses back to the same tree
        for s in ["int(x^2, x, 0, 2)", "sum(x^n/n!, n, 0, 6)", "prod(k, k, 1, 4)", "sum(n+1, n, 1, 3)*x"] {
            let e = parse(s).unwrap();
            assert_eq!(parse(&to_latex(&e)).unwrap(), e, "{s}");
        }
    }
}
