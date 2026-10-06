//! Parametric forms: range restrictions and axis-equation pairs.
//!
//! * A trailing `{ ... }` group of comparisons restricts the parameter of a parametric curve,
//!   a polar curve or a parametric surface: `(t^2, 2t) {-3<=t<=3}`, `r=theta {0<=theta<=6pi}`,
//!   `(u, v, u v) {0<=u<=1, 0<=v<=1}`. The parser desugars it to
//!   `domain(body, clause, ...)` ([`crate::ast::DOMAIN_FN`]); [`unwrap_domain`] takes it apart
//!   again and [`analyze`](crate::analyze::analyze) reports the ranges next to the item's kind.
//! * `x=3cos(t), y=2sin(t)` (and `x=.., y=.., z=..`) in one item is the tuple of the right
//!   sides, i.e. the same parametric curve as `(3cos(t), 2sin(t))`; [`axis_equation`] lets a
//!   document builder fuse two or three such items too.

use crate::analyze::Kind;
use crate::ast::{Expr, Rel, DOMAIN_FN};

/// A range for one parameter. A missing end takes the item's default (see
/// [`resolve_range`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub var: String,
    pub lo: Option<Expr>,
    pub hi: Option<Expr>,
}

/// Names that are taken as the range variable of a comparison when both sides are names.
const KNOWN_VARS: &[&str] = &["t", "theta", "u", "v", "x", "y", "z", "r"];

/// If `src` ends in a `{ ... }` group of comparisons (plain `{}` or LaTeX `\left\{ \right\}`),
/// returns the text before it and the text inside. Script groups (`x^{2}`) and groups without
/// a comparison are not ranges.
pub fn split_domain_src(src: &str) -> Option<(&str, &str)> {
    if !src.contains('{') && !src.contains("brace") {
        return None;
    }
    let b = src.as_bytes();
    let mut i = 0;
    let mut depth = 0usize;
    let mut prev_sig = b' ';
    // pending `\left` / `\right` positions directly before a brace
    let mut left_at: Option<usize> = None;
    let mut right_at: Option<usize> = None;
    // (start of group incl. `\left`, content start, content end, end after the close)
    let mut open: Option<(usize, usize, bool)> = None;
    let mut last: Option<(usize, usize, usize, usize)> = None;
    while i < b.len() {
        let c = b[i];
        if c == b'\\' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j == i + 1 {
                // escaped punctuation: `\{` and `\}` are braces
                if j < b.len() && (b[j] == b'{' || b[j] == b'}') {
                    let open_brace = b[j] == b'{';
                    let end = j + 1;
                    if open_brace {
                        if depth == 0 {
                            let start = left_at.take().unwrap_or(i);
                            open = Some((start, end, prev_sig == b'^' || prev_sig == b'_'));
                        }
                        depth += 1;
                    } else if depth > 0 {
                        depth -= 1;
                        if depth == 0 {
                            if let Some((s, cs, script)) = open.take() {
                                if !script {
                                    last = Some((s, cs, right_at.take().unwrap_or(i), end));
                                }
                            }
                        }
                    }
                    left_at = None;
                    right_at = None;
                    prev_sig = b'}';
                    i = end;
                    continue;
                }
                i = (j + 1).min(b.len());
                continue;
            }
            match &src[i + 1..j] {
                "left" => left_at = Some(i),
                "right" => right_at = Some(i),
                // MathLive spells a typed brace `\lbrace` / `\rbrace`
                "lbrace" => {
                    if depth == 0 {
                        open = Some((left_at.take().unwrap_or(i), j, prev_sig == b'^' || prev_sig == b'_'));
                    }
                    depth += 1;
                    left_at = None;
                    right_at = None;
                    prev_sig = b'}';
                }
                "rbrace" => {
                    if depth > 0 {
                        depth -= 1;
                        if depth == 0 {
                            if let Some((st, cs, script)) = open.take() {
                                if !script {
                                    last = Some((st, cs, right_at.take().unwrap_or(i), j));
                                }
                            }
                        }
                    }
                    left_at = None;
                    right_at = None;
                    prev_sig = b'}';
                }
                _ => {
                    left_at = None;
                    right_at = None;
                    prev_sig = b'a';
                }
            }
            if matches!(&src[i + 1..j], "left" | "right") {
                prev_sig = b'a';
            }
            i = j;
            continue;
        }
        match c {
            b'{' => {
                if depth == 0 {
                    let start = left_at.take().unwrap_or(i);
                    open = Some((start, i + 1, prev_sig == b'^' || prev_sig == b'_'));
                }
                depth += 1;
            }
            b'}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some((s, cs, script)) = open.take() {
                            if !script {
                                last = Some((s, cs, right_at.take().unwrap_or(i), i + 1));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        if !c.is_ascii_whitespace() {
            prev_sig = c;
            left_at = None;
            right_at = None;
        }
        i += 1;
    }
    let (start, cs, ce, end) = last?;
    if depth != 0 || !src[end..].trim().is_empty() || cs > ce {
        return None;
    }
    let inner = &src[cs..ce];
    let relational = inner.contains('<')
        || inner.contains('>')
        || inner.contains("\\le")
        || inner.contains("\\ge")
        || inner.contains('\u{2264}')
        || inner.contains('\u{2265}');
    let body = src[..start].trim_end();
    (relational && !body.is_empty()).then_some((body, inner))
}

/// Splits `text` at commas and semicolons that are outside every bracket.
pub fn split_top_level(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut from) = (0i32, 0usize);
    let mut prev_backslash = false;
    for (i, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' | ';' if depth == 0 && !prev_backslash => {
                out.push(text[from..i].trim());
                from = i + 1;
            }
            _ => {}
        }
        prev_backslash = c == '\\';
    }
    out.push(text[from..].trim());
    out
}

fn is_known(e: &Expr) -> bool {
    matches!(e, Expr::Var(n) if KNOWN_VARS.contains(&n.as_str()))
}

/// The ranges a list of comparison clauses (`-3<=t<=3`, `t>=0`, `u<=pi`) describe, merged per
/// variable. Errors name the clause that is not a range.
pub fn ranges_of(clauses: &[Expr]) -> Result<Vec<Range>, String> {
    let mut out: Vec<Range> = Vec::new();
    for clause in clauses {
        let parts: Vec<&Expr> = match crate::ast::rel_chain(clause) {
            Some(p) => p.iter().collect(),
            None => vec![clause],
        };
        for part in parts {
            let Expr::Rel(rel, l, r) = part else {
                return Err("a range must be a comparison such as -3<=t<=3".into());
            };
            if *rel == Rel::Eq {
                return Err("a range uses <= or >=, not =".into());
            }
            // which side is the parameter
            let var_on_left = match (&**l, &**r) {
                (Expr::Var(_), Expr::Var(_)) if is_known(l) || !is_known(r) => true,
                (Expr::Var(_), Expr::Var(_)) => false,
                (Expr::Var(_), _) => true,
                (_, Expr::Var(_)) => false,
                _ => return Err("a range needs the parameter on one side, e.g. -3<=t<=3".into()),
            };
            let (name, bound, var_is_small) = if var_on_left {
                let Expr::Var(n) = &**l else { unreachable!() };
                (n.clone(), (**r).clone(), matches!(rel, Rel::Lt | Rel::Le))
            } else {
                let Expr::Var(n) = &**r else { unreachable!() };
                (n.clone(), (**l).clone(), matches!(rel, Rel::Gt | Rel::Ge))
            };
            if bound.contains_var(&name) {
                return Err(format!("the range of {name} cannot depend on {name}"));
            }
            // `t <= b` bounds from above; `a <= t` from below
            let idx = match out.iter().position(|x| x.var == name) {
                Some(i) => i,
                None => {
                    out.push(Range { var: name, lo: None, hi: None });
                    out.len() - 1
                }
            };
            if var_is_small {
                out[idx].hi = Some(bound);
            } else {
                out[idx].lo = Some(bound);
            }
        }
    }
    Ok(out)
}

/// Splits a parsed expression into its body and the ranges of a trailing `{...}` (none when it
/// has no range). Malformed clauses cannot occur: the parser validated them.
pub fn unwrap_domain(e: &Expr) -> (&Expr, Vec<Range>) {
    match e {
        Expr::Call(n, args) if n == DOMAIN_FN && args.len() >= 2 => {
            (&args[0], ranges_of(&args[1..]).unwrap_or_default())
        }
        _ => (e, Vec::new()),
    }
}

/// `x = f(t)`, `y = g(t)` or `z = h(t)` where the right side uses `t` (or `u`/`v`) and none of
/// x, y, z: the axis index and the right side. Such a row is half of a parametric curve.
pub fn axis_equation(kind: &Kind) -> Option<(usize, &Expr)> {
    let (axis, rhs) = match kind {
        Kind::ExplicitX { rhs } => (0, rhs),
        Kind::ExplicitY { rhs } => (1, rhs),
        Kind::ExplicitZ { rhs } => (2, rhs),
        _ => return None,
    };
    let uses_param = ["t", "u", "v"].iter().any(|p| rhs.contains_var(p));
    let uses_space = ["x", "y", "z"].iter().any(|p| rhs.contains_var(p));
    (uses_param && !uses_space).then_some((axis, rhs))
}

/// The curve bounds `[lo, hi]` for a parameter: the item's own range where given, filling a
/// missing end from `default` (`[0, 2pi]` and friends). A single given end moves the other
/// default so the interval stays non-empty (`t>=5` is `[5, 5 + span]`, `t<=-1` is
/// `[-1 - span, -1]`). `lo`/`hi` are the evaluated bounds.
pub fn resolve_range(lo: Option<f64>, hi: Option<f64>, default: [f64; 2]) -> Result<[f64; 2], String> {
    let span = default[1] - default[0];
    let (a, b) = match (lo, hi) {
        (None, None) => (default[0], default[1]),
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, if a >= default[1] { a + span } else { default[1] }),
        (None, Some(b)) => (if b <= default[0] { b - span } else { default[0] }, b),
    };
    if !(a.is_finite() && b.is_finite()) {
        return Err("the range is not a finite number".into());
    }
    if b <= a {
        return Err(format!("empty range: {a} is not below {b}"));
    }
    Ok([a, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn splits_trailing_domain_only() {
        assert_eq!(split_domain_src("(t,t) {-3<=t<=3}"), Some(("(t,t)", "-3<=t<=3")));
        assert_eq!(split_domain_src("(t,t){t>0}"), Some(("(t,t)", "t>0")));
        assert_eq!(split_domain_src(r"(t,t)\left\{-3\le t\le 3\right\}"), Some(("(t,t)", r"-3\le t\le 3")));
        assert_eq!(split_domain_src(r"(t,t)\{t\ge 0\}"), Some(("(t,t)", r"t\ge 0")));
        assert_eq!(split_domain_src(r"(t,t)\lbrace -3\le t\le 3\rbrace"), Some(("(t,t)", r" -3\le t\le 3")));
        assert_eq!(split_domain_src(r"(t,t)\left\lbrace t\ge 0\right\rbrace"), Some(("(t,t)", r" t\ge 0")));
        // not ranges: scripts, no comparison, text after the group, nothing before it
        assert_eq!(split_domain_src("x^{2}"), None);
        assert_eq!(split_domain_src("x^{a<b}"), None);
        assert_eq!(split_domain_src("2{x}"), None);
        assert_eq!(split_domain_src("(t,t) {t>0} + 1"), None);
        assert_eq!(split_domain_src("{t>0}"), None);
        assert_eq!(split_domain_src("(t,t) {t>0"), None);
    }

    #[test]
    fn mathlive_braces_parse_as_ranges() {
        let a = parse(r"\left(t^2,2t\right)\lbrace -3\le t\le 3\rbrace").unwrap();
        assert_eq!(unwrap_domain(&a).1.len(), 1);
        let b = parse(r"(vcos(u),vsin(u),0.5u)\lbrace-6\le u\le6,-5\le v\le5\rbrace").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(unwrap_domain(&b).1.len(), 2);
    }

    #[test]
    fn top_level_split_respects_brackets() {
        assert_eq!(split_top_level("x=1, y=f(a,b); z=[1,2]"), vec!["x=1", "y=f(a,b)", "z=[1,2]"]);
        assert_eq!(split_top_level("a"), vec!["a"]);
    }

    #[test]
    fn ranges_of_chains_and_halves() {
        let e = parse("(t,t) {-3<=t<=2, u>=1, v<pi}").unwrap();
        let (body, r) = unwrap_domain(&e);
        assert!(matches!(body, Expr::Tuple(_)));
        assert_eq!(r.len(), 3);
        assert_eq!((r[0].var.as_str(), r[0].lo.is_some(), r[0].hi.is_some()), ("t", true, true));
        assert_eq!((r[1].var.as_str(), r[1].lo.is_some(), r[1].hi.is_some()), ("u", true, false));
        assert_eq!((r[2].var.as_str(), r[2].lo.is_some(), r[2].hi.is_some()), ("v", false, true));
        // reversed and slider bounds
        let e = parse("(t,t) {a>=t>=-1}").unwrap();
        let (_, r) = unwrap_domain(&e);
        assert_eq!(r[0].lo, Some(Expr::Neg(Box::new(Expr::Num(1.0)))));
        assert_eq!(r[0].hi, Some(Expr::var("a")));
    }

    #[test]
    fn malformed_ranges_are_parse_errors() {
        assert!(parse("(t,t) {1<2}").is_err());
        assert!(parse("(t,t) {t<t+1}").is_err());
    }

    #[test]
    fn resolves_defaults_and_half_ranges() {
        let d = [0.0, 6.0];
        assert_eq!(resolve_range(None, None, d), Ok([0.0, 6.0]));
        assert_eq!(resolve_range(Some(-3.0), Some(3.0), d), Ok([-3.0, 3.0]));
        assert_eq!(resolve_range(Some(2.0), None, d), Ok([2.0, 6.0]));
        assert_eq!(resolve_range(Some(7.0), None, d), Ok([7.0, 13.0]));
        assert_eq!(resolve_range(None, Some(-1.0), d), Ok([-7.0, -1.0]));
        assert!(resolve_range(Some(3.0), Some(3.0), d).is_err());
        assert!(resolve_range(Some(f64::NAN), None, d).is_err());
    }
}
