//! Desmos-style actions: `a -> a+1, b -> 2b`.
//!
//! An action is a comma-separated list of assignments `target -> expression` (the arrow may
//! also be `\to`, `\rightarrow` or `→`). Targets are existing slider names or plain numeric
//! definitions (`a=3`). Assignments are SIMULTANEOUS: every right side is evaluated against the
//! old values, then all results are applied together.
//!
//! This module only splits the source text (respecting `()[]{}` nesting) and delegates each
//! right side to the ordinary parser, so the expression grammar stays in one place. Evaluation
//! is pure ([`plan`]); the caller applies the resulting [`Change`]s.

use crate::analyze::{analyze, Kind};
use crate::compile::{compile, Angle};
use crate::doc::{AngleMode, Doc, ItemKind};
use crate::parse::{parse_with, ParseCtx};
use crate::resolve::Defs;
use crate::Expr;
use std::collections::BTreeSet;

/// Most assignments one action may hold.
pub const MAX_ASSIGNMENTS: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: String,
    pub rhs: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub assignments: Vec<Assignment>,
}

/// Where a computed value goes.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetKind {
    Slider,
    /// A plain numeric definition living in item `item`.
    Definition { item: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub name: String,
    pub value: f64,
    pub target: TargetKind,
}

/// A top-level arrow found at byte range `start..end`.
fn arrow_at(src: &str, i: usize) -> Option<usize> {
    let rest = &src[i..];
    if rest.starts_with("->") {
        return Some(2);
    }
    if rest.starts_with('\u{2192}') {
        return Some('\u{2192}'.len_utf8());
    }
    if let Some(cmd) = rest.strip_prefix('\\') {
        let name: String = cmd.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if matches!(name.as_str(), "to" | "rightarrow" | "mapsto") {
            return Some(1 + name.len());
        }
    }
    None
}

/// True when the text contains a top-level action arrow (a cheap check used to route items).
pub fn looks_like_action(src: &str) -> bool {
    split_clauses(src).map(|c| c.iter().any(|(_, r)| r.is_some())).unwrap_or(false)
}

type Clause<'a> = (&'a str, Option<&'a str>);

/// Splits on top-level commas, then each clause on its first top-level arrow:
/// `(lhs, Some(rhs))`, or `(whole, None)` when a clause has no arrow.
fn push_clause<'a>(src: &'a str, end: usize, arrow: Option<(usize, usize)>, start: usize, out: &mut Vec<Clause<'a>>) {
    match arrow {
        Some((a0, a1)) => out.push((&src[start..a0], Some(&src[a1..end]))),
        None => out.push((&src[start..end], None)),
    }
}

fn split_clauses(src: &str) -> Result<Vec<Clause<'_>>, String> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let (mut start, mut arrow): (usize, Option<(usize, usize)>) = (0, None);
    let mut i = 0;
    let bytes = src.as_bytes();
    while i < src.len() {
        if let Some(len) = arrow_at(src, i) {
            if depth == 0 {
                if arrow.is_some() {
                    return Err("an action clause has two arrows (separate assignments with commas)".into());
                }
                arrow = Some((i, i + len));
            }
            i += len;
            continue;
        }
        let c = bytes[i];
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                push_clause(src, i, arrow.take(), start, &mut out);
                start = i + 1;
            }
            _ => {}
        }
        // Advance one char (source may hold multi-byte characters).
        i += src[i..].chars().next().map_or(1, char::len_utf8);
    }
    if depth != 0 {
        return Err("unbalanced brackets in action".into());
    }
    push_clause(src, src.len(), arrow, start, &mut out);
    Ok(out)
}

/// Accepts `a`, `k_1`, `a_{1}`; returns the plain name.
fn target_name(s: &str) -> Option<String> {
    let n: String = s.trim().chars().filter(|c| *c != '{' && *c != '}').collect();
    let ok = !n.is_empty()
        && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    ok.then_some(n)
}

/// Parses an action. Right sides use `ctx` (so user functions are callable).
pub fn parse_action(src: &str, ctx: &ParseCtx) -> Result<Action, String> {
    let clauses = split_clauses(src)?;
    let mut assignments: Vec<Assignment> = Vec::new();
    for (lhs, rhs) in clauses {
        let Some(rhs) = rhs else {
            if lhs.trim().is_empty() && assignments.is_empty() {
                continue;
            }
            return Err(format!("'{}' has no arrow (write name -> expression)", lhs.trim()));
        };
        let target = target_name(lhs).ok_or_else(|| format!("'{}' is not a variable name", lhs.trim()))?;
        if assignments.iter().any(|a| a.target == target) {
            return Err(format!("'{target}' is assigned twice"));
        }
        if rhs.trim().is_empty() {
            return Err(format!("missing expression after '{target} ->'"));
        }
        let rhs = parse_with(rhs, ctx).map_err(|e| format!("{target}: {e}"))?;
        assignments.push(Assignment { target, rhs });
    }
    if assignments.is_empty() {
        return Err("empty action".into());
    }
    if assignments.len() > MAX_ASSIGNMENTS {
        return Err(format!("too many assignments (max {MAX_ASSIGNMENTS})"));
    }
    Ok(Action { assignments })
}

fn angle_of(a: AngleMode) -> Angle {
    match a {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    }
}

fn is_value_item(k: ItemKind) -> bool {
    !matches!(k, ItemKind::Folder | ItemKind::Note | ItemKind::Action | ItemKind::Table | ItemKind::Complex)
}

/// Parse context and definitions of every ordinary item of `doc`, with slider values applied.
/// Also returns the plain numeric definitions as `(name, item id)`.
fn scope(doc: &Doc) -> (ParseCtx, Defs, Vec<(String, String)>) {
    let none = BTreeSet::new();
    let items: Vec<_> = doc
        .items
        .iter()
        .filter(|i| is_value_item(i.kind) && !i.latex.trim().is_empty() && !looks_like_action(&i.latex))
        .collect();
    let mut ctx = ParseCtx::new();
    for it in &items {
        if let Ok(e) = parse_with(&it.latex, &ParseCtx::new()) {
            if let Kind::Definition { name, params, .. } = analyze(&e, &none).kind {
                if !params.is_empty() {
                    ctx = ctx.with_function(&name);
                }
            }
        }
    }
    let mut kinds = Vec::new();
    let mut numeric = Vec::new();
    for it in items {
        if let Ok(e) = parse_with(&it.latex, &ctx) {
            let k = analyze(&e, &none).kind;
            if let Kind::Definition { name, params, body } = &k {
                if params.is_empty() && body.free_vars().is_empty() && !numeric.iter().any(|(n, _): &(String, String)| n == name) {
                    numeric.push((name.clone(), it.id.clone()));
                }
            }
            kinds.push(k);
        }
    }
    let mut defs = Defs::from_kinds(kinds.iter());
    defs.set_angle(angle_of(doc.view.angle));
    for (n, c) in &doc.sliders {
        defs.set_slider(n, c.value);
    }
    (ctx, defs, numeric)
}

/// Parses the text of action item `id` (with the document's functions in scope).
pub fn parse_item(doc: &Doc, id: &str) -> Result<Action, String> {
    let it = doc.items.iter().find(|i| i.id == id).ok_or_else(|| format!("no item with id '{id}'"))?;
    if it.kind != ItemKind::Action && !looks_like_action(&it.latex) {
        return Err(format!("item '{id}' is not an action"));
    }
    let (ctx, _, _) = scope(doc);
    parse_action(&it.latex, &ctx)
}

/// Rounds to 12 significant digits so repeated `a -> a+0.1` does not accumulate float noise.
pub fn tidy(v: f64) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return if v == 0.0 { 0.0 } else { v };
    }
    format!("{v:.11e}").parse().unwrap_or(v)
}

/// Evaluates every right side against the CURRENT values and returns the changes to apply.
/// Atomic: any bad target or non-finite result is an error and nothing should be applied.
pub fn plan(doc: &Doc, action: &Action) -> Result<Vec<Change>, String> {
    let (_, defs, numeric) = scope(doc);
    let mut out = Vec::with_capacity(action.assignments.len());
    for a in &action.assignments {
        let target = if doc.sliders.contains_key(&a.target) {
            TargetKind::Slider
        } else if let Some((_, item)) = numeric.iter().find(|(n, _)| *n == a.target) {
            TargetKind::Definition { item: item.clone() }
        } else {
            return Err(format!("'{}' is not a slider or a plain numeric definition", a.target));
        };
        let resolved = defs.resolve(&a.rhs).map_err(|e| format!("{}: {e}", a.target))?;
        let prog = compile(&resolved, &[], angle_of(doc.view.angle)).map_err(|e| format!("{}: {e}", a.target))?;
        let v = prog.eval(&[]);
        if !v.is_finite() {
            return Err(format!("{} -> ... is not a finite number ({v})", a.target));
        }
        out.push(Change { name: a.target.clone(), value: tidy(v), target });
    }
    Ok(out)
}

/// The text a rewritten numeric definition gets.
pub fn definition_text(name: &str, value: f64) -> String {
    let v = tidy(value);
    format!("{name}={}", if v == 0.0 { 0.0 } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::{Item, SliderCfg};

    fn doc_with(sliders: &[(&str, f64)], items: &[(&str, ItemKind, &str)]) -> Doc {
        let mut d = Doc::new_default();
        for (n, v) in sliders {
            d.sliders.insert((*n).into(), SliderCfg { min: -100.0, max: 100.0, step: None, value: *v });
        }
        for (id, k, l) in items {
            d.add_item(Item::new(id, *k, l)).unwrap();
        }
        d
    }

    fn pa(s: &str) -> Result<Action, String> {
        parse_action(s, &ParseCtx::new())
    }

    #[test]
    fn syntax_variants() {
        for s in ["a -> a+1", "a\\to a+1", "a\u{2192}a+1", "a \\rightarrow a+1", "a_{1}->1"] {
            assert_eq!(pa(s).unwrap().assignments.len(), 1, "{s}");
        }
        assert_eq!(pa("a -> a+1, b -> 2b").unwrap().assignments.len(), 2);
        assert_eq!(pa("a_{1}->1").unwrap().assignments[0].target, "a_1");
    }

    #[test]
    fn commas_inside_brackets_do_not_split() {
        let a = pa("a -> max(a, 3), b -> sum([1,2,3])").unwrap();
        assert_eq!(a.assignments.len(), 2);
        assert_eq!(a.assignments[0].target, "a");
    }

    #[test]
    fn rejects_bad_actions() {
        assert!(pa("a+1").is_err());
        assert!(pa("a -> 1, a -> 2").unwrap_err().contains("twice"));
        assert!(pa("2a -> 1").is_err());
        assert!(pa("a -> ").is_err());
        assert!(pa("a -> (1").is_err());
        assert!(pa("a -> 1 -> 2").is_err());
        assert!(pa("").is_err());
    }

    #[test]
    fn looks_like_action_routing() {
        assert!(looks_like_action("a -> a+1"));
        assert!(looks_like_action("a \\to 1, b \\to 2"));
        assert!(!looks_like_action("y = a x"));
        assert!(!looks_like_action("f(a -> 1)"), "arrow nested in parentheses is not top level");
    }

    #[test]
    fn simultaneous_assignment() {
        // swap: a -> b, b -> a must read the OLD values.
        let d = doc_with(&[("a", 1.0), ("b", 2.0)], &[("act", ItemKind::Action, "a -> b, b -> a")]);
        let act = parse_item(&d, "act").unwrap();
        let ch = plan(&d, &act).unwrap();
        let get = |n: &str| ch.iter().find(|c| c.name == n).unwrap().value;
        assert_eq!((get("a"), get("b")), (2.0, 1.0));
        // dependent update uses the old a, not the new one.
        let act = pa("a -> a+1, b -> a*10").unwrap();
        let ch = plan(&d, &act).unwrap();
        assert_eq!((ch[0].value, ch[1].value), (2.0, 10.0));
    }

    #[test]
    fn definition_targets_and_errors() {
        let d = doc_with(
            &[("a", 1.0)],
            &[
                ("d", ItemKind::Equation, "k=3"),
                ("e", ItemKind::Equation, "m=k+1"),
                ("f", ItemKind::Equation, "g(x)=x^2"),
                ("act", ItemKind::Action, "k -> k+1"),
            ],
        );
        let ch = plan(&d, &pa("k -> k+1").unwrap()).unwrap();
        assert_eq!(ch[0].target, TargetKind::Definition { item: "d".into() });
        assert_eq!(ch[0].value, 4.0);
        assert_eq!(definition_text("k", 4.0), "k=4");
        // derived definitions, functions, unknown names and NaN are diagnostics.
        assert!(plan(&d, &pa("m -> 1").unwrap()).unwrap_err().contains("not a slider"));
        assert!(plan(&d, &pa("g -> 1").unwrap()).is_err());
        assert!(plan(&d, &pa("zz -> 1").unwrap()).is_err());
        assert!(plan(&d, &pa("a -> 0/0").unwrap()).unwrap_err().contains("finite"));
        assert!(plan(&d, &pa("a -> q+1").unwrap()).is_err());
        // atomic: a bad second assignment fails the whole action.
        assert!(plan(&d, &pa("a -> 5, zz -> 1").unwrap()).is_err());
    }

    #[test]
    fn user_functions_and_tidy() {
        let d = doc_with(&[("a", 0.1)], &[("f", ItemKind::Equation, "f(x)=2x"), ("act", ItemKind::Action, "a -> f(a)+0.2")]);
        let ch = plan(&d, &parse_item(&d, "act").unwrap()).unwrap();
        assert_eq!(ch[0].value, 0.4);
        assert_eq!(tidy(0.1 + 0.2), 0.3);
        assert!(parse_item(&d, "f").is_err());
    }
}
