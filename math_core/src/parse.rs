//! Text and LaTeX-subset parser (hand-written Pratt parser).
//!
//! Binding powers: relation 10, `+ -` 20, `* /` and implicit multiplication 30,
//! prefix minus 35, function argument without parentheses 36, `^` 40 (right associative).

use crate::ast::{is_builtin_func, BinOp, Expr, Rel, BUILTIN_FUNCS, NAMED_SYMBOLS};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub pos: usize,
    pub msg: String,
}

impl ParseError {
    fn new(pos: usize, msg: impl Into<String>) -> Self {
        ParseError { pos, msg: msg.into() }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at {})", self.msg, self.pos)
    }
}

impl std::error::Error for ParseError {}

/// Names the parser should treat as callable user functions, e.g. `f` after `f(x)=x^2`.
#[derive(Debug, Clone, Default)]
pub struct ParseCtx {
    pub functions: HashSet<String>,
}

impl ParseCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_function(mut self, name: &str) -> Self {
        self.functions.insert(name.to_string());
        self
    }
}

pub fn parse(src: &str) -> Result<Expr, ParseError> {
    parse_with(src, &ParseCtx::default())
}

pub fn parse_with(src: &str, ctx: &ParseCtx) -> Result<Expr, ParseError> {
    // A trailing `{a<=t<=b, ...}` restricts the parameters: `domain(body, clause, ...)`.
    if let Some((body, inner)) = crate::param::split_domain_src(src) {
        let mut parts = vec![parse_body(body, ctx)?];
        for clause in crate::param::split_top_level(inner).into_iter().filter(|c| !c.is_empty()) {
            parts.push(parse_body(clause, ctx)?);
        }
        if parts.len() < 2 {
            return Err(ParseError::new(body.len(), "empty range"));
        }
        crate::param::ranges_of(&parts[1..]).map_err(|m| ParseError::new(body.len(), m))?;
        return Ok(Expr::Call(crate::ast::DOMAIN_FN.to_string(), parts));
    }
    parse_body(src, ctx)
}

fn parse_body(src: &str, ctx: &ParseCtx) -> Result<Expr, ParseError> {
    let norm = normalize(src)?;
    match parse_normalized(&norm, ctx) {
        Ok(e) => Ok(e),
        // `x=3cos(t), y=2sin(t)`: a bare top-level comma was always an error, now it joins
        // axis equations into the tuple of their right sides.
        Err(err) => parse_axis_pair(&norm, ctx).ok_or(err),
    }
}

fn parse_normalized(norm: &str, ctx: &ParseCtx) -> Result<Expr, ParseError> {
    let toks = lex(norm)?;
    let mut ctx = ctx.clone();
    if let Some(name) = definition_prefix(&toks) {
        ctx.functions.insert(name);
    }
    let mut p = Parser { toks, i: 0, ctx: &ctx, abs_depth: 0, depth: 0 };
    let e = p.expr(0)?;
    match p.peek().tok {
        Tok::Eof => Ok(e),
        _ => Err(ParseError::new(p.peek().pos, "unexpected trailing input")),
    }
}

/// `x=f, y=g` or `x=f, y=g, z=h` (any order, separated by `,` or `;`) as `(f, g)` / `(f, g, h)`.
fn parse_axis_pair(norm: &str, ctx: &ParseCtx) -> Option<Expr> {
    let parts = crate::param::split_top_level(norm);
    if !(2..=3).contains(&parts.len()) {
        return None;
    }
    let mut slots: [Option<Expr>; 3] = [None, None, None];
    for part in parts {
        let Expr::Rel(Rel::Eq, lhs, rhs) = parse_normalized(part, ctx).ok()? else { return None };
        let axis = match &*lhs {
            Expr::Var(n) if n == "x" => 0,
            Expr::Var(n) if n == "y" => 1,
            Expr::Var(n) if n == "z" => 2,
            _ => return None,
        };
        if slots[axis].is_some() || ["x", "y", "z"].iter().any(|v| rhs.contains_var(v)) {
            return None;
        }
        slots[axis] = Some(*rhs);
    }
    let [x, y, z] = slots;
    Some(Expr::Tuple(vec![x?, y?].into_iter().chain(z).collect()))
}

// ---------------------------------------------------------------- LaTeX normalization

const LATEX_FUNCS: &[&str] = &[
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "exp", "ln", "log", "min", "max",
];

const LATEX_SYMBOLS: &[&str] = &["pi", "tau", "theta"];

/// Rewrites the supported LaTeX subset into plain text. Plain text passes through unchanged.
pub fn normalize(src: &str) -> Result<String, ParseError> {
    if !src.contains('\\') && !src.contains('{') && !src.contains('}') {
        return Ok(src.to_string());
    }
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    transform(&chars, &mut i, false)
}

fn read_group(chars: &[char], i: &mut usize) -> Result<String, ParseError> {
    skip_spaces(chars, i);
    if *i >= chars.len() || chars[*i] != '{' {
        return Err(ParseError::new(*i, "expected '{'"));
    }
    let start = *i + 1;
    let mut depth = 0usize;
    let mut j = *i;
    while j < chars.len() {
        match chars[j] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let inner: String = chars[start..j].iter().collect();
                    *i = j + 1;
                    return Ok(inner);
                }
            }
            _ => {}
        }
        j += 1;
    }
    Err(ParseError::new(*i, "unbalanced '{'"))
}

/// `\frac{d}{dx}` and `\frac{d^2}{dx^2}` become `d/dx ` and `d^2/dx^2 ` (the Leibniz operator
/// the parser understands). Anything else stays an ordinary fraction.
fn leibniz_frac(a: &str, b: &str) -> Option<String> {
    let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace() && *c != '{' && *c != '}').collect::<String>();
    let (a, b) = (strip(a), strip(b));
    let order = a.strip_prefix('d')?;
    let rest = b.strip_prefix('d')?;
    let (var, bpow) = match rest.split_once('^') {
        Some((v, p)) => (v, Some(p)),
        None => (rest, None),
    };
    let var = match var {
        "\\theta" => "theta",
        v if v.chars().count() == 1 && v.chars().all(|c| c.is_ascii_alphabetic()) => v,
        _ => return None,
    };
    let apow = match order {
        "" => None,
        o => Some(o.strip_prefix('^')?),
    };
    if apow != bpow || apow.is_some_and(|p| p.is_empty() || !p.chars().all(|c| c.is_ascii_digit())) {
        return None;
    }
    Some(match apow {
        Some(n) => format!(" d^{n}/d{var}^{n} "),
        None => format!(" d/d{var} "),
    })
}

/// The `_lo` and `^hi` arguments of `\\int`, `\\sum`, `\\prod` (either order, each a `{group}` or one
/// character/command). Missing limits come back empty.
fn read_limits(chars: &[char], i: &mut usize) -> Result<(String, String), ParseError> {
    let (mut lo, mut hi) = (String::new(), String::new());
    for _ in 0..2 {
        skip_spaces(chars, i);
        let which = match chars.get(*i) {
            Some('_') => &mut lo,
            Some('^') => &mut hi,
            _ => break,
        };
        *i += 1;
        skip_spaces(chars, i);
        match chars.get(*i) {
            Some('{') => *which = read_group(chars, i)?,
            Some('\\') => {
                let start = *i;
                *i += 1;
                while *i < chars.len() && chars[*i].is_ascii_alphabetic() {
                    *i += 1;
                }
                *which = chars[start..*i].iter().collect();
            }
            Some(c) => {
                *which = c.to_string();
                *i += 1;
            }
            None => return Err(ParseError::new(*i, "missing limit")),
        }
    }
    Ok((lo, hi))
}

/// True when `\\int`/`\\sum`/`\\prod` is directly followed by `(` or `\\left(` whose group holds a
/// top-level comma (an argument list), as opposed to a bracketed integrand or limit-less body.
fn is_call_form(chars: &[char], at: usize) -> bool {
    let mut j = at;
    skip_spaces(chars, &mut j);
    if chars[j..].starts_with(&['\\', 'l', 'e', 'f', 't']) {
        j += 5;
        skip_spaces(chars, &mut j);
    }
    if chars.get(j) != Some(&'(') {
        return false;
    }
    let (mut paren, mut brace) = (0i32, 0i32);
    while j < chars.len() {
        match chars[j] {
            '\\' => j += 1,
            '(' => paren += 1,
            ')' => {
                paren -= 1;
                if paren == 0 {
                    return false;
                }
            }
            '{' => brace += 1,
            '}' => brace -= 1,
            ',' if paren == 1 && brace == 0 => return true,
            _ => {}
        }
        j += 1;
    }
    false
}

/// Private-use markers the normaliser leaves for a piecewise group `{c: v, ...}`.
const PIECE_OPEN: char = '\u{2983}';
const PIECE_CLOSE: char = '\u{2984}';

/// For a brace group whose content starts at `from`: the index where the content ends (before
/// the closing brace, `\}` or `\rbrace`) and the index after the closer. Counts `{`, `\{`
/// and `\lbrace` as openers.
fn find_brace_close(chars: &[char], from: usize) -> Option<(usize, usize)> {
    let mut depth = 0usize;
    let mut i = from;
    while i < chars.len() {
        match chars[i] {
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return Some((i, i + 1));
                }
                depth -= 1;
            }
            '\\' => {
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if j == start {
                    match chars.get(j) {
                        Some('{') => depth += 1,
                        Some('}') => {
                            if depth == 0 {
                                return Some((i, j + 1));
                            }
                            depth -= 1;
                        }
                        _ => {}
                    }
                    i = j + 1;
                    continue;
                }
                let name: String = chars[start..j].iter().collect();
                match name.as_str() {
                    "lbrace" => depth += 1,
                    "rbrace" => {
                        if depth == 0 {
                            return Some((i, j));
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// A brace group is a piecewise value when, outside every bracket, it has a `:` or a comparison
/// (`{x<0: -x, x}`, `{x>0}`). Other groups (`{x}`) stay plain parentheses.
fn is_piece_group(inner: &str) -> bool {
    let chars: Vec<char> = inner.chars().collect();
    let mut depth = 0i32;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ':' if depth <= 0 => return true,
            '<' | '>' | '=' | '\u{2264}' | '\u{2265}' | '\u{2260}' if depth <= 0 => return true,
            '!' if depth <= 0 && chars.get(i + 1) == Some(&'=') => return true,
            '\\' => {
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if j == start {
                    match chars.get(j) {
                        Some('{') => depth += 1,
                        Some('}') => depth -= 1,
                        _ => {}
                    }
                    i = j + 1;
                    continue;
                }
                let name: String = chars[start..j].iter().collect();
                match name.as_str() {
                    "lbrace" => depth += 1,
                    "rbrace" => depth -= 1,
                    "le" | "leq" | "leqslant" | "ge" | "geq" | "geqslant" | "lt" | "gt" | "ne" | "neq" if depth <= 0 => return true,
                    _ => {}
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn skip_spaces(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i] == ' ' {
        *i += 1;
    }
}

fn sub_normalize(inner: &str) -> Result<String, ParseError> {
    let chars: Vec<char> = inner.chars().collect();
    let mut i = 0;
    transform(&chars, &mut i, true)
}

fn transform(chars: &[char], i: &mut usize, _nested: bool) -> Result<String, ParseError> {
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        match c {
            '\\' => {
                *i += 1;
                let start = *i;
                while *i < chars.len() && chars[*i].is_ascii_alphabetic() {
                    *i += 1;
                }
                let name: String = chars[start..*i].iter().collect();
                if name.is_empty() {
                    // Escaped punctuation such as `\,` `\ ` `\{`.
                    if *i < chars.len() {
                        let p = chars[*i];
                        *i += 1;
                        if p == '{' {
                            if let Some((end, after)) = find_brace_close(chars, *i) {
                                let inner: String = chars[*i..end].iter().collect();
                                if is_piece_group(&inner) {
                                    out.push(PIECE_OPEN);
                                    out.push_str(&sub_normalize(&inner)?);
                                    out.push(PIECE_CLOSE);
                                    *i = after;
                                    continue;
                                }
                            }
                        }
                        match p {
                            '{' => out.push('('),
                            '}' => out.push(')'),
                            // Thin/medium spaces separate tokens (`x^2\,dx`).
                            ',' | ';' | ':' | ' ' | '!' => out.push(' '),
                            // `\%` is MathLive's typed percent sign: the modulo operator
                            '%' => out.push('%'),
                            _ => {}
                        }
                    }
                    continue;
                }
                match name.as_str() {
 "lbrace" | "rbrace" => {
                        // MathLive's typed brace: the same as `\{` / `\}`
                        let open = name == "lbrace";
                        let mut handled = false;
                        if open {
                            if let Some((end, after)) = find_brace_close(chars, *i) {
                                let inner: String = chars[*i..end].iter().collect();
                                if is_piece_group(&inner) {
                                    out.push(PIECE_OPEN);
                                    out.push_str(&sub_normalize(&inner)?);
                                    out.push(PIECE_CLOSE);
                                    *i = after;
                                    handled = true;
                                }
                            }
                        }
                        if !handled {
                            out.push(if open { '(' } else { ')' });
                        }
                    }
                    "left" | "right" | "limits" | "nolimits" | "displaystyle" | "textstyle" | "quad" | "qquad" => {}
                    "int" | "sum" | "prod" if is_call_form(chars, *i) => {
                        // Pasted call form `\int(x^2,x,0,2)`: the plain `int(...)` call.
                        out.push(' ');
                        out.push_str(&name);
                    }
                    "int" | "sum" | "prod" => {
                        let glyph = match name.as_str() {
                            "int" => '\u{222b}',
                            "sum" => '\u{2211}',
                            _ => '\u{220f}',
                        };
                        let (lo, hi) = read_limits(chars, i)?;
                        out.push(' ');
                        out.push(glyph);
                        out.push_str(&format!("({})({}) ", sub_normalize(&lo)?, sub_normalize(&hi)?));
                    }
                    "ne" | "neq" => out.push_str(" != "),
                    // MathLive writes a typed `[` and `]` as these (lists: `[(1,2),(3,4)]`).
                    "lbrack" => out.push('['),
                    "rbrack" => out.push(']'),
                    "lt" => out.push('<'),
                    "gt" => out.push('>'),
                    "cdot" | "times" => out.push('*'),
                    "ldots" | "dots" | "cdots" => out.push_str("..."),
                    "div" => out.push('/'),
                    "le" | "leq" | "leqslant" => out.push_str(" <= "),
                    "ge" | "geq" | "geqslant" => out.push_str(" >= "),
                    "sim" => out.push('~'),
                    "frac" => {
                        let a = read_group(chars, i)?;
                        let b = read_group(chars, i)?;
                        if let Some(d) = leibniz_frac(&a, &b) {
                            out.push_str(&d);
                            continue;
                        }
                        out.push_str(&format!("(({})/({}))", sub_normalize(&a)?, sub_normalize(&b)?));
                    }
                    "binom" | "dbinom" | "tbinom" => {
                        let a = read_group(chars, i)?;
                        let b = read_group(chars, i)?;
                        out.push_str(&format!(" nCr({}, {}) ", sub_normalize(&a)?, sub_normalize(&b)?));
                    }
                    "sqrt" => {
                        skip_spaces(chars, i);
                        if *i < chars.len() && chars[*i] == '[' {
                            let close = chars[*i..]
                                .iter()
                                .position(|c| *c == ']')
                                .ok_or_else(|| ParseError::new(*i, "unbalanced '['"))?;
                            let n: String = chars[*i + 1..*i + close].iter().collect();
                            *i += close + 1;
                            let a = read_group(chars, i)?;
                            out.push_str(&format!(
                                "(({})^(1/({})))",
                                sub_normalize(&a)?,
                                sub_normalize(&n)?
                            ));
                        } else {
                            let a = read_group(chars, i)?;
                            out.push_str(&format!("sqrt({})", sub_normalize(&a)?));
                        }
                    }
                    "operatorname" | "mathrm" | "text" => {
                        let a = read_group(chars, i)?;
                        // A leading space keeps `a\operatorname{sin}` from gluing into `asin`.
                        out.push(' ');
                        out.push_str(&a);
                        if name == "operatorname" {
                            out.push(' ');
                        }
                    }
                    // Spaces on both sides: `a\sin(x)` is a times sin, never the glued name `asin`.
                    n if LATEX_FUNCS.contains(&n) || LATEX_SYMBOLS.contains(&n) => {
                        out.push(' ');
                        out.push_str(n);
                        out.push(' ');
                    }
                    other => {
                        return Err(ParseError::new(*i, format!("unsupported LaTeX command \\{other}")))
                    }
                }
            }
            '^' | '_' => {
                out.push(c);
                *i += 1;
                if *i < chars.len() && chars[*i] == '{' {
                    let inner = read_group(chars, i)?;
                    let inner = sub_normalize(&inner)?;
                    if c == '^' {
                        out.push('(');
                        out.push_str(&inner);
                        out.push(')');
                    } else {
                        out.push_str(&inner);
                    }
                }
            }
            '{' => {
                if let Some((end, after)) = find_brace_close(chars, *i + 1) {
                    let inner: String = chars[*i + 1..end].iter().collect();
                    if is_piece_group(&inner) {
                        out.push(PIECE_OPEN);
                        out.push_str(&sub_normalize(&inner)?);
                        out.push(PIECE_CLOSE);
                        *i = after;
                        continue;
                    }
                }
                out.push('(');
                *i += 1;
            }
            '}' => {
                out.push(')');
                *i += 1;
            }
            _ => {
                out.push(c);
                *i += 1;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Ellipsis,
    For,
    Pipe,
    /// `\u{222b}`, `\u{2211}`, `\u{220f}` produced by the normaliser for `\\int`, `\\sum`, `\\prod`.
    BigOp(char),
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    /// `!=` (the `!` directly followed by `=`) or `≠`.
    Ne,
    Percent,
    Tilde,
    Prime,
    Bang,
    /// Piecewise group `{ ... }` (the normaliser's private markers) and its `:`.
    LBrace,
    RBrace,
    Colon,
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    pos: usize,
    /// Whitespace directly before this token (a `[` only indexes when there is none).
    space_before: bool,
}

fn lex(s: &str) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let push = |out: &mut Vec<Token>, tok: Tok, pos: usize| out.push(Token { tok, pos, space_before: false });
    let mut ws = false;
    // Open `[` count: `for` is a keyword only inside a list literal.
    let mut bracket_depth = 0usize;
    while i < chars.len() {
        let (pos, c) = chars[i];
        let n0 = out.len();
        match c {
            c if c.is_whitespace() => i += 1,
            '.' if i + 2 < chars.len() && chars[i + 1].1 == '.' && chars[i + 2].1 == '.' => {
                push(&mut out, Tok::Ellipsis, pos);
                i += 3;
            }
            '0'..='9' | '.' => {
                let mut text = String::new();
                let mut seen_dot = false;
                while i < chars.len() {
                    let d = chars[i].1;
                    if d.is_ascii_digit() {
                        text.push(d);
                    } else if d == '.' && !seen_dot && chars.get(i + 1).map(|c| c.1) != Some('.') {
                        seen_dot = true;
                        text.push(d);
                    } else {
                        break;
                    }
                    i += 1;
                }
                if !text.chars().any(|d| d.is_ascii_digit()) {
                    return Err(ParseError::new(pos, "stray '.'"));
                }
                // Exponent only when digits follow directly, so `2e` stays 2 * e.
                if i + 1 < chars.len()
                    && (chars[i].1 == 'e' || chars[i].1 == 'E')
                    && chars[i + 1].1.is_ascii_digit()
                {
                    text.push('e');
                    i += 1;
                    while i < chars.len() && chars[i].1.is_ascii_digit() {
                        text.push(chars[i].1);
                        i += 1;
                    }
                }
                let v: f64 = text
                    .parse()
                    .map_err(|_| ParseError::new(pos, format!("bad number '{text}'")))?;
                push(&mut out, Tok::Num(v), pos);
            }
            'π' => {
                push(&mut out, Tok::Ident("pi".into()), pos);
                i += 1;
            }
            'θ' => {
                push(&mut out, Tok::Ident("theta".into()), pos);
                i += 1;
            }
            'τ' => {
                push(&mut out, Tok::Ident("tau".into()), pos);
                i += 1;
            }
            '√' => {
                push(&mut out, Tok::Ident("sqrt".into()), pos);
                i += 1;
            }
            c if c.is_alphabetic() => {
                // Longest builtin/symbol prefix of the ASCII letter run wins; otherwise one letter.
                let run: String = chars[i..]
                    .iter()
                    .map(|(_, ch)| *ch)
                    .take_while(|ch| ch.is_ascii_alphanumeric())
                    .collect();
                if run == "for" && bracket_depth > 0 {
                    push(&mut out, Tok::For, pos);
                    i += 3;
                } else {
                let best = BUILTIN_FUNCS
                    .iter()
                    .chain(NAMED_SYMBOLS.iter())
                    .filter(|n| run.starts_with(**n))
                    .max_by_key(|n| n.len());
                if let Some(name) = best {
                    push(&mut out, Tok::Ident(canonical_alias(name).to_string()), pos);
                    i += name.chars().count();
                } else {
                    let mut name = c.to_string();
                    i += 1;
                    if i < chars.len() && chars[i].1 == '_' {
                        let mut j = i + 1;
                        let mut sub = String::new();
                        while j < chars.len() && chars[j].1.is_alphanumeric() {
                            sub.push(chars[j].1);
                            j += 1;
                        }
                        if !sub.is_empty() {
                            name = format!("{name}_{sub}");
                            i = j;
                        }
                    }
                    push(&mut out, Tok::Ident(name), pos);
                }
                }
            }
            '\u{222b}' | '\u{2211}' | '\u{220f}' => {
                push(&mut out, Tok::BigOp(c), pos);
                i += 1;
            }
            '+' => {
                push(&mut out, Tok::Plus, pos);
                i += 1;
            }
            '-' | '−' => {
                push(&mut out, Tok::Minus, pos);
                i += 1;
            }
            '*' | '·' | '×' => {
                push(&mut out, Tok::Star, pos);
                i += 1;
            }
            '/' | '÷' => {
                push(&mut out, Tok::Slash, pos);
                i += 1;
            }
            '^' => {
                push(&mut out, Tok::Caret, pos);
                i += 1;
            }
            '(' => {
                push(&mut out, Tok::LParen, pos);
                i += 1;
            }
            ')' => {
                push(&mut out, Tok::RParen, pos);
                i += 1;
            }
            '[' => {
                push(&mut out, Tok::LBracket, pos);
                bracket_depth += 1;
                i += 1;
            }
            ']' => {
                push(&mut out, Tok::RBracket, pos);
                bracket_depth = bracket_depth.saturating_sub(1);
                i += 1;
            }
            ',' => {
                push(&mut out, Tok::Comma, pos);
                i += 1;
            }
            '|' => {
                push(&mut out, Tok::Pipe, pos);
                i += 1;
            }
            '=' => {
                push(&mut out, Tok::Eq, pos);
                i += 1;
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '=' {
                    push(&mut out, Tok::Le, pos);
                    i += 2;
                } else {
                    push(&mut out, Tok::Lt, pos);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1].1 == '=' {
                    push(&mut out, Tok::Ge, pos);
                    i += 2;
                } else {
                    push(&mut out, Tok::Gt, pos);
                    i += 1;
                }
            }
            '~' => {
                push(&mut out, Tok::Tilde, pos);
                i += 1;
            }
            // `!` directly followed by `=` is the relation; `n! = 3` (a space) stays a factorial.
            '!' if chars.get(i + 1).map(|c| c.1) == Some('=') => {
                push(&mut out, Tok::Ne, pos);
                i += 2;
            }
            '\u{2260}' => {
                push(&mut out, Tok::Ne, pos);
                i += 1;
            }
            '%' => {
                push(&mut out, Tok::Percent, pos);
                i += 1;
            }
            '!' => {
                push(&mut out, Tok::Bang, pos);
                i += 1;
            }
            '\'' | '′' => {
                push(&mut out, Tok::Prime, pos);
                i += 1;
            }
            '\u{2983}' => {
                push(&mut out, Tok::LBrace, pos);
                i += 1;
            }
            '\u{2984}' => {
                push(&mut out, Tok::RBrace, pos);
                i += 1;
            }
            ':' => {
                push(&mut out, Tok::Colon, pos);
                i += 1;
            }
            '≤' => {
                push(&mut out, Tok::Le, pos);
                i += 1;
            }
            '≥' => {
                push(&mut out, Tok::Ge, pos);
                i += 1;
            }
            other => return Err(ParseError::new(pos, format!("unexpected character '{other}'"))),
        }
        if out.len() > n0 {
            out[n0].space_before = ws;
            ws = false;
        }
        if c.is_whitespace() {
            ws = true;
        }
    }
    out.push(Token { tok: Tok::Eof, pos: s.len(), space_before: ws });
    Ok(out)
}

/// Maps alias spellings onto the canonical function name so the AST has one form of each.
fn canonical_alias(name: &str) -> &str {
    match name {
        "arcsin" => "asin",
        "arccos" => "acos",
        "arctan" => "atan",
        "sgn" => "sign",
        other => other,
    }
}

/// Detects `f(x, y) =` and returns `f`, so the call parses as a call and not as `f * (x)`.
fn definition_prefix(toks: &[Token]) -> Option<String> {
    let name = match toks.first().map(|t| &t.tok) {
        Some(Tok::Ident(n)) if !is_builtin_func(n) && !NAMED_SYMBOLS.contains(&n.as_str()) => n.clone(),
        _ => return None,
    };
    if !matches!(toks.get(1).map(|t| &t.tok), Some(Tok::LParen)) {
        return None;
    }
    let mut i = 2;
    loop {
        if !matches!(toks.get(i).map(|t| &t.tok), Some(Tok::Ident(_))) {
            return None;
        }
        i += 1;
        match toks.get(i).map(|t| &t.tok) {
            Some(Tok::Comma) => i += 1,
            Some(Tok::RParen) => {
                i += 1;
                break;
            }
            _ => return None,
        }
    }
    if matches!(toks.get(i).map(|t| &t.tok), Some(Tok::Eq)) {
        Some(name)
    } else {
        None
    }
}

// ---------------------------------------------------------------- parser

struct Parser<'a> {
    toks: Vec<Token>,
    i: usize,
    ctx: &'a ParseCtx,
    abs_depth: usize,
    depth: usize,
}

/// Nesting limit so hostile input errors instead of overflowing the stack.
const MAX_DEPTH: usize = 128;

fn is_rel(t: &Tok) -> Option<Rel> {
    match t {
        Tok::Eq => Some(Rel::Eq),
        Tok::Lt => Some(Rel::Lt),
        Tok::Le => Some(Rel::Le),
        Tok::Gt => Some(Rel::Gt),
        Tok::Ge => Some(Rel::Ge),
        Tok::Ne => Some(Rel::Ne),
        _ => None,
    }
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Token {
        &self.toks[self.i]
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.i].clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }

    fn expect(&mut self, want: Tok, what: &str) -> Result<(), ParseError> {
        if self.peek().tok == want {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::new(self.peek().pos, format!("expected {what}")))
        }
    }

    fn starts_primary(&self, t: &Tok) -> bool {
        match t {
            Tok::Num(_) | Tok::Ident(_) | Tok::LParen | Tok::LBracket | Tok::LBrace | Tok::BigOp(_) => true,
            Tok::Pipe => self.abs_depth == 0,
            _ => false,
        }
    }

    fn expr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ParseError::new(self.peek().pos, "expression nested too deeply"));
        }
        let r = self.expr_inner(min_bp);
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.prefix()?;
        loop {
            let t = self.peek().tok.clone();
            match t {
                Tok::Plus | Tok::Minus => {
                    if 20 < min_bp {
                        break;
                    }
                    self.bump();
                    let rhs = self.expr(21)?;
                    let op = if t == Tok::Plus { BinOp::Add } else { BinOp::Sub };
                    lhs = Expr::bin(op, lhs, rhs);
                }
                Tok::Star | Tok::Slash => {
                    if 30 < min_bp {
                        break;
                    }
                    self.bump();
                    let rhs = self.expr(31)?;
                    let op = if t == Tok::Star { BinOp::Mul } else { BinOp::Div };
                    lhs = Expr::bin(op, lhs, rhs);
                }
                // `a % b` is `mod(a, b)`, at the level of `*` and `/` (left associative)
                Tok::Percent => {
                    if 30 < min_bp {
                        break;
                    }
                    self.bump();
                    let rhs = self.expr(31)?;
                    lhs = Expr::call("mod", vec![lhs, rhs]);
                }
                Tok::Caret => {
                    if 40 < min_bp {
                        break;
                    }
                    self.bump();
                    let rhs = self.expr(40)?;
                    lhs = Expr::bin(BinOp::Pow, lhs, rhs);
                }
                Tok::Tilde => {
                    if 10 < min_bp {
                        break;
                    }
                    self.bump();
                    let rhs = self.expr(11)?;
                    if is_rel(&self.peek().tok).is_some() || self.peek().tok == Tok::Tilde {
                        return Err(ParseError::new(self.peek().pos, "a comparison chain cannot be combined with ~"));
                    }
                    lhs = Expr::call("regress", vec![lhs, rhs]);
                }
                ref r if is_rel(r).is_some() => {
                    if 10 < min_bp {
                        break;
                    }
                    let rel = is_rel(r).unwrap();
                    self.bump();
                    let rhs = self.expr(11)?;
                    if is_rel(&self.peek().tok).is_none() {
                        lhs = Expr::Rel(rel, Box::new(lhs), Box::new(rhs));
                        continue;
                    }
                    lhs = self.rel_chain(lhs, rel, rhs)?;
                }
                ref other if self.starts_primary(other) => {
                    if 30 < min_bp {
                        break;
                    }
                    let rhs = self.expr(31)?;
                    lhs = Expr::bin(BinOp::Mul, lhs, rhs);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// `a < b <= c`: the first comparison and its right side are parsed and another comparison
    /// follows. Builds `and(a<b, b<=c)` (at most three comparisons, none of them `=`).
    fn rel_chain(&mut self, first: Expr, rel: Rel, second: Expr) -> Result<Expr, ParseError> {
        let mut operands = vec![first, second];
        let mut rels = vec![rel];
        while let Some(r) = is_rel(&self.peek().tok) {
            let pos = self.peek().pos;
            if rels.len() == 3 {
                return Err(ParseError::new(pos, "a chained comparison can have at most three comparisons"));
            }
            rels.push(r);
            self.bump();
            operands.push(self.expr(11)?);
        }
        if rels.contains(&Rel::Eq) || rels.contains(&Rel::Ne) {
            return Err(ParseError::new(
                self.peek().pos,
                "'=' and '!=' cannot be part of a chained comparison; chain only <, <=, > and >=",
            ));
        }
        let parts = rels
            .iter()
            .enumerate()
            .map(|(i, r)| Expr::Rel(*r, Box::new(operands[i].clone()), Box::new(operands[i + 1].clone())))
            .collect();
        Ok(Expr::Call(crate::ast::CHAIN_FN.to_string(), parts))
    }

    /// A primary followed by any number of postfix factorials: `n!`, `(n+1)!`, `f(x)!`, `n!!`.
    /// `!` binds tighter than `^` (as in Desmos), so `2^3!` is `2^(3!)` and `x^n!` is `x^(n!)`.
    fn prefix(&mut self) -> Result<Expr, ParseError> {
        let signed = matches!(self.peek().tok, Tok::Minus | Tok::Plus);
        let mut e = self.prefix_inner()?;
        if !signed {
            while self.peek().tok == Tok::Bang {
                self.bump();
                e = Expr::call("factorial", vec![e]);
            }
        }
        Ok(e)
    }

    fn prefix_inner(&mut self) -> Result<Expr, ParseError> {
        let t = self.bump();
        match t.tok {
            Tok::Num(v) => Ok(Expr::Num(v)),
            Tok::Minus => Ok(Expr::neg(self.expr(35)?)),
            Tok::Plus => self.expr(35),
            Tok::Ident(name) => {
                if name == "d" {
                    if let Some(e) = self.leibniz()? {
                        return Ok(e);
                    }
                }
                self.ident(name)
            }
            Tok::LParen => {
                let first = self.expr(0)?;
                if self.peek().tok == Tok::Comma {
                    let mut items = vec![first];
                    while self.peek().tok == Tok::Comma {
                        self.bump();
                        items.push(self.expr(0)?);
                    }
                    self.expect(Tok::RParen, "')'")?;
                    Ok(Expr::Tuple(items))
                } else {
                    self.expect(Tok::RParen, "')'")?;
                    self.postfix(first)
                }
            }
            Tok::LBracket => {
                let e = self.list_literal()?;
                self.postfix(e)
            }
            Tok::Pipe => {
                self.abs_depth += 1;
                let inner = self.expr(11)?;
                self.abs_depth -= 1;
                self.expect(Tok::Pipe, "closing '|'")?;
                Ok(Expr::call("abs", vec![inner]))
            }
            Tok::BigOp(g) => self.big_op(g, t.pos),
            Tok::LBrace => self.piecewise(t.pos),
            Tok::Eof => Err(ParseError::new(t.pos, "unexpected end of input")),
            _ => Err(ParseError::new(t.pos, "unexpected token")),
        }
    }

    /// `{c1: v1, c2: v2, d}` after the opening brace: `piece(c1, v1, c2, v2, d)`. A branch without
    /// a colon is a default value when it is last, and `1` under its condition when it is a
    /// comparison (`{x>0}`).
    fn piecewise(&mut self, pos: usize) -> Result<Expr, ParseError> {
        let mut args: Vec<Expr> = Vec::new();
        let mut has_default = false;
        loop {
            if self.peek().tok == Tok::RBrace {
                break;
            }
            let e = self.expr(0)?;
            if self.peek().tok == Tok::Colon {
                self.bump();
                let v = self.expr(0)?;
                args.push(e);
                args.push(v);
            } else if matches!(e, Expr::Rel(..)) || crate::ast::rel_chain(&e).is_some() {
                args.push(e);
                args.push(Expr::Num(1.0));
            } else {
                if self.peek().tok != Tok::RBrace {
                    return Err(ParseError::new(
                        self.peek().pos,
                        "in a piecewise {..} only the last value may have no condition (write condition: value)",
                    ));
                }
                args.push(e);
                has_default = true;
                break;
            }
            match self.peek().tok {
                Tok::Comma => {
                    self.bump();
                }
                Tok::RBrace => break,
                _ => return Err(ParseError::new(self.peek().pos, "expected ',' or '}' in a piecewise {..}")),
            }
        }
        self.expect(Tok::RBrace, "'}'")?;
        if args.is_empty() {
            return Err(ParseError::new(pos, "an empty piecewise {} has no values"));
        }
        let _ = has_default;
        Ok(Expr::Call(crate::ast::PIECE_FN.to_string(), args))
    }

    /// `\\int_a^b body\\,dx`, `\\sum_{n=a}^b body`, `\\prod_{n=a}^b body` (the normaliser left
    /// `GLYPH(lo)(hi)`). An integral's body runs up to its matching `d<var>`; a sum's or product's
    /// body is the following product-level term (`\\sum_{n=1}^3 n+1` is `(sum n) + 1`, as in Desmos).
    fn big_op(&mut self, glyph: char, pos: usize) -> Result<Expr, ParseError> {
        let kind = match glyph {
            '\u{222b}' => "integral",
            '\u{2211}' => "sum",
            _ => "product",
        };
        let limit = |p: &mut Self| -> Result<Option<Expr>, ParseError> {
            p.expect(Tok::LParen, "'('")?;
            if p.peek().tok == Tok::RParen {
                p.bump();
                return Ok(None);
            }
            let e = p.expr(0)?;
            p.expect(Tok::RParen, "')'")?;
            Ok(Some(e))
        };
        let lo = limit(self)?;
        let hi = limit(self)?;
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return Err(ParseError::new(
                pos,
                if glyph == '\u{222b}' {
                    "an integral needs both limits, as in \\int_{0}^{1} f dx (indefinite integrals are not supported)".to_string()
                } else {
                    format!("a {kind} needs both limits, as in \\{}_{{n=1}}^{{5}}", if glyph == '\u{2211}' { "sum" } else { "prod" })
                },
            ));
        };
        if glyph != '\u{222b}' {
            let name = if glyph == '\u{2211}' { "sum" } else { "prod" };
            let Expr::Rel(Rel::Eq, var, start) = lo else {
                return Err(ParseError::new(pos, format!("the lower limit of a {kind} looks like n=1")));
            };
            let Expr::Var(v) = *var else {
                return Err(ParseError::new(pos, format!("the lower limit of a {kind} looks like n=1")));
            };
            let body = self.expr(30)?;
            return Ok(Expr::call(name, vec![body, Expr::Var(v), *start, hi]));
        }
        // integral: find the matching differential
        let start = self.i;
        let (mut depth, mut nested) = (0i32, 0usize);
        let mut found = None;
        for k in start..self.toks.len() {
            match &self.toks[k].tok {
                Tok::LParen | Tok::LBracket => depth += 1,
                Tok::RParen | Tok::RBracket => depth -= 1,
                Tok::BigOp('\u{222b}') if depth == 0 => nested += 1,
                Tok::Ident(d) if d == "d" && depth == 0 => {
                    if let Some(Token { tok: Tok::Ident(v), .. }) = self.toks.get(k + 1) {
                        if nested == 0 {
                            found = Some((k, v.clone()));
                            break;
                        }
                        nested -= 1;
                    }
                }
                Tok::Eof => break,
                _ => {}
            }
            if depth < 0 {
                break;
            }
        }
        let Some((dpos, var)) = found else {
            return Err(ParseError::new(pos, "an integral needs a differential at the end, as in \\int_{0}^{1} f dx"));
        };
        if dpos == start {
            return Err(ParseError::new(pos, "the integral has nothing to integrate before the differential"));
        }
        let mut sub = self.toks[start..dpos].to_vec();
        sub.push(Token { tok: Tok::Eof, pos: self.toks[dpos].pos, space_before: false });
        let mut p = Parser { toks: sub, i: 0, ctx: self.ctx, abs_depth: self.abs_depth, depth: self.depth };
        let body = p.expr(0)?;
        if p.peek().tok != Tok::Eof {
            return Err(ParseError::new(p.peek().pos, "unexpected trailing input in the integrand"));
        }
        self.i = dpos + 2;
        Ok(Expr::call("int", vec![body, Expr::Var(var), lo, hi]))
    }

    /// After `[`: a list literal, a range (`[a...b]`, `[a,b...c]`) or a comprehension
    /// (`[expr for n=source]`).
    fn list_literal(&mut self) -> Result<Expr, ParseError> {
        let mut items = Vec::new();
        if self.peek().tok == Tok::RBracket {
            self.bump();
            return Ok(Expr::List(items));
        }
        items.push(self.expr(11)?);
        match self.peek().tok {
            Tok::For => {
                self.bump();
                let var = match self.bump() {
                    Token { tok: Tok::Ident(n), .. } => n,
                    t => return Err(ParseError::new(t.pos, "expected a loop variable after 'for'")),
                };
                self.expect(Tok::Eq, "'=' after the loop variable")?;
                let source = self.expr(11)?;
                self.expect(Tok::RBracket, "']'")?;
                return Ok(Expr::call("for", vec![items.pop().unwrap(), Expr::Var(var), source]));
            }
            Tok::Ellipsis => {
                self.bump();
                let end = self.expr(11)?;
                self.expect(Tok::RBracket, "']'")?;
                return Ok(Expr::call("range", vec![items.pop().unwrap(), end]));
            }
            _ => {}
        }
        while self.peek().tok == Tok::Comma {
            self.bump();
            items.push(self.expr(11)?);
            if self.peek().tok == Tok::Ellipsis {
                if items.len() != 2 {
                    return Err(ParseError::new(self.peek().pos, "a range looks like [a...b] or [a, b...c]"));
                }
                self.bump();
                let end = self.expr(11)?;
                self.expect(Tok::RBracket, "']'")?;
                let mut it = items.into_iter();
                let (a, b) = (it.next().unwrap(), it.next().unwrap());
                return Ok(Expr::call("range", vec![a, b, end]));
            }
        }
        self.expect(Tok::RBracket, "']'")?;
        Ok(Expr::List(items))
    }

    /// Postfix indexing: a `[` directly after (no whitespace) a variable, call, group, list or
    /// index. Never after a number, so `2[1,2,3]` stays 2 times the list.
    fn postfix(&mut self, mut e: Expr) -> Result<Expr, ParseError> {
        while self.peek().tok == Tok::LBracket && !self.peek().space_before {
            self.bump();
            let first = self.expr(0)?;
            let idx = match self.peek().tok {
                Tok::Ellipsis => {
                    self.bump();
                    let end = self.expr(0)?;
                    Expr::call("range", vec![first, end])
                }
                Tok::Comma => {
                    self.bump();
                    let second = self.expr(0)?;
                    if self.peek().tok != Tok::Ellipsis {
                        return Err(ParseError::new(self.peek().pos, "an index range looks like L[a...b] or L[a, b...c]"));
                    }
                    self.bump();
                    let end = self.expr(0)?;
                    Expr::call("range", vec![first, second, end])
                }
                _ => first,
            };
            self.expect(Tok::RBracket, "']'")?;
            e = Expr::call("index", vec![e, idx]);
        }
        Ok(e)
    }

    fn call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect(Tok::LParen, "'('")?;
        let mut args = Vec::new();
        if self.peek().tok != Tok::RParen {
            args.push(self.expr(11)?);
            while self.peek().tok == Tok::Comma {
                self.bump();
                args.push(self.expr(11)?);
            }
        }
        self.expect(Tok::RParen, "')'")?;
        Ok(args)
    }

    fn ident(&mut self, name: String) -> Result<Expr, ParseError> {
        let e = self.ident_inner(name)?;
        if matches!(e, Expr::Var(_) | Expr::Call(..)) {
            self.postfix(e)
        } else {
            Ok(e)
        }
    }

    /// Leibniz notation after the leading `d`: `d/dx f`, `d^2/dx^2 f`. The operand is a product
    /// term (`d/dx 3x^2 + 1` differentiates `3x^2` only). Not a derivative, so `None` (and the
    /// tokens untouched), unless the shape is exactly `d [^n] / d <letter> [^n]` followed by an
    /// operand.
    fn leibniz(&mut self) -> Result<Option<Expr>, ParseError> {
        let at = |k: usize| self.toks.get(self.i + k).map(|t| &t.tok);
        let mut k = 0;
        let mut order = 1.0;
        if at(k) == Some(&Tok::Caret) {
            match at(k + 1) {
                Some(Tok::Num(v)) if *v >= 1.0 && *v <= 8.0 && v.fract() == 0.0 => order = *v,
                _ => return Ok(None),
            }
            k += 2;
        }
        if at(k) != Some(&Tok::Slash) || at(k + 1) != Some(&Tok::Ident("d".into())) {
            return Ok(None);
        }
        let var = match at(k + 2) {
            Some(Tok::Ident(v)) if !is_builtin_func(v) && v != "d" => v.clone(),
            _ => return Ok(None),
        };
        k += 3;
        if order != 1.0 || at(k) == Some(&Tok::Caret) {
            match (at(k), at(k + 1)) {
                (Some(Tok::Caret), Some(Tok::Num(v))) if *v == order => k += 2,
                _ => return Ok(None),
            }
        }
        match at(k) {
            Some(t) if self.starts_primary(t) || *t == Tok::Minus => {}
            _ => return Ok(None),
        }
        self.i += k;
        let mut e = self.expr(30)?;
        for _ in 0..order as usize {
            e = Expr::call("deriv", vec![e, Expr::Var(var.clone())]);
        }
        Ok(Some(e))
    }

    fn ident_inner(&mut self, name: String) -> Result<Expr, ParseError> {
        let builtin = is_builtin_func(&name);
        let user = self.ctx.functions.contains(&name);
        if !builtin && !user {
            return Ok(Expr::Var(name));
        }
        // f'(a), f''(a): the derivative of f evaluated at a.
        if self.peek().tok == Tok::Prime {
            let mut primes = 0;
            while self.peek().tok == Tok::Prime {
                self.bump();
                primes += 1;
            }
            if primes > 8 {
                return Err(ParseError::new(self.peek().pos, "too many primes"));
            }
            if self.peek().tok != Tok::LParen {
                return Err(ParseError::new(self.peek().pos, "expected '(' after the prime"));
            }
            let mut args = self.call_args()?;
            if args.len() != 1 {
                return Err(ParseError::new(self.peek().pos, "a primed function takes one argument"));
            }
            let x = Expr::var("x");
            let mut f = Expr::Call(name, vec![x.clone()]);
            for _ in 1..primes {
                f = Expr::call("deriv", vec![f, x.clone()]);
            }
            return Ok(Expr::call("deriv", vec![f, x, args.pop().unwrap()]));
        }
        // sin^2 x and sin^-1 x
        if builtin && self.peek().tok == Tok::Caret {
            self.bump();
            let exp = self.expr(40)?;
            let inverse = matches!(&exp, Expr::Neg(inner) if **inner == Expr::Num(1.0));
            let fname = match (inverse, name.as_str()) {
                (true, "sin") => "asin".to_string(),
                (true, "cos") => "acos".to_string(),
                (true, "tan") => "atan".to_string(),
                _ => name.clone(),
            };
            let arg = self.func_arg()?;
            let call = Expr::call(&fname, arg);
            return Ok(if inverse && fname != name { call } else { Expr::bin(BinOp::Pow, call, exp) });
        }
        if self.peek().tok == Tok::LParen {
            let args = self.call_args()?;
            return Ok(Expr::Call(name, args));
        }
        if user {
            return Ok(Expr::Var(name));
        }
        let arg = self.func_arg()?;
        Ok(Expr::Call(name, arg))
    }

    fn func_arg(&mut self) -> Result<Vec<Expr>, ParseError> {
        if self.peek().tok == Tok::LParen {
            return self.call_args();
        }
        let t = self.peek().tok.clone();
        if !self.starts_primary(&t) && t != Tok::Minus {
            return Err(ParseError::new(self.peek().pos, "expected function argument"));
        }
        Ok(vec![self.expr(36)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp::*;

    fn p(s: &str) -> Expr {
        parse(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"))
    }

    fn v(n: &str) -> Expr {
        Expr::var(n)
    }
    fn n(x: f64) -> Expr {
        Expr::num(x)
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(p("1+2*3"), Expr::bin(Add, n(1.0), Expr::bin(Mul, n(2.0), n(3.0))));
        assert_eq!(p("2^3^2"), Expr::bin(Pow, n(2.0), Expr::bin(Pow, n(3.0), n(2.0))));
        assert_eq!(p("-x^2"), Expr::neg(Expr::bin(Pow, v("x"), n(2.0))));
        assert_eq!(p("a-b-c"), Expr::bin(Sub, Expr::bin(Sub, v("a"), v("b")), v("c")));
        assert_eq!(p("2^-x"), Expr::bin(Pow, n(2.0), Expr::neg(v("x"))));
    }

    #[test]
    fn implicit_multiplication() {
        assert_eq!(p("2x"), Expr::bin(Mul, n(2.0), v("x")));
        assert_eq!(p("xy"), Expr::bin(Mul, v("x"), v("y")));
        assert_eq!(p("2x^2"), Expr::bin(Mul, n(2.0), Expr::bin(Pow, v("x"), n(2.0))));
        assert_eq!(p("(x+1)(x-1)"), Expr::bin(Mul, Expr::bin(Add, v("x"), n(1.0)), Expr::bin(Sub, v("x"), n(1.0))));
        assert_eq!(p("2pi"), Expr::bin(Mul, n(2.0), v("pi")));
        assert_eq!(p("2e"), Expr::bin(Mul, n(2.0), v("e")));
        assert_eq!(p("1e3"), n(1000.0));
    }

    #[test]
    fn functions() {
        assert_eq!(p("sin(x)"), Expr::call("sin", vec![v("x")]));
        assert_eq!(p("sin x"), Expr::call("sin", vec![v("x")]));
        assert_eq!(p("sinx"), Expr::call("sin", vec![v("x")]));
        assert_eq!(p("sin x^2"), Expr::call("sin", vec![Expr::bin(Pow, v("x"), n(2.0))]));
        assert_eq!(p("sin^2 x"), Expr::bin(Pow, Expr::call("sin", vec![v("x")]), n(2.0)));
        assert_eq!(p("sin^-1(x)"), Expr::call("asin", vec![v("x")]));
        assert_eq!(p("max(1,2)"), Expr::call("max", vec![n(1.0), n(2.0)]));
        assert_eq!(p("atan2(y,x)"), Expr::call("atan2", vec![v("y"), v("x")]));
        assert_eq!(p("x2"), Expr::bin(Mul, v("x"), n(2.0)));
        assert_eq!(p("sin2x"), Expr::bin(Mul, Expr::call("sin", vec![n(2.0)]), v("x")));
        assert_eq!(p("sqrt(x)+1"), Expr::bin(Add, Expr::call("sqrt", vec![v("x")]), n(1.0)));
    }

    #[test]
    fn absolute_value() {
        assert_eq!(p("|x|"), Expr::call("abs", vec![v("x")]));
        assert_eq!(p("|x-1|+2"), Expr::bin(Add, Expr::call("abs", vec![Expr::bin(Sub, v("x"), n(1.0))]), n(2.0)));
        assert_eq!(p("2|x|"), Expr::bin(Mul, n(2.0), Expr::call("abs", vec![v("x")])));
        assert_eq!(
            p("||x|-1|"),
            Expr::call("abs", vec![Expr::bin(Sub, Expr::call("abs", vec![v("x")]), n(1.0))])
        );
    }

    #[test]
    fn relations_points_lists() {
        assert_eq!(p("y=x^2"), Expr::Rel(Rel::Eq, Box::new(v("y")), Box::new(Expr::bin(Pow, v("x"), n(2.0)))));
        assert_eq!(p("x<=3"), Expr::Rel(Rel::Le, Box::new(v("x")), Box::new(n(3.0))));
        assert_eq!(p("(1,2)"), Expr::Tuple(vec![n(1.0), n(2.0)]));
        assert_eq!(p("[1,2,3]"), Expr::List(vec![n(1.0), n(2.0), n(3.0)]));
        assert!(parse("0<x<5").is_ok(), "comparison chains are supported");
    }

    #[test]
    fn user_function_definitions() {
        let e = p("f(x)=x^2");
        assert_eq!(e, Expr::Rel(Rel::Eq, Box::new(Expr::call("f", vec![v("x")])), Box::new(Expr::bin(Pow, v("x"), n(2.0)))));
        // Without a definition in scope, f(x) is f times (x).
        assert_eq!(p("f(x)"), Expr::bin(Mul, v("f"), v("x")));
        let ctx = ParseCtx::new().with_function("f");
        assert_eq!(parse_with("f(2)", &ctx).unwrap(), Expr::call("f", vec![n(2.0)]));
    }

    #[test]
    fn latex_subset() {
        assert_eq!(p("\\frac{1}{2}x"), p("((1)/(2))x"));
        assert_eq!(p("\\sqrt{x+1}"), p("sqrt(x+1)"));
        assert_eq!(p("x^{2}"), p("x^2"));
        assert_eq!(p("2\\cdot 3"), p("2*3"));
        assert_eq!(p("\\sin\\left(x\\right)"), p("sin(x)"));
        assert_eq!(p("\\left|x\\right|"), p("|x|"));
        assert_eq!(p("\\pi r^{2}"), p("pi r^2"));
        assert_eq!(p("x\\le 3"), p("x<=3"));
        assert_eq!(p("\\operatorname{floor}\\left(x\\right)"), p("floor(x)"));
        assert!(parse("\\bogus{x}").is_err());
    }

    #[test]
    fn errors_carry_positions() {
        let e = parse("1+").unwrap_err();
        assert_eq!(e.pos, 2);
        assert!(parse("(1+2").is_err());
        assert!(parse("1 $ 2").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn hostile_nesting_is_an_error_not_a_stack_overflow() {
        assert!(parse(&"(".repeat(100_000)).is_err());
        assert!(parse(&"[".repeat(100_000)).is_err());
        let deep = format!("{}1{}", "(".repeat(500), ")".repeat(500));
        assert!(parse(&deep).is_err());
        let ok = format!("{}1{}", "(".repeat(50), ")".repeat(50));
        assert!(parse(&ok).is_ok());
        assert!(parse(&"-".repeat(100_000)).is_err());
    }

    #[test]
    fn ranges() {
        assert_eq!(p("[1...10]"), Expr::call("range", vec![n(1.0), n(10.0)]));
        assert_eq!(p("[1.5...10]"), Expr::call("range", vec![n(1.5), n(10.0)]));
        assert_eq!(p("[1,3...11]"), Expr::call("range", vec![n(1.0), n(3.0), n(11.0)]));
        assert_eq!(p("[10, 8 ... 0]"), Expr::call("range", vec![n(10.0), n(8.0), n(0.0)]));
        assert_eq!(p("[-1...n]"), Expr::call("range", vec![Expr::neg(n(1.0)), v("n")]));
        assert_eq!(p("[1\\ldots 5]"), p("[1...5]"));
        assert_eq!(p("[1\\cdots 5]"), p("[1...5]"));
        assert_eq!(p("[1\\dots 5]"), p("[1...5]"));
        assert!(parse("[1,2,3...9]").is_err());
        assert!(parse("[1...]").is_err());
        assert!(parse("[1..5]").is_err());
        assert_eq!(p("[1,2,3]"), Expr::List(vec![n(1.0), n(2.0), n(3.0)]));
    }

    #[test]
    fn indexing() {
        let l = || v("L");
        assert_eq!(p("L[3]"), Expr::call("index", vec![l(), n(3.0)]));
        assert_eq!(p("L[2...4]"), Expr::call("index", vec![l(), Expr::call("range", vec![n(2.0), n(4.0)])]));
        assert_eq!(p("L[1,3...9]"), Expr::call("index", vec![l(), Expr::call("range", vec![n(1.0), n(3.0), n(9.0)])]));
        assert_eq!(p("L[[1,3]]"), Expr::call("index", vec![l(), Expr::List(vec![n(1.0), n(3.0)])]));
        assert_eq!(p("L[L>3]"), Expr::call("index", vec![l(), Expr::Rel(Rel::Gt, Box::new(l()), Box::new(n(3.0)))]));
        assert_eq!(
            p("L[L^2<10]"),
            Expr::call("index", vec![l(), Expr::Rel(Rel::Lt, Box::new(Expr::bin(Pow, l(), n(2.0))), Box::new(n(10.0)))])
        );
        assert_eq!(p("L[i+1]"), Expr::call("index", vec![l(), Expr::bin(Add, v("i"), n(1.0))]));
        // after a `)`, a call and a `]`
        assert_eq!(p("(L)[1]"), Expr::call("index", vec![l(), n(1.0)]));
        assert_eq!(p("sort(L)[1]"), Expr::call("index", vec![Expr::call("sort", vec![l()]), n(1.0)]));
        assert_eq!(p("[1,2,3][2]"), Expr::call("index", vec![Expr::List(vec![n(1.0), n(2.0), n(3.0)]), n(2.0)]));
        assert_eq!(
            p("M[1][2]"),
            Expr::call("index", vec![Expr::call("index", vec![v("M"), n(1.0)]), n(2.0)])
        );
        // not after a number or after whitespace: implicit multiplication
        assert_eq!(p("2[1,2,3]"), Expr::bin(Mul, n(2.0), Expr::List(vec![n(1.0), n(2.0), n(3.0)])));
        assert_eq!(p("L [1,2]"), Expr::bin(Mul, v("L"), Expr::List(vec![n(1.0), n(2.0)])));
        assert!(parse("L[1").is_err());
        assert!(parse("L[]").is_err());
    }

    #[test]
    fn comprehension() {
        assert_eq!(
            p("[n^2 for n=[1...5]]"),
            Expr::call("for", vec![Expr::bin(Pow, v("n"), n(2.0)), v("n"), Expr::call("range", vec![n(1.0), n(5.0)])])
        );
        assert_eq!(
            p("[2k for k=L]"),
            Expr::call("for", vec![Expr::bin(Mul, n(2.0), v("k")), v("k"), v("L")])
        );
        assert_eq!(p("[x for x=L[L>0]]"), p("[x for x=L[L>0]]"));
        assert!(parse("[x for]").is_err());
        assert!(parse("[x for 3=L]").is_err());
        assert!(parse("[x for x L]").is_err());
        // `for` is only a keyword inside a list literal; elsewhere it splits into letters.
        assert_eq!(p("for"), Expr::bin(Mul, Expr::bin(Mul, v("f"), v("o")), v("r")));
    }

    #[test]
    fn list_function_names_and_lexing_tradeoffs() {
        assert_eq!(p("mean(L)"), Expr::call("mean", vec![v("L")]));
        assert_eq!(p("normalcdf(-1,1,0,1)").to_string(), "normalcdf(-1, 1, 0, 1)");
        assert_eq!(p("sort(L,K)"), Expr::call("sort", vec![v("L"), v("K")]));
        // Existing identifiers do not regress.
        assert_eq!(p("xy"), Expr::bin(Mul, v("x"), v("y")));
        assert_eq!(p("ax"), Expr::bin(Mul, v("a"), v("x")));
        assert_eq!(p("a*b"), Expr::bin(Mul, v("a"), v("b")));
        // Known trade-off: a builtin name is a function wherever a run of letters starts with
        // it, so `total` is the function (not t*o*t*a*l) and `vara` is var(a).
        assert_eq!(p("total(L)"), Expr::call("total", vec![v("L")]));
        assert!(parse("a*total").is_err(), "`total` is the function, so a bare `total` has no argument");
        assert_eq!(p("vara"), Expr::call("var", vec![v("a")]));
    }

    #[test]
    fn factorial_postfix() {
        let f = |e: Expr| Expr::call("factorial", vec![e]);
        assert_eq!(p("n!"), f(v("n")));
        assert_eq!(p("5!"), f(n(5.0)));
        assert_eq!(p("(n+1)!"), f(Expr::bin(Add, v("n"), n(1.0))));
        assert_eq!(p("factorial(n)"), f(v("n")));
        assert_eq!(p("n!!"), f(f(v("n"))));
        assert_eq!(p("sin(x)!"), f(Expr::call("sin", vec![v("x")])));
        // `!` binds tighter than `*`, `/`, `^` and unary minus (Desmos): 2^3! = 2^(3!)
        assert_eq!(p("2^3!"), Expr::bin(Pow, n(2.0), f(n(3.0))));
        assert_eq!(p("x^n!"), Expr::bin(Pow, v("x"), f(v("n"))));
        assert_eq!(p("n!^2"), Expr::bin(Pow, f(v("n")), n(2.0)));
        assert_eq!(p("-n!"), Expr::neg(f(v("n"))));
        assert_eq!(p("2n!"), Expr::bin(Mul, n(2.0), f(v("n"))));
        assert_eq!(p("x^n/n!"), Expr::bin(Div, Expr::bin(Pow, v("x"), v("n")), f(v("n"))));
        assert_eq!(p("|x|!"), f(Expr::call("abs", vec![v("x")])));
        assert_eq!(p("sum(x^n/n!,n,0,6)").called_functions().len(), 2);
        // `!` directly followed by `=` is the relation `!=`; with a space it is a factorial
        assert_eq!(p("x!=3"), Expr::Rel(Rel::Ne, Box::new(v("x")), Box::new(n(3.0))));
        assert_eq!(p("x! = 3"), Expr::Rel(Rel::Eq, Box::new(f(v("x"))), Box::new(n(3.0))));
        assert_eq!(p("(x!)=3"), Expr::Rel(Rel::Eq, Box::new(f(v("x"))), Box::new(n(3.0))));
        assert!(parse("!x").is_err());
        assert_eq!(p("nCr(5,2)"), Expr::call("nCr", vec![n(5.0), n(2.0)]));
        assert_eq!(p("nPr(n,k)"), Expr::call("nPr", vec![v("n"), v("k")]));
        assert_eq!(p("\\binom{n}{k}"), Expr::call("nCr", vec![v("n"), v("k")]));
        assert_eq!(p("\\binom{n+1}{2}"), Expr::call("nCr", vec![Expr::bin(Add, v("n"), n(1.0)), n(2.0)]));
    }

    #[test]
    fn chained_comparisons() {
        let rel = |r: Rel, a: Expr, b: Expr| Expr::Rel(r, Box::new(a), Box::new(b));
        assert_eq!(
            p("0<=y<=x^2"),
            Expr::call(
                "and",
                vec![
                    rel(Rel::Le, n(0.0), v("y")),
                    rel(Rel::Le, v("y"), Expr::bin(Pow, v("x"), n(2.0))),
                ]
            )
        );
        assert_eq!(p("a<x<b"), Expr::call("and", vec![rel(Rel::Lt, v("a"), v("x")), rel(Rel::Lt, v("x"), v("b"))]));
        // mixed directions and three comparisons
        assert!(matches!(&p("1>x<=2"), Expr::Call(n, a) if n == "and" && a.len() == 2));
        assert!(matches!(&p("0<x<y<=3"), Expr::Call(n, a) if n == "and" && a.len() == 3));
        assert_eq!(p("1<x^2+y^2<=4").free_vars().len(), 2);
        // single comparisons are unchanged
        assert_eq!(p("y<=x"), rel(Rel::Le, v("y"), v("x")));
        // rejected: `=` inside a chain, too many parts
        let e = parse("0<y=x").unwrap_err();
        assert!(e.msg.contains("'='"), "{e}");
        assert!(parse("0=y<x").is_err());
        assert!(parse("0<x<y<z<5").unwrap_err().msg.contains("at most three"));
        assert!(parse("y~x<1<2").is_err());
    }

    // ---- MathLive-emitted LaTeX (what the web shell actually feeds the engine)

    fn same(latex: &str, text: &str) {
        let a = parse(latex).unwrap_or_else(|e| panic!("latex {latex:?}: {e}"));
        let b = parse(text).unwrap_or_else(|e| panic!("text {text:?}: {e}"));
        assert_eq!(a, b, "{latex:?} vs {text:?}");
    }

    #[test]
    fn mathlive_square_brackets_are_lists() {
        // MathLive writes a typed `[` / `]` as \lbrack / \rbrack
        same(r"\lbrack(1,2),(3,4)\rbrack", "[(1,2),(3,4)]");
        same(r"\lbrack1,2,3\rbrack", "[1,2,3]");
        same(r"x\lbrack2\rbrack", "x[2]");
    }

    #[test]
    fn mathlive_integrals() {
        same(r"\int_0^2x^2dx", "int(x^2, x, 0, 2)");
        same(r"\int_{0}^{2}x^{2}\,dx", "int(x^2, x, 0, 2)");
        same(r"\int_{0}^{2}x^{2}\mathrm{d}x", "int(x^2, x, 0, 2)");
        same(r"\int_a^b f\,dx", "int(f, x, a, b)");
        same(r"\int_{a}^{b}f\left(x\right)dx", "int(f*(x), x, a, b)");
        same(r"\int_{0}^{\pi}\sin\left(x\right)\,dx", "int(sin(x), x, 0, pi)");
        same(r"\int_{1}^{3}\frac{1}{x}dx", "int(1/x, x, 1, 3)");
        same(r"\int_{0}^{1}\left(x+1\right)^{2}\mathrm{d}x", "int((x+1)^2, x, 0, 1)");
        same(r"\int_0^1 t^2\,dt+1", "int(t^2, t, 0, 1)+1");
        same(r"2\int_0^1x\,dx", "2*int(x, x, 0, 1)");
        same(r"\int_0^1\int_0^2xy\,dx\,dy", "int(int(x*y, x, 0, 2), y, 0, 1)");
        same(r"\int_{0}^{\theta}\sin\left(u\right)\,du", "int(sin(u), u, 0, theta)");
        same(r"y=\int_0^x t\,dt", "y=int(t, t, 0, x)");
        assert_eq!(p(r"\int_0^2x^2dx"), Expr::call("int", vec![Expr::bin(Pow, v("x"), n(2.0)), v("x"), n(0.0), n(2.0)]));
    }

    #[test]
    fn latex_call_forms_for_big_operators() {
        same(r"\int(x^2,x,0,2)", "int(x^2,x,0,2)");
        same(r"\sum(x^n,n,0,6)", "sum(x^n,n,0,6)");
        same(r"\prod(k,k,1,4)", "prod(k,k,1,4)");
        same(r"\int\left(x^2,x,0,2\right)", "int(x^2,x,0,2)");
        same(r"\sum\left(x^{n},n,0,6\right)", "sum(x^n,n,0,6)");
        same(r"\prod \left( k , k , 1 , 4 \right)", "prod(k,k,1,4)");
        same(r"\int (x^2, x, 0, 2)", "int(x^2,x,0,2)");
        same(r"\int(x^2,x,0,2)+1", "int(x^2,x,0,2)+1");
        same(r"2\int(x,x,0,1)", "2*int(x,x,0,1)");
        same(r"\int(\sin\left(x\right),x,0,\pi)", "int(sin(x),x,0,pi)");
        same(r"\sum(\prod(k,k,1,n),n,1,3)", "sum(prod(k,k,1,n),n,1,3)");
        same(r"\int(\int(x*y,x,0,2),y,0,1)", "int(int(x*y,x,0,2),y,0,1)");
        // limit forms and indefinite integrals are unchanged
        same(r"\int_0^1\left(x+1\right)dx", "int(x+1,x,0,1)");
        assert!(parse(r"\int(x+1)dx").unwrap_err().msg.contains("integral"));
    }

    #[test]
    fn mathlive_integral_errors() {
        for bad in [r"\int x\,dx", r"\int_0^1 x", r"\int_0 x\,dx", r"\int_0^1\,dx"] {
            let e = parse(bad).unwrap_err();
            assert!(e.msg.contains("integral"), "{bad}: {e}");
        }
    }

    #[test]
    fn mathlive_sums_and_products() {
        same(r"\sum_{n=0}^{6}\frac{x^{n}}{n!}", "sum(x^n/n!, n, 0, 6)");
        same(r"\sum_{n=1}^{5}n^2", "sum(n^2, n, 1, 5)");
        same(r"\sum_{n=1}^{3}n+1", "sum(n, n, 1, 3)+1");
        same(r"\sum_{n=1}^{3}2n\cdot x", "sum(2*n*x, n, 1, 3)");
        same(r"\sum_{n=1}^{3}\left(n+1\right)", "sum(n+1, n, 1, 3)");
        same(r"\prod_{k=1}^{4}k", "prod(k, k, 1, 4)");
        same(r"\prod_{n=a}^{b}\left(n+1\right)", "prod(n+1, n, a, b)");
        same(r"\sum_{k=1}^{4}k^2\,", "sum(k^2, k, 1, 4)");
        assert!(parse(r"\sum n").is_err());
        assert!(parse(r"\sum_{n}^{3}n").is_err());
    }

    #[test]
    fn latex_function_after_letter_is_a_product() {
        let prod = |a: &str, f: &str, arg: &str| Expr::bin(Mul, v(a), Expr::call(f, vec![arg.parse::<f64>().map(Expr::num).unwrap_or_else(|_| v(arg))]));
        assert_eq!(p(r"a\sin\left(x\right)"), prod("a", "sin", "x"));
        assert_eq!(p(r"a\sin(x)"), prod("a", "sin", "x"));
        assert_eq!(p(r"a\cos x"), prod("a", "cos", "x"));
        assert_eq!(p(r"b\sin x"), prod("b", "sin", "x"));
        assert_eq!(p(r"a\ln x"), prod("a", "ln", "x"));
        assert_eq!(p(r"y=a\sin\left(x\right)"), Expr::Rel(Rel::Eq, Box::new(v("y")), Box::new(prod("a", "sin", "x"))));
        assert_eq!(p(r"k\tan\left(x\right)"), prod("k", "tan", "x"));
        assert_eq!(p(r"a\operatorname{sin}(x)"), prod("a", "sin", "x"));
        // real inverse trig keeps working in every spelling
        for s in [r"\arcsin\left(x\right)", "asin(x)", "arcsin(x)", r"\operatorname{arcsin}\left(x\right)"] {
            assert_eq!(p(s), Expr::call("asin", vec![v("x")]), "{s}");
        }
        assert_eq!(p(r"a\arcsin x"), Expr::bin(Mul, v("a"), Expr::call("asin", vec![v("x")])));
        assert_eq!(p(r"x\pi"), Expr::bin(Mul, v("x"), v("pi")));
        assert_eq!(p(r"a\theta"), Expr::bin(Mul, v("a"), v("theta")));
    }

    #[test]
    fn mathlive_leibniz() {
        same(r"\frac{d}{dx}x^2", "deriv(x^2, x)");
        same(r"\frac{d}{dx}\left(x^3\right)", "deriv(x^3, x)");
        same(r"\frac{d^2}{dx^2}\sin x", "deriv(deriv(sin(x), x), x)");
        same(r"\frac{d}{dx}\sin\left(x\right)", "deriv(sin(x), x)");
    }

    #[test]
    fn mathlive_relation_chains() {
        let chain = p("0<=y<=x^2");
        for s in [
            r"0\le y\le x^2",
            r"0\leq y\leq x^{2}",
            r"0\le y\le x^{2}",
            r"x^2\ge y\ge0",
            r"x^{2}\geq y\geq 0",
            "0<=y<=x^2",
        ] {
            let e = p(s);
            assert!(crate::ast::rel_chain(&e).is_some(), "{s}: {e:?}");
        }
        assert_eq!(p(r"0\le y\le x^2"), chain);
        assert_eq!(p(r"0\leq y\leq x^{2}"), chain);
        assert!(crate::ast::rel_chain(&p(r"1<x^2+y^2\le4")).is_some());
        assert_eq!(p(r"y\ge x"), p("y>=x"));
        assert_eq!(p(r"y\geq x"), p("y>=x"));
        assert_eq!(p(r"y\le x"), p("y<=x"));
        // `\ne` used to be rejected; it is the relation `!=` now
        assert_eq!(p(r"x\ne 2"), p("x!=2"));
    }
}
