//! Safe WGSL emission from a compiled [`Program`].
//!
//! Security model: the emitter consumes ONLY a `Program` (closed opcode set plus f64 constants).
//! Every piece of emitted text comes from a fixed table or from a formatter: inputs are `v0..vN`,
//! temporaries are `t0..tN`, constants are validated finite and printed by [`format_f32`]. No
//! user-controlled string (variable names, function names from the source expression) is ever
//! copied into the output. The one caller-supplied string, the emitted function's own name, is
//! validated against a strict identifier grammar and a reserved-word list.
//!
//! Non-finite constants (NaN, inf, or values that overflow f32) are rejected with
//! [`WgslError::NonFiniteConstant`]. Runtime NaN/inf produced by the helpers use bitcasts.
//!
//! Semantics follow `Program::eval`: degree-mode trig multiplies by the exact `to_rad()` literal
//! (inverse trig divides), `round` is half-away-from-zero, `mod` is floored, `sign(0) = 0`, and
//! `pow` follows `f64::powf` for negative bases with integer exponents. Results are f32, so they
//! match the CPU only to f32 precision. Shader compilers may assume no NaN/inf (fast-math).

use crate::compile::{Angle, Op, Program, F1, F2};
use std::fmt::Write;

/// Helper functions required by emitted code. Contains no comments and no user data.
pub const PRELUDE: &str = "\
fn am_nan() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn am_inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn am_pow(a: f32, b: f32) -> f32 {
    if (a != a || b != b) {
        return am_nan();
    }
    if (b == 0.0) {
        return 1.0;
    }
    if (a == 0.0) {
        if (b > 0.0) {
            return 0.0;
        }
        return am_inf();
    }
    if (a > 0.0) {
        return pow(a, b);
    }
    if (floor(b) == b) {
        let r = pow(-a, b);
        if (fract(b * 0.5) != 0.0) {
            return -r;
        }
        return r;
    }
    return am_nan();
}

fn am_cbrt(x: f32) -> f32 {
    if (x == 0.0 || x != x) {
        return x;
    }
    var s = 1.0;
    if (x < 0.0) {
        s = -1.0;
    }
    return s * exp2(log2(abs(x)) / 3.0);
}

fn am_sign0(x: f32) -> f32 {
    if (x != x) {
        return x;
    }
    if (x > 0.0) {
        return 1.0;
    }
    if (x < 0.0) {
        return -1.0;
    }
    return 0.0;
}

fn am_round_away(x: f32) -> f32 {
    let ax = abs(x);
    let f = floor(ax);
    var r = f;
    if (ax - f >= 0.5) {
        r = f + 1.0;
    }
    if (x < 0.0) {
        return -r;
    }
    return r;
}

fn am_mod_floor(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn am_log10(x: f32) -> f32 {
    return log2(x) * 0.30103;
}

fn am_gamma_pos(z: f32) -> f32 {
    let x = z - 1.0;
    let t = x + 7.5;
    var a: f32 = 0.99999999999980993;
    a += 676.5203681218851 / (x + 1.0);
    a += -1259.1392167224028 / (x + 2.0);
    a += 771.32342877765313 / (x + 3.0);
    a += -176.61502916214059 / (x + 4.0);
    a += 12.507343278686905 / (x + 5.0);
    a += -0.13857109526572012 / (x + 6.0);
    a += 9.9843695780195716e-6 / (x + 7.0);
    a += 1.5056327351493116e-7 / (x + 8.0);
    return 2.5066282746310002 * exp((x + 0.5) * log(t) - t) * a;
}

fn am_fact(x: f32) -> f32 {
    if (x != x) {
        return x;
    }
    if (x == floor(x)) {
        if (x < 0.0) {
            return am_nan();
        }
        if (x > 34.0) {
            return am_inf();
        }
        var r: f32 = 1.0;
        for (var i: f32 = 2.0; i <= x; i += 1.0) {
            r = r * i;
        }
        return r;
    }
    let z = x + 1.0;
    if (z < 0.5) {
        return 3.14159265 / (sin(3.14159265 * z) * am_gamma_pos(1.0 - z));
    }
    return am_gamma_pos(z);
}

fn am_choose(n: f32, k: f32) -> f32 {
    if (n != n || k != k || n != floor(n) || k != floor(k) || n < 0.0) {
        return am_nan();
    }
    if (k < 0.0 || k > n) {
        return 0.0;
    }
    let m = min(k, n - k);
    if (m > 256.0) {
        return am_inf();
    }
    var r: f32 = 1.0;
    for (var i: f32 = 1.0; i <= m; i += 1.0) {
        r = r * (n - m + i) / i;
    }
    return r;
}

fn am_perm(n: f32, k: f32) -> f32 {
    if (n != n || k != k || n != floor(n) || k != floor(k) || n < 0.0) {
        return am_nan();
    }
    if (k < 0.0 || k > n) {
        return 0.0;
    }
    if (k > 256.0) {
        return am_inf();
    }
    var r: f32 = 1.0;
    for (var i: f32 = 0.0; i < k; i += 1.0) {
        r = r * (n - i);
    }
    return r;
}

// Abramowitz and Stegun 7.1.26: absolute error below 1.5e-7, enough for f32 drawing.
fn am_erf(x: f32) -> f32 {
    let s = select(1.0, -1.0, x < 0.0);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-a * a));
}

fn am_gcd(a: f32, b: f32) -> f32 {
    if (a != a || b != b || a != floor(a) || b != floor(b)) {
        return am_nan();
    }
    var p: f32 = abs(a);
    var q: f32 = abs(b);
    for (var i: i32 = 0; i < 64; i += 1) {
        if (q == 0.0) {
            break;
        }
        let r = p - q * floor(p / q);
        p = q;
        q = r;
    }
    return p;
}

fn am_lcm(a: f32, b: f32) -> f32 {
    let g = am_gcd(a, b);
    if (g != g) {
        return am_nan();
    }
    if (g == 0.0) {
        return 0.0;
    }
    return abs(a / g * b);
}

";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgslError {
    /// Function name failed validation (grammar, length, reserved word).
    InvalidName,
    /// The same function name was given twice to `emit_module`.
    DuplicateName,
    /// A constant was NaN, infinite, or overflows f32.
    NonFiniteConstant,
    /// Malformed program (stack underflow, leftover values, bad `Load` index, empty).
    BadProgram(&'static str),
    /// A construct with no shader form (piecewise).
    Unsupported(&'static str),
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::InvalidName => write!(f, "invalid WGSL function name"),
            WgslError::DuplicateName => write!(f, "duplicate WGSL function name"),
            WgslError::NonFiniteConstant => write!(f, "constant is not finite in f32"),
            WgslError::BadProgram(m) => write!(f, "malformed program: {m}"),
            WgslError::Unsupported(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for WgslError {}

const RESERVED: &[&str] = &[
    "alias", "break", "case", "const", "const_assert", "continue", "continuing", "default",
    "diagnostic", "discard", "else", "enable", "false", "fn", "for", "if", "let", "loop",
    "override", "requires", "return", "struct", "switch", "true", "var", "while", "f32", "f16",
    "i32", "u32", "bool", "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4", "array", "ptr",
    "atomic", "sampler", "texture_2d", "bitcast", "select", "sin", "cos", "tan", "asin", "acos",
    "atan", "atan2", "sinh", "cosh", "tanh", "exp", "exp2", "log", "log2", "sqrt", "abs",
    "floor", "ceil", "round", "fract", "pow", "min", "max", "sign", "clamp", "mix", "step",
    "main", "vs_main", "fs_main", "NULL", "auto", "do", "enum", "class", "new", "this", "null",
];

fn is_reserved_pattern(n: &str) -> bool {
    let numbered = |p: char| {
        n.strip_prefix(p).is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
    };
    n.starts_with("am_") || n.starts_with("__") || n == "_" || numbered('v') || numbered('t')
}

/// Validates an emitted function name: `^[A-Za-z_][A-Za-z0-9_]{0,40}$`, not reserved.
pub fn validate_name(name: &str) -> Result<(), WgslError> {
    let b = name.as_bytes();
    let ok = !b.is_empty()
        && b.len() <= 41
        && (b[0].is_ascii_alphabetic() || b[0] == b'_')
        && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_');
    if !ok || RESERVED.contains(&name) || is_reserved_pattern(name) {
        return Err(WgslError::InvalidName);
    }
    Ok(())
}

/// Formats a constant as an f32 WGSL literal: always float-typed (has '.' or an exponent and an
/// `f` suffix), negative values parenthesised, round-trips through f32.
pub fn format_f32(c: f64) -> Result<String, WgslError> {
    let f = c as f32;
    if !c.is_finite() || !f.is_finite() {
        return Err(WgslError::NonFiniteConstant);
    }
    let mut s = format!("{f:?}");
    if !s.contains(['.', 'e', 'E']) {
        s.push_str(".0");
    }
    s.push('f');
    if f.is_sign_negative() {
        s = format!("({s})");
    }
    Ok(s)
}

fn f1_expr(f: F1, a: &str, angle: Angle) -> Result<String, WgslError> {
    let k = format_f32(angle.to_rad())?;
    let deg = angle == Angle::Deg;
    let scaled = |name: &str| {
        if deg { format!("{name}({a} * {k})") } else { format!("{name}({a})") }
    };
    let unscaled = |name: &str| {
        if deg { format!("{name}({a}) / {k}") } else { format!("{name}({a})") }
    };
    Ok(match f {
        F1::Sin => scaled("sin"),
        F1::Cos => scaled("cos"),
        F1::Tan => scaled("tan"),
        F1::Asin => unscaled("asin"),
        F1::Acos => unscaled("acos"),
        F1::Atan => unscaled("atan"),
        F1::Sinh => format!("sinh({a})"),
        F1::Cosh => format!("cosh({a})"),
        F1::Tanh => format!("tanh({a})"),
        F1::Asinh => format!("asinh({a})"),
        F1::Acosh => format!("acosh({a})"),
        F1::Atanh => format!("atanh({a})"),
        F1::Erf => format!("am_erf({a})"),
        F1::Exp => format!("exp({a})"),
        F1::Ln => format!("log({a})"),
        F1::Log10 => format!("am_log10({a})"),
        F1::Sqrt => format!("sqrt({a})"),
        F1::Cbrt => format!("am_cbrt({a})"),
        F1::Abs => format!("abs({a})"),
        F1::Floor => format!("floor({a})"),
        F1::Ceil => format!("ceil({a})"),
        F1::Round => format!("am_round_away({a})"),
        F1::Sign => format!("am_sign0({a})"),
        F1::Factorial => format!("am_fact({a})"),
    })
}

fn f2_expr(f: F2, a: &str, b: &str) -> String {
    match f {
        F2::Atan2 => format!("atan2({a}, {b})"),
        F2::Min => format!("min({a}, {b})"),
        F2::Max => format!("max({a}, {b})"),
        F2::Mod => format!("am_mod_floor({a}, {b})"),
        F2::Choose => format!("am_choose({a}, {b})"),
        F2::Perm => format!("am_perm({a}, {b})"),
        F2::Gcd => format!("am_gcd({a}, {b})"),
        F2::Lcm => format!("am_lcm({a}, {b})"),
    }
}

/// Emits `fn <name>(v0: f32, ...) -> f32 { ... }` by simulating the postfix stack.
/// Requires [`PRELUDE`] to be present in the same module.
pub fn emit_function(name: &str, p: &Program) -> Result<String, WgslError> {
    validate_name(name)?;
    let nvars = p.vars.len();
    let mut out = String::new();
    let params: Vec<String> = (0..nvars).map(|i| format!("v{i}: f32")).collect();
    let _ = writeln!(out, "fn {name}({}) -> f32 {{", params.join(", "));
    let mut stack: Vec<usize> = Vec::new();
    let underflow = WgslError::BadProgram("stack underflow");
    for (n, op) in p.ops.iter().enumerate() {
        let rhs = match *op {
            Op::Const(c) => format_f32(c)?,
            Op::Load(i) => {
                if i >= nvars {
                    return Err(WgslError::BadProgram("load index out of range"));
                }
                format!("v{i}")
            }
            Op::Neg => {
                let a = stack.pop().ok_or(underflow.clone())?;
                format!("-t{a}")
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow => {
                let b = stack.pop().ok_or(underflow.clone())?;
                let a = stack.pop().ok_or(underflow.clone())?;
                match op {
                    Op::Add => format!("t{a} + t{b}"),
                    Op::Sub => format!("t{a} - t{b}"),
                    Op::Mul => format!("t{a} * t{b}"),
                    Op::Div => format!("t{a} / t{b}"),
                    _ => format!("am_pow(t{a}, t{b})"),
                }
            }
            Op::F1(f) => {
                let a = stack.pop().ok_or(underflow.clone())?;
                f1_expr(f, &format!("t{a}"), p.angle)?
            }
            Op::F2(f) => {
                let b = stack.pop().ok_or(underflow.clone())?;
                let a = stack.pop().ok_or(underflow.clone())?;
                f2_expr(f, &format!("t{a}"), &format!("t{b}"))
            }
            Op::Stat(_) => {
                return Err(WgslError::BadProgram("statistical distribution functions are not supported on the GPU"))
            }
            Op::Reduce(..) => {
                return Err(WgslError::BadProgram("integrals, sums and products are not supported on the GPU"))
            }
            Op::Cmp(..) | Op::And | Op::Piece { .. } => {
                return Err(WgslError::Unsupported("piecewise {..} is not supported on the GPU (NaN gaps are unreliable in shaders)"))
            }
        };
        let _ = writeln!(out, "    let t{n} = {rhs};");
        stack.push(n);
    }
    match stack.as_slice() {
        [r] => {
            let _ = writeln!(out, "    return t{r};\n}}\n");
            Ok(out)
        }
        [] => Err(WgslError::BadProgram("empty program")),
        _ => Err(WgslError::BadProgram("leftover values on stack")),
    }
}

/// Emits [`PRELUDE`] followed by every function. Names must be valid and unique.
pub fn emit_module(functions: &[(&str, &Program)]) -> Result<String, WgslError> {
    let mut out = String::from(PRELUDE);
    let mut seen: Vec<&str> = Vec::new();
    for (name, p) in functions {
        if seen.contains(name) {
            return Err(WgslError::DuplicateName);
        }
        seen.push(name);
        out.push_str(&emit_function(name, p)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile;
    use crate::parse::parse;

    const CORPUS: &[&str] = &[
        "x^2+y", "x^3-2x+y^2", "sin(x)*cos(y)", "sin(x)+sin(y)^2", "1/x", "x/(y+3)",
        "sqrt(x)+y", "sqrt(x^2+y^2)-2", "abs(x)-y", "exp(x)/(1+y^2)", "tan(x)", "ln(x)+y",
        "x^y", "x^2.5", "atan(x*y)", "floor(x)+y", "ceil(x*y)", "(x+1)(x-1)", "sin(1/x)",
        "min(x,y)*max(x,y)", "mod(x,3)+y", "sinh(x)-cosh(y)", "tanh(x*y)",
        "asin(x/5)+acos(y/5)", "x^-2", "round(x)+sign(y)", "cbrt(x)*y", "log(x)+y",
        "sec(x)+csc(y)+cot(x)", "x*y/(x^2+y^2+1)", "2^x", "x^(x^y)",
        "-3*x - -2", "2e-8*x+1e30", "x+y+z", "5", "x!+y", "(x+y)!/2", "nCr(x,y)+nPr(x,2)",
    ];

    fn prog(src: &str, vars: &[&str], angle: Angle) -> Program {
        compile(&parse(src).unwrap(), vars, angle).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    fn check_naga(src: &str) {
        let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }

    #[test]
    fn prelude_valid() {
        check_naga(PRELUDE);
    }

    #[test]
    fn corpus_validates() {
        for angle in [Angle::Rad, Angle::Deg] {
            for src in CORPUS {
                let p = prog(src, &["x", "y", "z"], angle);
                let m = emit_module(&[("f", &p)]).unwrap();
                check_naga(&m);
                let p = prog(src, &["x", "y", "z"][..if src.contains('z') { 3 } else { 2 }], angle);
                check_naga(&emit_module(&[("g", &p)]).unwrap());
            }
        }
        let p = prog("5", &[], Angle::Rad);
        check_naga(&emit_module(&[("c", &p)]).unwrap());
    }

    #[test]
    fn stat_functions_are_rejected_for_the_gpu() {
        let p = prog("normalpdf(x,0,1)", &["x"], Angle::Rad);
        let e = emit_function("f", &p).unwrap_err();
        assert!(e.to_string().contains("GPU"), "{e}");
    }

    #[test]
    fn atan2_handbuilt() {
        let ops = vec![Op::Load(0), Op::Load(1), Op::F2(F2::Atan2)];
        let p = Program { ops, vars: vec!["y".into(), "x".into()], angle: Angle::Rad, subs: vec![] };
        let m = emit_module(&[("f", &p)]).unwrap();
        assert!(m.contains("atan2(t0, t1)"));
        check_naga(&m);
    }

    #[test]
    fn multi_function_module() {
        let a = prog("x+1", &["x"], Angle::Rad);
        let b = prog("sin(x)", &["x"], Angle::Deg);
        check_naga(&emit_module(&[("fa", &a), ("fb", &b)]).unwrap());
        assert_eq!(emit_module(&[("fa", &a), ("fa", &b)]), Err(WgslError::DuplicateName));
    }

    /// Words allowed in an emitted function body.
    fn body_ok(body: &str, fname: &str) {
        const OK: &[&str] = &[
            "fn", "let", "return", "f32", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "sqrt", "abs", "floor",
            "ceil", "min", "max", "am_pow", "am_cbrt", "am_sign0", "am_round_away", "am_mod_floor",
            "am_log10", "am_fact", "am_choose", "am_perm", "am_erf", "am_gcd", "am_lcm",
        ];
        let b: Vec<char> = body.chars().collect();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c.is_ascii_digit() {
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    i += 1;
                }
                if i < b.len() && b[i] == 'e' {
                    i += 1;
                    if i < b.len() && (b[i] == '-' || b[i] == '+') {
                        i += 1;
                    }
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && b[i] == 'f' {
                    i += 1;
                }
            } else if c.is_ascii_alphabetic() || c == '_' {
                let s = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                let w: String = b[s..i].iter().collect();
                let num = |p: char| w.strip_prefix(p).is_some_and(|r| !r.is_empty() && r.bytes().all(|x| x.is_ascii_digit()));
                assert!(OK.contains(&w.as_str()) || num('v') || num('t') || w == fname, "unexpected word {w:?} in\n{body}");
            } else {
                assert!(" \n(),;:=+-*/{}>".contains(c), "unexpected char {c:?} in\n{body}");
                i += 1;
            }
        }
    }

    #[test]
    fn hostile_inputs() {
        let deep = format!("{}x{}", "(".repeat(300), ")".repeat(300));
        let hostile: Vec<String> = vec![
            "x;discard".into(), "}fn evil(){".into(), "x\"//".into(), "f(x)}{".into(),
            "x+1e999".into(), "1e999".into(), "-1e999*x".into(), "ünï+x".into(), "x😀".into(),
            "evil(x)".into(), "x /* y */".into(), "sin(x);fn z(){}".into(), "x\n\0y".into(),
            "{x}".into(), "x@y".into(), "1e39*x".into(), "1e-60*x".into(), deep,
            "discard".into(), "x=y;".into(), "return x".into(), "x_evil + y".into(),
        ];
        let mut emitted = 0;
        for h in &hostile {
            let Ok(e) = parse(h) else { continue };
            for angle in [Angle::Rad, Angle::Deg] {
                let Ok(p) = compile(&e, &["x", "y", "z"], angle) else { continue };
                let Ok(m) = emit_module(&[("f", &p)]) else { continue };
                emitted += 1;
                check_naga(&m);
                body_ok(&m[PRELUDE.len()..], "f");
            }
        }
        assert!(emitted > 0);
        // Variable names that look like injection never appear: bind them as inputs.
        let p = compile(&parse("a+b").unwrap(), &["a", "b"], Angle::Rad).unwrap();
        let m = emit_module(&[("f", &p)]).unwrap();
        body_ok(&m[PRELUDE.len()..], "f");
        // Non-finite constants rejected.
        for c in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
            let p = Program { ops: vec![Op::Const(c)], vars: vec![], angle: Angle::Rad, subs: vec![] };
            assert_eq!(emit_function("f", &p), Err(WgslError::NonFiniteConstant));
        }
    }

    #[test]
    fn malformed_programs_rejected() {
        let mk = |ops| Program { ops, vars: vec!["x".into()], angle: Angle::Rad, subs: vec![] };
        assert!(emit_function("f", &mk(vec![])).is_err());
        assert!(emit_function("f", &mk(vec![Op::Add])).is_err());
        assert!(emit_function("f", &mk(vec![Op::Load(1)])).is_err());
        assert!(emit_function("f", &mk(vec![Op::Load(0), Op::Load(0)])).is_err());
    }

    #[test]
    fn invalid_names_rejected() {
        let p = prog("x", &["x"], Angle::Rad);
        for bad in [
            "", "1a", "a-b", "a b", "a;b", "fn", "let", "discard", "main", "sin", "v0", "t12",
            "am_pow", "__x", "_", "a\"", "é", "f()", "a{", &"a".repeat(42),
        ] {
            assert_eq!(emit_function(bad, &p), Err(WgslError::InvalidName), "{bad:?}");
        }
        for good in ["f", "_f", "height_1", &"a".repeat(41), "v", "t"] {
            assert!(emit_function(good, &p).is_ok(), "{good:?}");
        }
    }

    #[test]
    fn literal_formatting() {
        assert_eq!(format_f32(3.0).unwrap(), "3.0f");
        assert_eq!(format_f32(0.0).unwrap(), "0.0f");
        assert_eq!(format_f32(-2.5).unwrap(), "(-2.5f)");
        assert_eq!(format_f32(1e30).unwrap(), "1e30f");
        assert_eq!(format_f32(2e-8).unwrap(), "2e-8f");
        assert_eq!(format_f32(1e-60).unwrap(), "0.0f");
        assert!(format_f32(1e39).is_err());
        for c in [0.1, 1.0 / 3.0, 123456789.0, 1e16, 1e-7, f32::MAX as f64, 1.17549435e-38, -0.0, 7.0, 100.0] {
            let s = format_f32(c).unwrap();
            let t = s.trim_matches(|ch| ch == '(' || ch == ')').trim_end_matches('f');
            assert!(t.contains(['.', 'e']), "{s}");
            assert_eq!(t.parse::<f32>().unwrap(), c as f32, "{s}");
        }
    }

    #[test]
    fn operand_order() {
        let p = prog("x-y", &["x", "y"], Angle::Rad);
        let s = emit_function("f", &p).unwrap();
        assert!(s.contains("let t0 = v0;") && s.contains("let t1 = v1;"), "{s}");
        assert!(s.contains("let t2 = t0 - t1;") && s.contains("return t2;"), "{s}");
        let p = prog("x/y", &["x", "y"], Angle::Rad);
        assert!(emit_function("f", &p).unwrap().contains("let t2 = t0 / t1;"));
        let p = prog("x^y", &["x", "y"], Angle::Rad);
        assert!(emit_function("f", &p).unwrap().contains("am_pow(t0, t1)"));
        let p = prog("sin(x)", &["x"], Angle::Deg);
        assert!(emit_function("f", &p).unwrap().contains("sin(t0 * 0.017453292f)"));
        let p = prog("asin(x)", &["x"], Angle::Deg);
        assert!(emit_function("f", &p).unwrap().contains("asin(t0) / 0.017453292f"));
    }

    /// Interprets the emitted `let tK = ...;` lines for the arithmetic subset in f32.
    fn run_emitted(text: &str, vars: &[f32]) -> f32 {
        let mut t: Vec<f32> = Vec::new();
        let val = |tok: &str, t: &[f32]| -> f32 {
            let tok = tok.trim_matches(|c| c == '(' || c == ')');
            if let Some(i) = tok.strip_prefix('t') {
                t[i.parse::<usize>().unwrap()]
            } else if let Some(i) = tok.strip_prefix('v') {
                vars[i.parse::<usize>().unwrap()]
            } else {
                tok.trim_end_matches('f').parse().unwrap()
            }
        };
        for line in text.lines().map(str::trim).filter(|l| l.starts_with("let ")) {
            let rhs = line.split_once(" = ").unwrap().1.trim_end_matches(';');
            let parts: Vec<&str> = rhs.split(' ').collect();
            let v = match parts.as_slice() {
                [a] if a.starts_with('-') && a.as_bytes().get(1) == Some(&b't') => -val(&a[1..], &t),
                [a] => val(a, &t),
                [a, op, b] => {
                    let (a, b) = (val(a, &t), val(b, &t));
                    match *op {
                        "+" => a + b,
                        "-" => a - b,
                        "*" => a * b,
                        "/" => a / b,
                        o => panic!("op {o}"),
                    }
                }
                _ => panic!("unhandled {rhs}"),
            };
            t.push(v);
        }
        *t.last().unwrap()
    }

    #[test]
    fn differential_arithmetic() {
        let srcs = ["x-y", "x/(y+3)", "(x+1)(x-1)", "x*y/(x*x+y*y+1)", "-x - -2*y + 0.25", "1/(x-y*2)-x/3"];
        for src in srcs {
            let p = prog(src, &["x", "y"], Angle::Rad);
            let text = emit_function("f", &p).unwrap();
            for i in -8..=8 {
                for j in -8..=8 {
                    let (x, y) = (i as f64 * 0.7 + 0.13, j as f64 * 0.45 - 0.2);
                    let want = p.eval(&[x, y]);
                    let got = run_emitted(&text, &[x as f32, y as f32]) as f64;
                    if want.is_finite() && want.abs() < 1e4 {
                        assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "{src} ({x},{y}): {got} vs {want}");
                    }
                }
            }
        }
    }
}
