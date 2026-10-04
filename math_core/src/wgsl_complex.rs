//! Safe WGSL emission from a compiled [`CProgram`], plus the domain-colouring function.
//!
//! Same security model as [`crate::wgsl`]: the emitter consumes only a `CProgram` (closed opcode
//! set plus finite constants); every emitted identifier is generated (`v0..`, `t0..`, fixed
//! helper names). The only caller-supplied string is the function's own name, validated by
//! [`crate::wgsl::validate_name`]. Helper names all start with `am_`, a prefix `validate_name`
//! reserves, so a user function can never shadow them. The prelude is standalone (it defines its
//! own `am_cnan` / `am_cinf`), so it can be used with or without `wgsl::PRELUDE`.
//!
//! A complex number is a `vec2<f32>` (x = real, y = imaginary). Results match
//! [`CProgram::eval`] to f32 precision only. Division by zero yields (inf, inf) (a pole reads
//! as bright in the colouring) or NaN for 0/0. Argument/`ln` on the negative real axis rely on
//! the GPU's `atan2` honouring the sign of a zero imaginary part; exactly on the branch cut the
//! sheet may differ from the CPU. NaN/inf are detected by bit pattern, so fast-math cannot
//! remove the checks.
//!
//! ## Domain colouring (`am_domain_color` / [`domain_color_cpu`])
//! With `m = |w|`, `l = log2 m`, `s = fract(l)` (a sawtooth in log-modulus, so contours at
//! `|w| = 2^k`), `g = m / (1 + m)` and `u = m / (m + 300)`:
//! * hue = `fract(arg(w) / 2pi)` (red on the positive real axis, counter-clockwise),
//! * saturation = `0.9 - 0.85 u^2` (high, fading to white only for very large `|w|`, i.e. poles),
//! * value = `(0.05 + 0.95 g^0.4) * (0.78 + 0.22 s)` (zeros dark, large `|w|` bright),
//! * NaN -> neutral grey `(0.5, 0.5, 0.5)`; infinite modulus -> white.

use crate::complex::{CProgram, C64, CF1, COp};
use crate::wgsl::{format_f32, validate_name, WgslError};
use std::fmt::Write;

/// Helper functions required by emitted complex code. No comments, no user data.
pub const COMPLEX_PRELUDE: &str = "\
fn am_cnan() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn am_cinf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn am_c_isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn am_c_isinf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

fn am_c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn am_c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (abs(b.x) >= abs(b.y)) {
        if (b.x == 0.0) {
            if (a.x == 0.0 && a.y == 0.0) {
                return vec2<f32>(am_cnan(), am_cnan());
            }
            return vec2<f32>(am_cinf(), am_cinf());
        }
        let r = b.y / b.x;
        let d = b.x + b.y * r;
        return vec2<f32>((a.x + a.y * r) / d, (a.y - a.x * r) / d);
    }
    let r = b.x / b.y;
    let d = b.x * r + b.y;
    return vec2<f32>((a.x * r + a.y) / d, (a.y * r - a.x) / d);
}

fn am_c_abs(z: vec2<f32>) -> f32 {
    let ax = abs(z.x);
    let ay = abs(z.y);
    if (am_c_isinf(ax) || am_c_isinf(ay)) {
        return am_cinf();
    }
    let m = max(ax, ay);
    if (m == 0.0) {
        return 0.0;
    }
    let a = ax / m;
    let b = ay / m;
    return m * sqrt(a * a + b * b);
}

fn am_c_arg(z: vec2<f32>) -> f32 {
    if (z.x == 0.0 && z.y == 0.0) {
        return 0.0;
    }
    return atan2(z.y, z.x);
}

fn am_c_conj(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x, -z.y);
}

fn am_c_exp(z: vec2<f32>) -> vec2<f32> {
    let e = exp(z.x);
    return vec2<f32>(e * cos(z.y), e * sin(z.y));
}

fn am_c_ln(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(log(am_c_abs(z)), am_c_arg(z));
}

fn am_c_sqrt(z: vec2<f32>) -> vec2<f32> {
    if (z.x == 0.0 && z.y == 0.0) {
        return vec2<f32>(0.0, z.y);
    }
    let h = am_c_abs(z);
    let t = sqrt((abs(z.x) + h) * 0.5);
    if (z.x >= 0.0) {
        return vec2<f32>(t, z.y / (2.0 * t));
    }
    var sy = t;
    if (z.y < 0.0) {
        sy = -t;
    }
    return vec2<f32>(abs(z.y) / (2.0 * t), sy);
}

fn am_c_powi(z: vec2<f32>, n: i32) -> vec2<f32> {
    var e = u32(abs(n));
    var base = z;
    var acc = vec2<f32>(1.0, 0.0);
    for (var k = 0; k < 7; k = k + 1) {
        if (e == 0u) {
            break;
        }
        if ((e & 1u) == 1u) {
            acc = am_c_mul(acc, base);
        }
        base = am_c_mul(base, base);
        e = e >> 1u;
    }
    if (n < 0) {
        return am_c_div(vec2<f32>(1.0, 0.0), acc);
    }
    return acc;
}

fn am_c_pow(z: vec2<f32>, w: vec2<f32>) -> vec2<f32> {
    if (w.y == 0.0 && w.x == floor(w.x) && abs(w.x) <= 64.0) {
        return am_c_powi(z, i32(w.x));
    }
    if (z.x == 0.0 && z.y == 0.0) {
        if (w.x > 0.0) {
            return vec2<f32>(0.0, 0.0);
        }
        return vec2<f32>(am_cnan(), am_cnan());
    }
    return am_c_exp(am_c_mul(w, am_c_ln(z)));
}

fn am_c_sin(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

fn am_c_cos(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(z.x) * cosh(z.y), -(sin(z.x) * sinh(z.y)));
}

fn am_c_sinh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}

fn am_c_cosh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cosh(z.x) * cos(z.y), sinh(z.x) * sin(z.y));
}

fn am_c_tanh(z: vec2<f32>) -> vec2<f32> {
    if (abs(z.x) > 12.0) {
        var s = 1.0;
        if (z.x < 0.0) {
            s = -1.0;
        }
        return vec2<f32>(s, 0.0);
    }
    let d = cosh(2.0 * z.x) + cos(2.0 * z.y);
    return vec2<f32>(sinh(2.0 * z.x) / d, sin(2.0 * z.y) / d);
}

fn am_c_tan(z: vec2<f32>) -> vec2<f32> {
    let t = am_c_tanh(vec2<f32>(-z.y, z.x));
    return vec2<f32>(t.y, -t.x);
}

fn am_hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return v * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

fn am_domain_color(w: vec2<f32>) -> vec3<f32> {
    if (am_c_isnan(w.x) || am_c_isnan(w.y)) {
        return vec3<f32>(0.5, 0.5, 0.5);
    }
    let m = am_c_abs(w);
    if (am_c_isinf(m)) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    let hue = fract(am_c_arg(w) / 6.28318531);
    let g = m / (1.0 + m);
    var s = 0.0;
    if (m > 0.0) {
        s = fract(log2(m));
    }
    let u = m / (m + 300.0);
    let sat = 0.9 - 0.85 * u * u;
    var gp = 0.0;
    if (g > 0.0) {
        gp = pow(g, 0.4);
    }
    let val = (0.05 + 0.95 * gp) * (0.78 + 0.22 * s);
    return am_hsv2rgb(hue, sat, val);
}

";

fn hsv2rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let fract = |x: f32| x - x.floor();
    let k = [1.0f32, 2.0 / 3.0, 1.0 / 3.0];
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let p = (fract(h + k[i]) * 6.0 - 3.0).abs();
        let c = (p - 1.0).clamp(0.0, 1.0);
        out[i] = v * (1.0 + (c - 1.0) * s);
    }
    out
}

/// CPU twin of `am_domain_color` (same formulas, f32 arithmetic); see the module docs.
pub fn domain_color_cpu(w: C64) -> [f32; 3] {
    if w.is_nan() {
        return [0.5, 0.5, 0.5];
    }
    let (re, im) = (w.re as f32, w.im as f32);
    let m = w.abs() as f32;
    if m.is_infinite() || re.is_infinite() || im.is_infinite() {
        return [1.0, 1.0, 1.0];
    }
    let arg = if re == 0.0 && im == 0.0 { 0.0 } else { im.atan2(re) };
    let t = arg / 6.283_185_5;
    let hue = t - t.floor();
    let g = m / (1.0 + m);
    let s = if m > 0.0 {
        let l = m.log2();
        l - l.floor()
    } else {
        0.0
    };
    let u = m / (m + 300.0);
    let sat = 0.9 - 0.85 * u * u;
    let gp = if g > 0.0 { g.powf(0.4) } else { 0.0 };
    let val = (0.05 + 0.95 * gp) * (0.78 + 0.22 * s);
    hsv2rgb(hue, sat, val)
}

fn cvec(c: C64) -> Result<String, WgslError> {
    Ok(format!("vec2<f32>({}, {})", format_f32(c.re)?, format_f32(c.im)?))
}

fn f1_expr(f: CF1, a: &str) -> String {
    match f {
        CF1::Sin => format!("am_c_sin({a})"),
        CF1::Cos => format!("am_c_cos({a})"),
        CF1::Tan => format!("am_c_tan({a})"),
        CF1::Sinh => format!("am_c_sinh({a})"),
        CF1::Cosh => format!("am_c_cosh({a})"),
        CF1::Tanh => format!("am_c_tanh({a})"),
        CF1::Exp => format!("am_c_exp({a})"),
        CF1::Ln => format!("am_c_ln({a})"),
        CF1::Sqrt => format!("am_c_sqrt({a})"),
        CF1::Abs => format!("vec2<f32>(am_c_abs({a}), 0.0)"),
        CF1::Arg => format!("vec2<f32>(am_c_arg({a}), 0.0)"),
        CF1::Re => format!("vec2<f32>({a}.x, 0.0)"),
        CF1::Im => format!("vec2<f32>({a}.y, 0.0)"),
        CF1::Conj => format!("am_c_conj({a})"),
    }
}

/// Emits `fn <name>(v0: vec2<f32>, ...) -> vec2<f32> { ... }` by simulating the postfix stack.
/// Requires [`COMPLEX_PRELUDE`] in the same module.
pub fn emit_complex_function(name: &str, p: &CProgram) -> Result<String, WgslError> {
    validate_name(name)?;
    let nvars = p.vars.len();
    let mut out = String::new();
    let params: Vec<String> = (0..nvars).map(|i| format!("v{i}: vec2<f32>")).collect();
    let _ = writeln!(out, "fn {name}({}) -> vec2<f32> {{", params.join(", "));
    let mut stack: Vec<usize> = Vec::new();
    let underflow = WgslError::BadProgram("stack underflow");
    for (n, op) in p.ops.iter().enumerate() {
        let rhs = match *op {
            COp::Const(c) => cvec(c)?,
            COp::Load(i) => {
                if i >= nvars {
                    return Err(WgslError::BadProgram("load index out of range"));
                }
                format!("v{i}")
            }
            COp::Neg => {
                let a = stack.pop().ok_or(underflow.clone())?;
                format!("-t{a}")
            }
            COp::Add | COp::Sub | COp::Mul | COp::Div | COp::Pow => {
                let b = stack.pop().ok_or(underflow.clone())?;
                let a = stack.pop().ok_or(underflow.clone())?;
                match op {
                    COp::Add => format!("t{a} + t{b}"),
                    COp::Sub => format!("t{a} - t{b}"),
                    COp::Mul => format!("am_c_mul(t{a}, t{b})"),
                    COp::Div => format!("am_c_div(t{a}, t{b})"),
                    _ => format!("am_c_pow(t{a}, t{b})"),
                }
            }
            COp::F1(f) => {
                let a = stack.pop().ok_or(underflow.clone())?;
                f1_expr(f, &format!("t{a}"))
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

/// A complete field module for the renderer: [`COMPLEX_PRELUDE`], the compiled function (named `cplx`), and `fn field_color(x: f32, y: f32) -> vec4<f32>` which evaluates it
/// at `z = x + iy` and returns the opaque domain colour. `p` must take exactly one variable.
pub fn emit_domain_module(p: &CProgram) -> Result<String, WgslError> {
    if p.vars.len() != 1 {
        return Err(WgslError::BadProgram("domain function needs exactly one variable"));
    }
    let mut out = String::from(COMPLEX_PRELUDE);
    out.push_str(&emit_complex_function("cplx", p)?);
    out.push_str(
        "fn field_color(x: f32, y: f32) -> vec4<f32> {\n    \
         return vec4<f32>(am_domain_color(cplx(vec2<f32>(x, y))), 1.0);\n}\n",
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::{compile_complex, parse_complex};
    use crate::parse::ParseCtx;
    use std::f64::consts::PI;

    const CORPUS: &[&str] = &[
        "z", "z^2-1", "1/z", "sin(z)", "cos(z)", "tan(z)", "e^z", "(z-1)/(z^2+z+1)", "sqrt(z)",
        "ln(z)", "z^z", "z^2.5", "z^-3", "sinh(z)+cosh(z)", "tanh(z)", "abs(z)", "arg(z)",
        "conj(z)*z", "re(z)+i*im(z)", "e^(i*z)", "2i", "pi*z", "log(z)", "sec(z)+csc(z)+cot(z)",
        "z^100", "z^(1+i)", "(z^2+1)/(z^2-1)", "-z", "5", "i",
    ];

    fn prog(src: &str) -> CProgram {
        let e = parse_complex(src, &ParseCtx::new()).unwrap_or_else(|e| panic!("{src}: {e}"));
        compile_complex(&e, &["z"]).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    fn check_naga(src: &str) {
        let m = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }

    #[test]
    fn prelude_valid_alone_and_with_real_prelude() {
        check_naga(COMPLEX_PRELUDE);
        check_naga(&format!("{}{}", crate::wgsl::PRELUDE, COMPLEX_PRELUDE));
    }

    #[test]
    fn corpus_validates() {
        for src in CORPUS {
            let p = prog(src);
            let f = emit_complex_function("cf", &p).unwrap();
            check_naga(&format!("{COMPLEX_PRELUDE}{f}"));
            let m = emit_domain_module(&p).unwrap();
            check_naga(&m);
            // The module the renderer builds alongside a real function must also be valid.
            let real = crate::compile::compile(
                &crate::parse::parse("x^2+y").unwrap(),
                &["x", "y"],
                crate::compile::Angle::Rad,
            )
            .unwrap();
            let both = format!("{}{}", crate::wgsl::emit_module(&[("field_fn", &real)]).unwrap(), m);
            check_naga(&both);
        }
        // Two variables work too.
        let e = parse_complex("z*w+1", &ParseCtx::new()).unwrap();
        let p = compile_complex(&e, &["z", "w"]).unwrap();
        check_naga(&format!("{COMPLEX_PRELUDE}{}", emit_complex_function("cf2", &p).unwrap()));
    }

    #[test]
    fn hostile_strings_cannot_reach_output() {
        let p = prog("z^2+1");
        for bad in ["", "1a", "fn", "a b", "a;b", "x}\nfn evil", "am_c_mul", "v0", "t1", "main", "a\u{0}b", "é", &"a".repeat(80)] {
            assert_eq!(emit_complex_function(bad, &p), Err(WgslError::InvalidName), "{bad:?}");
        }
        // Variable names from the source are never copied into the output.
        let e = parse_complex("q_evil+1", &ParseCtx::new()).unwrap();
        let p = compile_complex(&e, &["q_evil"]).unwrap();
        let out = emit_complex_function("cf", &p).unwrap();
        assert!(!out.contains("q_evil") && !out.contains("evil"), "{out}");
        // Hostile source text parses to something whose output only has generated identifiers.
        for s in ["}; evil(); {", "z; discard; z", "z//", "fn x() {}"] {
            if let Ok(e) = parse_complex(s, &ParseCtx::new()) {
                if let Ok(p) = compile_complex(&e, &["z"]) {
                    let m = emit_domain_module(&p).unwrap();
                    assert!(!m.contains("evil") && !m.contains("discard"));
                    check_naga(&m);
                }
            }
        }
    }

    #[test]
    fn non_finite_constants_rejected() {
        for c in [f64::NAN, f64::INFINITY, 1e300, -1e300] {
            let p = CProgram { ops: vec![COp::Const(C64::new(c, 0.0))], vars: vec![] };
            assert_eq!(emit_complex_function("cf", &p), Err(WgslError::NonFiniteConstant));
            let p = CProgram { ops: vec![COp::Const(C64::new(0.0, c))], vars: vec![] };
            assert_eq!(emit_complex_function("cf", &p), Err(WgslError::NonFiniteConstant));
        }
        let e = parse_complex("1e300*z", &ParseCtx::new()).unwrap();
        let p = compile_complex(&e, &["z"]).unwrap();
        assert!(emit_complex_function("cf", &p).is_err());
    }

    #[test]
    fn malformed_programs_rejected() {
        let mk = |ops: Vec<COp>, nv: usize| CProgram { ops, vars: (0..nv).map(|i| format!("v{i}")).collect() };
        assert!(emit_complex_function("cf", &mk(vec![], 1)).is_err());
        assert!(emit_complex_function("cf", &mk(vec![COp::Add], 1)).is_err());
        assert!(emit_complex_function("cf", &mk(vec![COp::Load(3)], 1)).is_err());
        assert!(emit_complex_function("cf", &mk(vec![COp::Load(0), COp::Load(0)], 1)).is_err());
        assert!(emit_domain_module(&mk(vec![COp::Load(0)], 2)).is_err());
    }

    // ---- CPU emulation of the emitted op order, in f32 -----------------------------------

    type V = [f32; 2];

    fn mul(a: V, b: V) -> V {
        [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
    }
    fn div(a: V, b: V) -> V {
        if b[0].abs() >= b[1].abs() {
            if b[0] == 0.0 {
                return if a == [0.0, 0.0] { [f32::NAN; 2] } else { [f32::INFINITY; 2] };
            }
            let r = b[1] / b[0];
            let d = b[0] + b[1] * r;
            [(a[0] + a[1] * r) / d, (a[1] - a[0] * r) / d]
        } else {
            let r = b[0] / b[1];
            let d = b[0] * r + b[1];
            [(a[0] * r + a[1]) / d, (a[1] * r - a[0]) / d]
        }
    }
    fn cabs(z: V) -> f32 {
        let (ax, ay) = (z[0].abs(), z[1].abs());
        let m = ax.max(ay);
        if m == 0.0 {
            return 0.0;
        }
        let (a, b) = (ax / m, ay / m);
        m * (a * a + b * b).sqrt()
    }
    fn carg(z: V) -> f32 {
        if z == [0.0, 0.0] {
            0.0
        } else {
            z[1].atan2(z[0])
        }
    }
    fn cexp(z: V) -> V {
        let e = z[0].exp();
        [e * z[1].cos(), e * z[1].sin()]
    }
    fn cln(z: V) -> V {
        [cabs(z).ln(), carg(z)]
    }
    fn csqrt(z: V) -> V {
        if z == [0.0, 0.0] {
            return [0.0, z[1]];
        }
        let t = ((z[0].abs() + cabs(z)) * 0.5).sqrt();
        if z[0] >= 0.0 {
            [t, z[1] / (2.0 * t)]
        } else {
            [z[1].abs() / (2.0 * t), if z[1] < 0.0 { -t } else { t }]
        }
    }
    fn powi(z: V, n: i32) -> V {
        let mut e = n.unsigned_abs();
        let (mut base, mut acc) = (z, [1.0f32, 0.0]);
        for _ in 0..7 {
            if e == 0 {
                break;
            }
            if e & 1 == 1 {
                acc = mul(acc, base);
            }
            base = mul(base, base);
            e >>= 1;
        }
        if n < 0 {
            div([1.0, 0.0], acc)
        } else {
            acc
        }
    }
    fn cpow(z: V, w: V) -> V {
        if w[1] == 0.0 && w[0] == w[0].floor() && w[0].abs() <= 64.0 {
            return powi(z, w[0] as i32);
        }
        if z == [0.0, 0.0] {
            return if w[0] > 0.0 { [0.0, 0.0] } else { [f32::NAN; 2] };
        }
        cexp(mul(w, cln(z)))
    }
    fn ctanh(z: V) -> V {
        if z[0].abs() > 12.0 {
            return [z[0].signum(), 0.0];
        }
        let d = (2.0 * z[0]).cosh() + (2.0 * z[1]).cos();
        [(2.0 * z[0]).sinh() / d, (2.0 * z[1]).sin() / d]
    }

    fn emulate(p: &CProgram, z: V) -> V {
        let mut st: Vec<V> = Vec::new();
        for op in &p.ops {
            match *op {
                COp::Const(c) => st.push([c.re as f32, c.im as f32]),
                COp::Load(_) => st.push(z),
                COp::Neg => {
                    let a = st.pop().unwrap();
                    st.push([-a[0], -a[1]]);
                }
                COp::Add | COp::Sub | COp::Mul | COp::Div | COp::Pow => {
                    let b = st.pop().unwrap();
                    let a = st.pop().unwrap();
                    st.push(match op {
                        COp::Add => [a[0] + b[0], a[1] + b[1]],
                        COp::Sub => [a[0] - b[0], a[1] - b[1]],
                        COp::Mul => mul(a, b),
                        COp::Div => div(a, b),
                        _ => cpow(a, b),
                    });
                }
                COp::F1(f) => {
                    let a = st.pop().unwrap();
                    st.push(match f {
                        CF1::Sin => [a[0].sin() * a[1].cosh(), a[0].cos() * a[1].sinh()],
                        CF1::Cos => [a[0].cos() * a[1].cosh(), -(a[0].sin() * a[1].sinh())],
                        CF1::Sinh => [a[0].sinh() * a[1].cos(), a[0].cosh() * a[1].sin()],
                        CF1::Cosh => [a[0].cosh() * a[1].cos(), a[0].sinh() * a[1].sin()],
                        CF1::Tanh => ctanh(a),
                        CF1::Tan => {
                            let t = ctanh([-a[1], a[0]]);
                            [t[1], -t[0]]
                        }
                        CF1::Exp => cexp(a),
                        CF1::Ln => cln(a),
                        CF1::Sqrt => csqrt(a),
                        CF1::Abs => [cabs(a), 0.0],
                        CF1::Arg => [carg(a), 0.0],
                        CF1::Re => [a[0], 0.0],
                        CF1::Im => [a[1], 0.0],
                        CF1::Conj => [a[0], -a[1]],
                    });
                }
            }
        }
        st[0]
    }

    #[test]
    fn emulated_op_order_matches_cpu_eval() {
        // The emulation mirrors the exact op sequence emit_complex_function walks.
        let mut checked = 0;
        for src in CORPUS {
            let p = prog(src);
            // Emission succeeds for the same program, so the emitted op order is the one emulated.
            emit_complex_function("cf", &p).unwrap();
            for ix in -12..=12 {
                for iy in -12..=12 {
                    let z = C64::new(ix as f64 * 0.23 + 0.011, iy as f64 * 0.19 - 0.007);
                    let want = p.eval(&[z]);
                    let got = emulate(&p, [z.re as f32, z.im as f32]);
                    let (gr, gi) = (got[0] as f64, got[1] as f64);
                    if !want.re.is_finite() || !want.im.is_finite() || want.abs() > 1e6 || want.abs() < 1e-4 {
                        continue;
                    }
                    // Near-pole values amplify f32 input rounding; use a tolerance scaled by
                    // magnitude and a generous relative term for cancellation-heavy functions.
                    let tol = 2e-3 * (1.0 + want.abs());
                    assert!(
                        (gr - want.re).abs() <= tol && (gi - want.im).abs() <= tol,
                        "{src} at {z:?}: emulated ({gr}, {gi}) vs {want:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 5000, "only {checked} points compared");
    }

    #[test]
    fn domain_color_hue_by_quadrant() {
        // Same modulus, so only the hue differs: hue = arg / 2pi.
        let rgb = |re: f64, im: f64| domain_color_cpu(C64::new(re, im));
        let hue_of = |c: [f32; 3]| {
            let (r, g, b) = (c[0], c[1], c[2]);
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            let d = mx - mn;
            let h = if mx == r {
                ((g - b) / d).rem_euclid(6.0)
            } else if mx == g {
                (b - r) / d + 2.0
            } else {
                (r - g) / d + 4.0
            };
            h / 6.0
        };
        let q1 = hue_of(rgb(1.5, 1.5)); // arg pi/4 -> 0.125
        let q2 = hue_of(rgb(-1.5, 1.5)); // 3pi/4 -> 0.375
        let q3 = hue_of(rgb(-1.5, -1.5)); // -3pi/4 -> 0.625
        let q4 = hue_of(rgb(1.5, -1.5)); // -pi/4 -> 0.875
        for (got, want) in [(q1, 0.125), (q2, 0.375), (q3, 0.625), (q4, 0.875)] {
            assert!((got - want).abs() < 0.01, "hue {got} vs {want}");
        }
        // Positive real axis is red, positive imaginary axis is yellow-green (hue 0.25).
        let red = rgb(2.0, 0.0);
        assert!(red[0] > red[1] && red[0] > red[2]);
        assert!((hue_of(rgb(0.0, 2.0)) - 0.25).abs() < 0.01);
    }

    #[test]
    fn domain_color_zero_dark_infinity_bright_nan_grey() {
        let lum = |c: [f32; 3]| c[0].max(c[1]).max(c[2]);
        assert!(lum(domain_color_cpu(C64::ZERO)) < 0.1);
        assert!(lum(domain_color_cpu(C64::new(1e-6, 0.0))) < 0.2);
        let big = domain_color_cpu(C64::new(1e6, 1e6));
        assert!(lum(big) > 0.85, "{big:?}");
        assert_eq!(domain_color_cpu(C64::new(f64::INFINITY, 0.0)), [1.0, 1.0, 1.0]);
        assert_eq!(domain_color_cpu(C64::new(f64::NAN, 0.0)), [0.5, 0.5, 0.5]);
        assert_eq!(domain_color_cpu(C64::new(1.0, f64::NAN)), [0.5, 0.5, 0.5]);
        // Magnitude contours: brightness jumps down across |w| = 2^k going outward.
        let below = lum(domain_color_cpu(C64::new(1.999, 0.0)));
        let above = lum(domain_color_cpu(C64::new(2.001, 0.0)));
        assert!(below > above, "{below} vs {above}");
        // All channels stay in [0, 1] over a sweep.
        for k in -30..30 {
            for a in 0..32 {
                let w = C64::from_polar(1.37f64.powi(k), a as f64 * 0.2 - 3.2);
                for c in domain_color_cpu(w) {
                    assert!((0.0..=1.0).contains(&c), "{w:?} -> {c}");
                }
            }
        }
        // Euler: arg is continuous across the negative real axis from either side in hue wrap.
        let up = domain_color_cpu(C64::new(-1.5, 1e-9));
        let dn = domain_color_cpu(C64::new(-1.5, -1e-9));
        assert!((up[0] - dn[0]).abs() < 0.2 && (up[2] - dn[2]).abs() < 0.2);
        let _ = PI;
    }
}
