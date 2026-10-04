//! Complex numbers (f64) and a compiler from a scalar [`Expr`] to complex postfix bytecode.
//!
//! Conventions:
//! * Complex trig/hyperbolic functions always take RADIANS (there is no degree mode).
//! * `ln` and `sqrt` are the principal branches; `arg` is in (-pi, pi]. The sign of a zero
//!   imaginary part is respected on the branch cut: `ln(-1 + 0i) = +i pi`, `ln(-1 - 0i) = -i pi`.
//! * `z^w` is `exp(w ln z)` (principal), except when `w` is a real integer with `|n| <= 64`,
//!   where exact square-and-multiply is used (so `z^2` is accurate and defined for `z = 0`).
//!   `0^w`: 1 for `w = 0`; 0 for `Re w > 0`; NaN otherwise (negative powers of 0 and non-real
//!   powers with `Re w <= 0` are undefined).
//! * Like `compile.rs`, the compiler emits only a closed opcode set and constants from the AST,
//!   so no user text reaches an evaluator or a generated shader.

use crate::ast::{constant_value, BinOp, Expr};
use crate::compile::CompileError;
use crate::parse::{parse_with, ParseCtx, ParseError};
use std::f64::consts::LN_10;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

/// Largest |n| for which an integer exponent uses repeated multiplication.
pub const MAX_INT_POW: f64 = 64.0;
/// |Re| beyond which `tanh` is `+-1` (avoids `cosh` overflow); shared with the WGSL prelude.
pub const TANH_SAT: f64 = 12.0;

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    pub const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    pub const fn real(re: f64) -> C64 {
        C64 { re, im: 0.0 }
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Principal argument in (-pi, pi] (uses the sign of a zero imaginary part).
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conj(self) -> C64 {
        C64::new(self.re, -self.im)
    }

    pub fn from_polar(r: f64, theta: f64) -> C64 {
        C64::new(r * theta.cos(), r * theta.sin())
    }

    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    pub fn exp(self) -> C64 {
        C64::from_polar(self.re.exp(), self.im)
    }

    pub fn ln(self) -> C64 {
        C64::new(self.abs().ln(), self.arg())
    }

    pub fn sqrt(self) -> C64 {
        let (a, b) = (self.re, self.im);
        if a == 0.0 && b == 0.0 {
            return C64::new(0.0, b);
        }
        if a.is_nan() || b.is_nan() {
            return C64::new(f64::NAN, f64::NAN);
        }
        if b.is_infinite() {
            return C64::new(f64::INFINITY, b);
        }
        let t = ((a.abs() + a.hypot(b)) * 0.5).sqrt();
        if a >= 0.0 {
            C64::new(t, b / (2.0 * t))
        } else {
            C64::new(b.abs() / (2.0 * t), t.copysign(b))
        }
    }

    /// Integer power by square-and-multiply (negative `n` divides 1 by the positive power).
    pub fn powi(self, n: i32) -> C64 {
        let mut e = n.unsigned_abs();
        let mut base = self;
        let mut acc = C64::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        if n < 0 {
            C64::ONE / acc
        } else {
            acc
        }
    }

    pub fn pow(self, w: C64) -> C64 {
        if w.im == 0.0 && w.re.fract() == 0.0 && w.re.abs() <= MAX_INT_POW {
            return self.powi(w.re as i32);
        }
        if self.re == 0.0 && self.im == 0.0 {
            return if w.re > 0.0 { C64::ZERO } else { C64::new(f64::NAN, f64::NAN) };
        }
        (w * self.ln()).exp()
    }

    pub fn sin(self) -> C64 {
        C64::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }

    pub fn cos(self) -> C64 {
        C64::new(self.re.cos() * self.im.cosh(), -(self.re.sin() * self.im.sinh()))
    }

    pub fn sinh(self) -> C64 {
        C64::new(self.re.sinh() * self.im.cos(), self.re.cosh() * self.im.sin())
    }

    pub fn cosh(self) -> C64 {
        C64::new(self.re.cosh() * self.im.cos(), self.re.sinh() * self.im.sin())
    }

    pub fn tanh(self) -> C64 {
        let (a, b) = (self.re, self.im);
        if a.abs() > TANH_SAT {
            return C64::new(a.signum(), 0.0_f64.copysign(b));
        }
        let d = (2.0 * a).cosh() + (2.0 * b).cos();
        C64::new((2.0 * a).sinh() / d, (2.0 * b).sin() / d)
    }

    /// `tan z = -i tanh(i z)`.
    pub fn tan(self) -> C64 {
        let t = C64::new(-self.im, self.re).tanh();
        C64::new(t.im, -t.re)
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        C64::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

/// Smith's algorithm: scales by the larger divisor component, so it neither overflows nor
/// underflows for large/small magnitudes. Division by zero gives inf/NaN like `f64`.
impl Div for C64 {
    type Output = C64;
    fn div(self, o: C64) -> C64 {
        if o.re.abs() >= o.im.abs() {
            if o.re == 0.0 {
                // o == 0 (|im| <= |re| = 0)
                return C64::new(self.re / 0.0, self.im / 0.0);
            }
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            C64::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            C64::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bytecode
// ---------------------------------------------------------------------------------------------

/// Unary complex functions (closed set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CF1 {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Sqrt,
    /// Modulus as a complex with zero imaginary part.
    Abs,
    /// Principal argument as a complex with zero imaginary part.
    Arg,
    Re,
    Im,
    Conj,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum COp {
    Const(C64),
    Load(usize),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    F1(CF1),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CProgram {
    pub ops: Vec<COp>,
    pub vars: Vec<String>,
}

fn f1_of(name: &str) -> Option<CF1> {
    Some(match name {
        "sin" => CF1::Sin,
        "cos" => CF1::Cos,
        "tan" => CF1::Tan,
        "sinh" => CF1::Sinh,
        "cosh" => CF1::Cosh,
        "tanh" => CF1::Tanh,
        "exp" => CF1::Exp,
        "ln" => CF1::Ln,
        "sqrt" => CF1::Sqrt,
        "abs" => CF1::Abs,
        "arg" => CF1::Arg,
        "re" => CF1::Re,
        "im" => CF1::Im,
        "conj" => CF1::Conj,
        _ => return None,
    })
}

/// The core lexer splits unknown words such as `conj` into single-letter variables, so the
/// complex-only function names are rewritten to these private one-letter function names before
/// parsing and mapped back to the real names in the resulting tree.
const ALIASES: &[(&str, char)] = &[("conj", 'Ω'), ("arg", 'Ξ'), ("re", 'Ψ'), ("im", 'Φ')];

fn rename_calls(e: Expr) -> Expr {
    let go = |v: Vec<Expr>| v.into_iter().map(rename_calls).collect::<Vec<_>>();
    match e {
        Expr::Num(_) | Expr::Var(_) => e,
        Expr::Neg(a) => Expr::Neg(Box::new(rename_calls(*a))),
        Expr::Bin(op, a, b) => Expr::Bin(op, Box::new(rename_calls(*a)), Box::new(rename_calls(*b))),
        Expr::Rel(r, a, b) => Expr::Rel(r, Box::new(rename_calls(*a)), Box::new(rename_calls(*b))),
        Expr::Tuple(v) => Expr::Tuple(go(v)),
        Expr::List(v) => Expr::List(go(v)),
        Expr::Call(n, args) => {
            let name = ALIASES
                .iter()
                .find(|(_, c)| n.chars().eq(std::iter::once(*c)))
                .map(|(real, _)| real.to_string())
                .unwrap_or(n);
            Expr::Call(name, go(args))
        }
    }
}

/// Parses `src` like [`parse_with`], additionally understanding the complex-only functions
/// `conj`, `arg`, `re`, `im` (a name followed by `(`, not part of a longer word).
pub fn parse_complex(src: &str, ctx: &ParseCtx) -> Result<Expr, ParseError> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        let prev_alnum = i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '\\');
        let hit = if prev_alnum {
            None
        } else {
            ALIASES.iter().find(|(name, _)| {
                let n = name.chars().count();
                chars.len() >= i + n
                    && chars[i..i + n].iter().copied().eq(name.chars())
                    && chars[i + n..].iter().find(|c| !c.is_whitespace()) == Some(&'(')
            })
        };
        match hit {
            Some((name, c)) => {
                out.push(*c);
                i += name.chars().count();
            }
            None => {
                out.push(chars[i]);
                i += 1;
            }
        }
    }
    let mut ctx = ctx.clone();
    for (_, c) in ALIASES {
        ctx.functions.insert(c.to_string());
    }
    parse_with(&out, &ctx).map(rename_calls)
}

const MAX_DEPTH: usize = 200;

fn arity(name: &str, expected: usize, args: &[Expr]) -> Result<(), CompileError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(CompileError::Arity { name: name.to_string(), expected, got: args.len() })
    }
}

fn emit(e: &Expr, vars: &[String], out: &mut Vec<COp>, depth: usize) -> Result<(), CompileError> {
    if depth > MAX_DEPTH {
        return Err(CompileError::NotScalar("an expression nested too deeply"));
    }
    let d = depth + 1;
    match e {
        Expr::Num(v) => out.push(COp::Const(C64::real(*v))),
        Expr::Var(n) => {
            if let Some(i) = vars.iter().position(|v| v == n) {
                out.push(COp::Load(i));
            } else if n == "i" {
                out.push(COp::Const(C64::I));
            } else if let Some(c) = constant_value(n) {
                out.push(COp::Const(C64::real(c)));
            } else {
                return Err(CompileError::UnboundVariable(n.clone()));
            }
        }
        Expr::Neg(a) => {
            emit(a, vars, out, d)?;
            out.push(COp::Neg);
        }
        Expr::Bin(op, a, b) => {
            emit(a, vars, out, d)?;
            emit(b, vars, out, d)?;
            out.push(match op {
                BinOp::Add => COp::Add,
                BinOp::Sub => COp::Sub,
                BinOp::Mul => COp::Mul,
                BinOp::Div => COp::Div,
                BinOp::Pow => COp::Pow,
            });
        }
        Expr::Call(name, args) => {
            let n = name.as_str();
            if let Some(f) = f1_of(n) {
                arity(n, 1, args)?;
                emit(&args[0], vars, out, d)?;
                out.push(COp::F1(f));
            } else {
                match n {
                    "sec" | "csc" => {
                        arity(n, 1, args)?;
                        out.push(COp::Const(C64::ONE));
                        emit(&args[0], vars, out, d)?;
                        out.push(COp::F1(if n == "sec" { CF1::Cos } else { CF1::Sin }));
                        out.push(COp::Div);
                    }
                    "cot" => {
                        arity(n, 1, args)?;
                        emit(&args[0], vars, out, d)?;
                        out.push(COp::F1(CF1::Cos));
                        emit(&args[0], vars, out, d)?;
                        out.push(COp::F1(CF1::Sin));
                        out.push(COp::Div);
                    }
                    "log" => {
                        arity(n, 1, args)?;
                        emit(&args[0], vars, out, d)?;
                        out.push(COp::F1(CF1::Ln));
                        out.push(COp::Const(C64::real(LN_10)));
                        out.push(COp::Div);
                    }
                    _ => return Err(CompileError::UnknownFunction(name.clone())),
                }
            }
        }
        Expr::Tuple(_) => return Err(CompileError::NotScalar("a point")),
        Expr::List(_) => return Err(CompileError::NotScalar("a list")),
        Expr::Rel(..) => return Err(CompileError::NotScalar("a relation")),
    }
    Ok(())
}

/// Compiles `expr` with `vars` as input slots. The name `i` is the imaginary unit unless it is
/// itself one of `vars`.
pub fn compile_complex(expr: &Expr, vars: &[&str]) -> Result<CProgram, CompileError> {
    let vars: Vec<String> = vars.iter().map(|s| s.to_string()).collect();
    let mut ops = Vec::new();
    emit(expr, &vars, &mut ops, 0)?;
    Ok(CProgram { ops, vars })
}

pub fn apply1(f: CF1, z: C64) -> C64 {
    match f {
        CF1::Sin => z.sin(),
        CF1::Cos => z.cos(),
        CF1::Tan => z.tan(),
        CF1::Sinh => z.sinh(),
        CF1::Cosh => z.cosh(),
        CF1::Tanh => z.tanh(),
        CF1::Exp => z.exp(),
        CF1::Ln => z.ln(),
        CF1::Sqrt => z.sqrt(),
        CF1::Abs => C64::real(z.abs()),
        CF1::Arg => C64::real(z.arg()),
        CF1::Re => C64::real(z.re),
        CF1::Im => C64::real(z.im),
        CF1::Conj => z.conj(),
    }
}

impl CProgram {
    /// Evaluates with `inputs` bound to `vars` in order. Missing inputs read as 0; a
    /// malformed program (impossible from `compile_complex`) yields NaN instead of panicking.
    pub fn eval(&self, inputs: &[C64]) -> C64 {
        let nan = C64::new(f64::NAN, f64::NAN);
        let mut st: Vec<C64> = Vec::with_capacity(16);
        for op in &self.ops {
            match *op {
                COp::Const(c) => st.push(c),
                COp::Load(i) => st.push(inputs.get(i).copied().unwrap_or(C64::ZERO)),
                COp::Neg => match st.pop() {
                    Some(a) => st.push(-a),
                    None => return nan,
                },
                COp::F1(f) => match st.pop() {
                    Some(a) => st.push(apply1(f, a)),
                    None => return nan,
                },
                COp::Add | COp::Sub | COp::Mul | COp::Div | COp::Pow => {
                    let (Some(b), Some(a)) = (st.pop(), st.pop()) else { return nan };
                    st.push(match op {
                        COp::Add => a + b,
                        COp::Sub => a - b,
                        COp::Mul => a * b,
                        COp::Div => a / b,
                        _ => a.pow(b),
                    });
                }
            }
        }
        match st.as_slice() {
            [r] => *r,
            _ => nan,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Angle};
    use crate::parse::{parse, ParseCtx};
    use std::f64::consts::PI;

    fn close(a: C64, b: C64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    fn assert_close(a: C64, b: C64, tol: f64) {
        assert!(close(a, b, tol), "{a:?} != {b:?}");
    }

    fn ev(src: &str, z: C64) -> C64 {
        let e = parse_complex(src, &ParseCtx::new()).unwrap_or_else(|e| panic!("{src}: {e}"));
        compile_complex(&e, &["z"]).unwrap_or_else(|e| panic!("{src}: {e}")).eval(&[z])
    }

    /// Deterministic pseudo-random points in [-3, 3]^2.
    fn pts(n: usize) -> Vec<C64> {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0
        };
        (0..n).map(|_| C64::new(next(), next())).collect()
    }

    #[test]
    fn euler_identity() {
        assert_close(C64::new(0.0, PI).exp(), C64::real(-1.0), 1e-12);
        assert_close(ev("e^(i*pi)", C64::ZERO), C64::real(-1.0), 1e-12);
    }

    #[test]
    fn basic_identities() {
        let a = C64::new(1.0, 1.0);
        assert_eq!(a.pow(C64::real(2.0)), C64::new(0.0, 2.0));
        assert_close(C64::real(-1.0).sqrt(), C64::I, 1e-15);
        assert_close(C64::new(-1.0, -0.0).sqrt(), -C64::I, 1e-15);
        assert_close(C64::new(3.0, 4.0).sqrt(), C64::new(2.0, 1.0), 1e-15);
        assert_close(C64::new(3.0, -4.0).sqrt(), C64::new(2.0, -1.0), 1e-15);
        assert_close(C64::new(-3.0, 4.0).sqrt() * C64::new(-3.0, 4.0).sqrt(), C64::new(-3.0, 4.0), 1e-14);
        assert_eq!(C64::ZERO.sqrt(), C64::ZERO);
    }

    #[test]
    fn ln_branch_cut_signed_zero() {
        let up = C64::new(-1.0, 0.0).ln();
        let dn = C64::new(-1.0, -0.0).ln();
        assert_close(up, C64::new(0.0, PI), 1e-15);
        assert_close(dn, C64::new(0.0, -PI), 1e-15);
        assert_close(C64::new(0.0, 1.0).ln(), C64::new(0.0, PI / 2.0), 1e-15);
    }

    #[test]
    fn sin_of_i() {
        assert_close(C64::I.sin(), C64::new(0.0, 1f64.sinh()), 1e-15);
        assert_close(C64::I.cos(), C64::real(1f64.cosh()), 1e-15);
    }

    #[test]
    fn pythagorean_and_hyperbolic_identities() {
        for z in pts(200) {
            let s = z.sin();
            let c = z.cos();
            assert_close(s * s + c * c, C64::ONE, 1e-12);
            let sh = z.sinh();
            let ch = z.cosh();
            assert_close(ch * ch - sh * sh, C64::ONE, 1e-11);
            assert_close(z.tan(), s / c, 1e-11);
            assert_close(z.tanh(), sh / ch, 1e-11);
        }
    }

    #[test]
    fn exp_ln_round_trip() {
        for z in pts(200) {
            assert_close(z.ln().exp(), z, 1e-12);
            let l = z.exp().ln();
            // principal ln(exp(z)) = z up to a multiple of 2 pi i in the imaginary part.
            assert!((l.re - z.re).abs() < 1e-12);
            let k = (l.im - z.im) / (2.0 * PI);
            assert!((k - k.round()).abs() < 1e-9);
        }
    }

    #[test]
    fn division() {
        for (a, b) in pts(100).into_iter().zip(pts(101).into_iter().skip(1)) {
            assert_close((a / b) * b, a, 1e-12);
        }
        assert_close(C64::new(1.0, 0.0) / C64::new(0.0, 1.0), C64::new(0.0, -1.0), 1e-15);
        // No overflow where the naive |b|^2 formula would overflow.
        let big = C64::new(1e200, 1e200);
        let q = big / big;
        assert_close(q, C64::ONE, 1e-14);
        let tiny = C64::new(1e-200, 1e-200);
        assert_close(tiny / tiny, C64::ONE, 1e-14);
        let inf = C64::ONE / C64::ZERO;
        assert!(inf.re.is_infinite() || inf.re.is_nan());
    }

    #[test]
    fn integer_pow_matches_repeated_multiplication() {
        for z in pts(50) {
            let mut acc = C64::ONE;
            for n in 0..=10 {
                assert_close(z.pow(C64::real(n as f64)), acc, 1e-13);
                acc = acc * z;
            }
            assert_close(z.pow(C64::real(-3.0)), C64::ONE / (z * z * z), 1e-12);
            // Integer path agrees with the generic exp/ln path.
            assert_close(z.powi(5), (C64::real(5.0) * z.ln()).exp(), 1e-10);
        }
        assert_eq!(C64::ZERO.pow(C64::ZERO), C64::ONE);
        assert_eq!(C64::ZERO.pow(C64::real(2.0)), C64::ZERO);
        assert_eq!(C64::ZERO.pow(C64::real(0.5)), C64::ZERO);
        assert!(C64::ZERO.pow(C64::real(-0.5)).is_nan());
        // z^(1/2) principal equals sqrt.
        assert_close(C64::new(-4.0, 1.0).pow(C64::real(0.5)), C64::new(-4.0, 1.0).sqrt(), 1e-13);
        // i^i = e^(-pi/2)
        assert_close(C64::I.pow(C64::I), C64::real((-PI / 2.0).exp()), 1e-13);
    }

    #[test]
    fn real_axis_agrees_with_real_program() {
        for src in ["sqrt(x)", "exp(x)", "sin(x)", "cos(x)", "tan(x)", "ln(x)", "x^2+1", "x^0.5", "sinh(x)", "cosh(x)", "tanh(x)", "abs(x)", "log(x)", "sec(x)+cot(x)", "1/x^3"] {
            let e = parse(src).unwrap();
            let rp = compile(&e, &["x"], Angle::Rad).unwrap();
            let cp = compile_complex(&e, &["x"]).unwrap();
            for k in 1..40 {
                let x = k as f64 * 0.137;
                let r = rp.eval(&[x]);
                let c = cp.eval(&[C64::real(x)]);
                assert!((c.re - r).abs() <= 1e-12 * (1.0 + r.abs()), "{src} at {x}: {c:?} vs {r}");
                assert!(c.im.abs() <= 1e-12 * (1.0 + r.abs()), "{src} at {x}: im {}", c.im);
            }
        }
    }

    #[test]
    fn parsed_expressions() {
        let z = C64::new(0.7, -1.3);
        assert_close(ev("z^2+1", z), z * z + C64::ONE, 1e-14);
        assert_close(ev("(z-1)/(z^2+z+1)", z), (z - C64::ONE) / (z * z + z + C64::ONE), 1e-13);
        assert_close(ev("e^(i*z)", z), (C64::I * z).exp(), 1e-13);
        assert_close(ev("abs(z)", z), C64::real(z.abs()), 1e-15);
        assert_close(ev("conj(z)*z", z), C64::real(z.abs() * z.abs()), 1e-14);
        assert_close(ev("arg(z)+re(z)+im(z)", z), C64::real(z.arg() + z.re + z.im), 1e-14);
        assert_close(ev("log(z)", z), z.ln() / C64::real(LN_10), 1e-14);
        assert_close(ev("2i", z), C64::new(0.0, 2.0), 1e-15);
        assert_close(ev("-z", z), -z, 0.0);
    }

    #[test]
    fn errors() {
        let c = |s: &str| compile_complex(&parse(s).unwrap(), &["z"]);
        assert_eq!(c("w+1"), Err(CompileError::UnboundVariable("w".into())));
        assert!(matches!(c("sin(z,z)"), Err(CompileError::Arity { .. })));
        assert!(matches!(c("(z,z)"), Err(CompileError::NotScalar(_))));
        assert!(matches!(c("z=1"), Err(CompileError::NotScalar(_))));
        assert!(matches!(c("floor(z)"), Err(CompileError::UnknownFunction(_))));
        // `i` is the unit unless it is an input variable.
        let p = compile_complex(&parse("i").unwrap(), &["i"]).unwrap();
        assert_eq!(p.eval(&[C64::real(5.0)]), C64::real(5.0));
        assert_eq!(c("i").unwrap().eval(&[C64::ZERO]), C64::I);
    }

    #[test]
    fn hostile_strings_never_panic() {
        let nasty = [
            "", "((((", "))))", "z^^2", "1/0", "z/(z-z)", "0^0", "ln(0)", "sqrt(", "1e999", "z^1e308",
            "z^(z^(z^z))", "𝔷^2", "\\frac{", "\\x", "z\u{0}+1", "fn main() {}", "}; evil(); {",
            "sin(cos(tan(exp(ln(sqrt(z))))))", &"(".repeat(100), &"z+".repeat(1000),
            &format!("{}z{}", "-(".repeat(60), ")".repeat(60)),
        ];
        for s in nasty {
            let Ok(e) = parse(s) else { continue };
            if let Ok(p) = compile_complex(&e, &["z"]) {
                let _ = p.eval(&[C64::new(0.3, -0.2)]);
                let _ = p.eval(&[]);
            }
        }
        let bad = CProgram { ops: vec![COp::Add], vars: vec![] };
        assert!(bad.eval(&[]).is_nan());
        let bad = CProgram { ops: vec![COp::Const(C64::ONE), COp::Const(C64::ONE)], vars: vec![] };
        assert!(bad.eval(&[]).is_nan());
    }
}
