//! Outward-rounded interval arithmetic.
//!
//! Soundness contract: for every point `p` inside the input boxes whose f64 evaluation is
//! not NaN, the interval result contains that f64 value. An empty interval (NaN bounds)
//! means the expression is undefined on the whole box. Used for implicit-curve pruning,
//! discontinuity detection and points of interest.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl PartialEq for Interval {
    fn eq(&self, o: &Self) -> bool {
        (self.is_empty() && o.is_empty()) || (self.lo == o.lo && self.hi == o.hi)
    }
}

// The arithmetic methods take `self` by value and mirror `std::ops` names on purpose: they
// return an enclosure, not a `Self`-closed operator result with operator-trait semantics.
#[allow(clippy::should_implement_trait)]
impl Interval {
    pub const EMPTY: Interval = Interval { lo: f64::NAN, hi: f64::NAN };
    pub const ENTIRE: Interval = Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY };

    pub fn new(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            Self::EMPTY
        } else {
            Interval { lo, hi }
        }
    }

    pub fn point(x: f64) -> Self {
        if x.is_nan() {
            Self::EMPTY
        } else {
            Interval { lo: x, hi: x }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.lo.is_nan() || self.hi.is_nan()
    }

    pub fn contains(&self, x: f64) -> bool {
        !self.is_empty() && self.lo <= x && x <= self.hi
    }

    pub fn contains_zero(&self) -> bool {
        self.contains(0.0)
    }

    pub fn width(&self) -> f64 {
        if self.is_empty() {
            f64::NAN
        } else {
            self.hi - self.lo
        }
    }

    pub fn mid(&self) -> f64 {
        if self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY {
            0.0
        } else if self.lo == f64::NEG_INFINITY {
            self.hi
        } else if self.hi == f64::INFINITY {
            self.lo
        } else {
            self.lo * 0.5 + self.hi * 0.5
        }
    }

    fn widen(self) -> Self {
        if self.is_empty() {
            self
        } else {
            Interval { lo: self.lo.next_down(), hi: self.hi.next_up() }
        }
    }

    fn widen_n(self, n: usize) -> Self {
        let mut r = self;
        for _ in 0..n {
            r = r.widen();
        }
        r
    }

    /// Image under a nondecreasing function, widened outward.
    fn inc(self, f: impl Fn(f64) -> f64) -> Self {
        if self.is_empty() {
            return self;
        }
        Interval::new(f(self.lo), f(self.hi)).widen()
    }

    pub fn neg(self) -> Self {
        if self.is_empty() {
            self
        } else {
            Interval { lo: -self.hi, hi: -self.lo }
        }
    }

    pub fn add(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        Interval::new(self.lo + o.lo, self.hi + o.hi).widen()
    }

    pub fn sub(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        Interval::new(self.lo - o.hi, self.hi - o.lo).widen()
    }

    pub fn mul(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        // 0 * inf is 0 in interval arithmetic.
        let m = |a: f64, b: f64| if a == 0.0 || b == 0.0 { 0.0 } else { a * b };
        let c = [m(self.lo, o.lo), m(self.lo, o.hi), m(self.hi, o.lo), m(self.hi, o.hi)];
        let lo = c.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Interval::new(lo, hi).widen()
    }

    pub fn div(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        if o.lo == 0.0 && o.hi == 0.0 {
            return Self::EMPTY;
        }
        if o.lo < 0.0 && o.hi > 0.0 {
            return Self::ENTIRE;
        }
        if o.lo == 0.0 {
            // o = [0, h], h > 0: a/o covers a half line when a excludes 0, otherwise everything.
            if self.lo > 0.0 {
                return Interval { lo: (self.lo / o.hi).next_down(), hi: f64::INFINITY };
            }
            if self.hi < 0.0 {
                return Interval { lo: f64::NEG_INFINITY, hi: (self.hi / o.hi).next_up() };
            }
            return Self::ENTIRE;
        }
        if o.hi == 0.0 {
            if self.lo > 0.0 {
                return Interval { lo: f64::NEG_INFINITY, hi: (self.lo / o.lo).next_up() };
            }
            if self.hi < 0.0 {
                return Interval { lo: (self.hi / o.lo).next_down(), hi: f64::INFINITY };
            }
            return Self::ENTIRE;
        }
        let recip = Interval::new(1.0 / o.hi, 1.0 / o.lo).widen();
        self.mul(recip)
    }

    pub fn abs(self) -> Self {
        if self.is_empty() || self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Interval { lo: 0.0, hi: (-self.lo).max(self.hi) }
        }
    }

    pub fn sqrt(self) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        Interval::new(self.lo.max(0.0).sqrt(), self.hi.sqrt()).widen()
    }

    pub fn exp(self) -> Self {
        self.inc(f64::exp)
    }

    fn log_with(self, f: impl Fn(f64) -> f64) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        let lo = if self.lo <= 0.0 { f64::NEG_INFINITY } else { f(self.lo) };
        Interval::new(lo, f(self.hi)).widen()
    }

    pub fn ln(self) -> Self {
        self.log_with(f64::ln)
    }

    pub fn log10(self) -> Self {
        self.log_with(f64::log10)
    }

    pub fn cbrt(self) -> Self {
        self.inc(f64::cbrt)
    }

    pub fn atan(self) -> Self {
        self.inc(f64::atan)
    }

    pub fn sinh(self) -> Self {
        self.inc(f64::sinh)
    }

    pub fn tanh(self) -> Self {
        self.inc(f64::tanh)
    }

    pub fn cosh(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let (a, b) = (self.lo.cosh(), self.hi.cosh());
        if self.lo >= 0.0 {
            Interval::new(a, b)
        } else if self.hi <= 0.0 {
            Interval::new(b, a)
        } else {
            Interval::new(1.0, a.max(b))
        }
        .widen()
    }

    pub fn asin(self) -> Self {
        if self.is_empty() || self.hi < -1.0 || self.lo > 1.0 {
            return Self::EMPTY;
        }
        Interval::new(self.lo.max(-1.0).asin(), self.hi.min(1.0).asin()).widen()
    }

    pub fn acos(self) -> Self {
        if self.is_empty() || self.hi < -1.0 || self.lo > 1.0 {
            return Self::EMPTY;
        }
        Interval::new(self.hi.min(1.0).acos(), self.lo.max(-1.0).acos()).widen()
    }

    pub fn floor(self) -> Self {
        if self.is_empty() {
            self
        } else {
            Interval { lo: self.lo.floor(), hi: self.hi.floor() }
        }
    }

    pub fn ceil(self) -> Self {
        if self.is_empty() {
            self
        } else {
            Interval { lo: self.lo.ceil(), hi: self.hi.ceil() }
        }
    }

    pub fn round(self) -> Self {
        if self.is_empty() {
            self
        } else {
            Interval { lo: self.lo.round(), hi: self.hi.round() }
        }
    }

    pub fn sign(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let s = |x: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
        Interval { lo: s(self.lo), hi: s(self.hi) }
    }

    /// Pads by a relative epsilon so that rounded π constants cannot misplace an extremum.
    fn trig_pad(self) -> Option<Self> {
        if self.is_empty() || !self.lo.is_finite() || !self.hi.is_finite() {
            return None;
        }
        let mag = self.lo.abs().max(self.hi.abs());
        if mag > 1.0e7 || self.hi - self.lo >= TAU {
            return None;
        }
        let d = 1.0e-15 * (1.0 + mag);
        Some(Interval { lo: self.lo - d, hi: self.hi + d })
    }

    pub fn cos(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let x = match self.trig_pad() {
            Some(x) => x,
            None => return Interval { lo: -1.0, hi: 1.0 },
        };
        let (c1, c2) = (x.lo.cos(), x.hi.cos());
        let mut lo = c1.min(c2);
        let mut hi = c1.max(c2);
        if (x.lo / TAU).ceil() * TAU <= x.hi {
            hi = 1.0;
        }
        if PI + ((x.lo - PI) / TAU).ceil() * TAU <= x.hi {
            lo = -1.0;
        }
        Interval { lo: lo.max(-1.0), hi: hi.min(1.0) }.widen_n(2)
    }

    pub fn sin(self) -> Self {
        if self.is_empty() {
            return self;
        }
        // sin(x) = cos(x - pi/2); trig_pad inside cos absorbs the rounding of pi/2.
        Interval { lo: self.lo - FRAC_PI_2, hi: self.hi - FRAC_PI_2 }.cos()
    }

    pub fn tan(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let x = match self.trig_pad() {
            Some(x) => x,
            None => return Self::ENTIRE,
        };
        if FRAC_PI_2 + ((x.lo - FRAC_PI_2) / PI).ceil() * PI <= x.hi {
            return Self::ENTIRE;
        }
        Interval::new(x.lo.tan(), x.hi.tan()).widen_n(2)
    }

    /// `self ^ n` for an integer exponent.
    fn powi(self, n: i32) -> Self {
        if self.is_empty() {
            return self;
        }
        if n == 0 {
            return Interval { lo: 1.0, hi: 1.0 };
        }
        if n < 0 {
            let p = self.powi(-n);
            return Interval { lo: 1.0, hi: 1.0 }.div(p);
        }
        let f = |x: f64| x.powi(n);
        if n % 2 == 1 {
            return Interval::new(f(self.lo), f(self.hi)).widen();
        }
        if self.lo >= 0.0 {
            Interval::new(f(self.lo), f(self.hi))
        } else if self.hi <= 0.0 {
            Interval::new(f(self.hi), f(self.lo))
        } else {
            Interval::new(0.0, f(self.lo).max(f(self.hi)))
        }
        .widen()
    }

    pub fn pow(self, e: Self) -> Self {
        if self.is_empty() || e.is_empty() {
            return Self::EMPTY;
        }
        if e.lo == e.hi && e.lo.fract() == 0.0 && e.lo.abs() <= 1.0e9 {
            return self.powi(e.lo as i32);
        }
        if self.lo < 0.0 {
            // Negative bases are defined only for integer exponents. A point non-integer
            // exponent makes them NaN, so clamp; an exponent range could hit integers.
            if e.lo != e.hi {
                return Self::ENTIRE;
            }
            if self.hi < 0.0 {
                return Self::EMPTY;
            }
        }
        let base = Interval { lo: self.lo.max(0.0), hi: self.hi };
        e.mul(base.ln()).exp()
    }

    /// `x!` = Gamma(x + 1). Gamma is convex and positive on `z > 0` with one minimum at
    /// `z0 ~ 1.4616` (value `0.88560319...`), so a box inside `x > -1` is bounded by its
    /// endpoint values and that minimum. A box reaching `x <= -1` crosses the poles and sign
    /// changes of the negative branches: whole line.
    pub fn factorial(self) -> Self {
        if self.is_empty() {
            return self;
        }
        if self.lo == self.hi && self.lo.is_finite() {
            return Interval::point(crate::stats::factorial(self.lo));
        }
        if self.lo <= -1.0 {
            return Self::ENTIRE;
        }
        const Z0: f64 = 1.461_632_144_968_362_3;
        const GAMMA_MIN: f64 = 0.885_603_194_410_888_7;
        let f = crate::stats::factorial;
        let (a, b) = (f(self.lo), f(self.hi));
        if a.is_nan() || b.is_nan() {
            return Self::ENTIRE;
        }
        // `lo` just above -1 (Gamma near its pole) evaluates to a huge but finite value.
        let hi = a.max(b);
        let lo = if self.lo + 1.0 <= Z0 && Z0 <= self.hi + 1.0 { GAMMA_MIN } else { a.min(b) };
        // Lanczos is not exactly monotone near the minimum, so pad by a relative 1e-10.
        let pad = |v: f64, up: bool| {
            if !v.is_finite() {
                v
            } else if up {
                v + v.abs() * 1e-10 + f64::MIN_POSITIVE
            } else {
                v - v.abs() * 1e-10 - f64::MIN_POSITIVE
            }
        };
        Interval::new(pad(lo, false), pad(hi, true))
    }

    /// Enclosure of `nCr` (`perm = false`) or `nPr(self, k)`: exact for point arguments,
    /// otherwise the loose but finite `[0, 1e300]` (an invalid argument is NaN, so the result is
    /// never negative); empty when `n` is entirely negative.
    pub fn comb_enclosure(self, k: Self, perm: bool) -> Self {
        if self.is_empty() || k.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        if self.lo == self.hi && k.lo == k.hi {
            let f = if perm { crate::stats::permute } else { crate::stats::choose };
            return Interval::point(f(self.lo, k.lo));
        }
        Interval::new(0.0, 1e300)
    }

    pub fn atan2(self, x: Self) -> Self {
        if self.is_empty() || x.is_empty() {
            return Self::EMPTY;
        }
        Interval { lo: -PI, hi: PI }.widen()
    }

    pub fn min(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        Interval { lo: self.lo.min(o.lo), hi: self.hi.min(o.hi) }
    }

    pub fn max(self, o: Self) -> Self {
        if self.is_empty() || o.is_empty() {
            return Self::EMPTY;
        }
        Interval { lo: self.lo.max(o.lo), hi: self.hi.max(o.hi) }
    }

    /// `a - b * floor(a / b)`.
    pub fn modulo(self, b: Self) -> Self {
        if self.is_empty() || b.is_empty() {
            return Self::EMPTY;
        }
        if b.lo > 0.0 {
            Interval { lo: 0.0, hi: b.hi }.widen()
        } else {
            Self::ENTIRE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(a: f64, b: f64) -> Interval {
        Interval::new(a, b)
    }

    #[test]
    fn basic_arithmetic_encloses() {
        let r = iv(1.0, 2.0).add(iv(3.0, 4.0));
        assert!(r.contains(4.0) && r.contains(6.0) && !r.contains(3.9));
        let r = iv(-2.0, 3.0).mul(iv(-1.0, 4.0));
        assert!(r.contains(-8.0) && r.contains(12.0));
    }

    #[test]
    fn division_by_zero_straddle_is_entire() {
        assert_eq!(iv(1.0, 2.0).div(iv(-1.0, 1.0)), Interval::ENTIRE);
        assert!(iv(1.0, 2.0).div(iv(0.0, 0.0)).is_empty());
        let r = iv(1.0, 2.0).div(iv(0.0, 4.0));
        assert!(r.contains(0.25) && r.hi == f64::INFINITY && r.lo < 0.26);
    }

    #[test]
    fn even_power_includes_zero_minimum() {
        let r = iv(-3.0, 2.0).pow(Interval::point(2.0));
        assert!(r.lo <= 0.0 && r.contains(9.0) && r.contains(0.0));
    }

    #[test]
    fn sqrt_clamps_negative_part() {
        let r = iv(-4.0, 9.0).sqrt();
        assert!(r.contains(0.0) && r.contains(3.0));
        assert!(iv(-4.0, -1.0).sqrt().is_empty());
    }

    #[test]
    fn trig_extrema() {
        let r = iv(0.0, 4.0).sin();
        assert!(r.contains(1.0) && r.contains(-0.7568));
        let r = iv(0.0, 1.0).cos();
        assert!(r.hi >= 1.0 && r.lo <= 0.54031 && r.lo > 0.5);
        assert_eq!(iv(0.0, 100.0).sin().hi.min(1.0), 1.0);
        assert!(iv(1.0, 2.0).tan() == Interval::ENTIRE);
        assert!(iv(0.1, 0.5).tan().hi < 1.0);
    }

    #[test]
    fn ln_domain() {
        assert!(iv(-2.0, -1.0).ln().is_empty());
        let r = iv(-1.0, 1.0).ln();
        assert_eq!(r.lo, f64::NEG_INFINITY);
        assert!(r.contains(0.0));
    }
}
