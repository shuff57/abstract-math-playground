//! Pure `f64` statistics: list aggregates and probability distributions.
//!
//! Conventions
//! * Aggregates take `&[f64]` and return `Result<_, StatsError>` (empty input, length
//!   mismatch, too few samples, bad argument).
//! * NaN handling: sums/means/variances propagate NaN naturally; order statistics (`min`,
//!   `max`, `median`, `quantile`, `quartile`) return NaN if any element is NaN; `sort` places
//!   NaN last (IEEE total order on positive NaN); `unique` treats NaN as one value.
//! * Quantile method: **linear interpolation between order statistics, R type 7**
//!   (`h = (n-1)p`, interpolate `x[floor h]` and `x[ceil h]`). This is the spreadsheet/NumPy
//!   default; the median is the usual median and `quartile(L, q) = quantile(L, q/4)`.
//!   (Desmos does not document an interpolation rule, so this is a documented choice.)
//! * Distribution functions take scalars and return NaN for invalid parameters
//!   (e.g. `sigma <= 0`, `p` outside `[0,1]`, non-positive `df`).
//! * Special functions are implemented here: `erf`/`erfc` (power series for `|x| < 1`,
//!   backward-evaluated continued fraction for `|x| >= 1`, ~1e-15 relative), `ln_gamma` (Lanczos g=7, n=9),
//!   regularised incomplete beta and gamma (continued fractions / series), `invnorm`
//!   (Acklam's rational approximation plus two Halley steps).

// `!(x > 0.0)` is deliberate: it is also true for NaN, which must be rejected.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::f64::consts::{PI, SQRT_2};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    Empty,
    LengthMismatch { a: usize, b: usize },
    TooFew { need: usize, got: usize },
    BadArg(&'static str),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::Empty => write!(f, "the list is empty"),
            StatsError::LengthMismatch { a, b } => write!(f, "list lengths differ ({a} vs {b})"),
            StatsError::TooFew { need, got } => write!(f, "need at least {need} values, got {got}"),
            StatsError::BadArg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StatsError {}

type R<T> = Result<T, StatsError>;

fn nonempty(x: &[f64]) -> R<()> {
    if x.is_empty() {
        Err(StatsError::Empty)
    } else {
        Ok(())
    }
}

fn at_least(x: &[f64], need: usize) -> R<()> {
    if x.len() < need {
        Err(StatsError::TooFew { need, got: x.len() })
    } else {
        Ok(())
    }
}

fn same_len(a: &[f64], b: &[f64]) -> R<()> {
    if a.len() != b.len() {
        Err(StatsError::LengthMismatch { a: a.len(), b: b.len() })
    } else {
        Ok(())
    }
}

// ------------------------------------------------------------------ aggregates

pub fn length(x: &[f64]) -> f64 {
    x.len() as f64
}

/// Neumaier compensated sum. The empty sum is 0.
pub fn total(x: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for &v in x {
        let t = s + v;
        if s.abs() >= v.abs() {
            c += (s - t) + v;
        } else {
            c += (v - t) + s;
        }
        s = t;
    }
    let r = s + c;
    if r.is_nan() && !x.iter().any(|v| v.is_nan()) {
        // inf - inf during compensation: fall back to the plain sum.
        x.iter().sum()
    } else {
        r
    }
}

pub fn mean(x: &[f64]) -> R<f64> {
    nonempty(x)?;
    Ok(total(x) / x.len() as f64)
}

fn has_nan(x: &[f64]) -> bool {
    x.iter().any(|v| v.is_nan())
}

pub fn min(x: &[f64]) -> R<f64> {
    nonempty(x)?;
    if has_nan(x) {
        return Ok(f64::NAN);
    }
    Ok(x.iter().copied().fold(f64::INFINITY, f64::min))
}

pub fn max(x: &[f64]) -> R<f64> {
    nonempty(x)?;
    if has_nan(x) {
        return Ok(f64::NAN);
    }
    Ok(x.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

fn sum_sq_dev(x: &[f64]) -> f64 {
    let m = total(x) / x.len() as f64;
    let sq: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
    total(&sq)
}

/// Sample variance (divisor n-1). Needs at least 2 values.
pub fn var(x: &[f64]) -> R<f64> {
    at_least(x, 2)?;
    Ok(sum_sq_dev(x) / (x.len() - 1) as f64)
}

/// Population variance (divisor n).
pub fn varp(x: &[f64]) -> R<f64> {
    nonempty(x)?;
    Ok(sum_sq_dev(x) / x.len() as f64)
}

pub fn stdev(x: &[f64]) -> R<f64> {
    var(x).map(f64::sqrt)
}

pub fn stdevp(x: &[f64]) -> R<f64> {
    varp(x).map(f64::sqrt)
}

/// Mean absolute deviation about the mean.
pub fn mad(x: &[f64]) -> R<f64> {
    let m = mean(x)?;
    let d: Vec<f64> = x.iter().map(|v| (v - m).abs()).collect();
    Ok(total(&d) / x.len() as f64)
}

/// Sorted copy (ascending, NaN last).
pub fn sort(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Quantile by linear interpolation (R type 7); see the module docs. `p` in `[0, 1]`.
pub fn quantile(x: &[f64], p: f64) -> R<f64> {
    nonempty(x)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::BadArg("quantile needs 0 <= p <= 1"));
    }
    if has_nan(x) {
        return Ok(f64::NAN);
    }
    let s = sort(x);
    let h = (s.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let frac = h - lo as f64;
    Ok(if frac == 0.0 { s[lo] } else { s[lo] + frac * (s[hi] - s[lo]) })
}

pub fn median(x: &[f64]) -> R<f64> {
    quantile(x, 0.5)
}

/// `q` in {0,1,2,3,4}: min, Q1, median, Q3, max.
pub fn quartile(x: &[f64], q: f64) -> R<f64> {
    if !(q == 0.0 || q == 1.0 || q == 2.0 || q == 3.0 || q == 4.0) {
        return Err(StatsError::BadArg("quartile index must be 0, 1, 2, 3 or 4"));
    }
    quantile(x, q / 4.0)
}

/// Sorts `x` by `keys` (stable, ascending keys).
pub fn sort_by_keys(x: &[f64], keys: &[f64]) -> R<Vec<f64>> {
    same_len(x, keys)?;
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| keys[i].total_cmp(&keys[j]));
    Ok(idx.into_iter().map(|i| x[i]).collect())
}

pub fn reverse(x: &[f64]) -> Vec<f64> {
    x.iter().rev().copied().collect()
}

pub fn join(parts: &[&[f64]]) -> Vec<f64> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// Sorted distinct values.
pub fn unique(x: &[f64]) -> Vec<f64> {
    let mut s = sort(x);
    s.dedup_by(|a, b| a == b || (a.is_nan() && b.is_nan()));
    s
}

/// Sample covariance (divisor n-1).
pub fn cov(x: &[f64], y: &[f64]) -> R<f64> {
    same_len(x, y)?;
    at_least(x, 2)?;
    let mx = total(x) / x.len() as f64;
    let my = total(y) / y.len() as f64;
    let p: Vec<f64> = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).collect();
    Ok(total(&p) / (x.len() - 1) as f64)
}

/// Pearson correlation. NaN when either list is constant.
pub fn corr(x: &[f64], y: &[f64]) -> R<f64> {
    same_len(x, y)?;
    at_least(x, 2)?;
    let c = cov(x, y)?;
    let r = c / (var(x)?.sqrt() * var(y)?.sqrt());
    Ok(if r.is_finite() { r.clamp(-1.0, 1.0) } else { f64::NAN })
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation (average ranks for ties).
pub fn spearman(x: &[f64], y: &[f64]) -> R<f64> {
    same_len(x, y)?;
    at_least(x, 2)?;
    corr(&ranks(x), &ranks(y))
}

// ------------------------------------------------------------------ special functions

const SQRT_PI: f64 = 1.772_453_850_905_516;

/// `erf(x)` for `0 <= x < 1` by the all-positive series
/// `2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (2n+1)!!`.
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0.0;
    while term.abs() > 1e-17 * sum.abs() && n < 200.0 {
        n += 1.0;
        term *= 2.0 * x2 / (2.0 * n + 1.0);
        sum += term;
    }
    2.0 / SQRT_PI * (-x2).exp() * sum
}

/// `erfc(x)` for `x >= 1` by the continued fraction
/// `e^{-x^2}/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`.
fn erfc_cf(x: f64) -> f64 {
    // Evaluated bottom-up with a fixed depth: backward recurrence does not accumulate
    // rounding error the way the forward Lentz product does.
    let n: u32 = if x < 2.0 {
        400
    } else if x < 4.0 {
        100
    } else {
        50
    };
    let mut t = x;
    for k in (1..=n).rev() {
        t = x + (k as f64 / 2.0) / t;
    }
    (-x * x).exp() / SQRT_PI / t
}

pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 1.0 {
        1.0 - erf_series(x)
    } else if x > 27.3 {
        0.0
    } else {
        erfc_cf(x)
    }
}

pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x < 1.0 {
        erf_series(x)
    } else {
        1.0 - erfc(x)
    }
}

const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of |Gamma(x)| (Lanczos, g = 7, 9 terms; reflection for x < 0.5).
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.5 {
        let s = (PI * x).sin();
        if s == 0.0 {
            return f64::INFINITY;
        }
        return (PI / s.abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + 7.5;
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The gamma function `Gamma(z)`: exact products for small positive integers, Lanczos (via
/// [`ln_gamma`]) elsewhere, reflection for `z < 0.5`. Non-positive integers (poles) are NaN, and
/// the result overflows to infinity past `z ~ 171.6`.
pub fn gamma(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == z.floor() {
        if z <= 0.0 {
            return f64::NAN;
        }
        if z <= 171.0 {
            return (2..z as u64).fold(1.0, |a, k| a * k as f64);
        }
        return f64::INFINITY;
    }
    if z < 0.5 {
        let s = (PI * z).sin();
        return PI / (s * gamma(1.0 - z));
    }
    ln_gamma(z).exp()
}

/// `x!` as `Gamma(x + 1)`: exact for integers up to 170, infinity beyond, NaN for negative
/// integers.
pub fn factorial(x: f64) -> f64 {
    gamma(x + 1.0)
}

/// Binomial coefficient `nCr(n, k)` for integer `n >= 0` and `k`; 0 outside `0 <= k <= n`, NaN
/// for non-integers or a negative `n`.
pub fn choose(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || n != n.floor() || k != k.floor() || n < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 || k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    if k > 1000.0 {
        return (ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)).exp().round();
    }
    let mut r = 1.0;
    for i in 1..=(k as u64) {
        r = r * (n - k + i as f64) / i as f64;
    }
    r.round()
}

/// Permutations `nPr(n, k) = n!/(n-k)!` for integer `n >= 0` and `k`; 0 outside `0 <= k <= n`.
/// `v` as the fraction `(numerator, denominator)` (denominator at least 2, at most 10 000) when
/// one reproduces it to about 1e-9 relative, by continued fractions; `None` for an integer, a
/// non-finite number or a value with no such fraction (`pi`, `sqrt(2)`).
pub fn as_fraction(v: f64) -> Option<(i64, u64)> {
    if !v.is_finite() || v == v.round() || v.abs() > 1e9 {
        return None;
    }
    let target = v.abs();
    let tol = 1e-9 * target.max(1.0);
    // convergents h/k of the continued fraction of `target`
    let (mut h0, mut k0, mut h1, mut k1) = (0i64, 1i64, 1i64, 0i64);
    let mut x = target;
    for _ in 0..40 {
        let a = x.floor();
        if a > 1e12 {
            return None;
        }
        let a = a as i64;
        let (h2, k2) = (a.checked_mul(h1)?.checked_add(h0)?, a.checked_mul(k1)?.checked_add(k0)?);
        if k2 > 10_000 {
            return None;
        }
        (h0, k0, h1, k1) = (h1, k1, h2, k2);
        if (h1 as f64 / k1 as f64 - target).abs() <= tol {
            if k1 < 2 {
                return None;
            }
            return Some((if v < 0.0 { -h1 } else { h1 }, k1 as u64));
        }
        let frac = x - a as f64;
        if frac < 1e-15 {
            return None;
        }
        x = 1.0 / frac;
    }
    None
}

/// Greatest common divisor of two integers (`gcd(0, 0)` is 0). NaN when either is not an integer.
pub fn gcd(a: f64, b: f64) -> f64 {
    if !(a.is_finite() && b.is_finite()) || a != a.trunc() || b != b.trunc() {
        return f64::NAN;
    }
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0.0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Least common multiple of two integers (0 when either is 0). NaN when either is not an integer.
pub fn lcm(a: f64, b: f64) -> f64 {
    let g = gcd(a, b);
    if g.is_nan() {
        return f64::NAN;
    }
    if g == 0.0 {
        return 0.0;
    }
    (a / g * b).abs()
}

pub fn permute(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || n != n.floor() || k != k.floor() || n < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 || k > n {
        return 0.0;
    }
    if k > 1000.0 {
        return (ln_gamma(n + 1.0) - ln_gamma(n - k + 1.0)).exp().round();
    }
    let mut r = 1.0;
    for i in 0..(k as u64) {
        r *= n - i as f64;
    }
    r
}

fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta `I_x(a, b)` where `y = 1 - x` is supplied separately so
/// callers that know `y` exactly keep full precision near `x = 1`.
pub fn inc_beta_xy(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * y.ln();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, y) / b
    }
}

pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    inc_beta_xy(a, b, x, 1.0 - x)
}

/// Regularised lower incomplete gamma `P(a, x)`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // series
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..100_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..100_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Regularised upper incomplete gamma `Q(a, x) = 1 - P(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x < a + 1.0 {
        1.0 - gamma_p(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

// ------------------------------------------------------------------ normal

pub fn normalpdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0) || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * PI).sqrt())
}

/// Standard normal CDF, accurate in both tails.
pub fn std_normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / SQRT_2)
}

/// `P(a < X < b)` for `X ~ N(mu, sigma^2)`; `a` and `b` may be infinite. If `a >= b` the
/// interval is empty and the probability is 0 (so the result always lies in `[0, 1]`).
pub fn normalcdf(a: f64, b: f64, mu: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0) || a.is_nan() || b.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if a >= b {
        return 0.0;
    }
    let (za, zb) = ((a - mu) / sigma, (b - mu) / sigma);
    let p = if za > 0.0 {
        std_normal_cdf(-za) - std_normal_cdf(-zb)
    } else {
        std_normal_cdf(zb) - std_normal_cdf(za)
    };
    p.clamp(0.0, 1.0)
}

const ACK_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];
const ACK_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];
const ACK_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];
const ACK_D: [f64; 4] =
    [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];

/// Inverse standard normal CDF: Acklam's approximation (rel. error ~1e-9) refined by two
/// Halley steps against the accurate `erfc`.
pub fn std_invnorm(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let plow = 0.02425;
    let mut x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q + ACK_C[3]) * q + ACK_C[4]) * q + ACK_C[5])
            / ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q + ACK_D[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((ACK_A[0] * r + ACK_A[1]) * r + ACK_A[2]) * r + ACK_A[3]) * r + ACK_A[4]) * r + ACK_A[5]) * q
            / (((((ACK_B[0] * r + ACK_B[1]) * r + ACK_B[2]) * r + ACK_B[3]) * r + ACK_B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q + ACK_C[3]) * q + ACK_C[4]) * q + ACK_C[5])
            / ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q + ACK_D[3]) * q + 1.0)
    };
    // Work in the tail that keeps precision: for p > 0.5 mirror to the lower tail.
    let (pp, sign) = if p > 0.5 { (1.0 - p, -1.0) } else { (p, 1.0) };
    if p > 0.5 {
        x = -x;
    }
    for _ in 0..3 {
        let e = std_normal_cdf(x) - pp;
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        let step = u / (1.0 + x * u / 2.0);
        if !step.is_finite() {
            break;
        }
        x -= step;
    }
    sign * x
}

pub fn invnorm(p: f64, mu: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0) || mu.is_nan() {
        return f64::NAN;
    }
    mu + sigma * std_invnorm(p)
}

// ------------------------------------------------------------------ discrete

fn is_int(k: f64) -> bool {
    k.is_finite() && k == k.floor()
}

fn valid_binom(n: f64, p: f64) -> bool {
    n >= 0.0 && n == n.floor() && n.is_finite() && (0.0..=1.0).contains(&p)
}

/// `P(X = k)` for `X ~ Binomial(n, p)`. Non-integer `k` has probability 0.
pub fn binompdf(n: f64, p: f64, k: f64) -> f64 {
    if !valid_binom(n, p) || k.is_nan() {
        return f64::NAN;
    }
    if !is_int(k) || k < 0.0 || k > n {
        return 0.0;
    }
    if p == 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    if n <= 60.0 {
        // Small n: multiplicative binomial coefficient (exact in f64 here) keeps simple
        // cases like binompdf(2, .5, 1) = 0.5 bit-exact.
        let kk = k.min(n - k) as i32;
        let mut c = 1.0f64;
        for i in 1..=kk {
            c = c * (n - kk as f64 + i as f64) / i as f64;
        }
        return c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
    }
    (ln_choose(n, k) + k * p.ln() + (n - k) * (1.0 - p).ln()).exp()
}

/// `P(X <= k)`; `k` is floored.
pub fn binomcdf(n: f64, p: f64, k: f64) -> f64 {
    if !valid_binom(n, p) || k.is_nan() {
        return f64::NAN;
    }
    let k = k.floor();
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    // P(X <= k) = I_{1-p}(n-k, k+1)
    inc_beta_xy(n - k, k + 1.0, 1.0 - p, p)
}

pub fn poissonpdf(lambda: f64, k: f64) -> f64 {
    if lambda.is_nan() || lambda < 0.0 || k.is_nan() {
        return f64::NAN;
    }
    if !is_int(k) || k < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    (k * lambda.ln() - lambda - ln_gamma(k + 1.0)).exp()
}

/// `P(X <= k)`; `k` is floored.
pub fn poissoncdf(lambda: f64, k: f64) -> f64 {
    if lambda.is_nan() || lambda < 0.0 || k.is_nan() {
        return f64::NAN;
    }
    let k = k.floor();
    if k < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 || k.is_infinite() {
        return 1.0;
    }
    gamma_q(k + 1.0, lambda)
}

// ------------------------------------------------------------------ uniform

pub fn uniformpdf(x: f64, a: f64, b: f64) -> f64 {
    if !(b > a) || x.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if x >= a && x <= b {
        1.0 / (b - a)
    } else {
        0.0
    }
}

/// `P(lo < X < hi)` for `X ~ Uniform(a, b)`.
pub fn uniformcdf(a: f64, b: f64, lo: f64, hi: f64) -> f64 {
    if !(b > a) || lo.is_nan() || hi.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    (hi.min(b) - lo.max(a)).max(0.0) / (b - a)
}

// ------------------------------------------------------------------ Student t

pub fn tpdf(x: f64, df: f64) -> f64 {
    if !(df > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    let ln_c = ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0) - 0.5 * (df * PI).ln();
    (ln_c - (df + 1.0) / 2.0 * (x * x / df).ln_1p()).exp()
}

/// Upper tail `P(T > x)` for `x >= 0`.
fn t_sf(x: f64, df: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let x2 = x * x;
    // x_beta = df/(df+x^2), y = x^2/(df+x^2): both exact-ish, no 1-x cancellation.
    let denom = df + x2;
    0.5 * inc_beta_xy(df / 2.0, 0.5, df / denom, x2 / denom)
}

/// `P(T <= x)`.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    if !(df > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        1.0 - t_sf(x, df)
    } else {
        t_sf(-x, df)
    }
}

/// `P(a < T < b)`; 0 when `a >= b`.
pub fn tcdf(a: f64, b: f64, df: f64) -> f64 {
    if !(df > 0.0) || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a >= b {
        return 0.0;
    }
    let p = if a > 0.0 { t_sf(a, df) - t_sf(b, df) } else { t_cdf(b, df) - t_cdf(a, df) };
    p.clamp(0.0, 1.0)
}

/// Quantile of Student t: the `x` with `P(T <= x) = p`. Solved on the accurate upper tail
/// by safeguarded bisection plus Newton polish.
pub fn invt(p: f64, df: f64) -> f64 {
    if !(df > 0.0) || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    let (tail, sign) = if p > 0.5 { (1.0 - p, 1.0) } else { (p, -1.0) };
    // find x >= 0 with t_sf(x) = tail
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    while t_sf(hi, df) > tail && hi < 1e300 {
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if t_sf(mid, df) > tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut x = 0.5 * (lo + hi);
    for _ in 0..2 {
        let d = tpdf(x, df);
        if d > 0.0 && d.is_finite() {
            let nx = x + (t_sf(x, df) - tail) / d;
            if nx.is_finite() && nx >= lo * 0.999_999 && nx <= hi * 1.000_001 + 1e-300 {
                x = nx;
            }
        }
    }
    sign * x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        if a == b {
            0.0
        } else {
            (a - b).abs() / b.abs().max(1e-300)
        }
    }
    fn near(a: f64, b: f64, tol: f64) {
        assert!(rel(a, b) <= tol, "{a} vs {b} (rel {})", rel(a, b));
    }

    #[test]
    fn aggregates_basic() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(length(&x), 8.0);
        assert_eq!(total(&x), 40.0);
        assert_eq!(mean(&x).unwrap(), 5.0);
        assert_eq!(median(&x).unwrap(), 4.5);
        assert_eq!(min(&x).unwrap(), 2.0);
        assert_eq!(max(&x).unwrap(), 9.0);
        assert_eq!(varp(&x).unwrap(), 4.0);
        assert_eq!(stdevp(&x).unwrap(), 2.0);
        near(var(&x).unwrap(), 32.0 / 7.0, 1e-15);
        near(stdev(&x).unwrap(), (32.0f64 / 7.0).sqrt(), 1e-15);
        near(mad(&x).unwrap(), 1.5, 1e-15);
    }

    #[test]
    fn errors() {
        assert_eq!(mean(&[]), Err(StatsError::Empty));
        assert_eq!(median(&[]), Err(StatsError::Empty));
        assert_eq!(var(&[1.0]), Err(StatsError::TooFew { need: 2, got: 1 }));
        assert_eq!(varp(&[1.0]).unwrap(), 0.0);
        assert!(matches!(corr(&[1.0, 2.0], &[1.0]), Err(StatsError::LengthMismatch { .. })));
        assert!(quantile(&[1.0], 1.5).is_err());
        assert!(quartile(&[1.0, 2.0], 2.5).is_err());
        assert_eq!(total(&[]), 0.0);
    }

    #[test]
    fn nan_handling() {
        assert!(mean(&[1.0, f64::NAN]).unwrap().is_nan());
        assert!(min(&[1.0, f64::NAN]).unwrap().is_nan());
        assert!(median(&[1.0, f64::NAN, 3.0]).unwrap().is_nan());
        assert_eq!(sort(&[2.0, f64::NAN, 1.0])[..2], [1.0, 2.0]);
        assert!(sort(&[2.0, f64::NAN, 1.0])[2].is_nan());
        assert!(total(&[f64::INFINITY, 1.0]).is_infinite());
    }

    #[test]
    fn quantiles_type7() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quartile(&x, 0.0).unwrap(), 1.0);
        assert_eq!(quartile(&x, 1.0).unwrap(), 2.0);
        assert_eq!(quartile(&x, 2.0).unwrap(), 3.0);
        assert_eq!(quartile(&x, 3.0).unwrap(), 4.0);
        assert_eq!(quartile(&x, 4.0).unwrap(), 5.0);
        let y = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&y, 0.5).unwrap(), 2.5);
        near(quantile(&y, 0.25).unwrap(), 1.75, 1e-15);
        near(quantile(&y, 0.9).unwrap(), 3.7, 1e-15);
        assert_eq!(quantile(&[7.0], 0.3).unwrap(), 7.0);
        // unsorted input
        assert_eq!(median(&[5.0, 1.0, 3.0]).unwrap(), 3.0);
    }

    #[test]
    fn sort_reverse_join_unique() {
        assert_eq!(sort(&[3.0, 1.0, 2.0]), vec![1.0, 2.0, 3.0]);
        assert_eq!(sort_by_keys(&[10.0, 20.0, 30.0], &[3.0, 1.0, 2.0]).unwrap(), vec![20.0, 30.0, 10.0]);
        assert!(sort_by_keys(&[1.0], &[1.0, 2.0]).is_err());
        assert_eq!(reverse(&[1.0, 2.0, 3.0]), vec![3.0, 2.0, 1.0]);
        assert_eq!(join(&[&[1.0], &[2.0, 3.0]]), vec![1.0, 2.0, 3.0]);
        assert_eq!(unique(&[3.0, 1.0, 3.0, 2.0, 1.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        near(corr(&x, &y).unwrap(), 1.0, 1e-15);
        let z = [5.0, 4.0, 3.0, 2.0, 1.0];
        near(corr(&x, &z).unwrap(), -1.0, 1e-15);
        near(cov(&x, &x).unwrap(), var(&x).unwrap(), 1e-15);
        assert!(corr(&x, &[1.0; 5]).unwrap().is_nan());
        let m = [1.0, 4.0, 9.0, 16.0, 25.0]; // monotone, not linear
        assert!(corr(&x, &m).unwrap() < 1.0);
        near(spearman(&x, &m).unwrap(), 1.0, 1e-15);
        near(spearman(&[1.0, 2.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0]).unwrap(), 1.0, 1e-15);
    }

    #[test]
    fn fractions_from_decimals() {
        assert_eq!(as_fraction(0.5), Some((1, 2)));
        assert_eq!(as_fraction(-0.75), Some((-3, 4)));
        assert_eq!(as_fraction(1.0 / 3.0), Some((1, 3)));
        assert_eq!(as_fraction(7.0 / 3.0), Some((7, 3)));
        assert_eq!(as_fraction(-22.0 / 7.0), Some((-22, 7)));
        assert_eq!(as_fraction(0.1 + 0.2), Some((3, 10)), "float dust is fine");
        assert_eq!(as_fraction(355.0 / 113.0), Some((355, 113)));
        // not fractions: integers, irrationals, huge denominators, non-finite
        assert_eq!(as_fraction(3.0), None);
        assert_eq!(as_fraction(std::f64::consts::PI), None);
        assert_eq!(as_fraction(2f64.sqrt()), None);
        assert_eq!(as_fraction(f64::NAN), None);
        assert_eq!(as_fraction(f64::INFINITY), None);
        assert_eq!(as_fraction(1.0 / 1_000_003.0), None);
    }

    #[test]
    fn erf_references() {
        near(erf(0.5), 0.5204998778130465, 1e-15);
        near(erf(1.0), 0.8427007929497149, 1e-15);
        near(erf(2.0), 0.9953222650189527, 1e-15);
        near(erfc(1.0), 0.15729920705028513, 1e-15);
        near(erfc(2.0), 0.004677734981047266, 2e-15);
        near(erfc(5.0), 1.537459794428035e-12, 2e-14);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(-1.0), -erf(1.0));
        near(erfc(-1.0), 2.0 - erfc(1.0), 1e-15);
        assert_eq!(erfc(40.0), 0.0);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn erf_series_and_cf_agree_across_the_seam() {
        near(1.0 - erf_series(0.999_999_999), erfc_cf(0.999_999_999), 2e-14);
        near(erfc(1.5), 0.033894853524689274, 1e-14);
        near(erf(1.5), 0.9661051464753108, 1e-15);
        // erf + erfc = 1 everywhere
        for i in 0..60 {
            let x = i as f64 / 10.0;
            assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn ln_gamma_values() {
        near(ln_gamma(1.0).abs() + 1.0, 1.0, 1e-14);
        near(ln_gamma(5.0), 24.0f64.ln(), 1e-14);
        near(ln_gamma(0.5), PI.sqrt().ln(), 1e-14);
        near(ln_gamma(10.5), 13.940625219403763, 1e-14);
        near(ln_gamma(-0.5), (2.0 * PI.sqrt()).ln(), 1e-13);
        assert!(ln_gamma(0.0).is_infinite());
    }

    #[test]
    fn normal_references() {
        near(normalcdf(f64::NEG_INFINITY, 1.96, 0.0, 1.0), 0.9750021048517795, 1e-15);
        near(normalpdf(0.0, 0.0, 1.0), 0.3989422804014327, 1e-15);
        near(normalpdf(1.0, 0.0, 1.0), 0.24197072451914337, 1e-15);
        near(invnorm(0.975, 0.0, 1.0), 1.959963984540054, 1e-14);
        near(invnorm(0.5 + 0.3413447460685429, 0.0, 1.0), 1.0, 1e-13);
        near(normalcdf(-1.0, 1.0, 0.0, 1.0), 0.6826894921370859, 1e-15);
        near(normalcdf(-1.96, 1.96, 0.0, 1.0), 0.950004209703559, 1e-14);
        // location/scale
        near(normalcdf(10.0, 14.0, 12.0, 2.0), 0.6826894921370859, 1e-14);
        near(normalpdf(3.0, 1.0, 2.0), 0.24197072451914337 / 2.0, 1e-15);
        // far tails stay relatively accurate
        near(normalcdf(8.0, f64::INFINITY, 0.0, 1.0), 6.220960574271786e-16, 1e-12);
        near(invnorm(1e-10, 0.0, 1.0), -6.361340902404056, 1e-13);
        assert_eq!(normalcdf(f64::NEG_INFINITY, f64::INFINITY, 0.0, 1.0), 1.0);
        assert_eq!(normalcdf(1.0, -1.0, 0.0, 1.0), 0.0);
        assert_eq!(tcdf(1.0, -1.0, 5.0), 0.0);
    }

    #[test]
    fn normal_edge_cases() {
        assert!(normalpdf(0.0, 0.0, 0.0).is_nan());
        assert!(normalpdf(0.0, 0.0, -1.0).is_nan());
        assert!(normalcdf(0.0, 1.0, 0.0, 0.0).is_nan());
        assert!(invnorm(0.5, 0.0, 0.0).is_nan());
        assert_eq!(invnorm(0.0, 0.0, 1.0), f64::NEG_INFINITY);
        assert_eq!(invnorm(1.0, 0.0, 1.0), f64::INFINITY);
        assert!(invnorm(-0.1, 0.0, 1.0).is_nan());
        assert!(invnorm(1.1, 0.0, 1.0).is_nan());
        assert_eq!(invnorm(0.5, 3.0, 2.0), 3.0);
    }

    #[test]
    fn binomial_values() {
        near(binompdf(10.0, 0.5, 5.0), 252.0 / 1024.0, 1e-13);
        near(binomcdf(10.0, 0.5, 5.0), 638.0 / 1024.0, 1e-13);
        near(binompdf(5.0, 0.3, 2.0), 10.0 * 0.09 * 0.343, 1e-13);
        assert_eq!(binompdf(5.0, 0.3, 2.5), 0.0);
        assert_eq!(binompdf(5.0, 0.3, 6.0), 0.0);
        assert_eq!(binomcdf(5.0, 0.3, -1.0), 0.0);
        assert_eq!(binomcdf(5.0, 0.3, 5.0), 1.0);
        assert_eq!(binompdf(5.0, 0.0, 0.0), 1.0);
        assert_eq!(binompdf(5.0, 1.0, 5.0), 1.0);
        assert_eq!(binomcdf(5.0, 1.0, 4.0), 0.0);
        assert!(binompdf(5.0, 1.5, 2.0).is_nan());
        assert!(binompdf(5.5, 0.5, 2.0).is_nan());
        for &(n, p) in &[(20.0, 0.3), (100.0, 0.07), (50.0, 0.9)] {
            let s: f64 = (0..=n as i64).map(|k| binompdf(n, p, k as f64)).sum();
            near(s, 1.0, 1e-12);
            let mut acc = 0.0;
            for k in 0..=n as i64 {
                acc += binompdf(n, p, k as f64);
                near(binomcdf(n, p, k as f64), acc.min(1.0), 1e-10);
            }
        }
    }

    #[test]
    fn poisson_values() {
        let e2 = (-2.0f64).exp();
        near(poissonpdf(2.0, 3.0), e2 * 8.0 / 6.0, 1e-13);
        near(poissoncdf(2.0, 3.0), e2 * (1.0 + 2.0 + 2.0 + 4.0 / 3.0), 1e-13);
        assert_eq!(poissonpdf(0.0, 0.0), 1.0);
        assert_eq!(poissoncdf(0.0, 3.0), 1.0);
        assert_eq!(poissoncdf(3.0, -1.0), 0.0);
        assert!(poissonpdf(-1.0, 1.0).is_nan());
        for &l in &[0.5, 4.0, 30.0] {
            let mut acc = 0.0;
            for k in 0..200 {
                acc += poissonpdf(l, k as f64);
                if k < 150 {
                    near(poissoncdf(l, k as f64), acc.min(1.0), 1e-10);
                }
            }
            near(acc, 1.0, 1e-12);
        }
    }

    #[test]
    fn uniform_values() {
        assert_eq!(uniformpdf(0.5, 0.0, 2.0), 0.5);
        assert_eq!(uniformpdf(3.0, 0.0, 2.0), 0.0);
        assert!(uniformpdf(0.5, 2.0, 2.0).is_nan());
        assert_eq!(uniformcdf(0.0, 4.0, 1.0, 2.0), 0.25);
        assert_eq!(uniformcdf(0.0, 4.0, -5.0, 9.0), 1.0);
        assert_eq!(uniformcdf(0.0, 4.0, 5.0, 9.0), 0.0);
        assert_eq!(uniformcdf(0.0, 4.0, 3.0, 1.0), 0.0);
    }

    #[test]
    fn t_distribution() {
        // df = 1 is Cauchy
        for &x in &[-5.0, -1.0, 0.0, 0.3, 2.0, 40.0] {
            near(tpdf(x, 1.0), 1.0 / (PI * (1.0 + x * x)), 1e-13);
            near(t_cdf(x, 1.0), 0.5 + x.atan() / PI, 1e-13);
        }
        // df = 2 closed form
        for &x in &[-3.0, -0.5, 1.0, 4.0] {
            near(t_cdf(x, 2.0), 0.5 + x / (2.0 * (2.0 + x * x).sqrt()), 1e-13);
        }
        // symmetry and total mass
        near(tcdf(-2.0, 2.0, 7.0), 1.0 - 2.0 * t_cdf(-2.0, 7.0), 1e-13);
        near(tpdf(1.3, 5.0), tpdf(-1.3, 5.0), 1e-15);
        near(tcdf(f64::NEG_INFINITY, f64::INFINITY, 5.0), 1.0, 1e-14);
        near(t_cdf(0.0, 9.0), 0.5, 1e-15);
        // textbook critical values
        near(invt(0.975, 10.0), 2.2281388519862744, 1e-12);
        near(invt(0.975, 1.0), (PI * 0.475).tan(), 1e-12);
        near(invt(0.975, 1.0), 12.706204736174696, 1e-12);
        near(invt(0.025, 10.0), -2.2281388519862744, 1e-12);
        near(invt(0.95, 5.0), 2.0150483733330242, 1e-12);
        // df -> large approaches normal
        near(t_cdf(1.96, 1e7), 0.9750021048517795, 1e-6);
        near(tpdf(0.5, 1e7), normalpdf(0.5, 0.0, 1.0), 1e-6);
        // edges
        assert!(tpdf(0.0, 0.0).is_nan());
        assert!(tcdf(-1.0, 1.0, -2.0).is_nan());
        assert_eq!(invt(0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(invt(1.0, 5.0), f64::INFINITY);
        assert!(invt(1.5, 5.0).is_nan());
        assert_eq!(invt(0.5, 5.0), 0.0);
    }

    #[test]
    fn property_cdf_monotone_and_inverse_round_trips() {
        let mut prev = -1.0;
        for i in -80..=80 {
            let x = i as f64 / 10.0;
            let c = normalcdf(f64::NEG_INFINITY, x, 0.0, 1.0);
            assert!(c >= prev);
            prev = c;
            if c > 1e-14 && c < 1.0 - 1e-8 {
                let back = invnorm(c, 0.0, 1.0);
                assert!((back - x).abs() < 1e-9 * (1.0 + x.abs()), "{x} -> {back}");
            }
        }
        let mut prev = -1.0;
        for i in -100..=100 {
            let x = i as f64 / 4.0;
            let c = t_cdf(x, 3.5);
            assert!(c >= prev);
            prev = c;
            let back = invt(c, 3.5);
            if c > 1e-12 && c < 1.0 - 1e-12 {
                assert!((back - x).abs() < 1e-7 * (1.0 + x.abs()), "{x} -> {back}");
            }
        }
        let mut prev = -1.0;
        for k in 0..60 {
            let c = binomcdf(50.0, 0.4, k as f64);
            assert!(c >= prev);
            prev = c;
        }
    }

    #[test]
    fn property_aggregates() {
        // deterministic pseudo-random data
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 * 100.0 - 50.0
        };
        for len in [2usize, 3, 10, 57] {
            let x: Vec<f64> = (0..len).map(|_| next()).collect();
            let (mn, mx) = (min(&x).unwrap(), max(&x).unwrap());
            for i in 0..=20 {
                let q = quantile(&x, i as f64 / 20.0).unwrap();
                assert!(q >= mn && q <= mx);
            }
            // var = E[x^2] - mean^2 (population) and sample/pop relation
            let m = mean(&x).unwrap();
            let ex2 = x.iter().map(|v| v * v).sum::<f64>() / len as f64;
            assert!((varp(&x).unwrap() - (ex2 - m * m)).abs() < 1e-9 * (1.0 + ex2));
            near(var(&x).unwrap(), varp(&x).unwrap() * len as f64 / (len - 1) as f64, 1e-12);
            // scaling and shifting
            let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 7.0).collect();
            near(stdev(&y).unwrap(), 3.0 * stdev(&x).unwrap(), 1e-12);
            near(var(&y).unwrap(), 9.0 * var(&x).unwrap(), 1e-12);
            near(mean(&y).unwrap(), 3.0 * m + 7.0, 1e-12);
            assert!((corr(&x, &y).unwrap() - 1.0).abs() < 1e-12);
            assert!(mad(&x).unwrap() <= stdevp(&x).unwrap() + 1e-12);
            let sorted = sort(&x);
            assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
            assert_eq!(sorted.len(), len);
        }
    }
}
