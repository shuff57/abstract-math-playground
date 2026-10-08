//! Domain restriction of explicit and implicit 2D items: `y=x^2 {x>0}`, `y=x^2 {-1<x<2}`,
//! `x=y^2 {y>=0}`, `x^2+y^2=4 {y>0}`, `y<x {x>0}`.
//!
//! The trailing `{...}` group is parsed exactly like a parametric range (see [`crate::param`]:
//! `domain(body, clause, ...)`); here its clauses on `x` and `y` become a [`Restrict`], an
//! axis-aligned box with optional sides. Curves are clipped to the box ([`Restrict::clip_polylines`],
//! [`Restrict::clip_segments`]) and the points where a curve ends on a side come back as
//! [`Marker`]s: a strict bound (`<`, `>`) is an OPEN endpoint, a non-strict one (`<=`, `>=`)
//! a FILLED one, as in Desmos. The renderer only has to draw points.
//!
//! The box is in world coordinates. Inequality fills use [`margin_expr`], which folds the box
//! into the item's margin function so the GPU field needs no special case.

use crate::ast::{BinOp, Expr};
use crate::compile::Program;
use crate::param::Range;

/// One side of the box: its value and whether the boundary itself is excluded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    pub value: f64,
    pub strict: bool,
}

/// A curve endpoint produced by a restriction. `open` is true for a strict bound (`<`, `>`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marker {
    pub pos: [f64; 2],
    pub open: bool,
}

/// A box in the x/y plane with optional sides: `lo[0]`/`hi[0]` bound x, `lo[1]`/`hi[1]` bound y.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Restrict {
    pub lo: [Option<Bound>; 2],
    pub hi: [Option<Bound>; 2],
}

/// A side of the box, for markers: the axis (0 = x, 1 = y) and whether it is the upper side.
#[derive(Clone, Copy)]
struct Edge {
    axis: usize,
    upper: bool,
}

struct Clip {
    t0: f64,
    t1: f64,
    enter: Option<Edge>,
    exit: Option<Edge>,
}

const MAX_MARKERS: usize = 64;

impl Restrict {
    /// The box a list of ranges describes. `eval` turns a bound expression into a number
    /// (sliders and definitions resolved by the caller). Errors: a range on a name other than
    /// `x`/`y`, a bound that is not a finite number, and an empty box.
    pub fn from_ranges(ranges: &[Range], eval: &dyn Fn(&Expr) -> Result<f64, String>) -> Result<Restrict, String> {
        let mut out = Restrict::default();
        for r in ranges {
            let a = match r.var.as_str() {
                "x" => 0,
                "y" => 1,
                other => {
                    return Err(format!(
                        "a range on '{other}' does not apply here (an explicit or implicit item is restricted by x or y)"
                    ))
                }
            };
            let get = |e: &Expr, strict: bool| -> Result<Bound, String> {
                let value = eval(e)?;
                if !value.is_finite() {
                    return Err(format!("the range of {} is not a finite number", r.var));
                }
                Ok(Bound { value, strict })
            };
            if let Some(e) = &r.lo {
                out.lo[a] = Some(get(e, r.lo_strict)?);
            }
            if let Some(e) = &r.hi {
                out.hi[a] = Some(get(e, r.hi_strict)?);
            }
            if let (Some(l), Some(h)) = (out.lo[a], out.hi[a]) {
                if l.value > h.value || (l.value == h.value && (l.strict || h.strict)) {
                    return Err(format!("empty range: {} is not below {}", l.value, h.value));
                }
            }
        }
        Ok(out)
    }

    /// True when no side is set.
    pub fn is_unbounded(&self) -> bool {
        self.lo.iter().chain(self.hi.iter()).all(Option::is_none)
    }

    fn axis_ok(&self, a: usize, v: f64) -> bool {
        if let Some(b) = self.lo[a] {
            if v < b.value || (b.strict && v == b.value) {
                return false;
            }
        }
        if let Some(b) = self.hi[a] {
            if v > b.value || (b.strict && v == b.value) {
                return false;
            }
        }
        true
    }

    /// Whether `(x, y)` satisfies every bound (strict bounds exclude their boundary).
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.axis_ok(0, x) && self.axis_ok(1, y)
    }

    /// `[x0, x1]` narrowed to the x bounds; `None` when nothing is left (or only one point).
    pub fn x_interval(&self, x0: f64, x1: f64) -> Option<(f64, f64)> {
        let a = self.lo[0].map_or(x0, |b| b.value.max(x0));
        let b = self.hi[0].map_or(x1, |b| b.value.min(x1));
        (b > a).then_some((a, b))
    }

    /// Liang-Barsky against the closed box, remembering which sides cut the segment.
    fn clip_seg(&self, p: [f64; 2], q: [f64; 2]) -> Option<Clip> {
        let d = [q[0] - p[0], q[1] - p[1]];
        let mut c = Clip { t0: 0.0, t1: 1.0, enter: None, exit: None };
        for a in 0..2 {
            for upper in [false, true] {
                let bound = if upper { self.hi[a] } else { self.lo[a] };
                let Some(b) = bound else { continue };
                // inside means s0 + t * s1 >= 0
                let (s0, s1) = if upper { (b.value - p[a], -d[a]) } else { (p[a] - b.value, d[a]) };
                if s1 == 0.0 {
                    if s0 < 0.0 {
                        return None;
                    }
                    continue;
                }
                let t = -s0 / s1;
                if s1 > 0.0 {
                    if t > c.t0 {
                        c.t0 = t;
                        c.enter = Some(Edge { axis: a, upper });
                    }
                } else if t < c.t1 {
                    c.t1 = t;
                    c.exit = Some(Edge { axis: a, upper });
                }
            }
        }
        (c.t0 <= c.t1).then_some(c)
    }

    fn marker_at(&self, e: Edge, pos: [f64; 2]) -> Marker {
        let b = if e.upper { self.hi[e.axis] } else { self.lo[e.axis] };
        Marker { pos, open: b.is_some_and(|b| b.strict) }
    }

    /// Clips polylines to the box. A curve that leaves or enters through a side gets a marker
    /// there (open for a strict bound).
    pub fn clip_polylines(&self, lines: &[Vec<[f64; 2]>]) -> (Vec<Vec<[f64; 2]>>, Vec<Marker>) {
        if self.is_unbounded() {
            return (lines.to_vec(), Vec::new());
        }
        let mut out = Vec::new();
        let mut marks = Vec::new();
        let flush = |cur: &mut Vec<[f64; 2]>, out: &mut Vec<Vec<[f64; 2]>>| {
            if cur.len() >= 2 {
                out.push(std::mem::take(cur));
            } else {
                cur.clear();
            }
        };
        for line in lines {
            let mut cur: Vec<[f64; 2]> = Vec::new();
            for w in line.windows(2) {
                let Some(c) = self.clip_seg(w[0], w[1]) else {
                    flush(&mut cur, &mut out);
                    continue;
                };
                let at = |t: f64| [w[0][0] + t * (w[1][0] - w[0][0]), w[0][1] + t * (w[1][1] - w[0][1])];
                let a = if c.t0 == 0.0 { w[0] } else { at(c.t0) };
                let b = if c.t1 == 1.0 { w[1] } else { at(c.t1) };
                if let Some(e) = c.enter {
                    flush(&mut cur, &mut out);
                    push_marker(&mut marks, self.marker_at(e, a));
                }
                if cur.is_empty() {
                    cur.push(a);
                }
                cur.push(b);
                if let Some(e) = c.exit {
                    push_marker(&mut marks, self.marker_at(e, b));
                    flush(&mut cur, &mut out);
                }
            }
            flush(&mut cur, &mut out);
        }
        (out, marks)
    }

    /// Clips loose segments (a contour) to the box, with the same markers.
    pub fn clip_segments(&self, segs: &[[[f64; 2]; 2]]) -> (Vec<[[f64; 2]; 2]>, Vec<Marker>) {
        if self.is_unbounded() {
            return (segs.to_vec(), Vec::new());
        }
        let mut out = Vec::new();
        let mut marks = Vec::new();
        for s in segs {
            let Some(c) = self.clip_seg(s[0], s[1]) else { continue };
            let at = |t: f64| [s[0][0] + t * (s[1][0] - s[0][0]), s[0][1] + t * (s[1][1] - s[0][1])];
            let a = if c.t0 == 0.0 { s[0] } else { at(c.t0) };
            let b = if c.t1 == 1.0 { s[1] } else { at(c.t1) };
            if let Some(e) = c.enter {
                push_marker(&mut marks, self.marker_at(e, a));
            }
            if let Some(e) = c.exit {
                push_marker(&mut marks, self.marker_at(e, b));
            }
            if a != b {
                out.push([a, b]);
            }
        }
        (out, marks)
    }

    /// Markers at the x bounds of an explicit curve `y = f(x)` over `[x0, x1]` (a bound outside
    /// the window has none), skipping a bound where `f` is undefined or the y bounds exclude it.
    pub fn explicit_markers(&self, f: &Program, x0: f64, x1: f64) -> Vec<Marker> {
        let mut out = Vec::new();
        for b in [self.lo[0], self.hi[0]].into_iter().flatten() {
            if b.value < x0 || b.value > x1 {
                continue;
            }
            let y = f.eval(&[b.value]);
            if y.is_finite() && self.axis_ok(1, y) {
                out.push(Marker { pos: [b.value, y], open: b.strict });
            }
        }
        out
    }

    /// Samples `y = f(x)` over `[x0, x1]` restricted to the box: the x interval is narrowed
    /// first (so the curve ends exactly on the bound), then clipped in y. Returns the polylines
    /// and the endpoint markers.
    pub fn sample_explicit(
        &self,
        f: &Program,
        x0: f64,
        x1: f64,
        width_px: usize,
        y_range: (f64, f64),
    ) -> (Vec<Vec<[f64; 2]>>, Vec<Marker>) {
        let Some((a, b)) = self.x_interval(x0, x1) else {
            return (Vec::new(), self.explicit_markers(f, x0, x1));
        };
        let share = ((b - a) / (x1 - x0)).clamp(0.0, 1.0);
        let px = ((width_px as f64 * share).ceil() as usize).max(16);
        let lines = crate::mesh::sample_explicit(f, a, b, px, y_range);
        let (lines, mut marks) = self.clip_polylines(&lines);
        for m in self.explicit_markers(f, x0, x1) {
            push_marker(&mut marks, m);
        }
        (lines, marks)
    }
}

/// A number with at most six decimals: the boundary a user typed (`-1`, `0.5`), as opposed to
/// the double next to it that bisection also lands on.
fn typed_number(c: f64) -> bool {
    c.is_finite() && c == (c * 1e6).round() / 1e6
}

/// Breakpoint markers of a piecewise explicit curve `y = f(x)` over `[x0, x1]` (call it only for
/// expressions that contain a piecewise node). Where the curve goes from defined to undefined,
/// or jumps between two finite values, the edge is found by bisection and marked at the
/// one-sided limit: OPEN when the function value at the edge is not that limit (a strict
/// condition), FILLED when it is (a non-strict one). Edges with an infinite limit or outside the
/// window get none. `samples` is the number of probe intervals.
pub fn piecewise_markers(f: &Program, x0: f64, x1: f64, y_range: (f64, f64), samples: usize) -> Vec<Marker> {
    let mut out = Vec::new();
    if !(x0.is_finite() && x1.is_finite() && x1 > x0) {
        return out;
    }
    let (ylo, yhi) = (y_range.0.min(y_range.1), y_range.0.max(y_range.1));
    let n = samples.clamp(16, 8192);
    let dx = (x1 - x0) / n as f64;
    let span = (yhi - ylo).max(1e-9);
    let ev = |x: f64| f.eval(&[x]);
    let mut xa = x0;
    let mut ya = ev(xa);
    let put = |out: &mut Vec<Marker>, x: f64, y: f64, open: bool| {
        if y.is_finite() && x >= x0 && x <= x1 && y >= ylo && y <= yhi {
            push_marker(out, Marker { pos: [x, y], open });
        }
    };
    for i in 1..=n {
        let xb = if i == n { x1 } else { x0 + dx * i as f64 };
        let yb = ev(xb);
        match (ya.is_finite(), yb.is_finite()) {
            (true, false) | (false, true) => {
                // d: the defined end, u: the undefined end
                let (mut d, mut u) = if ya.is_finite() { (xa, xb) } else { (xb, xa) };
                for _ in 0..1200 {
                    let m = 0.5 * (d + u);
                    if m == d || m == u {
                        break;
                    }
                    if ev(m).is_finite() {
                        d = m;
                    } else {
                        u = m;
                    }
                }
                let b = if typed_number(u) { u } else if typed_number(d) { d } else { u };
                put(&mut out, b, ev(d), b != d);
            }
            (true, true) if (yb - ya).abs() > 0.02 * span => {
                let (mut lo, mut hi, mut ylo_, mut yhi_) = (xa, xb, ya, yb);
                for _ in 0..1200 {
                    let m = 0.5 * (lo + hi);
                    if m == lo || m == hi {
                        break;
                    }
                    let ym = ev(m);
                    if !ym.is_finite() {
                        break;
                    }
                    if (ym - ylo_).abs() >= (yhi_ - ym).abs() {
                        hi = m;
                        yhi_ = ym;
                    } else {
                        lo = m;
                        ylo_ = ym;
                    }
                }
                let big = ylo_.abs().max(yhi_.abs());
                if (yhi_ - ylo_).abs() > 1e-6 * (1.0 + big) && big < 1e6 {
                    // the typed number is where the edge belongs: its value is the filled dot
                    if typed_number(lo) && !typed_number(hi) {
                        put(&mut out, lo, ylo_, false);
                        put(&mut out, lo, yhi_, true);
                    } else {
                        put(&mut out, hi, yhi_, false);
                        put(&mut out, hi, ylo_, true);
                    }
                }
            }
            _ => {}
        }
        xa = xb;
        ya = yb;
    }
    out
}

fn push_marker(marks: &mut Vec<Marker>, m: Marker) {
    if marks.len() >= MAX_MARKERS || !m.pos.iter().all(|v| v.is_finite()) {
        return;
    }
    let near = |a: &Marker| {
        let tol = 1e-9 * (1.0 + a.pos[0].abs().max(a.pos[1].abs()));
        (a.pos[0] - m.pos[0]).abs() <= tol && (a.pos[1] - m.pos[1]).abs() <= tol
    };
    if !marks.iter().any(near) {
        marks.push(m);
    }
}

/// An expression that is `<= 0` exactly inside the box of `ranges` (`lo - x`, `x - hi`, folded
/// with `max`), or `None` when the ranges bound neither x nor y. Bounds stay symbolic so the GPU
/// field keeps sliders as uniform parameters. Ranges on other names are ignored.
pub fn margin_expr(ranges: &[Range]) -> Option<Expr> {
    let mut parts: Vec<Expr> = Vec::new();
    for r in ranges.iter().filter(|r| r.var == "x" || r.var == "y") {
        if let Some(lo) = &r.lo {
            parts.push(Expr::bin(BinOp::Sub, lo.clone(), Expr::var(&r.var)));
        }
        if let Some(hi) = &r.hi {
            parts.push(Expr::bin(BinOp::Sub, Expr::var(&r.var), hi.clone()));
        }
    }
    parts.into_iter().reduce(|a, b| Expr::call("max", vec![a, b]))
}

/// The margin of an inequality region `f < 0` (or `f > 0` when `greater`) cut to the box.
/// `m` is [`margin_expr`]: the region is where `f` and the box both hold.
pub fn restrict_margin(f: Expr, greater: bool, m: Expr) -> Expr {
    if greater {
        Expr::call("min", vec![f, Expr::neg(m)])
    } else {
        Expr::call("max", vec![f, m])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{analyze, Kind};
    use crate::compile::{compile, Angle};
    use crate::parse::parse;
    use std::collections::BTreeSet;

    fn ev(e: &Expr) -> Result<f64, String> {
        compile(e, &[], Angle::Rad).map(|p| p.eval(&[])).map_err(|e| e.to_string())
    }

    fn restrict(src: &str) -> Restrict {
        let a = analyze(&parse(src).unwrap(), &BTreeSet::new());
        Restrict::from_ranges(&a.domain, &ev).unwrap()
    }

    #[test]
    fn explicit_and_implicit_items_keep_their_kind_and_carry_ranges() {
        for (src, n) in [("y=x^2 {x>0}", 1), ("y=x^2 {-1<x<2}", 1), ("x=y^2 {y>=0}", 1), ("x^2+y^2=4 {y>0}", 1), ("y<x {x>0, y<3}", 2)] {
            let a = analyze(&parse(src).unwrap(), &BTreeSet::new());
            assert_eq!(a.domain.len(), n, "{src}");
            assert!(!matches!(a.kind, Kind::Parametric { .. }), "{src}");
        }
        assert!(matches!(analyze(&parse("y=x^2 {x>0}").unwrap(), &BTreeSet::new()).kind, Kind::ExplicitY { .. }));
        assert!(matches!(analyze(&parse("x=y^2 {y>=0}").unwrap(), &BTreeSet::new()).kind, Kind::ExplicitX { .. }));
        assert!(matches!(analyze(&parse("x^2+y^2=4 {y>0}").unwrap(), &BTreeSet::new()).kind, Kind::Implicit { .. }));
        assert!(matches!(analyze(&parse("y<x {x>0}").unwrap(), &BTreeSet::new()).kind, Kind::Inequality { .. }));
    }

    #[test]
    fn bounds_remember_strictness() {
        let r = restrict("y=x^2 {-1<x<=2}");
        assert_eq!(r.lo[0], Some(Bound { value: -1.0, strict: true }));
        assert_eq!(r.hi[0], Some(Bound { value: 2.0, strict: false }));
        let r = restrict("y=x {3>=x>1}");
        assert_eq!(r.hi[0], Some(Bound { value: 3.0, strict: false }));
        assert_eq!(r.lo[0], Some(Bound { value: 1.0, strict: true }));
        // two clauses on one variable merge
        let r = restrict("y=x {x>0, x<5}");
        assert!(r.lo[0].is_some() && r.hi[0].is_some());
        assert!(r.contains(1.0, 0.0) && !r.contains(0.0, 0.0) && !r.contains(5.0, 0.0));
    }

    #[test]
    fn rejects_other_names_and_empty_boxes() {
        let a = analyze(&parse("y=x {t>0}").unwrap(), &BTreeSet::new());
        assert!(Restrict::from_ranges(&a.domain, &ev).unwrap_err().contains("'t'"));
        let a = analyze(&parse("y=x {x>2, x<1}").unwrap(), &BTreeSet::new());
        assert!(Restrict::from_ranges(&a.domain, &ev).unwrap_err().contains("empty"));
        let a = analyze(&parse("y=x {2<x<2}").unwrap(), &BTreeSet::new());
        assert!(Restrict::from_ranges(&a.domain, &ev).is_err());
        let a = analyze(&parse("y=x {x>b}").unwrap(), &BTreeSet::new());
        assert!(Restrict::from_ranges(&a.domain, &ev).is_err()); // b is not defined
    }

    fn square() -> Program {
        compile(&parse("x^2").unwrap(), &["x"], Angle::Rad).unwrap()
    }

    #[test]
    fn explicit_samples_stay_inside_the_range() {
        let r = restrict("y=x^2 {x>0}");
        let (lines, marks) = r.sample_explicit(&square(), -5.0, 5.0, 400, (-5.0, 25.0));
        assert!(!lines.is_empty());
        for l in &lines {
            for p in l {
                assert!(p[0] >= 0.0 && p[0] <= 5.0, "{p:?}");
            }
        }
        assert_eq!(marks, vec![Marker { pos: [0.0, 0.0], open: true }]);

        let r = restrict("y=x^2 {-1<=x<2}");
        let (lines, marks) = r.sample_explicit(&square(), -5.0, 5.0, 400, (-5.0, 25.0));
        let xs: Vec<f64> = lines.iter().flatten().map(|p| p[0]).collect();
        assert!(xs.iter().all(|x| (-1.0..=2.0).contains(x)));
        assert!(xs.iter().any(|x| (*x + 1.0).abs() < 1e-9) && xs.iter().any(|x| (*x - 2.0).abs() < 1e-9));
        assert_eq!(marks.len(), 2);
        assert!(marks.contains(&Marker { pos: [-1.0, 1.0], open: false }));
        assert!(marks.contains(&Marker { pos: [2.0, 4.0], open: true }));
    }

    #[test]
    fn explicit_y_bound_clips_and_marks() {
        let r = restrict("y=x^2 {y<4}");
        let (lines, marks) = r.sample_explicit(&square(), -5.0, 5.0, 400, (-5.0, 25.0));
        assert_eq!(lines.len(), 1);
        for p in lines.iter().flatten() {
            assert!(p[1] <= 4.0 + 1e-12);
        }
        assert_eq!(marks.len(), 2);
        assert!(marks.iter().all(|m| m.open && (m.pos[1] - 4.0).abs() < 1e-12));
    }

    #[test]
    fn window_outside_the_range_draws_nothing() {
        let r = restrict("y=x^2 {x>10}");
        let (lines, marks) = r.sample_explicit(&square(), -5.0, 5.0, 400, (-5.0, 25.0));
        assert!(lines.is_empty() && marks.is_empty());
    }

    #[test]
    fn contour_segments_are_clipped_with_markers() {
        let r = restrict("x^2+y^2=4 {y>=0}");
        let p = compile(&parse("x^2+y^2-4").unwrap(), &["x", "y"], Angle::Rad).unwrap();
        let segs = crate::mesh::contour_2d(&p, (-3.0, 3.0), (-3.0, 3.0), 0.01, 10, 400_000);
        assert!(segs.iter().any(|s| s[0][1] < 0.0 || s[1][1] < 0.0), "the full circle has a lower half");
        let (kept, marks) = r.clip_segments(&segs);
        assert!(!kept.is_empty());
        for s in &kept {
            assert!(s[0][1] >= 0.0 && s[1][1] >= 0.0, "{s:?}");
        }
        assert_eq!(marks.len(), 2, "{marks:?}");
        assert!(marks.iter().all(|m| !m.open && m.pos[1].abs() < 1e-9 && (m.pos[0].abs() - 2.0).abs() < 0.02));
        // strict: open
        let (_, marks) = restrict("x^2+y^2=4 {y>0}").clip_segments(&segs);
        assert!(marks.iter().all(|m| m.open));
    }

    #[test]
    fn x_equals_g_of_y_restricted_by_y() {
        // x = y^2, y >= 0: the upper branch only; its end at the origin is filled
        let r = restrict("x=y^2 {y>=0}");
        let p = compile(&parse("x-y^2").unwrap(), &["x", "y"], Angle::Rad).unwrap();
        let segs = crate::mesh::contour_2d(&p, (-1.0, 4.0), (-3.0, 3.0), 0.01, 10, 400_000);
        let (kept, marks) = r.clip_segments(&segs);
        assert!(!kept.is_empty() && kept.iter().all(|s| s[0][1] >= 0.0 && s[1][1] >= 0.0));
        assert_eq!(marks.len(), 1);
        assert!(!marks[0].open && marks[0].pos[0].abs() < 0.02);
    }

    #[test]
    fn polyline_clip_splits_and_joins() {
        let r = restrict("y=x {-1<x<1}");
        // a zig-zag that leaves and returns through the right side
        let line = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let (out, marks) = r.clip_polylines(&[line]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![[0.0, 0.0], [1.0, 0.0]]);
        assert_eq!(out[1], vec![[1.0, 1.0], [0.0, 1.0]]);
        assert_eq!(marks.len(), 2);
        // an unbounded box changes nothing
        let u = Restrict::default();
        let l = vec![vec![[0.0, 0.0], [9.0, 9.0]]];
        assert_eq!(u.clip_polylines(&l).0, l);
    }

    #[test]
    fn margin_cuts_inequality_fills() {
        let a = analyze(&parse("y<x {x>0}").unwrap(), &BTreeSet::new());
        let Kind::Inequality { f, .. } = &a.kind else { panic!() };
        let m = margin_expr(&a.domain).unwrap();
        let g = compile(&restrict_margin(f.clone(), false, m.clone()), &["x", "y"], Angle::Rad).unwrap();
        assert!(g.eval(&[1.0, 0.0]) < 0.0, "inside both");
        assert!(g.eval(&[-1.0, -5.0]) > 0.0, "y<x holds but x<=0");
        assert!(g.eval(&[1.0, 5.0]) > 0.0, "x>0 but y>=x");
        let gt = compile(&restrict_margin(f.clone(), true, m), &["x", "y"], Angle::Rad).unwrap();
        // `>` regions are where the margin is positive: outside the box it must not be
        assert!(gt.eval(&[1.0, 5.0]) > 0.0 && gt.eval(&[-1.0, 5.0]) <= 0.0);
        assert!(margin_expr(&[]).is_none());
    }

    fn pw(src: &str) -> Vec<Marker> {
        let p = compile(&parse(src).unwrap(), &["x"], Angle::Rad).unwrap();
        piecewise_markers(&p, -10.0, 10.0, (-8.0, 8.0), 800)
    }

    #[test]
    fn piecewise_gap_edges_get_open_or_filled_markers() {
        let m = pw("{x<-1:-x, x>1:x}");
        assert_eq!(m.len(), 2, "{m:?}");
        assert!(m.iter().all(|k| k.open));
        assert!(m.iter().any(|k| (k.pos[0] + 1.0).abs() < 1e-9 && (k.pos[1] - 1.0).abs() < 1e-6));
        assert!(m.iter().any(|k| (k.pos[0] - 1.0).abs() < 1e-9 && (k.pos[1] - 1.0).abs() < 1e-6));
        let m = pw("{x<=-1:-x, x>=1:x}");
        assert_eq!(m.len(), 2, "{m:?}");
        assert!(m.iter().all(|k| !k.open));
    }

    #[test]
    fn piecewise_jump_has_an_open_and_a_filled_marker() {
        let m = pw("{x<0:x^2, x^2+1}");
        assert_eq!(m.len(), 2, "{m:?}");
        assert!(m.iter().any(|k| k.open && k.pos[0].abs() < 1e-9 && k.pos[1].abs() < 1e-6));
        assert!(m.iter().any(|k| !k.open && k.pos[0].abs() < 1e-9 && (k.pos[1] - 1.0).abs() < 1e-9));
        // continuous join: nothing
        assert!(pw("{x<0:-x, x}").is_empty());
    }
}
