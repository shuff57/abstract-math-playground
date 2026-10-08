//! Adapts the hit-test curves ([`ExplicitCurve`], [`ShapeCurve`]) to `math_core::intersect` and
//! finds where one curve meets the others. Everything here is in world coordinates: shape curves
//! hold display-space programs, which are wrapped with the axis map (logarithmic axes).

use crate::axis_map::AxisMap;
use crate::scene::{ExplicitCurve, ShapeCurve, ShapeKind};
use math_core::intersect::{intersections, Curve, Shape, Window};

/// Other curves checked against the selected one, at most.
pub const MAX_PARTNERS: usize = 8;

/// A curve as owned closures, so a [`Curve`] can borrow them.
enum Owned<'a> {
    Explicit(Box<dyn Fn(f64) -> f64 + 'a>),
    Implicit(Box<dyn Fn(f64, f64) -> f64 + 'a>),
    Param { x: Box<dyn Fn(f64) -> f64 + 'a>, y: Box<dyn Fn(f64) -> f64 + 'a>, t1: f64 },
}

impl<'a> Owned<'a> {
    fn curve(&self) -> Curve<'_> {
        Curve::new(match self {
            Owned::Explicit(f) => Shape::Explicit(f.as_ref()),
            Owned::Implicit(f) => Shape::Implicit(f.as_ref()),
            Owned::Param { x, y, t1 } => Shape::Param { x: x.as_ref(), y: y.as_ref(), t0: 0.0, t1: *t1 },
        })
    }
}

fn explicit<'a>(c: &'a ExplicitCurve) -> Owned<'a> {
    Owned::Explicit(Box::new(move |x| c.prog.eval(&[x])))
}

fn shape<'a>(c: &'a ShapeCurve, m: AxisMap) -> Owned<'a> {
    match &c.kind {
        ShapeKind::Implicit(p) => Owned::Implicit(Box::new(move |x, y| p.eval(&[m.fwd(0, x), m.fwd(1, y)]))),
        ShapeKind::Parametric { px, py, t_end } => Owned::Param {
            x: Box::new(move |t| m.inv(0, px.eval(&[t]))),
            y: Box::new(move |t| m.inv(1, py.eval(&[t]))),
            t1: *t_end,
        },
    }
}

/// Intersections of curve `id` with every other curve in `partners` order (the caller sorts
/// them; at most [`MAX_PARTNERS`] are used): `(partner id, x, y)` in world coordinates, sorted by
/// x. `win` is `[xmin, xmax, ymin, ymax]` in world coordinates.
pub fn with_others(
    id: &str,
    explicit_curves: &[ExplicitCurve],
    shapes: &[&ShapeCurve],
    order: &dyn Fn(&str) -> usize,
    m: AxisMap,
    win: [f64; 4],
) -> Vec<(String, f64, f64)> {
    let mut all: Vec<(&str, Owned)> = explicit_curves.iter().map(|c| (c.id.as_str(), explicit(c))).collect();
    all.extend(shapes.iter().map(|c| (c.id.as_str(), shape(c, m))));
    let Some(me) = all.iter().position(|(i, _)| *i == id) else { return Vec::new() };
    let mut others: Vec<usize> = (0..all.len()).filter(|&i| i != me).collect();
    others.sort_by_key(|&i| order(all[i].0));
    others.truncate(MAX_PARTNERS);
    let w = Window { x0: win[0], x1: win[1], y0: win[2], y1: win[3] };
    let a = all[me].1.curve();
    let mut out = Vec::new();
    for i in others {
        for (x, y) in intersections(&a, &all[i].1.curve(), w) {
            out.push((all[i].0.to_string(), x, y));
        }
    }
    out.sort_by(|p, q| p.1.total_cmp(&q.1).then(p.2.total_cmp(&q.2)));
    out
}
