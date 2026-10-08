//! Adapts the hit-test curves ([`ExplicitCurve`], [`ShapeCurve`]) to `math_core::intersect` and
//! finds where one curve meets the others. Everything here is in world coordinates: shape curves
//! hold display-space programs, which are wrapped with the axis map (logarithmic axes).

use crate::axis_map::AxisMap;
use crate::scene::{ExplicitCurve, ShapeCurve, ShapeKind};
use math_core::intersect::{intersections, Curve, Shape, Window};
use math_core::restrict::Restrict;

/// Other curves checked against the selected one, at most.
pub const MAX_PARTNERS: usize = 8;

/// The geometry of a curve as owned closures, so a [`Curve`] can borrow them.
enum Geom<'a> {
    Explicit(Box<dyn Fn(f64) -> f64 + 'a>),
    Implicit(Box<dyn Fn(f64, f64) -> f64 + 'a>),
    Param { x: Box<dyn Fn(f64) -> f64 + 'a>, y: Box<dyn Fn(f64) -> f64 + 'a>, t1: f64 },
}

/// A curve and the `{x>0}` restriction it was drawn with, if any: an intersection on the part
/// that is clipped away is not an intersection (it is not drawn, so it must not be listed).
struct Owned<'a> {
    geom: Geom<'a>,
    clip: Option<Box<dyn Fn(f64, f64) -> bool + 'a>>,
}

impl<'a> Owned<'a> {
    fn curve(&self) -> Curve<'_> {
        let mut c = Curve::new(match &self.geom {
            Geom::Explicit(f) => Shape::Explicit(f.as_ref()),
            Geom::Implicit(f) => Shape::Implicit(f.as_ref()),
            Geom::Param { x, y, t1 } => Shape::Param { x: x.as_ref(), y: y.as_ref(), t0: 0.0, t1: *t1 },
        });
        c.clip = self.clip.as_deref();
        c
    }
}

fn clip_of(r: &Option<Restrict>) -> Option<Box<dyn Fn(f64, f64) -> bool + '_>> {
    r.as_ref().filter(|r| !r.is_unbounded()).map(|r| Box::new(move |x, y| r.contains(x, y)) as Box<dyn Fn(f64, f64) -> bool>)
}

fn explicit<'a>(c: &'a ExplicitCurve) -> Owned<'a> {
    Owned { geom: Geom::Explicit(Box::new(move |x| c.prog.eval(&[x]))), clip: clip_of(&c.restrict) }
}

fn shape<'a>(c: &'a ShapeCurve, m: AxisMap) -> Owned<'a> {
    let geom = match &c.kind {
        ShapeKind::Implicit(p) => Geom::Implicit(Box::new(move |x, y| p.eval(&[m.fwd(0, x), m.fwd(1, y)]))),
        ShapeKind::Parametric { px, py, t_end } => Geom::Param {
            x: Box::new(move |t| m.inv(0, px.eval(&[t]))),
            y: Box::new(move |t| m.inv(1, py.eval(&[t]))),
            t1: *t_end,
        },
    };
    Owned { geom, clip: clip_of(&c.restrict) }
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
