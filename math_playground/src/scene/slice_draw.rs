//! Slice drawing: the overlay in the main scene (plane quad / cut line, exact intersection
//! curves, root dots) and the secondary inset panel (the same slice as a flat 2D graph).
//!
//! Curves are exact, not mesh-clipped: every item is turned into a scalar `F` (`y = f(x)` becomes
//! `y - f(x)`, an implicit item is its own `lhs - rhs`), the fixed axes are substituted by their
//! constants, and the remaining function of the free axes is contoured with
//! `mesh::contour_2d` (2D slice) or root-found (1D slice).

use super::*;
use math_core::slice::{axis_name, ResolvedSlice};
use math_core::view::Rig;
use math_core::Interval;

/// Plane quad, cut line and outline colour.
pub const SLICE_COLOR: [f32; 4] = [0.95, 0.58, 0.08, 1.0];
const PLANE_ALPHA: f32 = 0.18;
/// Extra room around the slice curve when the inset auto-fits it.
const INSET_FIT_PAD: f64 = 0.2;
/// Weight of the cut (plane / surface intersection) in the main scene, before its halo.
const CUT_MIN_W: f32 = 3.0;
const CUT_MAX_W: f32 = 4.0;
/// The cut: opaque dark navy, which separates from a surface of any hue and from the plane's orange.
const CUT_COLOR: [f32; 4] = [0.07, 0.09, 0.22, 1.0];
const SLICE_CURVE_W: f32 = 4.5;
const SLICE_DOT_W: f32 = 13.0;
const INSET_CURVE_W: f32 = 2.5;
/// Samples used to estimate the value range of a 1D graph.
/// NDC depth the slice curves are pulled towards the camera.
const DEPTH_BIAS: f32 = 1.0e-3;
/// Stronger pull for the cut so it is not broken up where it grazes the surface.
const CUT_BIAS: f32 = 6.0e-3;
const RANGE_SAMPLES: usize = 600;
/// Highlight of the intervals of a line slice where an inequality holds.
const INTERVAL_W: f32 = 9.0;
const INTERVAL_ALPHA: f32 = 0.55;

/// A 2D raster of an item over the slice plane (hue field, inequality region, domain colouring).
pub struct SliceField {
    pub kind: FieldKind,
    /// The item's expression with the fixed axes replaced by `$slice_*` stand-ins and the free
    /// axes renamed `x`, `y` (see `math_core::slice::virtualize`); unresolved, so sliders stay
    /// uniforms. For a complex item this is the complex expression itself.
    pub raw: Expr,
    /// Stand-in variable names and their values (the resolved slice constants).
    pub consts: Vec<(String, f64)>,
    /// The same function with everything folded to numbers (used if the uniform form is rejected).
    pub folded: Option<Expr>,
    /// Domain colouring of a complex item.
    pub complex: bool,
}

/// One visible item restricted to the slice.
pub struct SliceItem {
    pub color: [f32; 4],
    pub line_w: f32,
    /// The restricted scalar `F` over the free axes (variables are among `x`, `y`, `z`), if the
    /// item has an implicit/explicit form.
    pub f: Option<Expr>,
    /// Ambient points on the slice: where a parametric/polar curve crosses it, and the members
    /// of a point / point list within [`math_core::slice::POINT_TOL_FRAC`] of it (snapped onto
    /// the slice).
    pub points: Vec<[f64; 3]>,
    /// 2D raster for a plane slice (hue field, inequality fill, domain colouring).
    pub field: Option<SliceField>,
    /// `Some(greater)` for an inequality: the region is where `f > 0` (`greater`) or `f < 0`.
    pub region: Option<bool>,
    /// A scalar field item restricted to the slice, plotted along a LINE slice.
    pub graph: Option<Expr>,
}

impl SliceItem {
    /// The function plotted in the inset of a line slice.
    pub fn line_fn(&self) -> Option<&Expr> {
        self.graph.as_ref().or(self.f.as_ref())
    }
}

/// Geometry of a slice in free-axis coordinates.
#[derive(Default)]
pub struct SliceGeo {
    /// Per item: contour segments `(u, v)` (2D slice).
    pub curves: Vec<Vec<[[f64; 2]; 2]>>,
    /// Per item: roots along the free axis (1D slice).
    pub roots: Vec<Vec<f64>>,
    /// Per item: intervals of the free axis where an inequality holds (1D slice).
    pub intervals: Vec<Vec<[f64; 2]>>,
}

impl SliceGeo {
    /// Items whose curve is non-empty, and the total number of marked points.
    pub fn counts(&self, items: &[SliceItem]) -> (usize, usize) {
        let curves = self.curves.iter().filter(|c| !c.is_empty()).count();
        let pts = self.roots.iter().map(Vec::len).sum::<usize>()
            + items.iter().map(|i| i.points.len()).sum::<usize>();
        (curves, pts)
    }
}

/// The window the slice is drawn over: the main window, except on a sloped plane where slots 0
/// and 1 carry the in-plane `(u, v)` ranges of the plane's section of the main window.
pub fn slice_window(rs: &ResolvedSlice, win: &Window3) -> Window3 {
    let Some(pl) = &rs.plane else { return *win };
    let [(u0, u1), (v0, v1)] = pl.uv_ranges(win.min, win.max);
    let mut w = *win;
    w.min[0] = u0;
    w.max[0] = u1;
    w.min[1] = v0;
    w.max[1] = v1;
    w
}

fn turn_of(angle: Angle) -> f64 {
    match angle {
        Angle::Rad => 2.0 * PI,
        Angle::Deg => 360.0,
    }
}

/// Restricts every visible item to the slice (colours match the main scene).
pub fn collect(
    items: &[Prepared],
    defs: &Defs,
    theme: &Theme,
    angle: Angle,
    rs: &ResolvedSlice,
    win: &Window3,
) -> Vec<SliceItem> {
    let free = rs.free_axes();
    let tol = rs.point_tolerances(win.min, win.max);
    let consts: Vec<(String, f64)> = rs
        .fixed_axes()
        .into_iter()
        .map(|a| (math_core::slice::const_var(a), rs.fixed[a].unwrap_or(0.0)))
        .collect();
    let renamed_ok = |e: &Expr| -> bool {
        e.free_vars().iter().all(|v| match v.as_str() {
            "x" => !free.is_empty(),
            "y" => free.len() == 2,
            "z" => false,
            _ => true,
        })
    };
    let spatial_ok = |e: &Expr| -> bool {
        e.free_vars()
            .iter()
            .all(|v| match math_core::slice::axis_index(v) {
                Some(i) => free.contains(&i),
                None => true,
            })
    };
    let sub = |a: &str, e: &Expr| Expr::bin(BinOp::Sub, Expr::var(a), e.clone());
    let mut out = Vec::new();
    let mut visible_idx = 0usize;
    for pr in items.iter().filter(|p| !p.item.hidden) {
        if matches!(pr.kind, Kind::Definition { .. }) {
            continue;
        }
        let mut color = pr
            .item
            .color
            .as_deref()
            .and_then(parse_hex_color)
            .unwrap_or_else(|| theme.color(visible_idx));
        visible_idx += 1;
        if let Some(o) = pr.item.style.opacity {
            color[3] *= o.clamp(0.0, 1.0) as f32;
        }
        let line_w = pr
            .item
            .style
            .line_width
            .map(|w| w as f32)
            .unwrap_or(SLICE_CURVE_W);
        let mut it = SliceItem {
            color,
            line_w,
            f: None,
            points: Vec::new(),
            field: None,
            region: None,
            graph: None,
        };
        if let Some(c) = &pr.complex {
            // A complex item is a function of x + iy (it ignores the sliced axis), so a plane
            // slice through x and y (z = c) shows its domain colouring; other slices skip it.
            if free == [0, 1] {
                it.field = Some(SliceField {
                    kind: FieldKind::Domain,
                    raw: c.clone(),
                    consts: Vec::new(),
                    folded: None,
                    complex: true,
                });
            }
            out.push(it);
            continue;
        }
        let res = |e: &Expr| defs.resolve(e).ok();
        let three = rs.mode == Mode::D3;
        let f_expr: Option<Expr> = match &pr.kind {
            Kind::ExplicitX { rhs } => res(rhs).map(|r| sub("x", &r)),
            Kind::ExplicitY { rhs } => res(rhs).map(|r| sub("y", &r)),
            Kind::ExplicitZ { rhs } if three => res(rhs).map(|r| sub("z", &r)),
            Kind::Implicit { f } | Kind::Inequality { f, .. } => res(f),
            _ => None,
        };
        if let Some(f) = f_expr {
            let r = rs.restrict(&f);
            if spatial_ok(&r) {
                it.f = Some(r);
            }
        }
        // Raster / graph forms of scalar fields and inequality regions.
        let raster: Option<(FieldKind, &Expr)> = match &pr.kind {
            Kind::Field { expr } => Some((FieldKind::Hue, expr)),
            Kind::Inequality { rel, f } => {
                let greater = matches!(rel, Rel::Gt | Rel::Ge);
                it.region = Some(greater);
                Some((FieldKind::Fill { greater }, f))
            }
            _ => None,
        };
        if let Some((kind, raw)) = raster {
            if free.len() == 2 {
                let virt = math_core::slice::virtualize(raw, rs);
                if renamed_ok(&virt) {
                    let folded =
                        res(raw).map(|r| math_core::slice::rename_free(&rs.restrict(&r), rs));
                    it.field = Some(SliceField {
                        kind,
                        raw: virt,
                        consts: consts.clone(),
                        folded,
                        complex: false,
                    });
                }
            } else if matches!(pr.kind, Kind::Field { .. }) {
                if let Some(rr) = res(raw).map(|r| rs.restrict(&r)) {
                    if spatial_ok(&rr) {
                        it.graph = Some(rr);
                    }
                }
            }
        }
        // Points and lists of points / tuples near the slice.
        for p in point_values(pr, defs, angle, three) {
            if rs.contains_point(p, tol) {
                it.points.push(rs.snap(p));
            }
        }
        // Parametric / polar curves cross a slice PLANE (one fixed axis, or a sloped plane) at
        // isolated points.
        if rs.fixed_axes().len() == 1 || rs.plane.is_some() {
            let axis = rs.fixed_axes().first().copied().unwrap_or(0);
            let c = rs.fixed[axis].unwrap_or(0.0);
            let curve: Option<(Vec<Expr>, &str, f64)> = match &pr.kind {
                Kind::Parametric { components }
                    if components.len() == 2 || (three && components.len() == 3) =>
                {
                    components
                        .iter()
                        .map(res)
                        .collect::<Option<Vec<_>>>()
                        .map(|v| (v, "t", turn_of(angle)))
                }
                Kind::Polar { rhs } if !three => res(rhs).map(|r| {
                    let k = if theta_outside_trig(&r) { 3.0 } else { 2.0 };
                    let th = || Expr::var("theta");
                    let xe = Expr::bin(BinOp::Mul, r.clone(), Expr::call("cos", vec![th()]));
                    let ye = Expr::bin(BinOp::Mul, r, Expr::call("sin", vec![th()]));
                    (vec![xe, ye], "theta", turn_of(angle) * k)
                }),
                _ => None,
            };
            if let Some((comps, var, t1)) = curve {
                let progs: Vec<Program> = comps
                    .iter()
                    .filter_map(|c| compile(c, &[var], angle).ok())
                    .collect();
                // `g` is zero where the curve meets the slice: its fixed coordinate minus the
                // constant, or its signed distance from a sloped plane.
                let g = match &rs.plane {
                    Some(pl) if comps.len() == 3 => {
                        let term = |a: usize| {
                            Expr::bin(BinOp::Mul, Expr::num(pl.n[a]), comps[a].clone())
                        };
                        let sum = Expr::bin(
                            BinOp::Add,
                            Expr::bin(BinOp::Add, term(0), term(1)),
                            term(2),
                        );
                        Some(Expr::bin(BinOp::Sub, sum, Expr::num(pl.d)))
                    }
                    Some(_) => None,
                    None if axis < comps.len() => {
                        Some(Expr::bin(BinOp::Sub, comps[axis].clone(), Expr::num(c)))
                    }
                    None => None,
                };
                if progs.len() == comps.len() {
                    if let Some(Ok(gp)) = g.map(|g| compile(&g, &[var], angle)) {
                        for t in find_roots(&gp, 0.0, t1) {
                            let mut p = [0.0; 3];
                            for (a, pg) in progs.iter().enumerate() {
                                p[a] = pg.eval(&[t]);
                            }
                            if rs.plane.is_some() {
                                p = rs.snap(p);
                            } else {
                                p[axis] = c;
                            }
                            if p.iter().all(|v| v.is_finite()) {
                                it.points.push(p);
                            }
                        }
                    }
                }
            }
        }
        out.push(it);
    }
    out
}

/// Ambient positions of a point item or the members of a list of points / tuples (2-tuples get
/// `z = 0`; in 2D only 2-tuples count, as in the main scene). Anything else gives nothing.
fn point_values(pr: &Prepared, defs: &Defs, angle: Angle, three: bool) -> Vec<[f64; 3]> {
    if !matches!(
        pr.kind,
        Kind::List { .. } | Kind::Point { .. } | Kind::Value { .. }
    ) {
        return Vec::new();
    }
    let Ok(r) = defs.resolve(&pr.expr) else {
        return Vec::new();
    };
    if ["x", "y", "z"].iter().any(|v| r.contains_var(v)) {
        return Vec::new();
    }
    let tuples: Vec<Vec<f64>> = if is_listish(&r) {
        match eval_value(&r, &Bindings::new().with_angle(angle)) {
            Ok(Value::Point(p)) => vec![p],
            Ok(Value::PointList(ps)) => ps,
            _ => Vec::new(),
        }
    } else if let Expr::Tuple(comps) = &r {
        let vals: Vec<f64> = comps
            .iter()
            .filter_map(|c| compile(c, &[], angle).ok().map(|p| p.eval(&[])))
            .collect();
        if vals.len() == comps.len() {
            vec![vals]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    tuples
        .into_iter()
        .filter(|t| {
            if three {
                matches!(t.len(), 2 | 3)
            } else {
                t.len() == 2
            }
        })
        .map(|t| [t[0], t[1], t.get(2).copied().unwrap_or(0.0)])
        .collect()
}

/// Intervals of `[lo, hi]` where the sign test holds, from the roots (the sign is constant
/// between consecutive roots, so each piece is decided by its midpoint).
fn region_intervals(p: &Program, roots: &[f64], lo: f64, hi: f64, greater: bool) -> Vec<[f64; 2]> {
    let mut cuts = vec![lo];
    let mut rs: Vec<f64> = roots
        .iter()
        .copied()
        .filter(|r| *r > lo && *r < hi)
        .collect();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cuts.extend(rs);
    cuts.push(hi);
    let mut out: Vec<[f64; 2]> = Vec::new();
    for w in cuts.windows(2) {
        let v = p.eval(&[0.5 * (w[0] + w[1])]);
        if v.is_finite() && (if greater { v > 0.0 } else { v < 0.0 }) {
            match out.last_mut() {
                Some(l) if l[1] == w[0] => l[1] = w[1],
                _ => out.push([w[0], w[1]]),
            }
        }
    }
    out
}

/// Cell size and depth for a contour over a `su` x `sv` world box shown in `px` pixels.
fn contour_opts(su: f64, sv: f64, px: (f64, f64)) -> (f64, u32) {
    let min_cell = (su / px.0.max(1.0)).min(sv / px.1.max(1.0)) * 2.0;
    let big = su.max(sv);
    let depth = if min_cell > 0.0 && min_cell.is_finite() {
        ((big / min_cell).log2().ceil() + 1.0).clamp(4.0, 16.0) as u32
    } else {
        10
    };
    (min_cell, depth)
}

/// Zero set of `p` that never changes sign (a tangent / double root such as `-x^2`, a flat
/// line `x^2 = 0`, `(x^2+y^2-1)^2`), which marching squares cannot see. Interval pruning keeps the
/// cells of a uniform grid whose range contains zero; in each, a small pattern search minimises
/// `|f|`, and a cell whose minimum is (relatively) zero is "touched" at that point. Touched points
/// of neighbouring cells are joined into segments. A single isolated touch (a tangent point)
/// gives no segment. Bounded work: at most `budget` interval evaluations.
fn touching_zero_set(
    p: &Program,
    u: (f64, f64),
    v: (f64, f64),
    min_cell: f64,
    budget: usize,
) -> Vec<[[f64; 2]; 2]> {
    use std::collections::HashMap;
    let (su, sv) = (u.1 - u.0, v.1 - v.0);
    if !(su.is_finite() && sv.is_finite() && su > 0.0 && sv > 0.0) {
        return Vec::new();
    }
    let big = su.max(sv);
    let target = (min_cell * 2.0).max(big / 512.0);
    let depth = ((big / target).log2().ceil().max(1.0) as u32).min(10);
    let (cw, ch) = (su / f64::from(1u32 << depth), sv / f64::from(1u32 << depth));
    let mut ist: Vec<Interval> = Vec::with_capacity(16);
    let mut stack = vec![(u.0, v.0, u.1, v.1, 0u32)];
    let mut evals = 0usize;
    let mut touched: HashMap<(i64, i64), [f64; 2]> = HashMap::new();
    while let Some((ax, ay, bx, by, d)) = stack.pop() {
        if evals >= budget {
            break;
        }
        evals += 1;
        let r = p.eval_interval_with(&[Interval::new(ax, bx), Interval::new(ay, by)], &mut ist);
        if r.is_empty() || !r.contains_zero() {
            continue;
        }
        if d < depth {
            let (mx, my) = (0.5 * (ax + bx), 0.5 * (ay + by));
            stack.push((ax, ay, mx, my, d + 1));
            stack.push((mx, ay, bx, my, d + 1));
            stack.push((ax, my, mx, by, d + 1));
            stack.push((mx, my, bx, by, d + 1));
            continue;
        }
        // Leaf: minimise |f| over the cell.
        let f = |x: f64, y: f64| p.eval(&[x, y]).abs();
        let mut best = (f64::INFINITY, [0.5 * (ax + bx), 0.5 * (ay + by)]);
        let mut worst = 0.0f64;
        for i in 0..5 {
            for j in 0..5 {
                let (x, y) = (
                    ax + (bx - ax) * i as f64 / 4.0,
                    ay + (by - ay) * j as f64 / 4.0,
                );
                let val = f(x, y);
                if !val.is_finite() {
                    continue;
                }
                worst = worst.max(val);
                if val < best.0 {
                    best = (val, [x, y]);
                }
            }
        }
        if !best.0.is_finite() {
            continue;
        }
        let mut step = 0.25 * (bx - ax).max(by - ay);
        let min_step = step * 1e-7;
        while step > min_step {
            let [bx0, by0] = best.1;
            let mut improved = false;
            for (dx, dy) in [
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (1.0, 1.0),
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
            ] {
                let (x, y) = (
                    (bx0 + dx * step).clamp(ax, bx),
                    (by0 + dy * step).clamp(ay, by),
                );
                let val = f(x, y);
                if val.is_finite() && val < best.0 {
                    best = (val, [x, y]);
                    improved = true;
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        if best.0 <= 1e-6 * worst.max(f64::MIN_POSITIVE) || best.0 == 0.0 {
            let key = (
                ((ax - u.0) / cw).round() as i64,
                ((ay - v.0) / ch).round() as i64,
            );
            touched.insert(key, best.1);
        }
    }
    let reach = 3.0 * cw.hypot(ch);
    let mut keys: Vec<_> = touched.keys().copied().collect();
    keys.sort_unstable();
    let mut segs = Vec::new();
    for k in keys {
        let a = touched[&k];
        for d in [(1, 0), (0, 1), (1, 1), (-1, 1)] {
            if let Some(b) = touched.get(&(k.0 + d.0, k.1 + d.1)) {
                let len = (a[0] - b[0]).hypot(a[1] - b[1]);
                if len > 1e-9 * big && len <= reach {
                    segs.push([a, *b]);
                }
            }
        }
    }
    segs
}

/// Contours / roots of every item over the free-axis ranges of `win`; `px` is the pixel size of
/// the free axes (for the contour resolution).
pub fn compute(
    rs: &ResolvedSlice,
    items: &[SliceItem],
    win: &Window3,
    px: (f64, f64),
    angle: Angle,
) -> SliceGeo {
    let free = rs.free_axes();
    let names: Vec<&str> = free.iter().map(|a| axis_name(*a)).collect();
    let mut geo = SliceGeo::default();
    for it in items {
        let mut curve = Vec::new();
        let mut roots = Vec::new();
        let mut intervals = Vec::new();
        if let Some(Ok(p)) = it.f.as_ref().map(|f| compile(f, &names, angle)) {
            if free.len() == 2 {
                let (u, v) = (free[0], free[1]);
                let (mc, depth) =
                    contour_opts(win.max[u] - win.min[u], win.max[v] - win.min[v], px);
                curve = mesh::contour_2d(
                    &p,
                    (win.min[u], win.max[u]),
                    (win.min[v], win.max[v]),
                    mc,
                    depth,
                    400_000,
                );
                if curve.is_empty() {
                    // No sign change anywhere: the zero set may still exist (tangent / double root).
                    curve = touching_zero_set(
                        &p,
                        (win.min[u], win.max[u]),
                        (win.min[v], win.max[v]),
                        mc,
                        60_000,
                    );
                }
            } else if free.len() == 1 {
                let (lo, hi) = (win.min[free[0]], win.max[free[0]]);
                roots = find_roots(&p, lo, hi);
                if let Some(greater) = it.region {
                    intervals = region_intervals(&p, &roots, lo, hi, greater);
                }
            }
        }
        geo.curves.push(curve);
        geo.roots.push(roots);
        geo.intervals.push(intervals);
    }
    geo
}

impl<'a> Builder<'a> {
    /// A translucent unlit quad through four ambient points.
    fn quad4(&mut self, corners: [[f64; 3]; 4], color: [f32; 4]) {
        let c = corners.map(|p| self.rb(p));
        if c.iter().flatten().any(|v| !v.is_finite()) {
            return;
        }
        let base = self.out.vertices.len() as u32;
        for p in c {
            self.out.vertices.push(MeshVertex::new(p, [0.0; 3], color));
        }
        self.out
            .flat_indices
            .extend([0, 1, 2, 0, 2, 3].map(|i| i + base));
    }

    /// A dot with a contrasting halo so it reads on top of a surface of the same colour.
    fn halo_dot(&mut self, p: [f64; 3], color: [f32; 4], w: f32) {
        let mut halo = self.theme.background;
        halo[3] = 0.95;
        let mut ring = self.theme.axis;
        ring[3] = 1.0;
        self.biased_seg(p, p, w + 4.0, halo, DEPTH_BIAS);
        self.biased_seg(p, p, w + 2.5, ring, DEPTH_BIAS);
        self.biased_seg(p, p, w, color, DEPTH_BIAS);
    }

    /// A segment pulled towards the camera (see `SegmentInstance::with_depth_bias`).
    fn biased_seg(&mut self, a: [f64; 3], b: [f64; 3], w: f32, color: [f32; 4], bias: f32) {
        let (p0, p1) = (self.rb(a), self.rb(b));
        if p0.iter().chain(p1.iter()).all(|v| v.is_finite()) {
            self.out
                .overlay_segments
                .push(SegmentInstance::new(p0, p1, w, color).with_depth_bias(bias));
        }
    }

    /// Main-scene overlay of a resolved slice.
    pub fn slice_overlay(&mut self, rs: &ResolvedSlice, items: &[SliceItem], geo: &SliceGeo) {
        let free = rs.free_axes();
        let win = self.win;
        let (lo, hi) = (win.min, win.max);
        let mut halo = self.theme.background;
        halo[3] = 1.0;
        // A sloped plane is drawn as its section of the window (a fan of triangles).
        let section = rs.plane.map(|pl| pl.section(lo, hi));
        let in_window = match &section {
            Some(poly) => !poly.is_empty(),
            None => rs.fixed_axes().iter().all(|a| {
                let c = rs.fixed[*a].unwrap_or(0.0);
                c >= lo[*a] && c <= hi[*a]
            }),
        };
        if free.len() == 2 {
            let (u, v) = (free[0], free[1]);
            if in_window {
                let corners: Vec<[f64; 3]> = match section {
                    Some(poly) => poly,
                    None => [
                        [lo[u], lo[v]],
                        [hi[u], lo[v]],
                        [hi[u], hi[v]],
                        [lo[u], hi[v]],
                    ]
                    .map(|c| rs.lift(&c))
                    .to_vec(),
                };
                let mut fill = SLICE_COLOR;
                fill[3] = PLANE_ALPHA;
                if corners.len() == 4 {
                    self.quad4([corners[0], corners[1], corners[2], corners[3]], fill);
                } else {
                    for i in 1..corners.len().saturating_sub(1) {
                        self.quad4([corners[0], corners[i], corners[i + 1], corners[i + 1]], fill);
                    }
                }
                // Thin (never above 1.6x) and in front of the box edges it runs along, so the
                // plane's rim stays one clean line at the heavy print weights.
                let rim = 2.0 * self.ext.line_mul.min(1.6) / self.ext.line_mul;
                for i in 0..corners.len() {
                    self.seg_front(corners[i], corners[(i + 1) % corners.len()], rim, SLICE_COLOR, 4.0e-4);
                }
            }
            for (it, segs) in items.iter().zip(&geo.curves) {
                // One opaque 3-4 px line in the contrasting slice colour, pulled towards the camera, with a
                // 1 px halo of the background colour: no dark groove, no white sliver.
                let w = it.line_w.clamp(CUT_MIN_W, CUT_MAX_W);
                for s in segs {
                    let (a, b) = (rs.lift(&s[0]), rs.lift(&s[1]));
                    self.biased_seg(a, b, w + 2.0, halo, CUT_BIAS);
                }
                for s in segs {
                    let (a, b) = (rs.lift(&s[0]), rs.lift(&s[1]));
                    self.biased_seg(a, b, w, CUT_COLOR, CUT_BIAS);
                }
            }
        } else if free.len() == 1 && in_window {
            let u = free[0];
            for (it, ivs) in items.iter().zip(&geo.intervals) {
                let mut c = it.color;
                c[3] *= INTERVAL_ALPHA;
                for iv in ivs {
                    self.seg(rs.lift(&[iv[0]]), rs.lift(&[iv[1]]), INTERVAL_W, c);
                }
            }
            self.seg(rs.lift(&[lo[u]]), rs.lift(&[hi[u]]), 3.0, SLICE_COLOR);
            for (it, roots) in items.iter().zip(&geo.roots) {
                for r in roots {
                    self.halo_dot(rs.lift(&[*r]), it.color, self.point_px(SLICE_DOT_W));
                }
            }
        }
        for it in items {
            for p in &it.points {
                self.halo_dot(*p, it.color, self.point_px(SLICE_DOT_W));
            }
        }
    }
}

/// The secondary inset: the slice drawn as a flat 2D graph in its own small window.
pub struct SlicePanel {
    pub geometry: SceneGeometry,
    /// 2D camera of the inset (its own window).
    pub rig: Rig,
    /// Render origin the geometry was built against (`rig.render_origin()`).
    pub origin: [f64; 3],
    /// `[x, y, w, h]` in pixels from the top-left corner of the canvas.
    pub rect: [u32; 4],
    /// Free-axis names, horizontal then vertical (`"f"` vertical for a 1D slice).
    pub axes: [String; 2],
    /// Items with a curve and total marked points (roots / crossings / nearby points).
    pub curves: usize,
    pub points: usize,
    /// The window the inset shows, `[xmin, xmax, ymin, ymax]` in free-axis (horizontal, vertical)
    /// values; for a line slice the vertical range is the plotted value `f`. In a plane slice
    /// the vertical span always follows the horizontal one and the inset's aspect.
    pub view: [f64; 4],
    /// Indices of the free axes (horizontal, vertical) this inset shows.
    pub free: Vec<usize>,
    /// True while the inset follows the main window / auto-fits (no explicit view applied).
    pub follow: bool,
}

/// An explicit inset view and the free axes it was set for (it only applies to that slice
/// orientation, so a new slice through other axes goes back to following).
#[derive(Clone, Debug, PartialEq)]
pub struct ViewReq {
    pub free: Vec<usize>,
    pub view: [f64; 4],
}

/// What the inset needs beyond the slice: definitions (with regression fits) and view request.
pub struct PanelCtx<'a> {
    pub defs: &'a Defs,
    pub pdefs: &'a Defs,
    pub sliders: BTreeMap<String, f64>,
    pub view: Option<&'a ViewReq>,
}

/// Inset rectangle `[x, y, w, h]` (top-left origin) for a canvas of `size`, top-left corner (the
/// right edge belongs to the floating controls, the bottom-left to the phone sheet's button, and
/// the plane's lower-left corner would sit under a bottom inset). At most 24% of the canvas width
/// and 180 x 180 px (square, so a round curve fills it); on a narrow canvas (480 px or less) about 110 x 88 px, so it does not cover the scene (less only on a canvas too small for it).
pub fn inset_rect(size: (u32, u32)) -> [u32; 4] {
    let (cw, ch) = (size.0.max(1) as f64, size.1.max(1) as f64);
    let m = (0.015 * cw.min(ch)).clamp(6.0, 18.0);
    let narrow = cw <= 480.0;
    let mut w = (if narrow { 0.26 } else { 0.24 } * cw)
        .min(180.0)
        .max(if narrow { 0.0 } else { 108.0_f64.min(0.3 * cw) })
        .min((cw - 2.0 * m).max(40.0));
    let mut h = w;
    if narrow {
        w = (0.3 * cw).clamp(100.0, 110.0).min((cw - 2.0 * m).max(40.0));
        h = w * 0.8;
    }
    if h > ch * 0.45 {
        h = (ch * 0.45).max(30.0);
        w = (h / 0.75).min(w);
    }
    let x = m.min((cw - w).max(0.0));
    let y = m.min((ch - h).max(0.0));
    [x as u32, y as u32, w.max(1.0) as u32, h.max(1.0) as u32]
}

fn robust_range(vals: &mut Vec<f64>) -> (f64, f64) {
    vals.retain(|v| v.is_finite());
    if vals.is_empty() {
        return (-1.0, 1.0);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let (mut lo, mut hi) = if n >= 40 {
        (vals[n * 3 / 100], vals[(n * 97 / 100).min(n - 1)])
    } else {
        (vals[0], vals[n - 1])
    };
    // The zero line (where the roots are) is always in view.
    lo = lo.min(0.0);
    hi = hi.max(0.0);
    let mut span = hi - lo;
    if !(span.is_finite() && span > 1e-12) {
        lo -= 1.0;
        hi += 1.0;
        span = hi - lo;
    }
    (lo - 0.12 * span, hi + 0.12 * span)
}

/// Builds the inset for `rs` over the main `window` (free-axis ranges are taken from it unless
/// `ctx.view` gives an explicit view for these free axes). `main_px` is the canvas size; the
/// panel is `inset_rect(main_px)`.
pub fn build_panel(
    doc: &Doc,
    rs: &ResolvedSlice,
    items: &[SliceItem],
    window: Window3,
    main_px: (u32, u32),
    theme: &Theme,
    ctx: &PanelCtx,
) -> SlicePanel {
    let rect = inset_rect(main_px);
    let (pw, ph) = (rect[2] as f64, rect[3] as f64);
    let aspect = pw / ph;
    let angle = match doc.view.angle {
        AngleMode::Rad => Angle::Rad,
        AngleMode::Deg => Angle::Deg,
    };
    let win = slice_window(rs, &window.sanitized());
    let free = rs.free_axes();
    let two_d = free.len() == 2;
    let u = free[0];
    let names: Vec<&str> = free.iter().map(|a| axis_name(*a)).collect();
    let labels = rs.free_names();
    let req = ctx
        .view
        .filter(|r| r.free == free && math_core::slice::view_valid(&r.view))
        .map(|r| r.view);
    let follow = req.is_none();
    // Horizontal range: the free axis; for a 2D slice widen so both ranges fit the aspect.
    let (mut u0, mut u1) = req
        .map(|v| (v[0], v[1]))
        .unwrap_or((win.min[u], win.max[u]));
    let (vc, k, v_lo_s, v_hi_s);
    let view;
    let fit = if two_d && req.is_none() {
        let g = compute(rs, items, &win, (pw, ph), angle);
        curve_fit(&g, free[0], free[1], &win)
    } else {
        None
    };
    if two_d {
        let v = free[1];
        match req {
            Some(r) => vc = 0.5 * (r[2] + r[3]),
            None => {
                let (v0, v1) = (win.min[v], win.max[v]);
                if let Some(f) = fit {
                    // Auto-fit: centre on the curve, ~20% padding, never wider than the window.
                    u0 = f[0];
                    u1 = f[1];
                    vc = 0.5 * (f[2] + f[3]);
                    let (fv0, fv1) = (f[2], f[3]);
                    let need_u = (fv1 - fv0) * aspect;
                    if need_u > u1 - u0 {
                        let c = 0.5 * (u0 + u1);
                        u0 = c - need_u / 2.0;
                        u1 = c + need_u / 2.0;
                    }
                } else {
                    let need_u = (v1 - v0) * aspect;
                    if need_u > u1 - u0 {
                        let c = 0.5 * (u0 + u1);
                        u0 = c - need_u / 2.0;
                        u1 = c + need_u / 2.0;
                    }
                    vc = 0.5 * (v0 + v1);
                }
            }
        }
        k = 1.0;
        let half = (u1 - u0) / aspect / 2.0;
        v_lo_s = vc - half;
        v_hi_s = vc + half;
        view = [u0, u1, v_lo_s, v_hi_s];
    } else {
        let (lo, hi) = match req {
            Some(r) => (r[2], r[3]),
            None => {
                let mut vals = Vec::new();
                for it in items {
                    if let Some(Ok(p)) = it.line_fn().map(|f| compile(f, &names, angle)) {
                        let mut st = Vec::with_capacity(16);
                        for i in 0..=RANGE_SAMPLES {
                            let x = u0 + (u1 - u0) * i as f64 / RANGE_SAMPLES as f64;
                            vals.push(p.eval_with(&[x], &mut st));
                        }
                    }
                }
                robust_range(&mut vals)
            }
        };
        vc = 0.5 * (lo + hi);
        let uspan = u1 - u0;
        k = (uspan / aspect) / (hi - lo);
        v_lo_s = -(uspan / aspect) / 2.0;
        v_hi_s = (uspan / aspect) / 2.0;
        view = [u0, u1, lo, hi];
    }
    // Panel window in scaled coordinates: x = u, y = (v - vc) * k.
    let panel_win = Window3::new([u0, v_lo_s, -1.0], [u1, v_hi_s, 1.0]);
    let mut rig = Rig::new(panel_win, Mode::D2);
    rig.set_aspect(aspect);
    let origin = rig.render_origin();
    let mut sliders = ctx.sliders.clone();
    for it in items {
        if let Some(f) = &it.field {
            sliders.extend(f.consts.iter().cloned());
        }
    }
    let mut b = Builder {
        hosts: Vec::new(),
        out: SceneGeometry::default(),
        origin,
        win: panel_win,
        vw: pw,
        vh: ph,
        theme,
        angle,
        pdefs: ctx.pdefs.clone(),
        sliders,
        // The inset follows the print weight like the main scene (lines, grid and point sizes).
        ext: {
            let (lm, gm, pm) = weight_mul(doc.view.weight);
            let scale = RENDER_SCALE.get();
            BuildExt {
                line_mul: lm * scale,
                grid_mul: gm * scale,
                point_mul: pm * scale,
                scale,
                ..Default::default()
            }
        },
    };
    // y of a free-axis value `v` in panel coordinates.
    let ys = |v: f64| if two_d { v } else { (v - vc) * k };
    // Opaque backdrop, so the inset reads over the 3D scene. The flat quad draws after fields,
    // so a panel with rasters gets an opaque one-colour domain field drawn first instead.
    let mut bg = theme.background;
    bg[3] = 1.0;
    let has_fields = two_d && items.iter().any(|i| i.field.is_some());
    if has_fields {
        b.backdrop_field(bg);
    } else {
        b.quad(u0, u1, v_lo_s, v_hi_s, bg);
    }
    if two_d {
        for it in items {
            if let Some(f) = &it.field {
                b.slice_field(f, it.color, ctx.defs);
            }
        }
    }
    let ts = doc.view.text_scale.clamp(0.5, 3.0);
    let tick_boxes = panel_axes(&mut b, two_d, (u0, u1), (v_lo_s, v_hi_s), vc, k, pw, ph, ts);

    let geo = compute(
        rs,
        items,
        &win_for_panel(&win, &free, (u0, u1), (v_lo_s, v_hi_s), two_d),
        (pw, ph),
        angle,
    );
    if two_d {
        for (it, segs) in items.iter().zip(&geo.curves) {
            for s in segs {
                b.seg(
                    [s[0][0], s[0][1], 0.0],
                    [s[1][0], s[1][1], 0.0],
                    INSET_CURVE_W,
                    it.color,
                );
            }
        }
        for it in items {
            for p in &it.points {
                let q = rs.project(*p);
                b.halo_dot([q[0], q[1], 0.0], it.color, b.point_px(8.0));
            }
        }
    } else {
        for (it, ivs) in items.iter().zip(&geo.intervals) {
            let mut c = it.color;
            c[3] *= INTERVAL_ALPHA;
            for iv in ivs {
                b.seg([iv[0], ys(0.0), 0.0], [iv[1], ys(0.0), 0.0], INTERVAL_W, c);
            }
        }
        for it in items {
            if let Some(Ok(_)) = it.line_fn().map(|f| compile(f, &names, angle)) {
                let scaled = Expr::bin(
                    BinOp::Mul,
                    Expr::bin(
                        BinOp::Sub,
                        it.line_fn().cloned().unwrap_or(Expr::num(0.0)),
                        Expr::num(vc),
                    ),
                    Expr::num(k),
                );
                if let Ok(p) = compile(&scaled, &names, angle) {
                    let lines = mesh::sample_explicit(&p, u0, u1, pw as usize, (v_lo_s, v_hi_s));
                    for l in &lines {
                        let pts: Vec<[f64; 3]> = l.iter().map(|q| [q[0], q[1], 0.0]).collect();
                        b.polyline(&pts, Style::new(it.color, INSET_CURVE_W));
                    }
                }
            }
        }
        for (it, roots) in items.iter().zip(&geo.roots) {
            for r in roots {
                b.halo_dot([*r, ys(0.0), 0.0], it.color, b.point_px(8.0));
            }
        }
        for it in items {
            for p in &it.points {
                b.halo_dot([p[u], ys(0.0), 0.0], it.color, b.point_px(8.0));
            }
        }
    }
    // Frame and axis names (placed so their text boxes lie inside the rectangle).
    let mut frame = theme.axis;
    frame[3] = 0.9;
    let corners = [[u0, v_lo_s], [u1, v_lo_s], [u1, v_hi_s], [u0, v_hi_s]];
    for i in 0..4 {
        let (a, c) = (corners[i], corners[(i + 1) % 4]);
        b.seg([a[0], a[1], 0.0], [c[0], c[1], 0.0], 2.0, frame);
    }
    let vname = if two_d {
        labels[1].to_string()
    } else {
        "f".to_string()
    };
    let (wu, wv) = ((u1 - u0) / pw, (v_hi_s - v_lo_s) / ph);
    // Axis names go at the far end of their axis; if a tick label is there, slide along the edge
    // (x name leftwards, y name downwards) until clear of every tick box.
    let xn = place_name(
        0,
        labels[0],
        (pw - NAME_INSET_X, ph - NAME_INSET_Y * ts),
        (-1.0, 0.0),
        &tick_boxes,
        &[],
        pw,
        ph,
        ts,
    );
    let xbox = label_box_s(0, labels[0], xn.0, xn.1, ts);
    let yn = place_name(
        1,
        &vname,
        (NAME_INSET_X, 0.5 * LABEL_H * ts + 3.0),
        (0.0, 1.0),
        &tick_boxes,
        &[xbox],
        pw,
        ph,
        ts,
    );
    b.label(
        [u0 + xn.0 * wu, v_hi_s - xn.1 * wv, 0.0],
        labels[0].to_string(),
        0,
    );
    b.label([u0 + yn.0 * wu, v_hi_s - yn.1 * wv, 0.0], vname.clone(), 1);
    let (curves, points) = geo.counts(items);
    let mut g = b.out;
    g.diagnostics.clear();
    SlicePanel {
        geometry: g,
        rig,
        origin,
        rect,
        axes: [labels[0].to_string(), vname],
        curves,
        points,
        view,
        free,
        follow,
    }
}

/// Approximate label box used to keep inset labels inside the rectangle (CSS pixels): width per
/// character and height. The anchor offsets mirror the label painters (x ticks hang below
/// their anchor, y ticks sit left of it).
pub const LABEL_CHAR_W: f64 = 7.0;
pub const LABEL_H: f64 = 12.0;
/// Pixel gap below an x-tick anchor / left of a y-tick anchor.
pub const LABEL_GAP_X: f64 = 5.0;
pub const LABEL_GAP_Y: f64 = 6.0;
const NAME_INSET_X: f64 = 14.0;
const NAME_INSET_Y: f64 = 24.0;

impl Builder<'_> {
    /// An opaque one-colour fill of the panel window, as a domain field (field passes draw
    /// before the flat/segment passes, so the rasters that follow draw over it).
    fn backdrop_field(&mut self, c: [f32; 4]) {
        let wgsl = format!(
            "fn field_color(x: f32, y: f32) -> vec4<f32> {{\n    return vec4<f32>({:?}, {:?}, {:?}, 1.0);\n}}\n",
            c[0], c[1], c[2]
        );
        let (lo, hi) = (self.win.min, self.win.max);
        let rb = |v: f64, a: usize| (v - self.origin[a]) as f32;
        self.out.fields.push(FieldSpec {
            kind: FieldKind::Domain,
            wgsl,
            params: Vec::new(),
            color: [1.0; 4],
            rect_min: [rb(lo[0], 0), rb(lo[1], 1)],
            rect_max: [rb(hi[0], 0), rb(hi[1], 1)],
            origin_xy: [self.origin[0], self.origin[1]],
        });
    }

    /// One raster of a plane slice over the panel window. A failure (for example a shader that
    /// cannot be expressed) is dropped: the inset just shows less.
    fn slice_field(&mut self, f: &SliceField, color: [f32; 4], defs: &Defs) {
        let st = Style::new(color, INSET_CURVE_W);
        if f.complex {
            let _ = self.draw_complex(&f.raw, defs, Mode::D2, st);
            return;
        }
        let n = self.out.fields.len();
        if self.field(&f.raw, defs, f.kind, color).is_err() {
            self.out.fields.truncate(n);
            if let Some(folded) = &f.folded {
                let _ = self.field(folded, defs, f.kind, color);
            }
        }
    }
}

/// Window handed to `compute` for the panel: free-axis ranges in WORLD (unscaled) values (for a
/// 1D slice only the horizontal range matters).
/// `[u0, u1, v0, v1]` hugging every slice curve with [`INSET_FIT_PAD`] around it, clamped to the
/// window. `None` when nothing is drawn or the curve already spans the window.
fn curve_fit(geo: &SliceGeo, u: usize, v: usize, win: &Window3) -> Option<[f64; 4]> {
    let mut e = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for s in geo.curves.iter().flatten() {
        for p in s {
            if p[0].is_finite() && p[1].is_finite() {
                e[0] = e[0].min(p[0]);
                e[1] = e[1].max(p[0]);
                e[2] = e[2].min(p[1]);
                e[3] = e[3].max(p[1]);
            }
        }
    }
    if !(e[0] <= e[1] && e[2] <= e[3]) {
        return None;
    }
    let span = (e[1] - e[0]).max(e[3] - e[2]);
    if !(span > 1e-9) {
        return None;
    }
    let (cu, cv) = (0.5 * (e[0] + e[1]), 0.5 * (e[2] + e[3]));
    let hu = ((e[1] - e[0]) * 0.5 * (1.0 + INSET_FIT_PAD)).max(span * 0.1);
    let hv = ((e[3] - e[2]) * 0.5 * (1.0 + INSET_FIT_PAD)).max(span * 0.1);
    let (wu, wv) = (win.max[u] - win.min[u], win.max[v] - win.min[v]);
    if hu * 2.0 >= wu && hv * 2.0 >= wv {
        return None;
    }
    Some([cu - hu, cu + hu, cv - hv, cv + hv])
}

fn win_for_panel(
    win: &Window3,
    free: &[usize],
    (u0, u1): (f64, f64),
    (v0, v1): (f64, f64),
    two_d: bool,
) -> Window3 {
    let mut w = *win;
    w.min[free[0]] = u0;
    w.max[free[0]] = u1;
    if two_d {
        w.min[free[1]] = v0;
        w.max[free[1]] = v1;
    }
    w
}

/// Grid, axes and tick labels of the inset. For a 1D slice the vertical axis shows the value
/// `v` while panel coordinates are `(v - vc) * k`, so the plot keeps square pixels. Tick labels
/// ride the zero axes but are held far enough inside the rectangle that their boxes (see
/// [`LABEL_CHAR_W`]) fit, and labels whose box would still leave the rectangle are not emitted.
#[allow(clippy::too_many_arguments)]
fn panel_axes(
    b: &mut Builder,
    two_d: bool,
    (u0, u1): (f64, f64),
    (y0, y1): (f64, f64),
    vc: f64,
    k: f64,
    pw: f64,
    ph: f64,
    ts: f64,
) -> Vec<[f64; 4]> {
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let (minor_c, major_c, axis_c) = (b.theme.grid_minor, b.theme.grid_major, b.theme.axis);
    // Value range shown vertically.
    let (v0, v1) = if two_d {
        (y0, y1)
    } else {
        (vc + y0 / k, vc + y1 / k)
    };
    let sy = |v: f64| if two_d { v } else { (v - vc) * k };
    let su = nice_step(u1 - u0, pw, 60.0 * ts);
    let sv = nice_step(v1 - v0, ph, 40.0 * ts);
    for (axis, step, lo, hi) in [(0usize, su, u0, u1), (1usize, sv, v0, v1)] {
        let div = minor_divisions(step);
        let minor = step / div as f64;
        for pass_major in [false, true] {
            for (kk, val) in multiples(lo, hi, minor) {
                let is_major = kk % div == 0;
                if is_major != pass_major {
                    continue;
                }
                let (w, c) = if is_major {
                    (MAJOR_W, major_c)
                } else {
                    (MINOR_W, minor_c)
                };
                if axis == 0 {
                    b.gseg([val, y0, 0.0], [val, y1, 0.0], w, c);
                } else {
                    b.gseg([u0, sy(val), 0.0], [u1, sy(val), 0.0], w, c);
                }
            }
        }
    }
    // Axes through zero (clamped into view).
    if v0 <= 0.0 && 0.0 <= v1 {
        b.seg([u0, sy(0.0), 0.0], [u1, sy(0.0), 0.0], AXIS_W, axis_c);
    }
    if u0 <= 0.0 && 0.0 <= u1 {
        b.seg([0.0, y0, 0.0], [0.0, y1, 0.0], AXIS_W, axis_c);
    }
    // Where the tick rows sit: on the zero axes while those are far enough from the edges.
    let (wu, wv) = ((u1 - u0) / pw, (v1 - v0) / ph);
    let row_lo = v0 + (LABEL_GAP_X * ts + LABEL_H * ts + 3.0) * wv;
    let row_hi = v1 - 3.0 * wv;
    let col_lo = u0 + ((LABEL_CHAR_W * 4.0 + LABEL_GAP_Y) * ts + 3.0) * wu;
    let col_hi = u1 - 3.0 * wu;
    let zy = if row_lo <= row_hi {
        0f64.max(row_lo).min(row_hi)
    } else {
        0.5 * (v0 + v1)
    };
    let zx = if col_lo <= col_hi {
        0f64.max(col_lo).min(col_hi)
    } else {
        0.5 * (u0 + u1)
    };
    // Pixel position of a label anchor inside the panel (from the top-left), for the box test.
    let fits = |axis: u8, text: &str, u: f64, v: f64| -> bool {
        let (ax, ay) = ((u - u0) / wu, (v1 - v) / wv);
        label_box_inside_s(axis, text, ax, ay, pw, ph, ts)
    };
    for (_, val) in multiples(u0, u1, su) {
        let text = format_tick(val, su);
        if fits(0, &text, val, zy) {
            boxes.push(label_box_s(0, &text, (val - u0) / wu, (v1 - zy) / wv, ts));
            b.label([val, sy(zy), 0.0], text, 0);
        }
    }
    for (kk, val) in multiples(v0, v1, sv) {
        if kk == 0 && two_d {
            continue;
        }
        let text = format_tick(val, sv);
        if fits(1, &text, zx, val) {
            boxes.push(label_box_s(1, &text, (zx - u0) / wu, (v1 - val) / wv, ts));
            b.label([zx, sy(val), 0.0], text, 1);
        }
    }
    boxes
}

/// Text box `[x0, y0, x1, y1]` (pixels from the panel's top-left) of a label anchored at `(x, y)`.
pub fn label_box(axis: u8, text: &str, x: f64, y: f64) -> [f64; 4] {
    label_box_s(axis, text, x, y, 1.0)
}

/// [`label_box`] for text drawn `ts` times larger (the view's `textScale`): the character width,
/// height and anchor gaps all scale.
pub fn label_box_s(axis: u8, text: &str, x: f64, y: f64, ts: f64) -> [f64; 4] {
    let (cw, h, gx, gy) = (LABEL_CHAR_W * ts, LABEL_H * ts, LABEL_GAP_X * ts, LABEL_GAP_Y * ts);
    let w = text.chars().count() as f64 * cw;
    let (x0, y0) = match axis {
        0 => (x - w / 2.0, y + gx),
        1 => (x - w - gy, y - h / 2.0),
        _ => (x - w / 2.0, y - h / 2.0),
    };
    [x0, y0, x0 + w, y0 + h]
}

fn boxes_overlap(a: &[f64; 4], b: &[f64; 4], pad: f64) -> bool {
    a[0] < b[2] + pad && b[0] < a[2] + pad && a[1] < b[3] + pad && b[1] < a[3] + pad
}

/// Anchor (pixels from the top-left) for an axis name: the preferred spot if its box is inside the
/// panel and clear of every tick box (and of `taken`), else the nearest clear spot on a sweep
/// along the axis' far end, else the preferred spot (nothing better exists).
#[allow(clippy::too_many_arguments)]
fn place_name(
    axis: u8,
    text: &str,
    pref: (f64, f64),
    sweep: (f64, f64),
    ticks: &[[f64; 4]],
    taken: &[[f64; 4]],
    pw: f64,
    ph: f64,
    ts: f64,
) -> (f64, f64) {
    for i in 0..=60 {
        let t = i as f64 * 6.0;
        for sign in [1.0, -1.0] {
            let (x, y) = (pref.0 + sign * t * sweep.0, pref.1 + sign * t * sweep.1);
            if !label_box_inside_s(axis, text, x, y, pw, ph, ts) {
                continue;
            }
            let bx = label_box_s(axis, text, x, y, ts);
            if ticks
                .iter()
                .chain(taken)
                .all(|o| !boxes_overlap(&bx, o, 2.0))
            {
                return (x, y);
            }
        }
    }
    pref
}

/// True when a label's text box, anchored at `(x, y)` pixels from the panel's top-left corner,
/// lies inside a `pw` x `ph` panel (`axis` 0 hangs below the anchor, 1 sits left of it, others
/// are centred).
pub fn label_box_inside(axis: u8, text: &str, x: f64, y: f64, pw: f64, ph: f64) -> bool {
    label_box_inside_s(axis, text, x, y, pw, ph, 1.0)
}

/// [`label_box_inside`] for text drawn `ts` times larger.
pub fn label_box_inside_s(axis: u8, text: &str, x: f64, y: f64, pw: f64, ph: f64, ts: f64) -> bool {
    let b = label_box_s(axis, text, x, y, ts);
    b[0] >= 0.0 && b[1] >= 0.0 && b[2] <= pw && b[3] <= ph
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_core::doc::{Item, ItemKind, SliderCfg};
    use math_core::slice::{parse_fixed_text, SliceCfg};

    fn doc_with(items: &[&str], slice: &str, mode: Mode) -> Doc {
        let mut d = Doc::new_default();
        for (i, l) in items.iter().enumerate() {
            d.add_item(Item::new(&format!("e{i}"), ItemKind::Equation, l))
                .unwrap();
        }
        d.slice = Some(SliceCfg::new(None, parse_fixed_text(slice).unwrap(), mode).unwrap());
        d
    }

    fn slider(d: &mut Doc, name: &str, v: f64) {
        d.sliders.insert(
            name.into(),
            SliderCfg {
                min: -10.0,
                max: 10.0,
                step: None,
                value: v,
            },
        );
    }

    fn win() -> Window3 {
        Window3::new([-3.0; 3], [3.0; 3])
    }

    /// Restricted items and their geometry for `doc` in `mode`.
    fn geo_of(d: &Doc, mode: Mode) -> (ResolvedSlice, Vec<SliceItem>, SliceGeo) {
        let mut diags = Vec::new();
        let (items, defs, _, _) = prepare(d, &mut diags);
        let rs =
            ResolvedSlice::resolve(d.slice.as_ref().unwrap(), mode, &defs, Angle::Rad).unwrap();
        let w = if mode == Mode::D3 {
            win()
        } else {
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0])
        };
        let si = collect(&items, &defs, &Theme::light(), Angle::Rad, &rs, &w);
        let geo = compute(&rs, &si, &slice_window(&rs, &w), (532.0, 532.0), Angle::Rad);
        (rs, si, geo)
    }

    #[test]
    fn sphere_ring_is_exact() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "z=1", Mode::D3);
        let (rs, _, geo) = geo_of(&d, Mode::D3);
        assert_eq!(rs.dim, 2);
        let segs = &geo.curves[0];
        assert!(segs.len() > 200, "ring has many segments: {}", segs.len());
        let r = 3f64.sqrt();
        for s in segs {
            for p in s {
                assert!(
                    (p[0].hypot(p[1]) - r).abs() < 1e-5,
                    "point off the circle: {p:?}"
                );
            }
        }
    }

    #[test]
    fn explicit_surface_and_slider_sweep() {
        // z = x^2+y^2 cut at z=a is the circle of radius sqrt(a); dragging a moves it.
        let mut d = doc_with(&["z=x^2+y^2"], "z=a", Mode::D3);
        for (a, r) in [(1.0, 1.0), (4.0, 2.0)] {
            slider(&mut d, "a", a);
            let (_, _, geo) = geo_of(&d, Mode::D3);
            assert!(!geo.curves[0].is_empty());
            for s in &geo.curves[0] {
                assert!((s[0][0].hypot(s[0][1]) - r).abs() < 1e-5, "a={a}");
            }
        }
        // The overlay in the main scene follows the slider too (plane quad at z=a).
        slider(&mut d, "a", 2.0);
        let g = build_scene(&d, Mode::D3, win(), [0.0; 3], (900, 600), &Theme::light());
        let zs: Vec<f32> = g.vertices.iter().rev().take(4).map(|v| v.pos[2]).collect();
        assert!(
            zs.iter().all(|z| (z - 2.0).abs() < 1e-6),
            "plane quad at z=2: {zs:?}"
        );
        assert!(!g.overlay_segments.is_empty());
    }

    #[test]
    fn plane_slice_axes_x_and_y() {
        // y = 1 through the sphere: circle in the (x, z) plane of radius sqrt(3).
        let d = doc_with(&["x^2+y^2+z^2=4"], "y=1", Mode::D3);
        let (rs, _, geo) = geo_of(&d, Mode::D3);
        assert_eq!(rs.free_axes(), vec![0, 2]);
        assert!(geo.curves[0]
            .iter()
            .all(|s| (s[0][0].hypot(s[0][1]) - 3f64.sqrt()).abs() < 1e-5));
        // x = 3 misses the radius-2 sphere entirely: no curve, no panic.
        let d = doc_with(&["x^2+y^2+z^2=4"], "x=3", Mode::D3);
        assert!(geo_of(&d, Mode::D3).2.curves[0].is_empty());
    }

    #[test]
    fn line_slice_in_2d_marks_roots() {
        let mut d = doc_with(&["y=x^2"], "y=a", Mode::D2);
        slider(&mut d, "a", 4.0);
        let (rs, _, geo) = geo_of(&d, Mode::D2);
        assert_eq!(rs.dim, 1);
        let mut r = geo.roots[0].clone();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(r.len(), 2);
        assert!(
            (r[0] + 2.0).abs() < 1e-9 && (r[1] - 2.0).abs() < 1e-9,
            "{r:?}"
        );
        // The main scene draws the cut line and two dots (each dot is several overlay passes).
        let g = build_scene(
            &d,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            [0.0; 3],
            (900, 600),
            &Theme::light(),
        );
        let dots = g.overlay_segments.iter().filter(|s| s.p0 == s.p1).count();
        assert_eq!(dots, 2 * 3, "two roots, three passes each");
        let line = g
            .segments
            .iter()
            .filter(|s| s.p0[1] == 4.0 && s.p1[1] == 4.0 && s.width == 3.0)
            .count();
        assert_eq!(line, 1, "the cut line at y=4");
        // Slider below the vertex: no intersection.
        slider(&mut d, "a", -1.0);
        assert!(geo_of(&d, Mode::D2).2.roots[0].is_empty());
    }

    #[test]
    fn line_slice_in_3d() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "y=1, z=0.5", Mode::D3);
        let (rs, _, geo) = geo_of(&d, Mode::D3);
        assert_eq!((rs.dim, rs.free_axes()), (1, vec![0]));
        let expect = (4.0f64 - 1.0 - 0.25).sqrt();
        let mut r = geo.roots[0].clone();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((r[0] + expect).abs() < 1e-9 && (r[1] - expect).abs() < 1e-9);
    }

    #[test]
    fn parametric_curve_crossings() {
        // Unit circle (cos t, sin t) cut by y = 0.5: two points with x = +-sqrt(0.75).
        let d = doc_with(&["(cos(t), sin(t))"], "y=0.5", Mode::D2);
        let (_, items, _) = geo_of(&d, Mode::D2);
        let mut xs: Vec<f64> = items[0].points.iter().map(|p| p[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(xs.len(), 2);
        assert!((xs[1] - 0.75f64.sqrt()).abs() < 1e-9);
        assert!(items[0].points.iter().all(|p| p[1] == 0.5));
    }

    #[test]
    fn mode_mismatch_is_an_error_not_a_panic() {
        let d = doc_with(&["y=x^2"], "z=1", Mode::D3);
        let mut diags = Vec::new();
        let (_, defs, _, _) = prepare(&d, &mut diags);
        assert!(
            ResolvedSlice::resolve(d.slice.as_ref().unwrap(), Mode::D2, &defs, Angle::Rad).is_err()
        );
        assert!(
            build_slice_panel(&d, Mode::D2, win(), (900, 600), &Theme::light())
                .unwrap()
                .is_err()
        );
        // And the main scene just ignores it.
        let g = build_scene(&d, Mode::D2, win(), [0.0; 3], (900, 600), &Theme::light());
        assert!(g.overlay_segments.is_empty());
    }

    #[test]
    fn no_slice_means_no_overlay_and_no_panel() {
        let mut d = Doc::new_default();
        d.add_item(Item::new("a", ItemKind::Equation, "y=x^2"))
            .unwrap();
        assert!(build_slice_panel(&d, Mode::D2, win(), (900, 600), &Theme::light()).is_none());
        let g = build_scene(&d, Mode::D2, win(), [0.0; 3], (900, 600), &Theme::light());
        assert!(g.overlay_segments.is_empty());
    }

    /// World position (panel coordinates) of a segment end, from origin-relative f32.
    fn abs(p: &SlicePanel, v: [f32; 3]) -> [f64; 2] {
        [v[0] as f64 + p.origin[0], v[1] as f64 + p.origin[1]]
    }

    #[test]
    fn tangent_and_double_root_zero_sets_are_drawn() {
        // y = x^2 in 3D is y - x^2; at y = 0 it is -x^2: zero on the line x = 0 with no sign change.
        let d = doc_with(&["y=x^2"], "y=0", Mode::D3);
        let (rs, si, geo) = geo_of(&d, Mode::D3);
        assert_eq!(rs.free_axes(), vec![0, 2]);
        assert_eq!(geo.counts(&si).0, 1, "one curve");
        let segs = &geo.curves[0];
        assert!(segs.len() >= 20);
        for s in segs {
            assert!(
                s[0][0].abs() < 1e-3 && s[1][0].abs() < 1e-3,
                "on x = 0: {s:?}"
            );
        }
        let (lo, hi) = segs.iter().fold((f64::MAX, f64::MIN), |(l, h), s| {
            (l.min(s[0][1].min(s[1][1])), h.max(s[0][1].max(s[1][1])))
        });
        assert!(lo < -2.5 && hi > 2.5, "spans the window along z: {lo} {hi}");
        // The inset reports the curve too.
        let out = build_slice_panel(&d, Mode::D3, win(), (900, 600), &Theme::light())
            .unwrap()
            .unwrap();
        assert_eq!(out.panel.curves, 1);
        // (x^2+y^2-1)^2 = 0 never changes sign: still the unit circle.
        let d = doc_with(&["(x^2+y^2-1)^2+z=0"], "z=0", Mode::D3);
        let (_, si, geo) = geo_of(&d, Mode::D3);
        assert_eq!(geo.counts(&si).0, 1);
        for s in &geo.curves[0] {
            for q in s {
                assert!((q[0].hypot(q[1]) - 1.0).abs() < 0.05, "{q:?}");
            }
        }
    }

    #[test]
    fn tangent_sphere_and_empty_cuts_do_not_panic() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "z=2", Mode::D3);
        let (_, si, geo) = geo_of(&d, Mode::D3);
        assert!(geo.counts(&si).0 <= 1);
        // Misses entirely: x^2+y^2+z^2+1 = 0 has no zero.
        let d = doc_with(&["x^2+y^2+z^2+1=0"], "z=0", Mode::D3);
        let (_, si, geo) = geo_of(&d, Mode::D3);
        assert_eq!(geo.counts(&si).0, 0);
        // A normal crossing is unchanged: still the one exact ring.
        let d = doc_with(&["x^2+y^2+z^2=4"], "z=1", Mode::D3);
        let (_, si, geo) = geo_of(&d, Mode::D3);
        assert_eq!(geo.counts(&si).0, 1);
    }

    #[test]
    fn panel_shows_the_circle_of_radius_sqrt3() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "z=1", Mode::D3);
        let out = build_slice_panel(&d, Mode::D3, win(), (900, 600), &Theme::light())
            .unwrap()
            .unwrap();
        let p = &out.panel;
        assert_eq!(p.axes, ["x".to_string(), "y".to_string()]);
        assert_eq!((p.curves, p.points), (1, 0));
        // Inside the canvas, top-left, 4:3, at most 180 px wide.
        assert!(p.rect[0] + p.rect[2] <= 900 && p.rect[1] + p.rect[3] <= 600);
        assert!(p.rect[0] < 450 && p.rect[1] < 100 && p.rect[2] <= 180);
        let ring: Vec<_> = p
            .geometry
            .segments
            .iter()
            .filter(|s| s.width == INSET_CURVE_W)
            .collect();
        assert!(ring.len() > 200);
        for s in ring {
            let a = abs(p, s.p0);
            assert!((a[0].hypot(a[1]) - 3f64.sqrt()).abs() < 1e-4, "{a:?}");
        }
        // Axis names and tick labels exist.
        assert!(p
            .geometry
            .labels
            .iter()
            .any(|l| l.text == "x" && l.axis == 0));
        assert!(p
            .geometry
            .labels
            .iter()
            .any(|l| l.text == "y" && l.axis == 1));
        assert!(p.geometry.labels.iter().any(|l| l.text.parse::<f64>().is_ok()));
        // An opaque background quad so it reads over the 3D scene.
        assert_eq!(p.geometry.flat_indices.len(), 6);
    }

    #[test]
    fn panel_follows_the_print_weight() {
        let mut d = doc_with(&["x^2+y^2+z^2=4"], "z=1", Mode::D3);
        let ring = |d: &Doc, m: f32| {
            let out = build_slice_panel(d, Mode::D3, win(), (900, 600), &Theme::light())
                .unwrap()
                .unwrap();
            let ws: Vec<f32> = out.panel.geometry.segments.iter().map(|s| s.width).collect();
            (ws.iter().filter(|w| (**w - INSET_CURVE_W * m).abs() < 1e-4).count(), ws)
        };
        let (n1, w1) = ring(&d, 1.0);
        assert!(n1 > 200);
        d.view.weight = math_core::doc::Weight::Extra;
        let (n2, w2) = ring(&d, 2.2);
        assert_eq!(n1, n2, "the ring is drawn at the line multiplier");
        assert!(w2.iter().any(|w| (*w - MINOR_W * 1.5).abs() < 1e-4), "grid uses the gentle one");
        assert!(w2.iter().any(|w| (*w - AXIS_W * 2.2).abs() < 1e-4), "axes: full multiplier");
        assert!(!w2.iter().any(|w| (*w - INSET_CURVE_W).abs() < 1e-4), "no unweighted curve");
        assert!(w1.iter().any(|w| (*w - MINOR_W).abs() < 1e-4));
    }

    #[test]
    fn panel_for_a_line_slice_plots_the_value_along_the_line() {
        let mut d = doc_with(&["y=x^2"], "y=a", Mode::D2);
        slider(&mut d, "a", 4.0);
        let out = build_slice_panel(
            &d,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            (900, 600),
            &Theme::light(),
        )
        .unwrap()
        .unwrap();
        let p = &out.panel;
        assert_eq!(p.axes, ["x".to_string(), "f".to_string()]);
        assert_eq!((p.curves, p.points), (0, 2));
        // The graph is F(x) = a - x^2 scaled in y; its roots sit on the zero line (two dots).
        let xs: Vec<f64> = p
            .geometry
            .overlay_segments
            .iter()
            .filter(|s| s.p0 == s.p1)
            .map(|s| abs(p, s.p0)[0])
            .collect();
        assert_eq!(xs.len(), 2 * 3, "two root dots, three passes each");
        assert!(xs.iter().all(|x| (x.abs() - 2.0).abs() < 1e-4), "{xs:?}");
        assert!(p.geometry.labels.iter().any(|l| l.text == "f"));
    }

    #[test]
    fn inset_rect_fits_any_canvas() {
        for size in [
            (900, 600),
            (300, 200),
            (100, 100),
            (1, 1),
            (2000, 300),
            (390, 800),
        ] {
            let r = inset_rect(size);
            assert!(r[2] >= 1 && r[3] >= 1);
            assert!(r[0] + r[2] <= size.0.max(r[0] + r[2]).max(1));
            if size.0 >= 100 && size.1 >= 100 {
                assert!(
                    r[0] + r[2] <= size.0 && r[1] + r[3] <= size.1,
                    "{size:?} {r:?}"
                );
            }
        }
    }

    #[test]
    fn inset_rect_is_capped_and_top_left_anchored() {
        let r = inset_rect((1600, 900));
        assert!(r[2] <= 180 && r[3] <= 180 && r[3] >= 160 && r[0] < 100 && r[1] < 40, "{r:?}");
        let r = inset_rect((375, 700));
        assert!(r[2] >= 100 && r[2] <= 110 && r[3] <= 90 && r[0] < 40, "{r:?}");
        // clear of the sheet and its button along the bottom
        assert!(r[1] < 40 && r[1] + r[3] + 200 <= 700, "{r:?}");
    }

    #[test]
    fn paraboloid_ellipse_fills_the_inset() {
        let d = doc_with(&["z=x^2+y^2"], "z=7.84", Mode::D3);
        let w = Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 8.0]);
        let p = panel_of(&d, Mode::D3, w, None);
        let (du, dv) = (p.view[1] - p.view[0], p.view[3] - p.view[2]);
        let aspect = p.rect[2] as f64 / p.rect[3] as f64;
        assert!((du / dv - aspect).abs() < 1e-6, "equal scale on both axes");
        assert!(5.6 / du > 0.6, "the r = 2.8 curve fills over 60% of the width: {du}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        // Constant function of the free axes, an item that does not mention the sliced axis,
        // an undefined name, and a complex item.
        let mut d = doc_with(&["y=2", "y=q*x", "c:z^2-1", "x^2+y^2=1"], "z=0", Mode::D3);
        d.items[2].kind = ItemKind::Complex;
        let g = build_scene(&d, Mode::D3, win(), [0.0; 3], (200, 150), &Theme::light());
        assert!(!g.overlay_segments.is_empty());
        let out = build_slice_panel(&d, Mode::D3, win(), (200, 150), &Theme::dark())
            .unwrap()
            .unwrap();
        assert!(out.panel.rect[2] >= 1);
    }

    fn panel_of(d: &Doc, mode: Mode, w: Window3, view: Option<&ViewReq>) -> SlicePanel {
        build_slice_panel_view(d, mode, w, (900, 600), &Theme::light(), view)
            .unwrap()
            .unwrap()
            .panel
    }

    #[test]
    fn hue_field_plane_slice_is_a_2d_field_in_the_inset() {
        let mut d = doc_with(&["sin(x)+y+z"], "z=1", Mode::D3);
        let p = panel_of(&d, Mode::D3, win(), None);
        // Backdrop (opaque domain field, first) then the hue field; no flat backdrop quad.
        assert_eq!(p.geometry.fields.len(), 2);
        assert_eq!(p.geometry.fields[1].kind, FieldKind::Hue);
        assert!(p.geometry.flat_indices.is_empty());
        let w = &p.geometry.fields[1].wgsl;
        assert!(
            w.contains("field_core") && w.contains("fp.params"),
            "constants are uniforms: {w}"
        );
        // Sweeping the slice constant changes the uniform, not the shader (no recompile).
        d.slice = Some(SliceCfg::new(None, parse_fixed_text("z=a").unwrap(), Mode::D3).unwrap());
        slider(&mut d, "a", 1.0);
        let f1 = panel_of(&d, Mode::D3, win(), None)
            .geometry
            .fields
            .remove(1);
        slider(&mut d, "a", 2.5);
        let f2 = panel_of(&d, Mode::D3, win(), None)
            .geometry
            .fields
            .remove(1);
        assert_eq!(f1.wgsl, f2.wgsl);
        assert_ne!(f1.params, f2.params);
        assert!(f2.params.contains(&2.5));
        // The field is NOT clutter in the main scene's slice overlay (no zero contour for it).
        let mut diags = Vec::new();
        let (items, defs, _, _) = prepare(&d, &mut diags);
        let rs =
            ResolvedSlice::resolve(d.slice.as_ref().unwrap(), Mode::D3, &defs, Angle::Rad).unwrap();
        let si = collect(&items, &defs, &Theme::light(), Angle::Rad, &rs, &win());
        assert!(si[0].f.is_none() && si[0].field.is_some());
    }

    #[test]
    fn rotated_plane_axes_keep_x_y_in_the_shader() {
        // y = 1 leaves (x, z) free: z must become the shader's second variable.
        let d = doc_with(&["x+10*z"], "y=1", Mode::D3);
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.geometry.fields[1].kind, FieldKind::Hue);
        assert_eq!(p.axes, ["x".to_string(), "z".to_string()]);
    }

    #[test]
    fn hue_field_line_slice_plots_the_value_along_the_line() {
        let d = doc_with(&["sin(x)+y+z"], "y=1, z=0.5", Mode::D3);
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.axes[1], "f");
        assert!(p.geometry.fields.is_empty());
        let curve = p
            .geometry
            .segments
            .iter()
            .filter(|s| s.width == INSET_CURVE_W)
            .count();
        assert!(curve > 50, "value graph drawn: {curve}");
        // sin(x)+1.5 lies in [0.5, 2.5]; the auto range includes it (and zero).
        assert!(p.view[2] <= 0.5 && p.view[3] >= 2.5, "{:?}", p.view);
        // 2D, line y = a: field x*y -> value a*x.
        let mut d = doc_with(&["x*y"], "y=a", Mode::D2);
        slider(&mut d, "a", 2.0);
        let p = panel_of(
            &d,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            None,
        );
        assert!(p.geometry.segments.iter().any(|s| s.width == INSET_CURVE_W));
    }

    #[test]
    fn inequality_region_plane_slice_fills_and_outlines() {
        let d = doc_with(&["x^2+y^2<4"], "z=1", Mode::D3);
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.geometry.fields.len(), 2);
        assert_eq!(
            p.geometry.fields[1].kind,
            FieldKind::Fill { greater: false }
        );
        assert_eq!(p.curves, 1, "the boundary outline is cut too");
        let (_, _, geo) = geo_of(&d, Mode::D3);
        for s in &geo.curves[0] {
            assert!((s[0][0].hypot(s[0][1]) - 2.0).abs() < 1e-5);
        }
        // `>` fills the outside.
        let d = doc_with(&["x^2+y^2>4"], "z=1", Mode::D3);
        assert_eq!(
            panel_of(&d, Mode::D3, win(), None).geometry.fields[1].kind,
            FieldKind::Fill { greater: true }
        );
        // The main scene still draws the outline on the plane and no fill field of its own.
        let g = build_scene(&d, Mode::D3, win(), [0.0; 3], (900, 600), &Theme::light());
        assert!(!g.overlay_segments.is_empty());
    }

    #[test]
    fn inequality_line_slice_highlights_intervals() {
        // Sphere interior along y=0, z=0 is x in (-2, 2).
        let d = doc_with(&["x^2+y^2+z^2<4"], "y=0, z=0", Mode::D3);
        let (_, _, geo) = geo_of(&d, Mode::D3);
        assert_eq!(geo.intervals[0].len(), 1);
        let iv = geo.intervals[0][0];
        assert!(
            (iv[0] + 2.0).abs() < 1e-6 && (iv[1] - 2.0).abs() < 1e-6,
            "{iv:?}"
        );
        // `>` gives the two outer pieces within the window [-3, 3].
        let d = doc_with(&["x^2+y^2+z^2>4"], "y=0, z=0", Mode::D3);
        let (_, _, geo) = geo_of(&d, Mode::D3);
        assert_eq!(geo.intervals[0].len(), 2);
        // 2D: y < x^2 cut at y = 4 holds for |x| > 2.
        let d = doc_with(&["y<x^2"], "y=4", Mode::D2);
        let (_, _, geo) = geo_of(&d, Mode::D2);
        assert_eq!(geo.intervals[0].len(), 2);
        // Main scene: the highlight segments are on the line; the inset has them on the zero line.
        let g = build_scene(
            &d,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            [0.0; 3],
            (900, 600),
            &Theme::light(),
        );
        assert_eq!(
            g.segments.iter().filter(|s| s.width == INTERVAL_W).count(),
            2
        );
        let p = panel_of(
            &d,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            None,
        );
        assert_eq!(
            p.geometry
                .segments
                .iter()
                .filter(|s| s.width == INTERVAL_W)
                .count(),
            2
        );
    }

    #[test]
    fn points_near_the_plane_show_with_a_span_proportional_tolerance() {
        let d = doc_with(
            &["(1,2,1.05)", "(1,2,1.5)", "[(0,0,1),(1,1,3),(2,-1,0.97)]"],
            "z=1",
            Mode::D3,
        );
        let (_, items, _) = geo_of(&d, Mode::D3);
        // Window span 6 -> tolerance 0.06.
        assert_eq!(
            items[0].points,
            vec![[1.0, 2.0, 1.0]],
            "snapped onto the plane"
        );
        assert!(items[1].points.is_empty());
        assert_eq!(items[2].points.len(), 2);
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.points, 3);
        let dots = p
            .geometry
            .overlay_segments
            .iter()
            .filter(|s| s.p0 == s.p1)
            .count();
        assert_eq!(dots, 3 * 3, "three dots, three passes each");
        // Zooming the main window in tightens the tolerance: 0.05 off no longer counts.
        let tight = Window3::new([-0.5; 3], [0.5; 3]);
        let mut diags = Vec::new();
        let (its, defs, _, _) = prepare(&d, &mut diags);
        let rs =
            ResolvedSlice::resolve(d.slice.as_ref().unwrap(), Mode::D3, &defs, Angle::Rad).unwrap();
        let si = collect(&its, &defs, &Theme::light(), Angle::Rad, &rs, &tight);
        assert!(si[0].points.is_empty(), "|1.05 - 1| > 0.01 * 1");
        assert_eq!(si[2].points.len(), 1, "(0,0,1) is exactly on it");
    }

    #[test]
    fn points_near_a_line_slice_and_in_2d() {
        let d = doc_with(
            &["(1,1,0.5)", "(1,1.5,0.5)", "[(2,1.01,0.5),(3,1,0.5)]"],
            "y=1, z=0.5",
            Mode::D3,
        );
        let (_, items, _) = geo_of(&d, Mode::D3);
        assert_eq!(items[0].points, vec![[1.0, 1.0, 0.5]]);
        assert!(items[1].points.is_empty());
        assert_eq!(items[2].points.len(), 2);
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.points, 3);
        // 2D: only 2-tuples; a 3-tuple is skipped as in the main scene.
        let d = doc_with(&["(1,2)", "(1,2,3)", "[(0,2.01),(0,5)]"], "y=2", Mode::D2);
        let (_, items, _) = geo_of(&d, Mode::D2);
        assert_eq!(items[0].points.len(), 1);
        assert!(items[1].points.is_empty());
        assert_eq!(items[2].points.len(), 1);
    }

    #[test]
    fn complex_item_is_domain_coloured_in_the_z_plane_slice() {
        let mut d = doc_with(&["z^2-1"], "z=1", Mode::D3);
        d.items[0].kind = ItemKind::Complex;
        let p = panel_of(&d, Mode::D3, win(), None);
        assert_eq!(p.geometry.fields.len(), 2);
        assert_eq!(p.geometry.fields[1].kind, FieldKind::Domain);
        assert!(p.geometry.fields[1].wgsl.contains("fn field_color"));
        // Other orientations and line slices skip it without a panic.
        let mut d = doc_with(&["z^2-1"], "y=1", Mode::D3);
        d.items[0].kind = ItemKind::Complex;
        assert_eq!(panel_of(&d, Mode::D3, win(), None).geometry.fields.len(), 0);
    }

    #[test]
    fn explicit_view_resamples_the_inset_geometry() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "z=1", Mode::D3);
        let auto = panel_of(&d, Mode::D3, win(), None);
        assert!(auto.follow);
        // Zoomed onto a small part of the ring: still on the circle, more detail per unit.
        let req = ViewReq {
            free: vec![0, 1],
            view: [1.0, 1.5, 0.8, 1.2],
        };
        let p = panel_of(&d, Mode::D3, win(), Some(&req));
        assert!(!p.follow);
        assert_eq!(p.view[0], 1.0);
        assert_eq!(p.view[1], 1.5);
        for s in p
            .geometry
            .segments
            .iter()
            .filter(|s| s.width == INSET_CURVE_W)
        {
            let a = abs(&p, s.p0);
            assert!((a[0].hypot(a[1]) - 3f64.sqrt()).abs() < 1e-4);
            assert!(
                a[0] > 0.9 && a[0] < 1.6,
                "only the visible part is sampled: {a:?}"
            );
        }
        // A request for other free axes is ignored (back to following).
        let other = ViewReq {
            free: vec![0, 2],
            view: [1.0, 1.5, 0.8, 1.2],
        };
        assert!(panel_of(&d, Mode::D3, win(), Some(&other)).follow);
        // An invalid one too.
        let bad = ViewReq {
            free: vec![0, 1],
            view: [1.0, 1.0, 0.0, 1.0],
        };
        assert!(panel_of(&d, Mode::D3, win(), Some(&bad)).follow);
        // 1D slice: the vertical range is the view's.
        let d1 = doc_with(&["y=x^2"], "y=1", Mode::D2);
        let req = ViewReq {
            free: vec![0],
            view: [-1.0, 1.0, -5.0, 5.0],
        };
        let p = panel_of(
            &d1,
            Mode::D2,
            Window3::new([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]),
            Some(&req),
        );
        assert_eq!(p.view, [-1.0, 1.0, -5.0, 5.0]);
    }
    #[test]
    fn sloped_plane_cuts_a_sphere_in_a_true_circle() {
        // z = x passes through the origin: a great circle of radius 2. z = x + 1 is 1/sqrt 2 from
        // it: radius sqrt(4 - 1/2). The inset coordinates are distances in the plane.
        for (plane, r) in [("z = x", 2.0), ("z = x + 1", 3.5f64.sqrt())] {
            let d = doc_with(&["x^2+y^2+z^2=4"], plane, Mode::D3);
            let (rs, _, geo) = geo_of(&d, Mode::D3);
            assert!(rs.plane.is_some() && rs.dim == 2, "{plane}");
            let segs = &geo.curves[0];
            assert!(segs.len() > 100, "{plane}: {}", segs.len());
            // The sphere is centred on the origin, which is the frame origin of both planes.
            let mid = (0.0, 0.0);
            for s in segs {
                for p in s {
                    let rr = (p[0] - mid.0).hypot(p[1] - mid.1);
                    assert!((rr - r).abs() < 5e-3, "{plane}: radius {rr} != {r}");
                }
            }
        }
    }

    #[test]
    fn sloped_plane_collects_points_and_draws_overlay_and_panel() {
        let d = doc_with(&["(1,1,2)", "(1,1,3)", "x^2+y^2+z^2=4"], "z = x + y", Mode::D3);
        let (rs, si, _) = geo_of(&d, Mode::D3);
        assert_eq!(si[0].points.len(), 1, "(1,1,2) is on z = x + y, (1,1,3) is not");
        let q = rs.project(si[0].points[0]);
        assert!((q[0].hypot(q[1]) - rs.plane.unwrap().project([1.0, 1.0, 2.0]).iter().map(|v| v * v).sum::<f64>().sqrt()).abs() < 1e-9);
        // Main-scene overlay: every plane vertex lies on the plane.
        let g = build_scene(&d, Mode::D3, win(), [0.0; 3], (900, 600), &Theme::light());
        assert!(!g.overlay_segments.is_empty());
        // The inset builds with u / v axis names.
        let out = build_slice_panel(&d, Mode::D3, win(), (900, 600), &Theme::light())
            .expect("has a slice")
            .expect("resolves");
        assert_eq!(out.panel.axes, ["u".to_string(), "v".to_string()]);
        assert!(out.panel.curves >= 1 && out.panel.points == 1);
    }

    #[test]
    fn sloped_plane_misses_window_without_panicking() {
        let d = doc_with(&["x^2+y^2+z^2=4"], "z = x + 100", Mode::D3);
        let (_, si, geo) = geo_of(&d, Mode::D3);
        assert!(si.len() == 1 && geo.curves[0].is_empty());
        let _ = build_scene(&d, Mode::D3, win(), [0.0; 3], (900, 600), &Theme::light());
    }

}
