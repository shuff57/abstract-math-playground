//! Dragging the selected curve `y = f(x)` to change its parameters (a child module of `app`, so
//! it works on `App`'s private state). The numerics live in `math_core::drag`.

use super::*;
use crate::scene::{drag_curve, DragCurve, DragShape, ParamSrc};
use math_core::compile::Program;
use math_core::drag::{gradient, project_to_zero, snap_to_steps, solve_with, SHAPE_PX, SHAPE_W, UNDEFINED_PX};

/// Points of the curve at the press, in display coordinates (the first is the grabbed one): the
/// shape the drag keeps while it moves the curve.
enum Anchor {
    /// `y = f(x)`: `(X, Y)`; the curve is sampled at the x of the moved point.
    Explicit(Vec<[f64; 2]>),
    /// Parametric and polar: `(t, (X, Y))`; the same `t` must land on the moved point.
    Param(Vec<(f64, [f64; 2])>),
    /// `F(x, y) = 0`: `((X, Y), |grad F|)` per display unit; `F` at the moved point must be 0.
    Implicit(Vec<([f64; 2], f64)>),
}

/// A curve drag in progress.
pub struct CurveGrab {
    curve: DragCurve,
    anchor: Anchor,
    /// Parameter values at the press (what Escape restores), and the text of numeric definitions.
    start: Vec<f64>,
    start_latex: Vec<Option<String>>,
    /// The continuous solution of the last move (the warm start of the next). Slider steps are
    /// applied only to what is written, so quantizing never accumulates error.
    cur: Vec<f64>,
    /// The values last written to the document.
    last: Vec<f64>,
    /// The pointer's display position at the press.
    press: [f64; 2],
    changed: bool,
}

/// Reusable evaluation buffers.
struct Eval {
    buf: Vec<f64>,
    st: Vec<f64>,
}

impl Eval {
    fn new() -> Self {
        Eval { buf: Vec::with_capacity(16), st: Vec::with_capacity(16) }
    }

    fn at(&mut self, prog: &Program, vars: &[f64], p: &[f64]) -> f64 {
        self.buf.clear();
        self.buf.extend_from_slice(vars);
        self.buf.extend_from_slice(p);
        prog.eval_with(&self.buf, &mut self.st)
    }
}

/// Pushes residual `v` of sample `k` (a sample the curve is undefined at costs a fixed amount,
/// but the grabbed point, `k == 0`, must exist).
fn put(out: &mut Vec<f64>, k: usize, v: f64) -> Option<()> {
    if v.is_finite() {
        out.push(v);
    } else if k == 0 {
        return None;
    } else {
        out.push(SHAPE_W * UNDEFINED_PX);
    }
    Some(())
}

/// The weighted pixel residuals of the curve with parameters `p` against the anchor moved by
/// `d` (display units); `u` is display units per pixel.
fn residual(
    curve: &DragCurve,
    anchor: &Anchor,
    d: [f64; 2],
    u: [f64; 2],
    map: AxisMap,
    ev: &mut Eval,
    p: &[f64],
    out: &mut Vec<f64>,
) -> Option<()> {
    out.clear();
    let w = |k: usize| if k == 0 { 1.0 } else { SHAPE_W };
    match (&curve.shape, anchor) {
        (DragShape::Explicit(prog), Anchor::Explicit(s)) => {
            for (k, q) in s.iter().enumerate() {
                let y = map.fwd(1, ev.at(prog, &[map.inv(0, q[0] + d[0])], p));
                put(out, k, w(k) * (y - (q[1] + d[1])) / u[1])?;
            }
        }
        (DragShape::Parametric { px, py, .. }, Anchor::Param(s)) => {
            for (k, (t, q)) in s.iter().enumerate() {
                let x = map.fwd(0, ev.at(px, &[*t], p));
                let y = map.fwd(1, ev.at(py, &[*t], p));
                put(out, k, w(k) * (x - (q[0] + d[0])) / u[0])?;
                put(out, k, w(k) * (y - (q[1] + d[1])) / u[1])?;
            }
        }
        (DragShape::Implicit(prog), Anchor::Implicit(s)) => {
            let ubar = 0.5 * (u[0] + u[1]);
            for (k, (q, gn)) in s.iter().enumerate() {
                let wld = map.inv3([q[0] + d[0], q[1] + d[1], 0.0]);
                let f = ev.at(prog, &[wld[0], wld[1]], p);
                put(out, k, w(k) * f / gn / ubar)?;
            }
        }
        _ => return None,
    }
    Some(())
}

impl App {
    fn world_at(&self, x: f64, y: f64, vp: (f64, f64)) -> [f64; 2] {
        let w = self.active_map().inv3(self.rig.pixel_to_world((x, y), vp));
        [w[0], w[1]]
    }

    /// Display units per pixel at the pointer.
    fn display_units(&self, x: f64, y: f64, vp: (f64, f64)) -> [f64; 2] {
        let a = self.rig.pixel_to_world((x, y), vp);
        let bx = self.rig.pixel_to_world((x + 1.0, y), vp);
        let by = self.rig.pixel_to_world((x, y + 1.0), vp);
        [(bx[0] - a[0]).abs().max(1e-300), (by[1] - a[1]).abs().max(1e-300)]
    }

    /// The curve's shape at the press, around the display point `h` nearest the pointer; `None`
    /// when the curve is undefined there.
    fn grab_anchor(curve: &DragCurve, start: &[f64], h: [f64; 2], u: [f64; 2], map: AxisMap) -> Option<Anchor> {
        let mut ev = Eval::new();
        match &curve.shape {
            DragShape::Explicit(prog) => {
                let mut s = Vec::new();
                for (k, o) in std::iter::once(0.0).chain(SHAPE_PX).enumerate() {
                    let xd = h[0] + o * u[0];
                    let y = map.fwd(1, ev.at(prog, &[map.inv(0, xd)], start));
                    if y.is_finite() {
                        s.push([xd, y]);
                    } else if k == 0 {
                        return None;
                    }
                }
                Some(Anchor::Explicit(s))
            }
            DragShape::Parametric { px, py, t_end } => {
                const N: usize = 3000;
                let mut pt = |t: f64| -> Option<[f64; 2]> {
                    let a = map.fwd(0, ev.at(px, &[t], start));
                    let b = map.fwd(1, ev.at(py, &[t], start));
                    (a.is_finite() && b.is_finite()).then_some([a, b])
                };
                let dist = |q: [f64; 2], c: [f64; 2]| ((q[0] - c[0]) / u[0]).hypot((q[1] - c[1]) / u[1]);
                let ts: Vec<f64> = (0..=N).map(|i| t_end * i as f64 / N as f64).collect();
                let pts: Vec<Option<[f64; 2]>> = ts.iter().map(|&t| pt(t)).collect();
                let key = |i: usize| pts[i].map_or(f64::INFINITY, |q| dist(q, h));
                let ia = (0..=N)
                    .filter(|&i| pts[i].is_some())
                    .min_by(|&i, &j| key(i).total_cmp(&key(j)))?;
                // Refine inside the neighbouring intervals.
                let (lo, hi) = (ts[ia.saturating_sub(1)], ts[(ia + 1).min(N)]);
                let (mut ta, mut pa, mut best) = (ts[ia], pts[ia]?, dist(pts[ia]?, h));
                for i in 0..=60 {
                    let t = lo + (hi - lo) * i as f64 / 60.0;
                    if let Some(q) = pt(t) {
                        if dist(q, h) < best {
                            (ta, pa, best) = (t, q, dist(q, h));
                        }
                    }
                }
                let mut s = vec![(ta, pa)];
                for dir in [-1i64, 1] {
                    let mut targets = [-SHAPE_PX[0], -SHAPE_PX[1]].into_iter().peekable();
                    let (mut acc, mut prev, mut i) = (0.0, pa, ia as i64 + if dir > 0 { 1 } else { 0 });
                    if dir > 0 {
                        targets = [SHAPE_PX[2], SHAPE_PX[3]].into_iter().peekable();
                    }
                    while (0..=N as i64).contains(&i) {
                        let Some(q) = pts[i as usize] else { break };
                        acc += dist(q, prev);
                        prev = q;
                        while targets.peek().is_some_and(|t| *t <= acc) {
                            targets.next();
                            s.push((ts[i as usize], q));
                        }
                        i += dir;
                    }
                }
                Some(Anchor::Param(s))
            }
            DragShape::Implicit(prog) => {
                let mut f = |a: f64, b: f64| {
                    let w = map.inv3([a, b, 0.0]);
                    ev.at(prog, &[w[0], w[1]], start)
                };
                let a0 = project_to_zero(&mut f, h, u)?;
                let g0 = gradient(&mut f, a0, u);
                let norm = |g: [f64; 2]| g[0].hypot(g[1]);
                if !(norm(g0) > 0.0 && norm(g0).is_finite()) {
                    return None;
                }
                // Along the tangent (measured in pixels), back onto the curve.
                let gp = [g0[0] * u[0], g0[1] * u[1]];
                let t = [-gp[1] / norm(gp), gp[0] / norm(gp)];
                let mut s = vec![(a0, norm(g0))];
                for o in SHAPE_PX {
                    let q = [a0[0] + o * t[0] * u[0], a0[1] + o * t[1] * u[1]];
                    if let Some(q) = project_to_zero(&mut f, q, u) {
                        let g = gradient(&mut f, q, u);
                        if norm(g) > 0.0 && norm(g).is_finite() {
                            s.push((q, norm(g)));
                        }
                    }
                }
                Some(Anchor::Implicit(s))
            }
        }
    }

    /// The parameters a drag of curve `id` would change: empty unless `id` is the selected curve.
    pub(super) fn grab_params(&mut self, id: &str) -> Vec<String> {
        if self.selected.as_deref() != Some(id) {
            return Vec::new();
        }
        if let Some((rev, i, names)) = &self.grab_probe {
            if *rev == self.build_rev && i == id {
                return names.clone();
            }
        }
        let names: Vec<String> = drag_curve(&self.doc, id)
            .map(|c| c.params.into_iter().map(|p| p.name).collect())
            .unwrap_or_default();
        self.grab_probe = Some((self.build_rev, id.to_string(), names.clone()));
        names
    }

    /// A press on the selected curve, when it has parameters to drag.
    pub(super) fn grab_curve(&mut self, x: f64, y: f64, vp: (f64, f64)) -> Option<CurveGrab> {
        let sel = self.selected.clone()?;
        let (id, wx, wy, ..) = self.curve_under(x, y, vp)?;
        if id != sel {
            return None;
        }
        let curve = drag_curve(&self.doc, &id)?;
        let start: Vec<f64> = curve.params.iter().map(|p| p.cfg.value).collect();
        let start_latex = curve
            .params
            .iter()
            .map(|p| match &p.src {
                ParamSrc::Def { item } => self.doc.items.iter().find(|i| i.id == *item).map(|i| i.latex.clone()),
                ParamSrc::Slider => None,
            })
            .collect();
        let map = self.active_map();
        let h = map.fwd3([wx, wy, 0.0]);
        let u = self.display_units(x, y, vp);
        let anchor = Self::grab_anchor(&curve, &start, [h[0], h[1]], u, map)?;
        let p = self.rig.pixel_to_world((x, y), vp);
        let g = CurveGrab {
            curve,
            anchor,
            cur: start.clone(),
            last: start.clone(),
            start,
            start_latex,
            press: [p[0], p[1]],
            changed: false,
        };
        self.emit_curve_drag(&g, true, false, x, y);
        Some(g)
    }

    fn emit_curve_drag(&mut self, g: &CurveGrab, active: bool, cancelled: bool, px: f64, py: f64) {
        let vals = if cancelled { &g.start } else { &g.last };
        self.outbox.push(Event::CurveDrag {
            item: g.curve.id.clone(),
            params: g
                .curve
                .params
                .iter()
                .zip(vals)
                .map(|(p, v)| ParamValue { name: p.name.clone(), value: *v })
                .collect(),
            active,
            cancelled,
            px,
            py,
        });
    }

    /// One pointer move of a curve drag: solves for the parameters that carry the grabbed point
    /// to the pointer and writes the ones that changed.
    pub(super) fn drag_curve(&mut self, x: f64, y: f64, vp: (f64, f64)) {
        let Some(mut g) = self.drag.as_mut().and_then(|d| d.curve.take()) else { return };
        // The target is the grabbed point moved by what the pointer moved in display (pixel)
        // space, so on logarithmic axes the curve stays under the pointer.
        let here = self.rig.pixel_to_world((x, y), vp);
        let d = [here[0] - g.press[0], here[1] - g.press[1]];
        let du = self.display_units(x, y, vp);
        let map = self.active_map();
        let w = self.world_at(x, y, vp);
        let wx_per_px = (self.world_at(x + 1.0, y, vp)[0] - w[0]).abs();
        let cfgs: Vec<_> = g.curve.params.iter().map(|p| p.cfg.clone()).collect();
        let mut ev = Eval::new();
        let solved = solve_with(&cfgs, &g.cur, |p, out| residual(&g.curve, &g.anchor, d, du, map, &mut ev, p, out));
        if let Some(cur) = solved {
            // Sliders land on the step grid value that fits best; `cur` stays continuous.
            let cur_snapped = snap_to_steps(&cfgs, &cur, |p, out| residual(&g.curve, &g.anchor, d, du, map, &mut ev, p, out));
            // Free numbers are rounded to what a pixel resolves, like a dragged point.
            let dec = (-wx_per_px.log10().floor()).clamp(0.0, 12.0) as usize;
            let k = 10f64.powi(dec as i32);
            let next: Vec<f64> = cur_snapped
                .iter()
                .zip(&g.curve.params)
                .map(|(v, p)| match p.src {
                    ParamSrc::Slider => *v,
                    ParamSrc::Def { .. } => (v * k).round() / k,
                })
                .collect();
            g.cur = cur;
            if next != g.last {
                self.write_params(&g.curve, &next, &g.last, dec);
                g.last = next;
                g.changed = true;
                self.mark_doc_changed();
                self.emit_curve_drag(&g, true, false, x, y);
            }
        }
        if let Some(d) = self.drag.as_mut() {
            d.curve = Some(g);
        }
    }

    /// Writes `values` into the document for the parameters that differ from `old`.
    fn write_params(&mut self, c: &DragCurve, values: &[f64], old: &[f64], decimals: usize) {
        for ((p, v), o) in c.params.iter().zip(values).zip(old) {
            if v == o {
                continue;
            }
            match &p.src {
                ParamSrc::Slider => {
                    if let Some(cfg) = self.doc.sliders.get_mut(&p.name) {
                        cfg.value = *v;
                        self.outbox.push(Event::SliderValue { name: p.name.clone(), value: *v });
                    }
                }
                ParamSrc::Def { item } => {
                    let latex = format!("{}={}", p.name, fmt_coord(*v, decimals));
                    self.set_item_text(item, latex);
                }
            }
        }
    }

    fn set_item_text(&mut self, id: &str, latex: String) {
        if let Some(it) = self.doc.items.iter_mut().find(|i| i.id == id) {
            if it.latex != latex {
                it.latex = latex.clone();
                self.outbox.push(Event::ItemEdited { id: id.to_string(), latex });
            }
        }
    }

    /// The release (or cancellation) of the pointer: closes a curve drag, restoring the starting
    /// values when `cancel` is set.
    pub(super) fn end_curve_drag(&mut self, cancel: bool, x: f64, y: f64) {
        let Some(g) = self.drag.as_mut().and_then(|d| d.curve.take()) else { return };
        self.finish_curve_drag(g, cancel, x, y);
    }

    /// Escape: abandons a drag in progress and restores its starting values.
    pub(super) fn cancel_curve_drag(&mut self) {
        let Some(g) = self.drag.as_mut().and_then(|d| d.curve.take()) else { return };
        // The press stays owned by the drag so later moves do not pan.
        self.drag = None;
        self.finish_curve_drag(g, true, 0.0, 0.0);
    }

    fn finish_curve_drag(&mut self, g: CurveGrab, cancel: bool, x: f64, y: f64) {
        let cancelled = cancel && g.changed;
        if cancelled {
            for ((p, v), l) in g.curve.params.iter().zip(&g.start).zip(&g.start_latex) {
                match (&p.src, l) {
                    (ParamSrc::Slider, _) => {
                        if let Some(cfg) = self.doc.sliders.get_mut(&p.name) {
                            if cfg.value != *v {
                                cfg.value = *v;
                                self.outbox.push(Event::SliderValue { name: p.name.clone(), value: *v });
                            }
                        }
                    }
                    (ParamSrc::Def { item }, Some(l)) => self.set_item_text(item, l.clone()),
                    _ => {}
                }
            }
            self.mark_doc_changed();
        }
        self.emit_curve_drag(&g, false, cancelled, x, y);
    }
}
