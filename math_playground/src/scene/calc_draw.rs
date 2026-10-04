//! Calculus and regression wiring for the scene builder (a child module of `scene`).
//!
//! * **CPU field fallback.** `int` / `sum` / `prod` have no WGSL form (`WgslError::BadProgram`),
//!   so a field or inequality that uses one is rasterised on the CPU into flat translucent quads
//!   instead of vanishing: [`Builder::field_cpu`]. Curves, contours and surfaces already run on
//!   the CPU and need no fallback. Complex domain colouring has no `sum`/`int`/`prod` at all and
//!   reports a clear diagnostic.
//! * **Read-outs** ([`collect_infos`]): the simplified symbolic derivative of every `d/dx`/`f'`
//!   call (`kind: "derivative"`) and the numeric value of scalar items such as
//!   `int(x^2,x,0,2)` (`kind: "value"`).
//! * **Regression** ([`fit_regressions`], [`Builder::draw_regression`]): `y_1 ~ a x_1 + b` is fitted
//!   against the document's lists and table columns. RULE: the parameters (names that no list,
//!   slider, definition or table column defines) are DEFINED AS NUMBERS WHILE THE REGRESSION ITEM
//!   IS VISIBLE, so any other item can use `a` and `b`; hiding or deleting the item undefines
//!   them. Two visible regressions that share a parameter name are fitted independently and the
//!   later one in document order supplies the value other items see. A name that is already a
//!   slider or definition is a fixed constant, not a parameter.

use std::collections::BTreeMap;

use math_core::analyze::Kind;
use math_core::ast::Expr;
use math_core::list::{eval_value, Bindings, Value};
use math_core::mesh;
use math_core::print::{to_latex, to_text};
use math_core::regress::{self, FitResult};
use math_core::resolve::Defs;
use math_core::view::Mode;

use super::{format_value, Builder, Prepared, Style};
use crate::geometry::{FieldKind, InfoParam, ItemInfo};

/// Time budget (seconds) for one CPU field raster.
const CPU_FIELD_BUDGET_S: f64 = 0.05;
/// Smallest and largest number of cells a CPU field raster may use.
const CPU_FIELD_MIN_CELLS: f64 = 1500.0;
const CPU_FIELD_MAX_CELLS: f64 = 250_000.0;
/// Opacity of the inequality fill (matches `field.wgsl`).
const FILL_ALPHA: f32 = 0.22;
/// Opacity of the hue field (matches `field.wgsl`).
const HUE_ALPHA: f32 = 0.9;
/// Residual tick width in physical pixels.
const RESIDUAL_W: f32 = 1.5;

fn children(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Num(_) | Expr::Var(_) => Vec::new(),
        Expr::Neg(a) => vec![a],
        Expr::Bin(_, a, b) | Expr::Rel(_, a, b) => vec![a, b],
        Expr::Call(_, args) => args.iter().collect(),
        Expr::Tuple(v) | Expr::List(v) => v.iter().collect(),
    }
}

/// True if the expression calls `int`, `sum` or `prod` (CPU only).
pub(super) fn uses_reduce(e: &Expr) -> bool {
    match e {
        Expr::Call(n, _) if matches!(n.as_str(), "int" | "sum" | "prod") => true,
        _ => children(e).into_iter().any(uses_reduce),
    }
}

fn collect_derivs<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
    if let Expr::Call(n, a) = e {
        // `f'(x)` parses as the 3-argument form evaluated at the variable itself; the read-out
        // is the symbolic derivative either way.
        if n == "deriv" && (a.len() == 2 || a.len() == 3) {
            out.push(e);
            return;
        }
    }
    for c in children(e) {
        collect_derivs(c, out);
    }
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let f = |k: f32| {
        let p = ((h + k).fract() * 6.0 - 3.0).abs();
        v * (1.0 - s + s * (p - 1.0).clamp(0.0, 1.0))
    };
    [f(1.0), f(2.0 / 3.0), f(1.0 / 3.0)]
}

/// Rounds to 4 significant digits (for displayed equations).
fn sig4(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let mag = 10f64.powi(3 - v.abs().log10().floor() as i32);
    (v * mag).round() / mag
}

impl<'a> Builder<'a> {
    /// CPU raster of a scalar field `e(x, y)` (all definitions and sliders already folded in)
    /// into flat quads over the window, at reduced resolution so a slow `int` stays interactive.
    /// Mirrors `field.wgsl`: the hue ramp for [`FieldKind::Hue`], translucent fill runs for
    /// [`FieldKind::Fill`].
    pub(super) fn field_cpu(&mut self, e: &Expr, kind: FieldKind, color: [f32; 4]) -> Result<(), String> {
        let p = self.prog(e, &["x", "y"])?;
        let (vw, vh) = self.px();
        let (lo, hi) = (self.win.min, self.win.max);
        let mut stack = Vec::with_capacity(16);
        let t0 = instant::Instant::now();
        for k in 0..24 {
            let x = lo[0] + (hi[0] - lo[0]) * (k as f64 + 0.5) / 24.0;
            p.eval_with(&[x, 0.5 * (lo[1] + hi[1])], &mut stack);
        }
        let per = (t0.elapsed().as_secs_f64() / 24.0).max(1e-9);
        let cells = (CPU_FIELD_BUDGET_S / per).clamp(CPU_FIELD_MIN_CELLS, CPU_FIELD_MAX_CELLS);
        let base: f64 = if matches!(kind, FieldKind::Fill { .. }) { 3.0 } else { 6.0 };
        let px = base.max((vw * vh / cells).sqrt());
        let nx = ((vw / px).ceil() as usize).clamp(1, 700);
        let ny = ((vh / px).ceil() as usize).clamp(1, 700);
        let (dx, dy) = ((hi[0] - lo[0]) / nx as f64, (hi[1] - lo[1]) / ny as f64);
        let mut v = vec![f64::NAN; nx * ny];
        for j in 0..ny {
            let y = lo[1] + dy * (j as f64 + 0.5);
            for i in 0..nx {
                v[j * nx + i] = p.eval_with(&[lo[0] + dx * (i as f64 + 0.5), y], &mut stack);
            }
        }
        match kind {
            FieldKind::Fill { greater } => {
                let col = [color[0], color[1], color[2], FILL_ALPHA * color[3]];
                for j in 0..ny {
                    let (y0, y1) = (lo[1] + dy * j as f64, lo[1] + dy * (j + 1) as f64);
                    let mut i = 0;
                    while i < nx {
                        let inside = |i: usize| {
                            let f = v[j * nx + i];
                            f.is_finite() && if greater { f > 0.0 } else { f < 0.0 }
                        };
                        if !inside(i) {
                            i += 1;
                            continue;
                        }
                        let start = i;
                        while i < nx && inside(i) {
                            i += 1;
                        }
                        self.quad(lo[0] + dx * start as f64, lo[0] + dx * i as f64, y0, y1, col);
                    }
                }
            }
            _ => {
                let u = |f: f64| f.signum() * (1.0 + f.abs()).ln() * 0.5;
                for j in 0..ny {
                    let (y0, y1) = (lo[1] + dy * j as f64, lo[1] + dy * (j + 1) as f64);
                    for i in 0..nx {
                        let f = v[j * nx + i];
                        if !f.is_finite() {
                            continue;
                        }
                        // Per-pixel change of the ramp coordinate (desaturates steep regions).
                        let nb = |ii: usize, jj: usize| {
                            let g = v[jj * nx + ii];
                            if g.is_finite() {
                                (u(g) - u(f)).abs()
                            } else {
                                0.0
                            }
                        };
                        let gu = nb((i + 1).min(nx - 1), j).max(nb(i, (j + 1).min(ny - 1))) / px;
                        let sat = (0.85 / (1.0 + 8.0 * gu)) as f32;
                        let rgb = hsv2rgb((u(f) + 0.62).rem_euclid(1.0) as f32, sat, 0.96);
                        let col = [rgb[0], rgb[1], rgb[2], HUE_ALPHA * color[3]];
                        self.quad(lo[0] + dx * i as f64, lo[0] + dx * (i + 1) as f64, y0, y1, col);
                    }
                }
            }
        }
        Ok(())
    }

    /// Draws a fitted regression: the curve (2D, and on the z = 0 plane in 3D) and, when the
    /// item asks for it, residual ticks from each data point to the curve.
    pub(super) fn draw_regression(
        &mut self,
        pr: &Prepared,
        fit: &FitResult,
        defs: &Defs,
        mode: Mode,
        st: Style,
    ) -> Result<(), String> {
        if mode == Mode::D1 {
            return Ok(());
        }
        let Some(curve) = fit.curve_expr("x") else {
            return Ok(()); // several data lists: no single curve to draw
        };
        let r = defs.resolve(&curve).map_err(|e| e.to_string())?;
        let prog = self.prog(&r, &["x"])?;
        let w = self.win;
        let lines = mesh::sample_explicit(&prog, w.min[0], w.max[0], self.vw.max(1.0) as usize, (w.min[1], w.max[1]));
        self.add_lines2(&lines, st);
        if pr.item.style.residuals {
            let xs = fit.data_vars.first().and_then(|n| {
                let body = defs.resolve(&Expr::var(n)).ok()?;
                match eval_value(&body, &Bindings::new().with_angle(self.angle)).ok()? {
                    Value::List(l) => Some(l),
                    _ => None,
                }
            });
            if let Some(xs) = xs {
                let col = [st.color[0], st.color[1], st.color[2], st.color[3] * 0.7];
                for ((x, f), e) in xs.iter().zip(&fit.fitted).zip(&fit.residuals) {
                    self.seg([*x, *f, 0.0], [*x, *f + *e, 0.0], RESIDUAL_W, col);
                }
            }
        }
        Ok(())
    }
}

/// Fits every visible regression item (document order) and defines the fitted parameters in
/// `defs` and `pdefs` (see the module docs for the rule). Failures become diagnostics.
pub(super) fn fit_regressions(
    items: &[Prepared],
    defs: &mut Defs,
    pdefs: &mut Defs,
    diags: &mut Vec<(String, String)>,
) -> BTreeMap<String, FitResult> {
    let base = defs.clone();
    let mut fits = BTreeMap::new();
    for pr in items.iter().filter(|p| !p.item.hidden) {
        let Kind::Regression { lhs, model } = &pr.kind else { continue };
        let params: Vec<String> = regress::infer_params(lhs, model, &base)
            .into_iter()
            .filter(|n| !matches!(n.as_str(), "x" | "y" | "z"))
            .collect();
        match regress::fit_regression(&pr.expr, &base, Some(&params)) {
            Ok(f) => {
                if f.params.iter().any(|(_, v)| !v.is_finite()) {
                    diags.push((pr.item.id.clone(), "regression: the fit did not produce finite parameters".into()));
                    continue;
                }
                for (n, v) in &f.params {
                    defs.set_slider(n, *v);
                    pdefs.define_var(n, Expr::Num(*v));
                }
                fits.insert(pr.item.id.clone(), f);
            }
            Err(e) => diags.push((pr.item.id.clone(), format!("regression: {e}"))),
        }
    }
    fits
}

/// Rewrites `a + (-c)` as `a - c` (and `a - (-c)` as `a + c`) for numeric `c`, so a fitted negative
/// coefficient reads `... - 0.03333` rather than `... + -0.03333`.
fn tidy_signs(e: Expr) -> Expr {
    use math_core::ast::BinOp;
    match e {
        Expr::Bin(op, a, b) => {
            let a = tidy_signs(*a);
            let b = tidy_signs(*b);
            match (op, &b) {
                (BinOp::Add, Expr::Num(c)) if *c < 0.0 => Expr::bin(BinOp::Sub, a, Expr::Num(-*c)),
                (BinOp::Sub, Expr::Num(c)) if *c < 0.0 => Expr::bin(BinOp::Add, a, Expr::Num(-*c)),
                _ => Expr::bin(op, a, b),
            }
        }
        Expr::Neg(a) => Expr::Neg(Box::new(tidy_signs(*a))),
        Expr::Call(n, args) => Expr::Call(n, args.into_iter().map(tidy_signs).collect()),
        other => other,
    }
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

fn regression_info(id: &str, lhs: &Expr, fit: &FitResult) -> ItemInfo {
    let mut model = fit.model.clone();
    for (n, v) in &fit.params {
        model = model.subst(n, &Expr::Num(sig4(*v)));
    }
    let model = tidy_signs(model);
    let plain: Vec<String> = fit.params.iter().map(|(n, v)| format!("{n} = {}", format_value(sig4(*v)))).collect();
    let mut text = format!("{} ~ {}; {}", to_text(lhs), to_text(&model), plain.join(", "));
    if fit.r2.is_finite() {
        text.push_str(&format!("; r2 = {}", format_value(sig4(fit.r2))));
    }
    text.push_str(&format!("; rmse = {}", format_value(sig4(fit.rmse))));
    let se = fit.std_errors.as_deref();
    ItemInfo {
        id: id.to_string(),
        kind: "regression".into(),
        latex: Some(format!("{} \\approx {}", to_latex(lhs), to_latex(&model))),
        text: Some(text),
        value: None,
        params: fit
            .params
            .iter()
            .enumerate()
            .map(|(i, (n, v))| InfoParam {
                name: n.clone(),
                value: *v,
                std_error: se.and_then(|s| s.get(i).copied()).and_then(finite),
            })
            .collect(),
        r2: finite(fit.r2),
        rmse: finite(fit.rmse),
        n: Some(fit.n),
    }
}

/// Read-outs for the shell: regression fits, symbolic derivatives and scalar values.
pub(super) fn collect_infos(
    items: &[Prepared],
    fits: &BTreeMap<String, FitResult>,
    defs: &Defs,
    pdefs: &Defs,
    angle: math_core::compile::Angle,
) -> Vec<ItemInfo> {
    let mut out = Vec::new();
    for pr in items.iter().filter(|p| !p.item.hidden) {
        let id = &pr.item.id;
        if let Kind::Regression { lhs, .. } = &pr.kind {
            if let Some(f) = fits.get(id) {
                out.push(regression_info(id, lhs, f));
            }
            continue;
        }
        if pr.complex.is_some() {
            continue;
        }
        let mut calls = Vec::new();
        collect_derivs(&pr.expr, &mut calls);
        let mut latex = Vec::new();
        let mut text = Vec::new();
        for c in calls {
            // Show the derivative function, not its value at a point.
            let c2;
            let c = match c {
                Expr::Call(n, a) if a.len() == 3 => {
                    c2 = Expr::Call(n.clone(), a[..2].to_vec());
                    &c2
                }
                _ => c,
            };
            let Ok(d) = pdefs.resolve(c) else { continue };
            let (l, t) = (format!("{} = {}", to_latex(c), to_latex(&d)), format!("{} = {}", to_text(c), to_text(&d)));
            if !latex.contains(&l) {
                latex.push(l);
                text.push(t);
            }
        }
        if !latex.is_empty() {
            out.push(ItemInfo {
                id: id.clone(),
                kind: "derivative".into(),
                latex: Some(latex.join(",\\ \\ ")),
                text: Some(text.join("; ")),
                ..Default::default()
            });
        }
        if let Kind::Value { .. } = &pr.kind {
            let Ok(r) = defs.resolve(&pr.expr) else { continue };
            if r.contains_var("x") || r.contains_var("y") || r.contains_var("z") {
                continue;
            }
            if let Ok(Value::Num(v)) = eval_value(&r, &Bindings::new().with_angle(angle)) {
                let shown = format_value(v);
                out.push(ItemInfo {
                    id: id.clone(),
                    kind: "value".into(),
                    latex: Some(format!("{} = {}", to_latex(&pr.expr), if shown.is_empty() { "\\text{undefined}".into() } else { shown.clone() })),
                    text: Some(if shown.is_empty() { "undefined".into() } else { shown }),
                    value: finite(v),
                    ..Default::default()
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn tidy_signs_turns_plus_negative_into_minus() {
        use math_core::ast::BinOp;
        use math_core::ast::Expr;
        use super::tidy_signs;
        let e = Expr::bin(BinOp::Add, Expr::var("x"), Expr::Num(-3.0));
        assert_eq!(math_core::print::to_text(&tidy_signs(e)), "x - 3");
    }

    use crate::geometry::{SceneGeometry, Theme};
    use crate::scene::build_scene;
    use math_core::doc::{Doc, Item, ItemKind};
    use math_core::table::Table;
    use math_core::view::{Mode, Window3};

    fn doc_with(items: &[(&str, &str)]) -> Doc {
        let mut d = Doc::new_default();
        for (id, l) in items {
            d.items.push(Item::new(id, ItemKind::Equation, l));
        }
        d
    }

    fn add_table(d: &mut Doc, cols: &[(&str, &[&str])]) {
        let names: Vec<String> = cols.iter().map(|c| c.0.to_string()).collect();
        let mut t = Table::new(&names, 0);
        for (ci, (_, cells)) in cols.iter().enumerate() {
            for (ri, v) in cells.iter().enumerate() {
                t.set_cell(ri, ci, v).unwrap();
            }
        }
        let mut it = Item::new("t", ItemKind::Table, "");
        it.table = Some(t);
        d.items.push(it);
    }

    fn build(d: &Doc, mode: Mode) -> SceneGeometry {
        build_scene(d, mode, Window3::default(), [0.0; 3], (800, 600), &Theme::light())
    }

    fn base_len(mode: Mode) -> usize {
        build(&Doc::new_default(), mode).segments.len()
    }

    fn info<'a>(g: &'a SceneGeometry, id: &str) -> &'a crate::geometry::ItemInfo {
        g.infos.iter().find(|i| i.id == id).unwrap_or_else(|| panic!("no info for {id}: {:?}", g.infos))
    }

    #[test]
    fn reduce_fields_fall_back_to_cpu_quads_not_nothing() {
        for src in ["y<=sum(x^n/n^2,n,1,3)", "sum(sin(k x)/k,k,1,3)+y", "y>int(cos(t),t,0,x)"] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert!(g.fields.is_empty(), "{src}: no GPU field expected");
            assert!(!g.flat_indices.is_empty() && !g.vertices.is_empty(), "{src}: CPU raster expected");
        }
        // Plain fields still use the GPU.
        let g = build(&doc_with(&[("a", "y<=x^2")]), Mode::D2);
        assert_eq!(g.fields.len(), 1);
    }

    #[test]
    fn fill_fallback_covers_the_right_side() {
        // y <= sum(...) fills below the curve: quads exist at negative y, none far above.
        let g = build(&doc_with(&[("a", "y<=sum(1/k,k,1,2)-8")]), Mode::D2); // y <= -6.5
        assert!(!g.vertices.is_empty());
        let ys: Vec<f32> = g.vertices.iter().map(|v| v.pos[1]).collect();
        assert!(ys.iter().all(|y| *y <= -6.0), "all fill is below y=-6.5 (to a cell)");
        assert!(ys.iter().any(|y| *y < -9.0));
    }

    #[test]
    fn curves_contours_and_surfaces_with_reduce_draw_in_every_mode() {
        let b2 = base_len(Mode::D2);
        for src in ["y=int(sin(t),t,0,x)", "y=sum(x^n/n^2,n,1,6)", "x^2+y^2=sum(1/k^2,k,1,5)"] {
            let g = build(&doc_with(&[("a", src)]), Mode::D2);
            assert!(g.diagnostics.is_empty(), "{src}: {:?}", g.diagnostics);
            assert!(g.segments.len() > b2 + 20, "{src} draws in 2D");
        }
        let g = build(&doc_with(&[("a", "y=sum(x^n/n^2,n,1,4)-2")]), Mode::D1);
        assert!(g.diagnostics.is_empty() && g.segments.len() > base_len(Mode::D1), "roots on the number line");
        let g = build(&doc_with(&[("a", "y=sum(x^n/n^2,n,1,3)")]), Mode::D3);
        assert!(g.diagnostics.is_empty() && !g.vertices.is_empty(), "surface in 3D: {:?}", g.diagnostics);
    }

    #[test]
    fn complex_sum_is_a_clear_diagnostic() {
        let mut d = Doc::new_default();
        d.items.push(Item::new("c", ItemKind::Complex, "sum(z^n,n,0,3)"));
        let g = build(&d, Mode::D2);
        assert_eq!(g.diagnostics.len(), 1);
        assert!(g.diagnostics[0].1.contains("complex items"), "{:?}", g.diagnostics);
    }

    #[test]
    fn derivative_items_draw_and_report_the_symbolic_derivative() {
        let d = doc_with(&[("f", "f(x)=x^3"), ("a", "y=f'(x)"), ("b", "y=d/dx x^3")]);
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        assert!(g.segments.len() > base_len(Mode::D2) + 20);
        for id in ["a", "b"] {
            let i = info(&g, id);
            assert_eq!(i.kind, "derivative");
            let l = i.latex.as_deref().unwrap();
            assert!(l.contains("3") && l.contains("x") && l.contains("frac"), "{l}");
        }
        assert!(g.infos.iter().all(|i| i.id != "f"), "x^3 itself has no derivative read-out");
    }

    #[test]
    fn tangent_line_with_slider_works_and_follows_it() {
        let mut d = doc_with(&[("f", "f(x)=x^3"), ("t", "y=f(a)+f'(a)(x-a)")]);
        d.sliders.insert("a".into(), math_core::doc::SliderCfg { min: -5.0, max: 5.0, step: None, value: 1.0 });
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        // At a = 1 the tangent is y = 3x - 2: it crosses y = -2 at x = 0.
        let hit = g.segments.iter().any(|s| s.p0[0].abs() < 0.2 && (s.p0[1] + 2.0).abs() < 0.2 && s.width == 2.5);
        assert!(hit, "tangent passes through (0, -2)");
    }

    #[test]
    fn degree_mode_derivative_is_scaled() {
        let mut d = doc_with(&[("a", "y=d/dx sin(x)")]);
        d.view.angle = math_core::doc::AngleMode::Deg;
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty());
        // d/dx sin(x deg) = (pi/180) cos(x): tiny, so the curve hugs the x axis.
        assert!(g.segments.iter().filter(|s| s.width == 2.5).all(|s| s.p0[1].abs() < 0.05));
    }

    #[test]
    fn definite_integral_reports_its_value() {
        let g = build(&doc_with(&[("i", "int(x^2,x,0,2)")]), Mode::D2);
        let i = info(&g, "i");
        assert_eq!(i.kind, "value");
        assert!((i.value.unwrap() - 8.0 / 3.0).abs() < 1e-9);
        assert!(i.text.as_deref().unwrap().starts_with("2.6666"));
        // Scalar items without integrals report too, and sliders flow in.
        let mut d = doc_with(&[("s", "2+3a")]);
        d.sliders.insert("a".into(), math_core::doc::SliderCfg { min: 0.0, max: 5.0, step: None, value: 2.0 });
        assert_eq!(info(&build(&d, Mode::D2), "s").value, Some(8.0));
        // Curves report no value.
        assert!(build(&doc_with(&[("c", "y=x^2")]), Mode::D2).infos.is_empty());
    }

    fn line_doc() -> Doc {
        let mut d = Doc::new_default();
        add_table(&mut d, &[("x_1", &["1", "2", "3", "4", "5"]), ("y_1", &["3", "5", "7", "9", "11"])]);
        d.items.push(Item::new("r", ItemKind::Equation, "y_1 ~ a x_1 + b"));
        d
    }

    #[test]
    fn regression_fits_draws_and_defines_parameters() {
        let mut d = line_doc();
        d.items.push(Item::new("u", ItemKind::Equation, "y=a*x+b+1"));
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let i = info(&g, "r");
        assert_eq!(i.kind, "regression");
        let p = |n: &str| i.params.iter().find(|p| p.name == n).unwrap().value;
        assert!((p("a") - 2.0).abs() < 1e-9 && (p("b") - 1.0).abs() < 1e-9);
        assert!((i.r2.unwrap() - 1.0).abs() < 1e-9 && i.rmse.unwrap() < 1e-9 && i.n == Some(5));
        assert!(i.latex.as_deref().unwrap().contains("approx"));
        // Curve + downstream item `y=a*x+b+1` (a, b defined by the visible regression).
        assert!(g.segments.iter().filter(|s| s.width == 2.5).count() > 100);
        // The curve passes through (3, 7) and the downstream line through (3, 8).
        let near = |x: f32, y: f32| g.segments.iter().any(|s| (s.p0[0] - x).abs() < 0.1 && (s.p0[1] - y).abs() < 0.1);
        assert!(near(3.0, 7.0) && near(3.0, 8.0));
    }

    #[test]
    fn hidden_regression_leaves_its_parameters_undefined() {
        let mut d = line_doc();
        d.items.push(Item::new("u", ItemKind::Equation, "y=a*x"));
        d.items.iter_mut().find(|i| i.id == "r").unwrap().hidden = true;
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.iter().any(|(id, m)| id == "u" && m.contains("'a'")), "{:?}", g.diagnostics);
        assert!(g.infos.is_empty());
    }

    #[test]
    fn slider_named_like_a_parameter_stays_a_fixed_constant() {
        let mut d = line_doc();
        d.sliders.insert("a".into(), math_core::doc::SliderCfg { min: -5.0, max: 5.0, step: None, value: 1.0 });
        let g = build(&d, Mode::D2);
        let i = info(&g, "r");
        assert_eq!(i.params.len(), 1, "only b is fitted: {:?}", i.params);
        assert_eq!(i.params[0].name, "b");
        assert!((i.params[0].value - 4.0).abs() < 1e-9, "with a = 1, b is the mean of y - x");
    }

    #[test]
    fn residual_ticks_follow_the_flag() {
        let mut d = Doc::new_default();
        add_table(&mut d, &[("x_1", &["1", "2", "3", "4"]), ("y_1", &["2", "5", "6", "9"])]);
        d.items.push(Item::new("r", ItemKind::Equation, "y_1 ~ a x_1 + b"));
        let off = build(&d, Mode::D2).segments.len();
        d.items.iter_mut().find(|i| i.id == "r").unwrap().style.residuals = true;
        let g = build(&d, Mode::D2);
        assert_eq!(g.segments.len(), off + 4, "one tick per data point");
    }

    #[test]
    fn regression_errors_are_diagnostics_not_panics() {
        for (xs, ys, model) in [
            (&["1"][..], &["2"][..], "y_1 ~ a x_1 + b"),                // too few points
            (&["1", "1", "1"][..], &["1", "2", "3"][..], "y_1 ~ a x_1 + b"), // singular
            (&["1", "2"][..], &["1", "2"][..], "y_1 ~ 3 x_1"),          // no parameters
            (&["1", "2"][..], &["1", "2"][..], "y_1 ~ a q_9 + b"),      // unknown list
        ] {
            let mut d = Doc::new_default();
            add_table(&mut d, &[("x_1", xs), ("y_1", ys)]);
            d.items.push(Item::new("r", ItemKind::Equation, model));
            let g = build(&d, Mode::D2);
            assert!(g.diagnostics.iter().any(|(id, m)| id == "r" && m.starts_with("regression:")), "{model}: {:?}", g.diagnostics);
            assert!(g.infos.is_empty());
        }
    }

    #[test]
    fn regression_in_1d_and_3d_does_not_break() {
        let d = line_doc();
        assert!(build(&d, Mode::D1).diagnostics.is_empty());
        let g = build(&d, Mode::D3);
        assert!(g.diagnostics.is_empty());
        assert!(g.segments.iter().any(|s| s.width == 2.5));
    }

    #[test]
    fn nonlinear_fit_draws_and_info_has_std_errors() {
        let mut d = Doc::new_default();
        let ys: Vec<String> = (1..=8).map(|x| format!("{:.4}", 1.5 * (0.3 * x as f64).exp())).collect();
        let xs: Vec<String> = (1..=8).map(|x| x.to_string()).collect();
        let (xr, yr): (Vec<&str>, Vec<&str>) = (xs.iter().map(String::as_str).collect(), ys.iter().map(String::as_str).collect());
        add_table(&mut d, &[("x_1", &xr), ("y_1", &yr)]);
        d.items.push(Item::new("r", ItemKind::Equation, "y_1 ~ a e^(b x_1)"));
        let g = build(&d, Mode::D2);
        assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
        let i = info(&g, "r");
        assert!((i.params[0].value - 1.5).abs() < 1e-2 && (i.params[1].value - 0.3).abs() < 1e-3, "{:?}", i.params);
    }

    fn cmd(a: &mut crate::app::App, json: &str) -> Vec<serde_json::Value> {
        serde_json::from_str(&a.dispatch(json)).unwrap()
    }

    #[test]
    fn info_event_flows_through_the_app_and_only_when_it_changes() {
        let mut a = crate::app::App::new((800, 600));
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"i","latex":"int(x^2,x,0,2)"}"#);
        let info = ev.iter().find(|e| e["t"] == "info").expect("info event");
        assert_eq!(info["items"][0]["id"], "i");
        assert_eq!(info["items"][0]["kind"], "value");
        assert!((info["items"][0]["value"].as_f64().unwrap() - 8.0 / 3.0).abs() < 1e-9);
        // An unrelated rebuild (pan) does not repeat it.
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"q","value":1}"#);
        assert!(!ev.iter().any(|e| e["t"] == "info"));
        // Removing the item clears the list.
        let ev = cmd(&mut a, r#"{"t":"removeItem","id":"i"}"#);
        let info = ev.iter().find(|e| e["t"] == "info").expect("clearing info event");
        assert_eq!(info["items"], serde_json::json!([]));
    }

    #[test]
    fn regression_commands_and_event_shape() {
        let mut a = crate::app::App::new((800, 600));
        cmd(&mut a, r#"{"t":"addTable","id":"t","columns":["x_1","y_1"],"data":[[1,3],[2,5],[3,7.5],[4,9]]}"#);
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"r","latex":"y_1 ~ a x_1 + b"}"#);
        let info = ev.iter().find(|e| e["t"] == "info").expect("info event");
        let r = &info["items"][0];
        assert_eq!(r["kind"], "regression");
        assert_eq!(r["params"][0]["name"], "a");
        assert!(r["params"][0]["stdError"].is_number());
        assert!(r["r2"].as_f64().unwrap() > 0.99 && r["rmse"].is_number() && r["n"] == 4);
        assert!(r["latex"].as_str().unwrap().contains("approx") && r["text"].is_string());
        // Residual ticks: the flag is stored on the item and survives a save/load.
        let ev = cmd(&mut a, r#"{"t":"setRegressionResiduals","id":"r","on":true}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"));
        let json = a.doc.clone();
        assert!(json.items.iter().find(|i| i.id == "r").unwrap().style.residuals);
        let text = math_core::doc::to_json(&a.doc);
        assert!(text.contains("\"residuals\":true"));
        assert!(math_core::doc::from_json(&text).unwrap().items.iter().any(|i| i.style.residuals));
        assert!(cmd(&mut a, r#"{"t":"setRegressionResiduals","id":"nope","on":true}"#).iter().any(|e| e["t"] == "error"));
        // Degenerate data: a diagnostic, never a panic.
        let ev = cmd(&mut a, r#"{"t":"setCell","id":"t","row":0,"col":0,"value":"2"}"#);
        let ev2 = cmd(&mut a, r#"{"t":"setCell","id":"t","row":1,"col":0,"value":"2"}"#);
        let ev3 = cmd(&mut a, r#"{"t":"setCell","id":"t","row":2,"col":0,"value":"2"}"#);
        let ev4 = cmd(&mut a, r#"{"t":"setCell","id":"t","row":3,"col":0,"value":"2"}"#);
        let all: Vec<_> = ev.iter().chain(&ev2).chain(&ev3).chain(&ev4).collect();
        let d = all.iter().rev().find(|e| e["t"] == "diagnostics").expect("diagnostics");
        assert!(d["items"][0]["message"].as_str().unwrap().starts_with("regression:"));
    }
}
