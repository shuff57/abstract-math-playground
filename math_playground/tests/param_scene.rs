//! Scene-level tests for parametric forms: `{a<=t<=b}` ranges, x=/y= equation pairs and
//! parametric surfaces, built through the real scene builder (no GPU).
mod common;
use common::*;
use math_core::doc::AngleMode;
use math_core::view::{Mode, Window3};
use math_playground_lib::geometry::SceneGeometry;

const W: [f64; 6] = [-10.0, 10.0, -8.0, 8.0, -8.0, 8.0];

fn win() -> Window3 {
    Window3::new([W[0], W[2], W[4]], [W[1], W[3], W[5]])
}

fn diags(g: &SceneGeometry) -> Vec<String> {
    g.diagnostics.iter().map(|d| d.1.clone()).collect()
}

/// Bounding box of the segment end points (absolute coordinates).
fn seg_box(g: &SceneGeometry, o: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for s in &g.segments {
        for p in [s.p0, s.p1] {
            for a in 0..3 {
                let v = p[a] as f64 + o[a];
                lo[a] = lo[a].min(v);
                hi[a] = hi[a].max(v);
            }
        }
    }
    (lo, hi)
}

fn curve_box(lines: &[&str], sliders: &[(&str, f64)], mode: Mode) -> ([f64; 3], [f64; 3], Vec<String>) {
    let d = make_doc(lines, sliders);
    let (g, o) = build(&d, mode, win());
    // drop the 3D box edges: curves only
    let curve: Vec<_> = g.segments.iter().filter(|s| s.width > 2.0).cloned().collect();
    let mut gg = SceneGeometry::default();
    gg.segments = curve;
    let (lo, hi) = seg_box(&gg, o);
    (lo, hi, diags(&g))
}

#[test]
fn parabola_range_gives_both_halves() {
    let (lo, hi, dg) = curve_box(&["(t^2, 2t) {-3<=t<=3}"], &[], Mode::D2);
    assert!(dg.is_empty(), "{dg:?}");
    assert!((lo[1] + 6.0).abs() < 0.05 && (hi[1] - 6.0).abs() < 0.05, "{lo:?} {hi:?}");
    assert!((hi[0] - 9.0).abs() < 0.05 && lo[0].abs() < 0.05);
    // without a range: the old default [0, 2pi]
    let (lo, hi, _) = curve_box(&["(t^2, 2t)"], &[], Mode::D2);
    assert!(lo[1] >= -0.01 && hi[1] > 12.0 && hi[1] <= 12.6 + 0.01, "{lo:?} {hi:?}");
}

#[test]
fn half_ranges_move_only_one_end() {
    // t>=-2 keeps the default end 2pi; t<=0 starts one turn earlier
    let (lo, hi, dg) = curve_box(&["(t, 0) {t>=-2}"], &[], Mode::D2);
    assert!(dg.is_empty());
    assert!((lo[0] + 2.0).abs() < 0.02 && (hi[0] - 2.0 * std::f64::consts::PI).abs() < 0.02);
    let (lo, hi, _) = curve_box(&["(t, 0) {t<=0}"], &[], Mode::D2);
    assert!((hi[0]).abs() < 0.02 && (lo[0] + 2.0 * std::f64::consts::PI).abs() < 0.02);
}

#[test]
fn bounds_may_use_pi_and_sliders() {
    let (lo, hi, dg) = curve_box(&["(t, 0) {0<=t<=a pi}"], &[("a", 1.5)], Mode::D2);
    assert!(dg.is_empty(), "{dg:?}");
    assert!(lo[0].abs() < 0.01 && (hi[0] - 1.5 * std::f64::consts::PI).abs() < 0.01);
    let (_, hi, _) = curve_box(&["(t, 0) {0<=t<=a pi}"], &[("a", 0.5)], Mode::D2);
    assert!((hi[0] - 0.5 * std::f64::consts::PI).abs() < 0.01);
    // an unknown name in a bound is the usual undefined-variable diagnostic (the app makes a slider)
    let (_, _, dg) = curve_box(&["(t, 0) {0<=t<=q}"], &[], Mode::D2);
    assert!(dg.iter().any(|m| m.contains("'q'")), "{dg:?}");
}

#[test]
fn degree_mode_ranges_are_in_degrees() {
    let mut d = make_doc(&["(t/10, 0) {0<=t<=90}"], &[]);
    d.view.angle = AngleMode::Deg;
    let (g, o) = build(&d, Mode::D2, win());
    let (lo, hi) = seg_box(&g, o);
    assert!(lo[0].abs() < 0.01 && (hi[0] - 9.0).abs() < 0.01, "{lo:?} {hi:?}");
    // default is one turn of 360
    let mut d = make_doc(&["(t/100, 0)"], &[]);
    d.view.angle = AngleMode::Deg;
    let (g, o) = build(&d, Mode::D2, win());
    let (_, hi) = seg_box(&g, o);
    assert!((hi[0] - 3.6).abs() < 0.01, "{hi:?}");
}

#[test]
fn polar_range_limits_theta() {
    // r = 4 over theta in [0, pi/2]: the first quadrant arc only
    let (lo, hi, dg) = curve_box(&["r=4 {0<=theta<=pi/2}"], &[], Mode::D2);
    assert!(dg.is_empty(), "{dg:?}");
    assert!(lo[0].abs() < 0.05 && lo[1].abs() < 0.05 && (hi[0] - 4.0).abs() < 0.05 && (hi[1] - 4.0).abs() < 0.05, "{lo:?} {hi:?}");
    // a spiral over several turns reaches r = 6pi' worth beyond the default 6pi
    let (_, hi, _) = curve_box(&["r=theta/3 {0<=theta<=24}"], &[], Mode::D2);
    assert!(hi[0].max(hi[1]) > 6.0, "{hi:?}");
}

#[test]
fn space_curve_range_runs_many_turns() {
    let (lo, hi, dg) = curve_box(&["(2cos(t), 2sin(t), t/4-6) {0<=t<=12pi}"], &[], Mode::D3);
    assert!(dg.is_empty(), "{dg:?}");
    // z from -6 to 12pi/4-6 = 3.42
    assert!((lo[2] + 6.0).abs() < 0.02 && (hi[2] - (3.0 * std::f64::consts::PI - 6.0)).abs() < 0.02, "{lo:?} {hi:?}");
    // the default is still one turn
    let (_, hi, _) = curve_box(&["(2cos(t), 2sin(t), t/4-6)"], &[], Mode::D3);
    assert!((hi[2] - (std::f64::consts::PI / 2.0 - 6.0)).abs() < 0.02);
}

#[test]
fn x_and_y_rows_are_one_curve() {
    for lines in [
        vec!["x=3cos(t)", "y=2sin(t)"],
        vec!["y=2sin(t)", "x=3cos(t)"],
        vec!["x=3cos(t), y=2sin(t)"],
        vec!["x=3cos(t); y=2sin(t)"],
    ] {
        let (lo, hi, dg) = curve_box(&lines, &[], Mode::D2);
        assert!(dg.is_empty(), "{lines:?}: {dg:?}");
        assert!((hi[0] - 3.0).abs() < 0.02 && (lo[0] + 3.0).abs() < 0.02, "{lines:?}: {lo:?} {hi:?}");
        assert!((hi[1] - 2.0).abs() < 0.02 && (lo[1] + 2.0).abs() < 0.02);
    }
    // with a range on one of the rows
    let (lo, hi, dg) = curve_box(&["x=t", "y=t^2 {0<=t<=2}"], &[], Mode::D2);
    assert!(dg.is_empty(), "{dg:?}");
    assert!(lo[0].abs() < 0.01 && (hi[0] - 2.0).abs() < 0.01 && (hi[1] - 4.0).abs() < 0.01);
    // the fused rows are one item: no extra colour slot is used
    let d = make_doc(&["x=3cos(t)", "y=2sin(t)", "y=x"], &[]);
    let (g, _) = build(&d, Mode::D2, win());
    assert_eq!(g.item_colors.len(), 2, "{:?}", g.item_colors);
}

#[test]
fn three_rows_make_a_space_curve() {
    let (lo, hi, dg) = curve_box(&["x=cos(t)", "y=sin(t)", "z=t/3"], &[], Mode::D3);
    assert!(dg.is_empty(), "{dg:?}");
    assert!((hi[0] - 1.0).abs() < 0.02 && (lo[2]).abs() < 0.02 && (hi[2] - 2.0 * std::f64::consts::PI / 3.0).abs() < 0.02, "{lo:?} {hi:?}");
    let (_, hi, _) = curve_box(&["x=cos(t), y=sin(t), z=t {0<=t<=10}"], &[], Mode::D3);
    assert!((hi[2] - 10.0).abs() < 0.02, "{hi:?}");
}

#[test]
fn ordinary_axis_equations_are_unchanged() {
    // no parameter: separate lines as before
    let d = make_doc(&["x=2", "y=3"], &[]);
    let (g, _) = build(&d, Mode::D2, win());
    assert!(g.diagnostics.is_empty());
    assert!(!g.segments.is_empty());
    // x=f(y) next to y=g(t) is not a pair (the x row uses y)
    let d = make_doc(&["x=y^2", "y=sin(t)"], &[]);
    let (g, _) = build(&d, Mode::D2, win());
    assert!(!g.segments.is_empty());
}

#[test]
fn bad_ranges_are_diagnostics_not_panics() {
    let (_, _, dg) = curve_box(&["(t, t) {3<=t<=1}"], &[], Mode::D2);
    assert!(dg.iter().any(|m| m.contains("empty range")), "{dg:?}");
    let (_, _, dg) = curve_box(&["(t, t) {0<=u<=1}"], &[], Mode::D2);
    assert!(dg.iter().any(|m| m.contains("does not apply")), "{dg:?}");
    let (_, _, dg) = curve_box(&["y=x^2 {x>0}"], &[], Mode::D2);
    assert!(dg.iter().any(|m| m.contains("range")), "{dg:?}");
    let (_, _, dg) = curve_box(&["(t, t) {t<=1/0}"], &[], Mode::D2);
    assert!(!dg.is_empty());
    let (_, _, dg) = curve_box(&["(t, t) {1<2}"], &[], Mode::D2);
    assert!(!dg.is_empty());
}

fn tris_box(g: &SceneGeometry, o: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for v in &g.vertices {
        for a in 0..3 {
            let p = v.pos[a] as f64 + o[a];
            lo[a] = lo[a].min(p);
            hi[a] = hi[a].max(p);
        }
    }
    (lo, hi)
}

const SPHERE: &str = "(a cos(u)cos(v), a cos(u)sin(v), a sin(u))";

#[test]
fn surface_follows_its_slider_and_range() {
    let d = make_doc(&[SPHERE], &[("a", 3.0)]);
    let (g, o) = build(&d, Mode::D3, win());
    assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
    let (lo, hi) = tris_box(&g, o);
    for a in 0..3 {
        assert!((hi[a] - 3.0).abs() < 0.05 && (lo[a] + 3.0).abs() < 0.05, "{lo:?} {hi:?}");
    }
    let d = make_doc(&[SPHERE], &[("a", 5.0)]);
    let (g, o) = build(&d, Mode::D3, win());
    let (_, hi) = tris_box(&g, o);
    assert!((hi[0] - 5.0).abs() < 0.05);
    // a range makes a hemisphere
    let d = make_doc(&["(3cos(u)cos(v), 3cos(u)sin(v), 3sin(u)) {0<=u<=pi/2, 0<=v<=2pi}"], &[]);
    let (g, o) = build(&d, Mode::D3, win());
    let (lo, hi) = tris_box(&g, o);
    assert!(lo[2] > -0.01 && (hi[2] - 3.0).abs() < 0.02, "{lo:?} {hi:?}");
}

#[test]
fn surface_is_clipped_to_the_window_box() {
    let d = make_doc(&["(u, v, u v/4) {-20<=u<=20, -20<=v<=20}"], &[]);
    let (g, o) = build(&d, Mode::D3, win());
    assert!(g.diagnostics.is_empty());
    let (lo, hi) = tris_box(&g, o);
    for a in 0..3 {
        assert!(lo[a] >= W[2 * a] - 1e-3 && hi[a] <= W[2 * a + 1] + 1e-3, "axis {a}: {lo:?} {hi:?}");
    }
    assert!(!g.indices.is_empty());
}

#[test]
fn u_and_v_sliders_do_not_break_a_surface() {
    // the app creates sliders for free names; u and v are the surface parameters regardless
    let d = make_doc(&[SPHERE], &[("a", 3.0), ("u", 1.0), ("v", 2.0)]);
    let (g, o) = build(&d, Mode::D3, win());
    assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
    let (_, hi) = tris_box(&g, o);
    assert!((hi[2] - 3.0).abs() < 0.05);
}

#[test]
fn surface_in_2d_draws_nothing_without_a_diagnostic() {
    let d = make_doc(&["(3cos(u)cos(v), 3cos(u)sin(v), 3sin(u))"], &[]);
    let (g, _) = build(&d, Mode::D2, win());
    assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
    assert!(g.indices.is_empty());
    assert!(g.segments.iter().all(|s| s.width <= 2.5 && s.color[3] < 1.1)); // grid/axes only
}

#[test]
fn surface_takes_item_colour_and_cooperates_with_other_items() {
    let mut d = make_doc(&["z=0", "(u, v, 1) {0<=u<=1, 0<=v<=1}"], &[]);
    d.items[1].color = Some("#ff0000".into());
    let (g, _) = build(&d, Mode::D3, win());
    assert!(g.diagnostics.is_empty());
    let red = g.vertices.iter().filter(|v| v.color[0] > 0.9 && v.color[1] < 0.1 && v.color[2] < 0.1).count();
    let other = g.vertices.len() - red;
    assert!(red > 0 && other > 0, "red {red}, other {other}");
    // and the pair form with u, v
    let d = make_doc(&["x=u", "y=v", "z=u v"], &[]);
    let (g, _) = build(&d, Mode::D3, win());
    assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
    assert!(!g.indices.is_empty());
}

#[test]
fn degenerate_and_empty_surfaces_are_quiet() {
    // all NaN
    let d = make_doc(&["(sqrt(-1-u*u), v, 0)"], &[]);
    let (g, _) = build(&d, Mode::D3, win());
    assert!(g.indices.is_empty());
    // a surface that collapses to a curve (v unused) and to a point
    for src in ["(cos(u), sin(u), 0*v)", "(1, 2, 3+0*u*v)"] {
        let d = make_doc(&[src], &[]);
        let (g, _) = build(&d, Mode::D3, win());
        assert!(g.indices.is_empty(), "{src}");
    }
    let d = make_doc(&["(u, v, 0) {2<=u<=1, 0<=v<=1}"], &[]);
    let (g, _) = build(&d, Mode::D3, win());
    assert!(!diags(&g).is_empty());
}
