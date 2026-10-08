//! Scene-level checks of piecewise `{c: v, ...}` items through the real scene builder.
mod common;
use common::*;
use math_core::view::{Mode, Window3};

fn win() -> Window3 {
    Window3::new([-10.0, -8.0, -8.0], [10.0, 8.0, 8.0])
}

fn run(src: &str) -> (Vec<[f64; 2]>, Vec<String>, String) {
    let d = make_doc(&[src], &[]);
    let (g, o) = build(&d, Mode::D2, win());
    let pts = g
        .segments
        .iter()
        .skip(g.backdrop_segments)
        .filter(|s| s.width > 2.45)
        .flat_map(|s| [s.p0, s.p1])
        .map(|p| [p[0] as f64 + o[0], p[1] as f64 + o[1]])
        .collect();
    (pts, g.diagnostics.iter().map(|d| d.1.clone()).collect(), field_desc(&g))
}

#[test]
fn piecewise_curve_has_a_gap_where_no_branch_matches() {
    let (pts, dg, _) = run("y={x<-2: x, x>2: x}");
    assert!(dg.is_empty(), "{dg:?}");
    assert!(!pts.is_empty());
    assert!(pts.iter().all(|p| p[0] < -2.0 + 1e-3 || p[0] > 2.0 - 1e-3), "a sample inside the gap");
    assert!(pts.iter().any(|p| p[0] < -5.0) && pts.iter().any(|p| p[0] > 5.0));
}

#[test]
fn absolute_value_piecewise_matches_the_v_shape() {
    let (pts, dg, _) = run("y={x<0: -x, x}");
    assert!(dg.is_empty(), "{dg:?}");
    assert!(pts.iter().all(|p| (p[1] - p[0].abs()).abs() < 0.05), "off the V");
}

#[test]
fn piecewise_with_a_range_restriction() {
    let (pts, dg, _) = run("y={x<0: -x, x} {x>-3}");
    assert!(dg.is_empty(), "{dg:?}");
    assert!(pts.iter().all(|p| p[0] >= -3.0 - 1e-6));
}

#[test]
fn inequality_with_piecewise_falls_back_to_the_cpu_field_not_an_error() {
    let (_, dg, _) = run("y<{x<0: -x, x}");
    assert!(dg.is_empty(), "{dg:?}");
}

#[test]
fn bad_piecewise_reports_a_diagnostic() {
    let (_, dg, _) = run("y={x<0: 1, 2, 3}");
    assert!(!dg.is_empty());
}
