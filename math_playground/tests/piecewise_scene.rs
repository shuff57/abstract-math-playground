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
        .filter(|s| s.width > 3.1)
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

/// (open interiors, filled dots) of the endpoint markers, in absolute coordinates.
fn markers(src: &str) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let d = make_doc(&[src], &[]);
    let (g, o) = build(&d, Mode::D2, win());
    let bg = math_playground_lib::geometry::Theme::light().background;
    let abs = |p: [f32; 3]| [p[0] as f64 + o[0], p[1] as f64 + o[1]];
    let (mut open, mut filled) = (Vec::new(), Vec::new());
    for s in g.segments.iter().skip(g.backdrop_segments).filter(|s| s.p0 == s.p1 && s.width > 8.0) {
        if s.color == bg {
            open.push(abs(s.p0));
        } else {
            filled.push(abs(s.p0));
        }
    }
    (open, filled)
}

fn has(v: &[[f64; 2]], p: [f64; 2]) -> bool {
    v.iter().any(|q| (q[0] - p[0]).abs() < 0.05 && (q[1] - p[1]).abs() < 0.05)
}

#[test]
fn strict_piecewise_ends_get_open_markers() {
    let (open, filled) = markers("y={x<-1:-x, x>1:x}");
    assert_eq!(open.len(), 2, "{open:?}");
    assert!(filled.is_empty());
    assert!(has(&open, [-1.0, 1.0]) && has(&open, [1.0, 1.0]));
}

#[test]
fn non_strict_piecewise_ends_get_filled_markers() {
    let (open, filled) = markers("y={x<=-1:-x, x>=1:x}");
    assert!(open.is_empty(), "{open:?}");
    assert_eq!(filled.len(), 2, "{filled:?}");
    assert!(has(&filled, [-1.0, 1.0]) && has(&filled, [1.0, 1.0]));
}

#[test]
fn piecewise_jump_is_open_at_one_branch_and_filled_at_the_other() {
    let (open, filled) = markers("y={x<0:x^2, x^2+1}");
    assert!(has(&open, [0.0, 0.0]), "{open:?}");
    assert!(has(&filled, [0.0, 1.0]), "{filled:?}");
    assert_eq!((open.len(), filled.len()), (1, 1));
    // a continuous join has no marker
    let (open, filled) = markers("y={x<0:-x, x}");
    assert!(open.is_empty() && filled.is_empty());
}

#[test]
fn non_piecewise_functions_gain_no_markers() {
    for src in ["y=sqrt(x)", "y=ln(x)", "y=1/x", "y=tan(x)"] {
        let (open, filled) = markers(src);
        assert!(open.is_empty() && filled.is_empty(), "{src}: {open:?} {filled:?}");
    }
}
