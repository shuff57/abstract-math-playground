//! Scene-level tests of domain restriction `{x>0}` on explicit and implicit items: the curve is
//! clipped, endpoint markers are open (strict) or filled (non-strict), unsupported cases keep
//! their diagnostic. Built through the real scene builder (no GPU).
mod common;
use common::*;
use math_core::view::{Mode, Window3};
use math_playground_lib::geometry::{SceneGeometry, Theme};
use math_playground_lib::scene::build_scene;

fn win() -> Window3 {
    Window3::new([-10.0, -8.0, -8.0], [10.0, 8.0, 8.0])
}

struct Drawn {
    /// end points of the curve segments (absolute coordinates)
    curve: Vec<[f64; 2]>,
    /// centres of filled dots
    filled: Vec<[f64; 2]>,
    /// centres of open rings (mean of the ring's segment ends)
    open: Vec<[f64; 2]>,
    /// centres of opaque background-coloured discs (the interior of an open ring)
    interiors: Vec<[f64; 2]>,
    diags: Vec<String>,
    g: SceneGeometry,
}

fn drawn(lines: &[&str]) -> Drawn {
    drawn_in(lines, &Theme::light())
}

fn drawn_in(lines: &[&str], theme: &Theme) -> Drawn {
    let d = make_doc(lines, &[]);
    let c = win().centre();
    let o = c;
    let g = build_scene(&d, Mode::D2, win(), c, VIEWPORT, theme);
    let abs = |p: [f32; 3]| [p[0] as f64 + o[0], p[1] as f64 + o[1]];
    let (mut curve, mut filled, mut ring, mut interiors) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in g.segments.iter().skip(g.backdrop_segments) {
        if s.p0 == s.p1 && s.width > 6.0 && s.color == theme.background {
            interiors.push(abs(s.p0));
        } else if s.p0 == s.p1 && s.width > 6.0 {
            filled.push(abs(s.p0));
        } else if s.width < 2.2 {
            ring.push(abs(s.p0));
            ring.push(abs(s.p1));
        } else {
            curve.push(abs(s.p0));
            curve.push(abs(s.p1));
        }
    }
    // group ring points into rings by proximity
    let mut open: Vec<([f64; 2], f64, usize)> = Vec::new();
    for p in ring {
        match open.iter_mut().find(|(c, _, _)| ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)).sqrt() < 0.5) {
            Some((c, _, n)) => {
                *n += 1;
                c[0] += (p[0] - c[0]) / *n as f64;
                c[1] += (p[1] - c[1]) / *n as f64;
            }
            None => open.push((p, 0.0, 1)),
        }
    }
    let diags = g.diagnostics.iter().map(|d| d.1.clone()).collect();
    Drawn { curve, filled, interiors, open: open.into_iter().map(|o| o.0).collect(), diags, g }
}

fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 0.15 && (a[1] - b[1]).abs() < 0.15
}

#[test]
fn parabola_right_half_with_an_open_endpoint() {
    let d = drawn(&["y=x^2 {x>0}"]);
    assert!(d.diags.is_empty(), "{:?}", d.diags);
    assert!(!d.curve.is_empty());
    assert!(d.curve.iter().all(|p| p[0] >= -1e-4), "a sample left of 0");
    assert!(d.curve.iter().any(|p| p[0] > 2.5), "the right half is drawn");
    assert!(d.filled.is_empty());
    assert_eq!(d.open.len(), 1, "{:?}", d.open);
    assert!(near(d.open[0], [0.0, 0.0]), "{:?}", d.open);
}

#[test]
fn closed_bound_gets_a_filled_dot_and_both_ends_are_marked() {
    let d = drawn(&["y=x^2 {-1<=x<2}"]);
    assert!(d.diags.is_empty(), "{:?}", d.diags);
    assert!(d.curve.iter().all(|p| p[0] >= -1.0 - 1e-4 && p[0] <= 2.0 + 1e-4));
    assert_eq!(d.filled.len(), 1, "{:?}", d.filled);
    assert!(near(d.filled[0], [-1.0, 1.0]));
    assert_eq!(d.open.len(), 1);
    assert!(near(d.open[0], [2.0, 4.0]));
}

#[test]
fn x_equals_g_of_y_and_implicit_items_are_clipped() {
    let d = drawn(&["x=y^2 {y>=0}"]);
    assert!(d.diags.is_empty(), "{:?}", d.diags);
    assert!(d.curve.iter().all(|p| p[1] >= -1e-3), "lower branch drawn");
    assert!(d.curve.iter().any(|p| p[1] > 2.0));
    assert_eq!(d.filled.len(), 1);
    assert!(near(d.filled[0], [0.0, 0.0]));

    let d = drawn(&["x^2+y^2=9 {y>0}"]);
    assert!(d.diags.is_empty(), "{:?}", d.diags);
    assert!(d.curve.iter().all(|p| p[1] >= -1e-3));
    assert_eq!(d.open.len(), 2, "{:?}", d.open);
    assert!(d.open.iter().all(|p| p[1].abs() < 0.15 && (p[0].abs() - 3.0).abs() < 0.15));
}

#[test]
fn inequality_fill_is_cut_and_keeps_one_fill() {
    let d = drawn(&["y<x {x>0}"]);
    assert!(d.diags.is_empty(), "{:?}", d.diags);
    assert_eq!(field_desc(&d.g), "fill(f<0)");
    let plain = drawn(&["y<x"]);
    assert_eq!(field_desc(&plain.g), "fill(f<0)");
    // the fill's shader differs: it carries the box
    assert_ne!(d.g.fields[0].wgsl, plain.g.fields[0].wgsl);
    // the boundary line y=x is drawn only for x>0
    assert!(d.curve.iter().all(|p| p[0] >= -1e-3));
}

#[test]
fn unsupported_cases_keep_the_diagnostic() {
    // a bare number item has no curve to clip
    let d = drawn(&["2+3 {x>0}"]);
    assert!(d.diags.iter().any(|m| m.contains("applies to")), "{:?}", d.diags);
    // a name that is not x or y
    let d = drawn(&["y=x {t>0}"]);
    assert!(d.diags.iter().any(|m| m.contains("'t'")), "{:?}", d.diags);
    // an empty range
    let d = drawn(&["y=x {x>2, x<1}"]);
    assert!(d.diags.iter().any(|m| m.contains("empty")), "{:?}", d.diags);
    assert!(d.curve.is_empty());
    // 3D: restriction is 2D-only
    let dd = make_doc(&["y=x^2 {x>0}"], &[]);
    let (g, _) = build(&dd, Mode::D3, win());
    assert!(g.diagnostics.iter().any(|m| m.1.contains("2D only")), "{:?}", g.diagnostics);
}

#[test]
fn slider_bounds_move_the_clip() {
    let dd = make_doc(&["y=x^2 {x>a}"], &[("a", 1.0)]);
    let (g, _) = build(&dd, Mode::D2, win());
    assert!(g.diagnostics.is_empty(), "{:?}", g.diagnostics);
}

#[test]
fn restricted_items_without_a_range_are_unchanged() {
    let a = drawn(&["y=x^2"]);
    assert!(a.open.is_empty() && a.filled.is_empty());
    assert!(a.curve.iter().any(|p| p[0] < -2.5));
}

#[test]
fn open_marker_has_an_opaque_background_interior_in_both_themes() {
    for theme in [Theme::light(), Theme::dark()] {
        let d = drawn_in(&["y=x^2 {x>0}"], &theme);
        assert_eq!(d.open.len(), 1);
        assert_eq!(d.interiors.len(), 1, "one opaque interior for the open ring");
        assert!(near(d.interiors[0], [0.0, 0.0]));
        // the interior is in the background colour, fully opaque, and drawn after the curve
        let i = d.g.segments.iter().position(|s| s.p0 == s.p1 && s.color == theme.background && s.width > 6.0).unwrap();
        assert_eq!(d.g.segments[i].color[3], 1.0);
        let last_curve = d.g.segments.iter().rposition(|s| s.p0 != s.p1 && s.width >= 2.2).unwrap();
        assert!(i > last_curve, "interior must cover the curve end");
        // the ring is drawn over the interior
        assert!(d.g.segments[i + 1..].iter().any(|s| s.width < 2.2 && s.p0 != s.p1));
        // the interior is as wide as the ring's outer diameter
        let ring_w = d.g.segments[i + 1].width;
        let ring_d = d.g.segments[i + 1..].iter().find(|s| s.width < 2.2).unwrap();
        assert!(d.g.segments[i].width > 2.0 * ring_w && ring_d.width >= 1.5);
        // a closed marker has no such interior
        let c = drawn_in(&["y=x^2 {x>=0}"], &theme);
        assert!(c.interiors.is_empty());
        assert_eq!(c.filled.len(), 1);
    }
}
