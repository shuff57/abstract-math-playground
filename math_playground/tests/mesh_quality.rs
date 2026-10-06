//! Scene-level checks of the mesher fixes: curves and graphs with poles, jumps and domain edges,
//! and the double-root shapes. (The conic and solid audits cover the broad matrix.)
mod common;
use common::*;
use math_core::view::{Mode, Window3};
use std::f64::consts::PI;

fn segs_2d(line: &str) -> Vec<([f64; 2], [f64; 2])> {
    let doc = make_doc(&[line], &[]);
    let (g, o) = build(&doc, Mode::D2, Window3::new([-6.0, -4.5, -1.0], [6.0, 4.5, 1.0]));
    g.segments
        .iter()
        .map(|s| ([s.p0[0] as f64 + o[0], s.p0[1] as f64 + o[1]], [s.p1[0] as f64 + o[0], s.p1[1] as f64 + o[1]]))
        .collect()
}

#[test]
fn explicit_2d_curves_do_not_join_across_poles() {
    // tan(x): no segment spans an asymptote x = pi/2 + k pi
    let segs = segs_2d("y=tan(x)");
    assert!(segs.len() > 100);
    let branch = |x: f64| ((x - PI / 2.0) / PI).floor();
    for (a, b) in &segs {
        assert_eq!(branch(a[0]), branch(b[0]), "{a:?} -> {b:?} joins two branches");
    }
    // 1/x: the two branches stay on their own side of x = 0
    let segs = segs_2d("y=1/x");
    assert!(segs.len() > 50);
    for (a, b) in &segs {
        assert!((a[0] < 0.0) == (b[0] < 0.0), "{a:?} -> {b:?}");
    }
    // no connector that flips from one end of the window to the other
    let segs = segs_2d("y=tan(x)+1/x");
    for (a, b) in &segs {
        assert!(!((a[1] < 0.0) != (b[1] < 0.0) && (a[1] - b[1]).abs() > 4.5), "{a:?} -> {b:?} is a connector");
    }
}

#[test]
fn degenerate_conics_are_drawn_as_a_dot_and_whole_lines() {
    let dot = segs_2d("(x-0.37)^2+(y-0.21)^2=0");
    assert!(!dot.is_empty());
    assert!(dot.iter().all(|(a, b)| (a[0] - 0.37).hypot(a[1] - 0.21) < 1e-3 && (b[0] - 0.37).hypot(b[1] - 0.21) < 1e-3));
    let line = segs_2d("(x-0.37)^2=0");
    let len: f64 = line.iter().map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1])).sum();
    assert!((len - 9.0).abs() < 0.2, "{len}");
    assert!(line.iter().all(|(a, b)| (a[0] - 0.37).abs() < 1e-3 && (b[0] - 0.37).abs() < 1e-3));
    let circle = segs_2d("(x^2+y^2-4)^2=0");
    let len: f64 = circle.iter().map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1])).sum();
    assert!((len - 4.0 * PI).abs() < 0.3, "{len}");
}

fn mesh_3d(line: &str) -> (Vec<[f64; 3]>, usize) {
    let doc = make_doc(&[line], &[]);
    let (g, o) = build(&doc, Mode::D3, Window3::new([-6.0; 3], [6.0; 3]));
    (g.vertices.iter().map(|v| [v.pos[0] as f64 + o[0], v.pos[1] as f64 + o[1], v.pos[2] as f64 + o[2]]).collect(), g.indices.len() / 3)
}

#[test]
fn explicit_surfaces_break_at_poles_jumps_and_domain_edges() {
    // tan: every triangle lies on one branch
    let doc = make_doc(&["z=tan(x)"], &[]);
    let (g, o) = build(&doc, Mode::D3, Window3::new([-6.0; 3], [6.0; 3]));
    let branch = |x: f32| ((x as f64 + o[0] - PI / 2.0) / PI).floor();
    for t in g.indices.chunks(3) {
        let b: Vec<f64> = t.iter().map(|&i| branch(g.vertices[i as usize].pos[0])).collect();
        assert!(b.iter().all(|&x| x == b[0]), "{b:?}");
    }
    // floor: flat treads only
    let doc = make_doc(&["z=floor(x)"], &[]);
    let (g, _) = build(&doc, Mode::D3, Window3::new([-6.0; 3], [6.0; 3]));
    for t in g.indices.chunks(3) {
        let z: Vec<f32> = t.iter().map(|&i| g.vertices[i as usize].pos[2]).collect();
        let r = z.iter().cloned().fold(f32::MIN, f32::max) - z.iter().cloned().fold(f32::MAX, f32::min);
        assert!(r < 1e-3, "a tread triangle spans {r}");
    }
    // sqrt rim: the mesh ends exactly on the circle x^2+y^2 = 16
    let (v, tris) = mesh_3d("z=sqrt(16-x^2-y^2)");
    assert!(tris > 1000);
    assert!(v.iter().all(|p| p[0].hypot(p[1]) <= 4.0 + 1e-4));
    assert!(v.iter().any(|p| (p[0].hypot(p[1]) - 4.0).abs() < 1e-4 && p[2].abs() < 1e-4), "no rim vertices");
    // y = f(x, z) and x = f(y, z) take the same path
    let (v, tris) = mesh_3d("y=sqrt(16-x^2-z^2)");
    assert!(tris > 1000 && v.iter().all(|p| p[0].hypot(p[2]) <= 4.0 + 1e-4));
    let (v, tris) = mesh_3d("x=sqrt(16-y^2-z^2)");
    assert!(tris > 1000 && v.iter().all(|p| p[1].hypot(p[2]) <= 4.0 + 1e-4));
}

#[test]
fn double_root_quadrics_are_drawn() {
    // point quadric: a small ball at the point
    let (v, tris) = mesh_3d("(x-0.37)^2+(y-0.21)^2+(z-0.11)^2=0");
    assert!(tris > 0);
    assert!(v.iter().all(|p| ((p[0] - 0.37).powi(2) + (p[1] - 0.21).powi(2) + (p[2] - 0.11).powi(2)).sqrt() < 0.3));
    // double sphere: the sphere, once
    let (v, tris) = mesh_3d("(x^2+y^2+z^2-9)^2=0");
    assert!(tris > 1000);
    assert!(v.iter().all(|p| (p[0].hypot(p[1]).hypot(p[2]) - 3.0).abs() < 1e-3));
}
