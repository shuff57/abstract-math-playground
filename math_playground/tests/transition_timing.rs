//! Scene build cost around a 2D -> 3D switch (preview vs full), for tuning the staged rebuild.
//!   cargo test -p math_playground --release --test transition_timing -- --ignored --nocapture
mod common;
use common::*;
use math_core::view::{Mode, Window3};
use math_playground_lib::geometry::Theme;
use math_playground_lib::scene::build_scene_preview;
use std::time::Instant;

#[test]
#[ignore = "timing report"]
fn preview_vs_full_build_cost() {
    let win = Window3::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
    for lines in [
        vec!["y=x^2"],
        vec!["z=sin(x)cos(y)"],
        vec!["x^2+y^2+z^2=9"],
        vec!["y=x^2", "z=sin(x)cos(y)", "x^2+y^2+z^2=9"],
    ] {
        let doc = make_doc(&lines, &[]);
        let o = win.centre();
        let theme = Theme::light();
        let t = Instant::now();
        let (_, coarse) = build_scene_preview(&doc, Mode::D3, win, o, (1280, 620), &theme);
        let pv = t.elapsed();
        let t = Instant::now();
        let _ = build(&doc, Mode::D3, win);
        let full = t.elapsed();
        let t = Instant::now();
        let _ = build(&doc, Mode::D2, win);
        let d2 = t.elapsed();
        eprintln!("{lines:?}: preview {pv:?} (coarse={coarse}), full {full:?}, 2D {d2:?}");
    }
}

use math_playground_lib::axis_map::AxisMap;
use math_playground_lib::scene::{build_scene_progressive_mapped, clear_surface_tiles, full_surface_depth};

fn tri_set(g: &math_playground_lib::geometry::SceneGeometry) -> Vec<[i64; 9]> {
    let q = |v: f32| (v as f64 * 1e3).round() as i64;
    let mut out: Vec<[i64; 9]> = g
        .indices
        .chunks(3)
        .map(|t| {
            let mut vs: Vec<[i64; 3]> = t
                .iter()
                .map(|&i| {
                    let p = g.vertices[i as usize].pos;
                    [q(p[0]), q(p[1]), q(p[2])]
                })
                .collect();
            vs.sort();
            [vs[0][0], vs[0][1], vs[0][2], vs[1][0], vs[1][1], vs[1][2], vs[2][0], vs[2][1], vs[2][2]]
        })
        .collect();
    out.sort();
    out
}

/// A surface meshed in time-sliced tiles is the same mesh as the one-shot build (no seams, no
/// holes, no extra triangles), however many calls it takes.
#[test]
fn tiled_progressive_surface_equals_the_one_shot_mesh() {
    let win = Window3::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
    let theme = Theme::light();
    let size = (700u32, 500u32);
    let depth = full_surface_depth((size.0 as f64, size.1 as f64));
    assert!(depth >= 6, "{depth}");
    for lines in [
        vec!["x^2+y^2+z^2=9"],
        vec!["x^2+y^2-z^3=1"],
        vec!["sin(x)+sin(y)+sin(z)=0"],
    ] {
        let doc = make_doc(&lines, &[]);
        let o = win.centre();
        let reference = math_playground_lib::scene::build_scene(&doc, Mode::D3, win, o, size, &theme);
        assert!(!reference.indices.is_empty(), "{lines:?}");
        clear_surface_tiles();
        let mut steps = 0;
        let got = loop {
            steps += 1;
            // A deadline already in the past: one tile per call, the slowest possible slicing.
            let r = build_scene_progressive_mapped(
                &doc, AxisMap::default(), Mode::D3, win, o, size, &theme, depth, Instant::now(), false,
            );
            if r.done {
                break r.geometry;
            }
            assert!(r.geometry.indices.is_empty(), "an unfinished surface must not be drawn");
            assert!(steps < 100_000);
        };
        assert!(steps > 3, "{lines:?} was not sliced ({steps} steps)");
        let (a, b) = (tri_set(&got), tri_set(&reference));
        let sa: std::collections::HashSet<_> = a.iter().collect();
        let sb: std::collections::HashSet<_> = b.iter().collect();
        let diff = sa.symmetric_difference(&sb).count();
        eprintln!("{lines:?}: tiled {} tris, one-shot {} tris, {diff} differ", a.len(), b.len());
        assert!(a == b, "{lines:?}: tiled {} vs one-shot {} triangles, {diff} differ", a.len(), b.len());
        clear_surface_tiles();
    }
}

/// A height field (`z = f(x, y)`) is one unsliceable step: a build told to defer heavy work
/// leaves it for later instead of stalling a tween, and does it once allowed.
#[test]
fn heavy_height_field_stage_can_be_deferred() {
    let win = Window3::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
    let theme = Theme::light();
    let doc = make_doc(&["z=sin(x)cos(y)"], &[]);
    let o = win.centre();
    let far = Instant::now() + std::time::Duration::from_secs(60);
    clear_surface_tiles();
    // The full depth for this viewport is heavy; a lower cap is not.
    let full = full_surface_depth((700.0, 500.0));
    let r = build_scene_progressive_mapped(&doc, AxisMap::default(), Mode::D3, win, o, (700, 500), &theme, full, far, true);
    assert!(!r.done && r.geometry.indices.is_empty());
    // Cap 3 meshes it at depth 5: light, goes through even while deferring.
    let r = build_scene_progressive_mapped(&doc, AxisMap::default(), Mode::D3, win, o, (700, 500), &theme, 3, far, true);
    assert!(r.done && !r.geometry.indices.is_empty());
    let r = build_scene_progressive_mapped(&doc, AxisMap::default(), Mode::D3, win, o, (700, 500), &theme, full, far, false);
    assert!(r.done && !r.geometry.indices.is_empty());
    clear_surface_tiles();
}

/// A generous deadline finishes in one call and a capped build is flagged as coarser.
#[test]
fn progressive_build_reports_done_and_capped() {
    let win = Window3::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
    let theme = Theme::light();
    let doc = make_doc(&["x^2+y^2+z^2=9"], &[]);
    let o = win.centre();
    clear_surface_tiles();
    let r = build_scene_progressive_mapped(
        &doc, AxisMap::default(), Mode::D3, win, o, (700, 500), &theme, 4,
        Instant::now() + std::time::Duration::from_secs(60), false,
    );
    assert!(r.done && r.capped && !r.geometry.indices.is_empty());
    clear_surface_tiles();
}

/// Boundary edges (used by one triangle) after welding vertices by position: cracks show up as
/// extra ones along the tile seams.
fn open_edges(g: &math_playground_lib::geometry::SceneGeometry) -> usize {
    use std::collections::HashMap;
    let q = |v: f32| (v as f64 * 1e4).round() as i64;
    let key = |i: u32| {
        let p = g.vertices[i as usize].pos;
        [q(p[0]), q(p[1]), q(p[2])]
    };
    let mut edges: HashMap<([i64; 3], [i64; 3]), u32> = HashMap::new();
    for t in g.indices.chunks(3) {
        for k in 0..3 {
            let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
            if a == b {
                continue;
            }
            *edges.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    edges.values().filter(|&&c| c == 1).count()
}

#[test]
#[ignore = "timing report"]
fn explicit_surface_full_build_cost() {
    let win = Window3::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
    let theme = Theme::light();
    for l in ["z=sin(x)cos(y)", "y=x^2", "y=sin(x)*z", "z=5*sin(3*x)*cos(3*y)", "x^2+y^2+z^2=9", "sin(x)+sin(y)+sin(z)=0"] {
        let doc = make_doc(&[l], &[]);
        let t = Instant::now();
        let _ = math_playground_lib::scene::build_scene(&doc, Mode::D3, win, win.centre(), (1280, 620), &theme);
        eprintln!("{l}: full build {:?}", t.elapsed());
    }
}
