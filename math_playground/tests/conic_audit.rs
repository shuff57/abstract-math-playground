//! Conic-section audit: builds the real 2D scene geometry for the exact text a user would type,
//! then checks the drawn contour against an analytic oracle (dense sample of the true curve):
//!   precision: every drawn point lies within ~3 px of the true curve (no extra branches/blobs),
//!   recall:    every true point inside the window is within ~3 px of something drawn
//!              (no holes, gaps, missing branches).
//! Cases flagged `gap: Some(..)` are known failures; each has an `#[ignore = "GAP: ..."]` test
//! below that fails when run with `--ignored` and starts passing when the engine is fixed.
//!
//! Run the full matrix with a printed table:
//!   cargo test -p math_playground --test conic_audit conic_report -- --ignored --nocapture
mod common;
use common::*;
use math_core::view::{Mode, Window3};
use std::f64::consts::PI;

struct Case {
    name: &'static str,
    lines: Vec<&'static str>,
    sliders: Vec<(&'static str, f64)>,
    win: [f64; 4],
    oracle: Vec<(f64, f64)>,
    expect_empty: bool,
    fill: Option<bool>,
    dashed: bool,
    gap: Option<&'static str>,
}

const W: [f64; 4] = [-10.0, 10.0, -7.5, 7.5];

fn case(name: &'static str, lines: &[&'static str], oracle: Vec<(f64, f64)>) -> Case {
    Case { name, lines: lines.to_vec(), sliders: vec![], win: W, oracle, expect_empty: false, fill: None, dashed: false, gap: None }
}
impl Case {
    fn sl(mut self, s: &[(&'static str, f64)]) -> Self {
        self.sliders = s.to_vec();
        self
    }
    fn win(mut self, w: [f64; 4]) -> Self {
        self.win = w;
        self
    }
    fn empty(mut self) -> Self {
        self.expect_empty = true;
        self
    }
    fn fill(mut self, greater: bool) -> Self {
        self.fill = Some(greater);
        self
    }
    fn dashed(mut self) -> Self {
        self.dashed = true;
        self
    }
    fn gap(mut self, why: &'static str) -> Self {
        self.gap = Some(why);
        self
    }
}

// ---- oracles ------------------------------------------------------------------------------

const N: usize = 40_000;
fn ellipse(cx: f64, cy: f64, a: f64, b: f64, rot: f64) -> Vec<(f64, f64)> {
    param(|t| {
        let (u, v) = (a * t.cos(), b * t.sin());
        (cx + u * rot.cos() - v * rot.sin(), cy + u * rot.sin() + v * rot.cos())
    }, 0.0, 2.0 * PI, N)
}
fn hyper_x(cx: f64, cy: f64, a: f64, b: f64) -> Vec<(f64, f64)> {
    let mut v = param(|s| (cx + a * s.cosh(), cy + b * s.sinh()), -5.0, 5.0, 2 * N);
    v.extend(param(|s| (cx - a * s.cosh(), cy + b * s.sinh()), -5.0, 5.0, 2 * N));
    v
}
fn hyper_y(cx: f64, cy: f64, a: f64, b: f64) -> Vec<(f64, f64)> {
    // y^2/a^2 - x^2/b^2 = 1
    let mut v = param(|s| (cx + b * s.sinh(), cy + a * s.cosh()), -5.0, 5.0, 2 * N);
    v.extend(param(|s| (cx + b * s.sinh(), cy - a * s.cosh()), -5.0, 5.0, 2 * N));
    v
}
fn line(f: impl Fn(f64) -> (f64, f64)) -> Vec<(f64, f64)> {
    param(f, -40.0, 40.0, 40_000)
}
fn parab_y(f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    param(|x| (x, f(x)), -12.0, 12.0, 60_000)
}
fn rot45(u: f64, v: f64) -> (f64, f64) {
    ((u + v) / 2f64.sqrt(), (u - v) / 2f64.sqrt())
}
fn polar(r: impl Fn(f64) -> f64, t0: f64, t1: f64) -> Vec<(f64, f64)> {
    param(|t| {
        let rr = r(t);
        (rr * t.cos(), rr * t.sin())
    }, t0, t1, 80_000)
}
fn cat(mut a: Vec<(f64, f64)>, b: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    a.extend(b);
    a
}

fn cases() -> Vec<Case> {
    let sq = (3.0f64).sqrt();
    let _ = sq;
    vec![
        // ---------- circles ----------
        case("circle_standard", &["x^2+y^2=9"], ellipse(0.0, 0.0, 3.0, 3.0, 0.0)),
        case("circle_shifted_vertex_form", &["(x-2)^2+(y+1)^2=4"], ellipse(2.0, -1.0, 2.0, 2.0, 0.0)),
        case("circle_general_form", &["x^2+y^2+4x-6y-12=0"], ellipse(-2.0, 3.0, 5.0, 5.0, 0.0)),
        case("circle_general_form_neg", &["x^2+y^2-2x+4y+1=0"], ellipse(1.0, -2.0, 2.0, 2.0, 0.0)),
        case("circle_slider_radius", &["x^2+y^2=a^2"], ellipse(0.0, 0.0, 4.0, 4.0, 0.0)).sl(&[("a", 4.0)]),
        case("circle_x_y_swapped_sides", &["9=x^2+y^2"], ellipse(0.0, 0.0, 3.0, 3.0, 0.0)),
        case("circle_huge_window", &["x^2+y^2=1000000"], ellipse(0.0, 0.0, 1000.0, 1000.0, 0.0)).win([-1500.0, 1500.0, -1125.0, 1125.0]),
        case("circle_tiny_window", &["x^2+y^2=0.000001"], ellipse(0.0, 0.0, 0.001, 0.001, 0.0)).win([-0.0015, 0.0015, -0.001125, 0.001125]),
        case("circle_far_from_origin", &["(x-1000)^2+(y-1000)^2=4"], ellipse(1000.0, 1000.0, 2.0, 2.0, 0.0)).win([990.0, 1010.0, 992.5, 1007.5]),
        // ---------- ellipses ----------
        case("ellipse_axis_aligned", &["x^2/9+y^2/4=1"], ellipse(0.0, 0.0, 3.0, 2.0, 0.0)),
        case("ellipse_tall", &["x^2/4+y^2/36=1"], ellipse(0.0, 0.0, 2.0, 6.0, 0.0)),
        case("ellipse_shifted", &["(x-3)^2/16+(y+2)^2/4=1"], ellipse(3.0, -2.0, 4.0, 2.0, 0.0)),
        case("ellipse_general_form", &["4x^2+9y^2-16x+18y-11=0"], ellipse(2.0, -1.0, 3.0, 2.0, 0.0)),
        case("ellipse_rotated_xy_term", &["x^2+xy+y^2=3"],
            param(|t| rot45(2f64.sqrt() * t.cos(), 6f64.sqrt() * t.sin()), 0.0, 2.0 * PI, N)),
        case("ellipse_rotated_big_xy", &["5x^2+6xy+5y^2=32"],
            // eigen 8 (u) and 2 (v): u^2/4 + v^2/16 = 1
            param(|t| rot45(2.0 * t.cos(), 4.0 * t.sin()), 0.0, 2.0 * PI, N)),
        case("ellipse_thin_semi_axis_0.05", &["x^2/64+400y^2=1"], ellipse(0.0, 0.0, 8.0, 0.05, 0.0)),
        case("ellipse_sharp_vertices_ecc_0.999", &["x^2/64+y^2/0.2=1"], ellipse(0.0, 0.0, 8.0, 0.4472136, 0.0)),
        case("ellipse_slider_axes", &["x^2/a^2+y^2/b^2=1"], ellipse(0.0, 0.0, 5.0, 2.0, 0.0)).sl(&[("a", 5.0), ("b", 2.0)]),
        // ---------- parabolas ----------
        case("parabola_y_ax2", &["y=x^2"], parab_y(|x| x * x)),
        case("parabola_y_general", &["y=2x^2-3x+1"], parab_y(|x| 2.0 * x * x - 3.0 * x + 1.0)),
        case("parabola_vertex_form_down", &["y=-(x-2)^2+3"], parab_y(|x| -(x - 2.0).powi(2) + 3.0)),
        case("parabola_x_ay2", &["x=y^2/4"], param(|y| (y * y / 4.0, y), -12.0, 12.0, 60_000)),
        case("parabola_x_left", &["x=-y^2+2"], param(|y| (-y * y + 2.0, y), -12.0, 12.0, 60_000)),
        case("parabola_implicit_horizontal", &["(y-1)^2=4(x+2)"], param(|y| ((y - 1.0).powi(2) / 4.0 - 2.0, y), -12.0, 12.0, 60_000)),
        case("parabola_implicit_down", &["x^2=-4y"], parab_y(|x| -x * x / 4.0)),
        case("parabola_implicit_left", &["y^2=-8x"], param(|y| (-y * y / 8.0, y), -12.0, 12.0, 60_000)),
        case("parabola_rotated_xy", &["x^2+2xy+y^2+x-y=0"],
            // (x+y)^2 = y-x  ->  u=(x+y)/sqrt2, v=(x-y)/sqrt2: 2u^2 = -sqrt2 v
            param(|u| rot45(u, -2f64.sqrt() * u * u), -12.0, 12.0, 60_000)),
        // ---------- hyperbolas ----------
        case("hyperbola_horizontal", &["x^2/4-y^2/9=1"], hyper_x(0.0, 0.0, 2.0, 3.0)),
        case("hyperbola_vertical", &["y^2/4-x^2/9=1"], hyper_y(0.0, 0.0, 2.0, 3.0)),
        case("hyperbola_shifted", &["(x-2)^2/4-(y+1)^2/9=1"], hyper_x(2.0, -1.0, 2.0, 3.0)),
        case("hyperbola_rectangular_xy_1", &["xy=1"],
            cat(param(|x| (x, 1.0 / x), 0.02, 60.0, 80_000), param(|x| (x, 1.0 / x), -60.0, -0.02, 80_000))),
        case("hyperbola_xy_neg2_spaced", &["x y=-2"],
            cat(param(|x| (x, -2.0 / x), 0.02, 60.0, 80_000), param(|x| (x, -2.0 / x), -60.0, -0.02, 80_000))),
        case("hyperbola_with_asymptotes", &["x^2/4-y^2/9=1", "y=3x/2", "y=-3x/2"],
            cat(cat(hyper_x(0.0, 0.0, 2.0, 3.0), line(|t| (t, 1.5 * t))), line(|t| (t, -1.5 * t)))),
        case("hyperbola_rotated_xy_x2_minus_y2_rot", &["x^2+3xy+y^2=5"],
            // 2.5u^2 - 0.5v^2 = 5 ->  u^2/2 - v^2/10 = 1
            cat(param(|s| rot45(2f64.sqrt() * s.cosh(), 10f64.sqrt() * s.sinh()), -5.0, 5.0, 2 * N),
                param(|s| rot45(-(2f64.sqrt()) * s.cosh(), 10f64.sqrt() * s.sinh()), -5.0, 5.0, 2 * N))),
        case("hyperbola_near_degenerate_x2_y2_0.01", &["x^2-y^2=0.01"], cat(param(|s| (0.1 * s.cosh(), 0.1 * s.sinh()), -7.0, 7.0, 80_000), param(|s| (-0.1 * s.cosh(), 0.1 * s.sinh()), -7.0, 7.0, 80_000))),
        case("hyperbola_hugging_asymptote_xy_0.001", &["xy=0.001"],
            cat(cat(param(|x| (x, 0.001 / x), 0.03, 60.0, 40_000), param(|x| (x, 0.001 / x), -60.0, -0.03, 40_000)),
            cat(param(|y| (0.001 / y, y), 0.03, 60.0, 40_000), param(|y| (0.001 / y, y), -60.0, -0.03, 40_000)))),
        // ---------- general second-degree with sliders ----------
        case("general_slider_ellipse", &["Ax^2+Bxy+Cy^2+Dx+Ey+F=0"], ellipse(0.0, 0.0, 3.0, 2.0, 0.0))
            .sl(&[("A", 4.0), ("B", 0.0), ("C", 9.0), ("D", 0.0), ("E", 0.0), ("F", -36.0)]),
        case("general_slider_parabola", &["Ax^2+Bxy+Cy^2+Dx+Ey+F=0"],
            param(|u| rot45(u, -2f64.sqrt() * u * u), -12.0, 12.0, 60_000))
            .sl(&[("A", 1.0), ("B", 2.0), ("C", 1.0), ("D", 1.0), ("E", -1.0), ("F", 0.0)]),
        case("general_slider_hyperbola", &["Ax^2+Bxy+Cy^2+Dx+Ey+F=0"], hyper_x(0.0, 0.0, 2.0, 3.0))
            .sl(&[("A", 9.0), ("B", 0.0), ("C", -4.0), ("D", 0.0), ("E", 0.0), ("F", -36.0)]),
        // ---------- parametric ----------
        case("param_ellipse_cos_sin", &["(3cos(t), 2sin(t))"], ellipse(0.0, 0.0, 3.0, 2.0, 0.0)),
        case("param_ellipse_slider_ab", &["(a cos(t), b sin(t))"], ellipse(0.0, 0.0, 5.0, 2.0, 0.0)).sl(&[("a", 5.0), ("b", 2.0)]),
        case("param_parabola_t2_2t_full", &["(t^2, 2t) {-3.5<=t<=3.5}"], param(|t| (t * t, 2.0 * t), -3.5, 3.5, 40_000)),
        case("param_parabola_workaround_shift_t", &["((t-pi)^2, 2(t-pi))"], param(|t| (t * t, 2.0 * t), -PI, PI, 40_000)),
        case("param_hyperbola_sec_tan", &["(2sec(t), 3tan(t))"], hyper_x(0.0, 0.0, 2.0, 3.0)),
        case("param_two_equations_x_eq_cos_t", &["x=3cos(t)", "y=2sin(t)"], ellipse(0.0, 0.0, 3.0, 2.0, 0.0)),
        // ---------- polar conics ----------
        case("polar_ellipse_e0.5", &["r=2/(1+0.5cos(theta))"], polar(|t| 2.0 / (1.0 + 0.5 * t.cos()), 0.0, 2.0 * PI)),
        case("polar_parabola_e1", &["r=2/(1+cos(theta))"], polar(|t| 2.0 / (1.0 + t.cos()), 1e-4, 2.0 * PI - 1e-4)),
        case("polar_hyperbola_e2", &["r=2/(1+2cos(theta))"], polar(|t| 2.0 / (1.0 + 2.0 * t.cos()), 0.0, 2.0 * PI)),
        case("polar_ellipse_sin", &["r=3/(1+0.5sin(theta))"], polar(|t| 3.0 / (1.0 + 0.5 * t.sin()), 0.0, 2.0 * PI)),
        case("polar_parabola_minus_cos", &["r=1.5/(1-cos(theta))"], polar(|t| 1.5 / (1.0 - t.cos()), 1e-4, 2.0 * PI - 1e-4)),
        case("polar_slider_e_d", &["r=E D/(1+E cos(theta))"], polar(|t| 3.0 / (1.0 + 0.5 * t.cos()), 0.0, 2.0 * PI)).sl(&[("E", 0.5), ("D", 6.0)]),
        // ---------- degenerate ----------
        case("degenerate_point_x2_y2_0", &["x^2+y^2=0"], vec![(0.0, 0.0)]),
        case("degenerate_point_shifted", &["(x-0.37)^2+(y-0.21)^2=0"], vec![(0.37, 0.21)]),
        case("degenerate_line_pair_x2_minus_y2", &["x^2-y^2=0"], cat(line(|t| (t, t)), line(|t| (t, -t)))),
        case("degenerate_line_pair_factored", &["(x-y)(x+2y-3)=0"], cat(line(|t| (t, t)), line(|t| (t, (3.0 - t) / 2.0)))),
        case("degenerate_parallel_lines_x2_4", &["x^2=4"], cat(line(|t| (2.0, t)), line(|t| (-2.0, t)))),
        case("degenerate_double_line_x_minus_y_sq", &["(x-y)^2=0"], line(|t| (t, t))),
        case("degenerate_double_line_x2", &["x^2=0"], line(|t| (0.0, t))),
        case("degenerate_double_line_shifted", &["(x-0.37)^2=0"], line(|t| (0.37, t))),
        case("degenerate_empty_x2_y2_eq_neg1", &["x^2+y^2=-1"], vec![]).empty(),
        case("degenerate_empty_ellipse", &["x^2/4+y^2/9+1=0"], vec![]).empty(),
        // ---------- inequalities (boundary is a contour, shading is a GPU field) ----------
        case("ineq_disk_lt", &["x^2+y^2<4"], ellipse(0.0, 0.0, 2.0, 2.0, 0.0)).fill(false).dashed(),
        case("ineq_ellipse_exterior_ge", &["x^2/9+y^2/4>=1"], ellipse(0.0, 0.0, 3.0, 2.0, 0.0)).fill(true),
        case("ineq_parabola_above", &["y>x^2"], parab_y(|x| x * x)).fill(true).dashed(),
        case("ineq_hyperbola_xy_gt1", &["xy>1"],
            cat(param(|x| (x, 1.0 / x), 0.02, 60.0, 80_000), param(|x| (x, 1.0 / x), -60.0, -0.02, 80_000))).fill(true).dashed(),
        case("ineq_annulus_chain", &["1<x^2+y^2<=4"], cat(ellipse(0.0, 0.0, 1.0, 1.0, 0.0), ellipse(0.0, 0.0, 2.0, 2.0, 0.0))).fill(false).dashed(),
        case("ineq_disk_le_solid_boundary", &["x^2+y^2<=4"], ellipse(0.0, 0.0, 2.0, 2.0, 0.0)).fill(false),
        case("ineq_empty_disk", &["x^2+y^2<-1"], vec![]).empty().fill(false),
    ]
}

// ---- runner -------------------------------------------------------------------------------

struct Out {
    kind: String,
    diags: Vec<String>,
    nseg: usize,
    fields: String,
    extras: usize,
    drawn_in: usize,
    miss: usize,
    checked: usize,
    pass: bool,
    why: String,
}

fn run(c: &Case) -> Out {
    let doc = make_doc(&c.lines, &c.sliders);
    let win = Window3::new([c.win[0], c.win[2], -1.0], [c.win[1], c.win[3], 1.0]);
    let (g, origin) = build(&doc, Mode::D2, win);
    let px = (c.win[1] - c.win[0]) / VIEWPORT.0 as f64;
    let tol = 3.0 * px;
    let (xmin, xmax, ymin, ymax) = (c.win[0], c.win[1], c.win[2], c.win[3]);
    // drawn points (absolute), segments subdivided to <= px/2 so recall is not limited by length
    let mut drawn: Vec<(f64, f64)> = Vec::new();
    for s in &g.segments {
        let a = (s.p0[0] as f64 + origin[0], s.p0[1] as f64 + origin[1]);
        let b = (s.p1[0] as f64 + origin[0], s.p1[1] as f64 + origin[1]);
        let len = (a.0 - b.0).hypot(a.1 - b.1);
        let n = ((len / (0.5 * px)).ceil() as usize).clamp(1, 4000);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            drawn.push((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
        }
    }
    let in_win = |p: &(f64, f64), m: f64| p.0 > xmin + m && p.0 < xmax - m && p.1 > ymin + m && p.1 < ymax - m;
    let oracle_in: Vec<(f64, f64)> = c.oracle.iter().copied().filter(|p| in_win(p, -tol)).collect();
    let og = Grid2::new(&oracle_in, 2.0 * tol);
    let dg = Grid2::new(&drawn, 2.0 * tol);
    let mut extras = 0;
    let mut drawn_in = 0;
    for p in drawn.iter().filter(|p| in_win(p, 0.0)) {
        drawn_in += 1;
        if !og.near(*p, tol) {
            extras += 1;
        }
    }
    let mut miss = 0;
    let mut checked = 0;
    for p in oracle_in.iter().filter(|p| in_win(p, tol)) {
        checked += 1;
        if !dg.near(*p, tol) {
            miss += 1;
        }
    }
    let diags: Vec<String> = g.diagnostics.iter().map(|d| d.1.clone()).collect();
    let fields = field_desc(&g);
    let mut why = String::new();
    let mut pass = diags.is_empty();
    if !diags.is_empty() {
        why += &format!("diag:{:?} ", diags);
    }
    if c.expect_empty {
        if !g.segments.is_empty() {
            pass = false;
            why += "drew something for an empty set ";
        }
    } else {
        if g.segments.is_empty() {
            pass = false;
            why += "nothing drawn ";
        }
        if extras > 0 {
            pass = false;
            why += &format!("{extras} drawn pts off the curve ");
        }
        if c.dashed {
            // strict inequality: boundary is dashed on purpose, so ~half is uncovered
            let f = miss as f64 / checked.max(1) as f64;
            if !(0.1..0.8).contains(&f) {
                pass = false;
                why += &format!("expected a dashed boundary, uncovered fraction {f:.2} ");
            }
        } else if miss > 0 {
            pass = false;
            why += &format!("{miss}/{checked} true pts uncovered ");
        }
    }
    if let Some(want) = c.fill {
        let ok = matches!(g.fields.as_slice(), [f] if f.kind == math_playground_lib::geometry::FieldKind::Fill { greater: want });
        if !ok {
            pass = false;
            why += &format!("fill field mismatch (want greater={want}, got [{fields}]) ");
        }
    }
    Out { kind: kind_name(c.lines[0]), diags, nseg: g.segments.len(), fields, extras, drawn_in, miss, checked, pass, why }
}

fn by_name(n: &str) -> Case {
    cases().into_iter().find(|c| c.name == n).unwrap_or_else(|| panic!("no case {n}"))
}

#[test]
fn param_parabola_with_t_range() {
    let o = run(&by_name("param_parabola_t2_2t_full"));
    assert!(o.pass, "{}", o.why);
}

#[test]
fn param_two_equations_are_one_curve() {
    let o = run(&by_name("param_two_equations_x_eq_cos_t"));
    assert!(o.pass, "{}", o.why);
}

/// Everything not marked as a gap must pass (this is the regression suite).
#[test]
fn conic_supported_cases_pass() {
    let mut bad = Vec::new();
    for c in cases().iter().filter(|c| c.gap.is_none()) {
        let o = run(c);
        if !o.pass {
            bad.push(format!("{}: {}", c.name, o.why));
        }
    }
    assert!(bad.is_empty(), "regressions:\n{}", bad.join("\n"));
}

/// Prints the whole matrix. `cargo test --test conic_audit conic_report -- --ignored --nocapture`
#[test]
#[ignore = "report only; run with --ignored --nocapture"]
fn conic_report() {
    println!("{:<44} {:<12} {:>6} {:<12} {:>7} {:>9} {}", "case", "kind", "segs", "fields", "extra", "miss", "verdict");
    for c in cases() {
        let o = run(&c);
        println!(
            "{:<44} {:<12} {:>6} {:<12} {:>3}/{:<4} {:>4}/{:<4} {} {}  [{}]",
            c.name, o.kind, o.nseg, o.fields, o.extras, o.drawn_in, o.miss, o.checked,
            if o.pass { "PASS" } else { "FAIL" }, o.why, c.lines.join(" ; ")
        );
        let _ = &o.diags;
    }
}

macro_rules! gap {
    ($test:ident, $name:literal, $why:literal) => {
        #[test]
        #[ignore = $why]
        fn $test() {
            let o = run(&by_name($name));
            assert!(o.pass, "{}", o.why);
        }
    };
}

// GAP: each of these fails today; run with `-- --ignored` to see why. Remove the ignore when fixed.
