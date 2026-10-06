//! 3D audit: builds the real 3D scene geometry (the mesher output the renderer would draw) for
//! the exact text a user would type and checks it against an analytic oracle:
//!   precision: every mesh vertex lies within ~2 cells of the true surface (no spurious sheets),
//!   recall:    every true surface point inside the window box has a mesh vertex nearby (no holes
//!              or missing sheets),
//!   closed:    a closed solid fully inside the window has no open (boundary) mesh edges,
//!   winding:   triangle winding agrees with the vertex normals.
//! Cases with `gap: Some(..)` are known failures, mirrored by `#[ignore = "GAP: ..."]` tests.
//!
//!   cargo test -p math_playground --release --test solid_audit solid_report -- --ignored --nocapture
mod common;
use common::*;
use math_core::view::{Mode, Window3};
use std::collections::HashMap;
use std::f64::consts::PI;

type P3 = [f64; 3];
struct Case {
    name: &'static str,
    lines: Vec<&'static str>,
    sliders: Vec<(&'static str, f64)>,
    win: [f64; 6],
    oracle: Vec<P3>,
    f: Option<fn(f64, f64, f64) -> f64>,
    closed: bool,
    expect_empty: bool,
    curve: bool,
    gap: Option<&'static str>,
}
const W6: [f64; 6] = [-6.0, 6.0, -6.0, 6.0, -6.0, 6.0];

fn case(name: &'static str, lines: &[&'static str], oracle: Vec<P3>) -> Case {
    Case { name, lines: lines.to_vec(), sliders: vec![], win: W6, oracle, f: None, closed: false, expect_empty: false, curve: false, gap: None }
}
impl Case {
    fn f(mut self, f: fn(f64, f64, f64) -> f64) -> Self {
        self.f = Some(f);
        self
    }
    fn closed(mut self) -> Self {
        self.closed = true;
        self
    }
    fn empty(mut self) -> Self {
        self.expect_empty = true;
        self
    }
    fn curve(mut self) -> Self {
        self.curve = true;
        self
    }
    fn win(mut self, w: [f64; 6]) -> Self {
        self.win = w;
        self
    }
    fn gap(mut self, why: &'static str) -> Self {
        self.gap = Some(why);
        self
    }
}

fn surf(n: usize, u0: f64, u1: f64, v0: f64, v1: f64, f: impl Fn(f64, f64) -> P3) -> Vec<P3> {
    let mut o = Vec::with_capacity((n + 1) * (n + 1));
    for i in 0..=n {
        for j in 0..=n {
            let p = f(u0 + (u1 - u0) * i as f64 / n as f64, v0 + (v1 - v0) * j as f64 / n as f64);
            if p.iter().all(|c| c.is_finite()) {
                o.push(p);
            }
        }
    }
    o
}
fn graph(n: usize, f: impl Fn(f64, f64) -> f64) -> Vec<P3> {
    surf(n, -8.0, 8.0, -8.0, 8.0, |x, y| [x, y, f(x, y)])
}
fn curve(n: usize, t0: f64, t1: f64, f: impl Fn(f64) -> P3) -> Vec<P3> {
    (0..=n).map(|i| f(t0 + (t1 - t0) * i as f64 / n as f64)).collect()
}
fn sphere(c: P3, r: f64) -> Vec<P3> {
    surf(500, 0.0, PI, 0.0, 2.0 * PI, |a, b| [c[0] + r * a.sin() * b.cos(), c[1] + r * a.sin() * b.sin(), c[2] + r * a.cos()])
}
fn ellipsoid(a: f64, b: f64, c: f64) -> Vec<P3> {
    surf(500, 0.0, PI, 0.0, 2.0 * PI, |p, t| [a * p.sin() * t.cos(), b * p.sin() * t.sin(), c * p.cos()])
}
fn torus(big: f64, small: f64) -> Vec<P3> {
    surf(600, 0.0, 2.0 * PI, 0.0, 2.0 * PI, |u, v| [(big + small * v.cos()) * u.cos(), (big + small * v.cos()) * u.sin(), small * v.sin()])
}
fn cat(mut a: Vec<P3>, b: Vec<P3>) -> Vec<P3> {
    a.extend(b);
    a
}

fn cases() -> Vec<Case> {
    vec![
        // ---------- planes ----------
        case("plane_z_eq_ax_by_c", &["z=0.5x-0.3y+1"], graph(300, |x, y| 0.5 * x - 0.3 * y + 1.0)).f(|x, y, z| z - 0.5 * x + 0.3 * y - 1.0),
        case("plane_general_ax_by_cz_d", &["x+2y+3z=6"], graph(300, |x, y| (6.0 - x - 2.0 * y) / 3.0)),
        case("plane_vertical_no_z", &["2x+3y=6"], surf(300, -8.0, 8.0, -8.0, 8.0, |x, z| [x, (6.0 - 2.0 * x) / 3.0, z])),
        case("plane_z_const", &["z=3"], graph(200, |_, _| 3.0)),
        case("plane_coordinate_x_eq_1", &["x=1"], surf(200, -8.0, 8.0, -8.0, 8.0, |y, z| [1.0, y, z])),
        // ---------- spheres / ellipsoids ----------
        case("sphere_origin", &["x^2+y^2+z^2=16"], sphere([0.0; 3], 4.0)).closed(),
        case("sphere_shifted_vertex_form", &["(x-1)^2+(y+2)^2+(z-1)^2=9"], sphere([1.0, -2.0, 1.0], 3.0)).closed(),
        case("sphere_general_form", &["x^2+y^2+z^2-2x+4y-6z-2=0"], sphere([1.0, -2.0, 3.0], 4.0)),
        case("sphere_default_window_pm10", &["x^2+y^2+z^2=36"], sphere([0.0; 3], 6.0)).closed().win([-10.0; 1].iter().chain(&[10.0, -10.0, 10.0, -10.0, 10.0]).copied().collect::<Vec<_>>().try_into().unwrap()),
        case("ellipsoid", &["x^2/9+y^2/4+z^2/16=1"], ellipsoid(3.0, 2.0, 4.0)).closed(),
        case("ellipsoid_thin", &["x^2/25+y^2/0.25+z^2/0.25=1"], ellipsoid(5.0, 0.5, 0.5)).closed(),
        // ---------- paraboloids ----------
        case("paraboloid_elliptic_z_eq", &["z=x^2/4+y^2/9"], graph(400, |x, y| x * x / 4.0 + y * y / 9.0)),
        case("paraboloid_implicit", &["x^2+y^2=z"], graph(400, |x, y| x * x + y * y)),
        case("paraboloid_hyperbolic_saddle", &["z=x^2/4-y^2/4"], graph(400, |x, y| x * x / 4.0 - y * y / 4.0)),
        // ---------- hyperboloids ----------
        case("hyperboloid_one_sheet", &["x^2/4+y^2/4-z^2/9=1"], surf(500, -3.0, 3.0, 0.0, 2.0 * PI, |v, u| [2.0 * v.cosh() * u.cos(), 2.0 * v.cosh() * u.sin(), 3.0 * v.sinh()])),
        case("hyperboloid_two_sheets", &["z^2/4-x^2-y^2=1"], cat(
            surf(500, 0.0, 3.0, 0.0, 2.0 * PI, |v, u| [v.sinh() * u.cos(), v.sinh() * u.sin(), 2.0 * v.cosh()]),
            surf(500, 0.0, 3.0, 0.0, 2.0 * PI, |v, u| [v.sinh() * u.cos(), v.sinh() * u.sin(), -2.0 * v.cosh()]))),
        // ---------- cones ----------
        case("cone_double_circular", &["x^2+y^2=z^2"], surf(500, -6.0, 6.0, 0.0, 2.0 * PI, |z, u| [z * u.cos(), z * u.sin(), z])),
        case("cone_single_nappe_sqrt", &["z=sqrt(x^2+y^2)"], surf(500, 0.0, 6.0, 0.0, 2.0 * PI, |z, u| [z * u.cos(), z * u.sin(), z])),
        case("cone_elliptic_double", &["x^2/4+y^2=z^2/9"], surf(500, -6.0, 6.0, 0.0, 2.0 * PI, |z, u| [2.0 * z / 3.0 * u.cos(), z / 3.0 * u.sin(), z])),
        // ---------- cylinders ----------
        case("cylinder_circular", &["x^2+y^2=4"], surf(400, -8.0, 8.0, 0.0, 2.0 * PI, |z, u| [2.0 * u.cos(), 2.0 * u.sin(), z])),
        case("cylinder_elliptic", &["x^2/9+y^2/4=1"], surf(400, -8.0, 8.0, 0.0, 2.0 * PI, |z, u| [3.0 * u.cos(), 2.0 * u.sin(), z])),
        case("cylinder_parabolic", &["y=x^2/2"], surf(400, -8.0, 8.0, -4.0, 4.0, |z, x| [x, x * x / 2.0, z])),
        case("cylinder_hyperbolic", &["x^2/4-y^2=1"], cat(
            surf(400, -8.0, 8.0, -3.0, 3.0, |z, s| [2.0 * s.cosh(), s.sinh(), z]),
            surf(400, -8.0, 8.0, -3.0, 3.0, |z, s| [-2.0 * s.cosh(), s.sinh(), z]))),
        // ---------- z = f(x,y) ----------
        case("surf_sin_x_cos_y", &["z=sin(x)cos(y)"], graph(500, |x, y| x.sin() * y.cos())),
        case("surf_radial_ripple", &["z=sin(sqrt(x^2+y^2))"], graph(500, |x, y| (x * x + y * y).sqrt().sin())),
        case("surf_abs_crease", &["z=abs(x)+abs(y)"], graph(400, |x, y| x.abs() + y.abs())),
        case("surf_gaussian", &["z=3exp(-(x^2+y^2)/4)"], graph(400, |x, y| 3.0 * (-(x * x + y * y) / 4.0).exp())),
        // Oracles for the steep parts are parametrised along the steep direction (a graph sampled on
        // an x/y grid is too sparse near the rim / the poles, where the surface is almost vertical).
        case("surf_hemisphere_domain_hole", &["z=sqrt(16-x^2-y^2)"], surf(700, 0.0, PI / 2.0, 0.0, 2.0 * PI, |t, u| [4.0 * t.sin() * u.cos(), 4.0 * t.sin() * u.sin(), 4.0 * t.cos()])),
        case("surf_tan_discontinuous", &["z=tan(x)"], (-3..=3).flat_map(|k| surf(500, -6.0, 6.0, -8.0, 8.0, move |z, y| [z.atan() + k as f64 * PI, y, z])).collect()),
        case("surf_reciprocal_pole", &["z=1/(x^2+y^2)"], graph(500, |x, y| 1.0 / (x * x + y * y))),
        case("surf_floor_steps", &["z=floor(x)"], graph(400, |x, _| x.floor())),
        case("surf_slider_a", &["z=a*sin(x)*cos(y)"], graph(500, |x, y| 2.0 * x.sin() * y.cos())).sl_(&[("a", 2.0)]),
        // ---------- implicit ----------
        case("implicit_torus_sqrt", &["(sqrt(x^2+y^2)-3)^2+z^2=1"], torus(3.0, 1.0)).closed(),
        case("implicit_torus_polynomial", &["(x^2+y^2+z^2+8)^2=36(x^2+y^2)"], torus(3.0, 1.0)).closed(),
        case("implicit_gyroid", &["sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=0"], vec![]).f(|x, y, z| x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()),
        case("implicit_quartic_superellipsoid", &["x^4+y^4+z^4=16"], vec![]).f(|x, y, z| x.powi(4) + y.powi(4) + z.powi(4) - 16.0).closed(),
        case("implicit_heart", &["(x^2+9/4y^2+z^2-1)^3-x^2z^3-9/80y^2z^3=0"], vec![])
            .f(|x, y, z| (x * x + 2.25 * y * y + z * z - 1.0).powi(3) - x * x * z.powi(3) - 9.0 / 80.0 * y * y * z.powi(3))
            .win([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]).closed(),
        case("implicit_revolution_vase", &["x^2+y^2=(2+sin(z))^2"], vec![]).f(|x, y, z| x * x + y * y - (2.0 + z.sin()).powi(2)),
        case("implicit_point_shifted", &["(x-0.37)^2+(y-0.21)^2+(z-0.11)^2=0"], vec![[0.37, 0.21, 0.11]]),
        case("implicit_line_shifted", &["(x-0.37)^2+(y-0.21)^2=0"], surf(400, -8.0, 8.0, 0.0, 0.0, |z, _| [0.37, 0.21, z])),
        case("implicit_double_sphere_shell", &["(x^2+y^2+z^2-9)^2=0"], sphere([0.0; 3], 3.0)),
        case("implicit_empty_sphere_neg", &["x^2+y^2+z^2=-1"], vec![]).empty(),
        // ---------- parametric (space curves) ----------
        case("curve_helix_one_turn", &["(cos(t), sin(t), t/3)"], curve(4000, 0.0, 2.0 * PI, |t| [t.cos(), t.sin(), t / 3.0])).curve(),
        case("curve_helix_many_turns", &["(2cos(t), 2sin(t), t/4-6) {0<=t<=6pi}"], curve(4000, 0.0, 6.0 * PI, |t| [2.0 * t.cos(), 2.0 * t.sin(), t / 4.0 - 6.0])).curve(),
        case("curve_trefoil", &["(sin(t)+2sin(2t), cos(t)-2cos(2t), -sin(3t))"], curve(4000, 0.0, 2.0 * PI, |t| [t.sin() + 2.0 * (2.0 * t).sin(), t.cos() - 2.0 * (2.0 * t).cos(), -(3.0 * t).sin()])).curve(),
        case("curve_line_3d", &["(t-3, 2t-6, t/2)"], curve(2000, 0.0, 2.0 * PI, |t| [t - 3.0, 2.0 * t - 6.0, t / 2.0])).curve(),
        // ---------- parametric surfaces (x(u,v),y(u,v),z(u,v)) ----------
        case("psurf_sphere_uv", &["(3cos(u)cos(v), 3cos(u)sin(v), 3sin(u))"], sphere([0.0; 3], 3.0)).closed(),
        case("psurf_torus_uv", &["((3+cos(v))cos(u), (3+cos(v))sin(u), sin(v))"], torus(3.0, 1.0)).closed(),
        case("psurf_helicoid_uv", &["(v cos(u), v sin(u), u/2) {-6<=u<=6, -5<=v<=5}"], surf(300, -6.0, 6.0, -5.0, 5.0, |u, v| [v * u.cos(), v * u.sin(), u / 2.0])),
        case("psurf_mobius_uv", &["((2+v cos(u/2))cos(u), (2+v cos(u/2))sin(u), v sin(u/2)) {-1<=v<=1}"], surf(400, 0.0, 2.0 * PI, -1.0, 1.0, |u, v| [(2.0 + v * (u / 2.0).cos()) * u.cos(), (2.0 + v * (u / 2.0).cos()) * u.sin(), v * (u / 2.0).sin()])),
        // ---------- other 3D item kinds ----------
        case("ineq3d_ball_boundary_only", &["x^2+y^2+z^2<9"], sphere([0.0; 3], 3.0)).closed(),
        case("ineq3d_xy_only_extruded", &["x^2+y^2<4"], surf(300, -6.0, 6.0, 0.0, 2.0 * PI, |z, u| [2.0 * u.cos(), 2.0 * u.sin(), z])),
        case("point_3d", &["(1,2,3)"], vec![[1.0, 2.0, 3.0]]).curve(),
        case("point_list_3d", &["[(0,0,0),(1,1,1),(2,-1,3)]"], vec![[0.0; 3], [1.0, 1.0, 1.0], [2.0, -1.0, 3.0]]).curve(),
        case("vector_field_3d_swirl", &["(-y,x,0)"], vec![]).curve(),
        case("vector_field_3d_radial", &["(x,y,z)"], vec![]).curve(),
        case("two_surfaces_no_intersection_curve", &["x^2+y^2+z^2=9", "z=1"], cat(sphere([0.0; 3], 3.0), graph(300, |_, _| 1.0))),
        // ---------- windows ----------
        case("window_huge_sphere", &["x^2+y^2+z^2=1000000"], sphere([0.0; 3], 1000.0)).closed().win([-1500.0, 1500.0, -1500.0, 1500.0, -1500.0, 1500.0]),
        case("window_tiny_sphere", &["x^2+y^2+z^2=0.000001"], sphere([0.0; 3], 0.001)).closed().win([-0.0015, 0.0015, -0.0015, 0.0015, -0.0015, 0.0015]),
        case("window_sphere_far_from_origin", &["(x-1000)^2+(y-1000)^2+(z-1000)^2=4"], sphere([1000.0; 3], 2.0)).closed().win([994.0, 1006.0, 994.0, 1006.0, 994.0, 1006.0]),
        case("window_sphere_clipped_by_box", &["x^2+y^2+z^2=64"], sphere([0.0; 3], 8.0)),
    ]
}

trait SlExt {
    fn sl_(self, s: &[(&'static str, f64)]) -> Self;
}
impl SlExt for Case {
    fn sl_(mut self, s: &[(&'static str, f64)]) -> Self {
        self.sliders = s.to_vec();
        self
    }
}

// ---- runner -------------------------------------------------------------------------------

struct Out {
    kind: String,
    verts: usize,
    tris: usize,
    segs: usize,
    ms: u128,
    far: usize,
    miss: usize,
    checked: usize,
    open_edges: usize,
    flips: usize,
    bad_normals: usize,
    pass: bool,
    why: String,
}

fn run(c: &Case) -> Out {
    let doc = make_doc(&c.lines, &c.sliders);
    let win = Window3::new([c.win[0], c.win[2], c.win[4]], [c.win[1], c.win[3], c.win[5]]);
    let t0 = std::time::Instant::now();
    let (g, o) = build(&doc, Mode::D3, win);
    let ms = t0.elapsed().as_millis();
    let span = (c.win[1] - c.win[0]).max(c.win[3] - c.win[2]).max(c.win[5] - c.win[4]);
    let cell = span / 128.0;
    let tol = 2.0 * cell;
    let inside = |p: &P3, m: f64| (0..3).all(|a| p[a] > c.win[2 * a] + m && p[a] < c.win[2 * a + 1] - m);
    let pts: Vec<P3> = if c.curve {
        let mut v = Vec::new();
        for s in &g.segments {
            let a = [s.p0[0] as f64 + o[0], s.p0[1] as f64 + o[1], s.p0[2] as f64 + o[2]];
            let b = [s.p1[0] as f64 + o[0], s.p1[1] as f64 + o[1], s.p1[2] as f64 + o[2]];
            let len = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let n = ((len / (0.25 * cell)).ceil() as usize).clamp(1, 500);
            for i in 0..=n {
                let t = i as f64 / n as f64;
                v.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
            }
        }
        v
    } else {
        g.vertices.iter().map(|v| [v.pos[0] as f64 + o[0], v.pos[1] as f64 + o[1], v.pos[2] as f64 + o[2]]).collect()
    };
    let mut far = 0;
    let mut miss = 0;
    let mut checked = 0;
    if !c.oracle.is_empty() {
        let og = Grid3::new(&c.oracle, 2.0 * tol);
        let vg = Grid3::new(&pts, 2.0 * tol);
        far = pts.iter().filter(|p| inside(p, -0.0) && !og.near(**p, tol)).count();
        for p in c.oracle.iter().filter(|p| inside(p, tol)) {
            checked += 1;
            if !vg.near(*p, tol * 1.2) {
                miss += 1;
            }
        }
    }
    if let Some(f) = c.f {
        // residual distance |F|/|grad F| for precision when there is no point oracle
        for p in pts.iter().filter(|p| inside(p, 0.0)) {
            let h = cell * 0.05;
            let g3 = [
                (f(p[0] + h, p[1], p[2]) - f(p[0] - h, p[1], p[2])) / (2.0 * h),
                (f(p[0], p[1] + h, p[2]) - f(p[0], p[1] - h, p[2])) / (2.0 * h),
                (f(p[0], p[1], p[2] + h) - f(p[0], p[1], p[2] - h)) / (2.0 * h),
            ];
            let gl = (g3[0] * g3[0] + g3[1] * g3[1] + g3[2] * g3[2]).sqrt();
            if gl > 1e-6 && (f(p[0], p[1], p[2]).abs() / gl) > tol * 0.5 {
                far += 1;
            }
        }
    }
    // topology / winding
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut flips = 0;
    for t in g.indices.chunks(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
        let v: Vec<_> = t.iter().map(|&i| &g.vertices[i as usize]).collect();
        let (e1, e2) = (
            [v[1].pos[0] - v[0].pos[0], v[1].pos[1] - v[0].pos[1], v[1].pos[2] - v[0].pos[2]],
            [v[2].pos[0] - v[0].pos[0], v[2].pos[1] - v[0].pos[1], v[2].pos[2] - v[0].pos[2]],
        );
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let vn = [
            v[0].normal[0] + v[1].normal[0] + v[2].normal[0],
            v[0].normal[1] + v[1].normal[1] + v[2].normal[1],
            v[0].normal[2] + v[1].normal[2] + v[2].normal[2],
        ];
        if n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] < 0.0 {
            flips += 1;
        }
    }
    let eps = cell * 0.02;
    let on_box = |i: u32| {
        let p = &g.vertices[i as usize].pos;
        (0..3).any(|a| ((p[a] as f64 + o[a]) - c.win[2 * a]).abs() < eps || ((p[a] as f64 + o[a]) - c.win[2 * a + 1]).abs() < eps)
    };
    let open_edges = edges.iter().filter(|(k, n)| **n == 1 && !(on_box(k.0) && on_box(k.1))).count();
    let bad_normals = g
        .vertices
        .iter()
        .filter(|v| {
            let l = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            !(l.is_finite() && (l - 1.0).abs() < 0.01)
        })
        .count();
    let diags: Vec<String> = g.diagnostics.iter().map(|d| d.1.clone()).collect();
    let mut why = String::new();
    let mut pass = diags.is_empty();
    if !diags.is_empty() {
        why += &format!("diag:{diags:?} ");
    }
    let drawn = if c.curve { !g.segments.is_empty() } else { !g.indices.is_empty() };
    if c.expect_empty {
        if drawn {
            pass = false;
            why += "drew something for an empty set ";
        }
    } else {
        if !drawn {
            pass = false;
            why += "nothing drawn ";
        }
        if far > 0 {
            pass = false;
            why += &format!("{far} pts off the surface ");
        }
        if miss > 0 {
            pass = false;
            why += &format!("{miss}/{checked} true pts uncovered ");
        }
        if c.closed && open_edges > 0 {
            pass = false;
            why += &format!("{open_edges} open edges on a closed solid ");
        }
        if flips > 0 {
            pass = false;
            why += &format!("{flips} winding flips ");
        }
        if bad_normals > 0 {
            pass = false;
            why += &format!("{bad_normals} non-unit normals ");
        }
    }
    Out { kind: kind_name(c.lines[0]), verts: g.vertices.len(), tris: g.indices.len() / 3, segs: g.segments.len(), ms, far, miss, checked, open_edges, flips, bad_normals, pass, why }
}

fn by_name(n: &str) -> Case {
    cases().into_iter().find(|c| c.name == n).unwrap_or_else(|| panic!("no case {n}"))
}

#[test]
fn param_curves_and_surfaces_pass() {
    for n in ["curve_helix_many_turns", "psurf_sphere_uv", "psurf_torus_uv", "psurf_helicoid_uv", "psurf_mobius_uv"] {
        let o = run(&by_name(n));
        assert!(o.pass, "{n}: {}", o.why);
    }
}

#[test]
fn solid_supported_cases_pass() {
    let mut bad = Vec::new();
    for c in cases().iter().filter(|c| c.gap.is_none()) {
        let o = run(c);
        if !o.pass {
            bad.push(format!("{}: {}", c.name, o.why));
        }
    }
    assert!(bad.is_empty(), "regressions:\n{}", bad.join("\n"));
}

#[test]
#[ignore = "report only; run with --ignored --nocapture"]
fn solid_report() {
    println!("{:<38} {:<12} {:>7} {:>7} {:>5} {:>5} {:>6} {:>6} {:>7} {:>5} {}", "case", "kind", "verts", "tris", "segs", "ms", "far", "miss", "open", "flip", "verdict");
    for c in cases() {
        let o = run(&c);
        println!(
            "{:<38} {:<12} {:>7} {:>7} {:>5} {:>5} {:>6} {:>3}/{:<4} {:>5} {:>4} {} {} [{}]",
            c.name, o.kind, o.verts, o.tris, o.segs, o.ms, o.far, o.miss, o.checked, o.open_edges, o.flips,
            if o.pass { "PASS" } else { "FAIL" }, o.why, c.lines.join(" ; ")
        );
        let _ = o.bad_normals;
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

