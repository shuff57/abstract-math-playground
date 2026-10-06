//! Mesher benchmark (before/after): the legacy mesher (verbatim copy of the pre-optimisation
//! code in `tests/legacy_mesh`) against the current `surface_3d`, plus the explicit-surface
//! height-field mesher.
//!
//!   cargo test -p math_core --release --test mesh_bench -- --ignored --nocapture
mod legacy_mesh;
use legacy_mesh::legacy_surface_3d;
use math_core::compile::{compile, Angle};
use math_core::mesh::surface_3d;
use math_core::parse::parse;

fn prog(src: &str) -> math_core::Program {
    compile(&parse(src).unwrap(), &["x", "y", "z"], Angle::Rad).unwrap()
}

/// CPU time of this thread in ms (immune to load from other processes).
fn cpu_ms() -> f64 {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    s.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1e6
}

fn best<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..3 {
        let t = cpu_ms();
        let r = f();
        best = best.min(cpu_ms() - t);
        out = Some(r);
    }
    (out.unwrap(), best)
}

/// Largest and mean |F| / |grad F| (distance estimate) over the vertices, in cells.
fn residual(p: &math_core::Program, pos: &[[f32; 3]], cell: f64) -> (f64, f64) {
    let (mut mx, mut sum) = (0.0f64, 0.0f64);
    let h = cell * 1e-3;
    for q in pos {
        let q = [q[0] as f64, q[1] as f64, q[2] as f64];
        let f = p.eval(&q);
        let mut g2 = 0.0;
        for a in 0..3 {
            let (mut u, mut d) = (q, q);
            u[a] += h;
            d[a] -= h;
            let gv = (p.eval(&u) - p.eval(&d)) / (2.0 * h);
            g2 += gv * gv;
        }
        let r = if g2 > 1e-12 { f.abs() / g2.sqrt() / cell } else { 0.0 };
        mx = mx.max(r);
        sum += r;
    }
    (mx, sum / pos.len().max(1) as f64)
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn surface_bench() {
    let cases = [
        ("sphere", "x^2+y^2+z^2-16"),
        ("hyperboloid", "x^2/4+y^2/4-z^2/9-1"),
        ("saddle z=x^2/4-y^2/4", "z-(x^2/4-y^2/4)"),
        ("cone", "x^2+y^2-z^2"),
        ("gyroid", "sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)"),
        ("torus", "(sqrt(x^2+y^2)-3)^2+z^2-1"),
    ];
    for depth in [7u32, 8] {
        let cell = 12.0 / (1u64 << depth) as f64;
        for (name, src) in cases {
            let p = prog(src);
            let (old, t_old) = best(|| legacy_surface_3d(&p, [-6.0; 3], [6.0; 3], depth, 300_000));
            let (new, t_new) = best(|| surface_3d(&p, [-6.0; 3], [6.0; 3], depth, 300_000));
            let (om, oa) = residual(&p, &old.positions, cell);
            let (nm, na) = residual(&p, &new.positions, cell);
            eprintln!(
                "BENCH depth {depth} {name:22} old {t_old:7.0} ms  new {t_new:7.0} ms  x{:4.1}  tris {:7} -> {:7}  vertex error (cells) max {om:.1e} -> {nm:.1e}, mean {oa:.1e} -> {na:.1e}",
                t_old / t_new,
                old.indices.len() / 3,
                new.indices.len() / 3
            );
        }
    }
}


/// Explicit surfaces `z = f(x, y)`: the legacy implicit mesher on `z - f` against the height field.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn explicit_bench() {
    use math_core::mesh::height_surface;
    let cases = [
        ("z=x^2/4-y^2/4", "z-(x^2/4-y^2/4)", "x^2/4-y^2/4"),
        ("z=sin(x)cos(y)", "z-sin(x)cos(y)", "sin(x)cos(y)"),
        ("z=sin(sqrt(x^2+y^2))", "z-sin(sqrt(x^2+y^2))", "sin(sqrt(x^2+y^2))"),
    ];
    for depth in [7u32, 8] {
        for (name, implicit, rhs) in cases {
            let p = prog(implicit);
            let h = compile(&parse(rhs).unwrap(), &["x", "y"], Angle::Rad).unwrap();
            let (old, t_old) = best(|| legacy_surface_3d(&p, [-6.0; 3], [6.0; 3], depth, 300_000));
            let (new, t_new) = best(|| height_surface(&h, [0, 1, 2], [-6.0; 3], [6.0; 3], depth));
            eprintln!(
                "BENCH depth {depth} {name:22} legacy implicit {t_old:7.0} ms  height field {t_new:7.0} ms  x{:5.1}  tris {:7} -> {:7}",
                t_old / t_new,
                old.indices.len() / 3,
                new.indices.len() / 3
            );
        }
    }
}
