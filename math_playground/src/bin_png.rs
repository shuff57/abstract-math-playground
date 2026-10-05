//! Renders expressions offscreen to PNG files, so output (including mid-transition frames) can
//! be inspected without a window.
//!
//!   render_png --mode 3d --out sphere.png "x^2+y^2+z^2=36"
//!   render_png --from 2d --to 3d --frames 6 --out-dir frames/ "y=x^2/4" "(3,2.25)"
//!   render_png --mode 2d "c:z^2-1"        (c: = complex, domain colouring)
//!   render_png --table "x_1=1,2,3;y_1=2,4,3" [--table-line] "histogram(x_1)"   (data table)
//!   render_png --slider a=0 --ticker "a -> a+0.5" --ticker-frames 8 --out t.png "y=a*x"
//!       (runs a ticker for N frames at 50 ms; writes t_before.png and t_after.png)
//!   render_png --mode 2d --angle deg --slider a=2 "y=a*sin(x)"
//!   render_png --mode 3d --window -3,-3,-3,3,3,3 --slice "z=1" --out s.png "x^2+y^2+z^2=4"
//!       (a slice: plane + exact cut curves in the scene, and the inset showing the flat graph)
//!   render_png --mode 2d --slider a=1 --slice "y=a" --out l.png "y=x^2"   (1D slice: cut line,
//!       intersection dots, inset graph of the value along the line)
//!   render_png --mode 3d --slice "z=1" --slice-view 0,2,0,1.5 --out z.png "x^2+y^2<4"
//!       (the inset's own view window xmin,xmax,ymin,ymax instead of following the main window;
//!       slices also show hue fields `sin(x)+y+z`, inequality regions, `c:` complex items and
//!       point lists `[(0,0,1),(1,2,1.02)]` near the plane)

use math_core::doc::{Doc, Item, ItemKind, SliderCfg};
use math_core::view::{Mode, Rig, Window3};
use math_playground_lib::geometry::{SceneGeometry, Theme};
use math_playground_lib::headless::{save_png, Headless};
use math_playground_lib::render::{crossfade, layer_lift, Inset, Layer};
use math_playground_lib::scene::{build_scene, build_slice_panel_view, ViewReq};
use std::path::PathBuf;

const VALUE_FLAGS: &[&str] =
    &["--mode", "--size", "--out", "--from", "--to", "--frames", "--out-dir", "--angle", "--slider", "--window", "--table", "--ticker", "--ticker-frames", "--slice", "--slice-view"];

fn parse_mode(s: &str) -> Mode {
    match s {
        "1d" => Mode::D1,
        "3d" => Mode::D3,
        _ => Mode::D2,
    }
}

fn geometry(doc: &Doc, mode: Mode, rig: &Rig, size: (u32, u32), theme: &Theme) -> SceneGeometry {
    build_scene(doc, mode, rig.window(), rig.render_origin(), size, theme)
}

/// The inset view requested with `--slice-view xmin,xmax,ymin,ymax`, for the slice's free axes.
fn view_req(doc: &Doc, mode: Mode, spec: Option<&str>) -> Result<Option<ViewReq>, String> {
    let (Some(spec), Some(cfg)) = (spec, doc.slice.as_ref()) else { return Ok(None) };
    let v: Vec<f64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let [a, b, c, d] = v[..] else { return Err("--slice-view needs 4 numbers: xmin,xmax,ymin,ymax".into()) };
    let free = (0..mode.dims() as usize).filter(|i| !cfg.fixed.contains_key(math_core::slice::axis_name(*i))).collect();
    Ok(Some(ViewReq { free, view: [a, b, c, d] }))
}

fn aspect_of(size: (u32, u32)) -> f64 {
    size.0 as f64 / size.1 as f64
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();

    let mut exprs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if VALUE_FLAGS.contains(&args[i].as_str()) {
            i += 2;
        } else if args[i].starts_with("--") {
            i += 1;
        } else {
            exprs.push(args[i].clone());
            i += 1;
        }
    }

    let size = get("--size")
        .and_then(|s| s.split_once('x').map(|(w, h)| (w.parse().unwrap_or(900), h.parse().unwrap_or(600))))
        .unwrap_or((900, 600));
    let theme = if args.iter().any(|a| a == "--dark") { Theme::dark() } else { Theme::light() };
    let window = match get("--window") {
        Some(w) => {
            let v: Vec<f64> = w.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if v.len() == 6 {
                Window3::new([v[0], v[1], v[2]], [v[3], v[4], v[5]])
            } else {
                return Err("--window needs 6 numbers: xmin,ymin,zmin,xmax,ymax,zmax".into());
            }
        }
        None => Window3::new([-10.0; 3], [10.0; 3]),
    };

    let mut doc = Doc::new_default();
    if get("--angle").as_deref() == Some("deg") {
        doc.view.angle = math_core::doc::AngleMode::Deg;
    }
    for (n, e) in exprs.iter().enumerate() {
        // A `c:` prefix marks a complex expression of z, drawn as a domain colouring.
        let (kind, src) = match e.strip_prefix("c:") {
            Some(rest) => (ItemKind::Complex, rest),
            None => (ItemKind::Equation, e.as_str()),
        };
        doc.add_item(Item::new(&format!("e{n}"), kind, src)).map_err(|e| e.to_string())?;
    }
    if let Some(spec) = get("--table") {
        // `name=cell,cell,...;name=...` : one column per `;` part.
        let cols: Vec<(&str, Vec<&str>)> =
            spec.split(';').filter_map(|c| c.split_once('=')).map(|(n, v)| (n, v.split(',').collect())).collect();
        let names: Vec<String> = cols.iter().map(|c| c.0.to_string()).collect();
        let mut t = math_core::table::Table::new(&names, 0);
        for (ci, (_, cells)) in cols.iter().enumerate() {
            for (ri, v) in cells.iter().enumerate() {
                t.set_cell(ri, ci, v.trim())?;
            }
        }
        if args.iter().any(|a| a == "--table-line") {
            t.style = math_core::table::TableStyle::Line;
        }
        let mut it = Item::new("table", ItemKind::Table, "");
        it.table = Some(t);
        doc.add_item(it).map_err(|e| e.to_string())?;
    }
    for (name, value) in args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--slider")
        .filter_map(|(i, _)| args.get(i + 1))
        .filter_map(|s| s.split_once('='))
        .filter_map(|(n, v)| v.parse::<f64>().ok().map(|v| (n.to_string(), v)))
    {
        doc.sliders.insert(name, SliderCfg { min: -10.0, max: 10.0, step: None, value });
    }

    if let Some(spec) = get("--slice") {
        let slice_mode = parse_mode(&get("--to").or_else(|| get("--mode")).unwrap_or_else(|| "2d".into()));
        let cfg = math_core::slice::parse_fixed_text(&spec)
            .and_then(|f| math_core::slice::SliceCfg::new(None, f, slice_mode))?;
        doc.slice = Some(cfg);
    }

    let vreq = view_req(&doc, parse_mode(&get("--to").or_else(|| get("--mode")).unwrap_or_else(|| "2d".into())), get("--slice-view").as_deref())?;

    if let Some(action) = get("--ticker") {
        // Drive the real App: load the doc, tick N frames, then render before and after.
        use math_playground_lib::app::{App, Command};
        let n: usize = get("--ticker-frames").and_then(|s| s.parse().ok()).unwrap_or(8);
        let mut app = App::new(size);
        app.run(Command::LoadDoc { json: math_core::doc::to_json(&doc) });
        let act = Item::new("ticker_action", ItemKind::Action, &action);
        let add = serde_json::json!({"t":"addItem","id":act.id,"kind":"action","latex":act.latex});
        for j in [
            add.to_string(),
            r#"{"t":"setTicker","action":"ticker_action","rate_ms":50,"running":true}"#.to_string(),
        ] {
            app.dispatch(&j);
        }
        let out = PathBuf::from(get("--out").unwrap_or_else(|| "out.png".into()));
        let mut gpu = Headless::new()?;
        let mode = parse_mode(&get("--mode").unwrap_or_else(|| "2d".into()));
        let mut shoot = |app: &mut App, suffix: &str| -> Result<(), String> {
            let doc = app.doc.clone();
            let mut rig = Rig::new(window, mode);
            rig.set_aspect(aspect_of(size));
            let g = geometry(&doc, mode, &rig, size, &theme);
            for (id, msg) in &g.diagnostics {
                eprintln!("diagnostic [{id}]: {msg}");
            }
            let panel = build_slice_panel_view(&doc, mode, rig.window(), size, &theme, vreq.as_ref()).and_then(|r| r.ok());
            let inset = panel.as_ref().map(|o| Inset {
                rect: o.panel.rect,
                rig: &o.panel.rig,
                layers: vec![Layer { geometry: &o.panel.geometry, fade: 1.0, origin: o.panel.origin, lift: 1.0 }],
            });
            let rgba = gpu.render_rgba_with_inset(
                size,
                &rig,
                &[Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }],
                theme.background,
                inset.as_ref(),
            );
            let path = out.with_file_name(format!("{}_{suffix}.png", out.file_stem().and_then(|s| s.to_str()).unwrap_or("ticker")));
            save_png(&path, size, &rgba)?;
            let vals: Vec<String> = doc.sliders.iter().map(|(k, v)| format!("{k}={}", v.value)).collect();
            eprintln!("wrote {} ({})", path.display(), vals.join(", "));
            Ok(())
        };
        shoot(&mut app, "before")?;
        for k in 0..=n {
            app.frame(k as f64 * 50.0);
        }
        shoot(&mut app, "after")?;
        return Ok(());
    }

    let mut gpu = Headless::new()?;
    eprintln!("adapter: {}", gpu.adapter_name);
    let aspect = size.0 as f64 / size.1 as f64;

    if let (Some(from), Some(to)) = (get("--from"), get("--to")) {
        let (from, to) = (parse_mode(&from), parse_mode(&to));
        let frames: usize = get("--frames").and_then(|s| s.parse().ok()).unwrap_or(6);
        let dir = PathBuf::from(get("--out-dir").unwrap_or_else(|| "frames".into()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut rig = Rig::new(window, from);
        rig.set_aspect(aspect);
        let ga = geometry(&doc, from, &rig, size, &theme);
        let gb = geometry(&doc, to, &rig, size, &theme);
        for g in [&ga, &gb] {
            for (id, msg) in &g.diagnostics {
                eprintln!("diagnostic [{id}]: {msg}");
            }
        }
        rig.set_mode(to, 0.0);
        for i in 0..frames {
            let t = i as f64 / (frames - 1).max(1) as f64;
            rig.update(t * rig.duration_ms);
            let p = rig.progress() as f32;
            let (fa, fb) = crossfade(p);
            let o = rig.render_origin();
            let layers = [
                Layer { geometry: &ga, fade: fa, origin: o, lift: layer_lift(from, rig.lift()) },
                Layer { geometry: &gb, fade: fb, origin: o, lift: layer_lift(to, rig.lift()) },
            ];
            let rgba = gpu.render_rgba(size, &rig, &layers, theme.background);
            let path = dir.join(format!("frame_{i:02}.png"));
            save_png(&path, size, &rgba)?;
            eprintln!("{}  progress={p:.2}", path.display());
        }
        return Ok(());
    }

    let mode = parse_mode(&get("--mode").unwrap_or_else(|| "2d".into()));
    let out = PathBuf::from(get("--out").unwrap_or_else(|| "out.png".into()));
    let mut rig = Rig::new(window, mode);
    rig.set_aspect(aspect);
    let t0 = std::time::Instant::now();
    let g = geometry(&doc, mode, &rig, size, &theme);
    eprintln!(
        "scene: {} segments, {} vertices, {} labels in {:?}",
        g.segments.len(),
        g.vertices.len(),
        g.labels.len(),
        t0.elapsed()
    );
    for (id, msg) in &g.diagnostics {
        eprintln!("diagnostic [{id}]: {msg}");
    }
    let panel = match build_slice_panel_view(&doc, mode, rig.window(), size, &theme, vreq.as_ref()) {
        Some(Ok(o)) => {
            eprintln!(
                "slice: dim {}, {} cut curve(s), {} marked point(s), inset {:?}, view {:?} (follow {})",
                o.rs.dim, o.panel.curves, o.panel.points, o.panel.rect, o.panel.view, o.panel.follow
            );
            Some(o)
        }
        Some(Err(e)) => return Err(format!("slice: {e}")),
        None => None,
    };
    let inset = panel.as_ref().map(|o| Inset {
        rect: o.panel.rect,
        rig: &o.panel.rig,
        layers: vec![Layer { geometry: &o.panel.geometry, fade: 1.0, origin: o.panel.origin, lift: 1.0 }],
    });
    let rgba = gpu.render_rgba_with_inset(
        size,
        &rig,
        &[Layer { geometry: &g, fade: 1.0, origin: rig.render_origin(), lift: 1.0 }],
        theme.background,
        inset.as_ref(),
    );
    save_png(&out, size, &rgba)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
