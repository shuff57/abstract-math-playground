//! Shared geometry-level harness for the conic and solid audits (no GPU needed).
#![allow(dead_code)]

use math_core::analyze::analyze;
use math_core::doc::{Doc, Item, ItemKind, SliderCfg};
use math_core::parse::parse;
use math_core::view::{Mode, Window3};
use math_playground_lib::geometry::{FieldKind, SceneGeometry, Theme};
use math_playground_lib::scene::build_scene;
use std::collections::{BTreeSet, HashMap};

pub const VIEWPORT: (u32, u32) = (800, 600);

pub fn kind_name(src: &str) -> String {
    match parse(src) {
        Err(e) => format!("PARSE-ERR({e})"),
        Ok(e) => {
            let s = format!("{:?}", analyze(&e, &BTreeSet::new()).kind);
            s.split(|c| c == ' ' || c == '{' || c == '(').next().unwrap_or("?").to_string()
        }
    }
}

pub fn make_doc(lines: &[&str], sliders: &[(&str, f64)]) -> Doc {
    let mut d = Doc::new_default();
    d.view.grid = false;
    d.view.axes = false;
    for (i, l) in lines.iter().enumerate() {
        d.items.push(Item::new(&format!("i{i}"), ItemKind::Equation, l));
    }
    for (n, v) in sliders {
        d.sliders.insert(n.to_string(), SliderCfg { min: -100.0, max: 100.0, step: None, value: *v });
    }
    d
}

pub fn build(doc: &Doc, mode: Mode, win: Window3) -> (SceneGeometry, [f64; 3]) {
    let c = win.centre();
    let g = build_scene(doc, mode, win, c, VIEWPORT, &Theme::light());
    (g, c)
}

/// Spatial hash for nearest-point queries.
pub struct Grid2 {
    cell: f64,
    m: HashMap<(i64, i64), Vec<(f64, f64)>>,
}
impl Grid2 {
    pub fn new(pts: &[(f64, f64)], cell: f64) -> Self {
        let mut m: HashMap<(i64, i64), Vec<(f64, f64)>> = HashMap::new();
        for p in pts {
            m.entry(((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64)).or_default().push(*p);
        }
        Grid2 { cell, m }
    }
    /// True when some stored point is within `r` (r <= cell).
    pub fn near(&self, p: (f64, f64), r: f64) -> bool {
        let (cx, cy) = ((p.0 / self.cell).floor() as i64, (p.1 / self.cell).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(v) = self.m.get(&(cx + dx, cy + dy)) {
                    if v.iter().any(|q| (q.0 - p.0).hypot(q.1 - p.1) <= r) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

pub struct Grid3 {
    cell: f64,
    m: HashMap<(i64, i64, i64), Vec<[f64; 3]>>,
}
impl Grid3 {
    pub fn new(pts: &[[f64; 3]], cell: f64) -> Self {
        let mut m: HashMap<(i64, i64, i64), Vec<[f64; 3]>> = HashMap::new();
        for p in pts {
            m.entry(Self::key(p, cell)).or_default().push(*p);
        }
        Grid3 { cell, m }
    }
    fn key(p: &[f64; 3], cell: f64) -> (i64, i64, i64) {
        ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)
    }
    pub fn near(&self, p: [f64; 3], r: f64) -> bool {
        let k = Self::key(&p, self.cell);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.m.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        if v.iter().any(|q| ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2)).sqrt() <= r) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

pub fn param<F: Fn(f64) -> (f64, f64)>(f: F, t0: f64, t1: f64, n: usize) -> Vec<(f64, f64)> {
    (0..=n)
        .map(|i| f(t0 + (t1 - t0) * i as f64 / n as f64))
        .filter(|p| p.0.is_finite() && p.1.is_finite())
        .collect()
}

pub fn field_desc(g: &SceneGeometry) -> String {
    g.fields
        .iter()
        .map(|f| match f.kind {
            FieldKind::Fill { greater } => format!("fill(f{}0)", if greater { ">" } else { "<" }),
            FieldKind::Hue => "hue".into(),
            FieldKind::Domain => "domain".into(),
        })
        .collect::<Vec<_>>()
        .join(",")
}
