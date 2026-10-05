//! Versioned document model, validation and share-link encoding.
//!
//! Everything here treats input as hostile (share links come from strangers):
//! size limits are enforced before parsing/decoding, decompression has a hard
//! output cap, and all numbers and strings are validated after deserialising.
//!
//! JSON nesting depth is bounded by serde_json's built-in recursion limit
//! (128 levels), so deeply nested input yields an error rather than a stack
//! overflow. Unknown JSON fields are ignored (forward compatibility); unknown
//! enum variants are errors.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

/// Current schema version.
pub const CURRENT_VERSION: u32 = 1;
/// Maximum accepted JSON size in bytes (also the decompression output cap).
pub const MAX_JSON_BYTES: usize = 100_000;
/// Maximum accepted share-hash length in chars, checked before decoding.
pub const MAX_HASH_CHARS: usize = 200_000;
/// Maximum number of items.
pub const MAX_ITEMS: usize = 500;
/// Maximum number of sliders.
pub const MAX_SLIDERS: usize = 200;
/// Maximum latex length in chars.
pub const MAX_LATEX_CHARS: usize = 2000;
/// Maximum id length in chars.
pub const MAX_ID_CHARS: usize = 64;

const HASH_PREFIX: &str = "v1.";
const MAX_REPORTED_ERRORS: usize = 10;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[serde(rename = "1d")]
    D1,
    #[serde(rename = "2d")]
    D2,
    #[serde(rename = "3d")]
    D3,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AngleMode {
    Rad,
    Deg,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewState {
    pub mode: Mode,
    pub window: WindowBox,
    pub angle: AngleMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Minor/major grid lines (additive field; saved only when false).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub grid: bool,
    /// Axis lines and their tick numbers (additive field; saved only when false).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub axes: bool,
    /// Tick numbers along the axes (additive field; saved only when false).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub axis_numbers: bool,
    /// Minor grid lines between the major ones (additive field; saved only when false).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub minor_grid: bool,
    /// Arrowheads at the ends of the 2D axes (additive field; saved only when true).
    #[serde(default, skip_serializing_if = "is_false")]
    pub arrows: bool,
    /// Names written at the ends of the x and y axes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_label: Option<String>,
    /// Fixed spacing of the major grid lines and tick numbers (`None`: automatic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_step: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_step: Option<f64>,
    /// Pan and zoom are ignored while set (additive field; saved only when true).
    #[serde(default, skip_serializing_if = "is_false")]
    pub lock: bool,
    /// Rectangular or polar 2D grid (additive field; saved only when polar).
    #[serde(default, skip_serializing_if = "GridKind::is_rect")]
    pub grid_kind: GridKind,
    /// Linear or logarithmic 2D axes (additive fields; saved only when logarithmic). A
    /// logarithmic axis needs a window whose minimum on that axis is greater than 0.
    #[serde(default, skip_serializing_if = "AxisScale::is_linear")]
    pub x_scale: AxisScale,
    #[serde(default, skip_serializing_if = "AxisScale::is_linear")]
    pub y_scale: AxisScale,
}

/// The 2D grid: lines parallel to the axes, or circles and spokes around the origin.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GridKind {
    #[default]
    Rect,
    Polar,
}

impl GridKind {
    pub fn is_rect(&self) -> bool {
        *self == GridKind::Rect
    }
}

/// How a 2D axis maps numbers to positions.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AxisScale {
    #[default]
    Linear,
    Log,
}

impl AxisScale {
    pub fn is_linear(&self) -> bool {
        *self == AxisScale::Linear
    }
    pub fn is_log(&self) -> bool {
        *self == AxisScale::Log
    }
}

impl ViewState {
    /// Checks that every logarithmic axis has a window minimum greater than 0 (`w`, or the
    /// view's own window). The message names the axis.
    pub fn check_log_window(&self, w: Option<&WindowBox>) -> Result<(), String> {
        let w = w.unwrap_or(&self.window);
        for (a, n, sc) in [(0, "x", self.x_scale), (1, "y", self.y_scale)] {
            if sc.is_log() && !(w.min[a] > 0.0 && w.max[a] > w.min[a]) {
                return Err(format!(
                    "a logarithmic {n} axis needs {n} min > 0 (the window has {n} from {} to {})",
                    w.min[a], w.max[a]
                ));
            }
        }
        Ok(())
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ItemKind {
    Equation,
    Expression,
    Complex,
    Points,
    VectorField,
    Action,
    Slider,
    Slice,
    Folder,
    Note,
    /// Columns of cells; each column defines a list named by its header (see [`crate::table`]).
    Table,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PointStyle {
    Dot,
    Circle,
    Cross,
    Square,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemStyle {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_style: Option<LineStyle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_style: Option<PointStyle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Regression items: draw a tick from each data point to the fitted curve.
    #[serde(default, skip_serializing_if = "is_false")]
    pub residuals: bool,
    /// Diameter of a drawn point in pixels (same units as `line_width`), `1..=40`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_size: Option<f64>,
    /// Opacity of an inequality / region shading in `[0, 1]` (default: the built-in 0.22).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_opacity: Option<f64>,
    /// Draw a text label next to the item's points (the `label` text, else the coordinates).
    #[serde(default, skip_serializing_if = "is_false")]
    pub show_label: bool,
    /// Regression items: plot the residual of each data point against its x value (the
    /// regression panel's "plot" button).
    #[serde(default, skip_serializing_if = "is_false")]
    pub residual_plot: bool,
}

/// Valid range of [`ItemStyle::point_size`].
pub const POINT_SIZE_RANGE: std::ops::RangeInclusive<f64> = 1.0..=40.0;

impl ItemStyle {
    fn is_empty(&self) -> bool {
        *self == ItemStyle::default()
    }

    /// Checks the numeric fields: `opacity` and `fillOpacity` in `[0, 1]`, `lineWidth` in
    /// `(0, 100]`, `pointSize` in `[1, 40]`, all finite. Returns every problem, `; `-joined.
    pub fn validate(&self) -> Result<(), String> {
        let mut e = Vec::new();
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if self.opacity.is_some_and(|o| !unit(o)) {
            e.push("opacity must be in [0,1]".to_string());
        }
        if let Some(w) = self.line_width {
            if !(w.is_finite() && w > 0.0 && w <= 100.0) {
                e.push("lineWidth must be in (0,100]".to_string());
            }
        }
        if self
            .point_size
            .is_some_and(|s| !(s.is_finite() && POINT_SIZE_RANGE.contains(&s)))
        {
            e.push("pointSize must be in [1,40]".to_string());
        }
        if self.fill_opacity.is_some_and(|o| !unit(o)) {
            e.push("fillOpacity must be in [0,1]".to_string());
        }
        if e.is_empty() {
            Ok(())
        } else {
            Err(e.join("; "))
        }
    }
}

impl WindowBox {
    /// Every axis finite with `min < max` and a sane span (as [`Doc::validate`] requires).
    pub fn validate(&self) -> Result<(), String> {
        let mut e = Vec::new();
        for a in 0..3 {
            let (lo, hi) = (self.min[a], self.max[a]);
            if !lo.is_finite() || !hi.is_finite() {
                e.push(format!("window axis {a}: non-finite bound"));
                continue;
            }
            let span = hi - lo;
            if lo >= hi || span <= 1e-300 || span >= 1e300 {
                e.push(format!("window axis {a}: need min < max with sane span"));
            }
        }
        if e.is_empty() {
            Ok(())
        } else {
            Err(e.join("; "))
        }
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Item {
    pub id: String,
    pub kind: ItemKind,
    pub latex: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub hidden: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "ItemStyle::is_empty")]
    pub style: ItemStyle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folder: Option<String>,
    /// Data for `ItemKind::Table` items (additive field: absent in older v1 documents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<crate::table::Table>,
}

impl Item {
    /// Minimal item with defaults for everything optional.
    pub fn new(id: &str, kind: ItemKind, latex: &str) -> Item {
        Item {
            id: id.to_string(),
            kind,
            latex: latex.to_string(),
            hidden: false,
            color: None,
            style: ItemStyle::default(),
            folder: None,
            table: None,
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SliderCfg {
    pub min: f64,
    pub max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    pub value: f64,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Doc {
    pub v: u32,
    pub view: ViewState,
    pub items: Vec<Item>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sliders: BTreeMap<String, SliderCfg>,
    /// Ticker configuration (additive field: absent in older v1 documents). Whether it is
    /// running is runtime state and is never saved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ticker: Option<TickerCfg>,
    /// Slice (additive field: absent in older v1 documents), see [`crate::slice`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<crate::slice::SliceCfg>,
}

/// Default ticker interval in milliseconds.
pub const TICKER_DEFAULT_RATE_MS: f64 = 50.0;
/// Default (and lowest sensible) ticker min step in milliseconds.
pub const TICKER_MIN_STEP_MS: f64 = 10.0;

fn default_rate_ms() -> f64 {
    TICKER_DEFAULT_RATE_MS
}
fn default_min_step_ms() -> f64 {
    TICKER_MIN_STEP_MS
}
fn default_true() -> bool {
    true
}
fn is_true(b: &bool) -> bool {
    *b
}
fn is_default_min_step(v: &f64) -> bool {
    *v == TICKER_MIN_STEP_MS
}

/// Which action item the ticker fires and how often.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerCfg {
    /// Id of the action item to fire, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Interval between steps in ms.
    #[serde(default = "default_rate_ms")]
    pub rate_ms: f64,
    /// Floor for the effective interval (`max(rate_ms, min_step_ms)`).
    #[serde(default = "default_min_step_ms", skip_serializing_if = "is_default_min_step")]
    pub min_step_ms: f64,
    /// Pause the ticker when the action errors (e.g. evaluates to NaN).
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub pause_on_error: bool,
}

impl Default for TickerCfg {
    fn default() -> Self {
        TickerCfg { action: None, rate_ms: TICKER_DEFAULT_RATE_MS, min_step_ms: TICKER_MIN_STEP_MS, pause_on_error: true }
    }
}

impl TickerCfg {
    /// Effective interval between steps: at least `min_step_ms` (itself at least 1 ms).
    pub fn interval_ms(&self) -> f64 {
        self.rate_ms.max(self.min_step_ms.max(1.0))
    }
}

/// Errors from parsing, validating, or decoding documents.
#[derive(Debug, Clone, PartialEq)]
pub enum DocError {
    /// Input (JSON, hash, or decompressed data) exceeds a size limit.
    TooLarge,
    /// Hash prefix or base64 is malformed.
    BadEncoding(String),
    /// Deflate stream is corrupt (or inflates past the limit).
    Decompress(String),
    /// JSON syntax or schema error (including unknown enum variants).
    Json(String),
    /// Document was written by a newer version.
    TooNew { found: u32 },
    /// Document version is missing or unusable.
    BadVersion(String),
    /// Semantic validation failure.
    Invalid(String),
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DocError::TooLarge => write!(f, "input too large"),
            DocError::BadEncoding(m) => write!(f, "bad encoding: {m}"),
            DocError::Decompress(m) => write!(f, "decompression failed: {m}"),
            DocError::Json(m) => write!(f, "invalid JSON: {m}"),
            DocError::TooNew { found } => write!(
                f,
                "document version {found} is newer than supported version {CURRENT_VERSION}"
            ),
            DocError::BadVersion(m) => write!(f, "bad document version: {m}"),
            DocError::Invalid(m) => write!(f, "invalid document: {m}"),
        }
    }
}

impl std::error::Error for DocError {}

// ---------------------------------------------------------------- migration

#[allow(unused_mut, unreachable_code, clippy::never_loop, unused_assignments, clippy::match_single_binding)]
/// Upgrade `value` in place, one version at a time, to [`CURRENT_VERSION`].
pub fn migrate(mut value: serde_json::Value) -> Result<serde_json::Value, DocError> {
    let found = value
        .get("v")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| DocError::BadVersion("missing or non-integer \"v\"".into()))?;
    if found == 0 {
        return Err(DocError::BadVersion("version 0 is not valid".into()));
    }
    if found > CURRENT_VERSION as u64 {
        return Err(DocError::TooNew {
            found: found.min(u32::MAX as u64) as u32,
        });
    }
    let mut ver = found as u32;
    while ver < CURRENT_VERSION {
        value = match ver {
            // Future steps plug in here, e.g.:
            // 1 => migrate_1_to_2(value)?,
            _ => return Err(DocError::BadVersion(format!("no migration from v{ver}"))),
        };
        ver += 1;
        value["v"] = serde_json::Value::from(ver);
    }
    Ok(value)
}

/// Documented no-op example of a migration step (v1 -> v2 would look like this:
/// reshape fields, then the loop in [`migrate`] bumps `"v"`). Not wired in
/// because no v2 exists yet.
#[allow(dead_code)]
fn migrate_1_to_2(value: serde_json::Value) -> Result<serde_json::Value, DocError> {
    Ok(value)
}

// ------------------------------------------------------------------- Doc API

pub(crate) fn is_hex_color(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 7 && b[0] == b'#' && b[1..].iter().all(|c| c.is_ascii_hexdigit())
}

fn is_slider_name(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 32
        && b[0].is_ascii_alphabetic()
        && b[1..].iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

struct Errs(Vec<String>);
impl Errs {
    fn push(&mut self, s: String) {
        if self.0.len() < MAX_REPORTED_ERRORS {
            self.0.push(s);
        }
    }
    fn finish(self) -> Result<(), DocError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(DocError::Invalid(self.0.join("; ")))
        }
    }
}

fn check_item_fields(it: &Item, ctx: &str, e: &mut Errs) {
    let idlen = it.id.chars().count();
    if idlen == 0 || idlen > MAX_ID_CHARS {
        e.push(format!("{ctx}: id must be 1..={MAX_ID_CHARS} chars"));
    }
    if it.latex.chars().count() > MAX_LATEX_CHARS {
        e.push(format!("{ctx}: latex longer than {MAX_LATEX_CHARS} chars"));
    }
    if let Some(t) = &it.table {
        if let Err(m) = t.validate() {
            e.push(format!("{ctx}: table: {m}"));
        }
    }
    if let Some(c) = &it.color {
        if !is_hex_color(c) {
            e.push(format!("{ctx}: color must match #rrggbb"));
        }
    }
    if let Err(m) = it.style.validate() {
        for part in m.split("; ") {
            e.push(format!("{ctx}: {part}"));
        }
    }
}

impl Doc {
    /// Moves every table's legacy table-wide style into its columns (see
    /// [`crate::table::Table::migrate_style`]); [`from_json`] does this on load.
    pub fn migrate_table_styles(&mut self) {
        for t in self.items.iter_mut().filter_map(|i| i.table.as_mut()) {
            t.migrate_style();
        }
    }
}

impl Doc {
    /// Empty 2D document with a [-10,10] window.
    pub fn new_default() -> Doc {
        Doc {
            v: CURRENT_VERSION,
            view: ViewState {
                mode: Mode::D2,
                window: WindowBox {
                    min: [-10.0; 3],
                    max: [10.0; 3],
                },
                angle: AngleMode::Rad,
                theme: None,
                grid: true,
                axes: true,
                axis_numbers: true,
                minor_grid: true,
                arrows: false,
                x_label: None,
                y_label: None,
                x_step: None,
                y_step: None,
                lock: false,
                grid_kind: GridKind::Rect,
                x_scale: AxisScale::Linear,
                y_scale: AxisScale::Linear,
            },
            items: Vec::new(),
            sliders: BTreeMap::new(),
            ticker: None,
            slice: None,
        }
    }

    /// Check all invariants; reports up to 10 problems with item index/id context.
    pub fn validate(&self) -> Result<(), DocError> {
        let mut e = Errs(Vec::new());
        if self.v != CURRENT_VERSION {
            e.push(format!("v is {} but expected {CURRENT_VERSION}", self.v));
        }
        if let Err(m) = self.view.window.validate() {
            for part in m.split("; ") {
                e.push(part.to_string());
            }
        }
        for (n, v) in [("x", self.view.x_step), ("y", self.view.y_step)] {
            if v.is_some_and(|v| !(v.is_finite() && v > 0.0)) {
                e.push(format!("view.{n}Step must be a positive number"));
            }
        }
        if let Err(m) = self.view.check_log_window(None) {
            e.push(format!("view: {m}"));
        }
        for (n, v) in [("x", &self.view.x_label), ("y", &self.view.y_label)] {
            if v.as_ref().is_some_and(|t| t.chars().count() > 64) {
                e.push(format!("view.{n}Label is longer than 64 characters"));
            }
        }
        if self.items.len() > MAX_ITEMS {
            e.push(format!("too many items ({} > {MAX_ITEMS})", self.items.len()));
            return e.finish(); // avoid O(n) work on absurd input
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut folders: HashSet<&str> = HashSet::new();
        for (i, it) in self.items.iter().enumerate() {
            let ctx = format!("item[{i}] id={:?}", truncate(&it.id));
            check_item_fields(it, &ctx, &mut e);
            if !seen.insert(it.id.as_str()) {
                e.push(format!("{ctx}: duplicate id"));
            }
            if it.kind == ItemKind::Folder {
                folders.insert(it.id.as_str());
            }
        }
        for (i, it) in self.items.iter().enumerate() {
            if let Some(f) = &it.folder {
                if !folders.contains(f.as_str()) || *f == it.id {
                    e.push(format!(
                        "item[{i}] id={:?}: folder {:?} is not an existing folder item",
                        truncate(&it.id),
                        truncate(f)
                    ));
                }
            }
        }
        if self.sliders.len() > MAX_SLIDERS {
            e.push(format!("too many sliders ({} > {MAX_SLIDERS})", self.sliders.len()));
            return e.finish();
        }
        if let Some(t) = &self.ticker {
            if !(t.rate_ms.is_finite() && t.rate_ms > 0.0 && t.rate_ms <= 1e9) {
                e.push("ticker: rateMs must be a positive finite number".into());
            }
            if !(t.min_step_ms.is_finite() && (0.0..=1e9).contains(&t.min_step_ms)) {
                e.push("ticker: minStepMs must be a finite number >= 0".into());
            }
            if t.action.as_ref().is_some_and(|a| a.chars().count() > MAX_ID_CHARS) {
                e.push("ticker: action id too long".into());
            }
        }
        if let Some(sl) = &self.slice {
            if let Err(m) = sl.validate() {
                e.push(format!("slice: {m}"));
            }
        }
        for (name, s) in &self.sliders {
            let ctx = format!("slider {:?}", truncate(name));
            if !is_slider_name(name) {
                e.push(format!("{ctx}: bad name"));
            }
            if !(s.min.is_finite() && s.max.is_finite() && s.value.is_finite()) {
                e.push(format!("{ctx}: non-finite number"));
            } else if s.min > s.max {
                e.push(format!("{ctx}: min > max"));
            }
            if let Some(st) = s.step {
                if !st.is_finite() {
                    e.push(format!("{ctx}: non-finite step"));
                }
            }
        }
        e.finish()
    }

    fn check_new_item(&self, item: &Item) -> Result<(), DocError> {
        let mut e = Errs(Vec::new());
        if self.items.len() >= MAX_ITEMS {
            e.push(format!("too many items (max {MAX_ITEMS})"));
        }
        let ctx = format!("item id={:?}", truncate(&item.id));
        check_item_fields(item, &ctx, &mut e);
        if self.items.iter().any(|i| i.id == item.id) {
            e.push(format!("{ctx}: duplicate id"));
        }
        if let Some(f) = &item.folder {
            let ok = *f != item.id
                && self
                    .items
                    .iter()
                    .any(|i| i.id == *f && i.kind == ItemKind::Folder);
            if !ok {
                e.push(format!("{ctx}: folder {:?} is not an existing folder", truncate(f)));
            }
        }
        e.finish()
    }

    /// Append an item, keeping invariants (unique id, limits, folder exists).
    pub fn add_item(&mut self, item: Item) -> Result<(), DocError> {
        self.check_new_item(&item)?;
        self.items.push(item);
        Ok(())
    }

    /// Remove and return an item. Removing a folder un-files its children.
    pub fn remove_item(&mut self, id: &str) -> Result<Item, DocError> {
        let pos = self
            .items
            .iter()
            .position(|i| i.id == id)
            .ok_or_else(|| DocError::Invalid(format!("no item with id {:?}", truncate(id))))?;
        let removed = self.items.remove(pos);
        if removed.kind == ItemKind::Folder {
            for i in &mut self.items {
                if i.folder.as_deref() == Some(id) {
                    i.folder = None;
                }
            }
        }
        Ok(removed)
    }

    /// Move an item to `to_index` (index in the resulting list; must be < len).
    pub fn move_item(&mut self, id: &str, to_index: usize) -> Result<(), DocError> {
        if to_index >= self.items.len() {
            return Err(DocError::Invalid(format!("index {to_index} out of range")));
        }
        let pos = self
            .items
            .iter()
            .position(|i| i.id == id)
            .ok_or_else(|| DocError::Invalid(format!("no item with id {:?}", truncate(id))))?;
        let it = self.items.remove(pos);
        self.items.insert(to_index, it);
        Ok(())
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(40).collect()
}

/// Serialise to compact JSON.
pub fn to_json(doc: &Doc) -> String {
    serde_json::to_string(doc).expect("Doc serialisation cannot fail")
}

/// Parse, migrate, deserialise and validate a document.
pub fn from_json(s: &str) -> Result<Doc, DocError> {
    if s.len() > MAX_JSON_BYTES {
        return Err(DocError::TooLarge);
    }
    let value: serde_json::Value =
        serde_json::from_str(s).map_err(|e| DocError::Json(e.to_string()))?;
    let value = migrate(value)?;
    let mut doc: Doc = serde_json::from_value(value).map_err(|e| DocError::Json(e.to_string()))?;
    doc.migrate_table_styles();
    doc.validate()?;
    Ok(doc)
}

/// Share-link payload: `"v1." + base64url(no pad)(deflate(json))`.
pub fn encode_hash(doc: &Doc) -> String {
    let json = to_json(doc);
    let z = miniz_oxide::deflate::compress_to_vec(json.as_bytes(), 9);
    format!("{HASH_PREFIX}{}", URL_SAFE_NO_PAD.encode(z))
}

/// Decode a share-link payload (optional leading `#`). Length-checked before
/// decoding; inflation is hard-capped at [`MAX_JSON_BYTES`].
pub fn decode_hash(s: &str) -> Result<Doc, DocError> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() > MAX_HASH_CHARS {
        return Err(DocError::TooLarge);
    }
    let body = s
        .strip_prefix(HASH_PREFIX)
        .ok_or_else(|| DocError::BadEncoding("missing \"v1.\" prefix".into()))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(body)
        .map_err(|e| DocError::BadEncoding(e.to_string()))?;
    let json = miniz_oxide::inflate::decompress_to_vec_with_limit(&bytes, MAX_JSON_BYTES)
        .map_err(|e| match e.status {
            miniz_oxide::inflate::TINFLStatus::HasMoreOutput => DocError::TooLarge,
            st => DocError::Decompress(format!("{st:?}")),
        })?;
    let json = String::from_utf8(json).map_err(|_| DocError::Json("not valid UTF-8".into()))?;
    from_json(&json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rich_doc() -> Doc {
        let mut d = Doc::new_default();
        d.view.theme = Some("dark".into());
        d.view.angle = AngleMode::Deg;
        d.view.mode = Mode::D3;
        let kinds = [
            ItemKind::Folder,
            ItemKind::Equation,
            ItemKind::Expression,
            ItemKind::Complex,
            ItemKind::Points,
            ItemKind::VectorField,
            ItemKind::Action,
            ItemKind::Slider,
            ItemKind::Slice,
            ItemKind::Note,
        ];
        for (i, k) in kinds.iter().enumerate() {
            let mut it = Item::new(&format!("i{i}"), *k, &format!("x^{i}+\\sin(a)"));
            if i > 0 {
                it.folder = Some("i0".into());
            }
            if i % 2 == 0 {
                it.color = Some("#a1B2c3".into());
                it.hidden = true;
                it.style = ItemStyle {
                    line_width: Some(2.5),
                    line_style: Some(LineStyle::Dashed),
                    point_style: Some(PointStyle::Cross),
                    opacity: Some(0.5),
                    label: Some("lbl".into()),
                    residuals: i % 4 == 0,
                    point_size: Some(12.0),
                    fill_opacity: Some(0.4),
                    show_label: true,
                    residual_plot: i % 4 == 0,
                };
            }
            d.add_item(it).unwrap();
        }
        d.sliders.insert(
            "a".into(),
            SliderCfg { min: 0.0, max: 5.0, step: Some(0.1), value: 1.0 },
        );
        d
    }

    const GOLDEN: &str = r##"{"v":1,"view":{"mode":"2d","window":{"min":[-10.0,-10.0,-10.0],"max":[10.0,10.0,10.0]},"angle":"rad"},"items":[{"id":"a","kind":"vectorField","latex":"y=x^2","color":"#ff0000","style":{"lineWidth":2.0,"lineStyle":"dotted"}}],"sliders":{"k":{"min":0.0,"max":1.0,"step":0.5,"value":0.25}}}"##;

    #[test]
    fn ticker_config_roundtrips_and_v1_still_decodes() {
        // Older v1 documents have no ticker.
        let d = from_json(GOLDEN).unwrap();
        assert!(d.ticker.is_none());
        let mut d = rich_doc();
        d.ticker = Some(TickerCfg { action: Some("i6".into()), rate_ms: 125.0, min_step_ms: 20.0, pause_on_error: false });
        let j = to_json(&d);
        assert!(j.contains("\"ticker\"") && j.contains("rateMs"));
        assert_eq!(from_json(&j).unwrap(), d);
        assert_eq!(decode_hash(&encode_hash(&d)).unwrap(), d);
        // Defaults are omitted and restored.
        let mut d2 = rich_doc();
        d2.ticker = Some(TickerCfg::default());
        let j2 = to_json(&d2);
        assert!(!j2.contains("minStepMs") && !j2.contains("pauseOnError"));
        assert_eq!(from_json(&j2).unwrap().ticker, Some(TickerCfg::default()));
        // Bad values are rejected.
        let bad = j.replace("125.0", "-1.0");
        assert!(matches!(from_json(&bad), Err(DocError::Invalid(_))));
    }

    #[test]
    fn grid_kind_and_axis_scales_round_trip_and_default_quietly() {
        // Defaults are not written, so older documents stay byte-identical.
        let j = to_json(&Doc::new_default());
        assert!(!j.contains("gridKind") && !j.contains("xScale") && !j.contains("yScale"));
        let mut d = rich_doc();
        d.view.grid_kind = GridKind::Polar;
        d.view.y_scale = AxisScale::Log;
        d.view.window.min[1] = 0.01;
        d.view.window.max[1] = 1000.0;
        let j = to_json(&d);
        assert!(j.contains(r#""gridKind":"polar""#) && j.contains(r#""yScale":"log""#), "{j}");
        assert!(!j.contains("xScale"));
        assert_eq!(from_json(&j).unwrap(), d);
        assert_eq!(decode_hash(&encode_hash(&d)).unwrap(), d);
        // A logarithmic axis whose window reaches 0 or below is rejected.
        let mut bad = d.clone();
        bad.view.window.min[1] = 0.0;
        let err = from_json(&to_json(&bad)).unwrap_err().to_string();
        assert!(err.contains("logarithmic y axis"), "{err}");
        assert!(from_json(&j.replace(r#""yScale":"log""#, r#""yScale":"ln""#)).is_err());
    }

    #[test]
    fn golden_fixture() {
        let d = from_json(GOLDEN).unwrap();
        assert_eq!(to_json(&d), GOLDEN);
        assert_eq!(d.items[0].kind, ItemKind::VectorField);
    }

    #[test]
    fn roundtrip_json_and_hash() {
        let d = rich_doc();
        assert_eq!(from_json(&to_json(&d)).unwrap(), d);
        let h = encode_hash(&d);
        assert!(h.starts_with("v1."));
        assert_eq!(decode_hash(&h).unwrap(), d);
        assert_eq!(decode_hash(&format!("#{h}")).unwrap(), d);
    }

    #[test]
    fn hash_is_short() {
        let mut d = Doc::new_default();
        for i in 0..10 {
            d.add_item(Item::new(&format!("e{i}"), ItemKind::Equation, "y=\\sin(x)+x^2")).unwrap();
        }
        let h = encode_hash(&d);
        assert!(h.len() < 2048, "len {}", h.len());
    }

    #[test]
    fn unknown_fields_ignored() {
        let j = r##"{"v":1,"future":{"a":1},"view":{"mode":"2d","extra":1,"window":{"min":[0,0,0],"max":[1,1,1]},"angle":"deg"},"items":[{"id":"a","kind":"note","latex":"hi","zzz":[1,2]}]}"##;
        let d = from_json(j).unwrap();
        assert_eq!(d.items.len(), 1);
        assert_eq!(d.view.angle, AngleMode::Deg);
    }

    #[test]
    fn unknown_variant_errors() {
        let j = GOLDEN.replace("vectorField", "hologram");
        match from_json(&j) {
            Err(DocError::Json(m)) => assert!(m.contains("hologram"), "{m}"),
            o => panic!("{o:?}"),
        }
        assert!(matches!(from_json(&GOLDEN.replace("\"2d\"", "\"4d\"")), Err(DocError::Json(_))));
    }

    #[test]
    fn versions() {
        let j = GOLDEN.replacen("\"v\":1", "\"v\":2", 1);
        assert_eq!(from_json(&j), Err(DocError::TooNew { found: 2 }));
        assert!(matches!(from_json(&GOLDEN.replacen("\"v\":1", "\"v\":0", 1)), Err(DocError::BadVersion(_))));
        assert!(matches!(from_json(&GOLDEN.replacen("\"v\":1,", "", 1)), Err(DocError::BadVersion(_))));
        assert!(matches!(from_json(&GOLDEN.replacen("\"v\":1", "\"v\":-1", 1)), Err(DocError::BadVersion(_))));
        let huge = GOLDEN.replacen("\"v\":1", "\"v\":99999999999", 1);
        assert!(matches!(from_json(&huge), Err(DocError::TooNew { .. })));
    }

    #[test]
    fn zip_bomb() {
        let zeros = vec![0u8; 50_000_000];
        let z = miniz_oxide::deflate::compress_to_vec(&zeros, 9);
        drop(zeros);
        let h = format!("v1.{}", URL_SAFE_NO_PAD.encode(z));
        assert!(h.len() < MAX_HASH_CHARS);
        let t = std::time::Instant::now();
        assert_eq!(decode_hash(&h), Err(DocError::TooLarge));
        assert!(t.elapsed().as_secs() < 2);
    }

    #[test]
    fn oversize_and_garbage_hash() {
        let big = format!("v1.{}", "A".repeat(MAX_HASH_CHARS));
        assert_eq!(decode_hash(&big), Err(DocError::TooLarge));
        assert!(matches!(decode_hash("v2.AAAA"), Err(DocError::BadEncoding(_))));
        assert!(matches!(decode_hash("v1.!!!***"), Err(DocError::BadEncoding(_))));
        assert!(decode_hash("v1.").is_err());
        assert!(decode_hash("").is_err());
        assert!(decode_hash("v1.AAAAAAAA").is_err());
        let h = encode_hash(&rich_doc());
        for cut in [5, h.len() / 2, h.len() - 3] {
            assert!(decode_hash(&h[..cut]).is_err());
        }
        // Valid base64 of non-deflate bytes, and deflate of non-JSON.
        let z = miniz_oxide::deflate::compress_to_vec(b"not json", 6);
        assert!(matches!(decode_hash(&format!("v1.{}", URL_SAFE_NO_PAD.encode(z))), Err(DocError::Json(_))));
        let z = miniz_oxide::deflate::compress_to_vec(&[0xff, 0xfe, 0xfd], 6);
        assert!(decode_hash(&format!("v1.{}", URL_SAFE_NO_PAD.encode(z))).is_err());
    }

    #[test]
    fn deep_nesting() {
        let n = 10_000;
        let j = format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(matches!(from_json(&j), Err(DocError::Json(_))));
        let j = format!("{{\"v\":1,\"x\":{}{}}}", "[".repeat(n), "]".repeat(n));
        assert!(matches!(from_json(&j), Err(DocError::Json(_))));
    }

    #[test]
    fn many_items() {
        let mut d = Doc::new_default();
        for i in 0..100_000 {
            d.items.push(Item::new(&format!("i{i}"), ItemKind::Note, ""));
        }
        assert!(matches!(d.validate(), Err(DocError::Invalid(m)) if m.contains("too many items")));
        // As JSON it is rejected by size before parsing.
        assert_eq!(from_json(&to_json(&d)), Err(DocError::TooLarge));
        // 501 items fits in size limit but fails validation.
        d.items.truncate(501);
        assert!(from_json(&to_json(&d)).is_err());
        d.items.truncate(500);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn non_finite_numbers() {
        assert!(from_json(&GOLDEN.replace("0.25", "1e999")).is_err());
        assert!(from_json(&GOLDEN.replace("0.25", "NaN")).is_err());
        assert!(from_json(&GOLDEN.replace("0.25", "Infinity")).is_err());
        assert!(from_json(&GOLDEN.replace("0.25", "null")).is_err());
        let mut d = rich_doc();
        d.view.window.max[0] = f64::INFINITY;
        assert!(d.validate().is_err());
        let mut d = rich_doc();
        d.sliders.get_mut("a").unwrap().value = f64::NAN;
        assert!(d.validate().is_err());
        let mut d = rich_doc();
        d.items[1].style.opacity = Some(f64::NAN);
        assert!(d.validate().is_err());
    }

    #[test]
    fn window_checks() {
        let mut d = Doc::new_default();
        d.view.window.min[1] = 10.0;
        assert!(d.validate().is_err());
        let mut d = Doc::new_default();
        d.view.window.min[2] = 1.0;
        d.view.window.max[2] = 1.0 + 1e-310;
        assert!(d.validate().is_err());
        let mut d = Doc::new_default();
        d.view.window = WindowBox { min: [-1e308; 3], max: [1e308; 3] };
        assert!(d.validate().is_err());
    }

    #[test]
    fn duplicate_ids_and_fields() {
        let mut d = Doc::new_default();
        d.add_item(Item::new("a", ItemKind::Note, "")).unwrap();
        assert!(d.add_item(Item::new("a", ItemKind::Note, "")).is_err());
        d.items.push(Item::new("a", ItemKind::Note, ""));
        match d.validate() {
            Err(DocError::Invalid(m)) => assert!(m.contains("item[1]") && m.contains("duplicate")),
            o => panic!("{o:?}"),
        }
        let mut d = Doc::new_default();
        d.items.push(Item::new("", ItemKind::Note, ""));
        assert!(d.validate().is_err());
        let mut d = Doc::new_default();
        d.items.push(Item::new(&"x".repeat(65), ItemKind::Note, ""));
        assert!(d.validate().is_err());
    }

    #[test]
    fn proto_keys_harmless() {
        let j = r##"{"__proto__":{"polluted":1},"constructor":{"x":1},"v":1,"view":{"__proto__":1,"mode":"1d","window":{"min":[0,0,0],"max":[1,1,1]},"angle":"rad"},"items":[{"id":"constructor","kind":"note","latex":"","__proto__":{"a":1}}],"sliders":{"__proto__":{"min":0,"max":1,"value":0}}}"##;
        // Slider name "__proto__" is invalid (must start with a letter).
        assert!(from_json(j).is_err());
        let j2 = j.replace(",\"sliders\":{\"__proto__\":{\"min\":0,\"max\":1,\"value\":0}}", "");
        let d = from_json(&j2).unwrap();
        assert_eq!(d.items[0].id, "constructor");
    }

    #[test]
    fn huge_latex_and_colors() {
        let mut d = Doc::new_default();
        d.items.push(Item::new("a", ItemKind::Note, &"x".repeat(2001)));
        assert!(d.validate().is_err());
        let j = to_json(&d).replace(&"x".repeat(2001), &"x".repeat(MAX_JSON_BYTES));
        assert_eq!(from_json(&j), Err(DocError::TooLarge));
        d.items[0].latex = "x".repeat(2000);
        assert!(d.validate().is_ok());
        for bad in ["red", "#12345", "#1234567", "#gggggg", "123456", "#12345\n", ""] {
            let mut d = Doc::new_default();
            let mut it = Item::new("a", ItemKind::Note, "");
            it.color = Some(bad.into());
            d.items.push(it);
            assert!(d.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn range_checks() {
        for (w, o) in [(Some(0.0), None), (Some(101.0), None), (None, Some(1.5)), (None, Some(-0.1))] {
            let mut d = Doc::new_default();
            let mut it = Item::new("a", ItemKind::Equation, "");
            it.style.line_width = w;
            it.style.opacity = o;
            d.items.push(it);
            assert!(d.validate().is_err());
        }
        let mut d = rich_doc();
        d.sliders.insert("1bad".into(), SliderCfg { min: 0.0, max: 1.0, step: None, value: 0.0 });
        assert!(d.validate().is_err());
        let mut d = rich_doc();
        d.sliders.insert("a".into(), SliderCfg { min: 2.0, max: 1.0, step: None, value: 0.0 });
        assert!(d.validate().is_err());
        let mut d = rich_doc();
        for i in 0..201 {
            d.sliders.insert(format!("s{i}"), SliderCfg { min: 0.0, max: 1.0, step: None, value: 0.0 });
        }
        assert!(d.validate().is_err());
    }

    #[test]
    fn folder_refs() {
        let mut d = Doc::new_default();
        d.add_item(Item::new("n", ItemKind::Note, "")).unwrap();
        let mut it = Item::new("a", ItemKind::Note, "");
        it.folder = Some("missing".into());
        assert!(d.add_item(it.clone()).is_err());
        it.folder = Some("n".into()); // exists but not a folder
        assert!(d.add_item(it.clone()).is_err());
        d.items.push(it);
        assert!(d.validate().is_err());
    }

    #[test]
    fn add_remove_move() {
        let mut d = Doc::new_default();
        d.add_item(Item::new("f", ItemKind::Folder, "")).unwrap();
        for id in ["a", "b", "c"] {
            let mut it = Item::new(id, ItemKind::Note, "");
            it.folder = Some("f".into());
            d.add_item(it).unwrap();
        }
        d.move_item("c", 0).unwrap();
        let ids: Vec<_> = d.items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["c", "f", "a", "b"]);
        assert!(d.move_item("c", 4).is_err());
        assert!(d.move_item("zz", 0).is_err());
        d.remove_item("f").unwrap();
        assert!(d.items.iter().all(|i| i.folder.is_none()));
        assert!(d.validate().is_ok());
        assert!(d.remove_item("f").is_err());
        d.remove_item("a").unwrap();
        assert_eq!(d.items.len(), 2);
        // Capacity limit.
        let mut d = Doc::new_default();
        for i in 0..MAX_ITEMS {
            d.add_item(Item::new(&format!("i{i}"), ItemKind::Note, "")).unwrap();
        }
        assert!(d.add_item(Item::new("extra", ItemKind::Note, "")).is_err());
    }
    #[test]
    fn table_items_round_trip_and_old_docs_still_load() {
        let mut d = Doc::new_default();
        let mut it = Item::new("t1", ItemKind::Table, "");
        let mut t = crate::table::Table::new(&[], 2);
        t.set_cell(0, 0, "1").unwrap();
        t.set_cell(0, 1, "2").unwrap();
        it.table = Some(t);
        d.add_item(it).unwrap();
        let back = from_json(&to_json(&d)).unwrap();
        assert_eq!(back, d);
        assert_eq!(decode_hash(&encode_hash(&d)).unwrap(), d);
        // A v1 document written before tables existed has no `table` key and no table items.
        let old = r#"{"v":1,"view":{"mode":"2d","window":{"min":[-1,-1,-1],"max":[1,1,1]},"angle":"rad"},"items":[{"id":"a","kind":"equation","latex":"y=x"}]}"#;
        assert!(from_json(old).unwrap().items[0].table.is_none());
        // Hostile table data is rejected.
        let mut bad = d.clone();
        bad.items[0].table.as_mut().unwrap().columns[0].name = "not a name".into();
        assert!(from_json(&to_json(&bad)).is_err());
    }

    #[test]
    fn column_styles_round_trip_and_legacy_table_style_migrates() {
        let mut d = Doc::new_default();
        let mut it = Item::new("t1", ItemKind::Table, "");
        let mut t = crate::table::Table::new(&["x_1".into(), "y_1".into(), "y_2".into()], 1);
        t.columns[2].style.color = Some("#6042a6".into());
        t.columns[2].style.lines = true;
        it.table = Some(t);
        d.add_item(it).unwrap();
        assert_eq!(from_json(&to_json(&d)).unwrap(), d);
        assert_eq!(decode_hash(&encode_hash(&d)).unwrap(), d);
        // A document saved with the old table-wide "line" style loads with lines on every column
        // (so it draws as before) and re-saves without the legacy key.
        let old = r#"{"v":1,"view":{"mode":"2d","window":{"min":[-1,-1,-1],"max":[1,1,1]},"angle":"rad"},"items":[{"id":"t","kind":"table","latex":"","table":{"columns":[{"name":"x_1","cells":["1","2"]},{"name":"y_1","cells":["1","4"]}],"style":"line"}}]}"#;
        let got = from_json(old).unwrap();
        let tb = got.items[0].table.as_ref().unwrap();
        assert_eq!(tb.style, crate::table::TableStyle::Points);
        assert!(tb.columns.iter().all(|c| c.style.lines && !c.style.hidden));
        let again = to_json(&got);
        assert!(!again.contains(r#""style":"line""#), "{again}");
        assert_eq!(from_json(&again).unwrap(), got);
        // Hostile column styles are rejected.
        let mut bad = d.clone();
        bad.items[0].table.as_mut().unwrap().columns[1].style.color = Some("javascript:".into());
        assert!(from_json(&to_json(&bad)).is_err());
    }

    #[test]
    fn style_and_view_flags_round_trip_and_defaults_stay_absent() {
        // Defaults are not written, so older documents and hashes are unchanged.
        let d = Doc::new_default();
        let j = to_json(&d);
        for k in [
            "grid",
            "axes",
            "axisNumbers",
            "pointSize",
            "fillOpacity",
            "showLabel",
        ] {
            assert!(!j.contains(k), "{k} in {j}");
        }
        let mut it = Item::new("a", ItemKind::Points, "(1,2)");
        it.style.line_style = Some(LineStyle::Solid);
        let mut d2 = d.clone();
        d2.add_item(it).unwrap();
        let j2 = to_json(&d2);
        assert!(!j2.contains("pointSize") && !j2.contains("showLabel"));
        // A document without the new keys loads with the defaults.
        let g = from_json(GOLDEN).unwrap();
        assert!(g.view.grid && g.view.axes && g.view.axis_numbers);
        assert!(g.items[0].style.point_size.is_none() && !g.items[0].style.show_label);
        // Non-defaults round-trip through JSON and the share hash.
        let mut d = rich_doc();
        d.view.grid = false;
        d.view.axes = false;
        d.view.axis_numbers = false;
        let j = to_json(&d);
        assert!(j.contains("\"grid\":false") && j.contains("\"axisNumbers\":false"));
        assert!(j.contains("\"pointSize\":12.0") && j.contains("\"fillOpacity\":0.4"));
        assert!(j.contains("\"showLabel\":true"));
        assert_eq!(from_json(&j).unwrap(), d);
        assert_eq!(decode_hash(&encode_hash(&d)).unwrap(), d);
        let mut d = rich_doc();
        d.view.axes = false;
        let back = from_json(&to_json(&d)).unwrap();
        assert!(back.view.grid && !back.view.axes && back.view.axis_numbers);
    }

    #[test]
    fn new_style_fields_are_validated() {
        let bad = |f: &dyn Fn(&mut ItemStyle)| {
            let mut d = Doc::new_default();
            let mut it = Item::new("a", ItemKind::Points, "(1,2)");
            f(&mut it.style);
            d.items.push(it);
            d.validate()
        };
        assert!(bad(&|_| {}).is_ok());
        for ok in [1.0, 9.0, 40.0] {
            assert!(bad(&|s| s.point_size = Some(ok)).is_ok(), "{ok}");
        }
        for v in [0.5, 40.5, f64::NAN, f64::INFINITY] {
            match bad(&|s| s.point_size = Some(v)) {
                Err(DocError::Invalid(m)) => assert!(m.contains("pointSize"), "{m}"),
                o => panic!("{v}: {o:?}"),
            }
        }
        for v in [-0.01, 1.01, f64::NAN] {
            match bad(&|s| s.fill_opacity = Some(v)) {
                Err(DocError::Invalid(m)) => assert!(m.contains("fillOpacity"), "{m}"),
                o => panic!("{v}: {o:?}"),
            }
        }
        assert!(bad(&|s| s.fill_opacity = Some(0.0)).is_ok());
        assert!(bad(&|s| s.opacity = Some(f64::NAN)).is_err());
        assert!(bad(&|s| s.line_width = Some(f64::INFINITY)).is_err());
        // Several problems are all reported.
        let e = bad(&|s| {
            s.opacity = Some(2.0);
            s.point_size = Some(0.0);
        });
        assert!(
            matches!(e, Err(DocError::Invalid(m)) if m.contains("opacity") && m.contains("pointSize"))
        );
        // Through JSON as well.
        let j = GOLDEN.replace("\"lineWidth\":2.0", "\"pointSize\":99");
        assert!(matches!(from_json(&j), Err(DocError::Invalid(_))));
        assert!(WindowBox {
            min: [0.0; 3],
            max: [1.0; 3]
        }
        .validate()
        .is_ok());
        assert!(WindowBox {
            min: [0.0; 3],
            max: [1.0, 0.0, 1.0]
        }
        .validate()
        .is_err());
    }
}
