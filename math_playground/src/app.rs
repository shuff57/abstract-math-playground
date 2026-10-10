//! Platform-independent application state: the document, the shared camera rig, theme, input
//! handling and throttled scene rebuilds. Native (winit) and web (wasm + DOM) both drive this
//! through [`App::dispatch`] (JSON commands in, JSON events out) and [`App::frame`], so the two
//! front ends cannot drift apart. No GPU or window types here; fully unit-testable.

use crate::geometry::{ItemInfo, SceneGeometry, Theme};
use crate::render::{layer_lift, mode_fades, Inset, Layer, ModeFades};
use crate::axis_map::AxisMap;
use crate::scene::{
    build_scene_capped_mapped, build_scene_mapped, build_scene_progressive_mapped,
    build_slice_panel_view, clear_surface_tiles, full_surface_depth, label_box_inside_s,
    FIRST_PREVIEW_DEPTH,
    point_handles, CoordSrc, PointHandle, SlicePanel, ViewReq,
};
use math_core::actions;
use math_core::doc::{self, AngleMode, Doc, Item, ItemKind, SliderCfg, TickerCfg};
use math_core::table::{Column, Table, TableStyle};
use math_core::view::{Mode, Rig, Window3, MAX_FRAME_DT_MS};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[path = "curve_grab.rs"]
mod curve_grab;

/// After the last input, wait this long before rebuilding an expensive scene.
const IDLE_REBUILD_MS: f64 = 100.0;
/// Most ticker steps fired by a single `frame` (a long stall drops the backlog instead of
/// replaying it).
pub const MAX_TICKER_STEPS_PER_FRAME: usize = 4;

/// Scenes that build faster than this rebuild on every input (cheap 2D scenes track live).
const CHEAP_BUILD_MS: f64 = 10.0;
/// Milliseconds per frame the staged refinement of a 3D scene may spend meshing: while a switch
/// or the user's input is running, and when the app is idle.
const REFINE_BUDGET_BUSY_MS: f64 = 3.0;
const REFINE_BUDGET_MS: f64 = 5.0;
const REFINE_BUDGET_IDLE_MS: f64 = 9.0;
/// A refined surface mesh replaces the coarser one over this long (see [`App::layers`]).
const SWAP_BLEND_MS: f64 = 160.0;
/// Two presses on the inset within this time and distance are a double-click (reset view).
const DOUBLE_CLICK_MS: f64 = 400.0;
const DOUBLE_CLICK_PX: f64 = 8.0;

/// Distinguishes a missing field (`None`) from an explicit `null` (`Some(None)`).
fn some_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "t", rename_all = "camelCase")]
pub enum Command {
    SetExpr {
        id: String,
        latex: String,
    },
    AddItem {
        id: String,
        kind: ItemKind,
        latex: String,
    },
    RemoveItem {
        id: String,
    },
    MoveItem {
        id: String,
        to: usize,
    },
    SetHidden {
        id: String,
        hidden: bool,
    },
    /// Selects a curve by item id (as a click on it does) so its special points are sent as
    /// `analysis`; a null id clears the selection. For keyboard users, who cannot click.
    SelectItem {
        #[serde(default)]
        id: Option<String>,
    },
    SetColor {
        id: String,
        color: Option<String>,
    },
    /// Regression items: draw (or stop drawing) a tick from each data point to the fitted curve.
    SetRegressionResiduals {
        id: String,
        on: bool,
    },
    /// Merges `style` (an object of `ItemStyle` keys in camelCase) into item `id`'s style; a key
    /// set to `null` goes back to its default. Validated as a whole: on any problem nothing
    /// changes and an `error` event is sent.
    SetStyle {
        id: String,
        style: serde_json::Map<String, serde_json::Value>,
    },
    /// Files item `id` in folder item `folder` (`null` or absent: top level).
    SetFolder {
        id: String,
        #[serde(default)]
        folder: Option<String>,
    },
    /// View settings; every field optional. `window` replaces the window like loading a
    /// document with it (the camera is reset to frame it) without touching anything else.
    SetView {
        #[serde(default)]
        grid: Option<bool>,
        #[serde(default)]
        axes: Option<bool>,
        #[serde(default, rename = "axisNumbers", alias = "axis_numbers")]
        axis_numbers: Option<bool>,
        #[serde(default)]
        window: Option<doc::WindowBox>,
        #[serde(default, rename = "minorGrid", alias = "minor_grid")]
        minor_grid: Option<bool>,
        #[serde(default)]
        arrows: Option<bool>,
        #[serde(default)]
        lock: Option<bool>,
        /// Axis names and steps: absent keeps the value, `null` or an empty string / `0` clears it.
        #[serde(default, rename = "xLabel", deserialize_with = "opt_opt")]
        x_label: Option<Option<String>>,
        #[serde(default, rename = "yLabel", deserialize_with = "opt_opt")]
        y_label: Option<Option<String>>,
        #[serde(default, rename = "xStep", deserialize_with = "opt_opt")]
        x_step: Option<Option<f64>>,
        #[serde(default, rename = "yStep", deserialize_with = "opt_opt")]
        y_step: Option<Option<f64>>,
        /// `"rect"` or `"polar"` 2D grid.
        #[serde(default, rename = "gridKind", alias = "grid_kind")]
        grid_kind: Option<doc::GridKind>,
        /// `"linear"` or `"log"` 2D axes. A logarithmic axis needs a window minimum > 0 on that
        /// axis (the current window, or `window` in the same command); otherwise the command is
        /// rejected and nothing changes.
        #[serde(default, rename = "xScale", alias = "x_scale")]
        x_scale: Option<doc::AxisScale>,
        #[serde(default, rename = "yScale", alias = "y_scale")]
        y_scale: Option<doc::AxisScale>,
        /// `"normal"`, `"bold"` or `"extra"` print weight (thicker lines and bigger points).
        /// Any other value is rejected with an `error` and nothing changes.
        #[serde(default)]
        weight: Option<doc::Weight>,
        /// Text size multiplier of everything drawn as text on the graph (tick numbers, axis
        /// names, item labels, tooltips, inset labels): 1 is the normal size, a number from 0.5
        /// to 3. Any other value is rejected with an `error` and nothing changes.
        #[serde(default, rename = "textScale", alias = "text_scale")]
        text_scale: Option<f64>,
        /// Keep the typed y range on linear axes even when the x and y scales then differ.
        #[serde(default, rename = "freeAspect", alias = "free_aspect")]
        free_aspect: Option<bool>,
    },
    /// Transient pixel scale of the drawn widths for an export rendered at a larger canvas size
    /// (`scale` 2 draws lines twice as wide in pixels, so they keep their look). Not saved in
    /// the document or the share link; 1 (the default) is the on-screen look.
    SetRenderScale {
        scale: f64,
    },
    SetSlider {
        name: String,
        value: f64,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    },
    RemoveSlider {
        name: String,
    },
    /// Saves a slider's playback settings: `mode` (`oscillate`, `loop`, `once`) and `speed`
    /// (a multiplier in (0, 20]). A missing key is left alone; the defaults (`oscillate`, `1`)
    /// are stored as nothing.
    SetSliderPlay {
        name: String,
        mode: Option<doc::SliderPlayMode>,
        speed: Option<f64>,
    },
    SetMode {
        mode: String,
    },
    SetOrtho {
        ortho: bool,
    },
    /// Measurement hook: `on` restores the behaviour of mode switches from before the staged
    /// refinement (blocking depth-5 preview and full rebuild, no frame clamp, no blend, plain
    /// crossfade, no 2D split). Only for before/after timing; never set by the shells.
    SetLegacyTransition {
        on: bool,
    },
    /// Reduced motion (the user's `prefers-reduced-motion`): mode switches and the ortho toggle
    /// jump straight to the end instead of animating.
    SetReducedMotion {
        on: bool,
    },
    SetTheme {
        dark: bool,
    },
    SetAngle {
        angle: String,
    },
    Resize {
        width: u32,
        height: u32,
    },
    Pointer {
        phase: String,
        x: f64,
        y: f64,
        #[serde(default)]
        button: i32,
        #[serde(default)]
        shift: bool,
    },
    /// A click (a press and release without a drag) at canvas pixel `(x, y)`: selects the curve
    /// there, or nothing. See the `analysis` event.
    Pick {
        x: f64,
        y: f64,
    },
    /// Abandons a curve drag in progress (the shell's Escape): the dragged parameters go back to
    /// their values at the press. Does nothing when no curve is being dragged.
    CancelDrag,
    Wheel {
        x: f64,
        y: f64,
        dy: f64,
    },
    Reset,
    LoadDoc {
        json: String,
    },
    LoadHash {
        hash: String,
    },
    /// Ask for the current document / share hash.
    Export,
    /// New data table. `columns` are header names (default `x_1`, `y_1`); `data` is row-major
    /// initial cells (strings or numbers); otherwise `rows` blank rows (default 3, or 0 when
    /// `data` is given).
    AddTable {
        id: String,
        #[serde(default)]
        columns: Vec<String>,
        #[serde(default)]
        data: Vec<Vec<serde_json::Value>>,
        rows: Option<usize>,
    },
    /// Sets one cell (`value` is a string or number); a row past the end grows the table.
    SetCell {
        id: String,
        row: usize,
        col: usize,
        value: serde_json::Value,
    },
    AddRow {
        id: String,
        at: Option<usize>,
    },
    RemoveRow {
        id: String,
        row: usize,
    },
    /// Appends a column; `name` defaults to the next free `x_n`/`y_n`-style header.
    AddColumn {
        id: String,
        name: Option<String>,
    },
    RemoveColumn {
        id: String,
        col: usize,
    },
    RenameColumn {
        id: String,
        col: usize,
        name: String,
    },
    /// Sets column `col`'s formula over the other columns' names (`x_1^2+1`), or clears it with
    /// `null` or blank text. A formula column's cells are computed and cannot be edited.
    SetColumnFormula {
        id: String,
        col: usize,
        formula: Option<String>,
    },
    /// Legacy table-wide style, applied to every column: `points`, `line` or `hidden`.
    SetTableStyle {
        id: String,
        style: String,
    },
    /// Merges `style` into column `col`'s [`math_core::table::ColumnStyle`] (`color`, `hidden`,
    /// `points`, `lines`, `pointStyle`, `pointSize`, `opacity`, `outline`; `null` resets a key).
    SetTableColumnStyle {
        id: String,
        col: usize,
        style: serde_json::Map<String, serde_json::Value>,
    },
    /// Evaluates action item `id` once (`a -> a+1, b -> 2b`) and applies the result.
    RunAction {
        id: String,
    },
    /// Configures the ticker. Every field is optional; `action: null` clears the action (and
    /// stops), a missing `action` leaves it alone. `rate_ms` is the interval (default 50, never
    /// faster than `min_step_ms`, default 10). `running` starts or stops it.
    SetTicker {
        #[serde(default, deserialize_with = "some_option")]
        action: Option<Option<String>>,
        #[serde(default, alias = "rateMs")]
        rate_ms: Option<f64>,
        #[serde(default, alias = "minStepMs")]
        min_step_ms: Option<f64>,
        #[serde(default, alias = "pauseOnError")]
        pause_on_error: Option<bool>,
        #[serde(default)]
        running: Option<bool>,
    },
    /// Fires the ticker action once now, whether or not the ticker is running.
    TickerStep,
    StartTicker,
    StopTicker,
    ToggleTicker,
    /// Shows a cross-section of the same items: in 3D `{"z":0.5}` (a plane, `dim` 2) or
    /// `{"y":1,"z":"a"}` (a line, `dim` 1); in 2D `{"y":"a"}` (a line, `dim` 1). A constant is a
    /// number or an expression of slider names, so dragging the slider sweeps the slice. `fixed`
    /// is an object or a string like `"y=1,z=0.5"` (aliases `plane`, `axis`, `at`); `dim` is
    /// optional (derived from the mode and the number of fixed axes, an error if it disagrees).
    SetSlice {
        #[serde(default)]
        dim: Option<u8>,
        #[serde(alias = "plane", alias = "axis", alias = "at", alias = "axes")]
        fixed: serde_json::Value,
    },
    ClearSlice,
    /// Sets the slice inset's own view window (it then stops following the main window).
    /// `min`/`max` are `[xmin, ymin]` / `[xmax, ymax]` in the inset's axes (horizontal and
    /// vertical free axis; for a line slice the vertical axis is the value `f`); `view` is the
    /// same as `[xmin, xmax, ymin, ymax]`. For a plane slice the vertical span is re-fitted to the
    /// inset's aspect around its centre. No-op error when there is no slice.
    SetSliceView {
        #[serde(default)]
        min: Option<[f64; 2]>,
        #[serde(default)]
        max: Option<[f64; 2]>,
        #[serde(default)]
        view: Option<[f64; 4]>,
    },
    /// Returns the inset to auto-follow (also a double-click on it).
    ResetSliceView,
    /// Shows or hides the slice inset (default shown). Hidden, the slice plane is still drawn in
    /// the 3D scene and the slice state is still reported; only the inset panel and its labels go.
    SetSliceInset {
        show: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ItemColor {
    pub id: String,
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiagItem {
    pub id: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelOut {
    pub pos: [f64; 3],
    pub text: String,
    pub axis: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScreenLabel {
    pub text: String,
    pub axis: u8,
    pub x: f64,
    pub y: f64,
    pub visible: bool,
    /// Opacity in [0, 1]: below 1 only while a mode switch fades the label in or out.
    pub alpha: f64,
    /// True for a label of the slice inset (its `x`/`y` are already offset into the canvas);
    /// axis 0/1 are its tick labels and axis names, 3 is reserved for a title.
    #[serde(default)]
    pub inset: bool,
    /// Rectangle `[x, y, w, h]` the label must stay inside (the inset's, for inset labels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip: Option<[f64; 4]>,
    /// Id of the item an item label (`showLabel`, axis 4/6) belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    /// The item's `labelOffset` `[dx, dy]` in CSS pixels (y down, clamped): the overlay adds it
    /// to `x`/`y` (canvas pixels / dpr) so the text is moved off its point. Absent when none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<[f64; 2]>,
    /// The item's `labelSize` as a text scale (0.75 small, 1.45 large). Absent for medium.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
}

/// A curve parameter and its value (see [`Event::CurveDrag`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParamValue {
    pub name: String,
    pub value: f64,
}

/// One special point of the selected curve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnalysisPoint {
    /// `root`, `y-intercept`, `minimum`, `maximum` or `inflection`.
    pub kind: &'static str,
    pub x: f64,
    pub y: f64,
    /// The point on the canvas in pixels from the top-left.
    pub px: f64,
    pub py: f64,
    /// For an `intersection`: the id of the other curve. Absent for every other kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub with: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "t", rename_all = "camelCase")]
pub enum Event {
    /// Per-item problems, shown inline in the expression list (never logged per frame).
    Diagnostics {
        items: Vec<DiagItem>,
    },
    /// Per-item read-outs (symbolic derivative, scalar value, regression fit), the full current
    /// list, sent whenever it changes (an empty list clears them). See `ItemInfo`.
    Info {
        items: Vec<ItemInfo>,
    },
    /// Resolved colour of every drawn item (`#rrggbb`), so list badges match the canvas. Sent
    /// only when it changes.
    Colors {
        items: Vec<ItemColor>,
    },
    /// Current mode and window, so a UI can show/edit them, and the view flags.
    View {
        mode: String,
        min: [f64; 3],
        max: [f64; 3],
        grid: bool,
        axes: bool,
        #[serde(rename = "axisNumbers")]
        axis_numbers: bool,
        #[serde(rename = "minorGrid")]
        minor_grid: bool,
        arrows: bool,
        lock: bool,
        #[serde(rename = "xLabel")]
        x_label: Option<String>,
        #[serde(rename = "yLabel")]
        y_label: Option<String>,
        #[serde(rename = "xStep")]
        x_step: Option<f64>,
        #[serde(rename = "yStep")]
        y_step: Option<f64>,
        #[serde(rename = "gridKind")]
        grid_kind: doc::GridKind,
        #[serde(rename = "xScale")]
        x_scale: doc::AxisScale,
        #[serde(rename = "yScale")]
        y_scale: doc::AxisScale,
        weight: doc::Weight,
        #[serde(rename = "textScale")]
        text_scale: f64,
    },
    /// Tick labels for the current scene in world coordinates, for a text overlay to project.
    Labels {
        labels: Vec<LabelOut>,
    },
    Doc {
        json: String,
    },
    Hash {
        hash: String,
    },
    /// Full state of a table, sent after every table change and for each table on load.
    Table {
        id: String,
        columns: Vec<Column>,
        rows: usize,
        style: String,
        /// True when a drag of a table point on the canvas caused it (the shell records the
        /// undo step, which it does itself for its own table commands).
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        drag: bool,
    },
    /// The pointer is over a curve `y = f(x)` (2D only): `item` is its id and `(x, y)` the point of
    /// the curve nearest the pointer in world coordinates; `px`/`py` is that point on the canvas in
    /// pixels from the top-left. `item` is null (and the numbers 0) when the pointer left every
    /// curve. Sent only when the result changes.
    Hover {
        item: Option<String>,
        x: f64,
        y: f64,
        px: f64,
        py: f64,
        /// The parameters a drag of this curve would change (names of sliders or numeric
        /// definitions in its equation): non-empty only when the curve is the selected one and
        /// has any, i.e. when a press here would drag it instead of panning.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        params: Vec<String>,
    },
    /// A curve drag started, moved or ended. `params` holds the dragged parameters' current
    /// values; `(px, py)` is the pointer on the canvas. `active` is false on the last event of
    /// the drag, with `cancelled` true when the values were restored.
    CurveDrag {
        item: String,
        params: Vec<ParamValue>,
        active: bool,
        cancelled: bool,
        px: f64,
        py: f64,
    },
    /// The special points of the selected curve over the visible window (an empty list and a null
    /// `item` when nothing is selected). Sent when the selection or the points change.
    Analysis {
        item: Option<String>,
        points: Vec<AnalysisPoint>,
    },
    /// A drag rewrote an item's text (a point `(a, b)` or a definition `a=3`): the shell should
    /// put `latex` into that item's input.
    ItemEdited {
        id: String,
        latex: String,
    },
    /// A drag moved a slider.
    SliderValue {
        name: String,
        value: f64,
    },
    Theme {
        dark: bool,
    },
    /// State of the slice, sent whenever it changes (also while a slider sweeps it). `active` is
    /// false when there is no slice or it does not fit the current mode (`error` says why).
    /// `fixed` holds the resolved constants, `free` the axes of the slice, `rect` the inset
    /// rectangle `[x, y, w, h]` in canvas pixels from the top-left, `curves` the items with a
    /// cut curve (2D slice) and `points` the marked points (1D slice roots, parametric
    /// crossings, points near the slice). `view` is the window the inset shows,
    /// `[xmin, xmax, ymin, ymax]` (null when inactive) and `follow` is true while it follows the
    /// main window / auto-fits (false after a drag, wheel zoom or `setSliceView`).
    Slice {
        active: bool,
        dim: u8,
        fixed: BTreeMap<String, f64>,
        free: Vec<String>,
        rect: Option<[u32; 4]>,
        curves: usize,
        points: usize,
        error: Option<String>,
        view: Option<[f64; 4]>,
        follow: bool,
    },
    /// The ticker started, stopped (also automatically, e.g. after an error) or was reconfigured.
    TickerState {
        running: bool,
        action: Option<String>,
    },
    Error {
        message: String,
    },
}

struct Built {
    geometry: SceneGeometry,
    origin: [f64; 3],
    mode: Mode,
    /// A flat (1D/2D) scene during a switch with 3D, split into its backdrop (grid, axes) and
    /// its items (curves, regions), which fade on different schedules (see [`mode_fades`]).
    parts: Option<Box<(SceneGeometry, SceneGeometry)>>,
}

/// A 3D scene's surfaces projected to the canvas: tick numbers inside them are not drawn.
struct SurfaceCover {
    tris: Vec<[[f64; 2]; 3]>,
    /// Screen bounds `[x0, y0, x1, y1]` of all the triangles.
    bounds: [f64; 4],
}

impl SurfaceCover {
    /// Whether the text box of a tick label anchored at `(x, y)` lies on a surface (its centre or
    /// the anchor point is inside a triangle).
    fn covers(&self, axis: u8, text: &str, x: f64, y: f64) -> bool {
        let tw = text.chars().count() as f64 * 7.5;
        let centre = match axis {
            0 => [x, y + 11.5],
            1 => [x - 6.0 - tw / 2.0, y],
            _ => [x, y],
        };
        [centre, [x, y]].iter().any(|p| self.hit(*p))
    }

    /// Whether the label's anchor lies inside the surface's screen bounds without being on it:
    /// such a number floats beside the object, so it is faded rather than dropped.
    fn near(&self, x: f64, y: f64) -> bool {
        let b = self.bounds;
        x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]
    }

    fn hit(&self, p: [f64; 2]) -> bool {
        self.tris.iter().any(|t| {
            if p[0] < t[0][0].min(t[1][0]).min(t[2][0])
                || p[0] > t[0][0].max(t[1][0]).max(t[2][0])
                || p[1] < t[0][1].min(t[1][1]).min(t[2][1])
                || p[1] > t[0][1].max(t[1][1]).max(t[2][1])
            {
                return false;
            }
            let e = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            let (d0, d1, d2) = (e(t[0], t[1]), e(t[1], t[2]), e(t[2], t[0]));
            (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
        })
    }
}

/// `g` split at its backdrop: (grid and axes, everything else).
fn split_flat(g: &SceneGeometry) -> (SceneGeometry, SceneGeometry) {
    let n = g.backdrop_segments.min(g.segments.len());
    let backdrop = SceneGeometry { segments: g.segments[..n].to_vec(), ..SceneGeometry::default() };
    let items = SceneGeometry {
        fields: g.fields.clone(),
        segments: g.segments[n..].to_vec(),
        overlay_segments: g.overlay_segments.clone(),
        vertices: g.vertices.clone(),
        indices: g.indices.clone(),
        flat_indices: g.flat_indices.clone(),
        ..SceneGeometry::default()
    };
    (backdrop, items)
}

/// A refined 3D scene is blending in over the coarser one it replaced.
struct Swap {
    old: Built,
    start_ms: f64,
}

/// Octree depths a 3D scene's surfaces are built at, in order: a first look ([`FIRST_PREVIEW_DEPTH`],
/// cheap enough to build between a click and the first moving frame), a middle one, then full
/// quality. Returns the depth after `cap`, or `None` when `cap` is already `full`.
pub fn next_refine_depth(cap: u32, full: u32) -> Option<u32> {
    const MID: u32 = 5;
    if cap >= full {
        None
    } else if cap < MID && MID < full {
        Some(MID)
    } else {
        Some(full)
    }
}

/// Smoothstep of how far a swap has progressed (`elapsed` ms into `SWAP_BLEND_MS`).
pub fn swap_blend(elapsed_ms: f64) -> f32 {
    let t = if elapsed_ms.is_finite() { (elapsed_ms / SWAP_BLEND_MS).clamp(0.0, 1.0) } else { 1.0 } as f32;
    t * t * (3.0 - 2.0 * t)
}

struct Drag {
    button: u8,
    shift: bool,
    last: (f64, f64),
    /// Set when the press grabbed a point item: the move edits it instead of panning.
    point: Option<PointGrab>,
    /// Set when the press grabbed the selected curve: the move changes its parameters.
    curve: Option<curve_grab::CurveGrab>,
}

struct PointGrab {
    handle: PointHandle,
    /// Point position minus pointer world position at the press, so the point does not jump.
    offset: [f64; 2],
}

/// Pointer distance (px) within which a press grabs a point.
const GRAB_PX: f64 = 14.0;

/// Pointer distance (px) within which a hover finds a curve.
const HOVER_PX: f64 = 10.0;

/// Formats `v` with `decimals` places, trailing zeros trimmed.
fn fmt_coord(v: f64, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, v);
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

fn cell_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub struct App {
    pub doc: Doc,
    pub rig: Rig,
    pub theme: Theme,
    size: (u32, u32),
    current: Option<Built>,
    /// Outgoing scene while a mode switch tweens.
    prev: Option<Built>,
    dirty: bool,
    redraw: bool,
    now_ms: f64,
    last_input_ms: f64,
    last_build_ms: f64,
    drag: Option<Drag>,
    diagnostics: Vec<DiagItem>,
    infos: Vec<ItemInfo>,
    colors: Vec<ItemColor>,
    outbox: Vec<Event>,
    /// Compiled `y = f(x)` curves for hover, with the `build_rev` they were made for.
    hover_curves: Option<(u64, Vec<crate::scene::ExplicitCurve>)>,
    /// Implicit, polar and parametric curves for hit-testing, with their polylines (display
    /// coordinates) once a hover needed them.
    hover_shapes: Option<(u64, Vec<(crate::scene::ShapeCurve, Option<Vec<Vec<[f64; 2]>>>)>)>,
    /// Counts rebuilds; a hover cache older than this is stale.
    build_rev: u64,
    /// The curve a click selected (its item id), whose special points are sent as `analysis`.
    selected: Option<String>,
    /// The last `analysis` sent, to send changes only.
    last_analysis: Option<(Option<String>, Vec<AnalysisPoint>)>,
    /// The last hover result sent (item and rounded pixel), to send changes only.
    last_hover: Option<(String, i64, i64, bool)>,
    /// The parameter names a drag of the selected curve would change, for the document revision
    /// and item they were worked out for.
    grab_probe: Option<(u64, String, Vec<String>)>,
    ticker_running: bool,
    /// Time accumulated towards the next ticker step.
    ticker_acc_ms: f64,
    /// `now_ms` at the previous ticker update (`None` until the first frame after starting).
    ticker_last_ms: Option<f64>,
    /// Last ticker error reported, so a persistent error is not repeated every step.
    ticker_err: Option<String>,
    /// Secondary inset for the slice, rebuilt with the main scene.
    panel: Option<SlicePanel>,
    /// The inset is folded away by the UI (the slice itself stays on).
    inset_hidden: bool,
    /// The 3D view was moved closer when the slice came on (undone when it goes).
    slice_zoomed: bool,
    /// The 3D box has not been touched since the document / mode was set up, so the first build
    /// with content may shrink the default cube to the content (see [`fit_default_cube`]).
    fit3d: bool,
    /// The window is the auto-fitted z = -1..5 one: an edit that draws below it gets the +-5 cube.
    low_z_fit: bool,
    /// Last `Slice` event sent, so it is only repeated when it changes.
    last_slice: Option<Event>,
    /// Explicit inset view (`None` follows the main window).
    slice_view: Option<ViewReq>,
    /// Pointer position of an inset drag in progress.
    inset_drag: Option<(f64, f64)>,
    /// Time and place of the last press on the inset (double-click detection).
    last_inset_press: Option<(f64, f64, f64)>,
    /// The inset view changed and its geometry must be re-sampled (throttled like the scene).
    panel_dirty: bool,
    last_panel_ms: f64,
    /// The current scene is a preview (coarser surfaces, built so a mode switch starts moving
    /// at once); it is rebuilt at full quality when the switch has finished.
    refine: bool,
    /// Longest time one frame advances a mode-switch tween (see [`Rig::max_frame_dt_ms`]).
    pub max_frame_dt_ms: f64,
    /// Surface depth the staged refinement is working towards next (valid while `refine`).
    refine_cap: u32,
    /// Meshing time spent on the running refinement, ms (it becomes `last_build_ms`).
    refine_ms: f64,
    /// A refined scene blending in over the one it replaced.
    swap: Option<Swap>,
    /// `setLegacyTransition` (measurement only).
    legacy_transition: bool,
    /// `setReducedMotion`; kept here because loading a document replaces the rig.
    reduced_motion: bool,
    /// World -> display map of logarithmic 2D axes: the rig's window is in display coordinates
    /// while in 2D (see [`AxisMap`]); fixed between `load` / `setView` so pans stay smooth.
    map: AxisMap,
    /// The world window before the axes became logarithmic, so switching back to linear
    /// restores the exact previous view.
    log_entry: Option<doc::WindowBox>,
    /// Transient pixel scale of drawn widths (export at a larger size); never saved.
    render_scale: f32,
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::D1 => "1d",
        Mode::D2 => "2d",
        Mode::D3 => "3d",
    }
}

fn parse_mode(s: &str) -> Option<Mode> {
    match s {
        "1d" => Some(Mode::D1),
        "2d" => Some(Mode::D2),
        "3d" => Some(Mode::D3),
        _ => None,
    }
}

fn doc_mode(m: Mode) -> doc::Mode {
    match m {
        Mode::D1 => doc::Mode::D1,
        Mode::D2 => doc::Mode::D2,
        Mode::D3 => doc::Mode::D3,
    }
}

fn view_mode(m: doc::Mode) -> Mode {
    match m {
        doc::Mode::D1 => Mode::D1,
        doc::Mode::D2 => Mode::D2,
        doc::Mode::D3 => Mode::D3,
    }
}

/// Default 3D dolly: the cube fills about 85% of the canvas height (the bare bounding-sphere
/// framing leaves it near half).
const FIT_3D: f64 = 1.35;

/// Largest distance from the origin of any item geometry (lit surfaces and item curves or dots,
/// not the grid, axes, slice overlay or flat plane), in world coordinates. `None` when nothing
/// is drawn.
/// Also returns the lowest `z` drawn.
fn content_extent(g: &SceneGeometry, origin: [f64; 3]) -> Option<(f64, f64)> {
    let mut m: f64 = 0.0;
    let mut zmin = f64::INFINITY;
    let mut upd = |p: [f32; 3]| {
        for a in 0..3 {
            let v = p[a] as f64 + origin[a];
            if v.is_finite() {
                m = m.max(v.abs());
                if a == 2 {
                    zmin = zmin.min(v);
                }
            }
        }
    };
    for &i in &g.indices {
        if let Some(v) = g.vertices.get(i as usize) {
            upd(v.pos);
        }
    }
    for s in g.segments.iter().skip(g.backdrop_segments) {
        upd(s.p0);
        upd(s.p1);
    }
    (m > 0.0).then_some((m, zmin))
}

/// The default +-10 cube shrunk to +-5 for content that fits well inside it; any other window
/// (the document set its own) is left alone. A surface that stays on or above the `xy` plane
/// (a paraboloid) gets `z` in -1..8, so its bowl (z = x^2+y^2 reaches 8 at r = 2.8) and the plane fit under the box top.
fn fit_default_cube(w: Window3, (extent, zmin): (f64, f64)) -> Option<Window3> {
    let default = w.min.iter().all(|v| *v == -10.0) && w.max.iter().all(|v| *v == 10.0);
    if !(default && (extent < 5.0 || extent >= 9.9)) {
        return None;
    }
    let mut c = Window3::new([-5.0; 3], [5.0; 3]);
    if zmin >= -1e-6 {
        c.min[2] = -1.0;
        c.max[2] = 8.0;
    }
    Some(c)
}

/// Extra dolly while a 3D slice is active.
const SLICE_ZOOM_3D: f64 = 1.25;

/// A rig framed for the default 3D view (see [`FIT_3D`]).
fn fitted_rig(window: Window3, mode: Mode) -> Rig {
    let mut rig = Rig::new(window, mode);
    rig.dolly(FIT_3D);
    rig
}

impl App {
    pub fn new(size: (u32, u32)) -> Self {
        let doc = Doc::new_default();
        let w = doc.view.window.clone();
        let mut rig = fitted_rig(Window3::new(w.min, w.max), view_mode(doc.view.mode));
        rig.set_aspect(size.0 as f64 / size.1.max(1) as f64);
        rig.max_frame_dt_ms = MAX_FRAME_DT_MS;
        let mut app = App {
            max_frame_dt_ms: MAX_FRAME_DT_MS,
            doc,
            rig,
            theme: Theme::light(),
            size,
            current: None,
            prev: None,
            dirty: true,
            redraw: true,
            now_ms: 0.0,
            last_input_ms: f64::NEG_INFINITY,
            last_build_ms: 0.0,
            drag: None,
            diagnostics: Vec::new(),
            infos: Vec::new(),
            colors: Vec::new(),
            outbox: Vec::new(),
            hover_curves: None,
            hover_shapes: None,
            selected: None,
            last_analysis: None,
            build_rev: 0,
            last_hover: None,
            grab_probe: None,
            ticker_running: false,
            ticker_acc_ms: 0.0,
            ticker_last_ms: None,
            ticker_err: None,
            panel: None,
            inset_hidden: false,
            slice_zoomed: false,
            fit3d: true,
            low_z_fit: false,
            last_slice: None,
            slice_view: None,
            inset_drag: None,
            last_inset_press: None,
            panel_dirty: false,
            last_panel_ms: 0.0,
            refine: false,
            refine_cap: 0,
            refine_ms: 0.0,
            swap: None,
            legacy_transition: false,
            reduced_motion: false,
            map: AxisMap::LINEAR,
            log_entry: None,
            render_scale: 1.0,
        };
        app.rebuild();
        app
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn background(&self) -> [f32; 4] {
        self.theme.background
    }

    pub fn mode(&self) -> Mode {
        self.rig.mode()
    }

    /// Takes the events produced since the last call.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.outbox)
    }

    /// [`App::take_events`] as a JSON array: events raised outside `dispatch` (inside `frame`:
    /// ticker `itemEdited`/`sliderValue`/`error`/`tickerState`, throttled slice rebuilds).
    pub fn drain_json(&mut self) -> String {
        serde_json::to_string(&self.take_events()).unwrap_or_else(|_| "[]".into())
    }

    /// Runs one JSON command and returns the resulting events as a JSON array. Malformed input
    /// yields an `error` event, never a panic.
    pub fn dispatch(&mut self, json: &str) -> String {
        match serde_json::from_str::<Command>(json) {
            Ok(cmd) => self.run(cmd),
            Err(e) => self.outbox.push(Event::Error {
                message: format!("bad command: {e}"),
            }),
        }
        let events = self.take_events();
        serde_json::to_string(&events).unwrap_or_else(|_| "[]".into())
    }

    /// Inset rectangle `[x, y, w, h]` if a slice inset is showing.
    fn inset_rect(&self) -> Option<[u32; 4]> {
        self.panel
            .as_ref()
            .filter(|_| !self.rig.is_animating() && !self.inset_hidden)
            .map(|p| p.rect)
    }

    /// The inset's current view with its free axes: the explicit request, else what is shown.
    fn current_inset_view(&self) -> Option<ViewReq> {
        let p = self.panel.as_ref()?;
        Some(self.slice_view.clone().unwrap_or(ViewReq {
            free: p.free.clone(),
            view: p.view,
        }))
    }

    /// Stores an explicit inset view and schedules the (throttled) re-sample.
    fn set_inset_view(&mut self, req: ViewReq) {
        self.slice_view = Some(req);
        self.panel_dirty = true;
        self.redraw = true;
    }

    fn reset_inset_view(&mut self) {
        if self.slice_view.take().is_some() {
            self.rebuild_panel(self.rig.mode());
        }
    }

    fn in_inset(&self, x: f64, y: f64) -> bool {
        self.inset_rect().is_some_and(|r| {
            x >= r[0] as f64
                && x <= (r[0] + r[2]) as f64
                && y >= r[1] as f64
                && y <= (r[1] + r[3]) as f64
        })
    }

    fn touch_input(&mut self) {
        self.last_input_ms = self.now_ms;
        self.redraw = true;
    }

    fn mark_doc_changed(&mut self) {
        self.dirty = true;
        self.redraw = true;
        // Edits rebuild immediately so typing feels live.
        self.rebuild();
    }

    pub fn run(&mut self, cmd: Command) {
        let vp = (self.size.0 as f64, self.size.1 as f64);
        match cmd {
            Command::SetExpr { id, latex } => {
                match self.doc.items.iter_mut().find(|i| i.id == id) {
                    Some(item) => item.latex = latex,
                    None => {
                        let item = Item::new(&id, ItemKind::Equation, &latex);
                        if let Err(e) = self.doc.add_item(item) {
                            self.outbox.push(Event::Error {
                                message: e.to_string(),
                            });
                            return;
                        }
                    }
                }
                self.mark_doc_changed();
            }
            Command::AddItem { id, kind, latex } => {
                let mut item = Item::new(&id, kind, &latex);
                if kind == ItemKind::Table {
                    item.table = Some(Table::new(&[], 3));
                }
                if let Err(e) = self.doc.add_item(item) {
                    self.outbox.push(Event::Error {
                        message: e.to_string(),
                    });
                    return;
                }
                self.mark_doc_changed();
                self.emit_table(&id);
            }
            Command::RemoveItem { id } => {
                if let Err(e) = self.doc.remove_item(&id) {
                    self.outbox.push(Event::Error {
                        message: e.to_string(),
                    });
                    return;
                }
                if self
                    .doc
                    .ticker
                    .as_ref()
                    .is_some_and(|t| t.action.as_deref() == Some(id.as_str()))
                {
                    self.configure_ticker(|t| t.action = None);
                    self.ticker_running = false;
                    self.emit_ticker_state();
                }
                self.mark_doc_changed();
            }
            Command::MoveItem { id, to } => {
                if let Err(e) = self.doc.move_item(&id, to) {
                    self.outbox.push(Event::Error {
                        message: e.to_string(),
                    });
                    return;
                }
                self.mark_doc_changed();
            }
            Command::SelectItem { id } => {
                let known = id.as_ref().is_none_or(|id| self.doc.items.iter().any(|i| &i.id == id));
                if known && self.selected != id {
                    self.selected = id;
                    self.refresh_analysis();
                }
            }
            Command::SetHidden { id, hidden } => {
                if let Some(i) = self.doc.items.iter_mut().find(|i| i.id == id) {
                    i.hidden = hidden;
                    self.mark_doc_changed();
                }
            }
            Command::SetRegressionResiduals { id, on } => {
                if let Some(i) = self.doc.items.iter_mut().find(|i| i.id == id) {
                    i.style.residuals = on;
                    self.mark_doc_changed();
                } else {
                    self.outbox.push(Event::Error {
                        message: format!("no item with id '{id}'"),
                    });
                }
            }
            Command::SetStyle { id, style } => match self.merge_style(&id, style) {
                Ok(()) => self.mark_doc_changed(),
                Err(message) => self.outbox.push(Event::Error { message }),
            },
            Command::SetFolder { id, folder } => match self.set_folder(&id, folder) {
                Ok(()) => self.mark_doc_changed(),
                Err(message) => self.outbox.push(Event::Error { message }),
            },
            Command::SetView {
                grid,
                axes,
                axis_numbers,
                window,
                minor_grid,
                arrows,
                lock,
                x_label,
                y_label,
                x_step,
                y_step,
                grid_kind,
                x_scale,
                y_scale,
                weight,
                text_scale,
                free_aspect,
            } => {
                if let Some(w) = &window {
                    if let Err(m) = w.validate() {
                        self.outbox.push(Event::Error {
                            message: format!("setView: {m}"),
                        });
                        return;
                    }
                }
                // Logarithmic axes need a positive window on that axis: check the result first.
                let scales_change = x_scale.is_some_and(|s| s != self.doc.view.x_scale)
                    || y_scale.is_some_and(|s| s != self.doc.view.y_scale);
                let mut next = self.doc.view.clone();
                next.x_scale = x_scale.unwrap_or(next.x_scale);
                next.y_scale = y_scale.unwrap_or(next.y_scale);
                let world_now = self.world_window();
                if let Err(m) = next.check_log_window(Some(window.as_ref().unwrap_or(&world_now))) {
                    self.outbox.push(Event::Error {
                        message: format!("setView: {m}"),
                    });
                    return;
                }
                for (n, st) in [("xStep", x_step), ("yStep", y_step)] {
                    if let Some(Some(v)) = st {
                        if v != 0.0 && !(v.is_finite() && v > 0.0) {
                            self.outbox.push(Event::Error {
                                message: format!("setView: {n} must be a positive number"),
                            });
                            return;
                        }
                    }
                }
                if let Some(t) = text_scale {
                    if !doc::valid_text_scale(t) {
                        self.outbox.push(Event::Error {
                            message: format!(
                                "setView: textScale must be a number from {} to {}",
                                doc::TEXT_SCALE_MIN,
                                doc::TEXT_SCALE_MAX
                            ),
                        });
                        return;
                    }
                }
                let aspect_change = free_aspect.is_some_and(|f| f != self.doc.view.free_aspect);
                let v = &mut self.doc.view;
                v.free_aspect = free_aspect.unwrap_or(v.free_aspect);
                v.text_scale = text_scale.unwrap_or(v.text_scale);
                v.grid = grid.unwrap_or(v.grid);
                v.axes = axes.unwrap_or(v.axes);
                v.axis_numbers = axis_numbers.unwrap_or(v.axis_numbers);
                v.minor_grid = minor_grid.unwrap_or(v.minor_grid);
                v.arrows = arrows.unwrap_or(v.arrows);
                v.lock = lock.unwrap_or(v.lock);
                let name = |t: Option<Option<String>>, cur: Option<String>| match t {
                    None => cur,
                    Some(t) => t
                        .map(|t| t.trim().chars().take(64).collect::<String>())
                        .filter(|t| !t.is_empty()),
                };
                v.x_label = name(x_label, v.x_label.take());
                v.y_label = name(y_label, v.y_label.take());
                let step = |t: Option<Option<f64>>, cur: Option<f64>| match t {
                    None => cur,
                    Some(t) => t.filter(|v| *v > 0.0),
                };
                v.x_step = step(x_step, v.x_step);
                v.y_step = step(y_step, v.y_step);
                v.grid_kind = grid_kind.unwrap_or(v.grid_kind);
                v.weight = weight.unwrap_or(v.weight);
                if aspect_change && !(scales_change || window.is_some()) {
                    // Only the aspect rule changed: re-frame the same world window under it.
                    self.frame_world_window(world_now.clone());
                }
                if scales_change || window.is_some() {
                    let was_linear =
                        self.doc.view.x_scale.is_linear() && self.doc.view.y_scale.is_linear();
                    self.doc.view.x_scale = next.x_scale;
                    self.doc.view.y_scale = next.y_scale;
                    let now_linear = next.x_scale.is_linear() && next.y_scale.is_linear();
                    // Back to linear: the exact view from before the axes became logarithmic (a
                    // linear view of a log window would be squashed into a sliver).
                    let entry = self.log_entry.take();
                    let restore = match &entry {
                        Some(before) if now_linear && window.is_none() => Some(before.clone()),
                        _ => None,
                    };
                    if !now_linear {
                        self.log_entry = entry;
                    }
                    let w = window.or(restore).unwrap_or(world_now.clone());
                    self.frame_world_window(w);
                    if was_linear && !now_linear {
                        self.log_entry = Some(world_now);
                    }
                }
                self.mark_doc_changed();
            }
            Command::SetRenderScale { scale } => {
                let scale = if scale.is_finite() && scale > 0.0 {
                    scale.min(8.0) as f32
                } else {
                    1.0
                };
                if scale != self.render_scale {
                    self.render_scale = scale;
                    self.dirty = true;
                    self.redraw = true;
                    self.rebuild();
                }
            }
            Command::SetColor { id, color } => {
                if let Some(i) = self.doc.items.iter_mut().find(|i| i.id == id) {
                    i.color = color;
                    self.mark_doc_changed();
                }
            }
            Command::SetSlider {
                name,
                value,
                min,
                max,
                step,
            } => {
                let cfg = self.doc.sliders.entry(name).or_insert(SliderCfg {
                    min: -10.0,
                    max: 10.0,
                    step: None,
                    value,
                });
                cfg.value = value;
                if let Some(m) = min {
                    cfg.min = m;
                }
                if let Some(m) = max {
                    cfg.max = m;
                }
                if step.is_some() {
                    cfg.step = step;
                }
                self.mark_doc_changed();
                self.emit_formula_tables();
            }
            Command::RemoveSlider { name } => {
                let had_play = self.doc.slider_play.remove(&name).is_some();
                if self.doc.sliders.remove(&name).is_some() || had_play {
                    self.mark_doc_changed();
                }
            }
            Command::SetSliderPlay { name, mode, speed } => {
                if !self.doc.sliders.contains_key(&name) {
                    self.outbox.push(Event::Error {
                        message: format!("setSliderPlay: no slider named '{name}'"),
                    });
                } else if speed.is_some_and(|s| !(s.is_finite() && s > 0.0 && s <= 20.0)) {
                    self.outbox.push(Event::Error {
                        message: "setSliderPlay: speed must be in (0, 20]".into(),
                    });
                } else {
                    let mut p = self.doc.slider_play.get(&name).cloned().unwrap_or_default();
                    if let Some(m) = mode {
                        p.mode = (m != doc::SliderPlayMode::Oscillate).then_some(m);
                    }
                    if let Some(s) = speed {
                        p.speed = (s != 1.0).then_some(s);
                    }
                    if p.is_default() {
                        self.doc.slider_play.remove(&name);
                    } else {
                        self.doc.slider_play.insert(name, p);
                    }
                    self.mark_doc_changed();
                }
            }
            Command::SetMode { mode } => match parse_mode(&mode) {
                Some(m) => self.set_mode(m),
                None => self.outbox.push(Event::Error {
                    message: format!("unknown mode '{mode}'"),
                }),
            },
            Command::SetOrtho { ortho } => {
                self.rig.reduced_motion = self.reduced_motion;
                self.rig.set_ortho3(ortho, self.now_ms);
                self.touch_input();
            }
            Command::SetLegacyTransition { on } => {
                self.legacy_transition = on;
                self.max_frame_dt_ms = if on { f64::INFINITY } else { MAX_FRAME_DT_MS };
                self.rig.max_frame_dt_ms = self.max_frame_dt_ms;
            }
            Command::SetReducedMotion { on } => {
                self.reduced_motion = on;
                self.rig.reduced_motion = on;
            }
            Command::SetTheme { dark } => {
                self.theme = if dark { Theme::dark() } else { Theme::light() };
                self.outbox.push(Event::Theme { dark });
                self.mark_doc_changed();
            }
            Command::SetAngle { angle } => {
                self.doc.view.angle = if angle == "deg" {
                    AngleMode::Deg
                } else {
                    AngleMode::Rad
                };
                self.mark_doc_changed();
            }
            Command::Resize { width, height } => {
                self.size = (width.max(1), height.max(1));
                self.rig.set_aspect(self.size.0 as f64 / self.size.1 as f64);
                self.dirty = true;
                self.touch_input();
                // A resize right after a mode switch (the layout changes with the mode) must not
                // block on the full 3D build: it restarts the staged refinement instead.
                let staged = !self.legacy_transition
                    && (self.rig.is_animating()
                        || self.refine
                        || (self.rig.mode() == Mode::D3 && self.last_build_ms >= CHEAP_BUILD_MS));
                self.rebuild_with(staged);
            }
            Command::Pointer {
                phase,
                x,
                y,
                button,
                shift,
            } => self.pointer(&phase, x, y, button.clamp(0, 255) as u8, shift, vp),
            Command::Pick { x, y } => self.pick(x, y, vp),
            Command::CancelDrag => self.cancel_curve_drag(),
            Command::Wheel { x, y, dy } if self.in_inset(x, y) => {
                if let (Some(r), Some(base)) = (self.inset_rect(), self.current_inset_view()) {
                    let factor = (-dy * 0.0015).exp();
                    let (fx, fy) = (
                        (x - r[0] as f64) / r[2] as f64,
                        1.0 - (y - r[1] as f64) / r[3] as f64,
                    );
                    let v = math_core::slice::zoom_view(
                        base.view,
                        fx.clamp(0.0, 1.0),
                        fy.clamp(0.0, 1.0),
                        factor,
                    );
                    self.set_inset_view(ViewReq {
                        free: base.free,
                        view: v,
                    });
                    self.touch_input();
                }
            }
            // A locked 2D / 1D view ignores zoom and pan; 3D can still be orbited.
            Command::Wheel { .. } if self.doc.view.lock && self.rig.mode() != Mode::D3 => {}
            Command::Wheel { x, y, dy } => {
                let factor = (-dy * 0.0015).exp();
                self.rig.zoom_at((x, y), factor, vp);
                self.dirty = true;
                self.touch_input();
            }
            Command::SetSlice { dim, fixed } => {
                let mode = self.rig.mode();
                let cfg = math_core::slice::fixed_from_json(&fixed)
                    .and_then(|f| math_core::slice::SliceCfg::new(dim, f, mode));
                match cfg {
                    Ok(c) => {
                        // A slice makes the object the subject: a sphere of radius 3 should fill
                        // about a quarter of the scene, not a sliver (undone on clear).
                        // (not for content that already fills the box, like a paraboloid in its z = -1..8 box: the
                        // extra zoom would push its top out of view)
                        let fills = self
                            .current
                            .as_ref()
                            .and_then(|b| content_extent(&b.geometry, b.origin))
                            .is_some_and(|(m, _)| {
                                let w = self.rig.window();
                                m >= 0.75 * w.min.iter().chain(w.max.iter()).fold(0.0f64, |a, v| a.max(v.abs()))
                            });
                        if mode == Mode::D3 && self.doc.slice.is_none() && !self.slice_zoomed && !fills {
                            self.rig.dolly(SLICE_ZOOM_3D);
                            self.slice_zoomed = true;
                            self.dirty = true;
                        }
                        self.doc.slice = Some(c);
                        self.mark_doc_changed();
                    }
                    Err(message) => self.outbox.push(Event::Error { message }),
                }
            }
            Command::ClearSlice => {
                if self.slice_zoomed {
                    self.slice_zoomed = false;
                    self.rig.dolly(1.0 / SLICE_ZOOM_3D);
                    self.dirty = true;
                }
                self.doc.slice = None;
                self.slice_view = None;
                self.mark_doc_changed();
            }
            Command::SetSliceView { min, max, view } => {
                let v = view.or_else(|| match (min, max) {
                    (Some(a), Some(b)) => Some([a[0], b[0], a[1], b[1]]),
                    _ => None,
                });
                match (v, self.current_inset_view()) {
                    (_, None) => self.outbox.push(Event::Error { message: "setSliceView: there is no slice inset".into() }),
                    (None, _) => self.outbox.push(Event::Error {
                        message: "setSliceView needs min and max ([x, y] each) or view [xmin, xmax, ymin, ymax]".into(),
                    }),
                    (Some(v), Some(base)) if !math_core::slice::view_valid(&v) => {
                        let _ = base;
                        self.outbox.push(Event::Error { message: "setSliceView: the view must be finite with max > min".into() });
                    }
                    (Some(v), Some(base)) => {
                        self.set_inset_view(ViewReq { free: base.free, view: v });
                        self.rebuild_panel(self.rig.mode());
                    }
                }
            }
            Command::SetSliceInset { show } => {
                if self.inset_hidden == show {
                    self.inset_hidden = !show;
                    self.redraw = true;
                }
            }
            Command::ResetSliceView => {
                if self.slice_view.take().is_some() {
                    self.rebuild_panel(self.rig.mode());
                }
            }
            Command::Reset => {
                self.rig.reset();
                self.rig.dolly(FIT_3D);
                self.slice_zoomed = false;
                self.dirty = true;
                self.touch_input();
            }
            Command::LoadDoc { json } => match doc::from_json(&json) {
                Ok(d) => self.load(d),
                Err(e) => self.outbox.push(Event::Error {
                    message: e.to_string(),
                }),
            },
            Command::LoadHash { hash } => match doc::decode_hash(&hash) {
                Ok(d) => self.load(d),
                Err(e) => self.outbox.push(Event::Error {
                    message: e.to_string(),
                }),
            },
            Command::AddTable {
                id,
                columns,
                data,
                rows,
            } => {
                let mut t = Table::new(
                    &columns,
                    rows.unwrap_or(if data.is_empty() { 3 } else { 0 })
                        .min(math_core::table::MAX_ROWS),
                );
                for n in &columns {
                    if !math_core::table::valid_column_name(n) {
                        self.outbox.push(Event::Error {
                            message: format!("'{n}' is not a valid column name"),
                        });
                        return;
                    }
                }
                for (ri, row) in data.iter().enumerate() {
                    for (ci, v) in row.iter().enumerate().take(t.columns.len()) {
                        if let Err(e) = t.set_cell(ri, ci, &cell_text(v)) {
                            self.outbox.push(Event::Error { message: e });
                            return;
                        }
                    }
                }
                let mut item = Item::new(&id, ItemKind::Table, "");
                item.table = Some(t);
                if let Err(e) = self.doc.add_item(item) {
                    self.outbox.push(Event::Error {
                        message: e.to_string(),
                    });
                    return;
                }
                self.mark_doc_changed();
                self.emit_table(&id);
            }
            Command::SetCell {
                id,
                row,
                col,
                value,
            } => {
                let v = cell_text(&value);
                self.edit_table(&id, |t| t.set_cell(row, col, &v));
            }
            Command::AddRow { id, at } => self.edit_table(&id, |t| t.add_row(at)),
            Command::RemoveRow { id, row } => self.edit_table(&id, |t| t.remove_row(row)),
            Command::AddColumn { id, name } => {
                self.edit_table(&id, |t| t.add_column(name.as_deref()))
            }
            Command::RemoveColumn { id, col } => self.edit_table(&id, |t| t.remove_column(col)),
            Command::RenameColumn { id, col, name } => {
                self.edit_table(&id, |t| t.rename_column(col, &name))
            }
            Command::SetColumnFormula { id, col, formula } => {
                self.edit_table(&id, |t| t.set_formula(col, formula.as_deref()))
            }
            Command::SetTableStyle { id, style } => match TableStyle::parse(&style) {
                Some(st) => self.edit_table(&id, |t| {
                    t.apply_table_style(st);
                    Ok(())
                }),
                None => self.outbox.push(Event::Error {
                    message: format!("unknown table style '{style}'"),
                }),
            },
            Command::SetTableColumnStyle { id, col, style } => {
                self.edit_table(&id, |t| t.set_column_style(col, &style))
            }
            Command::RunAction { id } => match self.step_action(&id) {
                Ok(()) => self.mark_doc_changed(),
                Err(message) => self.outbox.push(Event::Error { message }),
            },
            Command::SetTicker {
                action,
                rate_ms,
                min_step_ms,
                pause_on_error,
                running,
            } => self.set_ticker(action, rate_ms, min_step_ms, pause_on_error, running),
            Command::TickerStep => match self.ticker_action_id() {
                Ok(id) => match self.step_action(&id) {
                    Ok(()) => self.mark_doc_changed(),
                    Err(message) => self.outbox.push(Event::Error { message }),
                },
                Err(message) => self.outbox.push(Event::Error { message }),
            },
            Command::StartTicker => self.set_ticker(None, None, None, None, Some(true)),
            Command::StopTicker => self.set_ticker(None, None, None, None, Some(false)),
            Command::ToggleTicker => {
                let run = !self.ticker_running;
                self.set_ticker(None, None, None, None, Some(run))
            }
            Command::Export => {
                self.sync_view_into_doc();
                self.outbox.push(Event::Doc {
                    json: doc::to_json(&self.doc),
                });
                self.outbox.push(Event::Hash {
                    hash: doc::encode_hash(&self.doc),
                });
            }
        }
    }

    /// `setStyle`: the item's style with `patch` merged in (`null` resets a key), validated.
    fn merge_style(
        &mut self,
        id: &str,
        patch: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        const KEYS: [&str; 16] = [
            "lineWidth",
            "lineStyle",
            "pointStyle",
            "opacity",
            "label",
            "residuals",
            "pointSize",
            "fillOpacity",
            "showLabel",
            "residualPlot",
            "labelOffset",
            "dragMode",
            "asFraction",
            "labelSize",
            "pointOutline",
            "logMode",
        ];
        let Some(item) = self.doc.items.iter_mut().find(|i| i.id == id) else {
            return Err(format!("no item with id '{id}'"));
        };
        if let Some(k) = patch.keys().find(|k| !KEYS.contains(&k.as_str())) {
            return Err(format!(
                "setStyle: unknown style key '{k}' (expected one of {})",
                KEYS.join(", ")
            ));
        }
        let mut merged = match serde_json::to_value(&item.style) {
            Ok(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        for (k, v) in &patch {
            if v.is_null() {
                merged.remove(k);
            } else {
                merged.insert(k.clone(), v.clone());
            }
        }
        let style: doc::ItemStyle = serde_json::from_value(serde_json::Value::Object(merged))
            .map_err(|e| format!("setStyle: {e}"))?;
        style.validate().map_err(|m| format!("setStyle: {m}"))?;
        let mut style = style;
        // An offset beyond the limit is pulled back, not refused (a drag may overshoot).
        style.label_offset = style
            .label_offset
            .map(|o| o.map(|v| v.clamp(-doc::LABEL_OFFSET_MAX, doc::LABEL_OFFSET_MAX)));
        if let Some(w) = patch.get("lineWidth").and(style.line_width) {
            if !(0.25..=20.0).contains(&w) {
                return Err("setStyle: lineWidth must be in [0.25,20]".into());
            }
        }
        if style
            .label
            .as_ref()
            .is_some_and(|l| l.chars().count() > doc::MAX_LATEX_CHARS)
        {
            return Err(format!(
                "setStyle: label longer than {} chars",
                doc::MAX_LATEX_CHARS
            ));
        }
        item.style = style;
        Ok(())
    }

    /// `setFolder`: files `id` in an existing folder item (not itself, not another folder), or
    /// un-files it with `None`.
    fn set_folder(&mut self, id: &str, folder: Option<String>) -> Result<(), String> {
        let Some(kind) = self.doc.items.iter().find(|i| i.id == id).map(|i| i.kind) else {
            return Err(format!("no item with id '{id}'"));
        };
        if let Some(f) = &folder {
            if f == id {
                return Err("setFolder: an item cannot be in itself".into());
            }
            if !self
                .doc
                .items
                .iter()
                .any(|i| i.id == *f && i.kind == ItemKind::Folder)
            {
                return Err(format!("setFolder: '{f}' is not a folder item"));
            }
            if kind == ItemKind::Folder {
                return Err("setFolder: folders cannot be nested".into());
            }
        }
        if let Some(i) = self.doc.items.iter_mut().find(|i| i.id == id) {
            i.folder = folder;
        }
        Ok(())
    }

    fn load(&mut self, d: Doc) {
        let w = d.view.window.clone();
        self.rig = fitted_rig(Window3::new(w.min, w.max), view_mode(d.view.mode));
        self.rig.set_aspect(self.size.0 as f64 / self.size.1 as f64);
        self.rig.max_frame_dt_ms = self.max_frame_dt_ms;
        self.doc = d;
        self.log_entry = None;
        self.frame_world_window(w);
        self.fit3d = true;
        self.prev = None;
        self.swap = None;
        self.ticker_running = false;
        self.ticker_err = None;
        self.mark_doc_changed();
        self.emit_ticker_state();
        let ids: Vec<String> = self
            .doc
            .items
            .iter()
            .filter(|i| i.table.is_some())
            .map(|i| i.id.clone())
            .collect();
        for id in ids {
            self.emit_table(&id);
        }
    }

    fn emit_ticker_state(&mut self) {
        let action = self.doc.ticker.as_ref().and_then(|t| t.action.clone());
        self.outbox.push(Event::TickerState {
            running: self.ticker_running,
            action,
        });
    }

    /// Edits the ticker config in place, dropping it from the document when it is all defaults.
    fn configure_ticker(&mut self, f: impl FnOnce(&mut TickerCfg)) {
        let mut cfg = self.doc.ticker.take().unwrap_or_default();
        f(&mut cfg);
        self.doc.ticker = (cfg != TickerCfg::default()).then_some(cfg);
    }

    /// The configured ticker action id, or why there is none.
    fn ticker_action_id(&self) -> Result<String, String> {
        self.doc
            .ticker
            .as_ref()
            .and_then(|t| t.action.clone())
            .ok_or_else(|| "the ticker has no action (setTicker {action: <item id>})".to_string())
    }

    fn set_ticker(
        &mut self,
        action: Option<Option<String>>,
        rate_ms: Option<f64>,
        min_step_ms: Option<f64>,
        pause_on_error: Option<bool>,
        running: Option<bool>,
    ) {
        for (name, v) in [("rate_ms", rate_ms), ("min_step_ms", min_step_ms)] {
            if let Some(v) = v {
                if !v.is_finite() || v < 0.0 || (name == "rate_ms" && v == 0.0) {
                    self.outbox.push(Event::Error {
                        message: format!("ticker {name} must be a positive number"),
                    });
                    return;
                }
            }
        }
        if let Some(Some(id)) = &action {
            if let Err(message) = actions::parse_item(&self.doc, id) {
                // A malformed action text still reports precisely why on first run; only an
                // unknown or non-action item is refused here.
                if !self.doc.items.iter().any(|i| i.id == *id) || message.contains("not an action")
                {
                    self.outbox.push(Event::Error { message });
                    return;
                }
            }
        }
        self.configure_ticker(|t| {
            if let Some(a) = action {
                t.action = a;
            }
            if let Some(r) = rate_ms {
                t.rate_ms = r.min(1e9);
            }
            if let Some(m) = min_step_ms {
                t.min_step_ms = m.min(1e9);
            }
            if let Some(p) = pause_on_error {
                t.pause_on_error = p;
            }
        });
        let mut want = running.unwrap_or(self.ticker_running);
        if self.ticker_action_id().is_err() {
            if want && running == Some(true) {
                self.outbox.push(Event::Error {
                    message: "the ticker has no action to run".into(),
                });
            }
            want = false;
        }
        if want && !self.ticker_running {
            self.ticker_acc_ms = 0.0;
            self.ticker_last_ms = None;
            self.ticker_err = None;
        }
        self.ticker_running = want;
        self.emit_ticker_state();
    }

    /// Evaluates action item `id` against the current values and applies the result: sliders
    /// through the slider table (so uniform-driven fields only change a parameter), numeric
    /// definitions by rewriting their item text. Atomic: on error nothing changes. The caller
    /// rebuilds.
    fn step_action(&mut self, id: &str) -> Result<(), String> {
        let action = actions::parse_item(&self.doc, id)?;
        let changes = actions::plan(&self.doc, &action)?;
        for c in changes {
            match c.target {
                actions::TargetKind::Slider => {
                    if let Some(cfg) = self.doc.sliders.get_mut(&c.name) {
                        if cfg.value != c.value {
                            cfg.value = c.value;
                            self.outbox.push(Event::SliderValue {
                                name: c.name,
                                value: c.value,
                            });
                        }
                    }
                }
                actions::TargetKind::Definition { item } => {
                    let latex = actions::definition_text(&c.name, c.value);
                    if let Some(it) = self.doc.items.iter_mut().find(|i| i.id == item) {
                        if it.latex != latex {
                            it.latex = latex.clone();
                            self.outbox.push(Event::ItemEdited { id: item, latex });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Fires due ticker steps (at most [`MAX_TICKER_STEPS_PER_FRAME`]; a longer stall drops the
    /// backlog) and marks the scene dirty so the normal throttled rebuild picks them up.
    fn tick(&mut self, now_ms: f64) {
        if !self.ticker_running {
            return;
        }
        let Ok(id) = self.ticker_action_id() else {
            self.ticker_running = false;
            self.emit_ticker_state();
            return;
        };
        let cfg = self.doc.ticker.clone().unwrap_or_default();
        let last = self.ticker_last_ms.replace(now_ms).unwrap_or(now_ms);
        self.ticker_acc_ms += (now_ms - last).max(0.0);
        let interval = cfg.interval_ms();
        let due = (self.ticker_acc_ms / interval).floor();
        let n = due.min(MAX_TICKER_STEPS_PER_FRAME as f64) as usize;
        if due > MAX_TICKER_STEPS_PER_FRAME as f64 {
            self.ticker_acc_ms = 0.0;
        } else {
            self.ticker_acc_ms -= n as f64 * interval;
        }
        let mut changed = false;
        for _ in 0..n {
            match self.step_action(&id) {
                Ok(()) => {
                    changed = true;
                    self.ticker_err = None;
                }
                Err(message) => {
                    if self.ticker_err.as_deref() != Some(message.as_str()) {
                        self.outbox.push(Event::Error {
                            message: format!("ticker: {message}"),
                        });
                        self.ticker_err = Some(message);
                    }
                    if cfg.pause_on_error {
                        self.ticker_running = false;
                        self.emit_ticker_state();
                    }
                    break;
                }
            }
        }
        if changed {
            self.dirty = true;
            self.redraw = true;
        }
    }

    /// Applies `f` to the table of item `id`, then rebuilds and reports the new table state.
    /// Errors (unknown id, bad name, limits) become an `error` event and change nothing.
    fn edit_table(&mut self, id: &str, f: impl FnOnce(&mut Table) -> Result<(), String>) {
        let Some(t) = self
            .doc
            .items
            .iter_mut()
            .find(|i| i.id == id)
            .and_then(|i| i.table.as_mut())
        else {
            self.outbox.push(Event::Error {
                message: format!("no table with id '{id}'"),
            });
            return;
        };
        let mut edited = t.clone();
        match f(&mut edited) {
            Ok(()) => *t = edited,
            Err(message) => {
                self.outbox.push(Event::Error { message });
                return;
            }
        }
        self.mark_doc_changed();
        self.emit_table(id);
    }

    fn emit_table(&mut self, id: &str) {
        self.emit_table_inner(id, false);
    }

    /// [`App::emit_table`] for a change made by dragging one of its points.
    fn emit_table_drag(&mut self, id: &str) {
        self.emit_table_inner(id, true);
    }

    fn emit_table_inner(&mut self, id: &str, drag: bool) {
        if let Some(t) = self
            .doc
            .items
            .iter()
            .find(|i| i.id == id)
            .and_then(|i| i.table.as_ref())
        {
            let mut columns = t.columns.clone();
            if columns.iter().any(|c| c.formula.is_some()) {
                // A formula column's cells are the computed values (the workspace shows them
                // read-only); the formula text itself rides on the column.
                let (parsed, _) = t.parse(&math_core::parse::ParseCtx::new());
                let mut bind = math_core::list::Bindings::new().with_angle(match self.doc.view.angle {
                    AngleMode::Rad => math_core::compile::Angle::Rad,
                    AngleMode::Deg => math_core::compile::Angle::Deg,
                });
                for (name, s) in &self.doc.sliders {
                    bind.set(name, math_core::list::Value::Num(s.value));
                }
                for (c, p) in columns.iter_mut().zip(&parsed) {
                    if c.formula.is_none() {
                        continue;
                    }
                    c.cells = p
                        .cells
                        .iter()
                        .map(|cell| match cell {
                            None => String::new(),
                            Some(e) => match math_core::list::eval_value(e, &bind) {
                                Ok(math_core::list::Value::Num(v)) => {
                                    crate::scene::format_value(v)
                                }
                                _ => String::new(),
                            },
                        })
                        .collect();
                }
            }
            let (rows, style) = (t.rows(), t.summary_style().name().to_string());
            self.outbox.push(Event::Table {
                id: id.to_string(),
                columns,
                rows,
                style,
                drag,
            });
        }
    }

    /// Re-sends every table that has a formula column (its computed cells follow the sliders).
    fn emit_formula_tables(&mut self) {
        let ids: Vec<String> = self
            .doc
            .items
            .iter()
            .filter(|i| {
                i.table
                    .as_ref()
                    .is_some_and(|t| t.columns.iter().any(|c| c.formula.is_some()))
            })
            .map(|i| i.id.clone())
            .collect();
        for id in ids {
            self.emit_table(&id);
        }
    }

    /// Writes a dragged point position back into the document (see [`CoordSrc`]) and tells the
    /// shell what changed.
    fn drag_point(&mut self, h: &PointHandle, world: [f64; 2]) {
        let w = self.rig.window();
        let upp = ((w.max[0] - w.min[0]) / self.size.0.max(1) as f64).max(1e-300);
        let m = self.active_map();
        let mut pieces = [h.text[0].clone(), h.text[1].clone()];
        let mut literal = false;
        let mut touched_tables: Vec<String> = Vec::new();
        for k in 0..2 {
            // World units per pixel at the point (on a log axis they grow with the value).
            let upp = if m.log[k] {
                (world[k].abs() * std::f64::consts::LN_10 / m.k[k] * upp).max(1e-300)
            } else {
                upp
            };
            let decimals = (-upp.log10().floor()).clamp(0.0, 12.0) as usize;
            let v = (world[k] * 10f64.powi(decimals as i32)).round() / 10f64.powi(decimals as i32);
            match &h.src[k] {
                CoordSrc::Literal => {
                    pieces[k] = fmt_coord(v, decimals);
                    literal = true;
                }
                CoordSrc::Slider(name) => {
                    if let Some(cfg) = self.doc.sliders.get_mut(name) {
                        let mut v = v.clamp(cfg.min, cfg.max);
                        if let Some(st) = cfg.step.filter(|s| *s > 0.0) {
                            v = ((v - cfg.min) / st).round() * st + cfg.min;
                        }
                        if cfg.value != v {
                            cfg.value = v;
                            self.outbox.push(Event::SliderValue {
                                name: name.clone(),
                                value: v,
                            });
                        }
                    }
                }
                CoordSrc::Def { item, name } => {
                    let latex = format!("{name}={}", fmt_coord(v, decimals));
                    if let Some(it) = self.doc.items.iter_mut().find(|i| i.id == *item) {
                        if it.latex != latex {
                            it.latex = latex.clone();
                            self.outbox.push(Event::ItemEdited {
                                id: item.clone(),
                                latex,
                            });
                        }
                    }
                }
                CoordSrc::Cell { item, col, row } => {
                    let text = fmt_coord(v, decimals);
                    if let Some(tb) = self
                        .doc
                        .items
                        .iter_mut()
                        .find(|i| i.id == *item)
                        .and_then(|i| i.table.as_mut())
                    {
                        if tb.columns.get(*col).and_then(|c| c.cells.get(*row)) != Some(&text) {
                            let _ = tb.set_cell(*row, *col, &text);
                            if !touched_tables.contains(item) {
                                touched_tables.push(item.clone());
                            }
                        }
                    }
                }
                CoordSrc::Fixed => {}
            }
        }
        for id in touched_tables {
            self.emit_table_drag(&id);
        }
        if literal {
            let latex = format!("({}, {})", pieces[0], pieces[1]);
            if let Some(it) = self.doc.items.iter_mut().find(|i| i.id == h.id) {
                if it.latex != latex {
                    it.latex = latex.clone();
                    self.outbox.push(Event::ItemEdited {
                        id: h.id.clone(),
                        latex,
                    });
                }
            }
        }
        self.mark_doc_changed();
    }

    /// Finds the curve `y = f(x)` nearest the pointer (within `HOVER_PX`, 2D only) and sends a
    /// `Hover` event when the result differs from the last one sent.
    fn hover(&mut self, x: f64, y: f64, vp: (f64, f64)) {
        let found = self.curve_under(x, y, vp);
        let params = found.as_ref().map(|f| self.grab_params(&f.0)).unwrap_or_default();
        let key = found
            .as_ref()
            .map(|(id, _, _, px, py)| (id.clone(), px.round() as i64, py.round() as i64, !params.is_empty()));
        if key == self.last_hover {
            return;
        }
        self.last_hover = key;
        self.outbox.push(match found {
            Some((id, wx, wy, px, py)) => Event::Hover { item: Some(id), x: wx, y: wy, px, py, params },
            None => Event::Hover { item: None, x: 0.0, y: 0.0, px: 0.0, py: 0.0, params: Vec::new() },
        });
    }

    /// Compiles the document's `y = f(x)` curves again if it was rebuilt since the last time.
    fn ensure_curves(&mut self) {
        let stale = self.hover_curves.as_ref().is_none_or(|(rev, _)| *rev != self.build_rev);
        if stale {
            self.hover_curves = Some((self.build_rev, crate::scene::explicit_curves(&self.doc)));
        }
        let stale = self.hover_shapes.as_ref().is_none_or(|(rev, _)| *rev != self.build_rev);
        if stale {
            let shapes = crate::scene::shape_curves(&self.doc, self.active_map());
            self.hover_shapes = Some((self.build_rev, shapes.into_iter().map(|c| (c, None)).collect()));
        }
    }

    /// A click: selects the curve under the pointer (or clears the selection) and sends its
    /// special points.
    fn pick(&mut self, x: f64, y: f64, vp: (f64, f64)) {
        self.selected = self.curve_under(x, y, vp).map(|c| c.0);
        self.refresh_analysis();
        // The hover tip says whether a drag would grab the curve, which the click just changed.
        self.last_hover = None;
        self.hover(x, y, vp);
    }

    /// Sends `analysis` for the selected curve over the visible window if it differs from the
    /// last one sent. A selection that no longer names a drawn curve (deleted, edited into
    /// another kind, hidden, or not in 2D) is dropped.
    fn refresh_analysis(&mut self) {
        use math_core::special::analyze;
        // Nothing selected and nothing to clear: no work (this runs after every rebuild).
        if self.selected.is_none() && self.last_analysis.as_ref().is_none_or(|l| l.0.is_none()) {
            return;
        }
        self.ensure_curves();
        let vp = (self.size.0 as f64, self.size.1 as f64);
        let m = self.active_map();
        let mut points: Vec<AnalysisPoint> = Vec::new();
        match (self.selected.clone(), self.rig.mode() == Mode::D2) {
            (Some(id), true) => {
                let curve = self.hover_curves.as_ref().and_then(|(_, cs)| cs.iter().find(|c| c.id == id));
                match curve {
                    Some(c) => {
                        let w = self.world_window();
                        let mut f = |x: f64| c.prog.eval(&[x]);
                        let mut g1 = c.d1.as_ref().map(|p| move |x: f64| p.eval(&[x]));
                        let mut g2 = c.d2.as_ref().map(|p| move |x: f64| p.eval(&[x]));
                        // A {x..}/{y..} range clips the curve: look only inside it.
                        let span = match &c.restrict {
                            Some(r) => r.x_interval(w.min[0], w.max[0]),
                            None => Some((w.min[0], w.max[0])),
                        };
                        let found = match span {
                            Some((a, b)) => analyze(
                                &mut f,
                                g1.as_mut().map(|g| g as &mut dyn FnMut(f64) -> f64),
                                g2.as_mut().map(|g| g as &mut dyn FnMut(f64) -> f64),
                                a,
                                b,
                            ),
                            None => Vec::new(),
                        };
                        points = found
                            .iter()
                            .filter(|s| c.restrict.as_ref().is_none_or(|r| r.contains(s.x, s.y)))
                            .map(|s| {
                                let (px, py) = self.rig.world_to_pixel(m.fwd3([s.x, s.y, 0.0]), vp);
                                AnalysisPoint { kind: s.kind.name(), x: s.x, y: s.y, px, py, with: None }
                            })
                            .collect();
                    }
                    // Implicit, polar and parametric curves can be picked and dragged but have
                    // no analysis (no special points).
                    None if self.hover_shapes.as_ref().is_some_and(|(_, cs)| cs.iter().any(|(c, _)| c.id == id)) => {}
                    None => self.selected = None,
                }
                if self.selected.is_some() {
                    self.add_intersections(&id, &mut points, vp);
                }
            }
            (Some(_), false) => self.selected = None,
            _ => {}
        }
        let now = (self.selected.clone(), points);
        if self.last_analysis.as_ref() == Some(&now) {
            return;
        }
        self.outbox.push(Event::Analysis { item: now.0.clone(), points: now.1.clone() });
        self.last_analysis = Some(now);
    }

    /// Appends `intersection` points of curve `id` with the other visible curves to `points`
    /// (see `curve_pairs`). Special points are trimmed so at least 16 slots stay free for them;
    /// the total never exceeds 48.
    fn add_intersections(&self, id: &str, points: &mut Vec<AnalysisPoint>, vp: (f64, f64)) {
        const CAP: usize = 48;
        const RESERVED: usize = 16;
        let m = self.active_map();
        let w = m.to_world(self.scene_window());
        let (Some((_, ex)), Some((_, sh))) = (self.hover_curves.as_ref(), self.hover_shapes.as_ref()) else { return };
        let shapes: Vec<&crate::scene::ShapeCurve> = sh.iter().map(|(c, _)| c).collect();
        let order = |other: &str| self.doc.items.iter().position(|i| i.id == other).unwrap_or(usize::MAX);
        let hits = crate::curve_pairs::with_others(id, ex, &shapes, &order, m, [w.min[0], w.max[0], w.min[1], w.max[1]]);
        let (sx, sy) = ((w.max[0] - w.min[0]).abs(), (w.max[1] - w.min[1]).abs());
        let mut found: Vec<AnalysisPoint> = Vec::new();
        for (other, x, y) in hits {
            // A root, intercept or extremum already listed at the same spot is not repeated.
            let dup = points.iter().chain(found.iter()).any(|p| (p.x - x).abs() <= 1e-6 * sx && (p.y - y).abs() <= 1e-6 * sy);
            if dup {
                continue;
            }
            let (px, py) = self.rig.world_to_pixel(m.fwd3([x, y, 0.0]), vp);
            found.push(AnalysisPoint { kind: "intersection", x, y, px, py, with: Some(other) });
        }
        points.truncate(CAP - found.len().min(RESERVED));
        found.truncate(CAP - points.len());
        points.extend(found);
    }

    /// `(item, world x, world y, pixel x, pixel y)` of the curve point nearest the pointer, looking
    /// `HOVER_PX` to each side of it so steep curves are found too.
    fn curve_under(&mut self, x: f64, y: f64, vp: (f64, f64)) -> Option<(String, f64, f64, f64, f64)> {
        if self.rig.mode() != Mode::D2 || self.rig.is_animating() {
            return None;
        }
        self.ensure_curves();
        let m = self.active_map();
        let mut best: Option<(f64, (String, f64, f64, f64, f64))> = None;
        for c in &self.hover_curves.as_ref()?.1 {
            for k in -(HOVER_PX as i32)..=(HOVER_PX as i32) {
                let wx = m.inv3(self.rig.pixel_to_world((x + k as f64, y), vp))[0];
                let wy = c.prog.eval(&[wx]);
                if !wy.is_finite() || c.restrict.as_ref().is_some_and(|r| !r.contains(wx, wy)) {
                    continue;
                }
                let (px, py) = self.rig.world_to_pixel(m.fwd3([wx, wy, 0.0]), vp);
                let d = (px - x).hypot(py - y);
                if d <= HOVER_PX && best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                    best = Some((d, (c.id.clone(), wx, wy, px, py)));
                }
            }
        }
        // Implicit, polar and parametric curves: distance to the drawn polyline, in pixels.
        let here = self.rig.pixel_to_world((x, y), vp);
        let (bx, by) = (self.rig.pixel_to_world((x + 1.0, y), vp), self.rig.pixel_to_world((x, y + 1.0), vp));
        let u = [(bx[0] - here[0]).abs().max(1e-300), (by[1] - here[1]).abs().max(1e-300)];
        let win = self.scene_window();
        let size = (self.size.0 as f64, self.size.1 as f64);
        if let Some((_, shapes)) = self.hover_shapes.as_mut() {
            for (c, lines) in shapes.iter_mut() {
                let lines = lines.get_or_insert_with(|| crate::scene::shape_lines(c, win, size));
                let mut found: Option<(f64, [f64; 2])> = None;
                for l in lines.iter() {
                    for seg in l.windows(2) {
                        let (a, b) = ((seg[0][0] - here[0]) / u[0], (seg[0][1] - here[1]) / u[1]);
                        let (c2, e) = ((seg[1][0] - seg[0][0]) / u[0], (seg[1][1] - seg[0][1]) / u[1]);
                        let len2 = c2 * c2 + e * e;
                        let t = if len2 > 0.0 { (-(a * c2 + b * e) / len2).clamp(0.0, 1.0) } else { 0.0 };
                        let d = (a + t * c2).hypot(b + t * e);
                        if d <= HOVER_PX && found.is_none_or(|(bd, _)| d < bd) {
                            found = Some((d, [seg[0][0] + t * (seg[1][0] - seg[0][0]), seg[0][1] + t * (seg[1][1] - seg[0][1])]));
                        }
                    }
                }
                if let Some((d, q)) = found {
                    if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                        let q = [q[0], q[1], 0.0];
                        let w = m.inv3(q);
                        let (px, py) = self.rig.world_to_pixel(q, vp);
                        best = Some((d, (c.id.clone(), w[0], w[1], px, py)));
                    }
                }
            }
        }
        best.map(|(_, r)| r)
    }

    /// The point item (if any) under the pointer, in 2D only.
    fn grab_point(&self, x: f64, y: f64, vp: (f64, f64)) -> Option<PointGrab> {
        if self.rig.mode() != Mode::D2 || self.rig.is_animating() {
            return None;
        }
        let here = self.rig.pixel_to_world((x, y), vp);
        let m = self.active_map();
        point_handles(&self.doc)
            .into_iter()
            .filter_map(|h| {
                // Handles are world positions; the rig works in display coordinates.
                let d = m.fwd3([h.pos[0], h.pos[1], 0.0]);
                let (px, py) = self.rig.world_to_pixel(d, vp);
                let d = (px - x).hypot(py - y);
                (d <= GRAB_PX).then_some((d, h))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, h)| {
                let d = m.fwd3([h.pos[0], h.pos[1], 0.0]);
                PointGrab {
                    offset: [d[0] - here[0], d[1] - here[1]],
                    handle: h,
                }
            })
    }

    fn sync_view_into_doc(&mut self) {
        let w = self.world_window();
        self.doc.view.mode = doc_mode(self.rig.mode());
        let v = &mut self.doc.view;
        for a in 0..3 {
            // A logarithmic axis keeps its last valid range (a 3D pan may have crossed 0).
            let log = a < 2 && [v.x_scale, v.y_scale][a].is_log();
            if !log || (w.min[a] > 0.0 && w.max[a] > w.min[a] && w.max[a].is_finite()) {
                v.window.min[a] = w.min[a];
                v.window.max[a] = w.max[a];
            }
        }
    }

    /// The axis map in effect: the logarithmic axes apply in 2D only.
    fn active_map(&self) -> AxisMap {
        if self.rig.mode() == Mode::D2 {
            self.map
        } else {
            AxisMap::LINEAR
        }
    }

    /// The rig's window in world coordinates.
    fn world_window(&self) -> doc::WindowBox {
        self.active_map().to_world(self.rig.window())
    }

    /// Sets the document's world window and a fresh rig framing it in the current mode (as
    /// `load` does), recomputing the axis map for the view's scales and the canvas aspect.
    fn frame_world_window(&mut self, w: doc::WindowBox) {
        let aspect = self.size.0 as f64 / self.size.1.max(1) as f64;
        self.doc.view.window = w.clone();
        self.map = AxisMap::for_view(&self.doc.view, aspect);
        let shown = if self.rig.mode() == Mode::D2 {
            self.map.to_display(&w)
        } else {
            Window3::new(w.min, w.max)
        };
        self.rig = fitted_rig(shown, self.rig.mode());
        self.rig.set_aspect(aspect);
        self.rig.reduced_motion = self.reduced_motion;
        self.rig.max_frame_dt_ms = self.max_frame_dt_ms;
        self.prev = None;
    }
    fn set_mode(&mut self, mode: Mode) {
        if mode == self.rig.mode() {
            return;
        }
        self.slice_view = None;
        if mode == Mode::D3 {
            self.fit3d = true;
        }
        // Logarithmic axes are 2D only: the shared window changes coordinates with the mode.
        if !self.map.is_linear() {
            let w = self.world_window();
            let shown = if mode == Mode::D2 {
                let d = self.map.to_display(&w);
                if d.min.iter().chain(d.max.iter()).all(|v| v.is_finite()) {
                    d
                } else {
                    self.map.to_display(&self.doc.view.window)
                }
            } else {
                Window3::new(w.min, w.max)
            };
            self.rig.set_window(shown);
        }
        // The outgoing scene keeps drawing (fading) while the camera tweens to the new mode.
        self.prev = self.current.take();
        self.swap = None;
        if let Some(p) = &mut self.prev {
            // A flat scene leaving for (or arriving from) 3D fades its grid and its curves on
            // different schedules.
            if p.mode != Mode::D3 && mode == Mode::D3 && !self.reduced_motion && !self.legacy_transition {
                p.parts = Some(Box::new(split_flat(&p.geometry)));
            }
        }
        // The tween's clock starts at the next frame (see `Tween::start_ms`), so the time spent
        // building below is not taken out of the animation.
        self.rig.reduced_motion = self.reduced_motion;
        self.rig.set_mode(mode, self.now_ms);
        self.dirty = true;
        // A 3D scene starts as a cheap first look so the gap between the command and the first
        // moving frame stays short (also without a tween: nothing blocks); its surfaces are then
        // refined in time slices (see `refine_step`).
        self.rebuild_with(mode == Mode::D3);
        self.redraw = true;
    }

    fn pointer(&mut self, phase: &str, x: f64, y: f64, button: u8, shift: bool, vp: (f64, f64)) {
        match phase {
            // The slice inset is a read-only panel: a press on it does not orbit/pan the scene.
            "down" if self.in_inset(x, y) => {
                self.drag = None;
                let double = self.last_inset_press.is_some_and(|(t, px, py)| {
                    self.now_ms - t <= DOUBLE_CLICK_MS && (x - px).hypot(y - py) <= DOUBLE_CLICK_PX
                });
                if double {
                    self.last_inset_press = None;
                    self.inset_drag = None;
                    self.reset_inset_view();
                } else {
                    self.last_inset_press = Some((self.now_ms, x, y));
                    self.inset_drag = Some((x, y));
                }
            }
            "dblclick" if self.in_inset(x, y) => {
                self.inset_drag = None;
                self.reset_inset_view();
            }
            "move" if self.inset_drag.is_some() => {
                let Some((lx, ly)) = self.inset_drag else {
                    return;
                };
                self.inset_drag = Some((x, y));
                if let (Some(r), Some(base)) = (self.inset_rect(), self.current_inset_view()) {
                    let v = math_core::slice::pan_view(
                        base.view,
                        (x - lx) / r[2] as f64,
                        -(y - ly) / r[3] as f64,
                    );
                    self.set_inset_view(ViewReq {
                        free: base.free,
                        view: v,
                    });
                    self.touch_input();
                }
            }
            "down" => {
                self.fit3d = false;
                self.inset_drag = None;
                self.last_hover = None;
                let point = if button == 0 && !shift {
                    self.grab_point(x, y, vp)
                } else {
                    None
                };
                let curve = if button == 0 && !shift && point.is_none() {
                    self.grab_curve(x, y, vp)
                } else {
                    None
                };
                self.drag = Some(Drag {
                    button,
                    shift,
                    last: (x, y),
                    point,
                    curve,
                });
            }
            "up" | "cancel" => {
                self.end_curve_drag(phase == "cancel", x, y);
                self.drag = None;
                self.inset_drag = None;
            }
            "move" => {
                let m = self.active_map();
                if self.drag.is_none() {
                    self.hover(x, y, vp);
                    return;
                }
                let Some(d) = self.drag.as_mut() else { return };
                let (dx, dy) = (x - d.last.0, y - d.last.1);
                d.last = (x, y);
                if d.curve.is_some() {
                    self.drag_curve(x, y, vp);
                    self.touch_input();
                    return;
                }
                if let Some(g) = &d.point {
                    let here = self.rig.pixel_to_world((x, y), vp);
                    let to = m.inv3([here[0] + g.offset[0], here[1] + g.offset[1], 0.0]);
                    let (handle, to) = (g.handle.clone(), [to[0], to[1]]);
                    self.drag_point(&handle, to);
                    self.touch_input();
                    return;
                }
                let (button, shift) = (d.button, d.shift);
                let orbit = self.rig.mode() == Mode::D3 && button == 0 && !shift;
                if orbit {
                    self.rig.orbit(-dx * 0.008, dy * 0.008);
                } else if !(self.doc.view.lock && self.rig.mode() != Mode::D3) {
                    self.rig.pan_pixels(dx, dy, vp);
                    self.dirty = true;
                }
                self.touch_input();
            }
            _ => {}
        }
    }

    /// Advances time. Returns true if the caller should redraw.
    pub fn frame(&mut self, now_ms: f64) -> bool {
        self.now_ms = now_ms;
        self.tick(now_ms);
        let was_animating = self.rig.is_animating();
        self.rig.update(now_ms);
        let animating = self.rig.is_animating();
        if was_animating && !animating {
            self.prev = None;
            if let Some(c) = &mut self.current {
                c.parts = None;
            }
            self.redraw = true;
        }
        if self.dirty {
            let idle = now_ms - self.last_input_ms >= IDLE_REBUILD_MS;
            // `last_build_ms` is not known for a scene only built as a preview yet: wait for a
            // pause in the input (an orbit right after the switch) before the full build.
            let cheap = self.last_build_ms < CHEAP_BUILD_MS && !self.refine;
            if idle || cheap {
                // An expensive 3D scene (and anything mid-switch) is rebuilt in stages.
                let staged = animating || (!cheap && self.rig.mode() == Mode::D3);
                self.rebuild_with(staged);
            }
        } else if self.refine && self.legacy_transition {
            // Before: the full scene is rebuilt in one go on the still picture after the switch.
            if !animating && now_ms - self.last_input_ms >= IDLE_REBUILD_MS {
                self.rebuild_with(false);
            }
        } else if self.refine {
            let recent = now_ms - self.last_input_ms < IDLE_REBUILD_MS;
            let budget = if recent {
                REFINE_BUDGET_BUSY_MS
            } else if animating {
                REFINE_BUDGET_MS
            } else {
                REFINE_BUDGET_IDLE_MS
            };
            self.refine_step(budget);
        }
        if self.swap.as_ref().is_some_and(|s| now_ms - s.start_ms >= SWAP_BLEND_MS) {
            self.swap = None;
            self.redraw = true;
        }
        if self.panel_dirty && !self.dirty {
            let idle = now_ms - self.last_input_ms >= IDLE_REBUILD_MS;
            if idle || self.last_panel_ms < CHEAP_BUILD_MS {
                self.rebuild_panel(self.rig.mode());
                self.redraw = true;
            }
        }
        let redraw = self.redraw || animating || self.dirty || self.panel_dirty || self.swap.is_some();
        self.redraw = false;
        redraw
    }

    /// True while the app has work that needs more frames even though nothing may need redrawing
    /// yet (a 3D scene being refined): the shell must keep calling [`App::frame`].
    pub fn busy(&self) -> bool {
        self.refine || self.swap.is_some() || self.dirty || self.panel_dirty || self.rig.is_animating()
    }

    /// Finishes everything pending at full quality, right now (a PNG export must not capture a
    /// first-look mesh): rebuilds a dirty scene in one go, or runs the refinement to the end
    /// without a time limit, and drops any blend.
    pub fn settle(&mut self) {
        self.swap = None;
        if self.dirty {
            self.rebuild_with(false);
        }
        let mut guard = 0;
        while self.refine && guard < 512 {
            self.refine_step(f64::INFINITY);
            self.swap = None;
            guard += 1;
        }
        self.redraw = true;
    }

    /// One time slice of the staged refinement: meshes the current scene's surfaces at
    /// `self.refine_cap` until `budget_ms` is used up. A finished stage replaces the scene
    /// (blending in over the old one) and starts the next; an unfinished one changes nothing on
    /// screen, the finished tiles are remembered for the next frame.
    fn refine_step(&mut self, budget_ms: f64) {
        let Some(cur) = &self.current else {
            self.refine = false;
            return;
        };
        let (mode, origin) = (cur.mode, cur.origin);
        let t0 = instant::Instant::now();
        let window = self.scene_window();
        let full = full_surface_depth((self.size.0 as f64, self.size.1 as f64));
        let cap = self.refine_cap.min(full);
        crate::scene::set_render_scale(self.render_scale);
        let r = build_scene_progressive_mapped(
            &self.doc,
            self.map,
            mode,
            window,
            origin,
            self.size,
            &self.theme,
            cap,
            t0 + std::time::Duration::from_secs_f64(if budget_ms.is_finite() { budget_ms.max(0.0) / 1000.0 } else { 3600.0 }),
            // Unsliceable work waits for the still picture after the tween.
            self.rig.is_animating() && budget_ms.is_finite(),
        );
        crate::scene::set_render_scale(1.0);
        self.refine_ms += t0.elapsed().as_secs_f64() * 1000.0;
        if !r.done {
            return;
        }
        let parts = None;
        let old = self.current.replace(Built { geometry: r.geometry, origin, mode, parts });
        self.redraw = true;
        if !self.reduced_motion && !self.legacy_transition {
            if let Some(old) = old {
                self.swap = Some(Swap { old, start_ms: self.now_ms });
            }
        }
        match next_refine_depth(cap, full).filter(|_| r.capped) {
            Some(next) => self.refine_cap = next,
            None => {
                self.refine = false;
                // What a full build costs, so live edits know whether to rebuild staged.
                self.last_build_ms = self.refine_ms;
                clear_surface_tiles();
            }
        }
    }

    /// The window the scene should cover. In 2D the camera shows the window's x range exactly and
    /// derives the y range from the canvas aspect, so a tall phone canvas sees more than the
    /// stored y range; build the grid and curves over what is actually visible.
    fn scene_window(&self) -> Window3 {
        let mut w = self.rig.window();
        if self.rig.mode() == Mode::D2 {
            let aspect = self.size.0 as f64 / self.size.1.max(1) as f64;
            let cy = 0.5 * (w.min[1] + w.max[1]);
            let half_y = 0.5 * (w.max[0] - w.min[0]) / aspect.max(1e-6);
            if half_y.is_finite() && half_y > 0.0 {
                w.min[1] = cy - half_y;
                w.max[1] = cy + half_y;
            }
        }
        w
    }

    fn rebuild(&mut self) {
        self.rebuild_with(false);
    }

    /// Rebuilds the scene; `preview` meshes surfaces coarser (see [`build_scene_preview`]) and
    /// leaves `refine` set when that made a difference.
    fn rebuild_with(&mut self, preview: bool) {
        self.build_rev += 1;
        let t0 = instant::Instant::now();
        let mode = self.rig.mode();
        let origin = self.rig.render_origin();
        let window = self.scene_window();
        // The scale only lasts for this build, so direct `build_scene` callers on the thread
        // always see 1.
        struct ResetScale;
        impl Drop for ResetScale {
            fn drop(&mut self) {
                crate::scene::set_render_scale(1.0);
            }
        }
        crate::scene::set_render_scale(self.render_scale);
        let _reset_scale = ResetScale;
        let geometry = if preview {
            let full = full_surface_depth((self.size.0 as f64, self.size.1 as f64));
            let first = if self.legacy_transition { crate::scene::PREVIEW_SURFACE_DEPTH } else { FIRST_PREVIEW_DEPTH }.min(full);
            let (g, coarse) = build_scene_capped_mapped(
                &self.doc, self.map, mode, window, origin, self.size, &self.theme, first,
            );
            self.refine = coarse;
            self.refine_cap = next_refine_depth(first, full).unwrap_or(full);
            self.refine_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if !coarse {
                // Nothing was coarser than a full build, so this IS the full cost.
                self.last_build_ms = self.refine_ms;
            }
            g
        } else {
            self.refine = false;
            let g = build_scene_mapped(
                &self.doc, self.map, mode, window, origin, self.size, &self.theme,
            );
            // Only full builds tell how expensive live rebuilds of this scene are.
            self.last_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
            g
        };

        let diags: Vec<DiagItem> = geometry
            .diagnostics
            .iter()
            .map(|(id, m)| DiagItem {
                id: id.clone(),
                message: m.clone(),
            })
            .collect();
        let mut diags = diags;
        for it in self.doc.items.iter().filter(|i| {
            (i.kind == ItemKind::Action || actions::looks_like_action(&i.latex))
                && !i.latex.trim().is_empty()
        }) {
            let res =
                actions::parse_item(&self.doc, &it.id).and_then(|a| actions::plan(&self.doc, &a));
            if let (Err(message), false) = (res, it.hidden) {
                diags.push(DiagItem {
                    id: it.id.clone(),
                    message,
                });
            }
        }
        if diags != self.diagnostics {
            self.diagnostics = diags.clone();
            self.outbox.push(Event::Diagnostics { items: diags });
        }
        let colors: Vec<ItemColor> = geometry
            .item_colors
            .iter()
            .map(|(id, c)| {
                let h = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                ItemColor {
                    id: id.clone(),
                    color: format!("#{:02x}{:02x}{:02x}", h(c[0]), h(c[1]), h(c[2])),
                }
            })
            .collect();
        if colors != self.colors {
            self.colors = colors.clone();
            self.outbox.push(Event::Colors { items: colors });
        }
        if geometry.infos != self.infos {
            self.infos = geometry.infos.clone();
            self.outbox.push(Event::Info {
                items: geometry.infos.clone(),
            });
        }
        // World coordinates; with logarithmic axes, the visible window (the y range is not
        // derived from the aspect by the shell then).
        let w = if self.active_map().is_linear() {
            self.rig.window()
        } else {
            let v = self.active_map().to_world(self.scene_window());
            Window3::new(v.min, v.max)
        };
        self.outbox.push(Event::View {
            mode: mode_name(mode).into(),
            min: w.min,
            max: w.max,
            grid: self.doc.view.grid,
            axes: self.doc.view.axes,
            axis_numbers: self.doc.view.axis_numbers,
            minor_grid: self.doc.view.minor_grid,
            arrows: self.doc.view.arrows,
            lock: self.doc.view.lock,
            x_label: self.doc.view.x_label.clone(),
            y_label: self.doc.view.y_label.clone(),
            x_step: self.doc.view.x_step,
            y_step: self.doc.view.y_step,
            grid_kind: self.doc.view.grid_kind,
            x_scale: self.doc.view.x_scale,
            y_scale: self.doc.view.y_scale,
            weight: self.doc.view.weight,
            text_scale: self.doc.view.text_scale,
        });
        self.refresh_analysis();
        self.outbox.push(Event::Labels {
            labels: geometry
                .labels
                .iter()
                .map(|l| LabelOut {
                    pos: l.pos,
                    text: l.text.clone(),
                    axis: l.axis,
                })
                .collect(),
        });
        self.rebuild_panel(mode);
        // A flat scene arriving from 3D fades its grid and curves on separate schedules.
        let parts = (mode != Mode::D3
            && !self.legacy_transition
            && self.rig.is_animating()
            && self.prev.as_ref().is_some_and(|p| p.mode == Mode::D3))
        .then(|| Box::new(split_flat(&geometry)));
        self.swap = None;
        self.current = Some(Built {
            geometry,
            origin,
            mode,
            parts,
        });
        self.dirty = false;
        self.redraw = true;
        if mode == Mode::D3 && self.fit3d && self.map.is_linear() {
            if let Some(b) = self.current.as_ref().and_then(|b| content_extent(&b.geometry, b.origin)) {
                self.fit3d = false;
                if let Some(w) = fit_default_cube(self.rig.window(), b) {
                    self.low_z_fit = w.min[2] == -1.0;
                    self.rig.set_window(w);
                    self.rebuild_with(preview);
                }
            }
        } else if mode == Mode::D3 && self.low_z_fit {
            let w = self.rig.window();
            let low = (w.min, w.max) == ([-5.0, -5.0, -1.0], [5.0, 5.0, 8.0]);
            self.low_z_fit = low;
            if low && self.current.as_ref().and_then(|b| content_extent(&b.geometry, b.origin)).is_some_and(|(_, z)| z < -1.0 + 1e-3) {
                self.low_z_fit = false;
                self.rig.set_window(Window3::new([-5.0; 3], [5.0; 3]));
                self.rebuild_with(preview);
            }
        }
    }

    /// Rebuilds the slice inset and reports the slice state when it changed.
    fn rebuild_panel(&mut self, mode: Mode) {
        let t0 = instant::Instant::now();
        self.panel_dirty = false;
        let window = self.scene_window();
        let res = if self.doc.slice.is_some() && !self.active_map().is_linear() {
            Some(Err(format!(
                "a slice is {}",
                crate::scene::axes_unsupported(&self.active_map())
            )))
        } else {
            build_slice_panel_view(
                &self.doc,
                mode,
                window,
                self.size,
                &self.theme,
                self.slice_view.as_ref(),
            )
        };
        let ev = match res {
            None => {
                self.panel = None;
                Event::Slice {
                    active: false,
                    dim: 0,
                    fixed: BTreeMap::new(),
                    free: Vec::new(),
                    rect: None,
                    curves: 0,
                    points: 0,
                    error: None,
                    view: None,
                    follow: true,
                }
            }
            Some(Err(message)) => {
                self.panel = None;
                // Dimension it would have in this mode (0 when it leaves nothing to show).
                let dim = self
                    .doc
                    .slice
                    .as_ref()
                    .map(|c| (mode.dims() as usize).saturating_sub(c.fixed.len()).min(2) as u8)
                    .unwrap_or(0);
                Event::Slice {
                    active: false,
                    dim,
                    fixed: BTreeMap::new(),
                    free: Vec::new(),
                    rect: None,
                    curves: 0,
                    points: 0,
                    error: Some(message),
                    view: None,
                    follow: true,
                }
            }
            Some(Ok(out)) => {
                let fixed = out
                    .rs
                    .fixed_axes()
                    .into_iter()
                    .map(|a| {
                        (
                            math_core::slice::axis_name(a).to_string(),
                            out.rs.fixed[a].unwrap_or(0.0),
                        )
                    })
                    .collect();
                let free = out
                    .rs
                    .free_axes()
                    .into_iter()
                    .map(|a| math_core::slice::axis_name(a).to_string())
                    .collect();
                let ev = Event::Slice {
                    active: true,
                    dim: out.rs.dim,
                    fixed,
                    free,
                    rect: Some(out.panel.rect),
                    curves: out.panel.curves,
                    points: out.panel.points,
                    error: None,
                    view: Some(out.panel.view),
                    follow: out.panel.follow,
                };
                // A request for other free axes no longer applies.
                if out.panel.follow {
                    self.slice_view = None;
                } else if let Some(sv) = &mut self.slice_view {
                    // Keep the request in its normalised (aspect-fitted) form.
                    sv.view = out.panel.view;
                }
                self.panel = Some(out.panel);
                ev
            }
        };
        if self.panel.is_none() {
            self.slice_view = None;
        }
        self.last_panel_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if self.last_slice.as_ref() != Some(&ev) {
            // Do not announce "no slice" at startup, only when a slice goes away.
            let quiet = matches!(
                &ev,
                Event::Slice {
                    active: false,
                    error: None,
                    ..
                }
            ) && self.last_slice.is_none();
            self.last_slice = Some(ev.clone());
            if !quiet {
                // An undelivered slice report (raised in frame(), not yet drained) describes an
                // older config/mode; the new one supersedes it so a late event is never stale.
                self.outbox.retain(|e| !matches!(e, Event::Slice { .. }));
                self.outbox.push(ev);
            }
        }
    }

    /// The slice inset to draw in a corner viewport after the main layers (see
    /// [`crate::render::Renderer::render_with_inset`]), or `None` when there is no slice, it does
    /// not fit the mode, or the camera is mid mode-switch.
    pub fn inset(&self) -> Option<Inset<'_>> {
        let p = self.panel.as_ref().filter(|_| !self.rig.is_animating() && !self.inset_hidden)?;
        Some(Inset {
            rect: p.rect,
            rig: &p.rig,
            layers: vec![Layer {
                geometry: &p.geometry,
                fade: 1.0,
                origin: p.origin,
                lift: 1.0,
            }],
        })
    }

    /// The opacities of the two scenes of a running mode switch (outgoing, incoming).
    fn switch_fades(&self) -> Option<ModeFades> {
        match (&self.prev, &self.current) {
            (Some(p), Some(c)) if self.rig.is_animating() => {
                let progress = self.rig.progress() as f32;
                Some(if self.legacy_transition {
                    let (o, i) = crate::render::crossfade(progress);
                    ModeFades { from_backdrop: o, from_items: o, to_backdrop: i, to_items: i }
                } else {
                    mode_fades(p.mode, c.mode, progress, self.rig.lift() as f32)
                })
            }
            _ => None,
        }
    }

    /// Appends the layers of one scene: `back` is the opacity of its grid and axes, `items` of
    /// the rest. Layers that are invisible anyway are left out (they would only cost draw time).
    fn push_built<'a>(
        &'a self,
        out: &mut Vec<Layer<'a>>,
        b: &'a Built,
        back: f32,
        items: f32,
        swap: Option<&'a Swap>,
    ) {
        const MIN_FADE: f32 = 0.003;
        let lift = layer_lift(b.mode, self.rig.lift());
        let mut push = |g: &'a SceneGeometry, origin: [f64; 3], fade: f32| {
            if fade > MIN_FADE {
                out.push(Layer { geometry: g, fade: fade.min(1.0), origin, lift });
            }
        };
        if let Some(parts) = b.parts.as_deref().filter(|_| self.rig.is_animating()) {
            push(&parts.0, b.origin, back);
            push(&parts.1, b.origin, items);
            return;
        }
        match swap {
            // The coarser mesh stays opaque underneath while the refined one blends in on top.
            Some(s) => {
                push(&s.old.geometry, s.old.origin, items);
                push(&b.geometry, b.origin, items * swap_blend(self.now_ms - s.start_ms));
            }
            None => push(&b.geometry, b.origin, items),
        }
    }

    /// Layers to draw this frame, oldest first.
    pub fn layers(&self) -> Vec<Layer<'_>> {
        let mut out = Vec::new();
        match (&self.prev, &self.current, self.switch_fades()) {
            (Some(p), Some(c), Some(f)) => {
                self.push_built(&mut out, p, f.from_backdrop, f.from_items, None);
                self.push_built(&mut out, c, f.to_backdrop, f.to_items, self.swap.as_ref());
            }
            (_, Some(c), _) => self.push_built(&mut out, c, 1.0, 1.0, self.swap.as_ref()),
            _ => {}
        }
        out
    }

    /// Tick/axis labels projected through the CURRENT camera (so they follow a mode-switch
    /// tween), in CSS pixels for a viewport of `size`. `visible` is false when the label falls
    /// outside the view or behind the camera. Cheap enough to call every frame.
    pub fn screen_labels(&self) -> Vec<ScreenLabel> {
        let Some(c) = &self.current else {
            return Vec::new();
        };
        let mut out = Vec::new();
        // Mid-switch both scenes' labels are shown, each at the opacity of its own grid / items,
        // so the old tick labels fade out as the new ones fade in.
        match (&self.prev, self.switch_fades()) {
            (Some(p), Some(f)) => {
                self.project_labels(p, f.from_backdrop, f.from_items, &mut out);
                self.project_labels(c, f.to_backdrop, f.to_items, &mut out);
            }
            _ => self.project_labels(c, 1.0, 1.0, &mut out),
        }
        out.extend(self.inset_screen_labels());
        out
    }

    /// Projects `b`'s labels through the current camera into `out`; tick labels and axis names
    /// get opacity `back`, an item's own labels `items`.
    fn project_labels(&self, b: &Built, back: f32, items: f32, out: &mut Vec<ScreenLabel>) {
        let aspect = self.size.0 as f64 / self.size.1.max(1) as f64;
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let rect = self.inset_rect();
        // Labels of a 3D scene ride the switch lift with its geometry.
        let lift = layer_lift(b.mode, self.rig.lift()) as f64;
        // 3D tick numbers that would sit on a plotted surface are dropped, not haloed.
        let cover = (b.mode == Mode::D3).then(|| self.surface_cover(b, aspect, (w, h), lift)).flatten();
        out.extend(b.geometry.labels.iter().filter_map(|l| {
            let mut alpha = if l.axis == 4 { items } else { back } as f64;
            if alpha <= 0.003 {
                return None;
            }
            let ndc = self
                .rig
                .project_ndc([l.pos[0], l.pos[1], l.pos[2] * lift], aspect);
            let (x, y) = ((ndc[0] * 0.5 + 0.5) * w, (1.0 - (ndc[1] * 0.5 + 0.5)) * h);
            let mut visible =
                ndc[0].abs() <= 1.0 && ndc[1].abs() <= 1.0 && (0.0..=1.0).contains(&ndc[2]);
            if let Some(c) = &cover {
                if l.axis <= 2 {
                    if c.covers(l.axis, &l.text, x, y) {
                        visible = false;
                    } else if self.doc.slice.is_some() && c.near(x, y) {
                        alpha *= 0.35;
                    }
                }
            }
            // Labels under the inset would show through it.
            if let Some(r) = rect {
                if x >= r[0] as f64
                    && x <= (r[0] + r[2]) as f64
                    && y >= r[1] as f64
                    && y <= (r[1] + r[3]) as f64
                {
                    visible = false;
                }
            }
            Some(ScreenLabel {
                text: l.text.clone(),
                axis: l.axis,
                x,
                y,
                visible,
                alpha,
                inset: false,
                clip: None,
                item: l.item.clone(),
                offset: (l.offset != [0.0; 2]).then_some(l.offset),
                size: (l.size != 1.0).then_some(l.size),
            })
        }));
    }

    /// The screen-space triangles of `b`'s lit surfaces (canvas pixels, y down), or `None` when
    /// there are none.
    fn surface_cover(&self, b: &Built, aspect: f64, (w, h): (f64, f64), lift: f64) -> Option<SurfaceCover> {
        let g = &b.geometry;
        if g.indices.len() < 3 || !g.labels.iter().any(|l| l.axis <= 2) {
            return None;
        }
        let m = self.rig.view_proj(aspect);
        let o = self.rig.render_origin();
        let d = [b.origin[0] - o[0], b.origin[1] - o[1], b.origin[2] - o[2]];
        let pts: Vec<[f64; 2]> = g
            .vertices
            .iter()
            .map(|v| {
                let q = [
                    v.pos[0] as f64 + d[0],
                    v.pos[1] as f64 + d[1],
                    (v.pos[2] as f64 + b.origin[2]) * lift - o[2],
                ];
                let c = |r: usize| m[0][r] * q[0] + m[1][r] * q[1] + m[2][r] * q[2] + m[3][r];
                let cw = c(3);
                if !(cw.abs() > 1e-12 && cw.is_finite()) {
                    return [f64::NAN; 2];
                }
                [(c(0) / cw * 0.5 + 0.5) * w, (1.0 - (c(1) / cw * 0.5 + 0.5)) * h]
            })
            .collect();
        let tris = g
            .indices
            .chunks_exact(3)
            .filter_map(|t| {
                let tri = [pts.get(t[0] as usize)?, pts.get(t[1] as usize)?, pts.get(t[2] as usize)?];
                if tri.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
                    return None;
                }
                Some([*tri[0], *tri[1], *tri[2]])
            })
            .collect();
        let tris: Vec<[[f64; 2]; 3]> = tris;
        let mut bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for p in tris.iter().flatten() {
            bounds = [bounds[0].min(p[0]), bounds[1].min(p[1]), bounds[2].max(p[0]), bounds[3].max(p[1])];
        }
        Some(SurfaceCover { tris, bounds })
    }

    /// Tick labels and axis names of the slice inset, in canvas pixels (empty without one).
    pub fn inset_screen_labels(&self) -> Vec<ScreenLabel> {
        let Some(p) = self.panel.as_ref().filter(|_| !self.rig.is_animating() && !self.inset_hidden) else {
            return Vec::new();
        };
        let r = p.rect;
        let aspect = r[2] as f64 / r[3].max(1) as f64;
        let (pw, ph) = (r[2] as f64, r[3] as f64);
        let ts = self.doc.view.text_scale;
        p.geometry
            .labels
            .iter()
            .map(|l| {
                let ndc = p.rig.project_ndc(l.pos, aspect);
                // Position inside the panel, from its top-left corner; the whole text box (not
                // just the anchor) must lie inside the rectangle or the label is hidden.
                let (lx, ly) = ((ndc[0] * 0.5 + 0.5) * pw, (1.0 - (ndc[1] * 0.5 + 0.5)) * ph);
                let visible = ndc[0].abs() <= 1.0
                    && ndc[1].abs() <= 1.0
                    && label_box_inside_s(l.axis, &l.text, lx, ly, pw, ph, ts);
                ScreenLabel {
                    text: l.text.clone(),
                    axis: l.axis,
                    x: r[0] as f64 + lx,
                    y: r[1] as f64 + ly,
                    visible,
                    alpha: 1.0,
                    inset: true,
                    clip: Some([r[0] as f64, r[1] as f64, pw, ph]),
                    item: None,
                    offset: None,
                    size: None,
                }
            })
            .collect()
    }

    /// Mode of the scene currently being drawn as the main layer.
    pub fn scene_mode(&self) -> Option<Mode> {
        self.current.as_ref().map(|c| c.mode)
    }
}

/// `Some(None)` for an explicit JSON `null` and `Some(Some(v))` for a value, so a command can tell
/// "clear this" from "leave it" (`#[serde(default)]` gives `None` when the field is absent).
fn opt_opt<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

#[cfg(test)]
mod tests {
    #[test]
    fn surface_cover_hides_ticks_inside_a_triangle_only() {
        let c = SurfaceCover { tris: vec![[[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]], bounds: [0.0, 0.0, 100.0, 100.0] };
        assert!(c.covers(2, "5", 20.0, 20.0));
        assert!(!c.covers(2, "5", 90.0, 90.0));
        // An x tick's text hangs below its anchor: the box centre, not the anchor, decides.
        assert!(c.covers(0, "5", 20.0, 10.0));
    }

    use super::*;

    fn app() -> App {
        let mut a = App::new((800, 600));
        // Tests fast-forward the clock; the frame clamp has its own tests.
        a.max_frame_dt_ms = f64::INFINITY;
        a.rig.max_frame_dt_ms = f64::INFINITY;
        a
    }

    fn cmd(app: &mut App, json: &str) -> Vec<serde_json::Value> {
        serde_json::from_str(&app.dispatch(json)).unwrap()
    }

    fn has(events: &[serde_json::Value], t: &str) -> bool {
        events.iter().any(|e| e["t"] == t)
    }

    #[test]
    fn empty_app_draws_grid_only() {
        let a = app();
        let layers = a.layers();
        assert_eq!(layers.len(), 1);
        assert!(!layers[0].geometry.segments.is_empty());
        assert!(layers[0].geometry.vertices.is_empty());
    }

    #[test]
    fn set_expr_adds_item_and_draws() {
        let mut a = app();
        let before = a.layers()[0].geometry.segments.len();
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        assert!(has(&ev, "view") && has(&ev, "labels"));
        assert!(a.layers()[0].geometry.segments.len() > before);
        assert_eq!(a.doc.items.len(), 1);
        // Editing the same id replaces, not duplicates.
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=2x"}"#);
        assert_eq!(a.doc.items.len(), 1);
        assert_eq!(a.doc.items[0].latex, "y=2x");
    }

    #[test]
    fn bad_expressions_report_diagnostics_not_panics() {
        let mut a = app();
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y="}"#);
        let diag = ev
            .iter()
            .find(|e| e["t"] == "diagnostics")
            .expect("diagnostics event");
        assert_eq!(diag["items"][0]["id"], "a");
        // Fixing it clears the diagnostics.
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x"}"#);
        let diag = ev.iter().find(|e| e["t"] == "diagnostics").unwrap();
        assert!(diag["items"].as_array().unwrap().is_empty());
    }

    #[test]
    fn malformed_commands_are_errors() {
        let mut a = app();
        assert!(has(&cmd(&mut a, "not json"), "error"));
        assert!(has(&cmd(&mut a, r#"{"t":"nope"}"#), "error"));
        assert!(has(&cmd(&mut a, r#"{"t":"setMode","mode":"9d"}"#), "error"));
        assert!(has(
            &cmd(&mut a, r#"{"t":"removeItem","id":"missing"}"#),
            "error"
        ));
    }

    #[test]
    fn mode_switch_tweens_with_two_layers_then_settles() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(a.rig.is_animating());
        assert_eq!(a.layers().len(), 2, "outgoing + incoming while tweening");
        assert!(a.frame(250.0));
        assert_eq!(a.layers().len(), 2);
        a.frame(10_000.0);
        assert!(!a.rig.is_animating());
        assert_eq!(a.layers().len(), 1);
        assert_eq!(a.scene_mode(), Some(Mode::D3));
        // The 3D scene reinterprets y=x^2 as a surface.
        assert!(!a.layers()[0].geometry.vertices.is_empty());
    }

    #[test]
    fn switching_mid_tween_keeps_working() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=25"}"#,
        );
        for (i, m) in ["3d", "2d", "1d", "3d", "2d"].iter().enumerate() {
            cmd(&mut a, &format!(r#"{{"t":"setMode","mode":"{m}"}}"#));
            a.frame(i as f64 * 50.0);
        }
        a.frame(10_000.0);
        assert_eq!(a.mode(), Mode::D2);
        assert_eq!(a.layers().len(), 1);
    }

    #[test]
    fn switch_after_a_long_idle_still_animates_from_the_first_frame() {
        // Native and viSHual stop calling `frame` while idle, so the command sees an old clock.
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(100.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(a.frame(60_000.0));
        assert!(a.rig.is_animating(), "the whole tween is still ahead");
        assert_eq!(a.rig.progress(), 0.0);
        // The first frame still shows the 2D scene as it was: grid and curves, no 3D yet.
        let l = a.layers();
        assert_eq!(l.len(), 2, "grid + curves of the 2D scene");
        assert!(l.iter().all(|x| x.fade == 1.0 && x.lift == 1.0));
        assert_eq!(a.current.as_ref().unwrap().mode, Mode::D3);
        drop(l);
        let mut t = 60_000.0;
        let mut frames = 0;
        while a.rig.is_animating() {
            t += 1000.0 / 60.0;
            a.frame(t);
            frames += 1;
        }
        assert!(frames >= 29, "{frames} frames");
    }

    /// Frames (16 ms apart from `t`) until the staged refinement has finished and its blend is
    /// over; the meshing slices are bounded in real time, so a debug build needs many.
    fn run_until_refined(a: &mut App, mut t: f64) -> f64 {
        let mut n = 0;
        while a.busy() {
            t += 16.0;
            a.frame(t);
            n += 1;
            assert!(n < 20_000, "refinement never finished");
        }
        t
    }

    #[test]
    fn switch_to_3d_previews_then_refines_at_full_quality() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"\\sin(x)\\cos(y)+\\sin(y)\\cos(z)+\\sin(z)\\cos(x)=0"}"#,
        );
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        let full = crate::scene::build_scene(
            &a.doc,
            Mode::D3,
            a.scene_window(),
            a.rig.render_origin(),
            a.size,
            &a.theme,
        );
        let preview = a.current.as_ref().unwrap().geometry.vertices.len();
        assert!(a.refine);
        assert!(
            preview > 0 && preview < full.vertices.len() / 4,
            "{preview} vs {}",
            full.vertices.len()
        );
        // Mid-switch: the 3D scene is partly lifted (and fading in), the outgoing 2D one is not.
        a.frame(16.0);
        a.frame(266.0);
        let l = a.layers();
        let lifted: Vec<_> = l.iter().filter(|x| x.lift < 1.0).collect();
        assert!(!lifted.is_empty() && lifted.iter().all(|x| x.lift > 0.0));
        assert!(l.iter().any(|x| x.lift == 1.0), "the flat scene is not lifted");
        drop(l);
        // The refinement runs in slices alongside the tween, not in one stall.
        assert!(a.refine, "no full rebuild in one frame");
        assert!(a.busy());
        run_until_refined(&mut a, 266.0);
        assert!(!a.rig.is_animating() && !a.refine && a.swap.is_none());
        let l = a.layers();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].lift, 1.0);
        assert_eq!(l[0].fade, 1.0);
        assert_eq!(l[0].geometry.indices.len(), full.indices.len());
        // Leaving 3D needs no preview (2D builds are cheap) and nothing to refine.
        drop(l);
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        assert!(!a.refine);
        assert!(!a.busy() || a.rig.is_animating());
    }

    #[test]
    fn refinement_swaps_the_mesh_in_with_a_short_blend() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=9"}"#,
        );
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(16.0);
        let mut t = 16.0;
        let mut saw_blend = false;
        let mut n = 0;
        while a.busy() {
            t += 16.0;
            a.frame(t);
            n += 1;
            assert!(n < 20_000);
            if a.swap.is_some() && !a.rig.is_animating() {
                let l = a.layers();
                // The coarse mesh stays opaque underneath, the refined one fades in over it.
                assert!(l.len() <= 2 && !l.is_empty());
                assert_eq!(l[0].fade, 1.0);
                assert!(l.len() == 1 || l[1].fade <= 1.0);
                saw_blend = true;
            }
        }
        assert!(saw_blend, "the refined mesh popped in");
        assert_eq!(a.layers().len(), 1);
    }

    #[test]
    fn settle_finishes_the_refinement_at_full_quality_at_once() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=9"}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(16.0);
        assert!(a.refine);
        a.settle();
        assert!(!a.refine && a.swap.is_none());
        let full = crate::scene::build_scene(&a.doc, Mode::D3, a.scene_window(), a.rig.render_origin(), a.size, &a.theme);
        assert_eq!(a.current.as_ref().unwrap().geometry.indices.len(), full.indices.len());
    }

    #[test]
    fn legacy_transition_hook_restores_the_old_switch() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=9"}"#);
        cmd(&mut a, r#"{"t":"setLegacyTransition","on":true}"#);
        assert!(a.rig.max_frame_dt_ms.is_infinite());
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(a.prev.as_ref().unwrap().parts.is_none(), "no 2D split");
        a.frame(16.0);
        a.frame(266.0);
        assert_eq!(a.layers().len(), 2, "plain two-layer crossfade (one flat layer, not grid + items)");
        assert!(a.refine);
        a.frame(100.0);
        assert!(a.refine, "no refinement while moving");
        a.frame(900.0);
        a.frame(1100.0);
        assert!(!a.refine, "rebuilt in one go on the still picture");
        assert!(a.swap.is_none(), "no blend");
        cmd(&mut a, r#"{"t":"setLegacyTransition","on":false}"#);
        assert_eq!(a.rig.max_frame_dt_ms, MAX_FRAME_DT_MS);
    }

    #[test]
    fn a_slice_does_not_zoom_in_on_content_that_fills_the_box() {
        // a paraboloid fills its z = -1..8 box: the slice zoom would cut its top off
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"z=x^2+y^2"}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(0.0);
        run_until_refined(&mut a, 16.0);
        cmd(&mut a, r#"{"t":"setSlice","dim":2,"fixed":{"y":0}}"#);
        assert!(!a.slice_zoomed, "paraboloid zoomed");
        // a small sphere still gets the zoom
        let mut b = app();
        cmd(&mut b, r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=4"}"#);
        cmd(&mut b, r#"{"t":"setMode","mode":"3d"}"#);
        b.frame(0.0);
        run_until_refined(&mut b, 16.0);
        cmd(&mut b, r#"{"t":"setSlice","dim":2,"fixed":{"y":0}}"#);
        assert!(b.slice_zoomed, "sphere not zoomed");
    }

    #[test]
    fn default_cube_shrinks_only_for_small_content() {
        let d = Window3::new([-10.0; 3], [10.0; 3]);
        assert_eq!(fit_default_cube(d, (3.0, -3.0)), Some(Window3::new([-5.0; 3], [5.0; 3])));
        assert_eq!(fit_default_cube(d, (7.0, -7.0)), None);
        assert!(fit_default_cube(d, (10.0, -10.0)).is_some(), "content clipped by the box");
        assert_eq!(fit_default_cube(Window3::new([-4.0; 3], [4.0; 3]), (1.0, -1.0)), None);
        let bowl = fit_default_cube(d, (10.0, 0.0)).unwrap();
        assert_eq!((bowl.min, bowl.max), ([-5.0, -5.0, -1.0], [5.0, 5.0, 8.0]), "non-negative surface");
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=4"}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert_eq!(a.rig.window().max, [5.0; 3]);
    }

    #[test]
    fn a_resize_during_the_switch_does_not_block_on_the_full_build() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"x^2+y^2+z^2=9"}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(16.0);
        // The stage is laid out differently in 3D: the canvas is resized right after the switch.
        cmd(&mut a, r#"{"t":"resize","width":700,"height":520}"#);
        assert!(a.refine, "resize rebuilt at full quality in the command");
        let n = a.current.as_ref().unwrap().geometry.indices.len();
        let full_n = crate::scene::build_scene(&a.doc, Mode::D3, a.scene_window(), a.rig.render_origin(), a.size, &a.theme).indices.len();
        assert!(n < full_n / 2, "first look expected, got {n} of {full_n}");
        run_until_refined(&mut a, 16.0);
        assert!(!a.refine);
        let full = crate::scene::build_scene(&a.doc, Mode::D3, a.scene_window(), a.rig.render_origin(), a.size, &a.theme);
        assert_eq!(a.current.as_ref().unwrap().geometry.indices.len(), full.indices.len());
        // At rest and cheap, a resize still rebuilds at once.
        let mut b = app();
        cmd(&mut b, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        cmd(&mut b, r#"{"t":"resize","width":390,"height":844}"#);
        assert!(!b.refine && !b.layers().is_empty());
    }

    #[test]
    fn an_unsliceable_stage_waits_for_the_tween_to_end() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"z=\\sin(x)\\cos(y)"}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        let full = crate::scene::build_scene(&a.doc, Mode::D3, a.scene_window(), a.rig.render_origin(), a.size, &a.theme);
        a.frame(16.0);
        let mut t = 16.0;
        while a.rig.is_animating() {
            t += 16.0;
            a.frame(t);
            if a.rig.is_animating() {
                // The heavy full-depth height field is not meshed in the middle of the motion.
                let n = a.current.as_ref().unwrap().geometry.indices.len();
                assert!(n < full.indices.len() / 2, "full-depth mesh arrived mid-tween: {n}");
                assert!(a.refine, "refinement is still pending");
            }
        }
        run_until_refined(&mut a, t);
        assert!(!a.refine);
        assert_eq!(a.current.as_ref().unwrap().geometry.indices.len(), full.indices.len());
    }

    #[test]
    fn next_refine_depth_walks_the_stages() {
        assert_eq!(next_refine_depth(4, 7), Some(5));
        assert_eq!(next_refine_depth(3, 7), Some(5));
        assert_eq!(next_refine_depth(5, 7), Some(7));
        assert_eq!(next_refine_depth(7, 7), None);
        assert_eq!(next_refine_depth(4, 5), Some(5));
        assert_eq!(next_refine_depth(3, 4), Some(4));
        assert_eq!(next_refine_depth(8, 7), None);
    }

    #[test]
    fn swap_blend_is_a_smooth_ramp() {
        assert_eq!(swap_blend(0.0), 0.0);
        assert_eq!(swap_blend(SWAP_BLEND_MS), 1.0);
        assert_eq!(swap_blend(1e9), 1.0);
        assert_eq!(swap_blend(-5.0), 0.0);
        assert_eq!(swap_blend(f64::NAN), 1.0);
        assert!((swap_blend(SWAP_BLEND_MS / 2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn refining_never_blocks_a_frame_for_the_whole_mesh() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"\\sin(x)\\cos(y)+\\sin(y)\\cos(z)+\\sin(z)\\cos(x)=0"}"#,
        );
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        let mut frames = 0;
        let mut t = 0.0;
        while a.refine {
            t += 16.0;
            a.frame(t);
            frames += 1;
            assert!(frames < 20_000);
        }
        assert!(frames >= 4, "the full mesh took {frames} frames: it was not sliced");
    }

    #[test]
    fn reduced_motion_switches_at_once_without_a_stall() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"z=\\sin(x)\\cos(y)"}"#,
        );
        cmd(&mut a, r#"{"t":"setReducedMotion","on":true}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(!a.rig.is_animating(), "no tween");
        assert!(a.refine, "the surface is built in slices, not in the command");
        a.frame(16.0);
        let l = a.layers();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].lift, 1.0);
        assert_eq!(l[0].fade, 1.0);
        drop(l);
        let mut t = 16.0;
        while a.refine {
            t += 16.0;
            a.frame(t);
            assert!(a.swap.is_none(), "no blend under reduced motion");
            assert!(t < 400_000.0);
        }
        // The setting outlives a document load (which replaces the camera rig).
        let json = doc::to_json(&a.doc);
        cmd(
            &mut a,
            &serde_json::json!({"t":"loadDoc","json":json}).to_string(),
        );
        a.frame(t + 16.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        assert!(!a.rig.is_animating());
        cmd(&mut a, r#"{"t":"setReducedMotion","on":false}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(a.rig.is_animating());
    }

    #[test]
    fn a_stalled_frame_does_not_use_up_the_switch() {
        let mut a = app();
        a.max_frame_dt_ms = MAX_FRAME_DT_MS;
        a.rig.max_frame_dt_ms = MAX_FRAME_DT_MS;
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(100.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(116.0); // first frame: starts the clock
        assert_eq!(a.rig.progress(), 0.0);
        a.frame(116.0 + 1850.0); // the 1.8 s stall measured in the browser
        assert!(a.rig.is_animating());
        let p = a.rig.progress();
        assert!(p > 0.0 && p <= MAX_FRAME_DT_MS / a.rig.duration_ms + 1e-9, "{p}");
    }

    #[test]
    fn a_2d_curve_stays_drawn_while_it_lifts_and_the_grid_goes_first() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(16.0);
        let mut t = 16.0;
        let (mut grid_gone_at, mut curve_gone_at, mut three_seen_at) = (None, None, None);
        while a.rig.is_animating() {
            t += 16.0;
            a.frame(t);
            if !a.rig.is_animating() {
                break;
            }
            let lift = a.rig.lift();
            let l = a.layers();
            let flat: Vec<_> = l.iter().filter(|x| x.lift == 1.0 && !x.geometry.segments.is_empty()).collect();
            // layer 0 of the flat scene is the backdrop, layer 1 the items
            let has_back = flat.iter().any(|x| x.geometry.segments.len() == a.prev.as_ref().unwrap().parts.as_ref().unwrap().0.segments.len());
            let has_items = l.iter().any(|x| std::ptr::eq(x.geometry, &a.prev.as_ref().unwrap().parts.as_ref().unwrap().1));
            if !has_back && grid_gone_at.is_none() {
                grid_gone_at = Some(lift);
            }
            if !has_items && curve_gone_at.is_none() {
                curve_gone_at = Some(lift);
            }
            if l.iter().any(|x| x.lift < 1.0) && three_seen_at.is_none() {
                three_seen_at = Some(lift);
            }
        }
        let (g, c, th) = (grid_gone_at.unwrap(), curve_gone_at.unwrap(), three_seen_at.unwrap());
        assert!(g < th + 0.25, "grid should go before / as the 3D scene appears: {g} vs {th}");
        assert!(th >= crate::render::FADE_3D_START as f64 - 0.05, "3D appears only once lifted: {th}");
        assert!(c > 0.8, "the curve stays until the form has risen: {c}");
    }

    #[test]
    fn labels_cross_fade_with_the_scenes() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(0.0);
        let flat = a.screen_labels();
        assert!(!flat.is_empty() && flat.iter().all(|l| l.alpha == 1.0));
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(16.0);
        // Start: the 2D tick labels are still there, the 3D ones not yet.
        let l0 = a.screen_labels();
        assert_eq!(l0.len(), flat.len());
        assert!(l0.iter().all(|l| l.alpha == 1.0));
        // Middle: both sets, partly faded.
        a.frame(16.0 + 250.0);
        let lm = a.screen_labels();
        let (mut old, mut new) = (0, 0);
        for l in &lm {
            assert!(l.alpha > 0.0 && l.alpha <= 1.0);
        }
        for l in &lm {
            if l.axis == 2 { new += 1 } else { old += 1 }
        }
        assert!(old > 0, "2D labels vanished at once");
        let _ = new;
        // End: only the 3D scene's labels, fully opaque.
        run_until_refined(&mut a, 266.0);
        let l1 = a.screen_labels();
        assert!(!l1.is_empty() && l1.iter().all(|l| l.alpha == 1.0));
        assert!(l1.iter().any(|l| l.axis == 2), "z tick labels of the 3D box");
    }

    #[test]
    fn pan_moves_window_and_keeps_stale_origin_until_rebuild() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        a.frame(0.0);
        let w0 = a.rig.window();
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":300,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":300,"y":300}"#);
        let w1 = a.rig.window();
        assert!(w1.min[0] > w0.min[0], "dragging left pans the window right");
        // Past the idle delay the dirty scene rebuilds whatever the measured build time was
        // (`last_build_ms` is a real clock reading, so "cheap, rebuild at once" is not
        // deterministic under load) and the origin follows the window.
        a.frame(1.0 + IDLE_REBUILD_MS);
        let o = a.layers()[0].origin;
        let c = a.rig.render_origin();
        assert!((o[0] - c[0]).abs() < 1e-9);
    }

    #[test]
    fn wheel_zooms_about_cursor_in_2d() {
        let mut a = app();
        let w0 = a.rig.window();
        cmd(&mut a, r#"{"t":"wheel","x":400,"y":300,"dy":-300}"#);
        let w1 = a.rig.window();
        assert!(
            w1.max[0] - w1.min[0] < w0.max[0] - w0.min[0],
            "negative dy zooms in"
        );
    }

    #[test]
    fn locked_view_ignores_wheel_and_pan_but_not_set_view_window() {
        let mut a = app();
        let ev = cmd(&mut a, r#"{"t":"setView","lock":true}"#);
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        let w0 = a.rig.window();
        cmd(&mut a, r#"{"t":"wheel","x":400,"y":300,"dy":-300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":460,"y":340}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":460,"y":340}"#);
        assert_eq!(a.rig.window(), w0, "locked: window unchanged");
        cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-2,-2,-1],"max":[2,2,1]}}"#,
        );
        assert_eq!(a.doc.view.window.max[0], 2.0, "the window can still be set");
        cmd(&mut a, r#"{"t":"setView","lock":false}"#);
        let w1 = a.rig.window();
        cmd(&mut a, r#"{"t":"wheel","x":400,"y":300,"dy":-300}"#);
        assert!(a.rig.window() != w1, "unlocked zooms again");
    }

    #[test]
    fn set_view_axis_names_steps_minor_grid_and_arrows() {
        let mut a = app();
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","minorGrid":false,"arrows":true,"xLabel":"time","yLabel":"  h ","xStep":2,"yStep":5}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        let v = &a.doc.view;
        assert!(!v.minor_grid && v.arrows);
        assert_eq!(
            (v.x_label.as_deref(), v.y_label.as_deref()),
            (Some("time"), Some("h"))
        );
        assert_eq!((v.x_step, v.y_step), (Some(2.0), Some(5.0)));
        // The view event reports them, and omitted fields keep their values.
        let ev = cmd(&mut a, r#"{"t":"setView","grid":true}"#);
        let view = ev.iter().find(|e| e["t"] == "view").unwrap();
        assert_eq!(view["xLabel"], "time");
        assert_eq!(view["xStep"], 2.0);
        assert_eq!(view["minorGrid"], false);
        // Round trip through export.
        let json = doc::to_json(&a.doc);
        let back = doc::from_json(&json).unwrap();
        assert_eq!(back.view.x_step, Some(2.0));
        assert_eq!(back.view.y_label.as_deref(), Some("h"));
        // null, empty and 0 clear; a negative step is an error and changes nothing.
        cmd(
            &mut a,
            r#"{"t":"setView","xLabel":null,"yLabel":"","xStep":0,"yStep":null}"#,
        );
        let v = &a.doc.view;
        assert!(v.x_label.is_none() && v.y_label.is_none());
        assert!(v.x_step.is_none() && v.y_step.is_none());
        let ev = cmd(&mut a, r#"{"t":"setView","xStep":-1,"arrows":false}"#);
        assert!(error_msg(&ev).is_some());
        assert!(a.doc.view.arrows, "a rejected command changes nothing");
    }

    #[test]
    fn set_view_text_scale_is_saved_reported_and_validated() {
        let mut a = app();
        assert!(!doc::to_json(&a.doc).contains("textScale"));
        let ev = cmd(&mut a, r#"{"t":"setView","textScale":1.6}"#);
        let view = ev.iter().rfind(|e| e["t"] == "view").expect("a view event");
        assert_eq!(view["textScale"], 1.6);
        assert_eq!(a.doc.view.text_scale, 1.6);
        let json = doc::to_json(&a.doc);
        assert!(json.contains(r#""textScale":1.6"#), "{json}");
        // Round trips through a reload (share links carry the document).
        let mut b = app();
        let ev = cmd(&mut b, &format!(r#"{{"t":"loadDoc","json":{}}}"#, serde_json::to_string(&json).unwrap()));
        assert!(!has(&ev, "error"), "{ev:?}");
        assert_eq!(b.doc.view.text_scale, 1.6);
        // Out of range, zero and non-numbers are errors and change nothing.
        for bad in ["0.4", "3.1", "0", "-1", "\"big\"", "null"] {
            let ev = cmd(&mut a, &format!(r#"{{"t":"setView","textScale":{bad}}}"#));
            if bad == "null" {
                continue; // null is "absent" for an optional number
            }
            assert!(has(&ev, "error"), "{bad}: {ev:?}");
            assert_eq!(a.doc.view.text_scale, 1.6, "{bad}");
        }
        for ok in ["0.5", "3"] {
            let ev = cmd(&mut a, &format!(r#"{{"t":"setView","textScale":{ok}}}"#));
            assert!(!has(&ev, "error"), "{ok}: {ev:?}");
        }
        let ev = cmd(&mut a, r#"{"t":"setView","textScale":1}"#);
        assert_eq!(ev.iter().rfind(|e| e["t"] == "view").unwrap()["textScale"], 1.0);
        assert!(!doc::to_json(&a.doc).contains("textScale"));
    }

    #[test]
    fn set_view_weight_is_saved_reported_and_validated() {
        let mut a = app();
        let ev = cmd(&mut a, r#"{"t":"setView","weight":"bold"}"#);
        let view = ev.iter().rfind(|e| e["t"] == "view").expect("a view event");
        assert_eq!(view["weight"], "bold");
        assert_eq!(a.doc.view.weight, math_core::doc::Weight::Bold);
        assert!(doc::to_json(&a.doc).contains(r#""weight":"bold""#));
        // Unknown values are an error and change nothing.
        let ev = cmd(&mut a, r#"{"t":"setView","weight":"heavy"}"#);
        assert!(has(&ev, "error"));
        assert_eq!(a.doc.view.weight, math_core::doc::Weight::Bold);
        let ev = cmd(&mut a, r#"{"t":"setView","weight":"normal"}"#);
        assert_eq!(ev.iter().rfind(|e| e["t"] == "view").unwrap()["weight"], "normal");
        assert!(!doc::to_json(&a.doc).contains("weight"));
    }

    #[test]
    fn render_scale_widens_lines_without_touching_the_document() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x"}"#);
        a.frame(0.0);
        let before = doc::to_json(&a.doc);
        let max_w = |a: &App| {
            a.layers().iter().flat_map(|l| l.geometry.segments.iter()).map(|s| s.width).fold(0.0f32, f32::max)
        };
        let w1 = max_w(&a);
        cmd(&mut a, r#"{"t":"setRenderScale","scale":2}"#);
        a.frame(0.0);
        assert_eq!(max_w(&a), w1 * 2.0);
        assert_eq!(doc::to_json(&a.doc), before, "the scale is never saved");
        cmd(&mut a, r#"{"t":"setRenderScale","scale":1}"#);
        a.frame(0.0);
        assert_eq!(max_w(&a), w1);
    }

    #[test]
    fn set_view_grid_kind_and_axis_scales_round_trip() {
        let mut a = app();
        let ev = cmd(&mut a, r#"{"t":"setView","gridKind":"polar"}"#);
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        let view = ev.iter().rev().find(|e| e["t"] == "view").unwrap();
        assert_eq!(view["gridKind"], "polar");
        assert_eq!(view["xScale"], "linear");
        assert!(a.layers()[0]
            .geometry
            .labels
            .iter()
            .any(|l| l.axis == crate::scene::POLAR_LABEL_AXIS));
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","yScale":"log","window":{"min":[-10,0.01,-10],"max":[10,1000,10]}}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        let view = ev.iter().find(|e| e["t"] == "view").unwrap();
        assert_eq!(view["yScale"], "log");
        // The view event reports the visible WORLD window: here exactly the one asked for.
        assert!(
            (view["min"][1].as_f64().unwrap() - 0.01).abs() < 1e-9,
            "{view}"
        );
        assert!(
            (view["max"][1].as_f64().unwrap() - 1000.0).abs() < 1e-6,
            "{view}"
        );
        assert!(
            (view["max"][0].as_f64().unwrap() - 10.0).abs() < 1e-9,
            "{view}"
        );
        // Export and reload keep scales, grid and window.
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let hash = ev.iter().find(|e| e["t"] == "hash").unwrap()["hash"]
            .as_str()
            .unwrap()
            .to_string();
        let d = doc::decode_hash(&hash).unwrap();
        assert!(d.view.y_scale.is_log() && d.view.grid_kind == doc::GridKind::Polar);
        assert!((d.view.window.min[1] - 0.01).abs() < 1e-9);
        let shown = a.rig.window();
        let mut b = app();
        cmd(
            &mut b,
            &serde_json::json!({ "t": "loadHash", "hash": hash }).to_string(),
        );
        let w = b.rig.window();
        for i in 0..2 {
            assert!(
                (w.min[i] - shown.min[i]).abs() < 1e-9 && (w.max[i] - shown.max[i]).abs() < 1e-9
            );
        }
    }

    #[test]
    fn log_axis_needs_a_positive_window_and_rejects_without_changing_anything() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setView","gridKind":"polar"}"#);
        let w0 = a.rig.window();
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","grid":false,"gridKind":"rect"}"#,
        );
        let m = error_msg(&ev).expect("x min is -10");
        assert!(m.contains("logarithmic x axis"), "{m}");
        assert!(a.doc.view.x_scale.is_linear() && a.doc.view.grid);
        assert_eq!(a.doc.view.grid_kind, doc::GridKind::Polar);
        assert_eq!(a.rig.window(), w0);
        // A window with min <= 0 on an axis that is already logarithmic is refused as well.
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[1,-5,-5],"max":[100,5,5]}}"#,
        );
        assert!(a.doc.view.x_scale.is_log());
        let w1 = a.rig.window();
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[0,-5,-5],"max":[100,5,5]}}"#,
        );
        assert!(error_msg(&ev).is_some());
        assert_eq!(a.rig.window(), w1);
        assert_eq!(a.doc.view.window.min[0], 1.0);
        // Unknown scale names are malformed commands.
        let ev = cmd(&mut a, r#"{"t":"setView","yScale":"ln"}"#);
        assert!(error_msg(&ev).is_some_and(|m| m.contains("bad command")));
    }

    #[test]
    fn switching_back_to_linear_restores_the_exact_view() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-3.25,-7,-10],"max":[17.5,9,10]}}"#,
        );
        let (w0, d0) = (a.rig.window(), a.doc.view.window.clone());
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[0.01,-7,-10],"max":[17.5,9,10]}}"#,
        );
        assert!(a.rig.window() != w0);
        cmd(
            &mut a,
            r#"{"t":"setView","yScale":"log","window":{"min":[0.01,0.1,-10],"max":[17.5,9,10]}}"#,
        );
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","xScale":"linear","yScale":"linear"}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        // Windows set in between do not matter: the linear view from before comes back.
        assert_eq!(a.rig.window(), w0);
        // Straight there and back: exactly the old view.
        cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-3.25,-7,-10],"max":[17.5,9,10]}}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[0.01,-7,-10],"max":[17.5,9,10]}}"#,
        );
        cmd(&mut a, r#"{"t":"setView","xScale":"linear"}"#);
        assert_eq!(a.rig.window(), w0);
        assert_eq!(a.doc.view.window, d0);
        // Also after a pan in log mode; a window sent with the switch wins.
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[1,-7,-10],"max":[100,9,10]}}"#,
        );
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":300,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":300,"y":300}"#);
        let world = a.world_window();
        assert!(world.min[0] > 1.0, "panned right by decades: {world:?}");
        cmd(&mut a, r#"{"t":"setView","xScale":"linear"}"#);
        assert_eq!(a.rig.window(), w0);
        cmd(&mut a, r#"{"t":"setView","xScale":"log","window":{"min":[1,-7,-10],"max":[100,9,10]}}"#);
        cmd(&mut a, r#"{"t":"setView","xScale":"linear","window":{"min":[-1,-7,-10],"max":[5,9,10]}}"#);
        assert_eq!(a.doc.view.window.min[0], -1.0);
        assert!(a.log_entry.is_none());
    }

    #[test]
    fn item_labels_follow_a_point_drag_and_a_slider_within_the_same_event() {
        let mut a = app();
        let label = |a: &App, item: &str| {
            a.screen_labels().into_iter().find(|l| l.item.as_deref() == Some(item)).unwrap()
        };
        cmd(&mut a, r#"{"t":"setSlider","name":"s","value":1,"min":-5,"max":5}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(1,1)"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"q","latex":"(s,2)"}"#);
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"showLabel":true,"labelOffset":[10,10]}}"#);
        cmd(&mut a, r#"{"t":"setStyle","id":"q","style":{"showLabel":true}}"#);
        // A slider move relabels (and moves) the point built from it, with no frame in between.
        let before = label(&a, "q");
        assert_eq!(before.text, "(1, 2)");
        cmd(&mut a, r#"{"t":"setSlider","name":"s","value":3}"#);
        let after = label(&a, "q");
        assert_eq!(after.text, "(3, 2)");
        assert!(after.x > before.x && (after.y - before.y).abs() < 1e-9);
        // Dragging a point updates its label on every move, offset kept.
        let vp = (800.0, 600.0);
        let (px, py) = a.rig.world_to_pixel(a.map.fwd3([1.0, 1.0, 0.0]), vp);
        let start = label(&a, "p");
        cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"down","x":{px},"y":{py}}}"#));
        let mut last = start.x;
        for k in 1..=3 {
            cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"move","x":{},"y":{}}}"#, px + 20.0 * k as f64, py));
            let l = label(&a, "p");
            assert!(l.x > last, "step {k}: the label moved with the point");
            assert_ne!(l.text, start.text, "step {k}: text is current");
            assert_eq!(l.offset, Some([10.0, 10.0]));
            last = l.x;
        }
        cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"up","x":{},"y":{py}}}"#, px + 60.0));
    }

    #[test]
    fn dragging_a_point_on_a_log_axis_maps_back_to_world_values() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(10,5)"}"#);
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[0.1,-10,-10],"max":[1000,10,10]}}"#,
        );
        let vp = (800.0, 600.0);
        let at = |a: &App, x: f64, y: f64| a.rig.world_to_pixel(a.map.fwd3([x, y, 0.0]), vp);
        let (px, py) = at(&a, 10.0, 5.0);
        let (tx, ty) = at(&a, 100.0, 5.0);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{px},"y":{py}}}"#),
        );
        let ev = cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"move","x":{tx},"y":{ty}}}"#),
        );
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"up","x":{tx},"y":{ty}}}"#),
        );
        let edited = ev
            .iter()
            .find(|e| e["t"] == "itemEdited")
            .expect("the point moved");
        let latex = edited["latex"].as_str().unwrap();
        let inner = latex.trim_start_matches('(').trim_end_matches(')');
        let (x, y) = inner.split_once(',').unwrap();
        let (x, y): (f64, f64) = (x.trim().parse().unwrap(), y.trim().parse().unwrap());
        assert!((x - 100.0).abs() < 1.0, "{latex}");
        assert!((y - 5.0).abs() < 0.05, "{latex}");
        // The rig was not panned by the drag.
        let w = a.world_window();
        assert!((w.min[0] - 0.1).abs() < 1e-9 && (w.max[0] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn free_aspect_honours_the_typed_y_range_and_keeps_world_coordinates() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(5,500)"}"#);
        let view = |ev: &[serde_json::Value]| {
            ev.iter().rev().find(|e| e["t"] == "view").cloned().expect("view event")
        };
        // equal scales first: the map is the identity
        cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-10,-1000,-1],"max":[10,1000,1]}}"#,
        );
        assert!(a.map.is_linear());
        let px_per_unit = |a: &App| {
            let (x0, y0) = a.rig.world_to_pixel(a.map.fwd3([0.0, 0.0, 0.0]), (800.0, 600.0));
            let (x1, y1) = a.rig.world_to_pixel(a.map.fwd3([1.0, 1.0, 0.0]), (800.0, 600.0));
            ((x1 - x0).abs(), (y1 - y0).abs())
        };
        let (sx, sy) = px_per_unit(&a);
        assert!((sx / sy - 1.0).abs() < 1e-6, "equal pixels per unit: {sx} vs {sy}");
        // free aspect: the typed range is the shown range exactly
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","freeAspect":true,"window":{"min":[-10,-1000,-1],"max":[10,1000,1]}}"#,
        );
        let v = view(&ev);
        // free aspect: a world x unit is many times wider on screen than a y unit
        let (sx, sy) = px_per_unit(&a);
        assert!(sx / sy > 50.0, "x {sx} px/unit vs y {sy} px/unit");
        let near = |x: &serde_json::Value, w: f64| (x.as_f64().unwrap() - w).abs() < 1e-6;
        assert!(near(&v["min"][0], -10.0) && near(&v["max"][0], 10.0), "{v}");
        assert!(near(&v["min"][1], -1000.0) && near(&v["max"][1], 1000.0), "{v}");
        assert!(!a.map.is_linear() && a.map.scale[1] != 1.0);
        // a point drag reads back in world units
        let vp = (800.0, 600.0);
        let at = |a: &App, x: f64, y: f64| a.rig.world_to_pixel(a.map.fwd3([x, y, 0.0]), vp);
        let (px, py) = at(&a, 5.0, 500.0);
        let (tx, ty) = at(&a, 8.0, 250.0);
        cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"down","x":{px},"y":{py}}}"#));
        let ev = cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"move","x":{tx},"y":{ty}}}"#));
        cmd(&mut a, &format!(r#"{{"t":"pointer","phase":"up","x":{tx},"y":{ty}}}"#));
        let edited = ev.iter().find(|e| e["t"] == "itemEdited").expect("the point moved");
        let latex = edited["latex"].as_str().unwrap();
        let inner = latex.trim_start_matches('(').trim_end_matches(')');
        let (x, y) = inner.split_once(',').unwrap();
        let (x, y): (f64, f64) = (x.trim().parse().unwrap(), y.trim().parse().unwrap());
        assert!((x - 8.0).abs() < 0.1 && (y - 250.0).abs() < 5.0, "{latex}");
        // the view is preserved by the drag, and the flag round-trips through the document
        let w = a.world_window();
        assert!((w.min[1] + 1000.0).abs() < 1e-6 && (w.max[1] - 1000.0).abs() < 1e-6);
        assert!(doc::to_json(&a.doc).contains("freeAspect"));
        // turning it off returns to equal scales
        let ev = cmd(&mut a, r#"{"t":"setView","freeAspect":false}"#);
        assert!(a.map.is_linear(), "{:?}", a.map);
        let (sx, sy) = px_per_unit(&a);
        assert!((sx / sy - 1.0).abs() < 1e-6, "equal again: {sx} vs {sy}");
        let _ = view(&ev);
    }

    #[test]
    fn log_axes_are_2d_only_and_survive_a_mode_round_trip() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","window":{"min":[0.1,-10,-10],"max":[1000,10,10]}}"#,
        );
        let shown = a.rig.window();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        let w3 = a.rig.window();
        assert!(
            (w3.min[0] - 0.1).abs() < 1e-9 && (w3.max[0] - 1000.0).abs() < 1e-6,
            "3D uses world x"
        );
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        let back = a.rig.window();
        assert!(
            (back.min[0] - shown.min[0]).abs() < 1e-9 && (back.max[0] - shown.max[0]).abs() < 1e-9
        );
        // A slice cannot be shown on log axes and says so.
        cmd(&mut a, r#"{"t":"setExpr","id":"s","latex":"y=x"}"#);
        let ev = cmd(&mut a, r#"{"t":"setSlice","dim":1,"fixed":{"y":1}}"#);
        let sl = ev
            .iter()
            .rev()
            .find(|e| e["t"] == "slice")
            .expect("a slice event");
        assert!(
            sl["error"].as_str().unwrap_or("").contains("logarithmic"),
            "{sl}"
        );
    }
    #[test]
    fn orbit_in_3d_does_not_dirty_the_scene() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(10_000.0);
        let (y0, p0) = (a.rig.yaw(), a.rig.pitch());
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":450,"y":280}"#);
        assert!(a.rig.yaw() != y0 || a.rig.pitch() != p0);
        assert!(!a.dirty, "orbiting changes only the camera");
    }

    #[test]
    fn sliders_and_definitions_flow_through() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"e","latex":"y=a*x"}"#);
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"a","value":2}"#);
        let diag = ev.iter().find(|e| e["t"] == "diagnostics").unwrap();
        assert!(
            diag["items"].as_array().unwrap().is_empty(),
            "slider supplies a"
        );
        cmd(
            &mut a,
            r#"{"t":"addItem","id":"d","kind":"expression","latex":"f(x)=x^2"}"#,
        );
        cmd(&mut a, r#"{"t":"setExpr","id":"g","latex":"y=f(x)"}"#);
        assert!(a.diagnostics.is_empty());
    }

    #[test]
    fn export_and_reload_round_trips() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(10_000.0);
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let hash = ev.iter().find(|e| e["t"] == "hash").unwrap()["hash"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(hash.starts_with("v1."));
        let mut b = app();
        let ev = cmd(
            &mut b,
            &serde_json::json!({"t":"loadHash","hash":hash}).to_string(),
        );
        assert!(!has(&ev, "error"));
        assert_eq!(b.doc.items.len(), 1);
        assert_eq!(b.mode(), Mode::D3);
        // Hostile hashes are rejected, not trusted.
        assert!(has(
            &cmd(&mut b, r#"{"t":"loadHash","hash":"v1.AAAA"}"#),
            "error"
        ));
        assert!(has(
            &cmd(&mut b, r#"{"t":"loadDoc","json":"{\"v\":99}"}"#),
            "error"
        ));
    }

    #[test]
    fn tall_canvas_grid_covers_the_visible_height() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"resize","width":390,"height":844}"#);
        let g = &a.layers()[0].geometry;
        let max_y = g
            .segments
            .iter()
            .flat_map(|s| [s.p0[1].abs(), s.p1[1].abs()])
            .fold(0.0f32, f32::max);
        // 20 units across 390 px means ~21.6 units of half-height... the visible y half-range
        // is 10 * 844 / 390 = 21.6, far beyond the stored +-10.
        assert!(
            max_y > 20.0,
            "grid should reach the top and bottom of a tall canvas, got {max_y}"
        );
    }

    #[test]
    fn slider_step_and_removal() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setSlider","name":"a","value":1,"step":0.25}"#,
        );
        assert_eq!(a.doc.sliders["a"].step, Some(0.25));
        cmd(&mut a, r#"{"t":"removeSlider","name":"a"}"#);
        assert!(a.doc.sliders.is_empty());
        // Removing a slider that does not exist is a quiet no-op.
        assert!(!has(
            &cmd(&mut a, r#"{"t":"removeSlider","name":"zzz"}"#),
            "error"
        ));
    }

    #[test]
    fn pointer_hover_with_no_button_is_accepted() {
        let mut a = app();
        let ev = cmd(
            &mut a,
            r#"{"t":"pointer","phase":"move","x":10,"y":10,"button":-1}"#,
        );
        assert!(!has(&ev, "error"));
    }

    fn hover_at(a: &mut App, x: f64, y: f64) -> Vec<serde_json::Value> {
        cmd(a, &format!(r#"{{"t":"pointer","phase":"move","x":{x},"y":{y},"button":-1}}"#))
    }

    fn hover_event(ev: &[serde_json::Value]) -> Option<&serde_json::Value> {
        ev.iter().find(|e| e["t"] == "hover")
    }

    #[test]
    fn hover_finds_the_curve_under_the_pointer() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        let (px, py) = a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0));
        let ev = hover_at(&mut a, px, py + 3.0);
        let h = hover_event(&ev).expect("a hover event");
        assert_eq!(h["item"], "a");
        assert!((h["x"].as_f64().unwrap() - 2.0).abs() < 0.3, "x near 2, got {}", h["x"]);
        let (x, y) = (h["x"].as_f64().unwrap(), h["y"].as_f64().unwrap());
        assert!((y - x * x).abs() < 1e-9, "the point lies on the curve");
        assert!((h["py"].as_f64().unwrap() - py).abs() < 8.0, "py is where the curve is drawn");
        // the same pointer position again sends nothing; moving off the curve clears it
        assert!(hover_event(&hover_at(&mut a, px, py + 3.0)).is_none());
        let ev = hover_at(&mut a, px, py + 200.0);
        assert_eq!(hover_event(&ev).expect("cleared")["item"], serde_json::Value::Null);
    }

    #[test]
    fn hover_follows_edits_and_is_2d_only() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2"}"#);
        let (px, py) = a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0));
        assert!(hover_event(&hover_at(&mut a, px, py)).is_some());
        // an edit re-compiles the curve: y = 2x passes through (2, 4) too, y = x + 10 does not
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x+10"}"#);
        let ev = hover_at(&mut a, px, py);
        assert_eq!(hover_event(&ev).expect("cleared after the edit")["item"], serde_json::Value::Null);
        // 3D has no hover
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(!has(&hover_at(&mut a, 400.0, 300.0), "error"));
    }

    fn pick_at(a: &mut App, x: f64, y: f64) -> Vec<serde_json::Value> {
        cmd(a, &format!(r#"{{"t":"pick","x":{x},"y":{y}}}"#))
    }

    fn analysis(ev: &[serde_json::Value]) -> Option<&serde_json::Value> {
        ev.iter().find(|e| e["t"] == "analysis")
    }

    fn kinds(an: &serde_json::Value, kind: &str) -> Vec<(f64, f64)> {
        an["points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["kind"] == kind)
            .map(|p| (p["x"].as_f64().unwrap(), p["y"].as_f64().unwrap()))
            .collect()
    }

    #[test]
    fn picking_a_curve_sends_its_special_points() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4"}"#);
        let (px, py) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let ev = pick_at(&mut a, px, py);
        let an = analysis(&ev).expect("an analysis event");
        assert_eq!(an["item"], "a");
        let roots = kinds(an, "root");
        assert_eq!(roots.len(), 2);
        assert!((roots[0].0 + 2.0).abs() < 1e-6 && (roots[1].0 - 2.0).abs() < 1e-6);
        let mins = kinds(an, "minimum");
        assert!(mins.len() == 1 && mins[0].0.abs() < 1e-6 && (mins[0].1 + 4.0).abs() < 1e-6);
        assert_eq!(kinds(an, "y-intercept").len(), 1);
        // picking the same curve again changes nothing, so nothing is sent
        assert!(analysis(&pick_at(&mut a, px, py)).is_none());
        // a click on empty space clears the selection
        let ev = pick_at(&mut a, 10.0, 10.0);
        let an = analysis(&ev).expect("cleared");
        assert!(an["item"].is_null() && an["points"].as_array().unwrap().is_empty());
    }

    #[test]
    fn hover_pick_and_analysis_respect_a_range_restriction() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4 {x>0}"}"#);
        // the clipped-off left branch cannot be hovered ...
        let (lx, ly) = a.rig.world_to_pixel([-3.0, 5.0, 0.0], (800.0, 600.0));
        let ev = hover_at(&mut a, lx, ly);
        assert!(hover_event(&ev).is_none_or(|h| h["item"].is_null()), "{ev:?}");
        // ... the kept one can
        let (rx, ry) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let ev = hover_at(&mut a, rx, ry);
        assert_eq!(hover_event(&ev).expect("hover")["item"], "a");
        // analysis lists the root at x=2 only, not the one at -2 or the vertex at x=0
        let ev = pick_at(&mut a, rx, ry);
        let an = analysis(&ev).expect("an analysis event");
        let roots = kinds(an, "root");
        assert_eq!(roots.len(), 1, "{an}");
        assert!((roots[0].0 - 2.0).abs() < 1e-6);
        assert!(kinds(an, "minimum").is_empty(), "{an}");
    }

    #[test]
    fn analysis_lists_intersections_with_the_other_curves() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"b","latex":"y=x"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"x^2+y^2=25"}"#);
        let (px, py) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let ev = pick_at(&mut a, px, py);
        let an = analysis(&ev).expect("an analysis event");
        let hits: Vec<&serde_json::Value> =
            an["points"].as_array().unwrap().iter().filter(|p| p["kind"] == "intersection").collect();
        // with y=x: x^2 - x - 4 = 0; with the circle: 4 points of x^2 + (x^2-4)^2 = 25
        let with = |w: &str| hits.iter().filter(|p| p["with"] == w).count();
        assert_eq!(with("b"), 2, "{an}");
        assert_eq!(with("c"), 2, "{an}");
        let r = (1.0 + 17f64.sqrt()) / 2.0;
        assert!(hits.iter().any(|p| p["with"] == "b" && (p["x"].as_f64().unwrap() - r).abs() < 1e-6));
        assert!(hits.iter().all(|p| p["px"].is_number() && p["py"].is_number()));
        assert!(an["points"].as_array().unwrap().len() <= 48);
        // The roots and the rest are still listed.
        assert_eq!(kinds(an, "root").len(), 2);
    }

    #[test]
    fn intersections_respect_a_range_restriction() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4 {x>0}"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"b","latex":"y=x"}"#);
        let (px, py) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let an = analysis(&pick_at(&mut a, px, py)).cloned().expect("selected");
        assert_eq!(an["item"], "a");
        // y=x meets y=x^2-4 at (1±sqrt(17))/2; only the x>0 one is on the drawn curve
        let xs: Vec<f64> = kinds(&an, "intersection").iter().map(|p| p.0).collect();
        assert_eq!(xs.len(), 1, "{an}");
        assert!((xs[0] - (1.0 + 17f64.sqrt()) / 2.0).abs() < 1e-6, "{an}");
        // and from the other side: picking y=x lists the same single point with a
        let (bx, by) = a.rig.world_to_pixel([-3.0, -3.0, 0.0], (800.0, 600.0));
        let an = analysis(&pick_at(&mut a, bx, by)).cloned().expect("selected");
        assert_eq!(an["item"], "b");
        assert_eq!(kinds(&an, "intersection").len(), 1, "{an}");
    }

    #[test]
    fn selecting_an_item_by_id_sends_its_special_points() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4"}"#);
        let ev = cmd(&mut a, r#"{"t":"selectItem","id":"a"}"#);
        let an = analysis(&ev).expect("analysis sent");
        assert_eq!(an["item"], "a");
        assert!(!kinds(an, "root").is_empty(), "{an}");
        let ev = cmd(&mut a, r#"{"t":"selectItem","id":null}"#);
        assert_eq!(analysis(&ev).expect("cleared")["item"], serde_json::Value::Null);
    }

    #[test]
    fn a_picked_implicit_curve_lists_its_intersections() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"x^2+y^2=25"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"d","latex":"(x-5)^2+y^2=25"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"e","latex":"y=x"}"#);
        let (px, py) = a.rig.world_to_pixel([-5.0, 0.0, 0.0], (800.0, 600.0));
        let ev = pick_at(&mut a, px, py);
        let an = analysis(&ev).expect("selected");
        assert_eq!(an["item"], "c");
        let n = |w: &str| an["points"].as_array().unwrap().iter().filter(|p| p["with"] == w).count();
        assert_eq!((n("d"), n("e")), (2, 2), "{an}");
    }

    #[test]
    fn hidden_curves_have_no_intersections() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=x^2-4"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"b","latex":"y=x"}"#);
        cmd(&mut a, r#"{"t":"setHidden","id":"b","hidden":true}"#);
        let (px, py) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let an = analysis(&pick_at(&mut a, px, py)).cloned().expect("selected");
        assert!(kinds(&an, "intersection").is_empty(), "{an}");
    }

    fn ptr_at(a: &mut App, phase: &str, x: f64, y: f64) -> Vec<serde_json::Value> {
        cmd(a, &format!(r#"{{"t":"pointer","phase":"{phase}","x":{x},"y":{y},"button":0}}"#))
    }

    fn slider_of(a: &App, name: &str) -> f64 {
        a.doc.sliders[name].value
    }

    /// `y = a(x-h)^2 + k` with sliders, selected by a click on its flank at x = 2.
    fn selected_parabola(a: &mut App) -> (f64, f64) {
        cmd(a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(a, r#"{"t":"setSlider","name":"a","value":1,"min":-5,"max":5}"#);
        cmd(a, r#"{"t":"setSlider","name":"h","value":0,"min":-10,"max":10}"#);
        cmd(a, r#"{"t":"setSlider","name":"k","value":0,"min":-10,"max":10}"#);
        cmd(a, r#"{"t":"setExpr","id":"c","latex":"y=a(x-h)^2+k"}"#);
        let (px, py) = a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0));
        pick_at(a, px, py);
        (px, py)
    }

    #[test]
    fn dragging_the_selected_curve_changes_its_sliders() {
        let mut a = app();
        let (px, py) = selected_parabola(&mut a);
        // hovering the selected curve announces what a drag would change
        hover_at(&mut a, 10.0, 10.0);
        let ev = hover_at(&mut a, px, py);
        let h = hover_event(&ev).expect("hover");
        assert_eq!(h["params"], serde_json::json!(["a", "h", "k"]));
        let ev = ptr_at(&mut a, "down", px, py);
        assert!(ev.iter().any(|e| e["t"] == "curveDrag" && e["active"] == true));
        // drag it 1 unit right and 1 up (world units to pixels)
        let (qx, qy) = a.rig.world_to_pixel([3.0, 5.0, 0.0], (800.0, 600.0));
        let ev = ptr_at(&mut a, "move", qx, qy);
        assert!(ev.iter().any(|e| e["t"] == "sliderValue"), "{ev:?}");
        let (aa, hh, kk) = (slider_of(&a, "a"), slider_of(&a, "h"), slider_of(&a, "k"));
        // the curve passes through the pointer and kept its shape: a translation
        assert!((aa * (3.0 - hh).powi(2) + kk - 5.0).abs() < 0.1, "a={aa} h={hh} k={kk}");
        assert!((aa - 1.0).abs() < 0.15 && (hh - 1.0).abs() < 0.2, "a={aa} h={hh} k={kk}");
        let ev = ptr_at(&mut a, "up", qx, qy);
        assert!(ev.iter().any(|e| e["t"] == "curveDrag" && e["active"] == false && e["cancelled"] == false));
        // the press did not pan
        assert_eq!(a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0)), (px, py));
    }

    #[test]
    fn a_second_drag_right_after_the_first_works_and_escape_restores_it() {
        let mut a = app();
        let (px, py) = selected_parabola(&mut a);
        let at = |a: &App, x: f64, y: f64| a.rig.world_to_pixel([x, y, 0.0], (800.0, 600.0));
        hover_at(&mut a, px, py);
        ptr_at(&mut a, "down", px, py);
        let (qx, qy) = at(&a, 4.0, 4.0);
        ptr_at(&mut a, "move", qx, qy);
        ptr_at(&mut a, "up", qx, qy);
        let after_first = (slider_of(&a, "h"), slider_of(&a, "k"));
        assert!((after_first.0 - 2.0).abs() < 0.2, "{after_first:?}");
        // second drag: press on the curve where the first one left it, no click in between
        let y = (4.0 - slider_of(&a, "h")).powi(2) * slider_of(&a, "a") + slider_of(&a, "k");
        let (rx, ry) = at(&a, 4.0, y);
        hover_at(&mut a, rx, ry);
        let ev = ptr_at(&mut a, "down", rx, ry);
        assert!(ev.iter().any(|e| e["t"] == "curveDrag"), "no drag started: {ev:?}");
        let (sx, sy) = at(&a, 6.0, y);
        ptr_at(&mut a, "move", sx, sy);
        assert!((slider_of(&a, "h") - after_first.0 - 2.0).abs() < 0.3, "h = {}", slider_of(&a, "h"));
        cmd(&mut a, r#"{"t":"cancelDrag"}"#);
        assert_eq!((slider_of(&a, "h"), slider_of(&a, "k")), after_first);
    }

    #[test]
    fn an_unselected_curve_is_panned_not_dragged() {
        let mut a = app();
        let (px, py) = selected_parabola(&mut a);
        pick_at(&mut a, 10.0, 10.0); // clears the selection
        hover_at(&mut a, 10.0, 10.0);
        assert!(hover_event(&hover_at(&mut a, px, py)).unwrap().get("params").is_none());
        ptr_at(&mut a, "down", px, py);
        let ev = ptr_at(&mut a, "move", px + 40.0, py + 20.0);
        assert!(!ev.iter().any(|e| e["t"] == "sliderValue" || e["t"] == "curveDrag"));
        assert_eq!(slider_of(&a, "a"), 1.0);
        assert_ne!(a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0)), (px, py));
    }

    #[test]
    fn a_curve_without_parameters_has_no_grab() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"y=2x^2"}"#);
        let (px, py) = a.rig.world_to_pixel([2.0, 8.0, 0.0], (800.0, 600.0));
        pick_at(&mut a, px, py);
        hover_at(&mut a, 10.0, 10.0);
        assert!(hover_event(&hover_at(&mut a, px, py)).unwrap().get("params").is_none());
        ptr_at(&mut a, "down", px, py);
        let ev = ptr_at(&mut a, "move", px + 30.0, py);
        assert!(!ev.iter().any(|e| e["t"] == "curveDrag"));
    }

    #[test]
    fn escape_restores_the_starting_values() {
        let mut a = app();
        let (px, py) = selected_parabola(&mut a);
        ptr_at(&mut a, "down", px, py);
        ptr_at(&mut a, "move", px + 60.0, py - 40.0);
        assert!(slider_of(&a, "h") != 0.0);
        let ev = cmd(&mut a, r#"{"t":"cancelDrag"}"#);
        assert!(ev.iter().any(|e| e["t"] == "curveDrag" && e["active"] == false && e["cancelled"] == true));
        assert_eq!((slider_of(&a, "a"), slider_of(&a, "h"), slider_of(&a, "k")), (1.0, 0.0, 0.0));
        // later moves of the same press do nothing
        let ev = ptr_at(&mut a, "move", px + 90.0, py);
        assert!(!ev.iter().any(|e| e["t"] == "sliderValue"));
    }

    const VP: (f64, f64) = (800.0, 600.0);

    fn disp_px(a: &App, x: f64, y: f64) -> (f64, f64) {
        a.rig.world_to_pixel(a.map.fwd3([x, y, 0.0]), VP)
    }

    fn set_sliders(a: &mut App, defs: &[(&str, f64, f64, f64)]) {
        for (n, v, lo, hi) in defs {
            cmd(a, &format!(r#"{{"t":"setSlider","name":"{n}","value":{v},"min":{lo},"max":{hi}}}"#));
        }
    }

    /// Selects the curve under `(px, py)` with a click and starts a drag there.
    fn select_and_press(a: &mut App, px: f64, py: f64) {
        hover_at(a, 10.0, 10.0);
        let ev = hover_at(a, px, py);
        assert!(hover_event(&ev).is_some_and(|h| h["item"] != serde_json::Value::Null), "no curve under the pointer: {ev:?}");
        pick_at(a, px, py);
        let ev = ptr_at(a, "down", px, py);
        assert!(ev.iter().any(|e| e["t"] == "curveDrag" && e["active"] == true), "no drag started: {ev:?}");
    }

    #[test]
    fn dragging_a_curve_on_log_axes_keeps_it_under_the_pointer() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        set_sliders(&mut a, &[("a", 1.0, 0.01, 100.0), ("b", 2.0, 0.0, 5.0)]);
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"y=a*x^b"}"#);
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","xScale":"log","yScale":"log","window":{"min":[0.1,0.01,-1],"max":[100,10000,1]}}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        assert!(!a.map.is_linear());
        let (px, py) = disp_px(&a, 1.0, 1.0);
        select_and_press(&mut a, px, py);
        // one decade right, two and a half decades... up: the curve must pass through (10, 500)
        let (qx, qy) = disp_px(&a, 10.0, 500.0);
        ptr_at(&mut a, "move", qx, qy);
        let (aa, bb) = (slider_of(&a, "a"), slider_of(&a, "b"));
        // a power law is a straight line in log-log display: translating it keeps the slope b
        assert!((bb - 2.0).abs() < 0.1, "b = {bb}");
        let (cx, cy) = disp_px(&a, 10.0, aa * 10f64.powf(bb));
        assert!((cx - qx).abs() < 1.0 && (cy - qy).abs() < 3.0, "curve at ({cx},{cy}), pointer ({qx},{qy}), a={aa} b={bb}");
        ptr_at(&mut a, "up", qx, qy);
    }

    #[test]
    fn a_coarse_slider_step_snaps_without_drift() {
        let run = |path: &[(f64, f64)]| {
            let mut a = app();
            cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
            cmd(&mut a, r#"{"t":"setSlider","name":"a","value":1,"min":-5,"max":5,"step":0.5}"#);
            cmd(&mut a, r#"{"t":"setSlider","name":"b","value":0,"min":-5,"max":5,"step":0.5}"#);
            cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"y=a*x+b"}"#);
            let (px, py) = disp_px(&a, 2.0, 2.0);
            select_and_press(&mut a, px, py);
            for (wx, wy) in path {
                let (qx, qy) = disp_px(&a, *wx, *wy);
                ptr_at(&mut a, "move", qx, qy);
                for n in ["a", "b"] {
                    let v = slider_of(&a, n);
                    assert!(((v / 0.5).round() * 0.5 - v).abs() < 1e-9, "{n} = {v} is off the 0.5 grid");
                }
            }
            (slider_of(&a, "a"), slider_of(&a, "b"))
        };
        // straight to the target, and by 40 small moves: the same step values, no drift
        let direct = run(&[(2.7, 3.3)]);
        let small: Vec<(f64, f64)> = (1..=40).map(|i| (2.0 + 0.7 * i as f64 / 40.0, 2.0 + 1.3 * i as f64 / 40.0)).collect();
        let stepped = run(&small);
        assert_eq!(direct, stepped, "path dependent: {direct:?} vs {stepped:?}");
        // and back to where it started: the starting values return exactly
        let mut round_trip = small.clone();
        round_trip.extend((0..=40).map(|i| (2.7 - 0.7 * i as f64 / 40.0, 3.3 - 1.3 * i as f64 / 40.0)));
        assert_eq!(run(&round_trip), (1.0, 0.0));
    }

    fn circle_setup(a: &mut App, latex: &str) {
        cmd(a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(a, &format!(r#"{{"t":"setExpr","id":"c","latex":"{latex}"}}"#));
    }

    #[test]
    fn dragging_a_circle_by_its_edge_moves_its_center() {
        let mut a = app();
        set_sliders(&mut a, &[("h", 0.0, -10.0, 10.0), ("k", 0.0, -10.0, 10.0), ("r", 3.0, 0.5, 10.0)]);
        circle_setup(&mut a, "(x-h)^2+(y-k)^2=r^2");
        let (px, py) = disp_px(&a, 3.0, 0.0);
        // a few pixels off the line still grabs it (10 px reach)
        select_and_press(&mut a, px + 4.0, py + 3.0);
        assert_eq!(a.selected.as_deref(), Some("c"), "selecting an implicit curve survives analysis");
        let (qx, qy) = disp_px(&a, 4.0, 1.0);
        ptr_at(&mut a, "move", qx + 4.0, qy + 3.0);
        let (h, k, r) = (slider_of(&a, "h"), slider_of(&a, "k"), slider_of(&a, "r"));
        assert!((h - 1.0).abs() < 0.15 && (k - 1.0).abs() < 0.15 && (r - 3.0).abs() < 0.15, "h={h} k={k} r={r}");
        ptr_at(&mut a, "up", qx, qy);
    }

    #[test]
    fn pulling_a_circle_radially_changes_its_radius_when_only_r_is_a_slider() {
        let mut a = app();
        set_sliders(&mut a, &[("r", 3.0, 0.5, 10.0)]);
        circle_setup(&mut a, "x^2+y^2=r^2");
        let (px, py) = disp_px(&a, 0.0, 3.0);
        select_and_press(&mut a, px, py);
        let (qx, qy) = disp_px(&a, 0.0, 5.0);
        ptr_at(&mut a, "move", qx, qy);
        let r = slider_of(&a, "r");
        assert!((r - 5.0).abs() < 0.2, "r = {r}");
        // Escape puts it back
        cmd(&mut a, r#"{"t":"cancelDrag"}"#);
        assert_eq!(slider_of(&a, "r"), 3.0);
    }

    #[test]
    fn dragging_a_parametric_ellipse_translates_it() {
        let mut a = app();
        set_sliders(
            &mut a,
            &[("h", 0.0, -10.0, 10.0), ("k", 0.0, -10.0, 10.0), ("a", 3.0, 0.5, 8.0), ("b", 2.0, 0.5, 8.0)],
        );
        circle_setup(&mut a, "(h+a\\\\cos(t),k+b\\\\sin(t))");
        let (px, py) = disp_px(&a, 3.0, 0.0);
        select_and_press(&mut a, px, py + 5.0);
        let (qx, qy) = disp_px(&a, 4.0, 1.0);
        ptr_at(&mut a, "move", qx, qy + 5.0);
        let v = |n| slider_of(&a, n);
        assert!((v("h") - 1.0).abs() < 0.15 && (v("k") - 1.0).abs() < 0.15, "h={} k={}", v("h"), v("k"));
        assert!((v("a") - 3.0).abs() < 0.15 && (v("b") - 2.0).abs() < 0.15, "a={} b={}", v("a"), v("b"));
        ptr_at(&mut a, "up", qx, qy);
    }

    #[test]
    fn dragging_a_polar_curve_changes_its_slider() {
        let mut a = app();
        set_sliders(&mut a, &[("a", 3.0, 0.5, 10.0)]);
        circle_setup(&mut a, "r=a\\\\cos(\\\\theta)");
        let (px, py) = disp_px(&a, 3.0, 0.0);
        select_and_press(&mut a, px, py);
        let (qx, qy) = disp_px(&a, 4.0, 0.0);
        ptr_at(&mut a, "move", qx, qy);
        let v = slider_of(&a, "a");
        assert!((v - 4.0).abs() < 0.35, "a = {v}");
    }

    #[test]
    fn implicit_curves_are_picked_on_log_axes_too() {
        let mut a = app();
        set_sliders(&mut a, &[("r", 3.0, 0.5, 10.0)]);
        circle_setup(&mut a, "x^2+y^2=r^2");
        cmd(&mut a, r#"{"t":"setView","xScale":"log","yScale":"log","window":{"min":[0.1,0.1,-1],"max":[100,100,1]}}"#);
        // (x, y) = (2, sqrt(5)) is on the circle
        let (px, py) = disp_px(&a, 2.0, 5f64.sqrt());
        let ev = hover_at(&mut a, px + 2.0, py - 2.0);
        assert_eq!(hover_event(&ev).expect("hover")["item"], "c", "{ev:?}");
        let ev = hover_at(&mut a, px + 80.0, py - 80.0);
        assert_eq!(hover_event(&ev).expect("cleared")["item"], serde_json::Value::Null);
    }

    #[test]
    fn dragging_a_numeric_definition_rewrites_its_item() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"d","latex":"b=0"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"y=x^2+b"}"#);
        let (px, py) = a.rig.world_to_pixel([2.0, 4.0, 0.0], (800.0, 600.0));
        pick_at(&mut a, px, py);
        ptr_at(&mut a, "down", px, py);
        let (_, qy) = a.rig.world_to_pixel([2.0, 6.0, 0.0], (800.0, 600.0));
        let ev = ptr_at(&mut a, "move", px, qy);
        let e = ev.iter().find(|e| e["t"] == "itemEdited" && e["id"] == "d").expect("itemEdited");
        let v: f64 = e["latex"].as_str().unwrap().trim_start_matches("b=").parse().unwrap();
        assert!((v - 2.0).abs() < 0.1, "{e}");
    }

    #[test]
    fn a_slider_at_its_range_end_is_not_driven_out_of_it() {
        let mut a = app();
        let (px, py) = selected_parabola(&mut a);
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":5,"min":-5,"max":5}"#);
        let (px, py) = (px, py);
        ptr_at(&mut a, "down", px, py);
        let (qx, qy) = a.rig.world_to_pixel([2.0, -8.0, 0.0], (800.0, 600.0));
        ptr_at(&mut a, "move", qx, qy);
        for n in ["a", "h", "k"] {
            let v = slider_of(&a, n);
            assert!((a.doc.sliders[n].min..=a.doc.sliders[n].max).contains(&v));
        }
    }

    #[test]
    fn the_selected_curves_points_follow_the_view_and_edits() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":"y=sin(x)"}"#);
        let (px, py) = a.rig.world_to_pixel([0.0, 0.0, 0.0], (800.0, 600.0));
        let an = analysis(&pick_at(&mut a, px, py)).cloned().expect("selected");
        let before = kinds(&an, "root").len();
        assert!(before >= 5, "sin has several roots in -10..10, got {before}");
        // zooming in to about -2..2 leaves only the root at 0
        let ev = cmd(&mut a, r#"{"t":"setView","window":{"min":[-2,-2,-1],"max":[2,2,1]}}"#);
        let an = analysis(&ev).expect("the points are recomputed for the new window");
        let roots = kinds(an, "root");
        assert!(roots.len() == 1 && roots[0].0.abs() < 1e-6, "only the root at 0 is in view, got {roots:?}");
        // deleting the curve clears the selection
        let ev = cmd(&mut a, r#"{"t":"setExpr","id":"a","latex":""}"#);
        let an = analysis(&ev).expect("cleared when the curve went away");
        assert!(an["item"].is_null());
    }

    #[test]
    fn screen_labels_track_the_camera() {
        let mut a = app();
        let labels = a.screen_labels();
        assert!(
            labels.iter().any(|l| l.visible && l.axis == 0),
            "x tick labels on screen"
        );
        // The label for x=0 sits where the world origin projects: the screen centre column.
        let zero = labels.iter().find(|l| l.text == "0" && l.axis == 0);
        if let Some(z) = zero {
            assert!(
                (z.x - 400.0).abs() < 2.0,
                "x=0 is at the horizontal centre, got {}",
                z.x
            );
        }
        // Panning right moves labels left (same labels, new positions).
        let before = labels.iter().find(|l| l.visible).unwrap().x;
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":350,"y":300}"#);
        let after = a
            .screen_labels()
            .into_iter()
            .find(|l| l.text == labels.iter().find(|v| v.visible).unwrap().text && l.axis == 0)
            .map(|l| l.x);
        if let Some(after) = after {
            assert!(after != before);
        }
    }

    #[test]
    fn theme_switch_recolours_scene() {
        let mut a = app();
        let light_bg = a.background();
        cmd(&mut a, r#"{"t":"setTheme","dark":true}"#);
        assert!(a.background() != light_bg);
    }

    #[test]
    fn resize_updates_aspect_and_rebuilds() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"resize","width":390,"height":844}"#);
        assert_eq!(a.size(), (390, 844));
        assert!(!a.layers().is_empty());
        // Degenerate sizes are clamped, not fatal.
        cmd(&mut a, r#"{"t":"resize","width":0,"height":0}"#);
        assert_eq!(a.size(), (1, 1));
    }
    fn ptr(a: &mut App, phase: &str, x: f64, y: f64) -> Vec<serde_json::Value> {
        cmd(
            a,
            &format!(r#"{{"t":"pointer","phase":"{phase}","x":{x},"y":{y}}}"#),
        )
    }

    #[test]
    fn dragging_a_literal_point_rewrites_its_text() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(2,3)"}"#);
        let w0 = a.rig.window();
        // (2, 3) is at pixel (480, 180) with 40 px per unit.
        ptr(&mut a, "down", 480.0, 180.0);
        let ev = ptr(&mut a, "move", 520.0, 140.0);
        let edit = ev
            .iter()
            .find(|e| e["t"] == "itemEdited")
            .expect("itemEdited event");
        assert_eq!(edit["id"], "p");
        assert_eq!(edit["latex"], "(3, 4)");
        assert_eq!(a.doc.items[0].latex, "(3, 4)");
        assert_eq!(a.rig.window(), w0, "grabbing a point must not pan");
        ptr(&mut a, "up", 520.0, 140.0);
        // Pressing on empty canvas still pans.
        ptr(&mut a, "down", 100.0, 500.0);
        ptr(&mut a, "move", 60.0, 500.0);
        assert!(a.rig.window() != w0);
    }

    #[test]
    fn a_formula_column_is_computed_and_follows_sliders() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setSlider","name":"k","value":2,"min":0,"max":10}"#);
        cmd(&mut a, r#"{"t":"addTable","id":"t"}"#);
        for (r, v) in ["1", "2", "3"].iter().enumerate() {
            cmd(&mut a, &format!(r#"{{"t":"setCell","id":"t","row":{r},"col":0,"value":"{v}"}}"#));
        }
        let ev = cmd(&mut a, r#"{"t":"setColumnFormula","id":"t","col":1,"formula":"k x_1^2"}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        let table = |ev: &[serde_json::Value]| {
            ev.iter().rev().find(|e| e["t"] == "table").cloned().expect("table event")
        };
        let t = table(&ev);
        assert_eq!(t["columns"][1]["formula"], "k x_1^2");
        let cells: Vec<String> = t["columns"][1]["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        assert_eq!(&cells[..3], ["2", "8", "18"]);
        // a slider move re-sends the table with the new values
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"k","value":3}"#);
        let cells: Vec<String> = table(&ev)["columns"][1]["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        assert_eq!(&cells[..3], ["3", "12", "27"]);
        // the cells are locked; clearing the formula unlocks them
        let ev = cmd(&mut a, r#"{"t":"setCell","id":"t","row":0,"col":1,"value":"7"}"#);
        assert!(ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        let ev = cmd(&mut a, r#"{"t":"setColumnFormula","id":"t","col":1,"formula":null}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        let ev = cmd(&mut a, r#"{"t":"setCell","id":"t","row":0,"col":1,"value":"7"}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        // a bad formula is an error and changes nothing
        let ev = cmd(&mut a, r#"{"t":"setColumnFormula","id":"t","col":1,"formula":"x_1+"}"#);
        assert!(ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
    }

    #[test]
    fn table_points_of_a_drag_column_move_and_write_back_into_their_cells() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"addTable","id":"t"}"#);
        for (r, (x, y)) in [("1", "2"), ("3", "4"), ("", "")].iter().enumerate() {
            cmd(&mut a, &format!(r#"{{"t":"setCell","id":"t","row":{r},"col":0,"value":"{x}"}}"#));
            cmd(&mut a, &format!(r#"{{"t":"setCell","id":"t","row":{r},"col":1,"value":"{y}"}}"#));
        }
        let vp = (800.0, 600.0);
        let at = |a: &App, x: f64, y: f64| a.rig.world_to_pixel(a.map.fwd3([x, y, 0.0]), vp);
        let drag = |a: &mut App, from: (f64, f64), to: (f64, f64)| {
            let (px, py) = at(a, from.0, from.1);
            let (tx, ty) = at(a, to.0, to.1);
            cmd(a, &format!(r#"{{"t":"pointer","phase":"down","x":{px},"y":{py}}}"#));
            let ev = cmd(a, &format!(r#"{{"t":"pointer","phase":"move","x":{tx},"y":{ty}}}"#));
            cmd(a, &format!(r#"{{"t":"pointer","phase":"up","x":{tx},"y":{ty}}}"#));
            ev
        };
        let cell = |a: &App, r: usize, c: usize| {
            a.doc.items.iter().find(|i| i.id == "t").unwrap().table.as_ref().unwrap().columns[c].cells[r].clone()
        };
        // not draggable until the column's Drag is on: the press pans instead
        let w0 = a.rig.window();
        drag(&mut a, (1.0, 2.0), (2.0, 5.0));
        assert_eq!((cell(&a, 0, 0).as_str(), cell(&a, 0, 1).as_str()), ("1", "2"));
        assert!(a.rig.window() != w0, "a press with no handle pans");
        cmd(&mut a, r#"{"t":"setView","window":{"min":[-10,-7.5,-1],"max":[10,7.5,1]}}"#);
        let ev = cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":1,"style":{"drag":true}}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        // now the point (1, 2) follows the pointer and rewrites its two cells
        let ev = drag(&mut a, (1.0, 2.0), (2.0, 5.0));
        let t = ev.iter().find(|e| e["t"] == "table").expect("table event");
        assert_eq!(t["drag"], true, "{t}");
        let (x, y): (f64, f64) = (cell(&a, 0, 0).parse().unwrap(), cell(&a, 0, 1).parse().unwrap());
        assert!((x - 2.0).abs() < 0.1 && (y - 5.0).abs() < 0.1, "{x} {y}");
        // the other row is untouched, the blank row has no handle
        assert_eq!((cell(&a, 1, 0).as_str(), cell(&a, 1, 1).as_str()), ("3", "4"));
        // a formula column cannot be dragged
        cmd(&mut a, r#"{"t":"setColumnFormula","id":"t","col":1,"formula":"x_1+1"}"#);
        let before = (cell(&a, 0, 0), cell(&a, 1, 0));
        drag(&mut a, (x, x + 1.0), (6.0, 6.0));
        assert_eq!((cell(&a, 0, 0), cell(&a, 1, 0)), before);
    }

    #[test]
    fn slider_play_settings_are_saved_in_the_document() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":1,"min":0,"max":5}"#);
        assert!(a.doc.slider_play.is_empty());
        let ev = cmd(&mut a, r#"{"t":"setSliderPlay","name":"a","mode":"loop","speed":2}"#);
        assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
        let p = &a.doc.slider_play["a"];
        assert_eq!((p.mode, p.speed), (Some(doc::SliderPlayMode::Loop), Some(2.0)));
        // it is part of the saved document and survives a reload
        let json = doc::to_json(&a.doc);
        assert!(json.contains("sliderPlay"), "{json}");
        let back = doc::from_json(&json).unwrap();
        assert_eq!(back.slider_play, a.doc.slider_play);
        // a key left out is left alone; the defaults are stored as nothing
        cmd(&mut a, r#"{"t":"setSliderPlay","name":"a","speed":1}"#);
        let p = &a.doc.slider_play["a"];
        assert_eq!((p.mode, p.speed), (Some(doc::SliderPlayMode::Loop), None));
        cmd(&mut a, r#"{"t":"setSliderPlay","name":"a","mode":"oscillate"}"#);
        assert!(a.doc.slider_play.is_empty());
        assert!(!doc::to_json(&a.doc).contains("sliderPlay"));
        // errors change nothing
        for bad in [
            r#"{"t":"setSliderPlay","name":"nope","mode":"loop"}"#,
            r#"{"t":"setSliderPlay","name":"a","speed":0}"#,
            r#"{"t":"setSliderPlay","name":"a","speed":99}"#,
        ] {
            let ev = cmd(&mut a, bad);
            assert!(ev.iter().any(|e| e["t"] == "error"), "{bad}");
        }
        assert!(a.doc.slider_play.is_empty());
        // removing the slider drops its settings
        cmd(&mut a, r#"{"t":"setSliderPlay","name":"a","mode":"once"}"#);
        cmd(&mut a, r#"{"t":"removeSlider","name":"a"}"#);
        assert!(a.doc.slider_play.is_empty());
    }

    #[test]
    fn drag_mode_limits_which_coordinates_a_point_moves() {
        // (2, 3) is at pixel (480, 180); one unit is 40 px.
        let drag = |mode: Option<&str>| {
            let mut a = app();
            cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(2,3)"}"#);
            if let Some(m) = mode {
                let ev = cmd(
                    &mut a,
                    &format!(r#"{{"t":"setStyle","id":"p","style":{{"dragMode":"{m}"}}}}"#),
                );
                assert!(!ev.iter().any(|e| e["t"] == "error"), "{ev:?}");
            }
            let w0 = a.rig.window();
            ptr(&mut a, "down", 480.0, 180.0);
            ptr(&mut a, "move", 520.0, 140.0);
            ptr(&mut a, "up", 520.0, 140.0);
            (a.doc.items[0].latex.clone(), a.rig.window() != w0)
        };
        assert_eq!(drag(None), ("(3, 4)".to_string(), false));
        assert_eq!(drag(Some("xy")), ("(3, 4)".to_string(), false));
        assert_eq!(drag(Some("x")), ("(3, 3)".to_string(), false));
        assert_eq!(drag(Some("y")), ("(2, 4)".to_string(), false));
        // `none` leaves the text alone and the press pans the view instead.
        assert_eq!(drag(Some("none")), ("(2,3)".to_string(), true));
        // The style round-trips and `null` goes back to dragging both.
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(2,3)"}"#);
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"dragMode":"x"}}"#);
        assert_eq!(a.doc.items[0].style.drag_mode, Some(doc::DragMode::X));
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"dragMode":null}}"#);
        assert_eq!(a.doc.items[0].style.drag_mode, None);
        let ev = cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"dragMode":"diagonal"}}"#);
        assert!(ev.iter().any(|e| e["t"] == "error"));
    }

    #[test]
    fn drag_mode_x_on_a_slider_point_leaves_the_other_slider_alone() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":1,"min":-5,"max":5}"#);
        cmd(&mut a, r#"{"t":"setSlider","name":"b","value":2,"min":-5,"max":5}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(a,b)"}"#);
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"dragMode":"y"}}"#);
        ptr(&mut a, "down", 440.0, 220.0);
        ptr(&mut a, "move", 520.0, 140.0);
        assert_eq!(a.doc.sliders["a"].value, 1.0);
        assert_eq!(a.doc.sliders["b"].value, 4.0);
    }

    #[test]
    fn dragging_a_point_built_from_a_slider_and_a_definition() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setSlider","name":"a","value":1,"min":-5,"max":5}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"addItem","id":"d","kind":"expression","latex":"b=2"}"#,
        );
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(a,b)"}"#);
        // (1, 2): pixel (440, 220).
        ptr(&mut a, "down", 440.0, 220.0);
        let ev = ptr(&mut a, "move", 640.0, 160.0);
        let slider = ev
            .iter()
            .find(|e| e["t"] == "sliderValue")
            .expect("slider event");
        assert_eq!(slider["name"], "a");
        assert_eq!(slider["value"], 5.0, "clamped to the slider range");
        let edit = ev
            .iter()
            .find(|e| e["t"] == "itemEdited")
            .expect("definition edit");
        assert_eq!(
            (edit["id"].as_str(), edit["latex"].as_str()),
            (Some("d"), Some("b=3.5"))
        );
        assert_eq!(a.doc.items[0].latex, "b=3.5");
        assert_eq!(a.doc.sliders["a"].value, 5.0);
        assert_eq!(a.doc.items[1].latex, "(a,b)", "the tuple text is unchanged");
    }

    #[test]
    fn formula_points_and_other_modes_are_not_draggable() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(1+1, 3)"}"#);
        // The y coordinate is a literal, so the point is draggable (x stays a formula).
        ptr(&mut a, "down", 480.0, 180.0);
        let ev = ptr(&mut a, "move", 560.0, 140.0);
        let edit = ev.iter().find(|e| e["t"] == "itemEdited").unwrap();
        assert_eq!(edit["latex"], "(1 + 1, 4)");
        ptr(&mut a, "up", 560.0, 140.0);
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(1+1, 3+0)"}"#);
        ptr(&mut a, "down", 480.0, 180.0);
        let ev = ptr(&mut a, "move", 560.0, 140.0);
        assert!(!has(&ev, "itemEdited"));
        ptr(&mut a, "up", 560.0, 140.0);
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(2,3)"}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(10_000.0);
        ptr(&mut a, "down", 480.0, 180.0);
        assert!(!has(&ptr(&mut a, "move", 560.0, 140.0), "itemEdited"));
    }

    #[test]
    fn table_commands_round_trip_with_events() {
        let mut a = app();
        let ev = cmd(
            &mut a,
            r#"{"t":"addTable","id":"t","data":[[1,2],[2,"4"]]}"#,
        );
        let t = ev.iter().find(|e| e["t"] == "table").expect("table event");
        assert_eq!(t["columns"][0]["name"], "x_1");
        assert_eq!(t["columns"][1]["cells"], serde_json::json!(["2", "4"]));
        assert_eq!(
            (t["rows"].as_u64(), t["style"].as_str()),
            (Some(2), Some("points"))
        );
        assert_eq!(
            a.layers()[0]
                .geometry
                .segments
                .iter()
                .filter(|s| s.p0 == s.p1 && s.width == 12.0)
                .count(),
            2
        );
        let ev = cmd(
            &mut a,
            r#"{"t":"setCell","id":"t","row":2,"col":0,"value":3}"#,
        );
        assert_eq!(ev.iter().find(|e| e["t"] == "table").unwrap()["rows"], 3);
        cmd(
            &mut a,
            r#"{"t":"setCell","id":"t","row":2,"col":1,"value":"a"}"#,
        );
        // `a` is undefined: the table reports a diagnostic, until a slider supplies it.
        assert!(a.diagnostics.iter().any(|d| d.id == "t"));
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":9}"#);
        assert!(a.diagnostics.is_empty(), "{:?}", a.diagnostics);
        cmd(&mut a, r#"{"t":"addColumn","id":"t"}"#);
        cmd(
            &mut a,
            r#"{"t":"renameColumn","id":"t","col":2,"name":"w_1"}"#,
        );
        cmd(&mut a, r#"{"t":"addRow","id":"t","at":0}"#);
        assert_eq!(a.doc.items[0].table.as_ref().unwrap().rows(), 4);
        cmd(&mut a, r#"{"t":"removeRow","id":"t","row":0}"#);
        cmd(&mut a, r#"{"t":"removeColumn","id":"t","col":2}"#);
        let ev = cmd(&mut a, r#"{"t":"setTableStyle","id":"t","style":"line"}"#);
        assert_eq!(
            ev.iter().find(|e| e["t"] == "table").unwrap()["style"],
            "line"
        );
        cmd(&mut a, r#"{"t":"setTableStyle","id":"t","style":"hidden"}"#);
        assert_eq!(
            a.layers()[0]
                .geometry
                .segments
                .iter()
                .filter(|s| s.width == 12.0)
                .count(),
            0
        );
        // Errors leave the table untouched.
        let before = a.doc.clone();
        for bad in [
            r#"{"t":"setCell","id":"nope","row":0,"col":0,"value":1}"#,
            r#"{"t":"setCell","id":"t","row":0,"col":9,"value":1}"#,
            r#"{"t":"addColumn","id":"t","name":"height"}"#,
            r#"{"t":"addColumn","id":"t","name":"x_1"}"#,
            r#"{"t":"removeRow","id":"t","row":99}"#,
            r#"{"t":"setTableStyle","id":"t","style":"zigzag"}"#,
            r#"{"t":"addTable","id":"u","columns":["bad name"]}"#,
            r#"{"t":"addTable","id":"t"}"#,
        ] {
            assert!(has(&cmd(&mut a, bad), "error"), "{bad}");
        }
        assert_eq!(a.doc, before);
    }

    #[test]
    fn table_columns_have_their_own_styles() {
        let mut a = app();
        let dots = |a: &App, w: f32| {
            a.layers()[0]
                .geometry
                .segments
                .iter()
                .filter(|s| s.p0 == s.p1 && s.width == w)
                .count()
        };
        let ev = cmd(
            &mut a,
            r#"{"t":"addTable","id":"t","columns":["x_1","y_1","y_2"],"data":[[1,1,5],[2,2,6],[3,4,7]]}"#,
        );
        // Every y column is its own point set, each with its own palette colour.
        assert_eq!(dots(&a, 12.0), 6);
        let colors = ev.iter().find(|e| e["t"] == "colors").expect("colors event");
        let color_of = |id: &str| {
            colors["items"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == id)
                .map(|c| c["color"].as_str().unwrap().to_string())
        };
        let (c1, c2) = (color_of("t").unwrap(), color_of("t#2").unwrap());
        assert_ne!(c1, c2);
        assert_eq!(color_of("t#1"), None, "the first y column uses the item's colour");

        let ev = cmd(
            &mut a,
            r##"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"color":"#c74440","pointSize":14,"lines":true,"pointStyle":"square"}}"##,
        );
        let t = ev.iter().find(|e| e["t"] == "table").expect("table event");
        assert_eq!(t["columns"][2]["style"]["color"], "#c74440");
        assert_eq!(t["columns"][2]["style"]["lines"], true);
        assert!(t["columns"][1].get("style").is_none(), "defaults are not echoed");
        let segs = a.layers()[0].geometry.segments.clone();
        let red = |s: &&crate::geometry::SegmentInstance| (s.color[0] - 0.78).abs() < 0.01 && s.color[1] < 0.3;
        assert!(segs.iter().filter(red).any(|s| s.width == crate::scene::CURVE_W && s.p0 != s.p1), "lines between the y_2 points");
        assert!(segs.iter().filter(red).any(|s| s.p0 != s.p1 && s.width != crate::scene::CURVE_W), "square markers");
        assert_eq!(dots(&a, 12.0), 3, "y_1 keeps its dots");
        // Points off for y_1; an outline ring for y_2 (a larger darker dot under the halo).
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":1,"style":{"points":false}}"#);
        assert_eq!(dots(&a, 12.0), 0);
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"outline":true,"pointStyle":null}}"#);
        assert_eq!(dots(&a, 14.0), 3);
        assert_eq!(dots(&a, 22.0), 3);
        // Hidden draws nothing for that column; the lists stay defined.
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"hidden":true}}"#);
        assert_eq!(dots(&a, 14.0), 0);
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"hidden":null}}"#);

        // Errors change nothing.
        let before = a.doc.clone();
        for bad in [
            r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"color":"red"}}"#,
            r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"glow":true}}"#,
            r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"opacity":3}}"#,
            r#"{"t":"setTableColumnStyle","id":"t","col":9,"style":{"lines":true}}"#,
            r#"{"t":"setTableColumnStyle","id":"nope","col":1,"style":{"lines":true}}"#,
        ] {
            assert!(has(&cmd(&mut a, bad), "error"), "{bad}");
        }
        assert_eq!(a.doc, before);

        // Export and reload keep every column's style.
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let hash = ev.iter().find(|e| e["t"] == "hash").unwrap()["hash"].as_str().unwrap().to_string();
        let mut b = app();
        cmd(&mut b, &serde_json::json!({"t":"loadHash","hash":hash}).to_string());
        assert_eq!(b.doc.items[0].table, a.doc.items[0].table);
        assert_eq!(dots(&b, 14.0), 3);

        // A legacy document with the table-wide "line" style loads as lines on every column.
        let legacy = r#"{"v":1,"view":{"mode":"2d","window":{"min":[-10,-10,-10],"max":[10,10,10]},"angle":"rad"},"items":[{"id":"t","kind":"table","latex":"","table":{"columns":[{"name":"x_1","cells":["1","2"]},{"name":"y_1","cells":["1","4"]}],"style":"line"}}]}"#;
        let mut c = app();
        let ev = cmd(&mut c, &serde_json::json!({"t":"loadDoc","json":legacy}).to_string());
        let t = ev.iter().find(|e| e["t"] == "table").expect("table event on load");
        assert_eq!(t["columns"][1]["style"]["lines"], true);
        assert_eq!(t["style"], "line", "legacy summary for old shells");
        assert!(c.layers()[0].geometry.segments.iter().any(|s| s.p0 != s.p1 && s.width == crate::scene::CURVE_W && s.color[3] > 0.9 && s.color[0] > 0.7));
    }

    #[test]
    fn tables_survive_export_and_reload_and_feed_lists() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"addTable","id":"t","columns":["x_1","y_1"],"data":[[1,1],[2,2],[2,3]]}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"h","latex":"histogram(x_1)"}"#,
        );
        assert!(a.diagnostics.is_empty(), "{:?}", a.diagnostics);
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"]
            .as_str()
            .unwrap()
            .to_string();
        let hash = ev.iter().find(|e| e["t"] == "hash").unwrap()["hash"]
            .as_str()
            .unwrap()
            .to_string();
        for load in [
            serde_json::json!({"t":"loadDoc","json":json}),
            serde_json::json!({"t":"loadHash","hash":hash}),
        ] {
            let mut b = app();
            let ev = cmd(&mut b, &load.to_string());
            assert!(!has(&ev, "error"));
            let t = ev
                .iter()
                .find(|e| e["t"] == "table")
                .expect("table event on load");
            assert_eq!(t["columns"][0]["cells"], serde_json::json!(["1", "2", "2"]));
            assert!(b.diagnostics.is_empty());
            assert!(
                !b.layers()[0].geometry.vertices.is_empty(),
                "histogram from the table column"
            );
        }
    }

    #[test]
    fn vector_field_item_draws_and_follows_sliders() {
        let mut a = app();
        let base = a.layers()[0].geometry.segments.len();
        cmd(&mut a, r#"{"t":"setExpr","id":"f","latex":"(k*y,x)"}"#);
        cmd(&mut a, r#"{"t":"setSlider","name":"k","value":-1}"#);
        assert!(a.diagnostics.is_empty());
        assert!(a.layers()[0].geometry.segments.len() > base + 300);
    }

    fn ticker_app(action: &str) -> App {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"addItem","id":"d","kind":"equation","latex":"k=3"}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"setSlider","name":"a","value":0,"min":-100,"max":100}"#,
        );
        cmd(&mut a, r#"{"t":"setSlider","name":"b","value":10}"#);
        let j = serde_json::json!({"t":"addItem","id":"act","kind":"action","latex":action});
        cmd(&mut a, &j.to_string());
        a
    }

    fn slider(a: &App, n: &str) -> f64 {
        a.doc.sliders[n].value
    }

    #[test]
    fn run_action_is_simultaneous_and_emits_events() {
        let mut a = ticker_app("a -> b, b -> a, k -> k+1");
        let ev = cmd(&mut a, r#"{"t":"runAction","id":"act"}"#);
        assert!(!has(&ev, "error"), "{ev:?}");
        assert_eq!((slider(&a, "a"), slider(&a, "b")), (10.0, 0.0));
        let sv: Vec<_> = ev.iter().filter(|e| e["t"] == "sliderValue").collect();
        assert_eq!(sv.len(), 2);
        let edit = ev.iter().find(|e| e["t"] == "itemEdited").unwrap();
        assert_eq!(
            (edit["id"].as_str(), edit["latex"].as_str()),
            (Some("d"), Some("k=4"))
        );
        assert_eq!(a.doc.items[0].latex, "k=4");
    }

    #[test]
    fn run_action_errors_change_nothing() {
        let mut a = ticker_app("a -> 5, nope -> 1");
        let ev = cmd(&mut a, r#"{"t":"runAction","id":"act"}"#);
        assert!(has(&ev, "error"));
        assert_eq!(slider(&a, "a"), 0.0, "atomic");
        assert!(has(
            &cmd(&mut a, r#"{"t":"runAction","id":"zzz"}"#),
            "error"
        ));
        // The bad action also shows as an inline diagnostic.
        assert!(a.diagnostics.iter().any(|d| d.id == "act"));
    }

    #[test]
    fn equation_kind_item_with_arrow_is_an_action_not_a_parse_error() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":1}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"q","latex":"a \\to a+1"}"#);
        assert!(a.diagnostics.is_empty(), "{:?}", a.diagnostics);
        cmd(&mut a, r#"{"t":"runAction","id":"q"}"#);
        assert_eq!(slider(&a, "a"), 2.0);
    }

    #[test]
    fn ticker_steps_at_rate_and_only_while_running() {
        let mut a = ticker_app("a -> a+0.5");
        let ev = cmd(&mut a, r#"{"t":"setTicker","action":"act","rate_ms":100}"#);
        let st = ev.iter().find(|e| e["t"] == "tickerState").unwrap();
        assert_eq!(
            (st["running"].as_bool(), st["action"].as_str()),
            (Some(false), Some("act"))
        );
        for t in [0.0, 100.0, 200.0] {
            a.frame(t);
        }
        assert_eq!(slider(&a, "a"), 0.0, "not running");
        let ev = cmd(&mut a, r#"{"t":"startTicker"}"#);
        assert_eq!(
            ev.iter().find(|e| e["t"] == "tickerState").unwrap()["running"],
            true
        );
        a.frame(300.0); // anchors
        a.frame(350.0); // 50 ms: not due
        assert_eq!(slider(&a, "a"), 0.0);
        a.frame(400.0);
        assert_eq!(slider(&a, "a"), 0.5);
        a.frame(520.0); // 120 ms since 400 -> 1 step, 20 ms carried
        assert_eq!(slider(&a, "a"), 1.0);
        a.frame(600.0); // carried 20 + 80 = 100 -> 1 step
        assert_eq!(slider(&a, "a"), 1.5);
        let ev = cmd(&mut a, r#"{"t":"toggleTicker"}"#);
        assert_eq!(
            ev.iter().find(|e| e["t"] == "tickerState").unwrap()["running"],
            false
        );
        a.frame(900.0);
        assert_eq!(slider(&a, "a"), 1.5);
        // one manual step works while stopped
        cmd(&mut a, r#"{"t":"tickerStep"}"#);
        assert_eq!(slider(&a, "a"), 2.0);
    }

    #[test]
    fn ticker_frame_emits_slider_events_and_rebuilds() {
        let mut a = ticker_app("a -> a+1");
        cmd(&mut a, r#"{"t":"setTicker","action":"act","running":true}"#);
        a.frame(0.0);
        a.take_events();
        assert!(a.frame(50.0));
        let ev = a.take_events();
        assert!(ev.iter().any(
            |e| matches!(e, Event::SliderValue { name, value } if name == "a" && *value == 1.0)
        ));
    }

    #[test]
    fn ticker_caps_steps_per_frame_and_drops_backlog() {
        let mut a = ticker_app("a -> a+1");
        cmd(
            &mut a,
            r#"{"t":"setTicker","action":"act","rate_ms":50,"running":true}"#,
        );
        a.frame(0.0);
        a.frame(10_000.0); // 200 steps due, only 4 fire
        assert_eq!(slider(&a, "a"), MAX_TICKER_STEPS_PER_FRAME as f64);
        a.frame(10_010.0); // backlog was dropped, not carried
        assert_eq!(slider(&a, "a"), 4.0);
        a.frame(10_050.0);
        assert_eq!(slider(&a, "a"), 5.0);
        // exactly the cap is kept (no spurious drop)
        a.frame(10_250.0);
        assert_eq!(slider(&a, "a"), 9.0);
    }

    #[test]
    fn ticker_min_step_floors_the_rate() {
        let mut a = ticker_app("a -> a+1");
        cmd(
            &mut a,
            r#"{"t":"setTicker","action":"act","rate_ms":1,"running":true}"#,
        );
        a.frame(0.0);
        a.frame(5.0);
        assert_eq!(slider(&a, "a"), 0.0, "default floor is 10 ms");
        a.frame(10.0);
        assert_eq!(slider(&a, "a"), 1.0);
        cmd(
            &mut a,
            r#"{"t":"setTicker","rate_ms":10,"min_step_ms":200}"#,
        );
        a.frame(100.0);
        a.frame(200.0);
        assert_eq!(slider(&a, "a"), 1.0);
        a.frame(209.0);
        assert_eq!(slider(&a, "a"), 1.0, "200 ms floor beats rate_ms 10");
        a.frame(210.0);
        assert_eq!(slider(&a, "a"), 2.0);
    }

    #[test]
    fn ticker_pauses_on_error_unless_disabled() {
        let mut a = ticker_app("a -> a+1, b -> 0/0");
        cmd(&mut a, r#"{"t":"setTicker","action":"act","running":true}"#);
        a.frame(0.0);
        a.take_events();
        a.frame(50.0);
        let ev = a.take_events();
        assert!(ev.iter().any(|e| matches!(e, Event::Error { .. })));
        assert!(ev
            .iter()
            .any(|e| matches!(e, Event::TickerState { running: false, .. })));
        assert_eq!(slider(&a, "a"), 0.0, "failed step changes nothing");
        // Keep running, report the error once.
        let ev = cmd(
            &mut a,
            r#"{"t":"setTicker","pause_on_error":false,"running":true}"#,
        );
        assert!(ev
            .iter()
            .any(|e| e["t"] == "tickerState" && e["running"] == true));
        a.frame(100.0);
        a.take_events();
        a.frame(150.0);
        assert!(
            a.take_events()
                .iter()
                .any(|e| matches!(e, Event::Error { .. })),
            "reported once"
        );
        a.frame(200.0);
        a.frame(250.0);
        let ev = a.take_events();
        assert!(
            !ev.iter().any(|e| matches!(e, Event::Error { .. })),
            "error not repeated"
        );
        assert!(a.ticker_running);
    }

    #[test]
    fn ticker_config_validation_and_cleanup() {
        let mut a = ticker_app("a -> a+1");
        assert!(has(
            &cmd(&mut a, r#"{"t":"setTicker","action":"nope"}"#),
            "error"
        ));
        assert!(
            has(&cmd(&mut a, r#"{"t":"setTicker","action":"d"}"#), "error"),
            "not an action"
        );
        assert!(has(
            &cmd(&mut a, r#"{"t":"setTicker","rate_ms":-5}"#),
            "error"
        ));
        assert!(
            has(&cmd(&mut a, r#"{"t":"startTicker"}"#), "error"),
            "no action yet"
        );
        cmd(&mut a, r#"{"t":"setTicker","action":"act","running":true}"#);
        assert!(a.ticker_running);
        // missing action leaves it; explicit null clears and stops
        cmd(&mut a, r#"{"t":"setTicker","rate_ms":80}"#);
        assert!(a.ticker_running);
        let ev = cmd(&mut a, r#"{"t":"setTicker","action":null}"#);
        assert!(!a.ticker_running);
        assert_eq!(
            ev.iter().find(|e| e["t"] == "tickerState").unwrap()["action"],
            serde_json::Value::Null
        );
        // removing the action item stops the ticker
        cmd(&mut a, r#"{"t":"setTicker","action":"act","running":true}"#);
        let ev = cmd(&mut a, r#"{"t":"removeItem","id":"act"}"#);
        assert!(!a.ticker_running);
        assert!(ev
            .iter()
            .any(|e| e["t"] == "tickerState" && e["running"] == false));
    }

    #[test]
    fn ticker_config_is_saved_but_not_running_state() {
        let mut a = ticker_app("a -> a+1");
        cmd(
            &mut a,
            r#"{"t":"setTicker","action":"act","rate_ms":75,"min_step_ms":20,"pause_on_error":false,"running":true}"#,
        );
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(json.contains("\"ticker\""));
        let mut b = app();
        let load = serde_json::json!({"t":"loadDoc","json":json});
        let ev = cmd(&mut b, &load.to_string());
        assert!(!has(&ev, "error"));
        let st = ev.iter().find(|e| e["t"] == "tickerState").unwrap();
        assert_eq!(
            (st["running"].as_bool(), st["action"].as_str()),
            (Some(false), Some("act"))
        );
        let t = b.doc.ticker.clone().unwrap();
        assert_eq!(
            (t.rate_ms, t.min_step_ms, t.pause_on_error),
            (75.0, 20.0, false)
        );
        // Starting the loaded ticker works.
        cmd(&mut b, r#"{"t":"startTicker"}"#);
        b.frame(0.0);
        b.frame(80.0);
        assert_eq!(b.doc.sliders["a"].value, 1.0);
    }

    // ---- slices ------------------------------------------------------------------------------

    fn slice_ev(ev: &[serde_json::Value]) -> Option<&serde_json::Value> {
        ev.iter().rev().find(|e| e["t"] == "slice")
    }

    #[test]
    fn hiding_the_inset_keeps_the_slice() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"s","latex":"x^2+y^2+z^2=4"}"#);
        cmd(&mut a, r#"{"t":"setSlice","dim":2,"fixed":{"z":1}}"#);
        a.frame(0.0);
        a.frame(10_000.0);
        assert!(a.inset().is_some());
        cmd(&mut a, r#"{"t":"setSliceInset","show":false}"#);
        assert!(a.inset().is_none() && a.inset_rect().is_none());
        assert!(a.inset_screen_labels().is_empty());
        assert!(a.doc.slice.is_some(), "the slice plane stays in the 3D scene");
        cmd(&mut a, r#"{"t":"setSliceInset","show":true}"#);
        assert!(a.inset().is_some());
    }

    #[test]
    fn set_slice_3d_plane_reports_state_and_shows_inset() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"s","latex":"x^2+y^2+z^2=4"}"#,
        );
        assert!(a.inset().is_none());
        let ev = cmd(&mut a, r#"{"t":"setSlice","dim":2,"fixed":{"z":1}}"#);
        let s = slice_ev(&ev).expect("slice event");
        assert_eq!(s["active"], true);
        assert_eq!(s["dim"], 2);
        assert_eq!(s["fixed"]["z"], 1.0);
        assert_eq!(s["free"], serde_json::json!(["x", "y"]));
        assert_eq!(s["curves"], 1);
        let r = s["rect"].as_array().unwrap();
        assert_eq!(r.len(), 4);
        assert!(a.doc.slice.is_some());
        // Mid mode-switch the camera is animating; settle it, then the inset is available.
        a.frame(0.0);
        a.frame(10_000.0);
        let inset = a.inset().expect("inset");
        assert_eq!(inset.rect[2] as usize, r[2].as_u64().unwrap() as usize);
        assert_eq!(inset.layers.len(), 1);
        // Inset labels are part of screen_labels, flagged.
        let labels = a.screen_labels();
        assert!(labels.iter().any(|l| l.inset && l.text == "x"));
        assert!(labels.iter().any(|l| !l.inset));
        // The main scene carries the overlay.
        assert!(!a
            .layers()
            .last()
            .unwrap()
            .geometry
            .overlay_segments
            .is_empty());
    }

    #[test]
    fn slider_sweeps_the_slice() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlider","name":"a","value":1}"#);
        let ev = cmd(&mut a, r#"{"t":"setSlice","fixed":"y=a"}"#);
        let s = slice_ev(&ev).unwrap();
        assert_eq!(
            (s["dim"].as_u64(), s["points"].as_u64()),
            (Some(1), Some(2))
        );
        assert_eq!(s["fixed"]["y"], 1.0);
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"a","value":-3}"#);
        let s = slice_ev(&ev).expect("slice event follows the slider");
        assert_eq!(s["fixed"]["y"], -3.0);
        assert_eq!(s["points"], 0);
        // Unchanged state is not repeated.
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"a","value":-3}"#);
        assert!(slice_ev(&ev).is_none());
    }

    #[test]
    fn clear_slice_removes_inset_and_reports_inactive() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":{"y":2}}"#);
        assert!(a.inset().is_some());
        let ev = cmd(&mut a, r#"{"t":"clearSlice"}"#);
        assert_eq!(slice_ev(&ev).unwrap()["active"], false);
        assert!(a.inset().is_none() && a.doc.slice.is_none());
        assert!(a.screen_labels().iter().all(|l| !l.inset));
    }

    #[test]
    fn bad_slice_commands_are_errors() {
        let mut a = app();
        for bad in [
            r#"{"t":"setSlice","fixed":{"w":1}}"#,
            r#"{"t":"setSlice","fixed":{}}"#,
            r#"{"t":"setSlice","fixed":{"z":1}}"#, // z does not exist in 2D
            r#"{"t":"setSlice","fixed":{"x":1,"y":2}}"#, // would leave 0 dims
            r#"{"t":"setSlice","dim":2,"fixed":{"y":1}}"#, // dim disagrees (2D + 1 fixed = 1D)
            r#"{"t":"setSlice","fixed":3}"#,
            r#"{"t":"setSlice"}"#,
        ] {
            let ev = cmd(&mut a, bad);
            assert!(has(&ev, "error"), "{bad}");
        }
        assert!(a.doc.slice.is_none());
        // A slice that no longer fits after a mode switch is reported, not drawn.
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"z=0.5"}"#);
        a.frame(0.0);
        a.frame(10_000.0);
        assert!(a.inset().is_some());
        let ev = cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        let s = slice_ev(&ev).unwrap();
        assert_eq!(s["active"], false);
        assert!(s["error"].as_str().is_some());
    }

    fn settle(a: &mut App) {
        let t = a.now_ms;
        a.frame(t + 1.0);
        a.frame(t + 100_000.0);
        a.frame(t + 200_000.0);
    }

    #[test]
    fn slice_follows_mode_round_trips_with_one_report_each() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"s","latex":"x^2+y^2+z^2=4"}"#,
        );
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1,z=0.5"}"#);
        settle(&mut a);
        assert_eq!(
            a.panel.as_ref().map(|p| p.axes[1].clone()),
            Some("f".into()),
            "line slice in 3D"
        );
        let seq = [
            ("2d", false),
            ("3d", true),
            ("1d", false),
            ("2d", false),
            ("3d", true),
        ];
        for (m, active) in seq {
            let mut ev = cmd(&mut a, &format!(r#"{{"t":"setMode","mode":"{m}"}}"#));
            settle(&mut a);
            ev.extend(
                a.take_events()
                    .into_iter()
                    .map(|e| serde_json::to_value(e).unwrap()),
            );
            let slices: Vec<_> = ev.iter().filter(|e| e["t"] == "slice").collect();
            assert_eq!(
                slices.len(),
                1,
                "{m}: exactly one slice report, got {slices:?}"
            );
            assert_eq!(slices[0]["active"], active, "{m}");
            assert!(!ev.iter().any(|e| e["t"] == "error"), "{m}: no error event");
            if !active {
                let msg = slices[0]["error"].as_str().unwrap();
                assert!(
                    msg.starts_with("Slice paused") && !msg.contains("does not fit"),
                    "{msg}"
                );
                assert!(a.doc.slice.is_some(), "config kept while paused");
                assert!(a.inset().is_none());
            } else {
                assert_eq!(slices[0]["dim"], 1);
                assert!(a.inset().is_some());
            }
        }
        // A y-only slice is a plane in 3D, a line in 2D, back to a plane: dim re-derived.
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        for (m, dim) in [("2d", 1), ("3d", 2), ("2d", 1)] {
            cmd(&mut a, &format!(r#"{{"t":"setMode","mode":"{m}"}}"#));
            settle(&mut a);
            let a_ev = a.last_slice.clone().unwrap();
            match a_ev {
                Event::Slice {
                    active,
                    dim: d,
                    error,
                    ..
                } => assert!(active && d == dim && error.is_none(), "{m}"),
                _ => unreachable!(),
            }
        }
    }

    /// Every slice event in `evs` must describe `fixed` (the config just set) when it is active.
    fn assert_slices_match(evs: &[serde_json::Value], fixed: &[(&str, f64)], ctx: &str) {
        for e in evs
            .iter()
            .filter(|e| e["t"] == "slice" && e["active"] == true)
        {
            let got: Vec<(String, f64)> = e["fixed"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect();
            let want: Vec<(String, f64)> = fixed.iter().map(|(k, v)| (k.to_string(), *v)).collect();
            assert_eq!(got, want, "{ctx}: stale slice event {e}");
        }
    }

    #[test]
    fn no_stale_slice_event_after_mode_round_trips() {
        let mut a = app();
        let mut all: Vec<serde_json::Value> = Vec::new();
        let step = |a: &mut App, j: &str, all: &mut Vec<serde_json::Value>| {
            let mut ev = cmd(a, j);
            ev.extend(serde_json::from_str::<Vec<serde_json::Value>>(&a.drain_json()).unwrap());
            all.extend(ev.clone());
            ev
        };
        step(&mut a, r#"{"t":"setMode","mode":"3d"}"#, &mut all);
        step(
            &mut a,
            r#"{"t":"setExpr","id":"s","latex":"x^2+y^2+z^2=4"}"#,
            &mut all,
        );
        step(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#, &mut all);
        step(&mut a, r#"{"t":"setMode","mode":"2d"}"#, &mut all);
        let ev = step(&mut a, r#"{"t":"setSlice","fixed":"x=2"}"#, &mut all);
        assert_slices_match(&ev, &[("x", 2.0)], "2d x=2");
        step(&mut a, r#"{"t":"setMode","mode":"3d"}"#, &mut all);
        step(&mut a, r#"{"t":"setMode","mode":"1d"}"#, &mut all);
        step(&mut a, r#"{"t":"setMode","mode":"2d"}"#, &mut all);
        step(&mut a, r#"{"t":"setMode","mode":"3d"}"#, &mut all);
        let mark = all.len();
        let ev = step(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#, &mut all);
        assert_slices_match(&ev, &[("z", 1.0)], "z=1 restored");
        // Late events from frame() / drain after the config was set.
        for t in [1.0, 100_000.0, 200_000.0, 300_000.0] {
            let n = a.now_ms;
            a.frame(n + t);
            let late: Vec<serde_json::Value> = serde_json::from_str(&a.drain_json()).unwrap();
            assert_slices_match(&late, &[("z", 1.0)], "late frame");
            all.extend(late);
        }
        assert!(all.len() > mark);
        // A pending (undrained) frame()-raised report must not survive a later config change.
        let (x, y) = inset_center(&a);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        cmd(
            &mut a,
            &format!(
                r#"{{"t":"pointer","phase":"move","x":{},"y":{y}}}"#,
                x + 15.0
            ),
        );
        let n = a.now_ms;
        a.frame(n + 400_000.0);
        assert!(
            a.outbox.iter().any(|e| matches!(e, Event::Slice { .. })),
            "pending frame report"
        );
        let ev = cmd(&mut a, r#"{"t":"setSlice","fixed":"x=2"}"#);
        assert_eq!(ev.iter().filter(|e| e["t"] == "slice").count(), 1, "{ev:?}");
        assert_slices_match(&ev, &[("x", 2.0)], "x=2 after pending");
        cmd(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#);
        settle(&mut a);
        let last = all.iter().rev().find(|e| e["t"] == "slice").unwrap();
        assert_eq!(last["active"], true, "{last}");
        assert_eq!(last["fixed"], serde_json::json!({"z": 1.0}), "{last}");
        match a.last_slice.clone().unwrap() {
            Event::Slice { fixed, active, .. } => {
                assert!(active && fixed.keys().collect::<Vec<_>>() == ["z"])
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn drain_returns_events_raised_in_frame_and_dispatch_still_includes_them() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"s","latex":"x^2+y^2+z^2=4"}"#,
        );
        cmd(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#);
        settle(&mut a);
        a.take_events();
        // A throttled inset re-sample runs inside frame(): its slice event is only pending.
        let (x, y) = inset_center(&a);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        let ev = cmd(
            &mut a,
            &format!(
                r#"{{"t":"pointer","phase":"move","x":{},"y":{y}}}"#,
                x - 60.0
            ),
        );
        assert!(ev.iter().all(|e| e["t"] != "slice"));
        settle(&mut a);
        // Frame-time events are pending, and a drain returns them once.
        let v: Vec<serde_json::Value> = serde_json::from_str(&a.drain_json()).unwrap();
        assert!(v.iter().any(|e| e["t"] == "slice"), "{v:?}");
        assert_eq!(a.drain_json(), "[]");
        // Without a drain, dispatch carries them.
        cmd(
            &mut a,
            &format!(
                r#"{{"t":"pointer","phase":"move","x":{},"y":{y}}}"#,
                x - 120.0
            ),
        );
        settle(&mut a);
        let ev = cmd(&mut a, r#"{"t":"setSlider","name":"q","value":1}"#);
        assert!(
            ev.iter().any(|e| e["t"] == "slice"),
            "pending frame events ride along on dispatch"
        );
    }

    #[test]
    fn inset_axis_names_never_overlap_tick_labels() {
        use crate::scene::label_box;
        let check = |a: &App, what: &str| {
            let r = a.inset().expect(what).rect;
            let labs: Vec<_> = a
                .inset_screen_labels()
                .into_iter()
                .filter(|l| l.visible)
                .collect();
            let boxes: Vec<(String, [f64; 4])> = labs
                .iter()
                .map(|l| {
                    (
                        l.text.clone(),
                        label_box(l.axis, &l.text, l.x - r[0] as f64, l.y - r[1] as f64),
                    )
                })
                .collect();
            let names = ["x", "y", "z", "f"];
            assert!(
                boxes.iter().any(|(t, _)| names.contains(&t.as_str())),
                "{what}: axis names present"
            );
            for (i, (ta, ba)) in boxes.iter().enumerate() {
                for (tb, bb) in boxes.iter().skip(i + 1) {
                    if !names.contains(&ta.as_str()) && !names.contains(&tb.as_str()) {
                        continue; // tick vs tick is not what this checks
                    }
                    let overlap = ba[0] < bb[2] && bb[0] < ba[2] && ba[1] < bb[3] && bb[1] < ba[3];
                    assert!(
                        !overlap,
                        "{what}: labels {ta:?} {ba:?} and {tb:?} {bb:?} overlap"
                    );
                }
            }
        };
        for (what, setup) in [
            ("1d line", r#"{"t":"setSlice","fixed":"y=1,z=0.5"}"#),
            ("plane", r#"{"t":"setSlice","fixed":"z=1"}"#),
        ] {
            let mut a = app();
            cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
            cmd(&mut a, r#"{"t":"setExpr","id":"s","latex":"y=x^2"}"#);
            cmd(&mut a, setup);
            settle(&mut a);
            check(&a, what);
            // Panned and zoomed insets, including the value axis of the line slice sitting at an edge.
            for view in [
                "[-2,6,-1,3]",
                "[0.5,4.5,-3,0.2]",
                "[-9,-1,2,6]",
                "[-0.5,0.5,-0.2,0.2]",
                "[-4,4,-40,0.1]",
            ] {
                cmd(&mut a, &format!(r#"{{"t":"setSliceView","view":{view}}}"#));
                settle(&mut a);
                check(&a, &format!("{what} panned {view}"));
            }
        }
        // 2D line slice too.
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"s","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        settle(&mut a);
        check(&a, "2d line");
        cmd(&mut a, r#"{"t":"setSliceView","view":[-3,3,-2,0.5]}"#);
        settle(&mut a);
        check(&a, "2d line panned");
    }

    #[test]
    fn slice_survives_document_roundtrip_and_old_docs_still_load() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(json.contains("\"slice\""));
        let mut b = app();
        cmd(
            &mut b,
            &serde_json::json!({"t":"loadDoc","json":json}).to_string(),
        );
        assert_eq!(b.doc.slice, a.doc.slice);
        assert!(b.inset().is_some());
        // A v1 document written before slices existed has no `slice` key.
        let old = r#"{"v":1,"view":{"mode":"2d","window":{"min":[-10,-10,-10],"max":[10,10,10]},"angle":"rad"},"items":[]}"#;
        let ev = cmd(
            &mut b,
            &serde_json::json!({"t":"loadDoc","json":old}).to_string(),
        );
        assert!(!has(&ev, "error"), "{ev:?}");
        assert!(b.doc.slice.is_none() && b.inset().is_none());
    }

    #[test]
    fn pointer_on_the_inset_does_not_move_the_scene() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        let r = a.inset().unwrap().rect;
        let (x, y) = ((r[0] + r[2] / 2) as f64, (r[1] + r[3] / 2) as f64);
        let before = a.rig.window();
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        cmd(
            &mut a,
            &format!(
                r#"{{"t":"pointer","phase":"move","x":{},"y":{y}}}"#,
                x - 40.0
            ),
        );
        cmd(
            &mut a,
            &format!(r#"{{"t":"wheel","x":{x},"y":{y},"dy":-300}}"#),
        );
        assert_eq!(a.rig.window(), before);
        // Elsewhere it still pans.
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":500,"y":400}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":460,"y":400}"#);
        assert_ne!(a.rig.window(), before);
    }

    // ---- inset view, labels, regression constants -------------------------------------------

    fn inset_center(a: &App) -> (f64, f64) {
        let r = a.inset().unwrap().rect;
        ((r[0] + r[2] / 2) as f64, (r[1] + r[3] / 2) as f64)
    }

    #[test]
    fn inset_pan_zoom_reset_leave_the_main_scene_alone() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        let (x, y) = inset_center(&a);
        let main = a.rig.window();
        let s = slice_ev(&cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#)).cloned();
        assert!(s.is_none(), "an unchanged slice is not repeated");
        let v0 = a.panel.as_ref().unwrap().view;
        assert!(a.panel.as_ref().unwrap().follow);
        // Drag the inset left by 60 px: the view window moves right (content follows the pointer).
        a.frame(1000.0);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        let ev = cmd(
            &mut a,
            &format!(
                r#"{{"t":"pointer","phase":"move","x":{},"y":{y}}}"#,
                x - 60.0
            ),
        );
        assert!(
            ev.iter().all(|e| e["t"] != "slice"),
            "re-sample is throttled to frame()"
        );
        a.frame(1200.0);
        let ev = a.take_events();
        let s = ev.iter().rev().find_map(|e| match e {
            Event::Slice { view, follow, .. } => Some((*view, *follow)),
            _ => None,
        });
        let (view, follow) = s.expect("slice event after the drag");
        assert!(!follow);
        let v1 = view.unwrap();
        assert!(
            v1[0] > v0[0] + 0.1 && (v1[1] - v1[0] - (v0[1] - v0[0])).abs() < 1e-9,
            "{v0:?} -> {v1:?}"
        );
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":0,"y":0}"#);
        assert_eq!(a.rig.window(), main, "the main scene did not move");
        // Wheel zoom in at the centre halves-ish the horizontal span.
        cmd(
            &mut a,
            &format!(r#"{{"t":"wheel","x":{x},"y":{y},"dy":-400}}"#),
        );
        a.frame(2000.0);
        let v2 = a.panel.as_ref().unwrap().view;
        assert!(v2[1] - v2[0] < 0.6 * (v1[1] - v1[0]), "{v1:?} -> {v2:?}");
        assert_eq!(a.rig.window(), main);
        // The main window moving does not move an explicit inset view.
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":500,"y":400}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":460,"y":400}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":460,"y":400}"#);
        a.frame(3000.0);
        assert_eq!(a.panel.as_ref().unwrap().view, v2);
        // Double-click returns to following.
        a.frame(4000.0);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"up","x":{x},"y":{y}}}"#),
        );
        a.frame(4100.0);
        cmd(
            &mut a,
            &format!(r#"{{"t":"pointer","phase":"down","x":{x},"y":{y}}}"#),
        );
        assert!(a.panel.as_ref().unwrap().follow);
        // resetSliceView does the same, and setSliceView sets an exact window.
        let ev = cmd(&mut a, r#"{"t":"setSliceView","min":[-1,-2],"max":[3,6]}"#);
        let s = slice_ev(&ev).expect("slice event");
        assert_eq!(s["follow"], false);
        assert_eq!(s["view"], serde_json::json!([-1.0, 3.0, -2.0, 6.0]));
        let ev = cmd(&mut a, r#"{"t":"resetSliceView"}"#);
        assert_eq!(slice_ev(&ev).unwrap()["follow"], true);
        // Bad views are errors.
        for bad in [
            r#"{"t":"setSliceView","min":[1,0],"max":[0,1]}"#,
            r#"{"t":"setSliceView"}"#,
        ] {
            assert!(has(&cmd(&mut a, bad), "error"), "{bad}");
        }
        cmd(&mut a, r#"{"t":"clearSlice"}"#);
        assert!(has(
            &cmd(&mut a, r#"{"t":"setSliceView","view":[0,1,0,1]}"#),
            "error"
        ));
    }

    #[test]
    fn plane_inset_view_keeps_its_aspect_and_follows_until_changed() {
        let mut a = App::new((900, 600));
        a.max_frame_dt_ms = f64::INFINITY;
        a.rig.max_frame_dt_ms = f64::INFINITY;
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(0.0);
        a.frame(2000.0);
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"p","latex":"x^2+y^2+z^2=4"}"#,
        );
        cmd(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#);
        let r = a.inset().unwrap().rect;
        let ev = cmd(&mut a, r#"{"t":"setSliceView","view":[-1,1,-100,100]}"#);
        let v: Vec<f64> = slice_ev(&ev).unwrap()["view"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!((v[0], v[1]), (-1.0, 1.0));
        let aspect = r[2] as f64 / r[3] as f64;
        assert!(
            ((v[1] - v[0]) / (v[3] - v[2]) - aspect).abs() < 1e-9,
            "vertical span refit to the aspect: {v:?}"
        );
        // A new slice through other axes goes back to following.
        let ev = cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        assert_eq!(slice_ev(&ev).unwrap()["follow"], true);
    }

    #[test]
    fn inset_labels_stay_inside_and_apart_at_a_larger_text_scale() {
        use crate::scene::{label_box_inside_s, label_box_s};
        for ts in [1.0, 1.3, 1.6] {
            let mut a = app();
            cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
            cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
            cmd(&mut a, &format!(r#"{{"t":"setView","textScale":{ts}}}"#));
            let r = a.inset().expect("an inset").rect;
            let labs: Vec<_> = a.inset_screen_labels().into_iter().filter(|l| l.visible).collect();
            assert!(!labs.is_empty(), "scale {ts}: no inset labels");
            let boxes: Vec<[f64; 4]> = labs
                .iter()
                .map(|l| {
                    let (x, y) = (l.x - r[0] as f64, l.y - r[1] as f64);
                    assert!(
                        label_box_inside_s(l.axis, &l.text, x, y, r[2] as f64, r[3] as f64, ts),
                        "scale {ts}: {:?} leaves the inset",
                        l.text
                    );
                    label_box_s(l.axis, &l.text, x, y, ts)
                })
                .collect();
            for i in 0..boxes.len() {
                for j in i + 1..boxes.len() {
                    let (p, q) = (boxes[i], boxes[j]);
                    let apart = p[2] <= q[0] || q[2] <= p[0] || p[3] <= q[1] || q[3] <= p[1];
                    assert!(apart, "scale {ts}: {:?} overlaps {:?}", labs[i].text, labs[j].text);
                }
            }
        }
    }

    #[test]
    fn inset_labels_stay_inside_the_rect_and_follow_the_view() {
        use crate::scene::label_box_inside;
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"y=1"}"#);
        let check = |a: &App, what: &str| -> usize {
            let r = a.inset().unwrap().rect;
            let labels: Vec<_> = a
                .inset_screen_labels()
                .into_iter()
                .filter(|l| l.visible)
                .collect();
            assert!(!labels.is_empty(), "{what}: no visible inset labels");
            for l in &labels {
                let (x, y) = (l.x - r[0] as f64, l.y - r[1] as f64);
                assert!(
                    l.inset && l.clip == Some([r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64])
                );
                assert!(
                    label_box_inside(l.axis, &l.text, x, y, r[2] as f64, r[3] as f64),
                    "{what}: label {:?} at ({x:.1},{y:.1}) leaves {}x{}",
                    l.text,
                    r[2],
                    r[3]
                );
            }
            // Hidden labels are exactly the ones that would leave the rectangle.
            for l in a.inset_screen_labels().iter().filter(|l| !l.visible) {
                let (x, y) = (l.x - r[0] as f64, l.y - r[1] as f64);
                assert!(
                    !label_box_inside(l.axis, &l.text, x, y, r[2] as f64, r[3] as f64)
                        || x < 0.0
                        || y < 0.0
                        || x > r[2] as f64
                        || y > r[3] as f64
                );
            }
            labels.len()
        };
        let n0 = check(&a, "follow");
        assert!(n0 >= 4);
        // Axis names are among them.
        assert!(a
            .inset_screen_labels()
            .iter()
            .any(|l| l.visible && l.text == "x"));
        assert!(a
            .inset_screen_labels()
            .iter()
            .any(|l| l.visible && l.text == "f"));
        // Zoomed far into a corner (the zero axes are off screen): tick labels are re-laid out.
        cmd(&mut a, r#"{"t":"setSliceView","view":[40,60,300,500]}"#);
        check(&a, "far from the origin");
        let ticks: Vec<String> = a
            .inset_screen_labels()
            .into_iter()
            .filter(|l| l.visible && l.text != "x" && l.text != "f")
            .map(|l| l.text)
            .collect();
        assert!(ticks.iter().any(|t| t == "50"), "{ticks:?}");
        cmd(
            &mut a,
            r#"{"t":"setSliceView","view":[-0.001,0.001,-0.0004,0.0004]}"#,
        );
        check(&a, "tiny window");
        // A plane slice too.
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(0.0);
        a.frame(2000.0);
        cmd(&mut a, r#"{"t":"setSlice","fixed":"z=1"}"#);
        check(&a, "plane");
        cmd(&mut a, r#"{"t":"setSliceView","view":[1,5,1,4]}"#);
        check(&a, "plane panned");
    }

    #[test]
    fn slice_constant_uses_regression_parameters_and_tracks_refits() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"addTable","id":"t","columns":["x_1","y_1"],"data":[[1,3],[2,5],[3,7],[4,9]]}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"r","latex":"y_1 ~ a x_1 + b"}"#,
        );
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"y=x^2"}"#);
        let ev = cmd(&mut a, r#"{"t":"setSlice","fixed":"y=a+b"}"#);
        let s = slice_ev(&ev).unwrap();
        assert_eq!(s["active"], true, "{s}");
        assert!(
            (s["fixed"]["y"].as_f64().unwrap() - 3.0).abs() < 1e-6,
            "{s}"
        );
        assert_eq!(s["points"], 2, "x^2 = 3 has two roots");
        // Refit: the data now lies on y = 3x + 1, so a + b = 4.
        let mut last = None;
        for (row, y) in [(0, 4), (1, 7), (2, 10), (3, 13)] {
            let ev = cmd(
                &mut a,
                &format!(r#"{{"t":"setCell","id":"t","row":{row},"col":1,"value":{y}}}"#),
            );
            if let Some(s) = slice_ev(&ev) {
                last = Some(s.clone());
            }
        }
        let s = last.expect("slice event follows the refit");
        assert!(
            (s["fixed"]["y"].as_f64().unwrap() - 4.0).abs() < 1e-6,
            "{s}"
        );
        // Hiding the regression removes the parameters: the slice reports an error, no panic.
        let ev = cmd(&mut a, r#"{"t":"setHidden","id":"r","hidden":true}"#);
        let s = slice_ev(&ev).unwrap();
        assert_eq!(s["active"], false);
        assert!(s["error"].is_string());
    }

    fn style_of(a: &App, id: &str) -> doc::ItemStyle {
        a.doc
            .items
            .iter()
            .find(|i| i.id == id)
            .unwrap()
            .style
            .clone()
    }

    fn error_msg(ev: &[serde_json::Value]) -> Option<String> {
        ev.iter()
            .find(|e| e["t"] == "error")
            .map(|e| e["message"].as_str().unwrap_or("").to_string())
    }

    #[test]
    fn label_offset_is_set_clamped_cleared_and_reported_on_the_screen_label() {
        let mut a = App::new((800, 600));
        cmd(&mut a, r#"{"t":"addItem","id":"p","kind":"points","latex":"(1,2)"}"#);
        let item_label = |a: &App| {
            a.screen_labels()
                .into_iter()
                .find(|l| l.axis == crate::scene::ITEM_LABEL_AXIS || l.axis == crate::scene::ITEM_LABEL_LEFT_AXIS)
        };
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"showLabel":true}}"#);
        let base = item_label(&a).unwrap();
        assert_eq!((base.item.as_deref(), base.offset), (Some("p"), None));
        // Set: the offset rides on the label, the anchor does not move.
        let ev = cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"labelOffset":[30,-12.5]}}"#);
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        assert_eq!(style_of(&a, "p").label_offset, Some([30.0, -12.5]));
        let l = item_label(&a).unwrap();
        assert_eq!((l.offset, l.x, l.y), (Some([30.0, -12.5]), base.x, base.y));
        // Beyond the limit: clamped.
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"labelOffset":[900,-1e9]}}"#);
        assert_eq!(style_of(&a, "p").label_offset, Some([400.0, -400.0]));
        assert_eq!(item_label(&a).unwrap().offset, Some([400.0, -400.0]));
        // Saved in the document.
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"].as_str().unwrap().to_string();
        assert!(json.contains("\"labelOffset\":[400.0,-400.0]"), "{json}");
        // Updates when the point moves: the label keeps its item and offset.
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(3,4)"}"#);
        let l = item_label(&a).unwrap();
        assert_eq!((l.text.as_str(), l.offset), ("(3, 4)", Some([400.0, -400.0])));
        // null clears it.
        cmd(&mut a, r#"{"t":"setStyle","id":"p","style":{"labelOffset":null}}"#);
        assert!(style_of(&a, "p").label_offset.is_none());
        assert_eq!(item_label(&a).unwrap().offset, None);
    }

    #[test]
    fn set_style_merges_resets_and_validates() {
        let mut a = app();
        cmd(&mut a, r#"{"t":"setExpr","id":"p","latex":"(1,2)"}"#);
        let n0 = a.layers()[0].geometry.segments.len();
        let ev = cmd(
            &mut a,
            r#"{"t":"setStyle","id":"p","style":{"pointStyle":"circle","pointSize":14,"showLabel":true,"label":"P"}}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        assert!(has(&ev, "labels"), "rebuilt");
        let st = style_of(&a, "p");
        assert_eq!(st.point_style, Some(doc::PointStyle::Circle));
        assert_eq!((st.point_size, st.show_label), (Some(14.0), true));
        assert!(a.layers()[0].geometry.segments.len() > n0, "ring drawn");
        assert!(a
            .screen_labels()
            .iter()
            .any(|l| l.axis == crate::scene::ITEM_LABEL_AXIS && l.text == "P" && l.visible));
        // Merge keeps the other keys; null resets one.
        cmd(
            &mut a,
            r#"{"t":"setStyle","id":"p","style":{"lineStyle":"dashed","opacity":0.5}}"#,
        );
        let st = style_of(&a, "p");
        assert_eq!(st.line_style, Some(doc::LineStyle::Dashed));
        assert_eq!(st.point_size, Some(14.0));
        cmd(
            &mut a,
            r#"{"t":"setStyle","id":"p","style":{"pointSize":null,"showLabel":null,"label":null}}"#,
        );
        let st = style_of(&a, "p");
        assert!(st.point_size.is_none() && !st.show_label && st.label.is_none());
        assert_eq!(st.opacity, Some(0.5));
        // Bad values and unknown ids/keys: an error and no change.
        let before = style_of(&a, "p");
        for bad in [
            r#"{"t":"setStyle","id":"p","style":{"pointSize":0}}"#,
            r#"{"t":"setStyle","id":"p","style":{"pointSize":41}}"#,
            r#"{"t":"setStyle","id":"p","style":{"opacity":2}}"#,
            r#"{"t":"setStyle","id":"p","style":{"fillOpacity":-1}}"#,
            r#"{"t":"setStyle","id":"p","style":{"lineWidth":0.1}}"#,
            r#"{"t":"setStyle","id":"p","style":{"lineWidth":30}}"#,
            r#"{"t":"setStyle","id":"p","style":{"lineStyle":"wavy"}}"#,
            r#"{"t":"setStyle","id":"p","style":{"showLabel":"yes"}}"#,
            r#"{"t":"setStyle","id":"p","style":{"colour":"red"}}"#,
            r#"{"t":"setStyle","id":"p","style":{"labelOffset":[1]}}"#,
            r#"{"t":"setStyle","id":"p","style":{"labelOffset":"up"}}"#,
            r#"{"t":"setStyle","id":"p","style":{"labelOffset":[1,null]}}"#,
            r#"{"t":"setStyle","id":"p","style":{"opacity":0.2,"pointSize":99}}"#,
            r#"{"t":"setStyle","id":"nope","style":{"opacity":0.2}}"#,
            r#"{"t":"setStyle","id":"p","style":3}"#,
        ] {
            let ev = cmd(&mut a, bad);
            assert!(error_msg(&ev).is_some(), "{bad}: {ev:?}");
            assert_eq!(style_of(&a, "p"), before, "{bad}");
        }
        assert!(
            error_msg(&cmd(&mut a, r#"{"t":"setStyle","id":"nope","style":{}}"#))
                .unwrap()
                .contains("nope")
        );
        // Tables too, and the style survives export.
        cmd(&mut a, r#"{"t":"addTable","id":"t","data":[[1,2],[3,4]]}"#);
        let ev = cmd(
            &mut a,
            r#"{"t":"setStyle","id":"t","style":{"pointStyle":"square","lineWidth":4,"fillOpacity":0.3}}"#,
        );
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"]
            .as_str()
            .unwrap()
            .to_string();
        let back = doc::from_json(&json).unwrap();
        let t = back.items.iter().find(|i| i.id == "t").unwrap();
        assert_eq!(t.style.point_style, Some(doc::PointStyle::Square));
        assert_eq!(
            (t.style.line_width, t.style.fill_opacity),
            (Some(4.0), Some(0.3))
        );
        assert!(json.contains("\"lineStyle\":\"dashed\""));
    }

    #[test]
    fn set_folder_files_items_and_hidden_folders_hide_them() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"addItem","id":"f","kind":"folder","latex":"F"}"#,
        );
        cmd(
            &mut a,
            r#"{"t":"addItem","id":"g","kind":"folder","latex":"G"}"#,
        );
        cmd(&mut a, r#"{"t":"setExpr","id":"c","latex":"y=x^2"}"#);
        cmd(&mut a, r#"{"t":"setExpr","id":"n","latex":"note"}"#);
        let ev = cmd(&mut a, r#"{"t":"setFolder","id":"c","folder":"f"}"#);
        assert!(error_msg(&ev).is_none(), "{ev:?}");
        assert_eq!(a.doc.items[2].folder.as_deref(), Some("f"));
        let drawn = |a: &App| a.colors.iter().any(|c| c.id == "c");
        assert!(drawn(&a));
        cmd(&mut a, r#"{"t":"setHidden","id":"f","hidden":true}"#);
        assert!(!drawn(&a), "hidden folder hides its items");
        cmd(&mut a, r#"{"t":"setHidden","id":"f","hidden":false}"#);
        assert!(drawn(&a));
        for bad in [
            r#"{"t":"setFolder","id":"c","folder":"c"}"#,
            r#"{"t":"setFolder","id":"c","folder":"n"}"#,
            r#"{"t":"setFolder","id":"c","folder":"zz"}"#,
            r#"{"t":"setFolder","id":"zz","folder":"f"}"#,
            r#"{"t":"setFolder","id":"g","folder":"f"}"#,
            r#"{"t":"setFolder","id":"f","folder":"f"}"#,
        ] {
            assert!(error_msg(&cmd(&mut a, bad)).is_some(), "{bad}");
            assert_eq!(a.doc.items[2].folder.as_deref(), Some("f"), "{bad}");
        }
        cmd(&mut a, r#"{"t":"setFolder","id":"c","folder":null}"#);
        assert!(a.doc.items[2].folder.is_none());
        cmd(&mut a, r#"{"t":"setFolder","id":"c","folder":"g"}"#);
        cmd(&mut a, r#"{"t":"setFolder","id":"c"}"#);
        assert!(a.doc.items[2].folder.is_none(), "missing folder un-files");
        assert!(a.doc.validate().is_ok());
    }

    #[test]
    fn set_view_flags_and_window() {
        let mut a = app();
        let view = |ev: &[serde_json::Value]| ev.iter().rev().find(|e| e["t"] == "view").cloned();
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","grid":false,"axisNumbers":false}"#,
        );
        let v = view(&ev).expect("view event");
        assert_eq!(
            (v["grid"].clone(), v["axes"].clone()),
            (false.into(), true.into())
        );
        assert_eq!(v["axisNumbers"], false);
        assert!(a.screen_labels().is_empty(), "no tick numbers");
        cmd(&mut a, r#"{"t":"setView","axes":false,"axisNumbers":true}"#);
        assert!(!a.doc.view.grid && !a.doc.view.axes && a.doc.view.axis_numbers);
        assert!(
            a.layers()[0].geometry.segments.is_empty(),
            "no grid, no axes in 2D"
        );
        // Window: exactly what the event and export report afterwards.
        let ev = cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-2,-1,-3],"max":[6,3,3]}}"#,
        );
        let v = view(&ev).unwrap();
        assert_eq!(v["min"], serde_json::json!([-2.0, -1.0, -3.0]));
        assert_eq!(v["max"], serde_json::json!([6.0, 3.0, 3.0]));
        let ev = cmd(&mut a, r#"{"t":"export"}"#);
        let json = ev.iter().find(|e| e["t"] == "doc").unwrap()["json"]
            .as_str()
            .unwrap()
            .to_string();
        let d = doc::from_json(&json).unwrap();
        assert_eq!(d.view.window.min, [-2.0, -1.0, -3.0]);
        assert_eq!(d.view.window.max, [6.0, 3.0, 3.0]);
        assert!(!d.view.grid && !d.view.axes && d.view.axis_numbers);
        assert!(json.contains("\"grid\":false") && !json.contains("axisNumbers"));
        // Loading the same window gives the same camera.
        let mut b = app();
        cmd(
            &mut b,
            &serde_json::json!({"t": "loadDoc", "json": json}).to_string(),
        );
        assert_eq!(b.rig.window().min, a.rig.window().min);
        assert_eq!(b.rig.render_origin(), a.rig.render_origin());
        // Items, sliders and the mode are untouched; bad windows are errors and change nothing.
        cmd(&mut a, r#"{"t":"setExpr","id":"e","latex":"y=x"}"#);
        cmd(&mut a, r#"{"t":"setSlider","name":"k","value":2}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(10_000.0);
        cmd(
            &mut a,
            r#"{"t":"setView","window":{"min":[-1,-1,-1],"max":[1,1,1]}}"#,
        );
        assert_eq!(a.mode(), Mode::D3);
        assert_eq!(a.doc.items.len(), 1);
        assert_eq!(a.doc.sliders["k"].value, 2.0);
        for bad in [
            r#"{"t":"setView","window":{"min":[1,-1,-1],"max":[1,1,1]}}"#,
            r#"{"t":"setView","window":{"min":[0,0,0],"max":[1,1,-1]}}"#,
            r#"{"t":"setView","window":{"min":[0,0],"max":[1,1]}}"#,
            r#"{"t":"setView","grid":"no"}"#,
        ] {
            assert!(error_msg(&cmd(&mut a, bad)).is_some(), "{bad}");
        }
        assert_eq!(a.rig.window().max, [1.0, 1.0, 1.0]);
        assert!(!a.doc.view.grid);
    }
}
