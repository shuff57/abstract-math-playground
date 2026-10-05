//! Platform-independent application state: the document, the shared camera rig, theme, input
//! handling and throttled scene rebuilds. Native (winit) and web (wasm + DOM) both drive this
//! through [`App::dispatch`] (JSON commands in, JSON events out) and [`App::frame`], so the two
//! front ends cannot drift apart. No GPU or window types here; fully unit-testable.

use crate::geometry::{ItemInfo, SceneGeometry, Theme};
use crate::render::{crossfade, layer_lift, Inset, Layer};
use crate::axis_map::AxisMap;
use crate::scene::{
    build_scene_mapped, build_scene_preview_mapped, build_slice_panel_view, label_box_inside,
    point_handles, CoordSrc, PointHandle, SlicePanel, ViewReq,
};
use math_core::actions;
use math_core::doc::{self, AngleMode, Doc, Item, ItemKind, SliderCfg, TickerCfg};
use math_core::table::{Column, Table, TableStyle};
use math_core::view::{Mode, Rig, Window3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// After the last input, wait this long before rebuilding an expensive scene.
const IDLE_REBUILD_MS: f64 = 100.0;
/// Most ticker steps fired by a single `frame` (a long stall drops the backlog instead of
/// replaying it).
pub const MAX_TICKER_STEPS_PER_FRAME: usize = 4;

/// Scenes that build faster than this rebuild on every input (cheap 2D scenes track live).
const CHEAP_BUILD_MS: f64 = 10.0;
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
    SetMode {
        mode: String,
    },
    SetOrtho {
        ortho: bool,
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
    /// True for a label of the slice inset (its `x`/`y` are already offset into the canvas);
    /// axis 0/1 are its tick labels and axis names, 3 is reserved for a title.
    #[serde(default)]
    pub inset: bool,
    /// Rectangle `[x, y, w, h]` the label must stay inside (the inset's, for inset labels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip: Option<[f64; 4]>,
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
}

struct Drag {
    button: u8,
    shift: bool,
    last: (f64, f64),
    /// Set when the press grabbed a point item: the move edits it instead of panning.
    point: Option<PointGrab>,
}

struct PointGrab {
    handle: PointHandle,
    /// Point position minus pointer world position at the press, so the point does not jump.
    offset: [f64; 2],
}

/// Pointer distance (px) within which a press grabs a point.
const GRAB_PX: f64 = 14.0;

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
    ticker_running: bool,
    /// Time accumulated towards the next ticker step.
    ticker_acc_ms: f64,
    /// `now_ms` at the previous ticker update (`None` until the first frame after starting).
    ticker_last_ms: Option<f64>,
    /// Last ticker error reported, so a persistent error is not repeated every step.
    ticker_err: Option<String>,
    /// Secondary inset for the slice, rebuilt with the main scene.
    panel: Option<SlicePanel>,
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
    /// `setReducedMotion`; kept here because loading a document replaces the rig.
    reduced_motion: bool,
    /// World -> display map of logarithmic 2D axes: the rig's window is in display coordinates
    /// while in 2D (see [`AxisMap`]); fixed between `load` / `setView` so pans stay smooth.
    map: AxisMap,
    /// The world window before the axes became logarithmic, so switching back to linear
    /// restores the exact previous view.
    log_entry: Option<doc::WindowBox>,
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

impl App {
    pub fn new(size: (u32, u32)) -> Self {
        let doc = Doc::new_default();
        let w = doc.view.window.clone();
        let mut rig = Rig::new(Window3::new(w.min, w.max), view_mode(doc.view.mode));
        rig.set_aspect(size.0 as f64 / size.1.max(1) as f64);
        let mut app = App {
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
            ticker_running: false,
            ticker_acc_ms: 0.0,
            ticker_last_ms: None,
            ticker_err: None,
            panel: None,
            last_slice: None,
            slice_view: None,
            inset_drag: None,
            last_inset_press: None,
            panel_dirty: false,
            last_panel_ms: 0.0,
            refine: false,
            reduced_motion: false,
            map: AxisMap::LINEAR,
            log_entry: None,
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
            .filter(|_| !self.rig.is_animating())
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
                let v = &mut self.doc.view;
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
            }
            Command::RemoveSlider { name } => {
                if self.doc.sliders.remove(&name).is_some() {
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
                self.rebuild();
            }
            Command::Pointer {
                phase,
                x,
                y,
                button,
                shift,
            } => self.pointer(&phase, x, y, button.clamp(0, 255) as u8, shift, vp),
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
                        self.doc.slice = Some(c);
                        self.mark_doc_changed();
                    }
                    Err(message) => self.outbox.push(Event::Error { message }),
                }
            }
            Command::ClearSlice => {
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
            Command::ResetSliceView => {
                if self.slice_view.take().is_some() {
                    self.rebuild_panel(self.rig.mode());
                }
            }
            Command::Reset => {
                self.rig.reset();
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
        const KEYS: [&str; 10] = [
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
        self.rig = Rig::new(Window3::new(w.min, w.max), view_mode(d.view.mode));
        self.rig.set_aspect(self.size.0 as f64 / self.size.1 as f64);
        self.doc = d;
        self.log_entry = None;
        self.frame_world_window(w);
        self.prev = None;
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
        if let Some(t) = self
            .doc
            .items
            .iter()
            .find(|i| i.id == id)
            .and_then(|i| i.table.as_ref())
        {
            self.outbox.push(Event::Table {
                id: id.to_string(),
                columns: t.columns.clone(),
                rows: t.rows(),
                style: t.summary_style().name().to_string(),
            });
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
                CoordSrc::Fixed => {}
            }
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
        self.rig = Rig::new(shown, self.rig.mode());
        self.rig.set_aspect(aspect);
        self.rig.reduced_motion = self.reduced_motion;
        self.prev = None;
    }
    fn set_mode(&mut self, mode: Mode) {
        if mode == self.rig.mode() {
            return;
        }
        self.slice_view = None;
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
        // The tween's clock starts at the next frame (see `Tween::start_ms`), so the time spent
        // building below is not taken out of the animation.
        self.rig.reduced_motion = self.reduced_motion;
        self.rig.set_mode(mode, self.now_ms);
        self.dirty = true;
        // A preview build keeps the gap between the command and the first moving frame short;
        // the full-quality scene follows when the switch has finished.
        let preview = self.rig.is_animating();
        self.rebuild_with(preview);
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
                self.inset_drag = None;
                let point = if button == 0 && !shift {
                    self.grab_point(x, y, vp)
                } else {
                    None
                };
                self.drag = Some(Drag {
                    button,
                    shift,
                    last: (x, y),
                    point,
                });
            }
            "up" | "cancel" => {
                self.drag = None;
                self.inset_drag = None;
            }
            "move" => {
                let m = self.active_map();
                let Some(d) = self.drag.as_mut() else { return };
                let (dx, dy) = (x - d.last.0, y - d.last.1);
                d.last = (x, y);
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
            self.redraw = true;
        } else if self.refine && !animating {
            // From the frame after the switch ended, so its final pose is shown on time.
            self.dirty = true;
        }
        if self.dirty {
            let idle = now_ms - self.last_input_ms >= IDLE_REBUILD_MS;
            // `last_build_ms` is not known for a scene only built as a preview yet: wait for a
            // pause in the input (an orbit right after the switch) before the full build.
            let cheap = self.last_build_ms < CHEAP_BUILD_MS && !self.refine;
            if idle || cheap {
                // Mid-switch (a slider ticking, say) only a preview fits in a frame.
                self.rebuild_with(animating);
            }
        }
        if self.panel_dirty && !self.dirty {
            let idle = now_ms - self.last_input_ms >= IDLE_REBUILD_MS;
            if idle || self.last_panel_ms < CHEAP_BUILD_MS {
                self.rebuild_panel(self.rig.mode());
                self.redraw = true;
            }
        }
        let redraw = self.redraw || animating || self.dirty || self.panel_dirty;
        self.redraw = false;
        redraw
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
        let t0 = instant::Instant::now();
        let mode = self.rig.mode();
        let origin = self.rig.render_origin();
        let window = self.scene_window();
        let geometry = if preview {
            let (g, coarse) = build_scene_preview_mapped(
                &self.doc, self.map, mode, window, origin, self.size, &self.theme,
            );
            self.refine = coarse;
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
        });
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
        self.current = Some(Built {
            geometry,
            origin,
            mode,
        });
        self.dirty = false;
        self.redraw = true;
    }

    /// Rebuilds the slice inset and reports the slice state when it changed.
    fn rebuild_panel(&mut self, mode: Mode) {
        let t0 = instant::Instant::now();
        self.panel_dirty = false;
        let window = self.scene_window();
        let res = if self.doc.slice.is_some() && !self.active_map().is_linear() {
            Some(Err(format!("a slice is {}", crate::scene::LOG_UNSUPPORTED)))
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
        let p = self.panel.as_ref().filter(|_| !self.rig.is_animating())?;
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

    /// Layers to draw this frame, oldest first.
    pub fn layers(&self) -> Vec<Layer<'_>> {
        let mut out = Vec::new();
        match (&self.prev, &self.current) {
            (Some(p), Some(c)) if self.rig.is_animating() => {
                let (fa, fb) = crossfade(self.rig.progress() as f32);
                let lift = self.rig.lift();
                out.push(Layer {
                    geometry: &p.geometry,
                    fade: fa,
                    origin: p.origin,
                    lift: layer_lift(p.mode, lift),
                });
                out.push(Layer {
                    geometry: &c.geometry,
                    fade: fb,
                    origin: c.origin,
                    lift: layer_lift(c.mode, lift),
                });
            }
            (_, Some(c)) => {
                let lift = layer_lift(c.mode, self.rig.lift());
                out.push(Layer {
                    geometry: &c.geometry,
                    fade: 1.0,
                    origin: c.origin,
                    lift,
                })
            }
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
        let aspect = self.size.0 as f64 / self.size.1.max(1) as f64;
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let rect = self.inset_rect();
        // Labels of a 3D scene ride the switch lift with its geometry.
        let lift = layer_lift(c.mode, self.rig.lift()) as f64;
        let mut out: Vec<ScreenLabel> = c
            .geometry
            .labels
            .iter()
            .map(|l| {
                let ndc = self
                    .rig
                    .project_ndc([l.pos[0], l.pos[1], l.pos[2] * lift], aspect);
                let (x, y) = ((ndc[0] * 0.5 + 0.5) * w, (1.0 - (ndc[1] * 0.5 + 0.5)) * h);
                let mut visible =
                    ndc[0].abs() <= 1.0 && ndc[1].abs() <= 1.0 && (0.0..=1.0).contains(&ndc[2]);
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
                ScreenLabel {
                    text: l.text.clone(),
                    axis: l.axis,
                    x,
                    y,
                    visible,
                    inset: false,
                    clip: None,
                }
            })
            .collect();
        out.extend(self.inset_screen_labels());
        out
    }

    /// Tick labels and axis names of the slice inset, in canvas pixels (empty without one).
    pub fn inset_screen_labels(&self) -> Vec<ScreenLabel> {
        let Some(p) = self.panel.as_ref().filter(|_| !self.rig.is_animating()) else {
            return Vec::new();
        };
        let r = p.rect;
        let aspect = r[2] as f64 / r[3].max(1) as f64;
        let (pw, ph) = (r[2] as f64, r[3] as f64);
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
                    && label_box_inside(l.axis, &l.text, lx, ly, pw, ph);
                ScreenLabel {
                    text: l.text.clone(),
                    axis: l.axis,
                    x: r[0] as f64 + lx,
                    y: r[1] as f64 + ly,
                    visible,
                    inset: true,
                    clip: Some([r[0] as f64, r[1] as f64, pw, ph]),
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
    use super::*;

    fn app() -> App {
        App::new((800, 600))
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
        let l = a.layers();
        assert_eq!(l.len(), 2);
        assert_eq!((l[0].fade, l[1].fade), crossfade(0.0));
        assert_eq!(l[1].lift, 0.0, "the 3D scene starts flat in the plane");
        let mut t = 60_000.0;
        let mut frames = 0;
        while a.rig.is_animating() {
            t += 1000.0 / 60.0;
            a.frame(t);
            frames += 1;
        }
        assert!(frames >= 29, "{frames} frames");
    }

    #[test]
    fn switch_to_3d_previews_then_refines_at_full_quality() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"z=\\sin(x)\\cos(y)"}"#,
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
        let preview = a.layers()[1].geometry.vertices.len();
        assert!(a.refine);
        assert!(
            preview > 0 && preview < full.vertices.len(),
            "{preview} vs {}",
            full.vertices.len()
        );
        // Mid-switch: the 3D scene is partly lifted, the outgoing 2D one is untouched.
        a.frame(16.0);
        a.frame(266.0);
        let l = a.layers();
        assert!(l[1].lift > 0.0 && l[1].lift < 1.0);
        assert_eq!(l[0].lift, 1.0);
        drop(l);
        assert!(a.refine, "no full rebuild while moving");
        a.frame(600.0);
        assert!(!a.rig.is_animating());
        assert!(a.refine, "the final pose is drawn first");
        a.frame(616.0);
        assert!(!a.refine);
        let l = a.layers();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].lift, 1.0);
        assert_eq!(l[0].geometry.vertices.len(), full.vertices.len());
        // Leaving 3D needs no preview (2D builds are cheap) and nothing to refine.
        drop(l);
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        assert!(!a.refine);
    }

    #[test]
    fn refine_waits_for_a_pause_in_the_input() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"z=\\sin(x)\\cos(y)"}"#,
        );
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        a.frame(10.0);
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":400,"y":300}"#);
        let mut t = 10.0;
        while t < 800.0 {
            t += 16.0;
            a.now_ms = t;
            cmd(
                &mut a,
                &format!(
                    r#"{{"t":"pointer","phase":"move","x":{},"y":300}}"#,
                    400.0 + t / 10.0
                ),
            );
            a.frame(t);
        }
        assert!(!a.rig.is_animating());
        assert!(a.refine, "no full build while the user is orbiting");
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":480,"y":300}"#);
        a.frame(t + 50.0);
        assert!(a.refine);
        a.frame(t + 200.0);
        assert!(!a.refine);
    }

    #[test]
    fn reduced_motion_switches_at_once_at_full_quality() {
        let mut a = app();
        cmd(
            &mut a,
            r#"{"t":"setExpr","id":"a","latex":"z=\\sin(x)\\cos(y)"}"#,
        );
        cmd(&mut a, r#"{"t":"setReducedMotion","on":true}"#);
        a.frame(0.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(!a.rig.is_animating());
        assert!(!a.refine);
        a.frame(16.0);
        let l = a.layers();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].lift, 1.0);
        drop(l);
        // The setting outlives a document load (which replaces the camera rig).
        let json = doc::to_json(&a.doc);
        cmd(
            &mut a,
            &serde_json::json!({"t":"loadDoc","json":json}).to_string(),
        );
        a.frame(32.0);
        cmd(&mut a, r#"{"t":"setMode","mode":"2d"}"#);
        assert!(!a.rig.is_animating());
        cmd(&mut a, r#"{"t":"setReducedMotion","on":false}"#);
        cmd(&mut a, r#"{"t":"setMode","mode":"3d"}"#);
        assert!(a.rig.is_animating());
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
        // The scene is cheap, so it rebuilds at once and the origin follows the window.
        a.frame(1.0);
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
                .filter(|s| s.p0 == s.p1 && s.width == 9.0)
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
                .filter(|s| s.width == 9.0)
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
        assert_eq!(dots(&a, 9.0), 6);
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
        assert!(segs.iter().filter(red).any(|s| s.width == 2.5 && s.p0 != s.p1), "lines between the y_2 points");
        assert!(segs.iter().filter(red).any(|s| s.p0 != s.p1 && s.width != 2.5), "square markers");
        assert_eq!(dots(&a, 9.0), 3, "y_1 keeps its dots");
        // Points off for y_1; an outline ring for y_2 (drawn as a larger background dot).
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":1,"style":{"points":false}}"#);
        assert_eq!(dots(&a, 9.0), 0);
        cmd(&mut a, r#"{"t":"setTableColumnStyle","id":"t","col":2,"style":{"outline":true,"pointStyle":null}}"#);
        assert_eq!(dots(&a, 14.0), 3);
        assert_eq!(dots(&a, 18.0), 3);
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
        assert!(c.layers()[0].geometry.segments.iter().any(|s| s.p0 != s.p1 && s.width == 2.5 && s.color[3] > 0.9 && s.color[0] > 0.7));
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
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":100,"y":100}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":60,"y":100}"#);
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
        cmd(&mut a, r#"{"t":"pointer","phase":"down","x":100,"y":100}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"move","x":60,"y":100}"#);
        cmd(&mut a, r#"{"t":"pointer","phase":"up","x":60,"y":100}"#);
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
