# Engine API

The calculator engine (`App` in `math_playground/src/app.rs`) is driven entirely by JSON. The native window and the web shell both use it, so everything a UI can do is a command here. Source of truth: the `Command` and `Event` enums in `app.rs` and `web.rs`.

- A **command** is a JSON object with a `t` field naming it (camelCase), plus its fields. Field names are exactly as listed below (snake_case where shown).
- `dispatch(json)` returns a JSON array of **events**. A malformed or unknown command does not throw; it returns an `error` event with `bad command: ...`.
- Events raised outside `dispatch` (inside `frame`: ticker results, throttled slice updates) are collected and returned by the next `dispatch`, or by `drain()`.

```js
const events = JSON.parse(calc.dispatch(JSON.stringify({ t: "setExpr", id: "a", latex: "y=x^2" })));
```

## Commands

Items and the list:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setExpr` | `id`, `latex` | Sets the text of item `id`. If no such item exists, one is added as an equation. |
| `addItem` | `id`, `kind`, `latex` | `kind` is one of `equation`, `expression`, `complex`, `points`, `vectorField`, `action`, `slider`, `slice`, `folder`, `note`, `table`. |
| `removeItem` | `id` | |
| `moveItem` | `id`, `to` | New index in the list. |
| `setHidden` | `id`, `hidden` | Hidden items are not drawn and report no errors, but their definitions stay in scope. |
| `setColor` | `id`, `color` | `color` is a string such as `"#ff0000"`, or `null` for the automatic colour. |
| `setRegressionResiduals` | `id`, `on` | Regression items: draw ticks from each data point to the fit. |
| `setStyle` | `id`, `style` | `style` is a partial `ItemStyle` object (camelCase keys as in the document: `lineWidth`, `lineStyle`, `pointStyle`, `pointSize`, `opacity`, `fillOpacity`, `label`, `showLabel`, `labelOffset`, `dragMode`, `asFraction`, `labelSize`, `pointOutline`, `residuals`). Given keys are merged into the item's style; a key set to `null` returns to its default. Works for every item kind, tables included. Unknown id, unknown key, wrong type or out-of-range value gives an `error` event and changes nothing. `lineWidth` set here must be in [0.25, 20]. `dragMode` is `none`, `x`, `y` or `xy` (default): which coordinates of a point item the pointer may change; `none` makes the point undraggable (a press on it pans), `x` and `y` freeze the other coordinate, and a coordinate that is a formula never moves. `labelOffset` is `[dx, dy]` (two finite numbers, CSS pixels, y down): it moves the item's label off its point; values beyond +-400 are clamped, `null` returns the label to its default place. Example: `{"t":"setStyle","id":"a","style":{"lineStyle":"dashed","pointSize":null}}`. |
| `setFolder` | `id`, `folder` | `folder` is the id of a folder item, or `null` (or absent) for the top level. Error if `id` is unknown, the folder is not a folder item, is the item itself, or the item is itself a folder (no nesting). Items in a hidden folder are not drawn (their definitions stay in scope). |

Sliders:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setSlider` | `name`, `value`, optional `min`, `max`, `step` | Creates the slider if needed (range defaults to -10..10). |
| `removeSlider` | `name` | Also drops the slider's saved playback settings. |
| `setSliderPlay` | `name`, optional `mode` (`oscillate`, `loop`, `once`), optional `speed` (multiplier in (0, 20]) | Saves how the workspace animates the slider (the engine itself does not animate); stored in the document's `sliderPlay` map, which keeps only non-default entries. An unknown slider or a bad speed gives an `error` event and changes nothing. |

View and theme:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setMode` | `mode` | `"1d"`, `"2d"` or `"3d"`. Starts the animated transition (see below). Anything else gives an `error` event. |
| `setOrtho` | `ortho` | Orthographic camera in 3D. |
| `setReducedMotion` | `on` | Mode switches and the ortho toggle jump to the end instead of animating. Send it from `prefers-reduced-motion` at start-up and when it changes (the TS shell does). Default off. |
| `setTheme` | `dark` | |
| `setView` | all optional: `grid`, `axes`, `axisNumbers` (booleans), `window` `{min:[x,y,z], max:[x,y,z]}` | Flags: `grid` false hides the minor/major grid lines, `axes` false hides the axis lines, ticks and tick numbers, `axisNumbers` false hides only the tick numbers (the 3D box edges always stay). `window` sets the window like loading a document with it (fresh camera framing it, current mode kept); items, sliders, ticker and slice are untouched. Each axis must be finite with min < max, else an `error` and no change. `weight` (`normal` default, `bold`, `extra`) is the print weight: curve lines, axes, grid, ticks, arrowheads and point sizes are drawn about 1.6x / 2.2x thicker (grid lines and the 3D box edges only 1.3x / 1.5x so the grid stays a background; points 1.3x / 1.6x, at most 40 px unless the item asked for more; the 3D slice inset follows the same weights); applied when drawing, so item styles are unchanged and `normal` restores them. Saved in the document (`view.weight`, omitted when normal) and the share link. Any other value is an `error` and no change. Followed by a `view` event. `textScale` (number 0.5 to 3, default 1) scales all text drawn on the graph: tick numbers, axis names, item labels (and their leader lines), tooltips and the slice inset labels. The engine widens the target gap between ticks and the inset label boxes by it, so larger text thins the tick numbers out instead of overlapping them; independent of `weight`. Saved in the document (`view.textScale`, omitted when 1; a saved value outside the range is clamped on load) and the share link. A value outside 0.5..3 or not a number is an `error` and no change. |
| `setRenderScale` | `scale` | Transient, never saved: draws widths `scale` times wider in pixels (1 default, clamped to 8) so an export rendered on a canvas `scale` times larger keeps the on-screen look (widths are physical pixels). Send it before `resize` for the export and `1` afterwards. |
| `setAngle` | `angle` | `"deg"`, otherwise radians. |
| `resize` | `width`, `height` | Physical pixels. |
| `reset` | none | Resets the camera. |

Input (pixel coordinates from the canvas top-left, physical pixels):

| `t` | Fields | Notes |
| --- | --- | --- |
| `pointer` | `phase`, `x`, `y`, optional `button` (default 0), `shift` (default false) | `phase` is `down`, `move`, `up`, `cancel` or `dblclick` (a double-click on the slice inset resets its view). Left-drag pans in 1D/2D, orbits in 3D; shift or another button pans. A press on a point item grabs it and edits its text (see `itemEdited`). A press on the **selected** curve (see `pick`) that has parameters (sliders or numeric definitions such as `a=3` used in its equation; plain numbers in the equation are never changed) drags it instead of panning: each move changes those parameters so the grabbed curve point follows the pointer and the curve keeps its shape around it (a translation when parameters allow), clamped to slider ranges and snapped to slider steps, answered with `sliderValue` / `itemEdited` and `curveDrag`. The target is computed in display (pixel) space through the axis map, so on logarithmic axes the curve stays under the pointer (a power law `y=a x^b` on log-log axes translates, keeping `b`). A slider with a `step` writes the grid values that fit best (the lattice neighbours of the continuous solution are compared by residual); the continuous solution is kept internally, so no error accumulates over many small moves and moving back restores the starting values. Draggable curve kinds: `y=f(x)`, `x=g(y)`, implicit `F(x,y)=0` (e.g. `(x-h)^2+(y-k)^2=r^2`), polar `r(theta)` and 2D parametric `(x(t),y(t))`; for the last three the grabbed point keeps its place on the curve (`F` stays 0 there; the same `t` lands on the moved point) and a few points around it move with it, so the curve translates when position parameters (`h`, `k`) exist and a radius slider such as `r` follows only when translation cannot carry the move (for example when it is the only parameter). A `cancel` phase restores the starting values. Pressing an unselected curve, or one without parameters, still pans. A `move` with no drag in progress is a hover in 2D: see the `hover` event. |
| `pick` | `x`, `y` | A click (press and release without a drag) at canvas pixel `(x, y)`. In 2D it selects the curve under the pointer (an explicit `y = f(x)`, or an implicit, polar or parametric one; same 10 px reach as `hover`), or clears the selection when there is none, and answers with an `analysis` event. Elsewhere it does nothing. The shell decides what counts as a click. |
| `cancelDrag` | none | Abandons a curve drag in progress (the shell's Escape): the dragged sliders and definitions go back to their values at the press, with a `curveDrag` event (`active` false, `cancelled` true). Later moves of that press do nothing. No effect when no curve is being dragged. |
| `wheel` | `x`, `y`, `dy` | Zoom at the cursor. Over the slice inset it zooms the inset instead. |

Documents:

| `t` | Fields | Notes |
| --- | --- | --- |
| `loadDoc` | `json` | A document as a JSON string (format below). Invalid input gives an `error` event. |
| `loadHash` | `hash` | A share hash (`v1.` + URL-safe base64 of deflated JSON). |
| `export` | none | Returns a `doc` and a `hash` event for the current state. |

Tables (`id` is a table item):

| `t` | Fields | Notes |
| --- | --- | --- |
| `addTable` | `id`, optional `columns` (names, default `x_1`, `y_1`), `data` (row-major, strings or numbers), `rows` | Blank rows default to 3, or 0 when `data` is given. Invalid column names give an `error`. |
| `setCell` | `id`, `row`, `col`, `value` | `value` is a string or number; a row past the end grows the table. |
| `addRow` | `id`, optional `at` | |
| `removeRow` | `id`, `row` | |
| `addColumn` | `id`, optional `name` | Name defaults to the next free header. |
| `removeColumn` | `id`, `col` | |
| `renameColumn` | `id`, `col`, `name` | |
| `setTableStyle` | `id`, `style` | `points`, `line` or `hidden`. |

Actions and ticker:

| `t` | Fields | Notes |
| --- | --- | --- |
| `runAction` | `id` | Evaluates action item `id` once and applies the result. |
| `setTicker` | all optional: `action`, `rate_ms` (alias `rateMs`), `min_step_ms` (alias `minStepMs`), `pause_on_error` (alias `pauseOnError`), `running` | `action` is an action item id; `null` clears the action and stops the ticker; a missing `action` leaves it alone. Defaults: rate 50 ms, min step 10 ms, pause on error true. |
| `tickerStep` | none | Fires the ticker action once now. |
| `startTicker`, `stopTicker`, `toggleTicker` | none | |

Slices:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setSlice` | `fixed` (aliases `plane`, `axis`, `at`, `axes`), optional `dim` | `fixed` is an object `{"z":0.5}` / `{"y":1,"z":"a"}` or a string `"y=1,z=0.5"`. Values are numbers or expressions of slider names; a single value that uses `x`, `y` or `z` and is linear (`{"z":"2x+y"}`) is a sloped plane in 3D (the `slice` event then has an empty `fixed` and `free` is `["u","v"]`). `dim` is derived from the mode and the number of fixed axes; a value that disagrees is an error. |
| `clearSlice` | none | |
| `setSliceView` | optional `min` `[xmin,ymin]`, `max` `[xmax,ymax]`, `view` `[xmin,xmax,ymin,ymax]` | Sets the inset's own window; it stops following the main window. Error if there is no slice. |
| `resetSliceView` | none | Back to auto-follow. |

## Events

Every event has a `t` field.

| `t` | Fields | Meaning |
| --- | --- | --- |
| `diagnostics` | `items`: `[{id, message}]` | Per-item problems (parse errors, unsupported constructs). Examples for the newer forms: `a {range} applies to parametric curves, polar curves and parametric surfaces` (a range on an item that cannot take one, such as `z=f(x,y)`), `a {range} on 'q' does not apply here (an explicit or implicit item is restricted by x or y)`, `empty range: 2 is not below 1`, `a {range} on x or y restricts curves in 2D only` (3D) and `... needs linear axes` (log axes), `a '!=' comparison ... is not a scalar expression` (`x!=3` alone), `in a piecewise {..} only the last value may have no condition`. |
| `info` | `items`: `[ItemInfo]` | Read-outs: derivative, scalar value, regression fit. The full current list, sent whenever it changes; an empty list clears them. |
| `colors` | `items`: `[{id, color}]` | Resolved `#rrggbb` colour of each drawn item. Sent only when it changes. |
| `view` | `mode`, `min` `[x,y,z]`, `max` `[x,y,z]`, `grid`, `axes`, `axisNumbers` | Current mode, window and view flags, plus `weight` (`normal`, `bold` or `extra`), and `textScale` (number, 1 normal). |
| `labels` | `labels`: `[{pos:[x,y,z], text, axis}]` | Tick labels (and item labels, axis 4) in world coordinates. |
| `doc` | `json` | Document JSON string (reply to `export`). |
| `hash` | `hash` | Share hash (reply to `export`). |
| `table` | `id`, `columns`: `[{name, cells}]`, `rows`, `style` | Full table state, after each change and once per table on load. |
| `itemEdited` | `id`, `latex` | A drag rewrote an item's text (a point or a definition); put `latex` in that item's input. |
| `hover` | `item`, `x`, `y`, `px`, `py`, optional `params` | The pointer is within 10 px of a visible curve (2D only): an explicit `y = f(x)` (an equation `y=...` or a bare expression of `x`), or an implicit (`F(x,y)=0`, `x=g(y)`), polar or parametric curve, found by pixel distance to the polyline the renderer draws (`mesh::sample_parametric`, `mesh::contour_2d`; implicit curves are sampled over the window of the last rebuild). `item` is the curve's id, `(x, y)` the curve point nearest the pointer in world coordinates (on logarithmic axes too), `px`/`py` that point in canvas pixels from the top-left. `item` is null (numbers 0) when the pointer is over no curve. Sent only when the result changes; a press resets it so the next move sends again. Hidden items, inequalities, 3D curves and 1D/3D modes give none. `params` (omitted when empty) lists the parameter names a drag would change: non-empty only when the curve is the selected one and has any, so the shell can show a grab cursor and hint. A `pick` re-sends hover so this follows selection changes. |
| `analysis` | `item`, `points` | The special points of the selected curve over the visible window: `points` is a list of `{kind, x, y, px, py}`, `kind` one of `root`, `y-intercept`, `minimum`, `maximum`, `inflection`, `intersection`, `(x, y)` in world coordinates and `px`/`py` in canvas pixels. An `intersection` also has `with`, the id of the other curve; it is where the selected curve meets another visible curve (explicit `y=f(x)`, implicit including circles, conics and `x=g(y)`, polar, parametric; see `math_core/src/intersect.rs`), inside the visible window, honouring each curve's visibility. Partners are the first 8 other visible curves in document order; a spot already listed (a root that is also a crossing with `y=0`) is not repeated. `item` is null and `points` empty when nothing is selected; an implicit, polar or parametric curve can be selected too; it has no roots or extrema, only intersections. Sent when the selection changes and again after every rebuild that changes the points or their pixels (pan, zoom, edits, slider moves); a selected curve that is deleted, hidden, edited into another kind or left in 2D is deselected. Roots are sign changes (a pole is not a root; a tangent root such as the one of `x^2` is found as a minimum, not a root); extrema and inflection points are sign changes of `f'` and `f''` (symbolic where possible, else finite differences). At most 48 points in all: when there are intersections, the special points are cut so that up to 16 slots stay free for them. Intersection method: pairs that reduce to one variable (explicit with anything, or implicit with parametric) are scanned and bisected, and tangent touches (a double root, such as a line tangent to a circle) are found as zero minima of the difference; implicit/implicit and parametric/parametric pairs are sampled on a 160x160 grid and refined by damped Gauss-Newton, so a tangent touch is found but only to about 1e-8. Curves that coincide (identical, or overlapping along a stretch) give no points. A pair gives at most 48 points. Limits: two crossings closer than a grid cell, or a closed loop smaller than a cell, can be missed in implicit/implicit pairs; points outside the window are not listed; inequalities are not curves here. || `sliderValue` | `name`, `value` | A drag or action moved a slider. |
| `curveDrag` | `item`, `params`, `active`, `cancelled`, `px`, `py` | A curve drag started (`active` true, before any change), moved (sent whenever a value changed) or ended (`active` false; `cancelled` true when Escape or a `cancel` phase restored the starting values). `params` is `[{name, value}]` for every parameter of the curve; `px`/`py` is the pointer in canvas pixels. The value changes themselves arrive as `sliderValue` / `itemEdited` events in the same batch. |
| `theme` | `dark` | |
| `slice` | `active`, `dim`, `fixed` (map axis to number), `free` (axis names), `rect` `[x,y,w,h]` or null, `curves`, `points`, `error`, `view` `[xmin,xmax,ymin,ymax]` or null, `follow` | Slice state, sent when it changes (also while a slider sweeps it). `active` is false with no slice or one that does not fit the mode (`error` says why). `rect` is the inset in canvas pixels. |
| `tickerState` | `running`, `action` | Ticker started, stopped (also automatically after an error) or reconfigured. |
| `error` | `message` | A problem with a command. Ticker problems are prefixed `ticker: `. |

`ItemInfo`: `id`, `kind` (`"derivative"`, `"value"` or `"regression"`), and as they apply `latex`, `text`, `value`, `params` (`[{name, value, stdError?}]`), `r2`, `rmse`, `n`. Absent fields are omitted.

`screen_labels_json()` (below) returns the labels already projected to screen: `{text, axis, x, y, visible, alpha, inset, clip?, item?, offset?, size?}`. `inset` is true for slice-inset labels (their `x`/`y` are already offset into the canvas), `clip` is the `[x,y,w,h]` rectangle the label must stay inside. Axis 0 and 1 are tick labels and axis names (and the value labels of 1D dots); 2 is the z tick labels in 3D; 3 is reserved for a title; 4 is an item label (`showLabel`): `pos` is the point itself (the overlay should offset the text so it does not cover the marker), the text is the item's `label` or the coordinates such as `(1, 2)`; at most 100 per item; a curve with `showLabel` and a `label` gets one at its first drawn sample inside the window. Item labels carry `item` (the item id) and, when the item has a `labelOffset`, `offset: [dx, dy]` (CSS pixels, y down, clamped to +-400): `x`/`y` stay the point's own position (canvas pixels), and the overlay (and any PNG export) draws the text at `x/dpr + dx`, `y/dpr + dy`, with a leader line back to `x/dpr, y/dpr` when the offset is large. Labels are rebuilt with the scene on every edit, point drag and slider change, so text and position are current in the next frame. `size`, when present, is the item's `labelSize` as a text scale (0.75 small, 1.45 large; absent for medium and for tick labels): the overlay multiplies its text size by it.

## Document format

`loadDoc` and `export` use a versioned JSON document (`math_core/src/doc.rs`, current `v` is 1):

```json
{
  "v": 1,
  "view": { "mode": "2d", "window": { "min": [-10,-10,-10], "max": [10,10,10] }, "angle": "rad", "grid": false },
  "items": [ { "id": "a", "kind": "equation", "latex": "y=x^2", "hidden": false, "color": "#ff0000",
               "style": { "lineWidth": 2.0, "lineStyle": "dotted" } } ],
  "sliders": { "k": { "min": 0, "max": 1, "step": 0.5, "value": 0.25 } },
  "ticker": { "action": "i6", "rateMs": 125 },
  "slice": { "dim": 1, "fixed": { "y": "a" } }
}
```

Optional item fields: `hidden`, `color`, `style` (`lineWidth` (0,100], `lineStyle` `solid|dashed|dotted`, `pointStyle` `dot|circle|cross|square|plus|triangle|diamond|star`, `pointSize` [1,40] (marker diameter in pixels, default 9), `opacity` [0,1], `fillOpacity` [0,1] (effective opacity of an inequality's shading, times `opacity`; absent = the built-in 0.22), `label` (text), `showLabel` (bool), `labelOffset` (`[dx, dy]` CSS pixels, each in [-400, 400]; absent = the default place), `dragMode` (`none|x|y|xy`; absent = `xy`), `asFraction` (bool: a number item shows `7/3` instead of `2.3333`, only when the value is a fraction with denominator up to 10 000), `labelSize` (`small|medium|large`; absent = medium: the label text is drawn at 0.75, 1 or 1.45 times the text size), `pointOutline` (bool: dot markers get a thin dark ring), `residuals`), `folder`, `table` (for `table` items). Style keys at their default are omitted. `view.theme` is optional; `view.grid`, `view.axes`, `view.axisNumbers` default to true and are written only when false.

Rendering of styles: `dashed`/`dotted` split curves into dashes measured in pixels (explicit, implicit and contour curves, polar, parametric, 3D curves, regression curves and table lines; surfaces, fields, vector fields and statistical plots stay solid). `pointStyle`/`pointSize` apply to point items, point lists, table points and 1D dots; `circle`, `cross`, `square`, `plus`, `triangle`, `diamond` and `star` are open outlines in 1D/2D (the closed shapes have an opaque background interior, like the circle) and fall back to a filled dot in 3D. `sliders`, `ticker` and `slice` are optional. Unknown JSON fields are ignored; unknown enum values are errors.

Input is treated as untrusted (share links come from strangers). Limits: JSON up to 100,000 bytes, share hash up to 200,000 characters, 500 items, 200 sliders, 2000 characters of text per item, ids up to 64 characters. Decompression output is capped.

## WASM `Calculator`

Built by `wasm-pack build --target web` from `math_playground/src/web.rs`. In the web shell it is wrapped by `math_playground_web/src/engine.ts`.

```js
import init, { create_calculator } from "../../math_playground/pkg/math_playground_lib.js";
await init(wasmUrl);
const calc = await create_calculator(canvasElement);   // async; rejects with a string on failure
```

`create_calculator(canvas)` takes the canvas's current `width`/`height` (physical pixels) as the initial size, installs a panic hook, and creates a WebGL2 surface (the only backend requested; see [ROADMAP.md](ROADMAP.md)). It rejects if no GPU adapter is available.

Methods on `Calculator`:

| Method | Returns | Description |
| --- | --- | --- |
| `dispatch(json: string)` | `string` (JSON array of events) | Runs one command (see above). Also returns any events still pending. |
| `drain()` | `string` (JSON array of events) | Events raised since the last `dispatch`/`drain`: ticker `itemEdited`, `sliderValue`, `error`, `tickerState`, and throttled `slice` updates from `frame`. Calling it is optional (`dispatch` returns pending events too) but never loses events. |
| `frame(now_ms: number)` | `boolean` | Advances the app (animation, ticker, throttled scene rebuilds) and renders if something changed. Returns whether a frame was drawn. The JS side owns `requestAnimationFrame` and passes the timestamp. If the surface is lost it reconfigures and retries once. |
| `resize(width, height)` | nothing | New size in physical pixels. Reconfigures the surface and sends the `resize` command. Ignored for zero sizes (the surface keeps its size). |
| `screen_labels_json()` | `string` | JSON array of tick labels projected to screen pixels, for an HTML overlay (each has `alpha`, see Mode transitions). `"[]"` before the first scene exists. |
| `busy()` | `boolean` | True while background work (a 3D scene being refined in time slices) needs more `frame` calls even though nothing was drawn. |
| `settle()` | nothing | Finishes any pending rebuild or refinement at full quality now, e.g. before exporting an image. |
| `backend_name()` | `string` | The wgpu backend picked, from `adapter.get_info().backend` (for example `Gl`). |

Typical loop:

```js
function tick(now) {
  const drawn = calc.frame(now);
  const events = JSON.parse(calc.drain());
  handle(events);
  if (drawn) overlay(JSON.parse(calc.screen_labels_json()));
  if (drawn || calc.busy()) requestAnimationFrame(tick); // busy: a 3D scene is still refining
}
requestAnimationFrame(tick);
```

The native build does not use `Calculator`; `native.rs` calls `App::dispatch` and `App::frame` directly.

### Mode transitions

`setMode` moves the camera to the new mode over 500 ms (ease-in-out cubic) while the outgoing scene fades out and the incoming one fades in. How it behaves with the frame loop:

- The transition's clock starts at the first `frame` after the command, not at the command. A loop may stop calling `frame` while nothing changes (the native shell does, and so may an embedding that sleeps its `requestAnimationFrame`), and the scene for the new mode is built inside `setMode`; neither eats into the animation. The first frame after the switch shows the starting pose; every later one moves.
- Frame time is clamped: one `frame` advances the tween by at most 50 ms (`Rig::max_frame_dt_ms`, `view::MAX_FRAME_DT_MS`), so a stalled frame (a first GPU upload, a long task, a hidden tab) costs the animation one frame instead of swallowing it and making the picture freeze then jump.
- Switching into 3D never blocks. `setMode` builds only a first look (surfaces at octree depth `scene::FIRST_PREVIEW_DEPTH` = 4, a few milliseconds) so the first moving frame comes at once, then `frame` refines the surfaces in time slices (3 ms per frame while the pointer is moving, 5 ms during the tween, 9 ms when idle): depth 5, then full quality. An implicit surface is meshed as lattice-aligned tiles that are identical to the one-shot mesh (no seams), a few per frame; an explicit height field (`z = f(x, y)`) cannot be tiled without seams, so it is one unsliceable step: at most one per frame from depth 5, and those at full depth (`HEAVY_HEIGHT_DEPTH`) wait until the tween has ended. An unfinished stage is never shown (no holes). A finished stage replaces the scene by blending in over the previous one for 160 ms (`SWAP_BLEND_MS`; none with reduced motion). `App::busy()` / `Calculator::busy()` is true while slices remain, and the shell must keep calling `frame` then. `App::settle()` / `Calculator::settle()` finishes everything at full quality at once (the viSHual export calls it before capturing a PNG). Leaving 3D builds the cheap 1D/2D scene directly.
- Layers during a switch: a 2D/1D scene leaving for (or arriving from) 3D is drawn as two layers, its backdrop (grid, axes) and its items (curves, regions, fields), which fade on different schedules; see `render::mode_fades`. The fades of a switch with 3D follow the LIFT, not the clock: the flat grid is gone first, the 3D scene fades in once it has height (lift 0.2 to 0.6, so a flat sheet never washes out the grid), and the 2D curves stay opaque until the 3D form has risen (lift 0.5 to 0.95), so `y = x^2` visibly turns into its wall. 1D and 2D switch with the plain staggered crossfade. Layers with no visible opacity are not drawn. The renderer caches the GPU buffers of each geometry (`GeoKey`), so a still scene and the layers of a tween upload once instead of every frame.
- 2D/1D to 3D: the 3D scene is drawn with its height scaled about z = 0 by the camera's lift (`Rig::lift`, 0 in 1D/2D, 1 in 3D, eased with the camera), so surfaces and the 3D box grow out of the plane and a curve's extruded sheet starts as the curve itself. 3D to 2D/1D flattens it back. z tick labels follow the same lift.
- `screen_labels_json()` entries have an `alpha` (1 except mid-switch): both scenes' labels are returned while switching, each at the opacity of its grid or items, so the old tick labels fade out as the new ones fade in. The shell sets the CSS opacity from it.
- A switch during a switch starts from the pose on screen, so the camera never jumps.
- With `setReducedMotion` on there is no animation and no blend: the switch is immediate, and the 3D scene is still built first-look-then-slices so the command does not stall.
