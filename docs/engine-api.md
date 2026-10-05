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

Sliders:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setSlider` | `name`, `value`, optional `min`, `max`, `step` | Creates the slider if needed (range defaults to -10..10). |
| `removeSlider` | `name` | |

View and theme:

| `t` | Fields | Notes |
| --- | --- | --- |
| `setMode` | `mode` | `"1d"`, `"2d"` or `"3d"`. Starts the animated transition (see below). Anything else gives an `error` event. |
| `setOrtho` | `ortho` | Orthographic camera in 3D. |
| `setReducedMotion` | `on` | Mode switches and the ortho toggle jump to the end instead of animating. Send it from `prefers-reduced-motion` at start-up and when it changes (the TS shell does). Default off. |
| `setTheme` | `dark` | |
| `setAngle` | `angle` | `"deg"`, otherwise radians. |
| `resize` | `width`, `height` | Physical pixels. |
| `reset` | none | Resets the camera. |

Input (pixel coordinates from the canvas top-left, physical pixels):

| `t` | Fields | Notes |
| --- | --- | --- |
| `pointer` | `phase`, `x`, `y`, optional `button` (default 0), `shift` (default false) | `phase` is `down`, `move`, `up`, `cancel` or `dblclick` (a double-click on the slice inset resets its view). Left-drag pans in 1D/2D, orbits in 3D; shift or another button pans. A press on a point item grabs it and edits its text (see `itemEdited`). |
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
| `setSlice` | `fixed` (aliases `plane`, `axis`, `at`, `axes`), optional `dim` | `fixed` is an object `{"z":0.5}` / `{"y":1,"z":"a"}` or a string `"y=1,z=0.5"`. Values are numbers or expressions of slider names. `dim` is derived from the mode and the number of fixed axes; a value that disagrees is an error. |
| `clearSlice` | none | |
| `setSliceView` | optional `min` `[xmin,ymin]`, `max` `[xmax,ymax]`, `view` `[xmin,xmax,ymin,ymax]` | Sets the inset's own window; it stops following the main window. Error if there is no slice. |
| `resetSliceView` | none | Back to auto-follow. |

## Events

Every event has a `t` field.

| `t` | Fields | Meaning |
| --- | --- | --- |
| `diagnostics` | `items`: `[{id, message}]` | Per-item problems (parse errors, unsupported constructs). |
| `info` | `items`: `[ItemInfo]` | Read-outs: derivative, scalar value, regression fit. The full current list, sent whenever it changes; an empty list clears them. |
| `colors` | `items`: `[{id, color}]` | Resolved `#rrggbb` colour of each drawn item. Sent only when it changes. |
| `view` | `mode`, `min` `[x,y,z]`, `max` `[x,y,z]` | Current mode and window. |
| `labels` | `labels`: `[{pos:[x,y,z], text, axis}]` | Tick labels in world coordinates. |
| `doc` | `json` | Document JSON string (reply to `export`). |
| `hash` | `hash` | Share hash (reply to `export`). |
| `table` | `id`, `columns`: `[{name, cells}]`, `rows`, `style` | Full table state, after each change and once per table on load. |
| `itemEdited` | `id`, `latex` | A drag rewrote an item's text (a point or a definition); put `latex` in that item's input. |
| `sliderValue` | `name`, `value` | A drag or action moved a slider. |
| `theme` | `dark` | |
| `slice` | `active`, `dim`, `fixed` (map axis to number), `free` (axis names), `rect` `[x,y,w,h]` or null, `curves`, `points`, `error`, `view` `[xmin,xmax,ymin,ymax]` or null, `follow` | Slice state, sent when it changes (also while a slider sweeps it). `active` is false with no slice or one that does not fit the mode (`error` says why). `rect` is the inset in canvas pixels. |
| `tickerState` | `running`, `action` | Ticker started, stopped (also automatically after an error) or reconfigured. |
| `error` | `message` | A problem with a command. Ticker problems are prefixed `ticker: `. |

`ItemInfo`: `id`, `kind` (`"derivative"`, `"value"` or `"regression"`), and as they apply `latex`, `text`, `value`, `params` (`[{name, value, stdError?}]`), `r2`, `rmse`, `n`. Absent fields are omitted.

`screen_labels_json()` (below) returns the labels already projected to screen: `{text, axis, x, y, visible, inset, clip?}`. `inset` is true for slice-inset labels (their `x`/`y` are already offset into the canvas), `clip` is the `[x,y,w,h]` rectangle the label must stay inside. Axis 0 and 1 are tick labels and axis names; 3 is reserved for a title.

## Document format

`loadDoc` and `export` use a versioned JSON document (`math_core/src/doc.rs`, current `v` is 1):

```json
{
  "v": 1,
  "view": { "mode": "2d", "window": { "min": [-10,-10,-10], "max": [10,10,10] }, "angle": "rad" },
  "items": [ { "id": "a", "kind": "equation", "latex": "y=x^2", "hidden": false, "color": "#ff0000",
               "style": { "lineWidth": 2.0, "lineStyle": "dotted" } } ],
  "sliders": { "k": { "min": 0, "max": 1, "step": 0.5, "value": 0.25 } },
  "ticker": { "action": "i6", "rateMs": 125 },
  "slice": { "dim": 1, "fixed": { "y": "a" } }
}
```

Optional item fields: `hidden`, `color`, `style` (`lineWidth`, `lineStyle` `solid|dashed|dotted`, `pointStyle` `dot|circle|cross|square`, `opacity`, `label`, `residuals`), `folder`, `table` (for `table` items). `view.theme` is optional. `sliders`, `ticker` and `slice` are optional. Unknown JSON fields are ignored; unknown enum values are errors.

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
| `screen_labels_json()` | `string` | JSON array of tick labels projected to screen pixels, for an HTML overlay. `"[]"` before the first scene exists. |
| `backend_name()` | `string` | The wgpu backend picked, from `adapter.get_info().backend` (for example `Gl`). |

Typical loop:

```js
function tick(now) {
  const drawn = calc.frame(now);
  const events = JSON.parse(calc.drain());
  handle(events);
  if (drawn) overlay(JSON.parse(calc.screen_labels_json()));
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
```

The native build does not use `Calculator`; `native.rs` calls `App::dispatch` and `App::frame` directly.

### Mode transitions

`setMode` moves the camera to the new mode over 500 ms (ease-in-out cubic) while the outgoing scene fades out and the incoming one fades in. How it behaves with the frame loop:

- The transition's clock starts at the first `frame` after the command, not at the command. A loop may stop calling `frame` while nothing changes (the native shell does, and so may an embedding that sleeps its `requestAnimationFrame`), and the scene for the new mode is built inside `setMode`; neither eats into the animation. The first frame after the switch shows the starting pose; every later one moves.
- Switching into 3D builds a preview first: implicit surfaces are meshed at a coarser octree depth (`scene::PREVIEW_SURFACE_DEPTH`), so motion starts within a frame or two even when the full 3D scene takes 100 ms or more. The full-quality scene is rebuilt in the first frame after the switch has finished, and only once the pointer has been still for 100 ms (an orbit straight after the switch is not interrupted). That rebuild costs what the switch itself used to cost, but it happens on a still picture. Leaving 3D needs no preview (1D/2D scenes are cheap).
- 2D/1D to 3D: the 3D scene is drawn with its height scaled about z = 0 by the camera's lift (`Rig::lift`, 0 in 1D/2D, 1 in 3D, eased with the camera), so surfaces and the 3D box grow out of the plane and a curve's extruded sheet starts as the curve itself. 3D to 2D/1D flattens it back. z tick labels follow the same lift.
- A switch during a switch starts from the pose on screen, so the camera never jumps.
- With `setReducedMotion` on there is no animation: the switch is immediate and builds the full scene directly.
