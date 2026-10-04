# Abstract Math Playground

Right now this project does not have a clear purpose or goal except to be a playground where I explore what makes me curious. It lives in the same realm as [Mr. Pullen's Graphing Calculator](https://calculator.mrpullen.com/). If you're interested in these kinds of things and want to discuss more, join us on Discord!

[![Join the Graphing Calculator Creators Discord server](https://invidget.switchblade.xyz/sgqwmkUQhQ)](https://discord.gg/sgqwmkUQhQ)

# Installation & Usage

To be honest, I don't remember all the exact installation steps. If someone installs this and can write down what the steps should be, that would be lovely.

Generally speaking, you want to do this:

## Install dependencies

- Install rust/cargo on your machine (I used rustup)
- Install node/npm (if you want to run the web version as well)

## Run the native version

```bash
cd math_playground
cargo run
```

## Run the web version

```bash
cd math_playground_web
npm run wasm-dev # Builds the wasm/webgpu code from `math_playground` once, then runs the vite dev server
```

# Graphing calculator

This branch adds a Desmos-style graphing calculator on top of the playground. You type expressions and they are drawn in 1D, 2D or 3D. Switching between the three modes is animated (the scene cross-fades while the camera moves), and the same expressions are kept across modes. It is written in Rust with [wgpu](https://wgpu.rs/) and [winit](https://github.com/rust-windowing/winit), runs natively and in the browser through WASM, and the web version has a TypeScript shell that uses [MathLive](https://cortexjs.io/mathlive/) for math input.

What it handles (details and examples in [docs/syntax.md](docs/syntax.md)): equations, inequalities, parametric curves, polar curves, points, lists and statistics, regression, complex-valued functions (domain colouring), derivatives, integrals/sums/products, sliders, actions and a ticker, data tables, vector fields, and slices (a lower-dimensional cross-section of the same items).

Other documents:

- [docs/syntax.md](docs/syntax.md): input syntax and its limits.
- [docs/engine-api.md](docs/engine-api.md): the JSON command/event protocol and the WASM `Calculator` methods.
- [docs/ROADMAP.md](docs/ROADMAP.md): known limitations and things not built yet.

## Architecture

The workspace has three crates/packages (see `Cargo.toml` at the root for the Rust ones):

| Part | What it is |
| --- | --- |
| `math_core/` | Pure-logic engine. No GPU, window or browser dependencies, so it unit-tests natively and compiles to WASM unchanged. Parsing (text and a LaTeX subset), analysis (what kind of item an expression is), compiling to bytecode and to WGSL, lists, statistics, regression, calculus, complex numbers, tables, slices, meshing, the camera/view model, and the versioned document format with share-link encoding. |
| `math_playground/` | The application. `app.rs` holds the platform-independent state (document, camera, theme, input handling, throttled rebuilds) and is driven by JSON commands in and JSON events out. `scene.rs` turns the document into drawable geometry, `render.rs` draws it with wgpu. It builds as a native app (`native.rs`, winit window), a headless PNG renderer (`headless.rs`, `bin_png.rs`), and a WASM library (`web.rs`, a `Calculator` class for JavaScript). `demo.rs` is the older standalone demo. |
| `math_playground_web/` | Vite + TypeScript shell. Loads the WASM package, owns the canvas, the expression list, MathLive input, sliders, tables, the slice panel and the tick-label overlay. It only talks to the engine through the JSON protocol. |

The native window and the web page both go through `App::dispatch` and `App::frame`, so they share behaviour.

## Build and run

You need a Rust toolchain (rustup). `~/.cargo/bin` must be on your `PATH` (cargo, rustc and wasm-pack all live there). For the web version you also need `wasm-pack` and Node.js with npm. The repo does not declare a Node version (`package.json` has no `engines` field); the web build uses Vite 5 and TypeScript 5, so use a Node version that Vite 5 supports (Node 18 or newer).

### Native

`math_playground` has two binaries, `math_playground_bin` (the window) and `render_png` (headless), and no `default-run`, so name the binary:

```bash
cargo run -p math_playground --bin math_playground_bin -- "y=x^2" "(1,1)" --mode 2d
```

Usage, from `bin.rs`:

```
math_playground_bin [EXPR...] [--mode 1d|2d|3d] [--dark] [--hash V1..] [--doc FILE.json] [--angle deg|rad] [--demo]
```

- `EXPR...` expressions to add on start-up.
- `--mode` starting mode.
- `--dark` dark theme.
- `--hash` load a share hash (the `v1.` string).
- `--doc` load a document from a JSON file.
- `--angle` angle unit for trig, `deg` or `rad`.
- `--demo` run the older wgpu demo instead of the calculator.

Keys in the native window (from `native.rs`): `1`, `2`, `3` switch mode, `d` toggles the theme, `o` toggles orthographic in 3D, `r` resets the view, `Esc` quits. Mouse: drag pans in 1D and 2D and orbits in 3D (shift-drag or a non-left button pans), the wheel zooms, and dragging a point item moves it.

### Headless PNG (`render_png`)

Renders offscreen to PNG so output can be checked without a window. It needs a usable wgpu adapter. Options are parsed by hand in `bin_png.rs`; value flags are:

```
--mode 1d|2d|3d        mode to render (default 2d)
--size WxH             image size (default 900x600)
--out FILE             output file (default out.png)
--dark                 dark theme (flag, no value)
--angle deg            degree mode (only "deg" is checked; anything else is radians)
--window a,b,c,d,e,f   view box xmin,ymin,zmin,xmax,ymax,zmax (default -10..10 on each axis)
--slider name=value    set a slider (range fixed to -10..10); repeatable
--from MODE --to MODE  render a mode transition as frames
--frames N             number of transition frames (default 6)
--out-dir DIR          directory for transition frames (default frames/)
--table "x_1=1,2,3;y_1=2,4,3"   add a data table; with --table-line the points are joined
--ticker "a -> a+0.5"  run a ticker action; writes OUT_before.png and OUT_after.png
--ticker-frames N      ticker steps to run (default 8, 50 ms each)
--slice "z=1"          add a slice (also "y=a", "y=1,z=0.5")
--slice-view a,b,c,d   the slice inset's own window xmin,xmax,ymin,ymax
```

Prefix an expression with `c:` to make it a complex item (domain colouring). Examples:

```bash
cargo run -p math_playground --bin render_png -- --mode 3d --out sphere.png "x^2+y^2+z^2=36"
cargo run -p math_playground --bin render_png -- --from 2d --to 3d --frames 6 --out-dir frames/ "y=x^2/4" "(3,2.25)"
cargo run -p math_playground --bin render_png -- --mode 2d "c:z^2-1"
```

### Web

```bash
cd math_playground_web
npm install
npm run wasm          # wasm-pack build --target web --dev, output in ../math_playground/pkg
npm run dev           # vite dev server
```

Scripts in `package.json`:

- `npm run wasm`: debug build of the WASM package (`wasm-pack build --target web --dev`, run in `math_playground/`).
- `npm run wasm-release`: the same with `--release`. Use this for anything you want to run at a reasonable speed or deploy.
- `npm run wasm-dev`: `wasm` then `vite` in one step (what the older text above refers to).
- `npm run dev`: Vite dev server only (needs the WASM package already built).
- `npm run build`: `tsc && vite build`, output in `dist/`.
- `npm run preview`: serve the built `dist/`.

The TypeScript imports the WASM package from `../../math_playground/pkg/`, so `npm run wasm` (or `wasm-release`) must run before `dev` or `build`. A page can open a shared document with a `#v1....` URL hash.

### Tests

```bash
cargo test --workspace
```

This runs the unit tests in `math_core` and `math_playground` (no separate `tests/` directories exist). The TypeScript side has no test script.

### Browser support

The WASM build requests only the WebGL2 backend (`wgpu::Backends::GL` in `web.rs`). WebGPU is not used yet: wgpu 0.19's WebGPU backend traps on current Chrome because it reads a limit (`maxInterStageShaderComponents`) that Chrome removed. Using WebGPU needs a wgpu upgrade. After startup, `Calculator.backend_name()` reports which backend was picked.
