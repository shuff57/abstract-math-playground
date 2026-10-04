# Roadmap and known limitations

A plain list of what does not work yet or has not been checked. Items marked "from the code" are stated in source comments or error messages; items marked "not verified" are things the repo has no evidence for either way.

## Rendering and platform

- **WebGPU is not used.** The WASM build asks for the WebGL2 backend only (`web.rs`). wgpu 0.19's WebGPU backend reads `maxInterStageShaderComponents`, which current Chrome removed, so it fails at adapter request. Moving to WebGPU needs a wgpu upgrade (and the API changes that come with it for `render.rs`, `headless.rs`, `native.rs`, `demo.rs` and `web.rs`). From the code.
- **Browser matrix not tested.** Only the WebGL2 path exists. Which browsers and GPUs work has not been recorded. Not verified.
- **Native on each OS not checked.** The native app uses wgpu's primary backends (`Backends::PRIMARY`) and winit 0.29. The repo records no results for Linux, Windows or macOS beyond whatever the author used. Not verified.
- **Mobile and touch not checked.** The web shell handles pointer events and a two-finger pinch (`canvas.ts`), and the engine itself takes `pointer` and `wheel` commands. Whether this works well on real phones and tablets, and whether the layout fits small screens, has not been recorded. Not verified.
- **`demo.rs` and `--demo`** are the original wgpu demo, kept alongside the calculator. They have TODO comments in the source and are not part of the calculator.
- **Release build size and load time** of the WASM package have not been measured or optimised.

## Precision

- **GPU fields are f32.** Hue fields, inequality fills and complex domain colouring are evaluated in f32 in a shader. Near the origin this is fine. At deep zoom far from 0 (window span below about 1e-5 of the origin's size) the coordinate runs out of mantissa and the field shows banding or blockiness. The CPU contours and curves are f64 and stay exact. From the code (`field.wgsl`). A fix would be to evaluate in the shader relative to the window centre with a double-single scheme, or to fall back to a CPU raster when the zoom is deep. Not started.
- **3D surfaces keep f32 vertices.** Mesh vertices from `math_core::mesh` are f32 and window-absolute, so far from the origin at extreme zoom a 3D surface has only f32 precision before it is re-based. From the code (`scene.rs`).
- **GPU results match the CPU only to f32 precision**, and shader compilers may assume no NaN or infinity (`wgsl.rs`). Exactly on a complex branch cut the sheet drawn on the GPU may differ from the CPU.

## Math coverage

- **Complex items lack inverse functions.** There is no `asin`, `acos`, `atan` or inverse hyperbolic functions in complex mode, and no `cbrt`, `floor`, `round`, `min` or `max`. Complex trig is radians only. Complex items cannot use `sum`, `int` or `prod`.
- **No 3D region fills.** In 3D, an inequality that uses only `x` and `y` is filled on the z=0 plane; one that uses `z` draws only its boundary surface (`scene.rs`). There is no filled volume.
- **Scalar fields use `x` and `y` only.** A field expression that mentions `z` is reported as an error.
- **Integrals, sums and products have no GPU form.** In fields and inequalities they are rasterised on the CPU at coarse resolution (about 50 ms budget). Statistical distribution functions are CPU-only too.
- **`int` and `sum`/`prod` limits:** no indefinite integrals; `sum` and `prod` give NaN past 1,000,000 terms; a list holds at most 10,000 elements.
- **Lists are flat.** Lists cannot contain lists (so no matrices as nested lists).
- **No `!=` or `≠`**, and no `%` operator (`mod(a, b)` exists).
- **Multi-letter names parse as products** (`ab` is `a*b`), and a name that starts with a built-in function name is read as that function. Use subscripts for names. This is a parser design choice, not a bug, but it differs from some other calculators.
- **Symbolic differentiation has gaps.** A term that cannot be differentiated becomes NaN or an error ("cannot differentiate ..."). Which functions are covered is defined in `calculus.rs`; this has not been listed here.
- **Regression** supports one dependent list and the fit methods described in `regress.rs` (linear least squares, Levenberg-Marquardt). Other fit types, weights, and residual plots beyond ticks are not built.

## Slices and modes

- Slices exist for 3D (plane or line) and 2D (line). There is no slice for a 1D document.
- The same expression is drawn differently per mode with no automatic conversion (a curve in 2D, a surface in 3D, roots on the number line in 1D). Some item kinds draw nothing in 1D (for example `histogram`).

## Web shell

- No automated tests for the TypeScript shell (`package.json` has no test script).
- Accessibility work exists (announcements, `aria` on sliders) but has not been audited against a checklist.
- Share links are a URL hash (`#v1.`) holding a deflated document; very large documents hit the 200,000 character hash limit.

## Documentation gaps

- The README's older setup text predates the second binary; `cargo run` inside `math_playground` needs `--bin math_playground_bin` because there is no `default-run`.
- `syntax-examples.txt` at the repo root is design notes for a possible future language (structs, conversions); none of it is implemented. It is not the calculator's input syntax. See [syntax.md](syntax.md) for that.
