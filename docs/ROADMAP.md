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

## Meshing and degenerate shapes

- **Touches of higher order or without curvature are only found when lucky.** The degenerate-shape search (a damped Newton minimiser plus the zero set of `w . grad F`, see `mesh_touch.rs`, `mesh_contour.rs`, `mesh_surface.rs`) needs `F` to touch zero like a square. `x^4+y^4=0` converges slowly and may draw nothing, and a non-smooth touch such as `abs(x)+abs(y)=0` is drawn only when the minimum falls on a grid corner. Dimension mixes beyond a point, a line and a surface (a double curve in 3D that is not a line quadric, found by tracing, can lose a branch at a crossing) are not covered.
- **A point or line quadric is drawn thicker than it is.** The ball and the tube have a fixed size of about 0.8 / 0.4 grid cells, so they are visible; they are not to scale when zoomed far in.
- **Poles and jumps in implicit surfaces cost a ragged cell.** The break is decided on a grid edge (a function value that does not shrink with the bracket), so the cut edge is up to one cell off the true discontinuity. Explicit graphs `z=f(x,y)` break exactly (interval enclosure for poles, a jump test for steps). A steep but continuous cliff (a very sharp sigmoid) can be mistaken for a jump and shown as a gap.
- **Implicit surfaces that end on a domain edge** (`sqrt`, `ln`, ...) still have a ragged rim of up to one cell, because a cell with an undefined corner is skipped. Only explicit height fields are trimmed to the exact edge.
- **A height field is one value per grid point.** A surface that folds back over itself in `z` (a graph cannot) is not what `z=f(x,y)` means anyway; thin features narrower than a cell (a sharp ridge) are smoothed, where the old implicit search would have kept them at cell resolution. Steep quads are refined (up to 8x), not everything.

## Parametric forms

- **Slices ignore parametric ranges.** The slice overlay (`slice_draw.rs`) finds where a parametric or polar curve crosses the slice plane over the default parameter span, not over a `{a<=t<=b}` range, and parametric surfaces are not cut by a slice plane at all.
- **Parametric surface sampling is a uniform grid.** The resolution is chosen from the surface size in the window (96 to 640 cells per direction, about 180,000 vertices at most) but is not adaptive per region, so a very fast-varying surface can still alias, and an asymptote is only caught by the long-edge guard.
- **A full `[0, 2pi]` range may cover a surface twice** (the sphere formula does), which draws two coincident sheets with opposite normals; write the range to cover it once.
- **No intersection curves of two surfaces and no volume fills** (also in the 3D list above).

## Intersections

- **Grid resolution.** Implicit/implicit and parametric/parametric intersections use a 160x160 grid over the window (or over the parameter ranges). Two crossings closer than a cell, a loop smaller than a cell, or a near miss tighter than the refinement tolerance can be missed or merged. Tangent touches are found but converge linearly, so they are good to about 1e-8, not full precision.
- **Partner and point caps.** The selected curve is checked against at most 8 other visible curves (document order), 48 points per pair and 48 points in all with the other special points. Intersection points beyond the window are not listed.
- **Overlaps** (identical curves, or a stretch where two curves coincide) give no points, by design, as in Desmos.
- **Not done:** intersections of a curve with an inequality boundary, 3D curves, and domain clips (`math_core::intersect::Curve::clip` takes one, but no document syntax feeds it yet).

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
