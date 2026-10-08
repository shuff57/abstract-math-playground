# Input syntax

This describes what the parser in `math_core` accepts, checked against `parse.rs`, `ast.rs`, `analyze.rs` and the unit tests. The input can be plain text (`x^2/2`) or the LaTeX that MathLive produces (`\frac{x^{2}}{2}`); both go through the same parser. Examples below use plain text.

If a rule here disagrees with the code, the code is right. Please fix this file.

## Operators

| Operator | Meaning |
| --- | --- |
| `+ - * /` | arithmetic. `·`, `×` also multiply, `÷` divides, `−` (U+2212) is minus |
| `^` | power, right associative: `2^3^2` is `2^(3^2)`. `2^-x` works |
| `x y`, `2x`, `(x+1)(x-1)` | implicit multiplication |
| `n!` | factorial, binds tighter than `*`, `/`, `^` and unary minus: `2^3!` is `2^(3!)`, `-n!` is `-(n!)` |
| `\|x\|` | absolute value (can nest: `\|\|x\|-1\|`) |
| `= < <= > >= !=` | relations. `≤`, `≥`, `≠` also accepted. `!=` is only a condition (see "Not equal" below), not a region |
| `a % b` | modulo, exactly `mod(a, b)`: the result has the sign of `b` (`-7 % 3` is 2). `\%` from MathLive works |
| `~` | regression (see below) |

Precedence, loosest to tightest: relations (including `!=`), `+ -`, `* / %` and implicit multiplication, prefix minus, function argument without parentheses, `^`. `%` is left associative at the level of `*` and `/`: `2*7%4` is `(2*7)%4` and `-7%3` is `(-7)%3`. Parentheses nest to a depth of 128; deeper input is an error, not a crash.

Numbers: `3`, `.5`, `1.5`, `1e3`. The exponent form needs digits right after the `e`, so `2e` is `2*e`.

### Not equal: `!=`

`!` immediately followed by `=` (no space) is the relation not-equal, so `n!=3` means `n != 3`, `5!=120` is true, and `x≠3` / `x\ne 3` are the same. A factorial that is then compared needs a space or brackets: `n! = 120`, `(n!)=120` or `factorial(n)=120`. `n!` on its own, `(n+1)!`, `3!!` and `2^3!` are unchanged.

`!=` is a condition, not a region or a curve. It works in piecewise conditions (`y={x!=2: x}` leaves a hole at x=2, `{x%2=0: 1, 0}` is a parity test) and in list masks (`L[L!=3]`). As an item on its own (`x!=3`) it reports "a '!=' comparison ... use it as a condition inside a piecewise", it cannot be chained (`1<x!=3`) and a `{...}` restriction range cannot use it (a hole is `y=x {x!=0}`, which reads as a piecewise condition because it is not a range). A comparison with an undefined operand (`sqrt(-4)!=1`) is false.

## Functions

The full list is `BUILTIN_FUNCS` in `math_core/src/ast.rs`:

- Trig: `sin cos tan sec csc cot asin acos atan arcsin arccos arctan sinh cosh tanh atan2`. `arcsin`/`arccos`/`arctan` are aliases of `asin`/`acos`/`atan`. `sin^2 x` means `(sin x)^2`; `sin^-1(x)` means `asin(x)`.
- Exponential and roots: `exp ln log sqrt cbrt`. `log` is base 10 (`F1::Log10` in `compile.rs`). `√` is accepted as `sqrt`.
- Rounding and sign: `abs floor ceil round sign sgn min max mod`. `sgn` is an alias of `sign`. `min`/`max` take two numbers, or one list.
- Lists and statistics: `length count total mean median var varp stdev stdevp mad quartile quantile sort reverse join unique corr cov`.
- Statistical plots: `histogram boxplot dotplot` (drawn, not evaluated to a number).
- Distributions: `normalpdf normalcdf invnorm binompdf binomcdf poissonpdf poissoncdf uniformpdf uniformcdf tpdf tcdf invt`.
- Calculus: `deriv int sum prod`.
- Combinatorics: `factorial nCr nPr`.

A function can be called as `sin(x)`, `sin x`, or `sinx`. A function name is recognised wherever a run of letters starts with it (longest match wins), so `sin2x` is `sin(2)*x`.

Anything you define yourself with `f(x)=...` is also callable (see Definitions).

## Constants

`pi` (or `π`), `e`, `tau` (or `τ`). `theta` (or `θ`) is a reserved name used by polar curves. In complex items, `i` is the imaginary unit. Greek letters beyond those three are not special.

## Names and the "multi-letter names are products" rule

A name that is not a function or one of `pi`, `tau`, `theta` is split into single letters multiplied together: `ab` is `a*b`, `xy` is `x*y`, `ax` is `a*x`. To get a multi-character variable use a subscript: `x_1`, `y_1`, `k_12`. Consequences:

- `total` is the function, not `t*o*t*a*l`; a bare `total` with no argument is an error. `vara` parses as `var(a)`. Avoid naming variables so that they start with a builtin function name.
- `for` is only a keyword inside `[...]`; elsewhere it is `f*o*r`.
- `f(x)` is `f*(x)` unless `f` has been defined as a function in the document (see below).

## Definitions and sliders

```
a = 3
f(x) = x^2 - a
g(x, y) = x*y
```

`a = 3` defines a number, `f(x) = ...` a function. Other items can use them. A definition's value shows in the list; the web UI offers a "Make slider for a" chip for undefined names, which creates a slider with range -10..10. A slider is a named number with `min`, `max`, optional `step` and a `value` (set it with the `setSlider` command or `--slider name=value` in `render_png`). Dragging a slider re-draws every item that uses its name.

```
y = a*sin(x)          (with a slider named a)
f(x) = x^2
y = f(x) + 1
```

A drawn point `(a, b)` or definition `a=3` can be dragged on the canvas; the engine then rewrites the item's text (the `itemEdited` event).

## What an equation means

The analyser (`analyze.rs`) picks a kind for each item:

| Input | Kind |
| --- | --- |
| `y = f(x)` | explicit curve (2D); a surface when 3D mode and `z = f(x,y)` |
| `x = f(y)` | explicit in x |
| `z = f(x, y)` | surface |
| `r = f(theta)` | polar curve |
| `x^2+y^2=9`, `x^2+y^2+z^2=36` | implicit curve / surface |
| `y < x^2`, `x^2+y^2<=4` | inequality region |
| `(cos(t), sin(t))`, `(cos(t), sin(t), t/4)` | parametric curve, 2D or 3D (uses `t`) |
| `x=3cos(t), y=2sin(t)` | the same parametric curve as `(3cos(t), 2sin(t))`; also as separate rows |
| `(cos(u)cos(v), sin(u)cos(v), sin(v))` | parametric surface (3 components using both `u` and `v`, 3D mode) |
| `(1, 2)`, `(1, 2, 3)` | point |
| `(-y, x)` | vector field (tuple of x, y(, z) without `t`), 1 to 3 components |
| `[1, 2, 3]` | list |
| `y_1 ~ a x_1 + b` | regression |
| `sin(x)+y`, `x^2+y^2` (no `=`) | scalar field coloured over the plane |
| `2+3`, `int(x^2, x, 0, 2)` | value (shown as a read-out) |

Parametric `t` runs over `[0, 2*pi]` (`[0, 360]` in degree mode) unless the item gives a range (next section). Polar `theta` runs over `[0, 4*pi]` when theta only appears inside trig functions, else `[0, 6*pi]` (for spirals); degree-mode equivalents in degree mode. Scalar fields can only use `x` and `y`; a field that mentions `z` is reported as an error.

### How curves and surfaces are drawn where they are awkward

* **Degenerate shapes that touch zero without changing sign are drawn.** `(x-0.37)^2+(y-0.21)^2=0` is a dot, `(x-y)^2=0` and `x^2=0` are the (double) line, `(x^2+y^2-4)^2=0` is the circle. In 3D a point quadric `(x-a)^2+(y-b)^2+(z-c)^2=0` is a small ball (about one grid cell across, so it is visible), a line quadric `(x-a)^2+(y-b)^2=0` is a thin tube, and a squared factor `(x^2+y^2+z^2-9)^2=0` is the surface itself, once. A feature smaller than one grid cell (`x^2+y^2=0.00000001`) is also drawn as a dot or ball instead of vanishing.
* **Graphs `y=f(x)`, `x=f(y)` and `z=f(x,y)` break where the function does.** No segment or sheet joins the two sides of a pole (`tan(x)`, `1/x`, `1/(x^2+y^2)`) or a jump (`floor(x)`, `sign(x)`: only the treads are drawn), and a surface ends exactly on the edge of its domain (`z=sqrt(16-x^2-y^2)` has a clean rim, no ragged edge). `y=f(x,z)` and `x=f(y,z)` surfaces behave the same. The same break is applied to implicit surfaces, such as `z=tan(x)` written as `tan(x)-z=0`, but there the cut edge is up to one grid cell ragged.
* **Surfaces are fast.** `z=f(x,y)` is sampled on a grid (one evaluation per node, finer on steep walls) instead of searched in 3D; implicit surfaces share corner values between cells and find each crossing in a few evaluations.

Examples:

```
y = x^2/4
r = 1 + cos(theta)
(3 cos(t), 2 sin(t))
```

## Chained inequalities

Up to three comparisons in a row, using only `<`, `<=`, `>`, `>=`:

```
0 <= y <= x^2
1 < x^2+y^2 <= 4
a < x < b
```

`=` inside a chain is an error, more than three parts is an error ("at most three"), and a chain cannot be combined with `~`.

## Lists, ranges, indexing, comprehensions

```
L = [1, 2, 3]
[1...10]                 range 1 to 10
[1, 3...11]              range with step (1, 3, 5, ...)
L[2]                     1-based index
L[2...4]                 slice by range
L[[1,3]]                 pick by list
L[L > 3]                 filter with a comparison
[n^2 for n=[1...5]]      comprehension
[2k for k=L]
```

Rules checked in `list.rs`: indexing is 1-based; a single out-of-range or non-integer index gives NaN; operators and functions broadcast over lists elementwise; lists hold numbers or points, not other lists (`lists cannot be nested`); a list is capped at 10,000 elements and evaluation has a total work budget (5,000,000 steps), after which you get an error. `L[1]` indexes only when `[` follows directly with no space and not after a number (`2[1,2,3]` and `L [1,2]` are products). A comparison only evaluates inside an index.

Lists of points: `[(0,0), (1,2), (2,1)]`. `join(A, B, ...)` joins lists (or point lists).

## Statistics and distributions

Aggregates take a list: `length(L)`, `count(L)`, `total(L)`, `mean(L)`, `median(L)`, `var(L)`, `varp(L)`, `stdev(L)`, `stdevp(L)`, `mad(L)`, `min(L)`, `max(L)`, `corr(X, Y)`, `cov(X, Y)`. `quantile(L, p)` and `quartile(L, q)` use linear interpolation (R type 7) and accept a list as the second argument. `sort(L)`, `sort(L, K)` (sort L by keys K), `reverse(L)`, `unique(L)`.

Plots: `histogram(L)` or `histogram(L, binwidth)`, `dotplot(L)` / `dotplot(L, binwidth)`, `boxplot(L)`. Drawn on the z=0 plane in 2D and 3D, nothing in 1D.

Distributions (argument order from `compile.rs` and `stats.rs`):

| Call | Meaning |
| --- | --- |
| `normalpdf(x, mu, sigma)` | density |
| `normalcdf(a, b, mu, sigma)` | P(a < X < b) |
| `invnorm(p, mu, sigma)` | inverse CDF |
| `binompdf(n, p, k)`, `binomcdf(n, p, k)` | binomial |
| `poissonpdf(lambda, k)`, `poissoncdf(lambda, k)` | Poisson |
| `uniformpdf(x, a, b)`, `uniformcdf(a, b, lo, hi)` | uniform |
| `tpdf(x, df)`, `tcdf(a, b, df)`, `invt(p, df)` | Student t |

Invalid parameters (for example `sigma <= 0`) give NaN. Examples:

```
normalcdf(-1, 1, 0, 1)
y = normalpdf(x, 0, 1)
histogram(x_1)
```

## Regression `~`

`y_1 ~ a x_1 + b` fits the model on the right to the list on the left. Names in the model that are not lists, sliders or definitions are the parameters. A model linear in its parameters is solved by least squares (QR); otherwise Levenberg-Marquardt is used. The item shows the fitted parameters, R squared and RMSE. While the item is visible, the fitted parameters are defined as numbers other items can use. Residual ticks can be turned on with `setRegressionResiduals`.

```
y_1 ~ a x_1 + b
y_1 ~ a x_1^2 + b x_1 + c
y_1 ~ a sin(x_1) + b cos(x_1)
```

Lists can come from a typed list or from table columns (below).

## Derivatives

```
d/dx x^3                  Leibniz form, operand is a product term
d^2/dx^2 sin(x)           second derivative (order 1 to 8)
f'(2)                     derivative of f evaluated at 2
f''(x)
deriv(x^3, x)             function form
deriv(x^3, x, 2)          derivative at x = 2
```

`d/dx 3x^2 + 1` differentiates only `3x^2`. The derivative is computed symbolically and simplified; the simplified result is shown as a read-out. An expression that cannot be differentiated gives a NaN term or an error ("cannot differentiate ..."). In degree mode the derivative of `sin(x)` carries a factor `pi/180`.

## Integrals, sums, products

```
int(x^2, x, 0, 2)         function form: int(body, var, lo, hi)
sum(n^2, n, 1, 10)
prod(n, n, 1, 5)
\int_{0}^{1} x^2 dx       LaTeX / MathLive form
\sum_{n=1}^{5} n^2
\prod_{n=1}^{5} n
```

Both limits are required; indefinite integrals are an error. A sum or product body is the following product-level term: `\sum_{n=1}^3 n+1` is `(sum of n) + 1`. Integrals use adaptive Gauss-Kronrod with a fallback for endpoint singularities and give NaN if divergent. `sum` and `prod` iterate integer bounds and give NaN beyond 1,000,000 terms.

Limits: `int`, `sum` and `prod` have no GPU form. When used inside a field or inequality they are rasterised on the CPU into coarse quads (time budget about 50 ms per raster, between 1,500 and 250,000 cells), so the result is lower resolution than a GPU field. Curves and surfaces already run on the CPU. They are not available in complex items (an error says so). Statistical distribution functions are also not supported in GPU shaders (`wgsl.rs` rejects them).

## Factorial, nCr, nPr

```
5!            120
(n+1)!
nCr(5, 2)     also \binom{5}{2}
nPr(n, k)
x^n/n!
```

`factorial(n)` is the same as `n!`.

## Complex mode

An item of kind `complex` is parsed with the complex-aware parser and drawn as a domain colouring of the plane, with `z` as the variable (hue is the argument, brightness the modulus). The web shell and the document format use the item kind `complex`; the `c:` prefix is a convenience of the `render_png` tool only (`render_png "c:z^2-1"`).

- Variable: `z`. Constants: `i`, `pi`, `e`, `tau`. Sliders can be used as real parameters.
- Operators: `+ - * / ^`. Integer powers up to 64 in size use exact repeated multiplication; others use `exp(w ln z)`.
- Functions: `sin cos tan sec csc cot sinh cosh tanh exp ln log sqrt abs arg re im conj`.
- Trig is always in radians in complex items (no degree mode).
- `ln` and `sqrt` use principal branches.

```
z^2 - 1
sin(z)
(z-1)/(z^2+z+1)
```

Not available: inverse trig and inverse hyperbolic functions (`asin`, `acos`, `atan` and friends), `cbrt`, `floor`/`round`, `min`/`max`, lists, relations, `sum`, `int`, `prod`. These are reported as unknown functions or not-scalar errors.

## Parametric curves and surfaces

### Ranges: a trailing `{ ... }`

A group of comparisons at the END of an item restricts its parameter (Desmos' domain restriction):

```
(t^2, 2t) {-3<=t<=3}                          both halves of a parabola
(2cos(t), 2sin(t), t/4-6) {0<=t<=12pi}        a helix of six turns
r = theta/3 {0<=theta<=24}                    a spiral over a chosen theta range
(cos(t), sin(t)) {0<=t<=a pi}                 a slider moves the end of the arc
(u, v, u v) {0<=u<=1, 0<=v<=1}                a surface patch
(cos(t), t) {t>=-2}                           one end only: the other keeps its default
```

* The group is `{...}` (LaTeX `\left\{ ... \right\}` works too) holding one or more comparisons separated by `,` or `;`. Each is `a<=t<=b`, `b>=t>=a` (strict `<`/`>` are the same here), or one-sided `t<=b` / `t>=a`. The parameter is `t` for parametric curves, `theta` for polar curves and `u`, `v` for surfaces; a range on another name is a diagnostic (`... does not apply here`).
* The bounds are ordinary expressions of constants, `pi`, sliders and definitions. A slider name that does not exist yet is the usual `undefined variable` diagnostic, so the app offers a slider for it. In degree mode the bounds are degrees (`{0<=t<=180}`).
* A one-sided range moves only that end: `t>=a` runs `[a, 2pi]` (or `[a, a+2pi]` when `a` is already past `2pi`); `t<=b` runs `[0, b]` (or `[b-2pi, b]` when `b` is not above 0). Polar curves use their own default span in the same way.
* An empty or non-finite range (`{3<=t<=1}`) is a diagnostic and draws nothing.
* A `{...}` group is only taken as a range when it is last, is not a `^{...}`/`_{...}` script, and contains a comparison. On a parametric curve, polar curve or parametric surface it restricts the parameter; on `y=f(x)`, `x=g(y)`, an implicit equation, an inequality or a bare expression of `x` it restricts the curve itself (see "Restricting x or y" below). On any other item (for example a number, or `z=f(x,y)`) it is reported (`a {range} applies to ...`).
* Long ranges get more samples (a curve over `k` default turns uses `k` times as many, up to 120,000 points). Closed and open arcs need nothing special.

### Restricting x or y: `y=x^2 {x>0}`

```
y=x^2 {x>0}                  right half of the parabola
y=x^2 {-1<x<2}               only between x=-1 and x=2
x=y^2 {y>=0}                 the upper branch of a sideways parabola
x^2+y^2=4 {y>0}              the upper half circle
y<x {x>0}                    the region y<x, but only where x>0
y=sin(x) {0<=x<=2pi, y>0}    bounds on both axes
```

* Bounds are on `x` and/or `y` (a range on another name is a diagnostic). Bounds are expressions of constants, `pi`, sliders and definitions. The curve is clipped to the box, so no sample lies outside it; hover, click-to-select and the special points of a selected curve respect the clip too.
* Endpoints are marked as in Desmos: a strict bound (`<`, `>`) gets an OPEN circle where the curve ends, a non-strict one (`<=`, `>=`, `≤`, `≥`) a FILLED dot. The markers use the item's colour and point size and are drawn with the ordinary point renderer. A curve that crosses a bound (a circle cut at `y>0`) gets a marker at each crossing; a bound at which the function is undefined (`y=1/x {x>0}`) gets none.
* An inequality fill is cut to the box. The edge of the box itself is not drawn, only the inequality's own boundary curve, clipped.
* An empty box (`{x>2, x<1}`) is a diagnostic and draws nothing. Restrictions apply in 2D with linear axes; in 3D or on logarithmic axes they are reported and ignored.

### `x=`, `y=`, `z=` pairs

`x=3cos(t), y=2sin(t)` in one row (separated by `,` or `;`, any order) is exactly the tuple `(3cos(t), 2sin(t))`; with `z=...` a space curve. Separate rows `x=3cos(t)` and `y=2sin(t)` (and `z=t/4`) are fused into one curve too: the first visible row of each axis is used, the rows must use `t` (or `u` and `v` for a surface) and not x, y or z on the right, and the fused curve takes the colour and style of the x row (a range may be written on any of the rows). `x=2`, `y=3` (no parameter) stay two lines.

### Parametric surfaces

`(x(u,v), y(u,v), z(u,v))`, three components that use both `u` and `v` and no `t`, is a surface in 3D mode:

```
(3cos(u)cos(v), 3cos(u)sin(v), 3sin(u))                          sphere
((3+cos(v))cos(u), (3+cos(v))sin(u), sin(v))                     torus
(v cos(u), v sin(u), u/2) {-6<=u<=6, -5<=v<=5}                   helicoid
((2+v cos(u/2))cos(u), (2+v cos(u/2))sin(u), v sin(u/2)) {-1<=v<=1}   Moebius strip
(v cos(u), v sin(u), v) {0<=v<=3}                                cone
(a u, v, u^2 - v^2) {-2<=u<=2, -2<=v<=2}                         `a` becomes a slider
```

* `u` and `v` default to `[0, 2*pi]` each (`[0, 360]` in degrees). Note that a full `[0, 2pi]` range of the sphere formula above covers the sphere twice; `{-pi/2<=u<=pi/2}` covers it once.
* `u` and `v` are always the surface parameters, even if a slider of that name exists. Other free names are sliders as usual.
* It is drawn with the same lighting, two-sided shading, per-item colour and opacity as other surfaces, and is clipped to the window box (triangles are cut at the box faces).
* The grid has at least 96 x 96 cells and refines itself from the surface's size in the window (up to 640 per direction and about 180,000 vertices, fewer while a preview is built). Cells with an undefined corner are left out (holes), triangles that leap across a discontinuity are dropped, and poles and cone apexes get a sound normal. Closed surfaces share their seam and pole vertices (no cracks, no open edges); the Moebius strip's seam keeps two normals but its edges meet in space.
* In 2D mode a parametric surface draws nothing and reports nothing (the same as `z=f(x,y)` in 2D: the same expression is drawn per mode, see ROADMAP). In 1D it draws nothing.
* A curve or surface in a tuple with both `t` and `u`/`v` is a curve in `t`; `(u, 2)` and `(a, b, c)` stay points. Not supported: intersection curves of two surfaces, filled volumes, slicing a parametric surface with the slice plane.

## Piecewise: `{x<0: -x, x}`

```
y={x<0: -x, x}                    absolute value: the last value has no condition and is the default
y={x<0: -x, x>=0: x}              the same with two conditions
y={x<-1: x, x>1: x}               nothing is defined for -1<=x<=1: the graph has a gap there
y={0<x<1: 5, 7}                   a chained condition
y=2{x<0: -x, x}+1                 pieces are ordinary values: they can be added, multiplied, ...
y={x<0: -x, x} {x>-3}             a piecewise value, restricted by a range
```

* A `{...}` group holding `condition: value` entries separated by `,`. The first entry whose condition holds gives the value; a last entry with no condition is the default; with no default and no matching condition the value is undefined (NaN), so a curve has a gap. A condition alone (`{x>0}`) is 1 where it holds and undefined elsewhere. LaTeX `\left\{ ... \right\}` works (MathLive writes `\lbrace`/`\rbrace` for a typed brace).
* How `{...}` is read: a group is a range (restriction) when it is last, is not a `^{...}`/`_{...}` script, has a comparison and has NO `:` outside brackets. A group with a `:` is piecewise, wherever it stands. A group without a `:` that has a comparison but is not a range (`{x>0}` inside a longer expression) is a piecewise too. A group with neither (`{x}`) is plain parentheses.
* Conditions are comparisons (`<`, `<=`, `>`, `>=`, `=`) or chains of them; a condition that is undefined (NaN) counts as false. Every branch is evaluated and then selected, so a branch that is not taken may be undefined without harm (`{x>0: sqrt(x), 0}`).
* Where it works: curves (`y={...}`, `x={...}`, bare expressions of `x`), implicit curves, inequality fills and fields. Fills and fields that contain a piecewise are drawn by the CPU raster (the shader has no reliable NaN), at the same coarse resolution as `sum` and `int`. Not supported: complex items (reported as an unknown function) and the GPU path used by plain fills. A jump at a branch boundary is joined by a near-vertical line when the jump is smaller than the view height, unlike Desmos, which leaves a gap.
* Breakpoint markers: an explicit curve `y={...}` marks where its branches end. At a strict end of a branch (`y={x<-1:-x, x>1:x}`) an OPEN circle sits at the one-sided limit, (-1,1) and (1,1); at a non-strict end (`x<=-1`) the point is on the curve and gets a FILLED dot. A jump between two branches (`y={x<0:x^2, x^2+1}`) is open at the end of one branch and filled at the start of the other. Edges whose limit is infinite or outside the view get none, and functions without a piecewise (`sqrt(x)`, `ln(x)`, `1/x`) get no markers.

## Tuples and vector fields

```
(1, 2)                     point
(1, 2, 3)                  3D point
(cos(t), sin(t))           parametric (has t); `(cos(t), sin(t)) {0<=t<=pi}` limits t
(cos(u), sin(u), v)        parametric surface (3 components with u and v)
(-y, x)                    2D vector field, arrows
(x, y, z)                  3D vector field
```

A vector field is drawn as arrows on a grid whose spacing is chosen to keep the arrow count bounded (in 3D at most 12 lattice points per axis).

## Tables

A table item has columns named like list variables (`x_1`, `y_1`, up to 16 characters), up to 12 columns, 500 rows and 120 characters per cell. Each column defines a list with its name, so `histogram(x_1)` or `y_1 ~ a x_1 + b` work as with a typed list. The first two columns are plotted as points. Style is `points` (default), `line` (points joined in row order) or `hidden` (not drawn; the lists stay defined). Tables are edited with the table commands in [engine-api.md](engine-api.md).

## Actions and ticker

An action is a comma-separated list of assignments `target -> expression`; the arrow may also be `→`, `\to`, `\rightarrow` or `\mapsto`.

```
a -> a + 1
a -> a + 1, b -> 2b
n -> mod(n + 1, 10)
```

Targets are existing slider names or plain numeric definitions (`a=3`). The assignments are simultaneous: every right side is evaluated against the old values, and then all results are applied. Run one with `runAction`. The ticker runs an action repeatedly: `setTicker` with `action`, `rate_ms` (default 50 ms, never faster than `min_step_ms`, default 10) and `running`. A ticker error stops it by default (`pause_on_error`). At most 4 ticker steps run per rendered frame; a longer stall drops the backlog.

## Slices

A slice shows a lower-dimensional cross-section of the same items in an inset panel. In 3D, fixing one axis (`z = 1`) gives a plane (2D slice); fixing two (`y = 1, z = 0.5`) gives a line (1D slice). In 2D, fixing one axis (`y = a`) gives a line. The constant is a number or an expression of slider names, so moving a slider sweeps the slice.

```
z = 1
y = 1, z = 0.5
y = a
```

Set it with `setSlice` (or `--slice` in `render_png`). The inset has its own view window; dragging or the wheel on the inset changes it, a double-click resets it to follow the main window.

## Angle mode

`rad` (default) or `deg`. In degree mode trig functions take degrees and inverse trig returns degrees, and the default parametric `t` (and `u`, `v`) range becomes 0 to 360 (ranges you write are in degrees too). Set it with `setAngle` (`--angle deg` in the native app and `render_png`). Complex items ignore it.

## Comments and notes

Items of kind `note` and `folder` carry text and grouping only; they are not parsed as math.

## Known limits

- Multi-letter names are products (see above); use subscripts for named variables.
- No indefinite integrals and no nested lists. `!=` is a condition only (not a region), `%` is modulo.
- Input length is capped at 2000 characters per item; a document holds at most 500 items and 200 sliders, and the JSON is limited to 100,000 bytes.
- Fields use `x` and `y` only; `z` is not supported in a field expression.
- `int`/`sum`/`prod` in fields are drawn by a coarse CPU raster, and are not available in complex items.
- GPU fields are computed in f32, so they match the CPU only to about 7 digits and can break up at deep zoom far from the origin (see [ROADMAP.md](ROADMAP.md)).
- Complex items have no inverse trig, and exactly on a branch cut the sheet drawn on the GPU may differ from the CPU.
- Statistical distribution functions are CPU-only.
