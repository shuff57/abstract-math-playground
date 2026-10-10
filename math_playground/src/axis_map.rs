//! Logarithmic 2D axes as a map between WORLD coordinates (the numbers in the document) and
//! DISPLAY coordinates (what the camera rig, the scene geometry and the labels use).
//!
//! On a logarithmic axis `display = k * log10(world)`; on a linear axis `display = world`. The
//! rig keeps working linearly (pan, zoom, equal scale on both display axes), so a logarithmic
//! axis pans and zooms by decades. `k` stretches a logarithmic axis so that the document's
//! world window fills the canvas on both axes (the rig always shows equal display scales); it
//! is chosen when the window or the scales are set (see [`AxisMap::for_view`]) and stays fixed
//! while the user pans and zooms.
//!
//! Curves are drawn by substituting the map into the expression ([`AxisMap::display_expr`]):
//! `y = f(x)` on a logarithmic x axis becomes `Y = f(10^(X/k))`, so the existing samplers work
//! in display space unchanged and keep their adaptive refinement.

use math_core::ast::{BinOp, Expr};
use math_core::doc::{ViewState, WindowBox};
use math_core::view::Window3;

const LN10: f64 = std::f64::consts::LN_10;

/// World <-> display map of the two 2D axes (identity on linear axes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisMap {
    /// Which of x and y are logarithmic.
    pub log: [bool; 2],
    /// Display units per decade on a logarithmic axis (unused on a linear one).
    pub k: [f64; 2],
    /// Display units per world unit on a LINEAR axis: 1 for equal scales, otherwise the axes
    /// have different scales (`freeAspect` views, where the typed y range is honoured).
    pub scale: [f64; 2],
}

impl Default for AxisMap {
    fn default() -> Self {
        AxisMap::LINEAR
    }
}

impl AxisMap {
    pub const LINEAR: AxisMap = AxisMap {
        log: [false, false],
        k: [1.0, 1.0],
        scale: [1.0, 1.0],
    };

    /// True when display coordinates ARE world coordinates: no logarithmic axis and no stretch.
    /// (Every place that treats the two as the same asks this.)
    pub fn is_linear(&self) -> bool {
        !self.log[0] && !self.log[1] && self.scale == [1.0, 1.0]
    }

    /// The map for `view` on a canvas of aspect `aspect` (width / height): the view's world
    /// window maps onto a display window whose x span / y span equals the aspect, so the camera
    /// shows the whole window on both axes. A window that is not valid for the scales (a
    /// logarithmic axis with min <= 0) gives the linear map.
    pub fn for_view(view: &ViewState, aspect: f64) -> AxisMap {
        let log = [view.x_scale.is_log(), view.y_scale.is_log()];
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        if !log[0] && !log[1] {
            // Linear axes: equal scales, unless the view asks for its y range to be honoured.
            if view.free_aspect {
                let w = &view.window;
                let (sx, sy) = (w.max[0] - w.min[0], w.max[1] - w.min[1]);
                if sx.is_finite() && sy.is_finite() && sx > 0.0 && sy > 0.0 {
                    let s = sx / (aspect * sy);
                    if s.is_finite() && s > 0.0 && (s - 1.0).abs() > 1e-9 {
                        return AxisMap { scale: [1.0, s], ..AxisMap::LINEAR };
                    }
                }
            }
            return AxisMap::LINEAR;
        }
        if view.check_log_window(None).is_err() {
            return AxisMap::LINEAR;
        }
        let w = &view.window;
        let span = |a: usize| {
            if log[a] {
                w.max[a].log10() - w.min[a].log10()
            } else {
                w.max[a] - w.min[a]
            }
        };
        let (sx, sy) = (span(0), span(1));
        if !(sx.is_finite() && sy.is_finite() && sx > 0.0 && sy > 0.0) {
            return AxisMap::LINEAR;
        }
        // Display spans must satisfy k_x * sx = aspect * k_y * sy; a linear axis has k = 1.
        let k = if log[0] && !log[1] {
            [aspect * sy / sx, 1.0]
        } else {
            [1.0, sx / (aspect * sy)]
        };
        if k.iter().all(|v| v.is_finite() && *v > 0.0) {
            AxisMap { log, k, scale: [1.0, 1.0] }
        } else {
            AxisMap::LINEAR
        }
    }

    /// World -> display on axis `a` (0 = x, 1 = y; other axes are unchanged). Non-positive
    /// values on a logarithmic axis give NaN or -inf (drawn as nothing).
    pub fn fwd(&self, a: usize, v: f64) -> f64 {
        if a < 2 && self.log[a] {
            if v > 0.0 {
                self.k[a] * v.log10()
            } else {
                f64::NAN
            }
        } else if a < 2 {
            v * self.scale[a]
        } else {
            v
        }
    }

    /// Display -> world on axis `a`.
    pub fn inv(&self, a: usize, d: f64) -> f64 {
        if a < 2 && self.log[a] {
            10f64.powf(d / self.k[a])
        } else if a < 2 {
            d / self.scale[a]
        } else {
            d
        }
    }

    pub fn fwd3(&self, p: [f64; 3]) -> [f64; 3] {
        [self.fwd(0, p[0]), self.fwd(1, p[1]), p[2]]
    }

    pub fn inv3(&self, p: [f64; 3]) -> [f64; 3] {
        [self.inv(0, p[0]), self.inv(1, p[1]), p[2]]
    }

    /// The display window of a world window.
    pub fn to_display(&self, w: &WindowBox) -> Window3 {
        Window3::new(self.fwd3(w.min), self.fwd3(w.max))
    }

    /// The world window of a display window.
    pub fn to_world(&self, w: Window3) -> WindowBox {
        WindowBox {
            min: self.inv3(w.min),
            max: self.inv3(w.max),
        }
    }

    /// World value of display variable `name` on axis `a` (`10^(X/k)` as `exp(X ln10 / k)`).
    fn world_of(&self, a: usize, name: &str) -> Expr {
        let v = Expr::var(name);
        if self.log[a] {
            Expr::call(
                "exp",
                vec![Expr::bin(BinOp::Mul, Expr::num(LN10 / self.k[a]), v)],
            )
        } else if self.scale[a] != 1.0 {
            Expr::bin(BinOp::Div, v, Expr::num(self.scale[a]))
        } else {
            v
        }
    }

    /// Display value of the world expression `e` on axis `a` (`k log10 e` as `k/ln10 ln e`).
    pub fn display_of(&self, a: usize, e: Expr) -> Expr {
        if a < 2 && self.log[a] {
            Expr::bin(
                BinOp::Mul,
                Expr::num(self.k[a] / LN10),
                Expr::call("ln", vec![e]),
            )
        } else if a < 2 && self.scale[a] != 1.0 {
            Expr::bin(BinOp::Mul, Expr::num(self.scale[a]), e)
        } else {
            e
        }
    }

    /// `e` with the world variables `x` and `y` replaced by their values in terms of the display
    /// coordinates, so that evaluating it at a display point gives `e` at the world point.
    pub fn display_expr(&self, e: &Expr) -> Expr {
        let mut out = e.clone();
        for (a, n) in [(0usize, "x"), (1, "y")] {
            if (self.log[a] || self.scale[a] != 1.0) && out.contains_var(n) {
                out = out.subst(n, &self.world_of(a, n));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math_core::doc::{AxisScale, Doc};

    fn view(xs: AxisScale, ys: AxisScale, min: [f64; 2], max: [f64; 2]) -> ViewState {
        let mut v = Doc::new_default().view;
        v.x_scale = xs;
        v.y_scale = ys;
        v.window.min[..2].copy_from_slice(&min);
        v.window.max[..2].copy_from_slice(&max);
        v
    }

    #[test]
    fn linear_is_identity_and_invalid_windows_stay_linear() {
        let v = Doc::new_default().view;
        assert_eq!(AxisMap::for_view(&v, 1.5), AxisMap::LINEAR);
        let bad = view(AxisScale::Log, AxisScale::Linear, [-1.0, -5.0], [10.0, 5.0]);
        assert!(AxisMap::for_view(&bad, 1.5).is_linear());
        assert_eq!(AxisMap::LINEAR.fwd(0, -3.0), -3.0);
    }

    #[test]
    fn log_window_fills_the_canvas_and_round_trips() {
        let v = view(
            AxisScale::Log,
            AxisScale::Linear,
            [0.01, -5.0],
            [100.0, 5.0],
        );
        let m = AxisMap::for_view(&v, 2.0);
        let d = m.to_display(&v.window);
        let (sx, sy) = (d.max[0] - d.min[0], d.max[1] - d.min[1]);
        assert!((sx / sy - 2.0).abs() < 1e-12, "{sx} {sy}");
        let back = m.to_world(d);
        for a in 0..2 {
            assert!((back.min[a] - v.window.min[a]).abs() < 1e-12 * v.window.max[a].abs().max(1.0));
            assert!((back.max[a] - v.window.max[a]).abs() < 1e-9);
        }
        assert!(m.fwd(0, 0.0).is_infinite() || m.fwd(0, 0.0).is_nan());
        assert!(m.fwd(0, -1.0).is_nan());
        // Both logarithmic: x keeps one display unit per decade.
        let v2 = view(AxisScale::Log, AxisScale::Log, [1.0, 1.0], [1e4, 1e2]);
        let m2 = AxisMap::for_view(&v2, 1.0);
        assert_eq!(m2.k[0], 1.0);
        assert!((m2.k[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn free_aspect_stretches_y_so_the_typed_range_fills_the_canvas() {
        let mut v = view(AxisScale::Linear, AxisScale::Linear, [-10.0, -1000.0], [10.0, 1000.0]);
        // off: equal scales, the identity
        assert_eq!(AxisMap::for_view(&v, 2.0), AxisMap::LINEAR);
        v.free_aspect = true;
        let m = AxisMap::for_view(&v, 2.0);
        assert!(!m.is_linear() && !m.log[0] && !m.log[1]);
        assert_eq!(m.scale[0], 1.0);
        let d = m.to_display(&v.window);
        // display spans keep the canvas aspect: 20 wide, 10 tall
        assert!(((d.max[0] - d.min[0]) / (d.max[1] - d.min[1]) - 2.0).abs() < 1e-12);
        let back = m.to_world(d);
        for a in 0..2 {
            assert!((back.min[a] - v.window.min[a]).abs() < 1e-9);
            assert!((back.max[a] - v.window.max[a]).abs() < 1e-9);
        }
        // a window that already has the canvas aspect is the identity (no needless stretch)
        let eq = {
            let mut e = view(AxisScale::Linear, AxisScale::Linear, [-10.0, -5.0], [10.0, 5.0]);
            e.free_aspect = true;
            e
        };
        assert_eq!(AxisMap::for_view(&eq, 2.0), AxisMap::LINEAR);
        // expressions follow: y = 100 x is Y = s * 100 * (X / 1) on the stretched axis
        let e = m.display_of(1, m.display_expr(&Expr::bin(
            BinOp::Mul,
            Expr::num(100.0),
            Expr::var("x"),
        )));
        let p = math_core::compile::compile(&e, &["x"], math_core::compile::Angle::Rad).unwrap();
        assert!((p.eval(&[3.0]) - m.fwd(1, 300.0)).abs() < 1e-9);
        // log axes win over the flag
        let l = {
            let mut l = view(AxisScale::Log, AxisScale::Linear, [1.0, -5.0], [100.0, 5.0]);
            l.free_aspect = true;
            l
        };
        assert!(AxisMap::for_view(&l, 2.0).log[0]);
    }

    #[test]
    fn display_expr_substitutes_world_coordinates() {
        let v = view(AxisScale::Log, AxisScale::Log, [1.0, 1.0], [1e3, 1e3]);
        let m = AxisMap::for_view(&v, 1.0);
        let e = m.display_of(1, m.display_expr(&Expr::var("x")));
        let p = math_core::compile::compile(&e, &["x"], math_core::compile::Angle::Rad).unwrap();
        // y = x on log-log axes is the line Y = X.
        for xd in [0.0, 0.5, 2.0] {
            assert!((p.eval(&[xd]) - xd).abs() < 1e-9);
        }
    }
}
