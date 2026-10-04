//! Camera / window / mode-transition model for seamless 1D <-> 2D <-> 3D views of ONE scene.
//!
//! # Model
//! * [`Window3`] (f64) is the single source of truth, shared by every mode. Switching modes never
//!   modifies it; 1D/2D simply do not *show* some of its extents.
//! * Math axes: x right, y up, z toward the viewer in 2D; z is "up" in 3D.
//! * 2D: orthographic, eye on +z looking toward -z, up=+y. The window's **x range is exact**; the
//!   shown y range is derived from the viewport aspect: `cy +- halfx/aspect` (pixels stay square).
//!   Interactions in 2D write that derived y range back to the window.
//! * 1D: same camera as 2D but vertically centred on y=0 (the window's y/z are kept in storage).
//! * 3D: perspective orbit camera (vertical FOV 50 deg, z up) around the window centre.
//!
//! # Projection blend
//! The camera is `(q, H, ortho_t, strip)`: orientation, half-height `H` of the *focal plane*
//! (eye distance `d = H / tan(fov/2)`), blend `ortho_t` in [0,1], and 1D strip factor. The
//! projection is `lerp(perspective, orthographic)` where both share identical x/y rows (the
//! ortho matrix is scaled by `d` so its clip w is `d`), i.e. the focal plane keeps its framing for
//! every `ortho_t`. At `ortho_t == 1` the matrix is exactly orthographic (w row = `[0,0,0,d]`).
//! A mode switch and the 3D perspective/ortho toggle are the same tween mechanism.
//!
//! # Matrix convention
//! `view_proj` is **column-major** (`m[col][row]`), maps **origin-relative** world coordinates
//! `p - render_origin` (render origin = window centre, computed in f64) to wgpu clip space
//! (x,y in [-1,1] after divide, depth in [0,1]; the OpenGL->wgpu fix `z' = (z + w)/2` is applied).
//! Use as `clip = view_proj * vec4(p - origin, 1)`.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

pub type Mat4 = [[f64; 4]; 4];

const MIN_HALF: f64 = 1e-30;
const MAX_HALF: f64 = 1e30;
const MAX_PITCH: f64 = PI / 2.0 - 0.01;

/// Axis-aligned window shared by all modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window3 {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Window3 {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Window3 { min, max }.sanitized()
    }
    /// Non-finite values replaced by +-1, min/max ordered.
    pub fn sanitized(mut self) -> Self {
        for i in 0..3 {
            let mut lo = if self.min[i].is_finite() { self.min[i] } else { -1.0 };
            let mut hi = if self.max[i].is_finite() { self.max[i] } else { 1.0 };
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
            self.min[i] = lo;
            self.max[i] = hi;
        }
        self
    }
    pub fn centre(&self) -> [f64; 3] {
        let mut c = [0.0; 3];
        for (i, v) in c.iter_mut().enumerate() {
            *v = self.min[i] / 2.0 + self.max[i] / 2.0;
        }
        c
    }
    /// Half extents, clamped to a sane positive range so downstream maths stays finite.
    pub fn half(&self) -> [f64; 3] {
        let mut h = [0.0; 3];
        for (i, v) in h.iter_mut().enumerate() {
            *v = (self.max[i] / 2.0 - self.min[i] / 2.0).clamp(MIN_HALF, MAX_HALF);
        }
        h
    }
    fn set_centre_half(&mut self, axis: usize, c: f64, h: f64) {
        self.min[axis] = c - h;
        self.max[axis] = c + h;
    }
}

impl Default for Window3 {
    fn default() -> Self {
        Window3 { min: [-10.0; 3], max: [10.0; 3] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    D1,
    D2,
    D3,
}

impl Mode {
    pub fn dims(self) -> u32 {
        match self {
            Mode::D1 => 1,
            Mode::D2 => 2,
            Mode::D3 => 3,
        }
    }
}

/// GPU uniform. `repr(C)`, 96 bytes. `view_proj` column-major, origin-relative, wgpu depth.
/// `origin` = window centre (f32 rounding for info only; the CPU rebases in f64).
/// `scale` = window half-extents xyz, and w = focal-plane half-height `H` (world units).
/// `mode` = 1, 2 or 3 (target mode).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub origin: [f32; 4],
    pub scale: [f32; 4],
    pub mode: u32,
    pub ortho_t: f32,
    pub _pad: [f32; 2],
}

/// Aspect-independent-ish snapshot of the interpolated camera.
#[derive(Clone, Copy, Debug)]
pub struct CamState {
    /// Camera-to-world orientation, [x,y,z,w].
    pub q: [f64; 4],
    /// Focal-plane half-height in world units.
    pub h: f64,
    pub ortho_t: f64,
    /// 0 = normal, 1 = 1D strip (view centred on y=0).
    pub strip: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Tween {
    pub from: CamState,
    pub start_ms: f64,
    pub duration_ms: f64,
}

/// Ease-in-out cubic. Clamped; ease(0)=0, ease(1)=1, zero slope at both ends.
pub fn ease(t: f64) -> f64 {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

// ---- quaternion helpers ([x,y,z,w]) ----
type Q = [f64; 4];
const Q_ID: Q = [0.0, 0.0, 0.0, 1.0];

fn qnorm(q: Q) -> Q {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 && n.is_finite() {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        Q_ID
    }
}

fn slerp(a: Q, mut b: Q, t: f64) -> Q {
    let mut d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if d < 0.0 {
        d = -d;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    let (wa, wb) = if d > 0.9995 {
        (1.0 - t, t)
    } else {
        let th = d.clamp(-1.0, 1.0).acos();
        let s = th.sin();
        (((1.0 - t) * th).sin() / s, (t * th).sin() / s)
    };
    qnorm([
        a[0] * wa + b[0] * wb,
        a[1] * wa + b[1] * wb,
        a[2] * wa + b[2] * wb,
        a[3] * wa + b[3] * wb,
    ])
}

/// Columns of the rotation matrix: camera x (right), y (up), z (toward viewer) in world coords.
fn axes(q: Q) -> [[f64; 3]; 3] {
    let [x, y, z, w] = qnorm(q);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)],
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn quat_from_cols(c: [[f64; 3]; 3]) -> Q {
    // m[row][col] = c[col][row]
    let m = |r: usize, k: usize| c[k][r];
    let tr = m(0, 0) + m(1, 1) + m(2, 2);
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, 0.25 * s]
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        [0.25 * s, (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s, (m(2, 1) - m(1, 2)) / s]
    } else if m(1, 1) > m(2, 2) {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        [(m(0, 1) + m(1, 0)) / s, 0.25 * s, (m(1, 2) + m(2, 1)) / s, (m(0, 2) - m(2, 0)) / s]
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        [(m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s, 0.25 * s, (m(1, 0) - m(0, 1)) / s]
    };
    qnorm(q)
}

/// Orbit orientation: eye direction `(cos p cos y, cos p sin y, sin p)` from the centre, z up.
fn orbit_quat(yaw: f64, pitch: f64) -> Q {
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let dir = [cp * cy, cp * sy, sp];
    let xa = {
        let v = [-dir[1], dir[0], 0.0];
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-12);
        [v[0] / n, v[1] / n, 0.0]
    };
    let ya = [
        dir[1] * xa[2] - dir[2] * xa[1],
        dir[2] * xa[0] - dir[0] * xa[2],
        dir[0] * xa[1] - dir[1] * xa[0],
    ];
    quat_from_cols([xa, ya, dir])
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn f32c(x: f64) -> f32 {
    if x.is_finite() {
        (x as f32).clamp(-f32::MAX, f32::MAX)
    } else {
        0.0
    }
}

fn clamp_aspect(a: f64) -> f64 {
    if a.is_finite() && a > 0.0 {
        a.clamp(1e-6, 1e6)
    } else {
        1.0
    }
}

fn viewport_ok(v: (f64, f64)) -> Option<(f64, f64)> {
    if v.0.is_finite() && v.1.is_finite() && v.0 > 0.0 && v.1 > 0.0 {
        Some(v)
    } else {
        None
    }
}

fn lerp_state(a: &CamState, b: &CamState, e: f64) -> CamState {
    let ratio = b.h / a.h;
    CamState {
        q: slerp(a.q, b.q, e),
        h: if ratio.is_finite() && ratio > 0.0 { a.h * ratio.powf(e) } else { a.h },
        ortho_t: a.ortho_t + (b.ortho_t - a.ortho_t) * e,
        strip: a.strip + (b.strip - a.strip) * e,
    }
}

/// Camera rig: shared window + orbit state + optional mode tween.
#[derive(Clone, Debug)]
pub struct Rig {
    window: Window3,
    home: Window3,
    mode: Mode,
    yaw: f64,
    pitch: f64,
    /// Dolly multiplier on the default 3D distance (1 = frames the window bounding sphere).
    dist: f64,
    ortho3: bool,
    pub fov_deg: f64,
    aspect: f64,
    pub duration_ms: f64,
    pub reduced_motion: bool,
    tween: Option<Tween>,
    now_ms: f64,
}

pub const DEFAULT_YAW: f64 = -PI / 4.0;
pub const DEFAULT_PITCH: f64 = 0.5;

impl Default for Rig {
    fn default() -> Self {
        Rig::new(Window3::default(), Mode::D2)
    }
}

impl Rig {
    pub fn new(window: Window3, mode: Mode) -> Self {
        let window = window.sanitized();
        Rig {
            window,
            home: window,
            mode,
            yaw: DEFAULT_YAW,
            pitch: DEFAULT_PITCH,
            dist: 1.0,
            ortho3: false,
            fov_deg: 50.0,
            aspect: 1.0,
            duration_ms: 500.0,
            reduced_motion: false,
            tween: None,
            now_ms: 0.0,
        }
    }

    pub fn window(&self) -> Window3 {
        self.window
    }
    /// Replace the shared window (no tween; continuity is the caller's concern).
    pub fn set_window(&mut self, w: Window3) {
        self.window = w.sanitized();
    }
    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn yaw(&self) -> f64 {
        self.yaw
    }
    pub fn pitch(&self) -> f64 {
        self.pitch
    }
    pub fn render_origin(&self) -> [f64; 3] {
        self.window.centre()
    }
    /// Renderer should call this on resize; it is used to retarget tweens (`set_mode`).
    pub fn set_aspect(&mut self, aspect: f64) {
        self.aspect = clamp_aspect(aspect);
    }
    pub fn is_animating(&self) -> bool {
        self.tween.is_some()
    }
    /// Linear time progress of the running tween in [0,1] (1 when idle).
    pub fn progress(&self) -> f64 {
        match &self.tween {
            Some(t) => ((self.now_ms - t.start_ms) / t.duration_ms).clamp(0.0, 1.0),
            None => 1.0,
        }
    }
    pub fn ortho_t(&self, aspect: f64) -> f64 {
        self.current_state(clamp_aspect(aspect)).ortho_t
    }

    /// Advance the clock; finishes the tween when its time is up.
    pub fn update(&mut self, now_ms: f64) {
        if now_ms.is_finite() {
            self.now_ms = now_ms;
        }
        if let Some(t) = &self.tween {
            if self.now_ms - t.start_ms >= t.duration_ms {
                self.tween = None;
            }
        }
    }

    fn target_state(&self, mode: Mode, aspect: f64) -> CamState {
        let hw = self.window.half();
        match mode {
            Mode::D1 | Mode::D2 => CamState {
                q: Q_ID,
                h: (hw[0] / aspect).max(MIN_HALF),
                ortho_t: 1.0,
                strip: if mode == Mode::D1 { 1.0 } else { 0.0 },
            },
            Mode::D3 => {
                let r = (hw[0] * hw[0] + hw[1] * hw[1] + hw[2] * hw[2]).sqrt();
                let half_fov = (self.fov_deg.clamp(1.0, 170.0) / 2.0).to_radians();
                CamState {
                    q: orbit_quat(self.yaw, self.pitch),
                    h: (self.dist * r / half_fov.cos()).max(MIN_HALF),
                    ortho_t: if self.ortho3 { 1.0 } else { 0.0 },
                    strip: 0.0,
                }
            }
        }
    }

    /// Current (possibly mid-tween) camera state for `aspect`.
    pub fn current_state(&self, aspect: f64) -> CamState {
        let aspect = clamp_aspect(aspect);
        let to = self.target_state(self.mode, aspect);
        match &self.tween {
            Some(t) => {
                let p = ((self.now_ms - t.start_ms) / t.duration_ms).clamp(0.0, 1.0);
                lerp_state(&t.from, &to, ease(p))
            }
            None => to,
        }
    }

    /// Capture the current state, apply `f`, then tween from the captured state to the new target.
    fn retarget(&mut self, now_ms: f64, f: impl FnOnce(&mut Self)) {
        self.update(now_ms);
        let from = self.current_state(self.aspect);
        f(self);
        if self.reduced_motion || !self.duration_ms.is_finite() || self.duration_ms <= 0.0 {
            self.tween = None;
        } else {
            self.tween = Some(Tween { from, start_ms: self.now_ms, duration_ms: self.duration_ms });
        }
    }

    /// Start (or retarget) an eased transition to `mode`. Interruptible without any visual jump.
    pub fn set_mode(&mut self, mode: Mode, now_ms: f64) {
        if mode == self.mode {
            self.update(now_ms);
            return;
        }
        self.retarget(now_ms, |r| r.mode = mode);
    }

    /// Perspective/orthographic toggle for 3D (same blend mechanism as the 2D<->3D switch).
    pub fn set_ortho3(&mut self, ortho: bool, now_ms: f64) {
        if ortho == self.ortho3 {
            self.update(now_ms);
            return;
        }
        self.retarget(now_ms, |r| r.ortho3 = ortho);
    }

    pub fn reset(&mut self) {
        self.window = self.home;
        self.yaw = DEFAULT_YAW;
        self.pitch = DEFAULT_PITCH;
        self.dist = 1.0;
        self.ortho3 = false;
        self.tween = None;
    }

    /// 3D orbit (radians). Pitch is clamped so the camera never flips. No-op outside 3D.
    pub fn orbit(&mut self, dyaw: f64, dpitch: f64) {
        if self.mode != Mode::D3 || !dyaw.is_finite() || !dpitch.is_finite() {
            return;
        }
        self.yaw = (self.yaw + dyaw).rem_euclid(2.0 * PI);
        if self.yaw > PI {
            self.yaw -= 2.0 * PI;
        }
        self.pitch = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// 3D dolly (>1 moves closer). Distinct from `zoom_at`, which scales the window.
    pub fn dolly(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.dist = (self.dist / factor.clamp(1e-3, 1e3)).clamp(1e-3, 1e3);
        }
    }

    fn sync_y(&mut self, aspect: f64) {
        let hx = self.window.half()[0];
        let cy = self.window.centre()[1];
        self.window.set_centre_half(1, cy, (hx / aspect).max(MIN_HALF));
    }

    /// Drag the content by (dx, dy) screen pixels (y down).
    pub fn pan_pixels(&mut self, dx: f64, dy: f64, viewport_px: (f64, f64)) {
        let Some((w, h)) = viewport_ok(viewport_px) else { return };
        if !dx.is_finite() || !dy.is_finite() {
            return;
        }
        let aspect = clamp_aspect(w / h);
        self.aspect = aspect;
        match self.mode {
            Mode::D1 | Mode::D2 => {
                let upp = 2.0 * self.window.half()[0] / w;
                if self.mode == Mode::D2 {
                    self.sync_y(aspect);
                    self.window.min[1] += dy * upp;
                    self.window.max[1] += dy * upp;
                }
                self.window.min[0] -= dx * upp;
                self.window.max[0] -= dx * upp;
            }
            Mode::D3 => {
                let st = self.current_state(aspect);
                let upp = 2.0 * st.h / h;
                let a = axes(st.q);
                for i in 0..3 {
                    let d = -dx * upp * a[0][i] + dy * upp * a[1][i];
                    self.window.min[i] += d;
                    self.window.max[i] += d;
                }
            }
        }
        self.window = self.window.sanitized();
    }

    /// `factor > 1` zooms in. 2D: the world point under the cursor stays fixed; 1D: x only;
    /// 3D: window scaled uniformly about its centre (cursor ignored).
    pub fn zoom_at(&mut self, cursor_px: (f64, f64), factor: f64, viewport_px: (f64, f64)) {
        let Some((w, h)) = viewport_ok(viewport_px) else { return };
        if !factor.is_finite() || factor <= 0.0 || !cursor_px.0.is_finite() || !cursor_px.1.is_finite() {
            return;
        }
        let f = factor.clamp(1e-3, 1e3);
        let aspect = clamp_aspect(w / h);
        self.aspect = aspect;
        match self.mode {
            Mode::D1 | Mode::D2 => {
                let p = self.pixel_to_world(cursor_px, viewport_px);
                if self.mode == Mode::D2 {
                    self.sync_y(aspect);
                }
                let axes_n = if self.mode == Mode::D2 { 2 } else { 1 };
                for i in 0..axes_n {
                    let (c, hf) = (self.window.centre()[i], self.window.half()[i]);
                    self.window.set_centre_half(i, p[i] + (c - p[i]) / f, (hf / f).max(MIN_HALF));
                }
            }
            Mode::D3 => {
                let (c, hw) = (self.window.centre(), self.window.half());
                for i in 0..3 {
                    self.window.set_centre_half(i, c[i], (hw[i] / f).max(MIN_HALF));
                }
            }
        }
        self.window = self.window.sanitized();
    }

    /// Top-down (2D/1D) pixel -> world mapping. In 1D the view is centred on y=0. z = window
    /// centre z. (In 3D this is the same top-down mapping, not a ray pick.)
    pub fn pixel_to_world(&self, px: (f64, f64), viewport_px: (f64, f64)) -> [f64; 3] {
        let c = self.window.centre();
        let Some((w, h)) = viewport_ok(viewport_px) else { return c };
        let hx = self.window.half()[0];
        let hy = hx * h / w;
        let cy = if self.mode == Mode::D1 { 0.0 } else { c[1] };
        let u = px.0 / w - 0.5;
        let v = px.1 / h - 0.5;
        [c[0] + u * 2.0 * hx, cy - v * 2.0 * hy, c[2]]
    }

    /// Inverse of [`Rig::pixel_to_world`] (ignores z).
    pub fn world_to_pixel(&self, p: [f64; 3], viewport_px: (f64, f64)) -> (f64, f64) {
        let Some((w, h)) = viewport_ok(viewport_px) else { return (0.0, 0.0) };
        let c = self.window.centre();
        let hx = self.window.half()[0];
        let hy = hx * h / w;
        let cy = if self.mode == Mode::D1 { 0.0 } else { c[1] };
        (((p[0] - c[0]) / (2.0 * hx) + 0.5) * w, (0.5 - (p[1] - cy) / (2.0 * hy)) * h)
    }

    fn view_proj_state(&self, st: &CamState, aspect: f64) -> Mat4 {
        let aspect = clamp_aspect(aspect);
        let hw = self.window.half();
        let r = (hw[0] * hw[0] + hw[1] * hw[1] + hw[2] * hw[2]).sqrt();
        let half_fov = (self.fov_deg.clamp(1.0, 170.0) / 2.0).to_radians();
        let h = st.h.clamp(MIN_HALF, 1e36);
        let d = h / half_fov.tan();
        let ax = axes(st.q);
        // Eye relative to the render origin (= window centre); strip moves the focus to y = 0.
        let c = self.window.centre();
        let focus = [0.0, -st.strip * c[1], 0.0];
        let eye = [focus[0] + ax[2][0] * d, focus[1] + ax[2][1] * d, focus[2] + ax[2][2] * d];
        let mut view = [[0.0; 4]; 4];
        for i in 0..3 {
            view[i] = [ax[i][0], ax[i][1], ax[i][2], -dot3(ax[i], eye)];
        }
        view[3] = [0.0, 0.0, 0.0, 1.0];

        let span = 1.1 * r;
        let (n_o, f_o) = (d - span, d + span);
        let (n_p, f_p) = ((d - span).max(0.01 * d), d + span);
        let t = st.ortho_t.clamp(0.0, 1.0);
        let l = |a: f64, b: f64| a * (1.0 - t) + b * t;
        let sx = d / (h * aspect);
        let sy = d / h;
        let z_row = [
            0.0,
            0.0,
            l((f_p + n_p) / (n_p - f_p), -2.0 * d / (f_o - n_o)),
            l(2.0 * f_p * n_p / (n_p - f_p), -d * (f_o + n_o) / (f_o - n_o)),
        ];
        let w_row = [0.0, 0.0, l(-1.0, 0.0), l(0.0, d)];
        // wgpu depth fix: z' = (z + w) / 2
        let z_fix = [
            0.5 * z_row[0] + 0.5 * w_row[0],
            0.5 * z_row[1] + 0.5 * w_row[1],
            0.5 * z_row[2] + 0.5 * w_row[2],
            0.5 * z_row[3] + 0.5 * w_row[3],
        ];
        let proj = [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], z_fix, w_row];
        mul(&proj, &view)
    }

    /// Row-major internally, returned **column-major** (`m[col][row]`), origin-relative, wgpu depth.
    pub fn view_proj(&self, aspect: f64) -> Mat4 {
        let st = self.current_state(aspect);
        let m = self.view_proj_state(&st, aspect);
        let mut o = [[0.0; 4]; 4];
        for (r, row) in m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                o[c][r] = if v.is_finite() { *v } else { 0.0 };
            }
        }
        o
    }

    /// Project a world point (f64, rebased on the origin in f64) to NDC (x, y, depth in [0,1]).
    /// Returns zeros if the point is at infinity.
    pub fn project_ndc(&self, p: [f64; 3], aspect: f64) -> [f64; 3] {
        let m = self.view_proj(aspect);
        let o = self.render_origin();
        let q = [p[0] - o[0], p[1] - o[1], p[2] - o[2], 1.0];
        let clip: Vec<f64> = (0..4).map(|r| (0..4).map(|c| m[c][r] * q[c]).sum()).collect();
        if clip[3].abs() < 1e-300 || !clip[3].is_finite() {
            return [0.0; 3];
        }
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    pub fn camera_uniform(&self, aspect: f64) -> CameraUniform {
        let m = self.view_proj(aspect);
        let mut vp = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                vp[c][r] = f32c(m[c][r]);
            }
        }
        let st = self.current_state(aspect);
        let o = self.render_origin();
        let hw = self.window.half();
        CameraUniform {
            view_proj: vp,
            origin: [f32c(o[0]), f32c(o[1]), f32c(o[2]), 0.0],
            scale: [f32c(hw[0]), f32c(hw[1]), f32c(hw[2]), f32c(st.h)],
            mode: self.mode.dims(),
            ortho_t: f32c(st.ortho_t),
            _pad: [0.0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win() -> Window3 {
        Window3::new([-4.0, -3.0, -2.0], [4.0, 3.0, 2.0])
    }
    fn mat_close(a: &Mat4, b: &Mat4, tol: f64) -> bool {
        (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j]).abs() <= tol))
    }
    const VP: (f64, f64) = (800.0, 600.0);

    #[test]
    fn pixel_round_trip() {
        let mut r = Rig::new(win(), Mode::D2);
        r.set_aspect(4.0 / 3.0);
        for p in [(0.0, 0.0), (800.0, 600.0), (123.4, 456.7)] {
            let w = r.pixel_to_world(p, VP);
            let b = r.world_to_pixel(w, VP);
            assert!((b.0 - p.0).abs() < 1e-9 && (b.1 - p.1).abs() < 1e-9);
        }
        let w = r.pixel_to_world((0.0, 0.0), VP);
        assert!((w[0] + 4.0).abs() < 1e-12 && (w[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn zoom_keeps_cursor_fixed() {
        for mode in [Mode::D2, Mode::D1] {
            let mut r = Rig::new(win(), mode);
            let cur = (200.0, 150.0);
            let before = r.pixel_to_world(cur, VP);
            r.zoom_at(cur, 1.7, VP);
            let after = r.pixel_to_world(cur, VP);
            assert!((before[0] - after[0]).abs() < 1e-9);
            if mode == Mode::D2 {
                assert!((before[1] - after[1]).abs() < 1e-9);
            }
            assert!((r.window().half()[0] - 4.0 / 1.7).abs() < 1e-9);
        }
    }

    #[test]
    fn pan_exact() {
        let mut r = Rig::new(win(), Mode::D2);
        let upp = 8.0 / 800.0;
        r.pan_pixels(10.0, -20.0, VP);
        let w = r.window();
        assert!((w.min[0] - (-4.0 - 10.0 * upp)).abs() < 1e-12);
        assert!((w.max[1] - (3.0 - 20.0 * upp)).abs() < 1e-12);
        let mut r1 = Rig::new(win(), Mode::D1);
        r1.pan_pixels(10.0, 50.0, VP);
        assert_eq!(r1.window().min[1], -3.0);
    }

    #[test]
    fn ease_sanity() {
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-15);
        assert!(ease(1e-3) < 1e-8 && 1.0 - ease(1.0 - 1e-3) < 1e-8);
        let mut prev = 0.0;
        for i in 1..=100 {
            let v = ease(i as f64 / 100.0);
            assert!(v >= prev);
            prev = v;
        }
        assert_eq!(ease(f64::NAN), 0.0);
    }

    #[test]
    fn tween_endpoints_and_progress() {
        let mut r = Rig::new(win(), Mode::D2);
        r.set_aspect(1.5);
        r.set_mode(Mode::D3, 100.0);
        assert!(r.is_animating());
        let mut prev = -1.0;
        for t in (100..=600).step_by(25) {
            r.update(t as f64);
            assert!(r.progress() >= prev);
            prev = r.progress();
        }
        assert!(!r.is_animating());
        let direct = Rig::new(win(), Mode::D3);
        assert!(mat_close(&r.view_proj(1.5), &direct.view_proj(1.5), 1e-12));
    }

    #[test]
    fn retarget_is_continuous() {
        let mut r = Rig::new(win(), Mode::D2);
        r.set_aspect(1.5);
        r.set_mode(Mode::D3, 0.0);
        r.update(180.0);
        let before = r.view_proj(1.5);
        r.set_mode(Mode::D1, 180.0);
        assert!(mat_close(&before, &r.view_proj(1.5), 1e-9));
        r.update(300.0);
        let before = r.view_proj(1.5);
        r.set_mode(Mode::D3, 300.0);
        assert!(mat_close(&before, &r.view_proj(1.5), 1e-9));
        r.update(300.0);
        r.set_ortho3(true, 400.0);
        r.update(450.0);
        let before = r.view_proj(1.5);
        r.set_ortho3(false, 450.0);
        assert!(mat_close(&before, &r.view_proj(1.5), 1e-9));
    }

    #[test]
    fn rapid_toggle_ends_at_last() {
        let mut r = Rig::new(win(), Mode::D2);
        let modes = [Mode::D3, Mode::D1, Mode::D2, Mode::D3, Mode::D1];
        let mut t = 0.0;
        for i in 0..200 {
            r.set_mode(modes[i % 5], t);
            t += 7.0;
            r.update(t);
            let m = r.view_proj(1.3);
            assert!(m.iter().flatten().all(|v| v.is_finite()));
        }
        let last = modes[199 % 5];
        assert_eq!(r.mode(), last);
        r.update(t + 1000.0);
        assert!(!r.is_animating());
        let direct = Rig::new(win(), last);
        assert!(mat_close(&r.view_proj(1.3), &direct.view_proj(1.3), 1e-12));
    }

    #[test]
    fn reduced_motion_instant() {
        let mut r = Rig::new(win(), Mode::D2);
        r.reduced_motion = true;
        r.set_mode(Mode::D3, 0.0);
        assert!(!r.is_animating());
        assert_eq!(r.mode(), Mode::D3);
        assert!(mat_close(&r.view_proj(1.0), &Rig::new(win(), Mode::D3).view_proj(1.0), 1e-12));
    }

    #[test]
    fn orbit_clamps_pitch() {
        let mut r = Rig::new(win(), Mode::D3);
        r.orbit(0.3, 100.0);
        assert!(r.pitch() <= PI / 2.0 - 0.0099);
        r.orbit(0.0, -1000.0);
        assert!(r.pitch() >= -PI / 2.0 + 0.0099);
        assert!(r.view_proj(1.0).iter().flatten().all(|v| v.is_finite()));
        let mut r2 = Rig::new(win(), Mode::D2);
        let y = r2.yaw();
        r2.orbit(1.0, 1.0);
        assert_eq!(r2.yaw(), y);
    }

    #[test]
    fn ortho_vs_perspective() {
        let mut r = Rig::new(win(), Mode::D3);
        let dir = [
            r.pitch().cos() * r.yaw().cos(),
            r.pitch().cos() * r.yaw().sin(),
            r.pitch().sin(),
        ];
        let p = [1.0, 0.5, 0.2];
        let q = [p[0] + 2.0 * dir[0], p[1] + 2.0 * dir[1], p[2] + 2.0 * dir[2]];
        let a = r.project_ndc(p, 1.0);
        let b = r.project_ndc(q, 1.0);
        assert!((a[0] - b[0]).abs() > 1e-3 || (a[1] - b[1]).abs() > 1e-3);
        r.reduced_motion = true;
        r.set_ortho3(true, 0.0);
        let a = r.project_ndc(p, 1.0);
        let b = r.project_ndc(q, 1.0);
        assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        let m = r.view_proj(1.0);
        assert!(m[0][3] == 0.0 && m[1][3] == 0.0 && m[2][3] == 0.0 && m[3][3] > 0.0);
    }

    #[test]
    fn mode_2d_corners_and_1d_x() {
        let r = Rig::new(win(), Mode::D2);
        let a = 4.0 / 3.0;
        let n = r.project_ndc([4.0, 3.0, 0.0], a);
        assert!((n[0] - 1.0).abs() < 1e-12 && (n[1] - 1.0).abs() < 1e-12);
        let n = r.project_ndc([-4.0, -3.0, 0.0], a);
        assert!((n[0] + 1.0).abs() < 1e-12 && (n[1] + 1.0).abs() < 1e-12);
        assert!(n[2] > 0.0 && n[2] < 1.0);
        let r1 = Rig::new(Window3::new([-4.0, 5.0, -2.0], [4.0, 9.0, 2.0]), Mode::D1);
        let n = r1.project_ndc([4.0, 0.0, 0.0], 2.0);
        assert!((n[0] - 1.0).abs() < 1e-12 && n[1].abs() < 1e-12);
        let n = r1.project_ndc([-4.0, 0.0, 0.0], 2.0);
        assert!((n[0] + 1.0).abs() < 1e-12);
        assert_eq!(r1.window().min[1], 5.0);
    }

    #[test]
    fn deep_zoom_precision() {
        let h = 2f64.powi(-31);
        let w = Window3::new([1e6 - h, -1.0, -1.0], [1e6 + h, 1.0, 1.0]);
        let r = Rig::new(w, Mode::D2);
        let p = [1e6 + 2f64.powi(-32), 0.0, 0.0];
        let n = r.project_ndc(p, 1.0);
        assert!((n[0] - 0.5).abs() < 1e-5, "{n:?}");
        let u = r.camera_uniform(1.0);
        let o = r.render_origin();
        let d = [(p[0] - o[0]) as f32, (p[1] - o[1]) as f32, (p[2] - o[2]) as f32, 1.0f32];
        let m = u.view_proj;
        let row = |i: usize| (0..4).map(|c| m[c][i] * d[c]).sum::<f32>();
        assert!((row(0) / row(3) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn degenerate_inputs_never_panic() {
        let wins = [
            Window3 { min: [0.0; 3], max: [0.0; 3] },
            Window3 { min: [-1e-300; 3], max: [1e-300; 3] },
            Window3 { min: [-1e300; 3], max: [1e300; 3] },
            Window3 { min: [f64::NAN; 3], max: [f64::INFINITY; 3] },
            Window3 { min: [5.0; 3], max: [-5.0; 3] },
        ];
        for w in wins {
            for mode in [Mode::D1, Mode::D2, Mode::D3] {
                let mut r = Rig::new(w, mode);
                for vp in [(0.0, 0.0), (-1.0, 5.0), (f64::NAN, 3.0), (800.0, 600.0)] {
                    r.pan_pixels(5.0, 5.0, vp);
                    r.zoom_at((1.0, 1.0), 2.0, vp);
                    let p = r.pixel_to_world((1.0, 1.0), vp);
                    assert!(p.iter().all(|v| v.is_finite()));
                    let _ = r.world_to_pixel(p, vp);
                }
                r.zoom_at((1.0, 1.0), f64::NAN, (10.0, 10.0));
                for a in [0.0, -1.0, f64::NAN, 1e-12, 1e12, 1.0] {
                    let u = r.camera_uniform(a);
                    assert!(u.view_proj.iter().flatten().all(|v| v.is_finite()));
                    assert!(u.scale.iter().chain(u.origin.iter()).all(|v| v.is_finite()));
                }
                r.set_mode(Mode::D3, 0.0);
                r.update(100.0);
                assert!(r.view_proj(1.0).iter().flatten().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn pan_3d_moves_centre_and_reset() {
        let mut r = Rig::new(win(), Mode::D3);
        r.pan_pixels(30.0, 10.0, VP);
        assert_ne!(r.window().centre(), [0.0; 3]);
        r.zoom_at((0.0, 0.0), 2.0, VP);
        r.reset();
        assert_eq!(r.window(), win());
    }
}
