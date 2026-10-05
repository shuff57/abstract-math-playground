// Shaders for the graphing scene. Vertex positions arrive ORIGIN-RELATIVE (the CPU subtracts the
// window centre in f64), and `cam.view_proj` is built for origin-relative coordinates, so only
// small deltas ever reach f32 even at deep zoom.

struct Camera {
    view_proj: mat4x4<f32>,
    origin: vec4<f32>,
    scale: vec4<f32>,
    mode: u32,
    ortho_t: f32,
    _pad: vec2<f32>,
};

struct Frame {
    viewport: vec2<f32>,
    fade: f32,
    // z scale about world z = 0 (offset.w): a 3D scene rises out of / settles into the plane
    // during a 2D <-> 3D switch. 1 otherwise.
    lift: f32,
    // xyz: geometry_origin - current_origin (computed in f64 on the CPU), so geometry built for
    // an older window keeps drawing correctly while a new one is rebuilt. w: world z = 0 relative
    // to the current origin.
    offset: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: Frame;

// Geometry position -> current-origin-relative position, with the switch lift applied.
fn placed(p: vec3<f32>) -> vec3<f32> {
    let q = p + frame.offset.xyz;
    return vec3<f32>(q.xy, frame.offset.w + (q.z - frame.offset.w) * frame.lift);
}

// ---------------------------------------------------------------- thick segments / dots

struct SegIn {
    @builtin(vertex_index) vi: u32,
    @location(0) p0: vec3<f32>,
    @location(1) width: f32,
    @location(2) p1: vec3<f32>,
    @location(3) color: vec4<f32>,
    // Depth bias (NDC z subtracted, so the segment wins against a surface it lies ON, e.g. a
    // slice curve on a mesh). 0 for ordinary segments.
    @location(4) bias: f32,
};

struct SegOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec2<f32>,
};

@vertex
fn vs_seg(in: SegIn) -> SegOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let q = corners[in.vi];
    let c0 = cam.view_proj * vec4<f32>(placed(in.p0), 1.0);
    let c1 = cam.view_proj * vec4<f32>(placed(in.p1), 1.0);

    var out: SegOut;
    out.color = in.color;
    // Behind the camera: push off-screen rather than draw garbage.
    if (c0.w <= 1e-5 || c1.w <= 1e-5) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local = vec2<f32>(0.0);
        out.params = vec2<f32>(0.0);
        return out;
    }

    let half_vp = 0.5 * frame.viewport;
    let s0 = c0.xy / c0.w * half_vp;
    let s1 = c1.xy / c1.w * half_vp;
    var d = s1 - s0;
    let len = length(d);
    if (len < 1e-4) {
        d = vec2<f32>(1.0, 0.0);
    } else {
        d = d / len;
    }
    let n = vec2<f32>(-d.y, d.x);
    let hw = max(in.width * 0.5, 0.5);
    let ext = hw + 1.0; // round cap plus one pixel of anti-aliasing margin
    let along = select(-ext, len + ext, q.x > 0.5);
    let across = q.y * ext;
    let p = s0 + d * along + n * across;
    let z = max(mix(c0.z / c0.w, c1.z / c1.w, q.x) - in.bias, 0.0);

    out.pos = vec4<f32>(p / half_vp, z, 1.0);
    out.local = vec2<f32>(along, across);
    out.params = vec2<f32>(len, hw);
    return out;
}

@fragment
fn fs_seg(in: SegOut) -> @location(0) vec4<f32> {
    let dx = in.local.x - clamp(in.local.x, 0.0, in.params.x);
    let dist = length(vec2<f32>(dx, in.local.y));
    let a = clamp(in.params.y + 0.5 - dist, 0.0, 1.0);
    let alpha = in.color.a * a * frame.fade;
    // Invisible fragments must not write depth, or they would hide what is behind them.
    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}

// ---------------------------------------------------------------- lit surfaces

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct MeshOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
    var out: MeshOut;
    out.pos = cam.view_proj * vec4<f32>(placed(in.pos), 1.0);
    // Normals of the z-scaled surface (inverse transpose of diag(1, 1, lift)); a zero normal
    // still marks unlit flat geometry, so the scale never reaches exactly 0.
    out.normal = vec3<f32>(in.normal.xy * max(frame.lift, 1.0e-3), in.normal.z);
    out.color = in.color;
    return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    // A zero normal marks flat, UNLIT geometry (2D bars): plain colour, alpha = colour alpha
    // times the crossfade.
    if (dot(in.normal, in.normal) < 1.0e-12) {
        let a = in.color.a * frame.fade;
        if (a < 0.01) {
            discard;
        }
        return vec4<f32>(in.color.rgb, a);
    }
    let light = normalize(vec3<f32>(0.35, -0.5, 0.8));
    let n = normalize(in.normal);
    // Two-sided: a surface has no inside, so shade by the absolute facing.
    let diffuse = abs(dot(n, light));
    let shade = 0.42 + 0.58 * diffuse;
    // Opaque on purpose: with depth writes, a translucent surface shows back faces through the
    // front ones depending on draw order. Only the mode-switch fade modulates alpha.
    if (frame.fade < 0.01) {
        discard;
    }
    return vec4<f32>(in.color.rgb * shade, frame.fade);
}
