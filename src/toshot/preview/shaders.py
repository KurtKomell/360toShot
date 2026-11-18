"""WGSL: dual fisheye → blended equirect look → rectilinear viewport (Vulkan/wgpu)."""

# Pre-stitched equirect (advanced / optical-flow path) → look viewport
EQUIRECT_LOOK_SHADER = """
struct Uniforms {
    yaw: f32,
    pitch: f32,
    roll: f32,
    fov_deg: f32,
    aspect: f32,
    eq_w: f32,
    eq_h: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var eq_tex: texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VSOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = p[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

fn deg2rad(d: f32) -> f32 { return d * 0.017453292519943295; }

fn rotate_ypr(v: vec3<f32>, yaw: f32, pitch: f32, roll: f32) -> vec3<f32> {
    let y = deg2rad(yaw);
    let p = deg2rad(pitch);
    let r = deg2rad(roll);
    let cy = cos(y); let sy = sin(y);
    let cp = cos(p); let sp = sin(p);
    let cr = cos(r); let sr = sin(r);
    var x = v.x; var yy = v.y; var z = v.z;
    let x1 =  cy * x + sy * z;
    let z1 = -sy * x + cy * z;
    x = x1; z = z1;
    let y2 =  cp * yy - sp * z;
    let z2 =  sp * yy + cp * z;
    yy = y2; z = z2;
    let x3 =  cr * x - sr * yy;
    let y3 =  sr * x + cr * yy;
    return vec3<f32>(x3, y3, z);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0);
    let fov = deg2rad(max(u.fov_deg, 1.0));
    let f = 1.0 / tan(fov * 0.5);
    var dir = normalize(vec3<f32>(ndc.x * u.aspect / f, ndc.y / f, 1.0));
    dir = rotate_ypr(dir, u.yaw, u.pitch, u.roll);
    let lon = atan2(dir.x, dir.z);
    let lat = asin(clamp(dir.y, -1.0, 1.0));
    let eq = clamp(vec2<f32>(
        lon / (2.0 * 3.141592653589793) + 0.5,
        0.5 - lat / 3.141592653589793
    ), vec2<f32>(0.0), vec2<f32>(1.0));
    let rgb = textureSampleLevel(eq_tex, samp, eq, 0.0).rgb;
    return vec4<f32>(rgb.bgr, 1.0);
}
"""

VIEWPORT_SHADER = """
struct Uniforms {
    yaw: f32,
    pitch: f32,
    roll: f32,
    fov_deg: f32,
    aspect: f32,
    eq_w: f32,
    eq_h: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var rear_tex: texture_2d<f32>;
@group(0) @binding(3) var front_tex: texture_2d<f32>;
@group(0) @binding(4) var rear_x: texture_2d<f32>;
@group(0) @binding(5) var rear_y: texture_2d<f32>;
@group(0) @binding(6) var front_x: texture_2d<f32>;
@group(0) @binding(7) var front_y: texture_2d<f32>;
@group(0) @binding(8) var weight_tex: texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VSOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = p[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    // Clip-space Y is up; keep UV as-is so screen-top → +lat (sky), not inverted.
    return out;
}

fn deg2rad(d: f32) -> f32 { return d * 0.017453292519943295; }

fn rotate_ypr(v: vec3<f32>, yaw: f32, pitch: f32, roll: f32) -> vec3<f32> {
    let y = deg2rad(yaw);
    let p = deg2rad(pitch);
    let r = deg2rad(roll);
    let cy = cos(y); let sy = sin(y);
    let cp = cos(p); let sp = sin(p);
    let cr = cos(r); let sr = sin(r);
    // R = Rz(roll) * Rx(pitch) * Ry(yaw) applied to camera forward
    var x = v.x; var yy = v.y; var z = v.z;
    // yaw about Y
    let x1 =  cy * x + sy * z;
    let z1 = -sy * x + cy * z;
    x = x1; z = z1;
    // pitch about X
    let y2 =  cp * yy - sp * z;
    let z2 =  sp * yy + cp * z;
    yy = y2; z = z2;
    // roll about Z
    let x3 =  cr * x - sr * yy;
    let y3 =  sr * x + cr * yy;
    return vec3<f32>(x3, y3, z);
}

fn eq_pixel(eq_uv: vec2<f32>, map_tex: texture_2d<f32>) -> vec2<i32> {
    // r32float maps are unfilterable — must use textureLoad, not textureSample
    let dims = textureDimensions(map_tex);
    let x = i32(clamp(eq_uv.x, 0.0, 0.99999) * f32(dims.x));
    let y = i32(clamp(eq_uv.y, 0.0, 0.99999) * f32(dims.y));
    return vec2<i32>(x, y);
}

fn sample_lens(tex: texture_2d<f32>, map_x: texture_2d<f32>, map_y: texture_2d<f32>, eq_uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(tex));
    let px = eq_pixel(eq_uv, map_x);
    let mx = textureLoad(map_x, px, 0).r;
    let my = textureLoad(map_y, px, 0).r;
    // maps store pixel coords; convert to normalized UV
    let uv = vec2<f32>((mx + 0.5) / dims.x, (my + 0.5) / dims.y);
    let inside = (uv.x >= 0.0) & (uv.x <= 1.0) & (uv.y >= 0.0) & (uv.y <= 1.0)
        & (mx >= 0.0) & (my >= 0.0);
    if (!inside) {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(tex, samp, uv, 0.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // NDC-like coords from uv
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0);
    let fov = deg2rad(max(u.fov_deg, 1.0));
    let f = 1.0 / tan(fov * 0.5);
    // Camera ray in view space (+Y up, +Z forward)
    var dir = normalize(vec3<f32>(ndc.x * u.aspect / f, ndc.y / f, 1.0));
    dir = rotate_ypr(dir, u.yaw, u.pitch, u.roll);

    // Direction → equirect UV
    let lon = atan2(dir.x, dir.z);           // -pi..pi
    let lat = asin(clamp(dir.y, -1.0, 1.0)); // -pi/2..pi/2
    var eq = vec2<f32>(
        lon / (2.0 * 3.141592653589793) + 0.5,
        0.5 - lat / 3.141592653589793
    );
    eq = clamp(eq, vec2<f32>(0.0), vec2<f32>(1.0));

    let c0 = sample_lens(rear_tex, rear_x, rear_y, eq);
    let c1 = sample_lens(front_tex, front_x, front_y, eq);
    let w = textureLoad(weight_tex, eq_pixel(eq, weight_tex), 0).r;
    let rgb = c0.rgb * w + c1.rgb * (1.0 - w);
    // BGR textures uploaded as RGBA with B,G,R → swap back to RGB for display
    return vec4<f32>(rgb.bgr, 1.0);
}
"""
