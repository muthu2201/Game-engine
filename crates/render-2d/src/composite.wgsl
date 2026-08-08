// Combines the lit scene with the light map.
//
// The scene pass draws sprites at full brightness and the light pass
// accumulates every light into a separate buffer. This multiplies one by the
// other, which is what makes an unlit corner dark and a lantern's pool of
// light warm — and it does so in one full-screen pass regardless of how many
// lights or sprites there were.

struct Settings {
    // Light present everywhere. The day/night cycle drives this, so noon needs
    // no lights at all and midnight leans entirely on them.
    ambient: vec4<f32>,
    // Multiplies the accumulated lights, for turning the whole system down
    // without editing every light.
    light_scale: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let uv = vec2<f32>(
        f32((index << 1u) & 2u),
        f32(index & 2u),
    );
    var out: VertexOutput;
    out.uv = uv;
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var light_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> settings: Settings;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv);
    let lights = textureSample(light_texture, scene_sampler, in.uv);

    // Ambient plus lights, then multiply. Clamping the total keeps a stack of
    // overlapping lanterns from blowing the scene out to white.
    let illumination = min(
        settings.ambient.rgb + lights.rgb * settings.light_scale.rgb,
        vec3<f32>(settings.light_scale.a)
    );

    return vec4<f32>(scene.rgb * illumination, scene.a);
}
