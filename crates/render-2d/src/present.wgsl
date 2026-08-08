// Full-screen blit used to scale the low-resolution frame onto the window.
//
// The triangle is generated from the vertex index rather than a vertex buffer:
// one oversized triangle covers the viewport with no buffer binding and no
// diagonal seam down the middle of the screen.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // (0,0), (2,0), (0,2) in UV space; the parts beyond 1 are clipped away.
    let uv = vec2<f32>(
        f32((index << 1u) & 2u),
        f32(index & 2u),
    );

    var out: VertexOutput;
    out.uv = uv;
    // UV origin is top-left, clip-space origin is centre with +Y up.
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Nearest sampling, and the destination is a whole multiple of the source,
    // so every texel lands on an exact block of screen pixels.
    return textureSample(source_texture, source_sampler, in.uv);
}
