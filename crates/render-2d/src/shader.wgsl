// Instanced sprite shader.
//
// One draw call renders every sprite in a batch: the quad's four corners come
// from the vertex buffer, and everything that varies per sprite — transform,
// UV window, tint, atlas layer — arrives as instance attributes. The atlas is a
// texture_2d_array, so a sprite selects its artwork with an integer index
// rather than a bind-group switch, which is what keeps the batch unbroken.

struct Globals {
    // World-to-clip transform, column-major.
    view_projection: mat4x4<f32>,
    // Multiplies every sprite's tint; drives day/night and damage flashes
    // without touching per-sprite data.
    global_tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var atlas: texture_2d_array<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VertexInput {
    // Unit quad corner, (0,0) to (1,1).
    @location(0) corner: vec2<f32>,
};

struct InstanceInput {
    // Packed 2x3 affine: x axis, y axis, translation.
    @location(1) model_x: vec2<f32>,
    @location(2) model_y: vec2<f32>,
    @location(3) model_translation: vec2<f32>,
    // Sub-rectangle of the atlas layer: origin then size.
    @location(4) uv_rect: vec4<f32>,
    @location(5) color: vec4<f32>,
    @location(6) layer: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) layer: u32,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Apply the affine transform to the unit corner. Two multiply-adds, versus
    // reconstructing a rotation from an angle for every vertex.
    let world = instance.model_x * vertex.corner.x
              + instance.model_y * vertex.corner.y
              + instance.model_translation;

    var out: VertexOutput;
    out.clip_position = globals.view_projection * vec4<f32>(world, 0.0, 1.0);
    out.uv = instance.uv_rect.xy + vertex.corner * instance.uv_rect.zw;
    out.color = instance.color;
    out.layer = instance.layer;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas, atlas_sampler, in.uv, in.layer);
    let result = texel * in.color * globals.global_tint;

    // Discard fully transparent fragments rather than blending them. For pixel
    // art with hard alpha edges this costs nothing and keeps the depth-free
    // painter's-algorithm ordering from accumulating invisible blend work.
    if (result.a < 0.004) {
        discard;
    }
    return result;
}
