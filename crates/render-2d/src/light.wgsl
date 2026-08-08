// Light accumulation.
//
// One instanced quad per light, sized to the light's reach, blended additively
// into a light map. The map is then multiplied over the scene by composite.wgsl.
//
// Lights are accumulated in their own pass rather than evaluated per sprite
// because the cost then scales with screen area covered by lights rather than
// with sprites times lights — a night scene with two hundred sprites under
// three lanterns pays for three quads, not six hundred evaluations.

struct Globals {
    view_projection: mat4x4<f32>,
};

struct LightInstance {
    // World-space centre.
    @location(0) center: vec2<f32>,
    // Colour, already multiplied by intensity.
    @location(1) color: vec4<f32>,
    // x: reach in world units
    // y: falloff exponent — 1 is linear, 2 is close to physical
    // z: cosine of the cone's half-angle; -1 means a full circle
    // w: the cone's direction in radians
    @location(2) params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: LightInstance,
) -> VertexOutput {
    // A quad from the vertex index: two triangles, no vertex buffer to bind.
    // Corners run (-1,-1), (1,-1), (-1,1), (-1,1), (1,-1), (1,1).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let radius = instance.params.x;
    let world = instance.center + corners[vertex_index] * radius;

    var out: VertexOutput;
    out.clip_position = globals.view_projection * vec4<f32>(world, 0.0, 1.0);
    out.world = world;
    out.center = instance.center;
    out.color = instance.color;
    out.params = instance.params;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius = in.params.x;
    let falloff = in.params.y;
    let cone_cos = in.params.z;
    let cone_direction = in.params.w;

    let offset = in.world - in.center;
    let distance = length(offset);
    if (distance >= radius) {
        discard;
    }

    // Smooth to zero at the edge. A linear ramp leaves a visible circle where
    // the light stops; raising it to a power softens that edge without
    // shrinking the lit area.
    let normalised = distance / max(radius, 0.0001);
    var attenuation = pow(1.0 - normalised, max(falloff, 0.0001));

    // Cone test for spot lights. cone_cos of -1 admits every direction, which
    // is how a point light is expressed without a second pipeline.
    if (cone_cos > -1.0 && distance > 0.0) {
        let direction = vec2<f32>(cos(cone_direction), sin(cone_direction));
        let alignment = dot(offset / distance, direction);
        if (alignment < cone_cos) {
            discard;
        }
        // Fade across the last of the cone so its edge is not a hard line.
        let edge = (alignment - cone_cos) / max(1.0 - cone_cos, 0.0001);
        attenuation = attenuation * clamp(edge * 3.0, 0.0, 1.0);
    }

    // Alpha stays at the accumulated weight so a caller can tell lit from
    // unlit; the composite pass reads only rgb.
    return vec4<f32>(in.color.rgb * attenuation * in.color.a, attenuation);
}
