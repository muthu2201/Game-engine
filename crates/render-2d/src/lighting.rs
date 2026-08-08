//! 2D lighting.
//!
//! Lights are accumulated into their own buffer and multiplied over the scene,
//! rather than evaluated while drawing sprites. That choice is what makes the
//! system affordable: cost scales with the screen area lights cover, not with
//! sprites multiplied by lights. A night scene with two hundred sprites under
//! three lanterns pays for three quads.
//!
//! ## The three passes
//!
//! ```text
//! sprites ──> scene target ─┐
//!                           ├─> composite ──> final
//! lights  ──> light map  ───┘
//! ```
//!
//! [`LightRenderer`] draws the middle row; [`Compositor`] joins them. Ambient
//! light lives in the composite rather than as a giant light, so a fully lit
//! midday scene costs one full-screen pass and no lights at all.
//!
//! ## Why lights are not sprites
//!
//! A glowing sprite drawn additively looks like a light until something moves
//! behind it. Real falloff has to be evaluated per fragment against the
//! light's world position, which is what [`Light`] carries and what the shader
//! does — and it is why a light's reach is a world-space radius rather than a
//! texture size.

use crate::gpu::GpuContext;
use crate::renderer::RenderTarget;
use crate::sprite::Color;
use bytemuck::{Pod, Zeroable};
use verdant_core_math::{Fx, Rect, Vec2};
use wgpu::util::DeviceExt;

/// How many lights a new renderer can hold before it grows its buffer.
pub const DEFAULT_LIGHT_CAPACITY: usize = 64;

/// A light in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Where it sits, in world units.
    pub position: Vec2,
    /// How far its light reaches, in world units.
    pub radius: Fx,
    /// Its colour.
    pub color: Color,
    /// How bright it is. One is full strength; above one over-drives and is
    /// clamped by the composite.
    pub intensity: f32,
    /// How sharply it falls off. One is linear; two is close to physical.
    pub falloff: f32,
    /// The cone, for a spot light.
    pub cone: Option<Cone>,
}

/// A spot light's cone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    /// Which way it points, in radians.
    pub direction: Fx,
    /// Half the cone's opening angle, in radians.
    pub half_angle: Fx,
}

impl Light {
    /// A point light.
    #[must_use]
    pub fn point(position: Vec2, radius: Fx, color: Color) -> Light {
        Light {
            position,
            radius,
            color,
            intensity: 1.0,
            // Quadratic by default: a linear falloff reads as a flat disc
            // rather than as something glowing.
            falloff: 2.0,
            cone: None,
        }
    }

    /// A spot light pointing along `direction`.
    #[must_use]
    pub fn spot(position: Vec2, radius: Fx, color: Color, direction: Fx, half_angle: Fx) -> Light {
        Light {
            cone: Some(Cone {
                direction,
                half_angle,
            }),
            ..Light::point(position, radius, color)
        }
    }

    /// Returns this light at a different brightness.
    #[must_use]
    pub const fn with_intensity(mut self, intensity: f32) -> Light {
        self.intensity = intensity;
        self
    }

    /// Returns this light with a different falloff exponent.
    #[must_use]
    pub const fn with_falloff(mut self, falloff: f32) -> Light {
        self.falloff = falloff;
        self
    }

    /// The world-space area this light can affect.
    ///
    /// Used for culling: a light whose bounds miss the view contributes
    /// nothing and never reaches the GPU.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        Rect::from_centre(self.position, Vec2::splat(self.radius * Fx::from_num(2)))
    }

    /// True when this light could affect anything inside `view`.
    #[must_use]
    pub fn is_visible(&self, view: Rect) -> bool {
        // A light with no reach or no brightness is invisible wherever it is,
        // and culling it here keeps degenerate lights out of the buffer.
        if self.radius <= Fx::ZERO || self.intensity <= 0.0 {
            return false;
        }
        self.bounds().intersects(view)
    }

    /// Packs this light for the GPU.
    #[must_use]
    fn to_instance(self) -> LightInstance {
        // -1 admits every direction, which expresses a point light without a
        // second pipeline.
        let (cone_cos, cone_direction) = match self.cone {
            Some(cone) => (cone.half_angle.cos().to_f32(), cone.direction.to_f32()),
            None => (-1.0, 0.0),
        };
        LightInstance {
            center: [self.position.x.to_f32(), self.position.y.to_f32()],
            color: [
                self.color.r,
                self.color.g,
                self.color.b,
                self.intensity.max(0.0),
            ],
            params: [
                self.radius.to_f32(),
                self.falloff.max(0.0001),
                cone_cos,
                cone_direction,
            ],
        }
    }
}

/// A light packed for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightInstance {
    center: [f32; 2],
    color: [f32; 4],
    params: [f32; 4],
}

/// The uniform the light pass reads.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightGlobals {
    view_projection: [[f32; 4]; 4],
}

/// The uniform the composite reads.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeSettings {
    ambient: [f32; 4],
    light_scale: [f32; 4],
}

/// How the composite combines the scene with its lights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSettings {
    /// Light present everywhere, before any light is added.
    ///
    /// The day/night cycle drives this, which is why a midday scene needs no
    /// lights at all and midnight leans entirely on them.
    pub ambient: Color,
    /// Multiplies every accumulated light, for turning the system down without
    /// editing each one.
    pub light_scale: f32,
    /// The brightest the combined illumination may reach.
    ///
    /// Above one, overlapping lanterns can deliberately over-brighten; at one
    /// they saturate instead of blowing out to white.
    pub maximum: f32,
}

impl Default for LightSettings {
    fn default() -> LightSettings {
        LightSettings {
            ambient: Color::WHITE,
            light_scale: 1.0,
            maximum: 1.0,
        }
    }
}

impl LightSettings {
    /// Settings for a scene lit only by its ambient light.
    #[must_use]
    pub const fn ambient_only(ambient: Color) -> LightSettings {
        LightSettings {
            ambient,
            light_scale: 1.0,
            maximum: 1.0,
        }
    }

    /// True when lights would make no visible difference.
    ///
    /// A fully lit scene can skip both the light pass and the composite
    /// entirely, which is the common case in daylight.
    #[must_use]
    pub fn is_fully_lit(&self) -> bool {
        self.ambient.r >= self.maximum
            && self.ambient.g >= self.maximum
            && self.ambient.b >= self.maximum
    }
}

/// Accumulates lights into a light map.
pub struct LightRenderer {
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    capacity: usize,
}

impl LightRenderer {
    /// Builds the light pipeline for a target of `format`.
    #[must_use]
    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat) -> LightRenderer {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("verdant light shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            });

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("verdant light globals layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verdant light globals"),
                contents: bytemuck::bytes_of(&LightGlobals {
                    view_projection: [[0.0; 4]; 4],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verdant light globals"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("verdant light pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("verdant light pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<LightInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x2,
                            1 => Float32x4,
                            2 => Float32x4,
                        ],
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // Additive: two lanterns overlapping are brighter than
                        // either alone, which is how light actually behaves.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant light instances"),
            size: (std::mem::size_of::<LightInstance>() * DEFAULT_LIGHT_CAPACITY)
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        LightRenderer {
            pipeline,
            globals,
            bind_group,
            instances,
            capacity: DEFAULT_LIGHT_CAPACITY,
        }
    }

    /// How many lights fit without the buffer growing.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Grows the instance buffer if `needed` will not fit.
    fn ensure_capacity(&mut self, gpu: &GpuContext, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        // Doubling rather than fitting exactly, so a scene that gains lights
        // one at a time does not reallocate on every frame.
        let capacity = needed.next_power_of_two();
        self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant light instances"),
            size: (std::mem::size_of::<LightInstance>() * capacity) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity = capacity;
    }

    /// Draws `lights` into `target`, clearing it first.
    ///
    /// The target is cleared to black: ambient light is applied by the
    /// composite, not accumulated here, so that a scene with no lights costs
    /// nothing beyond the clear.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        target: &RenderTarget,
        camera: &crate::Camera2D,
        lights: &[Light],
    ) {
        let view = camera.visible_bounds();
        let instances: Vec<LightInstance> = lights
            .iter()
            .filter(|light| light.is_visible(view))
            .map(|light| light.to_instance())
            .collect();

        self.ensure_capacity(gpu, instances.len());
        gpu.queue.write_buffer(
            &self.globals,
            0,
            bytemuck::bytes_of(&LightGlobals {
                view_projection: camera.view_projection(),
            }),
        );
        if !instances.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("light pass"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("verdant light pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if !instances.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.instances.slice(..));
                let count = u32::try_from(instances.len()).unwrap_or(u32::MAX);
                pass.draw(0..6, 0..count);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}

impl std::fmt::Debug for LightRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightRenderer")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

/// Multiplies a scene by its light map.
pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    settings: wgpu::Buffer,
}

impl Compositor {
    /// Builds the composite pipeline for a destination of `format`.
    #[must_use]
    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat) -> Compositor {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("verdant composite shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
            });

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("verdant composite layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("verdant composite pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("verdant composite pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("verdant composite sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let settings = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verdant composite settings"),
                contents: bytemuck::bytes_of(&CompositeSettings {
                    ambient: [1.0; 4],
                    light_scale: [1.0; 4],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Compositor {
            pipeline,
            layout,
            sampler,
            settings,
        }
    }

    /// Binds a scene and a light map for compositing.
    ///
    /// Depends only on the two views, so it is built once per pair of targets
    /// rather than once per frame.
    #[must_use]
    pub fn bind(
        &self,
        gpu: &GpuContext,
        scene: &RenderTarget,
        lights: &RenderTarget,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verdant composite inputs"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(lights.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.settings.as_entire_binding(),
                },
            ],
        })
    }

    /// Writes the lit scene into `destination`.
    pub fn composite(
        &self,
        gpu: &GpuContext,
        destination: &wgpu::TextureView,
        inputs: &wgpu::BindGroup,
        settings: LightSettings,
    ) {
        gpu.queue.write_buffer(
            &self.settings,
            0,
            bytemuck::bytes_of(&CompositeSettings {
                ambient: settings.ambient.to_array(),
                light_scale: [
                    settings.light_scale,
                    settings.light_scale,
                    settings.light_scale,
                    settings.maximum,
                ],
            }),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite pass"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("verdant composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, inputs, &[]);
            pass.draw(0..3, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}

impl std::fmt::Debug for Compositor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compositor").finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert exact values the packing code writes as literals \
              — a sentinel cone cosine, a clamped intensity — and a tolerance \
              would stop the test checking what it claims to."
)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    #[test]
    fn a_point_light_admits_every_direction() {
        // -1 as the cone cosine is how a point light avoids needing its own
        // pipeline.
        let light = Light::point(Vec2::ZERO, fx(10), Color::WHITE);
        assert_eq!(light.to_instance().params[2], -1.0);
    }

    #[test]
    fn a_spot_light_carries_its_cone() {
        let light = Light::spot(Vec2::ZERO, fx(10), Color::WHITE, Fx::ZERO, Fx::PI / fx(4));
        let params = light.to_instance().params;
        // cos(45 degrees) is about 0.707.
        assert!(
            (params[2] - 0.707).abs() < 0.01,
            "cone cosine {}",
            params[2]
        );
    }

    #[test]
    fn a_lights_bounds_cover_its_reach() {
        let light = Light::point(Vec2::from_ints(100, 50), fx(20), Color::WHITE);
        let bounds = light.bounds();
        assert_eq!(bounds.min, Vec2::from_ints(80, 30));
        assert_eq!(bounds.max(), Vec2::from_ints(120, 70));
    }

    #[test]
    fn a_light_outside_the_view_is_culled() {
        let view = Rect::from_ints(0, 0, 100, 100);
        let near = Light::point(Vec2::from_ints(50, 50), fx(10), Color::WHITE);
        let far = Light::point(Vec2::from_ints(500, 500), fx(10), Color::WHITE);
        assert!(near.is_visible(view));
        assert!(!far.is_visible(view));
    }

    #[test]
    fn a_light_just_off_screen_still_counts() {
        // Its reach crosses the edge, so culling it would make light pop in
        // as the camera moves.
        let view = Rect::from_ints(0, 0, 100, 100);
        let light = Light::point(Vec2::from_ints(-15, 50), fx(20), Color::WHITE);
        assert!(light.is_visible(view));
    }

    #[test]
    fn a_light_with_no_reach_is_culled() {
        let view = Rect::from_ints(0, 0, 100, 100);
        let dark = Light::point(Vec2::from_ints(50, 50), Fx::ZERO, Color::WHITE);
        assert!(!dark.is_visible(view));
    }

    #[test]
    fn a_light_with_no_intensity_is_culled() {
        let view = Rect::from_ints(0, 0, 100, 100);
        let dark = Light::point(Vec2::from_ints(50, 50), fx(10), Color::WHITE).with_intensity(0.0);
        assert!(!dark.is_visible(view));
    }

    #[test]
    fn intensity_reaches_the_gpu() {
        let light = Light::point(Vec2::ZERO, fx(10), Color::WHITE).with_intensity(0.25);
        assert!((light.to_instance().color[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_negative_intensity_cannot_darken_the_scene() {
        // Subtractive light is not a thing, and letting it through would make
        // an unlit area darker than black.
        let light = Light::point(Vec2::ZERO, fx(10), Color::WHITE).with_intensity(-5.0);
        assert_eq!(light.to_instance().color[3], 0.0);
    }

    #[test]
    fn a_zero_falloff_cannot_divide_by_zero_in_the_shader() {
        let light = Light::point(Vec2::ZERO, fx(10), Color::WHITE).with_falloff(0.0);
        assert!(light.to_instance().params[1] > 0.0);
    }

    #[test]
    fn the_default_falloff_is_not_linear() {
        // A linear falloff reads as a flat disc rather than as a glow.
        assert!(Light::point(Vec2::ZERO, fx(10), Color::WHITE).falloff > 1.0);
    }

    #[test]
    fn a_fully_lit_scene_can_skip_the_whole_system() {
        // The common daylight case: no light pass, no composite.
        assert!(LightSettings::default().is_fully_lit());
        assert!(LightSettings::ambient_only(Color::WHITE).is_fully_lit());
        assert!(!LightSettings::ambient_only(Color::rgb(0.3, 0.3, 0.4)).is_fully_lit());
    }

    #[test]
    fn a_partially_lit_channel_still_needs_lights() {
        // Dusk tints one channel down; treating that as fully lit would drop
        // every lantern in the scene.
        let dusk = LightSettings::ambient_only(Color::rgb(1.0, 0.8, 0.6));
        assert!(!dusk.is_fully_lit());
    }

    #[test]
    fn light_instances_are_packed_tightly() {
        // The vertex layout hard-codes these offsets, so a change in the
        // struct that is not mirrored there would read garbage.
        assert_eq!(std::mem::size_of::<LightInstance>(), 40);
    }

    #[test]
    fn a_lights_position_reaches_the_gpu_unchanged() {
        let light = Light::point(Vec2::from_ints(-12, 34), fx(5), Color::WHITE);
        let instance = light.to_instance();
        assert!((instance.center[0] + 12.0).abs() < 1e-6);
        assert!((instance.center[1] - 34.0).abs() < 1e-6);
        assert!((instance.params[0] - 5.0).abs() < 1e-6);
    }
}
