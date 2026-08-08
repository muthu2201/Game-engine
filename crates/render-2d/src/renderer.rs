//! The render pipeline and the offscreen target sprites are drawn into.

use crate::camera::Camera2D;
use crate::gpu::{GpuContext, GpuError};
use crate::sprite::{Color, SpriteInstance};
use crate::texture::TextureArray;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// The per-frame uniform block, matching `Globals` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Globals {
    view_projection: [[f32; 4]; 4],
    global_tint: [f32; 4],
}

/// The unit quad every sprite is an instance of.
///
/// Two triangles from four vertices via an index buffer, rather than six
/// vertices: fewer bytes and the vertex cache does the rest.
const QUAD_VERTICES: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// Indices for the quad's two triangles.
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

/// Number of instances the buffer starts out able to hold.
///
/// Grows geometrically when exceeded; starting non-trivially avoids a burst of
/// reallocations over the first few frames.
const INITIAL_INSTANCE_CAPACITY: usize = 1024;

/// An offscreen colour target that can be read back.
///
/// The game renders at a low internal resolution into one of these, and the
/// result is scaled up by a whole number to fill the window. Rendering directly
/// at window resolution and scaling the *sprites* instead would put pixel-art
/// texels on fractional screen positions, which is exactly the shimmering the
/// engine's pixel-snapping exists to avoid.
pub struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RenderTarget {
    /// The format every target and pipeline uses.
    ///
    /// Non-sRGB, because colours are converted to linear on the CPU once and
    /// the GPU must not apply the transfer function a second time.
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// Creates a target of the given size.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> RenderTarget {
        assert!(
            width > 0 && height > 0,
            "RenderTarget: dimensions must be non-zero"
        );
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("verdant render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RenderTarget::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        RenderTarget {
            texture,
            view,
            width,
            height,
        }
    }

    /// The target's view, for use as a render attachment.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Target width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Target height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Copies the target back to the CPU as RGBA8 rows.
    ///
    /// This is what makes the renderer testable: a golden-image test renders a
    /// scene, reads it back, and compares against a reference. It is a slow,
    /// synchronising operation and has no place in a shipped frame.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Operation`] if the readback cannot be mapped.
    pub fn read_pixels(&self, gpu: &GpuContext) -> Result<Vec<u8>, GpuError> {
        // Buffer rows must be aligned to 256 bytes, so the staging buffer is
        // usually wider than the image and is trimmed after mapping.
        let unpadded_row = self.width * 4;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(alignment) * alignment;

        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant readback"),
            size: u64::from(padded_row) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // A send failure only means the receiver went away, which cannot
            // happen while this function is still on the stack.
            let _ = sender.send(result);
        });
        gpu.wait_for_idle()?;
        receiver
            .recv()
            .map_err(|error| GpuError::Operation(error.to_string()))?
            .map_err(|error| GpuError::Operation(error.to_string()))?;

        let padded = slice
            .get_mapped_range()
            .map_err(|error| GpuError::Operation(error.to_string()))?;
        let mut pixels = Vec::with_capacity((unpadded_row * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded_row) as usize;
            pixels.extend_from_slice(&padded[start..start + unpadded_row as usize]);
        }
        drop(padded);
        buffer.unmap();
        Ok(pixels)
    }
}

impl std::fmt::Debug for RenderTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderTarget")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

/// The frame-wide settings a render pass applies.
///
/// Grouped rather than passed positionally because "which of these two colours
/// was the clear and which was the tint" is exactly the kind of thing a long
/// argument list gets wrong silently.
#[derive(Clone, Copy, Debug)]
pub struct FrameSettings {
    /// Background colour the target is cleared to.
    pub clear: Color,
    /// Multiplies every sprite's tint. The day/night cycle and weather are
    /// applied here, in one place, rather than per sprite.
    pub global_tint: Color,
}

impl Default for FrameSettings {
    fn default() -> FrameSettings {
        FrameSettings {
            clear: Color::BLACK,
            global_tint: Color::WHITE,
        }
    }
}

/// Draws batched sprites into a [`RenderTarget`].
pub struct SpriteRenderer {
    pipeline: wgpu::RenderPipeline,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    atlas_layout: wgpu::BindGroupLayout,
    quad_vertices: wgpu::Buffer,
    quad_indices: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
}

impl SpriteRenderer {
    /// Builds the pipeline.
    #[must_use]
    pub fn new(gpu: &GpuContext) -> SpriteRenderer {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("verdant sprite shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        let globals_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("verdant globals layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let atlas_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("verdant atlas layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let globals_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verdant globals bind group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("verdant sprite pipeline layout"),
                bind_group_layouts: &[Some(&globals_layout), Some(&atlas_layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("verdant sprite pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[
                        // The unit quad.
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                        }),
                        // Per-sprite instance data.
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<SpriteInstance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![
                                1 => Float32x2,
                                2 => Float32x2,
                                3 => Float32x2,
                                4 => Float32x4,
                                5 => Float32x4,
                                6 => Uint32,
                            ],
                        }),
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RenderTarget::FORMAT,
                        // Straight (non-premultiplied) alpha, which is what the
                        // procedural art generator produces.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // No back-face culling: a horizontally flipped sprite has a
                    // negative x axis and therefore reversed winding, and culling
                    // would make every mirrored sprite vanish.
                    cull_mode: None,
                    ..Default::default()
                },
                // No depth buffer: order comes from the batcher's sort, which is
                // what a 2D painter's-algorithm renderer wants.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let quad_vertices = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verdant quad vertices"),
                contents: bytemuck::cast_slice(&QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let quad_indices = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verdant quad indices"),
                contents: bytemuck::cast_slice(&QUAD_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        let instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant sprite instances"),
            size: (INITIAL_INSTANCE_CAPACITY * std::mem::size_of::<SpriteInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        SpriteRenderer {
            pipeline,
            globals_buffer,
            globals_bind_group,
            atlas_layout,
            quad_vertices,
            quad_indices,
            instances,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
        }
    }

    /// Creates the bind group for an atlas.
    ///
    /// Held by the caller so an atlas that never changes is bound without
    /// rebuilding the group each frame.
    #[must_use]
    pub fn bind_atlas(&self, gpu: &GpuContext, atlas: &TextureArray) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verdant atlas bind group"),
            layout: &self.atlas_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(atlas.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(atlas.sampler()),
                },
            ],
        })
    }

    /// Grows the instance buffer if `count` will not fit.
    ///
    /// Doubling rather than fitting exactly means a scene whose sprite count
    /// creeps upward does not reallocate every frame.
    fn ensure_capacity(&mut self, gpu: &GpuContext, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        let capacity = count.next_power_of_two();
        self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verdant sprite instances"),
            size: (capacity * std::mem::size_of::<SpriteInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = capacity;
    }

    /// Renders `instances` into `target`.
    ///
    /// `clear` is the background colour; `global_tint` multiplies every sprite,
    /// which is how the day/night cycle and weather are applied in one place.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        target: &RenderTarget,
        atlas_bind_group: &wgpu::BindGroup,
        camera: &Camera2D,
        instances: &[SpriteInstance],
        settings: FrameSettings,
    ) {
        self.ensure_capacity(gpu, instances.len());

        let globals = Globals {
            view_projection: camera.view_projection(),
            global_tint: settings.global_tint.to_array(),
        };
        gpu.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
        if !instances.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(instances));
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sprite pass"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("verdant sprite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(settings.clear.r),
                            g: f64::from(settings.clear.g),
                            b: f64::from(settings.clear.b),
                            a: f64::from(settings.clear.a),
                        }),
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
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                pass.set_bind_group(1, atlas_bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vertices.slice(..));
                pass.set_vertex_buffer(1, self.instances.slice(..));
                pass.set_index_buffer(self.quad_indices.slice(..), wgpu::IndexFormat::Uint16);
                // The whole batch in one call: this is the point of the design.
                let count = u32::try_from(instances.len()).unwrap_or(u32::MAX);
                let index_count = u32::try_from(QUAD_INDICES.len()).unwrap_or(6);
                pass.draw_indexed(0..index_count, 0, 0..count);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
    }

    /// How many instances the buffer can currently hold.
    #[must_use]
    pub const fn instance_capacity(&self) -> usize {
        self.instance_capacity
    }
}

impl std::fmt::Debug for SpriteRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpriteRenderer")
            .field("instance_capacity", &self.instance_capacity)
            .finish()
    }
}

/// The largest whole-number scale at which `internal` fits inside `window`.
///
/// Integer scaling is what keeps pixel art crisp: at 2.5x, some source pixels
/// would cover three screen pixels and others two, which reads as uneven
/// letter strokes and wobbling outlines. The remainder is letterboxed.
#[must_use]
pub fn integer_scale(internal: (u32, u32), window: (u32, u32)) -> u32 {
    if internal.0 == 0 || internal.1 == 0 {
        return 1;
    }
    (window.0 / internal.0).min(window.1 / internal.1).max(1)
}

/// Where a scaled internal resolution sits inside a window.
///
/// Returns the top-left corner and the scaled size, centring the image so the
/// letterbox is even on both sides.
#[must_use]
pub fn letterbox(internal: (u32, u32), window: (u32, u32)) -> ((u32, u32), (u32, u32)) {
    let scale = integer_scale(internal, window);
    let size = (internal.0 * scale, internal.1 * scale);
    let origin = (
        window.0.saturating_sub(size.0) / 2,
        window.1.saturating_sub(size.1) / 2,
    );
    (origin, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_scaling_never_goes_fractional() {
        // A 320x180 game in a 1280x720 window scales exactly 4x.
        assert_eq!(integer_scale((320, 180), (1280, 720)), 4);
        // In a 1600x900 window it is 5x.
        assert_eq!(integer_scale((320, 180), (1600, 900)), 5);
        // A window that is not a whole multiple takes the largest that fits.
        assert_eq!(integer_scale((320, 180), (1000, 700)), 3);
    }

    #[test]
    fn scaling_is_limited_by_the_tighter_axis() {
        // Wide but short: the height limits the scale.
        assert_eq!(integer_scale((320, 180), (4000, 400)), 2);
    }

    #[test]
    fn a_window_smaller_than_the_internal_size_still_scales_by_one() {
        assert_eq!(integer_scale((320, 180), (100, 100)), 1);
        assert_eq!(integer_scale((0, 0), (100, 100)), 1);
    }

    #[test]
    fn letterboxing_centres_the_image() {
        // The Steam Deck's 1280x800 is 16:10, so a 16:9 game letterboxes.
        let (origin, size) = letterbox((320, 180), (1280, 800));
        assert_eq!(size, (1280, 720), "scaled 4x to fill the width");
        assert_eq!(origin, (0, 40), "with 40 pixels of border above and below");
    }

    #[test]
    fn an_exact_fit_has_no_border() {
        let (origin, size) = letterbox((320, 180), (1280, 720));
        assert_eq!(origin, (0, 0));
        assert_eq!(size, (1280, 720));
    }

    #[test]
    fn the_globals_block_matches_the_shader_layout() {
        // mat4x4 plus vec4, both 16-byte aligned.
        assert_eq!(std::mem::size_of::<Globals>(), 64 + 16);
    }
}
