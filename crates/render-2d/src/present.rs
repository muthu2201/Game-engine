//! Scaling the low-resolution frame up onto the window.
//!
//! The sprite pass always draws at the game's internal resolution. This is the
//! second pass that puts that image on screen: a full-screen triangle sampled
//! with nearest-neighbour filtering, drawn into an integer-scaled, centred
//! viewport, with the surrounding letterbox left at the clear colour.
//!
//! Doing the upscale as its own pass rather than by rendering the world at
//! window size is what makes the pixel grid exact. Every source texel covers
//! precisely the same whole number of screen pixels, so outlines stay one pixel
//! thick everywhere on screen.

use crate::gpu::GpuContext;
use crate::renderer::{letterbox, RenderTarget};
use crate::sprite::Color;

/// Upscales a [`RenderTarget`] onto another texture or a swapchain frame.
pub struct Presenter {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl Presenter {
    /// Builds the blit pipeline for a destination of `format`.
    ///
    /// The format must match the texture this presenter will draw into —
    /// usually [`crate::SurfaceContext::format`].
    #[must_use]
    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat) -> Presenter {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("verdant present shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("present.wgsl").into()),
            });

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("verdant present layout"),
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
                ],
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("verdant present pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("verdant present pipeline"),
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

        // Nearest in both directions: any filtering here would undo the whole
        // point of rendering at a low resolution.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("verdant present sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Presenter {
            pipeline,
            layout,
            sampler,
        }
    }

    /// Binds a source frame for presentation.
    ///
    /// The bind group depends only on the target's view, so it can be built
    /// once and reused for every frame drawn into the same target.
    #[must_use]
    pub fn bind_source(&self, gpu: &GpuContext, source: &RenderTarget) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verdant present source"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Draws the bound source into `destination`, scaled and centred.
    ///
    /// `internal` is the source frame's size and `destination_size` the
    /// destination's, both in pixels. `border` fills the letterbox.
    pub fn present(
        &self,
        gpu: &GpuContext,
        destination: &wgpu::TextureView,
        source: &wgpu::BindGroup,
        internal: (u32, u32),
        destination_size: (u32, u32),
        border: Color,
    ) {
        let (origin, size) = letterbox(internal, destination_size);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present pass"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("verdant present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(border.r),
                            g: f64::from(border.g),
                            b: f64::from(border.b),
                            a: f64::from(border.a),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // A viewport rather than scaled vertices: the hardware then does
            // the centring, and the shader stays a plain full-screen blit.
            if size.0 > 0 && size.1 > 0 {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "viewport coordinates are pixel counts far below f32's exact range"
                )]
                pass.set_viewport(
                    origin.0 as f32,
                    origin.1 as f32,
                    size.0 as f32,
                    size.1 as f32,
                    0.0,
                    1.0,
                );
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, source, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}

impl std::fmt::Debug for Presenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Presenter").finish_non_exhaustive()
    }
}
