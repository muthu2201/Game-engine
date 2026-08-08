//! The windowed path: a swapchain surface and the device that drives it.
//!
//! [`GpuContext`] deliberately knows nothing about windows, because everything
//! the renderer does — batching, atlases, the sprite pass — happens offscreen
//! and is tested that way. This module is the thin layer that puts the result
//! on a screen.
//!
//! ## Why a non-sRGB surface format
//!
//! The sprite pass draws into an `Rgba8Unorm` target and its bytes are what
//! the golden images in CI capture. Presenting through an sRGB swapchain would
//! ask the hardware to encode those bytes a second time, so what a player saw
//! would not be what CI verified. [`SurfaceContext`] therefore prefers a
//! linear-storage format and passes the offscreen bytes through unchanged.

use crate::gpu::{GpuContext, GpuError};
use std::sync::Arc;

/// A window surface together with the device configured for it.
pub struct SurfaceContext {
    /// The device and queue, shared with the rest of the renderer.
    pub gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl SurfaceContext {
    /// Creates a surface for `target` and acquires a device that can present
    /// to it.
    ///
    /// `width` and `height` are the window's physical pixel size. A zero in
    /// either is clamped to one, because a minimised window still has to
    /// produce a valid swapchain configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no backend can draw to the target,
    /// [`GpuError::DeviceRequest`] when the adapter refuses a device, and
    /// [`GpuError::Operation`] when the surface itself cannot be created or is
    /// unsupported by the chosen adapter.
    pub fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<SurfaceContext, GpuError> {
        pollster::block_on(SurfaceContext::new_async(target, width, height))
    }

    /// Async form of [`SurfaceContext::new`].
    ///
    /// # Errors
    ///
    /// See [`SurfaceContext::new`].
    pub async fn new_async(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<SurfaceContext, GpuError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(descriptor);

        let surface = instance
            .create_surface(target)
            .map_err(|error| GpuError::Operation(error.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::NoAdapter(error.to_string()))?;

        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("verdant windowed device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::DeviceRequest(error.to_string()))?;

        let capabilities = surface.get_capabilities(&adapter);
        let format = preferred_format(&capabilities.formats)
            .ok_or_else(|| GpuError::Operation("surface offers no usable format".to_owned()))?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            // The presenter writes the offscreen bytes through unchanged, so
            // the swapchain must not apply an encoding of its own.
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            // Fifo is guaranteed on every backend and is what a fixed-timestep
            // game wants: the simulation already paces itself, so tearing buys
            // nothing.
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: Vec::new(),
        };

        let gpu = GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        };
        surface.configure(&gpu.device, &config);

        Ok(SurfaceContext {
            gpu,
            surface,
            config,
        })
    }

    /// The swapchain's texture format.
    ///
    /// Pipelines that draw to the window must be created against this.
    #[must_use]
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The surface's current size in physical pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after the window changes size.
    ///
    /// A zero in either dimension is ignored rather than clamped: a minimised
    /// window should keep its last good configuration and simply stop drawing.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Reapplies the current configuration, which recovers a lost swapchain.
    pub fn reconfigure(&self) {
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Acquires the next frame to draw into, recovering the swapchain if it
    /// has gone stale.
    ///
    /// Returns `None` when this frame should simply be skipped — the window is
    /// occluded, the compositor timed out, or the swapchain had to be rebuilt.
    /// None of those are errors: they happen routinely while a window is being
    /// resized or minimised, and the right response is to try again next frame
    /// rather than to tear down the renderer.
    pub fn acquire(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => Some(frame),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Usable now, but the configuration no longer matches the
                // surface. Draw this frame, then rebuild for the next one.
                self.reconfigure();
                Some(frame)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => None,
        }
    }
}

impl std::fmt::Debug for SurfaceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceContext")
            .field("format", &self.config.format)
            .field("size", &(self.config.width, self.config.height))
            .field("adapter", &self.gpu.adapter_info.name)
            .finish()
    }
}

/// Chooses the swapchain format, preferring one that stores the offscreen
/// bytes unchanged.
///
/// Exposed to the crate so the choice can be tested without a window.
pub(crate) fn preferred_format(
    formats: &[wgpu::TextureFormat],
) -> Option<wgpu::TextureFormat> {
    // In preference order: the offscreen format itself, its swapped-channel
    // twin, then any other non-sRGB format, then whatever is on offer.
    const LINEAR: [wgpu::TextureFormat; 2] = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ];
    for wanted in LINEAR {
        if formats.contains(&wanted) {
            return Some(wanted);
        }
    }
    formats
        .iter()
        .find(|format| !format.is_srgb())
        .or_else(|| formats.first())
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::TextureFormat;

    #[test]
    fn the_offscreen_format_is_preferred_when_offered() {
        let offered = [
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8Unorm,
        ];
        assert_eq!(preferred_format(&offered), Some(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn a_swapped_channel_linear_format_is_the_next_choice() {
        let offered = [TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm];
        assert_eq!(preferred_format(&offered), Some(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn any_linear_format_beats_an_srgb_one() {
        let offered = [TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgb10a2Unorm];
        assert_eq!(
            preferred_format(&offered),
            Some(TextureFormat::Rgb10a2Unorm)
        );
    }

    #[test]
    fn an_srgb_only_surface_still_gets_a_format() {
        // Correct-looking is better than not drawing at all, so an sRGB-only
        // surface is accepted rather than refused.
        let offered = [TextureFormat::Bgra8UnormSrgb];
        assert_eq!(
            preferred_format(&offered),
            Some(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn a_surface_with_no_formats_yields_none() {
        assert_eq!(preferred_format(&[]), None);
    }
}
