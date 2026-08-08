//! Texture arrays, the atlas strategy the renderer batches against.
//!
//! # Why a texture array rather than one big atlas page
//!
//! Packing everything into a single large 2D texture is the traditional
//! approach, but it forces every sprite to carry a UV rectangle within a page
//! whose layout changes whenever art is added, and it makes tiling and
//! wrapping impossible without bleeding into a neighbour.
//!
//! A `texture_2d_array` gives each sprite (or each animation strip) its own
//! layer at a uniform size. Every sprite in the world can then be drawn in one
//! call regardless of which layer it samples, because the layer index is per
//! *instance* rather than per bind group. That is what collapses a frame down
//! to a single draw.
//!
//! The constraint is that all layers share one size. For a pixel-art game whose
//! art is authored on a fixed grid that is a feature, not a limitation.

use crate::gpu::{GpuContext, GpuError};
use crate::sprite::Color;

/// A uniform-size array of texture layers.
pub struct TextureArray {
    /// The GPU texture.
    texture: wgpu::Texture,
    /// A view over every layer.
    view: wgpu::TextureView,
    /// The sampler layers are read through.
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
    layers: u32,
}

impl TextureArray {
    /// Creates an empty array of `layers` transparent layers.
    ///
    /// Sampling is nearest-neighbour with no mipmaps: pixel art must not be
    /// filtered, or every sprite turns to mush the moment it is scaled.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(gpu: &GpuContext, width: u32, height: u32, layers: u32) -> TextureArray {
        assert!(
            width > 0 && height > 0 && layers > 0,
            "TextureArray: dimensions and layer count must be non-zero"
        );

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("verdant sprite atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Non-sRGB: the CPU converts palettes to linear once at authoring
            // time, so the GPU must not apply the transfer function again.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("verdant sprite atlas view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("verdant nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        TextureArray {
            texture,
            view,
            sampler,
            width,
            height,
            layers,
        }
    }

    /// Uploads RGBA8 pixel data into one layer.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Operation`] when the layer index is out of range or
    /// the data length does not match the layer's size — both of which would
    /// otherwise corrupt whatever happened to follow in the buffer.
    pub fn write_layer(&self, gpu: &GpuContext, layer: u32, pixels: &[u8]) -> Result<(), GpuError> {
        if layer >= self.layers {
            return Err(GpuError::Operation(format!(
                "layer {layer} is out of range for an array of {} layers",
                self.layers
            )));
        }
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if pixels.len() != expected {
            return Err(GpuError::Operation(format!(
                "layer data is {} bytes but {}x{} RGBA needs {expected}",
                pixels.len(),
                self.width,
                self.height
            )));
        }

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Fills a layer with a single colour.
    ///
    /// Useful for solid-colour sprites — shadows, UI panels, damage flashes —
    /// which would otherwise need generated artwork.
    ///
    /// # Errors
    ///
    /// See [`TextureArray::write_layer`].
    pub fn fill_layer(&self, gpu: &GpuContext, layer: u32, color: Color) -> Result<(), GpuError> {
        let quantise = |channel: f32| {
            // Clamped before the cast so an out-of-range colour saturates
            // rather than wrapping to an unrelated value.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                (channel.clamp(0.0, 1.0) * 255.0).round() as u8
            }
        };
        let texel = [
            quantise(color.r),
            quantise(color.g),
            quantise(color.b),
            quantise(color.a),
        ];
        let pixels: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((self.width as usize) * (self.height as usize) * 4)
            .collect();
        self.write_layer(gpu, layer, &pixels)
    }

    /// The array view, for binding.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The sampler, for binding.
    #[must_use]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Layer width in texels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Layer height in texels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of layers.
    #[must_use]
    pub const fn layers(&self) -> u32 {
        self.layers
    }
}

impl std::fmt::Debug for TextureArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureArray")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layers", &self.layers)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acquires a device, or returns `None` where none exists.
    fn gpu() -> Option<GpuContext> {
        GpuContext::headless().ok()
    }

    #[test]
    fn a_texture_array_reports_its_shape() {
        let Some(gpu) = gpu() else {
            eprintln!("no graphics adapter; skipping");
            return;
        };
        let array = TextureArray::new(&gpu, 16, 16, 4);
        assert_eq!((array.width(), array.height(), array.layers()), (16, 16, 4));
    }

    #[test]
    fn writing_a_layer_of_the_right_size_succeeds() {
        let Some(gpu) = gpu() else {
            eprintln!("no graphics adapter; skipping");
            return;
        };
        let array = TextureArray::new(&gpu, 8, 8, 2);
        let pixels = vec![255u8; 8 * 8 * 4];
        assert!(array.write_layer(&gpu, 0, &pixels).is_ok());
        assert!(array.write_layer(&gpu, 1, &pixels).is_ok());
    }

    #[test]
    fn writing_out_of_range_is_rejected_rather_than_corrupting() {
        let Some(gpu) = gpu() else {
            eprintln!("no graphics adapter; skipping");
            return;
        };
        let array = TextureArray::new(&gpu, 8, 8, 2);
        let pixels = vec![0u8; 8 * 8 * 4];

        let error = array.write_layer(&gpu, 5, &pixels).unwrap_err();
        assert!(error.to_string().contains("out of range"));

        let error = array.write_layer(&gpu, 0, &pixels[..10]).unwrap_err();
        assert!(error.to_string().contains("bytes"));
    }

    #[test]
    fn a_layer_can_be_filled_with_a_flat_colour() {
        let Some(gpu) = gpu() else {
            eprintln!("no graphics adapter; skipping");
            return;
        };
        let array = TextureArray::new(&gpu, 4, 4, 1);
        assert!(array.fill_layer(&gpu, 0, Color::WHITE).is_ok());
        gpu.wait_for_idle().expect("the upload should complete");
    }
}
