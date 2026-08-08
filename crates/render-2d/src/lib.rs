//! # Verdant 2D renderer
//!
//! An instanced sprite renderer built on wgpu, designed around pixel art.
//!
//! ## The one-draw-call design
//!
//! Everything that varies per sprite — transform, UV window, tint, artwork —
//! travels as instance data, and the artwork lives in a `texture_2d_array`
//! whose layer is selected by an integer per instance. Nothing about drawing a
//! sprite requires changing a bind group, so an entire frame's worth of sprites
//! is one `draw_indexed` call with a large instance count.
//!
//! ## Pixel-perfect rendering
//!
//! Three things together keep pixel art crisp, and all three are necessary:
//!
//! 1. The world renders into a low-resolution [`RenderTarget`], never directly
//!    at window size.
//! 2. The camera snaps to whole pixels ([`Camera2D`]), so texels never land on
//!    fractional positions and shimmer as the view scrolls.
//! 3. The target is scaled to the window by a whole number ([`integer_scale`]),
//!    with the remainder letterboxed rather than stretched.
//!
//! ## Colour
//!
//! Colours are linear on the GPU and converted from sRGB once at authoring
//! time via [`Color::from_srgb_hex`]. Blending sRGB values directly produces
//! visibly dark edges wherever sprites overlap.
//!
//! ## Testing
//!
//! The renderer has no window dependency: [`GpuContext::headless`] acquires a
//! device with no surface, and [`RenderTarget::read_pixels`] copies a frame
//! back to the CPU. That is what lets the golden-image tests run in CI against
//! Mesa's lavapipe software Vulkan driver.
//!
//! ## Example
//!
//! ```no_run
//! use verdant_core_math::{fx, Vec2};
//! use verdant_render_2d::{
//!     Camera2D, Color, DrawSprite, FrameSettings, GpuContext, RenderTarget, SpriteBatcher,
//!     SpriteRenderer, TextureArray,
//! };
//!
//! let gpu = GpuContext::headless()?;
//! let atlas = TextureArray::new(&gpu, 16, 16, 1);
//! atlas.fill_layer(&gpu, 0, Color::from_srgb_hex(0x8F_BC_5A))?;
//!
//! let mut renderer = SpriteRenderer::new(&gpu);
//! let bind_group = renderer.bind_atlas(&gpu, &atlas);
//! let target = RenderTarget::new(&gpu, 320, 180);
//!
//! let mut batcher = SpriteBatcher::new();
//! batcher.push(DrawSprite::new(Vec2::from_ints(160, 120), Vec2::from_ints(16, 16), 0));
//!
//! renderer.render(
//!     &gpu,
//!     &target,
//!     &bind_group,
//!     &Camera2D::new(320, 180),
//!     batcher.prepare(),
//!     FrameSettings::default(),
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_no_source)]

pub mod camera;
pub mod gpu;
pub mod present;
pub mod renderer;
pub mod sprite;
pub mod surface;
pub mod texture;

pub use camera::Camera2D;
pub use gpu::{GpuContext, GpuError};
pub use present::Presenter;
pub use renderer::{integer_scale, letterbox, FrameSettings, RenderTarget, SpriteRenderer};
pub use sprite::{Color, DrawSprite, SpriteBatcher, SpriteInstance};
pub use surface::SurfaceContext;
pub use texture::TextureArray;
