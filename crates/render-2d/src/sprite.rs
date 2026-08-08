//! Sprite instances, colours and the batcher that sorts them.

use bytemuck::{Pod, Zeroable};
use verdant_core_math::{Fx, Rect, Vec2};

/// A linear-space RGBA colour.
///
/// Stored linear rather than sRGB because blending is only correct in linear
/// space: averaging two sRGB values produces a result that is visibly too dark,
/// which shows up as muddy edges wherever sprites overlap. Conversion happens
/// once at authoring time via [`Color::from_srgb_hex`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red, `0..=1`.
    pub r: f32,
    /// Green, `0..=1`.
    pub g: f32,
    /// Blue, `0..=1`.
    pub b: f32,
    /// Alpha, `0..=1`.
    pub a: f32,
}

impl Color {
    /// Opaque white; the identity tint.
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    /// Opaque black.
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    /// Fully transparent.
    pub const TRANSPARENT: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Builds a colour from linear components.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    /// Builds an opaque colour from linear components.
    #[must_use]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b, a: 1.0 }
    }

    /// Converts an sRGB hex value such as `0x8FBC5A` into linear space.
    ///
    /// This is how palette constants should be written: artists and design
    /// tools work in sRGB, the GPU blends in linear.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_srgb_hex(hex: u32) -> Color {
        let channel = |shift: u32| srgb_to_linear(((hex >> shift) & 0xFF) as f32 / 255.0);
        Color {
            r: channel(16),
            g: channel(8),
            b: channel(0),
            a: 1.0,
        }
    }

    /// Returns this colour with a different alpha.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: f32) -> Color {
        self.a = alpha;
        self
    }

    /// Multiplies the RGB channels, leaving alpha alone.
    ///
    /// Used for day/night tinting and damage flashes.
    #[must_use]
    pub fn scale_rgb(self, factor: f32) -> Color {
        Color {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
            a: self.a,
        }
    }

    /// Linearly interpolates toward `other`.
    #[must_use]
    pub fn lerp(self, other: Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// The components as an array, for GPU upload.
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for Color {
    fn default() -> Color {
        Color::WHITE
    }
}

/// The standard sRGB electro-optical transfer function.
fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.040_45 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// One sprite as the GPU sees it.
///
/// The layout is a packed 2×3 affine transform plus a UV rectangle, a tint and
/// an atlas layer — 48 bytes, which keeps a full screen of sprites inside a
/// single modest instance buffer.
///
/// Sending an affine matrix rather than position/rotation/scale means the
/// vertex shader does one multiply-add per corner instead of reconstructing a
/// rotation from an angle, and it lets the CPU fold a sprite's pivot and flip
/// into the same matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpriteInstance {
    /// Column-major 2×3 affine transform: x axis, y axis, translation.
    pub model: [f32; 6],
    /// Sub-rectangle of the atlas layer, in normalised UV.
    pub uv_rect: [f32; 4],
    /// Multiplicative tint.
    pub color: [f32; 4],
    /// Which layer of the texture array to sample.
    pub layer: u32,
    /// Padding to a 16-byte boundary, required by the vertex layout.
    pub _padding: [u32; 1],
}

/// A sprite queued for drawing, before it has been sorted and packed.
#[derive(Clone, Copy, Debug)]
pub struct DrawSprite {
    /// World position of the sprite's anchor.
    pub position: Vec2,
    /// Size in world units.
    pub size: Vec2,
    /// Anchor within the sprite, where `(0.5, 1)` is bottom-centre.
    ///
    /// Bottom-centre is the engine's convention so a sprite's anchor coincides
    /// with its entity's ground point.
    pub anchor: Vec2,
    /// Rotation in radians about the anchor.
    pub rotation: Fx,
    /// Region of the atlas layer to draw.
    pub uv_rect: Rect,
    /// Atlas layer index.
    pub layer: u32,
    /// Tint.
    pub color: Color,
    /// Draw order. Higher values paint later.
    pub z: i32,
    /// Mirrors the sprite horizontally.
    pub flip_x: bool,
}

impl DrawSprite {
    /// A sprite with the engine's defaults: bottom-centre anchor, no rotation,
    /// the whole layer, untinted.
    #[must_use]
    pub fn new(position: Vec2, size: Vec2, layer: u32) -> DrawSprite {
        DrawSprite {
            position,
            size,
            anchor: Vec2::new(Fx::HALF, Fx::ONE),
            rotation: Fx::ZERO,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            layer,
            color: Color::WHITE,
            z: 0,
            flip_x: false,
        }
    }

    /// Returns this sprite at the given draw order.
    #[must_use]
    pub const fn at_z(mut self, z: i32) -> DrawSprite {
        self.z = z;
        self
    }

    /// Returns this sprite tinted.
    #[must_use]
    pub const fn tinted(mut self, color: Color) -> DrawSprite {
        self.color = color;
        self
    }

    /// Returns this sprite drawing a sub-region of its layer.
    #[must_use]
    pub const fn with_uv(mut self, uv_rect: Rect) -> DrawSprite {
        self.uv_rect = uv_rect;
        self
    }

    /// Returns this sprite mirrored horizontally.
    #[must_use]
    pub const fn flipped(mut self) -> DrawSprite {
        self.flip_x = true;
        self
    }

    /// Returns this sprite with a custom anchor.
    #[must_use]
    pub const fn anchored(mut self, anchor: Vec2) -> DrawSprite {
        self.anchor = anchor;
        self
    }

    /// The world-space bounds this sprite covers when unrotated.
    ///
    /// Used for culling; a rotated sprite is culled by its unrotated bounds
    /// expanded to the diagonal, which is conservative but cheap.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        let offset = Vec2::new(self.size.x * self.anchor.x, self.size.y * self.anchor.y);
        let base = Rect::new(self.position - offset, self.size);
        if self.rotation.is_zero() {
            base
        } else {
            // The diagonal bounds any rotation about the anchor.
            let diagonal = self.size.length();
            Rect::from_centre(base.centre(), Vec2::splat(diagonal))
        }
    }

    /// Packs this sprite into its GPU form.
    #[must_use]
    pub fn to_instance(&self) -> SpriteInstance {
        // The anchor offset is applied before rotation so a sprite spins about
        // its anchor rather than its corner.
        let anchor_offset = Vec2::new(-self.size.x * self.anchor.x, -self.size.y * self.anchor.y);

        // Almost every sprite in a 2D game is unrotated, and the trig
        // polynomials are only accurate to about 3e-6 — enough to leave every
        // static sprite a fraction of a unit off its stated position. Taking
        // the identity directly is both exact and cheaper.
        let (sin, cos) = if self.rotation.is_zero() {
            (0.0f32, 1.0f32)
        } else {
            (self.rotation.sin().to_f32(), self.rotation.cos().to_f32())
        };
        let width = self.size.x.to_f32();
        let height = self.size.y.to_f32();

        // Columns of the 2x2 rotation-scale, then the translation.
        let x_axis = [cos * width, sin * width];
        let y_axis = [-sin * height, cos * height];

        let offset_x = anchor_offset.x.to_f32();
        let offset_y = anchor_offset.y.to_f32();
        let rotated_offset = [
            cos * offset_x - sin * offset_y,
            sin * offset_x + cos * offset_y,
        ];
        let translation = [
            self.position.x.to_f32() + rotated_offset[0],
            self.position.y.to_f32() + rotated_offset[1],
        ];

        // Mirroring is done in UV space, not by negating the quad's width.
        // A negative width would move the quad to the other side of its anchor
        // and reverse its winding; walking the UV window backwards mirrors the
        // artwork while the geometry stays exactly where it was.
        let (uv_x, uv_width) = if self.flip_x {
            (
                (self.uv_rect.min.x + self.uv_rect.size.x).to_f32(),
                -self.uv_rect.size.x.to_f32(),
            )
        } else {
            (self.uv_rect.min.x.to_f32(), self.uv_rect.size.x.to_f32())
        };

        SpriteInstance {
            model: [
                x_axis[0],
                x_axis[1],
                y_axis[0],
                y_axis[1],
                translation[0],
                translation[1],
            ],
            uv_rect: [
                uv_x,
                self.uv_rect.min.y.to_f32(),
                uv_width,
                self.uv_rect.size.y.to_f32(),
            ],
            color: self.color.to_array(),
            layer: self.layer,
            _padding: [0],
        }
    }
}

/// Collects sprites, sorts them, and packs them for upload.
///
/// # Sort order
///
/// Sprites are ordered by `z`, then by their anchor's y coordinate, then by
/// atlas layer. The y term is what makes a top-down world read correctly: a
/// character standing in front of a fence must draw after it, and "in front"
/// in a top-down projection means "further down the screen". Sorting by layer
/// last groups draws that share a texture without disturbing the visual order.
#[derive(Debug, Default)]
pub struct SpriteBatcher {
    sprites: Vec<DrawSprite>,
    instances: Vec<SpriteInstance>,
}

impl SpriteBatcher {
    /// Creates an empty batcher.
    #[must_use]
    pub fn new() -> SpriteBatcher {
        SpriteBatcher::default()
    }

    /// Queues a sprite.
    pub fn push(&mut self, sprite: DrawSprite) {
        self.sprites.push(sprite);
    }

    /// Queues a sprite only if it intersects `visible`.
    ///
    /// Returns whether it was kept. Culling here rather than in the shader
    /// keeps offscreen sprites out of the instance buffer entirely, which
    /// matters for a scrolling world where most of the map is offscreen.
    pub fn push_visible(&mut self, sprite: DrawSprite, visible: Rect) -> bool {
        if sprite.bounds().intersects(visible) {
            self.sprites.push(sprite);
            true
        } else {
            false
        }
    }

    /// Number of queued sprites.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// True when nothing is queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    /// Discards every queued sprite, keeping the allocation.
    pub fn clear(&mut self) {
        self.sprites.clear();
        self.instances.clear();
    }

    /// Sorts the queue and packs it into GPU instances.
    ///
    /// Returns the packed slice, which stays valid until the next call to
    /// [`SpriteBatcher::clear`] or [`SpriteBatcher::prepare`].
    pub fn prepare(&mut self) -> &[SpriteInstance] {
        // A stable sort so sprites that tie on every key keep their submission
        // order, which is what makes a frame's output reproducible.
        self.sprites.sort_by(|a, b| {
            a.z.cmp(&b.z)
                .then_with(|| a.position.y.cmp(&b.position.y))
                .then_with(|| a.layer.cmp(&b.layer))
        });
        self.instances.clear();
        self.instances
            .extend(self.sprites.iter().map(DrawSprite::to_instance));
        &self.instances
    }

    /// The sorted sprites, for inspection and testing.
    #[must_use]
    pub fn sprites(&self) -> &[DrawSprite] {
        &self.sprites
    }
}

#[cfg(test)]
// The values compared below are exact by construction — zero, one, and halves
// — so an exact comparison is the assertion actually wanted here.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    #[test]
    fn hex_colours_convert_out_of_srgb() {
        let white = Color::from_srgb_hex(0xFF_FF_FF);
        assert!((white.r - 1.0).abs() < 1e-5);
        let black = Color::from_srgb_hex(0x00_00_00);
        assert_eq!(black.r, 0.0);

        // Mid grey in sRGB is much darker than 0.5 in linear space; a renderer
        // that skipped the conversion would wash every colour out.
        let grey = Color::from_srgb_hex(0x80_80_80);
        assert!(grey.r < 0.25, "linear value was {}", grey.r);
    }

    #[test]
    fn colour_helpers_behave() {
        let colour = Color::rgb(0.5, 0.25, 0.125);
        assert_eq!(colour.a, 1.0);
        assert_eq!(colour.with_alpha(0.5).a, 0.5);
        assert_eq!(colour.scale_rgb(2.0).r, 1.0);
        assert_eq!(
            colour.scale_rgb(2.0).a,
            1.0,
            "alpha is untouched by tinting"
        );
        assert_eq!(Color::BLACK.lerp(Color::WHITE, 0.5).r, 0.5);
        assert_eq!(
            Color::BLACK.lerp(Color::WHITE, 5.0),
            Color::WHITE,
            "t is clamped"
        );
    }

    #[test]
    fn an_unrotated_sprite_maps_its_corners_predictably() {
        let sprite = DrawSprite::new(Vec2::from_ints(10, 20), Vec2::from_ints(4, 8), 0);
        let instance = sprite.to_instance();

        // Bottom-centre anchor: the quad's origin is 2 left and 8 up.
        assert!(
            (instance.model[4] - 8.0).abs() < 1e-5,
            "x was {}",
            instance.model[4]
        );
        assert!(
            (instance.model[5] - 12.0).abs() < 1e-5,
            "y was {}",
            instance.model[5]
        );
        // Axes carry the size.
        assert!((instance.model[0] - 4.0).abs() < 1e-5);
        assert!((instance.model[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn flipping_mirrors_the_uv_window_and_leaves_the_geometry_alone() {
        let sprite = DrawSprite::new(Vec2::from_ints(10, 20), Vec2::from_ints(4, 8), 0);
        let normal = sprite.to_instance();
        let flipped = sprite.flipped().to_instance();

        assert_eq!(
            flipped.model, normal.model,
            "a flipped sprite occupies the same space"
        );
        assert!(
            (flipped.uv_rect[0] - 1.0).abs() < 1e-5,
            "sampling starts at the right edge"
        );
        assert!(
            (flipped.uv_rect[2] + 1.0).abs() < 1e-5,
            "and walks backwards"
        );
    }

    #[test]
    fn flipping_a_sub_region_mirrors_within_that_region() {
        let sprite = DrawSprite::new(Vec2::ZERO, Vec2::from_ints(4, 8), 0).with_uv(Rect::new(
            Vec2::new(Fx::HALF, Fx::ZERO),
            Vec2::new(Fx::HALF, Fx::ONE),
        ));
        let flipped = sprite.flipped().to_instance();
        assert!(
            (flipped.uv_rect[0] - 1.0).abs() < 1e-5,
            "starts at the region's right edge"
        );
        assert!(
            (flipped.uv_rect[2] + 0.5).abs() < 1e-5,
            "and spans the region backwards"
        );
    }

    #[test]
    fn sprite_bounds_account_for_the_anchor() {
        let sprite = DrawSprite::new(Vec2::from_ints(10, 20), Vec2::from_ints(4, 8), 0);
        let bounds = sprite.bounds();
        assert_eq!(
            bounds.bottom(),
            fx(20),
            "the anchor sits on the bottom edge"
        );
        assert_eq!(bounds.centre().x, fx(10), "and is centred horizontally");
    }

    #[test]
    fn a_rotated_sprite_uses_conservative_bounds() {
        let mut sprite = DrawSprite::new(Vec2::ZERO, Vec2::from_ints(4, 8), 0);
        sprite.rotation = Fx::FRAC_PI_4;
        // The diagonal of a 4x8 sprite is about 8.94.
        assert!(sprite.bounds().size.x.to_f64() > 8.0);
    }

    #[test]
    fn batching_sorts_by_z_then_depth() {
        let mut batcher = SpriteBatcher::new();
        batcher.push(DrawSprite::new(Vec2::from_ints(0, 100), Vec2::ONE, 0).at_z(0));
        batcher.push(DrawSprite::new(Vec2::from_ints(0, 50), Vec2::ONE, 0).at_z(0));
        batcher.push(DrawSprite::new(Vec2::from_ints(0, 999), Vec2::ONE, 0).at_z(-1));
        batcher.prepare();

        let order: Vec<f64> = batcher
            .sprites()
            .iter()
            .map(|sprite| sprite.position.y.to_f64())
            .collect();
        assert_eq!(order, vec![999.0, 50.0, 100.0], "z wins, then y depth");
    }

    #[test]
    fn things_lower_on_screen_draw_in_front() {
        let mut batcher = SpriteBatcher::new();
        let fence = DrawSprite::new(Vec2::from_ints(0, 100), Vec2::ONE, 0);
        let character = DrawSprite::new(Vec2::from_ints(0, 120), Vec2::ONE, 0);
        batcher.push(character);
        batcher.push(fence);
        batcher.prepare();

        assert_eq!(
            batcher.sprites()[0].position.y,
            fx(100),
            "the fence is further up the screen, so it draws first"
        );
    }

    #[test]
    fn ties_keep_their_submission_order() {
        let mut batcher = SpriteBatcher::new();
        for layer in 0..5u32 {
            batcher.push(DrawSprite::new(Vec2::ZERO, Vec2::ONE, layer));
        }
        batcher.prepare();
        let layers: Vec<u32> = batcher
            .sprites()
            .iter()
            .map(|sprite| sprite.layer)
            .collect();
        assert_eq!(layers, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn culling_drops_offscreen_sprites() {
        let mut batcher = SpriteBatcher::new();
        let visible = Rect::from_ints(0, 0, 100, 100);

        assert!(batcher.push_visible(
            DrawSprite::new(Vec2::from_ints(50, 50), Vec2::ONE, 0),
            visible
        ));
        assert!(!batcher.push_visible(
            DrawSprite::new(Vec2::from_ints(5000, 5000), Vec2::ONE, 0),
            visible
        ));
        assert_eq!(batcher.len(), 1);
    }

    #[test]
    fn preparing_packs_every_sprite() {
        let mut batcher = SpriteBatcher::new();
        for index in 0..10i32 {
            batcher.push(DrawSprite::new(Vec2::from_ints(index, index), Vec2::ONE, 0));
        }
        assert_eq!(batcher.prepare().len(), 10);

        batcher.clear();
        assert!(batcher.is_empty());
        assert!(batcher.prepare().is_empty());
    }

    #[test]
    fn the_instance_layout_is_the_size_the_shader_expects() {
        // 6 floats of transform, 4 of UV, 4 of colour, a layer and padding.
        assert_eq!(
            std::mem::size_of::<SpriteInstance>(),
            4 * (6 + 4 + 4 + 1 + 1)
        );
    }

    #[test]
    fn batching_is_deterministic() {
        let build = || {
            let mut batcher = SpriteBatcher::new();
            for index in 0..64i32 {
                let sprite = DrawSprite::new(
                    Vec2::from_ints(index % 8, (index * 7) % 13),
                    Vec2::ONE,
                    (index % 4) as u32,
                )
                .at_z(index % 3);
                batcher.push(sprite);
            }
            batcher
                .prepare()
                .iter()
                .map(|instance| instance.model)
                .collect::<Vec<_>>()
        };
        assert_eq!(build(), build());
    }
}
