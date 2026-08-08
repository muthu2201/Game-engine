//! End-to-end rendering tests.
//!
//! These render real frames on a real device and inspect the pixels that come
//! back. In CI the device is Mesa's lavapipe software Vulkan implementation, so
//! the whole pipeline — shader compilation, vertex layout, blending, the atlas
//! bindings — is exercised on a machine with no GPU.
//!
//! # Why assertions about content, not a stored reference image
//!
//! A byte-for-byte reference PNG would be the strictest check, but it also
//! fails for reasons that have nothing to do with correctness: a driver update
//! changes rasterisation by a subpixel, and every test goes red at once. These
//! tests instead assert the properties a correct frame must have — this pixel
//! is this colour, this sprite occludes that one, a flip mirrors the image —
//! which catches real regressions and survives driver churn. Where an exact
//! image *is* wanted, [`render_to_png`] writes one out for inspection.

use verdant_core_math::{fx, Fx, Rect, Vec2};
use verdant_render_2d::{
    Camera2D, Color, Compositor, DrawSprite, FrameSettings, GpuContext, Light, LightRenderer,
    LightSettings, Presenter, RenderTarget, SpriteBatcher, SpriteRenderer, TextureArray,
};

/// The scene fixture the tests draw with.
struct Harness {
    gpu: GpuContext,
    renderer: SpriteRenderer,
    atlas: TextureArray,
    bind_group: wgpu::BindGroup,
    target: RenderTarget,
}

impl Harness {
    /// Builds a harness, or `None` when no adapter is available.
    ///
    /// Returning `None` rather than panicking means the suite still passes on a
    /// machine with no graphics stack at all, while CI — which installs
    /// lavapipe — always exercises the real path.
    fn new(width: u32, height: u32, layers: u32) -> Option<Harness> {
        let gpu = match GpuContext::headless() {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skipping renderer test: {error}");
                return None;
            }
        };
        eprintln!("rendering on {}", gpu.describe());

        let atlas = TextureArray::new(&gpu, 8, 8, layers);
        let renderer = SpriteRenderer::new(&gpu);
        let bind_group = renderer.bind_atlas(&gpu, &atlas);
        let target = RenderTarget::new(&gpu, width, height);
        Some(Harness {
            gpu,
            renderer,
            atlas,
            bind_group,
            target,
        })
    }

    /// Renders a batch and reads the frame back.
    fn render(&mut self, camera: &Camera2D, batcher: &mut SpriteBatcher, clear: Color) -> Frame {
        self.render_tinted(camera, batcher, clear, Color::WHITE)
    }

    /// Renders with a global tint applied.
    fn render_tinted(
        &mut self,
        camera: &Camera2D,
        batcher: &mut SpriteBatcher,
        clear: Color,
        tint: Color,
    ) -> Frame {
        self.renderer.render(
            &self.gpu,
            &self.target,
            &self.bind_group,
            camera,
            batcher.prepare(),
            FrameSettings {
                clear: Some(clear),
                global_tint: tint,
            },
        );
        let pixels = self
            .target
            .read_pixels(&self.gpu)
            .expect("the frame should read back");
        Frame {
            pixels,
            width: self.target.width(),
            height: self.target.height(),
        }
    }
}

/// A frame that has been read back to the CPU.
struct Frame {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Frame {
    /// The RGBA bytes at a pixel.
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) is outside the frame"
        );
        let index = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
            self.pixels[index + 3],
        ]
    }

    /// How many pixels are not the given colour.
    fn count_differing_from(&self, colour: [u8; 4]) -> usize {
        self.pixels
            .chunks_exact(4)
            .filter(|pixel| *pixel != colour)
            .count()
    }

    /// Writes the frame to a PNG, for eyeballing a failure.
    #[allow(dead_code)]
    fn write_png(&self, path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::File::create(path).expect("the diff directory is writable");
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .expect("the PNG header is valid")
            .write_image_data(&self.pixels)
            .expect("the pixel data matches the header");
    }
}

/// Channels are compared with a tolerance: a software rasteriser and a hardware
/// GPU can disagree by a least-significant bit on a blended edge, and a test
/// that failed on that would be measuring the driver rather than the renderer.
fn assert_colour_near(actual: [u8; 4], expected: [u8; 4], context: &str) {
    let close = actual
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| a.abs_diff(*b) <= 2);
    assert!(
        close,
        "{context}: got {actual:?}, expected about {expected:?}"
    );
}

#[test]
fn an_empty_scene_renders_the_clear_colour() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let mut batcher = SpriteBatcher::new();
    let frame = harness.render(&Camera2D::new(64, 64), &mut batcher, Color::BLACK);

    assert_eq!(
        frame.count_differing_from([0, 0, 0, 255]),
        0,
        "nothing should have been drawn"
    );
}

#[test]
fn the_clear_colour_is_honoured() {
    let Some(mut harness) = Harness::new(16, 16, 1) else {
        return;
    };
    let mut batcher = SpriteBatcher::new();
    // A colour with exactly representable channels, so quantisation is exact.
    let clear = Color::new(1.0, 0.0, 0.0, 1.0);
    let frame = harness.render(&Camera2D::new(16, 16), &mut batcher, clear);
    assert_colour_near(frame.at(8, 8), [255, 0, 0, 255], "the cleared background");
}

#[test]
fn a_sprite_appears_where_the_camera_puts_it() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::new(0.0, 1.0, 0.0, 1.0))
        .expect("the atlas layer is writable");

    let mut camera = Camera2D::new(64, 64);
    camera.position = Vec2::from_ints(32, 32);

    let mut batcher = SpriteBatcher::new();
    // A 16x16 sprite centred in the view: bottom-centre anchored at (32, 40)
    // puts it over rows 24..40, columns 24..40.
    batcher.push(DrawSprite::new(
        Vec2::from_ints(32, 40),
        Vec2::from_ints(16, 16),
        0,
    ));

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);

    assert_colour_near(frame.at(32, 32), [0, 255, 0, 255], "the sprite's centre");
    assert_colour_near(
        frame.at(2, 2),
        [0, 0, 0, 255],
        "a corner well outside the sprite",
    );
}

#[test]
fn a_tint_multiplies_the_sprite() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    batcher.push(
        DrawSprite::new(Vec2::from_ints(16, 24), Vec2::from_ints(16, 16), 0)
            .tinted(Color::new(1.0, 0.0, 0.0, 1.0)),
    );

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);
    assert_colour_near(
        frame.at(16, 16),
        [255, 0, 0, 255],
        "a white sprite tinted red",
    );
}

#[test]
fn the_global_tint_darkens_the_whole_scene() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    batcher.push(DrawSprite::new(
        Vec2::from_ints(16, 24),
        Vec2::from_ints(16, 16),
        0,
    ));

    // This is how the day/night cycle is applied: one uniform, every sprite.
    let night = Color::new(0.5, 0.5, 0.5, 1.0);
    let frame = harness.render_tinted(&camera, &mut batcher, Color::BLACK, night);
    assert_colour_near(
        frame.at(16, 16),
        [128, 128, 128, 255],
        "a half-tinted white sprite",
    );
}

#[test]
fn sprites_draw_in_depth_order() {
    let Some(mut harness) = Harness::new(32, 32, 2) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::new(1.0, 0.0, 0.0, 1.0))
        .expect("writable");
    harness
        .atlas
        .fill_layer(&harness.gpu, 1, Color::new(0.0, 0.0, 1.0, 1.0))
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    // The blue sprite is further down the screen, so it must draw in front.
    batcher.push(DrawSprite::new(
        Vec2::from_ints(16, 20),
        Vec2::from_ints(20, 20),
        1,
    ));
    batcher.push(DrawSprite::new(
        Vec2::from_ints(16, 18),
        Vec2::from_ints(20, 20),
        0,
    ));

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);
    assert_colour_near(
        frame.at(16, 16),
        [0, 0, 255, 255],
        "the nearer sprite occludes the further",
    );
}

#[test]
fn transparent_texels_do_not_paint_over_the_background() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::TRANSPARENT)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    batcher.push(DrawSprite::new(
        Vec2::from_ints(16, 24),
        Vec2::from_ints(16, 16),
        0,
    ));

    let clear = Color::new(0.0, 1.0, 0.0, 1.0);
    let frame = harness.render(&camera, &mut batcher, clear);
    assert_colour_near(
        frame.at(16, 16),
        [0, 255, 0, 255],
        "a fully transparent sprite is invisible",
    );
}

#[test]
fn a_uv_window_selects_part_of_a_layer() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    // Left half red, right half blue, so a UV window can be seen to pick one.
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for y in 0..8 {
        for x in 0..8 {
            let index = (y * 8 + x) * 4;
            let colour: [u8; 4] = if x < 4 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            pixels[index..index + 4].copy_from_slice(&colour);
        }
    }
    harness
        .atlas
        .write_layer(&harness.gpu, 0, &pixels)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    // Sample only the right half of the layer.
    batcher.push(
        DrawSprite::new(Vec2::from_ints(16, 24), Vec2::from_ints(16, 16), 0).with_uv(Rect::new(
            Vec2::new(Fx::HALF, Fx::ZERO),
            Vec2::new(Fx::HALF, Fx::ONE),
        )),
    );

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);
    assert_colour_near(
        frame.at(16, 16),
        [0, 0, 255, 255],
        "only the blue half was sampled",
    );
}

#[test]
fn a_flipped_sprite_mirrors_its_artwork() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for y in 0..8 {
        for x in 0..8 {
            let index = (y * 8 + x) * 4;
            let colour: [u8; 4] = if x < 4 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            pixels[index..index + 4].copy_from_slice(&colour);
        }
    }
    harness
        .atlas
        .write_layer(&harness.gpu, 0, &pixels)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut normal = SpriteBatcher::new();
    normal.push(DrawSprite::new(
        Vec2::from_ints(16, 24),
        Vec2::from_ints(16, 16),
        0,
    ));
    let unflipped = harness.render(&camera, &mut normal, Color::BLACK);

    let mut mirrored = SpriteBatcher::new();
    mirrored.push(DrawSprite::new(Vec2::from_ints(16, 24), Vec2::from_ints(16, 16), 0).flipped());
    let flipped = harness.render(&camera, &mut mirrored, Color::BLACK);

    // Sample a point inside the left half of the sprite's footprint.
    let left = (12, 16);
    assert_colour_near(
        unflipped.at(left.0, left.1),
        [255, 0, 0, 255],
        "unflipped left is red",
    );
    assert_colour_near(
        flipped.at(left.0, left.1),
        [0, 0, 255, 255],
        "flipped left is blue",
    );
}

#[test]
fn moving_the_camera_moves_the_scene() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut batcher = SpriteBatcher::new();
    batcher.push(DrawSprite::new(
        Vec2::from_ints(32, 40),
        Vec2::from_ints(16, 16),
        0,
    ));

    let mut camera = Camera2D::new(64, 64);
    camera.position = Vec2::from_ints(32, 32);
    let centred = harness.render(&camera, &mut batcher, Color::BLACK);
    assert_colour_near(
        centred.at(32, 32),
        [255, 255, 255, 255],
        "the sprite starts centred",
    );

    // Pan right by more than the sprite's width; it must leave the centre.
    let mut panned_batch = SpriteBatcher::new();
    panned_batch.push(DrawSprite::new(
        Vec2::from_ints(32, 40),
        Vec2::from_ints(16, 16),
        0,
    ));
    camera.position = Vec2::from_ints(60, 32);
    let panned = harness.render(&camera, &mut panned_batch, Color::BLACK);
    assert_colour_near(
        panned.at(32, 32),
        [0, 0, 0, 255],
        "panning moved the sprite away",
    );
}

#[test]
fn a_large_batch_renders_in_a_single_pass() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(64, 64);
    camera.position = Vec2::from_ints(32, 32);

    // More sprites than the instance buffer's initial capacity, to exercise
    // the growth path.
    let mut batcher = SpriteBatcher::new();
    for index in 0..2000i32 {
        batcher.push(DrawSprite::new(
            Vec2::from_ints(index % 64, (index % 64) + 4),
            Vec2::from_ints(4, 4),
            0,
        ));
    }
    let frame = harness.render(&camera, &mut batcher, Color::BLACK);

    assert!(
        harness.renderer.instance_capacity() >= 2000,
        "the buffer should have grown"
    );
    assert!(
        frame.count_differing_from([0, 0, 0, 255]) > 0,
        "something should have been drawn"
    );
}

#[test]
fn rendering_the_same_scene_twice_produces_identical_pixels() {
    let Some(mut harness) = Harness::new(48, 48, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(48, 48);
    camera.position = Vec2::from_ints(24, 24);

    let build = || {
        let mut batcher = SpriteBatcher::new();
        for index in 0..20i32 {
            batcher.push(
                DrawSprite::new(
                    Vec2::from_ints(index * 2, index * 2 + 8),
                    Vec2::from_ints(6, 6),
                    0,
                )
                .at_z(index % 3),
            );
        }
        batcher
    };

    let first = harness.render(&camera, &mut build(), Color::BLACK);
    let second = harness.render(&camera, &mut build(), Color::BLACK);
    assert_eq!(
        first.pixels, second.pixels,
        "the same scene must render identically"
    );
}

#[test]
fn a_sprite_outside_the_view_is_culled_before_upload() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);
    let visible = camera.visible_bounds();

    let mut batcher = SpriteBatcher::new();
    assert!(batcher.push_visible(
        DrawSprite::new(Vec2::from_ints(16, 24), Vec2::from_ints(8, 8), 0),
        visible
    ));
    assert!(!batcher.push_visible(
        DrawSprite::new(Vec2::from_ints(9000, 9000), Vec2::from_ints(8, 8), 0),
        visible
    ));
    assert_eq!(batcher.len(), 1);

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);
    assert!(frame.count_differing_from([0, 0, 0, 255]) > 0);
}

/// Renders a small scene and writes it out, so a human can look at what the
/// renderer actually produces.
///
/// Ignored by default because it writes a file; run it with
/// `cargo test -p verdant-render-2d -- --ignored` to regenerate.
#[test]
#[ignore = "writes a PNG artifact rather than asserting"]
fn render_to_png() {
    let Some(mut harness) = Harness::new(128, 128, 3) else {
        return;
    };
    for (layer, colour) in [(0, 0x6B_8E_23u32), (1, 0x8F_BC_5A), (2, 0xD2_69_1E)] {
        harness
            .atlas
            .fill_layer(&harness.gpu, layer, Color::from_srgb_hex(colour))
            .expect("writable");
    }

    let mut camera = Camera2D::new(128, 128);
    camera.position = Vec2::from_ints(64, 64);

    let mut batcher = SpriteBatcher::new();
    for row in 0..8i32 {
        for column in 0..8i32 {
            batcher.push(DrawSprite::new(
                Vec2::new(fx(column * 16 + 8), fx(row * 16 + 16)),
                Vec2::from_ints(16, 16),
                ((row + column) % 3) as u32,
            ));
        }
    }

    let frame = harness.render(&camera, &mut batcher, Color::BLACK);
    frame.write_png(std::path::Path::new("target/golden-diffs/scene.png"));
}

// ---------------------------------------------------------------------------
// The present pass
// ---------------------------------------------------------------------------

/// Renders a low-resolution frame and presents it into a larger target.
///
/// The destination is an ordinary [`RenderTarget`] rather than a swapchain, so
/// the upscale is verifiable with no window — which is the only way it can be
/// tested in CI at all.
fn present_into(
    harness: &mut Harness,
    camera: &Camera2D,
    batcher: &mut SpriteBatcher,
    clear: Color,
    destination: (u32, u32),
    border: Color,
) -> Frame {
    harness.renderer.render(
        &harness.gpu,
        &harness.target,
        &harness.bind_group,
        camera,
        batcher.prepare(),
        FrameSettings::clearing(clear),
    );

    let window = RenderTarget::new(&harness.gpu, destination.0, destination.1);
    let presenter = Presenter::new(&harness.gpu, RenderTarget::FORMAT);
    let source = presenter.bind_source(&harness.gpu, &harness.target);
    presenter.present(
        &harness.gpu,
        window.view(),
        &source,
        (harness.target.width(), harness.target.height()),
        destination,
        border,
    );

    let pixels = window
        .read_pixels(&harness.gpu)
        .expect("the presented frame should read back");
    Frame {
        pixels,
        width: destination.0,
        height: destination.1,
    }
}

#[test]
fn presenting_scales_the_frame_by_a_whole_number() {
    // A 32x32 frame into a 128x128 window is exactly 4x, filling it entirely.
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::from_srgb_hex(0xD2_69_1E))
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    // A 16x16 sprite in the top-left quadrant of the low-res frame.
    batcher.push(DrawSprite::new(
        Vec2::from_ints(8, 16),
        Vec2::from_ints(16, 16),
        0,
    ));

    let frame = present_into(
        &mut harness,
        &camera,
        &mut batcher,
        Color::BLACK,
        (128, 128),
        Color::BLACK,
    );

    // The sprite covered the low-res rect x 0..16, y 0..16; at 4x that is
    // 0..64 in both axes of the window.
    let sprite = frame.at(32, 32);
    assert!(
        sprite[0] > 100,
        "the scaled sprite should be here: {sprite:?}"
    );
    assert_eq!(
        frame.at(96, 96),
        [0, 0, 0, 255],
        "the rest of the frame should be the clear colour"
    );
}

#[test]
fn a_source_texel_covers_an_exact_block_of_screen_pixels() {
    // This is what integer scaling buys, and the property a fractional scale
    // would break: one texel must not spill a row into its neighbour.
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let mut batcher = SpriteBatcher::new();
    batcher.push(DrawSprite::new(
        Vec2::from_ints(8, 16),
        Vec2::from_ints(16, 16),
        0,
    ));

    let frame = present_into(
        &mut harness,
        &camera,
        &mut batcher,
        Color::BLACK,
        (128, 128),
        Color::BLACK,
    );

    // The edge sits at low-res x = 16, so window x = 64 exactly. The pixel
    // before it is inside the sprite and the pixel at it is outside.
    assert!(
        frame.at(63, 32)[0] > 100,
        "x=63 should be inside the sprite"
    );
    assert_eq!(
        frame.at(64, 32),
        [0, 0, 0, 255],
        "x=64 should be the first pixel past the sprite"
    );
}

#[test]
fn a_mismatched_window_letterboxes_rather_than_stretching() {
    // 32x32 into 200x100: the largest whole scale that fits is 3x, giving a
    // 96x96 image centred with 52 pixels of border either side.
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    let mut batcher = SpriteBatcher::new();
    let frame = present_into(
        &mut harness,
        &Camera2D::new(32, 32),
        &mut batcher,
        Color::from_srgb_hex(0x40_80_C0),
        (200, 100),
        Color::BLACK,
    );

    assert_eq!(
        frame.at(2, 50),
        [0, 0, 0, 255],
        "the left border should be the letterbox colour"
    );
    assert_eq!(
        frame.at(197, 50),
        [0, 0, 0, 255],
        "the right border should be the letterbox colour"
    );
    assert!(
        frame.at(100, 50)[2] > 100,
        "the centre should hold the presented frame"
    );
    assert_eq!(
        frame.at(100, 1),
        [0, 0, 0, 255],
        "the top border should be the letterbox colour"
    );
}

#[test]
fn presenting_is_reproducible() {
    let Some(mut harness) = Harness::new(32, 32, 1) else {
        return;
    };
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::from_srgb_hex(0x8F_BC_5A))
        .expect("writable");

    let mut camera = Camera2D::new(32, 32);
    camera.position = Vec2::from_ints(16, 16);

    let sprite = DrawSprite::new(Vec2::from_ints(12, 20), Vec2::from_ints(12, 12), 0);
    let mut first_batch = SpriteBatcher::new();
    first_batch.push(sprite);
    let first = present_into(
        &mut harness,
        &camera,
        &mut first_batch,
        Color::BLACK,
        (96, 96),
        Color::BLACK,
    );

    let mut second_batch = SpriteBatcher::new();
    second_batch.push(sprite);
    let second = present_into(
        &mut harness,
        &camera,
        &mut second_batch,
        Color::BLACK,
        (96, 96),
        Color::BLACK,
    );

    assert_eq!(first.pixels, second.pixels);
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

/// Renders a flat white scene, accumulates `lights`, and composites the two.
///
/// The scene is deliberately featureless so that every difference in the
/// result comes from the lighting rather than from the sprites.
fn light_scene(
    harness: &mut Harness,
    camera: &Camera2D,
    lights: &[Light],
    settings: LightSettings,
) -> Frame {
    harness
        .atlas
        .fill_layer(&harness.gpu, 0, Color::WHITE)
        .expect("writable");

    // One sprite covering the whole view, sized from the target so the frame
    // is entirely scene and every difference in it comes from the lighting.
    let width = i32::try_from(harness.target.width()).expect("small");
    let height = i32::try_from(harness.target.height()).expect("small");
    let mut batcher = SpriteBatcher::new();
    batcher.push(DrawSprite::new(
        Vec2::from_ints(width / 2, height),
        Vec2::from_ints(width, height),
        0,
    ));
    harness.renderer.render(
        &harness.gpu,
        &harness.target,
        &harness.bind_group,
        camera,
        batcher.prepare(),
        FrameSettings::default(),
    );

    let light_map = RenderTarget::new(
        &harness.gpu,
        harness.target.width(),
        harness.target.height(),
    );
    let mut light_renderer = LightRenderer::new(&harness.gpu, RenderTarget::FORMAT);
    light_renderer.render(&harness.gpu, &light_map, camera, lights);

    let final_target = RenderTarget::new(
        &harness.gpu,
        harness.target.width(),
        harness.target.height(),
    );
    let compositor = Compositor::new(&harness.gpu, RenderTarget::FORMAT);
    let inputs = compositor.bind(&harness.gpu, &harness.target, &light_map);
    compositor.composite(&harness.gpu, final_target.view(), &inputs, settings);

    let pixels = final_target
        .read_pixels(&harness.gpu)
        .expect("the composited frame should read back");
    Frame {
        pixels,
        width: final_target.width(),
        height: final_target.height(),
    }
}

/// A camera looking at the 64x64 world the light tests draw.
fn light_camera() -> Camera2D {
    let mut camera = Camera2D::new(64, 64);
    camera.position = Vec2::from_ints(32, 32);
    camera
}

#[test]
fn an_unlit_scene_is_dark() {
    // The property the whole system rests on: with no ambient and no lights,
    // a white sprite renders black.
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[],
        LightSettings::ambient_only(Color::BLACK),
    );
    assert_eq!(frame.at(32, 32)[0], 0, "an unlit scene should be black");
}

#[test]
fn ambient_light_lifts_the_whole_scene() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[],
        LightSettings::ambient_only(Color::rgb(0.5, 0.5, 0.5)),
    );
    let centre = frame.at(32, 32);
    let corner = frame.at(2, 2);
    assert!(centre[0] > 100 && centre[0] < 160, "centre {centre:?}");
    assert_eq!(centre, corner, "ambient light should be even");
}

#[test]
fn a_point_light_is_brightest_at_its_centre() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let light = Light::point(Vec2::from_ints(32, 32), fx(24), Color::WHITE);
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[light],
        LightSettings::ambient_only(Color::BLACK),
    );

    let centre = frame.at(32, 32)[0];
    let middle = frame.at(32, 44)[0];
    let edge = frame.at(32, 56)[0];
    assert!(
        centre > middle,
        "centre {centre} should beat middle {middle}"
    );
    assert!(middle > edge, "middle {middle} should beat edge {edge}");
    assert!(centre > 200, "the centre should be near full brightness");
}

#[test]
fn a_point_light_does_not_reach_past_its_radius() {
    // Otherwise a light's cost and its visible effect stop matching, and
    // culling by radius would produce visible popping.
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let light = Light::point(Vec2::from_ints(16, 16), fx(12), Color::WHITE);
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[light],
        LightSettings::ambient_only(Color::BLACK),
    );
    assert_eq!(frame.at(56, 56)[0], 0, "far corner should be untouched");
}

#[test]
fn a_coloured_light_tints_what_it_lights() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    // A warm lantern: strong red, weak blue.
    let light = Light::point(Vec2::from_ints(32, 32), fx(24), Color::rgb(1.0, 0.6, 0.2));
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[light],
        LightSettings::ambient_only(Color::BLACK),
    );
    let lit = frame.at(32, 32);
    assert!(lit[0] > lit[1] && lit[1] > lit[2], "warm tint: {lit:?}");
}

#[test]
fn two_lights_add_where_they_overlap() {
    // Additive blending is what makes light behave like light rather than
    // like a decal.
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let camera = light_camera();
    let dim = Color::rgb(0.4, 0.4, 0.4);

    let one = light_scene(
        &mut harness,
        &camera,
        &[Light::point(Vec2::from_ints(32, 32), fx(24), dim).with_falloff(1.0)],
        LightSettings::ambient_only(Color::BLACK),
    )
    .at(32, 32)[0];

    let both = light_scene(
        &mut harness,
        &camera,
        &[
            Light::point(Vec2::from_ints(32, 32), fx(24), dim).with_falloff(1.0),
            Light::point(Vec2::from_ints(32, 32), fx(24), dim).with_falloff(1.0),
        ],
        LightSettings::ambient_only(Color::BLACK),
    )
    .at(32, 32)[0];

    assert!(both > one, "two lights ({both}) should beat one ({one})");
}

#[test]
fn illumination_saturates_rather_than_wrapping() {
    // A stack of overlapping lanterns must clamp, not overflow into a dark
    // band, which is what an unclamped multiply would do.
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let lights: Vec<Light> = (0..8)
        .map(|_| Light::point(Vec2::from_ints(32, 32), fx(30), Color::WHITE))
        .collect();
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &lights,
        LightSettings::ambient_only(Color::BLACK),
    );
    assert!(
        frame.at(32, 32)[0] > 240,
        "should be saturated, not wrapped"
    );
}

#[test]
fn a_spot_light_only_lights_its_cone() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    // Pointing straight down (+Y in world space), a narrow cone.
    let light = Light::spot(
        Vec2::from_ints(32, 20),
        fx(36),
        Color::WHITE,
        Fx::PI / fx(2),
        Fx::PI / fx(8),
    );
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &[light],
        LightSettings::ambient_only(Color::BLACK),
    );

    // Sampled near the light rather than at the far end of its reach: the
    // default falloff is quadratic, so a point at half the radius is already
    // down to a fifth of full brightness and says less about the cone than
    // about the falloff curve.
    let inside = frame.at(32, 30)[0];
    let far_inside = frame.at(32, 44)[0];
    let outside = frame.at(4, 20)[0];

    assert!(inside > 100, "inside the cone should be well lit: {inside}");
    assert_eq!(outside, 0, "outside the cone should be dark");
    assert!(
        far_inside > 0 && far_inside < inside,
        "the cone should still fall off with distance: {far_inside} vs {inside}"
    );
}

#[test]
fn lighting_is_reproducible() {
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let camera = light_camera();
    let lights = [
        Light::point(Vec2::from_ints(20, 20), fx(18), Color::rgb(1.0, 0.7, 0.3)),
        Light::point(Vec2::from_ints(44, 40), fx(14), Color::rgb(0.4, 0.6, 1.0)),
    ];
    let settings = LightSettings {
        ambient: Color::rgb(0.2, 0.2, 0.3),
        light_scale: 1.0,
        maximum: 1.0,
    };
    let first = light_scene(&mut harness, &camera, &lights, settings);
    let second = light_scene(&mut harness, &camera, &lights, settings);
    assert_eq!(first.pixels, second.pixels);
}

#[test]
fn more_lights_than_the_buffer_holds_still_render() {
    // The instance buffer has to grow; dropping the overflow would silently
    // darken a busy scene.
    let Some(mut harness) = Harness::new(64, 64, 1) else {
        return;
    };
    let lights: Vec<Light> = (0..200)
        .map(|index| {
            Light::point(
                Vec2::from_ints(32, 32),
                fx(20) + fx(index % 4),
                Color::rgb(0.05, 0.05, 0.05),
            )
        })
        .collect();
    let frame = light_scene(
        &mut harness,
        &light_camera(),
        &lights,
        LightSettings::ambient_only(Color::BLACK),
    );
    assert!(frame.at(32, 32)[0] > 100, "200 lights should be visible");
}

#[test]
#[ignore = "writes a PNG artifact rather than asserting"]
fn render_lit_scene_to_png() {
    let Some(mut harness) = Harness::new(128, 128, 1) else {
        return;
    };
    let mut camera = Camera2D::new(128, 128);
    camera.position = Vec2::from_ints(64, 64);
    let frame = light_scene(
        &mut harness,
        &camera,
        &[
            Light::point(Vec2::from_ints(40, 50), fx(40), Color::rgb(1.0, 0.75, 0.4)),
            Light::point(Vec2::from_ints(90, 80), fx(30), Color::rgb(0.4, 0.6, 1.0)),
        ],
        LightSettings {
            ambient: Color::rgb(0.15, 0.16, 0.28),
            light_scale: 1.0,
            maximum: 1.0,
        },
    );
    frame.write_png(std::path::Path::new("target/golden-diffs/lit-scene.png"));
}
