//! Renders frames of Verdant Hollow to PNG files, with no window.
//!
//! This is how the game's look is reviewed: the tests assert that a frame has
//! the right *structure*, but only an image shows whether it is any good. CI
//! runs this against the software Vulkan driver and keeps the PNGs as build
//! artifacts, so a visual regression is visible in the pull request rather
//! than only on someone's desktop.
//!
//! ```text
//! cargo run -p verdant-hollow --example screenshot -- target/shots
//! ```

use verdant_hollow::render::{Art, GameRenderer, Scene, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use verdant_hollow::sim::Game;
use verdant_render_2d::{GpuContext, RenderTarget};

fn main() {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/screenshots".to_owned());
    std::fs::create_dir_all(&directory).expect("the output directory is writable");

    let gpu = match GpuContext::headless() {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("no graphics device: {error}");
            std::process::exit(1);
        }
    };
    eprintln!("rendering on {}", gpu.describe());

    let mut game = Game::new(20_260_808);
    let seeds: Vec<u64> = game
        .villagers
        .iter()
        .map(|villager| villager.appearance_seed)
        .collect();
    let art = Art::generate(game.seed, &seeds);
    let mut renderer =
        GameRenderer::new(&gpu, &art, RenderTarget::FORMAT).expect("the atlas uploads");
    let window = RenderTarget::new(&gpu, INTERNAL_WIDTH * 2, INTERNAL_HEIGHT * 2);

    // Morning on the farm, with a few plots worked so the farming art shows.
    let plot = game.layout.farm_area.0 + verdant_core_math::IVec2::new(2, 2);
    for offset in 0..4 {
        let cell = plot + verdant_core_math::IVec2::new(offset, 0);
        game.farm.till(cell);
        game.farm.water(cell);
    }

    let mut scene = Scene::new();
    scene.snap_to(&game);

    // Draw the phone layout too, so the touch controls get the same visual
    // review as everything else rather than being verified only by tests.
    scene.show_touch_controls = true;

    for (name, minute) in [
        ("morning", 8 * 60),
        ("afternoon", 14 * 60),
        ("dusk", 19 * 60),
        ("night", 23 * 60),
    ] {
        game.calendar.minute = minute;
        renderer.draw(
            &mut scene,
            &game,
            &art,
            window.view(),
            (window.width(), window.height()),
        );
        let pixels = renderer.read_frame().expect("the frame reads back");
        let path = format!("{directory}/{name}.png");
        write_png(&path, &pixels, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        eprintln!("wrote {path}");
    }
}

/// Writes RGBA bytes out as a PNG.
fn write_png(path: &str, pixels: &[u8], width: u32, height: u32) {
    let file = std::fs::File::create(path).expect("the file is writable");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("the header is valid")
        .write_image_data(pixels)
        .expect("the data matches the header");
}
