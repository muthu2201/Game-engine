//! Renders a contact sheet of every generator, for visual inspection.
//!
//! Grid arithmetic here casts freely between index types; this is a developer
//! tool laying out a fixed, small sheet, not shipped code.
#![allow(clippy::cast_possible_truncation)]

//!
//! Run with `cargo run -p verdant-procgen-art --example contact_sheet`.
//! Automated tests can confirm a sprite is symmetric or non-blank; only a
//! person can confirm it looks like a character.

use verdant_core_math::Rng;
use verdant_procgen_art::{
    generate_character, generate_crop, generate_item_icon, generate_terrain_tile, generate_tree,
    Canvas, CharacterStyle, Palette, Rgba,
};

/// Scale each sprite up so the pixels are visible in an image viewer.
const ZOOM: u32 = 4;

fn main() {
    let cell = 40u32;
    let columns = 16u32;
    let rows = 10u32;
    let mut sheet = vec![24u8; (cell * columns * ZOOM * cell * rows * ZOOM * 4) as usize];
    let sheet_width = cell * columns * ZOOM;

    let mut place = |canvas: &Canvas, palette: &Palette, column: u32, row: u32| {
        let pixels = canvas.to_rgba(palette);
        for y in 0..canvas.height() {
            for x in 0..canvas.width() {
                let source = ((y * canvas.width() + x) * 4) as usize;
                if pixels[source + 3] == 0 {
                    continue;
                }
                for zy in 0..ZOOM {
                    for zx in 0..ZOOM {
                        let px = (column * cell + x) * ZOOM + zx;
                        let py = (row * cell + y) * ZOOM + zy;
                        let target = ((py * sheet_width + px) * 4) as usize;
                        if target + 4 <= sheet.len() {
                            sheet[target..target + 4].copy_from_slice(&pixels[source..source + 4]);
                        }
                    }
                }
            }
        }
    };

    // Row 0-3: four randomised villagers, all four directions and walk frames.
    let mut rng = Rng::new(20_260_808);
    for villager in 0..4u32 {
        let style = CharacterStyle::random(&mut rng);
        let sprite = generate_character(u64::from(villager) + 1, 16, 24, style);
        for (index, frame) in sprite.frames.iter().enumerate() {
            place(frame, &sprite.palette, index as u32, villager);
        }
    }

    // Row 4: terrain tiles.
    for (column, colour) in [
        0x6B_8E_23u32,
        0x8F_BC_5A,
        0x4A_6B_2A,
        0x9C_8A_5A,
        0x5A_43_2E,
        0x7A_7A_8C,
    ]
    .iter()
    .enumerate()
    {
        let tile = generate_terrain_tile(column as u64, 16, Rgba::hex(*colour), 70);
        place(&tile.frames[0], &tile.palette, column as u32, 4);
    }

    // Rows 5-6: crop growth stages.
    for (row, (leaf, fruit)) in [(0x4A_8C_3Au32, 0xD9_4A_3Au32), (0x5A_9C_4A, 0xE8_C4_3D)]
        .iter()
        .enumerate()
    {
        let crop = generate_crop(row as u64 + 10, 16, 6, Rgba::hex(*leaf), Rgba::hex(*fruit));
        for (stage, frame) in crop.frames.iter().enumerate() {
            place(frame, &crop.palette, stage as u32, 5 + row as u32);
        }
    }

    // Row 7-8: trees.
    for column in 0..5u32 {
        let tree = generate_tree(u64::from(column) + 100, 24, 32, Rgba::hex(0x3A_6B_2A));
        place(&tree.frames[0], &tree.palette, column, 7);
    }

    // Row 9: item icons.
    for column in 0..10u32 {
        let icon = generate_item_icon(u64::from(column) + 200, 16, Rgba::hex(0xC9_A0_3D));
        place(&icon.frames[0], &icon.palette, column, 9);
    }

    let path = std::path::Path::new("target/contact-sheet.png");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("target is writable");
    }
    let file = std::fs::File::create(path).expect("the sheet is writable");
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        sheet_width,
        cell * rows * ZOOM,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("valid header")
        .write_image_data(&sheet)
        .expect("matching data");

    println!("wrote {}", path.display());
}
