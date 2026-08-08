//! A 5×7 bitmap font.
//!
//! Every other sprite in this crate is generated from a seed, but a font is
//! the one kind of pixel art that must not be: letterforms are recognised, not
//! sampled, and a procedurally "varied" `e` is simply a broken `e`. So the
//! glyphs are authored here as bit patterns and shipped as data.
//!
//! Glyphs are drawn into a [`Canvas`] like everything else, which means text
//! goes through the same palette discipline as the rest of the art and lands
//! in the same atlas.
//!
//! ```
//! use verdant_procgen_art::font;
//!
//! let label = font::render_text("Hello", verdant_procgen_art::PaletteIndex(1));
//! assert_eq!(label.width(), font::text_width("Hello"));
//! ```

use crate::canvas::{Canvas, PaletteIndex};

/// Width of a glyph's cell, in pixels.
pub const GLYPH_WIDTH: u32 = 5;

/// Height of a glyph's cell, in pixels.
pub const GLYPH_HEIGHT: u32 = 7;

/// Horizontal distance from one glyph's left edge to the next, in pixels.
///
/// One more than [`GLYPH_WIDTH`], which is the inter-letter gap.
pub const GLYPH_ADVANCE: u32 = GLYPH_WIDTH + 1;

/// Vertical distance between consecutive baselines, in pixels.
pub const LINE_HEIGHT: u32 = GLYPH_HEIGHT + 2;

/// The number of rows in a glyph, as a `usize` for indexing.
const ROWS: usize = GLYPH_HEIGHT as usize;

/// The glyph drawn in place of any character the font does not cover.
const FALLBACK: [u8; ROWS] = [
    0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100,
];

/// The printable ASCII range this font covers, from `' '` to `'~'`.
///
/// Rows run top to bottom; within a row, bit 4 is the leftmost pixel.
#[rustfmt::skip]
const GLYPHS: [[u8; ROWS]; 95] = [
    // ' '
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '!'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    // '"'
    [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '#'
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
    // '$'
    [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
    // '%'
    [0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011],
    // '&'
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
    // '\''
    [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '('
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    // ')'
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    // '*'
    [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000],
    // '+'
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
    // ','
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000],
    // '-'
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    // '.'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
    // '/'
    [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
    // '0'
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    // '1'
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // '2'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // '3'
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    // '4'
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // '5'
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // '6'
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // '7'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // '8'
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    // '9'
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    // ':'
    [0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000],
    // ';'
    [0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00100, 0b01000],
    // '<'
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
    // '='
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    // '>'
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
    // '?'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
    // '@'
    [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
    // 'A'
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // 'B'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    // 'C'
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    // 'D'
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    // 'E'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    // 'F'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'G'
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    // 'H'
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // 'I'
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'J'
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'K'
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    // 'L'
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    // 'M'
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    // 'N'
    [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    // 'O'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'P'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'Q'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    // 'R'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    // 'S'
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    // 'T'
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // 'U'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'V'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'W'
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    // 'X'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    // 'Y'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    // 'Z'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    // '['
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    // '\\'
    [0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001],
    // ']'
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    // '^'
    [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
    // '_'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    // '`'
    [0b01000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // 'a'
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
    // 'b'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
    // 'c'
    [0b00000, 0b00000, 0b01111, 0b10000, 0b10000, 0b10000, 0b01111],
    // 'd'
    [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
    // 'e'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
    // 'f'
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
    // 'g'
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'h'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    // 'i'
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'j'
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'k'
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
    // 'l'
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'm'
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
    // 'n'
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    // 'o'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'p'
    [0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000],
    // 'q'
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001],
    // 'r'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
    // 's'
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
    // 't'
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
    // 'u'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
    // 'v'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'w'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    // 'x'
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    // 'y'
    [0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'z'
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
    // '{'
    [0b00110, 0b01000, 0b01000, 0b11000, 0b01000, 0b01000, 0b00110],
    // '|'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // '}'
    [0b11000, 0b00100, 0b00100, 0b00110, 0b00100, 0b00100, 0b11000],
    // '~'
    [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000],
];

/// The bit rows making up `ch`, top row first.
///
/// Characters outside the printable ASCII range render as `?` rather than
/// disappearing, so a missing glyph shows up on screen instead of silently
/// shortening a line.
#[must_use]
pub fn glyph(ch: char) -> [u8; ROWS] {
    let code = ch as u32;
    if (0x20..0x7F).contains(&code) {
        let index = usize::try_from(code - 0x20).unwrap_or(0);
        GLYPHS[index]
    } else {
        FALLBACK
    }
}

/// The width `text` occupies, in pixels.
///
/// The trailing inter-letter gap is not counted, so a string's width is the
/// distance from its first lit column to one past its last.
#[must_use]
pub fn text_width(text: &str) -> u32 {
    let count = u32::try_from(text.chars().count()).unwrap_or(u32::MAX);
    match count {
        0 => 0,
        n => n * GLYPH_ADVANCE - (GLYPH_ADVANCE - GLYPH_WIDTH),
    }
}

/// Draws `text` into `canvas` with its top-left corner at `(x, y)`.
///
/// Pixels outside the canvas are dropped, matching [`Canvas::set`].
pub fn draw_text(canvas: &mut Canvas, x: i32, y: i32, text: &str, index: PaletteIndex) {
    let advance = i32::try_from(GLYPH_ADVANCE).unwrap_or(6);
    for (position, ch) in text.chars().enumerate() {
        let origin = x + i32::try_from(position).unwrap_or(0) * advance;
        let rows = glyph(ch);
        for (row, bits) in rows.iter().enumerate() {
            for column in 0..GLYPH_WIDTH {
                // Bit 4 is the leftmost pixel.
                let mask = 1u8 << (GLYPH_WIDTH - 1 - column);
                if bits & mask != 0 {
                    canvas.set(
                        origin + i32::try_from(column).unwrap_or(0),
                        y + i32::try_from(row).unwrap_or(0),
                        index,
                    );
                }
            }
        }
    }
}

/// Renders `text` onto a canvas sized exactly to fit it.
///
/// An empty string yields a 1×[`GLYPH_HEIGHT`] canvas rather than a zero-width
/// one, because a zero-sized canvas cannot be uploaded as a texture.
#[must_use]
pub fn render_text(text: &str, index: PaletteIndex) -> Canvas {
    let mut canvas = Canvas::new(text_width(text).max(1), GLYPH_HEIGHT);
    draw_text(&mut canvas, 0, 0, text, index);
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ink colour used throughout these tests.
    const INK: PaletteIndex = PaletteIndex(1);

    #[test]
    fn the_table_covers_every_printable_ascii_character() {
        assert_eq!(GLYPHS.len(), 0x7F - 0x20);
    }

    #[test]
    fn no_glyph_uses_a_column_outside_its_cell() {
        for (index, rows) in GLYPHS.iter().enumerate() {
            for (row, bits) in rows.iter().enumerate() {
                assert_eq!(
                    bits & !0b11111,
                    0,
                    "glyph {index} row {row} sets a bit outside the 5-pixel cell"
                );
            }
        }
    }

    #[test]
    fn every_visible_character_draws_something() {
        // A blank glyph in the middle of a string would read as a missing
        // letter, which is the defect this catches.
        for code in 0x21..0x7Fu32 {
            let ch = char::from_u32(code).expect("ASCII is valid UTF-8");
            let canvas = render_text(&ch.to_string(), INK);
            assert!(canvas.coverage() > 0, "{ch:?} renders as a blank cell");
        }
    }

    #[test]
    fn space_draws_nothing() {
        assert_eq!(render_text(" ", INK).coverage(), 0);
    }

    #[test]
    fn unknown_characters_fall_back_rather_than_vanishing() {
        assert_eq!(glyph('\u{2603}'), FALLBACK);
        assert!(render_text("\u{2603}", INK).coverage() > 0);
    }

    #[test]
    fn width_accounts_for_gaps_but_not_a_trailing_one() {
        assert_eq!(text_width(""), 0);
        assert_eq!(text_width("A"), GLYPH_WIDTH);
        assert_eq!(text_width("AB"), GLYPH_WIDTH * 2 + 1);
        assert_eq!(text_width("ABC"), GLYPH_WIDTH * 3 + 2);
    }

    #[test]
    fn a_rendered_string_is_exactly_as_wide_as_measured() {
        let canvas = render_text("Verdant Hollow", INK);
        assert_eq!(canvas.width(), text_width("Verdant Hollow"));
        assert_eq!(canvas.height(), GLYPH_HEIGHT);
    }

    #[test]
    fn an_empty_string_still_yields_a_usable_canvas() {
        let canvas = render_text("", INK);
        assert_eq!(canvas.width(), 1);
        assert_eq!(canvas.coverage(), 0);
    }

    #[test]
    fn letters_do_not_run_into_each_other() {
        // Two solid-edged letters side by side must leave a blank column
        // between them, or text turns into a single connected smear.
        let canvas = render_text("HH", INK);
        let gap = i32::try_from(GLYPH_WIDTH).expect("small");
        for y in 0..i32::try_from(GLYPH_HEIGHT).expect("small") {
            assert_eq!(
                canvas.get(gap, y),
                PaletteIndex::TRANSPARENT,
                "column {gap} should separate the two letters"
            );
        }
    }

    #[test]
    fn drawing_off_the_edge_is_clipped_rather_than_panicking() {
        let mut canvas = Canvas::new(8, 8);
        draw_text(&mut canvas, -20, -20, "clipped", INK);
        draw_text(&mut canvas, 100, 100, "clipped", INK);
        assert_eq!(canvas.coverage(), 0);
    }

    #[test]
    fn glyphs_are_positioned_at_their_advance() {
        // The second letter of a two-letter string must match the same letter
        // rendered alone, shifted by exactly one advance.
        let pair = render_text("AB", INK);
        let single = render_text("B", INK);
        let advance = i32::try_from(GLYPH_ADVANCE).expect("small");
        for y in 0..i32::try_from(GLYPH_HEIGHT).expect("small") {
            for x in 0..i32::try_from(GLYPH_WIDTH).expect("small") {
                assert_eq!(pair.get(x + advance, y), single.get(x, y));
            }
        }
    }

    #[test]
    fn digits_are_all_distinct() {
        // A copy-paste slip in the table would give two digits the same shape,
        // which is invisible in a screenshot but wrong in every readout.
        let digits: Vec<[u8; ROWS]> = ('0'..='9').map(glyph).collect();
        for (i, a) in digits.iter().enumerate() {
            for (j, b) in digits.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "digits {i} and {j} share a glyph");
            }
        }
    }

    #[test]
    fn letters_are_all_distinct() {
        let letters: Vec<[u8; ROWS]> =
            ('A'..='Z').chain('a'..='z').map(glyph).collect();
        for (i, a) in letters.iter().enumerate() {
            for (j, b) in letters.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "letters at {i} and {j} share a glyph");
            }
        }
    }
}
