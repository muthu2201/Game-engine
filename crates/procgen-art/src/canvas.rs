//! Palettes and the indexed canvas every generator draws onto.

use serde::{Deserialize, Serialize};

/// An 8-bit RGBA colour, as authored.
///
/// Stored sRGB here because this is the authoring side; conversion to linear
/// happens when the pixels reach the renderer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct Rgba {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
    /// Alpha; `0` is fully transparent.
    pub a: u8,
}

impl Rgba {
    /// Fully transparent.
    pub const TRANSPARENT: Rgba = Rgba {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };

    /// An opaque colour from a hex literal such as `0x8F_BC_5A`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn hex(value: u32) -> Rgba {
        Rgba {
            r: ((value >> 16) & 0xFF) as u8,
            g: ((value >> 8) & 0xFF) as u8,
            b: (value & 0xFF) as u8,
            a: 255,
        }
    }

    /// The colour as RGBA bytes.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Blends toward `other` by `amount` in 0..=255.
    ///
    /// Integer arithmetic so shading is reproducible bit-for-bit, which matters
    /// because a sprite's bytes are content-addressed by the asset database.
    #[must_use]
    pub fn blend(self, other: Rgba, amount: u8) -> Rgba {
        let mix = |a: u8, b: u8| {
            let a = u32::from(a);
            let b = u32::from(b);
            let t = u32::from(amount);
            #[allow(clippy::cast_possible_truncation)]
            {
                ((a * (255 - t) + b * t) / 255) as u8
            }
        };
        Rgba {
            r: mix(self.r, other.r),
            g: mix(self.g, other.g),
            b: mix(self.b, other.b),
            a: mix(self.a, other.a),
        }
    }

    /// Scales the RGB channels by `factor / 255`, leaving alpha alone.
    #[must_use]
    pub fn shade(self, factor: u8) -> Rgba {
        let scale = |channel: u8| {
            #[allow(clippy::cast_possible_truncation)]
            {
                ((u32::from(channel) * u32::from(factor)) / 255) as u8
            }
        };
        Rgba {
            r: scale(self.r),
            g: scale(self.g),
            b: scale(self.b),
            a: self.a,
        }
    }

    /// Moves the colour toward white by `amount / 255`.
    #[must_use]
    pub fn tint(self, amount: u8) -> Rgba {
        self.blend(
            Rgba {
                r: 255,
                g: 255,
                b: 255,
                a: self.a,
            },
            amount,
        )
    }

    /// True when the colour is fully transparent.
    #[must_use]
    pub const fn is_transparent(self) -> bool {
        self.a == 0
    }
}

/// Index into a [`Palette`].
///
/// Index `0` is always transparent, so an untouched canvas is empty rather than
/// a field of colour zero.
#[derive(
    Clone, Copy, PartialEq, Eq, Debug, Default, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PaletteIndex(pub u8);

impl PaletteIndex {
    /// The transparent entry.
    pub const TRANSPARENT: PaletteIndex = PaletteIndex(0);
}

/// A fixed set of colours a sprite may use.
///
/// # Why enforce a palette
///
/// Restricting output to a small palette is what makes generated pixel art read
/// as *art* rather than as noise: coherent colour is most of what the eye reads
/// as intentional. It is also the property general image models struggle with,
/// and the reason purpose-built pixel-art pipelines quantise so aggressively.
/// Here the constraint is structural — a canvas stores palette *indices*, so
/// producing an off-palette colour is not expressible.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Palette {
    /// Entry `0` is always transparent.
    colors: Vec<Rgba>,
    /// A name, recorded in provenance metadata.
    pub name: String,
}

impl Palette {
    /// Creates a palette from a list of opaque colours.
    ///
    /// The transparent entry is prepended automatically.
    ///
    /// # Panics
    ///
    /// Panics if more than 255 colours are supplied, since indices are `u8`.
    #[must_use]
    pub fn new(name: impl Into<String>, colors: &[Rgba]) -> Palette {
        assert!(
            colors.len() <= 255,
            "Palette: at most 255 colours are addressable"
        );
        let mut all = Vec::with_capacity(colors.len() + 1);
        all.push(Rgba::TRANSPARENT);
        all.extend_from_slice(colors);
        Palette {
            colors: all,
            name: name.into(),
        }
    }

    /// The colour at an index, or transparent when out of range.
    ///
    /// Clamping rather than panicking keeps a generator that computes an index
    /// arithmetically from producing a crash on an edge case.
    #[must_use]
    pub fn color(&self, index: PaletteIndex) -> Rgba {
        self.colors
            .get(index.0 as usize)
            .copied()
            .unwrap_or(Rgba::TRANSPARENT)
    }

    /// Number of entries, including the transparent one.
    #[must_use]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// True when only the transparent entry is present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.len() <= 1
    }

    /// A ramp of `steps` indices, from a base colour to progressively darker
    /// shades, appended to the palette.
    ///
    /// Ramps are how the generators shade: a sprite picks a base index and
    /// walks the ramp for its lit and shadowed faces, which keeps every sprite
    /// shaded consistently.
    pub fn add_ramp(&mut self, base: Rgba, steps: u8) -> Vec<PaletteIndex> {
        let mut indices = Vec::with_capacity(steps as usize);
        for step in 0..steps {
            // Darken toward 45% at the far end of the ramp; going further
            // makes shadows read as holes rather than form.
            let factor = 255 - (step as u32 * 140 / u32::from(steps.max(1)));
            #[allow(clippy::cast_possible_truncation)]
            let shaded = base.shade(factor as u8);
            let index = u8::try_from(self.colors.len()).unwrap_or(255);
            self.colors.push(shaded);
            indices.push(PaletteIndex(index));
        }
        indices
    }

    /// Appends a colour and returns its index.
    ///
    /// # Panics
    ///
    /// Panics when the palette is already full.
    pub fn push(&mut self, color: Rgba) -> PaletteIndex {
        assert!(self.colors.len() < 256, "Palette: no free indices remain");
        let index = u8::try_from(self.colors.len()).unwrap_or(255);
        self.colors.push(color);
        PaletteIndex(index)
    }
}

/// An indexed pixel buffer.
///
/// Every generator draws onto one of these. Because cells hold palette indices
/// rather than colours, the palette constraint cannot be violated, and the
/// whole sprite can be recoloured — for seasons, or a dyed shirt — by swapping
/// the palette without touching the artwork.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<PaletteIndex>,
}

impl Canvas {
    /// Creates a fully transparent canvas.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Canvas {
        assert!(
            width > 0 && height > 0,
            "Canvas: dimensions must be non-zero"
        );
        Canvas {
            width,
            height,
            pixels: vec![PaletteIndex::TRANSPARENT; (width as usize) * (height as usize)],
        }
    }

    /// Canvas width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Canvas height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// True when `(x, y)` lies inside the canvas.
    #[must_use]
    pub const fn contains(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }

    /// The index at `(x, y)`, or transparent outside the canvas.
    #[must_use]
    pub fn get(&self, x: i32, y: i32) -> PaletteIndex {
        if !self.contains(x, y) {
            return PaletteIndex::TRANSPARENT;
        }
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Writes an index, ignoring writes outside the canvas.
    ///
    /// Silent clipping lets a generator draw a shape that overhangs the edge
    /// without each one carrying its own bounds check.
    pub fn set(&mut self, x: i32, y: i32, index: PaletteIndex) {
        if self.contains(x, y) {
            let offset = (y as usize) * (self.width as usize) + (x as usize);
            self.pixels[offset] = index;
        }
    }

    /// Writes an index only where the canvas is still transparent.
    ///
    /// Used to lay a base shape down without erasing detail drawn earlier.
    pub fn set_if_empty(&mut self, x: i32, y: i32, index: PaletteIndex) {
        if self.get(x, y) == PaletteIndex::TRANSPARENT {
            self.set(x, y, index);
        }
    }

    /// Fills a rectangle.
    pub fn fill_rect(&mut self, x: i32, y: i32, width: i32, height: i32, index: PaletteIndex) {
        for row in y..y + height {
            for column in x..x + width {
                self.set(column, row, index);
            }
        }
    }

    /// Draws a filled ellipse inscribed in a rectangle.
    ///
    /// Ellipses rather than circles because most organic shapes in a sprite —
    /// a tree canopy, a character's head, a fruit — are wider than they are
    /// tall or the reverse.
    pub fn fill_ellipse(&mut self, cx: i32, cy: i32, rx: i32, ry: i32, index: PaletteIndex) {
        if rx <= 0 || ry <= 0 {
            return;
        }
        for y in -ry..=ry {
            for x in -rx..=rx {
                // Compare in a common denominator to stay in integers.
                let inside = (x * x) * (ry * ry) + (y * y) * (rx * rx) <= (rx * rx) * (ry * ry);
                if inside {
                    self.set(cx + x, cy + y, index);
                }
            }
        }
    }

    /// Draws a line with Bresenham's algorithm.
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, index: PaletteIndex) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let step_x = if x0 < x1 { 1 } else { -1 };
        let step_y = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        let (mut x, mut y) = (x0, y0);

        loop {
            self.set(x, y, index);
            if x == x1 && y == y1 {
                break;
            }
            let doubled = error * 2;
            if doubled >= dy {
                error += dy;
                x += step_x;
            }
            if doubled <= dx {
                error += dx;
                y += step_y;
            }
        }
    }

    /// Mirrors the left half onto the right.
    ///
    /// Bilateral symmetry is most of what makes a generated character read as a
    /// character: generate half, mirror it, then break the symmetry with a few
    /// asymmetric details.
    pub fn mirror_horizontally(&mut self) {
        let half = self.width / 2;
        for y in 0..self.height as i32 {
            for x in 0..half as i32 {
                let source = self.get(x, y);
                let target_x = (self.width as i32) - 1 - x;
                self.set(target_x, y, source);
            }
        }
    }

    /// Replaces every occurrence of one index with another.
    pub fn replace(&mut self, from: PaletteIndex, to: PaletteIndex) {
        for pixel in &mut self.pixels {
            if *pixel == from {
                *pixel = to;
            }
        }
    }

    /// Outlines every non-transparent region with `index`.
    ///
    /// A dark outline is what separates a sprite from the background at small
    /// sizes; without one, characters visually dissolve into the terrain.
    pub fn outline(&mut self, index: PaletteIndex) {
        let original = self.clone();
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                if original.get(x, y) != PaletteIndex::TRANSPARENT {
                    continue;
                }
                // Orthogonal neighbours only: including diagonals thickens the
                // outline at corners and reads as a blob.
                let touches_shape = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                    .iter()
                    .any(|(dx, dy)| original.get(x + dx, y + dy) != PaletteIndex::TRANSPARENT);
                if touches_shape {
                    self.set(x, y, index);
                }
            }
        }
    }

    /// Copies another canvas onto this one at an offset, skipping transparency.
    pub fn blit(&mut self, source: &Canvas, offset_x: i32, offset_y: i32) {
        for y in 0..source.height as i32 {
            for x in 0..source.width as i32 {
                let index = source.get(x, y);
                if index != PaletteIndex::TRANSPARENT {
                    self.set(offset_x + x, offset_y + y, index);
                }
            }
        }
    }

    /// Shifts the whole canvas, leaving vacated pixels transparent.
    ///
    /// Used to build walk-cycle frames by bobbing a body up and down.
    #[must_use]
    pub fn translated(&self, dx: i32, dy: i32) -> Canvas {
        let mut shifted = Canvas::new(self.width, self.height);
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                shifted.set(x + dx, y + dy, self.get(x, y));
            }
        }
        shifted
    }

    /// Resolves the canvas into RGBA bytes, row-major.
    ///
    /// This is the form the renderer uploads.
    #[must_use]
    pub fn to_rgba(&self, palette: &Palette) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.pixels.len() * 4);
        for index in &self.pixels {
            bytes.extend_from_slice(&palette.color(*index).to_bytes());
        }
        bytes
    }

    /// Number of pixels that are not transparent.
    #[must_use]
    pub fn coverage(&self) -> usize {
        self.pixels
            .iter()
            .filter(|index| **index != PaletteIndex::TRANSPARENT)
            .count()
    }

    /// The raw index buffer.
    #[must_use]
    pub fn pixels(&self) -> &[PaletteIndex] {
        &self.pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette() -> Palette {
        Palette::new("test", &[Rgba::hex(0xFF_00_00), Rgba::hex(0x00_FF_00)])
    }

    #[test]
    fn index_zero_is_always_transparent() {
        let palette = palette();
        assert!(palette.color(PaletteIndex::TRANSPARENT).is_transparent());
        assert_eq!(palette.color(PaletteIndex(1)), Rgba::hex(0xFF_00_00));
    }

    #[test]
    fn an_out_of_range_index_reads_as_transparent() {
        assert!(palette().color(PaletteIndex(200)).is_transparent());
    }

    #[test]
    fn hex_colours_unpack_correctly() {
        let colour = Rgba::hex(0x12_34_56);
        assert_eq!(
            (colour.r, colour.g, colour.b, colour.a),
            (0x12, 0x34, 0x56, 255)
        );
    }

    #[test]
    fn blending_hits_both_endpoints() {
        let red = Rgba::hex(0xFF_00_00);
        let blue = Rgba::hex(0x00_00_FF);
        assert_eq!(red.blend(blue, 0), red);
        assert_eq!(red.blend(blue, 255), blue);
        let midpoint = red.blend(blue, 128);
        assert!(midpoint.r > 100 && midpoint.b > 100);
    }

    #[test]
    fn shading_darkens_without_touching_alpha() {
        let colour = Rgba {
            r: 200,
            g: 200,
            b: 200,
            a: 128,
        };
        let shaded = colour.shade(128);
        assert!(shaded.r < colour.r);
        assert_eq!(shaded.a, 128);
    }

    #[test]
    fn a_ramp_grows_progressively_darker() {
        let mut palette = palette();
        let ramp = palette.add_ramp(Rgba::hex(0xFF_FF_FF), 4);
        assert_eq!(ramp.len(), 4);

        let brightness: Vec<u8> = ramp.iter().map(|index| palette.color(*index).r).collect();
        for pair in brightness.windows(2) {
            assert!(pair[1] < pair[0], "ramp brightness was {brightness:?}");
        }
    }

    #[test]
    fn a_new_canvas_is_empty() {
        let canvas = Canvas::new(8, 8);
        assert_eq!(canvas.coverage(), 0);
        assert_eq!(canvas.get(0, 0), PaletteIndex::TRANSPARENT);
    }

    #[test]
    fn drawing_outside_the_canvas_is_clipped_not_a_panic() {
        let mut canvas = Canvas::new(4, 4);
        canvas.set(-1, -1, PaletteIndex(1));
        canvas.set(99, 99, PaletteIndex(1));
        canvas.fill_rect(-10, -10, 100, 100, PaletteIndex(1));
        assert_eq!(canvas.coverage(), 16, "the fill covered exactly the canvas");
    }

    #[test]
    fn set_if_empty_preserves_existing_detail() {
        let mut canvas = Canvas::new(4, 4);
        canvas.set(1, 1, PaletteIndex(2));
        canvas.set_if_empty(1, 1, PaletteIndex(1));
        assert_eq!(canvas.get(1, 1), PaletteIndex(2));
        canvas.set_if_empty(2, 2, PaletteIndex(1));
        assert_eq!(canvas.get(2, 2), PaletteIndex(1));
    }

    #[test]
    fn an_ellipse_is_filled_and_bounded() {
        let mut canvas = Canvas::new(16, 16);
        canvas.fill_ellipse(8, 8, 4, 2, PaletteIndex(1));

        assert_eq!(canvas.get(8, 8), PaletteIndex(1), "the centre is filled");
        assert_eq!(
            canvas.get(8, 12),
            PaletteIndex::TRANSPARENT,
            "past the vertical radius"
        );
        assert_ne!(
            canvas.get(11, 8),
            PaletteIndex::TRANSPARENT,
            "inside the horizontal radius"
        );
        assert_eq!(canvas.get(14, 8), PaletteIndex::TRANSPARENT, "past it");
    }

    #[test]
    fn a_degenerate_ellipse_draws_nothing() {
        let mut canvas = Canvas::new(8, 8);
        canvas.fill_ellipse(4, 4, 0, 3, PaletteIndex(1));
        assert_eq!(canvas.coverage(), 0);
    }

    #[test]
    fn a_line_connects_its_endpoints() {
        let mut canvas = Canvas::new(16, 16);
        canvas.draw_line(2, 2, 12, 9, PaletteIndex(1));
        assert_eq!(canvas.get(2, 2), PaletteIndex(1));
        assert_eq!(canvas.get(12, 9), PaletteIndex(1));
        assert!(canvas.coverage() >= 11, "the line should be continuous");
    }

    #[test]
    fn mirroring_produces_bilateral_symmetry() {
        let mut canvas = Canvas::new(8, 4);
        canvas.set(1, 1, PaletteIndex(1));
        canvas.set(3, 2, PaletteIndex(2));
        canvas.mirror_horizontally();

        assert_eq!(
            canvas.get(6, 1),
            PaletteIndex(1),
            "column 1 mirrors to column 6"
        );
        assert_eq!(
            canvas.get(4, 2),
            PaletteIndex(2),
            "column 3 mirrors to column 4"
        );
    }

    #[test]
    fn outlining_surrounds_a_shape_without_overwriting_it() {
        let mut canvas = Canvas::new(8, 8);
        canvas.fill_rect(3, 3, 2, 2, PaletteIndex(1));
        let before = canvas.coverage();

        canvas.outline(PaletteIndex(2));

        assert_eq!(canvas.get(3, 3), PaletteIndex(1), "the shape is untouched");
        assert_eq!(canvas.get(2, 3), PaletteIndex(2), "and is now bordered");
        assert!(canvas.coverage() > before);
        // Orthogonal only, so the diagonal corner stays clear.
        assert_eq!(canvas.get(2, 2), PaletteIndex::TRANSPARENT);
    }

    #[test]
    fn blitting_skips_transparent_source_pixels() {
        let mut base = Canvas::new(8, 8);
        base.fill_rect(0, 0, 8, 8, PaletteIndex(1));

        let mut stamp = Canvas::new(4, 4);
        stamp.set(0, 0, PaletteIndex(2));

        base.blit(&stamp, 2, 2);
        assert_eq!(base.get(2, 2), PaletteIndex(2), "the stamp's pixel landed");
        assert_eq!(
            base.get(3, 3),
            PaletteIndex(1),
            "and its transparency did not erase"
        );
    }

    #[test]
    fn translating_shifts_and_leaves_a_gap() {
        let mut canvas = Canvas::new(8, 8);
        canvas.set(1, 1, PaletteIndex(1));
        let shifted = canvas.translated(2, 3);

        assert_eq!(shifted.get(3, 4), PaletteIndex(1));
        assert_eq!(shifted.get(1, 1), PaletteIndex::TRANSPARENT);
        assert_eq!(
            shifted.coverage(),
            1,
            "content shifted off the edge is dropped"
        );
    }

    #[test]
    fn resolving_to_rgba_produces_four_bytes_per_pixel() {
        let mut canvas = Canvas::new(2, 2);
        canvas.set(0, 0, PaletteIndex(1));
        let bytes = canvas.to_rgba(&palette());

        assert_eq!(bytes.len(), 2 * 2 * 4);
        assert_eq!(
            &bytes[0..4],
            &[255, 0, 0, 255],
            "the set pixel resolved to red"
        );
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "and the rest to transparency");
    }

    #[test]
    fn replacing_swaps_every_occurrence() {
        let mut canvas = Canvas::new(4, 4);
        canvas.fill_rect(0, 0, 4, 4, PaletteIndex(1));
        canvas.replace(PaletteIndex(1), PaletteIndex(2));
        assert_eq!(canvas.get(2, 2), PaletteIndex(2));
    }

    #[test]
    fn the_palette_constraint_cannot_be_violated() {
        // A canvas stores indices, so an off-palette colour is not expressible:
        // the worst an out-of-range index can do is read as transparent.
        let mut canvas = Canvas::new(2, 2);
        canvas.set(0, 0, PaletteIndex(250));
        let bytes = canvas.to_rgba(&palette());
        assert_eq!(&bytes[0..4], &[0, 0, 0, 0]);
    }
}
