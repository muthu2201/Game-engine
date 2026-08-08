//! The sprite generators.
//!
//! Every generator is a pure function of `(seed, parameters)`, so the same
//! inputs always produce the same pixels. That is what lets the asset database
//! content-address generated art, what lets a bug in a sprite be reproduced
//! from its provenance record, and what makes regenerating a whole atlas at
//! startup cheaper than shipping one.

use crate::canvas::{Canvas, Palette, PaletteIndex, Rgba};
use verdant_core_math::Rng;

/// The four directions a character sprite is generated for.
///
/// Four rather than eight: at 16 pixels tall the difference between a
/// three-quarter view and a side view is a handful of pixels, and four
/// directions is what top-down games of this scale have always used.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Facing {
    /// Facing the camera.
    South,
    /// Facing away.
    North,
    /// Facing left.
    West,
    /// Facing right.
    East,
}

impl Facing {
    /// All four, in a fixed order.
    pub const ALL: [Facing; 4] = [Facing::South, Facing::North, Facing::West, Facing::East];

    /// The index this facing occupies in a generated sprite sheet.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The colours a character is generated from.
#[derive(Clone, Copy, Debug)]
pub struct CharacterStyle {
    /// Skin tone.
    pub skin: Rgba,
    /// Hair colour.
    pub hair: Rgba,
    /// Shirt colour.
    pub shirt: Rgba,
    /// Trousers colour.
    pub trousers: Rgba,
    /// Boot colour.
    pub boots: Rgba,
}

impl CharacterStyle {
    /// A default villager palette.
    #[must_use]
    pub fn villager() -> CharacterStyle {
        CharacterStyle {
            skin: Rgba::hex(0xE8_B0_8C),
            hair: Rgba::hex(0x6B_43_2A),
            shirt: Rgba::hex(0x4A_7C_B5),
            trousers: Rgba::hex(0x3A_3F_52),
            boots: Rgba::hex(0x4A_36_25),
        }
    }

    /// A style with randomised colouring, for generated NPCs.
    ///
    /// Hues are drawn from curated lists rather than the full colour space: a
    /// uniformly random RGB triple produces muddy, clashing villagers, whereas
    /// picking from a hand-chosen set keeps every generated character looking
    /// like it belongs to the same world.
    #[must_use]
    pub fn random(rng: &mut Rng) -> CharacterStyle {
        const SKINS: [u32; 6] = [
            0xF2_C9_A0, 0xE8_B0_8C, 0xC6_8A_63, 0x9C_63_42, 0x6F_45_2C, 0x4A_2F_1E,
        ];
        const HAIRS: [u32; 7] = [
            0x2B_1D_14, 0x6B_43_2A, 0xA8_6C_3D, 0xD9_A6_5B, 0x8C_8C_8C, 0x4A_2C_5A, 0xB5_4A_3A,
        ];
        const CLOTH: [u32; 8] = [
            0x4A_7C_B5, 0xB5_4A_4A, 0x5A_8C_4A, 0xC9_8A_3D, 0x7A_5A_9C, 0x3D_8C_8C, 0xC4_6A_8C,
            0x8C_7A_5A,
        ];

        let pick =
            |rng: &mut Rng, options: &[u32]| Rgba::hex(*rng.pick(options).unwrap_or(&options[0]));
        CharacterStyle {
            skin: pick(rng, &SKINS),
            hair: pick(rng, &HAIRS),
            shirt: pick(rng, &CLOTH),
            trousers: pick(rng, &CLOTH),
            boots: Rgba::hex(0x4A_36_25),
        }
    }
}

/// A generated sprite: the artwork plus the palette it resolves against.
#[derive(Clone, Debug)]
pub struct GeneratedSprite {
    /// The indexed artwork, one canvas per frame.
    pub frames: Vec<Canvas>,
    /// The palette every frame resolves against.
    pub palette: Palette,
}

impl GeneratedSprite {
    /// Resolves one frame into RGBA bytes.
    ///
    /// Returns `None` when the index is out of range.
    #[must_use]
    pub fn frame_rgba(&self, frame: usize) -> Option<Vec<u8>> {
        self.frames
            .get(frame)
            .map(|canvas| canvas.to_rgba(&self.palette))
    }

    /// Number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Frames per direction in a generated walk cycle.
///
/// Four is the classic top-down cycle: contact, passing, contact, passing, with
/// the two passing poses mirrored. Fewer reads as a stutter; more is invisible
/// at this sprite size.
pub const WALK_FRAMES: usize = 4;

/// Generates a character with a four-direction walk cycle.
///
/// The returned sprite holds `4 * WALK_FRAMES` frames, ordered by
/// [`Facing::index`] then by frame within the cycle.
///
/// # Panics
///
/// Panics if `width` or `height` is zero.
#[must_use]
pub fn generate_character(
    seed: u64,
    width: u32,
    height: u32,
    style: CharacterStyle,
) -> GeneratedSprite {
    assert!(
        width > 0 && height > 0,
        "generate_character: dimensions must be non-zero"
    );
    let rng = Rng::new(seed);

    let mut palette = Palette::new("character", &[]);
    let outline = palette.push(Rgba::hex(0x1A_14_1A));
    let skin = palette.add_ramp(style.skin, 3);
    let hair = palette.add_ramp(style.hair, 3);
    let shirt = palette.add_ramp(style.shirt, 3);
    let trousers = palette.add_ramp(style.trousers, 3);
    let boots = palette.add_ramp(style.boots, 2);

    let parts = Parts {
        outline,
        skin: &skin,
        hair: &hair,
        shirt: &shirt,
        trousers: &trousers,
        boots: &boots,
    };

    let mut frames = Vec::with_capacity(4 * WALK_FRAMES);
    for facing in Facing::ALL {
        for frame in 0..WALK_FRAMES {
            // Each frame draws from its own stream, so the scattered hair
            // strands stay put across the cycle instead of shimmering.
            let mut frame_rng = rng.derive(&format!("facing-{}", facing.index()));
            frames.push(draw_character_frame(
                &mut frame_rng,
                width,
                height,
                facing,
                frame,
                &parts,
            ));
        }
    }

    GeneratedSprite { frames, palette }
}

/// The palette ramps a character body is drawn from.
struct Parts<'a> {
    outline: PaletteIndex,
    skin: &'a [PaletteIndex],
    hair: &'a [PaletteIndex],
    shirt: &'a [PaletteIndex],
    trousers: &'a [PaletteIndex],
    boots: &'a [PaletteIndex],
}

/// Reads a ramp entry, clamping rather than indexing out of range.
fn ramp(indices: &[PaletteIndex], step: usize) -> PaletteIndex {
    indices
        .get(step)
        .copied()
        .unwrap_or(PaletteIndex::TRANSPARENT)
}

/// The proportions a character is laid out against.
///
/// Expressed as fractions of the sprite's size so one generator serves 16, 24
/// and 32 pixel characters. The head takes a full third of the height: real
/// human proportions read as spindly and unappealing at this scale, and every
/// top-down game of this kind exaggerates the head for the same reason.
struct Proportions {
    width: i32,
    height: i32,
    centre: i32,
    head_height: i32,
    head_half_width: i32,
    torso_top: i32,
    torso_half_width: i32,
    leg_top: i32,
    foot_top: i32,
}

impl Proportions {
    /// Derives the layout for a sprite of the given size.
    fn new(width: u32, height: u32) -> Proportions {
        let w = width as i32;
        let h = height as i32;
        let head_height = (h * 3 / 8).max(4);
        let torso_height = (h * 5 / 16).max(3);
        Proportions {
            width: w,
            height: h,
            centre: w / 2,
            head_height,
            head_half_width: (w * 5 / 16).max(2),
            torso_top: head_height,
            torso_half_width: (w * 3 / 16).max(2),
            leg_top: head_height + torso_height,
            foot_top: h - (h / 12).max(1) - 1,
        }
    }
}

/// How far each leg is displaced on one frame of the walk cycle.
///
/// Frames 0 and 2 are the contact poses, identical to standing, so an idle
/// character reuses frame 0 and no separate idle art is needed. Frames 1 and 3
/// are the passing poses, mirrored from each other.
const LEG_POSE: [(i32, i32); WALK_FRAMES] = [(0, 0), (1, -1), (0, 0), (-1, 1)];

/// Draws one fully posed frame.
///
/// Every frame is drawn from scratch rather than patched onto a standing pose:
/// patching leaves the limbs overlapping the body they were meant to replace,
/// which is why an earlier version of this generator produced a walk cycle with
/// no visible motion.
#[allow(clippy::too_many_lines)]
fn draw_character_frame(
    rng: &mut Rng,
    width: u32,
    height: u32,
    facing: Facing,
    frame: usize,
    parts: &Parts<'_>,
) -> Canvas {
    let p = Proportions::new(width, height);
    let mut canvas = Canvas::new(width, height);

    // Passing poses lift the whole body by a pixel, which is most of what
    // reads as a walk at this scale.
    let bob = i32::from(matches!(frame, 1 | 3));
    let (front_leg, back_leg) = LEG_POSE[frame % WALK_FRAMES];

    // ---- Legs and boots -----------------------------------------------------
    let leg_width = (p.width / 6).max(2);
    let leg_gap = 1;
    let left_leg_x = p.centre - leg_gap - leg_width;
    let right_leg_x = p.centre + leg_gap;

    // Side views step forward and back along x; front and back views step up
    // and down, since a leg moving toward the camera is foreshortened.
    let (left_dx, right_dx, left_dy, right_dy) = match facing {
        Facing::West | Facing::East => (front_leg, back_leg, 0, 0),
        Facing::North | Facing::South => (0, 0, -front_leg.max(0), -back_leg.max(0)),
    };

    for (x, dx, dy, shade) in [
        (left_leg_x, left_dx, left_dy, 1usize),
        (right_leg_x, right_dx, right_dy, 0usize),
    ] {
        canvas.fill_rect(
            x + dx,
            p.leg_top - bob,
            leg_width,
            p.foot_top - p.leg_top + dy.abs(),
            ramp(parts.trousers, shade),
        );
        canvas.fill_rect(
            x + dx,
            p.foot_top + dy,
            leg_width,
            p.height - p.foot_top - 1,
            ramp(parts.boots, 0),
        );
    }

    // ---- Torso --------------------------------------------------------------
    let torso_top = p.torso_top - bob;
    let torso_height = p.leg_top - p.torso_top;
    canvas.fill_rect(
        p.centre - p.torso_half_width,
        torso_top,
        p.torso_half_width * 2,
        torso_height,
        ramp(parts.shirt, 0),
    );
    // One shade darker down the left, so every sprite in the game is lit from
    // the same direction.
    canvas.fill_rect(
        p.centre - p.torso_half_width,
        torso_top,
        (p.torso_half_width * 2 / 5).max(1),
        torso_height,
        ramp(parts.shirt, 1),
    );

    // ---- Arms ---------------------------------------------------------------
    // Drawn outside the torso rather than over it, so they are actually
    // visible; arms tucked inside the silhouette read as a featureless block.
    let arm_width = (p.width / 8).max(1);
    let arm_height = torso_height * 3 / 4;
    let arm_swing = match facing {
        // Arms counter-swing against the legs.
        Facing::West | Facing::East => (-front_leg, -back_leg),
        Facing::North | Facing::South => (0, 0),
    };
    for (x, swing, shade) in [
        (
            p.centre - p.torso_half_width - arm_width,
            arm_swing.0,
            2usize,
        ),
        (p.centre + p.torso_half_width, arm_swing.1, 1usize),
    ] {
        canvas.fill_rect(
            x + swing,
            torso_top + 1,
            arm_width,
            arm_height,
            ramp(parts.shirt, shade),
        );
        // A hand at the end of each sleeve.
        canvas.fill_rect(
            x + swing,
            torso_top + 1 + arm_height,
            arm_width,
            1,
            ramp(parts.skin, 1),
        );
    }

    // ---- Head ---------------------------------------------------------------
    // Profiles shift the head off-centre, which is what sells a side view more
    // than any amount of facial detail at this size.
    let head_shift = match facing {
        Facing::West => -1,
        Facing::East => 1,
        _ => 0,
    };
    let head_centre_x = p.centre + head_shift;
    let head_centre_y = p.head_height / 2 - bob;
    canvas.fill_ellipse(
        head_centre_x,
        head_centre_y,
        p.head_half_width,
        p.head_height / 2,
        ramp(parts.skin, 0),
    );
    // Shade the side away from the light.
    canvas.fill_ellipse(
        head_centre_x + 1,
        head_centre_y + 1,
        p.head_half_width - 1,
        p.head_height / 2 - 1,
        ramp(parts.skin, 1),
    );
    canvas.fill_ellipse(
        head_centre_x - 1,
        head_centre_y - 1,
        p.head_half_width - 1,
        (p.head_height / 2 - 1).max(1),
        ramp(parts.skin, 0),
    );

    // ---- Hair ---------------------------------------------------------------
    // How far down the skull the hair reaches, per direction. Seeing the back
    // of a head means seeing mostly hair, which is the clearest signal that a
    // character has turned away.
    let hair_depth = match facing {
        Facing::North => p.head_height * 4 / 5,
        Facing::South => p.head_height * 2 / 5,
        Facing::West | Facing::East => p.head_height / 2,
    };
    for y in 0..hair_depth {
        // Taper the fringe at the very top so the hairline is rounded rather
        // than a flat cap.
        let inset = i32::from(y == 0);
        canvas.fill_rect(
            head_centre_x - p.head_half_width + inset,
            y - bob,
            p.head_half_width * 2 - inset * 2,
            1,
            ramp(parts.hair, 0),
        );
    }
    // Sideburns on the profile views, which break the head's symmetry and stop
    // west and east reading as the same sprite.
    if matches!(facing, Facing::West | Facing::East) {
        let side = if facing == Facing::West { 1 } else { -1 };
        canvas.fill_rect(
            head_centre_x + side * (p.head_half_width - 1),
            hair_depth - bob,
            1,
            (p.head_height / 4).max(1),
            ramp(parts.hair, 0),
        );
    }
    // A few lighter strands, so hair is not a flat helmet.
    for _ in 0..p.head_half_width.max(1) {
        let x = head_centre_x - p.head_half_width + rng.range(0, (p.head_half_width * 2).max(1));
        canvas.set(
            x,
            rng.range(0, hair_depth.max(1)) - bob,
            ramp(parts.hair, 1),
        );
    }

    // ---- Face ---------------------------------------------------------------
    let eye_y = head_centre_y + (p.head_height / 8).max(0);
    let eye_offset = (p.head_half_width / 2).max(1);
    match facing {
        Facing::South => {
            canvas.set(head_centre_x - eye_offset, eye_y, parts.outline);
            canvas.set(head_centre_x + eye_offset, eye_y, parts.outline);
        }
        // A profile shows one eye, set forward on the face.
        Facing::West => {
            canvas.set(head_centre_x - eye_offset, eye_y, parts.outline);
        }
        Facing::East => {
            canvas.set(head_centre_x + eye_offset, eye_y, parts.outline);
        }
        // Facing away, there is no face to draw.
        Facing::North => {}
    }

    canvas.outline(parts.outline);
    canvas
}

/// Generates a terrain tile with subtle per-pixel variation.
///
/// The noise is what stops a field of grass reading as a flat colour: without
/// it, a tiled floor looks like graph paper.
///
/// # Panics
///
/// Panics if `size` is zero.
#[must_use]
pub fn generate_terrain_tile(seed: u64, size: u32, base: Rgba, roughness: u8) -> GeneratedSprite {
    assert!(size > 0, "generate_terrain_tile: size must be non-zero");
    let mut rng = Rng::new(seed);

    let mut palette = Palette::new("terrain", &[]);
    let shades = palette.add_ramp(base, 4);

    let mut canvas = Canvas::new(size, size);
    for y in 0..size as i32 {
        for x in 0..size as i32 {
            // Most pixels take the base shade; the rest scatter across the ramp.
            let top_shade = i32::try_from(shades.len().saturating_sub(1)).unwrap_or(1);
            let shade = if rng.chance(u32::from(roughness), 255) {
                usize::try_from(rng.range(1, top_shade)).unwrap_or(0)
            } else {
                0
            };
            canvas.set(x, y, ramp(&shades, shade));
        }
    }

    GeneratedSprite {
        frames: vec![canvas],
        palette,
    }
}

/// Generates the growth stages of a crop, from sprout to harvestable.
///
/// Returns one frame per stage. The plant grows taller and gains its fruit
/// colour only at the final stage, so a player can read ripeness at a glance —
/// which is the entire point of a farming game's visual language.
///
/// # Panics
///
/// Panics if `size` is zero or `stages` is zero.
#[must_use]
pub fn generate_crop(
    seed: u64,
    size: u32,
    stages: usize,
    leaf: Rgba,
    fruit: Rgba,
) -> GeneratedSprite {
    assert!(size > 0, "generate_crop: size must be non-zero");
    assert!(stages > 0, "generate_crop: at least one stage is required");
    let rng = Rng::new(seed);

    let mut palette = Palette::new("crop", &[]);
    let outline = palette.push(Rgba::hex(0x1A2A16));
    let leaves = palette.add_ramp(leaf, 3);
    let fruits = palette.add_ramp(fruit, 2);
    let soil = palette.push(Rgba::hex(0x5A_43_2E));

    let mut frames = Vec::with_capacity(stages);
    for stage in 0..stages {
        let mut stage_rng = rng.derive(&format!("stage-{stage}"));
        let mut canvas = Canvas::new(size, size);
        let width = size as i32;
        let height = size as i32;
        let centre = width / 2;

        // A patch of tilled soil under every stage.
        canvas.fill_rect(centre - width / 3, height - 2, (width * 2) / 3, 2, soil);

        // Height grows linearly with the stage.
        let progress = i32::try_from(stage + 1).unwrap_or(i32::MAX);
        let total = i32::try_from(stages).unwrap_or(i32::MAX);
        let plant_height = ((height - 3) * progress / total).max(1);
        let top = height - 2 - plant_height;

        // Stem.
        canvas.draw_line(centre, height - 2, centre, top, ramp(&leaves, 1));

        // Leaves, more of them as the plant matures.
        let leaf_count = progress.max(1);
        for index in 0..leaf_count {
            let y = top + (plant_height * index / leaf_count.max(1));
            let span = 1 + (plant_height / 4).min(3);
            let side = if index % 2 == 0 { 1 } else { -1 };
            canvas.fill_ellipse(centre + side * (span / 2 + 1), y, span, 1, ramp(&leaves, 0));
        }

        // Fruit only on the final stage.
        if stage + 1 == stages {
            let fruit_count = 1 + stage_rng.range(0, 2);
            for _ in 0..fruit_count {
                let x = centre + stage_rng.range(-2, 2);
                let y = top + stage_rng.range(0, plant_height.max(1) / 2);
                canvas.fill_ellipse(x, y, 1, 1, ramp(&fruits, 0));
            }
        }

        canvas.outline(outline);
        frames.push(canvas);
    }

    GeneratedSprite { frames, palette }
}

/// Generates a tree: trunk, canopy, and a shaded side.
///
/// # Panics
///
/// Panics if either dimension is zero.
#[must_use]
pub fn generate_tree(seed: u64, width: u32, height: u32, foliage: Rgba) -> GeneratedSprite {
    assert!(
        width > 0 && height > 0,
        "generate_tree: dimensions must be non-zero"
    );
    let mut rng = Rng::new(seed);

    let mut palette = Palette::new("tree", &[]);
    let outline = palette.push(Rgba::hex(0x16_20_12));
    let leaves = palette.add_ramp(foliage, 4);
    let bark = palette.add_ramp(Rgba::hex(0x6B_4A_2E), 2);

    let mut canvas = Canvas::new(width, height);
    let w = width as i32;
    let h = height as i32;
    let centre = w / 2;

    // Trunk.
    let trunk_width = (w / 6).max(2);
    let trunk_top = h / 2;
    canvas.fill_rect(
        centre - trunk_width / 2,
        trunk_top,
        trunk_width,
        h - trunk_top - 1,
        ramp(&bark, 0),
    );
    canvas.fill_rect(
        centre - trunk_width / 2,
        trunk_top,
        trunk_width / 3,
        h - trunk_top - 1,
        ramp(&bark, 1),
    );

    // Canopy: overlapping ellipses, which reads as foliage where a single
    // ellipse reads as a balloon.
    let canopy_radius = (w / 2 - 1).max(2);
    canvas.fill_ellipse(centre, h / 3, canopy_radius, h / 3, ramp(&leaves, 1));
    for _ in 0..5 {
        let x = centre + rng.range(-canopy_radius / 2, canopy_radius / 2);
        let y = h / 3 + rng.range(-h / 8, h / 8);
        let radius = (canopy_radius / 2).max(1) + rng.range(0, 2);
        canvas.fill_ellipse(x, y, radius, radius * 3 / 4, ramp(&leaves, 0));
    }
    // A darker underside, so the canopy has a light direction.
    canvas.fill_ellipse(
        centre + 1,
        h / 3 + h / 8,
        canopy_radius / 2,
        h / 8,
        ramp(&leaves, 2),
    );

    canvas.outline(outline);
    GeneratedSprite {
        frames: vec![canvas],
        palette,
    }
}

/// Integer square root, by Newton's method.
///
/// Used instead of `f64::sqrt` so a generated silhouette is bit-identical
/// everywhere, for the same reason the simulation avoids floats.
fn isqrt(value: i32) -> i32 {
    if value <= 0 {
        return 0;
    }
    let mut guess = value;
    let mut next = (guess + 1) / 2;
    while next < guess {
        guess = next;
        next = (guess + value / guess) / 2;
    }
    guess
}

/// Generates a symmetric item icon.
///
/// The shape is built as a vertically varying *silhouette width* rather than by
/// scattering pixels: each row gets one contiguous run whose width changes
/// smoothly, so the result is a single rounded object. Scattering per pixel
/// produces horizontal stripes that read as noise, which is what an earlier
/// version of this generator did.
///
/// # Panics
///
/// Panics if `size` is zero.
#[must_use]
pub fn generate_item_icon(seed: u64, size: u32, base: Rgba) -> GeneratedSprite {
    assert!(size > 0, "generate_item_icon: size must be non-zero");
    let mut rng = Rng::new(seed);

    let mut palette = Palette::new("item", &[]);
    let outline = palette.push(Rgba::hex(0x1A_14_1A));
    let shades = palette.add_ramp(base, 3);

    let mut canvas = Canvas::new(size, size);
    let half = (size / 2) as i32;
    let height = size as i32;
    let top = 2;
    let bottom = height - 2;
    let span = (bottom - top).max(1);

    // A base elliptical profile, perturbed by a small random walk so no two
    // items share a silhouette while every one stays a single blob.
    let mut wobble = 0i32;
    for y in top..bottom {
        // Position within the shape, 0 at the top, 1 at the bottom.
        let t = (y - top) * 2 - span;
        // Elliptical half-width: widest in the middle, tapering to the ends.
        // Integer square root, so the silhouette is identical on every machine
        // rather than depending on the platform's floating-point sqrt.
        let elliptical = isqrt((span * span - t * t).max(0)) * half / span.max(1);

        wobble = (wobble + rng.range(-1, 1)).clamp(-1, 1);
        let width = (elliptical + wobble).clamp(1, half);

        for x in (half - width)..half {
            // Shade by distance from the lit edge, so the object reads as
            // rounded rather than flat.
            let depth = usize::from(x < half - width * 2 / 3);
            canvas.set(x, y, ramp(&shades, depth));
        }
    }

    // A highlight on the upper-left, the same light direction every other
    // generator uses.
    canvas.fill_ellipse(
        half - half / 2,
        top + span / 3,
        (half / 4).max(1),
        (span / 8).max(1),
        ramp(&shades, 0),
    );

    canvas.mirror_horizontally();
    canvas.outline(outline);

    GeneratedSprite {
        frames: vec![canvas],
        palette,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_character_has_a_frame_for_every_direction_and_step() {
        let sprite = generate_character(1, 16, 24, CharacterStyle::villager());
        assert_eq!(sprite.frame_count(), 4 * WALK_FRAMES);
        assert!(
            sprite.frames.iter().all(|frame| frame.coverage() > 0),
            "no frame is blank"
        );
    }

    #[test]
    fn character_frames_resolve_to_the_right_number_of_bytes() {
        let sprite = generate_character(2, 16, 24, CharacterStyle::villager());
        let bytes = sprite.frame_rgba(0).expect("frame 0 exists");
        assert_eq!(bytes.len(), 16 * 24 * 4);
        assert_eq!(sprite.frame_rgba(999), None);
    }

    #[test]
    fn generation_is_deterministic() {
        let first = generate_character(42, 16, 24, CharacterStyle::villager());
        let second = generate_character(42, 16, 24, CharacterStyle::villager());
        assert_eq!(
            first.frames, second.frames,
            "the same seed must produce the same art"
        );
    }

    #[test]
    fn different_seeds_produce_different_characters() {
        let first = generate_character(1, 16, 24, CharacterStyle::villager());
        let second = generate_character(2, 16, 24, CharacterStyle::villager());
        assert_ne!(first.frames, second.frames);
    }

    #[test]
    fn walk_frames_differ_from_the_standing_pose() {
        let sprite = generate_character(7, 16, 24, CharacterStyle::villager());
        // Within one direction, the passing poses must differ from contact.
        assert_ne!(
            sprite.frames[0], sprite.frames[1],
            "frame 1 should be a passing pose"
        );
        assert_ne!(
            sprite.frames[1], sprite.frames[3],
            "the two passing poses differ"
        );
    }

    #[test]
    fn each_direction_looks_different() {
        let sprite = generate_character(9, 16, 24, CharacterStyle::villager());
        let south = &sprite.frames[Facing::South.index() * WALK_FRAMES];
        let north = &sprite.frames[Facing::North.index() * WALK_FRAMES];
        assert_ne!(
            south, north,
            "the back of a character must not match the front"
        );
    }

    #[test]
    fn a_character_is_outlined() {
        let sprite = generate_character(11, 16, 24, CharacterStyle::villager());
        let frame = &sprite.frames[0];
        // The outline index is 1 in the character palette.
        let has_outline = frame.pixels().contains(&PaletteIndex(1));
        assert!(
            has_outline,
            "sprites need an outline to read against terrain"
        );
    }

    #[test]
    fn random_styles_stay_within_the_curated_sets() {
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let style = CharacterStyle::random(&mut rng);
            // Every curated colour is opaque; a fully transparent or black
            // result would mean the pick fell through.
            assert_eq!(style.skin.a, 255);
            assert!(style.hair.r > 0 || style.hair.g > 0 || style.hair.b > 0);
        }
    }

    #[test]
    fn a_terrain_tile_is_fully_opaque() {
        let sprite = generate_terrain_tile(3, 16, Rgba::hex(0x6B_8E_23), 60);
        let tile = &sprite.frames[0];
        assert_eq!(
            tile.coverage(),
            16 * 16,
            "terrain must not have holes in it"
        );
    }

    #[test]
    fn a_terrain_tile_has_visible_variation() {
        let sprite = generate_terrain_tile(4, 16, Rgba::hex(0x6B_8E_23), 90);
        let tile = &sprite.frames[0];
        let distinct: std::collections::HashSet<_> = tile.pixels().iter().collect();
        assert!(distinct.len() > 1, "a flat tile would read as graph paper");
    }

    #[test]
    fn zero_roughness_produces_a_flat_tile() {
        let sprite = generate_terrain_tile(5, 8, Rgba::hex(0x6B_8E_23), 0);
        let distinct: std::collections::HashSet<_> = sprite.frames[0].pixels().iter().collect();
        assert_eq!(distinct.len(), 1);
    }

    #[test]
    fn crops_grow_taller_at_each_stage() {
        let sprite = generate_crop(6, 16, 5, Rgba::hex(0x4A_8C_3A), Rgba::hex(0xD9_4A_3A));
        assert_eq!(sprite.frame_count(), 5);

        /// The topmost non-transparent row of a canvas.
        fn top_row(canvas: &Canvas) -> i32 {
            for y in 0..canvas.height() as i32 {
                for x in 0..canvas.width() as i32 {
                    if canvas.get(x, y) != PaletteIndex::TRANSPARENT {
                        return y;
                    }
                }
            }
            canvas.height() as i32
        }

        let tops: Vec<i32> = sprite.frames.iter().map(top_row).collect();
        assert!(
            tops[0] > tops[4],
            "the mature crop should reach higher: {tops:?}"
        );
    }

    #[test]
    fn only_the_final_crop_stage_bears_fruit() {
        let sprite = generate_crop(8, 16, 4, Rgba::hex(0x4A_8C_3A), Rgba::hex(0xD9_4A_3A));
        // The fruit ramp starts after the outline and the three leaf shades.
        let fruit_indices = [PaletteIndex(5), PaletteIndex(6)];
        let bears_fruit = |canvas: &Canvas| {
            canvas
                .pixels()
                .iter()
                .any(|index| fruit_indices.contains(index))
        };
        assert!(
            !bears_fruit(&sprite.frames[0]),
            "a sprout must not have fruit"
        );
        assert!(bears_fruit(&sprite.frames[3]), "the mature stage should");
    }

    #[test]
    fn a_tree_has_a_trunk_beneath_its_canopy() {
        let sprite = generate_tree(10, 24, 32, Rgba::hex(0x3A_6B_2A));
        let tree = &sprite.frames[0];
        // The bottom rows should be occupied by trunk, the top by canopy.
        let bottom_filled = (0..24).any(|x| tree.get(x, 30) != PaletteIndex::TRANSPARENT);
        let top_filled = (0..24).any(|x| tree.get(x, 6) != PaletteIndex::TRANSPARENT);
        assert!(bottom_filled && top_filled);
    }

    #[test]
    fn an_item_icon_is_bilaterally_symmetric() {
        let sprite = generate_item_icon(12, 16, Rgba::hex(0xC9_A0_3D));
        let icon = &sprite.frames[0];
        for y in 0..16i32 {
            for x in 0..8i32 {
                assert_eq!(
                    icon.get(x, y),
                    icon.get(15 - x, y),
                    "the icon is not symmetric at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn an_item_icon_is_not_blank() {
        for seed in 0..20u64 {
            let sprite = generate_item_icon(seed, 16, Rgba::hex(0xC9_A0_3D));
            assert!(
                sprite.frames[0].coverage() > 4,
                "seed {seed} produced an empty icon"
            );
        }
    }

    /// The widest run of non-transparent pixels in a row band.
    fn widest_row(canvas: &Canvas, from_y: i32, to_y: i32) -> i32 {
        let mut widest = 0;
        for y in from_y..to_y {
            let mut leftmost = None;
            let mut rightmost = 0;
            for x in 0..canvas.width() as i32 {
                if canvas.get(x, y) != PaletteIndex::TRANSPARENT {
                    leftmost.get_or_insert(x);
                    rightmost = x;
                }
            }
            if let Some(left) = leftmost {
                widest = widest.max(rightmost - left + 1);
            }
        }
        widest
    }

    #[test]
    fn a_characters_head_is_not_narrower_than_its_torso() {
        // The first version of this generator produced a torso wider than the
        // head, which read as a blob rather than a person. Appeal at this
        // sprite size depends on an exaggerated head, so the proportion is
        // asserted rather than left to drift.
        let sprite = generate_character(3, 16, 24, CharacterStyle::villager());
        let frame = &sprite.frames[0];
        let head = widest_row(frame, 0, 9);
        let torso = widest_row(frame, 10, 14);
        assert!(
            head >= torso,
            "head was {head} wide but the torso was {torso}"
        );
    }

    #[test]
    fn walk_frames_move_the_legs_visibly() {
        // Patching a standing pose left the limbs overlapping the body they
        // replaced, so the cycle animated by almost nothing. Require a real
        // difference in the leg band.
        let sprite = generate_character(4, 16, 24, CharacterStyle::villager());
        for facing in Facing::ALL {
            let base = facing.index() * WALK_FRAMES;
            let contact = &sprite.frames[base];
            let passing = &sprite.frames[base + 1];

            let differing = (16..24i32)
                .flat_map(|y| (0..16i32).map(move |x| (x, y)))
                .filter(|(x, y)| contact.get(*x, *y) != passing.get(*x, *y))
                .count();
            assert!(
                differing >= 4,
                "{facing:?} moved only {differing} leg pixels between frames"
            );
        }
    }

    #[test]
    fn a_character_has_arms_outside_its_torso() {
        // Arms drawn inside the silhouette are invisible; they must widen it.
        let sprite = generate_character(5, 16, 24, CharacterStyle::villager());
        let frame = &sprite.frames[0];
        let with_arms = widest_row(frame, 10, 14);
        let legs = widest_row(frame, 18, 22);
        assert!(
            with_arms > legs,
            "the torso-and-arms band should be the widest part"
        );
    }

    #[test]
    fn an_item_icon_is_a_single_connected_shape() {
        // Scattering pixels per row produced horizontal stripes that read as
        // noise. A real object is one connected region.
        for seed in 0..12u64 {
            let sprite = generate_item_icon(seed, 16, Rgba::hex(0xC9_A0_3D));
            let icon = &sprite.frames[0];

            // Flood fill from the first filled pixel and confirm it reaches all.
            let filled: Vec<(i32, i32)> = (0..16i32)
                .flat_map(|y| (0..16i32).map(move |x| (x, y)))
                .filter(|(x, y)| icon.get(*x, *y) != PaletteIndex::TRANSPARENT)
                .collect();
            let start = filled[0];

            let mut reached = vec![start];
            let mut pending = vec![start];
            while let Some((x, y)) = pending.pop() {
                for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
                    let next = (x + dx, y + dy);
                    if icon.get(next.0, next.1) != PaletteIndex::TRANSPARENT
                        && !reached.contains(&next)
                    {
                        reached.push(next);
                        pending.push(next);
                    }
                }
            }
            assert_eq!(
                reached.len(),
                filled.len(),
                "seed {seed} produced {} disconnected fragments",
                filled.len() - reached.len()
            );
        }
    }

    #[test]
    fn generators_work_at_several_sprite_sizes() {
        for size in [8u32, 16, 24, 32] {
            let character = generate_character(1, size, size * 3 / 2, CharacterStyle::villager());
            assert!(
                character.frames[0].coverage() > 0,
                "size {size} produced nothing"
            );

            let crop = generate_crop(1, size, 4, Rgba::hex(0x4A_8C_3A), Rgba::hex(0xD9_4A_3A));
            assert!(crop.frames[0].coverage() > 0);

            let tree = generate_tree(1, size, size, Rgba::hex(0x3A_6B_2A));
            assert!(tree.frames[0].coverage() > 0);
        }
    }

    #[test]
    fn every_generator_is_reproducible() {
        assert_eq!(
            generate_terrain_tile(1, 16, Rgba::hex(0x6B_8E_23), 50).frames,
            generate_terrain_tile(1, 16, Rgba::hex(0x6B_8E_23), 50).frames
        );
        assert_eq!(
            generate_crop(1, 16, 4, Rgba::hex(0x4A_8C_3A), Rgba::hex(0xD9_4A_3A)).frames,
            generate_crop(1, 16, 4, Rgba::hex(0x4A_8C_3A), Rgba::hex(0xD9_4A_3A)).frames
        );
        assert_eq!(
            generate_tree(1, 24, 32, Rgba::hex(0x3A_6B_2A)).frames,
            generate_tree(1, 24, 32, Rgba::hex(0x3A_6B_2A)).frames
        );
        assert_eq!(
            generate_item_icon(1, 16, Rgba::hex(0xC9_A0_3D)).frames,
            generate_item_icon(1, 16, Rgba::hex(0xC9_A0_3D)).frames
        );
    }
}
