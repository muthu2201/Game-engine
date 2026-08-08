//! Turning game state into pixels.
//!
//! The module is deliberately split in two:
//!
//! * [`Art`] and [`Scene`] are pure CPU. Art generates every sprite the game
//!   needs from the world seed and packs it into atlas pages; a scene walks the
//!   game state and produces a list of [`DrawSprite`]s. Neither touches a GPU,
//!   which is why the layout, the depth sorting and the HUD can all be tested
//!   in CI on a machine with no graphics stack.
//! * [`GameRenderer`] is the GPU half: it uploads the atlas once, renders a
//!   scene into a low-resolution target, and presents that target to a window.
//!
//! ## No asset files
//!
//! Nothing here loads an image. Every tile, crop, character and glyph is
//! generated at startup from the world seed by
//! [`verdant_procgen_art`], which takes a few milliseconds and means the game
//! has no art directory to ship, no loader to fail, and no possibility of a
//! sprite going missing at runtime.

use crate::calendar::Weather;
use crate::farm::Plot;
use crate::inventory::HOTBAR_SLOTS;
use crate::items::{crop, item, CropId, ItemId, CROPS, ITEMS};
use crate::sim::{ActionOutcome, FacingState, Game, MAX_ENERGY, MAX_FRIENDSHIP};
use crate::world::{tiles, TILE_SIZE};
use std::collections::BTreeMap;
use verdant_core_math::{Fx, IVec2, Rect, Rng, Vec2};
use verdant_input::{actions, Thumbstick, TouchButton, TouchLayout, TouchState};
use verdant_procgen_art::{
    font, generate_character, generate_crop, generate_item_icon, generate_terrain_tile,
    generate_tree, Canvas, CharacterStyle, Palette, PaletteIndex, Rgba, WALK_FRAMES,
};
use verdant_render_2d::{
    Camera2D, Color, Compositor, DrawSprite, Light, LightRenderer, LightSettings, SpriteBatcher,
    TextureArray,
};
use verdant_tilemap::TileId;

/// Width of the frame the game renders into, in pixels.
///
/// 480×270 is exactly one quarter of 1920×1080, so a fullscreen window on the
/// most common display scales by a whole 4× with no letterbox at all.
pub const INTERNAL_WIDTH: u32 = 480;

/// Height of the frame the game renders into, in pixels.
pub const INTERNAL_HEIGHT: u32 = 270;

/// Edge length of one atlas page, in pixels.
///
/// 256 is the largest 2D-array layer size guaranteed by wgpu's downlevel
/// limits, which is the limit set the engine targets so that the game runs
/// identically on a software rasteriser and on a desktop GPU.
pub const ATLAS_PAGE: u32 = 256;

/// How many alternative sprites each ground tile gets.
///
/// Four is the point where a wide field stops reading as a repeating stamp;
/// beyond it the atlas grows faster than the eye notices.
pub const TERRAIN_VARIANTS: usize = 4;

/// How many tree shapes are generated.
pub const TREE_VARIANTS: usize = 4;

/// Draw order for ground tiles.
const Z_GROUND: i32 = 0;
/// Draw order for things painted onto the ground: tilled soil, water.
const Z_OVERLAY: i32 = 10;
/// Draw order for everything standing on the ground. Sorted by world Y within
/// this band, which is what makes a character walk behind a tree and in front
/// of the next one down.
const Z_STANDING: i32 = 20;
/// Draw order for the HUD's panels.
const Z_HUD_PANEL: i32 = 100;
/// Draw order for HUD text, above its panel.
const Z_HUD_TEXT: i32 = 110;
/// Draw order for the on-screen touch controls, above everything.
const Z_TOUCH: i32 = 120;

/// A packed sprite's place in the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Region {
    /// Which atlas page holds it.
    pub layer: u32,
    /// Where on that page, in texels.
    pub rect: (u32, u32, u32, u32),
}

impl Region {
    /// The region's size in pixels.
    #[must_use]
    pub fn size(&self) -> Vec2 {
        Vec2::new(
            Fx::from_num(i32::try_from(self.rect.2).unwrap_or(i32::MAX)),
            Fx::from_num(i32::try_from(self.rect.3).unwrap_or(i32::MAX)),
        )
    }

    /// The region as normalised texture coordinates on its page.
    #[must_use]
    pub fn uv(&self, page: u32) -> Rect {
        let texel = |value: u32| Fx::from_num(i32::try_from(value).unwrap_or(i32::MAX));
        let scale = texel(page);
        Rect::new(
            Vec2::new(texel(self.rect.0) / scale, texel(self.rect.1) / scale),
            Vec2::new(texel(self.rect.2) / scale, texel(self.rect.3) / scale),
        )
    }
}

/// Packs generated sprites into fixed-size atlas pages.
///
/// A shelf packer rather than a general bin packer: sprites here are all small
/// and mostly the same handful of sizes, so shelves waste a few percent of a
/// page and cost a dozen lines instead of a few hundred.
#[derive(Debug)]
pub struct AtlasBuilder {
    page: u32,
    pages: Vec<Vec<u8>>,
    /// Top of the shelf currently being filled.
    shelf_y: u32,
    /// Next free column on that shelf.
    cursor_x: u32,
    /// The shelf's height, set by the tallest sprite on it.
    shelf_height: u32,
}

/// A sprite too large for an atlas page.
#[derive(Debug, PartialEq, Eq)]
pub struct TooLarge {
    /// The offending size.
    pub size: (u32, u32),
    /// The page size it did not fit.
    pub page: u32,
}

impl std::fmt::Display for TooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a {}x{} sprite does not fit a {}x{} atlas page",
            self.size.0, self.size.1, self.page, self.page
        )
    }
}

impl std::error::Error for TooLarge {}

impl AtlasBuilder {
    /// Starts an atlas of `page`-square pages.
    ///
    /// # Panics
    ///
    /// Panics if `page` is zero.
    #[must_use]
    pub fn new(page: u32) -> AtlasBuilder {
        assert!(page > 0, "an atlas page must have a non-zero size");
        AtlasBuilder {
            page,
            pages: vec![vec![0u8; (page * page * 4) as usize]],
            shelf_y: 0,
            cursor_x: 0,
            shelf_height: 0,
        }
    }

    /// The page size this atlas was built with.
    #[must_use]
    pub const fn page_size(&self) -> u32 {
        self.page
    }

    /// How many pages have been opened.
    #[must_use]
    pub fn page_count(&self) -> u32 {
        u32::try_from(self.pages.len()).unwrap_or(u32::MAX)
    }

    /// Copies `canvas` into the atlas and returns where it landed.
    ///
    /// # Errors
    ///
    /// Returns [`TooLarge`] when the canvas cannot fit a page even on its own.
    pub fn insert(&mut self, canvas: &Canvas, palette: &Palette) -> Result<Region, TooLarge> {
        let (width, height) = (canvas.width(), canvas.height());
        if width > self.page || height > self.page {
            return Err(TooLarge {
                size: (width, height),
                page: self.page,
            });
        }

        // A one-texel gutter around every sprite. Without it, the renderer's
        // nearest sampling can pick up a neighbour's edge texel wherever a
        // sprite is drawn at a size other than its own, which shows up as a
        // stray coloured line along one edge.
        let stride = width + 1;
        if self.cursor_x + stride > self.page {
            // Next shelf.
            self.shelf_y += self.shelf_height + 1;
            self.cursor_x = 0;
            self.shelf_height = 0;
        }
        if self.shelf_y + height + 1 > self.page {
            // Next page.
            self.pages
                .push(vec![0u8; (self.page * self.page * 4) as usize]);
            self.shelf_y = 0;
            self.cursor_x = 0;
            self.shelf_height = 0;
        }

        let layer = self.page_count() - 1;
        let (x, y) = (self.cursor_x, self.shelf_y);
        self.blit(layer, x, y, canvas, palette);

        self.cursor_x += stride;
        self.shelf_height = self.shelf_height.max(height);

        Ok(Region {
            layer,
            rect: (x, y, width, height),
        })
    }

    /// Copies one canvas into a page's byte buffer.
    fn blit(&mut self, layer: u32, x: u32, y: u32, canvas: &Canvas, palette: &Palette) {
        let rgba = canvas.to_rgba(palette);
        let page = self.page;
        let Some(buffer) = self.pages.get_mut(layer as usize) else {
            return;
        };
        for row in 0..canvas.height() {
            let source = (row * canvas.width() * 4) as usize;
            let destination = (((y + row) * page + x) * 4) as usize;
            let width = (canvas.width() * 4) as usize;
            buffer[destination..destination + width].copy_from_slice(&rgba[source..source + width]);
        }
    }

    /// The raw bytes of every page, in layer order.
    #[must_use]
    pub fn pages(&self) -> &[Vec<u8>] {
        &self.pages
    }

    /// Uploads the pages to a texture array.
    ///
    /// # Errors
    ///
    /// Returns the renderer's error if a layer will not accept its bytes,
    /// which can only happen if the page size and the array disagree.
    pub fn upload(
        &self,
        gpu: &verdant_render_2d::GpuContext,
    ) -> Result<TextureArray, verdant_render_2d::GpuError> {
        let array = TextureArray::new(gpu, self.page, self.page, self.page_count());
        for (index, bytes) in self.pages.iter().enumerate() {
            let layer = u32::try_from(index).unwrap_or(u32::MAX);
            array.write_layer(gpu, layer, bytes)?;
        }
        Ok(array)
    }
}

/// Every sprite the game draws, generated once at startup.
#[derive(Debug)]
pub struct Art {
    atlas: AtlasBuilder,
    /// Ground tiles, by tile id. Each has [`TERRAIN_VARIANTS`] alternatives so
    /// a field of grass does not read as one texture stamped in a grid.
    terrain: BTreeMap<u16, Vec<Region>>,
    /// Tilled soil, dry then watered.
    soil: [Region; 2],
    /// Crop growth stages, keyed by crop and stage.
    crops: BTreeMap<(u16, usize), Region>,
    /// Item icons.
    items: BTreeMap<u16, Region>,
    /// Trees, anchored at their base. Several shapes, so a wood is a wood
    /// rather than one tree repeated.
    trees: Vec<Region>,
    /// A rock.
    rock: Region,
    /// A vein of ore.
    ore: Region,
    /// A building's wall.
    wall: Region,
    /// A building's roof.
    roof: Region,
    /// A door.
    door: Region,
    /// A ladder down into the mine.
    ladder: Region,
    /// The player's walk cycle: four facings of [`WALK_FRAMES`] frames.
    player: Vec<Region>,
    /// One walk cycle per villager, in the game's villager order.
    villagers: Vec<Vec<Region>>,
    /// A single opaque white texel, for HUD panels and bars.
    white: Region,
    /// The font's glyphs, indexed by `ch as usize - 0x20`.
    glyphs: Vec<Region>,
}

/// The colours the valley is built from.
///
/// Gathered in one place because a palette is a design decision, not an
/// implementation detail scattered across a generator call site.
mod palette {
    use verdant_procgen_art::Rgba;

    /// Meadow grass.
    pub const GRASS: Rgba = Rgba::hex(0x6C_A9_4E);
    /// Bare earth around the farm.
    pub const SOIL: Rgba = Rgba::hex(0x93_74_4E);
    /// Trodden path and the village square.
    pub const PATH: Rgba = Rgba::hex(0xC0_A8_7C);
    /// River water.
    pub const WATER: Rgba = Rgba::hex(0x3E_7C_C4);
    /// Cliff stone.
    pub const STONE: Rgba = Rgba::hex(0x86_86_8E);
    /// The mine's floor.
    pub const CAVE_FLOOR: Rgba = Rgba::hex(0x4E_47_52);
    /// The mine's walls.
    pub const CAVE_WALL: Rgba = Rgba::hex(0x2C_28_31);
    /// Bridge planking.
    pub const BRIDGE: Rgba = Rgba::hex(0x9A_6A_3C);
    /// Freshly tilled earth.
    pub const TILLED: Rgba = Rgba::hex(0x6B_4E_33);
    /// Tilled earth after watering.
    pub const WATERED: Rgba = Rgba::hex(0x46_31_20);
    /// Tree foliage.
    pub const FOLIAGE: Rgba = Rgba::hex(0x35_71_38);
    /// Loose rock.
    pub const ROCK: Rgba = Rgba::hex(0x8E_8E_98);
    /// Exposed ore.
    pub const ORE: Rgba = Rgba::hex(0xC8_7A_3C);
    /// Building walls.
    pub const WALL: Rgba = Rgba::hex(0xC8_AE_86);
    /// Building roofs.
    pub const ROOF: Rgba = Rgba::hex(0xA4_45_3C);
    /// Doors and ladders.
    pub const TIMBER: Rgba = Rgba::hex(0x7A_52_2E);
}

impl Art {
    /// Generates every sprite for a world.
    ///
    /// `villager_seeds` supplies one appearance seed per villager, in the same
    /// order the game holds them, so a villager's look is tied to the world
    /// seed rather than to the order they happened to be drawn in.
    ///
    /// # Panics
    ///
    /// Panics if a generated sprite is too large for an atlas page, which is a
    /// programming error in this module rather than a runtime condition.
    #[must_use]
    pub fn generate(seed: u64, villager_seeds: &[u64]) -> Art {
        let mut atlas = AtlasBuilder::new(ATLAS_PAGE);
        let tile = u32::try_from(TILE_SIZE.to_int()).unwrap_or(16);

        let mut pack = |canvas: &Canvas, palette: &Palette| -> Region {
            atlas
                .insert(canvas, palette)
                .expect("generated sprites fit an atlas page")
        };

        // --- Ground -------------------------------------------------------
        let mut terrain = BTreeMap::new();
        for (id, base, roughness) in [
            (tiles::GRASS, palette::GRASS, 46),
            (tiles::SOIL, palette::SOIL, 34),
            (tiles::PATH, palette::PATH, 28),
            (tiles::WATER, palette::WATER, 30),
            (tiles::STONE, palette::STONE, 40),
            (tiles::CAVE_FLOOR, palette::CAVE_FLOOR, 44),
            (tiles::CAVE_WALL, palette::CAVE_WALL, 36),
            (tiles::BRIDGE, palette::BRIDGE, 22),
        ] {
            let variants = (0..TERRAIN_VARIANTS)
                .map(|variant| {
                    let tile_seed = seed
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(u64::from(id.0))
                        .wrapping_add(u64::try_from(variant).unwrap_or(0).wrapping_mul(7_907));
                    let generated = generate_terrain_tile(tile_seed, tile, base, roughness);
                    let mut canvas = generated.frames[0].clone();
                    // The first variant of each tile stays plain, so a large
                    // area still reads as one surface; the rest carry detail.
                    let mut tile_palette = generated.palette.clone();
                    if variant > 0 {
                        scatter_detail(
                            &mut canvas,
                            &mut tile_palette,
                            id,
                            variant,
                            &mut Rng::new(tile_seed),
                        );
                    }
                    pack(&canvas, &tile_palette)
                })
                .collect();
            terrain.insert(id.0, variants);
        }

        // --- Worked soil --------------------------------------------------
        let soil = [palette::TILLED, palette::WATERED].map(|base| {
            let generated = generate_terrain_tile(seed ^ u64::from(base.r), tile, base, 20);
            let mut canvas = generated.frames[0].clone();
            furrow(&mut canvas, &generated.palette);
            pack(&canvas, &generated.palette)
        });

        // --- Crops --------------------------------------------------------
        let mut crops = BTreeMap::new();
        for definition in CROPS {
            let generated = generate_crop(
                seed.wrapping_add(u64::from(definition.id as u16).wrapping_mul(7919)),
                tile,
                definition.stage_count(),
                definition.leaf_tint,
                definition.fruit_tint,
            );
            for (stage, frame) in generated.frames.iter().enumerate() {
                let region = pack(frame, &generated.palette);
                crops.insert((definition.id as u16, stage), region);
            }
        }

        // --- Items --------------------------------------------------------
        let mut items = BTreeMap::new();
        for definition in ITEMS {
            let generated = generate_item_icon(
                seed.wrapping_add(u64::from(definition.id as u16).wrapping_mul(104_729)),
                tile,
                definition.tint,
            );
            items.insert(
                definition.id as u16,
                pack(&generated.frames[0], &generated.palette),
            );
        }

        // --- Scenery ------------------------------------------------------
        let trees = (0..TREE_VARIANTS)
            .map(|variant| {
                let index = u64::try_from(variant).unwrap_or(0);
                // Alternating foliage tones and slightly different heights:
                // the two cheapest changes that stop a wood looking stamped.
                let foliage = match variant % 3 {
                    0 => palette::FOLIAGE,
                    1 => palette::FOLIAGE.tint(28),
                    _ => palette::FOLIAGE.shade(210),
                };
                let height = tile * 3 - u32::try_from(variant % 2).unwrap_or(0) * 4;
                let sprite = generate_tree(
                    seed.wrapping_add(11)
                        .wrapping_add(index.wrapping_mul(5_147)),
                    tile * 2,
                    height,
                    foliage,
                );
                pack(&sprite.frames[0], &sprite.palette)
            })
            .collect();

        let rock_sprite = generate_item_icon(seed.wrapping_add(13), tile, palette::ROCK);
        let rock = pack(&rock_sprite.frames[0], &rock_sprite.palette);

        let ore_sprite = generate_item_icon(seed.wrapping_add(17), tile, palette::ORE);
        let ore = pack(&ore_sprite.frames[0], &ore_sprite.palette);

        // --- Buildings ----------------------------------------------------
        let (wall_canvas, wall_palette) = wall_tile(tile, palette::WALL);
        let wall = pack(&wall_canvas, &wall_palette);
        let (roof_canvas, roof_palette) = roof_tile(tile, palette::ROOF);
        let roof = pack(&roof_canvas, &roof_palette);
        let (door_canvas, door_palette) = door_tile(tile, palette::TIMBER);
        let door = pack(&door_canvas, &door_palette);
        let (ladder_canvas, ladder_palette) = ladder_tile(tile, palette::TIMBER);
        let ladder = pack(&ladder_canvas, &ladder_palette);

        // --- Characters ---------------------------------------------------
        let character_height = tile + tile / 2;
        let player_sprite = generate_character(
            seed.wrapping_add(1),
            tile,
            character_height,
            CharacterStyle {
                // The player is deliberately the one character not randomised,
                // so they are recognisable in a crowd of villagers.
                skin: Rgba::hex(0xE8_B8_92),
                hair: Rgba::hex(0x53_34_22),
                shirt: Rgba::hex(0x3C_7A_C8),
                trousers: Rgba::hex(0x36_3E_5A),
                boots: Rgba::hex(0x4A_33_22),
            },
        );
        let player = player_sprite
            .frames
            .iter()
            .map(|frame| pack(frame, &player_sprite.palette))
            .collect();

        let villagers = villager_seeds
            .iter()
            .map(|villager_seed| {
                let mut rng = Rng::new(*villager_seed);
                let sprite = generate_character(
                    *villager_seed,
                    tile,
                    character_height,
                    CharacterStyle::random(&mut rng),
                );
                sprite
                    .frames
                    .iter()
                    .map(|frame| pack(frame, &sprite.palette))
                    .collect()
            })
            .collect();

        // --- HUD ----------------------------------------------------------
        let mut white_palette = Palette::new("white", &[]);
        let white_index = white_palette.push(Rgba::hex(0xFF_FF_FF));
        let mut white_canvas = Canvas::new(1, 1);
        white_canvas.set(0, 0, white_index);
        let white = pack(&white_canvas, &white_palette);

        let mut ink = Palette::new("ink", &[]);
        let ink_index = ink.push(Rgba::hex(0xFF_FF_FF));
        let glyphs = (0x20u32..0x7F)
            .map(|code| {
                let ch = char::from_u32(code).unwrap_or('?');
                let canvas = font::render_text(&ch.to_string(), ink_index);
                pack(&canvas, &ink)
            })
            .collect();

        Art {
            atlas,
            terrain,
            soil,
            crops,
            items,
            trees,
            rock,
            ore,
            wall,
            roof,
            door,
            ladder,
            player,
            villagers,
            white,
            glyphs,
        }
    }

    /// The packed atlas, ready to upload.
    #[must_use]
    pub const fn atlas(&self) -> &AtlasBuilder {
        &self.atlas
    }

    /// The ground sprite for a tile at a cell.
    ///
    /// The variant is chosen from the cell's coordinates rather than at
    /// random, so the same patch of ground looks the same every time it comes
    /// back into view.
    #[must_use]
    pub fn terrain(&self, tile: TileId, cell: IVec2) -> Option<Region> {
        let variants = self.terrain.get(&tile.0)?;
        variants.get(variant_index(cell, variants.len())).copied()
    }

    /// The tree sprite for a cell.
    #[must_use]
    pub fn tree(&self, cell: IVec2) -> Region {
        self.trees[variant_index(cell, self.trees.len())]
    }

    /// The sprite for a crop at a growth stage.
    ///
    /// Stages past the last one clamp to the mature sprite, so a regrowing
    /// crop never falls back to nothing.
    #[must_use]
    pub fn crop(&self, id: CropId, stage: usize) -> Option<Region> {
        let last = crop(id).stage_count().saturating_sub(1);
        self.crops.get(&(id as u16, stage.min(last))).copied()
    }

    /// An item's icon.
    #[must_use]
    pub fn item(&self, id: ItemId) -> Option<Region> {
        self.items.get(&(id as u16)).copied()
    }

    /// A character frame for the player.
    #[must_use]
    pub fn player_frame(&self, facing: FacingState, frame: usize) -> Region {
        let index = facing.to_art().index() * WALK_FRAMES + frame % WALK_FRAMES;
        self.player[index.min(self.player.len() - 1)]
    }

    /// A character frame for a villager.
    #[must_use]
    pub fn villager_frame(&self, villager: usize, facing: FacingState, frame: usize) -> Region {
        let cycle = self.villagers.get(villager).unwrap_or(&self.player);
        let index = facing.to_art().index() * WALK_FRAMES + frame % WALK_FRAMES;
        cycle[index.min(cycle.len() - 1)]
    }

    /// The single white texel used for HUD fills.
    #[must_use]
    pub const fn white(&self) -> Region {
        self.white
    }

    /// A glyph's region, falling back to `?` outside printable ASCII.
    #[must_use]
    pub fn glyph(&self, ch: char) -> Region {
        let code = ch as u32;
        let index = if (0x20..0x7F).contains(&code) {
            usize::try_from(code - 0x20).unwrap_or(0)
        } else {
            usize::try_from(u32::from(b'?') - 0x20).unwrap_or(0)
        };
        self.glyphs[index.min(self.glyphs.len() - 1)]
    }
}

/// Cuts horizontal furrows into a soil tile so tilled ground reads as worked.
fn furrow(canvas: &mut Canvas, palette: &Palette) {
    // A mid-ramp shade rather than the darkest: full-strength furrows on
    // already-dark soil turn a worked plot into a black hole on the map.
    let dark = PaletteIndex(u8::try_from(palette.len().saturating_sub(2)).unwrap_or(1));
    let height = i32::try_from(canvas.height()).unwrap_or(16);
    let width = i32::try_from(canvas.width()).unwrap_or(16);
    let mut row = 3;
    while row < height {
        for x in 0..width {
            canvas.set(x, row, dark);
        }
        row += 4;
    }
}

/// A plastered wall with a timber frame.
fn wall_tile(size: u32, base: Rgba) -> (Canvas, Palette) {
    let mut palette = Palette::new("wall", &[]);
    let shades = palette.add_ramp(base, 3);
    let beam = palette.push(Rgba::hex(0x6B_47_2A));

    let mut canvas = Canvas::new(size, size);
    let extent = i32::try_from(size).unwrap_or(16);
    canvas.fill_rect(0, 0, extent, extent, shades[0]);
    // A vertical beam on one edge and a horizontal one across the middle: two
    // strokes are enough for a wall of these tiles to read as timber framing
    // rather than as a flat block of colour.
    canvas.fill_rect(0, 0, 1, extent, beam);
    canvas.fill_rect(0, extent / 2 - 1, extent, 2, beam);
    // A little shading under the beam gives the plaster depth.
    canvas.fill_rect(0, extent / 2 + 1, extent, 1, shades[shades.len() - 1]);
    (canvas, palette)
}

/// A tiled roof with an overhanging eave.
fn roof_tile(size: u32, base: Rgba) -> (Canvas, Palette) {
    let mut palette = Palette::new("roof", &[]);
    let shades = palette.add_ramp(base, 3);
    let shadow = palette.push(Rgba::hex(0x35_20_1E));

    let mut canvas = Canvas::new(size, size);
    let extent = i32::try_from(size).unwrap_or(16);
    canvas.fill_rect(0, 0, extent, extent, shades[0]);
    // Staggered courses of tiles.
    let mut row = 2;
    let mut offset = 0;
    while row < extent {
        for x in (offset..extent).step_by(4) {
            canvas.fill_rect(x, row, 1, 2, shades[shades.len() - 1]);
        }
        canvas.fill_rect(0, row, extent, 1, shades[1.min(shades.len() - 1)]);
        row += 4;
        offset = if offset == 0 { 2 } else { 0 };
    }
    // The eave: a dark line along the bottom edge, which is what separates a
    // roof from the wall below it at a glance.
    canvas.fill_rect(0, extent - 1, extent, 1, shadow);
    (canvas, palette)
}

/// A planked door with a handle.
fn door_tile(size: u32, base: Rgba) -> (Canvas, Palette) {
    let mut palette = Palette::new("door", &[]);
    let shades = palette.add_ramp(base, 3);
    let frame = palette.push(Rgba::hex(0x3A2616));
    let handle = palette.push(Rgba::hex(0xE0_C8_60));

    let mut canvas = Canvas::new(size, size);
    let extent = i32::try_from(size).unwrap_or(16);
    canvas.fill_rect(0, 0, extent, extent, frame);
    canvas.fill_rect(2, 2, extent - 4, extent - 2, shades[0]);
    // Plank seams.
    for x in (4..extent - 4).step_by(4) {
        canvas.fill_rect(x, 2, 1, extent - 2, shades[shades.len() - 1]);
    }
    canvas.set(extent - 5, extent / 2, handle);
    (canvas, palette)
}

/// A ladder leading down.
fn ladder_tile(size: u32, base: Rgba) -> (Canvas, Palette) {
    let mut palette = Palette::new("ladder", &[]);
    let shades = palette.add_ramp(base, 2);
    let dark = palette.push(Rgba::hex(0x14_10_18));

    let mut canvas = Canvas::new(size, size);
    let extent = i32::try_from(size).unwrap_or(16);
    // A dark opening the ladder descends into.
    canvas.fill_rect(0, 0, extent, extent, dark);
    // Two rails and the rungs between them.
    canvas.fill_rect(3, 0, 2, extent, shades[0]);
    canvas.fill_rect(extent - 5, 0, 2, extent, shades[0]);
    let mut rung = 2;
    while rung < extent {
        canvas.fill_rect(3, rung, extent - 6, 1, shades[shades.len() - 1]);
        rung += 4;
    }
    (canvas, palette)
}

/// Radius of the on-screen thumbstick, in internal pixels.
///
/// Generous for a 480-pixel-wide frame: on a phone held in two hands the
/// stick sits under a thumb that cannot see it, so it has to be findable by
/// feel rather than by aim.
pub const STICK_RADIUS: i32 = 30;

/// Edge length of an on-screen action button.
pub const TOUCH_BUTTON_SIZE: i32 = 44;

/// The on-screen control layout for the game's internal resolution.
///
/// The single source of truth for both hit testing and drawing. A layout
/// function rather than two constants because a button whose artwork and hit
/// region disagree is a button the player presses and nothing happens.
#[must_use]
pub fn touch_layout() -> TouchLayout {
    let width = i32::try_from(INTERNAL_WIDTH).unwrap_or(480);
    let height = i32::try_from(INTERNAL_HEIGHT).unwrap_or(270);

    // Bottom-left for the stick, bottom-right for the buttons: the standard
    // arrangement, and the one a player's hands already expect.
    let stick_centre = Vec2::from_ints(STICK_RADIUS + 22, height - STICK_RADIUS - 22);
    let size = TOUCH_BUTTON_SIZE;
    let margin = 12;

    // The primary action sits lowest and furthest right, under the thumb's
    // resting position; the rest fan up and left from it.
    let use_button = Rect::from_ints(width - size - margin, height - size - margin, size, size);
    let interact_button = Rect::from_ints(
        width - size * 2 - margin - 8,
        height - size - margin - 18,
        size,
        size,
    );
    let sprint_button = Rect::from_ints(
        width - size - margin - 6,
        height - size * 2 - margin - 20,
        size,
        size,
    );

    TouchLayout::new(Thumbstick::new(stick_centre, Fx::from_num(STICK_RADIUS)))
        .with_button(TouchButton::new(actions::USE, use_button))
        .with_button(TouchButton::new(actions::INTERACT, interact_button))
        .with_button(TouchButton::new(actions::SPRINT, sprint_button))
}

/// The label drawn on a touch button.
///
/// Short words rather than icons: a generated icon at this size is a smudge,
/// and a word is unambiguous.
#[must_use]
pub fn touch_button_label(action: verdant_input::Action) -> &'static str {
    match action {
        a if a == actions::USE => "USE",
        a if a == actions::INTERACT => "TALK",
        a if a == actions::SPRINT => "RUN",
        other => other.name(),
    }
}

/// A rectangle in HUD pixel space.
///
/// A named type rather than four loose integers: `fill(x, y, width, height)`
/// and `fill(left, top, right, bottom)` are indistinguishable at a call site,
/// and getting them the wrong way round is silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Panel {
    /// Distance from the left edge of the frame.
    x: i32,
    /// Distance from the top edge of the frame.
    y: i32,
    /// Width in pixels.
    width: i32,
    /// Height in pixels.
    height: i32,
}

impl Panel {
    /// A panel at a position with a size.
    const fn new(x: i32, y: i32, width: i32, height: i32) -> Panel {
        Panel {
            x,
            y,
            width,
            height,
        }
    }
}

/// Builds the frame's sprite list from game state.
///
/// Holds the batcher between frames so its allocation is reused rather than
/// regrown sixty times a second.
#[derive(Debug)]
pub struct Scene {
    /// The world, which the lighting pass acts on.
    batcher: SpriteBatcher,
    /// The interface, drawn after lighting and therefore never dimmed by it.
    ///
    /// A separate batch rather than a compensating tint: the HUD is not part
    /// of the world, and the honest way to keep it readable at midnight is to
    /// draw it outside the lighting rather than to undo the lighting on it.
    hud: SpriteBatcher,
    /// Lights gathered from the world this frame.
    lights: Vec<Light>,
    /// The camera, kept across frames so its smoothing has history.
    pub camera: Camera2D,
    /// The on-screen control layout, drawn when `show_touch_controls` is set.
    pub touch: TouchLayout,
    /// Whether to draw the on-screen controls.
    ///
    /// Off by default and switched on by the Android entry point, so a
    /// desktop player never sees a thumbstick they cannot use, and a phone
    /// never lacks one.
    pub show_touch_controls: bool,
    /// Where the stick's thumb marker should be drawn.
    touch_thumb: Option<Vec2>,
    /// Which touch buttons are currently held, for their pressed look.
    touch_held: Vec<verdant_input::Action>,
}

impl Default for Scene {
    fn default() -> Scene {
        Scene::new()
    }
}

impl Scene {
    /// Creates a scene builder at the game's internal resolution.
    #[must_use]
    pub fn new() -> Scene {
        Scene {
            batcher: SpriteBatcher::new(),
            hud: SpriteBatcher::new(),
            lights: Vec::new(),
            camera: Camera2D::new(INTERNAL_WIDTH, INTERNAL_HEIGHT),
            touch: touch_layout(),
            show_touch_controls: cfg!(target_os = "android"),
            touch_thumb: None,
            touch_held: Vec::new(),
        }
    }

    /// Moves the camera toward the player and keeps it inside the map.
    ///
    /// `dt` is real elapsed time rather than simulation time: the camera is
    /// presentation, and smoothing it on the fixed step would make it stutter
    /// whenever a frame and a tick disagree.
    pub fn follow(&mut self, game: &Game, dt: Fx) {
        // A short half-life: long enough to take the edge off direction
        // changes, short enough that the player never feels dragged.
        self.camera
            .follow(game.player.position, Fx::from_ratio(1, 12), dt);
        self.camera.clamp_to(game.map.bounds());
    }

    /// Places the camera on the player immediately, with no smoothing.
    ///
    /// Used when a scene starts or the player changes level, where easing in
    /// from the previous position would read as the camera flying across the
    /// map.
    pub fn snap_to(&mut self, game: &Game) {
        self.camera.position = game.player.position;
        self.camera.clamp_to(game.map.bounds());
    }

    /// Builds this frame and returns the sorted sprites.
    pub fn build(&mut self, game: &Game, art: &Art) -> &mut SpriteBatcher {
        self.batcher.clear();
        self.hud.clear();
        self.collect_lights(game);
        self.draw_world(game, art);
        self.draw_farm(game, art);
        self.draw_characters(game, art);
        self.draw_hud(game, art);
        &mut self.batcher
    }

    /// The world sprites built by the last call to [`Scene::build`].
    #[must_use]
    pub fn sprites(&self) -> &[DrawSprite] {
        self.batcher.sprites()
    }

    /// The interface sprites built by the last call to [`Scene::build`].
    #[must_use]
    pub fn hud_sprites(&self) -> &[DrawSprite] {
        self.hud.sprites()
    }

    /// The lights gathered by the last call to [`Scene::build`].
    #[must_use]
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    /// The interface batch, for the pass that draws it after lighting.
    pub fn hud_batch(&mut self) -> &mut SpriteBatcher {
        &mut self.hud
    }

    /// Gathers the lights the world is casting right now.
    ///
    /// Nothing is lit in daylight: the ambient term already covers the scene,
    /// and a lantern that glows at noon reads as a bug rather than as detail.
    /// Their strength rises as the ambient light falls, so lights fade in
    /// through dusk instead of switching on.
    fn collect_lights(&mut self, game: &Game) {
        self.lights.clear();

        let ambient = game.calendar.ambient_light();
        // How dark it is, from the brightest channel: a warm dusk still has a
        // strong red, and darkness should follow the brightest light present
        // rather than an average that dusk would exaggerate.
        let brightest = ambient.r.max(ambient.g).max(ambient.b);
        let darkness = (1.0 - brightest).clamp(0.0, 1.0);
        if darkness <= 0.02 {
            return;
        }

        let lantern = Color::from_srgb_hex(0xFF_C8_6E);
        let window = Color::from_srgb_hex(0xFF_D8_8A);

        // Windows in the village, and the farmhouse.
        let mut lit_buildings: Vec<IVec2> = game.layout.homes.clone();
        lit_buildings.push(game.layout.shop_door);
        lit_buildings.push(game.layout.farmhouse_door);
        for cell in lit_buildings {
            self.lights.push(
                Light::point(tile_anchor(cell), TILE_SIZE * Fx::from_num(4), window)
                    .with_intensity(darkness * 0.85),
            );
        }

        // The player carries a lantern, so walking home in the dark works.
        self.lights.push(
            Light::point(game.player.position, TILE_SIZE * Fx::from_num(5), lantern)
                .with_intensity(darkness),
        );
    }

    /// Queues one sprite for an atlas region at a world position.
    fn push(&mut self, art: &Art, region: Region, position: Vec2, z: i32, tint: Color) {
        let page = art.atlas.page_size();
        self.batcher.push(
            DrawSprite::new(position, region.size(), region.layer)
                .with_uv(region.uv(page))
                .at_z(z)
                .tinted(tint),
        );
    }

    /// Ground tiles and the scenery standing on them.
    fn draw_world(&mut self, game: &Game, art: &Art) {
        let Some(ground) = game.ground() else {
            return;
        };
        let visible = self.camera.visible_bounds();
        // One tile of margin so a sprite taller than its cell — a tree — is
        // still drawn when its base is just off screen.
        let margin = 3;
        let min = game.map.cell_at(visible.min) - IVec2::new(margin, margin);
        let max = game.map.cell_at(visible.max()) + IVec2::new(margin, margin);

        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let cell = IVec2::new(x, y);
                if !ground.contains(cell) {
                    continue;
                }
                let tile = ground.get(cell);
                let base = tile_ground(tile);
                let position = tile_anchor(cell);

                if let Some(region) = art.terrain(base, cell) {
                    self.push(art, region, position, Z_GROUND, Color::WHITE);
                }

                match tile {
                    tiles::TREE => {
                        self.push(art, art.tree(cell), position, Z_STANDING, Color::WHITE);
                    }
                    tiles::ROCK => self.push(art, art.rock, position, Z_STANDING, Color::WHITE),
                    tiles::ORE => self.push(art, art.ore, position, Z_STANDING, Color::WHITE),
                    tiles::DOOR => self.push(art, art.door, position, Z_STANDING, Color::WHITE),
                    tiles::LADDER => self.push(art, art.ladder, position, Z_OVERLAY, Color::WHITE),
                    tiles::BUILDING => {
                        // A building tile with open sky above it is the roof
                        // line; the rest is wall. Two sprites and one lookup
                        // give the village silhouette real depth.
                        let above = ground.get(cell - IVec2::new(0, 1));
                        let region = if above == tiles::BUILDING || above == tiles::DOOR {
                            art.wall
                        } else {
                            art.roof
                        };
                        self.push(art, region, position, Z_STANDING, Color::WHITE);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Tilled plots and whatever is growing in them.
    fn draw_farm(&mut self, game: &Game, art: &Art) {
        let visible = self.camera.visible_bounds();
        for (cell, plot) in game.farm.iter() {
            let position = tile_anchor(cell);
            // Plots are cheap to skip and there can be hundreds of them.
            if !visible.contains_point(position) {
                continue;
            }
            let soil = art.soil[usize::from(plot.watered)];
            self.push(art, soil, position, Z_OVERLAY, Color::WHITE);
            self.draw_planting(art, plot, position);
        }
    }

    /// The crop in a plot, if any.
    fn draw_planting(&mut self, art: &Art, plot: &Plot, position: Vec2) {
        let Some(planting) = plot.planting.as_ref() else {
            return;
        };
        let Some(stage) = plot.growth_stage() else {
            return;
        };
        if let Some(region) = art.crop(planting.crop, stage) {
            self.push(art, region, position, Z_STANDING, Color::WHITE);
        }
    }

    /// The player and the villagers.
    fn draw_characters(&mut self, game: &Game, art: &Art) {
        let visible = self.camera.visible_bounds();

        for (index, villager) in game.villagers.iter().enumerate() {
            if !visible.contains_point(villager.position) {
                continue;
            }
            let region = art.villager_frame(index, villager.facing, walk_frame(villager.position));
            self.push(art, region, villager.position, Z_STANDING, Color::WHITE);
        }

        let region = art.player_frame(game.player.facing, game.player.animation_frame());
        self.push(art, region, game.player.position, Z_STANDING, Color::WHITE);

        // The tile the current tool would act on, so the player can see their
        // reach instead of guessing at it.
        let target = tile_anchor(game.player.target_tile());
        self.push(
            art,
            art.white,
            target,
            Z_OVERLAY,
            Color::WHITE.with_alpha(0.18),
        );
    }

    /// Draws a filled rectangle in screen space, for HUD panels and bars.
    fn fill(&mut self, art: &Art, panel: Panel, color: Color, z: i32) {
        let Panel {
            x,
            y,
            width,
            height,
        } = panel;
        let page = art.atlas.page_size();
        let position = self.screen_to_world(Vec2::from_ints(x, y + height));
        self.hud.push(
            DrawSprite::new(position, Vec2::from_ints(width, height), art.white.layer)
                .with_uv(art.white.uv(page))
                .anchored(Vec2::new(Fx::ZERO, Fx::ONE))
                .at_z(z)
                .tinted(color),
        );
    }

    /// Draws a string in screen space, one sprite per glyph.
    fn text(&mut self, art: &Art, x: i32, y: i32, string: &str, color: Color, z: i32) {
        let page = art.atlas.page_size();
        let advance = i32::try_from(font::GLYPH_ADVANCE).unwrap_or(6);
        let height = i32::try_from(font::GLYPH_HEIGHT).unwrap_or(7);
        for (index, ch) in string.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            let region = art.glyph(ch);
            let offset = x + i32::try_from(index).unwrap_or(0) * advance;
            let position = self.screen_to_world(Vec2::from_ints(offset, y + height));
            self.hud.push(
                DrawSprite::new(position, region.size(), region.layer)
                    .with_uv(region.uv(page))
                    .anchored(Vec2::new(Fx::ZERO, Fx::ONE))
                    .at_z(z)
                    .tinted(color),
            );
        }
    }

    /// Draws a string with a one-pixel drop shadow.
    ///
    /// Pixel text over a detailed background is unreadable without one; the
    /// shadow is what lets the HUD sit on grass, water or stone unchanged.
    fn shadowed_text(&mut self, art: &Art, x: i32, y: i32, string: &str, color: Color, z: i32) {
        self.text(art, x + 1, y + 1, string, Color::BLACK.with_alpha(0.65), z);
        self.text(art, x, y, string, color, z);
    }

    /// Converts a HUD pixel coordinate into the world position that lands on
    /// it.
    ///
    /// The HUD shares the world's camera rather than a second orthographic
    /// pass, which keeps the whole frame in one draw call.
    fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        self.camera.screen_to_world(screen)
    }

    /// The status bars, the clock and the hotbar.
    fn draw_hud(&mut self, game: &Game, art: &Art) {
        let width = i32::try_from(INTERNAL_WIDTH).unwrap_or(480);
        let height = i32::try_from(INTERNAL_HEIGHT).unwrap_or(270);

        self.draw_clock(game, art, width);
        self.draw_energy(game, art, width, height);
        self.draw_hotbar(game, art, width, height);
        self.draw_message(game, art, height);
        if self.show_touch_controls {
            self.draw_touch_controls(art);
        }
    }

    /// The on-screen thumbstick and action buttons.
    ///
    /// Drawn from the same [`TouchLayout`] that resolves presses, so what the
    /// player sees is exactly what they can hit.
    fn draw_touch_controls(&mut self, art: &Art) {
        // These sit over a world that can be any colour, so each control
        // paints its own dark ground first and its bright parts on top.
        // Without that the stick reads as a few pale specks in the grass —
        // which is exactly what an earlier version of this did.
        let ground = Color::from_srgb_hex(0x14_10_1C).with_alpha(0.38);
        let ring = Color::from_srgb_hex(0xF4_EC_DC).with_alpha(0.72);
        let thumb = Color::from_srgb_hex(0xFF_FF_FF).with_alpha(0.85);
        let face = Color::from_srgb_hex(0x14_10_1C).with_alpha(0.46);
        let edge = Color::from_srgb_hex(0xF4_EC_DC).with_alpha(0.55);
        let held_face = Color::from_srgb_hex(0xF0_D8_A0).with_alpha(0.72);
        let label = Color::from_srgb_hex(0xF8_F0_DC);

        let centre = self.touch.stick.centre;
        let radius = self.touch.stick.radius.to_int();

        // The stick's ground, then its rim.
        self.fill_disc(art, centre, radius, ground, Z_TOUCH);
        self.stroke_circle(art, centre, radius, ring, Z_TOUCH);

        // The thumb, at the finger when one is down and at rest otherwise,
        // kept inside the rim so it cannot wander off across the screen when
        // a thumb drags well past the edge.
        let thumb_at = self.touch_thumb.unwrap_or(centre);
        let offset = thumb_at - centre;
        let clamped = if offset.length() > self.touch.stick.radius {
            centre + offset.clamp_length(self.touch.stick.radius)
        } else {
            thumb_at
        };
        self.fill_disc(art, clamped, (radius / 3).max(5), thumb, Z_TOUCH);

        // The action buttons.
        for index in 0..self.touch.buttons.len() {
            let button = self.touch.buttons[index];
            let held = self.touch_held.contains(&button.action);
            let bounds = Panel::new(
                button.bounds.min.x.to_int(),
                button.bounds.min.y.to_int(),
                button.bounds.size.x.to_int(),
                button.bounds.size.y.to_int(),
            );

            // A pressed button lights up: on a touchscreen there is no travel
            // to feel, so the only confirmation a player gets is visual.
            self.fill(art, bounds, if held { held_face } else { face }, Z_TOUCH);
            self.stroke_rect(art, bounds, edge, Z_TOUCH);

            let text = touch_button_label(button.action);
            let text_width = i32::try_from(font::text_width(text)).unwrap_or(0);
            let glyph_height = i32::try_from(font::GLYPH_HEIGHT).unwrap_or(7);
            let colour = if held {
                Color::from_srgb_hex(0x2A_20_18)
            } else {
                label
            };
            self.shadowed_text(
                art,
                bounds.x + (bounds.width - text_width) / 2,
                bounds.y + (bounds.height - glyph_height) / 2,
                text,
                colour,
                Z_TOUCH,
            );
        }
    }

    /// Fills a disc in screen space, as one horizontal span per row.
    ///
    /// Spans rather than a circle of blocks: a ring of separate squares
    /// leaves gaps that read as noise against a textured background, and a
    /// solid shape is what makes a control look like a control.
    fn fill_disc(&mut self, art: &Art, centre: Vec2, radius: i32, color: Color, z: i32) {
        if radius <= 0 {
            return;
        }
        let (cx, cy) = (centre.x.to_int(), centre.y.to_int());
        for row in -radius..=radius {
            // The engine's fixed-point square root rather than a float one,
            // so the shape is identical on every machine.
            let half = Fx::from_num((radius * radius - row * row).max(0))
                .sqrt()
                .to_int();
            if half <= 0 {
                continue;
            }
            self.fill(art, Panel::new(cx - half, cy + row, half * 2, 1), color, z);
        }
    }

    /// Draws a one-pixel circular outline in screen space.
    fn stroke_circle(&mut self, art: &Art, centre: Vec2, radius: i32, color: Color, z: i32) {
        if radius <= 0 {
            return;
        }
        // Enough segments that consecutive marks touch: roughly the
        // circumference in pixels, so each step advances about one pixel and
        // the outline has no gaps for the background to show through.
        let scale = Fx::from_num(radius);
        let segments = (radius * 7).max(24);
        for step in 0..segments {
            let angle = Fx::TAU * Fx::from_ratio(step, segments);
            let x = centre.x + angle.cos() * scale;
            let y = centre.y + angle.sin() * scale;
            self.fill(art, Panel::new(x.to_int(), y.to_int(), 2, 2), color, z);
        }
    }

    /// Draws a one-pixel rectangular outline in screen space.
    fn stroke_rect(&mut self, art: &Art, panel: Panel, color: Color, z: i32) {
        let Panel {
            x,
            y,
            width,
            height,
        } = panel;
        self.fill(art, Panel::new(x, y, width, 1), color, z);
        self.fill(art, Panel::new(x, y + height - 1, width, 1), color, z);
        self.fill(art, Panel::new(x, y, 1, height), color, z);
        self.fill(art, Panel::new(x + width - 1, y, 1, height), color, z);
    }

    /// Records where the player's fingers are, for the next frame's controls.
    ///
    /// Presentation only: the simulation already has the resolved direction,
    /// and this is what makes the drawn stick follow the thumb.
    pub fn observe_touch(&mut self, touch: &TouchState) {
        self.touch_thumb = touch.stick_position();
        self.touch_held.clear();
        self.touch_held.extend_from_slice(touch.pressed());
    }

    /// The date, time and weather panel.
    fn draw_clock(&mut self, game: &Game, art: &Art, width: i32) {
        let panel_width = 116;
        let x = width - panel_width - 6;
        self.fill(
            art,
            Panel::new(x, 6, panel_width, 34),
            Color::from_srgb_hex(0x1E_18_24).with_alpha(0.72),
            Z_HUD_PANEL,
        );

        let (hour, minute) = game.calendar.clock();
        let (display_hour, suffix) = twelve_hour(hour);
        self.shadowed_text(
            art,
            x + 6,
            11,
            &format!(
                "{} {}",
                game.calendar.season().name(),
                game.calendar.day_of_season() + 1
            ),
            Color::from_srgb_hex(0xF4_E8_C8),
            Z_HUD_TEXT,
        );
        self.shadowed_text(
            art,
            x + 6,
            21,
            &format!(
                "{display_hour}:{minute:02}{suffix}  {}",
                game.calendar.weather.name()
            ),
            Color::from_srgb_hex(0xC8_D8_F0),
            Z_HUD_TEXT,
        );
        self.shadowed_text(
            art,
            x + 6,
            31,
            &format!("{}g", game.player.inventory.gold),
            Color::from_srgb_hex(0xF0_D0_60),
            Z_HUD_TEXT,
        );
    }

    /// The energy bar, drawn as a vertical gauge on the right.
    fn draw_energy(&mut self, game: &Game, art: &Art, width: i32, height: i32) {
        let (bar_width, bar_height) = (10, 92);
        let x = width - bar_width - 8;
        let y = height - bar_height - 40;

        self.fill(
            art,
            Panel::new(x - 2, y - 2, bar_width + 4, bar_height + 4),
            Color::from_srgb_hex(0x1E_18_24).with_alpha(0.78),
            Z_HUD_PANEL,
        );

        let filled = (bar_height * game.player.energy.clamp(0, MAX_ENERGY)) / MAX_ENERGY;
        // Green while rested, amber when tiring, red when nearly spent: the
        // colour is the warning, since the player is watching the world rather
        // than the number.
        let colour = match game.player.energy * 100 / MAX_ENERGY {
            0..=20 => Color::from_srgb_hex(0xD8_4C_3C),
            21..=50 => Color::from_srgb_hex(0xE0_A0_38),
            _ => Color::from_srgb_hex(0x6C_C0_50),
        };
        self.fill(
            art,
            Panel::new(x, y + (bar_height - filled), bar_width, filled),
            colour,
            Z_HUD_TEXT,
        );
    }

    /// The hotbar along the bottom of the screen.
    fn draw_hotbar(&mut self, game: &Game, art: &Art, width: i32, height: i32) {
        let slot = 22;
        let gap = 2;
        let slots = i32::try_from(HOTBAR_SLOTS).unwrap_or(6);
        let total = slots * slot + (slots - 1) * gap;
        let x = (width - total) / 2;
        let y = height - slot - 8;

        for index in 0..slots {
            let left = x + index * (slot + gap);
            let selected =
                usize::try_from(index).unwrap_or(0) == game.player.inventory.selected_index();
            let background = if selected {
                Color::from_srgb_hex(0xF0_D8_A0).with_alpha(0.92)
            } else {
                Color::from_srgb_hex(0x1E_18_24).with_alpha(0.72)
            };
            self.fill(
                art,
                Panel::new(left, y, slot, slot),
                background,
                Z_HUD_PANEL,
            );

            let Some(stack) = game
                .player
                .inventory
                .slots()
                .get(usize::try_from(index).unwrap_or(0))
                .copied()
                .flatten()
            else {
                continue;
            };
            if let Some(region) = art.item(stack.item) {
                let page = art.atlas.page_size();
                let position = self.screen_to_world(Vec2::from_ints(left + 3, y + slot - 3));
                self.hud.push(
                    DrawSprite::new(position, region.size(), region.layer)
                        .with_uv(region.uv(page))
                        .anchored(Vec2::new(Fx::ZERO, Fx::ONE))
                        .at_z(Z_HUD_TEXT)
                        .tinted(Color::WHITE),
                );
            }
            if stack.count > 1 {
                let label = stack.count.to_string();
                let label_width = i32::try_from(font::text_width(&label)).unwrap_or(0);
                self.shadowed_text(
                    art,
                    left + slot - label_width - 2,
                    y + slot - 9,
                    &label,
                    Color::WHITE,
                    Z_HUD_TEXT,
                );
            }
        }

        // The selected item's name, centred above the hotbar.
        if let Some(stack) = game.player.inventory.selected() {
            let name = item(stack.item).name;
            let label_width = i32::try_from(font::text_width(name)).unwrap_or(0);
            self.shadowed_text(
                art,
                (width - label_width) / 2,
                y - 12,
                name,
                Color::from_srgb_hex(0xF4_E8_C8),
                Z_HUD_TEXT,
            );
        }
    }

    /// The last action's result, as a line of text at the top-left.
    fn draw_message(&mut self, game: &Game, art: &Art, height: i32) {
        let Some(message) = describe(&game.last_outcome) else {
            return;
        };
        let text_width = i32::try_from(font::text_width(&message)).unwrap_or(0);
        self.fill(
            art,
            Panel::new(6, height - 26, text_width + 10, 15),
            Color::from_srgb_hex(0x1E_18_24).with_alpha(0.78),
            Z_HUD_PANEL,
        );
        self.shadowed_text(
            art,
            11,
            height - 22,
            &message,
            Color::from_srgb_hex(0xF4_E8_C8),
            Z_HUD_TEXT,
        );
    }
}

/// Picks one of `count` variants for a cell.
///
/// A hash of the coordinates rather than a sequence: the choice must not
/// depend on the order cells are drawn in, or scrolling the camera would make
/// the ground change underfoot.
fn variant_index(cell: IVec2, count: usize) -> usize {
    if count <= 1 {
        return 0;
    }
    // A small integer hash. The multipliers are odd and coprime so adjacent
    // cells land in different variants instead of forming stripes.
    let mixed = (cell.x.wrapping_mul(73_856_093)) ^ (cell.y.wrapping_mul(19_349_663));
    usize::try_from(mixed.unsigned_abs()).unwrap_or(0) % count
}

/// Adds a little life to a terrain variant: blades on grass, ripples on water,
/// grit on stone.
///
/// Terrain generation gives each tile per-pixel noise, which reads as texture
/// but not as *content*. These few marks are what make a field look like grass
/// rather than green static.
fn scatter_detail(
    canvas: &mut Canvas,
    palette: &mut Palette,
    tile: TileId,
    variant: usize,
    rng: &mut Rng,
) {
    let width = i32::try_from(canvas.width()).unwrap_or(16);
    let height = i32::try_from(canvas.height()).unwrap_or(16);
    let last = PaletteIndex(u8::try_from(palette.len().saturating_sub(1)).unwrap_or(1));
    let light = PaletteIndex(1);

    match tile {
        tiles::GRASS => {
            // Blades: a three-pixel vertical stroke, dark at the base and
            // light at the tip. Enough of them to break up the flat field,
            // few enough that the tile still reads as one surface.
            for _ in 0..rng.range(6, 10) {
                let x = rng.range(1, width - 2);
                let y = rng.range(2, height - 4);
                canvas.set(x, y + 2, last);
                canvas.set(x, y + 1, last);
                canvas.set(x, y, light);
            }
            // One variant carries wildflowers. Restricting them to a quarter
            // of the tiles is what keeps them a detail rather than a lawn of
            // confetti.
            if variant == TERRAIN_VARIANTS - 1 {
                let colours = [
                    Rgba::hex(0xE8_D8_60),
                    Rgba::hex(0xE0_86_C8),
                    Rgba::hex(0xF0_F0_E0),
                ];
                let picked = colours[usize::try_from(rng.range(0, 2)).unwrap_or(0)];
                let petal = palette.push(picked);
                let heart = palette.push(picked.shade(180));
                let x = rng.range(3, width - 4);
                let y = rng.range(3, height - 4);
                canvas.set(x, y, petal);
                canvas.set(x + 1, y, petal);
                canvas.set(x, y + 1, heart);
                canvas.set(x + 1, y + 1, petal);
            }
        }
        tiles::WATER => {
            // Horizontal ripples read as a surface; vertical marks read as
            // noise, which is why these are always two pixels wide.
            for _ in 0..rng.range(2, 4) {
                let x = rng.range(1, width - 3);
                let y = rng.range(1, height - 2);
                canvas.set(x, y, light);
                canvas.set(x + 1, y, light);
            }
        }
        tiles::STONE | tiles::PATH | tiles::CAVE_FLOOR => {
            for _ in 0..rng.range(2, 5) {
                canvas.set(rng.range(1, width - 2), rng.range(1, height - 2), last);
            }
        }
        _ => {}
    }
}

/// The ground tile drawn beneath a tile that is really an object.
///
/// A tree stands on grass and a ladder on a cave floor; without this every
/// object would sit on a transparent hole.
fn tile_ground(tile: TileId) -> TileId {
    match tile {
        tiles::TREE | tiles::ROCK => tiles::GRASS,
        tiles::BUILDING | tiles::DOOR => tiles::SOIL,
        tiles::ORE | tiles::LADDER => tiles::CAVE_FLOOR,
        other => other,
    }
}

/// The world position a tile's sprite anchors to: the bottom centre of its
/// cell, matching the renderer's anchor convention.
fn tile_anchor(cell: IVec2) -> Vec2 {
    Vec2::new(
        Fx::from_num(cell.x) * TILE_SIZE + TILE_SIZE * Fx::HALF,
        Fx::from_num(cell.y) * TILE_SIZE + TILE_SIZE,
    )
}

/// The walk frame a villager should be on.
///
/// Derived from position rather than from a timer, so a villager standing
/// still stands still and one walking cycles at a rate that matches their
/// speed — the same rule the player's animation follows.
fn walk_frame(position: Vec2) -> usize {
    let travelled = usize::try_from((position.x + position.y).to_int().unsigned_abs()).unwrap_or(0);
    (travelled / 6) % WALK_FRAMES
}

/// Converts a 24-hour clock into the 12-hour form the HUD shows.
fn twelve_hour(hour: u32) -> (u32, &'static str) {
    let suffix = if hour < 12 { "am" } else { "pm" };
    let display = match hour % 12 {
        0 => 12,
        other => other,
    };
    (display, suffix)
}

/// The HUD line for an action's result, or `None` when there is nothing to
/// say.
#[must_use]
pub fn describe(outcome: &ActionOutcome) -> Option<String> {
    match outcome {
        ActionOutcome::Nothing => None,
        ActionOutcome::Farm(action) => describe_farm(*action),
        ActionOutcome::Gathered { item: id, count } => {
            Some(format!("Got {} x{}", item(*id).name, count))
        }
        ActionOutcome::Talked { name, line } => Some(format!("{name}: {line}")),
        ActionOutcome::Gifted { name, points } => {
            Some(format!("{name} liked that. (+{points} friendship)"))
        }
        ActionOutcome::Shipped { gold } => Some(format!("Shipped for {gold}g")),
        ActionOutcome::TooTired => Some("Too tired. Get some sleep.".to_owned()),
    }
}

/// The HUD line for a farming action.
fn describe_farm(action: crate::farm::FarmAction) -> Option<String> {
    use crate::farm::FarmAction;
    match action {
        FarmAction::Nothing => None,
        FarmAction::Tilled => Some("Tilled the soil.".to_owned()),
        FarmAction::Watered => Some("Watered.".to_owned()),
        FarmAction::Planted(id) => Some(format!("Planted {}.", crop(id).name)),
        FarmAction::Harvested {
            item: produce,
            count,
        } => Some(format!("Harvested {} x{}", item(produce).name, count)),
        FarmAction::Cleared => Some("Cleared the plot.".to_owned()),
    }
}

/// The clear colour for the world pass.
///
/// The world is drawn at full brightness and darkened by the lighting pass, so
/// this is only what shows outside the map. It follows the ambient light so
/// the edge of the world reads as distance rather than as a black void.
#[must_use]
pub fn frame_settings(game: &Game) -> verdant_render_2d::FrameSettings {
    let ambient = game.calendar.ambient_light();
    verdant_render_2d::FrameSettings::clearing(
        Color::from_srgb_hex(0x10_18_28).scale_rgb(0.5 + ambient.r * 0.5),
    )
}

/// How the composite should light this frame.
///
/// Time of day and weather reach the screen through exactly one value, which
/// is why a scene never has to know what time it is.
#[must_use]
pub fn light_settings(game: &Game) -> LightSettings {
    LightSettings {
        ambient: game.calendar.ambient_light(),
        light_scale: 1.0,
        // Slightly above one so a lantern can lift its immediate surroundings
        // past the ambient level rather than merely matching it.
        maximum: 1.15,
    }
}

/// How friendship reads on the HUD, as a count of filled hearts out of five.
#[must_use]
pub fn hearts(friendship: i32) -> u32 {
    let clamped = friendship.clamp(0, MAX_FRIENDSHIP);
    u32::try_from(clamped * 5 / MAX_FRIENDSHIP).unwrap_or(0)
}

/// True when the weather should draw falling particles over the scene.
#[must_use]
pub const fn has_precipitation(weather: Weather) -> bool {
    matches!(weather, Weather::Rain | Weather::Storm | Weather::Snow)
}

/// The GPU half: uploads the atlas and draws frames.
///
/// Separate from [`Scene`] so that everything above can be tested without a
/// graphics device.
pub struct GameRenderer {
    gpu: verdant_render_2d::GpuContext,
    renderer: verdant_render_2d::SpriteRenderer,
    lights: LightRenderer,
    compositor: Compositor,
    presenter: verdant_render_2d::Presenter,
    atlas_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    source_bind_group: wgpu::BindGroup,
    /// The world, drawn at full brightness.
    scene: verdant_render_2d::RenderTarget,
    /// Every light, accumulated additively.
    light_map: verdant_render_2d::RenderTarget,
    /// The lit world, and then the HUD on top of it.
    target: verdant_render_2d::RenderTarget,
    /// Kept alive because the bind group borrows it.
    _atlas: TextureArray,
}

impl GameRenderer {
    /// Builds the GPU-side renderer for a surface's format.
    ///
    /// # Errors
    ///
    /// Returns the renderer's error when the atlas cannot be uploaded.
    pub fn new(
        gpu: &verdant_render_2d::GpuContext,
        art: &Art,
        surface_format: wgpu::TextureFormat,
    ) -> Result<GameRenderer, verdant_render_2d::GpuError> {
        let atlas = art.atlas().upload(gpu)?;
        let renderer = verdant_render_2d::SpriteRenderer::new(gpu);
        let atlas_bind_group = renderer.bind_atlas(gpu, &atlas);

        let format = verdant_render_2d::RenderTarget::FORMAT;
        let scene = verdant_render_2d::RenderTarget::new(gpu, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let light_map = verdant_render_2d::RenderTarget::new(gpu, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let target = verdant_render_2d::RenderTarget::new(gpu, INTERNAL_WIDTH, INTERNAL_HEIGHT);

        let lights = LightRenderer::new(gpu, format);
        let compositor = Compositor::new(gpu, format);
        let composite_bind_group = compositor.bind(gpu, &scene, &light_map);

        let presenter = verdant_render_2d::Presenter::new(gpu, surface_format);
        let source_bind_group = presenter.bind_source(gpu, &target);

        Ok(GameRenderer {
            gpu: gpu.clone(),
            renderer,
            lights,
            compositor,
            presenter,
            atlas_bind_group,
            composite_bind_group,
            source_bind_group,
            scene,
            light_map,
            target,
            _atlas: atlas,
        })
    }

    /// Renders a prepared scene into the low-resolution target and presents it
    /// to `destination` at `destination_size`.
    pub fn draw(
        &mut self,
        scene: &mut Scene,
        game: &Game,
        art: &Art,
        destination: &wgpu::TextureView,
        destination_size: (u32, u32),
    ) {
        let camera = scene.camera;

        // 1. The world, at full brightness, into its own target.
        let world = scene.build(game, art).prepare();
        self.renderer.render(
            &self.gpu,
            &self.scene,
            &self.atlas_bind_group,
            &camera,
            world,
            frame_settings(game),
        );

        // 2. Every light, accumulated additively.
        self.lights
            .render(&self.gpu, &self.light_map, &camera, scene.lights());

        // 3. Scene times illumination.
        self.compositor.composite(
            &self.gpu,
            self.target.view(),
            &self.composite_bind_group,
            light_settings(game),
        );

        // 4. The interface, over the lit world and untouched by it. Loading
        //    rather than clearing is what keeps the composited frame beneath.
        let hud = scene.hud_batch().prepare();
        self.renderer.render(
            &self.gpu,
            &self.target,
            &self.atlas_bind_group,
            &camera,
            hud,
            verdant_render_2d::FrameSettings::loading(),
        );

        self.presenter.present(
            &self.gpu,
            destination,
            &self.source_bind_group,
            (INTERNAL_WIDTH, INTERNAL_HEIGHT),
            destination_size,
            Color::BLACK,
        );
    }

    /// Reads the low-resolution frame back to the CPU.
    ///
    /// # Errors
    ///
    /// Returns the renderer's error if the copy fails.
    pub fn read_frame(&self) -> Result<Vec<u8>, verdant_render_2d::GpuError> {
        self.target.read_pixels(&self.gpu)
    }
}

impl std::fmt::Debug for GameRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameRenderer")
            .field("internal", &(INTERNAL_WIDTH, INTERNAL_HEIGHT))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world with art, as most tests need both.
    fn fixture() -> (Game, Art) {
        let game = Game::new(20_260_808);
        let seeds: Vec<u64> = game.villagers.iter().map(|v| v.appearance_seed).collect();
        let art = Art::generate(game.seed, &seeds);
        (game, art)
    }

    #[test]
    fn the_internal_resolution_scales_cleanly_to_common_displays() {
        // 480x270 into 1920x1080 is exactly 4x with nothing left over, which
        // is the whole reason for that resolution.
        assert_eq!(
            verdant_render_2d::integer_scale((INTERNAL_WIDTH, INTERNAL_HEIGHT), (1920, 1080)),
            4
        );
        assert_eq!(
            verdant_render_2d::letterbox((INTERNAL_WIDTH, INTERNAL_HEIGHT), (1920, 1080)),
            ((0, 0), (1920, 1080))
        );
    }

    #[test]
    fn every_sprite_the_game_needs_is_generated() {
        let (_, art) = fixture();

        for id in [
            tiles::GRASS,
            tiles::SOIL,
            tiles::PATH,
            tiles::WATER,
            tiles::STONE,
            tiles::CAVE_FLOOR,
            tiles::CAVE_WALL,
            tiles::BRIDGE,
        ] {
            assert!(
                art.terrain(id, IVec2::ZERO).is_some(),
                "tile {} has no sprite",
                id.0
            );
        }
        for definition in ITEMS {
            assert!(
                art.item(definition.id).is_some(),
                "{} has no icon",
                definition.name
            );
        }
        for definition in CROPS {
            for stage in 0..definition.stage_count() {
                assert!(
                    art.crop(definition.id, stage).is_some(),
                    "{} stage {stage} has no sprite",
                    definition.name
                );
            }
        }
    }

    #[test]
    fn every_object_tile_draws_over_a_ground_tile() {
        // A tree on nothing would be a tree floating over a transparent hole.
        for tile in [
            tiles::TREE,
            tiles::ROCK,
            tiles::BUILDING,
            tiles::DOOR,
            tiles::ORE,
            tiles::LADDER,
        ] {
            let ground = tile_ground(tile);
            assert_ne!(ground, tile, "tile {} needs a ground beneath it", tile.0);
        }
    }

    #[test]
    fn packed_regions_never_overlap() {
        let (_, art) = fixture();
        // Every region on the same page must own its texels exclusively, or
        // two sprites would share pixels and one would corrupt the other.
        let mut by_layer: BTreeMap<u32, Vec<(u32, u32, u32, u32)>> = BTreeMap::new();
        let mut regions: Vec<Region> = Vec::new();
        regions.extend(art.terrain.values().flatten().copied());
        regions.extend(art.soil);
        regions.extend(art.crops.values().copied());
        regions.extend(art.items.values().copied());
        regions.extend(art.player.iter().copied());
        regions.extend(art.villagers.iter().flatten().copied());
        regions.extend(art.glyphs.iter().copied());
        regions.extend([
            art.rock, art.ore, art.wall, art.roof, art.door, art.ladder, art.white,
        ]);
        regions.extend(art.trees.iter().copied());

        for region in regions {
            let entry = by_layer.entry(region.layer).or_default();
            for existing in entry.iter() {
                let separate = region.rect.0 + region.rect.2 <= existing.0
                    || existing.0 + existing.2 <= region.rect.0
                    || region.rect.1 + region.rect.3 <= existing.1
                    || existing.1 + existing.3 <= region.rect.1;
                assert!(separate, "{:?} overlaps {existing:?}", region.rect);
            }
            entry.push(region.rect);
        }
    }

    #[test]
    fn every_region_lies_inside_its_page() {
        let (_, art) = fixture();
        let page = art.atlas().page_size();
        for region in art.glyphs.iter().chain(art.terrain.values().flatten()) {
            assert!(region.rect.0 + region.rect.2 <= page);
            assert!(region.rect.1 + region.rect.3 <= page);
            assert!(region.layer < art.atlas().page_count());
        }
    }

    #[test]
    fn uv_coordinates_stay_within_the_unit_square() {
        let (_, art) = fixture();
        let page = art.atlas().page_size();
        for region in art.items.values() {
            let uv = region.uv(page);
            assert!(uv.min.x >= Fx::ZERO && uv.min.y >= Fx::ZERO);
            assert!(uv.max().x <= Fx::ONE && uv.max().y <= Fx::ONE);
        }
    }

    #[test]
    fn an_oversized_sprite_is_refused_rather_than_corrupting_a_page() {
        let mut atlas = AtlasBuilder::new(16);
        let palette = Palette::new("p", &[Rgba::hex(0xFF_FF_FF)]);
        let canvas = Canvas::new(32, 8);
        assert_eq!(
            atlas.insert(&canvas, &palette),
            Err(TooLarge {
                size: (32, 8),
                page: 16
            })
        );
    }

    #[test]
    fn the_atlas_opens_a_new_page_when_one_fills_up() {
        let mut atlas = AtlasBuilder::new(16);
        let mut palette = Palette::new("p", &[]);
        let index = palette.push(Rgba::hex(0xFF_FF_FF));
        let mut canvas = Canvas::new(8, 8);
        canvas.fill_rect(0, 0, 8, 8, index);

        assert_eq!(atlas.page_count(), 1);
        // Each 8x8 sprite claims a 9x9 cell, so a 16px page holds one per
        // shelf and one shelf; the third must open a page.
        for _ in 0..4 {
            atlas.insert(&canvas, &palette).expect("fits");
        }
        assert!(atlas.page_count() > 1, "the atlas should have grown");
    }

    #[test]
    fn a_frame_is_built_without_a_gpu() {
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.snap_to(&game);
        let sprites = scene.build(&game, &art).sprites().len();
        assert!(sprites > 100, "a frame should have real content: {sprites}");
    }

    #[test]
    fn the_scene_is_sorted_back_to_front() {
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art).prepare();

        let sprites = scene.sprites();
        for pair in sprites.windows(2) {
            let (first, second) = (&pair[0], &pair[1]);
            assert!(
                (first.z, first.position.y) <= (second.z, second.position.y),
                "sprites are out of order: {:?} then {:?}",
                (first.z, first.position.y),
                (second.z, second.position.y)
            );
        }
    }

    #[test]
    fn the_hud_draws_above_the_world() {
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art).prepare();

        // The two live in separate batches so the HUD can be drawn after the
        // lighting composite rather than through it.
        assert!(!scene.sprites().is_empty(), "the world should be drawn");
        assert!(!scene.hud_sprites().is_empty(), "the HUD should be drawn");
    }

    #[test]
    fn building_roofs_sit_above_their_walls() {
        // The roof/wall split is decided by the tile above, so a column of
        // building tiles must produce exactly one roof at its top.
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);

        let roofs = scene
            .sprites()
            .iter()
            .filter(|sprite| sprite.uv_rect == art.roof.uv(ATLAS_PAGE))
            .count();
        let walls = scene
            .sprites()
            .iter()
            .filter(|sprite| sprite.uv_rect == art.wall.uv(ATLAS_PAGE))
            .count();
        assert!(
            roofs > 0,
            "the farmhouse should be in view with a visible roof"
        );
        assert!(walls > 0, "and walls beneath it");
    }

    #[test]
    fn the_camera_stays_inside_the_map() {
        let (mut game, art) = fixture();
        let mut scene = Scene::new();
        // Walk the player into the top-left corner and make sure the camera
        // does not follow them past the map edge into empty space.
        game.player.position = Vec2::from_ints(4, 4);
        scene.snap_to(&game);
        scene.build(&game, &art);

        let visible = scene.camera.visible_bounds();
        assert!(visible.min.x >= Fx::ZERO, "camera ran off the left edge");
        assert!(visible.min.y >= Fx::ZERO, "camera ran off the top edge");
    }

    #[test]
    fn each_facing_draws_a_different_frame() {
        let (_, art) = fixture();
        let south = art.player_frame(FacingState::South, 0);
        let north = art.player_frame(FacingState::North, 0);
        let east = art.player_frame(FacingState::East, 0);
        assert_ne!(south, north);
        assert_ne!(south, east);
        assert_ne!(north, east);
    }

    #[test]
    fn walk_frames_cycle_and_stay_in_range() {
        let (_, art) = fixture();
        for frame in 0..16 {
            // Out-of-range frame indices must wrap rather than panic, since
            // the animation counter is unbounded.
            let region = art.player_frame(FacingState::South, frame);
            assert!(region.rect.2 > 0);
        }
        assert_eq!(
            art.player_frame(FacingState::South, 0),
            art.player_frame(FacingState::South, WALK_FRAMES)
        );
    }

    #[test]
    fn a_villager_index_past_the_end_still_draws_something() {
        // Better a placeholder character than a panic mid-frame.
        let (_, art) = fixture();
        let region = art.villager_frame(999, FacingState::South, 0);
        assert!(region.rect.2 > 0);
    }

    #[test]
    fn every_printable_character_has_a_glyph_region() {
        let (_, art) = fixture();
        for code in 0x20u32..0x7F {
            let ch = char::from_u32(code).expect("ASCII");
            let region = art.glyph(ch);
            assert!(region.rect.3 == font::GLYPH_HEIGHT);
        }
        // And an unmapped character falls back rather than panicking.
        assert_eq!(art.glyph('\u{2603}'), art.glyph('?'));
    }

    #[test]
    fn crop_stages_past_maturity_clamp_to_the_ripe_sprite() {
        let (_, art) = fixture();
        let ripe = crop(CropId::Parsnip).stage_count() - 1;
        assert_eq!(
            art.crop(CropId::Parsnip, ripe),
            art.crop(CropId::Parsnip, ripe + 5)
        );
    }

    #[test]
    fn outcomes_that_matter_produce_a_message() {
        assert_eq!(describe(&ActionOutcome::Nothing), None);
        assert!(describe(&ActionOutcome::TooTired).is_some());
        assert!(describe(&ActionOutcome::Shipped { gold: 120 })
            .expect("a message")
            .contains("120"));
        assert!(describe(&ActionOutcome::Gathered {
            item: ItemId::Wood,
            count: 2
        })
        .expect("a message")
        .contains("Wood"));
    }

    #[test]
    fn the_clock_reads_as_a_twelve_hour_time() {
        assert_eq!(twelve_hour(0), (12, "am"));
        assert_eq!(twelve_hour(6), (6, "am"));
        assert_eq!(twelve_hour(12), (12, "pm"));
        assert_eq!(twelve_hour(23), (11, "pm"));
    }

    #[test]
    fn hearts_fill_as_friendship_grows() {
        assert_eq!(hearts(0), 0);
        assert_eq!(hearts(MAX_FRIENDSHIP), 5);
        assert_eq!(hearts(MAX_FRIENDSHIP / 2), 2);
        // Out-of-range values clamp rather than overflowing the display.
        assert_eq!(hearts(-500), 0);
        assert_eq!(hearts(MAX_FRIENDSHIP * 3), 5);
    }

    #[test]
    fn night_darkens_the_frame() {
        let (mut game, _) = fixture();
        let noon = light_settings(&game).ambient;
        // Wind the clock to late evening.
        game.calendar.minute = 23 * 60;
        let night = light_settings(&game).ambient;
        assert!(
            night.r < noon.r && night.b <= noon.b,
            "evening should be darker than midday: {night:?} vs {noon:?}"
        );
    }

    #[test]
    fn a_frame_builds_the_same_way_twice() {
        // The frame is a pure function of the game state, which is what lets a
        // rendering bug be reproduced from a save.
        let (game, art) = fixture();
        let mut first = Scene::new();
        first.snap_to(&game);
        first.build(&game, &art).prepare();
        let a: Vec<_> = first.sprites().to_vec();

        let mut second = Scene::new();
        second.snap_to(&game);
        second.build(&game, &art).prepare();
        let b: Vec<_> = second.sprites().to_vec();

        assert_eq!(a.len(), b.len());
        for (left, right) in a.iter().zip(b.iter()) {
            assert_eq!(left.position, right.position);
            assert_eq!(left.uv_rect, right.uv_rect);
            assert_eq!(left.layer, right.layer);
            assert_eq!(left.z, right.z);
        }
    }

    #[test]
    fn art_generation_is_deterministic() {
        let seeds = [1u64, 2, 3, 4];
        let first = Art::generate(99, &seeds);
        let second = Art::generate(99, &seeds);
        assert_eq!(first.atlas().pages(), second.atlas().pages());
    }

    #[test]
    fn a_different_world_seed_produces_different_art() {
        let seeds = [1u64, 2, 3, 4];
        let first = Art::generate(99, &seeds);
        let second = Art::generate(100, &seeds);
        assert_ne!(first.atlas().pages(), second.atlas().pages());
    }

    #[test]
    fn the_atlas_stays_within_the_engines_guaranteed_layer_count() {
        // wgpu's downlevel defaults guarantee 256 array layers; needing more
        // would mean the game no longer runs on the limits the engine targets.
        let (_, art) = fixture();
        assert!(
            art.atlas().page_count() <= 256,
            "the atlas grew to {} pages",
            art.atlas().page_count()
        );
    }

    #[test]
    fn precipitation_is_recognised() {
        assert!(has_precipitation(Weather::Rain));
        assert!(has_precipitation(Weather::Storm));
        assert!(!has_precipitation(Weather::Clear));
    }

    #[test]
    fn a_scene_off_the_edge_of_the_map_still_builds() {
        // Cell lookups outside the map must be skipped, not panic.
        let (mut game, art) = fixture();
        game.player.position = Vec2::from_ints(-200, -200);
        let mut scene = Scene::new();
        scene.camera.position = game.player.position;
        scene.build(&game, &art);
    }

    #[test]
    fn hotbar_slots_all_fit_on_screen() {
        let slot = 22;
        let gap = 2;
        let slots = i32::try_from(HOTBAR_SLOTS).unwrap_or(6);
        let total = slots * slot + (slots - 1) * gap;
        assert!(
            total < i32::try_from(INTERNAL_WIDTH).unwrap_or(480),
            "the hotbar is {total}px wide"
        );
    }

    #[test]
    fn text_is_measured_the_same_way_it_is_drawn() {
        // The panel behind a message is sized from `text_width`, so the two
        // must agree or the panel will clip the last letter.
        let message = "Harvested Parsnip x2";
        let drawn = font::render_text(message, PaletteIndex(1));
        assert_eq!(drawn.width(), font::text_width(message));
    }

    #[test]
    fn a_stack_of_one_shows_no_count() {
        // Cosmetic, but it is the difference between a HUD that reads as
        // designed and one that reads as a debug overlay.
        let (mut game, art) = fixture();
        game.player.inventory = crate::inventory::Inventory::new(0);
        game.player.inventory.add(ItemId::Wood, 1);
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);

        let ones = scene
            .hud_sprites()
            .iter()
            .filter(|sprite| sprite.uv_rect == art.glyph('1').uv(ATLAS_PAGE))
            .count();
        assert_eq!(ones, 0, "a single item should not be labelled");
    }

    #[test]
    fn the_selected_hotbar_slot_is_highlighted() {
        let (mut game, art) = fixture();
        let mut scene = Scene::new();

        game.player.inventory.select(0);
        scene.snap_to(&game);
        scene.build(&game, &art);
        let first: Vec<_> = scene
            .hud_sprites()
            .iter()
            .filter(|sprite| sprite.z == Z_HUD_PANEL)
            .map(|sprite| sprite.color.to_array())
            .collect();

        game.player.inventory.select(2);
        scene.build(&game, &art);
        let second: Vec<_> = scene
            .hud_sprites()
            .iter()
            .filter(|sprite| sprite.z == Z_HUD_PANEL)
            .map(|sprite| sprite.color.to_array())
            .collect();

        assert_ne!(first, second, "the highlight should have moved");
    }

    #[test]
    fn neighbouring_cells_do_not_all_pick_the_same_variant() {
        // The whole point of variants is that a field is not one stamp; a hash
        // that collapsed to a constant would silently undo that.
        let mut seen = std::collections::BTreeSet::new();
        for y in 0..8 {
            for x in 0..8 {
                seen.insert(variant_index(IVec2::new(x, y), TERRAIN_VARIANTS));
            }
        }
        assert_eq!(
            seen.len(),
            TERRAIN_VARIANTS,
            "an 8x8 patch should use every variant"
        );
    }

    #[test]
    fn the_variant_for_a_cell_never_changes() {
        // If it did, the ground would shimmer as the camera moved.
        let cell = IVec2::new(-13, 41);
        let first = variant_index(cell, TERRAIN_VARIANTS);
        for _ in 0..8 {
            assert_eq!(variant_index(cell, TERRAIN_VARIANTS), first);
        }
        // And negative coordinates must be handled, since the camera can look
        // past the map edge.
        assert!(variant_index(IVec2::new(-1, -1), TERRAIN_VARIANTS) < TERRAIN_VARIANTS);
    }

    #[test]
    fn a_single_variant_is_always_index_zero() {
        assert_eq!(variant_index(IVec2::new(5, 9), 1), 0);
        assert_eq!(variant_index(IVec2::new(5, 9), 0), 0);
    }

    #[test]
    fn terrain_variants_are_actually_different_sprites() {
        let (_, art) = fixture();
        let grass: std::collections::BTreeSet<Region> = (0..16)
            .map(|x| {
                art.terrain(tiles::GRASS, IVec2::new(x, 0))
                    .expect("grass has sprites")
            })
            .collect();
        assert!(
            grass.len() > 1,
            "grass should draw from more than one sprite"
        );
    }

    #[test]
    fn the_hud_is_drawn_outside_the_lighting() {
        // The HUD is not part of the world. It used to be tinted by the time
        // of day and then divided back out again; drawing it in its own pass
        // after the composite is both simpler and actually correct.
        let (mut game, art) = fixture();
        game.calendar.minute = 23 * 60;
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);

        assert!(!scene.hud_sprites().is_empty(), "the HUD should exist");
        for sprite in scene.hud_sprites() {
            assert!(
                sprite.z >= Z_HUD_PANEL,
                "a world sprite leaked into the HUD batch at z {}",
                sprite.z
            );
        }
        for sprite in scene.sprites() {
            assert!(
                sprite.z < Z_HUD_PANEL,
                "a HUD sprite leaked into the world batch at z {}",
                sprite.z
            );
        }
    }

    #[test]
    fn nothing_is_lit_in_daylight() {
        // A lantern glowing at noon reads as a bug rather than as detail.
        let (mut game, art) = fixture();
        game.calendar.minute = 12 * 60;
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);
        assert!(scene.lights().is_empty(), "daylight needs no lights");
    }

    #[test]
    fn the_village_and_the_player_light_up_at_night() {
        let (mut game, art) = fixture();
        game.calendar.minute = 23 * 60;
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);

        let lights = scene.lights();
        // One per home, the shop, the farmhouse, and the player's lantern.
        let expected = game.layout.homes.len() + 3;
        assert_eq!(lights.len(), expected, "got {lights:?}");
        assert!(
            lights
                .iter()
                .any(|light| light.position == game.player.position),
            "the player should carry a lantern"
        );
    }

    #[test]
    fn lights_fade_in_through_dusk_rather_than_switching_on() {
        let (mut game, art) = fixture();
        let mut scene = Scene::new();
        scene.snap_to(&game);

        game.calendar.minute = 19 * 60;
        scene.build(&game, &art);
        let dusk = scene
            .lights()
            .iter()
            .map(|light| light.intensity)
            .fold(0.0f32, f32::max);

        game.calendar.minute = 23 * 60;
        scene.build(&game, &art);
        let night = scene
            .lights()
            .iter()
            .map(|light| light.intensity)
            .fold(0.0f32, f32::max);

        assert!(dusk > 0.0, "dusk should already be lighting up");
        assert!(night > dusk, "night ({night}) should beat dusk ({dusk})");
    }

    #[test]
    fn every_light_has_real_reach_and_brightness() {
        // A light with no radius or no intensity costs a draw and shows
        // nothing, and would be culled by the renderer anyway.
        let (mut game, art) = fixture();
        game.calendar.minute = 23 * 60;
        let mut scene = Scene::new();
        scene.snap_to(&game);
        scene.build(&game, &art);
        for light in scene.lights() {
            assert!(light.radius > Fx::ZERO, "{light:?} has no reach");
            assert!(light.intensity > 0.0, "{light:?} is not lit");
        }
    }

    #[test]
    fn the_ambient_light_drives_the_composite() {
        let (mut game, _) = fixture();
        game.calendar.minute = 12 * 60;
        let noon = light_settings(&game);
        game.calendar.minute = 23 * 60;
        let night = light_settings(&game);
        assert!(night.ambient.r < noon.ambient.r);
        // Not `is_fully_lit`: a season grades the whole day, so spring tints
        // blue to 0.96 even at noon. What matters is that the brightest
        // channel is at full strength, since that is what decides whether any
        // light is emitted.
        let brightest = |colour: Color| colour.r.max(colour.g).max(colour.b);
        assert!(
            brightest(noon.ambient) >= 1.0,
            "noon should be at full strength"
        );
        assert!(brightest(night.ambient) < 0.7, "night should be well down");
    }

    #[test]
    fn every_touch_control_is_on_screen() {
        // A control drawn off the edge is a control that cannot be pressed.
        let layout = touch_layout();
        let width = Fx::from_num(i32::try_from(INTERNAL_WIDTH).expect("small"));
        let height = Fx::from_num(i32::try_from(INTERNAL_HEIGHT).expect("small"));

        let stick = layout.stick;
        assert!(stick.centre.x - stick.radius >= Fx::ZERO);
        assert!(stick.centre.y + stick.radius <= height);

        for button in &layout.buttons {
            assert!(
                button.bounds.min.x >= Fx::ZERO,
                "{:?} runs off the left",
                button.action
            );
            assert!(
                button.bounds.min.y >= Fx::ZERO,
                "{:?} runs off the top",
                button.action
            );
            assert!(
                button.bounds.max().x <= width,
                "{:?} runs off the right",
                button.action
            );
            assert!(
                button.bounds.max().y <= height,
                "{:?} runs off the bottom",
                button.action
            );
        }
    }

    #[test]
    fn touch_buttons_do_not_overlap_each_other() {
        // Overlapping buttons mean one is unreachable, since the later one
        // always wins the press.
        let layout = touch_layout();
        for (index, a) in layout.buttons.iter().enumerate() {
            for b in layout.buttons.iter().skip(index + 1) {
                assert!(
                    !a.bounds.intersects(b.bounds),
                    "{:?} overlaps {:?}",
                    a.action,
                    b.action
                );
            }
        }
    }

    #[test]
    fn the_buttons_do_not_overlap_the_stick() {
        // A press meant for the stick that lands on a button, or the reverse,
        // is the most frustrating failure a touch layout can have.
        let layout = touch_layout();
        for button in &layout.buttons {
            let closest = button.bounds.clamp_point(layout.stick.centre);
            let distance = closest.distance(layout.stick.centre);
            assert!(
                distance > layout.stick.radius + layout.stick.grab_margin,
                "{:?} is only {distance} from the stick",
                button.action
            );
        }
    }

    #[test]
    fn pressing_where_a_button_is_drawn_triggers_it() {
        // The property the shared layout exists to guarantee.
        let layout = touch_layout();
        let mut state = TouchState::new();
        for (index, button) in layout.buttons.iter().enumerate() {
            state.clear();
            state.handle(
                verdant_input::TouchEvent {
                    id: verdant_input::TouchId(1),
                    position: button.bounds.centre(),
                    phase: verdant_input::TouchPhase::Started,
                },
                &layout,
            );
            assert!(
                state.is_pressed(button.action),
                "button {index} ({:?}) did not respond at its own centre",
                button.action
            );
        }
    }

    #[test]
    fn the_touch_controls_are_hidden_by_default_on_desktop() {
        // A desktop player should never see a thumbstick they cannot use.
        let scene = Scene::new();
        assert_eq!(scene.show_touch_controls, cfg!(target_os = "android"));
    }

    #[test]
    fn the_touch_controls_draw_above_the_hud() {
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.show_touch_controls = true;
        scene.snap_to(&game);
        scene.build(&game, &art);

        let controls = scene
            .hud_sprites()
            .iter()
            .filter(|sprite| sprite.z >= Z_TOUCH)
            .count();
        assert!(controls > 0, "the controls should have been drawn");
    }

    #[test]
    fn hiding_the_touch_controls_draws_nothing_extra() {
        let (game, art) = fixture();
        let mut scene = Scene::new();
        scene.show_touch_controls = false;
        scene.snap_to(&game);
        scene.build(&game, &art);
        assert_eq!(scene.sprites().iter().filter(|s| s.z >= Z_TOUCH).count(), 0);
    }

    #[test]
    fn the_drawn_thumb_follows_the_finger() {
        let (game, art) = fixture();
        let layout = touch_layout();
        let mut touch = TouchState::new();
        touch.handle(
            verdant_input::TouchEvent {
                id: verdant_input::TouchId(1),
                position: layout.stick.centre + Vec2::from_ints(10, 0),
                phase: verdant_input::TouchPhase::Started,
            },
            &layout,
        );

        let mut scene = Scene::new();
        scene.show_touch_controls = true;
        scene.snap_to(&game);
        scene.observe_touch(&touch);
        scene.build(&game, &art);
        let with_finger: Vec<Vec2> = scene
            .hud_sprites()
            .iter()
            .filter(|s| s.z >= Z_TOUCH)
            .map(|s| s.position)
            .collect();

        touch.clear();
        scene.observe_touch(&touch);
        scene.build(&game, &art);
        let at_rest: Vec<Vec2> = scene
            .hud_sprites()
            .iter()
            .filter(|s| s.z >= Z_TOUCH)
            .map(|s| s.position)
            .collect();

        assert_ne!(with_finger, at_rest, "the thumb marker should have moved");
    }

    #[test]
    fn the_drawn_thumb_stays_inside_its_ring() {
        // A thumb dragged across the screen must not drag the marker with it.
        let (game, art) = fixture();
        let layout = touch_layout();
        let mut touch = TouchState::new();
        touch.handle(
            verdant_input::TouchEvent {
                id: verdant_input::TouchId(1),
                position: layout.stick.centre,
                phase: verdant_input::TouchPhase::Started,
            },
            &layout,
        );
        touch.handle(
            verdant_input::TouchEvent {
                id: verdant_input::TouchId(1),
                position: Vec2::from_ints(2000, 2000),
                phase: verdant_input::TouchPhase::Moved,
            },
            &layout,
        );

        let mut scene = Scene::new();
        scene.show_touch_controls = true;
        scene.snap_to(&game);
        scene.observe_touch(&touch);
        scene.build(&game, &art);

        let visible = Rect::from_ints(
            -8,
            -8,
            i32::try_from(INTERNAL_WIDTH).expect("small") + 16,
            i32::try_from(INTERNAL_HEIGHT).expect("small") + 16,
        );
        for sprite in scene.hud_sprites().iter().filter(|s| s.z >= Z_TOUCH) {
            let screen = scene.camera.world_to_screen(sprite.position);
            assert!(
                visible.contains_point(screen),
                "a control was drawn at {screen:?}, off screen"
            );
        }
    }

    #[test]
    fn a_held_button_looks_different_from_a_released_one() {
        // On a touchscreen there is no travel to feel; the only confirmation
        // a player gets is visual.
        let (game, art) = fixture();
        let layout = touch_layout();
        let mut scene = Scene::new();
        scene.show_touch_controls = true;
        scene.snap_to(&game);

        scene.build(&game, &art);
        let released: Vec<[f32; 4]> = scene
            .hud_sprites()
            .iter()
            .filter(|s| s.z >= Z_TOUCH)
            .map(|s| s.color.to_array())
            .collect();

        let mut touch = TouchState::new();
        touch.handle(
            verdant_input::TouchEvent {
                id: verdant_input::TouchId(1),
                position: layout.buttons[0].bounds.centre(),
                phase: verdant_input::TouchPhase::Started,
            },
            &layout,
        );
        scene.observe_touch(&touch);
        scene.build(&game, &art);
        let held: Vec<[f32; 4]> = scene
            .hud_sprites()
            .iter()
            .filter(|s| s.z >= Z_TOUCH)
            .map(|s| s.color.to_array())
            .collect();

        assert_ne!(released, held, "a pressed button should light up");
    }

    #[test]
    fn every_touch_button_has_a_readable_label() {
        let layout = touch_layout();
        for button in &layout.buttons {
            let label = touch_button_label(button.action);
            assert!(!label.is_empty());
            let width = i32::try_from(font::text_width(label)).expect("small");
            assert!(
                width <= button.bounds.size.x.to_int(),
                "{label:?} is {width}px wide but its button is {}px",
                button.bounds.size.x.to_int()
            );
        }
    }

    #[test]
    fn tile_anchors_land_on_the_bottom_of_their_cell() {
        // The renderer anchors sprites at bottom-centre, so a tile's anchor
        // must be the bottom of its cell or the whole map sits half a tile
        // high.
        let anchor = tile_anchor(IVec2::new(3, 4));
        assert_eq!(anchor.x, Fx::from_num(3) * TILE_SIZE + TILE_SIZE * Fx::HALF);
        assert_eq!(anchor.y, Fx::from_num(4) * TILE_SIZE + TILE_SIZE);
    }
}
