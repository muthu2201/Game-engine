//! Tile storage.

use serde::{Deserialize, Serialize};
use verdant_core_math::{Fx, IVec2, Rect, Vec2};
use verdant_physics_2d::SolidGrid;

/// Index of a tile within a tileset.
///
/// [`TileId::EMPTY`] is reserved: it means "nothing here", not "tile zero".
/// Reserving a value rather than wrapping every cell in an `Option` halves the
/// memory a large map needs and keeps the layer a flat `Vec<u16>`.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct TileId(pub u16);

impl TileId {
    /// The absence of a tile.
    pub const EMPTY: TileId = TileId(0);

    /// True when this cell holds no tile.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == TileId::EMPTY.0
    }
}

/// What a tile does to movement and sight.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct TileProperties {
    /// Blocks movement.
    pub solid: bool,
    /// Blocks line of sight.
    pub opaque: bool,
    /// Can be swum through rather than walked on.
    pub liquid: bool,
    /// Extra movement cost for the pathfinder, where `0` is normal ground.
    ///
    /// Used to make NPCs prefer paths rather than trample crops, without
    /// forbidding the shortcut outright.
    pub extra_cost: u16,
}

impl TileProperties {
    /// Ordinary walkable ground.
    pub const GROUND: TileProperties = TileProperties {
        solid: false,
        opaque: false,
        liquid: false,
        extra_cost: 0,
    };
    /// A solid, sight-blocking wall.
    pub const WALL: TileProperties = TileProperties {
        solid: true,
        opaque: true,
        liquid: false,
        extra_cost: 0,
    };
    /// Water: passable only by swimmers, and expensive to path through.
    pub const WATER: TileProperties = TileProperties {
        solid: false,
        opaque: false,
        liquid: true,
        extra_cost: 40,
    };
}

/// One layer of tiles.
///
/// Layers are the engine's drawing and collision order: the ground layer paints
/// first, decoration over it, and only layers marked [`TileLayer::collides`]
/// contribute to the solver.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileLayer {
    /// Human-readable name, used by the editor and the MCP tool surface.
    pub name: String,
    /// Row-major tile indices, `width * height` entries.
    tiles: Vec<TileId>,
    /// Whether this layer's solid tiles block movement.
    pub collides: bool,
    /// Draw order; higher values paint later.
    pub z: i16,
    width: u32,
    height: u32,
}

impl TileLayer {
    /// Creates an empty layer.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> TileLayer {
        assert!(
            width > 0 && height > 0,
            "TileLayer: dimensions must be non-zero"
        );
        let count = (width as usize) * (height as usize);
        TileLayer {
            name: name.into(),
            tiles: vec![TileId::EMPTY; count],
            collides: true,
            z: 0,
            width,
            height,
        }
    }

    /// Returns this layer with collision disabled.
    #[must_use]
    pub fn decorative(mut self) -> TileLayer {
        self.collides = false;
        self
    }

    /// Returns this layer at the given draw order.
    #[must_use]
    pub fn at_z(mut self, z: i16) -> TileLayer {
        self.z = z;
        self
    }

    /// Layer width in tiles.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Layer height in tiles.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// True when `cell` lies inside the layer.
    #[inline]
    #[must_use]
    pub const fn contains(&self, cell: IVec2) -> bool {
        cell.x >= 0 && cell.y >= 0 && (cell.x as u32) < self.width && (cell.y as u32) < self.height
    }

    /// Flat index of a cell, or `None` when it is out of bounds.
    #[inline]
    fn index_of(&self, cell: IVec2) -> Option<usize> {
        if !self.contains(cell) {
            return None;
        }
        Some((cell.y as usize) * (self.width as usize) + (cell.x as usize))
    }

    /// The tile at `cell`, or [`TileId::EMPTY`] outside the layer.
    #[inline]
    #[must_use]
    pub fn get(&self, cell: IVec2) -> TileId {
        self.index_of(cell)
            .map_or(TileId::EMPTY, |index| self.tiles[index])
    }

    /// Writes a tile, returning `false` when the cell is out of bounds.
    ///
    /// Silently ignoring out-of-bounds writes rather than panicking is what
    /// lets generators paint shapes that overhang the map edge without every
    /// one of them needing its own clipping.
    pub fn set(&mut self, cell: IVec2, tile: TileId) -> bool {
        match self.index_of(cell) {
            Some(index) => {
                self.tiles[index] = tile;
                true
            }
            None => false,
        }
    }

    /// Fills a rectangular region, clipped to the layer.
    pub fn fill(&mut self, region: Rect, tile: TileId) {
        let min_x = region.left().floor_int().max(0);
        let min_y = region.top().floor_int().max(0);
        let max_x = region.right().ceil_int().min(self.width as i32);
        let max_y = region.bottom().ceil_int().min(self.height as i32);
        for y in min_y..max_y {
            for x in min_x..max_x {
                self.set(IVec2::new(x, y), tile);
            }
        }
    }

    /// Replaces every cell with `tile`.
    pub fn clear_to(&mut self, tile: TileId) {
        self.tiles.fill(tile);
    }

    /// Every cell in row-major order, paired with its coordinate.
    pub fn iter(&self) -> impl Iterator<Item = (IVec2, TileId)> + '_ {
        let width = self.width;
        self.tiles.iter().enumerate().map(move |(index, tile)| {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            (
                IVec2::new((index % width) as i32, (index / width) as i32),
                *tile,
            )
        })
    }

    /// The raw tile buffer, for renderer upload.
    #[must_use]
    pub fn tiles(&self) -> &[TileId] {
        &self.tiles
    }
}

/// A stack of tile layers plus the properties of each tile index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tilemap {
    /// The layers, kept sorted by [`TileLayer::z`].
    layers: Vec<TileLayer>,
    /// Properties indexed by [`TileId`].
    properties: Vec<TileProperties>,
    /// World units per tile.
    tile_size: Fx,
    width: u32,
    height: u32,
}

impl Tilemap {
    /// Creates a map with no layers.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero or `tile_size` is not positive.
    #[must_use]
    pub fn new(width: u32, height: u32, tile_size: Fx) -> Tilemap {
        assert!(
            width > 0 && height > 0,
            "Tilemap: dimensions must be non-zero"
        );
        assert!(
            tile_size.is_positive(),
            "Tilemap: tile size must be positive"
        );
        Tilemap {
            layers: Vec::new(),
            // Index 0 is TileId::EMPTY, which is never solid.
            properties: vec![TileProperties::GROUND],
            tile_size,
            width,
            height,
        }
    }

    /// Adds a layer, keeping the stack ordered by draw order.
    ///
    /// # Panics
    ///
    /// Panics if the layer's dimensions differ from the map's.
    pub fn push_layer(&mut self, layer: TileLayer) {
        assert_eq!(
            (layer.width(), layer.height()),
            (self.width, self.height),
            "layer `{}` does not match the map's dimensions",
            layer.name
        );
        self.layers.push(layer);
        // A stable sort keeps insertion order among layers sharing a z.
        self.layers.sort_by_key(|layer| layer.z);
    }

    /// Registers the properties of a tile index, growing the table as needed.
    pub fn set_properties(&mut self, tile: TileId, properties: TileProperties) {
        let index = tile.0 as usize;
        if index >= self.properties.len() {
            self.properties.resize(index + 1, TileProperties::GROUND);
        }
        self.properties[index] = properties;
    }

    /// The properties of a tile index, defaulting to ordinary ground.
    #[inline]
    #[must_use]
    pub fn properties(&self, tile: TileId) -> TileProperties {
        self.properties
            .get(tile.0 as usize)
            .copied()
            .unwrap_or(TileProperties::GROUND)
    }

    /// The layers, in draw order.
    #[must_use]
    pub fn layers(&self) -> &[TileLayer] {
        &self.layers
    }

    /// A layer by name.
    #[must_use]
    pub fn layer(&self, name: &str) -> Option<&TileLayer> {
        self.layers.iter().find(|layer| layer.name == name)
    }

    /// A mutable layer by name.
    pub fn layer_mut(&mut self, name: &str) -> Option<&mut TileLayer> {
        self.layers.iter_mut().find(|layer| layer.name == name)
    }

    /// Map width in tiles.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Map height in tiles.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// World units per tile.
    #[inline]
    #[must_use]
    pub const fn tile_size(&self) -> Fx {
        self.tile_size
    }

    /// True when `cell` lies inside the map.
    #[inline]
    #[must_use]
    pub const fn contains(&self, cell: IVec2) -> bool {
        cell.x >= 0 && cell.y >= 0 && (cell.x as u32) < self.width && (cell.y as u32) < self.height
    }

    /// The combined properties of every layer at `cell`.
    ///
    /// Solidity and opacity are the union across layers, so a decorative rock
    /// drawn above walkable grass still blocks. Movement cost takes the maximum
    /// rather than the sum, so stacking three cosmetic layers does not make a
    /// tile three times as expensive to walk.
    #[must_use]
    pub fn properties_at(&self, cell: IVec2) -> TileProperties {
        let mut combined = TileProperties::GROUND;
        for layer in &self.layers {
            let tile = layer.get(cell);
            if tile.is_empty() {
                continue;
            }
            let properties = self.properties(tile);
            if layer.collides {
                combined.solid |= properties.solid;
            }
            combined.opaque |= properties.opaque;
            combined.liquid |= properties.liquid;
            combined.extra_cost = combined.extra_cost.max(properties.extra_cost);
        }
        combined
    }

    /// True when `cell` blocks movement, treating outside the map as solid.
    #[inline]
    #[must_use]
    pub fn is_solid(&self, cell: IVec2) -> bool {
        if !self.contains(cell) {
            // A closed world: the map edge is a wall, so a body can never walk
            // into unallocated space.
            return true;
        }
        self.properties_at(cell).solid
    }

    /// True when `cell` can be walked on.
    #[inline]
    #[must_use]
    pub fn is_walkable(&self, cell: IVec2) -> bool {
        self.contains(cell) && {
            let properties = self.properties_at(cell);
            !properties.solid && !properties.liquid
        }
    }

    /// The world-space bounds of the whole map.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        Rect::new(
            Vec2::ZERO,
            Vec2::new(
                self.tile_size * i32::try_from(self.width).unwrap_or(i32::MAX),
                self.tile_size * i32::try_from(self.height).unwrap_or(i32::MAX),
            ),
        )
    }

    /// The cell containing a world position.
    #[inline]
    #[must_use]
    pub fn cell_at(&self, position: Vec2) -> IVec2 {
        let (x, y) = position.to_tile(self.tile_size);
        IVec2::new(x, y)
    }

    /// Every cell overlapping a world-space rectangle, clipped to the map.
    #[must_use]
    pub fn cells_in(&self, area: Rect) -> Vec<IVec2> {
        let min = self.cell_at(area.min);
        let max = self.cell_at(area.max());
        let mut cells = Vec::new();
        for y in min.y.max(0)..=max.y.min(self.height as i32 - 1) {
            for x in min.x.max(0)..=max.x.min(self.width as i32 - 1) {
                cells.push(IVec2::new(x, y));
            }
        }
        cells
    }
}

impl SolidGrid for Tilemap {
    fn tile_size(&self) -> Fx {
        self.tile_size
    }

    fn is_solid(&self, cell: IVec2) -> bool {
        Tilemap::is_solid(self, cell)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    const GRASS: TileId = TileId(1);
    const STONE: TileId = TileId(2);
    const POND: TileId = TileId(3);

    fn map() -> Tilemap {
        let mut map = Tilemap::new(8, 8, fx(16));
        map.set_properties(GRASS, TileProperties::GROUND);
        map.set_properties(STONE, TileProperties::WALL);
        map.set_properties(POND, TileProperties::WATER);
        let mut ground = TileLayer::new("ground", 8, 8);
        ground.clear_to(GRASS);
        map.push_layer(ground);
        map
    }

    #[test]
    fn tiles_read_back_after_writing() {
        let mut map = map();
        let ground = map.layer_mut("ground").unwrap();
        assert!(ground.set(IVec2::new(3, 4), STONE));
        assert_eq!(ground.get(IVec2::new(3, 4)), STONE);
        assert_eq!(ground.get(IVec2::new(0, 0)), GRASS);
    }

    #[test]
    fn out_of_bounds_access_is_clamped_rather_than_panicking() {
        let mut map = map();
        let ground = map.layer_mut("ground").unwrap();
        assert!(!ground.set(IVec2::new(-1, 0), STONE));
        assert!(!ground.set(IVec2::new(8, 0), STONE));
        assert_eq!(ground.get(IVec2::new(-1, -1)), TileId::EMPTY);
        assert_eq!(ground.get(IVec2::new(99, 99)), TileId::EMPTY);
    }

    #[test]
    fn the_map_edge_is_solid_so_bodies_cannot_leave() {
        let map = map();
        assert!(!Tilemap::is_solid(&map, IVec2::new(0, 0)));
        assert!(Tilemap::is_solid(&map, IVec2::new(-1, 0)));
        assert!(Tilemap::is_solid(&map, IVec2::new(8, 0)));
    }

    #[test]
    fn solidity_is_the_union_across_layers() {
        let mut map = map();
        let mut decor = TileLayer::new("decor", 8, 8).at_z(1);
        decor.set(IVec2::new(2, 2), STONE);
        map.push_layer(decor);

        // Walkable grass below, a solid rock above: the cell blocks.
        assert!(Tilemap::is_solid(&map, IVec2::new(2, 2)));
        assert!(!Tilemap::is_solid(&map, IVec2::new(3, 3)));
    }

    #[test]
    fn a_decorative_layer_never_blocks() {
        let mut map = map();
        let mut decor = TileLayer::new("decor", 8, 8).at_z(1).decorative();
        decor.set(IVec2::new(2, 2), STONE);
        map.push_layer(decor);
        assert!(!Tilemap::is_solid(&map, IVec2::new(2, 2)));
    }

    #[test]
    fn movement_cost_takes_the_maximum_not_the_sum() {
        let mut map = map();
        for name in ["a", "b"] {
            let mut layer = TileLayer::new(name, 8, 8).at_z(1).decorative();
            layer.set(IVec2::new(1, 1), POND);
            map.push_layer(layer);
        }
        assert_eq!(
            map.properties_at(IVec2::new(1, 1)).extra_cost,
            TileProperties::WATER.extra_cost,
            "stacking cosmetic layers must not multiply the path cost"
        );
    }

    #[test]
    fn water_is_not_solid_but_is_not_walkable() {
        let mut map = map();
        map.layer_mut("ground").unwrap().set(IVec2::new(4, 4), POND);
        assert!(!Tilemap::is_solid(&map, IVec2::new(4, 4)));
        assert!(!map.is_walkable(IVec2::new(4, 4)));
        assert!(map.is_walkable(IVec2::new(0, 0)));
    }

    #[test]
    fn layers_are_kept_in_draw_order() {
        let mut map = map();
        map.push_layer(TileLayer::new("top", 8, 8).at_z(10));
        map.push_layer(TileLayer::new("under", 8, 8).at_z(-5));
        let order: Vec<&str> = map
            .layers()
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(order, vec!["under", "ground", "top"]);
    }

    #[test]
    #[should_panic(expected = "does not match the map's dimensions")]
    fn a_mismatched_layer_is_rejected() {
        let mut map = map();
        map.push_layer(TileLayer::new("wrong", 4, 4));
    }

    #[test]
    fn filling_a_region_clips_to_the_layer() {
        let mut map = map();
        let ground = map.layer_mut("ground").unwrap();
        ground.fill(Rect::from_ints(-2, -2, 4, 4), STONE);
        assert_eq!(ground.get(IVec2::new(0, 0)), STONE);
        assert_eq!(ground.get(IVec2::new(1, 1)), STONE);
        assert_eq!(ground.get(IVec2::new(3, 3)), GRASS);
    }

    #[test]
    fn world_positions_map_onto_cells() {
        let map = map();
        assert_eq!(map.cell_at(Vec2::from_ints(0, 0)), IVec2::new(0, 0));
        assert_eq!(map.cell_at(Vec2::from_ints(31, 15)), IVec2::new(1, 0));
        assert_eq!(map.bounds().size, Vec2::from_ints(128, 128));
    }

    #[test]
    fn cells_in_a_region_are_clipped_to_the_map() {
        let map = map();
        let cells = map.cells_in(Rect::from_ints(-100, -100, 140, 140));
        assert!(cells.contains(&IVec2::new(0, 0)));
        assert!(cells.iter().all(|cell| map.contains(*cell)));
    }

    #[test]
    fn iteration_visits_every_cell_in_row_major_order() {
        let layer = TileLayer::new("l", 3, 2);
        let coords: Vec<IVec2> = layer.iter().map(|(cell, _)| cell).collect();
        assert_eq!(
            coords,
            vec![
                IVec2::new(0, 0),
                IVec2::new(1, 0),
                IVec2::new(2, 0),
                IVec2::new(0, 1),
                IVec2::new(1, 1),
                IVec2::new(2, 1),
            ]
        );
    }

    #[test]
    fn the_map_serves_as_a_solid_grid_for_the_solver() {
        let mut map = map();
        map.layer_mut("ground")
            .unwrap()
            .set(IVec2::new(2, 2), STONE);
        let grid: &dyn SolidGrid = &map;
        assert_eq!(grid.tile_size(), fx(16));
        assert!(grid.is_solid(IVec2::new(2, 2)));
        assert!(!grid.is_solid(IVec2::new(0, 0)));
    }
}
