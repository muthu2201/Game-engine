//! World generation: the valley the game is played in, and the mine below it.

use serde::{Deserialize, Serialize};
use verdant_core_math::{fbm, Fx, IVec2, Rng, ValueNoise};
use verdant_tilemap::{TileLayer, TileProperties, Tilemap};

/// World units per tile. Sixteen matches the generated sprite grid.
pub const TILE_SIZE: Fx = Fx::from_num(16);

/// The valley's width in tiles.
pub const VALLEY_WIDTH: u32 = 64;
/// The valley's height in tiles.
pub const VALLEY_HEIGHT: u32 = 48;

/// Tile indices used across the game's maps.
pub mod tiles {
    use verdant_tilemap::TileId;

    /// Ordinary grass.
    pub const GRASS: TileId = TileId(1);
    /// Tilled soil.
    pub const SOIL: TileId = TileId(2);
    /// A worn path.
    pub const PATH: TileId = TileId(3);
    /// Water. Not walkable, but fishable from the bank.
    pub const WATER: TileId = TileId(4);
    /// Solid rock.
    pub const STONE: TileId = TileId(5);
    /// A tree trunk. Blocks movement until chopped.
    pub const TREE: TileId = TileId(6);
    /// A boulder. Blocks movement until broken.
    pub const ROCK: TileId = TileId(7);
    /// The floor of a mine level.
    pub const CAVE_FLOOR: TileId = TileId(8);
    /// A mine wall.
    pub const CAVE_WALL: TileId = TileId(9);
    /// A wall with ore in it.
    pub const ORE: TileId = TileId(10);
    /// The ladder down to the next mine level.
    pub const LADDER: TileId = TileId(11);
    /// A building's wall.
    pub const BUILDING: TileId = TileId(12);
    /// A doorway into a building.
    pub const DOOR: TileId = TileId(13);
    /// A plank bridge over water.
    pub const BRIDGE: TileId = TileId(14);
}

/// Registers every tile's properties on a map.
fn register_properties(map: &mut Tilemap) {
    use tiles as t;
    map.set_properties(t::GRASS, TileProperties::GROUND);
    map.set_properties(t::SOIL, TileProperties::GROUND);
    // A path is cheaper to walk, which is what makes NPCs follow roads.
    map.set_properties(
        t::PATH,
        TileProperties {
            extra_cost: 0,
            ..TileProperties::GROUND
        },
    );
    map.set_properties(t::WATER, TileProperties::WATER);
    map.set_properties(t::STONE, TileProperties::WALL);
    map.set_properties(t::TREE, TileProperties::WALL);
    map.set_properties(t::ROCK, TileProperties::WALL);
    map.set_properties(t::CAVE_FLOOR, TileProperties::GROUND);
    map.set_properties(t::CAVE_WALL, TileProperties::WALL);
    map.set_properties(t::ORE, TileProperties::WALL);
    map.set_properties(t::LADDER, TileProperties::GROUND);
    map.set_properties(t::BUILDING, TileProperties::WALL);
    map.set_properties(t::DOOR, TileProperties::GROUND);
    map.set_properties(t::BRIDGE, TileProperties::GROUND);
    // Grass costs a little more than a path, so NPCs prefer the road without
    // being forbidden from cutting across a field.
    map.set_properties(
        t::GRASS,
        TileProperties {
            extra_cost: 4,
            ..TileProperties::GROUND
        },
    );
}

/// Where things are in the generated valley.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValleyLayout {
    /// Where the player starts and returns to sleep.
    pub farmhouse_door: IVec2,
    /// The rectangle of tillable land around the farmhouse.
    pub farm_area: (IVec2, IVec2),
    /// The shop's doorway.
    pub shop_door: IVec2,
    /// The entrance to the mine.
    pub mine_entrance: IVec2,
    /// Tiles adjacent to water, where fishing is possible.
    pub fishing_spots: Vec<IVec2>,
    /// Where each villager's home is.
    pub homes: Vec<IVec2>,
    /// The village square, where villagers gather in the afternoon.
    pub square: IVec2,
}

/// A generated valley: its tiles and where everything is.
pub struct Valley {
    /// The tile map.
    pub map: Tilemap,
    /// Landmark positions.
    pub layout: ValleyLayout,
}

/// Generates the valley.
///
/// The layout is fixed and the *detail* is seeded: the farm is always west of
/// the river and the shop always east, so a player's mental map of the place
/// survives between saves, while the trees, rocks and river's exact course
/// differ per world.
///
/// # Panics
///
/// Panics if the generated layout leaves the farmhouse unreachable, which would
/// mean the generator itself is broken rather than the seed being unlucky.
#[must_use]
pub fn generate_valley(seed: u64) -> Valley {
    let mut rng = Rng::new(seed).derive("valley");
    let mut map = Tilemap::new(VALLEY_WIDTH, VALLEY_HEIGHT, TILE_SIZE);
    register_properties(&mut map);

    let mut ground = TileLayer::new("ground", VALLEY_WIDTH, VALLEY_HEIGHT);
    ground.clear_to(tiles::GRASS);

    // ---- The river ---------------------------------------------------------
    // A meandering vertical river splits the valley. Its course comes from
    // noise rather than a random walk, so it curves smoothly instead of
    // jittering pixel by pixel.
    let noise = ValueNoise::new(u32::try_from(seed & 0xFFFF_FFFF).unwrap_or(0));
    let river_settings = verdant_core_math::FbmSettings {
        octaves: 3,
        frequency: Fx::from_ratio(1, 24),
        ..verdant_core_math::FbmSettings::default()
    };
    // Centred between the farm on the west bank and the village on the east,
    // with a meander narrow enough that it cannot wander into either. A river
    // that crosses a settlement cuts its houses off from their own road.
    let river_centre = (VALLEY_WIDTH as i32) / 2;
    for y in 0..VALLEY_HEIGHT as i32 {
        let wobble = fbm(noise, Fx::from_num(y), Fx::ZERO, river_settings);
        let offset = ((wobble - Fx::HALF) * 10).round_int();
        let x = river_centre + offset;
        for width in -1..=1 {
            ground.set(IVec2::new(x + width, y), tiles::WATER);
        }
    }

    // ---- Landmarks ---------------------------------------------------------
    let farmhouse_door = IVec2::new(10, 12);
    let farm_area = (IVec2::new(4, 14), IVec2::new(22, 34));
    // Well clear of the river's widest meander, which reaches x = 32 +/- 5.
    let square = IVec2::new((VALLEY_WIDTH as i32) - 16, 22);
    let shop_door = IVec2::new(square.x + 5, square.y - 3);
    let mine_entrance = IVec2::new((VALLEY_WIDTH as i32) - 5, 6);

    // ---- Scatter trees and rocks over the west bank ------------------------
    // Kept clear of the farm plot and the paths, so a new player is not walled
    // in by their own scenery.
    for y in 2..(VALLEY_HEIGHT as i32) - 2 {
        for x in 2..(VALLEY_WIDTH as i32) - 2 {
            let cell = IVec2::new(x, y);
            if ground.get(cell) != tiles::GRASS {
                continue;
            }
            let in_farm = x >= farm_area.0.x - 1
                && x <= farm_area.1.x + 1
                && y >= farm_area.0.y - 1
                && y <= farm_area.1.y + 1;
            let near_landmark = cell.chebyshev_distance(farmhouse_door) < 4
                || cell.chebyshev_distance(square) < 5
                || cell.chebyshev_distance(shop_door) < 3
                || cell.chebyshev_distance(mine_entrance) < 3;
            if in_farm || near_landmark {
                continue;
            }

            if rng.chance(9, 100) {
                ground.set(cell, tiles::TREE);
            } else if rng.chance(4, 100) {
                ground.set(cell, tiles::ROCK);
            }
        }
    }

    // ---- Buildings ---------------------------------------------------------
    // Drawn after the scatter so nothing overlaps them.
    let mut homes = Vec::new();
    build_house(&mut ground, farmhouse_door, 6, 4);
    build_house(&mut ground, shop_door, 7, 5);
    for index in 0..4i32 {
        let door = IVec2::new(square.x - 6 + index * 4, square.y + 6);
        build_house(&mut ground, door, 3, 3);
        homes.push(door);
    }

    // ---- Paths -------------------------------------------------------------
    // A road from the farmhouse to the square, and on to the mine. Carving it
    // last means it cuts through whatever the scatter placed.
    carve_path(&mut ground, farmhouse_door, square);
    carve_path(&mut ground, square, mine_entrance);
    for home in &homes {
        carve_path(&mut ground, *home, square);
    }

    // ---- Fishing spots -----------------------------------------------------
    // Land tiles adjacent to water, which is where a rod can be cast from.
    let mut fishing_spots = Vec::new();
    for y in 1..(VALLEY_HEIGHT as i32) - 1 {
        for x in 1..(VALLEY_WIDTH as i32) - 1 {
            let cell = IVec2::new(x, y);
            if !matches!(ground.get(cell), tiles::GRASS | tiles::PATH | tiles::BRIDGE) {
                continue;
            }
            let beside_water = IVec2::CARDINALS
                .iter()
                .any(|offset| ground.get(cell + *offset) == tiles::WATER);
            if beside_water {
                fishing_spots.push(cell);
            }
        }
    }

    map.push_layer(ground);

    Valley {
        map,
        layout: ValleyLayout {
            farmhouse_door,
            farm_area,
            shop_door,
            mine_entrance,
            fishing_spots,
            homes,
            square,
        },
    }
}

/// Stamps a building with its doorway at `door`.
///
/// The door tile itself stays walkable, so a building is always enterable —
/// a generated house you cannot get into is a bug the player experiences as a
/// dead end.
fn build_house(layer: &mut TileLayer, door: IVec2, width: i32, height: i32) {
    let left = door.x - width / 2;
    let top = door.y - height;
    for y in top..door.y {
        for x in left..left + width {
            layer.set(IVec2::new(x, y), tiles::BUILDING);
        }
    }
    layer.set(door, tiles::DOOR);
}

/// Carves an L-shaped path between two points.
///
/// Horizontal then vertical, rather than a diagonal, because a tile path reads
/// as a road when it runs along the grid and as a staircase when it does not.
fn carve_path(layer: &mut TileLayer, from: IVec2, to: IVec2) {
    let step = |a: i32, b: i32| if a < b { 1 } else { -1 };

    let mut x = from.x;
    while x != to.x {
        paint_path(layer, IVec2::new(x, from.y));
        x += step(x, to.x);
    }
    let mut y = from.y;
    while y != to.y {
        paint_path(layer, IVec2::new(to.x, y));
        y += step(y, to.y);
    }
    paint_path(layer, to);
}

/// Paints one path tile.
///
/// Water becomes a bridge rather than being skipped: a road that stops at the
/// bank is a road to nowhere, and carving bridges as a separate pass only works
/// if it happens to guess the row the road actually crosses on.
fn paint_path(layer: &mut TileLayer, cell: IVec2) {
    let existing = layer.get(cell);
    // A doorway is already walkable and must keep its own tile.
    if existing == tiles::DOOR {
        return;
    }
    let tile = if existing == tiles::WATER {
        tiles::BRIDGE
    } else {
        tiles::PATH
    };
    layer.set(cell, tile);
}

/// One level of the mine.
pub struct MineLevel {
    /// The level's tiles.
    pub map: Tilemap,
    /// Where the player arrives.
    pub entrance: IVec2,
    /// Where the ladder down is.
    pub ladder: IVec2,
    /// Where creatures start.
    pub spawns: Vec<IVec2>,
}

/// Mine level width in tiles.
pub const MINE_WIDTH: u32 = 40;
/// Mine level height in tiles.
pub const MINE_HEIGHT: u32 = 32;

/// Generates one level of the mine.
///
/// Cellular-automaton cave carving: start from noise, then repeatedly replace
/// each cell with the majority of its neighbourhood. Four passes turns static
/// into rounded, connected caverns — the standard technique, and far better
/// suited to organic caves than room-and-corridor generation.
///
/// The entrance and the ladder are always connected, which the function
/// guarantees by carving a corridor between them rather than by rejecting
/// unlucky seeds.
///
/// # Panics
///
/// Panics if the level dimensions are zero, which is a programming error.
#[must_use]
pub fn generate_mine_level(seed: u64, depth: u32) -> MineLevel {
    let mut rng = Rng::new(seed).derive(&format!("mine-{depth}"));
    let mut map = Tilemap::new(MINE_WIDTH, MINE_HEIGHT, TILE_SIZE);
    register_properties(&mut map);

    let width = MINE_WIDTH as i32;
    let height = MINE_HEIGHT as i32;

    // Seed the grid: true means solid rock.
    let mut solid = vec![false; (MINE_WIDTH * MINE_HEIGHT) as usize];
    let index_of = |cell: IVec2| (cell.y as usize) * (MINE_WIDTH as usize) + (cell.x as usize);
    for y in 0..height {
        for x in 0..width {
            let cell = IVec2::new(x, y);
            // The border is always solid, so a level is a closed space.
            let is_border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
            // Deeper levels start denser, so the mine narrows as it descends.
            let fill = 42 + depth.min(20);
            solid[index_of(cell)] = is_border || rng.chance(fill, 100);
        }
    }

    // Smooth. Each pass replaces a cell with the majority of its 3x3
    // neighbourhood, which erodes single-pixel noise into caverns.
    for _ in 0..4 {
        let previous = solid.clone();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let cell = IVec2::new(x, y);
                let mut walls = 0;
                for offset in IVec2::NEIGHBOURS {
                    if previous[index_of(cell + offset)] {
                        walls += 1;
                    }
                }
                solid[index_of(cell)] = walls >= 5;
            }
        }
    }

    let entrance = IVec2::new(3, 3);
    let ladder = IVec2::new(width - 4, height - 4);

    // Guarantee a route. Carving is cheaper and more reliable than rejecting
    // seeds until one happens to connect.
    carve_corridor(&mut solid, index_of, entrance, ladder);
    for offset in IVec2::NEIGHBOURS {
        solid[index_of(entrance + offset)] = false;
        solid[index_of(ladder + offset)] = false;
    }

    let mut layer = TileLayer::new("cave", MINE_WIDTH, MINE_HEIGHT);
    let mut spawns = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let cell = IVec2::new(x, y);
            if solid[index_of(cell)] {
                // Ore appears in walls, more often the deeper the level.
                let ore_chance = 3 + depth.min(12);
                let tile = if rng.chance(ore_chance, 100) {
                    tiles::ORE
                } else {
                    tiles::CAVE_WALL
                };
                layer.set(cell, tile);
            } else {
                layer.set(cell, tiles::CAVE_FLOOR);
                // Creatures spawn away from the entrance, so arriving is not
                // an immediate ambush.
                if cell.manhattan_distance(entrance) > 8 && rng.chance(3, 100) {
                    spawns.push(cell);
                }
            }
        }
    }
    layer.set(ladder, tiles::LADDER);

    map.push_layer(layer);
    MineLevel {
        map,
        entrance,
        ladder,
        spawns,
    }
}

/// Carves a two-tile-wide corridor between two points.
fn carve_corridor(solid: &mut [bool], index_of: impl Fn(IVec2) -> usize, from: IVec2, to: IVec2) {
    let step = |a: i32, b: i32| if a < b { 1 } else { -1 };
    let mut carve = |cell: IVec2| {
        for offset in [IVec2::ZERO, IVec2::new(1, 0), IVec2::new(0, 1)] {
            let target = cell + offset;
            if target.x > 0
                && target.y > 0
                && target.x < (MINE_WIDTH as i32) - 1
                && target.y < (MINE_HEIGHT as i32) - 1
            {
                solid[index_of(target)] = false;
            }
        }
    };

    let mut x = from.x;
    while x != to.x {
        carve(IVec2::new(x, from.y));
        x += step(x, to.x);
    }
    let mut y = from.y;
    while y != to.y {
        carve(IVec2::new(to.x, y));
        y += step(y, to.y);
    }
    carve(to);
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_tilemap::{find_path, Movement};

    #[test]
    fn the_valley_generates_at_the_expected_size() {
        let valley = generate_valley(1);
        assert_eq!(valley.map.width(), VALLEY_WIDTH);
        assert_eq!(valley.map.height(), VALLEY_HEIGHT);
    }

    #[test]
    fn generation_is_deterministic() {
        let first = generate_valley(42);
        let second = generate_valley(42);
        let ground = |valley: &Valley| {
            valley
                .map
                .layer("ground")
                .expect("the ground layer exists")
                .tiles()
                .to_vec()
        };
        assert_eq!(ground(&first), ground(&second));
    }

    #[test]
    fn different_seeds_produce_different_valleys() {
        let first = generate_valley(1);
        let second = generate_valley(2);
        let ground = |valley: &Valley| {
            valley
                .map
                .layer("ground")
                .expect("the ground layer exists")
                .tiles()
                .to_vec()
        };
        assert_ne!(ground(&first), ground(&second));
    }

    #[test]
    fn landmarks_stay_in_the_same_place_across_seeds() {
        // The layout is fixed on purpose: a player's mental map of the valley
        // should survive starting a new save.
        let first = generate_valley(1);
        let second = generate_valley(9999);
        assert_eq!(first.layout.farmhouse_door, second.layout.farmhouse_door);
        assert_eq!(first.layout.square, second.layout.square);
        assert_eq!(first.layout.mine_entrance, second.layout.mine_entrance);
    }

    #[test]
    fn every_landmark_is_walkable() {
        for seed in 0..8u64 {
            let valley = generate_valley(seed);
            let layout = &valley.layout;
            for (name, cell) in [
                ("farmhouse door", layout.farmhouse_door),
                ("shop door", layout.shop_door),
                ("mine entrance", layout.mine_entrance),
                ("square", layout.square),
            ] {
                assert!(
                    valley.map.is_walkable(cell),
                    "seed {seed}: the {name} at {cell:?} is not walkable"
                );
            }
        }
    }

    #[test]
    fn the_player_can_walk_from_home_to_the_shop_and_the_mine() {
        // The single most important property of a generated world: everywhere
        // the game sends the player must be reachable.
        for seed in 0..8u64 {
            let valley = generate_valley(seed);
            let layout = &valley.layout;

            for (name, destination) in [
                ("the square", layout.square),
                ("the shop", layout.shop_door),
                ("the mine", layout.mine_entrance),
            ] {
                let route = find_path(
                    &valley.map,
                    layout.farmhouse_door,
                    destination,
                    Movement::Orthogonal,
                    40_000,
                );
                assert!(route.is_some(), "seed {seed}: no route from home to {name}");
            }
        }
    }

    #[test]
    fn every_home_is_reachable_from_the_square() {
        for seed in 0..4u64 {
            let valley = generate_valley(seed);
            for home in &valley.layout.homes {
                let route = find_path(
                    &valley.map,
                    valley.layout.square,
                    *home,
                    Movement::Orthogonal,
                    40_000,
                );
                assert!(
                    route.is_some(),
                    "seed {seed}: {home:?} is cut off from the square"
                );
            }
        }
    }

    #[test]
    fn the_farm_plot_is_clear_of_scenery() {
        for seed in 0..6u64 {
            let valley = generate_valley(seed);
            let (min, max) = valley.layout.farm_area;
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let cell = IVec2::new(x, y);
                    assert!(
                        valley.map.is_walkable(cell),
                        "seed {seed}: {cell:?} in the farm plot is blocked"
                    );
                }
            }
        }
    }

    #[test]
    fn the_valley_has_water_to_fish_in() {
        for seed in 0..6u64 {
            let valley = generate_valley(seed);
            assert!(
                !valley.layout.fishing_spots.is_empty(),
                "seed {seed}: nowhere to fish"
            );
            // And every listed spot really is beside water.
            let ground = valley.map.layer("ground").unwrap();
            for spot in &valley.layout.fishing_spots {
                let beside_water = IVec2::CARDINALS
                    .iter()
                    .any(|offset| ground.get(*spot + *offset) == tiles::WATER);
                assert!(beside_water, "{spot:?} is not beside water");
            }
        }
    }

    #[test]
    fn a_bridge_crosses_the_river() {
        // Without one, the road to the shop would end at the bank.
        for seed in 0..6u64 {
            let valley = generate_valley(seed);
            let route = find_path(
                &valley.map,
                valley.layout.farmhouse_door,
                valley.layout.square,
                Movement::Orthogonal,
                40_000,
            );
            assert!(route.is_some(), "seed {seed}: the river is uncrossable");
        }
    }

    #[test]
    fn a_mine_level_generates_and_connects() {
        for depth in 0..12u32 {
            let level = generate_mine_level(7, depth);
            assert!(
                level.map.is_walkable(level.entrance),
                "depth {depth}: no entrance"
            );
            assert!(
                level.map.is_walkable(level.ladder),
                "depth {depth}: no ladder"
            );

            let route = find_path(
                &level.map,
                level.entrance,
                level.ladder,
                Movement::Orthogonal,
                40_000,
            );
            assert!(route.is_some(), "depth {depth}: the ladder is unreachable");
        }
    }

    #[test]
    fn mine_levels_are_enclosed() {
        let level = generate_mine_level(3, 1);
        let width = MINE_WIDTH as i32;
        let height = MINE_HEIGHT as i32;
        for x in 0..width {
            assert!(!level.map.is_walkable(IVec2::new(x, 0)));
            assert!(!level.map.is_walkable(IVec2::new(x, height - 1)));
        }
        for y in 0..height {
            assert!(!level.map.is_walkable(IVec2::new(0, y)));
            assert!(!level.map.is_walkable(IVec2::new(width - 1, y)));
        }
    }

    #[test]
    fn creatures_never_spawn_on_top_of_the_entrance() {
        for depth in 0..8u32 {
            let level = generate_mine_level(11, depth);
            for spawn in &level.spawns {
                assert!(
                    spawn.manhattan_distance(level.entrance) > 8,
                    "depth {depth}: a creature spawned at the entrance"
                );
                assert!(
                    level.map.is_walkable(*spawn),
                    "a creature spawned inside rock"
                );
            }
        }
    }

    #[test]
    fn deeper_levels_hold_more_ore() {
        let count_ore = |depth: u32| {
            let level = generate_mine_level(5, depth);
            level
                .map
                .layer("cave")
                .expect("the cave layer exists")
                .tiles()
                .iter()
                .filter(|tile| **tile == tiles::ORE)
                .count()
        };
        assert!(
            count_ore(10) > count_ore(0),
            "the mine should get richer with depth"
        );
    }

    #[test]
    fn mine_generation_is_deterministic() {
        let tiles_of = |seed: u64, depth: u32| {
            generate_mine_level(seed, depth)
                .map
                .layer("cave")
                .expect("the cave layer exists")
                .tiles()
                .to_vec()
        };
        assert_eq!(tiles_of(99, 3), tiles_of(99, 3));
        assert_ne!(tiles_of(99, 3), tiles_of(99, 4), "each level differs");
    }
}
