//! Wave function collapse, simple tiled model.
//!
//! Given a set of tiles and which tiles may sit next to which, WFC fills a grid
//! so that every adjacency is satisfied. The engine uses it for cave and ruin
//! layouts: the generator states the *rules* ("a corridor connects to a room or
//! another corridor, never to solid rock") and WFC produces layouts that obey
//! them, rather than a hand-written generator that must be re-tuned whenever a
//! tile is added.
//!
//! # The algorithm
//!
//! Every cell starts as a superposition of all tiles. Then, repeatedly:
//!
//! 1. **Observe** — pick the cell with the fewest remaining options (lowest
//!    entropy) and collapse it to one tile, weighted by frequency.
//! 2. **Propagate** — remove now-impossible options from its neighbours, and
//!    from their neighbours, until nothing changes.
//!
//! # Contradictions
//!
//! Propagation can leave a cell with no legal tile. That is not a bug; it is
//! inherent to the method. [`Wfc::run`] restarts from the seed with a different
//! sub-stream, up to a bounded number of attempts, and reports failure rather
//! than looping forever — a generator that hangs is far worse than one that
//! says "these rules are too tight".

use verdant_core_math::{IVec2, Rng};

/// Which side of a cell an adjacency rule applies to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// Toward negative y.
    North,
    /// Toward positive x.
    East,
    /// Toward positive y.
    South,
    /// Toward negative x.
    West,
}

impl Direction {
    /// All four directions, in a fixed order.
    pub const ALL: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    /// The offset to the neighbour in this direction.
    #[must_use]
    pub const fn offset(self) -> IVec2 {
        match self {
            Direction::North => IVec2::new(0, -1),
            Direction::East => IVec2::new(1, 0),
            Direction::South => IVec2::new(0, 1),
            Direction::West => IVec2::new(-1, 0),
        }
    }

    /// The direction pointing back the other way.
    #[must_use]
    pub const fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }

    /// Index into a per-direction array.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The tiles and adjacency rules WFC works from.
#[derive(Clone, Debug, Default)]
pub struct TileSet {
    /// How often each tile should appear, relative to the others.
    weights: Vec<u32>,
    /// `allowed[tile][direction]` is the bitset of tiles permitted on that side.
    ///
    /// A bitset (rather than a `Vec<usize>`) makes propagation an intersection
    /// of machine words, which is what keeps the inner loop fast enough to fill
    /// a cave every time the player descends.
    allowed: Vec<[u64; 4]>,
}

impl TileSet {
    /// Creates a tile set where nothing may sit next to anything.
    ///
    /// # Panics
    ///
    /// Panics if `count` exceeds 64, the width of the adjacency bitset.
    #[must_use]
    pub fn new(count: usize) -> TileSet {
        assert!(
            count <= 64,
            "TileSet: at most 64 tiles are supported, got {count}"
        );
        TileSet {
            weights: vec![1; count],
            allowed: vec![[0; 4]; count],
        }
    }

    /// Number of tiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True when the set holds no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Sets how often a tile should appear relative to the others.
    ///
    /// # Panics
    ///
    /// Panics if `tile` is out of range.
    pub fn set_weight(&mut self, tile: usize, weight: u32) {
        assert!(
            tile < self.len(),
            "TileSet::set_weight: tile {tile} is out of range"
        );
        self.weights[tile] = weight;
    }

    /// Permits `neighbour` to sit on the `direction` side of `tile`.
    ///
    /// The mirrored rule is added automatically: if A may sit east of B, then B
    /// may sit west of A. Requiring both to be stated by hand is a reliable
    /// source of asymmetric rule sets that produce contradictions.
    ///
    /// # Panics
    ///
    /// Panics if either tile index is out of range.
    pub fn allow(&mut self, tile: usize, direction: Direction, neighbour: usize) {
        assert!(
            tile < self.len() && neighbour < self.len(),
            "TileSet::allow: tile index out of range"
        );
        self.allowed[tile][direction.index()] |= 1 << neighbour;
        self.allowed[neighbour][direction.opposite().index()] |= 1 << tile;
    }

    /// Permits `tile` and `neighbour` to sit next to each other in every
    /// direction.
    pub fn allow_all_directions(&mut self, tile: usize, neighbour: usize) {
        for direction in Direction::ALL {
            self.allow(tile, direction, neighbour);
        }
    }

    /// The bitset of tiles permitted on one side of `tile`.
    #[must_use]
    fn permitted(&self, tile: usize, direction: Direction) -> u64 {
        self.allowed[tile][direction.index()]
    }

    /// A bitset with every tile set.
    #[must_use]
    fn full_mask(&self) -> u64 {
        if self.len() >= 64 {
            u64::MAX
        } else {
            (1u64 << self.len()) - 1
        }
    }
}

/// Why a WFC run failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WfcError {
    /// Every attempt hit a contradiction.
    ///
    /// Usually means the rules are too tight for the grid — a tile with no
    /// legal neighbour on some side, or two regions that cannot meet.
    Contradiction {
        /// How many attempts were made.
        attempts: u32,
    },
    /// The tile set was empty, so there is nothing to place.
    EmptyTileSet,
}

impl std::fmt::Display for WfcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WfcError::Contradiction { attempts } => {
                write!(f, "wave function collapse failed after {attempts} attempts")
            }
            WfcError::EmptyTileSet => write!(f, "wave function collapse needs at least one tile"),
        }
    }
}

impl std::error::Error for WfcError {}

/// A grid being collapsed.
pub struct Wfc<'a> {
    tiles: &'a TileSet,
    width: u32,
    height: u32,
    /// Remaining options per cell, as a bitset.
    wave: Vec<u64>,
}

impl<'a> Wfc<'a> {
    /// Prepares a grid where every cell may be any tile.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn new(tiles: &'a TileSet, width: u32, height: u32) -> Wfc<'a> {
        assert!(width > 0 && height > 0, "Wfc: dimensions must be non-zero");
        let mask = tiles.full_mask();
        Wfc {
            tiles,
            width,
            height,
            wave: vec![mask; (width as usize) * (height as usize)],
        }
    }

    /// Forces a cell to a specific tile before collapsing.
    ///
    /// This is how a generator pins entrances, exits and pre-placed rooms, and
    /// lets WFC fill only what is left.
    ///
    /// # Panics
    ///
    /// Panics if the cell is out of range.
    pub fn constrain(&mut self, cell: IVec2, tile: usize) {
        let index = self
            .index_of(cell)
            .expect("Wfc::constrain: cell out of range");
        self.wave[index] = 1 << tile;
    }

    /// Flat index of a cell, or `None` when out of range.
    fn index_of(&self, cell: IVec2) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || (cell.x as u32) >= self.width
            || (cell.y as u32) >= self.height
        {
            return None;
        }
        Some((cell.y as usize) * (self.width as usize) + (cell.x as usize))
    }

    /// The cell at a flat index.
    fn cell_of(&self, index: usize) -> IVec2 {
        let index = u32::try_from(index).unwrap_or(u32::MAX);
        IVec2::new((index % self.width) as i32, (index / self.width) as i32)
    }

    /// Collapses the grid, retrying on contradiction.
    ///
    /// Returns one tile index per cell, row-major.
    ///
    /// # Errors
    ///
    /// Returns [`WfcError::Contradiction`] when every attempt fails, and
    /// [`WfcError::EmptyTileSet`] when there are no tiles to place.
    pub fn run(&self, rng: &mut Rng, attempts: u32) -> Result<Vec<usize>, WfcError> {
        if self.tiles.is_empty() {
            return Err(WfcError::EmptyTileSet);
        }
        for attempt in 0..attempts.max(1) {
            // Each attempt draws from its own stream, so a retry explores a
            // genuinely different layout while the whole run stays a pure
            // function of the caller's seed.
            let mut attempt_rng = rng.derive(&format!("wfc-attempt-{attempt}"));
            if let Some(result) = self.attempt(&mut attempt_rng) {
                return Ok(result);
            }
        }
        Err(WfcError::Contradiction {
            attempts: attempts.max(1),
        })
    }

    /// One collapse attempt, or `None` on contradiction.
    fn attempt(&self, rng: &mut Rng) -> Option<Vec<usize>> {
        let mut wave = self.wave.clone();
        // The initial constraints must be propagated before the first choice,
        // or a pinned cell would not restrict its neighbours.
        self.propagate_all(&mut wave)?;

        loop {
            let Some(index) = self.lowest_entropy(&mut wave.clone(), &wave) else {
                // Nothing left to collapse: read out the result.
                return wave
                    .iter()
                    .map(|options| {
                        if options.count_ones() == 1 {
                            Some(options.trailing_zeros() as usize)
                        } else {
                            None
                        }
                    })
                    .collect();
            };

            let chosen = self.choose(wave[index], rng)?;
            wave[index] = 1 << chosen;
            self.propagate_from(&mut wave, index)?;
        }
    }

    /// The undecided cell with the fewest options, or `None` when all are
    /// decided.
    ///
    /// Ties break by the lowest flat index, so the traversal order is a
    /// function of the wave rather than of iteration order.
    fn lowest_entropy(&self, _scratch: &mut [u64], wave: &[u64]) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for (index, options) in wave.iter().enumerate() {
            let count = options.count_ones();
            if count <= 1 {
                continue;
            }
            if best.is_none_or(|(best_count, _)| count < best_count) {
                best = Some((count, index));
            }
        }
        best.map(|(_, index)| index)
    }

    /// Picks one tile from a bitset, weighted by frequency.
    fn choose(&self, options: u64, rng: &mut Rng) -> Option<usize> {
        let candidates: Vec<usize> = (0..self.tiles.len())
            .filter(|tile| options & (1 << tile) != 0)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let weights: Vec<u32> = candidates
            .iter()
            .map(|tile| self.tiles.weights[*tile].max(1))
            .collect();
        rng.pick_weighted(&weights)
            .map(|position| candidates[position])
    }

    /// Propagates constraints from every cell, for the initial pass.
    fn propagate_all(&self, wave: &mut [u64]) -> Option<()> {
        for index in 0..wave.len() {
            self.propagate_from(wave, index)?;
        }
        Some(())
    }

    /// Removes now-impossible options, spreading outward from one cell.
    fn propagate_from(&self, wave: &mut [u64], origin: usize) -> Option<()> {
        let mut pending = vec![origin];
        while let Some(index) = pending.pop() {
            let options = wave[index];
            if options == 0 {
                return None;
            }
            let cell = self.cell_of(index);

            for direction in Direction::ALL {
                let Some(neighbour_index) = self.index_of(cell + direction.offset()) else {
                    continue;
                };
                // The union of what each surviving option permits on this side.
                let mut permitted = 0u64;
                for tile in 0..self.tiles.len() {
                    if options & (1 << tile) != 0 {
                        permitted |= self.tiles.permitted(tile, direction);
                    }
                }
                let before = wave[neighbour_index];
                let after = before & permitted;
                if after == 0 {
                    return None;
                }
                if after != before {
                    wave[neighbour_index] = after;
                    pending.push(neighbour_index);
                }
            }
        }
        Some(())
    }

    /// Grid width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Grid height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tiles where ground and wall may not touch directly; a shore tile
    /// must separate them. Tight enough to exercise propagation, loose enough
    /// to always have a solution.
    fn shore_rules() -> TileSet {
        const GROUND: usize = 0;
        const SHORE: usize = 1;
        const WALL: usize = 2;
        let mut tiles = TileSet::new(3);
        tiles.allow_all_directions(GROUND, GROUND);
        tiles.allow_all_directions(GROUND, SHORE);
        tiles.allow_all_directions(SHORE, SHORE);
        tiles.allow_all_directions(SHORE, WALL);
        tiles.allow_all_directions(WALL, WALL);
        tiles
    }

    #[test]
    fn a_permissive_rule_set_fills_the_grid() {
        let mut tiles = TileSet::new(2);
        tiles.allow_all_directions(0, 0);
        tiles.allow_all_directions(0, 1);
        tiles.allow_all_directions(1, 1);

        let wfc = Wfc::new(&tiles, 8, 8);
        let result = wfc
            .run(&mut Rng::new(1), 10)
            .expect("permissive rules always succeed");
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|tile| *tile < 2));
    }

    #[test]
    fn the_output_satisfies_every_adjacency_rule() {
        let tiles = shore_rules();
        let wfc = Wfc::new(&tiles, 12, 12);
        let result = wfc
            .run(&mut Rng::new(7), 20)
            .expect("the shore rules have solutions");

        for y in 0..12i32 {
            for x in 0..12i32 {
                let here = result[(y as usize) * 12 + (x as usize)];
                for direction in Direction::ALL {
                    let neighbour = IVec2::new(x, y) + direction.offset();
                    if neighbour.x < 0 || neighbour.y < 0 || neighbour.x >= 12 || neighbour.y >= 12
                    {
                        continue;
                    }
                    let there = result[(neighbour.y as usize) * 12 + (neighbour.x as usize)];
                    assert!(
                        tiles.permitted(here, direction) & (1 << there) != 0,
                        "tile {here} at ({x}, {y}) illegally borders {there} to the {direction:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn constraints_are_respected_and_propagated() {
        let tiles = shore_rules();
        let mut wfc = Wfc::new(&tiles, 6, 6);
        wfc.constrain(IVec2::new(0, 0), 0); // ground
        wfc.constrain(IVec2::new(5, 5), 2); // wall

        let result = wfc.run(&mut Rng::new(3), 20).expect("a solution exists");
        assert_eq!(result[0], 0, "the pinned ground must survive");
        assert_eq!(result[35], 2, "the pinned wall must survive");
        // Ground cannot touch wall, so the pinned ground's neighbours cannot be
        // wall either.
        assert_ne!(result[1], 2);
        assert_ne!(result[6], 2);
    }

    #[test]
    fn an_impossible_rule_set_reports_a_contradiction() {
        // Two tiles, neither of which may neighbour anything.
        let tiles = TileSet::new(2);
        let wfc = Wfc::new(&tiles, 4, 4);
        let error = wfc
            .run(&mut Rng::new(1), 3)
            .expect_err("nothing can be placed");
        assert_eq!(error, WfcError::Contradiction { attempts: 3 });
        assert!(error.to_string().contains("3 attempts"));
    }

    #[test]
    fn an_empty_tile_set_is_reported_distinctly() {
        let tiles = TileSet::new(0);
        let wfc = Wfc::new(&tiles, 4, 4);
        assert_eq!(wfc.run(&mut Rng::new(1), 3), Err(WfcError::EmptyTileSet));
    }

    #[test]
    fn contradictory_constraints_fail_rather_than_hanging() {
        let tiles = shore_rules();
        let mut wfc = Wfc::new(&tiles, 2, 1);
        // Ground beside wall, which the rules forbid outright.
        wfc.constrain(IVec2::new(0, 0), 0);
        wfc.constrain(IVec2::new(1, 0), 2);
        assert!(wfc.run(&mut Rng::new(1), 5).is_err());
    }

    #[test]
    fn determinism_the_same_seed_produces_the_same_layout() {
        let tiles = shore_rules();
        let wfc = Wfc::new(&tiles, 10, 10);
        let first = wfc.run(&mut Rng::new(42), 20).unwrap();
        let second = wfc.run(&mut Rng::new(42), 20).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_produce_different_layouts() {
        let tiles = shore_rules();
        let wfc = Wfc::new(&tiles, 10, 10);
        let first = wfc.run(&mut Rng::new(1), 20).unwrap();
        let second = wfc.run(&mut Rng::new(2), 20).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn weights_bias_the_output() {
        let mut tiles = TileSet::new(2);
        tiles.allow_all_directions(0, 0);
        tiles.allow_all_directions(0, 1);
        tiles.allow_all_directions(1, 1);
        tiles.set_weight(0, 50);
        tiles.set_weight(1, 1);

        let wfc = Wfc::new(&tiles, 16, 16);
        let result = wfc.run(&mut Rng::new(11), 10).unwrap();
        let common = result.iter().filter(|tile| **tile == 0).count();
        assert!(
            common > result.len() / 2,
            "the heavily weighted tile should dominate"
        );
    }

    #[test]
    fn rules_are_mirrored_automatically() {
        let mut tiles = TileSet::new(2);
        tiles.allow(0, Direction::East, 1);
        // Stating A-east-of-B must imply B-west-of-A.
        assert!(tiles.permitted(1, Direction::West) & 1 != 0);
    }

    #[test]
    #[should_panic(expected = "at most 64 tiles")]
    fn oversized_tile_sets_are_rejected() {
        let _ = TileSet::new(65);
    }
}
