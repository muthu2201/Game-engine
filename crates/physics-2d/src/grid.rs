//! The interface between the solver and whatever holds the world's tiles.

use verdant_core_math::{Fx, IVec2};

/// A source of tile-based solidity.
///
/// The solver needs to ask "is this cell solid?" without knowing how the map is
/// stored — the game's tilemap, a procedurally generated cave, or a test
/// fixture all answer the same question. Keeping it a trait is also what lets
/// the collision tests run against hand-built layouts.
pub trait SolidGrid {
    /// The world-space size of one tile. Must be positive.
    fn tile_size(&self) -> Fx;

    /// True when `cell` blocks movement.
    ///
    /// Cells outside the map should return `true` for a closed world, which is
    /// what keeps a body from walking off the edge into unallocated space.
    fn is_solid(&self, cell: IVec2) -> bool;
}

/// A grid where nothing is ever solid.
///
/// Used for entities that move in open space — projectiles above the ground
/// layer, UI-space effects — so they can share the solver.
#[derive(Clone, Copy, Debug)]
pub struct OpenGrid {
    tile_size: Fx,
}

impl OpenGrid {
    /// Creates an open grid with the given tile size.
    #[inline]
    #[must_use]
    pub const fn new(tile_size: Fx) -> OpenGrid {
        OpenGrid { tile_size }
    }
}

impl SolidGrid for OpenGrid {
    fn tile_size(&self) -> Fx {
        self.tile_size
    }

    fn is_solid(&self, _cell: IVec2) -> bool {
        false
    }
}

/// A grid backed by an explicit list of solid cells.
///
/// Available outside `cfg(test)` because the procedural cave generator builds
/// candidate layouts with it before committing them to a real tilemap.
#[derive(Clone, Debug)]
pub struct TestGrid {
    tile_size: Fx,
    solids: Vec<IVec2>,
}

impl TestGrid {
    /// A grid with no solid cells.
    #[must_use]
    pub fn empty(tile_size: Fx) -> TestGrid {
        TestGrid {
            tile_size,
            solids: Vec::new(),
        }
    }

    /// A grid with the listed cells solid.
    #[must_use]
    pub fn with_solids(tile_size: Fx, solids: &[IVec2]) -> TestGrid {
        TestGrid {
            tile_size,
            solids: solids.to_vec(),
        }
    }

    /// Marks a cell solid.
    pub fn set_solid(&mut self, cell: IVec2) {
        if !self.solids.contains(&cell) {
            self.solids.push(cell);
        }
    }
}

impl SolidGrid for TestGrid {
    fn tile_size(&self) -> Fx {
        self.tile_size
    }

    fn is_solid(&self, cell: IVec2) -> bool {
        self.solids.contains(&cell)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    #[test]
    fn an_open_grid_is_never_solid() {
        let grid = OpenGrid::new(fx(16));
        assert_eq!(grid.tile_size(), fx(16));
        assert!(!grid.is_solid(IVec2::new(0, 0)));
        assert!(!grid.is_solid(IVec2::new(-99, 99)));
    }

    #[test]
    fn a_test_grid_reports_only_its_listed_cells() {
        let mut grid = TestGrid::with_solids(fx(8), &[IVec2::new(1, 1)]);
        assert!(grid.is_solid(IVec2::new(1, 1)));
        assert!(!grid.is_solid(IVec2::new(0, 0)));

        grid.set_solid(IVec2::new(0, 0));
        grid.set_solid(IVec2::new(0, 0));
        assert!(grid.is_solid(IVec2::new(0, 0)));
    }
}
