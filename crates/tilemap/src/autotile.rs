//! Autotiling: choosing a tile's artwork from its neighbours.
//!
//! When a player tills a patch of soil or a cave generator carves a room, the
//! result is a *region*, not a set of individually chosen sprites. Autotiling
//! turns the region into artwork: each cell looks at which of its neighbours
//! belong to the same region and picks the variant with matching edges.
//!
//! # The 47-tile blob set
//!
//! Eight neighbours give 256 combinations, but the four diagonals only matter
//! when both of their adjacent orthogonals are filled — a corner piece is only
//! visible if the two edges meeting there are present. Collapsing the redundant
//! cases leaves **47** distinct tiles, the standard "blob" set that pixel-art
//! tools and tilesets are authored against.
//!
//! This module computes the 8-bit neighbour mask and maps it onto a stable
//! index in `0..47`, so a tileset laid out in the conventional order can be
//! indexed directly.

use verdant_core_math::IVec2;

/// Neighbour bits, in the order the mask packs them.
///
/// North is the low bit and the order proceeds clockwise. Any consistent order
/// works; this one matches the layout most blob tilesets are authored in.
pub mod bits {
    /// The cell above.
    pub const NORTH: u8 = 1 << 0;
    /// Above and to the right.
    pub const NORTH_EAST: u8 = 1 << 1;
    /// To the right.
    pub const EAST: u8 = 1 << 2;
    /// Below and to the right.
    pub const SOUTH_EAST: u8 = 1 << 3;
    /// The cell below.
    pub const SOUTH: u8 = 1 << 4;
    /// Below and to the left.
    pub const SOUTH_WEST: u8 = 1 << 5;
    /// To the left.
    pub const WEST: u8 = 1 << 6;
    /// Above and to the left.
    pub const NORTH_WEST: u8 = 1 << 7;
}

/// Neighbour offsets in the same order as [`bits`].
const NEIGHBOUR_OFFSETS: [IVec2; 8] = [
    IVec2::new(0, -1),
    IVec2::new(1, -1),
    IVec2::new(1, 0),
    IVec2::new(1, 1),
    IVec2::new(0, 1),
    IVec2::new(-1, 1),
    IVec2::new(-1, 0),
    IVec2::new(-1, -1),
];

/// Builds the 8-bit neighbour mask for a cell.
///
/// `belongs` answers "is this cell part of the same region?". Cells outside the
/// map should usually answer `true` for terrain (so a region does not draw an
/// edge against the map border) and `false` for objects.
#[must_use]
pub fn neighbour_mask(cell: IVec2, belongs: impl Fn(IVec2) -> bool) -> u8 {
    let mut mask = 0u8;
    for (index, offset) in NEIGHBOUR_OFFSETS.iter().enumerate() {
        if belongs(cell + *offset) {
            mask |= 1 << index;
        }
    }
    mask
}

/// Discards diagonal bits whose two adjacent orthogonals are not both set.
///
/// This is the reduction that takes 256 combinations down to 47: a corner
/// sprite is only distinguishable when both edges meeting at that corner are
/// filled, so every other diagonal bit is noise.
#[must_use]
pub fn canonical_mask(mask: u8) -> u8 {
    let mut canonical = mask & (bits::NORTH | bits::EAST | bits::SOUTH | bits::WEST);
    if mask & bits::NORTH != 0 && mask & bits::EAST != 0 {
        canonical |= mask & bits::NORTH_EAST;
    }
    if mask & bits::EAST != 0 && mask & bits::SOUTH != 0 {
        canonical |= mask & bits::SOUTH_EAST;
    }
    if mask & bits::SOUTH != 0 && mask & bits::WEST != 0 {
        canonical |= mask & bits::SOUTH_WEST;
    }
    if mask & bits::WEST != 0 && mask & bits::NORTH != 0 {
        canonical |= mask & bits::NORTH_WEST;
    }
    canonical
}

/// The 47 canonical masks, ascending.
///
/// Computed once and searched by binary search, so mask-to-index is a handful
/// of comparisons rather than a 256-entry table that has to be kept in sync
/// with [`canonical_mask`].
static CANONICAL_MASKS: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
    let mut masks: Vec<u8> = (0..=255u8).map(canonical_mask).collect();
    masks.sort_unstable();
    masks.dedup();
    masks
});

/// Number of distinct tiles in a blob set.
pub const BLOB_TILE_COUNT: usize = 47;

/// Maps a neighbour mask onto a tile index in `0..47`.
///
/// The index is stable: it depends only on the canonical mask's numeric value,
/// so a tileset authored against this ordering keeps working as long as the
/// mask convention does.
///
/// # Panics
///
/// Panics only if the canonical-mask table is inconsistent, which would be an
/// internal invariant violation rather than a caller error.
#[must_use]
pub fn blob_index(mask: u8) -> usize {
    let canonical = canonical_mask(mask);
    CANONICAL_MASKS
        .binary_search(&canonical)
        .expect("every canonical mask is present in the table by construction")
}

/// Convenience wrapper: the blob index for a cell, given a membership test.
#[must_use]
pub fn blob_index_at(cell: IVec2, belongs: impl Fn(IVec2) -> bool) -> usize {
    blob_index(neighbour_mask(cell, belongs))
}

/// The simpler 16-tile set, using only the four orthogonal neighbours.
///
/// Enough for fences, paths and anything whose corners are not visually
/// distinct — and a quarter of the artwork to draw.
#[must_use]
pub fn wang_index(mask: u8) -> usize {
    let north = usize::from(mask & bits::NORTH != 0);
    let east = usize::from(mask & bits::EAST != 0);
    let south = usize::from(mask & bits::SOUTH != 0);
    let west = usize::from(mask & bits::WEST != 0);
    north | (east << 1) | (south << 2) | (west << 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn an_isolated_cell_has_no_neighbours() {
        assert_eq!(neighbour_mask(IVec2::ZERO, |_| false), 0);
        assert_eq!(blob_index(0), 0, "the empty mask is the first tile");
    }

    #[test]
    fn a_fully_surrounded_cell_sets_every_bit() {
        assert_eq!(neighbour_mask(IVec2::ZERO, |_| true), 0xFF);
    }

    #[test]
    fn the_mask_records_the_right_directions() {
        // Only the cell directly above belongs.
        let mask = neighbour_mask(IVec2::ZERO, |cell| cell == IVec2::new(0, -1));
        assert_eq!(mask, bits::NORTH);

        let mask = neighbour_mask(IVec2::ZERO, |cell| cell == IVec2::new(-1, 0));
        assert_eq!(mask, bits::WEST);
    }

    #[test]
    fn a_diagonal_without_both_edges_is_discarded() {
        // North-east set, but neither north nor east: the corner is invisible.
        assert_eq!(canonical_mask(bits::NORTH_EAST), 0);
        // With only one edge it is still invisible.
        assert_eq!(canonical_mask(bits::NORTH_EAST | bits::NORTH), bits::NORTH);
        // With both edges it survives.
        let full = bits::NORTH_EAST | bits::NORTH | bits::EAST;
        assert_eq!(canonical_mask(full), full);
    }

    #[test]
    fn there_are_exactly_forty_seven_distinct_tiles() {
        let distinct: HashSet<u8> = (0..=255u8).map(canonical_mask).collect();
        assert_eq!(distinct.len(), BLOB_TILE_COUNT);
    }

    #[test]
    fn every_mask_maps_into_the_tile_range() {
        for mask in 0..=255u8 {
            assert!(
                blob_index(mask) < BLOB_TILE_COUNT,
                "mask {mask} escaped the range"
            );
        }
    }

    #[test]
    fn masks_that_look_the_same_share_an_index() {
        // These differ only in a diagonal bit that cannot be seen.
        let without = bits::NORTH;
        let with = bits::NORTH | bits::SOUTH_EAST;
        assert_eq!(blob_index(without), blob_index(with));
    }

    #[test]
    fn masks_that_look_different_have_different_indices() {
        let indices: HashSet<usize> = [0, bits::NORTH, bits::EAST, bits::NORTH | bits::EAST, 0xFF]
            .iter()
            .map(|mask| blob_index(*mask))
            .collect();
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn indexing_is_stable_across_calls() {
        let first: Vec<usize> = (0..=255u8).map(blob_index).collect();
        let second: Vec<usize> = (0..=255u8).map(blob_index).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn the_wang_set_has_sixteen_entries() {
        let distinct: HashSet<usize> = (0..=255u8).map(wang_index).collect();
        assert_eq!(distinct.len(), 16);
        assert!(distinct.iter().all(|index| *index < 16));
    }

    #[test]
    fn wang_indexing_ignores_diagonals() {
        assert_eq!(
            wang_index(bits::NORTH),
            wang_index(bits::NORTH | bits::SOUTH_EAST)
        );
    }

    #[test]
    fn a_region_edge_picks_a_different_tile_than_its_interior() {
        // A 3x3 filled block: the centre is surrounded, the corners are not.
        let belongs = |cell: IVec2| (0..3).contains(&cell.x) && (0..3).contains(&cell.y);
        let centre = blob_index_at(IVec2::new(1, 1), belongs);
        let corner = blob_index_at(IVec2::new(0, 0), belongs);
        let edge = blob_index_at(IVec2::new(1, 0), belongs);
        assert_ne!(centre, corner);
        assert_ne!(centre, edge);
        assert_ne!(corner, edge);
    }

    #[test]
    fn opposite_corners_of_a_block_use_distinct_tiles() {
        let belongs = |cell: IVec2| (0..3).contains(&cell.x) && (0..3).contains(&cell.y);
        let top_left = blob_index_at(IVec2::new(0, 0), belongs);
        let bottom_right = blob_index_at(IVec2::new(2, 2), belongs);
        assert_ne!(top_left, bottom_right, "corners face different ways");
    }
}
