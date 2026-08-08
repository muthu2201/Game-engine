//! Spatial hashing, so overlap queries do not scan every entity.
//!
//! A farm at harvest time holds hundreds of crops, dropped items, NPCs and
//! particles. Testing every pair each tick is quadratic and becomes the
//! simulation's dominant cost well before that. The spatial hash buckets bodies
//! by the grid cells their bounds cover, so a query only examines the handful of
//! bodies sharing a cell with it.
//!
//! # Choosing a cell size
//!
//! The cell should be roughly the size of the largest *common* body. Too small
//! and a large body is inserted into many cells; too large and each cell holds
//! too many bodies to have narrowed anything. Two tiles is the engine's default
//! and matches the size of a character plus its interaction radius.

use std::collections::HashMap;
use verdant_core_ecs::Entity;
use verdant_core_math::{Fx, IVec2, Rect};

/// One body registered in the broadphase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BroadphaseEntry {
    /// The entity this body belongs to.
    pub entity: Entity,
    /// Its world-space bounds at insertion time.
    pub bounds: Rect,
    /// The layer bits it occupies.
    pub layer: u32,
}

/// A uniform spatial hash over world space.
#[derive(Debug)]
pub struct SpatialHash {
    cell_size: Fx,
    /// Cell coordinate to the indices of the entries it contains.
    cells: HashMap<IVec2, Vec<usize>>,
    entries: Vec<BroadphaseEntry>,
}

impl SpatialHash {
    /// Creates an empty hash.
    ///
    /// # Panics
    ///
    /// Panics if `cell_size` is not positive.
    #[must_use]
    pub fn new(cell_size: Fx) -> SpatialHash {
        assert!(
            cell_size.is_positive(),
            "SpatialHash: cell size must be positive"
        );
        SpatialHash {
            cell_size,
            cells: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Removes every entry, keeping allocated capacity.
    ///
    /// Called once per tick before re-inserting the moving bodies. Retaining
    /// the bucket `Vec`s is what keeps a full rebuild allocation-free after the
    /// first few frames.
    pub fn clear(&mut self) {
        for bucket in self.cells.values_mut() {
            bucket.clear();
        }
        self.entries.clear();
    }

    /// Inserts a body.
    pub fn insert(&mut self, entity: Entity, bounds: Rect, layer: u32) {
        let index = self.entries.len();
        self.entries.push(BroadphaseEntry {
            entity,
            bounds,
            layer,
        });
        for cell in self.covered_cells(bounds) {
            self.cells.entry(cell).or_default().push(index);
        }
    }

    /// The cells a rectangle overlaps.
    fn covered_cells(&self, bounds: Rect) -> Vec<IVec2> {
        let min_x = (bounds.left() / self.cell_size).floor_int();
        let min_y = (bounds.top() / self.cell_size).floor_int();
        let max_x = (bounds.right() / self.cell_size).floor_int();
        let max_y = (bounds.bottom() / self.cell_size).floor_int();

        let mut cells = Vec::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                cells.push(IVec2::new(x, y));
            }
        }
        cells
    }

    /// Every body whose bounds overlap `area` and whose layer matches `mask`.
    ///
    /// Results are sorted by entity so the order is a function of the world's
    /// contents rather than of hash-map iteration — a query result that feeds
    /// gameplay must not vary between runs.
    #[must_use]
    pub fn query(&self, area: Rect, mask: u32) -> Vec<BroadphaseEntry> {
        let mut seen = Vec::new();
        let mut results = Vec::new();
        for cell in self.covered_cells(area) {
            let Some(bucket) = self.cells.get(&cell) else {
                continue;
            };
            for index in bucket {
                // A body spanning several cells appears in each of their
                // buckets, so duplicates must be filtered.
                if seen.contains(index) {
                    continue;
                }
                seen.push(*index);
                let entry = self.entries[*index];
                if (entry.layer & mask) != 0 && entry.bounds.intersects(area) {
                    results.push(entry);
                }
            }
        }
        results.sort_unstable_by_key(|entry| entry.entity.to_bits());
        results
    }

    /// The bounds of every body overlapping `area` on `mask`, excluding one
    /// entity.
    ///
    /// This is the exact shape the solver wants: the blocker list for a body,
    /// which must not include the body itself.
    #[must_use]
    pub fn blockers_for(&self, area: Rect, mask: u32, exclude: Entity) -> Vec<Rect> {
        self.query(area, mask)
            .into_iter()
            .filter(|entry| entry.entity != exclude)
            .map(|entry| entry.bounds)
            .collect()
    }

    /// Number of registered bodies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no bodies are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The configured cell size.
    #[must_use]
    pub fn cell_size(&self) -> Fx {
        self.cell_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    fn entity(index: u32) -> Entity {
        Entity::from_parts(index, 0)
    }

    #[test]
    fn a_query_finds_overlapping_bodies() {
        let mut hash = SpatialHash::new(fx(32));
        hash.insert(entity(1), Rect::from_ints(0, 0, 10, 10), 1);
        hash.insert(entity(2), Rect::from_ints(100, 100, 10, 10), 1);

        let found = hash.query(Rect::from_ints(5, 5, 10, 10), 1);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].entity, entity(1));
    }

    #[test]
    fn a_query_respects_the_layer_mask() {
        let mut hash = SpatialHash::new(fx(32));
        hash.insert(entity(1), Rect::from_ints(0, 0, 10, 10), 0b001);
        hash.insert(entity(2), Rect::from_ints(0, 0, 10, 10), 0b010);

        assert_eq!(hash.query(Rect::from_ints(0, 0, 10, 10), 0b001).len(), 1);
        assert_eq!(hash.query(Rect::from_ints(0, 0, 10, 10), 0b011).len(), 2);
        assert_eq!(hash.query(Rect::from_ints(0, 0, 10, 10), 0b100).len(), 0);
    }

    #[test]
    fn a_body_spanning_several_cells_is_reported_once() {
        let mut hash = SpatialHash::new(fx(8));
        // 40 units wide across a cell size of 8: five cells.
        hash.insert(entity(1), Rect::from_ints(0, 0, 40, 40), 1);
        let found = hash.query(Rect::from_ints(0, 0, 40, 40), 1);
        assert_eq!(found.len(), 1, "duplicates across buckets must be filtered");
    }

    #[test]
    fn bodies_that_only_share_a_cell_are_not_reported() {
        let mut hash = SpatialHash::new(fx(64));
        // Both land in cell (0, 0) but their bounds are far apart.
        hash.insert(entity(1), Rect::from_ints(0, 0, 4, 4), 1);
        let found = hash.query(Rect::from_ints(50, 50, 4, 4), 1);
        assert!(
            found.is_empty(),
            "the broadphase must still confirm the overlap"
        );
    }

    #[test]
    fn query_results_are_sorted_for_determinism() {
        let mut hash = SpatialHash::new(fx(32));
        for index in [7u32, 2, 9, 1, 5] {
            hash.insert(entity(index), Rect::from_ints(0, 0, 10, 10), 1);
        }
        let found = hash.query(Rect::from_ints(0, 0, 10, 10), 1);
        let ids: Vec<u64> = found.iter().map(|entry| entry.entity.to_bits()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn blockers_exclude_the_querying_body() {
        let mut hash = SpatialHash::new(fx(32));
        hash.insert(entity(1), Rect::from_ints(0, 0, 10, 10), 1);
        hash.insert(entity(2), Rect::from_ints(2, 2, 10, 10), 1);

        let blockers = hash.blockers_for(Rect::from_ints(0, 0, 10, 10), 1, entity(1));
        assert_eq!(blockers.len(), 1, "a body must not block itself");
        assert_eq!(blockers[0], Rect::from_ints(2, 2, 10, 10));
    }

    #[test]
    fn clearing_empties_the_hash_but_keeps_it_usable() {
        let mut hash = SpatialHash::new(fx(32));
        hash.insert(entity(1), Rect::from_ints(0, 0, 10, 10), 1);
        hash.clear();
        assert!(hash.is_empty());
        assert!(hash.query(Rect::from_ints(0, 0, 10, 10), 1).is_empty());

        hash.insert(entity(2), Rect::from_ints(0, 0, 10, 10), 1);
        assert_eq!(hash.query(Rect::from_ints(0, 0, 10, 10), 1).len(), 1);
    }

    #[test]
    fn negative_coordinates_are_bucketed_correctly() {
        let mut hash = SpatialHash::new(fx(16));
        hash.insert(entity(1), Rect::from_ints(-40, -40, 8, 8), 1);
        let found = hash.query(Rect::from_ints(-40, -40, 8, 8), 1);
        assert_eq!(found.len(), 1, "flooring must handle negative cells");
    }

    #[test]
    #[should_panic(expected = "cell size must be positive")]
    fn a_non_positive_cell_size_is_rejected() {
        let _ = SpatialHash::new(Fx::ZERO);
    }
}
