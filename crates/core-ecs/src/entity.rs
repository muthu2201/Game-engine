//! Entity identity and allocation.

use std::fmt;

/// A handle to an entity: an index paired with a generation counter.
///
/// The generation is what makes the handle safe to keep around. When an entity
/// is despawned its index goes back on the free list, but the generation stored
/// in the allocator is bumped; a stale handle held by a quest script or a
/// targeting component therefore compares unequal to whatever now lives at that
/// index, and [`World::contains`](crate::World::contains) reports it as dead
/// instead of silently addressing an unrelated entity.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    /// A handle that is never valid in any world.
    ///
    /// Useful as a "no target" sentinel in components that would otherwise need
    /// an `Option<Entity>` and an extra word of storage.
    pub const NULL: Entity = Entity {
        index: u32::MAX,
        generation: u32::MAX,
    };

    /// Builds a handle from its parts.
    ///
    /// Only [`EntityAllocator`] and the deserialiser should call this — a
    /// hand-built handle is not guaranteed to refer to a live entity.
    #[inline]
    #[must_use]
    pub const fn from_parts(index: u32, generation: u32) -> Entity {
        Entity { index, generation }
    }

    /// The entity's slot index. Stable for the lifetime of the entity.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation this handle was minted at.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// True for [`Entity::NULL`].
    #[inline]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.index == u32::MAX && self.generation == u32::MAX
    }

    /// Packs the handle into a single `u64`, for hashing and save files.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        ((self.generation as u64) << 32) | (self.index as u64)
    }

    /// Unpacks a handle produced by [`Entity::to_bits`].
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_bits(bits: u64) -> Entity {
        Entity {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "Entity::NULL")
        } else {
            write!(f, "Entity({}v{})", self.index, self.generation)
        }
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Where an entity's component data currently lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct EntityLocation {
    /// Index into [`Archetypes`](crate::storage::Archetypes).
    pub archetype: u32,
    /// Row within that archetype's columns.
    pub row: u32,
}

/// The state of one entity slot.
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// In use, with data at this location.
    Live {
        generation: u32,
        location: EntityLocation,
    },
    /// Free, waiting to be recycled at this generation.
    Free { generation: u32 },
}

/// Allocates entity handles and tracks where each entity's data lives.
///
/// Freed indices are recycled in last-in-first-out order. LIFO rather than FIFO
/// is deliberate: reusing a just-freed slot keeps the live set densely packed at
/// the front of the array, so iteration touches fewer cache lines in a world
/// that spawns and despawns heavily — which a game with projectiles, particles
/// and dropped items does constantly.
#[derive(Debug, Default)]
pub struct EntityAllocator {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live_count: u32,
}

impl EntityAllocator {
    /// Creates an empty allocator.
    #[must_use]
    pub fn new() -> EntityAllocator {
        EntityAllocator::default()
    }

    /// Reserves a new handle. The caller must immediately call
    /// [`EntityAllocator::set_location`] once the data has been stored.
    pub(crate) fn allocate(&mut self) -> Entity {
        self.live_count += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            let generation = match *slot {
                Slot::Free { generation } => generation,
                Slot::Live { .. } => unreachable!("a slot on the free list cannot be live"),
            };
            *slot = Slot::Live {
                generation,
                location: EntityLocation {
                    archetype: 0,
                    row: 0,
                },
            };
            Entity { index, generation }
        } else {
            // A world with more than 4 billion simultaneous entities is far
            // outside what this engine targets; failing loudly beats wrapping.
            let index = u32::try_from(self.slots.len())
                .expect("entity index space exhausted (more than u32::MAX entities)");
            self.slots.push(Slot::Live {
                generation: 0,
                location: EntityLocation {
                    archetype: 0,
                    row: 0,
                },
            });
            Entity {
                index,
                generation: 0,
            }
        }
    }

    /// Reserves a *specific* handle, growing the slot table as needed.
    ///
    /// Only a restore should call this. It exists so a loaded world hands back
    /// the same [`Entity`] values that were captured, keeping entity-to-entity
    /// references — quest targets, an NPC's home, a projectile's owner — valid
    /// across a save and load.
    ///
    /// Returns `false` when the slot is already live at a different generation,
    /// which would mean the snapshot disagrees with the world it is restoring
    /// into.
    pub(crate) fn allocate_at(&mut self, entity: Entity) -> bool {
        if entity.is_null() {
            return false;
        }
        let index = entity.index() as usize;
        if index >= self.slots.len() {
            // Slots between the end and the requested index are free, at
            // generation zero, and go on the free list so they are not leaked.
            for slot_index in self.slots.len()..=index {
                self.slots.push(Slot::Free { generation: 0 });
                if slot_index != index {
                    self.free
                        .push(u32::try_from(slot_index).expect("entity index space exhausted"));
                }
            }
        }
        match self.slots[index] {
            Slot::Live { generation, .. } => generation == entity.generation(),
            Slot::Free { .. } => {
                // Drop this index from the free list if it is queued there.
                self.free.retain(|queued| *queued != entity.index());
                self.slots[index] = Slot::Live {
                    generation: entity.generation(),
                    location: EntityLocation {
                        archetype: 0,
                        row: 0,
                    },
                };
                self.live_count += 1;
                true
            }
        }
    }

    /// Releases a handle, bumping its generation so stale copies stop resolving.
    ///
    /// Returns the location the entity occupied, or `None` when the handle was
    /// already dead — which makes a double despawn a no-op rather than a panic.
    pub(crate) fn free(&mut self, entity: Entity) -> Option<EntityLocation> {
        let slot = self.slots.get_mut(entity.index as usize)?;
        match *slot {
            Slot::Live {
                generation,
                location,
            } if generation == entity.generation => {
                // Saturating rather than wrapping: a slot that has been reused
                // four billion times stops being recycled instead of handing
                // out a handle that collides with an ancient one.
                let next = generation.saturating_add(1);
                *slot = Slot::Free { generation: next };
                if next != u32::MAX {
                    self.free.push(entity.index);
                }
                self.live_count -= 1;
                Some(location)
            }
            _ => None,
        }
    }

    /// True when the handle refers to a live entity.
    #[must_use]
    pub fn contains(&self, entity: Entity) -> bool {
        matches!(
            self.slots.get(entity.index as usize),
            Some(Slot::Live { generation, .. }) if *generation == entity.generation
        )
    }

    /// The location of a live entity's data.
    pub(crate) fn location(&self, entity: Entity) -> Option<EntityLocation> {
        match self.slots.get(entity.index as usize) {
            Some(Slot::Live {
                generation,
                location,
            }) if *generation == entity.generation => Some(*location),
            _ => None,
        }
    }

    /// Records where a live entity's data now lives.
    pub(crate) fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        if let Some(Slot::Live {
            generation,
            location: slot_location,
        }) = self.slots.get_mut(entity.index as usize)
        {
            if *generation == entity.generation {
                *slot_location = location;
            }
        }
    }

    /// Rewrites the location of whatever entity currently occupies `index`.
    ///
    /// Used when a swap-remove moves the last row of an archetype into the hole
    /// left by a despawn.
    pub(crate) fn relocate_by_index(&mut self, index: u32, location: EntityLocation) {
        if let Some(Slot::Live {
            location: slot_location,
            ..
        }) = self.slots.get_mut(index as usize)
        {
            *slot_location = location;
        }
    }

    /// Number of live entities.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.live_count
    }

    /// True when no entities are alive.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Total number of slots ever allocated, live or recycled.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Iterates every live entity in ascending index order.
    ///
    /// The ordering is part of the determinism contract: anything that walks
    /// all entities — save serialisation, the state hasher — must see them in
    /// the same sequence on every machine.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot {
                Slot::Live { generation, .. } => Some(Entity {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    generation: *generation,
                }),
                Slot::Free { .. } => None,
            })
    }

    /// Removes every entity, keeping the allocated slot capacity.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.live_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_unique_while_live() {
        let mut allocator = EntityAllocator::new();
        let entities: Vec<Entity> = (0..100).map(|_| allocator.allocate()).collect();
        let mut sorted = entities.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            entities.len(),
            "allocator handed out a duplicate"
        );
        assert_eq!(allocator.len(), 100);
    }

    #[test]
    fn freed_indices_are_recycled_with_a_new_generation() {
        let mut allocator = EntityAllocator::new();
        let first = allocator.allocate();
        allocator.free(first);
        let second = allocator.allocate();
        assert_eq!(second.index(), first.index(), "the index should be reused");
        assert_ne!(
            second.generation(),
            first.generation(),
            "but not the generation"
        );
    }

    #[test]
    fn a_stale_handle_stops_resolving_once_its_slot_is_reused() {
        let mut allocator = EntityAllocator::new();
        let stale = allocator.allocate();
        allocator.free(stale);
        let fresh = allocator.allocate();
        assert!(!allocator.contains(stale), "the old handle must be dead");
        assert!(allocator.contains(fresh));
        assert_eq!(allocator.location(stale), None);
    }

    #[test]
    fn freeing_twice_is_a_no_op_rather_than_a_panic() {
        let mut allocator = EntityAllocator::new();
        let entity = allocator.allocate();
        assert!(allocator.free(entity).is_some());
        assert!(allocator.free(entity).is_none());
        assert_eq!(allocator.len(), 0);
    }

    #[test]
    fn freeing_an_unknown_handle_is_a_no_op() {
        let mut allocator = EntityAllocator::new();
        assert!(allocator.free(Entity::from_parts(999, 0)).is_none());
        assert!(allocator.free(Entity::NULL).is_none());
    }

    #[test]
    fn locations_round_trip() {
        let mut allocator = EntityAllocator::new();
        let entity = allocator.allocate();
        let location = EntityLocation {
            archetype: 3,
            row: 7,
        };
        allocator.set_location(entity, location);
        assert_eq!(allocator.location(entity), Some(location));
    }

    #[test]
    fn iteration_is_in_ascending_index_order() {
        let mut allocator = EntityAllocator::new();
        let entities: Vec<Entity> = (0..10).map(|_| allocator.allocate()).collect();
        allocator.free(entities[3]);
        allocator.free(entities[7]);
        let live: Vec<u32> = allocator.iter().map(Entity::index).collect();
        assert_eq!(live, vec![0, 1, 2, 4, 5, 6, 8, 9]);
    }

    #[test]
    fn the_null_handle_is_never_live() {
        let mut allocator = EntityAllocator::new();
        allocator.allocate();
        assert!(!allocator.contains(Entity::NULL));
        assert!(Entity::NULL.is_null());
    }

    #[test]
    fn bit_packing_round_trips() {
        for (index, generation) in [(0u32, 0u32), (7, 3), (u32::MAX - 1, 12345)] {
            let entity = Entity::from_parts(index, generation);
            assert_eq!(Entity::from_bits(entity.to_bits()), entity);
        }
        assert_eq!(Entity::from_bits(Entity::NULL.to_bits()), Entity::NULL);
    }

    #[test]
    fn recycling_is_last_in_first_out() {
        let mut allocator = EntityAllocator::new();
        let a = allocator.allocate();
        let b = allocator.allocate();
        allocator.free(a);
        allocator.free(b);
        // b was freed last, so its index comes back first.
        assert_eq!(allocator.allocate().index(), b.index());
        assert_eq!(allocator.allocate().index(), a.index());
    }
}
