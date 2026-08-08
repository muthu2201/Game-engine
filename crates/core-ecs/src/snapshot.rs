//! Whole-world capture and restore.
//!
//! Snapshots back three features that share one mechanism: save files, the
//! editor's undo stack, and the replay tests that compare a re-simulated run
//! against a reference.
//!
//! # Opt-in by component
//!
//! Only components registered with [`SnapshotRegistry::register`] are captured.
//! That is deliberate rather than a limitation. Much of what a world holds is
//! derived state — cached bounding boxes, interpolation buffers, GPU handles —
//! that is cheaper to recompute than to serialise, and some of it (a texture
//! handle) is meaningless after a restore. Keeping the snapshot set minimal is
//! also what keeps rollback affordable, since a rollback captures state every
//! tick.
//!
//! # Entity identity is preserved
//!
//! A restore recreates entities with the *same* [`Entity`] handles they had
//! when captured, so a component holding a reference to another entity — a
//! quest target, an NPC's home, a projectile's owner — still resolves. A
//! restore that renumbered entities would silently corrupt every such link.

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;

/// A captured world state.
///
/// Serialisable as a whole, which is what the save system writes to disk.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    /// The world's logical tick at capture time.
    pub tick: u32,
    /// Every live entity, as packed bits, in ascending index order.
    pub entities: Vec<u64>,
    /// Component values, keyed by type name then by entity.
    ///
    /// `BTreeMap` rather than `HashMap` so the serialised form is byte-stable:
    /// a save file that reorders its keys between runs defeats content
    /// addressing and makes diffs unreadable.
    pub components: BTreeMap<String, BTreeMap<u64, serde_json::Value>>,
}

impl Snapshot {
    /// Number of entities captured.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Total number of component values captured.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components.values().map(BTreeMap::len).sum()
    }

    /// The component type names present, sorted.
    #[must_use]
    pub fn component_types(&self) -> Vec<&str> {
        self.components.keys().map(String::as_str).collect()
    }
}

/// How one registered component type is captured and restored.
struct SnapshotEntry {
    /// Stable identifier written into the snapshot.
    name: String,
    /// Reads the component off an entity, if present.
    capture: fn(&World, Entity) -> Option<serde_json::Value>,
    /// Writes the component back onto an entity.
    ///
    /// Returns `false` when the stored value no longer matches the component's
    /// shape, which is how a save from an older build is detected.
    restore: fn(&mut World, Entity, &serde_json::Value) -> bool,
}

/// The set of component types a snapshot covers.
#[derive(Default)]
pub struct SnapshotRegistry {
    entries: Vec<SnapshotEntry>,
}

impl SnapshotRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> SnapshotRegistry {
        SnapshotRegistry::default()
    }

    /// Includes `T` in future snapshots.
    ///
    /// Registering the same type twice is a no-op, so setup code can be called
    /// idempotently.
    pub fn register<T>(&mut self)
    where
        T: Component + Serialize + DeserializeOwned,
    {
        let name = std::any::type_name::<T>().to_string();
        if self.entries.iter().any(|entry| entry.name == name) {
            return;
        }
        self.entries.push(SnapshotEntry {
            name,
            capture: |world, entity| {
                world
                    .get::<T>(entity)
                    .and_then(|value| serde_json::to_value(value).ok())
            },
            restore: |world, entity, value| match serde_json::from_value::<T>(value.clone()) {
                Ok(component) => world.insert(entity, component),
                Err(_) => false,
            },
        });
    }

    /// Number of registered component types.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is registered, so a snapshot would be empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The registered type names, in registration order.
    #[must_use]
    pub fn registered_types(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect()
    }

    /// Captures every live entity and its registered components.
    #[must_use]
    pub fn capture(&self, world: &World) -> Snapshot {
        let entities: Vec<Entity> = world.entities().collect();
        let mut components: BTreeMap<String, BTreeMap<u64, serde_json::Value>> = BTreeMap::new();

        for entry in &self.entries {
            let mut values = BTreeMap::new();
            for entity in &entities {
                if let Some(value) = (entry.capture)(world, *entity) {
                    values.insert(entity.to_bits(), value);
                }
            }
            // Skip types no entity currently carries, so the snapshot stays
            // proportional to what exists rather than to what is registered.
            if !values.is_empty() {
                components.insert(entry.name.clone(), values);
            }
        }

        Snapshot {
            tick: world.tick().0,
            entities: entities.iter().map(|entity| entity.to_bits()).collect(),
            components,
        }
    }

    /// Replaces the world's contents with a captured state.
    ///
    /// Existing entities are removed first. Resources are left untouched — they
    /// hold application-level state such as the asset registry, which must
    /// survive a load rather than be reset by one.
    ///
    /// Returns the number of component values that could not be restored
    /// because their stored shape no longer matches the current type. A
    /// non-zero count means the save predates a component change and the
    /// migration for it is missing.
    pub fn restore(&self, world: &mut World, snapshot: &Snapshot) -> usize {
        world.clear_entities();
        world.set_tick(snapshot.tick);

        for bits in &snapshot.entities {
            world.spawn_at(Entity::from_bits(*bits));
        }

        let mut failures = 0;
        for entry in &self.entries {
            let Some(values) = snapshot.components.get(&entry.name) else {
                continue;
            };
            for (bits, value) in values {
                let entity = Entity::from_bits(*bits);
                if !(entry.restore)(world, entity, value) {
                    failures += 1;
                }
            }
        }
        failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Position {
        x: i32,
        y: i32,
    }
    impl Component for Position {}

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Name(String);
    impl Component for Name {}

    /// Deliberately not registered, to prove capture is opt-in.
    #[derive(Debug, PartialEq)]
    struct RenderCache(u64);
    impl Component for RenderCache {}

    fn registry() -> SnapshotRegistry {
        let mut registry = SnapshotRegistry::new();
        registry.register::<Position>();
        registry.register::<Name>();
        registry
    }

    #[test]
    fn capture_and_restore_round_trips() {
        let mut world = World::new();
        let first = world.spawn((Position { x: 1, y: 2 }, Name("first".into())));
        let second = world.spawn((Position { x: 3, y: 4 },));

        let registry = registry();
        let snapshot = registry.capture(&world);

        let mut restored = World::new();
        assert_eq!(registry.restore(&mut restored, &snapshot), 0);

        assert_eq!(restored.entity_count(), 2);
        assert_eq!(
            restored.get::<Position>(first),
            Some(&Position { x: 1, y: 2 })
        );
        assert_eq!(restored.get::<Name>(first), Some(&Name("first".into())));
        assert_eq!(
            restored.get::<Position>(second),
            Some(&Position { x: 3, y: 4 })
        );
        assert_eq!(restored.get::<Name>(second), None);
    }

    #[test]
    fn entity_handles_survive_a_round_trip() {
        let mut world = World::new();
        let a = world.spawn((Position { x: 0, y: 0 },));
        let b = world.spawn((Position { x: 1, y: 1 },));
        // Recycle a slot so the generations are not all zero.
        world.despawn(a);
        let c = world.spawn((Position { x: 2, y: 2 },));
        assert_eq!(
            c.index(),
            a.index(),
            "the test needs a recycled index to be meaningful"
        );

        let registry = registry();
        let snapshot = registry.capture(&world);
        let mut restored = World::new();
        registry.restore(&mut restored, &snapshot);

        assert!(restored.contains(b));
        assert!(restored.contains(c));
        assert!(
            !restored.contains(a),
            "the stale handle must stay dead across a restore"
        );
        assert_eq!(restored.get::<Position>(c), Some(&Position { x: 2, y: 2 }));
    }

    #[test]
    fn unregistered_components_are_not_captured() {
        let mut world = World::new();
        let entity = world.spawn((Position { x: 1, y: 1 }, RenderCache(999)));

        let registry = registry();
        let snapshot = registry.capture(&world);
        assert_eq!(
            snapshot.component_types(),
            vec![std::any::type_name::<Position>()]
        );

        let mut restored = World::new();
        registry.restore(&mut restored, &snapshot);
        assert!(restored.get::<Position>(entity).is_some());
        assert!(restored.get::<RenderCache>(entity).is_none());
    }

    #[test]
    fn restoring_replaces_rather_than_merges() {
        let registry = registry();
        let mut source = World::new();
        source.spawn((Position { x: 1, y: 1 },));
        let snapshot = registry.capture(&source);

        let mut target = World::new();
        for _ in 0..5 {
            target.spawn((Position { x: 9, y: 9 },));
        }
        registry.restore(&mut target, &snapshot);
        assert_eq!(
            target.entity_count(),
            1,
            "pre-existing entities must be cleared"
        );
    }

    #[test]
    fn the_tick_is_part_of_the_snapshot() {
        let mut world = World::new();
        for _ in 0..7 {
            world.advance_tick();
        }
        world.spawn((Position { x: 0, y: 0 },));
        let registry = registry();
        let snapshot = registry.capture(&world);

        let mut restored = World::new();
        registry.restore(&mut restored, &snapshot);
        assert_eq!(restored.tick(), world.tick());
    }

    #[test]
    fn registering_twice_is_idempotent() {
        let mut registry = SnapshotRegistry::new();
        registry.register::<Position>();
        registry.register::<Position>();
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn a_mismatched_stored_value_is_counted_rather_than_panicking() {
        let mut world = World::new();
        let entity = world.spawn((Position { x: 1, y: 2 },));
        let registry = registry();
        let mut snapshot = registry.capture(&world);

        // Simulate a save written by a build where Position had a different
        // shape.
        let key = std::any::type_name::<Position>().to_string();
        snapshot.components.get_mut(&key).unwrap().insert(
            entity.to_bits(),
            serde_json::json!({ "totally": "different" }),
        );

        let mut restored = World::new();
        assert_eq!(registry.restore(&mut restored, &snapshot), 1);
        assert!(
            restored.contains(entity),
            "the entity still exists, just without the component"
        );
        assert_eq!(restored.get::<Position>(entity), None);
    }

    #[test]
    fn snapshots_serialise_to_stable_json() {
        let mut world = World::new();
        world.spawn((Position { x: 1, y: 2 }, Name("a".into())));
        world.spawn((Position { x: 3, y: 4 }, Name("b".into())));
        let registry = registry();

        let first = serde_json::to_string(&registry.capture(&world)).unwrap();
        let second = serde_json::to_string(&registry.capture(&world)).unwrap();
        assert_eq!(first, second, "key ordering must not vary between captures");

        let parsed: Snapshot = serde_json::from_str(&first).unwrap();
        assert_eq!(parsed.entity_count(), 2);
        assert_eq!(parsed.component_count(), 4);
    }

    #[test]
    fn an_empty_world_captures_and_restores_cleanly() {
        let registry = registry();
        let snapshot = registry.capture(&World::new());
        assert_eq!(snapshot.entity_count(), 0);
        assert_eq!(snapshot.component_count(), 0);

        let mut world = World::new();
        world.spawn((Position { x: 1, y: 1 },));
        registry.restore(&mut world, &snapshot);
        assert!(world.is_empty());
    }
}
