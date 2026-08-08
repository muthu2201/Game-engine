//! Integration tests for the world's archetype bookkeeping.
//!
//! These target the paths that are easy to get subtly wrong and hard to notice:
//! swap-removes that relocate a *different* entity, migrations that must carry
//! some components and drop others, and the interaction between the two storage
//! kinds. A bug in any of them shows up much later as a component that silently
//! belongs to the wrong entity.

use verdant_core_ecs::{Commands, Component, Entity, StorageKind, World};

#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i32);
impl Component for Position {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Velocity(i32);
impl Component for Velocity {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Health(i32);
impl Component for Health {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Name(&'static str);
impl Component for Name {}

/// A high-churn marker in non-fragmenting storage.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Stunned(u32);
impl Component for Stunned {
    const STORAGE: StorageKind = StorageKind::Sparse;
}

#[test]
fn components_survive_an_archetype_migration() {
    let mut world = World::new();
    let entity = world.spawn((Position(1), Velocity(2)));

    // Adding a third component moves the entity to a new archetype; the first
    // two must come with it.
    world.insert(entity, Health(30));

    assert_eq!(world.get::<Position>(entity), Some(&Position(1)));
    assert_eq!(world.get::<Velocity>(entity), Some(&Velocity(2)));
    assert_eq!(world.get::<Health>(entity), Some(&Health(30)));
}

#[test]
fn removing_a_component_keeps_the_others_intact() {
    let mut world = World::new();
    let entity = world.spawn((Position(1), Velocity(2), Health(3)));

    assert_eq!(world.remove::<Velocity>(entity), Some(Velocity(2)));

    assert_eq!(world.get::<Velocity>(entity), None);
    assert_eq!(world.get::<Position>(entity), Some(&Position(1)));
    assert_eq!(world.get::<Health>(entity), Some(&Health(3)));
    assert!(!world.has::<Velocity>(entity));
}

#[test]
fn removing_a_component_the_entity_lacks_returns_none() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    assert_eq!(world.remove::<Health>(entity), None);
    assert_eq!(world.get::<Position>(entity), Some(&Position(1)));
}

#[test]
fn a_migration_relocates_the_entity_swapped_into_the_hole() {
    let mut world = World::new();
    // Three entities sharing one archetype.
    let first = world.spawn((Position(1), Velocity(1)));
    let second = world.spawn((Position(2), Velocity(2)));
    let third = world.spawn((Position(3), Velocity(3)));

    // Moving `first` out swap-removes its row, dragging `third` into it. If the
    // world forgets to update `third`'s recorded location, the next read
    // returns the wrong entity's data.
    world.insert(first, Health(99));

    assert_eq!(world.get::<Position>(first), Some(&Position(1)));
    assert_eq!(world.get::<Position>(second), Some(&Position(2)));
    assert_eq!(world.get::<Position>(third), Some(&Position(3)));
    assert_eq!(world.get::<Velocity>(third), Some(&Velocity(3)));
}

#[test]
fn a_despawn_relocates_the_entity_swapped_into_the_hole() {
    let mut world = World::new();
    let first = world.spawn((Position(1),));
    let second = world.spawn((Position(2),));
    let third = world.spawn((Position(3),));

    world.despawn(first);

    assert!(!world.contains(first));
    assert_eq!(world.get::<Position>(second), Some(&Position(2)));
    assert_eq!(world.get::<Position>(third), Some(&Position(3)));
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn every_entity_stays_addressable_through_heavy_churn() {
    let mut world = World::new();
    let mut tracked: Vec<(Entity, i32)> = Vec::new();

    for value in 0..200i32 {
        let entity = world.spawn((Position(value),));
        tracked.push((entity, value));
    }
    // Give every third entity an extra component, forcing migrations.
    for (index, (entity, _)) in tracked.iter().enumerate() {
        if index % 3 == 0 {
            world.insert(*entity, Velocity(7));
        }
    }
    // Despawn every fifth, forcing swap-removes.
    let mut survivors = Vec::new();
    for (index, (entity, value)) in tracked.iter().enumerate() {
        if index % 5 == 0 {
            world.despawn(*entity);
        } else {
            survivors.push((*entity, *value));
        }
    }

    for (entity, value) in survivors {
        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position(value)),
            "{entity:?} lost track of its position after churn"
        );
    }
}

#[test]
fn inserting_an_existing_component_overwrites_without_migrating() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    let archetypes_before = world.archetype_count();

    world.insert(entity, Position(42));

    assert_eq!(world.get::<Position>(entity), Some(&Position(42)));
    assert_eq!(
        world.archetype_count(),
        archetypes_before,
        "overwriting a component must not create an archetype"
    );
}

#[test]
fn operations_on_a_dead_entity_are_rejected_not_panics() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    world.despawn(entity);

    assert!(!world.insert(entity, Health(1)));
    assert_eq!(world.get::<Position>(entity), None);
    assert_eq!(world.remove::<Position>(entity), None);
    assert!(!world.has::<Position>(entity));
    assert!(!world.despawn(entity));
}

#[test]
fn queries_visit_every_matching_entity_exactly_once() {
    let mut world = World::new();
    // Entities spread across three different archetypes.
    world.spawn((Position(1),));
    world.spawn((Position(2), Velocity(0)));
    world.spawn((Position(3), Velocity(0), Health(0)));
    world.spawn((Velocity(0),)); // no Position: must not match

    let mut seen: Vec<i32> = world.query::<(&Position,)>().map(|(_, (p,))| p.0).collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![1, 2, 3]);
}

#[test]
fn multi_component_queries_pair_the_right_values_together() {
    let mut world = World::new();
    for value in 1..=50i32 {
        world.spawn((Position(value), Velocity(value * 10)));
    }
    // Some entities in a different archetype, to force multi-table iteration.
    for value in 51..=60i32 {
        world.spawn((Position(value), Velocity(value * 10), Health(1)));
    }

    let mut pairs: Vec<(i32, i32)> = world
        .query::<(&Position, &Velocity)>()
        .map(|(_, (p, v))| (p.0, v.0))
        .collect();
    pairs.sort_unstable();

    assert_eq!(pairs.len(), 60);
    for (position, velocity) in pairs {
        assert_eq!(
            velocity,
            position * 10,
            "a query mismatched two entities' components"
        );
    }
}

#[test]
fn mutable_queries_write_through_to_the_world() {
    let mut world = World::new();
    let entity = world.spawn((Position(0), Velocity(5)));

    for _ in 0..10 {
        for (_entity, (position, velocity)) in world.query::<(&mut Position, &Velocity)>() {
            position.0 += velocity.0;
        }
    }
    assert_eq!(world.get::<Position>(entity), Some(&Position(50)));
}

#[test]
fn query_filters_narrow_by_presence_and_absence() {
    let mut world = World::new();
    world.spawn((Position(1), Velocity(0)));
    world.spawn((Position(2),));
    world.spawn((Position(3), Velocity(0), Health(0)));

    let with_velocity: Vec<i32> = world
        .query_filtered::<(&Position,)>()
        .with::<Velocity>()
        .iter()
        .map(|(_, (p,))| p.0)
        .collect();
    let mut with_velocity = with_velocity;
    with_velocity.sort_unstable();
    assert_eq!(with_velocity, vec![1, 3]);

    let without_health: Vec<i32> = world
        .query_filtered::<(&Position,)>()
        .without::<Health>()
        .iter()
        .map(|(_, (p,))| p.0)
        .collect();
    let mut without_health = without_health;
    without_health.sort_unstable();
    assert_eq!(without_health, vec![1, 2]);
}

#[test]
fn read_only_queries_can_overlap() {
    let mut world = World::new();
    world.spawn((Position(1), Name("a")));
    world.spawn((Position(2), Name("b")));

    // Two live read-only queries plus point lookups, all at once. This is the
    // pattern a system needs when it inspects a second entity mid-iteration.
    let positions: Vec<i32> = world
        .query_ref::<(&Position,)>()
        .map(|(_, (p,))| p.0)
        .collect();
    let names: Vec<&str> = world.query_ref::<(&Name,)>().map(|(_, (n,))| n.0).collect();
    assert_eq!(positions.len(), 2);
    assert_eq!(names.len(), 2);
}

#[test]
fn a_read_only_query_for_an_unknown_component_matches_nothing() {
    let world = World::new();
    assert_eq!(world.query_ref::<(&Position,)>().count(), 0);
}

// -----------------------------------------------------------------------------
// Sparse storage
// -----------------------------------------------------------------------------

#[test]
fn sparse_components_do_not_fragment_archetypes() {
    let mut world = World::new();
    let entity = world.spawn((Position(1), Velocity(2)));
    let archetypes_before = world.archetype_count();

    // Toggling a sparse component many times must never create an archetype —
    // that is the whole reason the storage kind exists.
    for tick in 0..100u32 {
        world.insert(entity, Stunned(tick));
        world.remove::<Stunned>(entity);
    }

    assert_eq!(world.archetype_count(), archetypes_before);
    assert_eq!(world.get::<Position>(entity), Some(&Position(1)));
}

#[test]
fn sparse_components_read_back_correctly() {
    let mut world = World::new();
    let a = world.spawn((Position(1),));
    let b = world.spawn((Position(2),));

    world.insert(a, Stunned(5));
    assert_eq!(world.get::<Stunned>(a), Some(&Stunned(5)));
    assert_eq!(world.get::<Stunned>(b), None);
    assert!(world.has::<Stunned>(a));
    assert!(!world.has::<Stunned>(b));

    assert_eq!(world.remove::<Stunned>(a), Some(Stunned(5)));
    assert_eq!(world.get::<Stunned>(a), None);
}

#[test]
fn sparse_components_are_dropped_when_the_entity_despawns() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    world.insert(entity, Stunned(1));
    world.despawn(entity);

    // A new entity recycling the same index must not inherit the marker.
    let recycled = world.spawn((Position(2),));
    assert_eq!(
        recycled.index(),
        entity.index(),
        "the test needs a recycled slot"
    );
    assert_eq!(world.get::<Stunned>(recycled), None);
}

#[test]
fn queries_can_filter_on_sparse_components() {
    let mut world = World::new();
    let stunned = world.spawn((Position(1),));
    world.spawn((Position(2),));
    let also_stunned = world.spawn((Position(3),));
    world.insert(stunned, Stunned(1));
    world.insert(also_stunned, Stunned(1));

    let mut affected: Vec<i32> = world
        .query_filtered::<(&Position,)>()
        .with::<Stunned>()
        .iter()
        .map(|(_, (p,))| p.0)
        .collect();
    affected.sort_unstable();
    assert_eq!(affected, vec![1, 3]);

    let mut unaffected: Vec<i32> = world
        .query_filtered::<(&Position,)>()
        .without::<Stunned>()
        .iter()
        .map(|(_, (p,))| p.0)
        .collect();
    unaffected.sort_unstable();
    assert_eq!(unaffected, vec![2]);
}

#[test]
#[should_panic(expected = "sparse storage and cannot be fetched")]
fn fetching_a_sparse_component_in_a_query_fails_loudly() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    world.insert(entity, Stunned(1));
    let _ = world.query::<(&Stunned,)>().count();
}

// -----------------------------------------------------------------------------
// Change detection
// -----------------------------------------------------------------------------

#[test]
fn change_ticks_track_writes() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    let spawn_tick = world
        .changed_tick::<Position>(entity)
        .expect("just inserted");

    world.advance_tick();
    world.advance_tick();

    // A read must not mark the component changed.
    assert_eq!(world.get::<Position>(entity), Some(&Position(1)));
    assert_eq!(world.changed_tick::<Position>(entity), Some(spawn_tick));

    // A mutable borrow must.
    world.get_mut::<Position>(entity).unwrap().0 = 2;
    let changed = world
        .changed_tick::<Position>(entity)
        .expect("still present");
    assert!(changed.is_newer_than(spawn_tick));
    // The added tick is not disturbed by a write.
    assert_eq!(world.added_tick::<Position>(entity), Some(spawn_tick));
}

#[test]
fn mutable_queries_stamp_the_change_tick() {
    let mut world = World::new();
    let entity = world.spawn((Position(1), Velocity(1)));
    let before = world.changed_tick::<Position>(entity).unwrap();
    world.advance_tick();

    for (_entity, (position,)) in world.query::<(&mut Position,)>() {
        position.0 += 1;
    }
    assert!(world
        .changed_tick::<Position>(entity)
        .unwrap()
        .is_newer_than(before));
    // The untouched component keeps its original tick.
    assert_eq!(world.changed_tick::<Velocity>(entity), Some(before));
}

#[test]
fn change_ticks_work_for_sparse_components_too() {
    let mut world = World::new();
    let entity = world.spawn((Position(1),));
    world.insert(entity, Stunned(1));
    let before = world.changed_tick::<Stunned>(entity).unwrap();

    world.advance_tick();
    world.get_mut::<Stunned>(entity).unwrap().0 = 2;
    assert!(world
        .changed_tick::<Stunned>(entity)
        .unwrap()
        .is_newer_than(before));
}

// -----------------------------------------------------------------------------
// Determinism
// -----------------------------------------------------------------------------

#[test]
fn determinism_query_order_is_reproducible() {
    /// Builds a world through an identical sequence of operations and records
    /// the order a query visits entities in.
    fn build_and_observe() -> Vec<u64> {
        let mut world = World::new();
        let mut entities = Vec::new();
        for value in 0..100i32 {
            entities.push(world.spawn((Position(value),)));
        }
        for (index, entity) in entities.iter().enumerate() {
            if index % 4 == 0 {
                world.insert(*entity, Velocity(1));
            }
            if index % 7 == 0 {
                world.insert(*entity, Health(1));
            }
        }
        for (index, entity) in entities.iter().enumerate() {
            if index % 11 == 0 {
                world.despawn(*entity);
            }
        }
        world
            .query::<(&Position,)>()
            .map(|(entity, _)| entity.to_bits())
            .collect()
    }

    let first = build_and_observe();
    let second = build_and_observe();
    assert_eq!(
        first, second,
        "identical operation sequences must yield identical iteration order"
    );
    assert!(!first.is_empty());
}

#[test]
fn determinism_commands_apply_in_recorded_order() {
    fn run() -> Vec<i32> {
        let mut world = World::new();
        for value in 0..20i32 {
            world.spawn((Position(value),));
        }
        let mut commands = Commands::new();
        for (entity, (position,)) in world.query::<(&Position,)>() {
            if position.0 % 2 == 0 {
                commands.insert(entity, Velocity(position.0));
            } else {
                commands.despawn(entity);
            }
        }
        commands.apply(&mut world);
        let mut remaining: Vec<i32> = world.query::<(&Position,)>().map(|(_, (p,))| p.0).collect();
        remaining.sort_unstable();
        remaining
    }

    assert_eq!(run(), run());
    assert_eq!(
        run(),
        (0..20).filter(|value| value % 2 == 0).collect::<Vec<i32>>()
    );
}

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct Calendar {
    day: u32,
}

#[test]
fn resources_are_reachable_from_systems() {
    let mut world = World::new();
    world.insert_resource(Calendar { day: 1 });
    world.expect_resource_mut::<Calendar>().day += 1;
    assert_eq!(world.resource::<Calendar>(), Some(&Calendar { day: 2 }));
}

#[test]
#[should_panic(expected = "has not been inserted")]
fn expecting_a_missing_resource_names_the_type() {
    let world = World::new();
    let _ = world.expect_resource::<Calendar>();
}

#[test]
fn clearing_entities_leaves_resources_alone() {
    let mut world = World::new();
    world.insert_resource(Calendar { day: 5 });
    world.spawn((Position(1),));

    world.clear_entities();

    assert!(world.is_empty());
    assert_eq!(world.resource::<Calendar>(), Some(&Calendar { day: 5 }));
}

#[test]
fn component_ids_lists_both_storage_kinds() {
    let mut world = World::new();
    let entity = world.spawn((Position(1), Velocity(2)));
    world.insert(entity, Stunned(1));

    let ids = world.component_ids(entity);
    assert_eq!(
        ids.len(),
        3,
        "expected Position, Velocity and the sparse Stunned"
    );

    // Sorted, so the listing is stable for the editor inspector.
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
}
