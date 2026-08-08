//! The ECS systems that drive the solver.
//!
//! Each tick the physics stage does three things, in this order:
//!
//! 1. Rebuild the broadphase from every collider's current bounds.
//! 2. Integrate each moving body's velocity through [`move_and_slide`],
//!    consulting the tile grid and the broadphase for blockers.
//! 3. Recompute sensor overlaps, so gameplay can ask what is inside a trigger.
//!
//! Rebuilding the broadphase wholesale rather than incrementally updating it is
//! a deliberate simplification: at the entity counts a 2D life-sim reaches, a
//! full rebuild is a few microseconds and it removes an entire class of bug
//! where a body's recorded bounds drift out of sync with its transform.

use crate::broadphase::SpatialHash;
use crate::components::{BodyKind, Collider, Transform, Velocity};
use crate::grid::SolidGrid;
use crate::solver::move_and_slide;
use verdant_core_ecs::{Entity, World};
use verdant_core_math::{Fx, Rect, Vec2};

/// Tunable constants for the physics stage.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsConfig {
    /// Broadphase cell size. See [`SpatialHash`] for how to choose one.
    pub cell_size: Fx,
    /// Speeds below this are treated as stationary.
    ///
    /// Without a floor, a body decelerating asymptotically keeps requesting
    /// sub-pixel movement forever, which keeps it out of the "idle" animation
    /// state and defeats any is-anything-moving optimisation.
    pub sleep_threshold: Fx,
}

impl Default for PhysicsConfig {
    fn default() -> PhysicsConfig {
        PhysicsConfig {
            cell_size: Fx::from_num(32),
            sleep_threshold: Fx::from_ratio(1, 1000),
        }
    }
}

/// A pair of overlapping bodies, at least one of which is a sensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SensorOverlap {
    /// The sensor.
    pub sensor: Entity,
    /// The body inside it.
    pub other: Entity,
}

/// Physics state that persists between ticks.
///
/// Held as an ECS resource. It owns the broadphase (so its allocations are
/// reused) and the overlap list produced by the most recent step.
pub struct PhysicsWorld {
    config: PhysicsConfig,
    broadphase: SpatialHash,
    overlaps: Vec<SensorOverlap>,
}

impl PhysicsWorld {
    /// Creates physics state with the given configuration.
    #[must_use]
    pub fn new(config: PhysicsConfig) -> PhysicsWorld {
        PhysicsWorld {
            broadphase: SpatialHash::new(config.cell_size),
            config,
            overlaps: Vec::new(),
        }
    }

    /// The broadphase, for gameplay queries such as "what is near the player".
    #[must_use]
    pub fn broadphase(&self) -> &SpatialHash {
        &self.broadphase
    }

    /// The sensor overlaps detected by the most recent step.
    ///
    /// Sorted, so iterating them drives gameplay deterministically.
    #[must_use]
    pub fn overlaps(&self) -> &[SensorOverlap] {
        &self.overlaps
    }

    /// True when `sensor` currently contains `other`.
    #[must_use]
    pub fn is_overlapping(&self, sensor: Entity, other: Entity) -> bool {
        self.overlaps
            .iter()
            .any(|pair| pair.sensor == sensor && pair.other == other)
    }

    /// Everything currently inside `sensor`.
    #[must_use]
    pub fn contents_of(&self, sensor: Entity) -> Vec<Entity> {
        self.overlaps
            .iter()
            .filter(|pair| pair.sensor == sensor)
            .map(|pair| pair.other)
            .collect()
    }

    /// Advances the physics stage by `dt` seconds.
    ///
    /// `solids` is the tile grid the world's static geometry lives in.
    pub fn step(&mut self, world: &mut World, solids: &dyn SolidGrid, dt: Fx) {
        self.rebuild_broadphase(world);
        self.integrate(world, solids, dt);
        // Bodies moved, so the broadphase is stale; sensor tests must see where
        // things ended up, not where they started.
        self.rebuild_broadphase(world);
        self.collect_overlaps(world);
    }

    /// Repopulates the broadphase from current transforms.
    fn rebuild_broadphase(&mut self, world: &mut World) {
        self.broadphase.clear();
        for (entity, (transform, collider)) in world.query::<(&Transform, &Collider)>() {
            self.broadphase.insert(
                entity,
                collider.world_bounds(transform.position),
                collider.layer,
            );
        }
    }

    /// Moves every non-static body by its velocity.
    fn integrate(&mut self, world: &mut World, solids: &dyn SolidGrid, dt: Fx) {
        // Collect first: the solver needs the broadphase while the query would
        // otherwise still be borrowing the world.
        let mut movers: Vec<(Entity, Transform, Velocity, Collider)> = Vec::new();
        for (entity, (transform, velocity, collider)) in
            world.query::<(&Transform, &Velocity, &Collider)>()
        {
            if collider.kind == BodyKind::Static || collider.sensor {
                continue;
            }
            if velocity.0.length_squared() < self.config.sleep_threshold {
                continue;
            }
            movers.push((entity, *transform, *velocity, *collider));
        }

        for (entity, transform, velocity, collider) in movers {
            let displacement = velocity.0 * dt;
            // Only bodies that block this one are consulted; a projectile that
            // ignores NPCs must not be stopped by one.
            let swept = collider
                .world_bounds(transform.position)
                .union(collider.world_bounds(transform.position + displacement));
            let blockers = self.solid_blockers(&collider, swept, entity, world);

            let result = move_and_slide(
                &collider,
                transform.position,
                displacement,
                solids,
                &blockers,
            );

            if let Some(target) = world.get_mut::<Transform>(entity) {
                target.position = result.position;
            }
            world.insert(entity, result.flags);
        }
    }

    /// The bounds of every body that should block `collider`.
    fn solid_blockers(
        &self,
        collider: &Collider,
        area: Rect,
        exclude: Entity,
        world: &World,
    ) -> Vec<Rect> {
        self.broadphase
            .query(area, collider.mask)
            .into_iter()
            .filter(|entry| entry.entity != exclude)
            .filter(|entry| {
                // Sensors never block, and the layer agreement must hold in
                // both directions.
                world
                    .get::<Collider>(entry.entity)
                    .is_some_and(|other| !other.sensor && collider.interacts_with(other))
            })
            .map(|entry| entry.bounds)
            .collect()
    }

    /// Recomputes which bodies are inside which sensors.
    fn collect_overlaps(&mut self, world: &mut World) {
        self.overlaps.clear();

        let sensors: Vec<(Entity, Rect, Collider)> = world
            .query::<(&Transform, &Collider)>()
            .filter(|(_, (_, collider))| collider.sensor)
            .map(|(entity, (transform, collider))| {
                (entity, collider.world_bounds(transform.position), *collider)
            })
            .collect();

        for (sensor, bounds, collider) in sensors {
            for entry in self.broadphase.query(bounds, collider.mask) {
                if entry.entity == sensor {
                    continue;
                }
                let interacts = world
                    .get::<Collider>(entry.entity)
                    .is_some_and(|other| collider.interacts_with(other));
                if interacts {
                    self.overlaps.push(SensorOverlap {
                        sensor,
                        other: entry.entity,
                    });
                }
            }
        }
        // The broadphase already sorts within a query; sorting the whole list
        // makes the ordering independent of sensor discovery order too.
        self.overlaps
            .sort_unstable_by_key(|pair| (pair.sensor.to_bits(), pair.other.to_bits()));
    }
}

impl Default for PhysicsWorld {
    fn default() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsConfig::default())
    }
}

/// Applies a velocity directly to a transform, ignoring collision.
///
/// For entities that should move without being stopped — floating damage
/// numbers, weather particles, camera targets.
pub fn integrate_unobstructed(world: &mut World, dt: Fx) {
    for (_entity, (transform, velocity)) in world.query::<(&mut Transform, &Velocity)>() {
        transform.position += velocity.0 * dt;
    }
}

/// Moves `position` toward `target` at `speed`, returning the new velocity.
///
/// Shared by NPC steering and projectile homing so both produce identical
/// motion for identical inputs.
#[must_use]
pub fn seek(position: Vec2, target: Vec2, speed: Fx) -> Velocity {
    let offset = target - position;
    if offset.is_zero() {
        return Velocity::ZERO;
    }
    Velocity(offset.normalize() * speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{layers, CollisionFlags};
    use crate::grid::{OpenGrid, TestGrid};
    use verdant_core_math::{fx, IVec2};

    fn walking_body() -> Collider {
        Collider::centred(Fx::ONE, Fx::ONE)
            .with_kind(BodyKind::Kinematic)
            .with_layers(layers::PLAYER, layers::ALL)
    }

    fn wall_body() -> Collider {
        Collider::centred(fx(4), fx(4)).with_layers(layers::TERRAIN, layers::ALL)
    }

    #[test]
    fn a_moving_body_advances_by_its_velocity() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        let entity = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(10), Fx::ZERO),
            walking_body(),
        ));

        let mut physics = PhysicsWorld::default();
        // A power-of-two timestep so the assertion can be exact: one tenth of a
        // second is not representable in binary fixed point, and comparing
        // against it would be testing the representation rather than the solver.
        physics.step(&mut world, &grid, Fx::from_ratio(1, 2));

        let position = world.get::<Transform>(entity).unwrap().position;
        assert_eq!(position.x, fx(5), "10 units/s for half a second is 5 units");
    }

    #[test]
    fn static_bodies_never_move() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        let entity = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(10), fx(10)),
            wall_body(),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);
        assert_eq!(world.get::<Transform>(entity).unwrap().position, Vec2::ZERO);
    }

    #[test]
    fn a_body_below_the_sleep_threshold_does_not_move() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        let entity = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(Fx::from_raw(4), Fx::ZERO),
            walking_body(),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);
        assert_eq!(world.get::<Transform>(entity).unwrap().position, Vec2::ZERO);
    }

    #[test]
    fn tiles_stop_a_moving_body_and_set_its_flags() {
        let mut world = World::new();
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0)]);
        let entity = world.spawn((
            Transform::at(Vec2::new(fx(8), fx(8))),
            Velocity::new(fx(100), Fx::ZERO),
            walking_body(),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);

        assert!(world.get::<CollisionFlags>(entity).unwrap().right);
        assert!(world.get::<Transform>(entity).unwrap().position.x < fx(16));
    }

    #[test]
    fn one_body_blocks_another() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        world.spawn((Transform::at(Vec2::from_ints(10, 0)), wall_body()));
        let mover = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(100), Fx::ZERO),
            walking_body(),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);

        let position = world.get::<Transform>(mover).unwrap().position;
        assert!(
            position.x < fx(10),
            "walked through a wall to {}",
            position.x.to_f64()
        );
        assert!(world.get::<CollisionFlags>(mover).unwrap().right);
    }

    #[test]
    fn a_body_passes_through_something_on_a_layer_it_ignores() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        // An item the player does not collide with.
        world.spawn((
            Transform::at(Vec2::from_ints(5, 0)),
            Collider::centred(fx(4), fx(4)).with_layers(layers::ITEM, layers::ITEM),
        ));
        let mover = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(10), Fx::ZERO),
            walking_body().with_layers(layers::PLAYER, layers::TERRAIN),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);
        assert_eq!(world.get::<Transform>(mover).unwrap().position.x, fx(10));
    }

    #[test]
    fn sensors_report_contents_without_blocking() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        // Wide enough that the walker is still inside it at the end of the step,
        // so the test checks non-blocking and containment independently.
        let trigger = world.spawn((
            Transform::at(Vec2::from_ints(5, 0)),
            Collider::centred(fx(24), fx(24))
                .with_layers(layers::INTERACTABLE, layers::PLAYER)
                .as_sensor(),
        ));
        let walker = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(10), Fx::ZERO),
            walking_body().with_layers(layers::PLAYER, layers::ALL),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::ONE);

        assert_eq!(
            world.get::<Transform>(walker).unwrap().position.x,
            fx(10),
            "a sensor must not block movement"
        );
        assert!(physics.is_overlapping(trigger, walker));
        assert_eq!(physics.contents_of(trigger), vec![walker]);
    }

    #[test]
    fn overlaps_are_cleared_when_a_body_leaves() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        let trigger = world.spawn((
            Transform::at(Vec2::ZERO),
            Collider::centred(fx(4), fx(4))
                .with_layers(layers::INTERACTABLE, layers::PLAYER)
                .as_sensor(),
        ));
        let walker = world.spawn((
            Transform::at(Vec2::ZERO),
            Velocity::new(fx(100), Fx::ZERO),
            walking_body(),
        ));

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::from_ratio(1, 100));
        assert!(physics.is_overlapping(trigger, walker));

        physics.step(&mut world, &grid, Fx::ONE);
        assert!(
            !physics.is_overlapping(trigger, walker),
            "the walker left the trigger"
        );
        assert!(physics.overlaps().is_empty());
    }

    #[test]
    fn overlap_order_is_deterministic() {
        let mut world = World::new();
        let grid = OpenGrid::new(fx(16));
        world.spawn((
            Transform::at(Vec2::ZERO),
            Collider::centred(fx(20), fx(20))
                .with_layers(layers::INTERACTABLE, layers::PLAYER)
                .as_sensor(),
        ));
        for offset in 0..8i32 {
            world.spawn((Transform::at(Vec2::from_ints(offset, 0)), walking_body()));
        }

        let mut physics = PhysicsWorld::default();
        physics.step(&mut world, &grid, Fx::from_ratio(1, 60));
        let first: Vec<u64> = physics
            .overlaps()
            .iter()
            .map(|pair| pair.other.to_bits())
            .collect();

        physics.step(&mut world, &grid, Fx::from_ratio(1, 60));
        let second: Vec<u64> = physics
            .overlaps()
            .iter()
            .map(|pair| pair.other.to_bits())
            .collect();

        assert_eq!(first, second);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(first, sorted);
    }

    #[test]
    fn unobstructed_integration_ignores_geometry() {
        let mut world = World::new();
        let entity = world.spawn((Transform::at(Vec2::ZERO), Velocity::new(fx(3), fx(4))));
        integrate_unobstructed(&mut world, Fx::ONE);
        assert_eq!(
            world.get::<Transform>(entity).unwrap().position,
            Vec2::from_ints(3, 4)
        );
    }

    #[test]
    fn seek_produces_a_velocity_of_the_requested_speed() {
        let velocity = seek(Vec2::ZERO, Vec2::from_ints(3, 4), fx(10));
        assert!((velocity.0.length().to_f64() - 10.0).abs() < 1e-5);
        assert_eq!(seek(Vec2::ZERO, Vec2::ZERO, fx(10)), Velocity::ZERO);
    }

    #[test]
    fn determinism_a_full_step_sequence_reproduces() {
        let run = || {
            let mut world = World::new();
            let grid = TestGrid::with_solids(fx(16), &[IVec2::new(2, 0), IVec2::new(2, 1)]);
            for index in 0..12i32 {
                world.spawn((
                    Transform::at(Vec2::from_ints(index, index * 2)),
                    Velocity::new(fx(5), fx(3)),
                    walking_body(),
                ));
            }
            let mut physics = PhysicsWorld::default();
            for _ in 0..120 {
                physics.step(&mut world, &grid, Fx::from_ratio(1, 60));
            }
            let mut positions: Vec<(i64, i64)> = world
                .query::<(&Transform,)>()
                .map(|(_, (transform,))| {
                    (transform.position.x.to_raw(), transform.position.y.to_raw())
                })
                .collect();
            positions.sort_unstable();
            positions
        };
        assert_eq!(run(), run());
    }
}
