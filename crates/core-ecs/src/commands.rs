//! Deferred structural changes.
//!
//! A [`Query`](crate::Query) borrows the world for as long as it is alive, so a
//! system cannot spawn, despawn, or add and remove components while iterating —
//! doing so would move rows between archetypes underneath the iterator. That is
//! the same constraint every archetype ECS has; the difference here is that the
//! borrow checker enforces it rather than leaving it as a documented hazard.
//!
//! [`Commands`] is the way through. A system records what it wants to happen,
//! and the recorded operations are applied at a merge point — the end of the
//! system, the end of a stage, or wherever the schedule chooses. Because the
//! queue is an ordered list that is replayed in insertion order, the resulting
//! world state is deterministic.

use crate::component::Component;
use crate::entity::Entity;
use crate::world::{Bundle, World};

/// One recorded structural change.
///
/// Boxed closures rather than an enum of concrete operations: an enum would
/// need a variant per component type, which is not expressible without making
/// the whole queue generic.
type Command = Box<dyn FnOnce(&mut World) + Send>;

/// A queue of structural changes to apply later.
///
/// ```
/// use verdant_core_ecs::{Commands, Component, World};
///
/// #[derive(Debug)] struct Health(i32);
/// impl Component for Health {}
/// #[derive(Debug)] struct Dead;
/// impl Component for Dead {}
///
/// let mut world = World::new();
/// world.spawn((Health(0),));
/// world.spawn((Health(5),));
///
/// let mut commands = Commands::new();
/// for (entity, (health,)) in world.query::<(&Health,)>() {
///     if health.0 <= 0 {
///         commands.insert(entity, Dead);
///     }
/// }
/// // The query's borrow has ended, so the changes can be applied.
/// commands.apply(&mut world);
/// ```
#[derive(Default)]
pub struct Commands {
    queue: Vec<Command>,
    /// Entities reserved by [`Commands::spawn`] but not yet created.
    ///
    /// Tracked only so [`Commands::len`] reports meaningful progress; the
    /// handles themselves are minted when the queue is applied.
    pending_spawns: usize,
}

impl Commands {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Commands {
        Commands::default()
    }

    /// Queues the creation of an entity carrying `bundle`.
    ///
    /// The entity's handle does not exist until the queue is applied, so this
    /// returns nothing. When a system needs the handle immediately — to store
    /// it in another component, say — spawn directly on the world before the
    /// query begins.
    pub fn spawn<B: Bundle + Send + 'static>(&mut self, bundle: B) {
        self.pending_spawns += 1;
        self.queue.push(Box::new(move |world| {
            world.spawn(bundle);
        }));
    }

    /// Queues the creation of an entity and passes its handle to `then`.
    ///
    /// This is the escape hatch for the case above: the callback runs at apply
    /// time with the freshly minted handle, so a spawned projectile can be
    /// registered with its emitter without a second pass.
    pub fn spawn_with<B, F>(&mut self, bundle: B, then: F)
    where
        B: Bundle + Send + 'static,
        F: FnOnce(&mut World, Entity) + Send + 'static,
    {
        self.pending_spawns += 1;
        self.queue.push(Box::new(move |world| {
            let entity = world.spawn(bundle);
            then(world, entity);
        }));
    }

    /// Queues the removal of an entity.
    ///
    /// Despawning an already-dead entity is a no-op, so two systems reacting to
    /// the same death in one tick is harmless.
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.push(Box::new(move |world| {
            world.despawn(entity);
        }));
    }

    /// Queues attaching `component` to `entity`.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.queue.push(Box::new(move |world| {
            world.insert(entity, component);
        }));
    }

    /// Queues attaching every component in `bundle` to `entity`.
    pub fn insert_bundle<B: Bundle + Send + 'static>(&mut self, entity: Entity, bundle: B) {
        self.queue.push(Box::new(move |world| {
            if world.contains(entity) {
                bundle.insert_into(world, entity);
            }
        }));
    }

    /// Queues detaching `T` from `entity`.
    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.queue.push(Box::new(move |world| {
            world.remove::<T>(entity);
        }));
    }

    /// Queues inserting or replacing a resource.
    pub fn insert_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.queue.push(Box::new(move |world| {
            world.insert_resource(resource);
        }));
    }

    /// Queues an arbitrary operation on the world.
    ///
    /// The general form the typed helpers above are built on. Reach for it when
    /// a change does not fit them — reordering a hierarchy, say.
    pub fn queue(&mut self, action: impl FnOnce(&mut World) + Send + 'static) {
        self.queue.push(Box::new(action));
    }

    /// Applies every queued change in the order it was recorded, then empties
    /// the queue so the buffer can be reused next tick.
    pub fn apply(&mut self, world: &mut World) {
        for command in self.queue.drain(..) {
            command(world);
        }
        self.pending_spawns = 0;
    }

    /// Number of queued operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// True when nothing is queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of entities queued for creation.
    #[must_use]
    pub fn pending_spawns(&self) -> usize {
        self.pending_spawns
    }

    /// Discards every queued operation without applying it.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.pending_spawns = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Debug, PartialEq)]
    struct Dead;
    impl Component for Dead {}

    #[test]
    fn queued_changes_do_nothing_until_applied() {
        let mut world = World::new();
        let mut commands = Commands::new();
        commands.spawn((Health(10),));
        assert_eq!(world.entity_count(), 0, "nothing happens before apply");
        assert_eq!(commands.len(), 1);

        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 1);
        assert!(commands.is_empty(), "applying drains the queue");
    }

    #[test]
    fn changes_are_applied_in_the_order_recorded() {
        let mut world = World::new();
        let entity = world.spawn((Health(1),));
        let mut commands = Commands::new();
        commands.insert(entity, Health(2));
        commands.insert(entity, Health(3));
        commands.apply(&mut world);
        assert_eq!(
            world.get::<Health>(entity),
            Some(&Health(3)),
            "the last write wins"
        );
    }

    #[test]
    fn tagging_entities_found_by_a_query() {
        let mut world = World::new();
        world.spawn((Health(0),));
        world.spawn((Health(7),));

        let mut commands = Commands::new();
        for (entity, (health,)) in world.query::<(&Health,)>() {
            if health.0 <= 0 {
                commands.insert(entity, Dead);
            }
        }
        commands.apply(&mut world);

        let dead: Vec<_> = world
            .query_filtered::<(&Health,)>()
            .with::<Dead>()
            .iter()
            .collect();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].1 .0, &Health(0));
    }

    #[test]
    fn despawning_an_already_dead_entity_is_harmless() {
        let mut world = World::new();
        let entity = world.spawn((Health(1),));
        let mut commands = Commands::new();
        commands.despawn(entity);
        commands.despawn(entity);
        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_with_receives_the_new_handle() {
        let mut world = World::new();
        let mut commands = Commands::new();
        commands.spawn_with((Health(4),), |world, entity| {
            world.insert(entity, Dead);
        });
        commands.apply(&mut world);

        let tagged: Vec<_> = world
            .query_filtered::<(&Health,)>()
            .with::<Dead>()
            .iter()
            .collect();
        assert_eq!(tagged.len(), 1);
    }

    #[test]
    fn inserting_a_bundle_onto_a_despawned_entity_is_skipped() {
        let mut world = World::new();
        let entity = world.spawn((Health(1),));
        let mut commands = Commands::new();
        commands.despawn(entity);
        commands.insert_bundle(entity, (Dead,));
        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn removal_is_queued_like_everything_else() {
        let mut world = World::new();
        let entity = world.spawn((Health(1), Dead));
        let mut commands = Commands::new();
        commands.remove::<Dead>(entity);
        assert!(world.has::<Dead>(entity));
        commands.apply(&mut world);
        assert!(!world.has::<Dead>(entity));
        assert_eq!(world.get::<Health>(entity), Some(&Health(1)));
    }

    #[test]
    fn clear_discards_without_applying() {
        let mut world = World::new();
        let mut commands = Commands::new();
        commands.spawn((Health(1),));
        commands.clear();
        commands.apply(&mut world);
        assert_eq!(world.entity_count(), 0);
        assert_eq!(commands.pending_spawns(), 0);
    }
}
