//! The world: entities, their components, and the shared resources systems read.

use crate::component::{Component, ComponentId, ComponentRegistry, StorageKind, Tick};
use crate::entity::{Entity, EntityAllocator, EntityLocation};
use crate::query::{
    split_archetype, Query, QueryBuilder, QueryData, QueryFilter, ReadOnlyQueryData, SparseGate,
};
use crate::resource::Resources;
use crate::storage::{Archetypes, SparseSet, SparseStorage, TypedColumn};
use std::collections::{HashMap, HashSet};

/// Everything that makes up one simulation state.
///
/// A world owns its entities, their components, and a set of singleton
/// resources. It is `Send`, so a world can be built on a worker thread — level
/// generation does exactly that — and moved to the main thread when ready.
///
/// # Structural changes during iteration
///
/// Spawning, despawning and adding or removing components move rows between
/// archetypes, which would invalidate a query mid-flight. The borrow checker
/// prevents it outright: a [`Query`] holds the world mutably. When a system
/// needs to make structural changes while iterating, it records them in a
/// [`Commands`] buffer and the changes are applied at the next merge point.
pub struct World {
    entities: EntityAllocator,
    archetypes: Archetypes,
    registry: ComponentRegistry,
    /// Non-fragmenting storage, one sparse set per sparse component type.
    sparse: HashMap<ComponentId, Box<dyn SparseStorage>>,
    /// Builders for empty sparse sets, captured at registration time so the
    /// world can create storage for a component id without being generic.
    sparse_factories: HashMap<ComponentId, fn() -> Box<dyn SparseStorage>>,
    resources: Resources,
    tick: Tick,
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

impl World {
    /// Creates an empty world.
    #[must_use]
    pub fn new() -> World {
        World {
            entities: EntityAllocator::new(),
            archetypes: Archetypes::new(),
            registry: ComponentRegistry::new(),
            sparse: HashMap::new(),
            sparse_factories: HashMap::new(),
            resources: Resources::new(),
            tick: Tick(1),
        }
    }

    // -------------------------------------------------------------------------
    // Ticks and change detection
    // -------------------------------------------------------------------------

    /// The world's current logical tick.
    #[inline]
    #[must_use]
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// Advances the logical clock. Called once per simulation step.
    #[inline]
    pub fn advance_tick(&mut self) {
        self.tick.advance();
    }

    /// Overwrites the logical clock.
    ///
    /// Used by [`SnapshotRegistry::restore`](crate::SnapshotRegistry::restore)
    /// so change detection stays consistent with the loaded state.
    #[inline]
    pub fn set_tick(&mut self, tick: u32) {
        self.tick = Tick(tick);
    }

    /// The tick at which `entity` last had `T` written, if it has one.
    ///
    /// Compare against a system's previously observed tick to find work:
    /// `world.changed_tick::<Sprite>(e).is_some_and(|t| t.is_newer_than(last))`.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    #[must_use]
    pub fn changed_tick<T: Component>(&self, entity: Entity) -> Option<Tick> {
        let id = self.registry.get_id::<T>()?;
        match T::STORAGE {
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let archetype = self.archetypes.get(location.archetype);
                let column = archetype.column_index(id)?;
                let typed = archetype.columns[column]
                    .as_any()
                    .downcast_ref::<TypedColumn<T>>()
                    .expect("column type mismatch");
                typed.changed.get(location.row as usize).copied()
            }
            StorageKind::Sparse => {
                let set = self.sparse_set::<T>()?;
                let position = set.entities().iter().position(|stored| *stored == entity)?;
                set.changed.get(position).copied()
            }
        }
    }

    /// The tick at which `entity` gained `T`, if it has one.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    #[must_use]
    pub fn added_tick<T: Component>(&self, entity: Entity) -> Option<Tick> {
        let id = self.registry.get_id::<T>()?;
        match T::STORAGE {
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let archetype = self.archetypes.get(location.archetype);
                let column = archetype.column_index(id)?;
                let typed = archetype.columns[column]
                    .as_any()
                    .downcast_ref::<TypedColumn<T>>()
                    .expect("column type mismatch");
                typed.added.get(location.row as usize).copied()
            }
            StorageKind::Sparse => {
                let set = self.sparse_set::<T>()?;
                let position = set.entities().iter().position(|stored| *stored == entity)?;
                set.added.get(position).copied()
            }
        }
    }

    // -------------------------------------------------------------------------
    // Entity lifecycle
    // -------------------------------------------------------------------------

    /// Creates an entity with no components.
    pub fn spawn_empty(&mut self) -> Entity {
        let entity = self.entities.allocate();
        let row = self.archetypes.push_entity(0, entity);
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype: 0,
                row: row_index(row),
            },
        );
        entity
    }

    /// Recreates an entity with a specific handle.
    ///
    /// Only a restore should call this; see
    /// [`EntityAllocator::allocate_at`](crate::EntityAllocator). Returns `false`
    /// when the handle collides with a live entity at a different generation.
    pub fn spawn_at(&mut self, entity: Entity) -> bool {
        if !self.entities.allocate_at(entity) {
            return false;
        }
        let row = self.archetypes.push_entity(0, entity);
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype: 0,
                row: row_index(row),
            },
        );
        true
    }

    /// Creates an entity and immediately attaches `bundle`'s components.
    ///
    /// ```
    /// use verdant_core_ecs::{Component, World};
    ///
    /// #[derive(Debug, PartialEq)] struct Name(&'static str);
    /// impl Component for Name {}
    /// #[derive(Debug, PartialEq)] struct Level(u32);
    /// impl Component for Level {}
    ///
    /// let mut world = World::new();
    /// let hero = world.spawn((Name("Robin"), Level(3)));
    /// assert_eq!(world.get::<Level>(hero), Some(&Level(3)));
    /// ```
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.spawn_empty();
        bundle.insert_into(self, entity);
        entity
    }

    /// Removes an entity and every component it carries.
    ///
    /// Returns `false` when the handle was already dead, which makes despawning
    /// twice harmless — a common situation when two systems both react to the
    /// same death event in one tick.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        let Some(location) = self.entities.free(entity) else {
            return false;
        };
        // Drop the entity's table components, repairing the location of
        // whichever entity the swap-remove moved into the hole.
        if let Some(moved) = self
            .archetypes
            .swap_remove(location.archetype, location.row as usize)
        {
            self.entities.relocate_by_index(moved.index(), location);
        }
        // Drop its sparse components too.
        for storage in self.sparse.values_mut() {
            storage.remove_entity(entity);
        }
        true
    }

    /// True when the handle refers to a live entity.
    #[inline]
    #[must_use]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entities.contains(entity)
    }

    /// Number of live entities.
    #[inline]
    #[must_use]
    pub fn entity_count(&self) -> u32 {
        self.entities.len()
    }

    /// True when the world holds no entities.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Every live entity, in ascending index order.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter()
    }

    /// Removes every entity and component, keeping registered types and
    /// resources.
    ///
    /// Used when loading a save into an already-running application.
    pub fn clear_entities(&mut self) {
        self.entities.clear();
        self.archetypes.clear();
        for storage in self.sparse.values_mut() {
            storage.clear();
        }
    }

    // -------------------------------------------------------------------------
    // Components
    // -------------------------------------------------------------------------

    /// Attaches `component` to `entity`, replacing any existing value.
    ///
    /// Returns `false` when the entity is dead.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> bool {
        match T::STORAGE {
            StorageKind::Table => self.insert_table(entity, component),
            StorageKind::Sparse => self.insert_sparse(entity, component),
        }
    }

    /// Inserts a table component, migrating the entity's archetype if needed.
    fn insert_table<T: Component>(&mut self, entity: Entity, component: T) -> bool {
        let Some(location) = self.entities.location(entity) else {
            return false;
        };
        let id = self.registry.register::<T>();
        let tick = self.tick;

        // Already present: overwrite in place, no archetype change.
        if let Some(column) = self.archetypes.get(location.archetype).column_index(id) {
            let typed = self.archetypes.get_mut(location.archetype).columns[column]
                .as_any_mut()
                .downcast_mut::<TypedColumn<T>>()
                .expect("column type mismatch");
            let row = location.row as usize;
            typed.data[row] = component;
            typed.changed[row] = tick;
            return true;
        }

        let destination = self.archetypes.archetype_with(location.archetype, id, &|| {
            Box::new(TypedColumn::<T> {
                data: Vec::new(),
                added: Vec::new(),
                changed: Vec::new(),
            })
        });
        let (new_row, displaced) = self.archetypes.migrate(
            location.archetype,
            location.row as usize,
            destination,
            entity,
        );

        if let Some(moved) = displaced {
            self.entities.relocate_by_index(moved.index(), location);
        }

        // Fill in the newly added column for this row.
        let column = self
            .archetypes
            .get(destination)
            .column_index(id)
            .expect("the destination archetype was built to contain this component");
        let typed = self.archetypes.get_mut(destination).columns[column]
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("column type mismatch");
        typed.push(component, tick);

        self.entities.set_location(
            entity,
            EntityLocation {
                archetype: destination,
                row: row_index(new_row),
            },
        );
        true
    }

    /// Inserts a sparse component. Never touches the entity's archetype.
    fn insert_sparse<T: Component>(&mut self, entity: Entity, component: T) -> bool {
        if !self.entities.contains(entity) {
            return false;
        }
        let id = self.registry.register::<T>();
        let tick = self.tick;
        self.sparse_factories
            .entry(id)
            .or_insert(|| Box::new(SparseSet::<T>::new()));
        let storage = self
            .sparse
            .entry(id)
            .or_insert_with(|| Box::new(SparseSet::<T>::new()) as Box<dyn SparseStorage>);
        storage
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("sparse storage type mismatch")
            .insert(entity, component, tick);
        true
    }

    /// Detaches `T` from `entity`, returning the value if it was present.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        match T::STORAGE {
            StorageKind::Table => self.remove_table(entity),
            StorageKind::Sparse => {
                let id = self.registry.get_id::<T>()?;
                self.sparse
                    .get_mut(&id)?
                    .as_any_mut()
                    .downcast_mut::<SparseSet<T>>()
                    .expect("sparse storage type mismatch")
                    .remove(entity)
            }
        }
    }

    /// Removes a table component, migrating the entity to a smaller archetype.
    ///
    /// The value is taken out of its column first, which leaves that column the
    /// same length as its siblings will be once the migration removes the row.
    /// The migration then moves the surviving components across.
    fn remove_table<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let id = self.registry.get_id::<T>()?;
        let location = self.entities.location(entity)?;
        let column = self.archetypes.get(location.archetype).column_index(id)?;
        let row = location.row as usize;

        let boxed = self.archetypes.get_mut(location.archetype).columns[column].take_row(row);
        let value = *boxed.downcast::<T>().expect("column type mismatch");

        let destination = self.archetypes.archetype_without(location.archetype, id);
        let (new_row, displaced) =
            self.archetypes
                .migrate(location.archetype, row, destination, entity);
        if let Some(moved) = displaced {
            self.entities.relocate_by_index(moved.index(), location);
        }
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype: destination,
                row: row_index(new_row),
            },
        );
        Some(value)
    }

    /// Borrows `entity`'s `T`, if it has one.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    #[must_use]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let id = self.registry.get_id::<T>()?;
        match T::STORAGE {
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let archetype = self.archetypes.get(location.archetype);
                let column = archetype.column_index(id)?;
                archetype.columns[column]
                    .as_any()
                    .downcast_ref::<TypedColumn<T>>()
                    .expect("column type mismatch")
                    .data
                    .get(location.row as usize)
            }
            StorageKind::Sparse => self.sparse_set::<T>()?.get(entity),
        }
    }

    /// Mutably borrows `entity`'s `T`, stamping its changed tick.
    ///
    /// # Panics
    ///
    /// Panics if a component column holds a value of a different type, which
    /// would mean the archetype signature and its columns had drifted apart.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let id = self.registry.get_id::<T>()?;
        let tick = self.tick;
        match T::STORAGE {
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let column = self.archetypes.get(location.archetype).column_index(id)?;
                let typed = self.archetypes.get_mut(location.archetype).columns[column]
                    .as_any_mut()
                    .downcast_mut::<TypedColumn<T>>()
                    .expect("column type mismatch");
                let row = location.row as usize;
                if row < typed.changed.len() {
                    typed.changed[row] = tick;
                }
                typed.data.get_mut(row)
            }
            StorageKind::Sparse => self
                .sparse
                .get_mut(&id)?
                .as_any_mut()
                .downcast_mut::<SparseSet<T>>()
                .expect("sparse storage type mismatch")
                .get_mut(entity, tick),
        }
    }

    /// True when `entity` carries `T`.
    #[must_use]
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        let Some(id) = self.registry.get_id::<T>() else {
            return false;
        };
        match T::STORAGE {
            StorageKind::Table => self.entities.location(entity).is_some_and(|location| {
                self.archetypes
                    .get(location.archetype)
                    .column_index(id)
                    .is_some()
            }),
            StorageKind::Sparse => self
                .sparse
                .get(&id)
                .is_some_and(|storage| storage.contains_entity(entity)),
        }
    }

    /// Borrows the sparse set backing `T`, if any values have been stored.
    fn sparse_set<T: Component>(&self) -> Option<&SparseSet<T>> {
        let id = self.registry.get_id::<T>()?;
        self.sparse.get(&id).map(|storage| {
            storage
                .as_any()
                .downcast_ref::<SparseSet<T>>()
                .expect("sparse storage type mismatch")
        })
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /// Iterates every entity carrying all of `Q`'s components.
    ///
    /// ```
    /// use verdant_core_ecs::{Component, World};
    ///
    /// #[derive(Debug)] struct Position { x: i32 }
    /// impl Component for Position {}
    /// #[derive(Debug)] struct Velocity { x: i32 }
    /// impl Component for Velocity {}
    ///
    /// let mut world = World::new();
    /// world.spawn((Position { x: 0 }, Velocity { x: 2 }));
    /// world.spawn((Position { x: 5 }, Velocity { x: -1 }));
    ///
    /// for (_entity, (position, velocity)) in world.query::<(&mut Position, &Velocity)>() {
    ///     position.x += velocity.x;
    /// }
    /// ```
    pub fn query<Q: QueryData>(&mut self) -> Query<'_, Q> {
        self.query_with_filter(QueryFilter::new())
    }

    /// Starts a query with additional `with`/`without` constraints.
    ///
    /// ```
    /// # use verdant_core_ecs::{Component, World};
    /// # #[derive(Debug)] struct Position;
    /// # impl Component for Position {}
    /// # #[derive(Debug)] struct Frozen;
    /// # impl Component for Frozen {}
    /// # let mut world = World::new();
    /// # world.spawn((Position,));
    /// let moving: Vec<_> = world
    ///     .query_filtered::<(&Position,)>()
    ///     .without::<Frozen>()
    ///     .iter()
    ///     .collect();
    /// assert_eq!(moving.len(), 1);
    /// ```
    pub fn query_filtered<Q: QueryData>(&mut self) -> QueryBuilder<'_, Q> {
        QueryBuilder {
            world: self,
            filter: QueryFilter::new(),
            marker: std::marker::PhantomData,
        }
    }

    /// Runs a query with a prepared filter.
    ///
    /// # Panics
    ///
    /// Panics if `Q` fetches a component declared [`StorageKind::Sparse`],
    /// which has no archetype column to iterate.
    pub(crate) fn query_with_filter<Q: QueryData>(&mut self, filter: QueryFilter) -> Query<'_, Q> {
        let mut ids = Vec::new();
        Q::register(&mut self.registry, &mut ids);
        for id in &ids {
            if let Some(info) = self.registry.info(*id) {
                assert!(
                    info.storage == StorageKind::Table,
                    "component `{}` uses sparse storage and cannot be fetched by a query; \
                     read it with World::get, or filter on it with .with()",
                    info.name
                );
            }
        }

        // Table filters narrow which archetypes match; sparse filters become a
        // per-entity gate, because sparse components leave no trace in the
        // archetype signature.
        let mut table_required = ids.clone();
        let mut sparse_required = Vec::new();
        for id in filter.required {
            if self.is_table_component(id) {
                table_required.push(id);
            } else {
                sparse_required.push(id);
            }
        }
        let mut table_excluded = Vec::new();
        let mut sparse_excluded = Vec::new();
        for id in filter.excluded {
            if self.is_table_component(id) {
                table_excluded.push(id);
            } else {
                sparse_excluded.push(id);
            }
        }

        let gate = self.build_sparse_gate(&sparse_required, &sparse_excluded);
        let matching = self.archetypes.matching(&table_required, &table_excluded);
        let tick = self.tick;

        // `iter_mut` hands out one `&mut Archetype` per element with disjoint
        // lifetimes, which is what makes building every bundle up front safe.
        let mut bundles = Vec::with_capacity(matching.len());
        for (index, archetype) in self.archetypes.iter_mut().enumerate() {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            if !matching.contains(&index) {
                continue;
            }
            let (signature, entities, columns) = split_archetype(archetype);
            bundles.push((
                entities.iter(),
                Q::make_iters(signature, columns, &ids, tick),
            ));
        }
        Query::new(bundles, gate)
    }

    /// Resolves sparse filter components into the entity sets the query gate
    /// tests against.
    fn build_sparse_gate(&self, required: &[ComponentId], excluded: &[ComponentId]) -> SparseGate {
        let mut allowed: Option<HashSet<Entity>> = None;
        for id in required {
            let members: HashSet<Entity> = match self.sparse.get(id) {
                Some(storage) => storage.entity_set(),
                // A required component with no storage at all matches nothing.
                None => HashSet::new(),
            };
            allowed = Some(match allowed {
                Some(existing) => existing.intersection(&members).copied().collect(),
                None => members,
            });
        }
        let mut denied = HashSet::new();
        for id in excluded {
            if let Some(storage) = self.sparse.get(id) {
                denied.extend(storage.entity_set());
            }
        }
        SparseGate { allowed, denied }
    }

    /// Iterates a read-only query against a shared borrow of the world.
    ///
    /// Because it does not borrow the world mutably, several read-only queries
    /// — and any number of [`World::get`] calls — can be alive at once. That is
    /// the escape hatch for a system that needs to look at a second entity
    /// while walking a first.
    ///
    /// ```
    /// # use verdant_core_ecs::{Component, World};
    /// # #[derive(Debug)] struct Position(i32);
    /// # impl Component for Position {}
    /// # let mut world = World::new();
    /// # world.spawn((Position(1),));
    /// # world.spawn((Position(2),));
    /// let total: i32 = world.query_ref::<(&Position,)>().map(|(_, (p,))| p.0).sum();
    /// assert_eq!(total, 3);
    /// ```
    pub fn query_ref<Q: ReadOnlyQueryData>(&self) -> Query<'_, Q> {
        // A component type the world has never seen has no instances, so an
        // unresolvable identifier means the query matches nothing.
        let Some(ids) = Q::lookup_ids(&self.registry) else {
            return Query::empty();
        };
        let matching = self.archetypes.matching(&ids, &[]);
        let mut bundles = Vec::with_capacity(matching.len());
        for index in matching {
            let archetype = self.archetypes.get(index);
            bundles.push((
                archetype.entities.iter(),
                Q::make_iters_ref(&archetype.components, &archetype.columns, &ids),
            ));
        }
        Query::new(bundles, SparseGate::default())
    }

    /// True when `id` is stored in archetype columns.
    fn is_table_component(&self, id: ComponentId) -> bool {
        self.registry
            .info(id)
            .is_none_or(|info| info.storage == StorageKind::Table)
    }

    // -------------------------------------------------------------------------
    // Resources
    // -------------------------------------------------------------------------

    /// Inserts or replaces a singleton resource.
    pub fn insert_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.resources.insert(resource);
    }

    /// Borrows a resource.
    #[must_use]
    pub fn resource<R: 'static + Send + Sync>(&self) -> Option<&R> {
        self.resources.get::<R>()
    }

    /// Mutably borrows a resource.
    pub fn resource_mut<R: 'static + Send + Sync>(&mut self) -> Option<&mut R> {
        self.resources.get_mut::<R>()
    }

    /// Removes a resource, returning it.
    pub fn remove_resource<R: 'static + Send + Sync>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// Borrows a resource, panicking with the type name when it is absent.
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted. Use for resources the
    /// application inserts during setup, where absence is a wiring bug.
    #[must_use]
    pub fn expect_resource<R: 'static + Send + Sync>(&self) -> &R {
        self.resources
            .get::<R>()
            .unwrap_or_else(|| panic!("resource `{}` has not been inserted", type_name_of::<R>()))
    }

    /// Mutable counterpart of [`World::expect_resource`].
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted.
    pub fn expect_resource_mut<R: 'static + Send + Sync>(&mut self) -> &mut R {
        let name = type_name_of::<R>();
        self.resources
            .get_mut::<R>()
            .unwrap_or_else(|| panic!("resource `{name}` has not been inserted"))
    }

    // -------------------------------------------------------------------------
    // Introspection
    // -------------------------------------------------------------------------

    /// The component registry, for editor inspection and serialisation.
    #[must_use]
    pub fn registry(&self) -> &ComponentRegistry {
        &self.registry
    }

    /// Mutable access to the registry, so components can be registered ahead of
    /// first use.
    pub fn registry_mut(&mut self) -> &mut ComponentRegistry {
        &mut self.registry
    }

    /// Registers `T` without attaching it to anything.
    ///
    /// Worth doing at startup for every component the save system handles, so
    /// that identifiers are assigned in a stable order.
    pub fn register_component<T: Component>(&mut self) -> ComponentId {
        let id = self.registry.register::<T>();
        if T::STORAGE == StorageKind::Sparse {
            self.sparse_factories
                .entry(id)
                .or_insert(|| Box::new(SparseSet::<T>::new()));
            self.sparse
                .entry(id)
                .or_insert_with(|| Box::new(SparseSet::<T>::new()) as Box<dyn SparseStorage>);
        }
        id
    }

    /// How many live entities carry `T`.
    ///
    /// Walks the matching archetypes rather than every entity, so it is cheap
    /// enough for a debug overlay to call every frame.
    #[must_use]
    pub fn component_count<T: Component>(&self) -> usize {
        let Some(id) = self.registry.get_id::<T>() else {
            return 0;
        };
        match T::STORAGE {
            StorageKind::Table => self
                .archetypes
                .matching(&[id], &[])
                .iter()
                .map(|index| self.archetypes.get(*index).len())
                .sum(),
            StorageKind::Sparse => self.sparse_set::<T>().map_or(0, SparseSet::len),
        }
    }

    /// Number of archetypes, including the empty root. A useful health metric:
    /// runaway archetype counts mean component churn that should move to
    /// sparse storage.
    #[must_use]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// The identifiers of every component `entity` currently carries.
    #[must_use]
    pub fn component_ids(&self, entity: Entity) -> Vec<ComponentId> {
        let mut ids = match self.entities.location(entity) {
            Some(location) => self.archetypes.get(location.archetype).components.clone(),
            None => return Vec::new(),
        };
        for (id, storage) in &self.sparse {
            if storage.contains_entity(entity) {
                ids.push(*id);
            }
        }
        ids.sort_unstable();
        ids
    }
}

/// Converts a row index to the `u32` the location record stores.
#[inline]
fn row_index(row: usize) -> u32 {
    u32::try_from(row).expect("archetype row index exceeded u32::MAX")
}

/// The short type name of `R`, for diagnostics.
fn type_name_of<R: 'static>() -> &'static str {
    std::any::type_name::<R>()
}

/// A group of components inserted together.
///
/// Implemented for tuples of up to twelve components, which is what lets
/// [`World::spawn`] take `(Position, Velocity, Sprite)` directly.
pub trait Bundle {
    /// Attaches every component in the bundle to `entity`.
    fn insert_into(self, world: &mut World, entity: Entity);
}

impl<T: Component> Bundle for T {
    fn insert_into(self, world: &mut World, entity: Entity) {
        world.insert(entity, self);
    }
}

macro_rules! impl_bundle {
    ($($param:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($param: Component),+> Bundle for ($($param,)+) {
            fn insert_into(self, world: &mut World, entity: Entity) {
                let ($($param,)+) = self;
                $(world.insert(entity, $param);)+
            }
        }
    };
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L);
