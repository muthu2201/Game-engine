//! Typed iteration over archetype columns.
//!
//! # How a query runs
//!
//! 1. The requested component types are resolved to [`ComponentId`]s.
//! 2. [`Archetypes::matching`] returns every archetype holding all of them (and
//!    none of the excluded ones), in ascending order.
//! 3. For each matching archetype, one iterator is built per requested
//!    component, walking that component's column directly.
//! 4. The query yields tuples drawn from those iterators in lockstep.
//!
//! # Why this is safe without `unsafe`
//!
//! Two facts make the borrows check out. Within one archetype, the requested
//! columns are distinct elements of a `Vec`, so
//! [`slice::get_disjoint_mut`] hands out simultaneous mutable
//! references to them — and rejects a query that asks for the same component
//! twice, which would be an aliasing bug. Across archetypes, the per-archetype
//! iterator bundles are built up front through `Vec::iter_mut`, which yields
//! one `&mut Archetype` per element with non-overlapping lifetimes.
//!
//! Building the bundles eagerly is what lets [`Query`] be a real
//! [`Iterator`] rather than a lending iterator or a closure callback: the items
//! it yields borrow from the archetypes, not from the query itself. The cost is
//! proportional to the number of *matching archetypes*, not entities.
//!
//! # Limits
//!
//! * Components declared [`StorageKind::Sparse`](crate::StorageKind::Sparse) do
//!   not live in archetype columns, so they cannot appear in the fetched tuple.
//!   Read them with [`World::get`](crate::World::get), or filter on them with
//!   [`QueryBuilder::with`].
//! * A query borrows the world for as long as it is alive, so a system that
//!   needs to read a *different* entity mid-iteration should collect what it
//!   needs first. [`World::query_ref`](crate::World::query_ref) exists for the
//!   read-only case, where several queries can be alive at once.

use crate::component::{Component, ComponentId, ComponentRegistry, Tick};
use crate::entity::Entity;
use crate::storage::{Archetype, Column, TypedColumn};

/// One component slot in a query's fetched tuple.
///
/// Implemented for `&T` and `&mut T`. Not part of the public extension surface:
/// the set of possible fetches is fixed by the storage layout.
pub trait QueryParam {
    /// What one row of this slot yields.
    type Item<'w>;
    /// The iterator walking this slot's column.
    type Iter<'w>;

    /// True when the fetch hands out mutable references.
    const MUTABLE: bool;

    /// Resolves (and registers, if needed) this slot's component identifier.
    fn register(registry: &mut ComponentRegistry) -> ComponentId;

    /// Resolves this slot's identifier without registering it.
    ///
    /// Returns `None` when the component type is unknown to the world, which
    /// necessarily means no entity carries it and the query matches nothing.
    fn lookup(registry: &ComponentRegistry) -> Option<ComponentId>;

    /// Builds the column iterator.
    ///
    /// `tick` is stamped onto every row a mutable fetch yields, which is what
    /// drives change detection.
    fn make_iter<'w>(column: &'w mut dyn Column, tick: Tick) -> Self::Iter<'w>;

    /// Advances the column iterator.
    fn next<'w>(iter: &mut Self::Iter<'w>) -> Option<Self::Item<'w>>;
}

/// A [`QueryParam`] that only reads, and so can run against a shared world.
pub trait ReadOnlyQueryParam: QueryParam {
    /// Builds the column iterator from a shared borrow.
    fn make_iter_ref<'w>(column: &'w dyn Column) -> Self::Iter<'w>;
}

impl<T: Component> QueryParam for &T {
    type Item<'w> = &'w T;
    type Iter<'w> = std::slice::Iter<'w, T>;

    const MUTABLE: bool = false;

    fn register(registry: &mut ComponentRegistry) -> ComponentId {
        registry.register::<T>()
    }

    fn lookup(registry: &ComponentRegistry) -> Option<ComponentId> {
        registry.get_id::<T>()
    }

    fn make_iter<'w>(column: &'w mut dyn Column, _tick: Tick) -> Self::Iter<'w> {
        downcast_mut::<T>(column).data.iter()
    }

    fn next<'w>(iter: &mut Self::Iter<'w>) -> Option<Self::Item<'w>> {
        iter.next()
    }
}

impl<T: Component> ReadOnlyQueryParam for &T {
    fn make_iter_ref<'w>(column: &'w dyn Column) -> Self::Iter<'w> {
        downcast_ref::<T>(column).data.iter()
    }
}

impl<T: Component> QueryParam for &mut T {
    type Item<'w> = &'w mut T;
    type Iter<'w> = ChangeTrackingIter<'w, T>;

    const MUTABLE: bool = true;

    fn register(registry: &mut ComponentRegistry) -> ComponentId {
        registry.register::<T>()
    }

    fn lookup(registry: &ComponentRegistry) -> Option<ComponentId> {
        registry.get_id::<T>()
    }

    fn make_iter<'w>(column: &'w mut dyn Column, tick: Tick) -> Self::Iter<'w> {
        // Split the column into its value and tick vectors so both can be
        // walked mutably at once.
        let TypedColumn { data, changed, .. } = downcast_mut::<T>(column);
        ChangeTrackingIter {
            values: data.iter_mut(),
            ticks: changed.iter_mut(),
            tick,
        }
    }

    fn next<'w>(iter: &mut Self::Iter<'w>) -> Option<Self::Item<'w>> {
        iter.next()
    }
}

/// Walks a column's values, stamping each row's changed tick as it goes.
///
/// The stamp is applied on *access*, not on write, so a system that takes
/// `&mut T` and leaves the value alone still marks it changed. That is the
/// conservative direction: a spurious change wastes a little downstream work,
/// whereas a missed change silently drops a UI refresh or a network update.
pub struct ChangeTrackingIter<'w, T> {
    values: std::slice::IterMut<'w, T>,
    ticks: std::slice::IterMut<'w, Tick>,
    tick: Tick,
}

impl<'w, T> Iterator for ChangeTrackingIter<'w, T> {
    type Item = &'w mut T;

    fn next(&mut self) -> Option<&'w mut T> {
        let value = self.values.next()?;
        if let Some(slot) = self.ticks.next() {
            *slot = self.tick;
        }
        Some(value)
    }
}

/// Downcasts a type-erased column, with a message that names the failure mode.
fn downcast_mut<T: 'static>(column: &mut dyn Column) -> &mut TypedColumn<T> {
    column
        .as_any_mut()
        .downcast_mut::<TypedColumn<T>>()
        .expect("component column type mismatch: a ComponentId was mapped to the wrong column")
}

/// Shared-borrow counterpart of [`downcast_mut`].
fn downcast_ref<T: 'static>(column: &dyn Column) -> &TypedColumn<T> {
    column
        .as_any()
        .downcast_ref::<TypedColumn<T>>()
        .expect("component column type mismatch: a ComponentId was mapped to the wrong column")
}

/// A tuple of [`QueryParam`]s that a query fetches together.
pub trait QueryData {
    /// One row of the fetch.
    type Item<'w>;
    /// The per-archetype bundle of column iterators.
    type Iters<'w>;

    /// Appends this tuple's component identifiers, in fetch order.
    fn register(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>);

    /// Resolves every slot's identifier without registering.
    ///
    /// Returns `None` if any slot's component type is unknown to the world.
    fn lookup_ids(registry: &ComponentRegistry) -> Option<Vec<ComponentId>>;

    /// Builds the column iterators for one archetype.
    fn make_iters<'w>(
        signature: &[ComponentId],
        columns: &'w mut [Box<dyn Column>],
        ids: &[ComponentId],
        tick: Tick,
    ) -> Self::Iters<'w>;

    /// Advances every column iterator in lockstep.
    fn next<'w>(iters: &mut Self::Iters<'w>) -> Option<Self::Item<'w>>;
}

/// A [`QueryData`] whose every slot only reads.
pub trait ReadOnlyQueryData: QueryData {
    /// Builds the column iterators from a shared archetype borrow.
    fn make_iters_ref<'w>(
        signature: &[ComponentId],
        columns: &'w [Box<dyn Column>],
        ids: &[ComponentId],
    ) -> Self::Iters<'w>;
}

/// Resolves a component's column position within an archetype.
fn column_position(signature: &[ComponentId], id: ComponentId) -> usize {
    signature
        .binary_search(&id)
        .expect("query matched an archetype that lacks a required column")
}

macro_rules! impl_query_data {
    ($(($param:ident, $index:tt)),+) => {
        impl<$($param: QueryParam),+> QueryData for ($($param,)+) {
            type Item<'w> = ($($param::Item<'w>,)+);
            type Iters<'w> = ($($param::Iter<'w>,)+);

            fn register(registry: &mut ComponentRegistry, out: &mut Vec<ComponentId>) {
                $(out.push($param::register(registry));)+
            }

            fn lookup_ids(registry: &ComponentRegistry) -> Option<Vec<ComponentId>> {
                Some(vec![$($param::lookup(registry)?),+])
            }

            #[allow(non_snake_case)]
            fn make_iters<'w>(
                signature: &[ComponentId],
                columns: &'w mut [Box<dyn Column>],
                ids: &[ComponentId],
                tick: Tick,
            ) -> Self::Iters<'w> {
                let positions = [$(column_position(signature, ids[$index])),+];
                let [$($param,)+] = columns.get_disjoint_mut(positions).expect(
                    "a query must not request the same component more than once",
                );
                ($(<$param as QueryParam>::make_iter(&mut **$param, tick),)+)
            }

            fn next<'w>(iters: &mut Self::Iters<'w>) -> Option<Self::Item<'w>> {
                Some(($($param::next(&mut iters.$index)?,)+))
            }
        }

        impl<$($param: ReadOnlyQueryParam),+> ReadOnlyQueryData for ($($param,)+) {
            #[allow(non_snake_case)]
            fn make_iters_ref<'w>(
                signature: &[ComponentId],
                columns: &'w [Box<dyn Column>],
                ids: &[ComponentId],
            ) -> Self::Iters<'w> {
                ($(
                    <$param as ReadOnlyQueryParam>::make_iter_ref(
                        &**&columns[column_position(signature, ids[$index])],
                    ),
                )+)
            }
        }
    };
}

impl_query_data!((A, 0));
impl_query_data!((A, 0), (B, 1));
impl_query_data!((A, 0), (B, 1), (C, 2));
impl_query_data!((A, 0), (B, 1), (C, 2), (D, 3));
impl_query_data!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4));
impl_query_data!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5));
impl_query_data!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6));
impl_query_data!(
    (A, 0),
    (B, 1),
    (C, 2),
    (D, 3),
    (E, 4),
    (F, 5),
    (G, 6),
    (H, 7)
);

/// One archetype's worth of iteration state.
type Bundle<'w, Q> = (std::slice::Iter<'w, Entity>, <Q as QueryData>::Iters<'w>);

/// An iterator over every entity matching a query, yielding `(Entity, Item)`.
///
/// Created by [`World::query`](crate::World::query).
pub struct Query<'w, Q: QueryData> {
    remaining: std::vec::IntoIter<Bundle<'w, Q>>,
    current: Option<Bundle<'w, Q>>,
    gate: SparseGate,
}

/// Per-entity admission test for filters on sparse components.
///
/// Sparse components live outside the archetypes, so they cannot narrow the set
/// of matching *tables* the way a table filter does. Instead the world resolves
/// them to entity sets up front and the query skips rows that fail the test.
/// Building the sets costs one pass over the relevant sparse storages, which is
/// far cheaper than the alternative of consulting a hash map per yielded row.
#[derive(Debug, Default)]
pub(crate) struct SparseGate {
    /// Entities carrying every required sparse component. `None` means no
    /// sparse component was required, so nothing is filtered in.
    pub(crate) allowed: Option<std::collections::HashSet<Entity>>,
    /// Entities carrying at least one excluded sparse component.
    pub(crate) denied: std::collections::HashSet<Entity>,
}

impl SparseGate {
    /// True when the gate never rejects anything, letting `next` skip the test.
    fn is_open(&self) -> bool {
        self.allowed.is_none() && self.denied.is_empty()
    }

    /// True when `entity` passes the filters.
    fn admits(&self, entity: Entity) -> bool {
        if self.denied.contains(&entity) {
            return false;
        }
        match &self.allowed {
            Some(allowed) => allowed.contains(&entity),
            None => true,
        }
    }
}

impl<'w, Q: QueryData> Query<'w, Q> {
    /// Wraps the per-archetype bundles produced by the world.
    pub(crate) fn new(bundles: Vec<Bundle<'w, Q>>, gate: SparseGate) -> Query<'w, Q> {
        let mut remaining = bundles.into_iter();
        let current = remaining.next();
        Query {
            remaining,
            current,
            gate,
        }
    }

    /// An empty query, for the case where a requested component is unknown.
    pub(crate) fn empty() -> Query<'w, Q> {
        Query::new(Vec::new(), SparseGate::default())
    }
}

impl<'w, Q: QueryData> Iterator for Query<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (entities, iters) = self.current.as_mut()?;
            if let Some(entity) = entities.next() {
                // The column iterators must advance in lockstep with the entity
                // list even for rows the gate rejects, or the two fall out of
                // alignment and every later row yields the wrong components.
                let item = Q::next(iters)
                    .expect("column and entity lengths disagree: archetype invariant violated");
                if self.gate.is_open() || self.gate.admits(*entity) {
                    return Some((*entity, item));
                }
                continue;
            }
            // This archetype is exhausted; advance to the next one.
            self.current = self.remaining.next();
        }
    }
}

/// Accumulates the extra `with`/`without` constraints of a query.
///
/// Created by [`World::query_filtered`](crate::World::query_filtered).
#[derive(Debug, Default, Clone)]
pub struct QueryFilter {
    /// Components an entity must have, beyond those being fetched.
    pub(crate) required: Vec<ComponentId>,
    /// Components an entity must not have.
    pub(crate) excluded: Vec<ComponentId>,
}

impl QueryFilter {
    /// An unconstrained filter.
    #[must_use]
    pub fn new() -> QueryFilter {
        QueryFilter::default()
    }
}

/// Builds a query with additional presence constraints.
///
/// Filters accept components in either storage, so a sparse marker such as a
/// status effect can gate a query even though it cannot be fetched by one.
pub struct QueryBuilder<'w, Q: QueryData> {
    pub(crate) world: &'w mut crate::World,
    pub(crate) filter: QueryFilter,
    pub(crate) marker: std::marker::PhantomData<fn() -> Q>,
}

impl<'w, Q: QueryData> QueryBuilder<'w, Q> {
    /// Restricts the query to entities that also carry `T`.
    #[must_use]
    pub fn with<T: Component>(mut self) -> QueryBuilder<'w, Q> {
        let id = self.world.registry_mut().register::<T>();
        self.filter.required.push(id);
        self
    }

    /// Restricts the query to entities that do **not** carry `T`.
    #[must_use]
    pub fn without<T: Component>(mut self) -> QueryBuilder<'w, Q> {
        let id = self.world.registry_mut().register::<T>();
        self.filter.excluded.push(id);
        self
    }

    /// Runs the query.
    #[must_use]
    pub fn iter(self) -> Query<'w, Q> {
        self.world.query_with_filter::<Q>(self.filter)
    }
}

impl<'w, Q: QueryData> IntoIterator for QueryBuilder<'w, Q> {
    type Item = (Entity, Q::Item<'w>);
    type IntoIter = Query<'w, Q>;

    fn into_iter(self) -> Query<'w, Q> {
        self.iter()
    }
}

/// Splits an archetype into the pieces a query bundle needs.
///
/// Destructuring rather than method calls is what lets the entity list be
/// borrowed immutably while the columns are borrowed mutably.
pub(crate) fn split_archetype(
    archetype: &mut Archetype,
) -> (&[ComponentId], &[Entity], &mut [Box<dyn Column>]) {
    let Archetype {
        components,
        entities,
        columns,
    } = archetype;
    (components, entities, columns)
}
