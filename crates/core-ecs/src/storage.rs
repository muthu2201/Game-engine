//! Columnar archetype tables and non-fragmenting sparse sets.
//!
//! # Layout
//!
//! Entities that carry exactly the same set of table components share an
//! archetype. An archetype stores its entities in one `Vec<Entity>` and each
//! component type in its own `Vec<T>` — structure of arrays, not array of
//! structures — so a system that reads only `Position` walks a tightly packed
//! run of positions rather than striding over unrelated fields.
//!
//! Component columns are type-erased behind the [`Column`] trait so an
//! archetype can hold an arbitrary mix of types. The erasure costs one virtual
//! call per column per archetype, paid once when a query starts iterating a
//! table, not once per entity.
//!
//! # Why swap-remove
//!
//! Removing a row swaps the last row into the hole. That keeps every column
//! densely packed with no tombstones, at the cost of reordering: the entity
//! that was last now sits at the removed row's index. Callers must therefore
//! update that entity's recorded location, which is why the removal methods
//! report which entity moved.

use crate::component::{ComponentId, Tick};
use crate::entity::Entity;
use std::any::Any;
use std::collections::HashMap;

/// Type-erased access to one component column.
///
/// Implemented only by [`TypedColumn`]; the trait exists so archetypes can hold
/// a heterogeneous set of columns in a single `Vec`.
///
/// Public because it appears in the signatures of [`QueryParam`](crate::QueryParam),
/// but not part of the supported surface: the set of column types is fixed by
/// the storage layout and cannot be extended from outside the crate.
#[doc(hidden)]
pub trait Column: Any + Send + Sync {
    /// Number of rows currently stored.
    fn len(&self) -> usize;

    /// True when the column holds no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes `row` by swapping the final row into its place.
    fn swap_remove(&mut self, row: usize);

    /// Removes `row` and returns its value boxed.
    ///
    /// This is how [`World::remove`](crate::World::remove) hands a component
    /// back to the caller: the row leaves this column exactly as a
    /// [`Column::swap_remove`] would, so the column stays aligned with its
    /// siblings for the archetype migration that follows.
    fn take_row(&mut self, row: usize) -> Box<dyn Any>;

    /// Moves `row` out of `self` and appends it to `destination`.
    ///
    /// Used when an entity gains or loses a component and must migrate to a
    /// different archetype. `destination` must hold the same component type;
    /// the implementation asserts this rather than corrupting memory.
    fn migrate_row(&mut self, row: usize, destination: &mut dyn Column);

    /// Creates another empty column of the same component type.
    fn empty_clone(&self) -> Box<dyn Column>;

    /// Upcast for downcasting to the concrete [`TypedColumn`].
    fn as_any(&self) -> &dyn Any;

    /// Mutable upcast.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A concrete component column: the values plus their change-detection ticks.
///
/// The three vectors are kept the same length and index-aligned. Storing the
/// ticks beside the data rather than inside it keeps `T` free of engine
/// bookkeeping, so a component stays a plain gameplay struct.
pub(crate) struct TypedColumn<T> {
    pub(crate) data: Vec<T>,
    pub(crate) added: Vec<Tick>,
    pub(crate) changed: Vec<Tick>,
}

impl<T> TypedColumn<T> {
    /// Creates an empty column.
    fn new() -> TypedColumn<T> {
        TypedColumn {
            data: Vec::new(),
            added: Vec::new(),
            changed: Vec::new(),
        }
    }

    /// Appends a value, stamping both ticks with `tick`.
    pub(crate) fn push(&mut self, value: T, tick: Tick) {
        self.data.push(value);
        self.added.push(tick);
        self.changed.push(tick);
    }
}

impl<T: 'static + Send + Sync> Column for TypedColumn<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn swap_remove(&mut self, row: usize) {
        self.data.swap_remove(row);
        self.added.swap_remove(row);
        self.changed.swap_remove(row);
    }

    fn take_row(&mut self, row: usize) -> Box<dyn Any> {
        self.added.swap_remove(row);
        self.changed.swap_remove(row);
        Box::new(self.data.swap_remove(row))
    }

    fn migrate_row(&mut self, row: usize, destination: &mut dyn Column) {
        let target = destination
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("migrate_row called with a column of a different component type");
        target.data.push(self.data.swap_remove(row));
        target.added.push(self.added.swap_remove(row));
        target.changed.push(self.changed.swap_remove(row));
    }

    fn empty_clone(&self) -> Box<dyn Column> {
        Box::new(TypedColumn::<T>::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A table holding every entity with an identical set of table components.
pub(crate) struct Archetype {
    /// Sorted component identifiers. Doubles as the archetype's signature.
    pub(crate) components: Vec<ComponentId>,
    /// The entities in this table, index-aligned with every column.
    pub(crate) entities: Vec<Entity>,
    /// One column per entry in `components`, in the same order.
    pub(crate) columns: Vec<Box<dyn Column>>,
}

impl Archetype {
    /// Creates an empty archetype for the given (already sorted) signature.
    fn new(components: Vec<ComponentId>, columns: Vec<Box<dyn Column>>) -> Archetype {
        debug_assert!(components.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert_eq!(components.len(), columns.len());
        Archetype {
            components,
            entities: Vec::new(),
            columns,
        }
    }

    /// Number of entities in this archetype.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entities.len()
    }

    /// Position of a component's column, or `None` when absent.
    ///
    /// Binary search rather than a hash lookup: signatures are short (a dozen
    /// components at most) and already sorted, so the search beats hashing.
    #[inline]
    pub(crate) fn column_index(&self, component: ComponentId) -> Option<usize> {
        self.components.binary_search(&component).ok()
    }

    /// True when this archetype holds every component in `required`.
    pub(crate) fn has_all(&self, required: &[ComponentId]) -> bool {
        required.iter().all(|id| self.column_index(*id).is_some())
    }

    /// True when this archetype holds none of the components in `excluded`.
    pub(crate) fn has_none(&self, excluded: &[ComponentId]) -> bool {
        excluded.iter().all(|id| self.column_index(*id).is_none())
    }

    /// Appends an entity with no component data yet, returning its row.
    fn push_entity(&mut self, entity: Entity) -> usize {
        self.entities.push(entity);
        self.entities.len() - 1
    }

    /// Removes a row from every column and the entity list.
    ///
    /// Returns the entity that was swapped into `row`, when the removed row was
    /// not the last one — its recorded location must be updated by the caller.
    fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        for column in &mut self.columns {
            column.swap_remove(row);
        }
        self.entities.swap_remove(row);
        self.entities.get(row).copied()
    }
}

/// Every archetype in a world, plus the indices that make lookups cheap.
#[derive(Default)]
pub(crate) struct Archetypes {
    list: Vec<Archetype>,
    /// Signature to archetype index.
    by_signature: HashMap<Vec<ComponentId>, u32>,
    /// For each component, the archetypes that contain it.
    ///
    /// Query matching intersects starting from the *shortest* of these lists
    /// rather than scanning every archetype. Without this index, a world that
    /// has accumulated hundreds of archetypes pays for all of them on every
    /// query — the scaling trap the engine's design notes call out.
    by_component: HashMap<ComponentId, Vec<u32>>,
    /// Cached "archetype + component" transitions, so repeatedly adding the
    /// same component to entities of the same shape does not re-derive the
    /// destination signature each time.
    add_edges: HashMap<(u32, ComponentId), u32>,
    remove_edges: HashMap<(u32, ComponentId), u32>,
}

impl Archetypes {
    /// Creates the archetype set, seeded with the empty archetype at index 0.
    pub(crate) fn new() -> Archetypes {
        let mut archetypes = Archetypes::default();
        archetypes.list.push(Archetype::new(Vec::new(), Vec::new()));
        archetypes.by_signature.insert(Vec::new(), 0);
        archetypes
    }

    /// Borrows an archetype by index.
    #[inline]
    pub(crate) fn get(&self, index: u32) -> &Archetype {
        &self.list[index as usize]
    }

    /// Mutably borrows an archetype by index.
    #[inline]
    pub(crate) fn get_mut(&mut self, index: u32) -> &mut Archetype {
        &mut self.list[index as usize]
    }

    /// Number of archetypes, including the empty one.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    /// The indices of every archetype matching a query's required and excluded
    /// component sets, in ascending order.
    ///
    /// Ascending order matters: it is what makes query iteration — and
    /// therefore any system that accumulates state while iterating — produce
    /// the same result on every machine.
    pub(crate) fn matching(&self, required: &[ComponentId], excluded: &[ComponentId]) -> Vec<u32> {
        // Start from the rarest required component's archetype list; every
        // match must appear in it, so it is an upper bound on the answer.
        let seed = required
            .iter()
            .filter_map(|id| self.by_component.get(id))
            .min_by_key(|candidates| candidates.len());

        let mut matches: Vec<u32> = match seed {
            Some(candidates) => candidates
                .iter()
                .copied()
                .filter(|index| {
                    let archetype = self.get(*index);
                    archetype.len() > 0
                        && archetype.has_all(required)
                        && archetype.has_none(excluded)
                })
                .collect(),
            // A query with no required components matches every non-empty
            // archetype that avoids the exclusions.
            None => (0..self.list.len())
                .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
                .filter(|index| {
                    let archetype = self.get(*index);
                    archetype.len() > 0 && archetype.has_none(excluded)
                })
                .collect(),
        };
        matches.sort_unstable();
        matches
    }

    /// Finds or creates the archetype for `signature`.
    ///
    /// `prototype_columns` supplies an empty column for each component in the
    /// signature, which is how the type information reaches a newly created
    /// archetype without the archetype layer being generic over component
    /// types.
    fn get_or_insert(
        &mut self,
        signature: Vec<ComponentId>,
        prototype_columns: Vec<Box<dyn Column>>,
    ) -> u32 {
        if let Some(existing) = self.by_signature.get(&signature) {
            return *existing;
        }
        let index = u32::try_from(self.list.len()).expect("archetype id space exhausted");
        for component in &signature {
            self.by_component.entry(*component).or_default().push(index);
        }
        self.by_signature.insert(signature.clone(), index);
        self.list.push(Archetype::new(signature, prototype_columns));
        index
    }

    /// The archetype reached by adding `component` to `from`.
    ///
    /// `make_column` is only called when the destination archetype does not yet
    /// exist, so the common path costs a single hash lookup.
    pub(crate) fn archetype_with(
        &mut self,
        from: u32,
        component: ComponentId,
        make_column: &dyn Fn() -> Box<dyn Column>,
    ) -> u32 {
        if let Some(cached) = self.add_edges.get(&(from, component)) {
            return *cached;
        }
        let mut signature = self.get(from).components.clone();
        let insert_at = match signature.binary_search(&component) {
            // Already present: the transition is a no-op.
            Ok(_) => {
                self.add_edges.insert((from, component), from);
                return from;
            }
            Err(position) => position,
        };
        signature.insert(insert_at, component);

        let mut columns: Vec<Box<dyn Column>> = self
            .get(from)
            .columns
            .iter()
            .map(|column| column.empty_clone())
            .collect();
        columns.insert(insert_at, make_column());

        let destination = self.get_or_insert(signature, columns);
        self.add_edges.insert((from, component), destination);
        destination
    }

    /// The archetype reached by removing `component` from `from`.
    pub(crate) fn archetype_without(&mut self, from: u32, component: ComponentId) -> u32 {
        if let Some(cached) = self.remove_edges.get(&(from, component)) {
            return *cached;
        }
        let mut signature = self.get(from).components.clone();
        let remove_at = match signature.binary_search(&component) {
            Ok(position) => position,
            // Not present: nothing to do.
            Err(_) => {
                self.remove_edges.insert((from, component), from);
                return from;
            }
        };
        signature.remove(remove_at);

        let mut columns: Vec<Box<dyn Column>> = self
            .get(from)
            .columns
            .iter()
            .map(|column| column.empty_clone())
            .collect();
        columns.remove(remove_at);

        let destination = self.get_or_insert(signature, columns);
        self.remove_edges.insert((from, component), destination);
        destination
    }

    /// Appends an entity to an archetype, returning its row.
    pub(crate) fn push_entity(&mut self, archetype: u32, entity: Entity) -> usize {
        self.get_mut(archetype).push_entity(entity)
    }

    /// Removes a row, reporting the entity swapped into its place.
    pub(crate) fn swap_remove(&mut self, archetype: u32, row: usize) -> Option<Entity> {
        self.get_mut(archetype).swap_remove(row)
    }

    /// Moves an entity's row from one archetype to another, carrying over every
    /// component the two signatures share.
    ///
    /// Returns the new row and the entity displaced by the swap-remove, if any.
    pub(crate) fn migrate(
        &mut self,
        from: u32,
        row: usize,
        to: u32,
        entity: Entity,
    ) -> (usize, Option<Entity>) {
        debug_assert_ne!(
            from, to,
            "migrate called with identical source and destination"
        );

        // Borrow both archetypes at once. `get_disjoint_mut` is what makes this
        // safe without unsafe code or a temporary buffer for the moved values.
        let (source, target) = {
            let [a, b] = self
                .list
                .get_disjoint_mut([from as usize, to as usize])
                .expect("archetype indices must be distinct and in range");
            (a, b)
        };

        // Walk both sorted signatures together, moving shared components.
        let mut source_column = 0usize;
        let mut target_column = 0usize;
        while source_column < source.components.len() && target_column < target.components.len() {
            let source_id = source.components[source_column];
            let target_id = target.components[target_column];
            match source_id.cmp(&target_id) {
                std::cmp::Ordering::Equal => {
                    let (source_columns, target_columns) =
                        (&mut source.columns, &mut target.columns);
                    source_columns[source_column]
                        .migrate_row(row, target_columns[target_column].as_mut());
                    source_column += 1;
                    target_column += 1;
                }
                // Present in the source only: dropped by this transition.
                std::cmp::Ordering::Less => source_column += 1,
                // Present in the target only: filled in by the caller.
                std::cmp::Ordering::Greater => target_column += 1,
            }
        }

        // Components present only in the source are dropped by this transition.
        // A column that migrated already shrank by one; one that did not still
        // holds the row, and is exactly one longer than the entity list will be.
        let rows_after_removal = source.entities.len() - 1;
        for column in &mut source.columns {
            if column.len() > rows_after_removal {
                column.swap_remove(row);
            }
        }

        source.entities.swap_remove(row);
        let displaced = source.entities.get(row).copied();
        let new_row = target.push_entity(entity);
        (new_row, displaced)
    }

    /// Removes every entity and archetype except the empty one.
    pub(crate) fn clear(&mut self) {
        *self = Archetypes::new();
    }

    /// Mutably iterates every archetype.
    ///
    /// Each yielded borrow is disjoint from the others, which is what lets a
    /// query build its per-archetype iterator bundles up front without unsafe.
    pub(crate) fn iter_mut(&mut self) -> std::slice::IterMut<'_, Archetype> {
        self.list.iter_mut()
    }
}

/// Type-erased access to one sparse component store.
pub(crate) trait SparseStorage: Any + Send + Sync {
    /// Drops the component for `entity`, if present.
    fn remove_entity(&mut self, entity: Entity);
    /// True when `entity` currently carries this component.
    fn contains_entity(&self, entity: Entity) -> bool;
    /// The entities carrying this component, as a set for query gating.
    fn entity_set(&self) -> std::collections::HashSet<Entity>;
    /// Drops every stored component.
    fn clear(&mut self);
    /// Upcast for downcasting to the concrete [`SparseSet`].
    fn as_any(&self) -> &dyn Any;
    /// Mutable upcast.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Sentinel meaning "this entity index has no entry in the dense arrays".
const ABSENT: u32 = u32::MAX;

/// A sparse set mapping entity indices to component values.
///
/// This is the non-fragmenting storage: adding or removing a component held
/// here does not touch the entity's archetype, so a status effect can be
/// toggled every tick without copying the entity's other components anywhere.
///
/// The `sparse` array is indexed by entity index and holds a position in the
/// dense arrays, so lookup is one bounds check and one indirection. The dense
/// arrays stay packed, so iteration over just this component is still linear.
pub(crate) struct SparseSet<T> {
    /// Entity index to dense position, or [`ABSENT`].
    sparse: Vec<u32>,
    /// Dense position to entity, so a swap-remove can fix the sparse array.
    dense_entities: Vec<Entity>,
    /// The component values, index-aligned with `dense_entities`.
    pub(crate) data: Vec<T>,
    /// Tick each value was added.
    pub(crate) added: Vec<Tick>,
    /// Tick each value was last written.
    pub(crate) changed: Vec<Tick>,
}

impl<T> SparseSet<T> {
    /// Creates an empty sparse set.
    pub(crate) fn new() -> SparseSet<T> {
        SparseSet {
            sparse: Vec::new(),
            dense_entities: Vec::new(),
            data: Vec::new(),
            added: Vec::new(),
            changed: Vec::new(),
        }
    }

    /// The dense position of an entity's value, validating the generation.
    #[inline]
    fn position(&self, entity: Entity) -> Option<usize> {
        let slot = *self.sparse.get(entity.index() as usize)?;
        if slot == ABSENT {
            return None;
        }
        // The generation check is what stops a recycled index from inheriting
        // the component of the entity that previously used that slot.
        let stored = *self.dense_entities.get(slot as usize)?;
        if stored == entity {
            Some(slot as usize)
        } else {
            None
        }
    }

    /// Inserts or overwrites the value for `entity`.
    pub(crate) fn insert(&mut self, entity: Entity, value: T, tick: Tick) {
        if let Some(position) = self.position(entity) {
            self.data[position] = value;
            self.changed[position] = tick;
            return;
        }
        let index = entity.index() as usize;
        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, ABSENT);
        }
        let slot = u32::try_from(self.dense_entities.len()).expect("sparse set overflow");
        self.sparse[index] = slot;
        self.dense_entities.push(entity);
        self.data.push(value);
        self.added.push(tick);
        self.changed.push(tick);
    }

    /// Borrows the value for `entity`.
    #[inline]
    pub(crate) fn get(&self, entity: Entity) -> Option<&T> {
        self.position(entity).map(|position| &self.data[position])
    }

    /// Mutably borrows the value for `entity`, stamping its changed tick.
    #[inline]
    pub(crate) fn get_mut(&mut self, entity: Entity, tick: Tick) -> Option<&mut T> {
        let position = self.position(entity)?;
        self.changed[position] = tick;
        Some(&mut self.data[position])
    }

    /// Removes and returns the value for `entity`.
    pub(crate) fn remove(&mut self, entity: Entity) -> Option<T> {
        let position = self.position(entity)?;
        let last = self.dense_entities.len() - 1;

        self.sparse[entity.index() as usize] = ABSENT;
        // Swap-remove keeps the dense arrays packed; the entity moved into the
        // hole needs its sparse entry repointed at the new position.
        if position != last {
            let moved = self.dense_entities[last];
            self.sparse[moved.index() as usize] =
                u32::try_from(position).expect("sparse set overflow");
        }
        self.dense_entities.swap_remove(position);
        self.added.swap_remove(position);
        self.changed.swap_remove(position);
        Some(self.data.swap_remove(position))
    }

    /// The entity at each dense position, for iteration.
    #[inline]
    pub(crate) fn entities(&self) -> &[Entity] {
        &self.dense_entities
    }

    /// Number of entities carrying this component.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T: 'static + Send + Sync> SparseStorage for SparseSet<T> {
    fn remove_entity(&mut self, entity: Entity) {
        self.remove(entity);
    }

    fn contains_entity(&self, entity: Entity) -> bool {
        self.position(entity).is_some()
    }

    fn entity_set(&self) -> std::collections::HashSet<Entity> {
        self.dense_entities.iter().copied().collect()
    }

    fn clear(&mut self) {
        self.sparse.clear();
        self.dense_entities.clear();
        self.data.clear();
        self.added.clear();
        self.changed.clear();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_column_swap_remove_keeps_the_vectors_aligned() {
        let mut column = TypedColumn::<u32>::new();
        for value in 0..5u32 {
            column.push(value, Tick(value));
        }
        column.swap_remove(1);
        assert_eq!(column.data, vec![0, 4, 2, 3]);
        assert_eq!(column.added, vec![Tick(0), Tick(4), Tick(2), Tick(3)]);
        assert_eq!(column.changed.len(), column.data.len());
    }

    #[test]
    fn migrating_a_row_moves_the_value_and_its_ticks() {
        let mut source = TypedColumn::<String>::new();
        source.push("a".to_string(), Tick(1));
        source.push("b".to_string(), Tick(2));
        let mut target = TypedColumn::<String>::new();

        source.migrate_row(0, &mut target);

        assert_eq!(target.data, vec!["a".to_string()]);
        assert_eq!(target.added, vec![Tick(1)]);
        assert_eq!(
            source.data,
            vec!["b".to_string()],
            "the row was removed from the source"
        );
    }

    #[test]
    #[should_panic(expected = "different component type")]
    fn migrating_between_mismatched_column_types_panics() {
        let mut source = TypedColumn::<u32>::new();
        source.push(1, Tick::ZERO);
        let mut target = TypedColumn::<String>::new();
        source.migrate_row(0, &mut target);
    }

    #[test]
    fn sparse_set_insert_and_lookup() {
        let mut set = SparseSet::<u32>::new();
        let a = Entity::from_parts(0, 0);
        let b = Entity::from_parts(5, 0);
        set.insert(a, 10, Tick(1));
        set.insert(b, 20, Tick(1));
        assert_eq!(set.get(a), Some(&10));
        assert_eq!(set.get(b), Some(&20));
        assert_eq!(set.get(Entity::from_parts(3, 0)), None);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn sparse_set_rejects_a_recycled_index_with_a_stale_generation() {
        let mut set = SparseSet::<u32>::new();
        let original = Entity::from_parts(4, 0);
        set.insert(original, 99, Tick(1));
        // The same slot, reused by a later entity.
        let recycled = Entity::from_parts(4, 1);
        assert_eq!(
            set.get(recycled),
            None,
            "the new entity must not inherit the old value"
        );
        assert_eq!(set.get(original), Some(&99));
    }

    #[test]
    fn sparse_set_reinsert_overwrites_without_growing() {
        let mut set = SparseSet::<u32>::new();
        let entity = Entity::from_parts(2, 0);
        set.insert(entity, 1, Tick(1));
        set.insert(entity, 2, Tick(5));
        assert_eq!(set.get(entity), Some(&2));
        assert_eq!(set.len(), 1);
        assert_eq!(set.changed[0], Tick(5));
        assert_eq!(
            set.added[0],
            Tick(1),
            "the added tick is preserved on overwrite"
        );
    }

    #[test]
    fn sparse_set_remove_repoints_the_swapped_entity() {
        let mut set = SparseSet::<u32>::new();
        let a = Entity::from_parts(0, 0);
        let b = Entity::from_parts(1, 0);
        let c = Entity::from_parts(2, 0);
        set.insert(a, 10, Tick::ZERO);
        set.insert(b, 20, Tick::ZERO);
        set.insert(c, 30, Tick::ZERO);

        assert_eq!(set.remove(a), Some(10));
        // `c` was swapped into a's slot; it must still resolve.
        assert_eq!(set.get(c), Some(&30));
        assert_eq!(set.get(b), Some(&20));
        assert_eq!(set.get(a), None);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn sparse_set_remove_of_an_absent_entity_is_none() {
        let mut set = SparseSet::<u32>::new();
        assert_eq!(set.remove(Entity::from_parts(7, 0)), None);
    }

    #[test]
    fn archetypes_start_with_an_empty_root() {
        let archetypes = Archetypes::new();
        assert_eq!(archetypes.len(), 1);
        assert!(archetypes.get(0).components.is_empty());
    }

    #[test]
    fn adding_a_component_creates_and_caches_a_transition() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let first = archetypes.archetype_with(0, ComponentId(1), &make);
        assert_ne!(first, 0);
        // The second call must hit the edge cache and not create a new table.
        let count_before = archetypes.len();
        let second = archetypes.archetype_with(0, ComponentId(1), &make);
        assert_eq!(first, second);
        assert_eq!(archetypes.len(), count_before);
    }

    #[test]
    fn adding_a_component_the_archetype_already_has_is_a_no_op() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let with_one = archetypes.archetype_with(0, ComponentId(1), &make);
        assert_eq!(
            archetypes.archetype_with(with_one, ComponentId(1), &make),
            with_one
        );
    }

    #[test]
    fn signatures_are_order_independent() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        // Build {1, 2} by adding in each order; both must land on one archetype.
        let one_then_two = {
            let a = archetypes.archetype_with(0, ComponentId(1), &make);
            archetypes.archetype_with(a, ComponentId(2), &make)
        };
        let two_then_one = {
            let a = archetypes.archetype_with(0, ComponentId(2), &make);
            archetypes.archetype_with(a, ComponentId(1), &make)
        };
        assert_eq!(one_then_two, two_then_one);
    }

    #[test]
    fn removing_a_component_returns_to_the_previous_archetype() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let with_one = archetypes.archetype_with(0, ComponentId(1), &make);
        assert_eq!(archetypes.archetype_without(with_one, ComponentId(1)), 0);
        // Removing something absent is a no-op.
        assert_eq!(
            archetypes.archetype_without(with_one, ComponentId(9)),
            with_one
        );
    }

    #[test]
    fn matching_skips_empty_archetypes() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let target = archetypes.archetype_with(0, ComponentId(1), &make);
        // Created but empty: a query must not report it.
        assert!(archetypes.matching(&[ComponentId(1)], &[]).is_empty());

        archetypes.push_entity(target, Entity::from_parts(0, 0));
        assert_eq!(archetypes.matching(&[ComponentId(1)], &[]), vec![target]);
    }

    #[test]
    fn matching_honours_exclusions() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let with_one = archetypes.archetype_with(0, ComponentId(1), &make);
        let with_both = archetypes.archetype_with(with_one, ComponentId(2), &make);
        archetypes.push_entity(with_one, Entity::from_parts(0, 0));
        archetypes.push_entity(with_both, Entity::from_parts(1, 0));

        let matches = archetypes.matching(&[ComponentId(1)], &[ComponentId(2)]);
        assert_eq!(matches, vec![with_one]);
    }

    #[test]
    fn matching_results_are_sorted() {
        let mut archetypes = Archetypes::new();
        let make = || Box::new(TypedColumn::<u32>::new()) as Box<dyn Column>;
        let mut created = Vec::new();
        for component in 1..6u32 {
            let index = archetypes.archetype_with(0, ComponentId(component), &make);
            archetypes.push_entity(index, Entity::from_parts(component, 0));
            created.push(index);
        }
        let matches = archetypes.matching(&[], &[]);
        let mut sorted = matches.clone();
        sorted.sort_unstable();
        assert_eq!(
            matches, sorted,
            "query iteration order must be deterministic"
        );
    }
}
