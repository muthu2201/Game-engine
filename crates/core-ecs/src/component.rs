//! Component registration, identity and change-detection ticks.

use std::any::{type_name, TypeId};
use std::collections::HashMap;

/// Marker trait for types that can be attached to an entity.
///
/// The blanket bounds are what the rest of the engine needs: `'static` so the
/// type can be keyed by [`TypeId`], and `Send + Sync` so a world can be moved
/// between threads (for background level generation or an off-thread save).
///
/// Implement it with the derive-free one-liner:
///
/// ```
/// use verdant_core_ecs::{Component, StorageKind};
///
/// #[derive(Debug, Clone, Copy)]
/// struct Health(i32);
/// impl Component for Health {}
///
/// // A high-churn marker that should not reshuffle archetypes on every add:
/// #[derive(Debug, Clone, Copy)]
/// struct Stunned;
/// impl Component for Stunned {
///     const STORAGE: StorageKind = StorageKind::Sparse;
/// }
/// ```
pub trait Component: 'static + Send + Sync {
    /// Where instances of this component are stored.
    ///
    /// Defaults to [`StorageKind::Table`], which is the right answer for
    /// anything a system iterates in bulk. See [`StorageKind`] for when to
    /// override it.
    const STORAGE: StorageKind = StorageKind::Table;
}

/// Which storage backend holds a component type.
///
/// This is the hybrid model the engine is built around: archetype tables by
/// default for cache-coherent iteration, with an opt-in non-fragmenting
/// alternative for component types whose add/remove churn would otherwise
/// dominate.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StorageKind {
    /// Stored in the archetype's columns, alongside every other table
    /// component of the same entity.
    ///
    /// Iteration is a linear walk over a contiguous `Vec<T>`, which is as
    /// cache-friendly as 2D gameplay gets. The cost is that adding or removing
    /// the component moves the entity to a different archetype, copying all of
    /// its other table components.
    Table,

    /// Stored in a standalone sparse set, keyed by entity index.
    ///
    /// Adding or removing does **not** change the entity's archetype — this is
    /// the non-fragmenting storage the blueprint calls for, and it is why a
    /// status effect that is applied and cleared many times per second does not
    /// thrash the tables.
    ///
    /// The trade-off is deliberate and load-bearing: sparse components cannot
    /// participate in the zipped column iteration a [`Query`](crate::Query)
    /// performs, because they do not live in the archetype. Reach them with
    /// [`World::get`](crate::World::get) / [`World::get_mut`](crate::World::get_mut),
    /// or filter on them with [`QueryBuilder::with`](crate::QueryBuilder::with).
    Sparse,
}

/// A dense, world-local identifier for a component type.
///
/// Interned per [`ComponentRegistry`] rather than derived from [`TypeId`] so
/// that archetype signatures are small integers that can be compared and sorted
/// cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ComponentId(pub(crate) u32);

impl ComponentId {
    /// The identifier as a plain index, for use as a slot in dense arrays.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Static facts about one registered component type.
#[derive(Clone, Debug)]
pub struct ComponentInfo {
    /// The interned identifier.
    pub id: ComponentId,
    /// The Rust path of the type, for diagnostics and the editor inspector.
    pub name: &'static str,
    /// Where instances live.
    pub storage: StorageKind,
}

/// Interns component types into [`ComponentId`]s.
#[derive(Debug, Default)]
pub struct ComponentRegistry {
    infos: Vec<ComponentInfo>,
    by_type: HashMap<TypeId, ComponentId>,
}

impl ComponentRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> ComponentRegistry {
        ComponentRegistry::default()
    }

    /// Returns the identifier for `T`, registering it on first use.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` component types are registered.
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(existing) = self.by_type.get(&type_id) {
            return *existing;
        }
        let id =
            ComponentId(u32::try_from(self.infos.len()).expect("component id space exhausted"));
        self.infos.push(ComponentInfo {
            id,
            name: type_name::<T>(),
            storage: T::STORAGE,
        });
        self.by_type.insert(type_id, id);
        id
    }

    /// Returns the identifier for `T` if it has already been registered.
    ///
    /// Queries use this rather than [`ComponentRegistry::register`] so that
    /// asking about a component type no entity has ever carried does not
    /// permanently add it to the registry.
    #[must_use]
    pub fn get_id<T: Component>(&self) -> Option<ComponentId> {
        self.by_type.get(&TypeId::of::<T>()).copied()
    }

    /// Looks up the recorded facts about a component type.
    #[must_use]
    pub fn info(&self, id: ComponentId) -> Option<&ComponentInfo> {
        self.infos.get(id.index())
    }

    /// The name of a registered component, or `"<unregistered>"`.
    #[must_use]
    pub fn name(&self, id: ComponentId) -> &'static str {
        self.infos
            .get(id.index())
            .map_or("<unregistered>", |info| info.name)
    }

    /// Number of registered component types.
    #[must_use]
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// True when nothing has been registered yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Every registered component, in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &ComponentInfo> {
        self.infos.iter()
    }
}

/// A monotonically increasing logical clock used for change detection.
///
/// The world advances its tick once per simulation step. Each component slot
/// records the tick it was added and the tick it was last written, so a system
/// can ask "what changed since I last ran?" by comparing against the tick it
/// saw previously — the same model Bevy uses, and cheap enough to maintain
/// unconditionally.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    Default,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Tick(pub u32);

impl Tick {
    /// The tick before any work has happened.
    pub const ZERO: Tick = Tick(0);

    /// Advances by one, wrapping at `u32::MAX`.
    ///
    /// Wrapping is safe here because ticks are only ever compared for recency
    /// over a short window; at 60 Hz the counter takes over two years of
    /// continuous play to wrap, and the comparison below tolerates it.
    #[inline]
    pub fn advance(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }

    /// True when this tick is strictly newer than `other`.
    ///
    /// Uses wrapping subtraction so the comparison stays correct across the
    /// counter's wrap point, where a naive `>` would report a fresh tick as
    /// ancient.
    #[inline]
    #[must_use]
    pub const fn is_newer_than(self, other: Tick) -> bool {
        self.0.wrapping_sub(other.0) > 0 && self.0.wrapping_sub(other.0) < u32::MAX / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position;
    impl Component for Position {}

    struct Velocity;
    impl Component for Velocity {}

    struct Stunned;
    impl Component for Stunned {
        const STORAGE: StorageKind = StorageKind::Sparse;
    }

    #[test]
    fn registering_the_same_type_twice_returns_the_same_id() {
        let mut registry = ComponentRegistry::new();
        let first = registry.register::<Position>();
        let second = registry.register::<Position>();
        assert_eq!(first, second);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn distinct_types_get_distinct_ids() {
        let mut registry = ComponentRegistry::new();
        assert_ne!(
            registry.register::<Position>(),
            registry.register::<Velocity>()
        );
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn lookup_does_not_register() {
        let mut registry = ComponentRegistry::new();
        assert_eq!(registry.get_id::<Position>(), None);
        assert_eq!(registry.len(), 0, "a lookup must not create an entry");
        let id = registry.register::<Position>();
        assert_eq!(registry.get_id::<Position>(), Some(id));
    }

    #[test]
    fn storage_kind_is_recorded_from_the_trait() {
        let mut registry = ComponentRegistry::new();
        let dense = registry.register::<Position>();
        let sparse = registry.register::<Stunned>();
        assert_eq!(registry.info(dense).unwrap().storage, StorageKind::Table);
        assert_eq!(registry.info(sparse).unwrap().storage, StorageKind::Sparse);
    }

    #[test]
    fn names_are_recorded_for_diagnostics() {
        let mut registry = ComponentRegistry::new();
        let id = registry.register::<Position>();
        assert!(registry.name(id).ends_with("Position"));
        assert_eq!(registry.name(ComponentId(999)), "<unregistered>");
    }

    #[test]
    fn ticks_compare_by_recency() {
        let old = Tick(5);
        let new = Tick(9);
        assert!(new.is_newer_than(old));
        assert!(!old.is_newer_than(new));
        assert!(!old.is_newer_than(old), "a tick is not newer than itself");
    }

    #[test]
    fn tick_comparison_survives_the_wrap_point() {
        let before_wrap = Tick(u32::MAX - 2);
        let mut after_wrap = before_wrap;
        for _ in 0..5 {
            after_wrap.advance();
        }
        assert_eq!(after_wrap.0, 2, "the counter wrapped as expected");
        assert!(
            after_wrap.is_newer_than(before_wrap),
            "a wrapped tick must still read as newer, not two billion steps older"
        );
    }
}
