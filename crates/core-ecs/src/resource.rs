//! Singleton values shared by every system in a world.
//!
//! Resources hold the state that is not *per entity*: the calendar, the input
//! snapshot, the world seed, the asset registry. Keying them by [`TypeId`]
//! means a system asks for what it needs by type rather than threading a
//! growing context struct through every call.

use std::any::{type_name, Any, TypeId};
use std::collections::HashMap;

/// A type-keyed map of singleton values.
#[derive(Default)]
pub struct Resources {
    /// Boxed values keyed by their concrete type.
    ///
    /// The `Send + Sync` bound is what keeps [`World`](crate::World) movable
    /// between threads, so a level can be generated off the main thread.
    values: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Type names, kept alongside for diagnostics and editor listings.
    names: HashMap<TypeId, &'static str>,
}

impl Resources {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Resources {
        Resources::default()
    }

    /// Inserts or replaces the value of type `R`.
    pub fn insert<R: 'static + Send + Sync>(&mut self, value: R) {
        let type_id = TypeId::of::<R>();
        self.names.insert(type_id, type_name::<R>());
        self.values.insert(type_id, Box::new(value));
    }

    /// Borrows the value of type `R`.
    #[must_use]
    pub fn get<R: 'static + Send + Sync>(&self) -> Option<&R> {
        self.values
            .get(&TypeId::of::<R>())
            .and_then(|value| value.downcast_ref::<R>())
    }

    /// Mutably borrows the value of type `R`.
    pub fn get_mut<R: 'static + Send + Sync>(&mut self) -> Option<&mut R> {
        self.values
            .get_mut(&TypeId::of::<R>())
            .and_then(|value| value.downcast_mut::<R>())
    }

    /// Removes and returns the value of type `R`.
    pub fn remove<R: 'static + Send + Sync>(&mut self) -> Option<R> {
        let boxed = self.values.remove(&TypeId::of::<R>())?;
        self.names.remove(&TypeId::of::<R>());
        boxed.downcast::<R>().ok().map(|value| *value)
    }

    /// True when a value of type `R` is present.
    #[must_use]
    pub fn contains<R: 'static + Send + Sync>(&self) -> bool {
        self.values.contains_key(&TypeId::of::<R>())
    }

    /// Borrows `R`, inserting the value produced by `default` if absent.
    ///
    /// # Panics
    ///
    /// Panics if the slot somehow holds a value of a different type, which
    /// would mean the type-id keying had been corrupted.
    pub fn get_or_insert_with<R: 'static + Send + Sync>(
        &mut self,
        default: impl FnOnce() -> R,
    ) -> &mut R {
        let type_id = TypeId::of::<R>();
        self.names.entry(type_id).or_insert_with(type_name::<R>);
        self.values
            .entry(type_id)
            .or_insert_with(|| Box::new(default()))
            .downcast_mut::<R>()
            .expect("resource type mismatch")
    }

    /// Number of stored resources.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when nothing is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The type names of every stored resource, sorted.
    ///
    /// Sorted because the underlying map has no stable order, and this feeds
    /// the editor's inspector and diagnostic dumps.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.names.values().copied().collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Calendar {
        day: u32,
    }

    #[derive(Debug, PartialEq)]
    struct Seed(u64);

    #[test]
    fn insert_and_borrow_round_trip() {
        let mut resources = Resources::new();
        resources.insert(Calendar { day: 3 });
        assert_eq!(resources.get::<Calendar>(), Some(&Calendar { day: 3 }));
        assert!(resources.contains::<Calendar>());
    }

    #[test]
    fn distinct_types_do_not_collide() {
        let mut resources = Resources::new();
        resources.insert(Calendar { day: 1 });
        resources.insert(Seed(42));
        assert_eq!(resources.get::<Calendar>().unwrap().day, 1);
        assert_eq!(resources.get::<Seed>().unwrap().0, 42);
        assert_eq!(resources.len(), 2);
    }

    #[test]
    fn inserting_twice_replaces() {
        let mut resources = Resources::new();
        resources.insert(Seed(1));
        resources.insert(Seed(2));
        assert_eq!(resources.get::<Seed>(), Some(&Seed(2)));
        assert_eq!(resources.len(), 1);
    }

    #[test]
    fn mutation_is_visible_afterwards() {
        let mut resources = Resources::new();
        resources.insert(Calendar { day: 1 });
        resources.get_mut::<Calendar>().unwrap().day = 9;
        assert_eq!(resources.get::<Calendar>().unwrap().day, 9);
    }

    #[test]
    fn removing_returns_the_value_and_clears_the_slot() {
        let mut resources = Resources::new();
        resources.insert(Seed(7));
        assert_eq!(resources.remove::<Seed>(), Some(Seed(7)));
        assert!(!resources.contains::<Seed>());
        assert_eq!(resources.remove::<Seed>(), None);
        assert!(resources.names().is_empty());
    }

    #[test]
    fn missing_resources_report_none_rather_than_panicking() {
        let resources = Resources::new();
        assert_eq!(resources.get::<Seed>(), None);
        assert!(resources.is_empty());
    }

    #[test]
    fn get_or_insert_with_only_calls_the_default_once() {
        let mut resources = Resources::new();
        let mut calls = 0;
        for _ in 0..3 {
            let value = resources.get_or_insert_with(|| {
                calls += 1;
                Seed(5)
            });
            value.0 += 1;
        }
        assert_eq!(calls, 1, "the default must only run when the slot is empty");
        assert_eq!(resources.get::<Seed>(), Some(&Seed(8)));
    }

    #[test]
    fn names_are_sorted_for_stable_diagnostics() {
        let mut resources = Resources::new();
        resources.insert(Seed(0));
        resources.insert(Calendar { day: 0 });
        let names = resources.names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(names.len(), 2);
    }
}
