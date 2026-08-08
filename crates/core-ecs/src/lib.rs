//! # Verdant core ECS
//!
//! A hybrid archetype / sparse-set entity component system.
//!
//! ## The hybrid storage model
//!
//! Benchmarks of ECS designs keep reaching the same conclusion: archetype
//! storage wins on iteration because entities with identical component sets
//! share tightly packed columns, and sparse-set storage wins on add/remove
//! because attaching a component never moves anything. Neither wins outright,
//! so this engine does both and lets the component type choose:
//!
//! ```
//! use verdant_core_ecs::{Component, StorageKind};
//!
//! // Iterated every frame by the movement system: table storage.
//! struct Position { x: f32, y: f32 }
//! impl Component for Position {}
//!
//! // Applied and cleared many times a second: sparse storage, so toggling it
//! // never rewrites the entity's other components.
//! struct Stunned { remaining_ticks: u32 }
//! impl Component for Stunned {
//!     const STORAGE: StorageKind = StorageKind::Sparse;
//! }
//! ```
//!
//! The trade-off is explicit and enforced: sparse components cannot be fetched
//! by a [`Query`] (they have no column to iterate), but they can gate one via
//! [`QueryBuilder::with`] and be read with [`World::get`]. A component that a
//! system iterates in bulk belongs in a table; a component that a system checks
//! for, or that churns, belongs in a sparse set.
//!
//! ## Determinism
//!
//! Everything that can affect simulation state is ordered:
//!
//! * Query iteration visits archetypes in ascending index order, and entities
//!   within an archetype in row order.
//! * [`Commands`] replays queued structural changes in the order recorded.
//! * [`Schedule`] runs systems in a declared order, and rejects an ambiguous
//!   ordering rather than picking one arbitrarily.
//!
//! Nothing here iterates a `HashMap` in a way that reaches simulation state.
//!
//! ## Worked example
//!
//! ```
//! use verdant_core_ecs::{Commands, Component, Schedule, World};
//!
//! #[derive(Debug)] struct Position { x: i32 }
//! impl Component for Position {}
//! #[derive(Debug)] struct Velocity { x: i32 }
//! impl Component for Velocity {}
//! #[derive(Debug)] struct OutOfBounds;
//! impl Component for OutOfBounds {}
//!
//! let mut world = World::new();
//! world.spawn((Position { x: 0 }, Velocity { x: 3 }));
//! world.spawn((Position { x: 9 }, Velocity { x: 3 }));
//!
//! let mut schedule = Schedule::new();
//! schedule.add_system("movement", |world: &mut World| {
//!     for (_entity, (position, velocity)) in world.query::<(&mut Position, &Velocity)>() {
//!         position.x += velocity.x;
//!     }
//! });
//! schedule.add_system_after("cull", "movement", |world: &mut World| {
//!     let mut commands = Commands::new();
//!     for (entity, (position,)) in world.query::<(&Position,)>() {
//!         if position.x > 10 {
//!             commands.insert(entity, OutOfBounds);
//!         }
//!     }
//!     commands.apply(world);
//! });
//!
//! schedule.run(&mut world);
//!
//! let flagged = world.query_filtered::<(&Position,)>().with::<OutOfBounds>().iter().count();
//! assert_eq!(flagged, 1);
//! ```

#![doc(html_no_source)]

pub mod commands;
pub mod component;
pub mod entity;
pub mod query;
pub mod resource;
pub mod schedule;
pub mod snapshot;
pub mod storage;
pub mod world;

pub use commands::Commands;
pub use component::{Component, ComponentId, ComponentInfo, ComponentRegistry, StorageKind, Tick};
pub use entity::{Entity, EntityAllocator};
pub use query::{Query, QueryBuilder, QueryData, QueryFilter, QueryParam, ReadOnlyQueryData};
pub use resource::Resources;
pub use schedule::{Schedule, SystemAccess, SystemId};
pub use snapshot::{Snapshot, SnapshotRegistry};
pub use world::{Bundle, World};
