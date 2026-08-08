//! # Verdant 2D physics
//!
//! Deterministic kinematic collision for tile-based 2D games.
//!
//! ## What this is, and what it is not
//!
//! This is a **kinematic** solver, not a rigid-body simulation. There are no
//! forces, no mass, no restitution and no constraint iteration. Bodies move
//! exactly where gameplay tells them to, and stop exactly at geometry.
//!
//! That is a deliberate trade. A general solver (Rapier, Box2D) buys realistic
//! stacking, joints and continuous collision between arbitrary convex shapes —
//! none of which a farming and exploration game uses — and costs an iteration
//! count that must converge, which is exactly the kind of thing that makes
//! cross-platform determinism hard. Every operation here is a fixed number of
//! fixed-point steps, so a replay reproduces bit-exactly.
//!
//! Reach for a general solver if the game needs ragdolls, rope, or realistic
//! stacking. For anything that moves like a character in a tile world, this is
//! both simpler and more controllable.
//!
//! ## Pieces
//!
//! | Module | Role |
//! |---|---|
//! | [`components`] | [`Transform`], [`Velocity`], [`Collider`], [`CollisionFlags`] |
//! | [`solver`] | [`move_and_slide`], the axis-separated sub-stepping solver |
//! | [`grid`] | [`SolidGrid`], the interface to the world's tiles |
//! | [`broadphase`] | [`SpatialHash`], so overlap queries stay sub-quadratic |
//! | [`systems`] | The ECS systems that wire the above into a schedule |
//!
//! ## Example
//!
//! ```
//! use verdant_core_math::{fx, IVec2, Vec2};
//! use verdant_physics_2d::{move_and_slide, Collider, TestGrid};
//!
//! // A one-unit body walking east into a wall at tile (1, 0).
//! let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0)]);
//! let body = Collider::centred(fx(1), fx(1));
//! let result = move_and_slide(&body, Vec2::new(fx(8), fx(8)), Vec2::from_ints(20, 0), &grid, &[]);
//!
//! assert!(result.flags.right, "the solver reports the blocked side");
//! assert!(result.position.x < fx(16), "and stops the body at the wall");
//! ```

#![doc(html_no_source)]

pub mod broadphase;
pub mod components;
pub mod grid;
pub mod solver;
pub mod systems;

pub use broadphase::{BroadphaseEntry, SpatialHash};
pub use components::{layers, BodyKind, Collider, CollisionFlags, Transform, Velocity};
pub use grid::{OpenGrid, SolidGrid, TestGrid};
pub use solver::{is_position_clear, move_and_slide, MoveResult};
pub use systems::{PhysicsConfig, PhysicsWorld};
