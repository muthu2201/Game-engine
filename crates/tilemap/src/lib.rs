//! # Verdant tilemaps
//!
//! Tile storage, autotiling, grid pathfinding and constraint-based layout
//! generation.
//!
//! | Module | Role |
//! |---|---|
//! | [`map`] | [`Tilemap`] and [`TileLayer`]; also the `SolidGrid` the collision solver reads |
//! | [`autotile`] | Neighbour masks and the 47-tile blob set |
//! | [`pathfinding`] | [`find_path`] (A*) and [`FlowField`] (Dijkstra) |
//! | [`wfc`] | [`Wfc`], wave function collapse for generated layouts |
//!
//! ## Example
//!
//! ```
//! use verdant_core_math::{fx, IVec2};
//! use verdant_tilemap::{
//!     find_path, Movement, TileId, TileLayer, TileProperties, Tilemap,
//! };
//!
//! const GRASS: TileId = TileId(1);
//! const WALL: TileId = TileId(2);
//!
//! let mut map = Tilemap::new(8, 8, fx(16));
//! map.set_properties(GRASS, TileProperties::GROUND);
//! map.set_properties(WALL, TileProperties::WALL);
//!
//! let mut ground = TileLayer::new("ground", 8, 8);
//! ground.clear_to(GRASS);
//! ground.set(IVec2::new(4, 3), WALL);
//! map.push_layer(ground);
//!
//! let path = find_path(&map, IVec2::new(0, 3), IVec2::new(7, 3), Movement::Orthogonal, 4096)
//!     .expect("a route around the wall exists");
//! assert!(!path.contains(&IVec2::new(4, 3)));
//! ```

#![doc(html_no_source)]

pub mod autotile;
pub mod map;
pub mod pathfinding;
pub mod wfc;

pub use autotile::{blob_index, blob_index_at, neighbour_mask, wang_index, BLOB_TILE_COUNT};
pub use map::{TileId, TileLayer, TileProperties, Tilemap};
pub use pathfinding::{find_path, FlowField, Movement};
pub use wfc::{Direction, TileSet, Wfc, WfcError};
