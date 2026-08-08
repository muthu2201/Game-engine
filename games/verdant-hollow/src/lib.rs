//! # Verdant Hollow
//!
//! A farming and life simulation built on the Verdant engine.
//!
//! The crate is split so that the whole simulation runs without a window: the
//! modules here own the game's state and rules, and the binary adds windowing,
//! rendering and input on top. That is what lets a full in-game year be
//! simulated in a test, and what keeps non-deterministic subsystems out of the
//! simulation.
//!
//! | Module | Role |
//! |---|---|
//! | [`calendar`] | Time of day, seasons, weather and the day/night tint |
//! | [`items`] | The item, crop and recipe catalogues |
//! | [`inventory`] | Carried items, the hotbar, buying, selling and cooking |
//! | [`farm`] | Tilled soil, planted crops and the overnight growth pass |
//! | [`world`] | Valley and mine generation |
#![doc(html_no_source)]

pub mod calendar;
pub mod farm;
pub mod inventory;
pub mod items;
pub mod world;
