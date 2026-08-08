//! # Verdant core math
//!
//! The deterministic numeric foundation of the Verdant engine.
//!
//! ## The determinism contract
//!
//! Everything in this crate is a pure function of its inputs and produces
//! bit-identical results on every platform. That property is what makes replay
//! files, rollback netcode and desync detection possible, and it is why the
//! simulation layer uses [`Fx`] fixed point rather than `f32`/`f64`.
//!
//! Concretely, code inside the determinism boundary — [`Fx`], [`Vec2`],
//! [`Rect`], [`Rng`], [`ValueNoise`] and everything built on them — must obey
//! three rules:
//!
//! 1. **No floating point.** Not in arithmetic, not in comparisons, not as an
//!    intermediate. [`Fx::from_f64`] and [`Fx::to_f64`] exist for asset import
//!    and for handing values to the GPU; calling them inside a simulation
//!    system reintroduces exactly the nondeterminism this crate removes.
//! 2. **No platform math library.** `sin`, `sqrt` and `atan2` are implemented
//!    here from integer operations, because vendor libm implementations
//!    disagree in the last ulp.
//! 3. **No unseeded randomness.** Every random draw comes from an [`Rng`] whose
//!    seed is part of saved and replayed state.
//!
//! Layers deliberately *outside* the boundary — rendering, audio output, the
//! editor — may use floats freely. They must never feed a value back into
//! simulation state except through recorded, replayable input.
//!
//! ## Module tour
//!
//! | Module | Contents |
//! |---|---|
//! | [`fixed`] | [`Fx`], the Q32.32 fixed-point scalar |
//! | [`vec2`] | [`Vec2`] world vectors and [`IVec2`] grid coordinates |
//! | [`geometry`] | [`Rect`], [`Circle`], ray casts and swept-AABB collision |
//! | [`rng`] | [`Rng`], a seekable PCG generator with derivable streams |
//! | [`noise`] | [`ValueNoise`] and fractal sums for terrain generation |
//! | [`easing`] | [`Easing`] curves for tweens and camera moves |
//!
//! ## Example
//!
//! ```
//! use verdant_core_math::{fx, Fx, Rng, Vec2};
//!
//! // A seeded generator: this sequence is identical on every machine.
//! let mut rng = Rng::new(20_260_808);
//! let direction = Vec2::from_angle(rng.range_fx(Fx::ZERO, Fx::TAU), Fx::ONE);
//! assert!((direction.length().to_f64() - 1.0).abs() < 1e-6);
//!
//! // Fixed-point arithmetic is exact for representable values.
//! assert_eq!(fx(3) * Fx::from_ratio(1, 2), Fx::from_ratio(3, 2));
//! ```

#![doc(html_no_source)]

pub mod easing;
pub mod fixed;
pub mod geometry;
pub mod noise;
pub mod rng;
pub mod vec2;

pub use easing::Easing;
pub use fixed::{fx, Fx, FRACTIONAL_BITS};
pub use geometry::{ray_vs_rect, sweep_rect_vs_rect, Circle, RayHit, Rect};
pub use noise::{fbm, ridged, FbmSettings, ValueNoise};
pub use rng::Rng;
pub use vec2::{IVec2, Vec2};

/// A deterministic 64-bit hash, used to fingerprint simulation state.
///
/// Replay and rollback tests hash the whole world after each tick and compare
/// the sequence against a reference run; a mismatch pinpoints the exact tick a
/// desync was introduced. FNV-1a is chosen because it is order-sensitive,
/// trivially incremental, and identical on every platform — cryptographic
/// strength is irrelevant here, reproducibility is everything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHasher {
    hash: u64,
}

impl StateHasher {
    /// The FNV-1a 64-bit offset basis.
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    /// The FNV-1a 64-bit prime.
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// Starts a new hash.
    #[inline]
    #[must_use]
    pub const fn new() -> StateHasher {
        StateHasher {
            hash: StateHasher::OFFSET_BASIS,
        }
    }

    /// Folds a byte slice into the hash.
    #[inline]
    pub fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(StateHasher::PRIME);
        }
    }

    /// Folds a `u64` into the hash, little-endian.
    #[inline]
    pub fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// Folds an `i64` into the hash.
    #[inline]
    pub fn write_i64(&mut self, value: i64) {
        self.write(&value.to_le_bytes());
    }

    /// Folds a `u32` into the hash.
    #[inline]
    pub fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    /// Folds a fixed-point value into the hash by its exact raw bits.
    #[inline]
    pub fn write_fx(&mut self, value: Fx) {
        self.write_i64(value.to_raw());
    }

    /// Folds a vector into the hash.
    #[inline]
    pub fn write_vec2(&mut self, value: Vec2) {
        self.write_fx(value.x);
        self.write_fx(value.y);
    }

    /// The hash accumulated so far.
    #[inline]
    #[must_use]
    pub const fn finish(&self) -> u64 {
        self.hash
    }
}

impl Default for StateHasher {
    fn default() -> StateHasher {
        StateHasher::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_order_sensitive() {
        let mut a = StateHasher::new();
        a.write_u32(1);
        a.write_u32(2);
        let mut b = StateHasher::new();
        b.write_u32(2);
        b.write_u32(1);
        assert_ne!(
            a.finish(),
            b.finish(),
            "a state hash must notice reordering"
        );
    }

    #[test]
    fn hashing_the_same_sequence_is_reproducible() {
        let build = || {
            let mut hasher = StateHasher::new();
            hasher.write_fx(Fx::from_ratio(1, 3));
            hasher.write_vec2(Vec2::from_ints(4, -7));
            hasher.write_u64(u64::MAX);
            hasher.finish()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn an_empty_hash_is_the_offset_basis() {
        assert_eq!(StateHasher::new().finish(), StateHasher::OFFSET_BASIS);
    }

    #[test]
    fn a_single_bit_change_changes_the_hash() {
        let mut a = StateHasher::new();
        a.write_fx(Fx::from_raw(1_000_000));
        let mut b = StateHasher::new();
        b.write_fx(Fx::from_raw(1_000_001));
        assert_ne!(a.finish(), b.finish());
    }
}
