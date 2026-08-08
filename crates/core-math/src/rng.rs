//! Deterministic pseudo-random number generation.
//!
//! The engine never calls the operating system's entropy source inside the
//! simulation. Every random decision — crop yields, mine layouts, NPC gift
//! preferences, loot rolls — comes from an explicitly seeded [`Rng`] that is
//! part of the saved and replayed state. That is what lets a replay reproduce a
//! session exactly, and what lets a player share a world seed.

/// A PCG-XSH-RR 64/32 generator.
///
/// PCG is used rather than a linear congruential generator or xorshift because
/// it passes the full TestU01 BigCrush battery while needing only 64 bits of
/// state, and because its output function is a fixed sequence of integer
/// operations — identical on every platform, unlike anything float-based.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rng {
    state: u64,
    /// The stream selector. Must be odd; enforced at construction.
    increment: u64,
}

/// PCG's multiplier constant, from the reference implementation.
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// Default stream selector, also from the reference implementation.
const PCG_DEFAULT_INCREMENT: u64 = 1_442_695_040_888_963_407;

impl Rng {
    /// Creates a generator from a seed on the default stream.
    #[must_use]
    pub fn new(seed: u64) -> Rng {
        Rng::with_stream(seed, PCG_DEFAULT_INCREMENT)
    }

    /// Creates a generator on an independent stream.
    ///
    /// Separate streams are how the engine keeps subsystems from perturbing one
    /// another: the mine generator drawing an extra number must not shift the
    /// weather sequence. Give each subsystem its own stream from the world seed.
    #[must_use]
    pub fn with_stream(seed: u64, stream: u64) -> Rng {
        // The low bit of the increment must be set for the full 2^64 period.
        let increment = (stream << 1) | 1;
        let mut rng = Rng {
            state: 0,
            increment,
        };
        // Standard PCG seeding procedure: step, add the seed, step again.
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Derives a named child generator from this one without consuming any of
    /// its output.
    ///
    /// The child is a pure function of `(self.state, label)`, so the same world
    /// seed always produces the same "cave level 7" generator regardless of how
    /// much other generation happened first.
    #[must_use]
    pub fn derive(&self, label: &str) -> Rng {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for byte in label.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        }
        Rng::with_stream(self.state ^ hash, hash | 1)
    }

    /// Returns the next 32-bit output and advances the state.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let previous = self.state;
        self.state = previous
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
        // XSH-RR output permutation: xorshift the high bits down, then rotate
        // by an amount taken from the very highest bits.
        #[allow(clippy::cast_possible_truncation)]
        let xorshifted = (((previous >> 18) ^ previous) >> 27) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let rotation = (previous >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    /// Returns the next 64-bit output, drawn as two 32-bit outputs.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let high = u64::from(self.next_u32());
        let low = u64::from(self.next_u32());
        (high << 32) | low
    }

    /// Returns a uniformly distributed value in `[0, bound)`.
    ///
    /// Uses rejection sampling rather than a plain modulo so the distribution is
    /// exactly uniform; a modulo would bias the low values whenever `bound` does
    /// not divide 2^32.
    ///
    /// # Panics
    ///
    /// Panics if `bound` is zero.
    pub fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "Rng::below: bound must be positive");
        // Values at or above this threshold would land in the short final
        // bucket, so they are redrawn.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let candidate = self.next_u32();
            if candidate >= threshold {
                return candidate % bound;
            }
        }
    }

    /// Returns a uniformly distributed integer in the inclusive range
    /// `[min, max]`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    pub fn range(&mut self, min: i32, max: i32) -> i32 {
        assert!(min <= max, "Rng::range: min must not exceed max");
        let span = (i64::from(max) - i64::from(min) + 1) as u64;
        // The span of two i32 values always fits in u32 plus one.
        if span > u64::from(u32::MAX) {
            return min.wrapping_add(self.next_u32() as i32);
        }
        #[allow(clippy::cast_possible_truncation)]
        let offset = self.below(span as u32);
        min + (offset as i32)
    }

    /// Returns a value in `[0, 1)` as a [`Fx`](crate::fixed::Fx).
    ///
    /// Never returns exactly `1`, so it is safe to use as a `lerp` factor or to
    /// index a table of length `n` after multiplying by `n`.
    #[inline]
    pub fn unit(&mut self) -> crate::fixed::Fx {
        // 32 random bits become exactly the fractional part of a Q32.32 value.
        crate::fixed::Fx::from_raw(i64::from(self.next_u32()))
    }

    /// Returns a value in `[min, max)`.
    #[inline]
    pub fn range_fx(&mut self, min: crate::fixed::Fx, max: crate::fixed::Fx) -> crate::fixed::Fx {
        min + (max - min) * self.unit()
    }

    /// Returns `true` with probability `numerator / denominator`.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        assert!(denominator > 0, "Rng::chance: denominator must be positive");
        self.below(denominator) < numerator
    }

    /// Returns `true` with 50% probability.
    #[inline]
    pub fn coin_flip(&mut self) -> bool {
        self.next_u32() & 1 == 1
    }

    /// Picks a uniformly random element, or `None` when the slice is empty.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        // A slice length always fits in u32 on the platforms the engine targets;
        // saturating keeps this total for hypothetical huge slices.
        let index = self.below(u32::try_from(items.len()).unwrap_or(u32::MAX));
        items.get(index as usize)
    }

    /// Picks an index according to integer weights.
    ///
    /// Returns `None` when the slice is empty or every weight is zero. Used for
    /// loot tables and crop-quality rolls, where the weights come from data.
    pub fn pick_weighted(&mut self, weights: &[u32]) -> Option<usize> {
        let total: u32 = weights.iter().copied().sum();
        if total == 0 {
            return None;
        }
        let mut roll = self.below(total);
        for (index, weight) in weights.iter().enumerate() {
            if roll < *weight {
                return Some(index);
            }
            roll -= *weight;
        }
        // Unreachable given `roll < total`, but returning the last non-zero
        // entry is the safe answer rather than panicking in a shipped build.
        weights.iter().rposition(|weight| *weight > 0)
    }

    /// Shuffles a slice in place with an unbiased Fisher-Yates pass.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for index in (1..items.len()).rev() {
            let swap_with = self.below(u32::try_from(index + 1).unwrap_or(u32::MAX)) as usize;
            items.swap(index, swap_with);
        }
    }

    /// Returns a point uniformly distributed inside the unit circle.
    ///
    /// Rejection sampling in the bounding square is used rather than the polar
    /// method, because the polar method needs a square root of a random value
    /// and would clump samples toward the centre if done naively.
    pub fn unit_circle(&mut self) -> crate::vec2::Vec2 {
        use crate::fixed::Fx;
        loop {
            let x = self.range_fx(Fx::NEG_ONE, Fx::ONE);
            let y = self.range_fx(Fx::NEG_ONE, Fx::ONE);
            let point = crate::vec2::Vec2::new(x, y);
            if point.length_squared() <= Fx::ONE {
                return point;
            }
        }
    }

    /// The generator's current internal state, for snapshotting.
    #[inline]
    #[must_use]
    pub fn state(&self) -> (u64, u64) {
        (self.state, self.increment)
    }

    /// Restores a state captured by [`Rng::state`].
    ///
    /// Used by the rollback and save systems to rewind the random sequence
    /// alongside the rest of the world.
    #[inline]
    pub fn restore(&mut self, state: (u64, u64)) {
        self.state = state.0;
        // Preserve the odd-increment invariant even if the input is corrupt.
        self.increment = state.1 | 1;
    }
}

impl Default for Rng {
    /// A generator on a fixed seed.
    ///
    /// Deliberately *not* seeded from the clock: a default that varies per run
    /// would make an accidentally-defaulted generator produce bugs that only
    /// reproduce once.
    fn default() -> Rng {
        Rng::new(0x5eed_1234_5678_9abc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Fx;

    #[test]
    fn the_same_seed_produces_the_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let a_values: Vec<u32> = (0..32).map(|_| a.next_u32()).collect();
        let b_values: Vec<u32> = (0..32).map(|_| b.next_u32()).collect();
        assert_ne!(a_values, b_values);
    }

    #[test]
    fn streams_are_independent() {
        let mut a = Rng::with_stream(7, 1);
        let mut b = Rng::with_stream(7, 2);
        let a_values: Vec<u32> = (0..32).map(|_| a.next_u32()).collect();
        let b_values: Vec<u32> = (0..32).map(|_| b.next_u32()).collect();
        assert_ne!(a_values, b_values);
    }

    #[test]
    fn derive_is_stable_and_does_not_consume_the_parent() {
        let parent = Rng::new(99);
        let first = parent.derive("cave-level-7").next_u32();
        let second = parent.derive("cave-level-7").next_u32();
        assert_eq!(first, second, "deriving twice gives the same child");
        assert_ne!(first, parent.derive("cave-level-8").next_u32());
        // The parent is untouched: `derive` takes &self.
        let mut check = parent.clone();
        assert_eq!(check.next_u32(), Rng::new(99).next_u32());
    }

    #[test]
    fn below_stays_in_bounds() {
        let mut rng = Rng::new(5);
        for bound in [1u32, 2, 3, 7, 100, 1000] {
            for _ in 0..500 {
                assert!(rng.below(bound) < bound);
            }
        }
    }

    #[test]
    fn below_is_close_to_uniform() {
        let mut rng = Rng::new(11);
        let mut buckets = [0u32; 6];
        const ROLLS: u32 = 60_000;
        for _ in 0..ROLLS {
            buckets[rng.below(6) as usize] += 1;
        }
        // Each bucket should hold ~10000; allow a generous 5% band so the test
        // is not flaky while still catching a genuinely skewed generator.
        for count in buckets {
            assert!(
                (9_500..10_500).contains(&count),
                "bucket counts were {buckets:?}"
            );
        }
    }

    #[test]
    fn range_covers_both_endpoints() {
        let mut rng = Rng::new(13);
        let mut saw_min = false;
        let mut saw_max = false;
        for _ in 0..1000 {
            let value = rng.range(-3, 3);
            assert!((-3..=3).contains(&value));
            saw_min |= value == -3;
            saw_max |= value == 3;
        }
        assert!(saw_min && saw_max, "range must be inclusive at both ends");
    }

    #[test]
    fn range_with_equal_bounds_returns_that_value() {
        let mut rng = Rng::new(17);
        assert_eq!(rng.range(5, 5), 5);
    }

    #[test]
    fn unit_stays_in_the_half_open_zero_one_interval() {
        let mut rng = Rng::new(19);
        for _ in 0..5000 {
            let value = rng.unit();
            assert!(
                value >= Fx::ZERO && value < Fx::ONE,
                "unit() produced {value}"
            );
        }
    }

    #[test]
    fn weighted_picks_never_select_a_zero_weight() {
        let mut rng = Rng::new(23);
        let weights = [0u32, 5, 0, 3, 0];
        for _ in 0..2000 {
            let index = rng
                .pick_weighted(&weights)
                .expect("weights are non-zero overall");
            assert!(weights[index] > 0, "selected zero-weight index {index}");
        }
        assert_eq!(rng.pick_weighted(&[0, 0, 0]), None);
        assert_eq!(rng.pick_weighted(&[]), None);
    }

    #[test]
    fn weighted_picks_respect_the_ratio() {
        let mut rng = Rng::new(29);
        let weights = [1u32, 9];
        let mut counts = [0u32; 2];
        for _ in 0..20_000 {
            counts[rng.pick_weighted(&weights).unwrap()] += 1;
        }
        let ratio = f64::from(counts[1]) / f64::from(counts[0]);
        assert!(
            (ratio - 9.0).abs() < 1.0,
            "expected roughly 9:1, got {ratio}"
        );
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(31);
        let mut items: Vec<u32> = (0..64).collect();
        rng.shuffle(&mut items);
        assert_ne!(
            items,
            (0..64).collect::<Vec<u32>>(),
            "shuffling should reorder"
        );
        items.sort_unstable();
        assert_eq!(
            items,
            (0..64).collect::<Vec<u32>>(),
            "no element lost or duplicated"
        );
    }

    #[test]
    fn shuffle_handles_degenerate_lengths() {
        let mut rng = Rng::new(37);
        let mut empty: [u32; 0] = [];
        rng.shuffle(&mut empty);
        let mut single = [1u32];
        rng.shuffle(&mut single);
        assert_eq!(single, [1]);
    }

    #[test]
    fn pick_returns_none_only_for_empty_slices() {
        let mut rng = Rng::new(41);
        assert_eq!(rng.pick::<u32>(&[]), None);
        assert_eq!(rng.pick(&[9u32]), Some(&9));
    }

    #[test]
    fn unit_circle_samples_lie_inside_the_circle() {
        let mut rng = Rng::new(43);
        for _ in 0..2000 {
            assert!(rng.unit_circle().length_squared() <= Fx::ONE);
        }
    }

    #[test]
    fn state_round_trips_through_snapshot_and_restore() {
        let mut rng = Rng::new(47);
        for _ in 0..10 {
            rng.next_u32();
        }
        let snapshot = rng.state();
        let expected: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        rng.restore(snapshot);
        let replayed: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        assert_eq!(
            expected, replayed,
            "restoring must rewind the sequence exactly"
        );
    }

    #[test]
    fn chance_respects_its_probability() {
        let mut rng = Rng::new(53);
        let hits = (0..10_000).filter(|_| rng.chance(1, 4)).count();
        assert!(
            (2_300..2_700).contains(&hits),
            "expected ~2500 hits, got {hits}"
        );
        // Degenerate probabilities must be absolute, not approximate.
        assert!((0..100).all(|_| !rng.chance(0, 10)));
        assert!((0..100).all(|_| rng.chance(10, 10)));
    }
}
