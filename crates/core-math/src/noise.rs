//! Deterministic coherent noise for terrain, weather and texture generation.
//!
//! Two generators are provided:
//!
//! * [`ValueNoise`] — smooth, cheap, hash-based. Used for elevation, moisture
//!   and the cloud-cover field.
//! * [`fbm`] — fractal Brownian motion, layering octaves of value noise to add
//!   detail without changing the large-scale shape.
//!
//! Both are integer-hash driven and evaluated in [`Fx`], so a world seed
//! reproduces the same terrain on every machine — which is the whole point of
//! sharing a seed with another player.

use crate::fixed::Fx;
use crate::vec2::Vec2;

/// A 2D value-noise field.
///
/// Value noise (interpolating hashed lattice *values*) is used rather than
/// Perlin/simplex (interpolating hashed *gradients*) because it needs no
/// gradient table, is trivially deterministic in fixed point, and its slightly
/// blockier character is invisible once octaves are layered and the result is
/// quantised to tiles.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValueNoise {
    seed: u32,
}

impl ValueNoise {
    /// Creates a noise field for the given seed.
    #[inline]
    #[must_use]
    pub const fn new(seed: u32) -> ValueNoise {
        ValueNoise { seed }
    }

    /// Hashes a lattice point to a value in `[0, 1)`.
    ///
    /// This is the only place randomness enters: everything else is
    /// interpolation, so the whole field is a pure function of `(seed, x, y)`.
    #[inline]
    #[must_use]
    fn lattice_value(self, x: i32, y: i32) -> Fx {
        // Integer avalanche mixer in the style of MurmurHash3's finaliser.
        let mut hash = (x as u32).wrapping_mul(0x1b87_3593);
        hash ^= (y as u32).wrapping_mul(0xcc9e_2d51);
        hash ^= self.seed.wrapping_mul(0x85eb_ca6b);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x2545_f491);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2_ae35);
        hash ^= hash >> 16;
        // The low 32 bits become the fractional part of a Q32.32 value.
        Fx::from_raw(i64::from(hash))
    }

    /// Samples the field at `(x, y)`, returning a value in `[0, 1]`.
    ///
    /// Lattice values are blended with a quintic smoothstep, whose first *and*
    /// second derivatives vanish at the lattice points — a cubic smoothstep
    /// would leave visible creases along the cell boundaries once the field is
    /// used to drive lighting or normals.
    #[must_use]
    pub fn sample(self, x: Fx, y: Fx) -> Fx {
        let cell_x = x.floor_int();
        let cell_y = y.floor_int();
        let fx_frac = x.fract();
        let fy_frac = y.fract();

        let u = quintic(fx_frac);
        let v = quintic(fy_frac);

        let top_left = self.lattice_value(cell_x, cell_y);
        let top_right = self.lattice_value(cell_x + 1, cell_y);
        let bottom_left = self.lattice_value(cell_x, cell_y + 1);
        let bottom_right = self.lattice_value(cell_x + 1, cell_y + 1);

        let top = top_left.lerp(top_right, u);
        let bottom = bottom_left.lerp(bottom_right, u);
        top.lerp(bottom, v).clamp(Fx::ZERO, Fx::ONE)
    }

    /// Samples at a world position.
    #[inline]
    #[must_use]
    pub fn sample_at(self, position: Vec2) -> Fx {
        self.sample(position.x, position.y)
    }

    /// Samples and remaps to `[-1, 1]`, the form terrain warping wants.
    #[inline]
    #[must_use]
    pub fn sample_signed(self, x: Fx, y: Fx) -> Fx {
        self.sample(x, y) * Fx::TWO - Fx::ONE
    }
}

/// Parameters for a fractal Brownian motion sum.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FbmSettings {
    /// Number of noise layers summed. More octaves means finer detail and
    /// proportionally more cost.
    pub octaves: u32,
    /// Frequency multiplier between successive octaves. `2` is the usual
    /// choice; values slightly off 2 avoid detail aligning to the lattice.
    pub lacunarity: Fx,
    /// Amplitude multiplier between successive octaves. Below `0.5` the higher
    /// octaves become imperceptible; above it the field turns noisy.
    pub gain: Fx,
    /// Frequency of the first octave, in cycles per world unit.
    pub frequency: Fx,
}

impl Default for FbmSettings {
    /// Four octaves of classic 2× lacunarity and ½ gain — the setting that
    /// produces recognisable rolling terrain.
    fn default() -> FbmSettings {
        FbmSettings {
            octaves: 4,
            lacunarity: Fx::TWO,
            gain: Fx::HALF,
            frequency: Fx::from_ratio(1, 32),
        }
    }
}

/// Sums octaves of [`ValueNoise`] into a fractal field, normalised to `[0, 1]`.
///
/// Normalisation divides by the total amplitude actually used, so changing the
/// octave count shifts the amount of detail without shifting the overall
/// brightness of the field — otherwise every terrain threshold would need
/// re-tuning whenever an octave was added.
#[must_use]
pub fn fbm(noise: ValueNoise, x: Fx, y: Fx, settings: FbmSettings) -> Fx {
    let mut total = Fx::ZERO;
    let mut amplitude = Fx::ONE;
    let mut total_amplitude = Fx::ZERO;
    let mut frequency = settings.frequency;

    for octave in 0..settings.octaves {
        // Offsetting each octave keeps their lattices from lining up, which
        // would otherwise leave a grid artefact at the origin.
        let offset = Fx::from_num((octave as i32) * 137);
        total += noise.sample(x * frequency + offset, y * frequency + offset) * amplitude;
        total_amplitude += amplitude;
        amplitude *= settings.gain;
        frequency *= settings.lacunarity;
    }

    if total_amplitude.is_zero() {
        Fx::HALF
    } else {
        (total / total_amplitude).clamp(Fx::ZERO, Fx::ONE)
    }
}

/// Ridged multifractal noise, for mountain ridges and cave walls.
///
/// Folding each octave around its midpoint (`1 - |2n - 1|`) turns the smooth
/// hills of [`fbm`] into sharp creases, which reads as rock rather than dunes.
#[must_use]
pub fn ridged(noise: ValueNoise, x: Fx, y: Fx, settings: FbmSettings) -> Fx {
    let mut total = Fx::ZERO;
    let mut amplitude = Fx::ONE;
    let mut total_amplitude = Fx::ZERO;
    let mut frequency = settings.frequency;

    for octave in 0..settings.octaves {
        let offset = Fx::from_num((octave as i32) * 191);
        let sample = noise.sample(x * frequency + offset, y * frequency + offset);
        let ridge = Fx::ONE - (sample * Fx::TWO - Fx::ONE).abs();
        total += ridge * amplitude;
        total_amplitude += amplitude;
        amplitude *= settings.gain;
        frequency *= settings.lacunarity;
    }

    if total_amplitude.is_zero() {
        Fx::ZERO
    } else {
        (total / total_amplitude).clamp(Fx::ZERO, Fx::ONE)
    }
}

/// The quintic smoothstep `6t^5 - 15t^4 + 10t^3`.
#[inline]
#[must_use]
fn quintic(t: Fx) -> Fx {
    let t3 = t * t * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    t5 * 6 - t4 * 15 + t3 * 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_stay_within_the_unit_interval() {
        let noise = ValueNoise::new(1234);
        for i in -200..200i32 {
            for j in -20..20i32 {
                let value = noise.sample(Fx::from_ratio(i, 7), Fx::from_ratio(j, 5));
                assert!(value >= Fx::ZERO && value <= Fx::ONE, "sample was {value}");
            }
        }
    }

    #[test]
    fn sampling_is_deterministic() {
        let a = ValueNoise::new(99);
        let b = ValueNoise::new(99);
        for i in 0..500i32 {
            let x = Fx::from_ratio(i, 13);
            let y = Fx::from_ratio(i * 3, 7);
            assert_eq!(a.sample(x, y), b.sample(x, y));
        }
    }

    #[test]
    fn different_seeds_produce_different_fields() {
        let a = ValueNoise::new(1);
        let b = ValueNoise::new(2);
        let differences = (0..100i32)
            .filter(|i| {
                let x = Fx::from_ratio(*i, 9);
                a.sample(x, Fx::ZERO) != b.sample(x, Fx::ZERO)
            })
            .count();
        assert!(differences > 90, "fields should differ nearly everywhere");
    }

    #[test]
    fn the_field_is_continuous_across_lattice_boundaries() {
        let noise = ValueNoise::new(7);
        // Step across an integer boundary in tiny increments; the field must
        // never jump, which is what a missing smoothstep would cause.
        let mut previous = noise.sample(Fx::from_ratio(99, 100), Fx::HALF);
        for step in 100..=110i32 {
            let value = noise.sample(Fx::from_ratio(step, 100), Fx::HALF);
            let jump = (value - previous).abs();
            assert!(jump < Fx::from_ratio(1, 10), "discontinuity of {jump} at step {step}");
            previous = value;
        }
    }

    #[test]
    fn integer_lattice_points_reproduce_their_hashed_values() {
        let noise = ValueNoise::new(3);
        // At an exact lattice point the interpolation weights are zero, so the
        // sample must equal that corner's hash exactly.
        for x in 0..10i32 {
            for y in 0..10i32 {
                let expected = noise.lattice_value(x, y);
                assert_eq!(noise.sample(Fx::from_num(x), Fx::from_num(y)), expected);
            }
        }
    }

    #[test]
    fn quintic_smoothstep_pins_its_endpoints() {
        assert_eq!(quintic(Fx::ZERO), Fx::ZERO);
        assert_eq!(quintic(Fx::ONE), Fx::ONE);
        // Symmetric about the midpoint.
        assert!((quintic(Fx::HALF).to_f64() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fbm_stays_normalised_regardless_of_octave_count() {
        let noise = ValueNoise::new(555);
        for octaves in 1..=8u32 {
            let settings = FbmSettings { octaves, ..FbmSettings::default() };
            for i in 0..100i32 {
                let value = fbm(noise, Fx::from_num(i), Fx::from_num(i * 2), settings);
                assert!(value >= Fx::ZERO && value <= Fx::ONE, "{octaves} octaves gave {value}");
            }
        }
    }

    #[test]
    fn fbm_mean_is_near_the_midpoint() {
        let noise = ValueNoise::new(2024);
        let settings = FbmSettings::default();
        let samples = 4000;
        let mut total = Fx::ZERO;
        for i in 0..samples {
            total += fbm(noise, Fx::from_ratio(i, 3), Fx::from_ratio(i * 7, 5), settings);
        }
        let mean = (total / Fx::from_num(samples)).to_f64();
        assert!((mean - 0.5).abs() < 0.1, "fbm mean drifted to {mean}");
    }

    #[test]
    fn fbm_with_zero_octaves_returns_the_midpoint() {
        let noise = ValueNoise::new(1);
        let settings = FbmSettings { octaves: 0, ..FbmSettings::default() };
        assert_eq!(fbm(noise, Fx::ZERO, Fx::ZERO, settings), Fx::HALF);
    }

    #[test]
    fn ridged_noise_stays_in_range() {
        let noise = ValueNoise::new(808);
        let settings = FbmSettings::default();
        for i in 0..500i32 {
            let value = ridged(noise, Fx::from_ratio(i, 11), Fx::from_ratio(i, 13), settings);
            assert!(value >= Fx::ZERO && value <= Fx::ONE);
        }
    }
}
