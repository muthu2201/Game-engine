//! Oscillators, and the table that keeps them identical on every machine.
//!
//! ## Why a table rather than `f32::sin`
//!
//! `sin` is provided by the platform's libm, and different libms disagree in
//! the last bit or two. That is inaudible, but it means two machines running
//! the same seed produce different bytes, which is exactly what the engine's
//! determinism guarantees rule out. The table here is built from
//! [`verdant_core_math::Fx::sin`] — a fixed-point polynomial with no platform
//! dependency at all — so every sample this crate produces is reproducible
//! anywhere.

use std::sync::OnceLock;
use verdant_core_math::{Fx, Rng};

/// Entries in the sine table.
///
/// 1024 points with linear interpolation between them keeps harmonic
/// distortion below about -70 dB, which is inaudible under a game mix and a
/// fraction of the memory a larger table would cost.
pub const TABLE_SIZE: usize = 1024;

/// The shared sine table, built on first use.
static SINE: OnceLock<[f32; TABLE_SIZE]> = OnceLock::new();

/// One cycle of a sine wave, sampled at [`TABLE_SIZE`] points.
#[must_use]
pub fn sine_table() -> &'static [f32; TABLE_SIZE] {
    SINE.get_or_init(|| {
        let mut table = [0.0f32; TABLE_SIZE];
        let step = Fx::TAU / Fx::from_num(i32::try_from(TABLE_SIZE).unwrap_or(1024));
        for (index, entry) in table.iter_mut().enumerate() {
            let angle = step * Fx::from_num(i32::try_from(index).unwrap_or(0));
            // Clamped: the fixed-point sine is a polynomial and overshoots
            // by a few millionths near its peak. Inaudible, but a sample
            // above full scale is a sample that clips once it is summed.
            *entry = angle.sin().to_f32().clamp(-1.0, 1.0);
        }
        table
    })
}

/// Reads the sine table at a fractional phase, interpolating between entries.
///
/// `phase` is in cycles; only its fractional part matters.
#[must_use]
pub fn sine(phase: f32) -> f32 {
    let table = sine_table();
    let wrapped = phase - phase.floor();
    #[allow(
        clippy::cast_precision_loss,
        reason = "TABLE_SIZE is 1024, far inside f32's exactly-representable range"
    )]
    let scaled = wrapped * TABLE_SIZE as f32;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled is in [0, TABLE_SIZE) because wrapped is in [0, 1)"
    )]
    let index = scaled as usize % TABLE_SIZE;
    #[allow(
        clippy::cast_precision_loss,
        reason = "index is below TABLE_SIZE and exactly representable"
    )]
    let fraction = scaled - index as f32;
    let a = table[index];
    let b = table[(index + 1) % TABLE_SIZE];
    a + (b - a) * fraction
}

/// The shape of an oscillator.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Waveform {
    /// A pure tone. Warm, and the basis of the additive instruments.
    #[default]
    Sine,
    /// Hollow and reedy; the classic chiptune lead.
    Square,
    /// Softer than a square, brighter than a sine.
    Triangle,
    /// Buzzy and rich in harmonics; good for basses and strings.
    Saw,
    /// A square with a quarter duty cycle, thinner than a square.
    Pulse,
    /// White noise, for percussion, wind and rain.
    Noise,
}

impl Waveform {
    /// Every waveform, for iteration in tests and editors.
    pub const ALL: [Waveform; 6] = [
        Waveform::Sine,
        Waveform::Square,
        Waveform::Triangle,
        Waveform::Saw,
        Waveform::Pulse,
        Waveform::Noise,
    ];

    /// Its name, for display.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Waveform::Sine => "sine",
            Waveform::Square => "square",
            Waveform::Triangle => "triangle",
            Waveform::Saw => "saw",
            Waveform::Pulse => "pulse",
            Waveform::Noise => "noise",
        }
    }

    /// Samples the waveform at `phase`, in cycles.
    ///
    /// `noise` is only consulted by [`Waveform::Noise`]; passing the same
    /// generator to a tuned waveform leaves it untouched, so a patch can be
    /// re-rendered without its random stream drifting.
    #[must_use]
    pub fn sample(self, phase: f32, noise: &mut Rng) -> f32 {
        let wrapped = phase - phase.floor();
        match self {
            Waveform::Sine => sine(wrapped),
            Waveform::Square => {
                if wrapped < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Pulse => {
                if wrapped < 0.25 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Triangle => {
                // Zero-crossing at the same phase as the sine, so the two can
                // be layered without one cancelling the other.
                if wrapped < 0.25 {
                    wrapped * 4.0
                } else if wrapped < 0.75 {
                    2.0 - wrapped * 4.0
                } else {
                    wrapped * 4.0 - 4.0
                }
            }
            Waveform::Saw => wrapped * 2.0 - 1.0,
            Waveform::Noise => {
                // The engine's PCG, so noise is as reproducible as everything
                // else and a percussion hit sounds the same on every machine.
                let raw = noise.next_u32();
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "the mantissa loss here is the point: it is noise"
                )]
                let unit = (raw >> 8) as f32 / f32::from(1u16 << 8) / 65_536.0;
                unit * 2.0 - 1.0
            }
        }
    }

    /// True when the waveform draws on the random stream.
    #[must_use]
    pub const fn is_random(self) -> bool {
        matches!(self, Waveform::Noise)
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "audio is float arithmetic end to end. These tests assert exact \
              values that the code returns as literals — an envelope that is \
              exactly zero, a gain that is exactly one — and a tolerance there \
              would stop the test checking what it claims to."
)]
mod tests {
    use super::*;

    /// A generator for waveforms that do not use one.
    fn quiet() -> Rng {
        Rng::new(1)
    }

    #[test]
    fn the_sine_table_is_a_full_cycle() {
        let table = sine_table();
        assert!(table[0].abs() < 1e-3, "starts at zero: {}", table[0]);
        assert!(
            (table[TABLE_SIZE / 4] - 1.0).abs() < 1e-3,
            "peaks at a quarter turn: {}",
            table[TABLE_SIZE / 4]
        );
        assert!(table[TABLE_SIZE / 2].abs() < 1e-3);
        assert!((table[TABLE_SIZE * 3 / 4] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn sine_interpolates_between_entries() {
        // Halfway between two table entries must land between them, not on
        // one of them, or the interpolation is not happening.
        let table = sine_table();
        let midpoint = sine(0.5 / TABLE_SIZE as f32);
        let (a, b) = (table[0], table[1]);
        assert!(midpoint > a.min(b) && midpoint < a.max(b));
    }

    #[test]
    fn sine_wraps_at_every_whole_cycle() {
        for phase in [0.0f32, 0.125, 0.4, 0.99] {
            let here = sine(phase);
            for cycles in [1.0f32, 7.0, -3.0] {
                assert!(
                    (sine(phase + cycles) - here).abs() < 1e-5,
                    "phase {phase} + {cycles} cycles should match"
                );
            }
        }
    }

    #[test]
    fn every_waveform_stays_inside_full_scale() {
        // A sample outside [-1, 1] clips audibly once it reaches the mixer.
        let mut rng = Rng::new(7);
        for waveform in Waveform::ALL {
            for step in 0..512 {
                let phase = step as f32 / 512.0;
                let sample = waveform.sample(phase, &mut rng);
                assert!(
                    (-1.0..=1.0).contains(&sample),
                    "{} produced {sample} at phase {phase}",
                    waveform.name()
                );
            }
        }
    }

    #[test]
    fn the_tuned_waveforms_are_symmetric_about_zero() {
        // A wave with a DC offset makes every mix drift toward a rail.
        for waveform in [
            Waveform::Sine,
            Waveform::Square,
            Waveform::Triangle,
            Waveform::Saw,
        ] {
            let mut rng = quiet();
            let sum: f32 = (0..1024)
                .map(|step| waveform.sample(step as f32 / 1024.0, &mut rng))
                .sum();
            assert!(
                sum.abs() < 2.0,
                "{} has a DC offset of {sum}",
                waveform.name()
            );
        }
    }

    #[test]
    fn a_square_is_two_flat_halves() {
        let mut rng = quiet();
        assert_eq!(Waveform::Square.sample(0.25, &mut rng), 1.0);
        assert_eq!(Waveform::Square.sample(0.75, &mut rng), -1.0);
    }

    #[test]
    fn a_pulse_is_thinner_than_a_square() {
        // A quarter duty cycle: high for the first quarter only.
        let mut rng = quiet();
        assert_eq!(Waveform::Pulse.sample(0.1, &mut rng), 1.0);
        assert_eq!(Waveform::Pulse.sample(0.4, &mut rng), -1.0);
    }

    #[test]
    fn a_triangle_peaks_a_quarter_of_the_way_through() {
        let mut rng = quiet();
        assert!((Waveform::Triangle.sample(0.25, &mut rng) - 1.0).abs() < 1e-6);
        assert!((Waveform::Triangle.sample(0.75, &mut rng) + 1.0).abs() < 1e-6);
        assert!(Waveform::Triangle.sample(0.0, &mut rng).abs() < 1e-6);
    }

    #[test]
    fn a_saw_rises_across_its_cycle() {
        let mut rng = quiet();
        let mut previous = f32::MIN;
        for step in 0..64 {
            let sample = Waveform::Saw.sample(step as f32 / 64.0, &mut rng);
            assert!(sample > previous, "the saw should rise monotonically");
            previous = sample;
        }
    }

    #[test]
    fn noise_is_reproducible_from_its_seed() {
        // The whole point: the same seed gives the same noise, so a thunder
        // clap in a replay is the thunder clap that was recorded.
        let render = || {
            let mut rng = Rng::new(4242);
            (0..256)
                .map(|_| Waveform::Noise.sample(0.0, &mut rng))
                .collect::<Vec<f32>>()
        };
        assert_eq!(render(), render());
    }

    #[test]
    fn noise_is_actually_noisy() {
        let mut rng = Rng::new(11);
        let samples: Vec<f32> = (0..512)
            .map(|_| Waveform::Noise.sample(0.0, &mut rng))
            .collect();
        let distinct = samples
            .iter()
            .map(|sample| sample.to_bits())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(distinct > 400, "only {distinct} distinct samples");
    }

    #[test]
    fn only_noise_consumes_the_random_stream() {
        // A tuned voice that drew from the generator would shift every noise
        // voice rendered after it.
        for waveform in Waveform::ALL {
            let mut rng = Rng::new(3);
            let before = rng.clone().next_u32();
            // The sample itself is irrelevant here; what is under test is
            // whether the call touched the random stream.
            let _ = waveform.sample(0.3, &mut rng);
            let after = rng.next_u32();
            assert_eq!(
                before == after,
                !waveform.is_random(),
                "{} handled the stream wrongly",
                waveform.name()
            );
        }
    }

    #[test]
    fn waveform_names_are_distinct() {
        let names: std::collections::BTreeSet<&str> =
            Waveform::ALL.iter().map(|w| w.name()).collect();
        assert_eq!(names.len(), Waveform::ALL.len());
    }
}
