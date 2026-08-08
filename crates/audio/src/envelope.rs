//! Amplitude envelopes.
//!
//! An oscillator alone sounds like a test tone: it starts and stops at full
//! volume, and the discontinuity at each end is an audible click. An envelope
//! is what turns a tone into an instrument — the difference between a plucked
//! string and a bowed one is almost entirely in how quickly the amplitude
//! rises and falls.

/// A four-stage attack/decay/sustain/release envelope.
///
/// Times are in seconds; [`Adsr::sustain`] is a level rather than a time,
/// because the sustain stage lasts as long as the note is held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adsr {
    /// Seconds from silence to full amplitude.
    pub attack: f32,
    /// Seconds from full amplitude down to the sustain level.
    pub decay: f32,
    /// The level held while the note is down, from zero to one.
    pub sustain: f32,
    /// Seconds from the sustain level back to silence after release.
    pub release: f32,
}

impl Default for Adsr {
    /// A short, percussive envelope: the shape most game sounds want.
    fn default() -> Adsr {
        Adsr {
            attack: 0.005,
            decay: 0.06,
            sustain: 0.6,
            release: 0.12,
        }
    }
}

impl Adsr {
    /// An envelope with the given stages, clamped to sane values.
    ///
    /// Negative times and out-of-range sustain levels are corrected rather
    /// than rejected: an envelope is data that a designer or a generator
    /// produces, and a slightly wrong number should make a slightly wrong
    /// sound, not stop the game.
    #[must_use]
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Adsr {
        Adsr {
            attack: attack.max(0.0),
            decay: decay.max(0.0),
            sustain: sustain.clamp(0.0, 1.0),
            release: release.max(0.0),
        }
    }

    /// A plucked shape: instant attack, quick decay, no sustain.
    #[must_use]
    pub const fn pluck(decay: f32) -> Adsr {
        Adsr {
            attack: 0.002,
            decay,
            sustain: 0.0,
            release: 0.02,
        }
    }

    /// A sustained shape, for pads and held notes.
    #[must_use]
    pub const fn pad(attack: f32, release: f32) -> Adsr {
        Adsr {
            attack,
            decay: 0.1,
            sustain: 0.85,
            release,
        }
    }

    /// The amplitude `time` seconds after the note began, given how long it is
    /// held.
    ///
    /// Past `hold + release` the result is exactly zero, which is what lets a
    /// voice be retired rather than mixed forever at an inaudible level.
    #[must_use]
    pub fn amplitude(&self, time: f32, hold: f32) -> f32 {
        if time < 0.0 {
            return 0.0;
        }
        if time >= hold {
            // Releasing, from wherever the envelope had reached.
            let level = self.level_at(hold);
            if self.release <= 0.0 {
                return 0.0;
            }
            let elapsed = time - hold;
            if elapsed >= self.release {
                return 0.0;
            }
            return level * (1.0 - elapsed / self.release);
        }
        self.level_at(time)
    }

    /// The level during the held part of the note, ignoring release.
    fn level_at(&self, time: f32) -> f32 {
        if time < self.attack {
            // A zero-length attack is instant rather than a division by zero.
            if self.attack <= 0.0 {
                return 1.0;
            }
            return time / self.attack;
        }
        let into_decay = time - self.attack;
        if into_decay < self.decay {
            if self.decay <= 0.0 {
                return self.sustain;
            }
            let fraction = into_decay / self.decay;
            return 1.0 + (self.sustain - 1.0) * fraction;
        }
        self.sustain
    }

    /// How long a note of this envelope lasts in total, held for `hold`
    /// seconds.
    ///
    /// This is the length a renderer must allocate: cutting a buffer at `hold`
    /// would chop the release off and leave a click.
    #[must_use]
    pub fn total_duration(&self, hold: f32) -> f32 {
        hold.max(0.0) + self.release
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

    /// An envelope with easy round numbers.
    fn envelope() -> Adsr {
        Adsr::new(0.1, 0.2, 0.5, 0.4)
    }

    #[test]
    fn an_envelope_starts_and_ends_silent() {
        let adsr = envelope();
        assert_eq!(adsr.amplitude(0.0, 1.0), 0.0);
        // A tolerance at the end rather than an exact zero: the release time
        // is not exactly representable, so the final sample lands a float
        // epsilon above silence. That is inaudible; a whole sample would not
        // be.
        assert!(adsr.amplitude(adsr.total_duration(1.0), 1.0) < 1e-6);
        // And stays silent well past the end, rather than wrapping around.
        assert_eq!(adsr.amplitude(100.0, 1.0), 0.0);
    }

    #[test]
    fn the_attack_reaches_full_amplitude() {
        let adsr = envelope();
        assert!((adsr.amplitude(0.1, 1.0) - 1.0).abs() < 1e-6);
        // And rises monotonically to get there.
        let mut previous = -1.0;
        for step in 0..=10 {
            let level = adsr.amplitude(step as f32 * 0.01, 1.0);
            assert!(level >= previous);
            previous = level;
        }
    }

    #[test]
    fn the_decay_falls_to_the_sustain_level() {
        let adsr = envelope();
        // Attack ends at 0.1, decay ends at 0.3.
        assert!((adsr.amplitude(0.3, 1.0) - 0.5).abs() < 1e-6);
        assert!((adsr.amplitude(0.6, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_release_falls_from_wherever_the_note_was() {
        let adsr = envelope();
        // Released at 1.0 from the sustain level, halfway through a 0.4s
        // release is half of 0.5.
        assert!((adsr.amplitude(1.2, 1.0) - 0.25).abs() < 1e-6);
        assert!(adsr.amplitude(1.4, 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_note_released_during_its_attack_releases_from_part_way_up() {
        // Otherwise a very short note would jump to full volume before
        // fading, which is an audible click.
        let adsr = envelope();
        let level = adsr.amplitude(0.05, 0.05);
        assert!(level < 0.6, "released mid-attack, got {level}");
        let after = adsr.amplitude(0.25, 0.05);
        assert!(after < level, "and should keep falling");
    }

    #[test]
    fn a_zero_length_stage_does_not_divide_by_zero() {
        let instant = Adsr::new(0.0, 0.0, 1.0, 0.0);
        assert!(instant.amplitude(0.0, 1.0).is_finite());
        assert_eq!(instant.amplitude(0.5, 1.0), 1.0);
        // With no release the note stops the moment it is let go.
        assert_eq!(instant.amplitude(1.0, 1.0), 0.0);
    }

    #[test]
    fn amplitudes_never_leave_the_unit_range() {
        let adsr = envelope();
        for step in 0..400 {
            let level = adsr.amplitude(step as f32 * 0.005, 1.0);
            assert!(
                (0.0..=1.0).contains(&level),
                "amplitude {level} at step {step}"
            );
        }
    }

    #[test]
    fn out_of_range_parameters_are_corrected() {
        let wild = Adsr::new(-1.0, -1.0, 5.0, -1.0);
        assert_eq!(wild.attack, 0.0);
        assert_eq!(wild.decay, 0.0);
        assert_eq!(wild.sustain, 1.0);
        assert_eq!(wild.release, 0.0);
    }

    #[test]
    fn negative_time_is_silent_rather_than_negative() {
        assert_eq!(envelope().amplitude(-0.5, 1.0), 0.0);
    }

    #[test]
    fn the_total_duration_includes_the_release() {
        let adsr = envelope();
        assert!((adsr.total_duration(1.0) - 1.4).abs() < 1e-6);
        // A buffer cut at `hold` would chop the release and click.
        assert!(adsr.amplitude(1.0 + adsr.release * 0.5, 1.0) > 0.0);
    }

    #[test]
    fn a_pluck_has_no_sustain() {
        let pluck = Adsr::pluck(0.3);
        assert_eq!(pluck.sustain, 0.0);
        assert!(
            pluck.amplitude(0.5, 2.0).abs() < 1e-6,
            "it should have died"
        );
    }

    #[test]
    fn a_pad_holds_its_level() {
        let pad = Adsr::pad(0.4, 0.8);
        assert!(pad.amplitude(2.0, 4.0) > 0.8);
    }
}
