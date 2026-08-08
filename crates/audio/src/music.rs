//! Procedural music.
//!
//! A game like this needs hours of background music that never grates. Written
//! themes loop and become obtrusive; generated ones drift into noise unless
//! they are constrained. The constraint here is harmonic: melodies are drawn
//! from a scale over a chord progression, so every note is consonant with what
//! is under it by construction, and the generator's freedom is limited to
//! *which* consonant note it picks.
//!
//! Everything is a function of a seed, so a piece can be regenerated exactly —
//! which is what lets a track be identified by a number rather than shipped as
//! a file.

use crate::envelope::Adsr;
use crate::synth::{layer, render, Instrument, Note, Sound};
use crate::wave::Waveform;
use verdant_core_math::Rng;

/// A scale, as semitone offsets from its root.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Scale {
    /// Bright and settled.
    Major,
    /// Darker; the natural minor.
    Minor,
    /// Minor with a raised sixth — wistful rather than sad.
    Dorian,
    /// Five notes, no semitone clashes at all. The safest choice for a
    /// generator, because no two notes of it can sound wrong together.
    Pentatonic,
    /// Major without the fourth and seventh: open and airy.
    MajorPentatonic,
}

impl Scale {
    /// Every scale, for iteration.
    pub const ALL: [Scale; 5] = [
        Scale::Major,
        Scale::Minor,
        Scale::Dorian,
        Scale::Pentatonic,
        Scale::MajorPentatonic,
    ];

    /// The semitone offsets of its degrees.
    #[must_use]
    pub const fn degrees(self) -> &'static [i32] {
        match self {
            Scale::Major => &[0, 2, 4, 5, 7, 9, 11],
            Scale::Minor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Scale::Pentatonic => &[0, 3, 5, 7, 10],
            Scale::MajorPentatonic => &[0, 2, 4, 7, 9],
        }
    }

    /// Its name, for display.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Scale::Major => "major",
            Scale::Minor => "minor",
            Scale::Dorian => "dorian",
            Scale::Pentatonic => "pentatonic",
            Scale::MajorPentatonic => "major pentatonic",
        }
    }

    /// The MIDI pitch of a scale degree above a root.
    ///
    /// Degrees past the end of the scale wrap into higher octaves, and
    /// negative degrees into lower ones, so a melody can range freely without
    /// the caller tracking octaves.
    #[must_use]
    pub fn pitch(self, root: u8, degree: i32) -> u8 {
        let steps = self.degrees();
        let count = i32::try_from(steps.len()).unwrap_or(7);
        let octave = degree.div_euclid(count);
        let index = usize::try_from(degree.rem_euclid(count)).unwrap_or(0);
        let semitones = i32::from(root) + steps[index] + octave * 12;
        u8::try_from(semitones.clamp(0, 127)).unwrap_or(60)
    }
}

/// A chord, as scale degrees above the key's root.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Chord {
    /// The chord's root, as a scale degree.
    pub root: i32,
}

impl Chord {
    /// The triad built on this chord's root, in scale degrees.
    ///
    /// Stacked thirds *within the scale*, which is what makes the chord fit
    /// the key without any need to know whether it is major or minor.
    #[must_use]
    pub const fn triad(self) -> [i32; 3] {
        [self.root, self.root + 2, self.root + 4]
    }
}

/// A sequence of chords, one per bar.
///
/// These four are the progressions almost all popular music is built from, and
/// each loops without a seam — which matters more than variety when a track
/// will play for an hour.
pub const PROGRESSIONS: [&[i32]; 4] = [
    &[0, 5, 3, 4], // I–vi–IV–V
    &[0, 3, 4, 0], // I–IV–V–I
    &[5, 3, 0, 4], // vi–IV–I–V
    &[0, 4, 5, 3], // I–V–vi–IV
];

/// The settings a piece is generated from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    /// The key's root, as a MIDI pitch.
    pub root: u8,
    /// Which scale the melody is drawn from.
    pub scale: Scale,
    /// Beats per minute.
    pub tempo: f32,
    /// How many bars long the piece is.
    pub bars: u32,
    /// Beats in a bar.
    pub beats_per_bar: u32,
}

impl Default for Theme {
    fn default() -> Theme {
        Theme {
            root: 60,
            scale: Scale::MajorPentatonic,
            tempo: 96.0,
            bars: 8,
            beats_per_bar: 4,
        }
    }
}

impl Theme {
    /// How long one beat lasts, in seconds.
    #[must_use]
    pub fn beat_seconds(&self) -> f32 {
        60.0 / self.tempo.max(1.0)
    }

    /// How long the whole piece lasts, in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "bar and beat counts are small integers"
        )]
        let beats = (self.bars * self.beats_per_bar) as f32;
        beats * self.beat_seconds()
    }
}

/// A generated piece, as its parts.
#[derive(Clone, Debug, Default)]
pub struct Score {
    /// The tune.
    pub melody: Vec<Note>,
    /// The chords under it.
    pub harmony: Vec<Note>,
    /// The bass line.
    pub bass: Vec<Note>,
}

impl Score {
    /// Every note in the score, for counting and inspection.
    pub fn notes(&self) -> impl Iterator<Item = &Note> {
        self.melody
            .iter()
            .chain(self.harmony.iter())
            .chain(self.bass.iter())
    }
}

/// Composes a piece from a seed.
///
/// The melody moves by step far more often than it leaps, which is the single
/// rule that most separates a tune from a random walk through a scale.
#[must_use]
pub fn compose(theme: &Theme, seed: u64) -> Score {
    let mut rng = Rng::new(seed).derive("compose");
    let beat = theme.beat_seconds();
    let progression = PROGRESSIONS[usize::try_from(
        rng.range(0, i32::try_from(PROGRESSIONS.len()).unwrap_or(4) - 1),
    )
    .unwrap_or(0)];

    let mut score = Score::default();
    let mut degree = 0i32;

    for bar in 0..theme.bars {
        let chord = Chord {
            root: progression[usize::try_from(bar).unwrap_or(0) % progression.len()],
        };
        #[allow(clippy::cast_precision_loss, reason = "bar counts are small integers")]
        let bar_start = (bar * theme.beats_per_bar) as f32 * beat;

        // --- Bass: the chord's root, one note per bar ---------------------
        score.bass.push(
            Note::new(
                theme.scale.pitch(theme.root, chord.root - 7),
                bar_start,
                beat * 2.0,
            )
            .at_velocity(0.8),
        );

        // --- Harmony: the triad, held across the bar ----------------------
        #[allow(
            clippy::cast_precision_loss,
            reason = "beats per bar is a small integer"
        )]
        let bar_length = theme.beats_per_bar as f32 * beat;
        for member in chord.triad() {
            score.harmony.push(
                Note::new(theme.scale.pitch(theme.root, member), bar_start, bar_length)
                    .at_velocity(0.35),
            );
        }

        // --- Melody -------------------------------------------------------
        let mut position = 0.0f32;
        #[allow(
            clippy::cast_precision_loss,
            reason = "beats per bar is a small integer"
        )]
        let bar_beats = theme.beats_per_bar as f32;
        while position < bar_beats {
            // Mostly quavers and crotchets; the occasional minim gives the
            // line somewhere to breathe.
            let length: f32 = match rng.range(0, 9) {
                0..=3 => 0.5,
                4..=7 => 1.0,
                _ => 2.0,
            };
            let length = length.min(bar_beats - position);

            // A rest now and then. A melody with no gaps is exhausting.
            if rng.chance(1, 6) {
                position += length;
                continue;
            }

            // Step by one degree most of the time; leap occasionally, and
            // land on a chord tone at the start of a bar so the tune stays
            // anchored to the harmony.
            degree = if position == 0.0 {
                chord.triad()[usize::try_from(rng.range(0, 2)).unwrap_or(0)] + 7
            } else if rng.chance(3, 4) {
                degree + if rng.chance(1, 2) { 1 } else { -1 }
            } else {
                degree + rng.range(-3, 3)
            };
            // Keep the tune inside a comfortable range rather than letting
            // the random walk wander off the top of the scale. The bound is
            // in scale degrees, and a five-note scale covers far more ground
            // per degree than a seven-note one, so it is set for the widest
            // case: thirteen degrees of a pentatonic is under two octaves.
            degree = degree.clamp(4, 13);

            score.melody.push(
                Note::new(
                    theme.scale.pitch(theme.root, degree),
                    bar_start + position * beat,
                    length * beat * 0.9,
                )
                .at_velocity(0.7),
            );
            position += length;
        }
    }

    score
}

/// The instruments a generated piece is rendered with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ensemble {
    /// Plays the melody.
    pub lead: Instrument,
    /// Plays the chords.
    pub pad: Instrument,
    /// Plays the bass line.
    pub bass: Instrument,
}

impl Default for Ensemble {
    /// A warm, soft ensemble: a plucked lead over a sine pad and a round bass.
    fn default() -> Ensemble {
        Ensemble {
            lead: Instrument::new(Waveform::Triangle, Adsr::new(0.01, 0.18, 0.35, 0.22))
                .with_gain(0.34)
                .with_harmonics(2, 5.0),
            pad: Instrument::new(Waveform::Sine, Adsr::pad(0.35, 0.6))
                .with_gain(0.12)
                .with_harmonics(3, 9.0),
            bass: Instrument::new(Waveform::Sine, Adsr::new(0.02, 0.25, 0.5, 0.2))
                .with_gain(0.3)
                .with_harmonics(2, 0.0),
        }
    }
}

/// Renders a score to audio.
///
/// The result is normalised and faded, so it can be looped without a click at
/// the seam.
#[must_use]
pub fn perform(score: &Score, ensemble: &Ensemble, seed: u64) -> Sound {
    let mut sound = layer(&[
        render(&ensemble.lead, &score.melody, seed ^ 0x1),
        render(&ensemble.pad, &score.harmony, seed ^ 0x2),
        render(&ensemble.bass, &score.bass, seed ^ 0x3),
    ]);
    // Headroom rather than full scale: this is background music, and it has
    // to sit under the effects bus without fighting it.
    sound.normalise(0.7);
    sound.fade_out(0.05);
    sound
}

/// Composes and renders in one step.
#[must_use]
pub fn generate(theme: &Theme, seed: u64) -> Sound {
    perform(&compose(theme, seed), &Ensemble::default(), seed)
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
    use crate::synth::note_frequency;

    /// A short theme, so the tests stay fast.
    fn theme() -> Theme {
        Theme {
            bars: 4,
            ..Theme::default()
        }
    }

    #[test]
    fn every_scale_starts_on_its_root() {
        for scale in Scale::ALL {
            assert_eq!(scale.degrees()[0], 0, "{} should start at 0", scale.name());
            assert_eq!(scale.pitch(60, 0), 60);
        }
    }

    #[test]
    fn every_scale_stays_inside_an_octave() {
        for scale in Scale::ALL {
            for degree in scale.degrees() {
                assert!(
                    (0..12).contains(degree),
                    "{} has a degree outside its octave: {degree}",
                    scale.name()
                );
            }
            // And its degrees rise, or `pitch` would not be monotonic.
            let mut previous = -1;
            for degree in scale.degrees() {
                assert!(*degree > previous);
                previous = *degree;
            }
        }
    }

    #[test]
    fn degrees_past_the_scale_wrap_into_the_next_octave() {
        let scale = Scale::Major;
        let count = i32::try_from(scale.degrees().len()).expect("small");
        assert_eq!(scale.pitch(60, count), 72);
        assert_eq!(scale.pitch(60, -count), 48);
    }

    #[test]
    fn pitches_rise_with_their_degree() {
        for scale in Scale::ALL {
            let mut previous = 0;
            for degree in 0..20 {
                let pitch = scale.pitch(60, degree);
                assert!(
                    pitch > previous,
                    "{} degree {degree} did not rise",
                    scale.name()
                );
                previous = pitch;
            }
        }
    }

    #[test]
    fn extreme_degrees_clamp_rather_than_overflowing() {
        // A melody that wandered far enough would otherwise wrap around into
        // a shriek or a rumble.
        assert_eq!(Scale::Major.pitch(60, 1000), 127);
        assert_eq!(Scale::Major.pitch(60, -1000), 0);
    }

    #[test]
    fn a_triad_stacks_thirds_within_the_scale() {
        let chord = Chord { root: 0 };
        assert_eq!(chord.triad(), [0, 2, 4]);
        // In C major that is C, E, G.
        let pitches = chord.triad().map(|d| Scale::Major.pitch(60, d));
        assert_eq!(pitches, [60, 64, 67]);
    }

    #[test]
    fn every_progression_is_four_bars() {
        for progression in PROGRESSIONS {
            assert_eq!(progression.len(), 4);
            for degree in progression {
                assert!((0..7).contains(degree), "degree {degree} is off the scale");
            }
        }
    }

    #[test]
    fn composing_fills_the_whole_piece() {
        let theme = theme();
        let score = compose(&theme, 1);
        assert!(!score.melody.is_empty());
        assert_eq!(score.bass.len(), theme.bars as usize);
        assert_eq!(score.harmony.len(), theme.bars as usize * 3);

        // And nothing starts after the piece has ended.
        for note in score.notes() {
            assert!(
                note.start < theme.duration(),
                "a note starts at {} in a {}s piece",
                note.start,
                theme.duration()
            );
        }
    }

    #[test]
    fn composing_is_deterministic() {
        let theme = theme();
        assert_eq!(compose(&theme, 42).melody, compose(&theme, 42).melody);
    }

    #[test]
    fn different_seeds_write_different_tunes() {
        let theme = theme();
        assert_ne!(compose(&theme, 1).melody, compose(&theme, 2).melody);
    }

    #[test]
    fn the_melody_moves_by_step_more_often_than_it_leaps() {
        // The one rule that separates a tune from a random walk.
        let score = compose(
            &Theme {
                bars: 32,
                ..theme()
            },
            7,
        );
        let mut steps = 0;
        let mut leaps = 0;
        for pair in score.melody.windows(2) {
            let interval = i32::from(pair[1].pitch).abs_diff(i32::from(pair[0].pitch));
            if interval <= 4 {
                steps += 1;
            } else {
                leaps += 1;
            }
        }
        assert!(
            steps > leaps * 2,
            "{steps} steps against {leaps} leaps is not a melody"
        );
    }

    #[test]
    fn the_melody_stays_in_a_singable_range() {
        let score = compose(
            &Theme {
                bars: 32,
                ..theme()
            },
            11,
        );
        let low = score.melody.iter().map(|n| n.pitch).min().unwrap_or(0);
        let high = score.melody.iter().map(|n| n.pitch).max().unwrap_or(0);
        assert!(high - low <= 24, "the tune spans {} semitones", high - low);
        assert!(note_frequency(low) > 100.0, "and does not rumble");
    }

    #[test]
    fn notes_never_overrun_their_bar_in_the_melody() {
        let theme = theme();
        let beat = theme.beat_seconds();
        let bar = f32::from(u16::try_from(theme.beats_per_bar).expect("small")) * beat;
        let score = compose(&theme, 3);
        for note in &score.melody {
            let bar_index = (note.start / bar).floor();
            let into_bar = note.start - bar_index * bar;
            assert!(
                into_bar + note.duration <= bar + 1e-3,
                "a note runs {} past its bar",
                into_bar + note.duration - bar
            );
        }
    }

    #[test]
    fn every_melody_note_belongs_to_the_key() {
        // Generated music is only reliably pleasant because this holds.
        let theme = Theme {
            scale: Scale::Minor,
            bars: 16,
            ..theme()
        };
        let score = compose(&theme, 5);
        let allowed: std::collections::BTreeSet<i32> = theme
            .scale
            .degrees()
            .iter()
            .map(|degree| (i32::from(theme.root) + degree).rem_euclid(12))
            .collect();
        for note in score.notes() {
            assert!(
                allowed.contains(&(i32::from(note.pitch).rem_euclid(12))),
                "pitch {} is outside the key",
                note.pitch
            );
        }
    }

    #[test]
    fn performing_produces_audible_sound() {
        let sound = generate(&Theme { bars: 2, ..theme() }, 1);
        assert!(!sound.is_empty());
        assert!(sound.peak() > 0.5, "peak was {}", sound.peak());
    }

    #[test]
    fn a_performance_leaves_headroom_for_the_effects_bus() {
        let sound = generate(&Theme { bars: 2, ..theme() }, 1);
        assert!(sound.peak() <= 0.71, "peak was {}", sound.peak());
    }

    #[test]
    fn a_performance_ends_at_silence_so_it_can_loop() {
        let sound = generate(&Theme { bars: 2, ..theme() }, 1);
        let last = sound.samples()[sound.len() - 1];
        assert!(last.abs() < 1e-4, "ends at {last}, which would click");
    }

    #[test]
    fn generation_is_reproducible() {
        let theme = Theme { bars: 2, ..theme() };
        assert_eq!(generate(&theme, 8).samples(), generate(&theme, 8).samples());
    }

    #[test]
    fn a_performance_stays_finite() {
        let sound = generate(&Theme { bars: 4, ..theme() }, 13);
        assert!(sound.samples().iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn the_duration_matches_the_tempo() {
        let theme = Theme {
            bars: 4,
            beats_per_bar: 4,
            tempo: 120.0,
            ..Theme::default()
        };
        // Sixteen beats at two per second.
        assert!((theme.duration() - 8.0).abs() < 1e-4);
    }

    #[test]
    fn a_zero_tempo_does_not_divide_by_zero() {
        let theme = Theme {
            tempo: 0.0,
            ..Theme::default()
        };
        assert!(theme.beat_seconds().is_finite());
    }

    #[test]
    fn a_piece_with_no_bars_is_empty_rather_than_a_panic() {
        let score = compose(&Theme { bars: 0, ..theme() }, 1);
        assert!(score.notes().count() == 0);
        assert!(generate(&Theme { bars: 0, ..theme() }, 1).is_empty());
    }

    #[test]
    fn scale_names_are_distinct() {
        let names: std::collections::BTreeSet<&str> = Scale::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(names.len(), Scale::ALL.len());
    }
}
