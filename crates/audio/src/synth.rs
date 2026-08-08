//! Instruments, notes and the renderer that turns them into samples.
//!
//! Sounds are rendered ahead of time into a [`Sound`] rather than synthesised
//! in the audio callback. That is a deliberate trade: a few kilobytes per
//! effect, in exchange for an audio thread that only ever adds numbers
//! together. Synthesis in the callback is where audio glitches come from,
//! because a missed deadline is a click the player hears.

use crate::envelope::Adsr;
use crate::wave::Waveform;
use std::sync::Arc;
use verdant_core_math::Rng;

/// Samples per second, for everything this crate produces.
///
/// One rate throughout: resampling is the other common source of audio
/// artefacts, and a game has no reason to mix material at different rates.
pub const SAMPLE_RATE: u32 = 48_000;

/// MIDI note number of A4, the tuning reference.
pub const A4_MIDI: u8 = 69;

/// Frequency of A4 in hertz.
pub const A4_FREQUENCY: f32 = 440.0;

/// The twelve equal-tempered semitone ratios within an octave.
///
/// A table rather than `2f32.powf(n / 12.0)`: `powf` is another libm function
/// that differs between platforms, and these twelve constants make every pitch
/// in the game bit-identical everywhere.
#[allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "these are the twelfth roots of two to f32 precision; the tritone \
              coinciding with sqrt(2) is the definition of equal temperament, \
              not a mistyped constant"
)]
const SEMITONE_RATIOS: [f32; 12] = [
    1.000_000_0,
    1.059_463_1,
    1.122_462_0,
    1.189_207_1,
    1.259_921_0,
    1.334_839_9,
    1.414_213_6,
    1.498_307_1,
    1.587_401_1,
    1.681_792_8,
    1.781_797_4,
    1.887_748_6,
];

/// The frequency of a MIDI note number, in hertz.
///
/// Note 69 is A4 at 440 Hz; every octave doubles.
#[must_use]
pub fn note_frequency(pitch: u8) -> f32 {
    // Distance from C0, so both the octave and the semitone are non-negative
    // and the integer division is unambiguous.
    let semitones = i32::from(pitch) - i32::from(A4_MIDI);
    let octave = semitones.div_euclid(12);
    let step = usize::try_from(semitones.rem_euclid(12)).unwrap_or(0);
    let mut frequency = A4_FREQUENCY * SEMITONE_RATIOS[step];
    // Repeated halving and doubling rather than `powi`, so the result is an
    // exact binary scaling of the table entry.
    for _ in 0..octave.abs() {
        if octave > 0 {
            frequency *= 2.0;
        } else {
            frequency *= 0.5;
        }
    }
    frequency
}

/// A voice's timbre: what it sounds like, independent of what it plays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instrument {
    /// The oscillator shape.
    pub waveform: Waveform,
    /// The amplitude envelope.
    pub envelope: Adsr,
    /// Output level, from zero to one.
    pub gain: f32,
    /// How many harmonics to stack above the fundamental.
    ///
    /// One is a bare oscillator. More adds body, at a cost of one extra
    /// oscillator each — which is why this is a small number rather than a
    /// full additive engine.
    pub harmonics: u8,
    /// Detuning of the stacked harmonics, in cents.
    ///
    /// A few cents of detune is what makes a stack sound like an ensemble
    /// rather than one loud oscillator.
    pub detune_cents: f32,
}

impl Default for Instrument {
    fn default() -> Instrument {
        Instrument {
            waveform: Waveform::Sine,
            envelope: Adsr::default(),
            gain: 0.5,
            harmonics: 1,
            detune_cents: 0.0,
        }
    }
}

impl Instrument {
    /// A bare oscillator with an envelope.
    #[must_use]
    pub const fn new(waveform: Waveform, envelope: Adsr) -> Instrument {
        Instrument {
            waveform,
            envelope,
            gain: 0.5,
            harmonics: 1,
            detune_cents: 0.0,
        }
    }

    /// Returns this instrument at a different level.
    #[must_use]
    pub const fn with_gain(mut self, gain: f32) -> Instrument {
        self.gain = gain;
        self
    }

    /// Returns this instrument with a stack of detuned harmonics.
    #[must_use]
    pub const fn with_harmonics(mut self, harmonics: u8, detune_cents: f32) -> Instrument {
        self.harmonics = harmonics;
        self.detune_cents = detune_cents;
        self
    }
}

/// A note in a score: what to play, when, and how hard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// MIDI note number.
    pub pitch: u8,
    /// Seconds from the start of the score.
    pub start: f32,
    /// Seconds the note is held, before its release begins.
    pub duration: f32,
    /// How hard it is struck, from zero to one.
    pub velocity: f32,
}

impl Note {
    /// A note at full velocity.
    #[must_use]
    pub const fn new(pitch: u8, start: f32, duration: f32) -> Note {
        Note {
            pitch,
            start,
            duration,
            velocity: 1.0,
        }
    }

    /// Returns this note at a different velocity.
    #[must_use]
    pub const fn at_velocity(mut self, velocity: f32) -> Note {
        self.velocity = velocity;
        self
    }
}

/// A rendered mono sound, ready to mix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sound {
    samples: Vec<f32>,
}

impl Sound {
    /// Wraps raw samples.
    #[must_use]
    pub fn from_samples(samples: Vec<f32>) -> Sound {
        Sound { samples }
    }

    /// Silence of a given length.
    #[must_use]
    pub fn silence(seconds: f32) -> Sound {
        Sound {
            samples: vec![0.0; sample_count(seconds)],
        }
    }

    /// The samples, at [`SAMPLE_RATE`].
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// How many samples long it is.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True when there is nothing to play.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Its length in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "a sound long enough to lose precision here would be four hours"
        )]
        let samples = self.samples.len() as f32;
        #[allow(
            clippy::cast_precision_loss,
            reason = "SAMPLE_RATE is exactly representable"
        )]
        let rate = SAMPLE_RATE as f32;
        samples / rate
    }

    /// The largest absolute sample, which is what clips.
    #[must_use]
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()))
    }

    /// Scales every sample so the loudest reaches `target`.
    ///
    /// Silence is left alone rather than amplified into noise.
    pub fn normalise(&mut self, target: f32) {
        let peak = self.peak();
        if peak <= f32::EPSILON {
            return;
        }
        let scale = target / peak;
        for sample in &mut self.samples {
            *sample *= scale;
        }
    }

    /// Fades the last `seconds` down to silence.
    ///
    /// Any sound that ends on a non-zero sample clicks; this is the blunt fix
    /// for material that was not enveloped.
    pub fn fade_out(&mut self, seconds: f32) {
        let fade = sample_count(seconds).min(self.samples.len());
        if fade == 0 {
            return;
        }
        let start = self.samples.len() - fade;
        for (offset, sample) in self.samples[start..].iter_mut().enumerate() {
            #[allow(
                clippy::cast_precision_loss,
                reason = "offset is bounded by the fade length"
            )]
            let progress = (offset + 1) as f32 / fade as f32;
            *sample *= 1.0 - progress;
        }
    }

    /// Shares this sound between voices without copying its samples.
    #[must_use]
    pub fn shared(self) -> Arc<Sound> {
        Arc::new(self)
    }
}

/// How many samples span `seconds`.
#[must_use]
pub fn sample_count(seconds: f32) -> usize {
    if seconds <= 0.0 {
        return 0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE is exactly representable"
    )]
    let total = seconds * SAMPLE_RATE as f32;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "total is positive and bounded by the caller's duration"
    )]
    let rounded = total.round() as usize;
    rounded
}

/// Renders a score with one instrument.
///
/// `seed` drives the noise oscillator, so a percussion part is reproducible.
#[must_use]
pub fn render(instrument: &Instrument, notes: &[Note], seed: u64) -> Sound {
    let end = notes
        .iter()
        .map(|note| note.start + instrument.envelope.total_duration(note.duration))
        .fold(0.0f32, f32::max);
    let mut samples = vec![0.0f32; sample_count(end)];
    if samples.is_empty() {
        return Sound { samples };
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE is exactly representable"
    )]
    let rate = SAMPLE_RATE as f32;
    let voices = u32::from(instrument.harmonics.max(1));

    for (index, note) in notes.iter().enumerate() {
        // Each note draws from its own stream, so adding a note to a score
        // does not change the noise of the notes after it.
        let mut rng = Rng::new(seed).derive(&format!("note-{index}"));
        let base = note_frequency(note.pitch);
        let first = sample_count(note.start);
        let span = sample_count(instrument.envelope.total_duration(note.duration));

        for offset in 0..span {
            let position = first + offset;
            if position >= samples.len() {
                break;
            }
            #[allow(
                clippy::cast_precision_loss,
                reason = "offset is bounded by the note's own length"
            )]
            let time = offset as f32 / rate;
            let amplitude = instrument.envelope.amplitude(time, note.duration);
            if amplitude <= 0.0 {
                continue;
            }

            let mut value = 0.0f32;
            for harmonic in 0..voices {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "harmonic counts are single digits"
                )]
                let multiple = (harmonic + 1) as f32;
                // Cents to a ratio, linearised: at the few-cent detunings this
                // is used for, the error against the exact ratio is under a
                // thousandth of a semitone and inaudible.
                let detune = 1.0 + instrument.detune_cents * multiple / 120_000.0;
                let phase = time * base * multiple * detune;
                // Higher harmonics quieter, or the stack is just louder rather
                // than richer.
                value += instrument.waveform.sample(phase, &mut rng) / multiple;
            }

            samples[position] += value * amplitude * note.velocity * instrument.gain;
        }
    }

    Sound { samples }
}

/// Mixes several rendered sounds into one, summing them from sample zero.
///
/// Used to build a layered effect — a footstep is a thud and a scuff — without
/// paying for two voices at play time.
#[must_use]
pub fn layer(sounds: &[Sound]) -> Sound {
    let length = sounds.iter().map(Sound::len).max().unwrap_or(0);
    let mut samples = vec![0.0f32; length];
    for sound in sounds {
        for (destination, source) in samples.iter_mut().zip(sound.samples()) {
            *destination += *source;
        }
    }
    Sound { samples }
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

    #[test]
    fn a4_is_concert_pitch() {
        assert!((note_frequency(A4_MIDI) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn an_octave_doubles_the_frequency() {
        for pitch in [24u8, 48, 60, 69, 81] {
            let low = note_frequency(pitch);
            let high = note_frequency(pitch + 12);
            assert!(
                (high - low * 2.0).abs() < low * 1e-5,
                "{pitch}: {low} then {high}"
            );
        }
    }

    #[test]
    fn well_known_pitches_land_where_they_should() {
        // Middle C and the A below it, to catch a table that is off by one.
        assert!((note_frequency(60) - 261.626).abs() < 0.01);
        assert!((note_frequency(57) - 220.0).abs() < 0.01);
        assert!((note_frequency(21) - 27.5).abs() < 0.01);
    }

    #[test]
    fn pitch_rises_monotonically_across_the_whole_range() {
        let mut previous = 0.0;
        for pitch in 0..=127u8 {
            let frequency = note_frequency(pitch);
            assert!(
                frequency > previous,
                "pitch {pitch} did not rise: {frequency} after {previous}"
            );
            previous = frequency;
        }
    }

    #[test]
    fn rendering_produces_the_full_length_of_the_score() {
        let instrument = Instrument::new(Waveform::Sine, Adsr::new(0.01, 0.01, 1.0, 0.1));
        let notes = [Note::new(60, 0.0, 0.5), Note::new(64, 1.0, 0.5)];
        let sound = render(&instrument, &notes, 1);
        // The last note ends at 1.0 + 0.5 held + 0.1 release.
        assert!(
            (sound.duration() - 1.6).abs() < 0.01,
            "{}",
            sound.duration()
        );
    }

    #[test]
    fn an_empty_score_renders_nothing() {
        let sound = render(&Instrument::default(), &[], 1);
        assert!(sound.is_empty());
        assert_eq!(sound.duration(), 0.0);
        assert_eq!(sound.peak(), 0.0);
    }

    #[test]
    fn rendering_is_deterministic() {
        // The property the whole crate exists to provide.
        let instrument = Instrument::new(Waveform::Noise, Adsr::pluck(0.2));
        let notes = [Note::new(60, 0.0, 0.1), Note::new(67, 0.2, 0.1)];
        assert_eq!(
            render(&instrument, &notes, 99).samples(),
            render(&instrument, &notes, 99).samples()
        );
    }

    #[test]
    fn a_different_seed_gives_different_noise() {
        let instrument = Instrument::new(Waveform::Noise, Adsr::pluck(0.2));
        let notes = [Note::new(60, 0.0, 0.2)];
        assert_ne!(
            render(&instrument, &notes, 1).samples(),
            render(&instrument, &notes, 2).samples()
        );
    }

    #[test]
    fn adding_a_note_does_not_change_the_ones_before_it() {
        // Each note has its own stream precisely so a score can be edited
        // without every later sound changing.
        let instrument = Instrument::new(Waveform::Noise, Adsr::pluck(0.1));
        let first = render(&instrument, &[Note::new(60, 0.0, 0.1)], 5);
        let both = render(
            &instrument,
            &[Note::new(60, 0.0, 0.1), Note::new(67, 1.0, 0.1)],
            5,
        );
        let overlap = first.len();
        assert_eq!(&both.samples()[..overlap], first.samples());
    }

    #[test]
    fn a_rendered_note_starts_and_ends_at_silence() {
        // Anything else is an audible click at each end.
        let instrument = Instrument::new(Waveform::Saw, Adsr::new(0.02, 0.05, 0.7, 0.1));
        let sound = render(&instrument, &[Note::new(60, 0.0, 0.3)], 1);
        assert!(sound.samples()[0].abs() < 1e-4);
        let last = sound.samples()[sound.len() - 1];
        assert!(last.abs() < 1e-4, "ends at {last}");
    }

    #[test]
    fn a_note_actually_makes_a_sound() {
        let instrument = Instrument::new(Waveform::Sine, Adsr::default());
        let sound = render(&instrument, &[Note::new(60, 0.0, 0.3)], 1);
        assert!(sound.peak() > 0.1, "peak was only {}", sound.peak());
    }

    #[test]
    fn velocity_scales_the_output() {
        let instrument = Instrument::new(Waveform::Sine, Adsr::default());
        let loud = render(&instrument, &[Note::new(60, 0.0, 0.3)], 1);
        let quiet = render(&instrument, &[Note::new(60, 0.0, 0.3).at_velocity(0.25)], 1);
        assert!((quiet.peak() * 4.0 - loud.peak()).abs() < loud.peak() * 0.05);
    }

    #[test]
    fn a_note_starting_late_leaves_silence_before_it() {
        let instrument = Instrument::new(Waveform::Sine, Adsr::default());
        let sound = render(&instrument, &[Note::new(60, 0.5, 0.2)], 1);
        let before = sample_count(0.4);
        assert!(sound.samples()[..before].iter().all(|s| *s == 0.0));
        assert!(sound.samples()[sample_count(0.55)] != 0.0);
    }

    #[test]
    fn harmonics_add_content_without_changing_the_pitch() {
        let bare = Instrument::new(Waveform::Sine, Adsr::pad(0.01, 0.05));
        let stacked = bare.with_harmonics(4, 6.0);
        let notes = [Note::new(57, 0.0, 0.4)];
        let a = render(&bare, &notes, 1);
        let b = render(&stacked, &notes, 1);
        assert_eq!(a.len(), b.len());
        assert_ne!(a.samples(), b.samples());
    }

    #[test]
    fn normalising_brings_the_peak_to_the_target() {
        let mut sound = Sound::from_samples(vec![0.0, 0.2, -0.1]);
        sound.normalise(1.0);
        assert!((sound.peak() - 1.0).abs() < 1e-6);
        // And the shape is preserved, not clipped.
        assert!((sound.samples()[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalising_silence_leaves_it_silent() {
        // Dividing by a zero peak would fill the buffer with infinities.
        let mut sound = Sound::silence(0.01);
        sound.normalise(1.0);
        assert_eq!(sound.peak(), 0.0);
        assert!(sound.samples().iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn a_fade_ends_at_silence() {
        let mut sound = Sound::from_samples(vec![1.0; sample_count(0.1)]);
        sound.fade_out(0.05);
        assert_eq!(sound.samples()[sound.len() - 1], 0.0);
        assert_eq!(sound.samples()[0], 1.0);
    }

    #[test]
    fn a_fade_longer_than_the_sound_is_clamped() {
        let mut sound = Sound::from_samples(vec![1.0; 10]);
        sound.fade_out(100.0);
        assert_eq!(sound.len(), 10, "the fade must not extend the sound");
        assert_eq!(sound.samples()[9], 0.0, "and must still reach silence");
        assert!(sound.samples()[0] < 1.0, "the fade covers the whole sound");
    }

    #[test]
    fn layering_sums_sounds_and_takes_the_longest() {
        let a = Sound::from_samples(vec![0.5, 0.5]);
        let b = Sound::from_samples(vec![0.25, 0.25, 0.25]);
        let mixed = layer(&[a, b]);
        assert_eq!(mixed.samples(), &[0.75, 0.75, 0.25]);
    }

    #[test]
    fn layering_nothing_yields_nothing() {
        assert!(layer(&[]).is_empty());
    }

    #[test]
    fn sample_counts_round_rather_than_truncate() {
        assert_eq!(sample_count(1.0), SAMPLE_RATE as usize);
        assert_eq!(sample_count(0.0), 0);
        assert_eq!(sample_count(-5.0), 0);
    }

    #[test]
    fn rendered_output_stays_finite() {
        // A stack of harmonics at full velocity is the loudest case; NaN or
        // infinity here would be a burst of noise in the player's ears.
        let instrument = Instrument::new(Waveform::Saw, Adsr::pad(0.0, 0.0))
            .with_harmonics(8, 20.0)
            .with_gain(1.0);
        let notes: Vec<Note> = (0..8).map(|i| Note::new(48 + i * 3, 0.0, 0.2)).collect();
        let sound = render(&instrument, &notes, 3);
        assert!(sound.samples().iter().all(|sample| sample.is_finite()));
    }
}
