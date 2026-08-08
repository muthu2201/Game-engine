//! The mixer: what the audio thread actually runs.
//!
//! Everything expensive — synthesis, envelopes, note scheduling — happens
//! before a sound reaches here. [`Mixer::fill`] only reads samples, multiplies
//! them by three gains and adds them, so it finishes in bounded time and never
//! allocates. That is the whole design: an audio callback that can miss its
//! deadline is a callback that clicks.

use crate::synth::Sound;
use std::sync::Arc;

/// Which group a sound belongs to, for independent volume control.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Bus {
    /// Background music.
    Music,
    /// One-shot effects.
    Effects,
    /// Ambience: rain, wind, the river.
    Ambience,
}

impl Bus {
    /// Every bus, for iteration in a settings screen.
    pub const ALL: [Bus; 3] = [Bus::Music, Bus::Effects, Bus::Ambience];

    /// Its name, for display and configuration files.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Bus::Music => "music",
            Bus::Effects => "effects",
            Bus::Ambience => "ambience",
        }
    }

    /// Its index into the mixer's gain array.
    const fn index(self) -> usize {
        match self {
            Bus::Music => 0,
            Bus::Effects => 1,
            Bus::Ambience => 2,
        }
    }
}

/// A handle to a playing voice.
///
/// Carries a generation alongside the slot so that stopping a finished voice
/// cannot silence whatever took its place — the same recycling problem entity
/// handles solve, and the same solution.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VoiceId {
    slot: usize,
    generation: u32,
}

/// How a sound should be played.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaySettings {
    /// Which bus it belongs to.
    pub bus: Bus,
    /// Level, from zero to one.
    pub gain: f32,
    /// Stereo position, `-1` hard left to `1` hard right.
    pub pan: f32,
    /// Whether it repeats until stopped.
    pub looping: bool,
}

impl Default for PlaySettings {
    fn default() -> PlaySettings {
        PlaySettings {
            bus: Bus::Effects,
            gain: 1.0,
            pan: 0.0,
            looping: false,
        }
    }
}

impl PlaySettings {
    /// Settings for a one-shot on a bus.
    #[must_use]
    pub fn on(bus: Bus) -> PlaySettings {
        PlaySettings {
            bus,
            ..PlaySettings::default()
        }
    }

    /// Returns these settings at a different level.
    #[must_use]
    pub const fn with_gain(mut self, gain: f32) -> PlaySettings {
        self.gain = gain;
        self
    }

    /// Returns these settings panned.
    #[must_use]
    pub const fn panned(mut self, pan: f32) -> PlaySettings {
        self.pan = pan;
        self
    }

    /// Returns these settings looping.
    #[must_use]
    pub const fn looping(mut self) -> PlaySettings {
        self.looping = true;
        self
    }
}

/// One sound in flight.
#[derive(Clone, Debug)]
struct Voice {
    sound: Arc<Sound>,
    position: usize,
    settings: PlaySettings,
    generation: u32,
    active: bool,
}

/// Mixes playing voices into an interleaved stereo buffer.
#[derive(Debug)]
pub struct Mixer {
    voices: Vec<Voice>,
    bus_gains: [f32; 3],
    master: f32,
    /// Voices beyond this are refused rather than queued.
    voice_limit: usize,
    /// How many voices were refused, which is a symptom worth logging.
    dropped: u64,
}

/// How many voices a mixer allows by default.
///
/// A cap rather than an unbounded list: without one, a bug that plays a sound
/// every frame degrades into a rising wall of noise instead of a missing
/// sound, and the second is far easier to notice and diagnose.
pub const DEFAULT_VOICE_LIMIT: usize = 64;

impl Default for Mixer {
    fn default() -> Mixer {
        Mixer::new()
    }
}

impl Mixer {
    /// An empty mixer at full volume.
    #[must_use]
    pub fn new() -> Mixer {
        Mixer {
            voices: Vec::new(),
            bus_gains: [1.0; 3],
            master: 1.0,
            voice_limit: DEFAULT_VOICE_LIMIT,
            dropped: 0,
        }
    }

    /// Sets how many voices may play at once.
    pub fn set_voice_limit(&mut self, limit: usize) {
        self.voice_limit = limit;
    }

    /// The master volume.
    #[must_use]
    pub const fn master(&self) -> f32 {
        self.master
    }

    /// Sets the master volume, clamped to `[0, 1]`.
    pub fn set_master(&mut self, gain: f32) {
        self.master = gain.clamp(0.0, 1.0);
    }

    /// A bus's volume.
    #[must_use]
    pub fn bus_gain(&self, bus: Bus) -> f32 {
        self.bus_gains[bus.index()]
    }

    /// Sets a bus's volume, clamped to `[0, 1]`.
    pub fn set_bus_gain(&mut self, bus: Bus, gain: f32) {
        self.bus_gains[bus.index()] = gain.clamp(0.0, 1.0);
    }

    /// Starts a sound, returning a handle to it.
    ///
    /// Returns `None` when the sound is empty or the voice limit is reached.
    /// An empty sound is refused rather than played silently so that a
    /// mistakenly empty asset shows up as a missing handle.
    pub fn play(&mut self, sound: Arc<Sound>, settings: PlaySettings) -> Option<VoiceId> {
        if sound.is_empty() {
            return None;
        }

        // Reuse a finished slot before growing, so a long session does not
        // accumulate dead voices.
        if let Some(slot) = self.voices.iter().position(|voice| !voice.active) {
            let voice = &mut self.voices[slot];
            voice.generation = voice.generation.wrapping_add(1);
            voice.sound = sound;
            voice.position = 0;
            voice.settings = settings;
            voice.active = true;
            return Some(VoiceId {
                slot,
                generation: voice.generation,
            });
        }

        if self.voices.len() >= self.voice_limit {
            self.dropped += 1;
            return None;
        }

        self.voices.push(Voice {
            sound,
            position: 0,
            settings,
            generation: 0,
            active: true,
        });
        Some(VoiceId {
            slot: self.voices.len() - 1,
            generation: 0,
        })
    }

    /// Stops a voice.
    ///
    /// A stale handle is ignored, which is what makes it safe to hold one for
    /// as long as you like.
    pub fn stop(&mut self, id: VoiceId) {
        if let Some(voice) = self.voices.get_mut(id.slot) {
            if voice.generation == id.generation {
                voice.active = false;
            }
        }
    }

    /// Stops everything on a bus.
    pub fn stop_bus(&mut self, bus: Bus) {
        for voice in &mut self.voices {
            if voice.settings.bus == bus {
                voice.active = false;
            }
        }
    }

    /// Stops everything.
    pub fn stop_all(&mut self) {
        for voice in &mut self.voices {
            voice.active = false;
        }
    }

    /// True when the handle still refers to a playing voice.
    #[must_use]
    pub fn is_playing(&self, id: VoiceId) -> bool {
        self.voices
            .get(id.slot)
            .is_some_and(|voice| voice.active && voice.generation == id.generation)
    }

    /// How many voices are currently sounding.
    #[must_use]
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|voice| voice.active).count()
    }

    /// How many `play` calls have been refused for want of a voice.
    #[must_use]
    pub const fn dropped_voices(&self) -> u64 {
        self.dropped
    }

    /// Mixes the next block into `output`, an interleaved stereo buffer.
    ///
    /// The buffer is overwritten rather than added to, so a caller never has
    /// to remember to clear it — forgetting is how the previous block's audio
    /// ends up repeating.
    ///
    /// An odd-length buffer has its trailing sample left silent rather than
    /// being treated as a left channel with no right.
    pub fn fill(&mut self, output: &mut [f32]) {
        output.fill(0.0);
        let master = self.master;

        for voice in &mut self.voices {
            if !voice.active {
                continue;
            }
            let bus = self.bus_gains[voice.settings.bus.index()];
            let (left_gain, right_gain) = pan_gains(voice.settings.pan);
            let gain = master * bus * voice.settings.gain;
            let samples = voice.sound.samples();

            for frame in output.chunks_exact_mut(2) {
                if voice.position >= samples.len() {
                    if voice.settings.looping && !samples.is_empty() {
                        voice.position = 0;
                    } else {
                        voice.active = false;
                        break;
                    }
                }
                let sample = samples[voice.position] * gain;
                frame[0] += sample * left_gain;
                frame[1] += sample * right_gain;
                voice.position += 1;
            }
        }
    }

    /// Removes finished voices, freeing their held sounds.
    ///
    /// Called from the game thread rather than the audio thread, because
    /// dropping the last handle to a sound frees memory, and allocation is
    /// exactly what an audio callback must not do.
    pub fn collect(&mut self) {
        self.voices.retain(|voice| voice.active);
    }
}

/// The left and right gains for a pan position.
///
/// Constant-power rather than linear: a linear pan dips about 3 dB in the
/// middle, so a sound sweeping across the stereo field audibly sags as it
/// passes the centre. The sine-law curve keeps the total power flat.
#[must_use]
pub fn pan_gains(pan: f32) -> (f32, f32) {
    let clamped = pan.clamp(-1.0, 1.0);
    // Map [-1, 1] onto a quarter turn, then take the two table entries a
    // quarter cycle apart, which is cosine and sine of the same angle.
    let position = (clamped + 1.0) * 0.125;
    let left = crate::wave::sine(position + 0.25);
    let right = crate::wave::sine(position);
    (left.max(0.0), right.max(0.0))
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

    /// A one-second sound of constant amplitude, easy to reason about.
    fn tone(amplitude: f32, seconds: f32) -> Arc<Sound> {
        Arc::new(Sound::from_samples(vec![
            amplitude;
            crate::synth::sample_count(seconds)
        ]))
    }

    #[test]
    fn an_empty_mixer_produces_silence() {
        let mut mixer = Mixer::new();
        let mut output = vec![1.0f32; 64];
        mixer.fill(&mut output);
        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn the_buffer_is_overwritten_rather_than_accumulated() {
        // Otherwise the previous block's audio repeats under the new one.
        let mut mixer = Mixer::new();
        mixer.play(tone(0.5, 1.0), PlaySettings::default());
        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        let first = output.clone();
        mixer.fill(&mut output);
        // Different position in the sound, but not first + second.
        assert!(output.iter().all(|sample| sample.abs() <= 0.51));
        assert_eq!(first.len(), output.len());
    }

    #[test]
    fn a_played_sound_is_audible() {
        let mut mixer = Mixer::new();
        assert!(mixer
            .play(tone(0.5, 1.0), PlaySettings::default())
            .is_some());
        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        assert!(output.iter().any(|sample| sample.abs() > 0.1));
    }

    #[test]
    fn an_empty_sound_is_refused() {
        // A silently-ignored empty asset is far harder to notice than a
        // handle that never appears.
        let mut mixer = Mixer::new();
        assert!(mixer
            .play(Arc::new(Sound::default()), PlaySettings::default())
            .is_none());
        assert_eq!(mixer.active_voices(), 0);
    }

    #[test]
    fn a_voice_retires_when_its_sound_ends() {
        let mut mixer = Mixer::new();
        let id = mixer
            .play(tone(0.5, 0.001), PlaySettings::default())
            .expect("plays");
        let mut output = vec![0.0f32; 1024];
        mixer.fill(&mut output);
        assert!(!mixer.is_playing(id));
        assert_eq!(mixer.active_voices(), 0);
    }

    #[test]
    fn a_looping_voice_keeps_going() {
        let mut mixer = Mixer::new();
        let id = mixer
            .play(tone(0.5, 0.001), PlaySettings::default().looping())
            .expect("plays");
        let mut output = vec![0.0f32; 4096];
        mixer.fill(&mut output);
        assert!(mixer.is_playing(id));
        assert!(output.iter().all(|sample| sample.abs() > 0.0));
    }

    #[test]
    fn stopping_a_voice_silences_it() {
        let mut mixer = Mixer::new();
        let id = mixer
            .play(tone(0.5, 1.0), PlaySettings::default())
            .expect("plays");
        mixer.stop(id);
        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn a_stale_handle_cannot_silence_a_new_voice() {
        // The bug generations exist to prevent: a slot recycled under a
        // handle someone kept.
        let mut mixer = Mixer::new();
        let old = mixer
            .play(tone(0.5, 0.001), PlaySettings::default())
            .expect("plays");
        let mut output = vec![0.0f32; 1024];
        mixer.fill(&mut output);
        assert!(!mixer.is_playing(old));

        let new = mixer
            .play(tone(0.5, 1.0), PlaySettings::default())
            .expect("plays");
        mixer.stop(old);
        assert!(mixer.is_playing(new), "the new voice was silenced");
    }

    #[test]
    fn finished_slots_are_reused_rather_than_leaked() {
        let mut mixer = Mixer::new();
        for _ in 0..100 {
            mixer.play(tone(0.5, 0.001), PlaySettings::default());
            let mut output = vec![0.0f32; 1024];
            mixer.fill(&mut output);
        }
        assert!(
            mixer.voices.len() <= 2,
            "grew to {} slots",
            mixer.voices.len()
        );
    }

    #[test]
    fn the_voice_limit_refuses_rather_than_growing_without_bound() {
        let mut mixer = Mixer::new();
        mixer.set_voice_limit(4);
        for _ in 0..4 {
            assert!(mixer
                .play(tone(0.1, 1.0), PlaySettings::default())
                .is_some());
        }
        assert!(mixer
            .play(tone(0.1, 1.0), PlaySettings::default())
            .is_none());
        assert_eq!(mixer.dropped_voices(), 1);
        assert_eq!(mixer.active_voices(), 4);
    }

    #[test]
    fn bus_gains_are_independent() {
        let mut mixer = Mixer::new();
        mixer.set_bus_gain(Bus::Music, 0.0);
        mixer.play(tone(1.0, 1.0), PlaySettings::on(Bus::Music));
        mixer.play(tone(1.0, 1.0), PlaySettings::on(Bus::Effects));

        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        // Only the effects voice survives, at its own level.
        let peak = output.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        assert!((peak - 0.707).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn the_master_volume_scales_everything() {
        let mut mixer = Mixer::new();
        mixer.set_master(0.0);
        mixer.play(tone(1.0, 1.0), PlaySettings::default());
        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn volumes_are_clamped_to_a_sane_range() {
        let mut mixer = Mixer::new();
        mixer.set_master(9.0);
        assert_eq!(mixer.master(), 1.0);
        mixer.set_bus_gain(Bus::Music, -3.0);
        assert_eq!(mixer.bus_gain(Bus::Music), 0.0);
    }

    #[test]
    fn stopping_a_bus_leaves_the_others_playing() {
        let mut mixer = Mixer::new();
        let music = mixer
            .play(tone(0.5, 1.0), PlaySettings::on(Bus::Music))
            .expect("plays");
        let effect = mixer
            .play(tone(0.5, 1.0), PlaySettings::on(Bus::Effects))
            .expect("plays");
        mixer.stop_bus(Bus::Music);
        assert!(!mixer.is_playing(music));
        assert!(mixer.is_playing(effect));
    }

    #[test]
    fn panning_hard_left_silences_the_right_channel() {
        let mut mixer = Mixer::new();
        mixer.play(tone(1.0, 1.0), PlaySettings::default().panned(-1.0));
        let mut output = vec![0.0f32; 64];
        mixer.fill(&mut output);
        assert!(output[0].abs() > 0.9, "left was {}", output[0]);
        assert!(output[1].abs() < 0.01, "right was {}", output[1]);
    }

    #[test]
    fn panning_holds_constant_power_across_the_field() {
        // A linear pan dips in the middle; this is the check that it does not.
        for step in 0..=20 {
            let pan = step as f32 / 10.0 - 1.0;
            let (left, right) = pan_gains(pan);
            let power = left * left + right * right;
            assert!(
                (power - 1.0).abs() < 0.02,
                "pan {pan} had power {power} ({left}, {right})"
            );
        }
    }

    #[test]
    fn pan_positions_outside_the_range_are_clamped() {
        assert_eq!(pan_gains(-5.0), pan_gains(-1.0));
        assert_eq!(pan_gains(5.0), pan_gains(1.0));
    }

    #[test]
    fn an_odd_length_buffer_does_not_panic() {
        // A stereo frame is two samples; a buffer with a spare one must not
        // be read as a half-frame.
        let mut mixer = Mixer::new();
        mixer.play(tone(0.5, 1.0), PlaySettings::default());
        let mut output = vec![0.0f32; 65];
        mixer.fill(&mut output);
        assert_eq!(output[64], 0.0, "the odd sample should stay silent");
    }

    #[test]
    fn collecting_frees_finished_voices() {
        let mut mixer = Mixer::new();
        mixer.play(tone(0.5, 0.001), PlaySettings::default());
        let mut output = vec![0.0f32; 1024];
        mixer.fill(&mut output);
        mixer.collect();
        assert_eq!(mixer.voices.len(), 0);
    }

    #[test]
    fn mixing_is_reproducible() {
        let render = || {
            let mut mixer = Mixer::new();
            mixer.play(tone(0.4, 0.05), PlaySettings::default().panned(-0.3));
            mixer.play(tone(0.3, 0.05), PlaySettings::on(Bus::Music).panned(0.6));
            let mut output = vec![0.0f32; 512];
            mixer.fill(&mut output);
            output
        };
        assert_eq!(render(), render());
    }

    #[test]
    fn a_full_mix_stays_finite() {
        let mut mixer = Mixer::new();
        for _ in 0..DEFAULT_VOICE_LIMIT {
            mixer.play(tone(1.0, 1.0), PlaySettings::default());
        }
        let mut output = vec![0.0f32; 128];
        mixer.fill(&mut output);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn bus_names_are_distinct() {
        let names: std::collections::BTreeSet<&str> = Bus::ALL.iter().map(|b| b.name()).collect();
        assert_eq!(names.len(), Bus::ALL.len());
    }
}
