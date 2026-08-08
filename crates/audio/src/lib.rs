//! # Verdant audio
//!
//! A deterministic software mixer with procedural synthesis and generated
//! music.
//!
//! ## Determinism
//!
//! Every sample this crate produces is a pure function of its inputs, on every
//! platform. Two things make that true, and both are deliberate:
//!
//! * **No libm.** `sin`, `powf` and friends differ between platforms in the
//!   last bits. Oscillators read a table built from
//!   [`verdant_core_math::Fx::sin`], and pitches come from a table of the
//!   twelve semitone ratios, so the arithmetic is plain multiplication and
//!   addition throughout.
//! * **Seeded noise.** The noise oscillator draws from the engine's PCG
//!   generator, with a separate stream per note, so a percussion part is
//!   reproducible and editing a score does not shift the sounds after the
//!   edit.
//!
//! The one exception is [`device`], which hands finished samples to the
//! operating system. Everything above it can be rendered and compared
//! byte-for-byte in a test.
//!
//! ## The shape of the pipeline
//!
//! ```text
//! Theme ──compose──> Score ──perform──> Sound ──play──> Mixer ──fill──> device
//!  (seed)            (notes)           (samples)      (voices)      (speakers)
//! ```
//!
//! Synthesis happens once, ahead of time. The audio callback only reads
//! samples, scales them and adds them — no allocation, no synthesis, no
//! locking beyond a single uncontended mutex. An audio callback that misses
//! its deadline is a click the player hears, so the work is moved out of it
//! rather than optimised inside it.
//!
//! ## Example
//!
//! ```
//! use verdant_audio::{music, AudioDevice, PlaySettings, Bus};
//!
//! // Eight bars in a major pentatonic, generated from a seed.
//! let theme = music::Theme { bars: 2, ..music::Theme::default() };
//! let track = music::generate(&theme, 20_260_808).shared();
//!
//! // The same seed always produces the same bytes.
//! assert_eq!(music::generate(&theme, 20_260_808).samples(), track.samples());
//!
//! // A silent device behaves exactly like a real one, minus the noise.
//! let audio = AudioDevice::silent();
//! audio.play(track, PlaySettings::on(Bus::Music).looping());
//! ```

#![doc(html_no_source)]

pub mod device;
pub mod envelope;
pub mod mixer;
pub mod music;
pub mod synth;
pub mod wave;

pub use device::{AudioDevice, AudioError, SharedMixer};
pub use envelope::Adsr;
pub use mixer::{pan_gains, Bus, Mixer, PlaySettings, VoiceId, DEFAULT_VOICE_LIMIT};
pub use music::{Chord, Ensemble, Scale, Score, Theme};
pub use synth::{
    layer, note_frequency, render, sample_count, Instrument, Note, Sound, SAMPLE_RATE,
};
pub use wave::Waveform;
