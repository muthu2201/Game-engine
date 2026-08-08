//! Playing the mixer's output through the machine's speakers.
//!
//! This is the only part of the crate that talks to the operating system, and
//! the only part that is not deterministic — the device chooses the sample
//! rate and the buffer size, and the callback runs on a thread the driver
//! owns. Everything else is arranged so that this layer stays as thin as
//! possible: the callback locks a mutex, calls [`Mixer::fill`], and returns.
//!
//! ## Failure is not fatal
//!
//! A machine with no sound card, a container with no audio server, a device
//! unplugged mid-session — all of these are normal, and none of them should
//! stop a game. [`AudioDevice::open`] returns an error the caller is expected
//! to log and ignore, and [`AudioDevice::silent`] gives back a working handle
//! that simply makes no noise.

use crate::mixer::{Mixer, PlaySettings, VoiceId};
use crate::synth::{Sound, SAMPLE_RATE};
use std::sync::{Arc, Mutex};

/// Why audio output could not be started.
#[derive(Debug)]
pub enum AudioError {
    /// The platform reported no output device.
    NoDevice,
    /// A device exists but would not accept a stream configuration.
    Unsupported(String),
    /// The stream was created but would not start.
    Stream(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::NoDevice => write!(f, "no audio output device is available"),
            AudioError::Unsupported(detail) => {
                write!(f, "no supported output configuration: {detail}")
            }
            AudioError::Stream(detail) => write!(f, "the audio stream failed: {detail}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// A handle to the mixer, shared with the audio thread.
///
/// A mutex rather than a lock-free queue: the critical section is a few
/// microseconds of adding floats, and a mutex that is never contended for long
/// is both simpler and easier to reason about than a ring buffer that has to
/// be right.
pub type SharedMixer = Arc<Mutex<Mixer>>;

/// An open output stream, or a silent stand-in.
pub struct AudioDevice {
    mixer: SharedMixer,
    /// The live stream. Dropping it stops playback, so it is held even though
    /// nothing reads it.
    #[cfg(feature = "device")]
    _stream: Option<cpal::Stream>,
    sample_rate: u32,
}

impl AudioDevice {
    /// A device that mixes nothing and plays nothing.
    ///
    /// Used when output cannot be opened, and by tests, so that game code
    /// never needs a branch for "audio is unavailable".
    #[must_use]
    pub fn silent() -> AudioDevice {
        AudioDevice {
            mixer: Arc::new(Mutex::new(Mixer::new())),
            #[cfg(feature = "device")]
            _stream: None,
            sample_rate: SAMPLE_RATE,
        }
    }

    /// The shared mixer, for queuing sounds from the game thread.
    #[must_use]
    pub fn mixer(&self) -> &SharedMixer {
        &self.mixer
    }

    /// The rate the device is actually running at.
    ///
    /// May differ from [`SAMPLE_RATE`]: the driver has the final say, and a
    /// mismatch means rendered sounds play back slightly fast or slow.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// True when this device is a silent stand-in rather than a real stream.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        #[cfg(feature = "device")]
        {
            self._stream.is_none()
        }
        #[cfg(not(feature = "device"))]
        {
            true
        }
    }

    /// Queues a sound, ignoring the result.
    ///
    /// The common case at a call site is "play this if you can", and a
    /// dropped voice is not something gameplay code can do anything about.
    pub fn play(&self, sound: Arc<Sound>, settings: PlaySettings) -> Option<VoiceId> {
        self.mixer.lock().ok()?.play(sound, settings)
    }

    /// Runs `edit` against the mixer, for volume changes and stops.
    ///
    /// Returns `None` if the audio thread panicked while holding the lock,
    /// which poisons it; the game carries on without sound rather than
    /// panicking in turn.
    pub fn with_mixer<T>(&self, edit: impl FnOnce(&mut Mixer) -> T) -> Option<T> {
        self.mixer.lock().ok().map(|mut mixer| edit(&mut mixer))
    }

    /// Frees voices that have finished.
    ///
    /// Call once a frame from the game thread. Dropping the last reference to
    /// a sound deallocates it, which is why this is not done in the callback.
    pub fn collect(&self) {
        if let Ok(mut mixer) = self.mixer.lock() {
            mixer.collect();
        }
    }
}

impl std::fmt::Debug for AudioDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioDevice")
            .field("sample_rate", &self.sample_rate)
            .field("silent", &self.is_silent())
            .finish()
    }
}

#[cfg(feature = "device")]
mod backend {
    use super::{AudioDevice, AudioError, Mixer, SharedMixer};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};

    impl AudioDevice {
        /// Opens the default output device.
        ///
        /// # Errors
        ///
        /// Returns [`AudioError::NoDevice`] when the platform reports no
        /// output, [`AudioError::Unsupported`] when no configuration is
        /// accepted, and [`AudioError::Stream`] when the stream will not
        /// start.
        pub fn open() -> Result<AudioDevice, AudioError> {
            let host = cpal::default_host();
            let device = host.default_output_device().ok_or(AudioError::NoDevice)?;
            let config = device
                .default_output_config()
                .map_err(|error| AudioError::Unsupported(error.to_string()))?;

            let sample_rate = config.sample_rate().0;
            let channels = config.channels() as usize;
            let mixer: SharedMixer = Arc::new(Mutex::new(Mixer::new()));
            let callback_mixer = Arc::clone(&mixer);

            // Scratch space allocated once, up front: allocating inside the
            // callback is the classic way to introduce an audible glitch.
            let mut scratch: Vec<f32> = Vec::new();

            let stream = device
                .build_output_stream(
                    &config.config(),
                    move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let frames = output.len() / channels.max(1);
                        scratch.resize(frames * 2, 0.0);

                        match callback_mixer.lock() {
                            Ok(mut mixer) => mixer.fill(&mut scratch),
                            Err(_) => {
                                // Poisoned: emit silence rather than
                                // whatever was left in the buffer.
                                output.fill(0.0);
                                return;
                            }
                        }

                        // Spread the stereo mix over however many channels
                        // the device actually has.
                        for (frame, block) in scratch
                            .chunks_exact(2)
                            .zip(output.chunks_exact_mut(channels.max(1)))
                        {
                            for (index, sample) in block.iter_mut().enumerate() {
                                *sample = frame[index % 2];
                            }
                        }
                    },
                    |error| log::error!("audio stream error: {error}"),
                    None,
                )
                .map_err(|error| AudioError::Stream(error.to_string()))?;

            stream
                .play()
                .map_err(|error| AudioError::Stream(error.to_string()))?;

            Ok(AudioDevice {
                mixer,
                _stream: Some(stream),
                sample_rate,
            })
        }

        /// Opens the default device, falling back to silence.
        ///
        /// This is what a game should call: sound is a nicety, and a machine
        /// without it should still play.
        #[must_use]
        pub fn open_or_silent() -> AudioDevice {
            match AudioDevice::open() {
                Ok(device) => device,
                Err(error) => {
                    log::warn!("audio unavailable, continuing without sound: {error}");
                    AudioDevice::silent()
                }
            }
        }
    }
}

#[cfg(not(feature = "device"))]
impl AudioDevice {
    /// Without the `device` feature there is no backend, so this is always
    /// silent.
    #[must_use]
    pub fn open_or_silent() -> AudioDevice {
        AudioDevice::silent()
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
    use crate::mixer::Bus;

    #[test]
    fn a_silent_device_still_accepts_sounds() {
        // Game code must not need a branch for "no audio".
        let device = AudioDevice::silent();
        let sound = Arc::new(Sound::from_samples(vec![0.5; 128]));
        assert!(device.play(sound, PlaySettings::default()).is_some());
        assert!(device.is_silent());
    }

    #[test]
    fn a_silent_device_reports_the_engines_sample_rate() {
        assert_eq!(AudioDevice::silent().sample_rate(), SAMPLE_RATE);
    }

    #[test]
    fn the_mixer_can_be_edited_through_the_handle() {
        let device = AudioDevice::silent();
        device.with_mixer(|mixer| mixer.set_bus_gain(Bus::Music, 0.25));
        let gain = device
            .with_mixer(|mixer| mixer.bus_gain(Bus::Music))
            .expect("the lock is healthy");
        assert!((gain - 0.25).abs() < 1e-6);
    }

    #[test]
    fn collecting_frees_finished_voices() {
        let device = AudioDevice::silent();
        let sound = Arc::new(Sound::from_samples(vec![0.5; 8]));
        device.play(sound, PlaySettings::default());
        device.with_mixer(|mixer| {
            let mut output = vec![0.0f32; 64];
            mixer.fill(&mut output);
        });
        device.collect();
        let active = device.with_mixer(|mixer| mixer.active_voices());
        assert_eq!(active, Some(0));
    }

    #[test]
    fn opening_never_panics_on_a_machine_without_audio() {
        // CI containers have no sound card; this must degrade, not crash.
        let device = AudioDevice::open_or_silent();
        assert!(device.sample_rate() > 0);
    }
}
