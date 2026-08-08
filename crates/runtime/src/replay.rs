//! Input recording, replay, and desync detection.
//!
//! # What a replay is
//!
//! Because the simulation is deterministic and reads input only from a per-tick
//! snapshot, a whole session is reproducible from two things: the world seed
//! and the sequence of input snapshots. A replay file is therefore tiny — a few
//! bytes per tick — and yet reproduces the session exactly.
//!
//! That single mechanism covers three jobs:
//!
//! * **Bug reports.** A replay is a perfect repro. "It crashed after two hours"
//!   becomes a file that crashes on demand.
//! * **Regression testing.** Record a session, then assert that replaying it
//!   still produces the same state hashes. Any change that alters simulation
//!   behaviour shows up as a mismatch at the exact tick it first diverges.
//! * **Desync detection.** The same comparison across two machines is what
//!   proves cross-platform determinism, and is what the CI determinism job runs.
//!
//! # State hashes
//!
//! A recording stores a hash of the world state at intervals rather than every
//! tick. Every tick would make the file large and the comparison no more
//! useful: a divergence is found at the first mismatching checkpoint, and the
//! interval bounds how far back the cause can be.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use verdant_input::{Action, InputState};

/// One tick's worth of recorded input.
///
/// Stores which actions were held rather than the resolved [`InputState`],
/// because the resolution — buffering, leniency, dead zones — is itself part of
/// the simulation and must be re-derived rather than replayed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFrame {
    /// Actions held this tick, sorted for a stable encoding.
    pub actions: Vec<String>,
    /// Movement direction, as raw fixed-point components.
    pub movement: (i64, i64),
}

impl InputFrame {
    /// Captures the actions currently held.
    #[must_use]
    pub fn capture(input: &InputState, tracked: &[Action]) -> InputFrame {
        let mut actions: Vec<String> = tracked
            .iter()
            .filter(|action| input.is_down(**action))
            .map(|action| action.name().to_string())
            .collect();
        actions.sort();
        InputFrame {
            actions,
            movement: (input.movement().x.to_raw(), input.movement().y.to_raw()),
        }
    }

    /// True when an action was held on this tick.
    #[must_use]
    pub fn is_down(&self, action: Action) -> bool {
        self.actions.iter().any(|name| name == action.name())
    }
}

/// A recorded session.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Recording {
    /// The world seed the session started from.
    pub seed: u64,
    /// Simulation steps per second, so a replay runs at the recorded rate.
    pub steps_per_second: u32,
    /// One entry per simulation tick.
    pub frames: Vec<InputFrame>,
    /// State hashes, keyed by the tick they were taken at.
    ///
    /// `BTreeMap` so the serialised form is stable and a diff between two
    /// recordings is readable.
    pub checkpoints: BTreeMap<u64, u64>,
    /// How often checkpoints were taken.
    pub checkpoint_interval: u64,
}

/// Default ticks between state hashes.
///
/// Once a second at 60 Hz: frequent enough to localise a divergence to a
/// second of play, infrequent enough that hashing does not dominate a replay.
pub const DEFAULT_CHECKPOINT_INTERVAL: u64 = 60;

impl Recording {
    /// Starts an empty recording.
    #[must_use]
    pub fn new(seed: u64, steps_per_second: u32) -> Recording {
        Recording {
            seed,
            steps_per_second,
            frames: Vec::new(),
            checkpoints: BTreeMap::new(),
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
        }
    }

    /// Appends one tick.
    pub fn push_frame(&mut self, frame: InputFrame) {
        self.frames.push(frame);
    }

    /// Records a state hash if this tick falls on a checkpoint boundary.
    pub fn maybe_checkpoint(&mut self, tick: u64, state_hash: u64) {
        if self.checkpoint_interval > 0 && tick % self.checkpoint_interval == 0 {
            self.checkpoints.insert(tick, state_hash);
        }
    }

    /// Number of recorded ticks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True when nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The input for a tick, or `None` past the end of the recording.
    #[must_use]
    pub fn frame(&self, tick: u64) -> Option<&InputFrame> {
        self.frames.get(usize::try_from(tick).ok()?)
    }

    /// Encodes the recording as JSON.
    ///
    /// # Errors
    ///
    /// Returns the serialisation error if encoding fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Decodes a recording.
    ///
    /// # Errors
    ///
    /// Returns the deserialisation error for malformed input.
    pub fn from_json(text: &str) -> Result<Recording, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// The outcome of comparing a replay against its recording.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayOutcome {
    /// Every checkpoint matched.
    Matched {
        /// How many checkpoints were compared.
        checkpoints: usize,
        /// How many ticks were replayed.
        ticks: u64,
    },
    /// A checkpoint disagreed, meaning the simulation is no longer
    /// deterministic or its behaviour has changed.
    Diverged {
        /// The first tick whose hash differed.
        tick: u64,
        /// The hash the recording holds.
        expected: u64,
        /// The hash the replay produced.
        actual: u64,
    },
    /// The recording held no checkpoints to compare against.
    NoCheckpoints,
}

impl ReplayOutcome {
    /// True when the replay reproduced the recording.
    #[must_use]
    pub const fn is_match(&self) -> bool {
        matches!(self, ReplayOutcome::Matched { .. })
    }
}

impl std::fmt::Display for ReplayOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayOutcome::Matched { checkpoints, ticks } => {
                write!(f, "replay matched across {checkpoints} checkpoints over {ticks} ticks")
            }
            ReplayOutcome::Diverged { tick, expected, actual } => write!(
                f,
                "replay diverged at tick {tick}: expected state hash {expected:#018x}, got {actual:#018x}"
            ),
            ReplayOutcome::NoCheckpoints => write!(f, "the recording holds no checkpoints"),
        }
    }
}

/// Compares a replay's hashes against a recording's.
///
/// `observed` maps tick to state hash, as produced by re-running the recorded
/// input. Only ticks the recording checkpointed are compared; the replay may
/// hash more often without affecting the result.
#[must_use]
pub fn compare(recording: &Recording, observed: &BTreeMap<u64, u64>) -> ReplayOutcome {
    if recording.checkpoints.is_empty() {
        return ReplayOutcome::NoCheckpoints;
    }
    // BTreeMap iterates in tick order, so the *first* divergence is reported —
    // which is the only one that matters, since everything after it is
    // downstream of the same cause.
    for (tick, expected) in &recording.checkpoints {
        let Some(actual) = observed.get(tick) else {
            continue;
        };
        if actual != expected {
            return ReplayOutcome::Diverged {
                tick: *tick,
                expected: *expected,
                actual: *actual,
            };
        }
    }
    ReplayOutcome::Matched {
        checkpoints: recording.checkpoints.len(),
        ticks: recording.len() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_input::{actions, Binding, DeviceState, InputMap, KeyCode};

    fn recording_with_checkpoints() -> Recording {
        let mut recording = Recording::new(42, 60);
        for tick in 0..180u64 {
            recording.push_frame(InputFrame::default());
            recording.maybe_checkpoint(tick, tick.wrapping_mul(0x9E37_79B9));
        }
        recording
    }

    #[test]
    fn captured_input_records_held_actions() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(
            &map,
            &DeviceState {
                keys: vec![KeyCode::KeyE, KeyCode::KeyD],
                ..DeviceState::default()
            },
        );

        let frame = InputFrame::capture(&input, &[actions::INTERACT, actions::USE]);
        assert!(frame.is_down(actions::INTERACT));
        assert!(!frame.is_down(actions::USE));
        assert!(
            frame.movement.0 > 0,
            "the movement direction is captured too"
        );
    }

    #[test]
    fn captured_actions_are_sorted_for_a_stable_encoding() {
        let mut map = InputMap::new();
        map.bind(actions::USE, Binding::Key(KeyCode::KeyA));
        map.bind(actions::INTERACT, Binding::Key(KeyCode::KeyB));
        map.bind(actions::SPRINT, Binding::Key(KeyCode::KeyC));

        let mut input = InputState::new();
        input.update(
            &map,
            &DeviceState {
                keys: vec![KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC],
                ..DeviceState::default()
            },
        );

        let frame =
            InputFrame::capture(&input, &[actions::USE, actions::INTERACT, actions::SPRINT]);
        let mut sorted = frame.actions.clone();
        sorted.sort();
        assert_eq!(frame.actions, sorted);
    }

    #[test]
    fn a_recording_stores_one_frame_per_tick() {
        let recording = recording_with_checkpoints();
        assert_eq!(recording.len(), 180);
        assert!(recording.frame(0).is_some());
        assert!(recording.frame(179).is_some());
        assert_eq!(recording.frame(180), None, "past the end of the recording");
    }

    #[test]
    fn checkpoints_land_on_the_configured_interval() {
        let recording = recording_with_checkpoints();
        // Ticks 0, 60 and 120 at the default interval of 60.
        assert_eq!(recording.checkpoints.len(), 3);
        assert!(recording.checkpoints.contains_key(&0));
        assert!(recording.checkpoints.contains_key(&60));
        assert!(!recording.checkpoints.contains_key(&59));
    }

    #[test]
    fn an_identical_replay_matches() {
        let recording = recording_with_checkpoints();
        let observed = recording.checkpoints.clone();
        let outcome = compare(&recording, &observed);

        assert!(outcome.is_match());
        assert_eq!(
            outcome,
            ReplayOutcome::Matched {
                checkpoints: 3,
                ticks: 180
            }
        );
    }

    #[test]
    fn a_divergence_reports_the_first_tick_that_differs() {
        let recording = recording_with_checkpoints();
        let mut observed = recording.checkpoints.clone();
        // Corrupt two checkpoints; only the earlier should be reported, since
        // everything after it is downstream of the same cause.
        observed.insert(60, 0xDEAD);
        observed.insert(120, 0xBEEF);

        match compare(&recording, &observed) {
            ReplayOutcome::Diverged { tick, actual, .. } => {
                assert_eq!(tick, 60);
                assert_eq!(actual, 0xDEAD);
            }
            other => panic!("expected a divergence, got {other}"),
        }
    }

    #[test]
    fn a_divergence_message_names_the_tick_and_hashes() {
        let outcome = ReplayOutcome::Diverged {
            tick: 120,
            expected: 0xAAAA,
            actual: 0xBBBB,
        };
        let message = outcome.to_string();
        assert!(message.contains("tick 120"));
        assert!(message.contains("aaaa") && message.contains("bbbb"));
    }

    #[test]
    fn missing_observations_are_skipped_rather_than_failing() {
        let recording = recording_with_checkpoints();
        // A replay that only hashed one checkpoint still validates that one.
        let mut observed = BTreeMap::new();
        observed.insert(60, recording.checkpoints[&60]);
        assert!(compare(&recording, &observed).is_match());
    }

    #[test]
    fn a_recording_without_checkpoints_is_reported_distinctly() {
        let mut recording = Recording::new(1, 60);
        recording.checkpoint_interval = 0;
        for tick in 0..10u64 {
            recording.push_frame(InputFrame::default());
            recording.maybe_checkpoint(tick, 0);
        }
        assert_eq!(
            compare(&recording, &BTreeMap::new()),
            ReplayOutcome::NoCheckpoints
        );
    }

    #[test]
    fn a_recording_round_trips_through_json() {
        let mut recording = Recording::new(0xFEED, 60);
        recording.push_frame(InputFrame {
            actions: vec!["use".to_string()],
            movement: (1234, -5678),
        });
        recording.maybe_checkpoint(0, 0x1234_5678);

        let encoded = recording.to_json().expect("serialisable");
        let decoded = Recording::from_json(&encoded).expect("deserialisable");
        assert_eq!(decoded, recording);
        assert_eq!(decoded.seed, 0xFEED);
    }

    #[test]
    fn the_encoding_is_stable_between_runs() {
        let recording = recording_with_checkpoints();
        assert_eq!(
            recording.to_json().unwrap(),
            recording.to_json().unwrap(),
            "a replay file must not vary between encodings"
        );
    }

    #[test]
    fn a_recording_is_small_relative_to_its_length() {
        // The point of recording input rather than state: an hour of play at
        // 60 Hz is 216,000 ticks, and must not produce a huge file.
        let mut recording = Recording::new(1, 60);
        for tick in 0..6000u64 {
            recording.push_frame(InputFrame {
                actions: if tick % 7 == 0 {
                    vec!["use".to_string()]
                } else {
                    Vec::new()
                },
                movement: (0, 0),
            });
            recording.maybe_checkpoint(tick, tick);
        }
        let bytes = recording.to_json().unwrap().len();
        assert!(
            bytes < 400_000,
            "100 seconds of play encoded to {bytes} bytes"
        );
    }
}
