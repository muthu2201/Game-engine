//! # Verdant runtime
//!
//! The game loop, and the replay machinery that keeps it honest.
//!
//! ## The loop
//!
//! ```text
//!   real time ──► Timestep ──► N fixed simulation steps ──► alpha
//!                                      │                      │
//!                                      ▼                      ▼
//!                                  Schedule              interpolated
//!                                 (systems)                render
//! ```
//!
//! The simulation advances in fixed steps regardless of frame rate, and the
//! renderer draws between the last two states using the leftover fraction.
//! See [`timestep`] for why, and for how the loop escapes a spiral of death.
//!
//! ## Determinism, end to end
//!
//! Every layer beneath this one was built to be reproducible: fixed-point
//! arithmetic, ordered ECS iteration, seeded randomness, input as a per-tick
//! snapshot. [`replay`] is what turns those guarantees into something testable
//! — record a session's input, replay it, and compare state hashes. The CI
//! determinism job runs exactly that across Linux, Windows and macOS.
//!
//! ## Example
//!
//! ```
//! use verdant_core_ecs::{Component, World};
//! use verdant_core_math::Fx;
//! use verdant_runtime::{App, Timestep};
//!
//! #[derive(Debug)] struct Ticks(u32);
//! impl Component for Ticks {}
//!
//! let mut app = App::new();
//! app.world_mut().spawn((Ticks(0),));
//! app.schedule_mut().add_system("count", |world: &mut World| {
//!     for (_entity, (ticks,)) in world.query::<(&mut Ticks,)>() {
//!         ticks.0 += 1;
//!     }
//! });
//!
//! // One second of real time at the standard 60 Hz rate.
//! for _ in 0..60 {
//!     app.advance(Fx::ONE / 60);
//! }
//! assert_eq!(app.tick(), 60);
//! ```

#![doc(html_no_source)]

pub mod replay;
pub mod timestep;

pub use replay::{compare, InputFrame, Recording, ReplayOutcome, DEFAULT_CHECKPOINT_INTERVAL};
pub use timestep::{StepPlan, Timestep, DEFAULT_STEPS_PER_SECOND};

use std::collections::BTreeMap;
use verdant_core_ecs::{Schedule, World};
use verdant_core_math::{Fx, StateHasher};

/// A world, a schedule, and the loop that drives them.
///
/// `App` owns the simulation side only. Windowing, rendering and audio output
/// live outside it and read the world after each advance — which is what keeps
/// the simulation testable without a window, and keeps non-deterministic
/// subsystems from being able to influence it.
pub struct App {
    world: World,
    schedule: Schedule,
    timestep: Timestep,
    tick: u64,
    recording: Option<Recording>,
    observed_hashes: BTreeMap<u64, u64>,
    hash_interval: u64,
}

impl Default for App {
    fn default() -> App {
        App::new()
    }
}

impl App {
    /// Creates an app at the standard simulation rate.
    #[must_use]
    pub fn new() -> App {
        App::with_timestep(Timestep::standard())
    }

    /// Creates an app with a custom timestep.
    #[must_use]
    pub fn with_timestep(timestep: Timestep) -> App {
        App {
            world: World::new(),
            schedule: Schedule::new(),
            timestep,
            tick: 0,
            recording: None,
            observed_hashes: BTreeMap::new(),
            hash_interval: DEFAULT_CHECKPOINT_INTERVAL,
        }
    }

    /// The simulation world.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Mutable access to the world, for setup and for the renderer's extract.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// The system schedule.
    #[must_use]
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Mutable access to the schedule, for registering systems.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// The timestep accumulator.
    #[must_use]
    pub fn timestep(&self) -> &Timestep {
        &self.timestep
    }

    /// Mutable access to the timestep.
    pub fn timestep_mut(&mut self) -> &mut Timestep {
        &mut self.timestep
    }

    /// Simulation ticks run so far.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// How far between the last two simulation states to draw, `[0, 1)`.
    #[must_use]
    pub fn interpolation_alpha(&self) -> Fx {
        self.timestep.alpha()
    }

    /// Feeds elapsed real time in and runs the resulting simulation steps.
    ///
    /// Returns what the frame decided to do, so a caller can report overload.
    pub fn advance(&mut self, elapsed: Fx) -> StepPlan {
        let plan = self.timestep.advance(elapsed);
        for _ in 0..plan.steps {
            self.step();
        }
        plan
    }

    /// Runs exactly one simulation step.
    ///
    /// Exposed separately from [`App::advance`] so tests and replays can drive
    /// the simulation tick by tick without going through a clock.
    pub fn step(&mut self) {
        self.schedule.run(&mut self.world);
        self.world.advance_tick();
        self.tick += 1;

        if self.hash_interval > 0 && self.tick % self.hash_interval == 0 {
            let hash = self.state_hash();
            self.observed_hashes.insert(self.tick, hash);
            if let Some(recording) = self.recording.as_mut() {
                recording.maybe_checkpoint(self.tick, hash);
            }
        }
    }

    /// Begins recording a session.
    ///
    /// The recording captures the seed and the checkpoints; the caller is
    /// responsible for pushing an [`InputFrame`] per tick, since only it knows
    /// which actions the game defines.
    pub fn start_recording(&mut self, seed: u64) {
        let mut recording = Recording::new(seed, DEFAULT_STEPS_PER_SECOND);
        recording.checkpoint_interval = self.hash_interval;
        self.recording = Some(recording);
    }

    /// Appends an input frame to the active recording, if any.
    pub fn record_input(&mut self, frame: InputFrame) {
        if let Some(recording) = self.recording.as_mut() {
            recording.push_frame(frame);
        }
    }

    /// Stops recording and returns what was captured.
    pub fn finish_recording(&mut self) -> Option<Recording> {
        self.recording.take()
    }

    /// The state hashes observed so far, for comparison against a recording.
    #[must_use]
    pub fn observed_hashes(&self) -> &BTreeMap<u64, u64> {
        &self.observed_hashes
    }

    /// Sets how often state is hashed. Zero disables hashing entirely.
    pub fn set_hash_interval(&mut self, interval: u64) {
        self.hash_interval = interval;
    }

    /// Fingerprints the world's current state.
    ///
    /// Hashes the entity set and the world's tick. Component *values* are
    /// deliberately not included here: a generic hash would have to walk
    /// type-erased storage, and a game knows far better which of its components
    /// actually matter. Games extend this by hashing their own state into the
    /// value returned — see the Verdant Hollow simulation for an example.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_u64(self.tick);
        hasher.write_u32(self.world.tick().0);
        hasher.write_u32(self.world.entity_count());
        // Entity iteration is in ascending index order, which is part of the
        // ECS's determinism contract.
        for entity in self.world.entities() {
            hasher.write_u64(entity.to_bits());
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_ecs::Component;

    #[derive(Debug)]
    struct Counter(u32);
    impl Component for Counter {}

    /// An app whose only system increments a counter.
    fn counting_app() -> App {
        let mut app = App::new();
        app.world_mut().spawn((Counter(0),));
        app.schedule_mut().add_system("count", |world: &mut World| {
            for (_entity, (counter,)) in world.query::<(&mut Counter,)>() {
                counter.0 += 1;
            }
        });
        app
    }

    #[test]
    fn advancing_runs_the_schedule_once_per_step() {
        let mut app = counting_app();
        for _ in 0..60 {
            app.advance(Fx::ONE / 60);
        }
        assert_eq!(app.tick(), 60);

        let counter = app
            .world()
            .query_ref::<(&Counter,)>()
            .map(|(_, (counter,))| counter.0)
            .next()
            .expect("the counter entity exists");
        assert_eq!(counter, 60);
    }

    #[test]
    fn a_short_frame_runs_nothing() {
        let mut app = counting_app();
        app.advance(Fx::ONE / 600);
        assert_eq!(app.tick(), 0);
    }

    #[test]
    fn stepping_directly_bypasses_the_clock() {
        let mut app = counting_app();
        for _ in 0..10 {
            app.step();
        }
        assert_eq!(app.tick(), 10);
    }

    #[test]
    fn the_world_tick_advances_with_the_app() {
        let mut app = counting_app();
        let before = app.world().tick();
        app.step();
        assert!(app.world().tick().is_newer_than(before));
    }

    #[test]
    fn state_hashes_are_recorded_at_the_configured_interval() {
        let mut app = counting_app();
        app.set_hash_interval(10);
        for _ in 0..25 {
            app.step();
        }
        assert_eq!(app.observed_hashes().len(), 2, "ticks 10 and 20");
        assert!(app.observed_hashes().contains_key(&10));
        assert!(app.observed_hashes().contains_key(&20));
    }

    #[test]
    fn hashing_can_be_disabled() {
        let mut app = counting_app();
        app.set_hash_interval(0);
        for _ in 0..100 {
            app.step();
        }
        assert!(app.observed_hashes().is_empty());
    }

    #[test]
    fn the_state_hash_reflects_the_entity_set() {
        let mut app = App::new();
        let before = app.state_hash();
        app.world_mut().spawn((Counter(0),));
        assert_ne!(
            app.state_hash(),
            before,
            "spawning must change the fingerprint"
        );
    }

    #[test]
    fn recording_captures_checkpoints_alongside_input() {
        let mut app = counting_app();
        app.set_hash_interval(5);
        app.start_recording(1234);

        for _ in 0..20 {
            app.record_input(InputFrame::default());
            app.step();
        }

        let recording = app.finish_recording().expect("recording was started");
        assert_eq!(recording.seed, 1234);
        assert_eq!(recording.len(), 20);
        assert_eq!(recording.checkpoints.len(), 4, "ticks 5, 10, 15 and 20");
    }

    #[test]
    fn input_is_only_recorded_while_recording() {
        let mut app = counting_app();
        app.record_input(InputFrame::default());
        assert!(app.finish_recording().is_none());
    }

    #[test]
    fn determinism_a_replayed_session_reproduces_its_checkpoints() {
        // The property the whole engine is built for: run a session, record it,
        // run it again, and get identical state hashes.
        let record_session = || {
            let mut app = counting_app();
            app.set_hash_interval(10);
            app.start_recording(999);
            for _ in 0..100 {
                app.record_input(InputFrame::default());
                app.step();
            }
            app.finish_recording().expect("recording was started")
        };

        let recording = record_session();

        // Replay: the same simulation driven the same way.
        let mut replay = counting_app();
        replay.set_hash_interval(10);
        for tick in 0..recording.len() as u64 {
            let _ = recording.frame(tick);
            replay.step();
        }

        let outcome = compare(&recording, replay.observed_hashes());
        assert!(outcome.is_match(), "{outcome}");
    }

    #[test]
    fn determinism_a_changed_simulation_is_detected() {
        // The failure the comparison exists to catch: a simulation whose
        // behaviour changed must not silently pass a replay.
        let mut recorded = counting_app();
        recorded.set_hash_interval(5);
        recorded.start_recording(1);
        for _ in 0..20 {
            recorded.step();
        }
        let recording = recorded.finish_recording().unwrap();

        // A "new build" that spawns an extra entity partway through.
        let mut changed = counting_app();
        changed.set_hash_interval(5);
        for tick in 0..20 {
            if tick == 7 {
                changed.world_mut().spawn((Counter(0),));
            }
            changed.step();
        }

        let outcome = compare(&recording, changed.observed_hashes());
        assert!(
            !outcome.is_match(),
            "the divergence should have been caught"
        );
        match outcome {
            ReplayOutcome::Diverged { tick, .. } => {
                assert_eq!(tick, 10, "the first checkpoint after the change");
            }
            other => panic!("expected a divergence, got {other}"),
        }
    }

    #[test]
    fn an_overloaded_frame_is_reported_rather_than_spiralling() {
        let mut app = counting_app();
        let plan = app.advance(Fx::from_num(10));
        assert!(plan.is_overloaded());
        assert!(app.tick() <= 5, "the step cap held");
    }
}
