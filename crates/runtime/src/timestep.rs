//! The fixed-timestep accumulator.
//!
//! # Why the simulation does not use the frame's delta time
//!
//! Advancing the simulation by however long the last frame took makes physics
//! and gameplay depend on the display's refresh rate and on momentary hitches.
//! Two players on 60 Hz and 144 Hz monitors get different results, a stutter
//! changes where a body ends up, and no replay reproduces.
//!
//! Instead the simulation runs at a fixed rate and the *renderer* absorbs the
//! variation: elapsed real time accumulates, whole simulation steps are taken
//! out of it, and whatever fraction of a step is left over becomes the
//! interpolation factor the renderer uses to draw between the last two states.
//!
//! # The spiral of death
//!
//! If a simulation step takes longer than the step it represents, the
//! accumulator grows faster than it drains and the loop tries to catch up
//! forever, freezing the game. [`Timestep::advance`] therefore caps how many
//! steps it will run in one frame and discards the excess. The game runs in
//! slow motion under sustained overload — which is recoverable — rather than
//! locking up, which is not.

use verdant_core_math::Fx;

/// Accumulates real time into fixed simulation steps.
#[derive(Clone, Debug)]
pub struct Timestep {
    /// Seconds each simulation step represents.
    step: Fx,
    /// Unconsumed real time.
    accumulator: Fx,
    /// Most steps to run in a single frame.
    max_steps_per_frame: u32,
    /// Total steps taken since the loop started.
    total_steps: u64,
    /// Steps dropped to escape a spiral, for diagnostics.
    dropped_steps: u64,
}

/// The engine's default simulation rate.
///
/// 60 Hz because it divides evenly into the common refresh rates and because
/// a 1/60 step is short enough that interpolation is imperceptible.
pub const DEFAULT_STEPS_PER_SECOND: u32 = 60;

impl Timestep {
    /// Creates a timestep running at `steps_per_second`.
    ///
    /// # Panics
    ///
    /// Panics if `steps_per_second` is zero.
    #[must_use]
    pub fn new(steps_per_second: u32) -> Timestep {
        assert!(
            steps_per_second > 0,
            "Timestep: the simulation rate must be positive"
        );
        Timestep {
            step: Fx::ONE / i32::try_from(steps_per_second).unwrap_or(i32::MAX),
            accumulator: Fx::ZERO,
            // Five steps is a twelfth of a second of catch-up at 60 Hz: enough
            // to ride out a garbage collection or a texture upload, short
            // enough that a genuinely overloaded frame degrades immediately
            // rather than compounding.
            max_steps_per_frame: 5,
            total_steps: 0,
            dropped_steps: 0,
        }
    }

    /// A timestep at [`DEFAULT_STEPS_PER_SECOND`].
    #[must_use]
    pub fn standard() -> Timestep {
        Timestep::new(DEFAULT_STEPS_PER_SECOND)
    }

    /// Sets how many steps may run in one frame before the excess is dropped.
    ///
    /// # Panics
    ///
    /// Panics if `max_steps` is zero, which would stop the simulation entirely.
    pub fn set_max_steps_per_frame(&mut self, max_steps: u32) {
        assert!(
            max_steps > 0,
            "Timestep: at least one step per frame must be allowed"
        );
        self.max_steps_per_frame = max_steps;
    }

    /// Seconds one simulation step represents.
    #[inline]
    #[must_use]
    pub fn step_seconds(&self) -> Fx {
        self.step
    }

    /// Feeds elapsed real time in and reports how many steps to run.
    ///
    /// Negative or non-finite input is ignored: a clock that jumps backwards —
    /// which happens on suspend and on some virtualised hosts — must not run
    /// the simulation in reverse.
    pub fn advance(&mut self, elapsed: Fx) -> StepPlan {
        if elapsed.is_positive() {
            self.accumulator += elapsed;
        }

        let mut steps = 0;
        while self.accumulator >= self.step && steps < self.max_steps_per_frame {
            self.accumulator -= self.step;
            steps += 1;
        }

        // Still behind after the cap: abandon the backlog rather than trying to
        // catch up on a later frame, which is what turns a hitch into a freeze.
        let mut dropped = 0;
        while self.accumulator >= self.step {
            self.accumulator -= self.step;
            dropped += 1;
        }

        self.total_steps += u64::from(steps);
        self.dropped_steps += u64::from(dropped);

        StepPlan {
            steps,
            dropped,
            // How far into the next step we are, for the renderer to
            // interpolate with. Always in [0, 1).
            alpha: self.accumulator / self.step,
        }
    }

    /// Unconsumed time, as a fraction of one step.
    #[must_use]
    pub fn alpha(&self) -> Fx {
        self.accumulator / self.step
    }

    /// Total simulation steps run.
    #[must_use]
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Steps discarded to escape a spiral.
    ///
    /// A non-zero count means frames are taking longer than the simulation
    /// rate allows, which is the number to watch when the game feels sluggish.
    #[must_use]
    pub fn dropped_steps(&self) -> u64 {
        self.dropped_steps
    }

    /// Simulated seconds elapsed, exactly.
    ///
    /// Derived from the step count rather than accumulated per frame, so it
    /// carries no drift no matter how long the session runs.
    #[must_use]
    pub fn simulated_seconds(&self) -> Fx {
        self.step * i32::try_from(self.total_steps).unwrap_or(i32::MAX)
    }

    /// Clears the accumulator without touching the step counters.
    ///
    /// Call after a long blocking operation — loading a save, generating a
    /// level — so the time it took is not treated as simulation backlog.
    pub fn discard_backlog(&mut self) {
        self.accumulator = Fx::ZERO;
    }
}

impl Default for Timestep {
    fn default() -> Timestep {
        Timestep::standard()
    }
}

/// What a frame should do, as decided by the accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepPlan {
    /// Simulation steps to run this frame.
    pub steps: u32,
    /// Steps abandoned because the frame was too far behind.
    pub dropped: u32,
    /// How far between the previous and next simulation state to draw, `[0, 1)`.
    pub alpha: Fx,
}

impl StepPlan {
    /// True when the loop had to abandon work to keep up.
    #[must_use]
    pub const fn is_overloaded(&self) -> bool {
        self.dropped > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sixtieth of a second, the standard step.
    fn one_step() -> Fx {
        Fx::ONE / 60
    }

    #[test]
    fn a_frame_worth_of_time_runs_exactly_one_step() {
        let mut timestep = Timestep::standard();
        let plan = timestep.advance(one_step());
        assert_eq!(plan.steps, 1);
        assert_eq!(plan.dropped, 0);
    }

    #[test]
    fn a_short_frame_runs_no_steps_but_banks_the_time() {
        let mut timestep = Timestep::standard();
        // Half a step: nothing to run yet.
        assert_eq!(timestep.advance(one_step() / 2).steps, 0);
        // The other half completes it.
        assert_eq!(timestep.advance(one_step() / 2).steps, 1);
    }

    #[test]
    fn a_long_frame_catches_up_with_several_steps() {
        let mut timestep = Timestep::standard();
        let plan = timestep.advance(one_step() * 3);
        assert_eq!(plan.steps, 3);
        assert!(!plan.is_overloaded());
    }

    #[test]
    fn the_step_count_is_capped_and_the_excess_dropped() {
        let mut timestep = Timestep::standard();
        // A two-second stall: without a cap this would try to run 120 steps.
        let plan = timestep.advance(Fx::TWO);
        assert_eq!(plan.steps, 5, "the cap holds");
        assert!(plan.dropped > 100, "the rest is abandoned, not banked");
        assert!(plan.is_overloaded());

        // Crucially, the next frame starts fresh rather than still behind.
        let next = timestep.advance(one_step());
        assert_eq!(next.steps, 1);
        assert_eq!(next.dropped, 0);
    }

    #[test]
    fn a_backwards_clock_does_not_run_the_simulation_in_reverse() {
        let mut timestep = Timestep::standard();
        timestep.advance(one_step() / 2);
        let plan = timestep.advance(Fx::from_num(-5));
        assert_eq!(plan.steps, 0);
        // The banked half-step is still there, not negative.
        assert!(!timestep.alpha().is_negative());
        assert_eq!(timestep.advance(one_step() / 2).steps, 1);
    }

    #[test]
    fn alpha_stays_within_the_unit_interval() {
        let mut timestep = Timestep::standard();
        for tick in 0..500i32 {
            // Deliberately irregular frame times.
            let elapsed = one_step() * Fx::from_ratio(80 + (tick % 40), 100);
            let plan = timestep.advance(elapsed);
            assert!(
                plan.alpha >= Fx::ZERO && plan.alpha < Fx::ONE,
                "alpha escaped at tick {tick}: {}",
                plan.alpha.to_f64()
            );
        }
    }

    #[test]
    fn simulated_time_does_not_drift() {
        // A power-of-two rate and frame time, so both are exactly representable
        // in binary fixed point. With 1/60 and 1/600 the accumulated total
        // falls a fraction short of a second, which would measure the number
        // format rather than the accumulator.
        let mut timestep = Timestep::new(64);
        let eighth_step = Fx::ONE / 512;
        for _ in 0..512 {
            timestep.advance(eighth_step);
        }
        assert_eq!(timestep.total_steps(), 64);
        // Exactly one second, derived from the step count rather than summed
        // per frame, so no amount of play time accumulates error.
        assert_eq!(timestep.simulated_seconds(), Fx::ONE);
    }

    #[test]
    fn discarding_the_backlog_clears_pending_time() {
        let mut timestep = Timestep::standard();
        timestep.advance(one_step() * 3 / 2);
        assert!(timestep.alpha().is_positive());
        timestep.discard_backlog();
        assert_eq!(timestep.alpha(), Fx::ZERO);
        // Counters are untouched.
        assert_eq!(timestep.total_steps(), 1);
    }

    #[test]
    fn a_custom_rate_changes_the_step_length() {
        let timestep = Timestep::new(30);
        assert_eq!(timestep.step_seconds(), Fx::ONE / 30);
    }

    #[test]
    #[should_panic(expected = "rate must be positive")]
    fn a_zero_rate_is_rejected() {
        let _ = Timestep::new(0);
    }

    #[test]
    #[should_panic(expected = "at least one step")]
    fn a_zero_step_cap_is_rejected() {
        Timestep::standard().set_max_steps_per_frame(0);
    }

    #[test]
    fn determinism_the_same_frame_times_produce_the_same_steps() {
        let run = || {
            let mut timestep = Timestep::standard();
            let mut trace = Vec::new();
            for tick in 0..300i32 {
                let elapsed = one_step() * Fx::from_ratio(70 + (tick * 7) % 60, 100);
                let plan = timestep.advance(elapsed);
                trace.push((plan.steps, plan.dropped, plan.alpha.to_raw()));
            }
            trace
        };
        assert_eq!(run(), run());
    }
}
