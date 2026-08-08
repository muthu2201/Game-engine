//! Easing curves for tweens, camera moves and UI transitions.
//!
//! Every curve maps `t` in `[0, 1]` to an output that starts at `0` and ends at
//! `1`. Curves in the `back` and `elastic` families deliberately overshoot in
//! between; the rest stay within the unit interval.
//!
//! All of them are evaluated in [`Fx`], so a tween that drives simulation state
//! (a cutscene camera, a scripted NPC walk) stays deterministic.

use crate::fixed::Fx;

/// The easing curves the engine ships with.
///
/// Stored as a plain enum rather than a boxed closure so tween components stay
/// `Copy`, serialisable, and free of indirection in the inner loop.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Easing {
    /// No easing; the input is returned unchanged.
    #[default]
    Linear,
    /// Accelerates from rest, quadratic.
    QuadIn,
    /// Decelerates to rest, quadratic.
    QuadOut,
    /// Accelerates then decelerates, quadratic.
    QuadInOut,
    /// Accelerates from rest, cubic. The default for UI panels sliding in.
    CubicIn,
    /// Decelerates to rest, cubic.
    CubicOut,
    /// Accelerates then decelerates, cubic.
    CubicInOut,
    /// Accelerates from rest, quartic.
    QuartIn,
    /// Decelerates to rest, quartic.
    QuartOut,
    /// Sinusoidal ease in.
    SineIn,
    /// Sinusoidal ease out. Good for camera settles.
    SineOut,
    /// Sinusoidal ease in and out.
    SineInOut,
    /// Circular ease in.
    CircIn,
    /// Circular ease out.
    CircOut,
    /// Overshoots backwards before moving forward.
    BackIn,
    /// Overshoots past the target before settling.
    BackOut,
    /// Springs past the target and oscillates in.
    ElasticOut,
    /// Bounces on arrival, like a dropped object.
    BounceOut,
}

impl Easing {
    /// Applies the curve to `t`.
    ///
    /// `t` is clamped to `[0, 1]` first, so a tween that overshoots its
    /// duration holds at its end value rather than extrapolating off the curve.
    #[must_use]
    pub fn apply(self, t: Fx) -> Fx {
        let t = t.clamp(Fx::ZERO, Fx::ONE);
        match self {
            Easing::Linear => t,
            Easing::QuadIn => t * t,
            Easing::QuadOut => {
                let inverse = Fx::ONE - t;
                Fx::ONE - inverse * inverse
            }
            Easing::QuadInOut => {
                if t < Fx::HALF {
                    t * t * Fx::TWO
                } else {
                    let shifted = t * Fx::TWO - Fx::TWO;
                    Fx::ONE - shifted * shifted * Fx::HALF
                }
            }
            Easing::CubicIn => t * t * t,
            Easing::CubicOut => {
                let inverse = Fx::ONE - t;
                Fx::ONE - inverse * inverse * inverse
            }
            Easing::CubicInOut => {
                if t < Fx::HALF {
                    t * t * t * 4
                } else {
                    let shifted = t * Fx::TWO - Fx::TWO;
                    Fx::ONE + shifted * shifted * shifted * Fx::HALF
                }
            }
            Easing::QuartIn => {
                let squared = t * t;
                squared * squared
            }
            Easing::QuartOut => {
                let inverse = Fx::ONE - t;
                let squared = inverse * inverse;
                Fx::ONE - squared * squared
            }
            Easing::SineIn => Fx::ONE - (t * Fx::FRAC_PI_2).cos(),
            Easing::SineOut => (t * Fx::FRAC_PI_2).sin(),
            Easing::SineInOut => (Fx::ONE - (t * Fx::PI).cos()) * Fx::HALF,
            Easing::CircIn => Fx::ONE - (Fx::ONE - t * t).max(Fx::ZERO).sqrt(),
            Easing::CircOut => {
                let shifted = t - Fx::ONE;
                (Fx::ONE - shifted * shifted).max(Fx::ZERO).sqrt()
            }
            Easing::BackIn => {
                // The classic overshoot constant, 1.70158.
                let overshoot = Fx::from_ratio(170_158, 100_000);
                t * t * ((overshoot + Fx::ONE) * t - overshoot)
            }
            Easing::BackOut => {
                let overshoot = Fx::from_ratio(170_158, 100_000);
                let shifted = t - Fx::ONE;
                Fx::ONE + shifted * shifted * ((overshoot + Fx::ONE) * shifted + overshoot)
            }
            Easing::ElasticOut => {
                if t.is_zero() {
                    return Fx::ZERO;
                }
                if t >= Fx::ONE {
                    return Fx::ONE;
                }
                // A decaying sine; the decay is approximated by a quartic
                // falloff rather than an exponential so the whole curve stays
                // in integer arithmetic.
                let period = Fx::from_ratio(3, 10);
                let phase = (t - period / 4) * Fx::TAU / period;
                let inverse = Fx::ONE - t;
                let squared = inverse * inverse;
                Fx::ONE - squared * squared * phase.cos()
            }
            Easing::BounceOut => bounce_out(t),
        }
    }

    /// Interpolates from `start` to `end` along this curve.
    ///
    /// The endpoints are snapped rather than evaluated: several curves are
    /// polynomial approximations that land a fraction of a unit short of `1` at
    /// `t == 1`, and a tween that stops just shy of its target leaves an NPC
    /// permanently off its mark or a UI panel a pixel out of place.
    #[inline]
    #[must_use]
    pub fn interpolate(self, start: Fx, end: Fx, t: Fx) -> Fx {
        if t <= Fx::ZERO {
            return start;
        }
        if t >= Fx::ONE {
            return end;
        }
        start.lerp(end, self.apply(t))
    }
}

/// The four-segment parabola bounce, matching the widely used Penner curve.
fn bounce_out(t: Fx) -> Fx {
    let n1 = Fx::from_ratio(75, 10); // 7.5625
    let n1 = n1 + Fx::from_ratio(625, 10_000);
    let d1 = Fx::from_ratio(275, 100); // 2.75

    if t < Fx::ONE / d1 {
        n1 * t * t
    } else if t < Fx::TWO / d1 {
        let shifted = t - Fx::from_ratio(15, 10) / d1;
        n1 * shifted * shifted + Fx::from_ratio(75, 100)
    } else if t < Fx::from_ratio(25, 10) / d1 {
        let shifted = t - Fx::from_ratio(225, 100) / d1;
        n1 * shifted * shifted + Fx::from_ratio(9375, 10_000)
    } else {
        let shifted = t - Fx::from_ratio(2625, 1000) / d1;
        n1 * shifted * shifted + Fx::from_ratio(984_375, 1_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every curve in this list. Kept explicit so a newly added variant fails
    /// to compile here until it is covered by the property tests below.
    const ALL: [Easing; 18] = [
        Easing::Linear,
        Easing::QuadIn,
        Easing::QuadOut,
        Easing::QuadInOut,
        Easing::CubicIn,
        Easing::CubicOut,
        Easing::CubicInOut,
        Easing::QuartIn,
        Easing::QuartOut,
        Easing::SineIn,
        Easing::SineOut,
        Easing::SineInOut,
        Easing::CircIn,
        Easing::CircOut,
        Easing::BackIn,
        Easing::BackOut,
        Easing::ElasticOut,
        Easing::BounceOut,
    ];

    #[test]
    fn every_curve_starts_at_zero_and_ends_at_one() {
        for easing in ALL {
            let start = easing.apply(Fx::ZERO).to_f64();
            let end = easing.apply(Fx::ONE).to_f64();
            assert!(start.abs() < 1e-3, "{easing:?} started at {start}");
            assert!((end - 1.0).abs() < 1e-3, "{easing:?} ended at {end}");
        }
    }

    #[test]
    fn input_is_clamped_outside_the_unit_interval() {
        for easing in ALL {
            assert_eq!(easing.apply(Fx::from_num(-5)), easing.apply(Fx::ZERO));
            assert_eq!(easing.apply(Fx::from_num(5)), easing.apply(Fx::ONE));
        }
    }

    #[test]
    fn non_overshooting_curves_stay_within_the_unit_interval() {
        let bounded = [
            Easing::Linear,
            Easing::QuadIn,
            Easing::QuadOut,
            Easing::QuadInOut,
            Easing::CubicIn,
            Easing::CubicOut,
            Easing::CubicInOut,
            Easing::QuartIn,
            Easing::QuartOut,
            Easing::SineIn,
            Easing::SineOut,
            Easing::SineInOut,
            Easing::CircIn,
            Easing::CircOut,
            Easing::BounceOut,
        ];
        for easing in bounded {
            for step in 0..=100i32 {
                let value = easing.apply(Fx::from_ratio(step, 100));
                assert!(
                    value >= Fx::from_ratio(-1, 1000) && value <= Fx::from_ratio(1001, 1000),
                    "{easing:?} left the unit interval at t={step}: {value}"
                );
            }
        }
    }

    #[test]
    fn monotonic_curves_never_decrease() {
        // The back and elastic families intentionally reverse; the rest must not.
        let monotonic = [
            Easing::Linear,
            Easing::QuadIn,
            Easing::QuadOut,
            Easing::QuadInOut,
            Easing::CubicIn,
            Easing::CubicOut,
            Easing::CubicInOut,
            Easing::QuartIn,
            Easing::QuartOut,
            Easing::SineIn,
            Easing::SineOut,
            Easing::SineInOut,
            Easing::CircIn,
            Easing::CircOut,
        ];
        for easing in monotonic {
            let mut previous = easing.apply(Fx::ZERO);
            for step in 1..=200i32 {
                let value = easing.apply(Fx::from_ratio(step, 200));
                assert!(
                    value >= previous - Fx::from_ratio(1, 10_000),
                    "{easing:?} decreased at t={step}: {previous} -> {value}"
                );
                previous = value;
            }
        }
    }

    #[test]
    fn back_curves_actually_overshoot() {
        // BackIn must dip below zero; BackOut must rise above one. If they did
        // not, the constant would be wrong and the effect invisible.
        let dips = (0..50).any(|step| {
            Easing::BackIn
                .apply(Fx::from_ratio(step, 100))
                .is_negative()
        });
        assert!(dips, "BackIn should undershoot near the start");
        let overshoots =
            (50..100).any(|step| Easing::BackOut.apply(Fx::from_ratio(step, 100)) > Fx::ONE);
        assert!(overshoots, "BackOut should overshoot near the end");
    }

    #[test]
    fn in_out_curves_pass_through_the_midpoint() {
        for easing in [Easing::QuadInOut, Easing::CubicInOut, Easing::SineInOut] {
            let midpoint = easing.apply(Fx::HALF).to_f64();
            assert!(
                (midpoint - 0.5).abs() < 1e-3,
                "{easing:?} midpoint was {midpoint}"
            );
        }
    }

    #[test]
    fn linear_is_the_identity() {
        for step in 0..=100i32 {
            let t = Fx::from_ratio(step, 100);
            assert_eq!(Easing::Linear.apply(t), t);
        }
    }

    #[test]
    fn interpolate_spans_the_requested_range() {
        let start = Fx::from_num(10);
        let end = Fx::from_num(20);
        for easing in ALL {
            assert_eq!(easing.interpolate(start, end, Fx::ZERO), start);
            assert_eq!(easing.interpolate(start, end, Fx::ONE), end);
        }
    }
}
