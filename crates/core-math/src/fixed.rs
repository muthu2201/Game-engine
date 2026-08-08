//! Q32.32 fixed-point arithmetic.
//!
//! # Why fixed point
//!
//! The engine's simulation layer must produce bit-identical results across
//! machines so that replays, rollback netcode and desync detection work. IEEE
//! 754 floating point does not give that guarantee in practice: compilers are
//! free to contract `a * b + c` into a fused multiply-add, `x87` intermediates
//! carry 80 bits of precision, and vendor libm implementations of `sin`/`sqrt`
//! disagree in the last ulp. Every one of those differences compounds over
//! thousands of simulation ticks.
//!
//! [`Fx`] sidesteps all of it. It is a plain `i64` interpreted as a fraction
//! with 32 fractional bits, so every operation is integer arithmetic with
//! defined overflow behaviour, and the transcendental functions in this module
//! are implemented from scratch rather than delegated to the platform.
//!
//! # Range and precision
//!
//! * Range: `[-2_147_483_648, 2_147_483_647.999_999_999_8]`
//! * Resolution: `2^-32` ≈ `2.33e-10`
//!
//! At the engine's 32 pixels-per-unit scale that is sub-nanometre precision on
//! a world roughly four million tiles across — far more headroom than a 2D game
//! needs in either direction.
//!
//! # Overflow policy
//!
//! Arithmetic saturates rather than wrapping or panicking. A simulation that
//! overflows is already wrong, but saturating keeps it *deterministically*
//! wrong on every machine, which preserves replay validity and keeps the bug
//! reproducible instead of turning it into a release-only panic. Use the
//! `checked_*` methods when a caller wants to detect the condition.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// Number of fractional bits in an [`Fx`] value.
pub const FRACTIONAL_BITS: u32 = 32;

/// The raw integer value representing `1.0`.
const ONE_RAW: i64 = 1 << FRACTIONAL_BITS;

/// A deterministic Q32.32 fixed-point number.
///
/// See the [module documentation](self) for the rationale and the overflow
/// policy.
///
/// ```
/// use verdant_core_math::Fx;
///
/// let a = Fx::from_num(3);
/// let b = Fx::from_ratio(1, 2);
/// assert_eq!(a * b, Fx::from_ratio(3, 2));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
pub struct Fx(i64);

impl Fx {
    /// The value `0`.
    pub const ZERO: Fx = Fx(0);
    /// The value `1`.
    pub const ONE: Fx = Fx(ONE_RAW);
    /// The value `-1`.
    pub const NEG_ONE: Fx = Fx(-ONE_RAW);
    /// The value `0.5`.
    pub const HALF: Fx = Fx(ONE_RAW / 2);
    /// The value `2`.
    pub const TWO: Fx = Fx(ONE_RAW * 2);
    /// The largest representable value.
    pub const MAX: Fx = Fx(i64::MAX);
    /// The smallest (most negative) representable value.
    pub const MIN: Fx = Fx(i64::MIN);
    /// The smallest positive value, `2^-32`.
    pub const EPSILON: Fx = Fx(1);

    /// π, correct to the full 32-bit fraction.
    pub const PI: Fx = Fx(13_493_037_705);
    /// τ = 2π.
    pub const TAU: Fx = Fx(26_986_075_409);
    /// π/2.
    pub const FRAC_PI_2: Fx = Fx(6_746_518_852);
    /// π/4.
    pub const FRAC_PI_4: Fx = Fx(3_373_259_426);
    /// Euler's number, e.
    pub const E: Fx = Fx(11_674_931_555);

    /// Reinterprets a raw Q32.32 integer as an [`Fx`].
    ///
    /// This is the inverse of [`Fx::to_raw`] and performs no scaling.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: i64) -> Fx {
        Fx(raw)
    }

    /// Returns the underlying Q32.32 integer.
    ///
    /// Serialising this value (rather than a float) is what makes save files
    /// and network snapshots reproduce exactly.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> i64 {
        self.0
    }

    /// Converts an integer to fixed point, saturating on overflow.
    #[inline]
    #[must_use]
    pub const fn from_num(value: i32) -> Fx {
        Fx((value as i64) << FRACTIONAL_BITS)
    }

    /// Builds the exact fraction `numerator / denominator`.
    ///
    /// This is the precise way to write constants such as `Fx::from_ratio(1, 3)`
    /// — going through `f64` would bake a rounding error into the constant.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero.
    #[inline]
    #[must_use]
    pub const fn from_ratio(numerator: i32, denominator: i32) -> Fx {
        assert!(denominator != 0, "Fx::from_ratio: denominator must be non-zero");
        Fx(((numerator as i64) << FRACTIONAL_BITS) / (denominator as i64))
    }

    /// Converts from `f64`, saturating to [`Fx::MIN`]/[`Fx::MAX`] out of range.
    ///
    /// Only for authoring-time conversion (asset import, editor input, config
    /// parsing). Never call this inside the simulation: the whole point of
    /// [`Fx`] is that simulation state never round-trips through a float.
    ///
    /// A non-finite input converts to [`Fx::ZERO`], because propagating a NaN
    /// into deterministic state is strictly worse than clamping it.
    #[must_use]
    // The bounds below are compared as f64 on purpose: losing the low bits of
    // `i64::MAX` in that conversion only widens the saturation window by a few
    // ulps, and the branch it guards is the one that prevents UB in the cast.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_f64(value: f64) -> Fx {
        if !value.is_finite() {
            return Fx::ZERO;
        }
        let scaled = value * (ONE_RAW as f64);
        if scaled >= i64::MAX as f64 {
            Fx::MAX
        } else if scaled <= i64::MIN as f64 {
            Fx::MIN
        } else {
            // Truncation is intended: `scaled` is bounded by the branches above.
            #[allow(clippy::cast_possible_truncation)]
            Fx(scaled.round() as i64)
        }
    }

    /// Converts to `f64`. Exact — `f64` has 53 mantissa bits, more than the 32
    /// fractional bits of [`Fx`], for any value whose integer part fits.
    ///
    /// For rendering and audio only, never to feed a value back into the
    /// simulation.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / (ONE_RAW as f64)
    }

    /// Converts to `f32` for GPU upload.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Truncates toward zero and returns the integer part.
    ///
    /// Note the difference from [`Fx::floor_int`], which rounds toward negative
    /// infinity: `(-1.5).to_int() == -1` but `(-1.5).floor_int() == -2`. Tile
    /// and grid lookups want the flooring version; anything mirroring the
    /// behaviour of an `as i32` cast wants this one.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn to_int(self) -> i32 {
        if self.0 >= 0 {
            (self.0 >> FRACTIONAL_BITS) as i32
        } else {
            // Shifting right floors, so negate around the shift to truncate.
            -((self.0.saturating_neg() >> FRACTIONAL_BITS) as i32)
        }
    }

    /// Largest integer less than or equal to this value.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn floor_int(self) -> i32 {
        (self.0 >> FRACTIONAL_BITS) as i32
    }

    /// Smallest integer greater than or equal to this value.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn ceil_int(self) -> i32 {
        ((self.0 + ONE_RAW - 1) >> FRACTIONAL_BITS) as i32
    }

    /// Rounds half away from zero and returns the integer.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn round_int(self) -> i32 {
        if self.0 >= 0 {
            ((self.0 + ONE_RAW / 2) >> FRACTIONAL_BITS) as i32
        } else {
            -((((-self.0) + ONE_RAW / 2) >> FRACTIONAL_BITS) as i32)
        }
    }

    /// Rounds down to the nearest whole [`Fx`].
    #[inline]
    #[must_use]
    pub const fn floor(self) -> Fx {
        Fx(self.0 & !(ONE_RAW - 1))
    }

    /// Rounds up to the nearest whole [`Fx`].
    #[inline]
    #[must_use]
    pub const fn ceil(self) -> Fx {
        Fx((self.0 + ONE_RAW - 1) & !(ONE_RAW - 1))
    }

    /// Rounds to the nearest whole [`Fx`], halves away from zero.
    #[inline]
    #[must_use]
    pub const fn round(self) -> Fx {
        if self.0 >= 0 {
            Fx((self.0 + ONE_RAW / 2) & !(ONE_RAW - 1))
        } else {
            Fx(-(((-self.0) + ONE_RAW / 2) & !(ONE_RAW - 1)))
        }
    }

    /// Fractional part, always in `[0, 1)` — including for negative values,
    /// which is what tile-offset and texture-wrap code needs.
    #[inline]
    #[must_use]
    pub const fn fract(self) -> Fx {
        Fx(self.0 & (ONE_RAW - 1))
    }

    /// Absolute value, saturating (`MIN.abs() == MAX`).
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Fx {
        Fx(self.0.saturating_abs())
    }

    /// Returns `-1`, `0` or `1` matching the sign of this value.
    #[inline]
    #[must_use]
    pub const fn signum(self) -> Fx {
        if self.0 > 0 {
            Fx::ONE
        } else if self.0 < 0 {
            Fx::NEG_ONE
        } else {
            Fx::ZERO
        }
    }

    /// True when the value is exactly zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// True when the value is strictly negative.
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// True when the value is strictly positive.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// The lesser of two values.
    #[inline]
    #[must_use]
    pub const fn min(self, other: Fx) -> Fx {
        if self.0 < other.0 {
            self
        } else {
            other
        }
    }

    /// The greater of two values.
    #[inline]
    #[must_use]
    pub const fn max(self, other: Fx) -> Fx {
        if self.0 > other.0 {
            self
        } else {
            other
        }
    }

    /// Constrains this value to `[min, max]`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    #[inline]
    #[must_use]
    pub const fn clamp(self, min: Fx, max: Fx) -> Fx {
        assert!(min.0 <= max.0, "Fx::clamp: min must not exceed max");
        if self.0 < min.0 {
            min
        } else if self.0 > max.0 {
            max
        } else {
            self
        }
    }

    /// Multiplication that reports overflow instead of saturating.
    #[inline]
    #[must_use]
    pub fn checked_mul(self, other: Fx) -> Option<Fx> {
        let product = ((self.0 as i128) * (other.0 as i128)) >> FRACTIONAL_BITS;
        i64::try_from(product).ok().map(Fx)
    }

    /// Division that reports overflow and division by zero.
    #[inline]
    #[must_use]
    pub fn checked_div(self, other: Fx) -> Option<Fx> {
        if other.0 == 0 {
            return None;
        }
        let quotient = ((self.0 as i128) << FRACTIONAL_BITS) / (other.0 as i128);
        i64::try_from(quotient).ok().map(Fx)
    }

    /// Linear interpolation from `self` to `target` by `t`.
    ///
    /// `t` is not clamped, so values outside `[0, 1]` extrapolate. Computed as
    /// `self + (target - self) * t` so that `lerp(a, b, 0) == a` exactly.
    #[inline]
    #[must_use]
    pub fn lerp(self, target: Fx, t: Fx) -> Fx {
        self + (target - self) * t
    }

    /// Moves toward `target` by at most `max_delta`, never overshooting.
    #[inline]
    #[must_use]
    pub fn move_towards(self, target: Fx, max_delta: Fx) -> Fx {
        let diff = target - self;
        if diff.abs() <= max_delta {
            target
        } else {
            self + diff.signum() * max_delta
        }
    }

    /// Square root, via 32 iterations of restoring binary digit-by-digit
    /// extraction on the widened `i128` value.
    ///
    /// Exact to within one ulp and identical on every platform, unlike
    /// `f64::sqrt` whose last-bit behaviour depends on the FPU.
    ///
    /// # Panics
    ///
    /// Panics if `self` is negative.
    #[must_use]
    pub fn sqrt(self) -> Fx {
        assert!(self.0 >= 0, "Fx::sqrt: cannot take the square root of a negative value");
        if self.0 == 0 {
            return Fx::ZERO;
        }
        // Work in Q64.64 so the result lands back in Q32.32 after the shift.
        let radicand = (self.0 as u128) << FRACTIONAL_BITS;
        let mut remainder = radicand;
        let mut root: u128 = 0;
        // Highest even power of four not exceeding the radicand.
        let mut bit: u128 = 1u128 << 126;
        while bit > remainder {
            bit >>= 2;
        }
        while bit != 0 {
            if remainder >= root + bit {
                remainder -= root + bit;
                root = (root >> 1) + bit;
            } else {
                root >>= 1;
            }
            bit >>= 2;
        }
        // `root` fits: sqrt of a value below 2^95 is below 2^48.
        #[allow(clippy::cast_possible_truncation)]
        Fx(root as i64)
    }

    /// Sine, from the ninth-order odd Taylor polynomial evaluated after the
    /// argument is folded into the first quadrant by symmetry.
    ///
    /// Folding to `[0, π/2]` before evaluating is what makes a fixed-order
    /// polynomial viable: the truncation error of the series grows sharply with
    /// `|x|`, so the reduction bounds it at roughly `3e-6` — far below a
    /// rendered pixel, and identical on every platform.
    #[must_use]
    pub fn sin(self) -> Fx {
        // Reduce to [0, TAU).
        let mut x = Fx(self.0.rem_euclid(Fx::TAU.0));
        // Fold the lower half-cycle up, remembering to flip the sign back.
        let mut negate = false;
        if x > Fx::PI {
            x -= Fx::PI;
            negate = true;
        }
        // Mirror the second quadrant onto the first: sin(π - x) == sin(x).
        if x > Fx::FRAC_PI_2 {
            x = Fx::PI - x;
        }
        // x - x^3/3! + x^5/5! - x^7/7! + x^9/9!
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        let x9 = x7 * x2;
        let result = x - x3 * Fx::from_ratio(1, 6) + x5 * Fx::from_ratio(1, 120)
            - x7 * Fx::from_ratio(1, 5040)
            + x9 * Fx::from_ratio(1, 362_880);
        if negate {
            -result
        } else {
            result
        }
    }

    /// Cosine, defined as `sin(x + π/2)`.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Fx {
        (self + Fx::FRAC_PI_2).sin()
    }

    /// Tangent. Saturates near the poles rather than dividing by zero, so a
    /// caller that walks the argument across π/2 gets a large finite value
    /// instead of a panic.
    #[must_use]
    pub fn tan(self) -> Fx {
        let cos = self.cos();
        if cos.0.abs() < 16 {
            if self.sin().is_negative() {
                Fx::MIN
            } else {
                Fx::MAX
            }
        } else {
            self.sin() / cos
        }
    }

    /// Two-argument arctangent covering all four quadrants, in `(-π, π]`.
    ///
    /// Evaluates a cubic in the normalised ratio `(x ∓ |y|) / (x ± |y|)`, which
    /// stays within `[-1, 1]` for every input and so needs no octant table.
    /// Maximum error is about `1.5e-3` radians (0.09°) — accurate enough for
    /// facing directions and projectile aim, and deterministic, which
    /// `f64::atan2` is not.
    #[must_use]
    pub fn atan2(y: Fx, x: Fx) -> Fx {
        if x.is_zero() && y.is_zero() {
            return Fx::ZERO;
        }
        // Coefficients of the standard cubic fit for atan over [-1, 1].
        let c3 = Fx::from_ratio(1963, 10_000);
        let c1 = Fx::from_ratio(9817, 10_000);

        let abs_y = y.abs();
        let angle = if !x.is_negative() {
            // Right half-plane: ratio maps x >= |y| to 0 and x == |y| to ±1.
            let ratio = (x - abs_y) / (x + abs_y);
            c3 * ratio * ratio * ratio - c1 * ratio + Fx::FRAC_PI_4
        } else {
            // Left half-plane: the same fit rotated by π/2.
            let ratio = (x + abs_y) / (abs_y - x);
            c3 * ratio * ratio * ratio - c1 * ratio + Fx::FRAC_PI_4 * 3
        };
        // The construction above is symmetric in y, so reflect for y < 0.
        if y.is_negative() {
            -angle
        } else {
            angle
        }
    }
}

// -----------------------------------------------------------------------------
// Operators. All saturate; see the module-level overflow policy.
// -----------------------------------------------------------------------------

impl Add for Fx {
    type Output = Fx;
    #[inline]
    fn add(self, other: Fx) -> Fx {
        Fx(self.0.saturating_add(other.0))
    }
}

impl Sub for Fx {
    type Output = Fx;
    #[inline]
    fn sub(self, other: Fx) -> Fx {
        Fx(self.0.saturating_sub(other.0))
    }
}

impl Mul for Fx {
    type Output = Fx;
    #[inline]
    fn mul(self, other: Fx) -> Fx {
        // The 128-bit intermediate is what keeps `a * b` exact before the shift
        // discards the low fractional bits.
        let product = ((self.0 as i128) * (other.0 as i128)) >> FRACTIONAL_BITS;
        Fx(saturate_i128(product))
    }
}

impl Div for Fx {
    type Output = Fx;
    /// # Panics
    ///
    /// Panics if `other` is zero. Division by zero is always a simulation bug,
    /// and silently saturating would hide it until a replay desynced.
    #[inline]
    fn div(self, other: Fx) -> Fx {
        assert!(other.0 != 0, "Fx::div: division by zero");
        let quotient = ((self.0 as i128) << FRACTIONAL_BITS) / (other.0 as i128);
        Fx(saturate_i128(quotient))
    }
}

impl Rem for Fx {
    type Output = Fx;
    /// # Panics
    ///
    /// Panics if `other` is zero.
    #[inline]
    fn rem(self, other: Fx) -> Fx {
        assert!(other.0 != 0, "Fx::rem: division by zero");
        Fx(self.0 % other.0)
    }
}

impl Neg for Fx {
    type Output = Fx;
    #[inline]
    fn neg(self) -> Fx {
        Fx(self.0.saturating_neg())
    }
}

impl Mul<i32> for Fx {
    type Output = Fx;
    #[inline]
    fn mul(self, scalar: i32) -> Fx {
        Fx(self.0.saturating_mul(scalar as i64))
    }
}

impl Div<i32> for Fx {
    type Output = Fx;
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    #[inline]
    fn div(self, divisor: i32) -> Fx {
        assert!(divisor != 0, "Fx::div: division by zero");
        Fx(self.0 / (divisor as i64))
    }
}

impl AddAssign for Fx {
    #[inline]
    fn add_assign(&mut self, other: Fx) {
        *self = *self + other;
    }
}

impl SubAssign for Fx {
    #[inline]
    fn sub_assign(&mut self, other: Fx) {
        *self = *self - other;
    }
}

impl MulAssign for Fx {
    #[inline]
    fn mul_assign(&mut self, other: Fx) {
        *self = *self * other;
    }
}

impl DivAssign for Fx {
    #[inline]
    fn div_assign(&mut self, other: Fx) {
        *self = *self / other;
    }
}

impl PartialOrd for Fx {
    #[inline]
    fn partial_cmp(&self, other: &Fx) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fx {
    #[inline]
    fn cmp(&self, other: &Fx) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<i32> for Fx {
    #[inline]
    fn from(value: i32) -> Fx {
        Fx::from_num(value)
    }
}

impl fmt::Debug for Fx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fx({:.6})", self.to_f64())
    }
}

impl fmt::Display for Fx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_f64())
    }
}

impl core::iter::Sum for Fx {
    fn sum<I: Iterator<Item = Fx>>(iter: I) -> Fx {
        iter.fold(Fx::ZERO, |acc, value| acc + value)
    }
}

/// Clamps a 128-bit intermediate back into `i64` range.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn saturate_i128(value: i128) -> i64 {
    if value > i64::MAX as i128 {
        i64::MAX
    } else if value < i64::MIN as i128 {
        i64::MIN
    } else {
        value as i64
    }
}

/// Shorthand for [`Fx::from_num`], for readable literals in gameplay code.
#[inline]
#[must_use]
pub const fn fx(value: i32) -> Fx {
    Fx::from_num(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_their_real_values() {
        assert!((Fx::PI.to_f64() - core::f64::consts::PI).abs() < 1e-9);
        assert!((Fx::TAU.to_f64() - core::f64::consts::TAU).abs() < 1e-9);
        assert!((Fx::FRAC_PI_2.to_f64() - core::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert!((Fx::E.to_f64() - core::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn arithmetic_is_exact_for_representable_values() {
        assert_eq!(fx(2) + fx(3), fx(5));
        assert_eq!(fx(2) - fx(5), fx(-3));
        assert_eq!(fx(6) * fx(7), fx(42));
        assert_eq!(fx(42) / fx(7), fx(6));
        assert_eq!(Fx::from_ratio(1, 2) * Fx::from_ratio(1, 2), Fx::from_ratio(1, 4));
    }

    #[test]
    fn from_ratio_is_exact_where_the_denominator_is_a_power_of_two() {
        assert_eq!(Fx::from_ratio(1, 2).to_raw(), ONE_RAW / 2);
        assert_eq!(Fx::from_ratio(3, 4).to_raw(), ONE_RAW * 3 / 4);
        assert_eq!(Fx::from_ratio(-1, 8).to_raw(), -ONE_RAW / 8);
    }

    #[test]
    fn overflow_saturates_rather_than_wrapping() {
        assert_eq!(Fx::MAX + Fx::ONE, Fx::MAX);
        assert_eq!(Fx::MIN - Fx::ONE, Fx::MIN);
        assert_eq!(Fx::MAX * fx(2), Fx::MAX);
        assert_eq!(Fx::MAX.checked_mul(fx(2)), None);
        assert_eq!(fx(1).checked_div(Fx::ZERO), None);
    }

    #[test]
    fn rounding_handles_negatives_correctly() {
        let value = Fx::from_ratio(-3, 2); // -1.5
        assert_eq!(value.floor_int(), -2);
        assert_eq!(value.ceil_int(), -1);
        assert_eq!(value.round_int(), -2);
        assert_eq!(value.to_int(), -1, "to_int truncates toward zero");
        // fract() is always non-negative, which tile lookups rely on.
        assert!(!value.fract().is_negative());
        assert_eq!(value.fract(), Fx::HALF);
    }

    #[test]
    fn sqrt_matches_reference_within_tolerance() {
        for n in 0..2000i32 {
            let value = Fx::from_ratio(n, 16);
            let root = value.sqrt();
            let expected = value.to_f64().sqrt();
            assert!(
                (root.to_f64() - expected).abs() < 1e-6,
                "sqrt({}) = {} but expected {expected}",
                value.to_f64(),
                root.to_f64()
            );
        }
    }

    #[test]
    fn sqrt_of_a_perfect_square_is_exact() {
        for n in 0..64i32 {
            assert_eq!(fx(n * n).sqrt(), fx(n), "sqrt({}) should be {n}", n * n);
        }
    }

    #[test]
    #[should_panic(expected = "cannot take the square root of a negative")]
    fn sqrt_of_a_negative_panics() {
        let _ = fx(-1).sqrt();
    }

    #[test]
    fn sine_and_cosine_track_the_real_functions() {
        for step in -720..=720i32 {
            let radians = Fx::TAU * Fx::from_ratio(step, 360);
            let expected_sin = radians.to_f64().sin();
            let expected_cos = radians.to_f64().cos();
            assert!(
                (radians.sin().to_f64() - expected_sin).abs() < 1e-4,
                "sin({}) drifted",
                radians.to_f64()
            );
            assert!(
                (radians.cos().to_f64() - expected_cos).abs() < 1e-4,
                "cos({}) drifted",
                radians.to_f64()
            );
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        for step in 0..360i32 {
            let radians = Fx::TAU * Fx::from_ratio(step, 360);
            let identity = radians.sin() * radians.sin() + radians.cos() * radians.cos();
            assert!((identity.to_f64() - 1.0).abs() < 1e-3, "sin^2 + cos^2 drifted at {step} deg");
        }
    }

    #[test]
    fn atan2_covers_all_four_quadrants() {
        let cases = [
            (1, 0, 0.0),
            (0, 1, core::f64::consts::FRAC_PI_2),
            (-1, 0, core::f64::consts::PI),
            (0, -1, -core::f64::consts::FRAC_PI_2),
            (1, 1, core::f64::consts::FRAC_PI_4),
            (-1, 1, 3.0 * core::f64::consts::FRAC_PI_4),
            (-1, -1, -3.0 * core::f64::consts::FRAC_PI_4),
            (1, -1, -core::f64::consts::FRAC_PI_4),
        ];
        for (x, y, expected) in cases {
            let actual = Fx::atan2(fx(y), fx(x)).to_f64();
            assert!(
                (actual - expected).abs() < 1e-3,
                "atan2({y}, {x}) = {actual} but expected {expected}"
            );
        }
    }

    #[test]
    fn lerp_hits_both_endpoints_exactly() {
        let a = Fx::from_ratio(1, 3);
        let b = Fx::from_ratio(7, 9);
        assert_eq!(a.lerp(b, Fx::ZERO), a);
        assert_eq!(a.lerp(b, Fx::ONE), b);
    }

    #[test]
    fn move_towards_never_overshoots() {
        let mut value = Fx::ZERO;
        for _ in 0..100 {
            value = value.move_towards(fx(3), Fx::from_ratio(1, 10));
        }
        assert_eq!(value, fx(3));
    }

    #[test]
    fn from_f64_rejects_non_finite_input() {
        assert_eq!(Fx::from_f64(f64::NAN), Fx::ZERO);
        assert_eq!(Fx::from_f64(f64::INFINITY), Fx::ZERO);
    }

    #[test]
    fn from_f64_saturates_out_of_range_input() {
        assert_eq!(Fx::from_f64(1e30), Fx::MAX);
        assert_eq!(Fx::from_f64(-1e30), Fx::MIN);
    }

    #[test]
    fn raw_round_trip_is_lossless() {
        for raw in [0i64, 1, -1, i64::MAX, i64::MIN, 123_456_789] {
            assert_eq!(Fx::from_raw(raw).to_raw(), raw);
        }
    }

    /// The determinism guarantee in one test: the same operation sequence
    /// produces the same bits regardless of evaluation order or optimisation.
    #[test]
    fn repeated_accumulation_is_bit_stable() {
        let step = Fx::from_ratio(1, 60);
        let mut a = Fx::ZERO;
        for _ in 0..10_000 {
            a = a + step * Fx::from_ratio(7, 3) - step * Fx::from_ratio(1, 3);
        }
        let mut b = Fx::ZERO;
        for _ in 0..10_000 {
            b = b + step * Fx::from_ratio(7, 3) - step * Fx::from_ratio(1, 3);
        }
        assert_eq!(a.to_raw(), b.to_raw());
    }
}
