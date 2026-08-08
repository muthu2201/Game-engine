//! Deterministic 2D vector built on [`Fx`].

use crate::fixed::Fx;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector with fixed-point components.
///
/// Every simulation position, velocity and offset in the engine uses this type.
/// Because [`Fx`] is deterministic, so is every operation here — including
/// [`Vec2::length`] and [`Vec2::normalize`], which route through
/// [`Fx::sqrt`] rather than the platform's libm.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec2 {
    /// Horizontal component; positive is right.
    pub x: Fx,
    /// Vertical component; positive is down, matching screen and tile space.
    pub y: Fx,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Vec2 = Vec2 {
        x: Fx::ZERO,
        y: Fx::ZERO,
    };
    /// `(1, 1)`.
    pub const ONE: Vec2 = Vec2 {
        x: Fx::ONE,
        y: Fx::ONE,
    };
    /// `(1, 0)`.
    pub const X: Vec2 = Vec2 {
        x: Fx::ONE,
        y: Fx::ZERO,
    };
    /// `(0, 1)`.
    pub const Y: Vec2 = Vec2 {
        x: Fx::ZERO,
        y: Fx::ONE,
    };
    /// Screen-space up, `(0, -1)`.
    pub const UP: Vec2 = Vec2 {
        x: Fx::ZERO,
        y: Fx::NEG_ONE,
    };
    /// Screen-space down, `(0, 1)`.
    pub const DOWN: Vec2 = Vec2 {
        x: Fx::ZERO,
        y: Fx::ONE,
    };
    /// `(-1, 0)`.
    pub const LEFT: Vec2 = Vec2 {
        x: Fx::NEG_ONE,
        y: Fx::ZERO,
    };
    /// `(1, 0)`.
    pub const RIGHT: Vec2 = Vec2 {
        x: Fx::ONE,
        y: Fx::ZERO,
    };

    /// Builds a vector from two fixed-point components.
    #[inline]
    #[must_use]
    pub const fn new(x: Fx, y: Fx) -> Vec2 {
        Vec2 { x, y }
    }

    /// Builds a vector from two integers.
    #[inline]
    #[must_use]
    pub const fn from_ints(x: i32, y: i32) -> Vec2 {
        Vec2 {
            x: Fx::from_num(x),
            y: Fx::from_num(y),
        }
    }

    /// A vector with both components set to `value`.
    #[inline]
    #[must_use]
    pub const fn splat(value: Fx) -> Vec2 {
        Vec2 { x: value, y: value }
    }

    /// The unit vector at `angle` radians, scaled by `length`.
    #[inline]
    #[must_use]
    pub fn from_angle(angle: Fx, length: Fx) -> Vec2 {
        Vec2 {
            x: angle.cos() * length,
            y: angle.sin() * length,
        }
    }

    /// Dot product.
    #[inline]
    #[must_use]
    pub fn dot(self, other: Vec2) -> Fx {
        self.x * other.x + self.y * other.y
    }

    /// The z component of the 3D cross product.
    ///
    /// Its sign says which side of `self` the vector `other` lies on, which is
    /// how the collision code chooses a separation direction.
    #[inline]
    #[must_use]
    pub fn cross(self, other: Vec2) -> Fx {
        self.x * other.y - self.y * other.x
    }

    /// Squared length. Prefer this for comparisons — it avoids the square root.
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> Fx {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length, via the deterministic [`Fx::sqrt`].
    #[inline]
    #[must_use]
    pub fn length(self) -> Fx {
        self.length_squared().sqrt()
    }

    /// Distance to `other`.
    #[inline]
    #[must_use]
    pub fn distance(self, other: Vec2) -> Fx {
        (other - self).length()
    }

    /// Squared distance to `other`. Prefer this for radius checks.
    #[inline]
    #[must_use]
    pub fn distance_squared(self, other: Vec2) -> Fx {
        (other - self).length_squared()
    }

    /// Manhattan (taxicab) distance, used by the grid pathfinder's heuristic.
    #[inline]
    #[must_use]
    pub fn manhattan_distance(self, other: Vec2) -> Fx {
        (other.x - self.x).abs() + (other.y - self.y).abs()
    }

    /// Returns this vector scaled to unit length.
    ///
    /// A zero vector is returned unchanged rather than producing a division by
    /// zero — callers that need to distinguish the case should check
    /// [`Vec2::is_zero`] first.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Vec2 {
        let length = self.length();
        if length.is_zero() {
            Vec2::ZERO
        } else {
            Vec2 {
                x: self.x / length,
                y: self.y / length,
            }
        }
    }

    /// Returns this vector with its length capped at `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length(self, max: Fx) -> Vec2 {
        let length_sq = self.length_squared();
        if length_sq > max * max {
            self.normalize() * max
        } else {
            self
        }
    }

    /// Rotates by `angle` radians (clockwise in screen space, where y is down).
    #[inline]
    #[must_use]
    pub fn rotate(self, angle: Fx) -> Vec2 {
        let (sin, cos) = (angle.sin(), angle.cos());
        Vec2 {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }

    /// The vector rotated 90° clockwise: `(x, y) -> (-y, x)`.
    #[inline]
    #[must_use]
    pub fn perpendicular(self) -> Vec2 {
        Vec2 {
            x: -self.y,
            y: self.x,
        }
    }

    /// Angle of this vector in radians, measured from the +X axis.
    #[inline]
    #[must_use]
    pub fn angle(self) -> Fx {
        Fx::atan2(self.y, self.x)
    }

    /// Component-wise absolute value.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Vec2 {
        Vec2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Component-wise floor.
    #[inline]
    #[must_use]
    pub fn floor(self) -> Vec2 {
        Vec2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Component-wise round.
    #[inline]
    #[must_use]
    pub fn round(self) -> Vec2 {
        Vec2 {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Component-wise minimum.
    #[inline]
    #[must_use]
    pub fn min(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Component-wise maximum.
    #[inline]
    #[must_use]
    pub fn max(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Component-wise multiplication.
    #[inline]
    #[must_use]
    pub fn mul_components(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    /// Linear interpolation toward `target`.
    #[inline]
    #[must_use]
    pub fn lerp(self, target: Vec2, t: Fx) -> Vec2 {
        Vec2 {
            x: self.x.lerp(target.x, t),
            y: self.y.lerp(target.y, t),
        }
    }

    /// Moves toward `target` by at most `max_delta`, never overshooting.
    #[must_use]
    pub fn move_towards(self, target: Vec2, max_delta: Fx) -> Vec2 {
        let delta = target - self;
        let distance = delta.length();
        if distance <= max_delta || distance.is_zero() {
            target
        } else {
            self + delta / distance * max_delta
        }
    }

    /// Reflects this vector about a surface with the given unit `normal`.
    #[inline]
    #[must_use]
    pub fn reflect(self, normal: Vec2) -> Vec2 {
        self - normal * (self.dot(normal) * Fx::TWO)
    }

    /// True when both components are exactly zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// The integer tile coordinate containing this position, given `tile_size`
    /// world units per tile.
    ///
    /// Uses flooring rather than truncation so that negative coordinates map to
    /// the tile they are actually inside.
    ///
    /// # Panics
    ///
    /// Panics if `tile_size` is zero.
    #[inline]
    #[must_use]
    pub fn to_tile(self, tile_size: Fx) -> (i32, i32) {
        (
            (self.x / tile_size).floor_int(),
            (self.y / tile_size).floor_int(),
        )
    }

    /// Converts to a pair of `f32` for GPU upload.
    #[inline]
    #[must_use]
    pub fn to_f32_array(self) -> [f32; 2] {
        [self.x.to_f32(), self.y.to_f32()]
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<Fx> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, scalar: Fx) -> Vec2 {
        Vec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Mul<i32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, scalar: i32) -> Vec2 {
        Vec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Div<Fx> for Vec2 {
    type Output = Vec2;
    /// # Panics
    ///
    /// Panics if `scalar` is zero.
    #[inline]
    fn div(self, scalar: Fx) -> Vec2 {
        Vec2 {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    #[inline]
    fn neg(self) -> Vec2 {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, other: Vec2) {
        *self = *self + other;
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, other: Vec2) {
        *self = *self - other;
    }
}

impl MulAssign<Fx> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, scalar: Fx) {
        *self = *self * scalar;
    }
}

impl fmt::Debug for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({:.4}, {:.4})", self.x.to_f64(), self.y.to_f64())
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x.to_f64(), self.y.to_f64())
    }
}

/// An integer 2D coordinate, used for tile and grid addressing.
///
/// Kept separate from [`Vec2`] so the type system distinguishes "a point in the
/// continuous world" from "a cell in the tile grid" — mixing those up is the
/// most common source of off-by-one bugs in tile-based games.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IVec2 {
    /// Column index.
    pub x: i32,
    /// Row index.
    pub y: i32,
}

impl IVec2 {
    /// The origin cell.
    pub const ZERO: IVec2 = IVec2 { x: 0, y: 0 };
    /// `(1, 1)`.
    pub const ONE: IVec2 = IVec2 { x: 1, y: 1 };

    /// The four orthogonal neighbour offsets, in a fixed order.
    ///
    /// Iteration order is part of the determinism contract: pathfinding expands
    /// neighbours in this order, so changing it changes generated paths.
    pub const CARDINALS: [IVec2; 4] = [
        IVec2 { x: 0, y: -1 },
        IVec2 { x: 1, y: 0 },
        IVec2 { x: 0, y: 1 },
        IVec2 { x: -1, y: 0 },
    ];

    /// The eight neighbour offsets including diagonals, in a fixed order.
    pub const NEIGHBOURS: [IVec2; 8] = [
        IVec2 { x: 0, y: -1 },
        IVec2 { x: 1, y: -1 },
        IVec2 { x: 1, y: 0 },
        IVec2 { x: 1, y: 1 },
        IVec2 { x: 0, y: 1 },
        IVec2 { x: -1, y: 1 },
        IVec2 { x: -1, y: 0 },
        IVec2 { x: -1, y: -1 },
    ];

    /// Builds a grid coordinate.
    #[inline]
    #[must_use]
    pub const fn new(x: i32, y: i32) -> IVec2 {
        IVec2 { x, y }
    }

    /// Converts to a world position at the cell's top-left corner.
    #[inline]
    #[must_use]
    pub fn to_world(self, tile_size: Fx) -> Vec2 {
        Vec2::new(
            Fx::from_num(self.x) * tile_size,
            Fx::from_num(self.y) * tile_size,
        )
    }

    /// Converts to the world position at the cell's centre.
    #[inline]
    #[must_use]
    pub fn to_world_centre(self, tile_size: Fx) -> Vec2 {
        self.to_world(tile_size) + Vec2::splat(tile_size * Fx::HALF)
    }

    /// Manhattan distance to `other`.
    #[inline]
    #[must_use]
    pub fn manhattan_distance(self, other: IVec2) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    /// Chebyshev (king-move) distance to `other`.
    #[inline]
    #[must_use]
    pub fn chebyshev_distance(self, other: IVec2) -> i32 {
        (self.x - other.x).abs().max((self.y - other.y).abs())
    }
}

impl Add for IVec2 {
    type Output = IVec2;
    #[inline]
    fn add(self, other: IVec2) -> IVec2 {
        IVec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for IVec2 {
    type Output = IVec2;
    #[inline]
    fn sub(self, other: IVec2) -> IVec2 {
        IVec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<i32> for IVec2 {
    type Output = IVec2;
    #[inline]
    fn mul(self, scalar: i32) -> IVec2 {
        IVec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::fx;

    #[test]
    fn length_of_a_three_four_five_triangle_is_exact() {
        assert_eq!(Vec2::from_ints(3, 4).length(), fx(5));
        assert_eq!(Vec2::from_ints(-3, -4).length(), fx(5));
    }

    #[test]
    fn normalize_produces_unit_length() {
        for (x, y) in [(3, 4), (1, 0), (0, -7), (-5, 12), (8, 15)] {
            let normalized = Vec2::from_ints(x, y).normalize();
            let length = normalized.length().to_f64();
            assert!(
                (length - 1.0).abs() < 1e-6,
                "({x}, {y}) normalized to length {length}"
            );
        }
    }

    #[test]
    fn normalize_of_zero_is_zero_not_a_division_by_zero() {
        assert_eq!(Vec2::ZERO.normalize(), Vec2::ZERO);
    }

    #[test]
    fn dot_and_cross_agree_with_geometry() {
        let a = Vec2::from_ints(1, 0);
        let b = Vec2::from_ints(0, 1);
        assert_eq!(
            a.dot(b),
            Fx::ZERO,
            "perpendicular vectors have zero dot product"
        );
        assert_eq!(a.cross(b), Fx::ONE);
        assert_eq!(b.cross(a), Fx::NEG_ONE);
    }

    #[test]
    fn perpendicular_is_orthogonal_and_preserves_length() {
        let v = Vec2::from_ints(3, 4);
        let p = v.perpendicular();
        assert_eq!(v.dot(p), Fx::ZERO);
        assert_eq!(p.length(), v.length());
    }

    #[test]
    fn rotating_by_a_full_turn_returns_close_to_the_original() {
        let v = Vec2::from_ints(10, 0);
        let rotated = v.rotate(Fx::TAU);
        assert!((rotated.x.to_f64() - 10.0).abs() < 1e-3);
        assert!(rotated.y.to_f64().abs() < 1e-3);
    }

    #[test]
    fn rotating_by_a_quarter_turn_maps_x_to_y() {
        let rotated = Vec2::X.rotate(Fx::FRAC_PI_2);
        assert!(rotated.x.to_f64().abs() < 1e-4);
        assert!((rotated.y.to_f64() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn clamp_length_caps_long_vectors_and_leaves_short_ones() {
        let long = Vec2::from_ints(30, 40).clamp_length(fx(5));
        assert!((long.length().to_f64() - 5.0).abs() < 1e-5);
        let short = Vec2::from_ints(1, 0);
        assert_eq!(short.clamp_length(fx(5)), short);
    }

    #[test]
    fn move_towards_reaches_the_target_without_overshooting() {
        let mut position = Vec2::ZERO;
        let target = Vec2::from_ints(3, 4);
        for _ in 0..100 {
            position = position.move_towards(target, Fx::from_ratio(1, 10));
        }
        assert_eq!(position, target);
    }

    #[test]
    fn reflect_bounces_off_a_flat_floor() {
        // Travelling down-right, hitting a floor whose normal points up.
        let velocity = Vec2::new(fx(1), fx(1));
        let reflected = velocity.reflect(Vec2::UP);
        assert_eq!(reflected.x, fx(1), "tangential component is preserved");
        assert_eq!(reflected.y, fx(-1), "normal component is inverted");
    }

    #[test]
    fn to_tile_floors_negative_coordinates() {
        let tile = fx(16);
        assert_eq!(Vec2::from_ints(0, 0).to_tile(tile), (0, 0));
        assert_eq!(Vec2::from_ints(15, 15).to_tile(tile), (0, 0));
        assert_eq!(Vec2::from_ints(16, 32).to_tile(tile), (1, 2));
        assert_eq!(Vec2::from_ints(-1, -1).to_tile(tile), (-1, -1));
        assert_eq!(Vec2::from_ints(-16, -17).to_tile(tile), (-1, -2));
    }

    #[test]
    fn ivec2_neighbour_tables_are_the_documented_length() {
        assert_eq!(IVec2::CARDINALS.len(), 4);
        assert_eq!(IVec2::NEIGHBOURS.len(), 8);
        // Every cardinal must also appear in the eight-way table.
        for cardinal in IVec2::CARDINALS {
            assert!(IVec2::NEIGHBOURS.contains(&cardinal));
        }
    }

    #[test]
    fn ivec2_world_conversion_round_trips() {
        let tile = fx(16);
        for (x, y) in [(0, 0), (3, 7), (-2, -5)] {
            let cell = IVec2::new(x, y);
            assert_eq!(cell.to_world(tile).to_tile(tile), (x, y));
            assert_eq!(cell.to_world_centre(tile).to_tile(tile), (x, y));
        }
    }
}
