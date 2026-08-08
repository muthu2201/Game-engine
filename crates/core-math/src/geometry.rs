//! Axis-aligned rectangles, circles and the intersection tests the physics and
//! culling layers are built from.

use crate::fixed::Fx;
use crate::vec2::Vec2;

/// An axis-aligned rectangle stored as a minimum corner plus a size.
///
/// Min/size rather than min/max because the physics broadphase reads the size
/// far more often than the far corner, and because a negative size is then
/// trivially detectable as invalid.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    /// Top-left corner (minimum on both axes).
    pub min: Vec2,
    /// Extent along each axis. Never negative for a valid rectangle.
    pub size: Vec2,
}

impl Rect {
    /// The degenerate rectangle at the origin.
    pub const ZERO: Rect = Rect { min: Vec2::ZERO, size: Vec2::ZERO };

    /// Builds a rectangle from its top-left corner and size.
    #[inline]
    #[must_use]
    pub const fn new(min: Vec2, size: Vec2) -> Rect {
        Rect { min, size }
    }

    /// Builds a rectangle from two opposite corners in any order.
    #[inline]
    #[must_use]
    pub fn from_corners(a: Vec2, b: Vec2) -> Rect {
        let min = a.min(b);
        Rect { min, size: a.max(b) - min }
    }

    /// Builds a rectangle centred on `centre`.
    #[inline]
    #[must_use]
    pub fn from_centre(centre: Vec2, size: Vec2) -> Rect {
        Rect { min: centre - size * Fx::HALF, size }
    }

    /// Builds a rectangle from integer components, the common case for tile and
    /// sprite bounds.
    #[inline]
    #[must_use]
    pub const fn from_ints(x: i32, y: i32, width: i32, height: i32) -> Rect {
        Rect { min: Vec2::from_ints(x, y), size: Vec2::from_ints(width, height) }
    }

    /// The maximum corner (`min + size`).
    #[inline]
    #[must_use]
    pub fn max(self) -> Vec2 {
        self.min + self.size
    }

    /// The centre point.
    #[inline]
    #[must_use]
    pub fn centre(self) -> Vec2 {
        self.min + self.size * Fx::HALF
    }

    /// Left edge (minimum x).
    #[inline]
    #[must_use]
    pub fn left(self) -> Fx {
        self.min.x
    }

    /// Right edge (maximum x).
    #[inline]
    #[must_use]
    pub fn right(self) -> Fx {
        self.min.x + self.size.x
    }

    /// Top edge (minimum y — y grows downward).
    #[inline]
    #[must_use]
    pub fn top(self) -> Fx {
        self.min.y
    }

    /// Bottom edge (maximum y).
    #[inline]
    #[must_use]
    pub fn bottom(self) -> Fx {
        self.min.y + self.size.y
    }

    /// Area of the rectangle.
    #[inline]
    #[must_use]
    pub fn area(self) -> Fx {
        self.size.x * self.size.y
    }

    /// True when either dimension is zero or negative.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        !self.size.x.is_positive() || !self.size.y.is_positive()
    }

    /// True when `point` lies inside, treating the min edges as inclusive and
    /// the max edges as exclusive.
    ///
    /// Half-open bounds mean adjacent tiles tile the plane without a point ever
    /// belonging to two of them.
    #[inline]
    #[must_use]
    pub fn contains_point(self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x < self.right()
            && point.y >= self.min.y
            && point.y < self.bottom()
    }

    /// True when `other` lies entirely within this rectangle.
    #[inline]
    #[must_use]
    pub fn contains_rect(self, other: Rect) -> bool {
        other.min.x >= self.min.x
            && other.min.y >= self.min.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// True when the two rectangles overlap by a non-zero area.
    ///
    /// Rectangles that merely touch along an edge do **not** count as
    /// overlapping, which is what keeps a body resting exactly on a floor from
    /// registering a collision every tick.
    #[inline]
    #[must_use]
    pub fn intersects(self, other: Rect) -> bool {
        self.min.x < other.right()
            && self.right() > other.min.x
            && self.min.y < other.bottom()
            && self.bottom() > other.min.y
    }

    /// The overlapping region, or `None` when the rectangles are disjoint.
    #[must_use]
    pub fn intersection(self, other: Rect) -> Option<Rect> {
        let min = self.min.max(other.min);
        let max = self.max().min(other.max());
        if min.x < max.x && min.y < max.y {
            Some(Rect { min, size: max - min })
        } else {
            None
        }
    }

    /// The smallest rectangle containing both inputs.
    #[must_use]
    pub fn union(self, other: Rect) -> Rect {
        let min = self.min.min(other.min);
        let max = self.max().max(other.max());
        Rect { min, size: max - min }
    }

    /// Grows the rectangle by `amount` on every side. A negative amount shrinks
    /// it, potentially to an empty rectangle.
    #[inline]
    #[must_use]
    pub fn expand(self, amount: Fx) -> Rect {
        Rect { min: self.min - Vec2::splat(amount), size: self.size + Vec2::splat(amount * Fx::TWO) }
    }

    /// Returns the rectangle translated by `offset`.
    #[inline]
    #[must_use]
    pub fn translate(self, offset: Vec2) -> Rect {
        Rect { min: self.min + offset, size: self.size }
    }

    /// Returns `point` clamped to lie within the rectangle.
    #[inline]
    #[must_use]
    pub fn clamp_point(self, point: Vec2) -> Vec2 {
        Vec2::new(
            point.x.clamp(self.min.x, self.right()),
            point.y.clamp(self.min.y, self.bottom()),
        )
    }

    /// The minimum translation that pushes `other` out of `self`, or `None`
    /// when they do not overlap.
    ///
    /// The returned vector always acts along the axis of least penetration,
    /// which is what makes a body slide along a wall rather than pop around a
    /// corner.
    #[must_use]
    pub fn penetration_vector(self, other: Rect) -> Option<Vec2> {
        let overlap = self.intersection(other)?;
        if overlap.size.x < overlap.size.y {
            // Push horizontally, in whichever direction is shorter.
            let sign = if other.centre().x < self.centre().x { Fx::NEG_ONE } else { Fx::ONE };
            Some(Vec2::new(overlap.size.x * sign, Fx::ZERO))
        } else {
            let sign = if other.centre().y < self.centre().y { Fx::NEG_ONE } else { Fx::ONE };
            Some(Vec2::new(Fx::ZERO, overlap.size.y * sign))
        }
    }
}

/// A circle, used for interaction radii, light volumes and audio falloff.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Circle {
    /// Centre point.
    pub centre: Vec2,
    /// Radius. Never negative for a valid circle.
    pub radius: Fx,
}

impl Circle {
    /// Builds a circle.
    #[inline]
    #[must_use]
    pub const fn new(centre: Vec2, radius: Fx) -> Circle {
        Circle { centre, radius }
    }

    /// True when `point` lies inside or on the boundary.
    #[inline]
    #[must_use]
    pub fn contains_point(self, point: Vec2) -> bool {
        self.centre.distance_squared(point) <= self.radius * self.radius
    }

    /// True when two circles overlap or touch.
    #[inline]
    #[must_use]
    pub fn intersects(self, other: Circle) -> bool {
        let sum = self.radius + other.radius;
        self.centre.distance_squared(other.centre) <= sum * sum
    }

    /// True when the circle overlaps an axis-aligned rectangle.
    ///
    /// Works by clamping the centre into the rectangle and measuring back — the
    /// standard test, correct for every relative placement including the circle
    /// fully inside the rectangle.
    #[must_use]
    pub fn intersects_rect(self, rect: Rect) -> bool {
        let closest = rect.clamp_point(self.centre);
        self.centre.distance_squared(closest) <= self.radius * self.radius
    }

    /// The axis-aligned bounding box of this circle.
    #[inline]
    #[must_use]
    pub fn bounds(self) -> Rect {
        Rect::from_centre(self.centre, Vec2::splat(self.radius * Fx::TWO))
    }
}

/// The result of a swept ray or box cast against a rectangle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RayHit {
    /// Fraction along the ray, in `[0, 1]`, at which contact occurs.
    pub time: Fx,
    /// Contact point in world space.
    pub point: Vec2,
    /// Unit surface normal at the contact point.
    pub normal: Vec2,
}

/// Casts a ray against an axis-aligned rectangle using the slab method.
///
/// `direction` is the full displacement of the ray, not a unit vector, so
/// [`RayHit::time`] is directly the fraction of the intended motion that is
/// safe to apply. Returns `None` when the ray misses, starts past the box, or
/// travels parallel to and outside a slab.
#[must_use]
pub fn ray_vs_rect(origin: Vec2, direction: Vec2, rect: Rect) -> Option<RayHit> {
    // Per-axis entry and exit times. A zero direction component means the ray
    // is parallel to that slab, so it either never enters or is always inside.
    let (near_x, far_x) = if direction.x.is_zero() {
        if origin.x < rect.left() || origin.x > rect.right() {
            return None;
        }
        (Fx::MIN, Fx::MAX)
    } else {
        let t1 = (rect.left() - origin.x) / direction.x;
        let t2 = (rect.right() - origin.x) / direction.x;
        (t1.min(t2), t1.max(t2))
    };

    let (near_y, far_y) = if direction.y.is_zero() {
        if origin.y < rect.top() || origin.y > rect.bottom() {
            return None;
        }
        (Fx::MIN, Fx::MAX)
    } else {
        let t1 = (rect.top() - origin.y) / direction.y;
        let t2 = (rect.bottom() - origin.y) / direction.y;
        (t1.min(t2), t1.max(t2))
    };

    let entry = near_x.max(near_y);
    let exit = far_x.min(far_y);

    // Missed entirely, or the box is behind us, or contact is past the end.
    if entry > exit || exit.is_negative() || entry > Fx::ONE {
        return None;
    }
    let time = entry.max(Fx::ZERO);

    // The axis with the later entry time is the face actually struck.
    let normal = if near_x > near_y {
        if direction.x.is_negative() {
            Vec2::RIGHT
        } else {
            Vec2::LEFT
        }
    } else if direction.y.is_negative() {
        Vec2::DOWN
    } else {
        Vec2::UP
    };

    Some(RayHit { time, point: origin + direction * time, normal })
}

/// Sweeps `moving` along `displacement` against the stationary rectangle
/// `obstacle`, returning the first contact.
///
/// Implemented by reducing the box-vs-box sweep to a ray cast against the
/// Minkowski sum of the two rectangles — the standard trick that makes swept
/// AABB collision exact rather than iterative.
#[must_use]
pub fn sweep_rect_vs_rect(moving: Rect, displacement: Vec2, obstacle: Rect) -> Option<RayHit> {
    if displacement.is_zero() {
        return None;
    }
    // Expanding the obstacle by the mover's size turns the mover into a point.
    let expanded = Rect::new(obstacle.min - moving.size * Fx::HALF, obstacle.size + moving.size);
    ray_vs_rect(moving.centre(), displacement, expanded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::fx;

    #[test]
    fn contains_point_uses_half_open_bounds() {
        let rect = Rect::from_ints(0, 0, 10, 10);
        assert!(rect.contains_point(Vec2::from_ints(0, 0)), "min corner is inside");
        assert!(rect.contains_point(Vec2::from_ints(9, 9)));
        assert!(!rect.contains_point(Vec2::from_ints(10, 5)), "max edge is outside");
        assert!(!rect.contains_point(Vec2::from_ints(5, 10)));
        assert!(!rect.contains_point(Vec2::from_ints(-1, 5)));
    }

    #[test]
    fn touching_rectangles_do_not_count_as_intersecting() {
        let floor = Rect::from_ints(0, 10, 10, 2);
        let resting = Rect::from_ints(0, 0, 10, 10);
        assert!(!floor.intersects(resting), "a body resting exactly on a floor is not colliding");
        let sunk = Rect::from_ints(0, 1, 10, 10);
        assert!(floor.intersects(sunk));
    }

    #[test]
    fn intersection_returns_the_overlapping_region() {
        let a = Rect::from_ints(0, 0, 10, 10);
        let b = Rect::from_ints(5, 5, 10, 10);
        let overlap = a.intersection(b).expect("rectangles overlap");
        assert_eq!(overlap, Rect::from_ints(5, 5, 5, 5));
        assert_eq!(a.intersection(Rect::from_ints(100, 100, 1, 1)), None);
    }

    #[test]
    fn union_covers_both_inputs() {
        let a = Rect::from_ints(0, 0, 4, 4);
        let b = Rect::from_ints(8, 8, 2, 2);
        let union = a.union(b);
        assert!(union.contains_rect(a));
        assert!(union.contains_rect(b));
        assert_eq!(union, Rect::from_ints(0, 0, 10, 10));
    }

    #[test]
    fn penetration_resolves_along_the_shallow_axis() {
        let wall = Rect::from_ints(0, 0, 100, 10);
        // Overlaps 2 units vertically but 50 horizontally: must push vertically.
        let body = Rect::from_ints(25, 8, 50, 10);
        let push = wall.penetration_vector(body).expect("bodies overlap");
        assert_eq!(push.x, Fx::ZERO);
        assert_eq!(push.y, fx(2), "pushed down and out by the overlap depth");
    }

    #[test]
    fn circle_rect_test_handles_corners_and_containment() {
        let rect = Rect::from_ints(0, 0, 10, 10);
        assert!(Circle::new(Vec2::from_ints(5, 5), fx(1)).intersects_rect(rect), "inside");
        assert!(Circle::new(Vec2::from_ints(-1, 5), fx(2)).intersects_rect(rect), "through an edge");
        assert!(Circle::new(Vec2::from_ints(-1, -1), fx(2)).intersects_rect(rect), "past a corner");
        assert!(
            !Circle::new(Vec2::from_ints(-5, -5), fx(2)).intersects_rect(rect),
            "clear of the corner"
        );
    }

    #[test]
    fn ray_hits_the_near_face_with_the_correct_normal() {
        let rect = Rect::from_ints(10, 0, 10, 10);
        let hit = ray_vs_rect(Vec2::from_ints(0, 5), Vec2::from_ints(20, 0), rect)
            .expect("ray travels into the box");
        assert_eq!(hit.time, Fx::HALF, "contact halfway along the 20-unit ray");
        assert_eq!(hit.point.x, fx(10));
        assert_eq!(hit.normal, Vec2::LEFT, "struck the left face");
    }

    #[test]
    fn ray_misses_return_none() {
        let rect = Rect::from_ints(10, 0, 10, 10);
        assert!(ray_vs_rect(Vec2::from_ints(0, 50), Vec2::from_ints(20, 0), rect).is_none());
        assert!(
            ray_vs_rect(Vec2::from_ints(0, 5), Vec2::from_ints(-20, 0), rect).is_none(),
            "pointing away"
        );
        assert!(
            ray_vs_rect(Vec2::from_ints(0, 5), Vec2::from_ints(5, 0), rect).is_none(),
            "stops short"
        );
    }

    #[test]
    fn ray_parallel_to_a_slab_still_hits_when_it_is_inside_it() {
        let rect = Rect::from_ints(10, 0, 10, 10);
        // Exactly horizontal, y inside the box's vertical slab.
        let hit = ray_vs_rect(Vec2::from_ints(0, 0), Vec2::from_ints(20, 0), rect);
        assert!(hit.is_some());
    }

    #[test]
    fn swept_box_stops_at_contact_rather_than_tunnelling() {
        // A 2x2 body moving 100 units right at a wall it would otherwise skip.
        let body = Rect::from_ints(0, 0, 2, 2);
        let wall = Rect::from_ints(50, 0, 10, 2);
        let hit = sweep_rect_vs_rect(body, Vec2::from_ints(100, 0), wall)
            .expect("the sweep must catch the wall");
        assert_eq!(hit.normal, Vec2::LEFT);
        // Contact when the body's right edge reaches the wall's left edge: the
        // centre travels from 1 to 49, i.e. 48 of the 100 units.
        assert!((hit.time.to_f64() - 0.48).abs() < 1e-6, "time was {}", hit.time.to_f64());
    }

    #[test]
    fn swept_box_with_no_displacement_reports_no_hit() {
        let body = Rect::from_ints(0, 0, 2, 2);
        let wall = Rect::from_ints(1, 0, 10, 2);
        assert!(sweep_rect_vs_rect(body, Vec2::ZERO, wall).is_none());
    }
}
