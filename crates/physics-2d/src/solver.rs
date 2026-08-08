//! The movement solver.
//!
//! # Approach
//!
//! Movement is resolved one axis at a time: apply the horizontal component,
//! push out of anything it entered, then apply the vertical component and do
//! the same. Axis separation is what produces sliding — a body walking into a
//! wall at an angle keeps the tangential part of its motion instead of
//! stopping dead — and it does so without a general constraint solver.
//!
//! Each axis step is further split into sub-steps small enough that a body
//! cannot cross a solid tile in one go. Without that, anything moving faster
//! than a tile per tick tunnels straight through walls, and the failure only
//! appears at high speed or low frame rates.
//!
//! # Why not an impulse solver
//!
//! A general rigid-body solver models restitution, friction and mass, and
//! produces motion that feels *simulated*. A farming and exploration game wants
//! motion that feels *controlled*: the character moves exactly where the stick
//! points and stops exactly at walls. Kinematic axis-separated resolution gives
//! that directly, and it is deterministic by construction — every step is
//! fixed-point arithmetic with no iteration count to converge.

use crate::components::{Collider, CollisionFlags};
use crate::grid::SolidGrid;
use verdant_core_math::{Fx, Rect, Vec2};

/// The outcome of moving a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveResult {
    /// Where the body ended up.
    pub position: Vec2,
    /// Which sides were blocked along the way.
    pub flags: CollisionFlags,
    /// The motion that was actually applied.
    ///
    /// Comparing this against the requested displacement is how a caller
    /// detects "I tried to move but could not", which drives push animations
    /// and the stuck-detection in NPC pathing.
    pub applied: Vec2,
}

/// A tiny gap left between a body and whatever stopped it.
///
/// Resting exactly flush means the next tick's overlap test sees a
/// zero-thickness contact, which fixed-point rounding can flip either way. The
/// gap is far below one screen pixel at any zoom, so it is invisible, but it
/// keeps contact classification stable from tick to tick.
const SKIN_WIDTH: Fx = Fx::from_raw(1 << 20); // 2^-12 world units

/// Moves a body along `displacement`, stopping at solid geometry and sliding.
///
/// `solids` supplies the tile grid; `blockers` supplies any additional
/// world-space boxes (other entities, placed buildings) that should stop this
/// body. Both are consulted for every sub-step.
///
/// The returned [`MoveResult::position`] is the body's new ground point.
pub fn move_and_slide(
    collider: &Collider,
    position: Vec2,
    displacement: Vec2,
    solids: &dyn SolidGrid,
    blockers: &[Rect],
) -> MoveResult {
    if displacement.is_zero() {
        return MoveResult {
            position,
            flags: CollisionFlags::default(),
            applied: Vec2::ZERO,
        };
    }

    // Sub-step so the body never advances further than half its own smallest
    // dimension (or half a tile, whichever is smaller) before being tested.
    // Half, not a whole, because a body that steps exactly its own width can
    // still straddle a thin wall without ever overlapping it at a sample point.
    let extent = collider
        .bounds
        .size
        .x
        .min(collider.bounds.size.y)
        .max(SKIN_WIDTH);
    let step_limit = extent.min(solids.tile_size()) * Fx::HALF;
    let distance = displacement.length();
    let steps = if step_limit.is_positive() {
        (distance / step_limit).ceil_int().max(1)
    } else {
        1
    };
    // A pathological displacement (a teleport expressed as movement) would
    // otherwise spend an unbounded time sub-stepping.
    let steps = steps.min(MAX_SUBSTEPS);

    let mut current = position;
    let mut flags = CollisionFlags::default();
    // Carrying the *remaining* displacement and letting the final sub-step
    // consume whatever is left keeps the sub-steps summing to exactly the
    // requested motion. Dividing once and adding the quotient back `steps`
    // times would leave a rounding residue, so an unobstructed body would drift
    // a fraction of a unit short of where gameplay asked it to go — and would
    // do so differently for different step counts.
    let mut remaining = displacement;

    for step_index in 0..steps {
        let steps_left = steps - step_index;
        let step = if steps_left == 1 {
            remaining
        } else {
            remaining / Fx::from_num(steps_left)
        };
        remaining -= step;

        let (moved, step_flags) = resolve_axis_x(collider, current, step.x, solids, blockers);
        current = moved;
        flags = flags.merge(step_flags);

        let (moved, step_flags) = resolve_axis_y(collider, current, step.y, solids, blockers);
        current = moved;
        flags = flags.merge(step_flags);
    }

    MoveResult {
        position: current,
        flags,
        applied: current - position,
    }
}

/// Upper bound on sub-steps per call, so a huge displacement cannot stall the
/// simulation. At the engine's tile size this still permits a body to cross
/// sixty-four tiles in a single tick.
const MAX_SUBSTEPS: i32 = 128;

/// Applies horizontal motion and pushes out of anything entered.
fn resolve_axis_x(
    collider: &Collider,
    position: Vec2,
    delta: Fx,
    solids: &dyn SolidGrid,
    blockers: &[Rect],
) -> (Vec2, CollisionFlags) {
    let mut flags = CollisionFlags::default();
    if delta.is_zero() {
        return (position, flags);
    }
    let candidate = Vec2::new(position.x + delta, position.y);
    let box_at = collider.world_bounds(candidate);

    let Some(obstruction) = first_obstruction(box_at, solids, blockers) else {
        return (candidate, flags);
    };

    // Snap flush against the obstruction's near face, less the skin gap.
    let resolved_x = if delta.is_positive() {
        flags.right = true;
        obstruction.left() - collider.bounds.right() - SKIN_WIDTH
    } else {
        flags.left = true;
        obstruction.right() - collider.bounds.left() + SKIN_WIDTH
    };

    // Only accept the snap if it does not push the body backwards past where it
    // started; a body that begins already overlapping must not be teleported.
    let resolved = if delta.is_positive() {
        Vec2::new(resolved_x.min(candidate.x).max(position.x), position.y)
    } else {
        Vec2::new(resolved_x.max(candidate.x).min(position.x), position.y)
    };
    (resolved, flags)
}

/// Applies vertical motion and pushes out of anything entered.
fn resolve_axis_y(
    collider: &Collider,
    position: Vec2,
    delta: Fx,
    solids: &dyn SolidGrid,
    blockers: &[Rect],
) -> (Vec2, CollisionFlags) {
    let mut flags = CollisionFlags::default();
    if delta.is_zero() {
        return (position, flags);
    }
    let candidate = Vec2::new(position.x, position.y + delta);
    let box_at = collider.world_bounds(candidate);

    let Some(obstruction) = first_obstruction(box_at, solids, blockers) else {
        return (candidate, flags);
    };

    let resolved_y = if delta.is_positive() {
        flags.down = true;
        obstruction.top() - collider.bounds.bottom() - SKIN_WIDTH
    } else {
        flags.up = true;
        obstruction.bottom() - collider.bounds.top() + SKIN_WIDTH
    };

    let resolved = if delta.is_positive() {
        Vec2::new(position.x, resolved_y.min(candidate.y).max(position.y))
    } else {
        Vec2::new(position.x, resolved_y.max(candidate.y).min(position.y))
    };
    (resolved, flags)
}

/// Finds the first thing `bounds` overlaps, if any.
///
/// Tiles are checked before entity blockers because tile geometry is the
/// overwhelming majority of the world and the lookup is a direct index rather
/// than a scan.
fn first_obstruction(bounds: Rect, solids: &dyn SolidGrid, blockers: &[Rect]) -> Option<Rect> {
    if let Some(tile) = overlapping_solid_tile(bounds, solids) {
        return Some(tile);
    }
    blockers
        .iter()
        .copied()
        .find(|blocker| bounds.intersects(*blocker))
}

/// Returns the world bounds of the first solid tile overlapping `bounds`.
///
/// Only the cells the box actually covers are visited, so cost is proportional
/// to the body's size rather than the map's.
fn overlapping_solid_tile(bounds: Rect, solids: &dyn SolidGrid) -> Option<Rect> {
    let tile_size = solids.tile_size();
    if !tile_size.is_positive() {
        return None;
    }
    let min_x = (bounds.left() / tile_size).floor_int();
    let min_y = (bounds.top() / tile_size).floor_int();
    // The max edge is exclusive: a box ending exactly on a tile boundary must
    // not be considered to be inside the next tile along.
    let max_x = ((bounds.right() - SKIN_WIDTH) / tile_size).floor_int();
    let max_y = ((bounds.bottom() - SKIN_WIDTH) / tile_size).floor_int();

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let cell = verdant_core_math::IVec2::new(x, y);
            if solids.is_solid(cell) {
                return Some(Rect::new(cell.to_world(tile_size), Vec2::splat(tile_size)));
            }
        }
    }
    None
}

/// True when a body of this shape could stand at `position` without overlapping
/// anything solid.
///
/// Used for spawn placement, teleport destinations and NPC path validation.
#[must_use]
pub fn is_position_clear(
    collider: &Collider,
    position: Vec2,
    solids: &dyn SolidGrid,
    blockers: &[Rect],
) -> bool {
    first_obstruction(collider.world_bounds(position), solids, blockers).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::TestGrid;
    use verdant_core_math::{fx, IVec2};

    /// A one-unit body, small enough to fit between the test grid's tiles.
    fn body() -> Collider {
        Collider::centred(Fx::ONE, Fx::ONE)
    }

    #[test]
    fn unobstructed_movement_is_applied_exactly() {
        let grid = TestGrid::empty(fx(16));
        let result = move_and_slide(&body(), Vec2::ZERO, Vec2::from_ints(5, 3), &grid, &[]);
        assert_eq!(result.position, Vec2::from_ints(5, 3));
        assert!(!result.flags.any());
        assert_eq!(result.applied, Vec2::from_ints(5, 3));
    }

    #[test]
    fn zero_displacement_is_a_no_op() {
        let grid = TestGrid::empty(fx(16));
        let result = move_and_slide(&body(), Vec2::from_ints(3, 3), Vec2::ZERO, &grid, &[]);
        assert_eq!(result.position, Vec2::from_ints(3, 3));
        assert_eq!(result.applied, Vec2::ZERO);
    }

    #[test]
    fn a_body_stops_against_a_solid_tile() {
        // Tile (1, 0) spans x in [16, 32).
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0)]);
        let start = Vec2::new(fx(8), fx(8));
        let result = move_and_slide(&body(), start, Vec2::from_ints(20, 0), &grid, &[]);

        assert!(result.flags.right, "the right side should report blocked");
        // The body's right edge stops at the tile's left edge.
        let right_edge = result.position.x + body().bounds.right();
        assert!(
            right_edge <= fx(16),
            "penetrated to {}",
            right_edge.to_f64()
        );
        assert!(
            right_edge > fx(16) - Fx::from_ratio(1, 100),
            "stopped short at {}",
            right_edge.to_f64()
        );
    }

    #[test]
    fn a_body_slides_along_a_wall_instead_of_stopping() {
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0), IVec2::new(1, 1)]);
        let start = Vec2::new(fx(8), fx(8));
        // Diagonal into the wall: the x component is blocked, y must survive.
        let result = move_and_slide(&body(), start, Vec2::from_ints(20, 4), &grid, &[]);

        assert!(result.flags.right);
        assert_eq!(
            result.position.y,
            fx(12),
            "the tangential component slid freely"
        );
    }

    #[test]
    fn a_fast_body_cannot_tunnel_through_a_thin_wall() {
        // One solid tile, and a body moving many tiles in a single call.
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(4, 0)]);
        let start = Vec2::new(fx(8), fx(8));
        let result = move_and_slide(&body(), start, Vec2::from_ints(500, 0), &grid, &[]);

        assert!(result.flags.right, "the sweep must catch the wall");
        assert!(
            result.position.x < fx(64),
            "tunnelled to x = {}",
            result.position.x.to_f64()
        );
    }

    #[test]
    fn movement_is_blocked_in_every_direction() {
        let grid = TestGrid::with_solids(
            fx(16),
            &[
                IVec2::new(1, 0),
                IVec2::new(-1, 0),
                IVec2::new(0, 1),
                IVec2::new(0, -1),
            ],
        );
        let start = Vec2::new(fx(8), fx(8));

        let right = move_and_slide(&body(), start, Vec2::from_ints(30, 0), &grid, &[]);
        assert!(right.flags.right);
        let left = move_and_slide(&body(), start, Vec2::from_ints(-30, 0), &grid, &[]);
        assert!(left.flags.left);
        let down = move_and_slide(&body(), start, Vec2::from_ints(0, 30), &grid, &[]);
        assert!(down.flags.down);
        let up = move_and_slide(&body(), start, Vec2::from_ints(0, -30), &grid, &[]);
        assert!(up.flags.up);
    }

    #[test]
    fn entity_blockers_stop_a_body_like_tiles_do() {
        let grid = TestGrid::empty(fx(16));
        let wall = Rect::from_ints(10, -10, 4, 40);
        let result = move_and_slide(
            &body(),
            Vec2::new(fx(0), fx(0)),
            Vec2::from_ints(20, 0),
            &grid,
            &[wall],
        );
        assert!(result.flags.right);
        assert!(result.position.x + body().bounds.right() <= fx(10));
    }

    #[test]
    fn a_body_never_moves_backwards_when_resolving() {
        // Start already overlapping a tile; the solver must not fling it away.
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(0, 0)]);
        let start = Vec2::new(fx(8), fx(8));
        let result = move_and_slide(&body(), start, Vec2::from_ints(4, 0), &grid, &[]);
        assert!(
            result.position.x >= start.x,
            "the body was pushed backwards to {}",
            result.position.x.to_f64()
        );
    }

    #[test]
    fn resting_against_a_wall_stays_stable_over_many_ticks() {
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0)]);
        let mut position = Vec2::new(fx(8), fx(8));
        // Push into the wall every tick; the position must converge and stop
        // changing rather than creeping or jittering.
        for _ in 0..10 {
            position = move_and_slide(&body(), position, Vec2::new(Fx::ONE, Fx::ZERO), &grid, &[])
                .position;
        }
        let settled = position;
        for _ in 0..10 {
            position = move_and_slide(&body(), position, Vec2::new(Fx::ONE, Fx::ZERO), &grid, &[])
                .position;
        }
        assert_eq!(position, settled, "a body resting on a wall must not drift");
    }

    #[test]
    fn clearance_testing_agrees_with_the_solver() {
        let grid = TestGrid::with_solids(fx(16), &[IVec2::new(1, 0)]);
        assert!(is_position_clear(
            &body(),
            Vec2::new(fx(8), fx(8)),
            &grid,
            &[]
        ));
        assert!(!is_position_clear(
            &body(),
            Vec2::new(fx(20), fx(8)),
            &grid,
            &[]
        ));
    }

    #[test]
    fn determinism_the_same_motion_produces_the_same_path() {
        let grid = TestGrid::with_solids(
            fx(16),
            &[
                IVec2::new(2, 0),
                IVec2::new(2, 1),
                IVec2::new(3, 2),
                IVec2::new(-1, 1),
            ],
        );
        let run = || {
            let mut position = Vec2::new(fx(8), fx(8));
            let mut trace = Vec::new();
            for step in 0..200i32 {
                let angle = Fx::TAU * Fx::from_ratio(step, 37);
                let displacement = Vec2::from_angle(angle, Fx::from_ratio(3, 2));
                position = move_and_slide(&body(), position, displacement, &grid, &[]).position;
                trace.push((position.x.to_raw(), position.y.to_raw()));
            }
            trace
        };
        assert_eq!(
            run(),
            run(),
            "identical inputs must trace an identical path"
        );
    }
}
