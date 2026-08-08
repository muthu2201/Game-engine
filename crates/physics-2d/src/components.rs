//! The spatial components every other subsystem reads.
//!
//! [`Transform`] lives here rather than in the ECS crate because position is a
//! *physical* property in this engine: the collision solver owns it, and the
//! renderer, audio panning and tile queries all read whatever the solver last
//! wrote. Keeping it beside the solver makes that ownership explicit.

use serde::{Deserialize, Serialize};
use verdant_core_ecs::{Component, StorageKind};
use verdant_core_math::{Fx, Rect, Vec2};

/// Where an entity is in the world.
///
/// Position is the entity's **feet** — the bottom-centre of its collider — not
/// its top-left or its centre. Top-down games sort sprites by the ground point
/// and place shadows there, so making it the canonical anchor removes a
/// per-sprite offset from every one of those calculations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transform {
    /// World position of the entity's ground point.
    pub position: Vec2,
    /// Rotation in radians. Most 2D sprites leave this at zero.
    pub rotation: Fx,
    /// Per-axis scale. Negative x is how sprites are mirrored.
    pub scale: Vec2,
}

impl Transform {
    /// A transform at the origin with unit scale.
    pub const IDENTITY: Transform = Transform {
        position: Vec2::ZERO,
        rotation: Fx::ZERO,
        scale: Vec2::ONE,
    };

    /// A transform at `position` with unit scale and no rotation.
    #[inline]
    #[must_use]
    pub const fn at(position: Vec2) -> Transform {
        Transform {
            position,
            rotation: Fx::ZERO,
            scale: Vec2::ONE,
        }
    }

    /// A transform at integer world coordinates.
    #[inline]
    #[must_use]
    pub const fn at_ints(x: i32, y: i32) -> Transform {
        Transform::at(Vec2::from_ints(x, y))
    }
}

impl Component for Transform {}

/// An entity's linear velocity, in world units per second.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Velocity(pub Vec2);

impl Velocity {
    /// A stationary velocity.
    pub const ZERO: Velocity = Velocity(Vec2::ZERO);

    /// Builds a velocity from components.
    #[inline]
    #[must_use]
    pub const fn new(x: Fx, y: Fx) -> Velocity {
        Velocity(Vec2::new(x, y))
    }
}

impl Component for Velocity {}

/// How the solver treats a body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyKind {
    /// Never moves. Walls, fences, buildings.
    ///
    /// Static bodies are the only ones stored in the broadphase's persistent
    /// grid, because rebuilding that grid every tick for things that cannot
    /// move would be the solver's dominant cost.
    #[default]
    Static,

    /// Moves under direct control, and is stopped by static geometry.
    ///
    /// The player, NPCs and pushed crates. Kinematic rather than dynamic
    /// because a farming game wants movement that responds exactly to input,
    /// not one mediated by forces and restitution.
    Kinematic,

    /// Moves and is stopped by geometry, but does not stop anything itself.
    ///
    /// Projectiles, thrown items, particles with collision.
    Projectile,
}

/// An axis-aligned collision box.
///
/// The box is expressed relative to the entity's [`Transform::position`], so a
/// character whose position is at its feet has a collider with a negative
/// `min.y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collider {
    /// Bounds relative to the entity's position.
    pub bounds: Rect,
    /// Which layers this body belongs to, as a bitmask.
    pub layer: u32,
    /// Which layers this body collides with, as a bitmask.
    ///
    /// Collision requires *both* directions to agree, so a body cannot be
    /// blocked by something that does not consider it blocking.
    pub mask: u32,
    /// How the solver treats this body.
    pub kind: BodyKind,
    /// When true, overlaps are reported but never resolved.
    ///
    /// Sensors are how interaction ranges, trigger zones and hitboxes are
    /// expressed without giving them physical presence.
    pub sensor: bool,
}

/// Collision layer bits.
///
/// Layers are a plain bitmask rather than an enum so a body can belong to more
/// than one, which the interaction system relies on: a harvestable crop is both
/// `TERRAIN` (it blocks nothing) and `INTERACTABLE`.
pub mod layers {
    /// The player character.
    pub const PLAYER: u32 = 1 << 0;
    /// Non-player characters.
    pub const NPC: u32 = 1 << 1;
    /// Hostile creatures.
    pub const ENEMY: u32 = 1 << 2;
    /// Solid world geometry that is not tile-based.
    pub const TERRAIN: u32 = 1 << 3;
    /// Items lying on the ground.
    pub const ITEM: u32 = 1 << 4;
    /// Things that respond to the interact button.
    pub const INTERACTABLE: u32 = 1 << 5;
    /// Damage-dealing volumes.
    pub const HITBOX: u32 = 1 << 6;
    /// Every layer.
    pub const ALL: u32 = u32::MAX;
}

impl Collider {
    /// A collider centred horizontally on the entity and rising from its feet.
    ///
    /// This is the shape almost every character wants, and computing it here
    /// keeps the sign convention in one place instead of at each call site.
    #[must_use]
    pub fn feet_anchored(width: Fx, height: Fx) -> Collider {
        Collider {
            bounds: Rect::new(
                Vec2::new(-width * Fx::HALF, -height),
                Vec2::new(width, height),
            ),
            layer: layers::TERRAIN,
            mask: layers::ALL,
            kind: BodyKind::Static,
            sensor: false,
        }
    }

    /// A collider centred on the entity's position.
    #[must_use]
    pub fn centred(width: Fx, height: Fx) -> Collider {
        Collider {
            bounds: Rect::from_centre(Vec2::ZERO, Vec2::new(width, height)),
            layer: layers::TERRAIN,
            mask: layers::ALL,
            kind: BodyKind::Static,
            sensor: false,
        }
    }

    /// Returns this collider with the given body kind.
    #[must_use]
    pub const fn with_kind(mut self, kind: BodyKind) -> Collider {
        self.kind = kind;
        self
    }

    /// Returns this collider on the given layer, colliding with `mask`.
    #[must_use]
    pub const fn with_layers(mut self, layer: u32, mask: u32) -> Collider {
        self.layer = layer;
        self.mask = mask;
        self
    }

    /// Returns this collider marked as a non-blocking sensor.
    #[must_use]
    pub const fn as_sensor(mut self) -> Collider {
        self.sensor = true;
        self
    }

    /// The collider's world-space bounds when the entity is at `position`.
    #[inline]
    #[must_use]
    pub fn world_bounds(&self, position: Vec2) -> Rect {
        self.bounds.translate(position)
    }

    /// True when these two colliders are configured to interact.
    ///
    /// Both masks must accept the other's layer. Requiring agreement in both
    /// directions is what stops a one-sided mask edit from silently making a
    /// body pass through walls.
    #[inline]
    #[must_use]
    pub const fn interacts_with(&self, other: &Collider) -> bool {
        (self.mask & other.layer) != 0 && (other.mask & self.layer) != 0
    }
}

impl Component for Collider {}

/// Which sides of a body touched something during the last solver step.
///
/// Gameplay reads these rather than re-testing geometry: "am I standing on
/// something" is a flag the solver already computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollisionFlags {
    /// Blocked while moving left.
    pub left: bool,
    /// Blocked while moving right.
    pub right: bool,
    /// Blocked while moving up.
    pub up: bool,
    /// Blocked while moving down — standing on ground, in a top-down game.
    pub down: bool,
}

impl CollisionFlags {
    /// True when any side is blocked.
    #[inline]
    #[must_use]
    pub const fn any(self) -> bool {
        self.left || self.right || self.up || self.down
    }

    /// True when movement was blocked horizontally.
    #[inline]
    #[must_use]
    pub const fn horizontal(self) -> bool {
        self.left || self.right
    }

    /// True when movement was blocked vertically.
    #[inline]
    #[must_use]
    pub const fn vertical(self) -> bool {
        self.up || self.down
    }

    /// Merges two sets of flags, keeping every blocked side.
    #[inline]
    #[must_use]
    pub const fn merge(self, other: CollisionFlags) -> CollisionFlags {
        CollisionFlags {
            left: self.left || other.left,
            right: self.right || other.right,
            up: self.up || other.up,
            down: self.down || other.down,
        }
    }
}

impl Component for CollisionFlags {
    // Written every tick for every moving body and read by a handful of
    // systems: sparse storage keeps it out of the movement archetypes.
    const STORAGE: StorageKind = StorageKind::Sparse;
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    #[test]
    fn feet_anchored_colliders_sit_above_the_position() {
        let collider = Collider::feet_anchored(fx(2), fx(4));
        let bounds = collider.world_bounds(Vec2::from_ints(10, 10));
        assert_eq!(
            bounds.bottom(),
            fx(10),
            "the box's base is at the entity's feet"
        );
        assert_eq!(bounds.top(), fx(6));
        assert_eq!(bounds.left(), fx(9), "and it is centred horizontally");
        assert_eq!(bounds.right(), fx(11));
    }

    #[test]
    fn centred_colliders_surround_the_position() {
        let bounds = Collider::centred(fx(2), fx(2)).world_bounds(Vec2::ZERO);
        assert_eq!(bounds.centre(), Vec2::ZERO);
    }

    #[test]
    fn interaction_requires_both_masks_to_agree() {
        let player = Collider::centred(fx(1), fx(1)).with_layers(layers::PLAYER, layers::TERRAIN);
        let wall = Collider::centred(fx(1), fx(1)).with_layers(layers::TERRAIN, layers::PLAYER);
        assert!(player.interacts_with(&wall));
        assert!(wall.interacts_with(&player));

        // The wall stops considering the player: neither direction interacts.
        let deaf_wall = wall.with_layers(layers::TERRAIN, layers::ENEMY);
        assert!(!player.interacts_with(&deaf_wall));
        assert!(!deaf_wall.interacts_with(&player));
    }

    #[test]
    fn collision_flags_merge_without_losing_sides() {
        let horizontal = CollisionFlags {
            left: true,
            ..CollisionFlags::default()
        };
        let vertical = CollisionFlags {
            down: true,
            ..CollisionFlags::default()
        };
        let merged = horizontal.merge(vertical);
        assert!(merged.left && merged.down);
        assert!(merged.any() && merged.horizontal() && merged.vertical());
        assert!(!CollisionFlags::default().any());
    }

    #[test]
    fn builders_compose() {
        let sensor = Collider::centred(fx(1), fx(1))
            .with_kind(BodyKind::Kinematic)
            .with_layers(layers::PLAYER, layers::ITEM)
            .as_sensor();
        assert_eq!(sensor.kind, BodyKind::Kinematic);
        assert_eq!(sensor.layer, layers::PLAYER);
        assert!(sensor.sensor);
    }
}
