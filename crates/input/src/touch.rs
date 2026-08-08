//! Touch input, and the virtual controls built on it.
//!
//! A phone has no keyboard, so something has to stand in for one. This module
//! is that something: it tracks active touch points and resolves them into a
//! thumbstick direction and a set of pressed buttons, which the rest of the
//! engine then treats exactly like a gamepad.
//!
//! ## Why the layout lives here rather than in the renderer
//!
//! Where a control *is* and where it is *drawn* have to agree exactly, or the
//! player presses a button and nothing happens. Keeping the geometry in one
//! place, in the crate that resolves input, means the HUD asks this module
//! where to draw rather than the other way round — so the two cannot drift
//! apart.
//!
//! ## Finger tracking
//!
//! Touches are identified by the platform's finger id, not by position. A
//! player whose thumb slides off the stick and back on must keep control of
//! it, and a second finger pressing a button must not steal the first one's
//! stick — both of which happen if you match touches by proximity each frame.

use crate::{Action, Vec2};
use std::collections::BTreeMap;
use verdant_core_math::{Fx, Rect};

/// A finger on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TouchId(pub u64);

/// What is happening to a touch point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TouchPhase {
    /// The finger just came down.
    Started,
    /// It is down and may have moved.
    Moved,
    /// It lifted.
    Ended,
    /// The platform took it away — a system gesture, or a call arriving.
    Cancelled,
}

/// One touch event from the platform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TouchEvent {
    /// Which finger.
    pub id: TouchId,
    /// Where, in the game's internal pixel space.
    pub position: Vec2,
    /// What it is doing.
    pub phase: TouchPhase,
}

/// A round region that acts as a thumbstick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thumbstick {
    /// Where the stick rests, in internal pixels.
    pub centre: Vec2,
    /// How far the thumb travels for full deflection.
    pub radius: Fx,
    /// How far outside `radius` a touch may start and still grab the stick.
    ///
    /// Generous, because the player's thumb is nowhere near as precise as a
    /// mouse and a stick that has to be hit exactly is a stick that is missed.
    pub grab_margin: Fx,
}

impl Thumbstick {
    /// A stick at a position.
    #[must_use]
    pub fn new(centre: Vec2, radius: Fx) -> Thumbstick {
        Thumbstick {
            centre,
            radius,
            grab_margin: radius,
        }
    }

    /// True when a touch starting here should take control of the stick.
    #[must_use]
    pub fn accepts(&self, position: Vec2) -> bool {
        position.distance(self.centre) <= self.radius + self.grab_margin
    }

    /// The direction a thumb at `position` is pushing, never longer than one.
    ///
    /// A thumb dragged past the edge keeps pushing at full deflection rather
    /// than losing the stick, which is what a physical stick does and what a
    /// player expects.
    #[must_use]
    pub fn direction(&self, position: Vec2) -> Vec2 {
        if self.radius <= Fx::ZERO {
            return Vec2::ZERO;
        }
        let offset = position - self.centre;
        let distance = offset.length();
        if distance <= Fx::ZERO {
            return Vec2::ZERO;
        }
        if distance >= self.radius {
            return offset / distance;
        }
        offset / self.radius
    }
}

/// A rectangular region that acts as a button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TouchButton {
    /// The action it triggers.
    pub action: Action,
    /// Where it is, in internal pixels.
    pub bounds: Rect,
}

impl TouchButton {
    /// A button covering a region.
    #[must_use]
    pub const fn new(action: Action, bounds: Rect) -> TouchButton {
        TouchButton { action, bounds }
    }

    /// True when a touch at this position presses the button.
    #[must_use]
    pub fn contains(&self, position: Vec2) -> bool {
        self.bounds.contains_point(position)
    }
}

/// Which control a finger has claimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Claim {
    /// It is driving the thumbstick.
    Stick,
    /// It is holding a button, identified by its index in the layout.
    Button(usize),
    /// It landed on nothing.
    None,
}

/// The on-screen control layout.
///
/// Built once for a screen size, then asked both what is pressed and where to
/// draw — the single source of truth that keeps the two in agreement.
#[derive(Clone, Debug, PartialEq)]
pub struct TouchLayout {
    /// The movement stick.
    pub stick: Thumbstick,
    /// The action buttons.
    pub buttons: Vec<TouchButton>,
}

impl TouchLayout {
    /// A layout with a stick and no buttons.
    #[must_use]
    pub fn new(stick: Thumbstick) -> TouchLayout {
        TouchLayout {
            stick,
            buttons: Vec::new(),
        }
    }

    /// Adds a button.
    #[must_use]
    pub fn with_button(mut self, button: TouchButton) -> TouchLayout {
        self.buttons.push(button);
        self
    }

    /// The index of the button a touch lands on, if any.
    ///
    /// Later buttons win, so a layout can place a small button over a larger
    /// one and have the small one take the press.
    #[must_use]
    pub fn button_at(&self, position: Vec2) -> Option<usize> {
        self.buttons
            .iter()
            .rposition(|button| button.contains(position))
    }
}

/// Tracks fingers and resolves them into a stick direction and pressed
/// actions.
#[derive(Clone, Debug, Default)]
pub struct TouchState {
    /// What each active finger has claimed, and where it is now.
    active: BTreeMap<TouchId, (Claim, Vec2)>,
    /// Actions held this tick.
    pressed: Vec<Action>,
    /// The stick's current direction.
    movement: Vec2,
}

impl TouchState {
    /// An empty tracker.
    #[must_use]
    pub fn new() -> TouchState {
        TouchState::default()
    }

    /// Applies one platform event against a layout.
    ///
    /// A finger's claim is decided when it goes down and kept until it lifts,
    /// so sliding off a button does not silently release it and sliding off
    /// the stick does not drop it.
    pub fn handle(&mut self, event: TouchEvent, layout: &TouchLayout) {
        match event.phase {
            TouchPhase::Started => {
                let claim = if layout.stick.accepts(event.position) {
                    Claim::Stick
                } else if let Some(index) = layout.button_at(event.position) {
                    Claim::Button(index)
                } else {
                    Claim::None
                };
                self.active.insert(event.id, (claim, event.position));
            }
            TouchPhase::Moved => {
                if let Some(entry) = self.active.get_mut(&event.id) {
                    entry.1 = event.position;
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.active.remove(&event.id);
            }
        }
        self.resolve(layout);
    }

    /// Forgets every finger.
    ///
    /// Called when the app loses focus: Android does not send a lift for a
    /// touch that was in progress when the activity paused, so without this a
    /// player returning to the game finds themselves walking into a wall.
    pub fn clear(&mut self) {
        self.active.clear();
        self.pressed.clear();
        self.movement = Vec2::ZERO;
    }

    /// Recomputes the resolved state from the active fingers.
    fn resolve(&mut self, layout: &TouchLayout) {
        self.pressed.clear();
        self.movement = Vec2::ZERO;

        for (claim, position) in self.active.values() {
            match claim {
                Claim::Stick => self.movement = layout.stick.direction(*position),
                Claim::Button(index) => {
                    if let Some(button) = layout.buttons.get(*index) {
                        if !self.pressed.contains(&button.action) {
                            self.pressed.push(button.action);
                        }
                    }
                }
                Claim::None => {}
            }
        }
    }

    /// The stick's direction this tick.
    #[must_use]
    pub const fn movement(&self) -> Vec2 {
        self.movement
    }

    /// The actions held this tick.
    #[must_use]
    pub fn pressed(&self) -> &[Action] {
        &self.pressed
    }

    /// True when an action is held by a finger.
    #[must_use]
    pub fn is_pressed(&self, action: Action) -> bool {
        self.pressed.contains(&action)
    }

    /// How many fingers are down.
    #[must_use]
    pub fn touch_count(&self) -> usize {
        self.active.len()
    }

    /// Whether any finger is on the stick.
    #[must_use]
    pub fn stick_held(&self) -> bool {
        self.active
            .values()
            .any(|(claim, _)| *claim == Claim::Stick)
    }

    /// Where the finger controlling the stick currently is.
    ///
    /// The HUD draws the thumb here, which is what makes the stick feel
    /// attached to the player's finger rather than painted on the glass.
    #[must_use]
    pub fn stick_position(&self) -> Option<Vec2> {
        self.active
            .values()
            .find(|(claim, _)| *claim == Claim::Stick)
            .map(|(_, position)| *position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions;
    use verdant_core_math::fx;

    /// A layout roughly like the game's: a stick on the left, two buttons on
    /// the right.
    fn layout() -> TouchLayout {
        TouchLayout::new(Thumbstick::new(Vec2::from_ints(60, 200), fx(32)))
            .with_button(TouchButton::new(
                actions::USE,
                Rect::from_ints(380, 180, 48, 48),
            ))
            .with_button(TouchButton::new(
                actions::INTERACT,
                Rect::from_ints(320, 210, 48, 48),
            ))
    }

    /// A touch event, tersely.
    fn touch(id: u64, x: i32, y: i32, phase: TouchPhase) -> TouchEvent {
        TouchEvent {
            id: TouchId(id),
            position: Vec2::from_ints(x, y),
            phase,
        }
    }

    #[test]
    fn a_touch_on_the_stick_produces_movement() {
        let layout = layout();
        let mut state = TouchState::new();
        // Straight down from the centre, at the edge of the stick.
        state.handle(touch(1, 60, 232, TouchPhase::Started), &layout);
        assert_eq!(state.movement(), Vec2::DOWN);
        assert!(state.stick_held());
    }

    #[test]
    fn a_partly_pushed_stick_walks_slowly() {
        // Otherwise every touch is a sprint, and fine positioning on a phone
        // becomes impossible.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 216, TouchPhase::Started), &layout);
        let length = state.movement().length();
        assert!(
            length > fx(0) && length < Fx::ONE,
            "half deflection gave {length}"
        );
    }

    #[test]
    fn a_thumb_dragged_past_the_edge_keeps_full_deflection() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 200, TouchPhase::Started), &layout);
        state.handle(touch(1, 60, 400, TouchPhase::Moved), &layout);
        let movement = state.movement();
        assert!((movement.length() - Fx::ONE).abs() < fx(1) / fx(100));
        assert!(movement.y > Fx::ZERO, "should still be pushing down");
    }

    #[test]
    fn the_stick_never_exceeds_full_deflection() {
        let layout = layout();
        for (x, y) in [(0, 0), (480, 270), (60, 260), (-500, -500)] {
            let mut state = TouchState::new();
            state.handle(touch(1, 60, 200, TouchPhase::Started), &layout);
            state.handle(touch(1, x, y, TouchPhase::Moved), &layout);
            assert!(
                state.movement().length() <= Fx::ONE + fx(1) / fx(100),
                "({x}, {y}) gave {}",
                state.movement().length()
            );
        }
    }

    #[test]
    fn the_stick_is_forgiving_about_where_a_thumb_lands() {
        // A stick that must be hit exactly is a stick that is missed.
        let layout = layout();
        let mut state = TouchState::new();
        // Outside the radius but inside the grab margin.
        state.handle(touch(1, 60, 250, TouchPhase::Started), &layout);
        assert!(state.stick_held());
    }

    #[test]
    fn a_touch_far_from_everything_claims_nothing() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 240, 20, TouchPhase::Started), &layout);
        assert_eq!(state.movement(), Vec2::ZERO);
        assert!(state.pressed().is_empty());
        assert_eq!(state.touch_count(), 1, "but it is still tracked");
    }

    #[test]
    fn a_touch_on_a_button_presses_it() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 400, 200, TouchPhase::Started), &layout);
        assert!(state.is_pressed(actions::USE));
        assert!(!state.is_pressed(actions::INTERACT));
    }

    #[test]
    fn lifting_a_finger_releases_its_button() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 400, 200, TouchPhase::Started), &layout);
        state.handle(touch(1, 400, 200, TouchPhase::Ended), &layout);
        assert!(!state.is_pressed(actions::USE));
        assert_eq!(state.touch_count(), 0);
    }

    #[test]
    fn a_cancelled_touch_releases_too() {
        // Android cancels touches when a system gesture takes over; a button
        // stuck down after that would be held forever.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 400, 200, TouchPhase::Started), &layout);
        state.handle(touch(1, 400, 200, TouchPhase::Cancelled), &layout);
        assert!(!state.is_pressed(actions::USE));
    }

    #[test]
    fn a_finger_keeps_its_button_when_it_slides_off() {
        // Thumbs move while pressing. Releasing on the slightest drift makes
        // a button feel broken.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 400, 200, TouchPhase::Started), &layout);
        state.handle(touch(1, 200, 100, TouchPhase::Moved), &layout);
        assert!(state.is_pressed(actions::USE), "the press should survive");
    }

    #[test]
    fn a_finger_that_started_on_nothing_cannot_press_by_sliding_on() {
        // The mirror of the rule above, and the reason claims are decided at
        // touch-down: a thumb resting on the stick sliding under a button
        // must not fire it.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 240, 20, TouchPhase::Started), &layout);
        state.handle(touch(1, 400, 200, TouchPhase::Moved), &layout);
        assert!(!state.is_pressed(actions::USE));
    }

    #[test]
    fn moving_and_acting_at_once_works() {
        // The whole point of tracking fingers separately: walking while
        // swinging a tool is the normal way to play.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 232, TouchPhase::Started), &layout);
        state.handle(touch(2, 400, 200, TouchPhase::Started), &layout);
        assert_eq!(state.movement(), Vec2::DOWN);
        assert!(state.is_pressed(actions::USE));
    }

    #[test]
    fn a_second_finger_does_not_steal_the_stick() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 232, TouchPhase::Started), &layout);
        state.handle(touch(2, 400, 200, TouchPhase::Started), &layout);
        state.handle(touch(2, 400, 210, TouchPhase::Moved), &layout);
        assert_eq!(state.movement(), Vec2::DOWN, "the stick should be unmoved");
    }

    #[test]
    fn lifting_the_stick_finger_stops_movement() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 232, TouchPhase::Started), &layout);
        state.handle(touch(1, 60, 232, TouchPhase::Ended), &layout);
        assert_eq!(state.movement(), Vec2::ZERO);
        assert!(!state.stick_held());
    }

    #[test]
    fn two_fingers_on_the_same_button_press_it_once() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 400, 200, TouchPhase::Started), &layout);
        state.handle(touch(2, 410, 210, TouchPhase::Started), &layout);
        assert_eq!(state.pressed().len(), 1);
    }

    #[test]
    fn overlapping_buttons_give_the_press_to_the_later_one() {
        let layout = TouchLayout::new(Thumbstick::new(Vec2::from_ints(0, 0), fx(1)))
            .with_button(TouchButton::new(
                actions::USE,
                Rect::from_ints(100, 100, 80, 80),
            ))
            .with_button(TouchButton::new(
                actions::INTERACT,
                Rect::from_ints(120, 120, 20, 20),
            ));
        let mut state = TouchState::new();
        state.handle(touch(1, 125, 125, TouchPhase::Started), &layout);
        assert!(state.is_pressed(actions::INTERACT));
        assert!(!state.is_pressed(actions::USE));
    }

    #[test]
    fn clearing_releases_everything() {
        // Android does not send a lift for a touch in progress when the
        // activity pauses.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 232, TouchPhase::Started), &layout);
        state.handle(touch(2, 400, 200, TouchPhase::Started), &layout);
        state.clear();
        assert_eq!(state.touch_count(), 0);
        assert_eq!(state.movement(), Vec2::ZERO);
        assert!(state.pressed().is_empty());
    }

    #[test]
    fn a_move_for_an_unknown_finger_is_ignored() {
        // Events can arrive out of order across an activity pause.
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(9, 60, 232, TouchPhase::Moved), &layout);
        assert_eq!(state.touch_count(), 0);
        assert_eq!(state.movement(), Vec2::ZERO);
    }

    #[test]
    fn the_stick_position_follows_the_finger() {
        let layout = layout();
        let mut state = TouchState::new();
        assert_eq!(state.stick_position(), None);
        state.handle(touch(1, 60, 200, TouchPhase::Started), &layout);
        state.handle(touch(1, 70, 210, TouchPhase::Moved), &layout);
        assert_eq!(state.stick_position(), Some(Vec2::from_ints(70, 210)));
    }

    #[test]
    fn a_stick_at_its_centre_reads_as_neutral() {
        let layout = layout();
        let mut state = TouchState::new();
        state.handle(touch(1, 60, 200, TouchPhase::Started), &layout);
        assert_eq!(state.movement(), Vec2::ZERO);
    }

    #[test]
    fn a_degenerate_stick_does_not_divide_by_zero() {
        let stick = Thumbstick::new(Vec2::ZERO, Fx::ZERO);
        assert_eq!(stick.direction(Vec2::from_ints(5, 5)), Vec2::ZERO);
    }
}
