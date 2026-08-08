//! # Verdant input
//!
//! Action maps, buffering and rebinding.
//!
//! ## Why actions rather than keys
//!
//! Gameplay asks "did the player try to use a tool this tick?", not "is the C
//! key down?". Routing everything through named [`Action`]s means rebinding,
//! gamepad support and touch controls are all configuration rather than code
//! changes, and it keeps device-specific handling out of gameplay systems
//! entirely.
//!
//! ## Why input is a snapshot
//!
//! [`InputState`] is a value captured once per simulation tick, not a live view
//! of the hardware. That is what makes input *recordable*: a replay is a list
//! of these snapshots, and feeding them back through the same simulation
//! reproduces the session exactly. A system that polled the device directly
//! would break that, so nothing here reads hardware — the platform layer
//! pushes events in, and the simulation only ever sees the snapshot.
//!
//! ## Buffering and leniency
//!
//! Two forms of forgiveness are built in, because both are the difference
//! between controls that feel responsive and controls that feel broken:
//!
//! * **Input buffering** — a press registered slightly *before* the game can
//!   act on it is remembered for a few ticks, so a swing queued during the tail
//!   of the previous animation still fires.
//! * **Coyote time** — a state that has just become false still reads as true
//!   for a few ticks, so an action begun a moment too late still counts.

#![doc(html_no_source)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use verdant_core_math::{Fx, Vec2};

/// A named thing the player can do.
///
/// Actions are `&'static str` rather than an enum so a game can define its own
/// without the engine enumerating them, while staying `Copy` and cheap to
/// compare — they are passed around constantly by gameplay code.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Action(pub &'static str);

impl Action {
    /// The action's name.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.0
    }

    /// Interns a runtime string into an [`Action`].
    ///
    /// Needed because a rebinding profile loaded from disk carries action names
    /// as owned strings, while [`Action`] holds a `&'static str` so it can stay
    /// `Copy`. Each distinct name is leaked exactly once and reused thereafter,
    /// so the memory is bounded by the number of distinct actions a game
    /// defines — a few dozen — rather than by how often profiles are loaded.
    ///
    /// # Panics
    ///
    /// Panics if the interner's lock has been poisoned by a panic in another
    /// thread while interning.
    #[must_use]
    pub fn intern(name: &str) -> Action {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};

        static INTERNED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
        let table = INTERNED.get_or_init(|| Mutex::new(HashSet::new()));
        let mut table = table.lock().expect("the action interner lock was poisoned");
        if let Some(existing) = table.get(name) {
            return Action(existing);
        }
        let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
        table.insert(leaked);
        Action(leaked)
    }
}

impl Serialize for Action {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Action, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(Action::intern(&name))
    }
}

/// The actions the engine itself understands.
///
/// A game is free to ignore these and define its own; they exist so that
/// engine-level features (menus, the debug overlay) have names to bind to.
pub mod actions {
    use super::Action;

    /// Confirm, interact, talk.
    pub const INTERACT: Action = Action("interact");
    /// Cancel, close, back out.
    pub const CANCEL: Action = Action("cancel");
    /// Use the held tool or item.
    pub const USE: Action = Action("use");
    /// Open the inventory.
    pub const INVENTORY: Action = Action("inventory");
    /// Open the pause menu.
    pub const MENU: Action = Action("menu");
    /// Run rather than walk.
    pub const SPRINT: Action = Action("sprint");
    /// Cycle to the next hotbar slot.
    pub const NEXT_SLOT: Action = Action("next_slot");
    /// Cycle to the previous hotbar slot.
    pub const PREVIOUS_SLOT: Action = Action("previous_slot");
}

/// A physical control that can be bound to an action.
///
/// Deliberately not tied to `winit` or any other windowing library: the
/// platform layer translates its own events into these, which keeps the
/// simulation crate free of a windowing dependency and lets the whole input
/// stack be tested without a window.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Binding {
    /// A keyboard key, identified by its physical position.
    ///
    /// Physical rather than logical so that WASD stays in the same place on an
    /// AZERTY keyboard, which is what players expect from movement keys.
    Key(KeyCode),
    /// A mouse button.
    MouseButton(MouseButton),
    /// A gamepad button.
    GamepadButton(GamepadButton),
    /// A gamepad axis pushed past its threshold in one direction.
    GamepadAxis {
        /// Which axis.
        axis: GamepadAxis,
        /// True for the positive direction.
        positive: bool,
    },
}

/// Physical keyboard positions, named for their US-layout legends.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum KeyCode {
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Comma,
    Period,
    Slash,
    Semicolon,
    Quote,
    BracketLeft,
    BracketRight,
    Backslash,
    Minus,
    Equal,
    Backquote,
}

/// Mouse buttons.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

/// Gamepad buttons, named in the Xbox convention.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GamepadButton {
    South,
    East,
    West,
    North,
    LeftShoulder,
    RightShoulder,
    LeftTrigger,
    RightTrigger,
    Select,
    Start,
    LeftStick,
    RightStick,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

/// Gamepad analogue axes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
}

/// Maps actions onto the controls that trigger them.
///
/// An action may have several bindings (WASD *and* the arrow keys *and* the
/// gamepad stick), and a control may drive several actions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputMap {
    /// Bindings per action. `BTreeMap` so serialised profiles are stable and
    /// diffable, and so rebinding UI lists actions in a fixed order.
    bindings: BTreeMap<Action, Vec<Binding>>,
    /// How far a stick must move before it counts as pressed.
    pub axis_threshold: Fx,
    /// Stick movement below this is treated as zero.
    ///
    /// Without a dead zone a worn stick drifts, and a character walks slowly in
    /// one direction whenever the player lets go.
    pub dead_zone: Fx,
}

impl InputMap {
    /// An empty map with sensible analogue thresholds.
    #[must_use]
    pub fn new() -> InputMap {
        InputMap {
            bindings: BTreeMap::new(),
            axis_threshold: Fx::HALF,
            dead_zone: Fx::from_ratio(15, 100),
        }
    }

    /// The engine's default bindings: WASD and arrows to move, plus the usual
    /// gamepad layout.
    #[must_use]
    pub fn with_defaults() -> InputMap {
        let mut map = InputMap::new();
        map.bind(actions::INTERACT, Binding::Key(KeyCode::KeyE));
        map.bind(
            actions::INTERACT,
            Binding::GamepadButton(GamepadButton::South),
        );
        map.bind(actions::USE, Binding::Key(KeyCode::Space));
        map.bind(actions::USE, Binding::MouseButton(MouseButton::Left));
        map.bind(actions::USE, Binding::GamepadButton(GamepadButton::West));
        map.bind(actions::CANCEL, Binding::Key(KeyCode::Escape));
        map.bind(actions::CANCEL, Binding::GamepadButton(GamepadButton::East));
        map.bind(actions::INVENTORY, Binding::Key(KeyCode::Tab));
        map.bind(
            actions::INVENTORY,
            Binding::GamepadButton(GamepadButton::North),
        );
        map.bind(actions::MENU, Binding::Key(KeyCode::Escape));
        map.bind(actions::MENU, Binding::GamepadButton(GamepadButton::Start));
        map.bind(actions::SPRINT, Binding::Key(KeyCode::ShiftLeft));
        map.bind(
            actions::SPRINT,
            Binding::GamepadButton(GamepadButton::LeftStick),
        );
        map.bind(actions::NEXT_SLOT, Binding::Key(KeyCode::BracketRight));
        map.bind(
            actions::NEXT_SLOT,
            Binding::GamepadButton(GamepadButton::RightShoulder),
        );
        map.bind(actions::PREVIOUS_SLOT, Binding::Key(KeyCode::BracketLeft));
        map.bind(
            actions::PREVIOUS_SLOT,
            Binding::GamepadButton(GamepadButton::LeftShoulder),
        );
        map
    }

    /// Adds a binding, ignoring exact duplicates.
    pub fn bind(&mut self, action: Action, binding: Binding) {
        let bindings = self.bindings.entry(action).or_default();
        if !bindings.contains(&binding) {
            bindings.push(binding);
        }
    }

    /// Removes one binding from an action.
    pub fn unbind(&mut self, action: Action, binding: Binding) {
        if let Some(bindings) = self.bindings.get_mut(&action) {
            bindings.retain(|existing| *existing != binding);
        }
    }

    /// Removes every binding for an action.
    pub fn clear_action(&mut self, action: Action) {
        self.bindings.remove(&action);
    }

    /// The bindings for an action.
    #[must_use]
    pub fn bindings_for(&self, action: Action) -> &[Binding] {
        self.bindings.get(&action).map_or(&[], Vec::as_slice)
    }

    /// Every bound action, in a stable order.
    pub fn actions(&self) -> impl Iterator<Item = Action> + '_ {
        self.bindings.keys().copied()
    }

    /// Every action a control currently triggers.
    ///
    /// Rebinding UI uses this to warn about conflicts before committing.
    #[must_use]
    pub fn actions_bound_to(&self, binding: Binding) -> Vec<Action> {
        self.bindings
            .iter()
            .filter(|(_, bindings)| bindings.contains(&binding))
            .map(|(action, _)| *action)
            .collect()
    }
}

/// Raw device state for one tick, before actions are resolved.
///
/// The platform layer fills this in; nothing else should write to it.
#[derive(Clone, Debug, Default)]
pub struct DeviceState {
    /// Keys currently held.
    pub keys: Vec<KeyCode>,
    /// Mouse buttons currently held.
    pub mouse_buttons: Vec<MouseButton>,
    /// Gamepad buttons currently held.
    pub gamepad_buttons: Vec<GamepadButton>,
    /// Gamepad axis positions, in `[-1, 1]`.
    pub gamepad_axes: BTreeMap<GamepadAxis, Fx>,
    /// Cursor position in window pixels.
    pub cursor: Vec2,
    /// Scroll wheel movement since the last tick.
    pub scroll: Fx,
}

impl DeviceState {
    /// True when a key is held.
    #[must_use]
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    /// The position of an axis, or zero when the pad is absent.
    #[must_use]
    pub fn axis(&self, axis: GamepadAxis) -> Fx {
        self.gamepad_axes.get(&axis).copied().unwrap_or(Fx::ZERO)
    }

    /// True when a binding is currently active.
    #[must_use]
    pub fn is_binding_active(&self, binding: Binding, threshold: Fx) -> bool {
        match binding {
            Binding::Key(key) => self.keys.contains(&key),
            Binding::MouseButton(button) => self.mouse_buttons.contains(&button),
            Binding::GamepadButton(button) => self.gamepad_buttons.contains(&button),
            Binding::GamepadAxis { axis, positive } => {
                let value = self.axis(axis);
                if positive {
                    value >= threshold
                } else {
                    value <= -threshold
                }
            }
        }
    }
}

/// How long a press stays buffered, in simulation ticks.
///
/// Six ticks is a tenth of a second at 60 Hz — long enough to absorb the gap
/// between a player's intent and the moment the game can act, short enough that
/// a queued action never feels like it fired on its own.
pub const DEFAULT_BUFFER_TICKS: u32 = 6;

/// Per-action state for one tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ActionState {
    /// Held this tick.
    down: bool,
    /// Held last tick.
    was_down: bool,
    /// Ticks remaining on the buffered press, zero when nothing is buffered.
    buffer: u32,
}

/// The resolved input for one simulation tick.
///
/// This is the only thing gameplay reads, and the only thing a replay records.
#[derive(Clone, Debug, Default)]
pub struct InputState {
    states: BTreeMap<Action, ActionState>,
    /// Movement direction, already dead-zoned and clamped to unit length.
    movement: Vec2,
    cursor: Vec2,
    scroll: Fx,
    buffer_ticks: u32,
}

impl InputState {
    /// Creates empty input with the default buffer window.
    #[must_use]
    pub fn new() -> InputState {
        InputState {
            buffer_ticks: DEFAULT_BUFFER_TICKS,
            ..InputState::default()
        }
    }

    /// Sets how many ticks a press stays buffered.
    pub fn set_buffer_ticks(&mut self, ticks: u32) {
        self.buffer_ticks = ticks;
    }

    /// Advances one tick, resolving `devices` through `map`.
    ///
    /// Call exactly once per simulation tick. Calling it twice would consume
    /// two ticks of buffer for one tick of gameplay, so presses would expire
    /// early.
    pub fn update(&mut self, map: &InputMap, devices: &DeviceState) {
        for action in map.actions() {
            let down = map
                .bindings_for(action)
                .iter()
                .any(|binding| devices.is_binding_active(*binding, map.axis_threshold));

            let state = self.states.entry(action).or_default();
            state.was_down = state.down;
            state.down = down;
            if down && !state.was_down {
                // A fresh press refills the buffer.
                state.buffer = self.buffer_ticks;
            } else {
                state.buffer = state.buffer.saturating_sub(1);
            }
        }

        self.movement = resolve_movement(map, devices);
        self.cursor = devices.cursor;
        self.scroll = devices.scroll;
    }

    /// True while the action is held.
    #[must_use]
    pub fn is_down(&self, action: Action) -> bool {
        self.states.get(&action).is_some_and(|state| state.down)
    }

    /// True on the tick the action was pressed.
    #[must_use]
    pub fn just_pressed(&self, action: Action) -> bool {
        self.states
            .get(&action)
            .is_some_and(|state| state.down && !state.was_down)
    }

    /// True on the tick the action was released.
    #[must_use]
    pub fn just_released(&self, action: Action) -> bool {
        self.states
            .get(&action)
            .is_some_and(|state| !state.down && state.was_down)
    }

    /// True when the action was pressed recently enough to still count.
    ///
    /// This is what gameplay should test for one-shot actions. Unlike
    /// [`InputState::just_pressed`], it forgives a press that arrived while the
    /// character was mid-animation and could not yet act on it.
    #[must_use]
    pub fn is_buffered(&self, action: Action) -> bool {
        self.states
            .get(&action)
            .is_some_and(|state| state.buffer > 0)
    }

    /// Consumes a buffered press so it cannot fire twice.
    ///
    /// Returns whether there was one. Every buffered action *must* be consumed
    /// when acted on, or the same press would trigger on each of the following
    /// buffered ticks.
    pub fn consume_buffered(&mut self, action: Action) -> bool {
        match self.states.get_mut(&action) {
            Some(state) if state.buffer > 0 => {
                state.buffer = 0;
                true
            }
            _ => false,
        }
    }

    /// The movement direction, dead-zoned and never longer than one unit.
    #[must_use]
    pub fn movement(&self) -> Vec2 {
        self.movement
    }

    /// Cursor position in window pixels.
    #[must_use]
    pub fn cursor(&self) -> Vec2 {
        self.cursor
    }

    /// Scroll movement this tick.
    #[must_use]
    pub fn scroll(&self) -> Fx {
        self.scroll
    }

    /// Clears every held state, keeping the bindings.
    ///
    /// Called when the window loses focus, so a key held at the moment of
    /// alt-tab does not stay stuck down while the player is elsewhere.
    pub fn release_all(&mut self) {
        for state in self.states.values_mut() {
            state.was_down = state.down;
            state.down = false;
            state.buffer = 0;
        }
        self.movement = Vec2::ZERO;
        self.scroll = Fx::ZERO;
    }
}

/// Resolves the movement vector from keys and sticks.
///
/// The analogue stick wins when it is pushed past the dead zone, so a player
/// resting a hand on the keyboard does not fight their own gamepad.
fn resolve_movement(map: &InputMap, devices: &DeviceState) -> Vec2 {
    let stick = Vec2::new(
        devices.axis(GamepadAxis::LeftStickX),
        devices.axis(GamepadAxis::LeftStickY),
    );
    if stick.length() > map.dead_zone {
        // Clamp rather than normalise: a partly pushed stick should walk slowly.
        return stick.clamp_length(Fx::ONE);
    }

    let mut direction = Vec2::ZERO;
    if devices.is_key_down(KeyCode::KeyW) || devices.is_key_down(KeyCode::ArrowUp) {
        direction.y -= Fx::ONE;
    }
    if devices.is_key_down(KeyCode::KeyS) || devices.is_key_down(KeyCode::ArrowDown) {
        direction.y += Fx::ONE;
    }
    if devices.is_key_down(KeyCode::KeyA) || devices.is_key_down(KeyCode::ArrowLeft) {
        direction.x -= Fx::ONE;
    }
    if devices.is_key_down(KeyCode::KeyD) || devices.is_key_down(KeyCode::ArrowRight) {
        direction.x += Fx::ONE;
    }
    // Normalising means diagonal movement is not faster than cardinal, which it
    // would be if both components were left at full magnitude.
    direction.normalize()
}

/// Remembers a condition for a few ticks after it stops being true.
///
/// The generalisation of "coyote time": a state that has just lapsed still
/// reads as available, so an action begun a moment too late still counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct Leniency {
    remaining: u32,
    window: u32,
}

impl Leniency {
    /// Creates a leniency window of the given length in ticks.
    #[must_use]
    pub const fn new(window: u32) -> Leniency {
        Leniency {
            remaining: 0,
            window,
        }
    }

    /// Advances one tick with the condition's current value.
    pub fn update(&mut self, condition: bool) {
        if condition {
            self.remaining = self.window;
        } else {
            self.remaining = self.remaining.saturating_sub(1);
        }
    }

    /// True while the condition holds or is still within its grace window.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.remaining > 0
    }

    /// Ends the grace period, so it cannot be used twice.
    pub fn consume(&mut self) {
        self.remaining = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pressing(keys: &[KeyCode]) -> DeviceState {
        DeviceState {
            keys: keys.to_vec(),
            ..DeviceState::default()
        }
    }

    #[test]
    fn an_action_reads_as_down_while_its_key_is_held() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyE]));

        assert!(input.is_down(actions::INTERACT));
        assert!(input.just_pressed(actions::INTERACT));
        assert!(!input.is_down(actions::USE));
    }

    #[test]
    fn just_pressed_only_fires_on_the_first_tick() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        let held = pressing(&[KeyCode::KeyE]);

        input.update(&map, &held);
        assert!(input.just_pressed(actions::INTERACT));
        input.update(&map, &held);
        assert!(
            !input.just_pressed(actions::INTERACT),
            "a held key presses once"
        );
        assert!(input.is_down(actions::INTERACT));
    }

    #[test]
    fn just_released_fires_on_the_tick_after_letting_go() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyE]));
        input.update(&map, &pressing(&[]));

        assert!(input.just_released(actions::INTERACT));
        assert!(!input.is_down(actions::INTERACT));
    }

    #[test]
    fn any_of_an_actions_bindings_triggers_it() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();

        input.update(&map, &pressing(&[KeyCode::Space]));
        assert!(input.is_down(actions::USE), "the keyboard binding works");

        let mouse = DeviceState {
            mouse_buttons: vec![MouseButton::Left],
            ..DeviceState::default()
        };
        input.update(&map, &mouse);
        assert!(input.is_down(actions::USE), "and so does the mouse binding");
    }

    #[test]
    fn a_press_stays_buffered_for_the_configured_window() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.set_buffer_ticks(3);

        input.update(&map, &pressing(&[KeyCode::KeyE]));
        assert!(input.is_buffered(actions::INTERACT));

        // Released immediately, but the press is remembered.
        for tick in 0..2 {
            input.update(&map, &pressing(&[]));
            assert!(
                input.is_buffered(actions::INTERACT),
                "expired at tick {tick}"
            );
        }
        input.update(&map, &pressing(&[]));
        assert!(
            !input.is_buffered(actions::INTERACT),
            "the window should have closed"
        );
    }

    #[test]
    fn consuming_a_buffered_press_stops_it_repeating() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyE]));

        assert!(input.consume_buffered(actions::INTERACT));
        assert!(!input.is_buffered(actions::INTERACT));
        assert!(
            !input.consume_buffered(actions::INTERACT),
            "a press fires once"
        );
    }

    #[test]
    fn movement_keys_produce_a_direction() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();

        input.update(&map, &pressing(&[KeyCode::KeyD]));
        assert_eq!(input.movement(), Vec2::RIGHT);

        input.update(&map, &pressing(&[KeyCode::KeyW]));
        assert_eq!(input.movement(), Vec2::UP);
    }

    #[test]
    fn arrow_keys_are_equivalent_to_wasd() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::ArrowLeft]));
        let arrows = input.movement();
        input.update(&map, &pressing(&[KeyCode::KeyA]));
        assert_eq!(arrows, input.movement());
    }

    #[test]
    fn diagonal_movement_is_not_faster_than_cardinal() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyW, KeyCode::KeyD]));

        let length = input.movement().length().to_f64();
        assert!((length - 1.0).abs() < 1e-5, "diagonal speed was {length}");
    }

    #[test]
    fn opposite_keys_cancel() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyA, KeyCode::KeyD]));
        assert_eq!(input.movement(), Vec2::ZERO);
    }

    #[test]
    fn stick_movement_inside_the_dead_zone_is_ignored() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        let mut devices = DeviceState::default();
        devices
            .gamepad_axes
            .insert(GamepadAxis::LeftStickX, Fx::from_ratio(5, 100));

        input.update(&map, &devices);
        assert_eq!(
            input.movement(),
            Vec2::ZERO,
            "a drifting stick must not walk"
        );
    }

    #[test]
    fn a_partly_pushed_stick_walks_slowly() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        let mut devices = DeviceState::default();
        devices
            .gamepad_axes
            .insert(GamepadAxis::LeftStickX, Fx::HALF);

        input.update(&map, &devices);
        assert_eq!(
            input.movement().x,
            Fx::HALF,
            "analogue input must stay analogue"
        );
    }

    #[test]
    fn a_pushed_stick_is_clamped_to_unit_length() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        let mut devices = DeviceState::default();
        devices
            .gamepad_axes
            .insert(GamepadAxis::LeftStickX, Fx::ONE);
        devices
            .gamepad_axes
            .insert(GamepadAxis::LeftStickY, Fx::ONE);

        input.update(&map, &devices);
        assert!(input.movement().length() <= Fx::ONE + Fx::from_ratio(1, 1000));
    }

    #[test]
    fn an_axis_binding_triggers_past_its_threshold() {
        let mut map = InputMap::new();
        map.bind(
            actions::SPRINT,
            Binding::GamepadAxis {
                axis: GamepadAxis::RightStickY,
                positive: false,
            },
        );
        let mut input = InputState::new();

        let mut devices = DeviceState::default();
        devices
            .gamepad_axes
            .insert(GamepadAxis::RightStickY, Fx::from_ratio(-2, 10));
        input.update(&map, &devices);
        assert!(!input.is_down(actions::SPRINT), "below the threshold");

        devices
            .gamepad_axes
            .insert(GamepadAxis::RightStickY, Fx::from_ratio(-9, 10));
        input.update(&map, &devices);
        assert!(input.is_down(actions::SPRINT));
    }

    #[test]
    fn rebinding_replaces_the_control() {
        let mut map = InputMap::new();
        map.bind(actions::USE, Binding::Key(KeyCode::Space));
        map.unbind(actions::USE, Binding::Key(KeyCode::Space));
        map.bind(actions::USE, Binding::Key(KeyCode::KeyF));

        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::Space]));
        assert!(!input.is_down(actions::USE));
        input.update(&map, &pressing(&[KeyCode::KeyF]));
        assert!(input.is_down(actions::USE));
    }

    #[test]
    fn duplicate_bindings_are_ignored() {
        let mut map = InputMap::new();
        map.bind(actions::USE, Binding::Key(KeyCode::Space));
        map.bind(actions::USE, Binding::Key(KeyCode::Space));
        assert_eq!(map.bindings_for(actions::USE).len(), 1);
    }

    #[test]
    fn binding_conflicts_can_be_discovered() {
        let map = InputMap::with_defaults();
        // Escape is deliberately bound to both cancel and menu.
        let conflicts = map.actions_bound_to(Binding::Key(KeyCode::Escape));
        assert!(conflicts.contains(&actions::CANCEL));
        assert!(conflicts.contains(&actions::MENU));
    }

    #[test]
    fn losing_focus_releases_everything() {
        let map = InputMap::with_defaults();
        let mut input = InputState::new();
        input.update(&map, &pressing(&[KeyCode::KeyE, KeyCode::KeyW]));
        assert!(input.is_down(actions::INTERACT));

        input.release_all();
        assert!(!input.is_down(actions::INTERACT));
        assert!(!input.is_buffered(actions::INTERACT));
        assert_eq!(input.movement(), Vec2::ZERO);
    }

    #[test]
    fn a_map_round_trips_through_serialisation() {
        let map = InputMap::with_defaults();
        let encoded = serde_json::to_string(&map).expect("the map is serialisable");
        let decoded: InputMap = serde_json::from_str(&encoded).expect("and deserialisable");
        assert_eq!(
            decoded.bindings_for(actions::INTERACT),
            map.bindings_for(actions::INTERACT)
        );
    }

    #[test]
    fn leniency_keeps_a_lapsed_condition_available() {
        let mut coyote = Leniency::new(3);
        coyote.update(true);
        assert!(coyote.is_available());

        for _ in 0..2 {
            coyote.update(false);
            assert!(coyote.is_available(), "still within the grace window");
        }
        coyote.update(false);
        assert!(!coyote.is_available());
    }

    #[test]
    fn consuming_leniency_ends_it_immediately() {
        let mut coyote = Leniency::new(5);
        coyote.update(true);
        coyote.consume();
        assert!(!coyote.is_available());
    }

    #[test]
    fn determinism_the_same_event_sequence_resolves_identically() {
        let map = InputMap::with_defaults();
        let sequence = [
            pressing(&[KeyCode::KeyW]),
            pressing(&[KeyCode::KeyW, KeyCode::KeyD]),
            pressing(&[KeyCode::KeyD, KeyCode::KeyE]),
            pressing(&[]),
        ];
        let run = || {
            let mut input = InputState::new();
            let mut trace = Vec::new();
            for devices in &sequence {
                input.update(&map, devices);
                trace.push((
                    input.movement().x.to_raw(),
                    input.movement().y.to_raw(),
                    input.is_down(actions::INTERACT),
                    input.is_buffered(actions::INTERACT),
                ));
            }
            trace
        };
        assert_eq!(run(), run());
    }
}
