//! The window, the frame loop and the input that drives the simulation.
//!
//! Everything here is presentation. The simulation in [`crate::sim`] never
//! learns that a window exists, which is what lets an entire in-game year run
//! in a test in a few milliseconds.
//!
//! This lives in the library rather than in `main.rs` because there are two
//! entry points into it — a desktop binary and an Android activity — and they
//! differ only in how the platform hands over control.
//!
//! ## The Android lifecycle
//!
//! Android destroys the native window whenever the activity is backgrounded
//! and creates a new one on return. Anything built against the old window —
//! the swapchain, and every pipeline compiled for its format — is invalid
//! afterwards. So the window-bound state is grouped into one value that is
//! dropped on `Suspended` and rebuilt on `Resumed` — grouped rather than held
//! as separate options, because then it is impossible to keep half of it by
//! accident. The game state is deliberately not part of it: a player who
//! takes a phone call comes back to the same day on the same farm.
//!
//! ## The loop
//!
//! Rendering is per frame; simulation is on a fixed 60 Hz step. A slow frame
//! runs several steps to catch up, capped so a stalled machine cannot spiral
//! into an ever-growing backlog. That separation is what keeps the game the
//! same speed on a 60 Hz laptop and a 240 Hz desktop, and what makes replays
//! reproduce exactly.
//!
//! ## Controls
//!
//! | Key | Action |
//! |---|---|
//! | WASD / arrows | Walk |
//! | Shift | Sprint |
//! | Space | Use the selected tool or seed |
//! | E | Interact: talk, give, harvest, ship, sleep |
//! | 1–6 | Select a hotbar slot |
//! | `[` `]` | Cycle hotbar slots |
//! | F5 / F9 | Save / load |
//! | Escape | Quit |

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::render::{Art, GameRenderer, Scene, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use crate::sim::{Game, SaveData, SAVE_VERSION};
use std::sync::Arc as StdArc;
use verdant_audio::{music, AudioDevice, Bus, PlaySettings, Sound};
use verdant_core_math::Fx;
use verdant_input::{
    actions, Binding, DeviceState, InputMap, InputState, KeyCode, TouchEvent, TouchId, TouchPhase,
    TouchState,
};
use verdant_render_2d::{window_to_internal, SurfaceContext};
use verdant_runtime::Timestep;
use verdant_save::{MigrationChain, SaveSlot};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

/// The action that puts the player to bed.
const SLEEP: verdant_input::Action = verdant_input::Action("sleep");

/// Where saves are written, relative to the user's data directory.
const SAVE_NAME: &str = "verdant-hollow";

/// Everything the game owns, across the whole life of the process.
pub struct App {
    /// `None` until the platform hands us a window.
    active: Option<Active>,
    /// The simulation, created before the window so a failure to open a
    /// display does not lose a world.
    game: Game,
    art: Art,
    scene: Scene,
    timestep: Timestep,
    input: InputState,
    map: InputMap,
    devices: DeviceState,
    /// When the last frame was drawn, for the real-time delta.
    last_frame: Instant,
    /// Fingers on the screen, resolved against the on-screen controls.
    touch: TouchState,
    /// Sound output, or a silent stand-in when the device has none.
    audio: AudioDevice,
    /// The generated track for the current season, and which season it is for.
    ///
    /// Regenerated when the season turns rather than every frame: composing
    /// and rendering eight bars costs a few milliseconds, which is nothing
    /// once a season but far too much sixty times a second.
    track: Option<(crate::calendar::Season, StdArc<Sound>)>,
    /// Where saves are written.
    save_directory: PathBuf,
}

/// The parts that are tied to a particular native window.
///
/// Dropped whenever Android takes the window away, which is why they are
/// grouped rather than held as separate `Option`s: it is impossible to keep
/// half of them by accident.
struct Active {
    window: Arc<Window>,
    surface: SurfaceContext,
    renderer: GameRenderer,
}

impl App {
    /// Builds the game and its art, with no window yet.
    ///
    /// Saves are written under `save_directory`, which the platform decides:
    /// a data directory on desktop, and the activity's private storage on
    /// Android, where nothing else is writable.
    #[must_use]
    pub fn new(save_directory: PathBuf) -> App {
        let seed = seed_from_environment();
        let game = Game::new(seed);
        let seeds: Vec<u64> = game
            .villagers
            .iter()
            .map(|villager| villager.appearance_seed)
            .collect();
        let art = Art::generate(game.seed, &seeds);

        let mut map = InputMap::with_defaults();
        // The engine's defaults cover movement and the shared verbs; these are
        // the game's own.
        map.bind(SLEEP, Binding::Key(KeyCode::KeyB));
        map.bind(actions::USE, Binding::Key(KeyCode::KeyC));

        let mut scene = Scene::new();
        scene.snap_to(&game);

        App {
            active: None,
            game,
            art,
            scene,
            timestep: Timestep::standard(),
            input: InputState::new(),
            map,
            devices: DeviceState::default(),
            last_frame: Instant::now(),
            touch: TouchState::new(),
            // Sound is a nicety: a machine without it should still play, so
            // failure here degrades to silence rather than stopping the game.
            audio: AudioDevice::open_or_silent(),
            track: None,
            save_directory,
        }
    }

    /// Shows or hides the on-screen touch controls.
    ///
    /// Set by the entry point rather than guessed at: a desktop player should
    /// never see a thumbstick they cannot use, and a phone must always have
    /// one.
    pub fn set_touch_controls(&mut self, shown: bool) {
        self.scene.show_touch_controls = shown;
    }

    /// Runs the game until the window closes.
    ///
    /// # Errors
    ///
    /// Returns the event loop's error if it stops abnormally.
    pub fn run(mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        // Poll rather than Wait: the game animates continuously, so there is
        // always another frame to draw.
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut self)
    }

    /// Runs the simulation for however many fixed steps are due, then draws.
    fn frame(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        self.last_frame = now;

        // Clamped before it reaches the timestep: a frame that took a whole
        // second — a window drag, a laptop waking up — should drop the time,
        // not simulate a second of the player walking into a wall.
        let dt = seconds(elapsed.as_secs_f64().min(0.25));

        // The touch layer resolves fingers into a direction and a set of
        // actions; from here down they are indistinguishable from a gamepad.
        self.devices.touch_movement = self.touch.movement();
        self.devices.touch_actions.clear();
        self.devices
            .touch_actions
            .extend_from_slice(self.touch.pressed());

        let plan = self.timestep.advance(dt);
        let step = self.timestep.step_seconds();
        for _ in 0..plan.steps {
            self.input.update(&self.map, &self.devices);
            self.simulate(step);
            // Weather advances on the fixed step with everything else, so a
            // replay's rain matches the recording's.
            self.scene.update_weather(&self.game, &self.art, step);
        }

        self.update_music();
        self.audio.collect();

        self.scene.observe_touch(&self.touch);
        self.scene.follow(&self.game, dt);
        self.draw();
    }

    /// One fixed simulation step, including whatever the player asked for.
    fn simulate(&mut self, step: Fx) {
        let movement = self.input.movement();
        let sprinting = self.input.is_down(actions::SPRINT);
        self.game.update(movement, sprinting, step);

        // Buffered rather than `just_pressed`: a tool press a few frames early
        // should still land, which is the difference between a game that feels
        // responsive and one that feels like it ignores you.
        if self.input.consume_buffered(actions::USE) {
            self.game.use_selected();
        }
        if self.input.consume_buffered(actions::INTERACT) {
            self.game.interact();
        }
        if self.input.consume_buffered(SLEEP) {
            self.game.sleep();
        }
        if self.input.just_pressed(actions::NEXT_SLOT) {
            self.game.player.inventory.cycle_selection(true);
        }
        if self.input.just_pressed(actions::PREVIOUS_SLOT) {
            self.game.player.inventory.cycle_selection(false);
        }
    }

    /// Draws one frame, skipping it if the swapchain is not ready.
    fn draw(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let Some(frame) = active.surface.acquire() else {
            // Occluded, resizing or rebuilt: try again next frame.
            return;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        active.renderer.draw(
            &mut self.scene,
            &self.game,
            &self.art,
            &view,
            active.surface.size(),
        );
        // Pre-present notify lets the compositor time the flip; without it the
        // frame is presented late on Wayland.
        active.window.pre_present_notify();
        active.surface.present(frame);
    }

    /// Starts the season's music, generating it the first time it is needed.
    ///
    /// Each season gets its own key and scale, so the valley sounds different
    /// in autumn than in spring without a single note being written by hand.
    fn update_music(&mut self) {
        use crate::calendar::Season;

        let season = self.game.calendar.season();
        if self
            .track
            .as_ref()
            .is_some_and(|(current, _)| *current == season)
        {
            return;
        }

        let theme = match season {
            // Bright and open for the growing seasons; darker and slower as
            // the year closes in.
            Season::Spring => music::Theme {
                root: 62,
                scale: music::Scale::MajorPentatonic,
                tempo: 104.0,
                ..music::Theme::default()
            },
            Season::Summer => music::Theme {
                root: 67,
                scale: music::Scale::Major,
                tempo: 112.0,
                ..music::Theme::default()
            },
            Season::Autumn => music::Theme {
                root: 60,
                scale: music::Scale::Dorian,
                tempo: 88.0,
                ..music::Theme::default()
            },
            Season::Winter => music::Theme {
                root: 57,
                scale: music::Scale::Minor,
                tempo: 76.0,
                ..music::Theme::default()
            },
        };

        // Seeded from the world and the season, so a given valley always
        // sounds like itself.
        let seed = self
            .game
            .seed
            .wrapping_mul(31)
            .wrapping_add(u64::from(season.index()));
        let sound = StdArc::new(music::generate(&theme, seed));

        self.audio.with_mixer(|mixer| mixer.stop_bus(Bus::Music));
        self.audio.play(
            StdArc::clone(&sound),
            PlaySettings::on(Bus::Music).with_gain(0.5).looping(),
        );
        self.track = Some((season, sound));
        log::info!("playing the {} theme", season.name());
    }

    /// The save file's full path.
    fn save_path(&self) -> PathBuf {
        self.save_directory.join(format!("{SAVE_NAME}.save"))
    }

    /// Routes a touch event through the on-screen control layout.
    ///
    /// Window coordinates are mapped onto the low-resolution frame first,
    /// because that is the space the controls are laid out in. A touch in the
    /// letterbox maps to nothing and is dropped rather than clamped to the
    /// nearest edge, which would fire whichever control happens to be there.
    fn touch(&mut self, event: &winit::event::Touch, window: (u32, u32)) {
        let Some(position) = window_to_internal(
            (INTERNAL_WIDTH, INTERNAL_HEIGHT),
            window,
            (event.location.x, event.location.y),
        ) else {
            return;
        };

        let phase = match event.phase {
            winit::event::TouchPhase::Started => TouchPhase::Started,
            winit::event::TouchPhase::Moved => TouchPhase::Moved,
            winit::event::TouchPhase::Ended => TouchPhase::Ended,
            winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
        };

        let layout = self.scene.touch.clone();
        self.touch.handle(
            TouchEvent {
                id: TouchId(event.id),
                position,
                phase,
            },
            &layout,
        );
    }

    /// Applies a key press or release to the device snapshot.
    fn key(&mut self, code: winit::keyboard::KeyCode, pressed: bool) {
        let Some(key) = translate_key(code) else {
            return;
        };
        if pressed {
            if !self.devices.keys.contains(&key) {
                self.devices.keys.push(key);
            }
            self.hotbar_shortcut(key);
        } else {
            self.devices.keys.retain(|held| *held != key);
        }
    }

    /// Number keys select a hotbar slot directly.
    ///
    /// Handled here rather than as bound actions because there are six of them
    /// and each maps to a distinct index, which a boolean action cannot carry.
    fn hotbar_shortcut(&mut self, key: KeyCode) {
        let slot = match key {
            KeyCode::Digit1 => 0,
            KeyCode::Digit2 => 1,
            KeyCode::Digit3 => 2,
            KeyCode::Digit4 => 3,
            KeyCode::Digit5 => 4,
            KeyCode::Digit6 => 5,
            _ => return,
        };
        self.game.player.inventory.select(slot);
    }

    /// Writes the world to disk.
    fn save(&self) {
        let slot = SaveSlot::new(self.save_path());
        match slot.write(&self.game.save_data(), SAVE_VERSION) {
            Ok(()) => log::info!("saved to {}", slot.path().display()),
            Err(error) => log::error!("could not save: {error}"),
        }
    }

    /// Reads the world back, leaving the current one alone on failure.
    ///
    /// A corrupt or missing save must never cost the player the session they
    /// are in the middle of, so the new game replaces the old one only once it
    /// has been read successfully.
    fn load(&mut self) {
        let slot = SaveSlot::new(self.save_path());
        // No migrations yet: version 1 is the first format. The chain is here
        // so that adding one is a data change rather than a code change.
        let chain = MigrationChain::new(SAVE_VERSION);
        match slot.read::<SaveData>(&chain) {
            Ok(data) => {
                self.game = Game::from_save(data);
                self.scene.snap_to(&self.game);
                log::info!("loaded {}", slot.path().display());
            }
            Err(error) => log::warn!("could not load: {error}"),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Verdant Hollow")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_min_inner_size(winit::dpi::LogicalSize::new(480, 270));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("could not create a window: {error}");
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        let surface = match SurfaceContext::new(Arc::clone(&window), size.width, size.height) {
            Ok(surface) => surface,
            Err(error) => {
                eprintln!("could not start the renderer: {error}");
                event_loop.exit();
                return;
            }
        };
        log::info!("rendering on {}", surface.gpu.describe());

        let renderer = match GameRenderer::new(&surface.gpu, &self.art, surface.format()) {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("could not upload the atlas: {error}");
                event_loop.exit();
                return;
            }
        };

        self.last_frame = Instant::now();
        self.active = Some(Active {
            window,
            surface,
            renderer,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(active) = self.active.as_mut() {
                    active.surface.resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(false) => {
                // Otherwise a key held at the moment of alt-tab stays held.
                self.devices.keys.clear();
                self.input.release_all();
                // Android does not send a lift for a touch in progress when
                // the activity loses focus, so a finger down at that moment
                // would keep walking forever.
                self.touch.clear();
            }
            WindowEvent::Touch(touch) => {
                let size = self
                    .active
                    .as_ref()
                    .map_or((INTERNAL_WIDTH, INTERNAL_HEIGHT), |active| {
                        active.surface.size()
                    });
                self.touch(&touch, size);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                // Key repeat must not re-trigger an action; only the first
                // press counts.
                if pressed && event.repeat {
                    return;
                }
                match code {
                    winit::keyboard::KeyCode::Escape if pressed => event_loop.exit(),
                    winit::keyboard::KeyCode::F5 if pressed => self.save(),
                    winit::keyboard::KeyCode::F9 if pressed => self.load(),
                    other => self.key(other, pressed),
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Android is about to destroy the native window. Everything built
        // against it — the swapchain and every pipeline compiled for its
        // format — has to go with it; the game state stays, so a player who
        // takes a phone call comes back to the same day on the same farm.
        log::info!("suspended: releasing the window and renderer");
        self.active = None;
        self.touch.clear();
        self.devices.keys.clear();
        self.input.release_all();
        // The clock would otherwise carry the whole backgrounded duration
        // into the next frame and simulate it in one go.
        self.timestep.discard_backlog();
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}

/// Converts winit's key code into the engine's.
///
/// Only the keys the game binds are translated; anything else is ignored
/// rather than mapped approximately, so an unbound key never acts as a
/// different one.
fn translate_key(code: winit::keyboard::KeyCode) -> Option<KeyCode> {
    use winit::keyboard::KeyCode as W;
    Some(match code {
        W::KeyA => KeyCode::KeyA,
        W::KeyB => KeyCode::KeyB,
        W::KeyC => KeyCode::KeyC,
        W::KeyD => KeyCode::KeyD,
        W::KeyE => KeyCode::KeyE,
        W::KeyQ => KeyCode::KeyQ,
        W::KeyS => KeyCode::KeyS,
        W::KeyW => KeyCode::KeyW,
        W::Digit1 => KeyCode::Digit1,
        W::Digit2 => KeyCode::Digit2,
        W::Digit3 => KeyCode::Digit3,
        W::Digit4 => KeyCode::Digit4,
        W::Digit5 => KeyCode::Digit5,
        W::Digit6 => KeyCode::Digit6,
        W::ArrowUp => KeyCode::ArrowUp,
        W::ArrowDown => KeyCode::ArrowDown,
        W::ArrowLeft => KeyCode::ArrowLeft,
        W::ArrowRight => KeyCode::ArrowRight,
        W::Space => KeyCode::Space,
        W::Enter => KeyCode::Enter,
        W::Tab => KeyCode::Tab,
        W::ShiftLeft => KeyCode::ShiftLeft,
        W::ShiftRight => KeyCode::ShiftRight,
        W::BracketLeft => KeyCode::BracketLeft,
        W::BracketRight => KeyCode::BracketRight,
        _ => return None,
    })
}

/// Converts a duration in seconds into the simulation's fixed point.
fn seconds(value: f64) -> Fx {
    Fx::from_f64(value)
}

/// The world seed, from `VERDANT_SEED` when set.
///
/// A fixed default rather than a random one: a new player and a bug report
/// should both land in the same valley unless someone asks otherwise.
fn seed_from_environment() -> u64 {
    std::env::var("VERDANT_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_260_808)
}

/// Where saves live on this platform.
///
/// `VERDANT_SAVE_DIR` overrides it everywhere. Otherwise the platform's data
/// directory, falling back to the working directory when the environment says
/// nothing. On Android this is not used: the activity's private storage is the
/// only writable location, and the entry point passes it in.
#[must_use]
pub fn default_save_directory() -> PathBuf {
    if let Ok(path) = std::env::var("VERDANT_SAVE_DIR") {
        return PathBuf::from(path);
    }
    let base = std::env::var("XDG_DATA_HOME")
        .or_else(|_| std::env::var("APPDATA"))
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("verdant-hollow")
}

/// Runs the game on a desktop platform.
///
/// # Errors
///
/// Returns an error when no display is available or the event loop stops
/// abnormally. A machine with no display is a normal condition — a CI runner,
/// a container — and the caller is expected to report it, not panic.
pub fn run_desktop() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(default_save_directory());
    // A desktop player has a keyboard; a thumbstick they cannot use would only
    // be in the way.
    app.set_touch_controls(false);
    app.run(event_loop)?;
    Ok(())
}

/// The Android activity's entry point.
///
/// Named `android_main` and exported unmangled because that is the symbol
/// `android-activity` looks up after loading the shared library. The Rust ABI
/// rather than `extern "C"`: the glue declares it in an `extern "Rust"` block
/// and calls it from Rust, so a C signature here would be both wrong and not
/// FFI-safe, since `AndroidApp` has no C representation.
#[cfg(target_os = "android")]
#[no_mangle]
pub fn android_main(android: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("VerdantHollow"),
    );

    // The activity's private storage is the only location an Android app can
    // reliably write to, and it survives the app being backgrounded and
    // killed — which is exactly what a save file needs.
    let save_directory = android
        .internal_data_path()
        .unwrap_or_else(|| PathBuf::from("."));
    log::info!("saving to {}", save_directory.display());

    let event_loop = match EventLoop::builder().with_android_app(android).build() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            log::error!("could not start the event loop: {error}");
            return;
        }
    };

    let mut app = App::new(save_directory);
    // There is no keyboard on a phone, so the on-screen controls are not
    // optional here.
    app.set_touch_controls(true);
    if let Err(error) = app.run(event_loop) {
        log::error!("the event loop stopped: {error}");
    }
}
