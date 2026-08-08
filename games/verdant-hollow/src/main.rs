//! Verdant Hollow's executable: a window, a frame loop and the input that
//! drives the simulation.
//!
//! Everything below is presentation. The simulation lives in the library and
//! never learns that a window exists, which is what lets an entire in-game
//! year run in a test in a few milliseconds.
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

use std::sync::Arc;
use std::time::Instant;

use verdant_core_math::Fx;
use verdant_hollow::render::{Art, GameRenderer, Scene};
use verdant_hollow::sim::{Game, SaveData, SAVE_VERSION};
use verdant_input::{actions, Binding, DeviceState, InputMap, InputState, KeyCode};
use verdant_render_2d::SurfaceContext;
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

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("verdant_hollow=info,warn"),
    )
    .init();

    let event_loop = match EventLoop::new() {
        Ok(loop_) => loop_,
        Err(error) => {
            // A headless machine has no display server; say so plainly rather
            // than panicking with a backtrace.
            eprintln!("could not open a window: {error}");
            eprintln!(
                "Verdant Hollow needs a display. To see the game without one, run:\n  \
                 cargo run -p verdant-hollow --example screenshot"
            );
            std::process::exit(1);
        }
    };
    // Poll rather than Wait: the game animates continuously, so there is
    // always another frame to draw.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("the event loop stopped: {error}");
        std::process::exit(1);
    }
}

/// Everything the window owns.
struct App {
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
}

/// The parts that only exist once a window does.
struct Active {
    window: Arc<Window>,
    surface: SurfaceContext,
    renderer: GameRenderer,
}

impl App {
    /// Builds the game and its art, with no window yet.
    fn new() -> App {
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
        }
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

        let plan = self.timestep.advance(dt);
        let step = self.timestep.step_seconds();
        for _ in 0..plan.steps {
            self.input.update(&self.map, &self.devices);
            self.simulate(step);
        }

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
        let slot = SaveSlot::new(save_path());
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
        let slot = SaveSlot::new(save_path());
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

/// The save file's full path.
fn save_path() -> std::path::PathBuf {
    save_directory().join(format!("{SAVE_NAME}.save"))
}

/// Where saves live.
///
/// `VERDANT_SAVE_DIR` overrides it; otherwise the platform's data directory,
/// falling back to the working directory when the environment says nothing.
fn save_directory() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("VERDANT_SAVE_DIR") {
        return std::path::PathBuf::from(path);
    }
    let base = std::env::var("XDG_DATA_HOME")
        .or_else(|_| std::env::var("APPDATA"))
        .map(std::path::PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME").map(|home| std::path::PathBuf::from(home).join(".local/share"))
        })
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    base.join("verdant-hollow")
}
