# Verdant

A deterministic 2D game engine written in Rust, and **Verdant Hollow**, the
farming and life sim built with it.

Nothing here loads an asset file. Every tile, character, crop, item icon and
note of music is generated from a seed at startup, which takes a few
milliseconds and means the game has no art directory to ship, no loader to
fail, and no possibility of a sprite going missing at runtime.

## Running it

```sh
cargo run -p verdant-hollow --release          # desktop
android/build-apk.sh debug                     # Android — see android/README.md
```

On a machine with no display — a CI runner, a container — render frames to PNG
instead:

```sh
cargo run -p verdant-hollow --example screenshot -- target/screenshots
```

### Controls

| Desktop | Android | Does |
|---|---|---|
| WASD / arrows | Thumbstick | Walk |
| Shift | **RUN** | Sprint |
| Space | **USE** | Swing the selected tool, or plant the selected seed |
| E | **TALK** | Interact: talk, give, harvest, ship, sleep |
| 1–6, `[` `]` | Hotbar | Select a slot |
| F5 / F9 | — | Save / load |

## The crates

| Crate | What it is |
|---|---|
| `core-math` | Q32.32 fixed point, vectors, geometry, seeded PCG, noise |
| `core-ecs` | Hybrid archetype/sparse-set ECS with safe disjoint queries |
| `physics-2d` | Swept AABB collision and `move_and_slide` |
| `tilemap` | Layers, A\*, flow fields, 47-tile autotiling, wave function collapse |
| `input` | Action maps, rebinding, buffering, coyote time, touch controls |
| `save` | Versioned, checksummed saves with atomic writes and migrations |
| `render-2d` | Instanced wgpu sprite renderer, pixel-perfect pipeline, bitmap font |
| `procgen-art` | Palette-enforced pixel-art generation with provenance records |
| `audio` | Deterministic mixer, procedural synthesis, generated music |
| `runtime` | Fixed-timestep loop and the replay/determinism harness |
| `games/verdant-hollow` | The game |

## Determinism

The same seed produces the same simulation, the same art and the same audio
samples — on Linux, macOS, Windows and Android alike. That is not a happy
accident of the implementation; it is the constraint the engine is built
around, and CI checks it on three operating systems on every push.

Three things make it hold:

- **No floats in the simulation.** Positions, velocities and timers are Q32.32
  fixed point, with `sqrt`, `sin`, `cos` and `atan2` implemented in integer
  arithmetic. Float rounding differs between compilers and architectures;
  integers do not.
- **No libm anywhere it matters.** Audio oscillators read a table built from
  the fixed-point sine, and pitches come from a table of the twelve semitone
  ratios, because `sin` and `powf` disagree between platforms in the last bits.
- **Ordered iteration and seeded randomness.** Queries walk archetypes in a
  defined order, and every random draw comes from a named child stream of the
  world seed, so adding a system cannot shift another system's rolls.

What that buys: a replay is a list of per-tick inputs plus periodic state
hashes, and `Recording::compare` reports the first tick at which two runs
diverge. A bug report can be a seed and an input log.

## Testing

```sh
cargo test --workspace
```

Just under 800 tests, all of which run without a GPU or a display. The renderer is
covered too: `GpuContext::headless` acquires a device with no surface and
`RenderTarget::read_pixels` copies a frame back, so CI exercises the real
shader pipeline against Mesa's lavapipe software Vulkan driver.

Tests assert that a frame has the right *structure*. Only an image shows
whether it looks like anything, so CI also renders the game to PNGs and keeps
them as artifacts — several real defects in this repository were invisible to a
passing test suite and obvious in a screenshot.

## Documentation

```sh
cargo doc --workspace --no-deps --open
```

Every public item is documented, and the docs explain *why* a thing is the way
it is rather than restating its signature. `android/README.md` covers the
Android build, the activity lifecycle and the touch layout.

## Licence

MIT or Apache-2.0, at your option.
