# Verdant Hollow on Android

The game runs on Android as a `NativeActivity`: there is no Java or Kotlin
source in this directory, only a manifest, an icon and a Gradle module that
packages the compiled Rust into an APK.

## Building

```sh
android/build-apk.sh debug      # or: release
```

The script does two things, in this order and no other:

1. **`cargo-ndk`** cross-compiles `verdant-hollow`'s `cdylib` once per ABI and
   writes `libverdant_hollow.so` into `app/src/main/jniLibs/<abi>/`.
2. **Gradle** packages that tree, the manifest and the icon into an APK.

Gradle never invokes Cargo. That is deliberate: a build failure has exactly one
owner, and the error message comes from the tool that caused it rather than
from a Gradle task wrapping it.

### What you need

| Requirement | Notes |
|---|---|
| Android NDK | Found via `ANDROID_NDK_HOME`, or `ANDROID_NDK_LATEST_HOME` as a fallback |
| `cargo-ndk` | `cargo install cargo-ndk --locked` |
| Rust targets | `rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android` |
| Gradle 8.x, JDK 17+ | The wrapper is generated on first run rather than committed, so the repository holds no binary jar |

CI builds the debug APK on every push and keeps it as a `verdant-hollow-debug-apk`
artifact. It is signed with the standard Android debug key, so it installs on
any device or emulator with `adb install`.

### ABIs

`arm64-v8a` covers every phone sold for years. `armeabi-v7a` keeps older
hardware working and costs one more compile. `x86_64` is for the emulator,
which is how most people will first see the game.

## Controls

There is no keyboard, so the game draws its own. The layout lives in
`verdant_input::TouchLayout` and the HUD reads the *same value* it is hit-tested
against — a button whose artwork and hit region disagree is a button the player
presses and nothing happens, and sharing one layout makes that impossible.

| Control | Does |
|---|---|
| Thumbstick, bottom left | Walk. Partial deflection walks slowly |
| **USE** | Swing the selected tool, or plant the selected seed |
| **TALK** | Interact: talk, give, harvest, ship, sleep |
| **RUN** | Sprint while held |

A finger's claim is decided when it goes down and kept until it lifts. That is
what lets a thumb slide off a button without releasing it, stops a second
finger stealing the stick, and prevents a thumb resting on the stick from
firing a button by sliding under it.

## The activity lifecycle

Android destroys the native window whenever the activity is backgrounded and
creates a new one on return. Everything built against the old window — the
swapchain, and every render pipeline compiled for its format — is invalid
afterwards.

`verdant_hollow::app` groups exactly those things into one value that is
dropped on `Suspended` and rebuilt on `Resumed`. They are grouped rather than
held as separate `Option`s so that it is impossible to keep half of them by
accident.

The game state is deliberately *not* part of that group: a player who takes a
phone call comes back to the same day on the same farm. Two smaller things go
with the teardown for the same reason:

- **Touches are cleared.** Android does not send a lift for a finger that was
  down when the activity paused, so without this a returning player finds
  themselves walking into a wall.
- **The timestep's backlog is discarded.** Otherwise the whole backgrounded
  duration arrives as one enormous frame and gets simulated in a single go.

## Saves

Saves go to the activity's private storage (`internal_data_path`), which is the
only location an Android app can reliably write to and which survives the app
being backgrounded and killed. The path is passed in by the entry point rather
than discovered by the game, because on desktop the answer is different and the
simulation should not know which platform it is on.

## Sound

`cpal` reaches Android's audio through [Oboe], which is C++, so building with
sound needs the NDK's C++ toolchain. The `sound` feature (on by default) gates
it:

```sh
# Compiles the Android target on a machine with no NDK C++ toolchain.
cargo check -p verdant-hollow --lib --target aarch64-linux-android --no-default-features
```

With the feature off the mixer is still compiled and the game still runs — it
just plays silence. That is what keeps the Android target checkable without a
full NDK install.

[Oboe]: https://github.com/google/oboe

## API level

`minSdk` is **26** (Android 8.0). That floor is set by audio, not by graphics:
cpal reaches the speakers through Oboe, which links `libaaudio`, and the NDK
sysroot for an earlier API level does not contain it. The level passed to
`cargo-ndk` in `build-apk.sh` must be kept in step with `minSdk` in
`app/build.gradle.kts` for the same reason.

## Permissions

None. The game runs entirely offline and generates all of its art and music
from a seed at startup, so there is nothing to ask for.
