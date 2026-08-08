#!/usr/bin/env bash
#
# Builds an installable Verdant Hollow APK.
#
# Two steps, in this order and no other: cargo-ndk cross-compiles the game into
# a shared library per ABI and writes it into the Gradle module's jniLibs tree,
# then Gradle packages that tree into an APK. Gradle never invokes Cargo, so a
# failure has exactly one owner and the error comes from the tool that caused
# it.
#
# Requirements:
#   * The Android NDK, located via ANDROID_NDK_HOME or ANDROID_NDK_LATEST_HOME.
#   * cargo-ndk         (cargo install cargo-ndk)
#   * The Rust targets  (rustup target add aarch64-linux-android armv7-linux-androideabi)
#   * Gradle 8.x and a JDK 17+.
#
# Usage:
#   android/build-apk.sh [debug|release]

set -euo pipefail

PROFILE="${1:-debug}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JNI_LIBS="$ROOT/android/app/src/main/jniLibs"

case "$PROFILE" in
    debug)   CARGO_PROFILE_FLAG="";          GRADLE_TASK="assembleDebug"   ;;
    release) CARGO_PROFILE_FLAG="--release"; GRADLE_TASK="assembleRelease" ;;
    *) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

if [[ -z "${ANDROID_NDK_HOME:-}" && -n "${ANDROID_NDK_LATEST_HOME:-}" ]]; then
    export ANDROID_NDK_HOME="$ANDROID_NDK_LATEST_HOME"
fi
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "error: set ANDROID_NDK_HOME to an Android NDK installation" >&2
    exit 1
fi
echo "Using NDK at $ANDROID_NDK_HOME"

# Stale libraries from a previous ABI set would be packaged alongside the new
# ones and shipped to devices that cannot load them.
rm -rf "$JNI_LIBS"
mkdir -p "$JNI_LIBS"

# arm64 covers every phone sold for years; armv7 keeps older hardware working
# and costs one more compile. x86_64 is here for the emulator, which is how
# most people will first see the game.
cargo ndk \
    --target arm64-v8a \
    --target armeabi-v7a \
    --target x86_64 \
    --platform 24 \
    --output-dir "$JNI_LIBS" \
    -- build --lib $CARGO_PROFILE_FLAG -p verdant-hollow

echo "Native libraries:"
find "$JNI_LIBS" -name '*.so' -exec ls -lh {} \;

cd "$ROOT/android"
if [[ ! -x ./gradlew ]]; then
    # Generate the wrapper rather than committing a binary jar to the
    # repository; the version is pinned in gradle/wrapper.
    gradle wrapper --gradle-version 8.11.1
fi
./gradlew --no-daemon "$GRADLE_TASK"

echo
echo "APK:"
find "$ROOT/android/app/build/outputs/apk" -name '*.apk' -exec ls -lh {} \;
