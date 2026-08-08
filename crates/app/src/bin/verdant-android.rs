//! Generates and builds an Android application for a game on the engine.
//!
//! ```text
//! verdant-android build  --name "Verdant Hollow" \
//!                        --package dev.verdant.hollow \
//!                        --crate verdant-hollow
//! verdant-android generate --name "My Game" --package com.example.game --crate my-game
//! ```
//!
//! Argument parsing is hand-rolled rather than pulled from a crate: this takes
//! eight flags, and a build tool that is part of the engine should not add a
//! dependency tree to the engine's own build.

use std::path::PathBuf;
use std::process::ExitCode;
use verdant_app::android::{Abi, AndroidApp, Orientation, Profile};

/// What the tool was asked to do.
enum Action {
    /// Write the Gradle project and stop.
    Generate,
    /// Write it, cross-compile and package an APK.
    Build,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The tool's real body, so failures become a message rather than a panic.
fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let action = match arguments.next().as_deref() {
        Some("generate") => Action::Generate,
        Some("build") => Action::Build,
        Some("--help" | "-h") | None => {
            println!("{USAGE}");
            return Ok(());
        }
        Some(other) => return Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };

    let mut name = None;
    let mut package = None;
    let mut crate_name = None;
    let mut lib_name = None;
    let mut out = PathBuf::from("target/android");
    let mut workspace = PathBuf::from(".");
    let mut profile = Profile::Debug;
    let mut orientation = Orientation::Landscape;
    let mut abis: Vec<Abi> = Vec::new();

    while let Some(flag) = arguments.next() {
        // Every flag that takes a value reports its own absence, rather than
        // silently using a default that would produce a wrong APK.
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--name" => name = Some(value()?),
            "--package" => package = Some(value()?),
            "--crate" => crate_name = Some(value()?),
            "--lib" => lib_name = Some(value()?),
            "--out" => out = PathBuf::from(value()?),
            "--workspace" => workspace = PathBuf::from(value()?),
            "--release" => profile = Profile::Release,
            "--portrait" => orientation = Orientation::Portrait,
            "--abi" => {
                let requested = value()?;
                let abi = Abi::ALL
                    .iter()
                    .find(|abi| abi.name() == requested)
                    .copied()
                    .ok_or_else(|| {
                        let known: Vec<&str> = Abi::ALL.iter().map(|abi| abi.name()).collect();
                        format!("unknown ABI {requested:?}; known: {}", known.join(", "))
                    })?;
                abis.push(abi);
            }
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
    }

    let name = name.ok_or("--name is required")?;
    let package = package.ok_or("--package is required")?;
    let crate_name = crate_name.ok_or("--crate is required")?;

    let mut app = AndroidApp::new(name, package, crate_name).with_orientation(orientation);
    if let Some(lib) = lib_name {
        app = app.with_lib_name(lib);
    }
    if !abis.is_empty() {
        app = app.with_abis(abis);
    }
    app.validate().map_err(|error| error.to_string())?;

    match action {
        Action::Generate => {
            app.generate(&out).map_err(|error| error.to_string())?;
            println!("generated {}", out.display());
        }
        Action::Build => {
            let apk = app
                .build(&workspace, &out, profile)
                .map_err(|error| error.to_string())?;
            println!("{}", apk.display());
        }
    }
    Ok(())
}

/// How to use the tool.
const USAGE: &str = "\
verdant-android — package a Verdant game for Android

USAGE:
    verdant-android generate [OPTIONS]   Write the Gradle project
    verdant-android build    [OPTIONS]   Write it, cross-compile and package an APK

REQUIRED:
    --name <NAME>          Name shown under the launcher icon
    --package <ID>         Android package, e.g. com.example.game
    --crate <CRATE>        Cargo package producing the game's cdylib

OPTIONS:
    --lib <NAME>           Library name, if it is not the crate name with
                           hyphens replaced by underscores
    --out <DIR>            Where to write the project [default: target/android]
    --workspace <DIR>      Cargo workspace holding the game [default: .]
    --abi <ABI>            Restrict architectures; repeatable
                           [default: arm64-v8a, armeabi-v7a, x86_64]
    --portrait             Run the game portrait rather than landscape
    --release              Build the release variant

REQUIREMENTS for `build`:
    The Android NDK (ANDROID_NDK_HOME), cargo-ndk, the Rust Android targets,
    Gradle 8.x and a JDK 17 or newer.
";
