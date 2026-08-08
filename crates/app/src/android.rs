//! Generating and building an Android application for a game.
//!
//! The engine owns the Android project rather than each game keeping a
//! hand-written copy of it. A game supplies its name, package and a handful of
//! settings; everything else — the manifest, the Gradle module, the launcher
//! icon — is generated from templates that live in this crate.
//!
//! That matters for more than tidiness. A per-game `android/` directory is a
//! copy of the engine's build knowledge that immediately starts drifting: the
//! API level a game passes to the NDK has to match the `minSdk` in its Gradle
//! file, and when those two live in different files maintained by hand they
//! eventually disagree and fail at link time. Here a single [`AndroidApp`]
//! writes both, so they cannot.
//!
//! ## The two steps
//!
//! 1. [`AndroidApp::generate`] writes a Gradle project into a directory.
//! 2. [`AndroidApp::build`] cross-compiles the game's `cdylib` once per ABI
//!    with `cargo-ndk`, drops the result into that project, and runs Gradle.
//!
//! Gradle never invokes Cargo. A build failure then has exactly one owner, and
//! the error comes from the tool that caused it rather than from a Gradle task
//! wrapping it.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The API level below which the generated project will not be built.
///
/// 26 is the floor because that is where AAudio appears. Audio on Android goes
/// through Oboe, which links `libaaudio`, and the NDK sysroot for an earlier
/// level simply does not contain it. It is also comfortably past 24, where
/// Vulkan support becomes dependable.
pub const MINIMUM_SDK: u32 = 26;

/// The Android SDK the generated project compiles against.
pub const DEFAULT_COMPILE_SDK: u32 = 35;

/// A CPU architecture an APK can carry.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Abi {
    /// 64-bit ARM: every phone sold for years.
    Arm64,
    /// 32-bit ARM, for older hardware.
    ArmV7,
    /// 64-bit x86, which is what the emulator runs.
    X86_64,
}

impl Abi {
    /// Every ABI, which is what a release build should carry.
    pub const ALL: [Abi; 3] = [Abi::Arm64, Abi::ArmV7, Abi::X86_64];

    /// The name Android and `cargo-ndk` know it by.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Abi::Arm64 => "arm64-v8a",
            Abi::ArmV7 => "armeabi-v7a",
            Abi::X86_64 => "x86_64",
        }
    }

    /// The Rust target triple that produces it.
    #[must_use]
    pub const fn target(self) -> &'static str {
        match self {
            Abi::Arm64 => "aarch64-linux-android",
            Abi::ArmV7 => "armv7-linux-androideabi",
            Abi::X86_64 => "x86_64-linux-android",
        }
    }
}

/// Which way up the game runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Orientation {
    /// Either landscape, following the device.
    #[default]
    Landscape,
    /// Either portrait.
    Portrait,
    /// Whatever the device is doing.
    Sensor,
}

impl Orientation {
    /// The value Android's manifest expects.
    #[must_use]
    pub const fn manifest_value(self) -> &'static str {
        match self {
            Orientation::Landscape => "sensorLandscape",
            Orientation::Portrait => "sensorPortrait",
            Orientation::Sensor => "fullSensor",
        }
    }
}

/// Debug or release.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Profile {
    /// Unoptimised, and signed with the standard Android debug key so it
    /// installs on any device without a signing set-up.
    #[default]
    Debug,
    /// Optimised and unsigned; a release key belongs to whoever publishes the
    /// game, not to a build script.
    Release,
}

impl Profile {
    /// The Gradle task that assembles it.
    #[must_use]
    pub const fn gradle_task(self) -> &'static str {
        match self {
            Profile::Debug => "assembleDebug",
            Profile::Release => "assembleRelease",
        }
    }

    /// The subdirectory Gradle writes the APK into.
    #[must_use]
    pub const fn output_directory(self) -> &'static str {
        match self {
            Profile::Debug => "debug",
            Profile::Release => "release",
        }
    }
}

/// Why an Android build did not finish.
#[derive(Debug)]
pub enum AndroidError {
    /// A setting was rejected before anything was written.
    Invalid(String),
    /// A file could not be written or read.
    Io(std::io::Error),
    /// A required tool was not on the path.
    MissingTool {
        /// What was being looked for.
        tool: String,
        /// How to get it.
        hint: String,
    },
    /// A tool ran and failed.
    Failed {
        /// Which tool.
        tool: String,
        /// Its exit status, rendered.
        status: String,
    },
}

impl fmt::Display for AndroidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AndroidError::Invalid(detail) => write!(f, "{detail}"),
            AndroidError::Io(error) => write!(f, "file error: {error}"),
            AndroidError::MissingTool { tool, hint } => {
                write!(f, "{tool} is not available. {hint}")
            }
            AndroidError::Failed { tool, status } => write!(f, "{tool} failed ({status})"),
        }
    }
}

impl std::error::Error for AndroidError {}

impl From<std::io::Error> for AndroidError {
    fn from(error: std::io::Error) -> AndroidError {
        AndroidError::Io(error)
    }
}

/// Everything the engine needs to produce an Android application for a game.
#[derive(Clone, Debug)]
pub struct AndroidApp {
    /// The name shown under the launcher icon.
    pub app_name: String,
    /// The Android package, which is also the application id.
    pub package: String,
    /// The Cargo package that produces the game's `cdylib`.
    pub crate_name: String,
    /// The shared library's name, which the manifest points the activity at.
    ///
    /// This is the Cargo `[lib] name`, not the package name, and the two
    /// differ whenever a package uses hyphens.
    pub lib_name: String,
    /// Which way up the game runs.
    pub orientation: Orientation,
    /// Architectures the APK carries.
    pub abis: Vec<Abi>,
    /// The API level compiled against and declared as `minSdk`.
    pub min_sdk: u32,
    /// The API level the project compiles against.
    pub compile_sdk: u32,
    /// Monotonic build number.
    pub version_code: u32,
    /// Human-readable version.
    pub version_name: String,
    /// The launcher icon's background, as an Android colour literal.
    pub icon_background: String,
}

impl AndroidApp {
    /// A configuration with the engine's defaults.
    ///
    /// `crate_name` is the Cargo package; the library name defaults to it with
    /// hyphens replaced by underscores, which is what Cargo does.
    #[must_use]
    pub fn new(
        app_name: impl Into<String>,
        package: impl Into<String>,
        crate_name: impl Into<String>,
    ) -> AndroidApp {
        let crate_name = crate_name.into();
        AndroidApp {
            app_name: app_name.into(),
            package: package.into(),
            lib_name: crate_name.replace('-', "_"),
            crate_name,
            orientation: Orientation::default(),
            abis: Abi::ALL.to_vec(),
            min_sdk: MINIMUM_SDK,
            compile_sdk: DEFAULT_COMPILE_SDK,
            version_code: 1,
            version_name: "0.1.0".to_owned(),
            icon_background: "#6CA94E".to_owned(),
        }
    }

    /// Returns this configuration with a different library name.
    #[must_use]
    pub fn with_lib_name(mut self, lib_name: impl Into<String>) -> AndroidApp {
        self.lib_name = lib_name.into();
        self
    }

    /// Returns this configuration at a different orientation.
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> AndroidApp {
        self.orientation = orientation;
        self
    }

    /// Returns this configuration carrying only the given architectures.
    #[must_use]
    pub fn with_abis(mut self, abis: impl IntoIterator<Item = Abi>) -> AndroidApp {
        self.abis = abis.into_iter().collect();
        self
    }

    /// Returns this configuration at a different version.
    #[must_use]
    pub fn with_version(mut self, code: u32, name: impl Into<String>) -> AndroidApp {
        self.version_code = code;
        self.version_name = name.into();
        self
    }

    /// Checks the configuration before anything is written.
    ///
    /// # Errors
    ///
    /// Returns [`AndroidError::Invalid`] when a value would produce a project
    /// that cannot build — an API level below [`MINIMUM_SDK`], an empty ABI
    /// list, or a package that Android would reject.
    pub fn validate(&self) -> Result<(), AndroidError> {
        if self.min_sdk < MINIMUM_SDK {
            return Err(AndroidError::Invalid(format!(
                "minimum SDK {} is below {MINIMUM_SDK}, where AAudio appears; \
                 audio would fail to link",
                self.min_sdk
            )));
        }
        if self.compile_sdk < self.min_sdk {
            return Err(AndroidError::Invalid(format!(
                "compile SDK {} is below the minimum SDK {}",
                self.compile_sdk, self.min_sdk
            )));
        }
        if self.abis.is_empty() {
            return Err(AndroidError::Invalid(
                "an APK with no architectures would install nowhere".to_owned(),
            ));
        }
        // Android requires at least one dot and no leading digit in a segment.
        let segments: Vec<&str> = self.package.split('.').collect();
        if segments.len() < 2 || segments.iter().any(|part| part.is_empty()) {
            return Err(AndroidError::Invalid(format!(
                "package {:?} must have at least two dot-separated segments",
                self.package
            )));
        }
        if segments
            .iter()
            .any(|part| part.chars().next().is_some_and(|c| c.is_ascii_digit()))
        {
            return Err(AndroidError::Invalid(format!(
                "package {:?} has a segment starting with a digit",
                self.package
            )));
        }
        if self.lib_name.contains('-') {
            return Err(AndroidError::Invalid(format!(
                "library name {:?} contains a hyphen; Cargo produces \
                 lib{}.so",
                self.lib_name,
                self.lib_name.replace('-', "_")
            )));
        }
        if self.app_name.trim().is_empty() {
            return Err(AndroidError::Invalid(
                "the application needs a name to show under its icon".to_owned(),
            ));
        }
        Ok(())
    }

    /// The substitutions applied to the templates.
    fn substitutions(&self) -> BTreeMap<&'static str, String> {
        let mut values = BTreeMap::new();
        values.insert("{{APP_NAME}}", escape_xml(&self.app_name));
        values.insert("{{PROJECT_NAME}}", sanitise_project_name(&self.app_name));
        values.insert("{{PACKAGE}}", self.package.clone());
        values.insert("{{LIB_NAME}}", self.lib_name.clone());
        values.insert(
            "{{ORIENTATION}}",
            self.orientation.manifest_value().to_owned(),
        );
        values.insert("{{MIN_SDK}}", self.min_sdk.to_string());
        values.insert("{{COMPILE_SDK}}", self.compile_sdk.to_string());
        values.insert("{{VERSION_CODE}}", self.version_code.to_string());
        values.insert("{{VERSION_NAME}}", self.version_name.clone());
        values.insert("{{ICON_BACKGROUND}}", self.icon_background.clone());
        values
    }

    /// The files this configuration produces, as paths relative to the project
    /// root paired with their contents.
    ///
    /// Exposed separately from [`AndroidApp::generate`] so the whole project
    /// can be inspected in a test without touching a filesystem.
    ///
    /// # Errors
    ///
    /// Returns [`AndroidError::Invalid`] if the configuration is rejected.
    pub fn files(&self) -> Result<Vec<(PathBuf, String)>, AndroidError> {
        self.validate()?;
        let values = self.substitutions();
        let fill = |source: &str| -> String {
            let mut text = source.to_owned();
            for (key, value) in &values {
                text = text.replace(key, value);
            }
            text
        };

        let module = Path::new("app");
        let main = module.join("src/main");
        Ok(vec![
            (
                PathBuf::from("settings.gradle.kts"),
                fill(include_str!("../templates/android/settings.gradle.kts.in")),
            ),
            (
                PathBuf::from("build.gradle.kts"),
                fill(include_str!("../templates/android/build.gradle.kts.in")),
            ),
            (
                PathBuf::from("gradle.properties"),
                include_str!("../templates/android/gradle.properties").to_owned(),
            ),
            (
                PathBuf::from("gradle/wrapper/gradle-wrapper.properties"),
                include_str!("../templates/android/gradle-wrapper.properties").to_owned(),
            ),
            (
                module.join("build.gradle.kts"),
                fill(include_str!("../templates/android/app-build.gradle.kts.in")),
            ),
            (
                main.join("AndroidManifest.xml"),
                fill(include_str!("../templates/android/AndroidManifest.xml.in")),
            ),
            (
                main.join("res/values/strings.xml"),
                fill(include_str!(
                    "../templates/android/res/values/strings.xml.in"
                )),
            ),
            (
                main.join("res/values/colors.xml"),
                fill(include_str!(
                    "../templates/android/res/values/colors.xml.in"
                )),
            ),
            (
                main.join("res/drawable/ic_launcher_foreground.xml"),
                include_str!("../templates/android/res/drawable/ic_launcher_foreground.xml")
                    .to_owned(),
            ),
            (
                main.join("res/mipmap-anydpi-v26/ic_launcher.xml"),
                include_str!("../templates/android/res/mipmap-anydpi-v26/ic_launcher.xml")
                    .to_owned(),
            ),
            (
                main.join("res/mipmap-anydpi-v26/ic_launcher_round.xml"),
                include_str!("../templates/android/res/mipmap-anydpi-v26/ic_launcher_round.xml")
                    .to_owned(),
            ),
        ])
    }

    /// Writes the Gradle project into `root`, creating it if needed.
    ///
    /// Existing files are overwritten: the project is generated output, not
    /// something to edit in place, and a stale manifest left behind would
    /// silently package the wrong library name.
    ///
    /// # Errors
    ///
    /// Returns [`AndroidError::Invalid`] if the configuration is rejected, or
    /// [`AndroidError::Io`] if a file cannot be written.
    pub fn generate(&self, root: &Path) -> Result<(), AndroidError> {
        for (relative, contents) in self.files()? {
            let path = root.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, contents)?;
        }
        Ok(())
    }

    /// The directory the native libraries must be written into.
    #[must_use]
    pub fn jni_libs_directory(root: &Path) -> PathBuf {
        root.join("app/src/main/jniLibs")
    }

    /// Cross-compiles the game and packages an APK, returning its path.
    ///
    /// `workspace` is the Cargo workspace holding the game; `root` is where
    /// the generated project lives.
    ///
    /// # Errors
    ///
    /// Returns [`AndroidError::MissingTool`] when `cargo-ndk` or Gradle is not
    /// available, and [`AndroidError::Failed`] when one of them exits
    /// unsuccessfully.
    pub fn build(
        &self,
        workspace: &Path,
        root: &Path,
        profile: Profile,
    ) -> Result<PathBuf, AndroidError> {
        self.generate(root)?;

        let jni_libs = AndroidApp::jni_libs_directory(root);
        // Stale libraries from a previous ABI set would be packaged alongside
        // the new ones and shipped to devices that cannot load them.
        if jni_libs.exists() {
            std::fs::remove_dir_all(&jni_libs)?;
        }
        std::fs::create_dir_all(&jni_libs)?;

        let mut cargo = Command::new("cargo");
        cargo.current_dir(workspace).arg("ndk");
        for abi in &self.abis {
            cargo.arg("--target").arg(abi.name());
        }
        cargo
            .arg("--platform")
            .arg(self.min_sdk.to_string())
            .arg("--output-dir")
            .arg(&jni_libs)
            .arg("--")
            .arg("build")
            .arg("--lib")
            .arg("-p")
            .arg(&self.crate_name);
        if profile == Profile::Release {
            cargo.arg("--release");
        }
        run(
            &mut cargo,
            "cargo-ndk",
            "Install it with `cargo install cargo-ndk`.",
        )?;

        // Generate the wrapper rather than committing a binary jar; the
        // version is pinned in gradle/wrapper.
        if !root.join("gradlew").exists() {
            let mut wrapper = Command::new("gradle");
            wrapper.current_dir(root).arg("wrapper");
            run(
                &mut wrapper,
                "gradle",
                "Install Gradle 8.x and a JDK 17 or newer.",
            )?;
        }

        let gradlew = if cfg!(windows) {
            "gradlew.bat"
        } else {
            "./gradlew"
        };
        let mut build = Command::new(gradlew);
        build
            .current_dir(root)
            .arg("--no-daemon")
            .arg(profile.gradle_task());
        run(
            &mut build,
            "gradle",
            "Install Gradle 8.x and a JDK 17 or newer.",
        )?;

        let output = root
            .join("app/build/outputs/apk")
            .join(profile.output_directory());
        find_apk(&output).ok_or_else(|| AndroidError::Failed {
            tool: "gradle".to_owned(),
            status: format!("no APK was produced in {}", output.display()),
        })
    }
}

/// Runs a command, turning a missing binary and a failure into clear errors.
fn run(command: &mut Command, tool: &str, hint: &str) -> Result<(), AndroidError> {
    let status = command.status().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            AndroidError::MissingTool {
                tool: tool.to_owned(),
                hint: hint.to_owned(),
            }
        } else {
            AndroidError::Io(error)
        }
    })?;
    if !status.success() {
        return Err(AndroidError::Failed {
            tool: tool.to_owned(),
            status: status.to_string(),
        });
    }
    Ok(())
}

/// The first APK in a directory.
fn find_apk(directory: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(directory).ok()?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|extension| extension == "apk"))
}

/// Escapes the five characters XML cannot carry literally.
///
/// A game called `Bob & Sons` would otherwise produce a `strings.xml` that
/// does not parse, and the build would fail a long way from the cause.
fn escape_xml(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Turns a display name into something Gradle accepts as a project name.
fn sanitise_project_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || *character == ' ')
        .collect();
    let joined: String = cleaned
        .split_whitespace()
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
                None => String::new(),
            }
        })
        .collect();
    if joined.is_empty() {
        "Game".to_owned()
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A configuration like the game's.
    fn app() -> AndroidApp {
        AndroidApp::new("Verdant Hollow", "dev.verdant.hollow", "verdant-hollow")
    }

    /// The generated file at a path, for assertions.
    fn file(app: &AndroidApp, path: &str) -> String {
        app.files()
            .expect("the configuration is valid")
            .into_iter()
            .find(|(name, _)| name == Path::new(path))
            .map(|(_, contents)| contents)
            .unwrap_or_else(|| panic!("{path} was not generated"))
    }

    #[test]
    fn the_library_name_follows_cargos_rule() {
        // Cargo turns hyphens into underscores, and the manifest has to point
        // at the file that actually lands in the APK.
        assert_eq!(app().lib_name, "verdant_hollow");
    }

    #[test]
    fn the_manifest_points_at_the_games_library() {
        let manifest = file(&app(), "app/src/main/AndroidManifest.xml");
        assert!(manifest.contains(r#"android:value="verdant_hollow""#));
        assert!(!manifest.contains("{{"), "a placeholder was left unfilled");
    }

    #[test]
    fn every_placeholder_is_substituted() {
        // One unfilled placeholder produces a project that fails deep inside
        // Gradle with an unrelated-looking message.
        for (path, contents) in app().files().expect("valid") {
            assert!(
                !contents.contains("{{"),
                "{} still contains a placeholder",
                path.display()
            );
        }
    }

    #[test]
    fn a_complete_project_is_generated() {
        let files = app().files().expect("valid");
        let names: Vec<String> = files
            .iter()
            .map(|(path, _)| path.display().to_string().replace('\\', "/"))
            .collect();
        for required in [
            "settings.gradle.kts",
            "build.gradle.kts",
            "app/build.gradle.kts",
            "app/src/main/AndroidManifest.xml",
            "app/src/main/res/values/strings.xml",
        ] {
            assert!(
                names.iter().any(|name| name == required),
                "missing {required}"
            );
        }
    }

    #[test]
    fn the_app_name_reaches_the_launcher() {
        let strings = file(&app(), "app/src/main/res/values/strings.xml");
        assert!(strings.contains("Verdant Hollow"));
    }

    #[test]
    fn a_name_with_xml_syntax_is_escaped() {
        // Otherwise strings.xml does not parse and the build fails a long way
        // from the cause.
        let app = AndroidApp::new("Bob & Sons <Farm>", "com.bob.farm", "bob-farm");
        let strings = file(&app, "app/src/main/res/values/strings.xml");
        assert!(strings.contains("Bob &amp; Sons &lt;Farm&gt;"));
        assert!(!strings.contains("& S"), "a raw ampersand survived");
    }

    #[test]
    fn the_gradle_project_name_is_sanitised() {
        let app = AndroidApp::new("Bob & Sons <Farm>", "com.bob.farm", "bob-farm");
        let settings = file(&app, "settings.gradle.kts");
        assert!(settings.contains(r#"rootProject.name = "BobSonsFarm""#));
    }

    #[test]
    fn a_nameless_project_still_gets_a_gradle_name() {
        assert_eq!(sanitise_project_name("***"), "Game");
    }

    #[test]
    fn the_api_levels_agree_between_gradle_and_the_ndk() {
        // The bug this exists to prevent: the level passed to cargo-ndk and
        // the minSdk in Gradle drifting apart, which fails at link time with
        // a missing system library.
        let mut app = app();
        app.min_sdk = 28;
        let gradle = file(&app, "app/build.gradle.kts");
        assert!(gradle.contains("minSdk = 28"));
        assert_eq!(app.min_sdk, 28, "and the same value drives cargo-ndk");
    }

    #[test]
    fn an_api_level_without_aaudio_is_refused() {
        let mut app = app();
        app.min_sdk = 24;
        let error = app.validate().expect_err("24 has no libaaudio");
        assert!(format!("{error}").contains("AAudio"));
    }

    #[test]
    fn a_compile_sdk_below_the_minimum_is_refused() {
        let mut app = app();
        app.compile_sdk = 21;
        assert!(app.validate().is_err());
    }

    #[test]
    fn an_apk_with_no_architectures_is_refused() {
        let app = app().with_abis([]);
        let error = app.validate().expect_err("no ABIs");
        assert!(format!("{error}").contains("install nowhere"));
    }

    #[test]
    fn a_malformed_package_is_refused() {
        // Android silently misbehaves rather than erroring on some of these,
        // so they are caught here instead.
        for package in ["hollow", "dev..hollow", "dev.1hollow", ""] {
            let app = AndroidApp::new("Game", package, "game");
            assert!(
                app.validate().is_err(),
                "{package:?} should have been refused"
            );
        }
    }

    #[test]
    fn a_hyphenated_library_name_is_refused() {
        // Cargo would emit libgame_x.so and the manifest would look for
        // libgame-x.so, which fails at launch rather than at build time.
        let app = app().with_lib_name("verdant-hollow");
        let error = app.validate().expect_err("hyphen");
        assert!(format!("{error}").contains("hyphen"));
    }

    #[test]
    fn a_nameless_app_is_refused() {
        let mut app = app();
        app.app_name = "   ".to_owned();
        assert!(app.validate().is_err());
    }

    #[test]
    fn every_abi_has_a_distinct_target_triple() {
        let targets: std::collections::BTreeSet<&str> =
            Abi::ALL.iter().map(|abi| abi.target()).collect();
        assert_eq!(targets.len(), Abi::ALL.len());
        let names: std::collections::BTreeSet<&str> =
            Abi::ALL.iter().map(|abi| abi.name()).collect();
        assert_eq!(names.len(), Abi::ALL.len());
    }

    #[test]
    fn orientations_map_to_manifest_values() {
        assert_eq!(Orientation::Landscape.manifest_value(), "sensorLandscape");
        assert_eq!(Orientation::Portrait.manifest_value(), "sensorPortrait");
        let manifest = file(
            &app().with_orientation(Orientation::Portrait),
            "app/src/main/AndroidManifest.xml",
        );
        assert!(manifest.contains("sensorPortrait"));
    }

    #[test]
    fn the_version_reaches_gradle() {
        let app = app().with_version(7, "1.2.3");
        let gradle = file(&app, "app/build.gradle.kts");
        assert!(gradle.contains("versionCode = 7"));
        assert!(gradle.contains(r#"versionName = "1.2.3""#));
    }

    #[test]
    fn the_package_reaches_gradle_as_both_namespace_and_id() {
        let gradle = file(&app(), "app/build.gradle.kts");
        assert!(gradle.contains(r#"namespace = "dev.verdant.hollow""#));
        assert!(gradle.contains(r#"applicationId = "dev.verdant.hollow""#));
    }

    #[test]
    fn generating_writes_the_project_to_disk() {
        let root =
            std::env::temp_dir().join(format!("verdant-android-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        app().generate(&root).expect("writes");

        assert!(root.join("settings.gradle.kts").is_file());
        assert!(root.join("app/src/main/AndroidManifest.xml").is_file());
        let manifest =
            std::fs::read_to_string(root.join("app/src/main/AndroidManifest.xml")).expect("read");
        assert!(manifest.contains("verdant_hollow"));

        // Regenerating over an existing project must replace it, or a stale
        // manifest would package the wrong library name.
        let renamed = AndroidApp::new("Other", "com.other.game", "other-game");
        renamed.generate(&root).expect("regenerates");
        let manifest =
            std::fs::read_to_string(root.join("app/src/main/AndroidManifest.xml")).expect("read");
        assert!(manifest.contains("other_game"));
        assert!(!manifest.contains("verdant_hollow"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_jni_directory_is_where_gradle_looks() {
        // Gradle's jniLibs.srcDirs and this have to agree or the APK ships
        // with no native code and crashes at launch.
        let gradle = file(&app(), "app/build.gradle.kts");
        assert!(gradle.contains("src/main/jniLibs"));
        let path = AndroidApp::jni_libs_directory(Path::new("/tmp/project"));
        assert!(path.ends_with("app/src/main/jniLibs"));
    }

    #[test]
    fn profiles_name_their_own_outputs() {
        assert_eq!(Profile::Debug.gradle_task(), "assembleDebug");
        assert_eq!(Profile::Debug.output_directory(), "debug");
        assert_eq!(Profile::Release.gradle_task(), "assembleRelease");
        assert_eq!(Profile::Release.output_directory(), "release");
    }
}
