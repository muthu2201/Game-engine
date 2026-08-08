//! # Verdant app
//!
//! Packaging a game built on the Verdant engine for the platforms it runs on.
//!
//! ## Why the engine owns this
//!
//! A game could keep its own `android/` directory with a hand-written manifest
//! and Gradle module — and that is exactly how build knowledge rots. The API
//! level passed to the NDK has to match the `minSdk` in Gradle, the manifest's
//! `android.app.lib_name` has to match Cargo's `[lib] name` after hyphen
//! substitution, and the `jniLibs` directory Gradle reads has to be the one the
//! cross-compiler writes to. Kept in separate hand-edited files, those three
//! pairs drift, and each one fails late and confusingly: a missing system
//! library at link time, a crash at launch, an APK with no native code in it.
//!
//! Here they are derived from one [`AndroidApp`] value, so they cannot
//! disagree, and every game on the engine gets a working Android build without
//! copying anything.
//!
//! ## Example
//!
//! ```no_run
//! use std::path::Path;
//! use verdant_app::android::{AndroidApp, Profile};
//!
//! let app = AndroidApp::new("Verdant Hollow", "dev.verdant.hollow", "verdant-hollow");
//! let apk = app.build(Path::new("."), Path::new("target/android"), Profile::Debug)?;
//! println!("built {}", apk.display());
//! # Ok::<(), verdant_app::android::AndroidError>(())
//! ```
//!
//! The same configuration can be inspected without running anything, which is
//! how the generated project is tested:
//!
//! ```
//! use verdant_app::android::AndroidApp;
//!
//! let app = AndroidApp::new("Verdant Hollow", "dev.verdant.hollow", "verdant-hollow");
//! let files = app.files().expect("a valid configuration");
//! assert!(files.iter().any(|(path, _)| path.ends_with("AndroidManifest.xml")));
//! ```

#![doc(html_no_source)]

pub mod android;

pub use android::{Abi, AndroidApp, AndroidError, Orientation, Profile};
