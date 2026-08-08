//! # Verdant save files
//!
//! Versioned, checksummed, atomically written save files with schema migration.
//!
//! ## What a save has to survive
//!
//! A save file outlives the build that wrote it. Players update mid-playthrough,
//! and a game that loses a fifty-hour farm because a component gained a field is
//! not shippable. Three mechanisms handle that:
//!
//! * **Versioning** — every file records the schema version that wrote it.
//! * **Migration** — [`MigrationChain`] holds a function per version step, and
//!   loading replays them in order to bring an old file up to date. Each
//!   migration only has to know about *its own* step, not every past format.
//! * **Checksums** — a truncated or bit-rotted file is detected on load rather
//!   than deserialising into nonsense that corrupts the session.
//!
//! Writes are atomic: the data goes to a temporary file which is then renamed
//! over the real one. A crash or power loss mid-save therefore leaves either
//! the old save or the new one, never a half-written file. Games that write in
//! place lose the save exactly when it matters most — during the autosave at
//! the end of a long day.
//!
//! ## Example
//!
//! ```
//! use serde::{Deserialize, Serialize};
//! use verdant_save::{MigrationChain, SaveFile};
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct Farm { gold: u32, day: u32 }
//!
//! let mut chain = MigrationChain::new(2);
//! // Version 1 had no `day` field; supply one when loading an old save.
//! chain.add(1, |value| {
//!     if let Some(object) = value.as_object_mut() {
//!         object.insert("day".into(), serde_json::json!(1));
//!     }
//!     Ok(())
//! });
//!
//! let file = SaveFile::create(&Farm { gold: 500, day: 12 }, 2).unwrap();
//! let encoded = file.to_bytes().unwrap();
//!
//! let loaded: Farm = SaveFile::from_bytes(&encoded).unwrap().load(&chain).unwrap();
//! assert_eq!(loaded, Farm { gold: 500, day: 12 });
//! ```

#![doc(html_no_source)]

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// A schema version number.
///
/// Incremented whenever the saved data's shape changes in a way that an older
/// build could not read correctly.
pub type Version = u32;

/// Anything that can go wrong saving or loading.
#[derive(Debug)]
pub enum SaveError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The bytes are not valid save data.
    Malformed(String),
    /// The stored checksum does not match the payload.
    ///
    /// The file is truncated or corrupt; loading it would produce a plausible
    /// but wrong world.
    ChecksumMismatch {
        /// What the file claims.
        expected: u64,
        /// What the payload actually hashes to.
        actual: u64,
    },
    /// The file is newer than this build understands.
    ///
    /// Migrations only run forwards, so a downgrade cannot be repaired — the
    /// honest answer is to refuse rather than silently drop the fields the
    /// older build does not know about.
    FromTheFuture {
        /// The version the file was written at.
        file_version: Version,
        /// The version this build supports.
        supported: Version,
    },
    /// No migration exists to move a file forward from this version.
    MissingMigration {
        /// The version that has no step defined.
        from: Version,
    },
    /// A migration function reported a problem.
    MigrationFailed {
        /// The version being migrated from.
        from: Version,
        /// What went wrong.
        reason: String,
    },
    /// The payload did not deserialise into the requested type.
    Deserialisation(String),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaveError::Io(error) => write!(f, "save file I/O failed: {error}"),
            SaveError::Malformed(detail) => write!(f, "save file is malformed: {detail}"),
            SaveError::ChecksumMismatch { expected, actual } => write!(
                f,
                "save file is corrupt: checksum {actual:#x} does not match the recorded {expected:#x}"
            ),
            SaveError::FromTheFuture { file_version, supported } => write!(
                f,
                "save file was written by a newer build (version {file_version}, this build supports {supported})"
            ),
            SaveError::MissingMigration { from } => {
                write!(f, "no migration is defined from save version {from}")
            }
            SaveError::MigrationFailed { from, reason } => {
                write!(f, "migration from save version {from} failed: {reason}")
            }
            SaveError::Deserialisation(detail) => {
                write!(f, "save payload did not match the expected shape: {detail}")
            }
        }
    }
}

impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SaveError::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SaveError {
    fn from(error: std::io::Error) -> SaveError {
        SaveError::Io(error)
    }
}

/// The result type used throughout this crate.
pub type SaveResult<T> = Result<T, SaveError>;

/// A save file: a versioned, checksummed payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveFile {
    /// Schema version the payload was written at.
    pub version: Version,
    /// FNV-1a hash of the serialised payload.
    pub checksum: u64,
    /// Seconds since the Unix epoch, for the load menu.
    pub written_at: u64,
    /// The saved data, still as JSON so migrations can rewrite it.
    pub payload: serde_json::Value,
}

impl SaveFile {
    /// Serialises `data` into a new save file at `version`.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Malformed`] if `data` cannot be serialised.
    pub fn create<T: Serialize>(data: &T, version: Version) -> SaveResult<SaveFile> {
        let payload =
            serde_json::to_value(data).map_err(|error| SaveError::Malformed(error.to_string()))?;
        let checksum = checksum_of(&payload)?;
        Ok(SaveFile {
            version,
            checksum,
            written_at: now_seconds(),
            payload,
        })
    }

    /// Encodes the file as bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Malformed`] if the envelope cannot be serialised.
    pub fn to_bytes(&self) -> SaveResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|error| SaveError::Malformed(error.to_string()))
    }

    /// Decodes a file, verifying its checksum.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Malformed`] for unparseable bytes and
    /// [`SaveError::ChecksumMismatch`] when the payload has been altered.
    pub fn from_bytes(bytes: &[u8]) -> SaveResult<SaveFile> {
        let file: SaveFile = serde_json::from_slice(bytes)
            .map_err(|error| SaveError::Malformed(error.to_string()))?;
        let actual = checksum_of(&file.payload)?;
        if actual != file.checksum {
            return Err(SaveError::ChecksumMismatch {
                expected: file.checksum,
                actual,
            });
        }
        Ok(file)
    }

    /// Migrates the payload up to the chain's current version and deserialises.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::FromTheFuture`] for a newer file,
    /// [`SaveError::MissingMigration`] when a step is absent,
    /// [`SaveError::MigrationFailed`] when one reports a problem, and
    /// [`SaveError::Deserialisation`] when the migrated payload still does not
    /// match `T`.
    pub fn load<T: DeserializeOwned>(self, chain: &MigrationChain) -> SaveResult<T> {
        let migrated = chain.migrate(self.payload, self.version)?;
        serde_json::from_value(migrated)
            .map_err(|error| SaveError::Deserialisation(error.to_string()))
    }

    /// The version without migrating or deserialising.
    ///
    /// Lets a load menu list saves, including ones too new to open.
    #[must_use]
    pub fn version(&self) -> Version {
        self.version
    }
}

/// FNV-1a over the payload's canonical serialisation.
///
/// `serde_json::Value` stores objects in a `BTreeMap`, so the serialisation —
/// and therefore the hash — is independent of the order fields were inserted.
/// A hash that varied with insertion order would flag valid saves as corrupt.
fn checksum_of(payload: &serde_json::Value) -> SaveResult<u64> {
    let bytes =
        serde_json::to_vec(payload).map_err(|error| SaveError::Malformed(error.to_string()))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(hash)
}

/// Seconds since the Unix epoch, or zero if the clock is before it.
fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// One schema step's transformation.
type MigrationFn = Box<dyn Fn(&mut serde_json::Value) -> Result<(), String> + Send + Sync>;

/// The ordered set of migrations from old versions up to the current one.
///
/// Each entry moves a payload from version `n` to version `n + 1`. Composing
/// single steps means a save from any supported version reaches the present by
/// replaying the steps between, and adding a new format only requires writing
/// the one new step.
pub struct MigrationChain {
    current: Version,
    /// Steps indexed by the version they migrate *from*.
    steps: std::collections::BTreeMap<Version, MigrationFn>,
}

impl MigrationChain {
    /// Creates a chain targeting `current`.
    #[must_use]
    pub fn new(current: Version) -> MigrationChain {
        MigrationChain {
            current,
            steps: std::collections::BTreeMap::new(),
        }
    }

    /// Registers the step from `from` to `from + 1`.
    ///
    /// Registering the same step twice replaces the earlier one.
    pub fn add(
        &mut self,
        from: Version,
        migration: impl Fn(&mut serde_json::Value) -> Result<(), String> + Send + Sync + 'static,
    ) {
        self.steps.insert(from, Box::new(migration));
    }

    /// The version this chain migrates to.
    #[must_use]
    pub fn current_version(&self) -> Version {
        self.current
    }

    /// True when a payload at `version` can be brought up to date.
    #[must_use]
    pub fn can_migrate(&self, version: Version) -> bool {
        version <= self.current
            && (version..self.current).all(|step| self.steps.contains_key(&step))
    }

    /// Replays every step from `version` up to the current one.
    ///
    /// # Errors
    ///
    /// See [`SaveFile::load`] for the error cases.
    pub fn migrate(
        &self,
        mut payload: serde_json::Value,
        version: Version,
    ) -> SaveResult<serde_json::Value> {
        if version > self.current {
            return Err(SaveError::FromTheFuture {
                file_version: version,
                supported: self.current,
            });
        }
        for step in version..self.current {
            let migration = self
                .steps
                .get(&step)
                .ok_or(SaveError::MissingMigration { from: step })?;
            migration(&mut payload)
                .map_err(|reason| SaveError::MigrationFailed { from: step, reason })?;
        }
        Ok(payload)
    }
}

impl fmt::Debug for MigrationChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationChain")
            .field("current", &self.current)
            .field("steps", &self.steps.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Reads and writes save files on disk.
#[derive(Clone, Debug)]
pub struct SaveSlot {
    path: PathBuf,
}

impl SaveSlot {
    /// Targets a specific file.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> SaveSlot {
        SaveSlot { path: path.into() }
    }

    /// The file's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// True when the file exists.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Writes `data` atomically.
    ///
    /// The bytes go to a sibling temporary file which is flushed and then
    /// renamed over the target. Rename is atomic on every platform the engine
    /// targets, so an interrupted save leaves the previous file intact rather
    /// than a truncated one.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Io`] if any filesystem step fails, or
    /// [`SaveError::Malformed`] if `data` cannot be serialised.
    pub fn write<T: Serialize>(&self, data: &T, version: Version) -> SaveResult<()> {
        use std::io::Write;

        let file = SaveFile::create(data, version)?;
        let bytes = file.to_bytes()?;

        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let temporary = self.path.with_extension("tmp");
        {
            let mut handle = std::fs::File::create(&temporary)?;
            handle.write_all(&bytes)?;
            // Without the sync, the rename can land before the data does, and a
            // power loss leaves an empty file where the save should be.
            handle.sync_all()?;
        }
        std::fs::rename(&temporary, &self.path)?;
        Ok(())
    }

    /// Reads and migrates the file.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Io`] if the file cannot be read, plus any of the
    /// errors described on [`SaveFile::from_bytes`] and [`SaveFile::load`].
    pub fn read<T: DeserializeOwned>(&self, chain: &MigrationChain) -> SaveResult<T> {
        let bytes = std::fs::read(&self.path)?;
        SaveFile::from_bytes(&bytes)?.load(chain)
    }

    /// Reads the envelope without migrating, for listing saves.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Io`] or [`SaveError::Malformed`].
    pub fn read_header(&self) -> SaveResult<SaveFile> {
        let bytes = std::fs::read(&self.path)?;
        SaveFile::from_bytes(&bytes)
    }

    /// Deletes the file, if it exists.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Io`] if deletion fails for a reason other than the
    /// file being absent.
    pub fn delete(&self) -> SaveResult<()> {
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(SaveError::Io(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct FarmV2 {
        gold: u32,
        day: u32,
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct FarmV1 {
        gold: u32,
    }

    /// A chain from version 1 to 2 that supplies the new `day` field.
    fn chain() -> MigrationChain {
        let mut chain = MigrationChain::new(2);
        chain.add(1, |value| {
            let object = value.as_object_mut().ok_or("payload is not an object")?;
            object.insert("day".to_string(), serde_json::json!(1));
            Ok(())
        });
        chain
    }

    /// A scratch directory that cleans itself up.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let path =
                std::env::temp_dir().join(format!("verdant-save-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&path).expect("the temp directory is writable");
            TempDir(path)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_save_round_trips_through_bytes() {
        let farm = FarmV2 { gold: 500, day: 12 };
        let bytes = SaveFile::create(&farm, 2).unwrap().to_bytes().unwrap();
        let loaded: FarmV2 = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load(&chain())
            .unwrap();
        assert_eq!(loaded, farm);
    }

    #[test]
    fn an_old_save_is_migrated_forward() {
        // Written by a build that only knew about `gold`.
        let bytes = SaveFile::create(&FarmV1 { gold: 250 }, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let loaded: FarmV2 = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load(&chain())
            .unwrap();
        assert_eq!(
            loaded,
            FarmV2 { gold: 250, day: 1 },
            "the new field was supplied"
        );
    }

    #[test]
    fn migrations_compose_across_several_versions() {
        let mut chain = MigrationChain::new(3);
        chain.add(1, |value| {
            value
                .as_object_mut()
                .unwrap()
                .insert("day".into(), serde_json::json!(1));
            Ok(())
        });
        chain.add(2, |value| {
            // Version 3 renamed `gold` to `coins`.
            let object = value.as_object_mut().unwrap();
            let gold = object.remove("gold").unwrap_or(serde_json::json!(0));
            object.insert("coins".into(), gold);
            Ok(())
        });

        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct FarmV3 {
            coins: u32,
            day: u32,
        }

        let bytes = SaveFile::create(&FarmV1 { gold: 99 }, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let loaded: FarmV3 = SaveFile::from_bytes(&bytes).unwrap().load(&chain).unwrap();
        assert_eq!(loaded, FarmV3 { coins: 99, day: 1 });
    }

    #[test]
    fn a_current_version_save_skips_migration_entirely() {
        let mut chain = MigrationChain::new(2);
        chain.add(1, |_| Err("this step must not run".to_string()));
        let bytes = SaveFile::create(&FarmV2 { gold: 1, day: 1 }, 2)
            .unwrap()
            .to_bytes()
            .unwrap();
        assert!(SaveFile::from_bytes(&bytes)
            .unwrap()
            .load::<FarmV2>(&chain)
            .is_ok());
    }

    #[test]
    fn corruption_is_detected_rather_than_loaded() {
        let bytes = SaveFile::create(&FarmV2 { gold: 500, day: 3 }, 2)
            .unwrap()
            .to_bytes()
            .unwrap();
        // Tamper with the payload, leaving the recorded checksum stale.
        let text = String::from_utf8(bytes).unwrap().replace("500", "999");

        let error = SaveFile::from_bytes(text.as_bytes()).expect_err("tampering must be caught");
        assert!(matches!(error, SaveError::ChecksumMismatch { .. }));
        assert!(error.to_string().contains("corrupt"));
    }

    #[test]
    fn truncated_files_are_rejected() {
        let bytes = SaveFile::create(&FarmV2 { gold: 1, day: 1 }, 2)
            .unwrap()
            .to_bytes()
            .unwrap();
        let truncated = &bytes[..bytes.len() / 2];
        assert!(matches!(
            SaveFile::from_bytes(truncated),
            Err(SaveError::Malformed(_))
        ));
    }

    #[test]
    fn a_save_from_a_newer_build_is_refused_clearly() {
        let bytes = SaveFile::create(&FarmV2 { gold: 1, day: 1 }, 99)
            .unwrap()
            .to_bytes()
            .unwrap();
        let error = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load::<FarmV2>(&chain())
            .expect_err("a downgrade cannot be repaired");
        assert!(matches!(
            error,
            SaveError::FromTheFuture {
                file_version: 99,
                supported: 2
            }
        ));
        assert!(error.to_string().contains("newer build"));
    }

    #[test]
    fn a_missing_migration_step_is_reported() {
        // A chain that jumps from 1 to 3 without defining the step from 2.
        let mut chain = MigrationChain::new(3);
        chain.add(1, |_| Ok(()));
        assert!(!chain.can_migrate(1));

        let bytes = SaveFile::create(&FarmV1 { gold: 1 }, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let error = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load::<FarmV2>(&chain)
            .unwrap_err();
        assert!(matches!(error, SaveError::MissingMigration { from: 2 }));
    }

    #[test]
    fn a_failing_migration_names_the_step() {
        let mut chain = MigrationChain::new(2);
        chain.add(1, |_| Err("the barn is on fire".to_string()));

        let bytes = SaveFile::create(&FarmV1 { gold: 1 }, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let error = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load::<FarmV2>(&chain)
            .unwrap_err();
        assert!(matches!(error, SaveError::MigrationFailed { from: 1, .. }));
        assert!(error.to_string().contains("barn is on fire"));
    }

    #[test]
    fn a_payload_of_the_wrong_shape_is_reported_distinctly() {
        let bytes = SaveFile::create(&serde_json::json!({ "unrelated": true }), 2)
            .unwrap()
            .to_bytes()
            .unwrap();
        let error = SaveFile::from_bytes(&bytes)
            .unwrap()
            .load::<FarmV2>(&chain())
            .unwrap_err();
        assert!(matches!(error, SaveError::Deserialisation(_)));
    }

    #[test]
    fn can_migrate_reports_reachability() {
        let chain = chain();
        assert!(chain.can_migrate(1));
        assert!(chain.can_migrate(2));
        assert!(!chain.can_migrate(3), "a future version is not reachable");
        assert_eq!(chain.current_version(), 2);
    }

    #[test]
    fn the_checksum_is_independent_of_field_order() {
        // Two payloads with the same content written in different orders must
        // hash the same, or valid saves would be flagged as corrupt.
        let first: serde_json::Value = serde_json::from_str(r#"{"gold":1,"day":2}"#).unwrap();
        let second: serde_json::Value = serde_json::from_str(r#"{"day":2,"gold":1}"#).unwrap();
        assert_eq!(checksum_of(&first).unwrap(), checksum_of(&second).unwrap());
    }

    #[test]
    fn a_slot_writes_and_reads_from_disk() {
        let dir = TempDir::new("roundtrip");
        let slot = SaveSlot::new(dir.file("farm.json"));
        assert!(!slot.exists());

        let farm = FarmV2 {
            gold: 1234,
            day: 56,
        };
        slot.write(&farm, 2).unwrap();
        assert!(slot.exists());

        let loaded: FarmV2 = slot.read(&chain()).unwrap();
        assert_eq!(loaded, farm);
        assert_eq!(slot.read_header().unwrap().version(), 2);
    }

    #[test]
    fn writing_creates_missing_directories() {
        let dir = TempDir::new("nested");
        let slot = SaveSlot::new(dir.file("deep/nested/farm.json"));
        slot.write(&FarmV2 { gold: 1, day: 1 }, 2).unwrap();
        assert!(slot.exists());
    }

    #[test]
    fn writing_leaves_no_temporary_file_behind() {
        let dir = TempDir::new("atomic");
        let slot = SaveSlot::new(dir.file("farm.json"));
        slot.write(&FarmV2 { gold: 1, day: 1 }, 2).unwrap();
        assert!(!slot.path().with_extension("tmp").exists());
    }

    #[test]
    fn overwriting_replaces_the_previous_save() {
        let dir = TempDir::new("overwrite");
        let slot = SaveSlot::new(dir.file("farm.json"));
        slot.write(&FarmV2 { gold: 1, day: 1 }, 2).unwrap();
        slot.write(&FarmV2 { gold: 2, day: 2 }, 2).unwrap();
        let loaded: FarmV2 = slot.read(&chain()).unwrap();
        assert_eq!(loaded, FarmV2 { gold: 2, day: 2 });
    }

    #[test]
    fn reading_a_missing_slot_reports_io_rather_than_panicking() {
        let dir = TempDir::new("missing");
        let slot = SaveSlot::new(dir.file("absent.json"));
        assert!(matches!(
            slot.read::<FarmV2>(&chain()),
            Err(SaveError::Io(_))
        ));
    }

    #[test]
    fn deleting_is_idempotent() {
        let dir = TempDir::new("delete");
        let slot = SaveSlot::new(dir.file("farm.json"));
        slot.write(&FarmV2 { gold: 1, day: 1 }, 2).unwrap();
        slot.delete().unwrap();
        assert!(!slot.exists());
        slot.delete()
            .expect("deleting an absent save is not an error");
    }
}
