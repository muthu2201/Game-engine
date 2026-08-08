//! Provenance records for generated assets.
//!
//! Every generated sprite carries a record of exactly how it was produced. The
//! blueprint calls for this in the context of cloud generation — where it
//! serves app-store disclosure and IP tracking — and it is just as valuable
//! here, for a different reason: because generation is deterministic, a
//! provenance record is a *complete recipe*. Anyone holding one can reproduce
//! the exact bytes, which turns "this sprite looks wrong" into a reproducible
//! bug rather than a screenshot.

use serde::{Deserialize, Serialize};

/// What kind of asset a record describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum AssetKind {
    /// A character with a directional walk cycle.
    Character,
    /// A tiling terrain texture.
    Terrain,
    /// A crop's growth stages.
    Crop,
    /// A tree or large prop.
    Tree,
    /// An inventory icon.
    Item,
}

impl AssetKind {
    /// A short, stable identifier used in asset keys.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            AssetKind::Character => "character",
            AssetKind::Terrain => "terrain",
            AssetKind::Crop => "crop",
            AssetKind::Tree => "tree",
            AssetKind::Item => "item",
        }
    }
}

/// The complete recipe for a generated asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// What was generated.
    pub kind: AssetKind,
    /// The seed the generator was given.
    pub seed: u64,
    /// Pixel dimensions of one frame.
    pub size: (u32, u32),
    /// Number of frames produced.
    pub frames: usize,
    /// The version of the generator that produced it.
    ///
    /// Bumped whenever a generator's output changes. Without it, a saved world
    /// referring to a sprite by seed would silently change appearance after an
    /// engine update — this makes the change detectable.
    pub generator_version: u32,
    /// A hash of the resolved pixels, for content addressing and verification.
    pub content_hash: u64,
}

/// The current generator version.
///
/// Increment whenever a change to [`generators`](crate::generators) alters the
/// pixels produced for an unchanged seed.
pub const GENERATOR_VERSION: u32 = 1;

impl Provenance {
    /// Builds a record for a generated sprite.
    #[must_use]
    pub fn new(kind: AssetKind, seed: u64, size: (u32, u32), frames: &[Vec<u8>]) -> Provenance {
        Provenance {
            kind,
            seed,
            size,
            frames: frames.len(),
            generator_version: GENERATOR_VERSION,
            content_hash: hash_frames(frames),
        }
    }

    /// A stable, human-readable key for this asset.
    ///
    /// Includes the generator version, so art from two engine versions cannot
    /// collide in a cache.
    #[must_use]
    pub fn asset_key(&self) -> String {
        format!(
            "{}/{:016x}/{}x{}/v{}",
            self.kind.slug(),
            self.seed,
            self.size.0,
            self.size.1,
            self.generator_version
        )
    }

    /// True when `frames` are the pixels this record describes.
    ///
    /// A cache uses this to confirm that stored bytes still match the recipe
    /// before trusting them.
    #[must_use]
    pub fn matches(&self, frames: &[Vec<u8>]) -> bool {
        self.frames == frames.len() && self.content_hash == hash_frames(frames)
    }
}

/// FNV-1a over every frame's bytes, with the frame count folded in.
///
/// Order-sensitive on purpose: two sprites with the same frames in a different
/// order are different sprites.
fn hash_frames(frames: &[Vec<u8>]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for count in (frames.len() as u64).to_le_bytes() {
        fold(count);
    }
    for frame in frames {
        for byte in frame {
            fold(*byte);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::{generate_character, CharacterStyle};

    /// Resolves every frame of a generated sprite to bytes.
    fn frames_of(sprite: &crate::GeneratedSprite) -> Vec<Vec<u8>> {
        (0..sprite.frame_count())
            .filter_map(|index| sprite.frame_rgba(index))
            .collect()
    }

    #[test]
    fn a_record_describes_what_was_generated() {
        let sprite = generate_character(42, 16, 24, CharacterStyle::villager());
        let frames = frames_of(&sprite);
        let record = Provenance::new(AssetKind::Character, 42, (16, 24), &frames);

        assert_eq!(record.seed, 42);
        assert_eq!(record.size, (16, 24));
        assert_eq!(record.frames, sprite.frame_count());
        assert_eq!(record.generator_version, GENERATOR_VERSION);
    }

    #[test]
    fn a_record_is_a_complete_recipe() {
        // The whole point: the record alone is enough to reproduce the bytes.
        let record = {
            let sprite = generate_character(1234, 16, 24, CharacterStyle::villager());
            Provenance::new(AssetKind::Character, 1234, (16, 24), &frames_of(&sprite))
        };

        let regenerated = generate_character(
            record.seed,
            record.size.0,
            record.size.1,
            CharacterStyle::villager(),
        );
        assert!(
            record.matches(&frames_of(&regenerated)),
            "regenerating from the record must reproduce the exact pixels"
        );
    }

    #[test]
    fn a_different_seed_produces_a_different_hash() {
        let first = generate_character(1, 16, 24, CharacterStyle::villager());
        let second = generate_character(2, 16, 24, CharacterStyle::villager());
        let a = Provenance::new(AssetKind::Character, 1, (16, 24), &frames_of(&first));
        let b = Provenance::new(AssetKind::Character, 2, (16, 24), &frames_of(&second));
        assert_ne!(a.content_hash, b.content_hash);
        assert_ne!(a.asset_key(), b.asset_key());
    }

    #[test]
    fn tampered_pixels_fail_verification() {
        let sprite = generate_character(7, 16, 24, CharacterStyle::villager());
        let frames = frames_of(&sprite);
        let record = Provenance::new(AssetKind::Character, 7, (16, 24), &frames);

        let mut tampered = frames.clone();
        tampered[0][0] = tampered[0][0].wrapping_add(1);
        assert!(!record.matches(&tampered));

        // Losing a frame is also detected.
        let mut truncated = frames;
        truncated.pop();
        assert!(!record.matches(&truncated));
    }

    #[test]
    fn frame_order_is_part_of_the_hash() {
        let a = vec![vec![1u8, 2, 3], vec![4, 5, 6]];
        let b = vec![vec![4u8, 5, 6], vec![1, 2, 3]];
        assert_ne!(hash_frames(&a), hash_frames(&b));
    }

    #[test]
    fn asset_keys_are_stable_and_versioned() {
        let record = Provenance {
            kind: AssetKind::Crop,
            seed: 0xDEAD_BEEF,
            size: (16, 16),
            frames: 5,
            generator_version: 1,
            content_hash: 0,
        };
        assert_eq!(record.asset_key(), "crop/00000000deadbeef/16x16/v1");

        // A generator version bump must change the key, or a stale cache entry
        // would be served for the new art.
        let bumped = Provenance {
            generator_version: 2,
            ..record.clone()
        };
        assert_ne!(record.asset_key(), bumped.asset_key());
    }

    #[test]
    fn records_round_trip_through_serialisation() {
        let record = Provenance {
            kind: AssetKind::Tree,
            seed: 99,
            size: (24, 32),
            frames: 1,
            generator_version: GENERATOR_VERSION,
            content_hash: 0x1234_5678,
        };
        let encoded = serde_json::to_string(&record).expect("serialisable");
        let decoded: Provenance = serde_json::from_str(&encoded).expect("deserialisable");
        assert_eq!(decoded, record);
    }

    #[test]
    fn every_asset_kind_has_a_distinct_slug() {
        let kinds = [
            AssetKind::Character,
            AssetKind::Terrain,
            AssetKind::Crop,
            AssetKind::Tree,
            AssetKind::Item,
        ];
        let slugs: std::collections::HashSet<&str> = kinds.iter().map(|kind| kind.slug()).collect();
        assert_eq!(slugs.len(), kinds.len());
    }
}
