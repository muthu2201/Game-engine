//! The farm: tilled soil, planted crops, and what happens overnight.
//!
//! Farm plots are stored in a map keyed by tile, not as entities. A farm has a
//! few hundred worked tiles at most, every one is addressed by coordinate far
//! more often than it is iterated, and the whole thing must serialise into a
//! save — all of which a keyed map does better than an entity per tile.

use crate::calendar::{Season, Weather};
use crate::items::{crop, CropId, ItemId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use verdant_core_math::{IVec2, Rng};

/// What is growing on one tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Planting {
    /// What was planted.
    pub crop: CropId,
    /// Days of growth accumulated. Only advances on watered days.
    pub growth_days: u32,
    /// Set once the crop has been harvested at least once, for regrowing crops.
    pub harvested_once: bool,
}

/// One worked tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Plot {
    /// Whether the soil was watered today.
    pub watered: bool,
    /// What is growing, if anything.
    pub planting: Option<Planting>,
}

impl Plot {
    /// Freshly tilled, empty soil.
    #[must_use]
    pub const fn tilled() -> Plot {
        Plot {
            watered: false,
            planting: None,
        }
    }

    /// True when a crop is present and ready to pick.
    #[must_use]
    pub fn is_harvestable(&self) -> bool {
        self.planting
            .is_some_and(|planting| crop(planting.crop).is_mature(planting.growth_days))
    }

    /// The growth stage to draw, or `None` for bare soil.
    #[must_use]
    pub fn growth_stage(&self) -> Option<usize> {
        self.planting
            .map(|planting| crop(planting.crop).stage_at(planting.growth_days))
    }
}

/// What happened when the player acted on a tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FarmAction {
    /// Soil was broken into a plot.
    Tilled,
    /// A plot was watered.
    Watered,
    /// A seed was planted.
    Planted(CropId),
    /// A crop was picked, yielding this item and count.
    Harvested {
        /// What was picked.
        item: ItemId,
        /// How many.
        count: u32,
    },
    /// The tile was cleared back to untilled ground.
    Cleared,
    /// Nothing applied here.
    Nothing,
}

/// A summary of what changed overnight.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct OvernightReport {
    /// Crops that advanced a day.
    pub grown: u32,
    /// Crops that did not, because they were not watered.
    pub thirsty: u32,
    /// Plots cleared because their crop cannot survive the new season.
    pub killed_by_season: u32,
    /// Plots that reverted to untilled ground.
    pub reverted: u32,
}

/// Every worked tile on the farm.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Farm {
    /// Worked tiles, keyed by coordinate.
    ///
    /// `BTreeMap` rather than `HashMap` so iteration is in a fixed coordinate
    /// order: the overnight pass draws from the random generator per plot, and
    /// an unordered walk would make the whole day non-reproducible.
    ///
    /// Persisted as a sequence rather than a map, because JSON object keys must
    /// be strings and a coordinate pair is not one.
    #[serde(with = "plot_map")]
    plots: BTreeMap<(i32, i32), Plot>,
}

/// Serialises the plot map as a sorted sequence of `(x, y, plot)` triples.
mod plot_map {
    use super::Plot;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    /// Writes the map as a sequence, in coordinate order.
    pub(super) fn serialize<S: Serializer>(
        plots: &BTreeMap<(i32, i32), Plot>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let entries: Vec<(i32, i32, Plot)> =
            plots.iter().map(|((x, y), plot)| (*x, *y, *plot)).collect();
        entries.serialize(serializer)
    }

    /// Rebuilds the map from that sequence.
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<(i32, i32), Plot>, D::Error> {
        let entries = Vec::<(i32, i32, Plot)>::deserialize(deserializer)?;
        Ok(entries
            .into_iter()
            .map(|(x, y, plot)| ((x, y), plot))
            .collect())
    }
}

impl Farm {
    /// An unworked farm.
    #[must_use]
    pub fn new() -> Farm {
        Farm::default()
    }

    /// The plot at a tile, if it has been worked.
    #[must_use]
    pub fn plot(&self, cell: IVec2) -> Option<&Plot> {
        self.plots.get(&(cell.x, cell.y))
    }

    /// Mutable access to a plot.
    pub fn plot_mut(&mut self, cell: IVec2) -> Option<&mut Plot> {
        self.plots.get_mut(&(cell.x, cell.y))
    }

    /// Every worked tile, in coordinate order.
    pub fn iter(&self) -> impl Iterator<Item = (IVec2, &Plot)> {
        self.plots
            .iter()
            .map(|((x, y), plot)| (IVec2::new(*x, *y), plot))
    }

    /// Number of worked tiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plots.len()
    }

    /// True when nothing has been worked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plots.is_empty()
    }

    /// Breaks ground into a plot.
    ///
    /// Tilling an existing plot does nothing rather than destroying whatever
    /// grows there — an accidental hoe swing over a ripe crop should not
    /// silently delete a season's work.
    pub fn till(&mut self, cell: IVec2) -> FarmAction {
        if self.plots.contains_key(&(cell.x, cell.y)) {
            return FarmAction::Nothing;
        }
        self.plots.insert((cell.x, cell.y), Plot::tilled());
        FarmAction::Tilled
    }

    /// Waters a plot.
    pub fn water(&mut self, cell: IVec2) -> FarmAction {
        match self.plots.get_mut(&(cell.x, cell.y)) {
            Some(plot) if !plot.watered => {
                plot.watered = true;
                FarmAction::Watered
            }
            _ => FarmAction::Nothing,
        }
    }

    /// Plants a seed on an empty plot.
    ///
    /// Refuses out of season, because a crop planted in the wrong season would
    /// simply never grow — telling the player up front is better than letting
    /// them discover it a week later.
    pub fn plant(&mut self, cell: IVec2, id: CropId, season: Season) -> FarmAction {
        if !crop(id).grows_in(season) {
            return FarmAction::Nothing;
        }
        match self.plots.get_mut(&(cell.x, cell.y)) {
            Some(plot) if plot.planting.is_none() => {
                plot.planting = Some(Planting {
                    crop: id,
                    growth_days: 0,
                    harvested_once: false,
                });
                FarmAction::Planted(id)
            }
            _ => FarmAction::Nothing,
        }
    }

    /// Harvests a mature crop.
    ///
    /// A crop with a regrow stage returns to that stage and keeps producing;
    /// everything else leaves bare tilled soil behind.
    pub fn harvest(&mut self, cell: IVec2, rng: &mut Rng) -> FarmAction {
        let Some(plot) = self.plots.get_mut(&(cell.x, cell.y)) else {
            return FarmAction::Nothing;
        };
        let Some(planting) = plot.planting else {
            return FarmAction::Nothing;
        };
        let definition = crop(planting.crop);
        if !definition.is_mature(planting.growth_days) {
            return FarmAction::Nothing;
        }

        // Most harvests yield one, with an occasional bonus. A flat yield makes
        // every harvest identical; a wide spread makes planning impossible.
        let count = if rng.chance(1, 5) { 2 } else { 1 };

        match definition.regrow_stage {
            Some(stage) => {
                // Rewind to the start of the regrow stage rather than to zero,
                // so a second crop arrives in days rather than a full cycle.
                let days: u32 = definition.stage_days.iter().take(stage as usize).sum();
                plot.planting = Some(Planting {
                    crop: planting.crop,
                    growth_days: days,
                    harvested_once: true,
                });
            }
            None => plot.planting = None,
        }

        FarmAction::Harvested {
            item: definition.produce,
            count,
        }
    }

    /// Clears a tile back to unworked ground, destroying anything on it.
    pub fn clear(&mut self, cell: IVec2) -> FarmAction {
        if self.plots.remove(&(cell.x, cell.y)).is_some() {
            FarmAction::Cleared
        } else {
            FarmAction::Nothing
        }
    }

    /// Runs the overnight pass.
    ///
    /// Crops advance only if watered, either by hand or by the weather; the
    /// watered flag then clears for the new day. Empty plots gradually revert
    /// to unworked ground, so a farm left alone does not stay tilled forever.
    pub fn advance_day(
        &mut self,
        season: Season,
        weather: Weather,
        rng: &mut Rng,
    ) -> OvernightReport {
        let mut report = OvernightReport::default();
        let rained = weather.waters_crops();
        let mut to_remove = Vec::new();

        // BTreeMap iteration is in coordinate order, so the random draws below
        // happen in the same sequence on every machine.
        for (position, plot) in &mut self.plots {
            let watered = plot.watered || rained;

            if let Some(planting) = plot.planting.as_mut() {
                let definition = crop(planting.crop);
                if !definition.grows_in(season) {
                    // The season turned under a growing crop; it dies.
                    plot.planting = None;
                    report.killed_by_season += 1;
                } else if watered {
                    planting.growth_days += 1;
                    report.grown += 1;
                } else {
                    report.thirsty += 1;
                }
            } else if rng.chance(1, 10) {
                // An empty plot slowly goes back to grass.
                to_remove.push(*position);
                report.reverted += 1;
            }

            plot.watered = false;
        }

        for position in to_remove {
            self.plots.remove(&position);
        }
        report
    }

    /// Folds the farm's state into a hash, for the determinism tests.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        let mut hasher = verdant_core_math::StateHasher::new();
        for ((x, y), plot) in &self.plots {
            hasher.write_u32(*x as u32);
            hasher.write_u32(*y as u32);
            hasher.write_u32(u32::from(plot.watered));
            match plot.planting {
                Some(planting) => {
                    hasher.write_u32(planting.crop as u32);
                    hasher.write_u32(planting.growth_days);
                    hasher.write_u32(u32::from(planting.harvested_once));
                }
                None => hasher.write_u32(0),
            }
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(x: i32, y: i32) -> IVec2 {
        IVec2::new(x, y)
    }

    /// A farm with one tilled, planted parsnip plot.
    fn planted_farm() -> Farm {
        let mut farm = Farm::new();
        farm.till(cell(0, 0));
        farm.plant(cell(0, 0), CropId::Parsnip, Season::Spring);
        farm
    }

    #[test]
    fn tilling_creates_a_plot() {
        let mut farm = Farm::new();
        assert_eq!(farm.till(cell(1, 1)), FarmAction::Tilled);
        assert!(farm.plot(cell(1, 1)).is_some());
        assert_eq!(farm.len(), 1);
    }

    #[test]
    fn tilling_an_existing_plot_does_not_destroy_its_crop() {
        let mut farm = planted_farm();
        assert_eq!(farm.till(cell(0, 0)), FarmAction::Nothing);
        assert!(
            farm.plot(cell(0, 0)).unwrap().planting.is_some(),
            "the crop survived"
        );
    }

    #[test]
    fn watering_only_applies_to_tilled_soil() {
        let mut farm = Farm::new();
        assert_eq!(
            farm.water(cell(5, 5)),
            FarmAction::Nothing,
            "untilled ground"
        );
        farm.till(cell(5, 5));
        assert_eq!(farm.water(cell(5, 5)), FarmAction::Watered);
        assert_eq!(
            farm.water(cell(5, 5)),
            FarmAction::Nothing,
            "already watered"
        );
    }

    #[test]
    fn planting_requires_tilled_and_empty_soil() {
        let mut farm = Farm::new();
        assert_eq!(
            farm.plant(cell(0, 0), CropId::Parsnip, Season::Spring),
            FarmAction::Nothing,
            "untilled ground cannot be planted"
        );

        farm.till(cell(0, 0));
        assert_eq!(
            farm.plant(cell(0, 0), CropId::Parsnip, Season::Spring),
            FarmAction::Planted(CropId::Parsnip)
        );
        assert_eq!(
            farm.plant(cell(0, 0), CropId::Cauliflower, Season::Spring),
            FarmAction::Nothing,
            "an occupied plot cannot be replanted"
        );
    }

    #[test]
    fn planting_out_of_season_is_refused() {
        let mut farm = Farm::new();
        farm.till(cell(0, 0));
        // Parsnips are a spring crop.
        assert_eq!(
            farm.plant(cell(0, 0), CropId::Parsnip, Season::Autumn),
            FarmAction::Nothing
        );
        assert!(farm.plot(cell(0, 0)).unwrap().planting.is_none());
    }

    #[test]
    fn a_watered_crop_grows_overnight() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(1);
        farm.water(cell(0, 0));

        let report = farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        assert_eq!(report.grown, 1);
        assert_eq!(
            farm.plot(cell(0, 0)).unwrap().planting.unwrap().growth_days,
            1
        );
    }

    #[test]
    fn an_unwatered_crop_does_not_grow() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(2);

        let report = farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        assert_eq!(report.thirsty, 1);
        assert_eq!(
            farm.plot(cell(0, 0)).unwrap().planting.unwrap().growth_days,
            0
        );
    }

    #[test]
    fn rain_waters_every_crop() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(3);

        let report = farm.advance_day(Season::Spring, Weather::Rain, &mut rng);
        assert_eq!(report.grown, 1, "rain should have watered it");
        assert_eq!(report.thirsty, 0);
    }

    #[test]
    fn snow_does_not_water_crops() {
        let mut farm = Farm::new();
        farm.till(cell(0, 0));
        farm.plant(cell(0, 0), CropId::Cranberry, Season::Autumn);
        let mut rng = Rng::new(4);

        let report = farm.advance_day(Season::Autumn, Weather::Snow, &mut rng);
        assert_eq!(report.thirsty, 1, "snow is not irrigation");
    }

    #[test]
    fn the_watered_flag_clears_each_morning() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(5);
        farm.water(cell(0, 0));
        assert!(farm.plot(cell(0, 0)).unwrap().watered);

        farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        assert!(
            !farm.plot(cell(0, 0)).unwrap().watered,
            "it must be watered again"
        );
    }

    #[test]
    fn a_crop_matures_after_its_stages_and_can_be_harvested() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(6);
        let days = crop(CropId::Parsnip).days_to_maturity();

        for _ in 0..days {
            farm.water(cell(0, 0));
            farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        }
        assert!(farm.plot(cell(0, 0)).unwrap().is_harvestable());

        match farm.harvest(cell(0, 0), &mut rng) {
            FarmAction::Harvested { item, count } => {
                assert_eq!(item, ItemId::Parsnip);
                assert!(count >= 1);
            }
            other => panic!("expected a harvest, got {other:?}"),
        }
    }

    #[test]
    fn an_immature_crop_cannot_be_harvested() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(7);
        assert_eq!(farm.harvest(cell(0, 0), &mut rng), FarmAction::Nothing);
        assert!(
            farm.plot(cell(0, 0)).unwrap().planting.is_some(),
            "and it stays planted"
        );
    }

    #[test]
    fn a_single_harvest_crop_leaves_bare_soil() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(8);
        for _ in 0..crop(CropId::Parsnip).days_to_maturity() {
            farm.water(cell(0, 0));
            farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        }
        farm.harvest(cell(0, 0), &mut rng);

        let plot = farm.plot(cell(0, 0)).expect("the plot remains");
        assert!(plot.planting.is_none(), "the plant is gone");
    }

    #[test]
    fn a_regrowing_crop_stays_and_ripens_again() {
        let mut farm = Farm::new();
        farm.till(cell(0, 0));
        farm.plant(cell(0, 0), CropId::Tomato, Season::Summer);
        let mut rng = Rng::new(9);

        let definition = crop(CropId::Tomato);
        for _ in 0..definition.days_to_maturity() {
            farm.water(cell(0, 0));
            farm.advance_day(Season::Summer, Weather::Clear, &mut rng);
        }
        assert!(matches!(
            farm.harvest(cell(0, 0), &mut rng),
            FarmAction::Harvested { .. }
        ));

        let planting = farm
            .plot(cell(0, 0))
            .unwrap()
            .planting
            .expect("the plant stays");
        assert!(planting.harvested_once);
        assert!(
            !farm.plot(cell(0, 0)).unwrap().is_harvestable(),
            "but is not ripe yet"
        );

        // A few more watered days and it is ready again, sooner than a fresh
        // planting would have been.
        let regrow_days = definition.days_to_maturity() - planting.growth_days;
        assert!(regrow_days < definition.days_to_maturity());
        for _ in 0..regrow_days {
            farm.water(cell(0, 0));
            farm.advance_day(Season::Summer, Weather::Clear, &mut rng);
        }
        assert!(farm.plot(cell(0, 0)).unwrap().is_harvestable());
    }

    #[test]
    fn a_crop_dies_when_its_season_ends() {
        let mut farm = planted_farm();
        let mut rng = Rng::new(10);
        farm.water(cell(0, 0));

        let report = farm.advance_day(Season::Summer, Weather::Clear, &mut rng);
        assert_eq!(report.killed_by_season, 1);
        assert!(farm.plot(cell(0, 0)).unwrap().planting.is_none());
    }

    #[test]
    fn empty_plots_eventually_revert_to_grass() {
        let mut farm = Farm::new();
        for x in 0..40 {
            farm.till(cell(x, 0));
        }
        let mut rng = Rng::new(11);
        for _ in 0..30 {
            farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        }
        assert!(
            farm.len() < 40,
            "an abandoned farm should not stay tilled forever"
        );
    }

    #[test]
    fn planted_plots_never_revert() {
        let mut farm = Farm::new();
        for x in 0..20 {
            farm.till(cell(x, 0));
            farm.plant(cell(x, 0), CropId::Parsnip, Season::Spring);
        }
        let mut rng = Rng::new(12);
        for _ in 0..30 {
            // Not watered, so nothing grows, but nothing should vanish either.
            farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
        }
        assert_eq!(farm.len(), 20, "a planted plot must not be reclaimed");
    }

    #[test]
    fn clearing_removes_the_plot_entirely() {
        let mut farm = planted_farm();
        assert_eq!(farm.clear(cell(0, 0)), FarmAction::Cleared);
        assert!(farm.plot(cell(0, 0)).is_none());
        assert_eq!(farm.clear(cell(0, 0)), FarmAction::Nothing);
    }

    #[test]
    fn iteration_is_in_coordinate_order() {
        let mut farm = Farm::new();
        for (x, y) in [(5, 5), (1, 1), (3, 2), (0, 9)] {
            farm.till(cell(x, y));
        }
        let order: Vec<(i32, i32)> = farm.iter().map(|(cell, _)| (cell.x, cell.y)).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(
            order, sorted,
            "overnight growth draws randomness in this order"
        );
    }

    #[test]
    fn determinism_a_full_season_reproduces() {
        let run = || {
            let mut farm = Farm::new();
            let mut rng = Rng::new(2024);
            for x in 0..12 {
                for y in 0..12 {
                    farm.till(cell(x, y));
                    if (x + y) % 3 == 0 {
                        farm.plant(cell(x, y), CropId::Parsnip, Season::Spring);
                    }
                }
            }
            for day in 0..28 {
                if day % 2 == 0 {
                    for x in 0..12 {
                        for y in 0..12 {
                            farm.water(cell(x, y));
                        }
                    }
                }
                farm.advance_day(Season::Spring, Weather::Clear, &mut rng);
                for x in 0..12 {
                    for y in 0..12 {
                        farm.harvest(cell(x, y), &mut rng);
                    }
                }
            }
            farm.state_hash()
        };
        assert_eq!(run(), run(), "a season of farming must reproduce exactly");
    }

    #[test]
    fn a_farm_round_trips_through_serialisation() {
        let mut farm = planted_farm();
        farm.water(cell(0, 0));
        let encoded = serde_json::to_string(&farm).expect("serialisable");
        let decoded: Farm = serde_json::from_str(&encoded).expect("deserialisable");
        assert_eq!(decoded.state_hash(), farm.state_hash());
    }
}
