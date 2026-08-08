//! Items, tools, crops and the recipes that connect them.
//!
//! Everything the player can hold is an [`ItemId`] plus a quantity. The catalogue
//! is static data rather than components, because an item's *definition* is
//! shared by every instance of it — a hundred parsnips in a chest are one
//! definition and one count, not a hundred entities.

use serde::{Deserialize, Serialize};
use verdant_procgen_art::Rgba;

/// A stable identifier for an item definition.
///
/// Explicit numeric values rather than a derived discriminant, because these
/// end up in save files: reordering the enum must not silently turn everyone's
/// parsnips into stone.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u16)]
pub enum ItemId {
    // ---- Tools -------------------------------------------------------------
    /// Breaks ground into tilled soil.
    Hoe = 1,
    /// Waters tilled soil.
    WateringCan = 2,
    /// Chops trees.
    Axe = 3,
    /// Breaks stone and ore.
    Pickaxe = 4,
    /// Catches fish.
    FishingRod = 5,
    /// A weapon for the mine.
    Sword = 6,

    // ---- Seeds -------------------------------------------------------------
    /// Spring crop, quick and cheap.
    ParsnipSeeds = 20,
    /// Spring crop, higher value.
    CauliflowerSeeds = 21,
    /// Summer crop that regrows after harvest.
    TomatoSeeds = 22,
    /// Summer crop, the season's staple.
    MelonSeeds = 23,
    /// Autumn crop, regrows.
    CranberrySeeds = 24,
    /// Autumn crop, the year's most valuable.
    PumpkinSeeds = 25,

    // ---- Produce -----------------------------------------------------------
    /// Harvested parsnip.
    Parsnip = 40,
    /// Harvested cauliflower.
    Cauliflower = 41,
    /// Harvested tomato.
    Tomato = 42,
    /// Harvested melon.
    Melon = 43,
    /// Harvested cranberry.
    Cranberry = 44,
    /// Harvested pumpkin.
    Pumpkin = 45,

    // ---- Gathered ----------------------------------------------------------
    /// Chopped from trees; the basic building material.
    Wood = 60,
    /// Broken from rocks.
    Stone = 61,
    /// Smelted into bars, or sold.
    CopperOre = 62,
    /// A deeper-mine ore.
    IronOre = 63,
    /// The deepest ore, and the most valuable.
    GoldOre = 64,
    /// Found in the mine; the mine's real prize.
    Gemstone = 65,

    // ---- Fish --------------------------------------------------------------
    /// Common river fish.
    Chub = 80,
    /// Uncommon fish.
    Bass = 81,
    /// Rare fish, worth a day's farming.
    Sturgeon = 82,

    // ---- Foraged and cooked -------------------------------------------------
    /// Spring forage.
    Daffodil = 100,
    /// Autumn forage.
    Blackberry = 101,
    /// Restores a large amount of energy.
    VegetableStew = 120,
    /// Restores energy and sells well.
    FishDinner = 121,
}

/// What an item is for, which decides how the game treats it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum ItemKind {
    /// Used on the world; never consumed.
    Tool,
    /// Planted on tilled soil.
    Seed {
        /// What it grows into.
        crop: CropId,
    },
    /// Sold, gifted, or cooked with.
    Produce,
    /// A raw material.
    Material,
    /// Eaten for energy.
    Food {
        /// Energy restored.
        energy: i32,
    },
}

/// A crop's identity, separate from its seed and its produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u16)]
pub enum CropId {
    /// Fast, cheap spring crop.
    Parsnip = 1,
    /// Slow, valuable spring crop.
    Cauliflower = 2,
    /// Summer crop that keeps producing.
    Tomato = 3,
    /// Large summer crop.
    Melon = 4,
    /// Autumn crop that keeps producing.
    Cranberry = 5,
    /// The autumn payday.
    Pumpkin = 6,
}

/// Everything static about an item.
#[derive(Clone, Copy, Debug)]
pub struct ItemDef {
    /// The item's identity.
    pub id: ItemId,
    /// Display name.
    pub name: &'static str,
    /// What the item is for.
    pub kind: ItemKind,
    /// Base price when sold. Zero means it cannot be sold.
    pub sell_price: u32,
    /// Price when bought, or `None` when it is not stocked.
    pub buy_price: Option<u32>,
    /// Most of this item that fits in one inventory slot.
    pub stack_size: u32,
    /// The colour its generated icon is built from.
    pub tint: Rgba,
}

/// Everything static about a crop.
#[derive(Clone, Copy, Debug)]
pub struct CropDef {
    /// The crop's identity.
    pub id: CropId,
    /// Display name.
    pub name: &'static str,
    /// The seed that plants it.
    pub seed: ItemId,
    /// What harvesting yields.
    pub produce: ItemId,
    /// Days in each growth stage; the crop is harvestable after their sum.
    pub stage_days: &'static [u32],
    /// Seasons it will grow in. Planted out of season, it never advances.
    pub seasons: &'static [crate::calendar::Season],
    /// When set, harvesting returns the plant to this stage instead of
    /// clearing it, so it produces again after a few more days.
    pub regrow_stage: Option<u32>,
    /// The colour its leaves are generated from.
    pub leaf_tint: Rgba,
    /// The colour its fruit is generated from.
    pub fruit_tint: Rgba,
}

impl CropDef {
    /// Total days from planting to first harvest.
    #[must_use]
    pub fn days_to_maturity(&self) -> u32 {
        self.stage_days.iter().sum()
    }

    /// Number of growth stages, including the mature one.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stage_days.len() + 1
    }

    /// The stage a crop of this kind is in after `days` of growth.
    #[must_use]
    pub fn stage_at(&self, days: u32) -> usize {
        let mut remaining = days;
        for (stage, cost) in self.stage_days.iter().enumerate() {
            if remaining < *cost {
                return stage;
            }
            remaining -= cost;
        }
        self.stage_days.len()
    }

    /// True when a crop of this kind has been growing long enough to harvest.
    #[must_use]
    pub fn is_mature(&self, days: u32) -> bool {
        days >= self.days_to_maturity()
    }

    /// True when this crop grows in `season`.
    #[must_use]
    pub fn grows_in(&self, season: crate::calendar::Season) -> bool {
        self.seasons.contains(&season)
    }
}

use crate::calendar::Season;

/// Every crop in the game.
pub static CROPS: &[CropDef] = &[
    CropDef {
        id: CropId::Parsnip,
        name: "Parsnip",
        seed: ItemId::ParsnipSeeds,
        produce: ItemId::Parsnip,
        stage_days: &[1, 1, 1, 1],
        seasons: &[Season::Spring],
        regrow_stage: None,
        leaf_tint: Rgba::hex(0x6F_A8_3D),
        fruit_tint: Rgba::hex(0xE8_D8_9A),
    },
    CropDef {
        id: CropId::Cauliflower,
        name: "Cauliflower",
        seed: ItemId::CauliflowerSeeds,
        produce: ItemId::Cauliflower,
        stage_days: &[2, 2, 3, 3, 2],
        seasons: &[Season::Spring],
        regrow_stage: None,
        leaf_tint: Rgba::hex(0x5A_8C_3A),
        fruit_tint: Rgba::hex(0xF0_EE_D8),
    },
    CropDef {
        id: CropId::Tomato,
        name: "Tomato",
        seed: ItemId::TomatoSeeds,
        produce: ItemId::Tomato,
        stage_days: &[2, 2, 2, 2, 3],
        seasons: &[Season::Summer],
        // Regrows a few days after each harvest, which is what makes it worth
        // its higher seed price over a whole season.
        regrow_stage: Some(3),
        leaf_tint: Rgba::hex(0x4A_8C_3A),
        fruit_tint: Rgba::hex(0xD9_3A_2A),
    },
    CropDef {
        id: CropId::Melon,
        name: "Melon",
        seed: ItemId::MelonSeeds,
        produce: ItemId::Melon,
        stage_days: &[2, 3, 3, 4],
        seasons: &[Season::Summer],
        regrow_stage: None,
        leaf_tint: Rgba::hex(0x3D_7A_2E),
        fruit_tint: Rgba::hex(0x8C_C4_4A),
    },
    CropDef {
        id: CropId::Cranberry,
        name: "Cranberry",
        seed: ItemId::CranberrySeeds,
        produce: ItemId::Cranberry,
        stage_days: &[2, 2, 2, 1],
        seasons: &[Season::Autumn],
        regrow_stage: Some(2),
        leaf_tint: Rgba::hex(0x4A_7A_3A),
        fruit_tint: Rgba::hex(0xC4_2A_3A),
    },
    CropDef {
        id: CropId::Pumpkin,
        name: "Pumpkin",
        seed: ItemId::PumpkinSeeds,
        produce: ItemId::Pumpkin,
        stage_days: &[2, 3, 3, 3, 2],
        seasons: &[Season::Autumn],
        regrow_stage: None,
        leaf_tint: Rgba::hex(0x3D_6B_2A),
        fruit_tint: Rgba::hex(0xE0_8A_1E),
    },
];

/// Looks up a crop definition.
///
/// # Panics
///
/// Panics if the crop is missing from [`CROPS`], which would be a data bug
/// rather than a runtime condition.
#[must_use]
pub fn crop(id: CropId) -> &'static CropDef {
    CROPS
        .iter()
        .find(|def| def.id == id)
        .unwrap_or_else(|| panic!("crop {id:?} is missing from the catalogue"))
}

/// Every item in the game.
pub static ITEMS: &[ItemDef] = &[
    // Tools are never sold and never stack.
    tool(ItemId::Hoe, "Hoe", 0x8C_6A_3D),
    tool(ItemId::WateringCan, "Watering Can", 0x5A_8C_A8),
    tool(ItemId::Axe, "Axe", 0x9C_9C_A8),
    tool(ItemId::Pickaxe, "Pickaxe", 0x8C_8C_9C),
    tool(ItemId::FishingRod, "Fishing Rod", 0x7A_5A_3D),
    tool(ItemId::Sword, "Sword", 0xC4_C4_D8),
    // Seeds: bought from the shop, planted on tilled soil.
    seed(
        ItemId::ParsnipSeeds,
        "Parsnip Seeds",
        CropId::Parsnip,
        10,
        20,
        0x8C_A8_5A,
    ),
    seed(
        ItemId::CauliflowerSeeds,
        "Cauliflower Seeds",
        CropId::Cauliflower,
        40,
        80,
        0x9C_B8_6A,
    ),
    seed(
        ItemId::TomatoSeeds,
        "Tomato Seeds",
        CropId::Tomato,
        25,
        50,
        0x7A_A8_4A,
    ),
    seed(
        ItemId::MelonSeeds,
        "Melon Seeds",
        CropId::Melon,
        40,
        80,
        0x6A_A8_3D,
    ),
    seed(
        ItemId::CranberrySeeds,
        "Cranberry Seeds",
        CropId::Cranberry,
        120,
        240,
        0x8C_A8_5A,
    ),
    seed(
        ItemId::PumpkinSeeds,
        "Pumpkin Seeds",
        CropId::Pumpkin,
        50,
        100,
        0x7A_9C_4A,
    ),
    // Produce: sold or cooked. Prices are tuned so a crop returns roughly two
    // to three times its seed cost, which keeps farming worthwhile without
    // making the first season trivial.
    produce(ItemId::Parsnip, "Parsnip", 35, 0xE8_D8_9A),
    produce(ItemId::Cauliflower, "Cauliflower", 175, 0xF0_EE_D8),
    produce(ItemId::Tomato, "Tomato", 60, 0xD9_3A_2A),
    produce(ItemId::Melon, "Melon", 250, 0x8C_C4_4A),
    produce(ItemId::Cranberry, "Cranberry", 75, 0xC4_2A_3A),
    produce(ItemId::Pumpkin, "Pumpkin", 320, 0xE0_8A_1E),
    // Materials.
    material(ItemId::Wood, "Wood", 2, 0x8C_6A_43),
    material(ItemId::Stone, "Stone", 2, 0x9C_9C_A0),
    material(ItemId::CopperOre, "Copper Ore", 5, 0xC4_7A_3D),
    material(ItemId::IronOre, "Iron Ore", 10, 0xA8_A8_B5),
    material(ItemId::GoldOre, "Gold Ore", 25, 0xE0_C4_4A),
    material(ItemId::Gemstone, "Gemstone", 150, 0x9C_5A_C4),
    // Fish.
    produce(ItemId::Chub, "Chub", 50, 0x6A_8C_A8),
    produce(ItemId::Bass, "Bass", 110, 0x5A_7A_9C),
    produce(ItemId::Sturgeon, "Sturgeon", 300, 0x4A_5A_6B),
    // Forage.
    produce(ItemId::Daffodil, "Daffodil", 30, 0xE8_D8_4A),
    produce(ItemId::Blackberry, "Blackberry", 20, 0x3A_2A_4A),
    // Cooked food.
    food(
        ItemId::VegetableStew,
        "Vegetable Stew",
        180,
        120,
        0xC4_7A_3D,
    ),
    food(ItemId::FishDinner, "Fish Dinner", 240, 150, 0xD8_C4_9C),
];

/// Builds a tool definition.
const fn tool(id: ItemId, name: &'static str, tint: u32) -> ItemDef {
    ItemDef {
        id,
        name,
        kind: ItemKind::Tool,
        sell_price: 0,
        buy_price: None,
        // Tools do not stack: two hoes in one slot would be meaningless, and
        // the hotbar addresses a slot rather than an item.
        stack_size: 1,
        tint: Rgba::hex(tint),
    }
}

/// Builds a seed definition.
const fn seed(
    id: ItemId,
    name: &'static str,
    crop: CropId,
    sell: u32,
    buy: u32,
    tint: u32,
) -> ItemDef {
    ItemDef {
        id,
        name,
        kind: ItemKind::Seed { crop },
        sell_price: sell,
        buy_price: Some(buy),
        stack_size: 999,
        tint: Rgba::hex(tint),
    }
}

/// Builds a produce definition.
const fn produce(id: ItemId, name: &'static str, sell: u32, tint: u32) -> ItemDef {
    ItemDef {
        id,
        name,
        kind: ItemKind::Produce,
        sell_price: sell,
        buy_price: None,
        stack_size: 999,
        tint: Rgba::hex(tint),
    }
}

/// Builds a material definition.
const fn material(id: ItemId, name: &'static str, sell: u32, tint: u32) -> ItemDef {
    ItemDef {
        id,
        name,
        kind: ItemKind::Material,
        sell_price: sell,
        buy_price: None,
        stack_size: 999,
        tint: Rgba::hex(tint),
    }
}

/// Builds a food definition.
const fn food(id: ItemId, name: &'static str, sell: u32, energy: i32, tint: u32) -> ItemDef {
    ItemDef {
        id,
        name,
        kind: ItemKind::Food { energy },
        sell_price: sell,
        buy_price: None,
        stack_size: 99,
        tint: Rgba::hex(tint),
    }
}

/// Looks up an item definition.
///
/// # Panics
///
/// Panics if the item is missing from [`ITEMS`], which is a data bug.
#[must_use]
pub fn item(id: ItemId) -> &'static ItemDef {
    ITEMS
        .iter()
        .find(|def| def.id == id)
        .unwrap_or_else(|| panic!("item {id:?} is missing from the catalogue"))
}

/// A recipe that turns ingredients into a product.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// What it makes.
    pub output: ItemId,
    /// How many it makes.
    pub output_count: u32,
    /// What it consumes, as `(item, count)` pairs.
    pub ingredients: &'static [(ItemId, u32)],
}

/// Every recipe the player can make.
pub static RECIPES: &[Recipe] = &[
    Recipe {
        output: ItemId::VegetableStew,
        output_count: 1,
        // Three parsnips rather than a cauliflower: a starter recipe should be
        // makeable from the crop the player already has on day one, and its
        // ingredients must be worth less than the result.
        ingredients: &[(ItemId::Parsnip, 3)],
    },
    Recipe {
        output: ItemId::FishDinner,
        output_count: 1,
        ingredients: &[(ItemId::Chub, 1), (ItemId::Parsnip, 1)],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_catalogued_item_is_unique() {
        let ids: HashSet<ItemId> = ITEMS.iter().map(|def| def.id).collect();
        assert_eq!(
            ids.len(),
            ITEMS.len(),
            "a duplicate item id would shadow a definition"
        );
    }

    #[test]
    fn every_item_can_be_looked_up() {
        for def in ITEMS {
            assert_eq!(item(def.id).id, def.id);
            assert!(!def.name.is_empty());
            assert!(def.stack_size > 0);
        }
    }

    #[test]
    fn every_crop_is_reachable_from_its_seed() {
        for def in CROPS {
            let seed_def = item(def.seed);
            match seed_def.kind {
                ItemKind::Seed { crop } => assert_eq!(crop, def.id),
                other => panic!("{} is catalogued as {other:?}, not a seed", seed_def.name),
            }
            // And its produce must exist too.
            assert!(
                item(def.produce).sell_price > 0,
                "{} sells for nothing",
                def.name
            );
        }
    }

    #[test]
    fn every_seed_grows_something() {
        for def in ITEMS {
            if let ItemKind::Seed { crop: id } = def.kind {
                assert_eq!(crop(id).seed, def.id, "{} does not round trip", def.name);
            }
        }
    }

    #[test]
    fn tools_do_not_stack_and_cannot_be_sold() {
        for def in ITEMS.iter().filter(|def| def.kind == ItemKind::Tool) {
            assert_eq!(def.stack_size, 1, "{} should not stack", def.name);
            assert_eq!(def.sell_price, 0, "{} should not be sellable", def.name);
        }
    }

    #[test]
    fn every_crop_turns_a_profit_over_a_season() {
        // The economy only works if farming pays; this catches a mistuned price
        // before it reaches a player. Regrowing crops must be judged on their
        // whole-season yield, not on one harvest — that is exactly what their
        // higher seed price buys.
        for def in CROPS {
            let seed_cost = item(def.seed).buy_price.expect("seeds are stocked");
            let per_harvest = item(def.produce).sell_price;
            let maturity = def.days_to_maturity();

            let harvests = match def.regrow_stage {
                Some(stage) => {
                    let regrow_days =
                        maturity - def.stage_days.iter().take(stage as usize).sum::<u32>();
                    1 + Season::DAYS.saturating_sub(maturity) / regrow_days.max(1)
                }
                None => 1,
            };
            let season_yield = per_harvest * harvests;
            assert!(
                season_yield > seed_cost,
                "{} yields {season_yield} over a season but its seeds cost {seed_cost}",
                def.name
            );
        }
    }

    #[test]
    fn regrowing_crops_out_earn_their_higher_seed_price() {
        // A tomato costs twice a parsnip's seed; over a season it must return
        // more, or there would be no reason to ever buy one.
        let parsnip = crop(CropId::Parsnip);
        let tomato = crop(CropId::Tomato);
        assert!(tomato.regrow_stage.is_some());
        assert!(item(tomato.seed).buy_price > item(parsnip.seed).buy_price);
        assert!(item(tomato.produce).sell_price > item(parsnip.produce).sell_price);
    }

    #[test]
    fn growth_stages_advance_and_then_stop() {
        let parsnip = crop(CropId::Parsnip);
        assert_eq!(parsnip.stage_at(0), 0);
        assert_eq!(parsnip.stage_at(1), 1);
        assert_eq!(parsnip.stage_at(3), 3);
        assert_eq!(parsnip.stage_at(4), 4, "the mature stage");
        assert_eq!(parsnip.stage_at(100), 4, "and it does not grow past it");
    }

    #[test]
    fn maturity_matches_the_sum_of_the_stages() {
        for def in CROPS {
            let total: u32 = def.stage_days.iter().sum();
            assert_eq!(def.days_to_maturity(), total);
            assert!(!def.is_mature(total - 1));
            assert!(def.is_mature(total));
            assert_eq!(def.stage_at(total), def.stage_count() - 1);
        }
    }

    #[test]
    fn every_crop_grows_in_at_least_one_season() {
        for def in CROPS {
            assert!(!def.seasons.is_empty(), "{} grows nowhere", def.name);
            assert!(
                !def.grows_in(Season::Winter),
                "nothing grows outdoors in winter"
            );
        }
    }

    #[test]
    fn regrowing_crops_return_to_a_valid_stage() {
        for def in CROPS {
            if let Some(stage) = def.regrow_stage {
                assert!(
                    (stage as usize) < def.stage_count() - 1,
                    "{} regrows to its mature stage, so it would never need to regrow",
                    def.name
                );
            }
        }
    }

    #[test]
    fn regrowing_crops_cost_more_than_single_harvest_ones() {
        // A crop that yields all season should not also be the cheapest.
        let single = item(crop(CropId::Parsnip).seed).buy_price.unwrap();
        let regrowing = item(crop(CropId::Tomato).seed).buy_price.unwrap();
        assert!(regrowing > single);
    }

    #[test]
    fn every_recipe_uses_real_items() {
        for recipe in RECIPES {
            assert!(recipe.output_count > 0);
            assert!(!recipe.ingredients.is_empty());
            let _ = item(recipe.output);
            for (ingredient, count) in recipe.ingredients {
                assert!(*count > 0);
                let _ = item(*ingredient);
            }
        }
    }

    #[test]
    fn cooked_food_is_worth_more_than_its_ingredients() {
        // Cooking must add value, or there is no reason to do it.
        for recipe in RECIPES {
            let ingredient_value: u32 = recipe
                .ingredients
                .iter()
                .map(|(id, count)| item(*id).sell_price * count)
                .sum();
            let output_value = item(recipe.output).sell_price * recipe.output_count;
            assert!(
                output_value > ingredient_value,
                "{:?} is worth {output_value} but costs {ingredient_value} to make",
                recipe.output
            );
        }
    }

    #[test]
    fn food_restores_energy() {
        for def in ITEMS {
            if let ItemKind::Food { energy } = def.kind {
                assert!(energy > 0, "{} restores nothing", def.name);
            }
        }
    }

    #[test]
    fn item_ids_are_stable_across_reordering() {
        // The numeric values end up in save files, so they are asserted here:
        // changing one silently rewrites what every existing save contains.
        assert_eq!(ItemId::Hoe as u16, 1);
        assert_eq!(ItemId::ParsnipSeeds as u16, 20);
        assert_eq!(ItemId::Parsnip as u16, 40);
        assert_eq!(ItemId::Wood as u16, 60);
        assert_eq!(ItemId::Chub as u16, 80);
    }
}
