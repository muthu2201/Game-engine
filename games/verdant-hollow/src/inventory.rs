//! The player's inventory, hotbar and wallet.

use crate::items::{item, ItemId, ItemKind, Recipe, RECIPES};
use serde::{Deserialize, Serialize};

/// One inventory slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Slot {
    /// What is in the slot.
    pub item: ItemId,
    /// How many. Never zero — an empty slot is `None`, not a zero count.
    pub count: u32,
}

/// How many slots the player carries.
///
/// Twelve is a deliberate constraint: an inventory that holds everything
/// removes the decision of what to bring into the mine, which is one of the
/// few real choices a farming day contains.
pub const INVENTORY_SLOTS: usize = 12;

/// How many of those slots are on the hotbar.
pub const HOTBAR_SLOTS: usize = 6;

/// The player's carried items and money.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    /// Slot contents; `None` is an empty slot.
    slots: Vec<Option<Slot>>,
    /// The currently selected hotbar slot.
    selected: usize,
    /// Money.
    pub gold: u32,
}

impl Inventory {
    /// An empty inventory with the starting purse.
    #[must_use]
    pub fn new(gold: u32) -> Inventory {
        Inventory {
            slots: vec![None; INVENTORY_SLOTS],
            selected: 0,
            gold,
        }
    }

    /// The inventory a new game begins with: the four basic tools and a few
    /// parsnip seeds, which is enough to plant on the first morning.
    #[must_use]
    pub fn starting() -> Inventory {
        let mut inventory = Inventory::new(500);
        for tool in [
            ItemId::Hoe,
            ItemId::WateringCan,
            ItemId::Axe,
            ItemId::Pickaxe,
        ] {
            inventory.add(tool, 1);
        }
        inventory.add(ItemId::ParsnipSeeds, 15);
        inventory
    }

    /// The slots, for the inventory screen.
    #[must_use]
    pub fn slots(&self) -> &[Option<Slot>] {
        &self.slots
    }

    /// The selected hotbar index.
    #[must_use]
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    /// The item in the selected slot, if any.
    #[must_use]
    pub fn selected(&self) -> Option<Slot> {
        self.slots.get(self.selected).copied().flatten()
    }

    /// Selects a hotbar slot, ignoring an out-of-range index.
    pub fn select(&mut self, index: usize) {
        if index < HOTBAR_SLOTS {
            self.selected = index;
        }
    }

    /// Moves the selection one slot along, wrapping at the ends.
    pub fn cycle_selection(&mut self, forward: bool) {
        self.selected = if forward {
            (self.selected + 1) % HOTBAR_SLOTS
        } else {
            (self.selected + HOTBAR_SLOTS - 1) % HOTBAR_SLOTS
        };
    }

    /// Adds items, returning how many did not fit.
    ///
    /// Filling partial stacks before opening a new slot is what keeps an
    /// inventory from filling up with six half-stacks of the same crop.
    pub fn add(&mut self, id: ItemId, mut count: u32) -> u32 {
        let stack_size = item(id).stack_size;

        for slot in self.slots.iter_mut().flatten() {
            if count == 0 {
                return 0;
            }
            if slot.item != id || slot.count >= stack_size {
                continue;
            }
            let space = stack_size - slot.count;
            let moved = space.min(count);
            slot.count += moved;
            count -= moved;
        }

        for slot in &mut self.slots {
            if count == 0 {
                return 0;
            }
            if slot.is_none() {
                let moved = stack_size.min(count);
                *slot = Some(Slot {
                    item: id,
                    count: moved,
                });
                count -= moved;
            }
        }
        count
    }

    /// True when `count` of an item could be added without overflowing.
    #[must_use]
    pub fn has_room_for(&self, id: ItemId, count: u32) -> bool {
        let stack_size = item(id).stack_size;
        let in_partial_stacks: u32 = self
            .slots
            .iter()
            .flatten()
            .filter(|slot| slot.item == id)
            .map(|slot| stack_size.saturating_sub(slot.count))
            .sum();
        let empty = self.slots.iter().filter(|slot| slot.is_none()).count();
        let in_empty = u32::try_from(empty)
            .unwrap_or(u32::MAX)
            .saturating_mul(stack_size);
        in_partial_stacks.saturating_add(in_empty) >= count
    }

    /// How many of an item the player is carrying.
    #[must_use]
    pub fn count_of(&self, id: ItemId) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.item == id)
            .map(|slot| slot.count)
            .sum()
    }

    /// Removes items, returning whether there were enough.
    ///
    /// All-or-nothing: a partial removal would leave a recipe half-consumed
    /// with nothing to show for it.
    pub fn remove(&mut self, id: ItemId, count: u32) -> bool {
        if self.count_of(id) < count {
            return false;
        }
        let mut remaining = count;
        for slot in &mut self.slots {
            if remaining == 0 {
                break;
            }
            let Some(contents) = slot else {
                continue;
            };
            if contents.item != id {
                continue;
            }
            let taken = contents.count.min(remaining);
            contents.count -= taken;
            remaining -= taken;
            if contents.count == 0 {
                *slot = None;
            }
        }
        true
    }

    /// Removes one item from a specific slot.
    pub fn consume_from_slot(&mut self, index: usize) -> Option<ItemId> {
        let slot = self.slots.get_mut(index)?;
        let contents = slot.as_mut()?;
        let id = contents.item;
        contents.count -= 1;
        if contents.count == 0 {
            *slot = None;
        }
        Some(id)
    }

    /// Sells `count` of an item, adding the proceeds to the purse.
    ///
    /// Returns the gold received, or `None` when the player does not have them
    /// or the item cannot be sold.
    pub fn sell(&mut self, id: ItemId, count: u32) -> Option<u32> {
        let price = item(id).sell_price;
        if price == 0 || !self.remove(id, count) {
            return None;
        }
        let proceeds = price.saturating_mul(count);
        self.gold = self.gold.saturating_add(proceeds);
        Some(proceeds)
    }

    /// Buys `count` of an item.
    ///
    /// Returns the gold spent, or `None` when the item is not stocked, the
    /// player cannot afford it, or there is no room for it.
    pub fn buy(&mut self, id: ItemId, count: u32) -> Option<u32> {
        let price = item(id).buy_price?;
        let cost = price.checked_mul(count)?;
        if self.gold < cost || !self.has_room_for(id, count) {
            return None;
        }
        self.gold -= cost;
        // The room check above guarantees this fits.
        self.add(id, count);
        Some(cost)
    }

    /// True when every ingredient for a recipe is carried.
    #[must_use]
    pub fn can_craft(&self, recipe: &Recipe) -> bool {
        recipe
            .ingredients
            .iter()
            .all(|(id, count)| self.count_of(*id) >= *count)
            && self.has_room_for(recipe.output, recipe.output_count)
    }

    /// Consumes a recipe's ingredients and adds its output.
    ///
    /// Returns whether the recipe was made.
    pub fn craft(&mut self, recipe: &Recipe) -> bool {
        if !self.can_craft(recipe) {
            return false;
        }
        for (id, count) in recipe.ingredients {
            // Guaranteed by can_craft; a failure here would mean the checks
            // and the mutation had drifted apart.
            debug_assert!(self.count_of(*id) >= *count);
            self.remove(*id, *count);
        }
        self.add(recipe.output, recipe.output_count);
        true
    }

    /// Every recipe the player can currently make.
    #[must_use]
    pub fn available_recipes(&self) -> Vec<&'static Recipe> {
        RECIPES
            .iter()
            .filter(|recipe| self.can_craft(recipe))
            .collect()
    }

    /// Eats the item in a slot, returning the energy it restores.
    ///
    /// Returns `None` when the slot is empty or holds something inedible.
    pub fn eat_from_slot(&mut self, index: usize) -> Option<i32> {
        let slot = self.slots.get(index).copied().flatten()?;
        let ItemKind::Food { energy } = item(slot.item).kind else {
            return None;
        };
        self.consume_from_slot(index);
        Some(energy)
    }

    /// Number of occupied slots.
    #[must_use]
    pub fn used_slots(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    /// True when every slot is occupied.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.used_slots() == self.slots.len()
    }
}

impl Default for Inventory {
    fn default() -> Inventory {
        Inventory::starting()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_game_starts_with_tools_and_seeds() {
        let inventory = Inventory::starting();
        assert_eq!(inventory.count_of(ItemId::Hoe), 1);
        assert_eq!(inventory.count_of(ItemId::WateringCan), 1);
        assert_eq!(inventory.count_of(ItemId::ParsnipSeeds), 15);
        assert!(inventory.gold > 0);
    }

    #[test]
    fn adding_fills_partial_stacks_before_opening_new_slots() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::Wood, 10);
        let used_before = inventory.used_slots();
        inventory.add(ItemId::Wood, 10);
        assert_eq!(
            inventory.used_slots(),
            used_before,
            "a second slot should not be opened"
        );
        assert_eq!(inventory.count_of(ItemId::Wood), 20);
    }

    #[test]
    fn adding_beyond_a_stack_opens_another_slot() {
        let mut inventory = Inventory::new(0);
        // Food stacks to 99.
        assert_eq!(inventory.add(ItemId::VegetableStew, 150), 0);
        assert_eq!(inventory.count_of(ItemId::VegetableStew), 150);
        assert_eq!(inventory.used_slots(), 2);
    }

    #[test]
    fn a_full_inventory_reports_the_overflow() {
        let mut inventory = Inventory::new(0);
        // Tools do not stack, so each takes a whole slot.
        for _ in 0..INVENTORY_SLOTS {
            inventory.add(ItemId::Hoe, 1);
        }
        assert!(inventory.is_full());
        assert_eq!(
            inventory.add(ItemId::Hoe, 3),
            3,
            "nothing should have fitted"
        );
    }

    #[test]
    fn room_is_reported_accurately() {
        let mut inventory = Inventory::new(0);
        assert!(inventory.has_room_for(ItemId::Wood, 100));
        for _ in 0..INVENTORY_SLOTS {
            inventory.add(ItemId::Hoe, 1);
        }
        assert!(!inventory.has_room_for(ItemId::Wood, 1));
    }

    #[test]
    fn removing_is_all_or_nothing() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::Wood, 5);
        assert!(!inventory.remove(ItemId::Wood, 10), "not enough to remove");
        assert_eq!(inventory.count_of(ItemId::Wood), 5, "and nothing was taken");
        assert!(inventory.remove(ItemId::Wood, 5));
        assert_eq!(inventory.count_of(ItemId::Wood), 0);
    }

    #[test]
    fn removing_drains_across_several_stacks() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::VegetableStew, 150);
        assert!(inventory.remove(ItemId::VegetableStew, 120));
        assert_eq!(inventory.count_of(ItemId::VegetableStew), 30);
        assert_eq!(inventory.used_slots(), 1, "the emptied slot was released");
    }

    #[test]
    fn selling_pays_the_catalogue_price() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::Parsnip, 4);
        let expected = item(ItemId::Parsnip).sell_price * 4;
        assert_eq!(inventory.sell(ItemId::Parsnip, 4), Some(expected));
        assert_eq!(inventory.gold, expected);
        assert_eq!(inventory.count_of(ItemId::Parsnip), 0);
    }

    #[test]
    fn selling_what_you_do_not_have_fails_without_paying() {
        let mut inventory = Inventory::new(100);
        assert_eq!(inventory.sell(ItemId::Parsnip, 1), None);
        assert_eq!(inventory.gold, 100);
    }

    #[test]
    fn tools_cannot_be_sold() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::Hoe, 1);
        assert_eq!(inventory.sell(ItemId::Hoe, 1), None);
        assert_eq!(inventory.count_of(ItemId::Hoe), 1, "and the tool is kept");
    }

    #[test]
    fn buying_deducts_gold_and_delivers_goods() {
        let mut inventory = Inventory::new(1000);
        let price = item(ItemId::ParsnipSeeds).buy_price.unwrap();
        assert_eq!(inventory.buy(ItemId::ParsnipSeeds, 5), Some(price * 5));
        assert_eq!(inventory.gold, 1000 - price * 5);
        assert_eq!(inventory.count_of(ItemId::ParsnipSeeds), 5);
    }

    #[test]
    fn buying_what_you_cannot_afford_fails() {
        let mut inventory = Inventory::new(5);
        assert_eq!(inventory.buy(ItemId::ParsnipSeeds, 100), None);
        assert_eq!(inventory.gold, 5, "and no gold changes hands");
        assert_eq!(inventory.count_of(ItemId::ParsnipSeeds), 0);
    }

    #[test]
    fn buying_something_unstocked_fails() {
        let mut inventory = Inventory::new(100_000);
        assert_eq!(
            inventory.buy(ItemId::Parsnip, 1),
            None,
            "produce is not stocked"
        );
    }

    #[test]
    fn buying_with_no_room_fails_before_taking_payment() {
        let mut inventory = Inventory::new(100_000);
        for _ in 0..INVENTORY_SLOTS {
            inventory.add(ItemId::Hoe, 1);
        }
        assert_eq!(inventory.buy(ItemId::ParsnipSeeds, 1), None);
        assert_eq!(inventory.gold, 100_000);
    }

    #[test]
    fn crafting_consumes_ingredients_and_yields_the_output() {
        let recipe = &RECIPES[0];
        let mut inventory = Inventory::new(0);
        for (id, count) in recipe.ingredients {
            inventory.add(*id, *count);
        }
        assert!(inventory.can_craft(recipe));
        assert!(inventory.craft(recipe));

        assert_eq!(inventory.count_of(recipe.output), recipe.output_count);
        for (id, _) in recipe.ingredients {
            assert_eq!(inventory.count_of(*id), 0, "ingredients should be consumed");
        }
    }

    #[test]
    fn crafting_without_ingredients_changes_nothing() {
        let recipe = &RECIPES[0];
        let mut inventory = Inventory::new(0);
        // One short of the first ingredient.
        let (first, count) = recipe.ingredients[0];
        inventory.add(first, count - 1);

        assert!(!inventory.can_craft(recipe));
        assert!(!inventory.craft(recipe));
        assert_eq!(inventory.count_of(first), count - 1, "nothing was consumed");
        assert_eq!(inventory.count_of(recipe.output), 0);
    }

    #[test]
    fn available_recipes_reflect_what_is_carried() {
        let mut inventory = Inventory::new(0);
        assert!(inventory.available_recipes().is_empty());

        let recipe = &RECIPES[0];
        for (id, count) in recipe.ingredients {
            inventory.add(*id, *count);
        }
        assert_eq!(inventory.available_recipes().len(), 1);
    }

    #[test]
    fn eating_food_restores_energy_and_consumes_it() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::VegetableStew, 2);
        let energy = inventory.eat_from_slot(0).expect("stew is edible");
        assert!(energy > 0);
        assert_eq!(inventory.count_of(ItemId::VegetableStew), 1);
    }

    #[test]
    fn inedible_items_cannot_be_eaten() {
        let mut inventory = Inventory::new(0);
        inventory.add(ItemId::Stone, 1);
        assert_eq!(inventory.eat_from_slot(0), None);
        assert_eq!(
            inventory.count_of(ItemId::Stone),
            1,
            "and the stone is kept"
        );
    }

    #[test]
    fn hotbar_selection_wraps_and_stays_in_range() {
        let mut inventory = Inventory::starting();
        assert_eq!(inventory.selected_index(), 0);

        inventory.cycle_selection(false);
        assert_eq!(
            inventory.selected_index(),
            HOTBAR_SLOTS - 1,
            "wraps backwards"
        );
        inventory.cycle_selection(true);
        assert_eq!(inventory.selected_index(), 0, "and forwards");

        inventory.select(3);
        assert_eq!(inventory.selected_index(), 3);
        inventory.select(999);
        assert_eq!(
            inventory.selected_index(),
            3,
            "an out-of-range index is ignored"
        );
    }

    #[test]
    fn the_selected_slot_reports_its_contents() {
        let mut inventory = Inventory::starting();
        assert_eq!(
            inventory.selected().map(|slot| slot.item),
            Some(ItemId::Hoe)
        );
        inventory.select(5);
        assert_eq!(inventory.selected(), None, "an empty slot selects nothing");
    }

    #[test]
    fn an_inventory_round_trips_through_serialisation() {
        let mut inventory = Inventory::starting();
        inventory.add(ItemId::Parsnip, 7);
        inventory.select(2);

        let encoded = serde_json::to_string(&inventory).expect("serialisable");
        let decoded: Inventory = serde_json::from_str(&encoded).expect("deserialisable");
        assert_eq!(decoded.count_of(ItemId::Parsnip), 7);
        assert_eq!(decoded.selected_index(), 2);
        assert_eq!(decoded.gold, inventory.gold);
    }
}
