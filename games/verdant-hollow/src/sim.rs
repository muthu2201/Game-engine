//! The player, the villagers, and the game state that ties everything together.

use crate::calendar::{Calendar, DayTransition};
use crate::farm::{Farm, FarmAction};
use crate::inventory::Inventory;
use crate::items::{item, ItemId, ItemKind};
use crate::world::{generate_valley, tiles, ValleyLayout, TILE_SIZE};
use serde::{Deserialize, Serialize};
use verdant_core_math::{Fx, IVec2, Rng, StateHasher, Vec2};
use verdant_procgen_art::Facing;
use verdant_tilemap::{find_path, Movement, TileLayer, Tilemap};

/// Walking speed in world units per second.
///
/// Four tiles a second: fast enough that crossing the valley is not a chore,
/// slow enough that the valley still feels like a place with distance in it.
pub const WALK_SPEED: Fx = Fx::from_num(64);

/// Multiplier applied while sprinting.
pub const SPRINT_MULTIPLIER: Fx = Fx::from_ratio(17, 10);

/// Energy the player wakes with.
pub const MAX_ENERGY: i32 = 270;

/// How far the player can reach to use a tool or interact, in tiles.
pub const REACH_TILES: i32 = 1;

/// The player's state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    /// World position of the player's feet.
    pub position: Vec2,
    /// Which way they are facing, which decides the tile a tool acts on.
    pub facing: FacingState,
    /// Remaining energy. Every tool swing costs some; at zero the player
    /// collapses and loses part of the next day.
    pub energy: i32,
    /// Carried items and money.
    pub inventory: Inventory,
    /// Total gold earned, for the end-of-game summary.
    pub lifetime_earnings: u32,
    /// How far down the mine the player has reached.
    pub deepest_mine_level: u32,
    /// Whether the player is currently underground.
    pub in_mine: bool,
    /// Which mine level, when underground.
    pub mine_level: u32,
    /// Seconds of the current walk cycle, for animation.
    animation_time: Fx,
    /// Whether the player moved on the last tick.
    moving: bool,
}

/// A serialisable facing, since [`Facing`] belongs to the art crate.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum FacingState {
    /// Toward the camera.
    South,
    /// Away from the camera.
    North,
    /// Left.
    West,
    /// Right.
    East,
}

impl FacingState {
    /// The art crate's equivalent.
    #[must_use]
    pub const fn to_art(self) -> Facing {
        match self {
            FacingState::South => Facing::South,
            FacingState::North => Facing::North,
            FacingState::West => Facing::West,
            FacingState::East => Facing::East,
        }
    }

    /// The tile offset this facing points at.
    #[must_use]
    pub const fn offset(self) -> IVec2 {
        match self {
            FacingState::South => IVec2::new(0, 1),
            FacingState::North => IVec2::new(0, -1),
            FacingState::West => IVec2::new(-1, 0),
            FacingState::East => IVec2::new(1, 0),
        }
    }

    /// The facing that best matches a movement direction.
    ///
    /// Vertical wins ties, because a character walking diagonally reads more
    /// naturally facing up or down than sideways.
    #[must_use]
    pub fn from_direction(direction: Vec2, current: FacingState) -> FacingState {
        if direction.is_zero() {
            return current;
        }
        if direction.y.abs() >= direction.x.abs() {
            if direction.y.is_negative() {
                FacingState::North
            } else {
                FacingState::South
            }
        } else if direction.x.is_negative() {
            FacingState::West
        } else {
            FacingState::East
        }
    }
}

impl Player {
    /// A player standing at `position` with a starting inventory.
    #[must_use]
    pub fn new(position: Vec2) -> Player {
        Player {
            position,
            facing: FacingState::South,
            energy: MAX_ENERGY,
            inventory: Inventory::starting(),
            lifetime_earnings: 0,
            deepest_mine_level: 0,
            in_mine: false,
            mine_level: 0,
            animation_time: Fx::ZERO,
            moving: false,
        }
    }

    /// The tile the player is standing on.
    #[must_use]
    pub fn tile(&self) -> IVec2 {
        let (x, y) = self.position.to_tile(TILE_SIZE);
        IVec2::new(x, y)
    }

    /// The tile a tool or interaction would act on.
    #[must_use]
    pub fn target_tile(&self) -> IVec2 {
        self.tile() + self.facing.offset() * REACH_TILES
    }

    /// The frame of the walk cycle to draw.
    ///
    /// Frames advance with distance walked rather than with time, so the feet
    /// stay planted: a character animating on a timer appears to skate when
    /// their speed changes.
    #[must_use]
    pub fn animation_frame(&self) -> usize {
        if !self.moving {
            // Frame 0 is the contact pose, which doubles as the idle pose.
            return 0;
        }
        const FRAMES_PER_SECOND: i32 = 8;
        ((self.animation_time * FRAMES_PER_SECOND).floor_int().rem_euclid(4)) as usize
    }

    /// True when the player is too tired to keep working.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.energy <= 0
    }

    /// Spends energy, never dropping below zero.
    pub fn spend_energy(&mut self, amount: i32) {
        self.energy = (self.energy - amount).max(0);
    }

    /// Restores energy, capped at the daily maximum.
    pub fn restore_energy(&mut self, amount: i32) {
        self.energy = (self.energy + amount).min(MAX_ENERGY);
    }
}

/// A villager.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Villager {
    /// Display name.
    pub name: String,
    /// Current world position.
    pub position: Vec2,
    /// Which way they are facing.
    pub facing: FacingState,
    /// Where they live.
    pub home: IVec2,
    /// Friendship, from zero to [`MAX_FRIENDSHIP`].
    pub friendship: i32,
    /// Whether they have been given something today. One gift a day, so
    /// friendship is earned over time rather than bought in an afternoon.
    pub gifted_today: bool,
    /// Whether they have been spoken to today.
    pub greeted_today: bool,
    /// What they most like to receive.
    pub favourite_gift: ItemId,
    /// The seed their appearance is generated from.
    pub appearance_seed: u64,
    /// The route they are currently walking.
    path: Vec<IVec2>,
    /// How far along that route they are.
    path_index: usize,
}

/// The highest friendship a villager will reach.
pub const MAX_FRIENDSHIP: i32 = 1000;

/// Friendship gained from a favourite gift.
pub const FAVOURITE_GIFT_POINTS: i32 = 80;
/// Friendship gained from any other gift.
pub const GIFT_POINTS: i32 = 20;
/// Friendship gained from the first conversation each day.
pub const GREETING_POINTS: i32 = 5;

/// Where a villager should be at a given hour.
///
/// A schedule is a list of `(hour, destination)` pairs rather than a script,
/// so a villager interrupted by terrain or a closed door simply re-paths to
/// wherever they should currently be instead of losing their place.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleEntry {
    /// The hour this destination takes effect.
    pub hour: u32,
    /// Where to go.
    pub destination: Destination,
}

/// A named place in a villager's schedule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Destination {
    /// Their own house.
    Home,
    /// The village square.
    Square,
    /// The shop.
    Shop,
}

/// The daily routine every villager follows.
///
/// One shared schedule rather than one per villager: with four villagers, the
/// variety comes from where their homes are and when they set out, not from
/// each having a bespoke itinerary.
pub static VILLAGE_SCHEDULE: &[ScheduleEntry] = &[
    ScheduleEntry { hour: 6, destination: Destination::Home },
    ScheduleEntry { hour: 9, destination: Destination::Square },
    ScheduleEntry { hour: 13, destination: Destination::Shop },
    ScheduleEntry { hour: 17, destination: Destination::Square },
    ScheduleEntry { hour: 20, destination: Destination::Home },
];

impl Villager {
    /// The destination this villager should currently be heading for.
    #[must_use]
    pub fn destination_at(&self, hour: u32, layout: &ValleyLayout) -> IVec2 {
        let mut current = Destination::Home;
        for entry in VILLAGE_SCHEDULE {
            if hour >= entry.hour {
                current = entry.destination;
            }
        }
        match current {
            Destination::Home => self.home,
            Destination::Square => layout.square,
            Destination::Shop => layout.shop_door,
        }
    }

    /// How the villager greets the player, based on friendship.
    #[must_use]
    pub fn greeting(&self) -> &'static str {
        match self.friendship {
            f if f >= 800 => "You're the best thing to happen to this valley.",
            f if f >= 500 => "Always good to see you! Stop by any time.",
            f if f >= 250 => "Morning! The farm's looking well.",
            f if f >= 80 => "Oh — hello again.",
            _ => "...hello.",
        }
    }

    /// Records a conversation, returning the friendship gained.
    pub fn greet(&mut self) -> i32 {
        if self.greeted_today {
            return 0;
        }
        self.greeted_today = true;
        self.friendship = (self.friendship + GREETING_POINTS).min(MAX_FRIENDSHIP);
        GREETING_POINTS
    }

    /// Accepts a gift, returning the friendship gained.
    ///
    /// Returns `None` when they have already been given something today, so
    /// friendship cannot be bought in a single afternoon.
    pub fn receive_gift(&mut self, gift: ItemId) -> Option<i32> {
        if self.gifted_today {
            return None;
        }
        self.gifted_today = true;
        let points =
            if gift == self.favourite_gift { FAVOURITE_GIFT_POINTS } else { GIFT_POINTS };
        self.friendship = (self.friendship + points).min(MAX_FRIENDSHIP);
        Some(points)
    }
}

/// What the player's action did, for the HUD to report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionOutcome {
    /// Nothing applied here.
    Nothing,
    /// Something happened on the farm.
    Farm(FarmAction),
    /// A tree or rock was cleared, yielding an item.
    Gathered {
        /// What was collected.
        item: ItemId,
        /// How many.
        count: u32,
    },
    /// The player spoke to a villager.
    Talked {
        /// Who.
        name: String,
        /// What they said.
        line: &'static str,
    },
    /// A gift was given.
    Gifted {
        /// Who received it.
        name: String,
        /// Friendship gained.
        points: i32,
    },
    /// Items were sold into the shipping bin.
    Shipped {
        /// Gold earned.
        gold: u32,
    },
    /// The player was too tired to act.
    TooTired,
}

/// The whole game state.
pub struct Game {
    /// The world seed, from which everything else is derived.
    pub seed: u64,
    /// The valley's tiles.
    pub map: Tilemap,
    /// Where things are.
    pub layout: ValleyLayout,
    /// The clock.
    pub calendar: Calendar,
    /// Worked farmland.
    pub farm: Farm,
    /// The player.
    pub player: Player,
    /// The villagers.
    pub villagers: Vec<Villager>,
    /// Gold earned from yesterday's shipment, shown on waking.
    pub last_shipment: u32,
    /// The simulation's random generator. Part of saved state, so a reload
    /// continues the same sequence rather than restarting it.
    pub rng: Rng,
    /// The most recent action's outcome, for the HUD.
    pub last_outcome: ActionOutcome,
}

impl Game {
    /// Starts a new game.
    #[must_use]
    pub fn new(seed: u64) -> Game {
        let valley = generate_valley(seed);
        let mut rng = Rng::new(seed).derive("simulation");

        // Villagers are generated from the world seed, so a given valley is
        // always populated by the same people.
        let names = ["Mara", "Tobin", "Silva", "Rook"];
        let favourites =
            [ItemId::Parsnip, ItemId::Cauliflower, ItemId::Chub, ItemId::Gemstone];
        let villagers = valley
            .layout
            .homes
            .iter()
            .enumerate()
            .map(|(index, home)| Villager {
                name: names.get(index).copied().unwrap_or("Villager").to_string(),
                position: home.to_world_centre(TILE_SIZE),
                facing: FacingState::South,
                home: *home,
                friendship: 0,
                gifted_today: false,
                greeted_today: false,
                favourite_gift: favourites
                    .get(index)
                    .copied()
                    .unwrap_or(ItemId::Parsnip),
                appearance_seed: seed
                    .wrapping_mul(31)
                    .wrapping_add(index as u64)
                    .wrapping_add(1),
                path: Vec::new(),
                path_index: 0,
            })
            .collect();

        let player = Player::new(valley.layout.farmhouse_door.to_world_centre(TILE_SIZE));

        Game {
            seed,
            map: valley.map,
            layout: valley.layout,
            calendar: Calendar::new(&mut rng),
            farm: Farm::new(),
            player,
            villagers,
            last_shipment: 0,
            rng,
            last_outcome: ActionOutcome::Nothing,
        }
    }

    /// Advances the simulation by one step.
    ///
    /// `movement` is the player's desired direction; the collision solver
    /// decides how much of it actually happens.
    pub fn update(&mut self, movement: Vec2, sprinting: bool, dt: Fx) {
        self.move_player(movement, sprinting, dt);
        self.move_villagers(dt);

        if self.calendar.advance(dt, &mut self.rng) == DayTransition::Collapsed {
            // Collapsing costs half the next day's energy.
            self.begin_day(MAX_ENERGY / 2);
        }
    }

    /// Moves the player, stopping at solid tiles.
    fn move_player(&mut self, movement: Vec2, sprinting: bool, dt: Fx) {
        let speed = if sprinting && !self.player.is_exhausted() {
            WALK_SPEED * SPRINT_MULTIPLIER
        } else {
            WALK_SPEED
        };
        let displacement = movement * speed * dt;
        self.player.moving = !displacement.is_zero();

        if self.player.moving {
            self.player.facing = FacingState::from_direction(movement, self.player.facing);
            self.player.animation_time += dt;

            // A collider narrower than a tile, so a one-tile gap is passable.
            let collider = verdant_physics_2d::Collider::feet_anchored(
                TILE_SIZE * Fx::from_ratio(5, 8),
                TILE_SIZE * Fx::from_ratio(3, 8),
            );
            let result = verdant_physics_2d::move_and_slide(
                &collider,
                self.player.position,
                displacement,
                &self.map,
                &[],
            );
            self.player.position = result.position;
        }
    }

    /// Walks each villager toward wherever their schedule says they should be.
    fn move_villagers(&mut self, dt: Fx) {
        let (hour, _) = self.calendar.clock();
        // Villagers move at two thirds of the player's pace, so the player can
        // always catch someone they want to talk to.
        let step = WALK_SPEED * Fx::from_ratio(2, 3) * dt;

        for index in 0..self.villagers.len() {
            let destination = self.villagers[index].destination_at(hour, &self.layout);
            let current_tile = {
                let (x, y) = self.villagers[index].position.to_tile(TILE_SIZE);
                IVec2::new(x, y)
            };

            // Re-path when the route is spent or was computed for somewhere
            // else. Re-deriving from the schedule means an interrupted
            // villager recovers instead of standing still.
            let needs_path = self.villagers[index].path.is_empty()
                || self.villagers[index].path_index >= self.villagers[index].path.len()
                || self.villagers[index].path.last() != Some(&destination);
            if needs_path {
                let route =
                    find_path(&self.map, current_tile, destination, Movement::Orthogonal, 8192);
                self.villagers[index].path = route.unwrap_or_default();
                self.villagers[index].path_index = 0;
            }

            let villager = &mut self.villagers[index];
            let Some(next) = villager.path.get(villager.path_index).copied() else {
                continue;
            };
            let target = next.to_world_centre(TILE_SIZE);
            let offset = target - villager.position;

            if offset.length() <= step {
                villager.position = target;
                villager.path_index += 1;
            } else {
                villager.position += offset.normalize() * step;
                villager.facing = FacingState::from_direction(offset, villager.facing);
            }
        }
    }

    /// Uses the selected item on the tile the player faces.
    pub fn use_selected(&mut self) -> ActionOutcome {
        if self.player.is_exhausted() {
            self.last_outcome = ActionOutcome::TooTired;
            return self.last_outcome.clone();
        }
        let Some(slot) = self.player.inventory.selected() else {
            self.last_outcome = ActionOutcome::Nothing;
            return self.last_outcome.clone();
        };
        let target = self.player.target_tile();
        let outcome = match item(slot.item).kind {
            ItemKind::Tool => self.use_tool(slot.item, target),
            ItemKind::Seed { crop } => {
                let action = self.farm.plant(target, crop, self.calendar.season());
                if action != FarmAction::Nothing {
                    self.player.inventory.remove(slot.item, 1);
                }
                ActionOutcome::Farm(action)
            }
            ItemKind::Food { .. } => {
                let index = self.player.inventory.selected_index();
                match self.player.inventory.eat_from_slot(index) {
                    Some(energy) => {
                        self.player.restore_energy(energy);
                        ActionOutcome::Nothing
                    }
                    None => ActionOutcome::Nothing,
                }
            }
            ItemKind::Produce | ItemKind::Material => ActionOutcome::Nothing,
        };
        self.last_outcome = outcome;
        self.last_outcome.clone()
    }

    /// Applies a tool to a tile.
    fn use_tool(&mut self, tool: ItemId, target: IVec2) -> ActionOutcome {
        // Energy costs are per swing, so a day's work is bounded by stamina
        // rather than by the clock alone.
        let cost = match tool {
            ItemId::Hoe | ItemId::WateringCan => 2,
            ItemId::Axe | ItemId::Pickaxe => 4,
            _ => 1,
        };

        let ground_tile = self
            .map
            .layer("ground")
            .map_or(verdant_tilemap::TileId::EMPTY, |layer| layer.get(target));

        let outcome = match tool {
            ItemId::Hoe => {
                // Only bare ground can be broken, and only inside the farm.
                if self.is_farmable(target) && ground_tile == tiles::GRASS {
                    ActionOutcome::Farm(self.farm.till(target))
                } else if self.farm.plot(target).is_some_and(|plot| plot.planting.is_none()) {
                    ActionOutcome::Farm(self.farm.clear(target))
                } else {
                    ActionOutcome::Nothing
                }
            }
            ItemId::WateringCan => ActionOutcome::Farm(self.farm.water(target)),
            ItemId::Axe if ground_tile == tiles::TREE => {
                self.replace_tile(target, tiles::GRASS);
                let count = 2 + self.rng.below(3);
                self.player.inventory.add(ItemId::Wood, count);
                ActionOutcome::Gathered { item: ItemId::Wood, count }
            }
            ItemId::Pickaxe if ground_tile == tiles::ROCK => {
                self.replace_tile(target, tiles::GRASS);
                let count = 1 + self.rng.below(2);
                self.player.inventory.add(ItemId::Stone, count);
                ActionOutcome::Gathered { item: ItemId::Stone, count }
            }
            _ => ActionOutcome::Nothing,
        };

        if outcome != ActionOutcome::Nothing {
            self.player.spend_energy(cost);
        }
        outcome
    }

    /// True when a tile is inside the farm plot.
    fn is_farmable(&self, cell: IVec2) -> bool {
        let (min, max) = self.layout.farm_area;
        cell.x >= min.x && cell.x <= max.x && cell.y >= min.y && cell.y <= max.y
    }

    /// Rewrites a ground tile.
    fn replace_tile(&mut self, cell: IVec2, tile: verdant_tilemap::TileId) {
        if let Some(layer) = self.map.layer_mut("ground") {
            layer.set(cell, tile);
        }
    }

    /// Interacts with whatever the player is facing.
    ///
    /// Harvesting, talking and shipping are all one button, resolved by what is
    /// actually there — which is far better than asking the player to remember
    /// three different keys.
    pub fn interact(&mut self) -> ActionOutcome {
        let target = self.player.target_tile();

        // A villager standing close enough to talk to.
        let reach = TILE_SIZE * Fx::from_ratio(3, 2);
        let nearby = self
            .villagers
            .iter()
            .position(|villager| villager.position.distance(self.player.position) <= reach);
        if let Some(index) = nearby {
            let held = self.player.inventory.selected();
            // Holding produce offers it as a gift; otherwise, just talk.
            if let Some(slot) = held {
                if matches!(item(slot.item).kind, ItemKind::Produce) {
                    if let Some(points) = self.villagers[index].receive_gift(slot.item) {
                        self.player.inventory.remove(slot.item, 1);
                        self.last_outcome = ActionOutcome::Gifted {
                            name: self.villagers[index].name.clone(),
                            points,
                        };
                        return self.last_outcome.clone();
                    }
                }
            }
            self.villagers[index].greet();
            self.last_outcome = ActionOutcome::Talked {
                name: self.villagers[index].name.clone(),
                line: self.villagers[index].greeting(),
            };
            return self.last_outcome.clone();
        }

        // The farmhouse door doubles as the shipping bin.
        if target == self.layout.farmhouse_door || self.player.tile() == self.layout.farmhouse_door
        {
            let gold = self.ship_everything();
            self.last_outcome = ActionOutcome::Shipped { gold };
            return self.last_outcome.clone();
        }

        // Otherwise, try to harvest.
        let action = self.farm.harvest(target, &mut self.rng);
        if let FarmAction::Harvested { item: produce, count } = action {
            self.player.inventory.add(produce, count);
        }
        self.last_outcome = ActionOutcome::Farm(action);
        self.last_outcome.clone()
    }

    /// Sells every sellable item the player is carrying.
    fn ship_everything(&mut self) -> u32 {
        let sellable: Vec<(ItemId, u32)> = self
            .player
            .inventory
            .slots()
            .iter()
            .flatten()
            .filter(|slot| item(slot.item).sell_price > 0)
            .map(|slot| (slot.item, slot.count))
            .collect();

        let mut total = 0;
        for (id, count) in sellable {
            if let Some(gold) = self.player.inventory.sell(id, count) {
                total += gold;
            }
        }
        self.player.lifetime_earnings = self.player.lifetime_earnings.saturating_add(total);
        self.last_shipment = total;
        total
    }

    /// Ends the day and starts the next one.
    pub fn sleep(&mut self) {
        self.calendar.sleep(&mut self.rng);
        self.begin_day(MAX_ENERGY);
    }

    /// Runs everything that happens between one day and the next.
    fn begin_day(&mut self, energy: i32) {
        let season = self.calendar.season();
        let weather = self.calendar.weather;
        self.farm.advance_day(season, weather, &mut self.rng);

        self.player.energy = energy;
        self.player.position = self.layout.farmhouse_door.to_world_centre(TILE_SIZE);
        self.player.in_mine = false;

        for villager in &mut self.villagers {
            villager.gifted_today = false;
            villager.greeted_today = false;
            // Villagers wake at home, so a day always starts from a known
            // arrangement rather than wherever pathing left them.
            villager.position = villager.home.to_world_centre(TILE_SIZE);
            villager.path.clear();
            villager.path_index = 0;
        }
    }

    /// Regrows a little of the valley's scenery overnight.
    ///
    /// Without this the valley is a strictly diminishing resource, and a player
    /// who clears it has nothing to gather for the rest of the game.
    pub fn regrow_scenery(&mut self) {
        let Some(layer) = self.map.layer_mut("ground") else {
            return;
        };
        let width = layer.width() as i32;
        let height = layer.height() as i32;
        for _ in 0..6 {
            let cell = IVec2::new(self.rng.range(1, width - 2), self.rng.range(1, height - 2));
            if layer.get(cell) == tiles::GRASS {
                let tile = if self.rng.coin_flip() { tiles::TREE } else { tiles::ROCK };
                layer.set(cell, tile);
            }
        }
    }

    /// Fingerprints the whole simulation, for the determinism tests.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_u64(self.seed);
        hasher.write_u32(self.calendar.day);
        hasher.write_u32(self.calendar.minute);
        hasher.write_u64(self.farm.state_hash());
        hasher.write_vec2(self.player.position);
        hasher.write_u32(self.player.energy.unsigned_abs());
        hasher.write_u32(self.player.inventory.gold);
        for villager in &self.villagers {
            hasher.write_vec2(villager.position);
            hasher.write_u32(villager.friendship.unsigned_abs());
        }
        let (state, increment) = self.rng.state();
        hasher.write_u64(state);
        hasher.write_u64(increment);
        hasher.finish()
    }

    /// The ground layer, for rendering.
    #[must_use]
    pub fn ground(&self) -> Option<&TileLayer> {
        self.map.layer("ground")
    }
}

/// A saved game.
///
/// Separate from [`Game`] because the map is regenerated from the seed rather
/// than stored: the valley is a pure function of its seed, so persisting three
/// thousand tiles would be storing something already derivable. Only the tiles
/// the player has *changed* need recording.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveData {
    /// The world seed.
    pub seed: u64,
    /// The clock.
    pub calendar: Calendar,
    /// Worked farmland.
    pub farm: Farm,
    /// The player.
    pub player: Player,
    /// The villagers.
    pub villagers: Vec<Villager>,
    /// Ground tiles the player has changed, as `(x, y, tile)`.
    pub tile_edits: Vec<(i32, i32, u16)>,
    /// The generator's state, so randomness continues rather than restarting.
    pub rng_state: (u64, u64),
}

/// The current save format version.
pub const SAVE_VERSION: u32 = 1;

impl Game {
    /// Captures the game into a saveable form.
    #[must_use]
    pub fn save_data(&self) -> SaveData {
        // Diff the live map against a freshly generated one, so only genuine
        // player edits are stored.
        let pristine = generate_valley(self.seed);
        let mut tile_edits = Vec::new();
        if let (Some(current), Some(original)) =
            (self.map.layer("ground"), pristine.map.layer("ground"))
        {
            for (cell, tile) in current.iter() {
                if original.get(cell) != tile {
                    tile_edits.push((cell.x, cell.y, tile.0));
                }
            }
        }

        SaveData {
            seed: self.seed,
            calendar: self.calendar.clone(),
            farm: self.farm.clone(),
            player: self.player.clone(),
            villagers: self.villagers.clone(),
            tile_edits,
            rng_state: self.rng.state(),
        }
    }

    /// Rebuilds a game from saved data.
    #[must_use]
    pub fn from_save(data: SaveData) -> Game {
        let mut game = Game::new(data.seed);
        game.calendar = data.calendar;
        game.farm = data.farm;
        game.player = data.player;
        game.villagers = data.villagers;
        game.rng.restore(data.rng_state);

        if let Some(layer) = game.map.layer_mut("ground") {
            for (x, y, tile) in data.tile_edits {
                layer.set(IVec2::new(x, y), verdant_tilemap::TileId(tile));
            }
        }
        game
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::CropId;

    /// One simulation step at the standard rate.
    fn step() -> Fx {
        Fx::ONE / 60
    }

    fn game() -> Game {
        Game::new(7)
    }

    #[test]
    fn a_new_game_starts_the_player_at_the_farmhouse() {
        let game = game();
        assert_eq!(game.player.tile(), game.layout.farmhouse_door);
        assert_eq!(game.player.energy, MAX_ENERGY);
        assert_eq!(game.villagers.len(), game.layout.homes.len());
    }

    #[test]
    fn facing_follows_the_direction_of_travel() {
        let mut game = game();
        game.update(Vec2::RIGHT, false, step());
        assert_eq!(game.player.facing, FacingState::East);
        game.update(Vec2::UP, false, step());
        assert_eq!(game.player.facing, FacingState::North);
    }

    #[test]
    fn the_target_tile_is_the_one_being_faced() {
        let mut game = game();
        game.player.facing = FacingState::East;
        assert_eq!(game.player.target_tile(), game.player.tile() + IVec2::new(1, 0));
    }

    #[test]
    fn the_player_cannot_walk_through_buildings() {
        let mut game = game();
        // The farmhouse sits directly north of its door.
        let start = game.player.position;
        for _ in 0..120 {
            game.update(Vec2::UP, false, step());
        }
        assert!(
            game.player.position.y > start.y - TILE_SIZE * 4,
            "the player walked into the farmhouse"
        );
    }

    #[test]
    fn sprinting_covers_more_ground() {
        let distance = |sprinting: bool| {
            let mut game = game();
            let start = game.player.position;
            for _ in 0..30 {
                game.update(Vec2::DOWN, sprinting, step());
            }
            game.player.position.distance(start)
        };
        assert!(distance(true) > distance(false));
    }

    #[test]
    fn an_exhausted_player_cannot_sprint() {
        let mut exhausted = game();
        let mut walker = game();

        exhausted.player.energy = 0;
        let start = exhausted.player.position;
        for _ in 0..30 {
            exhausted.update(Vec2::DOWN, true, step());
        }
        let sprinted = exhausted.player.position.distance(start);

        let start = walker.player.position;
        for _ in 0..30 {
            walker.update(Vec2::DOWN, false, step());
        }
        assert_eq!(sprinted, walker.player.position.distance(start));
    }

    #[test]
    fn the_hoe_tills_farmland_and_costs_energy() {
        let mut game = game();
        // Stand in the farm plot facing an untilled tile.
        let plot = game.layout.farm_area.0 + IVec2::new(2, 2);
        game.player.position = (plot - IVec2::new(0, 1)).to_world_centre(TILE_SIZE);
        game.player.facing = FacingState::South;
        game.player.inventory.select(0);

        let energy_before = game.player.energy;
        assert_eq!(game.use_selected(), ActionOutcome::Farm(FarmAction::Tilled));
        assert!(game.farm.plot(plot).is_some());
        assert!(game.player.energy < energy_before, "tilling should cost energy");
    }

    #[test]
    fn the_hoe_does_nothing_outside_the_farm() {
        let mut game = game();
        // Far from the farm plot.
        let outside = IVec2::new(game.layout.square.x, game.layout.square.y + 2);
        game.player.position = outside.to_world_centre(TILE_SIZE);
        game.player.facing = FacingState::South;
        game.player.inventory.select(0);

        assert_eq!(game.use_selected(), ActionOutcome::Nothing);
    }

    #[test]
    fn a_tired_player_cannot_work() {
        let mut game = game();
        game.player.energy = 0;
        assert_eq!(game.use_selected(), ActionOutcome::TooTired);
    }

    #[test]
    fn seeds_plant_and_are_consumed() {
        let mut game = game();
        let plot = game.layout.farm_area.0 + IVec2::new(3, 3);
        game.farm.till(plot);
        game.player.position = (plot - IVec2::new(0, 1)).to_world_centre(TILE_SIZE);
        game.player.facing = FacingState::South;

        // Select the seed slot.
        let seeds_before = game.player.inventory.count_of(ItemId::ParsnipSeeds);
        game.player.inventory.select(4);
        assert_eq!(
            game.use_selected(),
            ActionOutcome::Farm(FarmAction::Planted(CropId::Parsnip))
        );
        assert_eq!(game.player.inventory.count_of(ItemId::ParsnipSeeds), seeds_before - 1);
    }

    #[test]
    fn harvesting_puts_produce_in_the_inventory() {
        let mut game = game();
        let plot = game.layout.farm_area.0 + IVec2::new(4, 4);
        game.farm.till(plot);
        game.farm.plant(plot, CropId::Parsnip, Season::Spring);
        for _ in 0..crate::items::crop(CropId::Parsnip).days_to_maturity() {
            game.farm.water(plot);
            game.farm.advance_day(Season::Spring, game.calendar.weather, &mut game.rng);
        }

        game.player.position = (plot - IVec2::new(0, 1)).to_world_centre(TILE_SIZE);
        game.player.facing = FacingState::South;
        let before = game.player.inventory.count_of(ItemId::Parsnip);
        game.interact();
        assert!(game.player.inventory.count_of(ItemId::Parsnip) > before);
    }

    #[test]
    fn shipping_sells_everything_sellable_and_keeps_tools() {
        let mut game = game();
        game.player.inventory.add(ItemId::Parsnip, 5);
        game.player.position = game.layout.farmhouse_door.to_world_centre(TILE_SIZE);

        let gold_before = game.player.inventory.gold;
        match game.interact() {
            ActionOutcome::Shipped { gold } => assert!(gold > 0),
            other => panic!("expected a shipment, got {other:?}"),
        }
        assert!(game.player.inventory.gold > gold_before);
        assert_eq!(game.player.inventory.count_of(ItemId::Parsnip), 0);
        assert_eq!(game.player.inventory.count_of(ItemId::Hoe), 1, "tools are not shipped");
    }

    #[test]
    fn talking_to_a_villager_builds_friendship_once_a_day() {
        let mut game = game();
        let home = game.villagers[0].home;
        game.player.position = home.to_world_centre(TILE_SIZE);
        game.villagers[0].position = home.to_world_centre(TILE_SIZE);

        assert!(matches!(game.interact(), ActionOutcome::Talked { .. }));
        let after_first = game.villagers[0].friendship;
        assert_eq!(after_first, GREETING_POINTS);

        game.interact();
        assert_eq!(game.villagers[0].friendship, after_first, "one greeting a day");
    }

    #[test]
    fn gifting_a_favourite_is_worth_more() {
        let mut villager = Villager {
            name: "Test".into(),
            position: Vec2::ZERO,
            facing: FacingState::South,
            home: IVec2::ZERO,
            friendship: 0,
            gifted_today: false,
            greeted_today: false,
            favourite_gift: ItemId::Parsnip,
            appearance_seed: 1,
            path: Vec::new(),
            path_index: 0,
        };
        assert_eq!(villager.receive_gift(ItemId::Parsnip), Some(FAVOURITE_GIFT_POINTS));

        villager.gifted_today = false;
        villager.friendship = 0;
        assert_eq!(villager.receive_gift(ItemId::Stone), Some(GIFT_POINTS));
    }

    #[test]
    fn only_one_gift_a_day_is_accepted() {
        let mut game = game();
        game.player.inventory.add(ItemId::Parsnip, 5);
        // Select the parsnip slot.
        let index = game
            .player
            .inventory
            .slots()
            .iter()
            .position(|slot| slot.is_some_and(|slot| slot.item == ItemId::Parsnip))
            .expect("parsnips are carried");
        game.player.inventory.select(index.min(crate::inventory::HOTBAR_SLOTS - 1));

        let home = game.villagers[0].home;
        game.player.position = home.to_world_centre(TILE_SIZE);
        game.villagers[0].position = home.to_world_centre(TILE_SIZE);

        if index < crate::inventory::HOTBAR_SLOTS {
            assert!(matches!(game.interact(), ActionOutcome::Gifted { .. }));
            // A second attempt falls through to conversation.
            assert!(matches!(game.interact(), ActionOutcome::Talked { .. }));
        }
    }

    #[test]
    fn greetings_warm_up_as_friendship_grows() {
        let mut villager = Villager {
            name: "Test".into(),
            position: Vec2::ZERO,
            facing: FacingState::South,
            home: IVec2::ZERO,
            friendship: 0,
            gifted_today: false,
            greeted_today: false,
            favourite_gift: ItemId::Parsnip,
            appearance_seed: 1,
            path: Vec::new(),
            path_index: 0,
        };
        let cold = villager.greeting();
        villager.friendship = MAX_FRIENDSHIP;
        assert_ne!(cold, villager.greeting());
    }

    #[test]
    fn villagers_follow_their_schedule_through_the_day() {
        let game = game();
        let villager = &game.villagers[0];
        assert_eq!(villager.destination_at(7, &game.layout), villager.home);
        assert_eq!(villager.destination_at(10, &game.layout), game.layout.square);
        assert_eq!(villager.destination_at(14, &game.layout), game.layout.shop_door);
        assert_eq!(villager.destination_at(22, &game.layout), villager.home);
    }

    #[test]
    fn villagers_actually_walk_toward_their_destination() {
        let mut game = game();
        // Mid-morning, so everyone heads for the square.
        game.calendar.minute = 10 * 60;
        let start = game.villagers[0].position.distance(game.layout.square.to_world_centre(TILE_SIZE));

        for _ in 0..600 {
            game.update(Vec2::ZERO, false, step());
        }
        let finish =
            game.villagers[0].position.distance(game.layout.square.to_world_centre(TILE_SIZE));
        assert!(finish < start, "the villager did not set off: {start:?} -> {finish:?}");
    }

    #[test]
    fn sleeping_restores_energy_and_advances_the_day() {
        let mut game = game();
        game.player.energy = 10;
        let day = game.calendar.day;

        game.sleep();
        assert_eq!(game.calendar.day, day + 1);
        assert_eq!(game.player.energy, MAX_ENERGY);
        assert_eq!(game.player.tile(), game.layout.farmhouse_door, "and wakes at home");
    }

    #[test]
    fn crops_grow_across_a_night() {
        let mut game = game();
        let plot = game.layout.farm_area.0 + IVec2::new(5, 5);
        game.farm.till(plot);
        game.farm.plant(plot, CropId::Parsnip, Season::Spring);
        game.farm.water(plot);

        game.sleep();
        assert_eq!(game.farm.plot(plot).unwrap().planting.unwrap().growth_days, 1);
    }

    #[test]
    fn gifts_reset_each_morning() {
        let mut game = game();
        game.villagers[0].gifted_today = true;
        game.villagers[0].greeted_today = true;
        game.sleep();
        assert!(!game.villagers[0].gifted_today);
        assert!(!game.villagers[0].greeted_today);
    }

    #[test]
    fn chopping_a_tree_yields_wood_and_clears_the_tile() {
        let mut game = game();
        // Find a tree next to walkable ground.
        let tree = game
            .ground()
            .expect("the ground layer exists")
            .iter()
            .find(|(cell, tile)| {
                *tile == tiles::TREE && game.map.is_walkable(*cell + IVec2::new(0, -1))
            })
            .map(|(cell, _)| cell)
            .expect("the valley has trees");

        game.player.position = (tree + IVec2::new(0, -1)).to_world_centre(TILE_SIZE);
        game.player.facing = FacingState::South;
        // Select the axe.
        game.player.inventory.select(2);

        match game.use_selected() {
            ActionOutcome::Gathered { item: ItemId::Wood, count } => assert!(count >= 2),
            other => panic!("expected wood, got {other:?}"),
        }
        assert!(game.map.is_walkable(tree), "the tile should now be clear");
    }

    #[test]
    fn a_save_round_trips_without_losing_state() {
        let mut game = game();
        // Make the world differ from a fresh generation in several ways.
        let plot = game.layout.farm_area.0 + IVec2::new(6, 6);
        game.farm.till(plot);
        game.farm.plant(plot, CropId::Parsnip, Season::Spring);
        game.player.inventory.add(ItemId::Parsnip, 12);
        game.player.energy = 111;
        game.villagers[0].friendship = 350;
        game.replace_tile(IVec2::new(30, 40), tiles::PATH);
        game.sleep();

        let data = game.save_data();
        let encoded = serde_json::to_string(&data).expect("serialisable");
        let decoded: SaveData = serde_json::from_str(&encoded).expect("deserialisable");
        let restored = Game::from_save(decoded);

        assert_eq!(restored.state_hash(), game.state_hash());
        assert_eq!(restored.player.inventory.count_of(ItemId::Parsnip), 12);
        assert_eq!(restored.villagers[0].friendship, 350);
        assert_eq!(
            restored.ground().unwrap().get(IVec2::new(30, 40)),
            tiles::PATH,
            "player edits to the map must survive"
        );
    }

    #[test]
    fn a_save_stores_only_what_the_player_changed() {
        // The valley is a pure function of its seed, so a save should not
        // contain three thousand unchanged tiles.
        let game = game();
        assert!(
            game.save_data().tile_edits.is_empty(),
            "an untouched world should record no tile edits"
        );
    }

    #[test]
    fn determinism_a_simulated_week_reproduces_exactly() {
        // The property the whole engine exists to provide, exercised through
        // the real game: identical inputs must produce an identical world.
        let run = || {
            let mut game = Game::new(20_260_808);
            for day in 0..7 {
                for tick in 0..900 {
                    let direction = match tick % 4 {
                        0 => Vec2::RIGHT,
                        1 => Vec2::DOWN,
                        2 => Vec2::LEFT,
                        _ => Vec2::UP,
                    };
                    game.update(direction, tick % 8 == 0, step());
                    if tick % 37 == 0 {
                        game.use_selected();
                    }
                    if tick % 53 == 0 {
                        game.interact();
                    }
                }
                let _ = day;
                game.sleep();
            }
            game.state_hash()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_full_year_can_be_simulated_without_panicking() {
        // Long-running stability: four seasons of days, with the calendar
        // rolling over, seasons turning under crops, and villagers pathing.
        let mut game = Game::new(4242);
        for _ in 0..(28 * 4) {
            for _ in 0..200 {
                game.update(Vec2::RIGHT, false, step());
            }
            game.sleep();
        }
        assert_eq!(game.calendar.year(), 2);
        assert_eq!(game.calendar.season(), Season::Spring);
    }
}
