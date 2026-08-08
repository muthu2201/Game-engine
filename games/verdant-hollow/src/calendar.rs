//! Time, seasons and weather.
//!
//! The calendar is the spine a life sim hangs everything from: crops grow
//! against it, shops open and close against it, festivals land on it, and the
//! player's day ends when it runs out. It is deliberately a plain integer
//! counter rather than wall-clock time, so a save records "day 47, minute 630"
//! and reloads exactly there.

use serde::{Deserialize, Serialize};
use verdant_core_math::{Fx, Rng};
use verdant_render_2d::Color;

/// Minutes of simulated time in one in-game day.
///
/// The day runs 06:00 to 02:00 — twenty hours — because the small hours are
/// dead time in a farming game and forcing the player through them is not
/// interesting. Collapsing them is what every game in the genre does.
pub const MINUTES_PER_DAY: u32 = 20 * 60;

/// The in-game minute the day starts at (06:00).
pub const DAY_START_MINUTE: u32 = 6 * 60;

/// The minute the player collapses if still awake (02:00, as minute 1560).
pub const COLLAPSE_MINUTE: u32 = DAY_START_MINUTE + MINUTES_PER_DAY;

/// Real seconds per in-game minute.
///
/// At 0.7 seconds a minute, a full day is about fourteen real minutes: long
/// enough that a day's work feels like a session, short enough that a season
/// passes in an evening.
pub const SECONDS_PER_GAME_MINUTE: Fx = Fx::from_ratio(7, 10);

/// Days in one season.
pub const DAYS_PER_SEASON: u32 = 28;

/// The four seasons.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Season {
    /// Growing season; most crops are planted here.
    Spring,
    /// Long days, the most productive season.
    Summer,
    /// Harvest, and the last chance to plant.
    Autumn,
    /// Nothing grows outdoors; the season for mining and socialising.
    Winter,
}

impl Season {
    /// Days in one season, mirrored from [`DAYS_PER_SEASON`] so economy code
    /// can reason about a season's length without importing the constant
    /// separately.
    pub const DAYS: u32 = DAYS_PER_SEASON;

    /// All four, in calendar order.
    pub const ALL: [Season; 4] = [
        Season::Spring,
        Season::Summer,
        Season::Autumn,
        Season::Winter,
    ];

    /// The season's index in the year.
    #[must_use]
    pub const fn index(self) -> u32 {
        self as u32
    }

    /// The season's display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Season::Spring => "Spring",
            Season::Summer => "Summer",
            Season::Autumn => "Autumn",
            Season::Winter => "Winter",
        }
    }

    /// The season after this one.
    #[must_use]
    pub const fn next(self) -> Season {
        match self {
            Season::Spring => Season::Summer,
            Season::Summer => Season::Autumn,
            Season::Autumn => Season::Winter,
            Season::Winter => Season::Spring,
        }
    }

    /// The tint applied to the whole scene in this season.
    ///
    /// A colour shift is a cheap and very effective way to make a season felt:
    /// the same tiles read as spring or autumn purely from the grade.
    #[must_use]
    pub fn grade(self) -> Color {
        match self {
            Season::Spring => Color::rgb(1.0, 1.0, 0.96),
            Season::Summer => Color::rgb(1.05, 1.0, 0.88),
            Season::Autumn => Color::rgb(1.05, 0.92, 0.78),
            Season::Winter => Color::rgb(0.88, 0.94, 1.08),
        }
    }
}

/// What the sky is doing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Weather {
    /// Clear skies.
    Clear,
    /// Overcast, but dry.
    Cloudy,
    /// Rain. Waters every crop for free, which is the day a player stays in
    /// the mine instead of hauling a watering can around.
    Rain,
    /// Rain with lightning.
    Storm,
    /// Winter precipitation. Looks like rain but waters nothing.
    Snow,
}

impl Weather {
    /// True when the weather waters crops on its own.
    #[must_use]
    pub const fn waters_crops(self) -> bool {
        matches!(self, Weather::Rain | Weather::Storm)
    }

    /// The weather's display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Weather::Clear => "Clear",
            Weather::Cloudy => "Cloudy",
            Weather::Rain => "Rain",
            Weather::Storm => "Storm",
            Weather::Snow => "Snow",
        }
    }

    /// How much this weather darkens the scene.
    #[must_use]
    pub fn dimming(self) -> f32 {
        match self {
            Weather::Clear => 1.0,
            Weather::Cloudy => 0.88,
            Weather::Rain => 0.74,
            Weather::Storm => 0.62,
            Weather::Snow => 0.92,
        }
    }

    /// Picks tomorrow's weather for a season.
    ///
    /// Probabilities are per season rather than uniform: rain in spring is
    /// common and welcome, rain in winter is snow, and a summer storm is rare
    /// enough to be an event.
    #[must_use]
    pub fn roll(season: Season, rng: &mut Rng) -> Weather {
        let roll = rng.below(100);
        match season {
            Season::Spring => match roll {
                0..=34 => Weather::Rain,
                35..=49 => Weather::Cloudy,
                50..=53 => Weather::Storm,
                _ => Weather::Clear,
            },
            Season::Summer => match roll {
                0..=17 => Weather::Rain,
                18..=27 => Weather::Cloudy,
                28..=33 => Weather::Storm,
                _ => Weather::Clear,
            },
            Season::Autumn => match roll {
                0..=29 => Weather::Rain,
                30..=49 => Weather::Cloudy,
                50..=54 => Weather::Storm,
                _ => Weather::Clear,
            },
            Season::Winter => match roll {
                0..=44 => Weather::Snow,
                45..=64 => Weather::Cloudy,
                _ => Weather::Clear,
            },
        }
    }
}

/// The game clock.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calendar {
    /// Days elapsed since the game began, starting at zero.
    pub day: u32,
    /// Minute within the current day, from [`DAY_START_MINUTE`].
    pub minute: u32,
    /// Today's weather.
    pub weather: Weather,
    /// Tomorrow's weather, known a day ahead so the forecast can show it.
    pub forecast: Weather,
    /// Fractional accumulator, so a partial simulation step is not lost.
    seconds_into_minute: Fx,
}

impl Calendar {
    /// A new game's calendar: spring, day one, six in the morning.
    #[must_use]
    pub fn new(rng: &mut Rng) -> Calendar {
        Calendar {
            day: 0,
            minute: DAY_START_MINUTE,
            // The first day is always clear, so a new player is not immediately
            // rained on before they own a watering can.
            weather: Weather::Clear,
            forecast: Weather::roll(Season::Spring, rng),
            seconds_into_minute: Fx::ZERO,
        }
    }

    /// The current season.
    #[must_use]
    pub fn season(&self) -> Season {
        Season::ALL[((self.day / DAYS_PER_SEASON) % 4) as usize]
    }

    /// The year, counting from one.
    #[must_use]
    pub fn year(&self) -> u32 {
        self.day / (DAYS_PER_SEASON * 4) + 1
    }

    /// The day within the season, counting from one.
    #[must_use]
    pub fn day_of_season(&self) -> u32 {
        self.day % DAYS_PER_SEASON + 1
    }

    /// The hour and minute on a 24-hour clock.
    #[must_use]
    pub fn clock(&self) -> (u32, u32) {
        ((self.minute / 60) % 24, self.minute % 60)
    }

    /// The date and time, formatted for the HUD.
    #[must_use]
    pub fn display(&self) -> String {
        let (hour, minute) = self.clock();
        format!(
            "{} {} - Y{} - {hour:02}:{minute:02}",
            self.season().name(),
            self.day_of_season(),
            self.year()
        )
    }

    /// Advances the clock, returning whether a new day began.
    ///
    /// The day rolls over on its own at [`COLLAPSE_MINUTE`]: a player who works
    /// past two in the morning collapses and wakes up having lost some of the
    /// next day, which is the genre's standard way of enforcing a bedtime
    /// without a hard stop.
    pub fn advance(&mut self, dt: Fx, rng: &mut Rng) -> DayTransition {
        self.seconds_into_minute += dt;
        let mut rolled_over = false;

        while self.seconds_into_minute >= SECONDS_PER_GAME_MINUTE {
            self.seconds_into_minute -= SECONDS_PER_GAME_MINUTE;
            self.minute += 1;
            if self.minute >= COLLAPSE_MINUTE {
                self.begin_next_day(rng);
                rolled_over = true;
            }
        }

        if rolled_over {
            DayTransition::Collapsed
        } else {
            DayTransition::None
        }
    }

    /// Ends the day deliberately, as sleeping does.
    pub fn sleep(&mut self, rng: &mut Rng) {
        self.begin_next_day(rng);
    }

    /// Rolls the calendar over to the next morning.
    fn begin_next_day(&mut self, rng: &mut Rng) {
        self.day += 1;
        self.minute = DAY_START_MINUTE;
        self.seconds_into_minute = Fx::ZERO;
        // Yesterday's forecast becomes today's weather, and a new forecast is
        // drawn for the season the *new* day falls in — which matters on the
        // last day of a season, when tomorrow is a different one.
        self.weather = self.forecast;
        self.forecast = Weather::roll(self.season(), rng);
    }

    /// How far through the day it is, from `0` at waking to `1` at collapse.
    #[must_use]
    pub fn day_progress(&self) -> Fx {
        let elapsed = self.minute.saturating_sub(DAY_START_MINUTE);
        Fx::from_num(i32::try_from(elapsed).unwrap_or(0))
            / Fx::from_num(i32::try_from(MINUTES_PER_DAY).unwrap_or(1))
    }

    /// The ambient light colour for the current time, season and weather.
    ///
    /// Combining all three in one tint is what lets the renderer apply the
    /// entire day/night cycle as a single uniform, with no per-sprite work.
    #[must_use]
    pub fn ambient_light(&self) -> Color {
        let (hour, _) = self.clock();
        // A five-stop curve through the day. Dawn and dusk are warm and dim,
        // midday is neutral and bright, night is dark and blue.
        let daylight = match hour {
            6 => Color::rgb(0.72, 0.62, 0.72),
            7 => Color::rgb(0.92, 0.82, 0.80),
            8..=16 => Color::rgb(1.0, 1.0, 1.0),
            17 => Color::rgb(1.0, 0.88, 0.74),
            18 => Color::rgb(0.92, 0.72, 0.62),
            19 => Color::rgb(0.72, 0.58, 0.62),
            20..=21 => Color::rgb(0.52, 0.48, 0.66),
            _ => Color::rgb(0.38, 0.40, 0.62),
        };

        let season = self.season().grade();
        let dimming = self.weather.dimming();
        Color::new(
            daylight.r * season.r * dimming,
            daylight.g * season.g * dimming,
            daylight.b * season.b * dimming,
            1.0,
        )
    }

    /// True when it is dark enough that lamps should be lit.
    #[must_use]
    pub fn is_night(&self) -> bool {
        let (hour, _) = self.clock();
        !(7..19).contains(&hour)
    }
}

/// What happened to the calendar during an advance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DayTransition {
    /// The day continued normally.
    None,
    /// The player stayed up past two and the day rolled over on its own.
    Collapsed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calendar() -> Calendar {
        Calendar::new(&mut Rng::new(1))
    }

    #[test]
    fn a_new_game_starts_on_the_first_morning_of_spring() {
        let calendar = calendar();
        assert_eq!(calendar.season(), Season::Spring);
        assert_eq!(calendar.year(), 1);
        assert_eq!(calendar.day_of_season(), 1);
        assert_eq!(calendar.clock(), (6, 0));
        assert_eq!(
            calendar.weather,
            Weather::Clear,
            "day one is never rained out"
        );
    }

    #[test]
    fn the_clock_advances_with_simulated_time() {
        let mut calendar = calendar();
        let mut rng = Rng::new(2);
        // Ten in-game minutes.
        calendar.advance(SECONDS_PER_GAME_MINUTE * 10, &mut rng);
        assert_eq!(calendar.clock(), (6, 10));
    }

    #[test]
    fn partial_minutes_accumulate_rather_than_being_lost() {
        let mut calendar = calendar();
        let mut rng = Rng::new(3);
        // Fed in twentieths, which is well under a minute each, so nothing
        // would advance at all if the remainder were discarded per call. The
        // count is deliberately more than twenty: 0.7 seconds is not exactly
        // representable in binary fixed point, so exactly twenty twentieths
        // land a hair short of a minute.
        for _ in 0..25 {
            calendar.advance(SECONDS_PER_GAME_MINUTE / 20, &mut rng);
        }
        assert_eq!(
            calendar.clock(),
            (6, 1),
            "the fractions must have accumulated"
        );
    }

    #[test]
    fn seasons_turn_over_after_their_days_run_out() {
        let mut calendar = calendar();
        calendar.day = DAYS_PER_SEASON - 1;
        assert_eq!(calendar.season(), Season::Spring);
        calendar.day = DAYS_PER_SEASON;
        assert_eq!(calendar.season(), Season::Summer);
        assert_eq!(calendar.day_of_season(), 1);
    }

    #[test]
    fn a_year_is_four_seasons() {
        let mut calendar = calendar();
        calendar.day = DAYS_PER_SEASON * 4 - 1;
        assert_eq!(calendar.year(), 1);
        assert_eq!(calendar.season(), Season::Winter);
        calendar.day = DAYS_PER_SEASON * 4;
        assert_eq!(calendar.year(), 2);
        assert_eq!(calendar.season(), Season::Spring);
    }

    #[test]
    fn staying_up_too_late_rolls_the_day_over() {
        let mut calendar = calendar();
        let mut rng = Rng::new(4);
        // Push right up to two in the morning.
        calendar.minute = COLLAPSE_MINUTE - 1;
        let transition = calendar.advance(SECONDS_PER_GAME_MINUTE, &mut rng);

        assert_eq!(transition, DayTransition::Collapsed);
        assert_eq!(calendar.day, 1);
        assert_eq!(
            calendar.clock(),
            (6, 0),
            "and wakes at six the next morning"
        );
    }

    #[test]
    fn sleeping_advances_to_the_next_morning() {
        let mut calendar = calendar();
        let mut rng = Rng::new(5);
        calendar.minute = DAY_START_MINUTE + 600;
        calendar.sleep(&mut rng);
        assert_eq!(calendar.day, 1);
        assert_eq!(calendar.clock(), (6, 0));
    }

    #[test]
    fn yesterdays_forecast_becomes_todays_weather() {
        let mut calendar = calendar();
        let mut rng = Rng::new(6);
        let forecast = calendar.forecast;
        calendar.sleep(&mut rng);
        assert_eq!(calendar.weather, forecast, "the forecast must be honest");
    }

    #[test]
    fn winter_brings_snow_and_never_rain() {
        let mut rng = Rng::new(7);
        let mut saw_snow = false;
        for _ in 0..300 {
            let weather = Weather::roll(Season::Winter, &mut rng);
            assert_ne!(weather, Weather::Rain, "winter precipitation is snow");
            assert_ne!(weather, Weather::Storm);
            saw_snow |= weather == Weather::Snow;
        }
        assert!(saw_snow);
    }

    #[test]
    fn spring_rains_more_than_summer() {
        let count_rain = |season: Season, seed: u64| {
            let mut rng = Rng::new(seed);
            (0..2000)
                .filter(|_| Weather::roll(season, &mut rng).waters_crops())
                .count()
        };
        assert!(count_rain(Season::Spring, 8) > count_rain(Season::Summer, 8));
    }

    #[test]
    fn only_rain_and_storms_water_crops() {
        assert!(Weather::Rain.waters_crops());
        assert!(Weather::Storm.waters_crops());
        assert!(!Weather::Clear.waters_crops());
        assert!(!Weather::Snow.waters_crops(), "snow must not water a crop");
    }

    #[test]
    fn day_progress_runs_from_zero_to_one() {
        let mut calendar = calendar();
        assert_eq!(calendar.day_progress(), Fx::ZERO);
        calendar.minute = DAY_START_MINUTE + MINUTES_PER_DAY / 2;
        assert!((calendar.day_progress().to_f64() - 0.5).abs() < 1e-6);
        calendar.minute = COLLAPSE_MINUTE - 1;
        assert!(calendar.day_progress() < Fx::ONE);
    }

    #[test]
    fn ambient_light_is_brightest_at_midday_and_dimmest_at_night() {
        let mut calendar = calendar();
        calendar.minute = 12 * 60;
        let noon = calendar.ambient_light();
        calendar.minute = 24 * 60;
        let midnight = calendar.ambient_light();
        assert!(
            noon.g > midnight.g,
            "midday should be brighter than midnight"
        );
        // Night is cooler than it is warm.
        assert!(midnight.b > midnight.r);
    }

    #[test]
    fn rain_darkens_the_scene() {
        let mut clear = calendar();
        clear.minute = 12 * 60;
        clear.weather = Weather::Clear;
        let mut wet = clear.clone();
        wet.weather = Weather::Storm;
        assert!(wet.ambient_light().g < clear.ambient_light().g);
    }

    #[test]
    fn night_is_reported_outside_daylight_hours() {
        let mut calendar = calendar();
        calendar.minute = 12 * 60;
        assert!(!calendar.is_night());
        calendar.minute = 22 * 60;
        assert!(calendar.is_night());
        calendar.minute = 6 * 60;
        assert!(calendar.is_night(), "six in the morning is still dark");
    }

    #[test]
    fn the_display_string_reads_as_a_date() {
        let mut calendar = calendar();
        calendar.day = DAYS_PER_SEASON + 4;
        calendar.minute = 14 * 60 + 30;
        assert_eq!(calendar.display(), "Summer 5 - Y1 - 14:30");
    }

    #[test]
    fn determinism_the_same_seed_produces_the_same_year() {
        let run = || {
            let mut rng = Rng::new(2024);
            let mut calendar = Calendar::new(&mut rng);
            let mut weather = Vec::new();
            for _ in 0..112 {
                calendar.sleep(&mut rng);
                weather.push(calendar.weather);
            }
            weather
        };
        assert_eq!(run(), run());
    }
}
