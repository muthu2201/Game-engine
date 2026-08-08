//! Particles.
//!
//! ## Deterministic by construction
//!
//! Particles here are simulated in fixed point from a seeded generator, so the
//! same seed produces the same rain on every machine and on every replay of a
//! recording. That is unusual — most engines simulate particles in floats on
//! whatever schedule the renderer happens to run at, which makes them
//! decorative but unreproducible. Here a particle can be part of a replay, and
//! a screenshot from a bug report can be regenerated exactly.
//!
//! Because of that, [`ParticleSystem::update`] must be driven from the fixed
//! simulation step rather than from frame time. Feeding it real elapsed
//! seconds would reintroduce exactly the non-determinism it avoids.
//!
//! ## Particles are sprites
//!
//! [`ParticleSystem::draw`] emits into an ordinary [`SpriteBatcher`], so
//! particles inherit the atlas, the depth sorting and the one-draw-call
//! batching that everything else uses. A thousand raindrops cost a thousand
//! instances in the same draw as the world, not a second pipeline.
//!
//! ## Pooling
//!
//! The pool is fixed at [`ParticleSystem::new`] and never grows. A system that
//! allocated on demand would let one runaway emitter consume memory until the
//! frame rate collapsed; a full pool simply refuses to spawn, which is visible
//! as thinner rain rather than as a stall.

use crate::sprite::{Color, DrawSprite, SpriteBatcher};
use verdant_core_math::{Fx, Rect, Rng, Vec2};

/// One live particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// Where it is, in world units.
    pub position: Vec2,
    /// How fast it is moving, in world units per second.
    pub velocity: Vec2,
    /// How long it has existed, in seconds.
    pub age: Fx,
    /// How long it will exist, in seconds.
    pub lifetime: Fx,
    /// Its size at birth, in world units.
    ///
    /// Two dimensions rather than one: a raindrop is a streak and a spark is
    /// a dot, and a system that can only make squares cannot draw rain at
    /// all — at one world unit across it is a single almost invisible pixel.
    pub start_size: Vec2,
    /// Its size at death.
    pub end_size: Vec2,
    /// Its tint at birth.
    pub start_color: Color,
    /// Its tint at death.
    pub end_color: Color,
    /// Current rotation, in radians.
    pub rotation: Fx,
    /// How fast it spins, in radians per second.
    pub spin: Fx,
    /// Constant acceleration acting on it, in world units per second squared.
    ///
    /// Carried per particle rather than read back from its emitter, so a
    /// particle keeps behaving as it was born even after the emitter moves or
    /// is reconfigured.
    pub gravity: Vec2,
    /// Fraction of velocity shed per second, from zero to one.
    pub drag: Fx,
    /// Which atlas layer it draws from.
    pub layer: u32,
    /// Which part of that layer.
    pub uv_rect: Rect,
    /// Draw order.
    pub z: i32,
}

impl Particle {
    /// How far through its life it is, from zero to one.
    #[must_use]
    pub fn progress(&self) -> Fx {
        if self.lifetime <= Fx::ZERO {
            return Fx::ONE;
        }
        (self.age / self.lifetime).clamp(Fx::ZERO, Fx::ONE)
    }

    /// True when it has outlived its lifetime.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.age >= self.lifetime
    }

    /// Its size right now.
    #[must_use]
    pub fn size(&self) -> Vec2 {
        let t = self.progress();
        self.start_size + (self.end_size - self.start_size) * t
    }

    /// Its tint right now.
    #[must_use]
    pub fn color(&self) -> Color {
        self.start_color
            .lerp(self.end_color, self.progress().to_f32())
    }

    /// Its sprite for this frame.
    #[must_use]
    pub fn to_sprite(&self) -> DrawSprite {
        DrawSprite::new(self.position, self.size(), self.layer)
            .with_uv(self.uv_rect)
            .at_z(self.z)
            .tinted(self.color())
            // Centred rather than bottom-anchored: a particle is a puff, not
            // something standing on the ground, and anchoring it at its feet
            // makes it appear to hover half its size too high.
            .anchored(Vec2::new(Fx::HALF, Fx::HALF))
    }
}

/// An inclusive range a value is drawn from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    /// The low end.
    pub min: Fx,
    /// The high end.
    pub max: Fx,
}

impl Range {
    /// A range between two values, in either order.
    #[must_use]
    pub fn new(a: Fx, b: Fx) -> Range {
        Range {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// A range with one value.
    #[must_use]
    pub const fn exactly(value: Fx) -> Range {
        Range {
            min: value,
            max: value,
        }
    }

    /// Draws a value from the range.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Fx {
        if self.min >= self.max {
            return self.min;
        }
        rng.range_fx(self.min, self.max)
    }
}

/// Where an emitter puts new particles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitShape {
    /// All from one spot.
    Point,
    /// Anywhere inside a circle.
    Circle {
        /// Its radius.
        radius: Fx,
    },
    /// Anywhere inside a rectangle, given as half-extents from the centre.
    Rect {
        /// Half its width and height.
        half_extents: Vec2,
    },
}

impl EmitShape {
    /// A position within the shape, relative to the emitter's origin.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec2 {
        match self {
            EmitShape::Point => Vec2::ZERO,
            EmitShape::Circle { radius } => {
                // The square root is what keeps the distribution even: taking
                // the radius directly clusters particles in the middle,
                // because the area of a ring grows with its radius.
                let angle = rng.range_fx(Fx::ZERO, Fx::TAU);
                let distance = rng.range_fx(Fx::ZERO, Fx::ONE).sqrt() * *radius;
                Vec2::new(angle.cos() * distance, angle.sin() * distance)
            }
            EmitShape::Rect { half_extents } => Vec2::new(
                rng.range_fx(-half_extents.x, half_extents.x),
                rng.range_fx(-half_extents.y, half_extents.y),
            ),
        }
    }
}

/// How an emitter behaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// Where it sits, in world units.
    pub position: Vec2,
    /// The area new particles appear in.
    pub shape: EmitShape,
    /// Particles per second. Zero emits only on demand.
    pub rate: Fx,
    /// How long particles live.
    pub lifetime: Range,
    /// How fast they start moving.
    pub speed: Range,
    /// Which way they go, in radians.
    pub direction: Fx,
    /// How far from that direction they may deviate, in radians.
    pub spread: Fx,
    /// Constant acceleration, in world units per second squared.
    pub gravity: Vec2,
    /// Fraction of velocity shed per second, from zero to one.
    pub drag: Fx,
    /// Size at birth, before [`Emitter::proportions`] is applied.
    pub start_size: Range,
    /// Size at death, as a multiple of the birth size.
    pub end_scale: Fx,
    /// Width and height multipliers applied to the sampled size.
    ///
    /// `(1, 1)` is square. Rain needs something like `(1, 5)`: a drop drawn
    /// square is one pixel and reads as nothing at all.
    pub proportions: Vec2,
    /// Tint at birth.
    pub start_color: Color,
    /// Tint at death. Fading the alpha to zero here is what stops particles
    /// vanishing abruptly.
    pub end_color: Color,
    /// How fast particles spin, in radians per second.
    pub spin: Range,
    /// Which atlas layer they draw from.
    pub layer: u32,
    /// Which part of that layer.
    pub uv_rect: Rect,
    /// Draw order.
    pub z: i32,
    /// Whether the emitter is currently producing.
    pub enabled: bool,
    /// Fractional particles carried between steps.
    ///
    /// Without this an emitter at three particles a second on a sixty hertz
    /// step would round to zero every step and emit nothing at all.
    debt: Fx,
}

impl Default for Emitter {
    fn default() -> Emitter {
        Emitter {
            position: Vec2::ZERO,
            shape: EmitShape::Point,
            rate: Fx::ZERO,
            lifetime: Range::exactly(Fx::ONE),
            speed: Range::exactly(Fx::ZERO),
            direction: Fx::ZERO,
            spread: Fx::ZERO,
            gravity: Vec2::ZERO,
            drag: Fx::ZERO,
            start_size: Range::exactly(Fx::ONE),
            end_scale: Fx::ONE,
            proportions: Vec2::ONE,
            start_color: Color::WHITE,
            end_color: Color::WHITE.with_alpha(0.0),
            spin: Range::exactly(Fx::ZERO),
            layer: 0,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            z: 0,
            enabled: true,
            debt: Fx::ZERO,
        }
    }
}

impl Emitter {
    /// An emitter at a position producing `rate` particles a second.
    #[must_use]
    pub fn new(position: Vec2, rate: Fx) -> Emitter {
        Emitter {
            position,
            rate,
            ..Emitter::default()
        }
    }

    /// Returns this emitter aimed along `direction` with a spread either side.
    #[must_use]
    pub const fn aimed(mut self, direction: Fx, spread: Fx) -> Emitter {
        self.direction = direction;
        self.spread = spread;
        self
    }

    /// Returns this emitter with a speed range.
    #[must_use]
    pub const fn with_speed(mut self, speed: Range) -> Emitter {
        self.speed = speed;
        self
    }

    /// Returns this emitter with a lifetime range.
    #[must_use]
    pub const fn with_lifetime(mut self, lifetime: Range) -> Emitter {
        self.lifetime = lifetime;
        self
    }

    /// Returns this emitter with a size range and end scale.
    #[must_use]
    pub const fn with_size(mut self, start: Range, end_scale: Fx) -> Emitter {
        self.start_size = start;
        self.end_scale = end_scale;
        self
    }

    /// Returns this emitter producing particles of the given proportions.
    #[must_use]
    pub const fn with_proportions(mut self, proportions: Vec2) -> Emitter {
        self.proportions = proportions;
        self
    }

    /// Returns this emitter fading from one colour to another.
    #[must_use]
    pub const fn with_colors(mut self, start: Color, end: Color) -> Emitter {
        self.start_color = start;
        self.end_color = end;
        self
    }

    /// Returns this emitter under a constant acceleration.
    #[must_use]
    pub const fn with_gravity(mut self, gravity: Vec2) -> Emitter {
        self.gravity = gravity;
        self
    }

    /// Returns this emitter drawing from an atlas region.
    #[must_use]
    pub const fn with_sprite(mut self, layer: u32, uv_rect: Rect, z: i32) -> Emitter {
        self.layer = layer;
        self.uv_rect = uv_rect;
        self.z = z;
        self
    }

    /// Builds one particle.
    fn spawn(&self, rng: &mut Rng) -> Particle {
        let offset = self.shape.sample(rng);
        let angle = if self.spread <= Fx::ZERO {
            self.direction
        } else {
            self.direction + rng.range_fx(-self.spread, self.spread)
        };
        let speed = self.speed.sample(rng);
        let base = self.start_size.sample(rng);
        let start_size = Vec2::new(base * self.proportions.x, base * self.proportions.y);

        Particle {
            position: self.position + offset,
            velocity: Vec2::new(angle.cos() * speed, angle.sin() * speed),
            age: Fx::ZERO,
            lifetime: self.lifetime.sample(rng).max(Fx::from_ratio(1, 1000)),
            start_size,
            end_size: start_size * self.end_scale,
            start_color: self.start_color,
            end_color: self.end_color,
            rotation: Fx::ZERO,
            spin: self.spin.sample(rng),
            gravity: self.gravity,
            drag: self.drag,
            layer: self.layer,
            uv_rect: self.uv_rect,
            z: self.z,
        }
    }
}

/// A pool of particles and the emitters filling it.
#[derive(Debug)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
    emitters: Vec<Emitter>,
    capacity: usize,
    rng: Rng,
    /// How many spawns were refused because the pool was full.
    dropped: u64,
}

impl ParticleSystem {
    /// A system holding at most `capacity` particles, seeded from `seed`.
    #[must_use]
    pub fn new(capacity: usize, seed: u64) -> ParticleSystem {
        ParticleSystem {
            particles: Vec::with_capacity(capacity),
            emitters: Vec::new(),
            capacity,
            rng: Rng::new(seed).derive("particles"),
            dropped: 0,
        }
    }

    /// Adds an emitter, returning its index.
    pub fn add_emitter(&mut self, emitter: Emitter) -> usize {
        self.emitters.push(emitter);
        self.emitters.len() - 1
    }

    /// An emitter, for moving or reconfiguring it.
    #[must_use]
    pub fn emitter_mut(&mut self, index: usize) -> Option<&mut Emitter> {
        self.emitters.get_mut(index)
    }

    /// How many emitters there are.
    #[must_use]
    pub fn emitter_count(&self) -> usize {
        self.emitters.len()
    }

    /// How many particles are alive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// True when nothing is alive.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// The most particles that can be alive at once.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many spawns have been refused for want of room.
    ///
    /// A rising count means an emitter is outpacing the pool, which shows up
    /// on screen as thinner effects rather than as a stall.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The live particles, for inspection and testing.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Removes every particle, leaving the emitters in place.
    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// Emits `count` particles from an emitter immediately.
    ///
    /// Returns how many were actually created, which is fewer than asked for
    /// when the pool fills.
    pub fn burst(&mut self, emitter: usize, count: usize) -> usize {
        let Some(config) = self.emitters.get(emitter).copied() else {
            return 0;
        };
        let mut created = 0;
        for _ in 0..count {
            if self.particles.len() >= self.capacity {
                self.dropped += 1;
                continue;
            }
            let particle = config.spawn(&mut self.rng);
            self.particles.push(particle);
            created += 1;
        }
        created
    }

    /// Advances every particle and emitter by one simulation step.
    ///
    /// `dt` must come from the fixed timestep, not from frame time: the whole
    /// point of simulating in fixed point from a seeded generator is lost if
    /// the step length varies with the frame rate.
    pub fn update(&mut self, dt: Fx) {
        if dt <= Fx::ZERO {
            return;
        }

        // Spawn first, so a burst emitted this step is visible this step
        // rather than a frame late.
        for index in 0..self.emitters.len() {
            let Some(emitter) = self.emitters.get_mut(index) else {
                continue;
            };
            if !emitter.enabled || emitter.rate <= Fx::ZERO {
                continue;
            }
            // Fractional particles carry over, so a slow emitter still emits.
            emitter.debt += emitter.rate * dt;
            let due = emitter.debt.floor_int().max(0);
            emitter.debt -= Fx::from_num(due);

            let config = *emitter;
            for _ in 0..due {
                if self.particles.len() >= self.capacity {
                    self.dropped += 1;
                    break;
                }
                let particle = config.spawn(&mut self.rng);
                self.particles.push(particle);
            }
        }

        // Then integrate. Backwards so a swap-remove cannot skip a particle.
        let mut index = self.particles.len();
        while index > 0 {
            index -= 1;
            let particle = &mut self.particles[index];

            particle.age += dt;
            if particle.is_dead() {
                self.particles.swap_remove(index);
                continue;
            }

            particle.velocity += particle.gravity * dt;
            if particle.drag > Fx::ZERO {
                // Shed a fraction of the velocity per second, clamped so a
                // large drag over a long step cannot reverse the motion.
                let kept = (Fx::ONE - particle.drag * dt).clamp(Fx::ZERO, Fx::ONE);
                particle.velocity *= kept;
            }
            particle.position += particle.velocity * dt;
            particle.rotation += particle.spin * dt;
        }
    }

    /// Queues every live particle into a batcher.
    pub fn draw(&self, batcher: &mut SpriteBatcher) {
        for particle in &self.particles {
            batcher.push(particle.to_sprite());
        }
    }

    /// Queues only the particles inside `visible`.
    ///
    /// Returns how many were drawn.
    pub fn draw_visible(&self, batcher: &mut SpriteBatcher, visible: Rect) -> usize {
        let mut drawn = 0;
        for particle in &self.particles {
            if batcher.push_visible(particle.to_sprite(), visible) {
                drawn += 1;
            }
        }
        drawn
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "colour assertions compare values the interpolation writes exactly \
              at the ends of a particle's life, where a tolerance would stop the \
              test checking what it claims to."
)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    /// An emitter that produces one particle a second, living one second.
    fn emitter() -> Emitter {
        Emitter::new(Vec2::ZERO, Fx::ONE)
    }

    /// One sixtieth of a second, the engine's step.
    fn step() -> Fx {
        Fx::from_ratio(1, 60)
    }

    #[test]
    fn a_new_system_is_empty() {
        let system = ParticleSystem::new(16, 1);
        assert!(system.is_empty());
        assert_eq!(system.capacity(), 16);
    }

    #[test]
    fn an_emitter_fills_the_pool_over_time() {
        let mut system = ParticleSystem::new(64, 1);
        system.add_emitter(Emitter::new(Vec2::ZERO, fx(10)));
        // Ten a second for one second.
        for _ in 0..60 {
            system.update(step());
        }
        assert!(
            system.len() >= 9,
            "expected about ten, got {}",
            system.len()
        );
    }

    #[test]
    fn a_slow_emitter_still_emits() {
        // The bug the fractional carry exists to prevent: three particles a
        // second on a sixty hertz step rounds to zero every step.
        let mut system = ParticleSystem::new(64, 1);
        let mut slow = Emitter::new(Vec2::ZERO, fx(3));
        slow.lifetime = Range::exactly(fx(10));
        system.add_emitter(slow);
        for _ in 0..60 {
            system.update(step());
        }
        assert!(system.len() >= 2, "a slow emitter emitted {}", system.len());
    }

    #[test]
    fn particles_die_at_their_lifetime() {
        let mut system = ParticleSystem::new(64, 1);
        let mut once = emitter();
        once.rate = Fx::ZERO;
        once.lifetime = Range::exactly(Fx::HALF);
        let index = system.add_emitter(once);
        system.burst(index, 5);
        assert_eq!(system.len(), 5);

        for _ in 0..31 {
            system.update(step());
        }
        assert_eq!(system.len(), 0, "they should all have expired");
    }

    #[test]
    fn a_burst_emits_immediately() {
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        assert_eq!(system.burst(index, 8), 8);
        assert_eq!(system.len(), 8);
    }

    #[test]
    fn a_burst_from_an_unknown_emitter_does_nothing() {
        let mut system = ParticleSystem::new(64, 1);
        assert_eq!(system.burst(99, 8), 0);
        assert!(system.is_empty());
    }

    #[test]
    fn the_pool_refuses_rather_than_growing() {
        // A runaway emitter should thin out, not consume memory until the
        // frame rate collapses.
        let mut system = ParticleSystem::new(10, 1);
        let index = system.add_emitter(emitter());
        assert_eq!(system.burst(index, 50), 10);
        assert_eq!(system.len(), 10);
        assert_eq!(system.dropped(), 40);
        assert_eq!(system.particles().len(), 10);
    }

    #[test]
    fn particles_move_along_their_velocity() {
        let mut system = ParticleSystem::new(16, 1);
        let mut sideways = emitter();
        sideways.rate = Fx::ZERO;
        sideways.speed = Range::exactly(fx(60));
        sideways.direction = Fx::ZERO; // +X
        sideways.lifetime = Range::exactly(fx(10));
        let index = system.add_emitter(sideways);
        system.burst(index, 1);

        for _ in 0..60 {
            system.update(step());
        }
        let particle = system.particles()[0];
        assert!(
            (particle.position.x - fx(60)).abs() < fx(2),
            "moved to {:?}",
            particle.position
        );
        assert!(particle.position.y.abs() < fx(1), "should not have drifted");
    }

    #[test]
    fn a_spread_scatters_directions() {
        let mut system = ParticleSystem::new(64, 1);
        let mut fan = emitter();
        fan.rate = Fx::ZERO;
        fan.speed = Range::exactly(fx(50));
        fan.spread = Fx::PI;
        fan.lifetime = Range::exactly(fx(10));
        let index = system.add_emitter(fan);
        system.burst(index, 32);

        let directions: std::collections::BTreeSet<i64> = system
            .particles()
            .iter()
            .map(|particle| particle.velocity.x.to_raw() / 100_000)
            .collect();
        assert!(directions.len() > 8, "only {} directions", directions.len());
    }

    #[test]
    fn no_spread_means_one_direction() {
        let mut system = ParticleSystem::new(64, 1);
        let mut straight = emitter();
        straight.rate = Fx::ZERO;
        straight.speed = Range::exactly(fx(50));
        straight.lifetime = Range::exactly(fx(10));
        let index = system.add_emitter(straight);
        system.burst(index, 8);

        let first = system.particles()[0].velocity;
        for particle in system.particles() {
            assert_eq!(particle.velocity, first);
        }
    }

    #[test]
    fn size_interpolates_across_a_life() {
        let mut particle = Particle {
            position: Vec2::ZERO,
            velocity: Vec2::ZERO,
            age: Fx::ZERO,
            lifetime: Fx::ONE,
            start_size: Vec2::splat(fx(4)),
            end_size: Vec2::splat(fx(8)),
            start_color: Color::WHITE,
            end_color: Color::WHITE,
            rotation: Fx::ZERO,
            spin: Fx::ZERO,
            gravity: Vec2::ZERO,
            drag: Fx::ZERO,
            layer: 0,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            z: 0,
        };
        assert_eq!(particle.size(), Vec2::splat(fx(4)));
        particle.age = Fx::HALF;
        assert_eq!(particle.size(), Vec2::splat(fx(6)));
        particle.age = Fx::ONE;
        assert_eq!(particle.size(), Vec2::splat(fx(8)));
    }

    #[test]
    fn colour_fades_across_a_life() {
        // Fading alpha to zero is what stops particles vanishing abruptly.
        let particle = Particle {
            position: Vec2::ZERO,
            velocity: Vec2::ZERO,
            age: Fx::ZERO,
            lifetime: Fx::ONE,
            start_size: Vec2::ONE,
            end_size: Vec2::ONE,
            start_color: Color::WHITE,
            end_color: Color::WHITE.with_alpha(0.0),
            rotation: Fx::ZERO,
            spin: Fx::ZERO,
            gravity: Vec2::ZERO,
            drag: Fx::ZERO,
            layer: 0,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            z: 0,
        };
        assert_eq!(particle.color().a, 1.0);

        let mut dying = particle;
        dying.age = Fx::ONE;
        assert_eq!(dying.color().a, 0.0);
    }

    #[test]
    fn a_zero_lifetime_does_not_divide_by_zero() {
        let particle = Particle {
            position: Vec2::ZERO,
            velocity: Vec2::ZERO,
            age: Fx::ZERO,
            lifetime: Fx::ZERO,
            start_size: Vec2::ONE,
            end_size: Vec2::ONE,
            start_color: Color::WHITE,
            end_color: Color::WHITE,
            rotation: Fx::ZERO,
            spin: Fx::ZERO,
            gravity: Vec2::ZERO,
            drag: Fx::ZERO,
            layer: 0,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            z: 0,
        };
        assert_eq!(particle.progress(), Fx::ONE);
        assert!(particle.is_dead());
    }

    #[test]
    fn a_spawned_particle_always_has_a_lifetime() {
        // Otherwise it dies on the step it was born and never appears.
        let mut system = ParticleSystem::new(16, 1);
        let mut instant = emitter();
        instant.rate = Fx::ZERO;
        instant.lifetime = Range::exactly(Fx::ZERO);
        let index = system.add_emitter(instant);
        system.burst(index, 1);
        assert!(system.particles()[0].lifetime > Fx::ZERO);
    }

    #[test]
    fn particles_are_drawn_centred_rather_than_standing_on_the_ground() {
        // A bottom-anchored puff appears to hover half its size too high.
        let particle = Particle {
            position: Vec2::from_ints(10, 20),
            velocity: Vec2::ZERO,
            age: Fx::ZERO,
            lifetime: Fx::ONE,
            start_size: Vec2::splat(fx(4)),
            end_size: Vec2::splat(fx(4)),
            start_color: Color::WHITE,
            end_color: Color::WHITE,
            rotation: Fx::ZERO,
            spin: Fx::ZERO,
            gravity: Vec2::ZERO,
            drag: Fx::ZERO,
            layer: 0,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
            z: 0,
        };
        let sprite = particle.to_sprite();
        assert_eq!(sprite.anchor, Vec2::new(Fx::HALF, Fx::HALF));
        assert_eq!(sprite.bounds().centre(), Vec2::from_ints(10, 20));
    }

    #[test]
    fn proportions_make_streaks_rather_than_squares() {
        // A raindrop drawn square is one pixel and reads as nothing at all,
        // which is exactly what the first version of this did.
        let mut system = ParticleSystem::new(16, 1);
        let mut rain = emitter();
        rain.rate = Fx::ZERO;
        rain.start_size = Range::exactly(fx(2));
        rain.proportions = Vec2::new(Fx::ONE, fx(5));
        let index = system.add_emitter(rain);
        system.burst(index, 1);

        let size = system.particles()[0].size();
        assert_eq!(size.x, fx(2));
        assert_eq!(size.y, fx(10));
        assert_eq!(system.particles()[0].to_sprite().size, size);
    }

    #[test]
    fn drawing_queues_every_live_particle() {
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        system.burst(index, 12);

        let mut batcher = SpriteBatcher::new();
        system.draw(&mut batcher);
        assert_eq!(batcher.len(), 12);
    }

    #[test]
    fn drawing_can_cull_to_the_view() {
        let mut system = ParticleSystem::new(64, 1);
        let mut here = emitter();
        here.rate = Fx::ZERO;
        here.shape = EmitShape::Rect {
            half_extents: Vec2::from_ints(500, 500),
        };
        let index = system.add_emitter(here);
        system.burst(index, 64);

        let mut batcher = SpriteBatcher::new();
        let visible = Rect::from_ints(-20, -20, 40, 40);
        let drawn = system.draw_visible(&mut batcher, visible);
        assert!(drawn < 64, "culling should have dropped some: {drawn}");
        assert_eq!(batcher.len(), drawn);
    }

    #[test]
    fn gravity_pulls_particles_down() {
        let mut system = ParticleSystem::new(16, 1);
        let mut falling = emitter();
        falling.rate = Fx::ZERO;
        falling.lifetime = Range::exactly(fx(10));
        falling.gravity = Vec2::new(Fx::ZERO, fx(100));
        let index = system.add_emitter(falling);
        system.burst(index, 1);

        for _ in 0..60 {
            system.update(step());
        }
        let particle = system.particles()[0];
        assert!(
            particle.velocity.y > fx(90),
            "gravity did not accelerate it"
        );
        assert!(particle.position.y > fx(40), "it did not fall");
    }

    #[test]
    fn drag_slows_particles_down() {
        let mut system = ParticleSystem::new(16, 1);
        let mut dragged = emitter();
        dragged.rate = Fx::ZERO;
        dragged.lifetime = Range::exactly(fx(10));
        dragged.speed = Range::exactly(fx(100));
        dragged.drag = fx(2);
        let index = system.add_emitter(dragged);
        system.burst(index, 1);

        for _ in 0..60 {
            system.update(step());
        }
        let speed = system.particles()[0].velocity.length();
        assert!(speed < fx(20), "drag left it at {speed}");
        assert!(speed >= Fx::ZERO, "drag must not reverse the motion");
    }

    #[test]
    fn heavy_drag_over_a_long_step_cannot_reverse_motion() {
        // Without the clamp, `1 - drag * dt` goes negative and the particle
        // flies backwards.
        let mut system = ParticleSystem::new(16, 1);
        let mut dragged = emitter();
        dragged.rate = Fx::ZERO;
        dragged.lifetime = Range::exactly(fx(10));
        dragged.speed = Range::exactly(fx(100));
        dragged.drag = fx(50);
        let index = system.add_emitter(dragged);
        system.burst(index, 1);

        system.update(Fx::HALF);
        assert!(
            system.particles()[0].velocity.x >= Fx::ZERO,
            "drag reversed the particle"
        );
    }

    #[test]
    fn a_particle_keeps_its_own_motion_after_its_emitter_changes() {
        let mut system = ParticleSystem::new(16, 1);
        let mut falling = emitter();
        falling.rate = Fx::ZERO;
        falling.lifetime = Range::exactly(fx(10));
        falling.gravity = Vec2::new(Fx::ZERO, fx(100));
        let index = system.add_emitter(falling);
        system.burst(index, 1);

        // Turn the emitter's gravity off after the fact.
        system.emitter_mut(index).expect("it exists").gravity = Vec2::ZERO;
        for _ in 0..30 {
            system.update(step());
        }
        assert!(
            system.particles()[0].velocity.y > Fx::ZERO,
            "the live particle should still be falling"
        );
    }

    #[test]
    fn particles_spin() {
        let mut system = ParticleSystem::new(16, 1);
        let mut spinning = emitter();
        spinning.rate = Fx::ZERO;
        spinning.lifetime = Range::exactly(fx(10));
        spinning.spin = Range::exactly(Fx::PI);
        let index = system.add_emitter(spinning);
        system.burst(index, 1);

        for _ in 0..60 {
            system.update(step());
        }
        let rotation = system.particles()[0].rotation;
        assert!(
            (rotation - Fx::PI).abs() < Fx::from_ratio(1, 10),
            "at {rotation}"
        );
    }

    #[test]
    fn simulation_is_deterministic() {
        // The property that lets a particle be part of a replay.
        let run = || {
            let mut system = ParticleSystem::new(256, 20_260_808);
            let mut smoke = Emitter::new(Vec2::ZERO, fx(30));
            smoke.speed = Range::new(fx(10), fx(40));
            smoke.spread = Fx::PI;
            smoke.lifetime = Range::new(Fx::HALF, fx(2));
            system.add_emitter(smoke);
            for _ in 0..120 {
                system.update(step());
            }
            system
                .particles()
                .iter()
                .map(|particle| (particle.position, particle.velocity, particle.age))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_different_seed_produces_different_particles() {
        let run = |seed: u64| {
            let mut system = ParticleSystem::new(256, seed);
            let mut smoke = Emitter::new(Vec2::ZERO, fx(30));
            smoke.speed = Range::new(fx(10), fx(40));
            smoke.spread = Fx::PI;
            system.add_emitter(smoke);
            for _ in 0..60 {
                system.update(step());
            }
            system
                .particles()
                .iter()
                .map(|particle| particle.position)
                .collect::<Vec<_>>()
        };
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn a_disabled_emitter_produces_nothing() {
        let mut system = ParticleSystem::new(64, 1);
        let mut off = Emitter::new(Vec2::ZERO, fx(60));
        off.enabled = false;
        system.add_emitter(off);
        for _ in 0..60 {
            system.update(step());
        }
        assert!(system.is_empty());
    }

    #[test]
    fn a_zero_step_changes_nothing() {
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        system.burst(index, 4);
        let before: Vec<Vec2> = system.particles().iter().map(|p| p.position).collect();
        system.update(Fx::ZERO);
        let after: Vec<Vec2> = system.particles().iter().map(|p| p.position).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn a_negative_step_changes_nothing() {
        // Time cannot run backwards, and letting it would age particles into
        // a negative lifetime.
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        system.burst(index, 4);
        system.update(-step());
        assert_eq!(system.len(), 4);
    }

    #[test]
    fn clearing_removes_particles_but_keeps_emitters() {
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        system.burst(index, 4);
        system.clear();
        assert!(system.is_empty());
        assert_eq!(system.emitter_count(), 1);
    }

    #[test]
    fn an_emitter_can_be_moved() {
        let mut system = ParticleSystem::new(64, 1);
        let index = system.add_emitter(emitter());
        system.emitter_mut(index).expect("it exists").position = Vec2::from_ints(100, 100);
        system.burst(index, 1);
        assert_eq!(system.particles()[0].position, Vec2::from_ints(100, 100));
    }

    #[test]
    fn a_circle_emitter_fills_its_area_evenly() {
        // Sampling the radius directly clusters particles in the middle,
        // because a ring's area grows with its radius.
        let mut rng = Rng::new(7);
        let shape = EmitShape::Circle { radius: fx(100) };
        let mut inner = 0;
        let mut outer = 0;
        for _ in 0..2000 {
            let point = shape.sample(&mut rng);
            // Half the radius bounds a quarter of the area.
            if point.length() < fx(50) {
                inner += 1;
            } else {
                outer += 1;
            }
        }
        assert!(
            outer > inner * 2,
            "expected roughly three times as many outside: {inner} inner, {outer} outer"
        );
    }

    #[test]
    fn every_emit_shape_stays_within_its_bounds() {
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            assert_eq!(EmitShape::Point.sample(&mut rng), Vec2::ZERO);

            let circle = EmitShape::Circle { radius: fx(10) };
            assert!(circle.sample(&mut rng).length() <= fx(10) + Fx::from_ratio(1, 100));

            let half = Vec2::from_ints(5, 3);
            let point = EmitShape::Rect { half_extents: half }.sample(&mut rng);
            assert!(point.x.abs() <= half.x && point.y.abs() <= half.y);
        }
    }

    #[test]
    fn a_range_orders_its_ends() {
        let range = Range::new(fx(10), fx(2));
        assert_eq!(range.min, fx(2));
        assert_eq!(range.max, fx(10));
    }

    #[test]
    fn a_single_valued_range_always_samples_the_same() {
        let mut rng = Rng::new(1);
        let range = Range::exactly(fx(7));
        for _ in 0..10 {
            assert_eq!(range.sample(&mut rng), fx(7));
        }
    }
}
