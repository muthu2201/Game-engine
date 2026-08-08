//! Ordered, deterministic system execution.
//!
//! # Execution model
//!
//! Systems are plain `FnMut(&mut World)` closures grouped into [`Stage`]s.
//! Stages run in the order they were declared; within a stage, systems run in a
//! topological order derived from their `after`/`before` constraints, with ties
//! broken by insertion order.
//!
//! Execution is **sequential by design**, not as a placeholder for parallelism.
//! The simulation must produce bit-identical results across machines for
//! replays and desync detection to work, and a work-stealing pool reorders
//! independent systems run to run. Sequential execution in a declared order
//! removes that entire class of nondeterminism.
//!
//! # Ambiguity detection
//!
//! Sequential execution makes conflicting access safe but not *correct*: if two
//! systems both write `Position` and neither is ordered relative to the other,
//! the result depends on insertion order, which is a latent bug the moment
//! someone reorders the registration calls. Systems can therefore declare what
//! they touch via [`SystemAccess`], and [`Schedule::ambiguities`] reports every
//! unordered conflicting pair. [`Schedule::run_strict`] turns those reports into
//! a hard error.

use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::world::World;
use std::collections::{HashMap, HashSet};

/// A system's position in a schedule.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SystemId(pub usize);

impl SystemId {
    /// The system's registration index within its schedule.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A named phase of the frame.
///
/// Stages exist so that ordering can be expressed coarsely ("all input before
/// all simulation") without every system needing a pairwise constraint.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Stage(pub &'static str);

impl Stage {
    /// Reads input devices and publishes the input snapshot.
    pub const INPUT: Stage = Stage("input");
    /// Advances the simulation. Runs zero or more times per rendered frame.
    pub const UPDATE: Stage = Stage("update");
    /// Resolves movement and collisions after gameplay has set velocities.
    pub const PHYSICS: Stage = Stage("physics");
    /// Applies deferred structural changes and maintains derived state.
    pub const MAINTAIN: Stage = Stage("maintain");
    /// Extracts what the renderer needs. Never mutates simulation state.
    pub const EXTRACT: Stage = Stage("extract");
}

/// What a system reads and writes, for ambiguity detection.
///
/// Declaring access is optional. A system that declares nothing is treated as
/// touching everything, so it is ordered-ambiguous with every other undeclared
/// system — which is the safe default, and a nudge to declare.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SystemAccess {
    /// Components the system reads.
    pub reads: Vec<ComponentId>,
    /// Components the system writes.
    pub writes: Vec<ComponentId>,
    /// True when the system did not declare its access.
    pub unrestricted: bool,
}

impl SystemAccess {
    /// Access covering everything. The default for undeclared systems.
    #[must_use]
    pub fn unrestricted() -> SystemAccess {
        SystemAccess {
            reads: Vec::new(),
            writes: Vec::new(),
            unrestricted: true,
        }
    }

    /// An empty declaration, to be extended with [`SystemAccess::reads`] and
    /// [`SystemAccess::writes`].
    #[must_use]
    pub fn none() -> SystemAccess {
        SystemAccess {
            reads: Vec::new(),
            writes: Vec::new(),
            unrestricted: false,
        }
    }

    /// Declares a read of `T`.
    #[must_use]
    pub fn read<T: Component>(mut self, registry: &mut ComponentRegistry) -> SystemAccess {
        self.reads.push(registry.register::<T>());
        self
    }

    /// Declares a write of `T`.
    #[must_use]
    pub fn write<T: Component>(mut self, registry: &mut ComponentRegistry) -> SystemAccess {
        self.writes.push(registry.register::<T>());
        self
    }

    /// True when running these two systems in either order could differ.
    ///
    /// Two reads never conflict; a write conflicts with any other access to the
    /// same component.
    #[must_use]
    pub fn conflicts_with(&self, other: &SystemAccess) -> bool {
        if self.unrestricted || other.unrestricted {
            return true;
        }
        let self_writes: HashSet<ComponentId> = self.writes.iter().copied().collect();
        let other_writes: HashSet<ComponentId> = other.writes.iter().copied().collect();
        self_writes
            .iter()
            .any(|id| other_writes.contains(id) || other.reads.contains(id))
            || other_writes.iter().any(|id| self.reads.contains(id))
    }
}

/// The boxed system body.
type SystemFn = Box<dyn FnMut(&mut World) + Send>;
/// The boxed run condition.
type ConditionFn = Box<dyn FnMut(&World) -> bool + Send>;

/// One registered system.
struct SystemEntry {
    name: String,
    stage: Stage,
    run: SystemFn,
    /// Names this system must run after.
    after: Vec<String>,
    /// Names this system must run before.
    before: Vec<String>,
    condition: Option<ConditionFn>,
    access: SystemAccess,
    enabled: bool,
}

/// An ordered collection of systems.
pub struct Schedule {
    systems: Vec<SystemEntry>,
    by_name: HashMap<String, usize>,
    stage_order: Vec<Stage>,
    /// Cached execution order, invalidated whenever a system is added.
    resolved: Option<Vec<usize>>,
}

impl Default for Schedule {
    fn default() -> Schedule {
        Schedule::new()
    }
}

impl Schedule {
    /// Creates a schedule with the default stage order.
    #[must_use]
    pub fn new() -> Schedule {
        Schedule {
            systems: Vec::new(),
            by_name: HashMap::new(),
            stage_order: vec![
                Stage::INPUT,
                Stage::UPDATE,
                Stage::PHYSICS,
                Stage::MAINTAIN,
                Stage::EXTRACT,
            ],
            resolved: None,
        }
    }

    /// Replaces the stage order.
    ///
    /// Systems in a stage not listed here run last, after every declared stage,
    /// in the order they were added — so a forgotten stage degrades to "runs at
    /// the end" rather than silently never running.
    pub fn set_stage_order(&mut self, stages: Vec<Stage>) {
        self.stage_order = stages;
        self.resolved = None;
    }

    /// Adds a system to [`Stage::UPDATE`].
    ///
    /// # Panics
    ///
    /// Panics if `name` is already registered — duplicate names would make the
    /// ordering constraints ambiguous.
    pub fn add_system(
        &mut self,
        name: &str,
        system: impl FnMut(&mut World) + Send + 'static,
    ) -> SystemId {
        self.add_system_to_stage(Stage::UPDATE, name, system)
    }

    /// Adds a system to a specific stage.
    ///
    /// # Panics
    ///
    /// Panics if `name` is already registered.
    pub fn add_system_to_stage(
        &mut self,
        stage: Stage,
        name: &str,
        system: impl FnMut(&mut World) + Send + 'static,
    ) -> SystemId {
        assert!(
            !self.by_name.contains_key(name),
            "a system named `{name}` is already registered in this schedule"
        );
        let id = SystemId(self.systems.len());
        self.by_name.insert(name.to_string(), id.0);
        self.systems.push(SystemEntry {
            name: name.to_string(),
            stage,
            run: Box::new(system),
            after: Vec::new(),
            before: Vec::new(),
            condition: None,
            access: SystemAccess::unrestricted(),
            enabled: true,
        });
        self.resolved = None;
        id
    }

    /// Adds a system constrained to run after another.
    ///
    /// # Panics
    ///
    /// Panics if `name` is already registered.
    pub fn add_system_after(
        &mut self,
        name: &str,
        after: &str,
        system: impl FnMut(&mut World) + Send + 'static,
    ) -> SystemId {
        let id = self.add_system(name, system);
        self.systems[id.0].after.push(after.to_string());
        id
    }

    /// Constrains an existing system to run after another.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered.
    pub fn order_after(&mut self, name: &str, after: &str) {
        let index = self.index_of(name);
        self.systems[index].after.push(after.to_string());
        self.resolved = None;
    }

    /// Constrains an existing system to run before another.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered.
    pub fn order_before(&mut self, name: &str, before: &str) {
        let index = self.index_of(name);
        self.systems[index].before.push(before.to_string());
        self.resolved = None;
    }

    /// Attaches a run condition, evaluated each time the schedule runs.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered.
    pub fn set_condition(
        &mut self,
        name: &str,
        condition: impl FnMut(&World) -> bool + Send + 'static,
    ) {
        let index = self.index_of(name);
        self.systems[index].condition = Some(Box::new(condition));
    }

    /// Declares what a system touches, enabling ambiguity detection for it.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered.
    pub fn set_access(&mut self, name: &str, access: SystemAccess) {
        let index = self.index_of(name);
        self.systems[index].access = access;
    }

    /// Enables or disables a system without removing it.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        let index = self.index_of(name);
        self.systems[index].enabled = enabled;
    }

    /// Number of registered systems.
    #[must_use]
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// True when no systems are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// The names of every system, in execution order.
    ///
    /// # Panics
    ///
    /// Panics if the ordering constraints contain a cycle.
    #[must_use]
    pub fn execution_order(&mut self) -> Vec<&str> {
        self.resolve();
        let order = self.resolved.as_ref().expect("resolve populates the cache");
        order
            .iter()
            .map(|index| self.systems[*index].name.as_str())
            .collect()
    }

    /// Runs every enabled system whose condition passes.
    ///
    /// # Panics
    ///
    /// Panics if the ordering constraints contain a cycle.
    pub fn run(&mut self, world: &mut World) {
        self.resolve();
        let order = self.resolved.clone().expect("resolve populates the cache");
        for index in order {
            let entry = &mut self.systems[index];
            if !entry.enabled {
                continue;
            }
            if let Some(condition) = entry.condition.as_mut() {
                if !condition(world) {
                    continue;
                }
            }
            (entry.run)(world);
        }
    }

    /// Runs the schedule, first rejecting any unordered conflicting pair.
    ///
    /// Worth calling once at startup, or in a test, rather than every frame.
    ///
    /// # Errors
    ///
    /// Returns every ambiguous pair when at least one exists; the schedule is
    /// not run in that case.
    ///
    /// # Panics
    ///
    /// Panics if the ordering constraints contain a cycle.
    pub fn run_strict(&mut self, world: &mut World) -> Result<(), Vec<(String, String)>> {
        let ambiguities = self.ambiguities();
        if !ambiguities.is_empty() {
            return Err(ambiguities);
        }
        self.run(world);
        Ok(())
    }

    /// Every pair of systems in the same stage that conflict on component
    /// access without an ordering constraint between them.
    ///
    /// # Panics
    ///
    /// Panics if the ordering constraints contain a cycle.
    #[must_use]
    pub fn ambiguities(&mut self) -> Vec<(String, String)> {
        self.resolve();
        let order = self.resolved.clone().expect("resolve populates the cache");
        let mut ambiguous = Vec::new();
        for (position, first) in order.iter().enumerate() {
            for second in order.iter().skip(position + 1) {
                let (a, b) = (&self.systems[*first], &self.systems[*second]);
                if a.stage != b.stage {
                    continue;
                }
                if !a.access.conflicts_with(&b.access) {
                    continue;
                }
                if self.is_ordered(*first, *second) {
                    continue;
                }
                ambiguous.push((a.name.clone(), b.name.clone()));
            }
        }
        ambiguous
    }

    /// True when an explicit constraint orders these two systems.
    fn is_ordered(&self, first: usize, second: usize) -> bool {
        let a = &self.systems[first];
        let b = &self.systems[second];
        a.after.contains(&b.name)
            || a.before.contains(&b.name)
            || b.after.contains(&a.name)
            || b.before.contains(&a.name)
    }

    /// Resolves the index of a registered system.
    ///
    /// # Panics
    ///
    /// Panics if the name is unknown, listing what is registered.
    fn index_of(&self, name: &str) -> usize {
        *self.by_name.get(name).unwrap_or_else(|| {
            let known: Vec<&str> = self.by_name.keys().map(String::as_str).collect();
            panic!("no system named `{name}` is registered; known systems: {known:?}")
        })
    }

    /// Computes and caches the execution order.
    ///
    /// # Panics
    ///
    /// Panics if the constraints contain a cycle, naming the systems involved.
    fn resolve(&mut self) {
        if self.resolved.is_some() {
            return;
        }

        // Systems are grouped by stage first; ordering constraints only apply
        // within a stage, since stages are already totally ordered.
        let mut stage_rank: HashMap<&Stage, usize> = HashMap::new();
        for (rank, stage) in self.stage_order.iter().enumerate() {
            stage_rank.insert(stage, rank);
        }
        let undeclared_rank = self.stage_order.len();

        let mut indices: Vec<usize> = (0..self.systems.len()).collect();
        // Stable sort keyed by stage rank: within a stage, insertion order is
        // preserved, which is the tie-break that makes the result deterministic.
        indices.sort_by_key(|index| {
            stage_rank
                .get(&self.systems[*index].stage)
                .copied()
                .unwrap_or(undeclared_rank)
        });

        // Build the dependency edges, both from `after` and from the inverse of
        // `before`, so the two spellings are equivalent.
        let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();
        for index in &indices {
            dependencies.entry(*index).or_default();
        }
        for index in &indices {
            let entry = &self.systems[*index];
            for predecessor in &entry.after {
                if let Some(other) = self.by_name.get(predecessor) {
                    if self.systems[*other].stage == entry.stage {
                        dependencies.entry(*index).or_default().insert(*other);
                    }
                }
            }
            for successor in &entry.before {
                if let Some(other) = self.by_name.get(successor) {
                    if self.systems[*other].stage == entry.stage {
                        dependencies.entry(*other).or_default().insert(*index);
                    }
                }
            }
        }

        // Kahn's algorithm, always taking the lowest-ranked ready system so the
        // output is a function of the constraints alone.
        let mut order = Vec::with_capacity(indices.len());
        let mut remaining: Vec<usize> = indices.clone();
        while !remaining.is_empty() {
            let ready = remaining.iter().position(|index| {
                dependencies
                    .get(index)
                    .is_none_or(|needs| needs.iter().all(|need| order.contains(need)))
            });
            match ready {
                Some(position) => {
                    let index = remaining.remove(position);
                    order.push(index);
                }
                None => {
                    let stuck: Vec<&str> = remaining
                        .iter()
                        .map(|index| self.systems[*index].name.as_str())
                        .collect();
                    panic!("system ordering constraints contain a cycle involving: {stuck:?}");
                }
            }
        }
        self.resolved = Some(order);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct Counter(u32);
    impl Component for Counter {}

    #[derive(Debug)]
    struct Other;
    impl Component for Other {}

    /// Records the order systems ran in.
    fn recording_system(
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
        name: &'static str,
    ) -> impl FnMut(&mut World) + Send + 'static {
        move |_world: &mut World| {
            log.lock().expect("log mutex poisoned").push(name);
        }
    }

    #[test]
    fn systems_run_in_insertion_order_by_default() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();
        schedule.add_system("first", recording_system(Arc::clone(&log), "first"));
        schedule.add_system("second", recording_system(Arc::clone(&log), "second"));
        schedule.add_system("third", recording_system(Arc::clone(&log), "third"));

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(*log.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn after_constraints_reorder_execution() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();
        schedule.add_system("late", recording_system(Arc::clone(&log), "late"));
        schedule.add_system("early", recording_system(Arc::clone(&log), "early"));
        schedule.order_after("late", "early");

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(*log.lock().unwrap(), vec!["early", "late"]);
    }

    #[test]
    fn before_and_after_are_equivalent_spellings() {
        let mut with_after = Schedule::new();
        with_after.add_system("a", |_: &mut World| {});
        with_after.add_system("b", |_: &mut World| {});
        with_after.order_after("a", "b");

        let mut with_before = Schedule::new();
        with_before.add_system("a", |_: &mut World| {});
        with_before.add_system("b", |_: &mut World| {});
        with_before.order_before("b", "a");

        assert_eq!(with_after.execution_order(), with_before.execution_order());
    }

    #[test]
    fn stages_run_in_declared_order_regardless_of_insertion() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();
        schedule.add_system_to_stage(
            Stage::EXTRACT,
            "render",
            recording_system(Arc::clone(&log), "render"),
        );
        schedule.add_system_to_stage(
            Stage::INPUT,
            "input",
            recording_system(Arc::clone(&log), "input"),
        );
        schedule.add_system_to_stage(
            Stage::UPDATE,
            "logic",
            recording_system(Arc::clone(&log), "logic"),
        );

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(*log.lock().unwrap(), vec!["input", "logic", "render"]);
    }

    #[test]
    fn systems_in_an_undeclared_stage_run_last() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();
        schedule.add_system_to_stage(
            Stage("custom"),
            "custom",
            recording_system(Arc::clone(&log), "custom"),
        );
        schedule.add_system_to_stage(
            Stage::UPDATE,
            "update",
            recording_system(Arc::clone(&log), "update"),
        );

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(*log.lock().unwrap(), vec!["update", "custom"]);
    }

    #[test]
    #[should_panic(expected = "contain a cycle")]
    fn a_cycle_is_reported_rather_than_silently_reordered() {
        let mut schedule = Schedule::new();
        schedule.add_system("a", |_: &mut World| {});
        schedule.add_system("b", |_: &mut World| {});
        schedule.order_after("a", "b");
        schedule.order_after("b", "a");
        let _ = schedule.execution_order();
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_system_names_are_rejected() {
        let mut schedule = Schedule::new();
        schedule.add_system("duplicate", |_: &mut World| {});
        schedule.add_system("duplicate", |_: &mut World| {});
    }

    #[test]
    #[should_panic(expected = "known systems")]
    fn ordering_against_an_unknown_system_names_what_exists() {
        let mut schedule = Schedule::new();
        schedule.add_system("real", |_: &mut World| {});
        schedule.order_after("missing", "real");
    }

    #[test]
    fn disabled_systems_are_skipped() {
        let ran = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&ran);
        let mut schedule = Schedule::new();
        schedule.add_system("counted", move |_: &mut World| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        schedule.set_enabled("counted", false);

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 0);

        schedule.set_enabled("counted", true);
        schedule.run(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn run_conditions_gate_execution() {
        let ran = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&ran);
        let mut schedule = Schedule::new();
        schedule.add_system("gated", move |_: &mut World| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        // Only run once the world has an entity.
        schedule.set_condition("gated", |world: &World| !world.is_empty());

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 0);

        world.spawn((Counter(0),));
        schedule.run(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn undeclared_access_makes_every_pair_ambiguous() {
        let mut schedule = Schedule::new();
        schedule.add_system("a", |_: &mut World| {});
        schedule.add_system("b", |_: &mut World| {});
        assert_eq!(schedule.ambiguities().len(), 1);
    }

    #[test]
    fn disjoint_declared_access_is_not_ambiguous() {
        let mut registry = ComponentRegistry::new();
        let mut schedule = Schedule::new();
        schedule.add_system("writes_counter", |_: &mut World| {});
        schedule.add_system("writes_other", |_: &mut World| {});
        schedule.set_access(
            "writes_counter",
            SystemAccess::none().write::<Counter>(&mut registry),
        );
        schedule.set_access(
            "writes_other",
            SystemAccess::none().write::<Other>(&mut registry),
        );
        assert!(schedule.ambiguities().is_empty());
    }

    #[test]
    fn two_readers_of_the_same_component_are_not_ambiguous() {
        let mut registry = ComponentRegistry::new();
        let mut schedule = Schedule::new();
        schedule.add_system("reader_a", |_: &mut World| {});
        schedule.add_system("reader_b", |_: &mut World| {});
        schedule.set_access(
            "reader_a",
            SystemAccess::none().read::<Counter>(&mut registry),
        );
        schedule.set_access(
            "reader_b",
            SystemAccess::none().read::<Counter>(&mut registry),
        );
        assert!(schedule.ambiguities().is_empty());
    }

    #[test]
    fn a_writer_and_a_reader_conflict_unless_ordered() {
        let mut registry = ComponentRegistry::new();
        let mut schedule = Schedule::new();
        schedule.add_system("writer", |_: &mut World| {});
        schedule.add_system("reader", |_: &mut World| {});
        schedule.set_access(
            "writer",
            SystemAccess::none().write::<Counter>(&mut registry),
        );
        schedule.set_access(
            "reader",
            SystemAccess::none().read::<Counter>(&mut registry),
        );
        assert_eq!(
            schedule.ambiguities(),
            vec![("writer".to_string(), "reader".to_string())]
        );

        // Declaring the order resolves it.
        schedule.order_after("reader", "writer");
        assert!(schedule.ambiguities().is_empty());
    }

    #[test]
    fn systems_in_different_stages_are_never_ambiguous() {
        let mut registry = ComponentRegistry::new();
        let mut schedule = Schedule::new();
        schedule.add_system_to_stage(Stage::UPDATE, "early", |_: &mut World| {});
        schedule.add_system_to_stage(Stage::PHYSICS, "late", |_: &mut World| {});
        schedule.set_access(
            "early",
            SystemAccess::none().write::<Counter>(&mut registry),
        );
        schedule.set_access("late", SystemAccess::none().write::<Counter>(&mut registry));
        assert!(
            schedule.ambiguities().is_empty(),
            "stages already order these"
        );
    }

    #[test]
    fn run_strict_reports_rather_than_running() {
        let ran = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&ran);
        let mut schedule = Schedule::new();
        schedule.add_system("a", move |_: &mut World| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        schedule.add_system("b", |_: &mut World| {});

        let mut world = World::new();
        assert!(schedule.run_strict(&mut world).is_err());
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "an ambiguous schedule must not run"
        );
    }

    #[test]
    fn the_resolved_order_is_stable_across_runs() {
        let mut schedule = Schedule::new();
        for name in ["e", "d", "c", "b", "a"] {
            schedule.add_system(name, |_: &mut World| {});
        }
        schedule.order_after("a", "c");
        schedule.order_after("c", "e");
        let first = schedule.execution_order().join(",");
        let second = schedule.execution_order().join(",");
        assert_eq!(first, second);
        // The constraints must actually be honoured.
        let order = schedule.execution_order();
        let position = |name: &str| order.iter().position(|entry| *entry == name).unwrap();
        assert!(position("e") < position("c"));
        assert!(position("c") < position("a"));
    }

    #[test]
    fn systems_actually_mutate_the_world() {
        let mut world = World::new();
        let entity = world.spawn((Counter(0),));
        let mut schedule = Schedule::new();
        schedule.add_system("increment", |world: &mut World| {
            for (_entity, (counter,)) in world.query::<(&mut Counter,)>() {
                counter.0 += 1;
            }
        });
        for _ in 0..5 {
            schedule.run(&mut world);
        }
        assert_eq!(world.get::<Counter>(entity).unwrap().0, 5);
    }
}
