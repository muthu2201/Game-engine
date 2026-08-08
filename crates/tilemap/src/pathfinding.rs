//! Grid pathfinding.
//!
//! Two complementary algorithms, because they answer different questions:
//!
//! * [`find_path`] — A*, for "how does *this* NPC get to *that* door". One
//!   agent, one goal, computed on demand.
//! * [`FlowField`] — Dijkstra from a goal outward, for "how do *all* of these
//!   creatures get to the player". Computed once, then every agent reads its
//!   own cell. In the mine, where a dozen creatures share one target, this
//!   turns a dozen searches into one.
//!
//! # Determinism
//!
//! Both walk neighbours in the fixed order of
//! [`IVec2::CARDINALS`](verdant_core_math::IVec2::CARDINALS), and A* breaks
//! equal-cost ties by cell coordinate rather than by insertion order. Two runs
//! over the same map therefore produce the identical path, which matters
//! because NPC movement is simulation state that replays must reproduce.

use crate::map::Tilemap;
use std::collections::BinaryHeap;
use verdant_core_math::IVec2;

/// Cost of one orthogonal step.
///
/// Scaled by ten so the diagonal cost (14) approximates `10 * sqrt(2)` in
/// integers — the standard trick that keeps A* free of floating point.
const STEP_COST: u32 = 10;
/// Cost of one diagonal step.
const DIAGONAL_COST: u32 = 14;

/// How an agent is allowed to move between cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Movement {
    /// Four-way movement.
    #[default]
    Orthogonal,
    /// Eight-way movement, but never cutting a corner between two walls.
    Diagonal,
}

/// A node on the open list.
///
/// `Ord` is written so the `BinaryHeap` (a max-heap) pops the *lowest* total
/// cost, and so equal costs are broken by cell coordinate rather than by
/// whatever order the heap happens to hold them in.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Candidate {
    estimated_total: u32,
    cost_so_far: u32,
    cell: IVec2,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Candidate) -> std::cmp::Ordering {
        other
            .estimated_total
            .cmp(&self.estimated_total)
            // Prefer the node closer to the goal when totals tie: it reaches an
            // answer sooner and, being deterministic, always the same answer.
            .then_with(|| other.cost_so_far.cmp(&self.cost_so_far))
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Candidate) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Finds a path from `start` to `goal`, or `None` when none exists.
///
/// The returned path includes both endpoints. `max_nodes` bounds the search so
/// an unreachable goal on a large map costs a predictable amount rather than
/// scanning every cell — an NPC that cannot find a route should give up and do
/// something else, not stall the frame.
#[must_use]
pub fn find_path(
    map: &Tilemap,
    start: IVec2,
    goal: IVec2,
    movement: Movement,
    max_nodes: usize,
) -> Option<Vec<IVec2>> {
    if !map.is_walkable(start) || !map.is_walkable(goal) {
        return None;
    }
    if start == goal {
        return Some(vec![start]);
    }

    let mut open = BinaryHeap::new();
    // Parent links and best-known cost, keyed by cell.
    let mut came_from: std::collections::HashMap<IVec2, IVec2> = std::collections::HashMap::new();
    let mut best_cost: std::collections::HashMap<IVec2, u32> = std::collections::HashMap::new();

    best_cost.insert(start, 0);
    open.push(Candidate {
        estimated_total: heuristic(start, goal, movement),
        cost_so_far: 0,
        cell: start,
    });

    let mut expanded = 0usize;
    while let Some(current) = open.pop() {
        if current.cell == goal {
            return Some(reconstruct(&came_from, goal));
        }
        // A stale heap entry: a cheaper route to this cell was found after it
        // was pushed. Skipping is cheaper than supporting decrease-key.
        if best_cost
            .get(&current.cell)
            .is_some_and(|best| current.cost_so_far > *best)
        {
            continue;
        }
        expanded += 1;
        if expanded > max_nodes {
            return None;
        }

        for (neighbour, step) in neighbours(map, current.cell, movement) {
            let cost =
                current.cost_so_far + step + u32::from(map.properties_at(neighbour).extra_cost);
            if best_cost.get(&neighbour).is_some_and(|best| cost >= *best) {
                continue;
            }
            best_cost.insert(neighbour, cost);
            came_from.insert(neighbour, current.cell);
            open.push(Candidate {
                estimated_total: cost + heuristic(neighbour, goal, movement),
                cost_so_far: cost,
                cell: neighbour,
            });
        }
    }
    None
}

/// The walkable neighbours of a cell, with the cost of stepping to each.
fn neighbours(map: &Tilemap, cell: IVec2, movement: Movement) -> Vec<(IVec2, u32)> {
    let mut found = Vec::with_capacity(8);
    for offset in IVec2::CARDINALS {
        let neighbour = cell + offset;
        if map.is_walkable(neighbour) {
            found.push((neighbour, STEP_COST));
        }
    }
    if movement == Movement::Diagonal {
        for offset in [
            IVec2::new(1, -1),
            IVec2::new(1, 1),
            IVec2::new(-1, 1),
            IVec2::new(-1, -1),
        ] {
            let neighbour = cell + offset;
            if !map.is_walkable(neighbour) {
                continue;
            }
            // Refuse to squeeze diagonally between two blocked cells: visually
            // the agent would clip through a wall corner, and the collision
            // solver would stop it anyway, leaving it stuck on its own path.
            let horizontal = IVec2::new(cell.x + offset.x, cell.y);
            let vertical = IVec2::new(cell.x, cell.y + offset.y);
            if map.is_walkable(horizontal) && map.is_walkable(vertical) {
                found.push((neighbour, DIAGONAL_COST));
            }
        }
    }
    found
}

/// An admissible distance estimate.
///
/// Must never overestimate, or A* stops being optimal: Manhattan for four-way
/// movement, octile for eight-way.
fn heuristic(from: IVec2, to: IVec2, movement: Movement) -> u32 {
    let dx = (from.x - to.x).unsigned_abs();
    let dy = (from.y - to.y).unsigned_abs();
    match movement {
        Movement::Orthogonal => (dx + dy) * STEP_COST,
        Movement::Diagonal => {
            let diagonal = dx.min(dy);
            let straight = dx.max(dy) - diagonal;
            diagonal * DIAGONAL_COST + straight * STEP_COST
        }
    }
}

/// Walks the parent links back from the goal.
fn reconstruct(came_from: &std::collections::HashMap<IVec2, IVec2>, goal: IVec2) -> Vec<IVec2> {
    let mut path = vec![goal];
    let mut current = goal;
    while let Some(parent) = came_from.get(&current) {
        path.push(*parent);
        current = *parent;
    }
    path.reverse();
    path
}

/// Distance-to-goal for every reachable cell, plus the direction to walk.
///
/// Built with a Dijkstra flood from the goal, so one construction serves any
/// number of agents heading for the same place.
#[derive(Clone, Debug)]
pub struct FlowField {
    width: u32,
    height: u32,
    /// Cost to reach the goal from each cell, `u32::MAX` when unreachable.
    costs: Vec<u32>,
    goal: IVec2,
}

impl FlowField {
    /// Floods outward from `goal` across every walkable cell.
    #[must_use]
    pub fn build(map: &Tilemap, goal: IVec2, movement: Movement) -> FlowField {
        let width = map.width();
        let height = map.height();
        let mut costs = vec![u32::MAX; (width as usize) * (height as usize)];

        let mut field = FlowField {
            width,
            height,
            costs: Vec::new(),
            goal,
        };
        if map.is_walkable(goal) {
            let mut queue = BinaryHeap::new();
            costs[field.index_of(goal)] = 0;
            queue.push(Candidate {
                estimated_total: 0,
                cost_so_far: 0,
                cell: goal,
            });

            while let Some(current) = queue.pop() {
                let index = field.index_of(current.cell);
                if current.cost_so_far > costs[index] {
                    continue;
                }
                for (neighbour, step) in neighbours(map, current.cell, movement) {
                    let cost = current.cost_so_far
                        + step
                        + u32::from(map.properties_at(neighbour).extra_cost);
                    let neighbour_index = field.index_of(neighbour);
                    if cost < costs[neighbour_index] {
                        costs[neighbour_index] = cost;
                        queue.push(Candidate {
                            estimated_total: cost,
                            cost_so_far: cost,
                            cell: neighbour,
                        });
                    }
                }
            }
        }
        field.costs = costs;
        field
    }

    /// Flat index of a cell.
    fn index_of(&self, cell: IVec2) -> usize {
        (cell.y as usize) * (self.width as usize) + (cell.x as usize)
    }

    /// True when `cell` lies inside the field.
    fn contains(&self, cell: IVec2) -> bool {
        cell.x >= 0 && cell.y >= 0 && (cell.x as u32) < self.width && (cell.y as u32) < self.height
    }

    /// The cost of reaching the goal from `cell`, or `None` when unreachable.
    #[must_use]
    pub fn cost(&self, cell: IVec2) -> Option<u32> {
        if !self.contains(cell) {
            return None;
        }
        let cost = self.costs[self.index_of(cell)];
        if cost == u32::MAX {
            None
        } else {
            Some(cost)
        }
    }

    /// The neighbouring cell to step to from `cell`, or `None` at the goal or
    /// somewhere unreachable.
    ///
    /// Ties go to the first neighbour in the fixed cardinal order, which is
    /// what makes a crowd of creatures funnel identically on every run.
    #[must_use]
    pub fn step_from(&self, cell: IVec2) -> Option<IVec2> {
        let current = self.cost(cell)?;
        if current == 0 {
            return None;
        }
        let mut best: Option<(u32, IVec2)> = None;
        for offset in IVec2::NEIGHBOURS {
            let neighbour = cell + offset;
            let Some(cost) = self.cost(neighbour) else {
                continue;
            };
            if cost < current && best.is_none_or(|(best_cost, _)| cost < best_cost) {
                best = Some((cost, neighbour));
            }
        }
        best.map(|(_, cell)| cell)
    }

    /// The cell this field leads to.
    #[must_use]
    pub fn goal(&self) -> IVec2 {
        self.goal
    }

    /// Number of cells that can reach the goal.
    #[must_use]
    pub fn reachable_count(&self) -> usize {
        self.costs.iter().filter(|cost| **cost != u32::MAX).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{TileId, TileLayer, TileProperties, Tilemap};
    use verdant_core_math::fx;

    const GRASS: TileId = TileId(1);
    const WALL: TileId = TileId(2);

    /// Builds a map from ASCII art: `#` is wall, anything else is ground.
    #[allow(clippy::cast_possible_truncation)]
    fn map_from(rows: &[&str]) -> Tilemap {
        let height = u32::try_from(rows.len()).unwrap();
        let width = u32::try_from(rows[0].len()).unwrap();
        let mut map = Tilemap::new(width, height, fx(16));
        map.set_properties(GRASS, TileProperties::GROUND);
        map.set_properties(WALL, TileProperties::WALL);

        let mut ground = TileLayer::new("ground", width, height);
        for (y, row) in rows.iter().enumerate() {
            for (x, glyph) in row.chars().enumerate() {
                let cell = IVec2::new(x as i32, y as i32);
                ground.set(cell, if glyph == '#' { WALL } else { GRASS });
            }
        }
        map.push_layer(ground);
        map
    }

    #[test]
    fn a_straight_path_across_open_ground() {
        let map = map_from(&[".....", ".....", "....."]);
        let path = find_path(
            &map,
            IVec2::new(0, 1),
            IVec2::new(4, 1),
            Movement::Orthogonal,
            1000,
        )
        .expect("open ground is always traversable");
        assert_eq!(path.first(), Some(&IVec2::new(0, 1)));
        assert_eq!(path.last(), Some(&IVec2::new(4, 1)));
        assert_eq!(path.len(), 5, "the shortest route is a straight line");
    }

    #[test]
    fn a_path_routes_around_an_obstacle() {
        let map = map_from(&[".....", "..#..", "....."]);
        let path = find_path(
            &map,
            IVec2::new(0, 1),
            IVec2::new(4, 1),
            Movement::Orthogonal,
            1000,
        )
        .expect("a detour exists");
        assert!(
            !path.contains(&IVec2::new(2, 1)),
            "the path must avoid the wall"
        );
        assert_eq!(path.last(), Some(&IVec2::new(4, 1)));
    }

    #[test]
    fn an_unreachable_goal_returns_none() {
        let map = map_from(&[".#.", ".#.", ".#."]);
        assert!(find_path(
            &map,
            IVec2::new(0, 0),
            IVec2::new(2, 0),
            Movement::Orthogonal,
            1000
        )
        .is_none());
    }

    #[test]
    fn paths_into_or_out_of_a_wall_are_rejected() {
        let map = map_from(&[".#.", "...", "..."]);
        assert!(find_path(
            &map,
            IVec2::new(1, 0),
            IVec2::new(0, 0),
            Movement::Orthogonal,
            100
        )
        .is_none());
        assert!(find_path(
            &map,
            IVec2::new(0, 0),
            IVec2::new(1, 0),
            Movement::Orthogonal,
            100
        )
        .is_none());
    }

    #[test]
    fn a_path_to_the_starting_cell_is_a_single_step() {
        let map = map_from(&["...", "...", "..."]);
        let path = find_path(
            &map,
            IVec2::new(1, 1),
            IVec2::new(1, 1),
            Movement::Orthogonal,
            100,
        )
        .unwrap();
        assert_eq!(path, vec![IVec2::new(1, 1)]);
    }

    #[test]
    fn the_node_budget_bounds_a_hopeless_search() {
        // A large open map with the goal walled off entirely.
        let mut rows = vec![".".repeat(40); 40];
        rows[20] = "#".repeat(40);
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        let map = map_from(&rows);
        assert!(
            find_path(
                &map,
                IVec2::new(0, 0),
                IVec2::new(0, 39),
                Movement::Orthogonal,
                50
            )
            .is_none(),
            "the search must give up rather than scan the map"
        );
    }

    #[test]
    fn diagonal_movement_produces_a_shorter_path() {
        let map = map_from(&["....", "....", "....", "...."]);
        let orthogonal = find_path(
            &map,
            IVec2::ZERO,
            IVec2::new(3, 3),
            Movement::Orthogonal,
            1000,
        )
        .unwrap();
        let diagonal = find_path(
            &map,
            IVec2::ZERO,
            IVec2::new(3, 3),
            Movement::Diagonal,
            1000,
        )
        .unwrap();
        assert!(diagonal.len() < orthogonal.len());
    }

    #[test]
    fn diagonal_movement_refuses_to_cut_a_wall_corner() {
        // Moving from (0,0) to (1,1) diagonally would squeeze between two walls.
        let map = map_from(&[".#.", "#..", "..."]);
        let path = find_path(
            &map,
            IVec2::ZERO,
            IVec2::new(1, 1),
            Movement::Diagonal,
            1000,
        );
        assert!(path.is_none(), "the corner is sealed by two walls");
    }

    #[test]
    fn the_pathfinder_prefers_cheaper_ground() {
        let mut map = map_from(&["...", "...", "..."]);
        // Make the direct middle row expensive; the route should detour.
        let costly = TileId(3);
        map.set_properties(
            costly,
            TileProperties {
                extra_cost: 100,
                ..TileProperties::GROUND
            },
        );
        map.layer_mut("ground")
            .unwrap()
            .set(IVec2::new(1, 1), costly);

        let path = find_path(
            &map,
            IVec2::new(0, 1),
            IVec2::new(2, 1),
            Movement::Orthogonal,
            1000,
        )
        .unwrap();
        assert!(
            !path.contains(&IVec2::new(1, 1)),
            "the expensive cell should be avoided"
        );
    }

    #[test]
    fn determinism_the_same_query_returns_the_same_path() {
        let map = map_from(&[
            "..........",
            ".##..##...",
            "..........",
            "..##...##.",
            "..........",
        ]);
        let run = || {
            find_path(
                &map,
                IVec2::ZERO,
                IVec2::new(9, 4),
                Movement::Diagonal,
                10_000,
            )
        };
        let first = run().expect("a route exists");
        for _ in 0..20 {
            assert_eq!(run().unwrap(), first, "A* must not vary between runs");
        }
    }

    #[test]
    fn a_flow_field_leads_every_cell_to_the_goal() {
        let map = map_from(&["....", ".##.", "....", "...."]);
        let field = FlowField::build(&map, IVec2::new(0, 0), Movement::Orthogonal);

        assert_eq!(field.cost(IVec2::new(0, 0)), Some(0));
        assert_eq!(field.goal(), IVec2::new(0, 0));

        // Walk downhill from a far corner; it must terminate at the goal.
        let mut cell = IVec2::new(3, 3);
        let mut steps = 0;
        while let Some(next) = field.step_from(cell) {
            cell = next;
            steps += 1;
            assert!(steps < 100, "the descent should not loop");
        }
        assert_eq!(cell, IVec2::new(0, 0));
    }

    #[test]
    fn unreachable_cells_have_no_cost_or_step() {
        let map = map_from(&[".#.", ".#.", ".#."]);
        let field = FlowField::build(&map, IVec2::new(0, 0), Movement::Orthogonal);
        assert!(field.cost(IVec2::new(2, 0)).is_none());
        assert!(field.step_from(IVec2::new(2, 0)).is_none());
        assert!(
            field.cost(IVec2::new(1, 0)).is_none(),
            "the wall itself is unreachable"
        );
        assert_eq!(
            field.reachable_count(),
            3,
            "only the left column is reachable"
        );
    }

    #[test]
    fn a_flow_field_with_an_unreachable_goal_is_empty() {
        let map = map_from(&["...", ".#.", "..."]);
        let field = FlowField::build(&map, IVec2::new(1, 1), Movement::Orthogonal);
        assert_eq!(field.reachable_count(), 0);
        assert!(field.step_from(IVec2::new(0, 0)).is_none());
    }

    #[test]
    fn flow_field_and_a_star_agree_on_reachability() {
        let map = map_from(&["......", ".####.", ".#....", ".#.##.", "......"]);
        let goal = IVec2::new(0, 0);
        let field = FlowField::build(&map, goal, Movement::Orthogonal);
        for y in 0..5i32 {
            for x in 0..6i32 {
                let cell = IVec2::new(x, y);
                if !map.is_walkable(cell) {
                    continue;
                }
                let by_search = find_path(&map, cell, goal, Movement::Orthogonal, 10_000).is_some();
                assert_eq!(
                    field.cost(cell).is_some(),
                    by_search,
                    "{cell:?} disagreed between the two algorithms"
                );
            }
        }
    }
}
