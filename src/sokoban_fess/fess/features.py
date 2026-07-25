"""Feature space coordinates and level preprocess (CoG §III-B).

Feature space is 4-dimensional:
  packed, connectivity, room_connectivity, oop

Hotspots and mobility are computed for advisors / tie-breaks but are not
feature-space axes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

from sokoban_fess.level import ACTION_DELTAS, ACTIONS, Level
from sokoban_fess.solver import Boxes, Cell, _INF, _is_open, _push_distances_to_targets

from sokoban_fess.fess.deadlocks import has_pattern_deadlock
from sokoban_fess.fess.packing import PackingPlan, build_packing_plan
from sokoban_fess.fess.rooms import RoomLayout, build_rooms, room_connectivity_with_open


class FeatureCell(NamedTuple):
    """One cell in FESS feature space (4-D CoG Sokoban configuration)."""

    packed: int
    connectivity: int
    room_connectivity: int
    oop: int


def connectivity_count(level: Level, boxes: Boxes) -> int:
    """Number of connected components among free (non-wall, non-box) cells."""
    remaining: set[Cell] = set()
    for row in range(level.height):
        for col in range(level.width):
            cell = (row, col)
            if cell not in level.walls and cell not in boxes:
                remaining.add(cell)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            pos = stack.pop()
            row, col = pos
            for nxt in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return components


def _target_cells(dist_to_target: dict[Cell, list[int]]) -> frozenset[Cell]:
    return frozenset(
        cell
        for cell, dists in dist_to_target.items()
        if any(d == 0 for d in dists)
    )


def _push_reachable(
    walls: Boxes, start: Cell, height: int, width: int
) -> set[Cell]:
    if not _is_open(start, walls, height, width):
        return set()
    seen = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        curr = queue.popleft()
        c_row, c_col = curr
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            dest = (c_row + d_row, c_col + d_col)
            stand = (c_row - d_row, c_col - d_col)
            if dest in seen:
                continue
            if not _is_open(dest, walls, height, width):
                continue
            if not _is_open(stand, walls, height, width):
                continue
            seen.add(dest)
            queue.append(dest)
    return seen


def build_hotspot_table(
    walls: Boxes,
    height: int,
    width: int,
    dist_to_target: dict[Cell, list[int]],
    *,
    max_seconds: float = 2.0,
) -> frozenset[tuple[Cell, Cell]]:
    """Pairs (X, Y) where Y is a hotspot square for a box on X.

    Aborts early if the pairwise scan exceeds ``max_seconds``.
    """
    deadline = time.perf_counter() + max_seconds
    # Only cells that can push toward some target matter.
    free = [
        cell
        for cell, dists in dist_to_target.items()
        if any(d < _INF for d in dists)
    ]
    targets = _target_cells(dist_to_target)
    if not targets or not free:
        return frozenset()

    base_reach: dict[Cell, int] = {}
    for x in free:
        reach = _push_reachable(walls, x, height, width) & targets
        base_reach[x] = len(reach)

    pairs: set[tuple[Cell, Cell]] = set()
    for yi, y in enumerate(free):
        if time.perf_counter() >= deadline:
            break
        # Skip target squares as blockers.
        if y in targets:
            continue
        blocked_walls = walls | {y}
        for x in free:
            if x == y or x in targets:
                continue
            if base_reach.get(x, 0) == 0:
                continue
            reach = _push_reachable(blocked_walls, x, height, width) & targets
            if len(reach) < base_reach[x]:
                pairs.add((x, y))
    return frozenset(pairs)


@dataclass(frozen=True)
class LevelAnalysis:
    """Static preprocess shared across a FESS run on one level."""

    level: Level
    rooms: RoomLayout
    packing: PackingPlan
    open_set: frozenset[Cell]
    dist_to_target: dict[Cell, list[int]]
    hotspot_pairs: frozenset[tuple[Cell, Cell]]

    def features(self, boxes: Boxes) -> FeatureCell:
        packed = self.packing.packed_count(boxes)
        conn = connectivity_count(self.level, boxes)
        if self.rooms.edges:
            room_conn = room_connectivity_with_open(
                self.rooms, boxes, set(self.open_set)
            )
        else:
            room_conn = 0
        oop = self.packing.oop_count(boxes)
        return FeatureCell(
            packed=packed,
            connectivity=conn,
            room_connectivity=room_conn,
            oop=oop,
        )

    def hotspot_count(self, boxes: Boxes) -> int:
        """Number of hotspot relations among current box pairs."""
        box_list = list(boxes)
        count = 0
        box_set = set(boxes)
        for x in box_list:
            for y in box_set:
                if x != y and (x, y) in self.hotspot_pairs:
                    count += 1
        return count

    def mobility(self, boxes: Boxes, player: Cell) -> int:
        """Approximate mobility: reachable box sides the player can stand on."""
        seen: set[Cell] = set()
        stack = [player]
        while stack:
            pos = stack.pop()
            if pos in seen:
                continue
            r, c = pos
            if not (0 <= r < self.level.height and 0 <= c < self.level.width):
                continue
            if pos in self.level.walls or pos in boxes:
                continue
            seen.add(pos)
            stack.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        mob = 0
        for box in boxes:
            br, bc = box
            for stand in ((br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)):
                if stand in seen:
                    mob += 1
        return mob

    def is_static_dead_cell(self, cell: Cell) -> bool:
        if cell in self.level.targets:
            return False
        distances = self.dist_to_target.get(cell)
        return distances is None or all(d >= _INF for d in distances)

    def has_deadlock(self, boxes: Boxes) -> bool:
        return has_pattern_deadlock(
            boxes,
            self.level.walls,
            frozenset(self.level.targets),
            self.level.height,
            self.level.width,
            dist_to_target=self.dist_to_target,
        )


_ANALYSIS_CACHE: dict[int, LevelAnalysis] = {}


def analyze_level(level: Level, *, build_hotspots: bool = True) -> LevelAnalysis:
    """Run FESS preprocess for ``level`` (rooms, packing plan, hotspots)."""
    open_set = frozenset(
        (r, c)
        for r in range(level.height)
        for c in range(level.width)
        if (r, c) not in level.walls
    )
    rooms = build_rooms(level.walls, level.height, level.width)
    packing = build_packing_plan(level, rooms)
    targets = tuple(sorted(level.targets))
    dist = _push_distances_to_targets(
        level.walls, targets, level.height, level.width
    )
    if build_hotspots:
        hotspots = build_hotspot_table(
            level.walls, level.height, level.width, dist
        )
    else:
        hotspots = frozenset()
    return LevelAnalysis(
        level=level,
        rooms=rooms,
        packing=packing,
        open_set=open_set,
        dist_to_target=dist,
        hotspot_pairs=hotspots,
    )


def get_analysis(level: Level) -> LevelAnalysis:
    """Cached ``analyze_level`` keyed by ``id(level)`` (immutable levels)."""
    key = id(level)
    cached = _ANALYSIS_CACHE.get(key)
    if cached is not None and cached.level is level:
        return cached
    analysis = analyze_level(level)
    if len(_ANALYSIS_CACHE) > 128:
        _ANALYSIS_CACHE.clear()
    _ANALYSIS_CACHE[key] = analysis
    return analysis


def feature_coordinates(level: Level, boxes: Boxes) -> FeatureCell:
    """True feature coordinates for a box placement (public helper / UI)."""
    return get_analysis(level).features(boxes)


def goal_direction_key(feat: FeatureCell) -> tuple[int, int, int, int]:
    """Lex key for progress toward the goal (smaller is better)."""
    return (feat.oop, -feat.packed, feat.connectivity, feat.room_connectivity)


def is_feature_progress(parent: FeatureCell, child: FeatureCell) -> bool:
    """True if ``child`` is closer to the projected goal than ``parent``."""
    return goal_direction_key(child) < goal_direction_key(parent)
