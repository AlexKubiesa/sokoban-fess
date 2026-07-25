"""Tests for pattern deadlocks and packing-plan FESS."""

from __future__ import annotations

from sokoban_fess.fess.deadlocks import has_2x2_deadlock, has_pattern_deadlock
from sokoban_fess.fess.features import analyze_level, goal_direction_key
from sokoban_fess.fess.packing import build_packing_plan
from sokoban_fess.fess.rooms import build_rooms
from sokoban_fess.fess.search import _FessSearcher
from sokoban_fess.level import parse_ascii


def test_2x2_deadlock_detected() -> None:
    walls = frozenset(
        {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
        }
    )
    # 2×2 of boxes in the open 2×2, none on targets.
    boxes = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
    targets = frozenset({(1, 1)})  # only one target — still off-target boxes freeze
    assert has_2x2_deadlock(boxes, walls, targets)


def test_corner_deadlock_in_pattern_check() -> None:
    level = parse_ascii(
        [
            "####",
            "#$ #",
            "#@.#",
            "####",
        ],
        name="corner_dead",
    )
    # Box sits in the upper-left inner corner, not on a target.
    assert has_pattern_deadlock(
        frozenset(level.boxes),
        level.walls,
        frozenset(level.targets),
        level.height,
        level.width,
    )


def test_packing_plan_has_ok_zones() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="pack_ok",
    )
    rooms = build_rooms(level.walls, level.height, level.width)
    plan = build_packing_plan(level, rooms)
    assert plan.packed_count(frozenset(level.boxes)) == 0
    assert len(plan.ok_zones) == len(plan.order) + 1
    assert plan.oop_count(frozenset(level.boxes)) >= 0


def test_best_ancestor_projection() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="proj",
    )
    searcher = _FessSearcher(
        level, max_states=100, collect_trace=False, max_events=0
    )
    start = searcher.features(frozenset(level.boxes))
    root = searcher._add_node(
        frozenset(level.boxes),
        level.player,
        0,
        0,
        start,
        start,
        None,
        None,
    )
    assert root is not None
    # Artificial worse child features → pin to ancestor cell.
    worse = type(start)(
        packed=start.packed,
        connectivity=start.connectivity + 1,
        room_connectivity=start.room_connectivity,
        oop=start.oop + 1,
    )
    assert goal_direction_key(worse) > goal_direction_key(start)
    projected = searcher.project_cell(root, worse)
    assert projected == start
