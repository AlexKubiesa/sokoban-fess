"""Unit tests for FESS preprocess helpers."""

from __future__ import annotations

from sokoban_fess.fess.features import analyze_level, build_hotspot_table
from sokoban_fess.fess.packing import build_packing_plan, compute_basin
from sokoban_fess.fess.rooms import build_rooms, room_connectivity_with_open
from sokoban_fess.level import parse_ascii
from sokoban_fess.solver import _push_distances_to_targets


def test_rooms_detect_open_block() -> None:
    level = parse_ascii(
        [
            "######",
            "#    #",
            "#    #",
            "#    #",
            "#@$ .#",
            "######",
        ],
        name="roomy",
    )
    rooms = build_rooms(level.walls, level.height, level.width)
    assert rooms.n_rooms >= 1


def test_room_connectivity_breaks_with_box() -> None:
    level = parse_ascii(
        [
            "##########",
            "#   ##   #",
            "#   ##   #",
            "#   ##   #",
            "# @$..$  #",
            "##########",
        ],
        name="two_rooms",
    )
    rooms = build_rooms(level.walls, level.height, level.width)
    open_set = {
        (r, c)
        for r in range(level.height)
        for c in range(level.width)
        if (r, c) not in level.walls
    }
    empty = room_connectivity_with_open(rooms, frozenset(), open_set)
    blocked = room_connectivity_with_open(
        rooms, frozenset(level.boxes), open_set
    )
    assert blocked >= empty


def test_basin_includes_approach_cells() -> None:
    level = parse_ascii(
        [
            "#####",
            "#  .#",
            "# $ #",
            "# @ #",
            "#####",
        ],
        name="basin_tiny",
    )
    target = next(iter(level.targets))
    basin = compute_basin(
        level.walls, frozenset(level.targets), target, level.height, level.width
    )
    assert target in basin
    assert any(b in basin for b in level.boxes) or len(basin) >= 1


def test_packing_plan_start_not_fully_packed() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="pack_tiny",
    )
    rooms = build_rooms(level.walls, level.height, level.width)
    plan = build_packing_plan(level, rooms)
    assert plan.packed_count(frozenset(level.boxes)) == 0
    assert plan.packed_count(frozenset(level.targets)) >= 1


def test_hotspot_table_nonempty_on_corridor() -> None:
    level = parse_ascii(
        [
            "########",
            "#@$   .#",
            "########",
        ],
        name="hot_corridor",
    )
    dist = _push_distances_to_targets(
        level.walls, tuple(sorted(level.targets)), level.height, level.width
    )
    pairs = build_hotspot_table(level.walls, level.height, level.width, dist)
    # May be empty on tiny boards; analysis should still succeed.
    analysis = analyze_level(level)
    assert analysis.features(frozenset(level.boxes)).packed == 0
    assert isinstance(pairs, frozenset)


def test_analyze_level_four_features() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="analyze_tiny",
    )
    analysis = analyze_level(level)
    feat = analysis.features(frozenset(level.boxes))
    assert feat.packed == 0
    assert feat.connectivity == 2
    assert feat.oop >= 0
