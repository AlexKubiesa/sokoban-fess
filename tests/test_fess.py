"""Tests for the FESS Sokoban solver (macro moves)."""

from __future__ import annotations

from dataclasses import replace

from sokoban_fess.fess import (
    FeatureCell,
    feature_coordinates,
    find_solution_fess,
    search_fess,
)
from sokoban_fess.fess.search import _FessSearcher
from sokoban_fess.level import parse_ascii, replay_solution
from sokoban_fess.levels import get_evaluation_level


def test_feature_coordinates() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="feat_tiny",
    )
    start = feature_coordinates(level, frozenset(level.boxes))
    assert start.packed == 0
    assert start.connectivity == 2
    assert start.room_connectivity >= 0
    assert start.oop >= 0

    packed = feature_coordinates(level, frozenset(level.targets))
    assert packed.packed >= 1
    assert packed.connectivity == 1


def test_fess_tiny_map() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    solution = find_solution_fess(level)
    assert solution == "R"
    assert replay_solution(replace(level, solution=solution))


def test_fess_already_solved() -> None:
    level = parse_ascii(
        [
            "###",
            "#*#",
            "#@#",
            "###",
        ],
        name="done",
    )
    assert find_solution_fess(level) == ""


def test_fess_wall_stuck_is_unsolvable() -> None:
    level = parse_ascii(
        [
            "#######",
            "#     #",
            "#$ @  #",
            "#    .#",
            "#     #",
            "#######",
        ],
        name="wall_stuck",
    )
    assert find_solution_fess(level) is None


def test_fess_eval_01() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    solution = find_solution_fess(level)
    assert solution is not None
    assert replay_solution(replace(level, solution=solution))


def test_fess_macro_multi_cell() -> None:
    """A corridor exposes macros that push one box several cells in one step."""
    level = parse_ascii(
        [
            "########",
            "#@$   .#",
            "########",
        ],
        name="macro_tunnel",
    )
    searcher = _FessSearcher(
        level,
        max_states=10_000,
        collect_trace=False,
        max_events=0,
    )
    macros = searcher.legal_macros(frozenset(level.boxes), level.player)
    assert any(m.push_count >= 2 for m in macros)
    assert any(
        abs(m.push_from[0] - m.push_to[0]) + abs(m.push_from[1] - m.push_to[1]) > 1
        for m in macros
    )

    result = search_fess(level, collect_trace=True)
    assert result.solution is not None
    assert replay_solution(replace(level, solution=result.solution))
    assert any(
        e.push_path is not None and len(e.push_path) > 2
        for e in result.events
        if e.kind in {"enqueue", "expand", "goal"}
    )
    # Path cells are orthogonal steps (no diagonal shortcuts).
    for e in result.events:
        if e.push_path is None or len(e.push_path) < 2:
            continue
        for a, b in zip(e.push_path, e.push_path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_fess_trace_events() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_trace",
    )
    result = search_fess(level, collect_trace=True)
    assert result.algorithm == "fess"
    assert result.solution == "R"
    assert result.stop_reason == "goal"
    kinds = {e.kind for e in result.events}
    assert "start" in kinds
    assert "goal" in kinds
    assert any(e.kind in {"expand", "enqueue"} for e in result.events)


def test_fess_trace_records_advisors() -> None:
    """Packing the sole box onto its target is packing-advisor endorsed."""
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_advisors",
    )
    result = search_fess(level, collect_trace=True)
    push_events = [
        e for e in result.events if e.kind in {"enqueue", "expand", "goal"}
    ]
    assert push_events
    assert any("packing" in e.advisors for e in push_events)
    endorsed = [e for e in push_events if e.advisors]
    assert all(e.weight >= 0 for e in endorsed)
    # Advisor macros add weight 0 from the root (CoG best setting).
    assert any(e.kind == "enqueue" and e.weight == 0 and e.advisors for e in endorsed)


def test_fess_respects_max_states() -> None:
    level = get_evaluation_level(11, height=None, width=None)
    assert level.name == "eval_12_factory"
    # Macros solve this in dozens of states; use a tiny budget to force a cap.
    result = search_fess(level, max_states=3, collect_trace=True)
    assert result.solution is None
    assert result.stop_reason == "capped"
    assert result.events[-1].kind == "capped"
    assert result.states_visited <= 10


def test_fess_respects_max_time() -> None:
    level = get_evaluation_level(11, height=None, width=None)
    result = search_fess(level, max_time=0.0, collect_trace=True)
    assert result.solution is None
    assert result.stop_reason == "timeout"
    assert result.events[-1].kind == "timeout"


def test_feature_cell_is_four_dimensional() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="feat4",
    )
    feat = feature_coordinates(level, frozenset(level.boxes))
    assert isinstance(feat, FeatureCell)
    assert len(feat) == 4
