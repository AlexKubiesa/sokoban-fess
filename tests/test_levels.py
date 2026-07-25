"""Tests for the fixed evaluation suite."""

from __future__ import annotations

import pytest

from sokoban_fess.level import DEFAULT_HEIGHT, DEFAULT_WIDTH, validate_level
from sokoban_fess.levels import (
    evaluation_level_count,
    get_evaluation_level,
    load_evaluation_levels,
)


def test_evaluation_count() -> None:
    assert evaluation_level_count() == 30
    levels = load_evaluation_levels()
    assert len(levels) == 30


def test_all_evaluation_levels_valid() -> None:
    levels = load_evaluation_levels()
    names = [lvl.name for lvl in levels]
    assert len(names) == len(set(names))
    for level in levels:
        validate_level(level)
        assert level.height <= DEFAULT_HEIGHT
        assert level.width <= DEFAULT_WIDTH
        assert len(level.boxes) == len(level.targets) >= 1
        assert level.difficulty >= 1


def test_padded_evaluation_levels() -> None:
    levels = load_evaluation_levels(height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH)
    assert levels
    for level in levels:
        validate_level(level)
        assert level.height == DEFAULT_HEIGHT
        assert level.width == DEFAULT_WIDTH


def test_natural_size_evaluation_levels() -> None:
    levels = load_evaluation_levels()
    assert levels
    assert any(lvl.height < DEFAULT_HEIGHT or lvl.width < DEFAULT_WIDTH for lvl in levels)
    for level in levels:
        validate_level(level)
        assert level.height <= DEFAULT_HEIGHT
        assert level.width <= DEFAULT_WIDTH


def test_get_evaluation_level_by_index() -> None:
    level = get_evaluation_level(0)
    assert level.name.startswith("eval_01")
    with pytest.raises(IndexError):
        get_evaluation_level(30)


def test_known_solution_replays() -> None:
    from sokoban_fess.level import replay_solution

    levels = load_evaluation_levels()
    with_solutions = [lvl for lvl in levels if lvl.solution]
    assert with_solutions, "expected at least one certified eval level"
    for level in with_solutions:
        assert replay_solution(level) is True
