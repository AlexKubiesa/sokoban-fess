"""Tests for the solver evaluation benchmark."""

from __future__ import annotations

from sokoban_fess.benchmark import (
    classify_outcome,
    count_pushes,
    format_report,
    report_to_dict,
    run_benchmark,
    run_level,
    verify_solution,
)
from sokoban_fess.fess import search_fess
from sokoban_fess.level import parse_ascii
from sokoban_fess.levels import get_evaluation_level


def test_classify_outcome() -> None:
    assert (
        classify_outcome(solution="R", stop_reason="goal", verified=True)
        == "solved"
    )
    assert (
        classify_outcome(solution="R", stop_reason="goal", verified=False)
        == "invalid"
    )
    assert (
        classify_outcome(solution=None, stop_reason="capped", verified=False)
        == "capped"
    )
    assert (
        classify_outcome(solution=None, stop_reason="timeout", verified=False)
        == "timeout"
    )
    assert (
        classify_outcome(solution=None, stop_reason="exhausted", verified=False)
        == "exhausted"
    )


def test_verify_and_count_pushes_tiny() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    assert verify_solution(level, "R")
    assert count_pushes(level, "R") == 1
    assert count_pushes(level, "") == 0


def test_verify_empty_solution_when_already_solved() -> None:
    level = parse_ascii(
        [
            "###",
            "#*#",
            "#@#",
            "###",
        ],
        name="done",
    )
    assert verify_solution(level, "")


def test_run_level_eval_01() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    result = run_level(level, index=0)
    assert result.status == "solved"
    assert result.name == "eval_01_push"
    assert result.moves is not None and result.moves >= 1
    assert result.pushes is not None and result.pushes >= 1
    assert result.states_visited >= 1
    assert result.time_s >= 0.0


def test_run_benchmark_subset_and_report() -> None:
    levels = [
        get_evaluation_level(0, height=None, width=None),
        get_evaluation_level(1, height=None, width=None),
    ]
    report = run_benchmark(levels, max_states=50_000, max_time=5.0)
    assert report.n_levels == 2
    assert report.n_solved == 2
    assert report.solve_rate == 1.0
    assert report.max_time == 5.0
    text = format_report(report)
    assert "solved 2/2" in text
    assert "max_time=" in text
    assert "By difficulty:" in text
    assert "FESS" in text
    payload = report_to_dict(report)
    assert payload["n_solved"] == 2
    assert payload["n_timeout"] == 0
    assert "solver" not in payload
    assert len(payload["results"]) == 2


def test_run_benchmark_progress() -> None:
    import io

    levels = [get_evaluation_level(0, height=None, width=None)]
    buf = io.StringIO()
    report = run_benchmark(levels, max_time=5.0, progress=buf)
    assert report.n_solved == 1
    text = buf.getvalue()
    assert "[1/1] eval_01_push ..." in text
    assert "solved" in text


def test_run_level_respects_max_states_cap() -> None:
    level = get_evaluation_level(11, height=None, width=None)
    assert level.name == "eval_12_factory"
    # Macros solve this in dozens of states; use a tiny budget to force a cap.
    result = run_level(level, index=11, max_states=3, max_time=None)
    assert result.status == "capped"
    assert result.moves is None
    assert result.states_visited <= 10


def test_search_respects_max_time() -> None:
    level = get_evaluation_level(11, height=None, width=None)
    assert level.name == "eval_12_factory"
    result = search_fess(level, max_states=500_000, max_time=0.0, collect_trace=True)
    assert result.solution is None
    assert result.stop_reason == "timeout"
    assert result.events[-1].kind == "timeout"


def test_run_level_respects_max_time() -> None:
    level = get_evaluation_level(11, height=None, width=None)
    result = run_level(level, index=11, max_states=500_000, max_time=0.0)
    assert result.status == "timeout"
    assert result.moves is None
