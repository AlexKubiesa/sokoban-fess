"""Tests for FESS search traces and the search viewer controller."""

from __future__ import annotations

from sokoban_fess.fess import search_fess
from sokoban_fess.level import LevelState, parse_ascii, replay_solution
from sokoban_fess.levels import get_evaluation_level
from sokoban_fess.visualize import (
    SearchViewer,
    build_solution_frames,
    event_to_state,
    feature_space_graph,
    heat_cells,
    layout_feature_space_graph,
    path_ghost_cells,
    path_indices,
    path_push_segments,
)


def test_search_trace_tiny_map() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    result = search_fess(level, collect_trace=True)
    assert result.solution == "R"
    kinds = [e.kind for e in result.events]
    assert kinds[0] == "start"
    assert "expand" in kinds or "enqueue" in kinds
    assert kinds[-1] == "goal"
    assert result.states_visited >= 1
    assert not result.truncated_trace
    assert all(e.features is not None for e in result.events)
    assert all(e.projected is not None for e in result.events)
    assert result.events[0].features == result.events[0].projected


def test_search_trace_includes_enqueue_or_deadlock() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    result = search_fess(level, collect_trace=True)
    assert result.solution is not None
    kinds = {e.kind for e in result.events}
    assert "start" in kinds
    assert "goal" in kinds
    assert "enqueue" in kinds or "expand" in kinds or "deadlock" in kinds
    from dataclasses import replace

    assert replay_solution(replace(level, solution=result.solution))


def test_already_solved_trace() -> None:
    level = parse_ascii(
        [
            "###",
            "#*#",
            "#@#",
            "###",
        ],
        name="done",
    )
    result = search_fess(level, collect_trace=True)
    assert result.solution == ""
    assert result.events[0].kind == "start"
    assert result.events[-1].kind == "goal"


def test_max_events_truncates_trace_but_still_solves() -> None:
    level = get_evaluation_level(2, height=None, width=None)
    result = search_fess(level, collect_trace=True, max_events=3)
    assert result.solution is not None
    assert len(result.events) <= 3
    assert result.truncated_trace is True


def test_parent_index_forms_path_to_goal() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    result = search_fess(level, collect_trace=True)
    assert result.events[0].parent_index is None
    goal = result.events[-1]
    assert goal.kind == "goal"
    assert goal.parent_index is not None
    chain = path_indices(result.events, len(result.events) - 1)
    assert chain[0] == 0
    assert chain[-1] == len(result.events) - 1
    for i in range(1, len(chain)):
        assert result.events[chain[i]].parent_index == chain[i - 1]


def test_parent_index_on_eval_level() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    result = search_fess(level, collect_trace=True)
    assert result.solution is not None
    goal_i = len(result.events) - 1
    chain = path_indices(result.events, goal_i)
    assert len(chain) >= 2
    assert result.events[chain[0]].kind == "start"
    assert result.events[chain[-1]].kind == "goal"
    expand_i = result.events[goal_i].parent_index
    assert expand_i is not None
    segments = path_push_segments(result.events, chain)
    ghosts = path_ghost_cells(result.events, chain)
    assert isinstance(segments, tuple)
    assert isinstance(ghosts, dict)


def test_fess_macro_push_path_has_corners() -> None:
    from sokoban_fess.visualize import event_push_path, path_push_paths

    level = parse_ascii(
        [
            "######",
            "#@$ .#",
            "#  $ #",
            "# .  #",
            "######",
        ],
        name="macro_corners",
    )
    result = search_fess(level, collect_trace=True, max_states=50_000)
    assert result.solution is not None
    bent = [
        e
        for e in result.events
        if e.push_path is not None and len(e.push_path) >= 3
    ]
    if bent:
        path = event_push_path(bent[0])
        assert path is not None
        assert len(path) >= 3
        rows = {c[0] for c in path}
        cols = {c[1] for c in path}
        assert len(rows) > 1 and len(cols) > 1

    goal_i = next(i for i, e in enumerate(result.events) if e.kind == "goal")
    chain = path_indices(result.events, goal_i)
    paths = path_push_paths(result.events, chain)
    assert all(len(p) >= 2 for p in paths)


def test_viewer_filters_enqueued_by_default() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    result = search_fess(level, collect_trace=True)
    assert any(e.kind == "enqueue" for e in result.events)

    viewer = SearchViewer(level, result)
    assert viewer.state.show_enqueued is False
    assert viewer.n_events() < len(result.events)
    kinds = [viewer.result.events[i].kind for i in viewer.visible_indices()]
    assert "enqueue" not in kinds
    assert "deadlock" not in kinds
    assert "expand" in kinds or "start" in kinds

    # Stepping never lands on an enqueue/deadlock while filtered.
    for i in range(viewer.n_events()):
        viewer.seek(i)
        assert viewer.event.kind not in {"enqueue", "deadlock"}

    # Toggle restores full enqueue timeline and can land on enqueues.
    viewer.seek(0)
    viewer.toggle_show_enqueued()
    assert viewer.state.show_enqueued is True
    assert viewer.n_events() == len(result.events)
    enqueue_i = next(i for i, e in enumerate(result.events) if e.kind == "enqueue")
    viewer.seek(enqueue_i)
    assert viewer.event.kind == "enqueue"

    # Turning the filter back off snaps off the enqueue frame.
    viewer.toggle_show_enqueued()
    assert viewer.state.show_enqueued is False
    assert viewer.event.kind not in {"enqueue", "deadlock"}


def test_viewer_step_and_seek() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    result = search_fess(level, collect_trace=True)
    viewer = SearchViewer(level, result)
    assert viewer.state.playing is False
    assert viewer.state.show_enqueued is False
    assert viewer.phase == "search"
    assert viewer.n_solution_frames() >= 1
    assert viewer.n_total() == viewer.n_events() + viewer.n_solution_frames()

    viewer.step(1)
    assert viewer.state.index == 1
    assert viewer.event_index == viewer.visible_indices()[1]
    viewer.seek(0)
    assert viewer.state.index == 0
    viewer.seek(10_000)
    assert viewer.state.index == viewer.n_total() - 1
    assert viewer.phase == "solution"
    viewer.toggle_play()
    assert viewer.state.playing is True
    viewer.toggle_play()
    assert viewer.state.playing is False

    viewer.seek(0)
    state = event_to_state(level, viewer.event)
    assert isinstance(state, LevelState)
    assert state.player == viewer.event.player
    heat = heat_cells(
        result.events,
        min(2, len(result.events) - 1),
        show_enqueued=False,
    )
    assert isinstance(heat, dict)

    viewer.seek(viewer.n_events() - 1)
    viewer.state.playing = True
    viewer.state.events_per_second = 1.0
    viewer.tick(1.0)
    assert viewer.phase == "solution"
    assert viewer.solution_frame_index == 0
    assert viewer.current_state().player == level.player


def test_feature_space_graph_grows_and_highlights_path() -> None:
    level = get_evaluation_level(0, height=None, width=None)
    result = search_fess(level, collect_trace=True)
    assert result.solution is not None
    assert len(result.events) >= 3

    early = feature_space_graph(result.events, 0)
    assert early.current == result.events[0].features
    assert early.current in early.nodes
    assert early.path_nodes == frozenset({early.current})

    mid = min(5, len(result.events) - 1)
    mid_graph = feature_space_graph(result.events, mid)
    late = feature_space_graph(result.events, len(result.events) - 1)
    assert set(mid_graph.nodes) <= set(late.nodes)
    assert late.current == result.events[-1].features
    assert late.current in late.path_nodes

    chain = path_indices(result.events, len(result.events) - 1)
    for idx in chain:
        cell = result.events[idx].features
        assert cell is not None
        assert cell in late.path_nodes

    rect = (0.0, 0.0, 200.0, 160.0)
    full_positions = layout_feature_space_graph(late, rect, pad=10.0)
    assert set(full_positions) == set(late.nodes)
    for x, y in full_positions.values():
        assert 0.0 <= x <= 200.0
        assert 0.0 <= y <= 160.0

    viewer = SearchViewer(level, result)
    expanded_late = feature_space_graph(
        result.events, len(result.events) - 1, show_enqueued=False
    )
    assert viewer.full_fs_graph().nodes == expanded_late.nodes
    viewer.seek(0)
    pos_start = layout_feature_space_graph(
        viewer.full_fs_graph(), rect, pad=10.0
    )
    viewer.seek(viewer.n_events() - 1)
    pos_end = layout_feature_space_graph(
        viewer.full_fs_graph(), rect, pad=10.0
    )
    assert pos_start == pos_end
    assert viewer.current_fs_graph().current is not None

    # With enqueued moves hidden, FS nodes come only from expanded states.
    assert any(e.kind == "enqueue" for e in result.events)
    viewer.seek(viewer.n_events() - 1)
    assert viewer.state.show_enqueued is False
    filtered = viewer.current_fs_graph()
    all_nodes = feature_space_graph(
        result.events, len(result.events) - 1, show_enqueued=True
    )
    assert set(filtered.nodes) <= set(all_nodes.nodes)
    # First expand should not yet include cells only seen on later enqueues.
    expands = [i for i, e in enumerate(result.events) if e.kind == "expand"]
    if expands:
        g_exp = feature_space_graph(
            result.events, expands[0], show_enqueued=False
        )
        g_all = feature_space_graph(
            result.events, expands[0], show_enqueued=True
        )
        assert set(g_exp.nodes) <= set(g_all.nodes)
        # Enqueue-only cells between start and first expand drop out.
        assert len(g_exp.nodes) <= len(g_all.nodes)


def test_feature_space_regression_uses_true_cell_and_projection_pin() -> None:
    """Level 8's 2nd expand regresses in true FS but pins projected cell."""
    level = get_evaluation_level(7, height=None, width=None)
    assert level.name == "eval_08_boxed_in"
    result = search_fess(level, collect_trace=True)
    expands = [i for i, e in enumerate(result.events) if e.kind == "expand"]
    assert len(expands) >= 2
    first, second = expands[0], expands[1]
    e1, e2 = result.events[first], result.events[second]
    assert e1.features == e1.projected
    assert e2.features != e2.projected  # regression pin

    g1 = feature_space_graph(result.events, first)
    g2 = feature_space_graph(result.events, second)
    assert g1.current == e1.features
    assert g2.current == e2.features
    assert g2.projected == e2.projected
    assert g2.current != g2.projected

    rect = (0.0, 0.0, 200.0, 160.0)
    late = feature_space_graph(result.events, len(result.events) - 1)
    positions = layout_feature_space_graph(late, rect, pad=10.0)
    # Closer-to-goal cells sit further right (FESS goal-direction order).
    from sokoban_fess.fess.features import FeatureCell, goal_direction_key

    a, b = e1.features, e2.features
    assert a is not None and b is not None
    # expand1 improved connectivity (0,1,…); expand2 regresses to (0,2,…).
    assert goal_direction_key(FeatureCell(*a)) < goal_direction_key(FeatureCell(*b))
    assert positions[a][0] > positions[b][0]
    # Distinct positions for every cell.
    assert len({(round(p[0], 3), round(p[1], 3)) for p in positions.values()}) == len(
        positions
    )


def test_build_solution_frames_replays_moves() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny_push",
    )
    frames = build_solution_frames(level, "R")
    assert len(frames) == 2
    assert frames[0].player == (1, 1)
    assert frames[0].boxes == {(1, 2)}
    assert frames[1].success
    assert frames[1].boxes == {(1, 3)}
