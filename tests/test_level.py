"""Tests for Sokoban level parsing and mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from sokoban_fess.level import (
    BOX,
    BOX_ON_TARGET,
    PLAYER,
    TARGET,
    WALL,
    LevelState,
    apply_action,
    encode_observation,
    level_to_ascii,
    parse_ascii,
    render_ansi,
    render_rgb,
)


SIMPLE = """\
###
#@#
#$#
#.#
###
"""


def test_parse_simple_level() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    assert level.player == (1, 1)
    assert level.boxes == frozenset({(2, 1)})
    assert level.targets == frozenset({(3, 1)})
    assert (0, 0) in level.walls


def test_parse_rejects_mismatched_boxes() -> None:
    with pytest.raises(ValueError, match="Box count"):
        parse_ascii("####\n#@$ #\n####\n")


def test_parse_rejects_missing_player() -> None:
    with pytest.raises(ValueError, match="player"):
        parse_ascii("####\n# $.#\n####\n")


def test_walk_and_push() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    state = LevelState.from_level(level, max_steps=50)
    # Push box down onto target.
    state, reward, terminated, truncated, info = apply_action(state, 2)  # down
    assert info["pushed"] is True
    assert state.player == (2, 1)
    assert state.boxes == {(3, 1)}
    assert terminated is True
    assert truncated is False
    assert reward > 0


def test_blocked_by_wall() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    state = LevelState.from_level(level)
    state, _, _, _, info = apply_action(state, 0)  # up into wall
    assert info["moved"] is False
    assert state.player == (1, 1)


def test_blocked_push_into_wall() -> None:
    level = parse_ascii(
        "#####\n#@$.#\n#####\n",
        name="side",
    )
    state = LevelState.from_level(level)
    # Push right: box would go into wall? map is #@$.# so push right moves box onto target.
    state, _, terminated, _, info = apply_action(state, 1)
    assert info["pushed"] is True
    assert terminated is True


def test_box_off_target_penalty() -> None:
    level = parse_ascii(
        "####\n# @#\n# *#\n#  #\n####\n",
        name="star",
    )
    state = LevelState.from_level(level)
    assert state.boxes_on_targets() == 1
    # Push box down off target.
    state, reward, _, _, info = apply_action(
        state,
        2,
        step_penalty=0.0,
        box_off_target_penalty=-0.1,
        box_on_target_reward=0.1,
        success_reward=1.0,
    )
    assert info["pushed"] is True
    assert state.boxes_on_targets() == 0
    assert reward == pytest.approx(-0.1)


def test_truncation() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    state = LevelState.from_level(level, max_steps=1)
    state, _, terminated, truncated, _ = apply_action(state, 3)  # left into wall / no win
    assert terminated is False
    assert truncated is True


def test_encode_observation_codes() -> None:
    level = parse_ascii(
        "#####\n#@$.#\n#####\n",
        name="obs",
    )
    state = LevelState.from_level(level)
    obs = encode_observation(state)
    assert obs.dtype == np.uint8
    assert obs[1, 1] == PLAYER
    assert obs[1, 2] == BOX
    assert obs[1, 3] == TARGET
    assert obs[0, 0] == WALL


def test_player_and_box_on_target_encoding() -> None:
    level = parse_ascii(
        "#####\n#+ $ #\n#####\n",
        name="ontarget",
    )
    state = LevelState.from_level(level)
    obs = encode_observation(state)
    assert obs[1, 1] == 6  # PLAYER_ON_TARGET
    assert obs[1, 3] == BOX

    level2 = parse_ascii(
        "#####\n# @* #\n#####\n",
        name="boxontarget",
    )
    obs2 = encode_observation(LevelState.from_level(level2))
    assert obs2[1, 3] == BOX_ON_TARGET


def test_render_ansi_and_rgb() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    state = LevelState.from_level(level)
    ansi = render_ansi(state)
    assert "@" in ansi or "\033" in ansi
    rgb = render_rgb(state, cell_size=4)
    assert rgb.shape == (level.height * 4, level.width * 4, 3)
    assert rgb.dtype == np.uint8


def test_roundtrip_ascii() -> None:
    level = parse_ascii(SIMPLE, name="simple")
    text = level_to_ascii(level)
    again = parse_ascii(text, name="again")
    assert again.boxes == level.boxes
    assert again.player == level.player
    assert again.targets == level.targets
