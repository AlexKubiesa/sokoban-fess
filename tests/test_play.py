"""Tests for the interactive evaluation player (no display required)."""

from __future__ import annotations

import pytest

from sokoban_fess.fess import feature_coordinates
from sokoban_fess.play import (
    ACTION_FOR_COMMAND,
    Command,
    EvaluationPlayer,
    Screen,
    map_key_to_command,
)


def test_map_key_to_command_wasd_and_arrows() -> None:
    assert map_key_to_command("w") is Command.MOVE_UP
    assert map_key_to_command("UP") is Command.MOVE_UP
    assert map_key_to_command("d") is Command.MOVE_RIGHT
    assert map_key_to_command("right") is Command.MOVE_RIGHT
    assert map_key_to_command("s") is Command.MOVE_DOWN
    assert map_key_to_command("down") is Command.MOVE_DOWN
    assert map_key_to_command("a") is Command.MOVE_LEFT
    assert map_key_to_command("left") is Command.MOVE_LEFT
    assert map_key_to_command("u") is Command.UNDO
    assert map_key_to_command("r") is Command.RESTART
    assert ACTION_FOR_COMMAND[Command.MOVE_UP] == 0
    assert ACTION_FOR_COMMAND[Command.MOVE_RIGHT] == 1
    assert ACTION_FOR_COMMAND[Command.MOVE_DOWN] == 2
    assert ACTION_FOR_COMMAND[Command.MOVE_LEFT] == 3


def test_picker_bounds_and_selection() -> None:
    player = EvaluationPlayer(visible_rows=5)
    assert player.state.screen is Screen.PICKER
    assert player.state.level_index == 0

    player.handle_command(Command.MOVE_UP)
    assert player.state.level_index == 0

    for _ in range(100):
        player.handle_command(Command.MOVE_DOWN)
    assert player.state.level_index == player.level_count - 1
    assert player.state.scroll_offset == player.level_count - player.visible_rows

    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.MODE_SELECT
    assert player.state.level_index == player.level_count - 1
    assert player.state.level_name
    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.PLAYING
    player.close()


def test_mode_select_play_and_solve() -> None:
    player = EvaluationPlayer()
    player.open_mode_select(0)
    assert player.state.screen is Screen.MODE_SELECT
    assert player.state.mode_option == 0

    player.handle_command(Command.MOVE_DOWN)
    assert player.state.mode_option == 1
    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.SEARCH
    assert player.search_viewer is not None
    assert player.search_viewer.n_events() >= 1
    assert player.search_viewer.result.algorithm == "fess"

    viewer = player.search_viewer
    viewer.step(1)
    assert viewer.state.index == 1
    player.handle_command(Command.RESTART)
    assert player.state.screen is Screen.SEARCH
    assert player.search_viewer is not None
    assert player.search_viewer is not viewer
    assert player.search_viewer.state.index == 0
    assert player.search_viewer.state.playing is False

    player.handle_command(Command.BACK)
    assert player.state.screen is Screen.MODE_SELECT
    # open_mode_select resets to Play; Select starts the level.
    assert player.state.mode_option == 0
    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.PLAYING
    player.close()


def test_mode_select_shows_features() -> None:
    player = EvaluationPlayer()
    player.open_mode_select(0)
    player.handle_command(Command.MOVE_DOWN)
    assert player.state.mode_option == 1
    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.SEARCH
    assert player.search_viewer is not None
    assert player.search_viewer.result.algorithm == "fess"
    viewer = player.search_viewer
    feat = feature_coordinates(viewer.level, frozenset(viewer.current_state().boxes))
    assert feat.packed >= 0
    assert feat.connectivity >= 1
    assert feat.room_connectivity >= 0
    assert feat.oop >= 0
    player.close()


def test_restart_resets_progress() -> None:
    player = EvaluationPlayer()
    player.start_level(0)
    # Move down once on eval_01 (push toward target, may complete).
    # Use a level and force a move then restart.
    player.start_level(2)  # eval_03 needs more than one move
    before_name = player.state.level_name
    player.handle_command(Command.MOVE_DOWN)
    steps_after = player.state.steps
    assert steps_after >= 1
    player.handle_command(Command.RESTART)
    assert player.state.screen is Screen.PLAYING
    assert player.state.level_name == before_name
    assert player.state.steps == 0
    assert player.state.boxes_on_targets == 0 or player.env.state is not None
    player.close()


def test_invalid_moves_do_not_increment_steps() -> None:
    player = EvaluationPlayer()
    player.start_level(0)  # eval_01: #@$ .# — wall left of player
    assert player.state.steps == 0
    player.handle_command(Command.MOVE_LEFT)
    assert player.state.steps == 0
    assert player.state.screen is Screen.PLAYING
    player.handle_command(Command.MOVE_RIGHT)
    assert player.state.steps == 1
    player.close()


def test_undo_restores_previous_move() -> None:
    player = EvaluationPlayer()
    player.start_level(0)  # eval_01: RR solves
    player.handle_command(Command.UNDO)
    assert player.state.steps == 0
    assert player.level_state is not None
    start_player = player.level_state.player
    start_boxes = set(player.level_state.boxes)

    player.handle_command(Command.MOVE_RIGHT)
    assert player.state.steps == 1
    assert player.level_state.player != start_player

    player.handle_command(Command.UNDO)
    assert player.state.screen is Screen.PLAYING
    assert player.state.steps == 0
    assert player.level_state.player == start_player
    assert player.level_state.boxes == start_boxes
    player.close()


def test_undo_after_clear_returns_to_playing() -> None:
    player = EvaluationPlayer()
    player.start_level(0)
    player.handle_command(Command.MOVE_RIGHT)
    player.handle_command(Command.MOVE_RIGHT)
    assert player.state.screen is Screen.COMPLETE
    assert player.state.success is True

    player.handle_command(Command.UNDO)
    assert player.state.screen is Screen.PLAYING
    assert player.state.success is False
    assert player.state.steps == 1
    player.close()


def test_complete_first_level_and_advance() -> None:
    player = EvaluationPlayer()
    player.start_level(0)  # eval_01_push — solution is Right, Right
    player.handle_command(Command.MOVE_RIGHT)
    player.handle_command(Command.MOVE_RIGHT)
    assert player.state.screen is Screen.COMPLETE
    assert player.state.success is True
    assert player.next_level_index() == 1

    player.handle_command(Command.SELECT)
    assert player.state.screen is Screen.MODE_SELECT
    assert player.state.level_index == 1
    player.close()


def test_final_level_returns_to_picker() -> None:
    player = EvaluationPlayer()
    last = player.level_count - 1
    player.state.level_index = last
    player.state.screen = Screen.COMPLETE
    player.state.success = True

    advanced = player.advance_to_next_level()
    assert advanced is False
    assert player.state.screen is Screen.PICKER
    assert player.state.level_index == last
    assert "Suite complete" in player.state.message
    player.close()


def test_complete_n_key_advances() -> None:
    player = EvaluationPlayer()
    player.start_level(0)
    player.handle_command(Command.MOVE_RIGHT)
    player.handle_command(Command.MOVE_RIGHT)
    assert player.state.screen is Screen.COMPLETE
    player.handle_command(Command.NEXT)
    assert player.state.level_index == 1
    assert player.state.screen is Screen.MODE_SELECT
    player.close()


def test_back_from_play_returns_to_mode_select() -> None:
    player = EvaluationPlayer()
    player.start_level(5)
    player.handle_command(Command.BACK)
    assert player.state.screen is Screen.MODE_SELECT
    assert player.state.level_index == 5
    player.handle_command(Command.BACK)
    assert player.state.screen is Screen.PICKER
    assert player.state.level_index == 5
    player.close()


def test_quit_command() -> None:
    player = EvaluationPlayer()
    player.handle_command(Command.QUIT)
    assert player.state.quit_requested is True
    player.close()


def test_start_level_out_of_range() -> None:
    player = EvaluationPlayer()
    with pytest.raises(IndexError):
        player.start_level(player.level_count)
    player.close()


def test_play_observation_uses_level_size() -> None:
    player = EvaluationPlayer()
    level = player.levels[0]
    player.start_level(0)
    obs = player.observation()
    assert obs is not None
    assert obs.shape == (level.height, level.width)
    player.close()


def test_search_enqueued_toggle_click() -> None:
    player = EvaluationPlayer()
    player.start_search(0)
    assert player.search_viewer is not None
    assert player.search_viewer.state.show_enqueued is False
    assert map_key_to_command("e") is Command.NONE
    message_before = player.state.message
    n_filtered = player.search_viewer.n_events()

    player.enqueued_toggle_rect = (100, 200, 160, 22)
    player.handle_search_click((120, 210))
    assert player.search_viewer.state.show_enqueued is True
    assert player.search_viewer.n_events() >= n_filtered
    assert player.state.message == message_before

    player.handle_search_click((120, 210))
    assert player.search_viewer.state.show_enqueued is False
    assert player.search_viewer.n_events() == n_filtered
    assert player.state.message == message_before
    player.close()


def test_search_help_toggle_and_back() -> None:
    player = EvaluationPlayer()
    player.start_search(0)
    assert player.state.screen is Screen.SEARCH
    assert player.state.help_open is False
    assert map_key_to_command("h") is Command.HELP
    assert map_key_to_command("?") is Command.HELP
    assert player.search_viewer is not None

    player.handle_command(Command.HELP)
    assert player.state.help_open is True
    # Stepping is blocked while help is open.
    index_before = player.search_viewer.state.index
    player.handle_command(Command.MOVE_RIGHT)
    assert player.search_viewer.state.index == index_before

    player.handle_command(Command.BACK)
    assert player.state.help_open is False
    assert player.state.screen is Screen.SEARCH

    player.handle_command(Command.HELP)
    assert player.state.help_open is True
    player.handle_command(Command.HELP)
    assert player.state.help_open is False
    player.close()


def test_search_help_click() -> None:
    player = EvaluationPlayer()
    player.start_search(0)
    player.help_button_rect = (100, 100, 26, 26)
    player.handle_search_click((110, 110))
    assert player.state.help_open is True
    player.handle_search_click((10, 10))
    assert player.state.help_open is False
    player.handle_search_click((10, 10))
    assert player.state.help_open is False
    player.close()


def test_draw_tiles_headless() -> None:
    pygame = pytest.importorskip("pygame")
    from sokoban_fess.level import BOX, BOX_ON_TARGET, FLOOR, PLAYER, PLAYER_ON_TARGET, TARGET, WALL
    from sokoban_fess.play import _draw_tile

    pygame.display.init()
    try:
        surface = pygame.Surface((64, 64))
        for tile in (WALL, FLOOR, TARGET, BOX, BOX_ON_TARGET, PLAYER, PLAYER_ON_TARGET):
            rect = pygame.Rect(8, 8, 48, 48)
            surface.fill((0, 0, 0))
            _draw_tile(pygame, surface, tile, rect)
            # Ensure something non-black was drawn for each tile type.
            assert surface.get_at((32, 32)) != (0, 0, 0, 255) or tile == WALL
    finally:
        pygame.display.quit()


def test_draw_search_help_modal_headless() -> None:
    pygame = pytest.importorskip("pygame")
    from sokoban_fess.play import SEARCH_STATS_HELP, _draw_search_help_modal, _wrap_text

    assert any(label == "packed" for label, _ in SEARCH_STATS_HELP)
    assert any(label == "Show enqueued moves" for label, _ in SEARCH_STATS_HELP)
    pygame.display.init()
    pygame.font.init()
    try:
        fonts = {
            "title": pygame.font.SysFont("dejavusans", 28, bold=True),
            "body": pygame.font.SysFont("dejavusans", 20),
            "small": pygame.font.SysFont("dejavusans", 16),
        }
        assert _wrap_text(fonts["small"], "one two three", 40)
        surface = pygame.Surface((880, 720))
        _draw_search_help_modal(pygame, surface, fonts=fonts)
        # Overlay should have painted something other than pure black.
        assert surface.get_at((440, 360)) != (0, 0, 0, 255)
    finally:
        pygame.font.quit()
        pygame.display.quit()


def test_draw_enqueued_toggle_headless() -> None:
    pygame = pytest.importorskip("pygame")
    from sokoban_fess.play import _draw_enqueued_toggle

    pygame.display.init()
    pygame.font.init()
    try:
        fonts = {"small": pygame.font.SysFont("dejavusans", 16)}
        surface = pygame.Surface((880, 720))
        panel = pygame.Rect(580, 90, 280, 600)
        pygame.draw.rect(surface, (36, 36, 48), panel)
        off_rect = _draw_enqueued_toggle(
            pygame, surface, fonts=fonts, panel=panel, y=140, on=False
        )
        on_rect = _draw_enqueued_toggle(
            pygame, surface, fonts=fonts, panel=panel, y=180, on=True
        )
        assert off_rect[2] > 0 and off_rect[3] > 0
        assert on_rect[2] > 0 and on_rect[3] > 0
    finally:
        pygame.font.quit()
        pygame.display.quit()
