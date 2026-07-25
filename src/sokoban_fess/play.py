"""Interactive Pygame player for the Sokoban FESS evaluation suite."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import pygame

from sokoban_fess.level import (
    BOX,
    BOX_ON_TARGET,
    FLOOR,
    PLAYER,
    PLAYER_ON_TARGET,
    TARGET,
    WALL,
    LevelState,
    apply_action,
    encode_observation,
)
from sokoban_fess.levels import evaluation_level_count, load_evaluation_levels
from sokoban_fess.fess import feature_coordinates, search_fess
from sokoban_fess.visualize import (
    DEADLOCK_TINT,
    ENQUEUE_TINT,
    EXPAND_TINT,
    FS_AXIS,
    FS_CURRENT,
    FS_EDGE,
    FS_GRAPH_H,
    FS_NODE,
    FS_PATH_EDGE,
    FS_PATH_NODE,
    FS_PROJECTED,
    HEAT_COLOR,
    KIND_LABELS,
    PATH_ARROW,
    PATH_PLAYER,
    PATH_TINT,
    PUSH_ARROW,
    SEARCH_BOARD_BOTTOM,
    SIDE_PANEL_W,
    SearchViewer,
    event_push_path,
    heat_cells,
    layout_feature_space_graph,
    path_ghost_cells,
    path_push_paths,
)

# Tile colors matching the existing RGB palette feel.
TILE_COLORS: dict[int, tuple[int, int, int]] = {
    WALL: (40, 40, 40),
    FLOOR: (220, 220, 220),
    TARGET: (80, 180, 80),
    BOX: (200, 140, 40),
    BOX_ON_TARGET: (220, 180, 60),
    PLAYER: (60, 140, 220),
    PLAYER_ON_TARGET: (40, 180, 220),
}

FLOOR_COLOR = (214, 214, 222)
FLOOR_EDGE = (190, 190, 200)
WALL_COLOR = (58, 58, 68)
WALL_HIGHLIGHT = (78, 78, 90)
WALL_MORTAR = (34, 34, 42)
TARGET_RING = (190, 55, 45)
TARGET_FILL = (220, 90, 70)
CRATE_WOOD = (176, 112, 42)
CRATE_WOOD_DARK = (132, 78, 28)
CRATE_WOOD_LIGHT = (210, 150, 70)
CRATE_ON_TARGET = (198, 150, 48)
PLAYER_BODY = (52, 124, 210)
PLAYER_BODY_DARK = (32, 88, 170)
PLAYER_SKIN = (240, 198, 160)
PLAYER_OUTLINE = (20, 40, 80)

BG_COLOR = (24, 24, 32)
PANEL_COLOR = (36, 36, 48)
TEXT_COLOR = (230, 230, 235)
MUTED_TEXT = (160, 160, 175)
ACCENT = (90, 160, 255)
SUCCESS_COLOR = (80, 200, 120)
WARN_COLOR = (220, 160, 60)

CELL_SIZE = 48
PREVIEW_CELL = 12
FPS = 60
WINDOW_WIDTH = 880
WINDOW_HEIGHT = 720
PICKER_VISIBLE = 10
BOARD_MARGIN = 24
BOARD_TOP = 90
BOARD_BOTTOM = 72

MODE_PLAY = 0
MODE_SOLVE = 1
MODE_OPTIONS = ("Play level", "Watch FESS solver")

# Side-panel help copy for the FESS search viewer.
SEARCH_STATS_HELP: tuple[tuple[str, str], ...] = (
    (
        "Phase",
        "Search = FESS events; Solution = replay of the found path.",
    ),
    (
        "Features",
        "4-D FESS coordinates for the current box layout:",
    ),
    ("packed", "packing-plan progress (higher better)"),
    ("connectivity", "free-cell components (lower better)"),
    ("room_conn", "broken room–room edges (lower better)"),
    ("oop", "out-of-plan boxes this packing step (lower better)"),
    (
        "Macro",
        "Box path under consideration (start cell → end cell).",
    ),
    (
        "Priority",
        "How FESS ranks the path to this node:",
    ),
    (
        "weight",
        "tree cost; advisor macros +0, else +1 (lower preferred)",
    ),
    (
        "pushes",
        "box pushes on the path here",
    ),
    (
        "Advisors",
        "Heuristics that endorsed this macro (weight 0).",
    ),
    (
        "Pending moves",
        "Macros still waiting in the open set.",
    ),
    (
        "Show enqueued moves",
        "Toggle: off = only macros chosen for expansion; on = every enqueued successor.",
    ),
    (
        "Tree nodes",
        "Distinct board states already visited (closed set).",
    ),
    (
        "Feature space",
        "True FS cells on the current timeline. Left→right follows FESS "
        "feature progress (oop, packed, connectivity, room_conn). With "
        "enqueued moves off, only expanded states. Yellow = path; cyan = "
        "current; violet ring = projected cell when FESS pins a regression.",
    ),
    (
        "Trace",
        "Shown when the event log hit max_events.",
    ),
)


class Screen(Enum):
    PICKER = auto()
    MODE_SELECT = auto()
    PLAYING = auto()
    COMPLETE = auto()
    SEARCH = auto()


class Command(Enum):
    NONE = auto()
    MOVE_UP = auto()
    MOVE_RIGHT = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    SELECT = auto()
    BACK = auto()
    RESTART = auto()
    UNDO = auto()
    NEXT = auto()
    QUIT = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    SPEED_DOWN = auto()
    SPEED_UP = auto()
    SEEK_START = auto()
    SEEK_END = auto()
    TOGGLE_PLAY = auto()
    HELP = auto()


ACTION_FOR_COMMAND = {
    Command.MOVE_UP: 0,
    Command.MOVE_RIGHT: 1,
    Command.MOVE_DOWN: 2,
    Command.MOVE_LEFT: 3,
}

PLAYING_HINT = "WASD/Arrows move · U undo · R restart · Esc mode select"


def _copy_level_state(state: LevelState) -> LevelState:
    """Deep-copy play state so undo history owns its own boxes set."""
    return LevelState(
        level=state.level,
        boxes=set(state.boxes),
        player=state.player,
        steps=state.steps,
        max_steps=state.max_steps,
    )


def map_key_to_command(key_name: str) -> Command:
    """Map a normalized key name to a player command."""
    key = key_name.lower()
    mapping = {
        "up": Command.MOVE_UP,
        "w": Command.MOVE_UP,
        "right": Command.MOVE_RIGHT,
        "d": Command.MOVE_RIGHT,
        "down": Command.MOVE_DOWN,
        "s": Command.MOVE_DOWN,
        "left": Command.MOVE_LEFT,
        "a": Command.MOVE_LEFT,
        "return": Command.SELECT,
        "enter": Command.SELECT,
        "space": Command.SELECT,
        "escape": Command.BACK,
        "backspace": Command.BACK,
        "r": Command.RESTART,
        "u": Command.UNDO,
        "n": Command.NEXT,
        "q": Command.QUIT,
        "pageup": Command.SCROLL_UP,
        "pagedown": Command.SCROLL_DOWN,
        "[": Command.SPEED_DOWN,
        "]": Command.SPEED_UP,
        "home": Command.SEEK_START,
        "end": Command.SEEK_END,
        "p": Command.TOGGLE_PLAY,
        "h": Command.HELP,
        "?": Command.HELP,
    }
    return mapping.get(key, Command.NONE)


@dataclass
class PlayerState:
    """Pure UI/game-flow state for the evaluation player."""

    screen: Screen = Screen.PICKER
    level_index: int = 0
    scroll_offset: int = 0
    mode_option: int = MODE_PLAY
    steps: int = 0
    boxes_on_targets: int = 0
    boxes_total: int = 0
    level_name: str = ""
    difficulty: int = 1
    success: bool = False
    truncated: bool = False
    help_open: bool = False
    message: str = ""
    quit_requested: bool = False


class EvaluationPlayer:
    """Controller for picker + mode select + play / FESS search."""

    def __init__(
        self,
        *,
        level_count: int | None = None,
        visible_rows: int = PICKER_VISIBLE,
        max_steps: int = 10_000,
        max_states: int = 500_000,
        max_events: int = 250_000,
    ) -> None:
        self.levels = load_evaluation_levels()
        self.level_count = level_count if level_count is not None else len(self.levels)
        if self.level_count < 1:
            raise ValueError("Evaluation suite is empty")
        self.max_steps = max_steps
        self.visible_rows = max(1, visible_rows)
        self.max_states = max_states
        self.max_events = max_events
        self.level_state: LevelState | None = None
        self._undo_stack: list[LevelState] = []
        self.search_viewer: SearchViewer | None = None
        self.help_button_rect: tuple[int, int, int, int] | None = None
        self.enqueued_toggle_rect: tuple[int, int, int, int] | None = None
        self.state = PlayerState()
        self._reset_to_picker(selected=0)

    def _clamp_scroll(self) -> None:
        max_offset = max(0, self.level_count - self.visible_rows)
        self.state.scroll_offset = max(0, min(self.state.scroll_offset, max_offset))
        if self.state.level_index < self.state.scroll_offset:
            self.state.scroll_offset = self.state.level_index
        elif self.state.level_index >= self.state.scroll_offset + self.visible_rows:
            self.state.scroll_offset = self.state.level_index - self.visible_rows + 1

    def _reset_to_picker(self, *, selected: int | None = None) -> None:
        if selected is not None:
            self.state.level_index = max(0, min(selected, self.level_count - 1))
        self.state.screen = Screen.PICKER
        self.state.success = False
        self.state.truncated = False
        self.state.help_open = False
        self.state.mode_option = MODE_PLAY
        self.search_viewer = None
        self.help_button_rect = None
        self.enqueued_toggle_rect = None
        self.state.message = "Select an evaluation level"
        self._clamp_scroll()

    def open_mode_select(self, index: int) -> None:
        """Show Play vs Solve choices for an evaluation level."""
        if index < 0 or index >= self.level_count:
            raise IndexError(
                f"level_index {index} out of range [0, {self.level_count})"
            )
        level = self.levels[index]
        self.search_viewer = None
        self.help_button_rect = None
        self.enqueued_toggle_rect = None
        self.state.screen = Screen.MODE_SELECT
        self.state.level_index = index
        self.state.level_name = level.name
        self.state.difficulty = level.difficulty
        self.state.mode_option = MODE_PLAY
        self.state.success = False
        self.state.truncated = False
        self.state.help_open = False
        self.state.steps = 0
        self.state.boxes_on_targets = 0
        self.state.boxes_total = len(level.boxes)
        self.state.message = "Choose Play or FESS · Esc back to picker"
        self._clamp_scroll()

    def start_level(self, index: int) -> None:
        """Load and start an evaluation level by zero-based index."""
        if index < 0 or index >= self.level_count:
            raise IndexError(
                f"level_index {index} out of range [0, {self.level_count})"
            )
        level = self.levels[index]
        self.search_viewer = None
        self.help_button_rect = None
        self.enqueued_toggle_rect = None
        self.level_state = LevelState.from_level(level, max_steps=self.max_steps)
        self._undo_stack.clear()
        self.state.screen = Screen.PLAYING
        self.state.level_index = index
        self.state.level_name = level.name
        self.state.difficulty = level.difficulty
        self.state.steps = self.level_state.steps
        self.state.boxes_on_targets = self.level_state.boxes_on_targets()
        self.state.boxes_total = len(self.level_state.boxes)
        self.state.success = False
        self.state.truncated = False
        self.state.help_open = False
        self.state.message = PLAYING_HINT
        self._clamp_scroll()

    def start_search(self, index: int) -> None:
        """Run FESS with a trace and open the search visualizer."""
        if index < 0 or index >= self.level_count:
            raise IndexError(
                f"level_index {index} out of range [0, {self.level_count})"
            )
        level = self.levels[index]
        result = search_fess(
            level,
            max_states=self.max_states,
            collect_trace=True,
            max_events=self.max_events,
        )
        self.search_viewer = SearchViewer(level, result)
        self.help_button_rect = None
        self.enqueued_toggle_rect = None
        self.state.mode_option = MODE_SOLVE
        self.state.screen = Screen.SEARCH
        self.state.level_index = index
        self.state.level_name = level.name
        self.state.difficulty = level.difficulty
        self.state.success = result.solution is not None
        self.state.truncated = result.truncated_trace
        self.state.help_open = False
        self.state.steps = 0
        self.state.boxes_on_targets = 0
        self.state.boxes_total = len(level.boxes)
        self.state.message = (
            "←/→ step · Space/P autoplay · [/] speed · H/? help · Esc mode select"
        )
        self._clamp_scroll()

    def restart_level(self) -> None:
        self.start_level(self.state.level_index)

    def restart_search(self) -> None:
        self.start_search(self.state.level_index)

    def next_level_index(self) -> int | None:
        nxt = self.state.level_index + 1
        if nxt >= self.level_count:
            return None
        return nxt

    def advance_to_next_level(self) -> bool:
        """Advance after completion. Returns False if suite is finished."""
        nxt = self.next_level_index()
        if nxt is None:
            self._reset_to_picker(selected=self.state.level_index)
            self.state.message = "Suite complete — pick another level"
            return False
        self.open_mode_select(nxt)
        return True

    def observation(self):
        if self.state.screen is Screen.SEARCH and self.search_viewer is not None:
            return self.search_viewer.observation()
        if self.level_state is None:
            return None
        return encode_observation(self.level_state)

    def handle_command(self, command: Command) -> PlayerState:
        if command is Command.NONE:
            return self.state
        if command is Command.QUIT:
            self.state.quit_requested = True
            return self.state

        if self.state.screen is Screen.PICKER:
            self._handle_picker(command)
        elif self.state.screen is Screen.MODE_SELECT:
            self._handle_mode_select(command)
        elif self.state.screen is Screen.PLAYING:
            self._handle_playing(command)
        elif self.state.screen is Screen.COMPLETE:
            self._handle_complete(command)
        elif self.state.screen is Screen.SEARCH:
            self._handle_search(command)
        return self.state

    def _handle_picker(self, command: Command) -> None:
        if command in {Command.MOVE_UP, Command.SCROLL_UP}:
            self.state.level_index = max(0, self.state.level_index - 1)
            self._clamp_scroll()
        elif command in {Command.MOVE_DOWN, Command.SCROLL_DOWN}:
            self.state.level_index = min(
                self.level_count - 1, self.state.level_index + 1
            )
            self._clamp_scroll()
        elif command is Command.SELECT:
            self.open_mode_select(self.state.level_index)
        elif command is Command.BACK:
            self.state.quit_requested = True

    def _handle_mode_select(self, command: Command) -> None:
        n_modes = len(MODE_OPTIONS)
        if command in {Command.MOVE_UP, Command.SCROLL_UP}:
            self.state.mode_option = (self.state.mode_option - 1) % n_modes
        elif command in {Command.MOVE_DOWN, Command.SCROLL_DOWN}:
            self.state.mode_option = (self.state.mode_option + 1) % n_modes
        elif command is Command.SELECT:
            if self.state.mode_option == MODE_SOLVE:
                self.start_search(self.state.level_index)
            else:
                self.start_level(self.state.level_index)
        elif command is Command.BACK:
            self._reset_to_picker(selected=self.state.level_index)

    def _sync_play_ui(self) -> None:
        """Refresh HUD fields from the current level state."""
        if self.level_state is None:
            return
        self.state.steps = self.level_state.steps
        self.state.boxes_on_targets = self.level_state.boxes_on_targets()
        self.state.boxes_total = len(self.level_state.boxes)
        self.state.success = self.level_state.success
        self.state.truncated = self.level_state.truncated and not self.level_state.success

    def _undo_move(self) -> bool:
        """Restore the previous successful move, if any. Returns True if undone."""
        if not self._undo_stack:
            return False
        self.level_state = self._undo_stack.pop()
        self._sync_play_ui()
        self.state.screen = Screen.PLAYING
        self.state.message = PLAYING_HINT
        return True

    def _handle_playing(self, command: Command) -> None:
        if command is Command.BACK:
            self.open_mode_select(self.state.level_index)
            return
        if command is Command.RESTART:
            self.restart_level()
            return
        if command is Command.UNDO:
            self._undo_move()
            return
        action = ACTION_FOR_COMMAND.get(command)
        if action is None or self.level_state is None:
            return
        before = _copy_level_state(self.level_state)
        self.level_state, _, terminated, truncated, info = apply_action(
            self.level_state, action
        )
        # Don't count wall / blocked-push attempts toward the step counter.
        if not info["moved"]:
            self.level_state = before
            return
        self._undo_stack.append(before)
        self._sync_play_ui()
        if terminated and self.state.success:
            self.state.screen = Screen.COMPLETE
            if self.next_level_index() is None:
                self.state.message = (
                    "Level cleared! Enter returns to picker · Esc mode select"
                )
            else:
                self.state.message = (
                    "Level cleared! Enter/N next · Esc mode select · R replay"
                )
        elif truncated:
            self.state.message = "Step limit reached · R restart · Esc mode select"

    def _handle_complete(self, command: Command) -> None:
        if command in {Command.SELECT, Command.NEXT}:
            self.advance_to_next_level()
        elif command is Command.RESTART:
            self.restart_level()
        elif command is Command.UNDO:
            self._undo_move()
        elif command is Command.BACK:
            self.open_mode_select(self.state.level_index)

    def toggle_search_help(self) -> None:
        """Open or close the search-stats help overlay."""
        if self.state.screen is not Screen.SEARCH:
            return
        self.state.help_open = not self.state.help_open
        if self.state.help_open and self.search_viewer is not None:
            self.search_viewer.state.playing = False

    def toggle_show_enqueued(self) -> None:
        """Toggle whether enqueue/deadlock frames appear in the search viewer."""
        viewer = self.search_viewer
        if self.state.screen is not Screen.SEARCH or viewer is None:
            return
        viewer.toggle_show_enqueued()

    @staticmethod
    def _point_in_rect(pos: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        x, y, w, h = rect
        px, py = pos
        return x <= px < x + w and y <= py < y + h

    def handle_search_click(self, pos: tuple[int, int]) -> None:
        """Handle a mouse click while watching the solver."""
        if self.state.screen is not Screen.SEARCH:
            return
        if self.state.help_open:
            self.state.help_open = False
            return
        if self.help_button_rect is not None and self._point_in_rect(
            pos, self.help_button_rect
        ):
            self.state.help_open = True
            if self.search_viewer is not None:
                self.search_viewer.state.playing = False
            return
        if self.enqueued_toggle_rect is not None and self._point_in_rect(
            pos, self.enqueued_toggle_rect
        ):
            self.toggle_show_enqueued()

    def _handle_search(self, command: Command) -> None:
        viewer = self.search_viewer
        if viewer is None:
            self.open_mode_select(self.state.level_index)
            return
        if command is Command.HELP:
            self.toggle_search_help()
            return
        if command is Command.BACK:
            if self.state.help_open:
                self.state.help_open = False
                return
            self.open_mode_select(self.state.level_index)
            return
        if self.state.help_open:
            # Keep navigation keys from acting under the overlay.
            return
        if command is Command.RESTART:
            self.restart_search()
            return
        if command in {Command.SELECT, Command.TOGGLE_PLAY}:
            viewer.toggle_play()
            return
        if command is Command.MOVE_LEFT:
            viewer.state.playing = False
            viewer.step(-1)
            return
        if command is Command.MOVE_RIGHT:
            viewer.state.playing = False
            viewer.step(1)
            return
        if command is Command.SPEED_DOWN:
            viewer.nudge_speed(0.5)
            return
        if command is Command.SPEED_UP:
            viewer.nudge_speed(2.0)
            return
        if command is Command.SEEK_START:
            viewer.state.playing = False
            viewer.seek(0)
            return
        if command is Command.SEEK_END:
            viewer.state.playing = False
            viewer.seek(viewer.n_total() - 1)

    def close(self) -> None:
        self.level_state = None
        self.search_viewer = None


def _draw_floor(pygame: Any, surface: Any, rect: Any) -> None:
    pygame.draw.rect(surface, FLOOR_COLOR, rect)
    pygame.draw.rect(surface, FLOOR_EDGE, rect, width=max(1, rect.width // 16))


def _draw_wall(pygame: Any, surface: Any, rect: Any) -> None:
    pygame.draw.rect(surface, WALL_COLOR, rect)
    inset = max(1, rect.width // 10)
    brick = pygame.Rect(
        rect.x + inset, rect.y + inset, rect.width - 2 * inset, rect.height - 2 * inset
    )
    pygame.draw.rect(surface, WALL_HIGHLIGHT, brick)
    # Mortar lines for a brick look (skip when cells are tiny).
    if rect.width >= 10:
        mid_y = brick.centery
        mid_x = brick.centerx
        pygame.draw.line(
            surface, WALL_MORTAR, (brick.left, mid_y), (brick.right - 1, mid_y), 1
        )
        pygame.draw.line(surface, WALL_MORTAR, (mid_x, brick.top), (mid_x, mid_y), 1)
        pygame.draw.line(
            surface,
            WALL_MORTAR,
            (brick.left, (brick.top + mid_y) // 2),
            (mid_x, (brick.top + mid_y) // 2),
            1,
        )


def _draw_target(pygame: Any, surface: Any, rect: Any) -> None:
    _draw_floor(pygame, surface, rect)
    pad = max(1, rect.width // 5)
    ring = pygame.Rect(
        rect.x + pad, rect.y + pad, rect.width - 2 * pad, rect.height - 2 * pad
    )
    width = max(1, rect.width // 10)
    pygame.draw.ellipse(surface, TARGET_RING, ring, width=width)
    # Inner cross marks the goal pad.
    if rect.width >= 8:
        cx, cy = ring.center
        arm = max(2, ring.width // 3)
        pygame.draw.line(surface, TARGET_FILL, (cx - arm, cy), (cx + arm, cy), width)
        pygame.draw.line(surface, TARGET_FILL, (cx, cy - arm), (cx, cy + arm), width)


def _draw_crate(
    pygame: Any, surface: Any, rect: Any, *, on_target: bool = False
) -> None:
    if on_target:
        _draw_target(pygame, surface, rect)
    else:
        _draw_floor(pygame, surface, rect)

    pad = max(1, rect.width // 8)
    box = pygame.Rect(
        rect.x + pad, rect.y + pad, rect.width - 2 * pad, rect.height - 2 * pad
    )
    wood = CRATE_ON_TARGET if on_target else CRATE_WOOD
    pygame.draw.rect(surface, wood, box, border_radius=max(1, box.width // 10))
    pygame.draw.rect(
        surface,
        CRATE_WOOD_DARK,
        box,
        width=max(1, box.width // 12),
        border_radius=max(1, box.width // 10),
    )

    if box.width >= 8:
        # Plank seams.
        third = box.height // 3
        pygame.draw.line(
            surface,
            CRATE_WOOD_DARK,
            (box.left + 2, box.top + third),
            (box.right - 3, box.top + third),
            1,
        )
        pygame.draw.line(
            surface,
            CRATE_WOOD_DARK,
            (box.left + 2, box.top + 2 * third),
            (box.right - 3, box.top + 2 * third),
            1,
        )
        # Diagonal brace.
        pygame.draw.line(
            surface,
            CRATE_WOOD_LIGHT,
            (box.left + 3, box.top + 3),
            (box.right - 4, box.bottom - 4),
            max(1, box.width // 10),
        )
        pygame.draw.line(
            surface,
            CRATE_WOOD_DARK,
            (box.right - 4, box.top + 3),
            (box.left + 3, box.bottom - 4),
            max(1, box.width // 14),
        )


def _draw_player(
    pygame: Any, surface: Any, rect: Any, *, on_target: bool = False
) -> None:
    if on_target:
        _draw_target(pygame, surface, rect)
    else:
        _draw_floor(pygame, surface, rect)

    cx, cy = rect.centerx, rect.centery
    # Scale body parts from cell size so preview and play boards both read clearly.
    head_r = max(2, rect.width // 6)
    body_w = max(3, rect.width // 3)
    body_h = max(4, rect.height // 3)
    leg_h = max(3, rect.height // 5)
    body_top = cy - body_h // 3
    body_rect = pygame.Rect(cx - body_w // 2, body_top, body_w, body_h)

    # Legs
    pygame.draw.line(
        surface,
        PLAYER_BODY_DARK,
        (cx - body_w // 3, body_rect.bottom - 1),
        (cx - body_w // 2, body_rect.bottom + leg_h - 1),
        max(2, rect.width // 12),
    )
    pygame.draw.line(
        surface,
        PLAYER_BODY_DARK,
        (cx + body_w // 3, body_rect.bottom - 1),
        (cx + body_w // 2, body_rect.bottom + leg_h - 1),
        max(2, rect.width // 12),
    )
    # Torso
    pygame.draw.rect(surface, PLAYER_BODY, body_rect, border_radius=max(1, body_w // 4))
    # Arms
    arm_y = body_rect.top + body_h // 3
    pygame.draw.line(
        surface,
        PLAYER_BODY,
        (body_rect.left, arm_y),
        (body_rect.left - body_w // 2, arm_y + body_h // 4),
        max(2, rect.width // 12),
    )
    pygame.draw.line(
        surface,
        PLAYER_BODY,
        (body_rect.right, arm_y),
        (body_rect.right + body_w // 2, arm_y + body_h // 4),
        max(2, rect.width // 12),
    )
    # Head
    head_center = (cx, body_rect.top - head_r // 2)
    pygame.draw.circle(surface, PLAYER_SKIN, head_center, head_r)
    pygame.draw.circle(
        surface, PLAYER_OUTLINE, head_center, head_r, width=max(1, head_r // 4)
    )


def _draw_tile(pygame: Any, surface: Any, tile: int, rect: Any) -> None:
    if tile == WALL:
        _draw_wall(pygame, surface, rect)
    elif tile == FLOOR:
        _draw_floor(pygame, surface, rect)
    elif tile == TARGET:
        _draw_target(pygame, surface, rect)
    elif tile == BOX:
        _draw_crate(pygame, surface, rect, on_target=False)
    elif tile == BOX_ON_TARGET:
        _draw_crate(pygame, surface, rect, on_target=True)
    elif tile == PLAYER:
        _draw_player(pygame, surface, rect, on_target=False)
    elif tile == PLAYER_ON_TARGET:
        _draw_player(pygame, surface, rect, on_target=True)
    else:
        pygame.draw.rect(surface, TILE_COLORS.get(tile, FLOOR_COLOR), rect)


def _draw_board(
    pygame: Any,
    surface: Any,
    obs,
    *,
    origin: tuple[int, int],
    cell: int,
) -> None:
    ox, oy = origin
    height, width = obs.shape
    for r in range(height):
        for c in range(width):
            rect = pygame.Rect(ox + c * cell, oy + r * cell, cell - 1, cell - 1)
            _draw_tile(pygame, surface, int(obs[r, c]), rect)


def _cell_rect(origin: tuple[int, int], cell: int, pos: tuple[int, int]) -> Any:
    r, c = pos
    ox, oy = origin
    return pygame.Rect(ox + c * cell, oy + r * cell, cell - 1, cell - 1)


def _draw_push_arrow(
    pygame: Any,
    surface: Any,
    *,
    origin: tuple[int, int],
    cell: int,
    push_from: tuple[int, int],
    push_to: tuple[int, int],
    color: tuple[int, int, int],
    width_div: int = 10,
) -> None:
    _draw_push_path(
        pygame,
        surface,
        origin=origin,
        cell=cell,
        cells=(push_from, push_to),
        color=color,
        width_div=width_div,
    )


def _draw_push_path(
    pygame: Any,
    surface: Any,
    *,
    origin: tuple[int, int],
    cell: int,
    cells: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    width_div: int = 10,
) -> None:
    """Draw a box route through ``cells`` (inclusive), with corners and an arrowhead."""
    if len(cells) < 2:
        return
    points = [_cell_rect(origin, cell, pos).center for pos in cells]
    width = max(2, cell // width_div)
    if len(points) == 2:
        pygame.draw.line(surface, color, points[0], points[1], width)
    else:
        pygame.draw.lines(surface, color, False, points, width)

    fx, fy = points[-2]
    tx, ty = points[-1]
    dx, dy = tx - fx, ty - fy
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    left = (
        tx - ux * cell * 0.35 + uy * cell * 0.22,
        ty - uy * cell * 0.35 - ux * cell * 0.22,
    )
    right = (
        tx - ux * cell * 0.35 - uy * cell * 0.22,
        ty - uy * cell * 0.35 + ux * cell * 0.22,
    )
    pygame.draw.polygon(surface, color, [points[-1], left, right])
    pygame.draw.circle(
        surface,
        color,
        points[0],
        max(3, cell // 8),
        width=max(1, cell // 16),
    )


def _draw_search_overlay(
    pygame: Any,
    surface: Any,
    *,
    origin: tuple[int, int],
    cell: int,
    viewer: SearchViewer,
) -> None:
    overlay = pygame.Surface((cell - 1, cell - 1), pygame.SRCALPHA)

    if viewer.phase == "solution":
        # Soft success wash on boxes during the solution coda.
        state = viewer.current_state()
        for pos in state.boxes:
            overlay.fill((SUCCESS_COLOR[0], SUCCESS_COLOR[1], SUCCESS_COLOR[2], 50))
            surface.blit(overlay, _cell_rect(origin, cell, pos))
        return

    event = viewer.event
    heat = heat_cells(
        viewer.result.events,
        viewer.event_index,
        show_enqueued=viewer.state.show_enqueued,
    )
    chain = viewer.path_indices()
    ghosts = path_ghost_cells(viewer.result.events, chain)

    for pos, intensity in heat.items():
        alpha = int(HEAT_COLOR[3] * intensity)
        overlay.fill((HEAT_COLOR[0], HEAT_COLOR[1], HEAT_COLOR[2], alpha))
        surface.blit(overlay, _cell_rect(origin, cell, pos))

    for pos, intensity in ghosts.items():
        alpha = int(PATH_TINT[3] * intensity)
        overlay.fill((PATH_TINT[0], PATH_TINT[1], PATH_TINT[2], alpha))
        surface.blit(overlay, _cell_rect(origin, cell, pos))

    # Path player trail (ancestors only).
    for idx in chain[:-1]:
        ev = viewer.result.events[idx]
        overlay.fill(PATH_PLAYER)
        surface.blit(overlay, _cell_rect(origin, cell, ev.player))

    for cells in path_push_paths(viewer.result.events, chain):
        _draw_push_path(
            pygame,
            surface,
            origin=origin,
            cell=cell,
            cells=cells,
            color=PATH_ARROW,
            width_div=12,
        )

    tint = {
        "expand": EXPAND_TINT,
        "enqueue": ENQUEUE_TINT,
        "deadlock": DEADLOCK_TINT,
    }.get(event.kind)
    if tint is not None:
        for pos in event.boxes:
            overlay.fill(tint)
            surface.blit(overlay, _cell_rect(origin, cell, pos))
        overlay.fill((*tint[:3], min(255, tint[3] + 40)))
        surface.blit(overlay, _cell_rect(origin, cell, event.player))

    current_path = event_push_path(event)
    if current_path is not None:
        _draw_push_path(
            pygame,
            surface,
            origin=origin,
            cell=cell,
            cells=current_path,
            color=PUSH_ARROW,
        )


def _kind_color(kind: str) -> tuple[int, int, int]:
    return {
        "start": ACCENT,
        "expand": ACCENT,
        "enqueue": SUCCESS_COLOR,
        "deadlock": (220, 90, 90),
        "goal": SUCCESS_COLOR,
        "capped": WARN_COLOR,
        "timeout": WARN_COLOR,
        "exhausted": (220, 90, 90),
    }.get(kind, TEXT_COLOR)


def _wrap_text(font: Any, text: str, max_width: int) -> list[str]:
    """Word-wrap ``text`` to fit ``max_width`` pixels."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_help_button(
    pygame: Any,
    surface: Any,
    *,
    fonts: dict[str, Any],
    panel: Any,
) -> tuple[int, int, int, int]:
    """Draw a ? button in the search stats panel header; return its rect."""
    size = 26
    rect = pygame.Rect(panel.right - size - 12, panel.y + 12, size, size)
    pygame.draw.rect(surface, (52, 52, 68), rect, border_radius=6)
    pygame.draw.rect(surface, ACCENT, rect, width=1, border_radius=6)
    mark = fonts["small"].render("?", True, ACCENT)
    surface.blit(
        mark,
        (
            rect.x + (rect.width - mark.get_width()) // 2,
            rect.y + (rect.height - mark.get_height()) // 2,
        ),
    )
    return (rect.x, rect.y, rect.width, rect.height)


def _draw_enqueued_toggle(
    pygame: Any,
    surface: Any,
    *,
    fonts: dict[str, Any],
    panel: Any,
    y: int,
    on: bool,
) -> tuple[int, int, int, int]:
    """Draw a labeled on/off switch for showing enqueued moves; return hit rect."""
    x = panel.x + 16
    track_w, track_h = 42, 22
    track = pygame.Rect(panel.right - track_w - 12, y, track_w, track_h)

    label = fonts["small"].render("Show enqueued moves", True, MUTED_TEXT)
    surface.blit(label, (x, y + 2))

    track_fill = SUCCESS_COLOR if on else (52, 52, 68)
    track_border = SUCCESS_COLOR if on else (90, 90, 110)
    pygame.draw.rect(surface, track_fill, track, border_radius=track_h // 2)
    pygame.draw.rect(surface, track_border, track, width=1, border_radius=track_h // 2)

    knob_r = 8
    knob_cx = track.right - 11 if on else track.left + 11
    pygame.draw.circle(surface, TEXT_COLOR, (knob_cx, track.centery), knob_r)

    # Whole row is clickable for easier hitting.
    return (x, y, panel.right - 12 - x, track_h)


def _draw_search_help_modal(
    pygame: Any,
    surface: Any,
    *,
    fonts: dict[str, Any],
) -> None:
    """Dim the screen and explain search-stats panel fields."""
    dim = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    dim.fill((8, 8, 14, 180))
    surface.blit(dim, (0, 0))

    modal_w = min(540, WINDOW_WIDTH - 40)
    modal_h = min(640, WINDOW_HEIGHT - 40)
    modal = pygame.Rect(
        (WINDOW_WIDTH - modal_w) // 2,
        (WINDOW_HEIGHT - modal_h) // 2,
        modal_w,
        modal_h,
    )
    pygame.draw.rect(surface, PANEL_COLOR, modal, border_radius=12)
    pygame.draw.rect(surface, ACCENT, modal, width=1, border_radius=12)

    pad = 20
    x = modal.x + pad
    y = modal.y + pad
    max_text_w = modal_w - 2 * pad
    surface.blit(fonts["title"].render("Search stats help", True, TEXT_COLOR), (x, y))
    y += 34
    surface.blit(
        fonts["small"].render(
            "Esc or click anywhere to close · H/? toggles", True, MUTED_TEXT
        ),
        (x, y),
    )
    y += 32

    nested = {"packed", "connectivity", "room_conn", "oop", "weight", "pushes"}
    for i, (label, blurb) in enumerate(SEARCH_STATS_HELP):
        indent = 14 if label in nested else 0
        if label in nested:
            line = f"- {label} — {blurb}"
            for wrapped in _wrap_text(fonts["small"], line, max_text_w - indent):
                surface.blit(
                    fonts["small"].render(wrapped, True, MUTED_TEXT),
                    (x + indent, y),
                )
                y += 17
        else:
            surface.blit(fonts["small"].render(label, True, TEXT_COLOR), (x, y))
            y += 17
            for wrapped in _wrap_text(fonts["small"], blurb, max_text_w):
                surface.blit(
                    fonts["small"].render(wrapped, True, MUTED_TEXT),
                    (x, y),
                )
                y += 17
        next_label = (
            SEARCH_STATS_HELP[i + 1][0] if i + 1 < len(SEARCH_STATS_HELP) else None
        )
        # Extra air before the next top-level section; nesteds stay tight.
        if next_label is not None and next_label not in nested:
            y += 12
        else:
            y += 4
        if y > modal.bottom - 24:
            break


def _draw_feature_space_graph(
    pygame: Any,
    surface: Any,
    *,
    fonts: dict[str, Any],
    panel: Any,
    graph_top: int,
    viewer: SearchViewer,
) -> None:
    """Draw discovered FS cells in the lower part of the search stats panel.

    Node positions are laid out from the **full** trace so the plot does not
    reflow as cells are discovered. Not-yet-seen cells are drawn in the plot
    background colour (invisible placeholders). Nodes use **true** features
    (matching the Features panel); a violet ring marks the projected cell when
    it differs after regression pinning.
    """
    plot = pygame.Rect(
        panel.x + 10,
        graph_top,
        panel.width - 20,
        min(FS_GRAPH_H, panel.bottom - graph_top - 10),
    )
    if plot.height < 80:
        return

    plot_bg = (28, 28, 38)
    pygame.draw.rect(surface, plot_bg, plot, border_radius=6)
    pygame.draw.rect(surface, FS_AXIS, plot, width=1, border_radius=6)

    title = fonts["small"].render("Feature space", True, MUTED_TEXT)
    surface.blit(title, (plot.x + 8, plot.y + 6))
    axis_label = fonts["small"].render("progress", True, FS_AXIS)
    arrow_y = plot.y + 30
    arrow_x0 = plot.x + 8
    arrow_x1 = plot.right - 12 - axis_label.get_width() - 6
    if arrow_x1 > arrow_x0 + 24:
        pygame.draw.line(surface, FS_AXIS, (arrow_x0, arrow_y), (arrow_x1, arrow_y), 1)
        pygame.draw.polygon(
            surface,
            FS_AXIS,
            [
                (arrow_x1, arrow_y),
                (arrow_x1 - 7, arrow_y - 4),
                (arrow_x1 - 7, arrow_y + 4),
            ],
        )
        surface.blit(
            axis_label, (arrow_x1 + 6, arrow_y - axis_label.get_height() // 2)
        )
    else:
        surface.blit(axis_label, (plot.x + 8, plot.y + 22))

    full = viewer.full_fs_graph()
    graph = viewer.current_fs_graph()
    inner = (plot.x + 4, plot.y + 40, plot.width - 8, plot.height - 48)
    positions = layout_feature_space_graph(full, inner, pad=14)

    if not positions:
        empty = fonts["small"].render("No FS cells yet", True, MUTED_TEXT)
        surface.blit(empty, (plot.x + 8, plot.centery))
        return

    visible_nodes = frozenset(graph.nodes)
    visible_edges = frozenset(graph.edges)

    # Future (not-yet-discovered) edges/nodes in plot background — holds layout.
    for a, b in full.edges:
        if (a, b) in visible_edges:
            continue
        pa, pb = positions.get(a), positions.get(b)
        if pa is None or pb is None:
            continue
        pygame.draw.aaline(surface, plot_bg, pa, pb)

    # Discovered non-path edges, then path edges on top.
    for a, b in graph.edges:
        if (a, b) in graph.path_edges:
            continue
        pa, pb = positions.get(a), positions.get(b)
        if pa is None or pb is None:
            continue
        pygame.draw.aaline(surface, FS_EDGE, pa, pb)
    for a, b in graph.path_edges:
        pa, pb = positions.get(a), positions.get(b)
        if pa is None or pb is None:
            continue
        pygame.draw.line(surface, FS_PATH_EDGE, pa, pb, 2)

    # Projection pin: true current → projected cell (regression).
    pin = graph.projected
    if (
        pin is not None
        and graph.current is not None
        and pin != graph.current
        and pin in positions
        and graph.current in positions
    ):
        pygame.draw.line(
            surface,
            FS_PROJECTED,
            positions[graph.current],
            positions[pin],
            1,
        )

    for cell, (px, py) in positions.items():
        if cell not in visible_nodes:
            pygame.draw.circle(surface, plot_bg, (int(px), int(py)), 4)
            continue
        on_path = cell in graph.path_nodes
        is_current = cell == graph.current
        is_projected = (
            pin is not None and cell == pin and cell != graph.current
        )
        if is_current:
            color = FS_CURRENT
            radius = 6
        elif on_path:
            color = FS_PATH_NODE
            radius = 5
        else:
            color = FS_NODE
            radius = 4
        pygame.draw.circle(surface, color, (int(px), int(py)), radius)
        if is_current:
            pygame.draw.circle(
                surface, TEXT_COLOR, (int(px), int(py)), radius + 2, width=1
            )
        elif is_projected:
            pygame.draw.circle(
                surface, FS_PROJECTED, (int(px), int(py)), radius + 3, width=2
            )


def _draw_search_hud(
    pygame: Any,
    surface: Any,
    player: EvaluationPlayer,
    *,
    fonts: dict[str, Any],
) -> None:
    viewer = player.search_viewer
    if viewer is None:
        return
    event = viewer.event
    vstate = viewer.state
    in_solution = viewer.phase == "solution"

    phase_title = "Solution replay" if in_solution else "Search"
    title = fonts["title"].render(
        f"{phase_title} · {viewer.level.name}", True, TEXT_COLOR
    )
    surface.blit(title, (24, 16))
    if in_solution:
        kind = fonts["body"].render("Playing found solution", True, SUCCESS_COLOR)
    else:
        kind = fonts["body"].render(
            KIND_LABELS.get(event.kind, event.kind),
            True,
            _kind_color(event.kind),
        )
    surface.blit(kind, (24, 52))
    if in_solution:
        frame_i = viewer.solution_frame_index + 1
        frame_n = viewer.n_solution_frames()
        progress = (
            f"Move {frame_i}/{frame_n}  ·  "
            f"{'Playing' if vstate.playing else 'Paused'}  ·  "
            f"{vstate.events_per_second:.0f} ev/s"
        )
    else:
        progress = (
            f"Event {vstate.index + 1}/{viewer.n_events()}  ·  "
            f"{'Playing' if vstate.playing else 'Paused'}  ·  "
            f"{vstate.events_per_second:.0f} ev/s"
        )
    surface.blit(fonts["small"].render(progress, True, MUTED_TEXT), (24, 78))

    panel = pygame.Rect(
        WINDOW_WIDTH - SIDE_PANEL_W - 16,
        BOARD_TOP,
        SIDE_PANEL_W,
        WINDOW_HEIGHT - BOARD_TOP - 24,
    )
    pygame.draw.rect(surface, PANEL_COLOR, panel, border_radius=8)

    if event.features is not None:
        feat = event.features
        packed, connectivity, room_conn, oop = feat
    else:
        computed = feature_coordinates(
            viewer.level, frozenset(viewer.current_state().boxes)
        )
        packed = computed.packed
        connectivity = computed.connectivity
        room_conn = computed.room_connectivity
        oop = computed.oop
    show_macro = not in_solution and bool(event.push_char)
    lines: list[tuple[str, str] | tuple[str, list[tuple[str, str]]]] = [
        ("Phase", "Solution" if in_solution else "Search"),
        (
            "Features",
            [
                ("packed", str(packed)),
                ("connectivity", str(connectivity)),
                ("room_conn", str(room_conn)),
                ("oop", str(oop)),
            ],
        ),
        (
            "Macro",
            f"{event.push_from} → {event.push_to}" if show_macro else "—",
        ),
        (
            "Priority",
            [
                (
                    "weight",
                    "∞" if event.weight < 0 else str(event.weight),
                ),
                (
                    "pushes",
                    (
                        str(event.pushes)
                        if not in_solution
                        else str(viewer.current_state().steps)
                    ),
                ),
            ],
        ),
        (
            "Advisors",
            ", ".join(event.advisors) if show_macro and event.advisors else "—",
        ),
        ("Pending moves", str(event.open_len)),
        ("Tree nodes", str(event.closed_len)),
    ]
    if viewer.result.truncated_trace:
        lines.append(("Trace", "truncated (hit max_events)"))

    y = panel.y + 16
    x = panel.x + 16
    graph_top = panel.bottom - FS_GRAPH_H - 8
    stats_bottom = graph_top - 4
    surface.blit(fonts["body"].render("Search stats", True, TEXT_COLOR), (x, y))
    player.help_button_rect = _draw_help_button(
        pygame, surface, fonts=fonts, panel=panel
    )
    y += 36
    player.enqueued_toggle_rect = _draw_enqueued_toggle(
        pygame,
        surface,
        fonts=fonts,
        panel=panel,
        y=y,
        on=vstate.show_enqueued,
    )
    y += 34
    compact = {"Phase", "Features", "Macro", "Priority", "Advisors"}
    extra_gap_before = {"Features": 6, "Priority": 6, "Pending moves": 10}
    for label, value in lines:
        if y >= stats_bottom:
            break
        if label in extra_gap_before:
            y += extra_gap_before[label]
        if isinstance(value, list):
            surface.blit(fonts["small"].render(label, True, MUTED_TEXT), (x, y))
            y += 20
            for sub_label, sub_value in value:
                if y >= stats_bottom:
                    break
                surface.blit(
                    fonts["small"].render(f"- {sub_label}", True, MUTED_TEXT),
                    (x, y),
                )
                surface.blit(
                    fonts["small"].render(sub_value, True, TEXT_COLOR),
                    (x + 128, y),
                )
                y += 20
            y += 8 if label in compact else 16
            continue
        surface.blit(fonts["small"].render(label, True, MUTED_TEXT), (x, y))
        shown = value if len(value) <= 28 else value[:25] + "…"
        surface.blit(fonts["small"].render(shown, True, TEXT_COLOR), (x, y + 16))
        y += 34 if label in compact else 40

    _draw_feature_space_graph(
        pygame,
        surface,
        fonts=fonts,
        panel=panel,
        graph_top=graph_top,
        viewer=viewer,
    )

    surface.blit(
        fonts["small"].render(player.state.message, True, MUTED_TEXT),
        (24, WINDOW_HEIGHT - 40),
    )

    if player.state.help_open:
        _draw_search_help_modal(pygame, surface, fonts=fonts)


def _run_pygame(player: EvaluationPlayer) -> int:
    pygame.init()
    pygame.display.set_caption("Sokoban FESS")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("dejavusans", 28, bold=True),
        "body": pygame.font.SysFont("dejavusans", 20),
        "small": pygame.font.SysFont("dejavusans", 16),
    }
    font = fonts["body"]
    small = fonts["small"]
    title_font = fonts["title"]

    key_names = {
        getattr(pygame, "K_UP", 273): "up",
        getattr(pygame, "K_DOWN", 274): "down",
        getattr(pygame, "K_LEFT", 276): "left",
        getattr(pygame, "K_RIGHT", 275): "right",
        getattr(pygame, "K_w", 119): "w",
        getattr(pygame, "K_a", 97): "a",
        getattr(pygame, "K_s", 115): "s",
        getattr(pygame, "K_d", 100): "d",
        getattr(pygame, "K_RETURN", 13): "return",
        getattr(pygame, "K_SPACE", 32): "space",
        getattr(pygame, "K_ESCAPE", 27): "escape",
        getattr(pygame, "K_BACKSPACE", 8): "backspace",
        getattr(pygame, "K_r", 114): "r",
        getattr(pygame, "K_n", 110): "n",
        getattr(pygame, "K_q", 113): "q",
        getattr(pygame, "K_p", 112): "p",
        getattr(pygame, "K_PAGEUP", 280): "pageup",
        getattr(pygame, "K_PAGEDOWN", 281): "pagedown",
        getattr(pygame, "K_LEFTBRACKET", 91): "[",
        getattr(pygame, "K_RIGHTBRACKET", 93): "]",
        getattr(pygame, "K_HOME", 278): "home",
        getattr(pygame, "K_END", 279): "end",
        getattr(pygame, "K_h", 104): "h",
        getattr(pygame, "K_QUESTION", 63): "?",
    }

    running = True
    while running and not player.state.quit_requested:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                player.handle_command(Command.QUIT)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if player.state.screen is Screen.SEARCH:
                    player.handle_search_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                name = key_names.get(event.key)
                if name is None and hasattr(event, "unicode") and event.unicode:
                    name = event.unicode
                if name:
                    player.handle_command(map_key_to_command(name))

        if player.state.screen is Screen.SEARCH and player.search_viewer is not None:
            player.search_viewer.tick(dt)

        screen.fill(BG_COLOR)
        state = player.state

        if state.screen is Screen.PICKER:
            title = title_font.render("Evaluation Levels", True, TEXT_COLOR)
            screen.blit(title, (24, 20))
            hint = small.render(
                "↑/↓ or W/S select · Enter choose mode · Q/Esc quit",
                True,
                MUTED_TEXT,
            )
            screen.blit(hint, (24, 56))

            list_x, list_y = 24, 96
            row_h = 36
            for row, idx in enumerate(
                range(
                    state.scroll_offset,
                    min(state.scroll_offset + player.visible_rows, player.level_count),
                )
            ):
                level = player.levels[idx]
                y = list_y + row * row_h
                selected = idx == state.level_index
                bg = ACCENT if selected else PANEL_COLOR
                pygame.draw.rect(
                    screen, bg, pygame.Rect(list_x, y, 360, row_h - 4), border_radius=6
                )
                label = f"{idx + 1:02d}. {level.name}  (d{level.difficulty})"
                color = (20, 20, 30) if selected else TEXT_COLOR
                screen.blit(font.render(label, True, color), (list_x + 12, y + 6))

            preview_level = player.levels[state.level_index]
            preview_obs = encode_observation(LevelState.from_level(preview_level))
            preview_origin = (420, 120)
            _draw_board(
                pygame, screen, preview_obs, origin=preview_origin, cell=PREVIEW_CELL
            )
            meta = small.render(
                f"Level {state.level_index + 1}/{player.level_count}",
                True,
                MUTED_TEXT,
            )
            screen.blit(meta, (420, 96))

        elif state.screen is Screen.MODE_SELECT:
            header = f"Level {state.level_index + 1}/{player.level_count}: {state.level_name}"
            screen.blit(title_font.render(header, True, TEXT_COLOR), (24, 16))
            screen.blit(
                small.render(
                    f"Difficulty {state.difficulty}  ·  ↑/↓ choose · Enter confirm · Esc picker",
                    True,
                    MUTED_TEXT,
                ),
                (24, 52),
            )

            preview_level = player.levels[state.level_index]
            preview_obs = encode_observation(LevelState.from_level(preview_level))
            cell = min(
                CELL_SIZE,
                (WINDOW_WIDTH - 2 * BOARD_MARGIN - 320) // preview_obs.shape[1],
                (WINDOW_HEIGHT - BOARD_TOP - BOARD_BOTTOM) // preview_obs.shape[0],
            )
            board_w = preview_obs.shape[1] * cell
            board_h = preview_obs.shape[0] * cell
            ox = BOARD_MARGIN
            oy = BOARD_TOP
            pygame.draw.rect(
                screen,
                PANEL_COLOR,
                pygame.Rect(ox - 8, oy - 8, board_w + 16, board_h + 16),
                border_radius=8,
            )
            _draw_board(pygame, screen, preview_obs, origin=(ox, oy), cell=cell)

            opt_x = ox + board_w + 40
            opt_y = BOARD_TOP
            for i, label in enumerate(MODE_OPTIONS):
                selected = state.mode_option == i
                bg = ACCENT if selected else PANEL_COLOR
                rect = pygame.Rect(opt_x, opt_y + i * 56, 280, 48)
                pygame.draw.rect(screen, bg, rect, border_radius=8)
                color = (20, 20, 30) if selected else TEXT_COLOR
                screen.blit(font.render(label, True, color), (rect.x + 16, rect.y + 12))

            screen.blit(
                font.render(state.message, True, TEXT_COLOR),
                (24, WINDOW_HEIGHT - 48),
            )

        elif state.screen is Screen.SEARCH:
            viewer = player.search_viewer
            obs = player.observation()
            if viewer is not None and obs is not None:
                board_area_w = WINDOW_WIDTH - SIDE_PANEL_W - 48
                cell = min(
                    CELL_SIZE,
                    (board_area_w - 2 * BOARD_MARGIN) // obs.shape[1],
                    (WINDOW_HEIGHT - BOARD_TOP - SEARCH_BOARD_BOTTOM) // obs.shape[0],
                )
                board_w = obs.shape[1] * cell
                board_h = obs.shape[0] * cell
                ox = BOARD_MARGIN + max(0, (board_area_w - board_w) // 2)
                oy = BOARD_TOP + 20
                pygame.draw.rect(
                    screen,
                    PANEL_COLOR,
                    pygame.Rect(ox - 8, oy - 8, board_w + 16, board_h + 16),
                    border_radius=8,
                )
                _draw_board(pygame, screen, obs, origin=(ox, oy), cell=cell)
                _draw_search_overlay(
                    pygame, screen, origin=(ox, oy), cell=cell, viewer=viewer
                )
            _draw_search_hud(pygame, screen, player, fonts=fonts)

        else:
            header = f"Level {state.level_index + 1}/{player.level_count}: {state.level_name}"
            screen.blit(title_font.render(header, True, TEXT_COLOR), (24, 16))
            stats = (
                f"Difficulty {state.difficulty}  ·  "
                f"Boxes {state.boxes_on_targets}/{state.boxes_total}  ·  "
                f"Steps {state.steps}"
            )
            screen.blit(small.render(stats, True, MUTED_TEXT), (24, 52))

            obs = player.observation()
            if obs is not None:
                cell = min(
                    CELL_SIZE,
                    (WINDOW_WIDTH - 2 * BOARD_MARGIN) // obs.shape[1],
                    (WINDOW_HEIGHT - BOARD_TOP - BOARD_BOTTOM) // obs.shape[0],
                )
                board_w = obs.shape[1] * cell
                board_h = obs.shape[0] * cell
                ox = (WINDOW_WIDTH - board_w) // 2
                oy = BOARD_TOP
                pygame.draw.rect(
                    screen,
                    PANEL_COLOR,
                    pygame.Rect(ox - 8, oy - 8, board_w + 16, board_h + 16),
                    border_radius=8,
                )
                _draw_board(pygame, screen, obs, origin=(ox, oy), cell=cell)

            msg_color = SUCCESS_COLOR if state.screen is Screen.COMPLETE else TEXT_COLOR
            if state.truncated and state.screen is Screen.PLAYING:
                msg_color = WARN_COLOR
            screen.blit(
                font.render(state.message, True, msg_color), (24, WINDOW_HEIGHT - 48)
            )

        pygame.display.flip()

    player.close()
    pygame.quit()
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for the interactive player."""
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Start directly on evaluation level 1..N (skips picker).",
    )


def run(args: argparse.Namespace) -> int:
    """Run the Pygame player from parsed CLI args."""
    count = evaluation_level_count()
    player = EvaluationPlayer()
    if args.level is not None:
        if args.level < 1 or args.level > count:
            print(f"--level must be between 1 and {count}", file=sys.stderr)
            return 2
        player.open_mode_select(args.level - 1)

    try:
        return _run_pygame(player)
    except SystemExit:
        player.close()
        raise
    except Exception:
        player.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play Sokoban FESS evaluation levels in Pygame."
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
