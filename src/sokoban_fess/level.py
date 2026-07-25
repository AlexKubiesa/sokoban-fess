"""Sokoban level representation, parsing, and observation encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Observation tile codes
WALL = 0
FLOOR = 1
TARGET = 2
BOX = 3
BOX_ON_TARGET = 4
PLAYER = 5
PLAYER_ON_TARGET = 6

TILE_CHARS = {
    "#": WALL,
    " ": FLOOR,
    ".": TARGET,
    "$": BOX,
    "*": BOX_ON_TARGET,
    "@": PLAYER,
    "+": PLAYER_ON_TARGET,
}

CHAR_TILES = {v: k for k, v in TILE_CHARS.items()}

# Action directions: up, right, down, left
ACTIONS = (0, 1, 2, 3)
ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (0, 1),  # right
    2: (1, 0),  # down
    3: (0, -1),  # left
}
ACTION_CHARS = {0: "U", 1: "R", 2: "D", 3: "L"}
CHAR_ACTIONS = {v: k for k, v in ACTION_CHARS.items()}

DEFAULT_HEIGHT = 20
DEFAULT_WIDTH = 20


@dataclass(frozen=True)
class Level:
    """Immutable Sokoban level definition."""

    name: str
    walls: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]
    height: int
    width: int
    solution: str | None = None
    difficulty: int = 1

    def __post_init__(self) -> None:
        validate_level(self)


@dataclass
class LevelState:
    """Mutable play state for a Sokoban level."""

    level: Level
    boxes: set[tuple[int, int]]
    player: tuple[int, int]
    steps: int = 0
    max_steps: int = 200

    @classmethod
    def from_level(cls, level: Level, *, max_steps: int = 200) -> LevelState:
        return cls(
            level=level,
            boxes=set(level.boxes),
            player=level.player,
            steps=0,
            max_steps=max_steps,
        )

    @property
    def success(self) -> bool:
        return self.boxes <= self.level.targets and len(self.boxes) == len(self.level.targets)

    @property
    def truncated(self) -> bool:
        return self.steps >= self.max_steps and not self.success

    def boxes_on_targets(self) -> int:
        return sum(1 for box in self.boxes if box in self.level.targets)

    def is_wall(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or c < 0 or r >= self.level.height or c >= self.level.width:
            return True
        return pos in self.level.walls

    def is_free(self, pos: tuple[int, int]) -> bool:
        return not self.is_wall(pos) and pos not in self.boxes


def validate_level(level: Level) -> None:
    """Raise ValueError if the level is structurally invalid."""
    if level.height <= 0 or level.width <= 0:
        raise ValueError("Level dimensions must be positive")
    if not level.boxes:
        raise ValueError("Level must contain at least one box")
    if len(level.boxes) != len(level.targets):
        raise ValueError(
            f"Box count ({len(level.boxes)}) must equal target count ({len(level.targets)})"
        )

    def _in_bounds(pos: tuple[int, int], label: str) -> None:
        r, c = pos
        if not (0 <= r < level.height and 0 <= c < level.width):
            raise ValueError(f"{label} {pos} is out of bounds")
        if pos in level.walls:
            raise ValueError(f"{label} {pos} overlaps a wall")

    _in_bounds(level.player, "Player")
    for box in level.boxes:
        _in_bounds(box, "Box")
    for target in level.targets:
        _in_bounds(target, "Target")

    if len(set(level.boxes)) != len(level.boxes):
        raise ValueError("Duplicate box positions")
    if level.player in level.boxes:
        raise ValueError("Player overlaps a box")

    # Playable cells must form a connected component containing player, boxes, and targets.
    playable = {
        (r, c)
        for r in range(level.height)
        for c in range(level.width)
        if (r, c) not in level.walls
    }
    if not playable:
        raise ValueError("Level has no playable cells")
    start = level.player
    seen: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        pos = stack.pop()
        if pos in seen or pos not in playable:
            continue
        seen.add(pos)
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            stack.append((r + dr, c + dc))
    required = set(level.boxes) | set(level.targets) | {level.player}
    missing = required - seen
    if missing:
        raise ValueError(f"Unreachable cells from player: {sorted(missing)}")


def parse_ascii(
    ascii_map: str | Iterable[str],
    *,
    name: str = "unnamed",
    solution: str | None = None,
    difficulty: int = 1,
    height: int | None = None,
    width: int | None = None,
) -> Level:
    """Parse an ASCII Sokoban map into a Level.

    Characters: ``#`` wall, `` `` floor, ``.`` target, ``$`` box,
    ``*`` box on target, ``@`` player, ``+`` player on target.
    """
    if isinstance(ascii_map, str):
        rows = [line.rstrip("\n") for line in ascii_map.strip("\n").splitlines()]
    else:
        rows = [str(line).rstrip("\n") for line in ascii_map]
    if not rows:
        raise ValueError("Empty ASCII map")

    raw_h = len(rows)
    raw_w = max(len(row) for row in rows)
    h = height if height is not None else raw_h
    w = width if width is not None else raw_w
    if raw_h > h or raw_w > w:
        raise ValueError(f"Map size {raw_h}x{raw_w} exceeds target {h}x{w}")

    # Pad shorter rows and pad the board to the requested size with walls.
    padded = [row.ljust(raw_w) for row in rows]
    while len(padded) < h:
        padded.append("#" * raw_w)
    padded = [row.ljust(w, "#") for row in padded]

    walls: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None

    for r, row in enumerate(padded):
        for c, ch in enumerate(row):
            if ch not in TILE_CHARS:
                raise ValueError(f"Unknown tile character {ch!r} at {(r, c)}")
            code = TILE_CHARS[ch]
            if code == WALL:
                walls.add((r, c))
            elif code == TARGET:
                targets.add((r, c))
            elif code == BOX:
                boxes.add((r, c))
            elif code == BOX_ON_TARGET:
                boxes.add((r, c))
                targets.add((r, c))
            elif code == PLAYER:
                if player is not None:
                    raise ValueError("Multiple players in map")
                player = (r, c)
            elif code == PLAYER_ON_TARGET:
                if player is not None:
                    raise ValueError("Multiple players in map")
                player = (r, c)
                targets.add((r, c))

    if player is None:
        raise ValueError("Map must contain a player (@ or +)")

    return Level(
        name=name,
        walls=frozenset(walls),
        targets=frozenset(targets),
        boxes=frozenset(boxes),
        player=player,
        height=h,
        width=w,
        solution=solution,
        difficulty=difficulty,
    )


def level_to_ascii(level: Level | LevelState) -> str:
    """Serialize a level or state to an ASCII map."""
    if isinstance(level, LevelState):
        state = level
        base = level.level
        boxes = state.boxes
        player = state.player
    else:
        base = level
        boxes = set(level.boxes)
        player = level.player

    lines: list[str] = []
    for r in range(base.height):
        chars: list[str] = []
        for c in range(base.width):
            pos = (r, c)
            if pos in base.walls:
                chars.append("#")
            elif pos == player and pos in base.targets:
                chars.append("+")
            elif pos == player:
                chars.append("@")
            elif pos in boxes and pos in base.targets:
                chars.append("*")
            elif pos in boxes:
                chars.append("$")
            elif pos in base.targets:
                chars.append(".")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)


def encode_observation(state: LevelState, *, height: int | None = None, width: int | None = None) -> np.ndarray:
    """Encode state as a uint8 HxW grid of tile codes, optionally zero-padded."""
    h = height if height is not None else state.level.height
    w = width if width is not None else state.level.width
    obs = np.full((h, w), WALL, dtype=np.uint8)
    for r in range(state.level.height):
        for c in range(state.level.width):
            pos = (r, c)
            if pos in state.level.walls:
                code = WALL
            elif pos == state.player and pos in state.level.targets:
                code = PLAYER_ON_TARGET
            elif pos == state.player:
                code = PLAYER
            elif pos in state.boxes and pos in state.level.targets:
                code = BOX_ON_TARGET
            elif pos in state.boxes:
                code = BOX
            elif pos in state.level.targets:
                code = TARGET
            else:
                code = FLOOR
            obs[r, c] = code
    return obs


def apply_action(
    state: LevelState,
    action: int,
    *,
    step_penalty: float = -0.01,
    box_on_target_reward: float = 0.1,
    box_off_target_penalty: float = -0.1,
    success_reward: float = 1.0,
) -> tuple[LevelState, float, bool, bool, dict]:
    """Apply a movement action and return (state, reward, terminated, truncated, info)."""
    if action not in ACTION_DELTAS:
        raise ValueError(f"Invalid action {action}")

    dr, dc = ACTION_DELTAS[action]
    pr, pc = state.player
    nxt = (pr + dr, pc + dc)
    pushed = False
    moved = False
    boxes_before = state.boxes_on_targets()

    new_boxes = set(state.boxes)
    new_player = state.player

    if state.is_wall(nxt):
        pass  # blocked
    elif nxt in state.boxes:
        beyond = (nxt[0] + dr, nxt[1] + dc)
        if state.is_free(beyond):
            new_boxes.remove(nxt)
            new_boxes.add(beyond)
            new_player = nxt
            pushed = True
            moved = True
    else:
        new_player = nxt
        moved = True

    new_state = LevelState(
        level=state.level,
        boxes=new_boxes,
        player=new_player,
        steps=state.steps + 1,
        max_steps=state.max_steps,
    )
    boxes_after = new_state.boxes_on_targets()
    delta = boxes_after - boxes_before

    reward = step_penalty
    if delta > 0:
        reward += box_on_target_reward * delta
    elif delta < 0:
        reward += box_off_target_penalty * (-delta)

    terminated = new_state.success
    truncated = new_state.truncated and not terminated
    if terminated:
        reward += success_reward

    info = {
        "moved": moved,
        "pushed": pushed,
        "boxes_on_targets": boxes_after,
        "boxes_total": len(new_state.boxes),
    }
    return new_state, float(reward), terminated, truncated, info


def replay_solution(level: Level, *, max_steps: int | None = None) -> bool:
    """Return True if replaying ``level.solution`` reaches a solved state."""
    if not level.solution:
        return False
    steps = max_steps if max_steps is not None else max(200, len(level.solution) + 10)
    state = LevelState.from_level(level, max_steps=steps)
    for ch in level.solution:
        if ch not in CHAR_ACTIONS:
            return False
        state, _, terminated, _, _ = apply_action(state, CHAR_ACTIONS[ch], step_penalty=0.0)
        if terminated:
            return True
    return state.success


def render_ansi(state: LevelState) -> str:
    """Render state as an ANSI-colored ASCII string."""
    colors = {
        "#": "\033[90m#\033[0m",
        " ": " ",
        ".": "\033[32m.\033[0m",
        "$": "\033[33m$\033[0m",
        "*": "\033[33m*\033[0m",
        "@": "\033[36m@\033[0m",
        "+": "\033[36m+\033[0m",
    }
    ascii_map = level_to_ascii(state)
    return "\n".join("".join(colors.get(ch, ch) for ch in line) for line in ascii_map.splitlines())


# Distinct RGB colors for each tile code
_RGB_PALETTE = np.array(
    [
        [40, 40, 40],  # WALL
        [220, 220, 220],  # FLOOR
        [80, 180, 80],  # TARGET
        [200, 140, 40],  # BOX
        [220, 180, 60],  # BOX_ON_TARGET
        [60, 140, 220],  # PLAYER
        [40, 180, 220],  # PLAYER_ON_TARGET
    ],
    dtype=np.uint8,
)


def render_rgb(state: LevelState, *, cell_size: int = 16) -> np.ndarray:
    """Render state as an RGB array without external graphics dependencies."""
    obs = encode_observation(state)
    h, w = obs.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = _RGB_PALETTE[int(obs[r, c])]
            img[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = color
    return img
