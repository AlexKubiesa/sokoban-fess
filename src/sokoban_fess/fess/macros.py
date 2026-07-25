"""Macro moves: push one box from cell A to any reachable cell B.

A macro is a sequence of pushes of the same box with no other box moved in
between (Shoham & Schaeffer, CoG 2020 §III-D).
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple, Protocol

from sokoban_fess.level import ACTION_CHARS, ACTION_DELTAS, ACTIONS
from sokoban_fess.solver import Boxes, Cell


class BoardView(Protocol):
    """Minimal board queries needed to enumerate macros."""

    def is_blocked(self, pos: Cell, obstacles: Boxes) -> bool: ...

    def reachable_cells(self, player: Cell, obstacles: Boxes) -> set[Cell]: ...

    def walk_path(self, start: Cell, goal: Cell, obstacles: Boxes) -> str | None: ...


class MacroCandidate(NamedTuple):
    """One legal macro from a position."""

    push_from: Cell
    push_to: Cell
    moves: str
    push_count: int
    push_path: tuple[Cell, ...]
    new_boxes: Boxes
    new_player: Cell


def _reach_with_parents(
    board: BoardView, player: Cell, obstacles: Boxes
) -> tuple[set[Cell], dict[Cell, tuple[Cell, int]]]:
    """BFS free-space reachability with parent pointers for path reconstruction."""
    came_from: dict[Cell, tuple[Cell, int]] = {}
    seen: set[Cell] = set()
    queue: deque[Cell] = deque([player])
    seen.add(player)
    while queue:
        pos = queue.popleft()
        row, col = pos
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            nxt = (row + d_row, col + d_col)
            if nxt in seen or board.is_blocked(nxt, obstacles):
                continue
            came_from[nxt] = (pos, action)
            seen.add(nxt)
            queue.append(nxt)
    return seen, came_from


def _path_string(
    start: Cell, goal: Cell, came_from: dict[Cell, tuple[Cell, int]]
) -> str | None:
    if start == goal:
        return ""
    if goal not in came_from and goal != start:
        return None
    chars: list[str] = []
    cur = goal
    while cur != start:
        parent_pos, act = came_from[cur]
        chars.append(ACTION_CHARS[act])
        cur = parent_pos
    chars.reverse()
    return "".join(chars)


def macros_for_box(
    board: BoardView, boxes: Boxes, player: Cell, box: Cell
) -> list[MacroCandidate]:
    """All destinations one box can reach by pushing it alone."""
    frozen = boxes - {box}
    visited: set[tuple[Cell, Cell]] = set()
    best: dict[Cell, tuple[int, str, Cell, tuple[Cell, ...]]] = {}

    obstacles0 = frozen | {box}
    reach0, _ = _reach_with_parents(board, player, obstacles0)
    if not reach0:
        return []
    start_key = (box, min(reach0))
    visited.add(start_key)
    queue: deque[tuple[Cell, Cell, str, int, tuple[Cell, ...]]] = deque(
        [(box, player, "", 0, (box,))]
    )

    while queue:
        bpos, ppos, moves, pcount, box_path = queue.popleft()
        obstacles = frozen | {bpos}
        reach, parents = _reach_with_parents(board, ppos, obstacles)

        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            stand = (bpos[0] - d_row, bpos[1] - d_col)
            dest = (bpos[0] + d_row, bpos[1] + d_col)
            if stand not in reach:
                continue
            if board.is_blocked(dest, frozen):
                continue
            walk = _path_string(ppos, stand, parents)
            if walk is None:
                continue

            push = ACTION_CHARS[action]
            new_moves = moves + walk + push
            new_player = bpos
            new_box = dest
            new_pcount = pcount + 1
            new_path = box_path + (new_box,)

            new_obstacles = frozen | {new_box}
            new_reach, _ = _reach_with_parents(board, new_player, new_obstacles)
            if not new_reach:
                continue
            new_key = (new_box, min(new_reach))
            if new_key in visited:
                continue
            visited.add(new_key)

            prev = best.get(new_box)
            if prev is None or (new_pcount, len(new_moves)) < (prev[0], len(prev[1])):
                best[new_box] = (new_pcount, new_moves, new_player, new_path)

            queue.append((new_box, new_player, new_moves, new_pcount, new_path))

    out: list[MacroCandidate] = []
    for dest, (pcount, moves, new_player, path) in best.items():
        out.append(
            MacroCandidate(
                push_from=box,
                push_to=dest,
                moves=moves,
                push_count=pcount,
                push_path=path,
                new_boxes=frozenset((boxes - {box}) | {dest}),
                new_player=new_player,
            )
        )
    return out


def legal_macros(
    board: BoardView, boxes: Boxes, player: Cell
) -> list[MacroCandidate]:
    """Every legal macro from ``boxes`` / ``player``."""
    macros: list[MacroCandidate] = []
    for box in boxes:
        macros.extend(macros_for_box(board, boxes, player, box))
    return macros


def one_step_push_count(board: BoardView, boxes: Boxes, player: Cell) -> int:
    """Count immediate single pushes (cheap explorer / mobility signal)."""
    reach = board.reachable_cells(player, boxes)
    count = 0
    for box in boxes:
        box_row, box_col = box
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            stand = (box_row - d_row, box_col - d_col)
            dest = (box_row + d_row, box_col + d_col)
            if stand in reach and not board.is_blocked(dest, boxes):
                count += 1
    return count
