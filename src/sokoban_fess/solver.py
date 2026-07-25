"""Shared Sokoban search types and static board helpers.

Solver algorithms (currently FESS) live in sibling modules and emit
``SearchEvent`` / ``SearchResult`` for visualization and benchmarking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, NamedTuple

from sokoban_fess.level import ACTION_DELTAS, ACTIONS, Level

# Sentinel for "unreachable" distances / costs.
_INF = 10**9

Cell = tuple[int, int]
Boxes = frozenset[Cell]


class StateKey(NamedTuple):
    """Canonical search state: box layout + a cell in the player's free region."""

    boxes: Boxes
    player: Cell  # lexicographically smallest reachable free cell


SearchEventKind = Literal[
    "start",
    "expand",
    "enqueue",
    "deadlock",
    "goal",
    "capped",
    "timeout",
    "exhausted",
]

StopReason = Literal["goal", "capped", "timeout", "exhausted"]
AlgorithmName = Literal["fess"]
AdvisorName = Literal[
    "packing",
    "connectivity",
    "room_connectivity",
    "hotspots",
    "explorer",
    "opener",
    "oop",
]


# True / projected FESS coordinates: (packed, connectivity, room_connectivity, oop).
FeatureCoords = tuple[int, int, int, int]


@dataclass(frozen=True)
class SearchEvent:
    """One step of search, suitable for visualization.

    ``pushes`` and ``weight`` follow FESS terminology (Shoham & Schaeffer):
    path push count and search-tree weight (advisor macros add 0; others add 1).
    """

    kind: SearchEventKind
    boxes: Boxes
    player: Cell
    pushes: int
    weight: int
    open_len: int
    closed_len: int
    # Push being considered (enqueue / deadlock), or the push that led here (expand/goal).
    push_from: Cell | None = None
    push_to: Cell | None = None
    push_char: str | None = None
    # Index of the parent state's expand/start event in the trace (search-tree parent).
    parent_index: int | None = None
    # Box cells visited by the push/macro (inclusive), for cornered path arrows.
    push_path: tuple[Cell, ...] | None = None
    # Advisors that endorsed this macro (empty if none / not a push event).
    advisors: tuple[AdvisorName, ...] = ()
    # True 4-D feature coordinates for ``boxes`` (None if not recorded).
    features: FeatureCoords | None = None
    # Projected FS cell after regression pinning (None if not recorded).
    projected: FeatureCoords | None = None


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a solver run plus an optional event trace for visualization."""

    solution: str | None
    events: tuple[SearchEvent, ...]
    states_visited: int
    truncated_trace: bool = False
    stop_reason: StopReason = "exhausted"
    algorithm: AlgorithmName = "fess"


def _is_open(cell: Cell, walls: Boxes, height: int, width: int) -> bool:
    """True if ``cell`` is in-bounds and not a wall."""
    row, col = cell
    return 0 <= row < height and 0 <= col < width and cell not in walls


def _push_distances_to_targets(
    walls: Boxes,
    targets: tuple[Cell, ...],
    height: int,
    width: int,
) -> dict[Cell, list[int]]:
    """For each free cell, min pushes to every target (static; walls only).

    Built by reverse BFS from each target: a box at ``prev`` can be pushed to
    ``curr`` only if the player stance behind ``prev`` is also open. Positions
    never reached are omitted (static deadlocks for off-target boxes).
    """
    n_targets = len(targets)
    dist_to_target: dict[Cell, list[int]] = {}

    for target_index, goal in enumerate(targets):
        if not _is_open(goal, walls, height, width):
            continue
        queue: deque[Cell] = deque([goal])
        dist: dict[Cell, int] = {goal: 0}
        while queue:
            curr = queue.popleft()
            here = dist[curr]
            c_row, c_col = curr
            for action in ACTIONS:
                d_row, d_col = ACTION_DELTAS[action]
                # Forward push was from prev → curr along (d_row, d_col).
                prev = (c_row - d_row, c_col - d_col)
                stand = (prev[0] - d_row, prev[1] - d_col)
                if prev in dist:
                    continue
                if not _is_open(prev, walls, height, width):
                    continue
                if not _is_open(stand, walls, height, width):
                    continue
                dist[prev] = here + 1
                queue.append(prev)

        for cell, distance in dist.items():
            row = dist_to_target.setdefault(cell, [_INF] * n_targets)
            row[target_index] = distance

    return dist_to_target


def search(
    level: Level,
    *,
    max_states: int = 500_000,
    max_time: float | None = None,
    collect_trace: bool = False,
    max_events: int = 250_000,
) -> SearchResult:
    """Run the default solver (FESS) and optionally collect a visualization trace."""
    from sokoban_fess.fess import search_fess

    return search_fess(
        level,
        max_states=max_states,
        max_time=max_time,
        collect_trace=collect_trace,
        max_events=max_events,
    )


def find_solution(
    level: Level,
    *,
    max_states: int = 500_000,
    max_time: float | None = None,
) -> str | None:
    """Return a move string that solves ``level``, or ``None`` if unsolvable / capped."""
    return search(
        level,
        max_states=max_states,
        max_time=max_time,
        collect_trace=False,
    ).solution


def is_solvable(
    level: Level,
    *,
    max_states: int = 500_000,
    max_time: float | None = None,
) -> bool:
    """Return True if a solution exists within the search budget."""
    return (
        find_solution(
            level,
            max_states=max_states,
            max_time=max_time,
        )
        is not None
    )
