"""Room partition and room-connectivity (CoG §III-B).

A room is a connected open region that contains at least one 2×3 (or 3×2)
open block. Room-connectivity counts how many static room–room edges are
broken by boxes in the current position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from sokoban_fess.solver import Boxes, Cell


@dataclass(frozen=True)
class RoomLayout:
    """Static room partition and adjacency for one level."""

    # Free cell → room id (corridors / non-room free cells omitted).
    cell_room: dict[Cell, int]
    # Undirected edges between room ids (empty-board connectivity).
    edges: frozenset[tuple[int, int]]
    n_rooms: int

    def room_of(self, cell: Cell) -> int | None:
        return self.cell_room.get(cell)


def _open_cells(walls: Boxes, height: int, width: int) -> set[Cell]:
    return {
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in walls
    }


def _is_open_rect(
    open_set: set[Cell], top: int, left: int, rows: int, cols: int
) -> bool:
    for r in range(top, top + rows):
        for c in range(left, left + cols):
            if (r, c) not in open_set:
                return False
    return True


def _seed_room_cells(open_set: set[Cell], height: int, width: int) -> set[Cell]:
    """Cells that participate in some 2×3 or 3×2 open rectangle."""
    seeds: set[Cell] = set()
    for r in range(height):
        for c in range(width):
            if _is_open_rect(open_set, r, c, 2, 3):
                for dr in range(2):
                    for dc in range(3):
                        seeds.add((r + dr, c + dc))
            if _is_open_rect(open_set, r, c, 3, 2):
                for dr in range(3):
                    for dc in range(2):
                        seeds.add((r + dr, c + dc))
    return seeds


def build_rooms(walls: Boxes, height: int, width: int) -> RoomLayout:
    """Partition the empty board into rooms and compute the room graph."""
    open_set = _open_cells(walls, height, width)
    seeds = _seed_room_cells(open_set, height, width)
    cell_room: dict[Cell, int] = {}
    room_id = 0
    remaining = set(seeds)
    while remaining:
        start = remaining.pop()
        stack = [start]
        members: list[Cell] = []
        while stack:
            pos = stack.pop()
            if pos in cell_room or pos not in seeds:
                continue
            cell_room[pos] = room_id
            members.append(pos)
            remaining.discard(pos)
            r, c = pos
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in seeds and nxt not in cell_room:
                    stack.append(nxt)
        if members:
            room_id += 1

    # Adjacency: path between rooms that does not enter a third room.
    edges: set[tuple[int, int]] = set()
    for a in range(room_id):
        for b in range(a + 1, room_id):
            if _rooms_adjacent(open_set, cell_room, a, b):
                edges.add((a, b))

    return RoomLayout(
        cell_room=cell_room, edges=frozenset(edges), n_rooms=room_id
    )


def _rooms_adjacent(
    open_set: set[Cell],
    cell_room: dict[Cell, int],
    room_a: int,
    room_b: int,
) -> bool:
    """True if room_a can reach room_b without entering any other room."""
    starts = [c for c, rid in cell_room.items() if rid == room_a]
    if not starts:
        return False
    allowed_rooms = {room_a, room_b}
    seen: set[Cell] = set()
    queue: deque[Cell] = deque(starts)
    seen.update(starts)
    while queue:
        pos = queue.popleft()
        if cell_room.get(pos) == room_b:
            return True
        r, c = pos
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in seen or nxt not in open_set:
                continue
            rid = cell_room.get(nxt)
            if rid is not None and rid not in allowed_rooms:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def room_connectivity_with_open(
    layout: RoomLayout, boxes: Boxes, open_set: set[Cell]
) -> int:
    """Broken room-edge count using the full empty-board open set."""
    if not layout.edges:
        return 0
    broken = 0
    for a, b in layout.edges:
        if not _edge_open_full(layout, boxes, open_set, a, b):
            broken += 1
    return broken


def _edge_open_full(
    layout: RoomLayout,
    boxes: Boxes,
    open_set: set[Cell],
    room_a: int,
    room_b: int,
) -> bool:
    starts = [
        c
        for c, rid in layout.cell_room.items()
        if rid == room_a and c not in boxes
    ]
    if not starts:
        return False
    allowed = {room_a, room_b}
    seen: set[Cell] = set()
    queue: deque[Cell] = deque(starts)
    seen.update(starts)
    while queue:
        pos = queue.popleft()
        if layout.cell_room.get(pos) == room_b:
            return True
        r, c = pos
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in seen or nxt not in open_set or nxt in boxes:
                continue
            rid = layout.cell_room.get(nxt)
            if rid is not None and rid not in allowed:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False
