"""Lightweight deadlock detectors used by FESS (CoG §II / §III).

CoG describes dead-end marking in the search tree. Here we also apply a few
local board checks that are easy to verify educationally:

- static unreachable-off-target cells (via push distances)
- classic 2×2 freeze (box+wall block with an off-target box)
- simple corner freeze (box in a corner not on a target)
"""

from __future__ import annotations

from sokoban_fess.solver import Boxes, Cell, _INF


def is_corner_deadlock(
    cell: Cell, walls: Boxes, targets: Boxes, height: int, width: int
) -> bool:
    """True if ``cell`` is a non-target corner (two adjacent walls)."""
    if cell in targets:
        return False
    row, col = cell
    blocked = []
    for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        r, c = nxt
        if not (0 <= r < height and 0 <= c < width) or nxt in walls:
            blocked.append(True)
        else:
            blocked.append(False)
    up, down, left, right = blocked
    return (up and left) or (up and right) or (down and left) or (down and right)


def has_2x2_deadlock(boxes: Boxes, walls: Boxes, targets: Boxes) -> bool:
    """True if any 2×2 of walls/boxes freezes an off-target box."""
    occupied = walls | boxes
    candidates: set[Cell] = set()
    for br, bc in boxes:
        for dr in (-1, 0):
            for dc in (-1, 0):
                candidates.add((br + dr, bc + dc))
    for top, left in candidates:
        cells = (
            (top, left),
            (top, left + 1),
            (top + 1, left),
            (top + 1, left + 1),
        )
        if any(c not in occupied for c in cells):
            continue
        for c in cells:
            if c in boxes and c not in targets:
                return True
    return False


def has_pattern_deadlock(
    boxes: Boxes,
    walls: Boxes,
    targets: Boxes,
    height: int,
    width: int,
    *,
    dist_to_target: dict[Cell, list[int]] | None = None,
) -> bool:
    """Combine static, corner, and 2×2 deadlock checks."""
    for box in boxes:
        if box in targets:
            continue
        if is_corner_deadlock(box, walls, targets, height, width):
            return True
        if dist_to_target is not None:
            distances = dist_to_target.get(box)
            if distances is None or all(d >= _INF for d in distances):
                return True
    if has_2x2_deadlock(boxes, walls, targets):
        return True
    return False
