"""FESS advisors — each endorses at most one macro (CoG §III-E).

Advisor-endorsed macros receive weight 0; all others receive weight 1
(CoG experimental setting).
"""

from __future__ import annotations

from typing import Protocol

from sokoban_fess.level import ACTION_DELTAS, ACTIONS
from sokoban_fess.solver import AdvisorName, Boxes, Cell

from sokoban_fess.fess.features import FeatureCell, LevelAnalysis
from sokoban_fess.fess.macros import MacroCandidate, one_step_push_count


class BoardView(Protocol):
    def is_blocked(self, pos: Cell, obstacles: Boxes) -> bool: ...

    def reachable_cells(self, player: Cell, obstacles: Boxes) -> set[Cell]: ...


def _immediate_pushes(
    board: BoardView, boxes: Boxes, player: Cell
) -> set[tuple[Cell, Cell]]:
    """Set of (box, dest) single-step pushes currently available."""
    reach = board.reachable_cells(player, boxes)
    out: set[tuple[Cell, Cell]] = set()
    for box in boxes:
        br, bc = box
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            stand = (br - d_row, bc - d_col)
            dest = (br + d_row, bc + d_col)
            if stand in reach and not board.is_blocked(dest, boxes):
                out.add((box, dest))
    return out


def endorse_macros(
    analysis: LevelAnalysis,
    board: BoardView,
    boxes: Boxes,
    player: Cell,
    candidates: list[MacroCandidate],
) -> dict[tuple[Cell, Cell], tuple[AdvisorName, ...]]:
    """Return macros endorsed by at least one advisor → which advisors."""
    if not candidates:
        return {}
    parent_feat = analysis.features(boxes)
    parent_options = one_step_push_count(board, boxes, player)
    parent_hotspots = analysis.hotspot_count(boxes)
    by_move: dict[tuple[Cell, Cell], list[AdvisorName]] = {}

    # Compute child features once per candidate (room/connectivity are expensive).
    child_feats: list[FeatureCell] = []
    for c in candidates:
        # Prefer searcher's cache when available.
        feat_fn = getattr(board, "features", None)
        if callable(feat_fn):
            child_feats.append(feat_fn(c.new_boxes))
        else:
            child_feats.append(analysis.features(c.new_boxes))
    child_hotspots: list[int] | None = None
    if analysis.hotspot_pairs:
        child_hotspots = [analysis.hotspot_count(c.new_boxes) for c in candidates]

    def endorse(index: int, name: AdvisorName) -> None:
        macro = candidates[index]
        key = (macro.push_from, macro.push_to)
        names = by_move.setdefault(key, [])
        if name not in names:
            names.append(name)

    # --- packing ---
    packing_idxs = [
        i
        for i, feat in enumerate(child_feats)
        if feat.packed > parent_feat.packed
    ]
    if packing_idxs:
        best_i = max(packing_idxs, key=lambda i: child_feats[i].packed)
        endorse(best_i, "packing")

    # --- connectivity ---
    conn_idxs = [
        i
        for i, feat in enumerate(child_feats)
        if feat.connectivity < parent_feat.connectivity
    ]
    if conn_idxs:
        best_i = min(conn_idxs, key=lambda i: child_feats[i].connectivity)
        endorse(best_i, "connectivity")

    # --- room connectivity ---
    room_idxs = [
        i
        for i, feat in enumerate(child_feats)
        if feat.room_connectivity < parent_feat.room_connectivity
    ]
    if room_idxs:
        best_i = min(room_idxs, key=lambda i: child_feats[i].room_connectivity)
        endorse(best_i, "room_connectivity")

    # --- hotspots ---
    if child_hotspots is not None and parent_hotspots > 0:
        hot_idxs = [
            i for i, h in enumerate(child_hotspots) if h < parent_hotspots
        ]
        if hot_idxs:
            best_i = min(hot_idxs, key=lambda i: child_hotspots[i])
            endorse(best_i, "hotspots")

    # --- explorer: open access enabling a previously impossible push ---
    explorers: list[tuple[int, int]] = []
    parent_push_set = _immediate_pushes(board, boxes, player)
    for i, c in enumerate(candidates):
        child_pushes = _immediate_pushes(board, c.new_boxes, c.new_player)
        new_options = len(child_pushes - parent_push_set)
        if new_options > 0:
            explorers.append((new_options, i))
    if explorers:
        best_i = max(explorers, key=lambda item: item[0])[1]
        endorse(best_i, "explorer")
    else:
        # Fallback: maximize raw one-step push count (previous heuristic).
        explorers2: list[tuple[int, int]] = []
        for i, c in enumerate(candidates):
            options = one_step_push_count(board, c.new_boxes, c.new_player)
            if options > parent_options:
                explorers2.append((options, i))
        if explorers2:
            best_i = max(explorers2, key=lambda item: item[0])[1]
            endorse(best_i, "explorer")

    # --- opener ---
    opener_i = _opener_index(
        analysis,
        candidates,
        child_feats,
        child_hotspots,
        parent_feat,
        boxes,
    )
    if opener_i is not None:
        endorse(opener_i, "opener")

    # --- OOP ---
    oop_idxs = [
        i for i, feat in enumerate(child_feats) if feat.oop < parent_feat.oop
    ]
    if oop_idxs:

        def oop_score(i: int) -> tuple[int, int]:
            feat = child_feats[i]
            into_basin = 1 if candidates[i].push_to in analysis.packing.basin else 0
            return (-feat.oop, into_basin)

        best_i = max(oop_idxs, key=oop_score)
        endorse(best_i, "oop")
    else:
        oop_boxes = _oop_boxes(analysis, boxes)
        if oop_boxes:
            clearing = [
                i
                for i, c in enumerate(candidates)
                if c.push_from not in oop_boxes
                and c.push_to in analysis.packing.basin
                and child_feats[i].connectivity <= parent_feat.connectivity
            ]
            if clearing:
                best_i = min(clearing, key=lambda i: child_feats[i].connectivity)
                endorse(best_i, "oop")

    return {key: tuple(names) for key, names in by_move.items()}


def _oop_boxes(analysis: LevelAnalysis, boxes: Boxes) -> set[Cell]:
    packed_cells = set(
        analysis.packing.order[: analysis.packing.packed_count(boxes)]
    )
    return {
        b
        for b in boxes
        if b not in packed_cells and b not in analysis.packing.basin
    }


def _opener_index(
    analysis: LevelAnalysis,
    candidates: list[MacroCandidate],
    child_feats: list[FeatureCell],
    child_hotspots: list[int] | None,
    parent_feat: FeatureCell,
    boxes: Boxes,
) -> int | None:
    """Index of macro that clears the hottest hotspot (opener advisor)."""
    if not analysis.hotspot_pairs or not boxes:
        return None
    heat: dict[Cell, int] = {b: 0 for b in boxes}
    box_set = set(boxes)
    for x, y in analysis.hotspot_pairs:
        if x in box_set and y in box_set:
            heat[y] = heat.get(y, 0) + 1
    if not heat or max(heat.values()) == 0:
        return None
    hottest = max(heat, key=heat.get)  # type: ignore[arg-type]

    def hot_of(i: int) -> int:
        if child_hotspots is not None:
            return child_hotspots[i]
        return analysis.hotspot_count(candidates[i].new_boxes)

    direct = [
        i
        for i, c in enumerate(candidates)
        if c.push_from == hottest
        and child_feats[i].connectivity <= parent_feat.connectivity
    ]
    if direct:
        return min(direct, key=hot_of)

    hr, hc = hottest
    nearby = {
        (hr - 1, hc),
        (hr + 1, hc),
        (hr, hc - 1),
        (hr, hc + 1),
        (hr - 1, hc - 1),
        (hr - 1, hc + 1),
        (hr + 1, hc - 1),
        (hr + 1, hc + 1),
    }
    clearing = [
        i
        for i, c in enumerate(candidates)
        if c.push_from in nearby
        and c.push_from != hottest
        and child_feats[i].connectivity <= parent_feat.connectivity
    ]
    if not clearing:
        return None
    return min(clearing, key=hot_of)
