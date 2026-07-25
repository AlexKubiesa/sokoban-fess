"""Packing plan via backward FESS (CoG §III-B packing / OOP).

Pull boxes from the solved position toward a sink basin. Feature space is
(boxes_on_board, boxes_on_targets); the packing-plan advisor prefers long pulls
from targets under push-accessibility constraints. The reversed removal /
parking path becomes the forward packing order.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import NamedTuple

from sokoban_fess.level import ACTION_DELTAS, ACTIONS, Level
from sokoban_fess.solver import Boxes, Cell, _is_open

from sokoban_fess.fess.rooms import RoomLayout


class PackFeature(NamedTuple):
    """Backward-search feature cell: minimize both axes."""

    boxes_on_board: int
    boxes_on_targets: int


@dataclass(frozen=True)
class PackingPlan:
    """Forward packing/parking order plus per-step OK zones for OOP."""

    order: tuple[Cell, ...]
    # Targets subset for quick membership.
    target_set: frozenset[Cell]
    # Primary sink basin (fallback OK zone).
    basin: frozenset[Cell]
    sink_room: int | None
    # ok_zones[k] = cells that are safe for unpacked boxes when packed_count==k.
    ok_zones: tuple[frozenset[Cell], ...]

    def packed_count(self, boxes: Boxes) -> int:
        """How many prefix plan steps are occupied (parked/packed)."""
        count = 0
        for cell in self.order:
            if cell in boxes:
                count += 1
            else:
                break
        return count

    def oop_count(self, boxes: Boxes) -> int:
        """Boxes neither plan-packed nor in the OK zone for the current step."""
        step = self.packed_count(boxes)
        packed_cells = set(self.order[:step])
        if self.ok_zones:
            zone = self.ok_zones[min(step, len(self.ok_zones) - 1)]
        else:
            zone = self.basin
        oop = 0
        for box in boxes:
            if box in packed_cells:
                continue
            if box in zone:
                continue
            oop += 1
        return oop


def compute_basin(
    walls: Boxes,
    targets: frozenset[Cell],
    goal: Cell,
    height: int,
    width: int,
) -> frozenset[Cell]:
    """Squares from which a box can reach ``goal`` with all targets occupied."""
    frozen = targets - {goal}
    if goal not in targets or not _is_open(goal, walls, height, width):
        return frozenset()
    dist: dict[Cell, int] = {goal: 0}
    queue: deque[Cell] = deque([goal])
    while queue:
        curr = queue.popleft()
        c_row, c_col = curr
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            prev = (c_row - d_row, c_col - d_col)
            stand = (prev[0] - d_row, prev[1] - d_col)
            if prev in dist:
                continue
            if not _is_open(prev, walls, height, width) or prev in frozen:
                continue
            if not _is_open(stand, walls, height, width) or stand in frozen:
                continue
            dist[prev] = dist[curr] + 1
            queue.append(prev)
    return frozenset(dist.keys())


def _nearest_target_distance(
    walls: Boxes, targets: frozenset[Cell], height: int, width: int
) -> dict[Cell, int]:
    dist: dict[Cell, int] = {}
    queue: deque[Cell] = deque()
    for t in targets:
        if _is_open(t, walls, height, width):
            dist[t] = 0
            queue.append(t)
    while queue:
        curr = queue.popleft()
        c_row, c_col = curr
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            prev = (c_row - d_row, c_col - d_col)
            stand = (prev[0] - d_row, prev[1] - d_col)
            if prev in dist:
                continue
            if not _is_open(prev, walls, height, width):
                continue
            if not _is_open(stand, walls, height, width):
                continue
            dist[prev] = dist[curr] + 1
            queue.append(prev)
    return dist


def _pull_destinations(
    walls: Boxes,
    boxes: set[Cell],
    box: Cell,
    height: int,
    width: int,
) -> list[tuple[Cell, int]]:
    """(dest, pull_steps) for pulls of ``box`` with others frozen."""
    frozen = boxes - {box}
    reach: dict[Cell, int] = {box: 0}
    queue: deque[Cell] = deque([box])
    while queue:
        curr = queue.popleft()
        c_row, c_col = curr
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            dest = (c_row + d_row, c_col + d_col)
            stand = (dest[0] + d_row, dest[1] + d_col)
            if dest in reach:
                continue
            if not _is_open(dest, walls, height, width) or dest in frozen:
                continue
            if not _is_open(stand, walls, height, width) or stand in frozen:
                continue
            reach[dest] = reach[curr] + 1
            queue.append(dest)
    return [(c, d) for c, d in reach.items() if c != box]


def _push_reachable_empty(
    walls: Boxes, start: Cell, height: int, width: int
) -> set[Cell]:
    """Cells a box at ``start`` can reach by pushes on the empty board."""
    if not _is_open(start, walls, height, width):
        return set()
    seen = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        curr = queue.popleft()
        c_row, c_col = curr
        for action in ACTIONS:
            d_row, d_col = ACTION_DELTAS[action]
            dest = (c_row + d_row, c_col + d_col)
            stand = (c_row - d_row, c_col - d_col)
            if dest in seen:
                continue
            if not _is_open(dest, walls, height, width):
                continue
            if not _is_open(stand, walls, height, width):
                continue
            seen.add(dest)
            queue.append(dest)
    return seen


def _connectivity_free(
    walls: Boxes, boxes: set[Cell], height: int, width: int
) -> int:
    remaining = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in walls and (r, c) not in boxes
    }
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            pos = stack.pop()
            r, c = pos
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return components


class _PendingPack(NamedTuple):
    weight: int
    boxes_on_board: int
    boxes_on_targets: int
    neg_distance: int
    tie: int
    parent_id: int
    new_boxes: frozenset[Cell]
    removed_target: Cell | None  # target cleared via sink, if any
    parked: Cell | None  # parking square discovered


@dataclass
class _PackNode:
    boxes: frozenset[Cell]
    feature: PackFeature
    weight: int
    parent_id: int | None
    removed_target: Cell | None
    parked: Cell | None
    true_feature: PackFeature


def _build_ok_zones(
    level: Level,
    order: tuple[Cell, ...],
    primary_basin: frozenset[Cell],
) -> tuple[frozenset[Cell], ...]:
    """Per packing-step OK zones for the OOP feature (simplified).

    At step k, parked/packed cells from order[:k] act as walls; cells that can
    still push to any remaining plan target form the OK zone (union primary
    basin).
    """
    walls = level.walls
    height, width = level.height, level.width
    targets = frozenset(level.targets)
    zones: list[frozenset[Cell]] = []
    for step in range(len(order) + 1):
        parked = set(order[:step])
        remaining = [c for c in order[step:] if c in targets]
        if not remaining:
            zones.append(frozenset(primary_basin) | parked)
            continue
        blocked = walls | parked
        ok: set[Cell] = set(primary_basin)
        for goal in remaining:
            # Reverse push BFS to goal with parked as walls.
            if not _is_open(goal, blocked, height, width):
                continue
            dist: dict[Cell, int] = {goal: 0}
            queue: deque[Cell] = deque([goal])
            while queue:
                curr = queue.popleft()
                c_row, c_col = curr
                for action in ACTIONS:
                    d_row, d_col = ACTION_DELTAS[action]
                    prev = (c_row - d_row, c_col - d_col)
                    stand = (prev[0] - d_row, prev[1] - d_col)
                    if prev in dist:
                        continue
                    if not _is_open(prev, blocked, height, width):
                        continue
                    if not _is_open(stand, blocked, height, width):
                        continue
                    dist[prev] = dist[curr] + 1
                    queue.append(prev)
            ok |= dist.keys()
        zones.append(frozenset(ok))
    return tuple(zones)


def build_packing_plan(
    level: Level,
    rooms: RoomLayout,
    *,
    max_expansions: int = 2_000,
) -> PackingPlan:
    """Backward FESS packing/parking plan (CoG packing-order pre-search)."""
    walls = level.walls
    height, width = level.height, level.width
    targets = frozenset(level.targets)
    target_list = tuple(sorted(targets))
    start_boxes = frozenset(level.boxes)

    basins = {
        t: compute_basin(walls, targets, t, height, width) for t in target_list
    }
    best_target = target_list[0] if target_list else None
    best_count = -1
    for t, basin in basins.items():
        count = sum(1 for b in level.boxes if b in basin)
        if count > best_count:
            best_count = count
            best_target = t
    if best_target is not None and basins.get(best_target):
        primary_basin: frozenset[Cell] = basins[best_target]
    elif basins:
        primary_basin = frozenset().union(*basins.values())
    else:
        primary_basin = frozenset(targets)

    sink_room: int | None = None
    if rooms.n_rooms and primary_basin:
        room_hits: dict[int, int] = {}
        for cell in primary_basin:
            rid = rooms.room_of(cell)
            if rid is not None:
                room_hits[rid] = room_hits.get(rid, 0) + 1
        if room_hits:
            sink_room = max(room_hits.items(), key=lambda kv: kv[1])[0]

    def in_sink(cell: Cell) -> bool:
        if sink_room is not None and rooms.room_of(cell) == sink_room:
            return True
        return cell in primary_basin and cell not in targets

    near_dist = _nearest_target_distance(walls, targets, height, width)
    # Empty-board push reachability for parking accessibility.
    push_from_start = {
        b: _push_reachable_empty(walls, b, height, width) for b in start_boxes
    }
    all_push_accessible = set().union(*push_from_start.values()) if push_from_start else set()

    def parking_ok(dest: Cell) -> bool:
        if dest in start_boxes:
            return True
        return dest in all_push_accessible

    def pack_feat(boxes: frozenset[Cell]) -> PackFeature:
        return PackFeature(
            boxes_on_board=len(boxes),
            boxes_on_targets=sum(1 for b in boxes if b in targets),
        )

    def is_pack_progress(parent: PackFeature, child: PackFeature) -> bool:
        if child.boxes_on_board < parent.boxes_on_board:
            return True
        return (
            child.boxes_on_board == parent.boxes_on_board
            and child.boxes_on_targets < parent.boxes_on_targets
        )

    # --- Mini FESS in pull mode ---
    root_boxes = frozenset(targets)
    nodes: list[_PackNode] = []
    pending: dict[PackFeature, list[_PendingPack]] = defaultdict(list)
    cell_order: list[PackFeature] = []
    seen: dict[frozenset[Cell], int] = {}
    tie = 0

    def ensure_cell(cell: PackFeature) -> None:
        if cell not in pending:
            pending[cell] = []
            cell_order.append(cell)

    def distance_score(boxes: frozenset[Cell]) -> int:
        return sum(near_dist.get(b, 0) for b in boxes)

    def enqueue(node_id: int) -> None:
        nonlocal tie
        node = nodes[node_id]
        boxes_set = set(node.boxes)
        parent_conn = _connectivity_free(walls, boxes_set, height, width)
        candidates: list[tuple[frozenset[Cell], Cell | None, Cell | None, bool]] = []

        # Enumerate pulls.
        for box in list(boxes_set):
            for dest, _steps in _pull_destinations(
                walls, boxes_set, box, height, width
            ):
                # Sink removal (rule 1).
                if box in targets and in_sink(dest):
                    new_boxes = frozenset(boxes_set - {box})
                    candidates.append((new_boxes, box, None, True))
                    continue
                # Ordinary pull.
                new_set = (boxes_set - {box}) | {dest}
                new_boxes = frozenset(new_set)
                parked = dest if box in targets and dest not in targets else None
                if parked is not None and not parking_ok(dest):
                    # Still legal, but not advisor-endorsed (restriction).
                    candidates.append((new_boxes, None, parked, False))
                else:
                    # Connectivity must not worsen for advisor pulls from targets.
                    advise = False
                    if box in targets:
                        new_conn = _connectivity_free(
                            walls, new_set, height, width
                        )
                        if new_conn <= parent_conn and (
                            parked is None or parking_ok(dest)
                        ):
                            advise = True
                    candidates.append((new_boxes, None, parked, advise))

        # Packing-plan advisor: among advisor-eligible, prefer farthest pull
        # from a target (max near_dist of dest / removal).
        advisor_idxs = [i for i, c in enumerate(candidates) if c[3]]
        endorsed: set[int] = set()
        if advisor_idxs:
            def adv_score(i: int) -> int:
                nb, rem, park, _ = candidates[i]
                if rem is not None:
                    return 10_000  # removals win
                if park is not None:
                    return near_dist.get(park, 0)
                # farthest box distance sum improvement
                return distance_score(nb)

            best_i = max(advisor_idxs, key=adv_score)
            endorsed.add(best_i)

        ensure_cell(node.feature)
        for i, (new_boxes, rem, park, _) in enumerate(candidates):
            if new_boxes in seen and seen[new_boxes] <= node.weight + (
                0 if i in endorsed else 1
            ):
                # Still enqueue if better weight possible — handled at expand.
                pass
            child_feat = pack_feat(new_boxes)
            delta = 0 if i in endorsed else 1
            weight = node.weight + delta
            projected = (
                child_feat
                if is_pack_progress(node.feature, child_feat)
                else node.feature
            )
            ensure_cell(projected)
            # Queue under parent's feature cell (FESS).
            tie += 1
            heapq.heappush(
                pending[node.feature],
                _PendingPack(
                    weight=weight,
                    boxes_on_board=child_feat.boxes_on_board,
                    boxes_on_targets=child_feat.boxes_on_targets,
                    neg_distance=-distance_score(new_boxes),
                    tie=tie,
                    parent_id=node_id,
                    new_boxes=new_boxes,
                    removed_target=rem,
                    parked=park,
                ),
            )

    root_feat = pack_feat(root_boxes)
    nodes.append(
        _PackNode(
            boxes=root_boxes,
            feature=root_feat,
            weight=0,
            parent_id=None,
            removed_target=None,
            parked=None,
            true_feature=root_feat,
        )
    )
    seen[root_boxes] = 0
    ensure_cell(root_feat)
    enqueue(0)

    expansions = 0
    cell_cursor = 0
    best_id = 0

    while expansions < max_expansions and any(pending.values()):
        move: _PendingPack | None = None
        tried = 0
        while tried < len(cell_order):
            cell = cell_order[cell_cursor % len(cell_order)]
            cell_cursor += 1
            tried += 1
            heap = pending.get(cell)
            if not heap:
                continue
            move = heapq.heappop(heap)
            break
        if move is None:
            break

        parent = nodes[move.parent_id]
        true_feat = pack_feat(move.new_boxes)
        # Best-ancestor projection for packing FESS.
        best_true = parent.true_feature
        best_proj = parent.feature
        cur: int | None = move.parent_id
        while cur is not None:
            n = nodes[cur]
            if (n.true_feature.boxes_on_board, n.true_feature.boxes_on_targets) < (
                best_true.boxes_on_board,
                best_true.boxes_on_targets,
            ):
                best_true = n.true_feature
                best_proj = n.feature
            cur = n.parent_id
        if is_pack_progress(best_true, true_feat):
            projected = true_feat
        else:
            projected = best_proj

        prev = seen.get(move.new_boxes)
        if prev is not None and move.weight >= prev:
            continue
        seen[move.new_boxes] = move.weight
        node_id = len(nodes)
        nodes.append(
            _PackNode(
                boxes=move.new_boxes,
                feature=projected,
                weight=move.weight,
                parent_id=move.parent_id,
                removed_target=move.removed_target,
                parked=move.parked,
                true_feature=true_feat,
            )
        )
        expansions += 1
        enqueue(node_id)

        # Track best: fewest boxes, then farthest from targets.
        best = nodes[best_id]
        cand = nodes[node_id]
        if cand.true_feature.boxes_on_board < best.true_feature.boxes_on_board or (
            cand.true_feature.boxes_on_board == best.true_feature.boxes_on_board
            and distance_score(cand.boxes) > distance_score(best.boxes)
        ):
            best_id = node_id

        if not cand.boxes:
            best_id = node_id
            break

    # Reconstruct parking/removal events along path to best.
    order_rev: list[Cell] = []
    cur_id: int | None = best_id
    while cur_id is not None:
        node = nodes[cur_id]
        if node.removed_target is not None:
            order_rev.append(node.removed_target)
        elif node.parked is not None:
            order_rev.append(node.parked)
        cur_id = node.parent_id

    parks = [c for c in reversed(order_rev) if c not in targets]
    packed_targets = [c for c in order_rev if c in targets]
    # order_rev was root←best, so removals are in reverse packing order.
    # parks discovered going backward appear in reverse; reverse parks for forward.
    parks = list(reversed([c for c in order_rev if c not in targets]))
    packed_targets = [c for c in order_rev if c in targets]  # already reverse-pull order = forward pack

    parks = [c for c in parks if c not in start_boxes]
    seen_cells: set[Cell] = set()
    order: list[Cell] = []
    for cell in parks + packed_targets:
        if cell not in seen_cells:
            seen_cells.add(cell)
            order.append(cell)
    for t in target_list:
        if t not in seen_cells:
            order.append(t)
            seen_cells.add(t)

    order_t = tuple(order)
    trial = PackingPlan(
        order=order_t,
        target_set=targets,
        basin=primary_basin if primary_basin else frozenset(targets),
        sink_room=sink_room,
        ok_zones=(),
    )
    if trial.packed_count(start_boxes) >= len(targets) and not (
        start_boxes <= targets
    ):
        order_t = target_list
        trial = PackingPlan(
            order=order_t,
            target_set=targets,
            basin=primary_basin if primary_basin else frozenset(targets),
            sink_room=sink_room,
            ok_zones=(),
        )

    ok_zones = _build_ok_zones(level, trial.order, trial.basin)
    return PackingPlan(
        order=trial.order,
        target_set=targets,
        basin=trial.basin,
        sink_room=trial.sink_room,
        ok_zones=ok_zones,
    )


def simple_target_plan(level: Level) -> PackingPlan:
    """Fallback plan: pack targets in sorted order."""
    order = tuple(sorted(level.targets))
    open_cells = frozenset(
        (r, c)
        for r in range(level.height)
        for c in range(level.width)
        if (r, c) not in level.walls
    )
    plan = PackingPlan(
        order=order,
        target_set=frozenset(level.targets),
        basin=open_cells,
        sink_room=None,
        ok_zones=(),
    )
    ok = _build_ok_zones(level, plan.order, plan.basin)
    return PackingPlan(
        order=plan.order,
        target_set=plan.target_set,
        basin=plan.basin,
        sink_room=None,
        ok_zones=ok,
    )
