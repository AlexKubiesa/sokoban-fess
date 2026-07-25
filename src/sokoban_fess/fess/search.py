"""Feature Space Search core loop (CoG Fig. 2).

Maintains a domain-space search tree and a projection onto feature space.
Cyclically visits FS cells; from each cell expands the least-weight pending
macro. Advisor macros add weight 0; others add 1.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import NamedTuple

from sokoban_fess.level import ACTION_CHARS, ACTION_DELTAS, ACTIONS, Level
from sokoban_fess.solver import (
    _INF,
    AdvisorName,
    Boxes,
    Cell,
    FeatureCoords,
    SearchEvent,
    SearchEventKind,
    SearchResult,
    StateKey,
    StopReason,
)

from sokoban_fess.fess.advisors import endorse_macros
from sokoban_fess.fess.features import (
    FeatureCell,
    LevelAnalysis,
    get_analysis,
    goal_direction_key,
    is_feature_progress,
)
from sokoban_fess.fess.macros import MacroCandidate, legal_macros as _legal_macros

# Advisor-endorsed macros add 0; all others add this (CoG best setting).
ADVISOR_MOVE_WEIGHT = 0
DIFFICULT_MOVE_WEIGHT = 1


class MacroLink(NamedTuple):
    """How we reached a child via one macro move from its parent."""

    prev_key: StateKey
    moves: str
    push_from: Cell
    push_to: Cell
    push_count: int
    push_path: tuple[Cell, ...]


class PendingMove(NamedTuple):
    """Unexpanded macro queued under a parent feature cell.

    Ordering (min-heap): weight, then CoG lex features, then pushes.
    """

    weight: int
    oop: int
    neg_packed: int
    connectivity: int
    room_connectivity: int
    hotspots: int
    neg_mobility: int
    pushes: int
    tie: int
    parent_id: int
    push_from: Cell
    push_to: Cell
    moves: str
    push_count: int
    push_path: tuple[Cell, ...]
    new_boxes: Boxes
    new_player: Cell
    advisors: tuple[AdvisorName, ...]


@dataclass
class _Node:
    """One position in the FESS search tree."""

    boxes: Boxes
    player: Cell
    key: StateKey
    weight: int
    pushes: int
    feature: FeatureCell  # projected FS cell
    true_feature: FeatureCell  # actual feature values
    parent_link: MacroLink | None
    parent_id: int | None
    dead: bool = False
    child_ids: list[int] | None = None
    pending_generated: bool = False


class _FessSearcher:
    """FESS over box→cell macro moves for one level."""

    def __init__(
        self,
        level: Level,
        *,
        max_states: int,
        collect_trace: bool,
        max_events: int,
        max_time: float | None = None,
        analysis: LevelAnalysis | None = None,
    ) -> None:
        self.level = level
        self.walls = level.walls
        self.targets = level.targets
        self.height = level.height
        self.width = level.width
        self.max_states = max_states
        self.max_time = max_time
        self.collect_trace = collect_trace
        self.max_events = max_events
        self._deadline: float | None = None

        self.analysis = analysis or get_analysis(level)

        self.events: list[SearchEvent] = []
        self.truncated_trace = False
        self.nodes: list[_Node] = []
        self.expand_index: dict[int, int] = {}
        self.seen: dict[StateKey, int] = {}
        self.pending: dict[FeatureCell, list[PendingMove]] = defaultdict(list)
        self.cell_order: list[FeatureCell] = []
        self.tie = 0
        # Live pending-move counts per parent (for dead-end detection).
        self._open_children: dict[int, int] = defaultdict(int)
        self._feature_cache: dict[Boxes, FeatureCell] = {}
        self._hotspot_cache: dict[Boxes, int] = {}
        self._mobility_cache: dict[tuple[Boxes, Cell], int] = {}
        # Stamped flood-fill buffers (avoid allocating a set each BFS).
        self._visit = [[0] * self.width for _ in range(self.height)]
        self._stamp = 0
        self._reach_buf: list[Cell] = []

    def features(self, boxes: Boxes) -> FeatureCell:
        cached = self._feature_cache.get(boxes)
        if cached is not None:
            return cached
        feat = self.analysis.features(boxes)
        if len(self._feature_cache) > 50_000:
            self._feature_cache.clear()
        self._feature_cache[boxes] = feat
        return feat

    def hotspot_count(self, boxes: Boxes) -> int:
        cached = self._hotspot_cache.get(boxes)
        if cached is not None:
            return cached
        value = self.analysis.hotspot_count(boxes)
        if len(self._hotspot_cache) > 50_000:
            self._hotspot_cache.clear()
        self._hotspot_cache[boxes] = value
        return value

    def mobility(self, boxes: Boxes, player: Cell) -> int:
        key = (boxes, player)
        cached = self._mobility_cache.get(key)
        if cached is not None:
            return cached
        value = self.analysis.mobility(boxes, player)
        if len(self._mobility_cache) > 50_000:
            self._mobility_cache.clear()
        self._mobility_cache[key] = value
        return value

    # --- board helpers -------------------------------------------------

    def is_blocked(self, pos: Cell, obstacles: Boxes) -> bool:
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            return True
        return pos in self.walls or pos in obstacles

    def reachable_cells(self, player: Cell, obstacles: Boxes) -> set[Cell]:
        self._stamp += 1
        if self._stamp > 2_000_000_000:
            for row in self._visit:
                for i in range(len(row)):
                    row[i] = 0
            self._stamp = 1
        stamp = self._stamp
        stack = [player]
        out: set[Cell] = set()
        while stack:
            pos = stack.pop()
            row, col = pos
            if not (0 <= row < self.height and 0 <= col < self.width):
                continue
            if self._visit[row][col] == stamp:
                continue
            if pos in self.walls or pos in obstacles:
                continue
            self._visit[row][col] = stamp
            out.add(pos)
            stack.extend(
                (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
            )
        return out

    def _reachable_list(self, player: Cell, obstacles: Boxes) -> list[Cell]:
        """Like ``reachable_cells`` but reuses a list buffer."""
        self._stamp += 1
        if self._stamp > 2_000_000_000:
            for row in self._visit:
                for i in range(len(row)):
                    row[i] = 0
            self._stamp = 1
        stamp = self._stamp
        stack = [player]
        self._reach_buf.clear()
        while stack:
            pos = stack.pop()
            row, col = pos
            if not (0 <= row < self.height and 0 <= col < self.width):
                continue
            if self._visit[row][col] == stamp:
                continue
            if pos in self.walls or pos in obstacles:
                continue
            self._visit[row][col] = stamp
            self._reach_buf.append(pos)
            stack.extend(
                (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
            )
        return self._reach_buf

    def state_key(self, boxes: Boxes, player: Cell) -> StateKey:
        cells = self._reachable_list(player, boxes)
        return StateKey(boxes, min(cells))

    def legal_macros(self, boxes: Boxes, player: Cell) -> list[MacroCandidate]:
        """Public wrapper used by tests / educational exploration."""
        return _legal_macros(self, boxes, player)

    def walk_path(self, start: Cell, goal: Cell, obstacles: Boxes) -> str | None:
        if start == goal:
            return ""
        if self.is_blocked(goal, obstacles):
            return None
        came_from: dict[Cell, tuple[Cell, int]] = {}
        queue: deque[Cell] = deque([start])
        seen = {start}
        while queue:
            pos = queue.popleft()
            row, col = pos
            for action in ACTIONS:
                d_row, d_col = ACTION_DELTAS[action]
                nxt = (row + d_row, col + d_col)
                if nxt in seen or self.is_blocked(nxt, obstacles):
                    continue
                came_from[nxt] = (pos, action)
                if nxt == goal:
                    chars: list[str] = []
                    cur = nxt
                    while cur != start:
                        parent_pos, act = came_from[cur]
                        chars.append(ACTION_CHARS[act])
                        cur = parent_pos
                    chars.reverse()
                    return "".join(chars)
                seen.add(nxt)
                queue.append(nxt)
        return None

    def project_cell(
        self, parent_id: int | None, child_true: FeatureCell
    ) -> FeatureCell:
        """Project onto FS, pinning regressions to the best ancestor (CoG Fig. 2)."""
        if parent_id is None:
            return child_true
        best_true: FeatureCell | None = None
        best_projected: FeatureCell | None = None
        cur: int | None = parent_id
        while cur is not None:
            node = self.nodes[cur]
            if best_true is None or goal_direction_key(
                node.true_feature
            ) < goal_direction_key(best_true):
                best_true = node.true_feature
                best_projected = node.feature
            cur = node.parent_id
        assert best_true is not None and best_projected is not None
        if is_feature_progress(best_true, child_true):
            return child_true
        return best_projected

    @staticmethod
    def _push_char(moves: str) -> str:
        return moves[-1] if moves else "?"

    @staticmethod
    def _coords(feat: FeatureCell) -> FeatureCoords:
        return (
            feat.packed,
            feat.connectivity,
            feat.room_connectivity,
            feat.oop,
        )

    def _event_features(
        self, boxes: Boxes, parent_id: int | None
    ) -> tuple[FeatureCoords, FeatureCoords]:
        """True and projected feature coordinates for a board layout."""
        true = self.features(boxes)
        projected = self.project_cell(parent_id, true)
        return self._coords(true), self._coords(projected)

    # --- trace helpers -------------------------------------------------

    def emit(self, event: SearchEvent) -> int | None:
        if not self.collect_trace:
            return None
        if len(self.events) >= self.max_events:
            self.truncated_trace = True
            return None
        self.events.append(event)
        return len(self.events) - 1

    def finish(
        self,
        solution: str | None,
        kind: SearchEventKind,
        boxes: Boxes,
        player: Cell,
        pushes: int,
        weight: int,
        *,
        parent_index: int | None = None,
        push_from: Cell | None = None,
        push_to: Cell | None = None,
        push_char: str | None = None,
        push_path: tuple[Cell, ...] | None = None,
        advisors: tuple[AdvisorName, ...] = (),
        open_len: int | None = None,
        features: FeatureCoords | None = None,
        projected: FeatureCoords | None = None,
        parent_id: int | None = None,
    ) -> SearchResult:
        if kind not in ("goal", "capped", "timeout", "exhausted"):
            raise ValueError(f"finish() requires a terminal kind, got {kind!r}")
        if features is None or projected is None:
            features, projected = self._event_features(boxes, parent_id)
        self.emit(
            SearchEvent(
                kind=kind,
                boxes=boxes,
                player=player,
                pushes=pushes,
                weight=weight,
                open_len=self._pending_count() if open_len is None else open_len,
                closed_len=len(self.nodes),
                push_from=push_from,
                push_to=push_to,
                push_char=push_char,
                parent_index=parent_index,
                push_path=push_path,
                advisors=advisors,
                features=features,
                projected=projected,
            )
        )
        assert kind in ("goal", "capped", "timeout", "exhausted")
        return SearchResult(
            solution=solution,
            events=tuple(self.events),
            states_visited=len(self.seen),
            truncated_trace=self.truncated_trace,
            stop_reason=kind,
            algorithm="fess",
        )

    def _pending_count(self) -> int:
        return sum(len(heap) for heap in self.pending.values())

    def _ensure_cell(self, cell: FeatureCell) -> None:
        if cell not in self.pending:
            self.pending[cell] = []
            self.cell_order.append(cell)

    def reconstruct_solution(self, node_id: int) -> str:
        parts: list[str] = []
        cur: int | None = node_id
        while cur is not None:
            node = self.nodes[cur]
            link = node.parent_link
            if link is None:
                break
            parts.append(link.moves)
            cur = node.parent_id
        parts.reverse()
        return "".join(parts)

    def _mark_dead(self, node_id: int) -> None:
        node = self.nodes[node_id]
        if node.dead:
            return
        node.dead = True
        parent_id = node.parent_id
        if parent_id is None:
            return
        parent = self.nodes[parent_id]
        if parent.child_ids is None:
            return
        # If all children dead and no open pending from parent, mark parent.
        if self._open_children.get(parent_id, 0) > 0:
            return
        if parent.child_ids and all(self.nodes[c].dead for c in parent.child_ids):
            if parent.pending_generated:
                self._mark_dead(parent_id)

    # --- search tree ---------------------------------------------------

    def _enqueue_moves(self, node_id: int) -> None:
        node = self.nodes[node_id]
        candidates = self.legal_macros(node.boxes, node.player)
        node.pending_generated = True
        if not candidates:
            self._mark_dead(node_id)
            return

        endorsed = endorse_macros(
            self.analysis, self, node.boxes, node.player, candidates
        )
        state_rep = self.expand_index.get(node_id)
        live = 0

        for macro in candidates:
            push_char = self._push_char(macro.moves)
            advisors = endorsed.get((macro.push_from, macro.push_to), ())
            if self.analysis.has_deadlock(macro.new_boxes):
                true_f, proj_f = self._event_features(macro.new_boxes, node_id)
                self.emit(
                    SearchEvent(
                        kind="deadlock",
                        boxes=macro.new_boxes,
                        player=macro.push_from,
                        pushes=node.pushes + macro.push_count,
                        weight=-1,
                        open_len=self._pending_count(),
                        closed_len=len(self.nodes),
                        push_from=macro.push_from,
                        push_to=macro.push_to,
                        push_char=push_char,
                        parent_index=state_rep,
                        push_path=macro.push_path,
                        advisors=advisors,
                        features=true_f,
                        projected=proj_f,
                    )
                )
                continue

            child_feat = self.features(macro.new_boxes)
            true_f = self._coords(child_feat)
            proj_f = self._coords(self.project_cell(node_id, child_feat))
            delta = (
                ADVISOR_MOVE_WEIGHT if advisors else DIFFICULT_MOVE_WEIGHT
            )
            weight = node.weight + delta
            total_pushes = node.pushes + macro.push_count
            hotspots = self.hotspot_count(macro.new_boxes)
            mobility = self.mobility(macro.new_boxes, macro.new_player)
            self.tie += 1
            self._ensure_cell(node.feature)
            heapq.heappush(
                self.pending[node.feature],
                PendingMove(
                    weight=weight,
                    oop=child_feat.oop,
                    neg_packed=-child_feat.packed,
                    connectivity=child_feat.connectivity,
                    room_connectivity=child_feat.room_connectivity,
                    hotspots=hotspots,
                    neg_mobility=-mobility,
                    pushes=total_pushes,
                    tie=self.tie,
                    parent_id=node_id,
                    push_from=macro.push_from,
                    push_to=macro.push_to,
                    moves=macro.moves,
                    push_count=macro.push_count,
                    push_path=macro.push_path,
                    new_boxes=macro.new_boxes,
                    new_player=macro.new_player,
                    advisors=advisors,
                ),
            )
            live += 1
            self.emit(
                SearchEvent(
                    kind="enqueue",
                    boxes=macro.new_boxes,
                    player=macro.new_player,
                    pushes=total_pushes,
                    weight=weight,
                    open_len=self._pending_count(),
                    closed_len=len(self.nodes),
                    push_from=macro.push_from,
                    push_to=macro.push_to,
                    push_char=push_char,
                    parent_index=state_rep,
                    push_path=macro.push_path,
                    advisors=advisors,
                    features=true_f,
                    projected=proj_f,
                )
            )

        self._open_children[node_id] = live
        if live == 0:
            self._mark_dead(node_id)

    def _add_node(
        self,
        boxes: Boxes,
        player: Cell,
        weight: int,
        pushes: int,
        feature: FeatureCell,
        true_feature: FeatureCell,
        parent_link: MacroLink | None,
        parent_id: int | None,
    ) -> int | None:
        key = self.state_key(boxes, player)
        best = self.seen.get(key)
        if best is not None and weight >= best:
            return None
        self.seen[key] = weight
        node_id = len(self.nodes)
        self.nodes.append(
            _Node(
                boxes=boxes,
                player=player,
                key=key,
                weight=weight,
                pushes=pushes,
                feature=feature,
                true_feature=true_feature,
                parent_link=parent_link,
                parent_id=parent_id,
                child_ids=[],
            )
        )
        if parent_id is not None:
            parent = self.nodes[parent_id]
            if parent.child_ids is not None:
                parent.child_ids.append(node_id)
        self._ensure_cell(feature)
        return node_id

    def _pop_best_from_cell(self, cell: FeatureCell) -> PendingMove | None:
        heap = self.pending.get(cell)
        if not heap:
            return None
        while heap:
            move = heapq.heappop(heap)
            parent = self.nodes[move.parent_id]
            if parent.dead:
                self._open_children[move.parent_id] = max(
                    0, self._open_children[move.parent_id] - 1
                )
                continue
            if self.seen.get(parent.key, _INF) < parent.weight:
                self._open_children[move.parent_id] = max(
                    0, self._open_children[move.parent_id] - 1
                )
                continue
            self._open_children[move.parent_id] = max(
                0, self._open_children[move.parent_id] - 1
            )
            return move
        return None

    def _expand_move(self, move: PendingMove) -> SearchResult | None:
        parent = self.nodes[move.parent_id]
        if parent.dead:
            return None
        parent_rep = self.expand_index.get(move.parent_id)

        child_true = self.features(move.new_boxes)
        projected = self.project_cell(move.parent_id, child_true)
        push_char = self._push_char(move.moves)
        link = MacroLink(
            prev_key=parent.key,
            moves=move.moves,
            push_from=move.push_from,
            push_to=move.push_to,
            push_count=move.push_count,
            push_path=move.push_path,
        )
        child_id = self._add_node(
            move.new_boxes,
            move.new_player,
            move.weight,
            move.pushes,
            projected,
            child_true,
            link,
            move.parent_id,
        )
        if child_id is None:
            # Transposition / dominated — treat as dead child for parent.
            if (
                parent.pending_generated
                and self._open_children.get(move.parent_id, 0) == 0
                and parent.child_ids is not None
                and all(self.nodes[c].dead for c in parent.child_ids)
            ):
                self._mark_dead(move.parent_id)
            return None

        expand_idx = self.emit(
            SearchEvent(
                kind="expand",
                boxes=move.new_boxes,
                player=move.new_player,
                pushes=move.pushes,
                weight=move.weight,
                open_len=self._pending_count(),
                closed_len=len(self.nodes),
                push_from=move.push_from,
                push_to=move.push_to,
                push_char=push_char,
                parent_index=parent_rep,
                push_path=move.push_path,
                advisors=move.advisors,
                features=self._coords(child_true),
                projected=self._coords(projected),
            )
        )
        if expand_idx is not None:
            self.expand_index[child_id] = expand_idx

        if move.new_boxes <= self.targets:
            return self.finish(
                self.reconstruct_solution(child_id),
                "goal",
                move.new_boxes,
                move.new_player,
                move.pushes,
                move.weight,
                parent_index=expand_idx if expand_idx is not None else parent_rep,
                push_from=move.push_from,
                push_to=move.push_to,
                push_char=push_char,
                push_path=move.push_path,
                advisors=move.advisors,
                features=self._coords(child_true),
                projected=self._coords(projected),
                parent_id=child_id,
            )

        self._enqueue_moves(child_id)
        return None

    def _budget_stop(self, kind: StopReason) -> SearchResult:
        boxes: Boxes = frozenset(self.level.boxes)
        player = self.level.player
        pushes = 0
        weight = 0
        parent_index: int | None = self.expand_index.get(0)
        parent_id: int | None = None
        if self.nodes:
            node = self.nodes[-1]
            boxes, player = node.boxes, node.player
            pushes, weight = node.pushes, node.weight
            parent_id = len(self.nodes) - 1
            parent_index = self.expand_index.get(parent_id, parent_index)
        return self.finish(
            None,
            kind,
            boxes,
            player,
            pushes,
            weight,
            parent_index=parent_index,
            parent_id=parent_id,
        )

    def run(self) -> SearchResult:
        start_boxes = frozenset(self.level.boxes)
        start_player = self.level.player
        start_feat = self.features(start_boxes)
        start_coords = self._coords(start_feat)

        if start_boxes <= self.targets:
            start_idx = self.emit(
                SearchEvent(
                    kind="start",
                    boxes=start_boxes,
                    player=start_player,
                    pushes=0,
                    weight=0,
                    open_len=0,
                    closed_len=1,
                    features=start_coords,
                    projected=start_coords,
                )
            )
            self.seen[self.state_key(start_boxes, start_player)] = 0
            return self.finish(
                "",
                "goal",
                start_boxes,
                start_player,
                0,
                0,
                parent_index=start_idx,
                features=start_coords,
                projected=start_coords,
            )

        if self.analysis.has_deadlock(start_boxes):
            self.emit(
                SearchEvent(
                    kind="start",
                    boxes=start_boxes,
                    player=start_player,
                    pushes=0,
                    weight=-1,
                    open_len=0,
                    closed_len=0,
                    features=start_coords,
                    projected=start_coords,
                )
            )
            return self.finish(
                None,
                "exhausted",
                start_boxes,
                start_player,
                0,
                0,
                features=start_coords,
                projected=start_coords,
            )

        root_id = self._add_node(
            start_boxes,
            start_player,
            0,
            0,
            start_feat,
            start_feat,
            None,
            None,
        )
        assert root_id is not None
        start_idx = self.emit(
            SearchEvent(
                kind="start",
                boxes=start_boxes,
                player=start_player,
                pushes=0,
                weight=0,
                open_len=0,
                closed_len=1,
                features=start_coords,
                projected=start_coords,
            )
        )
        if start_idx is not None:
            self.expand_index[root_id] = start_idx
        self._enqueue_moves(root_id)

        if self.max_time is not None:
            self._deadline = time.perf_counter() + self.max_time

        cell_cursor = 0
        while self._pending_count() > 0:
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                return self._budget_stop("timeout")
            if len(self.seen) > self.max_states:
                return self._budget_stop("capped")

            if not self.cell_order:
                break

            tried = 0
            move: PendingMove | None = None
            while tried < len(self.cell_order):
                cell = self.cell_order[cell_cursor % len(self.cell_order)]
                cell_cursor += 1
                tried += 1
                move = self._pop_best_from_cell(cell)
                if move is not None:
                    break
            if move is None:
                break

            result = self._expand_move(move)
            if result is not None:
                return result

        return self._budget_stop("exhausted")


def search_fess(
    level: Level,
    *,
    max_states: int = 500_000,
    max_time: float | None = None,
    collect_trace: bool = False,
    max_events: int = 250_000,
) -> SearchResult:
    """Run FESS with macro moves and optionally collect a visualization trace."""
    return _FessSearcher(
        level,
        max_states=max_states,
        max_time=max_time,
        collect_trace=collect_trace,
        max_events=max_events,
    ).run()


def find_solution_fess(
    level: Level,
    *,
    max_states: int = 500_000,
    max_time: float | None = None,
) -> str | None:
    """Return a move string that solves ``level`` via FESS, or ``None``."""
    return search_fess(
        level,
        max_states=max_states,
        max_time=max_time,
        collect_trace=False,
    ).solution
