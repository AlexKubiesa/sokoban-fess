"""Search-trace viewer helpers for the Sokoban FESS Pygame app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sokoban_fess.level import (
    CHAR_ACTIONS,
    Level,
    LevelState,
    apply_action,
    encode_observation,
)
from sokoban_fess.solver import FeatureCoords, SearchEvent, SearchResult

# Overlay / HUD colors for search events.
EXPAND_TINT = (70, 130, 255, 70)
ENQUEUE_TINT = (80, 200, 120, 90)
DEADLOCK_TINT = (220, 70, 70, 110)
PUSH_ARROW = (255, 220, 90)
HEAT_COLOR = (255, 140, 60, 55)
# Lighter blue / yellow than expand & current push, so the path reads as history.
PATH_TINT = (150, 195, 255, 85)
PATH_ARROW = (255, 236, 165)
PATH_PLAYER = (150, 195, 255, 100)

# Feature-space graph (2-D projection of true FS cells).
FS_EDGE = (90, 100, 130)
FS_PATH_EDGE = (255, 220, 90)
FS_NODE = (120, 140, 180)
FS_PATH_NODE = (255, 210, 80)
FS_CURRENT = (90, 200, 255)
FS_PROJECTED = (160, 140, 220)
FS_AXIS = (110, 110, 130)

KIND_LABELS = {
    "start": "Start",
    "expand": "Expand (pop open)",
    "enqueue": "Enqueue successor",
    "deadlock": "Prune deadlock",
    "goal": "Goal reached",
    "capped": "Search capped",
    "timeout": "Time limit",
    "exhausted": "No solution",
}

# Enqueue / deadlock frames are hidden unless ``show_enqueued`` is on.
ENQUEUED_KINDS = frozenset({"enqueue", "deadlock"})

SIDE_PANEL_W = 300
SEARCH_BOARD_BOTTOM = 96
FS_GRAPH_H = 168

ViewerPhase = Literal["search", "solution"]


@dataclass(frozen=True)
class FeatureSpaceGraph:
    """Discovered feature-space cells and edges up to a trace index.

    Nodes are **true** FS coordinates (matching the Features panel). Layout
    places cells left→right by FESS feature-progress order (worse→better),
    keeping every cell visually distinct. When the current event is
    projection-pinned, ``projected`` may differ from ``current``.
    """

    nodes: tuple[FeatureCoords, ...]
    edges: tuple[tuple[FeatureCoords, FeatureCoords], ...]
    path_nodes: frozenset[FeatureCoords]
    path_edges: frozenset[tuple[FeatureCoords, FeatureCoords]]
    current: FeatureCoords | None
    projected: FeatureCoords | None = None


def _fs_cell(event: SearchEvent) -> FeatureCoords | None:
    """True feature cell for the graph (falls back to projected if needed)."""
    if event.features is not None:
        return event.features
    return event.projected


def _fs_progress_key(cell: FeatureCoords) -> tuple[int, int, int, int]:
    """Lex progress key matching FESS goal direction (smaller = more progress)."""
    packed, connectivity, room_conn, oop = cell
    return (oop, -packed, connectivity, room_conn)


def feature_space_graph(
    events: tuple[SearchEvent, ...],
    upto: int,
    *,
    path: tuple[int, ...] | None = None,
    show_enqueued: bool = True,
) -> FeatureSpaceGraph:
    """Build the FS graph discovered in ``events[0..upto]``.

    Edges follow ``parent_index`` links. ``path`` (raw event indices) marks the
    highlighted root→current chain; defaults to ``path_indices(events, upto)``.

    When ``show_enqueued`` is False, enqueue/deadlock events are ignored so the
    graph only contains expanded (and terminal) states — matching the viewer
    filter.
    """
    if upto < 0 or not events:
        return FeatureSpaceGraph((), (), frozenset(), frozenset(), None, None)

    end = min(upto, len(events) - 1)
    nodes: set[FeatureCoords] = set()
    edges: set[tuple[FeatureCoords, FeatureCoords]] = set()
    for i in range(end + 1):
        ev = events[i]
        if not show_enqueued and ev.kind in ENQUEUED_KINDS:
            continue
        cell = _fs_cell(ev)
        if cell is None:
            continue
        nodes.add(cell)
        # Projection pins may reference a cell that never appears as a true
        # feature on this prefix; keep it in the node set for the pin marker.
        if ev.projected is not None:
            nodes.add(ev.projected)
        if ev.parent_index is None:
            continue
        parent = events[ev.parent_index]
        parent_cell = _fs_cell(parent)
        if parent_cell is None or parent_cell == cell:
            continue
        edges.add((parent_cell, cell))

    if path is None:
        path = path_indices(events, end)
    path_node_list: list[FeatureCoords] = []
    for idx in path:
        if idx < 0 or idx > end:
            continue
        if not show_enqueued and events[idx].kind in ENQUEUED_KINDS:
            continue
        cell = _fs_cell(events[idx])
        if cell is not None:
            path_node_list.append(cell)
    path_nodes = frozenset(path_node_list)
    path_edges: set[tuple[FeatureCoords, FeatureCoords]] = set()
    for a, b in zip(path_node_list, path_node_list[1:]):
        if a != b:
            path_edges.add((a, b))

    current = _fs_cell(events[end]) if end >= 0 else None
    projected = events[end].projected if end >= 0 else None
    return FeatureSpaceGraph(
        nodes=tuple(sorted(nodes)),
        edges=tuple(sorted(edges)),
        path_nodes=path_nodes,
        path_edges=frozenset(path_edges),
        current=current,
        projected=projected,
    )


def layout_feature_space_graph(
    graph: FeatureSpaceGraph,
    rect: tuple[float, float, float, float],
    *,
    pad: float = 18.0,
) -> dict[FeatureCoords, tuple[float, float]]:
    """Place FS nodes into ``rect`` (x, y, w, h).

    Horizontal order follows FESS feature progress (less progress left, more
    right), using the same lex order as search: oop ↓, packed ↑, connectivity ↓,
    room_connectivity ↓. Nodes are evenly spaced on x so every cell is distinct;
    y only separates them for readability (not a second progress axis).
    """
    x0, y0, w, h = rect
    if not graph.nodes:
        return {}

    # Larger progress key = less feature progress. Put those on the left.
    ordered = sorted(graph.nodes, key=_fs_progress_key, reverse=True)
    n = len(ordered)
    inner_w = max(1.0, w - 2 * pad)
    inner_h = max(1.0, h - 2 * pad)
    mid_y = y0 + h * 0.5
    # Vertical fan so edges don't all pile on one line; amplitude scales with n.
    amp = min(inner_h * 0.42, 12.0 + 4.0 * min(n, 8))

    positions: dict[FeatureCoords, tuple[float, float]] = {}
    for i, cell in enumerate(ordered):
        if n == 1:
            px = x0 + w * 0.5
        else:
            px = x0 + pad + (i / (n - 1)) * inner_w
        # Alternate above/below centre; step pattern keeps neighbours apart.
        if n == 1:
            py = mid_y
        else:
            lane = (i % 3) - 1  # -1, 0, +1
            py = mid_y + lane * (amp / 2.0)
        positions[cell] = (px, py)
    return positions


@dataclass
class ViewerState:
    """UI state for stepping through a search trace (+ optional solution coda)."""

    index: int = 0
    playing: bool = False
    events_per_second: float = 12.0
    accum: float = 0.0
    # When False (default), step only expands / terminals — not every enqueue.
    show_enqueued: bool = False


def visible_event_indices(
    events: tuple[SearchEvent, ...],
    *,
    show_enqueued: bool,
) -> tuple[int, ...]:
    """Raw event indices included in the viewer timeline."""
    if show_enqueued:
        return tuple(range(len(events)))
    return tuple(i for i, ev in enumerate(events) if ev.kind not in ENQUEUED_KINDS)


def event_to_state(level: Level, event: SearchEvent) -> LevelState:
    """Build a board state for rendering a search event."""
    return LevelState(
        level=level,
        boxes=set(event.boxes),
        player=event.player,
        steps=max(0, event.pushes),
        max_steps=10_000,
    )


def _box_push(
    prev_boxes: frozenset[tuple[int, int]],
    boxes: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    gone = prev_boxes - boxes
    came = boxes - prev_boxes
    if len(gone) == 1 and len(came) == 1:
        return next(iter(gone)), next(iter(came))
    return None, None


def heat_cells(
    events: tuple[SearchEvent, ...],
    upto: int,
    *,
    window: int = 40,
    show_enqueued: bool = True,
) -> dict[tuple[int, int], float]:
    """Recent expand/enqueue box cells → intensity in (0, 1].

    When ``show_enqueued`` is False, only expand events contribute heat.
    """
    start = max(0, upto - window + 1)
    heat: dict[tuple[int, int], float] = {}
    span = max(1, upto - start)
    allowed = (
        {"expand", "enqueue", "deadlock"} if show_enqueued else {"expand"}
    )
    for i in range(start, upto + 1):
        ev = events[i]
        if ev.kind not in allowed:
            continue
        weight = 0.35 + 0.65 * ((i - start) / span)
        for cell in ev.boxes:
            heat[cell] = max(heat.get(cell, 0.0), weight)
        heat[ev.player] = max(heat.get(ev.player, 0.0), weight * 0.7)
    return heat


def path_indices(events: tuple[SearchEvent, ...], index: int) -> tuple[int, ...]:
    """Return search-tree ancestor indices from root → ``index``."""
    if index < 0 or index >= len(events):
        return ()
    chain: list[int] = []
    seen: set[int] = set()
    cur: int | None = index
    while cur is not None and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        parent = events[cur].parent_index
        if parent is None or parent < 0 or parent >= len(events):
            break
        cur = parent
    chain.reverse()
    return tuple(chain)


def event_push_path(
    event: SearchEvent,
) -> tuple[tuple[int, int], ...] | None:
    """Box cells visited by the event's push/macro, if any."""
    if event.push_path is not None and len(event.push_path) >= 2:
        return event.push_path
    if event.push_from is not None and event.push_to is not None:
        return (event.push_from, event.push_to)
    return None


def path_push_segments(
    events: tuple[SearchEvent, ...],
    indices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Push arrows along consecutive nodes of a path-to-current chain.

    Prefer ``event_push_path`` endpoints for compatibility; prefer
    ``path_push_paths`` when drawing cornered macros.
    """
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for path in path_push_paths(events, indices):
        segments.append((path[0], path[-1]))
    return tuple(segments)


def path_push_paths(
    events: tuple[SearchEvent, ...],
    indices: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Box paths (inclusive cell sequences) along a path-to-current chain."""
    paths: list[tuple[tuple[int, int], ...]] = []
    for i, idx in enumerate(indices):
        ev = events[idx]
        path = event_push_path(ev)
        if path is not None:
            paths.append(path)
            continue
        if i == 0:
            continue
        prev = events[indices[i - 1]]
        frm, to = _box_push(prev.boxes, ev.boxes)
        if frm is not None and to is not None:
            paths.append((frm, to))
    return tuple(paths)


def path_ghost_cells(
    events: tuple[SearchEvent, ...],
    indices: tuple[int, ...],
) -> dict[tuple[int, int], float]:
    """Ancestor box/player cells → intensity fading toward the root."""
    if len(indices) <= 1:
        return {}
    # Skip the current event; ghosts show where the path came from.
    ancestors = indices[:-1]
    n = len(ancestors)
    heat: dict[tuple[int, int], float] = {}
    for i, idx in enumerate(ancestors):
        weight = 0.25 + 0.75 * ((i + 1) / n)
        ev = events[idx]
        for cell in ev.boxes:
            heat[cell] = max(heat.get(cell, 0.0), weight)
        heat[ev.player] = max(heat.get(ev.player, 0.0), weight * 0.65)
    return heat


def build_solution_frames(level: Level, solution: str) -> tuple[LevelState, ...]:
    """Board states for replaying ``solution``, including the initial state."""
    frames: list[LevelState] = [
        LevelState.from_level(level, max_steps=max(10_000, len(solution) + 10))
    ]
    state = frames[0]
    for ch in solution:
        if ch not in CHAR_ACTIONS:
            break
        state, _, terminated, _, _ = apply_action(
            state, CHAR_ACTIONS[ch], step_penalty=0.0
        )
        frames.append(state)
        if terminated:
            break
    return tuple(frames)


class SearchViewer:
    """Step / autoplay controller over a ``SearchResult`` trace and solution coda."""

    def __init__(self, level: Level, result: SearchResult) -> None:
        if not result.events:
            raise ValueError("SearchResult has no events to visualize")
        self.level = level
        self.result = result
        self.state = ViewerState()
        if result.solution is not None:
            self.solution_frames: tuple[LevelState, ...] = build_solution_frames(
                level, result.solution
            )
        else:
            self.solution_frames = ()
        # Full-trace FS graphs for stable layout (with / without enqueue nodes).
        last = len(result.events) - 1
        self._full_fs_graph_all = feature_space_graph(
            result.events, last, show_enqueued=True
        )
        self._full_fs_graph_expanded = feature_space_graph(
            result.events, last, show_enqueued=False
        )

    def visible_indices(self) -> tuple[int, ...]:
        """Raw ``result.events`` indices on the current display timeline."""
        return visible_event_indices(
            self.result.events, show_enqueued=self.state.show_enqueued
        )

    @property
    def phase(self) -> ViewerPhase:
        return "search" if self.state.index < self.n_events() else "solution"

    @property
    def event_index(self) -> int:
        """Raw index into ``result.events`` for the current search frame."""
        visible = self.visible_indices()
        if self.phase == "search":
            return visible[self.state.index]
        return visible[-1]

    @property
    def event(self) -> SearchEvent:
        """Current search event, or the last visible search event in the coda."""
        return self.result.events[self.event_index]

    @property
    def solution_frame_index(self) -> int:
        if self.phase != "solution":
            return -1
        return self.state.index - self.n_events()

    def n_events(self) -> int:
        return len(self.visible_indices())

    def n_solution_frames(self) -> int:
        return len(self.solution_frames)

    def n_total(self) -> int:
        return self.n_events() + self.n_solution_frames()

    def seek(self, index: int) -> None:
        self.state.index = max(0, min(index, self.n_total() - 1))
        self.state.accum = 0.0

    def step(self, delta: int = 1) -> None:
        self.seek(self.state.index + delta)

    def toggle_play(self) -> None:
        self.state.playing = not self.state.playing
        self.state.accum = 0.0

    def toggle_show_enqueued(self) -> None:
        """Show or hide enqueue/deadlock frames; keep the same tree node if possible."""
        if self.phase == "solution":
            sol_i = self.solution_frame_index
            self.state.show_enqueued = not self.state.show_enqueued
            self.seek(self.n_events() + sol_i)
            return
        raw = self.event_index
        self.state.show_enqueued = not self.state.show_enqueued
        visible = self.visible_indices()
        # Prefer the same raw event; otherwise the nearest earlier visible one.
        for i, idx in enumerate(visible):
            if idx >= raw:
                self.seek(i if idx == raw else max(0, i - 1))
                return
        self.seek(len(visible) - 1)

    def nudge_speed(self, factor: float) -> None:
        self.state.events_per_second = max(1.0, min(240.0, self.state.events_per_second * factor))

    def tick(self, dt: float) -> None:
        if not self.state.playing:
            return
        if self.state.index >= self.n_total() - 1:
            self.state.playing = False
            return
        self.state.accum += dt * self.state.events_per_second
        while self.state.accum >= 1.0 and self.state.index < self.n_total() - 1:
            self.state.accum -= 1.0
            self.state.index += 1
        if self.state.index >= self.n_total() - 1:
            self.state.playing = False
            self.state.accum = 0.0

    def current_state(self) -> LevelState:
        if self.phase == "solution":
            return self.solution_frames[self.solution_frame_index]
        return event_to_state(self.level, self.event)

    def path_indices(self) -> tuple[int, ...]:
        if self.phase != "search":
            return ()
        return path_indices(self.result.events, self.event_index)

    def current_fs_graph(self) -> FeatureSpaceGraph:
        """FS cells / edges discovered up to the current search event."""
        raw = self.event_index
        chain = path_indices(self.result.events, raw)
        return feature_space_graph(
            self.result.events,
            raw,
            path=chain,
            show_enqueued=self.state.show_enqueued,
        )

    def full_fs_graph(self) -> FeatureSpaceGraph:
        """All FS cells / edges in the trace (stable layout reference).

        Respects ``show_enqueued``: when off, only expanded/terminal states.
        """
        if self.state.show_enqueued:
            return self._full_fs_graph_all
        return self._full_fs_graph_expanded

    def observation(self):
        return encode_observation(self.current_state())
