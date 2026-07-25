# Feature Space Search (FESS)

This repository’s Sokoban solver is an educational implementation of **Feature Space Search** (Shoham & Schaeffer, CoG 2020).

Primary reference:

- [The FESS Algorithm: A Feature Based Approach to Single-Agent Search](https://ieee-cog.org/2020/papers/paper_44.pdf) (CoG 2020 PDF)
- DOI: [10.1109/CoG47356.2020.9231929](https://doi.org/10.1109/CoG47356.2020.9231929)

```bibtex
@inproceedings{shoham2020fess,
  title     = {The {FESS} Algorithm: A Feature Based Approach to Single-Agent Search},
  author    = {Shoham, Yaron and Schaeffer, Jonathan},
  booktitle = {2020 IEEE Conference on Games (CoG)},
  pages     = {96--103},
  year      = {2020},
  doi       = {10.1109/CoG47356.2020.9231929}
}
```

The code is organized to mirror the paper’s concepts. Read this page alongside
`src/sokoban_fess/fess/`.

## Domain space vs feature space


| Space                  | What lives there                          | Module                   |
| ---------------------- | ----------------------------------------- | ------------------------ |
| **Domain space (DS)**  | Board positions, macro moves, search tree | `macros.py`, `search.py` |
| **Feature space (FS)** | Coarse progress coordinates               | `features.py`            |


FESS does **not** search feature space directly. It keeps a normal search tree
in DS and uses FS as multi-objective guidance: cells that look closer to the
goal get a permanent share of expansion time.

## Macro moves

A **macro** pushes one box from cell A to any other cell B reachable by a
sequence of pushes of that box alone (other boxes frozen). See `macros.py`
(CoG §III-D).

Macros raise the branching factor but make feature changes visible in one
tree edge — the DS move unit used in the paper.

## The FESS loop (`search.py`)

Aligned with CoG Figure 2:

1. Root = start position; weight 0; project onto an FS cell.
2. Assign weights to all macros from the new node (via advisors).
3. Cyclically pick the next non-empty FS cell.
4. Among unexpanded macros whose parents project to that cell, expand the
  **least accumulated weight**.
5. Child weight = parent weight + move weight; project child onto FS
  (regressions pin to the best-ancestor cell).
6. Repeat until solved (or budget exhausted).

### Weights


| Move kind                      | Weight added |
| ------------------------------ | ------------ |
| Endorsed by ≥1 advisor         | `0`          |
| All other (“difficult”) macros | `1`          |


This matches the CoG experimental setting: weight counts difficult moves;
advisor-only paths stay at weight 0 and are preferred via feature tie-breaks.

### Tie-breaks

When weights are equal, prefer lower OOP, higher packed, lower connectivity,
lower room-connectivity, lower hotspots, higher mobility, then fewer pushes
(lexicographic multi-feature ordering as in the paper’s Sokoban discussion).

## Feature space (4-D)


| Feature               | Goal direction | Source                                                 |
| --------------------- | -------------- | ------------------------------------------------------ |
| **packed**            | maximize       | Progress along the packing/parking plan (`packing.py`) |
| **connectivity**      | minimize       | Free-cell connected components                         |
| **room_connectivity** | minimize       | Broken room–room edges (`rooms.py`)                    |
| **oop**               | minimize       | Out-of-plan boxes for the current packing step         |


Hotspots and mobility are **not** FS axes; they feed advisors and tie-breaks.

Goal-directed “progress” for projection uses the lex key
`(oop, −packed, connectivity, room_connectivity)`.

## Rooms (`rooms.py`)

Rooms are connected open regions that contain a 2×3 or 3×2 open block
(CoG room-connectivity feature). The static room graph’s edges may be broken
by boxes; the count of broken edges is room-connectivity.

## Packing plan & OOP (`packing.py`)

CoG’s packing feature counts boxes placed according to a packing order from
retrograde analysis. Here that plan is built by a **backward FESS** in pull
mode: features `(boxes_on_board, boxes_on_targets)`, with an advisor that
prefers long pulls from targets under connectivity / push-accessibility
constraints. Reversing the best path yields the forward packing/parking order.

- **packed** — how many prefix steps of that order are occupied  
- **oop** — boxes outside the plan prefix and outside the OK zone for the
current packing step (parked cells treated as walls; remaining targets
define reachable sources)

## Deadlocks (`deadlocks.py`)

CoG describes dead-end marking in the search tree (no valid moves; propagate
when all children are dead). This port also applies a small set of local
board checks before expanding a macro: static unreachable-off-target cells,
corner freezes, and 2×2 freezes.

## Projection

Regressions pin to the **best ancestor**’s feature cell (CoG §II / Fig. 2
discussion of discouraging feature-space diversions), not only the immediate
parent — the ancestor with the best true feature values on the path to the root.

## Advisors (`advisors.py`)

CoG: each advisor suggests at most one move; advisor moves get weight 0.
This port uses seven advisors aligned with the paper’s Sokoban roles:

1. `packing` — increase plan progress
2. `connectivity` — reduce free-space components
3. `room_connectivity` — repair room edges
4. `hotspots` — reduce hotspot count
5. `explorer` — unlock a previously impossible single-step push
6. `opener` — clear boxes around the hottest hotspot
7. `oop` — reduce out-of-plan boxes / clear paths into the basin

Hotspot preprocess always runs (with a time abort on large boards).

## Watching the search

```bash
uv sync --extra dev
uv run sokoban-fess play
```

Pick a level → **Watch FESS solver**. Space pauses autoplay; ←/→ steps one
event; use the **Show enqueued moves** switch to include not-yet-expanded
macros (default: off — only expands); `H` / `?` (or the `?` button) opens the
same legend as below.

### How to read the viewer

The board shows the current search event. Overlays tint recent activity and
draw the macro under consideration (box path / push arrow). The side panel
is the FESS HUD:


| Panel field             | Meaning                                                                                                                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase**               | *Search* = stepping FESS events; *Solution* = replaying the found path                                                                                                                                                        |
| **Features**            | True 4-D coordinates of the current box layout (not the projected cell)                                                                                                                                                       |
| `packed`                | Packing-plan progress — **higher** is better                                                                                                                                                                                  |
| `connectivity`          | Free-cell connected components — **lower** is better                                                                                                                                                                          |
| `room_conn`             | Broken room–room edges — **lower** is better                                                                                                                                                                                  |
| `oop`                   | Out-of-plan boxes for this packing step — **lower** is better                                                                                                                                                                 |
| **Macro**               | Box path for this event (`start → end` cells)                                                                                                                                                                                 |
| `weight`                | Tree cost so far: advisor macros add `0`, others add `1` (prefer lower)                                                                                                                                                       |
| `pushes`                | Box pushes on the path from the root to this node                                                                                                                                                                             |
| **Advisors**            | Which heuristics endorsed this macro (empty ⇒ weight +1)                                                                                                                                                                      |
| **Pending moves**       | Macros still waiting in the open set                                                                                                                                                                                          |
| **Show enqueued moves** | Off (default): step expands only; On: include every enqueue/deadlock                                                                                                                                                          |
| **Tree nodes**          | Distinct board states already visited (closed set)                                                                                                                                                                            |
| **Feature space**       | True FS cells on the current timeline. Left→right = FESS feature progress (lex order). With enqueued moves off, only expanded states. Yellow = path; cyan = current; violet ring = projected cell when a regression is pinned |
| **Trace**               | Shown when the event log hit `max_events` (viewer truncated)                                                                                                                                                                  |


Event kinds in the trace: *Start*, *Enqueue successor*, *Expand*, *Prune
deadlock*, then a terminal *Goal* / *capped* / *timeout* / *exhausted*.
By default the viewer hides *Enqueue* / *Prune deadlock* so you step through
macros chosen for expansion; flip **Show enqueued moves** to show every
enqueued successor.

When reading the panel, ask: did an advisor endorse this macro? Did features
move toward the goal (higher packed / lower oop & connectivity)? That is the
same guidance FESS uses when cycling feature cells and picking least-weight
pending macros.

## Benchmark

```bash
uv run sokoban-fess benchmark
```

## Completeness

FESS does not prune non-dead macros; it only deprioritizes them with weight.
Given a finite state space and enough time, it eventually expands every
non-deadlocked path (CoG §II-D). Zero-weight advisor moves still terminate
because the number of positions is finite.

## What is *not* implemented

Relative to the CoG 2020 paper. This is an educational port of the FESS
*algorithm* and the Sokoban feature/advisor *ideas* in that paper — not a
reproduction of the authors’ full experimental solver or XSokoban-90 timing
claims.

### Stated in CoG, only partly realized here


| CoG concept                                                                          | In this port                                                                                                                |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Packing feature = boxes on targets **in packing-plan order** (retrograde pre-search) | Bounded backward mini-FESS; plan quality and parking model are simplified                                                   |
| Room connectivity                                                                    | Approximate room partition (2×3/3×2 seeds + components); may differ from the paper’s room graph on junctions/corridors      |
| Out-of-plan (OOP)                                                                    | Per-step OK zones derived from the packing plan; not every nuance of the paper’s basin / “soon-to-be-blocked” examples      |
| Seven advisors (feature axes + clear-path / force-entry styles)                      | Same roles by name; predicates are thinner than a production implementation                                                 |
| Macro moves A→B                                                                      | Implemented; we also expand concrete LURD walk+push strings for replay (the paper treats the macro as the abstract DS edge) |
| Dead-end nodes in the tree                                                           | Implemented, plus a few local board freezes; CoG does not specify a large pattern database, and we do not claim one         |


### CoG experimental claims we do not target

- Solving all 90 XSokoban levels, or matching the paper’s node/time tables  
- Solution-length competitiveness with the paper’s reported averages  
- Multi-core / large-memory engineering details implied by their runtime setup

### Deliberate non-goals

- **Optimal** push or player-move length (FESS finds *a* solution via feature
progress, not A*-style optimality)  
- Bit-level or other low-level speed engineering from any reference solver

When reading CoG, treat Figure 2 and §III (Sokoban features, macros, advisors,
weights 0/1) as the map for this codebase; treat the paper’s end-to-end
XSokoban performance numbers as out of scope for this teaching port.