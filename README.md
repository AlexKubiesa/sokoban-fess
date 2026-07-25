# Sokoban FESS

Educational **Feature Space Search (FESS)** Sokoban solver
(Shoham & Schaeffer, CoG 2020).

The learning path is the FESS solver and its visualizer — not a black-box A*
port. Start here:

1. Read **[docs/fess.md](docs/fess.md)** — algorithm walkthrough mapped to modules  
   (includes **what the CoG paper describes that this port does not fully implement**)  
2. Explore **`src/sokoban_fess/fess/`** — macros, features, advisors, packing, search  
3. Watch a solve: `uv run sokoban-fess play` → **Watch FESS solver**

## Citation

Paper (primary reference):

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

This repository is a teaching port of the algorithm and Sokoban feature/advisor
ideas — not a reproduction of the authors’ XSokoban-90 experimental solver.

## FESS in one paragraph

Positions are projected onto a 4-D **feature space** (packed boxes along a
packing plan, connectivity, room-connectivity, out-of-plan). The search tree
still lives in the board domain, using **macro moves** (one box A→B). FESS
cycles through feature cells and expands the least-**weight** pending macro;
advisors mark promising macros with weight `0`, others get `1`.

```
src/sokoban_fess/fess/
  macros.py      # box→cell macros
  rooms.py       # 2×3 rooms + room-connectivity
  packing.py     # basins, sink, packing plan, OOP
  features.py    # FeatureCell + preprocess (hotspots, mobility)
  advisors.py    # seven advisors (≤1 macro each)
  search.py      # Fig. 2 loop, weights, projection
```

## Install

```bash
uv sync --extra play --extra dev
```

Solver-only (no Pygame viewer):

```bash
uv sync --extra dev
```

## Play & watch FESS

```bash
uv run sokoban-fess play
# or start on a specific level (1–30):
uv run sokoban-fess play --level 1
```

Pick a level → **Watch FESS solver**. Press `H` or `?` in the search viewer for
a legend of the side-panel stats (features, weight, advisors, feature-space
graph). Details: [How to read the viewer](docs/fess.md#how-to-read-the-viewer).

Controls:

| Context | Keys |
|---------|------|
| Level picker | ↑/↓ or W/S select, Enter/Space play, Q/Esc quit |
| Mode select | Choose **Play level** or **Watch FESS solver** |
| Playing | WASD or arrow keys move, U undo, R restart, Esc back to picker |
| Level cleared | Enter/N next level, R replay, U undo, Esc picker |
| Search viewer | Space pause, ←/→ step, Show enqueued moves toggle, H/? or ? button for stats help, Esc back |

Clearing a level advances to the next evaluation level in suite order. After level 30, the player returns to the picker.

## Solver API (quick check)

```python
from sokoban_fess import parse_ascii, search_fess
from sokoban_fess.fess import feature_coordinates

level = parse_ascii(
    [
        "#####",
        "#@$.#",
        "#####",
    ],
    name="tiny",
)
print(feature_coordinates(level, frozenset(level.boxes)))
result = search_fess(level, max_time=2.0)
print(result.solution, result.stop_reason)
```

Or run the demo:

```bash
uv run python main.py
```

## Solver benchmark

```bash
uv run sokoban-fess benchmark
```

Useful flags: `--max-states N`, `--max-time SECONDS` (default 5; `0` =
unlimited), `--json`, `--level N` (repeatable, 1-based).

## ASCII legend

```
# wall   ` ` floor   . target
$ box    * box on target
@ player + player on target
```

## Evaluation suite

Thirty fixed levels ship in `sokoban_fess/levels/evaluation.json`.

## Tests

```bash
uv run pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
