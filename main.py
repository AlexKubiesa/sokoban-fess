"""Small demo: print features and solve a tiny Sokoban with FESS."""

from __future__ import annotations

from sokoban_fess import parse_ascii, search_fess
from sokoban_fess.fess import feature_coordinates
from sokoban_fess.level import render_ansi, LevelState
from sokoban_fess.levels import get_evaluation_level


def main() -> None:
    level = parse_ascii(
        [
            "#####",
            "#@$.#",
            "#####",
        ],
        name="tiny",
    )
    print(f"Level: {level.name}")
    print(render_ansi(LevelState.from_level(level)))
    feat = feature_coordinates(level, frozenset(level.boxes))
    print(f"Features: {feat}")

    result = search_fess(level, max_time=2.0)
    print(f"Solution: {result.solution!r}  ({result.stop_reason})")

    eval_level = get_evaluation_level(0)
    print(f"\nEvaluation level: {eval_level.name} ({eval_level.height}x{eval_level.width})")
    print(render_ansi(LevelState.from_level(eval_level)))


if __name__ == "__main__":
    main()
