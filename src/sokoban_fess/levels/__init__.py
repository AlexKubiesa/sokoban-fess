"""Load the fixed Sokoban evaluation suite shipped with the package."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from sokoban_fess.level import Level, parse_ascii

__all__ = [
    "load_evaluation_levels",
    "get_evaluation_level",
    "evaluation_level_count",
]


@lru_cache(maxsize=4)
def _load_raw() -> tuple[dict, ...]:
    package = resources.files("sokoban_fess.levels")
    data = package.joinpath("evaluation.json").read_text(encoding="utf-8")
    payload = json.loads(data)
    if not isinstance(payload, list):
        raise ValueError("evaluation.json must contain a list of levels")
    return tuple(payload)


def load_evaluation_levels(
    *,
    height: int | None = None,
    width: int | None = None,
) -> list[Level]:
    """Return packaged evaluation levels at authored dimensions.

    Pass ``height`` / ``width`` to pad each map with walls to a fixed size.
    """
    levels: list[Level] = []
    for entry in _load_raw():
        level = parse_ascii(
            entry["map"],
            name=entry["name"],
            solution=entry.get("solution"),
            difficulty=int(entry.get("difficulty", 1)),
            height=height,
            width=width,
        )
        levels.append(level)
    if len(levels) < 1:
        raise RuntimeError("Evaluation suite is empty")
    return levels


def get_evaluation_level(
    index: int,
    *,
    levels: list[Level] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> Level:
    """Return evaluation level by index."""
    suite = levels if levels is not None else load_evaluation_levels(height=height, width=width)
    if index < 0 or index >= len(suite):
        raise IndexError(f"level_index {index} out of range [0, {len(suite)})")
    return suite[index]


def evaluation_level_count() -> int:
    return len(_load_raw())
