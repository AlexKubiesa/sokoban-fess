"""Feature Space Search (FESS) Sokoban solver — educational CoG 2020 port.

Projects positions onto a 4-D feature grid (packed, connectivity,
room-connectivity, OOP) and expands one pending **macro move** per
feature-cell visit, preferring advisor-endorsed moves (weight 0 vs 1).

See ``docs/fess.md`` for a walkthrough aligned with Shoham & Schaeffer
(CoG 2020).
"""

from __future__ import annotations

from sokoban_fess.fess.features import FeatureCell, feature_coordinates, get_analysis
from sokoban_fess.fess.search import (
    DIFFICULT_MOVE_WEIGHT,
    find_solution_fess,
    search_fess,
)

__all__ = [
    "DIFFICULT_MOVE_WEIGHT",
    "FeatureCell",
    "feature_coordinates",
    "find_solution_fess",
    "get_analysis",
    "search_fess",
]
