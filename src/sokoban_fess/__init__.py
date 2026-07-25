"""Sokoban FESS: educational FESS Sokoban solver for teaching Feature Space Search.

See ``docs/fess.md`` and ``sokoban_fess.fess``. Play evaluation levels or watch
the solver with ``sokoban-fess play``; solve programmatically via ``search_fess``.
"""

from sokoban_fess.benchmark import BenchmarkReport, LevelResult, run_benchmark
from sokoban_fess.fess import find_solution_fess, search_fess
from sokoban_fess.level import Level, LevelState, parse_ascii, replay_solution
from sokoban_fess.levels import evaluation_level_count, load_evaluation_levels
from sokoban_fess.solver import SearchEvent, SearchResult, find_solution, search

__all__ = [
    "Level",
    "LevelState",
    "parse_ascii",
    "replay_solution",
    "load_evaluation_levels",
    "evaluation_level_count",
    "find_solution",
    "search",
    "find_solution_fess",
    "search_fess",
    "SearchEvent",
    "SearchResult",
    "BenchmarkReport",
    "LevelResult",
    "run_benchmark",
]

__version__ = "0.1.0"
