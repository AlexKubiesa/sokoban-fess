"""Benchmark the FESS Sokoban solver on the fixed evaluation suite."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Literal, Sequence, TextIO

from sokoban_fess.fess import search_fess
from sokoban_fess.level import (
    CHAR_ACTIONS,
    Level,
    LevelState,
    apply_action,
    replay_solution,
)
from sokoban_fess.levels import load_evaluation_levels
from sokoban_fess.solver import StopReason

Status = Literal["solved", "capped", "timeout", "exhausted", "invalid"]

# Default wall-clock budget per level so a full-suite run stays bounded.
DEFAULT_MAX_TIME_S = 5.0


@dataclass(frozen=True)
class LevelResult:
    """Per-level solver outcome and metrics."""

    index: int
    name: str
    difficulty: int
    status: Status
    time_s: float
    states_visited: int
    moves: int | None = None
    pushes: int | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Full-suite solver benchmark results."""

    results: tuple[LevelResult, ...]
    max_states: int
    max_time: float | None

    @property
    def n_levels(self) -> int:
        return len(self.results)

    @property
    def n_solved(self) -> int:
        return sum(1 for r in self.results if r.status == "solved")

    @property
    def n_capped(self) -> int:
        return sum(1 for r in self.results if r.status == "capped")

    @property
    def n_timeout(self) -> int:
        return sum(1 for r in self.results if r.status == "timeout")

    @property
    def n_exhausted(self) -> int:
        return sum(1 for r in self.results if r.status == "exhausted")

    @property
    def n_invalid(self) -> int:
        return sum(1 for r in self.results if r.status == "invalid")

    @property
    def solve_rate(self) -> float:
        return self.n_solved / self.n_levels if self.n_levels else 0.0

    @property
    def total_time_s(self) -> float:
        return sum(r.time_s for r in self.results)

    def by_difficulty(self) -> dict[int, tuple[LevelResult, ...]]:
        bands: dict[int, list[LevelResult]] = {}
        for result in self.results:
            bands.setdefault(result.difficulty, []).append(result)
        return {d: tuple(rows) for d, rows in sorted(bands.items())}


def classify_outcome(
    *,
    solution: str | None,
    stop_reason: StopReason,
    verified: bool,
) -> Status:
    """Map a ``SearchResult`` (+ verification) to a benchmark status."""
    if solution is not None:
        return "solved" if verified else "invalid"
    if stop_reason == "capped":
        return "capped"
    if stop_reason == "timeout":
        return "timeout"
    return "exhausted"


def verify_solution(level: Level, solution: str) -> bool:
    """Return True if ``solution`` solves ``level`` (including the empty path)."""
    if solution == "":
        return LevelState.from_level(level).success
    return replay_solution(replace(level, solution=solution))


def count_pushes(level: Level, solution: str) -> int:
    """Count box pushes while replaying ``solution``."""
    if not solution:
        return 0
    state = LevelState.from_level(
        level, max_steps=max(200, len(solution) + 10)
    )
    pushes = 0
    for ch in solution:
        if ch not in CHAR_ACTIONS:
            break
        state, _, _, _, info = apply_action(
            state, CHAR_ACTIONS[ch], step_penalty=0.0
        )
        if info.get("pushed"):
            pushes += 1
    return pushes


def run_level(
    level: Level,
    *,
    index: int,
    max_states: int = 500_000,
    max_time: float | None = DEFAULT_MAX_TIME_S,
) -> LevelResult:
    """Solve one level with FESS and collect timing / size metrics."""
    t0 = time.perf_counter()
    result = search_fess(
        level,
        max_states=max_states,
        max_time=max_time,
        collect_trace=False,
    )
    elapsed = time.perf_counter() - t0

    verified = False
    moves: int | None = None
    pushes: int | None = None
    if result.solution is not None:
        verified = verify_solution(level, result.solution)
        if verified:
            moves = len(result.solution)
            pushes = count_pushes(level, result.solution)

    status = classify_outcome(
        solution=result.solution,
        stop_reason=result.stop_reason,
        verified=verified,
    )
    return LevelResult(
        index=index,
        name=level.name,
        difficulty=level.difficulty,
        status=status,
        time_s=elapsed,
        states_visited=result.states_visited,
        moves=moves,
        pushes=pushes,
    )


def run_benchmark(
    levels: Sequence[Level] | None = None,
    *,
    indices: Sequence[int] | None = None,
    max_states: int = 500_000,
    max_time: float | None = DEFAULT_MAX_TIME_S,
    progress: TextIO | None = None,
) -> BenchmarkReport:
    """Run FESS on every evaluation level (or a provided subset).

    When ``progress`` is set, each level prints a live status line there
    (typically ``sys.stderr``) before and after solving.
    """
    suite = (
        list(levels)
        if levels is not None
        else load_evaluation_levels(height=None, width=None)
    )
    index_list = list(indices) if indices is not None else list(range(len(suite)))
    if len(index_list) != len(suite):
        raise ValueError("indices must match the number of levels")

    total = len(suite)
    results: list[LevelResult] = []
    for pos, (level, index) in enumerate(zip(suite, index_list, strict=True), start=1):
        if progress is not None:
            print(
                f"[{pos}/{total}] {level.name} ...",
                file=progress,
                end="",
                flush=True,
            )
        result = run_level(
            level,
            index=index,
            max_states=max_states,
            max_time=max_time,
        )
        if progress is not None:
            print(f" {_format_progress_result(result)}", file=progress, flush=True)
        results.append(result)

    return BenchmarkReport(
        results=tuple(results),
        max_states=max_states,
        max_time=max_time,
    )


def _format_progress_result(result: LevelResult) -> str:
    detail = f"{_fmt_seconds(result.time_s)}, {result.states_visited:,} states"
    if result.status == "solved" and result.moves is not None:
        pushes = result.pushes if result.pushes is not None else 0
        detail += f", {result.moves} moves / {pushes} pushes"
    return f"{result.status} ({detail})"


def _fmt_seconds(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def format_report(report: BenchmarkReport) -> str:
    """Render a human-readable benchmark summary."""
    time_budget = (
        "unlimited" if report.max_time is None else _fmt_seconds(report.max_time)
    )
    lines: list[str] = []
    lines.append("Sokoban FESS solver benchmark (FESS)")
    lines.append(
        f"levels={report.n_levels}  "
        f"max_states={report.max_states:,}  "
        f"max_time={time_budget}"
    )
    lines.append("")

    header = (
        f"{'#':>3}  {'name':<22}  {'diff':>4}  {'status':<9}  "
        f"{'time':>8}  {'states':>8}  {'moves':>6}  {'pushes':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.results:
        moves = "-" if r.moves is None else str(r.moves)
        pushes = "-" if r.pushes is None else str(r.pushes)
        lines.append(
            f"{r.index + 1:3d}  {r.name:<22}  {r.difficulty:4d}  {r.status:<9}  "
            f"{_fmt_seconds(r.time_s):>8}  {r.states_visited:8,d}  "
            f"{moves:>6}  {pushes:>6}"
        )

    lines.append("")
    lines.append(
        f"solved {report.n_solved}/{report.n_levels} "
        f"({100.0 * report.solve_rate:.1f}%)  "
        f"capped={report.n_capped}  timeout={report.n_timeout}  "
        f"exhausted={report.n_exhausted}  invalid={report.n_invalid}"
    )
    lines.append(f"total time {_fmt_seconds(report.total_time_s)}")

    solved = [r for r in report.results if r.status == "solved"]
    if solved:
        times = [r.time_s for r in solved]
        states = [float(r.states_visited) for r in solved]
        moves_vals = [float(r.moves or 0) for r in solved]
        push_vals = [float(r.pushes or 0) for r in solved]
        lines.append(
            "solved means — "
            f"time {_fmt_seconds(_mean(times) or 0)} "
            f"(median {_fmt_seconds(_median(times) or 0)}), "
            f"states {_mean(states):,.0f} "
            f"(median {_median(states):,.0f}), "
            f"moves {_mean(moves_vals):.1f}, "
            f"pushes {_mean(push_vals):.1f}"
        )

    lines.append("")
    lines.append("By difficulty:")
    for difficulty, rows in report.by_difficulty().items():
        n = len(rows)
        n_ok = sum(1 for r in rows if r.status == "solved")
        band_time = sum(r.time_s for r in rows)
        band_states = [r.states_visited for r in rows if r.status == "solved"]
        states_note = (
            f", mean states {statistics.fmean(band_states):,.0f}"
            if band_states
            else ""
        )
        lines.append(
            f"  d={difficulty}: {n_ok}/{n} solved "
            f"({100.0 * n_ok / n:.0f}%), "
            f"time {_fmt_seconds(band_time)}{states_note}"
        )

    failures = [r for r in report.results if r.status != "solved"]
    if failures:
        lines.append("")
        lines.append("Unresolved:")
        for r in failures:
            lines.append(
                f"  {r.index + 1:02d} {r.name} [{r.status}] "
                f"states={r.states_visited:,} time={_fmt_seconds(r.time_s)}"
            )

    return "\n".join(lines)


def report_to_dict(report: BenchmarkReport) -> dict:
    """JSON-serializable view of a benchmark report."""
    return {
        "max_states": report.max_states,
        "max_time": report.max_time,
        "n_levels": report.n_levels,
        "n_solved": report.n_solved,
        "n_capped": report.n_capped,
        "n_timeout": report.n_timeout,
        "n_exhausted": report.n_exhausted,
        "n_invalid": report.n_invalid,
        "solve_rate": report.solve_rate,
        "total_time_s": report.total_time_s,
        "results": [asdict(r) for r in report.results],
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for the solver benchmark."""
    parser.add_argument(
        "--max-states",
        type=int,
        default=500_000,
        help="State budget per level (default: 500000).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=DEFAULT_MAX_TIME_S,
        metavar="SECONDS",
        help=(
            f"Wall-clock budget per level in seconds "
            f"(default: {DEFAULT_MAX_TIME_S:g}; use 0 for unlimited)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table.",
    )
    parser.add_argument(
        "--level",
        type=int,
        action="append",
        dest="levels",
        metavar="N",
        help="Only run evaluation level N (1-based). Repeatable.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the solver benchmark from parsed CLI args."""
    max_time: float | None = None if args.max_time <= 0 else args.max_time

    suite = load_evaluation_levels(height=None, width=None)
    selected = suite
    indices: list[int] | None = None
    if args.levels:
        selected = []
        indices = []
        for n in args.levels:
            if n < 1 or n > len(suite):
                print(
                    f"--level must be between 1 and {len(suite)} (got {n})",
                    file=sys.stderr,
                )
                return 2
            selected.append(suite[n - 1])
            indices.append(n - 1)

    report = run_benchmark(
        selected,
        indices=indices,
        max_states=args.max_states,
        max_time=max_time,
        progress=sys.stderr,
    )

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_report(report))

    return 0 if report.n_solved == report.n_levels else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark FESS on the Sokoban FESS evaluation suite."
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
