"""Unified ``sokoban-fess`` command-line interface."""

from __future__ import annotations

import argparse

from sokoban_fess import benchmark, play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sokoban-fess",
        description="Sokoban FESS: play evaluation levels, watch FESS, or benchmark the solver.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser(
        "play",
        help="Play evaluation levels in Pygame.",
        description="Play Sokoban FESS evaluation levels in Pygame.",
    )
    play.add_arguments(play_parser)
    play_parser.set_defaults(_run=play.run)

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark FESS on the evaluation suite.",
        description="Benchmark FESS on the Sokoban FESS evaluation suite.",
    )
    benchmark.add_arguments(bench_parser)
    bench_parser.set_defaults(_run=benchmark.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())
