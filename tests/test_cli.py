"""Tests for the unified ``sokoban-fess`` CLI."""

from __future__ import annotations

import sokoban_fess.cli as cli


def test_cli_requires_subcommand() -> None:
    try:
        cli.main([])
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 2


def test_cli_benchmark_help() -> None:
    try:
        cli.main(["benchmark", "--help"])
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_play_help() -> None:
    try:
        cli.main(["play", "--help"])
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_benchmark_single_level() -> None:
    code = cli.main(["benchmark", "--level", "1", "--max-time", "1"])
    assert code == 0
