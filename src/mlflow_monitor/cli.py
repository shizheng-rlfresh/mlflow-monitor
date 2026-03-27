"""Repo-local command-line entrypoint for the current monitoring slice."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from mlflow_monitor import monitor
from mlflow_monitor.domain import LifecycleStatus


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(prog="mlflow-monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute the monitoring run workflow for one source run.",
    )
    run_parser.add_argument("--subject", required=True, help="Monitored subject identifier.")
    run_parser.add_argument("--source-run", required=True, help="Training run ID to evaluate.")
    run_parser.add_argument(
        "--baseline",
        help="Baseline training run ID required for the first monitoring run.",
    )

    return parser


def _execute_run_command(args: argparse.Namespace) -> int:
    """Execute the `run` subcommand and return a process exit code."""
    result = monitor.run(
        subject_id=args.subject,
        source_run_id=args.source_run,
        baseline_source_run_id=args.baseline,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    if result.lifecycle_status is LifecycleStatus.FAILED:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, execute the command, and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run":
        return _execute_run_command(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
