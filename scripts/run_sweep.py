#!/usr/bin/env python3
"""Run the simulations of one experiment, or of all experiments.

Every case runs in its own directory results/runs/<experiment>/<case_id>/.  Cases that
already have a valid result are skipped, so an interrupted sweep can simply be restarted.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import ModuleType

import sweeplib
from sweeplib import Case


def _parser(experiments: dict[str, ModuleType]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=[*experiments, "all"],
        help="experiment to run, or 'all' for every experiment",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=f"only these model configurations (default: all of {', '.join(sweeplib.MODELS)})",
    )
    parser.add_argument(
        "--kv",
        nargs="+",
        type=int,
        metavar="TOKENS",
        help="only these KV histories (default: the histories of each experiment)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, metavar="N", help="simulations run in parallel (default: 1)"
    )
    parser.add_argument(
        "--status", action="store_true", help="print the number of complete cases and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the selected cases without running them"
    )
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="keep the command trace of every run (several GiB for the largest configurations)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=sweeplib.DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"time limit of one simulation (default: {sweeplib.DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=sweeplib.RUNS_DIR,
        help="directory of the run directories (default: results/runs)",
    )
    return parser


def _selected(
    experiments: dict[str, ModuleType], args: argparse.Namespace
) -> dict[str, list[Case]]:
    names = list(experiments) if args.experiment == "all" else [args.experiment]
    return {name: experiments[name].cases(args) for name in names}


def _print_status(selection: dict[str, list[Case]], runs_dir: Path) -> None:
    width = max(len(name) for name in [*selection, "total"])
    for name, cases in selection.items():
        done = sum(sweeplib.load_result(case, runs_dir) is not None for case in cases)
        print(f"{name:<{width}}  {done:>4}/{len(cases):<4} complete")
    if len(selection) > 1:
        unique = sweeplib.unique_cases(case for cases in selection.values() for case in cases)
        done = sum(sweeplib.load_result(case, runs_dir) is not None for case in unique)
        print(f"{'total':<{width}}  {done:>4}/{len(unique):<4} complete (shared cases counted once)")


def _print_cases(cases: list[Case], runs_dir: Path) -> None:
    for case in cases:
        state = "complete" if sweeplib.load_result(case, runs_dir) is not None else "pending"
        overrides = sweeplib.format_overrides(case.overrides)
        print(
            f"{case.experiment}/{case.case_id}: model={case.model} kv={case.kv_history} "
            f"overrides={overrides} [{state}]"
        )
    print(f"{len(cases)} case(s)")


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    experiments = sweeplib.discover_experiments()
    parser = _parser(experiments)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        selection = _selected(experiments, args)
    except ValueError as exc:
        parser.error(str(exc))
    if not any(selection.values()):
        parser.error("--models/--kv select no case of the chosen experiment(s)")

    if args.status:
        _print_status(selection, args.runs_dir)
        return 0
    cases = sweeplib.unique_cases(case for cases in selection.values() for case in cases)
    # Smallest configurations first, so that partial sweeps yield complete small models.
    cases.sort(key=lambda case: (sweeplib.MODELS.index(case.model), case.kv_history))
    if args.dry_run:
        _print_cases(cases, args.runs_dir)
        return 0

    signal.signal(signal.SIGTERM, _raise_interrupt)
    outcome = sweeplib.run_cases(
        cases,
        jobs=args.jobs,
        keep_traces=args.keep_traces,
        timeout_s=args.timeout,
        runs_dir=args.runs_dir,
    )
    print(
        f"{len(outcome.completed)} run, {len(outcome.skipped)} skipped, "
        f"{len(outcome.failed)} failed"
    )
    for case, message in outcome.failed:
        first_line = message.splitlines()[0] if message else ""
        print(f"  failed: {case.experiment}/{case.case_id}: {first_line}")
    if outcome.interrupted:
        return 130
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    sys.exit(main())
