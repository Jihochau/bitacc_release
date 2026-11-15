#!/usr/bin/env python3
"""Turn the finished runs in results/runs/ into the CSV files in results/csv/.

With --check, every produced value is compared with the reference in expected/paper/:
cycle counts and other integers must be equal, and derived floating-point values must
agree within a relative tolerance of 1e-9.  Reference rows without a finished run are
reported as "not run" and do not fail the check.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import sweeplib

RESOURCES_CSV = "run_resources.csv"
RESOURCES_COLUMNS = (
    "experiment",
    "case_id",
    "model",
    "kv_history",
    "final_cycle",
    "wall_time_s",
    "peak_rss_bytes",
    "peak_rss_generator_bytes",
    "git_commit",
)
RELATIVE_TOLERANCE = 1e-9


@dataclass
class Comparison:
    """Outcome of comparing one produced CSV file with its reference."""

    name: str
    matched: int = 0
    not_run: int = 0
    unreferenced: int = 0
    mismatches: list[str] = field(default_factory=list)


def _is_int(text: str) -> bool:
    return text.lstrip("+-").isdigit()


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def cells_equal(expected: str, produced: str) -> bool:
    """Integers (cycle counts) must be identical, floats equal within the tolerance."""
    if _is_int(expected) or _is_int(produced):
        return _is_int(expected) and _is_int(produced) and int(expected) == int(produced)
    if _is_float(expected) and _is_float(produced):
        return math.isclose(
            float(expected), float(produced), rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0
        )
    return expected == produced


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def compare_csv(expected_path: Path, produced_path: Path, keys: Sequence[str]) -> Comparison:
    """Compare a produced CSV file with its reference, aligning the rows by ``keys``."""
    comparison = Comparison(expected_path.name)
    expected = sweeplib.read_csv(expected_path)
    if not produced_path.is_file():
        comparison.not_run = len(expected)
        return comparison
    produced = sweeplib.read_csv(produced_path)
    columns = _header(expected_path)
    if _header(produced_path) != columns:
        comparison.mismatches.append(
            f"columns differ: expected {columns}, got {_header(produced_path)}"
        )
        return comparison

    produced_by_key = {tuple(row[key] for key in keys): row for row in produced}
    expected_keys = set()
    for row in expected:
        key = tuple(row[name] for name in keys)
        expected_keys.add(key)
        actual = produced_by_key.get(key)
        if actual is None:
            comparison.not_run += 1
            continue
        differences = [
            f"{column}: expected {row[column]!r}, got {actual[column]!r}"
            for column in columns
            if not cells_equal(row[column], actual[column])
        ]
        if differences:
            label = ", ".join(f"{name}={value}" for name, value in zip(keys, key))
            comparison.mismatches.append(f"[{label}] " + "; ".join(differences))
        else:
            comparison.matched += 1
    comparison.unreferenced = sum(key not in expected_keys for key in produced_by_key)
    return comparison


def write_resources(runs_dir: Path, csv_dir: Path) -> Path:
    """Wall time and peak memory of every finished run (not part of --check)."""
    rows = []
    for experiment_dir in sorted(path for path in runs_dir.glob("*") if path.is_dir()):
        rows.extend(sweeplib.load_results(runs_dir, experiment_dir.name))
    return sweeplib.write_csv(csv_dir / RESOURCES_CSV, RESOURCES_COLUMNS, rows)


def check(csv_dir: Path, expected_dir: Path, keys: dict[str, Sequence[str]]) -> bool:
    """Compare every reference CSV with the produced one; return True if nothing differs."""
    references = sorted(expected_dir.glob("*.csv"))
    if not references:
        print(f"check: no reference CSV files in {expected_dir}")
        return False
    passed = True
    for reference in references:
        if reference.name not in keys:
            print(f"check: {reference.name}: FAIL (no experiment declares its key columns)")
            passed = False
            continue
        result = compare_csv(reference, csv_dir / reference.name, keys[reference.name])
        verdict = "FAIL" if result.mismatches else "ok"
        print(
            f"check: {result.name}: {verdict} ({result.matched} rows match, "
            f"{len(result.mismatches)} mismatch, {result.not_run} not run, "
            f"{result.unreferenced} without reference)"
        )
        for message in result.mismatches:
            print(f"  MISMATCH {message}")
        passed = passed and not result.mismatches
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the produced CSV files with expected/paper/ and exit non-zero on a mismatch",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=sweeplib.RUNS_DIR, help="directory of the finished runs"
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=sweeplib.CSV_DIR, help="output directory of the CSV files"
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=sweeplib.EXPECTED_DIR,
        help="directory of the reference CSV files used by --check",
    )
    args = parser.parse_args()

    keys: dict[str, Sequence[str]] = {}
    written: list[Path] = []
    for name, module in sweeplib.discover_experiments().items():
        paths = module.parse(args.runs_dir, args.csv_dir)
        undeclared = [path.name for path in paths if path.name not in module.CSV_KEYS]
        if undeclared:
            raise ValueError(f"experiment {name}: CSV_KEYS does not list {undeclared}")
        keys.update(module.CSV_KEYS)
        written.extend(paths)
    written.append(write_resources(args.runs_dir, args.csv_dir))
    for path in written:
        rows = len(sweeplib.read_csv(path))
        print(f"wrote {path} ({rows} rows)")

    if args.check and not check(args.csv_dir, args.expected_dir, keys):
        print("check: FAILED")
        return 1
    if args.check:
        print("check: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
