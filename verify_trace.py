#!/usr/bin/env python3
"""Verify a generated BitNet-125M KV-history-2 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPECTED = ROOT / "expected" / "bitnet_125m_kv2.json"
PROTOCOL_COMMANDS = {"ACTAB", "PREA", "REFab"}


def inspect_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = normalized.decode("utf-8").splitlines()
    counts: Counter[str] = Counter()
    cycles: list[int] = []
    parse_errors: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            parse_errors.append(f"line {line_number}: expected comma-separated cycle and command")
            continue
        try:
            cycles.append(int(fields[0]))
        except ValueError:
            parse_errors.append(f"line {line_number}: invalid cycle {fields[0]!r}")
        counts[fields[1]] += 1

    protocol_line_count = sum(counts[name] for name in PROTOCOL_COMMANDS)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "line_count": len(lines),
        "semantic_line_count": len(lines) - protocol_line_count,
        "protocol_line_count": protocol_line_count,
        "last_cycle": cycles[-1] if cycles else None,
        "max_cycle": max(cycles) if cycles else None,
        "command_counts": dict(sorted(counts.items())),
        "parse_errors": parse_errors,
    }


def verify_trace(
    trace_path: str | Path, expected_path: str | Path = DEFAULT_EXPECTED
) -> dict[str, Any]:
    trace = Path(trace_path)
    expected_file = Path(expected_path)
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    actual = inspect_trace(trace)
    errors: list[str] = list(actual["parse_errors"])

    for key in (
        "normalized_sha256",
        "line_count",
        "semantic_line_count",
        "protocol_line_count",
        "last_cycle",
    ):
        if actual[key] != expected[key]:
            errors.append(f"{key}: expected {expected[key]!r}, got {actual[key]!r}")

    expected_counts = expected["command_counts"]
    if actual["command_counts"] != expected_counts:
        names = sorted(set(actual["command_counts"]) | set(expected_counts))
        for name in names:
            observed = actual["command_counts"].get(name, 0)
            wanted = expected_counts.get(name, 0)
            if observed != wanted:
                errors.append(f"command {name}: expected {wanted}, got {observed}")

    if actual["max_cycle"] != actual["last_cycle"]:
        errors.append(
            f"trace is not cycle-complete: max cycle {actual['max_cycle']} != last cycle {actual['last_cycle']}"
        )

    return {
        "ok": not errors,
        "status": "pass" if not errors else "fail",
        "trace": str(trace.resolve()),
        "expected": str(expected_file.resolve()),
        "actual": actual,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    args = parser.parse_args()
    result = verify_trace(args.trace, args.expected)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
