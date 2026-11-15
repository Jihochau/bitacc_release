"""Sensitivity of the decoding throughput to the DIMM-level PE (R4).

Every variant changes only the listed configuration values of the base design:

* ``dimm_x2`` / ``dimm_x4``: the DIMM-level PE runs at 2x / 4x its clock frequency, a
  stand-in for a 2x / 4x wider vector datapath.
* ``dimm_ideal``: the arithmetic of the DIMM-level PE takes (almost) no time: its clock is
  1000x faster and a complete table tile is generated per clock.  Bus transfers are
  unchanged.
* ``tablegen_x2`` / ``tablegen_x4`` / ``tablegen_x8``: only the table generation of the
  DIMM-level PE is 2x / 4x / 8x faster.

The reference point of every variant is the base design, shared with ``context``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import sweeplib
from sweeplib import GEOMEAN_LABEL, MODELS, Case

NAME = "dimm"

DIMM_CSV = "dimm_sensitivity.csv"
CSV_KEYS = {DIMM_CSV: ("variant", "model", "kv_history")}

KV_HISTORIES = (128, 1024)
BASE = "base"
DIMM_PE_CLOCK_HZ = 666_666_667  # dimm_pe_clock_hz of config/pim/bitacc_runtime_default.json
VARIANTS: dict[str, dict[str, Any]] = {
    "dimm_x2": {"pim_runtime": {"dimm_pe_clock_hz": 1_333_333_334}},
    "dimm_x4": {"pim_runtime": {"dimm_pe_clock_hz": 2_666_666_668}},
    "dimm_ideal": {
        "hardware": {"lut_pair_gen_per_cycle": 32},
        "pim_runtime": {"dimm_pe_clock_hz": 1000 * DIMM_PE_CLOCK_HZ},
    },
    "tablegen_x8": {"hardware": {"lut_pair_gen_per_cycle": 8}},
}

COLUMNS = (
    "variant",
    "model",
    "kv_history",
    "final_cycle",
    "base_final_cycle",
    "speedup_over_base",
)


def cases(args: argparse.Namespace) -> list[Case]:
    selected: list[Case] = []
    for model in sweeplib.select_models(args):
        for kv in sweeplib.select_kv(args, KV_HISTORIES):
            selected.append(sweeplib.base_case(model, kv))
            for variant, overrides in VARIANTS.items():
                selected.append(
                    Case(
                        experiment=NAME,
                        case_id=f"{variant}_{sweeplib.model_slug(model)}_kv{kv}",
                        model=model,
                        kv_history=kv,
                        overrides=overrides,
                        labels={"variant": variant},
                    )
                )
    return selected


def dimm_rows(
    base_results: Sequence[Mapping[str, Any]], variant_results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Rows of dimm_sensitivity.csv: throughput of every variant normalized to the base."""
    cycles = {
        (BASE, result["model"], result["kv_history"]): result["final_cycle"]
        for result in base_results
        if result["kv_history"] in KV_HISTORIES
    }
    for result in variant_results:
        key = (result["labels"]["variant"], result["model"], result["kv_history"])
        cycles[key] = result["final_cycle"]

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for kv in KV_HISTORIES:
            speedups = []
            for model in MODELS:
                base = cycles.get((BASE, model, kv))
                final_cycle = cycles.get((variant, model, kv))
                if base is None or final_cycle is None:
                    continue
                speedups.append(base / final_cycle)
                rows.append(
                    {
                        "variant": variant,
                        "model": model,
                        "kv_history": kv,
                        "final_cycle": final_cycle,
                        "base_final_cycle": base,
                        "speedup_over_base": speedups[-1],
                    }
                )
            if len(speedups) == len(MODELS):
                rows.append(
                    {
                        "variant": variant,
                        "model": GEOMEAN_LABEL,
                        "kv_history": kv,
                        "speedup_over_base": sweeplib.geomean(speedups),
                    }
                )
    return rows


def parse(runs_dir: Path, csv_dir: Path) -> list[Path]:
    base_results = [
        result for result in sweeplib.load_results(runs_dir, "context") if not result["overrides"]
    ]
    variant_results = [
        result
        for result in sweeplib.load_results(runs_dir, NAME)
        if result["overrides"] == VARIANTS.get(result["labels"].get("variant"))
    ]
    rows = dimm_rows(base_results, variant_results)
    return [sweeplib.write_csv(csv_dir / DIMM_CSV, COLUMNS, rows)]
