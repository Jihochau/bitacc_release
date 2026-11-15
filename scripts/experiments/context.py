"""Decoding throughput of BitAcc for every model configuration and KV history (R1, R2).

The 84 base configurations are the 12 model configurations at KV histories of 32-2048
tokens.  Their results give the context-length sensitivity (speedup over the published
BitNet.cpp throughput on an Apple M2 Ultra) and, at a 128-token history, the end-to-end
throughput comparison with the published per-kernel throughput.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import sweeplib
from sweeplib import GEOMEAN_LABEL, KV_HISTORIES, MODELS, Case

NAME = "context"

CONTEXT_CSV = "context_sensitivity.csv"
END_TO_END_CSV = "end_to_end_throughput.csv"
CSV_KEYS = {
    CONTEXT_CSV: ("model", "kv_history"),
    END_TO_END_CSV: ("model", "kv_history"),
}

END_TO_END_KV = 128
ALL_HISTORIES = "all"
# Systems of the end-to-end comparison: column prefix -> kernel in baselines/kernels_m2_ultra.csv.
KERNELS = {"fp16": "float16", "t_mac": "t_mac", "bitnet_cpp_i2_s": "i2_s"}
REFERENCE_KERNEL = "bitnet_cpp_i2_s"

CONTEXT_COLUMNS = (
    "model",
    "kv_history",
    "final_cycle",
    "latency_ms",
    "tokens_per_s",
    "bitnet_cpp_tokens_per_s",
    "speedup_over_bitnet_cpp",
)
END_TO_END_COLUMNS = (
    "model",
    "kv_history",
    "final_cycle",
    "bitacc_tokens_per_s",
    *(f"{name}_tokens_per_s" for name in KERNELS),
    "bitacc_normalized",
    *(f"{name}_normalized" for name in KERNELS),
)


def cases(args: argparse.Namespace) -> list[Case]:
    histories = sweeplib.select_kv(args, KV_HISTORIES, allow_other=True)
    return [
        sweeplib.base_case(model, kv) for model in sweeplib.select_models(args) for kv in histories
    ]


def load_bitnet_cpp() -> dict[str, float]:
    """Published BitNet.cpp throughput (unlimited threads) of the 12 configurations."""
    rows = sweeplib.read_csv(sweeplib.BASELINES_DIR / "bitnet_cpp_m2_ultra.csv")
    return {row["model"]: float(row["tokens_per_s"]) for row in rows if row["threads"] == "unlimited"}


def load_kernels() -> dict[str, dict[str, float]]:
    """Published per-kernel throughput: model -> kernel -> tokens/s."""
    table: dict[str, dict[str, float]] = {}
    for row in sweeplib.read_csv(sweeplib.BASELINES_DIR / "kernels_m2_ultra.csv"):
        table.setdefault(row["model"], {})[row["kernel"]] = float(row["tokens_per_s"])
    return table


def context_rows(
    results: Sequence[Mapping[str, Any]], bitnet_cpp: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Rows of context_sensitivity.csv from the results of the base configurations."""
    by_key = {(result["model"], result["kv_history"]): result for result in results}
    histories = sorted({kv for _, kv in by_key})
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for kv in histories:
            result = by_key.get((model, kv))
            if result is None:
                continue
            rows.append(
                {
                    "model": model,
                    "kv_history": kv,
                    "final_cycle": result["final_cycle"],
                    "latency_ms": result["latency_ns"] / 1e6,
                    "tokens_per_s": result["tokens_per_s"],
                    "bitnet_cpp_tokens_per_s": bitnet_cpp[model],
                    "speedup_over_bitnet_cpp": result["tokens_per_s"] / bitnet_cpp[model],
                }
            )

    speedups = {(row["model"], row["kv_history"]): row["speedup_over_bitnet_cpp"] for row in rows}
    for kv in histories:
        if all((model, kv) in speedups for model in MODELS):
            rows.append(_geomean_row(kv, [speedups[(model, kv)] for model in MODELS]))
    paper_configurations = [(model, kv) for model in MODELS for kv in KV_HISTORIES]
    if all(key in speedups for key in paper_configurations):
        rows.append(_geomean_row(ALL_HISTORIES, [speedups[key] for key in paper_configurations]))
    return rows


def _geomean_row(kv_history: int | str, speedups: Sequence[float]) -> dict[str, Any]:
    return {
        "model": GEOMEAN_LABEL,
        "kv_history": kv_history,
        "speedup_over_bitnet_cpp": sweeplib.geomean(speedups),
    }


def end_to_end_rows(
    results: Sequence[Mapping[str, Any]], kernels: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    """Rows of end_to_end_throughput.csv: throughput normalized to the I2_S kernel."""
    by_model = {
        result["model"]: result for result in results if result["kv_history"] == END_TO_END_KV
    }
    models = [model for model in MODELS if model in kernels]
    rows: list[dict[str, Any]] = []
    for model in models:
        result = by_model.get(model)
        if result is None:
            continue
        reference = kernels[model][KERNELS[REFERENCE_KERNEL]]
        row: dict[str, Any] = {
            "model": model,
            "kv_history": END_TO_END_KV,
            "final_cycle": result["final_cycle"],
            "bitacc_tokens_per_s": result["tokens_per_s"],
            "bitacc_normalized": result["tokens_per_s"] / reference,
        }
        for name, kernel in KERNELS.items():
            published = kernels[model].get(kernel)  # None: not reported for this model.
            row[f"{name}_tokens_per_s"] = published
            row[f"{name}_normalized"] = None if published is None else published / reference
        rows.append(row)

    if len(rows) == len(models):
        summary: dict[str, Any] = {"model": GEOMEAN_LABEL, "kv_history": END_TO_END_KV}
        for name in ("bitacc", *KERNELS):
            column = f"{name}_normalized"
            summary[column] = sweeplib.geomean(
                row[column] for row in rows if row[column] is not None
            )
        rows.append(summary)
    return rows


def parse(runs_dir: Path, csv_dir: Path) -> list[Path]:
    results = [
        result for result in sweeplib.load_results(runs_dir, NAME) if not result["overrides"]
    ]
    return [
        sweeplib.write_csv(
            csv_dir / CONTEXT_CSV, CONTEXT_COLUMNS, context_rows(results, load_bitnet_cpp())
        ),
        sweeplib.write_csv(
            csv_dir / END_TO_END_CSV, END_TO_END_COLUMNS, end_to_end_rows(results, load_kernels())
        ),
    ]
