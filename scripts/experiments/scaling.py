"""Scaling of the decoding throughput with chips and bank-level PEs per chip.

The 192 points are the 12 model configurations with 1, 2, 4, or 8 chips and 4, 8, 16, or 32
active bank-level PEs per chip, at a KV history of 1024 tokens.

Unlike the other experiments, these points are not simulated: the all-bank commands of the
simulator always address the 32 banks of the 8 chips of a rank.  Every point instead
rebuilds the command plan of the trace generator for its geometry (head waves, per-chip
head slices, local output groups, rank I/O width, row mapping) and reports the cycles of
that plan, ``host_pre + layers * block + host_post``.  These are the cycles of the
simulated trace without the DDR5 protocol overhead (e.g., row activation, precharge,
refresh) that only the simulator adds.  The 8-chip/32-PE point therefore equals
``cycle_breakdown.nominal_decode_full_token_cycles`` of the ``context`` base case, not its
``final_cycle``, and every ratio of this experiment is a ratio of plan cycles.  The plan of
one transformer block is built once and counted ``layers`` times, because all blocks of a
model are mapped identically.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import sweeplib
from sweeplib import GEOMEAN_LABEL, MODELS, Case

NAME = "scaling"

SCALING_CSV = "scaling.csv"
CSV_KEYS = {SCALING_CSV: ("model", "kv_history", "chips", "bank_pes")}

KV_HISTORIES = (1024,)
CHIPS = (1, 2, 4, 8)
BANK_PES = (4, 8, 16, 32)
# DRAM organization of the command plans: the largest x8 device, which holds every model.
PLAN_DRAM_ORG = "DDR5_32GB_x8"
CYCLE_SOURCE = "command_plan"

# Ratios that are summarized by their geometric mean across the models.
RATIO_COLUMNS = (
    "normalized_throughput",
    "parallel_efficiency_pct",
    "speedup_over_1_chip",
    "speedup_over_4_bank_pes",
)
COLUMNS = (
    "model",
    "kv_history",
    "chips",
    "bank_pes",
    "final_cycle",
    "tokens_per_s",
    *RATIO_COLUMNS,
    "peak_internal_bandwidth_tb_s",
    "achieved_internal_bandwidth_tb_s",
    "internal_bandwidth_utilization_pct",
)


def _overrides(chips: int, bank_pes: int) -> dict[str, Any]:
    return {"hardware": {"num_banks": bank_pes}, "ramulator": {"chips_per_dimm": chips}}


def cases(args: argparse.Namespace) -> list[Case]:
    return [
        Case(
            experiment=NAME,
            case_id=f"c{chips}_pe{bank_pes}_{sweeplib.model_slug(model)}_kv{kv}",
            model=model,
            kv_history=kv,
            overrides=_overrides(chips, bank_pes),
            labels={"chips": chips, "bank_pes": bank_pes},
        )
        for model in sweeplib.select_models(args)
        for kv in sweeplib.select_kv(args, KV_HISTORIES)
        for chips in CHIPS
        for bank_pes in BANK_PES
    ]


def _internal_payload_bytes(plan: Mapping[str, Any], chips: int, bank_pes: int) -> int:
    """Bytes that the bank-level PEs of one transformer block read from or write to their banks.

    A BitLinear group reads one weight group per active bank and chip, a KV read a complete
    per-chip bank row, and the write of the current K or V its head slice.  Rank transfers
    and the operations of the DIMM-level PE and the host are not bank-internal.
    """
    total = 0
    for event in plan["events"]:
        if event["scope"] != "block":
            continue
        terms = event.get("formula_terms", {})
        count = int(event["trace_tokens"][0]["count"])
        active = int(terms.get("active_banks", bank_pes)) * chips
        if event["command"] == "ALL_BANK_BITLINEAR":
            total += count * active * int(terms["weight_bytes_per_group"])
        elif event["command"] == "ALL_BANK_RD_K_V_CACHE":
            total += count * active * int(terms["bank_row_bytes"])
        elif event["command"] in ("ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT"):
            total += active * int(plan["parameters"]["kv_slice_bytes_per_chip"])
    return total


def plan_point(model: str, kv_history: int, chips: int, bank_pes: int) -> dict[str, Any]:
    """Cycles and bank-internal traffic of the command plan of one scaled configuration."""
    if str(sweeplib.ROOT) not in sys.path:
        sys.path.insert(0, str(sweeplib.ROOT))
    # Imported here so that listing and parsing the experiment do not load the generator.
    from src.config import load_runtime_bundle
    from src.runtime.latest_command_trace import build_latest_command_plan
    from src.runtime.ramulator_ddr import derive_ddr_burst_bytes, derive_ddr_row_bytes
    from src.token_lifecycle_trace import _modelinfos, _pim_config

    _run, hardware, ramulator, cpu, pim, models = load_runtime_bundle()
    model_config = models[sweeplib.model_name(model)]
    ramulator = replace(ramulator, org=PLAN_DRAM_ORG)
    pim_config = _pim_config(model_config, hardware, cpu, pim, ramulator)
    io_bits_per_edge = chips * int(pim_config["dq_bits_per_chip"])
    pim_config.update(
        {
            "BITACC_CHIP_COUNT": chips,
            "bank_count_per_chip": bank_pes,
            "controller_io_bits_per_edge": io_bits_per_edge,
            "rank_io_bits_per_clock": 2 * io_bits_per_edge,  # Two DDR edges per clock.
            "qkt_reduce_additions_per_wave": chips - 1,
        }
    )
    modelinfos = _modelinfos(model_config)
    layers = int(modelinfos["ndec"])
    modelinfos["ndec"] = 1
    plan = build_latest_command_plan(
        modelinfos=modelinfos,
        pim_config=pim_config,
        lin=kv_history,
        lout=1,
        burst_bytes=derive_ddr_burst_bytes(
            PLAN_DRAM_ORG, chips, io_bits_per_edge, int(ramulator.internal_prefetch_size)
        ),
        row_bytes=derive_ddr_row_bytes(PLAN_DRAM_ORG, chips),
        bank_layout={"total_banks": bank_pes},
        topology={"channels": int(ramulator.channel), "ranks": int(ramulator.rank)},
        allow_hardware_scaling=True,
    )
    parameters = plan["parameters"]
    return {
        "final_cycle": int(plan["host_pre_cycles"])
        + layers * int(plan["block_cycles"])
        + int(plan["host_post_cycles"]),
        "cycle_breakdown": {
            "embedding_cycles": int(plan["host_pre_cycles"]),
            "per_transformer_block_cycles": int(plan["block_cycles"]),
            "transformer_block_count": layers,
            "lm_head_cycles": int(plan["host_post_cycles"]),
        },
        "mapping": {
            "head_dim_per_chip": int(parameters["Dhc"]),
            "head_waves": int(parameters["head_waves"]),
            "heads_per_wave": list(parameters["heads_per_wave"]),
            "weight_rows_per_layer": int(parameters["weight_rows_per_layer"]),
            "kv_rows_per_layer": int(parameters["kv_rows_per_layer"]),
        },
        "internal_payload_bytes_per_token": layers
        * _internal_payload_bytes(plan, chips, bank_pes),
        "peak_internal_bytes_per_s": bank_pes
        * chips
        * int(parameters["bank_row_bytes"])
        * int(parameters["bank_pe_clock_hz"]),
    }


def run_case(
    case: Case,
    *,
    runs_dir: Path = sweeplib.RUNS_DIR,
    revision: str | None = None,
    **_simulator_options: Any,
) -> dict[str, Any]:
    """Evaluate the command plan of one case and store its result.

    Takes the place of :func:`sweeplib.run_case`; the options of a simulator run
    (``keep_traces``, ``timeout_s``, ``stop``) do not apply to a command plan.
    """
    run_dir = case.run_dir(runs_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    point = plan_point(
        case.model, case.kv_history, case.labels["chips"], case.labels["bank_pes"]
    )
    tck_ns = sweeplib.clock_period_ns(case.overrides)
    latency_ns, tokens_per_s = sweeplib.timing(point["final_cycle"], tck_ns)
    result = {
        "schema_version": sweeplib.RESULT_SCHEMA_VERSION,
        "status": "ok",
        **case.definition(),
        "model_name": sweeplib.model_name(case.model),
        "cycle_source": CYCLE_SOURCE,
        "clock_period_ns": tck_ns,
        "latency_ns": latency_ns,
        "tokens_per_s": tokens_per_s,
        "dram_org": PLAN_DRAM_ORG,
        **point,
        "wall_time_s": round(time.monotonic() - started, 3),
        "git_commit": revision,
        "platform": platform.platform(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sweeplib.write_json_atomic(run_dir / sweeplib.RESULT_NAME, result)
    return result


def scaling_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows of scaling.csv: throughput of every point normalized to its reference points."""
    points = {
        (
            result["model"],
            result["kv_history"],
            result["labels"]["chips"],
            result["labels"]["bank_pes"],
        ): result
        for result in results
    }
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for model, kv, chips, bank_pes in sorted(
        points, key=lambda key: (MODELS.index(key[0]), *key[1:])
    ):
        references = [
            points.get((model, kv, CHIPS[0], BANK_PES[0])),
            points.get((model, kv, CHIPS[0], bank_pes)),
            points.get((model, kv, chips, BANK_PES[0])),
        ]
        if any(reference is None for reference in references):
            continue
        result = points[(model, kv, chips, bank_pes)]
        throughput = result["tokens_per_s"]
        smallest, one_chip, four_bank_pes = (ref["tokens_per_s"] for ref in references)
        resources = (chips * bank_pes) / (CHIPS[0] * BANK_PES[0])
        achieved = result["internal_payload_bytes_per_token"] * throughput
        rows[(model, kv, chips, bank_pes)] = {
            "model": model,
            "kv_history": kv,
            "chips": chips,
            "bank_pes": bank_pes,
            "final_cycle": result["final_cycle"],
            "tokens_per_s": throughput,
            "normalized_throughput": throughput / smallest,
            "parallel_efficiency_pct": 100.0 * throughput / smallest / resources,
            "speedup_over_1_chip": throughput / one_chip,
            "speedup_over_4_bank_pes": throughput / four_bank_pes,
            "peak_internal_bandwidth_tb_s": result["peak_internal_bytes_per_s"] / 1e12,
            "achieved_internal_bandwidth_tb_s": achieved / 1e12,
            "internal_bandwidth_utilization_pct": 100.0
            * achieved
            / result["peak_internal_bytes_per_s"],
        }

    summaries: list[dict[str, Any]] = []
    for kv in sorted({key[1] for key in rows}):
        for chips in CHIPS:
            for bank_pes in BANK_PES:
                keys = [(model, kv, chips, bank_pes) for model in MODELS]
                if not all(key in rows for key in keys):
                    continue
                summary: dict[str, Any] = {
                    "model": GEOMEAN_LABEL,
                    "kv_history": kv,
                    "chips": chips,
                    "bank_pes": bank_pes,
                }
                for column in RATIO_COLUMNS:
                    summary[column] = sweeplib.geomean(rows[key][column] for key in keys)
                summaries.append(summary)
    return [*rows.values(), *summaries]


def _is_scaling_point(result: Mapping[str, Any]) -> bool:
    chips, bank_pes = result["labels"].get("chips"), result["labels"].get("bank_pes")
    return (
        result.get("cycle_source") == CYCLE_SOURCE
        and chips in CHIPS
        and bank_pes in BANK_PES
        and result["overrides"] == _overrides(chips, bank_pes)
    )


def parse(runs_dir: Path, csv_dir: Path) -> list[Path]:
    results = [
        result for result in sweeplib.load_results(runs_dir, NAME) if _is_scaling_point(result)
    ]
    return [sweeplib.write_csv(csv_dir / SCALING_CSV, COLUMNS, scaling_rows(results))]
