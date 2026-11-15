"""Contribution of operand locality and breakdown of the decoding time (R3).

Both analyses use the command trace of the base design (shared with ``context``) at KV
histories of 128, 512, and 1024 tokens.  Every interval between two issued commands is
charged to the command that starts it.  ``trace_statistics`` reduces the trace to the
cycles per command while the trace still exists; ``parse`` works on these statistics.

``locality_breakdown.csv`` compares four design points that use the same compute units and
differ only in which operands are local to them:

* ``no_pim``: weights and KV cache cross the off-chip channel,
* ``fc_pim``: the weights are local, the KV cache crosses the channel,
* ``attention_pim``: the KV cache is local, the weights cross the channel,
* ``bitacc``: both are local (the simulated design).

Only ``bitacc`` is simulated.  The other design points are derived from it: an operand that
is not local is streamed over the off-chip channel, which takes its number of bytes divided
by the channel bandwidth; of this time, the part that exceeds the computation on the
operand is off-chip data movement.  ``fc_pim`` and ``attention_pim`` additionally move
Q, K, V, and the attention output between the two domains once per layer.

``time_breakdown.csv`` gives the share of every activity in the decoding time of BitAcc.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import sweeplib
from sweeplib import MODELS, Case

NAME = "locality"

LOCALITY_CSV = "locality_breakdown.csv"
TIME_CSV = "time_breakdown.csv"
CSV_KEYS = {
    LOCALITY_CSV: ("model", "history_tokens", "design"),
    TIME_CSV: ("model", "kv_history"),
}

KV_HISTORIES = (128, 512, 1024)
MEAN_LABEL = "Mean"
DESIGNS = ("no_pim", "fc_pim", "attention_pim", "bitacc")
COMPONENTS = (
    "fc_computing",
    "attention_computing",
    "onchip_data_movement",
    "offchip_data_movement",
    "others",
)

# Categories of the trace commands.  Commands that are not listed (host operations,
# refresh, markers) belong to OTHER.
FC_ONCHIP = "fc_onchip"
ATTENTION_ONCHIP = "attention_onchip"
FC_COMMUNICATION = "fc_communication"
ATTENTION_COMMUNICATION = "attention_communication"
SHARED_COMMUNICATION = "shared_communication"
OTHER = "etc"
CATEGORY_COMMANDS: dict[str, tuple[str, ...]] = {
    FC_ONCHIP: (
        "QUANT",
        "ALL_BANK_BITLINEAR",
        "ALL_BANK_DEQUANT",
        "RMS_MLP",
        "MLP_FUSION",
        "FINAL_RES",
    ),
    ATTENTION_ONCHIP: (
        "RMS_ATTN",
        "ALL_BANK_ROPE",
        "ALL_BANK_WR_K_CURRENT",
        "ALL_BANK_WR_V_CURRENT",
        "ALL_BANK_RD_K_V_CACHE",
        "ALL_BANK_QKT",
        "QKT_REDUCE",
        "ALL_BANK_POST",
        "ALL_BANK_PV",
    ),
    FC_COMMUNICATION: ("ALL_BANK_LUT_LOAD", "LINEAR_XFER"),
    ATTENTION_COMMUNICATION: ("QKT_XFER", "POST_SCORE_BCAST", "PV_XFER"),
    SHARED_COMMUNICATION: ("HOST_XFER",),
}
CATEGORY = {
    command: category for category, commands in CATEGORY_COMMANDS.items() for command in commands
}
ACTIVATE = "ACTAB"
PRECHARGE = "PREA"
MARKERS = ("BLOCK_END", "TOKEN_DONE")  # Take no time; never start an interval of their own.
TRACE_START = "TRACE_START"  # Pseudo command: the cycles before the first command.

FC_COMPUTE_COMMANDS = ("ALL_BANK_BITLINEAR", "ALL_BANK_DEQUANT")
ATTENTION_COMPUTE_COMMANDS = ("ALL_BANK_ROPE", "ALL_BANK_QKT", "ALL_BANK_PV")
KV_COMMANDS = ("ALL_BANK_RD_K_V_CACHE", "ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT")
WEIGHT_COMMAND = "ALL_BANK_BITLINEAR"

# Activities of the decoding-time breakdown.  Commands that are not listed belong to HOST.
HOST = "host"
ACTIVITY_COMMANDS: dict[str, tuple[str, ...]] = {
    "bank_bitlinear": FC_COMPUTE_COMMANDS,
    "bank_attention": (*ATTENTION_COMPUTE_COMMANDS, *KV_COMMANDS),
    "dimm_tiles": ("ALL_BANK_LUT_LOAD",),
    "dimm_vector": (
        "QUANT",
        "RMS_ATTN",
        "RMS_MLP",
        "MLP_FUSION",
        "FINAL_RES",
        "QKT_REDUCE",
        "ALL_BANK_POST",
    ),
    "transfer": ("LINEAR_XFER", "QKT_XFER", "PV_XFER", "POST_SCORE_BCAST"),
    "dram": (ACTIVATE, PRECHARGE, "REFab"),
}
ACTIVITY = {
    command: activity for activity, commands in ACTIVITY_COMMANDS.items() for command in commands
}
ACTIVITIES = (*ACTIVITY_COMMANDS, HOST)
LEVELS = {
    "bank_level": ("bank_bitlinear", "bank_attention"),
    "dimm_level": ("dimm_tiles", "dimm_vector"),
}

LOCALITY_COLUMNS = (
    "model",
    "history_tokens",
    "design",
    *(f"{name}_cycles" for name in COMPONENTS),
    "total_cycles",
    "total_ms_per_token",
    "normalized_time_vs_no_pim",
    "normalized_time_within_case_max",
    "normalization_reference_design",
    "normalization_reference_cycles",
    "speedup_vs_no_pim",
    "controller_freq_hz",
    "offchip_bandwidth_bytes_per_s",
    "fc_trace_accounted_bytes",
    "attention_trace_accounted_bytes",
    "partial_fc_attention_boundary_bytes",
    "partial_fc_attention_boundary_cycles",
    "full_bitacc_trace_cycles",
    *(f"{name}_normalized" for name in COMPONENTS),
)
TIME_COLUMNS = (
    "model",
    "kv_history",
    "total_cycles",
    *(f"{name}_cycles" for name in ACTIVITIES),
    *(f"{name}_percent" for name in ACTIVITIES),
    *(f"{name}_percent" for name in LEVELS),
)


def cases(args: argparse.Namespace) -> list[Case]:
    histories = sweeplib.select_kv(args, KV_HISTORIES)
    return [
        sweeplib.base_case(model, kv) for model in sweeplib.select_models(args) for kv in histories
    ]


# --------------------------------------------------------------------------------------
# Statistics of the command trace
# --------------------------------------------------------------------------------------


def _owners(trace: Path) -> Iterator[tuple[int, str]]:
    """Issue cycle and owning command of every interval of a command trace.

    Commands issued in the same cycle form one interval, owned by the last of them that is
    not a marker.
    """
    cycle = -1
    owner = ""
    with trace.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split(",", 2)
            command = fields[1].strip() if len(fields) > 1 else ""
            if not fields[0].strip().isdigit() or not command:
                raise ValueError(f"{trace.name}:{number}: malformed trace line")
            issued = int(fields[0])
            if issued < cycle:
                raise ValueError(f"{trace.name}:{number}: cycles are not sorted")
            if issued > cycle:
                if owner:
                    yield cycle, owner
                cycle, owner = issued, command
            elif command not in MARKERS or owner in MARKERS:
                owner = command
    if owner:
        yield cycle, owner


def trace_statistics(trace: Path, final_cycle: int) -> dict[str, Any]:
    """Cycles of a command trace per command and category.

    An interval belongs to the category of its owning command.  The interval of a row
    activation is charged to the next and that of a precharge to the previous command of
    one of the categories other than OTHER, because it opens or closes the rows for it.
    """
    intervals: dict[tuple[str, str], list[int]] = {}

    def charge(command: str, category: str, cycles: int, count: int = 1) -> None:
        entry = intervals.setdefault((command, category), [0, 0])
        entry[0] += count
        entry[1] += cycles

    neighbour = OTHER
    activations = [0, 0]  # Number and cycles of the activations that wait for their command.
    previous: tuple[int, str] | None = None
    for cycle, owner in chain(_owners(trace), [(final_cycle, "")]):
        if previous is None:
            if not owner:
                raise ValueError(f"{trace.name}: empty command trace")
            if cycle > 0:
                charge(TRACE_START, OTHER, cycle)
        else:
            start, command = previous
            if cycle < start:
                raise ValueError(f"{trace.name}: ends after the final cycle {final_cycle}")
            if command == ACTIVATE:
                activations[0] += 1
                activations[1] += cycle - start
            elif command == PRECHARGE:
                charge(command, neighbour, cycle - start)
            else:
                category = CATEGORY.get(command, OTHER)
                charge(command, category, cycle - start)
                if category != OTHER:
                    neighbour = category
                    if activations[0]:
                        charge(ACTIVATE, category, activations[1], activations[0])
                        activations = [0, 0]
        previous = (cycle, owner)
    if activations[0]:
        charge(ACTIVATE, OTHER, activations[1], activations[0])

    return {
        "total_cycles": final_cycle,
        "intervals": [
            {"command": command, "category": category, "count": count, "cycles": cycles}
            for (command, category), (count, cycles) in sorted(intervals.items())
        ],
    }


# --------------------------------------------------------------------------------------
# Operand locality
# --------------------------------------------------------------------------------------


def trace_components(statistics: Mapping[str, Any]) -> dict[str, int]:
    """Cycles of the simulated design per component of the locality analysis.

    The bank-level computation and the transfers are counted per command; row activation,
    precharge, and the KV accesses are on-chip data movement of their domain.  Everything
    else (DIMM-level PE, refresh, host) is ``others``.
    """
    cycles = dict.fromkeys(
        (
            "fc_computing",
            "attention_computing",
            "fc_onchip_movement",
            "attention_onchip_movement",
            FC_COMMUNICATION,
            ATTENTION_COMMUNICATION,
            SHARED_COMMUNICATION,
        ),
        0,
    )
    for interval in statistics["intervals"]:
        command, category = interval["command"], interval["category"]
        if category in (FC_COMMUNICATION, ATTENTION_COMMUNICATION, SHARED_COMMUNICATION):
            cycles[category] += interval["cycles"]
        elif command in FC_COMPUTE_COMMANDS:
            cycles["fc_computing"] += interval["cycles"]
        elif command in ATTENTION_COMPUTE_COMMANDS:
            cycles["attention_computing"] += interval["cycles"]
        elif command in KV_COMMANDS:
            cycles["attention_onchip_movement"] += interval["cycles"]
        elif command in (ACTIVATE, PRECHARGE) and category == FC_ONCHIP:
            cycles["fc_onchip_movement"] += interval["cycles"]
        elif command in (ACTIVATE, PRECHARGE) and category == ATTENTION_ONCHIP:
            cycles["attention_onchip_movement"] += interval["cycles"]
    cycles["others"] = statistics["total_cycles"] - sum(cycles.values())
    return cycles


def accessed_bytes(summary: Mapping[str, Any]) -> tuple[int, int]:
    """Bytes of the weights and of the KV cache that the bank-level PEs access per token."""
    chips = int(summary["runtime_inputs"]["C"])
    weight_bytes = kv_bytes = 0
    for event in summary["phase_report"]:
        for token in event["trace_tokens"]:
            if token["name"] != WEIGHT_COMMAND and token["name"] not in KV_COMMANDS:
                continue
            terms = event["formula_terms"]
            accessed = (
                int(token["count"])
                * int(terms["active_banks"])
                * chips
                * int(terms["array_access_bytes"])
            )
            if token["name"] == WEIGHT_COMMAND:
                weight_bytes += accessed
            else:
                kv_bytes += accessed
    return weight_bytes, kv_bytes


def design_cycles(
    trace: Mapping[str, int], weight_cycles: int, kv_cycles: int, boundary_cycles: int
) -> dict[str, dict[str, int]]:
    """Cycles per component of the four design points.

    ``weight_cycles`` and ``kv_cycles`` are the times to stream the operand over the
    off-chip channel.  The computation on an operand overlaps with its stream, so that only
    the remainder of the stream is counted as off-chip data movement.
    """
    weight_stream = max(0, weight_cycles - trace["fc_computing"])
    kv_stream = max(0, kv_cycles - trace["attention_computing"])
    shared = trace[SHARED_COMMUNICATION]
    movement = {  # design -> (on-chip, off-chip) data movement
        "no_pim": (0, weight_stream + kv_stream),
        "fc_pim": (
            trace["fc_onchip_movement"],
            kv_stream + trace[FC_COMMUNICATION] + shared + boundary_cycles,
        ),
        "attention_pim": (
            trace["attention_onchip_movement"],
            weight_stream + trace[ATTENTION_COMMUNICATION] + shared + boundary_cycles,
        ),
        "bitacc": (
            trace["fc_onchip_movement"] + trace["attention_onchip_movement"],
            trace[FC_COMMUNICATION] + trace[ATTENTION_COMMUNICATION] + shared,
        ),
    }
    return {
        design: {
            "fc_computing": trace["fc_computing"],
            "attention_computing": trace["attention_computing"],
            "onchip_data_movement": onchip,
            "offchip_data_movement": offchip,
            "others": trace["others"],
        }
        for design, (onchip, offchip) in movement.items()
    }


def locality_rows(
    result: Mapping[str, Any], summary: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Rows of locality_breakdown.csv (one per design point) of one base configuration."""
    runtime = summary["runtime_inputs"]
    memory = config["ramulator"]
    clock_hz = 1e9 / result["clock_period_ns"]
    bandwidth = (
        int(memory["channel"])
        * int(runtime["burst_bytes"])
        / (int(memory["nBL"]) * result["clock_period_ns"] * 1e-9)
    )
    weight_bytes, kv_bytes = accessed_bytes(summary)
    # Q, K, V, and the attention output, each of the hidden dimension, in every layer.
    value_bytes = math.ceil(int(config["hardware"]["data_w_bits"]) / 8)
    boundary_bytes = 4 * int(runtime["H"]) * value_bytes * int(runtime["L"])
    boundary_cycles = math.ceil(boundary_bytes / int(runtime["burst_bytes"]))

    statistics = result["trace_statistics"][NAME]
    designs = design_cycles(
        trace_components(statistics),
        weight_cycles=math.ceil(weight_bytes * clock_hz / bandwidth),
        kv_cycles=math.ceil(kv_bytes * clock_hz / bandwidth),
        boundary_cycles=boundary_cycles,
    )
    totals = {design: sum(cycles.values()) for design, cycles in designs.items()}
    slowest = max(DESIGNS, key=totals.__getitem__)
    rows: list[dict[str, Any]] = []
    for design in DESIGNS:
        row: dict[str, Any] = {
            "model": result["model"],
            "history_tokens": result["kv_history"],
            "design": design,
            "total_cycles": totals[design],
            "total_ms_per_token": 1000.0 * totals[design] / clock_hz,
            "normalized_time_vs_no_pim": totals[design] / totals["no_pim"],
            "normalized_time_within_case_max": totals[design] / totals[slowest],
            "normalization_reference_design": slowest,
            "normalization_reference_cycles": totals[slowest],
            "speedup_vs_no_pim": totals["no_pim"] / totals[design],
            "controller_freq_hz": clock_hz,
            "offchip_bandwidth_bytes_per_s": bandwidth,
            "fc_trace_accounted_bytes": weight_bytes,
            "attention_trace_accounted_bytes": kv_bytes,
            "partial_fc_attention_boundary_bytes": boundary_bytes,
            "partial_fc_attention_boundary_cycles": boundary_cycles,
            "full_bitacc_trace_cycles": statistics["total_cycles"],
        }
        for name in COMPONENTS:
            row[f"{name}_cycles"] = designs[design][name]
            row[f"{name}_normalized"] = designs[design][name] / totals[slowest]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Decoding-time breakdown
# --------------------------------------------------------------------------------------


def time_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows of time_breakdown.csv: share of every activity in the decoding time of BitAcc."""
    by_key = {(result["model"], result["kv_history"]): result for result in results}
    rows: list[dict[str, Any]] = []
    for kv in sorted({kv for _, kv in by_key}):
        shares: list[dict[str, float]] = []
        for model in MODELS:
            result = by_key.get((model, kv))
            if result is None:
                continue
            statistics = result["trace_statistics"][NAME]
            cycles = dict.fromkeys(ACTIVITIES, 0)
            for interval in statistics["intervals"]:
                cycles[ACTIVITY.get(interval["command"], HOST)] += interval["cycles"]
            total = statistics["total_cycles"]
            share = {name: 100.0 * cycles[name] / total for name in ACTIVITIES}
            for level, activities in LEVELS.items():
                share[level] = sum(share[name] for name in activities)
            shares.append(share)
            rows.append(
                {
                    "model": model,
                    "kv_history": kv,
                    "total_cycles": total,
                    **{f"{name}_cycles": value for name, value in cycles.items()},
                    **{f"{name}_percent": value for name, value in share.items()},
                }
            )
        if len(shares) == len(MODELS):
            mean: dict[str, Any] = {"model": MEAN_LABEL, "kv_history": kv}
            for name in (*ACTIVITIES, *LEVELS):
                mean[f"{name}_percent"] = sum(share[name] for share in shares) / len(shares)
            rows.append(mean)
    return rows


def parse(runs_dir: Path, csv_dir: Path) -> list[Path]:
    results = [
        result
        for result in sweeplib.load_results(runs_dir, "context")
        if not result["overrides"] and NAME in result.get("trace_statistics", {})
    ]
    results.sort(key=lambda result: (MODELS.index(result["model"]), result["kv_history"]))
    config = json.loads(sweeplib.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    locality = [
        row
        for result in results
        for row in locality_rows(result, sweeplib.load_trace_summary(runs_dir, result), config)
    ]
    return [
        sweeplib.write_csv(csv_dir / LOCALITY_CSV, LOCALITY_COLUMNS, locality),
        sweeplib.write_csv(csv_dir / TIME_CSV, TIME_COLUMNS, time_rows(results)),
    ]
