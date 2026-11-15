"""Energy per token of the BitAcc system (R6).

The energy is computed from the statistics of the 84 base configurations simulated by
``context``; this experiment adds no simulation.  For every run, the model charges

* the host: its active power while it executes its own operations (the host cycles of
  ``cycle_breakdown``) and its idle power for the rest of the decoding time;
* the DRAM array: one row cycle per bank and chip for every activate/precharge pair, and
  every byte read from or written to the array by the bank-level PEs;
* the transfers: every byte moved between the DIMM-level PE and the chips, and between the
  host and the DIMM;
* the refresh: every all-bank refresh command, per chip, as the current above active
  standby for tRFC1 (``nRFC1`` of the simulator configuration of the run);
* the DRAM background: the precharge-standby power of every chip for the decoding time;
* the PEs: an energy per operation (ternary projection, attention, table generation), an
  energy per byte read from or written to the DIMM-level buffers, and the leakage of all
  PEs and buffers for the decoding time.

Command counts are those of the simulator (``full_trace.cmd_counter``); bytes and operation
counts follow from the per-command terms of the trace generator (``phase_report``).  The
constants are listed, with their sources, in ``config/energy.json`` and below.  The model
is an analytical estimate with assumed host powers, not a measurement.

The comparison uses the published energy per token of BitNet.cpp on an Apple M2 Ultra
(``baselines/bitnet_cpp_energy_m2_ultra.csv``), which exists for three configurations.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import sweeplib
from sweeplib import GEOMEAN_LABEL, KV_HISTORIES, MODELS, Case

NAME = "energy"

ENERGY_CSV = "energy_per_token.csv"
SENSITIVITY_CSV = "energy_sensitivity.csv"
CSV_KEYS = {
    ENERGY_CSV: ("model", "kv_history"),
    SENSITIVITY_CSV: ("host_idle_power_w", "model"),
}

CONSTANTS_PATH = sweeplib.ROOT / "config" / "energy.json"
RUNTIME_CONFIG_PATH = sweeplib.ROOT / "config" / "pim" / "bitacc_runtime_default.json"
BASELINE_PATH = sweeplib.BASELINES_DIR / "bitnet_cpp_energy_m2_ultra.csv"
SIMULATOR_CONFIG_NAME = "trace_out.yaml"

# Per-event energy coefficients, read from RUNTIME_CONFIG_PATH.  They are the coefficients
# of the AttAcc simulator (Park et al., ASPLOS 2024, github.com/scale-snu/attacc_simulator),
# whose DRAM terms come from Fine-Grained DRAM (O'Connor et al., MICRO 2017):
ROW_CYCLE_PJ_PER_BIT = "dram_activation_precharge_pj_per_bit"  # activate + precharge of a row
ARRAY_ACCESS_PJ_PER_BIT = "dram_cell_rw_pj_per_bit"  # read/write between array and bank PE
# 1.01 (column to bank group) + 1.23 (bank group to global I/O) + 0.30 (off-die wiring):
MODULE_TRANSFER_PJ_PER_BIT = "module_transfer_pj_per_bit"
HOST_TRANSFER_PJ_PER_BIT = "external_transfer_pj_per_bit"  # off-die wiring only
PROJECTION_PJ_PER_OPERATION = "bitlinear_energy_pj_per_operation"
ATTENTION_PJ_PER_OPERATION = "attention_mac_energy_pj_per_operation"
TABLE_PJ_PER_OPERATION = "lut_generation_energy_pj_per_operation"
BUFFER_READ_PJ_PER_BYTE = "sram_read_pj_per_byte"
BUFFER_WRITE_PJ_PER_BYTE = "sram_write_pj_per_byte"

# Simulator commands.  An all-bank command acts on every bank of every chip.
ACTIVATE_COMMANDS = ("ACT", "ACTAB", "ACTSB", "ACTPB")
PRECHARGE_COMMANDS = ("PRE", "PREA", "PRESB", "PREPB")
ALL_BANK_ROW_COMMANDS = ("ACTAB", "PREA")
REFRESH_COMMAND = "REFab"
UNMODELED_COMMANDS = ("REFsb",)  # no same-bank refresh current in config/energy.json
MODULE_TRANSFER_COMMANDS = ("QKT_XFER", "POST_SCORE_BCAST", "PV_XFER", "LINEAR_XFER")
KV_WRITE_COMMANDS = ("ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT")
ATTENTION_COMMANDS = ("ALL_BANK_QKT", "ALL_BANK_PV")
HOST_CYCLE_FIELDS = (
    "embedding_cycles",
    "positional_encoding_cycles",
    "lm_head_cycles",
    "runtime_dispatch_host_overhead_cycles",
)

COMPONENTS = (
    "host_active_j",
    "host_idle_j",
    "dram_array_j",
    "dram_transfer_j",
    "dram_refresh_j",
    "dram_background_j",
    "pe_logic_j",
    "pe_buffer_j",
    "pe_leakage_j",
)
ENERGY_COLUMNS = (
    "model",
    "kv_history",
    "final_cycle",
    "latency_s",
    "host_active_s",
    *COMPONENTS,
    "total_j",
    "average_power_w",
    "bitnet_cpp_j_per_token",
    "energy_vs_bitnet_cpp",
    "reduction_over_bitnet_cpp",
)
SENSITIVITY_COLUMNS = (
    "host_idle_power_w",
    "model",
    "bitacc_j_per_token",
    "bitnet_cpp_j_per_token",
    "reduction_over_bitnet_cpp",
)


@dataclass(frozen=True)
class Activity:
    """Bytes moved and operations executed during one decoded token."""

    array_bytes: int
    module_transfer_bytes: int
    host_transfer_bytes: int
    projection_operations: int
    attention_operations: int
    table_operations: int
    buffer_read_bytes: int
    buffer_write_bytes: int


def cases(args: argparse.Namespace) -> list[Case]:
    return [
        sweeplib.base_case(model, kv)
        for model in sweeplib.select_models(args)
        for kv in sweeplib.select_kv(args, KV_HISTORIES)
    ]


def load_constants() -> dict[str, Any]:
    """Host powers, PE leakage and DDR5 device currents (config/energy.json)."""
    return json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))


def load_coefficients() -> dict[str, float]:
    """Per-event energy coefficients (pJ) of the runtime configuration."""
    runtime = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    names = (
        ROW_CYCLE_PJ_PER_BIT,
        ARRAY_ACCESS_PJ_PER_BIT,
        MODULE_TRANSFER_PJ_PER_BIT,
        HOST_TRANSFER_PJ_PER_BIT,
        PROJECTION_PJ_PER_OPERATION,
        ATTENTION_PJ_PER_OPERATION,
        TABLE_PJ_PER_OPERATION,
        BUFFER_READ_PJ_PER_BYTE,
        BUFFER_WRITE_PJ_PER_BYTE,
    )
    return {name: float(runtime[name]) for name in names}


def load_bitnet_cpp_energy() -> dict[str, float]:
    """Published BitNet.cpp energy per token (J): model -> value."""
    return {row["model"]: float(row["j_per_token"]) for row in sweeplib.read_csv(BASELINE_PATH)}


def refresh_cycles(run_dir: Path) -> int:
    """Duration of an all-bank refresh (``nRFC1``, clock cycles) in the simulator configuration."""
    text = (run_dir / SIMULATOR_CONFIG_NAME).read_text(encoding="utf-8")
    values = re.findall(r"^\s*nRFC1:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{run_dir / SIMULATOR_CONFIG_NAME}: expected exactly one nRFC1 entry")
    return int(values[0])


def activity(summary: Mapping[str, Any]) -> Activity:
    """Count the bytes and operations of one decoded token from the generator statistics.

    Every entry of ``phase_report`` describes ``count`` identical commands; its
    ``formula_terms`` give the bytes or elements one command handles.  A bank-level command
    runs on ``active_banks`` banks of each of the ``C`` chips.
    """
    runtime = summary["runtime_inputs"]
    if runtime["command_count_scope"] != "full_decode_token" or runtime["lout"] != 1:
        raise ValueError("the energy model requires the trace of exactly one decoded token")
    chips = int(runtime["C"])
    weight_bits = int(runtime["weight_bits"])
    activation_bytes = int(runtime["activation_bytes"])
    transfer_bytes_per_clock = -(-int(runtime["rank_io_bits_per_clock"]) // 8)

    totals = dict.fromkeys(Activity.__dataclass_fields__, 0)
    for event in summary["phase_report"]:
        terms = event["formula_terms"]
        for token in event["trace_tokens"]:
            name, count = token["name"], int(token["count"])
            if name == "ALL_BANK_BITLINEAR":
                weight_bytes = count * terms["active_banks"] * chips * terms["array_access_bytes"]
                operations = weight_bytes * 8 // weight_bits
                totals["array_bytes"] += weight_bytes
                totals["projection_operations"] += operations
                # One table entry of activation width is read per pair of ternary weights.
                totals["buffer_read_bytes"] += (operations + 1) // 2 * activation_bytes
            elif name == "ALL_BANK_RD_K_V_CACHE" or name in KV_WRITE_COMMANDS:
                totals["array_bytes"] += (
                    count * terms["active_banks"] * chips * terms["array_access_bytes"]
                )
            elif name == "ALL_BANK_LUT_LOAD":
                elements = terms["lut_pairs"] * terms["lut_elements_per_pair"]
                tile_bytes = -(-elements * terms["lut_entry_bits"] // 8)
                totals["module_transfer_bytes"] += (
                    count * terms["lut_xfer_clocks"] * transfer_bytes_per_clock
                )
                totals["table_operations"] += count * elements
                totals["buffer_write_bytes"] += count * tile_bytes * terms["active_banks"] * chips
            elif name in MODULE_TRANSFER_COMMANDS:
                totals["module_transfer_bytes"] += (
                    count * terms["transfer_clocks"] * transfer_bytes_per_clock
                )
            elif name == "HOST_XFER":
                totals["host_transfer_bytes"] += (
                    count * terms["transfer_clocks"] * transfer_bytes_per_clock
                )
            elif name in ATTENTION_COMMANDS:
                totals["attention_operations"] += (
                    count * terms["active_banks"] * chips * terms["S"] * terms["Dhc"]
                )
            elif name == "QKT_REDUCE":
                totals["attention_operations"] += count * terms["logic_additions"]
    return Activity(**totals)


def row_events(counts: Mapping[str, int], commands: Sequence[str], banks: int, chips: int) -> int:
    """Number of (bank, chip) pairs on which the given row commands act."""
    return sum(
        int(counts.get(name, 0)) * (banks if name in ALL_BANK_ROW_COMMANDS else 1) * chips
        for name in commands
    )


def energy_components(
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    n_rfc1: int,
    constants: Mapping[str, Any],
    coefficients: Mapping[str, float],
    host_idle_power_w: float,
) -> dict[str, float]:
    """Energy per token (J) of every component, plus ``latency_s`` and ``host_active_s``."""
    runtime = summary["runtime_inputs"]
    counts = summary["full_trace"]["cmd_counter"]
    banks, chips = int(runtime["banks"]), int(runtime["C"])
    clock_s = result["clock_period_ns"] * 1e-9
    latency_s = result["final_cycle"] * clock_s
    host_active_s = sum(result["cycle_breakdown"][name] for name in HOST_CYCLE_FIELDS) * clock_s
    if not 0.0 <= host_active_s <= latency_s:
        raise ValueError(f"{result['case_id']}: host time {host_active_s} s exceeds the latency")
    unmodeled = [name for name in UNMODELED_COMMANDS if counts.get(name)]
    if unmodeled:
        raise ValueError(f"{result['case_id']}: no energy model for the commands {unmodeled}")

    work = activity(summary)
    row_cycle_pj = int(runtime["bank_row_bytes"]) * 8 * coefficients[ROW_CYCLE_PJ_PER_BIT]
    # A row cycle is charged in two halves, to the activation and to the precharge.
    row_pj = (
        row_events(counts, ACTIVATE_COMMANDS, banks, chips)
        + row_events(counts, PRECHARGE_COMMANDS, banks, chips)
    ) * (row_cycle_pj / 2.0)
    array_pj = work.array_bytes * 8 * coefficients[ARRAY_ACCESS_PJ_PER_BIT]
    transfer_pj = (
        work.module_transfer_bytes * 8 * coefficients[MODULE_TRANSFER_PJ_PER_BIT]
        + work.host_transfer_bytes * 8 * coefficients[HOST_TRANSFER_PJ_PER_BIT]
    )
    logic_pj = (
        work.projection_operations * coefficients[PROJECTION_PJ_PER_OPERATION]
        + work.attention_operations * coefficients[ATTENTION_PJ_PER_OPERATION]
        + work.table_operations * coefficients[TABLE_PJ_PER_OPERATION]
    )
    buffer_pj = (
        work.buffer_read_bytes * coefficients[BUFFER_READ_PJ_PER_BYTE]
        + work.buffer_write_bytes * coefficients[BUFFER_WRITE_PJ_PER_BYTE]
    )

    device, host, leakage = constants["ddr5_device"], constants["host"], constants["pe_leakage"]
    refresh_w = 1e-3 * (  # per chip, above active standby
        device["vdd_v"] * (device["idd5b_ma"] - device["idd3n_ma"])
        + device["vpp_v"] * (device["ipp5b_ma"] - device["ipp3n_ma"])
        + device["vddq_v"] * (device["iddq5b_ma"] - device["iddq3n_ma"])
    )
    background_w = 1e-3 * chips * (  # precharge standby
        device["vdd_v"] * device["idd2n_ma"]
        + device["vpp_v"] * device["ipp2n_ma"]
        + device["vddq_v"] * device["iddq2n_ma"]
    )
    leakage_w = (
        banks * chips * leakage["bank_pe_w"] + leakage["dimm_pe_w"] + leakage["dimm_buffers_w"]
    )
    refresh_chip_commands = int(counts.get(REFRESH_COMMAND, 0)) * chips
    return {
        "latency_s": latency_s,
        "host_active_s": host_active_s,
        "host_active_j": host["active_power_w"] * host_active_s,
        "host_idle_j": host_idle_power_w * (latency_s - host_active_s),
        "dram_array_j": (row_pj + array_pj) * 1e-12,
        "dram_transfer_j": transfer_pj * 1e-12,
        "dram_refresh_j": refresh_chip_commands * refresh_w * n_rfc1 * clock_s,
        "dram_background_j": background_w * latency_s,
        "pe_logic_j": logic_pj * 1e-12,
        "pe_buffer_j": buffer_pj * 1e-12,
        "pe_leakage_j": leakage_w * latency_s,
    }


def total_energy(components: Mapping[str, float], host_idle_power_w: float | None = None) -> float:
    """Energy per token (J); ``host_idle_power_w`` replaces the idle power of ``components``."""
    if host_idle_power_w is None:
        return sum(components[name] for name in COMPONENTS)
    idle_s = components["latency_s"] - components["host_active_s"]
    return host_idle_power_w * idle_s + sum(
        components[name] for name in COMPONENTS if name != "host_idle_j"
    )


def energy_rows(
    runs: Mapping[tuple[str, int], Mapping[str, Any]], bitnet_cpp: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Rows of energy_per_token.csv from the components of every finished base run."""
    histories = sorted({kv for _, kv in runs})
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for kv in histories:
            components = runs.get((model, kv))
            if components is None:
                continue
            total = total_energy(components)
            published = bitnet_cpp.get(model)  # None: no published energy for this model.
            rows.append(
                {
                    "model": model,
                    "kv_history": kv,
                    **components,
                    "total_j": total,
                    "average_power_w": total / components["latency_s"],
                    "bitnet_cpp_j_per_token": published,
                    "energy_vs_bitnet_cpp": None if published is None else total / published,
                    "reduction_over_bitnet_cpp": None if published is None else published / total,
                }
            )

    ratios = {
        (row["model"], row["kv_history"]): row["energy_vs_bitnet_cpp"]
        for row in rows
        if row["energy_vs_bitnet_cpp"] is not None
    }
    for kv in histories:
        if all((model, kv) in ratios for model in bitnet_cpp):
            ratio = sweeplib.geomean(ratios[(model, kv)] for model in bitnet_cpp)
            rows.append(
                {
                    "model": GEOMEAN_LABEL,
                    "kv_history": kv,
                    "energy_vs_bitnet_cpp": ratio,
                    "reduction_over_bitnet_cpp": 1.0 / ratio,
                }
            )
    return rows


def sensitivity_rows(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    bitnet_cpp: Mapping[str, float],
    host_idle_powers_w: Sequence[float],
) -> list[dict[str, Any]]:
    """Rows of energy_sensitivity.csv: reduction over BitNet.cpp for every host idle power.

    The energy of a model is the geometric mean over the KV histories of the paper; a
    model is listed only when all of them are finished.
    """
    rows: list[dict[str, Any]] = []
    for idle_w in host_idle_powers_w:
        reductions = []
        for model in bitnet_cpp:
            if not all((model, kv) in runs for kv in KV_HISTORIES):
                continue
            energy = sweeplib.geomean(
                total_energy(runs[(model, kv)], idle_w) for kv in KV_HISTORIES
            )
            reductions.append(bitnet_cpp[model] / energy)
            rows.append(
                {
                    "host_idle_power_w": idle_w,
                    "model": model,
                    "bitacc_j_per_token": energy,
                    "bitnet_cpp_j_per_token": bitnet_cpp[model],
                    "reduction_over_bitnet_cpp": reductions[-1],
                }
            )
        if len(reductions) == len(bitnet_cpp):
            rows.append(
                {
                    "host_idle_power_w": idle_w,
                    "model": GEOMEAN_LABEL,
                    "reduction_over_bitnet_cpp": sweeplib.geomean(reductions),
                }
            )
    return rows


def parse(runs_dir: Path, csv_dir: Path) -> list[Path]:
    constants, coefficients = load_constants(), load_coefficients()
    bitnet_cpp = load_bitnet_cpp_energy()
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for result in sweeplib.load_results(runs_dir, "context"):
        if result["overrides"]:
            continue
        run_dir = runs_dir / result["experiment"] / result["case_id"]
        components = energy_components(
            result,
            sweeplib.load_trace_summary(runs_dir, result),
            refresh_cycles(run_dir),
            constants,
            coefficients,
            constants["host"]["idle_power_w"],
        )
        runs[(result["model"], result["kv_history"])] = {
            "final_cycle": result["final_cycle"],
            **components,
        }
    return [
        sweeplib.write_csv(csv_dir / ENERGY_CSV, ENERGY_COLUMNS, energy_rows(runs, bitnet_cpp)),
        sweeplib.write_csv(
            csv_dir / SENSITIVITY_CSV,
            SENSITIVITY_COLUMNS,
            sensitivity_rows(runs, bitnet_cpp, constants["host"]["idle_power_sensitivity_w"]),
        ),
    ]
