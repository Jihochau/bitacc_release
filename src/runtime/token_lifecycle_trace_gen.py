#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

from .dse import _load_dse_config
from .dse_fused_trace_gen import (
    _addr_for_bank,
    _autotune_bank_shift,
    _bitlinear_trace_plan,
    _extract_memory_cycles,
    _make_yaml,
    _parse_cmd_counter,
    _parse_cmd_trace,
    _parse_run_stats,
    _run_ramulator,
)
from .ramulator_ddr import (
    ddr_bank_groups_for_org,
    ddr_banks_per_group_for_org,
    derive_ddr_burst_bytes,
    derive_ddr_row_bytes,
    derive_trace_address_layout,
)
from .pim_micro_ops import rtl_bank_pe as rtl_bank_ops
from .pim_micro_ops import rtl_dimm_pe as rtl_dimm_ops
from ..cycle_model import CycleModelParams, transformer_13_phase_cycle_model


def _emit_reqs(
    f,
    token: str,
    bank_id: int,
    nbytes: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    base_addr: int,
    bank_req_idx: Dict[int, int],
) -> int:
    nbytes = max(0, int(nbytes))
    if nbytes <= 0:
        return 0
    bursts = int(math.ceil(nbytes / float(max(1, burst_bytes))))
    for _ in range(bursts):
        addr = _addr_for_bank(
            bank_id=bank_id,
            req_idx_for_bank=bank_req_idx[bank_id],
            bank_shift=bank_shift,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            base_addr=base_addr,
            total_banks=max(1, len(bank_req_idx)),
        )
        bank_req_idx[bank_id] += 1
        f.write(f"{token} 0x{addr:x}\n")
    return bursts


class _TraceBuffer:
    """Tiny file-like sink used to reuse the existing trace emitters."""

    def __init__(self):
        self.lines: List[str] = []

    def write(self, text: str) -> None:
        for line in text.splitlines():
            line = line.strip()
            if line:
                self.lines.append(line)


def _resolve_channelized_artifact_paths(path_hint: str | None) -> List[str]:
    if not path_hint:
        return []
    if os.path.exists(path_hint):
        return [path_hint]
    channel_glob = f"{path_hint}.ch*"
    if str(path_hint).endswith(".ch0"):
        channel_glob = f"{str(path_hint)[:-4]}.ch*"
    return sorted(glob.glob(channel_glob))


def _write_aggregated_cmd_counter(counter_files: List[str], aggregate_path: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for counter_file in counter_files:
        with open(counter_file, "r", encoding="utf-8") as f:
            for line in f:
                if "," not in line:
                    continue
                name, raw_val = line.split(",", 1)
                name = name.strip()
                raw_val = raw_val.strip()
                if not name:
                    continue
                try:
                    counts[name] = counts.get(name, 0) + int(raw_val)
                except ValueError:
                    continue
    with open(aggregate_path, "w", encoding="utf-8") as f:
        for name in sorted(counts):
            f.write(f"{name}, {int(counts[name])}\n")
    return counts


def _write_merged_cmd_trace(trace_files: List[str], merged_path: str) -> str | None:
    if not trace_files:
        return None
    merged_lines: List[Tuple[int, int, str]] = []
    for file_idx, trace_file in enumerate(trace_files):
        with open(trace_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    cycle = int(line.split(",", 1)[0].strip())
                except ValueError:
                    cycle = 10**18
                merged_lines.append((cycle, file_idx, line))
    merged_lines.sort(key=lambda item: (item[0], item[1], item[2]))
    with open(merged_path, "w", encoding="utf-8") as f:
        for _, _, line in merged_lines:
            f.write(line + "\n")
    return merged_path


def _run_runtime_trace_lines(
    lines: Iterable[str],
    out_dir: str,
    tag: str,
    ramulator_bin: str,
    dram_impl: str,
    dram_org: str,
    dram_timing: str,
    chips_per_dimm: int,
    channel_width: int | None,
    channel: int,
    rank: int,
    n_lut_bcast: int,
    n_rdpim: int,
    n_macab: int,
    timeout_s: int,
    extra_timings: Dict[str, int] | None = None,
    extra_commands_to_count: List[str] | None = None,
    keep_yaml: bool = False,
    keep_cmd_counter: bool = False,
    keep_cmd_trace: bool = False,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{tag}.trace")
    run_out_path = os.path.join(out_dir, f"{tag}_run.out")

    with open(trace_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    yaml_path = os.path.join(out_dir, f"{tag}.yaml") if keep_yaml else None
    cmd_counter_path = os.path.join(out_dir, f"{tag}_cmd_counter.txt") if keep_cmd_counter else None
    cmd_trace_prefix = os.path.join(out_dir, tag) if keep_cmd_trace else None

    with tempfile.TemporaryDirectory() as td:
        tmp_yaml_path = yaml_path or os.path.join(td, f"{tag}.yaml")
        tmp_cmd_counter_path = cmd_counter_path or os.path.join(td, f"{tag}_cmd_counter.txt")
        tmp_cmd_trace_prefix = cmd_trace_prefix or os.path.join(td, tag)

        yaml_text = _make_yaml(
            path_trace=os.path.abspath(trace_path),
            path_cmd_counter=os.path.abspath(tmp_cmd_counter_path),
            path_cmd_trace_prefix=os.path.abspath(tmp_cmd_trace_prefix),
            enable_refresh=True,
            n_lut_bcast=int(n_lut_bcast),
            n_rdpim=int(n_rdpim),
            n_macab=int(n_macab),
            n_refi=3900,
            n_rfc1=295,
            chips_per_dimm=int(chips_per_dimm),
            channel_width=(None if channel_width is None else int(channel_width)),
            extra_timings=extra_timings,
            extra_commands_to_count=extra_commands_to_count,
        )
        # Override the DRAM preset from main.py runtime configs.
        yaml_text = yaml_text.replace("preset: DDR5_8Gb_x8", f"preset: {dram_org}")
        yaml_text = yaml_text.replace("preset: DDR5_3200AN", f"preset: {dram_timing}")
        yaml_text = yaml_text.replace("impl: DDR5-BITACC", f"impl: {dram_impl}")
        yaml_text = yaml_text.replace("channel: 1", f"channel: {int(channel)}")
        yaml_text = yaml_text.replace("rank: 2", f"rank: {int(rank)}")
        with open(tmp_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        timed_out = False
        out_text = ""
        try:
            out_text = _run_ramulator(ramulator_bin, tmp_yaml_path, timeout_s)
        except subprocess.TimeoutExpired as te:
            timed_out = True
            out_text = (te.stdout or "") if isinstance(te.stdout, str) else ""
            out_text += "\n[runtime-token-lifecycle] Ramulator run timed out.\n"
        except subprocess.CalledProcessError as cpe:
            out_text = cpe.output if isinstance(cpe.output, str) else ""
            if not out_text:
                out_text = f"[runtime-token-lifecycle] Ramulator failed with exit code {cpe.returncode}.\n"
            with open(run_out_path, "w", encoding="utf-8") as f:
                f.write(out_text)
            raise RuntimeError(
                f"Ramulator failed for tag={tag}; see {run_out_path}"
            ) from cpe

        counter_files = _resolve_channelized_artifact_paths(tmp_cmd_counter_path)
        cmd_trace_files = _resolve_channelized_artifact_paths(f"{tmp_cmd_trace_prefix}.ch0")

        if cmd_counter_path and len(counter_files) > 1:
            _write_aggregated_cmd_counter(counter_files, cmd_counter_path)
        elif cmd_counter_path and len(counter_files) == 1 and counter_files[0] != cmd_counter_path:
            _write_aggregated_cmd_counter(counter_files, cmd_counter_path)

        merged_cmd_trace_path = None
        if cmd_trace_prefix and cmd_trace_files:
            merged_target = (
                f"{cmd_trace_prefix}.merged"
                if len(cmd_trace_files) > 1
                else cmd_trace_files[0]
            )
            if len(cmd_trace_files) > 1:
                merged_cmd_trace_path = _write_merged_cmd_trace(cmd_trace_files, merged_target)
            else:
                merged_cmd_trace_path = merged_target

        cmd_counter = _parse_cmd_counter(tmp_cmd_counter_path)
        cmd_analysis = _parse_cmd_trace(f"{tmp_cmd_trace_prefix}.ch0")

    with open(run_out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    return {
        "timed_out": timed_out,
        "memory_system_cycles": _extract_memory_cycles(out_text),
        "cmd_counter": cmd_counter,
        "cmd_analysis": cmd_analysis,
        "run_stats": _parse_run_stats(out_text),
        "files": {
            "trace": trace_path,
            "yaml": yaml_path,
            "run_out": run_out_path,
            "cmd_counter": cmd_counter_path,
            "cmd_counter_channels": counter_files,
            "cmd_trace": merged_cmd_trace_path if cmd_trace_prefix is not None else None,
            "cmd_trace_channels": cmd_trace_files,
        },
    }


def _cycles_to_phase_entry(label: str, start_cycle: int, cycles: int, key_events: List[Tuple[str, int]]) -> Dict:
    cycles = max(0, int(cycles))
    start = int(start_cycle)
    end = start + max(0, cycles) - 1 if cycles > 0 else start - 1
    return {
        "label": label,
        "start_cycle": start,
        "end_cycle": end,
        "cycles": cycles,
        "key_events": [{"name": name, "cycle": int(cyc)} for name, cyc in key_events],
    }


def _ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = max(1, int(b))
    return int(math.ceil(a / float(b)))


def _trace_ctrl_freq_mhz(pim_config: Dict) -> float:
    return float(
        pim_config.get(
            "BITACC_TRACE_CTRL_FREQ_MHZ",
            pim_config.get("BITACC_TRACE_MEM_FREQ_MHZ", 1600.0),
        )
    )


def _trace_bank_pe_freq_mhz(pim_config: Dict) -> float:
    return float(pim_config.get("BITACC_TRACE_BANK_PE_FREQ_MHZ", 200.0))


def _trace_dimm_pe_freq_mhz(pim_config: Dict) -> float:
    return float(
        pim_config.get(
            "BITACC_TRACE_DIMM_PE_FREQ_MHZ",
            pim_config.get("BITACC_TRACE_PE_FREQ_MHZ", 666.0),
        )
    )


def _pe_to_ctrl_cycles(pe_cycles: int, ctrl_freq_mhz: float, pe_freq_mhz: float) -> int:
    pe_cycles = max(0, int(pe_cycles))
    if pe_cycles <= 0:
        return 0
    ratio = float(ctrl_freq_mhz) / max(1e-9, float(pe_freq_mhz))
    return int(math.ceil(pe_cycles * ratio))


def _bank_pe_to_mem_cycles(pe_cycles: int, pim_config: Dict) -> int:
    return _pe_to_ctrl_cycles(
        pe_cycles,
        _trace_ctrl_freq_mhz(pim_config),
        _trace_bank_pe_freq_mhz(pim_config),
    )


def _dimm_pe_to_mem_cycles(pe_cycles: int, pim_config: Dict) -> int:
    return _pe_to_ctrl_cycles(
        pe_cycles,
        _trace_ctrl_freq_mhz(pim_config),
        _trace_dimm_pe_freq_mhz(pim_config),
    )


def _pe_to_mem_cycles(pe_cycles: int, pim_config: Dict) -> int:
    return _dimm_pe_to_mem_cycles(pe_cycles, pim_config)


def _phase_entry_from_pe(label: str,
                         start_cycle_mem: int,
                         pe_cycles: int,
                         key_events_pe: List[Tuple[str, int]],
                         pim_config: Dict) -> Dict:
    mem_cycles = _pe_to_mem_cycles(pe_cycles, pim_config)
    start = int(start_cycle_mem)
    end = start + max(0, mem_cycles) - 1 if mem_cycles > 0 else start - 1
    key_events = []
    for name, rel_pe in key_events_pe:
        rel_pe = max(0, int(rel_pe))
        ev_cycle = start + max(0, _pe_to_mem_cycles(rel_pe, pim_config) - 1)
        key_events.append({"name": name, "cycle": int(ev_cycle), "pe_cycle": int(rel_pe)})
    return {
        "label": label,
        "start_cycle": start,
        "end_cycle": end,
        "cycles": int(mem_cycles),
        "logic_cycles": int(mem_cycles),
        "memory_cycles": 0,
        "pe_cycles": int(pe_cycles),
        "key_events": key_events,
    }


def _append_trace_report_to_run_out(
    run_out_path: str,
    phase_report: List[Dict],
    expected_cmd_counts: Dict[str, int],
    observed_cmd_counts: Dict[str, int],
    ramulator_memory_cycles: int,
    formula_memory_cycles: int,
) -> None:
    with open(run_out_path, "a", encoding="utf-8") as f:
        f.write("\n=== TRACE COMMAND COUNTER ===\n")
        for cmd in sorted(observed_cmd_counts):
            expected = int(expected_cmd_counts.get(cmd, 0))
            observed = int(observed_cmd_counts.get(cmd, 0))
            mark = "OK" if expected == observed else "MISMATCH"
            f.write(f"{cmd}: observed={observed} expected={expected} [{mark}]\n")

        f.write("\n=== TRACE CYCLE SUMMARY ===\n")
        f.write(f"formula_memory_cycles: {int(formula_memory_cycles)}\n")
        f.write(f"ramulator_memory_cycles: {int(ramulator_memory_cycles)}\n")
        if ramulator_memory_cycles >= 0:
            f.write(
                "ramulator_minus_formula_cycles: {}\n".format(
                    int(ramulator_memory_cycles - formula_memory_cycles)
                )
            )

        f.write("\n=== PHASE SUMMARY ===\n")
        for item in phase_report:
            f.write(
                "{}: cycles={} logic_cycles={} nominal_trace_memory={}\n".format(
                    item["label"],
                    int(item.get("cycles", 0)),
                    int(item.get("logic_cycles", 0)),
                    int(item.get("memory_cycles", 0)),
                )
            )


def _append_trace_report_markers(
    *,
    trace_path: str | None,
    cmd_trace_path: str | None,
    markers: List[Dict],
    end_cycle: int,
) -> None:
    if end_cycle < 0:
        return
    if trace_path:
        with open(trace_path, "a", encoding="utf-8") as f:
            if markers:
                f.write("\n# ABSTRACT_COMMANDS\n")
            for marker in markers:
                f.write(
                    "@{} {} report_cycles={} end_cycle={}\n".format(
                        int(marker["start_cycle"]),
                        str(marker["command"]),
                        int(marker["cycles"]),
                        int(marker["end_cycle"]),
                    )
                )
            f.write("@{} TRANSFORMER_BLOCK_END\n".format(int(end_cycle)))
    if cmd_trace_path:
        with open(cmd_trace_path, "a", encoding="utf-8") as f:
            if markers:
                f.write("\n")
            for marker in markers:
                f.write(
                    "{}, {}, abstract, report_cycles={}, end_cycle={}\n".format(
                        int(marker["start_cycle"]),
                        str(marker["command"]),
                        int(marker["cycles"]),
                        int(marker["end_cycle"]),
                    )
                )
            f.write("{}, TRANSFORMER_BLOCK_END\n".format(int(end_cycle)))


def _resolve_runtime_output_end_cycle(
    *,
    memory_system_cycles: int,
    cmd_trace_path: str | None,
    phase_report: List[Dict],
) -> int:
    if int(memory_system_cycles) >= 0:
        return int(memory_system_cycles)

    if cmd_trace_path and os.path.exists(cmd_trace_path):
        last_cycle = -1
        with open(cmd_trace_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or "abstract" in line or "TRANSFORMER_BLOCK_END" in line:
                    continue
                parts = [x.strip() for x in line.split(",", 1)]
                if not parts or not parts[0]:
                    continue
                try:
                    last_cycle = max(last_cycle, int(parts[0]))
                except ValueError:
                    continue
        if last_cycle >= 0:
            return int(last_cycle)

    if phase_report:
        return int(phase_report[-1]["end_cycle"])
    return -1


def _write_command_trace_file(path: str, phase_report: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# TOKEN LIFECYCLE COMMAND TRACE\n")
        for item in phase_report:
            f.write(
                "@{} CMD[{}] {} cycles={} logic_cycles={} nominal_trace_memory={} formula={}\n".format(
                    int(item.get("start_cycle", 0)),
                    int(item.get("command_id", -1)),
                    str(item.get("command", "")),
                    int(item.get("cycles", 0)),
                    int(item.get("logic_cycles", 0)),
                    int(item.get("memory_cycles", 0)),
                    str(item.get("formula", "")),
                )
            )
            for term_name, term_cycles in item.get("formula_terms", {}).items():
                f.write("  TERM {}={}\n".format(str(term_name), int(term_cycles)))
            for tok in item.get("trace_tokens", []):
                f.write(
                    "  TRACE {} count={} pe_each={} mem_each={} pe_total={} mem_total={}\n".format(
                        str(tok.get("name", "")),
                        int(tok.get("count", 0)),
                        int(tok.get("pe_cycles_each", 0)),
                        int(tok.get("memory_cycles_each", 0)),
                        int(tok.get("pe_cycles_total", 0)),
                        int(tok.get("memory_cycles_total", 0)),
                    )
                )
            for ev in item.get("key_events", []):
                f.write(
                    "  EVENT {} cycle={} pe_cycle={}\n".format(
                        str(ev.get("name", "")),
                        int(ev.get("cycle", 0)),
                        int(ev.get("pe_cycle", 0)),
                    )
                )
            f.write("\n")


def _runtime_burst_bytes(
    pim_config: Dict,
    *,
    dram_org: str,
    chips_per_dimm: int,
    channel_width: int | None,
) -> int:
    explicit_burst = pim_config.get("BITACC_TRACE_BURST_BYTES")
    if explicit_burst is not None:
        return max(1, int(explicit_burst))
    return derive_ddr_burst_bytes(
        dram_org=dram_org,
        chips_per_dimm=max(1, int(chips_per_dimm)),
        channel_width=(None if channel_width is None else int(channel_width)),
    )


_WEIGHT_TRACE_BASE_ADDR = {
    "qkv": 0x0,
    "wo": 0x0,
    "wup": 0x0,
    "wup_wgate": 0x0,
    "wgate": 0x0,
    "wdown": 0x0,
}

_KV_TRACE_BASE_ADDR = {
    "k_current": 0x0,
    "v_current": 0x0,
    "k_history": 0x0,
    "v_history": 0x0,
}

_STAGE_TRACE_BASE_ADDR = {
    "startup": 0x34000000,
    "pre_attn_norm": 0x34400000,
    "pre_attn_norm_tail": 0x34800000,
    "rope": 0x34C00000,
    "dequant": 0x34E00000,
    "qkt": 0x35000000,
    "post": 0x35400000,
    "pv": 0x35800000,
    "pre_mlp_norm": 0x35C00000,
    "pre_mlp_norm_tail": 0x36000000,
    "mlp_fusion": 0x36400000,
    "mlp_fusion_tail": 0x36800000,
}

_DEFAULT_STAGE_TRACE_BASE_ADDR = 0x36C00000

_WEIGHT_TENSOR_ROW_BASE = {
    "qkv": 0,
    "wo": 1024,
    "wup": 2048,
    "wup_wgate": 2048,
    "wgate": 3072,
    "wdown": 4096,
}

_DIMM_LEVEL_STAGE_TAGS = {
    "pre_attn_norm",
    "pre_attn_norm_tail",
    "pre_mlp_norm",
    "pre_mlp_norm_tail",
    "mlp_fusion",
    "mlp_fusion_tail",
}


def _resolve_bank_layout(dram_org: str, banks: int) -> Dict[str, int]:
    bankgroups = ddr_bank_groups_for_org(dram_org)
    banks_per_group = ddr_banks_per_group_for_org(dram_org)
    total_banks = max(1, int(bankgroups) * int(banks_per_group))
    if total_banks <= 0:
        total_banks = max(1, int(banks))
        bankgroups = int(total_banks)
        banks_per_group = 1
    return {
        "bankgroups": int(bankgroups),
        "banks_per_group": int(banks_per_group),
        "total_banks": int(total_banks),
    }


def _row_bytes_for_org(dram_org: str, chips_per_dimm: int) -> int:
    return derive_ddr_row_bytes(dram_org, chips_per_dimm)


def _row_col_from_ordinal(ordinal: int, burst_bytes: int, row_bytes: int) -> Tuple[int, int]:
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    ordinal = max(0, int(ordinal))
    return int(ordinal // bursts_per_row), int((ordinal % bursts_per_row) * max(1, int(burst_bytes)))


def _resolve_memory_topology(channel: int, rank: int) -> Dict[str, int]:
    channels = max(1, int(channel))
    ranks = max(1, int(rank))
    return {
        "channels": channels,
        "ranks": ranks,
        "slots": int(channels * ranks),
    }


def _channel_rank_for_trace(bank_id: int, stripe_ordinal: int, topology: Dict[str, int]) -> Tuple[int, int]:
    channels = max(1, int(topology.get("channels", 1)))
    ranks = max(1, int(topology.get("ranks", 1)))
    slots = max(1, int(topology.get("slots", channels * ranks)))
    stripe = (max(0, int(bank_id)) + max(0, int(stripe_ordinal))) % slots
    channel_id = stripe % channels
    rank_id = (stripe // channels) % ranks
    return int(channel_id), int(rank_id)


def _bank_addr_vec(
    bank_id: int,
    row_idx: int,
    col_idx: int,
    bank_layout: Dict[str, int],
    topology: Dict[str, int],
    stripe_ordinal: int = 0,
) -> List[int]:
    total_banks = max(1, int(bank_layout["total_banks"]))
    bank_id = max(0, int(bank_id)) % total_banks
    banks_per_group = max(1, int(bank_layout["banks_per_group"]))
    channel_id, rank_id = _channel_rank_for_trace(bank_id, stripe_ordinal, topology)
    return [
        int(channel_id),
        int(rank_id),
        int(bank_id // banks_per_group),
        int(bank_id % banks_per_group),
        max(0, int(row_idx)),
        max(0, int(col_idx)),
    ]


def _all_bank_addr_vec(row_idx: int, col_idx: int, topology: Dict[str, int], stripe_ordinal: int = 0) -> List[int]:
    channel_id, rank_id = _channel_rank_for_trace(0, stripe_ordinal, topology)
    return [int(channel_id), int(rank_id), -1, -1, max(0, int(row_idx)), max(0, int(col_idx))]


def _no_target_addr_vec(topology: Dict[str, int], stripe_ordinal: int = 0) -> List[int]:
    channel_id, rank_id = _channel_rank_for_trace(0, stripe_ordinal, topology)
    return [int(channel_id), int(rank_id), -1, -1, -1, -1]


def _compose_trace_addr(
    *,
    bank_id: int,
    row_idx: int,
    col_off: int,
    row_bytes: int,
    bank_shift: int,
    base_addr: int,
    total_banks: int = 1,
    row_shift: int | None = None,
) -> int:
    layout = derive_trace_address_layout(
        row_bytes=row_bytes,
        total_banks=total_banks,
        bank_shift=bank_shift,
        row_shift=row_shift,
    )
    return (
        int(base_addr)
        + (max(0, int(row_idx)) << int(layout["row_shift"]))
        + (max(0, int(bank_id)) << int(layout["bank_shift"]))
        + max(0, int(col_off))
    )


def _stage_trace_addr(
    *,
    stage_tag: str,
    ordinal: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    bank_id: int = 0,
) -> int:
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    ordinal = max(0, int(ordinal))
    row_idx = ordinal // bursts_per_row
    col_off = (ordinal % bursts_per_row) * burst_bytes
    base_addr = int(_STAGE_TRACE_BASE_ADDR.get(stage_tag, _DEFAULT_STAGE_TRACE_BASE_ADDR))
    return _compose_trace_addr(
        bank_id=bank_id,
        row_idx=row_idx,
        col_off=col_off,
        row_bytes=row_bytes,
        bank_shift=bank_shift,
        base_addr=base_addr,
    )


def _stage_trace_loc(
    *,
    stage_tag: str,
    ordinal: int,
    burst_bytes: int,
    row_bytes: int,
) -> Tuple[int, int]:
    if str(stage_tag) in _DIMM_LEVEL_STAGE_TAGS:
        return -1, -1
    _ = burst_bytes
    _ = row_bytes
    base_addr = int(_STAGE_TRACE_BASE_ADDR.get(stage_tag, _DEFAULT_STAGE_TRACE_BASE_ADDR))
    return int(base_addr >> 16), max(0, int(ordinal))


def _weight_lut_trace_addr(
    *,
    tensor_tag: str,
    tile_idx: int,
    bank_shift: int,
    row_bytes: int,
    burst_bytes: int,
    bank_id: int = 0,
) -> int:
    _ = row_bytes
    _ = burst_bytes
    return _compose_trace_addr(
        bank_id=bank_id,
        row_idx=max(0, int(tile_idx)),
        col_off=0,
        row_bytes=row_bytes,
        bank_shift=bank_shift,
        base_addr=int(_WEIGHT_TRACE_BASE_ADDR[tensor_tag]),
    )


def _weight_lut_trace_loc(*, tile_idx: int) -> Tuple[int, int]:
    _ = tile_idx
    return -1, -1


def _weight_rect_trace_addr(
    *,
    tensor_tag: str,
    tile_idx: int,
    group_idx: int,
    groups_per_tile: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    bank_id: int = 0,
) -> int:
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    groups_per_tile = max(1, int(groups_per_tile))
    groups_row_span = _ceil_div(groups_per_tile, bursts_per_row)
    group_idx = max(0, int(group_idx))
    row_idx = (max(0, int(tile_idx)) * groups_row_span) + (group_idx // bursts_per_row)
    col_off = (group_idx % bursts_per_row) * burst_bytes
    return _compose_trace_addr(
        bank_id=bank_id,
        row_idx=row_idx,
        col_off=col_off,
        row_bytes=row_bytes,
        bank_shift=bank_shift,
        base_addr=int(_WEIGHT_TRACE_BASE_ADDR[tensor_tag]),
    )


def _weight_rect_trace_loc(
    *,
    tensor_row_base: int,
    burst_ordinal: int,
    burst_bytes: int,
    row_bytes: int,
) -> Tuple[int, int]:
    row_idx, col_idx = _row_col_from_ordinal(burst_ordinal, burst_bytes, row_bytes)
    return int(tensor_row_base) + row_idx, col_idx


def _kv_trace_addr(
    *,
    kv_tag: str,
    token_slot: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    bank_id: int = 0,
) -> int:
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    token_slot = max(0, int(token_slot))
    row_idx = token_slot // bursts_per_row
    col_off = (token_slot % bursts_per_row) * burst_bytes
    return _compose_trace_addr(
        bank_id=bank_id,
        row_idx=row_idx,
        col_off=col_off,
        row_bytes=row_bytes,
        bank_shift=bank_shift,
        base_addr=int(_KV_TRACE_BASE_ADDR[kv_tag]),
    )


def _kv_trace_loc(
    *,
    token_slot: int,
    burst_bytes: int,
    row_bytes: int,
    row_base: int = 0,
) -> Tuple[int, int]:
    row_idx, col_idx = _row_col_from_ordinal(token_slot, burst_bytes, row_bytes)
    return int(row_base) + row_idx, col_idx


def _semantic_trace_addr_vec(
    op: Dict,
    ordinal: int,
    *,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    bank_layout: Dict[str, int],
    topology: Dict[str, int],
) -> List[int]:
    _ = bank_shift
    addr_kind = str(op.get("addr_kind", "stage"))
    bank_id = int(op.get("bank_id", 0))
    stripe_ordinal = int(op.get("topology_ordinal_base", 0)) + max(0, int(ordinal))

    if addr_kind == "weight_lut":
        return _no_target_addr_vec(topology, stripe_ordinal=stripe_ordinal)

    if addr_kind == "weight_rect":
        row_idx, col_idx = _weight_rect_trace_loc(
            tensor_row_base=int(op.get("tensor_row_base", 0)),
            burst_ordinal=int(op.get("burst_ordinal", ordinal)),
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
        )
        if str(op.get("token", "")).startswith("ALL_BANK_"):
            # Use a concrete representative bank address for ALL_BANK weight
            # fetch commands so DDR row-open/precharge machinery (ACTAB/PREA)
            # can be resolved safely when the command is modeled as a column
            # access requiring all banks to be open.
            return _bank_addr_vec(bank_id, row_idx, col_idx, bank_layout, topology, stripe_ordinal)
        return _bank_addr_vec(bank_id, row_idx, col_idx, bank_layout, topology, stripe_ordinal)

    if addr_kind == "kv":
        token_slot = int(op.get("token_slot", 0))
        if bool(op.get("advance_with_ordinal", False)):
            token_slot += max(0, int(ordinal))
        row_idx, col_idx = _kv_trace_loc(
            token_slot=token_slot,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            row_base=int(op.get("row_base", 0)),
        )
        if str(op.get("token", "")).startswith("ALL_BANK_"):
            # Use a concrete representative bank address so Ramulator can
            # recurse into bank-level row-open logic (ACTAB/PREA) while the
            # command semantics still mean "all banks at this row/column".
            return _bank_addr_vec(bank_id, row_idx, col_idx, bank_layout, topology, stripe_ordinal)
        return _bank_addr_vec(bank_id, row_idx, col_idx, bank_layout, topology, stripe_ordinal)

    if str(op.get("token", "")).startswith("ALL_BANK_"):
        return _no_target_addr_vec(topology, stripe_ordinal=stripe_ordinal)
    row_idx, col_idx = _stage_trace_loc(
        stage_tag=str(op.get("stage_tag", op.get("token", "stage"))).lower(),
        ordinal=ordinal,
        burst_bytes=burst_bytes,
        row_bytes=row_bytes,
    )
    if row_idx < 0 or col_idx < 0:
        return _no_target_addr_vec(topology, stripe_ordinal=stripe_ordinal)
    return _bank_addr_vec(bank_id, row_idx, col_idx, bank_layout, topology, stripe_ordinal)


def _merge_token_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, val in src.items():
        dst[str(key)] = int(dst.get(str(key), 0)) + int(val)


def _count_input_trace_tokens(trace_path: str | None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not trace_path or not os.path.exists(trace_path):
        return counts
    with open(trace_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            token = line.split(None, 1)[0]
            counts[token] = counts.get(token, 0) + 1
    return counts


def _emit_phase_trace_ops(
    trace_ops: List[Dict],
    writer: _TraceBuffer,
    banks: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    bank_req_idx: Dict[int, int],
    token_timings: Dict[str, int],
    token_counts: Dict[str, int],
) -> None:
    for op in trace_ops:
        kind = str(op.get("kind", "pseudo"))
        token = str(op["token"])
        if kind == "pseudo":
            _emit_pseudo_repeated(
                writer=writer,
                token=token,
                count=int(op["count"]),
                bank_start=0,
                banks=banks,
                bank_shift=bank_shift,
                row_bytes=row_bytes,
                base_addr=int(op.get("base_addr", 0x70000000)),
                bank_req_idx=bank_req_idx,
                token_timings=token_timings,
                token_counts=token_counts,
                mem_cycles_each=int(op["memory_cycles_each"]),
            )
            continue

        if kind == "semantic_repeated":
            _emit_semantic_repeated(
                writer=writer,
                token=token,
                count=int(op["count"]),
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                token_timings=token_timings,
                token_counts=token_counts,
                mem_cycles_each=int(op["memory_cycles_each"]),
                op=op,
                topology=op["topology"],
            )
            continue

        if kind == "all_bank_bursts":
            bytes_per_bank = max(0, int(op.get("bytes_per_bank", 0)))
            if bytes_per_bank <= 0:
                continue
            base_addr = int(op.get("base_addr", 0x70000000))
            total_bursts = 0
            for bank_id in range(max(1, int(op.get("banks", banks)))):
                total_bursts += _emit_reqs(
                    f=writer,
                    token=token,
                    bank_id=bank_id,
                    nbytes=bytes_per_bank,
                    burst_bytes=burst_bytes,
                    row_bytes=row_bytes,
                    bank_shift=bank_shift,
                    base_addr=base_addr,
                    bank_req_idx=bank_req_idx,
                )
            token_counts[token] = token_counts.get(token, 0) + int(total_bursts)
            continue

        raise ValueError(f"Unsupported phase trace op kind={kind}")


def _emit_logic_tokens(
    writer: _TraceBuffer,
    token: str,
    total_mem_cycles: int,
    chunk_mem_cycles: int,
    bank_id: int,
    bank_shift: int,
    row_bytes: int,
    base_addr: int,
    bank_req_idx: Dict[int, int],
    token_timings: Dict[str, int],
    token_counts: Dict[str, int],
) -> None:
    total_mem_cycles = max(0, int(total_mem_cycles))
    if total_mem_cycles <= 0:
        return
    chunk_mem_cycles = max(1, int(chunk_mem_cycles))
    full_chunks = total_mem_cycles // chunk_mem_cycles
    tail_cycles = total_mem_cycles % chunk_mem_cycles
    if full_chunks > 0:
        token_timings[token] = chunk_mem_cycles
    for _ in range(full_chunks):
        addr = _addr_for_bank(
            bank_id=bank_id,
            req_idx_for_bank=bank_req_idx[bank_id],
            bank_shift=bank_shift,
            burst_bytes=row_bytes,
            row_bytes=row_bytes,
            base_addr=base_addr,
            total_banks=max(1, len(bank_req_idx)),
        )
        bank_req_idx[bank_id] += 1
        writer.write(f"{token} 0x{addr:x}\n")
        token_counts[token] = token_counts.get(token, 0) + 1
    if tail_cycles > 0:
        tail_token = f"{token}_TAIL"
        token_timings[tail_token] = tail_cycles
        addr = _addr_for_bank(
            bank_id=bank_id,
            req_idx_for_bank=bank_req_idx[bank_id],
            bank_shift=bank_shift,
            burst_bytes=64,
            row_bytes=row_bytes,
            base_addr=base_addr,
            total_banks=max(1, len(bank_req_idx)),
        )
        bank_req_idx[bank_id] += 1
        writer.write(f"{tail_token} 0x{addr:x}\n")
        token_counts[tail_token] = token_counts.get(tail_token, 0) + 1


def _emit_pseudo_repeated(
    writer: _TraceBuffer,
    token: str,
    count: int,
    bank_start: int,
    banks: int,
    bank_shift: int,
    row_bytes: int,
    base_addr: int,
    bank_req_idx: Dict[int, int],
    token_timings: Dict[str, int],
    token_counts: Dict[str, int],
    mem_cycles_each: int,
) -> int:
    count = max(0, int(count))
    mem_cycles_each = max(0, int(mem_cycles_each))
    if count <= 0 or mem_cycles_each <= 0:
        return int(bank_start)
    token_timings[token] = mem_cycles_each
    # These pseudo-commands model rank-wide/all-bank broadcasts. Use one
    # representative bank address stream instead of round-robining the bank id,
    # otherwise the trace falsely suggests bank-private commands.
    bank_cursor = int(bank_start)
    for _ in range(count):
        bank_id = 0
        addr = _addr_for_bank(
            bank_id=bank_id,
            req_idx_for_bank=bank_req_idx[bank_id],
            bank_shift=bank_shift,
            burst_bytes=64,
            row_bytes=row_bytes,
            base_addr=base_addr,
            total_banks=max(1, int(banks)),
        )
        bank_req_idx[bank_id] += 1
        writer.write(f"{token} 0x{addr:x}\n")
        token_counts[token] = token_counts.get(token, 0) + 1
        bank_cursor += 1
    return bank_cursor


def _emit_semantic_repeated(
    writer: _TraceBuffer,
    token: str,
    count: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    token_timings: Dict[str, int],
    token_counts: Dict[str, int],
    mem_cycles_each: int,
    op: Dict,
    topology: Dict[str, int],
) -> None:
    count = max(0, int(count))
    mem_cycles_each = max(0, int(mem_cycles_each))
    if count <= 0 or mem_cycles_each <= 0:
        return
    token_timings[token] = mem_cycles_each
    for ordinal in range(count):
        addr_vec = _semantic_trace_addr_vec(
            op=op,
            ordinal=ordinal,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            bank_layout=op["bank_layout"],
            topology=topology,
        )
        writer.write(f"{token} {' '.join(str(int(v)) for v in addr_vec)}\n")
        token_counts[token] = token_counts.get(token, 0) + 1


def _all_bank_fp16_write_bursts(elems_per_bank: int, burst_bytes: int) -> int:
    bytes_per_bank = max(0, int(elems_per_bank)) * 2
    return max(1, _ceil_div(bytes_per_bank, burst_bytes))


def _ddr_burst_mem_cycles() -> int:
    # DDR5 BL8: one data burst for 1024b on a 128B trace burst.
    return 8


def _abstract_bus_transfer_cycles(
    steps: int,
    *,
    parallel_lanes: int,
    value_bits: int,
    io_bits: int,
    ctrl_cycles: int = 1,
) -> int:
    steps = max(0, int(steps))
    parallel_lanes = max(1, int(parallel_lanes))
    value_bits = max(1, int(value_bits))
    io_bits = max(1, int(io_bits))
    ctrl_cycles = max(0, int(ctrl_cycles))
    per_step_cycles = _ceil_div(parallel_lanes * value_bits, io_bits)
    return int((steps * per_step_cycles) + ctrl_cycles)


def _abstract_vector_move_cycles(
    vec_elems: int,
    *,
    elems_per_cycle: int,
    ctrl_cycles: int = 1,
) -> int:
    vec_elems = max(0, int(vec_elems))
    elems_per_cycle = max(1, int(elems_per_cycle))
    ctrl_cycles = max(0, int(ctrl_cycles))
    return int(_ceil_div(vec_elems, elems_per_cycle) + ctrl_cycles)


def _sum_breakdown_cycles(items: List[Tuple[str, int]]) -> int:
    return int(sum(max(0, int(cycles)) for _, cycles in items))


def _dict_breakdown_items(raw: Dict | None) -> List[Tuple[str, int]]:
    if not isinstance(raw, dict):
        return []
    return [
        (str(name), max(0, int(cycles)))
        for name, cycles in raw.items()
        if max(0, int(cycles)) > 0
    ]


def _qkv_trace_formula(
    *,
    d_model: int,
    banks: int,
    outputs_per_launch: int,
    tile_scalars: int,
    launch_pe_cycles: int,
    trace_input_dim: int,
) -> Dict:
    trace_input_dim = max(1, min(int(d_model), int(trace_input_dim)))
    return _rtl_linear_formula(
        in_dim=trace_input_dim,
        out_dim=3 * trace_input_dim,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=0,
        local_cols_override=3 * _ceil_div(trace_input_dim, banks),
    )


def _emit_step_reqs(
    f,
    token: str,
    steps: int,
    bytes_per_step: int,
    banks: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    base_addr: int,
    bank_req_idx: Dict[int, int],
) -> int:
    total = 0
    for step_idx in range(max(0, int(steps))):
        bank_id = step_idx % max(1, int(banks))
        total += _emit_reqs(
            f=f,
            token=token,
            bank_id=bank_id,
            nbytes=bytes_per_step,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_addr=base_addr,
            bank_req_idx=bank_req_idx,
        )
    return total


def _rtl_linear_formula(in_dim: int,
                        out_dim: int,
                        banks: int,
                        outputs_per_launch: int,
                        tile_scalars: int,
                        launch_pe_cycles: int,
                        tail_pe_cycles: int,
                        local_cols_override: int = 0) -> Dict:
    frontend = rtl_dimm_ops.linear_frontend_formula(
        in_dim=in_dim,
        out_dim=out_dim,
        banks=banks,
        n_tile=outputs_per_launch,
        pairs_per_rd=max(1, _ceil_div(tile_scalars, 2) * max(1, int(outputs_per_launch))),
        local_cols_override=local_cols_override,
    )
    lut_load_pairs = int(frontend["k_tile_pairs"])
    tiles = int(frontend["tiles"])
    local_cols = int(frontend["local_cols"])
    groups_per_tile = int(frontend["groups_per_tile"])
    launches = int(frontend["launches"])
    load_pe_cycles = int(frontend["load_cycles"])
    launch_total_pe_cycles = launches * max(1, int(launch_pe_cycles))
    total_pe_cycles = load_pe_cycles + launch_total_pe_cycles + max(0, int(tail_pe_cycles))
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "banks": banks,
        "outputs_per_launch": outputs_per_launch,
        "tile_scalars": tile_scalars,
        "lut_load_pairs": lut_load_pairs,
        "tiles": tiles,
        "local_cols": local_cols,
        "groups_per_tile": groups_per_tile,
        "launches": launches,
        "load_pe_cycles": load_pe_cycles,
        "launch_pe_cycles": launch_total_pe_cycles,
        "tail_pe_cycles": int(tail_pe_cycles),
        "total_pe_cycles": total_pe_cycles,
    }


def _apply_linear_burst_scaling(
    formula: Dict,
    *,
    burst_bytes: int,
    reference_burst_bytes: int = 128,
) -> Dict:
    """Reduce memory-side linear fetch groups when DDR bursts get wider."""
    scaled = dict(formula)
    ref = max(1, int(reference_burst_bytes))
    burst = max(1, int(burst_bytes))
    burst_scale = max(1, int(_ceil_div(burst, ref)))
    groups_per_tile = max(1, int(formula["groups_per_tile"]))
    memory_groups_per_tile = max(1, int(_ceil_div(groups_per_tile, burst_scale)))
    scaled["reference_burst_bytes"] = int(ref)
    scaled["burst_scale"] = int(burst_scale)
    scaled["memory_groups_per_tile"] = int(memory_groups_per_tile)
    scaled["memory_total_bursts"] = int(max(1, int(formula["tiles"])) * memory_groups_per_tile)
    return scaled


def _weight_tensor_layouts(
    *,
    burst_bytes: int,
    row_bytes: int,
    formulas: List[Tuple[str, Dict]],
    explicit_layout: Dict[str, Dict[str, int]] | None = None,
) -> Dict[str, Dict[str, int]]:
    layouts: Dict[str, Dict[str, int]] = {}
    next_row_base = 0
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    explicit_layout = explicit_layout or {}
    for tensor_tag, formula in formulas:
        tag = str(tensor_tag)
        if tag in explicit_layout:
            row_base = int(explicit_layout[tag].get("row_base", next_row_base))
            row_span = int(explicit_layout[tag].get("rows", 0))
            row_span = max(1, row_span)
            total_bursts = int(row_span * bursts_per_row)
            layouts[tag] = {
                "row_base": row_base,
                "total_bursts": total_bursts,
                "row_span": row_span,
            }
            next_row_base = max(next_row_base, row_base + row_span)
            continue

        total_bursts = int(
            formula.get(
                "memory_total_bursts",
                max(1, int(formula["tiles"]) * int(formula["groups_per_tile"])),
            )
        )
        row_span = _ceil_div(total_bursts, bursts_per_row)
        layouts[tag] = {
            "row_base": int(next_row_base),
            "total_bursts": int(total_bursts),
            "row_span": int(row_span),
        }
        next_row_base += row_span
    return layouts


def _build_rtl_trace_cfg(modelinfos: Dict,
                         pim_config: Dict,
                         lin: int,
                         lout: int,
                         batch: int) -> Dict:
    d_model = int(modelinfos["hdim"])
    num_heads = int(modelinfos["num_heads"])
    head_dim = int(modelinfos.get("dhead", max(1, d_model // max(1, num_heads))))
    ffn_dim = int(round(float(modelinfos["ff_scale"]) * d_model))
    kv_cache_max_tokens = int(pim_config.get("BITACC_KV_CACHE_MAX_TOKENS", 0) or 0)
    include_current_token = bool(pim_config.get("BITACC_TRACE_INCLUDE_CURRENT_TOKEN", True))

    banks = max(1, int(pim_config.get("BITACC_TRACE_BANKS", pim_config.get("BITACC_NUM_BANKS", 32))))
    outputs_per_launch = max(1, int(pim_config.get(
        "BITACC_TRACE_OUTPUTS_PER_LAUNCH",
        pim_config.get("BITACC_PE_N_TILE", 1),
    )))
    pairs_per_rd = max(1, int(pim_config.get(
        "BITACC_TRACE_PAIRS_PER_RD",
        pim_config.get("BITACC_PE_PAIRS_PER_RD", pim_config.get("BITACC_TRACE_TILE_SCALARS", 256)),
    )))
    k_tile = max(1, _ceil_div(pairs_per_rd, outputs_per_launch))
    tile_scalars = max(2, 2 * k_tile)
    tok_par = max(1, int(pim_config.get("BITACC_TRACE_TOK_PAR", pim_config.get("BITACC_PE_TOK_PAR", 2))))
    qkt_tok_par = max(1, int(pim_config.get("BITACC_TRACE_QKT_TOK_PAR", tok_par)))
    tp_par = max(1, int(pim_config.get("BITACC_TRACE_TP_PAR", tok_par)))
    qkt_dot_par = max(1, int(pim_config.get("BITACC_TRACE_QKT_DOT_PAR", head_dim)))
    pv_dim_par = max(1, int(pim_config.get("BITACC_TRACE_PV_DIM_PAR", tok_par)))
    launch_pe_cycles = max(1, int(pim_config.get("BITACC_TRACE_LAUNCH_PE_CYCLES", rtl_dimm_ops.LINEAR_FRONTEND_RD_LATENCY)))
    # Audience-facing lifecycle traces model dequant as folded into the final
    # read/launch path instead of a standalone tail command.
    linear_tail_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_LINEAR_TAIL_PE_CYCLES", 19)))
    qkv_ready_tail_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_QKV_READY_TAIL_PE_CYCLES", rtl_bank_ops.launch_to_dequant_cycles())))
    down_tail_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_DOWN_TAIL_PE_CYCLES", 18)))
    mlp_residual_pe_cycles = max(
        0,
        int(pim_config.get("BITACC_TRACE_MLP_RESIDUAL_PE_CYCLES", d_model + 1)),
    )
    norm_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_NORM_CTRL_PE_CYCLES", rtl_dimm_ops.NORM_CTRL_CYCLES)))
    rope_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_ROPE_CTRL_PE_CYCLES", rtl_bank_ops.ROPE_CTRL_CYCLES)))
    qkt_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_QKT_CTRL_PE_CYCLES", rtl_bank_ops.QKT_CTRL_CYCLES)))
    post_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_POST_CTRL_PE_CYCLES", rtl_dimm_ops.POST_CTRL_CYCLES)))
    pv_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_PV_CTRL_PE_CYCLES", rtl_bank_ops.PV_CTRL_CYCLES)))
    startup_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_STARTUP_PE_CYCLES", rtl_dimm_ops.STARTUP_CYCLES)))
    mlp_fusion_ctrl_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_MLP_FUSION_CTRL_PE_CYCLES", rtl_dimm_ops.MLP_FUSION_CTRL_CYCLES)))
    cfg_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_CFG_PE_CYCLES", rtl_dimm_ops.CFG_CYCLES)))
    current_kv_write_lanes_per_cycle = max(1, int(pim_config.get("BITACC_TRACE_KV_WRITE_LANES_PER_CYCLE", 2)))
    dimm_buffer_write_lanes_per_cycle = max(1, int(pim_config.get("BITACC_TRACE_DIMM_BUFFER_WRITE_LANES_PER_CYCLE", 2)))
    qkv_local_cols = max(0, int(pim_config.get("BITACC_TRACE_QKV_LOCAL_COLS", 3 * _ceil_div(d_model, banks))))
    wo_local_cols = max(0, int(pim_config.get("BITACC_TRACE_WO_LOCAL_COLS", _ceil_div(d_model, banks))))
    mlp_local_cols = max(0, int(pim_config.get("BITACC_TRACE_MLP_LOCAL_COLS", _ceil_div(ffn_dim, banks))))
    down_local_cols = max(0, int(pim_config.get("BITACC_TRACE_DOWN_LOCAL_COLS", _ceil_div(d_model, banks))))
    mlp_dim = max(0, int(pim_config.get("BITACC_TRACE_MLP_DIM", 0)))
    if mlp_dim <= 0:
        if mlp_local_cols > 0:
            mlp_dim = banks * mlp_local_cols
        else:
            mlp_dim = ffn_dim
    mlp_dim = max(ffn_dim, mlp_dim)
    history_tokens = max(0, int(lin) + max(0, int(lout) - 1))
    if kv_cache_max_tokens > 0:
        history_tokens = min(history_tokens, kv_cache_max_tokens)
    visible_tokens = max(1, history_tokens + (1 if include_current_token else 0))
    bank_local_cols = _ceil_div(d_model, banks)
    local_cols_max = _ceil_div(mlp_dim, banks)
    cpu_token_embed_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_CPU_TOKEN_EMBED_PE_CYCLES", 0)))
    cpu_pos_enc_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_CPU_POS_ENC_PE_CYCLES", 0)))
    cpu_lm_head_pe_cycles = max(0, int(pim_config.get("BITACC_TRACE_CPU_LM_HEAD_PE_CYCLES", 0)))

    return {
        "batch": int(batch),
        "lin": int(lin),
        "lout": int(lout),
        "d_model": d_model,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "ffn_dim": ffn_dim,
        "mlp_dim": int(mlp_dim),
        "banks": banks,
        "pairs_per_rd": int(pairs_per_rd),
        "k_tile": int(k_tile),
        "bank_local_cols": int(bank_local_cols),
        "local_cols_max": int(local_cols_max),
        "outputs_per_launch": outputs_per_launch,
        "tile_scalars": tile_scalars,
        "tok_par": tok_par,
        "qkt_tok_par": int(qkt_tok_par),
        "tp_par": int(tp_par),
        "qkt_dot_par": int(qkt_dot_par),
        "pv_dim_par": int(pv_dim_par),
        "visible_history_tokens": int(history_tokens),
        "visible_tokens": int(visible_tokens),
        "kv_cache_max_tokens": int(kv_cache_max_tokens),
        "launch_pe_cycles": launch_pe_cycles,
        "linear_tail_pe_cycles": linear_tail_pe_cycles,
        "qkv_ready_tail_pe_cycles": qkv_ready_tail_pe_cycles,
        "down_tail_pe_cycles": down_tail_pe_cycles,
        "mlp_residual_pe_cycles": mlp_residual_pe_cycles,
        "norm_ctrl_pe_cycles": norm_ctrl_pe_cycles,
        "rope_ctrl_pe_cycles": rope_ctrl_pe_cycles,
        "qkt_ctrl_pe_cycles": qkt_ctrl_pe_cycles,
        "post_ctrl_pe_cycles": post_ctrl_pe_cycles,
        "pv_ctrl_pe_cycles": pv_ctrl_pe_cycles,
        "startup_pe_cycles": startup_pe_cycles,
        "cfg_pe_cycles": cfg_pe_cycles,
        "mlp_fusion_ctrl_pe_cycles": mlp_fusion_ctrl_pe_cycles,
        "current_kv_write_lanes_per_cycle": current_kv_write_lanes_per_cycle,
        "dimm_buffer_write_lanes_per_cycle": dimm_buffer_write_lanes_per_cycle,
        "qkv_local_cols": int(qkv_local_cols),
        "wo_local_cols": int(wo_local_cols),
        "mlp_local_cols": int(mlp_local_cols),
        "down_local_cols": int(down_local_cols),
        "cpu_token_embed_pe_cycles": cpu_token_embed_pe_cycles,
        "cpu_pos_enc_pe_cycles": cpu_pos_enc_pe_cycles,
        "cpu_lm_head_pe_cycles": cpu_lm_head_pe_cycles,
        "dequant_integrated_into_launch": True,
    }


def _emit_rtl_linear_phase(f,
                           phase_name: str,
                           in_dim: int,
                           out_dim: int,
                           banks: int,
                           outputs_per_launch: int,
                           tile_scalars: int,
                           burst_bytes: int,
                           row_bytes: int,
                           bank_shift: int,
                           base_lut_addr: int,
                           base_mac_addr: int,
                           bank_req_idx: Dict[int, int],
                           lut_req_idx: int) -> Tuple[Dict[str, int], int, Dict]:
    formula = _rtl_linear_formula(
        in_dim=in_dim,
        out_dim=out_dim,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=4,
        tail_pe_cycles=0,
    )

    counts = {"phase": phase_name, "BITACC_LUT_BCAST": 0, "BITACC_FC_RD": 0}
    for _tile_idx in range(int(formula["tiles"])):
        lut_addr = _addr_for_bank(
            bank_id=0,
            req_idx_for_bank=lut_req_idx,
            bank_shift=bank_shift,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            base_addr=base_lut_addr,
            total_banks=max(1, int(banks)),
        )
        lut_req_idx += 1
        f.write(f"BITACC_LUT_BCAST 0x{lut_addr:x}\n")
        counts["BITACC_LUT_BCAST"] += 1
        for group_idx in range(int(formula["groups_per_tile"])):
            bank_id = group_idx % max(1, int(banks))
            addr = _addr_for_bank(
                bank_id=bank_id,
                req_idx_for_bank=bank_req_idx[bank_id],
                bank_shift=bank_shift,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                base_addr=base_mac_addr,
                total_banks=max(1, int(banks)),
            )
            bank_req_idx[bank_id] += 1
            f.write(f"BITACC_FC_RD 0x{addr:x}\n")
            counts["BITACC_FC_RD"] += 1
    return counts, lut_req_idx, formula


def generate_runtime_token_lifecycle_trace(
    modelinfos: Dict,
    pim_config: Dict,
    mem_model,
    lin: int,
    lout: int,
    batch: int,
    out_dir: str,
    timeout_s: int | None = None,
    write_summary_json: bool = False,
) -> Dict:
    """Generate one-block RTL-style lifecycle validation trace.

    The phase schedule follows the current RTL realization confirmed by the
    project notes:
      phase 0 startup
      phase 13 pre-attn norm
      phase 1 ST_WAIT_QKV
      phase 2 ST_WAIT_QKT
      phase 3 ST_WAIT_POST
      phase 4 ST_WAIT_PV
      phase 14/15 CFG_WO + WAIT_WO
      phase 5 PRE_MLP_NORM
      phase 6/7 CFG_UP + WAIT_UP
      phase 8/9 CFG_GATE + WAIT_GATE
      phase 10 MLP_FUSION
      phase 11/12 CFG_DOWN + WAIT_DOWN
      phase 16 MLP_RESIDUAL

    Final MLP residual is now emitted as an explicit abstract command so
    trace readers can see the block-completion step directly, even though it
    is still modeled outside the DRAM request stream.

    All user-visible phase durations are reported in memory cycles after
    converting from PE cycles with:
      1 PE cycle @ 666 MHz = 3200 / 666 memory cycles @ 3.2 GHz
    """
    os.makedirs(out_dir, exist_ok=True)
    rtl = _build_rtl_trace_cfg(modelinfos, pim_config, lin, lout, batch)
    d_model = int(rtl["d_model"])
    num_heads = int(rtl["num_heads"])
    head_dim = int(rtl["head_dim"])
    ffn_dim = int(rtl["ffn_dim"])
    mlp_dim = int(rtl["mlp_dim"])
    banks = int(rtl["banks"])
    visible_tokens = int(rtl["visible_tokens"])
    visible_history_tokens = int(rtl["visible_history_tokens"])
    pairs_per_rd = int(rtl["pairs_per_rd"])
    k_tile = int(rtl["k_tile"])
    bank_local_cols = int(rtl["bank_local_cols"])
    local_cols_max = int(rtl["local_cols_max"])
    outputs_per_launch = int(rtl["outputs_per_launch"])
    tile_scalars = int(rtl["tile_scalars"])
    tok_par = int(rtl["tok_par"])
    qkt_tok_par = int(rtl["qkt_tok_par"])
    tp_par = int(rtl["tp_par"])
    qkt_dot_par = int(rtl["qkt_dot_par"])
    pv_dim_par = int(rtl["pv_dim_par"])
    launch_pe_cycles = int(rtl["launch_pe_cycles"])
    linear_tail_pe_cycles = int(rtl["linear_tail_pe_cycles"])
    qkv_ready_tail_pe_cycles = int(rtl["qkv_ready_tail_pe_cycles"])
    down_tail_pe_cycles = int(rtl["down_tail_pe_cycles"])
    mlp_residual_pe_cycles = int(rtl["mlp_residual_pe_cycles"])
    norm_ctrl_pe_cycles = int(rtl["norm_ctrl_pe_cycles"])
    rope_ctrl_pe_cycles = int(rtl["rope_ctrl_pe_cycles"])
    qkt_ctrl_pe_cycles = int(rtl["qkt_ctrl_pe_cycles"])
    post_ctrl_pe_cycles = int(rtl["post_ctrl_pe_cycles"])
    pv_ctrl_pe_cycles = int(rtl["pv_ctrl_pe_cycles"])
    startup_pe_cycles = int(rtl["startup_pe_cycles"])
    mlp_fusion_ctrl_pe_cycles = int(rtl["mlp_fusion_ctrl_pe_cycles"])
    cfg_pe_cycles = int(rtl["cfg_pe_cycles"])
    current_kv_write_lanes_per_cycle = int(rtl["current_kv_write_lanes_per_cycle"])
    trace_bus_io_bits = int(pim_config.get("BITACC_TRACE_IO_BITS", 64))
    trace_score_bits = int(pim_config.get("BITACC_TRACE_SCORE_BITS", 48))
    trace_prob_bits = int(
        pim_config.get(
            "BITACC_TRACE_PROB_BITS",
            pim_config.get("BITACC_RTL_SOFTMAX_PROB_BITS", 16),
        )
    )
    trace_vector_elems_per_cycle = int(pim_config.get("BITACC_TRACE_VECTOR_ELEMS_PER_CYCLE", 64))
    trace_vector_group_elems = int(pim_config.get("BITACC_TRACE_VECTOR_GROUP_ELEMS", 4))

    dram_impl = getattr(mem_model, "dram_impl", "DDR5-BITACC")
    dram_org = getattr(mem_model, "dram_org", "DDR5_8Gb_x8")
    dram_timing = getattr(mem_model, "dram_timing", "DDR5_3200AN")
    dram_channel = getattr(mem_model, "channel", 1)
    dram_rank = getattr(mem_model, "rank", 1)
    chips_per_dimm = int(getattr(mem_model, "chips_per_dimm", pim_config.get("DDR_CHIPS_PER_DIMM", 8)))
    channel_width_bits = getattr(mem_model, "channel_width", None)

    bank_shift = max(12, int(pim_config.get("BITACC_RAMULATOR_BANK_ADDR_SHIFT", 16)))
    n_lut_bcast = int(pim_config.get("BITACC_TRACE_N_LUT_BCAST", 32))
    n_rdpim = int(pim_config.get("BITACC_TRACE_N_RDPIM", 12))
    n_macab = max(1, int(pim_config.get("BITACC_ACC_UNITS", 1) or 1))
    burst_bytes = _runtime_burst_bytes(
        pim_config,
        dram_org=dram_org,
        chips_per_dimm=chips_per_dimm,
        channel_width=channel_width_bits,
    )
    row_bytes = _row_bytes_for_org(dram_org, chips_per_dimm)
    bank_layout = _resolve_bank_layout(dram_org, banks)
    memory_topology = _resolve_memory_topology(dram_channel, dram_rank)
    ramulator_bin = getattr(mem_model, "ramulator_bin", "ramulator2/ramulator2")
    timeout_s = int(timeout_s if timeout_s is not None else getattr(mem_model, "ramulator_timeout_s", 300))

    qkv_formula = _rtl_linear_formula(
        in_dim=d_model,
        out_dim=3 * d_model,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=0,
        local_cols_override=int(rtl["qkv_local_cols"]),
    )
    qkv_formula = _apply_linear_burst_scaling(qkv_formula, burst_bytes=burst_bytes)
    qkv_trace_input_dim = max(
        1,
        min(
            int(d_model),
            int(
                pim_config.get(
                    "BITACC_TRACE_QKV_INPUT_DIM",
                    pim_config.get(
                        "BITACC_AREA_REF_INPUT_DIM",
                        pim_config.get("bitacc_area_ref_input_dim", d_model),
                    ),
                )
            ),
        ),
    )
    qkv_trace_formula = _qkv_trace_formula(
        d_model=d_model,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        trace_input_dim=qkv_trace_input_dim,
    )
    qkv_trace_formula = _apply_linear_burst_scaling(qkv_trace_formula, burst_bytes=burst_bytes)
    wo_formula = _rtl_linear_formula(
        in_dim=d_model,
        out_dim=d_model,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=linear_tail_pe_cycles,
        local_cols_override=int(rtl["wo_local_cols"]),
    )
    wo_formula = _apply_linear_burst_scaling(wo_formula, burst_bytes=burst_bytes)
    up_formula = _rtl_linear_formula(
        in_dim=d_model,
        out_dim=mlp_dim,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=linear_tail_pe_cycles,
        local_cols_override=int(rtl["mlp_local_cols"]),
    )
    up_formula = _apply_linear_burst_scaling(up_formula, burst_bytes=burst_bytes)
    gate_formula = _rtl_linear_formula(
        in_dim=d_model,
        out_dim=mlp_dim,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=linear_tail_pe_cycles,
        local_cols_override=int(rtl["mlp_local_cols"]),
    )
    gate_formula = _apply_linear_burst_scaling(gate_formula, burst_bytes=burst_bytes)
    up_gate_formula = _rtl_linear_formula(
        in_dim=d_model,
        out_dim=2 * mlp_dim,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=linear_tail_pe_cycles,
        local_cols_override=2 * int(rtl["mlp_local_cols"]),
    )
    up_gate_formula = _apply_linear_burst_scaling(up_gate_formula, burst_bytes=burst_bytes)
    down_formula = _rtl_linear_formula(
        in_dim=mlp_dim,
        out_dim=d_model,
        banks=banks,
        outputs_per_launch=outputs_per_launch,
        tile_scalars=tile_scalars,
        launch_pe_cycles=launch_pe_cycles,
        tail_pe_cycles=0,
        local_cols_override=int(rtl["down_local_cols"]),
    )
    down_formula = _apply_linear_burst_scaling(down_formula, burst_bytes=burst_bytes)
    attention_parallel_heads = max(
        1,
        int(
            pim_config.get(
                "BITACC_TRACE_MAX_PARALLEL_HEADS",
                pim_config.get("BITACC_TRACE_ATTN_PARALLEL_HEADS", banks),
            )
        ),
    )
    serialize_attention_heads = int(num_heads) > int(attention_parallel_heads)
    attention_head_chunks: List[int] = []
    heads_remaining = int(num_heads)
    while heads_remaining > 0:
        chunk_heads = (
            min(int(attention_parallel_heads), heads_remaining)
            if serialize_attention_heads
            else heads_remaining
        )
        attention_head_chunks.append(int(chunk_heads))
        heads_remaining -= int(chunk_heads)
        if not serialize_attention_heads:
            break
    if not attention_head_chunks:
        attention_head_chunks = [int(num_heads)]

    qkv_trace_formulas: List[Dict] = []
    qkv_logic_formulas: List[Dict] = []
    qkv_burst_bases: List[int] = []
    qkv_burst_cursor = 0
    for chunk_heads in attention_head_chunks:
        if serialize_attention_heads:
            active_banks = max(1, int(chunk_heads))
            chunk_formula = _rtl_linear_formula(
                in_dim=d_model,
                out_dim=3 * int(chunk_heads) * int(head_dim),
                banks=active_banks,
                outputs_per_launch=outputs_per_launch,
                tile_scalars=tile_scalars,
                launch_pe_cycles=launch_pe_cycles,
                tail_pe_cycles=0,
                local_cols_override=3 * int(head_dim),
            )
            chunk_formula = _apply_linear_burst_scaling(chunk_formula, burst_bytes=burst_bytes)
            qkv_logic_formulas.append(chunk_formula)
            qkv_trace_formulas.append(chunk_formula)
        else:
            qkv_logic_formulas.append(qkv_formula)
            qkv_trace_formulas.append(qkv_trace_formula)
        qkv_burst_bases.append(int(qkv_burst_cursor))
        qkv_burst_cursor += int(
            qkv_trace_formulas[-1].get(
                "memory_total_bursts",
                int(qkv_trace_formulas[-1]["tiles"]) * int(qkv_trace_formulas[-1]["groups_per_tile"]),
            )
        )

    explicit_weight_layout = pim_config.get("BITACC_TRACE_WEIGHT_ROW_LAYOUT") or {}
    if "qkv" not in explicit_weight_layout:
        qkv_rows = _ceil_div(max(1, int(qkv_burst_cursor)), max(1, row_bytes // max(1, burst_bytes)))
        explicit_weight_layout = {
            **explicit_weight_layout,
            "qkv": {
                "row_base": 0,
                "rows": int(qkv_rows),
            },
        }
    weight_tensor_layouts = _weight_tensor_layouts(
        burst_bytes=burst_bytes,
        row_bytes=row_bytes,
        formulas=[
            ("qkv", qkv_formula),
            ("wo", wo_formula),
            ("wup_wgate", up_gate_formula),
            ("wdown", down_formula),
        ],
        explicit_layout=explicit_weight_layout,
    )

    score_steps = _ceil_div(visible_tokens, qkt_tok_par)
    linear_tail_breakdown_cfg = _dict_breakdown_items(
        pim_config.get("BITACC_TRACE_LINEAR_TAIL_BREAKDOWN")
    )
    residual_tail_breakdown_cfg = _dict_breakdown_items(
        pim_config.get("BITACC_TRACE_RESIDUAL_TAIL_BREAKDOWN")
    )

    def _trace_token(name: str,
                     count: int,
                     pe_cycles_each: int,
                     *,
                     domain: str = "dimm",
                     formula: str | None = None) -> Dict:
        count = max(1, int(count))
        pe_cycles_each = max(0, int(pe_cycles_each))
        if str(domain) == "bank":
            mem_cycles_each = _bank_pe_to_mem_cycles(pe_cycles_each, pim_config)
        else:
            mem_cycles_each = _dimm_pe_to_mem_cycles(pe_cycles_each, pim_config)
        return {
            "name": name,
            "count": count,
            "domain": str(domain),
            "pe_cycles_each": pe_cycles_each,
            "memory_cycles_each": mem_cycles_each,
            "pe_cycles_total": count * pe_cycles_each,
            "memory_cycles_total": count * mem_cycles_each,
            **({"formula": formula} if formula else {}),
        }

    def _trace_mem_token(name: str,
                         count: int,
                         memory_cycles_each: int,
                         *,
                         formula: str | None = None) -> Dict:
        count = max(1, int(count))
        memory_cycles_each = max(0, int(memory_cycles_each))
        return {
            "name": name,
            "count": count,
            "pe_cycles_each": 0,
            "memory_cycles_each": memory_cycles_each,
            "pe_cycles_total": 0,
            "memory_cycles_total": count * memory_cycles_each,
            **({"formula": formula} if formula else {}),
        }

    def _semantic_op(token: str,
                     count: int,
                     memory_cycles_each: int,
                     *,
                     addr_kind: str,
                     **kwargs) -> Dict:
        return {
            "kind": "semantic_repeated",
            "token": str(token),
            "count": int(count),
            "memory_cycles_each": int(memory_cycles_each),
            "addr_kind": str(addr_kind),
            **kwargs,
        }

    def _tail_breakdown_items(total_cycles: int) -> List[Tuple[str, int]]:
        total_cycles = max(0, int(total_cycles))
        if linear_tail_breakdown_cfg:
            used = _sum_breakdown_cycles(linear_tail_breakdown_cfg)
            items = list(linear_tail_breakdown_cfg)
            if total_cycles > used:
                items.append(("EPILOGUE", total_cycles - used))
            return items

        items = [
            ("INPUT_BUFFER", int(rtl_bank_ops.INPUT_BUFFER_STAGES)),
            ("TERNARY_PAIR_PATH", int(rtl_bank_ops.TERNARY_PAIR_PATH_CYCLES)),
            ("PSUM_CORE", int(rtl_bank_ops.PSUM_CORE_CYCLES)),
            ("DEQUANT_PIPE", int(rtl_bank_ops.DEQUANT_PIPE_CYCLES)),
        ]
        used = _sum_breakdown_cycles(items)
        if total_cycles > used:
            items.append(("EPILOGUE", total_cycles - used))
        return [(name, cyc) for name, cyc in items if cyc > 0]

    def _residual_breakdown_items(total_cycles: int) -> List[Tuple[str, int]]:
        total_cycles = max(0, int(total_cycles))
        if residual_tail_breakdown_cfg:
            used = _sum_breakdown_cycles(residual_tail_breakdown_cfg)
            items = list(residual_tail_breakdown_cfg)
            if total_cycles > used:
                items.append(("TOKEN_COMMIT", total_cycles - used))
            return items
        if total_cycles <= 0:
            return []
        if total_cycles == 1:
            return [("RESIDUAL_ALIGN", 1)]
        if total_cycles == 2:
            return [("RESIDUAL_ALIGN", 1), ("TOKEN_COMMIT", 1)]
        return [
            ("RESIDUAL_ALIGN", 1),
            ("RESIDUAL_PIPE", total_cycles - 2),
            ("TOKEN_COMMIT", 1),
        ]

    def _trace_mem_breakdown(
        *,
        token_prefix: str,
        stage_prefix: str,
        breakdown: List[Tuple[str, int]],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, int], List[str]]:
        trace_tokens: List[Dict] = []
        trace_ops: List[Dict] = []
        formula_terms: Dict[str, int] = {}
        formula_labels: List[str] = []
        for raw_name, cycles in breakdown:
            cycles = max(0, int(cycles))
            if cycles <= 0:
                continue
            name = str(raw_name).upper()
            term_key = name.lower()
            token_name = f"{token_prefix}_{name}"
            trace_tokens.append(
                _trace_mem_token(
                    token_name,
                    1,
                    cycles,
                    formula=f"{term_key} = {cycles}",
                )
            )
            trace_ops.append(
                _semantic_op(
                    token_name,
                    1,
                    cycles,
                    addr_kind="stage",
                    stage_tag=f"{stage_prefix}_{term_key}",
                )
            )
            formula_terms[term_key] = cycles
            formula_labels.append(term_key)
        return trace_tokens, trace_ops, formula_terms, formula_labels

    def _trace_pe_breakdown(
        *,
        token_prefix: str,
        stage_prefix: str,
        breakdown: List[Tuple[str, int]],
        domain: str,
    ) -> Tuple[List[Dict], List[Dict], Dict[str, int], List[str]]:
        trace_tokens: List[Dict] = []
        trace_ops: List[Dict] = []
        formula_terms: Dict[str, int] = {}
        formula_labels: List[str] = []
        for raw_name, cycles in breakdown:
            cycles = max(0, int(cycles))
            if cycles <= 0:
                continue
            name = str(raw_name).upper()
            term_key = name.lower()
            token_name = f"{token_prefix}_{name}"
            trace_tokens.append(
                _trace_token(
                    token_name,
                    1,
                    cycles,
                    domain=domain,
                    formula=f"{term_key} = {cycles}",
                )
            )
            if str(domain) == "bank":
                mem_cycles_each = _bank_pe_to_mem_cycles(cycles, pim_config)
            else:
                mem_cycles_each = _dimm_pe_to_mem_cycles(cycles, pim_config)
            trace_ops.append(
                _semantic_op(
                    token_name,
                    1,
                    mem_cycles_each,
                    addr_kind="stage",
                    stage_tag=f"{stage_prefix}_{term_key}",
                )
            )
            formula_terms[term_key] = cycles
            formula_labels.append(term_key)
        return trace_tokens, trace_ops, formula_terms, formula_labels

    def _bus_transfer_command(
        command_id: int,
        command: str,
        *,
        steps: int,
        step_cycles: int,
        ctrl_cycles: int,
        token_prefix: str,
        stage_prefix: str,
        rtl_phase_ids: List[int],
        rtl_phase_names: List[str],
        ready_event: str,
        value_term: str,
        formula_terms: Dict[str, int],
    ) -> Dict:
        trace_tokens, _trace_ops_unused, phase_terms, phase_labels = _trace_mem_breakdown(
            token_prefix=token_prefix,
            stage_prefix=stage_prefix,
            breakdown=[
                ("DATA", max(0, int(steps)) * max(0, int(step_cycles))),
                ("CTRL", max(0, int(ctrl_cycles))),
            ],
        )
        merged_terms = dict(formula_terms)
        merged_terms.update(phase_terms)
        return _command(
            command_id,
            command,
            pe_cycles=(max(0, int(steps)) * max(0, int(step_cycles))) + max(0, int(ctrl_cycles)),
            formula_text=" + ".join([value_term, *phase_labels]),
            formula_terms=merged_terms,
            rtl_phase_ids=rtl_phase_ids,
            rtl_phase_names=rtl_phase_names,
            trace_tokens=trace_tokens,
            trace_ops=[],
            key_events_pe=[(ready_event, (max(0, int(steps)) * max(0, int(step_cycles))) + max(0, int(ctrl_cycles)))],
            cycle_eval_mode="memory_detailed",
        )

    def _vector_move_command(
        command_id: int,
        command: str,
        *,
        vec_elems: int,
        elems_per_cycle: int,
        ctrl_cycles: int,
        token_prefix: str,
        stage_prefix: str,
        rtl_phase_ids: List[int],
        rtl_phase_names: List[str],
        ready_event: str,
        dim_term: str,
    ) -> Dict:
        move_steps = _ceil_div(vec_elems, elems_per_cycle)
        trace_tokens, _trace_ops_unused, phase_terms, phase_labels = _trace_mem_breakdown(
            token_prefix=token_prefix,
            stage_prefix=stage_prefix,
            breakdown=[
                ("MOVE", move_steps),
                ("CTRL", max(0, int(ctrl_cycles))),
            ],
        )
        formula_terms = {
            dim_term: int(vec_elems),
            "vector_lanes": int(elems_per_cycle),
            **phase_terms,
        }
        return _command(
            command_id,
            command,
            pe_cycles=move_steps + max(0, int(ctrl_cycles)),
            formula_text=" + ".join(["vector_move_steps", *phase_labels]),
            formula_terms=formula_terms,
            rtl_phase_ids=rtl_phase_ids,
            rtl_phase_names=rtl_phase_names,
            trace_tokens=trace_tokens,
            trace_ops=[],
            key_events_pe=[(ready_event, move_steps + max(0, int(ctrl_cycles)))],
            cycle_eval_mode="memory_detailed",
        )

    def _linear_trace_tokens(
        *,
        formula: Dict,
        trace_setup_token: str,
        trace_body_token: str,
    ) -> List[Dict]:
        tokens: List[Dict] = []
        tiles = max(1, int(formula["tiles"]))
        groups_per_tile = max(
            1,
            int(formula.get("memory_groups_per_tile", formula["groups_per_tile"])),
        )
        for tile_idx in range(tiles):
            tokens.append(
                _trace_token(
                    trace_setup_token,
                    1,
                    int(formula["lut_load_pairs"]),
                    domain="bank",
                    formula=f"tile={tile_idx} load k_tile_pairs={formula['lut_load_pairs']}",
                )
            )
            tokens.append(
                _trace_token(
                    trace_body_token,
                    groups_per_tile,
                    launch_pe_cycles,
                    domain="bank",
                    formula=f"tile={tile_idx} groups={groups_per_tile} x {launch_pe_cycles} PE cycles",
                )
            )
        return tokens

    def _generic_trace_ops(trace_tokens: List[Dict]) -> List[Dict]:
        return [
            _semantic_op(
                str(s["name"]),
                int(s["count"]),
                int(s["memory_cycles_each"]),
                addr_kind="stage",
                stage_tag=str(s["name"]).lower(),
            )
            for s in trace_tokens
        ]

    def _linear_trace_ops(
        *,
        tensor_tag: str,
        formula: Dict,
        tensor_row_base: int,
        trace_setup_token: str,
        trace_body_token: str,
        burst_ordinal_base: int = 0,
    ) -> List[Dict]:
        lut_mem_cycles = _bank_pe_to_mem_cycles(int(formula["lut_load_pairs"]), pim_config)
        body_mem_cycles = _bank_pe_to_mem_cycles(launch_pe_cycles, pim_config)
        ops: List[Dict] = []
        tiles = max(1, int(formula["tiles"]))
        groups_per_tile = max(
            1,
            int(formula.get("memory_groups_per_tile", formula["groups_per_tile"])),
        )
        for tile_idx in range(tiles):
            ops.append(
                _semantic_op(
                    trace_setup_token,
                    1,
                    lut_mem_cycles,
                    addr_kind="weight_lut",
                    tensor_tag=str(tensor_tag),
                    tile_idx=tile_idx,
                )
            )
            for group_idx in range(groups_per_tile):
                burst_ordinal = int(burst_ordinal_base) + (tile_idx * groups_per_tile) + group_idx
                ops.append(
                    _semantic_op(
                        trace_body_token,
                        1,
                        body_mem_cycles,
                        addr_kind="weight_rect",
                        tensor_tag=str(tensor_tag),
                        tensor_row_base=int(tensor_row_base),
                        burst_ordinal=int(burst_ordinal),
                    )
                )
        return ops

    def _command(command_id: int,
                 command: str,
                 *,
                 pe_cycles: int,
                 formula_text: str,
                 formula_terms: Dict[str, int] | None = None,
                 rtl_phase_ids: List[int] | None = None,
                 rtl_phase_names: List[str] | None = None,
                 trace_tokens: List[Dict] | None = None,
                 trace_ops: List[Dict] | None = None,
                 key_events_pe: List[Tuple[str, int]] | None = None,
                 cycle_eval_mode: str = "pe") -> Dict:
        trace_tokens = trace_tokens or []
        trace_ops = _generic_trace_ops(trace_tokens) if trace_ops is None else trace_ops
        formula_terms = formula_terms or {}
        rtl_phase_ids = rtl_phase_ids or []
        rtl_phase_names = rtl_phase_names or []
        key_events_pe = key_events_pe or []
        mem_total = int(sum(int(s["memory_cycles_total"]) for s in trace_tokens))
        return {
            "command_id": int(command_id),
            "command": command,
            "formula": formula_text,
            "formula_terms": {str(k): int(v) for k, v in formula_terms.items()},
            "rtl_phase_ids": [int(v) for v in rtl_phase_ids],
            "rtl_phases": [str(v) for v in rtl_phase_names],
            "trace_mode": "explicit_subcommands",
            "pe_cycles": int(pe_cycles),
            "memory_cycles": mem_total,
            "trace_tokens": trace_tokens,
            "trace_ops": trace_ops,
            "key_events": key_events_pe,
            "cycle_eval_mode": str(cycle_eval_mode),
        }

    def _linear_command(command_id: int,
                        command: str,
                        formula: Dict,
                        *,
                        tensor_tag: str,
                        rtl_phase_ids: List[int],
                        rtl_phase_names: List[str],
                        trace_setup_token: str,
        trace_body_token: str,
        cfg_pe: int = 0,
        tail_breakdown: List[Tuple[str, int]] | None = None,
        tail_token_prefix: str | None = None,
        tail_stage_prefix: str | None = None,
        key_events_pe: List[Tuple[str, int]] | None = None,
        cycle_eval_mode: str = "pe") -> Dict:
        trace_tokens = _linear_trace_tokens(
            formula=formula,
            trace_setup_token=trace_setup_token,
            trace_body_token=trace_body_token,
        )
        trace_ops = _linear_trace_ops(
            tensor_tag=tensor_tag,
            formula=formula,
            tensor_row_base=int(weight_tensor_layouts[str(tensor_tag)]["row_base"]),
            trace_setup_token=trace_setup_token,
            trace_body_token=trace_body_token,
            burst_ordinal_base=0,
        )
        formula_terms = {
            "cfg": int(cfg_pe),
            "load": int(formula["load_pe_cycles"]),
            "launch": int(formula["launch_pe_cycles"]),
        }
        formula_labels: List[str] = []
        if cfg_pe > 0:
            formula_labels.append("cfg")
        formula_labels.extend(["linear_load", "linear_launch"])
        if int(formula["tail_pe_cycles"]) > 0:
            if tail_breakdown:
                tail_tokens, _tail_ops_unused, tail_terms, tail_labels = _trace_pe_breakdown(
                    token_prefix=str(tail_token_prefix or str(command).split("[", 1)[0]),
                    stage_prefix=str(tail_stage_prefix or str(tensor_tag)),
                    breakdown=tail_breakdown,
                    domain="bank",
                )
                trace_tokens.extend(tail_tokens)
                formula_terms.update(tail_terms)
                formula_labels.extend(tail_labels)
            else:
                formula_terms["tail"] = int(formula["tail_pe_cycles"])
                formula_labels.append("linear_tail")
        return _command(
            command_id,
            command,
            pe_cycles=int(cfg_pe) + int(formula["total_pe_cycles"]),
            formula_text=" + ".join(formula_labels),
            formula_terms=formula_terms,
            rtl_phase_ids=rtl_phase_ids,
            rtl_phase_names=rtl_phase_names,
            trace_tokens=trace_tokens,
            trace_ops=trace_ops,
            key_events_pe=key_events_pe or [],
            cycle_eval_mode=cycle_eval_mode,
        )

    def _abstract_command(command_id: int,
                          command: str,
                          *,
                          pe_cycles: int,
                          formula_text: str,
                          formula_terms: Dict[str, int] | None = None,
                          rtl_phase_ids: List[int] | None = None,
                          rtl_phase_names: List[str] | None = None,
                          key_events_pe: List[Tuple[str, int]] | None = None) -> Dict:
        return {
            "command_id": int(command_id),
            "command": str(command),
            "formula": str(formula_text),
            "formula_terms": {str(k): int(v) for k, v in (formula_terms or {}).items()},
            "rtl_phase_ids": [int(v) for v in (rtl_phase_ids or [])],
            "rtl_phases": [str(v) for v in (rtl_phase_names or [])],
            "trace_mode": "report_only",
            "pe_cycles": int(max(0, pe_cycles)),
            "memory_cycles": 0,
            "trace_tokens": [],
            "trace_ops": [],
            "key_events": key_events_pe or [],
            "cycle_eval_mode": "pe",
        }

    dequant_pe_cycles = int(rtl_bank_ops.DEQUANT_PIPE_CYCLES)
    current_kv_write_mem_cycles_each = _ddr_burst_mem_cycles()
    pre_attn_rms_pe_cycles = int(rtl_dimm_ops.rmsnorm_cycles(d_model))
    pre_mlp_rms_pe_cycles = int(rtl_dimm_ops.rmsnorm_cycles(d_model))
    history_k_read_mem_cycles_each = _ddr_burst_mem_cycles()
    qkt_compute_pe_cycles = int(
        rtl_bank_ops.qkt_compute_cycles(
            visible_tokens,
            qkt_tok_par,
            head_dim=head_dim,
            dot_par=qkt_dot_par,
        )
    )
    qkt_phase_pe_cycles = int(qkt_compute_pe_cycles)
    history_v_read_mem_cycles_each = _ddr_burst_mem_cycles()
    pv_compute_pe_cycles = int(
        rtl_bank_ops.pv_compute_cycles(
            visible_tokens,
            tok_par,
            head_dim=head_dim,
            tp_par=tp_par,
            pv_dim_par=pv_dim_par,
        )
    )
    pv_phase_pe_cycles = int(pv_compute_pe_cycles)
    vector_move_ctrl_cycles = 1
    wo_concat_vector_pe_cycles = _abstract_vector_move_cycles(
        d_model,
        elems_per_cycle=trace_vector_elems_per_cycle,
        ctrl_cycles=vector_move_ctrl_cycles,
    )
    up_gather_pe_cycles = _abstract_vector_move_cycles(
        mlp_dim,
        elems_per_cycle=trace_vector_elems_per_cycle,
        ctrl_cycles=vector_move_ctrl_cycles,
    )
    gate_gather_pe_cycles = _abstract_vector_move_cycles(
        mlp_dim,
        elems_per_cycle=trace_vector_elems_per_cycle,
        ctrl_cycles=vector_move_ctrl_cycles,
    )
    down_gather_pe_cycles = _abstract_vector_move_cycles(
        d_model,
        elems_per_cycle=trace_vector_elems_per_cycle,
        ctrl_cycles=vector_move_ctrl_cycles,
    )
    linear_tail_breakdown = _tail_breakdown_items(linear_tail_pe_cycles)
    residual_tail_breakdown = _residual_breakdown_items(mlp_residual_pe_cycles)
    (
        mlp_residual_trace_tokens,
        _mlp_residual_trace_ops_unused,
        mlp_residual_formula_terms,
        mlp_residual_formula_labels,
    ) = _trace_mem_breakdown(
        token_prefix="MLP_RESIDUAL",
        stage_prefix="mlp_residual",
        breakdown=residual_tail_breakdown,
    )
    kv_row_bases = pim_config.get("BITACC_TRACE_KV_ROW_BASES") or {}
    weight_rows_end = max(
        int(layout["row_base"]) + int(layout["row_span"])
        for layout in weight_tensor_layouts.values()
    )
    k_row_base = int(kv_row_bases.get("k", weight_rows_end))
    v_row_base = int(kv_row_bases.get("v", 256))
    bursts_per_row = max(1, row_bytes // max(1, burst_bytes))
    history_chunk_row_span = _ceil_div(max(1, visible_history_tokens), bursts_per_row)
    linear_weight_body_token = "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"
    current_token_slot = max(0, visible_tokens - 1)

    attention_chunk_specs: List[Dict] = []
    total_attention_history_row_span = max(1, len(attention_head_chunks)) * history_chunk_row_span
    current_k_row_base = int(k_row_base + total_attention_history_row_span) if serialize_attention_heads else int(k_row_base)
    current_v_row_base = int(v_row_base + total_attention_history_row_span) if serialize_attention_heads else int(v_row_base)
    for pass_idx, chunk_heads in enumerate(attention_head_chunks):
        chunk_heads = int(chunk_heads)
        qkv_logic_formula_i = dict(qkv_logic_formulas[pass_idx])
        qkv_trace_formula_i = dict(qkv_trace_formulas[pass_idx])
        chunk_kv_local_cols = int(head_dim if serialize_attention_heads else bank_local_cols)
        chunk_rope_pe_cycles = int(rtl_bank_ops.rope_cycles(chunk_kv_local_cols))
        chunk_current_kv_write_bursts = _all_bank_fp16_write_bursts(chunk_kv_local_cols, burst_bytes)
        qkv_linear_pe_cycles_i = int(qkv_logic_formula_i["load_pe_cycles"] + qkv_logic_formula_i["launch_pe_cycles"])
        qkv_total_pe_cycles_i = int(qkv_linear_pe_cycles_i + dequant_pe_cycles + chunk_rope_pe_cycles)
        q_rot_ready_rel_i = int(qkv_linear_pe_cycles_i + dequant_pe_cycles + chunk_rope_pe_cycles)
        attention_chunk_specs.append({
            "pass_idx": int(pass_idx),
            "chunk_heads": int(chunk_heads),
            "qkv_logic_formula": qkv_logic_formula_i,
            "qkv_trace_formula": qkv_trace_formula_i,
            "qkv_burst_ordinal_base": int(qkv_burst_bases[pass_idx]),
            "kv_local_cols": int(chunk_kv_local_cols),
            "rope_pe_cycles": int(chunk_rope_pe_cycles),
            "qkv_linear_pe_cycles": int(qkv_linear_pe_cycles_i),
            "qkv_total_pe_cycles": int(qkv_total_pe_cycles_i),
            "q_rot_ready_rel": int(q_rot_ready_rel_i),
            "current_kv_write_bursts": int(chunk_current_kv_write_bursts),
            "history_k_read_reqs": int(max(0, visible_history_tokens)),
            "history_v_read_reqs": int(max(0, visible_history_tokens)),
            "history_k_row_base": int(k_row_base + (pass_idx * history_chunk_row_span)) if serialize_attention_heads else int(k_row_base),
            "history_v_row_base": int(v_row_base + (pass_idx * history_chunk_row_span)) if serialize_attention_heads else int(v_row_base),
            "current_k_row_base": int(current_k_row_base),
            "current_v_row_base": int(current_v_row_base),
            "current_k_token_slot": int((pass_idx * chunk_current_kv_write_bursts) if serialize_attention_heads else current_token_slot),
            "current_v_token_slot": int((pass_idx * chunk_current_kv_write_bursts) if serialize_attention_heads else current_token_slot),
        })

    fp16_bits = 16
    score_bus_heads_per_step = max(
        1,
        int(trace_bus_io_bits) // max(1, int(qkt_tok_par) * int(trace_score_bits)),
    )
    prob_bus_heads_per_step = max(
        1,
        int(trace_bus_io_bits) // max(1, int(pv_dim_par) * int(trace_prob_bits)),
    )
    vector_bus_heads_per_step = max(
        1,
        int(trace_bus_io_bits) // max(1, int(trace_vector_group_elems) * int(fp16_bits)),
    )
    attn_gather_cycles = int(
        1
        + _ceil_div(head_dim, trace_vector_group_elems)
        * _ceil_div(num_heads, vector_bus_heads_per_step)
    )
    mlp_gather_cycles = int(
        1
        + _ceil_div(_ceil_div(mlp_dim, num_heads), trace_vector_group_elems)
        * _ceil_div(num_heads, vector_bus_heads_per_step)
    )

    active_formula_banks = max(1, min(int(num_heads), int(banks)))
    phase_cycle_rows = transformer_13_phase_cycle_model(
        CycleModelParams(
            d_model=int(d_model),
            mlp_dim=int(mlp_dim),
            banks=int(active_formula_banks),
            visible_tokens=int(visible_tokens),
            k_tile=int(k_tile),
            n_tile=int(outputs_per_launch),
            launch_cycles=int(launch_pe_cycles),
            tq_par=int(qkt_tok_par),
            tp_par=int(tp_par),
            bank_fp16_lanes=int(qkt_dot_par),
            bank_io_bits=int(trace_bus_io_bits),
            acc_w_bits=int(trace_score_bits),
            fp16_w_bits=int(trace_prob_bits),
            bank_deq_tail=int(qkv_ready_tail_pe_cycles),
            bank_wr2rd_turn=int(pim_config.get("BITACC_TRACE_BANK_WR2RD_TURN_CYCLES", 1)),
            lut_pair_gen_per_cycle=int(pim_config.get("BITACC_TRACE_LUT_PAIR_GEN_PER_CYCLE", 1)),
        )
    )
    phase_report_13: List[Dict] = []
    stage_start_cycle = 1
    for row in phase_cycle_rows:
        entry = _cycles_to_phase_entry(
            f"phase_{int(row['index'])}_{row['name']}",
            stage_start_cycle,
            int(row["cycles"]),
            [],
        )
        entry["domain"] = "13_phase_cycle_model"
        entry["formula"] = str(row["formula"])
        entry["formula_terms"] = {str(k): int(v) for k, v in row["formula_terms"].items()}
        phase_report_13.append(entry)
        stage_start_cycle = int(entry["end_cycle"]) + 1

    def _attention_label(prefix: str, spec: Dict, suffix: str) -> str:
        if not serialize_attention_heads:
            return f"{prefix}{suffix}"
        start_head = int(spec["pass_idx"]) * int(attention_parallel_heads)
        end_head = start_head + int(spec["chunk_heads"]) - 1
        return f"{prefix}[heads={start_head}-{end_head};pass={int(spec['pass_idx']) + 1}/{len(attention_chunk_specs)}]{suffix}"

    def _attention_qkv_command(command_id: int, spec: Dict) -> Dict:
        qkv_logic_formula_i = dict(spec["qkv_logic_formula"])
        qkv_trace_formula_i = dict(spec["qkv_trace_formula"])
        qkv_linear_pe_cycles_i = int(spec["qkv_linear_pe_cycles"])
        qkv_total_pe_cycles_i = int(spec["qkv_total_pe_cycles"])
        chunk_rope_pe_cycles = int(spec["rope_pe_cycles"])
        chunk_current_kv_write_bursts = int(spec["current_kv_write_bursts"])
        chunk_kv_local_cols = int(spec["kv_local_cols"])
        return _command(
            command_id,
            _attention_label("QKV", spec, f"[GEMV+DEQUANT({dequant_pe_cycles})+ROPE({chunk_rope_pe_cycles})+KV_WRITE]"),
            pe_cycles=int(qkv_total_pe_cycles_i),
            formula_text="qkv_linear + dequant + rope",
            formula_terms={
                "active_heads": int(spec["chunk_heads"]),
                "qkv_load": int(qkv_logic_formula_i["load_pe_cycles"]),
                "qkv_launch": int(qkv_logic_formula_i["launch_pe_cycles"]),
                "dequant": int(dequant_pe_cycles),
                "rope": int(chunk_rope_pe_cycles),
            },
            rtl_phase_ids=[1],
            rtl_phase_names=["ST_WAIT_QKV"],
            trace_tokens=[
                *_linear_trace_tokens(
                    formula=qkv_trace_formula_i,
                    trace_setup_token="ALL_BANK_LUT_GEN_AND_LOAD",
                    trace_body_token=linear_weight_body_token,
                ),
                _trace_token("ALL_BANK_DEQUANT", 1, dequant_pe_cycles, domain="bank", formula=f"dequant_pipe = {dequant_pe_cycles}"),
                _trace_token("ALL_BANK_ROPE", 1, chunk_rope_pe_cycles, domain="bank", formula=f"ceil(chunk_kv_local_cols/2)+ctrl = ceil({chunk_kv_local_cols}/2)+{rope_ctrl_pe_cycles}"),
                _trace_mem_token(
                    "ALL_BANK_WRITE_K_CURRENT",
                    chunk_current_kv_write_bursts,
                    current_kv_write_mem_cycles_each,
                    formula=f"ceil(({chunk_kv_local_cols} fp16 x 2B)/{burst_bytes}B) requests x {_ddr_burst_mem_cycles()} controller cycles/request",
                ),
                _trace_mem_token(
                    "ALL_BANK_WRITE_V_CURRENT",
                    chunk_current_kv_write_bursts,
                    current_kv_write_mem_cycles_each,
                    formula=f"ceil(({chunk_kv_local_cols} fp16 x 2B)/{burst_bytes}B) requests x {_ddr_burst_mem_cycles()} controller cycles/request",
                ),
            ],
            trace_ops=[
                *_linear_trace_ops(
                    tensor_tag="qkv",
                    formula=qkv_trace_formula_i,
                    tensor_row_base=int(weight_tensor_layouts["qkv"]["row_base"]),
                    trace_setup_token="ALL_BANK_LUT_GEN_AND_LOAD",
                    trace_body_token=linear_weight_body_token,
                    burst_ordinal_base=int(spec["qkv_burst_ordinal_base"]),
                ),
                _semantic_op(
                    "ALL_BANK_DEQUANT",
                    1,
                    _bank_pe_to_mem_cycles(dequant_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="dequant",
                ),
                _semantic_op(
                    "ALL_BANK_ROPE",
                    1,
                    _bank_pe_to_mem_cycles(chunk_rope_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="rope",
                ),
                _semantic_op(
                    "ALL_BANK_WRITE_K_CURRENT",
                    chunk_current_kv_write_bursts,
                    current_kv_write_mem_cycles_each,
                    addr_kind="kv",
                    kv_tag="k_current",
                    token_slot=int(spec["current_k_token_slot"]),
                    advance_with_ordinal=True,
                    bank_id=0,
                    row_base=int(spec["current_k_row_base"]),
                ),
                _semantic_op(
                    "ALL_BANK_WRITE_V_CURRENT",
                    chunk_current_kv_write_bursts,
                    current_kv_write_mem_cycles_each,
                    addr_kind="kv",
                    kv_tag="v_current",
                    token_slot=int(spec["current_v_token_slot"]),
                    advance_with_ordinal=True,
                    bank_id=0,
                    row_base=int(spec["current_v_row_base"]),
                ),
            ],
            key_events_pe=[
                ("QKV_LINEAR_DONE", qkv_linear_pe_cycles_i),
                ("DEQUANT_DONE", int(qkv_linear_pe_cycles_i + dequant_pe_cycles)),
                ("ROPE_DONE", int(spec["q_rot_ready_rel"])),
                ("K_CACHE_WRITE_DONE", int(spec["q_rot_ready_rel"])),
                ("V_CACHE_WRITE_DONE", int(spec["q_rot_ready_rel"])),
            ],
        )

    def _attention_qkt_command(command_id: int, spec: Dict) -> Dict:
        history_k_read_reqs_i = int(spec["history_k_read_reqs"])
        return _command(
            command_id,
            _attention_label("QKT", spec, f"[READ_K_CACHE({history_k_read_reqs_i}x{history_k_read_mem_cycles_each})+SCORE({qkt_compute_pe_cycles})]"),
            pe_cycles=int(qkt_phase_pe_cycles),
            formula_text="score_loop + qkt_ctrl",
            formula_terms={
                "active_heads": int(spec["chunk_heads"]),
                "score_seq_steps": int(_ceil_div(visible_tokens, qkt_tok_par)),
                "score_dot_steps": int(_ceil_div(head_dim, qkt_dot_par)),
                "qkt_ctrl": int(qkt_ctrl_pe_cycles),
            },
            rtl_phase_ids=[2],
            rtl_phase_names=["ST_WAIT_QKT"],
            trace_tokens=[
                _trace_mem_token(
                    "ALL_BANK_READ_K_CACHE",
                    history_k_read_reqs_i,
                    history_k_read_mem_cycles_each,
                    formula=f"{visible_history_tokens} all-bank burst requests x {_ddr_burst_mem_cycles()} controller cycles/request",
                ),
                _trace_token("ALL_BANK_QKT", 1, qkt_compute_pe_cycles, domain="bank", formula=f"ceil(seq_len/tq_par)*ceil(head_dim/qkt_dot_par)+ctrl = ceil({visible_tokens}/{qkt_tok_par})*ceil({head_dim}/{qkt_dot_par})+{qkt_ctrl_pe_cycles}"),
            ],
            trace_ops=[
                _semantic_op(
                    "ALL_BANK_READ_K_CACHE",
                    history_k_read_reqs_i,
                    history_k_read_mem_cycles_each,
                    addr_kind="kv",
                    kv_tag="k_history",
                    token_slot=0,
                    advance_with_ordinal=True,
                    bank_id=0,
                    row_base=int(spec["history_k_row_base"]),
                ),
                _semantic_op(
                    "ALL_BANK_QKT",
                    1,
                    _bank_pe_to_mem_cycles(qkt_compute_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="qkt",
                ),
            ],
            key_events_pe=[("QKT_VALID", qkt_phase_pe_cycles)],
            cycle_eval_mode="pe",
        )

    def _attention_pv_command(command_id: int, spec: Dict) -> Dict:
        history_v_read_reqs_i = int(spec["history_v_read_reqs"])
        return _command(
            command_id,
            _attention_label("PV", spec, f"[READ_V_CACHE({history_v_read_reqs_i}x{history_v_read_mem_cycles_each})+CONTEXT({pv_compute_pe_cycles})]"),
            pe_cycles=int(pv_phase_pe_cycles),
            formula_text="context_loop + pv_ctrl",
            formula_terms={
                "active_heads": int(spec["chunk_heads"]),
                "pv_head_blocks": int(_ceil_div(head_dim, tp_par)),
                "pv_seq_sweeps": int(_ceil_div(visible_tokens, pv_dim_par)),
                "pv_ctrl": int(pv_ctrl_pe_cycles),
            },
            rtl_phase_ids=[4],
            rtl_phase_names=["ST_WAIT_PV"],
            trace_tokens=[
                _trace_mem_token(
                    "ALL_BANK_READ_V_CACHE",
                    history_v_read_reqs_i,
                    history_v_read_mem_cycles_each,
                    formula=f"{visible_history_tokens} all-bank burst requests x {_ddr_burst_mem_cycles()} controller cycles/request",
                ),
                _trace_token("ALL_BANK_PV", 1, pv_compute_pe_cycles, domain="bank", formula=f"ceil(head_dim/tp_par)*ceil(seq_len/pv_dim_par)+ctrl = ceil({head_dim}/{tp_par})*ceil({visible_tokens}/{pv_dim_par})+{pv_ctrl_pe_cycles}"),
            ],
            trace_ops=[
                _semantic_op(
                    "ALL_BANK_READ_V_CACHE",
                    history_v_read_reqs_i,
                    history_v_read_mem_cycles_each,
                    addr_kind="kv",
                    kv_tag="v_history",
                    token_slot=0,
                    advance_with_ordinal=True,
                    bank_id=0,
                    row_base=int(spec["history_v_row_base"]),
                ),
                _semantic_op(
                    "ALL_BANK_PV",
                    1,
                    _bank_pe_to_mem_cycles(pv_compute_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="pv",
                ),
            ],
            key_events_pe=[("PV_VALID", pv_phase_pe_cycles)],
            cycle_eval_mode="pe",
        )
    command_schedule: List[Dict] = []
    next_command_id = 0

    command_schedule.append(
        _command(
            next_command_id,
            "STARTUP",
            pe_cycles=int(startup_pe_cycles),
            formula_text="startup_fixed",
            formula_terms={"startup": int(startup_pe_cycles)},
            rtl_phase_ids=[0],
            rtl_phase_names=["STARTUP"],
        )
    )
    next_command_id += 1

    command_schedule.append(
        _command(
            next_command_id,
            f"PRE_ATTN_RMS[2d+norm_ctrl={pre_attn_rms_pe_cycles}]",
            pe_cycles=int(pre_attn_rms_pe_cycles),
            formula_text="rms_sumsq + rms_scale_out + norm_ctrl",
            formula_terms={
                "rms_sumsq": int(d_model),
                "rms_scale_out": int(d_model),
                "norm_ctrl": int(norm_ctrl_pe_cycles),
            },
            rtl_phase_ids=[13],
            rtl_phase_names=["PRE_ATTN_NORM"],
            trace_tokens=[
                _trace_token("PRE_ATTN_NORM", 2, d_model, formula=f"2 x d = 2 x {d_model}"),
                _trace_token("PRE_ATTN_NORM_TAIL", 1, norm_ctrl_pe_cycles, formula=f"norm_ctrl = {norm_ctrl_pe_cycles}"),
            ],
            trace_ops=[
                _semantic_op(
                    "PRE_ATTN_NORM",
                    2,
                    _dimm_pe_to_mem_cycles(d_model, pim_config),
                    addr_kind="stage",
                    stage_tag="pre_attn_norm",
                ),
                _semantic_op(
                    "PRE_ATTN_NORM_TAIL",
                    1,
                    _dimm_pe_to_mem_cycles(norm_ctrl_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="pre_attn_norm_tail",
                ),
            ],
            key_events_pe=[("RMS_DONE", pre_attn_rms_pe_cycles)],
            cycle_eval_mode="memory_detailed",
        )
    )
    next_command_id += 1

    for spec in attention_chunk_specs:
        chunk_heads = int(spec["chunk_heads"])
        chunk_post_cycles = int(rtl_dimm_ops.qkt_postproc_cycles(visible_tokens, qkt_tok_par))
        chunk_score_transfer_step_cycles = _ceil_div(
            int(chunk_heads) * int(qkt_tok_par) * int(trace_score_bits),
            trace_bus_io_bits,
        )
        chunk_prob_transfer_step_cycles = _ceil_div(
            int(chunk_heads) * int(pv_dim_par) * int(trace_prob_bits),
            trace_bus_io_bits,
        )
        command_schedule.append(_attention_qkv_command(next_command_id, spec))
        next_command_id += 1
        command_schedule.append(_attention_qkt_command(next_command_id, spec))
        next_command_id += 1
        command_schedule.append(
            _bus_transfer_command(
                next_command_id,
                _attention_label("SCORE_TRANSFER", spec, "[BANK_TO_DIMM_BUS]"),
                steps=int(score_steps),
                step_cycles=int(chunk_score_transfer_step_cycles),
                ctrl_cycles=1,
                token_prefix="SCORE_TRANSFER_BANK_TO_DIMM",
                stage_prefix="score_transfer_bank_to_dimm",
                value_term="score_steps_x_bus_words",
                formula_terms={
                    "score_steps": int(score_steps),
                    "score_bus_heads_per_step": int(score_bus_heads_per_step),
                    "chunk_heads": int(chunk_heads),
                    "tq_par": int(qkt_tok_par),
                    "score_bits": int(trace_score_bits),
                    "io_bits": int(trace_bus_io_bits),
                },
                rtl_phase_ids=[5],
                rtl_phase_names=["SCORE_TRANSFER"],
                ready_event="SCORE_DIMM_READY",
            )
        )
        next_command_id += 1
        command_schedule.append(
            _command(
                next_command_id,
                _attention_label("POST", spec, f"[SOFTMAX_POST={chunk_post_cycles}]"),
                pe_cycles=chunk_post_cycles,
                formula_text="4*post_steps + post_ctrl",
                formula_terms={
                    "post_steps": int(score_steps),
                    "post_ctrl": int(post_ctrl_pe_cycles),
                },
                rtl_phase_ids=[3],
                rtl_phase_names=["ST_WAIT_POST"],
                trace_tokens=[
                    _trace_token("ALL_BANK_POST", 1, chunk_post_cycles, formula=f"4*ceil({visible_tokens}/{qkt_tok_par})+{post_ctrl_pe_cycles}"),
                ],
                trace_ops=[
                    _semantic_op(
                        "ALL_BANK_POST",
                        1,
                        _dimm_pe_to_mem_cycles(chunk_post_cycles, pim_config),
                        addr_kind="stage",
                        stage_tag="post",
                    ),
                ],
                key_events_pe=[("POST_VALID", chunk_post_cycles)],
                cycle_eval_mode="memory_detailed",
            )
        )
        next_command_id += 1
        command_schedule.append(
            _bus_transfer_command(
                next_command_id,
                _attention_label("PROBABILITY_TRANSFER", spec, "[DIMM_TO_BANK_BUS]"),
                steps=int(_ceil_div(head_dim, tp_par)),
                step_cycles=int(chunk_prob_transfer_step_cycles),
                ctrl_cycles=1,
                token_prefix="PROBABILITY_TRANSFER_DIMM_TO_BANK",
                stage_prefix="probability_transfer_dimm_to_bank",
                value_term="pv_head_blocks_x_bus_words",
                formula_terms={
                    "pv_head_blocks": int(_ceil_div(head_dim, tp_par)),
                    "prob_bus_heads_per_step": int(prob_bus_heads_per_step),
                    "chunk_heads": int(chunk_heads),
                    "pv_dim_par": int(pv_dim_par),
                    "prob_bits": int(trace_prob_bits),
                    "io_bits": int(trace_bus_io_bits),
                },
                rtl_phase_ids=[6],
                rtl_phase_names=["PROB_TRANSFER"],
                ready_event="PROB_BANK_READY",
            )
        )
        next_command_id += 1
        command_schedule.append(_attention_pv_command(next_command_id, spec))
        next_command_id += 1

    command_schedule.append(
        _vector_move_command(
            next_command_id,
            "CONCAT_ATTENTION_H[MOVE]",
            vec_elems=int(d_model),
            elems_per_cycle=int(trace_vector_elems_per_cycle),
            ctrl_cycles=int(vector_move_ctrl_cycles),
            token_prefix="CONCAT_ATTENTION_H",
            stage_prefix="concat_attention_h",
            rtl_phase_ids=[7],
            rtl_phase_names=["CONCAT_H"],
            ready_event="CONCAT_H_READY",
            dim_term="d_model",
        )
    )
    next_command_id += 1

    command_schedule.append(
        _linear_command(
            next_command_id,
            "WO[GEMV]",
            wo_formula,
            tensor_tag="wo",
            rtl_phase_ids=[14, 15],
            rtl_phase_names=["CFG_WO", "WAIT_WO"],
            trace_setup_token="ALL_BANK_LUT_GEN_AND_LOAD",
            trace_body_token=linear_weight_body_token,
            cfg_pe=int(cfg_pe_cycles),
            tail_breakdown=linear_tail_breakdown,
            tail_token_prefix="WO_TAIL",
            tail_stage_prefix="wo_tail",
            key_events_pe=[("WO_GEMV_READY", int(cfg_pe_cycles) + int(wo_formula["total_pe_cycles"]))],
        )
    )
    next_command_id += 1

    command_schedule.append(
        _command(
            next_command_id,
            f"PRE_MLP_RMS[2d+norm_ctrl={pre_mlp_rms_pe_cycles}]",
            pe_cycles=int(pre_mlp_rms_pe_cycles),
            formula_text="rms_sumsq + rms_scale_out + norm_ctrl",
            formula_terms={
                "rms_sumsq": int(d_model),
                "rms_scale_out": int(d_model),
                "norm_ctrl": int(norm_ctrl_pe_cycles),
            },
            rtl_phase_ids=[5],
            rtl_phase_names=["PRE_MLP_NORM"],
            trace_tokens=[
                _trace_token("PRE_MLP_NORM", 2, d_model, formula=f"2 x d = 2 x {d_model}"),
                _trace_token("PRE_MLP_NORM_TAIL", 1, norm_ctrl_pe_cycles, formula=f"norm_ctrl = {norm_ctrl_pe_cycles}"),
            ],
            trace_ops=[
                _semantic_op(
                    "PRE_MLP_NORM",
                    2,
                    _dimm_pe_to_mem_cycles(d_model, pim_config),
                    addr_kind="stage",
                    stage_tag="pre_mlp_norm",
                ),
                _semantic_op(
                    "PRE_MLP_NORM_TAIL",
                    1,
                    _dimm_pe_to_mem_cycles(norm_ctrl_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="pre_mlp_norm_tail",
                ),
            ],
            key_events_pe=[("MLP_RMS_DONE", pre_mlp_rms_pe_cycles)],
            cycle_eval_mode="memory_detailed",
        )
    )
    next_command_id += 1

    command_schedule.append(
        _linear_command(
            next_command_id,
            "WUP_WGATE[GEMV_INTERLEAVED]",
            up_gate_formula,
            tensor_tag="wup_wgate",
            rtl_phase_ids=[6, 7, 8, 9],
            rtl_phase_names=["CFG_UP_GATE", "WAIT_UP_GATE"],
            trace_setup_token="ALL_BANK_LUT_GEN_AND_LOAD",
            trace_body_token=linear_weight_body_token,
            cfg_pe=int(cfg_pe_cycles),
            tail_breakdown=linear_tail_breakdown,
            tail_token_prefix="WUP_WGATE_TAIL",
            tail_stage_prefix="wup_wgate_tail",
            key_events_pe=[
                ("UP_GATE_GEMV_READY", int(cfg_pe_cycles) + int(up_gate_formula["total_pe_cycles"])),
            ],
        )
    )
    next_command_id += 1

    command_schedule.append(
        _vector_move_command(
            next_command_id,
            "WUP_WGATE_GATHER[MOVE]",
            vec_elems=2 * int(mlp_dim),
            elems_per_cycle=int(trace_vector_elems_per_cycle),
            ctrl_cycles=2 * int(vector_move_ctrl_cycles),
            token_prefix="WUP_WGATE_GATHER",
            stage_prefix="wup_wgate_gather",
            rtl_phase_ids=[7, 9],
            rtl_phase_names=["GATHER_UP", "GATHER_GATE"],
            ready_event="UP_GATE_READY",
            dim_term="2*mlp_dim",
        )
    )
    next_command_id += 1

    command_schedule.append(
        _command(
            next_command_id,
            f"ACT_MUL[SiLU+MUL+ctrl={int(rtl_dimm_ops.mlp_fusion_cycles(mlp_dim))}]",
            pe_cycles=int(rtl_dimm_ops.mlp_fusion_cycles(mlp_dim)),
            formula_text="silu + mul + fusion_ctrl",
            formula_terms={
                "silu": int(mlp_dim),
                "mul": int(mlp_dim),
                "fusion_ctrl": int(mlp_fusion_ctrl_pe_cycles),
            },
            rtl_phase_ids=[10],
            rtl_phase_names=["MLP_FUSION"],
            trace_tokens=[
                _trace_token("MLP_FUSION", 2, mlp_dim, formula=f"2 x mlp_dim = 2 x {mlp_dim}"),
                _trace_token("MLP_FUSION_TAIL", 1, mlp_fusion_ctrl_pe_cycles, formula=f"fusion_ctrl = {mlp_fusion_ctrl_pe_cycles}"),
            ],
            trace_ops=[
                _semantic_op(
                    "MLP_FUSION",
                    2,
                    _dimm_pe_to_mem_cycles(mlp_dim, pim_config),
                    addr_kind="stage",
                    stage_tag="mlp_fusion",
                ),
                _semantic_op(
                    "MLP_FUSION_TAIL",
                    1,
                    _dimm_pe_to_mem_cycles(mlp_fusion_ctrl_pe_cycles, pim_config),
                    addr_kind="stage",
                    stage_tag="mlp_fusion_tail",
                ),
            ],
            key_events_pe=[("MLP_OUT", int(rtl_dimm_ops.mlp_fusion_cycles(mlp_dim)))],
            cycle_eval_mode="memory_detailed",
        )
    )
    next_command_id += 1

    command_schedule.append(
        _linear_command(
            next_command_id,
            "WDOWN[GEMV]",
            down_formula,
            tensor_tag="wdown",
            rtl_phase_ids=[11, 12],
            rtl_phase_names=["CFG_DOWN", "WAIT_DOWN"],
            trace_setup_token="ALL_BANK_LUT_GEN_AND_LOAD",
            trace_body_token=linear_weight_body_token,
            cfg_pe=int(cfg_pe_cycles),
            key_events_pe=[
                ("DOWN_GEMV_READY", int(cfg_pe_cycles) + int(down_formula["total_pe_cycles"])),
            ],
        )
    )
    next_command_id += 1

    command_schedule.append(
        _vector_move_command(
            next_command_id,
            "WDOWN_GATHER[MOVE]",
            vec_elems=int(d_model),
            elems_per_cycle=int(trace_vector_elems_per_cycle),
            ctrl_cycles=int(vector_move_ctrl_cycles),
            token_prefix="WDOWN_GATHER",
            stage_prefix="wdown_gather",
            rtl_phase_ids=[12],
            rtl_phase_names=["GATHER_DOWN"],
            ready_event="DOWN_READY",
            dim_term="d_model",
        )
    )
    next_command_id += 1

    command_schedule.append(
        _command(
            next_command_id,
            "MLP_RESIDUAL[TAIL]",
            pe_cycles=int(mlp_residual_pe_cycles),
            formula_text=" + ".join(mlp_residual_formula_labels),
            formula_terms=mlp_residual_formula_terms,
            rtl_phase_ids=[16],
            rtl_phase_names=["MLP_RESIDUAL"],
            trace_tokens=mlp_residual_trace_tokens,
            trace_ops=[],
            key_events_pe=[
                ("MLP_RESIDUAL_DONE", int(mlp_residual_pe_cycles)),
                ("TOKEN_DONE", int(mlp_residual_pe_cycles)),
            ],
            cycle_eval_mode="memory_detailed",
        )
    )

    full_writer = _TraceBuffer()
    full_bank_req_idx = {b: 0 for b in range(banks)}
    token_timings: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}

    phase_report: List[Dict] = []
    start_cycle = 1
    for item in command_schedule:
        phase_trace_writer = _TraceBuffer()
        phase_token_counts: Dict[str, int] = {}
        phase_token_timings: Dict[str, int] = {}
        for tok in item.get("trace_tokens", []):
            tok_name = str(tok.get("name", ""))
            if not tok_name:
                continue
            phase_token_timings[tok_name] = int(tok.get("memory_cycles_each", 0))
            phase_token_counts[tok_name] = phase_token_counts.get(tok_name, 0) + int(tok.get("count", 0))
        emitted_token_timings: Dict[str, int] = {}
        emitted_token_counts: Dict[str, int] = {}
        trace_mode = str(item["trace_mode"])
        if trace_mode == "explicit_subcommands":
            phase_ops = []
            for raw_op in item.get("trace_ops", []):
                op = dict(raw_op)
                op["bank_layout"] = dict(bank_layout)
                op["topology"] = dict(memory_topology)
                phase_ops.append(op)
            _emit_phase_trace_ops(
                trace_ops=phase_ops,
                writer=phase_trace_writer,
                banks=banks,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                bank_req_idx=full_bank_req_idx,
                token_timings=emitted_token_timings,
                token_counts=emitted_token_counts,
            )
        elif trace_mode != "report_only":
            raise ValueError(f"Unsupported trace_mode={trace_mode}")

        _merge_token_counts(token_counts, emitted_token_counts)
        for key, val in emitted_token_timings.items():
            token_timings[str(key)] = int(val)
        full_writer.lines.extend(phase_trace_writer.lines)

        pe_cycles = int(item["pe_cycles"])
        trace_nominal_memory_cycles = int(sum(
            int(phase_token_timings.get(tok, 0)) * int(cnt)
            for tok, cnt in phase_token_counts.items()
            if tok in phase_token_timings
        ))
        cycle_eval_mode = str(item.get("cycle_eval_mode", "pe"))
        if cycle_eval_mode == "memory_detailed":
            phase_total_cycles = int(trace_nominal_memory_cycles)
        else:
            phase_total_cycles = int(pe_cycles)
        phase_label = f"command {int(item['command_id'])} ({item['command']})"
        logic_end_cycle = int(start_cycle + max(0, pe_cycles) - 1 if pe_cycles > 0 else start_cycle - 1)
        entry = {
            "label": phase_label,
            "command_id": int(item["command_id"]),
            "command": str(item["command"]),
            "formula": str(item.get("formula", "")),
            "formula_terms": dict(item.get("formula_terms", {})),
            "rtl_phase_ids": list(item.get("rtl_phase_ids", [])),
            "rtl_phases": list(item.get("rtl_phases", [])),
            "start_cycle": int(start_cycle),
            "end_cycle": int(start_cycle + max(0, phase_total_cycles) - 1 if phase_total_cycles > 0 else start_cycle - 1),
            "cycles": int(phase_total_cycles),
            "logic_cycles": int(pe_cycles),
            "memory_cycles": int(trace_nominal_memory_cycles),
            "pe_cycles": int(pe_cycles),
            "phase_trace_tokens": dict(phase_token_counts),
            "trace_tokens": [dict(tok) for tok in item.get("trace_tokens", [])],
            "key_events": [],
        }
        for name, rel_pe in item.get("key_events", []):
            rel_pe = max(0, int(rel_pe))
            ev_cycle = int(start_cycle + rel_pe - 1 if rel_pe > 0 else start_cycle - 1)
            entry["key_events"].append({"name": name, "cycle": ev_cycle, "pe_cycle": rel_pe})
        phase_report.append(entry)
        start_cycle = entry["end_cycle"] + 1

    extra_commands = sorted(token_timings.keys())
    extra_timings = {f"n{cmd}": cyc for cmd, cyc in token_timings.items()}
    full_run = _run_runtime_trace_lines(
        lines=full_writer.lines,
        out_dir=out_dir,
        tag="trace_out",
        ramulator_bin=ramulator_bin,
        dram_impl=dram_impl,
        dram_org=dram_org,
        dram_timing=dram_timing,
        chips_per_dimm=chips_per_dimm,
        channel_width=(None if channel_width_bits is None else int(channel_width_bits)),
        channel=dram_channel,
        rank=dram_rank,
        n_lut_bcast=n_lut_bcast,
        n_rdpim=n_rdpim,
        n_macab=n_macab,
        timeout_s=timeout_s,
        extra_timings=extra_timings,
        extra_commands_to_count=extra_commands,
        keep_yaml=True,
        keep_cmd_counter=True,
        keep_cmd_trace=True,
    )

    transformer_block_cycles_pe = int(sum(int(item["pe_cycles"]) for item in command_schedule))
    transformer_block_cycles_logic = int(sum(int(item["logic_cycles"]) for item in phase_report))
    transformer_block_cycles_mem = int(sum(int(item["memory_cycles"]) for item in phase_report))
    transformer_block_cycles_merged = int(sum(int(item["cycles"]) for item in phase_report))
    report_only_ids = {
        int(item["command_id"])
        for item in command_schedule
        if str(item.get("trace_mode", "")) == "report_only"
    }
    report_markers = [
        {
            "command": str(item["command"]),
            "start_cycle": int(item["start_cycle"]),
            "end_cycle": int(item["end_cycle"]),
            "cycles": int(item["cycles"]),
        }
        for item in phase_report
        if int(item["command_id"]) in report_only_ids
    ]
    expected_cmd_counts = {str(k): int(v) for k, v in token_counts.items()}
    observed_cmd_counts = {
        str(k): int(full_run.get("cmd_counter", {}).get(k, 0))
        for k in expected_cmd_counts
    }
    input_trace_token_counts = _count_input_trace_tokens(full_run["files"]["trace"])
    input_trace_command_lines = int(sum(int(v) for v in input_trace_token_counts.values()))
    input_trace_count_match = {
        str(k): (int(input_trace_token_counts.get(str(k), 0)) == int(v))
        for k, v in expected_cmd_counts.items()
    }
    trace_nominal_memory_cycles = int(sum(int(item["memory_cycles"]) for item in phase_report))
    ramulator_memory_cycles = int(full_run.get("memory_system_cycles", -1))
    runtime_output_end_cycle = _resolve_runtime_output_end_cycle(
        memory_system_cycles=ramulator_memory_cycles,
        cmd_trace_path=full_run["files"]["cmd_trace"],
        phase_report=phase_report,
    )
    trace_vs_formula = {
        "expected_command_counts": expected_cmd_counts,
        "observed_command_counts": observed_cmd_counts,
        "input_trace_token_counts": {str(k): int(v) for k, v in input_trace_token_counts.items()},
        "input_trace_command_lines": int(input_trace_command_lines),
        "input_trace_count_match": input_trace_count_match,
        "count_match": {
            str(k): (observed_cmd_counts.get(str(k), 0) == int(v))
            for k, v in expected_cmd_counts.items()
        },
        "nominal_trace_memory_cycles": trace_nominal_memory_cycles,
        "formula_memory_cycles": int(transformer_block_cycles_mem),
        "formula_memory_match": (trace_nominal_memory_cycles == int(transformer_block_cycles_mem)),
        "ramulator_memory_cycles": ramulator_memory_cycles,
        "rtl_phase_cycles": int(transformer_block_cycles_pe),
        "ramulator_minus_formula_cycles": (
            None if ramulator_memory_cycles < 0 else int(ramulator_memory_cycles - transformer_block_cycles_mem)
        ),
    }
    cpu_embed_pe = int(rtl["cpu_token_embed_pe_cycles"])
    cpu_pos_pe = int(rtl["cpu_pos_enc_pe_cycles"])
    cpu_lm_head_pe = int(rtl["cpu_lm_head_pe_cycles"])
    cpu_frontend_pe = cpu_embed_pe + cpu_pos_pe
    ctrl_freq_hz = float(_trace_ctrl_freq_mhz(pim_config) * 1e6)
    cpu_embed_mem = _pe_to_mem_cycles(cpu_embed_pe, pim_config)
    cpu_pos_mem = _pe_to_mem_cycles(cpu_pos_pe, pim_config)
    cpu_lm_head_mem = _pe_to_mem_cycles(cpu_lm_head_pe, pim_config)
    cpu_frontend_mem = cpu_embed_mem + cpu_pos_mem
    full_token_pe = cpu_frontend_pe + (int(modelinfos["ndec"]) * transformer_block_cycles_pe) + cpu_lm_head_pe
    full_token_mem = (
        cpu_frontend_mem
        + int(modelinfos["ndec"]) * transformer_block_cycles_merged
        + cpu_lm_head_mem
    )
    transformer_block_cycles_13_phase = int(sum(int(item["cycles"]) for item in phase_report_13))
    steady_state_per_block_cycles = int(transformer_block_cycles_13_phase)
    steady_state_full_token_cycles = (
        cpu_frontend_mem + int(modelinfos["ndec"]) * steady_state_per_block_cycles + cpu_lm_head_mem
    )
    component_cycle_breakdown = {
        "formula": (
            "decode_full_token_cycles = embedding_cycles + positional_encoding_cycles "
            "+ transformer_block_count * per_transformer_block_cycles + lm_head_cycles "
            "+ runtime_dispatch_host_overhead_cycles"
        ),
        "embedding_cycles": int(cpu_embed_mem),
        "positional_encoding_cycles": int(cpu_pos_mem),
        "frontend_cycles": int(cpu_frontend_mem),
        "per_transformer_block_cycles": int(steady_state_per_block_cycles),
        "per_transformer_block_source": "13_phase_cycle_model",
        "ramulator_memory_system_cycles_for_validation": int(ramulator_memory_cycles),
        "merged_logic_plus_nominal_trace_memory_cycles": int(transformer_block_cycles_merged),
        "transformer_block_count": int(modelinfos["ndec"]),
        "all_transformer_blocks_cycles": int(int(modelinfos["ndec"]) * steady_state_per_block_cycles),
        "lm_head_cycles": int(cpu_lm_head_mem),
        "runtime_dispatch_host_overhead_cycles": 0,
        "decode_full_token_cycles": int(steady_state_full_token_cycles),
        "decode_run_tokens": int(lout),
        "decode_run_cycles": int(steady_state_full_token_cycles * max(1, int(lout))),
    }
    steady_state_decode_latency_ms = (1000.0 * steady_state_full_token_cycles / ctrl_freq_hz) if ctrl_freq_hz > 0 else None
    steady_state_decode_tps = (ctrl_freq_hz / steady_state_full_token_cycles) if steady_state_full_token_cycles > 0 else None

    summary = {
        "runtime_inputs": {
            "model": modelinfos["name"],
            "ndec": int(modelinfos["ndec"]),
            "batch": int(batch),
            "lin": int(lin),
            "lout": int(lout),
            "dram_impl": str(dram_impl),
            "dram_org": str(dram_org),
            "dram_timing": str(dram_timing),
            "dram_channel": int(dram_channel),
            "dram_rank": int(dram_rank),
            "chips_per_dimm": int(chips_per_dimm),
            "channel_width_bits": (
                None if channel_width_bits is None else int(channel_width_bits)
            ),
            "burst_bytes": int(burst_bytes),
            "row_bytes": int(row_bytes),
            "mlp_residual_pe_cycles": int(mlp_residual_pe_cycles),
            "trace_io_bits": int(trace_bus_io_bits),
            "trace_score_bits": int(trace_score_bits),
            "trace_prob_bits": int(trace_prob_bits),
            "trace_vector_elems_per_cycle": int(trace_vector_elems_per_cycle),
            "trace_vector_group_elems": int(trace_vector_group_elems),
            "attention_parallel_heads": int(attention_parallel_heads),
            "attention_head_chunks": [int(v) for v in attention_head_chunks],
            "serialize_attention_heads": bool(serialize_attention_heads),
            **rtl,
        },
        "phase_report": phase_report,
        "phase_report_13": phase_report_13,
        "rtl_phase_report": phase_report,
        "command_schedule": command_schedule,
        "full_trace": full_run,
        "trace_counts": {
            "pseudo_phase_tokens": token_counts,
            "input_trace_token_counts": input_trace_token_counts,
            "input_trace_command_lines": int(input_trace_command_lines),
            "input_trace_count_match": input_trace_count_match,
            "command_granularity_bytes": int(burst_bytes),
            "input_trace_bytes_approx": int(input_trace_command_lines * burst_bytes),
        },
        "validation": trace_vs_formula,
        "design_derivation": {
            "input_dim": d_model,
            "n_tile": outputs_per_launch,
            "banks": banks,
            "pairs_per_rd": pairs_per_rd,
            "k_tile": k_tile,
            "loaded_scalars_per_tile": tile_scalars,
            "head_dim_per_bank": bank_local_cols,
            "mlp_dim": mlp_dim,
            "local_cols_max": local_cols_max,
            "visible_history_tokens": visible_history_tokens,
            "seq_len": visible_tokens,
            "qkt_tok_par": int(qkt_tok_par),
            "tp_par": int(tp_par),
            "qkt_dot_par": int(qkt_dot_par),
            "pv_dim_par": int(pv_dim_par),
            "score_bus_heads_per_step": int(score_bus_heads_per_step),
            "prob_bus_heads_per_step": int(prob_bus_heads_per_step),
            "vector_bus_heads_per_step": int(vector_bus_heads_per_step),
            "attention_parallel_heads": int(attention_parallel_heads),
            "attention_head_chunks": [int(v) for v in attention_head_chunks],
            "serialize_attention_heads": bool(serialize_attention_heads),
        },
        "files": {
            "input_trace": full_run["files"]["trace"],
            "trace_yaml": full_run["files"]["yaml"],
            "run_out": full_run["files"]["run_out"],
            "command_counter": full_run["files"]["cmd_counter"],
            "command_counter_channels": full_run["files"].get("cmd_counter_channels", []),
            "command_trace": full_run["files"]["cmd_trace"],
            "command_trace_channels": full_run["files"].get("cmd_trace_channels", []),
            "full_trace": full_run["files"]["trace"],
            "full_yaml": full_run["files"]["yaml"],
            "full_run_out": full_run["files"]["run_out"],
            "full_cmd_trace": full_run["files"]["cmd_trace"],
            "trace_out_ch0": full_run["files"]["cmd_trace"],
            "command_timeline_trace": None,
        },
        "estimates": {
            "controller_freq_hz": int(round(ctrl_freq_hz)),
            "one_transformer_block_logic_cycles": int(transformer_block_cycles_logic),
            "one_transformer_block_memory_cycles": int(transformer_block_cycles_mem),
            "one_transformer_block_merged_cycles": int(transformer_block_cycles_merged),
            "one_transformer_block_pe_cycles": int(transformer_block_cycles_pe),
            "one_transformer_block_13_phase_cycles": int(transformer_block_cycles_13_phase),
            "total_cycles_to_next_token": int(runtime_output_end_cycle + 1 if runtime_output_end_cycle >= 0 else 0),
            "one_transformer_block_trace_nominal_memory_cycles": int(trace_nominal_memory_cycles),
            "one_transformer_block_ramulator_memory_cycles": int(ramulator_memory_cycles),
            "all_transformer_blocks_logic_cycles": int(modelinfos["ndec"] * transformer_block_cycles_logic),
            "all_transformer_blocks_memory_cycles": int(modelinfos["ndec"] * transformer_block_cycles_mem),
            "all_transformer_blocks_merged_cycles": int(modelinfos["ndec"] * transformer_block_cycles_merged),
            "all_transformer_blocks_pe_cycles": int(modelinfos["ndec"] * transformer_block_cycles_pe),
            "all_transformer_blocks_13_phase_cycles": int(modelinfos["ndec"] * transformer_block_cycles_13_phase),
            "cpu_token_embedding_memory_cycles": int(cpu_embed_mem),
            "cpu_token_embedding_pe_cycles": int(cpu_embed_pe),
            "cpu_positional_encoding_memory_cycles": int(cpu_pos_mem),
            "cpu_positional_encoding_pe_cycles": int(cpu_pos_pe),
            "cpu_lm_head_memory_cycles": int(cpu_lm_head_mem),
            "cpu_lm_head_pe_cycles": int(cpu_lm_head_pe),
            "full_token_all_24_blocks_memory_cycles": int(full_token_mem),
            "full_token_all_24_blocks_pe_cycles": int(full_token_pe),
            "steady_state_full_token_cycles": int(steady_state_full_token_cycles),
            "steady_state_one_token_decode_latency_ms": (
                None if steady_state_decode_latency_ms is None else float(steady_state_decode_latency_ms)
            ),
            "steady_state_decode_tokens_per_second": (
                None if steady_state_decode_tps is None else float(steady_state_decode_tps)
            ),
            "component_cycle_breakdown": component_cycle_breakdown,
        },
        "notes": {
            "cycle_reporting": "phase_report is the command-timeline view; phase_report_13 is the paper 13-phase cycle model used for steady-state decode latency. TRANSFORMER_BLOCK_END and total_cycles_to_next_token still record the single traced transformer-block Ramulator completion point for validation.",
            "total_cycles_to_next_token_note": "This field matches the single traced transformer block end cycle, not the full steady-state next-token decode cost across all decoder layers. Use steady_state_full_token_cycles / steady_state_decode_tokens_per_second for end-to-end decode estimates.",
            "cycle_conversion": (
                f"bank PE: 1 cycle @ {_trace_bank_pe_freq_mhz(pim_config):.3f} MHz -> "
                f"{_trace_ctrl_freq_mhz(pim_config):.3f}/{_trace_bank_pe_freq_mhz(pim_config):.3f} controller cycles; "
                f"dimm PE: 1 cycle @ {_trace_dimm_pe_freq_mhz(pim_config):.3f} MHz -> "
                f"{_trace_ctrl_freq_mhz(pim_config):.3f}/{_trace_dimm_pe_freq_mhz(pim_config):.3f} controller cycles"
            ),
            "trace_scope": "Ramulator trace validates one transformer block using command-level pseudo tokens plus refresh interleaving.",
            "mlp_dim_note": f"logical_ffn_dim={ffn_dim}, trace_mlp_dim={mlp_dim}",
            "kv_writeback_model": "Current-token K/V cache commit is still emitted into the semantic trace for Ramulator visibility, but it is no longer counted as a standalone RTL timing item in the 13-phase lifecycle model.",
            "rope_model": "RoPE is emitted as ALL_BANK_ROPE under the high-level QKV command after an explicit ALL_BANK_DEQUANT step.",
            "kv_read_model": "Attention-window access is emitted as one ALL_BANK_READ_K_CACHE / ALL_BANK_READ_V_CACHE request per history-token burst with explicit row/column addr_vec.",
            "attention_head_scheduling": "If num_heads exceeds the bank-parallel head budget, QKV/QKT/PV are serialized in head chunks so each bank PE handles at most one head per pass; WO and MLP stages remain unchanged.",
            "trace_addressing": "Semantic runtime trace emits explicit addr_vec entries and now stripes requests across the configured channel/rank topology while preserving concrete bank/row/column locations, so multi-channel and multi-rank DDR5 sweeps change both command addresses and achievable concurrency.",
            "mlp_residual_model": "Final MLP residual is emitted as explicit subcommands after WDOWN, and bus-transfer/gather/concat stages now generate real trace tokens; Ramulator TraceRecorder output is the command-level source of truth.",
            "not_in_trace": [
                "CPU token embedding / positional encoding / lm_head undo embedding",
            ],
        },
    }
    _append_trace_report_to_run_out(
        run_out_path=full_run["files"]["run_out"],
        phase_report=phase_report,
        expected_cmd_counts=expected_cmd_counts,
        observed_cmd_counts=observed_cmd_counts,
        ramulator_memory_cycles=ramulator_memory_cycles,
        formula_memory_cycles=int(transformer_block_cycles_mem),
    )
    _append_trace_report_markers(
        trace_path=full_run["files"]["trace"],
        cmd_trace_path=full_run["files"]["cmd_trace"],
        markers=report_markers,
        end_cycle=int(runtime_output_end_cycle),
    )
    summary["files"]["summary_json"] = None
    if write_summary_json:
        summary_path = os.path.join(out_dir, "token_lifecycle_runtime_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summary["files"]["summary_json"] = summary_path
    return summary


def _emit_fc_phase(
    f,
    phase_name: str,
    in_dim: int,
    out_dim: int,
    act_vectors: int,
    banks: int,
    n_macab: int,
    burst_bytes: int,
    row_bytes: int,
    bank_shift: int,
    base_lut_addr: int,
    base_mac_addr: int,
    bank_req_idx: Dict[int, int],
    lut_req_idx: int,
    lut_tiling_elements: int,
    lut_macab_span_elements: int,
) -> Tuple[Dict[str, int], int]:
    cols_per_bank = int(math.ceil(max(1, int(out_dim)) / float(max(1, int(banks)))))
    plan = _bitlinear_trace_plan(
        k_total=max(1, int(in_dim)),
        columns_per_bank=max(1, cols_per_bank),
        act_vectors=max(1, int(act_vectors)),
        pe_config=max(1, int(n_macab)),
        fixed_lut_elements=(int(lut_tiling_elements) or None),
        macab_span_elements=max(1, int(lut_macab_span_elements)),
    )

    counts = {"BITACC_LUT_BCAST": 0, "BITACC_FC_RD": 0}
    for _av in range(int(plan["act_vectors"])):
        for tile in plan["tile_sequence"]:
            lut_addr = _addr_for_bank(
                bank_id=0,
                req_idx_for_bank=lut_req_idx,
                bank_shift=bank_shift,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                base_addr=base_lut_addr,
                total_banks=max(1, int(banks)),
            )
            lut_req_idx += 1
            f.write(f"BITACC_LUT_BCAST 0x{lut_addr:x}\n")
            counts["BITACC_LUT_BCAST"] += 1

            for group_idx in range(cols_per_bank):
                bank_id = group_idx % max(1, banks)
                for _ in range(max(1, int(tile["macab_per_column"]))):
                    addr = _addr_for_bank(
                        bank_id=bank_id,
                        req_idx_for_bank=bank_req_idx[bank_id],
                        bank_shift=bank_shift,
                        burst_bytes=burst_bytes,
                        row_bytes=row_bytes,
                        base_addr=base_mac_addr,
                        total_banks=max(1, int(banks)),
                    )
                    bank_req_idx[bank_id] += 1
                    f.write(f"BITACC_FC_RD 0x{addr:x}\n")
                    counts["BITACC_FC_RD"] += 1

    counts["phase"] = phase_name
    return counts, lut_req_idx


def generate_token_lifecycle_trace(
    dse_config: str,
    out_dir: str,
    ramulator_bin: str,
    timeout_s: int,
    bank_shift: int | None,
    n_lut_bcast: int,
    n_rdpim: int,
    n_macab: int,
    force_visible_tokens: int | None = None,
) -> Dict:
    cfg = _load_dse_config(dse_config)
    w = cfg["workload"]
    d = cfg["dram"]
    bw = cfg["bit_widths"]
    a = cfg["architecture"]
    e = cfg["evaluation"]

    d_model = int(w.d_model)
    banks = max(1, int(a.banks))
    n_heads = max(1, int(w.n_heads))
    head_dim = max(1, d_model // n_heads)
    ffn_dim = max(1, int(w.ffn_mult) * d_model)
    burst_bytes = max(1, int(d.burst_bytes))
    row_bytes = max(1, int(d.row_bytes))
    out_bits = max(1, int(bw.out_bits))

    history_tokens = max(0, int(e.nth_token) - 1)
    if int(a.kv_cache_max_tokens) > 0:
        history_tokens = min(history_tokens, int(a.kv_cache_max_tokens))
    visible_tokens = history_tokens + 1
    if force_visible_tokens is not None and force_visible_tokens > 0:
        visible_tokens = int(force_visible_tokens)

    os.makedirs(out_dir, exist_ok=True)
    if bank_shift is None:
        bank_shift = _autotune_bank_shift(
            ramulator_bin=ramulator_bin,
            out_dir=out_dir,
            burst_bytes=burst_bytes,
            banks=banks,
            row_bytes=row_bytes,
            timeout_s=timeout_s,
        )

    trace_path = os.path.join(out_dir, "token_lifecycle.trace")
    yaml_path = os.path.join(out_dir, "token_lifecycle.yaml")
    run_out_path = os.path.join(out_dir, "token_lifecycle_run.out")
    cmd_counter_path = os.path.join(out_dir, "token_lifecycle_cmd_counter.txt")
    cmd_trace_prefix = os.path.join(out_dir, "token_lifecycle_cmd_trace")
    summary_path = os.path.join(out_dir, "token_lifecycle_summary.json")

    base_lut = 0x10000000
    base_fc = 0x20000000
    base_kvwb = 0x30000000
    base_attn = 0x40000000
    base_attn_wr = 0x50000000
    base_pv_wr = 0x60000000

    bank_req_idx_fc = {b: 0 for b in range(banks)}
    bank_req_idx_kv = {b: 0 for b in range(banks)}
    bank_req_idx_attn = {b: 0 for b in range(banks)}
    bank_req_idx_attn_wr = {b: 0 for b in range(banks)}
    lut_req_idx = 0

    phase_counts: List[Dict[str, int]] = []
    req_counts = {
        "BITACC_LUT_BCAST": 0,
        "BITACC_FC_RD": 0,
        "BITACC_KV_WB": 0,
        "BITACC_ATTN_RD": 0,
        "BITACC_ATTN_WR": 0,
    }

    with open(trace_path, "w", encoding="utf-8") as f:
        # 1) Q/K/V linear projection (bank-level BLU+ACC+dequant pipeline).
        qkv_counts, lut_req_idx = _emit_fc_phase(
            f=f,
            phase_name="QKV_LINEAR",
            in_dim=d_model,
            out_dim=3 * d_model,
            act_vectors=1,
            banks=banks,
            n_macab=n_macab,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_lut_addr=base_lut,
            base_mac_addr=base_fc,
            bank_req_idx=bank_req_idx_fc,
            lut_req_idx=lut_req_idx,
            lut_tiling_elements=int(getattr(a, "lut_tiling_elements", 0)),
            lut_macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
        )
        phase_counts.append(qkv_counts)
        req_counts["BITACC_LUT_BCAST"] += qkv_counts["BITACC_LUT_BCAST"]
        req_counts["BITACC_FC_RD"] += qkv_counts["BITACC_FC_RD"]

        # 2) K/V writeback to local bank KV rows after dequant.
        kv_bytes_total = int(math.ceil((2 * d_model) * out_bits / 8.0))
        kv_bytes_per_bank = int(math.ceil(kv_bytes_total / float(banks)))
        kv_reqs = 0
        for b in range(banks):
            kv_reqs += _emit_reqs(
                f=f,
                token="BITACC_KV_WB",
                bank_id=b,
                nbytes=kv_bytes_per_bank,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                base_addr=base_kvwb,
                bank_req_idx=bank_req_idx_kv,
            )
        phase_counts.append({"phase": "KV_WRITEBACK", "BITACC_KV_WB": kv_reqs})
        req_counts["BITACC_KV_WB"] += kv_reqs

        # 3) QK^T read path (bank attention PE reads K cache and writes local score stats).
        k_bytes_per_bank = int(math.ceil(visible_tokens * head_dim * (out_bits / 8.0)))
        qkt_rd = 0
        qkt_wr = 0
        for b in range(banks):
            qkt_rd += _emit_reqs(
                f=f,
                token="BITACC_ATTN_RD",
                bank_id=b,
                nbytes=k_bytes_per_bank,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                base_addr=base_attn,
                bank_req_idx=bank_req_idx_attn,
            )
            # One scalar packet (max/sumexp) per bank.
            qkt_wr += _emit_reqs(
                f=f,
                token="BITACC_ATTN_WR",
                bank_id=b,
                nbytes=8,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                base_addr=base_attn_wr,
                bank_req_idx=bank_req_idx_attn_wr,
            )
        phase_counts.append(
            {"phase": "QKT_SCORE", "BITACC_ATTN_RD": qkt_rd, "BITACC_ATTN_WR": qkt_wr}
        )
        req_counts["BITACC_ATTN_RD"] += qkt_rd
        req_counts["BITACC_ATTN_WR"] += qkt_wr

        # 4) Softmax post-process is DIMM/global logic only (no DRAM token).
        phase_counts.append({"phase": "SOFTMAX_POST", "DRAM_REQS": 0})

        # 5) P*V read/write path.
        v_bytes_per_bank = int(math.ceil(visible_tokens * head_dim * (out_bits / 8.0)))
        pv_out_bytes_per_bank = int(math.ceil(head_dim * (out_bits / 8.0)))
        pv_rd = 0
        pv_wr = 0
        for b in range(banks):
            pv_rd += _emit_reqs(
                f=f,
                token="BITACC_ATTN_RD",
                bank_id=b,
                nbytes=v_bytes_per_bank,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                base_addr=base_attn + 0x01000000,
                bank_req_idx=bank_req_idx_attn,
            )
            pv_wr += _emit_reqs(
                f=f,
                token="BITACC_ATTN_WR",
                bank_id=b,
                nbytes=pv_out_bytes_per_bank,
                burst_bytes=burst_bytes,
                row_bytes=row_bytes,
                bank_shift=bank_shift,
                base_addr=base_pv_wr,
                bank_req_idx=bank_req_idx_attn_wr,
            )
        phase_counts.append({"phase": "PV_CONTEXT", "BITACC_ATTN_RD": pv_rd, "BITACC_ATTN_WR": pv_wr})
        req_counts["BITACC_ATTN_RD"] += pv_rd
        req_counts["BITACC_ATTN_WR"] += pv_wr

        # 6) Wo projection.
        wo_counts, lut_req_idx = _emit_fc_phase(
            f=f,
            phase_name="WO_LINEAR",
            in_dim=d_model,
            out_dim=d_model,
            act_vectors=1,
            banks=banks,
            n_macab=n_macab,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_lut_addr=base_lut + 0x01000000,
            base_mac_addr=base_fc + 0x01000000,
            bank_req_idx=bank_req_idx_fc,
            lut_req_idx=lut_req_idx,
            lut_tiling_elements=int(getattr(a, "lut_tiling_elements", 0)),
            lut_macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
        )
        phase_counts.append(wo_counts)
        req_counts["BITACC_LUT_BCAST"] += wo_counts["BITACC_LUT_BCAST"]
        req_counts["BITACC_FC_RD"] += wo_counts["BITACC_FC_RD"]

        # 7) Residual + LayerNorm is DIMM/global logic only.
        phase_counts.append({"phase": "RES_LN", "DRAM_REQS": 0})

        # 8) MLP Wup.
        wup_counts, lut_req_idx = _emit_fc_phase(
            f=f,
            phase_name="WUP_LINEAR",
            in_dim=d_model,
            out_dim=ffn_dim,
            act_vectors=1,
            banks=banks,
            n_macab=n_macab,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_lut_addr=base_lut + 0x02000000,
            base_mac_addr=base_fc + 0x02000000,
            bank_req_idx=bank_req_idx_fc,
            lut_req_idx=lut_req_idx,
            lut_tiling_elements=int(getattr(a, "lut_tiling_elements", 0)),
            lut_macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
        )
        phase_counts.append(wup_counts)
        req_counts["BITACC_LUT_BCAST"] += wup_counts["BITACC_LUT_BCAST"]
        req_counts["BITACC_FC_RD"] += wup_counts["BITACC_FC_RD"]

        # 9) MLP Wgate.
        wgate_counts, lut_req_idx = _emit_fc_phase(
            f=f,
            phase_name="WGATE_LINEAR",
            in_dim=d_model,
            out_dim=ffn_dim,
            act_vectors=1,
            banks=banks,
            n_macab=n_macab,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_lut_addr=base_lut + 0x03000000,
            base_mac_addr=base_fc + 0x03000000,
            bank_req_idx=bank_req_idx_fc,
            lut_req_idx=lut_req_idx,
            lut_tiling_elements=int(getattr(a, "lut_tiling_elements", 0)),
            lut_macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
        )
        phase_counts.append(wgate_counts)
        req_counts["BITACC_LUT_BCAST"] += wgate_counts["BITACC_LUT_BCAST"]
        req_counts["BITACC_FC_RD"] += wgate_counts["BITACC_FC_RD"]

        # 10) MLP activation/gating fusion is logic only.
        phase_counts.append({"phase": "MLP_ACT_GATE", "DRAM_REQS": 0})

        # 11) MLP Wdown.
        wdown_counts, lut_req_idx = _emit_fc_phase(
            f=f,
            phase_name="WDOWN_LINEAR",
            in_dim=ffn_dim,
            out_dim=d_model,
            act_vectors=1,
            banks=banks,
            n_macab=n_macab,
            burst_bytes=burst_bytes,
            row_bytes=row_bytes,
            bank_shift=bank_shift,
            base_lut_addr=base_lut + 0x04000000,
            base_mac_addr=base_fc + 0x04000000,
            bank_req_idx=bank_req_idx_fc,
            lut_req_idx=lut_req_idx,
            lut_tiling_elements=int(getattr(a, "lut_tiling_elements", 0)),
            lut_macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
        )
        phase_counts.append(wdown_counts)
        req_counts["BITACC_LUT_BCAST"] += wdown_counts["BITACC_LUT_BCAST"]
        req_counts["BITACC_FC_RD"] += wdown_counts["BITACC_FC_RD"]

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(
            _make_yaml(
                path_trace=os.path.abspath(trace_path),
                path_cmd_counter=os.path.abspath(cmd_counter_path),
                path_cmd_trace_prefix=os.path.abspath(cmd_trace_prefix),
                enable_refresh=bool(d.enable_refresh),
                n_lut_bcast=int(n_lut_bcast),
                n_rdpim=int(n_rdpim),
                n_macab=int(n_macab),
                n_refi=int(d.tREFI),
                n_rfc1=int(d.tRFC),
            )
        )

    timed_out = False
    try:
        out_text = _run_ramulator(ramulator_bin, yaml_path, timeout_s)
    except subprocess.TimeoutExpired as te:
        timed_out = True
        out_text = (te.stdout or "") if isinstance(te.stdout, str) else ""
        out_text += "\n[token-lifecycle-trace] Ramulator run timed out.\n"
    with open(run_out_path, "w", encoding="utf-8") as rf:
        rf.write(out_text)

    cmd_trace_path = f"{cmd_trace_prefix}.ch0"
    cmd_info = _parse_cmd_trace(cmd_trace_path)
    cmd_counter = _parse_cmd_counter(cmd_counter_path)
    run_stats = _parse_run_stats(out_text)

    summary = {
        "source_dse_config": dse_config,
        "timed_out": timed_out,
        "memory_system_cycles": _extract_memory_cycles(out_text),
        "request_counts": req_counts,
        "phase_counts": phase_counts,
        "cmd_counter": cmd_counter,
        "cmd_analysis": cmd_info,
        "run_stats": run_stats,
        "derived": {
            "d_model": d_model,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "ffn_dim": ffn_dim,
            "banks": banks,
            "visible_tokens": int(visible_tokens),
            "history_tokens": int(max(0, visible_tokens - 1)),
            "burst_bytes": burst_bytes,
            "row_bytes": row_bytes,
            "out_bits": out_bits,
            "bank_shift": int(bank_shift),
        },
        "files": {
            "trace": trace_path,
            "yaml": yaml_path,
            "run_out": run_out_path,
            "cmd_counter": cmd_counter_path,
            "cmd_trace": cmd_trace_path,
            "summary": summary_path,
        },
        "dse_snapshot": {
            "workload": asdict(w),
            "dram": asdict(d),
            "bit_widths": asdict(bw),
            "architecture": asdict(a),
            "evaluation": asdict(e),
        },
        "notes": {
            "mapped_to_dram_tokens": [
                "QKV/WO/WUP/WGATE/WDOWN => BITACC_LUT_BCAST + BITACC_FC_RD",
                "KV writeback => BITACC_KV_WB",
                "QK^T / P*V memory traffic => BITACC_ATTN_RD + BITACC_ATTN_WR",
            ],
            "logic_only_not_in_trace": [
                "softmax postproc on DIMM/global PE",
                "residual + layernorm",
                "mlp activation + gate fusion",
            ],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate one end-to-end token lifecycle trace (QKV->QKT->PV->WO->MLP) and run Ramulator2."
    )
    p.add_argument("--dse_config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ramulator_bin", default="ramulator2/build/ramulator2")
    p.add_argument("--timeout_s", type=int, default=1200)
    p.add_argument("--bank_shift", type=int, default=None, help="address bank-tag shift; autotuned if omitted")
    p.add_argument("--n_lut_bcast", type=int, default=32)
    p.add_argument("--n_rdpim", type=int, default=12)
    p.add_argument("--n_macab", type=int, default=2)
    p.add_argument("--visible_tokens", type=int, default=None, help="force visible KV tokens (history+current)")
    args = p.parse_args()

    summary = generate_token_lifecycle_trace(
        dse_config=args.dse_config,
        out_dir=args.out_dir,
        ramulator_bin=args.ramulator_bin,
        timeout_s=args.timeout_s,
        bank_shift=args.bank_shift,
        n_lut_bcast=args.n_lut_bcast,
        n_rdpim=args.n_rdpim,
        n_macab=args.n_macab,
        force_visible_tokens=args.visible_tokens,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
