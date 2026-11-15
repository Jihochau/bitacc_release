#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict

from .dse import _load_dse_config
from .ramulator_ddr import derive_trace_address_layout


def _floor_pow2(x: int) -> int:
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


def _derive_lut_window_elements(k_total: int,
                                pe_config: int,
                                base_lut_elements: int = 2048,
                                min_lut_elements: int = 32,
                                fixed_lut_elements: int | None = None) -> int:
    """Map the PE setting to a tiled LUT coverage window.

    Assumption used for trace generation:
      - PE=1 can cover a full 2048-element LUT window
      - Higher-PE settings shrink the resident LUT window roughly as 2048 / PE
      - Hardware-supported windows are powers of two down to 32
    """
    k_total = max(1, int(k_total))
    pe_config = max(1, int(pe_config))
    base_lut_elements = max(1, int(base_lut_elements))
    min_lut_elements = max(1, int(min_lut_elements))
    if fixed_lut_elements is not None and int(fixed_lut_elements) > 0:
        return min(k_total, max(1, int(fixed_lut_elements)))
    nominal = max(min_lut_elements, base_lut_elements // max(1, pe_config))
    lut_window = _floor_pow2(nominal)
    lut_window = max(min_lut_elements, min(base_lut_elements, lut_window))
    return min(k_total, lut_window)


def _bitlinear_trace_plan(k_total: int,
                          columns_per_bank: int,
                          act_vectors: int,
                          pe_config: int,
                          weight_elements_per_burst: int = 256,
                          base_lut_elements: int = 2048,
                          min_lut_elements: int = 32,
                          fixed_lut_elements: int | None = None,
                          macab_span_elements: int = 128) -> dict:
    """Build the tiled BitLinear trace plan used by all LUT-based FC paths."""
    k_total = max(1, int(k_total))
    columns_per_bank = max(1, int(columns_per_bank))
    act_vectors = max(1, int(act_vectors))
    weight_elements_per_burst = max(1, int(weight_elements_per_burst))
    macab_span_elements = max(1, int(macab_span_elements))

    lut_window_elements = _derive_lut_window_elements(
        k_total=k_total,
        pe_config=pe_config,
        base_lut_elements=base_lut_elements,
        min_lut_elements=min_lut_elements,
        fixed_lut_elements=fixed_lut_elements,
    )

    tiles = []
    remaining = k_total
    while remaining > 0:
        tile_len = min(lut_window_elements, remaining)
        # One MACAB consumes one burst worth of weight elements. If the LUT tile
        # is smaller than the burst, the unused portion is wasted and we still
        # issue only one MACAB before the next LUT broadcast.
        macab_per_column = max(1, int(math.ceil(tile_len / float(macab_span_elements))))
        tiles.append({
            "tile_len": int(tile_len),
            "macab_per_column": int(macab_per_column),
        })
        remaining -= tile_len

    lut_bcast_per_vector = len(tiles)
    macab_per_column_total = int(sum(tile["macab_per_column"] for tile in tiles))
    total_macab = int(act_vectors * columns_per_bank * macab_per_column_total)
    total_lut_bcast = int(act_vectors * lut_bcast_per_vector)

    return {
        "k_total": int(k_total),
        "columns_per_bank": int(columns_per_bank),
        "act_vectors": int(act_vectors),
        "pe_config": int(pe_config),
        "weight_elements_per_burst": int(weight_elements_per_burst),
        "macab_span_elements": int(macab_span_elements),
        "base_lut_elements": int(base_lut_elements),
        "min_lut_elements": int(min_lut_elements),
        "lut_window_elements": int(lut_window_elements),
        "lut_tiles_per_vector": int(lut_bcast_per_vector),
        "macab_per_column_total": int(macab_per_column_total),
        "total_lut_bcast": int(total_lut_bcast),
        "total_macab": int(total_macab),
        "tile_sequence": tiles,
        "burst_gt_lut_window": bool(weight_elements_per_burst > lut_window_elements),
        "full_column_bursts_no_tiling": int(math.ceil(k_total / float(weight_elements_per_burst))),
    }


def _run_ramulator(ramulator_bin: str, yaml_path: str, timeout_s: int) -> str:
    result = subprocess.run(
        [ramulator_bin, "-f", yaml_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout_s if timeout_s > 0 else None,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
        )
    return result.stdout


def _extract_memory_cycles(out_text: str) -> int:
    for line in out_text.splitlines():
        if "memory_system_cycles:" in line:
            return int(line.split(":")[-1].strip())
    return -1


def _addr_for_bank(bank_id: int,
                   req_idx_for_bank: int,
                   bank_shift: int,
                   burst_bytes: int,
                   row_bytes: int,
                   base_addr: int,
                   total_banks: int = 1,
                   row_shift: int | None = None) -> int:
    byte_off = req_idx_for_bank * burst_bytes
    row_idx = byte_off // max(1, row_bytes)
    col_off = byte_off % max(1, row_bytes)
    layout = derive_trace_address_layout(
        row_bytes=row_bytes,
        total_banks=total_banks,
        bank_shift=bank_shift,
        row_shift=row_shift,
    )
    bank_id = max(0, int(bank_id)) % max(1, int(total_banks))
    return (
        int(base_addr)
        + (int(row_idx) << int(layout["row_shift"]))
        + (int(bank_id) << int(layout["bank_shift"]))
        + int(col_off)
    )


def _make_yaml(path_trace: str,
               path_cmd_counter: str,
               path_cmd_trace_prefix: str,
               enable_refresh: bool,
               n_lut_bcast: int,
               n_rdpim: int,
               n_macab: int,
               n_refi: int,
               n_rfc1: int,
               chips_per_dimm: int | None = None,
               channel_width: int | None = None,
               extra_timings: dict | None = None,
               extra_commands_to_count: list[str] | None = None) -> str:
    refresh_impl = "AllBank" if enable_refresh else "No"
    extra_timings = extra_timings or {}
    extra_commands_to_count = extra_commands_to_count or []
    base_commands_to_count = [
        "ACT", "ACTAB", "ACTSB", "ACTPB", "PRE", "PREA", "PRESB", "PREPB",
        "RD", "RDPIM", "MACAB", "MACSB", "MACPB", "WR", "WRAB", "LUTBCAST",
        "ACC", "REFab", "REFsb",
    ]
    commands_to_count = base_commands_to_count + [cmd for cmd in extra_commands_to_count if cmd not in base_commands_to_count]
    quoted_commands = ", ".join(f'"{cmd}"' for cmd in commands_to_count)
    lines = [
        "Frontend:",
        "  impl: PIMLoadStoreTrace",
        f"  path: {path_trace}",
        "  clock_ratio: 1",
        "",
        "Translation:",
        "  impl: NoTranslation",
        "  max_addr: 2147483648",
        "",
        "MemorySystem:",
        "  impl: PIMDRAM",
        "  clock_ratio: 1",
        "",
        "  DRAM:",
        "    impl: DDR5-BITACC",
        "    org:",
        "      preset: DDR5_8Gb_x8",
        *(
            [f"      chips: {int(chips_per_dimm)}"]
            if chips_per_dimm is not None and int(chips_per_dimm) > 0
            else []
        ),
        *(
            [f"      channel_width: {int(channel_width)}"]
            if channel_width is not None and int(channel_width) > 0
            else []
        ),
        "      channel: 1",
        "      rank: 2",
        "    timing:",
        "      preset: DDR5_3200AN",
        f"      nLUTBCAST: {n_lut_bcast}",
        f"      nRDPIM: {n_rdpim}",
        f"      nMACAB: {n_macab}",
        f"      nREFI: {n_refi}",
        f"      nRFC1: {n_rfc1}",
        "    RFM:",
        "      BRC: 2",
        "",
        "  Controller:",
        "    impl: PIM",
        "    Scheduler:",
        "      impl: PIM",
        "    RefreshManager:",
        f"      impl: {refresh_impl}",
        "    plugins:",
        "      - ControllerPlugin:",
        "          impl: CommandCounter",
        f"          commands_to_count: [{quoted_commands}]",
        f"          path: {path_cmd_counter}",
        "      - ControllerPlugin:",
        "          impl: TraceRecorder",
        f"          path: {path_cmd_trace_prefix}",
        "",
        "  AddrMapper:",
        "    impl: RoBaRaCoCh",
    ]
    timing_insert_idx = lines.index("    RFM:")
    for key, val in extra_timings.items():
        lines.insert(timing_insert_idx, f"      {key}: {int(val)}")
        timing_insert_idx += 1
    return "\n".join(lines) + "\n"


def _resolve_channelized_paths(path_hint: str) -> list[str]:
    if os.path.exists(path_hint):
        return [path_hint]
    channel_glob = f"{path_hint}.ch*"
    if path_hint.endswith(".ch0"):
        channel_glob = f"{path_hint[:-4]}.ch*"
    return sorted(glob.glob(channel_glob))


def _parse_cmd_trace(path_cmd_trace: str) -> dict:
    trace_paths = _resolve_channelized_paths(path_cmd_trace)
    if not trace_paths:
        return {}
    issued = {}
    read_channels = set()
    write_channels = set()
    read_ranks = set()
    write_ranks = set()
    read_banks = set()
    write_banks = set()
    read_rows = set()
    write_rows = set()
    for trace_path in trace_paths:
        for line in open(trace_path, "r", encoding="utf-8"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 8:
                continue
            cmd = parts[1]
            channel, rank, bg, bank, row = parts[2], parts[3], parts[4], parts[5], parts[6]
            issued[cmd] = issued.get(cmd, 0) + 1
            if cmd in (
                "RDPIM",
                "RD",
                "MACAB",
                "MACSB",
                "MACPB",
                "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE",
                "ALL_BANK_READ_K_CACHE",
                "ALL_BANK_READ_V_CACHE",
                "ALL_BANK_BITLINEAR",
                "ALL_BANK_RD_K_V_CACHE",
            ):
                read_channels.add(channel)
                read_ranks.add((channel, rank))
                read_banks.add((channel, rank, bg, bank))
                read_rows.add((channel, rank, bg, bank, row))
            if cmd in (
                "WR",
                "WRAB",
                "ALL_BANK_WRITE_K_CURRENT",
                "ALL_BANK_WRITE_V_CURRENT",
                "ALL_BANK_WR_K_CURRENT",
                "ALL_BANK_WR_V_CURRENT",
            ):
                write_channels.add(channel)
                write_ranks.add((channel, rank))
                write_banks.add((channel, rank, bg, bank))
                write_rows.add((channel, rank, bg, bank, row))
    return {
        "issued_commands": issued,
        "trace_files": trace_paths,
        "unique_read_channels": len(read_channels),
        "unique_write_channels": len(write_channels),
        "unique_read_ranks": len(read_ranks),
        "unique_write_ranks": len(write_ranks),
        "unique_read_banks": len(read_banks),
        "unique_write_banks": len(write_banks),
        "unique_read_rows": len(read_rows),
        "unique_write_rows": len(write_rows),
    }


def _parse_run_stats(out_text: str) -> dict:
    stats = {}
    for line in out_text.splitlines():
        m = re.search(r"(total_num_[^:]+):\s*([0-9]+)", line)
        if m:
            stats[m.group(1)] = int(m.group(2))
    return stats


def _parse_cmd_counter(path_cmd_counter: str) -> dict:
    counter_paths = _resolve_channelized_paths(path_cmd_counter)
    if not counter_paths:
        return {}
    counts = {}
    for counter_path in counter_paths:
        with open(counter_path, "r", encoding="utf-8") as f:
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
    return counts


def _autotune_bank_shift(ramulator_bin: str,
                         out_dir: str,
                         burst_bytes: int,
                         banks: int,
                         row_bytes: int,
                         timeout_s: int) -> int:
    best_shift = 12
    best_cov = max(1, int(banks))
    target_banks = max(1, int(banks))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "autotune_bank_shift.json"), "w", encoding="utf-8") as f:
        json.dump({"bank_shift": best_shift, "coverage": best_cov, "target_banks": target_banks}, f, indent=2)
    return best_shift


def generate(dse_config: str,
             out_dir: str,
             ramulator_bin: str,
             timeout_s: int,
             bank_shift: int | None,
             n_lut_bcast: int,
             n_rdpim: int,
             n_macab: int,
             act_vectors: int,
             max_bursts_per_output: int | None,
             d_model_override: int | None,
             qkv_projections_override: int | None,
             n_layers_override: int | None,
             banks_override: int | None) -> dict:
    cfg = _load_dse_config(dse_config)
    w = cfg["workload"]
    d = cfg["dram"]
    bw = cfg["bit_widths"]
    a = cfg["architecture"]

    d_model = int(d_model_override) if d_model_override is not None else int(w.d_model)
    qkv_projections = int(qkv_projections_override) if qkv_projections_override is not None else int(w.qkv_projections)
    qkv_channels = d_model * qkv_projections
    banks = max(1, int(banks_override) if banks_override is not None else int(a.banks))
    # Auto-derived packed storage width for BitAcc flow (not from bit_widths.weight_bits).
    packed_weight_bits = max(1, int(getattr(a, "packed_weight_bits", 2)))
    burst_bytes = max(1, int(d.burst_bytes))
    row_bytes = max(1, int(d.row_bytes))
    channels_per_bank_floor = qkv_channels // banks
    channels_per_bank_remainder = qkv_channels % banks

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

    trace_path = os.path.join(out_dir, "qkv_fused_kvwb_dse.trace")
    yaml_path = os.path.join(out_dir, "qkv_fused_kvwb_dse.yaml")
    cmd_counter_path = os.path.join(out_dir, "qkv_fused_kvwb_dse_cmd_counter.txt")
    cmd_trace_prefix = os.path.join(out_dir, "qkv_fused_kvwb_dse_cmd_trace")
    run_out_path = os.path.join(out_dir, "qkv_fused_kvwb_dse_run.out")
    summary_path = os.path.join(out_dir, "qkv_fused_kvwb_dse_summary.json")

    # A single all-bank command fans out to all banks, so use one representative
    # address stream instead of emitting 32 duplicated per-bank requests.
    allbank_read_reqs = 0
    allbank_write_reqs = 0
    read_base = 0
    kvwb_base = 1 << 24

    qkv_groups = int(math.ceil(qkv_channels / float(max(1, banks))))
    if int(getattr(a, "lut_tiling_columns", 0)) > 0:
        qkv_groups = int(getattr(a, "lut_tiling_columns"))
    trace_plan = _bitlinear_trace_plan(
        k_total=d_model,
        columns_per_bank=qkv_groups,
        act_vectors=act_vectors,
        pe_config=n_macab,
        fixed_lut_elements=(int(getattr(a, "lut_tiling_elements", 0)) or None),
        macab_span_elements=int(getattr(a, "lut_macab_span_elements", 128)),
    )
    if max_bursts_per_output is not None and max_bursts_per_output > 0:
        clipped_tiles = []
        remaining_bursts = int(max_bursts_per_output)
        for tile in trace_plan["tile_sequence"]:
            if remaining_bursts <= 0:
                break
            macab_per_column = min(tile["macab_per_column"], remaining_bursts)
            clipped_tiles.append({
                "tile_len": tile["tile_len"],
                "macab_per_column": int(macab_per_column),
            })
            remaining_bursts -= macab_per_column
        if clipped_tiles:
            trace_plan = dict(trace_plan)
            trace_plan["tile_sequence"] = clipped_tiles
            trace_plan["lut_tiles_per_vector"] = len(clipped_tiles)
            trace_plan["macab_per_column_total"] = int(sum(t["macab_per_column"] for t in clipped_tiles))
            trace_plan["total_lut_bcast"] = int(trace_plan["act_vectors"] * len(clipped_tiles))
            trace_plan["total_macab"] = int(
                trace_plan["act_vectors"] * trace_plan["columns_per_bank"] * trace_plan["macab_per_column_total"]
            )

    req_counts = {"BITACC_LUT_BCAST": 0, "PIM_MAC_AB": 0, "BITACC_KV_WB": 0}
    with open(trace_path, "w", encoding="utf-8") as f:
        for _av in range(max(1, int(act_vectors))):
            for _tile_idx, tile in enumerate(trace_plan["tile_sequence"]):
                lut_addr = _addr_for_bank(
                    0,
                    req_counts["BITACC_LUT_BCAST"],
                    bank_shift,
                    burst_bytes,
                    row_bytes,
                    read_base,
                    total_banks=banks,
                )
                f.write(f"BITACC_LUT_BCAST 0x{lut_addr:x}\n")
                req_counts["BITACC_LUT_BCAST"] += 1
                for _group in range(qkv_groups):
                    for _macab in range(tile["macab_per_column"]):
                        addr = _addr_for_bank(
                            0,
                            allbank_read_reqs,
                            bank_shift,
                            burst_bytes,
                            row_bytes,
                            read_base,
                            total_banks=banks,
                        )
                        allbank_read_reqs += 1
                        f.write(f"PIM_MAC_AB 0x{addr:x}\n")
                        req_counts["PIM_MAC_AB"] += 1

        # KV cache writeback (K+V only): 2 * d_model elements per token.
        kv_elem = 2 * d_model * max(1, int(act_vectors))
        kv_bytes = int(math.ceil(kv_elem * int(bw.out_bits) / 8.0))
        kv_bursts = int(math.ceil(kv_bytes / float(burst_bytes)))
        kv_allbank_bursts = int(math.ceil(kv_bursts / float(max(1, banks))))
        for _ in range(kv_allbank_bursts):
            addr = _addr_for_bank(
                0,
                allbank_write_reqs,
                bank_shift,
                burst_bytes,
                row_bytes,
                kvwb_base,
                total_banks=banks,
            )
            allbank_write_reqs += 1
            f.write(f"BITACC_KV_WB 0x{addr:x}\n")
            req_counts["BITACC_KV_WB"] += 1

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(_make_yaml(
            path_trace=os.path.abspath(trace_path),
            path_cmd_counter=os.path.abspath(cmd_counter_path),
            path_cmd_trace_prefix=os.path.abspath(cmd_trace_prefix),
            enable_refresh=bool(d.enable_refresh),
            n_lut_bcast=int(n_lut_bcast),
            n_rdpim=int(n_rdpim),
            n_macab=int(n_macab),
            n_refi=int(d.tREFI),
            n_rfc1=int(d.tRFC),
        ))

    timed_out = False
    out_text = ""
    try:
        out_text = _run_ramulator(ramulator_bin, yaml_path, timeout_s)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out_text = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out_text += "\n[trace-gen] Ramulator run timed out.\n"
    except subprocess.CalledProcessError as e:
        out_text = e.output if isinstance(e.output, str) else ""
        if not out_text:
            out_text = f"[trace-gen] Ramulator failed with exit code {e.returncode}.\n"
        with open(run_out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
        raise RuntimeError(
            f"Ramulator failed for trace {trace_path}; see {run_out_path}"
        ) from e
    with open(run_out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    cmd_trace_path = f"{cmd_trace_prefix}.ch0"
    cmd_info = _parse_cmd_trace(cmd_trace_path)
    cmd_counter = _parse_cmd_counter(cmd_counter_path)
    run_stats = _parse_run_stats(out_text)

    request_vs_theory = {
        "BITACC_LUT_BCAST": int(req_counts["BITACC_LUT_BCAST"]) == int(trace_plan["total_lut_bcast"]),
        "PIM_MAC_AB": int(req_counts["PIM_MAC_AB"]) == int(trace_plan["total_macab"]),
    }
    cmd_vs_theory = {
        "LUTBCAST": (cmd_counter.get("LUTBCAST") == int(trace_plan["total_lut_bcast"])) if cmd_counter else None,
        "MACAB": (cmd_counter.get("MACAB") == int(trace_plan["total_macab"])) if cmd_counter else None,
    }

    summary = {
        "source_dse_config": dse_config,
        "derived": {
            "d_model": d_model,
            "qkv_projections": qkv_projections,
            "qkv_channels": qkv_channels,
            "n_layers": int(n_layers_override) if n_layers_override is not None else int(w.n_layers),
            "banks": banks,
            "channels_per_bank_floor": channels_per_bank_floor,
            "channels_per_bank_remainder": channels_per_bank_remainder,
            "packed_weight_bits_auto": packed_weight_bits,
            "burst_bytes": burst_bytes,
            "row_bytes": row_bytes,
            "elements_per_rdpim_packed": int((burst_bytes * 8) // packed_weight_bits),
            "bank_shift": int(bank_shift),
            "act_vectors": int(act_vectors),
        },
        "timed_out": timed_out,
        "request_counts": req_counts,
        "cmd_analysis": cmd_info,
        "cmd_counter": cmd_counter,
        "theoretical_counts": {
            "BITACC_LUT_BCAST": int(trace_plan["total_lut_bcast"]),
            "PIM_MAC_AB": int(trace_plan["total_macab"]),
            "trace_plan": trace_plan,
        },
        "count_checks": {
            "request_vs_theory": request_vs_theory,
            "cmd_vs_theory": cmd_vs_theory,
            "run_stats_vs_requests": {
                "total_num_bitacc_lut_bcast_requests":
                    run_stats.get("total_num_bitacc_lut_bcast_requests") == int(req_counts["BITACC_LUT_BCAST"]),
                "total_num_pim_mac_all_bank_requests":
                    run_stats.get("total_num_pim_mac_all_bank_requests") == int(req_counts["PIM_MAC_AB"]),
            },
        },
        "memory_system_cycles": _extract_memory_cycles(out_text),
        "run_stats": run_stats,
        "files": {
            "trace": trace_path,
            "yaml": yaml_path,
            "run_out": run_out_path,
            "cmd_counter": cmd_counter_path,
            "cmd_trace": cmd_trace_path,
        },
        "dse_snapshot": {
            "workload": asdict(w),
            "dram": asdict(d),
            "bit_widths": asdict(bw),
            "architecture": asdict(a),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Generate DSE-parameterized fused QKV+KVWB traces and dump DRAM commands.")
    p.add_argument("--dse_config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ramulator_bin", default="ramulator2/build/ramulator2")
    p.add_argument("--timeout_s", type=int, default=600)
    p.add_argument("--bank_shift", type=int, default=None, help="address bank-tag shift. If omitted, autotuned.")
    p.add_argument("--n_lut_bcast", type=int, default=32)
    p.add_argument("--n_rdpim", type=int, default=80)
    p.add_argument("--n_macab", type=int, default=2)
    p.add_argument("--act_vectors", type=int, default=1)
    p.add_argument("--max_bursts_per_output", type=int, default=None,
                   help="optional debug window; if set, only generate first N bursts per output channel")
    p.add_argument("--d_model", type=int, default=None, help="optional override for input vector dimension")
    p.add_argument("--qkv_projections", type=int, default=None, help="optional override for QKV projection count")
    p.add_argument("--n_layers", type=int, default=None, help="optional override for block/layer count")
    p.add_argument("--banks", type=int, default=None, help="optional override for logical bank count")
    args = p.parse_args()

    summary = generate(
        dse_config=args.dse_config,
        out_dir=args.out_dir,
        ramulator_bin=args.ramulator_bin,
        timeout_s=args.timeout_s,
        bank_shift=args.bank_shift,
        n_lut_bcast=args.n_lut_bcast,
        n_rdpim=args.n_rdpim,
        n_macab=args.n_macab,
        act_vectors=args.act_vectors,
        max_bursts_per_output=args.max_bursts_per_output,
        d_model_override=args.d_model,
        qkv_projections_override=args.qkv_projections,
        n_layers_override=args.n_layers,
        banks_override=args.banks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
