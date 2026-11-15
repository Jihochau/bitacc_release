from __future__ import annotations

import json
import math
import os
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, List

from .ramulator_ddr import ddr_rows_for_org

from .token_lifecycle_trace_gen import (
    _all_bank_addr_vec,
    _count_input_trace_tokens,
    _no_target_addr_vec,
    _resolve_bank_layout,
    _resolve_memory_topology,
    _row_bytes_for_org,
    _run_runtime_trace_lines,
    _runtime_burst_bytes,
)


LATEST_COMMANDS = (
    "EMBEDDING",
    "HOST_XFER",
    "CPU_NORM",
    "LM_HEAD",
    "TOPK",
    "TOKEN_DONE",
    "RMS_ATTN",
    "RMS_MLP",
    "QUANT",
    "MLP_FUSION",
    "FINAL_RES",
    "BLOCK_END",
    "ALL_BANK_LUT_LOAD",
    "ALL_BANK_BITLINEAR",
    "ALL_BANK_DEQUANT",
    "ALL_BANK_ROPE",
    "ALL_BANK_WR_K_CURRENT",
    "ALL_BANK_WR_V_CURRENT",
    "ALL_BANK_RD_K_V_CACHE",
    "ALL_BANK_QKT",
    "QKT_XFER",
    "QKT_REDUCE",
    "ALL_BANK_POST",
    "POST_SCORE_BCAST",
    "ALL_BANK_PV",
    "PV_XFER",
    "LINEAR_XFER",
)


class _LayeredTraceLines:
    """Re-iterable full-token trace backed by one transformer-block template."""

    def __init__(
        self,
        *,
        host_pre: List[str],
        block: List[str],
        host_post: List[str],
        layers: int,
        weight_rows_per_layer: int,
        kv_rows_per_layer: int,
    ) -> None:
        self.host_pre = tuple(host_pre)
        self.block = tuple(block)
        self.host_post = tuple(host_post)
        self.layers = int(layers)
        self.weight_rows_per_layer = int(weight_rows_per_layer)
        self.kv_rows_per_layer = int(kv_rows_per_layer)

    def _remap(self, line: str, layer_id: int) -> str:
        if layer_id == 0:
            return line
        fields = line.split()
        if len(fields) not in {7, 8}:
            raise ValueError(f"unexpected semantic trace line: {line}")
        row = int(fields[5])
        if row >= 0:
            command = fields[0]
            if command == "ALL_BANK_BITLINEAR":
                row += layer_id * self.weight_rows_per_layer
            elif command in {
                "ALL_BANK_WR_K_CURRENT",
                "ALL_BANK_WR_V_CURRENT",
                "ALL_BANK_RD_K_V_CACHE",
            }:
                row += layer_id * self.kv_rows_per_layer
            else:
                raise ValueError(f"unexpected concrete-address command: {command}")
            fields[5] = str(row)
        return " ".join(fields)

    def __iter__(self) -> Iterator[str]:
        yield from self.host_pre
        for layer_id in range(self.layers):
            for line in self.block:
                yield self._remap(line, layer_id)
        yield from self.host_post

LEGACY_COMMANDS = (
    "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE",
    "ALL_BANK_LUT_GEN_AND_LOAD",
    "ALL_BANK_DEQUANT_ROPE",
    "ALL_BANK_WRITE_K_CURRENT",
    "ALL_BANK_WRITE_V_CURRENT",
    "ALL_BANK_READ_K_CACHE",
    "ALL_BANK_READ_V_CACHE",
)


def _ceil_div(a: int, b: int) -> int:
    return (max(0, int(a)) + max(1, int(b)) - 1) // max(1, int(b))


def _cycles_at_controller(local_cycles: int, controller_hz: float, local_hz: float) -> int:
    return max(1, int(math.ceil(max(0, int(local_cycles)) * controller_hz / max(1.0, local_hz))))


def _cpu_clock_cycles(ops: int, cpu_clock_hz: float, cpu_ops_per_second: float) -> int:
    return max(1, int(math.ceil(max(0, int(ops)) * cpu_clock_hz / max(1.0, cpu_ops_per_second))))


def _require_positive(config: Dict[str, Any], key: str) -> int:
    value = int(config.get(key, 0) or 0)
    if value <= 0:
        raise ValueError(f"latest command scheme requires positive config value {key}")
    return value


def derive_latest_address_requirements(
    *,
    modelinfos: Dict[str, Any],
    pim_config: Dict[str, Any],
    lin: int,
    row_bytes: int,
) -> Dict[str, int]:
    h = int(modelinfos["hdim"])
    layers = int(modelinfos["ndec"])
    heads = int(modelinfos["num_heads"])
    head_dim = int(modelinfos.get("dhead", _ceil_div(h, heads)))
    mlp_dim = int(round(float(modelinfos["ff_scale"]) * h))
    chips = _require_positive(pim_config, "BITACC_CHIP_COUNT")
    banks = _require_positive(pim_config, "bank_count_per_chip")
    activation_bytes = _require_positive(pim_config, "activation_bytes")
    _require_positive(pim_config, "bitacc_weight_bits")
    pairs_per_group = _require_positive(pim_config, "lut_pairs_per_group")
    input_elements_per_pair = _require_positive(
        pim_config, "input_elements_per_lut_pair"
    )
    scalars_per_group = pairs_per_group * input_elements_per_pair
    bank_array_bits = _require_positive(pim_config, "bank_array_bits_per_pe_cycle")
    if bank_array_bits % 8:
        raise ValueError("bank_array_bits_per_pe_cycle must be byte aligned")
    bank_array_bytes = bank_array_bits // 8
    weight_base_row = _require_positive(pim_config, "weight_base_row")
    bank_row_bytes = _ceil_div(row_bytes, chips)
    groups_per_row = bank_row_bytes // bank_array_bytes
    if groups_per_row <= 0:
        raise ValueError("one Bank-PE array access is larger than one per-chip DRAM row")

    dhc = _ceil_div(head_dim, chips)
    head_waves = _ceil_div(heads, banks)
    h_groups = _ceil_div(h, scalars_per_group)
    m_groups = _ceil_div(mlp_dim, scalars_per_group)
    qkv_local_columns = 3 * dhc
    output_local_columns = _ceil_div(h, banks * chips)
    mlp_local_columns = _ceil_div(mlp_dim, banks * chips)
    qkv_accesses = h_groups * qkv_local_columns
    output_accesses = h_groups * output_local_columns
    gate_accesses = h_groups * mlp_local_columns
    down_accesses = m_groups * output_local_columns
    qkv_rows = head_waves * _ceil_div(qkv_accesses, groups_per_row)
    output_rows = _ceil_div(output_accesses, groups_per_row)
    gate_rows = _ceil_div(gate_accesses, groups_per_row)
    down_rows = _ceil_div(down_accesses, groups_per_row)

    weight_rows_per_layer = qkv_rows + output_rows + 2 * gate_rows + down_rows
    kv_slice_payload_bytes = dhc * activation_bytes
    kv_slice_storage_bytes = _ceil_div(kv_slice_payload_bytes, bank_array_bytes) * bank_array_bytes
    kv_pair_bytes = 2 * kv_slice_storage_bytes
    kv_allocated_rows_per_wave = _ceil_div((int(lin) + 1) * kv_pair_bytes, bank_row_bytes)
    kv_rows_per_layer = head_waves * kv_allocated_rows_per_wave
    kv_cache_base_row = weight_base_row + layers * weight_rows_per_layer
    required_max_row = kv_cache_base_row + layers * kv_rows_per_layer - 1
    return {
        "bank_row_bytes": bank_row_bytes,
        "groups_per_weight_row": groups_per_row,
        "lut_pairs_per_group": pairs_per_group,
        "input_elements_per_lut_pair": input_elements_per_pair,
        "input_scalars_per_lut_group": scalars_per_group,
        "h_lut_groups": h_groups,
        "m_lut_groups": m_groups,
        "head_waves": head_waves,
        "Dhc": dhc,
        "qkv_local_columns": qkv_local_columns,
        "output_local_columns": output_local_columns,
        "mlp_local_columns": mlp_local_columns,
        "qkv_rows_per_layer": qkv_rows,
        "output_rows_per_layer": output_rows,
        "gate_rows_per_layer": gate_rows,
        "down_rows_per_layer": down_rows,
        "weight_rows_per_layer": weight_rows_per_layer,
        "kv_allocated_rows_per_wave": kv_allocated_rows_per_wave,
        "kv_rows_per_layer": kv_rows_per_layer,
        "kv_cache_base_row": kv_cache_base_row,
        "required_max_row": required_max_row,
    }


def build_latest_command_plan(
    *,
    modelinfos: Dict[str, Any],
    pim_config: Dict[str, Any],
    lin: int,
    lout: int,
    burst_bytes: int,
    row_bytes: int,
    bank_layout: Dict[str, int],
    topology: Dict[str, int],
    compact_layers: bool = False,
    allow_hardware_scaling: bool = False,
) -> Dict[str, Any]:
    if int(lout) != 1:
        raise ValueError(
            f"latest decode trace requires lout=1, got {int(lout)}; generate one token at a time"
        )
    if int(lin) <= 0:
        raise ValueError(f"latest decode trace requires lin>0 cached KV tokens, got {int(lin)}")
    h = int(modelinfos["hdim"])
    layers = int(modelinfos["ndec"])
    heads = int(modelinfos["num_heads"])
    head_dim = int(modelinfos.get("dhead", _ceil_div(h, heads)))
    mlp_dim = int(round(float(modelinfos["ff_scale"]) * h))
    vocab = int(modelinfos["vocab_size"])
    chips = _require_positive(pim_config, "BITACC_CHIP_COUNT")
    banks_per_chip = _require_positive(pim_config, "bank_count_per_chip")
    activation_bytes = _require_positive(pim_config, "activation_bytes")
    weight_bits = _require_positive(pim_config, "bitacc_weight_bits")
    quant_lanes = _require_positive(pim_config, "quant_lanes")
    residual_lanes = _require_positive(pim_config, "residual_lanes")
    post_pass_count = _require_positive(pim_config, "post_pass_count")
    post_token_parallelism = _require_positive(pim_config, "post_token_parallelism")
    post_overhead = int(pim_config.get("post_overhead", 0) or 0)
    lut_pairs_per_group = _require_positive(pim_config, "lut_pairs_per_group")
    input_elements_per_lut_pair = _require_positive(
        pim_config, "input_elements_per_lut_pair"
    )
    scalars_per_lut_tile = lut_pairs_per_group * input_elements_per_lut_pair
    configured_scalars_per_tile = _require_positive(
        pim_config, "input_scalars_per_lut_tile"
    )
    if configured_scalars_per_tile != scalars_per_lut_tile:
        raise ValueError(
            "input_scalars_per_lut_tile must equal "
            "lut_pairs_per_group * input_elements_per_lut_pair"
        )
    lut_pairs_per_dimm_clock = _require_positive(pim_config, "lut_pairs_per_dimm_clock")
    lut_elements_per_pair = _require_positive(pim_config, "lut_elements_per_pair")
    lut_entry_bits = _require_positive(pim_config, "bitacc_lut_entry_bits")
    rank_io_bits_per_clock = _require_positive(pim_config, "rank_io_bits_per_clock")
    controller_io_bits_per_edge = _require_positive(
        pim_config, "controller_io_bits_per_edge"
    )
    if rank_io_bits_per_clock != 2 * controller_io_bits_per_edge:
        raise ValueError(
            "rank_io_bits_per_clock must equal two DDR edges per controller clock"
        )
    dq_bits_per_chip = _require_positive(pim_config, "dq_bits_per_chip")
    dram_data_rate_mtps = _require_positive(pim_config, "dram_data_rate_mtps")
    if (
        not allow_hardware_scaling
        and (chips != 8 or banks_per_chip != 32 or dq_bits_per_chip != 8)
    ):
        raise ValueError(
            "latest command scheme is fixed to one rank with 8 x8 chips and 32 banks/chip"
        )
    if controller_io_bits_per_edge != chips * dq_bits_per_chip:
        raise ValueError(
            "controller_io_bits_per_edge must equal chip_count * dq_bits_per_chip"
        )
    if not bool(pim_config.get("merged_command_trace", False)):
        raise ValueError("latest command scheme requires merged_command_trace=true")
    bitlinear_cycles_per_group = _require_positive(pim_config, "bitlinear_cycles_per_group")
    bitlinear_weight_bytes = _require_positive(pim_config, "bitlinear_weight_bytes_per_group")
    bank_array_bits_per_cycle = _require_positive(
        pim_config, "bank_array_bits_per_pe_cycle"
    )
    if bank_array_bits_per_cycle % 8:
        raise ValueError("bank_array_bits_per_pe_cycle must be byte aligned")
    bank_array_bytes_per_cycle = bank_array_bits_per_cycle // 8
    pair_weight_bits = weight_bits * input_elements_per_lut_pair
    if bank_array_bits_per_cycle != lut_pairs_per_group * pair_weight_bits:
        raise ValueError(
            "bank_array_bits_per_pe_cycle must equal lut_pairs_per_group times "
            "the physical bits per pair cell (weight_bits * input_elements_per_lut_pair)"
        )
    if bitlinear_weight_bytes % bank_array_bytes_per_cycle:
        raise ValueError(
            "bitlinear_weight_bytes_per_group must be divisible by the bank-array access width"
        )
    bitlinear_fetches_per_group = bitlinear_weight_bytes // bank_array_bytes_per_cycle
    if bitlinear_fetches_per_group != 1:
        raise ValueError(
            "latest command scheme requires one Bank-PE array access per "
            "ALL_BANK_BITLINEAR"
        )
    scheme_version = (
        "v14"
        if lut_pairs_per_group == 32 and bank_array_bits_per_cycle == 128
        else "v13"
        if lut_pairs_per_group == 16 and bank_array_bits_per_cycle == 128
        else "v12"
        if lut_pairs_per_group == 8 and bank_array_bits_per_cycle == 64
        else "custom"
    )
    command_scheme = (
        f"latest_cmd_{scheme_version}_{lut_pairs_per_group}pair_merged_duration"
    )
    kv_internal_bank_cycles = _require_positive(pim_config, "kv_internal_bank_cycles")
    weight_base_row = _require_positive(pim_config, "weight_base_row")
    bank_pe_clock_hz = float(_require_positive(pim_config, "bank_pe_clock_hz"))
    dimm_pe_clock_hz = float(_require_positive(pim_config, "dimm_pe_clock_hz"))
    controller_hz = float(pim_config.get("BITACC_TRACE_CTRL_FREQ_MHZ", 1600.0)) * 1_000_000.0
    expected_controller_hz = dram_data_rate_mtps * 1_000_000.0 / 2.0
    if not math.isclose(controller_hz, expected_controller_hz, rel_tol=0.0, abs_tol=1.0):
        raise ValueError(
            "controller clock must be half the configured DDR transfer rate"
        )
    cpu_clock_hz = float(_require_positive(pim_config, "BITACC_CPU_CLOCK_HZ"))
    cpu_ops_per_second = float(_require_positive(pim_config, "BITACC_CPU_ARITHMETIC_OPS_PER_SECOND"))
    cpu_rsqrt_ops = int(pim_config.get("cpu_rsqrt_ops", 0) or 0)
    topk_ops = _require_positive(pim_config, "topk_ops")
    n_tile = max(1, int(pim_config.get("BITACC_PE_N_TILE", 1)))
    if n_tile != 1:
        raise ValueError("fixed Bank-PE datapath requires BITACC_PE_N_TILE=1")
    qkt_tok_par = max(1, int(pim_config.get("BITACC_TRACE_QKT_TOK_PAR", 1)))
    qkt_dot_par = max(1, int(pim_config.get("BITACC_TRACE_QKT_DOT_PAR", 1)))
    tp_par = max(1, int(pim_config.get("BITACC_TRACE_TP_PAR", 1)))
    pv_dim_par = max(1, int(pim_config.get("BITACC_TRACE_PV_DIM_PAR", 1)))
    qkt_ctrl = max(0, int(pim_config.get("BITACC_TRACE_QKT_CTRL_PE_CYCLES", 0)))
    pv_ctrl = max(0, int(pim_config.get("BITACC_TRACE_PV_CTRL_PE_CYCLES", 0)))
    rope_ctrl = max(0, int(pim_config.get("BITACC_TRACE_ROPE_CTRL_PE_CYCLES", 0)))
    dequant_cycles = max(0, int(pim_config.get("BITACC_TRACE_DEQUANT_PE_CYCLES", 2)))
    lut_control_dimm_cycles = max(
        0, int(pim_config.get("lut_control_dimm_cycles", 1) or 0)
    )
    qkt_reduce_additions = max(
        0, int(pim_config.get("qkt_reduce_additions_per_wave", chips - 1) or 0)
    )
    if qkt_reduce_additions != chips - 1:
        raise ValueError(
            "qkt_reduce_additions_per_wave must equal chip_count - 1 for chip-partial reduction"
        )

    # The trace represents the first decode output token: S_kv=lin and S=S_kv+1.
    s_kv = max(0, int(lin))
    seq_len = s_kv + 1
    dhc = _ceil_div(head_dim, chips)
    head_waves = _ceil_div(heads, banks_per_chip)
    heads_per_wave = [
        min(banks_per_chip, heads - wave * banks_per_chip)
        for wave in range(head_waves)
    ]
    kv_values_per_bank_chip = dhc
    kv_slice_bytes_per_chip = kv_values_per_bank_chip * activation_bytes
    bank_row_bytes = _ceil_div(row_bytes, chips)
    groups_per_weight_row = bank_row_bytes // bank_array_bytes_per_cycle
    if groups_per_weight_row <= 0:
        raise ValueError("BitLinear weight group is larger than one per-chip DRAM row")

    weight_row_cursor = 0

    def reserve_rows(group_count: int) -> List[int]:
        nonlocal weight_row_cursor
        count = _ceil_div(group_count, groups_per_weight_row)
        offsets = list(range(weight_row_cursor, weight_row_cursor + count))
        weight_row_cursor += count
        return offsets

    qkv_tiles = _ceil_div(h, scalars_per_lut_tile)
    qkv_groups_per_tile_per_wave = 3 * dhc
    qkv_wave_projections: List[List[Dict[str, Any]]] = []
    for head_wave, active_heads in enumerate(heads_per_wave):
        qkv_wave_projections.append(
            [
                {
                    "name": f"qkv_wave_{head_wave}",
                    "local_output_cols": qkv_groups_per_tile_per_wave,
                    "row_offsets": reserve_rows(
                        qkv_tiles * qkv_groups_per_tile_per_wave
                    ),
                    "head_wave": head_wave,
                    "active_banks": active_heads,
                }
            ]
        )

    output_tiles = _ceil_div(h, scalars_per_lut_tile)
    output_groups_per_tile = _ceil_div(h, banks_per_chip * chips)
    output_projection = [{
        "name": "o",
        "local_output_cols": output_groups_per_tile,
        "row_offsets": reserve_rows(output_tiles * output_groups_per_tile),
    }]

    mlp_groups_per_tile = _ceil_div(mlp_dim, banks_per_chip * chips)
    mlp_tiles = _ceil_div(h, scalars_per_lut_tile)
    gate_up_projections = [
        {
            "name": "gate",
            "local_output_cols": mlp_groups_per_tile,
            "row_offsets": reserve_rows(mlp_tiles * mlp_groups_per_tile),
        },
        {
            "name": "up",
            "local_output_cols": mlp_groups_per_tile,
            "row_offsets": reserve_rows(mlp_tiles * mlp_groups_per_tile),
        },
    ]

    down_tiles = _ceil_div(mlp_dim, scalars_per_lut_tile)
    down_groups_per_tile = _ceil_div(h, banks_per_chip * chips)
    down_projection = [{
        "name": "down",
        "local_output_cols": down_groups_per_tile,
        "row_offsets": reserve_rows(down_tiles * down_groups_per_tile),
    }]
    weight_rows_per_layer = weight_row_cursor
    kv_cache_base_row = weight_base_row + layers * weight_rows_per_layer
    physical_banks = int(bank_layout["total_banks"])
    if banks_per_chip != physical_banks:
        raise ValueError(
            f"bank_count_per_chip={banks_per_chip} must match Ramulator physical banks={physical_banks} "
            "for ALL_BANK command fanout"
        )
    if int(topology.get("channels", 0)) != 1 or int(topology.get("ranks", 0)) != 1:
        raise ValueError("latest command scheme requires exactly one memory channel/DIMM and one rank")

    rank_addr = _no_target_addr_vec(topology)
    bitlinear_mem_cycles = _cycles_at_controller(
        bitlinear_cycles_per_group, controller_hz, bank_pe_clock_hz
    )
    dimm_unit_cycles = _cycles_at_controller(1, controller_hz, dimm_pe_clock_hz)
    bank_unit_cycles = _cycles_at_controller(1, controller_hz, bank_pe_clock_hz)
    kv_command_cycles = _cycles_at_controller(
        kv_internal_bank_cycles, controller_hz, bank_pe_clock_hz
    )

    lines: List[str] = []
    events: List[Dict[str, Any]] = []
    command_counts: Counter[str] = Counter()
    token_timings: Dict[str, int] = {}
    timeline_cycle = 1

    def emit(
        name: str,
        count: int,
        memory_cycles_each: int,
        *,
        scope: str,
        domain: str,
        formula: str,
        addresses: Iterable[List[int]] | None = None,
        details: Dict[str, Any] | None = None,
        allow_zero_cycles: bool = False,
        dynamic_duration: bool = False,
    ) -> None:
        nonlocal timeline_cycle
        count = max(0, int(count))
        memory_cycles_each = int(memory_cycles_each)
        if memory_cycles_each < 0 or (memory_cycles_each == 0 and not allow_zero_cycles):
            raise ValueError(f"invalid timing for {name}: {memory_cycles_each}")
        if count <= 0:
            return
        addr_iter = iter(addresses) if addresses is not None else None
        event_start = timeline_cycle
        for _ in range(count):
            addr_vec = rank_addr if addr_iter is None else next(addr_iter)
            duration_suffix = f" {memory_cycles_each}" if dynamic_duration else ""
            lines.append(
                f"{name} {' '.join(str(int(v)) for v in addr_vec)}{duration_suffix}"
            )
        command_counts[name] += count
        ramulator_timing_each = 1 if dynamic_duration else memory_cycles_each
        previous_timing = token_timings.get(name)
        if previous_timing is not None and previous_timing != ramulator_timing_each:
            raise ValueError(
                f"Ramulator timing for {name} must be invariant, got "
                f"{previous_timing} and {ramulator_timing_each}"
            )
        token_timings[name] = ramulator_timing_each
        total_cycles = count * memory_cycles_each
        events.append(
            {
                "command_id": len(events),
                "command": name,
                "scope": scope,
                "domain": domain,
                "formula": formula,
                "formula_terms": details or {},
                "start_cycle": event_start,
                "end_cycle": event_start + max(0, total_cycles - 1),
                "cycles": total_cycles,
                "logic_cycles": total_cycles,
                "memory_cycles": total_cycles,
                "pe_cycles": total_cycles,
                "trace_tokens": [
                    {
                        "name": name,
                        "count": count,
                        "domain": domain,
                        "pe_cycles_each": memory_cycles_each,
                        "memory_cycles_each": memory_cycles_each,
                        "ramulator_timing_each": ramulator_timing_each,
                        "dynamic_duration": dynamic_duration,
                        "occupancy_cycles_each": (
                            memory_cycles_each if dynamic_duration else 0
                        ),
                        "pe_cycles_total": total_cycles,
                        "memory_cycles_total": total_cycles,
                    }
                ],
            }
        )
        timeline_cycle += total_cycles

    def emit_merged(
        name: str,
        duration_cycles: int,
        *,
        scope: str,
        domain: str,
        formula: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"merged_trace_command": True, "duration_cycles": duration_cycles}
        merged_details.update(details or {})
        emit(
            name,
            1,
            duration_cycles,
            scope=scope,
            domain=domain,
            formula=formula,
            details=merged_details,
            dynamic_duration=True,
        )

    def cpu_cycles(ops: int) -> tuple[int, int]:
        local = _cpu_clock_cycles(ops, cpu_clock_hz, cpu_ops_per_second)
        return local, _cycles_at_controller(local, controller_hz, cpu_clock_hz)

    def emit_dimm_beats(name: str, raw_cycles: int, *, scope: str, formula: str, details: Dict[str, Any]) -> None:
        visible_cycles = _cycles_at_controller(
            raw_cycles, controller_hz, dimm_pe_clock_hz
        )
        emit_merged(
            name,
            visible_cycles,
            scope=scope,
            domain="dimm",
            formula=formula,
            details={
                **details,
                "raw_dimm_cycles": raw_cycles,
                "visible_controller_cycles": visible_cycles,
            },
        )

    def emit_transfer(
        name: str,
        elems: int,
        *,
        scope: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        clocks = _ceil_div(elems * activation_bytes * 8, rank_io_bits_per_clock)
        transfer_details = {
            "elements": elems,
            "activation_bytes": activation_bytes,
            "rank_io_bits_per_clock": rank_io_bits_per_clock,
            "transfer_clocks": clocks,
            "serialized_bits": elems * activation_bytes * 8,
            "payload_bytes": elems * activation_bytes,
        }
        transfer_details.update(details or {})
        emit_merged(
            name,
            clocks,
            scope=scope,
            domain="rank_io",
            formula="ceil(elements * activation_bytes * 8 / rank_io_bits_per_clock)",
            details=transfer_details,
        )

    def emit_banked_transfer(
        name: str,
        elems: int,
        *,
        scope: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Serialize all Bank-PE slices over DQ, represented by one trace command."""

        elements_per_bank = _ceil_div(elems, banks_per_chip)
        clocks_per_bank = _ceil_div(
            elements_per_bank * activation_bytes * 8,
            rank_io_bits_per_clock,
        )
        clocks = banks_per_chip * clocks_per_bank
        serialized_bits = (
            banks_per_chip * elements_per_bank * activation_bytes * 8
        )
        transfer_details = {
            "elements": elems,
            "activation_bytes": activation_bytes,
            "bank_count": banks_per_chip,
            "elements_per_bank": elements_per_bank,
            "elements_per_bank_per_chip": _ceil_div(elements_per_bank, chips),
            "rank_io_bits_per_clock": rank_io_bits_per_clock,
            "controller_io_bits_per_edge": controller_io_bits_per_edge,
            "transfer_clocks_per_bank": clocks_per_bank,
            "transfer_clocks": clocks,
            "serialized_bits": serialized_bits,
            "payload_bytes": serialized_bits // 8,
            "bank_transfer_policy": "bank_major_equal_slices",
        }
        transfer_details.update(details or {})
        emit_merged(
            name,
            clocks,
            scope=scope,
            domain="rank_io",
            formula=(
                "B * ceil(ceil(elements/B) * activation_bytes * 8 "
                "/ rank_io_bits_per_clock)"
            ),
            details=transfer_details,
        )

    def emit_linear(
        *,
        stage: str,
        input_dim: int,
        projections: List[Dict[str, Any]],
        stage_details: Dict[str, Any] | None = None,
    ) -> None:
        tiles = _ceil_div(input_dim, scalars_per_lut_tile)
        lut_pairs = lut_pairs_per_group
        lut_gen_dimm_clocks = _ceil_div(lut_pairs, lut_pairs_per_dimm_clock)
        lut_gen_controller_clocks = _cycles_at_controller(
            lut_gen_dimm_clocks, controller_hz, dimm_pe_clock_hz
        )
        lut_bits_per_pair = lut_elements_per_pair * lut_entry_bits
        lut_xfer_clocks = _ceil_div(
            lut_pairs * lut_bits_per_pair * chips, rank_io_bits_per_clock
        )
        lut_pipeline_clocks = max(lut_gen_controller_clocks, lut_xfer_clocks)
        lut_control_controller_clocks = (
            0
            if lut_control_dimm_cycles == 0
            else _cycles_at_controller(
                lut_control_dimm_cycles, controller_hz, dimm_pe_clock_hz
            )
        )
        lut_load_cycles = lut_pipeline_clocks + lut_control_controller_clocks
        projection_ordinals = {str(projection["name"]): 0 for projection in projections}
        groups_per_physical_row = bank_row_bytes // bank_array_bytes_per_cycle
        if groups_per_physical_row <= 0:
            raise ValueError("Bank-array access is larger than one per-chip DRAM row")
        for tile_id in range(tiles):
            emit_merged(
                "ALL_BANK_LUT_LOAD",
                lut_load_cycles,
                scope="block",
                domain="rank_io",
                formula=(
                    "max(lut_generation_controller_clocks, lut_transfer_clocks) "
                    "+ lut_control_controller_clocks"
                ),
                details={
                    "stage": stage,
                    "tile_id": tile_id,
                    "lut_pairs": lut_pairs,
                    "input_elements_per_lut_pair": input_elements_per_lut_pair,
                    "input_scalars_per_lut_tile": scalars_per_lut_tile,
                    "lut_elements_per_pair": lut_elements_per_pair,
                    "lut_entry_bits": lut_entry_bits,
                    "lut_bits_per_pair": lut_bits_per_pair,
                    "lut_payload_bits_per_chip": lut_pairs * lut_bits_per_pair,
                    "lut_payload_bits_rank": lut_pairs * lut_bits_per_pair * chips,
                    "lut_payload_bytes_rank": lut_pairs * lut_bits_per_pair * chips // 8,
                    "lut_gen_dimm_clocks": lut_gen_dimm_clocks,
                    "lut_gen_controller_clocks": lut_gen_controller_clocks,
                    "lut_xfer_clocks": lut_xfer_clocks,
                    "lut_pipeline_clocks": lut_pipeline_clocks,
                    "lut_control_dimm_clocks": lut_control_dimm_cycles,
                    "lut_control_controller_clocks": lut_control_controller_clocks,
                    "transfer_clocks": lut_xfer_clocks,
                    "serialized_bits": lut_pairs * lut_bits_per_pair * chips,
                    "active_banks": banks_per_chip,
                    **(stage_details or {}),
                },
            )
            for projection in projections:
                name = str(projection["name"])
                tile_ids = [int(value) for value in projection.get("tile_ids", range(tiles))]
                if tile_id not in tile_ids:
                    continue
                local_output_cols = int(projection["local_output_cols"])
                row_offsets = [int(value) for value in projection["row_offsets"]]
                groups = local_output_cols
                total_groups = len(tile_ids) * groups
                required_rows = _ceil_div(total_groups, groups_per_physical_row)
                if required_rows > len(row_offsets):
                    raise ValueError(
                        f"{name} requires {required_rows} rows but only {len(row_offsets)} are reserved"
                    )
                addresses: List[List[int]] = []
                for _ in range(groups):
                    ordinal = projection_ordinals[name]
                    projection_ordinals[name] += 1
                    row_slot = ordinal // groups_per_physical_row
                    col_group = ordinal % groups_per_physical_row
                    row = weight_base_row + row_offsets[row_slot]
                    col = col_group * bank_array_bytes_per_cycle
                    addresses.append(_all_bank_addr_vec(row, col, topology))
                physical_fetches = groups
                emit(
                    "ALL_BANK_BITLINEAR",
                    physical_fetches,
                    bitlinear_mem_cycles,
                    scope="block",
                    domain="bank",
                    formula=(
                        "active_input_groups * local_output_columns"
                    ),
                    addresses=addresses,
                    details={
                        "stage": stage,
                        "projection": name,
                        "tile_id": tile_id,
                        "input_dim": input_dim,
                        "active_tiles": len(tile_ids),
                        "local_output_cols": local_output_cols,
                        "groups_per_tile": groups,
                        "physical_fetches_per_tile": physical_fetches,
                        "fetches_per_logical_group": 1,
                        "array_access_bits": bank_array_bits_per_cycle,
                        "array_access_bytes": bank_array_bytes_per_cycle,
                        "reserved_row_offsets": row_offsets,
                        "groups_per_physical_row": groups_per_physical_row,
                        "weight_bytes_per_group": bank_array_bytes_per_cycle,
                        "active_banks": banks_per_chip,
                        **(stage_details or {}),
                    },
                )

    def emit_dequant(stage: str, details: Dict[str, Any] | None = None) -> None:
        dequant_details = {"stage": stage, "raw_bank_cycles": dequant_cycles}
        dequant_details.update(details or {})
        emit(
            "ALL_BANK_DEQUANT",
            1,
            _cycles_at_controller(dequant_cycles, controller_hz, bank_pe_clock_hz),
            scope="block",
            domain="bank",
            formula="dequant_pipeline_cycles",
            details=dequant_details,
        )

    embedding_ops = 2 * h
    embedding_cpu, embedding_ctrl = cpu_cycles(embedding_ops)
    emit(
        "EMBEDDING",
        1,
        embedding_ctrl,
        scope="host_pre",
        domain="cpu",
        formula="CPU_CLK(2H)",
        details={"ops": embedding_ops, "cpu_cycles": embedding_cpu},
    )
    emit_transfer("HOST_XFER", h, scope="host_pre")
    block_line_start = len(lines)
    block_event_start = len(events)

    emit(
        "RMS_ATTN",
        1,
        _cycles_at_controller(2 * h + 2, controller_hz, dimm_pe_clock_hz),
        scope="block",
        domain="dimm",
        formula="2H + 2",
        details={"H": h, "raw_dimm_cycles": 2 * h + 2},
    )
    quant_h = 2 * _ceil_div(h, quant_lanes) + 1
    emit_dimm_beats(
        "QUANT",
        quant_h,
        scope="block",
        formula="2*ceil(H/quant_lanes)+1",
        details={"X": h, "quant_lanes": quant_lanes, "raw_dimm_cycles": quant_h},
    )
    weights_end_row = weight_base_row + layers * weight_rows_per_layer
    if kv_cache_base_row < weights_end_row:
        raise ValueError(
            f"kv_cache_base_row={kv_cache_base_row} overlaps weights ending at row "
            f"{weights_end_row - 1}"
        )
    kv_slice_storage_bytes = (
        _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle)
        * bank_array_bytes_per_cycle
    )
    kv_pair_bytes = 2 * kv_slice_storage_bytes
    kv_history_bytes_per_wave = s_kv * kv_pair_bytes
    kv_history_rows_per_wave = _ceil_div(kv_history_bytes_per_wave, bank_row_bytes)
    kv_allocated_rows_per_wave = _ceil_div(seq_len * kv_pair_bytes, bank_row_bytes)
    kv_read_rows_per_layer = head_waves * kv_history_rows_per_wave
    kv_read_accesses_per_wave = _ceil_div(
        kv_history_bytes_per_wave, bank_array_bytes_per_cycle
    )
    kv_read_accesses_per_layer = head_waves * kv_read_accesses_per_wave
    kv_rows_per_layer = head_waves * kv_allocated_rows_per_wave
    address_requirements = derive_latest_address_requirements(
        modelinfos=modelinfos,
        pim_config=pim_config,
        lin=lin,
        row_bytes=row_bytes,
    )
    if int(address_requirements["weight_rows_per_layer"]) != weight_rows_per_layer:
        raise ValueError("runtime address derivation disagrees with weight row allocation")
    if int(address_requirements["kv_rows_per_layer"]) != kv_rows_per_layer:
        raise ValueError("runtime address derivation disagrees with KV row allocation")
    qkt_xfer_beats_per_wave = [
        _ceil_div(active_heads * seq_len * chips * activation_bytes * 8, rank_io_bits_per_clock)
        for active_heads in heads_per_wave
    ]
    pv_xfer_beats_per_wave = [
        _ceil_div(active_heads * head_dim * activation_bytes * 8, rank_io_bits_per_clock)
        for active_heads in heads_per_wave
    ]
    post_score_bcast_beats_per_wave = [
        _ceil_div(
            active_heads * seq_len * chips * activation_bytes * 8,
            rank_io_bits_per_clock,
        )
        for active_heads in heads_per_wave
    ]

    def kv_current_addr_vecs(
        head_wave: int, value_offset: int
    ) -> Iterable[List[int]]:
        wave_base = kv_cache_base_row + head_wave * kv_allocated_rows_per_wave
        value_base = (seq_len - 1) * kv_pair_bytes + value_offset
        for local_offset in range(
            0, kv_slice_bytes_per_chip, bank_array_bytes_per_cycle
        ):
            byte_offset = value_base + local_offset
            yield _all_bank_addr_vec(
                wave_base + byte_offset // bank_row_bytes,
                byte_offset % bank_row_bytes,
                topology,
            )

    def kv_access_addr_vecs(head_wave: int) -> Iterable[List[int]]:
        wave_base = kv_cache_base_row + head_wave * kv_allocated_rows_per_wave
        for byte_offset in range(
            0, kv_history_bytes_per_wave, bank_array_bytes_per_cycle
        ):
            yield _all_bank_addr_vec(
                wave_base + byte_offset // bank_row_bytes,
                byte_offset % bank_row_bytes,
                topology,
            )

    kv_head_waves_per_layer = head_waves
    current_token_id = seq_len - 1
    rope_raw = _ceil_div(dhc, 2) + rope_ctrl
    qkt_raw = _ceil_div(seq_len, qkt_tok_par) * _ceil_div(dhc, qkt_dot_par) + qkt_ctrl
    post_raw = post_pass_count * _ceil_div(seq_len, post_token_parallelism) + post_overhead
    pv_raw = _ceil_div(dhc, tp_par) * _ceil_div(seq_len, pv_dim_par) + pv_ctrl
    qkt_reduce_cycles = _cycles_at_controller(
        qkt_reduce_additions, controller_hz, dimm_pe_clock_hz
    )

    for head_wave, active_heads in enumerate(heads_per_wave):
        wave_details = {"head_wave": head_wave, "active_banks": active_heads}
        emit_linear(
            stage="qkv",
            input_dim=h,
            projections=qkv_wave_projections[head_wave],
            stage_details=wave_details,
        )
        emit_dequant("qkv", wave_details)
        emit(
            "ALL_BANK_ROPE",
            1,
            _cycles_at_controller(rope_raw, controller_hz, bank_pe_clock_hz),
            scope="block",
            domain="bank",
            formula="ceil(Dhc/2) + rope_ctrl",
            details={
                "Dhc": dhc,
                "rope_ctrl": rope_ctrl,
                "raw_bank_cycles": rope_raw,
                **wave_details,
            },
        )
        emit(
            "ALL_BANK_WR_K_CURRENT",
            _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle),
            kv_command_cycles,
            scope="block",
            domain="bank",
            formula="one head wave; active banks write their local K chip slice",
            addresses=kv_current_addr_vecs(head_wave, 0),
            details={
                "token_id": current_token_id,
                "row_base": kv_cache_base_row,
                "value_offset": 0,
                "array_access_bits": bank_array_bits_per_cycle,
                "array_access_bytes": bank_array_bytes_per_cycle,
                **wave_details,
            },
        )
        emit(
            "ALL_BANK_WR_V_CURRENT",
            _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle),
            kv_command_cycles,
            scope="block",
            domain="bank",
            formula="one head wave; active banks write their local V chip slice",
            addresses=kv_current_addr_vecs(head_wave, kv_slice_storage_bytes),
            details={
                "token_id": current_token_id,
                "row_base": kv_cache_base_row,
                "value_offset": kv_slice_storage_bytes,
                "array_access_bits": bank_array_bits_per_cycle,
                "array_access_bytes": bank_array_bytes_per_cycle,
                **wave_details,
            },
        )
        emit(
            "ALL_BANK_RD_K_V_CACHE",
            kv_read_accesses_per_wave,
            kv_command_cycles,
            scope="block",
            domain="bank",
            formula=(
                "S_kv * 2 * ceil(Dhc * activation_bytes "
                "/ bank_array_bytes_per_cycle) per wave"
            ),
            addresses=kv_access_addr_vecs(head_wave),
            details={
                "token_first": 0,
                "token_last": s_kv - 1,
                "row_base": kv_cache_base_row,
                "kv_pair_bytes": kv_pair_bytes,
                "kv_history_bytes_per_wave": kv_history_bytes_per_wave,
                "kv_history_rows_per_wave": kv_history_rows_per_wave,
                "kv_allocated_rows_per_wave": kv_allocated_rows_per_wave,
                "kv_read_accesses_per_wave": kv_read_accesses_per_wave,
                "array_access_bits": bank_array_bits_per_cycle,
                "array_access_bytes": bank_array_bytes_per_cycle,
                "bank_row_bytes": bank_row_bytes,
                **wave_details,
            },
        )
        emit(
            "ALL_BANK_QKT",
            1,
            _cycles_at_controller(qkt_raw, controller_hz, bank_pe_clock_hz),
            scope="block",
            domain="bank",
            formula="ceil(S/qkt_tok_par)*ceil(Dhc/qkt_dot_par)+qkt_ctrl",
            details={
                "S": seq_len,
                "Dhc": dhc,
                "raw_bank_cycles": qkt_raw,
                **wave_details,
            },
        )
        emit_transfer(
            "QKT_XFER",
            active_heads * seq_len * chips,
            scope="block",
            details={
                "head_wave": head_wave,
                "active_heads": active_heads,
                "chip_partials_per_score": chips,
                "operation": "transfer all chip partial scores to the DIMM PE",
            },
        )
        emit_merged(
            "QKT_REDUCE",
            qkt_reduce_cycles,
            scope="block",
            domain="dimm",
            formula="ceil((C-1) * controller_hz / dimm_pe_clock_hz)",
            details={
                "head_wave": head_wave,
                "active_heads": active_heads,
                "S": seq_len,
                "chip_count": chips,
                "raw_dimm_cycles": qkt_reduce_additions,
                "reduce_additions_per_head": qkt_reduce_additions,
                "parallel_adders": banks_per_chip,
                "logic_additions": active_heads * seq_len * qkt_reduce_additions,
            },
        )
        emit(
            "ALL_BANK_POST",
            1,
            _cycles_at_controller(post_raw, controller_hz, dimm_pe_clock_hz),
            scope="block",
            domain="dimm",
            formula="post_pass_count*ceil(S/post_token_parallelism)+post_overhead",
            details={
                "head_wave": head_wave,
                "active_heads": active_heads,
                "S": seq_len,
                "post_pass_count": post_pass_count,
                "post_token_parallelism": post_token_parallelism,
                "post_overhead": post_overhead,
                "T_post_raw": post_raw,
                "timing_policy": "conservative_raw",
            },
        )
        emit_transfer(
            "POST_SCORE_BCAST",
            active_heads * seq_len * chips,
            scope="block",
            details={
                "head_wave": head_wave,
                "active_heads": active_heads,
                "broadcast_chip_count": chips,
                "operation": "broadcast each post-processed score vector to all chips for PV",
            },
        )
        emit(
            "ALL_BANK_PV",
            1,
            _cycles_at_controller(pv_raw, controller_hz, bank_pe_clock_hz),
            scope="block",
            domain="bank",
            formula="ceil(Dhc/tp_par)*ceil(S/pv_dim_par)+pv_ctrl",
            details={
                "S": seq_len,
                "Dhc": dhc,
                "raw_bank_cycles": pv_raw,
                "head_wave": head_wave,
                "active_banks": active_heads,
            },
        )
        emit_transfer(
            "PV_XFER",
            active_heads * head_dim,
            scope="block",
            details={
                "head_wave": head_wave,
                "active_heads": active_heads,
                "chip_slice_elements": dhc,
                "operation": "concatenate chip PV slices in the DIMM PE",
            },
        )

    emit_dimm_beats(
        "QUANT",
        quant_h,
        scope="block",
        formula="2*ceil(H/quant_lanes)+1",
        details={"X": h, "quant_lanes": quant_lanes, "raw_dimm_cycles": quant_h},
    )
    emit_linear(
        stage="output_projection",
        input_dim=h,
        projections=output_projection,
    )
    emit_dequant("output_projection")
    emit_banked_transfer("LINEAR_XFER", h, scope="block")
    emit(
        "RMS_MLP",
        1,
        _cycles_at_controller(2 * h + 2, controller_hz, dimm_pe_clock_hz),
        scope="block",
        domain="dimm",
        formula="2H + 2",
        details={"H": h, "raw_dimm_cycles": 2 * h + 2},
    )
    emit_dimm_beats(
        "QUANT",
        quant_h,
        scope="block",
        formula="2*ceil(H/quant_lanes)+1",
        details={"X": h, "quant_lanes": quant_lanes, "raw_dimm_cycles": quant_h},
    )
    emit_linear(
        stage="up_gate",
        input_dim=h,
        projections=gate_up_projections,
    )
    emit_dequant("up_gate")
    emit_banked_transfer("LINEAR_XFER", 2 * mlp_dim, scope="block")
    mlp_fusion_raw = 2 * mlp_dim + 4
    emit(
        "MLP_FUSION",
        1,
        _cycles_at_controller(mlp_fusion_raw, controller_hz, dimm_pe_clock_hz),
        scope="block",
        domain="dimm",
        formula="2M + 4",
        details={"M": mlp_dim, "raw_dimm_cycles": mlp_fusion_raw},
    )
    quant_m = 2 * _ceil_div(mlp_dim, quant_lanes) + 1
    emit_dimm_beats(
        "QUANT",
        quant_m,
        scope="block",
        formula="2*ceil(M/quant_lanes)+1",
        details={"X": mlp_dim, "quant_lanes": quant_lanes, "raw_dimm_cycles": quant_m},
    )
    emit_linear(
        stage="down_projection",
        input_dim=mlp_dim,
        projections=down_projection,
    )
    emit_dequant("down_projection")
    emit_banked_transfer("LINEAR_XFER", h, scope="block")
    final_res_raw = _ceil_div(h, residual_lanes) + 1
    emit(
        "FINAL_RES",
        1,
        _cycles_at_controller(final_res_raw, controller_hz, dimm_pe_clock_hz),
        scope="block",
        domain="dimm",
        formula="ceil(H/residual_lanes)+1",
        details={"H": h, "residual_lanes": residual_lanes, "raw_dimm_cycles": final_res_raw},
    )
    emit(
        "BLOCK_END",
        1,
        0,
        scope="block",
        domain="dimm",
        formula="zero-latency block completion marker",
        allow_zero_cycles=True,
    )

    single_block_lines = list(lines[block_line_start:])
    single_block_events = deepcopy(events[block_event_start:])
    single_block_cycles = sum(int(event["memory_cycles"]) for event in single_block_events)
    if weight_rows_per_layer <= 0:
        raise ValueError("weight row allocation must be positive")
    for event in events[block_event_start:]:
        event.setdefault("formula_terms", {})["layer_id"] = 0

    def remap_layer_line(line: str, layer_id: int) -> str:
        fields = line.split()
        if len(fields) not in {7, 8}:
            raise ValueError(f"unexpected semantic trace line: {line}")
        row = int(fields[5])
        if row >= 0:
            command = fields[0]
            if command == "ALL_BANK_BITLINEAR":
                row += layer_id * weight_rows_per_layer
            elif command in {
                "ALL_BANK_WR_K_CURRENT",
                "ALL_BANK_WR_V_CURRENT",
                "ALL_BANK_RD_K_V_CACHE",
            }:
                row += layer_id * kv_rows_per_layer
            else:
                raise ValueError(f"unexpected concrete-address command: {command}")
            fields[5] = str(row)
        return " ".join(fields)

    if compact_layers:
        for template in single_block_events:
            token = template["trace_tokens"][0]
            command_counts[str(template["command"])] += (
                (layers - 1) * int(token["count"])
            )
        timeline_cycle += (layers - 1) * single_block_cycles
    else:
        for layer_id in range(1, layers):
            lines.extend(remap_layer_line(line, layer_id) for line in single_block_lines)
            for template in single_block_events:
                event = deepcopy(template)
                event["command_id"] = len(events)
                event.setdefault("formula_terms", {})["layer_id"] = layer_id
                total_cycles = int(event["memory_cycles"])
                start_cycle = timeline_cycle
                event["start_cycle"] = start_cycle
                event["end_cycle"] = start_cycle + max(0, total_cycles - 1)
                events.append(event)
                timeline_cycle += total_cycles
                token = event["trace_tokens"][0]
                command_counts[str(event["command"])] += int(token["count"])

    host_post_line_start = len(lines)

    cpu_norm_ops = 4 * h + cpu_rsqrt_ops
    cpu_norm_cpu, cpu_norm_ctrl = cpu_cycles(cpu_norm_ops)
    emit(
        "CPU_NORM",
        1,
        cpu_norm_ctrl,
        scope="host_post",
        domain="cpu",
        formula="CPU_CLK(4H + cpu_rsqrt_ops)",
        details={"ops": cpu_norm_ops, "cpu_cycles": cpu_norm_cpu},
    )
    lm_head_ops = 2 * vocab * h
    lm_head_cpu, lm_head_ctrl = cpu_cycles(lm_head_ops)
    emit(
        "LM_HEAD",
        1,
        lm_head_ctrl,
        scope="host_post",
        domain="cpu",
        formula="CPU_CLK(2*Vocab*H)",
        details={"ops": lm_head_ops, "cpu_cycles": lm_head_cpu},
    )
    topk_cpu, topk_ctrl = cpu_cycles(topk_ops)
    emit(
        "TOPK",
        1,
        topk_ctrl,
        scope="host_post",
        domain="cpu",
        formula="CPU_CLK(topk_ops)",
        details={"ops": topk_ops, "cpu_cycles": topk_cpu},
    )
    emit(
        "TOKEN_DONE",
        1,
        0,
        scope="host_post",
        domain="cpu",
        formula="zero-latency token completion marker issued after TOPK timing completes",
        allow_zero_cycles=True,
    )

    def layer_dependency_checks(layer_id: int) -> bool:
        layer_events = [
            event
            for event in events
            if event["scope"] == "block"
            and int(event.get("formula_terms", {}).get("layer_id", -1)) == layer_id
        ]

        def positions(command: str, head_wave: int | None = None) -> List[int]:
            return [
                index
                for index, event in enumerate(layer_events)
                if event["command"] == command
                and (
                    head_wave is None
                    or int(event.get("formula_terms", {}).get("head_wave", -1)) == head_wave
                )
            ]

        previous_wave_end = -1
        for head_wave in range(head_waves):
            lut = positions("ALL_BANK_LUT_LOAD", head_wave)
            bitlinear = positions("ALL_BANK_BITLINEAR", head_wave)
            ordered_after_qkv = [
                positions("ALL_BANK_DEQUANT", head_wave),
                positions("ALL_BANK_ROPE", head_wave),
                positions("ALL_BANK_WR_K_CURRENT", head_wave),
                positions("ALL_BANK_WR_V_CURRENT", head_wave),
                positions("ALL_BANK_RD_K_V_CACHE", head_wave),
                positions("ALL_BANK_QKT", head_wave),
                positions("QKT_XFER", head_wave),
                positions("QKT_REDUCE", head_wave),
                positions("ALL_BANK_POST", head_wave),
                positions("POST_SCORE_BCAST", head_wave),
                positions("ALL_BANK_PV", head_wave),
                positions("PV_XFER", head_wave),
            ]
            if not lut or not bitlinear or any(not values for values in ordered_after_qkv):
                return False
            if previous_wave_end >= min(lut):
                return False
            if any(
                lut_pos + 1 >= len(layer_events)
                or layer_events[lut_pos + 1]["command"] != "ALL_BANK_BITLINEAR"
                for lut_pos in lut
            ):
                return False
            if max(bitlinear) >= min(ordered_after_qkv[0]):
                return False
            if any(
                max(left) >= min(right)
                for left, right in zip(ordered_after_qkv, ordered_after_qkv[1:])
            ):
                return False
            previous_wave_end = max(ordered_after_qkv[-1])
        return True

    block_end_events = [event for event in events if event["command"] == "BLOCK_END"]
    token_done_events = [event for event in events if event["command"] == "TOKEN_DONE"]
    first_cpu_norm = next(
        index for index, event in enumerate(events) if event["command"] == "CPU_NORM"
    )
    last_block_event = max(
        index for index, event in enumerate(events) if event["scope"] == "block"
    )
    checks = {
        "legacy_commands_absent": not any(command in command_counts for command in LEGACY_COMMANDS),
        "bitlinear_present": command_counts["ALL_BANK_BITLINEAR"] > 0,
        "split_dequant_and_rope_present": (
            command_counts["ALL_BANK_DEQUANT"] == (head_waves + 3) * layers
            and command_counts["ALL_BANK_ROPE"] == head_waves * layers
        ),
        "qkv_repeats_per_head_wave": (
            command_counts["ALL_BANK_LUT_LOAD"]
            == layers * (head_waves * qkv_tiles + output_tiles + mlp_tiles + down_tiles)
        ),
        "attention_transfers_cover_all_waves": (
            command_counts["QKT_XFER"] == layers * head_waves
            and command_counts["POST_SCORE_BCAST"]
            == layers * head_waves
            and command_counts["PV_XFER"] == layers * head_waves
            and command_counts["QKT_REDUCE"] == layers * head_waves
            and command_counts["ALL_BANK_POST"] == layers * head_waves
        ),
        "all_layer_dependencies_are_conservative": all(
            layer_dependency_checks(layer_id)
            for layer_id in (range(1) if compact_layers else range(layers))
        ),
        "all_transformer_layers_are_traced": (
            command_counts["BLOCK_END"] == layers
            if compact_layers
            else len(block_end_events) == layers
        ),
        "block_end_has_zero_modeled_latency": all(
            int(event["memory_cycles"]) == 0 for event in block_end_events
        ),
        "token_done_follows_topk_and_has_zero_modeled_latency": (
            len(token_done_events) == 1
            and int(token_done_events[0]["memory_cycles"]) == 0
            and next(index for index, event in enumerate(events) if event["command"] == "TOPK")
            < next(index for index, event in enumerate(events) if event["command"] == "TOKEN_DONE")
        ),
        "cpu_postprocessing_after_all_layers": last_block_event < first_cpu_norm,
        "bank_array_width_matches_lut_pair_cells": (
            bank_array_bits_per_cycle == lut_pairs_per_group * pair_weight_bits
        ),
        "bitlinear_commands_are_one_bank_array_access_each": all(
            int(event.get("formula_terms", {}).get("array_access_bits", 0))
            == bank_array_bits_per_cycle
            and int(event["trace_tokens"][0]["count"])
            == int(event.get("formula_terms", {}).get("physical_fetches_per_tile", -1))
            for event in events
            if event["command"] == "ALL_BANK_BITLINEAR"
        ),
        "kv_count_is_bank_array_access_granular": (
            command_counts["ALL_BANK_RD_K_V_CACHE"]
            == layers * kv_read_accesses_per_layer
            and command_counts["ALL_BANK_WR_K_CURRENT"]
            == layers
            * head_waves
            * _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle)
            and command_counts["ALL_BANK_WR_V_CURRENT"]
            == layers
            * head_waves
            * _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle)
        ),
        "linear_xfer_is_equal_per_bank": all(
            int(event["trace_tokens"][0]["count"]) == 1
            and int(event["memory_cycles"])
            == banks_per_chip
            * int(event.get("formula_terms", {}).get("transfer_clocks_per_bank", -1))
            for event in events
            if event["command"] == "LINEAR_XFER"
        ),
        "merged_rank_transfers_are_single_commands": all(
            int(event["trace_tokens"][0]["count"]) == 1
            and bool(event.get("formula_terms", {}).get("merged_trace_command"))
            and int(event["memory_cycles"])
            == int(event.get("formula_terms", {}).get("transfer_clocks", -1))
            for event in events
            if event["command"]
            in {"HOST_XFER", "QKT_XFER", "POST_SCORE_BCAST", "PV_XFER", "LINEAR_XFER"}
        ),
        "lut_geometry_and_transfer_duration_match_config": all(
            int(event.get("formula_terms", {}).get("lut_pairs", 0))
            == lut_pairs_per_group
            and int(event.get("formula_terms", {}).get("lut_xfer_clocks", 0))
            == _ceil_div(
                lut_pairs_per_group * lut_elements_per_pair * lut_entry_bits * chips,
                rank_io_bits_per_clock,
            )
            for event in events
            if event["command"] == "ALL_BANK_LUT_LOAD"
        ),
        "controller_clock_contains_two_io_edges": (
            rank_io_bits_per_clock == 2 * controller_io_bits_per_edge
        ),
        "post_uses_conservative_raw": any(
            event["command"] == "ALL_BANK_POST"
            and event["formula_terms"].get("timing_policy") == "conservative_raw"
            for event in events
        ),
        "chip_slices_cover_head_dim": chips * dhc >= head_dim and (chips - 1) * dhc < head_dim,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"latest command scheme plan validation failed: {', '.join(failed)}")

    warnings: List[str] = []
    if chips * dhc != head_dim:
        warnings.append(
            f"head dimension {head_dim} is padded to {chips * dhc} values across {chips} chips"
        )
    if bool(pim_config.get("BITACC_STAGE_OVERLAP", pim_config.get("bitacc_stage_overlap", False))):
        warnings.append("configured stage overlap is intentionally disabled by conservative command dependencies")

    host_pre_cycles = sum(int(event["memory_cycles"]) for event in events if event["scope"] == "host_pre")
    host_post_cycles = sum(int(event["memory_cycles"]) for event in events if event["scope"] == "host_post")
    returned_lines: Iterable[str] = lines
    returned_events = events
    if compact_layers:
        host_pre_lines = list(lines[:block_line_start])
        host_post_lines = list(lines[host_post_line_start:])
        returned_lines = _LayeredTraceLines(
            host_pre=host_pre_lines,
            block=single_block_lines,
            host_post=host_post_lines,
            layers=layers,
            weight_rows_per_layer=weight_rows_per_layer,
            kv_rows_per_layer=kv_rows_per_layer,
        )
        compact_events: List[Dict[str, Any]] = []
        compact_timeline = 1
        for template in events:
            event = deepcopy(template)
            multiplier = layers if event["scope"] == "block" else 1
            if multiplier > 1:
                event.setdefault("formula_terms", {})["layer_instances"] = layers
                for token in event.get("trace_tokens", []):
                    for key in (
                        "count",
                        "pe_cycles_total",
                        "memory_cycles_total",
                    ):
                        token[key] = int(token.get(key, 0)) * multiplier
                for key in ("cycles", "logic_cycles", "memory_cycles", "pe_cycles"):
                    event[key] = int(event.get(key, 0)) * multiplier
            total_cycles = int(event["memory_cycles"])
            event["command_id"] = len(compact_events)
            event["start_cycle"] = compact_timeline
            event["end_cycle"] = compact_timeline + max(0, total_cycles - 1)
            compact_timeline += total_cycles
            compact_events.append(event)
        returned_events = compact_events
    return {
        "lines": returned_lines,
        "events": returned_events,
        "command_counts": dict(command_counts),
        "token_timings": token_timings,
        "nominal_trace_cycles": (
            host_pre_cycles + layers * single_block_cycles + host_post_cycles
        ),
        "block_cycles": single_block_cycles,
        "host_pre_cycles": host_pre_cycles,
        "host_post_cycles": host_post_cycles,
        "checks": checks,
        "warnings": warnings,
        "compact_layers": bool(compact_layers),
        "parameters": {
            "command_scheme": command_scheme,
            "hardware_scaling_cycle_model": bool(allow_hardware_scaling),
            "H": h,
            "L": layers,
            "A": heads,
            "Dh": head_dim,
            "M": mlp_dim,
            "Vocab": vocab,
            "S_kv": s_kv,
            "S": seq_len,
            "C": chips,
            "B": banks_per_chip,
            "Dhc": dhc,
            "head_waves": head_waves,
            "heads_per_wave": heads_per_wave,
            "qkv_groups_per_tile_per_wave": qkv_groups_per_tile_per_wave,
            "activation_bytes": activation_bytes,
            "weight_bits": weight_bits,
            "kv_slice_bytes_per_chip": kv_slice_bytes_per_chip,
            "kv_slice_storage_bytes_per_chip": kv_slice_storage_bytes,
            "kv_values_per_bank_chip": kv_values_per_bank_chip,
            "kv_head_waves_per_layer": kv_head_waves_per_layer,
            "kv_cache_write_commands_per_layer": (
                2
                * kv_head_waves_per_layer
                * _ceil_div(kv_slice_bytes_per_chip, bank_array_bytes_per_cycle)
            ),
            "kv_array_read_commands_per_wave": kv_read_accesses_per_wave,
            "kv_array_read_commands_per_layer": kv_read_accesses_per_layer,
            "kv_execution": "head_wave_masked_banks_with_per_chip_head_slices",
            "weight_base_row": weight_base_row,
            "weight_rows_per_layer": weight_rows_per_layer,
            "kv_cache_base_row": kv_cache_base_row,
            "kv_pair_bytes": kv_pair_bytes,
            "bank_row_bytes": bank_row_bytes,
            "kv_history_rows_per_wave": kv_history_rows_per_wave,
            "kv_allocated_rows_per_wave": kv_allocated_rows_per_wave,
            "kv_read_rows_per_layer": kv_read_rows_per_layer,
            "kv_history_bytes_per_wave": kv_history_bytes_per_wave,
            "kv_read_accesses_per_wave": kv_read_accesses_per_wave,
            "kv_read_accesses_per_layer": kv_read_accesses_per_layer,
            "kv_rows_per_layer": kv_rows_per_layer,
            "required_max_row": int(address_requirements["required_max_row"]),
            "lin_semantics": "lin cached KV tokens are present before this one-token decode trace",
            "qkt_xfer_beats_per_wave": qkt_xfer_beats_per_wave,
            "post_score_bcast_beats_per_wave": post_score_bcast_beats_per_wave,
            "pv_xfer_beats_per_wave": pv_xfer_beats_per_wave,
            "controller_hz": int(controller_hz),
            "bank_pe_clock_hz": int(bank_pe_clock_hz),
            "dimm_pe_clock_hz": int(dimm_pe_clock_hz),
            "cpu_clock_hz": int(cpu_clock_hz),
            "cpu_arithmetic_ops_per_second": int(cpu_ops_per_second),
            "input_scalars_per_lut_tile": scalars_per_lut_tile,
            "lut_pairs_per_group": lut_pairs_per_group,
            "input_elements_per_lut_pair": input_elements_per_lut_pair,
            "n_tile": n_tile,
            "lut_pairs_per_dimm_clock": lut_pairs_per_dimm_clock,
            "lut_elements_per_pair": lut_elements_per_pair,
            "lut_entry_bits": lut_entry_bits,
            "controller_io_bits_per_edge": controller_io_bits_per_edge,
            "rank_io_bits_per_clock": rank_io_bits_per_clock,
            "dq_bits_per_chip": dq_bits_per_chip,
            "dram_data_rate_mtps": dram_data_rate_mtps,
            "lut_control_dimm_cycles": lut_control_dimm_cycles,
            "qkt_reduce_additions_per_wave": qkt_reduce_additions,
            "qkt_reduce_controller_cycles": qkt_reduce_cycles,
            "bitlinear_weight_bytes_per_group": bitlinear_weight_bytes,
            "pair_cell_weight_bits": pair_weight_bits,
            "bank_array_bits_per_pe_cycle": bank_array_bits_per_cycle,
            "bank_array_bytes_per_pe_cycle": bank_array_bytes_per_cycle,
            "bitlinear_fetches_per_group": bitlinear_fetches_per_group,
        },
    }


def generate_latest_command_trace(
    modelinfos: Dict[str, Any],
    pim_config: Dict[str, Any],
    mem_model: Any,
    lin: int,
    lout: int,
    batch: int,
    out_dir: str,
    timeout_s: int | None = None,
    write_summary_json: bool = False,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    dram_org = str(getattr(mem_model, "dram_org", "DDR5_8Gb_x8"))
    chips_per_dimm = int(getattr(mem_model, "chips_per_dimm", pim_config["BITACC_CHIP_COUNT"]))
    if chips_per_dimm != int(pim_config["BITACC_CHIP_COUNT"]):
        raise ValueError(
            f"chip_count mismatch: Ramulator={chips_per_dimm}, config={pim_config['BITACC_CHIP_COUNT']}"
        )
    channel_width = getattr(mem_model, "channel_width", None)
    burst_bytes = _runtime_burst_bytes(
        pim_config,
        dram_org=dram_org,
        chips_per_dimm=chips_per_dimm,
        channel_width=channel_width,
    )
    row_bytes = _row_bytes_for_org(dram_org, chips_per_dimm)
    bank_layout = _resolve_bank_layout(dram_org, int(pim_config["bank_count_per_chip"]))
    topology = _resolve_memory_topology(
        int(getattr(mem_model, "channel", 1)), int(getattr(mem_model, "rank", 1))
    )
    plan = build_latest_command_plan(
        modelinfos=modelinfos,
        pim_config=pim_config,
        lin=lin,
        lout=lout,
        burst_bytes=burst_bytes,
        row_bytes=row_bytes,
        bank_layout=bank_layout,
        topology=topology,
        compact_layers=True,
    )
    command_scheme = str(plan["parameters"]["command_scheme"])
    row_capacity = ddr_rows_for_org(dram_org)
    required_max_row = int(plan["parameters"]["required_max_row"])
    if required_max_row >= row_capacity:
        raise ValueError(
            f"{dram_org} has rows 0..{row_capacity - 1}, but {modelinfos['name']} "
            f"with lin={int(lin)} requires row {required_max_row}; choose a larger DRAM org"
        )
    extra_timings = {f"n{name}": cycles for name, cycles in plan["token_timings"].items()}
    full_run = _run_runtime_trace_lines(
        lines=plan["lines"],
        out_dir=out_dir,
        tag="trace_out",
        ramulator_bin=str(getattr(mem_model, "ramulator_bin")),
        dram_impl=str(getattr(mem_model, "dram_impl", "DDR5-BITACC")),
        dram_org=dram_org,
        dram_timing=str(getattr(mem_model, "dram_timing", "DDR5_3200AN")),
        chips_per_dimm=chips_per_dimm,
        channel_width=(None if channel_width is None else int(channel_width)),
        channel=int(getattr(mem_model, "channel", 1)),
        rank=int(getattr(mem_model, "rank", 1)),
        n_lut_bcast=int(pim_config.get("BITACC_TRACE_N_LUT_BCAST", 1)),
        n_rdpim=int(pim_config.get("BITACC_TRACE_N_RDPIM", 1)),
        n_macab=max(1, int(pim_config.get("BITACC_ACC_UNITS", 1) or 1)),
        timeout_s=int(timeout_s or getattr(mem_model, "ramulator_timeout_s", 300)),
        extra_timings=extra_timings,
        extra_commands_to_count=list(LATEST_COMMANDS),
        keep_yaml=True,
        keep_cmd_counter=True,
        keep_cmd_trace=True,
    )
    input_counts = _count_input_trace_tokens(full_run["files"]["trace"])
    expected = {name: int(count) for name, count in plan["command_counts"].items()}
    observed = {name: int(full_run.get("cmd_counter", {}).get(name, 0)) for name in expected}
    nominal = int(plan["nominal_trace_cycles"])
    ramulator_cycles = int(full_run.get("memory_system_cycles", -1))
    layers = int(modelinfos["ndec"])
    nominal_decode_full_token_cycles = (
        int(plan["host_pre_cycles"])
        + layers * int(plan["block_cycles"])
        + int(plan["host_post_cycles"])
    )
    ramulator_protocol_overhead_cycles = ramulator_cycles - nominal_decode_full_token_cycles
    component = {
        "formula": (
            "decode_full_token_cycles = host_pre + L * conservative_block + host_post "
            "+ ramulator_protocol_overhead"
        ),
        "embedding_cycles": int(plan["host_pre_cycles"]),
        "positional_encoding_cycles": 0,
        "per_transformer_block_cycles": int(plan["block_cycles"]),
        "per_transformer_block_source": command_scheme,
        "transformer_block_count": layers,
        "all_transformer_blocks_cycles": layers * int(plan["block_cycles"]),
        "lm_head_cycles": int(plan["host_post_cycles"]),
        "runtime_dispatch_host_overhead_cycles": 0,
        "ramulator_protocol_overhead_cycles": ramulator_protocol_overhead_cycles,
        "nominal_decode_full_token_cycles": nominal_decode_full_token_cycles,
        "decode_full_token_cycles": ramulator_cycles,
        "decode_run_tokens": max(1, int(lout)),
        "decode_run_cycles": ramulator_cycles * max(1, int(lout)),
    }
    count_match = {name: observed[name] == count for name, count in expected.items()}
    input_match = {name: int(input_counts.get(name, 0)) == count for name, count in expected.items()}
    summary: Dict[str, Any] = {
        "runtime_inputs": {
            "model": modelinfos["name"],
            "ndec": layers,
            "batch": int(batch),
            "lin": int(lin),
            "lout": int(lout),
            "burst_bytes": burst_bytes,
            "row_bytes": row_bytes,
            "command_scheme": command_scheme,
            "command_count_scope": "full_decode_token",
            "traced_transformer_blocks": layers,
            "phase_report_encoding": "layer_template_aggregated_counts",
            "banks": int(plan["parameters"]["B"]),
            **plan["parameters"],
        },
        "phase_report": plan["events"],
        "phase_report_13": [],
        "command_schedule": [],
        "full_trace": full_run,
        "trace_counts": {
            "pseudo_phase_tokens": expected,
            "input_trace_token_counts": input_counts,
            "input_trace_command_lines": sum(int(value) for value in input_counts.values()),
            "input_trace_count_match": input_match,
            "command_granularity_bytes": int(
                plan["parameters"]["bank_array_bytes_per_pe_cycle"]
            ),
            "command_granularity_scope": "mixed_physical_array_and_merged_duration",
            "bank_array_command_granularity_bytes": int(
                plan["parameters"]["bank_array_bytes_per_pe_cycle"]
            ),
            "rank_transfer_command_granularity_bytes": int(
                plan["parameters"]["rank_io_bits_per_clock"] // 8
            ),
            "logical_bitlinear_group_bytes": int(
                plan["parameters"]["bitlinear_weight_bytes_per_group"]
            ),
            "dram_burst_bytes": int(burst_bytes),
            "command_granularity_policy": (
                "ALL_BANK_BITLINEAR and KV array commands each consume one "
                f"{int(plan['parameters']['bank_array_bytes_per_pe_cycle'])}-byte slice "
                "per active bank/chip and 200 MHz Bank-PE cycle. LUT and "
                "off-chip transfers are one semantic trace line with an explicit "
                "1.6 GHz duration; each DQ clock serializes 16 rank bytes."
            ),
        },
        "validation": {
            "expected_command_counts": expected,
            "observed_command_counts": observed,
            "input_trace_token_counts": input_counts,
            "input_trace_command_lines": sum(int(value) for value in input_counts.values()),
            "input_trace_count_match": input_match,
            "count_match": count_match,
            "nominal_trace_memory_cycles": nominal,
            "formula_memory_cycles": nominal,
            "formula_memory_match": True,
            "ramulator_memory_cycles": ramulator_cycles,
            "ramulator_minus_formula_cycles": (
                None if ramulator_cycles < 0 else ramulator_cycles - nominal
            ),
            "latest_scheme_checks": plan["checks"],
        },
        "design_derivation": {
            "input_dim": int(plan["parameters"]["H"]),
            "mlp_dim": int(plan["parameters"]["M"]),
            "chip_count": int(plan["parameters"]["C"]),
            "bank_count_per_chip": int(plan["parameters"]["B"]),
            "head_dim_per_chip": int(plan["parameters"]["Dhc"]),
            "seq_len": int(plan["parameters"]["S"]),
            "kv_address_mapping": (
                "starts immediately after all layer weights; each head wave maps one head per active bank, "
                "each chip stores Dh/C K and V values for that head, and history K/V are interleaved per token"
            ),
            "weight_address_mapping": (
                "per-chip rows use 1024 addressable bytes and "
                f"{int(plan['parameters']['bank_array_bytes_per_pe_cycle'])}-byte/"
                f"{int(plan['parameters']['bank_array_bits_per_pe_cycle'])}-bit Bank-PE accesses; "
                f"each {int(plan['parameters']['lut_pairs_per_group'])}-pair LUT group is followed "
                "by one access per local output column; QKV rows are "
                "reserved per head wave, while WO/WGATE/WUP/WDOWN remain distributed once across all banks/chips; "
                f"this model reserves {int(plan['parameters']['weight_rows_per_layer'])} rows/layer"
            ),
        },
        "estimates": {
            "controller_freq_hz": int(plan["parameters"]["controller_hz"]),
            "one_transformer_block_logic_cycles": int(plan["block_cycles"]),
            "one_transformer_block_memory_cycles": int(plan["block_cycles"]),
            "one_transformer_block_merged_cycles": int(plan["block_cycles"]),
            "one_transformer_block_pe_cycles": int(plan["block_cycles"]),
            "one_transformer_block_13_phase_cycles": int(plan["block_cycles"]),
            "one_transformer_block_trace_nominal_memory_cycles": int(plan["block_cycles"]),
            "one_transformer_block_ramulator_memory_cycles": None,
            "representative_full_trace_nominal_cycles": nominal,
            "representative_full_trace_ramulator_cycles": ramulator_cycles,
            "full_decode_token_ramulator_memory_cycles": ramulator_cycles,
            "total_cycles_to_next_token": max(0, ramulator_cycles),
            "steady_state_full_token_cycles": ramulator_cycles,
            "steady_state_decode_tokens_per_second": (
                float(plan["parameters"]["controller_hz"]) / ramulator_cycles
            ),
            "component_cycle_breakdown": component,
        },
        "files": {
            "input_trace": full_run["files"]["trace"],
            "trace_yaml": full_run["files"]["yaml"],
            "run_out": full_run["files"]["run_out"],
            "command_counter": full_run["files"]["cmd_counter"],
            "command_trace": full_run["files"]["cmd_trace"],
            "trace_out_ch0": full_run["files"]["cmd_trace"],
        },
        "warnings": plan["warnings"],
        "notes": {
            "scope": "one full decode token: host pre, all transformer layers, then host post",
            "phase_report_encoding": (
                "one transformer-layer event template with trace counts and active cycles "
                "aggregated across all layers; physical Ramulator input still contains every layer"
            ),
            "dependency_policy": "fully conservative; no K-read/QKT, QKT/POST, V-read/PV overlap",
            "post_policy": "T_post_raw only; optimized tail timing is not used",
            "lut_policy": (
                f"{int(plan['parameters']['lut_pairs_per_group'])} pairs per LUT group; "
                "one pair generated per DIMM clock and each pair "
                "is four 16-bit entries replicated to eight x8 chips. Generation and DQ "
                "broadcast overlap, followed by the configured DIMM control latency."
            ),
            "kv_address_policy": (
                "K/V are interleaved in each bank row; ALL_BANK_RD_K_V_CACHE streams "
                f"{int(plan['parameters']['bank_array_bytes_per_pe_cycle'])} bytes from every "
                "active bank/chip per 200 MHz Bank-PE cycle"
            ),
            "block_end_policy": "zero-latency marker; Ramulator may co-issue it with the next real command",
        },
    }
    if not all(count_match.values()) or not all(input_match.values()):
        raise ValueError(
            "latest command scheme count validation failed: "
            f"Ramulator={count_match}, input={input_match}"
        )
    if ramulator_cycles <= 0 or bool(full_run.get("timed_out")):
        raise RuntimeError(
            f"latest command scheme Ramulator replay did not complete: cycles={ramulator_cycles}, "
            f"timed_out={full_run.get('timed_out')}"
        )
    if write_summary_json:
        summary_path = os.path.join(out_dir, "latest_command_trace_summary.json")
        with open(summary_path, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        summary["files"]["summary_json"] = summary_path
    return summary
