from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
import csv
import itertools
import json


@dataclass(frozen=True)
class Workload:
    # Transformer geometry. The baseline matches the current 2K hidden example.
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 12
    ffn_mult: int = 4
    tile_cols: int = 64
    qkv_projections: int = 3


@dataclass(frozen=True)
class DRAM:
    # DDR burst + row timing knobs.
    burst_bytes: int = 64
    row_bytes: int = 8192
    tRCD: int = 24
    tRP: int = 24
    tCCD: int = 1
    enable_refresh: bool = True
    tREFI: int = 12480
    tRFC: int = 350


@dataclass(frozen=True)
class Host:
    # Host/controller-side timing knobs.
    prep_cycles: int = 900
    prep_cycles_mode: str = "fixed"
    pass_cycles_2048: int = 200
    lut_gen_cycles: int = 200
    act_passes_8192: int = 2
    head_dim: int = 64
    cpu_overhead_per_token: int = 1


@dataclass(frozen=True)
class BitWidths:
    # Bus-visible precisions for Q/output movement and stored weight size.
    q_bits: int = 16
    out_bits: int = 16
    weight_bits: int = 16


@dataclass(frozen=True)
class Architecture:
    # PIM/NoC-side architectural knobs.
    banks: int = 16
    # Effective packed weight storage bits used by memory traffic models.
    # For BitAcc ternary-pair LUT flow, this is typically 2 bits/weight.
    packed_weight_bits: int = 2
    lut_bytes_per_bank: int = 8192
    qkv_tile_cost: int = 2088
    kv_writeback_bytes_per_tile: int = 8192
    kv_writeback_cost_cycles: Optional[int] = 152
    kv_model: str = "row_per_token"
    kv_bytes_per_history_token_per_tile: int = 0
    # Sliding KV cache window. 0 means full history (no eviction).
    kv_cache_max_tokens: int = 0
    init_serial_across_banks: bool = True
    # Init-cycle model selection:
    # - "analytic": existing burst/ACT/PRE(+refresh) idealized formula.
    # - "ramulator_counts": use command-count based model calibrated by Ramulator plugin dumps.
    # - "ramulator_measured": directly use measured per-bank cycles from Ramulator run.
    # - "ramulator_total": directly use measured model-level init cycles (no init math in DSE).
    init_cycle_model: str = "analytic"
    # Command-count model inputs (per-bank run, from CommandCounter plugin):
    init_wr_count: int = 0
    init_act_count: int = 0
    init_prea_count: int = 0
    init_refab_count: int = 0
    # Timing knobs for command-count model:
    init_write_spacing_cycles: int = 0
    init_act_cycles: int = 24
    init_pre_cycles: int = 24
    init_refab_cycles: int = 0
    # Measured model input:
    init_measured_per_bank_cycles: float = 0.0
    # Optional reference bytes to scale measured cycles linearly when bytes change.
    init_measured_per_bank_bytes: int = 0
    # Direct model-level measured init cycles (highest priority in ramulator_total mode).
    init_measured_model_cycles: float = 0.0
    # Optional per-stage cycle overrides measured from Ramulator traces.
    # Keeps the same formula pipeline but replaces selected stage constants.
    use_trace_stage_cycles: bool = False
    trace_attn_round_parallel_cycles: float = 0.0
    trace_attn_round_bus_cycles: float = 0.0
    trace_mlp_wup_cycles: float = 0.0
    trace_mlp_wgate_cycles: float = 0.0
    trace_mlp_act_gate_cycles: float = 0.0
    trace_mlp_wdown_cycles: float = 0.0
    # Optional trace-generation overrides for tiled LUT scheduling.
    lut_tiling_elements: int = 0
    lut_tiling_columns: int = 0
    # Weight elements consumed by one MACAB burst.
    # Example: burst_elements=128 gives:
    #   LUT=128 -> 1 MACAB per column per LUT
    #   LUT=64  -> 1 MACAB per column per LUT (remaining 64 elements wasted)
    #   LUT=256 -> 2 MACAB per column per LUT
    lut_macab_span_elements: int = 128


@dataclass(frozen=True)
class Evaluation:
    # Reporting knobs for the DSE run.
    n_tokens: int = 1000
    nth_token: int = 1000
    freq_ghz: float = 3.2
    topk: int = 10


def _dataclass_from_dict(cls, data: Optional[Dict[str, Any]]):
    if not data:
        return cls()
    valid = {field.name for field in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**data)


def _load_dse_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if raw is None:
        raw = {}

    bit_widths = raw.get("bit_widths", raw.get("bw", {}))
    architecture = raw.get("architecture", raw.get("arch", {}))

    return {
        "workload": _dataclass_from_dict(Workload, raw.get("workload", {})),
        "dram": _dataclass_from_dict(DRAM, raw.get("dram", {})),
        "host": _dataclass_from_dict(Host, raw.get("host", {})),
        "bit_widths": _dataclass_from_dict(BitWidths, bit_widths),
        "architecture": _dataclass_from_dict(Architecture, architecture),
        "evaluation": _dataclass_from_dict(Evaluation, raw.get("evaluation", {})),
        "sweep": raw.get("sweep", {}),
    }


def dram_stream_cycles(bytes_total: int,
                       dram: DRAM,
                       include_pre: bool = True) -> int:
    # Stream bytes over the DDR interface and add ACT/PRE overhead.
    bursts = ceil(bytes_total / dram.burst_bytes)
    bursts_per_row = max(1, dram.row_bytes // dram.burst_bytes)
    rows = ceil(bursts / bursts_per_row)

    total = bursts * dram.tCCD + rows * dram.tRCD
    if include_pre:
        total += max(0, rows - 1) * dram.tRP

    if dram.enable_refresh and dram.tREFI > 0 and dram.tRFC > 0:
        total += (total // dram.tREFI) * dram.tRFC
    return total


def latch_cycles(arch: Architecture, dram: DRAM) -> int:
    # Copy one bank-local LUT buffer over the shared bus.
    return ceil(arch.lut_bytes_per_bank / dram.burst_bytes) * dram.tCCD


def tile_write_cycles_64elems(elem_bits: int, dram: DRAM) -> int:
    # Move one 64-element tile over the DDR/main bus.
    bytes_total = ceil(64 * elem_bits / 8)
    return ceil(bytes_total / dram.burst_bytes)


def kv_writeback_cycles(arch: Architecture, dram: DRAM) -> int:
    if arch.kv_writeback_cost_cycles is not None:
        return arch.kv_writeback_cost_cycles
    return dram_stream_cycles(arch.kv_writeback_bytes_per_tile,
                              dram,
                              include_pre=False)


def kv_read_cycles_row_per_token(history_tokens: int, dram: DRAM) -> int:
    # Reproduce the row-per-history-token model from the user's derivation.
    if history_tokens <= 0:
        return 0

    row_cost = dram.tRCD + (dram.row_bytes // dram.burst_bytes) * dram.tCCD + dram.tRP
    total = row_cost * history_tokens - dram.tRP

    if dram.enable_refresh and dram.tREFI > 0 and dram.tRFC > 0:
        total += (total // dram.tREFI) * dram.tRFC
    return total


def kv_read_cycles_byte_model(history_tokens: int,
                              arch: Architecture,
                              dram: DRAM) -> int:
    # Optional byte-accurate KV model for future experiments.
    if history_tokens <= 0:
        return 0
    total_bytes = history_tokens * arch.kv_bytes_per_history_token_per_tile
    return dram_stream_cycles(total_bytes, dram, include_pre=True)


def cpu_attention_cycles_per_tile(history_tokens: int, host: Host) -> int:
    return history_tokens * (host.head_dim + host.cpu_overhead_per_token)


def attention_prep_cycles(workload: Workload, host: Host) -> int:
    if host.prep_cycles_mode == "fixed":
        return int(host.prep_cycles)
    if host.prep_cycles_mode == "auto_quant_lut":
        # Attention prep on host: quantize current activation + build LUT.
        return int(2 * pass_cycles(workload.d_model, host, workload) + host.lut_gen_cycles)
    raise ValueError(
        f"Unknown host.prep_cycles_mode={host.prep_cycles_mode}. "
        "Use one of: fixed, auto_quant_lut"
    )


def attention_block_cycles(token_index: int,
                           workload: Workload,
                           arch: Architecture,
                           dram: DRAM,
                           host: Host,
                           widths: BitWidths) -> Dict[str, int]:
    history_tokens_raw = token_index - 1
    if arch.kv_cache_max_tokens > 0:
        history_tokens = min(history_tokens_raw, arch.kv_cache_max_tokens)
    else:
        history_tokens = history_tokens_raw

    t_latch = latch_cycles(arch, dram)
    t_qkv_tile = arch.qkv_tile_cost
    t_kv_write = kv_writeback_cycles(arch, dram)
    t_round_parallel = t_latch + t_qkv_tile + t_kv_write

    t_q_to_host = tile_write_cycles_64elems(widths.q_bits, dram)
    t_out_write = tile_write_cycles_64elems(widths.out_bits, dram)
    if arch.kv_model == "row_per_token":
        t_kv_read = kv_read_cycles_row_per_token(history_tokens, dram)
    elif arch.kv_model == "byte_model":
        t_kv_read = kv_read_cycles_byte_model(history_tokens, arch, dram)
    else:
        raise ValueError(f"Unknown kv_model={arch.kv_model}")

    t_cpu_attention = cpu_attention_cycles_per_tile(history_tokens, host)
    t_tile_bus = t_q_to_host + t_kv_read + t_cpu_attention + t_out_write
    t_round_bus = arch.banks * t_tile_bus
    if arch.use_trace_stage_cycles:
        if arch.trace_attn_round_parallel_cycles > 0:
            t_round_parallel = int(round(arch.trace_attn_round_parallel_cycles))
        if arch.trace_attn_round_bus_cycles > 0:
            t_round_bus = int(round(arch.trace_attn_round_bus_cycles))

    # Attention rounds follow head-to-bank mapping.
    # If banks >= n_heads, one round is sufficient.
    rounds = max(1, ceil(workload.n_heads / max(1, arch.banks)))
    t_prep = attention_prep_cycles(workload, host)
    t_attention = t_prep + rounds * t_round_parallel + rounds * t_round_bus
    return {
        "history_tokens_raw": history_tokens_raw,
        "history_tokens": history_tokens,
        "rounds": rounds,
        "prep": t_prep,
        "round_parallel": t_round_parallel,
        "tile_bus": t_tile_bus,
        "round_bus": t_round_bus,
        "attention_block": t_attention,
    }


def pass_cycles(length: int, host: Host, workload: Workload) -> int:
    return int(host.pass_cycles_2048 * (length / workload.d_model))


def mlp_block_cycles(workload: Workload,
                     arch: Architecture,
                     dram: DRAM,
                     host: Host,
                     widths: BitWidths) -> Dict[str, int]:
    d_model = workload.d_model
    ffn_dim = workload.ffn_mult * d_model
    t_latch = latch_cycles(arch, dram)

    def gemv_compute_cycles(out_dim: int) -> int:
        tiles_out = ceil(out_dim / workload.tile_cols)
        rounds = ceil(tiles_out / arch.banks)
        return rounds * arch.qkv_tile_cost

    def gemv_write_cycles(out_dim: int) -> int:
        tiles_out = ceil(out_dim / workload.tile_cols)
        return tiles_out * tile_write_cycles_64elems(widths.out_bits, dram)

    # MLP flow (gated): Wup + Wgate + activation*gate + Wdown.
    # Residual + LN is paid once before entering Wup.
    t_residual_norm = 4 * pass_cycles(d_model, host, workload)
    t_norm_d = 3 * pass_cycles(d_model, host, workload)
    t_quant_d = 2 * pass_cycles(d_model, host, workload)
    t_norm_ffn = 3 * pass_cycles(ffn_dim, host, workload)
    t_quant_ffn = 2 * pass_cycles(ffn_dim, host, workload)
    t_lut = host.lut_gen_cycles

    # Wup: (d -> 4d)
    t_prep_wup = t_residual_norm + t_quant_d + t_lut
    t_wup = t_prep_wup + t_latch + gemv_compute_cycles(ffn_dim) + \
        gemv_write_cycles(ffn_dim) + pass_cycles(ffn_dim, host, workload)

    # Wgate: (d -> 4d) with fresh SubLN+quant+LUT before this linear path.
    t_prep_wgate = t_norm_d + t_quant_d + t_lut
    t_wgate = t_prep_wgate + t_latch + gemv_compute_cycles(ffn_dim) + \
        gemv_write_cycles(ffn_dim) + pass_cycles(ffn_dim, host, workload)

    # CPU-side nonlinearity + elementwise gating on 4d vector.
    t_act_gate = host.act_passes_8192 * pass_cycles(ffn_dim, host, workload)
    t_gate_mul = pass_cycles(ffn_dim, host, workload)

    # Wdown: (4d -> d)
    t_prep_wdown = t_norm_ffn + t_quant_ffn + t_lut
    t_wdown = t_prep_wdown + t_latch + gemv_compute_cycles(d_model) + \
        gemv_write_cycles(d_model) + pass_cycles(d_model, host, workload) + \
        t_residual_norm

    if arch.use_trace_stage_cycles:
        if arch.trace_mlp_wup_cycles > 0:
            t_wup = int(round(arch.trace_mlp_wup_cycles))
        if arch.trace_mlp_wgate_cycles > 0:
            t_wgate = int(round(arch.trace_mlp_wgate_cycles))
        if arch.trace_mlp_act_gate_cycles > 0:
            t_act_gate = int(round(arch.trace_mlp_act_gate_cycles))
            t_gate_mul = 0
        if arch.trace_mlp_wdown_cycles > 0:
            t_wdown = int(round(arch.trace_mlp_wdown_cycles))

    t_mlp = t_wup + t_wgate + t_act_gate + t_gate_mul + t_wdown
    return {
        "wup": t_wup,
        "wgate": t_wgate,
        "act_gate": t_act_gate + t_gate_mul,
        "wdown": t_wdown,
        "mlp_block": t_mlp,
    }


def init_weights_cycles(workload: Workload,
                        arch: Architecture,
                        dram: DRAM,
                        widths: BitWidths) -> Dict[str, int]:
    d_model = workload.d_model
    ffn_dim = workload.ffn_mult * d_model
    packed_weight_bits = max(1, int(arch.packed_weight_bits))

    def matrix_bytes(rows: int, cols: int) -> int:
        return ceil(rows * cols * packed_weight_bits / 8)

    qkv_bytes = matrix_bytes(d_model, workload.qkv_projections * d_model)
    wup_bytes = matrix_bytes(d_model, ffn_dim)
    wgate_bytes = matrix_bytes(d_model, ffn_dim)
    wdown_bytes = matrix_bytes(ffn_dim, d_model)
    block_bytes = qkv_bytes + wup_bytes + wgate_bytes + wdown_bytes

    per_bank_bytes = ceil(block_bytes / arch.banks)

    if arch.init_cycle_model == "ramulator_total":
        if arch.init_measured_model_cycles <= 0:
            raise ValueError("init_measured_model_cycles must be > 0 when init_cycle_model=ramulator_total")
        model_cycles = int(round(arch.init_measured_model_cycles))
        block_cycles = int(round(model_cycles / max(1, workload.n_layers)))
        per_bank_cycles = int(round(block_cycles / max(1, arch.banks)))
    elif arch.init_cycle_model == "analytic":
        per_bank_cycles = dram_stream_cycles(per_bank_bytes, dram, include_pre=True)
        if arch.init_serial_across_banks:
            block_cycles = arch.banks * per_bank_cycles
        else:
            block_cycles = per_bank_cycles
        model_cycles = workload.n_layers * block_cycles
    elif arch.init_cycle_model == "ramulator_counts":
        write_spacing = arch.init_write_spacing_cycles if arch.init_write_spacing_cycles > 0 else dram.tCCD
        per_bank_cycles = int(
            arch.init_wr_count * write_spacing
            + arch.init_act_count * arch.init_act_cycles
            + arch.init_prea_count * arch.init_pre_cycles
            + arch.init_refab_count * arch.init_refab_cycles
        )
        if arch.init_serial_across_banks:
            block_cycles = arch.banks * per_bank_cycles
        else:
            block_cycles = per_bank_cycles
        model_cycles = workload.n_layers * block_cycles
    elif arch.init_cycle_model == "ramulator_measured":
        if arch.init_measured_per_bank_cycles <= 0:
            raise ValueError("init_measured_per_bank_cycles must be > 0 when init_cycle_model=ramulator_measured")
        per_bank_cycles = float(arch.init_measured_per_bank_cycles)
        if arch.init_measured_per_bank_bytes > 0:
            per_bank_cycles *= (per_bank_bytes / arch.init_measured_per_bank_bytes)
        per_bank_cycles = int(round(per_bank_cycles))
        if arch.init_serial_across_banks:
            block_cycles = arch.banks * per_bank_cycles
        else:
            block_cycles = per_bank_cycles
        model_cycles = workload.n_layers * block_cycles
    else:
        raise ValueError(
            f"Unknown init_cycle_model={arch.init_cycle_model}. "
            "Use one of: analytic, ramulator_counts, ramulator_measured, ramulator_total"
        )

    return {
        "block_bytes": block_bytes,
        "per_bank_bytes": per_bank_bytes,
        "per_bank_cycles": per_bank_cycles,
        "init_block_cycles": block_cycles,
        "init_model_cycles": model_cycles,
    }


def token_cycles_n(token_index: int,
                   workload: Workload,
                   arch: Architecture,
                   dram: DRAM,
                   host: Host,
                   widths: BitWidths) -> Dict[str, Any]:
    attention = attention_block_cycles(token_index, workload, arch, dram, host,
                                       widths)
    mlp = mlp_block_cycles(workload, arch, dram, host, widths)
    per_block = attention["attention_block"] + mlp["mlp_block"]
    token_total = workload.n_layers * per_block
    return {
        "token_index": token_index,
        "history_tokens": attention["history_tokens"],
        "history_tokens_raw": token_index - 1,
        "attention_block_cycles": attention["attention_block"],
        "mlp_block_cycles": mlp["mlp_block"],
        "block_cycles": per_block,
        "token_cycles": token_total,
        "breakdown": {
            "attention": attention,
            "mlp": mlp,
        }
    }


def total_cycles_for_tokens(n_tokens: int,
                            workload: Workload,
                            arch: Architecture,
                            dram: DRAM,
                            host: Host,
                            widths: BitWidths) -> Dict[str, Any]:
    init = init_weights_cycles(workload, arch, dram, widths)
    token_cycles = []
    total = init["init_model_cycles"]
    for token_index in range(1, n_tokens + 1):
        token_total = token_cycles_n(token_index, workload, arch, dram, host,
                                     widths)["token_cycles"]
        token_cycles.append(token_total)
        total += token_total
    return {
        "init_model_cycles": init["init_model_cycles"],
        "total_cycles": total,
        "per_token_cycles": token_cycles,
    }


def _rebuild_dataclasses(workload: Workload,
                         dram: DRAM,
                         host: Host,
                         widths: BitWidths,
                         arch: Architecture,
                         sweep_values: Dict[str, Any]):
    sections = {
        "workload": asdict(workload),
        "dram": asdict(dram),
        "host": asdict(host),
        "bit_widths": asdict(widths),
        "architecture": asdict(arch),
    }

    for dotted_key, value in sweep_values.items():
        prefix, field_name = dotted_key.split(".", 1)
        if prefix == "bw":
            prefix = "bit_widths"
        elif prefix == "arch":
            prefix = "architecture"
        if prefix not in sections:
            raise ValueError(f"Unsupported sweep key prefix: {prefix}")
        if field_name not in sections[prefix]:
            raise ValueError(f"Unknown sweep field: {dotted_key}")
        sections[prefix][field_name] = value

    return (
        Workload(**sections["workload"]),
        DRAM(**sections["dram"]),
        Host(**sections["host"]),
        BitWidths(**sections["bit_widths"]),
        Architecture(**sections["architecture"]),
    )


def run_dse(workload: Workload,
            dram: DRAM,
            host: Host,
            widths: BitWidths,
            arch: Architecture,
            evaluation: Evaluation,
            sweep: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(sweep.keys())
    if not keys:
        keys = []
        combinations = [()]
    else:
        combinations = itertools.product(*[sweep[key] for key in keys])

    results = []
    freq_hz = evaluation.freq_ghz * 1e9
    for combo in combinations:
        overrides = dict(zip(keys, combo))
        run_workload, run_dram, run_host, run_widths, run_arch = _rebuild_dataclasses(
            workload, dram, host, widths, arch, overrides)

        totals = total_cycles_for_tokens(evaluation.n_tokens, run_workload,
                                         run_arch, run_dram, run_host,
                                         run_widths)
        nth = token_cycles_n(evaluation.nth_token, run_workload, run_arch,
                             run_dram, run_host, run_widths)
        first_token = token_cycles_n(1, run_workload, run_arch, run_dram,
                                     run_host, run_widths)
        if run_arch.kv_cache_max_tokens > 0:
            steady_state_index = run_arch.kv_cache_max_tokens + 1
        else:
            steady_state_index = evaluation.nth_token
        steady_state = token_cycles_n(steady_state_index, run_workload, run_arch,
                                      run_dram, run_host, run_widths)
        total_seconds = totals["total_cycles"] / freq_hz
        tokens_per_second = evaluation.n_tokens / total_seconds
        nth_ms = nth["token_cycles"] * 1000 / freq_hz
        steady_state_ms = steady_state["token_cycles"] * 1000 / freq_hz
        steady_state_tokens_per_second = freq_hz / steady_state["token_cycles"]
        time_to_first_token_seconds = (
            totals["init_model_cycles"] + first_token["token_cycles"]) / freq_hz

        results.append({
            "tokens_per_second": tokens_per_second,
            "total_seconds": total_seconds,
            "total_cycles": totals["total_cycles"],
            "init_cycles": totals["init_model_cycles"],
            "nth_token_cycles": nth["token_cycles"],
            "nth_token_ms": nth_ms,
            "steady_state_token_index": steady_state_index,
            "steady_state_token_cycles": steady_state["token_cycles"],
            "steady_state_token_ms": steady_state_ms,
            "steady_state_tokens_per_second": steady_state_tokens_per_second,
            "time_to_first_token_seconds": time_to_first_token_seconds,
            "overrides": overrides,
            "config": {
                "workload": asdict(run_workload),
                "dram": asdict(run_dram),
                "host": asdict(run_host),
                "bit_widths": asdict(run_widths),
                "architecture": asdict(run_arch),
            }
        })

    results.sort(key=lambda item: item["tokens_per_second"], reverse=True)
    return results[:evaluation.topk]


def _flatten_dict(prefix: str, value: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, item in value.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            _flatten_dict(flat_key, item, out)
        else:
            out[flat_key] = item


def write_dse_csv(path: str, results: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        row: Dict[str, Any] = {"rank": rank}
        for key in [
                "tokens_per_second", "total_seconds", "total_cycles",
                "init_cycles", "nth_token_cycles", "nth_token_ms",
                "steady_state_token_index", "steady_state_token_cycles",
                "steady_state_token_ms", "steady_state_tokens_per_second",
                "time_to_first_token_seconds"
        ]:
            row[key] = result[key]
        _flatten_dict("override", result.get("overrides", {}), row)
        _flatten_dict("config", result.get("config", {}), row)
        rows.append(row)

    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dse_from_config(path: str) -> Dict[str, Any]:
    config = _load_dse_config(path)
    workload = config["workload"]
    dram = config["dram"]
    host = config["host"]
    widths = config["bit_widths"]
    arch = config["architecture"]
    evaluation = config["evaluation"]
    freq_hz = evaluation.freq_ghz * 1e9

    baseline_nth = token_cycles_n(evaluation.nth_token, workload, arch, dram,
                                  host, widths)
    baseline_first = token_cycles_n(1, workload, arch, dram, host, widths)
    if arch.kv_cache_max_tokens > 0:
        baseline_steady_index = arch.kv_cache_max_tokens + 1
    else:
        baseline_steady_index = evaluation.nth_token
    baseline_steady = token_cycles_n(baseline_steady_index, workload, arch, dram,
                                     host, widths)
    baseline_totals = total_cycles_for_tokens(evaluation.n_tokens, workload,
                                              arch, dram, host, widths)
    baseline_seconds = baseline_totals["total_cycles"] / freq_hz

    return {
        "workload": asdict(workload),
        "dram": asdict(dram),
        "host": asdict(host),
        "bit_widths": asdict(widths),
        "architecture": asdict(arch),
        "evaluation": asdict(evaluation),
        "baseline": {
            "nth_token_cycles": baseline_nth["token_cycles"],
            "nth_token_ms": baseline_nth["token_cycles"] * 1000 / freq_hz,
            "total_cycles": baseline_totals["total_cycles"],
            "total_seconds": baseline_seconds,
            "tokens_per_second": evaluation.n_tokens / baseline_seconds,
            "steady_state_token_index": baseline_steady_index,
            "steady_state_token_cycles": baseline_steady["token_cycles"],
            "steady_state_token_ms": baseline_steady["token_cycles"] * 1000 / freq_hz,
            "steady_state_tokens_per_second": freq_hz / baseline_steady["token_cycles"],
            "time_to_first_token_seconds": (
                baseline_totals["init_model_cycles"] + baseline_first["token_cycles"]
            ) / freq_hz,
        },
        "top_results": run_dse(workload, dram, host, widths, arch, evaluation,
                               config["sweep"]),
    }


def write_dse_report(report: Dict[str, Any],
                     json_path: Optional[str] = None,
                     csv_path: Optional[str] = None) -> None:
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if csv_path:
        write_dse_csv(csv_path, report.get("top_results", []))
