from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .types import CPUConfig, HardwareConfig, ModelConfig, PIMConfig, RamulatorConfig, RunConfig
from .runtime.ramulator_ddr import (
    ddr5_org_specs,
    ddr_banks_per_group_for_org,
    ddr_bank_groups_for_org,
    ddr_dq_bits_for_org,
    derive_ddr_burst_bytes,
    derive_ddr_row_bytes,
    ddr_rows_for_org,
)


ROOT = Path(__file__).resolve().parents[1]


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + max(1, int(b)) - 1) // max(1, int(b))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _maybe_load_json(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _local_ramulator_bin(preferred: str | Path | None) -> str | None:
    candidates = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates.extend(
        [
            ROOT / "ramulator2" / "ramulator2",
            ROOT / "ramulator2" / "build" / "ramulator2",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _add_ramulator_library_path(ramulator_bin: str | Path) -> None:
    bin_path = Path(ramulator_bin)
    candidates = [bin_path.parent, bin_path.parent.parent]
    lib_dirs = [str(path) for path in candidates if (path / "libramulator.so").exists()]
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(os.pathsep) if part]
    merged = lib_dirs + [part for part in current_parts if part not in lib_dirs]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(merged)


def _modelinfos(model: ModelConfig) -> dict[str, Any]:
    hidden = int(model.hidden_size)
    return {
        "name": str(model.name).upper(),
        "ndec": int(model.n_layers),
        "hdim": hidden,
        "num_heads": int(model.num_heads),
        "dhead": int(model.head_dim),
        "ff_scale": float(model.ffn_size) / float(max(1, hidden)),
        "gqa_size": 1,
        "num_key_value_heads": int(model.num_heads),
        "vocab_size": int(model.vocab_size),
        "dtype": "W2A8",
    }


def _ddr_config(ramulator: RamulatorConfig) -> dict[str, Any]:
    channel_width = _trace_channel_width(ramulator)
    return {
        "impl": str(ramulator.impl),
        "org": str(ramulator.org),
        "chips_per_dimm": int(ramulator.chips_per_dimm),
        "channel_width": channel_width,
        "timing": str(ramulator.timing),
        "channel": int(ramulator.channel),
        "rank": int(ramulator.rank),
        "tck_ns": float(ramulator.tck_ns),
    }


def _trace_channel_width(ramulator: RamulatorConfig) -> int | None:
    channel_width = int(ramulator.channel_width)
    default_width = int(ramulator.chips_per_dimm) * int(ramulator.dq_bits)
    if channel_width == default_width:
        return None
    return channel_width


def _controller_io_bits_per_clock(ramulator: RamulatorConfig) -> int:
    """Return one full controller-clock transfer, including both DDR edges."""

    edge_bits = int(ramulator.channel_width)
    if edge_bits <= 0:
        edge_bits = int(ramulator.chips_per_dimm) * int(ramulator.dq_bits)
    return 2 * edge_bits


def _pim_config(
    model: ModelConfig,
    hardware: HardwareConfig,
    cpu: CPUConfig,
    pim: PIMConfig | None,
    ramulator: RamulatorConfig,
    pim_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .runtime.bitacc_layout import apply_bitacc_pe_config
    from .runtime.pim_micro_ops import rtl_bank_pe as rtl_bank_ops
    from .runtime.pim_micro_ops import rtl_dimm_pe as rtl_dimm_ops

    pim_config = _maybe_load_json(ROOT / "config" / "pim" / "bitacc_runtime_default.json")
    pim_config.update(pim_runtime or {})
    pe_config = _maybe_load_json(ROOT / "config" / "pe" / "current_pe.json")
    ddr_config = _ddr_config(ramulator)

    pe_config["banks"] = int(hardware.num_banks)
    pe_config["n_tile"] = int(hardware.n_channels_per_bank)
    pe_config["pairs_per_rd"] = int(hardware.k_tile) * int(hardware.n_channels_per_bank)
    pe_config["tok_par"] = int(hardware.qkt_tok_parallel)
    pe_config["tq_par"] = int(hardware.qkt_tok_parallel)
    pe_config["tp_par"] = int(hardware.tp_parallel)
    pe_config["qkt_dot_par"] = int(hardware.qkt_dot_parallel)
    pe_config["pv_dim_par"] = int(hardware.pv_dim_parallel)
    pe_config["wio"] = int(hardware.bank_io_bits)
    modelinfos = _modelinfos(model)
    apply_bitacc_pe_config(pim_config, pe_config, modelinfos, ddr_config)

    ddr_logical_banks = (
        ddr_bank_groups_for_org(ramulator.org) * ddr_banks_per_group_for_org(ramulator.org)
    )
    trace_banks = max(1, int(hardware.num_banks))
    if trace_banks != ddr_logical_banks:
        raise ValueError(
            f"bank mismatch: hardware.num_banks={trace_banks}, {ramulator.org} "
            f"provides {ddr_logical_banks} banks/rank"
        )
    controller_io_bits = _controller_io_bits_per_clock(ramulator)
    pim_config["BITACC_TRACE_IO_BITS"] = controller_io_bits
    pim_config["controller_io_bits_per_edge"] = controller_io_bits // 2
    attention_parallel_heads = max(1, min(int(model.num_heads), trace_banks))
    pim_config["BITACC_LOGICAL_BANKS"] = trace_banks
    pim_config["BITACC_NUM_BANKS"] = trace_banks
    pim_config["BITACC_TRACE_BANKS"] = trace_banks
    pim_config["bank_count_per_chip"] = trace_banks
    pim_config["BITACC_ATTN_PARALLEL_HEADS"] = attention_parallel_heads
    pim_config["BITACC_TRACE_ATTN_PARALLEL_HEADS"] = attention_parallel_heads
    pim_config["BITACC_TRACE_MAX_PARALLEL_HEADS"] = attention_parallel_heads
    pim_config["BITACC_N_CHANNELS_PER_BANK"] = max(1, _ceil_div(int(model.hidden_size), trace_banks))
    pim_config["BITACC_TRACE_QKV_LOCAL_COLS"] = max(1, 3 * int(model.head_dim))
    pim_config["BITACC_TRACE_WO_LOCAL_COLS"] = max(1, int(model.head_dim))
    pim_config["BITACC_TRACE_DOWN_LOCAL_COLS"] = max(1, _ceil_div(int(model.hidden_size), trace_banks))
    pim_config["BITACC_TRACE_MLP_LOCAL_COLS"] = max(1, _ceil_div(int(model.ffn_size), trace_banks))
    pim_config["BITACC_TRACE_MLP_DIM"] = int(model.ffn_size)
    pim_config["BITACC_CHIP_COUNT"] = int(ramulator.chips_per_dimm)
    cpu_clock_hz = float(cpu.clock_hz) if float(cpu.clock_hz) > 0 else float(hardware.host_clock_hz)
    effective_cpu_ops = (
        float(cpu.flops_per_s)
        * max(0.0, float(cpu.compute_util))
        * max(1, int(cpu.num_cpu))
    )
    if effective_cpu_ops <= 0:
        raise ValueError("CPU effective arithmetic throughput must be positive")
    pim_config["BITACC_CPU_CLOCK_HZ"] = cpu_clock_hz
    pim_config["BITACC_CPU_ARITHMETIC_OPS_PER_SECOND"] = effective_cpu_ops
    pim_config.setdefault("input_scalars_per_lut_tile", 2 * int(hardware.k_tile))
    pim_config["lut_pairs_per_dimm_clock"] = int(hardware.lut_pair_gen_per_cycle)
    pim_config["rank_io_bits_per_clock"] = controller_io_bits

    if pim is not None:
        pim_config["dram_share_host"] = float(pim.dram_share_host)
        pim_config["dram_share_pim"] = float(pim.dram_share_pim)
        pim_config["total_power_mw"] = float(pim.static_power_mw)

    controller_mhz = float(pim_config["controller_clock_hz"]) / 1_000_000.0
    bank_pe_mhz = float(pim_config["bank_pe_clock_hz"]) / 1_000_000.0
    dimm_pe_mhz = float(pim_config["dimm_pe_clock_hz"]) / 1_000_000.0
    pim_config["BITACC_TRACE_CTRL_FREQ_MHZ"] = controller_mhz
    pim_config["BITACC_TRACE_BANK_PE_FREQ_MHZ"] = bank_pe_mhz
    pim_config["BITACC_TRACE_DIMM_PE_FREQ_MHZ"] = dimm_pe_mhz
    pim_config["BITACC_TRACE_PE_FREQ_MHZ"] = dimm_pe_mhz
    pim_config["BITACC_TRACE_MEM_FREQ_MHZ"] = controller_mhz
    pim_config["BITACC_TRACE_LAUNCH_PE_CYCLES"] = int(rtl_bank_ops.TERNARY_PAIR_PATH_CYCLES)
    pim_config["BITACC_TRACE_QKV_READY_TAIL_PE_CYCLES"] = int(rtl_bank_ops.launch_to_dequant_cycles())
    pim_config["BITACC_TRACE_NORM_CTRL_PE_CYCLES"] = int(rtl_dimm_ops.NORM_CTRL_CYCLES)
    pim_config["BITACC_TRACE_DEQUANT_PE_CYCLES"] = int(pim_config["dequant_cycles"])
    pim_config["BITACC_TRACE_ROPE_CTRL_PE_CYCLES"] = int(pim_config["rope_ctrl_cycles"])
    pim_config["BITACC_TRACE_QKT_CTRL_PE_CYCLES"] = int(pim_config["qkt_ctrl_cycles"])
    pim_config["BITACC_TRACE_POST_CTRL_PE_CYCLES"] = int(rtl_dimm_ops.POST_CTRL_CYCLES)
    pim_config["BITACC_TRACE_PV_CTRL_PE_CYCLES"] = int(pim_config["pv_ctrl_cycles"])
    pim_config["BITACC_TRACE_STARTUP_PE_CYCLES"] = int(rtl_dimm_ops.STARTUP_CYCLES)
    pim_config["BITACC_TRACE_CFG_PE_CYCLES"] = int(rtl_dimm_ops.CFG_CYCLES)
    pim_config["BITACC_TRACE_MLP_FUSION_CTRL_PE_CYCLES"] = int(rtl_dimm_ops.MLP_FUSION_CTRL_CYCLES)
    return pim_config


def _ramulator_with_org(ramulator: RamulatorConfig, org: str) -> RamulatorConfig:
    dq_bits = ddr_dq_bits_for_org(org)
    channel_width = int(ramulator.chips_per_dimm) * dq_bits
    burst_bytes = derive_ddr_burst_bytes(
        org,
        ramulator.chips_per_dimm,
        channel_width,
        ramulator.internal_prefetch_size,
    )
    bandwidth = (
        int(ramulator.channel)
        * burst_bytes
        / (max(1, int(ramulator.nBL)) * float(ramulator.tck_ns) * 1e-9)
    )
    return replace(
        ramulator,
        org=org,
        dq_bits=dq_bits,
        channel_width=channel_width,
        burst_bytes=burst_bytes,
        bandwidth_bytes_per_s=bandwidth,
    )


def resolve_ramulator_for_workload(
    *,
    model: ModelConfig,
    run: RunConfig,
    hardware: HardwareConfig,
    cpu: CPUConfig,
    pim: PIMConfig | None,
    ramulator: RamulatorConfig,
    pim_runtime: dict[str, Any] | None = None,
) -> tuple[RamulatorConfig, HardwareConfig, dict[str, int]]:
    from .runtime.latest_command_trace import derive_latest_address_requirements

    if int(run.lout) != 1:
        raise ValueError(
            f"BitAcc decode trace requires lout=1, got {int(run.lout)}; run one output token at a time"
        )
    if int(run.lin) <= 0:
        raise ValueError(f"lin must be positive cached-KV token count, got {int(run.lin)}")
    if int(run.batch) != 1:
        raise ValueError(f"latest BitAcc trace currently requires batch=1, got {int(run.batch)}")
    if int(ramulator.chips_per_dimm) <= 0:
        raise ValueError("chips_per_dimm must be positive")
    fixed_topology = {
        "channel": int(ramulator.channel),
        "rank": int(ramulator.rank),
        "chips_per_dimm": int(ramulator.chips_per_dimm),
        "dq_bits": int(ramulator.dq_bits),
        "banks_per_chip": int(hardware.num_banks),
        "bank_array_bits": int(hardware.bank_io_bits),
        "k_tile_pairs": int(hardware.k_tile),
        "outputs_per_bank_pe": int(hardware.n_channels_per_bank),
    }
    expected_topology = {
        "channel": 1,
        "rank": 1,
        "chips_per_dimm": 8,
        "dq_bits": 8,
        "banks_per_chip": 32,
        "bank_array_bits": 128,
        "k_tile_pairs": 32,
        "outputs_per_bank_pe": 1,
    }
    if fixed_topology != expected_topology:
        raise ValueError(
            "v14 requires fixed 1-DIMM/1-rank topology: "
            "1 channel, 8 x8 chips, 32 banks/chip, 128-bit Bank PE, 32 pairs; "
            f"got {fixed_topology}"
        )

    def requirements_for(candidate: RamulatorConfig) -> dict[str, int]:
        physical_banks = (
            ddr_bank_groups_for_org(candidate.org)
            * ddr_banks_per_group_for_org(candidate.org)
        )
        if physical_banks != int(hardware.num_banks):
            raise ValueError(
                f"bank mismatch: configured {hardware.num_banks}, {candidate.org} has "
                f"{physical_banks} banks/rank"
            )
        pim_config = _pim_config(model, hardware, cpu, pim, candidate, pim_runtime)
        pim_config["BITACC_KV_CACHE_MAX_TOKENS"] = max(
            int(pim_config.get("BITACC_KV_CACHE_MAX_TOKENS", 0)),
            int(run.lin) + 1,
        )
        return derive_latest_address_requirements(
            modelinfos=_modelinfos(model),
            pim_config=pim_config,
            lin=int(run.lin),
            row_bytes=derive_ddr_row_bytes(candidate.org, candidate.chips_per_dimm),
        )

    if str(ramulator.org).lower() != "auto":
        selected = ramulator
        requirements = requirements_for(selected)
        row_capacity = ddr_rows_for_org(selected.org)
        if int(requirements["required_max_row"]) >= row_capacity:
            raise ValueError(
                f"explicit {selected.org} has rows 0..{row_capacity - 1}, but {model.name} "
                f"lin={run.lin} requires row {requirements['required_max_row']}"
            )
    else:
        candidates = []
        for name, spec in ddr5_org_specs().items():
            if int(spec["dq"]) != int(ramulator.dq_bits):
                continue
            if int(spec["bankgroup"]) * int(spec["bank"]) != int(hardware.num_banks):
                continue
            candidates.append((int(spec["density"]), name))
        candidates.sort()
        if not candidates:
            raise ValueError(
                f"no DDR5 preset matches dq={ramulator.dq_bits} and banks={hardware.num_banks}"
            )

        selected = None
        requirements = None
        attempted = []
        for _density, name in candidates:
            candidate = _ramulator_with_org(ramulator, name)
            candidate_requirements = requirements_for(candidate)
            row_capacity = ddr_rows_for_org(name)
            attempted.append(
                f"{name}: requires row {candidate_requirements['required_max_row']}, "
                f"capacity row {row_capacity - 1}"
            )
            if int(candidate_requirements["required_max_row"]) < row_capacity:
                selected = candidate
                requirements = candidate_requirements
                break
        if selected is None or requirements is None:
            raise ValueError("no DDR5 preset can hold workload; " + "; ".join(attempted))

    resolved_hardware = replace(
        hardware,
        dram_bandwidth_bytes_per_s=float(selected.bandwidth_bytes_per_s),
        bank_io_bits=128,
    )
    return selected, resolved_hardware, requirements


def _rewrite_yaml_trace_path(path: Path, trace_path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^(\s*path:\s*).*/trace_out\.trace\s*$",
        lambda match: f"{match.group(1)}{trace_path.resolve()}",
        text,
    )
    text = re.sub(
        r"(?ms)^      - ControllerPlugin:\n          impl: CommandCounter\n.*?(?=^      - ControllerPlugin:|^  AddrMapper:)",
        "",
        text,
    )
    path.write_text(text, encoding="utf-8")


def write_token_lifecycle_trace(
    *,
    model: ModelConfig,
    run: RunConfig,
    hardware: HardwareConfig,
    cpu: CPUConfig,
    ramulator: RamulatorConfig,
    pim: PIMConfig | None,
    result_dir: str | Path,
    ramulator_bin: str | Path | None,
    timeout_s: int = 3600,
    pim_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the command-level token lifecycle trace with local runtime code."""
    from .runtime.latest_command_trace import generate_latest_command_trace

    resolved_bin = _local_ramulator_bin(ramulator_bin)
    if resolved_bin is None:
        raise FileNotFoundError(
            "no local Ramulator2 binary found; build bitacc_inference/ramulator2 first "
            "or pass --ramulator-bin inside bitacc_inference"
        )
    _add_ramulator_library_path(resolved_bin)

    mem_model = SimpleNamespace(
        ramulator_bin=resolved_bin,
        ramulator_timeout_s=int(timeout_s),
        dram_impl=str(ramulator.impl),
        dram_org=str(ramulator.org),
        dram_timing=str(ramulator.timing),
        channel=int(ramulator.channel),
        rank=int(ramulator.rank),
        chips_per_dimm=int(ramulator.chips_per_dimm),
        channel_width=_trace_channel_width(ramulator),
    )
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_pim_config = _pim_config(model, hardware, cpu, pim, ramulator, pim_runtime)
    runtime_pim_config["BITACC_KV_CACHE_MAX_TOKENS"] = max(
        int(runtime_pim_config.get("BITACC_KV_CACHE_MAX_TOKENS", 0)),
        int(run.lin) + 1,
    )

    summary = generate_latest_command_trace(
        modelinfos=_modelinfos(model),
        pim_config=runtime_pim_config,
        mem_model=mem_model,
        lin=int(run.lin),
        lout=int(run.lout),
        batch=int(run.batch),
        out_dir=str(out_dir),
        timeout_s=int(timeout_s),
        write_summary_json=True,
    )

    ramulator_trace = out_dir / "ramulator_trace.txt"
    ramulator_config = out_dir / "ramulator_config.yaml"
    aliases = {
        "trace_out.trace": ramulator_trace.name,
        "trace_out.yaml": ramulator_config.name,
    }
    for src_name, dst_name in aliases.items():
        src = out_dir / src_name
        dst = out_dir / dst_name
        if src.exists() and src != dst:
            shutil.copyfile(src, dst)
    _rewrite_yaml_trace_path(out_dir / "trace_out.yaml", ramulator_trace)
    _rewrite_yaml_trace_path(ramulator_config, ramulator_trace)

    estimates = summary.get("estimates", {})
    files = summary.get("files", {})
    return {
        "source": "token_lifecycle_trace_gen",
        "ramulator_bin": resolved_bin,
        "ramulator_input_trace": str(ramulator_trace),
        "ramulator_yaml": str(out_dir / "trace_out.yaml"),
        "trace_out_ch0": str(files.get("trace_out_ch0", out_dir / "trace_out.ch0")),
        "trace_out_cycles": int(estimates.get("total_cycles_to_next_token", 0) or 0),
        "ramulator_memory_cycles": int(
            estimates.get("full_decode_token_ramulator_memory_cycles")
            or estimates.get("representative_full_trace_ramulator_cycles")
            or estimates.get("one_transformer_block_ramulator_memory_cycles")
            or 0
        ),
        "summary": summary,
    }
