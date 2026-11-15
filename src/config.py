from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import CPUConfig, HardwareConfig, ModelConfig, PIMConfig, RamulatorConfig, RunConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.json"
DEFAULT_MODELS = ROOT / "config" / "models.json"

DDR5_ORG_PRESETS = {
    "DDR5_8GB_X4": {"dq": 4},
    "DDR5_8GB_X8": {"dq": 8},
    "DDR5_8GB_X16": {"dq": 16},
    "DDR5_16GB_X4": {"dq": 4},
    "DDR5_16GB_X8": {"dq": 8},
    "DDR5_16GB_X16": {"dq": 16},
    "DDR5_32GB_X4": {"dq": 4},
    "DDR5_32GB_X8": {"dq": 8},
    "DDR5_32GB_X16": {"dq": 16},
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _direct_or_section(data: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        section = data.get(name)
        if isinstance(section, dict):
            return section
    return data


def _ddr5_dq_bits(org: str) -> int:
    key = str(org).upper()
    if key not in DDR5_ORG_PRESETS:
        raise KeyError(f"unknown DDR5 org preset {org}")
    return int(DDR5_ORG_PRESETS[key]["dq"])


def _load_ramulator_config(raw: dict[str, Any]) -> RamulatorConfig:
    org = str(raw.get("org", "DDR5_8Gb_x8"))
    dq_bits = int(raw.get("dq_bits", 8)) if org.lower() == "auto" else _ddr5_dq_bits(org)
    chips = int(raw.get("chips_per_dimm", 8))
    channel_width_raw = raw.get("channel_width")
    channel_width = (
        int(channel_width_raw)
        if channel_width_raw is not None and int(channel_width_raw) > 0
        else chips * dq_bits
    )
    internal_prefetch_size = int(raw.get("internal_prefetch_size", 16))
    nbl = int(raw.get("nBL", raw.get("nbl", 8)))
    channel = int(raw.get("channel", 1))
    burst_bytes = max(1, internal_prefetch_size * channel_width // 8)
    tck_ns = float(raw.get("tck_ns", 0.625))
    bandwidth = channel * burst_bytes / (max(1, nbl) * tck_ns * 1e-9)
    return RamulatorConfig(
        impl=str(raw.get("impl", "DDR5-BITACC")),
        org=org,
        timing=str(raw.get("timing", "DDR5_3200AN")),
        chips_per_dimm=chips,
        channel_width=channel_width,
        channel=channel,
        rank=int(raw.get("rank", 1)),
        tck_ns=tck_ns,
        internal_prefetch_size=internal_prefetch_size,
        nBL=nbl,
        dq_bits=dq_bits,
        burst_bytes=burst_bytes,
        bandwidth_bytes_per_s=bandwidth,
    )


def _bytes_from_gb(raw: dict[str, Any], key: str, default_gb: float) -> int:
    if key in raw:
        return int(float(raw[key]))
    bytes_key = f"{key}_bytes"
    if bytes_key in raw:
        return int(float(raw[bytes_key]))
    gb_key = f"{key}_gb"
    if gb_key in raw:
        return int(float(raw[gb_key]) * 1024 * 1024 * 1024)
    return int(float(default_gb) * 1024 * 1024 * 1024)


def _bytes_per_s_from_gbps(raw: dict[str, Any], key: str, default_gbps: float) -> float:
    if key in raw:
        return float(raw[key])
    bytes_key = f"{key}_bytes_per_s"
    if bytes_key in raw:
        return float(raw[bytes_key])
    gbps_key = f"{key}_gbps"
    if gbps_key in raw:
        return float(raw[gbps_key]) * 1000 * 1000 * 1000
    return float(default_gbps) * 1000 * 1000 * 1000


def _load_cpu_config(raw: dict[str, Any] | None = None) -> CPUConfig:
    raw = raw or {}
    mem_cap_bytes = raw.get("memory_capacity_bytes", raw.get("mem_cap_bytes"))
    if mem_cap_bytes is None and "memory_capacity_gb" in raw:
        mem_cap_bytes = float(raw["memory_capacity_gb"]) * 1024 * 1024 * 1024
    if mem_cap_bytes is None:
        mem_cap_bytes = _bytes_from_gb(raw, "mem_cap", 24)

    mem_bw = raw.get(
        "cpu_mem_bw_bytes_per_s",
        raw.get("memory_bandwidth_bytes_per_s", raw.get("mem_bw_bytes_per_s")),
    )
    if mem_bw is None:
        mem_bw = _bytes_per_s_from_gbps(raw, "mem_bw", 100)

    effective_ops = raw.get("effective_ops_per_s", raw.get("effective_bitlinear_ops_per_s"))
    if effective_ops is None:
        effective_ops = raw.get(
            "flops",
            raw.get("FLOPS_PER_DEVICE", float(raw.get("flops_tflops", 1.0)) * 1e12),
        )

    return CPUConfig(
        num_cpu=int(raw.get("num_cpu", raw.get("NUM_DEVICE", 1))),
        num_core=int(raw.get("num_core", raw.get("NUM_CORE", 8))),
        mem_cap_bytes=int(float(mem_cap_bytes)),
        mem_bw_bytes_per_s=float(mem_bw),
        flops_per_s=float(effective_ops),
        power_w=float(raw.get("power_w", raw.get("tdp_w", 65.0))),
        compute_util=float(raw.get("compute_util", 1.0)),
        mem_util=float(raw.get("mem_util", 1.0)),
        nonoverlap_runtime_overhead_s=float(raw.get("nonoverlap_runtime_overhead_s", 0.0)),
        ramulator_max_trace_requests=int(raw.get("ramulator_max_trace_requests", 0) or 0),
        ramulator_trace_read_ratio=float(raw.get("ramulator_trace_read_ratio", 0.98)),
        clock_hz=float(raw.get("cpu_clock_hz", raw.get("clock_hz", 0.0)) or 0.0),
    )


def _load_pim_config(raw: dict[str, Any] | None = None) -> PIMConfig:
    raw = raw or {}
    static_power_mw = raw.get("static_power_mw", raw.get("total_power_mw"))
    if static_power_mw is None:
        bank_um2 = float(raw.get("bitacc_area_bank_level_ref_um2", raw.get("BITACC_AREA_BANK_LEVEL_REF_UM2", 288356.642067)))
        dimm_um2 = float(raw.get("bitacc_area_dimm_level_ref_um2", raw.get("BITACC_AREA_DIMM_LEVEL_REF_UM2", 1253567.962366)))
        bank_density = float(raw.get("bitacc_area_bank_power_density_mw_per_mm2", 350.0))
        dimm_density = float(raw.get("bitacc_area_dimm_power_density_mw_per_mm2", 250.0))
        static_power_mw = (bank_um2 / 1_000_000.0) * bank_density + (dimm_um2 / 1_000_000.0) * dimm_density

    def optional_float(key: str) -> float | None:
        value = raw.get(key)
        return None if value is None else float(value)

    logic_energy = float(raw.get("logic_energy_pj_per_operation", 0.32))
    idd_profile = raw.get("ddr5_idd_profile") or {}
    if not isinstance(idd_profile, dict):
        raise ValueError("ddr5_idd_profile must be a JSON object")
    mode_power = raw.get("pim_mode_dynamic_power_mw") or {}
    if not isinstance(mode_power, dict):
        raise ValueError("pim_mode_dynamic_power_mw must be a JSON object")
    return PIMConfig(
        enabled=bool(raw.get("enabled", True)),
        num_attacc=int(raw.get("num_attacc", raw.get("NUM_ATTACC", 8))),
        num_dram=int(raw.get("num_dram", raw.get("NUM_DRAM", 1))),
        dram_share_host=float(raw.get("dram_share_host", raw.get("DRAM_SHARE_HOST", 0.5))),
        dram_share_pim=float(raw.get("dram_share_pim", raw.get("DRAM_SHARE_PIM", 0.5))),
        power_constraint=bool(raw.get("power_constraint", raw.get("POWER_CONSTRAINT", False))),
        static_power_mw=float(static_power_mw),
        dram_activation_precharge_pj_per_bit=float(
            raw.get("dram_activation_precharge_pj_per_bit", 0.11)
        ),
        dram_cell_rw_pj_per_bit=float(raw.get("dram_cell_rw_pj_per_bit", 0.44)),
        module_transfer_pj_per_bit=float(raw.get("module_transfer_pj_per_bit", 2.54)),
        external_transfer_pj_per_bit=float(raw.get("external_transfer_pj_per_bit", 0.30)),
        logic_energy_pj_per_operation=logic_energy,
        include_placeholder_static_power=bool(raw.get("include_placeholder_static_power", False)),
        include_static_power=bool(
            raw.get(
                "include_static_power",
                raw.get("include_placeholder_static_power", False),
            )
        ),
        dram_activate_pj_per_bank_command=optional_float(
            "dram_activate_pj_per_bank_command"
        ),
        dram_precharge_pj_per_bank_command=optional_float(
            "dram_precharge_pj_per_bank_command"
        ),
        dram_refresh_pj_per_bank_command=optional_float(
            "dram_refresh_pj_per_bank_command"
        ),
        sram_read_pj_per_byte=float(raw.get("sram_read_pj_per_byte", 0.0034)),
        sram_write_pj_per_byte=float(raw.get("sram_write_pj_per_byte", 0.0034)),
        bitlinear_energy_pj_per_operation=float(
            raw.get("bitlinear_energy_pj_per_operation", logic_energy)
        ),
        attention_mac_energy_pj_per_operation=float(
            raw.get("attention_mac_energy_pj_per_operation", logic_energy)
        ),
        lut_generation_energy_pj_per_operation=float(
            raw.get("lut_generation_energy_pj_per_operation", logic_energy)
        ),
        dram_background_power_w=optional_float("dram_background_power_w"),
        peak_power_budget_w=optional_float("peak_power_budget_w"),
        static_power_source=(
            None if raw.get("static_power_source") is None else str(raw["static_power_source"])
        ),
        ddr5_idd_profile=dict(idd_profile),
        pim_mode_dynamic_power_mw=dict(mode_power),
        pim_mode_power_source=(
            None
            if raw.get("pim_mode_power_source") is None
            else str(raw["pim_mode_power_source"])
        ),
        dynamic_crosscheck_relative_tolerance=float(
            raw.get("dynamic_crosscheck_relative_tolerance", 0.20)
        ),
    )


def _apply_pim_overrides(hw_raw: dict[str, Any], pim_raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(hw_raw)
    mapping = {
        "banks": "num_banks",
        "BITACC_NUM_BANKS": "num_banks",
        "n_tile": "n_channels_per_bank",
        "BITACC_N_CHANNELS_PER_BANK": "n_channels_per_bank",
        "k_tile": "k_tile",
        "BITACC_K_TILE": "k_tile",
        "pairs_per_rd": "pairs_per_cycle",
        "BITACC_PAIRS_PER_RD": "pairs_per_cycle",
        "tq_par": "qkt_tok_parallel",
        "BITACC_TRACE_QKT_TOK_PAR": "qkt_tok_parallel",
        "qkt_dot_par": "qkt_dot_parallel",
        "BITACC_TRACE_QKT_DOT_PAR": "qkt_dot_parallel",
        "pv_dim_par": "pv_dim_parallel",
        "BITACC_TRACE_PV_DIM_PAR": "pv_dim_parallel",
        "tp_par": "tp_parallel",
        "BITACC_TRACE_TP_PAR": "tp_parallel",
        "tok_par": "attention_parallel_heads",
        "bitacc_stage_overlap": "stage_overlap",
        "BITACC_STAGE_OVERLAP": "stage_overlap",
        "wio": "bank_io_bits",
        "bank_io_bits": "bank_io_bits",
        "launch_gap_cycles": "launch_gap_cycles",
        "LAUNCH_GAP_CYCLES": "launch_gap_cycles",
        "drain_cycles": "drain_cycles",
        "DRAIN": "drain_cycles",
        "lut_pair_gen_per_cycle": "lut_pair_gen_per_cycle",
        "LUT_PAIR_GEN_PER_CYCLE": "lut_pair_gen_per_cycle",
        "cpipe_qkv_cycles": "cpipe_qkv_cycles",
    }
    for src, dst in mapping.items():
        if src in pim_raw:
            out[dst] = pim_raw[src]
    if "banks" in pim_raw:
        out["max_active_banks"] = pim_raw["banks"]
    if "BITACC_NUM_BANKS" in pim_raw:
        out["max_active_banks"] = pim_raw["BITACC_NUM_BANKS"]
    return out


def load_bundle(
    config_path: str | Path | None = None,
) -> tuple[RunConfig, HardwareConfig, RamulatorConfig, dict[str, ModelConfig]]:
    cfg = load_json(DEFAULT_CONFIG)
    if config_path:
        cfg = deep_update(cfg, load_json(config_path))

    run_raw = cfg.get("run", {})
    hw_raw = cfg.get("hardware", {})
    ramulator = _load_ramulator_config(cfg.get("ramulator", {}))
    models = load_models(DEFAULT_MODELS)

    run = RunConfig(
        model=str(run_raw.get("model", "BITNET-125M")).upper(),
        lin=int(run_raw.get("lin", 128)),
        lout=int(run_raw.get("lout", 16)),
        batch=int(run_raw.get("batch", 1)),
    )
    hardware = HardwareConfig(
        clock_hz=float(hw_raw["clock_hz"]),
        host_clock_hz=float(hw_raw["host_clock_hz"]),
        dram_bandwidth_bytes_per_s=ramulator.bandwidth_bytes_per_s,
        num_banks=int(hw_raw["num_banks"]),
        n_channels_per_bank=int(hw_raw["n_channels_per_bank"]),
        k_tile=int(hw_raw["k_tile"]),
        pairs_per_cycle=int(hw_raw["pairs_per_cycle"]),
        vector_width=int(hw_raw["vector_width"]),
        qkt_tok_parallel=int(hw_raw["qkt_tok_parallel"]),
        qkt_dot_parallel=int(hw_raw["qkt_dot_parallel"]),
        pv_dim_parallel=int(hw_raw["pv_dim_parallel"]),
        attention_parallel_heads=int(hw_raw["attention_parallel_heads"]),
        stage_overlap=bool(hw_raw["stage_overlap"]),
        bank_io_bits=int(hw_raw.get("bank_io_bits", 1024)),
        acc_w_bits=int(hw_raw.get("acc_w_bits", 32)),
        fp16_w_bits=int(hw_raw.get("fp16_w_bits", 16)),
        data_w_bits=int(hw_raw.get("data_w_bits", 16)),
        launch_gap_cycles=int(hw_raw.get("launch_gap_cycles", 4)),
        drain_cycles=int(hw_raw.get("drain_cycles", 16)),
        cpipe_qkv_cycles=int(hw_raw.get("cpipe_qkv_cycles", 8)),
        lut_pair_gen_per_cycle=int(hw_raw.get("lut_pair_gen_per_cycle", 1)),
        max_active_banks=int(hw_raw.get("max_active_banks", 32)),
        tp_parallel=int(hw_raw.get("tp_parallel", 2)),
    )
    return run, hardware, ramulator, models


def load_runtime_bundle(
    config_path: str | Path | None = None,
    dram_config_path: str | Path | None = None,
    cpu_config_path: str | Path | None = None,
    pim_config_path: str | Path | None = None,
) -> tuple[RunConfig, HardwareConfig, RamulatorConfig, CPUConfig, PIMConfig, dict[str, ModelConfig]]:
    cfg = load_json(DEFAULT_CONFIG)
    if config_path:
        cfg = deep_update(cfg, load_json(config_path))

    cpu_raw: dict[str, Any] = {}
    pim_raw: dict[str, Any] = {}
    if cpu_config_path:
        cpu_raw = _direct_or_section(load_json(cpu_config_path), "cpu", "CPU")
    if pim_config_path:
        pim_raw = _direct_or_section(load_json(pim_config_path), "pim", "PIM", "hardware")

    if dram_config_path:
        dram_raw = _direct_or_section(load_json(dram_config_path), "ramulator", "dram", "DRAM")
        cfg["ramulator"] = deep_update(cfg.get("ramulator", {}), dram_raw)

    run_raw = cfg.get("run", {})
    hw_raw = _apply_pim_overrides(cfg.get("hardware", {}), pim_raw)
    ramulator = _load_ramulator_config(cfg.get("ramulator", {}))
    models = load_models(DEFAULT_MODELS)

    run = RunConfig(
        model=str(run_raw.get("model", "BITNET-125M")).upper(),
        lin=int(run_raw.get("lin", 128)),
        lout=int(run_raw.get("lout", 16)),
        batch=int(run_raw.get("batch", 1)),
    )
    hardware = HardwareConfig(
        clock_hz=float(hw_raw["clock_hz"]),
        host_clock_hz=float(hw_raw["host_clock_hz"]),
        dram_bandwidth_bytes_per_s=ramulator.bandwidth_bytes_per_s,
        num_banks=int(hw_raw["num_banks"]),
        n_channels_per_bank=int(hw_raw["n_channels_per_bank"]),
        k_tile=int(hw_raw["k_tile"]),
        pairs_per_cycle=int(hw_raw["pairs_per_cycle"]),
        vector_width=int(hw_raw["vector_width"]),
        qkt_tok_parallel=int(hw_raw["qkt_tok_parallel"]),
        qkt_dot_parallel=int(hw_raw["qkt_dot_parallel"]),
        pv_dim_parallel=int(hw_raw["pv_dim_parallel"]),
        attention_parallel_heads=int(hw_raw["attention_parallel_heads"]),
        stage_overlap=bool(hw_raw["stage_overlap"]),
        bank_io_bits=int(hw_raw.get("bank_io_bits", 1024)),
        acc_w_bits=int(hw_raw.get("acc_w_bits", 32)),
        fp16_w_bits=int(hw_raw.get("fp16_w_bits", 16)),
        data_w_bits=int(hw_raw.get("data_w_bits", 16)),
        launch_gap_cycles=int(hw_raw.get("launch_gap_cycles", 4)),
        drain_cycles=int(hw_raw.get("drain_cycles", 16)),
        cpipe_qkv_cycles=int(hw_raw.get("cpipe_qkv_cycles", 8)),
        lut_pair_gen_per_cycle=int(hw_raw.get("lut_pair_gen_per_cycle", 1)),
        max_active_banks=int(hw_raw.get("max_active_banks", 32)),
        tp_parallel=int(hw_raw.get("tp_parallel", 2)),
    )
    return run, hardware, ramulator, _load_cpu_config(cpu_raw), _load_pim_config(pim_raw), models


def load_models(path: str | Path) -> dict[str, ModelConfig]:
    raw = load_json(path)
    models: dict[str, ModelConfig] = {}
    for name, item in raw.items():
        models[str(name).upper()] = ModelConfig(
            name=str(name).upper(),
            n_layers=int(item["n_layers"]),
            hidden_size=int(item["hidden_size"]),
            num_heads=int(item["num_heads"]),
            head_dim=int(item["head_dim"]),
            ffn_size=int(item["ffn_size"]),
            vocab_size=int(item["vocab_size"]),
            parameter_count=(
                None if item.get("parameter_count") is None else int(item["parameter_count"])
            ),
        )
    return models
