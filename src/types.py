from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DataType(Enum):
    W2A8 = "W2A8"


class PhaseKind(Enum):
    NORM = "NORM"
    FC = "FC"
    KV_WRITEBACK = "KV_WRITEBACK"
    ATTN_PE = "ATTN_PE"
    SOFTMAX = "SOFTMAX"
    ACT = "ACT"
    GATHER = "GATHER"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    ffn_size: int
    vocab_size: int
    parameter_count: int | None = None
    dtype: DataType = DataType.W2A8


@dataclass(frozen=True)
class RamulatorConfig:
    impl: str
    org: str
    timing: str
    chips_per_dimm: int
    channel_width: int
    channel: int
    rank: int
    tck_ns: float
    internal_prefetch_size: int
    nBL: int
    dq_bits: int
    burst_bytes: int
    bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class HardwareConfig:
    clock_hz: float
    host_clock_hz: float
    dram_bandwidth_bytes_per_s: float
    num_banks: int
    n_channels_per_bank: int
    k_tile: int
    pairs_per_cycle: int
    vector_width: int
    qkt_tok_parallel: int
    qkt_dot_parallel: int
    pv_dim_parallel: int
    attention_parallel_heads: int
    stage_overlap: bool
    bank_io_bits: int = 1024
    acc_w_bits: int = 48
    fp16_w_bits: int = 16
    data_w_bits: int = 16
    launch_gap_cycles: int = 4
    drain_cycles: int = 16
    cpipe_qkv_cycles: int = 8
    lut_pair_gen_per_cycle: int = 1
    max_active_banks: int = 32
    tp_parallel: int = 2


@dataclass(frozen=True)
class CPUConfig:
    num_cpu: int
    num_core: int
    mem_cap_bytes: int
    mem_bw_bytes_per_s: float
    flops_per_s: float
    power_w: float
    compute_util: float = 1.0
    mem_util: float = 1.0
    nonoverlap_runtime_overhead_s: float = 0.0
    ramulator_max_trace_requests: int = 0
    ramulator_trace_read_ratio: float = 0.98
    clock_hz: float = 0.0


@dataclass(frozen=True)
class PIMConfig:
    enabled: bool
    num_attacc: int
    num_dram: int
    dram_share_host: float
    dram_share_pim: float
    power_constraint: bool
    static_power_mw: float
    logic_energy_pj_per_operation: float = 0.32
    include_placeholder_static_power: bool = False
    include_static_power: bool = False
    dram_activate_pj_per_bank_command: float | None = None
    dram_precharge_pj_per_bank_command: float | None = None
    dram_refresh_pj_per_bank_command: float | None = None
    sram_read_pj_per_byte: float = 0.0034
    sram_write_pj_per_byte: float = 0.0034
    bitlinear_energy_pj_per_operation: float = 0.32
    attention_mac_energy_pj_per_operation: float = 0.32
    lut_generation_energy_pj_per_operation: float = 0.32
    dram_background_power_w: float | None = None
    peak_power_budget_w: float | None = None
    static_power_source: str | None = None
    ddr5_idd_profile: dict[str, object] = field(default_factory=dict)
    pim_mode_dynamic_power_mw: dict[str, object] = field(default_factory=dict)
    pim_mode_power_source: str | None = None
    dynamic_crosscheck_relative_tolerance: float = 0.20


@dataclass(frozen=True)
class RunConfig:
    model: str
    lin: int
    lout: int
    batch: int


@dataclass(frozen=True)
class PhaseResult:
    index: int
    name: str
    kind: PhaseKind
    logic_cycles: int
    memory_cycles: int
    cycles: int
    bytes_moved: int


@dataclass(frozen=True)
class RuntimeOutputs:
    result_dir: Path
    input_trace: Path
    recorder_trace: Path
    ramulator_yaml: Path
    report_json: Path
    report_yaml: Path
