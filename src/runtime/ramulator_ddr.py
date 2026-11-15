"""Ramulator2 DDR memory timing model wrapper.

This module provides DDRTraceModel which wraps the Ramulator2 DRAM simulator
for cycle-accurate memory timing measurement.

Functionality:
  - Generates synthetic memory access traces (load/store requests)
  - Configures Ramulator2 with DRAM parameters (DDR5, timing, channel/rank)
  - Executes Ramulator2 to get cycle-accurate memory latency
  - Caches results to avoid re-running simulation for identical requests
  - Converts cycles to seconds using tck (time per clock)

This is used by local runtime trace generation when Ramulator-backed memory
timing is requested.

The memory timing from Ramulator2 is combined with logic timing (from
pim_micro_ops for BitAcc) in devices.PIM._bitacc_time_energy().
"""
import os
import tempfile
import subprocess
import math
import re
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DDR5_BITACC_IMPL_PATH = _PROJECT_ROOT / "ramulator2" / "src" / "dram" / "impl" / "DDR5-BITACC.cpp"
_ORG_PRESET_RE = re.compile(
    r'\{"(?P<name>DDR5_[^"]+)",\s*\{(?P<density>[^,]+),\s*(?P<dq>\d+),\s*\{(?P<counts>[^}]*)\}\}\}'
)


def _ceil_log2(value: int) -> int:
    value = max(1, int(value))
    return 0 if value <= 1 else int((value - 1).bit_length())


def _eval_cpp_int_expr(raw: str) -> int:
    expr = str(raw).strip()
    if not expr:
        raise ValueError("Empty C++ integer expression")
    if "<<" in expr:
        lhs, rhs = expr.split("<<", 1)
        return int(_eval_cpp_int_expr(lhs)) << int(_eval_cpp_int_expr(rhs))
    return int(expr, 0)


@lru_cache(maxsize=1)
def _load_ddr5_org_specs() -> dict[str, dict[str, int]]:
    if not _DDR5_BITACC_IMPL_PATH.exists():
        raise FileNotFoundError(f"DDR5 org preset source not found: {_DDR5_BITACC_IMPL_PATH}")
    text = _DDR5_BITACC_IMPL_PATH.read_text(encoding="utf-8")
    specs: dict[str, dict[str, int]] = {}
    for match in _ORG_PRESET_RE.finditer(text):
        counts = [_eval_cpp_int_expr(token) for token in match.group("counts").split(",")]
        if len(counts) != 6:
            continue
        specs[match.group("name")] = {
            "density": _eval_cpp_int_expr(match.group("density")),
            "dq": int(match.group("dq")),
            "channel": int(counts[0]),
            "rank": int(counts[1]),
            "bankgroup": int(counts[2]),
            "bank": int(counts[3]),
            "row": int(counts[4]),
            "column": int(counts[5]),
        }
    if not specs:
        raise ValueError(f"Failed to parse org_presets from {_DDR5_BITACC_IMPL_PATH}")
    return specs


def _get_ddr5_org_spec(dram_org: str) -> dict[str, int]:
    return dict(_load_ddr5_org_specs().get(str(dram_org), {}))


def ddr5_org_specs() -> dict[str, dict[str, int]]:
    return {name: dict(spec) for name, spec in _load_ddr5_org_specs().items()}


def ddr_dq_bits_for_org(dram_org: str) -> int:
    spec = _get_ddr5_org_spec(dram_org)
    return max(1, int(spec.get("dq", 8)))


def ddr_bank_groups_for_org(dram_org: str) -> int:
    spec = _get_ddr5_org_spec(dram_org)
    return max(1, int(spec.get("bankgroup", 8)))


def ddr_banks_per_group_for_org(dram_org: str) -> int:
    spec = _get_ddr5_org_spec(dram_org)
    return max(1, int(spec.get("bank", 4)))


def ddr_rows_for_org(dram_org: str) -> int:
    spec = _get_ddr5_org_spec(dram_org)
    if not spec:
        raise KeyError(f"unknown DDR5 org preset {dram_org}")
    return int(spec["row"])


def derive_ddr_logical_banks(dram_org: str, channel: int = 1, rank: int = 1) -> int:
    banks_per_rank = ddr_bank_groups_for_org(dram_org) * ddr_banks_per_group_for_org(dram_org)
    return max(1, int(channel)) * max(1, int(rank)) * max(1, int(banks_per_rank))


def derive_ddr_channel_width_bits(dram_org: str, chips_per_dimm: int, channel_width: int | None = None) -> int:
    if channel_width is not None and int(channel_width) > 0:
        return int(channel_width)
    return max(1, int(chips_per_dimm)) * ddr_dq_bits_for_org(dram_org)


def derive_ddr_burst_bytes(
    dram_org: str,
    chips_per_dimm: int,
    channel_width: int | None = None,
    internal_prefetch_size: int = 16,
) -> int:
    width_bits = derive_ddr_channel_width_bits(dram_org, chips_per_dimm, channel_width)
    return max(1, int(internal_prefetch_size) * int(width_bits) // 8)


def derive_ddr_row_bytes(dram_org: str, chips_per_dimm: int) -> int:
    spec = _get_ddr5_org_spec(dram_org)
    dq_bits = max(1, int(spec.get("dq", 8)))
    cols = max(1, int(spec.get("column", 1 << 10)))
    chips = max(1, int(chips_per_dimm))
    return int(cols * chips * dq_bits / 8)


def derive_trace_address_layout(
    *,
    row_bytes: int,
    total_banks: int,
    bank_shift: int | None = None,
    row_shift: int | None = None,
) -> dict[str, int]:
    col_bits = _ceil_log2(max(1, int(row_bytes)))
    bank_bits = _ceil_log2(max(1, int(total_banks)))
    resolved_bank_shift = max(col_bits, int(bank_shift)) if bank_shift is not None else col_bits
    resolved_row_shift = (
        max(resolved_bank_shift + bank_bits, int(row_shift))
        if row_shift is not None
        else resolved_bank_shift + bank_bits
    )
    return {
        "col_bits": int(col_bits),
        "bank_bits": int(bank_bits),
        "bank_shift": int(resolved_bank_shift),
        "row_shift": int(resolved_row_shift),
    }


class DDRTraceModel:
    def __init__(self,
                 ramulator_bin="ramulator2/ramulator2",
                 dram_impl="DDR5",
                 dram_org="DDR5_8Gb_x8",
                 dram_timing="DDR5_3200AN",
                 channel=1,
                 rank=1,
                 chips_per_dimm=8,
                 channel_width=None,
                 tck_ns=0.625,
                 max_addr=2147483648,
                 trace_granularity=4096,
                 host_read_ratio=0.7,
                 pim_read_ratio=0.7,
                 ramulator_timeout_s=300):
        self.ramulator_bin = ramulator_bin
        self.dram_impl = dram_impl
        self.dram_org = dram_org
        self.dram_timing = dram_timing
        self.channel = max(1, int(channel))
        self.rank = max(1, int(rank))
        self.chips_per_dimm = max(1, int(chips_per_dimm))
        self.channel_width = derive_ddr_channel_width_bits(
            self.dram_org,
            self.chips_per_dimm,
            channel_width,
        )
        self.burst_bytes = derive_ddr_burst_bytes(
            self.dram_org,
            self.chips_per_dimm,
            self.channel_width,
        )
        self.row_bytes = derive_ddr_row_bytes(self.dram_org, self.chips_per_dimm)
        self.logical_banks = derive_ddr_logical_banks(
            self.dram_org,
            self.channel,
            self.rank,
        )
        self.tck_ns = tck_ns
        self.max_addr = max_addr
        self.trace_granularity = trace_granularity
        self.host_read_ratio = host_read_ratio
        self.pim_read_ratio = pim_read_ratio
        self.ramulator_timeout_s = ramulator_timeout_s
        self._cache = {}

        if not os.path.exists(self.ramulator_bin):
            raise FileNotFoundError(f"ramulator2 binary not found: {self.ramulator_bin}")

    def _write_trace(self, trace_path, n_req, read_ratio):
        line_size = self.burst_bytes
        n_read = int(math.ceil(n_req * read_ratio))
        with open(trace_path, "w") as f:
            for i in range(n_req):
                op = "LD" if i < n_read else "ST"
                addr = (i * line_size) % self.max_addr
                f.write(f"{op} 0x{addr:x}\n")

    def _write_yaml(self, yaml_path, trace_path):
        yaml = []
        yaml.append("Frontend:")
        yaml.append("  impl: LoadStoreTrace")
        yaml.append("  path: {}".format(trace_path))
        yaml.append("  clock_ratio: 1")
        yaml.append("")
        yaml.append("Translation:")
        yaml.append("  impl: NoTranslation")
        yaml.append("  max_addr: {}".format(self.max_addr))
        yaml.append("")
        yaml.append("MemorySystem:")
        yaml.append("  impl: GenericDRAM")
        yaml.append("  clock_ratio: 1")
        yaml.append("")
        yaml.append("  DRAM:")
        yaml.append("    impl: {}".format(self.dram_impl))
        yaml.append("    org:")
        yaml.append("      preset: {}".format(self.dram_org))
        yaml.append("      chips: {}".format(int(self.chips_per_dimm)))
        yaml.append("      channel_width: {}".format(int(self.channel_width)))
        yaml.append("      channel: {}".format(self.channel))
        yaml.append("      rank: {}".format(self.rank))
        yaml.append("    timing:")
        yaml.append("      preset: {}".format(self.dram_timing))
        yaml.append("    RFM:")
        yaml.append("      BRC: 2")
        yaml.append("")
        yaml.append("  Controller:")
        yaml.append("    impl: Generic")
        yaml.append("    Scheduler:")
        yaml.append("      impl: FRFCFS")
        yaml.append("    RefreshManager:")
        yaml.append("      impl: AllBank")
        yaml.append("    plugins:")
        yaml.append("")
        yaml.append("  AddrMapper:")
        yaml.append("    impl: RoBaRaCoCh")
        with open(yaml_path, "w") as f:
            f.write("\n".join(yaml) + "\n")

    def _run_ramulator(self, yaml_path):
        try:
            result = subprocess.run(
                [self.ramulator_bin, "-f", yaml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=self.ramulator_timeout_s if self.ramulator_timeout_s and self.ramulator_timeout_s > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Ramulator2 timed out after {self.ramulator_timeout_s}s for config {yaml_path}"
            ) from exc
        cycles = None
        # Parse from both stdout and stderr; different Ramulator builds/loggers may differ.
        merged = (result.stdout or "") + "\n" + (result.stderr or "")
        for line in merged.splitlines():
            if "memory_system_cycles" not in line:
                continue
            m = re.search(r"memory_system_cycles\s*[:=]?\s*([0-9][0-9,]*)", line)
            if m is not None:
                try:
                    cycles = int(m.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass
        if cycles is None:
            tail = "\n".join(merged.splitlines()[-40:])
            raise RuntimeError(
                "Failed to parse memory_system_cycles from ramulator output "
                f"(returncode={result.returncode}). Tail:\n{tail}"
            )
        return cycles

    def timing_for_bytes(self, total_bytes, read_ratio, max_trace_requests=0):
        if total_bytes <= 0:
            return {
                "time_s": 0.0,
                "trace_out_cycles": 0.0,
                "ramulator_raw_cycles": 0,
                "trace_requests_total": 0,
                "trace_requests_emitted": 0,
                "trace_scale_factor": 1.0,
                "command_granularity_bytes": self.burst_bytes,
            }
        max_trace_requests = int(max_trace_requests or 0)
        key = (int(total_bytes), round(float(read_ratio), 6), max_trace_requests)
        if key in self._cache:
            return self._cache[key]
        total_requests = max(1, int(math.ceil(total_bytes / self.burst_bytes)))
        emitted_requests = total_requests
        if max_trace_requests > 0:
            emitted_requests = min(total_requests, max(1, max_trace_requests))
        scale = total_requests / max(1, emitted_requests)
        with tempfile.TemporaryDirectory() as td:
            trace_path = os.path.join(td, "mem.trace")
            yaml_path = os.path.join(td, "mem.yaml")
            self._write_trace(trace_path, emitted_requests, read_ratio)
            self._write_yaml(yaml_path, trace_path)
            cycles = self._run_ramulator(yaml_path)
        scaled_cycles = float(cycles) * float(scale)
        timing = {
            "time_s": scaled_cycles * (self.tck_ns * 1e-9),
            "trace_out_cycles": scaled_cycles,
            "ramulator_raw_cycles": int(cycles),
            "trace_requests_total": int(total_requests),
            "trace_requests_emitted": int(emitted_requests),
            "trace_scale_factor": float(scale),
            "command_granularity_bytes": int(self.burst_bytes),
        }
        self._cache[key] = timing
        return timing

    def time_for_bytes(self, total_bytes, read_ratio):
        return float(self.timing_for_bytes(total_bytes, read_ratio)["time_s"])
