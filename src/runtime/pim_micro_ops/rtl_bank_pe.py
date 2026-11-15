"""RTL-aligned bank-level PE helpers for ``bank_level_pe_top``.

This module is the current source of truth for the bank-local timing model.
Compatibility ``pim_micro_ops`` modules (``blu.py``, ``acc.py``, ``attn_pe.py``,
``dqu.py``) can stay as compatibility shims, but new code should map cycle
assumptions through the current RTL names:

  - ``qkt_cache_engine``
  - ``pv_cache_engine``
  - ``bank_level_pe_top``

KV-cache writeback / read latency is intentionally not modeled here anymore.
Those accesses are assumed instantaneous at the RTL helper layer and should be
timed by Ramulator instead.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


FREQ_GHZ: float = 1.0

# bank_level_pe_top pipe landmarks
INPUT_BUFFER_STAGES: int = 4
TERNARY_PAIR_PATH_CYCLES: int = 1
PSUM_CORE_CYCLES: int = 1
DEQUANT_PIPE_CYCLES: int = 2

# Rope / cache engines
ROPE_CTRL_CYCLES: int = 9
QKT_CTRL_CYCLES: int = 1
PV_CTRL_CYCLES: int = 1
TOK_PAR_DEFAULT: int = 2
TP_PAR_DEFAULT: int = 2
QKT_DOT_PAR_DEFAULT: int = 8
PV_DIM_PAR_DEFAULT: int = 8


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(max(0, int(a)) / float(max(1, int(b)))))


def launch_to_dequant_cycles() -> int:
    return int(
        INPUT_BUFFER_STAGES
        + TERNARY_PAIR_PATH_CYCLES
        + PSUM_CORE_CYCLES
        + DEQUANT_PIPE_CYCLES
    )


def encode_ternary_lane(weight: int) -> int:
    weight = int(weight)
    if weight == 1:
        return 0b01
    if weight == -1:
        return 0b10
    if weight == 0:
        return 0b00
    raise ValueError(f"Invalid ternary weight {weight}; expected -1/0/1")


def decode_ternary_lane(code: int) -> int:
    code = int(code) & 0b11
    if code == 0b01:
        return 1
    if code == 0b10:
        return -1
    return 0


def encode_weight_pair(w_a: int, w_b: int) -> int:
    return encode_ternary_lane(w_a) | (encode_ternary_lane(w_b) << 2)


def decode_weight_pair(weight_4b: int) -> Tuple[int, int]:
    weight_4b = int(weight_4b) & 0xF
    return (
        decode_ternary_lane(weight_4b & 0b11),
        decode_ternary_lane((weight_4b >> 2) & 0b11),
    )


def lut_inputs_from_pair(a_val: int, b_val: int) -> Tuple[int, int, int, int]:
    """Return ``(lut_ab_p, lut_ab_m, lut_a, lut_b)``."""
    a_val = int(a_val)
    b_val = int(b_val)
    return (a_val + b_val, a_val - b_val, a_val, b_val)


def ternary_pair_eval(
    *,
    lut_ab_p: int,
    lut_ab_m: int,
    lut_a: int,
    lut_b: int,
    weight_4b: int,
) -> int:
    code_a, code_b = decode_weight_pair(weight_4b)
    if code_a == 0 and code_b == 0:
        return 0
    if code_a == 1 and code_b == 0:
        return int(lut_a)
    if code_a == -1 and code_b == 0:
        return -int(lut_a)
    if code_a == 0 and code_b == 1:
        return int(lut_b)
    if code_a == 0 and code_b == -1:
        return -int(lut_b)
    if code_a == 1 and code_b == 1:
        return int(lut_ab_p)
    if code_a == -1 and code_b == -1:
        return -int(lut_ab_p)
    if code_a == 1 and code_b == -1:
        return int(lut_ab_m)
    if code_a == -1 and code_b == 1:
        return -int(lut_ab_m)
    return 0


def rope_cycles(head_dim_local: int) -> int:
    return int(rope_phase_breakdown(head_dim_local)["total_cycles"])


def rope_phase_breakdown(head_dim_local: int,
                         ctrl_cycles: int = ROPE_CTRL_CYCLES) -> Dict[str, int]:
    pair_rot_cycles = ceil_div(head_dim_local, 2)
    total_cycles = pair_rot_cycles + max(0, int(ctrl_cycles))
    return {
        "engine": "rope_rotate",
        "head_dim_local": max(0, int(head_dim_local)),
        "pair_rot_cycles": int(pair_rot_cycles),
        "ctrl_cycles": int(max(0, int(ctrl_cycles))),
        "total_cycles": int(total_cycles),
    }


def qkt_read_cycles(seq_len: int, tok_par: int = TOK_PAR_DEFAULT) -> int:
    del seq_len, tok_par
    return 0


def qkt_compute_cycles(seq_len: int,
                       tok_par: int = TOK_PAR_DEFAULT,
                       *,
                       head_dim: int | None = None,
                       dot_par: int | None = None,
                       ctrl_cycles: int = QKT_CTRL_CYCLES) -> int:
    return int(
        qkt_phase_breakdown(
            seq_len,
            tok_par,
            head_dim=head_dim,
            dot_par=dot_par,
            ctrl_cycles=ctrl_cycles,
        )["total_cycles"]
    )


def qkt_phase_breakdown(seq_len: int,
                        tok_par: int = TOK_PAR_DEFAULT,
                        *,
                        head_dim: int | None = None,
                        dot_par: int | None = None,
                        ctrl_cycles: int = QKT_CTRL_CYCLES) -> Dict[str, Any]:
    seq_steps = ceil_div(seq_len, tok_par)
    dot_par_use = (
        max(1, int(dot_par))
        if dot_par is not None and int(dot_par) > 0
        else QKT_DOT_PAR_DEFAULT
    )
    inner_steps = 1
    if head_dim is not None:
        inner_steps = ceil_div(head_dim, dot_par_use)
    body_cycles = seq_steps * inner_steps
    total_cycles = body_cycles + max(0, int(ctrl_cycles))
    return {
        "engine": "qkt_cache_engine",
        "phase_order": ["ctrl", "token_block_dot_sweep"],
        "seq_len": max(0, int(seq_len)),
        "tok_par": max(1, int(tok_par)),
        "token_blocks": int(seq_steps),
        "head_dim": None if head_dim is None else max(0, int(head_dim)),
        "dot_par": int(dot_par_use),
        "cycles_per_token_block": int(inner_steps),
        "body_cycles": int(body_cycles),
        "ctrl_cycles": int(max(0, int(ctrl_cycles))),
        "kv_cache_cycles": 0,
        "total_cycles": int(total_cycles),
    }


def pv_read_cycles(seq_len: int, tok_par: int = TOK_PAR_DEFAULT) -> int:
    del seq_len, tok_par
    return 0


def pv_compute_cycles(seq_len: int,
                      tok_par: int = TOK_PAR_DEFAULT,
                      *,
                      head_dim: int | None = None,
                      tp_par: int | None = None,
                      pv_dim_par: int | None = None,
                      ctrl_cycles: int = PV_CTRL_CYCLES) -> int:
    return int(
        pv_phase_breakdown(
            seq_len,
            tok_par,
            head_dim=head_dim,
            tp_par=tp_par,
            pv_dim_par=pv_dim_par,
            ctrl_cycles=ctrl_cycles,
        )["total_cycles"]
    )


def pv_phase_breakdown(seq_len: int,
                       tok_par: int = TOK_PAR_DEFAULT,
                       *,
                       head_dim: int | None = None,
                       tp_par: int | None = None,
                       pv_dim_par: int | None = None,
                       ctrl_cycles: int = PV_CTRL_CYCLES) -> Dict[str, Any]:
    tp_par_use = (
        max(1, int(tp_par))
        if tp_par is not None and int(tp_par) > 0
        else max(1, int(tok_par)) if int(tok_par) > 0 else TP_PAR_DEFAULT
    )
    pv_dim_par_use = (
        max(1, int(pv_dim_par))
        if pv_dim_par is not None and int(pv_dim_par) > 0
        else PV_DIM_PAR_DEFAULT
    )
    dim_blocks = 1
    if head_dim is not None:
        dim_blocks = ceil_div(head_dim, tp_par_use)
    seq_sweeps = ceil_div(seq_len, pv_dim_par_use)
    body_cycles = dim_blocks * seq_sweeps
    total_cycles = body_cycles + max(0, int(ctrl_cycles))
    return {
        "engine": "pv_cache_engine",
        "phase_order": ["ctrl", "output_dim_block_seq_sweep"],
        "seq_len": max(0, int(seq_len)),
        "tok_par": max(1, int(tok_par)),
        "tp_par": int(tp_par_use),
        "pv_dim_par": int(pv_dim_par_use),
        "head_dim": None if head_dim is None else max(0, int(head_dim)),
        "output_dim_blocks": int(dim_blocks),
        "sequence_sweeps_per_dim_block": int(seq_sweeps),
        "body_cycles": int(body_cycles),
        "ctrl_cycles": int(max(0, int(ctrl_cycles))),
        "kv_cache_cycles": 0,
        "total_cycles": int(total_cycles),
    }


def bank_pipeline_cycles_per_launch() -> Dict[str, int]:
    return {
        "pair_path": int(TERNARY_PAIR_PATH_CYCLES),
        "psum": int(PSUM_CORE_CYCLES),
        "dequant": int(DEQUANT_PIPE_CYCLES),
        "launch_to_dequant": int(launch_to_dequant_cycles()),
    }
