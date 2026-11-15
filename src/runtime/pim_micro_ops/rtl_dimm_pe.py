"""RTL-aligned DIMM-level PE helpers for ``dimm_level_pe_top``."""

from __future__ import annotations

from typing import Any, Dict

from .rtl_bank_pe import ceil_div


FREQ_GHZ: float = 1.0

LINEAR_FRONTEND_RD_LATENCY: int = 4
STARTUP_CYCLES: int = 2
CFG_CYCLES: int = 1
NORM_CTRL_CYCLES: int = 1
POST_CTRL_CYCLES: int = 3
MLP_FUSION_CTRL_CYCLES: int = 2
QKT_POSTPROC_SWEEPS: int = 4
QKT_POSTPROC_RECIP_CYCLES: int = 1
QKT_POSTPROC_DONE_CYCLES: int = 1


def linear_frontend_formula(
    *,
    in_dim: int,
    out_dim: int,
    banks: int,
    n_tile: int,
    pairs_per_rd: int,
    local_cols_override: int = 0,
) -> Dict[str, int]:
    in_dim = max(1, int(in_dim))
    out_dim = max(1, int(out_dim))
    banks = max(1, int(banks))
    n_tile = max(1, int(n_tile))
    pairs_per_rd = max(1, int(pairs_per_rd))

    k_tile_pairs = ceil_div(pairs_per_rd, n_tile)
    tile_scalars = max(2, 2 * k_tile_pairs)
    tiles = ceil_div(in_dim, tile_scalars)
    local_cols = (
        max(1, int(local_cols_override))
        if int(local_cols_override) > 0
        else ceil_div(out_dim, banks)
    )
    groups_per_tile = ceil_div(local_cols, n_tile)
    launches = tiles * groups_per_tile
    load_cycles = tiles * k_tile_pairs
    launch_cycles = launches * LINEAR_FRONTEND_RD_LATENCY
    total_cycles = load_cycles + launch_cycles
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "banks": banks,
        "n_tile": n_tile,
        "pairs_per_rd": pairs_per_rd,
        "k_tile_pairs": k_tile_pairs,
        "tile_scalars": tile_scalars,
        "tiles": tiles,
        "local_cols": local_cols,
        "groups_per_tile": groups_per_tile,
        "launches": launches,
        "load_cycles": load_cycles,
        "launch_cycles": launch_cycles,
        "total_cycles": total_cycles,
    }


def linear_frontend_phase_breakdown(
    *,
    in_dim: int,
    out_dim: int,
    banks: int,
    n_tile: int,
    pairs_per_rd: int,
    local_cols_override: int = 0,
    drain_cycles: int = 0,
) -> Dict[str, Any]:
    formula = linear_frontend_formula(
        in_dim=in_dim,
        out_dim=out_dim,
        banks=banks,
        n_tile=n_tile,
        pairs_per_rd=pairs_per_rd,
        local_cols_override=local_cols_override,
    )
    total_with_drain = int(formula["total_cycles"]) + max(0, int(drain_cycles))
    return {
        **formula,
        "engine": "dimm_linear_frontend",
        "phase_order": ["lut_pair_load", "launch", "drain"],
        "drain_cycles": int(max(0, int(drain_cycles))),
        "total_with_drain_cycles": int(total_with_drain),
    }


def rmsnorm_cycles(vec_dim: int) -> int:
    return int(rmsnorm_phase_breakdown(vec_dim)["total_cycles"])


def rmsnorm_phase_breakdown(vec_dim: int) -> Dict[str, int]:
    vec_dim = max(0, int(vec_dim))
    total_cycles = (2 * vec_dim) + NORM_CTRL_CYCLES
    return {
        "engine": "dimm_residual_rmsnorm",
        "phase_order": ["sumsq_pass", "scale_out_pass"],
        "vec_dim": int(vec_dim),
        "sumsq_cycles": int(vec_dim),
        "scale_out_cycles": int(vec_dim),
        "ctrl_cycles": int(NORM_CTRL_CYCLES),
        "total_cycles": int(total_cycles),
    }


def qkt_postproc_cycles(seq_len: int, tok_par: int) -> int:
    return int(qkt_postproc_phase_breakdown(seq_len, tok_par)["total_cycles"])


def qkt_postproc_phase_breakdown(seq_len: int, tok_par: int) -> Dict[str, int]:
    steps = ceil_div(seq_len, tok_par)
    setup_cycles = int(max(0, POST_CTRL_CYCLES - QKT_POSTPROC_RECIP_CYCLES - QKT_POSTPROC_DONE_CYCLES))
    prep_cycles = int(steps)
    findmax_cycles = int(steps)
    exp_cycles = int(steps)
    recip_cycles = int(QKT_POSTPROC_RECIP_CYCLES)
    norm_cycles = int(steps)
    done_cycles = int(QKT_POSTPROC_DONE_CYCLES)
    total_cycles = setup_cycles + prep_cycles + findmax_cycles + exp_cycles + recip_cycles + norm_cycles + done_cycles
    return {
        "engine": "dimm_qkt_postproc",
        "phase_order": ["setup", "prep", "findmax", "exp", "recip", "norm", "done"],
        "seq_len": max(0, int(seq_len)),
        "tok_par": max(1, int(tok_par)),
        "token_blocks": int(steps),
        "setup_cycles": setup_cycles,
        "prep_cycles": prep_cycles,
        "findmax_cycles": findmax_cycles,
        "exp_cycles": exp_cycles,
        "recip_cycles": recip_cycles,
        "norm_cycles": norm_cycles,
        "done_cycles": done_cycles,
        "ctrl_cycles": int(POST_CTRL_CYCLES),
        "total_cycles": int(total_cycles),
    }


def mlp_fusion_cycles(vec_dim: int) -> int:
    return int(mlp_fusion_phase_breakdown(vec_dim)["total_cycles"])


def mlp_fusion_phase_breakdown(vec_dim: int) -> Dict[str, int]:
    vec_dim = max(0, int(vec_dim))
    total_cycles = (2 * vec_dim) + MLP_FUSION_CTRL_CYCLES
    return {
        "engine": "dimm_mlp_fusion",
        "phase_order": ["silu_pass", "gate_mul_pass"],
        "vec_dim": int(vec_dim),
        "silu_cycles": int(vec_dim),
        "gate_mul_cycles": int(vec_dim),
        "ctrl_cycles": int(MLP_FUSION_CTRL_CYCLES),
        "total_cycles": int(total_cycles),
    }
