from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def ceil_div(a: int | float, b: int | float) -> int:
    return int((int(a) + max(1, int(b)) - 1) // max(1, int(b)))


@dataclass(frozen=True)
class CycleModelParams:
    d_model: int
    mlp_dim: int
    banks: int
    visible_tokens: int
    k_tile: int
    n_tile: int
    launch_cycles: int
    tq_par: int
    tp_par: int
    bank_fp16_lanes: int
    bank_io_bits: int
    acc_w_bits: int
    fp16_w_bits: int
    bank_deq_tail: int
    bank_wr2rd_turn: int = 1
    lut_pair_gen_per_cycle: int = 1


def c_lut(input_dim: int, params: CycleModelParams) -> int:
    tiles = ceil_div(max(1, int(input_dim)), 2 * max(1, int(params.k_tile)))
    per_tile = ceil_div(max(1, int(params.k_tile)), max(1, int(params.lut_pair_gen_per_cycle)))
    return int(tiles * per_tile)


def c_launch(input_dim: int, local_cols: int, projections: int, params: CycleModelParams) -> int:
    tiles = ceil_div(max(1, int(input_dim)), 2 * max(1, int(params.k_tile)))
    groups = ceil_div(max(1, int(local_cols)) * max(1, int(projections)), max(1, int(params.n_tile)))
    return int(tiles * groups * max(1, int(params.launch_cycles)))


def c_cat(vec_dim: int, params: CycleModelParams) -> int:
    banks = max(1, int(params.banks))
    local_elems = ceil_div(max(1, int(vec_dim)), banks)
    lanes = max(1, int(params.bank_fp16_lanes))
    bus_groups = max(1, int(params.bank_io_bits) // max(1, lanes * int(params.fp16_w_bits)))
    return int(1 + ceil_div(local_elems, lanes) * ceil_div(banks, bus_groups))


def transformer_13_phase_cycle_model(params: CycleModelParams) -> list[dict[str, Any]]:
    d = max(1, int(params.d_model))
    m = max(1, int(params.mlp_dim))
    banks = max(1, int(params.banks))
    h = ceil_div(d, banks)
    seq = max(1, int(params.visible_tokens))
    tq_par = max(1, int(params.tq_par))
    tp_par = max(1, int(params.tp_par))
    lanes = max(1, int(params.bank_fp16_lanes))
    bank_io_bits = max(1, int(params.bank_io_bits))
    acc_w = max(1, int(params.acc_w_bits))
    fp16_w = max(1, int(params.fp16_w_bits))

    m_local = ceil_div(m, banks)
    qkt_steps = ceil_div(seq, tq_par)
    pv_steps = ceil_div(seq, tp_par)

    rows = [
        (
            "pre_attn_rmsnorm",
            "2d + 2",
            {"d": d},
            2 * d + 2,
        ),
        (
            "qkv_lut_generation_pe_launch",
            "C_lut(d) + C_launch(d,h,3)",
            {"C_lut(d)": c_lut(d, params), "C_launch(d,h,3)": c_launch(d, h, 3, params)},
            c_lut(d, params) + c_launch(d, h, 3, params),
        ),
        (
            "qkv_dequant_rope_kv_write",
            "BANK_DEQ_TAIL + 4ceil(h/BANK_FP16_LANES)",
            {
                "BANK_DEQ_TAIL": int(params.bank_deq_tail),
                "h": h,
                "BANK_FP16_LANES": lanes,
            },
            int(params.bank_deq_tail) + 4 * ceil_div(h, lanes),
        ),
        (
            "qkt_score",
            "ceil((KV+1)/TQ_PAR) * ceil(TQ_PAR*h/BANK_FP16_LANES) + 2 + BANK_WR2RD_TURN",
            {
                "KV+1": seq,
                "TQ_PAR": tq_par,
                "h": h,
                "BANK_FP16_LANES": lanes,
                "BANK_WR2RD_TURN": int(params.bank_wr2rd_turn),
            },
            qkt_steps * ceil_div(tq_par * h, lanes) + 2 + int(params.bank_wr2rd_turn),
        ),
        (
            "score_transfer_post_softmax",
            "ceil((KV+1)/TQ_PAR)*ceil(B*TQ_PAR*ACC_W/BANK_IO_BITS) + 1 + 4ceil((KV+1)/TP_PAR) + 3",
            {
                "KV+1": seq,
                "B": banks,
                "TQ_PAR": tq_par,
                "TP_PAR": tp_par,
                "ACC_W": acc_w,
                "BANK_IO_BITS": bank_io_bits,
            },
            qkt_steps * ceil_div(banks * tq_par * acc_w, bank_io_bits)
            + 1
            + 4 * pv_steps
            + 3,
        ),
        (
            "probability_transfer_pv_context",
            "ceil((KV+1)/TP_PAR)*ceil(B*TP_PAR*FP16_W/BANK_IO_BITS) + 1 + ceil((KV+1)/TP_PAR)*ceil(TP_PAR*h/BANK_FP16_LANES) + 1",
            {
                "KV+1": seq,
                "B": banks,
                "TP_PAR": tp_par,
                "FP16_W": fp16_w,
                "BANK_IO_BITS": bank_io_bits,
                "h": h,
                "BANK_FP16_LANES": lanes,
            },
            pv_steps * ceil_div(banks * tp_par * fp16_w, bank_io_bits)
            + 1
            + pv_steps * ceil_div(tp_par * h, lanes)
            + 1,
        ),
        (
            "attention_concat",
            "C_cat(d)",
            {"C_cat(d)": c_cat(d, params)},
            c_cat(d, params),
        ),
        (
            "wo_projection",
            "C_lut(d) + C_launch(d,h,1)",
            {"C_lut(d)": c_lut(d, params), "C_launch(d,h,1)": c_launch(d, h, 1, params)},
            c_lut(d, params) + c_launch(d, h, 1, params),
        ),
        (
            "pre_mlp_rmsnorm",
            "2d + 2",
            {"d": d},
            2 * d + 2,
        ),
        (
            "wup_projection_concat",
            "C_lut(d) + C_launch(d,m/B,1) + C_cat(m)",
            {
                "C_lut(d)": c_lut(d, params),
                "C_launch(d,m/B,1)": c_launch(d, m_local, 1, params),
                "C_cat(m)": c_cat(m, params),
            },
            c_lut(d, params) + c_launch(d, m_local, 1, params) + c_cat(m, params),
        ),
        (
            "wgate_projection_concat",
            "C_lut(d) + C_launch(d,m/B,1) + C_cat(m)",
            {
                "C_lut(d)": c_lut(d, params),
                "C_launch(d,m/B,1)": c_launch(d, m_local, 1, params),
                "C_cat(m)": c_cat(m, params),
            },
            c_lut(d, params) + c_launch(d, m_local, 1, params) + c_cat(m, params),
        ),
        (
            "mlp_fusion_silu_mul",
            "2m + 4",
            {"m": m},
            2 * m + 4,
        ),
        (
            "wdown_projection_final_output",
            "C_lut(m) + C_launch(m,h,1) + C_cat(d)",
            {
                "C_lut(m)": c_lut(m, params),
                "C_launch(m,h,1)": c_launch(m, h, 1, params),
                "C_cat(d)": c_cat(d, params),
            },
            c_lut(m, params) + c_launch(m, h, 1, params) + c_cat(d, params),
        ),
    ]

    out: list[dict[str, Any]] = []
    for index, (name, formula, terms, cycles) in enumerate(rows, start=1):
        out.append(
            {
                "index": index,
                "name": name,
                "formula": formula,
                "formula_terms": {str(k): int(v) for k, v in terms.items()},
                "cycles": int(cycles),
            }
        )
    return out

