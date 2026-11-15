"""Shared BitAcc bank-layout helpers.

These helpers keep the simulator's bank-parallel assumptions consistent across
runtime config parsing, functional micro-op accounting, and trace timing.
"""

from __future__ import annotations

from typing import Any, Dict

from .ramulator_ddr import derive_ddr_logical_banks


ATTN_Q_BANKED_LAYERS = {"q_proj", "o_proj"}
ATTN_KV_BANKED_LAYERS = {"k_proj", "v_proj"}


def ceil_div(a: int, b: int) -> int:
    return (int(a) + max(1, int(b)) - 1) // max(1, int(b))


def derive_bitacc_logical_banks(ddr_config: Dict[str, Any]) -> int:
    return derive_ddr_logical_banks(
        dram_org=str(ddr_config.get("org", "DDR5_8Gb_x8")),
        channel=int(ddr_config.get("channel", 1)),
        rank=int(ddr_config.get("rank", 1)),
    )


def derive_attention_parallel_heads(active_heads: int, total_banks: int) -> int:
    return max(1, min(int(active_heads), int(total_banks)))


def derive_attention_rounds(active_heads: int, total_banks: int) -> int:
    parallel_heads = derive_attention_parallel_heads(active_heads, total_banks)
    return max(1, ceil_div(int(active_heads), parallel_heads))


def derive_fc_bank_layout(
    *,
    layer_name: str,
    out_features: int,
    total_banks: int,
    default_n_ch_per_bank: int,
    model_num_heads: int,
    model_num_kv_heads: int,
    model_head_dim: int,
) -> Dict[str, int]:
    total_banks = max(1, int(total_banks))
    out_features = max(1, int(out_features))
    default_n_ch_per_bank = max(1, int(default_n_ch_per_bank))
    model_num_heads = max(1, int(model_num_heads))
    model_num_kv_heads = max(1, int(model_num_kv_heads))
    model_head_dim = max(1, int(model_head_dim))

    if layer_name in ATTN_Q_BANKED_LAYERS:
        logical_units = model_num_heads
        local_cols = model_head_dim
        parallel_banks = min(total_banks, logical_units)
        rounds = ceil_div(logical_units, parallel_banks)
        return {
            "parallel_banks": int(parallel_banks),
            "local_cols": int(local_cols),
            "logical_units": int(logical_units),
            "rounds": int(rounds),
        }

    if layer_name in ATTN_KV_BANKED_LAYERS:
        logical_units = model_num_kv_heads
        local_cols = model_head_dim
        parallel_banks = min(total_banks, logical_units)
        rounds = ceil_div(logical_units, parallel_banks)
        return {
            "parallel_banks": int(parallel_banks),
            "local_cols": int(local_cols),
            "logical_units": int(logical_units),
            "rounds": int(rounds),
        }

    needed_banks = max(1, ceil_div(out_features, default_n_ch_per_bank))
    parallel_banks = min(total_banks, needed_banks)
    rounds = ceil_div(needed_banks, parallel_banks)
    return {
        "parallel_banks": int(parallel_banks),
        "local_cols": int(default_n_ch_per_bank),
        "logical_units": int(needed_banks),
        "rounds": int(rounds),
    }


def apply_bitacc_pe_config(
    pim_config: Dict[str, Any],
    pe_json: Dict[str, Any],
    modelinfos: Dict[str, Any],
    ddr_config: Dict[str, Any],
) -> None:
    hdim = int(modelinfos["hdim"])
    ffn_dim = int(round(float(modelinfos["ff_scale"]) * hdim))
    num_heads = int(modelinfos["num_heads"])
    num_kv_heads = int(modelinfos.get("num_key_value_heads") or (num_heads // max(1, int(modelinfos.get("gqa_size", 1) or 1))))
    head_dim = max(1, int(hdim // max(1, num_heads)))

    pim_config["BITACC_MODEL_INPUT_DIM"] = hdim
    pim_config["BITACC_AREA_INPUT_DIM"] = hdim
    pim_config["BITACC_MODEL_NUM_HEADS"] = num_heads
    pim_config["BITACC_MODEL_NUM_KV_HEADS"] = num_kv_heads
    pim_config["BITACC_MODEL_HEAD_DIM"] = head_dim

    def pick(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in obj:
                return obj[key]
        return default

    if pe_json:
        stage_overlap = pick(pe_json, "stage_mem_logic_overlap")
        if stage_overlap is not None:
            pim_config["BITACC_STAGE_OVERLAP"] = str(stage_overlap).lower() == "max"
        if "bitacc_pipeline" in pe_json:
            pim_config["BITACC_PIPELINE"] = bool(pe_json["bitacc_pipeline"])

    logical_banks = derive_bitacc_logical_banks(ddr_config)
    pim_config["BITACC_LOGICAL_BANKS"] = int(logical_banks)
    pim_config["BITACC_NUM_BANKS"] = int(logical_banks)
    pim_config["BITACC_TRACE_BANKS"] = int(logical_banks)
    attn_parallel_heads = int(min(logical_banks, num_heads))
    pim_config["BITACC_ATTN_PARALLEL_HEADS"] = attn_parallel_heads
    pim_config["BITACC_TRACE_ATTN_PARALLEL_HEADS"] = attn_parallel_heads
    pim_config["BITACC_TRACE_MAX_PARALLEL_HEADS"] = attn_parallel_heads

    n_tile = pick(pe_json or {}, "n_tile", "outputs_per_launch", "N_TILE")
    if n_tile is None:
        n_tile = int(pim_config.get("BITACC_TRACE_OUTPUTS_PER_LAUNCH", 1))
    n_tile = max(1, int(n_tile))
    pim_config["BITACC_PE_N_TILE"] = n_tile
    pim_config["BITACC_TRACE_OUTPUTS_PER_LAUNCH"] = n_tile

    pairs_per_rd = pick(pe_json or {}, "pairs_per_rd", "PAIRS_PER_RD")
    if pairs_per_rd is None:
        pairs_per_rd = int(pim_config.get("BITACC_TRACE_PAIRS_PER_RD", 256))
    pairs_per_rd = max(1, int(pairs_per_rd))
    pim_config["BITACC_PE_PAIRS_PER_RD"] = pairs_per_rd
    pim_config["BITACC_TRACE_PAIRS_PER_RD"] = pairs_per_rd

    rtl_k_tile_pairs = max(1, ceil_div(pairs_per_rd, n_tile))
    pim_config["BITACC_PE_K_TILE_PAIRS"] = rtl_k_tile_pairs
    pim_config["BITACC_K_TILE"] = 2 * rtl_k_tile_pairs
    pim_config["BITACC_TRACE_TILE_SCALARS"] = 2 * rtl_k_tile_pairs

    tok_par = pick(pe_json or {}, "tok_par", "ts_par", "TS_PAR")
    if tok_par is not None:
        pim_config["BITACC_PE_TOK_PAR"] = max(1, int(tok_par))
        pim_config["BITACC_TRACE_TOK_PAR"] = max(1, int(tok_par))

    qkt_tok_par = pick(pe_json or {}, "qkt_tok_par", "QKT_TOK_PAR", "tq_par", "TQ_PAR", "tok_par", "ts_par", "TS_PAR")
    if qkt_tok_par is not None:
        pim_config["BITACC_TRACE_QKT_TOK_PAR"] = max(1, int(qkt_tok_par))
        if tok_par is None:
            pim_config["BITACC_PE_TOK_PAR"] = max(1, int(qkt_tok_par))
            pim_config["BITACC_TRACE_TOK_PAR"] = max(1, int(qkt_tok_par))

    tp_par = pick(pe_json or {}, "tp_par", "TP_PAR")
    if tp_par is not None:
        pim_config["BITACC_TRACE_TP_PAR"] = max(1, int(tp_par))

    qkt_dot_par = pick(pe_json or {}, "qkt_dot_par", "QKT_DOT_PAR")
    if qkt_dot_par is not None:
        pim_config["BITACC_TRACE_QKT_DOT_PAR"] = max(1, int(qkt_dot_par))

    pv_dim_par = pick(pe_json or {}, "pv_dim_par", "PV_DIM_PAR")
    if pv_dim_par is not None:
        pim_config["BITACC_TRACE_PV_DIM_PAR"] = max(1, int(pv_dim_par))

    io_bits = pick(pe_json or {}, "io_bits", "IO_BITS", "wio", "Wio")
    if io_bits is not None:
        pim_config["BITACC_TRACE_IO_BITS"] = max(1, int(io_bits))

    max_kv_cache_tokens = pick(
        pe_json or {},
        "max_kv_cache_tokens",
        "kv_cache_max_tokens",
        "MAX_KV_CACHE_TOKENS",
    )
    if max_kv_cache_tokens is not None:
        pim_config["BITACC_KV_CACHE_MAX_TOKENS"] = max(0, int(max_kv_cache_tokens))

    active_banks = max(1, int(logical_banks))
    pim_config["BITACC_N_CHANNELS_PER_BANK"] = max(1, ceil_div(hdim, active_banks))
    pim_config["BITACC_TRACE_QKV_LOCAL_COLS"] = max(1, 3 * head_dim)
    pim_config["BITACC_TRACE_WO_LOCAL_COLS"] = max(1, head_dim)
    pim_config["BITACC_TRACE_DOWN_LOCAL_COLS"] = max(1, ceil_div(hdim, active_banks))
    pim_config["BITACC_TRACE_MLP_LOCAL_COLS"] = max(1, ceil_div(ffn_dim, active_banks))
    pim_config["BITACC_TRACE_MLP_DIM"] = ffn_dim

    pim_config["BITACC_LGU_UNITS"] = 1
    pim_config["BITACC_BLU_UNITS"] = n_tile
    pim_config["BITACC_ACC_UNITS"] = n_tile
