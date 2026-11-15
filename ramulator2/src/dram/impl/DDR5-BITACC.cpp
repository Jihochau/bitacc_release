#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class DDR5BitAcc : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, DDR5BitAcc, "DDR5-BITACC", "DDR5 Device Model with BitAcc request types")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      //   name         density   DQ   Ch Ra Bg Ba   Ro     Co
      {"DDR5_8Gb_x4",   {8<<10,   4,  {1, 1, 8, 2, 1<<16, 1<<11}}},
      {"DDR5_8Gb_x8",   {8<<10,   8,  {1, 1, 8, 4, 1<<15, 1<<10}}},
      {"DDR5_8Gb_x16",  {8<<10,   16, {1, 1, 4, 2, 1<<16, 1<<10}}},
      {"DDR5_16Gb_x4",  {16<<10,  4,  {1, 1, 8, 4, 1<<16, 1<<11}}},
      {"DDR5_16Gb_x8",  {16<<10,  8,  {1, 1, 8, 4, 1<<16, 1<<10}}},
      {"DDR5_16Gb_x16", {16<<10,  16, {1, 1, 4, 4, 1<<16, 1<<10}}},
      {"DDR5_32Gb_x4",  {32<<10,  4,  {1, 1, 8, 4, 1<<17, 1<<11}}},
      {"DDR5_32Gb_x8",  {32<<10,  8,  {1, 1, 8, 4, 1<<17, 1<<10}}},
      {"DDR5_32Gb_x16", {32<<10,  16, {1, 1, 4, 4, 1<<17, 1<<10}}},
      {"DDR5_64Gb_x4",  {64<<10,  4,  {1, 1, 8, 4, 1<<18, 1<<11}}},
      {"DDR5_64Gb_x8",  {64<<10,  8,  {1, 1, 8, 4, 1<<18, 1<<10}}},
      {"DDR5_64Gb_x16", {64<<10,  16, {1, 1, 4, 4, 1<<18, 1<<10}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name         rate   nBL  nCL nRCD   nRP  nRAS   nRC   nWR  nRTP nCWL nPPD nCCDS nCCDS_WR nCCDS_WTR nCCDL nCCDL_WR nCCDL_WTR nRRDS nRRDL nFAW nRFC1 nRFC2 nRFCsb nREFI nREFSBRD nRFM1 nRFM2 nRFMsb nDRFMab nDRFMsb nLUTGEN nLUTBCAST nBLU nACC nRDPIM nMACAB nMACSB nMACPB nCS, tCK_ps
      {"DDR5_3200AN",  {3200,   8,  24,  24,   24,   52,   75,   48,   12,  22,  2,    8,     8,     22+8+4,    8,     16,    22+8+16,   8,   -1,   -1,  -1,   -1,   -1,    -1,     30,    -1,   -1,   -1,     -1,     -1,     -1,       -1,  -1, -1,    -1,    -1,    -1,    -1,   2,   625}},
      {"DDR5_3200BN",  {3200,   8,  26,  26,   26,   52,   77,   48,   12,  24,  2,    8,     8,     24+8+4,    8,     16,    24+8+16,   8,   -1,   -1,  -1,   -1,   -1,    -1,     30,    -1,   -1,   -1,     -1,     -1,     -1,       -1,  -1, -1,    -1,    -1,    -1,    -1,   2,   625}},
      {"DDR5_3200C",   {3200,   8,  28,  28,   28,   52,   79,   48,   12,  26,  2,    8,     8,     26+8+4,    8,     16,    26+8+16,   8,   -1,   -1,  -1,   -1,   -1,    -1,     30,    -1,   -1,   -1,     -1,     -1,     -1,       -1,  -1, -1,    -1,    -1,    -1,    -1,   2,   625}},
    };

  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 16;

    inline static constexpr ImplDef m_levels = {
      "channel", "rank", "bankgroup", "bank", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT", "ACTAB", "ACTSB", "ACTPB",
      "PRE", "PREA", "PREsb", "PRESB", "PREPB",
      "RD",  "WR",  "RDA",  "WRA", "WRAB", "RDPIM", "MACAB", "MACSB", "MACPB",
      "LUTGEN", "LUTBCAST", "BLU", "ACC",
      "PRE_ATTN_NORM", "PRE_ATTN_NORM_TAIL",
      "ST_WAIT_QKV", "ST_WAIT_QKV_TAIL",
      "ALL_BANK_QKT", "ALL_BANK_POST", "ALL_BANK_WRITE_P_BACK",
      "ALL_BANK_PV", "ALL_BANK_WRITE_H_EQ_PV",
      "WAIT_WO", "WAIT_WO_TAIL",
      "PRE_MLP_NORM", "PRE_MLP_NORM_TAIL",
      "WAIT_UP", "WAIT_UP_TAIL",
      "WAIT_GATE", "WAIT_GATE_TAIL",
      "MLP_FUSION", "MLP_FUSION_TAIL",
      "WAIT_DOWN", "WAIT_DOWN_TAIL",
      "MLP_RESIDUAL_SUM", "MLP_RESIDUAL_SUM_TAIL",
      "ALL_BANK_LUT_GEN_AND_LOAD",
      "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE",
      "ALL_BANK_ROPE",
      "ALL_BANK_DEQUANT",
      "ALL_BANK_WRITE_K_CURRENT",
      "ALL_BANK_WRITE_V_CURRENT",
      "ALL_BANK_READ_K_CACHE",
      "ALL_BANK_READ_V_CACHE",
      "DIMM_BUFFER_CONCAT_WRITE",
      // Latest conservative command scheme. Unchanged names such as
      // ALL_BANK_QKT/POST/PV and MLP_FUSION reuse their definitions above.
      "EMBEDDING", "HOST_XFER", "CPU_NORM", "LM_HEAD", "TOPK",
      "RMS_ATTN", "RMS_MLP", "QUANT", "FINAL_RES", "BLOCK_END",
      "ALL_BANK_LUT_LOAD", "ALL_BANK_BITLINEAR", "ALL_BANK_DEQUANT_ROPE",
      "ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT",
      "ALL_BANK_RD_K_CACHE", "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE",
      "ATTN_XFER", "LINEAR_XFER", "QKT_XFER", "PV_XFER", "POST_SCORE_BCAST",
      "TOKEN_DONE", "QKT_REDUCE",
      "REFab",  "REFsb",
      "RFMab",  "RFMsb",
      "DRFMab", "DRFMsb",
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},    {"ACTAB", "row"}, {"ACTSB", "row"}, {"ACTPB", "row"},
        {"PRE",   "bank"},   {"PREA",   "rank"},   {"PREsb", "bankgroup"}, {"PRESB", "bank"}, {"PREPB", "bank"},
        {"RD",    "column"}, {"WR",      "column"}, {"RDA",   "column"}, {"WRA",   "column"}, {"WRAB",  "column"}, {"RDPIM", "column"},
        {"MACAB", "column"}, {"MACSB",   "column"}, {"MACPB", "column"},
        {"LUTGEN","bank"},   {"LUTBCAST","bank"},   {"BLU",   "bank"},   {"ACC",   "bank"},
        {"PRE_ATTN_NORM","rank"}, {"PRE_ATTN_NORM_TAIL","rank"},
        {"ST_WAIT_QKV","bank"}, {"ST_WAIT_QKV_TAIL","bank"},
        {"ALL_BANK_QKT","rank"}, {"ALL_BANK_POST","rank"}, {"ALL_BANK_WRITE_P_BACK","rank"},
        {"ALL_BANK_PV","rank"}, {"ALL_BANK_WRITE_H_EQ_PV","rank"},
        {"WAIT_WO","bank"}, {"WAIT_WO_TAIL","bank"},
        {"PRE_MLP_NORM","rank"}, {"PRE_MLP_NORM_TAIL","rank"},
        {"WAIT_UP","bank"}, {"WAIT_UP_TAIL","bank"},
        {"WAIT_GATE","bank"}, {"WAIT_GATE_TAIL","bank"},
        {"MLP_FUSION","rank"}, {"MLP_FUSION_TAIL","rank"},
        {"WAIT_DOWN","bank"}, {"WAIT_DOWN_TAIL","bank"},
        {"MLP_RESIDUAL_SUM","bank"}, {"MLP_RESIDUAL_SUM_TAIL","bank"},
        {"ALL_BANK_LUT_GEN_AND_LOAD","rank"},
        {"ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE","column"},
        {"ALL_BANK_ROPE","rank"},
        {"ALL_BANK_DEQUANT","rank"},
        {"ALL_BANK_WRITE_K_CURRENT","column"},
        {"ALL_BANK_WRITE_V_CURRENT","column"},
        {"ALL_BANK_READ_K_CACHE","column"},
        {"ALL_BANK_READ_V_CACHE","column"},
        {"DIMM_BUFFER_CONCAT_WRITE","rank"},
        {"EMBEDDING","rank"}, {"HOST_XFER","rank"}, {"CPU_NORM","rank"},
        {"LM_HEAD","rank"}, {"TOPK","rank"},
        {"RMS_ATTN","rank"}, {"RMS_MLP","rank"}, {"QUANT","rank"},
        {"FINAL_RES","rank"}, {"BLOCK_END","rank"},
        {"ALL_BANK_LUT_LOAD","rank"}, {"ALL_BANK_BITLINEAR","column"},
        {"ALL_BANK_DEQUANT_ROPE","rank"},
        {"ALL_BANK_WR_K_CURRENT","column"}, {"ALL_BANK_WR_V_CURRENT","column"},
        {"ALL_BANK_RD_K_CACHE","column"}, {"ALL_BANK_RD_V_CACHE","column"}, {"ALL_BANK_RD_K_V_CACHE","column"},
        {"ATTN_XFER","rank"}, {"LINEAR_XFER","rank"},
        {"QKT_XFER","rank"}, {"PV_XFER","rank"}, {"POST_SCORE_BCAST","rank"},
        {"TOKEN_DONE","rank"}, {"QKT_REDUCE","rank"},
        {"REFab",  "rank"},  {"REFsb",  "bankgroup"},  
        {"RFMab",  "rank"},  {"RFMsb",  "bankgroup"},  
        {"DRFMab", "rank"},  {"DRFMsb", "bankgroup"},  
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT",   {true,   false,   false,   false}},
        {"ACTAB", {true,   false,   false,   false}},
        {"ACTSB", {true,   false,   false,   false}},
        {"ACTPB", {true,   false,   false,   false}},
        {"PRE",   {false,  true,    false,   false}},
        {"PREA",  {false,  true,    false,   false}},
        {"PREsb", {false,  true,    false,   false}},
        {"PRESB", {false,  true,    false,   false}},
        {"PREPB", {false,  true,    false,   false}},
        {"RD",    {false,  false,   true,    false}},
        {"WR",    {false,  false,   true,    false}},
        {"RDA",   {false,  true,    true,    false}},
        {"WRA",   {false,  true,    true,    false}},
        {"WRAB",  {false,  false,   true,    false}},
        {"RDPIM", {false,  false,   true,    false}},
        {"MACAB", {false,  false,   true,    false}},
        {"MACSB", {false,  false,   true,    false}},
        {"MACPB", {false,  false,   true,    false}},
        {"LUTGEN",{false,  false,   false,   false}},
        {"LUTBCAST",{false,false,   false,   false}},
        {"BLU",   {false,  false,   false,   false}},
        {"ACC",   {false,  false,   false,   false}},
        {"PRE_ATTN_NORM",{false,false,false,false}},
        {"PRE_ATTN_NORM_TAIL",{false,false,false,false}},
        {"ST_WAIT_QKV",{false,false,false,false}},
        {"ST_WAIT_QKV_TAIL",{false,false,false,false}},
        {"ALL_BANK_QKT",{false,false,false,false}},
        {"ALL_BANK_POST",{false,false,false,false}},
        {"ALL_BANK_WRITE_P_BACK",{false,false,false,false}},
        {"ALL_BANK_PV",{false,false,false,false}},
        {"ALL_BANK_WRITE_H_EQ_PV",{false,false,false,false}},
        {"WAIT_WO",{false,false,false,false}},
        {"WAIT_WO_TAIL",{false,false,false,false}},
        {"PRE_MLP_NORM",{false,false,false,false}},
        {"PRE_MLP_NORM_TAIL",{false,false,false,false}},
        {"WAIT_UP",{false,false,false,false}},
        {"WAIT_UP_TAIL",{false,false,false,false}},
        {"WAIT_GATE",{false,false,false,false}},
        {"WAIT_GATE_TAIL",{false,false,false,false}},
        {"MLP_FUSION",{false,false,false,false}},
        {"MLP_FUSION_TAIL",{false,false,false,false}},
        {"WAIT_DOWN",{false,false,false,false}},
        {"WAIT_DOWN_TAIL",{false,false,false,false}},
        {"MLP_RESIDUAL_SUM",{false,false,false,false}},
        {"MLP_RESIDUAL_SUM_TAIL",{false,false,false,false}},
        {"ALL_BANK_LUT_GEN_AND_LOAD",{false,false,false,false}},
        {"ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE",{false,false,true,false}},
        {"ALL_BANK_ROPE",{false,false,false,false}},
        {"ALL_BANK_DEQUANT",{false,false,false,false}},
        {"ALL_BANK_WRITE_K_CURRENT",{false,false,false,false}},
        {"ALL_BANK_WRITE_V_CURRENT",{false,false,false,false}},
        {"ALL_BANK_READ_K_CACHE",{false,false,false,false}},
        {"ALL_BANK_READ_V_CACHE",{false,false,false,false}},
        {"DIMM_BUFFER_CONCAT_WRITE",{false,false,false,false}},
        {"EMBEDDING",{false,false,false,false}},
        {"HOST_XFER",{false,false,false,false}},
        {"CPU_NORM",{false,false,false,false}},
        {"LM_HEAD",{false,false,false,false}},
        {"TOPK",{false,false,false,false}},
        {"RMS_ATTN",{false,false,false,false}},
        {"RMS_MLP",{false,false,false,false}},
        {"QUANT",{false,false,false,false}},
        {"FINAL_RES",{false,false,false,false}},
        {"BLOCK_END",{false,false,false,false}},
        {"ALL_BANK_LUT_LOAD",{false,false,false,false}},
        {"ALL_BANK_BITLINEAR",{false,false,true,false}},
        {"ALL_BANK_DEQUANT_ROPE",{false,false,false,false}},
        {"ALL_BANK_WR_K_CURRENT",{false,false,true,false}},
        {"ALL_BANK_WR_V_CURRENT",{false,false,true,false}},
        {"ALL_BANK_RD_K_CACHE",{false,false,true,false}},
        {"ALL_BANK_RD_V_CACHE",{false,false,true,false}},
        {"ALL_BANK_RD_K_V_CACHE",{false,false,true,false}},
        {"ATTN_XFER",{false,false,false,false}},
        {"LINEAR_XFER",{false,false,false,false}},
        {"QKT_XFER",{false,false,false,false}},
        {"PV_XFER",{false,false,false,false}},
        {"POST_SCORE_BCAST",{false,false,false,false}},
        {"TOKEN_DONE",{false,false,false,false}},
        {"QKT_REDUCE",{false,false,false,false}},
        {"REFab", {false,  false,   false,   true }},
      }
    );

    inline static constexpr ImplDef m_requests = {
      // DRAM requests (align with Request::Type)
      "read", "write", "all-bank-refresh", "same-bank-refresh",
      // Compatibility PIM requests
      "pim-mac-all-bank", "pim-mac-same-bank", "pim-mac-per-bank",
      "pim-write-to-gemv-buffer", "pim-move-to-softmax-buffer", "pim-move-to-gemv-buffer",
      "pim-softmax", "pim-set-model", "pim-set-head", "pim-barrier",
      // BitAcc requests
      "bitacc-lut-gen", "bitacc-lut-bcast", "bitacc-blu", "bitacc-acc",
      "bitacc-kv-wb", "bitacc-attn-rd", "bitacc-attn-wr", "bitacc-fc-rd",
      "bitacc-pre-attn-norm", "bitacc-pre-attn-norm-tail",
      "bitacc-st-wait-qkv", "bitacc-st-wait-qkv-tail",
      "bitacc-all-bank-qkt", "bitacc-all-bank-post", "bitacc-all-bank-write-p-back",
      "bitacc-all-bank-pv", "bitacc-all-bank-write-h-eq-pv",
      "bitacc-wait-wo", "bitacc-wait-wo-tail",
      "bitacc-pre-mlp-norm", "bitacc-pre-mlp-norm-tail",
      "bitacc-wait-up", "bitacc-wait-up-tail",
      "bitacc-wait-gate", "bitacc-wait-gate-tail",
      "bitacc-mlp-fusion", "bitacc-mlp-fusion-tail",
      "bitacc-wait-down", "bitacc-wait-down-tail",
      "bitacc-mlp-residual-sum", "bitacc-mlp-residual-sum-tail",
      "bitacc-all-bank-lut-gen-and-load",
      "bitacc-all-bank-lut-weight-fetch-and-compute",
      "bitacc-all-bank-dequant",
      "bitacc-all-bank-write-k-current",
      "bitacc-all-bank-write-v-current",
      "bitacc-dimm-buffer-concat-write",
      "bitacc-all-bank-rope",
      "bitacc-all-bank-read-k-cache",
      "bitacc-all-bank-read-v-cache",
      "bitacc-embedding", "bitacc-host-xfer", "bitacc-cpu-norm", "bitacc-lm-head", "bitacc-topk",
      "bitacc-rms-attn", "bitacc-rms-mlp", "bitacc-quant", "bitacc-mlp-fusion-v2",
      "bitacc-final-res", "bitacc-block-end", "bitacc-all-bank-lut-load",
      "bitacc-all-bank-bitlinear", "bitacc-all-bank-dequant-rope",
      "bitacc-all-bank-wr-k-current", "bitacc-all-bank-wr-v-current",
      "bitacc-all-bank-rd-k-cache", "bitacc-all-bank-qkt-v2", "bitacc-all-bank-post-v2",
      "bitacc-all-bank-rd-v-cache", "bitacc-all-bank-pv-v2",
      "bitacc-attn-xfer", "bitacc-linear-xfer", "bitacc-all-bank-rd-k-v-cache",
      "bitacc-qkt-xfer", "bitacc-pv-xfer", "bitacc-post-score-bcast",
      "bitacc-token-done", "bitacc-qkt-reduce"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"},
        {"all-bank-refresh", "REFab"}, {"same-bank-refresh", "REFsb"},
        {"pim-mac-all-bank", "MACAB"}, {"pim-mac-same-bank", "MACSB"}, {"pim-mac-per-bank", "MACPB"},
        {"pim-write-to-gemv-buffer", "WRAB"}, {"pim-move-to-softmax-buffer", "RD"}, {"pim-move-to-gemv-buffer", "RD"},
        {"pim-softmax", "RD"}, {"pim-set-model", "RD"}, {"pim-set-head", "RD"}, {"pim-barrier", "RD"},
        {"bitacc-lut-gen", "LUTGEN"}, {"bitacc-lut-bcast", "LUTBCAST"}, {"bitacc-blu", "BLU"}, {"bitacc-acc", "ACC"},
        {"bitacc-kv-wb", "WRAB"}, {"bitacc-attn-rd", "RD"}, {"bitacc-attn-wr", "WR"},
        {"bitacc-fc-rd", "MACAB"},
        {"bitacc-pre-attn-norm", "PRE_ATTN_NORM"}, {"bitacc-pre-attn-norm-tail", "PRE_ATTN_NORM_TAIL"},
        {"bitacc-st-wait-qkv", "ST_WAIT_QKV"}, {"bitacc-st-wait-qkv-tail", "ST_WAIT_QKV_TAIL"},
        {"bitacc-all-bank-qkt", "ALL_BANK_QKT"}, {"bitacc-all-bank-post", "ALL_BANK_POST"},
        {"bitacc-all-bank-write-p-back", "ALL_BANK_WRITE_P_BACK"},
        {"bitacc-all-bank-pv", "ALL_BANK_PV"}, {"bitacc-all-bank-write-h-eq-pv", "ALL_BANK_WRITE_H_EQ_PV"},
        {"bitacc-wait-wo", "WAIT_WO"}, {"bitacc-wait-wo-tail", "WAIT_WO_TAIL"},
        {"bitacc-pre-mlp-norm", "PRE_MLP_NORM"}, {"bitacc-pre-mlp-norm-tail", "PRE_MLP_NORM_TAIL"},
        {"bitacc-wait-up", "WAIT_UP"}, {"bitacc-wait-up-tail", "WAIT_UP_TAIL"},
        {"bitacc-wait-gate", "WAIT_GATE"}, {"bitacc-wait-gate-tail", "WAIT_GATE_TAIL"},
        {"bitacc-mlp-fusion", "MLP_FUSION"}, {"bitacc-mlp-fusion-tail", "MLP_FUSION_TAIL"},
        {"bitacc-wait-down", "WAIT_DOWN"}, {"bitacc-wait-down-tail", "WAIT_DOWN_TAIL"},
        {"bitacc-mlp-residual-sum", "MLP_RESIDUAL_SUM"}, {"bitacc-mlp-residual-sum-tail", "MLP_RESIDUAL_SUM_TAIL"},
        {"bitacc-all-bank-lut-gen-and-load", "ALL_BANK_LUT_GEN_AND_LOAD"},
        {"bitacc-all-bank-lut-weight-fetch-and-compute", "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"},
        {"bitacc-all-bank-rope", "ALL_BANK_ROPE"},
        {"bitacc-all-bank-dequant", "ALL_BANK_DEQUANT"},
        {"bitacc-all-bank-write-k-current", "ALL_BANK_WRITE_K_CURRENT"},
        {"bitacc-all-bank-write-v-current", "ALL_BANK_WRITE_V_CURRENT"},
        {"bitacc-all-bank-read-k-cache", "ALL_BANK_READ_K_CACHE"},
        {"bitacc-all-bank-read-v-cache", "ALL_BANK_READ_V_CACHE"},
        {"bitacc-dimm-buffer-concat-write", "DIMM_BUFFER_CONCAT_WRITE"},
        {"bitacc-embedding", "EMBEDDING"}, {"bitacc-host-xfer", "HOST_XFER"},
        {"bitacc-cpu-norm", "CPU_NORM"}, {"bitacc-lm-head", "LM_HEAD"},
        {"bitacc-topk", "TOPK"}, {"bitacc-rms-attn", "RMS_ATTN"},
        {"bitacc-rms-mlp", "RMS_MLP"}, {"bitacc-quant", "QUANT"},
        {"bitacc-mlp-fusion-v2", "MLP_FUSION"}, {"bitacc-final-res", "FINAL_RES"},
        {"bitacc-block-end", "BLOCK_END"}, {"bitacc-all-bank-lut-load", "ALL_BANK_LUT_LOAD"},
        {"bitacc-all-bank-bitlinear", "ALL_BANK_BITLINEAR"},
        {"bitacc-all-bank-dequant-rope", "ALL_BANK_DEQUANT_ROPE"},
        {"bitacc-all-bank-wr-k-current", "ALL_BANK_WR_K_CURRENT"},
        {"bitacc-all-bank-wr-v-current", "ALL_BANK_WR_V_CURRENT"},
        {"bitacc-all-bank-rd-k-cache", "ALL_BANK_RD_K_CACHE"},
        {"bitacc-all-bank-qkt-v2", "ALL_BANK_QKT"},
        {"bitacc-all-bank-post-v2", "ALL_BANK_POST"},
        {"bitacc-all-bank-rd-v-cache", "ALL_BANK_RD_V_CACHE"},
        {"bitacc-all-bank-pv-v2", "ALL_BANK_PV"},
        {"bitacc-attn-xfer", "ATTN_XFER"}, {"bitacc-linear-xfer", "LINEAR_XFER"},
        {"bitacc-all-bank-rd-k-v-cache", "ALL_BANK_RD_K_V_CACHE"},
        {"bitacc-qkt-xfer", "QKT_XFER"}, {"bitacc-pv-xfer", "PV_XFER"},
        {"bitacc-post-score-bcast", "POST_SCORE_BCAST"},
        {"bitacc-token-done", "TOKEN_DONE"},
        {"bitacc-qkt-reduce", "QKT_REDUCE"}
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCD", "nRP", "nRAS", "nRC", "nWR", "nRTP", "nCWL",
      "nPPD",
      "nCCDS", "nCCDS_WR", "nCCDS_WTR", 
      "nCCDL", "nCCDL_WR", "nCCDL_WTR", 
      "nRRDS", "nRRDL",
      "nFAW",
      "nRFC1", "nRFC2", "nRFCsb", "nREFI", "nREFSBRD",
      "nRFM1", "nRFM2", "nRFMsb", 
      "nDRFMab", "nDRFMsb", 
      "nLUTGEN", "nLUTBCAST", "nBLU", "nACC", "nRDPIM", "nMACAB", "nMACSB", "nMACPB",
      "nPRE_ATTN_NORM", "nPRE_ATTN_NORM_TAIL",
      "nST_WAIT_QKV", "nST_WAIT_QKV_TAIL",
      "nALL_BANK_QKT", "nALL_BANK_POST", "nALL_BANK_WRITE_P_BACK",
      "nALL_BANK_PV", "nALL_BANK_WRITE_H_EQ_PV",
      "nWAIT_WO", "nWAIT_WO_TAIL",
      "nPRE_MLP_NORM", "nPRE_MLP_NORM_TAIL",
      "nWAIT_UP", "nWAIT_UP_TAIL",
      "nWAIT_GATE", "nWAIT_GATE_TAIL",
      "nMLP_FUSION", "nMLP_FUSION_TAIL",
      "nWAIT_DOWN", "nWAIT_DOWN_TAIL",
      "nMLP_RESIDUAL_SUM", "nMLP_RESIDUAL_SUM_TAIL",
      "nALL_BANK_LUT_GEN_AND_LOAD",
      "nALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE",
      "nALL_BANK_ROPE",
      "nALL_BANK_DEQUANT",
      "nALL_BANK_WRITE_K_CURRENT",
      "nALL_BANK_WRITE_V_CURRENT",
      "nALL_BANK_READ_K_CACHE",
      "nALL_BANK_READ_V_CACHE",
      "nDIMM_BUFFER_CONCAT_WRITE",
      "nEMBEDDING", "nHOST_XFER", "nCPU_NORM", "nLM_HEAD", "nTOPK",
      "nRMS_ATTN", "nRMS_MLP", "nQUANT", "nFINAL_RES", "nBLOCK_END",
      "nALL_BANK_LUT_LOAD", "nALL_BANK_BITLINEAR", "nALL_BANK_DEQUANT_ROPE",
      "nALL_BANK_WR_K_CURRENT", "nALL_BANK_WR_V_CURRENT",
      "nALL_BANK_RD_K_CACHE", "nALL_BANK_RD_V_CACHE", "nALL_BANK_RD_K_V_CACHE",
      "nATTN_XFER", "nLINEAR_XFER", "nQKT_XFER", "nPV_XFER", "nPOST_SCORE_BCAST",
      "nTOKEN_DONE", "nQKT_REDUCE",
      "nCS",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "PowerUp", "N/A"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",   "N/A"}, 
        {"rank",      "PowerUp"},
        {"bankgroup", "N/A"},
        {"bank",      "Closed"},
        {"row",       "Closed"},
        {"column",    "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<DDR5BitAcc> {
      Node(DDR5BitAcc* dram, Node* parent, int level, int id) : DRAMNodeBase<DDR5BitAcc>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  /************************************************
   *                 RFM Related
   ***********************************************/
  public:
    int m_BRC = 2;


  private:
    AddrVec_t resolve_all_bank_addr(const AddrVec_t& addr_vec) const {
      AddrVec_t resolved = addr_vec;
      if (resolved[m_levels["row"]] >= 0) {
        if (resolved[m_levels["bankgroup"]] < 0) {
          resolved[m_levels["bankgroup"]] = 0;
        }
        if (resolved[m_levels["bank"]] < 0) {
          resolved[m_levels["bank"]] = 0;
        }
      }
      return resolved;
    }


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      const AddrVec_t resolved = resolve_all_bank_addr(addr_vec);
      int channel_id = resolved[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, resolved, m_clk);
      m_channels[channel_id]->update_states(command, resolved, m_clk);
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      const AddrVec_t resolved = resolve_all_bank_addr(addr_vec);
      int channel_id = resolved[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, resolved, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      const AddrVec_t resolved = resolve_all_bank_addr(addr_vec);
      int channel_id = resolved[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, resolved, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      const AddrVec_t resolved = resolve_all_bank_addr(addr_vec);
      int channel_id = resolved[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, resolved, m_clk);
    };

  private:
    void set_organization() {
      int configured_channel_width = param_group("org").param<int>("channel_width").default_val(-1);
      int chips = param_group("org").param<int>("chips").default_val(-1);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      if (configured_channel_width > 0) {
        m_channel_width = configured_channel_width;
      } else if (chips > 0 && m_organization.dq > 0) {
        m_channel_width = chips * m_organization.dq;
      } else {
        m_channel_width = 32;
      }

      // Sanity check: is the calculated chip density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} chip density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 2);
      m_timing_vals("tCK_ps") = tCK_ps;

      // Load the organization specific timings
      int dq_id = [](int dq) -> int {
        switch (dq) {
          case 4:  return 0;
          case 8:  return 1;
          case 16: return 2;
          default: return -1;
        }
      }(m_organization.dq);

      int rate_id = [](int rate) -> int {
        switch (rate) {
          case 3200:  return 0;
          default:    return -1;
        }
      }(m_timing_vals("rate"));

      constexpr int nRRDL_TABLE[3][1] = {
      // 3200  
        { 5, },  // x4
        { 5, },  // x8
        { 5, },  // x16
      };
      constexpr int nFAW_TABLE[3][1] = {
      // 3200  
        { 40, },  // x4
        { 32, },  // x8
        { 32, },  // x16
      };

      if (dq_id != -1 && rate_id != -1) {
        m_timing_vals("nRRDL") = nRRDL_TABLE[dq_id][rate_id];
        m_timing_vals("nFAW")  = nFAW_TABLE [dq_id][rate_id];
      }

      // tCCD_L_WR2 (with RMW) table
      constexpr int nCCD_L_WR2_TABLE[1] = {
      // 3200  
        32,
      };
      if (dq_id == 0) {
        m_timing_vals("nCCDL_WR") = nCCD_L_WR2_TABLE[rate_id];
      }

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFC_TABLE[2][3] = {
      //  8Gb   16Gb  32Gb  
        { 195,  295,  410 }, // Normal refresh (tRFC1)
        { 130,  160,  220 }, // FGR 2x (tRFC2)
      };

      // tRFCsb table (unit is nanosecond!)
      constexpr int tRFCsb_TABLE[1][3] = {
      //  8Gb   16Gb  32Gb  
        { 115,  130,  190 }, // Normal refresh (tRFC1)
      };

      // tREFI(base) table (unit is nanosecond!)
      constexpr int tREFI_BASE = 3900;
      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 8192:  return 0;
          case 16384: return 1;
          case 32768: return 2;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFC1")  = JEDEC_rounding_DDR5(tRFC_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nRFC2")  = JEDEC_rounding_DDR5(tRFC_TABLE[1][density_id], tCK_ps);
      m_timing_vals("nRFCsb") = JEDEC_rounding_DDR5(tRFCsb_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nREFI")  = JEDEC_rounding_DDR5(tREFI_BASE, tCK_ps);

      m_timing_vals("nRFM1")  = m_timing_vals("nRFC1");
      m_timing_vals("nRFM2")  = m_timing_vals("nRFC2");
      m_timing_vals("nRFMsb") = m_timing_vals("nRFCsb");

      // Placeholder BitAcc command timings. These are intentionally explicit
      // knobs so later HLS/circuit results can replace them without changing
      // the command semantics in the DRAM model.
      m_timing_vals("nLUTGEN")   = 8;
      m_timing_vals("nLUTBCAST") = 32;
      m_timing_vals("nBLU")      = 64;
      m_timing_vals("nACC")      = 16;
      // Fused FC read+compute command latency placeholder.
      m_timing_vals("nRDPIM")    = 80;
      // Near-bank PIM compute commands are modeled as local PE issue slots.
      m_timing_vals("nMACAB")    = 1;
      m_timing_vals("nMACSB")    = 1;
      m_timing_vals("nMACPB")    = 1;
      m_timing_vals("nPRE_ATTN_NORM") = 1;
      m_timing_vals("nPRE_ATTN_NORM_TAIL") = 1;
      m_timing_vals("nST_WAIT_QKV") = 1;
      m_timing_vals("nST_WAIT_QKV_TAIL") = 1;
      m_timing_vals("nALL_BANK_QKT") = 1;
      m_timing_vals("nALL_BANK_POST") = 1;
      m_timing_vals("nALL_BANK_WRITE_P_BACK") = 1;
      m_timing_vals("nALL_BANK_PV") = 1;
      m_timing_vals("nALL_BANK_WRITE_H_EQ_PV") = 1;
      m_timing_vals("nWAIT_WO") = 1;
      m_timing_vals("nWAIT_WO_TAIL") = 1;
      m_timing_vals("nPRE_MLP_NORM") = 1;
      m_timing_vals("nPRE_MLP_NORM_TAIL") = 1;
      m_timing_vals("nWAIT_UP") = 1;
      m_timing_vals("nWAIT_UP_TAIL") = 1;
      m_timing_vals("nWAIT_GATE") = 1;
      m_timing_vals("nWAIT_GATE_TAIL") = 1;
      m_timing_vals("nMLP_FUSION") = 1;
      m_timing_vals("nMLP_FUSION_TAIL") = 1;
      m_timing_vals("nWAIT_DOWN") = 1;
      m_timing_vals("nWAIT_DOWN_TAIL") = 1;
      m_timing_vals("nMLP_RESIDUAL_SUM") = 1;
      m_timing_vals("nMLP_RESIDUAL_SUM_TAIL") = 1;
      m_timing_vals("nALL_BANK_LUT_GEN_AND_LOAD") = 1;
      m_timing_vals("nALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE") = 1;
      m_timing_vals("nALL_BANK_ROPE") = 1;
      m_timing_vals("nALL_BANK_DEQUANT") = 1;
      m_timing_vals("nALL_BANK_WRITE_K_CURRENT") = 1;
      m_timing_vals("nALL_BANK_WRITE_V_CURRENT") = 1;
      m_timing_vals("nALL_BANK_READ_K_CACHE") = 1;
      m_timing_vals("nALL_BANK_READ_V_CACHE") = 1;
      m_timing_vals("nALL_BANK_RD_K_V_CACHE") = 1;
      m_timing_vals("nDIMM_BUFFER_CONCAT_WRITE") = 1;
      m_timing_vals("nQKT_REDUCE") = 1;

      // tRRF table (unit is nanosecond!)
      constexpr int tRRFsb_TABLE[2][3] = {
      //  8Gb 16Gb 32Gb  
        { 70,  70,  70 }, // tRRFab
        { 60,  60,  60 }, // tRRFsb
      };
      m_BRC = param_group("RFM").param<int>("BRC").default_val(2);
      m_timing_vals("nDRFMab") = m_BRC * JEDEC_rounding_DDR5(tRRFsb_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nDRFMsb") = m_BRC * JEDEC_rounding_DDR5(tRRFsb_TABLE[1][density_id], tCK_ps);


      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding_DDR5(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      auto all_commands = std::vector<std::string_view>(m_commands.begin(), m_commands.end());
      populate_timingcons(this, {
          /*** Channel ***/ 
          // Two-Cycle Commands
          {.level = "channel", .preceding = {"ACT", "ACTAB", "ACTSB", "ACTPB", "RD", "RDA", "WR", "WRA", "WRAB", "RDPIM"}, .following = all_commands, .latency = 2},
          // LUT broadcast occupies the off-bank path / command pipeline but does
          // not access DRAM rows. Model it as a channel-visible config event.
          {.level = "channel", .preceding = {"LUTBCAST"}, .following = all_commands, .latency = V("nLUTBCAST")},
          // Near-bank PIM compute commands have their own issue slots and do not
          // consume the normal DDR column-data path.
          {.level = "channel", .preceding = {"MACAB"}, .following = {"MACAB"}, .latency = V("nMACAB")},
          {.level = "channel", .preceding = {"MACSB"}, .following = {"MACSB"}, .latency = V("nMACSB")},

          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "channel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"WRAB"}, .following = {"WRAB"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"RDPIM"}, .following = {"RDPIM", "RD", "RDA"}, .latency = V("nRDPIM")},

          /*** Rank (or different BankGroup) ***/ 
          // CAS <-> CAS
          /// nCCDS is the minimal latency for column commands 
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "rank", .preceding = {"RDPIM"}, .following = {"RDPIM", "RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDS_WR")},
          /// RD <-> WR, Minimum Read to Write, Assuming Read DQS Offset = 0, tRPST = 0.5, tWPRE = 2 tCK                          
          {.level = "rank", .preceding = {"RD", "RDA", "RDPIM"}, .following = {"WR", "WRA"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL") + 2},   // nCCDS_RTW
          /// WR <-> RD, Minimum Read after Write
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS_WTR")},
          /// CAS <-> CAS between sibling ranks, nCS (rank switching) is needed for new DQS
          {.level = "rank", .preceding = {"RD", "RDA", "RDPIM"}, .following = {"RD", "RDA", "RDPIM", "WR", "WRA"}, .latency = V("nBL") + V("nCS"), .is_sibling = true},
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCL")  + V("nBL") + V("nCS") - V("nCWL"), .is_sibling = true},
          /// CAS <-> PREab
          {.level = "rank", .preceding = {"RD", "RDPIM"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "rank", .preceding = {"WR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},          
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},          
          {.level = "rank", .preceding = {"ACT", "ACTAB"}, .following = {"PREA"}, .latency = V("nRAS")},
          {.level = "rank", .preceding = {"PREA"}, .following = {"ACT", "ACTAB"}, .latency = V("nRP")},
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACT", "ACTAB"}, .following = {"REFab", "RFMab", "DRFMab"}, .latency = V("nRC")},
          {.level = "rank", .preceding = {"PRE", "PREA"}, .following = {"REFab", "RFMab", "DRFMab"}, .latency = V("nRP")},          
          {.level = "rank", .preceding = {"RDA"}, .following = {"REFab", "RFMab", "DRFMab"}, .latency = V("nRP") + V("nRTP")},          
          {.level = "rank", .preceding = {"WRA"}, .following = {"REFab", "RFMab", "DRFMab"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},          
          // During all-bank refresh windows, the whole rank is busy.
          // Enforce refresh-to-any-command blackout so back-to-back REFab in
          // adjacent cycles cannot occur.
          {.level = "rank", .preceding = {"REFab"}, .following = all_commands, .latency = V("nRFC1")},
          {.level = "rank", .preceding = {"RFMab"}, .following = all_commands, .latency = V("nRFM1")},
          {.level = "rank", .preceding = {"DRFMab"}, .following = all_commands, .latency = V("nDRFMab")},
          {.level = "rank", .preceding = {"PRE_ATTN_NORM"}, .following = all_commands, .latency = V("nPRE_ATTN_NORM")},
          {.level = "rank", .preceding = {"PRE_ATTN_NORM_TAIL"}, .following = all_commands, .latency = V("nPRE_ATTN_NORM_TAIL")},
          {.level = "rank", .preceding = {"ST_WAIT_QKV"}, .following = all_commands, .latency = V("nST_WAIT_QKV")},
          {.level = "rank", .preceding = {"ST_WAIT_QKV_TAIL"}, .following = all_commands, .latency = V("nST_WAIT_QKV_TAIL")},
          {.level = "rank", .preceding = {"ALL_BANK_QKT"}, .following = all_commands, .latency = V("nALL_BANK_QKT")},
          {.level = "rank", .preceding = {"ALL_BANK_POST"}, .following = all_commands, .latency = V("nALL_BANK_POST")},
          {.level = "rank", .preceding = {"ALL_BANK_WRITE_P_BACK"}, .following = all_commands, .latency = V("nALL_BANK_WRITE_P_BACK")},
          {.level = "rank", .preceding = {"ALL_BANK_PV"}, .following = all_commands, .latency = V("nALL_BANK_PV")},
          {.level = "rank", .preceding = {"ALL_BANK_WRITE_H_EQ_PV"}, .following = all_commands, .latency = V("nALL_BANK_WRITE_H_EQ_PV")},
          {.level = "rank", .preceding = {"WAIT_WO"}, .following = all_commands, .latency = V("nWAIT_WO")},
          {.level = "rank", .preceding = {"WAIT_WO_TAIL"}, .following = all_commands, .latency = V("nWAIT_WO_TAIL")},
          {.level = "rank", .preceding = {"PRE_MLP_NORM"}, .following = all_commands, .latency = V("nPRE_MLP_NORM")},
          {.level = "rank", .preceding = {"PRE_MLP_NORM_TAIL"}, .following = all_commands, .latency = V("nPRE_MLP_NORM_TAIL")},
          {.level = "rank", .preceding = {"WAIT_UP"}, .following = all_commands, .latency = V("nWAIT_UP")},
          {.level = "rank", .preceding = {"WAIT_UP_TAIL"}, .following = all_commands, .latency = V("nWAIT_UP_TAIL")},
          {.level = "rank", .preceding = {"WAIT_GATE"}, .following = all_commands, .latency = V("nWAIT_GATE")},
          {.level = "rank", .preceding = {"WAIT_GATE_TAIL"}, .following = all_commands, .latency = V("nWAIT_GATE_TAIL")},
          {.level = "rank", .preceding = {"MLP_FUSION"}, .following = all_commands, .latency = V("nMLP_FUSION")},
          {.level = "rank", .preceding = {"MLP_FUSION_TAIL"}, .following = all_commands, .latency = V("nMLP_FUSION_TAIL")},
          {.level = "rank", .preceding = {"WAIT_DOWN"}, .following = all_commands, .latency = V("nWAIT_DOWN")},
          {.level = "rank", .preceding = {"WAIT_DOWN_TAIL"}, .following = all_commands, .latency = V("nWAIT_DOWN_TAIL")},
          {.level = "rank", .preceding = {"MLP_RESIDUAL_SUM"}, .following = all_commands, .latency = V("nMLP_RESIDUAL_SUM")},
          {.level = "rank", .preceding = {"MLP_RESIDUAL_SUM_TAIL"}, .following = all_commands, .latency = V("nMLP_RESIDUAL_SUM_TAIL")},
          {.level = "rank", .preceding = {"ALL_BANK_LUT_GEN_AND_LOAD"}, .following = all_commands, .latency = V("nALL_BANK_LUT_GEN_AND_LOAD")},
          {.level = "rank", .preceding = {"ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"}, .following = all_commands, .latency = V("nALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE")},
          {.level = "rank", .preceding = {"ALL_BANK_ROPE"}, .following = all_commands, .latency = V("nALL_BANK_ROPE")},
          {.level = "rank", .preceding = {"ALL_BANK_DEQUANT"}, .following = all_commands, .latency = V("nALL_BANK_DEQUANT")},
          {.level = "rank", .preceding = {"ALL_BANK_WRITE_K_CURRENT"}, .following = all_commands, .latency = V("nALL_BANK_WRITE_K_CURRENT")},
          {.level = "rank", .preceding = {"ALL_BANK_WRITE_V_CURRENT"}, .following = all_commands, .latency = V("nALL_BANK_WRITE_V_CURRENT")},
          {.level = "rank", .preceding = {"ALL_BANK_READ_K_CACHE"}, .following = all_commands, .latency = V("nALL_BANK_READ_K_CACHE")},
          {.level = "rank", .preceding = {"ALL_BANK_READ_V_CACHE"}, .following = all_commands, .latency = V("nALL_BANK_READ_V_CACHE")},
          {.level = "rank", .preceding = {"DIMM_BUFFER_CONCAT_WRITE"}, .following = all_commands, .latency = V("nDIMM_BUFFER_CONCAT_WRITE")},
          {.level = "rank", .preceding = {"EMBEDDING"}, .following = all_commands, .latency = V("nEMBEDDING")},
          {.level = "rank", .preceding = {"HOST_XFER"}, .following = all_commands, .latency = V("nHOST_XFER")},
          {.level = "rank", .preceding = {"CPU_NORM"}, .following = all_commands, .latency = V("nCPU_NORM")},
          {.level = "rank", .preceding = {"LM_HEAD"}, .following = all_commands, .latency = V("nLM_HEAD")},
          {.level = "rank", .preceding = {"TOPK"}, .following = all_commands, .latency = V("nTOPK")},
          {.level = "rank", .preceding = {"RMS_ATTN"}, .following = all_commands, .latency = V("nRMS_ATTN")},
          {.level = "rank", .preceding = {"RMS_MLP"}, .following = all_commands, .latency = V("nRMS_MLP")},
          {.level = "rank", .preceding = {"QUANT"}, .following = all_commands, .latency = V("nQUANT")},
          {.level = "rank", .preceding = {"FINAL_RES"}, .following = all_commands, .latency = V("nFINAL_RES")},
          {.level = "rank", .preceding = {"BLOCK_END"}, .following = all_commands, .latency = V("nBLOCK_END")},
          {.level = "rank", .preceding = {"ALL_BANK_LUT_LOAD"}, .following = all_commands, .latency = V("nALL_BANK_LUT_LOAD")},
          {.level = "rank", .preceding = {"ALL_BANK_BITLINEAR"}, .following = all_commands, .latency = V("nALL_BANK_BITLINEAR")},
          {.level = "rank", .preceding = {"ALL_BANK_DEQUANT_ROPE"}, .following = all_commands, .latency = V("nALL_BANK_DEQUANT_ROPE")},
          {.level = "rank", .preceding = {"ALL_BANK_WR_K_CURRENT"}, .following = all_commands, .latency = V("nALL_BANK_WR_K_CURRENT")},
          {.level = "rank", .preceding = {"ALL_BANK_WR_V_CURRENT"}, .following = all_commands, .latency = V("nALL_BANK_WR_V_CURRENT")},
          {.level = "rank", .preceding = {"ALL_BANK_RD_K_CACHE"}, .following = all_commands, .latency = V("nALL_BANK_RD_K_CACHE")},
          {.level = "rank", .preceding = {"ALL_BANK_RD_V_CACHE"}, .following = all_commands, .latency = V("nALL_BANK_RD_V_CACHE")},
          {.level = "rank", .preceding = {"ALL_BANK_RD_K_V_CACHE"}, .following = all_commands, .latency = V("nALL_BANK_RD_K_V_CACHE")},
          {.level = "rank", .preceding = {"ATTN_XFER"}, .following = all_commands, .latency = V("nATTN_XFER")},
          {.level = "rank", .preceding = {"LINEAR_XFER"}, .following = all_commands, .latency = V("nLINEAR_XFER")},
          {.level = "rank", .preceding = {"QKT_XFER"}, .following = all_commands, .latency = V("nQKT_XFER")},
          {.level = "rank", .preceding = {"PV_XFER"}, .following = all_commands, .latency = V("nPV_XFER")},
          {.level = "rank", .preceding = {"POST_SCORE_BCAST"}, .following = all_commands, .latency = V("nPOST_SCORE_BCAST")},
          {.level = "rank", .preceding = {"TOKEN_DONE"}, .following = all_commands, .latency = V("nTOKEN_DONE")},
          {.level = "rank", .preceding = {"QKT_REDUCE"}, .following = all_commands, .latency = V("nQKT_REDUCE")},

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"RDPIM"}, .following = {"RDPIM", "RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDL_WR")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL_WTR")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDL")},  

          /*** Bank ***/ 
          {.level = "bank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACTAB"}, .following = {"MACAB", "WRAB"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACTAB"}, .following = {"ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE", "ALL_BANK_WRITE_K_CURRENT", "ALL_BANK_WRITE_V_CURRENT", "ALL_BANK_READ_K_CACHE", "ALL_BANK_READ_V_CACHE", "ALL_BANK_BITLINEAR", "ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT", "ALL_BANK_RD_K_CACHE", "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE"}, .latency = V("nRCD")},
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"ACTSB", "MACSB"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"ACTPB", "MACPB"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"RD", "RDA", "WR", "WRA", "RDPIM"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"PRE"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"PRESB"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"PREPB"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"MACAB"}, .following = {"MACAB"}, .latency = V("nMACAB")},  
          {.level = "bank", .preceding = {"MACSB"}, .following = {"MACSB"}, .latency = V("nMACSB")},  
          {.level = "bank", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nMACPB")},  
          {.level = "bank", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"RD", "RDPIM"},  .following = {"PRE"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"MACAB"}, .following = {"PREA"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"MACSB"}, .following = {"PRESB"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"MACPB"}, .following = {"PREPB"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"WRAB"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE", "ALL_BANK_READ_K_CACHE", "ALL_BANK_READ_V_CACHE"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "bank", .preceding = {"ALL_BANK_WRITE_K_CURRENT", "ALL_BANK_WRITE_V_CURRENT"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},
          {.level = "bank", .preceding = {"ALL_BANK_BITLINEAR", "ALL_BANK_RD_K_CACHE", "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "bank", .preceding = {"ALL_BANK_WR_K_CURRENT", "ALL_BANK_WR_V_CURRENT"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},
          {.level = "bank", .preceding = {"WR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"RDA"}, .following = {"ACT"}, .latency = V("nRTP") + V("nRP")},  
          {.level = "bank", .preceding = {"WRA"}, .following = {"ACT"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},  
          {.level = "bank", .preceding = {"PRESB"}, .following = {"ACTSB"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"PREPB"}, .following = {"ACTPB"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"WR"},  .following = {"RDA"}, .latency = V("nCWL") + V("nBL") + V("nWR") - V("nRTP")},  

          // BitAcc near-bank commands:
          // - LUTBCAST occupies the config/broadcast path but not row commands.
          // - ACC is a local compute-pipe event, so it should not fully block
          //   subsequent DRAM fetches from the same bank.
          {.level = "bank", .preceding = {"LUTGEN"},   .following = all_commands, .latency = V("nLUTGEN")},
          {.level = "bank", .preceding = {"LUTBCAST"}, .following = all_commands, .latency = V("nLUTBCAST")},
          {.level = "bank", .preceding = {"BLU"},      .following = all_commands, .latency = V("nBLU")},
          {.level = "bank", .preceding = {"ACC"},      .following = {"ACC"},      .latency = V("nACC")},

          /// Same-bank refresh/RFM timings. The timings of the bank in other BGs will be updated by action function
          {.level = "bank", .preceding = {"REFsb"},  .following = {"ACT", "REFsb", "RFMsb", "DRFMsb"}, .latency = V("nRFCsb")},  
          {.level = "bank", .preceding = {"RFMsb"},  .following = {"ACT", "REFsb", "RFMsb", "DRFMsb"}, .latency = V("nRFMsb")},  
          {.level = "bank", .preceding = {"DRFMsb"}, .following = {"ACT", "REFsb", "RFMsb", "DRFMsb"}, .latency = V("nDRFMsb")},  
        }
      );
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Rank Actions
      m_actions[m_levels["rank"]][m_commands["PREA"]] = Lambdas::Action::Rank::PREA<DDR5BitAcc>;

      // Same-Bank Actions.
      m_actions[m_levels["bankgroup"]][m_commands["PREsb"]] = Lambdas::Action::BankGroup::PREsb<DDR5BitAcc>;
      m_actions[m_levels["bankgroup"]][m_commands["MACSB"]] = Lambdas::Action::BankGroup::PIMSameBankActions<DDR5BitAcc>;
      m_actions[m_levels["bankgroup"]][m_commands["MACPB"]] = Lambdas::Action::BankGroup::PIMPerBankActions<DDR5BitAcc>;

      // We call update_timing for the banks in other BGs here
      m_actions[m_levels["bankgroup"]][m_commands["REFsb"]]  = Lambdas::Action::BankGroup::SameBankActions<DDR5BitAcc>;
      m_actions[m_levels["bankgroup"]][m_commands["RFMsb"]]  = Lambdas::Action::BankGroup::SameBankActions<DDR5BitAcc>;
      m_actions[m_levels["bankgroup"]][m_commands["DRFMsb"]] = Lambdas::Action::BankGroup::SameBankActions<DDR5BitAcc>;

      // Bank actions
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["ACTAB"]] = Lambdas::Action::Bank::ACTAB<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["ACTSB"]] = Lambdas::Action::Bank::ACTSB<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["ACTPB"]] = Lambdas::Action::Bank::ACTPB<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["PRESB"]] = Lambdas::Action::Bank::PRESB<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["PREPB"]] = Lambdas::Action::Bank::PREPB<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["RDA"]] = Lambdas::Action::Bank::PRE<DDR5BitAcc>;
      m_actions[m_levels["bank"]][m_commands["WRA"]] = Lambdas::Action::Bank::PRE<DDR5BitAcc>;
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Rank Preqs
      m_preqs[m_levels["rank"]][m_commands["REFab"]]  = Lambdas::Preq::Rank::RequireAllBanksClosed<DDR5BitAcc>;
      m_preqs[m_levels["rank"]][m_commands["RFMab"]]  = Lambdas::Preq::Rank::RequireAllBanksClosed<DDR5BitAcc>;
      m_preqs[m_levels["rank"]][m_commands["DRFMab"]] = Lambdas::Preq::Rank::RequireAllBanksClosed<DDR5BitAcc>;

      // Same-Bank Preqs.
      m_preqs[m_levels["bankgroup"]][m_commands["REFsb"]]  = Lambdas::Preq::Rank::RequireSameBanksClosed<DDR5BitAcc>;
      m_preqs[m_levels["bankgroup"]][m_commands["RFMsb"]]  = Lambdas::Preq::Rank::RequireSameBanksClosed<DDR5BitAcc>;
      m_preqs[m_levels["bankgroup"]][m_commands["DRFMsb"]] = Lambdas::Preq::Rank::RequireSameBanksClosed<DDR5BitAcc>;

      // Bank Preqs
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["RDPIM"]] = Lambdas::Preq::Bank::RequireRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["WRAB"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::Preq::Bank::RequirePIMSameBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::Preq::Bank::RequirePIMPerBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["LUTGEN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return cmd;
      };
      m_preqs[m_levels["bank"]][m_commands["LUTBCAST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return cmd;
      };
      m_preqs[m_levels["bank"]][m_commands["BLU"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return cmd;
      };
      m_preqs[m_levels["bank"]][m_commands["ACC"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return cmd;
      };
      m_preqs[m_levels["bank"]][m_commands["PRE_ATTN_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["PRE_ATTN_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ST_WAIT_QKV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ST_WAIT_QKV_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_QKT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_POST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_WRITE_P_BACK"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_WRITE_H_EQ_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_WO"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_WO_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["PRE_MLP_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["PRE_MLP_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_UP"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_UP_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_GATE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_GATE_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["MLP_FUSION"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["MLP_FUSION_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_DOWN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["WAIT_DOWN_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_LUT_GEN_AND_LOAD"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_ROPE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_DEQUANT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_WRITE_K_CURRENT"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_WRITE_V_CURRENT"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_READ_K_CACHE"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_READ_V_CACHE"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["ALL_BANK_RD_K_V_CACHE"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      m_preqs[m_levels["bank"]][m_commands["DIMM_BUFFER_CONCAT_WRITE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      auto latest_passthrough_preq = [] (Node* node, int cmd, int target_id, Clk_t clk) { return cmd; };
      for (const auto& command : {"EMBEDDING", "HOST_XFER", "CPU_NORM", "LM_HEAD", "TOPK",
                                  "RMS_ATTN", "RMS_MLP", "QUANT", "FINAL_RES", "BLOCK_END",
                                  "ALL_BANK_LUT_LOAD", "ALL_BANK_DEQUANT_ROPE",
                                  "ATTN_XFER", "LINEAR_XFER", "QKT_XFER", "PV_XFER",
                                  "POST_SCORE_BCAST", "TOKEN_DONE", "QKT_REDUCE"}) {
        m_preqs[m_levels["bank"]][m_commands[command]] = latest_passthrough_preq;
      }
      for (const auto& command : {"ALL_BANK_BITLINEAR", "ALL_BANK_WR_K_CURRENT",
                                  "ALL_BANK_WR_V_CURRENT", "ALL_BANK_RD_K_CACHE",
                                  "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE"}) {
        m_preqs[m_levels["bank"]][m_commands[command]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR5BitAcc>;
      }
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["RDPIM"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["WRAB"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["LUTGEN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowhits[m_levels["bank"]][m_commands["LUTBCAST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowhits[m_levels["bank"]][m_commands["BLU"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowhits[m_levels["bank"]][m_commands["ACC"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowhits[m_levels["bank"]][m_commands["PRE_ATTN_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["PRE_ATTN_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ST_WAIT_QKV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ST_WAIT_QKV_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_QKT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_POST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_WRITE_P_BACK"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_WRITE_H_EQ_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_WO"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_WO_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["PRE_MLP_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["PRE_MLP_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_UP"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_UP_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_GATE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_GATE_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["MLP_FUSION"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["MLP_FUSION_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_DOWN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["WAIT_DOWN_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_LUT_GEN_AND_LOAD"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_ROPE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_DEQUANT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_WRITE_K_CURRENT"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_WRITE_V_CURRENT"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_READ_K_CACHE"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_READ_V_CACHE"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["ALL_BANK_RD_K_V_CACHE"]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      m_rowhits[m_levels["bank"]][m_commands["DIMM_BUFFER_CONCAT_WRITE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      auto latest_never_rowhit = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      for (const auto& command : {"EMBEDDING", "HOST_XFER", "CPU_NORM", "LM_HEAD", "TOPK",
                                  "RMS_ATTN", "RMS_MLP", "QUANT", "FINAL_RES", "BLOCK_END",
                                  "ALL_BANK_LUT_LOAD", "ALL_BANK_DEQUANT_ROPE",
                                  "ATTN_XFER", "LINEAR_XFER", "QKT_XFER", "PV_XFER",
                                  "POST_SCORE_BCAST", "TOKEN_DONE", "QKT_REDUCE"}) {
        m_rowhits[m_levels["bank"]][m_commands[command]] = latest_never_rowhit;
      }
      for (const auto& command : {"ALL_BANK_BITLINEAR", "ALL_BANK_WR_K_CURRENT",
                                  "ALL_BANK_WR_V_CURRENT", "ALL_BANK_RD_K_CACHE",
                                  "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE"}) {
        m_rowhits[m_levels["bank"]][m_commands[command]] = Lambdas::RowHit::Bank::RDWR<DDR5BitAcc>;
      }
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["RDPIM"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["WRAB"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["LUTGEN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowopens[m_levels["bank"]][m_commands["LUTBCAST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowopens[m_levels["bank"]][m_commands["BLU"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowopens[m_levels["bank"]][m_commands["ACC"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        return false;
      };
      m_rowopens[m_levels["bank"]][m_commands["PRE_ATTN_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["PRE_ATTN_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ST_WAIT_QKV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ST_WAIT_QKV_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_QKT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_POST"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_WRITE_P_BACK"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_WRITE_H_EQ_PV"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_WO"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_WO_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["PRE_MLP_NORM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["PRE_MLP_NORM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_UP"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_UP_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_GATE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_GATE_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["MLP_FUSION"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["MLP_FUSION_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_DOWN"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["WAIT_DOWN_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["MLP_RESIDUAL_SUM_TAIL"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_LUT_GEN_AND_LOAD"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_ROPE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_DEQUANT"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_WRITE_K_CURRENT"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_WRITE_V_CURRENT"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_READ_K_CACHE"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_READ_V_CACHE"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["ALL_BANK_RD_K_V_CACHE"]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      m_rowopens[m_levels["bank"]][m_commands["DIMM_BUFFER_CONCAT_WRITE"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      auto latest_never_rowopen = [] (Node* node, int cmd, int target_id, Clk_t clk) { return false; };
      for (const auto& command : {"EMBEDDING", "HOST_XFER", "CPU_NORM", "LM_HEAD", "TOPK",
                                  "RMS_ATTN", "RMS_MLP", "QUANT", "FINAL_RES", "BLOCK_END",
                                  "ALL_BANK_LUT_LOAD", "ALL_BANK_DEQUANT_ROPE",
                                  "ATTN_XFER", "LINEAR_XFER", "QKT_XFER", "PV_XFER",
                                  "POST_SCORE_BCAST", "TOKEN_DONE", "QKT_REDUCE"}) {
        m_rowopens[m_levels["bank"]][m_commands[command]] = latest_never_rowopen;
      }
      for (const auto& command : {"ALL_BANK_BITLINEAR", "ALL_BANK_WR_K_CURRENT",
                                  "ALL_BANK_WR_V_CURRENT", "ALL_BANK_RD_K_CACHE",
                                  "ALL_BANK_RD_V_CACHE", "ALL_BANK_RD_K_V_CACHE"}) {
        m_rowopens[m_levels["bank"]][m_commands[command]] = Lambdas::RowOpen::Bank::RDWR<DDR5BitAcc>;
      }
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
      }
    };
};


}        // namespace Ramulator
