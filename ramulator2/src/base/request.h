#ifndef     RAMULATOR_BASE_REQUEST_H
#define     RAMULATOR_BASE_REQUEST_H

#include <vector>
#include <list>
#include <string>

#include "base/base.h"

namespace Ramulator {

struct Request { 
  Addr_t    addr = -1;
  AddrVec_t addr_vec {};

  // Basic request id convention
  // 0 = Read, 1 = Write. The device spec defines all others
  struct Type {
    enum : int {
      // DRAM requests
      Read = 0, 
      Write = 1,

      // PIM requests
      PIM_MAC_AB = 4,
      PIM_MAC_SB = 5,
      PIM_MAC_PB = 6,
      PIM_WR_GB = 7,
      PIM_MV_SB = 8,
      PIM_MV_GB = 9,
      PIM_SFM = 10,
      PIM_SET_MODEL = 11,
      PIM_SET_HEAD = 12,
      PIM_BARRIER = 13,

      // BitAcc requests
      BITACC_LUT_GEN = 14,
      BITACC_LUT_BCAST = 15,
      BITACC_BLU = 16,
      BITACC_ACC = 17,
      BITACC_KV_WB = 18,
      BITACC_ATTN_RD = 19,
      BITACC_ATTN_WR = 20,
      BITACC_FC_RD = 21,
      BITACC_PRE_ATTN_NORM = 22,
      BITACC_PRE_ATTN_NORM_TAIL = 23,
      BITACC_ST_WAIT_QKV = 24,
      BITACC_ST_WAIT_QKV_TAIL = 25,
      BITACC_ALL_BANK_QKT = 26,
      BITACC_ALL_BANK_POST = 27,
      BITACC_ALL_BANK_WRITE_P_BACK = 28,
      BITACC_ALL_BANK_PV = 29,
      BITACC_ALL_BANK_WRITE_H_EQ_PV = 30,
      BITACC_WAIT_WO = 31,
      BITACC_WAIT_WO_TAIL = 32,
      BITACC_PRE_MLP_NORM = 33,
      BITACC_PRE_MLP_NORM_TAIL = 34,
      BITACC_WAIT_UP = 35,
      BITACC_WAIT_UP_TAIL = 36,
      BITACC_WAIT_GATE = 37,
      BITACC_WAIT_GATE_TAIL = 38,
      BITACC_MLP_FUSION = 39,
      BITACC_MLP_FUSION_TAIL = 40,
      BITACC_WAIT_DOWN = 41,
      BITACC_WAIT_DOWN_TAIL = 42,
      BITACC_MLP_RESIDUAL_SUM = 43,
      BITACC_MLP_RESIDUAL_SUM_TAIL = 44,
      BITACC_ALL_BANK_LUT_GEN_AND_LOAD = 45,
      BITACC_ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE = 46,
      BITACC_ALL_BANK_DEQUANT = 47,
      BITACC_ALL_BANK_WRITE_K_CURRENT = 48,
      BITACC_ALL_BANK_WRITE_V_CURRENT = 49,
      BITACC_DIMM_BUFFER_CONCAT_WRITE = 50,
      BITACC_ALL_BANK_ROPE = 51,
      BITACC_ALL_BANK_READ_K_CACHE = 52,
      BITACC_ALL_BANK_READ_V_CACHE = 53,

      // Latest conservative command scheme (legacy request ids above remain
      // available only for replaying archived traces).
      BITACC_EMBEDDING = 54,
      BITACC_HOST_XFER = 55,
      BITACC_CPU_NORM = 56,
      BITACC_LM_HEAD = 57,
      BITACC_TOPK = 58,
      BITACC_RMS_ATTN = 59,
      BITACC_RMS_MLP = 60,
      BITACC_QUANT = 61,
      BITACC_MLP_FUSION_V2 = 62,
      BITACC_FINAL_RES = 63,
      BITACC_BLOCK_END = 64,
      BITACC_ALL_BANK_LUT_LOAD = 65,
      BITACC_ALL_BANK_BITLINEAR = 66,
      BITACC_ALL_BANK_DEQUANT_ROPE = 67,
      BITACC_ALL_BANK_WR_K_CURRENT = 68,
      BITACC_ALL_BANK_WR_V_CURRENT = 69,
      BITACC_ALL_BANK_RD_K_CACHE = 70,
      BITACC_ALL_BANK_QKT_V2 = 71,
      BITACC_ALL_BANK_POST_V2 = 72,
      BITACC_ALL_BANK_RD_V_CACHE = 73,
      BITACC_ALL_BANK_PV_V2 = 74,
      BITACC_ATTN_XFER = 75,
      BITACC_LINEAR_XFER = 76,
      BITACC_ALL_BANK_RD_K_V_CACHE = 77,
      BITACC_QKT_XFER = 78,
      BITACC_PV_XFER = 79,
      BITACC_POST_SCORE_BCAST = 80,
      BITACC_TOKEN_DONE = 81,
      BITACC_QKT_REDUCE = 82
    };
  };

  int type_id = -1;    // An identifier for the type of the request
  int source_id = -1;  // An identifier for where the request is coming from (e.g., which core)

  int command = -1;          // The command that need to be issued to progress the request
  int final_command = -1;    // The final command that is needed to finish the request

  Clk_t arrive = -1;   // Clock cycle when the request arrive at the memory controller
  Clk_t depart = -1;   // Clock cycle when the request depart the memory controller

  // Optional controller occupancy carried by semantic traces. A value of zero
  // keeps the legacy command-name timing behavior; a positive value blocks the
  // controller for exactly this many controller cycles after the final command
  // is successfully issued.
  Clk_t occupancy_cycles = 0;

  std::function<void(Request&)> callback;

  Request(Addr_t addr, int type);
  Request(AddrVec_t addr_vec, int type);
  Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback);
};


struct ReqBuffer {
  std::list<Request> buffer;
  size_t max_size = 32;


  using iterator = std::list<Request>::iterator;
  iterator begin() { return buffer.begin(); };
  iterator end() { return buffer.end(); };


  size_t size() const { return buffer.size(); }

  bool enqueue(const Request& request) {
    if (buffer.size() <= max_size) {
      buffer.push_back(request);
      return true;
    } else {
      return false;
    }
  }

  void remove(iterator it) {
    buffer.erase(it);
  }
};

}        // namespace Ramulator


#endif   // RAMULATOR_BASE_REQUEST_H
