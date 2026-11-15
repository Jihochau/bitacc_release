#include <filesystem>
#include <iostream>
#include <fstream>

#include "frontend/frontend.h"
#include "base/exception.h"

namespace Ramulator {

namespace fs = std::filesystem;

class PIMLoadStoreTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PIMLoadStoreTrace, "PIMLoadStoreTrace", "PIM/Load/Store memory address trace.")

  private:
    struct Trace {
      int req_type;
      Addr_t addr;
      AddrVec_t addr_vec;
      bool has_addr_vec = false;
      Clk_t occupancy_cycles = 0;
    };
    std::vector<Trace> m_trace;

    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;

    size_t m_trace_count = 0;

    Logger_t m_logger;

  public:
    void init() override {
      std::string trace_path_str = param<std::string>("path").desc("Path to the load store trace file.").required();
      m_clock_ratio = param<unsigned int>("clock_ratio").required();

      m_logger = Logging::create_logger("LoadStoreTrace");
      m_logger->info("Loading trace file {} ...", trace_path_str);
      init_trace(trace_path_str);
      m_logger->info("Loaded {} lines.", m_trace.size());
    };


    void tick() override {
      if (is_finished()) {
        return;
      }
      bool req_full = false;
      while(!req_full && !is_finished()) {
        const Trace& t = m_trace[m_curr_trace_idx];
        bool request_sent = false;
        auto send_req = [&] (int req_type) -> bool {
          Request req = t.has_addr_vec ? Request(t.addr_vec, req_type)
                                       : Request(t.addr, req_type);
          req.occupancy_cycles = t.occupancy_cycles;
          return m_memory_system->send(req);
        };
        switch (t.req_type) {
          case  0: request_sent = send_req(Request::Type::Read); break;
          case  1: request_sent = send_req(Request::Type::Write); break;
          case  4: request_sent = send_req(Request::Type::PIM_MAC_AB); break;
          case  5: request_sent = send_req(Request::Type::PIM_MAC_SB); break;
          case  6: request_sent = send_req(Request::Type::PIM_MAC_PB); break;
          case  7: request_sent = send_req(Request::Type::PIM_WR_GB); break;
          case  8: request_sent = send_req(Request::Type::PIM_MV_SB); break;
          case  9: request_sent = send_req(Request::Type::PIM_MV_GB); break;
          case 10: request_sent = send_req(Request::Type::PIM_SFM); break;
          case 11: request_sent = send_req(Request::Type::PIM_SET_MODEL); break;
          case 12: request_sent = send_req(Request::Type::PIM_SET_HEAD); break;
          case 13: request_sent = send_req(Request::Type::PIM_BARRIER); break;
          case 14: throw ConfigurationError("BITACC_LUT_GEN is deprecated: LUT generation is host-side and must not be emitted into Ramulator traces.");
          case 15: request_sent = send_req(Request::Type::BITACC_LUT_BCAST); break;
          case 16: throw ConfigurationError("BITACC_BLU is deprecated: BLU matmul must be modeled as LD requests plus compute latency, not a standalone trace command.");
          case 17: request_sent = send_req(Request::Type::BITACC_ACC); break;
          case 18: request_sent = send_req(Request::Type::BITACC_KV_WB); break;
          case 19: request_sent = send_req(Request::Type::BITACC_ATTN_RD); break;
          case 20: request_sent = send_req(Request::Type::BITACC_ATTN_WR); break;
          case 21: request_sent = send_req(Request::Type::BITACC_FC_RD); break;
          case 22: request_sent = send_req(Request::Type::BITACC_PRE_ATTN_NORM); break;
          case 23: request_sent = send_req(Request::Type::BITACC_PRE_ATTN_NORM_TAIL); break;
          case 24: request_sent = send_req(Request::Type::BITACC_ST_WAIT_QKV); break;
          case 25: request_sent = send_req(Request::Type::BITACC_ST_WAIT_QKV_TAIL); break;
          case 26: request_sent = send_req(Request::Type::BITACC_ALL_BANK_QKT); break;
          case 27: request_sent = send_req(Request::Type::BITACC_ALL_BANK_POST); break;
          case 28: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WRITE_P_BACK); break;
          case 29: request_sent = send_req(Request::Type::BITACC_ALL_BANK_PV); break;
          case 30: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WRITE_H_EQ_PV); break;
          case 31: request_sent = send_req(Request::Type::BITACC_WAIT_WO); break;
          case 32: request_sent = send_req(Request::Type::BITACC_WAIT_WO_TAIL); break;
          case 33: request_sent = send_req(Request::Type::BITACC_PRE_MLP_NORM); break;
          case 34: request_sent = send_req(Request::Type::BITACC_PRE_MLP_NORM_TAIL); break;
          case 35: request_sent = send_req(Request::Type::BITACC_WAIT_UP); break;
          case 36: request_sent = send_req(Request::Type::BITACC_WAIT_UP_TAIL); break;
          case 37: request_sent = send_req(Request::Type::BITACC_WAIT_GATE); break;
          case 38: request_sent = send_req(Request::Type::BITACC_WAIT_GATE_TAIL); break;
          case 39: request_sent = send_req(Request::Type::BITACC_MLP_FUSION); break;
          case 40: request_sent = send_req(Request::Type::BITACC_MLP_FUSION_TAIL); break;
          case 41: request_sent = send_req(Request::Type::BITACC_WAIT_DOWN); break;
          case 42: request_sent = send_req(Request::Type::BITACC_WAIT_DOWN_TAIL); break;
          case 43: request_sent = send_req(Request::Type::BITACC_MLP_RESIDUAL_SUM); break;
          case 44: request_sent = send_req(Request::Type::BITACC_MLP_RESIDUAL_SUM_TAIL); break;
          case 45: request_sent = send_req(Request::Type::BITACC_ALL_BANK_LUT_GEN_AND_LOAD); break;
          case 46: request_sent = send_req(Request::Type::BITACC_ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE); break;
          case 47: request_sent = send_req(Request::Type::BITACC_ALL_BANK_DEQUANT); break;
          case 48: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WRITE_K_CURRENT); break;
          case 49: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WRITE_V_CURRENT); break;
          case 50: request_sent = send_req(Request::Type::BITACC_DIMM_BUFFER_CONCAT_WRITE); break;
          case 51: request_sent = send_req(Request::Type::BITACC_ALL_BANK_ROPE); break;
          case 52: request_sent = send_req(Request::Type::BITACC_ALL_BANK_READ_K_CACHE); break;
          case 53: request_sent = send_req(Request::Type::BITACC_ALL_BANK_READ_V_CACHE); break;
          case 54: request_sent = send_req(Request::Type::BITACC_EMBEDDING); break;
          case 55: request_sent = send_req(Request::Type::BITACC_HOST_XFER); break;
          case 56: request_sent = send_req(Request::Type::BITACC_CPU_NORM); break;
          case 57: request_sent = send_req(Request::Type::BITACC_LM_HEAD); break;
          case 58: request_sent = send_req(Request::Type::BITACC_TOPK); break;
          case 59: request_sent = send_req(Request::Type::BITACC_RMS_ATTN); break;
          case 60: request_sent = send_req(Request::Type::BITACC_RMS_MLP); break;
          case 61: request_sent = send_req(Request::Type::BITACC_QUANT); break;
          case 62: request_sent = send_req(Request::Type::BITACC_MLP_FUSION_V2); break;
          case 63: request_sent = send_req(Request::Type::BITACC_FINAL_RES); break;
          case 64: request_sent = send_req(Request::Type::BITACC_BLOCK_END); break;
          case 65: request_sent = send_req(Request::Type::BITACC_ALL_BANK_LUT_LOAD); break;
          case 66: request_sent = send_req(Request::Type::BITACC_ALL_BANK_BITLINEAR); break;
          case 67: request_sent = send_req(Request::Type::BITACC_ALL_BANK_DEQUANT_ROPE); break;
          case 68: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WR_K_CURRENT); break;
          case 69: request_sent = send_req(Request::Type::BITACC_ALL_BANK_WR_V_CURRENT); break;
          case 70: request_sent = send_req(Request::Type::BITACC_ALL_BANK_RD_K_CACHE); break;
          case 71: request_sent = send_req(Request::Type::BITACC_ALL_BANK_QKT_V2); break;
          case 72: request_sent = send_req(Request::Type::BITACC_ALL_BANK_POST_V2); break;
          case 73: request_sent = send_req(Request::Type::BITACC_ALL_BANK_RD_V_CACHE); break;
          case 74: request_sent = send_req(Request::Type::BITACC_ALL_BANK_PV_V2); break;
          case 75: request_sent = send_req(Request::Type::BITACC_ATTN_XFER); break;
          case 76: request_sent = send_req(Request::Type::BITACC_LINEAR_XFER); break;
          case 77: request_sent = send_req(Request::Type::BITACC_ALL_BANK_RD_K_V_CACHE); break;
          case 78: request_sent = send_req(Request::Type::BITACC_QKT_XFER); break;
          case 79: request_sent = send_req(Request::Type::BITACC_PV_XFER); break;
          case 80: request_sent = send_req(Request::Type::BITACC_POST_SCORE_BCAST); break;
          case 81: request_sent = send_req(Request::Type::BITACC_TOKEN_DONE); break;
          case 82: request_sent = send_req(Request::Type::BITACC_QKT_REDUCE); break;
          default:;
        }
        if (request_sent) {
          m_curr_trace_idx = (m_curr_trace_idx + 1) % m_trace_length;
          m_trace_count++;
        }
        else {
          req_full = true;
        }
      }
    };


  private:
    void init_trace(const std::string& file_path_str) {
      fs::path trace_path(file_path_str);
      if (!fs::exists(trace_path)) {
        throw ConfigurationError("Trace {} does not exist!", file_path_str);
      }

      std::ifstream trace_file(trace_path);
      if (!trace_file.is_open()) {
        throw ConfigurationError("Trace {} cannot be opened!", file_path_str);
      }

      std::string line;
      while (std::getline(trace_file, line)) {
        std::vector<std::string> tokens;
        tokenize(tokens, line, " ");

        // TODO: Add line number here for better error messages
        if (tokens.size() != 2 && tokens.size() != 7 && tokens.size() != 8) {
          throw ConfigurationError("Trace {} format invalid!", file_path_str);
        }

        int req_type = -1; 
        if (tokens[0] == "LD") {
          req_type = 0;
        } else if (tokens[0] == "ST") {
          req_type = 1;
        } else if (tokens[0] == "PIM_MAC_AB") {
          req_type = 4;
        } else if (tokens[0] == "PIM_MAC_SB") {
          req_type = 5;
        } else if (tokens[0] == "PIM_MAC_PB") {
          req_type = 6;
        } else if (tokens[0] == "PIM_WR_GB") {
          req_type = 7;
        } else if (tokens[0] == "PIM_MV_SB") {
          req_type = 8;
        } else if (tokens[0] == "PIM_MV_GB") {
          req_type = 9;
        } else if (tokens[0] == "PIM_SFM") {
          req_type = 10;
        } else if (tokens[0] == "PIM_SET_MODEL") {
          req_type = 11;
        } else if (tokens[0] == "PIM_SET_HEAD") {
          req_type = 12;
        } else if (tokens[0] == "PIM_BARRIER") {
          req_type = 13;
        } else if (tokens[0] == "BITACC_LUT_GEN") {
          throw ConfigurationError("Trace {} contains deprecated token BITACC_LUT_GEN. LUT generation is host-side and must not enter Ramulator command flow.", file_path_str);
        } else if (tokens[0] == "BITACC_LUT_BCAST") {
          req_type = 15;
        } else if (tokens[0] == "BITACC_BLU") {
          throw ConfigurationError("Trace {} contains deprecated token BITACC_BLU. Model BLU as LD traffic plus compute latency.", file_path_str);
        } else if (tokens[0] == "BITACC_ACC") {
          req_type = 17;
        } else if (tokens[0] == "BITACC_KV_WB") {
          req_type = 18;
        } else if (tokens[0] == "BITACC_ATTN_RD") {
          req_type = 19;
        } else if (tokens[0] == "BITACC_ATTN_WR") {
          req_type = 20;
        } else if (tokens[0] == "BITACC_FC_RD") {
          req_type = 21;
        } else if (tokens[0] == "PRE_ATTN_NORM") {
          req_type = 22;
        } else if (tokens[0] == "PRE_ATTN_NORM_TAIL") {
          req_type = 23;
        } else if (tokens[0] == "ST_WAIT_QKV") {
          req_type = 24;
        } else if (tokens[0] == "ST_WAIT_QKV_TAIL") {
          req_type = 25;
        } else if (tokens[0] == "ALL_BANK_QKT") {
          req_type = 26;
        } else if (tokens[0] == "ALL_BANK_POST") {
          req_type = 27;
        } else if (tokens[0] == "ALL_BANK_WRITE_P_BACK") {
          req_type = 28;
        } else if (tokens[0] == "ALL_BANK_PV") {
          req_type = 29;
        } else if (tokens[0] == "ALL_BANK_WRITE_H_EQ_PV") {
          req_type = 30;
        } else if (tokens[0] == "WAIT_WO") {
          req_type = 31;
        } else if (tokens[0] == "WAIT_WO_TAIL") {
          req_type = 32;
        } else if (tokens[0] == "PRE_MLP_NORM") {
          req_type = 33;
        } else if (tokens[0] == "PRE_MLP_NORM_TAIL") {
          req_type = 34;
        } else if (tokens[0] == "WAIT_UP") {
          req_type = 35;
        } else if (tokens[0] == "WAIT_UP_TAIL") {
          req_type = 36;
        } else if (tokens[0] == "WAIT_GATE") {
          req_type = 37;
        } else if (tokens[0] == "WAIT_GATE_TAIL") {
          req_type = 38;
        } else if (tokens[0] == "MLP_FUSION") {
          req_type = 39;
        } else if (tokens[0] == "MLP_FUSION_TAIL") {
          req_type = 40;
        } else if (tokens[0] == "WAIT_DOWN") {
          req_type = 41;
        } else if (tokens[0] == "WAIT_DOWN_TAIL") {
          req_type = 42;
        } else if (tokens[0] == "MLP_RESIDUAL_SUM") {
          req_type = 43;
        } else if (tokens[0] == "MLP_RESIDUAL_SUM_TAIL") {
          req_type = 44;
        } else if (tokens[0] == "ALL_BANK_LUT_GEN_AND_LOAD") {
          req_type = 45;
        } else if (tokens[0] == "ALL_BANK_LUT_WEIGHT_FETCH_AND_COMPUTE" ||
                   tokens[0] == "ALL_BANK_READ_AND_LAUNCH") {
          req_type = 46;
        } else if (tokens[0] == "ALL_BANK_DEQUANT") {
          req_type = 47;
        } else if (tokens[0] == "ALL_BANK_WRITE_K_CURRENT") {
          req_type = 48;
        } else if (tokens[0] == "ALL_BANK_WRITE_V_CURRENT") {
          req_type = 49;
        } else if (tokens[0] == "DIMM_BUFFER_CONCAT_WRITE") {
          req_type = 50;
        } else if (tokens[0] == "ALL_BANK_ROPE") {
          req_type = 51;
        } else if (tokens[0] == "ALL_BANK_READ_K_CACHE") {
          req_type = 52;
        } else if (tokens[0] == "ALL_BANK_READ_V_CACHE") {
          req_type = 53;
        } else if (tokens[0] == "EMBEDDING") {
          req_type = 54;
        } else if (tokens[0] == "HOST_XFER") {
          req_type = 55;
        } else if (tokens[0] == "CPU_NORM") {
          req_type = 56;
        } else if (tokens[0] == "LM_HEAD") {
          req_type = 57;
        } else if (tokens[0] == "TOPK") {
          req_type = 58;
        } else if (tokens[0] == "RMS_ATTN") {
          req_type = 59;
        } else if (tokens[0] == "RMS_MLP") {
          req_type = 60;
        } else if (tokens[0] == "QUANT") {
          req_type = 61;
        } else if (tokens[0] == "MLP_FUSION") {
          req_type = 62;
        } else if (tokens[0] == "FINAL_RES") {
          req_type = 63;
        } else if (tokens[0] == "BLOCK_END") {
          req_type = 64;
        } else if (tokens[0] == "ALL_BANK_LUT_LOAD") {
          req_type = 65;
        } else if (tokens[0] == "ALL_BANK_BITLINEAR") {
          req_type = 66;
        } else if (tokens[0] == "ALL_BANK_DEQUANT_ROPE") {
          req_type = 67;
        } else if (tokens[0] == "ALL_BANK_WR_K_CURRENT") {
          req_type = 68;
        } else if (tokens[0] == "ALL_BANK_WR_V_CURRENT") {
          req_type = 69;
        } else if (tokens[0] == "ALL_BANK_RD_K_CACHE") {
          req_type = 70;
        } else if (tokens[0] == "ALL_BANK_QKT") {
          req_type = 71;
        } else if (tokens[0] == "ALL_BANK_POST") {
          req_type = 72;
        } else if (tokens[0] == "ALL_BANK_RD_V_CACHE") {
          req_type = 73;
        } else if (tokens[0] == "ALL_BANK_PV") {
          req_type = 74;
        } else if (tokens[0] == "ATTN_XFER") {
          req_type = 75;
        } else if (tokens[0] == "LINEAR_XFER") {
          req_type = 76;
        } else if (tokens[0] == "ALL_BANK_RD_K_V_CACHE") {
          req_type = 77;
        } else if (tokens[0] == "QKT_XFER") {
          req_type = 78;
        } else if (tokens[0] == "PV_XFER") {
          req_type = 79;
        } else if (tokens[0] == "POST_SCORE_BCAST") {
          req_type = 80;
        } else if (tokens[0] == "TOKEN_DONE") {
          req_type = 81;
        } else if (tokens[0] == "QKT_REDUCE") {
          req_type = 82;
        } else {
          throw ConfigurationError("Trace {} format invalid!", file_path_str);
        }

        Clk_t occupancy_cycles = 0;
        if (tokens.size() == 8) {
          occupancy_cycles = std::stoll(tokens[7]);
          if (occupancy_cycles < 0) {
            throw ConfigurationError("Trace {} has a negative command occupancy!", file_path_str);
          }
        }

        if (tokens.size() == 7 || tokens.size() == 8) {
          AddrVec_t addr_vec;
          addr_vec.reserve(6);
          for (size_t i = 1; i <= 6; i++) {
            addr_vec.push_back(std::stoll(tokens[i]));
          }
          m_trace.push_back({req_type, -1, addr_vec, true, occupancy_cycles});
        } else {
          Addr_t addr = -1;
          if (tokens[1].compare(0, 2, "0x") == 0 || tokens[1].compare(0, 2, "0X") == 0) {
            addr = std::stoll(tokens[1].substr(2), nullptr, 16);
          } else {
            addr = std::stoll(tokens[1]);
          }
          m_trace.push_back({req_type, addr, {}, false, 0});
        }
      }

      trace_file.close();

      m_trace_length = m_trace.size();
    };

    // TODO: FIXME
    bool is_finished() override {
      return m_trace_count >= m_trace_length; 
    };
};

}        // namespace Ramulator
