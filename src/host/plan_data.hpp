// The plan's data (DESIGN §4, §6): everything imd_plan_create derives from the descriptors
// and options on the host, as plain CUDA-free C++17 (also included by the nvcc translation
// units through src/plan.hpp). libimmerdink_host builds it (analysis.hpp plan_data) and
// sizes it (plan_host_bytes); libimmerdink adds the device-side parts (K4 parameters,
// resident vectors) and runs it.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "immerdink/immerdink_host.h"
#include "index_map.hpp"
#include "workspace.hpp"

namespace imd::detail {

// v1.2 K0 (DESIGN §4): the §5 step 9 fused modes, post-swap, per class I, J, K, L (in
// LabelClass order): modes [first[c], first[c] + count[c]) of `modes`, first fastest, with
// the extent and the stride in A, B and C (0 where the tensor does not carry the class).
// The scatter vectors are a pure function of it (index_map.hpp build_scatter_vectors).
struct VecMode {
  int64_t extent = 0;
  int64_t stride[3] = {0, 0, 0};
};
constexpr int kMaxVecModes = 256;  // >= 4 classes x 62 fused modes (extent >= 2, product < 2^63)
struct VecDesc {
  int32_t first[4] = {0, 0, 0, 0}, count[4] = {0, 0, 0, 0};
  std::vector<VecMode> modes;
};

// K4's launch parameters, built at plan creation (gemm_k4.cu); opaque elsewhere.
struct K4Plan;

enum class Path : int {
  kNoop,    // empty output (b*m*n = 0): nothing is touched
  kGemm,    // EMULATE with k >= 2: K0 K1 (K2 K3 K4)* K5      (alpha = 0 or k = 0 -> kScale)
  kDirect,  // mode DIRECT or K-empty (k = 1 after lowering): K0 + native kernel
};

struct ExecPlan {
  Path path = Path::kNoop;
  bool swapped = false;  // §5 step 8 role swap: the GEMM's A operand is the caller's B
  int64_t m = 0, n = 0, k = 0, b = 0, k_hat = 0, n_hat = 0;  // post-swap GEMM view
  int N = 0, g = 0;      // number of moduli, residue group size in [1, N]
  bool trunc = false;    // rounding = TRUNC
  uint8_t seq = 0;       // issue 21: the instruction-sequence set of the device class (host/seq_config.hpp kSeq*), set by imd_plan_create; 0 = baseline. (In the padding after trunc: sizeof(ExecPlan) is unchanged.)
  // v1.13 §4b: the plan's K alignment A (16 or 64), resolved once by libimmerdink_host
  // (analysis.hpp Analysis::k_align; plan_data), k_hat = ceil(k)_A. Here, not in K4Plan:
  // sizeof(K4Plan) is kK4PlanBytes (plan host bytes). libimmerdink never resolves A itself.
  // (In the padding after trunc: sizeof(ExecPlan) is unchanged.)
  int32_t k_align = 16;
  WorkspaceLayout layout;

  // K2 / K3 gather tiling (index_map.hpp pack_pivot), for the GEMM's A and B operands.
  bool pack_along_rows[2] = {false, false};
  int64_t pack_radix[2] = {1, 1};
  // K5 thread mapping: lambda fastest (C's fastest label is a batch label) or j-bar fastest.
  bool k5_batch_fast = false;
  // v1.4 (DESIGN §4 "KF"): K0 (generate mode), K1 and group 1's K2/K3 run as one fused
  // launch. Chosen at plan time from the lowering (analysis.hpp fused_path).
  bool fused = false;
  // v1.6 (DESIGN §4 "KL"): bit x (0 = the GEMM's A, 1 = B) set iff operand x is scaled and
  // packed for group 1 by one KL launch instead of K1 + its group-1 pack. Chosen at plan time
  // from the lowering (analysis.hpp fused_large).
  int32_t fused_large = 0;
  // Test hook only (imd_test_set_fused 3): log2 of KL's tile height (rows per tile), -1 = the
  // launch rule. Forcing 64-row tiles makes short rows span several chunks.
  int32_t kl_lg_rows = -1;
  // v1.5 (DESIGN §4 "K4 configurations"): the K4 configuration (IMD_K4_*) the plan runs:
  // k4_select for the device's SM count, or the fastest one timed with k4_tune. Set by
  // libimmerdink at plan creation (plan_data leaves IMD_K4_128x256).
  int32_t k4_config = IMD_K4_128x256;
  // v1.7 (DESIGN §6b): bit x (0 = the GEMM's A, 1 = B, post-swap) set iff operand x is read
  // from a packed operand: no K1 / KL / KF share / pack launch for it, no sX / fX / RX
  // region; its sigma, flags and residues are at pack_off (sig, flag, res) in the caller's
  // packed buffer of pack_bytes bytes (§6b.4; B role read at batch stride n k_hat).
  int32_t packed = 0;
  uint64_t pack_off[2][3] = {{0, 0, 0}, {0, 0, 0}};
  uint64_t pack_bytes[2] = {0, 0};

  // §5 step 10 scatter vectors (§4a vec order, vec_count entries), from vec_desc: generated
  // by K0 into ws.vec per call (IMD_VEC_GENERATE), or built once at plan creation into
  // vec_dev, plan-owned device memory freed by the plan (IMD_VEC_RESIDENT, v1.2).
  VecDesc vec_desc;
  int64_t vec_count = 0;
  bool vec_resident = false;
  uint64_t device_bytes = 0;  // §4a "plan device bytes" (vec_dev's size), from plan_data
  int64_t* vec_dev = nullptr;

  int sms = 0;                         // SM count of the device at plan creation (K1 grids)
  int64_t l2_bytes = 0;                // L2 bytes of the device at plan creation (K1 rows in flight, issue 21; 0: the reference 72 MiB)
  std::shared_ptr<const K4Plan> k4;    // kGemm only: every K4 launch shape's parameters

  // §3.2 step 4: u = largest integer with (S << lhs_shift) * 4^u <= rhs, where
  // rhs = R * 2^F (little-endian u64 limbs, rhs_bits = bitlen(rhs) <= 217) and
  // lhs_shift = 2 (RN: 4 S, R = (2T - kappa)^2) or 0 (TRUNC: S, R = T^2).
  uint64_t rhs[4] = {0, 0, 0, 0};
  int rhs_bits = 0;
  int lhs_shift = 0;
};

// Offsets (in elements) of the nine scatter vectors inside the vec region.
struct VecOffsets {
  int64_t a_row, a_k, a_bat, b_k, b_col, b_bat, c_row, c_col, c_bat;
};
inline VecOffsets vec_offsets(const ExecPlan& p) {
  VecOffsets o{};
  o.a_row = 0;
  o.a_k = o.a_row + p.m;
  o.a_bat = o.a_k + p.k;
  o.b_k = o.a_bat + p.b;
  o.b_col = o.b_k + p.k;
  o.b_bat = o.b_col + p.n;
  o.c_row = o.b_bat + p.b;
  o.c_col = o.c_row + p.m;
  o.c_bat = o.c_col + p.n;
  return o;
}

// An owning copy of an imd_tensor_t (the caller's arrays need not outlive the plan).
struct TensorCopy {
  std::vector<int64_t> extent, stride;
  std::vector<int32_t> label;

  explicit TensorCopy(const imd_tensor_t& t)
      : extent(t.extent, t.extent + t.rank),
        stride(t.stride, t.stride + t.rank),
        label(t.label, t.label + t.rank) {}
};

}  // namespace imd::detail

// The plan record (opaque imd_plan_t of the public header). Read-only after
// imd_plan_create returns (DESIGN v1.2 §6): imd_contract never writes it. Defined here,
// CUDA-free, so that libimmerdink_host can size it (plan_host_bytes); the destructor, which
// frees the resident vectors, is libimmerdink's (api.cpp).
struct imd_plan_st {
  imd_options_t options{};  // residue_group normalized to [1, N]
  imd::detail::Lowered lowered;
  imd::detail::ExecPlan exec;  // exec.vec_dev: IMD_VEC_RESIDENT device vectors, owned
  imd::detail::TensorCopy A, B, C;  // for M3's aliasing check (§5 row 6)
  // v1.7 (§6b): the compatibility key of the caller's A / B slot (empty: packed in-line),
  // copied at creation; imd_contract_packed compares the handle it is given with it.
  std::vector<int64_t> slot_key[2];
  // Issue 18: the device the plan was made for, its table-set selection, the code kinds of the
  // library's kernels there and the plan's K4 configuration (imd_plan_info).
  imd_device_info_t info{};

  imd_plan_st(const imd_options_t& o, const imd::detail::Lowered& lw, const imd_tensor_t& a,
              const imd_tensor_t& b, const imd_tensor_t& c)
      : options(o), lowered(lw), A(a), B(b), C(c) {}
  imd_plan_st(const imd_plan_st&) = delete;
  imd_plan_st& operator=(const imd_plan_st&) = delete;
  ~imd_plan_st();
};
