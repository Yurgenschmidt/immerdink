// Execute-time plan data shared by the host API (api.cpp) and the CUDA translation units
// (contract.cu, gemm_k4.cu). Plain data; C++17 (it is included by nvcc TUs).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "immerdink/immerdink.h"
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
  WorkspaceLayout layout;

  // K2 / K3 gather tiling (index_map.hpp pack_pivot), for the GEMM's A and B operands.
  bool pack_along_rows[2] = {false, false};
  int64_t pack_radix[2] = {1, 1};
  // K5 thread mapping: lambda fastest (C's fastest label is a batch label) or j-bar fastest.
  bool k5_batch_fast = false;

  // §5 step 10 scatter vectors (§4a vec order, vec_count entries), from vec_desc: generated
  // by K0 into ws.vec per call (IMD_VEC_GENERATE), or built once at plan creation into
  // vec_dev, plan-owned device memory freed by the plan (IMD_VEC_RESIDENT, v1.2).
  VecDesc vec_desc;
  int64_t vec_count = 0;
  bool vec_resident = false;
  int64_t* vec_dev = nullptr;

  int sms = 0;                         // SM count of the device at plan creation (K1 grids)
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

// Enqueue K0-K5 (contract.cu). A and B are the GEMM's operands (already swapped by the
// caller when p.swapped). Reads `p` only. Returns SUCCESS or CUDA_ERROR (launch errors
// only; no sync).
imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream);

// K0 (contract.cu): write the p.vec_count scatter vectors to `vec` from p.vec_desc, on
// `stream`. Also builds the resident copy at plan creation. SUCCESS or CUDA_ERROR.
imd_status_t k0_generate(const ExecPlan& p, int64_t* vec, cudaStream_t stream);

// Launch limits of K4 (gemm_k4.cu). Production values are the hardware / DESIGN limits:
// K-chunk K_c' (§3.5), batch count <= 65535 (gridDim.z) and N tiles <= 65535 (gridDim.y)
// per launch. Only the test hook imd_test_set_limits (test_hooks.cpp) changes them, to
// exercise the chunk / batch / n-hat splits on small shapes; results must not change.
// k4_prepare reads them, so they apply to plans created afterwards.
struct LaunchLimits {
  int64_t k_chunk;      // multiple of 16 in [16, K_c']
  int64_t max_batch;    // [1, 65535]
  int64_t max_n_tiles;  // [1, 65535]
};
LaunchLimits& launch_limits();
LaunchLimits default_launch_limits();

// K4 setup at plan creation (gemm_k4.cu, DESIGN v1.2 §6): for every launch shape of the
// call (K-chunk length, batch count, n-hat part; store-mod / add-mod), the CUTLASS kernel
// parameters, built from the device's SM count and the kernels' occupancy, after setting
// their dynamic shared-memory attribute. All CUTLASS host state is touched here, never in
// imd_contract. SUCCESS, CUDA_ERROR, or NOT_SUPPORTED if CUTLASS rejects a shape.
imd_status_t k4_prepare(const ExecPlan& p, std::shared_ptr<const K4Plan>* out);

// K4 for one residue group (gemm_k4.cu): every K-chunk (store-mod, then add-mod) of the
// batched int8 TN GEMMs W_l[lambda] = RA_s[lambda] RB_s[lambda]^T mod p_l, l = g0 + s,
// s < gsize, over RA [s][lambda][m][k_hat], RB [s][lambda][n_hat][k_hat] and the u8 planes
// W [l][lambda][m][n_hat]. Splits the batch count (<= 65535 per launch) and n_hat
// (gridDim.y <= 65535). Copies the prepared parameters, sets this call's pointers and
// launches; reads `k4` only. Returns SUCCESS or CUDA_ERROR.
imd_status_t k4_group(const K4Plan& k4, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
                      int gsize, cudaStream_t stream);

// The execution data of a plan (api.cpp), for the test hooks; nullptr for a null plan.
const ExecPlan* exec_of(imd_plan_t plan);

}  // namespace imd::detail
