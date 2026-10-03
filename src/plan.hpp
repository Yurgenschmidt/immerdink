// Execute-time entry points shared by the host API (api.cpp) and the CUDA translation units
// (contract.cu, gemm_k4.cu). The plan's data (ExecPlan, imd_plan_st) is CUDA-free and lives
// in libimmerdink_host (src/host/plan_data.hpp); C++17 (it is included by nvcc TUs).
#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "host/k4_launches.hpp"
#include "host/plan_data.hpp"
#include "immerdink/immerdink.h"

namespace imd::detail {

// Enqueue K0-K5 (contract.cu). A and B are the GEMM's operands (already swapped by the
// caller when p.swapped). Reads `p` only. Returns SUCCESS or CUDA_ERROR (launch errors
// only; no sync).
imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream);

// K0 (contract.cu): write the p.vec_count scatter vectors to `vec` from p.vec_desc, on
// `stream`. Also builds the resident copy at plan creation. SUCCESS or CUDA_ERROR.
imd_status_t k0_generate(const ExecPlan& p, int64_t* vec, cudaStream_t stream);

// K4's current launch limits (gemm_k4.cu; host/k4_launches.hpp LaunchLimits): the
// production values unless the test hook imd_test_set_limits changed them. k4_prepare reads
// them, so they apply to plans created afterwards.
LaunchLimits& launch_limits();
LaunchLimits default_launch_limits();

// K4 setup at plan creation (gemm_k4.cu, DESIGN v1.2 §6): for every launch shape of the
// call (K-chunk length, batch count, n-hat part; store-mod / add-mod), the kernel parameters
// of K4 configuration `cfg` (IMD_K4_*, DESIGN v1.5 §4), built from the device's SM count and
// the kernels' occupancy, after setting their dynamic shared-memory attribute. All of this
// host state is built here, never in imd_contract. SUCCESS, CUDA_ERROR, INVALID_VALUE (no
// such configuration), or NOT_SUPPORTED if a shape is out of K4's limits.
imd_status_t k4_prepare(const ExecPlan& p, int cfg, std::shared_ptr<const K4Plan>* out);

// Plan-time autotuning (imd_options_t.k4_tune, DESIGN v1.5 §6): with k4_tune_bytes(p) > 0,
// allocate that much device memory, time group 1's K4 launches once (after one warm-up run)
// in every configuration on cudaStreamPerThread, free it, and set *cfg to the fastest (the
// incoming *cfg, the rule's choice, unless another is strictly faster). With
// k4_tune_bytes(p) = 0, nothing runs and *cfg is unchanged. SUCCESS, ALLOC_FAILED (the
// temporary allocation) or CUDA_ERROR. `report` (optional): the times, for benchmarks.
struct K4TuneReport {
  bool tuned = false;
  float ms[IMD_K4_CONFIG_COUNT] = {};
};
imd_status_t k4_tune(const ExecPlan& p, int32_t* cfg, K4TuneReport* report);

// Test / benchmark hook state (gemm_k4.cu; test_hooks.cpp imd_test_set_k4_config): -1 = the
// plan-time rule or k4_tune (production), else the IMD_K4_* every later plan runs. Read by
// imd_plan_create only.
int& k4_config_override();

// K4 for one residue group (gemm_k4.cu): every K-chunk (store-mod, then add-mod) of the
// batched int8 TN GEMMs W_l[lambda] = RA_s[lambda] RB_s[lambda]^T mod p_l, l = g0 + s,
// s < gsize, over RA [s][lambda][m][k_hat], RB [s][lambda][n_hat][k_hat] and the u8 planes
// W [l][lambda][m][n_hat]. Splits the batch count (<= 65535 per launch) and n_hat
// (gridDim.y <= 65535). Copies the prepared parameters, sets this call's pointers and
// launches; reads `k4` only. Returns SUCCESS or CUDA_ERROR.
imd_status_t k4_group(const K4Plan& k4, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
                      int gsize, cudaStream_t stream);

// Test hook state (api.cpp; test_hooks.cpp imd_test_set_fused): -1 = the plan-time rule
// (production), 0 = never take the v1.4 fused path. Read by imd_plan_create only.
int& fused_override();
// K5's one-element-per-thread limit (contract.cu; elements per SM), read by imd_contract;
// changed only by the test hook imd_test_set_k5_one_elem (not thread-safe, like the hooks).
int64_t& k5_one_elem_per_sm();

// The execution data of a plan (api.cpp), for the test hooks; nullptr for a null plan.
const ExecPlan* exec_of(imd_plan_t plan);

// The K4 parameter sets a plan holds (gemm_k4.cu): store-mod and add-mod shape keys, the
// launches of one call, and the host payload of the K4Plan (sizeof(K4Plan) plus its map
// entries), for the test hook that checks libimmerdink_host's sizing against a real plan.
struct K4Footprint {
  int64_t store_shapes = 0, add_shapes = 0, launches = 0;
  uint64_t bytes = 0;
  int32_t config = 0;  // the configuration whose parameters the plan holds (v1.5)
};
K4Footprint k4_footprint(const K4Plan& k4, int N, int g);

}  // namespace imd::detail
