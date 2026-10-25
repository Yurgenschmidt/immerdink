// Execute-time entry points shared by the host API (api.cpp) and the CUDA translation units
// (contract.cu, gemm_k4.cu). The plan's data (ExecPlan, imd_plan_st) is CUDA-free and lives
// in libimmerdink_host (src/host/plan_data.hpp); C++17 (it is included by nvcc TUs).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "host/k4_launches.hpp"
#include "host/pack.hpp"
#include "host/plan_data.hpp"
#include "immerdink/immerdink.h"

namespace imd::detail {

// Enqueue K0-K5 (contract.cu). A and B are the GEMM's operands (already swapped by the
// caller when p.swapped). Reads `p` only. Returns SUCCESS or CUDA_ERROR (launch errors
// only; no sync). v1.7 (DESIGN §6b): packed[x] is the packed buffer of the GEMM's operand x
// (post-swap) when p.packed has bit x (read only: its sigma, flags and residues at
// p.pack_off[x]); nullptr otherwise or when it is not read.
imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream,
                              const void* const* packed = nullptr);

// imd_pack (contract.cu, DESIGN v1.7 §6b): the packed bytes of X on `stream`. `p` is the pack's
// pack_exec (libimmerdink_host) with kl_lg_rows from pack_kl_lg_rows; `pl` its layout
// (offsets, region bytes, total). Writes X's scatter vectors into `ws` (pack_ws_bytes), every
// gap and the tail of `packed` with 0, then one KL launch: sigma, flags and all N residue
// planes. SUCCESS or CUDA_ERROR (launch errors only; no sync, no allocation).
imd_status_t enqueue_pack(const ExecPlan& p, const PackLayout& pl, const double* X, void* packed,
                          void* ws, cudaStream_t stream);
// The KL tile height of imd_pack (a launch choice): reading along rows, rows of fewer than
// 512 k-bar get taller tiles (up to 64 rows: whole short rows side by side); -1 = KL's rule.
int32_t pack_kl_lg_rows(const ExecPlan& p);

// v1.10 (DESIGN §6c.1, benchmark use: imd_prof_kernel_occupancy): out[0..3] = CTAs per SM,
// registers per thread, static shared memory bytes and threads of the launch, of kernel
// IMD_PERF_KERNEL_* (the K4 ones excepted: cudaErrorInvalidValue).
cudaError_t kernel_occupancy(int32_t kernel, int32_t* out);
// v1.16 §3.4.1: the same row for one pack-emitter word count W in {2, 3} of a pack kernel
// (KF16, KF4, KL8, KL4, PACK); W = 0 is kernel_occupancy's row (a pack kernel's most
// constrained W: fewest CTAs per SM, then most registers). cudaErrorInvalidValue otherwise.
cudaError_t kernel_occupancy_w(int32_t kernel, int32_t w, int32_t* out);

// Issue 18 (docs/NOTES-18.md D4): the binaryVersion / ptxVersion cudaFuncGetAttributes reports for
// kernel IMD_PERF_KERNEL_* (K4 excepted, see k4_ops.hpp K4Ops::versions); w as in kernel_occupancy_w (a pack
// kernel is looked at for one W instantiation; 0 for the others). The error of the CUDA call, else success.
cudaError_t kernel_versions(int32_t kernel, int32_t w, int32_t* binary, int32_t* ptx);
// The same for the launches without an IMD_PERF_KERNEL_* id (index 0 .. kExtraKernels - 1; cudaErrorInvalidValue
// beyond). Their failing_kernel id in imd_device_info_t is IMD_PERF_KERNEL_COUNT + index.
constexpr int32_t kExtraKernels = 11;
cudaError_t extra_kernel_versions(int32_t index, int32_t* binary, int32_t* ptx);

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
// allocate that much device memory, time group 1's K4 launches in every configuration on
// cudaStreamPerThread (a warm-up run, then 3 interleaved rounds of ~0.1 ms batches; the
// minimum per run), free it, and set *cfg to the fastest (the incoming *cfg, the rule's
// choice, unless another is strictly faster). With
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
// launches; reads `k4` only. Returns SUCCESS or CUDA_ERROR. batch_a / batch_b (v1.7, a packed
// operand, DESIGN §6b.4): the operand's batch stride z -> z * batch (0: the workspace's
// m k_hat / n_hat k_hat).
imd_status_t k4_group(const K4Plan& k4, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
                      int gsize, cudaStream_t stream, int64_t batch_a, int64_t batch_b);

// v1.13 §4b.5 (hygiene): the operand-pointer check (k4_align_check) of a whole call, run on
// the host BEFORE anything of the call is queued: for every residue group g0 (groups of g of
// N moduli) and every K4 launch, the A and B pointers call_of forms from RA + g0 group_a and
// RB + g0 group_b (group_x: the per-modulus offset of a packed operand's planes, b batch_x;
// 0 for the workspace's slots) are = 0 mod k_align. Only when the check
// is on (debug builds, or the test hook) and the plan's K-chunk is a multiple of k_align;
// counts every checked launch. SUCCESS, or CUDA_ERROR on a violation in a debug build (the
// call then fails with nothing queued; release builds only count it).
imd_status_t k4_check_call(const K4Plan& k4, int N, int g, const int8_t* RA, const int8_t* RB,
                           uint8_t* W, int64_t group_a, int64_t group_b, int64_t batch_a,
                           int64_t batch_b, int32_t k_align);

// v1.13 §4b.5 (gemm_k4.cu; test hook imd_test_k4_align_check): mode 1 = enable the check that
// every K4 operand pointer is = 0 mod the plan's A and reset its counts, 0 = disable, -1 =
// query only (default: on in debug builds), 2 = return the launches checked since the last
// reset, 3 = as 1 with a test mutation (the check sees the A pointer + 8 bytes; the launch is
// unchanged: every checked launch is a violation). Returns the violations counted since the
// last reset (mode 2: the checked launches).
int64_t k4_align_check(int mode);

// Test hook state (api.cpp; test_hooks.cpp imd_test_set_fused): -1 = the plan-time rule
// (production), 0 = no fused launch (KF, KL), 2 / 3 = KL for both operands of every emulated
// plan (3: 64-row KL tiles). Read by imd_plan_create only.
int& fused_override();
// K5's one-element-per-thread limit (contract.cu; elements per SM), read by imd_contract;
// changed only by the test hook imd_test_set_k5_one_elem (not thread-safe, like the hooks).
int64_t& k5_one_elem_per_sm();
// A5: K5's elements per thread forced to 1, 2 or 4 (0: the launch rule), read by
// imd_contract; changed only by the hooks imd_test_set_k5_elems / imd_prof_set_k5_elems.
int& k5_elems_override();
std::atomic<int>& seq_override();      // issue 21: -1 the plan's set (host/seq_config.hpp), else a forced set of kSeq* bits for every plan (hooks only; relaxed)
std::atomic<int>& k5_crt_override();   // issue 21: 0 integer CRT, 1 binary64-limb CRT (process-wide: all devices, streams and plans; a graph captures it at capture time)
// v1.13 §4b (hygiene; contract.cu, test hook imd_test_set_pad_mutation): -1 (production) or
// the post-swap operand x (0 = the GEMM's A, 1 = its B) whose K2 / K3 pack stores only k-bar
// < ceil(k)_16 (Pipeline neg_a64_pad_old_k3: [ceil(k)_16, k_hat) left unwritten on one side
// only, which the bit-exact check cannot see; initcheck must). Read by imd_contract with a
// relaxed load (an atomic so a hook call racing a contract on another thread is not a data race).
std::atomic<int>& pad_mutation();

// The execution data of a plan (api.cpp), for the test hooks; nullptr for a null plan.
const ExecPlan* exec_of(imd_plan_t plan);

// A pack handle as plan creation's analysis sees it (api.cpp): its labels by position and its
// lowering. The handle must be non-null and outlive the PackSource.
PackSource pack_source_of(imd_pack_t pack);

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
