// IMD_HWGEMM (docs/DESIGN.md §6i): the hardware FP64 tensor-core GEMM of one contraction.
//
// This header is CUTLASS-free: contract.cu fills an HwGemmCall from the plan and calls
// hwgemm_enqueue; the CUTLASS instantiations live in src/hwgemm/*.cu.
//
// The GEMM view (post-swap roles, DESIGN §5): C[i, j] = sum_h A(i, h) B(j, h) for each batch
// index lambda. Each operand is a "rows x K" view; C is "I x J". An affine view (§6i.2,
// spec/IndexMap.tla HwAcc) is read through a 2-D layout:
//   k_contig (ROW): element (r, h) at base + r * ld + h        (K contiguous)
//   otherwise (COL): element (r, h) at base + r + h * ld       (rows contiguous)
// and batch lambda adds lambda * bstride (any sign; 0 = broadcast).
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "immerdink/immerdink_host.h"
#include "host/hw_rules.hpp"

namespace imd::detail {

struct HwView2 {
  const double* base = nullptr;  // element (0, 0) of batch 0
  int64_t ld = 0;                // the non-unit stride (>= 0)
  bool k_contig = true;          // ROW: second index contiguous
  int64_t bstride = 0;           // batch stride (any sign)
};

// Tile configurations (§6i.7). sm_90 (Sm90 16x8x4 kernel): all three; sm_80 / sm_89 (Sm80
// 8x8x4): kHw64x64 only (correctness, not tuned). Bit-identical to each other and to DIRECT
// without slabs (§6i.4; bench/hwgemm_bitid.cu).
// HwTile, kHw*: host/hw_rules.hpp (shared with the host library's performance model)

struct HwGemmCall {
  int64_t m = 0, n = 0, k = 0, b = 0;  // each <= 2^31 - 1 (§6i.6, checked at plan time)
  HwView2 A, B;                        // A: m x k, B: n x k (rows x K views)
  double* C = nullptr;                 // element (0, 0) of batch 0
  int64_t ldc = 0;                     // C's non-unit stride
  bool c_j_contig = true;              // ROW: (i, j) at i * ldc + j; COL: i + j * ldc
  int64_t c_bstride = 0;
  double alpha = 1.0, beta = 0.0;      // §3.7 store_out: E = a == 1 ? D : RN(a D); C = b == 0 ? E : fma(b, C, E)
  int32_t tile = -1;                   // HwTile, or -1: the rule (hwgemm_tile)
};

// The provisional tile rule (NOTES-HWGEMM §4.3 gate, 6 shapes on H100; the per-shape rule is
// a deliverable of the tuning step): 64x64 when a 64-row tile already covers m or n, else
// 64x128 when 128x128 tiles would fill fewer than half the SMs, else 128x128. sm_80 / sm_89:
// always 64x64.
int32_t hwgemm_tile(int64_t m, int64_t n, int64_t b, int cc_major, int sms);

// Test / benchmark hook (imd_test_set_hw_tile, imd_prof_set_hw_tile): the tile of HWGEMM plans created afterwards,
// -1 = the rule. sm_80 / sm_89 have kHw64x64 only (another tile: INVALID_VALUE per call).
int& hw_tile_override();

// Queue the GEMM(s) of one call on `stream` (chunks of <= 65535 batches on grid.z and of
// <= 65535 column tiles on grid.y). SUCCESS, NOT_SUPPORTED (no kernel for this device), or
// CUDA_ERROR (a launch failed). Graph-capture safe: no allocation, no synchronization.
imd_status_t hwgemm_enqueue(const HwGemmCall& c, cudaStream_t stream);

// §3.7 alpha = 0 on a fully affine C (no vectors): C <- beta == 0 ? +0 : RN(beta C), the
// operations of contract.cu k5_beta_only; A and B are not read.
imd_status_t hwgemm_beta(const HwGemmCall& c, cudaStream_t stream);

// ---- Stage 2 (§6i.2): calls that are not fully affine -------------------------------------
// One GEMM operand as the own kernel reads it: a 2-D view (an in-place affine operand, or a
// pack: base = the PA / PB region, ld = k_hat, ROW) and the batch as a stride or, when the
// batch has several modes and the operand is in place, per-lambda offsets (the vec region).
struct HwOwnOperand {
  const double* base = nullptr;
  int64_t ld = 0;
  bool k_contig = true;
  int64_t bstride = 0;
  const int64_t* bat = nullptr;  // non-null: element offset of batch lambda = bat[lambda]
};
struct HwOwnCall {
  int64_t m = 0, n = 0, k = 0, b = 0;  // k: the GEMM's K extent (k, or k_hat when both packed)
  HwOwnOperand A, B;                   // A: m x k, B: n x k
  double* C = nullptr;
  int64_t c_si = 0, c_sj = 0, c_bstride = 0;                 // affine C
  const int64_t *c_row = nullptr, *c_col = nullptr, *c_bat = nullptr;  // non-null: vectors
  double alpha = 1.0, beta = 0.0;
  int32_t tile = -1;
};
// The own GEMM kernel (src/hwgemm/hw_own.cuh): SUCCESS, NOT_SUPPORTED (no kernel for this
// device), INVALID_VALUE (tile), CUDA_ERROR. Same chunking as hwgemm_enqueue.
imd_status_t hwgemm_own_enqueue(const HwOwnCall& c, cudaStream_t stream);

// The FP64 pack (§6i.2): dst[(lambda * rows + r) * k_hat + h] = src[bat[lambda] + row[r] +
// kk[h]] for h < k, and `pad` for k <= h < k_hat (A: -0.0, B: +0.0, §6i.4). Element offsets.
struct HwPackCall {
  const double* src = nullptr;
  const int64_t *bat = nullptr, *row = nullptr, *kk = nullptr;
  int64_t rows = 0, k = 0, k_hat = 0, b = 0;
  double pad = 0.0;
  double* dst = nullptr;
  // host/index_map.hpp pack_pivot: the source's fastest index runs along the packed rows
  // rho = r + rows * lambda, as rho = lo + radix * d (true: the tiled smem-transpose pack;
  // false: K is fastest and the element-wise pack's reads are already coalesced)
  bool along_rows = false;
  int64_t radix = 1;
};
imd_status_t hwgemm_pack(const HwPackCall& c, cudaStream_t stream);

// Test hook (imd_test_set_hw_own): 1 runs fully affine calls through the own kernel too (the
// bit comparison against stage 1's CUTLASS GemmUniversal); 0 (default): GemmUniversal.
int& hw_own_override();
// Test hook (imd_test_set_hw_own_vec2): 1 (default) lets the own kernel's epilogue store a
// thread's two adjacent C columns as one 16-byte store when C is affine with unit column
// stride (hw_own.cuh); 0: element-wise stores (the A/B timing; the values are the same).
bool& hw_own_vec2();

// Device-status coverage (issue 18 D4, src/device_state.cpp): the code versions of one kernel
// per HWGEMM module: 0 hw_beta_kernel (hwgemm.cu), 1 the Sm80 GEMM (hw_sm80.cu), 2 the Sm90
// GEMM (hw_sm90.cu), 3 hw_pack_kernel (hwgemm.cu), 4 / 5 the own Sm80 / Sm90 GEMM
// (hw_own_sm80.cu, hw_own_sm90.cu), 6 hw_pack_rows_kernel (hwgemm.cu). cudaErrorInvalidValue
// for another index.
constexpr int32_t kHwKernelVersionCount = 7;
cudaError_t hw_kernel_versions(int32_t index, int32_t* binary, int32_t* ptx);

}  // namespace imd::detail
