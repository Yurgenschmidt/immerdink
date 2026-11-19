// IMD_HWGEMM stage 2 (docs/DESIGN.md §6i.2, §6i.7): the immerdink-owned FP64 tensor-core GEMM
// kernel for calls that are not fully affine.
//
// Why a kernel of our own. Stage 1 runs CUTLASS's GemmUniversal, which needs every operand,
// the batch and C to be 2-D strided. Stage 2 adds three things it cannot express without a
// fork: per-lambda base offsets of a multi-mode batch (read from the scatter vectors), a C
// stored through its scatter vectors (c_bat[l] + c_row[i] + c_col[j]), and packed operands
// (which ARE strided, so they need nothing new). The mainloop is CUTLASS's own
// threadblock::DefaultMma, the exact type GemmUniversal instantiates for the same (arch, tile,
// warp, instruction, stages, layouts), called the same way (accumulator from +0, K iterations
// from 0): the sums are bit-identical to stage 1's (tests/test_hwgemm.py compares them). Only
// the prologue (operand base per lambda) and the epilogue (a register-only store_out per
// element, no shared memory) differ. Precedent: src/kernels/k4_gemm.cuh.
//
// Accumulator layout (the epilogue's coordinates). Warp-level MmaTensorOp keeps mma tile
// (mi, nj) of the warp tile at fragment offset (mi + nj * kMi) * kE (mma_tensor_op.h, the
// non-row-major accumulator branch); inside one tile, element e of a thread (lane = 4 g + t)
// is at row (e / 2) * 8 + g, column 2 t + e % 2: the PTX f64 accumulator layouts of
// m16n8k4 (kE = 4: rows g, g + 8) and m8n8k4 (kE = 2: row g). The bit comparison against
// stage 1 on every tile and layout is the probe of this mapping.
#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/layout/matrix.h"

#include "../hwgemm.hpp"

namespace imd::kernels {

// Experiment knob (issue #75 own-kernel epilogue): the paired 16-byte store is compiled in;
// hw_own_vec2() (imd_test_set_hw_own_vec2, default on) switches it at launch for A/B timing.
constexpr bool kVecStore = true;

template <class Arch, class TB_, class Warp_, class Inst_, int kStages, class LA, class LB>
struct HwOwn {
  using TB = TB_;
  using Warp = Warp_;
  using Inst = Inst_;
  // GemmUniversal's mainloop for these parameters (kernel/default_gemm.h: DefaultMma with
  // alignment 1, RowMajor accumulators, OpMultiplyAdd, SharedMemoryClearOption::kNone).
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      double, LA, 1, double, LB, 1, double, cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp, Arch, TB, Warp, Inst, kStages,
      cutlass::arch::OpMultiplyAdd, false, cutlass::gemm::SharedMemoryClearOption::kNone>::ThreadblockMma;
  using WarpCount = typename Mma::WarpCount;
  static_assert(WarpCount::kK == 1, "no split-K warps");
  static constexpr int kThreads = 32 * WarpCount::kCount;
  static constexpr int kSmemBytes = int(sizeof(typename Mma::SharedStorage));
  static constexpr int kMi = Warp::kM / Inst::kM;
  static constexpr int kNj = Warp::kN / Inst::kN;
  static constexpr int kE = Inst::kM * Inst::kN / 32;  // accumulators per thread per mma tile
  static_assert(Inst::kN == 8 && (Inst::kM == 16 || Inst::kM == 8), "f64 mma shapes only");
  static_assert(sizeof(typename Mma::FragmentC) == sizeof(double) * kMi * kNj * kE,
                "unexpected accumulator fragment");

  struct Params {
    typename Mma::IteratorA::Params pa;
    typename Mma::IteratorB::Params pb;
    const double *A, *B;
    int64_t a_bs, b_bs;                // batch strides (used when a_bat / b_bat is null)
    const int64_t *a_bat, *b_bat;      // per-lambda element offsets, or null
    double* C;
    int64_t c_si, c_sj, c_bs;          // affine C (when c_row / c_col / c_bat is null)
    const int64_t *c_row, *c_col, *c_bat;
    int m, n, k;                       // k: the GEMM's K extent (k or k_hat)
    int k_iters;
    int64_t z0;                        // lambda of blockIdx.z = 0
    int j0;                            // column of blockIdx.y = 0
    double alpha, beta;
    bool vec2;                         // C affine with c_sj == 1 (the paired store may apply)
  };

  CUTLASS_DEVICE static double store(Params const& p, double d, double c) {
    const double e = p.alpha == 1.0 ? d : __dmul_rn(p.alpha, d);  // §3.7 store_out
    return p.beta == 0.0 ? e : __fma_rn(p.beta, c, e);
  }

  CUTLASS_DEVICE static void run(Params const& p, typename Mma::SharedStorage& smem) {
    const int tid = threadIdx.x;
    const int warp_idx = __shfl_sync(0xffffffffu, tid / 32, 0);
    const int lane = tid % 32;
    const int64_t lam = p.z0 + blockIdx.z;
    const int m0 = blockIdx.x * TB::kM;
    const int n0 = p.j0 + blockIdx.y * TB::kN;

    const double* A = p.A + (p.a_bat ? p.a_bat[lam] : lam * p.a_bs);
    const double* B = p.B + (p.b_bat ? p.b_bat[lam] : lam * p.b_bs);
    typename Mma::IteratorA ia(p.pa, const_cast<double*>(A), {p.m, p.k}, tid, {m0, 0});
    typename Mma::IteratorB ib(p.pb, const_cast<double*>(B), {p.k, p.n}, tid, {0, n0});
    Mma mma(smem, tid, warp_idx, lane);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(p.k_iters, acc, ia, ib, acc);

    // Register-only epilogue (header comment: the accumulator layout).
    const int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    const int warp_m = warp_mn % WarpCount::kM, warp_n = warp_mn / WarpCount::kM;
    const int g = lane >> 2, t = lane & 3;
    const int64_t cb = p.c_bat ? p.c_bat[lam] : lam * p.c_bs;
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kMi; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < kE / 2; ++h) {
        const int i = m0 + warp_m * Warp::kM + mi * Inst::kM + h * 8 + g;
        if (i >= p.m) continue;
        const int64_t ci = cb + (p.c_row ? p.c_row[i] : i * p.c_si);
        CUTLASS_PRAGMA_UNROLL
        for (int nj = 0; nj < kNj; ++nj) {
          const int j = n0 + warp_n * Warp::kN + nj * 8 + 2 * t;  // even; the pair is (j, j + 1)
          const int a = (mi + nj * kMi) * kE + 2 * h;
          double* const dst = p.C + ci + (p.c_col ? p.c_col[j] : j * p.c_sj);
          if (kVecStore && p.vec2 && j + 1 < p.n &&
              (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
            // C affine with unit column stride: one 16-byte store, the same two values
            double2 o;
            if (p.beta == 0.0) {
              o.x = store(p, acc[a], 0.0), o.y = store(p, acc[a + 1], 0.0);
            } else {
              const double2 c = *reinterpret_cast<const double2*>(dst);
              o.x = store(p, acc[a], c.x), o.y = store(p, acc[a + 1], c.y);
            }
            *reinterpret_cast<double2*>(dst) = o;
            continue;
          }
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < 2; ++e) {
            if (j + e >= p.n) continue;
            double* const de = e == 0 ? dst : p.C + ci + (p.c_col ? p.c_col[j + 1] : (j + 1) * p.c_sj);
            *de = store(p, acc[a + e], p.beta == 0.0 ? 0.0 : *de);
          }
        }
      }
    }
  }

  static cudaError_t versions(int32_t* binary, int32_t* ptx);
  // v16 (DESIGN §6i.5): the occupancy row (CTAs per SM, registers, shared memory, threads) and
  // the code versions, for the performance model's kernel.hw_gemm_* lines.
  static cudaError_t occupancy(int32_t* out, int32_t* binary, int32_t* ptx);
  static imd_status_t launch(const imd::detail::HwOwnCall& c, cudaStream_t stream);
};

template <class K>
__global__ void __launch_bounds__(K::kThreads) hw_own_kernel(typename K::Params p) {
  extern __shared__ __align__(16) unsigned char hw_smem[];
  K::run(p, *reinterpret_cast<typename K::Mma::SharedStorage*>(hw_smem));
}

template <class Arch, class TB, class Warp, class Inst, int S, class LA, class LB>
cudaError_t HwOwn<Arch, TB, Warp, Inst, S, LA, LB>::versions(int32_t* binary, int32_t* ptx) {
  cudaFuncAttributes a{};
  const cudaError_t e = cudaFuncGetAttributes(&a, hw_own_kernel<HwOwn>);
  if (e == cudaSuccess) *binary = a.binaryVersion, *ptx = a.ptxVersion;
  return e;
}

template <class Arch, class TB, class Warp, class Inst, int S, class LA, class LB>
cudaError_t HwOwn<Arch, TB, Warp, Inst, S, LA, LB>::occupancy(int32_t* out, int32_t* binary,
                                                               int32_t* ptx) {
  using K = HwOwn;
  cudaFuncAttributes a{};
  cudaError_t e = cudaFuncGetAttributes(&a, hw_own_kernel<K>);
  if (e != cudaSuccess) return e;
  *binary = a.binaryVersion, *ptx = a.ptxVersion;
  if (kSmemBytes > 48 * 1024)
    e = cudaFuncSetAttribute(hw_own_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes);
  int r = 0;
  if (e == cudaSuccess)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, hw_own_kernel<K>, kThreads, kSmemBytes);
  if (e != cudaSuccess) return e;
  out[0] = r, out[1] = a.numRegs, out[2] = static_cast<int32_t>(a.sharedSizeBytes) + kSmemBytes;
  out[3] = kThreads;
  return cudaSuccess;
}

// One call: chunks of <= 65535 batches (grid.z) and <= 65535 column tiles (grid.y), the
// same chunking (hence launch count) as stage 1 (§6i.6).
template <class Arch, class TB, class Warp, class Inst, int S, class LA, class LB>
imd_status_t HwOwn<Arch, TB, Warp, Inst, S, LA, LB>::launch(const imd::detail::HwOwnCall& c,
                                                             cudaStream_t stream) {
  using K = HwOwn;
  static const cudaError_t attr = kSmemBytes > 48 * 1024
      ? cudaFuncSetAttribute(hw_own_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes)
      : cudaSuccess;
  if (attr != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
  Params p{};
  // A view ROW (K contiguous) = CUTLASS RowMajor m x k; COL = ColumnMajor. B view ROW
  // (j * ld + h) = ColumnMajor k x n; COL (j + h * ld) = RowMajor k x n.
  p.pa = typename Mma::IteratorA::Params(LA(c.A.ld));
  p.pb = typename Mma::IteratorB::Params(LB(c.B.ld));
  p.A = c.A.base, p.B = c.B.base;
  p.a_bs = c.A.bstride, p.b_bs = c.B.bstride, p.a_bat = c.A.bat, p.b_bat = c.B.bat;
  p.C = c.C, p.c_si = c.c_si, p.c_sj = c.c_sj, p.c_bs = c.c_bstride;
  p.c_row = c.c_row, p.c_col = c.c_col, p.c_bat = c.c_bat;
  p.m = static_cast<int>(c.m), p.n = static_cast<int>(c.n), p.k = static_cast<int>(c.k);
  p.k_iters = static_cast<int>((c.k + TB::kK - 1) / TB::kK);
  p.alpha = c.alpha, p.beta = c.beta;
  p.vec2 = c.c_row == nullptr && c.c_col == nullptr && c.c_sj == 1 && imd::detail::hw_own_vec2();
  constexpr int64_t kMaxGrid = 65535;
  const int64_t tiles_m = (c.m + TB::kM - 1) / TB::kM;
  if (tiles_m > INT32_MAX) return IMD_STATUS_INVALID_VALUE;
  constexpr int64_t col_chunk = kMaxGrid * 64;  // as stage 1: a multiple of every tile's N
  static_assert(TB::kN % 64 == 0 || 64 % TB::kN == 0, "tile N must divide the column chunk");
  for (int64_t z0 = 0; z0 < c.b; z0 += kMaxGrid) {
    const int64_t nb = c.b - z0 < kMaxGrid ? c.b - z0 : kMaxGrid;
    for (int64_t j0 = 0; j0 < c.n; j0 += col_chunk) {
      const int64_t nn = c.n - j0 < col_chunk ? c.n - j0 : col_chunk;
      p.z0 = z0, p.j0 = static_cast<int>(j0);
      const dim3 grid(static_cast<unsigned>(tiles_m),
                      static_cast<unsigned>((nn + TB::kN - 1) / TB::kN), static_cast<unsigned>(nb));
      hw_own_kernel<K><<<grid, kThreads, kSmemBytes, stream>>>(p);
      if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
    }
  }
  return IMD_STATUS_SUCCESS;
}

}  // namespace imd::kernels
