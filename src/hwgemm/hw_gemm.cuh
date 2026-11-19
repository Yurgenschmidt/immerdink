// IMD_HWGEMM CUTLASS kernels (docs/DESIGN.md §6i.7): CUTLASS 2.x GemmUniversal FP64 tensor-op
// GEMMs with the library's own epilogue (§3.7 store_out), one instantiation per (arch, tile,
// A layout, B layout). C is always row-major here: hwgemm.cu transposes a column-major C
// (C^T = B^T A^T swaps the two operands; the K order, hence every sum, is unchanged).
#pragma once

#include <cmath>
#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "../hwgemm.hpp"

namespace imd::kernels {

// §3.7 output scaling as an epilogue output operator: E = alpha == 1 ? D : RN(alpha D);
// C = beta == 0 ? E : fma(beta, C, E) — the same operations as contract.cu store_out, so a
// tile's result is bit-identical to DIRECT's (CUTLASS's LinearCombination computes
// fma(alpha, D, RN(beta C)) instead). beta == 0: the source is not read (is_source_needed).
template <int Count>
class HwStoreOut {
 public:
  using ElementOutput = double;
  using ElementSource = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
  using ElementScalar = double;
  using ElementC = double;
  using ElementD = double;
  static int const kCount = Count;
  static cutlass::epilogue::thread::ScaleType::Kind const kScale =
      cutlass::epilogue::thread::ScaleType::Default;
  using FragmentOutput = cutlass::Array<double, kCount>;
  using FragmentSource = cutlass::Array<double, kCount>;
  using FragmentAccumulator = cutlass::Array<double, kCount>;
  using FragmentCompute = cutlass::Array<double, kCount>;
  static cutlass::FloatRoundStyle const kRound = cutlass::FloatRoundStyle::round_to_nearest;

  struct Params {
    double alpha = 1.0;
    double beta = 0.0;
    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE Params(double a, double b) : alpha(a), beta(b) {}
  };

  CUTLASS_HOST_DEVICE explicit HwStoreOut(Params const& p, int = 0)
      : alpha_(p.alpha), beta_(p.beta) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return beta_ != 0.0; }

  // Serial split-K only (never used: HWGEMM has no split-K, §6i.4).
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc,
                                                FragmentSource const& src) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) out[i] = store(acc[i], src[i]);
    return out;
  }

  CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) out[i] = scale(acc[i]);
    return out;
  }

 private:
  double alpha_, beta_;

  CUTLASS_HOST_DEVICE double scale(double d) const {
#if defined(__CUDA_ARCH__)
    return alpha_ == 1.0 ? d : __dmul_rn(alpha_, d);
#else
    return alpha_ == 1.0 ? d : alpha_ * d;
#endif
  }
  CUTLASS_HOST_DEVICE double store(double d, double c) const {
    const double e = scale(d);
#if defined(__CUDA_ARCH__)
    return beta_ == 0.0 ? e : __fma_rn(beta_, c, e);
#else
    return beta_ == 0.0 ? e : std::fma(beta_, c, e);
#endif
  }
};

template <class Arch, class TB, class Warp, class Inst, int Stages, class LA, class LB>
struct HwKernel {
  using Gemm = cutlass::gemm::device::GemmUniversal<
      double, LA, double, LB, double, cutlass::layout::RowMajor, double,
      cutlass::arch::OpClassTensorOp, Arch, TB, Warp, Inst, HwStoreOut<1>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, Stages, 1, 1,
      cutlass::arch::OpMultiplyAdd>;
  static constexpr int kTileN = TB::kN;

  // The PTX version the kernel was compiled from (cudaFuncAttributes::ptxVersion), so that the
  // dispatcher never launches an Sm90 kernel JIT-compiled from older PTX (it would trap).
  static int ptx_version() {
    cudaFuncAttributes a{};
    if (cudaFuncGetAttributes(&a, cutlass::Kernel2<typename Gemm::GemmKernel>) != cudaSuccess) {
      (void)cudaGetLastError();
      return 0;
    }
    return a.ptxVersion;
  }

  static cudaError_t versions(int32_t* binary, int32_t* ptx) {
    cudaFuncAttributes a{};
    const cudaError_t e = cudaFuncGetAttributes(&a, cutlass::Kernel2<typename Gemm::GemmKernel>);
    if (e == cudaSuccess) *binary = a.binaryVersion, *ptx = a.ptxVersion;
    return e;
  }

  // One launch: m x n x k, `batches` batches (<= 65535), C row-major. The caller has chunked.
  static imd_status_t run(int64_t m, int64_t n, int64_t k, int64_t batches, const double* A,
                          int64_t lda, int64_t bsa, const double* B, int64_t ldb, int64_t bsb,
                          double* C, int64_t ldc, int64_t bsc, double alpha, double beta,
                          cudaStream_t stream) {
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kBatched,
        cutlass::gemm::GemmCoord(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)),
        static_cast<int>(batches), typename HwStoreOut<1>::Params(alpha, beta), A, B, C, C, bsa,
        bsb, bsc, bsc, lda, ldb, ldc, ldc);
    Gemm op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return IMD_STATUS_NOT_SUPPORTED;
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
      return IMD_STATUS_CUDA_ERROR;
    if (op.run(stream) != cutlass::Status::kSuccess) return IMD_STATUS_CUDA_ERROR;
    return IMD_STATUS_SUCCESS;
  }
};

}  // namespace imd::kernels

namespace imd::detail {

// A normalized call (C row-major) for one arch's kernels: hw_sm80.cu / hw_sm90.cu.
struct HwNorm {
  int64_t m, n, k, b;
  HwView2 A, B;  // A: m x k, B: n x k rows x K views
  double* C;
  int64_t ldc, c_bstride;
  double alpha, beta;
  int32_t tile;
};

imd_status_t hw_sm80_run(const HwNorm& c, cudaStream_t stream);
imd_status_t hw_own_sm80_run(const HwOwnCall& c, cudaStream_t stream);  // hw_own_sm80.cu
imd_status_t hw_own_sm90_run(const HwOwnCall& c, cudaStream_t stream);  // hw_own_sm90.cu
cudaError_t hw_own_sm80_versions(int32_t* binary, int32_t* ptx);
cudaError_t hw_own_sm90_versions(int32_t* binary, int32_t* ptx);
cudaError_t hw_own_sm80_occupancy(int32_t* out, int32_t* binary, int32_t* ptx);  // 64x64
cudaError_t hw_own_sm90_occupancy(int32_t tile, int32_t* out, int32_t* binary, int32_t* ptx);
imd_status_t hw_sm90_run(const HwNorm& c, cudaStream_t stream);
bool hw_sm90_usable();  // the Sm90 kernels carry sm_90+ code (ptxVersion >= 90)
cudaError_t hw_sm80_versions(int32_t* binary, int32_t* ptx);  // one kernel of each module
cudaError_t hw_sm90_versions(int32_t* binary, int32_t* ptx);

// Chunk one normalized call over batches (grid.z <= 65535) and column tiles (grid.y <=
// 65535) and run kernel K of each (A, B) layout pair.
template <class KRR, class KRC, class KCR, class KCC>
imd_status_t hw_dispatch_layouts(const HwNorm& c, cudaStream_t stream) {
  // A view ROW (K contiguous) = CUTLASS RowMajor m x k; COL = ColumnMajor. B view ROW
  // (j * ld + h) = ColumnMajor k x n; COL (j + h * ld) = RowMajor k x n. KXY: X = A's
  // CUTLASS layout, Y = B's (R = RowMajor, C = ColumnMajor).
  constexpr int64_t kMaxGrid = 65535;
  // Column chunks of 65535 * 64 (a multiple of every tile's N, and <= 65535 tiles of each):
  // the same count for every tile, so the planner reports the launches without the tile.
  static_assert(KRR::kTileN % 64 == 0, "tile N must divide the column chunk");
  constexpr int64_t col_chunk = kMaxGrid * 64;
  for (int64_t z0 = 0; z0 < c.b; z0 += kMaxGrid) {
    const int64_t nb = c.b - z0 < kMaxGrid ? c.b - z0 : kMaxGrid;
    for (int64_t j0 = 0; j0 < c.n; j0 += col_chunk) {
      const int64_t nn = c.n - j0 < col_chunk ? c.n - j0 : col_chunk;
      const double* A = c.A.base + z0 * c.A.bstride;
      const double* B = c.B.base + z0 * c.B.bstride + j0 * (c.B.k_contig ? c.B.ld : 1);
      double* C = c.C + z0 * c.c_bstride + j0;
      imd_status_t st;
      const auto go = [&](auto kernel) {
        return decltype(kernel)::run(c.m, nn, c.k, nb, A, c.A.ld, c.A.bstride, B, c.B.ld,
                                     c.B.bstride, C, c.ldc, c.c_bstride, c.alpha, c.beta, stream);
      };
      if (c.A.k_contig)
        st = c.B.k_contig ? go(KRC{}) : go(KRR{});
      else
        st = c.B.k_contig ? go(KCC{}) : go(KCR{});
      if (st != IMD_STATUS_SUCCESS) return st;
    }
  }
  return IMD_STATUS_SUCCESS;
}

}  // namespace imd::detail
