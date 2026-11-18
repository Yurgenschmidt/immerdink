// IMD_HWGEMM stage 2 on sm_80 / sm_89 (DESIGN §6i.7): the own kernel (hw_own.cuh) with the
// stage-1 Sm80 configuration (64x64x16, warp 32x32, DMMA m8n8k4, 4 stages).
#include "hw_gemm.cuh"
#include "hw_own.cuh"

namespace imd::detail {
namespace {
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
template <class LA, class LB>
using O64 = imd::kernels::HwOwn<cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 16>,
                                cutlass::gemm::GemmShape<32, 32, 16>,
                                cutlass::gemm::GemmShape<8, 8, 4>, 4, LA, LB>;
}  // namespace

cudaError_t hw_own_sm80_versions(int32_t* binary, int32_t* ptx) {
  return O64<RM, RM>::versions(binary, ptx);
}

imd_status_t hw_own_sm80_run(const HwOwnCall& c, cudaStream_t stream) {
  if (c.tile != kHw64x64) return IMD_STATUS_INVALID_VALUE;
  // A ROW = RowMajor; B ROW (K contiguous per column j) = ColumnMajor (hw_gemm.cuh).
  if (c.A.k_contig)
    return c.B.k_contig ? O64<RM, CM>::launch(c, stream) : O64<RM, RM>::launch(c, stream);
  return c.B.k_contig ? O64<CM, CM>::launch(c, stream) : O64<CM, RM>::launch(c, stream);
}

}  // namespace imd::detail
