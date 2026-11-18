// IMD_HWGEMM stage 2 on sm_90 (DESIGN §6i.7): the own kernel (hw_own.cuh) with the stage-1
// Sm90 configurations (DMMA m16n8k4, 4 stages): 128x128 (warp 32x64), 64x128 (warp 32x64),
// 64x64 (warp 32x32). The caller gates launches on hw_sm90_usable().
#include "hw_gemm.cuh"
#include "hw_own.cuh"

namespace imd::detail {
namespace {
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
template <int TM, int TN, int WM, int WN, class LA, class LB>
using O = imd::kernels::HwOwn<cutlass::arch::Sm90, cutlass::gemm::GemmShape<TM, TN, 16>,
                              cutlass::gemm::GemmShape<WM, WN, 16>,
                              cutlass::gemm::GemmShape<16, 8, 4>, 4, LA, LB>;
template <int TM, int TN, int WM, int WN>
imd_status_t run_tile(const HwOwnCall& c, cudaStream_t s) {
  if (c.A.k_contig)
    return c.B.k_contig ? O<TM, TN, WM, WN, RM, CM>::launch(c, s) : O<TM, TN, WM, WN, RM, RM>::launch(c, s);
  return c.B.k_contig ? O<TM, TN, WM, WN, CM, CM>::launch(c, s) : O<TM, TN, WM, WN, CM, RM>::launch(c, s);
}
}  // namespace

cudaError_t hw_own_sm90_versions(int32_t* binary, int32_t* ptx) {
  return O<64, 64, 32, 32, RM, RM>::versions(binary, ptx);
}

imd_status_t hw_own_sm90_run(const HwOwnCall& c, cudaStream_t stream) {
  switch (c.tile) {
    case kHw128x128: return run_tile<128, 128, 32, 64>(c, stream);
    case kHw64x128: return run_tile<64, 128, 32, 64>(c, stream);
    case kHw64x64: return run_tile<64, 64, 32, 32>(c, stream);
    default: return IMD_STATUS_INVALID_VALUE;
  }
}

}  // namespace imd::detail
