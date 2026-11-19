// IMD_HWGEMM on sm_90 (DESIGN §6i.7): CUTLASS's Sm90 FP64 tensor-op GEMM (DMMA m16n8k4, at
// about 99 % of the FP64 tensor-core peak on H100 / H200: NOTES-HWGEMM §4.3), three tiles.
// Built for every architecture of the library; the instructions exist from sm_90 on, so
// hw_sm90_usable() gates the launches on the kernels' PTX version.
#include "hw_gemm.cuh"

namespace imd::detail {
namespace {
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using I1684 = cutlass::gemm::GemmShape<16, 8, 4>;
template <int TM, int TN, int WM, int WN, class LA, class LB>
using K = imd::kernels::HwKernel<cutlass::arch::Sm90, cutlass::gemm::GemmShape<TM, TN, 16>,
                                 cutlass::gemm::GemmShape<WM, WN, 16>, I1684, 4, LA, LB>;
// The gate's best configurations (NOTES-HWGEMM §4.3): 128x128 (warp 32x64), 64x128 (warp
// 32x64), 64x64 (warp 32x32), all 4 stages.
template <int TM, int TN, int WM, int WN>
imd_status_t run_tile(const HwNorm& c, cudaStream_t s) {
  return hw_dispatch_layouts<K<TM, TN, WM, WN, RM, RM>, K<TM, TN, WM, WN, RM, CM>,
                             K<TM, TN, WM, WN, CM, RM>, K<TM, TN, WM, WN, CM, CM>>(c, s);
}
}  // namespace

bool hw_sm90_usable() {
  static const int v = K<64, 64, 32, 32, RM, RM>::ptx_version();
  return v >= 90;
}

cudaError_t hw_sm90_versions(int32_t* binary, int32_t* ptx) {
  return K<64, 64, 32, 32, RM, RM>::versions(binary, ptx);
}

imd_status_t hw_sm90_run(const HwNorm& c, cudaStream_t stream) {
  switch (c.tile) {
    case kHw128x128: return run_tile<128, 128, 32, 64>(c, stream);
    case kHw64x128: return run_tile<64, 128, 32, 64>(c, stream);
    case kHw64x64: return run_tile<64, 64, 32, 32>(c, stream);
    default: return IMD_STATUS_INVALID_VALUE;
  }
}

}  // namespace imd::detail
