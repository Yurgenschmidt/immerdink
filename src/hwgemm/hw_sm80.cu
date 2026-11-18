// IMD_HWGEMM on sm_80 / sm_89 (DESIGN §6i.7): CUTLASS's Sm80 FP64 tensor-op GEMM (DMMA
// m8n8k4), one tile (64x64x16, warp 32x32, 4 stages) for correctness; not tuned.
#include "hw_gemm.cuh"

namespace imd::detail {
namespace {
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using I884 = cutlass::gemm::GemmShape<8, 8, 4>;
template <class LA, class LB>
using K64 = imd::kernels::HwKernel<cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 64, 16>,
                                   cutlass::gemm::GemmShape<32, 32, 16>, I884, 4, LA, LB>;
}  // namespace

cudaError_t hw_sm80_versions(int32_t* binary, int32_t* ptx) {
  return K64<RM, RM>::versions(binary, ptx);
}

imd_status_t hw_sm80_run(const HwNorm& c, cudaStream_t stream) {
  return hw_dispatch_layouts<K64<RM, RM>, K64<RM, CM>, K64<CM, RM>, K64<CM, CM>>(c, stream);
}

}  // namespace imd::detail
