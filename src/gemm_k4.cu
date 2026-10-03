// K4 (DESIGN §3.5, §4): per residue group and K-chunk, a CUTLASS 2.x batched int8 TN GEMM
// with the mod-p -> u8 epilogue (store-mod for chunk 0, add-mod after), batch z = s*b +
// lambda, modulus l = g0 + z / b from __constant__ (imd::kernels::VisitorModP).
//
// One tile configuration for now: 128x256x64, 3 stages (72 KB smem), verified in M0
// (tests/test_cutlass_mod_epilogue.cu). TODO(M4): the DESIGN §4 tile menu chosen by wave
// quantization; 128-bit epilogue stores.
#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "immerdink/detail/constants.hpp"
#include "kernels/mod_epilogue.cuh"  // also brings in the Sm80 EVT visitors
#include "plan.hpp"

namespace imd::detail {
namespace {

using namespace cute;
namespace c = imd::constants;

using TB = cutlass::gemm::GemmShape<128, 256, 64>;
using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;
constexpr int kEVTStages = 1;
constexpr int kAlign = 16;  // int8 / u8 elements per 128-bit access
constexpr auto kRound = cutlass::FloatRoundStyle::round_toward_zero;

using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, uint8_t, kAlign,
                                                                         kEVTStages>;
using StrideW = cute::Stride<int64_t, _1, int64_t>;  // (ld = n_hat, 1, plane = m * n_hat)
using LoadW = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, uint8_t, StrideW>;
using StoreW = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, uint8_t, kRound, StrideW>;
using ModP = imd::kernels::VisitorModP;
using EVTStore = cutlass::epilogue::threadblock::Sm80EVT<StoreW, ModP>;
using EVTAdd =
    cutlass::epilogue::threadblock::Sm80EVT<StoreW, cutlass::epilogue::threadblock::Sm80EVT<ModP, LoadW>>;

template <class EVT>
using GemmFor = cutlass::gemm::device::GemmUniversalAdapter<
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlign,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlign,
        uint8_t, cutlass::layout::RowMajor, kAlign, int32_t, int32_t,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, EVT,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
        cutlass::arch::OpMultiplyAddSaturate, kEVTStages>::GemmKernel>;
using GemmStore = GemmFor<EVTStore>;
using GemmAdd = GemmFor<EVTAdd>;

constexpr int64_t kMaxBatch = 65535;                 // gridDim.z
constexpr int64_t kMaxNPerLaunch = 65535 * TB::kN;   // gridDim.y (identity swizzle: N tiles)

template <class Gemm>
bool run(typename Gemm::Arguments const& args, cudaStream_t stream) {
  Gemm op;
  cutlass::Status st = Gemm::can_implement(args);
  if (st == cutlass::Status::kSuccess) st = op.initialize(args, nullptr, stream);
  if (st == cutlass::Status::kSuccess) st = op.run(stream);
  return st == cutlass::Status::kSuccess && cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace

imd_status_t k4_group(const int8_t* RA, const int8_t* RB, uint8_t* W, int64_t m, int64_t n_hat,
                      int64_t k_hat, int64_t b, int g0, int gsize, cudaStream_t stream) {
  const int64_t batches = int64_t{gsize} * b;
  const int64_t plane_a = m * k_hat, plane_b = n_hat * k_hat, plane_w = m * n_hat;
  for (int64_t k0 = 0; k0 < k_hat; k0 += c::kKcPrime) {  // K-chunks (§3.5), 16-aligned
    const int64_t kc = std::min<int64_t>(c::kKcPrime, k_hat - k0);
    for (int64_t z0 = 0; z0 < batches; z0 += kMaxBatch) {  // batch-count split (§4)
      const int64_t zc = std::min(kMaxBatch, batches - z0);
      for (int64_t j0 = 0; j0 < n_hat; j0 += kMaxNPerLaunch) {
        const int64_t nc = std::min(kMaxNPerLaunch, n_hat - j0);
        const cutlass::gemm::GemmCoord size{static_cast<int>(m), static_cast<int>(nc),
                                            static_cast<int>(kc)};
        const int8_t* A = RA + z0 * plane_a + k0;
        const int8_t* B = RB + z0 * plane_b + j0 * k_hat + k0;
        uint8_t* Wp = W + (int64_t{g0} * b + z0) * plane_w + j0;
        const ModP::Arguments mod{g0, b, z0};
        const typename StoreW::Arguments store{Wp, {n_hat, _1{}, plane_w}};
        bool ok;
        if (k0 == 0) {
          typename EVTStore::Arguments cb{mod, store};
          ok = run<GemmStore>({cutlass::gemm::GemmUniversalMode::kBatched, size,
                               static_cast<int>(zc), cb, A, B, nullptr, nullptr, plane_a,
                               plane_b, 0, 0, k_hat, k_hat, 0, 0},
                              stream);
        } else {
          typename EVTAdd::Arguments cb{{{Wp, uint8_t(0), {n_hat, _1{}, plane_w}}, mod}, store};
          ok = run<GemmAdd>({cutlass::gemm::GemmUniversalMode::kBatched, size,
                             static_cast<int>(zc), cb, A, B, nullptr, nullptr, plane_a, plane_b,
                             0, 0, k_hat, k_hat, 0, 0},
                            stream);
        }
        if (!ok) return IMD_STATUS_CUDA_ERROR;
      }
    }
  }
  return IMD_STATUS_SUCCESS;
}

}  // namespace imd::detail
