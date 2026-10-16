// The Hopper K4 prototype's building blocks (issue 17, docs/NOTES-17.md): K4's epilogue as a CUTLASS Hopper collective
// epilogue (K4ModEpilogue) and the kernel type of one variant (Variant). Used by k4_hopper_proto.cu (compile-only) and
// rung0.cu (the known-answer program of R1). See k4_hopper_proto.cu for the design notes.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cute/tensor.hpp"

#include "device/constants.cuh"
#include "k4/hopper/epilogue_wg.cuh"

namespace imd::hopper {

using namespace cute;

// Does the TiledMma's accumulator layout equal the wgmma order, for warpgroups of 64 rows stacked down M (cooperative
// kernels stack two) and N columns per instruction: (thread t, value v) -> row 16 (t / 32) + (t % 32) / 4 +
// 8 ((v / 2) % 2), column 8 (v / 4) + 2 (t % 4) + v % 2 of the tile the TiledMma covers (linear index m + M n)?
// Compared structurally, at compile time, against the layout CuTe builds for the TiledMma (the wgmma atom's
// CLayout, tests/test_wgmma_layout.py, repeated over the warpgroups); tests/test_wgmma_layout.py checks the formula
// itself against CuTe for every atom.
template <class Mma, int N>
constexpr bool layout_ok() {
  constexpr int Mt = int(cute::tile_size<0>(Mma{}));
  constexpr int W = int(size(Mma{})) / 32;   // warps
  auto expected = Layout<Shape<Shape<_4, _8, Int<W>>, Shape<_2, _2, Int<N / 8>>>,
                         Stride<Stride<Int<2 * Mt>, _1, _16>, Stride<Int<Mt>, _8, Int<8 * Mt>>>>{};
  auto tv = Mma{}.get_layoutC_TV();
  return coalesce(tv) == coalesce(expected);
}

// The epilogue arguments, one type for the store-mod and the add-mod kernels.
struct K4Args {
  uint8_t* W = nullptr;     // modulus plane of z = 0
  int64_t ldw = 0;          // = n_hat
  int64_t plane_w = 0;      // elements between the planes of consecutive z
  int slot0 = 0;            // modulus index of slot 0 (the launch's g0)
  int64_t b = 1;            // batches per modulus slot
  int64_t batch0 = 0;       // the launch's first batch (gridDim.z parts): the modulus of z is slot0 + (batch0 + z) / b
};

// The epilogue: DefaultEpilogue's shape (cutlass/epilogue/collective/default_epilogue.hpp) with K4's arithmetic.
template <bool kAdd, bool kFast>
class K4ModEpilogue {
 public:
  using EpilogueSchedule = cutlass::gemm::EpilogueDefault;
  using DispatchPolicy = EpilogueSchedule;
  using ElementC = uint8_t;
  using ElementD = uint8_t;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;  // (M, N, L): row stride ldw, plane stride
  using StrideD = StrideC;
  using GmemElementC = ElementC;
  struct SharedStorage {};
  using TensorStorage = SharedStorage;
  struct ThreadEpilogueOp {};  // named by the device-level adapter only; the arithmetic is in operator()

  using Arguments = K4Args;
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& a, void*) { return a; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  // The fast path stores 128 bits at a time (HopperEpilogue): the plane needs 16-byte alignment, row and plane strides multiples of
  // 16, and n a multiple of 16 (n_hat in K4). The generic path needs none of it.
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& shape, Arguments const& a) {
    if constexpr (!kFast) return a.W != nullptr;
    const int64_t n = get<1>(shape);
    return a.W != nullptr && reinterpret_cast<uintptr_t>(a.W) % 16 == 0 && a.ldw % 16 == 0 && a.plane_w % 16 == 0 &&
           n % 16 == 0 && a.ldw >= n && a.b >= 1 && a.batch0 >= 0;
  }

  CUTLASS_HOST_DEVICE K4ModEpilogue(Params const& p, SharedStorage const& = SharedStorage()) : params(p) {}
  CUTLASS_DEVICE bool is_source_needed() { return kAdd; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout,
            class TiledMma, class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK blk_shape_MNK,
                                 BlockCoordMNKL blk_coord_mnkl, cute::Tensor<FrgEngine, FrgLayout> const& acc,
                                 TiledMma tiled_mma, ResidueMNK, int thread_idx, char*) {
    const int M = get<0>(problem_shape_mnkl), N = get<1>(problem_shape_mnkl);
    const auto [m_coord, n_coord, k_coord, z] = blk_coord_mnkl;
    const int row0 = m_coord * int(size<0>(blk_shape_MNK)), col0 = n_coord * int(size<1>(blk_shape_MNK));
    const int l = params.slot0 + int((params.batch0 + z) / params.b);     // the modulus of this batch (k4_gemm.cuh tile_of)
    uint8_t* const plane = params.W + int64_t(z) * params.plane_w;
    const uint32_t mod = imd::device::c_tables.moduli[l];
    if constexpr (kFast) {
      // The accumulators are the wgmma order of one 64 x N instruction per warpgroup, repeated kRep times down M
      // (register index v + (N / 2) rep: CuTe's (MMA, MMA_M, MMA_N) order); kWG warpgroups stack along M inside the
      // TiledMma (cooperative: 2), so the CTA tile has 64 kWG kRep rows.
      constexpr int kN = int(size<1>(BlockShapeMNK{}));
      constexpr int kRep = int(size(FrgLayout{})) / (kN / 2);
      constexpr int kWG = int(int(cute::tile_size<0>(TiledMma{}))) / 64;
      static_assert(kN % 64 == 0 && kN <= 256, "fast path: 64-column groups");
      static_assert(layout_ok<TiledMma, kN>(), "fast path: the TiledMma accumulator layout is not the wgmma 64 x N order");
      static_assert(64 * kWG * kRep == int(size<0>(BlockShapeMNK{})), "tile rows = 64 x warpgroups x repetitions");
      const int wg = thread_idx / 128, warp = (thread_idx % 128) / 32, lane = thread_idx % 32;
      CUTLASS_PRAGMA_UNROLL
      for (int rep = 0; rep < kRep; ++rep) {
        int32_t r[kN / 2];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kN / 2; ++i) r[i] = acc(rep * (kN / 2) + i);
        imd::kernels::HopperEpilogue::run<kN, kAdd>(r, warp, lane, row0 + 64 * (wg + kWG * rep), col0, plane,
                                                    params.ldw, M, N, mod, imd::device::c_tables.epi_bias[l],
                                                    imd::device::c_tables.epi_magic[l]);
      }
    } else {
      // generic: one byte per accumulator at its CuTe coordinate
      auto thr = tiled_mma.get_thread_slice(thread_idx);
      Tensor cD = make_identity_tensor(make_shape(size<0>(blk_shape_MNK), size<1>(blk_shape_MNK)));
      Tensor tC = thr.partition_C(cD);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < int(size(acc)); ++i) {
        const int r = row0 + int(get<0>(tC(i))), c = col0 + int(get<1>(tC(i)));
        if (r < M && c < N) {
          uint8_t* dst = plane + int64_t(r) * params.ldw + c;
          int32_t v = acc(i);
          if constexpr (kAdd) v += int32_t(*dst);
          *dst = mod == 256u ? uint8_t(uint32_t(v) & 255u)
                             : uint8_t(imd::device::mod_p_barrett_neg(v, 0u - mod, imd::device::c_tables.epi_bias[l],
                                                                      imd::device::c_tables.epi_magic[l]));
        }
      }
    }
  }

 private:
  Params params;
};

// One Hopper K4 kernel: tile, cluster, schedule, epilogue body.
template <class Tile, class Cluster, class Sched, bool kAdd, bool kFast>
struct Variant {
  using Epi = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<K4ModEpilogue<kAdd, kFast>>;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, cutlass::layout::RowMajor, 16, int8_t,
      cutlass::layout::ColumnMajor, 16, int32_t, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Sched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Main, Epi>;
};

}  // namespace imd::hopper

