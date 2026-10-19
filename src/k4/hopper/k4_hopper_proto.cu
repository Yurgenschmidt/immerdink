// COMPILE-ONLY prototype of the Hopper K4 (issue 17, docs/NOTES-17.md; the design check "does CUTLASS 4.8's
// Hopper main loop accept our epilogue?"). Not part of libimmerdink and never run: it is built for sm_90a only
// by tools/check_hopper_variants.sh, which compiles every variant, rejects register spills, and looks for the
// wgmma (IGMMA) and TMA (UTMALDG) instructions in the machine code.
//
// What is stock: the main loop, TMA loads, mbarrier pipeline and wgmma (cutlass::gemm::collective::CollectiveBuilder,
// KernelTmaWarpSpecialized{Cooperative,Pingpong}), the persistent tile scheduler.
// What is ours: the epilogue. K4's mod-p -> u8 epilogue (K4ModEpilogue below) is plugged in through CUTLASS's own
// extension point for register epilogues (cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter around a
// DefaultEpilogue-shaped class): it receives the accumulators in registers, the tile coordinate (the batch z picks the
// modulus) and writes the u8 residues of W straight from registers. It uses no shared memory, so the stages left for
// the main loop are the same as with no epilogue and, in a persistent kernel, the producer warp keeps loading the next
// tile's stages while the consumers run the epilogue (the "epilogue overlap" of K4's A2, for free).
//
// Two epilogue bodies:
//  * generic: every accumulator register is stored as one byte at the coordinate CuTe's partition says it has. Correct
//    for any TiledMma. Slow (one byte store per element). The reference for the layout check below.
//  * fast (kFast): HopperEpilogue::run (epilogue_wg.cuh): the quad-transposed 128-bit stores of the mma.sync kernel,
//    valid iff the TiledMma's accumulator layout is the wgmma 64 x N order. A constexpr check against CuTe's own
//    layout (layout_ok) is a static_assert, so a TiledMma where the assumption fails does not compile.
#include "k4/hopper/k4_hopper.cuh"

using namespace imd::hopper;
using cute::Int;
using cute::Shape;
namespace cg = cutlass::gemm;
using Coop = cg::KernelTmaWarpSpecializedCooperative;
using Ping = cg::KernelTmaWarpSpecializedPingpong;
using _1 = Int<1>;
using _2 = Int<2>;

// Force the instantiation of each kernel (one __global__ symbol per variant, nothing calls it). K4H_SELECT = -1 builds all
// of them, K4H_SELECT = i only the i-th line of k4_hopper_variants.def (tools/check_hopper_variants.sh: one ptxas report per variant).
#ifndef K4H_SELECT
#define K4H_SELECT -1
#endif
// the index of each variant = its line in the .def file
enum {
#define K4H_VARIANT(NAME, ...) K4H_INDEX_OF_##NAME,
#include "k4/hopper/k4_hopper_variants.def"
#undef K4H_VARIANT
};

#define K4H_VARIANT(NAME, TILE_M, TILE_N, TILE_K, CLM, CLN, SCHED, ADD, FAST)                                         \
  template <int kSel>                                                                                                \
  const void* k4h_##NAME##_t() {                                                                                     \
    if constexpr (kSel < 0 || kSel == K4H_INDEX_OF_##NAME) {                                                         \
      using V = Variant<Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>, Shape<Int<CLM>, Int<CLN>, _1>, SCHED, ADD, FAST>; \
      return reinterpret_cast<const void*>(&cutlass::device_kernel<typename V::Kernel>);                             \
    } else {                                                                                                         \
      return nullptr;                                                                                                \
    }                                                                                                                \
  }                                                                                                                  \
  extern "C" const void* k4h_##NAME() { return k4h_##NAME##_t<K4H_SELECT>(); }


#include "k4/hopper/k4_hopper_variants.def"
