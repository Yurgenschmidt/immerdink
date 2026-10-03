// K4 epilogue (DESIGN §3.5, §4): int32 accumulator -> non-negative residue mod p_l -> u8.
//
// A CUTLASS 2.x Sm80 epilogue-visitor node. One batched launch covers several moduli:
// batch z = s*b + lambda (s = slot in the residue group, b = batch count of the GEMM view),
// and the node reads p_l and its reduction constants from the __constant__ table at
// l = slot0 + z / b. A VisitorScalarBroadcast cannot express z / b (DESIGN §4). When the
// batch count is split over several launches, batch0 is the z of the launch's first batch.
//
//   store-mod (chunk c = 0):  Sm80EVT<AuxStore<u8>, VisitorModP>                 W = acc mod p
//   add-mod   (chunk c > 0):  Sm80EVT<AuxStore<u8>, Sm80EVT<VisitorModP, AuxLoad<u8>>>
//                                                                              W = (W + acc) mod p
// The node's output is already in [0, p) subset [0, 255], so the u8 conversion done by the
// store (a saturating cvt) is exact. Requires the identity threadblock swizzle in kBatched
// mode, where the tile offset's k() is the batch index z.
#pragma once

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
// visitors.hpp is not self-contained; the visitor kernel header provides its prerequisites.
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "device/constants.cuh"

namespace imd::kernels {

struct VisitorModP {
  struct SharedStorage {};

  struct Arguments {
    int32_t slot0 = 0;             // modulus index of slot 0 (g0)
    int64_t batches_per_slot = 1;  // b
    int64_t batch0 = 0;            // z of this launch's batch 0 (batch-count split, §4)
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args,
                                                  void*) {
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const& args) {
    return args.slot0 >= 0 && args.batches_per_slot >= 1;
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*,
                                              cudaStream_t, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE VisitorModP() {}
  CUTLASS_HOST_DEVICE VisitorModP(Params const& params, SharedStorage const&)
      : params_ptr(&params) {}

  Params const* params_ptr = nullptr;

  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    // Modulus p_l of this CTA's batch and its §3.5 constants, read from the __constant__
    // table once per CTA. (A2: reading them inside the element loop re-issued two indexed
    // LDCs and the p = 256 branch per element, ~12 instructions per element instead of 4.)
    uint32_t p, bias;
    uint64_t magic;

    CUTLASS_DEVICE explicit Callbacks(int l)
        : p(imd::device::c_tables.moduli[l]),
          bias(imd::device::c_tables.epi_bias[l]),
          magic(imd::device::c_tables.epi_magic[l]) {}

    // frg_prev: zero inputs (store-mod) or the stored u8 residues (add-mod). Both branches
    // compute imd::device::mod_p_epilogue; p is CTA-uniform, so the p = 256 test is one
    // uniform branch per fragment.
    template <class ElementAccumulator, int FragmentSize, class... Prev>
    CUTLASS_DEVICE cutlass::Array<uint8_t, FragmentSize> visit(
        int, int, int, int, cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
        Prev const&... frg_prev) {
      static_assert(sizeof...(Prev) <= 1, "VisitorModP takes at most one child (the old W)");
      cutlass::Array<uint8_t, FragmentSize> out;
      if (p == 256u) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          int32_t v = static_cast<int32_t>(frg_acc[i]);
          ((v += static_cast<int32_t>(frg_prev[i])), ...);  // W + acc stays in int32 (cert_kc)
          out[i] = static_cast<uint8_t>(static_cast<uint32_t>(v) & 255u);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          int32_t v = static_cast<int32_t>(frg_acc[i]);
          ((v += static_cast<int32_t>(frg_prev[i])), ...);
          out[i] = static_cast<uint8_t>(imd::device::mod_p_barrett(v, p, bias, magic));
        }
      }
      return out;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset, int,
                                    ProblemShape) {
    const int64_t z = params_ptr->batch0 + threadblock_tile_offset.k();
    return Callbacks(params_ptr->slot0 + static_cast<int>(z / params_ptr->batches_per_slot));
  }
};

}  // namespace imd::kernels
