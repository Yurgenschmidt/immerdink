// K4 (DESIGN §3.5, §4): per residue group and K-chunk, a CUTLASS 2.x batched int8 TN GEMM
// with the mod-p -> u8 epilogue (store-mod for chunk 0, add-mod after), batch z = s*b +
// lambda, modulus l = g0 + z / b from __constant__ (imd::kernels::VisitorModP).
//
// One tile configuration: 128x256x64, 3 stages (72 KB smem), verified in M0
// (tests/test_cutlass_mod_epilogue.cu). M4 A/B runs (docs/NOTES-M4.md) found 256x128x64 s3
// within noise of it and 128x128x64 s4/s5 slower on every benchmark shape, so the DESIGN §4
// tile menu is not instantiated. The epilogue already stores 16 u8 per access.
//
// DESIGN v1.2 §6: imd_plan_create builds the kernel parameters of every launch shape
// (k4_prepare: SM count, occupancy, smem attribute, CUTLASS Params); imd_contract copies
// them, sets its pointers with Params::update and launches Kernel2 directly. CUTLASS's
// GemmUniversalBase::initialize (and its device-property statics) is never called.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "immerdink/detail/constants.hpp"
#include "host/analysis.hpp"  // kK4*Bytes: libimmerdink_host's plan host bytes
#include "kernels/mod_epilogue.cuh"  // also brings in the Sm80 EVT visitors
#include "plan.hpp"
#include "profile.hpp"

namespace imd::detail {
namespace {

using namespace cute;
namespace c = imd::constants;

// Tile and stages are overridable at compile time for A/B benchmarks only.
#ifndef IMD_K4_TM
#  define IMD_K4_TM 128
#  define IMD_K4_TN 256
#  define IMD_K4_WM 64
#  define IMD_K4_WN 64
#  define IMD_K4_STAGES 3
#  define IMD_K4_DEFAULT_TILE 1
#endif
using TB = cutlass::gemm::GemmShape<IMD_K4_TM, IMD_K4_TN, 64>;
using Warp = cutlass::gemm::GemmShape<IMD_K4_WM, IMD_K4_WN, 64>;
using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = IMD_K4_STAGES;
constexpr int kEVTStages = 1;
// Identity swizzle with up to 8 N tiles per M step (CUTLASS picks log_tile from the tiled
// shape): a wave of 128 CTAs covers ~16 x 8 tiles instead of 128 x 1, so the A and B
// panels it streams stay in L2 at large k (16384^3: 1 GB less DRAM traffic per modulus).
// The tile offset's k() is still blockIdx.z = the batch index that VisitorModP reads.
#ifndef IMD_K4_SWIZZLE
#  define IMD_K4_SWIZZLE 8  // overridable for A/B benchmarks only
#endif
constexpr int kSwizzle = IMD_K4_SWIZZLE;
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
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<kSwizzle>, kStages,
        cutlass::arch::OpMultiplyAddSaturate, kEVTStages>::GemmKernel>;
using GemmStore = GemmFor<EVTStore>;
using GemmAdd = GemmFor<EVTAdd>;

// The launch structure (K-chunks, batch split, n-hat split) is libimmerdink_host's
// (host/k4_launches.hpp), so the host queries count exactly these launches. Its tile width
// width is this kernel's (a benchmark-only tile override passes its own to k4_grid; the
// host queries then differ only in the n-hat split, i.e. for n_hat > 65535 x tile N).
#ifdef IMD_K4_DEFAULT_TILE
static_assert(TB::kN == kK4TileN, "host/k4_launches.hpp kK4TileN != K4 tile N");
#endif

LaunchLimits g_limits = production_launch_limits();

// One K4 launch of a call (host/k4_launches.hpp K4Launch). Its CUTLASS parameters depend
// only on the shape key (kc, zc, nc; store-mod for chunk 0, add-mod after); the pointers and
// the epilogue's modulus offsets are set per call.
using Launch = K4Launch;
struct ShapeKey {
  int64_t kc, zc, nc;
  bool operator<(const ShapeKey& o) const {
    return kc != o.kc ? kc < o.kc : zc != o.zc ? zc < o.zc : nc < o.nc;
  }
};

constexpr int kStoreSmem = int(sizeof(typename GemmStore::GemmKernel::SharedStorage));
constexpr int kAddSmem = int(sizeof(typename GemmAdd::GemmKernel::SharedStorage));

}  // namespace

// Everything imd_contract needs to launch K4, fixed at plan creation (DESIGN v1.2 §6).
struct K4Plan {
  int64_t m = 0, n_hat = 0, k_hat = 0, b = 0;
  K4Grid grid;  // n_hat, k_hat, b and the launch limits at creation
  std::map<ShapeKey, typename GemmStore::GemmKernel::Params> store;  // chunk 0
  std::map<ShapeKey, typename GemmAdd::GemmKernel::Params> add;      // chunks > 0

  // The launches of one group of gsize moduli, in order: K-chunk, batch part, n-hat part.
  template <class F>
  void for_each_launch(int gsize, F&& f) const {
    k4_for_each_launch(grid, gsize, f);
  }
};

// libimmerdink_host's plan_host_bytes counts the K4Plan and its map entries with these
// sizes (host/analysis.hpp); a CUTLASS or K4Plan change must update them there.
static_assert(sizeof(K4Plan) == kK4PlanBytes, "update kK4PlanBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::store)::value_type) == kK4StoreEntryBytes,
              "update kK4StoreEntryBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::add)::value_type) == kK4AddEntryBytes,
              "update kK4AddEntryBytes in src/host/analysis.hpp");

namespace {

// The CUTLASS arguments of launch L (null pointers at plan creation).
template <class Gemm, class EVTArgs>
typename Gemm::Arguments gemm_args(const K4Plan& k, const Launch& L, EVTArgs const& cb,
                                   const int8_t* A, const int8_t* B) {
  const int64_t plane_a = k.m * k.k_hat, plane_b = k.n_hat * k.k_hat;
  const cutlass::gemm::GemmCoord size{static_cast<int>(k.m), static_cast<int>(L.nc),
                                      static_cast<int>(L.kc)};
  return {cutlass::gemm::GemmUniversalMode::kBatched, size, static_cast<int>(L.zc), cb, A, B,
          nullptr, nullptr, plane_a, plane_b, 0, 0, k.k_hat, k.k_hat, 0, 0};
}

// This call's operand and plane pointers for launch L of group g0.
struct CallPtrs {
  const int8_t* A;
  const int8_t* B;
  uint8_t* W;
};
CallPtrs call_ptrs(const K4Plan& k, const Launch& L, const int8_t* RA, const int8_t* RB,
                   uint8_t* W, int g0) {
  const int64_t plane_a = k.m * k.k_hat, plane_b = k.n_hat * k.k_hat, plane_w = k.m * k.n_hat;
  return {RA + L.z0 * plane_a + L.k0, RB + L.z0 * plane_b + L.j0 * k.k_hat + L.k0,
          W + (int64_t{g0} * k.b + L.z0) * plane_w + L.j0};
}

// Epilogue arguments: store-mod (chunk 0) or add-mod (later chunks) into the planes at Wp,
// modulus l = g0 + (z0 + blockIdx.z) / b.
typename EVTStore::Arguments store_args(const K4Plan& k, const Launch& L, uint8_t* Wp, int g0) {
  return {ModP::Arguments{g0, k.b, L.z0}, {Wp, {k.n_hat, _1{}, k.m * k.n_hat}}};
}
typename EVTAdd::Arguments add_args(const K4Plan& k, const Launch& L, uint8_t* Wp, int g0) {
  return {{{Wp, uint8_t(0), {k.n_hat, _1{}, k.m * k.n_hat}}, ModP::Arguments{g0, k.b, L.z0}},
          {Wp, {k.n_hat, _1{}, k.m * k.n_hat}}};
}

// Device setup of one kernel at plan creation: dynamic shared memory above 48 KB, then the
// occupancy CUTLASS's own initialization would query (thread-safe CUDA runtime calls).
template <class Gemm>
cudaError_t kernel_setup(int smem, int* occupancy) {
  using K = typename Gemm::GemmKernel;
  cudaError_t e = cudaSuccess;
  if (smem >= (48 << 10))
    e = cudaFuncSetAttribute(cutlass::Kernel2<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e == cudaSuccess)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        occupancy, cutlass::Kernel2<K>, K::kThreadCount, smem, cudaOccupancyDisableCachingOverride);
  return e;
}

// Copy the prepared parameters, set this call's pointers (Params::update: operand pointers,
// batch strides, epilogue arguments) and launch. The copy is local: the plan is not written.
template <class Gemm>
bool launch(typename Gemm::GemmKernel::Params params, typename Gemm::Arguments const& args,
            int smem, cudaStream_t stream) {
  using K = typename Gemm::GemmKernel;
  params.update(args);
  cutlass::Kernel2<K><<<params.get_grid_dims(), dim3(K::kThreadCount, 1, 1), smem, stream>>>(params);
  return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace

LaunchLimits& launch_limits() { return g_limits; }
LaunchLimits default_launch_limits() { return production_launch_limits(); }

imd_status_t k4_prepare(const ExecPlan& p, std::shared_ptr<const K4Plan>* out) {
  auto k = std::make_shared<K4Plan>();
  k->m = p.m, k->n_hat = p.n_hat, k->k_hat = p.k_hat, k->b = p.b;
  k->grid = k4_grid(p.n_hat, p.k_hat, p.b, g_limits, TB::kN);
  int occ_store = 0, occ_add = 0;
  if (kernel_setup<GemmStore>(kStoreSmem, &occ_store) != cudaSuccess ||
      kernel_setup<GemmAdd>(kAddSmem, &occ_add) != cudaSuccess) {
    cudaGetLastError();  // clear the (non-sticky) error state
    return IMD_STATUS_CUDA_ERROR;
  }
  bool ok = true;
  for (int g0 = 0; g0 < p.N; g0 += p.g) {
    k->for_each_launch(std::min(p.g, p.N - g0), [&](const Launch& L) {
      const ShapeKey key{L.kc, L.zc, L.nc};
      if (L.k0 == 0 && !k->store.count(key)) {
        const auto args = gemm_args<GemmStore>(*k, L, store_args(*k, L, nullptr, 0), nullptr, nullptr);
        ok = ok && GemmStore::can_implement(args) == cutlass::Status::kSuccess;
        k->store.emplace(key, typename GemmStore::GemmKernel::Params(args, p.sms, occ_store));
      } else if (L.k0 != 0 && !k->add.count(key)) {
        const auto args = gemm_args<GemmAdd>(*k, L, add_args(*k, L, nullptr, 0), nullptr, nullptr);
        ok = ok && GemmAdd::can_implement(args) == cutlass::Status::kSuccess;
        k->add.emplace(key, typename GemmAdd::GemmKernel::Params(args, p.sms, occ_add));
      }
    });
  }
  if (!ok) return IMD_STATUS_NOT_SUPPORTED;
  *out = std::move(k);
  return IMD_STATUS_SUCCESS;
}

K4Footprint k4_footprint(const K4Plan& k, int N, int g) {
  K4Footprint f;
  f.store_shapes = static_cast<int64_t>(k.store.size());
  f.add_shapes = static_cast<int64_t>(k.add.size());
  for (int g0 = 0; g0 < N; g0 += g) k.for_each_launch(std::min(g, N - g0), [&](const Launch&) { ++f.launches; });
  f.bytes = sizeof(K4Plan) + k.store.size() * sizeof(decltype(k.store)::value_type) +
            k.add.size() * sizeof(decltype(k.add)::value_type);
  return f;
}

imd_status_t k4_group(const K4Plan& k, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
                      int gsize, cudaStream_t stream) {
  bool ok = true;
  k.for_each_launch(gsize, [&](const Launch& L) {
    if (!ok) return;
    const ShapeKey key{L.kc, L.zc, L.nc};
    const CallPtrs q = call_ptrs(k, L, RA, RB, W, g0);
    if (L.k0 == 0) {
      const auto it = k.store.find(key);
      ok = it != k.store.end() &&
           launch<GemmStore>(it->second,
                             gemm_args<GemmStore>(k, L, store_args(k, L, q.W, g0), q.A, q.B),
                             kStoreSmem, stream);
    } else {
      const auto it = k.add.find(key);
      ok = it != k.add.end() &&
           launch<GemmAdd>(it->second, gemm_args<GemmAdd>(k, L, add_args(k, L, q.W, g0), q.A, q.B),
                           kAddSmem, stream);
    }
    IMD_PROF(stream, L.k0 == 0 ? "K4 gemm store" : "K4 gemm add");
  });
  return ok ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace imd::detail
