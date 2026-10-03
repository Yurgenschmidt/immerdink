// K4 (DESIGN §3.5, §4): per residue group and K-chunk, a batched int8 TN GEMM with the
// mod-p -> u8 epilogue (store-mod for chunk 0, add-mod after), batch z = s*b + lambda,
// modulus l = g0 + z / b from __constant__.
//
// A2: the kernel is imd::kernels::K4Gemm (kernels/k4_gemm.cuh): CUTLASS 2.x's multistage
// int8 main loop in a persistent kernel whose epilogue works in registers (no shared memory,
// no barriers) and overlaps the next tile's cp.async prologue. It replaced CUTLASS's
// GemmUniversal + Sm80 EVT epilogue (kernels/mod_epilogue.cuh, still tested by
// tests/test_cutlass_mod_epilogue.cu), which left K4 at 60-66 % of its bound at k = 1024
// (docs/NOTES-A2.md). Same arithmetic, bit-identical results.
//
// One tile configuration: 128x256x64, 3 stages (72 KB smem), 8 warps of 64x64. M4 A/B runs
// (docs/NOTES-M4.md) and A2's (docs/NOTES-A2.md) found no better one for the benchmark
// shapes, so the DESIGN §4 tile menu is not instantiated.
//
// DESIGN v1.2 §6: imd_plan_create builds the kernel parameters of every launch shape
// (k4_prepare: SM count, occupancy, smem attribute, Params); imd_contract copies them, sets
// its pointers and launches. No CUTLASS device-level host state is used.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "immerdink/detail/constants.hpp"
#include "host/analysis.hpp"  // kK4*Bytes: libimmerdink_host's plan host bytes
#include "kernels/k4_gemm.cuh"
#include "plan.hpp"
#include "profile.hpp"

#ifdef IMD_K4_NOSAT_MMA
// A2 benchmark only (-DIMD_K4_NOSAT_MMA): CUTLASS defines the sm80 int8 m16n8k32 MMA only
// with .satfinite; this is the same instruction without it, to measure what the saturation
// costs (with K_c chunking no partial sum overflows, so the bits are the same). Not used by
// any production build.
namespace cutlass::arch {
template <>
struct Mma<gemm::GemmShape<16, 8, 32>, 32, int8_t, layout::RowMajor, int8_t, layout::ColumnMajor,
           int, layout::RowMajor, OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 8, 32>;
  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;
  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;
  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;
  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;
  CUTLASS_HOST_DEVICE void operator()(FragmentC& d, FragmentA const& a, FragmentB const& b,
                                      FragmentC const& c) const {
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    int const* C = reinterpret_cast<int const*>(&c);
    int* D = reinterpret_cast<int*>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]),
          "r"(C[1]), "r"(C[2]), "r"(C[3]));
  }
};
}  // namespace cutlass::arch
#  define IMD_K4_OP cutlass::arch::OpMultiplyAdd
#else
#  define IMD_K4_OP cutlass::arch::OpMultiplyAddSaturate
#endif

namespace imd::detail {
namespace {

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
using G = imd::kernels::K4Gemm<TB, Warp, IMD_K4_STAGES, IMD_K4_OP>;
using Params = typename G::Params;

// The launch structure (K-chunks, batch split, n-hat split) is libimmerdink_host's
// (host/k4_launches.hpp), so the host queries count exactly these launches. Its tile width
// is this kernel's (a benchmark-only tile override passes its own to k4_grid; the host
// queries then differ only in the n-hat split, i.e. for n_hat > 65535 x tile N). The
// persistent kernel itself has no grid limits; the splits are kept so the launch structure
// (and the specs and host queries that count it) is unchanged.
#ifdef IMD_K4_DEFAULT_TILE
static_assert(TB::kN == kK4TileN, "host/k4_launches.hpp kK4TileN != K4 tile N");
#endif

LaunchLimits g_limits = production_launch_limits();

// One K4 launch of a call (host/k4_launches.hpp K4Launch). Its kernel parameters depend
// only on the shape key (kc, zc, nc; store-mod for chunk 0, add-mod after); the pointers and
// the epilogue's modulus offsets are set per call.
using Launch = K4Launch;
struct ShapeKey {
  int64_t kc, zc, nc;
  bool operator<(const ShapeKey& o) const {
    return kc != o.kc ? kc < o.kc : zc != o.zc ? zc < o.zc : nc < o.nc;
  }
};

constexpr int kSmem = G::kSmemBytes;

}  // namespace

// Everything imd_contract needs to launch K4, fixed at plan creation (DESIGN v1.2 §6).
struct K4Plan {
  int64_t m = 0, n_hat = 0, k_hat = 0, b = 0;
  K4Grid grid;  // n_hat, k_hat, b and the launch limits at creation
  std::map<ShapeKey, Params> store;  // chunk 0
  std::map<ShapeKey, Params> add;    // chunks > 0

  // The launches of one group of gsize moduli, in order: K-chunk, batch part, n-hat part.
  template <class F>
  void for_each_launch(int gsize, F&& f) const {
    k4_for_each_launch(grid, gsize, f);
  }
};

// libimmerdink_host's plan_host_bytes counts the K4Plan and its map entries with these
// sizes (host/analysis.hpp); a kernel-parameter or K4Plan change must update them there.
static_assert(sizeof(K4Plan) == kK4PlanBytes, "update kK4PlanBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::store)::value_type) == kK4StoreEntryBytes,
              "update kK4StoreEntryBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::add)::value_type) == kK4AddEntryBytes,
              "update kK4AddEntryBytes in src/host/analysis.hpp");

namespace {

// Device setup of one kernel at plan creation: dynamic shared memory above 48 KB, then its
// occupancy (thread-safe CUDA runtime calls).
template <bool kAdd>
cudaError_t kernel_setup(int* occupancy) {
  void (*kernel)(Params) = imd::kernels::k4_kernel<G, kAdd>;
  cudaError_t e = cudaSuccess;
  if (kSmem >= (48 << 10))
    e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (e == cudaSuccess)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(occupancy, kernel, G::kThreads, kSmem,
                                                               cudaOccupancyDisableCachingOverride);
  return e;
}

// This call's parameters for launch L of group g0: operand, plane and modulus offsets.
Params call_params(const K4Plan& k, Params p, const Launch& L, const int8_t* RA, const int8_t* RB,
                   uint8_t* W, int g0) {
  const int64_t plane_a = k.m * k.k_hat, plane_b = k.n_hat * k.k_hat, plane_w = k.m * k.n_hat;
  G::set_call(p, RA + L.z0 * plane_a + L.k0, RB + L.z0 * plane_b + L.j0 * k.k_hat + L.k0,
              W + (int64_t{g0} * k.b + L.z0) * plane_w + L.j0, plane_a, plane_b, plane_w, g0, k.b,
              L.z0);
  return p;
}

template <bool kAdd>
bool launch(Params const& p, cudaStream_t stream) {
  imd::kernels::k4_kernel<G, kAdd><<<p.grid, G::kThreads, kSmem, stream>>>(p);
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
  if (kernel_setup<false>(&occ_store) != cudaSuccess || kernel_setup<true>(&occ_add) != cudaSuccess) {
    cudaGetLastError();  // clear the (non-sticky) error state
    return IMD_STATUS_CUDA_ERROR;
  }
  if (occ_store < 1 || occ_add < 1) return IMD_STATUS_NOT_SUPPORTED;
  // GEMM sizes are ints (DESIGN §4 launch limits: m, n_hat <= 2^31 - 1, kc <= K_c'); k_hat
  // and n_hat are multiples of 16, so every 128-bit access is aligned.
  bool ok = k->m >= 1 && k->m <= INT32_MAX && k->n_hat % 16 == 0 && k->k_hat % 16 == 0;
  for (int g0 = 0; ok && g0 < p.N; g0 += p.g) {
    k->for_each_launch(std::min(p.g, p.N - g0), [&](const Launch& L) {
      const ShapeKey key{L.kc, L.zc, L.nc};
      auto& map = L.k0 == 0 ? k->store : k->add;
      if (map.count(key)) return;
      ok = ok && L.nc <= INT32_MAX && L.kc <= INT32_MAX && L.nc % 16 == 0;
      const int64_t resident = int64_t{p.sms} * (L.k0 == 0 ? occ_store : occ_add);
      map.emplace(key, G::make_params(static_cast<int>(k->m), static_cast<int>(L.nc),
                                      static_cast<int>(L.kc), k->k_hat, k->k_hat, k->n_hat, L.zc,
                                      resident));
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
    const auto& map = L.k0 == 0 ? k.store : k.add;
    const auto it = map.find(key);
    ok = it != map.end();
    if (ok) {
      const Params p = call_params(k, it->second, L, RA, RB, W, g0);
      ok = L.k0 == 0 ? launch<false>(p, stream) : launch<true>(p, stream);
    }
    IMD_PROF(stream, L.k0 == 0 ? "K4 gemm store" : "K4 gemm add");
  });
  return ok ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace imd::detail
