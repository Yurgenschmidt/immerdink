// One K4 configuration (host/k4_config.hpp kK4Configs[IMD_K4_CFG]) as a K4Ops entry. Each
// src/k4/k4_cfg<i>.cu defines IMD_K4_CFG and includes this file, so every CUTLASS
// instantiation compiles in its own translation unit (parallel builds, bounded compiler
// memory per job).
#pragma once

#ifndef IMD_K4_CFG
#  error "define IMD_K4_CFG (the configuration index) before including k4/k4_cfg.cuh"
#endif

#include <cstring>
#include <type_traits>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "host/k4_config.hpp"
#include "k4/k4_ops.hpp"
#include "kernels/k4_gemm.cuh"

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

constexpr K4ConfigDesc kCfg = kK4Configs[IMD_K4_CFG];
using TB = cutlass::gemm::GemmShape<kCfg.tile_m, kCfg.tile_n, 64>;
using Warp = cutlass::gemm::GemmShape<kCfg.warp_m, kCfg.warp_n, 64>;
using G = imd::kernels::K4Gemm<TB, Warp, kCfg.stages, IMD_K4_OP>;
using Params = typename G::Params;
static_assert(G::kThreads == k4_threads(kCfg), "host/k4_config.hpp warp count");
static_assert(sizeof(Params) <= kK4BlobBytes && alignof(Params) <= alignof(K4Blob),
              "K4Gemm::Params does not fit k4/k4_ops.hpp K4Blob");
static_assert(std::is_trivially_copyable_v<Params>, "K4Blob holds Params by bytes");

template <bool kAdd>
cudaError_t setup_one(int* occupancy) {
  void (*kernel)(Params) = imd::kernels::k4_kernel<G, kAdd>;
  cudaError_t e = cudaSuccess;
  if (G::kSmemBytes >= (48 << 10))
    e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
  if (e == cudaSuccess)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(occupancy, kernel, G::kThreads,
                                                               G::kSmemBytes,
                                                               cudaOccupancyDisableCachingOverride);
  return e;
}

cudaError_t setup(int* occ_store, int* occ_add) {
  const cudaError_t e = setup_one<false>(occ_store);
  return e != cudaSuccess ? e : setup_one<true>(occ_add);
}

void make(K4Blob* out, int m, int n, int k, int64_t lda, int64_t ldb, int64_t ldw,
          int64_t batches, int64_t resident) {
  const Params p = G::make_params(m, n, k, lda, ldb, ldw, batches, resident);
  std::memset(out->bytes, 0, sizeof out->bytes);
  std::memcpy(out->bytes, &p, sizeof p);
}

bool launch(const K4Blob& blob, bool add, const K4Call& c, cudaStream_t stream) {
  Params p;
  std::memcpy(&p, blob.bytes, sizeof p);
  G::set_call(p, c.A, c.B, c.W, c.batch_a, c.batch_b, c.plane_w, c.slot0, c.b, c.batch0);
  if (add)
    imd::kernels::k4_kernel<G, true><<<p.grid, G::kThreads, G::kSmemBytes, stream>>>(p);
  else
    imd::kernels::k4_kernel<G, false><<<p.grid, G::kThreads, G::kSmemBytes, stream>>>(p);
  return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace

#define IMD_K4_CAT2(a, b) a##b
#define IMD_K4_CAT(a, b) IMD_K4_CAT2(a, b)
const K4Ops& IMD_K4_CAT(k4_ops_cfg, IMD_K4_CFG)() {
  static const K4Ops ops{G::kThreads, G::kSmemBytes, &setup, &make, &launch};
  return ops;
}

}  // namespace imd::detail
