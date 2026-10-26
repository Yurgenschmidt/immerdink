// One tile variant of the Hopper K4 backend (src/k4/hopper/k4_wgmma_ops.hpp) as a K4WgmmaOps entry (issue 22, docs/NOTES-22.md "Library
// integration (design)"). src/k4/hopper/k4_wgmma_v<i>.cu defines the variant's tile / cluster / schedule and includes this file, so every
// CUTLASS instantiation compiles in its own translation unit. The kernels are NOT copied: this is k4_hopper.cuh's `Variant` (CUTLASS 3.x
// main loop + K4ModEpilogue with the fast HopperEpilogue), the type rung 0 / rung 1 ran on the cards; only the launch is new. It builds
// the kernel parameters (TMA descriptors included) on the stack per launch and launches with cudaLaunchKernelEx (cluster attribute), the
// way GemmUniversalAdapter::run does, but without the adapter's per-launch cudaFuncSetAttribute and without any query: everything per
// device is done once in `setup` at plan creation. Compiled for sm_90a only (CMake target imd_k4_hopper); the code is never reached on
// another device (gemm_k4.cu's gate), where the kernel image does not exist and a launch would fail with NoKernelImageForDevice.
#pragma once

#ifndef IMD_K4W_TILE
#  error "define IMD_K4W_TILE, IMD_K4W_CLUSTER, IMD_K4W_SCHED and IMD_K4W_NAME before including k4/hopper/k4_wgmma_impl.cuh"
#endif

#include <cstring>

#include <cuda_runtime.h>

#include "host/k4_wgmma.hpp"
#include "k4/hopper/k4_hopper.cuh"
#include "k4/hopper/k4_wgmma_ops.hpp"

namespace imd::detail {
namespace {

namespace cg = cutlass::gemm;
using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;

template <bool kAdd>
struct W {
  using V = imd::hopper::Variant<IMD_K4W_TILE, IMD_K4W_CLUSTER, IMD_K4W_SCHED, kAdd, /*kFast=*/true>;
  using Kernel = typename V::Kernel;
  static_assert(Kernel::SharedStorageSize <= 227 * 1024, "the kernel's shared memory exceeds the sm_90 limit");
  static_assert(Kernel::MaxThreadsPerBlock == 384, "cooperative: 2 consumer warpgroups + 1 producer warpgroup");

  static const void* fn() { return reinterpret_cast<const void*>(&cutlass::device_kernel<Kernel>); }

  static cudaLaunchConfig_t config(cudaLaunchAttribute* at, dim3 grid, cudaStream_t stream) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(Kernel::MaxThreadsPerBlock, 1, 1);
    cfg.dynamicSmemBytes = Kernel::SharedStorageSize;
    cfg.stream = stream;
    at->id = cudaLaunchAttributeClusterDimension;
    at->val.clusterDim.x = unsigned(cute::size<0>(typename Kernel::ClusterShape{}));
    at->val.clusterDim.y = unsigned(cute::size<1>(typename Kernel::ClusterShape{}));
    at->val.clusterDim.z = unsigned(cute::size<2>(typename Kernel::ClusterShape{}));
    cfg.attrs = at;
    cfg.numAttrs = 1;
    return cfg;
  }

  static cudaError_t setup(int* clusters, int* binary) {
    cudaError_t e = cudaFuncSetAttribute(fn(), cudaFuncAttributeMaxDynamicSharedMemorySize, int(Kernel::SharedStorageSize));
    if (e != cudaSuccess) return e;
    cudaFuncAttributes fa{};
    if ((e = cudaFuncGetAttributes(&fa, fn())) != cudaSuccess) return e;
    *binary = fa.binaryVersion;
    cudaLaunchAttribute at{};
    cudaLaunchConfig_t cfg = config(&at, dim3(unsigned(cute::size<0>(typename Kernel::ClusterShape{})),
                                              unsigned(cute::size<1>(typename Kernel::ClusterShape{})),
                                              unsigned(cute::size<2>(typename Kernel::ClusterShape{}))), nullptr);
    return cudaOccupancyMaxActiveClusters(clusters, fn(), &cfg);
  }

  static typename Kernel::Arguments arguments(const K4WgmmaArgs& a, const K4Call& c) {
    typename Kernel::Arguments args{};
    args.mode = cg::GemmUniversalMode::kGemm;
    args.problem_shape = {a.m, a.nc, a.kc, a.zc};
    args.mainloop.ptr_A = c.A;
    args.mainloop.dA = cute::make_stride(a.k_hat, cute::Int<1>{}, c.batch_a);  // A[z][m][k]: row stride k_hat, batch stride batch_a
    args.mainloop.ptr_B = c.B;
    args.mainloop.dB = cute::make_stride(a.k_hat, cute::Int<1>{}, c.batch_b);
    imd::hopper::K4Args ea{};
    ea.W = c.W, ea.ldw = a.n_hat, ea.plane_w = a.plane_w, ea.slot0 = c.slot0, ea.b = c.b, ea.batch0 = c.batch0;
    args.epilogue = ea;
    args.hw_info.device_id = 0;  // unused: sm_count and max_active_clusters are given
    args.hw_info.sm_count = a.sms;
    args.hw_info.max_active_clusters = a.max_clusters;
    args.scheduler.max_swizzle_size = a.swizzle;
    args.scheduler.raster_order = a.raster == kRasterAlongM ? Raster::AlongM : a.raster == kRasterAlongN ? Raster::AlongN : Raster::Heuristic;
    return args;
  }

  static bool launch(const K4WgmmaArgs& a, const K4Call& c, cudaStream_t stream) {
    const auto args = arguments(a, c);
    if (!Kernel::can_implement(args)) return false;
    const typename Kernel::Params params = Kernel::to_underlying_arguments(args, nullptr);  // the TMA descriptors, on the stack
    cudaLaunchAttribute at{};
    const cudaLaunchConfig_t cfg = config(&at, Kernel::get_grid_shape(params), stream);
    void* kargs[] = {const_cast<typename Kernel::Params*>(&params)};
    return cudaLaunchKernelExC(&cfg, fn(), kargs) == cudaSuccess && cudaPeekAtLastError() == cudaSuccess;
  }

  static bool dry_run(const K4WgmmaArgs& a, const K4Call& c, bool encode, int* gx, int* gy, int* gz) {
    const auto args = arguments(a, c);
    if (!Kernel::can_implement(args)) return false;
    if (!encode) return true;
    const typename Kernel::Params params = Kernel::to_underlying_arguments(args, nullptr);
    const dim3 g = Kernel::get_grid_shape(params);
    *gx = int(g.x), *gy = int(g.y), *gz = int(g.z);
    return true;
  }
};

cudaError_t setup(int* max_clusters, int* binary_version) {
  int cs = 0, ca = 0, bs = 0, ba = 0;
  cudaError_t e = W<false>::setup(&cs, &bs);
  if (e == cudaSuccess) e = W<true>::setup(&ca, &ba);
  if (e != cudaSuccess) return e;
  *max_clusters = cs < ca ? cs : ca;
  *binary_version = bs == ba ? bs : -1;
  return cudaSuccess;
}

bool launch(const K4WgmmaArgs& a, const K4Call& c, bool add, cudaStream_t stream) {
  return add ? W<true>::launch(a, c, stream) : W<false>::launch(a, c, stream);
}

bool dry_run(const K4WgmmaArgs& a, const K4Call& c, bool add, bool encode, int* gx, int* gy, int* gz) {
  return add ? W<true>::dry_run(a, c, encode, gx, gy, gz) : W<false>::dry_run(a, c, encode, gx, gy, gz);
}

}  // namespace

#define IMD_K4W_CAT2(a, b) a##b
#define IMD_K4W_CAT(a, b) IMD_K4W_CAT2(a, b)
const K4WgmmaOps& IMD_K4W_CAT(k4_wgmma_ops_, IMD_K4W_NAME)() {
  static const K4WgmmaOps ops{int(W<false>::Kernel::SharedStorageSize), int(W<false>::Kernel::MaxThreadsPerBlock), &setup, &launch, &dry_run};
  return ops;
}

}  // namespace imd::detail
