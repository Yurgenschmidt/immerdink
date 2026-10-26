// The Hopper K4 backend as a C++-only interface (issue 22, docs/NOTES-22.md "Library integration (design)"): one entry per tile
// variant of host/k4_wgmma.hpp, implemented in src/k4/hopper/k4_wgmma_v<i>.cu (sm_90a only, CUTLASS 3.x wgmma kernels with K4's register
// epilogue; src/k4/hopper/k4_hopper.cuh). Present only in builds with IMMERDINK_K4_WGMMA (IMD_K4_WGMMA); gemm_k4.cu never names
// the CUTLASS types, so the rest of the library compiles the same with or without it.
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "k4/k4_ops.hpp"

namespace imd::detail {

// What a launch needs besides the call's pointers (K4Call): the GEMM extents of this launch and the strides, taken from the plan.
struct K4WgmmaArgs {
  int m, nc, kc, zc;           // rows, columns of this n-hat part, K of this chunk, batches of this part
  int64_t k_hat, n_hat;        // operand row stride, W row stride
  int64_t plane_w;             // W plane stride
  int sms, max_clusters;       // the device's SMs, the queried persistent-grid cluster count
  int swizzle, raster;         // CUTLASS scheduler arguments (host/k4_wgmma.hpp K4WgmmaPick)
};

struct K4WgmmaOps {
  int smem_bytes, threads;
  // Plan creation, on the current device: the dynamic shared-memory attribute of the store-mod and add-mod kernels, their code
  // version (cudaFuncGetAttributes binaryVersion must be 90: the sm_90a image), and the cluster count the persistent grid is sized
  // by (cudaOccupancyMaxActiveClusters, the minimum over the two kernels). Any CUDA error: the backend is unavailable.
  cudaError_t (*setup)(int* max_clusters, int* binary_version);
  // Build the arguments (TMA descriptors included) on the stack, launch; false on a launch error. Never called for a launch
  // host/k4_wgmma.hpp k4_wgmma_illegal rejects.
  bool (*launch)(const K4WgmmaArgs& a, const K4Call& call, bool add, cudaStream_t stream);
  // Test only: everything `launch` does except the launch: can_implement and, if `encode`, to_underlying_arguments (the host-side TMA
  // encoding; cuTensorMapEncodeTiled is refused by the driver on a device below sm_90, so the caller passes encode only on one at or
  // above it) and the persistent grid (left 0 without encode). False if CUTLASS's can_implement refuses.
  bool (*dry_run)(const K4WgmmaArgs& a, const K4Call& call, bool add, bool encode, int* grid_x, int* grid_y, int* grid_z);
};

// nullptr for a variant this build does not have.
const K4WgmmaOps* k4_wgmma_ops(int variant);

}  // namespace imd::detail
