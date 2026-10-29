// The K4 configurations as a run-time table (DESIGN v1.5 §4 "K4 configurations"): one
// K4Gemm instantiation per host/k4_config.hpp kK4Configs entry, each compiled in its own
// translation unit (src/k4/k4_cfg<i>.cu, from k4_cfg.cuh), behind this C++-only interface,
// so gemm_k4.cu can build parameters for, launch and time any of them.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace imd::detail {

// One launch's kernel parameters (K4Gemm<...>::Params, trivially copyable), type-erased so
// that a plan's parameter maps have one entry size whatever the configuration (the plan
// host bytes of libimmerdink_host do not depend on the choice).
constexpr size_t kK4BlobBytes = 256;
struct alignas(8) K4Blob {
  unsigned char bytes[kK4BlobBytes];
};

// This call's operand, plane and modulus offsets of one launch (host/k4_launches.hpp K4Launch).
struct K4Call {
  const int8_t* A;
  const int8_t* B;
  uint8_t* W;
  int64_t batch_a, batch_b, plane_w;
  int slot0;
  int64_t b, batch0;
};

struct K4Ops {
  int threads, smem_bytes;
  // Plan creation: the dynamic shared-memory attribute, then the occupancy of the store-mod
  // and add-mod kernels (thread-safe runtime calls).
  cudaError_t (*setup)(int* occ_store, int* occ_add);
  // The code the store-mod and add-mod kernels run as on the current device (cudaFuncGetAttributes):
  // binaryVersion / ptxVersion of each (host/device_class.cpp imd_kernel_code_kind reads them).
  cudaError_t (*versions)(int* binary_store, int* ptx_store, int* binary_add, int* ptx_add);
  // The parameters of a launch of GEMM size m x n x k over `batches` batches with
  // `resident` persistent CTAs.
  void (*make)(K4Blob* out, int m, int n, int k, int64_t lda, int64_t ldb, int64_t ldw,
               int64_t batches, int64_t resident);
  // Copy the parameters, set the call's pointers and launch; false on a launch error.
  bool (*launch)(const K4Blob& params, bool add, const K4Call& call, cudaStream_t stream);
};

// The configuration `cfg` (IMD_K4_*, 0 <= cfg < IMD_K4_CONFIG_COUNT).
const K4Ops& k4_ops(int cfg);

}  // namespace imd::detail
