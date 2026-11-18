// IMD_HWGEMM dispatcher (DESIGN §6i): normalize C to row-major, pick the arch's kernels and
// the tile, chunk, launch. See hwgemm.hpp.
#include "hwgemm.hpp"

#include <utility>

#include "hwgemm/hw_gemm.cuh"

namespace imd::detail {
namespace {

struct DevInfo {
  int device = -1, major = 0, sms = 0;
};

// The current device's compute capability and SM count, cached per host thread and device.
bool dev_info(DevInfo* out) {
  thread_local DevInfo cache;
  int d = 0;
  if (cudaGetDevice(&d) != cudaSuccess) return false;
  if (cache.device != d) {
    DevInfo x;
    x.device = d;
    if (cudaDeviceGetAttribute(&x.major, cudaDevAttrComputeCapabilityMajor, d) != cudaSuccess ||
        cudaDeviceGetAttribute(&x.sms, cudaDevAttrMultiProcessorCount, d) != cudaSuccess)
      return false;
    cache = x;
  }
  *out = cache;
  return true;
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

int32_t hwgemm_tile(int64_t m, int64_t n, int64_t b, int cc_major, int sms) {
  if (cc_major < 9) return kHw64x64;
  if (m <= 64 || n <= 64) return kHw64x64;
  const int64_t ctas128 = ceil_div(m, 128) * ceil_div(n, 128) * b;
  return 2 * ctas128 < sms ? kHw64x128 : kHw128x128;
}

int& hw_tile_override() {
  static int v = -1;
  return v;
}

namespace {

__global__ void __launch_bounds__(256) hw_beta_kernel(double* C, int64_t m, int64_t n, int64_t b,
                                                      int64_t ldc, bool row, int64_t cbs,
                                                      double beta) {
  const int64_t total = b * m * n;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t j = idx % n, t = idx / n, i = t % m, lam = t / m;
    double* c = C + lam * cbs + (row ? i * ldc + j : i + j * ldc);
    *c = beta == 0.0 ? 0.0 : __dmul_rn(beta, *c);
  }
}

template <class Kernel>
cudaError_t versions_of(Kernel kernel, int32_t* binary, int32_t* ptx) {
  cudaFuncAttributes a{};
  const cudaError_t e = cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(kernel));
  if (e == cudaSuccess) *binary = a.binaryVersion, *ptx = a.ptxVersion;
  return e;
}

}  // namespace

cudaError_t hw_kernel_versions(int32_t index, int32_t* binary, int32_t* ptx) {
  switch (index) {
    case 0: return versions_of(hw_beta_kernel, binary, ptx);
    case 1: return hw_sm80_versions(binary, ptx);
    case 2: return hw_sm90_versions(binary, ptx);
    default: return cudaErrorInvalidValue;
  }
}

imd_status_t hwgemm_beta(const HwGemmCall& c, cudaStream_t stream) {
  const int64_t total = c.b * c.m * c.n;
  if (total == 0) return IMD_STATUS_SUCCESS;
  const int64_t blocks = ceil_div(total, 256);
  const unsigned grid = static_cast<unsigned>(blocks < 65535 * 8 ? blocks : 65535 * 8);
  hw_beta_kernel<<<grid, 256, 0, stream>>>(c.C, c.m, c.n, c.b, c.ldc, c.c_j_contig, c.c_bstride,
                                           c.beta);
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

imd_status_t hwgemm_enqueue(const HwGemmCall& call, cudaStream_t stream) {
  if (call.b == 0 || call.m == 0 || call.n == 0) return IMD_STATUS_SUCCESS;
  DevInfo di;
  if (!dev_info(&di)) return IMD_STATUS_CUDA_ERROR;
  const bool sm90 = di.major >= 9 && hw_sm90_usable();
  if (!sm90 && di.major < 8) return IMD_STATUS_NOT_SUPPORTED;
  HwNorm c{call.m, call.n, call.k, call.b, call.A, call.B, call.C, call.ldc, call.c_bstride,
           call.alpha, call.beta, call.tile};
  if (!call.c_j_contig) {  // C^T = B^T A^T: row-major C^T, same K order
    std::swap(c.m, c.n);
    std::swap(c.A, c.B);
  }
  if (c.tile < 0) c.tile = hwgemm_tile(c.m, c.n, c.b, sm90 ? 9 : 8, di.sms);
  if (!sm90 && c.tile != kHw64x64) return IMD_STATUS_INVALID_VALUE;
  return sm90 ? hw_sm90_run(c, stream) : hw_sm80_run(c, stream);
}

}  // namespace imd::detail
