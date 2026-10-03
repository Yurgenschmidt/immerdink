// CUDA smoke test: the sm_89 toolchain builds and runs, the statically initialized
// __constant__ tables equal the host tables byte for byte, and the §3.5 epilogue
// reduction (imd::device::mod_p) is exact over EVERY reachable input of every modulus.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "device/constants.cuh"

namespace {

namespace c = imd::constants;

#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    const cudaError_t err_ = (x);                                                  \
    if (err_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x,              \
                   cudaGetErrorString(err_));                                      \
      std::exit(2);                                                                \
    }                                                                              \
  } while (0)

__global__ void copy_tables(c::Tables* out) {
  // Byte-wise copy so every table byte is read from constant memory on the device.
  const auto* src = reinterpret_cast<const unsigned char*>(&imd::device::c_tables);
  auto* dst = reinterpret_cast<unsigned char*>(out);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizeof(c::Tables);
       i += size_t{gridDim.x} * blockDim.x)
    dst[i] = src[i];
}

// For modulus l, every v in [epi_vmin[l], epi_vmax[l]]: mod_p(v, l) must equal the
// non-negative remainder computed independently with the hardware 32-bit `%`.
__global__ void mod_p_exhaustive(int l, unsigned long long* mismatches,
                                 unsigned long long* evaluated, int* first_bad) {
  const int64_t lo = imd::device::c_tables.epi_vmin[l];
  const int64_t hi = imd::device::c_tables.epi_vmax[l];
  const auto p = static_cast<int32_t>(imd::device::c_tables.moduli[l]);
  const int64_t stride = int64_t{gridDim.x} * blockDim.x;
  unsigned long long bad = 0, count = 0;
  for (int64_t v64 = lo + blockIdx.x * int64_t{blockDim.x} + threadIdx.x; v64 <= hi; v64 += stride) {
    const auto v = static_cast<int32_t>(v64);
    int32_t want = v % p;
    if (want < 0) want += p;
    const uint32_t got = imd::device::mod_p(v, l);
    if (got != static_cast<uint32_t>(want)) {
      ++bad;
      atomicCAS(first_bad, 0x7fffffff, v);
    }
    ++count;
  }
  atomicAdd(mismatches, bad);
  atomicAdd(evaluated, count);
}

}  // namespace

int main() {
  int failures = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  c::Tables* d_tables = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tables, sizeof(c::Tables)));
  copy_tables<<<32, 256>>>(d_tables);
  CUDA_CHECK(cudaGetLastError());
  auto* h_tables = new c::Tables;
  CUDA_CHECK(cudaMemcpy(h_tables, d_tables, sizeof(c::Tables), cudaMemcpyDeviceToHost));
  if (std::memcmp(h_tables, &c::kTables, sizeof(c::Tables)) != 0) {
    std::fprintf(stderr, "FAIL: device constant tables differ from host kTables\n");
    ++failures;
  } else {
    std::printf("device __constant__ tables == host tables (%zu bytes)\n", sizeof(c::Tables));
  }
  delete h_tables;
  CUDA_CHECK(cudaFree(d_tables));

  unsigned long long* d_counts = nullptr;
  int* d_first = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counts, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_first, sizeof(int)));
  unsigned long long total = 0;
  for (int l = 0; l < c::kNMax; ++l) {
    const int first_init = 0x7fffffff;
    CUDA_CHECK(cudaMemset(d_counts, 0, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_first, &first_init, sizeof(int), cudaMemcpyHostToDevice));
    mod_p_exhaustive<<<prop.multiProcessorCount * 8, 256>>>(l, d_counts, d_counts + 1, d_first);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long counts[2];
    int first = 0;
    CUDA_CHECK(cudaMemcpy(counts, d_counts, sizeof counts, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&first, d_first, sizeof first, cudaMemcpyDeviceToHost));
    const auto expected =
        static_cast<unsigned long long>(int64_t{c::kTables.epi_vmax[l]} - c::kTables.epi_vmin[l] + 1);
    if (counts[0] != 0 || counts[1] != expected) {
      std::fprintf(stderr, "FAIL: p=%u: %llu mismatches (first v=%d), %llu of %llu evaluated\n",
                   c::kTables.moduli[l], counts[0], first, counts[1], expected);
      ++failures;
    }
    total += counts[1];
  }
  CUDA_CHECK(cudaFree(d_counts));
  CUDA_CHECK(cudaFree(d_first));
  std::printf("mod_p exhaustive over the reachable epilogue range: %llu values, 20 moduli\n", total);
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
