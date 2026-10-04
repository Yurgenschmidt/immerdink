// CUDA smoke test: the sm_89 toolchain builds and runs, the statically initialized
// __constant__ tables equal the host tables byte for byte, and the §3.5 epilogue
// reduction (imd::device::mod_p) is exact over EVERY reachable input of every modulus, and
// (A5) the 64-bit square_term_n equals the 128-bit square_term on 2^28 sampled inputs.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "device/constants.cuh"
#include "kernels/numerics.cuh"

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

// A5: square_term_n (two 64-bit words, normalized significand) equals the reference
// square_term (128-bit) for finite nonzero x and gamma >= ilogb(x): pseudo-random bit
// patterns (every exponent, an eighth subnormal, both signs) with gamma - ilogb(x) both small
// (0..71: every branch of the 64-bit shift) and anywhere up to 1023, plus fixed edge values.
__device__ __forceinline__ uint64_t xs64(uint64_t& x) {
  x ^= x << 13, x ^= x >> 7, x ^= x << 17;
  return x;
}
__global__ void square_term_check(uint64_t seed, int iters, unsigned long long* bad,
                                  unsigned long long* count, unsigned long long* first_bits,
                                  int* first_gamma) {
  using namespace imd::kernels;
  const uint64_t edges[] = {1, kFracMask, uint64_t{1} << 52, 0x7FEFFFFFFFFFFFFFull,
                            0x3FF0000000000000ull, 0x3FFFFFFFFFFFFFFFull, 0x0000000080000000ull,
                            0x000FFFFF00000001ull};
  const uint64_t tid = blockIdx.x * uint64_t{blockDim.x} + threadIdx.x;
  uint64_t x = seed ^ ((tid + 1) * 0x9E3779B97F4A7C15ull);
  unsigned long long nbad = 0, n = 0;
  for (int it = 0; it < iters; ++it) {
    uint64_t b = xs64(x);
    const uint64_t r = xs64(x);
    if (tid == 0 && it < 8 * 72) b = edges[it / 72] | (r & kSignBit);  // edges, d = 0..71
    else if ((r & 7) == 0) b &= ~(uint64_t{0x7FF} << 52);                // subnormal
    if (!finite_nonzero(b)) continue;
    const int il = ilogb_bits(b);
    int d = tid == 0 && it < 8 * 72 ? it % 72
            : (r >> 3) & 1 ? static_cast<int>((r >> 4) % 72) : static_cast<int>((r >> 4) % 2100);
    const int gamma = il + d > imd::constants::kGammaMax ? imd::constants::kGammaMax : il + d;
    const u128 want = square_term(b, gamma);
    const uint64_t got = square_term_n(b, gamma);
    if (want != static_cast<u128>(got)) {
      ++nbad;
      if (atomicCAS(first_gamma, 0x7fffffff, gamma) == 0x7fffffff) *first_bits = b;
    }
    ++n;
  }
  atomicAdd(bad, nbad);
  atomicAdd(count, n);
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
  std::printf("mod_p exhaustive over the reachable epilogue range: %llu values, 20 moduli\n", total);

  {  // A5: square_term_n == square_term
    unsigned long long* d_bits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, sizeof(unsigned long long)));
    const int first_init = 0x7fffffff;
    CUDA_CHECK(cudaMemset(d_counts, 0, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_first, &first_init, sizeof(int), cudaMemcpyHostToDevice));
    square_term_check<<<prop.multiProcessorCount * 8, 256>>>(0x5EED5EEDull, 1024, d_counts,
                                                             d_counts + 1, d_bits, d_first);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long counts[2], bits = 0;
    int gamma = 0;
    CUDA_CHECK(cudaMemcpy(counts, d_counts, sizeof counts, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&bits, d_bits, sizeof bits, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gamma, d_first, sizeof gamma, cudaMemcpyDeviceToHost));
    if (counts[0] != 0 || counts[1] < (1ull << 24)) {
      std::fprintf(stderr, "FAIL: square_term_n: %llu mismatches of %llu (first x=0x%016llx, "
                   "gamma=%d)\n", counts[0], counts[1], bits, gamma);
      ++failures;
    } else {
      std::printf("square_term_n == square_term on %llu (x, gamma) pairs\n", counts[1]);
    }
    CUDA_CHECK(cudaFree(d_bits));
  }
  CUDA_CHECK(cudaFree(d_counts));
  CUDA_CHECK(cudaFree(d_first));
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
