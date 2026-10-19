// Memory-bandwidth microbenchmark for the memory-bound kernels of immerdink (issue 21, docs/NOTES-21.md): how much bandwidth can this
// GPU's DRAM and L2 deliver to kernels shaped like the pack (KL / K2 / K3: read FP64 rows, write N narrow planes) and the CRT (K5: read N
// narrow planes, write FP64), as a function of the launch shape (blocks per SM, threads per block, loads in flight per thread)?
//
// Why: on the RTX 4090 the pack and CRT kernels run at 85-87% of the DRAM peak (ncu, docs/NOTES-21.md): bandwidth-bound. On H100 and
// H200 they take the same ~1.2 ms at 4096^3 although H200 has 43% more bandwidth, so something else limits them there. This program
// separates "the memory system cannot deliver more" from "this kernel shape cannot ask for more": the best read / write / copy / fan-out /
// fan-in bandwidth over all launch shapes, and the bandwidth of the shape the library uses. Compare with the library's phase times
// (imd_bench phases) offline.
//
// Patterns (all on buffers far larger than L2 unless noted):
//   rd      read  uint4 vectors (16 B), xor-reduce            wr   write uint4 vectors            cp   read uint4, write uint4
//   fan     read 16 B (two FP64), write 2 bytes into each of F = 14 planes (the pack's shape)
//   fanin   read 2 bytes from each of F = 14 planes, write 16 B (the CRT's shape)
//   fan16   read 8 x 16 B (16 FP64), write 16 B into each of F planes (the library's shape: wide stores per plane); ILP 1, 2, 4
//   fanin16 read 16 B from each of F planes, write 8 x 16 B (the CRT with wide loads); ILP 1, 2, 4
//   rd_L2   read of a buffer of L2/4, L2/2, 2 L2, 8 L2 (bandwidth against working-set size: the L2 cliff)
// Sweep: blocks per SM {1, 2, 4, 8, 16} x threads {128, 256, 512, 1024} x vectors in flight per thread {1, 2, 4, 8}.
// Output: one JSON line per (pattern, shape): bytes moved, time (median of the repetitions), GB/s, % of the datasheet-derived peak
// (memory clock x bus width x 2), plus a line with the device (name, SMs, L2, peak). Exit 0.
//   imd_membw [--out FILE] [--mib 0] [--reps 10] [--quick]   (--mib 0: the larger of 512 and 9 x L2, so the 8 x L2 point exists; FILE is truncated)
// Build: nvcc -O3 -std=c++17 -arch=sm_XX bench/membw/imd_membw.cu -o imd_membw (tools/build_membw.sh).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                                                \
  do {                                                                                       \
    const cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                                 \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                          \
    }                                                                                        \
  } while (0)

constexpr int kF = 14;  // planes of the fan-out / fan-in patterns (N = 14 moduli)

// ----------------------------------------------------------------------------------------------- kernels
// Each thread handles kIlp independent vectors per loop iteration (loads issued back to back: kIlp loads in flight).
template <bool kStream>
__device__ __forceinline__ uint4 ld(const uint4* p) {
  return kStream ? __ldcs(p) : __ldg(p);   // streaming (evict-first) for the DRAM patterns; ordinary for the L2-residency test
}
template <int kIlp, bool kStream = true>
__global__ void k_rd(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride * kIlp) {
    uint4 v[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) v[j] = (i + j * stride < n) ? ld<kStream>(p + i + j * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < kIlp; ++j) acc.x ^= v[j].x, acc.y ^= v[j].y, acc.z ^= v[j].z, acc.w ^= v[j].w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B1u && threadIdx.x == 12345) *sink = 1;  // keeps the loads alive
}
template <int kIlp>
__global__ void k_wr(uint4* __restrict__ p, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride * kIlp) {
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
      if (i + j * stride < n) __stcs(p + i + j * stride, make_uint4(unsigned(i), 1u, 2u, 3u));
  }
}
template <int kIlp>
__global__ void k_cp(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride * kIlp) {
    uint4 v[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) v[j] = (i + j * stride < n) ? __ldcs(a + i + j * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
      if (i + j * stride < n) __stcs(b + i + j * stride, v[j]);
  }
}
// fan-out: element pair e (16 B) -> 2 bytes (ushort) at planes[f][e], f = 0..kF-1
template <int kIlp>
__global__ void k_fan(const uint4* __restrict__ a, ushort* __restrict__ planes, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride * kIlp) {
    uint4 v[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) v[j] = (i + j * stride < n) ? __ldcs(a + i + j * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
      if (i + j * stride < n)
#pragma unroll
        for (int f = 0; f < kF; ++f) planes[size_t(f) * n + i + j * stride] = ushort((v[j].x >> f) ^ (v[j].y << f));
  }
}
// fan-in: 2 bytes from each plane -> 16 B
template <int kIlp>
__global__ void k_fanin(const ushort* __restrict__ planes, uint4* __restrict__ out, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride * kIlp) {
#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
      if (i + j * stride >= n) continue;
      unsigned acc = 0;
#pragma unroll
      for (int f = 0; f < kF; ++f) acc += unsigned(__ldcs(planes + size_t(f) * n + i + j * stride)) << (f & 7);
      __stcs(out + i + j * stride, make_uint4(acc, acc >> 1, acc >> 2, acc >> 3));
    }
  }
}

// wide fan-out: a group of 8 vectors (16 FP64) -> one 16 B vector in each of kF planes (planes hold n / 8 vectors each: the same bytes as k_fan)
template <int kIlp>
__global__ void k_fan16(const uint4* __restrict__ a, uint4* __restrict__ planes, size_t n) {
  const size_t ng = n / 8, stride = size_t(gridDim.x) * blockDim.x;
  for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < ng; g += stride * kIlp) {
    uint4 v[kIlp][8];
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) v[j][e] = (g + j * stride < ng) ? __ldcs(a + (g + j * stride) * 8 + e) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int j = 0; j < kIlp; ++j)
      if (g + j * stride < ng)
#pragma unroll
        for (int f = 0; f < kF; ++f) {
          uint4 o = make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int e = 0; e < 8; ++e) o.x ^= (v[j][e].x >> f) ^ (v[j][e].y << f), o.y ^= v[j][e].z >> f, o.z ^= v[j][e].w << f, o.w ^= o.x + e;
          __stcs(planes + size_t(f) * ng + g + j * stride, o);
        }
  }
}
// wide fan-in: one 16 B vector from each of kF planes -> a group of 8 vectors
template <int kIlp>
__global__ void k_fanin16(const uint4* __restrict__ planes, uint4* __restrict__ out, size_t n) {
  const size_t ng = n / 8, stride = size_t(gridDim.x) * blockDim.x;
  for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < ng; g += stride * kIlp) {
#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
      if (g + j * stride >= ng) continue;
      uint4 acc = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int f = 0; f < kF; ++f) {
        const uint4 v = __ldcs(planes + size_t(f) * ng + g + j * stride);
        acc.x += v.x << (f & 7), acc.y += v.y << (f & 7), acc.z += v.z << (f & 7), acc.w += v.w << (f & 7);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) __stcs(out + (g + j * stride) * 8 + e, make_uint4(acc.x + e, acc.y, acc.z, acc.w));
    }
  }
}

// ----------------------------------------------------------------------------------------------- driver
struct Dev {
  std::string name;
  int sms = 0;
  long long l2 = 0;
  double peak_gbs = 0;
};

std::FILE* g_out = stdout;
Dev dev;
int g_reps = 10;

template <class Launch>
double time_ms(Launch&& launch) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<float> t;
  launch();  // warm-up
  if (cudaGetLastError() != cudaSuccess) return -1.0;   // launch configuration refused (e.g. too many registers for this block size): no line
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < g_reps; ++r) {
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    t.push_back(ms);
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

void emit(const char* pattern, int bps, int threads, int ilp, double bytes, double ms, double footprint) {
  if (ms <= 0) return;
  const double gbs = bytes / (ms * 1e6);
  std::fprintf(g_out,
               "{\"kind\": \"membw\", \"pattern\": \"%s\", \"blocks_per_sm\": %d, \"threads\": %d, \"ilp\": %d, \"bytes\": %.0f, "
               "\"footprint_bytes\": %.0f, \"ms\": %.6f, \"gbs\": %.2f, \"pct_of_peak\": %.2f, \"gpu\": \"%s\"}\n",
               pattern, bps, threads, ilp, bytes, footprint, ms, gbs, 100.0 * gbs / dev.peak_gbs, dev.name.c_str());
  std::fflush(g_out);
}

template <int kIlp>
void sweep_one(int bps, int threads, size_t nvec, uint4* A, uint4* B, ushort* P, unsigned* sink, bool l2_only) {
  const int blocks = bps * dev.sms;
  const double fv = double(nvec) * 16;
  if (l2_only) {
    const double ms = time_ms([&] { k_rd<kIlp, false><<<blocks, threads>>>(A, nvec, sink); });
    emit("rd_small", bps, threads, kIlp, fv, ms, fv);
    return;
  }
  double ms = time_ms([&] { k_rd<kIlp><<<blocks, threads>>>(A, nvec, sink); });
  emit("rd", bps, threads, kIlp, fv, ms, fv);
  ms = time_ms([&] { k_wr<kIlp><<<blocks, threads>>>(B, nvec); });
  emit("wr", bps, threads, kIlp, fv, ms, fv);
  ms = time_ms([&] { k_cp<kIlp><<<blocks, threads>>>(A, B, nvec); });
  emit("cp", bps, threads, kIlp, 2 * fv, ms, 2 * fv);
  const double fan_bytes = fv + double(kF) * nvec * 2;
  ms = time_ms([&] { k_fan<kIlp><<<blocks, threads>>>(A, P, nvec); });
  emit("fan", bps, threads, kIlp, fan_bytes, ms, fan_bytes);
  ms = time_ms([&] { k_fanin<kIlp><<<blocks, threads>>>(P, B, nvec); });
  emit("fanin", bps, threads, kIlp, fan_bytes, ms, fan_bytes);
  if constexpr (kIlp <= 4) {   // 8 x kIlp live vectors per thread: ILP 8 would spill
    ms = time_ms([&] { k_fan16<kIlp><<<blocks, threads>>>(A, reinterpret_cast<uint4*>(P), nvec); });
    emit("fan16", bps, threads, kIlp, fan_bytes, ms, fan_bytes);
    ms = time_ms([&] { k_fanin16<kIlp><<<blocks, threads>>>(reinterpret_cast<const uint4*>(P), B, nvec); });
    emit("fanin16", bps, threads, kIlp, fan_bytes, ms, fan_bytes);
  }
}

void sweep(int bps, int threads, int ilp, size_t nvec, uint4* A, uint4* B, ushort* P, unsigned* sink, bool l2_only) {
  switch (ilp) {
    case 1: sweep_one<1>(bps, threads, nvec, A, B, P, sink, l2_only); break;
    case 2: sweep_one<2>(bps, threads, nvec, A, B, P, sink, l2_only); break;
    case 4: sweep_one<4>(bps, threads, nvec, A, B, P, sink, l2_only); break;
    default: sweep_one<8>(bps, threads, nvec, A, B, P, sink, l2_only); break;
  }
}

int main(int argc, char** argv) {
  size_t mib = 0;
  bool quick = false;
  const char* out = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--mib") && i + 1 < argc) mib = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--quick")) quick = true;
    else { std::fprintf(stderr, "usage: imd_membw [--out FILE] [--mib 0] [--reps 10] [--quick]\n"); return 2; }
  }
  if (out) g_out = std::fopen(out, "w");
  if (!g_out) { std::perror("open"); return 2; }
  cudaDeviceProp p{};
  int d = 0, mclk = 0, bus = 0;
  CK(cudaGetDevice(&d));
  CK(cudaGetDeviceProperties(&p, d));
  CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, d));
  CK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, d));
  dev.name = p.name, dev.sms = p.multiProcessorCount, dev.l2 = p.l2CacheSize;
  // kHz x 2 (DDR) x bytes per transfer. The factor 2 is the DDR rate; GDDR6X / GDDR7 signal differently, so on those cards pct_of_peak is
  // only as good as the attribute: the analysis leans on the measured read bandwidth (pct_of_rd), not on this figure.
  dev.peak_gbs = double(mclk) * 1e3 * 2.0 * (double(bus) / 8.0) / 1e9;
  if (!(dev.peak_gbs > 0)) dev.peak_gbs = 1e9;   // attribute not reported: percentages become ~0 instead of dividing by zero
  if (mib == 0) mib = std::max<size_t>(512, size_t(9.0 * double(dev.l2) / 1048576.0) + 1);
  std::fprintf(g_out, "{\"kind\": \"membw_device\", \"gpu\": \"%s\", \"sm_count\": %d, \"l2_bytes\": %lld, \"mem_clock_khz\": %d, \"bus_bits\": %d, \"peak_gbs\": %.1f}\n",
               p.name, dev.sms, dev.l2, mclk, bus, dev.peak_gbs);
  const size_t nvec = mib * 1048576ull / 16;
  uint4 *A, *B;
  ushort* P;
  unsigned* sink;
  CK(cudaMalloc(&A, nvec * 16));
  CK(cudaMalloc(&B, nvec * 16));
  CK(cudaMalloc(&P, size_t(kF) * nvec * 2));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(A, 0x5A, nvec * 16));
  CK(cudaMemset(B, 0, nvec * 16));
  CK(cudaMemset(P, 0x11, size_t(kF) * nvec * 2));
  const std::vector<int> bpss = quick ? std::vector<int>{2, 8} : std::vector<int>{1, 2, 4, 8, 16};
  const std::vector<int> ths = quick ? std::vector<int>{256, 1024} : std::vector<int>{128, 256, 512, 1024};
  const std::vector<int> ilps = quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 4, 8};
  for (int bps : bpss)
    for (int th : ths) {
      if (bps * th > 2048 * 2) continue;     // no more than two waves of threads per SM
      for (int ilp : ilps) sweep(bps, th, ilp, nvec, A, B, P, sink, false);
    }
  // bandwidth against working-set size (best shape of the sweep is not known here: a fixed good one)
  for (double f : {0.25, 0.5, 0.75, 1.0, 2.0, 8.0}) {
    const size_t nv = size_t(double(dev.l2) * f / 16);
    if (nv > nvec) continue;
    sweep(8, 512, 4, nv, A, B, P, sink, true);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}
