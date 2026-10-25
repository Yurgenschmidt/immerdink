// GPU unit test of sequence F1 (issue 21, src/kernels/numerics.cuh finish_f1_try / crt_round): the CRT finish through a
// hardware u64 -> f64 conversion against the existing crt_finish + round_scaled (which the suite proves bit-identical to
// spec/oracle.py) on the SAME limbs and exponent, for every N = 2..20. Inputs: x in [0, P) of every magnitude (|z| = x or
// P - x), around the centering boundary H, 0, P - 1, all-ones and power-of-two magnitudes, exact ties and their
// neighbours, and exponents that put the leading bit near the subnormal and overflow edges of binary64 (so both the fast
// path and the fallback run). The count of fast results is printed and must be > 0 and < total for every N.
// Mutations (the Mut template parameter): each planted bug must produce mismatches, or the test checks nothing.
#undef NDEBUG
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

struct Case {
  uint32_t x[c::kMaxLimbs];
  int e;
};
struct Out {
  uint64_t ref, f1;
  int fast;
};

template <int N, int Mut>
__global__ void f1_kernel(const Case* cs, int n, Out* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  constexpr int L = c::kTables.L32[N];
  uint32_t x[L];
#pragma unroll
  for (int w = 0; w < L; ++w) x[w] = cs[t].x[w];
  uint32_t mag[L];
  bool neg;
  imd::kernels::crt_finish<L>(x, N, mag, &neg);
  const double ref = imd::kernels::round_scaled<L>(mag, neg, cs[t].e);
  double f1 = 0;
  const bool fast = imd::kernels::finish_f1_try<L, N, Mut>(x, cs[t].e, &f1);
  const double total = fast ? f1 : ref;  // the fallback is the reference itself
  out[t].ref = static_cast<uint64_t>(__double_as_longlong(ref));
  out[t].f1 = static_cast<uint64_t>(__double_as_longlong(total));
  out[t].fast = fast;
}

// ---- host: little-endian u32 limb integers ------------------------------------------------------------------------
struct Big {
  uint32_t w[c::kMaxLimbs + 1] = {0, 0, 0, 0, 0, 0};
};
int cmp(const Big& a, const Big& b) {
  for (int i = c::kMaxLimbs; i >= 0; --i)
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
  return 0;
}
Big sub(const Big& a, const Big& b) {  // a >= b
  Big r;
  int64_t br = 0;
  for (int i = 0; i <= c::kMaxLimbs; ++i) {
    const int64_t t = int64_t{a.w[i]} - b.w[i] - br;
    r.w[i] = static_cast<uint32_t>(t), br = t < 0;
  }
  return r;
}
Big add_small(const Big& a, int64_t k) {  // a + k, k small (either sign), result >= 0
  Big r = a;
  if (k >= 0) {
    uint64_t cy = static_cast<uint64_t>(k);
    for (int i = 0; i <= c::kMaxLimbs && cy; ++i) {
      const uint64_t t = uint64_t{r.w[i]} + cy;
      r.w[i] = static_cast<uint32_t>(t), cy = t >> 32;
    }
  } else {
    Big m;
    m.w[0] = static_cast<uint32_t>(-k);
    r = sub(a, m);
  }
  return r;
}
int bitlen(const Big& a) {
  for (int i = c::kMaxLimbs; i >= 0; --i)
    if (a.w[i]) return 32 * i + 32 - __builtin_clz(a.w[i]);
  return 0;
}
Big pow2(int b) {
  Big r;
  r.w[b / 32] = 1u << (b % 32);
  return r;
}
Big ones(int b) {  // 2^b - 1
  return add_small(pow2(b), -1);
}
Big shl(const Big& a, int s) {
  Big r;
  for (int i = c::kMaxLimbs; i >= 0; --i) {
    const int src = i - s / 32;
    if (src < 0) continue;
    uint64_t v = uint64_t{a.w[src]} << (s % 32);
    if (s % 32 && src > 0) v |= a.w[src - 1] >> (32 - s % 32);
    r.w[i] = static_cast<uint32_t>(v);
  }
  return r;
}
Big addb(const Big& a, const Big& b) {
  Big r;
  uint64_t cy = 0;
  for (int i = 0; i <= c::kMaxLimbs; ++i) {
    const uint64_t t = uint64_t{a.w[i]} + b.w[i] + cy;
    r.w[i] = static_cast<uint32_t>(t), cy = t >> 32;
  }
  return r;
}
uint64_t xs(uint64_t& x) {
  x ^= x << 13, x ^= x >> 7, x ^= x << 17;
  return x;
}
Big rand_bits(int b, uint64_t& rng) {  // random with bit length exactly b
  Big r;
  for (int i = 0; i <= (b - 1) / 32; ++i) r.w[i] = static_cast<uint32_t>(xs(rng));
  const int top = (b - 1) / 32, tb = (b - 1) % 32;
  r.w[top] &= tb == 31 ? 0xFFFFFFFFu : ((1u << (tb + 1)) - 1);
  r.w[top] |= 1u << tb;
  return r;
}

Big load(const uint32_t* a) {
  Big r;
  for (int i = 0; i < c::kMaxLimbs; ++i) r.w[i] = a[i];
  return r;
}

template <int N, int Mut>
long run_n(uint64_t seed, long* total, long* fast_total, bool verbose) {
  const Big P = load(c::kTables.P[N]), H = load(c::kTables.H[N]);
  const int Lb = bitlen(H);
  uint64_t rng = seed * 0x9E3779B97F4A7C15ull + N;
  std::vector<Case> cs;
  const auto push = [&](Big x, int e) {
    if (cmp(x, P) >= 0) x = sub(x, P);
    Case k{};
    for (int i = 0; i < c::kMaxLimbs; ++i) k.x[i] = x.w[i];
    k.e = e;
    cs.push_back(k);
  };
  const auto with_e = [&](const Big& x, uint64_t& r) {
    const Big mag = cmp(x, H) > 0 ? sub(P, x) : x;
    const int bl = bitlen(mag) ? bitlen(mag) : 1;
    static const int tg[] = {-1074, -1075, -1022, -1021, -1020, 0, 1022, 1023, 1024, 970, 971, 972, 1000, -1074 + 52, -1023, 1500, -1500};
    for (int k = 0; k < 4; ++k) {
      const int tgt = tg[xs(r) % (sizeof tg / sizeof tg[0])];
      static const int d[] = {0, 0, 0, -1, 1, -2, 2};
      int e = tgt - (bl - 1) + d[xs(r) % 7];
      if (xs(r) % 10 < 3) e = static_cast<int>(xs(r) % 140001) - 70000;
      push(x, e);
    }
    for (int e : {0, -52, -400, 300, -600, 200}) push(x, e);
  };
  std::vector<Big> xsv;
  Big zero;
  xsv.push_back(zero);
  for (int k = 0; k < 4; ++k) {
    xsv.push_back(add_small(zero, k));
    xsv.push_back(add_small(H, k));
    xsv.push_back(add_small(H, -k));
    xsv.push_back(add_small(P, -1 - k));
  }
  for (int i = 0; i < 1500; ++i) {
    const int b = 1 + static_cast<int>(xs(rng) % Lb);
    Big mag = rand_bits(b, rng);
    switch (xs(rng) % 6) {
      case 0: mag = ones(b); break;
      case 1: {  // an exact tie and its neighbours: m 2^k + half (+ -1, 0, 1), m of 53 bits
        const int k = b > 53 ? b - 53 : 0;
        if (k == 0) break;
        Big m = rand_bits(53, rng);
        mag = shl(m, k);
        Big half = pow2(k - 1);
        mag = addb(mag, half);
        mag = add_small(mag, static_cast<int64_t>(xs(rng) % 5) - 2);
        break;
      }
      case 2: mag = pow2(b - 1); break;
      case 3: mag = add_small(pow2(b - 1), static_cast<int64_t>(xs(rng) % 5) - 2); break;
      default: break;
    }
    if (cmp(mag, H) > 0) mag = H;
    if (bitlen(mag) == 0) mag = add_small(zero, 1);
    xsv.push_back(mag);
    xsv.push_back(sub(P, mag));
  }
  for (const Big& x : xsv) with_e(x, rng);
  // the typical e of a real call: |sigma_i + tau_j| in a few hundred
  for (int i = 0; i < 4000; ++i) {
    Big x = rand_bits(1 + static_cast<int>(xs(rng) % (bitlen(P))), rng);
    push(x, static_cast<int>(xs(rng) % 801) - 600);
  }
  const int n = static_cast<int>(cs.size());
  Case* d_cs;
  Out* d_out;
  CUDA_CHECK(cudaMalloc(&d_cs, sizeof(Case) * n));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(Out) * n));
  CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), sizeof(Case) * n, cudaMemcpyHostToDevice));
  f1_kernel<N, Mut><<<(n + 127) / 128, 128>>>(d_cs, n, d_out);
  CUDA_CHECK(cudaGetLastError());
  std::vector<Out> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(Out) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cs));
  CUDA_CHECK(cudaFree(d_out));
  long bad = 0, fast = 0;
  for (int i = 0; i < n; ++i) {
    fast += out[i].fast;
    if (out[i].ref != out[i].f1) {
      if (bad < 3 && verbose)
        std::printf("  N=%d case %d e=%d x0=%08x: ref %016llx f1 %016llx\n", N, i, cs[i].e, cs[i].x[0],
                    static_cast<unsigned long long>(out[i].ref), static_cast<unsigned long long>(out[i].f1));
      ++bad;
    }
  }
  *total += n, *fast_total += fast;
  if (verbose)
    std::printf("N=%2d: %d cases, %ld on the fast path (%.1f%%), %ld mismatches\n", N, n, fast, 100.0 * fast / n, bad);
  if (verbose && (fast == 0 || fast == n)) {
    std::printf("  N=%d: the fast path or the fallback was never taken\n", N);
    ++bad;
  }
  return bad;
}

template <int Mut, int N = c::kNMin>
long run_all(long* total, long* fast, bool verbose) {
  long bad = run_n<N, Mut>(7, total, fast, verbose);
  if constexpr (N < c::kNMax) bad += run_all<Mut, N + 1>(total, fast, verbose);
  return bad;
}

}  // namespace

int main() {
  long total = 0, fast = 0;
  long bad = run_all<0>(&total, &fast, true);
  std::printf("F1 finish: %ld cases, %ld on the fast path, %ld mismatches against crt_finish + round_scaled\n", total, fast, bad);
  int fails = bad != 0;
  // mutations: each must be caught
  const char* names[] = {"sticky dropped", "exponent +1", "upper bound 2047", "sign dropped", "lower bound allows field 0"};
  long mb[5];
  long t2 = 0, f2 = 0;
  mb[0] = run_all<1>(&t2, &f2, false);
  mb[1] = run_all<2>(&t2, &f2, false);
  mb[2] = run_all<3>(&t2, &f2, false);
  mb[3] = run_all<4>(&t2, &f2, false);
  mb[4] = run_all<5>(&t2, &f2, false);
  for (int i = 0; i < 5; ++i) {
    std::printf("mutation \"%s\": %s (%ld mismatches)\n", names[i], mb[i] ? "caught" : "NOT CAUGHT", mb[i]);
    fails += mb[i] == 0;
  }
  std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
