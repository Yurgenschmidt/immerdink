// GPU unit test of the pack emitter (DESIGN v1.14 §3.4.1, src/kernels/pack_emitter.cuh):
// PackEmitterT<E, W> for E = 16, 8, 4 and every W >= W(N): W = 1 (v1.16, N <= 8), W = 2
// (N <= 16), W = 3 (every N), RN and TRUNC,
// against an exact host reference: |a'| from the double's significand and exponent in
// u128 (§3.3: RN ties-to-even or truncation toward zero), r = |a'| mod p, SMod, the sign.
// Device asserts are ON (NDEBUG undefined below): X's bits past kPackXBits[W] must be zero.
//
// Inputs per emitter thread (a sigma and E doubles), with |a'| <= T(N) for every element
// (the §3.2 guarantee the emitter relies on; elements beyond it are zeroed on the host):
//   mode 0  subnormals m 2^-1074 at sigma = 1074 + eps0: the device's eps is eps0 for every
//           eps0 in 0..eps_max(N) (word selects q = 1, 2 at N = 17-20; every W = 2 shift up to
//           62 at N = 16), m of every bit length allowed by m 2^eps0 <= T, m = T >> eps0;
//   mode 1  normals with eps = e + sigma from -70 (rounding path, ties) up to
//           bitlen(T) - 53, the largest m 2^eps <= T among them;
//   mode 2  T's top 53 bits (the largest reachable a' <= T), a' = 2^53 - 1, and the tie
//           2^52 - 1/2 (RN rounds up to 2^52);
// with random signs, and zeros, -0, +-Inf and NaN in some lanes. Consecutive threads take
// consecutive modes, so every warp mixes all eps values (lane-divergent eps).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "device/constants.cuh"
#include "kernels/numerics.cuh"
#include "kernels/pack_emitter.cuh"

namespace {

namespace c = imd::constants;
__extension__ typedef unsigned __int128 u128;

#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    const cudaError_t err_ = (x);                                                  \
    if (err_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x,              \
                   cudaGetErrorString(err_));                                      \
      std::exit(2);                                                                \
    }                                                                              \
  } while (0)

template <int E, int W>
__global__ void emit_kernel(const uint64_t* bits, const int* sigma, int threads, int N, bool trunc,
                            int8_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= threads) return;
  uint64_t b[E];
#pragma unroll
  for (int e = 0; e < E; ++e) b[e] = bits[int64_t{t} * E + e];
  imd::kernels::PackEmitterT<E, W> em;
  em.load(b, sigma[t], trunc);
  const int64_t plane = int64_t{threads} * E;
  for (int l = 0; l < N; ++l) em.store(l, out + l * plane + int64_t{t} * E);
}

uint64_t xs(uint64_t& x) {
  x ^= x << 13, x ^= x >> 7, x ^= x << 17;
  return x;
}

u128 T_of(int N) {
  u128 t = 0;
  for (int w = c::kMaxLimbs - 1; w >= 0; --w) t = (t << 32) | c::kTables.T[N][w];
  return t;
}
int bitlen(u128 x) {
  int n = 0;
  while (x != 0) ++n, x >>= 1;
  return n;
}

// The device's decomposition (numerics.cuh decompose): |x| = m 2^e.
void decomp(uint64_t b, uint64_t* m, int* e) {
  const int E = static_cast<int>((b >> 52) & 0x7FF);
  const uint64_t f = b & ((uint64_t{1} << 52) - 1);
  if (E != 0) *m = f | (uint64_t{1} << 52), *e = E - 1075;
  else *m = f, *e = -1074;
}

// §3.3 exactly: |a'| (u128) and the device's eps (0 on the rounding path).
u128 ref_abs(uint64_t b, int sigma, bool trunc, int* eps) {
  *eps = 0;
  if (((b >> 52) & 0x7FF) == 0x7FF || (b << 1) == 0) return 0;  // non-finite, +-0
  uint64_t m;
  int e;
  decomp(b, &m, &e);
  const int s = e + sigma;
  if (s >= 0) {
    *eps = s;
    return s >= 120 ? ~u128{0} : static_cast<u128>(m) << s;
  }
  const int sh = -s;
  if (sh >= 64) return 0;  // m < 2^53: below half (RN) or zero (TRUNC)
  uint64_t r = m >> sh;
  if (!trunc) {
    const uint64_t rem = m & ((uint64_t{1} << sh) - 1), half = uint64_t{1} << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
  }
  return r;
}

int8_t ref_res(uint64_t b, u128 a, uint32_t p) {
  const uint32_t r = static_cast<uint32_t>(a % p);
  int32_t u = r >= (p + 1) / 2 ? static_cast<int32_t>(r) - static_cast<int32_t>(p)
                               : static_cast<int32_t>(r);
  if (b >> 63) u = -u;
  return static_cast<int8_t>(static_cast<uint8_t>(u & 255));
}

uint64_t dbl_bits(double x) {
  uint64_t b;
  std::memcpy(&b, &x, 8);
  return b;
}
// m 2^k as a double (exact: m < 2^53 and the result in range), sign s.
uint64_t make(uint64_t m, int k, bool s) {
  const double v = std::ldexp(static_cast<double>(m), k);
  return dbl_bits(s ? -v : v);
}

struct Coverage {
  std::vector<long> eps;  // device eps histogram of nonzero elements
  long q1 = 0, q2 = 0, rounding = 0, ties = 0, top = 0;
};

// One emitter thread's inputs for N; every element has |a'| <= T.
void gen_thread(int N, int mode, int E, bool trunc, uint64_t& rng, int* sigma,
                std::vector<uint64_t>& bits, Coverage& cov, int eps_cycle) {
  const u128 T = T_of(N);
  const int eps_max = c::kTables.eps_max[N], tb = bitlen(T);
  std::vector<uint64_t> el(E);
  if (mode == 0) {  // subnormals: device eps = eps0 exactly
    const int eps0 = eps_cycle % (eps_max + 1);
    *sigma = 1074 + eps0;
    const u128 mcap128 = T >> eps0;  // m 2^eps0 <= T
    const uint64_t mcap = mcap128 > ((u128{1} << 52) - 1) ? (uint64_t{1} << 52) - 1
                                                           : static_cast<uint64_t>(mcap128);
    for (int e = 0; e < E; ++e) {
      uint64_t m;
      const uint64_t r = xs(rng);
      if (e == 0) m = mcap;               // the largest allowed
      else if (e == 1) m = 1;
      else {
        const int L = 1 + static_cast<int>(r % static_cast<uint64_t>(bitlen(mcap)));
        m = (xs(rng) & ((uint64_t{1} << L) - 1)) | (uint64_t{1} << (L - 1));
        if (m > mcap) m = mcap;
      }
      el[e] = m == 0 ? 0 : (m | ((r >> 63) << 63));  // a subnormal bit pattern, random sign
    }
  } else if (mode == 1) {  // normals, eps from -70 to tb - 53
    *sigma = static_cast<int>(xs(rng) % 401) - 200;
    for (int e = 0; e < E; ++e) {
      const uint64_t r = xs(rng);
      const int hi = tb - 53;  // may be negative (small N): then only the rounding path
      int s = hi - static_cast<int>(r % 74);
      uint64_t m = (xs(rng) & ((uint64_t{1} << 52) - 1)) | (uint64_t{1} << 52);
      if ((r >> 20) % 5 == 0) m |= 1;                                 // odd
      if ((r >> 24) % 7 == 0 && s < 0 && s > -50)                     // an exact tie
        m = (m >> (-s)) << (-s) | (uint64_t{1} << (-s - 1));
      if (s >= 0 && (static_cast<u128>(m) << s) > T) m = static_cast<uint64_t>(T >> s);
      if (s >= 0 && m < (uint64_t{1} << 52)) s = -1;                  // keep m normal
      el[e] = make(m, s - *sigma, (r >> 62) & 1);
    }
  } else {  // T's top 53 bits; 2^53 - 1 and the tie that rounds to 2^53
    *sigma = 1000;
    for (int e = 0; e < E; ++e) {
      const uint64_t r = xs(rng);
      const bool sg = (r >> 61) & 1;
      if (e % 3 == 0 && tb >= 53) {
        const int s = tb - 53;
        el[e] = make(static_cast<uint64_t>(T >> s), s - *sigma, sg);
      } else if (e % 3 == 1) {
        el[e] = make((uint64_t{1} << 53) - 1, -*sigma, sg);              // a' = 2^53 - 1
      } else {
        // x 2^sigma = 2^52 - 1/2: a tie on the rounding path, RN to even 2^52 (a carry into
        // a new bit), TRUNC 2^52 - 1
        el[e] = make((uint64_t{1} << 53) - 1, -1 - *sigma, sg);
      }
    }
  }
  for (int e = 0; e < E; ++e) {  // specials in some lanes
    const uint64_t r = xs(rng);
    if (r % 23 == 0) {
      const uint64_t sp[] = {0, uint64_t{1} << 63, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
                             0x7FF8000000000001ull, 0xFFF8000000000000ull};
      el[e] = sp[(r >> 8) % 6];
    }
    int eps;
    const u128 a = ref_abs(el[e], *sigma, trunc, &eps);
    if (a > T) {  // out of the §3.2 contract (cannot happen for modes 0, 2): a zero instead
      el[e] = 0;
      bits.push_back(0);
      continue;
    }
    if (a != 0) {
      ++cov.eps[eps];
      cov.q1 += (eps >> 5) == 1, cov.q2 += (eps >> 5) == 2;
      cov.top += bitlen(a) == tb;
    }
    uint64_t m;
    int ee;
    decomp(el[e], &m, &ee);
    if (a != 0 && ee + *sigma < 0) ++cov.rounding;
    if (ee + *sigma < 0 && ee + *sigma > -64 && (m & ((uint64_t{1} << -(ee + *sigma)) - 1)) ==
                                                     (uint64_t{1} << (-(ee + *sigma) - 1)))
      ++cov.ties;
    bits.push_back(el[e]);
  }
}

template <int E, int W>
long run(int N, bool trunc, int threads, uint64_t seed, Coverage& cov, long* checked) {
  uint64_t rng = seed;
  std::vector<uint64_t> bits;
  std::vector<int> sig(threads);
  bits.reserve(static_cast<size_t>(threads) * E);
  for (int t = 0; t < threads; ++t)
    gen_thread(N, t % 3, E, trunc, rng, &sig[t], bits, cov, t / 3);
  uint64_t* d_bits;
  int* d_sig;
  int8_t* d_out;
  const size_t plane = static_cast<size_t>(threads) * E;
  CUDA_CHECK(cudaMalloc(&d_bits, bits.size() * 8));
  CUDA_CHECK(cudaMalloc(&d_sig, sig.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, plane * N));
  CUDA_CHECK(cudaMemcpy(d_bits, bits.data(), bits.size() * 8, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sig, sig.data(), sig.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0x5A, plane * N));
  emit_kernel<E, W><<<(threads + 127) / 128, 128>>>(d_bits, d_sig, threads, N, trunc, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());  // a device assert surfaces here
  std::vector<int8_t> out(plane * N);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_bits));
  CUDA_CHECK(cudaFree(d_sig));
  CUDA_CHECK(cudaFree(d_out));
  long bad = 0;
  for (int t = 0; t < threads; ++t)
    for (int e = 0; e < E; ++e) {
      const uint64_t b = bits[static_cast<size_t>(t) * E + e];
      int eps;
      const u128 a = ref_abs(b, sig[t], trunc, &eps);
      for (int l = 0; l < N; ++l) {
        const int8_t want = ref_res(b, a, c::kTables.moduli[l]);
        const int8_t got = out[l * plane + static_cast<size_t>(t) * E + e];
        if (got != want) {
          if (bad < 5)
            std::fprintf(stderr, "  N=%d E=%d W=%d %s t=%d e=%d l=%d x=0x%016llx sigma=%d eps=%d: "
                         "got %d want %d\n", N, E, W, trunc ? "TRUNC" : "RN", t, e, l,
                         static_cast<unsigned long long>(b), sig[t], eps, got, want);
          ++bad;
        }
        ++*checked;
      }
    }
  return bad;
}

template <int E>
long run_n(int N, bool trunc, int threads, uint64_t seed, Coverage* cov, long* checked) {
  // cov[W - 1]: the run with W words, for every W >= W(N) (a larger W is valid too: W = 3 at
  // every N, W = 2 at N <= 16, W = 1 at N <= 8); coverage is checked per W.
  long bad = run<E, 3>(N, trunc, threads, seed, cov[2], checked);
  if (imd::kernels::pack_words(N) <= 2)
    bad += run<E, 2>(N, trunc, threads, seed + 1, cov[1], checked);
  if (imd::kernels::pack_words(N) == 1)
    bad += run<E, 1>(N, trunc, threads, seed + 2, cov[0], checked);
  return bad;
}

}  // namespace

int main() {
  static_assert(imd::kernels::pack_words(8) == 1 && imd::kernels::pack_words(9) == 2 &&
                    imd::kernels::pack_words(16) == 2 && imd::kernels::pack_words(17) == 3,
                "W rule (cert_pack_words)");
  long bad = 0, checked = 0;
  int fails = 0;
  for (int N = c::kNMin; N <= c::kNMax; ++N) {
    Coverage covw[3];  // [W - 1] (review L1: "all eps hit" per W)
    for (Coverage& cv : covw) cv.eps.assign(c::kEpsMax + 1, 0);
    const int w_min = imd::kernels::pack_words(N);
    long bn = 0;
    for (int tr = 0; tr < 2; ++tr) {
      const bool trunc = tr == 1;
      const uint64_t seed = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(4 * N + tr + 1);
      bn += run_n<16>(N, trunc, 3 * 1024, seed, covw, &checked);
      bn += run_n<8>(N, trunc, 3 * 512, seed ^ 0x1234, covw, &checked);
      bn += run_n<4>(N, trunc, 3 * 512, seed ^ 0x5678, covw, &checked);
    }
    int missing = 0;
    bool cov_ok = true;
    for (int w = w_min - 1; w < 3; ++w) {
      const Coverage& cv = covw[w];
      int miss_w = 0;
      for (int e = 0; e <= c::kTables.eps_max[N]; ++e) miss_w += cv.eps[e] == 0;
      missing += miss_w;
      cov_ok = cov_ok && miss_w == 0 && cv.top > 0 && cv.rounding > 0 && cv.ties > 0 &&
               (N < 9 || cv.q1 > 0) && (N < 17 || cv.q2 > 0);
    }
    Coverage cov = covw[2];  // the report: every W summed
    for (int w = w_min - 1; w < 2; ++w) {
      cov.q1 += covw[w].q1, cov.q2 += covw[w].q2, cov.top += covw[w].top;
      cov.rounding += covw[w].rounding, cov.ties += covw[w].ties;
    }
    std::printf("N=%2d W=%d..3: %ld mismatches; eps 0..%d all hit: %s (q=1: %ld, q=2: %ld, |a'| at "
                "bitlen T: %ld, rounding path: %ld, ties: %ld)\n",
                N, imd::kernels::pack_words(N), bn, c::kTables.eps_max[N], missing ? "NO" : "yes",
                cov.q1, cov.q2, cov.top, cov.rounding, cov.ties);
    bad += bn;
    fails += bn != 0 || !cov_ok;
  }
  std::printf("pack emitter: %ld residues checked, %ld mismatches\n%s\n", checked, bad,
              fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
