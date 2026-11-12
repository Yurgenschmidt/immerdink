// GPU unit test of the pack emitter (DESIGN v1.14 §3.4.1, src/kernels/pack_emitter.cuh):
// PackEmitterT<E, W> for E = 16, 8, 4 and W = 2 (N <= 16) / W = 3 (every N), RN and TRUNC,
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
#include <string>
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

template <int E, int W, int S, int Mut, int StMut>
__global__ void emit_kernel(const uint64_t* bits, const int* sigma, int threads, int N, bool trunc,
                            int8_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= threads) return;
  uint64_t b[E];
#pragma unroll
  for (int e = 0; e < E; ++e) b[e] = bits[int64_t{t} * E + e];
  imd::kernels::PackEmitterT<E, W, S, Mut> em;
  em.load(b, sigma[t], trunc);
  const int64_t plane = int64_t{threads} * E;
  // store_slots (the launches' addressing) in two parts, so that a start slot other than 0 is exercised too
  const int split = N / 3;
  imd::kernels::store_slots<decltype(em), StMut>(em, 0, split, N, out + int64_t{t} * E, plane);
  imd::kernels::store_slots<decltype(em), StMut>(em, 0, 0, split, out + int64_t{t} * E, plane);
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
  } else if (mode == 3) {  // issue 21: the certified sigma range (Q1's two power-of-two factors), any value the contract allows
    static const int sg[] = {c::kSigmaMin, c::kSigmaMin + 1, -1000, -513, -512, -2, -1, 0, 1, 2, 511, 512, 1000, 1074, c::kSigmaMax - 1, c::kSigmaMax};
    *sigma = sg[xs(rng) % (sizeof sg / sizeof sg[0])];
    for (int e = 0; e < E; ++e) {
      const uint64_t r = xs(rng);
      const int te = static_cast<int>(r % static_cast<uint64_t>(tb + 4)) - 3;  // target bit position of |a'|
      const int ex = te - *sigma;                                          // exponent of the double's top bit
      uint64_t m = (xs(rng) & ((uint64_t{1} << 52) - 1)) | (uint64_t{1} << 52);
      if ((r >> 20) % 4 == 0) m = (uint64_t{1} << 53) - 1;
      if (ex < -1022) {                                                    // a subnormal (or 0 below the grid)
        const int sh = -1022 - ex;
        const uint64_t f = sh > 52 ? 0 : m >> sh;
        el[e] = f == 0 ? 0 : (f | (((r >> 62) & 1) << 63));
      }
      else if (ex > 1023) el[e] = 0x7FEFFFFFFFFFFFFFull;                   // the largest double
      else el[e] = make(m, ex - 52, (r >> 62) & 1);
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

bool g_quiet = false;  // the mutation checks expect mismatches
Coverage g_scratch;  // coverage of the non-baseline sequences is not asserted (their inputs are the baseline's)

template <int E, int W, int S, int Mut = 0, int StMut = 0>
long run(int N, bool trunc, int threads, uint64_t seed, Coverage& cov, long* checked) {
  uint64_t rng = seed;
  std::vector<uint64_t> bits;
  std::vector<int> sig(threads);
  bits.reserve(static_cast<size_t>(threads) * E);
  for (int t = 0; t < threads; ++t)
    gen_thread(N, t % 4, E, trunc, rng, &sig[t], bits, cov, t / 4);
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
  emit_kernel<E, W, S, Mut, StMut><<<(threads + 127) / 128, 128>>>(d_bits, d_sig, threads, N, trunc, d_out);
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
          if (bad < 5 && !g_quiet)
            std::fprintf(stderr, "  N=%d E=%d W=%d S=%d %s t=%d e=%d l=%d x=0x%016llx sigma=%d eps=%d: "
                         "got %d want %d\n", N, E, W, S, trunc ? "TRUNC" : "RN", t, e, l,
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
  // cov[W - 1]: the run with W words, for every W >= W(N) (W = 3 is valid at every N, W = 2
  // at N <= 16); coverage is checked per W (baseline sequence). Issue 21: the same inputs through the
  // other instruction sequences (S = 1 the E1 emitter, S = 3 E1 + Q1 quantize, W = 2 only).
  long bad = run<E, 3, 0>(N, trunc, threads, seed, cov[2], checked);
  bad += run<E, 3, 1>(N, trunc, threads, seed, g_scratch, checked);
  if (imd::kernels::pack_words(N) == 2) {
    bad += run<E, 2, 0>(N, trunc, threads, seed + 1, cov[1], checked);
    bad += run<E, 2, 1>(N, trunc, threads, seed + 1, g_scratch, checked);
    bad += run<E, 2, 3>(N, trunc, threads, seed + 1, g_scratch, checked);
    bad += run<E, 2, 5>(N, trunc, threads, seed + 1, g_scratch, checked);  // O1 (task 17): offset-binary input, integer quantize
    bad += run<E, 2, 7>(N, trunc, threads, seed + 1, g_scratch, checked);  // O1 + Q1: the signed F2I
  }
  return bad;
}


// ---- Issue 21, sequence E1: the fused emitter on EVERY reduced value d = sum_j dp4a(x_j, w_j) --------------------------
// pack_smod (modp.hpp), the function PackEmitterT<.., S & 1> runs, on the real __constant__ table c_pack_emit, for every
// d in [0, dmax(W, l)], both signs, every modulus and both word counts (11.4 M values), against SMod((sigma d) mod p).
// dmax comes from the weights computed here by repeated doubling, not from the table under test.
template <int W>
__global__ void d_kernel(int l, uint32_t dmax, int8_t* out) {
  const imd::device::PackEmit& c = imd::device::c_pack_emit.e[W - 2][l];
  for (uint32_t d = blockIdx.x * blockDim.x + threadIdx.x; d <= dmax; d += gridDim.x * blockDim.x) {
    out[d] = static_cast<int8_t>(imd::device::pack_smod(d, 1u, c));
    out[dmax + 1 + d] = static_cast<int8_t>(imd::device::pack_smod(d, 0xFFFFFFFFu, c));
  }
}

template <int W>
long check_d(int l, int8_t* d_out, std::vector<int8_t>& h) {
  const uint32_t p = c::kTables.moduli[l];
  uint32_t pw = 1 % p, dmax = 0;
  for (int i = 0; i < 4 * W; ++i) {
    if (!(W == 3 && i >= 10)) dmax += 255u * pw;
    for (int k = 0; k < 8; ++k) pw = (2 * pw) % p;
  }
  d_kernel<W><<<256, 256>>>(l, dmax, d_out);
  CUDA_CHECK(cudaGetLastError());
  h.resize(2 * (static_cast<size_t>(dmax) + 1));
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, h.size(), cudaMemcpyDeviceToHost));
  long bad = 0;
  const int32_t half = static_cast<int32_t>(p / 2);
  for (uint32_t d = 0; d <= dmax; ++d)
    for (int sg = 0; sg < 2; ++sg) {
      const int64_t v = sg == 0 ? static_cast<int64_t>(d) : -static_cast<int64_t>(d);
      int32_t r = static_cast<int32_t>(((v % static_cast<int64_t>(p)) + p) % p);
      int32_t u = r >= (static_cast<int32_t>(p) + 1) / 2 ? r - static_cast<int32_t>(p) : r;  // SMod
      (void)half;
      if (h[sg * (static_cast<size_t>(dmax) + 1) + d] != static_cast<int8_t>(u)) ++bad;
    }
  return bad;
}

// O1 (task 17): pack_smod_o on every start-shifted dot value s' = C2 + d, d in [0, dmax(2, l)], on the real table c_pack_emit_o,
// against SMod((d - 2^63) mod p): d is the dot of Y = a' + 2^63, so this also checks C2 = -2^63 mod p.
__global__ void d_kernel_o(int l, uint32_t dmax, int8_t* out) {
  const imd::device::PackEmit& c = imd::device::c_pack_emit_o.e[l];
  for (uint32_t d = blockIdx.x * blockDim.x + threadIdx.x; d <= dmax; d += gridDim.x * blockDim.x)
    out[d] = static_cast<int8_t>(imd::device::pack_smod_o(d + c.C, c));
}

long check_d_o(int l, int8_t* d_out, std::vector<int8_t>& h) {
  const uint32_t p = c::kTables.moduli[l];
  uint32_t pw = 1 % p, dmax = 0, p63 = 1 % p;
  for (int i = 0; i < 63; ++i) p63 = (2 * p63) % p;
  for (int i = 0; i < 8; ++i) {
    dmax += 255u * pw;
    for (int k = 0; k < 8; ++k) pw = (2 * pw) % p;
  }
  d_kernel_o<<<256, 256>>>(l, dmax, d_out);
  CUDA_CHECK(cudaGetLastError());
  h.resize(static_cast<size_t>(dmax) + 1);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, h.size(), cudaMemcpyDeviceToHost));
  long bad = 0;
  for (uint32_t d = 0; d <= dmax; ++d) {
    const int32_t r = static_cast<int32_t>((d % p + p - p63) % p);
    const int32_t u = r >= (static_cast<int32_t>(p) + 1) / 2 ? r - static_cast<int32_t>(p) : r;  // SMod
    if (h[d] != static_cast<int8_t>(u)) ++bad;
  }
  return bad;
}

long check_all_d(long* total) {
  int8_t* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 2 * 12 * 255 * 255 + 64));
  std::vector<int8_t> h;
  long bad = 0;
  for (int l = 0; l < c::kNMax; ++l) {
    bad += check_d<3>(l, d_out, h), *total += static_cast<long>(h.size());
    if (l < 16) bad += check_d<2>(l, d_out, h), *total += static_cast<long>(h.size());
    if (l < 16) bad += check_d_o(l, d_out, h), *total += static_cast<long>(h.size());
  }
  CUDA_CHECK(cudaFree(d_out));
  return bad;
}

// Mutations of the real constants (cudaMemcpyToSymbol on this TU's copy of c_pack_emit): each one must be caught by the
// d sweep or by the end-to-end X -> byte runs above. A check that cannot fail checks nothing.
int mutation_checks() {
  imd::device::PackEmitTable good = imd::device::kPackEmit;
  enum Kind { kHmMinus, kM32Minus, kCMinusP, kNegpPlus, kWeightPlus, kCPlus, kHmPlus };
  struct Mut { const char* what; Kind kind; int w, l; };
  const Mut muts[] = {{"hm - m32 (h one too small)", kHmMinus, 3, 5}, {"m32 - 1", kM32Minus, 3, 9},
                      {"C - p (C < dmax)", kCMinusP, 3, 3}, {"negp + 1", kNegpPlus, 2, 0},
                      {"weight lane +1", kWeightPlus, 3, 7}, {"C + 1 (not a multiple of p)", kCPlus, 2, 12},
                      {"hm + m32 (h one too large), p = 256", kHmPlus, 3, 19},
                      {"weight lane +1, W = 2", kWeightPlus, 2, 10},
                      {"m32 - 1, p = 256 (the low-byte path)", kM32Minus, 3, 19},
                      {"C - p, p = 256", kCMinusP, 3, 19}};
  int fails = 0;
  g_quiet = true;
  int8_t* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 2 * 12 * 255 * 255 + 64));
  std::vector<int8_t> h;
  for (const Mut& m : muts) {
    imd::device::PackEmitTable t = good;
    imd::device::PackEmit& e = t.e[m.w - 2][m.l];
    switch (m.kind) {
      case kHmMinus: e.hm -= e.m32; break;
      case kM32Minus: e.m32 -= 1; break;
      case kCMinusP: e.C -= 0u - e.negp; break;
      case kNegpPlus: e.negp += 1; break;
      case kWeightPlus: e.w[1] += 1; break;
      case kCPlus: e.C += 1; break;
      case kHmPlus: e.hm += e.m32; break;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(imd::device::c_pack_emit, &t, sizeof t));
    long bad = m.w == 3 ? check_d<3>(m.l, d_out, h) : check_d<2>(m.l, d_out, h);
    long checked = 0;
    Coverage cv;
    cv.eps.assign(c::kEpsMax + 1, 0);
    const int N = 20;
    bad += m.w == 3 ? run<16, 3, 1>(N, false, 4 * 1024, 77, cv, &checked) : run<16, 2, 1>(16, false, 4 * 1024, 78, cv, &checked);
    std::printf("mutation \"%s\" (W=%d, modulus %d): %s (%ld mismatches)\n", m.what, m.w, m.l, bad ? "caught" : "NOT CAUGHT", bad);
    fails += bad == 0;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(imd::device::c_pack_emit, &good, sizeof good));
  // O1's table (task 17): C2 off by one, m32 - 1, a weight lane +1; each caught by the O1 d sweep or the O1 end-to-end run
  const imd::device::PackEmitTableO good_o = imd::device::kPackEmitO;
  const struct { const char* what; int kind, l; } muts_o[] = {{"O1 C2 + 1", 0, 4}, {"O1 C2 - 1 (mod p), p = 256", 1, 0},
                                                              {"O1 m32 - 1", 2, 9}, {"O1 weight lane +1", 3, 11}};
  for (const auto& m : muts_o) {
    imd::device::PackEmitTableO t = good_o;
    imd::device::PackEmit& e = t.e[m.l];
    const uint32_t p = 0u - e.negp;
    switch (m.kind) {
      case 0: e.C = (e.C + 1) % p; break;
      case 1: e.C = (e.C + p - 1) % p; break;
      case 2: e.m32 -= 1; break;
      case 3: e.w[1] += 1; break;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(imd::device::c_pack_emit_o, &t, sizeof t));
    long bad = check_d_o(m.l, d_out, h), checked = 0;
    Coverage cv;
    cv.eps.assign(c::kEpsMax + 1, 0);
    bad += run<16, 2, 5>(16, false, 4 * 1024, 79, cv, &checked) + run<16, 2, 7>(16, true, 4 * 1024, 80, cv, &checked);
    std::printf("mutation \"%s\" (modulus %d): %s (%ld mismatches)\n", m.what, m.l, bad ? "caught" : "NOT CAUGHT", bad);
    fails += bad == 0;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(imd::device::c_pack_emit_o, &good_o, sizeof good_o));
  CUDA_CHECK(cudaFree(d_out));
  // Mutants of the sequences' code (the Mut template values: tests of the test, planted in the production templates at compile
  // time): every one must produce mismatches, except the one marked equivalent, which must produce none.
  Coverage cv;
  cv.eps.assign(c::kEpsMax + 1, 0);
  const auto go = [&](const char* what, bool expect_caught, const auto& f) {
    long checked = 0, bad = 0;
    for (int tr = 0; tr < 2; ++tr) bad += f(tr == 1, &checked);
    const bool ok = expect_caught ? bad != 0 : bad == 0;
    std::printf("mutant \"%s\": %s (%ld mismatches)\n", what, ok ? (expect_caught ? "caught" : "equivalent, as expected") : "WRONG", bad);
    fails += !ok;
  };
  const int U = 4 * 1024;
  go("E1 sign polarity (sg inverted)", true, [&](bool t, long* c_) { return run<16, 2, 1, 11>(16, t, U, 5, cv, c_); });
  go("E1 byte_perm assembly (two bytes swapped)", true, [&](bool t, long* c_) { return run<8, 3, 1, 12>(20, t, U, 6, cv, c_); });
  go("store_slots start slot not advanced", true, [&](bool t, long* c_) { return run<16, 2, 1, 0, 1>(16, t, U, 7, cv, c_); });
  go("Q1 without the f2 factor", true, [&](bool t, long* c_) { return run<16, 2, 3, 1>(16, t, U, 8, cv, c_); });
  go("Q1 RN for TRUNC", true, [&](bool t, long* c_) { return t ? run<16, 2, 3, 2>(16, t, U, 9, cv, c_) : 0L; });
  go("Q1 without the Inf/NaN mask", true, [&](bool t, long* c_) { return run<16, 2, 3, 3>(16, t, U, 10, cv, c_); });
  go("Q1 s2 one too large", true, [&](bool t, long* c_) { return run<16, 2, 3, 5>(16, t, U, 11, cv, c_); });
  go("Q1 sigma / 2 instead of sigma >> 1 (an equivalent split)", false, [&](bool t, long* c_) { return run<16, 2, 3, 4>(16, t, U, 12, cv, c_); });
  go("O1 top bit not flipped (integer quantize)", true, [&](bool t, long* c_) { return run<16, 2, 5, 13>(16, t, U, 13, cv, c_); });
  go("O1 top bit not flipped (signed F2I)", true, [&](bool t, long* c_) { return run<8, 2, 7, 13>(16, t, U, 14, cv, c_); });
  go("O1 + Q1 without the f2 factor", true, [&](bool t, long* c_) { return run<4, 2, 7, 1>(16, t, U, 15, cv, c_); });
  go("O1 + Q1 RN for TRUNC", true, [&](bool t, long* c_) { return t ? run<16, 2, 7, 2>(16, t, U, 16, cv, c_) : 0L; });
  go("O1 + Q1 without the Inf/NaN mask", true, [&](bool t, long* c_) { return run<16, 2, 7, 3>(16, t, U, 17, cv, c_); });
  go("O1 sign polarity mutant of E1 (Mut 11: no sg in O1)", false, [&](bool t, long* c_) { return run<16, 2, 5, 11>(16, t, U, 18, cv, c_); });
  g_quiet = false;
  return fails;
}

// v1.16 §3.4.1 (review 2026-10-06): shl_clamped is PTX shl.b64, x << n for n < 64 and 0 for
// every n >= 64 (the C++ shift is undefined there). Counts 0..255 and 2^32 - 1 on a few x.
__global__ void shl_kernel(const uint64_t* x, int nx, const uint32_t* n, int nn, uint64_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nx * nn) return;
  out[t] = imd::kernels::shl_clamped(x[t / nn], n[t % nn]);
}

int check_shl_clamped() {
  const std::vector<uint64_t> xs = {1, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull,
                                    0x0123456789ABCDEFull, (1ull << 53) - 1};
  std::vector<uint32_t> ns;
  for (uint32_t n = 0; n < 256; ++n) ns.push_back(n);
  ns.push_back(0xFFFFFFFFu);
  ns.push_back(0x80000000u);
  const int nx = static_cast<int>(xs.size()), nn = static_cast<int>(ns.size()), tot = nx * nn;
  uint64_t *dx, *dout;
  uint32_t* dn;
  CUDA_CHECK(cudaMalloc(&dx, sizeof(uint64_t) * nx));
  CUDA_CHECK(cudaMalloc(&dn, sizeof(uint32_t) * nn));
  CUDA_CHECK(cudaMalloc(&dout, sizeof(uint64_t) * tot));
  CUDA_CHECK(cudaMemcpy(dx, xs.data(), sizeof(uint64_t) * nx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dn, ns.data(), sizeof(uint32_t) * nn, cudaMemcpyHostToDevice));
  shl_kernel<<<(tot + 255) / 256, 256>>>(dx, nx, dn, nn, dout);
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint64_t> out(tot);
  CUDA_CHECK(cudaMemcpy(out.data(), dout, sizeof(uint64_t) * tot, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dn));
  CUDA_CHECK(cudaFree(dout));
  int bad = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < nn; ++j) {
      const uint64_t want = ns[j] >= 64 ? 0 : xs[i] << ns[j];
      if (out[i * nn + j] != want) {
        if (bad < 5)
          std::printf("shl_clamped(%#llx, %u) = %#llx, want %#llx\n",
                      static_cast<unsigned long long>(xs[i]), ns[j],
                      static_cast<unsigned long long>(out[i * nn + j]),
                      static_cast<unsigned long long>(want));
        ++bad;
      }
    }
  std::printf("shl_clamped: %d counts x %d values (0..255, 2^31, 2^32 - 1), %d mismatches\n",
              nn, nx, bad);
  return bad;
}

}  // namespace

int main() {
  static_assert(imd::kernels::pack_words(2) == 2 && imd::kernels::pack_words(16) == 2 &&
                    imd::kernels::pack_words(17) == 3,
                "W rule (cert_pack_words)");
  long bad = 0, checked = 0;
  g_scratch.eps.assign(c::kEpsMax + 1, 0);
  int fails = check_shl_clamped() != 0;
  long dtotal = 0;
  const long dbad = check_all_d(&dtotal);
  std::printf("E1 emitter on every reduced d: %ld (d, sign) values over 20 moduli and W = 2, 3, %ld mismatches\n", dtotal, dbad);
  fails += dbad != 0;
  for (int N = c::kNMin; N <= c::kNMax; ++N) {
    Coverage covw[3];  // [W - 1], W = 2, 3 (review L1: "all eps hit" per W)
    for (Coverage& cv : covw) cv.eps.assign(c::kEpsMax + 1, 0);
    const int w_min = imd::kernels::pack_words(N);
    long bn = 0;
    for (int tr = 0; tr < 2; ++tr) {
      const bool trunc = tr == 1;
      const uint64_t seed = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(4 * N + tr + 1);
      bn += run_n<16>(N, trunc, 4 * 1024, seed, covw, &checked);
      bn += run_n<8>(N, trunc, 4 * 512, seed ^ 0x1234, covw, &checked);
      bn += run_n<4>(N, trunc, 4 * 512, seed ^ 0x5678, covw, &checked);
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
  fails += mutation_checks();
  std::printf("pack emitter: %ld residues checked, %ld mismatches\n%s\n", checked, bad,
              fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
