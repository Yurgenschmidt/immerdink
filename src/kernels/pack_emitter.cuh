// The pack emitter of DESIGN §3.3 + §3.4 (v1.14 §3.4.1): E consecutive k-bar of one row are
// quantized once, then their int8 residues are emitted per modulus. The only residue code of
// K2/K3 (k23_pack), KF (kf_scale_pack), KL (kl_scale_pack) and so imd_pack (src/contract.cu),
// and of the GPU unit test tests/test_pack_emitter.cu.
//
// v1.14: |a'| = mu 2^eps is held exactly as X in W little-endian u32 words (W = 2 for
// N <= kPackW2MaxN = 16, since |a'| <= T(N) < 2^63; W = 3 up to N = 20, T < 2^78:
// cert_pack_words), and the residue of modulus l is sum_j dp4a(X_j, w_j(l)) reduced by one
// umulhi (device/modp.hpp mod_p_words32): the same instructions for every element and every
// data, no eps registers, no eps > 0 branch, no lane-divergent pow2_mod load. Same
// congruence class as §3.4 `residue`, so the same int8 bytes as the v1.13 emitter.
#pragma once

#include <cassert>
#include <cstdint>

#include "device/constants.cuh"
#include "kernels/numerics.cuh"

namespace imd::kernels {

// v1.14 §3.4.1: the emitter's word count for N moduli (a function of N alone).
__host__ __device__ constexpr int pack_words(int N) {
  return N <= imd::constants::kPackW2MaxN ? 2 : 3;
}

// v1.16 §3.4.1: x << n with PTX shl semantics: n >= 64 gives 0, never undefined behaviour.
__device__ __forceinline__ uint64_t shl_clamped(uint64_t x, uint32_t n) {
  uint64_t r;
  asm("shl.b64 %0, %1, %2;" : "=l"(r) : "l"(x), "r"(n));
  return r;
}

template <int E, int W>  // E consecutive k-bar of one row: 16 (one uint4 per modulus), 8 or 4
struct PackEmitterT {
  static_assert(E == 16 || E == 8 || E == 4, "16, 8 or 4 values per emitter");
  static_assert(W == 2 || W == 3, "X = |a'| in 2 (N <= 16) or 3 (N <= 20) u32 words");
  uint32_t x[W][E];  // X_e = |a'_e| = sum_j x[j][e] 2^(32 j)
  uint32_t neg = 0;  // bit e: a'_e < 0

  __device__ __forceinline__ void load(const uint64_t* bits, int sigma, bool trunc) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      uint64_t mu;  // <= 2^53
      int eps;      // bitlen(mu) + eps <= kPackXBits[W] (|a'| <= T(N): cert_pack_words)
      quantize(bits[e], sigma, trunc, &mu, &eps);  // non-finite / zero -> mu = 0, eps = 0
#ifndef NDEBUG
      // X's bits at and above kPackXBits[W] are zero: no bit of mu 2^eps is lost below.
      assert(mu == 0 || (64 - __clzll(static_cast<long long>(mu))) + eps <=
                            (W == 2 ? imd::constants::kPackXBits2 : imd::constants::kPackXBits3));
#endif
      neg |= static_cast<uint32_t>((bits[e] & kSignBit) != 0) << e;
      if constexpr (W == 2) {
        // eps <= 62 in the contract (|a'| <= T). v1.16: PTX shl, whose count is clamped at 64
        // (result 0), instead of the C++ shift, undefined for eps >= 64: a sigma bug cannot
        // turn into undefined behaviour in builds without the assert above (wrong bytes).
        const uint64_t X = shl_clamped(mu, static_cast<uint32_t>(eps));
        x[0][e] = static_cast<uint32_t>(X);
        x[1][e] = static_cast<uint32_t>(X >> 32);
      } else {
        // mu's words shifted by b = eps mod 32, placed by q = eps / 32 in {0, 1, 2} with
        // selects (static indices only: no local memory). Words past x[2] are zero (X < 2^78).
        // Defined for every eps (v1.16): b <= 31, and q >= 3 selects as q = 2.
        const uint32_t m0 = static_cast<uint32_t>(mu), m1 = static_cast<uint32_t>(mu >> 32);
        const int q = eps >> 5, b = eps & 31;
        const uint32_t s0 = m0 << b;
        const uint32_t s1 = __funnelshift_l(m0, m1, b);  // (m1:m0) << b, high word
        const uint32_t s2 = __funnelshift_l(m1, 0u, b);  // m1 >> (32 - b); 0 for b = 0
        x[0][e] = q == 0 ? s0 : 0u;
        x[1][e] = q == 0 ? s1 : (q == 1 ? s0 : 0u);
        x[2][e] = q == 0 ? s2 : (q == 1 ? s1 : s0);
      }
    }
  }
  // The E int8 residues for modulus l, as E / 4 little-endian u32 words. Branch-free for
  // every modulus and every element (p = 256 included: its weights are 1, 0, ..., so s is the
  // low byte and the 2^24 magic gives quotient 0). The sign is applied after SMod (A5):
  // SMod(-r mod p) = -SMod(r) for odd p (symmetric range), and for p = 256 both give the byte
  // (-r) & 255 (the one asymmetric value, r = 128, is the byte 0x80 either way).
  __device__ __forceinline__ void emit(int l, uint32_t (&w)[E / 4]) const {
    using imd::device::c_tables;
    const uint32_t p = c_tables.moduli[l], half = (p + 1) / 2;
    const uint32_t m32 = imd::device::c_magic32.m[l];
    uint32_t wt[W];
#pragma unroll
    for (int j = 0; j < W; ++j) wt[j] = imd::device::c_pack_weights.w[j][l];
    uint32_t r[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
      uint32_t xe[W];
#pragma unroll
      for (int j = 0; j < W; ++j) xe[j] = x[j][e];
      r[e] = imd::device::mod_p_words32<W>(xe, wt, p, m32);
    }
#pragma unroll
    for (int q = 0; q < E / 4; ++q) w[q] = 0;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const int32_t u = r[e] >= half ? static_cast<int32_t>(r[e]) - static_cast<int32_t>(p)
                                     : static_cast<int32_t>(r[e]);  // SMod(r)
      const int32_t sm = (neg >> e) & 1u ? -u : u;  // as an int8 bit pattern (low byte)
      w[e / 4] |= (static_cast<uint32_t>(sm) & 255u) << (8 * (e % 4));
    }
  }
  // Store the residues for modulus l at dst (aligned to E bytes).
  __device__ __forceinline__ void store(int l, int8_t* dst) const {
    uint32_t w[E / 4];
    emit(l, w);
    if constexpr (E == 16)
      *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    else if constexpr (E == 8)
      *reinterpret_cast<uint2*>(dst) = make_uint2(w[0], w[1]);
    else
      *reinterpret_cast<uint32_t*>(dst) = w[0];
  }
};

}  // namespace imd::kernels
