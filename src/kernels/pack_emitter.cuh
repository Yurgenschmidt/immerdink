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

// Instruction sequences (issue 21, docs/NOTES-21.md "Pack emitter and finish"). S selects among
// implementations of the same function; every S gives the same bytes. The launch picks S per device
// class (host/seq_config.hpp); S = 0 is the baseline and stays the choice wherever a sequence is not
// at least as fast.
constexpr int kSeqE1 = 1;  // fused residue emitter (device/residue.hpp PackEmit, modp.hpp pack_smod)
constexpr int kSeqQ1 = 2;  // quantize by DMUL, DMUL, F2I.U64 (numerics.cuh quantize_cvt); W = 2 only

template <int E, int W, int S = 0>  // E consecutive k-bar of one row: 16 (one uint4 per modulus), 8 or 4
struct PackEmitterT {
  static_assert(E == 16 || E == 8 || E == 4, "16, 8 or 4 values per emitter");
  static_assert(W == 2 || W == 3, "X = |a'| in 2 (N <= 16) or 3 (N <= 20) u32 words");
  static_assert(S >= 0 && S <= 3 && ((S & kSeqQ1) == 0 || W == 2), "S: bit 0 E1, bit 1 Q1 (W = 2 only)");
  static constexpr bool kE1 = (S & kSeqE1) != 0, kQ1 = (S & kSeqQ1) != 0;
  uint32_t x[W][E];  // X_e = |a'_e| = sum_j x[j][e] 2^(32 j)
  uint32_t neg = 0;  // bit e: a'_e < 0 (baseline emitter)
  uint32_t sg[E];    // E1: sigma_e = +1 or -1 (two's complement) per element

  __device__ __forceinline__ void load(const uint64_t* bits, int sigma, bool trunc) {
    if constexpr (kQ1) {
      const QuantScale qs = quant_scale(sigma);
      if (trunc) load_cvt<true>(bits, qs);
      else load_cvt<false>(bits, qs);
    } else {
      load_int(bits, sigma, trunc);
    }
  }
  __device__ __forceinline__ void load_int(const uint64_t* bits, int sigma, bool trunc) {
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
      set_sign(e, bits[e]);
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
  __device__ __forceinline__ void set_sign(int e, uint64_t b) {
    if constexpr (kE1) sg[e] = (b >> 63) != 0 ? 0xFFFFFFFFu : 1u;
    else neg |= static_cast<uint32_t>((b & kSignBit) != 0) << e;
  }
  // Q1 (W = 2): X from the conversion of |x| 2^sigma.
  template <bool TRUNC>
  __device__ __forceinline__ void load_cvt(const uint64_t* bits, const QuantScale& qs) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const uint64_t X = quantize_cvt<TRUNC>(bits[e], qs);
#ifndef NDEBUG
      assert(X >> imd::constants::kPackXBits2 == 0);
#endif
      set_sign(e, bits[e]);
      x[0][e] = static_cast<uint32_t>(X);
      x[1][e] = static_cast<uint32_t>(X >> 32);
    }
  }
  // The E int8 residues for modulus l, as E / 4 little-endian u32 words. Branch-free for
  // every modulus and every element (p = 256 included: its weights are 1, 0, ..., so s is the
  // low byte and the 2^24 magic gives quotient 0). The sign is applied after SMod (A5):
  // SMod(-r mod p) = -SMod(r) for odd p (symmetric range), and for p = 256 both give the byte
  // (-r) & 255 (the one asymmetric value, r = 128, is the byte 0x80 either way).
  __device__ __forceinline__ void emit(int l, uint32_t (&w)[E / 4]) const {
    if constexpr (kE1) {
      emit_e1(l, w);
      return;
    }
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
  // E1: d = dp4a chain, s' = sigma d + C, q = hi32(s' m32 + h m32), u = s' - p q = SMod(sigma d, p); the int8 is u's
  // low byte, four of them per word by three byte permutes. No sign select, no SMod compare, no negation.
  __device__ __forceinline__ void emit_e1(int l, uint32_t (&w)[E / 4]) const {
    const imd::device::PackEmit& c = imd::device::c_pack_emit.e[W - 2][l];
    uint32_t u[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
      uint32_t xe[W];
#pragma unroll
      for (int j = 0; j < W; ++j) xe[j] = x[j][e];
      u[e] = static_cast<uint32_t>(imd::device::pack_smod(imd::device::pack_dot<W>(xe, c), sg[e], c));
    }
#pragma unroll
    for (int q = 0; q < E / 4; ++q) {
      const uint32_t lo = __byte_perm(u[4 * q], u[4 * q + 1], 0x0040u);      // bytes: u0.b0 u1.b0 0 0
      const uint32_t hi = __byte_perm(u[4 * q + 2], u[4 * q + 3], 0x0040u);  // bytes: u2.b0 u3.b0 0 0
      w[q] = __byte_perm(lo, hi, 0x5410u);
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
