// Exact constant-divisor reductions of DESIGN §3.5 and §3.6, usable from host C++ and
// CUDA device code (the CPU exhaustive test runs this exact code; cert_barrett and
// cert_crt_y_reduction in spec/params.py prove it).
#pragma once

#include <cstdint>

#include "device/residue.hpp"

#if defined(__CUDACC__)
#  define IMD_HD __host__ __device__ __forceinline__
#else
#  define IMD_HD inline
#endif

namespace imd::device {

// floor(a * b / 2^64).
IMD_HD uint64_t umulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#else
  __extension__ typedef unsigned __int128 u128;
  return static_cast<uint64_t>((static_cast<u128>(a) * b) >> 64);
#endif
}

// umulhi64(a, n) for a 32-bit n, from 32-bit words (A2): with a = ah 2^32 + al,
// floor(a n / 2^64) = floor((ah n + floor(al n / 2^32)) / 2^32), because dropping the
// fraction al n / 2^32 - floor(..) < 1 from the integer ah n 2^32 + al n cannot cross a
// multiple of 2^64; ah n + floor(al n / 2^32) < 2^64. Same value as umulhi64(a, n) (so
// cert_barrett's proof applies unchanged). On the device: t = umulhi(al, n), then the low
// and high halves of ah n + t as one carry chain (mad.lo.cc, madc.hi), 3 multiplies instead
// of the generic 64 x 64 high product (5 instructions + carries).
IMD_HD uint32_t umulhi64_u32(uint64_t a, uint32_t n) {
  const auto al = static_cast<uint32_t>(a), ah = static_cast<uint32_t>(a >> 32);
#if defined(__CUDA_ARCH__)
  const uint32_t t = __umulhi(al, n);
  uint32_t q;
  asm("{\n\t.reg .u32 lo;\n\t"
      "mad.lo.cc.u32 lo, %1, %2, %3;\n\t"
      "madc.hi.u32 %0, %1, %2, 0;\n\t}"
      : "=r"(q)
      : "r"(ah), "r"(n), "r"(t));
  return q;
#else
  const auto t = static_cast<uint32_t>((uint64_t{al} * n) >> 32);
  return static_cast<uint32_t>((uint64_t{ah} * n + t) >> 32);
#endif
}

// §3.5 epilogue for p != 256: n = v + bias in [0, 2^32), q = umulhi64(magic, n) = n div p,
// v mod p = n - p q (all u32 words), exact for v in [epi_vmin, epi_vmax] (cert_barrett).
// neg_p = 2^32 - p, so n - p q = n + neg_p q mod 2^32 (one IMAD; callers hoist neg_p).
IMD_HD uint32_t mod_p_barrett_neg(int32_t v, uint32_t neg_p, uint32_t bias, uint64_t magic) {
  const uint32_t n = static_cast<uint32_t>(v) + bias;
  return n + neg_p * umulhi64_u32(magic, n);
}
IMD_HD uint32_t mod_p_barrett(int32_t v, uint32_t p, uint32_t bias, uint64_t magic) {
  return mod_p_barrett_neg(v, 0u - p, bias, magic);
}

// §3.5 epilogue: the non-negative residue v mod p for v in the certified reachable range
// [epi_vmin, epi_vmax]. p = 256: two's-complement mask; otherwise mod_p_barrett.
IMD_HD uint32_t mod_p_epilogue(int32_t v, uint32_t p, uint32_t bias, uint64_t magic) {
  if (p == 256u) return static_cast<uint32_t>(v) & 255u;
  return mod_p_barrett(v, p, bias, magic);
}

// §3.6: n mod p for n in [0, 2^16) (n = q_l W_l), same magic, bias 0.
IMD_HD uint32_t mod_p_small(uint32_t n, uint32_t p, uint64_t magic) {
  if (p == 256u) return n & 255u;
  const auto q = static_cast<uint32_t>(umulhi64(magic, n));
  return n - p * q;
}

// §3.6 y = n mod p for n < 2^16 with the 32-bit magic m32 = ceil(2^32/p): n m32 = (n 2^32 +
// n e) / p with 0 <= e < p, and n e < 2^16 * 2^8 <= 2^32 / p for p <= 256, so
// floor(n m32 / 2^32) = floor(n / p). p = 256: the low byte. (Exhaustive CPU test.)
IMD_HD uint32_t mod_p_small32(uint32_t n, uint32_t p, uint32_t m32) {
  if (p == 256u) return n & 255u;
#if defined(__CUDA_ARCH__)
  const uint32_t q = __umulhi(n, m32);
#else
  const auto q = static_cast<uint32_t>((uint64_t{n} * m32) >> 32);
#endif
  return n - p * q;
}

// §3.4 residue of a quantized magnitude mu < 2^56, by byte weights (DESIGN §3.4 allows any
// exact reduction): mu = sum_i b_i 2^(8i), so mu = sum_i b_i w_i (mod p) with w_i = 2^(8i)
// mod p. lo = bits 0..31 and hi = bits 32..63 of mu (hi < 2^24); wlo / whi pack w_0..w_3 /
// w_4..w_6 (and 0) as u8 lanes. The dot product is < 7 * 255^2 < 2^19, and
// umulhi64(ceil(2^64/p), s) = floor(s / p) exactly for every s < 2^32 and p <= 256
// (s ceil(2^64/p) = (s 2^64 + s e) / p with 0 <= e < p and s e < 2^64), so the result is
// the non-negative residue mu mod p. p = 256: the low byte.
IMD_HD uint32_t dot_bytes(uint32_t x, uint32_t w) {
#if defined(__CUDA_ARCH__)
  return __dp4a(x, w, 0u);
#else
  uint32_t s = 0;
  for (int i = 0; i < 4; ++i) s += ((x >> (8 * i)) & 255u) * ((w >> (8 * i)) & 255u);
  return s;
#endif
}
IMD_HD uint32_t mod_p_bytes(uint32_t lo, uint32_t hi, uint32_t wlo, uint32_t whi, uint32_t p,
                            uint64_t magic) {
  if (p == 256u) return lo & 255u;
  const uint32_t s = dot_bytes(lo, wlo) + dot_bytes(hi, whi);
  return s - p * static_cast<uint32_t>(umulhi64(magic, s));
}

// A5: the same residue with the 32-bit magic m32 = ceil(2^32/p) (Magic32: 2^24 for p = 256),
// one multiply-high instead of the 64-bit one, branch-free for every p. Exact for s < 2^24:
// s m32 = (s 2^32 + s e) / p with 0 <= e < p <= 256 (e = 0 for p = 256), and s e < 2^24 2^8
// = 2^32, so floor(s m32 / 2^32) = floor(s / p) (as for mod_p_small32); here s < 7 * 255^2
// < 2^19. For p = 256 the weights are 1, 0, ... so s = lo & 255 and the quotient is 0.
// (Exhaustive over [0, 7 * 255^2] and end to end: tests/test_mod_exhaustive.cpp.)
IMD_HD uint32_t mod_p_bytes32(uint32_t lo, uint32_t hi, uint32_t wlo, uint32_t whi, uint32_t p,
                              uint32_t m32) {
  const uint32_t s = dot_bytes(lo, wlo) + dot_bytes(hi, whi);
#if defined(__CUDA_ARCH__)
  const uint32_t q = __umulhi(s, m32);
#else
  const auto q = static_cast<uint32_t>((uint64_t{s} * m32) >> 32);
#endif
  return s - p * q;
}

// v1.14 §3.4.1: X mod p for X = sum_j x[j] 2^(32 j) < 2^(32 W) whose bytes 10, 11 are zero
// when W = 3 (X < 2^78: cert_pack_words), with the pack weights wt[j] (u8 lanes 2^(8(4j+i))
// mod p, device/residue.hpp PackWeights). s = sum_j dp4a(x[j], wt[j]) = X (mod p) and
// s <= 12 * 255^2 < 2^20, inside mod_p_bytes32's exact range s < 2^24 (same quotient by
// the 32-bit magic m32 = ceil(2^32/p), 2^24 for p = 256). Returns the non-negative X mod p.
// (Exhaustive over [0, 12 * 255^2] and end to end: tests/test_mod_exhaustive.cpp.)
template <int W>
IMD_HD uint32_t mod_p_words32(const uint32_t (&x)[W], const uint32_t (&wt)[W], uint32_t p,
                              uint32_t m32) {
  static_assert(W == 2 || W == 3, "2 or 3 words");
  uint32_t s = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int j = 0; j < W; ++j) s = __dp4a(x[j], wt[j], s);
  const uint32_t q = __umulhi(s, m32);
#else
  for (int j = 0; j < W; ++j) s += dot_bytes(x[j], wt[j]);
  const auto q = static_cast<uint32_t>((uint64_t{s} * m32) >> 32);
#endif
  return s - p * q;
}

// E1 (device/residue.hpp PackEmit has the math): the signed residue from the dp4a sum d.
// d = sum_j dp4a(x_j, w_j) (unsigned lanes): 0 <= d <= dmax of the (W, modulus) constants.
template <int W>
IMD_HD uint32_t pack_dot(const uint32_t (&x)[W], const PackEmit& c) {
  static_assert(W == 2 || W == 3, "2 or 3 words");
  uint32_t d = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int j = 0; j < W; ++j) d = __dp4a(x[j], c.w[j], d);
#else
  for (int j = 0; j < W; ++j) d += dot_bytes(x[j], c.w[j]);
#endif
  return d;
}
// u = SMod(sigma d, p) for sigma = +1 (sg = 1) or -1 (sg = 0xFFFFFFFF), as an int32 in [-128, 127]; its low byte is
// the int8 of the pack. s' + h < 2^24 (pack_emit_bounds_ok), so the 64-bit multiply-add never wraps.
IMD_HD int32_t pack_smod(uint32_t d, uint32_t sg, const PackEmit& c) {
  const uint32_t s1 = d * sg + c.C;
  const auto q = static_cast<uint32_t>((uint64_t{s1} * c.m32 + c.hm) >> 32);
  return static_cast<int32_t>(s1 + c.negp * q);
}
// O1 (residue.hpp PackEmitTableO): s' = C2 + sum_j dp4a(y_j, w_j), the start value riding in the chain's accumulator, and
// u = SMod(s', p) by the same quotient as pack_smod without the sign multiply. s' + h < 2^24 (pack_emit_o_bounds_ok).
IMD_HD uint32_t pack_dot_o(const uint32_t (&y)[2], const PackEmit& c) {
#if defined(__CUDA_ARCH__)
  return __dp4a(y[1], c.w[1], __dp4a(y[0], c.w[0], c.C));
#else
  return c.C + dot_bytes(y[0], c.w[0]) + dot_bytes(y[1], c.w[1]);
#endif
}
IMD_HD int32_t pack_smod_o(uint32_t s1, const PackEmit& c) {
  const auto q = static_cast<uint32_t>((uint64_t{s1} * c.m32 + c.hm) >> 32);
  return static_cast<int32_t>(s1 + c.negp * q);
}

}  // namespace imd::device
