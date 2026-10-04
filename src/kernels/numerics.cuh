// Device numerics of DESIGN §3.1-§3.7, written for exactness (performance is M4):
//   binary64 field access, the §3.2 integer shift rule, §3.3 quantization, §3.4 residues,
//   the §3.6 multi-word running CRT and the §3.7 single correctly rounded conversion.
// spec/oracle.py is the bit-level reference for every function here.
#pragma once

#include <climits>
#include <cstdint>

#include "device/constants.cuh"

namespace imd::kernels {

__extension__ typedef unsigned __int128 u128;

constexpr uint64_t kSignBit = uint64_t{1} << 63;
constexpr uint64_t kFracMask = (uint64_t{1} << 52) - 1;

__device__ __forceinline__ uint64_t to_bits(double x) {
  return static_cast<uint64_t>(__double_as_longlong(x));
}
__device__ __forceinline__ double from_bits(uint64_t b) {
  return __longlong_as_double(static_cast<long long>(b));
}
// §3.1: Inf or NaN, by the exponent field only.
__device__ __forceinline__ bool non_finite(uint64_t b) { return ((b >> 52) & 0x7FF) == 0x7FF; }
// Finite and nonzero (either sign of zero is zero).
__device__ __forceinline__ bool finite_nonzero(uint64_t b) {
  return !non_finite(b) && (b & ~kSignBit) != 0;
}
// |x| = mant * 2^exp exactly, mant < 2^53 (integer significand incl. the implicit bit).
__device__ __forceinline__ void decompose(uint64_t b, uint64_t* mant, int* exp) {
  const int E = static_cast<int>((b >> 52) & 0x7FF);
  const uint64_t f = b & kFracMask;
  if (E != 0) {
    *mant = f | (uint64_t{1} << 52);
    *exp = E - 1075;
  } else {
    *mant = f;
    *exp = -1074;
  }
}
// floor(log2 |x|) for finite nonzero x (subnormals included).
__device__ __forceinline__ int ilogb_bits(uint64_t b) {
  const int E = static_cast<int>((b >> 52) & 0x7FF);
  if (E != 0) return E - 1023;
  return -1075 + (64 - __clzll(static_cast<long long>(b & kFracMask)));
}

// §3.2 step 3 term: ceil(m^2 * 2^t), t = 2(e - gamma) + F, for finite nonzero x with
// ilogb(x) <= gamma. The result is < 2^62 (cert kTermMax); every nonzero entry gives >= 1.
__device__ __forceinline__ u128 square_term(uint64_t b, int gamma) {
  uint64_t m;
  int e;
  decompose(b, &m, &e);
  const u128 sq = static_cast<u128>(m) * m;  // < 2^106
  const int t = 2 * (e - gamma) + imd::constants::kF;
  if (t >= 0) return sq << t;  // t <= F here, and the value is < 2^62
  const int s = -t;
  if (s >= 128) return 1;      // 0 < m^2 2^t < 1
  const u128 q = sq >> s;
  return (sq & ((static_cast<u128>(1) << s) - 1)) != 0 ? q + 1 : q;
}

// A5: the same term with the significand normalized to 53 bits, in two 64-bit words (the
// generic 128-bit shifts above are ~70 instructions per value; KF, KL and K1 run this once
// per operand element). m' = m 2^z in [2^52, 2^53) (z > 0 only for subnormals) has
// ilogb(x) = e' + 52 for e' = e - z, so the same real number m^2 2^t = m'^2 / 2^s with
// s = 2(gamma - ilogb(x)) + 104 - F >= 104 - F = 44. m'^2 < 2^106: s >= 106 gives 1 (0 <
// value < 1); otherwise the value is < 2^(106 - 44) = 2^62 and the result is
// floor(m'^2 / 2^s) + (remainder != 0) from hi:lo = m'^2. Equal to square_term for every
// finite nonzero x with ilogb(x) <= gamma (GPU test: tests/test_device_constants.cu).
static_assert(imd::constants::kF >= 0 && imd::constants::kF <= 61,
              "square_term_n: s >= 104 - F >= 43 and results below 2^(F + 2) <= 2^63");
__device__ __forceinline__ uint64_t square_term_n(uint64_t b, int gamma) {
  const int E = static_cast<int>((b >> 52) & 0x7FF);
  const uint64_t f = b & kFracMask;
  uint64_t m;
  int il;  // ilogb(x)
  if (E != 0) {
    m = f | (uint64_t{1} << 52);
    il = E - 1023;
  } else {  // subnormal, f != 0: bitlen(f) = 64 - lz
    const int lz = __clzll(static_cast<long long>(f));
    m = f << (lz - 11);
    il = -1075 + (64 - lz);
  }
  const int s = 2 * (gamma - il) + 104 - imd::constants::kF;
  if (s >= 106) return 1;
  const uint64_t lo = m * m, hi = __umul64hi(m, m);
  if (s < 64) {
    const uint64_t q = (hi << (64 - s)) | (lo >> s);
    return q + ((lo << (64 - s)) != 0 ? 1 : 0);
  }
  const int r = s - 64;
  const uint64_t q = hi >> r;
  return q + ((lo != 0 || (r != 0 && (hi << (64 - r)) != 0)) ? 1 : 0);
}

__device__ __forceinline__ int bitlen128(u128 x) {
  const auto hi = static_cast<uint64_t>(x >> 64), lo = static_cast<uint64_t>(x);
  if (hi != 0) return 128 - __clzll(static_cast<long long>(hi));
  if (lo != 0) return 64 - __clzll(static_cast<long long>(lo));
  return 0;
}

// 256-bit little-endian helpers for the §3.2 step 4 compare.
struct U256 {
  uint64_t w[4];
};
__device__ __forceinline__ int bitlen256(const U256& x) {
  for (int i = 3; i >= 0; --i)
    if (x.w[i] != 0) return 64 * i + 64 - __clzll(static_cast<long long>(x.w[i]));
  return 0;
}
// out = x << s; returns false if a set bit would be shifted out (out is then unspecified).
__device__ __forceinline__ bool shl256(const U256& x, int s, U256* out) {
  const int bl = bitlen256(x);
  if (bl == 0) {
    *out = x;
    return true;
  }
  if (s < 0 || bl + s > 256) return false;
  const int ws = s / 64, bs = s % 64;
  // Word shift by selection over the 4 possible ws (static indices only: a dynamic x.w[i - ws]
  // would put the words in local memory, which is costly when every thread of a warp runs it).
#pragma unroll
  for (int i = 3; i >= 0; --i) {
    uint64_t v = 0;
#pragma unroll
    for (int j = 0; j <= 3; ++j) {
      if (j == ws && i - j >= 0) {
        v = x.w[i - j] << bs;
        if (bs != 0 && i - j >= 1) v |= x.w[i - j - 1] >> (64 - bs);
      }
    }
    out->w[i] = v;
  }
  return true;
}
__device__ __forceinline__ bool le256(const U256& a, const U256& b) {
  for (int i = 3; i >= 0; --i)
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
  return true;
}

// §3.2 step 4: the largest integer u with L * 4^u <= rhs, L = S << lhs_shift > 0, by the
// bit-length estimate c = floor((bitlen(rhs) - bitlen(L) + 1) / 2) (floor, also when the
// difference is negative) and one exact multi-word compare: u = Rule(c) ? c : c - 1.
// rhs = R * 2^F has <= 217 bits, L < 2^127 (S < 2^125 for k < 2^63), so with this c the
// shifted operands stay below 2^219 and never leave 256 bits.
__device__ __forceinline__ int largest_u(u128 S, int lhs_shift, const uint64_t (&rhs)[4],
                                         int rhs_bits) {
  const u128 L = S << lhs_shift;
  const int d = rhs_bits - bitlen128(L) + 1;
  const int c = d >= 0 ? d / 2 : -((1 - d) / 2);  // floor(d / 2)
  const U256 l{{static_cast<uint64_t>(L), static_cast<uint64_t>(L >> 64), 0, 0}};
  const U256 r{{rhs[0], rhs[1], rhs[2], rhs[3]}};
  U256 t;
  bool rule;
  if (c >= 0)
    rule = shl256(l, 2 * c, &t) && le256(t, r);  // L 4^c <= rhs
  else
    rule = !shl256(r, -2 * c, &t) || le256(l, t);  // L <= rhs 4^-c
  return rule ? c : c - 1;
}

// §3.3: a' = RN_even(x 2^sigma) or trunc(x 2^sigma) for finite x, as |a'| = mu * 2^eps.
// eps <= floor(log2 T) <= kEpsMax by the norm budget (cert_pow2 / cert_shift_rule_guarantee).
__device__ __forceinline__ void quantize(uint64_t b, int sigma, bool trunc, uint64_t* mu,
                                         int* eps) {
  *mu = 0;
  *eps = 0;
  if (!finite_nonzero(b)) return;
  uint64_t m;
  int e;
  decompose(b, &m, &e);
  const int s = e + sigma;
  if (s >= 0) {
    *mu = m;
    *eps = s;
    return;
  }
  const int sh = -s;
  if (sh >= 60) return;  // m < 2^53 < half an ulp of the integer grid: rounds to 0
  uint64_t r = m >> sh;
  if (!trunc) {
    const uint64_t rem = m & ((uint64_t{1} << sh) - 1), half = uint64_t{1} << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
  }
  *mu = r;
}

// §3.4: SMod((mu mod p)(2^eps mod p), p) as int8 for a' = (neg ? -1 : 1) mu 2^eps.
__device__ __forceinline__ int8_t residue(uint64_t mu, bool neg, int eps, int l) {
  using imd::device::c_tables;
  const uint32_t p = c_tables.moduli[l];
  uint32_t r = static_cast<uint32_t>(mu % p);
  if (neg && r != 0) r = p - r;
  r = imd::device::mod_p_small(r * c_tables.pow2_mod[l][eps], p, c_tables.epi_magic[l]);
  return static_cast<int8_t>(r >= (p + 1) / 2 ? static_cast<int32_t>(r) - static_cast<int32_t>(p)
                                               : static_cast<int32_t>(r));
}

// §3.7: D = RN(sign * mag * 2^e), ONE rounding at the final exponent: gradual underflow
// (lsb never below 2^-1074), overflow to +-Inf, a zero result keeps the sign; mag = 0 -> +0.
// mag: L little-endian u32 limbs.
template <int L>
__device__ __forceinline__ double round_scaled(const uint32_t (&x)[L], bool neg, int e) {
  // The top nonzero limb, by a fully unrolled scan (static indices only: a dynamic x[top]
  // would put the limbs in local memory).
  int top = -1;
  uint32_t xt = 0;
#pragma unroll
  for (int w = 0; w < L; ++w)
    if (x[w] != 0) top = w, xt = x[w];
  if (top < 0) return 0.0;
  const int bl = top * 32 + (32 - __clz(static_cast<int>(xt)));
  uint64_t t;  // the top 64 bits of mag (exact when bl <= 64), lsb weight 2^(bl - 64 + e)
  bool sticky = false;
  if (bl <= 64) {
    uint64_t lo = x[0];
    if (L > 1) lo |= static_cast<uint64_t>(x[L > 1 ? 1 : 0]) << 32;
    t = lo << (64 - bl);
  } else {
    // t = floor(mag / 2^s) (< 2^64 since mag < 2^bl), sticky = (mag mod 2^s != 0), limb by
    // limb with runtime shift amounts only: selecting x[s / 32 + j] was compiled into a
    // dynamically indexed copy of x in local memory (K5's hottest loads).
    const int s = bl - 64;
    t = 0;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = 32 * i - s;  // bit position of limb i relative to bit s
      const uint64_t v = x[i];
      if (r >= 0) {
        if (r < 64) t |= v << r;
      } else if (r > -32) {
        t |= v >> -r;
        if ((v & ((uint64_t{1} << -r) - 1)) != 0) sticky = true;
      } else if (v != 0) {
        sticky = true;
      }
    }
  }
  const int lsb = bl - 64 + e;                  // exponent of t's lsb
  const int q = max(bl - 1 + e - 52, -1074);    // exponent of the result's lsb
  const int d = q - lsb;                        // >= 11 bits are dropped from t
  uint64_t r;
  if (d >= 65) {
    r = 0;  // t + sticky < 2^64 <= half
  } else if (d == 64) {
    const uint64_t half = uint64_t{1} << 63;
    r = (t > half || (t == half && sticky)) ? 1 : 0;  // tie -> even (0)
  } else {
    r = t >> d;
    const uint64_t rem = t & ((uint64_t{1} << d) - 1), half = uint64_t{1} << (d - 1);
    if (rem > half || (rem == half && (sticky || (r & 1)))) ++r;
  }
  if (r == 0) return neg ? -0.0 : 0.0;
  const int rbl = 64 - __clzll(static_cast<long long>(r));
  if (rbl - 1 + q >= 1024) return neg ? -__longlong_as_double(0x7FF0000000000000LL)
                                      : __longlong_as_double(0x7FF0000000000000LL);
  // r <= 2^53; r < 2^52 only when q = -1074 (subnormal); r = 2^53 carries into the exponent.
  const uint64_t bits = (static_cast<uint64_t>(q + 1074) << 52) + r;
  return from_bits(neg ? bits | kSignBit : bits);
}

// §3.6 running CRT over the N planes for one element: x <- 0; for l: y = (q_l W_l) mod p_l;
// x += M_l y (one carry chain); if x >= P: x -= P. Then z = x <= H ? x : x - P, returned as
// (|z| limbs, sign). L = L32(N) limbs suffice (2P <= 2^(32 L)).
// x += M y; if x >= P: x -= P, on L u32 limbs, as one PTX carry chain (two mad chains for
// the low and high halves of M_w y, then x - P with the borrow deciding the select). All
// arithmetic is mod 2^(32 L) and the true x + M y < 2P <= 2^(32 L) (cert_crt_peak), so
// carries out of the top limb are zero in the exact sum and dropping them is exact; the
// borrow of x - P is exactly the predicate x < P of §3.6.
template <int L>
__device__ __forceinline__ void crt_mac_sub(uint32_t (&x)[L], const uint32_t (&M)[L], uint32_t y,
                                            const uint32_t (&P)[L]);
template <>
__device__ __forceinline__ void crt_mac_sub<1>(uint32_t (&x)[1], const uint32_t (&M)[1], uint32_t y,
                                              const uint32_t (&P)[1]) {
  asm("{\n\t.reg .u32 t0, bw;\n\t.reg .pred p_ge;\n\tmad.lo.u32 %0, %1, %2, %0;\n\tsub.cc.u32 t0, %0, %3;\n\tsubc.u32 bw, 0, 0;\n\tsetp.eq.u32 p_ge, bw, 0;\n\tselp.u32 %0, t0, %0, p_ge;\n\t}" : "+r"(x[0]) : "r"(M[0]), "r"(y), "r"(P[0]));
}
template <>
__device__ __forceinline__ void crt_mac_sub<2>(uint32_t (&x)[2], const uint32_t (&M)[2], uint32_t y,
                                              const uint32_t (&P)[2]) {
  asm("{\n\t.reg .u32 t<2>, bw;\n\t.reg .pred p_ge;\n\tmad.lo.cc.u32 %0, %2, %4, %0;\n\tmadc.lo.u32 %1, %3, %4, %1;\n\tmad.hi.u32 %1, %2, %4, %1;\n\tsub.cc.u32 t0, %0, %5;\n\tsubc.cc.u32 t1, %1, %6;\n\tsubc.u32 bw, 0, 0;\n\tsetp.eq.u32 p_ge, bw, 0;\n\tselp.u32 %0, t0, %0, p_ge;\n\tselp.u32 %1, t1, %1, p_ge;\n\t}" : "+r"(x[0]), "+r"(x[1]) : "r"(M[0]), "r"(M[1]), "r"(y), "r"(P[0]), "r"(P[1]));
}
template <>
__device__ __forceinline__ void crt_mac_sub<3>(uint32_t (&x)[3], const uint32_t (&M)[3], uint32_t y,
                                              const uint32_t (&P)[3]) {
  asm("{\n\t.reg .u32 t<3>, bw;\n\t.reg .pred p_ge;\n\tmad.lo.cc.u32 %0, %3, %6, %0;\n\tmadc.lo.cc.u32 %1, %4, %6, %1;\n\tmadc.lo.u32 %2, %5, %6, %2;\n\tmad.hi.cc.u32 %1, %3, %6, %1;\n\tmadc.hi.u32 %2, %4, %6, %2;\n\tsub.cc.u32 t0, %0, %7;\n\tsubc.cc.u32 t1, %1, %8;\n\tsubc.cc.u32 t2, %2, %9;\n\tsubc.u32 bw, 0, 0;\n\tsetp.eq.u32 p_ge, bw, 0;\n\tselp.u32 %0, t0, %0, p_ge;\n\tselp.u32 %1, t1, %1, p_ge;\n\tselp.u32 %2, t2, %2, p_ge;\n\t}" : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]) : "r"(M[0]), "r"(M[1]), "r"(M[2]), "r"(y), "r"(P[0]), "r"(P[1]), "r"(P[2]));
}
template <>
__device__ __forceinline__ void crt_mac_sub<4>(uint32_t (&x)[4], const uint32_t (&M)[4], uint32_t y,
                                              const uint32_t (&P)[4]) {
  asm("{\n\t.reg .u32 t<4>, bw;\n\t.reg .pred p_ge;\n\tmad.lo.cc.u32 %0, %4, %8, %0;\n\tmadc.lo.cc.u32 %1, %5, %8, %1;\n\tmadc.lo.cc.u32 %2, %6, %8, %2;\n\tmadc.lo.u32 %3, %7, %8, %3;\n\tmad.hi.cc.u32 %1, %4, %8, %1;\n\tmadc.hi.cc.u32 %2, %5, %8, %2;\n\tmadc.hi.u32 %3, %6, %8, %3;\n\tsub.cc.u32 t0, %0, %9;\n\tsubc.cc.u32 t1, %1, %10;\n\tsubc.cc.u32 t2, %2, %11;\n\tsubc.cc.u32 t3, %3, %12;\n\tsubc.u32 bw, 0, 0;\n\tsetp.eq.u32 p_ge, bw, 0;\n\tselp.u32 %0, t0, %0, p_ge;\n\tselp.u32 %1, t1, %1, p_ge;\n\tselp.u32 %2, t2, %2, p_ge;\n\tselp.u32 %3, t3, %3, p_ge;\n\t}" : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3]) : "r"(M[0]), "r"(M[1]), "r"(M[2]), "r"(M[3]), "r"(y), "r"(P[0]), "r"(P[1]), "r"(P[2]), "r"(P[3]));
}
template <>
__device__ __forceinline__ void crt_mac_sub<5>(uint32_t (&x)[5], const uint32_t (&M)[5], uint32_t y,
                                              const uint32_t (&P)[5]) {
  asm("{\n\t.reg .u32 t<5>, bw;\n\t.reg .pred p_ge;\n\tmad.lo.cc.u32 %0, %5, %10, %0;\n\tmadc.lo.cc.u32 %1, %6, %10, %1;\n\tmadc.lo.cc.u32 %2, %7, %10, %2;\n\tmadc.lo.cc.u32 %3, %8, %10, %3;\n\tmadc.lo.u32 %4, %9, %10, %4;\n\tmad.hi.cc.u32 %1, %5, %10, %1;\n\tmadc.hi.cc.u32 %2, %6, %10, %2;\n\tmadc.hi.cc.u32 %3, %7, %10, %3;\n\tmadc.hi.u32 %4, %8, %10, %4;\n\tsub.cc.u32 t0, %0, %11;\n\tsubc.cc.u32 t1, %1, %12;\n\tsubc.cc.u32 t2, %2, %13;\n\tsubc.cc.u32 t3, %3, %14;\n\tsubc.cc.u32 t4, %4, %15;\n\tsubc.u32 bw, 0, 0;\n\tsetp.eq.u32 p_ge, bw, 0;\n\tselp.u32 %0, t0, %0, p_ge;\n\tselp.u32 %1, t1, %1, p_ge;\n\tselp.u32 %2, t2, %2, p_ge;\n\tselp.u32 %3, t3, %3, p_ge;\n\tselp.u32 %4, t4, %4, p_ge;\n\t}" : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3]), "+r"(x[4]) : "r"(M[0]), "r"(M[1]), "r"(M[2]), "r"(M[3]), "r"(M[4]), "r"(y), "r"(P[0]), "r"(P[1]), "r"(P[2]), "r"(P[3]), "r"(P[4]));
}


// One §3.6 step for plane l: y = (q_l W_l) mod p_l (q_l W_l < 2^16: the 32-bit magic,
// device/modp.hpp mod_p_small32); x += M_l y; if x >= P: x -= P.
template <int L>
__device__ __forceinline__ void crt_step(uint32_t (&x)[L], uint32_t Wl, int l, int N) {
  using imd::device::c_tables;
  const uint32_t p = c_tables.moduli[l];
  const uint32_t y =
      imd::device::mod_p_small32(c_tables.q[N][l] * Wl, p, imd::device::c_magic32.m[l]);
  uint32_t M[L], P[L];
#pragma unroll
  for (int w = 0; w < L; ++w) {
    M[w] = c_tables.M[N][l][w];
    P[w] = c_tables.P[N][w];
  }
  crt_mac_sub<L>(x, M, y, P);
}

template <int L>
__device__ __forceinline__ void crt_finish(const uint32_t (&x)[L], int N, uint32_t (&mag)[L],
                                           bool* neg);

template <int L>
__device__ __forceinline__ void crt(const uint8_t* W, int64_t plane, int N, uint32_t (&mag)[L],
                                    bool* neg) {
  uint32_t x[L];
#pragma unroll
  for (int w = 0; w < L; ++w) x[w] = 0;
  for (int l = 0; l < N; ++l) crt_step<L>(x, W[l * plane], l, N);
  crt_finish<L>(x, N, mag, neg);
}

// §3.6 centering: z = x <= H ? x : x - P, as (|z| limbs, sign).
template <int L>
__device__ __forceinline__ void crt_finish(const uint32_t (&x)[L], int N, uint32_t (&mag)[L],
                                           bool* neg) {
  using imd::device::c_tables;
  bool gt = false;  // x > H
#pragma unroll
  for (int w = L - 1; w >= 0; --w) {
    if (x[w] != c_tables.H[N][w]) {
      gt = x[w] > c_tables.H[N][w];
      break;
    }
  }
  *neg = gt;
  if (gt) {  // |z| = P - x
    uint64_t borrow = 0;
#pragma unroll
    for (int w = 0; w < L; ++w) {
      const uint64_t t = uint64_t{c_tables.P[N][w]} - x[w] - borrow;
      mag[w] = static_cast<uint32_t>(t);
      borrow = (t >> 32) & 1;
    }
  } else {
#pragma unroll
    for (int w = 0; w < L; ++w) mag[w] = x[w];
  }
}

}  // namespace imd::kernels
