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
  for (int i = 3; i >= 0; --i) {
    const int src = i - ws;
    uint64_t v = 0;
    if (src >= 0) {
      v = x.w[src] << bs;
      if (bs != 0 && src >= 1) v |= x.w[src - 1] >> (64 - bs);
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
  int top = L - 1;
  while (top >= 0 && x[top] == 0) --top;
  if (top < 0) return 0.0;
  const int bl = top * 32 + (32 - __clz(static_cast<int>(x[top])));
  uint64_t t;  // the top 64 bits of mag (exact when bl <= 64), lsb weight 2^(bl - 64 + e)
  bool sticky = false;
  if (bl <= 64) {
    uint64_t lo = x[0];
    if (L > 1) lo |= static_cast<uint64_t>(x[L > 1 ? 1 : 0]) << 32;
    t = lo << (64 - bl);
  } else {
    const int s = bl - 64, ws = s / 32, bs = s % 32;
    uint32_t w0 = 0, w1 = 0, w2 = 0;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i == ws) w0 = x[i];
      if (i == ws + 1) w1 = x[i];
      if (i == ws + 2) w2 = x[i];
      if (i < ws && x[i] != 0) sticky = true;
    }
    if (bs != 0 && (w0 & ((1u << bs) - 1)) != 0) sticky = true;
    const uint64_t lo = static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32);
    t = (lo >> bs) | (bs != 0 ? static_cast<uint64_t>(w2) << (64 - bs) : 0);
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
template <int L>
__device__ __forceinline__ void crt(const uint8_t* W, int64_t plane, int N, uint32_t (&mag)[L],
                                    bool* neg) {
  using imd::device::c_tables;
  uint32_t x[L];
#pragma unroll
  for (int w = 0; w < L; ++w) x[w] = 0;
  for (int l = 0; l < N; ++l) {
    const uint32_t p = c_tables.moduli[l];
    const uint32_t y = imd::device::mod_p_small(c_tables.q[N][l] * W[l * plane], p,
                                                c_tables.epi_magic[l]);
    uint64_t carry = 0;
#pragma unroll
    for (int w = 0; w < L; ++w) {
      const uint64_t t = uint64_t{x[w]} + uint64_t{c_tables.M[N][l][w]} * y + carry;
      x[w] = static_cast<uint32_t>(t);
      carry = t >> 32;
    }
    // carry == 0: the peak x + M_l y < 2P <= 2^(32 L) (cert_crt_peak)
    bool ge = true;  // x >= P
#pragma unroll
    for (int w = L - 1; w >= 0; --w) {
      if (x[w] != c_tables.P[N][w]) {
        ge = x[w] > c_tables.P[N][w];
        break;
      }
    }
    if (ge) {
      uint64_t borrow = 0;
#pragma unroll
      for (int w = 0; w < L; ++w) {
        const uint64_t t = uint64_t{x[w]} - c_tables.P[N][w] - borrow;
        x[w] = static_cast<uint32_t>(t);
        borrow = (t >> 32) & 1;
      }
    }
  }
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
