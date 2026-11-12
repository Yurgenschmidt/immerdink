// Byte weights for mod_p_bytes (device/modp.hpp) and (v1.14) the pack emitter's
// mod_p_words32, derived at compile time from the generated table pow2_mod (2^e mod p_l,
// e = 0..kEpsMax): no new constants.
#pragma once

#include <cstdint>

#include "immerdink/detail/constants.hpp"

namespace imd::device {

struct ByteWeights {
  uint32_t lo[imd::constants::kNMax];  // u8 lanes: 2^0, 2^8, 2^16, 2^24 mod p_l
  uint32_t hi[imd::constants::kNMax];  // u8 lanes: 2^32, 2^40, 2^48 mod p_l, 0
};

constexpr ByteWeights make_byte_weights(const imd::constants::Tables& t) {
  static_assert(imd::constants::kEpsMax >= 48, "pow2_mod must reach 2^48");
  ByteWeights w{};
  for (int l = 0; l < imd::constants::kNMax; ++l) {
    for (int i = 0; i < 4; ++i) w.lo[l] |= static_cast<uint32_t>(t.pow2_mod[l][8 * i]) << (8 * i);
    for (int i = 0; i < 3; ++i)
      w.hi[l] |= static_cast<uint32_t>(t.pow2_mod[l][32 + 8 * i]) << (8 * i);
  }
  return w;
}

// ceil(2^32 / p_l) for mod_p_small32, from epi_magic = ceil(2^64 / p_l):
// ceil(ceil(2^64/p) / 2^32) = ceil(2^32/p). For p = 256 it is 2^24 = 2^32 / 256 exactly, so
// umulhi(n, 2^24) = n >> 8 = floor(n / 256) for every n < 2^32: mod_p_small32 reduces p = 256
// by its mask, mod_p_bytes32 (A5) by the same multiply-high as the other moduli.
struct Magic32 {
  uint32_t m[imd::constants::kNMax];
};
constexpr Magic32 make_magic32(const imd::constants::Tables& t) {
  Magic32 r{};
  for (int l = 0; l < imd::constants::kNMax; ++l)
    r.m[l] = t.moduli[l] == 256u ? (1u << 24)
                                 : static_cast<uint32_t>((t.epi_magic[l] >> 32) +
                                                         ((t.epi_magic[l] & 0xFFFFFFFFull) != 0));
  return r;
}
inline constexpr Magic32 kMagic32 = make_magic32(imd::constants::kTables);

inline constexpr ByteWeights kByteWeights = make_byte_weights(imd::constants::kTables);

// v1.14 §3.4.1: the pack emitter's weights. Word j of modulus l packs the u8 lanes
// 2^(8(4j + i)) mod p_l, i = 0..3, for the bytes 0..kPackWeightBytes-1 = 0..9 of X = |a'|
// (X < 2^78, cert_pack_words); the lanes of bytes 10 and 11 are 0. Derived from pow2_mod.
struct PackWeights {
  uint32_t w[3][imd::constants::kNMax];
};
constexpr PackWeights make_pack_weights(const imd::constants::Tables& t) {
  static_assert(imd::constants::kEpsMax >= 72, "pow2_mod must reach 2^72 (byte 9 of X)");
  static_assert(imd::constants::kPackWeightBytes == 10 &&
                    8 * (imd::constants::kPackWeightBytes - 1) <= imd::constants::kEpsMax,
                "weights 2^0 .. 2^72 from pow2_mod");
  static_assert(imd::constants::kPackXBits3 <= 8 * imd::constants::kPackWeightBytes &&
                    imd::constants::kPackXBits2 < 64,
                "X < 2^(8 kPackWeightBytes); W = 2 keeps X below 2^63");
  PackWeights w{};
  for (int l = 0; l < imd::constants::kNMax; ++l)
    for (int i = 0; i < imd::constants::kPackWeightBytes; ++i)
      w.w[i / 4][l] |= static_cast<uint32_t>(t.pow2_mod[l][8 * i]) << (8 * (i % 4));
  return w;
}
inline constexpr PackWeights kPackWeights = make_pack_weights(imd::constants::kTables);

// ---- Sequence E1 (issue 21, docs/NOTES-21.md "Pack emitter and finish"): the fused residue emitter's
// per-(W, modulus) constants, derived at compile time from the tables above. ONE place for the math:
//   d  = sum_j dp4a(x_j, w_j) = X (mod p), 0 <= d <= dmax = 255 (sum of the weight lanes)
//   s' = sigma d + C        sigma = +1 / -1, C = p ceil(dmax / p): a multiple of p >= dmax, so s' >= 0 and s' = sigma d (mod p)
//   q  = hi32(s' m32 + h m32)   h = floor(p / 2): the 32-bit magic m32 = ceil(2^32 / p) (2^24 for p = 256) gives
//                               floor((s' + h) / p) exactly for s' + h < 2^24 (A5: e = m32 p - 2^32 < p <= 256 so (s' + h) e < 2^32)
//   u  = s' - p q = ((s' + h) mod p) - h = SMod(sigma d, p), the signed residue; its low byte is the int8.
// Hardware-specific part (not the math): the 64-bit addend h m32 rides in the multiply-add (IMAD.HI.U32 with a register pair;
// measured with nvcc 13.4 on sm_89, sm_90, sm_120, docs/NOTES-21.md). A walk-back to the plain sequence (q = umulhi(s, m32),
// r = s - p q, the SMod compare, the sign select) is the baseline emitter in kernels/pack_emitter.cuh, which stays.
struct PackEmit {            // 32 bytes: the emitter reads it with two 16-byte constant loads
  uint32_t w[3];             // the pack weights (u8 lanes), w[2] = 0 for W = 2
  uint32_t m32;              // Magic32
  uint64_t hm;               // floor(p / 2) m32
  uint32_t C;                // p ceil(dmax / p)
  uint32_t negp;             // 2^32 - p, so s' + negp q = s' - p q
};
struct alignas(16) PackEmitTable {
  PackEmit e[2][imd::constants::kNMax];  // [W - 2][l]
};
constexpr PackEmitTable make_pack_emit(const imd::constants::Tables& t) {
  PackEmitTable r{};
  for (int W = 2; W <= 3; ++W)
    for (int l = 0; l < imd::constants::kNMax; ++l) {
      PackEmit c{};
      const uint32_t p = t.moduli[l];
      uint32_t dmax = 0;
      for (int j = 0; j < W; ++j) {
        c.w[j] = kPackWeights.w[j][l];
        for (int i = 0; i < 4; ++i) dmax += 255u * ((c.w[j] >> (8 * i)) & 255u);
      }
      c.m32 = kMagic32.m[l];
      c.hm = uint64_t{p / 2} * c.m32;
      c.C = p * ((dmax + p - 1) / p);
      c.negp = 0u - p;
      r.e[W - 2][l] = c;
    }
  return r;
}
inline constexpr PackEmitTable kPackEmit = make_pack_emit(imd::constants::kTables);
// The bounds the math needs, checked at compile time for every W and modulus: s' + h < 2^24 (the A5 range; also
// s' m32 + h m32 < 2^64, trivially), C >= dmax (s' >= 0).
constexpr bool pack_emit_bounds_ok(const PackEmitTable& t) {
  for (int W = 2; W <= 3; ++W)
    for (int l = 0; l < imd::constants::kNMax; ++l) {
      const PackEmit& c = t.e[W - 2][l];
      const uint32_t p = 0u - c.negp;
      uint32_t dmax = 0;
      for (int j = 0; j < W; ++j)
        for (int i = 0; i < 4; ++i) dmax += 255u * ((c.w[j] >> (8 * i)) & 255u);
      if (c.C < dmax || c.C % p != 0 || c.C - dmax + 2 * dmax + p / 2 >= (1u << 24)) return false;
    }
  return true;
}
static_assert(pack_emit_bounds_ok(kPackEmit), "E1: C >= dmax, C multiple of p, C + dmax + h < 2^24");

// ---- Sequence O1 (issue 21, task 17): the offset-binary input of the E1 emitter, W = 2 only. The quantize writes
// Y = a' + 2^63 (the two's-complement a' with its top bit flipped; |a'| <= T(N) < 2^63 for W = 2, so 0 < Y < 2^64) instead of
// |a'| and a sign. The sign then needs no per-modulus multiply: with C2 = (-2^63) mod p as the dp4a chain's start value,
//   s' = C2 + sum_j dp4a(y_j, w_j) = Y - 2^63 = a' (mod p),  0 <= s' <= C2 + dmax < p + dmax,
// and q, u as for E1: u = SMod(a', p). p = 256: C2 = 0, the low byte of Y is the low byte of a'. Same struct as E1, C holds C2.
constexpr uint32_t pow2_63_mod(uint32_t p) {
  uint32_t r = 1 % p;
  for (int i = 0; i < 63; ++i) r = (2 * r) % p;
  return r;
}
struct alignas(16) PackEmitTableO {
  PackEmit e[imd::constants::kNMax];  // [l]; the W = 2 weights (meaningful for l < kPackW2MaxN)
};
constexpr PackEmitTableO make_pack_emit_o(const PackEmitTable& t2, const imd::constants::Tables& t) {
  PackEmitTableO r{};
  for (int l = 0; l < imd::constants::kNMax; ++l) {
    r.e[l] = t2.e[0][l];
    const uint32_t p = t.moduli[l];
    r.e[l].C = (p - pow2_63_mod(p)) % p;
  }
  return r;
}
inline constexpr PackEmitTableO kPackEmitO = make_pack_emit_o(kPackEmit, imd::constants::kTables);
// O1's bounds: C2 < p and C2 = -2^63 (mod p); s' + h <= C2 + dmax + floor(p / 2) < 2^24 (the A5 range of E1).
constexpr bool pack_emit_o_bounds_ok(const PackEmitTableO& t) {
  for (int l = 0; l < imd::constants::kNMax; ++l) {
    const PackEmit& c = t.e[l];
    const uint32_t p = 0u - c.negp;
    uint32_t dmax = 0;
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 4; ++i) dmax += 255u * ((c.w[j] >> (8 * i)) & 255u);
    if (c.w[2] != 0 || c.C >= p || (c.C + pow2_63_mod(p)) % p != 0 || c.C + dmax + p / 2 >= (1u << 24)) return false;
  }
  return true;
}
static_assert(pack_emit_o_bounds_ok(kPackEmitO), "O1: C2 = -2^63 mod p, C2 + dmax + h < 2^24");

// ---- Sequence FQ (issue 21, task 18; "F1" in the issue-21 findings, renamed here: F1 is the finish set, bit 4): the O1 emitter with its
// quotient in binary32. The emit loop of KL is fmaheavy-bound on H200, and O1's quotient q = hi32(s' m32 + h) is an IMAD.WIDE on that pipe;
// an FFMA can issue to fmalite too. The dp4a chain starts at C = 0x4B000000 + C3 with C3 = (C2 - 2^23) mod p, so the chain's sum s is the
// bit pattern of the binary32 T = 2^23 + s'' (s'' = C3 + d < 2^23: exponent 150, mantissa s'') and T = 2^23 + C3 + d = C2 + d = a' (mod p).
// Then qm = fma_rn(T, rcp, 1.5 * 2^23) with rcp = RN32(1/p) holds M + RN(T rcp) in [2^23, 2^24) (ulp 1; M = 1.5 * 2^23, 0x4B400000), and
//   u = s + negp * bits(qm) = (T - 2^23 + 0x4B000000) - p (0x4B400000 + q) = T - p q (mod 256),
// since 0x4B000000, 2^23 and 0x4B400000 are 0 mod 256: u's low byte is the int8 of T - p q, which is SMod(a', p) when q = RN(T / p).
// q = RN(T rcp) = RN(T / p) for every T in range (pack_emit_f_bounds_ok, an exact criterion; p = 256: rcp = 2^-8 is exact, and the tie
// T - 256 q = +-128 stores 0x80 either way). Same struct, C = 0x4B000000 + C3, m32 = the bits of rcp, hm = 0 (unused). W = 2, l < kPackW2MaxN.
constexpr uint32_t kFqBias = 0x4B000000u;  // the bits of 2^23
constexpr uint32_t kFqMagic = 0x4B400000u; // the bits of 1.5 * 2^23
// RN32(1/p) for 128 < p <= 256 as bits: 1/p = 2^-8 (2^8 / p), significand R = RN(2^31 / p) in [2^23, 2^24) (ties to even; p odd has none).
constexpr uint32_t rcp_bits(uint32_t p) {
  uint32_t R = (1u << 31) / p;
  const uint32_t rem = (1u << 31) % p;
  if (2 * rem > p || (2 * rem == p && (R & 1u) != 0)) ++R;
  return ((127u - 8u) << 23) | (R - (1u << 23));
}
struct alignas(16) PackEmitTableF {
  PackEmit e[imd::constants::kNMax];  // [l]; meaningful for l < kPackW2MaxN (the moduli 128 < p <= 256 the rcp form needs)
};
constexpr PackEmitTableF make_pack_emit_f(const PackEmitTableO& to, const imd::constants::Tables& t) {
  PackEmitTableF r{};
  for (int l = 0; l < imd::constants::kNMax; ++l) {
    r.e[l] = to.e[l];
    if (l >= imd::constants::kPackW2MaxN) continue;
    const uint32_t p = t.moduli[l];
    r.e[l].C = kFqBias + (to.e[l].C + p - (1u << 23) % p) % p;
    r.e[l].m32 = rcp_bits(p);
    r.e[l].hm = 0;
  }
  return r;
}
inline constexpr PackEmitTableF kPackEmitF = make_pack_emit_f(kPackEmitO, imd::constants::kTables);
// FQ's bounds and its quotient, exactly. With rcp = R 2^-31 and e = R p - 2^31, T rcp = T / p + T e / (p 2^31). For odd p, T / p is never a
// half-integer, and RN(T rcp) != RN(T / p) needs T rcp to reach the half-integer next to T / p: for T = p n + r the distance is |p - 2 r| / (2 p),
// so a wrong q needs T |e| >= 2^30 |p - 2 r| with r on the side e points to. The check: every T <= Tmax with r = (p -+ 1) / 2 (|p - 2r| = 1,
// the side of e) has T |e| < 2^30, and Tmax |e| < 3 * 2^30 (every other r). Tmax = 2^23 + C3 + dmax; also C3 + dmax < 2^23 (T's mantissa).
constexpr bool pack_emit_f_bounds_ok(const PackEmitTableF& t) {
  for (int l = 0; l < imd::constants::kPackW2MaxN; ++l) {
    const PackEmit& c = t.e[l];
    const uint32_t p = 0u - c.negp;
    uint32_t dmax = 0;
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 4; ++i) dmax += 255u * ((c.w[j] >> (8 * i)) & 255u);
    // odd p (T / p never a half-integer) or p = 256 (rcp exact): the criterion's premise, checked (review of 5c27189)
    if (p <= 128 || p > 256 || (p % 2 == 0 && p != 256) || c.w[2] != 0 || c.hm != 0 || c.C < kFqBias) return false;
    const uint32_t C3 = c.C - kFqBias;
    if (C3 >= p || (C3 + (1u << 23) % p + pow2_63_mod(p)) % p != 0 || C3 + dmax >= (1u << 23) || c.m32 != rcp_bits(p)) return false;
    const uint64_t R = (c.m32 & 0x7FFFFFu) | (1u << 23);
    const int64_t e = static_cast<int64_t>(R * p) - (int64_t{1} << 31);
    const uint64_t ae = static_cast<uint64_t>(e < 0 ? -e : e);
    const uint64_t Tmax = (1u << 23) + C3 + dmax;
    const uint64_t r = e > 0 ? (p - 1) / 2 : (p + 1) / 2;
    const uint64_t Tr = Tmax - (Tmax + p - r) % p;  // the largest T <= Tmax with T = r (mod p)
    if (Tr * ae >= (uint64_t{1} << 30) || Tmax * ae >= 3 * (uint64_t{1} << 30)) return false;
  }
  return true;
}
static_assert(pack_emit_f_bounds_ok(kPackEmitF), "FQ: C = 2^23 bits + C3, C3 = C2 - 2^23 mod p, C3 + dmax < 2^23, q = RN(T / p) on the whole range");

}  // namespace imd::device
