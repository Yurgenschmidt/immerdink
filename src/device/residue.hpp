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

}  // namespace imd::device
