// Byte weights for mod_p_bytes (device/modp.hpp), derived at compile time from the
// generated table pow2_mod (2^e mod p_l, e = 0..kEpsMax >= 48): no new constants.
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

}  // namespace imd::device
