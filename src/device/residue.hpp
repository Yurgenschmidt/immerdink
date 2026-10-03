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

inline constexpr ByteWeights kByteWeights = make_byte_weights(imd::constants::kTables);

}  // namespace imd::device
