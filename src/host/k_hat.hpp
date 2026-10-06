// The K extent and row stride of every residue row (DESIGN §4): k_hat = ceil(k)_A. The single
// source of this rounding in libimmerdink_host (workspace sizing, the plan's GEMM view, the
// pack layout and the pack lowering); spec/params.py k_hat_of mirrors it. A is the plan's K
// alignment, a power of two >= 16 (CUTLASS's 16-byte pointer rule); today every plan uses
// kKHatAlign = 16 (DESIGN v1.13 §4b: a later stage resolves A per plan).
#pragma once

#include <cstdint>

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {

inline constexpr int64_t kKHatAlign = imd::constants::kKAlign;

// ceil(k)_a; false when it does not fit uint64.
inline bool k_hat_of(uint64_t k, uint64_t a, uint64_t* k_hat) { return align_u64(k, a, k_hat); }

// ceil(k)_a for 0 <= k; false when it does not fit int64.
inline bool k_hat_of(int64_t k, int64_t a, int64_t* k_hat) {
  uint64_t u = 0;
  if (k < 0 || !align_u64(static_cast<uint64_t>(k), static_cast<uint64_t>(a), &u) ||
      u > static_cast<uint64_t>(INT64_MAX))
    return false;
  *k_hat = static_cast<int64_t>(u);
  return true;
}

}  // namespace imd::detail
