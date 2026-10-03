// Overflow-checked integer helpers (DESIGN §4a / §5 row 5 report overflow as a status).
#pragma once

#include <cstdint>

namespace imd::detail {

__extension__ typedef unsigned __int128 u128;  // GNU extension (g++, nvcc host side)
__extension__ typedef __int128 i128;

// |x| as uint64, defined for INT64_MIN.
constexpr uint64_t abs_u64(int64_t x) {
  return x < 0 ? uint64_t{0} - static_cast<uint64_t>(x) : static_cast<uint64_t>(x);
}

// Each returns false on overflow (and then leaves *r unspecified).
inline bool mul_i64(int64_t a, int64_t b, int64_t* r) { return !__builtin_mul_overflow(a, b, r); }
inline bool add_i64(int64_t a, int64_t b, int64_t* r) { return !__builtin_add_overflow(a, b, r); }
inline bool mul_u64(uint64_t a, uint64_t b, uint64_t* r) { return !__builtin_mul_overflow(a, b, r); }
inline bool add_u64(uint64_t a, uint64_t b, uint64_t* r) { return !__builtin_add_overflow(a, b, r); }

// ceil(x / a) * a for a power of two a.
inline bool align_u64(uint64_t x, uint64_t a, uint64_t* r) {
  return add_u64(x, a - 1, r) && ((*r &= ~(a - 1)), true);
}

}  // namespace imd::detail
