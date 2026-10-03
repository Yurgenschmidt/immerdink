#include "k4_config.hpp"

#include <cstdint>

#include "analysis.hpp"
#include "checked.hpp"

namespace imd::detail {

int32_t k4_select(const ExecPlan& e, int32_t sms) {
  if (!emulates(e)) return IMD_K4_128x256;
  (void)sms;
  return IMD_K4_128x256;
}

uint64_t k4_tune_bytes(const ExecPlan& e) {
  if (!emulates(e)) return 0;
  // Group 1 (g moduli): RA g b m k_hat, RB g b n_hat k_hat, W g b m n_hat, each 256-aligned
  // inside one allocation. Each region fits uint64 (the §4a layout does), the sum may not.
  const uint64_t gb = static_cast<uint64_t>(e.g) * static_cast<uint64_t>(e.b);
  const uint64_t m = static_cast<uint64_t>(e.m), nh = static_cast<uint64_t>(e.n_hat),
                 kh = static_cast<uint64_t>(e.k_hat);
  uint64_t total = 0;
  for (const uint64_t rows_cols : {m * kh, nh * kh, m * nh}) {
    uint64_t r = 0;
    if (!mul_u64(gb, rows_cols, &r) || r > kK4TuneMaxBytes) return 0;
    total += (r + 255) / 256 * 256;
  }
  return total <= kK4TuneMaxBytes ? total : 0;
}

}  // namespace imd::detail
