#include "k4_config.hpp"

#include <algorithm>
#include <cstdint>

#include "analysis.hpp"
#include "checked.hpp"

namespace imd::detail {
namespace {

// Saturating unsigned 128-bit arithmetic: every value is clamped to kSat, so the estimate
// below is the same exact integer function in any implementation that clamps the same way
// (tests/test_host.py transcribes it in Python).
constexpr u128 kSat = (u128{1} << 127) - 1;
u128 mul_sat(u128 a, u128 b) {
  u128 r = 0;
  return __builtin_mul_overflow(a, b, &r) || r > kSat ? kSat : r;
}
u128 add_sat(u128 a, u128 b) { return a > kSat - b ? kSat : a + b; }
u128 div_up(u128 a, u128 b) { return a / b + (a % b != 0); }

}  // namespace

K4Rounds k4_rounds(int c, int64_t m, int64_t n_hat, int64_t k_hat, int64_t gb, int32_t sms) {
  const K4ConfigDesc& d = kK4Configs[c];
  // Tiles of one launch over a group's g b batches; the busiest SM gets j of them: j / occ
  // full rounds of occ concurrent CTAs, then a round of the remaining j mod occ.
  const u128 tiles = mul_sat(mul_sat(div_up(static_cast<u128>(m), d.tile_m),
                                     div_up(static_cast<u128>(n_hat), d.tile_n)),
                             static_cast<u128>(gb));
  const u128 j = div_up(tiles, static_cast<u128>(sms));
  const u128 occ = static_cast<u128>(d.ctas_per_sm);
  K4Rounds r;
  r.full = j / occ;
  r.rest = static_cast<int32_t>(j % occ);
  r.tile_macs = mul_sat(static_cast<u128>(d.tile_m) * d.tile_n, static_cast<u128>(k_hat));
  return r;
}

u128 k4_estimate_fs(int c, int64_t m, int64_t n_hat, int64_t k_hat, int64_t gb, int32_t sms) {
  const K4Cost& cost = kK4Costs[c];
  const K4Rounds r = k4_rounds(c, m, n_hat, k_hat, gb, sms);
  const auto round_fs = [&](int q) {  // one round of q concurrent CTAs
    return add_sat(mul_sat(static_cast<u128>(cost.round_ns), 1000000),
                   mul_sat(mul_sat(r.tile_macs, static_cast<u128>(q)), static_cast<u128>(cost.mac_fs[q])));
  };
  u128 est = r.full == 0 ? 0 : mul_sat(r.full, round_fs(kK4Configs[c].ctas_per_sm));
  if (r.rest != 0) est = add_sat(est, round_fs(r.rest));
  return est;
}

int32_t k4_select(const ExecPlan& e, int32_t sms) {
  if (!emulates(e)) return IMD_K4_128x256;
  if (sms < 1) sms = kK4ReferenceSms;
  const int64_t gb = static_cast<int64_t>(std::min(e.g, e.N)) * e.b;  // fits: the RA region does
  const u128 base = k4_estimate_fs(IMD_K4_128x256, e.m, e.n_hat, e.k_hat, gb, sms);
  // A smaller tile is taken only when its estimate beats 128x256's by the margin; among
  // those, the smallest estimate (ties: the lower id).
  int32_t best = IMD_K4_128x256;
  u128 best_est = base;
  for (int32_t c = 1; c < kK4ConfigCount; ++c) {
    const u128 est = k4_estimate_fs(c, e.m, e.n_hat, e.k_hat, gb, sms);
    if (mul_sat(est, 1000) < mul_sat(base, 1000 - kK4MarginPermille) && est < best_est)
      best = c, best_est = est;
  }
  return best;
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
