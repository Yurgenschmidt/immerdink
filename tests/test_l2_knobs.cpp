// Issue 21 (docs/NOTES-21.md): K1's two L2 budgets are fractions of the card's L2. On the RTX 4090 (72 MiB) they must equal the constants
// they replaced (40 MiB and 64 MiB) EXACTLY, so that every 4090 launch is unchanged; on other cards they scale with the L2 and with
// nothing else. CPU only.
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "analysis.hpp"

namespace {
int failures = 0;
#define CHECK(x)                                                         \
  do {                                                                   \
    if (!(x)) {                                                          \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);         \
      ++failures;                                                        \
    }                                                                    \
  } while (0)
constexpr double MiB = 1048576.0;
}  // namespace

int main() {
  using namespace imd::detail;
  const int64_t l2_4090 = 75497472;     // cudaDevAttrL2CacheSize of the RTX 4090
  CHECK(k1_l2_budget_bytes(l2_4090) == 64.0 * MiB);
  CHECK(k1_l2_fit_bytes(l2_4090) == 40.0 * MiB);
  CHECK(k1_l2_budget_bytes(0) == 64.0 * MiB && k1_l2_fit_bytes(0) == 40.0 * MiB);     // unknown: the reference
  CHECK(k1_l2_budget_bytes(-5) == 64.0 * MiB);
  // the other cards (L2 sizes of issue 21): proportional
  struct Card { const char* name; int64_t l2; double budget_mib, fit_mib; } cards[] = {
      {"H100", int64_t{50} << 20, 50.0 * 8 / 9, 50.0 * 5 / 9},
      {"H200", int64_t{60} << 20, 60.0 * 8 / 9, 60.0 * 5 / 9},
      {"RTX PRO 6000", int64_t{128} << 20, 128.0 * 8 / 9, 128.0 * 5 / 9}};
  for (const Card& c : cards) {
    CHECK(std::fabs(k1_l2_budget_bytes(c.l2) / MiB - c.budget_mib) < 1e-9);
    CHECK(std::fabs(k1_l2_fit_bytes(c.l2) / MiB - c.fit_mib) < 1e-9);
    CHECK(k1_l2_fit_bytes(c.l2) < k1_l2_budget_bytes(c.l2) && k1_l2_budget_bytes(c.l2) < double(c.l2));
  }
  // monotone in the L2 size
  double pb = 0, pf = 0;
  for (int64_t l2 = int64_t{1} << 20; l2 <= int64_t{256} << 20; l2 += int64_t{1} << 20) {
    CHECK(k1_l2_budget_bytes(l2) > pb && k1_l2_fit_bytes(l2) > pf);
    pb = k1_l2_budget_bytes(l2), pf = k1_l2_fit_bytes(l2);
  }
  if (failures) return 1;
  std::printf("PASS  K1's L2 budgets: the 4090 gives 40 / 64 MiB exactly; H100, H200, RTX PRO 6000 scale with their L2\n");
  return 0;
}
