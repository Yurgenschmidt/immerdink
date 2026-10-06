// Independent C++ re-check of the generated constant header against its definitions
// (DESIGN §2, §3.4, §3.5): multi-limb arithmetic, no Python. Guards against a broken
// emitter (wrong limb order, truncation, misplaced rows) rather than re-proving params.py.
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "immerdink/detail/constants.hpp"

namespace {

namespace c = imd::constants;
using Limbs = std::vector<uint32_t>;  // little-endian, arbitrary length
__extension__ typedef unsigned __int128 u128;

int failures = 0;
#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      if (++failures <= 25) {                               \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                  \
        std::fprintf(stderr, "\n");                         \
      }                                                     \
    }                                                       \
  } while (0)

Limbs trim(Limbs x) {
  while (!x.empty() && x.back() == 0) x.pop_back();
  return x;
}
Limbs from_row(const uint32_t (&row)[c::kMaxLimbs]) { return trim(Limbs(row, row + c::kMaxLimbs)); }

Limbs mul(const Limbs& a, const Limbs& b) {
  Limbs r(a.size() + b.size() + 1, 0);
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < b.size(); ++j) {
      const uint64_t t = uint64_t{a[i]} * b[j] + r[i + j] + carry;
      r[i + j] = uint32_t(t);
      carry = t >> 32;
    }
    for (size_t k = i + b.size(); carry; ++k) {
      const uint64_t t = uint64_t{r[k]} + carry;
      r[k] = uint32_t(t);
      carry = t >> 32;
    }
  }
  return trim(r);
}
Limbs add(const Limbs& a, const Limbs& b) {
  Limbs r(std::max(a.size(), b.size()) + 1, 0);
  uint64_t carry = 0;
  for (size_t i = 0; i < r.size(); ++i) {
    const uint64_t t = uint64_t{i < a.size() ? a[i] : 0} + (i < b.size() ? b[i] : 0) + carry;
    r[i] = uint32_t(t);
    carry = t >> 32;
  }
  return trim(r);
}
int cmp(const Limbs& a, const Limbs& b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (size_t i = a.size(); i-- > 0;)
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}
uint32_t mod_small(const Limbs& a, uint32_t p) {
  uint64_t r = 0;
  for (size_t i = a.size(); i-- > 0;) r = ((r << 32) | a[i]) % p;
  return uint32_t(r);
}
Limbs pow2(int e) {
  Limbs r(size_t(e / 32) + 1, 0);
  r[size_t(e / 32)] = 1u << (e % 32);
  return r;
}
size_t bit_length(const Limbs& a) {
  if (a.empty()) return 0;
  return 32 * (a.size() - 1) + (32 - size_t(__builtin_clz(a.back())));
}

void check_moduli() {
  const auto& t = c::kTables;
  for (int l = 0; l < c::kNMax; ++l) {
    CHECK(t.moduli[l] >= 2 && t.moduli[l] <= 256, "p_%d = %u", l, t.moduli[l]);
    for (int j = 0; j < l; ++j)
      CHECK(std::gcd(t.moduli[l], t.moduli[j]) == 1, "gcd(p_%d, p_%d) != 1", l, j);
    if (l > 0) CHECK(t.moduli[l] < t.moduli[l - 1], "moduli not descending at %d", l);
  }
  CHECK(t.moduli[0] == 256 && t.moduli[c::kNMax - 1] == 173, "moduli table ends");
}

void check_per_n() {
  const auto& t = c::kTables;
  for (int N = 0; N <= c::kNMax; ++N) {
    if (N < c::kNMin) {
      CHECK(from_row(t.P[N]).empty() && t.L32[N] == 0, "row %d not zero", N);
      continue;
    }
    Limbs P{1};
    for (int l = 0; l < N; ++l) P = mul(P, {t.moduli[l]});
    CHECK(cmp(P, from_row(t.P[N])) == 0, "N=%d: P != prod p_l", N);
    // H = P/2 - 1  <=>  2(H + 1) = P
    CHECK(cmp(mul(add(from_row(t.H[N]), {1}), {2}), P) == 0, "N=%d: H != P/2 - 1", N);
    // T = isqrt(H): T^2 <= H < (T+1)^2
    const Limbs T = from_row(t.T[N]), H = from_row(t.H[N]), T1 = add(T, {1});
    CHECK(cmp(mul(T, T), H) <= 0 && cmp(H, mul(T1, T1)) < 0, "N=%d: T != isqrt(H)", N);
    // L32 minimal with 2P <= 2^(32 L32)
    const Limbs P2 = mul(P, {2});
    CHECK(cmp(P2, pow2(32 * t.L32[N])) <= 0, "N=%d: 2P > 2^(32 L32)", N);
    CHECK(t.L32[N] == 1 || cmp(P2, pow2(32 * (t.L32[N] - 1))) > 0, "N=%d: L32 not minimal", N);
    CHECK(t.L32[N] <= c::kMaxLimbs, "N=%d: L32 > kMaxLimbs", N);
    // eps_max = floor(log2 T)
    CHECK(int(bit_length(T)) - 1 == t.eps_max[N], "N=%d: eps_max", N);
    for (int l = 0; l < c::kNMax; ++l) {
      const Limbs M = from_row(t.M[N][l]);
      if (l >= N) {
        CHECK(M.empty() && t.q[N][l] == 0, "N=%d l=%d: padding not zero", N, l);
        continue;
      }
      const uint32_t p = t.moduli[l], q = t.q[N][l];
      CHECK(cmp(mul(M, {p}), P) == 0, "N=%d l=%d: M_l p_l != P", N, l);
      CHECK(q >= 1 && q < p, "N=%d l=%d: q_l range", N, l);
      CHECK(uint64_t{mod_small(M, p)} * q % p == 1, "N=%d l=%d: M_l q_l != 1 mod p_l", N, l);
      for (int j = 0; j < N; ++j)
        if (j != l) CHECK(mod_small(M, t.moduli[j]) == 0, "N=%d l=%d j=%d: p_j does not divide M_l", N, l, j);
    }
  }
  CHECK(c::kMaxLimbs == c::kTables.L32[c::kNMax], "kMaxLimbs");
  CHECK(c::kEpsMax == c::kTables.eps_max[c::kNMax], "kEpsMax");
}

void check_epilogue() {
  const auto& t = c::kTables;
  // v1.13 §4b: K_c' = floor(K_c/64)*64 = 131008, the largest multiple of the 64-wide K tile
  CHECK(c::kKc == ((int64_t{1} << 31) - 1) / (128 * 128) && c::kKChunkAlign == 64 &&
            c::kKcPrime == c::kKc / c::kKChunkAlign * c::kKChunkAlign && c::kKcPrime == 131008 &&
            c::kKcPrime % 16 == 0 && c::kKc - c::kKcPrime < c::kKChunkAlign,
        "K_c, K_c'");
  CHECK(t.epi_vmin[0] == -2129666048 && t.epi_vmax[0] == 2146435327, "p=256 range literals");
  for (int l = 0; l < c::kNMax; ++l) {
    const int64_t p = t.moduli[l];
    const int64_t lo = -(p / 2), hi = (p + 1) / 2 - 1;  // §3.4 symmetric residue range
    const int64_t pmax = std::max(lo * lo, hi * hi), pmin = lo * hi;
    CHECK(t.epi_vmin[l] == pmin * c::kKcPrime, "p=%lld vmin", (long long)p);
    CHECK(t.epi_vmax[l] == pmax * c::kKcPrime + p - 1, "p=%lld vmax", (long long)p);
    if (p == c::kModMaskModulus) {
      CHECK(t.epi_bias[l] == 0 && t.epi_magic[l] == 0, "p=256 uses the mask");
      continue;
    }
    const int64_t want_bias = p * (((int64_t{1} << 31) + p - 1) / p);
    CHECK(int64_t{t.epi_bias[l]} == want_bias, "p=%lld bias", (long long)p);
    const u128 mp = u128{t.epi_magic[l]} * u128(p);
    const u128 two64 = u128{1} << 64;
    CHECK(mp >= two64 && mp - two64 < u128(p), "p=%lld magic != ceil(2^64/p)", (long long)p);
    // n = v + bias stays in [0, 2^32) over the reachable v.
    CHECK(t.epi_vmin[l] + want_bias >= 0 && t.epi_vmax[l] + want_bias < (int64_t{1} << 32),
          "p=%lld: reachable n leaves [0, 2^32)", (long long)p);
  }
}

void check_pow2() {
  const auto& t = c::kTables;
  for (int l = 0; l < c::kNMax; ++l) {
    uint32_t v = 1 % t.moduli[l];
    for (int e = 0; e <= c::kEpsMax; ++e) {
      CHECK(t.pow2_mod[l][e] == v, "2^%d mod %u", e, t.moduli[l]);
      v = 2 * v % t.moduli[l];
    }
  }
}

}  // namespace

int main() {
  check_moduli();
  check_per_n();
  check_epilogue();
  check_pow2();
  std::printf("constant tables: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
