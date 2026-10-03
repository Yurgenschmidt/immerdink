#include "workspace.hpp"

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {
namespace {

// Product of nonnegative int64 factors as uint64, nullopt on overflow.
template <class... Ts>
std::optional<uint64_t> product(Ts... xs) {
  uint64_t r = 1;
  for (const uint64_t x : {static_cast<uint64_t>(xs)...})
    if (!mul_u64(r, x, &r)) return std::nullopt;
  return r;
}

}  // namespace

std::optional<WorkspaceLayout> workspace_layout(const WorkspaceShape& s) {
  namespace c = imd::constants;
  WorkspaceLayout L;
  if (s.b == 0 || s.m == 0 || s.n == 0) return L;  // empty output: no-op

  const uint64_t kAlign = c::kRegionAlign;
  // k_hat = ceil(k)_16, n_hat = ceil(n)_16; k, n < 2^63 so these cannot overflow uint64.
  const uint64_t k_hat = (static_cast<uint64_t>(s.k) + (c::kKAlign - 1)) & ~uint64_t{c::kKAlign - 1};
  const uint64_t n_hat = (static_cast<uint64_t>(s.n) + (c::kNAlign - 1)) & ~uint64_t{c::kNAlign - 1};

  // vec = 8 (2m + 2n + 2k + 3b)
  std::optional<uint64_t> vec_terms[4] = {product(2, s.m), product(2, s.n), product(2, s.k),
                                          product(3, s.b)};
  uint64_t vec = 0;
  for (const auto& t : vec_terms)
    if (!t || !add_u64(vec, *t, &vec)) return std::nullopt;
  if (!mul_u64(vec, 8, &vec)) return std::nullopt;

  std::optional<uint64_t> bytes[static_cast<int>(Region::count)] = {};
  if (s.direct || s.k <= 1) {
    bytes[static_cast<int>(Region::vec)] = vec;
  } else {
    const auto g = static_cast<uint64_t>(s.residue_group);
    const auto N = static_cast<uint64_t>(s.num_moduli);
    bytes[static_cast<int>(Region::sA)] = product(2, s.b, s.m);
    bytes[static_cast<int>(Region::sB)] = product(2, s.b, s.n);
    bytes[static_cast<int>(Region::fA)] = product(s.b, s.m);
    bytes[static_cast<int>(Region::fB)] = product(s.b, s.n);
    bytes[static_cast<int>(Region::vec)] = vec;
    bytes[static_cast<int>(Region::RA)] = product(g, s.b, s.m, k_hat);
    bytes[static_cast<int>(Region::RB)] = product(g, s.b, n_hat, k_hat);
    bytes[static_cast<int>(Region::W)] = product(N, s.b, s.m, n_hat);
    for (const auto& x : bytes)
      if (!x) return std::nullopt;
  }

  uint64_t off = 0;
  for (int r = 0; r < static_cast<int>(Region::count); ++r) {
    if (!bytes[r] || *bytes[r] == 0) continue;  // absent (regions are never empty here)
    L.offset[r] = off;
    L.bytes[r] = *bytes[r];
    uint64_t aligned = 0;
    if (!align_u64(*bytes[r], kAlign, &aligned) || !add_u64(off, aligned, &off))
      return std::nullopt;
  }
  L.total = off;
  return L;
}

}  // namespace imd::detail
