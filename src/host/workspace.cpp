#include "workspace.hpp"

#include "checked.hpp"
#include "k_hat.hpp"
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

// Cumulative 256-aligned offsets over the present regions (nullopt entries and 0 are absent).
std::optional<WorkspaceLayout> place(const std::optional<uint64_t> (&bytes)[static_cast<int>(Region::count)]) {
  WorkspaceLayout L;
  uint64_t off = 0;
  for (int r = 0; r < static_cast<int>(Region::count); ++r) {
    if (!bytes[r] || *bytes[r] == 0) continue;
    L.offset[r] = off;
    L.bytes[r] = *bytes[r];
    uint64_t aligned = 0;
    if (!align_u64(*bytes[r], imd::constants::kRegionAlign, &aligned) || !add_u64(off, aligned, &off))
      return std::nullopt;
  }
  L.total = off;
  return L;
}

std::optional<uint64_t> vec_bytes(int64_t m, int64_t n, int64_t k, int64_t b) {
  // vec = 8 (2m + 2n + 2k + 3b)
  const std::optional<uint64_t> terms[4] = {product(2, m), product(2, n), product(2, k), product(3, b)};
  uint64_t vec = 0;
  for (const auto& t : terms)
    if (!t || !add_u64(vec, *t, &vec)) return std::nullopt;
  if (!mul_u64(vec, 8, &vec)) return std::nullopt;
  return vec;
}

}  // namespace

std::optional<WorkspaceLayout> hw_workspace_layout(const HwWorkspaceShape& s) {
  if (s.b == 0 || s.m == 0 || s.n == 0) return WorkspaceLayout{};
  const uint64_t k_hat = (static_cast<uint64_t>(s.k) + (kHwQ - 1)) / kHwQ * kHwQ;  // k < 2^63
  std::optional<uint64_t> bytes[static_cast<int>(Region::count)] = {};
  // Sized whenever needed, even when resident: an overflowing vec is INVALID_VALUE either way.
  if (s.need_vec) {
    const auto v = vec_bytes(s.m, s.n, s.k, s.b);
    if (!v) return std::nullopt;
    if (!s.vec_resident) bytes[static_cast<int>(Region::vec)] = v;
  }
  if (s.pack_a && !(bytes[static_cast<int>(Region::PA)] = product(8, s.b, s.m, k_hat)))
    return std::nullopt;
  if (s.pack_b && !(bytes[static_cast<int>(Region::PB)] = product(8, s.b, s.n, k_hat)))
    return std::nullopt;
  return place(bytes);
}

std::optional<WorkspaceLayout> workspace_layout(const WorkspaceShape& s) {
  namespace c = imd::constants;
  WorkspaceLayout L;
  if (s.k_align != kKAlign16 && s.k_align != kKAlign64) return std::nullopt;
  if (s.b == 0 || s.m == 0 || s.n == 0) return L;  // empty output: no-op

  // k_hat = ceil(k)_A (k_hat.hpp, A = the plan's resolved k_align), n_hat = ceil(n)_16; k, n <
  // 2^63 so these cannot overflow uint64.
  uint64_t k_hat = 0;
  if (!k_hat_of(static_cast<uint64_t>(s.k), static_cast<uint64_t>(s.k_align), &k_hat))
    return std::nullopt;
  const uint64_t n_hat = (static_cast<uint64_t>(s.n) + (c::kNAlign - 1)) & ~uint64_t{c::kNAlign - 1};

  const auto vec = vec_bytes(s.m, s.n, s.k, s.b);
  if (!vec) return std::nullopt;

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
    for (int r = 0; r <= static_cast<int>(Region::W); ++r)  // PA, PB: HWGEMM only
      if (!bytes[r]) return std::nullopt;
  }
  // v1.2 IMD_VEC_RESIDENT: the vectors are plan memory; still sized above, so an
  // overflowing vec is INVALID_VALUE in both modes (the plan would need those bytes).
  if (s.vec_resident) bytes[static_cast<int>(Region::vec)] = std::nullopt;
  // v1.7 (§6b.6): a packed operand's shifts, flags and residues are in the caller's packed
  // buffer (sized above all the same: an overflow is INVALID_VALUE as for the in-line plan).
  if (s.packed & 1)
    for (const Region r : {Region::sA, Region::fA, Region::RA}) bytes[static_cast<int>(r)] = std::nullopt;
  if (s.packed & 2)
    for (const Region r : {Region::sB, Region::fB, Region::RB}) bytes[static_cast<int>(r)] = std::nullopt;

  return place(bytes);
}

}  // namespace imd::detail
