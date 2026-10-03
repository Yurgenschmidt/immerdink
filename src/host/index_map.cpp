#include "index_map.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>

#include "checked.hpp"

namespace imd::detail {
namespace {

constexpr int kA = 0, kB = 1, kC = 2;
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

// §5 step 9 for one label appended to a group's fused modes (canonical order): merge into the
// last mode iff s_X(new) = s_X(last) e(last) for every tensor X (stride-0 carriers included;
// a tensor that does not carry the group has stride 0 on both, where the test holds).
void fuse_append(std::vector<FusedMode>& f, const LabelInfo& li) {
  bool merge = !f.empty();
  for (int x = 0; merge && x < 3; ++x) {
    int64_t next = 0;
    merge = mul_i64(f.back().stride[x], f.back().extent, &next) && next == li.stride[x];
  }
  if (merge) {
    f.back().extent *= li.extent;  // a factor of the group size: no overflow
  } else {
    FusedMode fm;
    fm.extent = li.extent;
    for (int x = 0; x < 3; ++x) fm.stride[x] = li.stride[x];
    f.push_back(fm);
  }
}

}  // namespace

bool descriptor_ok(const imd_tensor_t& t) {
  if (t.rank < 0) return false;
  if (t.rank == 0) return true;
  if (t.extent == nullptr || t.stride == nullptr || t.label == nullptr) return false;
  for (int32_t d = 0; d < t.rank; ++d)
    if (t.extent[d] < 0) return false;
  return true;
}

bool tensor_empty(const imd_tensor_t& t) {
  for (int32_t d = 0; d < t.rank; ++d)
    if (t.extent[d] == 0) return true;
  return false;
}

namespace {

// §5 row 4, for a non-empty tensor: over the modes with extent > 1 sorted by |stride|
// (ties in mode order), |s_(t)| > sum_{u<t} |s_(u)| (e_(u) - 1). spec/IndexMap.tla
// StrideRule. The prefix sum saturates at 2^64, where the rule has already failed.
bool sorted_stride_rule(const imd_tensor_t& t) {
  std::vector<int32_t> modes;
  for (int32_t d = 0; d < t.rank; ++d)
    if (t.extent[d] > 1) modes.push_back(d);
  std::stable_sort(modes.begin(), modes.end(), [&](int32_t x, int32_t y) {
    return abs_u64(t.stride[x]) < abs_u64(t.stride[y]);
  });
  u128 below = 0;  // sum over earlier modes, capped at 2^64
  constexpr u128 kCap = static_cast<u128>(1) << 64;
  for (int32_t d : modes) {
    const uint64_t s = abs_u64(t.stride[d]);
    if (static_cast<u128>(s) <= below) return false;
    below += static_cast<u128>(s) * static_cast<uint64_t>(t.extent[d] - 1);
    below = std::min(below, kCap);
  }
  return true;
}

}  // namespace

// §5 row 5: the element span sum_d |s_d| (e_d - 1) of a non-empty tensor fits int64.
bool span_fits(const imd_tensor_t& t) {
  if (tensor_empty(t)) return true;
  int64_t span = 0;
  for (int32_t d = 0; d < t.rank; ++d) {
    if (t.extent[d] <= 1) continue;
    const uint64_t s = abs_u64(t.stride[d]);
    if (s > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
    int64_t term = 0;
    if (!mul_i64(static_cast<int64_t>(s), t.extent[d] - 1, &term)) return false;
    if (!add_i64(span, term, &span)) return false;
  }
  return true;
}

imd_status_t lower(const imd_tensor_t& A, const imd_tensor_t& B, const imd_tensor_t& C,
                   Lowered* out) {
  const imd_tensor_t* tensors[3] = {&A, &B, &C};
  for (const imd_tensor_t* t : tensors)
    if (!descriptor_ok(*t)) return IMD_STATUS_INVALID_VALUE;

  // Steps 1-2: one entry per distinct label; extents must agree (row 1); a label
  // repeated within A or B is a diagonal, its merged stride is the sum of its strides.
  std::vector<LabelInfo> labels;
  bool merged_stride_overflow = false;
  int pos = 0;
  for (int x = 0; x < 3; ++x) {
    const imd_tensor_t& t = *tensors[x];
    for (int32_t d = 0; d < t.rank; ++d, ++pos) {
      auto it = std::find_if(labels.begin(), labels.end(),
                             [&](const LabelInfo& li) { return li.label == t.label[d]; });
      if (it == labels.end()) {
        LabelInfo li;
        li.label = t.label[d];
        li.extent = t.extent[d];
        li.first_pos = pos;
        labels.push_back(li);
        it = labels.end() - 1;
      } else if (it->extent != t.extent[d]) {
        return IMD_STATUS_INVALID_VALUE;  // row 1: extent mismatch
      }
      it->count[x] += 1;
      if (!add_i64(it->stride[x], t.stride[d], &it->stride[x])) merged_stride_overflow = true;
    }
  }

  // Step 3 / row 2: a repeated label within C, any extent.
  for (const LabelInfo& li : labels)
    if (li.count[kC] > 1) return IMD_STATUS_NOT_SUPPORTED;

  // Step 7 / row 3: a label only in C with extent > 1 (broadcast; a v1 policy).
  for (const LabelInfo& li : labels)
    if (li.count[kC] > 0 && li.count[kA] == 0 && li.count[kB] == 0 && li.extent > 1)
      return IMD_STATUS_NOT_SUPPORTED;

  // Row 4: the sorted-|stride| rule on a non-empty C (sufficient for injective writes).
  const bool c_empty = tensor_empty(C);
  if (!c_empty && !sorted_stride_rule(C)) return IMD_STATUS_INVALID_VALUE;

  // Row 5 (part): spans and merged diagonal strides fit int64.
  if (merged_stride_overflow) return IMD_STATUS_INVALID_VALUE;
  for (const imd_tensor_t* t : tensors)
    if (!span_fits(*t)) return IMD_STATUS_INVALID_VALUE;

  // Steps 4-7: classify by membership; extent-1 labels are dropped (they never enter a
  // product, step 4). A-only / B-only labels are summed (K, stride 0 in
  // the other operand). A C-only label left here has extent 0 (C is empty): it is
  // lowered to I with stride 0 in A, so m = 0 and the plan is a no-op (IndexMap.tla).
  bool zero[4] = {false, false, false, false};  // indexed by LabelClass
  for (LabelInfo& li : labels) {
    const bool a = li.count[kA] > 0, b = li.count[kB] > 0, c = li.count[kC] > 0;
    if (a && b && c) li.cls = LabelClass::L;
    else if (a && c) li.cls = LabelClass::I;
    else if (b && c) li.cls = LabelClass::J;
    else if (c) li.cls = LabelClass::I;  // C-only, extent 0
    else li.cls = LabelClass::K;         // {A,B}, {A} or {B}
    if (li.extent == 0) zero[static_cast<int>(li.cls)] = true;
  }
  int64_t size[4] = {1, 1, 1, 1};
  for (int g = 0; g < 4; ++g) {
    if (zero[g]) {
      size[g] = 0;  // exactly 0, even if the other extents' product would overflow
      continue;
    }
    for (const LabelInfo& li : labels)
      if (li.extent > 1 && static_cast<int>(li.cls) == g && !mul_i64(size[g], li.extent, &size[g]))
        return IMD_STATUS_INVALID_VALUE;  // row 5: extent product exceeds int64
  }

  Lowered r;
  r.m = size[static_cast<int>(LabelClass::I)];
  r.n = size[static_cast<int>(LabelClass::J)];
  r.k = size[static_cast<int>(LabelClass::K)];
  r.b = size[static_cast<int>(LabelClass::L)];

  // Step 8 (coalescing): if C's smallest-|stride| label is in I, swap the operand roles so
  // that j-bar runs along C's fastest axis. On a non-empty C that passed row 4 the smallest
  // |stride| among modes with extent > 1 is unique. The canonical orders are computed in
  // the post-swap frame by build_scatter_vectors.
  if (!c_empty) {
    std::optional<int32_t> fastest;
    for (int32_t d = 0; d < C.rank; ++d) {
      if (C.extent[d] <= 1) continue;
      if (!fastest || abs_u64(C.stride[d]) < abs_u64(C.stride[*fastest])) fastest = d;
    }
    if (fastest) {
      const int32_t lab = C.label[*fastest];
      const auto it = std::find_if(labels.begin(), labels.end(),
                                   [&](const LabelInfo& li) { return li.label == lab; });
      if (it->cls == LabelClass::I) r.swapped = true;
    }
  }
  if (r.swapped) {
    std::swap(r.m, r.n);
    for (LabelInfo& li : labels) {
      std::swap(li.count[kA], li.count[kB]);
      std::swap(li.stride[kA], li.stride[kB]);
      if (li.cls == LabelClass::I) li.cls = LabelClass::J;
      else if (li.cls == LabelClass::J) li.cls = LabelClass::I;
    }
  }

  // Row 5 (rest): lowered operand sizes fit int64. The GemmCoord part is gemm_dims_ok.
  int64_t t0 = 0, t1 = 0;
  if (!mul_i64(r.b, r.m, &t0) || !mul_i64(t0, r.k, &t1)) return IMD_STATUS_INVALID_VALUE;
  if (!mul_i64(r.b, r.n, &t0) || !mul_i64(t0, r.k, &t1)) return IMD_STATUS_INVALID_VALUE;
  if (!mul_i64(r.b, r.m, &t0) || !mul_i64(t0, r.n, &t1)) return IMD_STATUS_INVALID_VALUE;
  // The vec entry count 2m + 2k + 2n + 3b fits int64 whenever the §4a vec byte size
  // (8x that, overflow-checked in uint64 by workspace_layout) does.

  // Step 8: canonical orders in the post-swap frame (the swap above came first).
  const int big = r.m >= r.n ? kA : kB, oth = big == kA ? kB : kA;
  for (int g = 0; g < 4; ++g) {
    std::vector<const LabelInfo*> grp;
    for (const LabelInfo& li : labels)
      if (static_cast<int>(li.cls) == g && li.extent != 1) grp.push_back(&li);
    auto key = [&](const LabelInfo* l) -> std::array<uint64_t, 4> {
      const uint64_t a = abs_u64(l->stride[kA]), b = abs_u64(l->stride[kB]),
                     c = abs_u64(l->stride[kC]), p = static_cast<uint64_t>(l->first_pos);
      switch (static_cast<LabelClass>(g)) {
        case LabelClass::I: return {c, a, p, 0};
        case LabelClass::J: return {c, b, p, 0};
        case LabelClass::L: return {c, a, b, p};
        case LabelClass::K: return {abs_u64(l->stride[big]), abs_u64(l->stride[oth]), p, 0};
      }
      return {};
    };
    // Keys are distinct: positions are.
    std::sort(grp.begin(), grp.end(),
              [&](const LabelInfo* x, const LabelInfo* y) { return key(x) < key(y); });
    // Step 9: fusion. A tensor that does not carry the group has stride 0 on both modes,
    // where the condition holds trivially, so checking all three tensors is the rule.
    for (const LabelInfo* li : grp) {
      r.order[g].push_back(li->label);
      fuse_append(r.fused[g], *li);
    }
  }

  r.labels = std::move(labels);
  *out = std::move(r);
  return IMD_STATUS_SUCCESS;
}

const LabelInfo* find_label(const Lowered& lw, int32_t label) {
  for (const LabelInfo& li : lw.labels)
    if (li.label == label) return &li;
  return nullptr;
}

void adopt_order(Lowered* lw, int g, const std::vector<int32_t>& order) {
  lw->order[g] = order;
  lw->fused[g].clear();
  for (const int32_t lab : order) fuse_append(lw->fused[g], *find_label(*lw, lab));
}

bool gemm_dims_ok(const Lowered& lw) { return lw.m <= kInt32Max && lw.n <= kInt32Max - 15; }

int64_t scatter_vector_count(const Lowered& lw) {
  return 2 * lw.m + 2 * lw.k + 2 * lw.n + 3 * lw.b;
}

namespace {

constexpr int kRowClass[2] = {static_cast<int>(LabelClass::I), static_cast<int>(LabelClass::J)};
constexpr int kL = static_cast<int>(LabelClass::L), kK = static_cast<int>(LabelClass::K);

// Mixed-radix scatter vector of tensor X over a group's fused modes (first fastest), by an
// odometer: no intermediate leaves the tensor's span (row 5), so int64 is exact.
int64_t* radix_vec(const std::vector<FusedMode>& modes, int X, int64_t size, int64_t* out) {
  std::vector<int64_t> digit(modes.size(), 0);
  int64_t cur = 0;
  for (int64_t q = 0; q < size; ++q) {
    out[q] = cur;
    for (size_t t = 0; t < modes.size(); ++t) {
      const int64_t e = modes[t].extent, s = modes[t].stride[X];
      if (digit[t] + 1 < e) {
        ++digit[t];
        cur += s;
        break;
      }
      cur -= s * (e - 1);
      digit[t] = 0;
    }
  }
  return out + size;
}

}  // namespace

bool k5_batch_fastest(const Lowered& lw) {
  const auto& L = lw.fused[kL];
  const auto& J = lw.fused[static_cast<int>(LabelClass::J)];
  if (L.empty()) return false;
  if (J.empty()) return true;
  return abs_u64(L.front().stride[kC]) < abs_u64(J.front().stride[kC]);
}

PackPivot pack_pivot(const Lowered& lw, int X) {
  // rho = r + rows * lambda: the row group's fused modes, then L's, first fastest.
  uint64_t best = UINT64_MAX;
  int64_t radix = 1, best_radix = 1;
  for (const int g : {kRowClass[X], kL}) {
    for (const FusedMode& fm : lw.fused[g]) {
      const uint64_t s = abs_u64(fm.stride[X]);
      if (s != 0 && s < best) {
        best = s;
        best_radix = radix;
      }
      radix *= fm.extent;  // a prefix product of b * rows, which fits int64
    }
  }
  const uint64_t h_step = lw.fused[kK].empty() ? 0 : abs_u64(lw.fused[kK].front().stride[X]);
  PackPivot p;
  if (best != UINT64_MAX && (h_step == 0 || best < h_step)) {
    p.along_rows = true;
    p.radix = best_radix;
  }
  return p;
}

void build_scatter_vectors(const Lowered& lw, int64_t* out) {
  const auto& I = lw.fused[static_cast<int>(LabelClass::I)];
  const auto& J = lw.fused[static_cast<int>(LabelClass::J)];
  const auto& K = lw.fused[kK];
  const auto& L = lw.fused[kL];
  out = radix_vec(I, kA, lw.m, out);
  out = radix_vec(K, kA, lw.k, out);
  out = radix_vec(L, kA, lw.b, out);
  out = radix_vec(K, kB, lw.k, out);
  out = radix_vec(J, kB, lw.n, out);
  out = radix_vec(L, kB, lw.b, out);
  out = radix_vec(I, kC, lw.m, out);
  out = radix_vec(J, kC, lw.n, out);
  radix_vec(L, kC, lw.b, out);
}

}  // namespace imd::detail
