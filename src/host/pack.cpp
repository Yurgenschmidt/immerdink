#include "pack.hpp"

#include <algorithm>
#include <array>

#include "analysis.hpp"  // kappa_at_least_T
#include "checked.hpp"
#include "immerdink/detail/constants.hpp"
#include "index_map.hpp"
#include "k_hat.hpp"

namespace imd::detail {
namespace {

namespace c = imd::constants;

// One distinct label of the pack's X after §5 step 2 (a repeated label is a diagonal: its
// strides are summed).
struct PackLabel {
  int32_t label = 0;
  int64_t extent = 0;
  int64_t stride = 0;
  int first = 0;  // first position in X's labels
  int cls = -1;   // kPackK / kPackL / kPackR, -1 = not listed
};

PackLabel* find(std::vector<PackLabel>& labs, int32_t label) {
  for (PackLabel& l : labs)
    if (l.label == label) return &l;
  return nullptr;
}

}  // namespace

std::optional<PackLayout> pack_layout(int64_t b, int64_t r, int64_t k, int N, int32_t k_align) {
  PackLayout L;
  if (k_align != kKAlign16 && k_align != kKAlign64) return std::nullopt;
  if (b == 0 || r == 0 || k == 0) return L;  // an empty pack: 0 bytes
  const auto ub = static_cast<uint64_t>(b), ur = static_cast<uint64_t>(r),
             uk = static_cast<uint64_t>(k);
  uint64_t kh = 0, rh = 0, br = 0, sig = 0, res = 0, tail = 0;
  if (!k_hat_of(uk, static_cast<uint64_t>(k_align), &kh) || !align_u64(ur, c::kNAlign, &rh) ||
      !mul_u64(ub, ur, &br) || !mul_u64(br, 2, &sig) ||
      !mul_u64(br, kh, &res) || !mul_u64(res, static_cast<uint64_t>(N), &res) ||
      !mul_u64(rh - ur, kh, &tail) || !add_u64(res, tail, &res))
    return std::nullopt;
  const uint64_t bytes[3] = {sig, br, res};
  uint64_t off = 0;
  for (int i = 0; i < 3; ++i) {
    L.offset[i] = off;
    L.bytes[i] = bytes[i];
    uint64_t a = 0;
    if (!align_u64(bytes[i], c::kRegionAlign, &a) || !add_u64(off, a, &off)) return std::nullopt;
  }
  L.total = off;
  return L;
}

std::optional<uint64_t> pack_ws_bytes(int64_t b, int64_t r, int64_t k) {
  if (b == 0 || r == 0 || k == 0) return uint64_t{0};
  uint64_t v = 0;
  if (!add_u64(static_cast<uint64_t>(r), static_cast<uint64_t>(k), &v) ||
      !add_u64(v, static_cast<uint64_t>(b), &v) || !mul_u64(v, 8, &v) ||
      !align_u64(v, c::kRegionAlign, &v))
    return std::nullopt;
  return v;
}

void pack_fuse_append(std::vector<PackMode>& f, int64_t extent, int64_t stride) {
  int64_t next = 0;
  if (!f.empty() && mul_i64(f.back().stride, f.back().extent, &next) && next == stride)
    f.back().extent *= extent;  // a factor of the class size: no overflow
  else
    f.push_back({extent, stride});
}

std::vector<int64_t> pack_key(int64_t format, int N, int32_t rounding, int64_t b, int64_t r,
                              int64_t k, const std::vector<PackMode>& L,
                              const std::vector<PackMode>& R, const std::vector<PackMode>& K) {
  std::vector<int64_t> w = {format, N, rounding == IMD_TRUNC ? 1 : 0, b, r, k};
  for (const std::vector<PackMode>* f : {&L, &R, &K}) {
    w.push_back(static_cast<int64_t>(f->size()));
    for (const PackMode& m : *f) {
      w.push_back(m.extent);
      w.push_back(m.stride);
    }
  }
  return w;
}

uint64_t key_digest(const std::vector<int64_t>& key) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (const int64_t w : key) {
    const auto u = static_cast<uint64_t>(w);
    for (int i = 0; i < 8; ++i) {
      h ^= (u >> (8 * i)) & 0xffu;
      h *= 0x100000001b3ull;
    }
  }
  return h;
}

int32_t key_mismatch(const std::vector<int64_t>& slot, const std::vector<int64_t>& pack) {
  if (slot == pack) return IMD_PACK_MISMATCH_NONE;
  // The reason of every word of the slot key: format, N, rounding, then EXTENTS for b, r, k,
  // the mode counts and extents, STRIDES for the strides.
  std::vector<int32_t> kind = {IMD_PACK_MISMATCH_FORMAT, IMD_PACK_MISMATCH_MODULI,
                               IMD_PACK_MISMATCH_ROUNDING, IMD_PACK_MISMATCH_EXTENTS,
                               IMD_PACK_MISMATCH_EXTENTS, IMD_PACK_MISMATCH_EXTENTS};
  for (size_t p = 6; p < slot.size();) {
    const int64_t cnt = slot[p];
    kind.push_back(IMD_PACK_MISMATCH_EXTENTS);
    for (int64_t t = 0; t < cnt; ++t) {
      kind.push_back(IMD_PACK_MISMATCH_EXTENTS);
      kind.push_back(IMD_PACK_MISMATCH_STRIDES);
    }
    p += 1 + 2 * static_cast<size_t>(cnt);
  }
  // Word 0 (the format) last: FORMAT means "the same tensor in the other K alignment".
  const size_t n = std::min({slot.size(), pack.size(), kind.size()});
  for (size_t i = 1; i < n; ++i)
    if (slot[i] != pack[i]) return kind[i];
  if (slot.size() != pack.size()) return IMD_PACK_MISMATCH_EXTENTS;  // a count differed
  return IMD_PACK_MISMATCH_FORMAT;
}

imd_status_t pack_lower(const imd_pack_desc_t& d, PackLowered* out) {
  return pack_lower(d, k_align_sel(), out);
}

imd_status_t pack_lower(const imd_pack_desc_t& d, const KAlignSel& sel, PackLowered* out) {
  *out = PackLowered{};
  if (d.struct_size < sizeof(imd_pack_desc_t)) return IMD_STATUS_INVALID_VALUE;
  if (d.num_moduli < c::kNMin || d.num_moduli > c::kNMax ||
      (d.rounding != IMD_RN && d.rounding != IMD_TRUNC) ||
      (d.order != IMD_PACK_ORDER_AUTO && d.order != IMD_PACK_ORDER_GIVEN))
    return IMD_STATUS_INVALID_VALUE;
  if (d.tensor == nullptr || !descriptor_ok(*d.tensor)) return IMD_STATUS_INVALID_VALUE;
  if (d.num_contracted < 0 || d.num_batch < 0 || d.num_rows < 0 ||
      (d.num_contracted > 0 && d.contracted == nullptr) ||
      (d.num_batch > 0 && d.batch == nullptr) || (d.num_rows > 0 && d.rows == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  if (d.order == IMD_PACK_ORDER_AUTO && d.num_rows != 0) return IMD_STATUS_INVALID_VALUE;
  const imd_tensor_t& X = *d.tensor;

  // §5 steps 1-2 on X alone: one entry per distinct label, equal extents, merged strides.
  std::vector<PackLabel> labs;
  bool stride_overflow = false;
  for (int32_t p = 0; p < X.rank; ++p) {
    PackLabel* l = find(labs, X.label[p]);
    if (l == nullptr) {
      labs.push_back({X.label[p], X.extent[p], 0, p, -1});
      l = &labs.back();
    } else if (l->extent != X.extent[p]) {
      return IMD_STATUS_INVALID_VALUE;  // row 1: a repeated label with two extents
    }
    if (!add_i64(l->stride, X.stride[p], &l->stride)) stride_overflow = true;
  }
  // The label lists: each listed label is X's, listed once overall.
  const auto mark = [&](const int32_t* list, int32_t n, int cls) {
    for (int32_t i = 0; i < n; ++i) {
      PackLabel* l = find(labs, list[i]);
      if (l == nullptr || l->cls != -1) return false;
      l->cls = cls;
    }
    return true;
  };
  if (!mark(d.contracted, d.num_contracted, kPackK) || !mark(d.batch, d.num_batch, kPackL) ||
      (d.order == IMD_PACK_ORDER_GIVEN && !mark(d.rows, d.num_rows, kPackR)))
    return IMD_STATUS_INVALID_VALUE;
  for (PackLabel& l : labs) {
    if (l.cls != -1) continue;
    if (d.order == IMD_PACK_ORDER_GIVEN) return IMD_STATUS_INVALID_VALUE;  // a row not listed
    l.cls = kPackR;                                                       // AUTO: the rest
  }
  // §5 row 5 on X: merged strides and the span fit int64, the class sizes too.
  if (stride_overflow || !span_fits(X)) return IMD_STATUS_INVALID_VALUE;
  int64_t size[3] = {1, 1, 1};
  bool zero[3] = {false, false, false};
  for (const PackLabel& l : labs)
    if (l.extent == 0) zero[l.cls] = true;
  for (int cl = 0; cl < 3; ++cl) {
    if (zero[cl]) {
      size[cl] = 0;  // exactly 0, even if the other extents' product would overflow
      continue;
    }
    for (const PackLabel& l : labs)
      if (l.cls == cl && l.extent > 1 && !mul_i64(size[cl], l.extent, &size[cl]))
        return IMD_STATUS_INVALID_VALUE;
  }
  PackLowered& o = *out;
  o.N = d.num_moduli;
  o.rounding = d.rounding;
  o.b = size[kPackL], o.r = size[kPackR], o.k = size[kPackK];
  // v1.13 §4b.5: the pack's A, resolved once here from the call's hook snapshot (the hook
  // if set, else AUTO(k)), then passed explicitly; its format word follows.
  o.k_align = sel.of(o.k);
  o.format = pack_format_of(o.k_align);
  if (!k_hat_of(o.k, int64_t{o.k_align}, &o.k_hat)) return IMD_STATUS_INVALID_VALUE;
  const auto layout = pack_layout(o.b, o.r, o.k, o.N, o.k_align);
  const auto ws = pack_ws_bytes(o.b, o.r, o.k);
  if (!layout || !ws) return IMD_STATUS_INVALID_VALUE;  // §6b.4 sizes exceed uint64
  o.layout = *layout;
  o.ws_bytes = *ws;
  o.kappa = o.k >= 1 ? ceil_sqrt_u64(static_cast<uint64_t>(o.k)) : 0;
  o.lowered = true;

  // §6b.3: K empty after lowering (every contraction with it is DIRECT), RN acceptance.
  const bool k_kept = std::any_of(labs.begin(), labs.end(), [](const PackLabel& l) {
    return l.cls == kPackK && l.extent != 1;
  });
  if (!k_kept) return IMD_STATUS_NOT_SUPPORTED;
  if (o.rounding == IMD_RN && o.k >= 2 && kappa_at_least_T(o.kappa, o.N))
    return IMD_STATUS_NOT_SUPPORTED;

  // Canonical orders: GIVEN as listed, AUTO by (|merged stride|, first position); extent-1
  // labels dropped (§5 step 4); then X's modes fused over X's strides alone.
  const int32_t* lists[3] = {d.contracted, d.batch, d.rows};
  const int32_t counts[3] = {d.num_contracted, d.num_batch, d.num_rows};
  for (int cl = 0; cl < 3; ++cl) {
    std::vector<const PackLabel*> grp;
    if (d.order == IMD_PACK_ORDER_GIVEN) {
      for (int32_t i = 0; i < counts[cl]; ++i) grp.push_back(find(labs, lists[cl][i]));
    } else {
      for (const PackLabel& l : labs)
        if (l.cls == cl) grp.push_back(&l);
      std::sort(grp.begin(), grp.end(), [](const PackLabel* x, const PackLabel* y) {
        const std::array<uint64_t, 2> kx = {abs_u64(x->stride), static_cast<uint64_t>(x->first)},
                                      ky = {abs_u64(y->stride), static_cast<uint64_t>(y->first)};
        return kx < ky;
      });
    }
    for (const PackLabel* l : grp) {
      if (l->extent == 1) continue;
      o.order[cl].push_back(l->label);
      pack_fuse_append(o.fused[cl], l->extent, l->stride);
    }
  }
  o.key = pack_key(o.format, o.N, o.rounding, o.b, o.r, o.k, o.fused[kPackL], o.fused[kPackR],
                   o.fused[kPackK]);
  return IMD_STATUS_SUCCESS;
}

}  // namespace imd::detail
