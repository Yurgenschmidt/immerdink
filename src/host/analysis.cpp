#include "analysis.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "checked.hpp"
#include "k_hat.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {
namespace {

namespace c = imd::constants;

bool options_ok(const imd_options_t& o) {
  return o.num_moduli >= c::kNMin && o.num_moduli <= c::kNMax && o.residue_group >= 0 &&
         o.residue_group <= o.num_moduli && (o.rounding == IMD_RN || o.rounding == IMD_TRUNC) &&
         (o.mode == IMD_EMULATE || o.mode == IMD_DIRECT) &&
         (o.vec_mode == IMD_VEC_GENERATE || o.vec_mode == IMD_VEC_RESIDENT) &&
         (o.k4_tune == IMD_K4_TUNE_OFF || o.k4_tune == IMD_K4_TUNE_ON);
}

// Little-endian multi-word unsigned integers for the plan-time constants of §3.2.
using Big = std::array<uint32_t, 8>;  // 256 bits

Big big_from_limbs(const uint32_t (&x)[c::kMaxLimbs]) {
  Big r{};
  for (int i = 0; i < c::kMaxLimbs; ++i) r[i] = x[i];
  return r;
}
Big big_from_u64(uint64_t x) {
  Big r{};
  r[0] = static_cast<uint32_t>(x);
  r[1] = static_cast<uint32_t>(x >> 32);
  return r;
}
bool big_ge(const Big& x, const Big& y) {
  for (int i = 7; i >= 0; --i)
    if (x[i] != y[i]) return x[i] > y[i];
  return true;
}
Big big_sub(const Big& x, const Big& y) {  // requires x >= y
  Big r{};
  uint64_t borrow = 0;
  for (int i = 0; i < 8; ++i) {
    const uint64_t t = uint64_t{x[i]} - y[i] - borrow;
    r[i] = static_cast<uint32_t>(t);
    borrow = (t >> 32) & 1;
  }
  return r;
}
Big big_mul(const Big& x, const Big& y) {  // requires x y < 2^256
  Big r{};
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; i + j < 8; ++j) {
      const uint64_t t = uint64_t{r[i + j]} + uint64_t{x[i]} * y[j] + carry;
      r[i + j] = static_cast<uint32_t>(t);
      carry = t >> 32;
    }
  }
  return r;
}
Big big_shl(const Big& x, int s) {  // requires 0 <= s < 256 and no overflow
  Big r{};
  const int ws = s / 32, bs = s % 32;
  for (int i = 7; i >= 0; --i) {
    const int src = i - ws;
    uint64_t v = 0;
    if (src >= 0) v = uint64_t{x[src]} << bs;
    if (src >= 1 && bs != 0) v |= x[src - 1] >> (32 - bs);
    r[i] = static_cast<uint32_t>(v);
  }
  return r;
}
int big_bitlen(const Big& x) {
  for (int i = 7; i >= 0; --i)
    if (x[i] != 0) return 32 * i + 32 - __builtin_clz(x[i]);
  return 0;
}

}  // namespace

imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                     const imd_options_t* options, Analysis* out) {
  return analyze(A, B, C, options, k_align_sel(), out);
}

imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                     const imd_options_t* options, const KAlignSel& sel, Analysis* out) {
  if (A == nullptr || B == nullptr || C == nullptr || options == nullptr)
    return IMD_STATUS_INVALID_VALUE;
  if (!options_ok(*options)) return IMD_STATUS_INVALID_VALUE;
  out->options = *options;
  if (out->options.residue_group == 0) out->options.residue_group = out->options.num_moduli;

  if (const imd_status_t st = lower(*A, *B, *C, &out->lowered); st != IMD_STATUS_SUCCESS)
    return st;

  const Lowered& lw = out->lowered;
  // §5 row 5, GemmCoord part: only when the emulated GEMM runs (EMULATE, non-empty output,
  // k >= 2); not for DIRECT, K-empty or k = 0.
  if (out->options.mode == IMD_EMULATE && !lw.empty_output() && lw.k >= 2 && !gemm_dims_ok(lw))
    return IMD_STATUS_INVALID_VALUE;
  // v1.13 §4b.5: the one resolution of A for this analysis, from the API call's one hook
  // snapshot `sel`; passed explicitly from here on.
  out->k_align = sel.of(lw.k);
  const auto layout = workspace_layout({lw.m, lw.n, lw.k, lw.b, out->options.num_moduli,
                                        out->options.residue_group,
                                        out->options.mode == IMD_DIRECT,
                                        out->options.vec_mode == IMD_VEC_RESIDENT, 0,
                                        out->k_align});
  if (!layout) return IMD_STATUS_INVALID_VALUE;
  if (layout->total > SIZE_MAX) return IMD_STATUS_INVALID_VALUE;
  out->layout = *layout;
  return IMD_STATUS_SUCCESS;
}

imd_status_t analyze_packed(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                            const imd_options_t* options, const PackSource* const src[2],
                            Analysis* out) {
  return analyze_packed(A, B, C, options, k_align_sel(), src, out);
}

imd_status_t analyze_packed(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                            const imd_options_t* options, const KAlignSel& sel,
                            const PackSource* const src[2], Analysis* out) {
  if (const imd_status_t st = analyze(A, B, C, options, sel, out); st != IMD_STATUS_SUCCESS)
    return st;
  return match_packs(A, B, src, out);
}

imd_status_t match_packs(const imd_tensor_t* A, const imd_tensor_t* B,
                         const PackSource* const src[2], Analysis* out) {
  out->packed = PackedSlots{};
  if (src[0] == nullptr && src[1] == nullptr) return IMD_STATUS_SUCCESS;
  Lowered& lw = out->lowered;
  PackedSlots& ps = out->packed;
  const imd_tensor_t* X[2] = {A, B};
  int role[2] = {0, 1};
  for (int x = 0; x < 2; ++x) {
    role[x] = (x == 0) != lw.swapped ? 0 : 1;  // post-swap role of the caller's operand x
    if (src[x] != nullptr) {
      ps.caller |= 1 << x;
      ps.role |= 1 << role[x];
    }
  }
  // §6b.6: the workspace without sX, fX, RX of the packed roles (no overflow: the in-line
  // layout of the same shape fitted).
  const auto layout = workspace_layout({lw.m, lw.n, lw.k, lw.b, out->options.num_moduli,
                                        out->options.residue_group,
                                        out->options.mode == IMD_DIRECT,
                                        out->options.vec_mode == IMD_VEC_RESIDENT, ps.role,
                                        out->k_align});
  if (!layout) return IMD_STATUS_INVALID_VALUE;
  out->layout = *layout;
  const auto mismatch = [&](int32_t reason) {
    ps.mismatch = reason;
    return IMD_STATUS_PACK_MISMATCH;
  };
  constexpr int kI = static_cast<int>(LabelClass::I), kJ = static_cast<int>(LabelClass::J),
                kK = static_cast<int>(LabelClass::K), kL = static_cast<int>(LabelClass::L);
  // Per packed operand (the caller's A, then B): an unknown format word (v1.13 §4b.5: a
  // future or foreign key layout; a known format of the other alignment is compared last), N,
  // rounding, then the classes, mapped onto the plan's labels by mode position in X.
  std::vector<int32_t> adopted[2][3];  // per caller operand: K, L, rows (plan labels)
  for (int x = 0; x < 2; ++x) {
    if (src[x] == nullptr) continue;
    const PackLowered& pl = *src[x]->pl;
    if (pl.key.empty() || pack_format_align(pl.key[0]) == 0)
      return mismatch(IMD_PACK_MISMATCH_FORMAT);
    if (pl.N != out->options.num_moduli) return mismatch(IMD_PACK_MISMATCH_MODULI);
    if (pl.rounding != out->options.rounding) return mismatch(IMD_PACK_MISMATCH_ROUNDING);
    const imd_tensor_t& T = *X[x];
    if (src[x]->rank != T.rank) return mismatch(IMD_PACK_MISMATCH_CLASSES);
    // pack label -> plan label, a bijection consistent at every position
    std::vector<std::pair<int32_t, int32_t>> map;
    for (int32_t p = 0; p < T.rank; ++p) {
      const int32_t lp = src[x]->label[p], lx = T.label[p];
      bool seen = false;
      for (const auto& [a, b] : map) {
        if ((a == lp) != (b == lx)) return mismatch(IMD_PACK_MISMATCH_CLASSES);
        seen = seen || a == lp;
      }
      if (!seen) map.emplace_back(lp, lx);
    }
    const auto to_plan = [&](int32_t lp) {
      for (const auto& [a, b] : map)
        if (a == lp) return b;
      return lp;  // unreachable: the pack's labels are X's
    };
    // a K label X does not carry (partner-only, §5 step 6): not supported (§6b.10 #10)
    for (const int32_t lab : lw.order[kK])
      if (find_label(lw, lab)->count[role[x]] == 0) return mismatch(IMD_PACK_MISMATCH_CLASSES);
    const int plan_cls[3] = {kK, kL, role[x] == 0 ? kI : kJ};
    for (int cl = 0; cl < 3; ++cl) {
      for (const int32_t lp : pl.order[cl]) adopted[x][cl].push_back(to_plan(lp));
      std::vector<int32_t> a = adopted[x][cl], b = lw.order[plan_cls[cl]];
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      if (a != b) return mismatch(IMD_PACK_MISMATCH_CLASSES);
    }
  }
  if (src[0] != nullptr && src[1] != nullptr &&
      (adopted[0][kPackK] != adopted[1][kPackK] || adopted[0][kPackL] != adopted[1][kPackL]))
    return mismatch(IMD_PACK_MISMATCH_ORDER);
  // Adopt the packs' orders (§6b.2); fusion is redone in them.
  for (int x = 0; x < 2; ++x) {
    if (src[x] == nullptr) continue;
    adopt_order(&lw, kK, adopted[x][kPackK]);
    adopt_order(&lw, kL, adopted[x][kPackL]);
    adopt_order(&lw, role[x] == 0 ? kI : kJ, adopted[x][kPackR]);
  }
  // The slot keys, from the adopted lowering restricted to X (X's merged strides alone), with
  // the PLAN's own format word (v1.13 §4b.5: of its resolved A, never IMD_PACK_FORMAT). That
  // is Analysis::k_align = sel.of(k), also for a plan without residue rows (DIRECT, K-empty)
  // whose ExecPlan::k_align / imd_lowering_t.k_align is 16: such a plan reads no residues,
  // and a pack must match it exactly when it matches the EMULATE plan of the same tensors
  // (the mode is not in the key, §6b.2). Keying with 16 would reject every AUTO pack of
  // k > 96 (format 2) from a DIRECT plan. DESIGN §6b.4 word 1.
  int32_t reason[2] = {0, 0};
  for (int x = 0; x < 2; ++x) {
    if (src[x] == nullptr) continue;
    std::vector<PackMode> f[3];
    const int plan_cls[3] = {kK, kL, role[x] == 0 ? kI : kJ};
    for (int cl = 0; cl < 3; ++cl)
      for (const int32_t lab : lw.order[plan_cls[cl]]) {
        const LabelInfo& li = *find_label(lw, lab);
        pack_fuse_append(f[cl], li.extent, li.stride[role[x]]);
      }
    const int64_t rows = role[x] == 0 ? lw.m : lw.n;
    ps.key[x] = pack_key(pack_format_of(out->k_align), out->options.num_moduli,
                         out->options.rounding, lw.b, rows, lw.k, f[kPackL], f[kPackR],
                         f[kPackK]);
    reason[x] = key_mismatch(ps.key[x], src[x]->pl->key);
  }
  // The other key words of both packs first, then the format 1 vs 2 of either (§4b.5: FORMAT
  // means "the same tensor in the other K alignment"; a k = 96 pack against a k = 97 plan is
  // EXTENTS).
  for (const bool format_pass : {false, true})
    for (int x = 0; x < 2; ++x)
      if (reason[x] != 0 && (reason[x] == IMD_PACK_MISMATCH_FORMAT) == format_pass) {
        ps.key[0].clear(), ps.key[1].clear();
        return mismatch(reason[x]);
      }
  // Drift check (§4b.5): equal keys (format and k) must mean equal k_hat.
  int64_t plan_k_hat = 0;
  if (!k_hat_of(lw.k, int64_t{out->k_align}, &plan_k_hat)) return IMD_STATUS_INVALID_VALUE;
  for (int x = 0; x < 2; ++x)
    if (src[x] != nullptr && src[x]->pl->k_hat != plan_k_hat) {
      ps.key[0].clear(), ps.key[1].clear();
      return mismatch(IMD_PACK_MISMATCH_FORMAT);
    }
  for (int x = 0; x < 2; ++x)
    if (src[x] != nullptr) ps.layout[role[x]] = src[x]->pl->layout;  // the plan's (b, r, k, N)
  return IMD_STATUS_SUCCESS;
}

bool kappa_at_least_T(uint64_t kappa, int N) {
  return big_ge(big_from_u64(kappa), big_from_limbs(c::kTables.T[N]));
}

uint64_t ceil_sqrt_u64(uint64_t k) {
  uint64_t r = 0;
  for (int bit = 31; bit >= 0; --bit) {  // floor(sqrt(k)) < 2^32 for k < 2^64
    const uint64_t t = r | (uint64_t{1} << bit);
    if (t * t <= k) r = t;
  }
  return r * r == k ? r : r + 1;
}

namespace {

// §3.2 step 4 right-hand side R 2^F: RN R = (2T - kappa)^2 (kappa < T), TRUNC R = T^2.
// T < 2^78, so the base is < 2^80, R < 2^160 and R 2^F < 2^220.
void set_shift_constants(int N, bool rn, uint64_t k, ExecPlan* e) {
  const Big T = big_from_limbs(c::kTables.T[N]);
  const Big kappa = big_from_u64(ceil_sqrt_u64(k));
  const Big base = rn ? big_sub(big_shl(T, 1), kappa) : T;
  const Big rhs = big_shl(big_mul(base, base), c::kF);
  for (int w = 0; w < 4; ++w)
    e->rhs[w] = uint64_t{rhs[2 * w]} | (uint64_t{rhs[2 * w + 1]} << 32);
  e->rhs_bits = big_bitlen(rhs);
  e->lhs_shift = rn ? 2 : 0;
}

}  // namespace

imd_status_t plan_data(const Analysis& an, ExecPlan* out) {
  const Lowered& lw = an.lowered;
  const int N = an.options.num_moduli;
  const bool rn = an.options.rounding == IMD_RN;
  ExecPlan& e = *out;
  e = ExecPlan{};
  e.swapped = lw.swapped;
  e.m = lw.m, e.n = lw.n, e.k = lw.k, e.b = lw.b;
  e.n_hat = (lw.n + c::kNAlign - 1) / c::kNAlign * c::kNAlign;
  e.N = N;
  e.g = an.options.residue_group;
  e.trunc = !rn;
  e.layout = an.layout;
  e.path = lw.empty_output()                                ? Path::kNoop
           : (an.options.mode == IMD_DIRECT || lw.k == 1) ? Path::kDirect
                                                          : Path::kGemm;
  // v1.13 §4b.5: A from the analysis (resolved once), for a plan with residue rows; 16 for
  // NOOP, SCALE (k = 0), DIRECT and K-empty, whatever the hook says.
  e.k_align = emulates(e) ? an.k_align : kKAlign16;
  if (e.k_align != kKAlign16 && e.k_align != kKAlign64) return IMD_STATUS_INVALID_VALUE;
  if (!k_hat_of(lw.k, int64_t{e.k_align}, &e.k_hat)) return IMD_STATUS_INVALID_VALUE;
  // v1.7 (§6b): the packed post-swap operands and where their regions are in the packs.
  e.packed = an.packed.role;
  for (int x = 0; x < 2; ++x) {
    if (!(e.packed & (1 << x))) continue;
    for (int r = 0; r < 3; ++r) e.pack_off[x][r] = an.packed.layout[x].offset[r];
    e.pack_bytes[x] = an.packed.layout[x].total;
  }
  // §3.2 RN acceptance (accuracy policy): reject kappa >= T when the emulated GEMM runs.
  const bool emulated = emulates(e);  // non-empty output, EMULATE, k >= 2
  if (emulated && rn && kappa_at_least_T(ceil_sqrt_u64(static_cast<uint64_t>(lw.k)), N))
    return IMD_STATUS_NOT_SUPPORTED;

  if (emulated) {
    set_shift_constants(N, rn, static_cast<uint64_t>(lw.k), &e);
    for (int x = 0; x < 2; ++x) {
      const PackPivot pv = pack_pivot(lw, x);
      e.pack_along_rows[x] = pv.along_rows;
      e.pack_radix[x] = pv.radix;
    }
    e.k5_batch_fast = k5_batch_fastest(lw);
    e.fused = fused_path(e);
    e.fused_large = fused_large(e, IMD_TABLE_SM89, kK4ReferenceSms);  // the reference class and SM count; api.cpp redoes it for the device's
    // v1.13 §4b.5 (hygiene: at plan time, not per call): KF's tile row covers the whole
    // padded row, so KF writes the zeros [k, k_hat). Unreachable (kf_lg_tk rounds up).
    if (e.fused && (int64_t{1} << kf_lg_tk(e.k_hat)) < e.k_hat) return IMD_STATUS_INVALID_VALUE;
  }
  if (e.path != Path::kNoop) {
    // §5 step 10 (v1.2): the fused-mode descriptor K0 generates the vectors from.
    e.vec_count = scatter_vector_count(lw);
    for (int g = 0; g < 4; ++g) {
      e.vec_desc.first[g] = static_cast<int32_t>(e.vec_desc.modes.size());
      e.vec_desc.count[g] = static_cast<int32_t>(lw.fused[g].size());
      for (const FusedMode& f : lw.fused[g])
        e.vec_desc.modes.push_back({f.extent, {f.stride[0], f.stride[1], f.stride[2]}});
    }
    if (e.vec_desc.modes.size() > static_cast<size_t>(kMaxVecModes))
      return IMD_STATUS_INVALID_VALUE;  // unreachable: at most 62 fused modes per class
    // §4a "plan device bytes" (v1.2): the resident vectors. Fits: analyze sized the vec
    // region (8 vec_count) without overflow in both vec modes.
    if (an.options.vec_mode == IMD_VEC_RESIDENT)
      e.device_bytes = static_cast<uint64_t>(e.vec_count) * sizeof(int64_t);
  }
  return IMD_STATUS_SUCCESS;
}

int64_t plan_launches(const ExecPlan& e, bool fused, int32_t fused_large, bool resident,
                      int64_t k4_launches) {
  if (e.path == Path::kNoop) return 0;
  const int64_t k0 = resident ? 0 : 1;
  if (!emulates(e)) return k0 + 1;  // DIRECT, K-empty or k = 0: K0 + one kernel
  const int64_t groups = (e.N + e.g - 1) / e.g;
  const int64_t zero_rows = e.n_hat > e.n ? 1 : 0;
  // Per in-line operand (v1.7: a packed one has none): its pack per group (B: + zero rows).
  const int64_t pack_a = (e.packed & 1) ? 0 : 1, pack_b = (e.packed & 2) ? 0 : 1 + zero_rows;
  const int64_t per_group = pack_a + pack_b;
  // KF, or K0 + one scale launch per in-line operand
  const int64_t head = fused ? 1 : k0 + ((e.packed & 1) ? 0 : 1) + ((e.packed & 2) ? 0 : 1);
  // Group 1: no pack after KF; after K1 / KL, the packs of the operands not done by KL.
  const int64_t group1 = fused ? 0
                               : ((fused_large & kFusedLargeA) ? 0 : pack_a) +
                                     ((fused_large & kFusedLargeB) ? 0 : pack_b);
  const int64_t packs = (groups - 1) * per_group + group1;
  if (k4_launches > INT64_MAX - head - packs - 1) return INT64_MAX;  // saturate like k4_counts
  return head + packs + k4_launches + 1;
}

K4Counts plan_k4_counts(const ExecPlan& e, const LaunchLimits& lim) {
  if (!emulates(e)) return {};
  return k4_counts(k4_grid(e.n_hat, e.k_hat, e.b, lim), e.N, e.g);
}

uint64_t plan_host_bytes(const imd_tensor_t& A, const imd_tensor_t& B, const imd_tensor_t& C,
                         const Analysis& an, const ExecPlan& e) {
  const Lowered& lw = an.lowered;
  uint64_t bytes = sizeof(imd_plan_st);
  const uint64_t modes = static_cast<uint64_t>(A.rank) + static_cast<uint64_t>(B.rank) +
                         static_cast<uint64_t>(C.rank);
  bytes += (sizeof(int64_t) + sizeof(int64_t) + sizeof(int32_t)) * modes;
  bytes += sizeof(LabelInfo) * lw.labels.size();
  uint64_t fused = 0;
  for (int g = 0; g < 4; ++g) {
    bytes += sizeof(int32_t) * lw.order[g].size();
    fused += lw.fused[g].size();
  }
  bytes += sizeof(FusedMode) * fused;
  if (e.path != Path::kNoop) bytes += sizeof(VecMode) * fused;
  if (emulates(e)) {
    const K4Counts kc = plan_k4_counts(e, production_launch_limits());
    bytes += kK4PlanBytes + kK4StoreEntryBytes * static_cast<uint64_t>(kc.store_shapes) +
             kK4AddEntryBytes * static_cast<uint64_t>(kc.add_shapes);
  }
  // v1.7: the copied slot keys of the packed operands (§6b.6)
  bytes += sizeof(int64_t) * (an.packed.key[0].size() + an.packed.key[1].size());
  return bytes;
}

void pack_exec(const PackLowered& pl, ExecPlan* out) {
  ExecPlan& e = *out;
  e = ExecPlan{};
  e.path = Path::kGemm;
  e.m = pl.r, e.n = 0, e.k = pl.k, e.b = pl.b;
  e.k_hat = pl.k_hat;
  e.k_align = pl.k_align;  // v1.13 §4b: the pack's resolved A
  e.N = pl.N;
  e.g = pl.N;  // all N planes at once: the format is g-free (§6b.4)
  e.trunc = pl.rounding == IMD_TRUNC;
  if (pl.k >= 2) set_shift_constants(pl.N, pl.rounding == IMD_RN, static_cast<uint64_t>(pl.k), &e);
  // X as the GEMM's A of a one-operand view: I = the pack's R, K, L (X's strides in slot A).
  Lowered lw;
  lw.m = pl.r, lw.k = pl.k, lw.b = pl.b;
  constexpr int cls[3] = {static_cast<int>(LabelClass::K), static_cast<int>(LabelClass::L),
                          static_cast<int>(LabelClass::I)};  // kPackK, kPackL, kPackR
  for (int cl = 0; cl < 3; ++cl)
    for (const PackMode& pm : pl.fused[cl]) {
      FusedMode f;
      f.extent = pm.extent;
      f.stride[0] = pm.stride;
      lw.fused[cls[cl]].push_back(f);
    }
  const PackPivot pv = pack_pivot(lw, 0);
  e.pack_along_rows[0] = pv.along_rows;
  e.pack_radix[0] = pv.radix;
  for (int g = 0; g < 4; ++g) {
    e.vec_desc.first[g] = static_cast<int32_t>(e.vec_desc.modes.size());
    e.vec_desc.count[g] = static_cast<int32_t>(lw.fused[g].size());
    for (const FusedMode& f : lw.fused[g])
      e.vec_desc.modes.push_back({f.extent, {f.stride[0], f.stride[1], f.stride[2]}});
  }
  e.vec_count = pl.r + pl.k + pl.b;
}

}  // namespace imd::detail
