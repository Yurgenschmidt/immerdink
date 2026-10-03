#include "analysis.hpp"

#include <array>
#include <cstdint>

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {
namespace {

namespace c = imd::constants;

bool options_ok(const imd_options_t& o) {
  return o.num_moduli >= c::kNMin && o.num_moduli <= c::kNMax && o.residue_group >= 0 &&
         o.residue_group <= o.num_moduli && (o.rounding == IMD_RN || o.rounding == IMD_TRUNC) &&
         (o.mode == IMD_EMULATE || o.mode == IMD_DIRECT) &&
         (o.vec_mode == IMD_VEC_GENERATE || o.vec_mode == IMD_VEC_RESIDENT);
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
  const auto layout = workspace_layout({lw.m, lw.n, lw.k, lw.b, out->options.num_moduli,
                                        out->options.residue_group,
                                        out->options.mode == IMD_DIRECT,
                                        out->options.vec_mode == IMD_VEC_RESIDENT});
  if (!layout) return IMD_STATUS_INVALID_VALUE;
  if (layout->total > SIZE_MAX) return IMD_STATUS_INVALID_VALUE;
  out->layout = *layout;
  return IMD_STATUS_SUCCESS;
}

uint64_t ceil_sqrt_u64(uint64_t k) {
  uint64_t r = 0;
  for (int bit = 31; bit >= 0; --bit) {  // floor(sqrt(k)) < 2^32 for k < 2^64
    const uint64_t t = r | (uint64_t{1} << bit);
    if (t * t <= k) r = t;
  }
  return r * r == k ? r : r + 1;
}

imd_status_t plan_data(const Analysis& an, ExecPlan* out) {
  const Lowered& lw = an.lowered;
  const int N = an.options.num_moduli;
  const bool rn = an.options.rounding == IMD_RN;
  ExecPlan& e = *out;
  e = ExecPlan{};
  e.swapped = lw.swapped;
  e.m = lw.m, e.n = lw.n, e.k = lw.k, e.b = lw.b;
  e.k_hat = (lw.k + c::kKAlign - 1) / c::kKAlign * c::kKAlign;
  e.n_hat = (lw.n + c::kNAlign - 1) / c::kNAlign * c::kNAlign;
  e.N = N;
  e.g = an.options.residue_group;
  e.trunc = !rn;
  e.layout = an.layout;
  e.path = lw.empty_output()                                ? Path::kNoop
           : (an.options.mode == IMD_DIRECT || lw.k == 1) ? Path::kDirect
                                                          : Path::kGemm;
  // §3.2 RN acceptance (accuracy policy): reject kappa >= T when the emulated GEMM runs.
  const bool emulated = emulates(e);  // non-empty output, EMULATE, k >= 2
  const Big T = big_from_limbs(c::kTables.T[N]);
  const Big kappa = big_from_u64(ceil_sqrt_u64(static_cast<uint64_t>(lw.k)));
  if (emulated && rn && big_ge(kappa, T)) return IMD_STATUS_NOT_SUPPORTED;

  if (emulated) {
    // §3.2 step 4 right-hand side R 2^F: RN R = (2T - kappa)^2 (kappa < T here), TRUNC
    // R = T^2. T < 2^78, so the base is < 2^80, R < 2^160 and R 2^F < 2^220.
    const Big base = rn ? big_sub(big_shl(T, 1), kappa) : T;
    const Big rhs = big_shl(big_mul(base, base), c::kF);
    for (int w = 0; w < 4; ++w)
      e.rhs[w] = uint64_t{rhs[2 * w]} | (uint64_t{rhs[2 * w + 1]} << 32);
    e.rhs_bits = big_bitlen(rhs);
    e.lhs_shift = rn ? 2 : 0;
    for (int x = 0; x < 2; ++x) {
      const PackPivot pv = pack_pivot(lw, x);
      e.pack_along_rows[x] = pv.along_rows;
      e.pack_radix[x] = pv.radix;
    }
    e.k5_batch_fast = k5_batch_fastest(lw);
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
  return bytes;
}

}  // namespace imd::detail
