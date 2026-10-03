// libimmerdink_host C ABI (include/immerdink/immerdink_host.h, docs/DESIGN.md §6a): pure,
// thread-safe host queries over analysis.hpp. No exception crosses this boundary; nothing
// here touches CUDA, a device or global state.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "analysis.hpp"
#include "immerdink/immerdink_host.h"

namespace {

using imd::detail::Analysis;
using imd::detail::ExecPlan;

// The version-1 size of imd_lowering_t (§6a): fields are only ever appended.
constexpr uint32_t kLoweringV1Size = 136;
constexpr uint32_t kLoweringV2Size = 152;  // + fused, reserved, kernel_launches (v1.4)
constexpr uint32_t kLoweringV3Size = 168;  // + sm_count, k4_config, k4_tune_bytes (v1.5)
constexpr uint32_t kLoweringV4Size = 176;  // + fused_large, reserved4 (v1.6)
constexpr uint32_t kLoweringV5Size = 184;  // + packed, pack_mismatch (v1.7)
static_assert(offsetof(imd_lowering_t, fused) == kLoweringV1Size, "imd_lowering_t v1 layout changed");
static_assert(offsetof(imd_lowering_t, sm_count) == kLoweringV2Size, "imd_lowering_t v2 layout changed");
static_assert(offsetof(imd_lowering_t, fused_large) == kLoweringV3Size, "imd_lowering_t v3 layout changed");
static_assert(offsetof(imd_lowering_t, packed) == kLoweringV4Size, "imd_lowering_t v4 layout changed");
static_assert(sizeof(imd_lowering_t) == kLoweringV5Size, "imd_lowering_t v5 layout changed");
static_assert(sizeof(imd_options_t) == 24, "imd_options_t layout changed");
static_assert(sizeof(imd_region_t) == 16, "imd_region_t layout changed");
// v1.7 (§6b.5): the version-1 pack structs (64-bit ABI).
static_assert(sizeof(imd_pack_desc_t) == 72, "imd_pack_desc_t v1 layout changed");
static_assert(sizeof(imd_pack_layout_t) == 136, "imd_pack_layout_t v1 layout changed");
constexpr uint32_t kPackLayoutV1Size = 136;

using imd::detail::PackLowered;
using imd::detail::PackSource;

// The packs of a host query (imd_packing_t: descriptors), lowered: each descriptor's own
// status first (imd_pack_layout's), then the plan's analyze_packed.
imd_status_t analyze_with(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                          const imd_options_t* options, const imd_packing_t* packing,
                          Analysis* an) {
  if (packing == nullptr || (packing->A == nullptr && packing->B == nullptr))
    return imd::detail::analyze(A, B, C, options, an);
  // §5 rows 1-5 of the plan first (as imd_workspace_size), then the pack descriptors.
  if (const imd_status_t st = imd::detail::analyze(A, B, C, options, an); st != IMD_STATUS_SUCCESS)
    return st;
  const imd_pack_desc_t* d[2] = {packing->A, packing->B};
  PackLowered pl[2];
  PackSource ps[2];
  const PackSource* src[2] = {nullptr, nullptr};
  for (int x = 0; x < 2; ++x) {
    if (d[x] == nullptr) continue;
    if (const imd_status_t st = imd::detail::pack_lower(*d[x], &pl[x]); st != IMD_STATUS_SUCCESS)
      return st;
    ps[x] = PackSource{d[x]->tensor->rank, d[x]->tensor->label, &pl[x]};
    src[x] = &ps[x];
  }
  return imd::detail::analyze_packed(A, B, C, options, src, an);
}

// Runs f() and maps a host allocation failure to ALLOC_FAILED (the queries allocate only
// O(rank) host memory for the lowering tables).
template <class F>
imd_status_t guarded(F&& f) {
  try {
    return f();
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_ALLOC_FAILED;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

}  // namespace

extern "C" {

uint32_t imd_host_api_version(void) { return IMD_HOST_API_VERSION; }

imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                const imd_tensor_t* C, const imd_options_t* options,
                                size_t* bytes) {
  if (bytes == nullptr) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    *bytes = static_cast<size_t>(an.layout.total);
    return IMD_STATUS_SUCCESS;
  });
}

const char* imd_region_name(int32_t region) {
  static const char* const kNames[IMD_REGION_COUNT] = {"sA", "sB", "fA", "fB",
                                                       "vec", "RA", "RB", "W"};
  return region >= 0 && region < IMD_REGION_COUNT ? kNames[region] : nullptr;
}

imd_status_t imd_plan_bytes(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                            const imd_options_t* options, size_t* device_bytes,
                            size_t* host_bytes) {
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    ExecPlan e;
    if (const imd_status_t st = imd::detail::plan_data(an, &e); st != IMD_STATUS_SUCCESS)
      return st;
    if (device_bytes != nullptr) *device_bytes = static_cast<size_t>(e.device_bytes);
    if (host_bytes != nullptr)
      *host_bytes = static_cast<size_t>(imd::detail::plan_host_bytes(*A, *B, *C, an, e));
    return IMD_STATUS_SUCCESS;
  });
}

}  // extern "C"

namespace {

// imd_lowering and imd_lowering_packed (packing NULL: the in-line plan).
imd_status_t lowering_impl(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                           const imd_options_t* options, const imd_packing_t* packing,
                           imd_lowering_t* out) {
  if (out == nullptr || out->struct_size < kLoweringV1Size) return IMD_STATUS_INVALID_VALUE;
  imd_lowering_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  // v3: sm_count is an input (0 or a version-1/2 caller: the reference device).
  const int32_t sms_in = out->struct_size >= kLoweringV3Size ? out->sm_count : 0;
  if (sms_in < 0) return IMD_STATUS_INVALID_VALUE;
  const int32_t sms = sms_in > 0 ? sms_in : imd::detail::kK4ReferenceSms;
  r.status = guarded([&] {
    Analysis an;
    const imd_status_t ast = analyze_with(A, B, C, options, packing, &an);
    // v1.7: a PACK_MISMATCH still has the plan's view (rows 1-5 passed) and its packed
    // layout; the summary is filled as for NOT_SUPPORTED (no plan bytes).
    if (ast != IMD_STATUS_SUCCESS && ast != IMD_STATUS_PACK_MISMATCH) return ast;
    r.packed = an.packed.caller;
    r.pack_mismatch = an.packed.mismatch;
    const imd::detail::Lowered& lw = an.lowered;
    ExecPlan e;
    const imd_status_t plan_st = ast != IMD_STATUS_SUCCESS
                                     ? (imd::detail::plan_data(an, &e), ast)
                                     : imd::detail::plan_data(an, &e);  // view, path: always set
    r.lowered = 1;
    r.num_moduli = e.N;
    r.residue_group = e.g;
    r.swapped = e.swapped ? 1 : 0;
    r.m = e.m, r.n = e.n, r.k = e.k, r.b = e.b;
    r.n_hat = e.n_hat, r.k_hat = e.k_hat;
    r.kappa = e.k >= 1 ? imd::detail::ceil_sqrt_u64(static_cast<uint64_t>(e.k)) : 0;
    r.workspace_bytes = an.layout.total;
    using imd::detail::Path;
    r.dispatch = e.path == Path::kNoop        ? IMD_DISPATCH_NOOP
                 : e.k == 0                   ? IMD_DISPATCH_SCALE
                 : imd::detail::emulates(e)   ? IMD_DISPATCH_EMULATE
                                              : IMD_DISPATCH_DIRECT;
    if (e.path != Path::kNoop) r.vec_entries = imd::detail::scatter_vector_count(lw);
    if (r.dispatch == IMD_DISPATCH_EMULATE) {
      r.num_groups = (e.N + e.g - 1) / e.g;
      const imd::detail::K4Counts kc =
          imd::detail::plan_k4_counts(e, imd::detail::production_launch_limits());
      r.k_chunks = kc.k_chunks;
      r.gemm_launches = kc.launches;
    }
    // v1.4: the fused launch and the kernel launches per call (contract.cu runs exactly these);
    // v1.6: the operands of the fused large launch KL (plan_data set e.fused_large).
    const bool fused = imd::detail::fused_path(e);
    r.fused = fused ? 1 : 0;
    r.fused_large = imd::detail::fused_large(e);
    r.kernel_launches = imd::detail::plan_launches(e, fused, r.fused_large,
                                                   an.options.vec_mode == IMD_VEC_RESIDENT,
                                                   r.gemm_launches);
    // v1.5: the K4 configuration of the plan-time rule (api.cpp uses the same function with
    // the device's SM count) and the autotuning scratch bytes.
    r.sm_count = sms;
    r.k4_config = imd::detail::k4_select(e, sms);
    r.k4_tune_bytes = imd::detail::k4_tune_bytes(e);
    if (plan_st != IMD_STATUS_SUCCESS) return plan_st;
    r.plan_device_bytes = e.device_bytes;
    r.plan_host_bytes = imd::detail::plan_host_bytes(*A, *B, *C, an, e);
    return IMD_STATUS_SUCCESS;
  });
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  return static_cast<imd_status_t>(r.status);
}

// imd_workspace_layout and imd_workspace_layout_packed.
imd_status_t layout_impl(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                         const imd_options_t* options, const imd_packing_t* packing,
                         imd_region_t* regions, int32_t capacity, int32_t* num_regions,
                         size_t* total) {
  if (capacity < 0 || (capacity > 0 && regions == nullptr)) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = analyze_with(A, B, C, options, packing, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    const int32_t n = std::min<int32_t>(capacity, IMD_REGION_COUNT);
    for (int32_t r = 0; r < n; ++r) regions[r] = {an.layout.offset[r], an.layout.bytes[r]};
    if (num_regions != nullptr) *num_regions = IMD_REGION_COUNT;
    if (total != nullptr) *total = static_cast<size_t>(an.layout.total);
    return IMD_STATUS_SUCCESS;
  });
}

}  // namespace

extern "C" {

imd_status_t imd_lowering(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                          const imd_options_t* options, imd_lowering_t* out) {
  return lowering_impl(A, B, C, options, nullptr, out);
}

imd_status_t imd_lowering_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                 const imd_tensor_t* C, const imd_options_t* options,
                                 const imd_packing_t* packing, imd_lowering_t* out) {
  return lowering_impl(A, B, C, options, packing, out);
}

imd_status_t imd_workspace_size_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       const imd_packing_t* packing, size_t* bytes) {
  if (bytes == nullptr) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = analyze_with(A, B, C, options, packing, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    *bytes = static_cast<size_t>(an.layout.total);
    return IMD_STATUS_SUCCESS;
  });
}

imd_status_t imd_workspace_layout(const imd_tensor_t* A, const imd_tensor_t* B,
                                  const imd_tensor_t* C, const imd_options_t* options,
                                  imd_region_t* regions, int32_t capacity,
                                  int32_t* num_regions, size_t* total) {
  return layout_impl(A, B, C, options, nullptr, regions, capacity, num_regions, total);
}

imd_status_t imd_workspace_layout_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                         const imd_tensor_t* C, const imd_options_t* options,
                                         const imd_packing_t* packing, imd_region_t* regions,
                                         int32_t capacity, int32_t* num_regions,
                                         size_t* total) {
  return layout_impl(A, B, C, options, packing, regions, capacity, num_regions, total);
}

imd_status_t imd_pack_layout(const imd_pack_desc_t* desc, imd_pack_layout_t* out) {
  if (desc == nullptr || out == nullptr || out->struct_size < kPackLayoutV1Size)
    return IMD_STATUS_INVALID_VALUE;
  imd_pack_layout_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = guarded([&] {
    PackLowered pl;
    const imd_status_t st = imd::detail::pack_lower(*desc, &pl);
    if (pl.lowered) {  // SUCCESS or NOT_SUPPORTED: sizes and layout
      r.format = IMD_PACK_FORMAT;
      r.num_moduli = pl.N;
      r.rounding = pl.rounding;
      r.b = pl.b, r.r = pl.r, r.k = pl.k, r.k_hat = pl.k_hat;
      r.kappa = pl.kappa;
      imd_region_t* reg[3] = {&r.sig, &r.flag, &r.res};
      for (int i = 0; i < 3; ++i) *reg[i] = {pl.layout.offset[i], pl.layout.bytes[i]};
      r.pack_bytes = pl.layout.total;
      r.pack_ws_bytes = pl.ws_bytes;
    }
    if (st == IMD_STATUS_SUCCESS) r.key_digest = imd::detail::key_digest(pl.key);
    return st;
  });
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  return static_cast<imd_status_t>(r.status);
}

imd_status_t imd_pack_desc_for(const imd_tensor_t* A, const imd_tensor_t* B,
                               const imd_tensor_t* C, const imd_options_t* options,
                               int32_t operand, imd_pack_desc_t* out, int32_t* labels,
                               int32_t capacity) {
  if ((operand != 0 && operand != 1) || out == nullptr ||
      out->struct_size < sizeof(imd_pack_desc_t) || capacity < 0 ||
      (capacity > 0 && labels == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    const imd::detail::Lowered& lw = an.lowered;
    const imd_tensor_t& X = operand == 0 ? *A : *B;
    const imd_tensor_t& P = operand == 0 ? *B : *A;
    const int role = (operand == 0) != lw.swapped ? 0 : 1;
    using imd::detail::LabelClass;
    const int cls[3] = {static_cast<int>(LabelClass::K), static_cast<int>(LabelClass::L),
                        role == 0 ? static_cast<int>(LabelClass::I) : static_cast<int>(LabelClass::J)};
    const auto in = [](const imd_tensor_t& t, int32_t lab) {
      for (int32_t p = 0; p < t.rank; ++p)
        if (t.label[p] == lab) return true;
      return false;
    };
    std::vector<int32_t> lists[3];  // K, L, R
    for (int cl = 0; cl < 3; ++cl)
      for (const int32_t lab : lw.order[cls[cl]])
        if (in(X, lab)) lists[cl].push_back(lab);  // a partner-only K label is not X's
    // X's labels not in the plan's orders (extent 1): appended by membership.
    for (int32_t p = 0; p < X.rank; ++p) {
      const int32_t lab = X.label[p];
      bool listed = false;
      for (const auto& l : lists) listed = listed || std::find(l.begin(), l.end(), lab) != l.end();
      if (listed) continue;
      lists[!in(*C, lab) ? 0 : in(P, lab) ? 1 : 2].push_back(lab);
    }
    const size_t need = lists[0].size() + lists[1].size() + lists[2].size();
    if (need > static_cast<size_t>(capacity)) return IMD_STATUS_INVALID_VALUE;
    int32_t* w = labels;
    const int32_t* starts[3];
    for (int cl = 0; cl < 3; ++cl) {
      starts[cl] = w;
      for (const int32_t lab : lists[cl]) *w++ = lab;
    }
    const uint32_t size = out->struct_size;
    std::memset(out, 0, sizeof(imd_pack_desc_t));
    out->struct_size = size;
    out->num_moduli = an.options.num_moduli;
    out->rounding = an.options.rounding;
    out->order = IMD_PACK_ORDER_GIVEN;
    out->tensor = operand == 0 ? A : B;
    out->num_contracted = static_cast<int32_t>(lists[0].size());
    out->contracted = lists[0].empty() ? nullptr : starts[0];
    out->num_batch = static_cast<int32_t>(lists[1].size());
    out->batch = lists[1].empty() ? nullptr : starts[1];
    out->num_rows = static_cast<int32_t>(lists[2].size());
    out->rows = lists[2].empty() ? nullptr : starts[2];
    return IMD_STATUS_SUCCESS;
  });
}

const char* imd_status_string(imd_status_t status) {
  switch (status) {
    case IMD_STATUS_SUCCESS: return "IMD_STATUS_SUCCESS";
    case IMD_STATUS_INVALID_VALUE: return "IMD_STATUS_INVALID_VALUE";
    case IMD_STATUS_NOT_SUPPORTED: return "IMD_STATUS_NOT_SUPPORTED";
    case IMD_STATUS_ALIASING: return "IMD_STATUS_ALIASING";
    case IMD_STATUS_INSUFFICIENT_WORKSPACE: return "IMD_STATUS_INSUFFICIENT_WORKSPACE";
    case IMD_STATUS_MISALIGNED: return "IMD_STATUS_MISALIGNED";
    case IMD_STATUS_CUDA_ERROR: return "IMD_STATUS_CUDA_ERROR";
    case IMD_STATUS_ALLOC_FAILED: return "IMD_STATUS_ALLOC_FAILED";
    case IMD_STATUS_PACK_MISMATCH: return "IMD_STATUS_PACK_MISMATCH";
  }
  return "IMD_STATUS_<unknown>";
}

}  // extern "C"
