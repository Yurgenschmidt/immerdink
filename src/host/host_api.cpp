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
static_assert(offsetof(imd_lowering_t, fused) == kLoweringV1Size, "imd_lowering_t v1 layout changed");
static_assert(offsetof(imd_lowering_t, sm_count) == kLoweringV2Size, "imd_lowering_t v2 layout changed");
static_assert(sizeof(imd_lowering_t) == kLoweringV3Size, "imd_lowering_t v3 layout changed");
static_assert(sizeof(imd_options_t) == 24, "imd_options_t layout changed");
static_assert(sizeof(imd_region_t) == 16, "imd_region_t layout changed");

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

imd_status_t imd_workspace_layout(const imd_tensor_t* A, const imd_tensor_t* B,
                                  const imd_tensor_t* C, const imd_options_t* options,
                                  imd_region_t* regions, int32_t capacity,
                                  int32_t* num_regions, size_t* total) {
  if (capacity < 0 || (capacity > 0 && regions == nullptr)) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    Analysis an;
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    const int32_t n = std::min<int32_t>(capacity, IMD_REGION_COUNT);
    for (int32_t r = 0; r < n; ++r) regions[r] = {an.layout.offset[r], an.layout.bytes[r]};
    if (num_regions != nullptr) *num_regions = IMD_REGION_COUNT;
    if (total != nullptr) *total = static_cast<size_t>(an.layout.total);
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

imd_status_t imd_lowering(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                          const imd_options_t* options, imd_lowering_t* out) {
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
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    const imd::detail::Lowered& lw = an.lowered;
    ExecPlan e;
    const imd_status_t plan_st = imd::detail::plan_data(an, &e);  // view and path: always set
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
    // v1.4: the fused launch and the kernel launches per call (contract.cu runs exactly these).
    const bool fused = imd::detail::fused_path(e);
    r.fused = fused ? 1 : 0;
    r.kernel_launches = imd::detail::plan_launches(e, fused, an.options.vec_mode == IMD_VEC_RESIDENT,
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
  }
  return "IMD_STATUS_<unknown>";
}

}  // extern "C"
