// C ABI entry points (docs/DESIGN.md §6). No exception crosses this boundary.
#include <cstdint>
#include <new>
#include <vector>

#include <cuda_runtime_api.h>

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "index_map.hpp"
#include "workspace.hpp"

struct imd_handle_st {
  int device = 0;
};

// An owning copy of an imd_tensor_t (the caller's arrays need not outlive the plan).
struct TensorCopy {
  std::vector<int64_t> extent, stride;
  std::vector<int32_t> label;

  explicit TensorCopy(const imd_tensor_t& t)
      : extent(t.extent, t.extent + t.rank),
        stride(t.stride, t.stride + t.rank),
        label(t.label, t.label + t.rank) {}
};

struct imd_plan_st {
  imd_options_t options{};  // residue_group normalized to [1, N]
  imd::detail::Lowered lowered;
  imd::detail::WorkspaceLayout layout;
  TensorCopy A, B, C;  // for M2's scatter vectors and M3's aliasing check
};

namespace {

using imd::detail::Lowered;
using imd::detail::WorkspaceLayout;
namespace c = imd::constants;

struct Analysis {
  imd_options_t options{};
  Lowered lowered;
  WorkspaceLayout layout;
};

bool options_ok(const imd_options_t& o) {
  return o.num_moduli >= c::kNMin && o.num_moduli <= c::kNMax && o.residue_group >= 0 &&
         o.residue_group <= o.num_moduli && (o.rounding == IMD_RN || o.rounding == IMD_TRUNC) &&
         (o.mode == IMD_EMULATE || o.mode == IMD_DIRECT);
}

// Shared by imd_workspace_size and imd_plan_create: argument checks, then the §5 legality
// rows 1-5 in precedence order (index_map), then the §4a layout (uint64 overflow is row 5).
imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                     const imd_options_t* options, Analysis* out) {
  if (A == nullptr || B == nullptr || C == nullptr || options == nullptr)
    return IMD_STATUS_INVALID_VALUE;
  if (!options_ok(*options)) return IMD_STATUS_INVALID_VALUE;
  out->options = *options;
  if (out->options.residue_group == 0) out->options.residue_group = out->options.num_moduli;

  if (const imd_status_t st = imd::detail::lower(*A, *B, *C, &out->lowered);
      st != IMD_STATUS_SUCCESS)
    return st;

  const Lowered& lw = out->lowered;
  const auto layout = imd::detail::workspace_layout({lw.m, lw.n, lw.k, lw.b,
                                                     out->options.num_moduli,
                                                     out->options.residue_group,
                                                     out->options.mode == IMD_DIRECT});
  if (!layout) return IMD_STATUS_INVALID_VALUE;
  if (layout->total > SIZE_MAX) return IMD_STATUS_INVALID_VALUE;
  out->layout = *layout;
  return IMD_STATUS_SUCCESS;
}

// Multi-limb (little-endian u32) comparison x >= y.
bool limbs_ge(uint64_t x, const uint32_t (&y)[c::kMaxLimbs]) {
  for (int i = c::kMaxLimbs - 1; i >= 2; --i)
    if (y[i] != 0) return false;
  const uint64_t y64 = (uint64_t{y[1]} << 32) | y[0];
  return x >= y64;
}

// kappa = ceil(sqrt(k)) exactly, k >= 1 (DESIGN §3.2).
uint64_t ceil_sqrt(uint64_t k) {
  uint64_t r = 0;
  for (int bit = 31; bit >= 0; --bit) {  // floor(sqrt(k)) < 2^32 for k < 2^64
    const uint64_t t = r | (uint64_t{1} << bit);
    if (t * t <= k) r = t;
  }
  return r * r == k ? r : r + 1;
}

bool aligned(const void* p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

}  // namespace

extern "C" {

imd_status_t imd_create(imd_handle_t* handle) {
  if (handle == nullptr) return IMD_STATUS_INVALID_VALUE;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
  auto* h = new (std::nothrow) imd_handle_st{device};
  if (h == nullptr) return IMD_STATUS_CUDA_ERROR;  // no allocation status (NOTES-M0)
  *handle = h;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_destroy(imd_handle_t handle) {
  delete handle;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                const imd_tensor_t* C, const imd_options_t* options,
                                size_t* bytes) {
  if (bytes == nullptr) return IMD_STATUS_INVALID_VALUE;
  try {
    Analysis an;
    if (const imd_status_t st = analyze(A, B, C, options, &an); st != IMD_STATUS_SUCCESS)
      return st;
    *bytes = static_cast<size_t>(an.layout.total);
    return IMD_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_CUDA_ERROR;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

imd_status_t imd_plan_create(imd_handle_t handle, const imd_tensor_t* A, const imd_tensor_t* B,
                             const imd_tensor_t* C, const imd_options_t* options,
                             imd_plan_t* plan) {
  if (handle == nullptr || plan == nullptr) return IMD_STATUS_INVALID_VALUE;
  try {
    Analysis an;
    if (const imd_status_t st = analyze(A, B, C, options, &an); st != IMD_STATUS_SUCCESS)
      return st;
    const Lowered& lw = an.lowered;
    // §3.2 RN acceptance (accuracy policy): reject kappa >= T when the emulated GEMM runs.
    const bool emulated = !lw.empty_output() && an.options.mode == IMD_EMULATE && lw.k > 1;
    if (emulated && an.options.rounding == IMD_RN &&
        limbs_ge(ceil_sqrt(static_cast<uint64_t>(lw.k)), c::kTables.T[an.options.num_moduli]))
      return IMD_STATUS_NOT_SUPPORTED;
    // TODO(M2): canonical orders, fusion, pinned scatter vectors, K-chunk / launch split.
    *plan = new imd_plan_st{an.options, an.lowered, an.layout, TensorCopy(*A), TensorCopy(*B),
                            TensorCopy(*C)};
    return IMD_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_CUDA_ERROR;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

imd_status_t imd_plan_destroy(imd_plan_t plan) {
  delete plan;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_contract(imd_plan_t plan, double alpha, const double* A, const double* B,
                          double beta, double* C, void* ws, size_t ws_bytes,
                          cudaStream_t stream) {
  (void)beta;
  (void)stream;
  if (plan == nullptr) return IMD_STATUS_INVALID_VALUE;
  const Lowered& lw = plan->lowered;
  if (lw.empty_output()) return IMD_STATUS_SUCCESS;  // empty C: no-op, nothing touched

  // §3.7: A and B are not read when alpha = 0 or k = 0.
  const bool reads_ab = alpha != 0.0 && lw.k != 0;
  const uint64_t need = plan->layout.total;
  if (C == nullptr || (reads_ab && (A == nullptr || B == nullptr)) ||
      (need > 0 && ws == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  // TODO(M3): §5 row 6, ALIASING (C vs A/B byte intervals, workspace vs A/B/C).
  // §5 row 7: FP64 tensors 8-byte aligned, workspace 256-byte aligned.
  if (!aligned(C, 8) || (reads_ab && (!aligned(A, 8) || !aligned(B, 8))) ||
      (need > 0 && !aligned(ws, c::kRegionAlign)))
    return IMD_STATUS_MISALIGNED;
  if (ws_bytes < need) return IMD_STATUS_INSUFFICIENT_WORKSPACE;
  return IMD_STATUS_NOT_SUPPORTED;  // TODO(M1): K0-K5 kernels
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
  }
  return "IMD_STATUS_<unknown>";
}

}  // extern "C"
