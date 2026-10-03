// C ABI entry points of libimmerdink (docs/DESIGN.md §6). No exception crosses this
// boundary. Argument checks, lowering, legality, the RN acceptance rule and all sizing are
// libimmerdink_host's (src/host/analysis.hpp, DESIGN §6a), the code behind
// imd_workspace_size and the host queries; this file adds the device-side plan parts.
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include <cuda_runtime_api.h>

#include "host/analysis.hpp"
#include "host/checked.hpp"
#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "plan.hpp"
#if defined(IMMERDINK_PROFILE) && IMMERDINK_PROFILE
#  include "profile_hooks.h"
#endif

struct imd_handle_st {
  int device = 0;
};

// The plan record is defined in host/plan_data.hpp (CUDA-free, so that libimmerdink_host
// can size it); its destructor frees the IMD_VEC_RESIDENT device vectors.
imd_plan_st::~imd_plan_st() {
  if (exec.vec_dev != nullptr) cudaFree(exec.vec_dev);
}

namespace {

using imd::detail::Analysis;
using imd::detail::Lowered;
using imd::detail::TensorCopy;
namespace c = imd::constants;

bool aligned(const void* p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

// §5 row 6: the bytes a strided FP64 tensor can address, [lo, hi), or none (lo == hi) for
// an empty tensor or a null pointer. Extent-1 modes contribute nothing; a rank-0 tensor
// is one element. Exact in 128 bits (spans fit int64 by row 5).
using imd::detail::i128;
struct ByteInterval {
  i128 lo = 0, hi = 0;
};
ByteInterval byte_interval(const TensorCopy& t, const void* p) {
  ByteInterval r;
  if (p == nullptr) return r;
  i128 neg = 0, pos = 0;
  for (size_t d = 0; d < t.extent.size(); ++d) {
    if (t.extent[d] == 0) return r;
    const i128 span = static_cast<i128>(t.stride[d]) * (t.extent[d] - 1);
    (span < 0 ? neg : pos) += span;
  }
  const i128 base = static_cast<i128>(reinterpret_cast<uintptr_t>(p));
  r.lo = base + 8 * neg;
  r.hi = base + 8 * pos + 8;
  return r;
}
bool overlap(const ByteInterval& x, const ByteInterval& y) {
  return x.lo < x.hi && y.lo < y.hi && x.lo < y.hi && y.lo < x.hi;
}

}  // namespace

// For src/test_hooks.cpp (same library, not exported): the plan's execution data.
namespace imd::detail {
const ExecPlan* exec_of(imd_plan_t plan) { return plan == nullptr ? nullptr : &plan->exec; }
int& fused_override() {
  static int v = -1;  // only the test hook changes it
  return v;
}
}  // namespace imd::detail

extern "C" {

imd_status_t imd_create(imd_handle_t* handle) {
  if (handle == nullptr) return IMD_STATUS_INVALID_VALUE;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
  auto* h = new (std::nothrow) imd_handle_st{device};
  if (h == nullptr) return IMD_STATUS_ALLOC_FAILED;
  *handle = h;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_destroy(imd_handle_t handle) {
  delete handle;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_plan_create(imd_handle_t handle, const imd_tensor_t* A, const imd_tensor_t* B,
                             const imd_tensor_t* C, const imd_options_t* options,
                             imd_plan_t* plan) {
  if (handle == nullptr || plan == nullptr) return IMD_STATUS_INVALID_VALUE;
  try {
    Analysis an;
    if (const imd_status_t st = imd::detail::analyze(A, B, C, options, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    // §3.2 RN acceptance, the GEMM view and every plan-time constant: libimmerdink_host.
    imd::detail::ExecPlan pd;
    if (const imd_status_t st = imd::detail::plan_data(an, &pd); st != IMD_STATUS_SUCCESS)
      return st;
    const bool emulated = imd::detail::emulates(pd);  // K4 parameters needed
    if (imd::detail::fused_override() == 0) pd.fused = false;  // test hook only

    auto p = std::make_unique<imd_plan_st>(an.options, an.lowered, *A, *B, *C);
    imd::detail::ExecPlan& e = p->exec;
    e = std::move(pd);
    if (e.path != imd::detail::Path::kNoop) {
      int device = 0;
      if (cudaGetDevice(&device) != cudaSuccess ||
          cudaDeviceGetAttribute(&e.sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        return IMD_STATUS_CUDA_ERROR;
      }
    }
    if (emulated) {  // §4 / §6 v1.2: every K4 launch shape's parameters, built here once
      // v1.5: the K4 configuration. libimmerdink_host's rule for this device's SM count (what
      // imd_lowering reports for sm_count = e.sms), or the fastest one timed (k4_tune).
      e.k4_config = imd::detail::k4_select(e, e.sms);
      if (imd::detail::k4_config_override() >= 0) {
        e.k4_config = imd::detail::k4_config_override();  // test / benchmark hook only
      } else if (an.options.k4_tune == IMD_K4_TUNE_ON) {
        if (const imd_status_t st = imd::detail::k4_tune(e, &e.k4_config, nullptr);
            st != IMD_STATUS_SUCCESS)
          return st;
      }
      if (const imd_status_t st = imd::detail::k4_prepare(e, e.k4_config, &e.k4);
          st != IMD_STATUS_SUCCESS)
        return st;
    }
    if (e.path != imd::detail::Path::kNoop && an.options.vec_mode == IMD_VEC_RESIDENT) {
      // §4 K0 v1.2 IMD_VEC_RESIDENT: build the vectors once in plan-owned device memory
      // (freed by the plan), on this thread's default stream, and wait for them.
      e.vec_resident = true;
      void* dev = nullptr;
      // Exactly the §4a plan device bytes of libimmerdink_host (imd_plan_bytes).
      const cudaError_t err = cudaMalloc(&dev, static_cast<size_t>(e.device_bytes));
      if (err != cudaSuccess) {
        cudaGetLastError();  // clear the (non-sticky) error state
        return err == cudaErrorMemoryAllocation ? IMD_STATUS_ALLOC_FAILED : IMD_STATUS_CUDA_ERROR;
      }
      e.vec_dev = static_cast<int64_t*>(dev);
      if (imd::detail::k0_generate(e, e.vec_dev, cudaStreamPerThread) != IMD_STATUS_SUCCESS ||
          cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        cudaGetLastError();
        return IMD_STATUS_CUDA_ERROR;  // the plan (and vec_dev) is freed by unique_ptr
      }
    }
    *plan = p.release();
    return IMD_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_ALLOC_FAILED;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

imd_status_t imd_plan_destroy(imd_plan_t plan) {
  delete plan;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_plan_device_bytes(imd_plan_t plan, size_t* bytes) {
  if (plan == nullptr || bytes == nullptr) return IMD_STATUS_INVALID_VALUE;
  // DESIGN §4a v1.2 "plan device bytes": exactly what imd_plan_create requested, as
  // computed by libimmerdink_host (imd_plan_bytes).
  *bytes = plan->exec.vec_resident ? static_cast<size_t>(plan->exec.device_bytes) : 0;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_contract(imd_plan_t plan, double alpha, const double* A, const double* B,
                          double beta, double* C, void* ws, size_t ws_bytes,
                          cudaStream_t stream) {
  // DESIGN §6 order: null pointers; empty output; ALIASING; MISALIGNED; workspace size.
  if (plan == nullptr) return IMD_STATUS_INVALID_VALUE;
  const imd::detail::ExecPlan& e = plan->exec;
  // §3.7: A and B are not read when alpha = 0 or k = 0; C is always required.
  const bool reads_ab = alpha != 0.0 && e.k != 0;
  const uint64_t need = e.layout.total;
  if (C == nullptr || (reads_ab && (A == nullptr || B == nullptr)) || (need > 0 && ws == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  if (e.path == imd::detail::Path::kNoop) return IMD_STATUS_SUCCESS;  // nothing touched
  // §5 row 6, ALIASING: C vs A and C vs B as byte intervals, also when A / B are not read
  // (alpha = 0 or k = 0), conservatively; a null A / B (allowed only when not read) has no
  // bytes. The workspace bytes actually used, [ws, ws + workspace_size), vs A, B and C.
  const ByteInterval iA = byte_interval(plan->A, A), iB = byte_interval(plan->B, B),
                     iC = byte_interval(plan->C, C);
  ByteInterval iW;
  if (need > 0) {
    iW.lo = static_cast<i128>(reinterpret_cast<uintptr_t>(ws));
    iW.hi = iW.lo + static_cast<i128>(need);
  }
  if (overlap(iC, iA) || overlap(iC, iB) || overlap(iW, iA) || overlap(iW, iB) ||
      overlap(iW, iC))
    return IMD_STATUS_ALIASING;
  // §5 row 7: A, B and C 8-byte aligned (a null pointer counts as aligned: it is only
  // allowed where it is not read), workspace 256-byte aligned.
  if (!aligned(A, 8) || !aligned(B, 8) || !aligned(C, 8) ||
      (need > 0 && !aligned(ws, c::kRegionAlign)))
    return IMD_STATUS_MISALIGNED;
  if (ws_bytes < need) return IMD_STATUS_INSUFFICIENT_WORKSPACE;
  // §5 step 8 role swap: the GEMM computes C^T = B^T A^T.
  const double* gA = e.swapped ? B : A;
  const double* gB = e.swapped ? A : B;
  try {
    return imd::detail::enqueue_contract(e, alpha, gA, gB, beta, C, ws, stream);
  } catch (...) {
    return IMD_STATUS_CUDA_ERROR;
  }
}

#if defined(IMMERDINK_PROFILE) && IMMERDINK_PROFILE
// Benchmark-only (src/profile_hooks.h): the plan's post-swap GEMM view.
IMD_API int imd_prof_plan_dims(imd_plan_t plan, int64_t* out) {
  if (plan == nullptr || out == nullptr) return 1;
  const imd::detail::ExecPlan& e = plan->exec;
  const int64_t v[7] = {e.m, e.n, e.k, e.b, e.n_hat, e.k_hat, static_cast<int64_t>(e.path)};
  for (int i = 0; i < 7; ++i) out[i] = v[i];
  return 0;
}
// Benchmark-only (v1.5): the K4 configuration a plan runs (IMD_K4_*; -1 without K4), and
// the configuration every later plan runs (-1: the rule / k4_tune).
IMD_API int imd_prof_plan_k4(imd_plan_t plan, int32_t* config) {
  if (plan == nullptr || config == nullptr) return 1;
  *config = plan->exec.k4 ? plan->exec.k4_config : -1;
  return 0;
}
IMD_API int imd_prof_set_k4_config(int32_t config) {
  if (config < -1 || config >= IMD_K4_CONFIG_COUNT) return 1;
  imd::detail::k4_config_override() = config;
  return 0;
}
#endif

}  // extern "C"
