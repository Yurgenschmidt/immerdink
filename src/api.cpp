// C ABI entry points of libimmerdink (docs/DESIGN.md §6). No exception crosses this
// boundary. Argument checks, lowering, legality, the RN acceptance rule and all sizing are
// libimmerdink_host's (src/host/analysis.hpp, DESIGN §6a), the code behind
// imd_workspace_size and the host queries; this file adds the device-side plan parts.
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include <cuda_runtime_api.h>

#include "host/analysis.hpp"
#include "device_state.hpp"
#include "host/checked.hpp"
#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "k4/k4_ops.hpp"
#include "plan.hpp"
#if defined(IMMERDINK_PROFILE) && IMMERDINK_PROFILE
#  include "profile_hooks.h"
#endif

struct imd_handle_st {
  int device = 0;
};

// v1.7 (DESIGN §6b): a pack handle. Immutable after imd_pack_create: the descriptor's tensor
// (labels by position for plans, extents and strides for imd_pack's aliasing check), its
// lowering (key, layout) and the launch data of imd_pack. No GPU work reads it.
struct imd_pack_s {
  imd::detail::TensorCopy tensor;
  imd::detail::PackLowered pl;
  imd::detail::ExecPlan exec;
  explicit imd_pack_s(const imd_tensor_t& t) : tensor(t) {}
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
// [p, p + n), or none for a null pointer or n = 0 (a workspace, a packed buffer).
ByteInterval bytes_at(const void* p, uint64_t n) {
  ByteInterval r;
  if (p == nullptr || n == 0) return r;
  r.lo = static_cast<i128>(reinterpret_cast<uintptr_t>(p));
  r.hi = r.lo + static_cast<i128>(n);
  return r;
}

}  // namespace

// For src/test_hooks.cpp (same library, not exported): the plan's execution data.
namespace imd::detail {
const ExecPlan* exec_of(imd_plan_t plan) { return plan == nullptr ? nullptr : &plan->exec; }
PackSource pack_source_of(imd_pack_t pack) {
  return {static_cast<int32_t>(pack->tensor.label.size()), pack->tensor.label.data(), &pack->pl};
}
int& fused_override() {
  static int v = -1;  // only the test hook changes it (-1, 0, 2, 3: test_hooks.cpp)
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

imd_status_t imd_device_info(imd_handle_t handle, imd_device_info_t* info) {
  if (handle == nullptr || info == nullptr || info->struct_size < sizeof(imd_device_info_t))
    return IMD_STATUS_INVALID_VALUE;
  int device = 0;
  imd_device_info_t r;
  if (cudaGetDevice(&device) != cudaSuccess ||
      imd::detail::device_state(device, &r) != cudaSuccess) {
    cudaGetLastError();
    return IMD_STATUS_CUDA_ERROR;
  }
  const auto caller_size = info->struct_size;   // a caller built against a later header keeps its own size field
  std::memcpy(info, &r, sizeof r);
  info->struct_size = caller_size;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_plan_info(imd_plan_t plan, imd_device_info_t* info) {
  if (plan == nullptr || info == nullptr || info->struct_size < sizeof(imd_device_info_t))
    return IMD_STATUS_INVALID_VALUE;
  const auto caller_size = info->struct_size;
  std::memcpy(info, &plan->info, sizeof plan->info);
  info->struct_size = caller_size;
  return IMD_STATUS_SUCCESS;
}

}  // extern "C"

namespace {

// imd_plan_create and imd_plan_create_packed (src: the packs of the caller's A and B, or null).
imd_status_t plan_create(imd_handle_t handle, const imd_tensor_t* A, const imd_tensor_t* B,
                         const imd_tensor_t* C, const imd_options_t* options,
                         const imd::detail::PackSource* const src[2], imd_plan_t* plan) {
  if (handle == nullptr || plan == nullptr) return IMD_STATUS_INVALID_VALUE;
  try {
    Analysis an;
    // §5 rows 1-5, then (v1.7) the packs' keys and the adopted orders (libimmerdink_host).
    if (const imd_status_t st = imd::detail::analyze_packed(A, B, C, options, src, &an);
        st != IMD_STATUS_SUCCESS)
      return st;
    // §3.2 RN acceptance, the GEMM view and every plan-time constant: libimmerdink_host.
    imd::detail::ExecPlan pd;
    if (const imd_status_t st = imd::detail::plan_data(an, &pd); st != IMD_STATUS_SUCCESS)
      return st;
    const bool emulated = imd::detail::emulates(pd);  // K4 parameters needed
    // Test hook only: 0 = no fused launch at all; 2 / 3 = KL for both operands of every
    // emulated plan (3: with 64-row KL tiles).
    if (imd::detail::fused_override() == 0) {
      pd.fused = false;
      pd.fused_large = 0;
    } else if (imd::detail::fused_override() >= 2 && emulated) {
      pd.fused = false;  // v1.7: KL only for in-line operands
      pd.fused_large = (imd::detail::kFusedLargeA | imd::detail::kFusedLargeB) & ~pd.packed;
      if (imd::detail::fused_override() == 3) pd.kl_lg_rows = 6;
    }

    auto p = std::make_unique<imd_plan_st>(an.options, an.lowered, *A, *B, *C);
    p->slot_key[0] = an.packed.key[0];  // v1.7: the packed slots' keys, copied
    p->slot_key[1] = an.packed.key[1];
    imd::detail::ExecPlan& e = p->exec;
    e = std::move(pd);
    // Issue 18: the device class and the kind of code the library's kernels run as on it. An
    // unusable device fails here, at plan creation, with NOT_SUPPORTED (info: imd_device_info).
    // A plan of an empty GEMM touches no CUDA (imd_contract returns SUCCESS without a device): no device query, no probe, info stays zero.
    if (e.path != imd::detail::Path::kNoop) {
      int device = 0;
      if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return IMD_STATUS_CUDA_ERROR;
      }
      if (imd::detail::device_state(device, &p->info) != cudaSuccess) {
        cudaGetLastError();
        return IMD_STATUS_CUDA_ERROR;
      }
      if (p->info.status == IMD_PLAN_UNSUPPORTED) return IMD_STATUS_NOT_SUPPORTED;
      e.sms = p->info.device.sm_count;
    } else {
      p->info.struct_size = sizeof p->info;
    }
    if (emulated) {  // §4 / §6 v1.2: every K4 launch shape's parameters, built here once
      // v1.5: the K4 configuration. libimmerdink_host's rule for this device's SM count (what
      // imd_lowering reports for sm_count = e.sms), or the fastest one timed (k4_tune).
      e.k4_config = imd::detail::k4_select(e, e.sms, p->info.table_id);
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
      p->info.k4_config = e.k4_config;
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

}  // namespace

extern "C" {

imd_status_t imd_plan_create(imd_handle_t handle, const imd_tensor_t* A, const imd_tensor_t* B,
                             const imd_tensor_t* C, const imd_options_t* options,
                             imd_plan_t* plan) {
  const imd::detail::PackSource* const none[2] = {nullptr, nullptr};
  return plan_create(handle, A, B, C, options, none, plan);
}

imd_status_t imd_plan_create_packed(imd_handle_t handle, const imd_tensor_t* A,
                                    const imd_tensor_t* B, const imd_tensor_t* C,
                                    const imd_options_t* options, imd_pack_t packA,
                                    imd_pack_t packB, imd_plan_t* plan) {
  imd::detail::PackSource ps[2];
  const imd::detail::PackSource* src[2] = {nullptr, nullptr};
  const imd_pack_t pk[2] = {packA, packB};
  for (int x = 0; x < 2; ++x) {
    if (pk[x] == nullptr) continue;
    ps[x] = imd::detail::pack_source_of(pk[x]);
    src[x] = &ps[x];
  }
  return plan_create(handle, A, B, C, options, src, plan);
}

imd_status_t imd_pack_create(imd_handle_t handle, const imd_pack_desc_t* desc, imd_pack_t* pack) {
  if (handle == nullptr || desc == nullptr || pack == nullptr) return IMD_STATUS_INVALID_VALUE;
  try {
    imd::detail::PackLowered pl;
    if (const imd_status_t st = imd::detail::pack_lower(*desc, &pl); st != IMD_STATUS_SUCCESS)
      return st;
    auto p = std::make_unique<imd_pack_s>(*desc->tensor);
    p->pl = std::move(pl);
    imd::detail::pack_exec(p->pl, &p->exec);  // libimmerdink_host: the launch data of imd_pack
    p->exec.kl_lg_rows = imd::detail::pack_kl_lg_rows(p->exec);
    *pack = p.release();
    return IMD_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_ALLOC_FAILED;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

imd_status_t imd_pack_destroy(imd_pack_t pack) {
  delete pack;
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_pack_bytes(imd_pack_t pack, size_t* pack_bytes, size_t* pack_ws_bytes) {
  if (pack == nullptr) return IMD_STATUS_INVALID_VALUE;
  if (pack_bytes != nullptr) *pack_bytes = static_cast<size_t>(pack->pl.layout.total);
  if (pack_ws_bytes != nullptr) *pack_ws_bytes = static_cast<size_t>(pack->pl.ws_bytes);
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_pack(imd_pack_t pack, const double* X, void* packed, size_t packed_bytes,
                      void* ws, size_t ws_bytes, cudaStream_t stream) {
  // DESIGN §6b.5 order: null pointers; empty pack; ALIASING; MISALIGNED; sizes.
  if (pack == nullptr) return IMD_STATUS_INVALID_VALUE;
  const uint64_t need = pack->pl.layout.total, need_ws = pack->pl.ws_bytes;
  if ((need > 0 && (X == nullptr || packed == nullptr)) || (need_ws > 0 && ws == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  if (need == 0) return IMD_STATUS_SUCCESS;  // b r k = 0: nothing touched
  const ByteInterval iX = byte_interval(pack->tensor, X), iP = bytes_at(packed, need),
                     iW = bytes_at(ws, need_ws);
  if (overlap(iP, iX) || overlap(iW, iX) || overlap(iW, iP)) return IMD_STATUS_ALIASING;
  if (!aligned(X, 8) || !aligned(packed, c::kRegionAlign) ||
      (need_ws > 0 && !aligned(ws, c::kRegionAlign)))
    return IMD_STATUS_MISALIGNED;
  if (packed_bytes < need || ws_bytes < need_ws) return IMD_STATUS_INSUFFICIENT_WORKSPACE;
  try {
    return imd::detail::enqueue_pack(pack->exec, pack->pl.layout, X, packed, ws, stream);
  } catch (...) {
    return IMD_STATUS_CUDA_ERROR;
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
  // v1.7 (§6b.5): a plan with a packed slot needs the packed pointers: imd_contract_packed.
  if (!plan->slot_key[0].empty() || !plan->slot_key[1].empty()) return IMD_STATUS_PACK_MISMATCH;
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

imd_status_t imd_contract_packed(imd_plan_t plan, double alpha, const imd_operand_t* A,
                                 const imd_operand_t* B, double beta, double* C, void* ws,
                                 size_t ws_bytes, cudaStream_t stream) {
  // DESIGN §6b.5 order: operands and packed slots; null pointers; empty output; ALIASING;
  // MISALIGNED; sizes.
  if (plan == nullptr || A == nullptr || B == nullptr) return IMD_STATUS_INVALID_VALUE;
  const imd::detail::ExecPlan& e = plan->exec;
  const imd_operand_t* op[2] = {A, B};
  int role[2];  // post-swap role of the caller's operand x
  for (int x = 0; x < 2; ++x) {
    role[x] = (x == 0) != e.swapped ? 0 : 1;
    if (plan->slot_key[x].empty()) {  // in-line slot: no packed operand may be passed
      if (op[x]->packed != nullptr || op[x]->pack != nullptr) return IMD_STATUS_PACK_MISMATCH;
    } else if (op[x]->pack == nullptr || op[x]->pack->pl.key != plan->slot_key[x]) {
      return IMD_STATUS_PACK_MISMATCH;  // host-only compare of O(rank) words
    }
  }
  // §6 null rule with `data`; a packed slot's buffer whenever it is read.
  const bool reads_ab = alpha != 0.0 && e.k != 0;
  const bool reads_pack = reads_ab && imd::detail::emulates(e);
  const uint64_t need = e.layout.total;
  if (C == nullptr || (reads_ab && (A->data == nullptr || B->data == nullptr)) ||
      (need > 0 && ws == nullptr))
    return IMD_STATUS_INVALID_VALUE;
  for (int x = 0; x < 2; ++x)
    if (!plan->slot_key[x].empty() && reads_pack && e.pack_bytes[role[x]] > 0 &&
        op[x]->packed == nullptr)
      return IMD_STATUS_INVALID_VALUE;
  if (e.path == imd::detail::Path::kNoop) return IMD_STATUS_SUCCESS;  // nothing touched
  // §5 row 6 (+ §6b.5): C vs every operand byte range; the workspace vs those and C. Packed
  // buffers may overlap each other and the FP64 operands (all are only read).
  const ByteInterval iC = byte_interval(plan->C, C), iW = bytes_at(ws, need);
  ByteInterval in[4] = {byte_interval(plan->A, A->data), byte_interval(plan->B, B->data), {}, {}};
  for (int x = 0; x < 2; ++x)
    if (!plan->slot_key[x].empty()) in[2 + x] = bytes_at(op[x]->packed, e.pack_bytes[role[x]]);
  if (overlap(iW, iC)) return IMD_STATUS_ALIASING;
  for (const ByteInterval& b : in)
    if (overlap(iC, b) || overlap(iW, b)) return IMD_STATUS_ALIASING;
  // §5 row 7: FP64 8 bytes, packed buffers and the workspace 256.
  if (!aligned(A->data, 8) || !aligned(B->data, 8) || !aligned(C, 8) ||
      !aligned(A->packed, c::kRegionAlign) || !aligned(B->packed, c::kRegionAlign) ||
      (need > 0 && !aligned(ws, c::kRegionAlign)))
    return IMD_STATUS_MISALIGNED;
  if (ws_bytes < need) return IMD_STATUS_INSUFFICIENT_WORKSPACE;
  for (int x = 0; x < 2; ++x)
    if (op[x]->packed != nullptr && op[x]->packed_bytes < e.pack_bytes[role[x]])
      return IMD_STATUS_INSUFFICIENT_WORKSPACE;
  // §5 step 8 role swap: the GEMM computes C^T = B^T A^T.
  const double* gA = e.swapped ? B->data : A->data;
  const double* gB = e.swapped ? A->data : B->data;
  const void* packed[2] = {nullptr, nullptr};  // by post-swap role
  for (int x = 0; x < 2; ++x)
    if (!plan->slot_key[x].empty()) packed[role[x]] = op[x]->packed;
  try {
    return imd::detail::enqueue_contract(e, alpha, gA, gB, beta, C, ws, stream, packed);
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
// Benchmark-only (A5): K5's elements per thread for later calls (1, 2, 4; 0: the rule).
IMD_API int imd_prof_set_k5_elems(int32_t elems) {
  if (elems != 0 && elems != 1 && elems != 2 && elems != 4) return 1;
  imd::detail::k5_elems_override() = elems;
  return 0;
}
// Benchmark-only (v1.10, DESIGN §6c.1): the measured occupancy of kernel IMD_PERF_KERNEL_*:
// out[0..3] = CTAs per SM, registers per thread, shared memory per CTA (static + dynamic),
// threads. K4: the store-mod kernel of the configuration (its occupancy from plan setup).
IMD_API int imd_prof_kernel_occupancy(int32_t kernel, int32_t* out) {
  if (out == nullptr || kernel < 0 || kernel >= IMD_PERF_KERNEL_COUNT) return 1;
  if (kernel >= IMD_PERF_KERNEL_K4 && kernel < IMD_PERF_KERNEL_K5) {
    const imd::detail::K4Ops& ops = imd::detail::k4_ops(kernel - IMD_PERF_KERNEL_K4);
    int occ_store = 0, occ_add = 0;
    if (ops.setup(&occ_store, &occ_add) != cudaSuccess) return 1;
    out[0] = occ_store, out[1] = 0, out[2] = ops.smem_bytes, out[3] = ops.threads;
    return 0;
  }
  return imd::detail::kernel_occupancy(kernel, out) == cudaSuccess ? 0 : 1;
}
// Benchmark-only (v1.16 §3.4.1): one W instantiation (2, 3) of a pack kernel (KF16, KF4,
// KL8, KL4, PACK); w = 0 is imd_prof_kernel_occupancy's row. Returns 0, 1 on error.
IMD_API int imd_prof_kernel_occupancy_w(int32_t kernel, int32_t w, int32_t* out) {
  if (out == nullptr || kernel < 0 || kernel >= IMD_PERF_KERNEL_COUNT) return 1;
  if (w == 0) return imd_prof_kernel_occupancy(kernel, out);
  return imd::detail::kernel_occupancy_w(kernel, w, out) == cudaSuccess ? 0 : 1;
}
#endif

}  // extern "C"
