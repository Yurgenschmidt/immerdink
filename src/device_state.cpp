// What the library knows about the CUDA device it runs on (issue 18, docs/NOTES-18.md): the device
// class, the table-set selection and the kind of code every kernel runs as. Computed once per device
// (the first plan creation or imd_device_info on it) and cached.
#include "device_state.hpp"

#include <cstring>
#include <mutex>

#include "host/k4_config.hpp"
#include "k4/k4_ops.hpp"
#include "plan.hpp"

namespace imd::detail {
namespace {

// The CUDA errors that mean "this device cannot run this binary": a property of the build and the
// driver, so they are cached like a success. Anything else (out of memory while loading, ...) is
// reported as it is and checked again next time.
bool is_permanent(cudaError_t e) {
  return e == cudaErrorNoKernelImageForDevice || e == cudaErrorUnsupportedPtxVersion ||
         e == cudaErrorInvalidPtx || e == cudaErrorInvalidKernelImage ||
         e == cudaErrorInvalidDeviceFunction || e == cudaErrorJitCompilerNotFound;
}

struct Slot {
  bool ready = false;
  imd_device_info_t info{};
};
constexpr int kMaxDevices = 64;
std::mutex g_mutex;
Slot g_slots[kMaxDevices];

void note_kernel(imd_device_info_t* r, int32_t id, cudaError_t e, int b, int p) {
  const int32_t code = imd_kernel_code_kind(static_cast<int32_t>(e), b, p, r->device.cc_major,
                                            r->device.cc_minor);
  ++r->kernels_checked;
  if (code > r->code_kind) r->code_kind = code;
  if (code == IMD_CODE_UNUSABLE && r->failing_kernel < 0) {
    r->failing_kernel = id;
    r->cuda_error = static_cast<int32_t>(e);
  }
}

// Fills `r` for `device`. Returns a CUDA error that must not be cached (transient), else success.
cudaError_t classify(int device, imd_device_info_t* r) {
  std::memset(r, 0, sizeof *r);
  r->struct_size = sizeof *r;
  r->failing_kernel = -1;
  r->k4_config = -1;
  imd_device_class_t& d = r->device;
  d.struct_size = sizeof d;
  int v = 0;
  cudaError_t e;
  const auto attr = [&](cudaDeviceAttr a, auto* dst) {
    e = cudaDeviceGetAttribute(&v, a, device);
    if (e == cudaSuccess) *dst = static_cast<std::remove_pointer_t<decltype(dst)>>(v);
    return e == cudaSuccess;
  };
  if (!attr(cudaDevAttrComputeCapabilityMajor, &d.cc_major) ||
      !attr(cudaDevAttrComputeCapabilityMinor, &d.cc_minor) ||
      !attr(cudaDevAttrMultiProcessorCount, &d.sm_count) ||
      !attr(cudaDevAttrL2CacheSize, &d.l2_bytes) ||
      !attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, &d.smem_optin_bytes)) {
    // With CUDA_MODULE_LOADING=EAGER a missing kernel image or too-new PTX shows at context creation.
    if (is_permanent(e)) {
      r->status = IMD_PLAN_UNSUPPORTED, r->code_kind = IMD_CODE_UNUSABLE;
      r->cuda_error = static_cast<int32_t>(e);
      r->selection.struct_size = sizeof r->selection;
      r->selection.match = IMD_MATCH_UNSUPPORTED, r->selection.table_id = -1, r->selection.row_index = -1;
      r->selection.approximate = 1;
      r->table_id = IMD_TABLE_SM89;
      return cudaSuccess;
    }
    return e;
  }
  cudaDeviceProp prop{};
  if ((e = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) return e;
  std::memcpy(d.name, prop.name, sizeof d.name - 1);  // 256 bytes in prop; d.name is zeroed, 63 kept
  d.name[sizeof d.name - 1] = '\0';
  r->selection.struct_size = sizeof r->selection;
  imd_device_select(&d, &r->selection);
  r->table_id = r->selection.table_id >= 0 ? r->selection.table_id : IMD_TABLE_SM89;
  if (r->selection.match != IMD_MATCH_UNSUPPORTED) {
    // Every kernel of the library, not only those of one plan: the plan's set depends on the path,
    // N, the pack word count, the fused launches and (K5) a variant picked at contract time.
    const int32_t first_w2[] = {IMD_PERF_KERNEL_KF16, IMD_PERF_KERNEL_KF4, IMD_PERF_KERNEL_KL8,
                                IMD_PERF_KERNEL_KL4, IMD_PERF_KERNEL_PACK};
    for (int32_t k = 0; k < IMD_PERF_KERNEL_COUNT; ++k) {
      if (k >= IMD_PERF_KERNEL_K4 && k < IMD_PERF_KERNEL_K4 + kK4ConfigCount) {
        int bs = 0, ps = 0, ba = 0, pa = 0;
        e = k4_ops(k - IMD_PERF_KERNEL_K4).versions(&bs, &ps, &ba, &pa);
        if (e != cudaSuccess && !is_permanent(e)) {
          cudaGetLastError();
          return e;
        }
        note_kernel(r, k, e, bs, ps);
        if (e == cudaSuccess) note_kernel(r, k, cudaSuccess, ba, pa);
        cudaGetLastError();
        continue;
      }
      bool pack = false;
      for (int32_t q : first_w2) pack = pack || q == k;
      for (int32_t w : {pack ? 2 : 0, 3}) {
        if (w == 3 && !pack) break;
        int32_t b = 0, p = 0;
        e = kernel_versions(k, w, &b, &p);
        if (e != cudaSuccess && !is_permanent(e)) {
          cudaGetLastError();
          return e;
        }
        note_kernel(r, k, e, b, p);
        cudaGetLastError();  // clear the (non-sticky) error state
      }
    }
  }
  if (r->selection.match != IMD_MATCH_UNSUPPORTED) {
    for (int32_t i = 0; i < kExtraKernels; ++i) {
      int32_t b = 0, p = 0;
      e = extra_kernel_versions(i, &b, &p);
      if (e != cudaSuccess && !is_permanent(e)) {
        cudaGetLastError();
        return e;
      }
      note_kernel(r, IMD_PERF_KERNEL_COUNT + i, e, b, p);
      cudaGetLastError();
    }
  }
  r->status = imd_plan_status_of(r->selection.match, r->code_kind);
  return cudaSuccess;
}

}  // namespace

cudaError_t device_state(int device, imd_device_info_t* out) {
  if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(g_mutex);
  Slot& s = g_slots[device];
  if (!s.ready) {
    imd_device_info_t r;
    const cudaError_t e = classify(device, &r);
    if (e != cudaSuccess) return e;
    s.info = r, s.ready = true;
  }
  *out = s.info;
  return cudaSuccess;
}

}  // namespace imd::detail
