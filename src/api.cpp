// C ABI entry points (docs/DESIGN.md §6). No exception crosses this boundary.
#include <array>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include <cuda_runtime_api.h>

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "index_map.hpp"
#include "plan.hpp"
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
  imd::detail::ExecPlan exec;  // exec.vec_host: pinned, owned (freed by the destructor)
  TensorCopy A, B, C;          // for M3's aliasing check (§5 row 6)

  imd_plan_st(const imd_options_t& o, const imd::detail::Lowered& lw, const imd_tensor_t& a,
              const imd_tensor_t& b, const imd_tensor_t& c)
      : options(o), lowered(lw), A(a), B(b), C(c) {}
  imd_plan_st(const imd_plan_st&) = delete;
  imd_plan_st& operator=(const imd_plan_st&) = delete;
  ~imd_plan_st() {
    if (exec.vec_host != nullptr) cudaFreeHost(const_cast<int64_t*>(exec.vec_host));
  }
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
  // §5 row 5, GemmCoord part: only when the emulated GEMM runs (EMULATE, non-empty output,
  // k >= 2); not for DIRECT, K-empty or k = 0.
  if (out->options.mode == IMD_EMULATE && !lw.empty_output() && lw.k >= 2 &&
      !imd::detail::gemm_dims_ok(lw))
    return IMD_STATUS_INVALID_VALUE;
  const auto layout = imd::detail::workspace_layout({lw.m, lw.n, lw.k, lw.b,
                                                     out->options.num_moduli,
                                                     out->options.residue_group,
                                                     out->options.mode == IMD_DIRECT});
  if (!layout) return IMD_STATUS_INVALID_VALUE;
  if (layout->total > SIZE_MAX) return IMD_STATUS_INVALID_VALUE;
  out->layout = *layout;
  return IMD_STATUS_SUCCESS;
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
    return IMD_STATUS_ALLOC_FAILED;
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
    const int N = an.options.num_moduli;
    const bool rn = an.options.rounding == IMD_RN;
    // §3.2 RN acceptance (accuracy policy): reject kappa >= T when the emulated GEMM runs.
    const bool emulated = !lw.empty_output() && an.options.mode == IMD_EMULATE && lw.k > 1;
    const Big T = big_from_limbs(c::kTables.T[N]);
    const Big kappa = big_from_u64(ceil_sqrt(static_cast<uint64_t>(lw.k)));
    if (emulated && rn && big_ge(kappa, T)) return IMD_STATUS_NOT_SUPPORTED;

    auto p = std::make_unique<imd_plan_st>(an.options, lw, *A, *B, *C);
    imd::detail::ExecPlan& e = p->exec;
    e.swapped = lw.swapped;
    e.m = lw.m, e.n = lw.n, e.k = lw.k, e.b = lw.b;
    e.k_hat = (lw.k + c::kKAlign - 1) / c::kKAlign * c::kKAlign;
    e.n_hat = (lw.n + c::kNAlign - 1) / c::kNAlign * c::kNAlign;
    e.N = N;
    e.g = an.options.residue_group;
    e.trunc = !rn;
    e.layout = an.layout;
    e.path = lw.empty_output()                                    ? imd::detail::Path::kNoop
             : (an.options.mode == IMD_DIRECT || lw.k == 1) ? imd::detail::Path::kDirect
                                                            : imd::detail::Path::kGemm;
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
        const imd::detail::PackPivot pv = imd::detail::pack_pivot(lw, x);
        e.pack_along_rows[x] = pv.along_rows;
        e.pack_radix[x] = pv.radix;
      }
    }
    if (e.path != imd::detail::Path::kNoop) {
      // §5 step 10: scatter vectors in pinned host memory (K0 copies them asynchronously).
      e.vec_count = imd::detail::scatter_vector_count(lw);
      void* host = nullptr;
      const cudaError_t err =
          cudaMallocHost(&host, static_cast<size_t>(e.vec_count) * sizeof(int64_t));
      if (err != cudaSuccess) {
        cudaGetLastError();  // clear the (non-sticky) error state
        return err == cudaErrorMemoryAllocation ? IMD_STATUS_ALLOC_FAILED : IMD_STATUS_CUDA_ERROR;
      }
      e.vec_host = static_cast<const int64_t*>(host);
      imd::detail::build_scatter_vectors(lw, static_cast<int64_t*>(host));
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
