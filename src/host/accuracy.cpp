// Accuracy-aware planning (DESIGN v1.12 §6f): imd_err_estimate, imd_err_min_moduli and the
// shared err_compute that imd_perf_predict / imd_perf_rank use. Pure: no global or static
// mutable state, thread-safe. The formulas are DESIGN §6f.2 (immerdink_host.h repeats them);
// the reference is spec/params.py error_estimate (exact rational enclosure; test host_accuracy).
#include "accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "analysis.hpp"  // kappa_at_least_T, ceil_sqrt_u64
#include "index_map.hpp"  // lower, gemm_dims_ok
#include "immerdink/detail/constants.hpp"

namespace imd::detail {
namespace {

namespace c = imd::constants;

constexpr double kU = 0x1p-53;                  // unit roundoff
constexpr double kInflate = 1.0 + 0x1p-40;      // covers the ~30 roundings below (each 2^-53)
constexpr double kDefaultFailProb = 0x1p-40;      // review 2026-10-05 (was 2^-20)
// ABSOLUTE bounds can be subnormal, where the relative inflation is a no-op: an absolute term.
// EMULATE: 2^-1075 covers a gradual underflow of D's single final rounding (|fl(y) - y| <=
// u|y| + 2^-1075) and the remaining 2^-1071 - 2^-1075 the few subnormal roundings of the
// computation (each <= 2^-1075). DIRECT adds (k + 16) 2^-1074 >= 2k 2^-1075: each of the k fmas
// may underflow (<= 2^-1075 each) plus the computation.
constexpr double kAbsTiny = 0x1p-1071;
constexpr uint32_t kErrRequestV8Size = sizeof(imd_err_request_t);  // 64; later: appended fields
constexpr uint32_t kErrV8Size = sizeof(imd_err_t);                 // 72
static_assert(kErrRequestV8Size == 64 && kErrV8Size == 72, "v8 error structs");
constexpr double kInf = std::numeric_limits<double>::infinity();

// T(N) as a double (relative error a few 2^-53: the limbs summed from the top).
double T_of(int N) {
  double t = 0;
  for (int w = c::kMaxLimbs - 1; w >= 0; --w) t = t * 4294967296.0 + c::kTables.T[N][w];
  return t;
}

bool finite_ge(double v, double lo) { return std::isfinite(v) && v >= lo; }

// The metric composition of a per-(i, j) coefficient E (EMULATE: the final rounding adds
// E u + u; DESIGN §6f.2, PROOFS L3c). E (1 + u) is written E + E u: 1 + 2^-53 rounds to 1.0.
double compose(const ErrReq& q, double E) {
  const double n = E + E * kU + kU;
  switch (q.metric) {
    case IMD_ERR_ELEMENTWISE: {
      const double Er = E * q.cond;
      return Er + Er * kU + kU;
    }
    case IMD_ERR_ABSOLUTE: return n * q.norm_product;
    default: return n;
  }
}

double scale_of(const ErrReq& q) {
  return q.metric == IMD_ERR_ELEMENTWISE ? q.cond
         : q.metric == IMD_ERR_ABSOLUTE  ? q.norm_product
                                         : 1.0;
}

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

bool out_valid(const imd_err_t* out) {
  return out != nullptr && out->struct_size >= kErrV8Size;
}

void finish(imd_err_t& r, imd_err_t* out) {
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
}

void fill(imd_err_t* r, int N, int32_t rounding, int32_t dispatch, int64_t k, int64_t outputs,
          const ErrOut& o) {
  r->num_moduli = N, r->rounding = rounding, r->dispatch = dispatch;
  r->k = k, r->outputs = outputs;
  r->bound = o.bound, r->typical = o.typical;
  r->unit_bound = o.unit_bound, r->unit_typical = o.unit_typical;
  r->err_status = o.err_status;
}

}  // namespace

bool err_request_valid(const imd_err_request_t* r, ErrReq* q) {
  *q = ErrReq{};
  if (r == nullptr) return true;
  if (r->struct_size < kErrRequestV8Size) return false;
  if (r->reserved != 0) return false;  // reserved for flags (e.g. an alpha rounding term)
  if (r->metric < IMD_ERR_NORMWISE || r->metric > IMD_ERR_ABSOLUTE) return false;
  if (!finite_ge(r->target, 0)) return false;
  if (r->metric == IMD_ERR_ELEMENTWISE && !finite_ge(r->cond, 1)) return false;
  if (r->metric == IMD_ERR_ABSOLUTE && !(std::isfinite(r->norm_product) && r->norm_product > 0))
    return false;
  for (const double v : {r->l1_ratio_a, r->l1_ratio_b})
    if (!(v == 0 || finite_ge(v, 1))) return false;
  if (r->basis != IMD_ERR_BASIS_BOUND && r->basis != IMD_ERR_BASIS_TYPICAL) return false;
  if (!(r->fail_prob == 0 || (r->fail_prob > 0 && r->fail_prob < 1))) return false;  // NaN: no
  q->metric = r->metric, q->basis = r->basis;
  q->target = r->target, q->cond = r->cond, q->norm_product = r->norm_product;
  q->nu_a = r->l1_ratio_a, q->nu_b = r->l1_ratio_b, q->fail_prob = r->fail_prob;
  return true;
}

ErrOut err_compute(int N, int32_t rounding, int32_t dispatch, int64_t k, int64_t outputs,
                   const ErrReq& q) {
  ErrOut o;
  if (dispatch == IMD_DISPATCH_NOOP || dispatch == IMD_DISPATCH_SCALE || k <= 0) {
    // no product: 0
  } else if (dispatch == IMD_DISPATCH_DIRECT) {
    // The native FMA chain (§3.7): gamma_k relative to sum |a b| <= ||a|| ||b||, the final
    // rounding included; no bound (+inf) once k u >= 1/2 (gamma_k >= 1 there anyway).
    const double ku = static_cast<double>(k) * kU;
    if (ku >= 0.5) {
      o.unit_bound = o.bound = kInf;
    } else {
      o.unit_bound = ku / (1 - ku) * kInflate;
      o.bound = ku / (1 - ku) * scale_of(q) * kInflate;
      if (q.metric == IMD_ERR_ABSOLUTE)
        o.bound += (static_cast<double>(k) + 16) * 0x1p-1074;  // k < 2^52: exact
    }
    o.unit_typical = o.unit_bound, o.typical = o.bound;
  } else {
    const bool rn = rounding == IMD_RN;
    const uint64_t kap = ceil_sqrt_u64(static_cast<uint64_t>(k));
    if (rn && kappa_at_least_T(kap, N)) {  // §3.2 RN acceptance: no bound
      o.unit_bound = o.bound = o.unit_typical = o.typical = kInf;
    } else {
      const double kd = static_cast<double>(k);
      const double T = T_of(N);
      const double delta = rn ? 0.5 : 1.0;
      const double d = rn ? T - 0.5 * static_cast<double>(kap) : T;
      const double s2 = 1 + kd * 0x1p-60;
      const double s = std::sqrt(s2);
      const double rk = std::sqrt(kd);
      const double nu_a = q.nu_a == 0 ? rk : std::min(q.nu_a, rk);
      const double nu_b = q.nu_b == 0 ? rk : std::min(q.nu_b, rk);
      const double second = 4 * delta * delta * kd * s2 / (d * d);
      const double E = 2 * delta * (nu_a + nu_b) * s / d + second;
      o.unit_bound = E * kInflate;
      o.bound = compose(q, E) * kInflate;
      if (q.metric == IMD_ERR_ABSOLUTE) o.bound += kAbsTiny;
      if (rn) {
        // Hoeffding + a union bound over the outputs (DESIGN §6f.2), capped at the bound.
        const double M = static_cast<double>(std::max<int64_t>(outputs, 1));
        const double p = q.fail_prob > 0 ? q.fail_prob : kDefaultFailProb;
        const double h = s / d * std::sqrt(8 * std::log(2 * M / p)) + second;
        o.unit_typical = std::min(o.unit_bound, h);
        o.typical = std::min(o.bound, compose(q, o.unit_typical) +
                                          (q.metric == IMD_ERR_ABSOLUTE ? kAbsTiny : 0.0));
      } else {
        o.unit_typical = o.unit_bound, o.typical = o.bound;
      }
    }
  }
  if (q.target > 0) {
    const double v = q.basis == IMD_ERR_BASIS_TYPICAL ? o.typical : o.bound;
    o.err_status = v <= q.target ? IMD_ERR_MET : IMD_ERR_MISSED;
  }
  return o;
}

}  // namespace imd::detail

extern "C" {

imd_status_t imd_err_estimate(int32_t num_moduli, int32_t rounding, int32_t dispatch, int64_t k,
                              int64_t outputs, const imd_err_request_t* request, imd_err_t* out) {
  using namespace imd::detail;
  if (!out_valid(out)) return IMD_STATUS_INVALID_VALUE;
  ErrReq q;
  if (num_moduli < c::kNMin || num_moduli > c::kNMax || (rounding != IMD_RN && rounding != IMD_TRUNC) ||
      dispatch < IMD_DISPATCH_NOOP || dispatch > IMD_DISPATCH_SCALE || k < 0 || outputs < 0 ||
      !err_request_valid(request, &q))
    return IMD_STATUS_INVALID_VALUE;
  imd_err_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  fill(&r, num_moduli, rounding, dispatch, k, outputs,
       err_compute(num_moduli, rounding, dispatch, k, outputs, q));
  r.status = IMD_STATUS_SUCCESS;
  finish(r, out);
  return IMD_STATUS_SUCCESS;
}

imd_status_t imd_err_min_moduli(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                                const imd_options_t* options, int32_t n_lo, int32_t n_hi,
                                const imd_err_request_t* request, imd_err_t* out) {
  using namespace imd::detail;
  if (!out_valid(out) || options == nullptr || request == nullptr) return IMD_STATUS_INVALID_VALUE;
  ErrReq q;
  if (!err_request_valid(request, &q) || !(q.target > 0)) return IMD_STATUS_INVALID_VALUE;
  const int lo = n_lo == 0 ? c::kNMin : n_lo, hi = n_hi == 0 ? c::kNMax : n_hi;
  if (lo < c::kNMin || hi > c::kNMax || lo > hi) return IMD_STATUS_INVALID_VALUE;
  imd_err_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = guarded([&] {
    // The lowering only (§5 rows 1-5 and the mode's GemmCoord rule; k, M and the dispatch do
    // not depend on N, g or vec_mode). Not the workspace layout: it depends on N and g, and
    // must not reject a shape that is valid at the N found. The §3.2 RN acceptance is the
    // estimate's own +inf at small N.
    const imd_options_t& o = *options;
    if ((o.rounding != IMD_RN && o.rounding != IMD_TRUNC) ||
        (o.mode != IMD_EMULATE && o.mode != IMD_DIRECT) || A == nullptr || B == nullptr ||
        C == nullptr)
      return IMD_STATUS_INVALID_VALUE;
    Lowered lw;
    if (const imd_status_t st = lower(*A, *B, *C, &lw); st != IMD_STATUS_SUCCESS) return st;
    if (o.mode == IMD_EMULATE && !lw.empty_output() && lw.k >= 2 && !gemm_dims_ok(lw))
      return IMD_STATUS_INVALID_VALUE;
    const int32_t dispatch = lw.empty_output()                     ? IMD_DISPATCH_NOOP
                             : lw.k == 0                           ? IMD_DISPATCH_SCALE
                             : (o.mode == IMD_DIRECT || lw.k == 1) ? IMD_DISPATCH_DIRECT
                                                                   : IMD_DISPATCH_EMULATE;
    const int64_t outputs = lw.b * lw.m * lw.n;
    for (int N = lo; N <= hi; ++N) {
      const ErrOut eo = err_compute(N, o.rounding, dispatch, lw.k, outputs, q);
      if (eo.err_status == IMD_ERR_MET || N == hi) {
        fill(&r, N, o.rounding, dispatch, lw.k, outputs, eo);
        if (eo.err_status == IMD_ERR_MET) return IMD_STATUS_SUCCESS;
        r.num_moduli = 0;
        return IMD_STATUS_NOT_SUPPORTED;
      }
    }
    return IMD_STATUS_INVALID_VALUE;  // unreachable
  });
  if (r.status != IMD_STATUS_SUCCESS && r.status != IMD_STATUS_NOT_SUPPORTED) {
    const uint32_t size = r.struct_size;
    const int32_t st = r.status;
    std::memset(&r, 0, sizeof r);
    r.struct_size = size, r.status = st;
  }
  finish(r, out);
  return static_cast<imd_status_t>(r.status);
}

}  // extern "C"
