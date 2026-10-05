// Accuracy-aware planning (DESIGN v1.12 §6f): the rigorous §3.8 error bound composed per metric
// and the typical (Hoeffding) estimate of one configuration, from scalars. The only
// implementation: imd_err_estimate, imd_err_min_moduli (accuracy.cpp), imd_perf_predict and
// imd_perf_rank (perf.cpp) all call err_compute. Reference: spec/params.py error_estimate.
#pragma once

#include <cstdint>

#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

// A validated imd_err_request_t (defaults: NORMWISE, no target, BOUND basis, p = 2^-20).
struct ErrReq {
  int32_t metric = IMD_ERR_NORMWISE, basis = IMD_ERR_BASIS_BOUND;
  double target = 0, cond = 0, norm_product = 0, nu_a = 0, nu_b = 0, fail_prob = 0;
};

// The estimates of one configuration (DESIGN §6f.2); err_status from the target and basis.
struct ErrOut {
  double bound = 0, typical = 0, unit_bound = 0, unit_typical = 0;
  int32_t err_status = IMD_ERR_UNCHECKED;
};

// false for an invalid request (the rules of immerdink_host.h); NULL is the default request.
IMD_HOST_INTERNAL bool err_request_valid(const imd_err_request_t* r, ErrReq* q);

// The estimates of (N, rounding, dispatch, k, outputs) under q. N in 2..20, k, outputs >= 0.
IMD_HOST_INTERNAL ErrOut err_compute(int N, int32_t rounding, int32_t dispatch, int64_t k,
                                     int64_t outputs, const ErrReq& q);

}  // namespace imd::detail
