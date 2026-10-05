// The performance model (DESIGN v1.8 §6c): per-phase features of a plan, exactly from its
// lowering, and the calibration (parameters per kernel class). The C ABI is in perf.cpp
// (imd_perf_*, include/immerdink/immerdink_host.h). Pure functions, no state.
#pragma once

#include <cstdint>

#include "analysis.hpp"
#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

// One phase of a call: phase / class ids, launches, the features x (the model is
// sum_j param[cls * IMD_PERF_MAX_PARAMS + j] x[j], in ns; K4: max of j < 4 and j = 4), and
// its MEMORY_MODEL §4 bound
// inputs: DRAM bytes and ops (peak: 0 none, 1 int8 tensor-core ops, 2 FP64 flops).
struct PerfPhase {
  int32_t phase = 0, cls = 0;
  int64_t launches = 0;
  double x[IMD_PERF_MAX_PARAMS] = {0, 0, 0, 0, 0, 0};
  double bytes = 0, ops = 0;
  int peak = 0;
};
struct PerfPhases {
  PerfPhase ph[IMD_PERF_MAX_PHASES];
  int32_t n = 0;
  int32_t k4_config = IMD_K4_128x256;  // the K4 configuration of the K4 phase
  int64_t launches = 0;                // kernel launches of the call
};

// The phases of one imd_contract of a SUCCESS plan (analysis `an`, plan data `e`), in launch
// order, the call term last. flags: IMD_PERF_BETA_NONZERO / IMD_PERF_ALPHA_ZERO; k4: -1 =
// k4_select for `sms`, else that configuration; l2: the device's L2 bytes (K4's DRAM term).
IMD_HOST_INTERNAL void perf_contract_phases(const Analysis& an, const ExecPlan& e, int32_t flags,
                                            int32_t k4, int32_t sms, double l2,
                                            PerfPhases* out);

// The phases of one imd_pack of a SUCCESS pack lowering (none for an empty pack).
IMD_HOST_INTERNAL void perf_pack_phases(const PackLowered& pl, PerfPhases* out);

// Predicted time of a phase in seconds under calibration c.
IMD_HOST_INTERNAL double perf_phase_seconds(const imd_perf_calib_t& c, const PerfPhase& p);

}  // namespace imd::detail
