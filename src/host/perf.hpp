// The performance model (DESIGN v1.8 / v1.10 §6c): per-phase features of a plan, exactly from
// its lowering and the shared launch rules (analysis.hpp), and the calibration (parameters per
// kernel class, measured occupancy, the clock / power, cache and band data). The C ABI is in
// perf.cpp (imd_perf_*, include/immerdink/immerdink_host.h). Pure functions, no state.
#pragma once

#include <cstdint>

#include "analysis.hpp"
#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

// One phase of a call: phase / class ids, launches, the features x (the model is
// sum_j param[cls * IMD_PERF_MAX_PARAMS + j] x[j], in ns, the clock-bound terms scaled by
// f_ref / f; K4: max of j != 4 and j = 4), its MEMORY_MODEL §4 bound inputs: DRAM bytes and
// ops (peak: 0 none, 1 int8 tensor-core ops, 2 FP64 flops), and (v1.10) the input bytes it
// reads first (the COLD / WARM term).
struct PerfPhase {
  int32_t phase = 0, cls = 0;
  int64_t launches = 0;
  double x[IMD_PERF_MAX_PARAMS] = {0, 0, 0, 0, 0, 0};
  double bytes = 0, ops = 0;
  int peak = 0;
  double in_bytes = 0;  // v1.10: the call's input bytes this phase reads first
};
struct PerfPhases {
  PerfPhase ph[IMD_PERF_MAX_PHASES];
  int32_t n = 0;
  int32_t k4_config = IMD_K4_128x256;  // the K4 configuration of the K4 phase
  int64_t launches = 0;                // kernel launches of the call
  // v1.10 (COLD / WARM): the call's input bytes I and its footprint F (inputs + workspace /
  // pack buffers + the output), DESIGN §6c.1 "Cache state".
  double inputs = 0, footprint = 0;
};

// The phases of one imd_contract of a SUCCESS plan (analysis `an`, plan data `e`), in launch
// order, the call term last. flags: IMD_PERF_BETA_NONZERO / IMD_PERF_ALPHA_ZERO; k4: -1 =
// k4_select for the calibration's sm_count, else that configuration; the calibration gives the
// SM count, the L2 of K4's DRAM term and the occupancy of the chains (cal.kernel, resolved:
// see perf_resolve_calib). v1.15 (DESIGN §6c.1 "The eps path — retired"): no eps fraction (the
// v1.14 pack emitter costs the same for every eps; the op_eps slots are 0).
IMD_HOST_INTERNAL void perf_contract_phases(const Analysis& an, const ExecPlan& e, int32_t flags,
                                            int32_t k4, const imd_perf_calib_t& cal,
                                            PerfPhases* out);

// The phases of one imd_pack of a SUCCESS pack lowering (none for an empty pack).
IMD_HOST_INTERNAL void perf_pack_phases(const PackLowered& pl, const imd_perf_calib_t& cal,
                                        PerfPhases* out);

// log2 T(N), exactly from the §2 table (not floor(log2 T)).
IMD_HOST_INTERNAL double perf_log2_T(int N);

// Predicted time of a phase in seconds under calibration c, its clock-bound terms multiplied
// by `scale` (= f_ref / f; 1 at the reference clock).
IMD_HOST_INTERNAL double perf_phase_seconds(const imd_perf_calib_t& c, const PerfPhase& p,
                                            double scale = 1.0);

}  // namespace imd::detail
