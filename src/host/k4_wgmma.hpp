// The Hopper (wgmma) K4 backend: which device classes use it, which launches it may serve (legality) and which of those
// it serves by default (profitability), and the tile / cluster / scheduler choice (issue 22, docs/NOTES-22.md "Library
// integration (design)"). CUDA-free, header only (C++17), so the CPU tests (tests/test_k4_wgmma_select.cpp) and the CUDA side
// (src/gemm_k4.cu, src/k4/hopper/) read one definition.
//
// The backend is a second implementation of ONE K4 launch (host/k4_launches.hpp K4Launch): same launch structure, same
// launch count, same plan host bytes; the mma.sync K4 (src/k4/k4_cfg.cuh) stays the fallback of every launch the
// backend does not serve. Both produce the same W planes bit for bit (docs/NOTES-22.md, "Why it is bit-identical").
#pragma once

#include <cstdint>

#include "immerdink/immerdink_host.h"

namespace imd::detail {

// ---- backends -------------------------------------------------------------------------------------------------
// The backend a PLAN tries first (K4Plan::backend). kK4BackendWgmmaForced is "wgmma, forced": the profitability floor is lifted for every
// launch of the plan (legality never is). It is a property of the plan, fixed at plan creation, so a benchmark that resets the process-wide
// override right after imd_plan_create still times the forced path.
enum K4Backend : int32_t { kK4BackendMma = 0, kK4BackendWgmma = 1, kK4BackendWgmmaForced = 2 };
// The process-wide override REQUEST (test / benchmark hooks only, read at plan creation): kK4BackendAuto = the device class's rule, kK4BackendMma,
// kK4BackendWgmma = force the Hopper backend (the plan then gets kK4BackendWgmmaForced where the device has it).
constexpr int32_t kK4BackendAuto = -1;
constexpr bool k4_backend_valid(int32_t v) { return v >= kK4BackendAuto && v <= kK4BackendWgmma; }

// ---- device classes -------------------------------------------------------------------------------------------
// One row per K4 table set of the registry (host/device_class.cpp, IMD_TABLE_*). A class defaults to the wgmma backend only where
// the R2 evidence is (H100 and H200 share IMD_TABLE_SM90); sm_89 and sm_120 (which maps onto the sm_89 row) never do.
// The wgmma instructions exist on compute capability 9.0 only (sm_90a SASS): `k4_wgmma_device_ok` adds that, so a table-set
// row selected by a generation match on another capability never runs the backend.
constexpr bool kK4WgmmaTable[IMD_TABLE_COUNT] = {
    false,  // IMD_TABLE_SM89: RTX 4090 (and sm_120 by the registry): mma.sync only
    true,   // IMD_TABLE_SM90: H100, H200
};
constexpr bool k4_wgmma_device_ok(int32_t table_id, int32_t cc_major, int32_t cc_minor) {
  return table_id >= 0 && table_id < IMD_TABLE_COUNT && kK4WgmmaTable[table_id] && cc_major == 9 && cc_minor == 0;
}

// The backend a plan runs: the override request if there is one (a forced wgmma on a device without it is still mma: `available` false;
// where the device has it the plan is kK4BackendWgmmaForced), else the class's default (kK4BackendWgmma: floor applies). `available` = the
// class allows it, the build has it, the probe found its kernels.
constexpr int32_t k4_backend_resolve(int32_t override_backend, bool available, bool class_default) {
  if (!available) return kK4BackendMma;
  if (override_backend == kK4BackendAuto) return class_default ? kK4BackendWgmma : kK4BackendMma;
  return override_backend == kK4BackendWgmma ? kK4BackendWgmmaForced : kK4BackendMma;
}

// ---- tile / cluster / scheduler table -------------------------------------------------------------------------
// Variants compiled into the backend (src/k4/hopper/k4_wgmma_v*.cu). One today: CTA tile 128 x 256 x 128, cluster 2 x 1,
// cooperative (two consumer warpgroups), the fast epilogue (src/k4/hopper/epilogue_wg.cuh), store-mod and add-mod.
// R2 (docs/NOTES-22.md "Library integration (design)", table of the choice) measured it best at 4096^3 batch 14 on both cards
// (1360 / 1357 TOPS-eq, spread 5 / 8 %) and within 1-4 % of the best row at 8192^3 once the scheduler is told to swizzle.
constexpr int kK4WgmmaVariants = 1;
constexpr int kK4WgmmaTileM = 128, kK4WgmmaTileN = 256, kK4WgmmaTileK = 128;
// CUTLASS raster orders as the harness numbers them (rung1 Sched::raster).
enum K4Raster : int32_t { kRasterHeuristic = 0, kRasterAlongM = 1, kRasterAlongN = 2 };

// One K4 launch as the host sees it (host/k4_launches.hpp K4Launch of a plan + the call's pointers and strides).
struct K4WgmmaLaunch {
  int64_t m, nc, kc, zc;       // GEMM rows, columns of this n-hat part, K of this chunk, batches of this part
  int64_t k_hat;               // row stride of both operands (bytes = elements: int8)
  int64_t n_hat;               // row stride of W (the plan's full n-hat)
  int64_t batch_a, batch_b;    // operand batch strides (elements), as K4Call::batch_a / batch_b
  int64_t plane_w;             // W plane stride
  uintptr_t a, b, w;           // operand and plane base addresses of the launch
  int sms;                     // the device's SM count
};

struct K4WgmmaPick {
  int variant = -1;                // -1: not served (the mma.sync K4 runs this launch)
  int32_t swizzle = 1;             // CUTLASS scheduler max_swizzle_size (power of two)
  int32_t raster = kRasterHeuristic;
  const char* why = "";            // the reason a launch is not served (tests, benchmark records)
};

constexpr int64_t kTmaStrideLimit = int64_t{1} << 40;  // cuTensorMapEncodeTiled: global strides < 2^40 bytes
constexpr int64_t kTmaDimLimit = int64_t{1} << 32;     // and dimensions <= 2^32 (all ours are far below)

// LEGALITY: can the wgmma kernel produce exactly this launch's W bytes? A pure function of the launch. Everything here is a hard
// requirement of TMA (cuTensorMapEncodeTiled: 16-byte global address, strides multiples of 16 bytes below 2^40; CUTLASS's
// encoder only prints on failure and asserts nothing under NDEBUG, so an illegal launch must never reach it), of the 128-bit
// stores of the epilogue (HopperEpilogue: 16-byte W base, ldw and plane stride multiples of 16, n multiple of 16) or of CUTLASS's
// 32-bit tile arithmetic.
constexpr const char* k4_wgmma_illegal(const K4WgmmaLaunch& L) {
  if (L.m < 1 || L.nc < 1 || L.kc < 1 || L.zc < 1) return "empty launch";
  if (L.m > INT32_MAX || L.nc > INT32_MAX || L.kc > INT32_MAX) return "extent beyond int32";
  if (L.zc > 65535) return "batch count beyond 65535";
  if (L.k_hat % 16 != 0 || L.kc % 16 != 0 || L.nc % 16 != 0 || L.n_hat % 16 != 0) return "extent or stride not a multiple of 16";
  if (L.k_hat < L.kc) return "chunk longer than the row";
  if (L.k_hat >= kTmaStrideLimit || L.batch_a >= kTmaStrideLimit || L.batch_b >= kTmaStrideLimit) return "TMA stride limit (2^40)";
  if (L.batch_a % 16 != 0 || L.batch_b % 16 != 0) return "operand batch stride not a multiple of 16";
  // Overlapping batch planes (a packed operand whose blocks are shorter than the GEMM view reads, docs/NOTES-22.md) are legal for
  // the hardware but not shown to be accepted by every driver's encoder: the mma.sync K4 serves them.
  if (L.batch_a < L.m * L.k_hat || L.batch_b < L.n_hat * L.k_hat) return "overlapping operand planes";
  if (L.a % 16 != 0 || L.b % 16 != 0) return "operand base not 16-byte aligned";
  if (L.w % 16 != 0 || L.plane_w % 16 != 0 || L.plane_w < L.m * L.n_hat) return "W base or plane stride";
  const int64_t tm = (L.m + kK4WgmmaTileM - 1) / kK4WgmmaTileM, tn = (L.nc + kK4WgmmaTileN - 1) / kK4WgmmaTileN;
  if (tm * tn * L.zc > INT32_MAX) return "tile count beyond int32";
  return nullptr;
}

// PROFITABILITY (the default; a forced backend skips it): serve a launch where the estimated wgmma time beats the mma.sync K4's with a
// margin, per device class (docs/NOTES-22.md "Profitability rule": data, fit, validation, limits, how to re-fit; the constants are the
// R3 floor sweep of H100 and H200, tools/fit_k4_wgmma_profit.py re-derives them). For a launch of T = tm * tn * zc output tiles (the 128 x 256
// CTA tile; the mma.sync K4 has the same tile) on a device of S SMs, kt = ceil(kc / 128) k steps, w = T / S (waves, fractional):
//     delta_us = launch_us + w * (wave_us + wave_k_us * kt)        wgmma minus mma.sync (negative: wgmma faster)
//     mma_us   = mma_wave_k_us * w * kt                            lower estimate of the mma.sync K4 time (no per-wave fixed cost)
// served iff  m, nc >= min_mn  and  kc >= min_kc  and  T >= min_fill * S  and  -delta_us >= max(min_gain_us, min_gain_rel * mma_us).
// The two margins are the conservatism: an absolute one (the model's launch-level residual, about a microsecond on the shapes at the
// boundary) and one relative to the K4 time (a K4 gain of 10 % is about 2 % of a call whose K4 share is 20 %, the smallest share measured).
// min_mn / min_kc are the measured envelope (smallest m and n-part 512, smallest K chunk 256): the linear model is not extrapolated below it.
// min_fill is a prior and not a fit: a launch with fewer tiles than half the SMs leaves most of the persistent 2x1-cluster grid idle; the
// one loser measured (1024^3, one modulus per launch, 32 tiles = 0.24 of the SMs) lies below it, the nearest winner (72 tiles) above.
struct K4WgmmaProfit {
  bool enabled;            // false: the class has no evidence (nothing is served by default)
  double launch_us;        // per-launch cost of the wgmma kernel over the mma.sync kernel
  double wave_us;          // per wave, tile-fixed part (negative: faster epilogue)
  double wave_k_us;        // per wave and k step of 128 (negative: faster mainloop)
  double mma_wave_k_us;    // mma.sync K4 per wave and k step (R3 K4 phase 965 / 995 TOPS-eq on H100 -> 1.11-1.15)
  double min_gain_us;      // absolute margin per launch
  double min_gain_rel;     // margin relative to mma_us
  double min_fill;         // fewest tiles per launch, as a fraction of the SMs
  int64_t min_mn;          // fewest rows m and columns of the n-part (the measured envelope)
  int64_t min_kc;          // shortest K chunk (the measured envelope)
};
// One row per table set, like kK4WgmmaTable; the sm_89 row has no backend and so no rule. H100 and H200 share the sm_90 row (fit on both).
constexpr K4WgmmaProfit kK4WgmmaProfit[IMD_TABLE_COUNT] = {
    {false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0},                    // IMD_TABLE_SM89
    {true, 2.40, -0.57, -0.51, 1.13, 0.75, 0.10, 0.5, 512, 256},        // IMD_TABLE_SM90 (R3, 19 shapes x 2 cards, N = 14)
};
constexpr int kK4WgmmaDefaultSms = 132;  // used when the plan does not know the SM count

struct K4WgmmaEstimate {
  double delta_us;  // wgmma minus mma.sync, estimated
  double mma_us;    // the mma.sync K4 time estimate the relative margin uses
  bool serve;       // the rule's verdict (envelope, fill and gain)
  bool envelope_ok; // m, nc, kc inside the measured envelope
  bool fill_ok;
};
constexpr K4WgmmaEstimate k4_wgmma_estimate(const K4WgmmaLaunch& L, const K4WgmmaProfit& P) {
  const int64_t tm = (L.m + kK4WgmmaTileM - 1) / kK4WgmmaTileM, tn = (L.nc + kK4WgmmaTileN - 1) / kK4WgmmaTileN;
  const double tiles = static_cast<double>(tm * tn * L.zc), sms = L.sms > 0 ? L.sms : kK4WgmmaDefaultSms;
  const double kt = static_cast<double>((L.kc + kK4WgmmaTileK - 1) / kK4WgmmaTileK), w = tiles / sms;
  K4WgmmaEstimate e{};
  e.delta_us = P.launch_us + w * (P.wave_us + P.wave_k_us * kt);
  e.mma_us = P.mma_wave_k_us * w * kt;
  e.envelope_ok = L.m >= P.min_mn && L.nc >= P.min_mn && L.kc >= P.min_kc;
  e.fill_ok = tiles >= P.min_fill * sms;
  const double need = P.min_gain_us > P.min_gain_rel * e.mma_us ? P.min_gain_us : P.min_gain_rel * e.mma_us;
  e.serve = P.enabled && e.envelope_ok && e.fill_ok && -e.delta_us >= need;
  return e;
}

// The tile / scheduler choice. Scheduler arguments are set explicitly (R2: at 8192^3 the default scheduler cost the 128x256 kernels 27 %
// on H100): tiles per batch >= 1024 (8192^2 has 2048, 4096^2 has 512) -> swizzle 8, raster AlongN; below -> CUTLASS's default (swizzle 1,
// heuristic), which R2 measured at 4096^3. 2x1 cluster, queried persistent grid.
constexpr int64_t kK4WgmmaSwizzleTiles = 1024;
constexpr K4WgmmaPick k4_wgmma_pick(const K4WgmmaLaunch& L, bool force, int32_t table_id = IMD_TABLE_SM90) {
  K4WgmmaPick p;
  if (const char* bad = k4_wgmma_illegal(L)) {
    p.why = bad;
    return p;
  }
  if (!force) {
    if (table_id < 0 || table_id >= IMD_TABLE_COUNT || !kK4WgmmaProfit[table_id].enabled) {
      p.why = "no profitability rule for this device class";
      return p;
    }
    const K4WgmmaEstimate e = k4_wgmma_estimate(L, kK4WgmmaProfit[table_id]);
    if (!e.envelope_ok) {
      p.why = "below the profitability rule (outside the measured envelope)";
      return p;
    }
    if (!e.fill_ok) {
      p.why = "below the profitability rule (fill)";
      return p;
    }
    if (!e.serve) {
      p.why = "below the profitability rule (estimated gain)";
      return p;
    }
  }
  p.variant = 0;
  const int64_t tm = (L.m + kK4WgmmaTileM - 1) / kK4WgmmaTileM, tn = (L.nc + kK4WgmmaTileN - 1) / kK4WgmmaTileN;
  if (tm * tn >= kK4WgmmaSwizzleTiles) p.swizzle = 8, p.raster = kRasterAlongN;
  p.why = force ? "forced" : "served";
  return p;
}

}  // namespace imd::detail
