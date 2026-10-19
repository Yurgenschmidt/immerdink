// K4 tile configurations (DESIGN §4 "K4 configurations", A3): the table every part of the
// library uses, and the plan-time selection rule. CUDA-free: libimmerdink_host owns the
// choice (k4_select, reported by imd_lowering), libimmerdink instantiates one K4Gemm per
// entry (src/k4/) and runs the plan's choice. Every configuration computes the same W
// planes bit for bit; only the tile -> CTA mapping differs.
#pragma once

#include <cstdint>

#include "checked.hpp"
#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

struct ExecPlan;

// One K4Gemm instantiation: threadblock tile (tile_k = 64), warp tile (warp_n = 64: the
// register epilogue's quad transpose), cp.async stages, and the CTAs per SM it is sized
// for (shared memory and registers; libimmerdink measures the real occupancy).
struct K4ConfigDesc {
  int32_t tile_m, tile_n, warp_m, warp_n, stages, ctas_per_sm;
};
constexpr int kK4ConfigCount = IMD_K4_CONFIG_COUNT;
constexpr K4ConfigDesc kK4Configs[kK4ConfigCount] = {
    {128, 256, 64, 64, 3, 1},  // IMD_K4_128x256: 8 warps, 72 KB smem (the A2 kernel)
    {128, 128, 64, 64, 3, 2},  // IMD_K4_128x128: 4 warps, 48 KB
    {64, 128, 32, 64, 4, 2},   // IMD_K4_64x128:  4 warps, 48 KB
    {64, 64, 32, 64, 4, 3},    // IMD_K4_64x64:   2 warps, 32 KB
};
constexpr int k4_threads(const K4ConfigDesc& c) {
  return 32 * (c.tile_m / c.warp_m) * (c.tile_n / c.warp_n);
}

// SM count the selection assumes when the caller gives none (imd_lowering_t sm_count = 0):
// the RTX 4090 the rule was fitted on.
constexpr int32_t kK4ReferenceSms = 128;

// The selection model (DESIGN §4 "K4 configurations"; fitted in docs/NOTES-A3.md). For
// configuration c over one group's launch (tiles = ceil(m/tile_m) ceil(n_hat/tile_n) g b),
// the busiest SM gets j = ceil(tiles / SMs) tiles: j / o full rounds of o = ctas_per_sm
// concurrent CTAs, then (if j mod o = r > 0) one round of r. A round of q CTAs costs
//     round_ns + q * tile_m * tile_n * k_hat * mac_fs[q]
// round_ns: fixed time per round (prologue, epilogue, launch); mac_fs[q]: time per MAC on an
// SM running q CTAs of c (index 0 unused). Least-squares fit (relative error) on nsys K4
// kernel times of every configuration on 84 (shape, N) cases, RTX 4090 at its power cap.
struct K4Cost {
  int64_t round_ns;
  int64_t mac_fs[4];
};
constexpr K4Cost kK4Costs[kK4ConfigCount] = {
    {2716, {0, 418, 0, 0}},       // 128x256
    {2550, {0, 453, 456, 0}},     // 128x128
    {1424, {0, 505, 612, 0}},     // 64x128
    {1333, {0, 944, 1088, 802}},  // 64x64
};
// The H100 table (issue 20, docs/NOTES-20.md): the fit of bench/k4_fit.py on the R1 tile sweep of 2026-10-19 on an H100 (132 SMs; all four tiles timed on 35 views at N = 8 and 14,
// K4 phase marks of at least 0.1 ms), under the same concurrency assumption as the 4090's (ctas_per_sm 1, 2, 2, 3). It changes two picks against the 4090 table (r_mm N=14 and x512_32k N=8)
// and was validated on an H200 (same picks, mean regret 2.20% to 0.80%).
constexpr K4Cost kK4CostsSm90[kK4ConfigCount] = {
    {3383, {0, 265, 0, 0}},       // 128x256
    {2838, {0, 464, 281, 0}},     // 128x128
    {737, {0, 584, 414, 0}},      // 64x128
    {363, {0, 949, 685, 505}},    // 64x64
};
// A configuration other than IMD_K4_128x256 is chosen only if its estimate is below
// (1000 - margin) / 1000 of 128x256's (model error and measurement noise).
constexpr int64_t kK4MarginPermille = 50;

// One fitted set of K4 constants per IMD_TABLE_* (issue 18, docs/NOTES-18.md D2). A table also owns
// the CTAs per SM the cost model assumes for each configuration: kK4Configs' value is a 4090 figure
// (the plan itself uses the measured occupancy), and another architecture may differ.
struct K4Table {
  K4Cost costs[kK4ConfigCount];
  int32_t ctas_per_sm[kK4ConfigCount];
  int64_t margin_permille;
};
constexpr K4Table kK4Tables[IMD_TABLE_COUNT] = {
    {{kK4Costs[0], kK4Costs[1], kK4Costs[2], kK4Costs[3]},
     {kK4Configs[0].ctas_per_sm, kK4Configs[1].ctas_per_sm, kK4Configs[2].ctas_per_sm,
      kK4Configs[3].ctas_per_sm},
     kK4MarginPermille},
    {{kK4CostsSm90[0], kK4CostsSm90[1], kK4CostsSm90[2], kK4CostsSm90[3]},
     {kK4Configs[0].ctas_per_sm, kK4Configs[1].ctas_per_sm, kK4Configs[2].ctas_per_sm,
      kK4Configs[3].ctas_per_sm},
     kK4MarginPermille},
};

// The round decomposition of the model above for one launch of configuration c over gb = g b
// batches: the busiest SM runs `full` rounds of ctas_per_sm concurrent CTAs, then (rest > 0)
// one round of `rest` CTAs; every CTA does tile_macs = tile_m tile_n k_hat MACs. Saturating
// unsigned 128-bit integers. Shared by k4_estimate_fs and the performance model (DESIGN §6c,
// perf.cpp: the same decomposition with calibrated constants).
struct K4Rounds {
  u128 full = 0;
  int32_t rest = 0;
  u128 tile_macs = 0;
};
IMD_HOST_INTERNAL K4Rounds k4_rounds(int c, int64_t m, int64_t n_hat, int64_t k_hat, int64_t gb,
                                     int32_t sms, int32_t table = IMD_TABLE_SM89);

// The estimate above in femtoseconds, computed in saturating unsigned 128-bit integers
// (clamped at 2^127 - 1 after every operation). gb = g b.
IMD_HOST_INTERNAL u128 k4_estimate_fs(int c, int64_t m, int64_t n_hat, int64_t k_hat,
                                      int64_t gb, int32_t sms, int32_t table = IMD_TABLE_SM89);

// The K4 configuration of a plan with the emulated GEMM (DESIGN §4): a deterministic
// function of the GEMM view (m, n_hat, k_hat, b), N, g and the device's SM count: the
// smallest estimate, where a configuration other than 128x256 must beat 128x256's by the
// margin. Returns IMD_K4_128x256 when the emulated GEMM does not run.
IMD_HOST_INTERNAL int32_t k4_select(const ExecPlan& e, int32_t sms, int32_t table = IMD_TABLE_SM89);

// Plan-time autotuning (imd_options_t.k4_tune, DESIGN §6): the device bytes imd_plan_create
// allocates temporarily to time the candidates on group 1's K4 launches (RA, RB and group
// 1's W planes, one allocation), or 0 if no tuning runs for this plan (no emulated GEMM, or
// more than kK4TuneMaxBytes). Independent of the k4_tune option itself.
constexpr uint64_t kK4TuneMaxBytes = uint64_t{256} << 20;
IMD_HOST_INTERNAL uint64_t k4_tune_bytes(const ExecPlan& e);

}  // namespace imd::detail
