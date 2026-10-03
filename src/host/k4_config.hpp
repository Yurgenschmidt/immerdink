// K4 tile configurations (DESIGN §4 "K4 configurations", A3): the table every part of the
// library uses, and the plan-time selection rule. CUDA-free: libimmerdink_host owns the
// choice (k4_select, reported by imd_lowering), libimmerdink instantiates one K4Gemm per
// entry (src/k4/) and runs the plan's choice. Every configuration computes the same W
// planes bit for bit; only the tile -> CTA mapping differs.
#pragma once

#include <cstdint>

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

// The K4 configuration of a plan with the emulated GEMM (DESIGN §4): a deterministic
// function of the GEMM view (m, n_hat, k_hat, b), N, g and the device's SM count. Returns
// IMD_K4_128x256 when the emulated GEMM does not run.
IMD_HOST_INTERNAL int32_t k4_select(const ExecPlan& e, int32_t sms);

// Plan-time autotuning (imd_options_t.k4_tune, DESIGN §6): the device bytes imd_plan_create
// allocates temporarily to time the candidates on group 1's K4 launches (RA, RB and group
// 1's W planes, one allocation), or 0 if no tuning runs for this plan (no emulated GEMM, or
// more than kK4TuneMaxBytes). Independent of the k4_tune option itself.
constexpr uint64_t kK4TuneMaxBytes = uint64_t{256} << 20;
IMD_HOST_INTERNAL uint64_t k4_tune_bytes(const ExecPlan& e);

}  // namespace imd::detail
