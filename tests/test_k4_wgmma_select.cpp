// Issue 22: the Hopper K4 backend's host-side rules (src/host/k4_wgmma.hpp), CUDA-free: the device-class table, the backend resolution,
// the legality of a launch (TMA / epilogue / CUTLASS requirements), the profitability rule (docs/NOTES-22.md "Profitability rule": the R3
// floor sweep of 19 shapes on H100 and H200 as expectations, margins and mutants of the constants) and the tile / scheduler choice with the
// R2 table (docs/NOTES-22.md "Library integration (design)") including edge shapes. Runs everywhere (no GPU, no CUTLASS).
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "host/k4_wgmma.hpp"

using namespace imd::detail;

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                         \
  do {                                                                                   \
    ++g_checks;                                                                          \
    if (!(c)) {                                                                          \
      ++g_fail;                                                                          \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                           \
    }                                                                                    \
  } while (0)

// A launch of the library as gemm_k4.cu forms it for an unpacked call: A[z][m][k_hat], B[z][n_hat][k_hat], W[z][m][n_hat], 256-byte bases.
static K4WgmmaLaunch lib_launch(int64_t m, int64_t n_hat, int64_t kc, int64_t zc, int64_t k_hat = 0, int sms = 132) {
  K4WgmmaLaunch L{};
  L.m = m, L.nc = n_hat, L.kc = kc, L.zc = zc;
  L.k_hat = k_hat ? k_hat : kc;
  L.n_hat = n_hat;
  L.batch_a = m * L.k_hat, L.batch_b = n_hat * L.k_hat, L.plane_w = m * n_hat;
  L.a = 0x7f0000000000ull, L.b = 0x7f0100000000ull, L.w = 0x7f0200000000ull;
  L.sms = sms;
  return L;
}

int main() {
  // ---- the device-class table: the sm_90 row only, compute capability 9.0 only
  static_assert(IMD_TABLE_COUNT == 2, "a new table set needs a row in host/k4_wgmma.hpp kK4WgmmaTable and a test here");
  CHECK(!kK4WgmmaTable[IMD_TABLE_SM89] && kK4WgmmaTable[IMD_TABLE_SM90]);
  CHECK(k4_wgmma_device_ok(IMD_TABLE_SM90, 9, 0));
  CHECK(!k4_wgmma_device_ok(IMD_TABLE_SM90, 9, 1) && !k4_wgmma_device_ok(IMD_TABLE_SM90, 8, 9) && !k4_wgmma_device_ok(IMD_TABLE_SM90, 10, 0));
  CHECK(!k4_wgmma_device_ok(IMD_TABLE_SM89, 8, 9) && !k4_wgmma_device_ok(IMD_TABLE_SM89, 12, 0) && !k4_wgmma_device_ok(IMD_TABLE_SM89, 9, 0));
  CHECK(!k4_wgmma_device_ok(-1, 9, 0) && !k4_wgmma_device_ok(IMD_TABLE_COUNT, 9, 0));
  // the real registry: the H100 row selects the sm_90 table, the RTX 4090 and an sm_120 card the sm_89 one
  {
    const struct { int major, minor; const char* name; bool wgmma; } dev[] = {
        {9, 0, "NVIDIA H100 80GB HBM3", true}, {9, 0, "NVIDIA H200", true}, {9, 0, "NVIDIA H100 PCIe", true},
        {8, 9, "NVIDIA GeForce RTX 4090", false}, {12, 0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", false}, {8, 0, "NVIDIA A100", false}};
    for (const auto& d : dev) {
      imd_device_class_t c{};
      c.struct_size = sizeof c, c.cc_major = d.major, c.cc_minor = d.minor, c.sm_count = 132;
      std::strncpy(c.name, d.name, sizeof c.name - 1);
      imd_device_selection_t sel{};
      sel.struct_size = sizeof sel;
      CHECK(imd_device_select(&c, &sel) == IMD_STATUS_SUCCESS);
      CHECK(sel.table_id < 0 || k4_wgmma_device_ok(sel.table_id, d.major, d.minor) == d.wgmma);
    }
  }
  // ---- backend resolution
  CHECK(k4_backend_resolve(kK4BackendAuto, true, true) == kK4BackendWgmma);
  CHECK(k4_backend_resolve(kK4BackendAuto, true, false) == kK4BackendMma);
  CHECK(k4_backend_resolve(kK4BackendMma, true, true) == kK4BackendMma);
  CHECK(k4_backend_resolve(kK4BackendWgmma, true, false) == kK4BackendWgmmaForced);  // forced where the device has it: a PLAN property (floor lifted)
  CHECK(k4_backend_resolve(kK4BackendWgmma, true, true) == kK4BackendWgmmaForced);
  CHECK(kK4BackendMma == 0 && kK4BackendWgmma == 1 && kK4BackendWgmmaForced == 2);  // the values the hooks and records report
  for (int o = -1; o <= 1; ++o)
    for (int c = 0; c < 2; ++c) CHECK(k4_backend_resolve(o, false, c != 0) == kK4BackendMma);  // unavailable: never wgmma
  CHECK(k4_backend_valid(-1) && k4_backend_valid(0) && k4_backend_valid(1) && !k4_backend_valid(2) && !k4_backend_valid(-2));

  // ---- the R2 table (batch 14, 4, 1)
  for (int64_t zc : {14, 4, 1}) {
    for (bool force : {false, true}) {
      K4WgmmaPick p = k4_wgmma_pick(lib_launch(4096, 4096, 4096, zc), force);
      CHECK(p.variant == 0 && p.swizzle == 1 && p.raster == kRasterHeuristic);  // 4096^3: the default scheduler (R2: 1360 / 1357 TOPS-eq)
      p = k4_wgmma_pick(lib_launch(8192, 8192, 8192, zc), force);
      CHECK(p.variant == 0 && p.swizzle == 8 && p.raster == kRasterAlongN);     // 8192^3: swizzle 8 AlongN (R2: 1408 / 1328)
    }
  }
  // tiles per batch 1024 is the swizzle threshold: 4096 x 8192 (512 x ... ) has 32 x 32 tiles = 1024
  CHECK(k4_wgmma_pick(lib_launch(4096, 8192, 4096, 14), false).swizzle == 8);
  CHECK(k4_wgmma_pick(lib_launch(4096, 7936, 4096, 14), false).swizzle == 1);   // 32 x 31 = 992 tiles
  // ---- profitability rule (auto) against force: spot checks (the full R3 table is below)
  CHECK(k4_wgmma_pick(lib_launch(512, 512, 512, 14), false).variant < 0);       // small: mma.sync (R3: 1.03, inside the margin)
  CHECK(k4_wgmma_pick(lib_launch(512, 512, 512, 14), true).variant == 0);       // forced: served
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 512, 14), false).variant == 0);    // short K: R3 measured +22 % (the old floor declined it)
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 1024, 14), false).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(1024, 1024, 4096, 1), false).variant < 0);     // 32 tiles of 132 SMs: below the fill prior
  CHECK(k4_wgmma_pick(lib_launch(1024, 1024, 4096, 14), false).variant == 0);   // 448 tiles
  CHECK(k4_wgmma_pick(lib_launch(2048, 2048, 2048, 14), false).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(8192, 8192, 8192, 1), false).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(16, 16, 16, 1), false).variant < 0);           // tiny: never by default
  CHECK(k4_wgmma_pick(lib_launch(1, 16, 16, 1), false).variant < 0);
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 16, 14), false).variant < 0);      // one k step: outside the measured envelope (R3 smallest K chunk 256)
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 128, 14), false).variant < 0 && k4_wgmma_pick(lib_launch(4096, 4096, 256, 14), false).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(256, 4096, 4096, 14), false).variant < 0 && k4_wgmma_pick(lib_launch(4096, 256, 4096, 14), false).variant < 0);  // m, n-part < 512: unmeasured
  // the device-class row: the sm_89 row has no rule (and no backend): nothing by default, force still serves a legal launch (the host gate
  // never reaches this on such a device; this is the table's own behaviour)
  CHECK(!kK4WgmmaProfit[IMD_TABLE_SM89].enabled && kK4WgmmaProfit[IMD_TABLE_SM90].enabled);
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 4096, 14), false, IMD_TABLE_SM89).variant < 0);
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 4096, 14), true, IMD_TABLE_SM89).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 4096, 14), false, -1).variant < 0 && k4_wgmma_pick(lib_launch(4096, 4096, 4096, 14), false, IMD_TABLE_COUNT).variant < 0);
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 4096, 14), false, IMD_TABLE_SM90).variant == 0);

  // ---- the R3 floor sweep (docs/NOTES-22.md "Profitability rule"): 19 shapes, N = 14, whole-call duel forced wgmma vs mma.sync on H100 / H200
  // (ratio > 1: wgmma faster). g = moduli per K4 launch: 14 = one launch, 4 = launches of 4, 4, 4, 2, 1 = fourteen launches of one.
  struct R3Row { const char* name; int64_t m, n, k; int g; double h100, h200; };
  static const R3Row r3[] = {
      {"r1024x1024x4096", 1024, 1024, 4096, 14, 1.198, 1.209}, {"r1024x4096x4096", 1024, 4096, 4096, 14, 1.268, 1.259},
      {"r1100x3000x2500", 1100, 3000, 2500, 14, 1.189, 1.198}, {"r2048x2048x256", 2048, 2048, 256, 14, 1.115, 1.112},
      {"r4096x1024x4096", 4096, 1024, 4096, 14, 1.255, 1.266}, {"r4096x4096x1024", 4096, 4096, 1024, 14, 1.272, 1.272},
      {"r4096x4096x512", 4096, 4096, 512, 14, 1.222, 1.222},   {"r4096x512x4096", 4096, 512, 4096, 14, 1.185, 1.185},
      {"r512x4096x4096", 512, 4096, 4096, 14, 1.199, 1.202},   {"sq1024", 1024, 1024, 1024, 14, 1.149, 1.134},
      {"sq1024g1", 1024, 1024, 1024, 1, 0.928, 0.930},         {"sq1536", 1536, 1536, 1536, 14, 1.192, 1.186},
      {"sq1536g1", 1536, 1536, 1536, 1, 1.060, 1.047},         {"sq2048", 2048, 2048, 2048, 14, 1.254, 1.258},
      {"sq2048g1", 2048, 2048, 2048, 1, 1.059, 1.068},         {"sq2048g4", 2048, 2048, 2048, 4, 1.162, 1.169},
      {"sq3072", 3072, 3072, 3072, 14, 1.312, 1.317},          {"sq512", 512, 512, 512, 14, 1.030, 1.031},
      {"sq768", 768, 768, 768, 14, 1.053, 1.061}};
  // the rule applied to a whole call: served iff every one of its K4 launches is served (as the library decides them, one by one)
  const auto call_served = [&](const R3Row& r, const K4WgmmaProfit& P, bool use_pick) {
    bool all = true;
    for (int z = 14; z > 0; z -= r.g) {
      const K4WgmmaLaunch L = lib_launch(r.m, (r.n + 15) / 16 * 16, (r.k + 15) / 16 * 16, z < r.g ? z : r.g);  // n-hat, k-hat as the plan pads them
      const bool s = use_pick ? k4_wgmma_pick(L, false).variant >= 0 : k4_wgmma_estimate(L, P).serve;
      all = all && s;
    }
    return all;
  };
  int n_served = 0, n_declined = 0;
  for (const R3Row& r : r3) {
    const double worst = std::fmin(r.h100, r.h200);
    const bool served = call_served(r, kK4WgmmaProfit[IMD_TABLE_SM90], true);
    served ? ++n_served : ++n_declined;
    if (served != (worst >= 1.045 && std::strcmp(r.name, "sq512") != 0 && std::strcmp(r.name, "sq1024g1") != 0)) std::printf("R3 shape %s: served %d\n", r.name, served);
    CHECK(served == call_served(r, kK4WgmmaProfit[IMD_TABLE_SM90], false));      // the pick and the estimate are one rule
    if (served) CHECK(worst >= 1.04);                                          // SAFETY: nothing served that measured slower or within noise (1.02), with a margin
    if (worst >= 1.045) CHECK(served);                                         // every gain of 4.5 % or more on both cards is served
    if (worst < 1.04) CHECK(!served);                                          // sq512 (1.030) and sq1024 g=1 (0.928) are declined
  }
  CHECK(n_served == 17 && n_declined == 2);
  {  // the one loser: 1024^3 with one modulus per launch, forced 0.928 / 0.930: declined, and the model predicts a loss (not merely a small gain)
    const K4WgmmaLaunch L = lib_launch(1024, 1024, 1024, 1);
    const K4WgmmaEstimate e = k4_wgmma_estimate(L, kK4WgmmaProfit[IMD_TABLE_SM90]);
    CHECK(!e.serve && !e.fill_ok && e.delta_us > 0);
    CHECK(k4_wgmma_pick(L, false).variant < 0 && std::strcmp(k4_wgmma_pick(L, false).why, "below the profitability rule (fill)") == 0);
    CHECK(k4_wgmma_pick(L, true).variant == 0);                                // forced is still served (that is what the duel measured)
    K4WgmmaProfit nofill = kK4WgmmaProfit[IMD_TABLE_SM90];
    nofill.min_fill = 0;                                                       // without the fill prior the fitted model alone still declines it
    CHECK(!k4_wgmma_estimate(L, nofill).serve && k4_wgmma_estimate(L, nofill).delta_us > 0);
  }
  {  // the rule's own numbers, restated (the fit of tools/fit_k4_wgmma_profit.py): 4096^3 N = 14: mma.sync K4 estimated at 1.96 ms
    const K4WgmmaEstimate e = k4_wgmma_estimate(lib_launch(4096, 4096, 4096, 14), kK4WgmmaProfit[IMD_TABLE_SM90]);
    const double w = 32.0 * 16 * 14 / 132, kt = 32;
    CHECK(std::fabs(e.delta_us - (2.40 + w * (-0.57 - 0.51 * kt))) < 1e-9 && std::fabs(e.mma_us - 1.13 * w * kt) < 1e-9);
    CHECK(e.serve && e.mma_us > 1900 && e.mma_us < 2000 && e.delta_us < -850 && e.delta_us > -1000);  // R3: K4 phase 1.99 -> 1.22 ms (965 -> 1580 TOPS): 0.77 ms saved, the model says 0.92 (optimistic by 20 % on this shape; far from the boundary)
  }
  // MUTANTS of the constants: each must change at least one outcome of the table (a test that cannot see a changed constant is not a test)
  {
    const auto outcomes = [&](const K4WgmmaProfit& P) {
      unsigned bits = 0;
      for (int i = 0; i < 19; ++i) bits |= (call_served(r3[i], P, false) ? 1u : 0u) << i;
      return bits;
    };
    const K4WgmmaProfit base = kK4WgmmaProfit[IMD_TABLE_SM90];
    const unsigned want = outcomes(base);
    CHECK(__builtin_popcount(want) == 17);
    const auto mutate = [&](auto&& f) { K4WgmmaProfit P = base; f(P); return outcomes(P) != want; };
    CHECK(mutate([](K4WgmmaProfit& P) { P.launch_us = 0.0; P.min_fill = 0.0; }));        // no launch cost and no fill prior: serves the loser
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_fill = 0.0; P.launch_us = 0.4; }));        // the leave-g1-out fit's neighbourhood alone is not enough either
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_fill = 1.0; }));                           // a full wave: loses the g = 1 winners
    CHECK(mutate([](K4WgmmaProfit& P) { P.launch_us = 6.0; }));                          // dearer launches: declines small winners
    CHECK(mutate([](K4WgmmaProfit& P) { P.wave_k_us = -0.2; }));                         // a weaker mainloop gain
    CHECK(mutate([](K4WgmmaProfit& P) { P.wave_us = 0.0; P.launch_us = 4.0; }));         // no tile-fixed gain
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_gain_us = 1.5; }));                        // the absolute margin is live (sq1536 g=1 needs 1.25)
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_gain_rel = 0.20; }));                      // the relative margin is live (sq1536 g=1 and sq768)
    CHECK(mutate([](K4WgmmaProfit& P) { P.enabled = false; }));
    CHECK(!mutate([](K4WgmmaProfit& P) { P.min_mn = 0; P.min_kc = 0; }));                // the envelope does not touch the 19 measured shapes (it only keeps the model out of the unmeasured)
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_mn = 1024; }));                            // ... and 512 is exactly the smallest measured extent that is served
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_kc = 512; }));
    CHECK(mutate([](K4WgmmaProfit& P) { P.wave_k_us = +0.51; }));                        // sign flipped
    CHECK(mutate([](K4WgmmaProfit& P) { P.min_gain_us = -5.0; P.min_gain_rel = 0.0; P.min_fill = 0.0; P.launch_us = 2.0; }));  // margins removed: serves sq512 (+3 %) or the loser
    // and the neighbourhood is a plateau: +-10 % on any constant and +-20 % on the margins keeps the 19 outcomes (the rule is not balanced on a knife edge)
    for (int which = 0; which < 6; ++which)
      for (double f : {0.9, 1.1}) {
        K4WgmmaProfit P = base;
        double* c[] = {&P.launch_us, &P.wave_us, &P.wave_k_us, &P.mma_wave_k_us, &P.min_gain_us, &P.min_gain_rel};
        *c[which] *= f;
        if (outcomes(P) != want) std::printf("constant %d x %.1f changes the R3 outcomes\n", which, f);
        CHECK(outcomes(P) == want);
      }
  }
  // MONOTONE: more K or more batches never turn a served launch into a declined one (the fitted b and a are negative and dominate the
  // relative margin), and a bigger device (more SMs, same tiles) is never "more profitable"
  for (int64_t zc : {1, 2, 4, 14, 56})
    for (int64_t m : {128, 640, 1024, 2048, 4096})
      for (int64_t kc : {128, 256, 512, 1024, 4096}) {
        const bool s = k4_wgmma_pick(lib_launch(m, 2048, kc, zc), false).variant >= 0;
        if (s) {
          CHECK(k4_wgmma_pick(lib_launch(m, 2048, kc * 2, zc), false).variant >= 0);
          CHECK(k4_wgmma_pick(lib_launch(m, 2048, kc, zc * 2), false).variant >= 0);
          CHECK(k4_wgmma_pick(lib_launch(m * 2, 2048, kc, zc), false).variant >= 0);
        }
      }
  // sms: the rule reads the plan's SM count (an H100 PCIe has 114); unknown (0) means 132
  CHECK(k4_wgmma_estimate(lib_launch(2048, 2048, 2048, 1, 0, 0), kK4WgmmaProfit[IMD_TABLE_SM90]).delta_us == k4_wgmma_estimate(lib_launch(2048, 2048, 2048, 1, 0, 132), kK4WgmmaProfit[IMD_TABLE_SM90]).delta_us);
  // ---- edge shapes: legal and forced
  CHECK(k4_wgmma_pick(lib_launch(1, 16, 16, 1), true).variant == 0);            // one row, one 16-wide chunk: TMA zero-fills the tile
  CHECK(k4_wgmma_pick(lib_launch(129, 272, 1040, 3, 1040), true).variant == 0); // m, n not multiples of the tile, k not a multiple of 128
  CHECK(k4_wgmma_pick(lib_launch(4097, 4112, 4112, 14), true).variant == 0);
  {  // the library chunk K_c' = 131008 inside a longer row (odd chunk starts at phase 64 mod 128): legal, a 16-byte aligned base
    K4WgmmaLaunch L = lib_launch(4096, 4096, 131008, 14, 262016);
    L.a += 131008, L.b += 131008;
    CHECK(k4_wgmma_pick(L, false).variant == 0);
  }
  // ---- illegal launches are never served, forced or not
  const auto never = [&](K4WgmmaLaunch L, const char* what) {
    for (bool force : {false, true}) {
      const K4WgmmaPick p = k4_wgmma_pick(L, force);
      if (p.variant >= 0) std::printf("served an illegal launch: %s\n", what);
      CHECK(p.variant < 0 && p.why[0] != '\0');
    }
    CHECK(k4_wgmma_illegal(L) != nullptr);
  };
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.a += 8; return L; }(), "A base 8 mod 16");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.b += 1; return L; }(), "B base 1 mod 16");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.w += 8; return L; }(), "W base 8 mod 16");
  never(lib_launch(4096, 4096, 4104, 14), "kc 8 mod 16");
  never(lib_launch(4096, 4090, 4096, 14), "nc not a multiple of 16");
  never(lib_launch(4096, 4096, 4096, 65536), "zc beyond gridDim.z");
  never(lib_launch(4096, 4096, 4096, 0), "no batches");
  never(lib_launch(0, 4096, 4096, 1), "no rows");
  never(lib_launch(int64_t{1} << 31, 4096, 4096, 1), "m beyond int32");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.k_hat = 4088; return L; }(), "k_hat 8 mod 16 and shorter than kc");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.batch_a -= 16; return L; }(), "overlapping A planes");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.batch_b = 4096ll * 4096 - 16; return L; }(), "overlapping B planes (a packed operand)");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.batch_a += 8; return L; }(), "A batch stride 8 mod 16");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.k_hat = int64_t{1} << 40; L.batch_a = L.m * L.k_hat; L.batch_b = L.n_hat * L.k_hat; return L; }(), "TMA stride limit");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.plane_w = 4096ll * 4096 - 16; return L; }(), "W planes overlap");
  never([] { auto L = lib_launch(4096, 4096, 4096, 14); L.n_hat = 4088; return L; }(), "n_hat 8 mod 16");
  {  // more tiles than int32: 2^20 tile rows x 2^11 tile columns
    K4WgmmaLaunch L = lib_launch(int64_t{1} << 27, 524288, 4096, 65535);
    never(L, "tile count beyond int32");
  }

  // ---- property: whatever a random launch is, a served one satisfies every requirement the encoders have (a re-statement, independent of
  // the code under test: 16-byte alignment of the three bases, strides multiples of 16 below 2^40, extents within int32)
  std::mt19937_64 rng(22);
  int served = 0, declined = 0;
  for (int it = 0; it < 200000; ++it) {
    K4WgmmaLaunch L{};
    const int64_t kinds[] = {1, 16, 48, 64, 128, 130, 256, 1000, 1024, 4096, 8192, 131008};
    L.m = rng() % 4 == 0 ? int64_t(rng() % 20000) : kinds[rng() % 12];
    L.nc = (rng() % 4 == 0 ? int64_t(rng() % 20000) : kinds[rng() % 12]) / 16 * 16 + (rng() % 8 == 0 ? 8 : 0);
    L.kc = (rng() % 4 == 0 ? int64_t(rng() % 40000) : kinds[rng() % 12]) / 16 * 16 + (rng() % 8 == 0 ? 4 : 0);
    L.zc = rng() % 6 == 0 ? int64_t(rng() % 70000) : int64_t(1 + rng() % 28);
    L.k_hat = L.kc + (rng() % 3 == 0 ? 16 * int64_t(rng() % 5) : 0) + (rng() % 16 == 0 ? 8 : 0);
    L.n_hat = L.nc + (rng() % 3 == 0 ? 16 * int64_t(rng() % 5) : 0);
    L.batch_a = L.m * L.k_hat + (rng() % 8 == 0 ? int64_t(rng() % 64) - 32 : 0);
    L.batch_b = L.n_hat * L.k_hat + (rng() % 8 == 0 ? int64_t(rng() % 64) - 32 : 0);
    L.plane_w = L.m * L.n_hat + (rng() % 8 == 0 ? int64_t(rng() % 48) - 16 : 0);
    L.a = 0x7f0000000000ull + 16 * (rng() % 100) + (rng() % 8 == 0 ? rng() % 16 : 0);
    L.b = 0x7f0100000000ull + 16 * (rng() % 100) + (rng() % 8 == 0 ? rng() % 16 : 0);
    L.w = 0x7f0200000000ull + 16 * (rng() % 100) + (rng() % 8 == 0 ? rng() % 16 : 0);
    L.sms = 114 + int(rng() % 40);
    const bool force = rng() % 2 != 0;
    const K4WgmmaPick p = k4_wgmma_pick(L, force);
    if (p.variant < 0) {
      ++declined;
      continue;
    }
    ++served;
    CHECK(L.a % 16 == 0 && L.b % 16 == 0 && L.w % 16 == 0);
    CHECK(L.k_hat % 16 == 0 && L.batch_a % 16 == 0 && L.batch_b % 16 == 0 && L.n_hat % 16 == 0 && L.plane_w % 16 == 0);
    CHECK(L.k_hat < (int64_t{1} << 40) && L.batch_a < (int64_t{1} << 40) && L.batch_b < (int64_t{1} << 40));
    CHECK(L.m >= 1 && L.nc >= 1 && L.kc >= 1 && L.zc >= 1 && L.zc <= 65535 && L.m <= INT32_MAX && L.nc <= INT32_MAX && L.kc <= INT32_MAX);
    CHECK(L.nc % 16 == 0 && L.kc % 16 == 0 && L.kc <= L.k_hat);
    CHECK(L.batch_a >= L.m * L.k_hat && L.batch_b >= L.n_hat * L.k_hat && L.plane_w >= L.m * L.n_hat);
    CHECK(p.swizzle == 1 || p.swizzle == 8);
    CHECK(p.raster == kRasterHeuristic || p.raster == kRasterAlongN);
    if (!force) {  // the rule restated independently of the header (integers where it can): tiles, k steps, fill, time estimate
      const double tiles = double(((L.m + 127) / 128) * ((L.nc + 255) / 256) * L.zc), kt = double((L.kc + 127) / 128), w = tiles / L.sms;
      const double delta = 2.40 + w * (-0.57 - 0.51 * kt), mma = 1.13 * w * kt;
      CHECK(tiles >= 0.5 * L.sms && -delta >= 0.75 && -delta >= 0.10 * mma - 1e-9);
    }
    // determinism and monotonicity in `force`: forcing never serves less
    CHECK(k4_wgmma_pick(L, true).variant >= 0);
    const K4WgmmaPick q = k4_wgmma_pick(L, force);
    CHECK(q.variant == p.variant && q.swizzle == p.swizzle && q.raster == p.raster);
  }
  CHECK(served > 1000 && declined > 1000);  // the generator reaches both sides
  std::printf("k4_wgmma_select: %d checks, %d failed (%d launches served, %d declined in the random sweep)\n", g_checks, g_fail, served, declined);
  return g_fail == 0 ? 0 : 1;
}
