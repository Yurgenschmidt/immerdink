// Issue 22: the Hopper K4 backend's host-side rules (src/host/k4_wgmma.hpp), CUDA-free: the device-class table, the backend resolution,
// the legality of a launch (TMA / epilogue / CUTLASS requirements), the profitability floor and the tile / scheduler choice with the R2
// table (docs/NOTES-22.md "Library integration (design)") including edge shapes. Runs everywhere (no GPU, no CUTLASS).
#include <cstdint>
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
  CHECK(k4_backend_resolve(kK4BackendWgmma, true, false) == kK4BackendWgmma);  // forced where the device has it
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
  // ---- profitability floor (auto) against force
  CHECK(k4_wgmma_pick(lib_launch(512, 512, 512, 14), false).variant < 0);       // small: mma.sync
  CHECK(k4_wgmma_pick(lib_launch(512, 512, 512, 14), true).variant == 0);       // forced: served
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 512, 14), false).variant < 0);     // short K
  CHECK(k4_wgmma_pick(lib_launch(4096, 4096, 1024, 14), false).variant == 0);
  CHECK(k4_wgmma_pick(lib_launch(1024, 1024, 4096, 1), false).variant < 0);     // 8 x 4 tiles = 32 < two waves of 132
  CHECK(k4_wgmma_pick(lib_launch(1024, 1024, 4096, 14), false).variant == 0);   // 448 tiles
  CHECK(k4_wgmma_pick(lib_launch(2048, 2048, 2048, 14), false).variant == 0);
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
    if (!force) CHECK(L.m >= 1024 && L.nc >= 1024 && L.kc >= 1024);
    // determinism and monotonicity in `force`: forcing never serves less
    CHECK(k4_wgmma_pick(L, true).variant >= 0);
    const K4WgmmaPick q = k4_wgmma_pick(L, force);
    CHECK(q.variant == p.variant && q.swizzle == p.swizzle && q.raster == p.raster);
  }
  CHECK(served > 1000 && declined > 1000);  // the generator reaches both sides
  std::printf("k4_wgmma_select: %d checks, %d failed (%d launches served, %d declined in the random sweep)\n", g_checks, g_fail, served, declined);
  return g_fail == 0 ? 0 : 1;
}
