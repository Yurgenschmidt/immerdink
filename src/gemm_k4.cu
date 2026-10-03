// K4 (DESIGN §3.5, §4): per residue group and K-chunk, a batched int8 TN GEMM with the
// mod-p -> u8 epilogue (store-mod for chunk 0, add-mod after), batch z = s*b + lambda,
// modulus l = g0 + z / b from __constant__.
//
// A2: the kernel is imd::kernels::K4Gemm (kernels/k4_gemm.cuh): CUTLASS 2.x's multistage
// int8 main loop in a persistent kernel whose epilogue works in registers (no shared memory,
// no barriers) and overlaps the next tile's cp.async prologue. It replaced CUTLASS's
// GemmUniversal + Sm80 EVT epilogue (kernels/mod_epilogue.cuh, still tested by
// tests/test_cutlass_mod_epilogue.cu), which left K4 at 60-66 % of its bound at k = 1024
// (docs/NOTES-A2.md). Same arithmetic, bit-identical results.
//
// A3 (DESIGN v1.5 §4 "K4 configurations"): K4Gemm is instantiated for every entry of
// host/k4_config.hpp kK4Configs (src/k4/k4_cfg<i>.cu, run-time table k4/k4_ops.hpp). A plan
// runs one of them: libimmerdink_host's k4_select for the device's SM count, or, with the
// plan option k4_tune, the fastest one timed at plan creation (k4_tune below). All give the
// same W planes bit for bit; the launch structure (host/k4_launches.hpp) is the same for
// every configuration.
//
// DESIGN v1.2 §6: imd_plan_create builds the kernel parameters of every launch shape
// (k4_prepare: SM count, occupancy, smem attribute, Params); imd_contract copies them, sets
// its pointers and launches. No CUTLASS device-level host state is used.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>

#include <cuda_runtime.h>

#include "host/analysis.hpp"  // kK4*Bytes, k4_tune_bytes: libimmerdink_host's sizing
#include "k4/k4_ops.hpp"
#include "plan.hpp"
#include "profile.hpp"

namespace imd::detail {

const K4Ops& k4_ops_cfg0();
const K4Ops& k4_ops_cfg1();
const K4Ops& k4_ops_cfg2();
const K4Ops& k4_ops_cfg3();
static_assert(kK4ConfigCount == 4, "one k4_ops_cfg<i> per host/k4_config.hpp entry");

const K4Ops& k4_ops(int cfg) {
  switch (cfg) {
    case 1: return k4_ops_cfg1();
    case 2: return k4_ops_cfg2();
    case 3: return k4_ops_cfg3();
    default: return k4_ops_cfg0();
  }
}

namespace {

LaunchLimits g_limits = production_launch_limits();

// One K4 launch of a call (host/k4_launches.hpp K4Launch). Its kernel parameters depend
// only on the shape key (kc, zc, nc; store-mod for chunk 0, add-mod after); the pointers and
// the epilogue's modulus offsets are set per call.
using Launch = K4Launch;
struct ShapeKey {
  int64_t kc, zc, nc;
  bool operator<(const ShapeKey& o) const {
    return kc != o.kc ? kc < o.kc : zc != o.zc ? zc < o.zc : nc < o.nc;
  }
};

}  // namespace

// Everything imd_contract needs to launch K4, fixed at plan creation (DESIGN v1.2 §6).
struct K4Plan {
  int64_t m = 0, n_hat = 0, k_hat = 0, b = 0;
  K4Grid grid;     // n_hat, k_hat, b and the launch limits at creation
  int32_t config = 0;  // IMD_K4_*: the K4Gemm instantiation the parameters belong to
  std::map<ShapeKey, K4Blob> store;  // chunk 0
  std::map<ShapeKey, K4Blob> add;    // chunks > 0

  // The launches of one group of gsize moduli, in order: K-chunk, batch part, n-hat part.
  template <class F>
  void for_each_launch(int gsize, F&& f) const {
    k4_for_each_launch(grid, gsize, f);
  }
};

// libimmerdink_host's plan_host_bytes counts the K4Plan and its map entries with these
// sizes (host/analysis.hpp); a kernel-parameter or K4Plan change must update them there.
// The entry size does not depend on the configuration (K4Blob).
static_assert(sizeof(K4Plan) == kK4PlanBytes, "update kK4PlanBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::store)::value_type) == kK4StoreEntryBytes,
              "update kK4StoreEntryBytes in src/host/analysis.hpp");
static_assert(sizeof(decltype(K4Plan::add)::value_type) == kK4AddEntryBytes,
              "update kK4AddEntryBytes in src/host/analysis.hpp");

namespace {

// This call's parameters for launch L of group g0: operand, plane and modulus offsets.
K4Call call_of(const K4Plan& k, const Launch& L, const int8_t* RA, const int8_t* RB, uint8_t* W,
               int g0) {
  const int64_t plane_a = k.m * k.k_hat, plane_b = k.n_hat * k.k_hat, plane_w = k.m * k.n_hat;
  return K4Call{RA + L.z0 * plane_a + L.k0,
                RB + L.z0 * plane_b + L.j0 * k.k_hat + L.k0,
                W + (int64_t{g0} * k.b + L.z0) * plane_w + L.j0,
                plane_a,
                plane_b,
                plane_w,
                g0,
                k.b,
                L.z0};
}

// The launches of one group (prof: the bench's phase marks).
bool run_group(const K4Plan& k, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
               int gsize, cudaStream_t stream, bool prof) {
  const K4Ops& ops = k4_ops(k.config);
  bool ok = true;
  k.for_each_launch(gsize, [&](const Launch& L) {
    if (!ok) return;
    const ShapeKey key{L.kc, L.zc, L.nc};
    const auto& map = L.k0 == 0 ? k.store : k.add;
    const auto it = map.find(key);
    ok = it != map.end() && ops.launch(it->second, L.k0 != 0, call_of(k, L, RA, RB, W, g0), stream);
    if (prof) IMD_PROF(stream, L.k0 == 0 ? "K4 gemm store" : "K4 gemm add");
  });
  return ok;
}

}  // namespace

LaunchLimits& launch_limits() { return g_limits; }
LaunchLimits default_launch_limits() { return production_launch_limits(); }

int& k4_config_override() {
  static int v = -1;  // only the test / benchmark hooks change it
  return v;
}

imd_status_t k4_prepare(const ExecPlan& p, int cfg, std::shared_ptr<const K4Plan>* out) {
  if (cfg < 0 || cfg >= kK4ConfigCount) return IMD_STATUS_INVALID_VALUE;
  auto k = std::make_shared<K4Plan>();
  k->m = p.m, k->n_hat = p.n_hat, k->k_hat = p.k_hat, k->b = p.b;
  // The launch structure is the configuration's-independent host one (n-hat split in
  // units of kK4TileN = 256 columns, a multiple of every tile width).
  k->grid = k4_grid(p.n_hat, p.k_hat, p.b, g_limits);
  k->config = cfg;
  const K4Ops& ops = k4_ops(cfg);
  int occ_store = 0, occ_add = 0;
  if (ops.setup(&occ_store, &occ_add) != cudaSuccess) {
    cudaGetLastError();  // clear the (non-sticky) error state
    return IMD_STATUS_CUDA_ERROR;
  }
  if (occ_store < 1 || occ_add < 1) return IMD_STATUS_NOT_SUPPORTED;
  // GEMM sizes are ints (DESIGN §4 launch limits: m, n_hat <= 2^31 - 1, kc <= K_c'); k_hat
  // and n_hat are multiples of 16, so every 128-bit access is aligned.
  bool ok = k->m >= 1 && k->m <= INT32_MAX && k->n_hat % 16 == 0 && k->k_hat % 16 == 0;
  for (int g0 = 0; ok && g0 < p.N; g0 += p.g) {
    k->for_each_launch(std::min(p.g, p.N - g0), [&](const Launch& L) {
      const ShapeKey key{L.kc, L.zc, L.nc};
      auto& map = L.k0 == 0 ? k->store : k->add;
      if (map.count(key)) return;
      ok = ok && L.nc <= INT32_MAX && L.kc <= INT32_MAX && L.nc % 16 == 0;
      const int64_t resident = int64_t{p.sms} * (L.k0 == 0 ? occ_store : occ_add);
      ops.make(&map[key], static_cast<int>(k->m), static_cast<int>(L.nc), static_cast<int>(L.kc),
               k->k_hat, k->k_hat, k->n_hat, L.zc, resident);
    });
  }
  if (!ok) return IMD_STATUS_NOT_SUPPORTED;
  *out = std::move(k);
  return IMD_STATUS_SUCCESS;
}

imd_status_t k4_tune(const ExecPlan& p, int32_t* cfg, K4TuneReport* report) {
  const uint64_t bytes = k4_tune_bytes(p);  // libimmerdink_host: what imd_lowering reports
  if (report != nullptr) *report = K4TuneReport{};
  if (bytes == 0) return IMD_STATUS_SUCCESS;  // no tuning for this plan: keep the rule's
  void* buf = nullptr;
  if (cudaMalloc(&buf, static_cast<size_t>(bytes)) != cudaSuccess) {
    cudaGetLastError();
    return IMD_STATUS_ALLOC_FAILED;
  }
  const int gsize = std::min(p.g, p.N);
  const auto align = [](uint64_t x) { return (x + 255) / 256 * 256; };
  const uint64_t gb = static_cast<uint64_t>(gsize) * static_cast<uint64_t>(p.b);
  const uint64_t ra = align(gb * p.m * p.k_hat), rb = align(gb * p.n_hat * p.k_hat);
  auto* RA = static_cast<int8_t*>(buf);
  auto* RB = RA + ra;
  auto* W = reinterpret_cast<uint8_t*>(RB + rb);
  cudaStream_t s = cudaStreamPerThread;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  imd_status_t st = IMD_STATUS_SUCCESS;
  if (cudaMemsetAsync(buf, 0, static_cast<size_t>(bytes), s) != cudaSuccess ||
      cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess)
    st = IMD_STATUS_CUDA_ERROR;
  // Candidates, the rule's choice first (another one replaces it only if strictly faster).
  struct Cand {
    int cfg;
    std::shared_ptr<const K4Plan> plan;
    int reps;
    float best_ms;
  };
  Cand cands[kK4ConfigCount];
  int nc = 0;
  for (int i = 0; i < kK4ConfigCount && st == IMD_STATUS_SUCCESS; ++i) {
    const int c = i == 0 ? *cfg : (i <= *cfg ? i - 1 : i);
    std::shared_ptr<const K4Plan> kp;
    const imd_status_t ps = k4_prepare(p, c, &kp);
    if (ps == IMD_STATUS_NOT_SUPPORTED) continue;  // not runnable here: not a candidate
    if (ps != IMD_STATUS_SUCCESS) st = ps;
    else cands[nc++] = Cand{c, std::move(kp), 1, 0.0f};
  }
  // Time `reps` back-to-back runs of group 1's launches; ms per run.
  const auto timed = [&](const Cand& c, float* ms) {
    bool ok = cudaEventRecord(e0, s) == cudaSuccess;
    for (int r = 0; ok && r < c.reps; ++r) ok = run_group(*c.plan, RA, RB, W, 0, gsize, s, false);
    ok = ok && cudaEventRecord(e1, s) == cudaSuccess && cudaEventSynchronize(e1) == cudaSuccess &&
         cudaEventElapsedTime(ms, e0, e1) == cudaSuccess;
    *ms /= static_cast<float>(c.reps);
    return ok;
  };
  // One warm-up run each (instruction cache, clocks), which also sizes the batches: about
  // kTuneBatchMs of back-to-back runs per timing (at most kTuneMaxReps), so launch jitter
  // and the event resolution stay small against short kernels. Then kTuneTrials rounds,
  // interleaved over the candidates (a clock ramp affects them alike); the minimum counts.
  constexpr float kTuneBatchMs = 0.1f;
  constexpr int kTuneMaxReps = 32, kTuneTrials = 3;
  for (int i = 0; i < nc && st == IMD_STATUS_SUCCESS; ++i) {
    float ms = 0;
    if (!timed(cands[i], &ms)) st = IMD_STATUS_CUDA_ERROR;
    const float want = ms > 0 ? kTuneBatchMs / ms : float(kTuneMaxReps);
    cands[i].reps = want >= kTuneMaxReps ? kTuneMaxReps : std::max(1, static_cast<int>(want + 0.999f));
  }
  for (int t = 0; t < kTuneTrials && st == IMD_STATUS_SUCCESS; ++t)
    for (int i = 0; i < nc && st == IMD_STATUS_SUCCESS; ++i) {
      float ms = 0;
      if (!timed(cands[i], &ms)) st = IMD_STATUS_CUDA_ERROR;
      if (t == 0 || ms < cands[i].best_ms) cands[i].best_ms = ms;
    }
  int best = *cfg;
  float best_ms = 0;
  for (int i = 0; i < nc && st == IMD_STATUS_SUCCESS; ++i) {
    if (report != nullptr) report->ms[cands[i].cfg] = cands[i].best_ms;
    if (i == 0 || cands[i].best_ms < best_ms) best_ms = cands[i].best_ms, best = cands[i].cfg;
  }
  if (e0 != nullptr) cudaEventDestroy(e0);
  if (e1 != nullptr) cudaEventDestroy(e1);
  if (cudaStreamSynchronize(s) != cudaSuccess) st = IMD_STATUS_CUDA_ERROR;
  cudaFree(buf);
  if (st != IMD_STATUS_SUCCESS) {
    cudaGetLastError();
    return st;
  }
  if (report != nullptr) report->tuned = nc > 0;
  *cfg = best;
  return IMD_STATUS_SUCCESS;
}

K4Footprint k4_footprint(const K4Plan& k, int N, int g) {
  K4Footprint f;
  f.store_shapes = static_cast<int64_t>(k.store.size());
  f.add_shapes = static_cast<int64_t>(k.add.size());
  for (int g0 = 0; g0 < N; g0 += g) k.for_each_launch(std::min(g, N - g0), [&](const Launch&) { ++f.launches; });
  f.bytes = sizeof(K4Plan) + k.store.size() * sizeof(decltype(k.store)::value_type) +
            k.add.size() * sizeof(decltype(k.add)::value_type);
  f.config = k.config;
  return f;
}

imd_status_t k4_group(const K4Plan& k, const int8_t* RA, const int8_t* RB, uint8_t* W, int g0,
                      int gsize, cudaStream_t stream) {
  return run_group(k, RA, RB, W, g0, gsize, stream, true) ? IMD_STATUS_SUCCESS
                                                            : IMD_STATUS_CUDA_ERROR;
}

}  // namespace imd::detail
