// Rung 1 of the Hopper K4 ladder (issue 22, docs/NOTES-22.md): the TIMING harness for our own wgmma K4 variants.
// Rung 0 (rung0.cu) checks exactness on a tiny problem; this program measures speed at the scale K4 runs: m = n = k in {4096, 8192},
// L = 14 independent GEMMs in one launch (the N moduli of the emulation; batch z uses modulus slot0 + z, b = 1), int8 residue operands
// A[z][m][k], B[z][n][k] (k contiguous), K4's modular epilogue (K4ModEpilogue: mod p -> u8 plane z of W), store mode, and, for reference,
// L = 4 and L = 1. Built for sm_90a only (tools/build_rung1.sh, one nvcc call, -O2 -DNDEBUG). On a device whose compute capability has no row
// in the table it does nothing and exits 3.
//
// Why: the stock CUTLASS wgmma ceilings (bench/ceiling) were built with CUTLASS's direct-store epilogue (256 scalar stores per thread) and
// default scheduler arguments, so they probably under-measure what a wgmma K4 can do. The theory (immerdink-eval/findings/offline-theory-
// 20261025.md section 3) says our register epilogue (HopperEpilogue, 16-byte stores) should reach 1260-1456 TOPS at 4096^3, L = 14, against
// 966 for today's mma.sync K4; it is falsified at <= 1100. This program measures that, and the two scheduler arguments the theory blames for
// the 8192^3 collapse of the stock kernels: max_swizzle_size (1, 2, 4, 8) and raster_order (AlongM / AlongN). The product's own mma.sync K4
// (k4_ops_cfg0, the A2 kernel, 128x256 tile) is a row of the same table ("m_..."), measured in the same process, so the baseline is same-session.
//
//   imd_rung1 [--quick] [--list] [--plan] [--variants A,B,..] [--shapes 4096,8192|MxNxK,..] [--batches 14,4,1] [--reps 12] [--warmup 3] [--loop 8]
//             [--max-seconds 8] [--samples 384] [--sweep 8192|none|SHAPES] [--sweep-variants LIST] [--clusters queried|heuristic|both]
//             [--sustain SECONDS] [--peak-seconds S] [--skip-last] [--no-nvml]
//   env: IMD_RUNG1_QUICK=1 (= --quick), BENCH_COMMIT, IMD_NVCC_VERSION, SLURM_JOB_ID (copied into the meta record / every record)
//   Unknown --keys are an error. --variants entries are the groups all / f / g / m / p / quick / sweep (exact words) or a name or substring of a
//   name (f_128x256 selects every f_ 128x256 row); default all. --sweep uses the selected rows flagged sweep unless --sweep-variants says otherwise.
//   --quick: 4096 only, L = 14 only, the rows flagged quick in the table, 10 reps, 128 samples, no sweep (about a minute).
//   The configurations (kEntries), the default shapes and batches and the sweep grid are data tables in one block ("DATA"); each row carries the
//   compute capability it is for, every JSON record carries the card's name and compute capability (gpu, cc).
//
// The p_ rows are NOT GEMMs: the SYNTHETIC wgmma INT8 PEAK (section "p_ rows" below; docs/NOTES-22.md "The synthetic peak"): wgmma issued back to back
// from shared-memory tiles, no memory traffic, no epilogue, accumulators in registers. They are an upper bound without memory system, TMA or epilogue,
// to read the real rows against instead of the 1979 TOPS datasheet figure. They take no shape or batch; --peak-seconds S sets the timed seconds per row
// (default 1.25: five launches of about 0.25 s, after a 0.3 s warm-up); --sustain also runs a sustained block per data kind (small, full, full4) on the
// best exact row of that kind. They were never run on Hopper, so they are NOT in the default selection ("all" = m_/f_/g_): select them with --variants p, and
// tools/build_rung1.sh runs them as a SECOND process after the primary pass, so a fault or hang in one cannot cost the primary results.
//
// Work order (so that a timeout or a hang keeps the most valuable lines, and the riskiest row is last): (0) with --variants p only: the p_ rows (about 3 s each), then with
// --sustain one sustained block per data kind; (1) default scheduler, first shape,
// L = first batch, every non-"last" f_ row and the m_ baseline, then the same at the other shapes; (2) the scheduler sweep (swizzle 1/2/4/8 x
// AlongM/AlongN, rows whose resolved (raster, swizzle) equals the default's are skipped as duplicates); (3) with --sustain: a sustained block
// for the best f_ row at the last shape; (4) the g_ rows; (5) the other batches (4, 1) for all of those; (6) the rows flagged "last" (cluster
// 4 x 1, never launched on any card before; --skip-last drops them). stderr gets "starting <variant> ..." BEFORE each row, so a hang names its row.
// Output is flushed per line. The first stdout line is a meta record (method "rung1_meta": driver, runtime, hostname, job id, commit, argv).
//
// Output: one JSON line per (variant, shape, batch, scheduler, mode) on stdout (method "rung1"), progress on stderr. mode "single": each
// repetition is one launch between two cudaEvents (includes the launch latency, a few us); mode "loop": each repetition is R back-to-back
// launches between two events, divided by R (amortises the launch; R = max(--loop, enough for 20 ms)); mode "sustain" (one record, with --sustain):
// back-to-back launches for the given seconds, tops over all of it and over its first and last second. Fields: tops = 2 m n k L / t (int8 TOPS,
// from the median), median/min/max in seconds (_s) and milliseconds (_ms), spread_pct = 100 (max - min) / median, reps.
// Scheduler fields: swizzle / raster are the REQUESTED values (raster "heuristic" = CUTLASS's default); swizzle_resolved / raster_resolved are
// what the scheduler actually uses for this tile, cluster and problem (the heuristic picks AlongM only when there are more N tiles than M tiles).
// Cluster rows: max_active_clusters_arg ("queried": hw_info.max_active_clusters from cudaOccupancyMaxActiveClusters with the kernel's real shared
// memory; "heuristic": left 0 as the rung-0 and stock-ceiling runs did), max_active_clusters, max_active_clusters_cutlass (CUTLASS's own query,
// which ignores dynamic shared memory), grid_x/y/z and ctas_launched (the persistent grid CUTLASS launches).
// p_ records (op "peak", mode "peak" = --peak-seconds worth of launches of about 0.25 s each, mode "sustain" with --sustain): n_tile, wg_per_cta, stages,
// ctas_per_sm, data (zero / small / full / full4), ceiling_kind (cold_burst_upper_bound / sustained), grid, iterations_blocks (loop count; 16 wgmma per block), wgmma_per_wg, ops_per_wgmma, tops = ops of one
// launch / median launch time (all SMs, all warpgroups), ops_clk_sm_clock64 (ops per SM clock per SM from the kernel's own clock64, no sampling: all the
// ops of an SM over (latest end - earliest start) of its warpgroups, after a bar.sync; datasheet 8192; valid only when cycles_max_over_min is about 1),
// sm_mhz_clock64, tops_per_ghz_nvml / ops_clk_sm_nvml (from NVML's clock), ceiling_tops_at_max_clock / _at_nvml_clock (SMs x 8192 x clock), datasheet_tops
// (1979 at 1.83 GHz, 132 SMs), sms_used / placement_ok (exactly the row's CTAs on every SM, in every logged launch), the NVML fields below, and "exact": EVERY accumulator of EVERY CTA and warpgroup equals iterations x (A.B^T of the tiles) mod 2^32 (int32 wraparound), after the
// timed block (its last launch, output poisoned beforehand), after the sustained block, and after a fresh launch at the end, plus the loop counter of every warpgroup; exact_check_trivial for the zero data (all results 0).
//
// "exact" and how it is decided (independent checks, all reported):
//   ref_mismatches   a sample of output bytes (--samples random ones plus corners and tile-edge positions) is compared with a HOST reference:
//                    sum over k of A * B (int64) reduced with the same modular definition ((s mod p) + p) mod p, p = moduli[slot0 + z].
//                    Operands come from a counter-based hash, so the host recomputes any element. This is the check against the definition.
//   hash_eq_first    a position-dependent sum-hash of the whole W plane stack equals the hash of the first row run on the same (shape, batch,
//                    slot0) (named in hash_ref_variant); all rows compute the same function, so equality across tile shapes, clusters, schedules,
//                    scheduler arguments and the mma.sync baseline is a check of every byte (up to hash collisions) relative to that first row.
//   exact_after_timing  the hash and the sampled reference are recomputed AFTER the timed loops (about a hundred launches later; catches a race
//                    that corrupts W only after warm-up or under sustained load) and must equal the first hash.
//   exact = (ref_mismatches == 0) && hash_eq_first && exact_after_timing. W is filled with 0xEE before the checked launch, so an unwritten tile
//   shows up. A failing row records the first sampled mismatch (first_bad_z/i/j, want, got). The spot-check cannot see a rare wrong byte that the
//   hash comparison also misses because the first row has the same fault: rung 0 is the known-answer test, but it is NOT a certificate for this
//   binary (a different binary and problem size, although both are built -DNDEBUG now).
//
// Clocks: cudaDevAttrClockRate (max SM clock, kHz) is always printed. If libnvidia-ml.so.1 can be dlopen'ed (no link-time dependency; --no-nvml
// to skip), a host thread polls it every 5 ms during each timed block: sm_mhz_min / mean, power_w_max, temp_c, throttle_reasons (OR of the NVML
// clocks-event reason bits EXCEPT 0x1 idle and 0x2 application clocks, which are reported apart as idle_app_flags: 0x4 SW power cap, 0x8 HW
// slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake). Without NVML those fields are null. nvmlDeviceGetPowerUsage is an average over
// about one second, so power_w_max lags a short block: use --sustain for a power-limited rate.
//
// Exit status: 0 done and every record exact and ok; 1 some record not exact; 2 usage or CUDA setup error; 3 no CUDA device or no table row
// for this device (the dev box; nothing is run); 4 a sticky device error (illegal address etc.) stopped the run after its record; 5 every
// measured row was exact but some record has status error / skipped (their count is printed). A hang is the caller's timeout (the build script
// runs this under `timeout -k 30`).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/kernel_hardware_info.h"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "k4/hopper/k4_hopper.cuh"
#include "k4/k4_ops.hpp"

namespace imd::detail {
const K4Ops& k4_ops_cfg0();  // src/k4/k4_cfg0.cu: the product's mma.sync K4, tile 128 x 256 (built into this binary by tools/build_rung1.sh)
}

namespace rung1 {  // (named, not anonymous: __global__ kernels in an anonymous namespace break the nvcc host stub)

using namespace imd::hopper;
namespace cg = cutlass::gemm;
using Coop = cg::KernelTmaWarpSpecializedCooperative;
using Ping = cg::KernelTmaWarpSpecializedPingpong;
using cute::Int;
using cute::Shape;
using _1 = Int<1>;
using _2 = Int<2>;
using _4 = Int<4>;
using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
using SchedParams = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params;

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };
struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };
struct Sticky : std::runtime_error { using std::runtime_error::runtime_error; };
#define CU(x)                                                                                                                  \
  do {                                                                                                                         \
    const cudaError_t e_ = (x);                                                                                                \
    if (e_ != cudaSuccess) throw Fail(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + #x + ": " + cudaGetErrorString(e_)); \
  } while (0)

// ------------------------------------------------------------------ data: a counter-based hash (matrix, row, k) -> int8 in [-127, 127]
__host__ __device__ inline uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__host__ __device__ inline int8_t val(uint64_t which, uint64_t row, uint64_t k) {
  return int8_t(int(mix((which << 56) ^ (row << 24) ^ k) % 255u) - 127);
}
__global__ void k_fill(int8_t* p, uint64_t which, uint64_t rows, uint64_t kk) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < rows * kk; i += uint64_t(gridDim.x) * blockDim.x)
    p[i] = val(which, i / kk, i % kk);
}
// position-dependent sum-hash of n 64-bit words
__global__ void k_hash(const uint64_t* w, uint64_t n, unsigned long long* out) {
  uint64_t h = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x) h += mix(w[i] ^ mix(i));
  for (int o = 16; o; o >>= 1) h += __shfl_down_sync(0xffffffffu, h, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, static_cast<unsigned long long>(h));
}

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit Buf(size_t n = 0) : bytes(n) { if (n) CU(cudaMalloc(&p, n)); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
  ~Buf() { if (p) cudaFree(p); }
  template <class T> T* as() const { return static_cast<T*>(p); }
};

// ------------------------------------------------------------------ JSON lines (the conventions of bench/ceiling: method, variant, tops, median_s, spread_pct, exact, status)
std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += (c == '\n') ? ' ' : c; }
  return o + "\"";
}
std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  return b;
}
struct Rec {
  std::vector<std::pair<std::string, std::string>> kv;
  Rec& s(const std::string& k, const std::string& v) { kv.emplace_back(k, jstr(v)); return *this; }
  Rec& n(const std::string& k, double v) { kv.emplace_back(k, jnum(v)); return *this; }
  Rec& i(const std::string& k, long long v) { kv.emplace_back(k, std::to_string(v)); return *this; }
  Rec& b(const std::string& k, bool v) { kv.emplace_back(k, v ? "true" : "false"); return *this; }
  Rec& null(const std::string& k) { kv.emplace_back(k, "null"); return *this; }
  Rec& raw(const std::string& k, const std::string& v) { kv.emplace_back(k, v); return *this; }
  std::string dump() const {
    std::string o = "{";
    for (size_t j = 0; j < kv.size(); ++j) o += (j ? ", " : "") + jstr(kv[j].first) + ": " + kv[j].second;
    return o + "}";
  }
};
void emit_line(const Rec& r) {
  std::printf("%s\n", r.dump().c_str());
  std::fflush(stdout);
}

// ------------------------------------------------------------------ NVML by dlopen (optional)
struct Nvml {
  using Ret = int;
  using Dev = void*;
  void* lib = nullptr;
  Dev dev = nullptr;
  Ret (*clock_info)(Dev, unsigned, unsigned*) = nullptr;
  Ret (*power)(Dev, unsigned*) = nullptr;
  Ret (*temp)(Dev, unsigned, unsigned*) = nullptr;
  Ret (*reasons)(Dev, unsigned long long*) = nullptr;
  std::string driver;
  bool ok = false;
  void open(int cuda_dev) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return;
    auto sym = [&](const char* n) { return dlsym(lib, n); };
    auto init = reinterpret_cast<Ret (*)()>(sym("nvmlInit_v2"));
    auto by_pci = reinterpret_cast<Ret (*)(const char*, Dev*)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    auto drv = reinterpret_cast<Ret (*)(char*, unsigned)>(sym("nvmlSystemGetDriverVersion"));
    clock_info = reinterpret_cast<decltype(clock_info)>(sym("nvmlDeviceGetClockInfo"));
    power = reinterpret_cast<decltype(power)>(sym("nvmlDeviceGetPowerUsage"));
    temp = reinterpret_cast<decltype(temp)>(sym("nvmlDeviceGetTemperature"));
    reasons = reinterpret_cast<decltype(reasons)>(sym("nvmlDeviceGetCurrentClocksEventReasons"));
    if (!reasons) reasons = reinterpret_cast<decltype(reasons)>(sym("nvmlDeviceGetCurrentClocksThrottleReasons"));
    char bus[64] = {};
    if (!init || !by_pci || !clock_info || cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) != cudaSuccess) return;
    if (init() != 0 || by_pci(bus, &dev) != 0) return;
    char v[96] = {};
    if (drv && drv(v, sizeof v) == 0) driver = v;
    ok = true;
  }
};
// polls the clock and power on a host thread while a timed block runs
struct Sampler {
  Nvml* nv = nullptr;
  std::thread th;
  std::atomic<bool> stop{false};
  double sum_mhz = 0, max_w = 0;
  unsigned min_mhz = ~0u, temp_c = 0, last_mhz = 0;
  double last_w = 0;
  unsigned long long reasons = 0;
  long long n = 0;
  void start(Nvml* nvml) {
    nv = nvml;
    if (!nv || !nv->ok) return;
    stop = false;
    th = std::thread([this] {
      while (!stop) {
        unsigned mhz = 0, mw = 0, t = 0;
        unsigned long long rs = 0;
        if (nv->clock_info(nv->dev, 1 /*NVML_CLOCK_SM*/, &mhz) == 0) { min_mhz = std::min(min_mhz, mhz); sum_mhz += mhz; ++n; last_mhz = mhz; }
        if (nv->power && nv->power(nv->dev, &mw) == 0) max_w = std::max(max_w, mw * 1e-3), last_w = mw * 1e-3;
        if (nv->temp && nv->temp(nv->dev, 0, &t) == 0) temp_c = t;
        if (nv->reasons && nv->reasons(nv->dev, &rs) == 0) reasons |= rs;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  void end() {
    if (th.joinable()) { stop = true; th.join(); }
  }
  ~Sampler() { end(); }  // an exception inside a timed block must not leave a joinable thread behind (std::terminate)
  void put(Rec& r) const {
    if (n == 0) {
      r.null("sm_mhz_min").null("sm_mhz_mean").null("power_w_max").null("temp_c").null("throttle_reasons").null("idle_app_flags");
      return;
    }
    char b[24], f[24];
    std::snprintf(b, sizeof b, "0x%llx", reasons & ~0x3ull);  // real throttle bits
    std::snprintf(f, sizeof f, "0x%llx", reasons & 0x3ull);   // 0x1 GPU idle, 0x2 application clocks setting: not throttling
    r.i("sm_mhz_min", min_mhz).n("sm_mhz_mean", sum_mhz / double(n)).n("power_w_max", max_w).i("temp_c", temp_c).s("throttle_reasons", b).s("idle_app_flags", f);
  }
};

// ------------------------------------------------------------------ options, context
struct Dims { long long m, n, k; };
bool operator<(const Dims& a, const Dims& b) { return std::tie(a.m, a.n, a.k) < std::tie(b.m, b.n, b.k); }
struct Sched { int swizzle = 1; int raster = 0; };  // raster: 0 heuristic (CUTLASS's default), 1 AlongM, 2 AlongN
const char* raster_name(int r) { return r == 1 ? "AlongM" : r == 2 ? "AlongN" : "heuristic"; }

struct Opt {
  int reps = 12, warmup = 3, loop = 8;
  double max_seconds = 8.0, sustain = 0;
  long long samples = 384;
  bool quick = false, nvml = true, skip_last = false;
  double peak_seconds = 1.25;  // p_ rows: total timed seconds per row (5 or more launches of about 0.25 s)
  std::string clusters = "queried";  // queried | heuristic | both
};
struct Fixture {
  Dims d;
  int lmax;
  Buf A, B, W;
  Fixture(Dims dd, int L)
      : d(dd), lmax(L), A(size_t(L) * dd.m * dd.k), B(size_t(L) * dd.n * dd.k), W(size_t(L) * dd.m * dd.n) {
    k_fill<<<2048, 256>>>(A.as<int8_t>(), 1, uint64_t(L) * d.m, d.k);  // batch z, row i is row z * m + i
    k_fill<<<2048, 256>>>(B.as<int8_t>(), 2, uint64_t(L) * d.n, d.k);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }
};
struct Entry;
struct Task {
  Dims d;
  int L;
  Sched sc;
  int occ = 0;       // cluster rows: 0 = hw_info.max_active_clusters queried, 1 = left 0 (CUTLASS's heuristic grid)
  int slot0 = 0;     // the first modulus slot: batch z uses moduli[slot0 + z] (0 holds p = 256, an epilogue-trivial modulus)
  double sustain = 0;  // > 0: also run a sustained block of this many seconds
  bool only_sustain = false;  // p_ rows: run the sustained block only (the work order's peak sustain item)
};
struct Ctx {
  Opt opt;
  std::string gpu, commit;
  int sms = 0, clock_khz = 0, mem_clock_khz = 0, cc_major = 0, cc_minor = 0;
  Nvml nvml;
  int lmax = 14, primary = 14;
  std::map<Dims, std::unique_ptr<Fixture>> fx;
  struct First { unsigned long long hash; std::string name; };
  std::map<std::tuple<long long, long long, long long, int, int>, First> first_hash;  // (m, n, k, L, slot0) -> hash of the first row run
  std::map<std::pair<const Entry*, long long>, double> loop_tops;  // (row, m) -> loop-mode tops of its primary-batch default-scheduler record
  std::map<const Entry*, double> peak_tops;                       // p_ row -> tops of its (exact) timed record
  long long n_rows = 0, n_inexact = 0, n_nonok = 0;
  Fixture& fixture(Dims d) {
    auto it = fx.find(d);
    if (it == fx.end()) it = fx.emplace(d, std::make_unique<Fixture>(d, lmax)).first;
    return *it->second;
  }
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
int mod_of(int64_t v, int p) { return int(((v % p) + p) % p); }

// ------------------------------------------------------------------ the rows
enum class Kind { Fast, Generic, Mma, Peak };
const char* kind_name(Kind k) {
  return k == Kind::Fast ? "fast_epilogue" : k == Kind::Generic ? "generic_epilogue" : k == Kind::Mma ? "mma_sync_product" : "wgmma_peak";
}

// One named configuration of the table kEntries (the only place that knows tile / cluster / schedule choices).
struct Entry {
  const char* name;
  int cc_major;  // the compute capability major this entry is for (9 = Hopper wgmma); the harness runs the entries of the device's major
  Kind kind;
  bool quick;    // part of --quick
  bool sweep;    // part of the scheduler sweep by default
  bool last;     // run after everything else (never launched on any card before: a hang here costs nothing already measured)
  int tile_m, tile_n, cl_m, cl_n;  // CTA tile rows / columns and cluster shape (to resolve the scheduler's raster order and swizzle)
  long long smem_bytes;  // dynamic shared memory per CTA, known at compile time: --list prints it without a GPU
  void (*run)(Ctx&, const Entry&, const Task&);
  int pk_data = -1;  // Kind::Peak rows: the operand data (0 zero, 1 small, 2 full range, 3 full range with 4 rotating stages), else -1
};

struct Prepared {  // what a kernel family supplies for one task
  std::function<void()> launch;  // one launch of the whole GEMM stack into the fixture's W
  Rec info;                      // extra record fields (registers, grid, ...)
  std::shared_ptr<void> keep;
};

struct Resolved { int raster; int swizzle; };  // raster 1 AlongM / 2 AlongN; swizzle = the max swizzle size the scheduler ends up using
Resolved resolve(const Entry& e, const Task& t) {
  if (e.kind == Kind::Mma || e.kind == Kind::Peak) return {0, 0};
  auto up = [](long long x, long long y) { return (x + y - 1) / y * y; };
  const long long tm = up((t.d.m + e.tile_m - 1) / e.tile_m, e.cl_m), tn = up((t.d.n + e.tile_n - 1) / e.tile_n, e.cl_n);
  const int lg = SchedParams::get_log_swizzle_size(int(tm), int(tn), t.sc.swizzle);
  const long long bm = up(tm, (1ll << lg) * e.cl_m), bn = up(tn, (1ll << lg) * e.cl_n);
  const Raster opt = t.sc.raster == 1 ? Raster::AlongM : t.sc.raster == 2 ? Raster::AlongN : Raster::Heuristic;
  const auto ro = SchedParams::get_rasterization_order(uint32_t(bm), uint32_t(bn), opt);
  return {ro == cutlass::gemm::kernel::detail::RasterOrder::AlongN ? 2 : 1, 1 << lg};
}

// ------------------------------------------------------------------ checking
struct Check {
  long long checked = 0, bad = 0;
  long long z = -1, i = -1, j = -1, want = -1, got = -1;  // the first sampled mismatch
  unsigned long long hash = 0;
};
Check verify(Ctx& c, Fixture& fx, const Task& t) {
  const int M = int(t.d.m), N = int(t.d.n), K = int(t.d.k), L = t.L;
  uint8_t* W = fx.W.as<uint8_t>();
  Check ck;
  const long long per = (long long)M * N;
  std::vector<std::array<long long, 3>> pos;  // (z, i, j)
  pos.push_back({0, 0, 0});
  pos.push_back({L - 1, M - 1, N - 1});
  pos.push_back({L - 1, 0, N - 1});
  pos.push_back({0, M - 1, 0});
  for (long long e : {127LL, 128LL, 255LL, 256LL, 511LL, 512LL})
    if (e < M && e < N) pos.push_back({(e + 3) % L, e, N - 1 - e});
  for (long long q = 0; q < c.opt.samples; ++q) {
    const uint64_t h = mix(uint64_t(q) * 7919u + 17);
    pos.push_back({(long long)(h % uint64_t(L)), (long long)((h >> 20) % uint64_t(M)), (long long)((h >> 40) % uint64_t(N))});
  }
  for (const auto& p : pos) {
    const long long z = p[0], i = p[1], j = p[2];
    long long s = 0;
    for (long long kk = 0; kk < K; ++kk) s += (long long)val(1, uint64_t(z * M + i), uint64_t(kk)) * (long long)val(2, uint64_t(z * N + j), uint64_t(kk));
    const int want = mod_of(s, int(imd::constants::kTables.moduli[t.slot0 + z]));
    uint8_t got = 0;
    CU(cudaMemcpy(&got, W + z * per + i * N + j, 1, cudaMemcpyDeviceToHost));
    ++ck.checked;
    if (got != want) {
      if (ck.bad == 0) ck.z = z, ck.i = i, ck.j = j, ck.want = want, ck.got = got;
      ++ck.bad;
    }
  }
  Buf hbuf(8);
  CU(cudaMemset(hbuf.p, 0, 8));
  k_hash<<<1024, 256>>>(fx.W.as<uint64_t>(), uint64_t(L) * M * N / 8, hbuf.as<unsigned long long>());
  CU(cudaMemcpy(&ck.hash, hbuf.p, 8, cudaMemcpyDeviceToHost));
  return ck;
}

// ------------------------------------------------------------------ timing
void ramp(const std::function<void()>& launch) {  // keep the card busy for ~0.3 s so the clocks are up before the first timed block
  const auto t0 = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 4; ++i) launch();
    CU(cudaDeviceSynchronize());
  } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3);
}

// the two timing modes; returns {single, loop} records; *est = single-mode median
std::vector<Rec> time_modes(Ctx& c, const std::function<void()>& launch, const Task& t, const Rec& base, double* est_out) {
  const double ops = 2.0 * double(t.d.m) * double(t.d.n) * double(t.d.k) * double(t.L);
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  struct Guard { cudaEvent_t a, b; ~Guard() { cudaEventDestroy(a); cudaEventDestroy(b); } } guard{e0, e1};
  for (int w = 0; w < c.opt.warmup; ++w) launch();
  CU(cudaDeviceSynchronize());
  ramp(launch);
  std::vector<Rec> out;
  double est = 0;
  for (int mode = 0; mode < 2; ++mode) {
    const int R = mode == 0 ? 1 : std::max(c.opt.loop, std::min(200, int(std::ceil(0.020 / std::max(est, 1e-6)))));
    std::vector<double> ts;
    double total = 0;
    Sampler sm;
    sm.start(c.opt.nvml ? &c.nvml : nullptr);
    for (int r = 0; r < std::max(c.opt.reps, 10); ++r) {  // at least 10 repetitions (the contract), --reps more if asked, capped by --max-seconds beyond 10
      CU(cudaEventRecord(e0, nullptr));
      for (int i = 0; i < R; ++i) launch();
      CU(cudaEventRecord(e1, nullptr));
      CU(cudaEventSynchronize(e1));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      ts.push_back(ms * 1e-3 / R);
      total += ms * 1e-3;
      if (int(ts.size()) >= 10 && total > c.opt.max_seconds) break;  // reps above 10 stop at the time cap
    }
    sm.end();
    CU(cudaGetLastError());
    const double med = median(ts), mn = *std::min_element(ts.begin(), ts.end()), mx = *std::max_element(ts.begin(), ts.end());
    if (mode == 0) est = med;
    Rec r = base;
    r.s("mode", mode == 0 ? "single" : "loop").i("launches_per_rep", R).i("reps", (long long)ts.size());
    r.n("tops", ops / med / 1e12).n("tops_best", ops / mn / 1e12).n("tops_worst", ops / mx / 1e12);
    r.n("median_s", med).n("min_s", mn).n("max_s", mx).n("median_ms", med * 1e3).n("min_ms", mn * 1e3).n("max_ms", mx * 1e3);
    r.n("spread_pct", 100.0 * (mx - mn) / med);
    sm.put(r);
    out.push_back(r);
  }
  *est_out = est;
  return out;
}

// a sustained block: chunks of ~100 ms of back-to-back launches for `seconds`; tops over all, the first second and the last second
Rec sustain_block(Ctx& c, const std::function<void()>& launch, const Task& t, const Rec& base, double est) {
  const double ops = 2.0 * double(t.d.m) * double(t.d.n) * double(t.d.k) * double(t.L);
  const int R = std::max(1, int(std::ceil(0.1 / std::max(est, 1e-6))));
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  struct Guard { cudaEvent_t a, b; ~Guard() { cudaEventDestroy(a); cudaEventDestroy(b); } } guard{e0, e1};
  std::vector<double> dt;  // seconds per chunk
  double total = 0;
  Sampler sm;
  sm.start(c.opt.nvml ? &c.nvml : nullptr);
  while (total < t.sustain) {
    CU(cudaEventRecord(e0, nullptr));
    for (int i = 0; i < R; ++i) launch();
    CU(cudaEventRecord(e1, nullptr));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    dt.push_back(ms * 1e-3);
    total += ms * 1e-3;
  }
  sm.end();
  CU(cudaGetLastError());
  auto window = [&](bool first) {
    double tt = 0;
    long long n = 0;
    if (first)
      for (size_t k = 0; k < dt.size() && tt < 1.0; ++k) tt += dt[k], ++n;
    else
      for (size_t k = dt.size(); k-- > 0 && tt < 1.0;) tt += dt[k], ++n;
    return ops * double(n) * R / tt / 1e12;
  };
  Rec r = base;
  r.s("mode", "sustain").i("launches_per_chunk", R).i("chunks", (long long)dt.size()).n("seconds", total);
  r.n("tops", ops * double(dt.size()) * R / total / 1e12).n("tops_first_1s", window(true)).n("tops_last_1s", window(false));
  sm.put(r);
  return r;
}

// ------------------------------------------------------------------ one row, one task
// prep builds the kernel's arguments for the fixture and returns the launch closure; everything else (checks, timing, records) is common
void measure(Ctx& c, const Entry& e, const Task& t, const std::function<Prepared(Fixture&)>& prep) {
  const int M = int(t.d.m), N = int(t.d.n), K = int(t.d.k), L = t.L;
  const Resolved rs = resolve(e, t);
  const bool cluster_row = e.cl_m * e.cl_n > 1;
  std::fprintf(stderr, "starting %s %dx%dx%d L=%d slot0=%d sw=%d %s%s%s\n", e.name, M, N, K, L, t.slot0, t.sc.swizzle, raster_name(t.sc.raster),
               cluster_row ? (t.occ ? " clusters=heuristic" : " clusters=queried") : "", t.sustain > 0 ? " +sustain" : "");
  Rec base;
  base.s("method", "rung1").s("variant", e.name).s("kind", kind_name(e.kind)).s("op", "store");
  base.i("m", M).i("n", N).i("k", K).i("batch", L).i("slot0", t.slot0);
  if (e.kind == Kind::Mma)
    base.null("swizzle").null("raster").null("swizzle_resolved").null("raster_resolved");
  else
    base.i("swizzle", t.sc.swizzle).s("raster", raster_name(t.sc.raster)).i("swizzle_resolved", rs.swizzle).s("raster_resolved", raster_name(rs.raster));
  base.s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).i("sm_count", c.sms);
  base.i("clock_khz_max", c.clock_khz).i("mem_clock_khz", c.mem_clock_khz);
  base.b("ndebug",
#ifdef NDEBUG
         true
#else
         false
#endif
  );
  if (!c.commit.empty()) base.s("commit", c.commit);
  ++c.n_rows;
  auto emit_status = [&](const std::string& st) {  // error / skipped: no exactness claim
    Rec r = base;
    r.null("exact").s("status", st);
    emit_line(r);
    ++c.n_nonok;
    std::fprintf(stderr, "  %s %dx%dx%d L=%d: %s\n", e.name, M, N, K, L, st.c_str());
  };
  try {
    Fixture& fx = c.fixture(t.d);
    uint8_t* W = fx.W.as<uint8_t>();
    Prepared pr = prep(fx);
    for (auto& kv : pr.info.kv) base.kv.push_back(kv);

    // ---- exactness before timing: a launch on a poisoned W, then the sampled host reference and the hash
    CU(cudaMemset(W, 0xEE, fx.W.bytes));
    pr.launch();
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    const Check ck = verify(c, fx, t);
    const auto key = std::make_tuple(t.d.m, t.d.n, t.d.k, L, t.slot0);
    const bool first = !c.first_hash.count(key);
    if (first) c.first_hash[key] = {ck.hash, e.name};
    const Ctx::First ref = c.first_hash[key];
    const bool heq = ref.hash == ck.hash;

    // ---- timing
    double est = 0;
    std::vector<Rec> recs = time_modes(c, pr.launch, t, base, &est);
    if (t.sustain > 0) recs.push_back(sustain_block(c, pr.launch, t, base, est));

    // ---- exactness after timing: the same checks on W as the timed launches left it
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    const Check ck2 = verify(c, fx, t);
    const bool after = ck2.bad == 0 && ck2.hash == ref.hash;
    const bool exact = ck.bad == 0 && heq && after;
    if (!exact) ++c.n_inexact;
    char hb[24], hb2[24];
    std::snprintf(hb, sizeof hb, "%016llx", ck.hash);
    std::snprintf(hb2, sizeof hb2, "%016llx", ck2.hash);
    for (Rec& r : recs) {
      r.b("exact", exact).i("check_elements", ck.checked).i("ref_mismatches", ck.bad).s("hash", hb).b("hash_eq_first", heq).b("is_first_hash", first);
      r.s("hash_ref_variant", ref.name).b("exact_after_timing", after).i("ref_mismatches_after", ck2.bad).s("hash_after", hb2);
      if (ck.bad) r.i("first_bad_z", ck.z).i("first_bad_i", ck.i).i("first_bad_j", ck.j).i("first_bad_want", ck.want).i("first_bad_got", ck.got);
      else if (ck2.bad) r.i("first_bad_z", ck2.z).i("first_bad_i", ck2.i).i("first_bad_j", ck2.j).i("first_bad_want", ck2.want).i("first_bad_got", ck2.got);
      r.s("status", "ok");
      emit_line(r);
    }
    auto tops_in = [](const Rec& r) {
      for (const auto& kv : r.kv)
        if (kv.first == "tops") return std::atof(kv.second.c_str());
      return 0.0;
    };
    const double ts = tops_in(recs[0]), tl = tops_in(recs[1]);
    if (t.sc.swizzle == 1 && t.sc.raster == 0 && t.occ == 0 && t.sustain == 0 && L == c.primary) c.loop_tops[{&e, t.d.m}] = tl;
    std::fprintf(stderr, "  %-26s %5dx%dx%d L=%-2d sw=%d %-9s exact=%d  single %.0f  loop %.0f TOPS\n", e.name, M, N, K, L, rs.swizzle,
                 raster_name(rs.raster), int(exact), ts, tl);
  } catch (const Skip& s) {
    emit_status(std::string("skipped: ") + s.what());
  } catch (const std::exception& ex) {
    cudaGetLastError();
    // a sticky error (illegal address, hardware exception) cannot be recovered from: report it and stop
    if (cudaDeviceSynchronize() != cudaSuccess) {
      emit_status(std::string("error (sticky): ") + ex.what());
      throw Sticky(ex.what());
    }
    emit_status(std::string("error: ") + ex.what());
  }
}

// ---- CUTLASS Hopper rows
template <class T, class C, class S, bool F>
struct VT {
  using Tile = T;
  using Cluster = C;
  using Sched = S;
  static constexpr bool kFast = F;
};

template <class V>
constexpr long long smem_of() {
  return (long long)Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>::Kernel::SharedStorageSize;
}

// hw_info.max_active_clusters as the product should pass it: the occupancy of the REAL kernel (its dynamic shared memory included)
template <class Kernel>
int occupancy_clusters() {
  const void* kernel = reinterpret_cast<const void*>(&cutlass::device_kernel<Kernel>);
  CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(Kernel::SharedStorageSize)));
  cudaLaunchConfig_t cfg{};
  const unsigned cx = unsigned(cute::size<0>(typename Kernel::ClusterShape{})), cy = unsigned(cute::size<1>(typename Kernel::ClusterShape{})),
                 cz = unsigned(cute::size<2>(typename Kernel::ClusterShape{}));
  cfg.gridDim = dim3(cx, cy, cz);
  cfg.blockDim = dim3(Kernel::MaxThreadsPerBlock, 1, 1);
  cfg.dynamicSmemBytes = Kernel::SharedStorageSize;
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeClusterDimension;
  at.val.clusterDim.x = cx, at.val.clusterDim.y = cy, at.val.clusterDim.z = cz;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  int n = 0;
  CU(cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
  return n;
}

template <class V>
void run_variant(Ctx& c, const Entry& e, const Task& t) {
  using Store = Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>;
  using Kernel = typename Store::Kernel;
  using Gemm = cg::device::GemmUniversalAdapter<Kernel>;
  measure(c, e, t, [&](Fixture& fx) -> Prepared {
    const int M = int(t.d.m), N = int(t.d.n), K = int(t.d.k), L = t.L;
    Prepared pr;
    cudaFuncAttributes fa{};
    CU(cudaFuncGetAttributes(&fa, cutlass::device_kernel<Kernel>));
    pr.info.i("regs", fa.numRegs).i("local_bytes", (long long)fa.localSizeBytes).i("smem_dynamic_bytes", (long long)Kernel::SharedStorageSize);
    pr.info.i("tile_m", e.tile_m).i("tile_n", e.tile_n).i("cluster_m", e.cl_m).i("cluster_n", e.cl_n);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = c.sms;
    const int occ = occupancy_clusters<Kernel>();
    hw.max_active_clusters = t.occ ? 0 : occ;
    pr.info.s("max_active_clusters_arg", t.occ ? "heuristic" : "queried").i("max_active_clusters", hw.max_active_clusters).i("max_active_clusters_occupancy", occ);
    pr.info.i("max_active_clusters_cutlass", cutlass::KernelHardwareInfo::query_device_max_active_clusters<Kernel>());
    const auto sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M) * K);
    const auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(N) * K);
    K4Args ea{};
    ea.W = fx.W.as<uint8_t>(), ea.ldw = N, ea.plane_w = int64_t(M) * N, ea.slot0 = t.slot0, ea.b = 1, ea.batch0 = 0;
    typename Gemm::Arguments args{};
    args.mode = cg::GemmUniversalMode::kGemm;
    args.problem_shape = {M, N, K, L};
    args.mainloop.ptr_A = fx.A.as<int8_t>();
    args.mainloop.dA = sA;
    args.mainloop.ptr_B = fx.B.as<int8_t>();
    args.mainloop.dB = sB;
    args.epilogue = ea;
    args.hw_info = hw;
    args.scheduler.max_swizzle_size = t.sc.swizzle;
    args.scheduler.raster_order = t.sc.raster == 1 ? Raster::AlongM : t.sc.raster == 2 ? Raster::AlongN : Raster::Heuristic;
    auto gemm = std::make_shared<Gemm>();
    if (gemm->can_implement(args) != cutlass::Status::kSuccess) throw Skip("can_implement failed");
    const dim3 g = Gemm::get_grid_shape(args);
    pr.info.i("grid_x", g.x).i("grid_y", g.y).i("grid_z", g.z).i("ctas_launched", (long long)g.x * g.y * g.z);
    auto ws = std::make_shared<Buf>(Gemm::get_workspace_size(args));
    if (gemm->initialize(args, ws->p) != cutlass::Status::kSuccess) throw Fail("initialize failed");
    pr.keep = ws;
    pr.launch = [gemm, ws] {
      if (gemm->run() != cutlass::Status::kSuccess) throw Fail("gemm.run failed");
    };
    return pr;
  });
}

// ---- the product's mma.sync K4 (configuration 0: 128 x 256 tile, 8 warps), one launch over the L batches, as the product's k4 launch does
void run_mma(Ctx& c, const Entry& e, const Task& t) {
  measure(c, e, t, [&](Fixture& fx) -> Prepared {
    const int M = int(t.d.m), N = int(t.d.n), K = int(t.d.k), L = t.L;
    const imd::detail::K4Ops& ops = imd::detail::k4_ops_cfg0();
    int occ_store = 0, occ_add = 0;
    CU(ops.setup(&occ_store, &occ_add));
    if (occ_store < 1) throw Skip("mma.sync K4 occupancy 0");
    auto blob = std::make_shared<imd::detail::K4Blob>();
    ops.make(blob.get(), M, N, K, K, K, N, L, int64_t(c.sms) * occ_store);
    const imd::detail::K4Call call{fx.A.as<int8_t>(), fx.B.as<int8_t>(), fx.W.as<uint8_t>(), int64_t(M) * K, int64_t(N) * K, int64_t(M) * N, t.slot0, 1, 0};
    Prepared pr;
    pr.info.i("smem_dynamic_bytes", ops.smem_bytes).i("threads", ops.threads).i("ctas_per_sm", occ_store).i("resident_ctas", int64_t(c.sms) * occ_store);
    pr.info.i("tile_m", e.tile_m).i("tile_n", e.tile_n);
    pr.launch = [blob, call, &ops] {
      if (!ops.launch(*blob, false, call, nullptr)) throw Fail("mma.sync K4 launch failed");
    };
    return pr;
  });
}

// ================================================================== p_ rows: the SYNTHETIC wgmma INT8 peak (no memory traffic, no epilogue)
// What is the fastest INT8 rate this card sustains with wgmma when nothing but the tensor cores is working? The CTA copies a few operand tiles into
// shared memory ONCE, then every consumer warpgroup issues wgmma m64nNk32 s32.s8.s8 (SS form: both operands from shared memory by descriptor, K-major,
// 128-byte swizzle) back to back on registers, `kPeakWgmmaPerBlock` = 16 per commit group, one group always in flight (wait_group 1), for a fixed number
// of blocks; the accumulators are written to global memory once at the end (so the loop cannot be removed, and so they can be checked). The shared
// memory layouts, descriptors, fragments and the mma call are CuTe/CUTLASS's own (tile_to_shape of GMMA::Layout_K_SW128_Atom, make_fragment_A/B,
// cute::gemm), the ones the stock sm90 TMA mainloop uses; there is no hand-written descriptor. docs/NOTES-22.md section "The synthetic peak".
constexpr int kPeakKTile = 128;         // K per tile: 128 int8 = 128 bytes = one SW128 atom wide
constexpr int kPeakTilesPerBlock = 4;   // K tiles per block (a block = one commit group)
constexpr int kPeakWgmmaPerBlock = kPeakTilesPerBlock * (kPeakKTile / 32);  // 16
constexpr double kDatasheetTops = 1979, kDatasheetClockGhz = 1.83;  // H100 SXM dense INT8 as printed: 132 SMs x 8192 x 1.83 GHz
constexpr int kDatasheetOpsClkSm = 8192;  // H100 SXM dense INT8: 4096 MAC = 8192 ops per clock per SM (1979 TOPS = 132 SMs x 8192 x 1.83 GHz)

struct PeakStat {  // one per (CTA, warpgroup), written by thread 0 of the warpgroup
  unsigned long long t0, t1;  // absolute clock64 (the SM-wide counter) at the start and at the end of the wgmma loop
  unsigned iters;             // the loop counter at exit (must equal the requested block count)
  unsigned smid;              // %smid
};

template <int N> struct PeakAtom;
template <> struct PeakAtom<256> { using type = cute::SM90_64x256x32_S32S8S8_SS_TN; };
template <> struct PeakAtom<128> { using type = cute::SM90_64x128x32_S32S8S8_SS_TN; };

template <int N, int NWG, int ST>
struct PeakCfg {
  using TiledMma = decltype(cute::make_tiled_mma(typename PeakAtom<N>::type{}));
  using Atom = cute::GMMA::Layout_K_SW128_Atom<int8_t>;
  using SmemA = decltype(cute::tile_to_shape(Atom{}, cute::make_shape(cute::Int<64>{}, cute::Int<kPeakKTile>{}, cute::Int<ST>{}), cute::Step<cute::_1, cute::_2, cute::_3>{}));
  using SmemB = decltype(cute::tile_to_shape(Atom{}, cute::make_shape(cute::Int<N>{}, cute::Int<kPeakKTile>{}, cute::Int<ST>{}), cute::Step<cute::_1, cute::_2, cute::_3>{}));
  static constexpr size_t kBytesB = size_t(ST) * N * kPeakKTile, kBytesAwg = size_t(ST) * 64 * kPeakKTile;
  static constexpr size_t kNeed = kBytesB + NWG * kBytesAwg + 1024;  // + 1 KiB: the swizzle pattern repeats every 1024 bytes, the base is aligned by hand
};
// dynamic shared memory of a launch: a one-CTA-per-SM row asks for more than half of the 227 KiB so that no second CTA can share an SM
// (CTAS = 2: 76 KiB each, so 2 fit (2 x 77 KiB) and 3 cannot (3 x 77 KiB > 228 KiB): the occupancy is exactly 2, checked at run time)
template <int N, int NWG, int ST, int CTAS>
constexpr long long peak_smem() {
  return (long long)(PeakCfg<N, NWG, ST>::kNeed > size_t(CTAS == 1 ? 116 * 1024 : 76 * 1024) ? PeakCfg<N, NWG, ST>::kNeed : size_t(CTAS == 1 ? 116 * 1024 : 76 * 1024));
}

// gA: [NWG][ST][64][128] int8 (row-major, k contiguous), gB: [ST][N][128]; out: [cta][wg][N/2 registers][128 threads] int32
template <int N, int NWG, int ST>
__global__ void __launch_bounds__(128 * NWG, 1)
    k_wgmma_peak(const int8_t* __restrict__ gA, const int8_t* __restrict__ gB, int nblk, int32_t* __restrict__ out, PeakStat* __restrict__ stat,
                  unsigned* __restrict__ plog) {  // plog: this launch's row of the placement log ([cta] -> %smid), or null
  using namespace cute;
  using Cfg = PeakCfg<N, NWG, ST>;
  extern __shared__ char peak_smem_raw[];
  const uint32_t sbase = cast_smem_ptr_to_uint(peak_smem_raw);
  char* const base = peak_smem_raw + ((1024u - (sbase & 1023u)) & 1023u);
  const int tid = threadIdx.x, wg = tid / 128;
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<int8_t*>(base)), typename Cfg::SmemB{});
  // the operand tiles, once: logical (row, k, stage) -> swizzled address by CuTe's layout
  for (int i = tid; i < ST * N * kPeakKTile; i += blockDim.x) {
    const int k = i % kPeakKTile, r = (i / kPeakKTile) % N, s = i / (kPeakKTile * N);
    sB(r, k, s) = gB[i];
  }
  for (int w = 0; w < NWG; ++w) {
    Tensor sAw = make_tensor(make_smem_ptr(reinterpret_cast<int8_t*>(base + Cfg::kBytesB + w * Cfg::kBytesAwg)), typename Cfg::SmemA{});
    for (int i = tid; i < ST * 64 * kPeakKTile; i += blockDim.x) {
      const int k = i % kPeakKTile, r = (i / kPeakKTile) % 64, s = i / (kPeakKTile * 64);
      sAw(r, k, s) = gA[(size_t(w) * ST * 64 + size_t(s) * 64 + r) * kPeakKTile + k];
    }
  }
  cutlass::arch::fence_view_async_shared();  // generic-proxy writes -> visible to the async proxy (wgmma reads shared memory through it)
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<int8_t*>(base + Cfg::kBytesB + wg * Cfg::kBytesAwg)), typename Cfg::SmemA{});
  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid % 128);
  Tensor tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA));  // (MMA, MMA_M, MMA_K, ST): descriptors
  Tensor tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB));  // (MMA, MMA_N, MMA_K, ST)
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<64>, Int<N>>{});
  static_assert(size(accum) == N / 2, "64 x N int32 accumulators per warpgroup: N / 2 registers per thread");
  static_assert(size<2>(tCrA) == kPeakKTile / 32, "four k32 instructions per 128-wide tile");
  clear(accum);
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;  // the accumulators start at zero, every instruction adds
  __syncthreads();  // bar.sync over all 128 NWG threads: every warpgroup of the CTA starts its timed loop together
  const long long t0 = clock64();
  int b = 0;
  for (; b < nblk; ++b) {
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    CUTE_UNROLL
    for (int j = 0; j < kPeakTilesPerBlock; ++j)  // tile j of the block lives in stage j % ST (ST = 1: the same tile four times; ST = 4: four different tiles)
      CUTE_UNROLL
      for (int k = 0; k < kPeakKTile / 32; ++k) cute::gemm(tiled_mma, tCrA(_, _, k, j % ST), tCrB(_, _, k, j % ST), accum);
    warpgroup_commit_batch();
    warpgroup_wait<1>();  // the previous group is done, this one is running while the next one is issued
    warpgroup_fence_operand(accum);
  }
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);
  const long long t1 = clock64();
  const int t = tid % 128;
  int32_t* o = out + (size_t(blockIdx.x) * NWG + wg) * (N / 2) * 128 + t;
  CUTE_UNROLL
  for (int v = 0; v < N / 2; ++v) o[size_t(v) * 128] = accum(v);
  if (t == 0) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    stat[size_t(blockIdx.x) * NWG + wg] = {static_cast<unsigned long long>(t0), static_cast<unsigned long long>(t1), unsigned(b), smid};
    if (plog && tid == 0) plog[blockIdx.x] = smid;
  }
}

int8_t peak_val(int data, uint64_t which, uint64_t row, uint64_t k) {
  if (data == 0) return 0;
  if (data == 1) return int8_t(int(mix((which << 56) ^ (row << 24) ^ k) % 7u) - 3);  // small random: -3 .. 3
  return val(which, row, k);                                                            // full range: -127 .. 127
}
const char* peak_data_name(int d) { return d == 0 ? "zero" : d == 1 ? "small" : d == 2 ? "full" : "full4"; }

struct PeakCheck {
  long long checked = 0, bad = 0, iters_bad = 0, sms_used = 0;
  long long cta = -1, wg = -1, thr = -1, reg = -1;
  long long want = 0, got = 0;
  // rate: ops per clock per SM from clock64 = (all ops of the SM) / (latest end - earliest start of the SM's warpgroups); only valid when cycles_max / cycles_min is about 1
  double rate_mean = 0, rate_min = 0, cycles_min = 0, cycles_mean = 0, cycles_max = 0;
  bool placed = false;  // this launch: exactly `ctas` CTAs on every one of the card's SMs
};

template <int N, int NWG, int ST>
void run_peak_impl(Ctx& c, const Entry& e, const Task& t, int ctas, int data) {
  constexpr int NACC = N / 2;
  const int grid = c.sms * ctas;
  const long long smem = e.smem_bytes;
  const double ops_wgmma = 2.0 * 64 * N * 32;
  std::fprintf(stderr, "starting %s N=%d wg/cta=%d stages=%d ctas/sm=%d data=%s grid=%d smem=%lld%s\n", e.name, N, NWG, ST, ctas, peak_data_name(data), grid, smem,
               t.only_sustain ? " sustain only" : "");
  Rec base;
  base.s("method", "rung1").s("variant", e.name).s("kind", kind_name(e.kind)).s("op", "peak");
  base.i("n_tile", N).i("wg_per_cta", NWG).i("stages", ST).i("ctas_per_sm", ctas).s("data", peak_data_name(data)).i("grid", grid).i("block_threads", 128 * NWG);
  base.s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).i("sm_count", c.sms);
  base.i("clock_khz_max", c.clock_khz).i("mem_clock_khz", c.mem_clock_khz);
  base.b("ndebug",
#ifdef NDEBUG
         true
#else
         false
#endif
  );
  if (!c.commit.empty()) base.s("commit", c.commit);
  ++c.n_rows;
  auto emit_status = [&](const std::string& st) {
    Rec r = base;
    r.null("exact").s("status", st);
    emit_line(r);
    ++c.n_nonok;
    std::fprintf(stderr, "  %s: %s\n", e.name, st.c_str());
  };
  try {
    auto kernel = &k_wgmma_peak<N, NWG, ST>;
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));
    if (smem > (long long)prop.sharedMemPerBlockOptin) throw Skip("needs " + std::to_string(smem) + " bytes of shared memory, the device allows " + std::to_string(prop.sharedMemPerBlockOptin));
    CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    cudaFuncAttributes fa{};
    CU(cudaFuncGetAttributes(&fa, kernel));
    int occ = 0;
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, 128 * NWG, size_t(smem)));
    base.i("regs", fa.numRegs).i("local_bytes", (long long)fa.localSizeBytes).i("smem_dynamic_bytes", smem).i("max_blocks_per_sm", occ);
    if (occ != ctas) throw Skip("occupancy " + std::to_string(occ) + " CTA(s) per SM, the row needs exactly " + std::to_string(ctas));

    // ---- data: tiles from the counter-based hash (or zeros / small values), the same bytes for every CTA
    std::vector<int8_t> hA(size_t(NWG) * ST * 64 * kPeakKTile), hB(size_t(ST) * N * kPeakKTile);
    for (int w = 0; w < NWG; ++w)
      for (int s = 0; s < ST; ++s)
        for (int r = 0; r < 64; ++r)
          for (int k = 0; k < kPeakKTile; ++k) hA[((size_t(w) * ST + s) * 64 + r) * kPeakKTile + k] = peak_val(data, 100 + w * 8 + s, r, k);
    for (int s = 0; s < ST; ++s)
      for (int r = 0; r < N; ++r)
        for (int k = 0; k < kPeakKTile; ++k) hB[(size_t(s) * N + r) * kPeakKTile + k] = peak_val(data, 200 + s, r, k);
    constexpr int kLogCap = 2048;  // launches whose placement ([cta] -> %smid) is logged on the device: the timed, sustained and final ones (not the warm-up)
    Buf dA(hA.size()), dB(hB.size()), dOut(size_t(grid) * NWG * NACC * 128 * 4), dStat(size_t(grid) * NWG * sizeof(PeakStat)), dLog(size_t(kLogCap) * grid * 4);
    int logged = 0, launches_total = 0;
    CU(cudaMemcpy(dA.p, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dB.p, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    // expected accumulator per block: S[w][row][col] = sum over the block's four tiles (stage j % ST) of (A_tile B_tile^T); int32 wraparound is the
    // hardware's (the non-saturating instruction wraps mod 2^32) and deterministic, so the check is exact: acc = nblk * S mod 2^32
    std::vector<int32_t> S(size_t(NWG) * 64 * N, 0);
    for (int w = 0; w < NWG; ++w)
      for (int j = 0; j < kPeakTilesPerBlock; ++j) {
        const int s = j % ST;
        for (int r = 0; r < 64; ++r) {
          const int8_t* a = &hA[((size_t(w) * ST + s) * 64 + r) * kPeakKTile];
          for (int q = 0; q < N; ++q) {
            const int8_t* b = &hB[(size_t(s) * N + q) * kPeakKTile];
            int32_t acc = 0;
            for (int k = 0; k < kPeakKTile; ++k) acc += int32_t(a[k]) * int32_t(b[k]);
            S[(size_t(w) * 64 + r) * N + q] += acc;
          }
        }
      }
    auto launch = [&](int nblk, bool log) {
      unsigned* plog = nullptr;
      if (log) {
        ++launches_total;
        if (logged < kLogCap) plog = dLog.as<unsigned>() + size_t(logged++) * grid;
      }
      kernel<<<grid, 128 * NWG, size_t(smem)>>>(dA.as<int8_t>(), dB.as<int8_t>(), nblk, dOut.as<int32_t>(), dStat.as<PeakStat>(), plog);
      CU(cudaGetLastError());
    };
    // exactly `ctas` CTAs on every one of the card's SMs, from a list of the CTAs' %smid
    auto placed = [&](const unsigned* smids) {
      std::map<unsigned, int> cnt;
      for (int i = 0; i < grid; ++i) ++cnt[smids[i]];
      if (int(cnt.size()) != c.sms) return false;
      for (const auto& kv : cnt)
        if (kv.second != ctas) return false;
      return true;
    };
    auto check = [&](int nblk) {  // a poisoned-output launch has run; compare every accumulator of every CTA and warpgroup
      PeakCheck ck;
      std::vector<int32_t> o(size_t(grid) * NWG * NACC * 128);
      std::vector<PeakStat> st(size_t(grid) * NWG);
      CU(cudaMemcpy(o.data(), dOut.p, o.size() * 4, cudaMemcpyDeviceToHost));
      CU(cudaMemcpy(st.data(), dStat.p, st.size() * sizeof(PeakStat), cudaMemcpyDeviceToHost));
      for (int cta = 0; cta < grid; ++cta)
        for (int w = 0; w < NWG; ++w)
          for (int v = 0; v < NACC; ++v)
            for (int th = 0; th < 128; ++th) {
              const int warp = th / 32, lane = th % 32;
              const int row = 16 * warp + lane / 4 + 8 * ((v / 2) % 2), col = 8 * (v / 4) + 2 * (lane % 4) + (v % 2);  // the PTX wgmma D fragment
              const uint32_t want = uint32_t(uint64_t(int64_t(S[(size_t(w) * 64 + row) * N + col])) * uint64_t(nblk));
              const uint32_t got = uint32_t(o[((size_t(cta) * NWG + w) * NACC + v) * 128 + th]);
              ++ck.checked;
              if (got != want) {
                if (ck.bad == 0) ck.cta = cta, ck.wg = w, ck.thr = th, ck.reg = v, ck.want = want, ck.got = got;
                ++ck.bad;
              }
            }
      struct SmAcc { double ops = 0; unsigned long long t0 = ~0ull, t1 = 0; };
      std::map<unsigned, SmAcc> per_sm;
      std::vector<unsigned> smids(grid);
      double cmin = 1e300, cmax = 0, csum = 0;
      for (size_t i = 0; i < st.size(); ++i) {
        const PeakStat& x = st[i];
        if (x.iters != unsigned(nblk)) ++ck.iters_bad;
        const double cyc = double(x.t1 - x.t0);
        cmin = std::min(cmin, cyc), cmax = std::max(cmax, cyc), csum += cyc;
        SmAcc& a = per_sm[x.smid];  // one common window per SM: all its ops over (latest end - earliest start) of its warpgroups
        a.ops += double(nblk) * kPeakWgmmaPerBlock * ops_wgmma;
        a.t0 = std::min(a.t0, x.t0), a.t1 = std::max(a.t1, x.t1);
        smids[i / NWG] = x.smid;
      }
      ck.placed = placed(smids.data());
      ck.sms_used = (long long)per_sm.size();
      ck.rate_min = 1e300;
      for (const auto& kv : per_sm) {
        const double r = kv.second.ops / double(std::max(1ull, kv.second.t1 - kv.second.t0));
        ck.rate_mean += r / double(per_sm.size()), ck.rate_min = std::min(ck.rate_min, r);
      }
      ck.cycles_min = cmin, ck.cycles_max = cmax, ck.cycles_mean = csum / double(st.size());
      return ck;
    };
    auto poison_launch = [&](int nblk) {
      CU(cudaMemset(dOut.p, 0xEE, dOut.bytes));
      CU(cudaMemset(dStat.p, 0, dStat.bytes));
      launch(nblk, true);
      CU(cudaDeviceSynchronize());
      CU(cudaGetLastError());
    };
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    struct Guard { cudaEvent_t a, b; ~Guard() { cudaEventDestroy(a); cudaEventDestroy(b); } } guard{e0, e1};
    auto time_launches = [&](int nblk, int n, bool log) {  // seconds of n back-to-back launches; the outputs are poisoned first (outside the event pair), so a launch that wrote nothing shows
      CU(cudaMemsetAsync(dOut.p, 0xEE, dOut.bytes, nullptr));
      CU(cudaMemsetAsync(dStat.p, 0, dStat.bytes, nullptr));
      CU(cudaEventRecord(e0, nullptr));
      for (int i = 0; i < n; ++i) launch(nblk, log);
      CU(cudaEventRecord(e1, nullptr));
      CU(cudaEventSynchronize(e1));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      return double(ms) * 1e-3;
    };

    // ---- warm-up: short launches for 0.3 s so the clocks are up, then the block count that makes one launch last about 0.25 s
    const int nblk0 = 1000;
    {
      const auto t0 = std::chrono::steady_clock::now();
      do { time_launches(nblk0, 8, false); } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3);
    }
    const double per_block = time_launches(nblk0, 8, false) / 8.0 / nblk0;
    const int nblk = std::max(2000, int(0.25 / std::max(per_block, 1e-9)));
    const double ops_launch = ops_wgmma * double(nblk) * kPeakWgmmaPerBlock * double(grid) * NWG;
    base.i("iterations_blocks", nblk).i("wgmma_per_wg", (long long)nblk * kPeakWgmmaPerBlock).i("wgmma_total_per_launch", (long long)nblk * kPeakWgmmaPerBlock * grid * NWG);
    base.n("ops_per_wgmma", ops_wgmma).n("ops_per_launch", ops_launch);
    const int reps = std::max(5, int(std::ceil(c.opt.peak_seconds / 0.25)));

    // ---- timing, straight after the warm-up (the clocks are up): `reps` launches of about 0.25 s each, an event pair around each
    std::vector<Rec> recs;
    std::vector<double> ts;
    PeakCheck ck, cstat;  // cstat: clock64 statistics and a full comparison of the last launch of the timed block / of the sustained block; ck: the first of them
    double med = 0, tops = 0;
    if (!t.only_sustain) {
      double total = 0;
      Sampler sm;
      sm.start(c.opt.nvml ? &c.nvml : nullptr);
      for (int r = 0; r < reps; ++r) {
        const double s = time_launches(nblk, 1, true);
        ts.push_back(s);
        total += s;
      }
      sm.end();
      CU(cudaGetLastError());
      cstat = check(nblk);
      ck = cstat;
      med = median(ts);
      const double mn = *std::min_element(ts.begin(), ts.end()), mx = *std::max_element(ts.begin(), ts.end());
      tops = ops_launch / med / 1e12;
      Rec r = base;
      r.s("mode", "peak").i("launches_per_rep", 1).i("reps", (long long)ts.size()).n("seconds", total);
      r.n("tops", tops).n("tops_best", ops_launch / mn / 1e12).n("tops_worst", ops_launch / mx / 1e12);
      r.n("median_s", med).n("min_s", mn).n("max_s", mx).n("median_ms", med * 1e3).n("min_ms", mn * 1e3).n("max_ms", mx * 1e3).n("spread_pct", 100.0 * (mx - mn) / med);
      sm.put(r);
      // clock-normalised: ops per clock per SM, from the SM's own clock64 (no sampling) and from NVML's clock; TOPS per GHz
      r.s("ceiling_kind", "cold_burst_upper_bound");  // a cold ~1 s burst: an upper bound without memory system, TMA or epilogue, never an "achievable" rate; the sustain records are the sustained ones
      r.n("ops_clk_sm_clock64", cstat.rate_mean).n("ops_clk_sm_clock64_min", cstat.rate_min).n("frac_of_datasheet_ops_clk", cstat.rate_mean / kDatasheetOpsClkSm);
      r.n("sm_mhz_clock64", cstat.cycles_mean / med / 1e6).n("cycles_mean", cstat.cycles_mean).n("cycles_min", cstat.cycles_min).n("cycles_max", cstat.cycles_max);
      r.n("cycles_max_over_min", cstat.cycles_max / std::max(cstat.cycles_min, 1.0));  // ops_clk_sm_clock64 is only valid when this is about 1
      if (sm.n > 0) {
        const double mhz = sm.sum_mhz / double(sm.n);
        r.n("tops_per_ghz_nvml", tops / (mhz / 1e3)).n("ops_clk_sm_nvml", tops * 1e12 / (double(cstat.sms_used) * mhz * 1e6));
        r.n("ceiling_tops_at_nvml_clock", double(c.sms) * kDatasheetOpsClkSm * mhz * 1e6 / 1e12);
      } else {
        r.null("tops_per_ghz_nvml").null("ops_clk_sm_nvml").null("ceiling_tops_at_nvml_clock");
      }
      r.n("ceiling_tops_at_max_clock", double(c.sms) * kDatasheetOpsClkSm * c.clock_khz * 1e3 / 1e12);
      r.n("datasheet_tops", kDatasheetTops).n("datasheet_ref_clock_ghz", kDatasheetClockGhz).i("datasheet_ref_sms", 132);
      recs.push_back(r);
    }
    // ---- sustained block (--sustain): single launches for the given seconds; tops over all of it and over its first and last second
    if (t.sustain > 0) {
      std::vector<double> dt;
      double total = 0;
      Sampler sm;
      sm.start(c.opt.nvml ? &c.nvml : nullptr);
      while (total < t.sustain) {
        const double s = time_launches(nblk, 1, true);
        dt.push_back(s);
        total += s;
      }
      sm.end();
      CU(cudaGetLastError());
      cstat = check(nblk);
      if (t.only_sustain) ck = cstat;
      auto window = [&](bool first) {
        double tt = 0;
        long long n = 0;
        if (first) for (size_t k = 0; k < dt.size() && tt < 1.0; ++k) tt += dt[k], ++n;
        else for (size_t k = dt.size(); k-- > 0 && tt < 1.0;) tt += dt[k], ++n;
        return ops_launch * double(n) / tt / 1e12;
      };
      Rec r = base;
      const double stops = ops_launch * double(dt.size()) / total / 1e12;
      r.s("mode", "sustain").i("launches_per_chunk", 1).i("chunks", (long long)dt.size()).n("seconds", total);
      r.n("tops", stops).n("tops_first_1s", window(true)).n("tops_last_1s", window(false));
      sm.put(r);
      r.s("ceiling_kind", "sustained");  // with seconds, temp_c (last sample), sm_mhz_last / power_w_last (last sample) and sm_mhz_min/mean: quote them next to the number
      r.n("ops_clk_sm_clock64_last", cstat.rate_mean).n("ops_clk_sm_clock64_min_last", cstat.rate_min).n("sm_mhz_clock64_last", cstat.cycles_mean / dt.back() / 1e6);
      r.n("cycles_min_last", cstat.cycles_min).n("cycles_max_last", cstat.cycles_max).n("cycles_max_over_min_last", cstat.cycles_max / std::max(cstat.cycles_min, 1.0));
      if (sm.n > 0) {
        r.n("tops_per_ghz_nvml", stops / (sm.sum_mhz / double(sm.n) / 1e3)).i("sm_mhz_last", sm.last_mhz).n("power_w_last", sm.last_w);
        r.n("ceiling_tops_at_nvml_clock", double(c.sms) * kDatasheetOpsClkSm * (sm.sum_mhz / double(sm.n)) * 1e6 / 1e12);
      } else {
        r.null("tops_per_ghz_nvml").null("sm_mhz_last").null("power_w_last").null("ceiling_tops_at_nvml_clock");
      }
      r.n("ceiling_tops_at_max_clock", double(c.sms) * kDatasheetOpsClkSm * c.clock_khz * 1e3 / 1e12).n("datasheet_tops", kDatasheetTops).n("datasheet_ref_clock_ghz", kDatasheetClockGhz).i("datasheet_ref_sms", 132);
      recs.push_back(r);
    }

    // ---- exactness after everything: a fresh poisoned launch (the last launch of the timed and of the sustained block were compared above)
    poison_launch(nblk);
    const PeakCheck ck2 = check(nblk);
    const bool before = ck.bad == 0 && ck.iters_bad == 0, after = ck2.bad == 0 && ck2.iters_bad == 0 && cstat.bad == 0 && cstat.iters_bad == 0;
    const bool exact = before && after;  // timed block's last launch, sustained block's last launch, and a fresh launch afterwards: all exact
    if (!exact) ++c.n_inexact;
    // placement of EVERY logged launch (timed, sustained, final): exactly `ctas` CTAs on each of the card's SMs
    bool all_placed = ck2.placed && cstat.placed;
    {
      std::vector<unsigned> hl(size_t(logged) * grid);
      if (logged) CU(cudaMemcpy(hl.data(), dLog.p, hl.size() * 4, cudaMemcpyDeviceToHost));
      for (int i = 0; i < logged; ++i) all_placed = all_placed && placed(&hl[size_t(i) * grid]);
    }
    for (Rec& r : recs) {
      r.b("exact", exact).i("check_elements", ck.checked).i("ref_mismatches", ck.bad).i("iters_mismatches", ck.iters_bad);
      r.b("exact_after_timing", after).i("ref_mismatches_after", ck2.bad).i("check_elements_after", ck2.checked);
      r.b("placement_ok", all_placed).i("placement_launches_checked", logged).i("placement_launches_total", launches_total).i("sms_used", cstat.sms_used);
      r.b("exact_check_trivial", data == 0);  // zero operands: every accumulator must be 0; only the iteration counter and the write-out are then tested
      r.s("exact_def", "accumulator[cta][wg][row][col] == nblk * sum_j (A_s(j) . B_s(j)^T) mod 2^32 (int32 wraparound), every element of every CTA and warpgroup");
      if (ck.bad) r.i("first_bad_cta", ck.cta).i("first_bad_wg", ck.wg).i("first_bad_thread", ck.thr).i("first_bad_reg", ck.reg).i("first_bad_want", ck.want).i("first_bad_got", ck.got);
      r.s("status", "ok");
      emit_line(r);
    }
    if (!t.only_sustain && exact) c.peak_tops[&e] = tops;
    if (!all_placed) std::fprintf(stderr, "  warning: placement is not exactly %d CTA(s) on each of the %d SMs (%lld distinct SMs in the last launch): the rate is wrong\n", ctas, c.sms, cstat.sms_used);
    if (cstat.cycles_max / std::max(cstat.cycles_min, 1.0) > 1.05) std::fprintf(stderr, "  note: warpgroup loop cycles max/min = %.3f: ops/clk/SM from clock64 is not valid for this row\n", cstat.cycles_max / std::max(cstat.cycles_min, 1.0));
    std::fprintf(stderr, "  %-26s exact=%d  %.0f TOPS  %.0f ops/clk/SM (datasheet %d)  %.0f MHz (clock64)  regs %d local %zu\n", e.name, int(exact), tops, cstat.rate_mean,
                 kDatasheetOpsClkSm, cstat.cycles_mean / std::max(med, 1e-9) / 1e6, fa.numRegs, (size_t)fa.localSizeBytes);
  } catch (const Skip& s) {
    emit_status(std::string("skipped: ") + s.what());
  } catch (const std::exception& ex) {
    cudaGetLastError();
    if (cudaDeviceSynchronize() != cudaSuccess) {
      emit_status(std::string("error (sticky): ") + ex.what());
      throw Sticky(ex.what());
    }
    emit_status(std::string("error: ") + ex.what());
  }
}
template <int N, int NWG, int ST, int CTAS, int DATA>
void run_peak(Ctx& c, const Entry& e, const Task& t) { run_peak_impl<N, NWG, ST>(c, e, t, CTAS, DATA); }

// ================================================================== DATA: the configuration table and the defaults
// Everything that is a choice (which kernels, which shapes, which scheduler knobs) is here. To add an architecture (Blackwell sm_100 / sm_120
// with tcgen05, another Hopper tile) add rows with its cc_major and a Variant-like type whose Kernel is a CUTLASS GemmUniversal with K4's epilogue,
// and guard them with the architecture's own compile condition; the harness runs the rows whose cc_major equals the device's and exits 3 if there are none.
// Rows are in priority order: if a run is cut off, the first rows have reported. Names: f_/g_ = fast / generic epilogue, m_ = the product's mma.sync K4,
// TILE_MxNxK, CLUSTER_MxN, coop/ping.
using T128x256 = Shape<Int<128>, Int<256>, Int<128>>;
using T128x256k64 = Shape<Int<128>, Int<256>, Int<64>>;
using T128x128 = Shape<Int<128>, Int<128>, Int<128>>;
using T256x128 = Shape<Int<256>, Int<128>, Int<128>>;
using C11 = Shape<_1, _1, _1>;
using C12 = Shape<_1, _2, _1>;
using C21 = Shape<_2, _1, _1>;
using C41 = Shape<_4, _1, _1>;
//          name                      cc  kind        quick  sweep  last   tile                        cluster
#define IMD_ROW(NAME, CC, KIND, QUICK, SWEEP, LAST, T, C, S)                                                                  \
  {NAME, CC, KIND, QUICK, SWEEP, LAST, int(cute::get<0>(T{})), int(cute::get<1>(T{})), int(cute::get<0>(C{})), int(cute::get<1>(C{})), \
   smem_of<VT<T, C, S, (KIND == Kind::Fast)>>(), &run_variant<VT<T, C, S, (KIND == Kind::Fast)>>}
// p_ rows (synthetic wgmma peak): N = instruction N (256 / 128), NWG = consumer warpgroups per CTA, ST = tile stages in shared memory, CTAS = CTAs per SM,
// DATA = 0 zero / 1 small random / 2 full-range random / 3 full-range random with four different tiles rotating (ST = 4)
#define IMD_PROW(NAME, N, NWG, ST, CTAS, DATA, QUICK)                                                                               \
  {NAME, 9, Kind::Peak, QUICK, false, false, 64 * NWG, N, 1, 1, peak_smem<N, NWG, ST, CTAS>(), &run_peak<N, NWG, ST, CTAS, DATA>, DATA}
const Entry kEntries[] = {
    // the peaks come first in the work order (cheap, about 3 s each, and they say what the ceiling is)
    IMD_PROW("p_n256_wg2_full", 256, 2, 1, 1, 2, false),   // the stock coop kernel's consumer structure: 2 warpgroups of 64 x 256 on one CTA per SM
    IMD_PROW("p_n256_wg2_zero", 256, 2, 1, 1, 0, false),
    IMD_PROW("p_n256_wg2_small", 256, 2, 1, 1, 1, false),
    IMD_PROW("p_n256_wg2_full4", 256, 2, 4, 1, 3, false),
    IMD_PROW("p_n256_wg3_full", 256, 3, 1, 1, 2, false),
    IMD_PROW("p_n256_wg3_zero", 256, 3, 1, 1, 0, false),
    IMD_PROW("p_n256_wg3_small", 256, 3, 1, 1, 1, false),
    IMD_PROW("p_n256_wg1_full", 256, 1, 1, 1, 2, false),
    IMD_PROW("p_n256_wg1_zero", 256, 1, 1, 1, 0, false),
    IMD_PROW("p_n256_wg1_small", 256, 1, 1, 1, 1, false),
    IMD_PROW("p_n256_wg1x2cta_full", 256, 1, 1, 2, 2, false),
    IMD_PROW("p_n256_wg1x2cta_zero", 256, 1, 1, 2, 0, false),
    IMD_PROW("p_n256_wg1x2cta_small", 256, 1, 1, 2, 1, false),
    IMD_PROW("p_n128_wg4_full", 128, 4, 1, 1, 2, false),
    IMD_PROW("p_n128_wg4_zero", 128, 4, 1, 1, 0, false),
    IMD_PROW("p_n128_wg4_small", 128, 4, 1, 1, 1, false),
    IMD_PROW("p_n128_wg4_full4", 128, 4, 4, 1, 3, false),
    IMD_PROW("p_n128_wg2_full", 128, 2, 1, 1, 2, false),
    IMD_PROW("p_n128_wg2_zero", 128, 2, 1, 1, 0, false),
    IMD_PROW("p_n128_wg2_small", 128, 2, 1, 1, 1, false),
    {"m_128x256x64_mma_sync", 9, Kind::Mma, true, false, false, 128, 256, 1, 1, 72 * 1024, &run_mma},
    IMD_ROW("f_256x128x128_2x1_coop", 9, Kind::Fast, true, true, false, T256x128, C21, Coop),
    IMD_ROW("f_128x128x128_2x1_coop", 9, Kind::Fast, true, true, false, T128x128, C21, Coop),
    IMD_ROW("f_128x256x128_1x2_coop", 9, Kind::Fast, true, true, false, T128x256, C12, Coop),
    IMD_ROW("f_128x256x128_1x1_coop", 9, Kind::Fast, true, true, false, T128x256, C11, Coop),
    IMD_ROW("f_128x128x128_1x1_ping", 9, Kind::Fast, true, true, false, T128x128, C11, Ping),
    IMD_ROW("f_128x256x128_2x1_coop", 9, Kind::Fast, false, true, false, T128x256, C21, Coop),
    IMD_ROW("f_128x256x64_1x2_coop", 9, Kind::Fast, false, false, false, T128x256k64, C12, Coop),
    IMD_ROW("g_256x128x128_2x1_coop", 9, Kind::Generic, false, false, false, T256x128, C21, Coop),
    IMD_ROW("g_128x128x128_2x1_coop", 9, Kind::Generic, false, false, false, T128x128, C21, Coop),
    IMD_ROW("g_128x256x128_1x2_coop", 9, Kind::Generic, false, false, false, T128x256, C12, Coop),
    IMD_ROW("g_128x256x128_1x1_coop", 9, Kind::Generic, false, false, false, T128x256, C11, Coop),
    IMD_ROW("g_128x128x128_1x1_ping", 9, Kind::Generic, false, false, false, T128x128, C11, Ping),
    IMD_ROW("f_128x256x128_4x1_coop", 9, Kind::Fast, false, false, true, T128x256, C41, Coop),  // last: untested cluster shape
};
#undef IMD_ROW
#undef IMD_PROW
struct Defaults {
  const char* shapes;        // m = n = k, or MxNxK
  const char* batches;       // the first is the primary one (the N moduli of the emulation); the others are references
  const char* sweep_shapes;  // shapes of the scheduler sweep ("none" = no sweep)
  const char* quick_shapes;
  const char* quick_batches;
};
constexpr Defaults kDefaults = {"4096,8192", "14,4,1", "8192", "4096", "14"};
// scheduler sweep: max_swizzle_size x raster order (1 = AlongM, 2 = AlongN); the default scheduler (swizzle 1, heuristic raster) runs in the main phase
constexpr int kSweepSwizzle[] = {1, 2, 4, 8};
constexpr int kSweepRaster[] = {1, 2};
constexpr int kRefSlot0 = 1;  // reference batches (L below the primary) start at modulus slot 1: moduli[0] = 256 makes the epilogue a mask, and L = 1 would measure only that

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> o;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');)
    if (!t.empty()) o.push_back(t);
  return o;
}
bool parse_dims(const std::string& t, Dims* d) {
  long long m, n, k;
  if (std::sscanf(t.c_str(), "%lldx%lldx%lld", &m, &n, &k) == 3) { *d = {m, n, k}; return true; }
  if (std::sscanf(t.c_str(), "%lld", &m) == 1 && m > 0) { *d = {m, m, m}; return true; }
  return false;
}
// entries of the device's architecture matching a spec: the groups all / f / g / m / quick / sweep (exact words), else a name or a substring of one
std::vector<const Entry*> select(const std::string& spec, int cc_major, std::vector<std::string>* unmatched) {
  static const std::set<std::string> groups = {"all", "f", "g", "m", "p", "quick", "sweep"};
  std::vector<const Entry*> out;
  for (const std::string& tok : split(spec)) {
    bool any = false;
    for (const Entry& e : kEntries) {
      if (e.cc_major != cc_major) continue;
      bool hit;
      if (groups.count(tok))
        hit = (tok == "all" && e.kind != Kind::Peak) || (tok == "f" && e.kind == Kind::Fast) || (tok == "g" && e.kind == Kind::Generic) || (tok == "m" && e.kind == Kind::Mma) || (tok == "p" && e.kind == Kind::Peak) ||
              (tok == "quick" && e.quick) || (tok == "sweep" && e.sweep);
      else
        hit = std::string(e.name).find(tok) != std::string::npos;
      if (hit) {
        any = true;
        if (std::find(out.begin(), out.end(), &e) == out.end()) out.push_back(&e);
      }
    }
    if (!any && unmatched) unmatched->push_back(tok);
  }
  return out;
}

}  // namespace rung1

using namespace rung1;

int main(int argc, char** argv) {
  static const std::set<std::string> flags = {"quick", "list", "plan", "no-nvml", "skip-last"};
  static const std::set<std::string> valued = {"variants", "shapes", "batches", "reps", "warmup", "loop", "max-seconds", "samples",
                                               "sweep", "sweep-variants", "clusters", "sustain", "peak-seconds"};
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--", 2) != 0) { std::fprintf(stderr, "unexpected argument %s (usage: see the header of src/k4/hopper/rung1.cu)\n", argv[i]); return 2; }
    const std::string key = argv[i] + 2;
    if (flags.count(key)) { a[key] = "1"; continue; }
    if (!valued.count(key)) { std::fprintf(stderr, "unknown option --%s (usage: see the header of src/k4/hopper/rung1.cu)\n", key.c_str()); return 2; }
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
    a[key] = argv[++i];
  }
  Ctx c;
  Opt& o = c.opt;
  const char* q = std::getenv("IMD_RUNG1_QUICK");
  o.quick = a.count("quick") || (q && *q && std::strcmp(q, "0") != 0);
  o.nvml = !a.count("no-nvml");
  o.skip_last = a.count("skip-last") > 0;
  if (o.quick) o.reps = 10, o.samples = 128;
  if (a.count("reps")) o.reps = std::max(1, std::atoi(a["reps"].c_str()));
  if (a.count("warmup")) o.warmup = std::max(1, std::atoi(a["warmup"].c_str()));
  if (a.count("loop")) o.loop = std::max(1, std::atoi(a["loop"].c_str()));
  if (a.count("max-seconds")) o.max_seconds = std::atof(a["max-seconds"].c_str());
  if (a.count("samples")) o.samples = std::atoll(a["samples"].c_str());
  if (a.count("sustain")) o.sustain = std::atof(a["sustain"].c_str());
  if (a.count("peak-seconds")) o.peak_seconds = std::max(0.25, std::atof(a["peak-seconds"].c_str()));
  if (a.count("clusters")) o.clusters = a["clusters"];
  if (o.clusters != "queried" && o.clusters != "heuristic" && o.clusters != "both") { std::fprintf(stderr, "--clusters is queried, heuristic or both\n"); return 2; }
  if (a.count("list")) {
    for (const Entry& e : kEntries) std::printf("%s cc%d smem_bytes=%lld tile=%dx%d cluster=%dx%d%s\n", e.name, e.cc_major, e.smem_bytes, e.tile_m, e.tile_n, e.cl_m, e.cl_n, e.last ? " last" : "");
    return 0;
  }

  const bool plan = a.count("plan") > 0;  // print the work list (as for a Hopper card) and exit; no GPU needed
  cudaDeviceProp prop{};
  int dev = 0;
  if (plan) {
    std::strcpy(prop.name, "plan (no device)");
    prop.major = 9, prop.multiProcessorCount = 132;
  } else if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    std::fprintf(stderr, "rung 1: no CUDA device\n");
    return 3;
  }
  std::vector<std::string> unmatched;
  const std::vector<const Entry*> vars = select(a.count("variants") ? a["variants"] : (o.quick ? "quick" : "all"), prop.major, &unmatched);
  if (vars.empty()) {
    std::fprintf(stderr, "rung 1: no configuration for this device (%s, sm_%d%d)%s: the table kEntries has rows for compute capability major 9 only\n",
                 prop.name, prop.major, prop.minor, unmatched.empty() ? "" : " / --variants matches nothing");
    return 3;
  }
  for (const std::string& u : unmatched) std::fprintf(stderr, "warning: --variants entry '%s' matches no row\n", u.c_str());

  c.gpu = prop.name;
  c.sms = prop.multiProcessorCount;
  c.cc_major = prop.major, c.cc_minor = prop.minor;
  if (!plan) {
    cudaDeviceGetAttribute(&c.clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&c.mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  }
  if (const char* cm = std::getenv("BENCH_COMMIT")) c.commit = cm;
  if (o.nvml && !plan) c.nvml.open(dev);

  // shapes, batches, sweep
  std::vector<Dims> shapes;
  for (const std::string& t : split(a.count("shapes") ? a["shapes"] : (o.quick ? kDefaults.quick_shapes : kDefaults.shapes))) {
    Dims d;
    if (!parse_dims(t, &d)) { std::fprintf(stderr, "bad shape %s\n", t.c_str()); return 2; }
    shapes.push_back(d);
  }
  std::vector<int> batches;
  for (const std::string& t : split(a.count("batches") ? a["batches"] : (o.quick ? kDefaults.quick_batches : kDefaults.batches))) batches.push_back(std::max(1, std::atoi(t.c_str())));
  if (shapes.empty() || batches.empty()) { std::fprintf(stderr, "empty --shapes / --batches\n"); return 2; }
  c.lmax = *std::max_element(batches.begin(), batches.end());
  c.primary = batches[0];
  if (c.lmax + kRefSlot0 > imd::constants::kNMax) { std::fprintf(stderr, "batch above the modulus table\n"); return 2; }
  std::vector<Dims> sweep_shapes;
  {
    const std::string s = a.count("sweep") ? a["sweep"] : (o.quick ? "none" : kDefaults.sweep_shapes);
    if (s != "none")
      for (const std::string& t : split(s)) {
        Dims d;
        if (!parse_dims(t, &d)) { std::fprintf(stderr, "bad --sweep %s\n", t.c_str()); return 2; }
        sweep_shapes.push_back(d);
      }
  }
  std::vector<const Entry*> sweep_vars;
  if (a.count("sweep-variants"))
    sweep_vars = select(a["sweep-variants"], prop.major, nullptr);
  else
    for (const Entry* e : vars)
      if (e->sweep) sweep_vars.push_back(e);

  // ---- the work list, in priority order (see the header)
  const int primary = batches[0];
  const bool cl_queried = o.clusters != "heuristic", cl_heur = o.clusters != "queried";
  auto is_cluster = [](const Entry* e) { return e->cl_m * e->cl_n > 1; };
  struct Work { const Entry* e; Task t; int peak_sustain = 0; };  // e == nullptr: a sustain item (the GEMM one, or with peak_sustain = 1 + data kind the p_ one of that data kind)
  std::vector<Work> work;
  auto add = [&](const Entry* e, Dims d, int L, Sched sc, int occ) {
    Task t{d, L, sc, occ, L < primary ? kRefSlot0 : 0, 0};
    work.push_back({e, t});
  };
  // the rows of one pass at a shape: every cluster row once per requested cluster mode
  auto pass = [&](const Entry* e, Dims d, int L, Sched sc) {
    if (!is_cluster(e)) { add(e, d, L, sc, 0); return; }
    if (cl_queried) add(e, d, L, sc, 0);
    if (cl_heur) add(e, d, L, sc, 1);
  };
  bool any_peak = false;
  for (const Entry* e : vars)  // (0) the synthetic wgmma peak rows (no shape, no batch), then with --sustain the sustained block of the best full-range one
    if (e->kind == Kind::Peak) { any_peak = true; work.push_back({e, Task{Dims{0, 0, 0}, 0, Sched{}, 0, 0, 0}}); }
  if (any_peak && o.sustain > 0)  // one sustained block per data kind (small, full, full4): the best exact row of that kind by its timed rate
    for (int dk = 1; dk <= 3; ++dk) work.push_back({nullptr, Task{Dims{0, 0, 0}, 0, Sched{}, 0, 0, o.sustain, true}, 1 + dk});
  for (const Dims& d : shapes)  // (1) m_ and f_ rows, default scheduler, primary batch
    for (const Entry* e : vars)
      if (e->kind != Kind::Generic && e->kind != Kind::Peak && !e->last) pass(e, d, primary, Sched{});
  for (const Dims& d : sweep_shapes)  // (2) the sweep, minus rows that resolve to the default's (raster, swizzle) or to an earlier sweep point
    for (const Entry* e : sweep_vars) {
      if (e->kind == Kind::Mma || e->kind == Kind::Peak || e->last) continue;
      const Resolved def = resolve(*e, Task{d, primary, Sched{}, 0, 0, 0});
      std::set<std::pair<int, int>> seen{{def.raster, def.swizzle}};
      for (int rst : kSweepRaster)
        for (int sw : kSweepSwizzle) {
          const Resolved r = resolve(*e, Task{d, primary, Sched{sw, rst}, 0, 0, 0});
          if (!seen.insert({r.raster, r.swizzle}).second) continue;
          pass(e, d, primary, Sched{sw, rst});
        }
    }
  bool any_fast = false;
  for (const Entry* e : vars) any_fast = any_fast || (e->kind == Kind::Fast && !e->last);
  if (o.sustain > 0 && any_fast) work.push_back({nullptr, Task{shapes.back(), primary, Sched{}, 0, 0, o.sustain}});  // (3)
  for (const Dims& d : shapes)  // (4) g_ rows
    for (const Entry* e : vars)
      if (e->kind == Kind::Generic && !e->last) pass(e, d, primary, Sched{});
  for (size_t bi = 1; bi < batches.size(); ++bi)  // (5) reference batches
    for (const Dims& d : shapes)
      for (const Entry* e : vars)
        if (!e->last && e->kind != Kind::Peak) pass(e, d, batches[bi], Sched{});
  if (!o.skip_last)  // (6) the riskiest rows
    for (const Dims& d : shapes)
      for (const Entry* e : vars)
        if (e->last) pass(e, d, primary, Sched{});

  if (plan) {
    for (const Work& w : work) {
      const Entry* e = w.e;
      if (!e && w.peak_sustain) { std::printf("sustain %gs on the best p_ row with data %s\n", w.t.sustain, peak_data_name(w.peak_sustain - 1)); continue; }
      if (!e) { std::printf("sustain %gs on the best f_ row at %lldx%lldx%lld\n", w.t.sustain, w.t.d.m, w.t.d.n, w.t.d.k); continue; }
      if (e->kind == Kind::Peak) { std::printf("%-26s synthetic wgmma peak: N=%d, %d warpgroup(s) per CTA, data %s, smem %lld\n", e->name, e->tile_n, e->tile_m / 64, peak_data_name(e->pk_data), e->smem_bytes); continue; }
      const Resolved r = resolve(*e, w.t);
      std::printf("%-26s %5lldx%lldx%lld L=%-2d slot0=%d sw=%d %-9s -> %s sw%d%s\n", e->name, w.t.d.m, w.t.d.n, w.t.d.k, w.t.L, w.t.slot0, w.t.sc.swizzle,
                  raster_name(w.t.sc.raster), raster_name(r.raster), r.swizzle, e->cl_m * e->cl_n > 1 ? (w.t.occ ? "  clusters=heuristic" : "  clusters=queried") : "");
    }
    std::printf("%zu tasks\n", work.size());
    return 0;
  }

  // ---- the meta record
  {
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    char host[256] = {};
    gethostname(host, sizeof host - 1);
    std::string cmd;
    for (int i = 1; i < argc; ++i) cmd += (i > 1 ? " " : "") + std::string(argv[i]);
    Rec m;
    m.s("method", "rung1_meta").s("harness", "rung1/3").s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).i("sm_count", c.sms);
    m.i("clock_khz_max", c.clock_khz).i("mem_clock_khz", c.mem_clock_khz).i("l2_bytes", prop.l2CacheSize);
    m.i("cuda_driver_version", drv).i("cuda_runtime_version", rt).s("hostname", host);
    if (c.nvml.ok) m.s("nvml_driver_version", c.nvml.driver); else m.null("nvml_driver_version");
    const char* sj = std::getenv("SLURM_JOB_ID");
    if (sj) m.s("slurm_job_id", sj); else m.null("slurm_job_id");
    const char* nv = std::getenv("IMD_NVCC_VERSION");
    if (nv) m.s("nvcc_version", nv); else m.null("nvcc_version");
    if (!c.commit.empty()) m.s("commit", c.commit); else m.null("commit");
    m.b("nvml", c.nvml.ok).s("argv", cmd).i("rows_in_table", (long long)(sizeof kEntries / sizeof kEntries[0])).i("rows_selected", (long long)vars.size());
    m.b("ndebug",
#ifdef NDEBUG
        true
#else
        false
#endif
    );
    emit_line(m);
  }
  std::fprintf(stderr, "imd_rung1: %s (%d SMs, max SM clock %d MHz), NVML %s, %zu rows, ndebug=%d\n", c.gpu.c_str(), c.sms, c.clock_khz / 1000,
               c.nvml.ok ? "on" : "off", vars.size(),
#ifdef NDEBUG
               1
#else
               0
#endif
  );
#ifndef NDEBUG
  std::fprintf(stderr, "WARNING: built without -DNDEBUG: CUTLASS's assert leaves a call in the main loop and ptxas serialises the wgmma (C7510). These numbers are not the product's.\n");
#endif

  int rc = 0;
  try {
    for (const Work& w : work) {
      const Entry* e = w.e;
      Task t = w.t;
      if (!e && w.peak_sustain) {  // the peak sustain item: the best exact p_ row of this data kind by its timed rate
        double best = 0;
        const int dk = w.peak_sustain - 1;
        for (const Entry* f : vars)
          if (f->kind == Kind::Peak && f->pk_data == dk) {
            auto it = c.peak_tops.find(f);
            if (it != c.peak_tops.end() && it->second > best) best = it->second, e = f;
          }
        if (!e) {  // say so in the output, not only on stderr
          const std::string why = std::string("skipped: peak sustain found no exact p_ row with data ") + peak_data_name(dk);
          std::fprintf(stderr, "%s\n", why.c_str());
          Rec r;
          r.s("method", "rung1").s("variant", std::string("peak_sustain_") + peak_data_name(dk)).s("kind", "wgmma_peak").s("op", "peak").s("mode", "sustain").s("data", peak_data_name(dk));
          r.s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).null("exact").s("status", why);
          emit_line(r);
          ++c.n_nonok;
          continue;
        }
        t.only_sustain = true;
      } else if (!e) {  // the sustain item: the best f_ row at the last shape by its loop-mode rate
        double best = 0;
        for (const Entry* f : vars)
          if (f->kind == Kind::Fast && !f->last) {
            auto it = c.loop_tops.find({f, t.d.m});
            if (it != c.loop_tops.end() && it->second > best) best = it->second, e = f;
          }
        if (!e) { std::fprintf(stderr, "sustain: no f_ row measured at the last shape, skipped\n"); continue; }
      }
      try {
        e->run(c, *e, t);
      } catch (const Sticky&) {
        throw;
      } catch (const std::exception& ex) {  // an allocation failure for a shape etc.
        cudaGetLastError();
        std::fprintf(stderr, "  %s: %s\n", e->name, ex.what());
        Rec r;
        r.s("method", "rung1").s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).s("variant", e->name).i("m", t.d.m)
            .i("n", t.d.n).i("k", t.d.k).i("batch", t.L).null("exact").s("status", std::string("error: ") + ex.what());
        emit_line(r);
        ++c.n_nonok;
      }
    }
  } catch (const Sticky& ex) {
    std::fprintf(stderr, "rung 1: sticky device error, stopping: %s\n", ex.what());
    rc = 4;
  }
  if (rc == 0 && c.n_inexact) rc = 1;
  if (rc == 0 && c.n_nonok) rc = 5;
  std::fprintf(stderr, "rung 1: %lld rows run, %lld not exact, %lld with status error / skipped -> exit %d (0 clean, 1 not exact, 4 device error, 5 error/skipped rows)\n",
               c.n_rows, c.n_inexact, c.n_nonok, rc);
  return rc;
}
