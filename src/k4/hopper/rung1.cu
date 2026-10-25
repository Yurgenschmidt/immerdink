// Rung 1 of the Hopper K4 ladder (issue 22, docs/NOTES-22.md): the TIMING harness for our own wgmma K4 variants.
// Rung 0 (rung0.cu) checks exactness on a tiny problem; this program measures speed at the scale K4 runs: m = n = k in {4096, 8192},
// L = 14 independent GEMMs in one launch (the N moduli of the emulation; batch z uses modulus slot0 + z, b = 1), int8 residue operands
// A[z][m][k], B[z][n][k] (k contiguous), K4's modular epilogue (K4ModEpilogue: mod p -> u8 plane z of W), store mode, and, for reference,
// L = 4 and L = 1. Built for sm_90a only (tools/build_rung1.sh, one nvcc call, -O2 -DNDEBUG). On a device whose compute capability has no row in the table it exits 3 and does nothing.
//
// Why: the stock CUTLASS wgmma ceilings (bench/ceiling) were built with CUTLASS's direct-store epilogue (256 scalar stores per thread) and
// default scheduler arguments, so they probably under-measure what a wgmma K4 can do. The theory (immerdink-eval/findings/offline-theory-
// 20261025.md section 3) says our register epilogue (HopperEpilogue, 16-byte stores) should reach 1260-1456 TOPS at 4096^3, L = 14, against
// 966 for today's mma.sync K4; it is falsified at <= 1100. This program measures that, and the two scheduler arguments the theory blames for
// the 8192^3 collapse of the stock kernels: max_swizzle_size (1, 2, 4, 8) and raster_order (AlongM / AlongN).
//
//   imd_rung1 [--quick] [--list] [--variants A,B,..|all|f|g] [--shapes 4096,8192|MxNxK,..] [--batches 14,4,1] [--reps 12] [--warmup 3]
//             [--loop 8] [--max-seconds 8] [--samples 384] [--sweep 8192|none|SHAPES] [--sweep-variants f|all|LIST] [--no-nvml]
//   env: IMD_RUNG1_QUICK=1 (= --quick), BENCH_COMMIT (copied into every record)
//   --variants entries match by name or substring (f_128x256 selects every f_ 128x256 variant), or the groups all / f / g / quick / sweep; default all.
//   --quick: 4096 only, L = 14 only, the rows flagged quick in the table, 10 reps, 128 samples, no sweep (about a minute).
//   The configurations (kEntries), the default shapes and batches and the sweep grid are data tables in one block ("DATA"); each row carries the
//   compute capability it is for, every JSON record carries the card's name and compute capability (gpu, cc).
//
// Work order (so that a timeout keeps the most valuable lines): (1) default scheduler, first shape, L = first batch, every variant;
// the same for the other shapes; (2) scheduler sweep (swizzle 1/2/4/8 x AlongM/AlongN) for the sweep variants at the --sweep shapes;
// (3) the other batches (4, 1) at every shape, default scheduler. Output is flushed per line.
//
// Output: one JSON line per (variant, shape, batch, scheduler, mode) on stdout (method "rung1"), progress on stderr. mode "single": each
// repetition is one launch between two cudaEvents (includes the launch latency, a few us); mode "loop": each repetition is R back-to-back
// launches between two events, divided by R (amortises the launch; R = max(--loop, enough for 20 ms)). Fields: tops = 2 m n k L / t
// (int8 TOPS, from the median), median/min/max in seconds (_s) and milliseconds (_ms), spread_pct = 100 (max - min) / median, reps.
//
// "exact" and how it is decided (two independent checks, both reported):
//   ref_mismatches   a sample of output bytes (--samples random ones plus corners and tile-edge positions) is compared with a HOST reference:
//                    sum over k of A * B (int64) reduced with the same modular definition ((s mod p) + p) mod p, p = moduli[slot0 + z]
//                    (p = 256 included whenever the table has it among the first L). Operands come from a counter-based hash, so the
//                    host recomputes any element. This is the check against the definition.
//   hash_eq_first    a position-dependent sum-hash of the whole W plane stack equals the hash of the first variant run on the same
//                    (shape, batch); all variants compute the same function, so equality across tile shapes, clusters, schedules and
//                    scheduler arguments is a check of every byte (up to hash collisions) relative to that first variant.
//   exact = (ref_mismatches == 0) && hash_eq_first. W is filled with 0xEE before the checked launch, so an unwritten tile shows up.
//   The spot-check cannot see a rare wrong byte that the hash comparison also misses because the first variant has the same fault: rung 0
//   is the full known-answer test and must have passed on the same card.
//
// Clocks: cudaDevAttrClockRate (max SM clock, kHz) is always printed. If libnvidia-ml.so.1 can be dlopen'ed (no link-time dependency; --no-nvml
// to skip), a host thread polls it every 5 ms during each timed block: sm_mhz_min / mean, power_w_max, temp_c, throttle_reasons (OR of the
// NVML clocks-event reason bits: 0x4 SW power cap, 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake). Without NVML those
// fields are null. This tells power throttling (the rival of the swizzle explanation at 8192) from locality.
//
// Exit status: 0 done and every record exact; 1 some record not exact; 2 usage or CUDA setup error; 3 no Hopper device; 4 a sticky device error
// (illegal address etc.) stopped the run after its record. A hang is the caller's timeout (the build script runs this under `timeout`).
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/kernel_hardware_info.h"
#include "k4/hopper/k4_hopper.cuh"

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

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };
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
  Rec& raw(const std::string& k, const std::string& v) { kv.emplace_back(k, v); return *this; }
  std::string dump() const {
    std::string o = "{";
    for (size_t j = 0; j < kv.size(); ++j) o += (j ? ", " : "") + jstr(kv[j].first) + ": " + kv[j].second;
    return o + "}";
  }
};

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
  bool ok = false;
  void open(int cuda_dev) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return;
    auto sym = [&](const char* n) { return dlsym(lib, n); };
    auto init = reinterpret_cast<Ret (*)()>(sym("nvmlInit_v2"));
    auto by_pci = reinterpret_cast<Ret (*)(const char*, Dev*)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    clock_info = reinterpret_cast<decltype(clock_info)>(sym("nvmlDeviceGetClockInfo"));
    power = reinterpret_cast<decltype(power)>(sym("nvmlDeviceGetPowerUsage"));
    temp = reinterpret_cast<decltype(temp)>(sym("nvmlDeviceGetTemperature"));
    reasons = reinterpret_cast<decltype(reasons)>(sym("nvmlDeviceGetCurrentClocksEventReasons"));
    if (!reasons) reasons = reinterpret_cast<decltype(reasons)>(sym("nvmlDeviceGetCurrentClocksThrottleReasons"));
    char bus[64] = {};
    if (!init || !by_pci || !clock_info || cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) != cudaSuccess) return;
    if (init() != 0 || by_pci(bus, &dev) != 0) return;
    ok = true;
  }
};
// polls the clock and power on a host thread while a timed block runs
struct Sampler {
  Nvml* nv = nullptr;
  std::thread th;
  std::atomic<bool> stop{false};
  double sum_mhz = 0, max_w = 0;
  unsigned min_mhz = ~0u, temp_c = 0;
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
        if (nv->clock_info(nv->dev, 1 /*NVML_CLOCK_SM*/, &mhz) == 0) { min_mhz = std::min(min_mhz, mhz); sum_mhz += mhz; ++n; }
        if (nv->power && nv->power(nv->dev, &mw) == 0) max_w = std::max(max_w, mw * 1e-3);
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
      r.raw("sm_mhz_min", "null").raw("sm_mhz_mean", "null").raw("power_w_max", "null").raw("temp_c", "null").raw("throttle_reasons", "null");
      return;
    }
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", reasons);
    r.i("sm_mhz_min", min_mhz).n("sm_mhz_mean", sum_mhz / double(n)).n("power_w_max", max_w).i("temp_c", temp_c).s("throttle_reasons", b);
  }
};

// ------------------------------------------------------------------ options, context
struct Dims { long long m, n, k; };
bool operator<(const Dims& a, const Dims& b) { return std::tie(a.m, a.n, a.k) < std::tie(b.m, b.n, b.k); }
struct Sched { int swizzle = 1; int raster = 0; };  // raster: 0 heuristic (CUTLASS's default), 1 AlongM, 2 AlongN
const char* raster_name(int r) { return r == 1 ? "AlongM" : r == 2 ? "AlongN" : "heuristic"; }

struct Opt {
  int reps = 12, warmup = 3, loop = 8;
  double max_seconds = 8.0;
  long long samples = 384;
  bool quick = false, nvml = true;
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
struct Ctx {
  Opt opt;
  std::string gpu, commit;
  int sms = 0, clock_khz = 0, mem_clock_khz = 0, cc_major = 0, cc_minor = 0;
  Nvml nvml;
  int lmax = 14;
  std::map<Dims, std::unique_ptr<Fixture>> fx;
  std::map<std::tuple<long long, long long, long long, int>, unsigned long long> first_hash;  // (m, n, k, L) -> hash of the first variant run
  bool any_inexact = false;
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

// ------------------------------------------------------------------ the variants
template <class T, class C, class S, bool F>
struct VT {
  using Tile = T;
  using Cluster = C;
  using Sched = S;
  static constexpr bool kFast = F;
};
using T128x256 = Shape<Int<128>, Int<256>, Int<128>>;
using T128x256k64 = Shape<Int<128>, Int<256>, Int<64>>;
using T128x128 = Shape<Int<128>, Int<128>, Int<128>>;
using T256x128 = Shape<Int<256>, Int<128>, Int<128>>;
using C11 = Shape<_1, _1, _1>;
using C12 = Shape<_1, _2, _1>;
using C21 = Shape<_2, _1, _1>;
using C41 = Shape<_4, _1, _1>;

struct Task {
  Dims d;
  int L;
  Sched sc;
};
// One named configuration of the table kEntries (the only place that knows tile / cluster / schedule choices).
struct Entry {
  const char* name;
  int cc_major;  // the compute capability major this entry is for (9 = Hopper wgmma); the harness runs the entries of the device's major
  bool fast;     // fast (register, 16-byte store) epilogue as opposed to the generic reference one
  bool quick;    // part of --quick
  bool sweep;    // part of the scheduler sweep by default
  long long smem_bytes;  // dynamic shared memory per CTA (Kernel::SharedStorageSize), known at compile time: --list prints it without a GPU
  void (*run)(Ctx&, const char*, const Task&);
};

// time `launch` (one GEMM) in the two modes; fills the record
void time_modes(Ctx& c, const std::function<void()>& launch, const Task& t, const Rec& base, const char* name,
                std::vector<Rec>& out) {
  const double ops = 2.0 * double(t.d.m) * double(t.d.n) * double(t.d.k) * double(t.L);
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  struct Guard { cudaEvent_t a, b; ~Guard() { cudaEventDestroy(a); cudaEventDestroy(b); } } guard{e0, e1};
  for (int w = 0; w < c.opt.warmup; ++w) launch();
  CU(cudaDeviceSynchronize());
  // clocks ramp: keep the card busy for ~0.3 s before the first timed block
  {
    const auto t0 = std::chrono::steady_clock::now();
    do {
      for (int i = 0; i < 4; ++i) launch();
      CU(cudaDeviceSynchronize());
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3);
  }
  double est = 0;  // one launch, from the single-mode median
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
  (void)name;
}

double tops_of(const Rec& r) {
  for (const auto& kv : r.kv)
    if (kv.first == "tops") return std::atof(kv.second.c_str());
  return 0;
}

template <class V>
constexpr long long smem_of() {
  return (long long)Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>::Kernel::SharedStorageSize;
}

template <class V>
void run_variant(Ctx& c, const char* name, const Task& t) {
  using Store = Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>;
  using Kernel = typename Store::Kernel;
  using Gemm = cg::device::GemmUniversalAdapter<Kernel>;
  const int M = int(t.d.m), N = int(t.d.n), K = int(t.d.k), L = t.L;
  Rec base;
  base.s("method", "rung1").s("variant", name).s("kind", V::kFast ? "fast_epilogue" : "generic_epilogue").s("op", "store");
  base.i("m", M).i("n", N).i("k", K).i("batch", L).i("swizzle", t.sc.swizzle).s("raster", raster_name(t.sc.raster));
  base.s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).i("sm_count", c.sms).i("clock_khz_max", c.clock_khz).i("mem_clock_khz", c.mem_clock_khz);
  base.b("ndebug",
#ifdef NDEBUG
         true
#else
         false
#endif
  );
  if (!c.commit.empty()) base.s("commit", c.commit);
  auto emit_status = [&](const std::string& st) {
    Rec r = base;
    r.s("status", st);
    std::printf("%s\n", r.dump().c_str());
    std::fflush(stdout);
    std::fprintf(stderr, "  %s %dx%dx%d L=%d sw=%d %s: %s\n", name, M, N, K, L, t.sc.swizzle, raster_name(t.sc.raster), st.c_str());
  };
  try {
    cudaFuncAttributes fa{};
    CU(cudaFuncGetAttributes(&fa, cutlass::device_kernel<Kernel>));
    base.i("regs", fa.numRegs).i("local_bytes", (long long)fa.localSizeBytes).i("smem_dynamic_bytes", (long long)Kernel::SharedStorageSize);

    Fixture& fx = c.fixture(t.d);
    uint8_t* W = fx.W.as<uint8_t>();
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = c.sms;
    const auto sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M) * K);
    const auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(N) * K);
    K4Args ea{};
    ea.W = W, ea.ldw = N, ea.plane_w = int64_t(M) * N, ea.slot0 = 0, ea.b = 1, ea.batch0 = 0;
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
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { emit_status("skipped: can_implement failed"); return; }
    Buf ws(Gemm::get_workspace_size(args));
    if (gemm.initialize(args, ws.p) != cutlass::Status::kSuccess) { emit_status("error: initialize failed"); return; }
    auto launch = [&] {
      if (gemm.run() != cutlass::Status::kSuccess) throw Fail("gemm.run failed");
    };

    // ---- exactness: a launch on a poisoned W, then the sampled host reference and the hash
    CU(cudaMemset(W, 0xEE, fx.W.bytes));
    launch();
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    long long checked = 0, bad = 0;
    {
      const long long per = (long long)M * N;
      std::vector<std::array<long long, 3>> pos;  // (z, i, j)
      pos.push_back({0, 0, 0});
      pos.push_back({L - 1, M - 1, N - 1});
      pos.push_back({L - 1, 0, N - 1});
      pos.push_back({0, M - 1, 0});
      for (long long e : {127LL, 128LL, 255LL, 256LL, 511LL, 512LL}) {
        if (e < M && e < N) pos.push_back({(e + 3) % L, e, N - 1 - e});
      }
      for (long long q = 0; q < c.opt.samples; ++q) {
        const uint64_t h = mix(uint64_t(q) * 7919u + 17);
        pos.push_back({(long long)(h % uint64_t(L)), (long long)((h >> 20) % uint64_t(M)), (long long)((h >> 40) % uint64_t(N))});
      }
      for (const auto& p : pos) {
        const long long z = p[0], i = p[1], j = p[2];
        long long s = 0;
        for (long long kk = 0; kk < K; ++kk) s += (long long)val(1, uint64_t(z * M + i), uint64_t(kk)) * (long long)val(2, uint64_t(z * N + j), uint64_t(kk));
        const int pm = int(imd::constants::kTables.moduli[z]);
        const int want = mod_of(s, pm);
        uint8_t got = 0;
        CU(cudaMemcpy(&got, W + z * per + i * N + j, 1, cudaMemcpyDeviceToHost));
        ++checked;
        bad += got != want;
      }
    }
    Buf hbuf(8);
    CU(cudaMemset(hbuf.p, 0, 8));
    k_hash<<<1024, 256>>>(fx.W.as<uint64_t>(), uint64_t(L) * M * N / 8, hbuf.as<unsigned long long>());
    unsigned long long hash = 0;
    CU(cudaMemcpy(&hash, hbuf.p, 8, cudaMemcpyDeviceToHost));
    const auto key = std::make_tuple(t.d.m, t.d.n, t.d.k, L);
    const bool first = !c.first_hash.count(key);
    if (first) c.first_hash[key] = hash;
    const bool heq = c.first_hash[key] == hash;
    const bool exact = bad == 0 && heq;
    if (!exact) c.any_inexact = true;
    char hb[24];
    std::snprintf(hb, sizeof hb, "%016llx", hash);
    base.b("exact", exact).i("check_elements", checked).i("ref_mismatches", bad).s("hash", hb).b("hash_eq_first", heq).b("is_first_hash", first);

    // ---- timing
    std::vector<Rec> recs;
    time_modes(c, [&] { launch(); }, t, base, name, recs);
    for (Rec& r : recs) {
      r.s("status", "ok");
      std::printf("%s\n", r.dump().c_str());
      std::fflush(stdout);
    }
    std::fprintf(stderr, "  %-26s %5dx%dx%d L=%-2d sw=%d %-9s regs=%d exact=%d  single %.0f  loop %.0f TOPS\n", name, M, N, K, L, t.sc.swizzle,
                 raster_name(t.sc.raster), int(fa.numRegs), int(exact), tops_of(recs[0]), tops_of(recs[1]));
  } catch (const std::exception& e) {
    cudaGetLastError();
    // a sticky error (illegal address, hardware exception) cannot be recovered from: report it and stop
    if (cudaDeviceSynchronize() != cudaSuccess) {
      emit_status(std::string("error (sticky): ") + e.what());
      throw Sticky(e.what());
    }
    emit_status(std::string("error: ") + e.what());
  }
}

// ================================================================== DATA: the configuration table and the defaults
// Everything that is a choice (which kernels, which shapes, which scheduler knobs) is here. To add an architecture (Blackwell sm_100 / sm_120
// with tcgen05, another Hopper tile) add rows with its cc_major and a Variant-like type whose Kernel is a CUTLASS GemmUniversal with K4's epilogue,
// and guard them with the architecture's own compile condition; the harness runs the rows whose cc_major equals the device's and exits 3 if there are none.
// Rows are in priority order: if a run is cut off, the first rows have reported. Names: f_/g_ = fast / generic epilogue, TILE_MxNxK, CLUSTER_MxN, coop/ping.
//                 name                      cc  fast   quick  sweep  tile      cluster schedule
#define IMD_V(NAME, CC, FAST, QUICK, SWEEP, ...) {NAME, CC, FAST, QUICK, SWEEP, smem_of<VT<__VA_ARGS__, FAST>>(), &run_variant<VT<__VA_ARGS__, FAST>>}
const Entry kEntries[] = {
    IMD_V("f_256x128x128_2x1_coop", 9, true, true, true, T256x128, C21, Coop),
    IMD_V("f_128x128x128_2x1_coop", 9, true, true, true, T128x128, C21, Coop),
    IMD_V("f_128x256x128_1x2_coop", 9, true, true, true, T128x256, C12, Coop),
    IMD_V("f_128x256x128_1x1_coop", 9, true, true, true, T128x256, C11, Coop),
    IMD_V("f_128x128x128_1x1_ping", 9, true, true, true, T128x128, C11, Ping),
    IMD_V("f_128x256x128_2x1_coop", 9, true, false, true, T128x256, C21, Coop),
    IMD_V("f_128x256x64_1x2_coop", 9, true, false, false, T128x256k64, C12, Coop),
    IMD_V("f_128x256x128_4x1_coop", 9, true, false, false, T128x256, C41, Coop),
    IMD_V("g_256x128x128_2x1_coop", 9, false, false, false, T256x128, C21, Coop),
    IMD_V("g_128x128x128_2x1_coop", 9, false, false, false, T128x128, C21, Coop),
    IMD_V("g_128x256x128_1x2_coop", 9, false, false, false, T128x256, C12, Coop),
    IMD_V("g_128x256x128_1x1_coop", 9, false, false, false, T128x256, C11, Coop),
    IMD_V("g_128x128x128_1x1_ping", 9, false, false, false, T128x128, C11, Ping),
};
#undef IMD_V
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
// entries of the device's architecture matching a spec: names / substrings, "all", "f", "g", "quick", "sweep"
std::vector<const Entry*> select(const std::string& spec, int cc_major) {
  std::vector<const Entry*> out;
  for (const std::string& tok : split(spec)) {
    for (const Entry& e : kEntries) {
      if (e.cc_major != cc_major) continue;
      const bool hit = tok == "all" || (tok == "f" && e.fast) || (tok == "g" && !e.fast) || (tok == "quick" && e.quick) ||
                       (tok == "sweep" && e.sweep) || std::string(e.name).find(tok) != std::string::npos;
      if (hit && std::find(out.begin(), out.end(), &e) == out.end()) out.push_back(&e);
    }
  }
  return out;
}

}  // namespace rung1

using namespace rung1;

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--", 2) != 0) { std::fprintf(stderr, "usage: see the header of src/k4/hopper/rung1.cu\n"); return 2; }
    const std::string key = argv[i] + 2;
    if (key == "quick" || key == "list" || key == "no-nvml") { a[key] = "1"; continue; }
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
    a[key] = argv[++i];
  }
  Ctx c;
  Opt& o = c.opt;
  const char* q = std::getenv("IMD_RUNG1_QUICK");
  o.quick = a.count("quick") || (q && *q && std::strcmp(q, "0") != 0);
  o.nvml = !a.count("no-nvml");
  if (o.quick) o.reps = 10, o.samples = 128;
  if (a.count("reps")) o.reps = std::max(1, std::atoi(a["reps"].c_str()));
  if (a.count("warmup")) o.warmup = std::max(1, std::atoi(a["warmup"].c_str()));
  if (a.count("loop")) o.loop = std::max(1, std::atoi(a["loop"].c_str()));
  if (a.count("max-seconds")) o.max_seconds = std::atof(a["max-seconds"].c_str());
  if (a.count("samples")) o.samples = std::atoll(a["samples"].c_str());
  if (a.count("list")) {
    for (const Entry& e : kEntries) std::printf("%s cc%d smem_bytes=%lld\n", e.name, e.cc_major, e.smem_bytes);
    return 0;
  }

  cudaDeviceProp prop{};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    std::fprintf(stderr, "rung 1: no CUDA device\n");
    return 3;
  }
  const std::vector<const Entry*> vars = select(a.count("variants") ? a["variants"] : (o.quick ? "quick" : "all"), prop.major);
  if (vars.empty()) {
    std::fprintf(stderr, "rung 1: no configuration for this device (%s, sm_%d%d): the table kEntries has rows for compute capability major %d only\n",
                 prop.name, prop.major, prop.minor, 9);
    return 3;
  }
  c.gpu = prop.name;
  c.sms = prop.multiProcessorCount;
  c.cc_major = prop.major, c.cc_minor = prop.minor;
  cudaDeviceGetAttribute(&c.clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&c.mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  if (const char* cm = std::getenv("BENCH_COMMIT")) c.commit = cm;
  if (o.nvml) c.nvml.open(dev);

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
  if (c.lmax > imd::constants::kNMax) { std::fprintf(stderr, "batch above kNMax\n"); return 2; }
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
  const std::vector<const Entry*> sweep_vars = select(a.count("sweep-variants") ? a["sweep-variants"] : "sweep", prop.major);

  std::fprintf(stderr, "imd_rung1: %s (%d SMs, max SM clock %d MHz), NVML %s, %zu variants, ndebug=%d\n", c.gpu.c_str(), c.sms, c.clock_khz / 1000,
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

  // the work list, in priority order
  std::vector<std::pair<const Entry*, Task>> work;
  const int primary = batches[0];
  for (const Dims& d : shapes)
    for (const Entry* e : vars) work.push_back({e, {d, primary, Sched{}}});
  for (const Dims& d : sweep_shapes)
    for (const Entry* e : sweep_vars)
      for (int rst : kSweepRaster)
        for (int sw : kSweepSwizzle) work.push_back({e, {d, primary, Sched{sw, rst}}});
  for (size_t bi = 1; bi < batches.size(); ++bi)
    for (const Dims& d : shapes)
      for (const Entry* e : vars) work.push_back({e, {d, batches[bi], Sched{}}});

  int rc = 0;
  try {
    for (const auto& w : work) {
      try {
        w.first->run(c, w.first->name, w.second);
      } catch (const Sticky&) {
        throw;
      } catch (const std::exception& e) {  // an allocation failure for a shape etc.
        cudaGetLastError();
        std::fprintf(stderr, "  %s: %s\n", w.first->name, e.what());
        Rec r;
        r.s("method", "rung1").s("gpu", c.gpu).s("cc", std::to_string(c.cc_major) + "." + std::to_string(c.cc_minor)).s("variant", w.first->name).i("m", w.second.d.m).i("n", w.second.d.n).i("k", w.second.d.k).i("batch", w.second.L)
            .s("status", std::string("error: ") + e.what());
        std::printf("%s\n", r.dump().c_str());
        std::fflush(stdout);
      }
    }
  } catch (const Sticky& e) {
    std::fprintf(stderr, "rung 1: sticky device error, stopping: %s\n", e.what());
    rc = 4;
  }
  if (rc == 0 && c.any_inexact) rc = 1;
  std::fprintf(stderr, rc == 0 ? "rung 1: done, every record exact\n" : "rung 1: finished with status %d (1 = some record not exact, 4 = device error)\n", rc);
  return rc;
}
