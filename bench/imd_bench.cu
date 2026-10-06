// imd_bench: time one FP64 contraction case with immerdink.
//
//   imd_bench --name sq4096 --spec ik,jk->ij --ext i=4096,j=4096,k=4096 \
//             --methods imd,bw --N 8,14,20 [options]
//
// Run it through bench/run_sweep.sh (which goes through tools/run_bench.sh); see
// docs/BENCHMARKING.md. Every operand is dense row-major in its label order (last label
// fastest). One JSON object per (method, variant) is appended to --out (default stdout).
// immerdink only: comparisons with other libraries live in a separate evaluation workspace.
//
// Methods
//   imd       immerdink imd_plan_create + imd_contract, N from --N. With a library built
//             by this project (IMMERDINK_PROFILE=ON), --phase-reps more calls are made with
//             per-phase event marks and the record gets "phases": per phase the median
//             time, its MEMORY_MODEL §4 lower bound, the time as % of that bound and the
//             performance model's predicted time (pred_ms; bounds and predictions from
//             libimmerdink_host imd_perf_predict, DESIGN §6c, shipped calibration); every
//             record has pred_ms / pred_err_pct of the whole call (the timed whole-call loop
//             runs with the marks switched off)
//   packed    v1.7 (DESIGN §6b, D1): the caller's A or B (--pack A,B) packed once with
//             imd_pack (imd_pack_desc_for: bit-identical C) and reused by R contractions
//             (--reuse 1,4,16): R x imd_contract vs imd_pack + R x imd_contract_packed,
//             interleaved per repetition, plus imd_pack / imd_contract / imd_contract_packed
//             alone; workspaces and pack bytes
//   duel      two immerdink configurations (--duel-a, --duel-b) timed INTERLEAVED, call by
//             call, in one process (docs/BENCHMARKING.md "Tiny shapes", NOTES-A5 §1.2): both
//             sides see the same GPU clock / DVFS / host-core state. A side is "default" or
//             a comma list of N=<n>, vec=generate|resident, g=<g> (0 = N),
//             k4=rule|tune|0..3, k5=0|1|2|4 (0 = the launch rule), ka=auto|16|64 (v1.13 §4b:
//             the side's K alignment, via the test hook at plan creation); unset keys keep the plan
//             defaults, N defaults to the --N value. Defaults: A = default (the K4 rule),
//             B = k4=tune (the plan-time K4 autotune); when both pick the same K4
//             configuration (k4_config in both records) the run is an A/A resolution check.
//             Records duel-a and duel-b; duel-b carries ratio_median = median over pairs of
//             t_a / t_b (> 1: B faster), its p10 / p90, and whether the two C are bit-identical
//   bw        device-to-device copy of A (achievable DRAM bandwidth; a hardware reference)
//
// --k-align auto|16|64 (DESIGN v1.13 §4b.6): the K alignment of every plan and pack the run
// creates (libimmerdink_host's test hook imd_test_set_k_align; auto = the library's rule, A =
// 64 iff k > 96). For the stage-2 A/B sweep and the A = 16-forced calibration cases; every
// imd / duel record carries the plan's k_align.
//
// Accuracy: for --samples output elements (all of them if C is smaller), the reference
// is a double-double (Dot2) dot product computed on the device; the host holds only the
// samples (no operand copies), so host RSS is small for every shape. Reported: max |c - r| / |r| and the
// normwise max |c - r| / (||a_i||_2 ||b_j||_2) of DESIGN §3.8.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/resource.h>
#include <immerdink/immerdink.h>

#ifdef IMD_BENCH_PROFILE
#include "profile_hooks.h"  // src/: per-phase event marks (libimmerdink built with IMMERDINK_PROFILE)
#endif

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// libimmerdink_host's test-only K-alignment hook (src/host/k_hat.cpp; not in a public header).
extern "C" int imd_test_set_k_align(int32_t a);

namespace {

// ---------------------------------------------------------------------------- errors ---
struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#define CU(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) throw Failure(std::string(#x) + ": " + cudaGetErrorString(e_));   \
  } while (0)
#define CI(x)                                                                                \
  do {                                                                                       \
    imd_status_t s_ = (x);                                                                   \
    if (s_ != IMD_STATUS_SUCCESS) throw Failure(std::string(#x) + ": " + imd_status_string(s_)); \
  } while (0)

// ----------------------------------------------------------------------- device buffer ---
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(size_t n) : bytes(n) {
    if (n) CU(cudaMalloc(&p, n));
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    std::swap(p, o.p);
    std::swap(bytes, o.bytes);
    return *this;
  }
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  template <class T> T* as() const { return static_cast<T*>(p); }
};

// ------------------------------------------------------------------------ JSON writer ---
std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c == '\n') o += "\\n";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += char(c);
  }
  return o + "\"";
}
std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  snprintf(b, sizeof b, "%.6g", v);
  return b;
}
struct Json {
  std::vector<std::pair<std::string, std::string>> kv;
  Json& raw(const std::string& k, const std::string& v) { kv.emplace_back(k, v); return *this; }
  Json& str(const std::string& k, const std::string& v) { return raw(k, jstr(v)); }
  Json& num(const std::string& k, double v) { return raw(k, jnum(v)); }
  Json& integer(const std::string& k, long long v) { return raw(k, std::to_string(v)); }
  std::string dump() const {
    std::string o = "{";
    for (size_t i = 0; i < kv.size(); ++i) o += (i ? ", " : "") + jstr(kv[i].first) + ": " + kv[i].second;
    return o + "}";
  }
};

std::string sh(const char* cmd) {
  std::string out;
  if (FILE* f = popen(cmd, "r")) {
    char b[512];
    while (fgets(b, sizeof b, f)) out += b;
    pclose(f);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  return out;
}
std::string read_file(const char* path) {
  std::string out;
  if (FILE* f = fopen(path, "r")) {
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
    fclose(f);
  }
  return out;
}
std::string getenv_or(const char* k, const char* d) {
  const char* v = std::getenv(k);
  return v ? v : d;
}
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, sep))
    if (!t.empty()) out.push_back(t);
  return out;
}

// --------------------------------------------------------------------- data generation ---
__device__ __host__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
// (u - 1/2) * exp(phi * g), u uniform in [0,1), g standard normal: GEMMul8's test data.
// dom > 0 (--data dominant): every dom-th element (one per row of k contiguous contracted
// elements, e.g. ik / jk operands) is multiplied by 2^20, so each row has one dominant entry.
__global__ void k_fill(double* p, size_t n, uint64_t seed, double phi, size_t dom) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    uint64_t h1 = splitmix64(seed ^ (2 * i)), h2 = splitmix64(seed ^ (2 * i + 1));
    double u = double(h1 >> 11) * 0x1.0p-53, u1 = (double(h2 >> 11) + 1.0) * 0x1.0p-53;
    double u2 = double(splitmix64(h2) >> 11) * 0x1.0p-53;
    double g = sqrt(-2.0 * log(u1)) * cospi(2.0 * u2);
    p[i] = (u - 0.5) * exp(phi * g) * ((dom != 0 && i % dom == 0) ? 1048576.0 : 1.0);
  }
}

// ----------------------------------------------------------------------- the contraction ---
struct Operand {
  std::string labels;          // one char per mode, outermost first
  std::vector<int64_t> extent; // per mode
  std::vector<int64_t> stride; // dense row-major
  size_t size = 1;
};

struct Case {
  std::string name, spec;
  std::map<char, int64_t> ext;
  Operand A, B, C;
  std::string I, J, K, L;  // GEMM-view label groups (I, J, L in C order; K in A order)
  int64_t m = 1, n = 1, k = 1, b = 1;
};

Operand make_operand(const std::string& labels, const std::map<char, int64_t>& ext) {
  Operand o;
  o.labels = labels;
  for (char c : labels) {
    auto it = ext.find(c);
    if (it == ext.end()) throw Failure(std::string("no extent for label ") + c);
    o.extent.push_back(it->second);
  }
  o.stride.assign(labels.size(), 1);
  for (int d = int(labels.size()) - 2; d >= 0; --d) o.stride[d] = o.stride[d + 1] * o.extent[d + 1];
  for (auto e : o.extent) o.size *= size_t(e);
  return o;
}

Case parse_case(const std::string& name, const std::string& spec, const std::string& exts) {
  Case c;
  c.name = name;
  c.spec = spec;
  for (auto& kv : split(exts, ',')) {
    auto eq = kv.find('=');
    if (eq != 1) throw Failure("bad --ext entry " + kv);
    c.ext[kv[0]] = std::stoll(kv.substr(2));
  }
  auto arrow = spec.find("->"), comma = spec.find(',');
  if (arrow == std::string::npos || comma == std::string::npos || comma > arrow)
    throw Failure("bad --spec " + spec);
  c.A = make_operand(spec.substr(0, comma), c.ext);
  c.B = make_operand(spec.substr(comma + 1, arrow - comma - 1), c.ext);
  c.C = make_operand(spec.substr(arrow + 2), c.ext);
  auto in = [](const std::string& s, char x) { return s.find(x) != std::string::npos; };
  for (char x : c.C.labels) {
    bool a = in(c.A.labels, x), bb = in(c.B.labels, x);
    if (a && bb) c.L += x;
    else if (a) c.I += x;
    else if (bb) c.J += x;
    else throw Failure("output-only label not supported by the bench");
  }
  for (char x : c.A.labels)
    if (!in(c.C.labels, x)) {
      if (!in(c.B.labels, x)) throw Failure("single-operand label not supported by the bench");
      c.K += x;
    }
  for (char x : c.B.labels)
    if (!in(c.C.labels, x) && !in(c.A.labels, x))
      throw Failure("single-operand label not supported by the bench");
  for (char x : c.I) c.m *= c.ext[x];
  for (char x : c.J) c.n *= c.ext[x];
  for (char x : c.K) c.k *= c.ext[x];
  for (char x : c.L) c.b *= c.ext[x];
  return c;
}

// ---------------------------------------------------------------------- device reference ---
// Dot2 (Ogita-Rump-Oishi): the result is as accurate as if computed in twice the working
// precision, then rounded; far below the FP64-level errors being measured. Computed on the
// device for the sampled outputs only (one thread per sample), so the host never holds an
// operand: host memory stays O(samples) for every shape. Intrinsics keep nvcc from fusing
// the error-free transformations into FMAs.
struct DD {
  double hi = 0, lo = 0;
  __device__ void add_prod(double a, double b) {
    double p = __dmul_rn(a, b), pe = __fma_rn(a, b, -p);
    double s = __dadd_rn(hi, p), bb = __dsub_rn(s, hi);
    double se = __dadd_rn(__dsub_rn(hi, __dsub_rn(s, bb)), __dsub_rn(p, bb));
    hi = s;
    lo = __dadd_rn(lo, __dadd_rn(pe, se));
  }
  __device__ void normalize() {  // TwoSum(hi, lo): hi = fl(hi + lo), lo = the exact remainder
    double s = __dadd_rn(hi, lo), bb = __dsub_rn(s, hi);
    lo = __dadd_rn(__dsub_rn(hi, __dsub_rn(s, bb)), __dsub_rn(lo, bb));
    hi = s;
  }
};

constexpr int kMaxKModes = 8;
struct KIter {  // mixed-radix iteration over the contracted labels
  int nd = 0;
  int64_t ext[kMaxKModes], sa[kMaxKModes], sb[kMaxKModes];
  int64_t k = 1;
};

// out[s] = {hi, lo, ||a_i||_2 ||b_j||_2} of sample s, whose A/B base offsets are offA/offB.
__global__ void k_reference(const double* A, const double* B, const int64_t* offA,
                            const int64_t* offB, size_t ns, KIter it, double* out) {
  size_t s = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (s >= ns) return;
  int64_t oa = offA[s], ob = offB[s], ctr[kMaxKModes] = {};
  DD acc;
  double na = 0, nb = 0;
  for (int64_t t = 0; t < it.k; ++t) {
    double a = A[oa], b = B[ob];
    acc.add_prod(a, b);
    na += a * a;
    nb += b * b;
    for (int d = it.nd - 1; d >= 0; --d) {  // increment the K counter
      oa += it.sa[d];
      ob += it.sb[d];
      if (++ctr[d] < it.ext[d]) break;
      oa -= it.sa[d] * it.ext[d];
      ob -= it.sb[d] * it.ext[d];
      ctr[d] = 0;
    }
  }
  acc.normalize();
  out[3 * s] = acc.hi;
  out[3 * s + 1] = acc.lo;
  out[3 * s + 2] = sqrt(na) * sqrt(nb);
}
__global__ void k_gather(const double* C, const size_t* idx, size_t ns, double* out) {
  size_t s = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (s < ns) out[s] = C[idx[s]];
}

struct Accuracy {
  double max_rel = 0, max_norm = 0;
  long long samples = 0, bad = 0;  // bad: non-finite results where the reference is finite
};

// Sampled reference: the C linear indices to check, the reference as a double-double
// (hi + lo, so errors are measured against the exact value, not a rounded one) and
// ||a_i||_2 ||b_j||_2. idx also lives on the device (dIdx) for the gather.
struct Reference {
  std::vector<size_t> idx;
  std::vector<double> hi, lo, nrm;
  DevBuf dIdx, dGot;
};

Reference make_reference(const Case& c, const double* dA, const double* dB, long long samples,
                         uint64_t seed, cudaStream_t st) {
  Reference R;
  size_t total = c.C.size;
  if (size_t(samples) >= total) {
    for (size_t i = 0; i < total; ++i) R.idx.push_back(i);
  } else {
    for (long long s = 0; s < samples; ++s) R.idx.push_back(splitmix64(seed + 7919 * s) % total);
  }
  KIter it;
  it.k = c.k;
  if (c.K.size() > size_t(kMaxKModes)) throw Failure("too many contracted labels");
  for (char x : c.K) {
    it.ext[it.nd] = c.ext.at(x);
    it.sa[it.nd] = c.A.stride[c.A.labels.find(x)];
    it.sb[it.nd] = c.B.stride[c.B.labels.find(x)];
    ++it.nd;
  }
  size_t ns = R.idx.size();
  std::vector<int64_t> offA(ns), offB(ns);
  for (size_t s = 0; s < ns; ++s) {
    size_t rem = R.idx[s];
    int64_t oa = 0, ob = 0;
    for (int d = int(c.C.labels.size()) - 1; d >= 0; --d) {
      int64_t v = int64_t(rem % size_t(c.C.extent[d]));
      rem /= size_t(c.C.extent[d]);
      char x = c.C.labels[d];
      auto pa = c.A.labels.find(x), pb = c.B.labels.find(x);
      if (pa != std::string::npos) oa += v * c.A.stride[pa];
      if (pb != std::string::npos) ob += v * c.B.stride[pb];
    }
    offA[s] = oa;
    offB[s] = ob;
  }
  DevBuf dOA(ns * 8), dOB(ns * 8), dOut(ns * 24);
  R.dIdx = DevBuf(ns * sizeof(size_t));
  R.dGot = DevBuf(ns * 8);
  CU(cudaMemcpy(dOA.p, offA.data(), ns * 8, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dOB.p, offB.data(), ns * 8, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(R.dIdx.p, R.idx.data(), ns * sizeof(size_t), cudaMemcpyHostToDevice));
  k_reference<<<unsigned((ns + 127) / 128), 128, 0, st>>>(dA, dB, dOA.as<int64_t>(), dOB.as<int64_t>(),
                                                          ns, it, dOut.as<double>());
  CU(cudaStreamSynchronize(st));
  CU(cudaGetLastError());
  std::vector<double> out(3 * ns);
  CU(cudaMemcpy(out.data(), dOut.p, ns * 24, cudaMemcpyDeviceToHost));
  R.hi.resize(ns);
  R.lo.resize(ns);
  R.nrm.resize(ns);
  for (size_t s = 0; s < ns; ++s) {
    R.hi[s] = out[3 * s];
    R.lo[s] = out[3 * s + 1];
    R.nrm[s] = out[3 * s + 2];
  }
  return R;
}

// Gather the sampled outputs of the device C and compare them with the reference.
Accuracy check(Reference& R, const double* dC, cudaStream_t st) {
  size_t ns = R.idx.size();
  k_gather<<<unsigned((ns + 255) / 256), 256, 0, st>>>(dC, R.dIdx.as<size_t>(), ns, R.dGot.as<double>());
  CU(cudaStreamSynchronize(st));
  CU(cudaGetLastError());
  std::vector<double> got(ns);
  CU(cudaMemcpy(got.data(), R.dGot.p, ns * 8, cudaMemcpyDeviceToHost));
  Accuracy a;
  for (size_t s = 0; s < ns; ++s) {
    double g = got[s], ref = R.hi[s], d = std::fabs((g - ref) - R.lo[s]);
    ++a.samples;
    if (!std::isfinite(g)) { ++a.bad; continue; }
    if (ref != 0) a.max_rel = std::max(a.max_rel, d / std::fabs(ref));
    if (R.nrm[s] != 0) a.max_norm = std::max(a.max_norm, d / R.nrm[s]);
  }
  return a;
}

// -------------------------------------------------------------------------------- timing ---
struct Timing {
  std::vector<double> t;  // seconds per repetition
  std::vector<double> mhz;  // v1.10: SM clock right after each timed call (clock probe), MHz
  double median() const {
    auto v = t;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n == 0 ? NAN : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
  }
  double min() const { return t.empty() ? NAN : *std::min_element(t.begin(), t.end()); }
  double max() const { return t.empty() ? NAN : *std::max_element(t.begin(), t.end()); }
};

struct Options {
  int reps = 10, warmup = 2, min_reps = 3;
  double max_seconds = 15;  // timed-loop budget per method/variant (after min_reps)
  long long samples = 256;
  int phase_reps = 5;  // profiled imd calls (per-phase marks), after the timed loop
  std::vector<int> vec_modes = {IMD_VEC_GENERATE};  // imd: --vec generate,resident (v1.2)
  int mode = IMD_EMULATE;  // imd: --mode emulate|direct (B2 calibration of the DIRECT kernel)
  std::vector<int> groups = {0};  // imd: --g 1,2,4,0 (residue group size g; 0 = N)
  // imd: --k4 rule,tune,0,1,2,3 (A3: the K4 configuration; -1 = the plan-time rule, -2 =
  // the k4_tune plan option, 0.. = forced through the benchmark hook)
  std::vector<int> k4 = {-1};
  double phi = 0.5;
  bool dominant = false;  // --data dominant (B2 eps-path calibration: one dominant entry per row)
  // v1.10 (DESIGN §6c): --clock-probe 1 records the SM clock after every timed call (also
  // IMD_BENCH_CLOCK=1); --l2 flush writes a 2 x L2 buffer before every timed imd call (outside
  // the timed window: the COLD state; default repeat); --sustain-s / --sustain-warm-s: method
  // sustained's measured window and warm-up.
  bool clock_probe = false;
  int k_align_hook = 0;  // v1.13 §4b: --k-align (0 = auto) / IMD_BENCH_K_ALIGN
  bool l2_flush = false;
  double sustain_s = 8, sustain_warm_s = 3;
  // Follow-up (hot clock, NOTES-B2 §9): --sustain-trace-s T adds a time series to method
  // sustained's record, one point per T seconds of the window: mean s per call, board power,
  // SM clock, GPU temperature, OR of the clock-event reasons, fan %. 0 = off.
  double sustain_trace_s = 0;
  uint64_t seed = 1;
  std::string out;
};

// A5 diagnostic (IMD_BENCH_CLOCK=1): the SM clock right after each timed call, measured by
// one warp spinning ~4 µs of %globaltimer and counting clock64 cycles (outside the timed
// window; it adds ~5 µs of GPU work between calls).
__global__ void k_clock_probe(unsigned long long* out) {
  unsigned long long t0, t1, c0, c1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  c0 = clock64();
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < 4000);
  c1 = clock64();
  if (threadIdx.x == 0) { out[0] = c1 - c0; out[1] = t1 - t0; }
}

Timing time_it(const std::function<void()>& run, cudaStream_t s, const Options& o,
               const std::function<void()>& pre = nullptr) {
  for (int w = 0; w < o.warmup; ++w) {
    if (pre) pre();
    run();
  }
  CU(cudaStreamSynchronize(s));
  CU(cudaGetLastError());
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  Timing T;
  double total = 0;
  const char* clk_env = getenv("IMD_BENCH_CLOCK");
  const bool clk = (clk_env && *clk_env == '1') || o.clock_probe;
  unsigned long long* clk_buf = nullptr;
  std::vector<double> mhz;
  if (clk) CU(cudaHostAlloc(&clk_buf, 2 * sizeof(unsigned long long), cudaHostAllocMapped));
  for (int r = 0; r < o.reps; ++r) {
    if (pre) pre();  // outside the timed window (v1.10: the L2 flush of --l2 flush)
    CU(cudaEventRecord(e0, s));
    run();
    CU(cudaEventRecord(e1, s));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    T.t.push_back(ms * 1e-3);
    if (clk) {
      k_clock_probe<<<1, 32, 0, s>>>(clk_buf);
      CU(cudaStreamSynchronize(s));
      mhz.push_back(1e3 * double(clk_buf[0]) / double(clk_buf[1]));
    }
    total += ms * 1e-3;
    if (r + 1 >= o.min_reps && total > o.max_seconds) break;
  }
  CU(cudaGetLastError());
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  if (const char* v = getenv("IMD_BENCH_TIMES"); v && *v == '1') {  // A5: every timed call, µs
    fprintf(stderr, "  times_us:");
    for (double t : T.t) fprintf(stderr, " %.1f", t * 1e6);
    fprintf(stderr, "\n");
  }
  T.mhz = mhz;
  if (clk && getenv("IMD_BENCH_CLOCK")) {
    fprintf(stderr, "  sm_mhz:");
    for (double m : mhz) fprintf(stderr, " %.0f", m);
    fprintf(stderr, "\n");
    cudaFreeHost(clk_buf);
  }
  return T;
}

// A side of method duel: one immerdink configuration. k4: -1 the plan-time rule, -2 the
// k4_tune plan option, >= 0 a forced configuration; k5: 0 the launch rule, 1 / 2 / 4
// elements per thread. Forced k4 / k5 go through the benchmark hooks (profiling build).
struct DuelSide {
  int N = 14, vec = IMD_VEC_GENERATE, g = 0, k4 = -1, k5 = 0;
  int ka = -1;  // v1.13 §4b: -1 = the run's --k-align, 0 = AUTO, 16 / 64 = forced (plan-time hook)
  std::string describe() const {
    std::string d = "N=" + std::to_string(N) + " g=" + ((g == 0 || g >= N) ? "N" : std::to_string(g));
    d += vec == IMD_VEC_RESIDENT ? " vec=resident" : " vec=generate";
    d += k4 == -1 ? " k4=rule" : k4 == -2 ? " k4=tune" : " k4=" + std::to_string(k4);
    d += k5 == 0 ? " k5=rule" : " k5=" + std::to_string(k5);
    if (ka >= 0) d += ka == 0 ? " ka=auto" : " ka=" + std::to_string(ka);
    return d;
  }
};

// "default" (the plan defaults at --N) or a comma list of key=value overrides.
DuelSide parse_duel_side(const std::string& spec, int N) {
  DuelSide d;
  d.N = N;
  if (spec.empty() || spec == "default") return d;
  for (auto& kv : split(spec, ',')) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) throw Failure("bad duel side entry '" + kv + "' (key=value)");
    const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    if (k == "N") d.N = std::stoi(v);
    else if (k == "vec" && (v == "generate" || v == "resident"))
      d.vec = v == "resident" ? IMD_VEC_RESIDENT : IMD_VEC_GENERATE;
    else if (k == "g") d.g = std::stoi(v);
    else if (k == "k4") d.k4 = v == "rule" ? -1 : v == "tune" ? -2 : std::stoi(v);
    else if (k == "k5" && (v == "0" || v == "1" || v == "2" || v == "4")) d.k5 = std::stoi(v);
    else if (k == "ka" && (v == "auto" || v == "16" || v == "64")) d.ka = v == "auto" ? 0 : std::stoi(v);
    else throw Failure("bad duel side entry '" + kv + "' (N=, vec=generate|resident, g=, "
                       "k4=rule|tune|0..3, k5=0|1|2|4, ka=auto|16|64)");
  }
  return d;
}

// v1.10: read n words (L2 eviction by clean lines); the store never happens for the zeroed
// buffer, but the compiler cannot drop the reads.
__global__ void k_touch(const unsigned long long* p, size_t n, unsigned long long* sink) {
  unsigned long long acc = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    acc += p[i];
  if (acc == 1) sink[0] = acc;
}

// v1.10 (method sustained): NVML through dlopen (no build dependency): board power and SM
// clock. Neither read disturbs the energy counter (evaluation workspace findings/energy).
struct Nvml {
  void* lib = nullptr;
  void* dev = nullptr;
  int (*power)(void*, unsigned*) = nullptr;
  int (*clock)(void*, int, unsigned*) = nullptr;
  int (*limit)(void*, unsigned*) = nullptr;
  int (*temp)(void*, int, unsigned*) = nullptr;
  int (*reasons)(void*, unsigned long long*) = nullptr;
  int (*fan)(void*, unsigned*) = nullptr;
  bool ok() const { return dev != nullptr; }
  Nvml() {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) return;
    auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
    auto handle = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    power = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
    clock = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib, "nvmlDeviceGetClockInfo"));
    limit = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetEnforcedPowerLimit"));
    temp = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib, "nvmlDeviceGetTemperature"));
    reasons = reinterpret_cast<int (*)(void*, unsigned long long*)>(
        dlsym(lib, "nvmlDeviceGetCurrentClocksThrottleReasons"));
    fan = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetFanSpeed"));
    if (!init || !handle || !power || !clock || init() != 0 || handle(0, &dev) != 0) dev = nullptr;
  }
  double power_w() const { unsigned mw = 0; return ok() && power(dev, &mw) == 0 ? mw * 1e-3 : NAN; }
  double sm_mhz() const { unsigned m = 0; return ok() && clock(dev, 1, &m) == 0 ? double(m) : NAN; }
  double limit_w() const { unsigned mw = 0; return ok() && limit && limit(dev, &mw) == 0 ? mw * 1e-3 : NAN; }
  double temp_c() const { unsigned t = 0; return ok() && temp && temp(dev, 0, &t) == 0 ? double(t) : NAN; }
  unsigned long long reason_bits() const {
    unsigned long long r = 0;
    return ok() && reasons && reasons(dev, &r) == 0 ? r : 0;
  }
  double fan_pct() const { unsigned f = 0; return ok() && fan && fan(dev, &f) == 0 ? double(f) : NAN; }
};

// Number of elements whose bit patterns differ (method duel: are the two sides' C identical).
__global__ void k_count_diff(const unsigned long long* x, const unsigned long long* y, size_t n,
                             unsigned long long* count) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    local += x[i] != y[i];
  if (local) atomicAdd(count, local);
}

// --------------------------------------------------------------------- the bench context ---
struct Bench {
  Case c;
  Options o;
  cudaStream_t s = nullptr;
  imd_handle_t imd = nullptr;
  DevBuf dA, dB, dC;
  Reference R;  // sampled reference (host memory O(samples); operands stay on the device)
  Json meta;     // fields common to every record
  FILE* out = stdout;

  // GEMM view (a case descriptor in every record): canonical labels, and which operands a
  // matrix-only library would have to permute to reach it.
  std::string cA, cB, cC;  // labels of the GEMM-view operands
  bool needA = false, needB = false, needC = false;

  void emit(Json j) {
    Json rec = meta;
    for (auto& kv : j.kv) rec.kv.push_back(kv);
    rec.str("timestamp", sh("date -Is"));
    fprintf(out, "%s\n", rec.dump().c_str());
    fflush(out);
    fprintf(stderr, "  %s\n", j.dump().c_str());
  }
  // Peak memory charged to this process' cgroup (the tools/memcap.sh scope: anon + page
  // cache + pinned), in MiB; NaN if not readable.
  static double cgroup_peak_mib() {
    std::string cg = read_file("/proc/self/cgroup");  // "0::/path\n"
    auto p = cg.find("::");
    if (p == std::string::npos) return NAN;
    std::string path = cg.substr(p + 2);
    while (!path.empty() && path.back() == '\n') path.pop_back();
    std::string v = read_file(("/sys/fs/cgroup" + path + "/memory.peak").c_str());
    return v.empty() ? NAN : std::stod(v) / 1048576.0;
  }
  static double host_maxrss_mib() {  // peak resident set of this process so far
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return double(u.ru_maxrss) / 1024.0;
  }
  std::string gpu_state() {
    return sh("nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,"
              "clocks_event_reasons.active --format=csv,noheader 2>/dev/null");
  }

  // Common record for a timed variant.
  void record(const std::string& method, const std::string& variant, int N, const Timing& T,
              const Accuracy* acc, double flops, size_t ws_bytes, Json extra = {}) {
    Json j;
    j.str("method", method).str("variant", variant).integer("N", N);
    j.integer("reps", (long long)T.t.size());
    j.num("median_s", T.median()).num("min_s", T.min()).num("max_s", T.max());
    j.num("spread_pct", 100.0 * (T.max() - T.min()) / T.median());
    if (!T.mhz.empty()) {  // v1.10: the SM clock of the timed calls (median of the probes)
      auto v = T.mhz;
      std::sort(v.begin(), v.end());
      j.num("sm_mhz_median", v[v.size() / 2]).num("sm_mhz_min", v.front()).num("sm_mhz_max", v.back());
    }
    j.num("tflops_eq", flops / T.median() / 1e12);
    j.integer("ws_bytes", (long long)ws_bytes);
    if (acc) {
      j.num("max_relerr", acc->max_rel).num("max_normerr", acc->max_norm);
      j.integer("err_samples", acc->samples).integer("nonfinite", acc->bad);
    }
    j.str("gpu_after", gpu_state()).num("host_maxrss_mib", host_maxrss_mib())
        .num("cgroup_peak_mib", cgroup_peak_mib()).str("status", "ok");
    for (auto& kv : extra.kv) j.kv.push_back(kv);
    emit(j);
  }
  void record_error(const std::string& method, const std::string& variant, int N,
                    const std::string& why) {
    Json j;
    j.str("method", method).str("variant", variant).integer("N", N).str("status", "error: " + why);
    emit(j);
  }

  double flops() const { return 2.0 * double(c.m) * double(c.n) * double(c.k) * double(c.b); }

  Accuracy result_accuracy() {
    return check(R, dC.as<double>(), s);
  }
  void poison_C() { CU(cudaMemset(dC.p, 0xff, dC.bytes)); }  // NaN everywhere

  // v1.10 (DESIGN §6c.1 "Cache state"): evict the L2 by writing a buffer of twice its size
  // (on the stream, before the timed window).
  DevBuf l2buf;
  int flush_n = 0;
  void flush_l2() {
    if (l2buf.bytes == 0) {
      cudaDeviceProp p{};
      CU(cudaGetDeviceProperties(&p, 0));
      l2buf = DevBuf(2 * size_t(p.l2CacheSize));
    }
    // Read (not written: dirty lines would be written back during the timed call); the
    // first use initializes the buffer.
    if (flush_n++ == 0) CU(cudaMemsetAsync(l2buf.p, 0, l2buf.bytes, s));
    k_touch<<<1024, 256, 0, s>>>(l2buf.as<const unsigned long long>(), l2buf.bytes / 8,
                                 l2buf.as<unsigned long long>());
  }

  // ------------------------------------------------------------------------ setup ---
  void setup() {
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CI(imd_create(&imd));

    dA = DevBuf(c.A.size * 8);
    dB = DevBuf(c.B.size * 8);
    dC = DevBuf(c.C.size * 8);
    const size_t dom = o.dominant ? size_t(c.k) : 0;
    k_fill<<<1024, 256, 0, s>>>(dA.as<double>(), c.A.size, splitmix64(o.seed * 3 + 0), o.phi, dom);
    k_fill<<<1024, 256, 0, s>>>(dB.as<double>(), c.B.size, splitmix64(o.seed * 3 + 1), o.phi, dom);
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    R = make_reference(c, dA.as<double>(), dB.as<double>(), o.samples, o.seed, s);

    // GEMM view. A: L ++ (I ++ K | K ++ I); B: L ++ (J ++ K | K ++ J); C: L ++ I ++ J.
    auto pick = [&](const std::string& have, const std::string& X) {
      return (have == c.L + X + c.K || have == c.L + c.K + X) ? have : c.L + X + c.K;
    };
    cA = pick(c.A.labels, c.I);
    cB = pick(c.B.labels, c.J);
    cC = c.L + c.I + c.J;
    needA = cA != c.A.labels;
    needB = cB != c.B.labels;
    needC = cC != c.C.labels;

    cudaDeviceProp p{};
    CU(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    std::string extstr;
    for (auto& [k, v] : c.ext) extstr += (extstr.empty() ? "" : ",") + std::string(1, k) + "=" + std::to_string(v);
    meta.str("case", c.name).str("spec", c.spec).str("ext", extstr);
    meta.integer("m", c.m).integer("n", c.n).integer("k", c.k).integer("b", c.b);
    meta.str("gemm_view", cA + "," + cB + "->" + cC);
    meta.raw("perm_needed", std::string("\"") + (needA ? "A" : "") + (needB ? "B" : "") + (needC ? "C" : "") + "\"");
    meta.num("phi", o.phi).integer("seed", (long long)o.seed);
    meta.integer("k_align_hook", o.k_align_hook);  // v1.13 §4b: 0 = AUTO, else forced A
    meta.str("data", o.dominant ? "dominant" : "normal");
    meta.str("gpu", p.name).integer("sm_count", p.multiProcessorCount);
    meta.integer("l2_bytes", (long long)p.l2CacheSize);
    meta.str("driver", sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null"));
    meta.integer("cuda_driver_api", drv).integer("cuda_runtime", rt);
    meta.str("gpu_max_clocks", sh("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem "
                                  "--format=csv,noheader 2>/dev/null"));
    meta.str("commit", getenv_or("BENCH_COMMIT", "unknown"));
    meta.str("env", getenv_or("BENCH_ENV", "unknown"));
    const char* ql = std::getenv("QUIET_LOG");
    meta.str("conditions", ql ? read_file(ql) : "not recorded (not run via tools/run_bench.sh)");
  }

  // k4: -1 the plan-time rule, -2 the k4_tune plan option, >= 0 a forced configuration
  // (benchmark hook; needs the profiling build).
  void method_imd(int N, int vec_mode, int g, int k4 = -1) {
    auto t = [&](const Operand& x, std::vector<int32_t>& lab) {
      lab.assign(x.labels.begin(), x.labels.end());
      return imd_tensor_t{int32_t(x.labels.size()), x.extent.data(), x.stride.data(), lab.data()};
    };
    std::vector<int32_t> la, lb, lc;
    imd_tensor_t A = t(c.A, la), B = t(c.B, lb), C = t(c.C, lc);
    imd_options_t opt{N, g, IMD_RN, o.mode, vec_mode, k4 == -2 ? IMD_K4_TUNE_ON : IMD_K4_TUNE_OFF};
    size_t wsb = 0;
    CI(imd_workspace_size(&A, &B, &C, &opt, &wsb));
    imd_lowering_t low{};
    low.struct_size = sizeof low;
    imd_lowering(&A, &B, &C, &opt, &low);
    imd_plan_t plan = nullptr;
#ifdef IMD_BENCH_PROFILE
    if (imd_prof_set_k4_config(k4 >= 0 ? k4 : -1) != 0) throw Failure("imd_prof_set_k4_config");
#else
    if (k4 >= 0) throw Failure("--k4 <config> needs the profiling build");
#endif
    const auto c0 = std::chrono::steady_clock::now();
    const imd_status_t pst = imd_plan_create(imd, &A, &B, &C, &opt, &plan);
    const double create_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
#ifdef IMD_BENCH_PROFILE
    imd_prof_set_k4_config(-1);
#endif
    CI(pst);
    size_t plan_dev = 0;
    CI(imd_plan_device_bytes(plan, &plan_dev));
    // A resident plan must outlive in-flight work (its kernels read the plan's vectors).
    auto destroy = [](imd_plan_t p) { cudaDeviceSynchronize(); imd_plan_destroy(p); };
    std::unique_ptr<imd_plan_st, decltype(destroy)> guard(plan, destroy);
    DevBuf ws(wsb);
    poison_C();
    Timing T = time_it([&] {
      CI(imd_contract(plan, 1.0, dA.as<double>(), dB.as<double>(), 0.0, dC.as<double>(), ws.p, wsb, s));
    }, s, o, o.l2_flush ? std::function<void()>([&] { flush_l2(); }) : nullptr);
    Accuracy acc = result_accuracy();
    CU(cudaStreamSynchronize(s));
    Json extra;
    // C's bits after the timed calls (FNV-1a 64 over the whole C buffer): equal hashes across
    // builds = bit-identical results (v1.14 A/B).
    std::string c_hash;
    {
      std::vector<unsigned char> hc(dC.bytes);
      CU(cudaMemcpy(hc.data(), dC.p, dC.bytes, cudaMemcpyDeviceToHost));
      uint64_t h = 0xcbf29ce484222325ull;
      for (unsigned char b : hc) h = (h ^ b) * 0x100000001b3ull;
      char hs[17];
      snprintf(hs, sizeof hs, "%016llx", static_cast<unsigned long long>(h));
      c_hash = hs;
    }
    // DESIGN §6c: the predicted call (shipped calibration) for the K4 configuration the plan runs.
    int32_t k4_plan = -1;
#ifdef IMD_BENCH_PROFILE
    imd_prof_plan_k4(plan, &k4_plan);
#endif
    // v1.10: the bench's state (the same call repeated) at the reference clock; the default
    // eps fraction (the v6 bench passed 0 here by value-initialization).
    imd_perf_request_t preq{sizeof(imd_perf_request_t), 0, k4_plan, 0, -1.0,
                            IMD_PERF_REGIME_BURST, o.l2_flush ? IMD_PERF_CACHE_COLD : IMD_PERF_CACHE_REPEAT, 0.0};
    imd_perf_t pred{};
    pred.struct_size = sizeof pred;
    imd_perf_predict(&A, &B, &C, &opt, nullptr, &preq, nullptr, &pred);
#ifdef IMD_BENCH_PROFILE
    extra = profile_imd(plan, N, pred, [&] {
      CI(imd_contract(plan, 1.0, dA.as<double>(), dB.as<double>(), 0.0, dC.as<double>(), ws.p, wsb, s));
    });
#endif
    extra.integer("plan_device_bytes", (long long)plan_dev).num("plan_create_ms", 1e3 * create_s);
    extra.str("c_hash", c_hash);
    extra.num("pred_ms", 1e3 * pred.time_s).num("pred_err_pct", 100.0 * (pred.time_s / T.median() - 1));
    const std::string gs = (g == 0 || g >= N) ? "RN g=N" : "RN g=" + std::to_string(g);
    extra.integer("g", (long long)((g == 0 || g >= N) ? N : g));
    // A3: the K4 configuration of the rule (host query) and the one the plan runs.
    int32_t k4_run = -1;
#ifdef IMD_BENCH_PROFILE
    imd_prof_plan_k4(plan, &k4_run);
#endif
    extra.integer("k4_rule", low.k4_config).integer("k4_config", k4_run)
        .integer("k4_tune_bytes", (long long)low.k4_tune_bytes)
        .integer("fused", low.fused).integer("fused_large", low.fused_large)
        .integer("k_align", low.k_align).integer("k_hat", (long long)low.k_hat);
    std::string vs = vec_mode == IMD_VEC_RESIDENT ? gs + " vec=resident" : gs;
    if (o.mode == IMD_DIRECT) vs += " mode=direct";
    if (k4 == -2) vs += " k4=tune";
    else if (k4 >= 0) vs += " k4=" + std::to_string(k4);
    if (o.l2_flush) vs += " l2=flush";
    record("imd", vs, N, T, &acc, flops(), wsb, extra);
  }

  // Method duel: two immerdink configurations timed INTERLEAVED in one process, call by
  // call, so both see the same GPU clock / DVFS / host-core state (tiny calls drift between
  // performance states that last ms to s, docs/NOTES-A5.md §1.2; a sequential comparison of
  // two processes or two blocks of calls samples different states). Each rep times one call
  // of each side, each its own event pair after an idle GPU, alternating which goes first.
  // Side A writes C, side B a second C buffer of the same size; afterwards both are checked
  // against the reference and compared bit for bit. Records "duel-a" and "duel-b" (medians
  // as usual); duel-b also carries ratio_median = median over reps of t_a / t_b (> 1: B
  // faster) with p10 / p90. A = B measures the case's own resolution (ratio ~1 and its spread).
  struct DuelPlan {
    DuelSide cfg;
    std::vector<int32_t> la, lb, lc;
    imd_plan_t plan = nullptr;
    size_t wsb = 0, plan_dev = 0;
    DevBuf ws;
    imd_lowering_t low{};
    int32_t k4_run = -1;
    double create_ms = 0;
    ~DuelPlan() {
      if (plan) { cudaDeviceSynchronize(); imd_plan_destroy(plan); }  // a resident plan outlives its work
    }
  };

  std::unique_ptr<DuelPlan> make_duel_plan(const DuelSide& d) {
    auto P = std::make_unique<DuelPlan>();
    P->cfg = d;
#ifndef IMD_BENCH_PROFILE
    if (d.k4 >= 0 || d.k5 != 0) throw Failure("duel k4=<config> / k5=<elems> need the profiling build");
#endif
    auto t = [&](const Operand& x, std::vector<int32_t>& lab) {
      lab.assign(x.labels.begin(), x.labels.end());
      return imd_tensor_t{int32_t(x.labels.size()), x.extent.data(), x.stride.data(), lab.data()};
    };
    imd_tensor_t A = t(c.A, P->la), B = t(c.B, P->lb), C = t(c.C, P->lc);
    imd_options_t opt{d.N, d.g, IMD_RN, IMD_EMULATE, d.vec, d.k4 == -2 ? IMD_K4_TUNE_ON : IMD_K4_TUNE_OFF};
    // ka=: the K alignment of this side's sizing, lowering and plan (the host library's test
    // hook, resolved once at plan creation; the plan keeps its A), restored to the run's
    // --k-align afterwards. Times A = 16 against A = 64 interleaved in one process.
    if (d.ka >= 0 && imd_test_set_k_align(d.ka) != 0) throw Failure("imd_test_set_k_align");
    struct KaRestore {
      int run; bool on;
      ~KaRestore() { if (on) imd_test_set_k_align(run); }
    } ka_restore{o.k_align_hook, d.ka >= 0};
    CI(imd_workspace_size(&A, &B, &C, &opt, &P->wsb));
    P->low.struct_size = sizeof P->low;
    imd_lowering(&A, &B, &C, &opt, &P->low);
#ifdef IMD_BENCH_PROFILE
    if (imd_prof_set_k4_config(d.k4 >= 0 ? d.k4 : -1) != 0) throw Failure("imd_prof_set_k4_config");
#endif
    const auto c0 = std::chrono::steady_clock::now();
    const imd_status_t pst = imd_plan_create(imd, &A, &B, &C, &opt, &P->plan);
    P->create_ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
#ifdef IMD_BENCH_PROFILE
    imd_prof_set_k4_config(-1);
#endif
    CI(pst);
    CI(imd_plan_device_bytes(P->plan, &P->plan_dev));
#ifdef IMD_BENCH_PROFILE
    imd_prof_plan_k4(P->plan, &P->k4_run);
#endif
    P->ws = DevBuf(P->wsb);
    return P;
  }

  void method_duel(int N, const std::string& spec_a, const std::string& spec_b) {
    auto pa = make_duel_plan(parse_duel_side(spec_a, N));
    auto pb = make_duel_plan(parse_duel_side(spec_b, N));
    DevBuf dC2(dC.bytes);
    const bool k5_hook = pa->cfg.k5 != 0 || pb->cfg.k5 != 0;
    auto set_k5 = [&](const DuelPlan& p) {
#ifdef IMD_BENCH_PROFILE
      if (k5_hook && imd_prof_set_k5_elems(p.cfg.k5) != 0) throw Failure("imd_prof_set_k5_elems");
#else
      (void)p;
#endif
    };
    auto call = [&](const DuelPlan& p, double* C) {
      CI(imd_contract(p.plan, 1.0, dA.as<double>(), dB.as<double>(), 0.0, C, p.ws.p, p.wsb, s));
    };
    double* Ca = dC.as<double>();
    double* Cb = dC2.as<double>();
    for (int w = 0; w < std::max(o.warmup, 20); ++w) {
      set_k5(*pa); call(*pa, Ca);
      set_k5(*pb); call(*pb, Cb);
    }
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    auto once = [&](const DuelPlan& p, double* C) {
      set_k5(p);  // host-side switch, outside the timed window
      CU(cudaEventRecord(e0, s));
      call(p, C);
      CU(cudaEventRecord(e1, s));
      CU(cudaEventSynchronize(e1));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      return double(ms) * 1e-3;
    };
    Timing TA, TB;
    std::vector<double> ratio;
    double total = 0;
    for (int r = 0; r < o.reps; ++r) {
      double ta, tb;
      if (r % 2 == 0) { ta = once(*pa, Ca); tb = once(*pb, Cb); }
      else { tb = once(*pb, Cb); ta = once(*pa, Ca); }
      TA.t.push_back(ta);
      TB.t.push_back(tb);
      ratio.push_back(ta / tb);
      total += ta + tb;
      if (r + 1 >= o.min_reps && total > o.max_seconds) break;
    }
    CU(cudaGetLastError());
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(ratio.begin(), ratio.end());
    const size_t n = ratio.size();
    const double rmed = n % 2 ? ratio[n / 2] : 0.5 * (ratio[n / 2 - 1] + ratio[n / 2]);
    // Accuracy of each side from one fresh call into a poisoned C, then bit identity.
    CU(cudaMemset(Ca, 0xff, dC.bytes));
    CU(cudaMemset(Cb, 0xff, dC2.bytes));
    set_k5(*pa); call(*pa, Ca);
    set_k5(*pb); call(*pb, Cb);
#ifdef IMD_BENCH_PROFILE
    if (k5_hook) imd_prof_set_k5_elems(0);
#endif
    CU(cudaStreamSynchronize(s));
    const Accuracy acc_a = check(R, Ca, s), acc_b = check(R, Cb, s);
    DevBuf cnt(sizeof(unsigned long long));
    CU(cudaMemsetAsync(cnt.p, 0, cnt.bytes, s));
    k_count_diff<<<1024, 256, 0, s>>>(reinterpret_cast<const unsigned long long*>(Ca),
                                      reinterpret_cast<const unsigned long long*>(Cb), c.C.size,
                                      cnt.as<unsigned long long>());
    unsigned long long differ = 0;
    CU(cudaMemcpyAsync(&differ, cnt.p, sizeof differ, cudaMemcpyDeviceToHost, s));
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    auto side_extra = [&](const DuelPlan& p, const char* which) {
      Json e;
      e.str("duel_side", which).str("duel_config", p.cfg.describe());
      e.integer("plan_device_bytes", (long long)p.plan_dev).num("plan_create_ms", p.create_ms);
      e.integer("g", (long long)((p.cfg.g == 0 || p.cfg.g >= p.cfg.N) ? p.cfg.N : p.cfg.g));
      e.integer("k4_rule", p.low.k4_config).integer("k4_config", p.k4_run);
      e.integer("kernel_launches", p.low.kernel_launches);
      e.integer("fused", p.low.fused).integer("fused_large", p.low.fused_large);
      e.integer("k_align", p.low.k_align).integer("k_hat", (long long)p.low.k_hat);
      return e;
    };
    Json ea = side_extra(*pa, "a"), eb = side_extra(*pb, "b");
    eb.str("duel_vs", pa->cfg.describe());
    eb.num("ratio_median", rmed).num("ratio_p10", ratio[n / 10]).num("ratio_p90", ratio[(9 * n) / 10]);
    eb.raw("bit_identical", differ == 0 ? "true" : "false").integer("elements_differing", (long long)differ);
    record("duel-a", pa->cfg.describe() + ", interleaved", pa->cfg.N, TA, &acc_a, flops(), pa->wsb, ea);
    record("duel-b", pb->cfg.describe() + ", interleaved", pb->cfg.N, TB, &acc_b, flops(), pb->wsb, eb);
  }

  // v1.7 (DESIGN §6b, ROADMAP D1): one operand packed once (imd_pack_desc_for: the plain
  // call's orders, so C is bit-identical) and reused by R contractions. `which`: 0 = the
  // caller's A, 1 = B. Per R in --reuse, interleaved in each repetition (same clock state):
  //   plain   R x imd_contract
  //   packed  imd_pack + R x imd_contract_packed (same partner, same C)
  // and, timed the same way, imd_pack alone, one imd_contract and one imd_contract_packed.
  // One record per R: medians, the ratio plain / packed, the workspaces and the pack bytes;
  // the packed C is checked bit for bit against the plain one.
  void method_packed(int N, int which, const std::vector<int>& reuse) {
    auto t = [&](const Operand& x, std::vector<int32_t>& lab) {
      lab.assign(x.labels.begin(), x.labels.end());
      return imd_tensor_t{int32_t(x.labels.size()), x.extent.data(), x.stride.data(), lab.data()};
    };
    std::vector<int32_t> la, lb, lc;
    imd_tensor_t A = t(c.A, la), B = t(c.B, lb), C = t(c.C, lc);
    const imd_options_t opt{N, 0, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
    imd_pack_desc_t desc{};
    desc.struct_size = sizeof desc;
    int32_t labels[64];
    CI(imd_pack_desc_for(&A, &B, &C, &opt, which, &desc, labels, 64));
    imd_pack_t pack = nullptr;
    CI(imd_pack_create(imd, &desc, &pack));
    std::unique_ptr<imd_pack_s, imd_status_t (*)(imd_pack_t)> pguard(pack, imd_pack_destroy);
    size_t pb = 0, pw = 0, wsb = 0, wsp = 0;
    CI(imd_pack_bytes(pack, &pb, &pw));
    const imd_packing_t packing = {which == 0 ? &desc : nullptr, which == 1 ? &desc : nullptr};
    CI(imd_workspace_size(&A, &B, &C, &opt, &wsb));
    CI(imd_workspace_size_packed(&A, &B, &C, &opt, &packing, &wsp));
    imd_lowering_t lu{}, lp{};
    lu.struct_size = lp.struct_size = sizeof lu;
    imd_lowering(&A, &B, &C, &opt, &lu);
    imd_lowering_packed(&A, &B, &C, &opt, &packing, &lp);
    imd_plan_t plain = nullptr, packed = nullptr;
    CI(imd_plan_create(imd, &A, &B, &C, &opt, &plain));
    std::unique_ptr<imd_plan_st, imd_status_t (*)(imd_plan_t)> g1(plain, imd_plan_destroy);
    CI(imd_plan_create_packed(imd, &A, &B, &C, &opt, which == 0 ? pack : nullptr,
                              which == 1 ? pack : nullptr, &packed));
    std::unique_ptr<imd_plan_st, imd_status_t (*)(imd_plan_t)> g2(packed, imd_plan_destroy);
    DevBuf ws(wsb), wsP(wsp), P(pb), PW(pw);
    const double* X = which == 0 ? dA.as<double>() : dB.as<double>();
    const imd_operand_t opX{X, P.p, pb, pack};
    const imd_operand_t opO{which == 0 ? dB.as<double>() : dA.as<double>(), nullptr, 0, nullptr};
    const imd_operand_t& oA = which == 0 ? opX : opO;
    const imd_operand_t& oB = which == 0 ? opO : opX;
    auto do_pack = [&] { CI(imd_pack(pack, X, P.p, pb, PW.p, pw, s)); };
    auto do_plain = [&] {
      CI(imd_contract(plain, 1.0, dA.as<double>(), dB.as<double>(), 0.0, dC.as<double>(), ws.p, wsb, s));
    };
    auto do_packed = [&] {
      CI(imd_contract_packed(packed, 1.0, &oA, &oB, 0.0, dC.as<double>(), wsP.p, wsp, s));
    };
    // bit-identity of the two paths (imd_pack_desc_for orders)
    poison_C();
    do_plain();
    CU(cudaStreamSynchronize(s));
    std::vector<double> ref(c.C.size), got(c.C.size);
    CU(cudaMemcpy(ref.data(), dC.p, c.C.size * 8, cudaMemcpyDeviceToHost));
    poison_C();
    do_pack();
    do_packed();
    CU(cudaStreamSynchronize(s));
    CU(cudaMemcpy(got.data(), dC.p, c.C.size * 8, cudaMemcpyDeviceToHost));
    const bool identical = std::memcmp(ref.data(), got.data(), c.C.size * 8) == 0;
    Accuracy acc = result_accuracy();
    // Timed units, interleaved per repetition.
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    auto timed = [&](const std::function<void()>& f) {
      CU(cudaEventRecord(e0, s));
      f();
      CU(cudaEventRecord(e1, s));
      CU(cudaEventSynchronize(e1));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      return double(ms) * 1e-3;
    };
    const int Rmax = *std::max_element(reuse.begin(), reuse.end());
    for (int w = 0; w < o.warmup; ++w) { do_pack(); do_plain(); do_packed(); }
    CU(cudaStreamSynchronize(s));
    Timing tp, tc, tcp;
    std::map<int, Timing> tplain, tpacked;
    double total = 0;
    for (int r = 0; r < o.reps; ++r) {
      tp.t.push_back(timed(do_pack));
      tc.t.push_back(timed(do_plain));
      tcp.t.push_back(timed(do_packed));
      for (int R : reuse) {
        const double a = timed([&] { for (int i = 0; i < R; ++i) do_plain(); });
        const double b = timed([&] { do_pack(); for (int i = 0; i < R; ++i) do_packed(); });
        tplain[R].t.push_back(a);
        tpacked[R].t.push_back(b);
        total += a + b;
      }
      if (r + 1 >= o.min_reps && total > o.max_seconds) break;
    }
    CU(cudaGetLastError());
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    (void)Rmax;
    // Per-phase event marks (profiling build): one plain call, one packed call, imd_pack.
    Json phases;
#ifdef IMD_BENCH_PROFILE
    if (o.phase_reps > 0) {
      const std::pair<const char*, std::function<void()>> runs[3] = {
          {"plain_", do_plain}, {"packed_", do_packed}, {"pack_", do_pack}};
      imd_perf_t preds[3] = {};
      for (auto& pr : preds) pr.struct_size = sizeof pr;
      const imd_perf_request_t rq{sizeof(imd_perf_request_t), 0, -1, 0, -1.0,
                                  IMD_PERF_REGIME_BURST, IMD_PERF_CACHE_REPEAT, 0.0};
      imd_perf_predict(&A, &B, &C, &opt, nullptr, &rq, nullptr, &preds[0]);
      imd_perf_predict(&A, &B, &C, &opt, &packing, &rq, nullptr, &preds[1]);
      imd_perf_predict_pack(&desc, &rq, nullptr, &preds[2]);
      int ri = 0;
      for (auto& [pre, fn] : runs) {
        fprintf(stderr, "  %sphases:\n", pre);
        Json pj = profile_imd(std::string(pre) == "plain_" ? plain : packed, N, preds[ri++], fn);
        for (auto& kv : pj.kv) phases.raw(pre + kv.first, kv.second);
      }
    }
#endif
    for (int R : reuse) {
      Json e;
      if (R == reuse.front())
        for (auto& kv : phases.kv) e.kv.push_back(kv);
      e.str("pack", which == 0 ? "A" : "B").integer("R", R);
      e.num("plain_R_ms", 1e3 * tplain[R].median()).num("packed_R_ms", 1e3 * tpacked[R].median());
      e.num("speedup_R", tplain[R].median() / tpacked[R].median());
      e.num("pack_ms", 1e3 * tp.median()).num("contract_ms", 1e3 * tc.median());
      e.num("contract_packed_ms", 1e3 * tcp.median());
      e.num("pack_vs_contract", tp.median() / tc.median());
      e.num("contract_packed_vs_contract", tcp.median() / tc.median());
      e.integer("ws_bytes_plain", (long long)wsb).integer("ws_bytes_packed", (long long)wsp);
      e.integer("pack_bytes", (long long)pb).integer("pack_ws_bytes", (long long)pw);
      e.integer("kernel_launches_plain", lu.kernel_launches).integer("kernel_launches_packed", lp.kernel_launches);
      e.integer("fused", lp.fused).integer("fused_large", lp.fused_large).integer("swapped", lp.swapped);
      e.raw("bit_identical", identical ? "true" : "false");
      record("packed", std::string("pack ") + (which == 0 ? "A" : "B") + " R=" + std::to_string(R), N,
             tpacked[R], &acc, R * flops(), wsp, e);
    }
    CU(cudaStreamSynchronize(s));
  }

#ifdef IMD_BENCH_PROFILE
  // Per-phase times from the library's event marks, against the MEMORY_MODEL §4 lower bounds
  // and the performance model's predicted phase times (DESIGN §6c: `pred` is imd_perf_predict /
  // imd_perf_predict_pack with the shipped calibration, the one implementation of the bound
  // formulas: the actual g, beta = 0, rho = 1, datasheet peaks).
  Json profile_imd(imd_plan_t plan, int N, const imd_perf_t& pred,
                   const std::function<void()>& call) {
    int64_t d[7];
    CI(imd_prof_plan_dims(plan, d) == 0 ? IMD_STATUS_SUCCESS : IMD_STATUS_INVALID_VALUE);
    std::map<std::string, double> lb, pm;  // phase -> bound, predicted time (s)
    for (int i = 0; i < pred.num_phases; ++i) {
      const char* nm = imd_perf_phase_name(pred.phase[i].phase);
      if (nm == nullptr) continue;
      lb[nm] = pred.phase[i].bound_s;
      pm[nm] = pred.phase[i].time_s;
    }

    if (imd_prof_enable(1) != 0) throw Failure("imd_prof_enable failed");
    std::map<std::string, std::vector<double>> per;  // phase -> per-call total ms
    std::map<std::string, int> launches;
    std::vector<std::string> order;
    std::vector<double> totals;
    for (int r = 0; r < o.phase_reps; ++r) {
      imd_prof_reset();
      call();
      const char* tags[4096];
      float ms[4096];
      const int cnt = imd_prof_read(4096, tags, ms);
      if (cnt < 0) throw Failure("imd_prof_read failed");
      std::map<std::string, double> sum;
      std::map<std::string, int> num;
      double tot = 0;
      for (int i = 0; i < cnt; ++i) {
        std::string t = tags[i];
        if (t.rfind("K4 gemm", 0) == 0) t = "K4 gemm";
        if (!sum.count(t) && std::find(order.begin(), order.end(), t) == order.end()) order.push_back(t);
        sum[t] += ms[i];
        num[t] += 1;
        tot += ms[i];
      }
      for (auto& [t, v] : sum) per[t].push_back(v);
      launches = num;
      totals.push_back(tot);
    }
    imd_prof_enable(0);
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v.empty() ? NAN : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
    };
    std::string js = "{";
    double lb_sum = 0;
    fprintf(stderr, "  phases (N=%d, m=%lld n=%lld k=%lld b=%lld):\n", N, (long long)d[0],
            (long long)d[1], (long long)d[2], (long long)d[3]);
    for (auto& t : order) {
      const double t_ms = median(per[t]);
      const double l_ms = lb.count(t) ? lb[t] * 1e3 : NAN;
      if (lb.count(t) && t != "K0 vec") lb_sum += l_ms;
      const double p_ms = pm.count(t) ? pm[t] * 1e3 : NAN;
      Json p;
      p.num("ms", t_ms).num("lb_ms", l_ms).num("pct_of_lb", 100.0 * l_ms / t_ms).integer("marks", launches[t]);
      p.num("pred_ms", p_ms);
      js += (js.size() > 1 ? ", " : "") + jstr(t) + ": " + p.dump();
      fprintf(stderr, "    %-14s %9.3f ms  LB %9.3f ms  %5.1f %% of LB  pred %9.3f ms  (%d marks)\n",
              t.c_str(), t_ms, l_ms, 100.0 * l_ms / t_ms, p_ms, launches[t]);
    }
    js += "}";
    const double tot = median(totals);
    fprintf(stderr, "    %-14s %9.3f ms  LB %9.3f ms  %5.1f %% of LB (excl. K0)\n", "total", tot, lb_sum,
            100.0 * lb_sum / tot);
    Json e;
    e.raw("phases", js).num("phase_total_ms", tot).num("phase_lb_ms", lb_sum).integer("phase_reps", o.phase_reps);
    e.str("phase_view", std::to_string(d[0]) + "x" + std::to_string(d[1]) + "x" + std::to_string(d[2]) +
                            " b=" + std::to_string(d[3]) + " n_hat=" + std::to_string(d[4]) +
                            " k_hat=" + std::to_string(d[5]));
    return e;
  }
#endif

  // ---- v1.10 (DESIGN §6c.1, §6c.2): calibration of occupancy, sustained clock / power, cache

  // An immerdink plan of this case (the rule's K4 configuration), with its workspace.
  struct SimplePlan {
    std::vector<int32_t> la, lb, lc;
    imd_tensor_t A{}, B{}, C{};
    imd_options_t opt{};
    imd_plan_t plan = nullptr;
    size_t wsb = 0;
    DevBuf ws;
    ~SimplePlan() {
      if (plan) { cudaDeviceSynchronize(); imd_plan_destroy(plan); }
    }
  };
  std::unique_ptr<SimplePlan> make_plan(int N, int vec_mode, int g) {
    auto P = std::make_unique<SimplePlan>();
    auto t = [&](const Operand& x, std::vector<int32_t>& lab) {
      lab.assign(x.labels.begin(), x.labels.end());
      return imd_tensor_t{int32_t(x.labels.size()), x.extent.data(), x.stride.data(), lab.data()};
    };
    P->A = t(c.A, P->la), P->B = t(c.B, P->lb), P->C = t(c.C, P->lc);
    P->opt = imd_options_t{N, g, IMD_RN, o.mode, vec_mode, IMD_K4_TUNE_OFF};
    CI(imd_workspace_size(&P->A, &P->B, &P->C, &P->opt, &P->wsb));
    CI(imd_plan_create(imd, &P->A, &P->B, &P->C, &P->opt, &P->plan));
    P->ws = DevBuf(P->wsb);
    return P;
  }

  // Method occupancy: the measured CTAs per SM, registers, shared memory and threads of every
  // kernel (IMD_PERF_KERNEL_*), for the calibration file's kernel.<name> lines. No timing.
  void method_occupancy() {
#ifdef IMD_BENCH_PROFILE
    std::string js = "{";
    for (int k = 0; k < IMD_PERF_KERNEL_COUNT; ++k) {
      int32_t v[4] = {0, 0, 0, 0};
      if (imd_prof_kernel_occupancy(k, v) != 0) throw Failure("imd_prof_kernel_occupancy");
      js += (k ? ", " : "") + jstr(imd_perf_kernel_name(k)) + ": [" + std::to_string(v[0]) + ", " +
            std::to_string(v[1]) + ", " + std::to_string(v[2]) + ", " + std::to_string(v[3]) + "]";
    }
    Json j;
    j.str("method", "occupancy").str("variant", "").integer("N", 0).raw("kernels", js + "}");
    j.str("status", "ok");
    emit(j);
#else
    throw Failure("--methods occupancy needs the profiling build");
#endif
  }

  // Method sustained: one plan back to back for --sustain-s seconds after --sustain-warm-s of
  // the same calls, queued ahead (two chunks of >= 20 ms in flight: the GPU never waits for
  // the host). Records the mean time per call and the mean board power and SM clock over the
  // window (NVML, sampled every 20 ms in a host thread; the energy counter is not read).
  void method_sustained(int N, int vec_mode, int g) {
    auto P = make_plan(N, vec_mode, g);
    Nvml nv;
    if (!nv.ok()) throw Failure("NVML not available (libnvidia-ml.so.1)");
    auto call = [&] {
      CI(imd_contract(P->plan, 1.0, dA.as<double>(), dB.as<double>(), 0.0, dC.as<double>(), P->ws.p,
                      P->wsb, s));
    };
    for (int w = 0; w < 3; ++w) call();
    CU(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1, chunk_ev[2];
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    for (auto& e : chunk_ev) CU(cudaEventCreate(&e));
    CU(cudaEventRecord(e0, s));
    for (int w = 0; w < 5; ++w) call();
    CU(cudaEventRecord(e1, s));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    const double t_est = std::max(1e-6, ms * 1e-3 / 5);
    const long chunk = std::max(1L, long(std::ceil(0.02 / t_est)));
    int cur = 0;
    // Trace (--sustain-trace-s): host time and calls completed at every chunk boundary of the
    // window (a chunk's event has completed when the host returns from its synchronize).
    using clk = std::chrono::steady_clock;
    clk::time_point tw0;
    bool tracing = false;
    std::vector<std::pair<double, long>> done;  // (s since window start, calls completed)
    // Chunks until `seconds` of host time have passed; returns the calls issued.
    auto run_for = [&](double seconds) {
      long calls = 0;
      const auto t0 = std::chrono::steady_clock::now();
      bool first = true;
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        for (long i = 0; i < chunk; ++i) call();
        calls += chunk;
        CU(cudaEventRecord(chunk_ev[cur], s));
        if (!first) {
          CU(cudaEventSynchronize(chunk_ev[cur ^ 1]));  // two chunks in flight
          if (tracing)
            done.emplace_back(std::chrono::duration<double>(clk::now() - tw0).count(), calls - chunk);
        }
        first = false;
        cur ^= 1;
      }
      return calls;
    };
    run_for(o.sustain_warm_s);
    std::atomic<bool> stop{false};
    std::vector<double> pw, mhz, ts, tc, fanv;
    std::vector<unsigned long long> rs;
    const bool trace = o.sustain_trace_s > 0;
    const double temp0 = nv.temp_c();
    tw0 = clk::now();
    tracing = trace;
    std::thread sampler([&] {
      int i = 0;
      while (!stop.load()) {
        pw.push_back(nv.power_w());
        mhz.push_back(nv.sm_mhz());
        if (trace) {
          ts.push_back(std::chrono::duration<double>(clk::now() - tw0).count());
          rs.push_back(nv.reason_bits());
          // temperature and fan change slowly: every 10th sample (200 ms)
          tc.push_back(i % 10 == 0 ? nv.temp_c() : NAN);
          fanv.push_back(i % 10 == 0 ? nv.fan_pct() : NAN);
        }
        ++i;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
    CU(cudaEventRecord(e0, s));
    const long calls = run_for(o.sustain_s);
    CU(cudaEventRecord(e1, s));
    CU(cudaEventSynchronize(e1));
    stop = true;
    sampler.join();
    CU(cudaEventElapsedTime(&ms, e0, e1));
    CU(cudaGetLastError());
    for (auto e : {e0, e1, chunk_ev[0], chunk_ev[1]}) cudaEventDestroy(e);
    auto mean = [](const std::vector<double>& v) {
      double a = 0;
      int n = 0;
      for (double x : v)
        if (std::isfinite(x)) a += x, ++n;
      return n ? a / n : NAN;
    };
    Timing T;
    T.t.push_back(ms * 1e-3 / double(calls));  // the back-to-back mean time per call
    Json e;
    e.num("window_s", ms * 1e-3).integer("calls", calls).integer("chunk", chunk);
    e.num("power_w_mean", mean(pw)).num("sm_mhz_mean", mean(mhz));
    e.integer("nvml_samples", (long long)pw.size());
    e.num("power_limit_w", nv.limit_w()).num("sustain_warm_s", o.sustain_warm_s);
    e.num("temp_c_start", temp0).num("temp_c_end", nv.temp_c());
    if (trace) {
      // One point per trace_s of the window: [t_mid s, s per call (completed calls between the
      // chunk boundaries nearest the bin edges), mean W, mean MHz, mean degC, OR of the clock-
      // event reason bits, mean fan %]. The energy counter is never read.
      std::string arr = "[";
      size_t j0 = 0;
      for (double b0 = 0; b0 < ms * 1e-3; b0 += o.sustain_trace_s) {
        const double b1 = b0 + o.sustain_trace_s;
        size_t j1 = j0;
        while (j1 + 1 < done.size() && done[j1 + 1].first <= b1) ++j1;
        double spc = NAN;
        if (j1 > j0 && done[j1].second > done[j0].second)
          spc = (done[j1].first - done[j0].first) / double(done[j1].second - done[j0].second);
        double sp = 0, sf = 0, st = 0, sfan = 0;
        int np = 0, nt = 0;
        unsigned long long rb = 0;
        for (size_t i = 0; i < ts.size() && i < pw.size(); ++i) {
          if (ts[i] < b0 || ts[i] >= b1) continue;
          if (std::isfinite(pw[i]) && std::isfinite(mhz[i])) sp += pw[i], sf += mhz[i], ++np;
          if (std::isfinite(tc[i])) st += tc[i], sfan += fanv[i], ++nt;
          rb |= rs[i];
        }
        char buf[256];
        snprintf(buf, sizeof buf, "%s[%.3f, %s, %s, %s, %s, %llu, %s]", arr.size() > 1 ? ", " : "",
                 0.5 * (b0 + b1), jnum(spc).c_str(), jnum(np ? sp / np : NAN).c_str(),
                 jnum(np ? sf / np : NAN).c_str(), jnum(nt ? st / nt : NAN).c_str(), rb,
                 jnum(nt ? sfan / nt : NAN).c_str());
        arr += buf;
        j0 = j1;
      }
      e.raw("trace", arr + "]").str("trace_cols", "t_s,s_per_call,power_w,sm_mhz,temp_c,reasons,fan_pct");
    }
    e.integer("g", (long long)((g == 0 || g >= N) ? N : g));
    std::string vs = (g == 0 || g >= N) ? "RN g=N" : "RN g=" + std::to_string(g);
    if (vec_mode == IMD_VEC_RESIDENT) vs += " vec=resident";
    record("sustained", vs + " sustained", N, T, nullptr, flops(), P->wsb, e);
  }

  // Method pair (cache state): C = A B^T with n = k, so C of one call is a valid A of the next
  // (the same plan). Per repetition, interleaved: COLD (L2 evicted, then the call reading C1
  // as its A), WARM (L2 evicted, call 1 writes C1, then the call reading C1) and REPEAT (the
  // same call again right after). Three records: cache = cold / warm / repeat.
  void method_pair(int N) {
    if (c.A.size != c.C.size || c.A.labels.size() != 2 || c.C.labels.size() != 2 || c.n != c.k)
      throw Failure("method pair needs ik,jk->ij with j = k (C of one call is A of the next)");
    auto P = make_plan(N, IMD_VEC_GENERATE, 0);
    DevBuf C1(dC.bytes);
    auto call = [&](const double* A, double* C) {
      CI(imd_contract(P->plan, 1.0, A, dB.as<double>(), 0.0, C, P->ws.p, P->wsb, s));
    };
    call(dA.as<double>(), C1.as<double>());  // C1: a valid A
    for (int w = 0; w < std::max(o.warmup, 3); ++w) call(C1.as<double>(), dC.as<double>());
    CU(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    auto timed = [&](const std::function<void()>& f) {
      CU(cudaEventRecord(e0, s));
      f();
      CU(cudaEventRecord(e1, s));
      CU(cudaEventSynchronize(e1));
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      return double(ms) * 1e-3;
    };
    Timing tc, tw, tr;
    double total = 0;
    auto second = [&] { call(C1.as<double>(), dC.as<double>()); };
    for (int r = 0; r < o.reps; ++r) {
      flush_l2();
      tc.t.push_back(timed(second));
      flush_l2();
      call(dA.as<double>(), C1.as<double>());
      tw.t.push_back(timed(second));
      tr.t.push_back(timed(second));
      total += tc.t.back() + tw.t.back() + tr.t.back();
      if (r + 1 >= o.min_reps && total > o.max_seconds) break;
    }
    CU(cudaGetLastError());
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const std::pair<const char*, Timing*> states[3] = {{"cold", &tc}, {"warm", &tw}, {"repeat", &tr}};
    for (const auto& [nm, T] : states) {
      Json e;
      e.str("cache", nm).integer("g", N);
      record("pair", std::string("RN g=N cache=") + nm, N, *T, nullptr, flops(), P->wsb, e);
    }
  }

  void method_bw() {
    DevBuf dst(dA.bytes);
    Timing T = time_it([&] { CU(cudaMemcpyAsync(dst.p, dA.p, dA.bytes, cudaMemcpyDeviceToDevice, s)); }, s, o);
    Json e;
    e.num("gbps", 2.0 * double(dA.bytes) / T.median() / 1e9).integer("bytes", (long long)dA.bytes);
    record("bw", "D2D cudaMemcpy of A (read+write)", 0, T, nullptr, NAN, 0, e);
  }
};

void usage() {
  fprintf(stderr,
          "usage: imd_bench --name NAME --spec 'ab,bc->ac' --ext a=..,b=..,c=.. [--methods LIST]\n"
          "  [--N 8,14,20] [--reps 10] [--warmup 2]\n"
          "  [--max-seconds 15] [--samples 256] [--phi 0.5] [--seed 1] [--phase-reps 5]\n"
          "  [--vec generate,resident] [--g 1,2,4,0 (0 = N)] [--k4 rule,tune,0,1,2,3]\n"
          "  [--mode emulate|direct] (method imd) [--data normal|dominant]\n"
          "  [--pack A,B] [--reuse 1,4,16] (method packed)\n"
          "  [--duel-a default] [--duel-b k4=tune] (method duel; a side: default or a comma\n"
          "      list of N=, vec=generate|resident, g=, k4=rule|tune|0..3, k5=0|1|2|4, ka=auto|16|64)\n"
          "  [--clock-probe 0|1] [--l2 repeat|flush] (v1.10: per-call SM clock; L2 evicted\n"
          "      before each timed imd call) [--sustain-s 8] [--sustain-warm-s 3] (sustained)\n"
          "  [--sustain-trace-s 0] (sustained: time series every T s of the window)\n"
          "  [--k-align auto|16|64] (v1.13 §4b: force the K alignment of every plan / pack)\n"
          "  methods: imd,packed,duel,bw,occupancy,sustained,pair (default imd)\n"
          "  [--out file.jsonl]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a{{"methods", "imd"},
                                       {"N", "14"},
                                       {"duel-a", "default"},
                                       {"duel-b", "k4=tune"}};
  static const char* const kKeys[] = {"name", "spec", "ext", "methods", "N", "reps", "warmup",
                                      "max-seconds", "samples", "phi", "seed", "phase-reps",
                                      "vec", "g", "k4", "pack", "reuse", "duel-a", "duel-b", "out",
                                      "mode", "data", "clock-probe", "l2", "sustain-s",
                                      "sustain-warm-s", "sustain-trace-s", "k-align"};
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) { usage(); return 2; }
    k = k.substr(2);
    if (std::find(std::begin(kKeys), std::end(kKeys), k) == std::end(kKeys)) {
      fprintf(stderr, "unknown option --%s\n", k.c_str());
      usage();
      return 2;
    }
    a[k] = argv[++i];
  }
  for (auto& m : split(a["methods"], ','))
    if (m != "imd" && m != "packed" && m != "duel" && m != "bw" && m != "occupancy" &&
        m != "sustained" && m != "pair") {
      fprintf(stderr, "unknown method %s (immerdink only: imd, packed, duel, bw, occupancy, "
                      "sustained, pair)\n", m.c_str());
      return 2;
    }
  if (!a.count("name") || !a.count("spec") || !a.count("ext")) { usage(); return 2; }
  Bench B;
  try {
    B.c = parse_case(a["name"], a["spec"], a["ext"]);
    if (a.count("reps")) B.o.reps = std::stoi(a["reps"]);
    if (a.count("warmup")) B.o.warmup = std::stoi(a["warmup"]);
    if (a.count("max-seconds")) B.o.max_seconds = std::stod(a["max-seconds"]);
    if (a.count("samples")) B.o.samples = std::stoll(a["samples"]);
    if (a.count("phase-reps")) B.o.phase_reps = std::stoi(a["phase-reps"]);
    if (a.count("vec")) {
      B.o.vec_modes.clear();
      for (auto& v : split(a["vec"], ','))
        B.o.vec_modes.push_back(v == "resident" ? IMD_VEC_RESIDENT : IMD_VEC_GENERATE);
    }
    if (a.count("g")) {
      B.o.groups.clear();
      for (auto& v : split(a["g"], ',')) B.o.groups.push_back(std::stoi(v));
    }
    if (a.count("k4")) {
      B.o.k4.clear();
      for (auto& v : split(a["k4"], ','))
        B.o.k4.push_back(v == "rule" ? -1 : v == "tune" ? -2 : std::stoi(v));
    }
    if (a.count("mode")) {
      if (a["mode"] != "emulate" && a["mode"] != "direct") throw Failure("--mode emulate|direct");
      B.o.mode = a["mode"] == "direct" ? IMD_DIRECT : IMD_EMULATE;
    }
    if (a.count("data")) {
      if (a["data"] != "normal" && a["data"] != "dominant") throw Failure("--data normal|dominant");
      B.o.dominant = a["data"] == "dominant";
    }
    if (a.count("phi")) B.o.phi = std::stod(a["phi"]);
    if (a.count("clock-probe")) B.o.clock_probe = a["clock-probe"] == "1";
    if (a.count("l2")) {
      if (a["l2"] != "flush" && a["l2"] != "repeat") throw Failure("--l2 flush|repeat");
      B.o.l2_flush = a["l2"] == "flush";
    }
    if (a.count("sustain-s")) B.o.sustain_s = std::stod(a["sustain-s"]);
    if (a.count("sustain-warm-s")) B.o.sustain_warm_s = std::stod(a["sustain-warm-s"]);
    if (a.count("sustain-trace-s")) B.o.sustain_trace_s = std::stod(a["sustain-trace-s"]);
    if (a.count("seed")) B.o.seed = std::stoull(a["seed"]);
    // v1.13 §4b: --k-align, else IMD_BENCH_K_ALIGN (tools/calibrate.sh --k-align), set
    // before any plan or pack is created.
    if (!a.count("k-align") && getenv("IMD_BENCH_K_ALIGN")) a["k-align"] = getenv("IMD_BENCH_K_ALIGN");
    if (a.count("k-align")) {
      const std::string& v = a["k-align"];
      if (v != "auto" && v != "0" && v != "16" && v != "64") throw Failure("--k-align auto|16|64");
      B.o.k_align_hook = v == "auto" ? 0 : std::stoi(v);
      if (imd_test_set_k_align(B.o.k_align_hook) != 0) throw Failure("imd_test_set_k_align");
    }
    B.o.min_reps = std::min(B.o.min_reps, B.o.reps);
    if (a.count("out")) {
      B.out = fopen(a["out"].c_str(), "a");
      if (!B.out) { perror(a["out"].c_str()); return 2; }
    }
    fprintf(stderr, "%s: %s  m=%lld n=%lld k=%lld b=%lld\n", B.c.name.c_str(), B.c.spec.c_str(),
            (long long)B.c.m, (long long)B.c.n, (long long)B.c.k, (long long)B.c.b);
    B.setup();
  } catch (const std::exception& e) {
    fprintf(stderr, "setup failed: %s\n", e.what());
    return 1;
  }
  std::vector<int> Ns;
  for (auto& x : split(a["N"], ',')) Ns.push_back(std::stoi(x));
  for (auto& m : split(a["methods"], ',')) {
    auto guarded = [&](const std::string& name, const std::string& variant, int N, auto&& fn) {
      try {
        fn();
      } catch (const std::exception& e) {
        cudaGetLastError();  // clear a sticky-free error so later methods can run
        B.record_error(name, variant, N, e.what());
      }
      cudaDeviceSynchronize();
    };
    if (m == "imd") {  // braces: the else-if chain below must not bind to the inner if
      for (int N : Ns)
        for (int v : B.o.vec_modes)
          for (int g : B.o.groups)
            for (int k4 : B.o.k4)
              if (g <= N) guarded(m, "", N, [&] { B.method_imd(N, v, g, k4); });
    } else if (m == "bw") guarded(m, "", 0, [&] { B.method_bw(); });
    else if (m == "occupancy") guarded(m, "", 0, [&] { B.method_occupancy(); });
    else if (m == "sustained") {
      for (int N : Ns)
        for (int v : B.o.vec_modes)
          for (int g : B.o.groups)
            if (g <= N) guarded(m, "", N, [&] { B.method_sustained(N, v, g); });
    } else if (m == "pair")
      for (int N : Ns) guarded(m, "", N, [&] { B.method_pair(N); });
    else if (m == "duel")
      for (int N : Ns) guarded(m, "", N, [&] { B.method_duel(N, a["duel-a"], a["duel-b"]); });
    else if (m == "packed") {  // v1.7 (D1): --pack A,B --reuse 1,4,16
      std::vector<int> reuse;
      for (auto& v : split(a.count("reuse") ? a["reuse"] : "1,4,16", ',')) reuse.push_back(std::stoi(v));
      for (int N : Ns)
        for (auto& w : split(a.count("pack") ? a["pack"] : "A", ','))
          guarded(m, "pack " + w, N, [&] { B.method_packed(N, w == "B" ? 1 : 0, reuse); });
    }
  }
  if (B.out != stdout) fclose(B.out);
  return 0;
}
