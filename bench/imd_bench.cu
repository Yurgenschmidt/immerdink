// imd_bench: time one FP64 contraction case with immerdink and the M4 baselines.
//
//   imd_bench --name sq4096 --spec ik,jk->ij --ext i=4096,j=4096,k=4096 \
//             --methods dgemm,emu,int8,gemmul8,cutensor,imd --N 8,14,20 [options]
//
// Run it through bench/run_sweep.sh (which goes through tools/run_bench.sh); see
// docs/BENCHMARKS.md. Every operand is dense row-major in its label order (last label
// fastest). One JSON object per (method, variant) is appended to --out (default stdout).
//
// Methods
//   dgemm     cuBLAS FP64 GEMM, CUBLAS_COMPUTE_64F_PEDANTIC (never emulated)
//   emu       cuBLAS FP64 fixed-point emulation, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
//             strategy EAGER; variants from --emu (dyn | fix<bits>)
//   int8      cuBLASLt int8 TN GEMM with int32 output on the GEMM view (m, n, k^, b):
//             the time of ONE modulus' GEMM (the GEMM-phase roofline is N times it)
//   gemmul8   GEMMul8 INT8 fast mode, one call per batch, N from --N
//   cutensor  cuTENSOR FP64 contraction on the original layouts
//   imd       immerdink imd_plan_create + imd_contract, N from --N
//   bw        device-to-device copy of A (achievable DRAM bandwidth)
// Matrix-only methods (dgemm, emu, gemmul8) need the GEMM view: A = [L][I|K], B = [L][J|K],
// C = [L][I][J] (either I/K or J/K order). If a case is not in that layout the method is
// timed twice: on pre-permuted operands ("gemm-view") and as "perm+<method>", which
// permutes A and B in, runs the GEMM and permutes C out (cuTENSOR permutations).
//
// Accuracy: for --samples output elements (all of them if C is smaller), the reference
// is a double-double (Dot2) dot product computed on the device; the host holds only the
// samples (no operand copies), so host RSS is small for every shape. Reported: max |c - r| / |r| and the
// normwise max |c - r| / (||a_i||_2 ||b_j||_2) of DESIGN §3.8.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <sys/resource.h>
#include <immerdink/immerdink.h>

#ifdef IMD_BENCH_GEMMUL8
#include <gemmul8.hpp>
#endif

#include <algorithm>
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
#include <stdexcept>
#include <string>
#include <vector>

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
#define CB(x)                                                                                \
  do {                                                                                       \
    cublasStatus_t s_ = (x);                                                                 \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                         \
      throw Failure(std::string(#x) + ": " + cublasGetStatusName(s_));                       \
  } while (0)
#define CT(x)                                                                                \
  do {                                                                                       \
    cutensorStatus_t s_ = (x);                                                               \
    if (s_ != CUTENSOR_STATUS_SUCCESS)                                                       \
      throw Failure(std::string(#x) + ": " + cutensorGetErrorString(s_));                    \
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
__global__ void k_fill(double* p, size_t n, uint64_t seed, double phi) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    uint64_t h1 = splitmix64(seed ^ (2 * i)), h2 = splitmix64(seed ^ (2 * i + 1));
    double u = double(h1 >> 11) * 0x1.0p-53, u1 = (double(h2 >> 11) + 1.0) * 0x1.0p-53;
    double u2 = double(splitmix64(h2) >> 11) * 0x1.0p-53;
    double g = sqrt(-2.0 * log(u1)) * cospi(2.0 * u2);
    p[i] = (u - 0.5) * exp(phi * g);
  }
}
__global__ void k_fill_i8(int8_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = int8_t(splitmix64(seed ^ i) & 0xff);
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
  double phi = 0.5;
  uint64_t seed = 1;
  std::string out;
};

Timing time_it(const std::function<void()>& run, cudaStream_t s, const Options& o) {
  for (int w = 0; w < o.warmup; ++w) run();
  CU(cudaStreamSynchronize(s));
  CU(cudaGetLastError());
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  Timing T;
  double total = 0;
  for (int r = 0; r < o.reps; ++r) {
    CU(cudaEventRecord(e0, s));
    run();
    CU(cudaEventRecord(e1, s));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    T.t.push_back(ms * 1e-3);
    total += ms * 1e-3;
    if (r + 1 >= o.min_reps && total > o.max_seconds) break;
  }
  CU(cudaGetLastError());
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return T;
}

// --------------------------------------------------------------------- the bench context ---
struct Bench {
  Case c;
  Options o;
  cudaStream_t s = nullptr;
  cublasHandle_t blas = nullptr;
  cublasLtHandle_t lt = nullptr;
  cutensorHandle_t tens = nullptr;
  imd_handle_t imd = nullptr;
  DevBuf dA, dB, dC;
  Reference R;  // sampled reference (host memory O(samples); operands stay on the device)
  Json meta;     // fields common to every record
  FILE* out = stdout;

  // GEMM view: canonical labels and whether each operand needs a permutation.
  std::string cA, cB, cC;  // labels of the GEMM-view operands
  bool needA = false, needB = false, needC = false;
  bool opA_kfast = true, opB_kfast = true;  // A = [L][I][K] (k fastest) or [L][K][I]
  DevBuf pA, pB, pC;                        // permuted operands (only when needed)
  cutensorPlan_t permA = nullptr, permB = nullptr, permC = nullptr;

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

  // ------------------------------------------------------------------------ setup ---
  void setup() {
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CB(cublasCreate(&blas));
    CB(cublasSetStream(blas, s));
    CB(cublasLtCreate(&lt));
    CT(cutensorCreate(&tens));
    CI(imd_create(&imd));

    dA = DevBuf(c.A.size * 8);
    dB = DevBuf(c.B.size * 8);
    dC = DevBuf(c.C.size * 8);
    k_fill<<<1024, 256, 0, s>>>(dA.as<double>(), c.A.size, splitmix64(o.seed * 3 + 0), o.phi);
    k_fill<<<1024, 256, 0, s>>>(dB.as<double>(), c.B.size, splitmix64(o.seed * 3 + 1), o.phi);
    CU(cudaStreamSynchronize(s));
    CU(cudaGetLastError());
    R = make_reference(c, dA.as<double>(), dB.as<double>(), o.samples, o.seed, s);

    // GEMM view. A: L ++ (I ++ K | K ++ I); B: L ++ (J ++ K | K ++ J); C: L ++ I ++ J.
    auto pick = [&](const std::string& have, const std::string& X, bool& kfast) {
      if (have == c.L + X + c.K) { kfast = true; return have; }
      if (have == c.L + c.K + X) { kfast = false; return have; }
      kfast = true;
      return c.L + X + c.K;
    };
    cA = pick(c.A.labels, c.I, opA_kfast);
    cB = pick(c.B.labels, c.J, opB_kfast);
    cC = c.L + c.I + c.J;
    needA = cA != c.A.labels;
    needB = cB != c.B.labels;
    needC = cC != c.C.labels;
    if (needA) { pA = DevBuf(dA.bytes); permA = make_perm(c.A, cA); }
    if (needB) { pB = DevBuf(dB.bytes); permB = make_perm(c.B, cB); }
    if (needC) { pC = DevBuf(dC.bytes); permC = make_perm(make_operand(cC, c.ext), c.C.labels); }

    cudaDeviceProp p{};
    CU(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    int cbv = 0;
    cublasGetVersion(blas, &cbv);
    std::string extstr;
    for (auto& [k, v] : c.ext) extstr += (extstr.empty() ? "" : ",") + std::string(1, k) + "=" + std::to_string(v);
    meta.str("case", c.name).str("spec", c.spec).str("ext", extstr);
    meta.integer("m", c.m).integer("n", c.n).integer("k", c.k).integer("b", c.b);
    meta.str("gemm_view", cA + "," + cB + "->" + cC);
    meta.raw("perm_needed", std::string("\"") + (needA ? "A" : "") + (needB ? "B" : "") + (needC ? "C" : "") + "\"");
    meta.num("phi", o.phi).integer("seed", (long long)o.seed);
    meta.str("gpu", p.name).integer("sm_count", p.multiProcessorCount);
    meta.str("driver", sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null"));
    meta.integer("cuda_driver_api", drv).integer("cuda_runtime", rt).integer("cublas", cbv);
    meta.integer("cutensor", (long long)cutensorGetVersion());
    meta.str("gpu_max_clocks", sh("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem "
                                  "--format=csv,noheader 2>/dev/null"));
    meta.str("commit", getenv_or("BENCH_COMMIT", "unknown"));
    meta.str("env", getenv_or("BENCH_ENV", "unknown"));
    const char* ql = std::getenv("QUIET_LOG");
    meta.str("conditions", ql ? read_file(ql) : "not recorded (not run via tools/run_bench.sh)");
  }

  cutensorPlan_t make_perm(const Operand& src, const std::string& dst_labels) {
    Operand dst = make_operand(dst_labels, c.ext);
    cutensorTensorDescriptor_t ds, dd;
    CT(cutensorCreateTensorDescriptor(tens, &ds, uint32_t(src.labels.size()), src.extent.data(),
                                      src.stride.data(), CUTENSOR_R_64F, 256));
    CT(cutensorCreateTensorDescriptor(tens, &dd, uint32_t(dst.labels.size()), dst.extent.data(),
                                      dst.stride.data(), CUTENSOR_R_64F, 256));
    std::vector<int32_t> ms(src.labels.begin(), src.labels.end()), md(dst.labels.begin(), dst.labels.end());
    cutensorOperationDescriptor_t op;
    CT(cutensorCreatePermutation(tens, &op, ds, ms.data(), CUTENSOR_OP_IDENTITY, dd, md.data(),
                                 CUTENSOR_COMPUTE_DESC_64F));
    cutensorPlanPreference_t pref;
    CT(cutensorCreatePlanPreference(tens, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    cutensorPlan_t plan;
    CT(cutensorCreatePlan(tens, &plan, op, pref, 0));
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(op);
    cutensorDestroyTensorDescriptor(ds);
    cutensorDestroyTensorDescriptor(dd);
    return plan;
  }
  void permute(cutensorPlan_t plan, const void* src, void* dst) {
    const double one = 1.0;
    CT(cutensorPermute(tens, plan, &one, src, dst, s));
  }

  // The matrix-only methods: `gemm(A, B, C)` on the GEMM view (row-major, see header).
  void run_gemm_family(const std::string& method, const std::string& variant, int N,
                       const std::function<void(const double*, const double*, double*)>& gemm,
                       size_t ws_bytes, Json extra = {}) {
    const double* a = needA ? pA.as<double>() : dA.as<double>();
    const double* bb = needB ? pB.as<double>() : dB.as<double>();
    double* cc = needC ? pC.as<double>() : dC.as<double>();
    if (needA) permute(permA, dA.p, pA.p);
    if (needB) permute(permB, dB.p, pB.p);
    poison_C();
    Timing T = time_it([&] { gemm(a, bb, cc); }, s, o);
    if (needC) permute(permC, cc, dC.p);
    Accuracy acc = result_accuracy();
    if (!(needA || needB || needC)) {
      record(method, variant, N, T, &acc, flops(), ws_bytes, extra);
      return;
    }
    Json e1 = extra;
    e1.str("layout", "gemm-view (operands pre-permuted, permutations not timed)");
    record(method, variant, N, T, &acc, flops(), ws_bytes, e1);
    poison_C();
    Timing P = time_it([&] {
      if (needA) permute(permA, dA.p, pA.p);
      if (needB) permute(permB, dB.p, pB.p);
      gemm(a, bb, cc);
      if (needC) permute(permC, cc, dC.p);
    }, s, o);
    acc = result_accuracy();
    Json e2 = extra;
    e2.str("layout", "original (explicit cuTENSOR permutations + GEMM, timed together)");
    record("perm+" + method, variant, N, P, &acc, flops(), ws_bytes, e2);
  }

  // Column-major call computing C^T (n x m) = op(B) * op(A), see the header comment.
  struct BlasArgs {
    cublasOperation_t ta, tb;  // for the cuBLAS "A" (= our B) and "B" (= our A)
    int64_t lda, ldb, ldc, sa, sb, sc;
  };
  BlasArgs blas_args() const {
    BlasArgs g;
    g.ta = opB_kfast ? CUBLAS_OP_T : CUBLAS_OP_N;
    g.lda = opB_kfast ? c.k : c.n;
    g.tb = opA_kfast ? CUBLAS_OP_N : CUBLAS_OP_T;
    g.ldb = opA_kfast ? c.k : c.m;
    g.ldc = c.n;
    g.sa = c.n * c.k;
    g.sb = c.m * c.k;
    g.sc = c.m * c.n;
    return g;
  }

  void run_cublas(cublasHandle_t h, const std::string& method, cublasComputeType_t ct,
                  const std::string& variant) {
    BlasArgs g = blas_args();
    const double one = 1.0, zero = 0.0;
    auto gemm = [&, g](const double* a, const double* bb, double* cc) {
      if (c.b == 1)
        CB(cublasGemmEx(h, g.ta, g.tb, int(c.n), int(c.m), int(c.k), &one, bb, CUDA_R_64F,
                        int(g.lda), a, CUDA_R_64F, int(g.ldb), &zero, cc, CUDA_R_64F, int(g.ldc),
                        ct, CUBLAS_GEMM_DEFAULT));
      else
        CB(cublasGemmStridedBatchedEx(h, g.ta, g.tb, int(c.n), int(c.m), int(c.k), &one, bb,
                                      CUDA_R_64F, int(g.lda), g.sa, a, CUDA_R_64F, int(g.ldb),
                                      g.sb, &zero, cc, CUDA_R_64F, int(g.ldc), g.sc, int(c.b), ct,
                                      CUBLAS_GEMM_DEFAULT));
    };
    run_gemm_family(method, variant, 0, gemm, 0);
  }

  void method_dgemm() {
    CB(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    run_cublas(blas, "dgemm", CUBLAS_COMPUTE_64F_PEDANTIC, "CUBLAS_COMPUTE_64F_PEDANTIC");
    CB(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  }

  void method_emu(const std::string& variant, size_t ws_limit) {
    // A handle of its own, so the emulation settings and workspace never leak into dgemm.
    cublasHandle_t eh;
    CB(cublasCreate(&eh));
    std::unique_ptr<cublasContext, cublasStatus_t (*)(cublasHandle_t)> guard(eh, cublasDestroy);
    CB(cublasSetStream(eh, s));
    DevBuf ws(ws_limit);  // the emulated path takes its scratch from the handle workspace
    CB(cublasSetWorkspace(eh, ws.p, ws.bytes));
    CB(cublasSetEmulationStrategy(eh, CUBLAS_EMULATION_STRATEGY_EAGER));
    if (variant == "dyn") {
      CB(cublasSetFixedPointEmulationMantissaControl(eh, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
    } else if (variant.rfind("fix", 0) == 0) {
      CB(cublasSetFixedPointEmulationMantissaControl(eh, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
      CB(cublasSetFixedPointEmulationMaxMantissaBitCount(eh, std::stoi(variant.substr(3))));
    } else {
      throw Failure("unknown --emu variant " + variant);
    }
    int maxbits = -1, offset = 0;
    cudaEmulationMantissaControl mc{};
    cublasGetFixedPointEmulationMaxMantissaBitCount(eh, &maxbits);
    cublasGetFixedPointEmulationMantissaBitOffset(eh, &offset);
    cublasGetFixedPointEmulationMantissaControl(eh, &mc);
    std::string v = variant + " (control=" + (mc == CUDA_EMULATION_MANTISSA_CONTROL_FIXED ? "FIXED" : "DYNAMIC") +
                    ", max_bits=" + std::to_string(maxbits) + ", offset=" + std::to_string(offset) +
                    ", strategy=EAGER, ws=" + std::to_string(ws.bytes >> 20) + "MiB)";
    run_cublas(eh, "emu", CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, v);
    CU(cudaStreamSynchronize(s));
  }

  void method_int8() {
    const int64_t kh = (c.k + 15) / 16 * 16;
    DevBuf a8(size_t(c.b * c.m * kh)), b8(size_t(c.b * c.n * kh)), c32(size_t(c.b * c.m * c.n) * 4);
    k_fill_i8<<<1024, 256, 0, s>>>(a8.as<int8_t>(), a8.bytes, 11);
    k_fill_i8<<<1024, 256, 0, s>>>(b8.as<int8_t>(), b8.bytes, 12);
    cublasLtMatmulDesc_t op;
    CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn));
    cublasLtMatrixLayout_t la, lb, lc;  // la = our B (k^ x n), lb = our A (k^ x m), lc = C^T
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, uint64_t(kh), uint64_t(c.n), kh));
    CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, uint64_t(kh), uint64_t(c.m), kh));
    CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, uint64_t(c.n), uint64_t(c.m), c.n));
    int32_t bc = int32_t(c.b);
    int64_t sa = c.n * kh, sb = c.m * kh, sc = c.m * c.n;
    for (auto [l, st] : {std::pair{la, sa}, std::pair{lb, sb}, std::pair{lc, sc}}) {
      CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
      CB(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof st));
    }
    size_t wsb = size_t(64) << 20;
    DevBuf ws(wsb);
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb));
    cublasLtMatmulHeuristicResult_t h{};
    int found = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &h, &found));
    if (!found) throw Failure("cuBLASLt: no int8 algorithm");
    const int32_t one = 1, zero = 0;
    Timing Tm = time_it([&] {
      CB(cublasLtMatmul(lt, op, &one, b8.p, la, a8.p, lb, &zero, c32.p, lc, c32.p, lc, &h.algo,
                        ws.p, wsb, s));
    }, s, o);
    double ops = 2.0 * double(c.m) * double(c.n) * double(kh) * double(c.b);
    Json e;
    e.num("int8_tops", ops / Tm.median() / 1e12).integer("k_hat", kh);
    e.str("layout", "GEMM view, int8 TN, int32 out; time of ONE modulus' GEMM");
    record("int8", "cublasLt int8 TN int32-out", 1, Tm, nullptr, NAN, wsb, e);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
  }

  void method_gemmul8(int N) {
#ifdef IMD_BENCH_GEMMUL8
    BlasArgs g = blas_args();
    size_t wsb = gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(
        size_t(c.n), size_t(c.m), size_t(c.k), N);
    DevBuf ws(wsb);
    const double one = 1.0, zero = 0.0;
    auto gemm = [&, g](const double* a, const double* bb, double* cc) {
      for (int64_t l = 0; l < c.b; ++l)
        gemmul8::gemm<double, gemmul8::Backend::INT8, double, double>(
            blas, g.ta, g.tb, size_t(c.n), size_t(c.m), size_t(c.k), &one, bb + l * g.sa,
            size_t(g.lda), a + l * g.sb, size_t(g.ldb), &zero, cc + l * g.sc, size_t(g.ldc), N,
            true, ws.p);
    };
    Json e;
    if (c.b > 1) e.str("note", "one GEMMul8 call per batch (no batched API)");
    run_gemm_family("gemmul8", "INT8 fastmode", N, gemm, wsb, e);
#else
    record_error("gemmul8", "INT8 fastmode", N, "built without GEMMul8 (tools/get_gemmul8.sh)");
#endif
  }

  void method_cutensor() {
    auto desc = [&](const Operand& x) {
      cutensorTensorDescriptor_t d;
      CT(cutensorCreateTensorDescriptor(tens, &d, uint32_t(x.labels.size()), x.extent.data(),
                                        x.stride.data(), CUTENSOR_R_64F, 256));
      return d;
    };
    auto da = desc(c.A), db = desc(c.B), dc = desc(c.C);
    std::vector<int32_t> ma(c.A.labels.begin(), c.A.labels.end()), mb(c.B.labels.begin(), c.B.labels.end()),
        mc(c.C.labels.begin(), c.C.labels.end());
    cutensorOperationDescriptor_t op;
    CT(cutensorCreateContraction(tens, &op, da, ma.data(), CUTENSOR_OP_IDENTITY, db, mb.data(),
                                 CUTENSOR_OP_IDENTITY, dc, mc.data(), CUTENSOR_OP_IDENTITY, dc,
                                 mc.data(), CUTENSOR_COMPUTE_DESC_64F));
    cutensorPlanPreference_t pref;
    CT(cutensorCreatePlanPreference(tens, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    uint64_t wsb = 0;
    CT(cutensorEstimateWorkspaceSize(tens, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &wsb));
    cutensorPlan_t plan;
    CT(cutensorCreatePlan(tens, &plan, op, pref, wsb));
    DevBuf ws(wsb);
    const double one = 1.0, zero = 0.0;
    poison_C();
    Timing T = time_it([&] {
      CT(cutensorContract(tens, plan, &one, dA.p, dB.p, &zero, dC.p, dC.p, ws.p, wsb, s));
    }, s, o);
    Accuracy acc = result_accuracy();
    record("cutensor", "CUTENSOR_COMPUTE_DESC_64F", 0, T, &acc, flops(), wsb);
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(op);
    for (auto d : {da, db, dc}) cutensorDestroyTensorDescriptor(d);
  }

  void method_imd(int N) {
    auto t = [&](const Operand& x, std::vector<int32_t>& lab) {
      lab.assign(x.labels.begin(), x.labels.end());
      return imd_tensor_t{int32_t(x.labels.size()), x.extent.data(), x.stride.data(), lab.data()};
    };
    std::vector<int32_t> la, lb, lc;
    imd_tensor_t A = t(c.A, la), B = t(c.B, lb), C = t(c.C, lc);
    imd_options_t opt{N, 0, IMD_RN, IMD_EMULATE};
    size_t wsb = 0;
    CI(imd_workspace_size(&A, &B, &C, &opt, &wsb));
    imd_plan_t plan = nullptr;
    CI(imd_plan_create(imd, &A, &B, &C, &opt, &plan));
    // The plan must outlive in-flight work (K0 reads its pinned buffer): sync first.
    auto destroy = [](imd_plan_t p) { cudaDeviceSynchronize(); imd_plan_destroy(p); };
    std::unique_ptr<imd_plan_st, decltype(destroy)> guard(plan, destroy);
    DevBuf ws(wsb);
    poison_C();
    Timing T = time_it([&] {
      CI(imd_contract(plan, 1.0, dA.as<double>(), dB.as<double>(), 0.0, dC.as<double>(), ws.p, wsb, s));
    }, s, o);
    Accuracy acc = result_accuracy();
    CU(cudaStreamSynchronize(s));
    record("imd", "RN g=N (M1, pre-optimization)", N, T, &acc, flops(), wsb);
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
          "  [--N 8,14,20] [--emu dyn,fix55] [--emu-ws-mib 2048] [--reps 10] [--warmup 2]\n"
          "  [--max-seconds 15] [--samples 256] [--phi 0.5] [--seed 1] [--out file.jsonl]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a{{"methods", "dgemm,emu,int8,gemmul8,cutensor,imd"},
                                       {"N", "14"},
                                       {"emu", "dyn"},
                                       {"emu-ws-mib", "2048"}};
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) { usage(); return 2; }
    a[k.substr(2)] = argv[++i];
  }
  if (!a.count("name") || !a.count("spec") || !a.count("ext")) { usage(); return 2; }
  Bench B;
  try {
    B.c = parse_case(a["name"], a["spec"], a["ext"]);
    if (a.count("reps")) B.o.reps = std::stoi(a["reps"]);
    if (a.count("warmup")) B.o.warmup = std::stoi(a["warmup"]);
    if (a.count("max-seconds")) B.o.max_seconds = std::stod(a["max-seconds"]);
    if (a.count("samples")) B.o.samples = std::stoll(a["samples"]);
    if (a.count("phi")) B.o.phi = std::stod(a["phi"]);
    if (a.count("seed")) B.o.seed = std::stoull(a["seed"]);
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
    if (m == "dgemm") guarded(m, "", 0, [&] { B.method_dgemm(); });
    else if (m == "emu")
      for (auto& v : split(a["emu"], ','))
        guarded(m, v, 0, [&] { B.method_emu(v, size_t(std::stoll(a["emu-ws-mib"])) << 20); });
    else if (m == "int8") guarded(m, "", 1, [&] { B.method_int8(); });
    else if (m == "gemmul8") for (int N : Ns) guarded(m, "", N, [&] { B.method_gemmul8(N); });
    else if (m == "cutensor") guarded(m, "", 0, [&] { B.method_cutensor(); });
    else if (m == "imd") for (int N : Ns) guarded(m, "", N, [&] { B.method_imd(N); });
    else if (m == "bw") guarded(m, "", 0, [&] { B.method_bw(); });
    else { fprintf(stderr, "unknown method %s\n", m.c_str()); return 2; }
  }
  if (B.out != stdout) fclose(B.out);
  return 0;
}
