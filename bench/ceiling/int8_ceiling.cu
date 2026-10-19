// INT8 tensor-core ceilings (issue 16): what rate do the INT8 matrix instructions and the stock GEMM kernels reach on this GPU,
// measured with the same inputs, timing and exactness check, so that K4's rate can be set against them.
//
//   imd_ceiling --variants LIST --shapes MxNxK,... [--batch B] [--out-type i32|i8] [--reps 20] [--warmup 3] [--max-seconds 2] [--keep-samples 1]
//               [--check-samples 4096] [--out file.jsonl]
//
// Variants (each = C[m][n] (int32) = A[m][k] (int8) * B[n][k]^T (int8), both operands k-contiguous, i.e. "TN"):
//   mma_peak        synthetic loop of mma.sync.m16n8k32 (K4's instruction) in registers, no memory traffic: the instruction ceiling
//                   (shape-independent; the shape argument is ignored, the result is checked in closed form)
//   cublaslt        cublasLtMatmul INT8 -> INT32 (a yardstick, never a fallback)
//   sm80:TBxTBN     stock CUTLASS sm80 GEMM (mma.sync, cp.async pipeline) with threadblock tile TBMxTBNx64, 3 stages: K4's own kind
//   hopper:TILE:CLUSTER:SCHED   stock CUTLASS 4.x Hopper GEMM (wgmma + TMA), only when built for sm_90a (IMD_CEIL_HOPPER);
//                   TILE = e.g. 128x256x128, CLUSTER = e.g. 1x2x1, SCHED = coop | ping
//   default list: mma_peak, cublaslt, sm80:128x256, sm80:128x128, and on Hopper builds four hopper variants
//
// --batch B runs B independent problems of the shape in one launch (K4 does one launch over all N moduli); --out-type i8 stores
// saturated int8 (K4 stores u8 residues: a quarter of the int32 store traffic). cublaslt is single-GEMM int32 only.
//
// Every record has "exact": the result is compared with a CPU reference on --check-samples output elements (all of them if the
// output is smaller); integer arithmetic, so equal means equal. A variant that fails the check is reported with exact=false
// and its rate must not be used. Records: method "ceiling", tops (INT8 trillion operations per second = 2mnk / time), median_s,
// spread_pct as in imd_bench, samples_s with --keep-samples. A variant that cannot run (not built, not implemented for the
// shape) gives a record with status "error: ..." / "skipped: ..." and the others still run.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#ifdef IMD_CEIL_HOPPER
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#endif

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw Fail(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };

// ------------------------------------------------------------------ data: a hash of (matrix, row, k) -> int8 in [-127, 127]
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

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit Buf(size_t n = 0) : bytes(n) { if (n) CU(cudaMalloc(&p, n)); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
  Buf& operator=(Buf&& o) noexcept {
    if (this != &o) { if (p) cudaFree(p); p = o.p; bytes = o.bytes; o.p = nullptr; }
    return *this;
  }
  ~Buf() { if (p) cudaFree(p); }
  template <class T> T* as() const { return static_cast<T*>(p); }
};

// ------------------------------------------------------------------ the variants
struct Problem { int64_t m, n, k; int64_t batch = 1; bool i8out = false; };
using Launch = std::function<void(cudaStream_t)>;
struct Prepared {  // a ready-to-run variant: launch() issues one GEMM into C; keep holds its workspaces and plans
  Launch launch;
  std::shared_ptr<void> keep;
  std::string note;
};
struct Operands { const int8_t* A; const int8_t* B; void* C; };

// --- synthetic mma.sync peak
__global__ void k_mma_peak(int iters, int32_t* out) {
  // A = B = all ones: every mma adds k = 32 to each accumulator element, so after `iters` rounds each is 32 * iters * chains
  uint32_t a0 = 0x01010101u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  int c[4][4] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  if (threadIdx.x < 32 && blockIdx.x == 0) {
    int s = 0;
    for (int j = 0; j < 4; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[threadIdx.x] = s;
  }
}

// --- cuBLASLt
struct LtPlan {
  cublasLtHandle_t h = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
  Buf ws;
  ~LtPlan() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (op) cublasLtMatmulDescDestroy(op);
    if (h) cublasLtDestroy(h);
  }
};

// --- CUTLASS sm80 (batched: one problem per modulus, like K4's single launch over all N moduli)
template <int TM, int TN, bool I8>
struct Sm80Gemm {
  using Out = std::conditional_t<I8, int8_t, int32_t>;
  using Op = std::conditional_t<I8, cutlass::epilogue::thread::LinearCombinationClamp<int8_t, 16, int32_t, float>,
                                cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>>;
  using Gemm = cutlass::gemm::device::GemmBatched<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, Out,
                                                  cutlass::layout::RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, 64>,
                                                  cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>, Op,
                                                  cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3, 16, 16,
                                                  cutlass::arch::OpMultiplyAddSaturate>;
};
template <int TM, int TN, bool I8>
Prepared make_sm80(const Problem& P, const Operands& o) {
  using S = Sm80Gemm<TM, TN, I8>;
  using G = typename S::Gemm;
  using Out = typename S::Out;
  auto g = std::make_shared<G>();
  const long long bs_a = P.m * P.k, bs_b = P.n * P.k, bs_c = P.m * P.n;
  typename G::Arguments args({int(P.m), int(P.n), int(P.k)}, {o.A, int(P.k)}, bs_a, {o.B, int(P.k)}, bs_b,
                             {static_cast<Out*>(o.C), int(P.n)}, bs_c, {static_cast<Out*>(o.C), int(P.n)}, bs_c,
                             {typename S::Op::ElementCompute(1),
                              typename S::Op::ElementCompute(0)}, int(P.batch));
  if (g->can_implement(args) != cutlass::Status::kSuccess) throw Fail("skipped: CUTLASS sm80 cannot implement this shape");
  auto ws = std::make_shared<Buf>(G::get_workspace_size(args));
  if (g->initialize(args, ws->p) != cutlass::Status::kSuccess) throw Fail("CUTLASS sm80 initialize failed");
  Prepared r;
  r.launch = [g](cudaStream_t s) { if ((*g)(s) != cutlass::Status::kSuccess) throw Fail("CUTLASS sm80 launch failed"); };
  r.keep = std::make_shared<std::pair<std::shared_ptr<G>, std::shared_ptr<Buf>>>(g, ws);
  return r;
}

#ifdef IMD_CEIL_HOPPER
// --- CUTLASS Hopper (wgmma + TMA)
template <class Tile, class Cluster, class Sched, bool I8>
struct HopperGemm {
  using Out = std::conditional_t<I8, int8_t, int32_t>;
  using Compute = std::conditional_t<I8, float, int32_t>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignC = I8 ? 16 : 4;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      int32_t, Compute, Out, LayoutC, AlignC, Out, LayoutC, AlignC, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, LayoutA, 16, int8_t, LayoutB, 16, int32_t, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      Sched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
template <class Tile, class Cluster, class Sched, bool I8>
Prepared make_hopper(const Problem& P, const Operands& o) {
  using H = HopperGemm<Tile, Cluster, Sched, I8>;
  using G = typename H::Gemm;
  using Out = typename H::Out;
  auto g = std::make_shared<G>();
  const int M = int(P.m), N = int(P.n), K = int(P.k), L = int(P.batch);
  // packed strides written out (cutlass/util/packed_stride.hpp lives under tools/util, which the PACE job's CUTLASS copy lacks):
  // A (M x K) and B (N x K) are K-contiguous, C and D (M x N) row-major, each batch after the other
  auto sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M) * K);
  auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(N) * K);
  auto sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(M) * N);
  auto sD = sC;
  typename G::Arguments args{L > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
                             {M, N, K, L}, {o.A, sA, o.B, sB},
                             {{typename H::Compute(1), typename H::Compute(0)}, static_cast<Out*>(o.C), sC, static_cast<Out*>(o.C), sD}};
  if (g->can_implement(args) != cutlass::Status::kSuccess) throw Fail("skipped: CUTLASS Hopper cannot implement this shape");
  auto ws = std::make_shared<Buf>(G::get_workspace_size(args));
  if (g->initialize(args, ws->p) != cutlass::Status::kSuccess) throw Fail("CUTLASS Hopper initialize failed");
  Prepared r;
  r.launch = [g](cudaStream_t s) { if (g->run(s) != cutlass::Status::kSuccess) throw Fail("CUTLASS Hopper run failed"); };
  r.keep = std::make_shared<std::pair<std::shared_ptr<G>, std::shared_ptr<Buf>>>(g, ws);
  return r;
}
using namespace cute;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// name -> factory; the tile is M x N x K of one threadblock, the cluster the thread-block cluster
std::map<std::string, Prepared (*)(const Problem&, const Operands&)> hopper_table(bool i8) {
  using F = Prepared (*)(const Problem&, const Operands&);
#define IMD_H(NAME, ...) \
  {NAME, i8 ? static_cast<F>(&make_hopper<__VA_ARGS__, true>) : static_cast<F>(&make_hopper<__VA_ARGS__, false>)}
  return {
      IMD_H("hopper:128x256x128:1x2x1:coop", Shape<_128, _256, _128>, Shape<_1, _2, _1>, Coop),
      IMD_H("hopper:128x256x128:1x1x1:coop", Shape<_128, _256, _128>, Shape<_1, _1, _1>, Coop),
      IMD_H("hopper:128x128x128:2x1x1:coop", Shape<_128, _128, _128>, Shape<_2, _1, _1>, Coop),
      IMD_H("hopper:128x128x128:1x1x1:ping", Shape<_128, _128, _128>, Shape<_1, _1, _1>, Ping),
      IMD_H("hopper:128x256x64:1x2x1:coop", Shape<_128, _256, _64>, Shape<_1, _2, _1>, Coop),
      IMD_H("hopper:256x128x128:2x1x1:coop", Shape<_256, _128, _128>, Shape<_2, _1, _1>, Coop),
  };
#undef IMD_H
}
#endif

// ------------------------------------------------------------------ measurement
struct Opt {
  int reps = 20, warmup = 3;
  double max_seconds = 2.0;
  long long check = 4096;
  bool keep = false;
  std::string out;
};
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += (c == '\n') ? ' ' : c; }
  return o + "\"";
}
static std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  snprintf(b, sizeof b, "%.9g", v);
  return b;
}
struct Rec {
  std::vector<std::pair<std::string, std::string>> kv;
  Rec& s(const std::string& k, const std::string& v) { kv.emplace_back(k, jstr(v)); return *this; }
  Rec& n(const std::string& k, double v) { kv.emplace_back(k, jnum(v)); return *this; }
  Rec& i(const std::string& k, long long v) { kv.emplace_back(k, std::to_string(v)); return *this; }
  Rec& raw(const std::string& k, const std::string& v) { kv.emplace_back(k, v); return *this; }
  std::string dump() const {
    std::string o = "{";
    for (size_t j = 0; j < kv.size(); ++j) o += (j ? ", " : "") + jstr(kv[j].first) + ": " + kv[j].second;
    return o + "}";
  }
};
static FILE* g_out = stdout;
static std::string g_gpu;
static int g_sms = 0;
static std::string gpu_state() {  // clock, power, temperature and the clock-event reasons right after the timed block (as imd_bench's gpu_after)
  std::string out;
  if (FILE* f = popen("nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_event_reasons.active "
                      "--format=csv,noheader 2>/dev/null", "r")) {
    char b[256];
    while (fgets(b, sizeof b, f)) out += b;
    pclose(f);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  return out;
}
static void emit(Rec r) {
  r.s("gpu", g_gpu).i("sm_count", g_sms);
  if (r.kv.size() > 3 && r.kv.back().first == "status") r.s("gpu_after", gpu_state());
  if (const char* c = std::getenv("BENCH_COMMIT")) r.s("commit", c);
  std::string line = r.dump();
  fprintf(g_out, "%s\n", line.c_str());
  fflush(g_out);
  fprintf(stderr, "  %s\n", line.substr(0, 220).c_str());
}

// CPU reference on sampled output elements (all of them for a small C). Returns the number of mismatches. With an int8 output the
// reference is the saturated value (what the clamping epilogue stores): a weaker check, so the int32 variants carry the exactness claim.
static long long check_result(const Problem& P, const void* hostC, long long samples, long long* checked) {
  const long long per = P.m * P.n, total = per * P.batch;
  const long long cnt = std::min<long long>(samples, total);
  long long bad = 0;
  for (long long q = 0; q < cnt; ++q) {
    long long pos = cnt == total ? q : (long long)(mix(uint64_t(q) * 7919u + 11) % uint64_t(total));
    const long long b = pos / per, r = pos % per, i = r / P.n, j = r % P.n;
    long long ref = 0;
    for (int64_t kk = 0; kk < P.k; ++kk) ref += (long long)val(1, b * P.m + i, kk) * (long long)val(2, b * P.n + j, kk);
    long long got = P.i8out ? (long long)static_cast<const int8_t*>(hostC)[pos] : (long long)static_cast<const int32_t*>(hostC)[pos];
    if (P.i8out) ref = std::max<long long>(-128, std::min<long long>(127, ref));
    if (ref != got) ++bad;
  }
  *checked = cnt;
  return bad;
}

struct Fixture {
  Problem P;
  Buf A, B, C;
  explicit Fixture(const Problem& p)
      : P(p), A(p.batch * p.m * p.k), B(p.batch * p.n * p.k), C(p.batch * p.m * p.n * (p.i8out ? 1 : 4)) {
    k_fill<<<2048, 256>>>(A.as<int8_t>(), 1, p.batch * p.m, p.k);   // batch b, row i is row b * m + i
    k_fill<<<2048, 256>>>(B.as<int8_t>(), 2, p.batch * p.n, p.k);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }
  Operands ops() const { return {A.as<int8_t>(), B.as<int8_t>(), C.p}; }
};

static Prepared make_cublaslt(const Problem& P, const Operands& o) {
  if (P.batch != 1 || P.i8out) throw Fail("skipped: the cuBLASLt yardstick is a single int32-output GEMM");
  auto pl = std::make_shared<LtPlan>();
  auto chk = [](cublasStatus_t s, const char* w) { if (s != CUBLAS_STATUS_SUCCESS) throw Fail(std::string(w) + " failed (" + std::to_string(int(s)) + ")"); };
  chk(cublasLtCreate(&pl->h), "cublasLtCreate");
  chk(cublasLtMatmulDescCreate(&pl->op, CUBLAS_COMPUTE_32I, CUDA_R_32I), "cublasLtMatmulDescCreate");
  // our row-major C[m][n] = A[m][k] B[n][k]^T is, column-major, C'[n][m] = op(B') A' with B' = B (k x n, T), A' = A (k x m, N)
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  chk(cublasLtMatmulDescSetAttribute(pl->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "TRANSA");
  chk(cublasLtMatmulDescSetAttribute(pl->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "TRANSB");
  chk(cublasLtMatrixLayoutCreate(&pl->la, CUDA_R_8I, P.k, P.n, P.k), "layout A");
  chk(cublasLtMatrixLayoutCreate(&pl->lb, CUDA_R_8I, P.k, P.m, P.k), "layout B");
  chk(cublasLtMatrixLayoutCreate(&pl->lc, CUDA_R_32I, P.n, P.m, P.n), "layout C");
  chk(cublasLtMatmulPreferenceCreate(&pl->pref), "preference");
  size_t wsb = size_t(256) << 20;
  chk(cublasLtMatmulPreferenceSetAttribute(pl->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb), "workspace pref");
  int found = 0;
  chk(cublasLtMatmulAlgoGetHeuristic(pl->h, pl->op, pl->la, pl->lb, pl->lc, pl->lc, pl->pref, 1, &pl->heur, &found), "heuristic");
  if (found == 0) throw Fail("skipped: cublasLt has no INT8 algorithm for this shape");
  pl->ws = Buf(pl->heur.workspaceSize);
  Prepared r;
  const int32_t one = 1, zero = 0;
  r.launch = [pl, o, one, zero](cudaStream_t s) {
    cublasStatus_t st = cublasLtMatmul(pl->h, pl->op, &one, o.B, pl->la, o.A, pl->lb, &zero, o.C, pl->lc, o.C, pl->lc,
                                       &pl->heur.algo, pl->ws.p, pl->ws.bytes, s);
    if (st != CUBLAS_STATUS_SUCCESS) throw Fail("cublasLtMatmul failed (" + std::to_string(int(st)) + ")");
  };
  r.keep = pl;
  return r;
}

static void time_variant(const std::string& name, Prepared& pr, const Problem& P, Fixture& fx, const Opt& o, double ops_override) {
  cudaStream_t s = nullptr;
  for (int w = 0; w < o.warmup; ++w) pr.launch(s);
  CU(cudaDeviceSynchronize());
  CU(cudaGetLastError());
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  std::vector<double> t;
  double total = 0;
  for (int r = 0; r < o.reps; ++r) {
    CU(cudaEventRecord(e0, s));
    pr.launch(s);
    CU(cudaEventRecord(e1, s));
    CU(cudaEventSynchronize(e1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    t.push_back(ms * 1e-3);
    total += ms * 1e-3;
    if (r + 1 >= 3 && total > o.max_seconds) break;
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  CU(cudaGetLastError());
  const double med = median(t);
  const double ops = ops_override > 0 ? ops_override : 2.0 * double(P.m) * double(P.n) * double(P.k) * double(P.batch);
  Rec rec;
  rec.s("method", "ceiling").s("variant", name).i("m", P.m).i("n", P.n).i("k", P.k).i("batch", P.batch).s("out", P.i8out ? "i8" : "i32").i("reps", (long long)t.size());
  rec.n("median_s", med).n("min_s", *std::min_element(t.begin(), t.end())).n("max_s", *std::max_element(t.begin(), t.end()));
  rec.n("spread_pct", 100.0 * (*std::max_element(t.begin(), t.end()) - *std::min_element(t.begin(), t.end())) / med);
  rec.n("tops", ops / med / 1e12);
  if (name != "mma_peak") {  // exactness against the CPU reference: the C of a fresh call
    CU(cudaMemset(fx.C.p, 0xff, fx.C.bytes));
    pr.launch(s);
    CU(cudaDeviceSynchronize());
    std::vector<char> h(fx.C.bytes);
    CU(cudaMemcpy(h.data(), fx.C.p, h.size(), cudaMemcpyDeviceToHost));
    long long checked = 0, bad = check_result(P, h.data(), o.check, &checked);
    rec.raw("exact", bad == 0 ? "true" : "false").i("check_elements", checked).i("check_mismatches", bad);
  }
  if (!pr.note.empty()) rec.s("note", pr.note);
  if (o.keep) rec.raw("samples_s", [&] { std::string a = "["; for (size_t i = 0; i < t.size(); ++i) a += (i ? ", " : "") + jnum(t[i]); return a + "]"; }());
  rec.s("status", "ok");
  emit(rec);
}

static void run_mma_peak(const Opt& o, int sms) {
  const int iters = 20000, blocks = sms * 4, threads = 256;
  Buf out(32 * 4);
  Fixture dummy({16, 16, 32, 1, false});
  Prepared pr;
  pr.launch = [&](cudaStream_t s) { k_mma_peak<<<blocks, threads, 0, s>>>(iters, out.as<int32_t>()); };
  // ops: blocks * (threads / 32 warps) * iters * 4 mma * (2 * 16 * 8 * 32 ops)
  const double ops = double(blocks) * (threads / 32) * iters * 4.0 * (2.0 * 16 * 8 * 32);
  time_variant("mma_peak", pr, {16, 16, 32, 1, false}, dummy, o, ops);
  // closed-form check of the accumulators (thread 0..31 of block 0 each hold 4 chains x 4 values of 32 * iters)
  std::vector<int32_t> h(32);
  CU(cudaMemcpy(h.data(), out.p, 128, cudaMemcpyDeviceToHost));
  const long long want = 16LL * 32 * iters;
  bool ok = true;
  for (int v : h) ok &= (v == want);
  Rec r;
  r.s("method", "ceiling").s("variant", "mma_peak_check").raw("exact", ok ? "true" : "false").i("expected", want).i("got", h[0]).s("status", "ok");
  emit(r);
}

int main(int argc, char** argv) {
  std::map<std::string, std::string> a{{"shapes", "4096x4096x4096,8192x8192x8192"}};
  for (int i = 1; i + 1 < argc; i += 2) {
    if (strncmp(argv[i], "--", 2) != 0) { fprintf(stderr, "usage: see the header of bench/ceiling/int8_ceiling.cu\n"); return 2; }
    a[argv[i] + 2] = argv[i + 1];
  }
  Opt o;
  if (a.count("reps")) o.reps = std::stoi(a["reps"]);
  if (a.count("warmup")) o.warmup = std::stoi(a["warmup"]);
  if (a.count("max-seconds")) o.max_seconds = std::stod(a["max-seconds"]);
  if (a.count("check-samples")) o.check = std::stoll(a["check-samples"]);
  if (a.count("keep-samples")) o.keep = a["keep-samples"] != "0";
  if (a.count("out")) { g_out = fopen(a["out"].c_str(), "a"); if (!g_out) { perror("out"); return 2; } }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
  g_gpu = prop.name;
  g_sms = prop.multiProcessorCount;
  std::vector<std::string> variants;
  {
    std::string v = a.count("variants") ? a["variants"] : "";
    if (v.empty()) {
      v = "mma_peak,cublaslt,sm80:128x256,sm80:128x128";
#ifdef IMD_CEIL_HOPPER
      v += ",hopper:128x256x128:1x2x1:coop,hopper:128x128x128:2x1x1:coop,hopper:128x128x128:1x1x1:ping,hopper:128x256x128:1x1x1:coop";
#endif
    }
    std::stringstream ss(v);
    for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) variants.push_back(t);
  }
  const long long batch = a.count("batch") ? std::stoll(a["batch"]) : 1;
  const bool i8out = a.count("out-type") && a["out-type"] == "i8";
  std::vector<Problem> shapes;
  {
    std::stringstream ss(a["shapes"]);
    for (std::string t; std::getline(ss, t, ',');) {
      long long m, n, k;
      if (sscanf(t.c_str(), "%lldx%lldx%lld", &m, &n, &k) != 3) { fprintf(stderr, "bad shape %s\n", t.c_str()); return 2; }
      shapes.push_back({m, n, k, batch, i8out});
    }
  }
#ifdef IMD_CEIL_HOPPER
  auto htab = hopper_table(i8out);
#endif
  fprintf(stderr, "imd_ceiling: %s (%d SMs), compute capability %d.%d\n", g_gpu.c_str(), g_sms, prop.major, prop.minor);
  for (const auto& v : variants) {
    if (v == "mma_peak") {
      try { run_mma_peak(o, g_sms); } catch (const std::exception& e) { cudaGetLastError(); Rec r; r.s("method", "ceiling").s("variant", v).s("status", std::string("error: ") + e.what()); emit(r); }
      continue;
    }
    for (const auto& P : shapes) {
      try {
        Fixture fx(P);
        Operands ops = fx.ops();
        Prepared pr;
        if (v == "cublaslt") pr = make_cublaslt(P, ops);
        else if (v == "sm80:128x256") pr = P.i8out ? make_sm80<128, 256, true>(P, ops) : make_sm80<128, 256, false>(P, ops);
        else if (v == "sm80:128x128") pr = P.i8out ? make_sm80<128, 128, true>(P, ops) : make_sm80<128, 128, false>(P, ops);
        else if (v == "sm80:256x128") pr = P.i8out ? make_sm80<256, 128, true>(P, ops) : make_sm80<256, 128, false>(P, ops);
        else if (v.rfind("hopper:", 0) == 0) {
#ifdef IMD_CEIL_HOPPER
          auto it = htab.find(v);
          if (it == htab.end()) throw Fail("skipped: unknown hopper variant " + v);
          pr = it->second(P, ops);
#else
          throw Fail("skipped: built without Hopper support (needs sm_90a)");
#endif
        } else throw Fail("skipped: unknown variant " + v);
        time_variant(v, pr, P, fx, o, 0);
      } catch (const std::exception& e) {
        cudaGetLastError();
        std::string w = e.what();
        Rec r;
        r.s("method", "ceiling").s("variant", v).i("m", P.m).i("n", P.n).i("k", P.k)
            .s("status", w.rfind("skipped", 0) == 0 ? w : "error: " + w);
        emit(r);
      }
      cudaDeviceSynchronize();
    }
  }
  if (g_out != stdout) fclose(g_out);
  return 0;
}
