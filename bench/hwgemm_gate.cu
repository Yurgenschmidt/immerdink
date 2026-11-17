// hwgemm_gate: the instruction / kernel gate of IMD_HWGEMM (issue #75, docs/NOTES-HWGEMM.md §4.3).
//
// Question: is CUTLASS 2.x's FP64 tensor-op GEMM (arch::Sm90, mma.sync m16n8k4) close enough to
// cuBLAS DGEMM on Hopper to build on, and do the k8 / k16 FP64 mma.sync shapes have more peak?
// Rule (NOTES-HWGEMM §4.3): k4 >= 95 % of the DMMA peak in the microbenchmark and the best CUTLASS
// configuration >= 90 % of cuBLAS on the large squares.
//
// Parts (all FP64, TFLOPS = 2 b m n k / time):
//   mma   per-instruction throughput of mma.sync f64 m8n8k4, m16n8k4, m16n8k8, m16n8k16
//         (8 independent accumulator chains per warp, no memory traffic in the loop)
//   gemm  cuBLAS DGEMM vs CUTLASS Gemm<Sm90, 16x8x4> over several tile configurations,
//         layouts TN (both operands K-contiguous: the packed-operand case) and NN
//   batch cuBLAS strided-batched vs CUTLASS GemmBatched: t_batch (b 16, 1024^3), s_batch (b 256, 64^3)
// Every CUTLASS result is checked against cuBLAS (max relative difference printed).
// Output: CSV lines on stdout, prefixed by the part name.
//
// Build (sm_90): nvcc -O3 -std=c++17 -arch=sm_90 -I<cutlass>/include -I<cutlass>/tools/util/include
//                hwgemm_gate.cu -lcublas -o hwgemm_gate
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(2); } } while (0)

// ------------------------------------------------------------------ timing --
template <class F> static double time_ms(F f) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();                 // warm up
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float one; CK(cudaEventElapsedTime(&one, a, b));
  int reps = std::max(5, std::min(200, (int)(150.0f / std::max(one, 1e-3f))));
  std::vector<float> t;
  for (int r = 0; r < 7; ++r) {                    // 7 samples of `reps` back-to-back calls
    CK(cudaEventRecord(a)); for (int i = 0; i < reps; ++i) f(); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms / reps);
  }
  std::sort(t.begin(), t.end());
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return t[3];                                     // median
}

// ------------------------------------------------------------- mma part --
template <int KIND> __device__ __forceinline__ void mma_step(double (&acc)[8][4], const double* a, const double* b) {
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    if constexpr (KIND == 0) {        // m8n8k4: A 1, B 1, C 2
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                   : "+d"(acc[c][0]), "+d"(acc[c][1]) : "d"(a[0]), "d"(b[0]));
    } else if constexpr (KIND == 1) { // m16n8k4: A 2, B 1, C 4
      asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64.rn {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                   : "+d"(acc[c][0]), "+d"(acc[c][1]), "+d"(acc[c][2]), "+d"(acc[c][3])
                   : "d"(a[0]), "d"(a[1]), "d"(b[0]));
    } else if constexpr (KIND == 2) { // m16n8k8: A 4, B 2, C 4
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+d"(acc[c][0]), "+d"(acc[c][1]), "+d"(acc[c][2]), "+d"(acc[c][3])
                   : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(b[0]), "d"(b[1]));
    } else {                          // m16n8k16: A 8, B 4, C 4
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3};\n"
                   : "+d"(acc[c][0]), "+d"(acc[c][1]), "+d"(acc[c][2]), "+d"(acc[c][3])
                   : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(a[4]), "d"(a[5]), "d"(a[6]), "d"(a[7]),
                     "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]));
    }
  }
}

template <int KIND> __global__ void mma_kernel(const double* in, double* out, int iters) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 || KIND == 0)
  double a[8], b[4], acc[8][4];
  for (int i = 0; i < 8; ++i) a[i] = in[(threadIdx.x + i) & 31];
  for (int i = 0; i < 4; ++i) b[i] = in[32 + ((threadIdx.x + i) & 31)];
  for (int c = 0; c < 8; ++c) for (int i = 0; i < 4; ++i) acc[c][i] = 0.0;
  for (int it = 0; it < iters; ++it) mma_step<KIND>(acc, a, b);
  double s = 0; for (int c = 0; c < 8; ++c) for (int i = 0; i < 4; ++i) s += acc[c][i];
  if (s == 1234.5) out[0] = s;                     // keeps the loop alive
#endif
}

static void run_mma(int sms) {
  static const char* name[4] = {"m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16"};
  static const int mnk[4] = {8 * 8 * 4, 16 * 8 * 4, 16 * 8 * 8, 16 * 8 * 16};
  double *in, *out; CK(cudaMalloc(&in, 64 * sizeof(double))); CK(cudaMalloc(&out, sizeof(double)));
  CK(cudaMemset(in, 0, 64 * sizeof(double)));
  const int iters = 4096;
  for (int kind = 0; kind < 4; ++kind) {
    for (int warps = 4; warps <= 8; warps *= 2) {    // warps per block (16 warps x 44+ FP64 registers exceeds the register file)
      int blocks = sms * 4, threads = 32 * warps;
      auto f = [&] {
        switch (kind) {
          case 0: mma_kernel<0><<<blocks, threads>>>(in, out, iters); break;
          case 1: mma_kernel<1><<<blocks, threads>>>(in, out, iters); break;
          case 2: mma_kernel<2><<<blocks, threads>>>(in, out, iters); break;
          default: mma_kernel<3><<<blocks, threads>>>(in, out, iters); break;
        }
      };
      double ms = time_ms(f); CK(cudaGetLastError());
      double flops = 2.0 * mnk[kind] * 8.0 * iters * (double)blocks * warps;
      printf("mma,%s,warps_per_block=%d,blocks=%d,%.3f ms,%.2f TFLOPS\n", name[kind], warps, blocks, ms, flops / ms / 1e9);
    }
  }
  CK(cudaFree(in)); CK(cudaFree(out));
}

// ------------------------------------------------------------ gemm part --
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;
using Epi = cutlass::epilogue::thread::LinearCombination<double, 1, double, double>;
using Swz = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using SwzB = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using I4 = cutlass::gemm::GemmShape<16, 8, 4>;

// A is m x k, B is k x n, C is m x n column-major (cuBLAS convention). LA / LB: CUTLASS layouts of
// A (m x k) and B (k x n): TN = A RowMajor (K-contiguous rows), B ColumnMajor (K-contiguous columns).
template <class LA, class LB, class TB, class WP, int ST>
struct Cfg {
  using G = cutlass::gemm::device::Gemm<double, LA, double, LB, double, ColM, double,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90, TB, WP, I4, Epi, Swz, ST>;
  using GB = cutlass::gemm::device::GemmBatched<double, LA, double, LB, double, ColM, double,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90, TB, WP, I4, Epi, SwzB, ST>;
  static void name(char* s) { snprintf(s, 64, "%dx%dx%d_%dx%dx%d_s%d", TB::kM, TB::kN, TB::kK, WP::kM, WP::kN, WP::kK, ST); }
};

struct Problem { const char* name; int m, n, k, b; };

static double max_rel(const double* x, const double* y, size_t n) {
  std::vector<double> hx(n), hy(n);
  CK(cudaMemcpy(hx.data(), x, n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hy.data(), y, n * 8, cudaMemcpyDeviceToHost));
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(hx[i]) || !std::isfinite(hy[i])) return INFINITY;   // std::max drops NaN
    num = std::max(num, std::fabs(hx[i] - hy[i])); den = std::max(den, std::fabs(hy[i]));
  }
  return den > 0 ? num / den : num;
}

__global__ void fill(double* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)(i * 2654435761u) ^ seed; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    p[i] = (double)(h & 0xffffff) / (double)0x1000000 - 0.5;
  }
}

template <class C, bool TN>
static void run_cfg(const Problem& p, const double* A, const double* B, double* Cc, const double* Cref, void* ws, size_t ws_bytes) {
  char nm[64]; C::name(nm);
  int lda = TN ? p.k : p.m, ldb = p.k, ldc = p.m;   // TN: A row-major m x k; NN: A col-major m x k
  long long sa = (long long)p.m * p.k, sb = (long long)p.k * p.n, sc = (long long)p.m * p.n;
  if (p.b == 1) {
    typename C::G op;
    typename C::G::Arguments args({p.m, p.n, p.k}, {A, lda}, {B, ldb}, {Cc, ldc}, {Cc, ldc}, {1.0, 0.0});
    if (op.can_implement(args) != cutlass::Status::kSuccess) { printf("gemm,%s,%s,%s,cannot_implement\n", p.name, TN ? "TN" : "NN", nm); return; }
    size_t need = C::G::get_workspace_size(args);
    if (need > ws_bytes) { printf("gemm,%s,%s,%s,workspace\n", p.name, TN ? "TN" : "NN", nm); return; }
    if (op.initialize(args, ws) != cutlass::Status::kSuccess) { printf("gemm,%s,%s,%s,init_failed\n", p.name, TN ? "TN" : "NN", nm); return; }
    if (op.run() != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass run failed: %s %s\n", p.name, nm); exit(4); }
    double ms = time_ms([&] { if (op.run() != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass run failed\n"); exit(4); } }); CK(cudaGetLastError());
    double tf = 2.0 * p.m * (double)p.n * p.k / ms / 1e9;
    printf("gemm,%s,%s,cutlass_%s,%.4f ms,%.2f TFLOPS,max_rel_vs_cublas=%.2e\n", p.name, TN ? "TN" : "NN", nm, ms, tf, max_rel(Cc, Cref, (size_t)sc));
  } else {
    typename C::GB op;
    typename C::GB::Arguments args({p.m, p.n, p.k}, {A, lda}, sa, {B, ldb}, sb, {Cc, ldc}, sc, {Cc, ldc}, sc, {1.0, 0.0}, p.b);
    if (op.can_implement(args) != cutlass::Status::kSuccess) { printf("gemm,%s,%s,%s,cannot_implement\n", p.name, TN ? "TN" : "NN", nm); return; }
    if (op.initialize(args) != cutlass::Status::kSuccess) { printf("gemm,%s,%s,%s,init_failed\n", p.name, TN ? "TN" : "NN", nm); return; }
    if (op.run() != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass run failed: %s %s\n", p.name, nm); exit(4); }
    double ms = time_ms([&] { if (op.run() != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass run failed\n"); exit(4); } }); CK(cudaGetLastError());
    double tf = 2.0 * p.b * p.m * (double)p.n * p.k / ms / 1e9;
    printf("gemm,%s,%s,cutlass_%s,%.4f ms,%.2f TFLOPS,max_rel_vs_cublas=%.2e\n", p.name, TN ? "TN" : "NN", nm, ms, tf, max_rel(Cc, Cref, (size_t)sc * p.b));
  }
}

template <bool TN>
static void run_problem(cublasHandle_t h, const Problem& p) {
  using LA = typename std::conditional<TN, RowM, ColM>::type;
  size_t na = (size_t)p.m * p.k * p.b, nb = (size_t)p.k * p.n * p.b, nc = (size_t)p.m * p.n * p.b;
  double *A, *B, *C, *Cref; CK(cudaMalloc(&A, na * 8)); CK(cudaMalloc(&B, nb * 8)); CK(cudaMalloc(&C, nc * 8)); CK(cudaMalloc(&Cref, nc * 8));
  fill<<<1024, 256>>>(A, na, 1u); fill<<<1024, 256>>>(B, nb, 2u); CK(cudaGetLastError());
  size_t ws_bytes = 64u << 20; void* ws; CK(cudaMalloc(&ws, ws_bytes));
  const double one = 1.0, zero = 0.0;
  cublasOperation_t ta = TN ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = TN ? p.k : p.m;
  auto cub = [&] {
    if (p.b == 1) CB(cublasDgemm(h, ta, CUBLAS_OP_N, p.m, p.n, p.k, &one, A, lda, B, p.k, &zero, Cref, p.m));
    else CB(cublasDgemmStridedBatched(h, ta, CUBLAS_OP_N, p.m, p.n, p.k, &one, A, lda, (long long)p.m * p.k,
                                      B, p.k, (long long)p.k * p.n, &zero, Cref, p.m, (long long)p.m * p.n, p.b));
  };
  double ms = time_ms(cub);
  printf("gemm,%s,%s,cublas,%.4f ms,%.2f TFLOPS\n", p.name, TN ? "TN" : "NN", ms, 2.0 * p.b * p.m * (double)p.n * p.k / ms / 1e9);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<32, 32, 16>, 4>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<128, 64, 16>, cutlass::gemm::GemmShape<64, 32, 16>, 4>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<64, 128, 16>, cutlass::gemm::GemmShape<32, 64, 16>, 4>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<32, 64, 16>, 4>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>, 3>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  run_cfg<Cfg<LA, ColM, cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<16, 16, 16>, 4>, TN>(p, A, B, C, Cref, ws, ws_bytes);
  CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(Cref)); CK(cudaFree(ws));
}

int main(int argc, char** argv) {
  bool do_mma = true, do_gemm = true;
  if (argc > 1 && !strcmp(argv[1], "mma")) do_gemm = false;
  if (argc > 1 && !strcmp(argv[1], "gemm")) do_mma = false;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  int ver = 0; cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasGetVersion(h, &ver));
  printf("device,%s,sm_%d%d,SMs=%d,max_clock_khz=%d,cublas=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, clk, ver);
  if (prop.major != 9) { fprintf(stderr, "hwgemm_gate: sm_90 only (got sm_%d%d)\n", prop.major, prop.minor); return 3; }
  if (do_mma) run_mma(prop.multiProcessorCount);
  if (do_gemm) {
    const Problem ps[] = {{"sq1024", 1024, 1024, 1024, 1}, {"sq2048", 2048, 2048, 2048, 1},
                          {"sq4096", 4096, 4096, 4096, 1}, {"sq8192", 8192, 8192, 8192, 1},
                          {"t_batch", 1024, 1024, 1024, 16}, {"s_batch", 64, 64, 64, 256}};
    for (const auto& p : ps) { run_problem<true>(h, p); run_problem<false>(h, p); fflush(stdout); }
  }
  CB(cublasDestroy(h));
  return 0;
}
