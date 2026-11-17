// hwgemm_bitid: is an FP64 tensor-core (DMMA) GEMM bit-identical to DIRECT? (issue #75, NOTES-HWGEMM §3)
//
// DIRECT (src/contract.cu native_dot) computes every output as acc = +0, then acc = fma(a_h, b_h, acc) for
// h = 0, 1, ..., k-1. If one mma.sync f64 instruction is itself such a chain (Khattak-Mikaitis, arXiv:2512.07004),
// a GEMM that feeds K in ascending order with the accumulator started at +0 (no split-K, no sliced-K) gives
// exactly DIRECT's bits. Correctness only; no timing.
//
// Part 1, instruction: one warp, one 8x8 tile, mma.sync m8n8k4 (sm_80+; on sm_90 also m16n8k4/k8/k16) chained
//   over K in steps of the instruction's k. Each result is compared bitwise with three host models:
//     seq   the ascending fma chain from +0 (DIRECT)
//     rev   the chain over each instruction's k products in descending order
//     fused per instruction: C + (exact sum of its k products), one rounding (__float128 approximation)
//   on several input distributions chosen to separate the models (wide exponents, cancellation, subnormal
//   products, signed zeros).
// Part 2, kernel: CUTLASS 2.x device::Gemm (Sm80 8x8x4 on sm_8x, Sm90 16x8x4 on sm_90), TN and NN, vs the
//   ascending fma chain on the device, bitwise, on odd shapes (k not a multiple of the K tile).
// Exit code 0 iff every comparison against `seq` matched.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_89 (or sm_90) -I<cutlass>/include hwgemm_bitid.cu -lquadmath -o hwgemm_bitid
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <quadmath.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint64_t bits(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }

// ------------------------------------------------------------------ inputs --
enum Dist { kUniform, kWideExp, kCancel, kSubnormal, kZeros, kDistCount };
static const char* dist_name[kDistCount] = {"uniform", "wide_exp", "cancel", "subnormal", "zeros"};

static double draw(Dist d, std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<int> e(-60, 60), es(-560, -500), coin(0, 3);
  switch (d) {
    case kUniform: return u(g);
    case kWideExp: return std::ldexp(u(g), e(g));
    case kCancel: {                       // large terms of both signs that cancel, plus small ones
      double m = 1.0 + std::ldexp(std::floor(std::fabs(u(g)) * 1024.0), -10);
      return coin(g) == 0 ? std::ldexp(u(g), -40) : (coin(g) & 1 ? m : -m) * std::ldexp(1.0, 20);
    }
    case kSubnormal: return std::ldexp(u(g), es(g));   // products ~2^-1060..-1000: subnormal / underflow
    default: { int c = coin(g); return c == 0 ? 0.0 : c == 1 ? -0.0 : u(g); }
  }
}

// ----------------------------------------------------------- part 1: mma --
// m8n8k4 .f64 fragments (PTX ISA): A row-major 8x4: lane holds A[lane/4][lane%4]; B col-major 4x8: lane holds
// B[lane%4][lane/4]; C/D 8x8: lane holds D[lane/4][2*(lane%4) + {0,1}].
__global__ void mma884_chain(const double* A, const double* B, double* D, int K) {   // A 8xK row-major, B Kx8 col-major
  const int lane = threadIdx.x;
  double d0 = 0.0, d1 = 0.0;
  for (int k0 = 0; k0 < K; k0 += 4) {
    double a = A[(lane / 4) * K + k0 + lane % 4];
    double b = B[(lane / 4) * K + k0 + lane % 4];    // column n = lane/4, row k = lane%4
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+d"(d0), "+d"(d1) : "d"(a), "d"(b));
#endif
  }
  D[(lane / 4) * 8 + 2 * (lane % 4)] = d0;
  D[(lane / 4) * 8 + 2 * (lane % 4) + 1] = d1;
}

static double model(int which, const double* A, const double* B, int K, int i, int j) {
  double acc = 0.0;
  for (int k0 = 0; k0 < K; k0 += 4) {
    if (which == 0) {
      for (int h = k0; h < k0 + 4; ++h) acc = std::fma(A[i * K + h], B[j * K + h], acc);
    } else if (which == 1) {
      for (int h = k0 + 3; h >= k0; --h) acc = std::fma(A[i * K + h], B[j * K + h], acc);
    } else {
      __float128 s = acc;
      for (int h = k0; h < k0 + 4; ++h) s += (__float128)A[i * K + h] * (__float128)B[j * K + h];
      acc = (double)s;
    }
  }
  return acc;
}

static bool part1(int trials) {
  printf("part 1: mma.sync m8n8k4 f64 chain vs host models (8x8 tile, K in {4, 8, 32})\n");
  bool ok = true;
  double *dA, *dB, *dD; CK(cudaMalloc(&dA, 8 * 32 * 8)); CK(cudaMalloc(&dB, 8 * 32 * 8)); CK(cudaMalloc(&dD, 64 * 8));
  for (int d = 0; d < kDistCount; ++d) {
    for (int K : {4, 8, 32}) {
      long n = 0, eq[3] = {0, 0, 0};
      std::mt19937_64 g(1000 + 10 * d + K);
      for (int t = 0; t < trials; ++t) {
        std::vector<double> A(8 * K), B(8 * K), D(64);
        for (auto& x : A) x = draw((Dist)d, g);
        for (auto& x : B) x = draw((Dist)d, g);
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
        mma884_chain<<<1, 32>>>(dA, dB, dD, K); CK(cudaGetLastError());
        CK(cudaMemcpy(D.data(), dD, 64 * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
          ++n;
          for (int m = 0; m < 3; ++m) eq[m] += bits(D[i * 8 + j]) == bits(model(m, A.data(), B.data(), K, i, j));
        }
      }
      printf("  %-10s K=%-3d outputs=%-7ld  =seq %-7ld =rev %-7ld =fused %-7ld%s\n", dist_name[d], K, n, eq[0], eq[1], eq[2],
             eq[0] == n ? "" : "   <-- seq mismatch");
      ok = ok && eq[0] == n;
    }
  }
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dD));
  return ok;
}

// -------------------------------------------------------- part 2: CUTLASS --
__global__ void ref_chain(const double* A, const double* B, double* C, int m, int n, int k, bool tn) {
  // C col-major m x n; A: TN row-major m x k (lda k), NN col-major m x k (lda m); B col-major k x n (ldb k)
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (i >= m || j >= n) return;
  double acc = 0.0;
  for (int h = 0; h < k; ++h) acc = __fma_rn(tn ? A[(size_t)i * k + h] : A[(size_t)h * m + i], B[(size_t)j * k + h], acc);
  C[(size_t)j * m + i] = acc;
}

#if defined(BITID_SM90)
using Arch = cutlass::arch::Sm90;
using Inst = cutlass::gemm::GemmShape<16, 8, 4>;
#else
using Arch = cutlass::arch::Sm80;
using Inst = cutlass::gemm::GemmShape<8, 8, 4>;
#endif
using Epi = cutlass::epilogue::thread::LinearCombination<double, 1, double, double>;
template <class LA>
using G = cutlass::gemm::device::Gemm<double, LA, double, cutlass::layout::ColumnMajor, double, cutlass::layout::ColumnMajor,
    double, cutlass::arch::OpClassTensorOp, Arch, cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<32, 32, 16>,
    Inst, Epi, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

template <bool TN>
static bool part2_one(int m, int n, int k, Dist d) {
  using LA = typename std::conditional<TN, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
  std::mt19937_64 g(7 + m + n + k + d);
  std::vector<double> A((size_t)m * k), B((size_t)k * n);
  for (auto& x : A) x = draw(d, g);
  for (auto& x : B) x = draw(d, g);
  double *dA, *dB, *dC, *dR;
  CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, B.size() * 8));
  CK(cudaMalloc(&dC, (size_t)m * n * 8)); CK(cudaMalloc(&dR, (size_t)m * n * 8));
  CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
  ref_chain<<<dim3((m + 127) / 128, n), 128>>>(dA, dB, dR, m, n, k, TN); CK(cudaGetLastError());
  G<LA> op;
  typename G<LA>::Arguments args({m, n, k}, {dA, TN ? k : m}, {dB, k}, {dC, m}, {dC, m}, {1.0, 0.0});
  if (op.can_implement(args) != cutlass::Status::kSuccess || op.initialize(args) != cutlass::Status::kSuccess ||
      op() != cutlass::Status::kSuccess) { printf("  cutlass failed m=%d n=%d k=%d\n", m, n, k); exit(4); }
  CK(cudaDeviceSynchronize());
  std::vector<double> C((size_t)m * n), R((size_t)m * n);
  CK(cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(R.data(), dR, R.size() * 8, cudaMemcpyDeviceToHost));
  size_t same = 0; double worst = 0;
  for (size_t q = 0; q < C.size(); ++q) {
    if (bits(C[q]) == bits(R[q])) ++same;
    else if (std::isfinite(R[q]) && R[q] != 0) worst = std::max(worst, std::fabs(C[q] - R[q]) / std::fabs(R[q]));
  }
  printf("  %s %-10s m=%-4d n=%-4d k=%-5d  bit-identical %zu / %zu%s", TN ? "TN" : "NN", dist_name[d], m, n, k, same, C.size(),
         same == C.size() ? "\n" : "");
  if (same != C.size()) printf("   worst rel diff %.2e  <-- mismatch\n", worst);
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC)); CK(cudaFree(dR));
  return same == C.size();
}

static bool part2() {
  printf("part 2: CUTLASS device::Gemm (%s, 64x64x16, 3 stages) vs ascending fma chain, bitwise\n",
#if defined(BITID_SM90)
         "Sm90 16x8x4"
#else
         "Sm80 8x8x4"
#endif
  );
  bool ok = true;
  const int shapes[][3] = {{64, 64, 16}, {200, 136, 333}, {129, 65, 1000}, {256, 256, 4097}};
  for (auto& s : shapes)
    for (int d = 0; d < kDistCount; ++d) {
      ok = part2_one<true>(s[0], s[1], s[2], (Dist)d) && ok;
      ok = part2_one<false>(s[0], s[1], s[2], (Dist)d) && ok;
    }
  return ok;
}

int main(int argc, char** argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 200;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device %s sm_%d%d\n", p.name, p.major, p.minor);
  bool ok1 = part1(trials), ok2 = part2();
  printf("RESULT: instruction %s, CUTLASS kernel %s (vs DIRECT's ascending fma chain)\n",
         ok1 ? "bit-identical" : "DIFFERS", ok2 ? "bit-identical" : "DIFFERS");
  return ok1 && ok2 ? 0 : 1;
}
