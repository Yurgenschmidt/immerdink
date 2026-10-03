// CUTLASS de-risking for M1 (part 1): the DESIGN §4 GEMM configuration runs on sm_89 and
// is exact. CUTLASS 2.x GemmUniversal, kBatched, arch::Sm80, int8 TN (A RowMajor,
// B ColumnMajor, both K-contiguous), OpMultiplyAddSaturate, int32 output, compared
// bit-exactly with a CPU reference. Also pins the .satfinite behaviour that makes K_c
// necessary for every modulus (DESIGN §2): exact at K = K_c', saturated just past 2^31-1.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "immerdink/detail/constants.hpp"

namespace {

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    const cudaError_t err_ = (x);                                                        \
    if (err_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_)); \
      std::exit(2);                                                                      \
    }                                                                                    \
  } while (0)

using Gemm = cutlass::gemm::device::GemmUniversal<
    int8_t, cutlass::layout::RowMajor,     // A: m x k, K-contiguous
    int8_t, cutlass::layout::ColumnMajor,  // B: k x n, K-contiguous
    int32_t, cutlass::layout::RowMajor,    // D: m x n
    int32_t,                               // accumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate>;

template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// D[z][i][j] = sum_h A[z][i][h] * B[z][j][h] on the GPU (A: [batch][m][k], B: [batch][n][k]).
std::vector<int32_t> gpu_gemm(int m, int n, int k, int batch, const std::vector<int8_t>& A,
                              const std::vector<int8_t>& B) {
  DeviceBuffer<int8_t> dA(A.size()), dB(B.size());
  DeviceBuffer<int32_t> dD(size_t(batch) * m * n);
  CUDA_CHECK(cudaMemcpy(dA.ptr, A.data(), A.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB.ptr, B.data(), B.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD.ptr, 0x5A, size_t(batch) * m * n * sizeof(int32_t)));
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched, {m, n, k}, batch,
                                {1, 0}, dA.ptr, dB.ptr, dD.ptr, dD.ptr, int64_t(m) * k,
                                int64_t(n) * k, int64_t(m) * n, int64_t(m) * n, k, k, n, n);
  Gemm op;
  cutlass::Status st = Gemm::can_implement(args);
  if (st == cutlass::Status::kSuccess) st = op.initialize(args, nullptr);
  if (st == cutlass::Status::kSuccess) st = op.run();
  if (st != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS error: %s\n", cutlassGetStatusString(st));
    std::exit(2);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int32_t> D(size_t(batch) * m * n);
  CUDA_CHECK(cudaMemcpy(D.data(), dD.ptr, D.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  return D;
}

// Exact sums in int64 (never saturating).
std::vector<int64_t> cpu_gemm(int m, int n, int k, int batch, const std::vector<int8_t>& A,
                              const std::vector<int8_t>& B) {
  std::vector<int64_t> D(size_t(batch) * m * n);
  for (int z = 0; z < batch; ++z)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        const int8_t* a = &A[(size_t(z) * m + i) * k];
        const int8_t* b = &B[(size_t(z) * n + j) * k];
        int64_t s = 0;
        for (int h = 0; h < k; ++h) s += int64_t{a[h]} * b[h];
        D[(size_t(z) * m + i) * n + j] = s;
      }
  return D;
}

int failures = 0;

void expect_equal(const char* name, const std::vector<int32_t>& got,
                  const std::vector<int64_t>& want) {
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (int64_t{got[i]} != want[i] && bad++ < 5)
      std::fprintf(stderr, "  %s[%zu]: got %d, want %lld\n", name, i, got[i], (long long)want[i]);
  std::printf("%-34s %zu elements, %zu mismatches\n", name, got.size(), bad);
  failures += bad != 0;
}

}  // namespace

int main() {
  namespace c = imd::constants;
  std::mt19937 rng(20261002);
  std::uniform_int_distribution<int> any_int8(-128, 127);

  {  // Random full-range int8 operands, predicated M, several batches.
    const int m = 100, n = 72, k = 272, batch = 3;
    std::vector<int8_t> A(size_t(batch) * m * k), B(size_t(batch) * n * k);
    for (auto& x : A) x = int8_t(any_int8(rng));
    for (auto& x : B) x = int8_t(any_int8(rng));
    expect_equal("random batched TN", gpu_gemm(m, n, k, batch, A, B), cpu_gemm(m, n, k, batch, A, B));
  }
  {  // Extremes at K = K_c': the largest exact sums, +128^2 K_c' and -128*127 K_c'.
    const int m = 16, n = 16, k = int(c::kKcPrime), batch = 1;
    std::vector<int8_t> A(size_t(m) * k), B(size_t(n) * k);
    for (int i = 0; i < m; ++i)
      for (int h = 0; h < k; ++h) A[size_t(i) * k + h] = int8_t(i % 2 ? 127 : -128);
    for (int j = 0; j < n; ++j)
      for (int h = 0; h < k; ++h) B[size_t(j) * k + h] = int8_t(j % 2 ? 127 : -128);
    const auto got = gpu_gemm(m, n, k, batch, A, B);
    expect_equal("K = K_c' extremes", got, cpu_gemm(m, n, k, batch, A, B));
    if (got[0] != 128 * 128 * c::kKcPrime || got[1] != -128 * 127 * c::kKcPrime) ++failures;
  }
  {  // Past K_c the true sum 128^2 * 131072 = 2^31 does not fit: .satfinite clamps it.
    const int m = 16, n = 16, k = 131072, batch = 1;
    std::vector<int8_t> A(size_t(m) * k, int8_t(-128)), B(size_t(n) * k, int8_t(-128));
    const auto got = gpu_gemm(m, n, k, batch, A, B);
    size_t sat = 0;
    for (int32_t v : got) sat += v == INT32_MAX;
    std::printf("%-34s %zu of %zu elements saturated to INT32_MAX (true sum 2^31)\n",
                "K = 131072 overflow (.satfinite)", sat, got.size());
    failures += sat != got.size();
  }
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
