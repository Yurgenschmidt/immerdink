// CUTLASS de-risking for M1 (part 2): the K4 GEMM with the DESIGN §3.5/§4 epilogue,
// run on sm_89 and compared bit-exactly with a CPU reference. Two realizations:
//  - CUTLASS GemmUniversal + the Sm80 EVT epilogue (kernels/mod_epilogue.cuh), K4 until A2;
//  - imd::kernels::K4Gemm (kernels/k4_gemm.cuh), K4 since A2: persistent CTAs (here also
//    with very few resident CTAs, so every CTA loops over many tiles and overlaps its next
//    prologue with the register epilogue), and with the batch count split over launches.
//
// One batched launch per K-chunk covers a residue group: batch z = s*b + lambda, modulus
// l = g0 + z / b from the __constant__ table (imd::kernels::VisitorModP). Chunk 0 stores
// W = acc mod p_l, later chunks W = (W + acc) mod p_l, into u8 planes W_l[lambda][i][n_hat].
// Expected: W = (sum over all of k of r_A r_B) mod p_l, independent of the chunking.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "kernels/k4_gemm.cuh"
#include "kernels/mod_epilogue.cuh"  // also brings in the Sm80 EVT visitors

namespace {

namespace c = imd::constants;
using cute::_1;

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    const cudaError_t err_ = (x);                                                        \
    if (err_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_)); \
      std::exit(2);                                                                      \
    }                                                                                    \
  } while (0)

// DESIGN §4 tile menu entry 128x256x64, 3 stages (72 KB smem: needs the opt-in > 48 KB).
using TB = cutlass::gemm::GemmShape<128, 256, 64>;
using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kStages = 3;
constexpr int kEVTStages = 1;
constexpr int kAlign = 16;  // int8 / u8 elements per 128-bit access
constexpr auto kRound = cutlass::FloatRoundStyle::round_toward_zero;

using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, uint8_t, kAlign,
                                                                         kEVTStages>;
using StrideW = cute::Stride<int64_t, _1, int64_t>;  // (ld = n_hat, 1, plane = m * n_hat)
using LoadW = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, uint8_t, StrideW>;
using StoreW = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, uint8_t, kRound, StrideW>;
using ModP = imd::kernels::VisitorModP;
using EVTStore = cutlass::epilogue::threadblock::Sm80EVT<StoreW, ModP>;
using EVTAdd =
    cutlass::epilogue::threadblock::Sm80EVT<StoreW, cutlass::epilogue::threadblock::Sm80EVT<ModP, LoadW>>;

template <class EVT>
using GemmFor = cutlass::gemm::device::GemmUniversalAdapter<
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlign,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlign,
        uint8_t, cutlass::layout::RowMajor, kAlign, int32_t, int32_t,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, EVT,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
        cutlass::arch::OpMultiplyAddSaturate, kEVTStages>::GemmKernel>;
using GemmStore = GemmFor<EVTStore>;
using GemmAdd = GemmFor<EVTAdd>;
using K4 = imd::kernels::K4Gemm<TB, Warp, kStages>;  // the production K4 kernel (A2)

template <class Gemm>
void run(typename Gemm::Arguments const& args) {
  Gemm op;
  cutlass::Status st = Gemm::can_implement(args);
  if (st == cutlass::Status::kSuccess) st = op.initialize(args, nullptr);
  if (st == cutlass::Status::kSuccess) st = op.run();
  if (st != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS error: %s\n", cutlassGetStatusString(st));
    std::exit(2);
  }
  CUDA_CHECK(cudaGetLastError());
}

// A residue-group GEMM problem: RA [s][lambda][m][k_hat], RB [s][lambda][n_hat][k_hat] (int8),
// W planes [l][lambda][m][n_hat] (u8) for all kNMax moduli.
struct Problem {
  int g0 = 0, slots = 1, b = 1, m = 1, n_hat = 16, k_hat = 16;
  std::vector<std::pair<int, int>> chunks;  // [begin, end) along k_hat, multiples of 16
  std::vector<int8_t> RA, RB;
  size_t a_index(int s, int lam, int i, int h) const {
    return ((size_t(s) * b + lam) * m + i) * k_hat + h;
  }
  size_t b_index(int s, int lam, int j, int h) const {
    return ((size_t(s) * b + lam) * n_hat + j) * k_hat + h;
  }
  size_t w_index(int l, int lam, int i, int j) const {
    return ((size_t(l) * b + lam) * m + i) * n_hat + j;
  }
};

// How K4 runs: the CUTLASS EVT kernel, or K4Gemm with `resident` persistent CTAs and at
// most `zsplit` batches per launch (0: all).
struct Kernel {
  bool k4gemm = false;
  int resident = 0, zsplit = 0;
};

template <bool kAdd>
void launch_k4gemm(const Problem& P, const Kernel& K, const int8_t* A, const int8_t* B,
                   uint8_t* W0, int kc) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(imd::kernels::k4_kernel<K4, false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, K4::kSmemBytes));
    CUDA_CHECK(cudaFuncSetAttribute(imd::kernels::k4_kernel<K4, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, K4::kSmemBytes));
    attr = true;
  }
  const int batch = P.slots * P.b, zs = K.zsplit > 0 ? K.zsplit : batch;
  const int64_t plane_a = int64_t(P.m) * P.k_hat, plane_b = int64_t(P.n_hat) * P.k_hat,
                plane_w = int64_t(P.m) * P.n_hat;
  for (int z0 = 0; z0 < batch; z0 += zs) {
    const int zc = std::min(zs, batch - z0);
    auto p = K4::make_params(P.m, P.n_hat, kc, P.k_hat, P.k_hat, P.n_hat, zc, K.resident);
    K4::set_call(p, A + z0 * plane_a, B + z0 * plane_b, W0 + z0 * plane_w, plane_a, plane_b,
                 plane_w, P.g0, P.b, z0);
    imd::kernels::k4_kernel<K4, kAdd><<<p.grid, K4::kThreads, K4::kSmemBytes>>>(p);
    CUDA_CHECK(cudaGetLastError());
  }
}

std::vector<uint8_t> gpu_k4(const Problem& P, const Kernel& K) {
  int8_t *dA = nullptr, *dB = nullptr;
  uint8_t* dW = nullptr;
  const size_t w_bytes = size_t(c::kNMax) * P.b * P.m * P.n_hat;
  CUDA_CHECK(cudaMalloc(&dA, P.RA.size()));
  CUDA_CHECK(cudaMalloc(&dB, P.RB.size()));
  CUDA_CHECK(cudaMalloc(&dW, w_bytes));
  CUDA_CHECK(cudaMemcpy(dA, P.RA.data(), P.RA.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, P.RB.data(), P.RB.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dW, 0xAB, w_bytes));  // planes outside the group must stay 0xAB

  const int batch = P.slots * P.b;
  const int64_t ld_w = P.n_hat, plane = int64_t(P.m) * P.n_hat;
  uint8_t* W0 = dW + size_t(P.g0) * P.b * plane;  // plane of (l = g0, lambda = 0)
  const ModP::Arguments mod{P.g0, P.b};
  for (size_t c = 0; c < P.chunks.size(); ++c) {
    const auto [k0, k1] = P.chunks[c];
    const cutlass::gemm::GemmCoord size{P.m, P.n_hat, k1 - k0};
    const int8_t* A = dA + k0;
    const int8_t* B = dB + k0;
    if (K.k4gemm) {
      if (c == 0)
        launch_k4gemm<false>(P, K, A, B, W0, k1 - k0);
      else
        launch_k4gemm<true>(P, K, A, B, W0, k1 - k0);
    } else if (c == 0) {
      typename EVTStore::Arguments cb{mod, {W0, {ld_w, _1{}, plane}}};
      run<GemmStore>({cutlass::gemm::GemmUniversalMode::kBatched, size, batch, cb, A, B, nullptr,
                      nullptr, int64_t(P.m) * P.k_hat, int64_t(P.n_hat) * P.k_hat, 0, 0, P.k_hat,
                      P.k_hat, 0, 0});
    } else {
      typename EVTAdd::Arguments cb{{{W0, uint8_t(0), {ld_w, _1{}, plane}}, mod},
                                    {W0, {ld_w, _1{}, plane}}};
      run<GemmAdd>({cutlass::gemm::GemmUniversalMode::kBatched, size, batch, cb, A, B, nullptr,
                    nullptr, int64_t(P.m) * P.k_hat, int64_t(P.n_hat) * P.k_hat, 0, 0, P.k_hat,
                    P.k_hat, 0, 0});
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> W(w_bytes);
  CUDA_CHECK(cudaMemcpy(W.data(), dW, w_bytes, cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dW);
  return W;
}

int64_t dot(const Problem& P, int s, int lam, int i, int j, int h0, int h1) {
  int64_t acc = 0;
  for (int h = h0; h < h1; ++h)
    acc += int64_t{P.RA[P.a_index(s, lam, i, h)]} * P.RB[P.b_index(s, lam, j, h)];
  return acc;
}

int failures = 0;

void check_with(const char* name, const Problem& P, const Kernel& K) {
  const auto W = gpu_k4(P, K);
  size_t bad = 0, checked = 0;
  for (int l = 0; l < c::kNMax; ++l) {
    const int s = l - P.g0;
    const bool in_group = s >= 0 && s < P.slots;
    const int64_t p = c::kTables.moduli[l];
    for (int lam = 0; lam < P.b; ++lam)
      for (int i = 0; i < P.m; ++i)
        for (int j = 0; j < P.n_hat; ++j) {
          const int64_t want = in_group ? ((dot(P, s, lam, i, j, 0, P.k_hat) % p) + p) % p : 0xAB;
          const uint8_t got = W[P.w_index(l, lam, i, j)];
          ++checked;
          if (got != want && bad++ < 5)
            std::fprintf(stderr, "  %s: l=%d lam=%d i=%d j=%d: got %u want %lld\n", name, l, lam,
                         i, j, got, (long long)want);
        }
  }
  std::printf("%-44s %-24s %zu bytes, %zu mismatches\n", name,
              !K.k4gemm ? "CUTLASS EVT" : K.resident > 8 ? "K4Gemm" : "K4Gemm, few CTAs + zsplit",
              checked, bad);
  failures += bad != 0;
}

// Every problem through the EVT kernel, K4Gemm with one CTA per SM, and K4Gemm with 3
// persistent CTAs and the batch count split into launches of 2.
void check(const char* name, const Problem& P) {
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  check_with(name, P, Kernel{});
  check_with(name, P, Kernel{true, sms, 0});
  check_with(name, P, Kernel{true, 3, 2});
}

// Uniform residues in p_l's symmetric range [-floor(p/2), ceil(p/2) - 1] (DESIGN §3.4).
void fill_residues(Problem& P, std::mt19937& rng) {
  P.RA.assign(size_t(P.slots) * P.b * P.m * P.k_hat, 0);
  P.RB.assign(size_t(P.slots) * P.b * P.n_hat * P.k_hat, 0);
  for (int s = 0; s < P.slots; ++s) {
    const int p = int(c::kTables.moduli[P.g0 + s]);
    std::uniform_int_distribution<int> r(-(p / 2), (p + 1) / 2 - 1);
    for (int lam = 0; lam < P.b; ++lam) {
      for (int i = 0; i < P.m; ++i)
        for (int h = 0; h < P.k_hat; ++h) P.RA[P.a_index(s, lam, i, h)] = int8_t(r(rng));
      for (int j = 0; j < P.n_hat; ++j)
        for (int h = 0; h < P.k_hat; ++h) P.RB[P.b_index(s, lam, j, h)] = int8_t(r(rng));
    }
  }
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  {  // All 20 moduli in one launch per chunk, b = 2, three chunks (store, add, add).
    Problem P;
    P.g0 = 0, P.slots = c::kNMax, P.b = 2, P.m = 37, P.n_hat = 48, P.k_hat = 96;
    P.chunks = {{0, 32}, {32, 64}, {64, 96}};
    fill_residues(P, rng);
    check("all moduli, b=2, 3 chunks", P);
  }
  {  // A group in the middle (g0 = 5), b = 3, two tiles in each of M and N.
    Problem P;
    P.g0 = 5, P.slots = 3, P.b = 3, P.m = 130, P.n_hat = 272, P.k_hat = 64;
    P.chunks = {{0, 16}, {16, 64}};
    fill_residues(P, rng);
    check("group g0=5 |G|=3, b=3, 2x2 tiles, 2 chunks", P);
    P.chunks = {{0, 64}};
    check("same, single store-mod chunk", P);
  }
  {  // The reachable extremes of the epilogue input v (DESIGN §3.5, params epilogue_range):
     // chunk 0 (16 wide) leaves W = p-1 at (0,0) and 0 at (0,1); chunk 1 (K_c' wide) adds
     // lo^2 K_c' at (0,0) -> v = v_max, and lo*hi K_c' at (0,1) -> v = v_min.
    Problem P;
    P.g0 = 0, P.slots = c::kNMax, P.b = 1, P.m = 16, P.n_hat = 16;
    P.k_hat = 16 + int(c::kKcPrime);
    P.chunks = {{0, 16}, {16, P.k_hat}};
    P.RA.assign(size_t(P.slots) * P.m * P.k_hat, 0);
    P.RB.assign(size_t(P.slots) * P.n_hat * P.k_hat, 0);
    for (int s = 0; s < P.slots; ++s) {
      const int p = int(c::kTables.moduli[s]), lo = -(p / 2), hi = (p + 1) / 2 - 1;
      P.RA[P.a_index(s, 0, 0, 0)] = -1;
      P.RB[P.b_index(s, 0, 0, 0)] = 1;
      for (int h = 16; h < P.k_hat; ++h) {
        P.RA[P.a_index(s, 0, 0, h)] = int8_t(lo);
        P.RA[P.a_index(s, 0, 1, h)] = int8_t(hi);
        P.RB[P.b_index(s, 0, 0, h)] = int8_t(lo);
        P.RB[P.b_index(s, 0, 1, h)] = int8_t(hi);
      }
      // The test must actually reach both ends of the certified range.
      const int64_t w00 = ((dot(P, s, 0, 0, 0, 0, 16) % p) + p) % p;
      const int64_t w01 = ((dot(P, s, 0, 0, 1, 0, 16) % p) + p) % p;
      const int64_t vmax = w00 + dot(P, s, 0, 0, 0, 16, P.k_hat);
      const int64_t vmin = w01 + dot(P, s, 0, 0, 1, 16, P.k_hat);
      if (vmax != c::kTables.epi_vmax[s] || vmin != c::kTables.epi_vmin[s]) {
        std::fprintf(stderr, "p=%d: test inputs miss the epilogue extremes\n", p);
        ++failures;
      }
    }
    check("epilogue extremes v_min/v_max, K = K_c'", P);
  }
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
