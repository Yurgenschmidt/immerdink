// Known-answer test of the Hopper K4 epilogue candidate on any sm_80+ GPU (issue 17, docs/NOTES-17.md).
//
// wgmma cannot run on the RTX 4090, but its accumulator layout is data, not hardware: thread t of a
// warpgroup holds, in register v, the element (16 (t / 32) + (t % 32) / 4 + 8 ((v / 2) % 2),
// 8 (v / 4) + 2 (t % 4) + v % 2) of the 64 x N tile (checked against CUTLASS's CuTe tables by
// tests/test_wgmma_layout.py). This test places a random integer matrix in the registers of a warpgroup in that
// order and runs HopperEpilogue (src/k4/hopper/epilogue_wg.cuh); it places the same matrix in the registers of
// the production kernel's mma.sync fragment order and runs K4Gemm's own epilogue. Both must write the same u8 plane,
// equal to the host reference ((acc [+ W]) mod p), for every modulus (p = 256 included), store and add mode, with
// partial rows and columns. This is the "permuted mma.sync layout" check of the issue; the same epilogue on
// accumulators of a real wgmma is rung 0 of the R1 campaign (docs/NOTES-17.md).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/gemm/gemm.h"
#include "device/constants.cuh"
#include "host/k4_config.hpp"
#include "k4/hopper/epilogue_wg.cuh"
#include "kernels/k4_gemm.cuh"

namespace {

#define CUDA_CHECK(x)                                                                        \
  do {                                                                                       \
    const cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                                 \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                          \
    }                                                                                        \
  } while (0)

using G = imd::kernels::K4Gemm<cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 3>;
constexpr int kM = 128, kN = 256;  // the tile both kernels cover
static_assert(G::kThreads == 256 && G::WarpCount::kM == 2 && G::kMi == 4 && G::kNj == 8, "tile geometry");

// Production: registers in CUTLASS MmaTensorOp order, element (16 mi + 8 h + g, 8 nj + 2 t + e) of warp (wm, wn).
template <bool kAdd>
__global__ void prod_kernel(G::Params p, const int32_t* acc, uint8_t* plane, int l) {
  G::Tile tile{0, 0, 0, l};
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  const int wm = warp % G::WarpCount::kM, wn = warp / G::WarpCount::kM;
  G::Mma::FragmentC f;
#pragma unroll
  for (int mi = 0; mi < G::kMi; ++mi)
#pragma unroll
    for (int nj = 0; nj < G::kNj; ++nj)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          f[4 * (mi + nj * G::kMi) + 2 * h + e] =
              acc[(wm * 64 + 16 * mi + 8 * h + g) * kN + wn * 64 + 8 * nj + 2 * t + e];
  p.W = plane;
  G::epilogue<kAdd>(p, tile, f, warp, lane);
}

// Candidate: registers in wgmma order for a warpgroup of 64 rows x kNW columns; two warpgroups cover 128 rows.
template <int kNW, bool kAdd>
__global__ void wg_kernel(const int32_t* acc, uint8_t* plane, int l, int m, int n, int col0) {
  const int wgi = threadIdx.x / 128, warp = (threadIdx.x % 128) / 32, lane = threadIdx.x % 32;
  const int g = lane >> 2, q = lane & 3;
  int32_t r[kNW / 2];
#pragma unroll
  for (int v = 0; v < kNW / 2; ++v) {
    const int row = 64 * wgi + 16 * warp + g + 8 * ((v / 2) % 2);
    const int col = col0 + 8 * (v / 4) + 2 * q + (v % 2);
    r[v] = acc[row * kN + col];
  }
  const uint32_t mod = imd::device::c_tables.moduli[l];
  imd::kernels::HopperEpilogue::run<kNW, kAdd>(
      r, warp, lane, 64 * wgi, col0, plane, kN, m, n, mod, imd::device::c_tables.epi_bias[l],
      imd::device::c_tables.epi_magic[l], imd::device::c_fold.f[l]);
}

int fails = 0;

template <bool kAdd>
void run_case(int l, int m, int n, std::mt19937_64& rng) {
  namespace c = imd::constants;
  const int p = int(c::kTables.moduli[l]);
  const int64_t lo = c::kTables.epi_vmin[l], hi = c::kTables.epi_vmax[l] - (kAdd ? 255 : 0);
  std::vector<int32_t> acc(kM * kN);
  std::vector<uint8_t> old(kM * kN), want(kM * kN, 0xEE);
  std::uniform_int_distribution<int64_t> d(lo, hi);
  for (int i = 0; i < kM; ++i)
    for (int j = 0; j < kN; ++j) {
      // edge values often, uniform otherwise
      const int64_t v = (rng() % 7 == 0) ? (rng() % 2 ? lo : hi) : d(rng);
      acc[i * kN + j] = int32_t(v);
      old[i * kN + j] = uint8_t(rng() % p);
      if (i < m && j < n) {
        int64_t x = v + (kAdd ? old[i * kN + j] : 0);
        want[i * kN + j] = uint8_t(((x % p) + p) % p);
      }
    }
  int32_t* dacc;
  uint8_t *pa, *pb;
  CUDA_CHECK(cudaMalloc(&dacc, acc.size() * 4));
  CUDA_CHECK(cudaMalloc(&pa, old.size()));
  CUDA_CHECK(cudaMalloc(&pb, old.size()));
  CUDA_CHECK(cudaMemcpy(dacc, acc.data(), acc.size() * 4, cudaMemcpyHostToDevice));
  std::vector<uint8_t> init = kAdd ? old : std::vector<uint8_t>(old.size(), 0xEE);
  CUDA_CHECK(cudaMemcpy(pa, init.data(), init.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pb, init.data(), init.size(), cudaMemcpyHostToDevice));
  G::Params gp = G::make_params(m, n, 64, 64, 64, kN, 1, 1);
  G::set_call(gp, nullptr, nullptr, nullptr, 0, 0, 0, l, 1, 0);
  prod_kernel<kAdd><<<1, 256>>>(gp, dacc, pa, l);
  // the candidate: groups of 256 columns in one go, and (second plane) the same tile as 64-column pieces
  wg_kernel<256, kAdd><<<1, 256>>>(dacc, pb, l, m, n, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> ga(old.size()), gb(old.size());
  CUDA_CHECK(cudaMemcpy(ga.data(), pa, ga.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gb.data(), pb, gb.size(), cudaMemcpyDeviceToHost));
  // rows / columns outside (m, n) keep their initial bytes
  size_t bad_prod = 0, bad_wg = 0, differ = 0;
  for (int i = 0; i < kM; ++i)
    for (int j = 0; j < kN; ++j) {
      const size_t k = size_t(i) * kN + j;
      const uint8_t expect = (i < m && j < n) ? want[k] : init[k];
      bad_prod += ga[k] != expect;
      bad_wg += gb[k] != expect;
      differ += ga[k] != gb[k];
    }
  if (bad_prod || bad_wg || differ) {
    std::fprintf(stderr, "FAIL modulus %d (p = %d) %s m=%d n=%d: production wrong %zu, wgmma-order wrong %zu, differ %zu\n",
                 l, p, kAdd ? "add" : "store", m, n, bad_prod, bad_wg, differ);
    ++fails;
  }
  cudaFree(dacc);
  cudaFree(pa);
  cudaFree(pb);
}

// narrower tiles and a column offset: the candidate on 64- and 128-column pieces of the same data
template <int kNW>
void run_piece(int l, std::mt19937_64& rng) {
  namespace c = imd::constants;
  const int p = int(c::kTables.moduli[l]);
  const int64_t lo = c::kTables.epi_vmin[l], hi = c::kTables.epi_vmax[l];
  std::vector<int32_t> acc(kM * kN);
  std::uniform_int_distribution<int64_t> d(lo, hi);
  for (auto& x : acc) x = int32_t(d(rng));
  int32_t* dacc;
  uint8_t* pb;
  CUDA_CHECK(cudaMalloc(&dacc, acc.size() * 4));
  CUDA_CHECK(cudaMalloc(&pb, acc.size()));
  CUDA_CHECK(cudaMemcpy(dacc, acc.data(), acc.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(pb, 0xEE, acc.size()));
  for (int col0 = 0; col0 + kNW <= kN; col0 += kNW) wg_kernel<kNW, false><<<1, 256>>>(dacc, pb, l, kM, kN, col0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> g(acc.size());
  CUDA_CHECK(cudaMemcpy(g.data(), pb, g.size(), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t k = 0; k < g.size(); ++k) bad += g[k] != uint8_t(((int64_t(acc[k]) % p) + p) % p);
  if (bad) {
    std::fprintf(stderr, "FAIL pieces of %d columns, modulus %d: %zu wrong bytes\n", kNW, l, bad);
    ++fails;
  }
  cudaFree(dacc);
  cudaFree(pb);
}

}  // namespace

int main() {
  std::mt19937_64 rng(20261015);
  int cases = 0;
  for (int l = 0; l < imd::constants::kNMax; ++l)
    for (const auto& mn : {std::pair<int, int>{kM, kN}, {100, 240}, {61, 16}, {128, 192}}) {
      run_case<false>(l, mn.first, mn.second, rng);
      run_case<true>(l, mn.first, mn.second, rng);
      cases += 2;
    }
  for (int l = 0; l < imd::constants::kNMax; l += 3) {
    run_piece<64>(l, rng);
    run_piece<128>(l, rng);
    cases += 2;
  }
  if (fails) {
    std::fprintf(stderr, "%d of %d cases FAILED\n", fails, cases);
    return 1;
  }
  std::printf("PASS  %d cases: the wgmma-order epilogue and K4's own epilogue write the same bytes, equal to the host reference\n", cases);
  return 0;
}
