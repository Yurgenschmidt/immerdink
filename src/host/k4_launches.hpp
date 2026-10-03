// K4's launch structure (DESIGN §3.5, §4): which batched int8 GEMM launches one
// imd_contract makes, as a CUDA-free function of the plan's GEMM view. gemm_k4.cu builds
// and runs exactly these launches (k4_for_each_launch); libimmerdink_host counts them for
// the lowering summary and the plan's host bytes (k4_counts), so the two cannot drift.
#pragma once

#include <algorithm>
#include <cstdint>

#include "checked.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {

// K4 tile width along n (K4Gemm threadblock N; gemm_k4.cu static_asserts it).
constexpr int64_t kK4TileN = 256;
constexpr int64_t kK4MaxBatch = 65535;   // gridDim.z
constexpr int64_t kK4MaxNTiles = 65535;  // gridDim.y (identity swizzle: N tiles)

// Launch limits of K4. Production values are the hardware / DESIGN limits: K-chunk K_c'
// (§3.5), batch count <= 65535 (gridDim.z) and N tiles <= 65535 (gridDim.y) per launch.
// Only the test hook imd_test_set_limits (test_hooks.cpp) changes libimmerdink's copy, to
// exercise the chunk / batch / n-hat splits on small shapes; results must not change. The
// host queries always use the production values.
struct LaunchLimits {
  int64_t k_chunk;      // multiple of 16 in [16, K_c']
  int64_t max_batch;    // [1, 65535]
  int64_t max_n_tiles;  // [1, 65535]
};
constexpr LaunchLimits production_launch_limits() {
  return {imd::constants::kKcPrime, kK4MaxBatch, kK4MaxNTiles};
}

// One K4 launch of a call: K-chunk [k0, k0 + kc), batches [z0, z0 + zc) of the group,
// columns [j0, j0 + nc) of n_hat. Its CUTLASS parameters depend only on the shape key
// (kc, zc, nc; store-mod for chunk 0, add-mod after).
struct K4Launch {
  int64_t k0, kc, z0, zc, j0, nc;
};

// The K4 grid of a plan: the post-swap n_hat, k_hat, b and the limits at plan creation.
struct K4Grid {
  int64_t n_hat = 0, k_hat = 0, b = 0;
  int64_t chunk = 0, max_batch = 0, n_per_launch = 0;
};
// tile_n: the K4 kernel's tile width (kK4TileN; only a benchmark-only tile override in
// gemm_k4.cu passes another).
inline K4Grid k4_grid(int64_t n_hat, int64_t k_hat, int64_t b, const LaunchLimits& lim,
                      int64_t tile_n = kK4TileN) {
  return {n_hat, k_hat, b, lim.k_chunk, lim.max_batch, lim.max_n_tiles * tile_n};
}

// The launches of one group of gsize moduli, in order: K-chunk, batch part, n-hat part.
template <class F>
void k4_for_each_launch(const K4Grid& G, int gsize, F&& f) {
  const int64_t batches = int64_t{gsize} * G.b;
  for (int64_t k0 = 0; k0 < G.k_hat; k0 += G.chunk) {  // K-chunks (§3.5), 16-aligned
    const int64_t kc = std::min<int64_t>(G.chunk, G.k_hat - k0);
    for (int64_t z0 = 0; z0 < batches; z0 += G.max_batch) {  // batch-count split (§4)
      const int64_t zc = std::min(G.max_batch, batches - z0);
      for (int64_t j0 = 0; j0 < G.n_hat; j0 += G.n_per_launch)  // n-hat split (gridDim.y)
        f(K4Launch{k0, kc, z0, zc, j0, std::min(G.n_per_launch, G.n_hat - j0)});
    }
  }
}

// Closed-form counts of k4_for_each_launch over all residue groups of (N, g), without
// enumerating (k_hat may need ~2^45 chunks): the K-chunks per group, the launches per call
// (saturating at INT64_MAX), and the distinct shape keys with k0 = 0 (store-mod) and k0 > 0
// (add-mod), i.e. the CUTLASS parameter sets k4_prepare builds.
struct K4Counts {
  int64_t k_chunks = 0, launches = 0, store_shapes = 0, add_shapes = 0;
};
inline K4Counts k4_counts(const K4Grid& G, int N, int g) {
  K4Counts r;
  if (G.k_hat <= 0 || G.b <= 0 || G.n_hat <= 0 || N <= 0 || g <= 0) return r;
  const auto parts = [](int64_t total, int64_t per) { return total / per + (total % per != 0); };
  // Distinct sizes of a split of `total` into parts of `per`: the full part and the rest.
  const auto sizes = [](int64_t total, int64_t per, int64_t out[2]) {
    int c = 0;
    if (total >= per) out[c++] = per;
    if (total % per != 0) out[c++] = total % per;
    return c;
  };
  r.k_chunks = parts(G.k_hat, G.chunk);
  const int64_t n_parts = parts(G.n_hat, G.n_per_launch);
  // Group sizes: full groups of g, and a last group of N mod g.
  const int full = N / g, rest = N % g;
  const int64_t gsizes[2] = {g, rest};
  const int64_t gcount[2] = {full, rest != 0 ? 1 : 0};
  int64_t zsizes[4];
  int nz = 0;
  int64_t launches = 0;
  bool sat = false;
  for (int i = 0; i < 2; ++i) {
    if (gcount[i] == 0) continue;
    const int64_t batches = gsizes[i] * G.b;  // fits: the RA region g b m k_hat fits uint64
    int64_t per_group = 0, t = 0;
    if (!mul_i64(r.k_chunks, parts(batches, G.max_batch), &t) || !mul_i64(t, n_parts, &t) ||
        !mul_i64(t, gcount[i], &per_group) || !add_i64(launches, per_group, &launches))
      sat = true;
    int64_t zs[2];
    const int c = sizes(batches, G.max_batch, zs);
    for (int j = 0; j < c; ++j)
      if (std::find(zsizes, zsizes + nz, zs[j]) == zsizes + nz) zsizes[nz++] = zs[j];
  }
  r.launches = sat ? INT64_MAX : launches;
  int64_t ns[2];
  const int nn = sizes(G.n_hat, G.n_per_launch, ns);
  // Store-mod: the first chunk only, kc = min(chunk, k_hat). Add-mod: chunks 1.., whose
  // lengths are the full chunk (if there are >= 3 chunks) and the last one (if >= 2).
  r.store_shapes = int64_t{nz} * nn;
  int add_k = 0;
  if (r.k_chunks >= 3) ++add_k;
  if (r.k_chunks >= 2) {
    const int64_t last = G.k_hat - (r.k_chunks - 1) * G.chunk;
    if (!(r.k_chunks >= 3 && last == G.chunk)) ++add_k;
  }
  r.add_shapes = int64_t{add_k} * nz * nn;
  return r;
}

}  // namespace imd::detail
