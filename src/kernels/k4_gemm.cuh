// K4 GEMM kernel (DESIGN §3.5, §4; A2): batched int8 TN GEMM per K-chunk with the mod-p -> u8
// epilogue, as a persistent kernel with a register-only epilogue.
//
// Main loop: CUTLASS 2.x MmaMultistage (cutlass::gemm::threadblock::DefaultMma, the same
// int8 mma.sync.m16n8k32 .satfinite main loop as DefaultGemmUniversal), unchanged.
//
// What differs from CUTLASS's GEMM kernel + Sm80 EVT epilogue (gemm_k4.cu before A2):
//  - Persistent CTAs: grid = min(tiles, SMs x occupancy); CTA c takes tiles c, c + grid, ...
//    Before the epilogue of tile t it issues the cp.async prologue of its next tile, so the
//    first stages' global-memory latency overlaps the epilogue instead of following it.
//  - The epilogue never touches shared memory (CUTLASS stages the int32 accumulators through
//    smem with two barriers per row group and bank conflicts at threadblock N = 256). Each
//    thread reduces its own accumulators mod p, packs the residues to bytes, and a 4 x 4
//    transpose of 32-bit words inside each lane quad (two __shfl_xor_sync butterfly stages)
//    gives every thread 16 consecutive output bytes of a row: one 128-bit store per row.
//    That leaves shared memory to the next tile's prologue.
//  - Same arithmetic as before: chunk 0 stores v mod p, later chunks (W + v) mod p with
//    v = acc (+ W) in int32 and imd::device::mod_p_barrett / the p = 256 mask (§3.5,
//    cert_kc, cert_barrett). Results are bit-identical.
//
// Tile order: tile index -> (z, n group of 8 tiles, m, n within the group), so the tiles of
// one wave of CTAs form a compact block (the identity-swizzle-8 order of M4): their A and B
// panels stay in L2. z is the batch of the launch; the modulus is l = slot0 + (batch0 + z) / b.
#pragma once

#include <cstdint>
#include <new>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/layout/matrix.h"
#include "device/constants.cuh"

namespace imd::kernels {

// Op: the MMA operator. Production: OpMultiplyAddSaturate (.satfinite, DESIGN §2 K_c);
// gemm_k4.cu's A2 benchmark-only IMD_K4_NOSAT_MMA build passes OpMultiplyAdd.
template <class TB_, class Warp_, int kStages_, class Op = cutlass::arch::OpMultiplyAddSaturate>
struct K4Gemm {
  using TB = TB_;
  using Warp = Warp_;
  static constexpr int kStages = kStages_;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16, int32_t,
      cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
      Inst, kStages, Op>::ThreadblockMma;
  using WarpCount = typename Mma::WarpCount;
  static_assert(WarpCount::kK == 1, "no split-K warps");
  static constexpr int kThreads = 32 * WarpCount::kCount;
  static constexpr int kSmemBytes = int(sizeof(typename Mma::SharedStorage));
  // Warp tile in mma tiles: kMi row tiles of 16, kNj column tiles of 8. The epilogue's
  // quad transpose needs 64 columns per warp (8 column tiles: 4 words of 2 tiles each).
  static constexpr int kMi = Warp::kM / 16;
  static constexpr int kNj = Warp::kN / 8;
  static_assert(Warp::kN == 64, "epilogue quad transpose assumes 64-column warp tiles");
  static_assert(sizeof(typename Mma::FragmentC) == sizeof(int32_t) * kMi * kNj * 4,
                "unexpected accumulator fragment");

  // Kernel parameters of one launch (host/k4_launches.hpp K4Launch); pointers and the
  // epilogue's modulus offsets are set per call (set_call).
  struct Params {
    typename Mma::IteratorA::Params pa;
    typename Mma::IteratorB::Params pb;
    const int8_t* A = nullptr;  // batch 0, k0 of this launch
    const int8_t* B = nullptr;
    uint8_t* W = nullptr;
    int64_t batch_a = 0, batch_b = 0, plane_w = 0, ldw = 0;
    int m = 0, n = 0, k = 0;  // launch GEMM size (n = nc, k = kc)
    int tiles_m = 0, tiles_n = 0;
    int64_t batches = 0;
    int64_t tiles = 0;  // tiles_m * tiles_n * batches
    int grid = 0;       // persistent CTAs
    int k_iters = 0;
    int last_group = 0, last_width = 0;  // the last N-tile group and its width (<= kGroup)
    int slot0 = 0;           // modulus index of slot 0 (g0)
    int64_t b = 1;           // batches per slot
    int64_t batch0 = 0;      // z of this launch's batch 0
    bool small = false;      // 32-bit tile arithmetic (set_call)
    cutlass::FastDivmod div_batch, div_group, div_last, div_b;  // tiles/batch, tiles_m*8, last_width, b
  };

  // lda = ldb = k_hat; ldw = n_hat; resident = SMs x occupancy.
  static constexpr int kGroup = 8;  // N tiles per group (M4's swizzle 8)

  static Params make_params(int m, int n, int k, int64_t lda, int64_t ldb, int64_t ldw,
                            int64_t batches, int64_t resident) {
    Params p;
    p.pa = typename Mma::IteratorA::Params(cutlass::layout::RowMajor(lda));
    p.pb = typename Mma::IteratorB::Params(cutlass::layout::ColumnMajor(ldb));
    p.ldw = ldw;
    p.m = m, p.n = n, p.k = k;
    p.tiles_m = (m + TB::kM - 1) / TB::kM;
    p.tiles_n = (n + TB::kN - 1) / TB::kN;
    p.batches = batches;
    p.tiles = int64_t{p.tiles_m} * p.tiles_n * batches;
    p.grid = static_cast<int>(p.tiles < resident ? p.tiles : resident);
    p.k_iters = (k + TB::kK - 1) / TB::kK;
    p.last_group = (p.tiles_n - 1) / kGroup;
    p.last_width = p.tiles_n - p.last_group * kGroup;
    // 32-bit tile arithmetic (FastDivmod: exact for dividends in [0, 2^31)) when every tile
    // index fits; otherwise the kernel divides in 64 bits.
    p.div_last = cutlass::FastDivmod(p.last_width);
    if (p.tiles <= INT32_MAX) {
      p.div_batch = cutlass::FastDivmod(p.tiles_m * p.tiles_n);
      p.div_group = cutlass::FastDivmod(p.tiles_m * kGroup);
    }
    return p;
  }
  static void set_call(Params& p, const int8_t* A, const int8_t* B, uint8_t* W, int64_t batch_a,
                       int64_t batch_b, int64_t plane_w, int slot0, int64_t b, int64_t batch0) {
    p.A = A, p.B = B, p.W = W;
    p.batch_a = batch_a, p.batch_b = batch_b, p.plane_w = plane_w;
    p.slot0 = slot0, p.b = b, p.batch0 = batch0;
    p.small = p.tiles <= INT32_MAX && batch0 + p.batches <= INT32_MAX;
    if (p.small) p.div_b = cutlass::FastDivmod(static_cast<int>(b < INT32_MAX ? b : INT32_MAX));
  }

  struct Tile {
    int64_t z;
    int m0, n0;  // first row / column
    int l;       // modulus index slot0 + (batch0 + z) / b
  };
  CUTLASS_DEVICE static Tile tile_of(Params const& p, int64_t t) {
    int64_t z, r, group, in;
    int l;
    if (p.small) {  // t < 2^31, batch0 + z < 2^31
      int zi, ri, gi, ii, q, rem;
      p.div_batch(zi, ri, static_cast<int>(t));
      p.div_group(gi, ii, ri);
      p.div_b(q, rem, static_cast<int>(p.batch0) + zi);
      z = zi, r = ri, group = gi, in = ii, l = p.slot0 + q;
    } else {
      const int64_t per_batch = int64_t{p.tiles_m} * p.tiles_n;
      z = t / per_batch;
      r = t - z * per_batch;
      group = r / (int64_t{p.tiles_m} * kGroup);
      in = r - group * (int64_t{p.tiles_m} * kGroup);
      l = p.slot0 + static_cast<int>((p.batch0 + z) / p.b);
    }
    int tm, tn;
    if (group != p.last_group) {
      tm = static_cast<int>(in >> 3), tn = static_cast<int>(group) * kGroup + static_cast<int>(in & 7);
    } else {
      int q, rem;
      p.div_last(q, rem, static_cast<int>(in));  // in < tiles_m * 8 <= 2^31 (m < 2^31)
      tm = q, tn = static_cast<int>(group) * kGroup + rem;
    }
    return {z, tm * TB::kM, tn * TB::kN, l};
  }

  // ----------------------------------------------------------------------------- epilogue ---

  // In-quad 4 x 4 transpose of 32-bit words: lane t of a quad holds w[0..3]; afterwards it
  // holds w[j] = (word t of lane j). Two butterfly stages: at distance s, entries whose
  // index bit s differs from the lane's bit s are exchanged with the partner lane t ^ s
  // (an element at (lane a, index b) moves to (a ^ d, b ^ d) with d = (a ^ b) & s).
  CUTLASS_DEVICE static void quad_transpose(uint32_t (&w)[4], int t) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 1; s <= 2; s <<= 1) {
      const bool hi = (t & s) != 0;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < 4; ++k) {
        if (k & s) continue;  // pair (k, k ^ s)
        const uint32_t send = hi ? w[k] : w[k ^ s];
        const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, s);
        if (hi)
          w[k] = recv;
        else
          w[k ^ s] = recv;
      }
    }
  }

  // Store-mod (kAdd = false) or add-mod (kAdd = true) of one tile from the accumulators.
  // The p = 256 test is made once per tile (as a per-element select it compiled to two
  // branches per element).
  template <bool kAdd>
  CUTLASS_DEVICE static void epilogue(Params const& p, Tile const& tile,
                                      typename Mma::FragmentC const& acc, int warp_idx, int lane) {
    const uint32_t mod = imd::device::c_tables.moduli[tile.l];
    if (mod == 256u)
      epilogue_rows<kAdd, true>(p, tile, acc, warp_idx, lane, 0u, 0u, 0u);
    else
      epilogue_rows<kAdd, false>(p, tile, acc, warp_idx, lane, 0u - mod,
                                 imd::device::c_tables.epi_bias[tile.l],
                                 imd::device::c_tables.epi_magic[tile.l]);
  }

  // neg_p = 2^32 - p, bias, magic: the §3.5 constants of p (unused for p = 256).
  template <bool kAdd, bool k256>
  CUTLASS_DEVICE static void epilogue_rows(Params const& p, Tile const& tile,
                                           typename Mma::FragmentC const& acc, int warp_idx,
                                           int lane, uint32_t neg_p, uint32_t bias, uint64_t magic) {
    const int warp_m = warp_idx % WarpCount::kM, warp_n = warp_idx / WarpCount::kM;
    const int g = lane >> 2, t = lane & 3;
    const int col = tile.n0 + warp_n * Warp::kN + 16 * t;  // this thread's 16 output bytes
    const bool col_ok = col < p.n;                         // n is a multiple of 16
    uint8_t* const plane = p.W + tile.z * p.plane_w;

    // Add-mod: the old residues of this thread's 16 bytes per row, loaded up front.
    uint4 old[kMi * 2];
    if constexpr (kAdd) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kMi * 2; ++r) {
        const int row = tile.m0 + warp_m * Warp::kM + (r >> 1) * 16 + (r & 1) * 8 + g;
        old[r] = make_uint4(0, 0, 0, 0);
        if (col_ok && row < p.m)
          old[r] = *reinterpret_cast<const uint4*>(plane + row * p.ldw + col);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kMi * 2; ++r) {  // the thread's rows: mma row tile r / 2, half r % 2
      const int mi = r >> 1, h = r & 1;
      const int row = tile.m0 + warp_m * Warp::kM + mi * 16 + h * 8 + g;
      // Accumulator (mma tile (mi, nj), element 2h + e) is at row 16 mi + 8 h + g, column
      // 8 nj + 2 t + e of the warp tile (PTX m16n8k32 s32 layout; CUTLASS MmaTensorOp keeps
      // mma tile (mi, nj) at fragment offset 4 (mi + nj kMi)).
      int32_t v[kNj][2];
      CUTLASS_PRAGMA_UNROLL
      for (int nj = 0; nj < kNj; ++nj) {
        v[nj][0] = acc[4 * (mi + nj * kMi) + 2 * h];
        v[nj][1] = acc[4 * (mi + nj * kMi) + 2 * h + 1];
      }
      if constexpr (kAdd) {
        // Inverse of the store's byte shuffle: word k of lane t = old bytes of column tiles
        // 2k, 2k + 1 at columns 2t, 2t + 1.
        uint32_t w[4] = {__byte_perm(old[r].x, old[r].z, 0x5410), __byte_perm(old[r].x, old[r].z, 0x7632),
                         __byte_perm(old[r].y, old[r].w, 0x5410), __byte_perm(old[r].y, old[r].w, 0x7632)};
        quad_transpose(w, t);
        CUTLASS_PRAGMA_UNROLL
        for (int nj = 0; nj < kNj; ++nj) {
          const uint32_t word = w[nj >> 1] >> (16 * (nj & 1));
          v[nj][0] += static_cast<int32_t>(word & 255u);  // W + acc stays in int32 (cert_kc)
          v[nj][1] += static_cast<int32_t>((word >> 8) & 255u);
        }
      }
      // Residues, packed: word k = column tiles 2k, 2k + 1, bytes (2t, 2t + 1) of each.
      uint32_t w[4];
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < 4; ++k) {
        uint32_t q[4];
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 4; ++e) {
          const int32_t x = v[2 * k + (e >> 1)][e & 1];
          if constexpr (k256)
            q[e] = static_cast<uint32_t>(x) & 255u;  // §3.5: p = 256 by mask
          else
            q[e] = imd::device::mod_p_barrett_neg(x, neg_p, bias, magic);
        }
        w[k] = __byte_perm(__byte_perm(q[0], q[1], 0x0040), __byte_perm(q[2], q[3], 0x0040), 0x5410);
      }
      // After the transpose, w[j] holds lane j's two bytes of column tiles 2t and 2t + 1:
      // output columns 16 t + 2 j (+1) and 16 t + 8 + 2 j (+1).
      quad_transpose(w, t);
      if (col_ok && row < p.m) {
        const uint4 out = make_uint4(__byte_perm(w[0], w[1], 0x5410), __byte_perm(w[2], w[3], 0x5410),
                                     __byte_perm(w[0], w[1], 0x7632), __byte_perm(w[2], w[3], 0x7632));
        *reinterpret_cast<uint4*>(plane + row * p.ldw + col) = out;
      }
    }
  }

  // ------------------------------------------------------------------------------ kernel ---

  template <bool kAdd>
  CUTLASS_DEVICE static void run(Params const& p, typename Mma::SharedStorage& smem) {
    int64_t t = blockIdx.x;
    if (t >= p.tiles) return;
    const int tid = threadIdx.x;
    const int warp_idx = __shfl_sync(0xffffffffu, tid / 32, 0);
    const int lane = tid % 32;

    Tile tile = tile_of(p, t);
    auto iter_a = [&](Tile const& c) {
      return typename Mma::IteratorA(p.pa, const_cast<int8_t*>(p.A) + c.z * p.batch_a,
                                     {p.m, p.k}, tid, {c.m0, 0});
    };
    auto iter_b = [&](Tile const& c) {
      return typename Mma::IteratorB(p.pb, const_cast<int8_t*>(p.B) + c.z * p.batch_b,
                                     {p.k, p.n}, tid, {0, c.n0});
    };
    typename Mma::IteratorA ia = iter_a(tile);
    typename Mma::IteratorB ib = iter_b(tile);
    int iters = p.k_iters;
    Mma mma(smem, tid, warp_idx, lane);
    mma.prologue(ia, ib, iters);

    typename Mma::FragmentC acc;
    for (;;) {
      mma.gmem_wait();
      acc.clear();
      mma.gemm_iters(iters, acc, ia, ib);  // ends with every cp.async drained + a barrier

      const Tile cur = tile;
      t += gridDim.x;
      const bool more = t < p.tiles;
      if (more) {  // next tile's prologue first; the epilogue below needs no shared memory
        tile = tile_of(p, t);
        // (CUTLASS's iterators are not assignable; all three are trivially destructible.)
        new (&ia) typename Mma::IteratorA(iter_a(tile));
        new (&ib) typename Mma::IteratorB(iter_b(tile));
        new (&mma) Mma(smem, tid, warp_idx, lane);
        iters = p.k_iters;
        mma.prologue(ia, ib, iters);
      }
      epilogue<kAdd>(p, cur, acc, warp_idx, lane);
      if (!more) break;
    }
  }
};

template <class G, bool kAdd>
__global__ void __launch_bounds__(G::kThreads, 1) k4_kernel(typename G::Params p) {
  extern __shared__ __align__(16) unsigned char k4_smem[];
  G::template run<kAdd>(p, *reinterpret_cast<typename G::Mma::SharedStorage*>(k4_smem));
}

}  // namespace imd::kernels
