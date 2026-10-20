// K0-K5 and the DIRECT kernel (DESIGN §3, §4). One stream, in order:
//   K0 vec    kernel: the scatter vectors -> ws.vec from the plan's fused-mode descriptor
//             (a kernel argument); skipped with IMD_VEC_RESIDENT (the plan's device copy)
//   K1 scale  A rows, B cols: non-finite flags and shifts sigma / tau (int16)
//   per residue group G:  K2 / K3 pack A, B -> int8 residues;  K4 GEMMs (gemm_k4.cu)
//   K5 crt    per (lambda, i, j): flagged ? native fma dot : exact CRT, one rounding; alpha/beta
// Fused variants (chosen at plan time, analysis.hpp): KF (v1.4, k_hat <= 1024) = K0 + K1 +
// group 1's packs in one launch; KL (v1.6, per operand, k_hat > 1024) = K1 + group 1's pack
// of that operand in one launch (one DRAM read of the operand instead of two).
// alpha = 0 or k = 0: only K0 and K5 in beta-only mode. DIRECT path (mode DIRECT or K-empty):
// K0 + the native kernel. Every operand access goes through the int64 additive scatter
// vectors off = v_bat[lambda] + v_row[i] + v_k[h], so arbitrary strides work.
//
// K2/K3 gather through a shared-memory tile, coalesced along the operand's smallest-|stride|
// mode, and write K-contiguous 16-byte vectors; residues by byte weights (dp4a). K1 reads
// over the same row tiles when the pack reads along rows, else one block per row. K5 takes
// 4 consecutive j-bar per thread; when C's fastest label is a batch label it stages a
// lambda x j-bar tile in shared memory and writes C lambda-fastest. Every launch choice
// here (tile shapes, grids, thread mappings) leaves the results bitwise unchanged.
// M4 measurements and the remaining gaps: docs/NOTES-M4.md.
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include "device/constants.cuh"
#include "host/analysis.hpp"  // kFusedKHatMax
#include "kernels/numerics.cuh"
#include "kernels/pack_emitter.cuh"
#include "plan.hpp"
#include "profile.hpp"

namespace imd::detail {
namespace {

using imd::device::c_tables;
using namespace imd::kernels;

constexpr int kThreads = kLaunchThreads;            // host/analysis.hpp (shared launch rules)

// v1.10: host/analysis.hpp launch_blocks is the one rule (the performance model uses it too).
unsigned blocks_for(int64_t work, int per_block = kThreads) {
  return static_cast<unsigned>(launch_blocks(work, per_block));
}

// q = a / b, r = a % b for a >= 0, b > 0: 32-bit division when both fit (the common case;
// a 64-bit division is a ~70-instruction subroutine, a 32-bit one ~20 instructions).
__device__ __forceinline__ void divmod(int64_t a, int64_t b, int64_t* q, int64_t* r) {
  if (((static_cast<uint64_t>(a) | static_cast<uint64_t>(b)) >> 32) == 0) {
    const uint32_t q32 = static_cast<uint32_t>(a) / static_cast<uint32_t>(b);
    *q = q32;
    *r = a - int64_t{q32} * b;
  } else {
    *q = a / b;
    *r = a - *q * b;
  }
}

// A strided operand seen as `rows` vectors of length k per batch:
// element (lambda, r, h) at X[v_bat[lambda] + v_row[r] + v_k[h]].
struct Operand {
  const double* X;
  const int64_t* v_bat;
  const int64_t* v_row;
  const int64_t* v_k;
  int64_t rows;
};

// ---- K1: §3.1 non-finite flags and §3.2 exact-integer shifts --------------------------

struct ScaleArgs {
  Operand op;
  int64_t b, k;
  int16_t* sigma;  // [lambda][rows]
  uint8_t* flag;   // [lambda][rows]
  uint64_t rhs[4];
  int rhs_bits, lhs_shift;
  int reverse;  // k1_scale: pass 2 walks each row backwards (launch choice, launch_k1)
};

__device__ __forceinline__ u128 shfl_down_u128(u128 v, int d) {
  const auto lo = __shfl_down_sync(0xffffffffu, static_cast<unsigned long long>(v), d);
  const auto hi = __shfl_down_sync(0xffffffffu, static_cast<unsigned long long>(v >> 64), d);
  return (static_cast<u128>(hi) << 64) | lo;
}

// One block of T threads per row (T = 1024 for long or few rows: more loads in flight per
// row while few rows are resident, so the second pass re-reads from L2).
template <int T>
__global__ void __launch_bounds__(T) k1_scale(ScaleArgs a) {
  constexpr int kW = T / 32;
  __shared__ int s_max[kW];
  __shared__ int s_flag[kW];
  __shared__ u128 s_sum[kW];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int64_t total = a.b * a.op.rows;
  for (int64_t r = blockIdx.x; r < total; r += gridDim.x) {
    const int64_t lam = r / a.op.rows, i = r % a.op.rows;
    const double* X = a.op.X + (a.op.v_bat[lam] + a.op.v_row[i]);
    // pass 1: flag and gamma = max ilogb over finite nonzero entries
    int gmax = INT_MIN, flag = 0;
#pragma unroll 4
    for (int64_t h = threadIdx.x; h < a.k; h += blockDim.x) {
      const uint64_t bits = to_bits(X[a.op.v_k[h]]);
      if (non_finite(bits)) flag = 1;
      else if (finite_nonzero(bits)) gmax = max(gmax, ilogb_bits(bits));
    }
    gmax = __reduce_max_sync(0xffffffffu, gmax);
    flag = __reduce_or_sync(0xffffffffu, static_cast<unsigned>(flag));
    if (lane == 0) {
      s_max[warp] = gmax;
      s_flag[warp] = flag;
    }
    __syncthreads();
    gmax = INT_MIN;
    flag = 0;
    for (int w = 0; w < kW; ++w) {
      gmax = max(gmax, s_max[w]);
      flag |= s_flag[w];
    }
    // pass 2: S = sum ceil(m_h^2 2^(2(e_h - gamma) + F)), exact (order-independent).
    // A3: with a.reverse (rows in flight larger than the L2 budget, launch_k1), backwards
    // along the row, so the part pass 1 read last (the part most likely still in L2) is
    // re-read first; S is an exact u128 sum, so the order does not change it.
    u128 S = 0;
    if (gmax != INT_MIN && a.reverse) {
      if (static_cast<int64_t>(threadIdx.x) < a.k) {
        const int64_t steps = (a.k - 1 - threadIdx.x) / blockDim.x;  // this thread's last h
#pragma unroll 4
        for (int64_t h = threadIdx.x + steps * blockDim.x; h >= 0; h -= blockDim.x) {
          const uint64_t bits = to_bits(X[a.op.v_k[h]]);
          if (finite_nonzero(bits)) S += square_term_n(bits, gmax);
        }
      }
    } else if (gmax != INT_MIN) {
#pragma unroll 4
      for (int64_t h = threadIdx.x; h < a.k; h += blockDim.x) {
        const uint64_t bits = to_bits(X[a.op.v_k[h]]);
        if (finite_nonzero(bits)) S += square_term_n(bits, gmax);
      }
    }
    if (gmax != INT_MIN) {
      for (int d = 16; d > 0; d /= 2) S += shfl_down_u128(S, d);
      if (lane == 0) s_sum[warp] = S;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int sigma = 0;  // zero row (all entries zero or non-finite): sigma := 0
      if (gmax != INT_MIN) {
        u128 tot = 0;
        for (int w = 0; w < kW; ++w) tot += s_sum[w];
        sigma = largest_u(tot, a.lhs_shift, a.rhs, a.rhs_bits) - gmax;
      }
      a.sigma[r] = static_cast<int16_t>(sigma);  // range certified (cert_shift_range)
      a.flag[r] = static_cast<uint8_t>(flag);
    }
    __syncthreads();  // shared arrays are reused by the next row
  }
}

// ---- K2 / K3: gather, §3.3 quantization and §3.4 residues -------------------------------
//
// A tiled gather-transpose (TBLIS style). The operand's rows are enumerated as
// rho = r + rows * lambda (index_map.hpp pack_pivot); a tile is th rows
// rho = lo + radix * d, d in [d0, d0 + th), by tk = 4096 / th consecutive k-bar.
//   1. row info: the gather base v_bat[lambda] + v_row[r], sigma, the output row; v_k[h];
//   2. read: raw binary64 bits into shared memory. A warp reads along d (along_rows: X's
//      smallest-|stride| mode is the row / batch mode at `radix`) or along k-bar, so global
//      reads are coalesced along that mode whichever axis of X is fastest;
//   3. write: each thread converts 16 consecutive k-bar of one row and stores them as one
//      16-byte vector per modulus of the group, K-contiguous; h in [k, k_hat) is zero.
// Tile shape: th = 64 when reading along k-bar; along rows th = the power of two >= D (the
// number of rows along the pivot), clamped to [8, 64], so a short pivot mode (e.g. a
// stride-1 batch label of extent 16) does not leave most tile rows empty (NOTES-M2M3 #6):
// the tile is wider in k-bar instead. Shared memory holds row dl, k-bar hl at
// dl * pitch + hl + hl / 16 (one pad word per 16: the 16-wide conversion reads of a warp
// hit distinct banks; pitch is odd, so the along-d reads do too).
// The tile is generic in what it emits: `PackEmitterT` converts 16 values for one modulus
// (an Ozaki-I slice mode would emit slices here and reuse the gather unchanged).
// B rows n..n_hat-1 (zero) are written by k23_zero_rows.

constexpr int kTileElems = 4096;  // th * tk; kTileElems / 16 == kThreads
static_assert(kTileElems == 1 << kPackTileLgElems, "host/analysis.hpp pack tile");
constexpr int kTileRowsMax = 64, kTileRowsMin = 8;
constexpr int kTileKMax = kTileElems / kTileRowsMin;  // 512
constexpr int kVec = 16;          // int8 per 16-byte store
constexpr int kTileSmem = kTileElems + kTileElems / kVec + kTileRowsMax;  // padded words
static_assert(kTileElems / kVec == kThreads, "one 16-byte chunk per thread");

// log2 of the pack tile height (see above): host/analysis.hpp pack_tile_lg_rows (v1.10: shared
// with the performance model, as pack_launch).

// K1 over the pack's row tiles (NOTES-M2M3 #5), used when the pack reads along rows (the
// operand's smallest-|stride| mode is a row / batch mode, so reading one row along k-bar
// would be strided). A block takes th rows rho = lo + radix * d, d in [d0, d0 + th), over
// all of k-bar (the pack's row enumeration; th = 8 here, see the launch); thread tid always
// works on row dl = tid mod th, so a warp reads th rows side by side (coalesced along the
// pivot) and 32 / th consecutive k-bar. Same two passes
// and the same exact integers as k1_scale: gamma (and the flag) per row, then S per row;
// the per-thread partial results are combined in shared memory (max, or, u128 sum: exact
// and order-independent).
template <int T>
__global__ void __launch_bounds__(T) k1_tiled(ScaleArgs a, int64_t radix, int lg_th) {
  __shared__ int s_max[T];
  __shared__ int s_flag[T];
  __shared__ u128 s_sum[T];
  __shared__ int64_t s_base[kTileRowsMax];
  __shared__ bool s_valid[kTileRowsMax];
  __shared__ int s_gamma[kTileRowsMax];
  __shared__ int s_rflag[kTileRowsMax];
  const int tid = threadIdx.x, th = 1 << lg_th, dl = tid & (th - 1);
  const int h_first = tid >> lg_th, h_step = T >> lg_th, per_row = T >> lg_th;
  const int64_t rows = a.op.rows, D = a.b * rows / radix, n_d = (D + th - 1) >> lg_th;
  const int64_t tiles = radix * n_d;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int64_t dt = t % n_d, lo = t / n_d;
    if (tid < th) {
      const int64_t d = (dt << lg_th) + tid;
      s_valid[tid] = d < D;
      if (d < D) {
        const int64_t rho = lo + radix * d, lam = rho / rows, r = rho % rows;
        s_base[tid] = a.op.v_bat[lam] + a.op.v_row[r];
      }
    }
    __syncthreads();
    const bool valid = s_valid[dl];
    const double* X = a.op.X + (valid ? s_base[dl] : 0);
    // pass 1: flag and gamma
    int gmax = INT_MIN, flag = 0;
    if (valid) {
#pragma unroll 4
      for (int64_t h = h_first; h < a.k; h += h_step) {
        const uint64_t bits = to_bits(X[a.op.v_k[h]]);
        if (non_finite(bits)) flag = 1;
        else if (finite_nonzero(bits)) gmax = max(gmax, ilogb_bits(bits));
      }
    }
    s_max[tid] = gmax;
    s_flag[tid] = flag;
    __syncthreads();
    if (tid < th) {
      int g = INT_MIN, f = 0;
      for (int u = 0; u < per_row; ++u) {
        g = max(g, s_max[tid + (u << lg_th)]);
        f |= s_flag[tid + (u << lg_th)];
      }
      s_gamma[tid] = g;
      s_rflag[tid] = f;
    }
    __syncthreads();
    // pass 2: S
    gmax = s_gamma[dl];
    u128 S = 0;
    if (valid && gmax != INT_MIN) {
#pragma unroll 4
      for (int64_t h = h_first; h < a.k; h += h_step) {
        const uint64_t bits = to_bits(X[a.op.v_k[h]]);
        if (finite_nonzero(bits)) S += square_term_n(bits, gmax);
      }
    }
    s_sum[tid] = S;
    __syncthreads();
    if (tid < th && s_valid[tid]) {
      const int64_t rho = lo + radix * ((dt << lg_th) + tid);
      const int g = s_gamma[tid];
      int sigma = 0;  // zero row: sigma := 0
      if (g != INT_MIN) {
        u128 tot = 0;
        for (int u = 0; u < per_row; ++u) tot += s_sum[tid + (u << lg_th)];
        sigma = largest_u(tot, a.lhs_shift, a.rhs, a.rhs_bits) - g;
      }
      a.sigma[rho] = static_cast<int16_t>(sigma);
      a.flag[rho] = static_cast<uint8_t>(s_rflag[tid]);
    }
    __syncthreads();  // shared arrays are reused by the next tile
  }
}

// Blocks for a two-pass kernel whose blocks each re-read `bytes_per_block`: enough to fill
// the GPU, but few enough that the rows in flight (about 40 MB) stay in the 72 MB L2 for
// the second pass. `sms` is the plan's SM count (read at plan creation).
// When the grid is capped (fewer blocks than work items) and the blocks will be 1024 threads
// wide (fewer than 2 per SM: one resident per SM), the cap is rounded down to a multiple of
// the SM count: a grid-strided block runs until all its rows are done, so the blocks beyond
// the first wave would start only when a resident block finishes all its rows, nearly
// doubling the time (A4: r_bigk, 2048 rows of 256 KB, had 160 blocks on 128 SMs).
// L2 budget for K1's rows in flight (about the 72 MB L2 less what the rest of the GPU keeps
// there), so that pass 2 re-reads from L2 (launch_k1, A3): the two-blocks-per-SM long-row
// variant only while 2 x SMs rows fit (r_bigk's 256 x 256 KB do, r_mm's 256 x 400 KB do not);
// beyond it, one 1024-thread block per SM walks pass 2 backwards.
// Issue 21: both budgets are fractions of the card's L2 (host/analysis.hpp k1_l2_budget_bytes, k1_l2_fit_bytes: the old constants on the 4090).
unsigned k1_blocks(int64_t work_items, double bytes_per_block, int sms, int64_t l2_bytes) {
  const double fit = imd::detail::k1_l2_fit_bytes(l2_bytes) / std::max(1.0, bytes_per_block);
  int64_t cap = std::max<int64_t>(sms, static_cast<int64_t>(fit));
  if (work_items > cap && cap < 2 * int64_t{sms}) cap = cap / sms * sms;
  return blocks_for(std::min(work_items, cap), 1);
}

struct PackArgs {
  Operand op;
  int64_t rows_pad, b, k, k_hat;  // rows_pad = m (A) or n_hat (B)
  int64_t k_store;                // k-bar < k_store are stored: k_hat (pad_mutation: test only)
  const int16_t* sigma;           // [lambda][rows] = [rho]
  int g0, gsize;
  bool trunc;
  int8_t* R;          // [s][lambda][rows_pad][k_hat]
  int64_t radix;      // rho = lo + radix * d
  bool along_rows;    // read phase: warp lanes along d (else along k-bar)
  int lg_th;          // tile height th = 2^lg_th rows, tk = kTileElems / th k-bar
};

// §3.3 + §3.4 for one row's E consecutive k-bar: quantize once, then residues per modulus:
// PackEmitterT<E, W> (kernels/pack_emitter.cuh, v1.14 §3.4.1: |a'| as W = pack_words(N) u32
// words, branch-free). Every pack kernel is instantiated for W = 2 and 3; the launch picks W
// from the plan's N.

template <int W>  // v1.14 §3.4.1: pack_words(N)
__global__ void __launch_bounds__(kThreads) k23_pack(PackArgs a) {
  __shared__ uint64_t tile[kTileSmem];
  __shared__ int64_t s_base[kTileRowsMax];  // v_bat[lambda] + v_row[r]
  __shared__ int64_t s_out[kTileRowsMax];   // (lambda rows_pad + r) k_hat
  __shared__ int s_sigma[kTileRowsMax];
  __shared__ bool s_valid[kTileRowsMax];
  __shared__ int64_t s_vk[kTileKMax];
  const int tid = threadIdx.x;
  const int lg_th = a.lg_th, th = 1 << lg_th, lg_tk = 12 - lg_th, tk = 1 << lg_tk;
  const int pitch = tk + tk / kVec + 1;
  const int64_t rows = a.op.rows, D = a.b * rows / a.radix;
  const int64_t n_d = (D + th - 1) >> lg_th, n_h = (a.k_hat + tk - 1) >> lg_tk;
  const int64_t tiles = a.radix * n_d * n_h, plane = a.b * a.rows_pad * a.k_hat;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int64_t ht = t % n_h, rest = t / n_h, dt = rest % n_d, lo = rest / n_d;
    const int64_t h0 = ht << lg_tk, d0 = dt << lg_th;
    // 1. row and column info
    if (tid < th) {
      const int64_t d = d0 + tid;
      const bool valid = d < D;
      s_valid[tid] = valid;
      if (valid) {
        const int64_t rho = lo + a.radix * d, lam = rho / rows, r = rho % rows;
        s_base[tid] = a.op.v_bat[lam] + a.op.v_row[r];
        s_out[tid] = (lam * a.rows_pad + r) * a.k_hat;
        s_sigma[tid] = a.sigma[rho];
      }
    }
    for (int hl = tid; hl < tk; hl += kThreads) {
      const int64_t h = h0 + hl;
      s_vk[hl] = h < a.k ? a.op.v_k[h] : 0;
    }
    __syncthreads();
    // 2. coalesced read into shared memory (zero outside the operand)
    const int64_t h_end = a.k - h0;  // valid k-bar in this tile: hl < h_end
#pragma unroll 4
    for (int it = 0; it < kTileElems / kThreads; ++it) {
      const int q = tid + it * kThreads;
      const int dl = a.along_rows ? (q & (th - 1)) : (q >> lg_tk);
      const int hl = a.along_rows ? (q >> lg_th) : (q & (tk - 1));
      tile[dl * pitch + hl + hl / kVec] =
          s_valid[dl] && hl < h_end ? to_bits(a.op.X[s_base[dl] + s_vk[hl]]) : 0;
    }
    __syncthreads();
    // 3. convert and store 16 K-contiguous int8 per modulus
    const int dl = tid >> (lg_tk - 4), hl = (tid & ((tk >> 4) - 1)) * kVec;
    if (s_valid[dl] && h0 + hl < a.k_store) {  // k_store = k_hat, a multiple of 16
      uint64_t bits[kVec];
      const uint64_t* src = tile + dl * pitch + hl + hl / kVec;
#pragma unroll
      for (int e = 0; e < kVec; ++e) bits[e] = src[e];
      PackEmitterT<kVec, W> em;
      em.load(bits, s_sigma[dl], a.trunc);
      int8_t* dst = a.R + s_out[dl] + h0 + hl;
      // Block row y of the grid emits the slots [y gsize / ny, (y + 1) gsize / ny): a small
      // operand is spread over more blocks (each re-reads its tile; same values).
      const int s_lo = int(blockIdx.y) * a.gsize / int(gridDim.y);
      const int s_hi = (int(blockIdx.y) + 1) * a.gsize / int(gridDim.y);
      for (int s = s_lo; s < s_hi; ++s)
        em.store(a.g0 + s, dst + s * plane);
    }
    __syncthreads();  // shared arrays are reused by the next tile
  }
}

// gridDim.y of a pack launch: with fewer tiles than 2 per SM, split the group's moduli over
// up to gsize block rows so that about 2 blocks per SM run (a launch choice only).
// (host/analysis.hpp pack_moduli_parts: the one rule, also used by the performance model.)
inline unsigned pack_split(int64_t tiles, int gsize, int sms) {
  return static_cast<unsigned>(pack_moduli_parts(tiles, gsize, sms));
}

// Zero rows [rows, rows_pad) of every (slot, lambda): the n_hat padding of RB.
__global__ void __launch_bounds__(kThreads) k23_zero_rows(int8_t* R, int64_t rows,
                                                          int64_t rows_pad, int64_t b,
                                                          int64_t k_hat, int gsize) {
  const int64_t pad_vecs = (rows_pad - rows) * k_hat / kVec;  // per (slot, lambda)
  const int64_t total = int64_t{gsize} * b * pad_vecs;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t v = idx % pad_vecs, sl = idx / pad_vecs;  // sl = s * b + lambda
    *reinterpret_cast<uint4*>(R + (sl * rows_pad + rows) * k_hat + v * kVec) =
        make_uint4(0, 0, 0, 0);
  }
}

// ---- K5 and DIRECT: §3.6 CRT, §3.7 output -----------------------------------------------

struct OutArgs {
  const double* A;
  const double* B;
  double* C;
  const int64_t *a_row, *a_k, *a_bat, *b_k, *b_col, *b_bat, *c_row, *c_col, *c_bat;
  int64_t m, n, k, b, n_hat;
  const int16_t* sA;
  const int16_t* sB;
  const uint8_t* fA;
  const uint8_t* fB;
  const uint8_t* W;  // [l][lambda][m][n_hat]
  int N;
  double alpha, beta;
};

// §3.7 native path: acc = +0; acc = fma(a_h, b_h, acc) in canonical k-bar order.
__device__ __forceinline__ double native_dot(const OutArgs& a, int64_t lam, int64_t i, int64_t j) {
  const double* A = a.A + (a.a_bat[lam] + a.a_row[i]);
  const double* B = a.B + (a.b_bat[lam] + a.b_col[j]);
  double acc = 0.0;
  for (int64_t h = 0; h < a.k; ++h) acc = __fma_rn(A[a.a_k[h]], B[a.b_k[h]], acc);
  return acc;
}

// §3.7 scaling: E = alpha == 1 ? D : RN(alpha D); C = beta == 0 ? E : fma(beta, C, E).
__device__ __forceinline__ void store_out(const OutArgs& a, int64_t off, double D) {
  const double E = a.alpha == 1.0 ? D : __dmul_rn(a.alpha, D);
  a.C[off] = a.beta == 0.0 ? E : __fma_rn(a.beta, a.C[off], E);  // C not read when beta = 0
}

// Largest N whose CRT word has L limbs (DESIGN §2 L_w), checked against the table.
constexpr int k5_max_n(int L) { return L == 1 ? 3 : L == 2 ? 7 : L == 3 ? 12 : L == 4 ? 16 : 20; }
constexpr bool k5_max_n_ok() {
  for (int N = imd::constants::kNMin; N <= imd::constants::kNMax; ++N)
    if (N > k5_max_n(imd::constants::kTables.L32[N])) return false;
  return true;
}
static_assert(k5_max_n_ok(), "k5_max_n must cover every N of its limb count");

// ---- K5, one kernel per N (A5, docs/NOTES-A5.md §2) ---------------------------------------
// The §3.6 running CRT (numerics.cuh crt_step) with N a template parameter: the plane loop,
// the plane addresses and every constant of a step (p_l, q_l, the 32-bit magic, M_l, P, H)
// are compile-time, so a step is its arithmetic only (no runtime-indexed constant loads, no
// per-step branches on l < N or p = 256, no predicated addresses for unused planes). Until
// A5 one kernel per limb count L took N as an argument (780 instructions per output at N = 8
// against ~120 for the steps themselves). Same integers in the same order as crt_step /
// crt_finish / round_scaled: bit-identical.
template <int N, int L, int l>
__device__ __forceinline__ void crt_step_n(uint32_t (&x)[L], uint32_t Wl) {
  constexpr uint32_t p = imd::constants::kTables.moduli[l];
  constexpr uint32_t q = imd::constants::kTables.q[N][l];
  constexpr uint32_t m32 = imd::device::kMagic32.m[l];
  const uint32_t y = imd::device::mod_p_small32(q * Wl, p, m32);
  uint32_t M[L], P[L];  // constant-bank operands: every index is compile-time after unrolling
#pragma unroll
  for (int w = 0; w < L; ++w) {
    M[w] = c_tables.M[N][l][w];
    P[w] = c_tables.P[N][w];
  }
  crt_mac_sub<L>(x, M, y, P);
}
// Issue 21 (docs/NOTES-21.md, spec/crt_f64.py): the CRT of element e of the packed planes wv with the sums in binary64 limbs. The same integer x in [0, P) as crt_planes_n
// (so crt_finish and round_scaled follow unchanged and the result is bit-identical): s_j = sum_l W_l k_lj by fma (exact), the quotient t = floor(fl(S) / P - 2^-30)
// (never above the true quotient, at most one below), r_j = s_j - t P_j (exact), the limbs r_j carried into 32-bit words, one conditional subtraction of P.
template <int N, int L>
__device__ __forceinline__ void crt_f64_elem(const uint32_t (&wv)[N], int e, uint32_t (&x)[L]) {
  constexpr int J = imd::constants::kF64Crt.J[N];
  constexpr int Wb = imd::constants::kF64LimbBits;
  double s[J];
#pragma unroll
  for (int j = 0; j < J; ++j) s[j] = 0.0;
#pragma unroll
  for (int l = 0; l < N; ++l) {
    const double Wd = __uint2double_rn((wv[l] >> (8 * e)) & 255u);
#pragma unroll
    for (int j = 0; j < J; ++j) s[j] = __fma_rn(Wd, imd::device::c_f64crt.k[N][l][j], s[j]);
  }
  double Sd = s[J - 1];
#pragma unroll
  for (int j = J - 2; j >= 0; --j) Sd = __fma_rn(Sd, 1099511627776.0, s[j]);   // 2^40 = 2^kF64LimbBits
  static_assert(Wb == 40, "the Horner base above is 2^40");
  const int t = __double2int_rd(__fma_rn(Sd, imd::device::c_f64crt.Pinv[N], -0x1p-30));      // floor(fl(S) Pinv - 2^-30): the product is fused with the bias, one rounding fewer
  const double td = static_cast<double>(t);
  long long carry = 0;
  uint64_t limb[J];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const long long a = __double2ll_rn(__fma_rn(-td, imd::device::c_f64crt.Pj[N][j], s[j])) + carry;   // r_j: an exact integer below 2^53
    limb[j] = static_cast<uint64_t>(a) & ((uint64_t{1} << Wb) - 1);
    carry = a >> Wb;
  }
  // the carry out of the top limb is zero: R < P (1 + 2^-29) < 2^(40 J) (cert_crt_f64)
#pragma unroll
  for (int k = 0; k < L; ++k) {
    const int bit = 32 * k, j0 = bit / Wb, off = bit % Wb;
    uint64_t v = j0 < J ? limb[j0] >> off : 0;
    if (j0 + 1 < J) v |= limb[j0 + 1] << (Wb - off);
    x[k] = static_cast<uint32_t>(v);
  }
  // R in [0, P (1 + 2^-29)): subtract P once if R >= P
  uint32_t d[L];
  uint32_t borrow = 0;
#pragma unroll
  for (int k = 0; k < L; ++k) {
    const uint64_t diff = static_cast<uint64_t>(x[k]) - c_tables.P[N][k] - borrow;
    d[k] = static_cast<uint32_t>(diff);
    borrow = static_cast<uint32_t>(diff >> 63);
  }
#pragma unroll
  for (int k = 0; k < L; ++k) x[k] = borrow ? x[k] : d[k];
}

// Planes l .. N-1 for E elements (interleaved: E independent chains per step).
template <int N, int L, int E, int l = 0>
__device__ __forceinline__ void crt_planes_n(uint32_t (&x)[E][L], const uint32_t (&wv)[N]) {
  if constexpr (l < N) {
#pragma unroll
    for (int e = 0; e < E; ++e) crt_step_n<N, L, l>(x[e], (wv[l] >> (8 * e)) & 255u);
    crt_planes_n<N, L, E, l + 1>(x, wv);
  }
}

// K5 core: the D values of the E consecutive j-bar j0..j0+E-1 of row (lambda, i), E = 1, 2
// or 4. j0 is a multiple of E, so each plane's E bytes are one aligned load (n_hat is a
// multiple of 16). Elements with j >= n are not computed (D[e] unspecified). Every index into
// x / D / wv is static after unrolling (no local memory).
template <int N, int E, bool F64 = false>
__device__ __forceinline__ void k5_elems_n(const OutArgs& a, int64_t lam, int64_t i, int64_t j0,
                                           double (&D)[E]) {
  static_assert(E == 1 || E == 2 || E == 4, "K5 takes 1, 2 or 4 consecutive j-bar per thread");
  constexpr int L = imd::constants::kTables.L32[N];
  const int64_t plane = a.b * a.m * a.n_hat, row = lam * a.m + i;
  const uint8_t* W = a.W + row * a.n_hat + j0;
  uint32_t wv[N];
#pragma unroll
  for (int l = 0; l < N; ++l) {
    if constexpr (E == 4) wv[l] = __ldg(reinterpret_cast<const uint32_t*>(W + l * plane));
    else if constexpr (E == 2) wv[l] = __ldg(reinterpret_cast<const uint16_t*>(W + l * plane));
    else wv[l] = __ldg(W + l * plane);
  }
  uint32_t x[E][L];
#pragma unroll
  for (int e = 0; e < E; ++e)
#pragma unroll
    for (int w = 0; w < L; ++w) x[e][w] = 0;
  if constexpr (F64) {
#pragma unroll
    for (int e = 0; e < E; ++e) crt_f64_elem<N, L>(wv, e, x[e]);
  } else {
    crt_planes_n<N, L, E>(x, wv);
  }
  const bool fa = a.fA[row] != 0;
  const int sa = a.sA[row];
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int64_t j = j0 + e;
    if (j < a.n) {
      if (fa || a.fB[lam * a.n + j] != 0) {
        D[e] = native_dot(a, lam, i, j);
      } else {
        uint32_t mag[L];
        bool neg;
        crt_finish<L>(x[e], N, mag, &neg);
        D[e] = round_scaled<L>(mag, neg, -(sa + int{a.sB[lam * a.n + j]}));
      }
    }
  }
}

// K5, j-bar fastest (C's fastest label is in J): thread = E consecutive j-bar of one row,
// items (lambda, i, j-bar group) in flattened order, so the W reads of a warp are 32 E
// contiguous bytes per plane. E = 1: thread t stores its own element (consecutive threads,
// consecutive j-bar). E = 4: a warp's 128 results are transposed through shared memory so
// that store r of lane writes slot lane + 32 r (consecutive lanes, consecutive j-bar: C
// stores and c_col loads are coalesced along C's fastest label). Each element is still
// written exactly once with the same value; only the thread that writes it changes.
// E = 2: each thread stores its own two elements. Shared slot of result s (= 4 item + e of the warp's 32 items): s + s / 16, which makes
// both the writes (stride 4 per lane) and the reads (stride 1) bank-conflict free.
constexpr int kK5WarpSlots = 128 + 8;
template <int N, int E, bool F64 = false>
__global__ void __launch_bounds__(kThreads) k5_crt_n(OutArgs a) {
  __shared__ double sD[E == 4 ? (kThreads / 32) * kK5WarpSlots : 1];
  const int64_t ng = (a.n + E - 1) / E, total = a.b * a.m * ng;
  const int lane = threadIdx.x & 31;
  double* wD = sD + (E == 4 ? (threadIdx.x / 32) * kK5WarpSlots : 0);
  for (int64_t base = blockIdx.x * int64_t{blockDim.x}; base < total;
       base += int64_t{gridDim.x} * blockDim.x) {
    const int64_t idx = base + threadIdx.x;
    const bool live = idx < total;
    int64_t jg = 0, cbase = 0;
    double D[E];
#pragma unroll
    for (int e = 0; e < E; ++e) D[e] = 0.0;
    if (live) {
      int64_t t, i, lam;
      divmod(idx, ng, &t, &jg);
      divmod(t, a.m, &lam, &i);
      k5_elems_n<N, E, F64>(a, lam, i, E * jg, D);
      cbase = a.c_bat[lam] + a.c_row[i];
    }
    if constexpr (E != 4) {
      if (live) {
#pragma unroll
        for (int e = 0; e < E; ++e)
          if (E * jg + e < a.n) store_out(a, cbase + a.c_col[E * jg + e], D[e]);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int s = 4 * lane + e;
        wD[s + s / 16] = D[e];
      }
      __syncwarp();
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int s = lane + 32 * r, src = s / 4;
        const int64_t cb = __shfl_sync(0xffffffffu, cbase, src);
        const int64_t j = 4 * __shfl_sync(0xffffffffu, jg, src) + (s & 3);
        const bool ok = __shfl_sync(0xffffffffu, live ? 1 : 0, src) != 0;
        if (ok && j < a.n) store_out(a, cb + a.c_col[j], wD[s + s / 16]);
      }
      __syncwarp();  // wD is rewritten by the next iteration
    }
  }
}

// K5, lambda fastest (C's fastest label is a batch label, NOTES-M2M3 #4): a block takes a
// tile of TL lambda x TJ j-bar of one row i (TL * TJ = 1024). It computes with j-bar
// fastest (coalesced W reads), stages D in shared memory and writes C with lambda fastest
// (coalesced C writes). TL = 2^lg_tl in [2, 32].
template <int N, bool F64 = false>
__global__ void __launch_bounds__(kThreads) k5_crt_bfast(OutArgs a, int lg_tl) {
  constexpr int kTile = 1024;
  __shared__ double sD[kTile + 32];
  const int tl = 1 << lg_tl, lg_tj = 10 - lg_tl, tj = 1 << lg_tj, pitch = tj + 1;
  const int64_t n_lt = (a.b + tl - 1) >> lg_tl, n_jt = (a.n + tj - 1) >> lg_tj;
  const int64_t tiles = a.m * n_lt * n_jt;
  const int tid = threadIdx.x;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int64_t jt = t % n_jt, rest = t / n_jt, lt = rest % n_lt, i = rest / n_lt;
    const int64_t lam0 = lt << lg_tl, j00 = jt << lg_tj;
    {  // compute: thread = (lambda_l, 4 consecutive j-bar), j-bar fastest
      const int jq = tid & ((tj >> 2) - 1), ll = tid >> (lg_tj - 2);
      const int64_t lam = lam0 + ll, j0 = j00 + 4 * jq;
      if (lam < a.b && j0 < a.n) {
        double D[4] = {0, 0, 0, 0};
        k5_elems_n<N, 4, F64>(a, lam, i, j0, D);
#pragma unroll
        for (int e = 0; e < 4; ++e) sD[ll * pitch + 4 * jq + e] = D[e];
      }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < kTile / kThreads; ++it) {  // store: lambda fastest
      const int q = tid + it * kThreads, ll = q & (tl - 1), jl = q >> lg_tl;
      const int64_t lam = lam0 + ll, j = j00 + jl;
      if (lam < a.b && j < a.n)
        store_out(a, a.c_bat[lam] + a.c_row[i] + a.c_col[j], sD[ll * pitch + jl]);
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(kThreads) k_direct(OutArgs a) {
  const int64_t total = a.b * a.m * a.n;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t j = idx % a.n, t = idx / a.n, i = t % a.m, lam = t / a.m;
    store_out(a, a.c_bat[lam] + a.c_row[i] + a.c_col[j], native_dot(a, lam, i, j));
  }
}

// §3.7 alpha = 0 or k = 0: C <- beta == 0 ? +0 : RN(beta C); A, B not read.
__global__ void __launch_bounds__(kThreads) k5_beta_only(OutArgs a) {
  const int64_t total = a.b * a.m * a.n;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t j = idx % a.n, t = idx / a.n, i = t % a.m, lam = t / a.m;
    const int64_t off = a.c_bat[lam] + a.c_row[i] + a.c_col[j];
    a.C[off] = a.beta == 0.0 ? 0.0 : __dmul_rn(a.beta, a.C[off]);
  }
}

// K5 j-bar fastest, elements per thread (a launch choice; the results do not depend on it):
// 1 while b m n <= k5_one_elem_per_sm() / 2 per SM, 2 while b m n <= 8 k5_one_elem_per_sm()
// per SM, else 4 (A5, NOTES-A5 §2: nsys K5 times, 1 / 2 / 4 per thread, 16 K to 16 M
// outputs). The test hooks: imd_test_set_k5_one_elem (0: always 4) and imd_test_set_k5_elems
// (force 1, 2 or 4).
// (kK5OneElemPerSm: host/analysis.hpp.)

// v1.10: the variant and grid are host/analysis.hpp k5_launch (shared with the performance
// model), with this library's test hooks.
template <int N>
void launch_k5(const ExecPlan& p, const OutArgs& out, cudaStream_t stream) {
  const K5Launch L = k5_launch(p.k5_batch_fast, p.b, p.m, p.n, p.sms, k5_one_elem_per_sm(),
                               k5_elems_override());
  const unsigned grid = static_cast<unsigned>(L.grid);
  if (k5_crt_override() == 1) {  // issue 21: the sums in binary64 limbs (test and benchmark hooks only)
    if (L.variant == 3)
      k5_crt_bfast<N, true><<<grid, kThreads, 0, stream>>>(out, L.lg_tl);
    else if (L.elems == 1)
      k5_crt_n<N, 1, true><<<grid, kThreads, 0, stream>>>(out);
    else if (L.elems == 2)
      k5_crt_n<N, 2, true><<<grid, kThreads, 0, stream>>>(out);
    else
      k5_crt_n<N, 4, true><<<grid, kThreads, 0, stream>>>(out);
    return;
  }
  if (L.variant == 3)
    k5_crt_bfast<N><<<grid, kThreads, 0, stream>>>(out, L.lg_tl);
  else if (L.elems == 1)
    k5_crt_n<N, 1><<<grid, kThreads, 0, stream>>>(out);
  else if (L.elems == 2)
    k5_crt_n<N, 2><<<grid, kThreads, 0, stream>>>(out);
  else
    k5_crt_n<N, 4><<<grid, kThreads, 0, stream>>>(out);
}

// K5 for the plan's N: the run-time N picks its kernel (false: N outside [kNMin, kNMax]).
template <int N = imd::constants::kNMin>
bool launch_k5_for(const ExecPlan& p, const OutArgs& out, cudaStream_t stream) {
  if (p.N == N) {
    launch_k5<N>(p, out, stream);
    return true;
  }
  if constexpr (N < imd::constants::kNMax) return launch_k5_for<N + 1>(p, out, stream);
  else return false;
}

// ---- K0: §5 step 10 scatter vectors from the fused-mode descriptor (DESIGN v1.2 §4) -------
//
// The nine vectors in §4a order, each over one class's fused modes (first fastest) with one
// tensor's strides: entry q of a vector is sum_t stride_X(mode_t) digit_t(q), the mixed-radix
// digits of q (index_map.cpp radix_vec computes the same integers incrementally). The
// descriptor is the kernel argument (by value): no host or plan memory is read on the GPU.
// Two capacities keep the argument small for the usual few modes; a class has at most 62
// fused modes (extents >= 2, product < 2^63), so 4 * 62 <= kK0MaxModes.
constexpr int kK0SmallModes = 8, kK0MaxModes = kMaxVecModes;
// Sub-vector v: its class (LabelClass order I, J, K, L) and tensor (A, B, C).
constexpr int8_t kVecClass[9] = {0, 2, 3, 2, 1, 3, 0, 1, 3};
constexpr int8_t kVecTensor[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

template <int Cap>
struct K0Args {
  int64_t start[10];  // sub-vector v covers entries [start[v], start[v + 1])
  int8_t cls[9], tensor[9];
  int32_t first[4], count[4];
  int64_t extent[Cap];
  int64_t stride[Cap][3];
  int64_t* out;
};

// Entry r of class c's vector for tensor x: sum_t stride_x(mode_t) digit_t(r).
template <int Cap>
__device__ __forceinline__ int64_t k0_offset(const K0Args<Cap>& a, int c, int x, int64_t r) {
  int64_t off = 0;
  for (int t = a.first[c]; t < a.first[c] + a.count[c]; ++t) {
    int64_t q, d;
    divmod(r, a.extent[t], &q, &d);
    off += a.stride[t][x] * d;
    r = q;
  }
  return off;
}

// Entry q of the vec region (§4a order).
template <int Cap>
__device__ __forceinline__ int64_t k0_entry(const K0Args<Cap>& a, int64_t q) {
  int v = 0;
  while (q >= a.start[v + 1]) ++v;  // empty sub-vectors are skipped
  return k0_offset(a, a.cls[v], a.tensor[v], q - a.start[v]);
}

template <int Cap>
__global__ void __launch_bounds__(kThreads) k0_vec(const __grid_constant__ K0Args<Cap> a) {
  const int64_t total = a.start[9];
  for (int64_t q = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; q < total;
       q += int64_t{gridDim.x} * blockDim.x)
    a.out[q] = k0_entry(a, q);
}

// The K0 descriptor argument (the plan's fused modes; Cap >= their count). out = vec.
template <int Cap>
K0Args<Cap> k0_args(const ExecPlan& p, int64_t* vec) {
  K0Args<Cap> a{};
  const VecOffsets o = vec_offsets(p);
  const int64_t starts[10] = {o.a_row, o.a_k, o.a_bat, o.b_k, o.b_col,
                              o.b_bat, o.c_row, o.c_col, o.c_bat, p.vec_count};
  for (int v = 0; v < 10; ++v) a.start[v] = starts[v];
  for (int v = 0; v < 9; ++v) a.cls[v] = kVecClass[v], a.tensor[v] = kVecTensor[v];
  for (int c = 0; c < 4; ++c) a.first[c] = p.vec_desc.first[c], a.count[c] = p.vec_desc.count[c];
  for (size_t t = 0; t < p.vec_desc.modes.size(); ++t) {
    a.extent[t] = p.vec_desc.modes[t].extent;
    for (int x = 0; x < 3; ++x) a.stride[t][x] = p.vec_desc.modes[t].stride[x];
  }
  a.out = vec;
  return a;
}

template <int Cap>
imd_status_t launch_k0(const ExecPlan& p, int64_t* vec, cudaStream_t stream) {
  k0_vec<Cap><<<blocks_for(p.vec_count), kThreads, 0, stream>>>(k0_args<Cap>(p, vec));
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

template <class T>
T* region(void* ws, const ExecPlan& p, Region r) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(ws) +
                              p.layout.offset[static_cast<int>(r)]);
}

// ---- KF: K0 + K1 + group 1's K2 / K3 in one launch (DESIGN v1.4 §4) ----------------------
//
// Chosen at plan time when k_hat <= kKfKMax = 1024 (analysis.hpp fused_path; 512 until A3).
// A block of 256 threads takes a tile of th rows by tk = pow2 >= k_hat columns (tk >= 16),
// th tk = 256 E: whole rows. Each thread converts E consecutive k-bar of one row (E = 16, or
// E = 4 for small operands: 4x the blocks and a 4x shorter chain per thread); a row's tk / E
// threads are consecutive lanes, of one warp or (tk / E > 32) of consecutive warps, whose
// partial results are combined in shared memory. A block (tile t, part y):
//   1. row bases v_bat[lambda] + v_row[r] and v_k[h] computed from the descriptor (kernel
//      argument), never read from the vec region (which this launch writes);
//   2. the tile read into shared memory as in k23_pack (coalesced along the pivot);
//   3. per row, the §3.1/§3.2 K1 integers over its lanes by warp shuffles: the flag (or),
//      gamma (max ilogb), S (exact u128 sum of the same terms, order-independent), then
//      sigma = largest_u(S) - gamma (by the row's first lane), exactly K1's value; zero
//      rows sigma = 0;
//   4. part 0 stores sigma and the flag (sA / fA, sB / fB) for the later kernels;
//   5. §3.3/§3.4 residues of the slots [y gsize / ny, (y + 1) gsize / ny) of group 1, as K2/K3.
// Blocks of part 0 then zero the B rows n..n_hat-1 of group 1 and (generate mode) write the
// vec region, grid-strided: the later packs and K5 read them after this launch retires.
// Every part recomputes sigma from the tile (identical integers), so no block reads another
// block's output (spec: Pipeline neg_fused_shared_sigma, neg_fused_reads_vec).

constexpr int kKfKMax = 1024;  // the largest tk: one row of a 4096-value tile per 4 rows
static_assert(kKfKMax == kFusedKHatMax, "KF tiles must hold whole rows of every fused plan");

template <int Cap>
struct FusedArgs {
  K0Args<Cap> d;          // descriptor; d.out = the vec region, or null (resident vectors)
  const double* X[2];     // the GEMM's A and B
  int16_t* sigma[2];
  uint8_t* flag[2];
  int8_t* R[2];
  int64_t rows[2], rows_pad[2], radix[2], tiles[2];
  int along_rows[2];
  int64_t b, k, k_hat;
  int lg_tk, gsize, trunc;
  uint64_t rhs[4];
  int rhs_bits, lhs_shift;
};

__device__ __forceinline__ u128 shfl_xor_u128(u128 v, int d) {
  const auto lo = __shfl_xor_sync(0xffffffffu, static_cast<unsigned long long>(v), d);
  const auto hi = __shfl_xor_sync(0xffffffffu, static_cast<unsigned long long>(v >> 64), d);
  return (static_cast<u128>(hi) << 64) | lo;
}

template <int Cap, int E, int W>  // W: v1.14 §3.4.1 pack_words(N)
__global__ void __launch_bounds__(kThreads) kf_scale_pack(const __grid_constant__ FusedArgs<Cap> a) {
  constexpr int kLgE = E == 16 ? 4 : 2, kTE = kThreads * E, kLgTE = 8 + kLgE;
  constexpr int kRowsMax = kTE / 16;  // th for tk = 16
  // 47.5 KB of static shared memory with E = 16 (the 48 KB limit): the output row offsets are
  // recomputed per thread in step 5 instead of being kept per row here.
  __shared__ uint64_t tile[kTE + kTE / kVec + kRowsMax];
  __shared__ int64_t s_base[kRowsMax];
  __shared__ bool s_valid[kRowsMax];
  __shared__ int64_t s_vk[kKfKMax];
  __shared__ int s_wg[kThreads / 32];
  __shared__ unsigned s_wf[kThreads / 32];
  __shared__ u128 s_ws[kThreads / 32];
  const int tid = threadIdx.x;
  const int lg_tk = a.lg_tk, tk = 1 << lg_tk, lg_th = kLgTE - lg_tk, th = 1 << lg_th;
  const int pitch = tk + tk / kVec + 1, G = tk / E;  // G lanes per row: G | 32 or 32 | G
  const int s_lo = int(blockIdx.y) * a.gsize / int(gridDim.y);
  const int s_hi = (int(blockIdx.y) + 1) * a.gsize / int(gridDim.y);
  const int64_t total = a.tiles[0] + a.tiles[1];
  for (int64_t t = blockIdx.x; t < total; t += gridDim.x) {  // block-uniform
    const int x = t < a.tiles[0] ? 0 : 1;
    const int64_t tt = x == 0 ? t : t - a.tiles[0];
    const int64_t rows = a.rows[x], radix = a.radix[x], D = a.b * rows / radix;
    const int64_t n_d = (D + th - 1) >> lg_th, dt = tt % n_d, lo = tt / n_d, d0 = dt << lg_th;
    // 1. row and column info from the descriptor (class I / J = x, K = 2, L = 3)
    if (tid < th) {
      const int64_t d = d0 + tid;
      const bool valid = d < D;
      s_valid[tid] = valid;
      if (valid) {
        const int64_t rho = lo + radix * d;
        int64_t lam, r;
        divmod(rho, rows, &lam, &r);
        s_base[tid] = k0_offset(a.d, 3, x, lam) + k0_offset(a.d, x, x, r);
      }
    }
    for (int hl = tid; hl < tk; hl += kThreads) s_vk[hl] = hl < a.k ? k0_offset(a.d, 2, x, hl) : 0;
    __syncthreads();
    // 2. coalesced read into shared memory (zero outside the operand)
    const double* X = a.X[x];
    const bool along = a.along_rows[x] != 0;
#pragma unroll 4
    for (int it = 0; it < E; ++it) {
      const int q = tid + it * kThreads;
      const int dl = along ? (q & (th - 1)) : (q >> lg_tk);
      const int hl = along ? (q >> lg_th) : (q & (tk - 1));
      tile[dl * pitch + hl + hl / kVec] =
          s_valid[dl] && hl < a.k ? to_bits(X[s_base[dl] + s_vk[hl]]) : 0;
    }
    __syncthreads();
    // 3. K1 per row over its G lanes
    const int dl = tid >> (lg_tk - kLgE), hl = (tid & (G - 1)) * E;
    uint64_t bits[E];
    const uint64_t* src = tile + dl * pitch + hl + hl / kVec;  // E <= 16: no pad inside
#pragma unroll
    for (int e = 0; e < E; ++e) bits[e] = src[e];
    int gmax = INT_MIN;
    unsigned fl = 0;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      if (non_finite(bits[e])) fl = 1;
      else if (finite_nonzero(bits[e])) gmax = max(gmax, ilogb_bits(bits[e]));
    }
    // Reduce over the row's lanes: within the warp (Gw = min(G, 32) lanes), then, for rows
    // wider than a warp (G = 64 .. 256: E = 4 with tk >= 256, E = 16 with tk = 1024), over the row's warps in
    // shared memory, combined by every thread in the same order (max, or, exact u128 sum).
    const int Gw = G < 32 ? G : 32, warp = tid / 32, lane = tid & 31;
    for (int o = Gw / 2; o > 0; o /= 2) {
      gmax = max(gmax, __shfl_xor_sync(0xffffffffu, gmax, o));
      fl |= __shfl_xor_sync(0xffffffffu, fl, o);
    }
    const int w0 = (dl * G) / 32, nw = G / 32;  // the row's warps (G > 32)
    if (G > 32) {
      if (lane == 0) s_wg[warp] = gmax, s_wf[warp] = fl;
      __syncthreads();
      for (int w = w0; w < w0 + nw; ++w) gmax = max(gmax, s_wg[w]), fl |= s_wf[w];
    }
    u128 S = 0;
    if (gmax != INT_MIN) {
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (finite_nonzero(bits[e])) S += square_term_n(bits[e], gmax);
    }
    for (int o = Gw / 2; o > 0; o /= 2) S += shfl_xor_u128(S, o);
    if (G > 32) {
      if (lane == 0) s_ws[warp] = S;
      __syncthreads();
      S = 0;
      for (int w = w0; w < w0 + nw; ++w) S += s_ws[w];
    }
    // sigma by the first lane of the row's piece in this warp (the 256-bit compare once per
    // row and warp), then broadcast within the warp.
    int sigma = 0;
    if ((lane & (Gw - 1)) == 0 && gmax != INT_MIN)
      sigma = largest_u(S, a.lhs_shift, a.rhs, a.rhs_bits) - gmax;
    sigma = __shfl_sync(0xffffffffu, sigma, lane & ~(Gw - 1));
    const bool valid = s_valid[dl];
    // 4. part 0 stores sigma and the flag (range certified, cert_shift_range)
    if (valid && blockIdx.y == 0 && (tid & (G - 1)) == 0) {
      const int64_t rho = lo + radix * (d0 + dl);
      a.sigma[x][rho] = static_cast<int16_t>(sigma);
      a.flag[x][rho] = static_cast<uint8_t>(fl);
    }
    // 5. residues of this part's slots (K padding h in [k, k_hat) is zero in the tile)
    if (valid && hl < a.k_hat) {
      PackEmitterT<E, W> em;
      em.load(bits, sigma, a.trunc != 0);
      const int64_t plane = a.b * a.rows_pad[x] * a.k_hat;
      int64_t lam, r;
      divmod(lo + radix * (d0 + dl), rows, &lam, &r);
      int8_t* dst = a.R[x] + (lam * a.rows_pad[x] + r) * a.k_hat + hl;
      for (int s = s_lo; s < s_hi; ++s) em.store(s, dst + s * plane);  // group 1: modulus = slot
    }
    __syncthreads();  // shared arrays are reused by the next tile
  }
  if (blockIdx.y != 0) return;
  const int64_t gt = blockIdx.x * int64_t{kThreads} + tid, gs = int64_t{gridDim.x} * kThreads;
  // B rows n..n_hat-1 of every (slot, lambda) of group 1: zero (as k23_zero_rows)
  const int64_t pad_vecs = (a.rows_pad[1] - a.rows[1]) * a.k_hat / kVec;
  for (int64_t idx = gt; idx < int64_t{a.gsize} * a.b * pad_vecs; idx += gs) {
    const int64_t v = idx % pad_vecs, sl = idx / pad_vecs;
    *reinterpret_cast<uint4*>(a.R[1] + (sl * a.rows_pad[1] + a.rows[1]) * a.k_hat + v * kVec) =
        make_uint4(0, 0, 0, 0);
  }
  // the vec region (IMD_VEC_GENERATE): K0's job
  if (a.d.out != nullptr)
    for (int64_t q = gt; q < a.d.start[9]; q += gs) a.d.out[q] = k0_entry(a.d, q);
}

// KF with E = 4 when the E = 16 grid would have fewer than kKfSmallBlocksPerSm blocks per SM
// even with every modulus of the group in its own block row (4x the blocks, a 4x shorter
// chain per thread; its 14 KB of shared memory allow more blocks per SM, so its moduli split
// targets 4 blocks per SM).
// (kKfSmallBlocksPerSm, kf_lg_tk, kf_operand_tiles, kf_small: host/analysis.hpp, shared with
// the performance model.)
template <int Cap>
imd_status_t launch_kf(const ExecPlan& p, const double* A, const double* B, void* ws,
                       int64_t* vec_out, cudaStream_t stream) {
  FusedArgs<Cap> a{};
  a.d = k0_args<Cap>(p, vec_out);
  a.X[0] = A, a.X[1] = B;
  a.sigma[0] = region<int16_t>(ws, p, Region::sA), a.sigma[1] = region<int16_t>(ws, p, Region::sB);
  a.flag[0] = region<uint8_t>(ws, p, Region::fA), a.flag[1] = region<uint8_t>(ws, p, Region::fB);
  a.R[0] = region<int8_t>(ws, p, Region::RA), a.R[1] = region<int8_t>(ws, p, Region::RB);
  a.b = p.b, a.k = p.k, a.k_hat = p.k_hat;
  // k_hat <= kKfKMax: lg_tk <= 10. v1.13 §4b.5: one tile row covers the whole padded row (KF
  // writes the zeros [k, k_hat)); plan_data checks this at plan time.
  const int lg_tk = kf_lg_tk(p.k_hat);
  a.lg_tk = lg_tk;
  a.gsize = std::min(p.g, p.N);
  a.trunc = p.trunc ? 1 : 0;
  for (int w = 0; w < 4; ++w) a.rhs[w] = p.rhs[w];
  a.rhs_bits = p.rhs_bits, a.lhs_shift = p.lhs_shift;
  const auto tiles_for = [&](int lg_te) {
    int64_t total = 0;
    for (int x = 0; x < 2; ++x) {
      a.rows[x] = x == 0 ? p.m : p.n;
      a.rows_pad[x] = x == 0 ? p.m : p.n_hat;
      a.radix[x] = p.pack_radix[x];
      a.along_rows[x] = p.pack_along_rows[x] ? 1 : 0;
      if (p.packed & (1 << x)) {  // v1.7: a packed operand has no tiles (and no RB rows)
        a.tiles[x] = 0;
        a.rows_pad[x] = a.rows[x];
        continue;
      }
      a.tiles[x] = kf_operand_tiles(lg_te, lg_tk, p.b, a.rows[x], a.radix[x]);
      total += a.tiles[x];
    }
    return total;
  };
  int64_t tiles = tiles_for(12);
  // (k5_one_elem_per_sm() == 0, test hook only: no small-problem variants anywhere.)
  const bool small = kf_small(tiles, a.gsize, p.sms) && k5_one_elem_per_sm() != 0;
  if (small) tiles = tiles_for(10);
  const dim3 grid(blocks_for(tiles, 1), pack_split(tiles, a.gsize, p.sms));
  const bool w2 = pack_words(p.N) == 2;  // v1.14 §3.4.1
  if (small) {
    if (w2) kf_scale_pack<Cap, 4, 2><<<grid, kThreads, 0, stream>>>(a);
    else kf_scale_pack<Cap, 4, 3><<<grid, kThreads, 0, stream>>>(a);
  } else {
    if (w2) kf_scale_pack<Cap, 16, 2><<<grid, kThreads, 0, stream>>>(a);
    else kf_scale_pack<Cap, 16, 3><<<grid, kThreads, 0, stream>>>(a);
  }
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

// ---- KL: K1 + group 1's K2 / K3 of ONE operand in one launch (DESIGN v1.6 §4) ------------
//
// For plans without KF (k_hat > kKfKMax), per operand with its fused_large bit (analysis.hpp
// fused_large). A block of 4096 / E threads owns whole rows: a tile of th rows by tk k-bar
// (th tk = kTileElems, the k23_pack tile, shared-memory layout and read phase; reading along
// k-bar tk = pow2 >= k_hat up to 4096, reading along rows th = 8), and a row of k_hat spans
// nc = ceil(k_hat / tk) chunks of tk. In every chunk each thread works on E consecutive k-bar
// of one row (E = 8, or 4 for rows of more than two chunks; a row's tk / E threads are
// consecutive lanes, of one warp or of consecutive warps). Per tile:
//   pass 1  chunks 0 .. nc-1 gathered into shared memory: the flag and gamma (max ilogb);
//           chunk 0 is kept in registers, chunk nc-1 stays in shared memory;
//   reduce  per row over its threads (warp shuffles, a shared-memory combine across warps);
//   pass 2  S over chunk 0 (registers), chunk nc-1 (shared memory), then chunks nc-2 .. 1
//           re-read, most recently read first (L2); sigma = largest_u(S) - gamma: exactly
//           K1's integers (max, or, exact u128 sum: order-independent);
//   pass 3  the residues of every slot of group 1, as K2 / K3: chunk 0 (registers), chunk 1
//           (in shared memory after pass 2, or the last chunk if nc = 2), then chunks
//           2 .. nc-1 re-read;
//   sigma and the flag are stored by the row's first thread.
// Offsets: v_bat / v_row from the vec region (written by K0, which has retired: KL is its own
// launch, spec Pipeline neg_large_with_k0), and v_k likewise, or h * stride when the K class
// is a single fused mode. One CTA computes, uses and stores a row's sigma: no block reads what
// another block of this launch writes (neg_large_row_split). KL_B then zeroes RB rows
// n..n_hat-1 of group 1, grid-strided (as k23_zero_rows).

struct LargeArgs {
  Operand op;                      // vectors: the vec region or the plan's resident copy
  int64_t b, k, k_hat, rows_pad, radix;
  int64_t k_stride;                // k_affine: v_k[h] = h * k_stride
  int k_affine, along_rows, lg_th, gsize, trunc;
  int16_t* sigma;                  // [lambda][rows]
  uint8_t* flag;                   // [lambda][rows]
  int8_t* R;                       // [s][lambda][rows_pad][k_hat], group 1: modulus = slot s
  uint64_t rhs[4];
  int rhs_bits, lhs_shift;
};

__device__ __forceinline__ void scan_bits(uint64_t bits, int& gmax, unsigned& fl) {
  if (non_finite(bits)) fl = 1;
  else if (finite_nonzero(bits)) gmax = max(gmax, ilogb_bits(bits));
}

template <int E, int W>  // values per thread: 8 (512 threads, 2 CTAs per SM) or 4 (1024
                         // threads); W: v1.14 §3.4.1 pack_words(N)
__global__ void __launch_bounds__(kTileElems / E, E == 4 ? 1 : 2)
    kl_scale_pack(const __grid_constant__ LargeArgs a) {
  constexpr int T = kTileElems / E, kLgE = E == 16 ? 4 : E == 8 ? 3 : 2;
  __shared__ uint64_t tile[kTileSmem];
  __shared__ int64_t s_base[kTileRowsMax];  // v_bat[lambda] + v_row[r]
  __shared__ bool s_valid[kTileRowsMax];
  __shared__ int s_wg[T / 32];
  __shared__ unsigned s_wf[T / 32];
  __shared__ u128 s_ws[T / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lg_th = a.lg_th, th = 1 << lg_th, lg_tk = 12 - lg_th, tk = 1 << lg_tk;
  const int pitch = tk + tk / kVec + 1;
  const int G = tk / E, Gw = G < 32 ? G : 32;  // threads per row; G | 32 or 32 | G
  const int nw = G > 32 ? G / 32 : 1, w0 = warp & ~(nw - 1);  // the row's warps (G > 32)
  const int dl = tid >> (lg_tk - kLgE), hl = (tid & (G - 1)) * E;  // this thread: row, k-bar
  const uint64_t* mine = tile + dl * pitch + hl + hl / kVec;
  const int64_t rows = a.op.rows, D = a.b * rows / a.radix;
  const int64_t n_d = (D + th - 1) >> lg_th, tiles = a.radix * n_d;
  const int64_t nc = (a.k_hat + tk - 1) >> lg_tk;
  const int64_t plane = a.b * a.rows_pad * a.k_hat;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {  // block-uniform
    const int64_t dt = t % n_d, lo = t / n_d, d0 = dt << lg_th;
    if (tid < th) {
      const int64_t d = d0 + tid;
      const bool valid = d < D;
      s_valid[tid] = valid;
      if (valid) {
        int64_t lam, r;
        divmod(lo + a.radix * d, rows, &lam, &r);
        s_base[tid] = a.op.v_bat[lam] + a.op.v_row[r];
      }
    }
    __syncthreads();
    // Gather chunk c into the tile (zero outside the operand), coalesced as in k23_pack.
    const auto load = [&](int64_t c) {
      const int64_t h0 = c << lg_tk, h_end = a.k - h0;
#pragma unroll 4
      for (int it = 0; it < E; ++it) {
        const int q = tid + it * T;
        const int ql = a.along_rows ? (q & (th - 1)) : (q >> lg_tk);
        const int qh = a.along_rows ? (q >> lg_th) : (q & (tk - 1));
        uint64_t v = 0;
        if (s_valid[ql] && qh < h_end) {
          const int64_t h = h0 + qh;
          v = to_bits(a.op.X[s_base[ql] + (a.k_affine ? h * a.k_stride : a.op.v_k[h])]);
        }
        tile[ql * pitch + qh + qh / kVec] = v;
      }
    };
    // pass 1: flag and gamma; chunk 0 into registers
    load(0);
    __syncthreads();
    uint64_t r0[E];
#pragma unroll
    for (int e = 0; e < E; ++e) r0[e] = mine[e];
    int gmax = INT_MIN;
    unsigned fl = 0;
#pragma unroll
    for (int e = 0; e < E; ++e) scan_bits(r0[e], gmax, fl);
    for (int64_t c = 1; c < nc; ++c) {
      __syncthreads();  // every thread has taken chunk c - 1
      load(c);
      __syncthreads();
#pragma unroll
      for (int e = 0; e < E; ++e) scan_bits(mine[e], gmax, fl);
    }
    for (int o = Gw / 2; o > 0; o /= 2) {
      gmax = max(gmax, __shfl_xor_sync(0xffffffffu, gmax, o));
      fl |= __shfl_xor_sync(0xffffffffu, fl, o);
    }
    if (G > 32) {  // block-uniform
      if (lane == 0) s_wg[warp] = gmax, s_wf[warp] = fl;
      __syncthreads();
      for (int w = w0; w < w0 + nw; ++w) gmax = max(gmax, s_wg[w]), fl |= s_wf[w];
    }
    // pass 2: S (exact, order-independent) from registers, shared memory, then L2
    const bool live = gmax != INT_MIN;  // the row has a finite nonzero entry
    u128 S = 0;
    if (live) {
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (finite_nonzero(r0[e])) S += square_term_n(r0[e], gmax);
    }
    if (nc > 1) {
      if (live) {  // chunk nc - 1, still in shared memory
#pragma unroll
        for (int e = 0; e < E; ++e) {
          const uint64_t bits = mine[e];
          if (finite_nonzero(bits)) S += square_term_n(bits, gmax);
        }
      }
      for (int64_t c = nc - 2; c >= 1; --c) {
        __syncthreads();
        load(c);
        __syncthreads();
        if (live) {
#pragma unroll
          for (int e = 0; e < E; ++e) {
            const uint64_t bits = mine[e];
            if (finite_nonzero(bits)) S += square_term_n(bits, gmax);
          }
        }
      }
    }
    for (int o = Gw / 2; o > 0; o /= 2) S += shfl_xor_u128(S, o);
    if (G > 32) {
      if (lane == 0) s_ws[warp] = S;
      __syncthreads();
      S = 0;
      for (int w = w0; w < w0 + nw; ++w) S += s_ws[w];
    }
    // sigma by the first lane of the row's piece in this warp, broadcast within the warp
    int sigma = 0;  // zero row (all entries zero or non-finite): sigma := 0
    if ((lane & (Gw - 1)) == 0 && live) sigma = largest_u(S, a.lhs_shift, a.rhs, a.rhs_bits) - gmax;
    sigma = __shfl_sync(0xffffffffu, sigma, lane & ~(Gw - 1));
    const bool valid = s_valid[dl];
    int64_t dst_off = 0;  // (lambda rows_pad + r) k_hat + hl
    if (valid) {
      const int64_t rho = lo + a.radix * (d0 + dl);
      if ((tid & (G - 1)) == 0) {  // range certified (cert_shift_range)
        a.sigma[rho] = static_cast<int16_t>(sigma);
        a.flag[rho] = static_cast<uint8_t>(fl);
      }
      int64_t lam, r;
      divmod(rho, rows, &lam, &r);
      dst_off = (lam * a.rows_pad + r) * a.k_hat + hl;
    }
    // pass 3: residues of every slot of group 1 (K padding h in [k, k_hat) is zero)
    const auto emit = [&](const uint64_t (&bits)[E], int64_t c) {
      if (valid && (c << lg_tk) + hl < a.k_hat) {  // k_hat is a multiple of 16
        PackEmitterT<E, W> em;
        em.load(bits, sigma, a.trunc != 0);
        int8_t* dst = a.R + dst_off + (c << lg_tk);
        for (int s = 0; s < a.gsize; ++s) em.store(s, dst + s * plane);
      }
    };
    emit(r0, 0);
    for (int64_t c = 1; c < nc; ++c) {
      if (c > 1) {  // chunk 1 is in shared memory after pass 2
        __syncthreads();
        load(c);
        __syncthreads();
      }
      uint64_t bits[E];
#pragma unroll
      for (int e = 0; e < E; ++e) bits[e] = mine[e];
      emit(bits, c);
    }
    __syncthreads();  // the tile and the row info are reused by the next tile
  }
  // B rows n..n_hat-1 of every (slot, lambda) of group 1: zero (as k23_zero_rows)
  if (a.rows_pad > rows) {
    const int64_t gt = blockIdx.x * int64_t{T} + tid, gs = int64_t{gridDim.x} * T;
    const int64_t pad_vecs = (a.rows_pad - rows) * a.k_hat / kVec;
    for (int64_t idx = gt; idx < int64_t{a.gsize} * a.b * pad_vecs; idx += gs) {
      const int64_t v = idx % pad_vecs, sl = idx / pad_vecs;
      *reinterpret_cast<uint4*>(a.R + (sl * a.rows_pad + rows) * a.k_hat + v * kVec) =
          make_uint4(0, 0, 0, 0);
    }
  }
}

// KL for operand x of the plan (DESIGN v1.6 §4): tile height and grid.
imd_status_t launch_kl(const ExecPlan& p, int x, const Operand& op, int16_t* sigma, uint8_t* flag,
                       int8_t* R, cudaStream_t stream) {
  LargeArgs a{};
  a.op = op;
  a.b = p.b, a.k = p.k, a.k_hat = p.k_hat;
  a.rows_pad = x == 0 ? p.m : p.n_hat;
  a.radix = p.pack_radix[x];
  a.along_rows = p.pack_along_rows[x] ? 1 : 0;
  a.gsize = std::min(p.g, p.N);
  a.trunc = p.trunc ? 1 : 0;
  a.sigma = sigma, a.flag = flag, a.R = R;
  for (int w = 0; w < 4; ++w) a.rhs[w] = p.rhs[w];
  a.rhs_bits = p.rhs_bits, a.lhs_shift = p.lhs_shift;
  // K class of one fused mode: v_k[h] = h * stride (no vector load per element).
  const VecDesc& vd = p.vec_desc;
  if (vd.count[2] == 1) {
    a.k_affine = 1;
    a.k_stride = vd.modes[static_cast<size_t>(vd.first[2])].stride[x];
  }
  // Tile: along rows 8 rows side by side (64-byte segments); along k-bar tk = pow2 >= k_hat,
  // at most the whole tile (th = 1). The test hook forces 64-row tiles (many chunks per row).
  // (host/analysis.hpp kl_tile_lg_k: the one rule, also used by the performance model.)
  // v1.10: the launch shape is host/analysis.hpp kl_launch (shared with the performance model).
  const KlLaunch L = kl_launch(a.along_rows != 0, p.kl_lg_rows, p.k_hat, p.b, op.rows, a.radix);
  a.lg_th = L.lg_th;
  // Threads per CTA (NOTES-D4 §2): rows held on chip (nc <= 2): 8 values per thread, 512
  // threads, two CTAs per SM; longer rows (chunks re-read from L2): 4 values per thread, 1024
  // threads, one CTA per SM, which also keeps the rows in flight at SMs x th x 8 k_hat bytes
  // (r_bigk 32 MB, r_mm 51 MB: inside the L2 budget of K1, A3).
  const unsigned grid = static_cast<unsigned>(L.grid);
  const bool w2 = pack_words(p.N) == 2;  // v1.14 §3.4.1 (imd_pack: the pack's N)
  if (L.values == 4) {
    if (w2) kl_scale_pack<4, 2><<<grid, kTileElems / 4, 0, stream>>>(a);
    else kl_scale_pack<4, 3><<<grid, kTileElems / 4, 0, stream>>>(a);
  } else {
    if (w2) kl_scale_pack<8, 2><<<grid, kTileElems / 8, 0, stream>>>(a);
    else kl_scale_pack<8, 3><<<grid, kTileElems / 8, 0, stream>>>(a);
  }
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace

// K1 of operand x (0 = A, 1 = B): flags and sigma / tau.
static void launch_k1(const ExecPlan& p, int x, const Operand& op, int16_t* sigma, uint8_t* flag,
                      cudaStream_t stream) {
  ScaleArgs sa{};
  sa.b = p.b, sa.k = p.k, sa.rhs_bits = p.rhs_bits, sa.lhs_shift = p.lhs_shift;
  for (int w = 0; w < 4; ++w) sa.rhs[w] = p.rhs[w];
  sa.op = op;
  sa.sigma = sigma;
  sa.flag = flag;
  const int64_t rows = p.b * sa.op.rows;
  if (p.pack_along_rows[x]) {  // rows are read side by side: tiles as in the pack
    const int64_t radix = p.pack_radix[x], D = rows / radix;
    // 8 rows side by side (64-byte segments, whole sectors) keep the per-block re-read
    // footprint small, so more rows fit in L2 for the second pass.
    const int lg_th = 3;
    const int64_t tiles = radix * ((D + (int64_t{1} << lg_th) - 1) >> lg_th);
    const unsigned blocks =
        k1_blocks(tiles, 8.0 * double(p.k) * double(int64_t{1} << lg_th), p.sms, p.l2_bytes);
    if (int64_t{blocks} < 2 * int64_t{p.sms})
      k1_tiled<1024><<<blocks, 1024, 0, stream>>>(sa, radix, lg_th);
    else
      k1_tiled<kThreads><<<blocks, kThreads, 0, stream>>>(sa, radix, lg_th);
  } else {
    const unsigned blocks = k1_blocks(rows, 8.0 * double(p.k), p.sms, p.l2_bytes);
    if (int64_t{blocks} < 2 * int64_t{p.sms} && rows > int64_t{blocks} && blocks >= unsigned(p.sms) &&
        2.0 * p.sms * 8.0 * double(p.k) <= k1_l2_budget_bytes(p.l2_bytes))
      // Long rows, capped grid: two 512-thread blocks per SM, so one block streams a row
      // while the other is in its reductions / sigma tail (A4, r_bigk). A3: only while the
      // 2 x SMs rows in flight fit the L2 budget; beyond it (r_mm: 400 KB rows, 102 MB in
      // flight) the second pass missed L2 and K1 read the operand twice from DRAM.
      k1_scale<512><<<2 * p.sms, 512, 0, stream>>>(sa);
    else if (int64_t{blocks} < 2 * int64_t{p.sms}) {
      // A3: rows in flight beyond the L2 budget (x300_100k: 128 rows of 800 KB): walk
      // pass 2 backwards (NOTES-A3 §4; forward is faster while the rows fit: r_mm).
      sa.reverse = double(std::min<int64_t>(blocks, rows)) * 8.0 * double(p.k) > k1_l2_budget_bytes(p.l2_bytes);
      k1_scale<1024><<<blocks, 1024, 0, stream>>>(sa);
      sa.reverse = 0;
    }
    else
      k1_scale<kThreads><<<blocks, kThreads, 0, stream>>>(sa);
  }
  IMD_PROF(stream, x == 0 ? "K1 scale A" : "K1 scale B");
}

int64_t& k5_one_elem_per_sm() {
  static int64_t v = kK5OneElemPerSm;  // only the test hook changes it
  return v;
}

int& k5_elems_override() {
  static int v = 0;  // only the test and benchmark hooks change it
  return v;
}

int& k5_crt_override() {
  static int v = 0;  // 0: the integer running CRT (default), 1: the sums in binary64 limbs (issue 21); only the hooks change it
  return v;
}

std::atomic<int>& pad_mutation() {
  static std::atomic<int> v{-1};  // only the test hook imd_test_set_pad_mutation changes it
  return v;
}

imd_status_t k0_generate(const ExecPlan& p, int64_t* vec, cudaStream_t stream) {
  if (p.vec_count == 0) return IMD_STATUS_SUCCESS;
  if (p.vec_desc.modes.size() <= static_cast<size_t>(kK0SmallModes))
    return launch_k0<kK0SmallModes>(p, vec, stream);
  if (p.vec_desc.modes.size() <= static_cast<size_t>(kK0MaxModes))
    return launch_k0<kK0MaxModes>(p, vec, stream);
  return IMD_STATUS_CUDA_ERROR;  // unreachable: at most 4 * 62 fused modes
}

namespace {

// The K4 operands of an emulated call: RA / RB (the workspace's slots, or v1.7 a packed
// operand's residue planes), the W planes, the K4 batch strides (0: the workspace's; a
// packed operand's r k_hat, §6b.4: the B role reads blocks at the pack's stride) and the
// offset per modulus of a residue group's first plane (a packed operand: b batch; the
// workspace's slots: 0, every group reuses them). enqueue_contract's launches and its
// pointer-check pre-pass (k4_check_call) both use this one computation.
struct K4Operands {
  int8_t* R[2] = {nullptr, nullptr};
  uint8_t* W = nullptr;
  int64_t batch[2] = {0, 0};
  int64_t group[2] = {0, 0};
};
K4Operands k4_operands(const ExecPlan& p, void* ws, const void* const* packed) {
  K4Operands o;
  o.R[0] = region<int8_t>(ws, p, Region::RA);
  o.R[1] = region<int8_t>(ws, p, Region::RB);
  o.W = region<uint8_t>(ws, p, Region::W);
  for (int x = 0; x < 2; ++x) {
    if (!(p.packed & (1 << x))) continue;
    auto* base = static_cast<unsigned char*>(const_cast<void*>(packed[x]));
    o.R[x] = reinterpret_cast<int8_t*>(base + p.pack_off[x][kPackRes]);
    o.batch[x] = (x == 0 ? p.m : p.n) * p.k_hat;
    o.group[x] = p.b * o.batch[x];
  }
  return o;
}

}  // namespace

imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream,
                              const void* const* packed) {
  if (p.path == Path::kNoop) return IMD_STATUS_SUCCESS;
  // v1.13 §4b.5 (hygiene): the K4 operand-pointer check (debug builds, test hook) of the whole
  // call, before anything is queued: a violation fails the call with nothing launched.
  K4Operands k4o{};
  const bool emulated = p.path == Path::kGemm && alpha != 0.0 && p.k >= 2 && p.k4 != nullptr;
  if (emulated) {
    k4o = k4_operands(p, ws, packed);
    if (const imd_status_t st = k4_check_call(*p.k4, p.N, p.g, k4o.R[0], k4o.R[1], k4o.W,
                                              k4o.group[0], k4o.group[1], k4o.batch[0],
                                              k4o.batch[1], p.k_align);
        st != IMD_STATUS_SUCCESS)
      return st;
  }
  // K0: generate the scatter vectors into ws.vec (IMD_VEC_GENERATE), or use the plan's
  // resident copy (IMD_VEC_RESIDENT: no K0, no vec region).
  IMD_PROF(stream, "begin");
  int64_t* vec = p.vec_resident ? p.vec_dev : region<int64_t>(ws, p, Region::vec);
  // v1.4: with the fused path, KF writes the vectors (and K1's and group 1's outputs).
  const bool use_kf = p.fused && alpha != 0.0;  // p.fused implies the emulated GEMM, k >= 2
  if (!p.vec_resident && !use_kf) {
    if (const imd_status_t st = k0_generate(p, vec, stream); st != IMD_STATUS_SUCCESS) return st;
    IMD_PROF(stream, "K0 vec");
  }
  const VecOffsets vo = vec_offsets(p);

  OutArgs out{};
  out.A = A;
  out.B = B;
  out.C = C;
  out.a_row = vec + vo.a_row, out.a_k = vec + vo.a_k, out.a_bat = vec + vo.a_bat;
  out.b_k = vec + vo.b_k, out.b_col = vec + vo.b_col, out.b_bat = vec + vo.b_bat;
  out.c_row = vec + vo.c_row, out.c_col = vec + vo.c_col, out.c_bat = vec + vo.c_bat;
  out.m = p.m, out.n = p.n, out.k = p.k, out.b = p.b, out.n_hat = p.n_hat;
  out.N = p.N;
  out.alpha = alpha;
  out.beta = beta;
  const unsigned out_blocks = blocks_for(p.b * p.m * p.n);

  if (alpha == 0.0 || p.k == 0) {  // §3.7 beta-only mode: K1-K4 are not launched
    k5_beta_only<<<out_blocks, kThreads, 0, stream>>>(out);
    IMD_PROF(stream, "K5 beta-only");
    return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
  }
  if (p.path == Path::kDirect) {
    k_direct<<<out_blocks, kThreads, 0, stream>>>(out);
    IMD_PROF(stream, "DIRECT");
    return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
  }

  // Emulated path.
  auto* sA = region<int16_t>(ws, p, Region::sA);
  auto* sB = region<int16_t>(ws, p, Region::sB);
  auto* fA = region<uint8_t>(ws, p, Region::fA);
  auto* fB = region<uint8_t>(ws, p, Region::fB);
  // v1.7 (DESIGN §6b): a packed operand's sigma, flags and residues come from its packed
  // buffer (only read; the workspace has no sX, fX, RX); k4_operands (also the pre-pass's).
  for (int x = 0; x < 2; ++x) {
    if (!(p.packed & (1 << x))) continue;
    auto* base = static_cast<unsigned char*>(const_cast<void*>(packed[x]));
    (x == 0 ? sA : sB) = reinterpret_cast<int16_t*>(base + p.pack_off[x][kPackSig]);
    (x == 0 ? fA : fB) = reinterpret_cast<uint8_t*>(base + p.pack_off[x][kPackFlag]);
  }
  if (!emulated) k4o = k4_operands(p, ws, packed);  // defensive: here k >= 2 and K4 exists
  int8_t* const RA = k4o.R[0];
  int8_t* const RB = k4o.R[1];
  uint8_t* const W = k4o.W;
  const int64_t* const batch = k4o.batch;
  // The GEMM operand of group g0: the workspace's slots, or the pack's planes g0...
  const auto operand = [&](int x, int g0) -> const int8_t* {
    return k4o.R[x] + int64_t{g0} * k4o.group[x];
  };
  const Operand opA{A, out.a_bat, out.a_row, out.a_k, p.m};
  const Operand opB{B, out.b_bat, out.b_col, out.b_k, p.n};

  if (use_kf) {  // KF: K0 + K1 + group 1's packs
    int64_t* vec_out = p.vec_resident ? nullptr : vec;
    const imd_status_t st =
        p.vec_desc.modes.size() <= static_cast<size_t>(kK0SmallModes)
            ? launch_kf<kK0SmallModes>(p, A, B, ws, vec_out, stream)
            : launch_kf<kK0MaxModes>(p, A, B, ws, vec_out, stream);
    if (st != IMD_STATUS_SUCCESS) return st;
    IMD_PROF(stream, "KF fused");
  } else {  // per operand: K1, or v1.6 KL (K1 + its group-1 pack; analysis.hpp fused_large)
    for (int x = 0; x < 2; ++x) {
      const Operand& op = x == 0 ? opA : opB;
      if (p.packed & (1 << x)) continue;  // v1.7: scaled and packed by imd_pack
      if (p.fused_large & (1 << x)) {
        if (const imd_status_t st = launch_kl(p, x, op, x == 0 ? sA : sB, x == 0 ? fA : fB,
                                              x == 0 ? RA : RB, stream);
            st != IMD_STATUS_SUCCESS)
          return st;
        IMD_PROF(stream, x == 0 ? "KL A" : "KL B");
      } else {
        launch_k1(p, x, op, x == 0 ? sA : sB, x == 0 ? fA : fB, stream);
        if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
      }
    }
  }

  // K2 / K3 / K4 per residue group
  for (int g0 = 0; g0 < p.N; g0 += p.g) {
    const int gsize = std::min(p.g, p.N - g0);
    if (use_kf && g0 == 0) {  // group 1 was packed by KF
      if (const imd_status_t st = k4_group(*p.k4, operand(0, g0), operand(1, g0), W, g0, gsize,
                                           stream, batch[0], batch[1]);
          st != IMD_STATUS_SUCCESS)
        return st;
      continue;
    }
    PackArgs pa{};
    pa.b = p.b, pa.k = p.k, pa.k_hat = p.k_hat, pa.g0 = g0, pa.gsize = gsize, pa.trunc = p.trunc;
    for (int x = 0; x < 2; ++x) {
      if (p.packed & (1 << x)) continue;                     // v1.7: its planes are in the pack
      if (g0 == 0 && (p.fused_large & (1 << x))) continue;  // group 1 of x was packed by KL
      pa.op = x == 0 ? opA : opB;
      pa.rows_pad = x == 0 ? p.m : p.n_hat;
      pa.sigma = x == 0 ? sA : sB;
      pa.R = x == 0 ? RA : RB;
      // production: every k-bar < k_hat (zeros past k); the test mutation stops at ceil(k)_16
      const int mutated = pad_mutation().load(std::memory_order_relaxed);
      pa.k_store = mutated == x ? (p.k + 15) / 16 * 16 : p.k_hat;
      pa.radix = p.pack_radix[x];
      pa.along_rows = p.pack_along_rows[x];
      // v1.10: host/analysis.hpp pack_launch (shared with the performance model).
      const PackLaunch L = pack_launch(pa.along_rows, p.b, pa.op.rows, pa.radix, p.k_hat, gsize,
                                       p.sms);
      pa.lg_th = L.lg_th;
      const dim3 grid(static_cast<unsigned>(L.grid_x), static_cast<unsigned>(L.parts));
      if (pack_words(p.N) == 2) k23_pack<2><<<grid, kThreads, 0, stream>>>(pa);  // v1.14 §3.4.1
      else k23_pack<3><<<grid, kThreads, 0, stream>>>(pa);
      if (pa.rows_pad > pa.op.rows)
        k23_zero_rows<<<blocks_for(gsize * p.b * (pa.rows_pad - pa.op.rows) * p.k_hat / kVec),
                        kThreads, 0, stream>>>(pa.R, pa.op.rows, pa.rows_pad, p.b, p.k_hat,
                                               gsize);
      IMD_PROF(stream, x == 0 ? "K2 pack A" : "K3 pack B");
    }
    if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
    if (const imd_status_t st = k4_group(*p.k4, operand(0, g0), operand(1, g0), W, g0, gsize,
                                         stream, batch[0], batch[1]);
        st != IMD_STATUS_SUCCESS)
      return st;
  }

  // K5
  out.sA = sA, out.sB = sB, out.fA = fA, out.fB = fB, out.W = W;
  if (!launch_k5_for(p, out, stream)) return IMD_STATUS_CUDA_ERROR;  // unreachable: N in 2..20
  IMD_PROF(stream, "K5 crt");
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

// ---- imd_pack (DESIGN v1.7 §6b): X's scatter vectors and the packed buffer's zero bytes,
// then one KL launch for sigma, the flags and all N residue planes ------------------------
//
// The pack is X as the GEMM's A of a one-operand view (pack_exec: rows m = r, g = N), so KL
// (v1.6) writes exactly the §6b.4 rows: sigma and the flag at lambda r + rho, plane l of row
// (lambda, rho) at l b r k_hat + (lambda r + rho) k_hat with zero K padding (rows_pad = r, the
// group being every plane). kp_prep writes everything else of [0, pack_bytes): the two gaps
// after sig and flag and, from the end of the planes, the tail and the last gap. No byte is
// written twice; no block reads what another writes (KL reads only X and the vectors, which
// kp_prep wrote in the previous launch).

namespace {

template <int Cap>
__global__ void __launch_bounds__(kThreads)
    kp_prep(const __grid_constant__ K0Args<Cap> a, uint8_t* packed, uint64_t g0_lo, uint64_t g0_hi,
            uint64_t g1_lo, uint64_t g1_hi, uint64_t t_lo, uint64_t t_hi) {
  const uint64_t gt = blockIdx.x * uint64_t{blockDim.x} + threadIdx.x;
  const uint64_t gs = uint64_t{gridDim.x} * blockDim.x;
  for (uint64_t q = gt; q < static_cast<uint64_t>(a.start[9]); q += gs)
    a.out[q] = k0_entry(a, static_cast<int64_t>(q));
  for (uint64_t i = g0_lo + gt; i < g0_hi; i += gs) packed[i] = 0;   // after sig (< 256 B)
  for (uint64_t i = g1_lo + gt; i < g1_hi; i += gs) packed[i] = 0;   // after flag (< 256 B)
  for (uint64_t i = t_lo + kVec * gt; i < t_hi; i += kVec * gs)      // tail + last gap
    *reinterpret_cast<uint4*>(packed + i) = make_uint4(0, 0, 0, 0);
}

// The vectors of a pack: rows (class I = the pack's R), k-bar (K) and batches (L), X's
// strides in slot A, in the vec region's first three sub-vectors (the others are empty).
template <int Cap>
K0Args<Cap> kp_args(const ExecPlan& p, int64_t* vec) {
  K0Args<Cap> a{};
  const int64_t r = p.m, k = p.k, b = p.b;
  for (int v = 0; v < 10; ++v) a.start[v] = v == 0 ? 0 : v == 1 ? r : v == 2 ? r + k : r + k + b;
  for (int v = 0; v < 9; ++v) a.cls[v] = kVecClass[v], a.tensor[v] = 0;
  for (int c = 0; c < 4; ++c) a.first[c] = p.vec_desc.first[c], a.count[c] = p.vec_desc.count[c];
  for (size_t t = 0; t < p.vec_desc.modes.size(); ++t) {
    a.extent[t] = p.vec_desc.modes[t].extent;
    for (int x = 0; x < 3; ++x) a.stride[t][x] = p.vec_desc.modes[t].stride[x];
  }
  a.out = vec;
  return a;
}
static_assert(kVecClass[0] == 0 && kVecClass[1] == 2 && kVecClass[2] == 3,
              "sub-vectors 0-2 are A's rows (I), k-bar (K) and batches (L)");

template <int Cap>
imd_status_t launch_kp_prep(const ExecPlan& p, const PackLayout& pl, uint8_t* packed,
                            int64_t* vec, cudaStream_t stream) {
  const uint64_t sig_end = pl.offset[kPackSig] + pl.bytes[kPackSig];
  const uint64_t flag_end = pl.offset[kPackFlag] + pl.bytes[kPackFlag];
  const uint64_t planes_end = pl.offset[kPackRes] + static_cast<uint64_t>(p.N) *
                                                        static_cast<uint64_t>(p.b * p.m * p.k_hat);
  const int64_t work = std::max<int64_t>(
      p.vec_count, static_cast<int64_t>((pl.total - planes_end) / kVec));
  kp_prep<Cap><<<blocks_for(work), kThreads, 0, stream>>>(
      kp_args<Cap>(p, vec), packed, sig_end, pl.offset[kPackFlag], flag_end, pl.offset[kPackRes],
      planes_end, pl.total);
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace

int32_t pack_kl_lg_rows(const ExecPlan& p) {
  // host/analysis.hpp (v1.10: shared with the performance model): 8 rows (tk = 512) .. 64
  // rows (tk = 64) side by side; -1 (KL's rule) along k-bar.
  return pack_kl_lg_rows_for(p.pack_along_rows[0], p.k_hat);
}

imd_status_t enqueue_pack(const ExecPlan& p, const PackLayout& pl, const double* X, void* packed,
                          void* ws, cudaStream_t stream) {
  if (pl.total == 0) return IMD_STATUS_SUCCESS;  // an empty pack
  auto* out = static_cast<uint8_t*>(packed);
  auto* vec = static_cast<int64_t*>(ws);
  const imd_status_t st = p.vec_desc.modes.size() <= static_cast<size_t>(kK0SmallModes)
                              ? launch_kp_prep<kK0SmallModes>(p, pl, out, vec, stream)
                              : launch_kp_prep<kK0MaxModes>(p, pl, out, vec, stream);
  if (st != IMD_STATUS_SUCCESS) return st;
  IMD_PROF(stream, "pack prep");
  const Operand op{X, vec + p.m + p.k, vec, vec + p.m, p.m};  // v_bat, v_row, v_k, rows = r
  if (const imd_status_t s2 = launch_kl(p, 0, op, reinterpret_cast<int16_t*>(out + pl.offset[kPackSig]),
                                        out + pl.offset[kPackFlag],
                                        reinterpret_cast<int8_t*>(out + pl.offset[kPackRes]), stream);
      s2 != IMD_STATUS_SUCCESS)
    return s2;
  IMD_PROF(stream, "pack KL");
  return IMD_STATUS_SUCCESS;
}


// ---- v1.10 (DESIGN §6c.1): occupancy of every kernel for the performance-model calibration --
namespace {

// The code version of the kernel the last occupancy_of call on this thread looked at (issue 18:
// kernel_versions); set whenever cudaFuncGetAttributes succeeded.
thread_local int32_t g_last_binary = 0, g_last_ptx = 0;

template <class Kernel>
cudaError_t occupancy_of(Kernel kernel, int threads, int32_t* out) {
  cudaFuncAttributes a{};
  cudaError_t e = cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(kernel));
  if (e == cudaSuccess) g_last_binary = a.binaryVersion, g_last_ptx = a.ptxVersion;
  int r = 0;
  if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, kernel, threads, 0);
  if (e != cudaSuccess) return e;
  out[0] = r, out[1] = a.numRegs, out[2] = static_cast<int32_t>(a.sharedSizeBytes);
  out[3] = threads;
  return cudaSuccess;
}

template <int N = imd::constants::kNMin>
cudaError_t k5_occupancy(int n, int variant, int32_t* out) {
  if (n == N) {
    switch (variant) {
      case 0: return occupancy_of(k5_crt_n<N, 1>, kThreads, out);
      case 1: return occupancy_of(k5_crt_n<N, 2>, kThreads, out);
      case 2: return occupancy_of(k5_crt_n<N, 4>, kThreads, out);
      default: return occupancy_of(k5_crt_bfast<N>, kThreads, out);
    }
  }
  if constexpr (N < imd::constants::kNMax) return k5_occupancy<N + 1>(n, variant, out);
  else return cudaErrorInvalidValue;
}

// v1.16 §3.4.1: one W instantiation (2, 3) of a pack kernel.
template <int W>
cudaError_t pack_occupancy(int32_t kernel, int32_t* out) {
  switch (kernel) {
    case IMD_PERF_KERNEL_KF16: return occupancy_of(kf_scale_pack<kK0SmallModes, 16, W>, kThreads, out);
    case IMD_PERF_KERNEL_KF4: return occupancy_of(kf_scale_pack<kK0SmallModes, 4, W>, kThreads, out);
    case IMD_PERF_KERNEL_KL8: return occupancy_of(kl_scale_pack<8, W>, kTileElems / 8, out);
    case IMD_PERF_KERNEL_KL4: return occupancy_of(kl_scale_pack<4, W>, kTileElems / 4, out);
    case IMD_PERF_KERNEL_PACK: return occupancy_of(k23_pack<W>, kThreads, out);
    default: return cudaErrorInvalidValue;
  }
}

bool is_pack_kernel(int32_t kernel) {
  return kernel == IMD_PERF_KERNEL_KF16 || kernel == IMD_PERF_KERNEL_KF4 ||
         kernel == IMD_PERF_KERNEL_KL8 || kernel == IMD_PERF_KERNEL_KL4 ||
         kernel == IMD_PERF_KERNEL_PACK;
}

}  // namespace

cudaError_t kernel_occupancy_w(int32_t kernel, int32_t w, int32_t* out) {
  if (w == 0) return kernel_occupancy(kernel, out);
  if (!is_pack_kernel(kernel)) return cudaErrorInvalidValue;
  switch (w) {
    case 2: return pack_occupancy<2>(kernel, out);
    case 3: return pack_occupancy<3>(kernel, out);
    default: return cudaErrorInvalidValue;
  }
}

namespace {
template <class Kernel>
cudaError_t versions_of(Kernel kernel, int32_t* binary, int32_t* ptx) {
  cudaFuncAttributes a{};
  const cudaError_t e = cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(kernel));
  if (e == cudaSuccess) *binary = a.binaryVersion, *ptx = a.ptxVersion;
  return e;
}
}  // namespace

// The launches that have no IMD_PERF_KERNEL_* id: the K1 kernels and the instantiations for more
// than kK0SmallModes modes (tests/check_kernel_sites.py keeps this list complete).
cudaError_t extra_kernel_versions(int32_t index, int32_t* binary, int32_t* ptx) {
  switch (index) {
    case 0: return versions_of(k0_vec<kK0MaxModes>, binary, ptx);
    case 1: return versions_of(kf_scale_pack<kK0MaxModes, 16, 2>, binary, ptx);
    case 2: return versions_of(kf_scale_pack<kK0MaxModes, 16, 3>, binary, ptx);
    case 3: return versions_of(kf_scale_pack<kK0MaxModes, 4, 2>, binary, ptx);
    case 4: return versions_of(kf_scale_pack<kK0MaxModes, 4, 3>, binary, ptx);
    case 5: return versions_of(kp_prep<kK0MaxModes>, binary, ptx);
    case 6: return versions_of(k1_tiled<1024>, binary, ptx);
    case 7: return versions_of(k1_tiled<kThreads>, binary, ptx);
    case 8: return versions_of(k1_scale<512>, binary, ptx);
    case 9: return versions_of(k1_scale<1024>, binary, ptx);
    case 10: return versions_of(k1_scale<kThreads>, binary, ptx);
    default: return cudaErrorInvalidValue;
  }
}

cudaError_t kernel_versions(int32_t kernel, int32_t w, int32_t* binary, int32_t* ptx) {
  int32_t row[4];
  g_last_binary = g_last_ptx = 0;
  const cudaError_t e = kernel_occupancy_w(kernel, w, row);
  if (e != cudaSuccess) return e;
  *binary = g_last_binary, *ptx = g_last_ptx;
  return cudaSuccess;
}

cudaError_t kernel_occupancy(int32_t kernel, int32_t* out) {
  // v1.16 §3.4.1: a pack kernel reports its most constrained W instantiation (fewest CTAs per
  // SM, then most registers; the first such W of 3, 2). On sm_89 the CTAs per SM are equal for
  // both W (contract_api checks it) and W = 3 has the most registers: the v1.14 row.
  if (is_pack_kernel(kernel)) {
    int32_t best[4] = {0, 0, 0, 0};
    for (int w = 3; w >= 2; --w) {
      int32_t r[4];
      const cudaError_t e = kernel_occupancy_w(kernel, w, r);
      if (e != cudaSuccess) return e;
      if (w == 3 || r[0] < best[0] || (r[0] == best[0] && r[1] > best[1]))
        for (int i = 0; i < 4; ++i) best[i] = r[i];
    }
    for (int i = 0; i < 4; ++i) out[i] = best[i];
    return cudaSuccess;
  }
  switch (kernel) {
    case IMD_PERF_KERNEL_K0: return occupancy_of(k0_vec<kK0SmallModes>, kThreads, out);
    case IMD_PERF_KERNEL_ZERO_ROWS: return occupancy_of(k23_zero_rows, kThreads, out);
    case IMD_PERF_KERNEL_DIRECT: return occupancy_of(k_direct, kThreads, out);
    case IMD_PERF_KERNEL_BETA: return occupancy_of(k5_beta_only, kThreads, out);
    case IMD_PERF_KERNEL_PACK_PREP: return occupancy_of(kp_prep<kK0SmallModes>, kThreads, out);
    default:
      if (kernel >= IMD_PERF_KERNEL_K5 && kernel < IMD_PERF_KERNEL_COUNT) {
        const int i = kernel - IMD_PERF_KERNEL_K5;
        return k5_occupancy(imd::constants::kNMin + i / 4, i % 4, out);
      }
      return cudaErrorInvalidValue;  // K4: gemm_k4 (api.cpp imd_prof_kernel_occupancy)
  }
}

}  // namespace imd::detail
