// K0-K5 and the DIRECT kernel (DESIGN §3, §4). One stream, in order:
//   K0 vec    kernel: the scatter vectors -> ws.vec from the plan's fused-mode descriptor
//             (a kernel argument); skipped with IMD_VEC_RESIDENT (the plan's device copy)
//   K1 scale  A rows, B cols: non-finite flags and shifts sigma / tau (int16)
//   per residue group G:  K2 / K3 pack A, B -> int8 residues;  K4 GEMMs (gemm_k4.cu)
//   K5 crt    per (lambda, i, j): flagged ? native fma dot : exact CRT, one rounding; alpha/beta
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
#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include "device/constants.cuh"
#include "kernels/numerics.cuh"
#include "plan.hpp"
#include "profile.hpp"

namespace imd::detail {
namespace {

using imd::device::c_tables;
using namespace imd::kernels;

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = int64_t{1} << 20;  // grid-stride loops beyond this

unsigned blocks_for(int64_t work, int per_block = kThreads) {
  const int64_t b = (work + per_block - 1) / per_block;
  return static_cast<unsigned>(std::max<int64_t>(1, std::min(b, kMaxBlocks)));
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
    // pass 2: S = sum ceil(m_h^2 2^(2(e_h - gamma) + F)), exact (order-independent)
    u128 S = 0;
    if (gmax != INT_MIN) {
#pragma unroll 4
      for (int64_t h = threadIdx.x; h < a.k; h += blockDim.x) {
        const uint64_t bits = to_bits(X[a.op.v_k[h]]);
        if (finite_nonzero(bits)) S += square_term(bits, gmax);
      }
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
// The tile is generic in what it emits: `PackEmitter` converts 16 values for one modulus
// (an Ozaki-I slice mode would emit slices here and reuse the gather unchanged).
// B rows n..n_hat-1 (zero) are written by k23_zero_rows.

constexpr int kTileElems = 4096;  // th * tk; kTileElems / 16 == kThreads
constexpr int kTileRowsMax = 64, kTileRowsMin = 8;
constexpr int kTileKMax = kTileElems / kTileRowsMin;  // 512
constexpr int kVec = 16;          // int8 per 16-byte store
constexpr int kTileSmem = kTileElems + kTileElems / kVec + kTileRowsMax;  // padded words
static_assert(kTileElems / kVec == kThreads, "one 16-byte chunk per thread");

// log2 of the pack tile height (see above).
inline int pack_tile_lg_rows(bool along_rows, int64_t D) {
  if (!along_rows) return 6;
  int lg = 3;
  while (lg < 6 && (int64_t{1} << lg) < D) ++lg;
  return lg;
}

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
        if (finite_nonzero(bits)) S += square_term(bits, gmax);
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
unsigned k1_blocks(int64_t work_items, double bytes_per_block, int sms) {
  const double fit = 40.0 * (1 << 20) / std::max(1.0, bytes_per_block);
  int64_t cap = std::max<int64_t>(sms, static_cast<int64_t>(fit));
  if (work_items > cap && cap < 2 * int64_t{sms}) cap = cap / sms * sms;
  return blocks_for(std::min(work_items, cap), 1);
}

struct PackArgs {
  Operand op;
  int64_t rows_pad, b, k, k_hat;  // rows_pad = m (A) or n_hat (B)
  const int16_t* sigma;           // [lambda][rows] = [rho]
  int g0, gsize;
  bool trunc;
  int8_t* R;          // [s][lambda][rows_pad][k_hat]
  int64_t radix;      // rho = lo + radix * d
  bool along_rows;    // read phase: warp lanes along d (else along k-bar)
  int lg_th;          // tile height th = 2^lg_th rows, tk = kTileElems / th k-bar
};

// §3.3 + §3.4 for one row's 16 consecutive k-bar: quantize once, then residues per modulus.
// The residue of |a'| = mu 2^eps is ((mu mod p) (2^eps mod p)) mod p; mu mod p by byte
// weights and dp4a (device/modp.hpp mod_p_bytes), the rare eps > 0 factor by the pow2_mod
// table, then the sign and SMod. Same congruence class as §3.4 `residue`, so the same int8.
template <int E>  // E consecutive k-bar of one row: 16 (one uint4 per modulus) or 4 (one u32)
struct PackEmitterT {
  static_assert(E == 16 || E == 4, "16 or 4 values per emitter");
  uint32_t lo[E], hi[E];  // mu = hi 2^32 + lo < 2^54
  int eps[E];
  uint32_t neg = 0, has_eps = 0;  // bit e: a'_e < 0 / eps_e > 0

  __device__ __forceinline__ void load(const uint64_t* bits, int sigma, bool trunc) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      uint64_t mu;
      quantize(bits[e], sigma, trunc, &mu, &eps[e]);  // non-finite / zero -> mu = 0, eps = 0
      lo[e] = static_cast<uint32_t>(mu);
      hi[e] = static_cast<uint32_t>(mu >> 32);
      neg |= static_cast<uint32_t>((bits[e] & kSignBit) != 0) << e;
      has_eps |= static_cast<uint32_t>(eps[e] != 0) << e;
    }
  }
  // The E int8 residues for modulus l, as E / 4 little-endian u32 words.
  __device__ __forceinline__ void emit(int l, uint32_t (&w)[E / 4]) const {
    const uint32_t p = c_tables.moduli[l], half = (p + 1) / 2;
    const uint64_t magic = c_tables.epi_magic[l];
    const uint32_t wlo = imd::device::c_byte_weights.lo[l], whi = imd::device::c_byte_weights.hi[l];
#pragma unroll
    for (int q = 0; q < E / 4; ++q) w[q] = 0;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      uint32_t r = imd::device::mod_p_bytes(lo[e], hi[e], wlo, whi, p, magic);
      if ((has_eps >> e) & 1u)
        r = imd::device::mod_p_small(r * c_tables.pow2_mod[l][eps[e]], p, magic);
      if (((neg >> e) & 1u) && r != 0) r = p - r;
      const uint32_t s = r >= half ? r - p : r;  // SMod as an int8 bit pattern (low byte)
      w[e / 4] |= (s & 255u) << (8 * (e % 4));
    }
  }
  // Store the residues for modulus l at dst (aligned to E bytes).
  __device__ __forceinline__ void store(int l, int8_t* dst) const {
    uint32_t w[E / 4];
    emit(l, w);
    if constexpr (E == 16)
      *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    else
      *reinterpret_cast<uint32_t*>(dst) = w[0];
  }
};
using PackEmitter = PackEmitterT<16>;

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
    if (s_valid[dl] && h0 + hl < a.k_hat) {  // k_hat is a multiple of 16
      uint64_t bits[kVec];
      const uint64_t* src = tile + dl * pitch + hl + hl / kVec;
#pragma unroll
      for (int e = 0; e < kVec; ++e) bits[e] = src[e];
      PackEmitter em;
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
inline unsigned pack_split(int64_t tiles, int gsize, int sms) {
  const int64_t want = 2 * int64_t{sms};
  if (tiles >= want) return 1;
  return static_cast<unsigned>(std::min<int64_t>(gsize, (want + tiles - 1) / tiles));
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

// K5 core: the D values of the E consecutive j-bar j0..j0+E-1 of row (lambda, i), E = 4 or
// 1. For E = 4, j0 is a multiple of 4, so each plane's 4 bytes are one aligned u32 load
// (n_hat is a multiple of 16). Elements with j >= n are not computed (D[e] unspecified).
// Every index into x / D / wv is static after unrolling (no local memory).
template <int L, int E>
__device__ __forceinline__ void k5_elems(const OutArgs& a, int64_t lam, int64_t i, int64_t j0,
                                         double (&D)[E]) {
  static_assert(E == 1 || E == 4, "K5 takes 1 or 4 consecutive j-bar per thread");
  const int64_t plane = a.b * a.m * a.n_hat, row = lam * a.m + i;
  const uint8_t* W = a.W + row * a.n_hat + j0;
  uint32_t x[E][L];
#pragma unroll
  for (int e = 0; e < E; ++e)
#pragma unroll
    for (int w = 0; w < L; ++w) x[e][w] = 0;
  // All N plane loads are issued before the CRT chain (N <= kMaxN for this L), so their
  // latencies overlap instead of serializing with the arithmetic.
  constexpr int kMaxN = k5_max_n(L);
  uint32_t wv[kMaxN];
#pragma unroll
  for (int l = 0; l < kMaxN; ++l) {
    if constexpr (E == 4)
      wv[l] = l < a.N ? __ldg(reinterpret_cast<const uint32_t*>(W + l * plane)) : 0u;
    else
      wv[l] = l < a.N ? uint32_t{__ldg(W + l * plane)} : 0u;
  }
#pragma unroll
  for (int l = 0; l < kMaxN; ++l) {
    if (l < a.N) {
#pragma unroll
      for (int e = 0; e < E; ++e) crt_step<L>(x[e], (wv[l] >> (8 * e)) & 255u, l, a.N);
    }
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
        crt_finish<L>(x[e], a.N, mag, &neg);
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
// Shared slot of result s (= 4 item + e of the warp's 32 items): s + s / 16, which makes
// both the writes (stride 4 per lane) and the reads (stride 1) bank-conflict free.
constexpr int kK5WarpSlots = 128 + 8;
template <int L, int E>
__global__ void __launch_bounds__(kThreads) k5_crt(OutArgs a) {
  __shared__ double sD[E == 4 ? (kThreads / 32) * kK5WarpSlots : 1];
  const int64_t ng = (a.n + E - 1) / E, total = a.b * a.m * ng;
  const int lane = threadIdx.x & 31;
  double* wD = sD + (E == 4 ? (threadIdx.x / 32) * kK5WarpSlots : 0);
  // Block-uniform loop: every lane of a warp takes part in each iteration's shuffles.
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
      k5_elems<L, E>(a, lam, i, E * jg, D);
      cbase = a.c_bat[lam] + a.c_row[i];
    }
    if constexpr (E == 1) {
      if (live) store_out(a, cbase + a.c_col[jg], D[0]);
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
template <int L>
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
        k5_elems<L, 4>(a, lam, i, j0, D);
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

// K5 j-bar fastest: one element per thread while b m n <= this many per SM (a launch choice;
// the results do not depend on it). The test hook imd_test_set_k5_one_elem changes it.
constexpr int64_t kK5OneElemPerSm = 2048;

template <int L>
void launch_k5(const ExecPlan& p, const OutArgs& out, cudaStream_t stream) {
  if (p.k5_batch_fast) {
    int lg_tl = 1;
    while (lg_tl < 5 && (int64_t{1} << lg_tl) < p.b) ++lg_tl;
    const int64_t tl = int64_t{1} << lg_tl, tj = 1024 / tl;
    const int64_t tiles = p.m * ((p.b + tl - 1) / tl) * ((p.n + tj - 1) / tj);
    k5_crt_bfast<L><<<blocks_for(tiles, 1), kThreads, 0, stream>>>(out, lg_tl);
  } else if (p.b * p.m * p.n <= k5_one_elem_per_sm() * p.sms) {
    // Few elements: one per thread (4x the threads of the quad mapping, a shorter chain each).
    k5_crt<L, 1><<<blocks_for(p.b * p.m * p.n), kThreads, 0, stream>>>(out);
  } else {
    k5_crt<L, 4><<<blocks_for(p.b * p.m * ((p.n + 3) / 4)), kThreads, 0, stream>>>(out);
  }
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
// Chosen at plan time when k_hat <= 512 (analysis.hpp fused_path). A block of 256 threads
// takes a tile of th rows by tk = pow2 >= k_hat columns (tk >= 16), th tk = 256 E: whole rows.
// Each thread converts E consecutive k-bar of one row (E = 16, or E = 4 for small operands
// with tk <= 128: 4x the blocks and a 4x shorter chain per thread), so a row's tk / E <= 32
// threads are consecutive lanes of one warp. A block (tile t, part y):
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

template <int Cap, int E>
__global__ void __launch_bounds__(kThreads) kf_scale_pack(const __grid_constant__ FusedArgs<Cap> a) {
  constexpr int kLgE = E == 16 ? 4 : 2, kTE = kThreads * E, kLgTE = 8 + kLgE;
  constexpr int kRowsMax = kTE / 16;  // th for tk = 16
  __shared__ uint64_t tile[kTE + kTE / kVec + kRowsMax];
  __shared__ int64_t s_base[kRowsMax];
  __shared__ int64_t s_out[kRowsMax];
  __shared__ bool s_valid[kRowsMax];
  __shared__ int64_t s_vk[kTileKMax];
  const int tid = threadIdx.x;
  const int lg_tk = a.lg_tk, tk = 1 << lg_tk, lg_th = kLgTE - lg_tk, th = 1 << lg_th;
  const int pitch = tk + tk / kVec + 1, G = tk / E;  // G lanes per row, G | 32
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
        s_out[tid] = (lam * a.rows_pad[x] + r) * a.k_hat;
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
    for (int o = G / 2; o > 0; o /= 2) {
      gmax = max(gmax, __shfl_xor_sync(0xffffffffu, gmax, o));
      fl |= __shfl_xor_sync(0xffffffffu, fl, o);
    }
    u128 S = 0;
    if (gmax != INT_MIN) {
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (finite_nonzero(bits[e])) S += square_term(bits[e], gmax);
    }
    for (int o = G / 2; o > 0; o /= 2) S += shfl_xor_u128(S, o);
    // sigma by the row's first lane (the 256-bit compare once per row), then broadcast.
    int sigma = 0;
    if ((tid & (G - 1)) == 0 && gmax != INT_MIN)
      sigma = largest_u(S, a.lhs_shift, a.rhs, a.rhs_bits) - gmax;
    sigma = __shfl_sync(0xffffffffu, sigma, (tid & 31) & ~(G - 1));
    const bool valid = s_valid[dl];
    // 4. part 0 stores sigma and the flag (range certified, cert_shift_range)
    if (valid && blockIdx.y == 0 && (tid & (G - 1)) == 0) {
      const int64_t rho = lo + radix * (d0 + dl);
      a.sigma[x][rho] = static_cast<int16_t>(sigma);
      a.flag[x][rho] = static_cast<uint8_t>(fl);
    }
    // 5. residues of this part's slots (K padding h in [k, k_hat) is zero in the tile)
    if (valid && hl < a.k_hat) {
      PackEmitterT<E> em;
      em.load(bits, sigma, a.trunc != 0);
      const int64_t plane = a.b * a.rows_pad[x] * a.k_hat;
      int8_t* dst = a.R[x] + s_out[dl] + hl;
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

// KF with E = 4 when one row fits 32 lanes of 4 (tk <= 128) and the E = 16 grid would have
// fewer than 2 blocks per SM even with every modulus of the group in its own block row.
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
  int lg_tk = 4;
  while ((int64_t{1} << lg_tk) < p.k_hat) ++lg_tk;  // k_hat <= 512: lg_tk <= 9
  a.lg_tk = lg_tk;
  a.gsize = std::min(p.g, p.N);
  a.trunc = p.trunc ? 1 : 0;
  for (int w = 0; w < 4; ++w) a.rhs[w] = p.rhs[w];
  a.rhs_bits = p.rhs_bits, a.lhs_shift = p.lhs_shift;
  const auto tiles_for = [&](int lg_te) {
    const int64_t th = int64_t{1} << (lg_te - lg_tk);
    int64_t total = 0;
    for (int x = 0; x < 2; ++x) {
      a.rows[x] = x == 0 ? p.m : p.n;
      a.rows_pad[x] = x == 0 ? p.m : p.n_hat;
      a.radix[x] = p.pack_radix[x];
      a.along_rows[x] = p.pack_along_rows[x] ? 1 : 0;
      const int64_t D = p.b * a.rows[x] / a.radix[x];
      a.tiles[x] = a.radix[x] * ((D + th - 1) / th);
      total += a.tiles[x];
    }
    return total;
  };
  int64_t tiles = tiles_for(12);
  // (k5_one_elem_per_sm() == 0, test hook only: no small-problem variants anywhere.)
  const bool small =
      lg_tk <= 7 && tiles * a.gsize < 2 * int64_t{p.sms} && k5_one_elem_per_sm() != 0;
  if (small) {
    tiles = tiles_for(10);
    kf_scale_pack<Cap, 4><<<dim3(blocks_for(tiles, 1), pack_split(tiles, a.gsize, p.sms)),
                            kThreads, 0, stream>>>(a);
  } else {
    kf_scale_pack<Cap, 16><<<dim3(blocks_for(tiles, 1), pack_split(tiles, a.gsize, p.sms)),
                             kThreads, 0, stream>>>(a);
  }
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace

// K1 (both operands; two launches): flags and sigma / tau.
static void launch_k1(const ExecPlan& p, const Operand& opA, const Operand& opB, int16_t* sA,
                      int16_t* sB, uint8_t* fA, uint8_t* fB, cudaStream_t stream) {
  ScaleArgs sa{};
  sa.b = p.b, sa.k = p.k, sa.rhs_bits = p.rhs_bits, sa.lhs_shift = p.lhs_shift;
  for (int w = 0; w < 4; ++w) sa.rhs[w] = p.rhs[w];
  for (int x = 0; x < 2; ++x) {
    sa.op = x == 0 ? opA : opB;
    sa.sigma = x == 0 ? sA : sB;
    sa.flag = x == 0 ? fA : fB;
    const int64_t rows = p.b * sa.op.rows;
    if (p.pack_along_rows[x]) {  // rows are read side by side: tiles as in the pack
      const int64_t radix = p.pack_radix[x], D = rows / radix;
      // 8 rows side by side (64-byte segments, whole sectors) keep the per-block re-read
      // footprint small, so more rows fit in L2 for the second pass.
      const int lg_th = 3;
      const int64_t tiles = radix * ((D + (int64_t{1} << lg_th) - 1) >> lg_th);
      const unsigned blocks =
          k1_blocks(tiles, 8.0 * double(p.k) * double(int64_t{1} << lg_th), p.sms);
      if (int64_t{blocks} < 2 * int64_t{p.sms})
        k1_tiled<1024><<<blocks, 1024, 0, stream>>>(sa, radix, lg_th);
      else
        k1_tiled<kThreads><<<blocks, kThreads, 0, stream>>>(sa, radix, lg_th);
    } else {
      const unsigned blocks = k1_blocks(rows, 8.0 * double(p.k), p.sms);
      if (int64_t{blocks} < 2 * int64_t{p.sms})
        k1_scale<1024><<<blocks, 1024, 0, stream>>>(sa);
      else
        k1_scale<kThreads><<<blocks, kThreads, 0, stream>>>(sa);
    }
    IMD_PROF(stream, x == 0 ? "K1 scale A" : "K1 scale B");
  }
}

int64_t& k5_one_elem_per_sm() {
  static int64_t v = kK5OneElemPerSm;  // only the test hook changes it
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

imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream) {
  if (p.path == Path::kNoop) return IMD_STATUS_SUCCESS;
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
  auto* RA = region<int8_t>(ws, p, Region::RA);
  auto* RB = region<int8_t>(ws, p, Region::RB);
  auto* W = region<uint8_t>(ws, p, Region::W);
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
  } else {  // K1
    launch_k1(p, opA, opB, sA, sB, fA, fB, stream);
    if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
  }

  // K2 / K3 / K4 per residue group
  for (int g0 = 0; g0 < p.N; g0 += p.g) {
    const int gsize = std::min(p.g, p.N - g0);
    if (use_kf && g0 == 0) {  // group 1 was packed by KF
      if (const imd_status_t st = k4_group(*p.k4, RA, RB, W, g0, gsize, stream);
          st != IMD_STATUS_SUCCESS)
        return st;
      continue;
    }
    PackArgs pa{};
    pa.b = p.b, pa.k = p.k, pa.k_hat = p.k_hat, pa.g0 = g0, pa.gsize = gsize, pa.trunc = p.trunc;
    for (int x = 0; x < 2; ++x) {
      pa.op = x == 0 ? opA : opB;
      pa.rows_pad = x == 0 ? p.m : p.n_hat;
      pa.sigma = x == 0 ? sA : sB;
      pa.R = x == 0 ? RA : RB;
      pa.radix = p.pack_radix[x];
      pa.along_rows = p.pack_along_rows[x];
      const int64_t D = p.b * pa.op.rows / pa.radix;
      pa.lg_th = pack_tile_lg_rows(pa.along_rows, D);
      const int64_t th = int64_t{1} << pa.lg_th, tk = kTileElems / th;
      const int64_t tiles = pa.radix * ((D + th - 1) / th) * ((p.k_hat + tk - 1) / tk);
      k23_pack<<<dim3(blocks_for(tiles, 1), pack_split(tiles, gsize, p.sms)), kThreads, 0,
                 stream>>>(pa);
      if (pa.rows_pad > pa.op.rows)
        k23_zero_rows<<<blocks_for(gsize * p.b * (pa.rows_pad - pa.op.rows) * p.k_hat / kVec),
                        kThreads, 0, stream>>>(pa.R, pa.op.rows, pa.rows_pad, p.b, p.k_hat,
                                               gsize);
      IMD_PROF(stream, x == 0 ? "K2 pack A" : "K3 pack B");
    }
    if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
    if (const imd_status_t st = k4_group(*p.k4, RA, RB, W, g0, gsize, stream);
        st != IMD_STATUS_SUCCESS)
      return st;
  }

  // K5
  out.sA = sA, out.sB = sB, out.fA = fA, out.fB = fB, out.W = W;
  switch (imd::constants::kTables.L32[p.N]) {
    case 1: launch_k5<1>(p, out, stream); break;
    case 2: launch_k5<2>(p, out, stream); break;
    case 3: launch_k5<3>(p, out, stream); break;
    case 4: launch_k5<4>(p, out, stream); break;
    case 5: launch_k5<5>(p, out, stream); break;
    default: return IMD_STATUS_CUDA_ERROR;  // unreachable: L32 in 1..5 for N in 2..20
  }
  IMD_PROF(stream, "K5 crt");
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace imd::detail
