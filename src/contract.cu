// K0-K5 and the DIRECT kernel (DESIGN §3, §4). One stream, in order:
//   K0 vec    cudaMemcpyAsync: the plan's pinned scatter vectors -> ws.vec
//   K1 scale  A rows, B cols: non-finite flags and shifts sigma / tau (int16)
//   per residue group G:  K2 / K3 pack A, B -> int8 residues;  K4 GEMMs (gemm_k4.cu)
//   K5 crt    per (lambda, i, j): flagged ? native fma dot : exact CRT, one rounding; alpha/beta
// alpha = 0 or k = 0: only K0 and K5 in beta-only mode. DIRECT path (mode DIRECT or K-empty):
// K0 + the native kernel. Every operand access goes through the int64 additive scatter
// vectors off = v_bat[lambda] + v_row[i] + v_k[h], so arbitrary strides work.
//
// K2/K3 gather through a shared-memory tile, coalesced along the operand's smallest-|stride|
// mode, and write K-contiguous 16-byte vectors. K5 maps consecutive threads to consecutive
// j-bar, which (after the §5 step 8 swap) runs along C's fastest label when that label is
// in I or J: C writes, W reads and the C scatter-vector reads are then coalesced.
// The kernels are written for exactness first. TODO(M4): pack tile shapes and the
// per-element modulo, single-pass K1 (its reads are not tiled yet), a register-resident
// CRT, wider K5 stores, a K5 mapping for C's fastest label in L.
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
// all of k-bar; thread tid always works on row dl = tid mod th, so a warp reads th rows
// side by side (coalesced along the pivot) and 32 / th consecutive k-bar. Same two passes
// and the same exact integers as k1_scale: gamma (and the flag) per row, then S per row;
// the per-thread partial results are combined in shared memory (max, or, u128 sum: exact
// and order-independent).
__global__ void __launch_bounds__(kThreads) k1_tiled(ScaleArgs a, int64_t radix, int lg_th) {
  __shared__ int s_max[kThreads];
  __shared__ int s_flag[kThreads];
  __shared__ u128 s_sum[kThreads];
  __shared__ int64_t s_base[kTileRowsMax];
  __shared__ bool s_valid[kTileRowsMax];
  __shared__ int s_gamma[kTileRowsMax];
  __shared__ int s_rflag[kTileRowsMax];
  const int tid = threadIdx.x, th = 1 << lg_th, dl = tid & (th - 1);
  const int h_first = tid >> lg_th, h_step = kThreads >> lg_th, per_row = kThreads >> lg_th;
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
// the second pass.
int sm_count() {
  int dev = 0, sms = 128;
  if (cudaGetDevice(&dev) == cudaSuccess)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return sms;
}

unsigned k1_blocks(int64_t work_items, double bytes_per_block) {
  const int sms = sm_count();
  const double fit = 40.0 * (1 << 20) / std::max(1.0, bytes_per_block);
  const int64_t cap = std::max<int64_t>(sms, static_cast<int64_t>(fit));
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
struct PackEmitter {
  uint32_t lo[kVec], hi[kVec];  // mu = hi 2^32 + lo < 2^54
  int eps[kVec];
  uint32_t neg = 0, has_eps = 0;  // bit e: a'_e < 0 / eps_e > 0

  __device__ __forceinline__ void load(const uint64_t* bits, int sigma, bool trunc) {
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      uint64_t mu;
      quantize(bits[e], sigma, trunc, &mu, &eps[e]);  // non-finite / zero -> mu = 0, eps = 0
      lo[e] = static_cast<uint32_t>(mu);
      hi[e] = static_cast<uint32_t>(mu >> 32);
      neg |= static_cast<uint32_t>((bits[e] & kSignBit) != 0) << e;
      has_eps |= static_cast<uint32_t>(eps[e] != 0) << e;
    }
  }
  __device__ __forceinline__ uint4 emit(int l) const {
    const uint32_t p = c_tables.moduli[l], half = (p + 1) / 2;
    const uint64_t magic = c_tables.epi_magic[l];
    const uint32_t wlo = imd::device::c_byte_weights.lo[l], whi = imd::device::c_byte_weights.hi[l];
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      uint32_t r = imd::device::mod_p_bytes(lo[e], hi[e], wlo, whi, p, magic);
      if ((has_eps >> e) & 1u)
        r = imd::device::mod_p_small(r * c_tables.pow2_mod[l][eps[e]], p, magic);
      if (((neg >> e) & 1u) && r != 0) r = p - r;
      const uint32_t s = r >= half ? r - p : r;  // SMod as an int8 bit pattern (low byte)
      w[e / 4] |= (s & 255u) << (8 * (e % 4));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
};

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
      for (int s = 0; s < a.gsize; ++s)
        *reinterpret_cast<uint4*>(dst + s * plane) = em.emit(a.g0 + s);
    }
    __syncthreads();  // shared arrays are reused by the next tile
  }
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

// K5 core: the D values of the 4 consecutive j-bar j0..j0+3 of row (lambda, i); j0 is a
// multiple of 4, so each plane's 4 bytes are one aligned u32 load (n_hat is a multiple of
// 16). Elements with j >= n are not computed (D[e] unspecified).
template <int L>
__device__ __forceinline__ void k5_quad(const OutArgs& a, int64_t lam, int64_t i, int64_t j0,
                                        double (&D)[4]) {
  const int64_t plane = a.b * a.m * a.n_hat, row = lam * a.m + i;
  const uint8_t* W = a.W + row * a.n_hat + j0;
  uint32_t x[4][L];
#pragma unroll
  for (int e = 0; e < 4; ++e)
#pragma unroll
    for (int w = 0; w < L; ++w) x[e][w] = 0;
  for (int l = 0; l < a.N; ++l) {
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t*>(W + l * plane));
#pragma unroll
    for (int e = 0; e < 4; ++e) crt_step<L>(x[e], (wv >> (8 * e)) & 255u, l, a.N);
  }
  const bool fa = a.fA[row] != 0;
  const int sa = a.sA[row];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int64_t j = j0 + e;
    if (j >= a.n) break;
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

// K5, j-bar fastest (C's fastest label is in J): thread = 4 consecutive j-bar. W reads of a
// warp are 128 contiguous bytes per plane; C writes run along C's fastest label.
template <int L>
__global__ void __launch_bounds__(kThreads) k5_crt(OutArgs a) {
  const int64_t nq = (a.n + 3) / 4, total = a.b * a.m * nq;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t jq = idx % nq, t = idx / nq, i = t % a.m, lam = t / a.m;
    double D[4] = {0, 0, 0, 0};
    k5_quad<L>(a, lam, i, 4 * jq, D);
    const int64_t base = a.c_bat[lam] + a.c_row[i];
#pragma unroll
    for (int e = 0; e < 4; ++e)
      if (4 * jq + e < a.n) store_out(a, base + a.c_col[4 * jq + e], D[e]);
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
        k5_quad<L>(a, lam, i, j0, D);
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

template <int L>
void launch_k5(const ExecPlan& p, const OutArgs& out, cudaStream_t stream) {
  if (p.k5_batch_fast) {
    int lg_tl = 1;
    while (lg_tl < 5 && (int64_t{1} << lg_tl) < p.b) ++lg_tl;
    const int64_t tl = int64_t{1} << lg_tl, tj = 1024 / tl;
    const int64_t tiles = p.m * ((p.b + tl - 1) / tl) * ((p.n + tj - 1) / tj);
    k5_crt_bfast<L><<<blocks_for(tiles, 1), kThreads, 0, stream>>>(out, lg_tl);
  } else {
    k5_crt<L><<<blocks_for(p.b * p.m * ((p.n + 3) / 4)), kThreads, 0, stream>>>(out);
  }
}

template <class T>
T* region(void* ws, const ExecPlan& p, Region r) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(ws) +
                              p.layout.offset[static_cast<int>(r)]);
}

}  // namespace

imd_status_t enqueue_contract(const ExecPlan& p, double alpha, const double* A, const double* B,
                              double beta, double* C, void* ws, cudaStream_t stream) {
  if (p.path == Path::kNoop) return IMD_STATUS_SUCCESS;
  // K0: scatter vectors -> ws.vec (pinned source: truly asynchronous)
  int64_t* vec = region<int64_t>(ws, p, Region::vec);
  IMD_PROF(stream, "begin");
  if (cudaMemcpyAsync(vec, p.vec_host, static_cast<size_t>(p.vec_count) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    return IMD_STATUS_CUDA_ERROR;
  IMD_PROF(stream, "K0 vec");
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

  // K1
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
      const int lg_th = pack_tile_lg_rows(true, D);
      const int64_t tiles = radix * ((D + (int64_t{1} << lg_th) - 1) >> lg_th);
      k1_tiled<<<k1_blocks(tiles, 8.0 * double(p.k) * double(int64_t{1} << lg_th)), kThreads, 0,
                 stream>>>(sa, radix, lg_th);
    } else {
      const unsigned blocks = k1_blocks(rows, 8.0 * double(p.k));
      if (int64_t{blocks} < 2 * int64_t{sm_count()})
        k1_scale<1024><<<blocks, 1024, 0, stream>>>(sa);
      else
        k1_scale<kThreads><<<blocks, kThreads, 0, stream>>>(sa);
    }
    IMD_PROF(stream, x == 0 ? "K1 scale A" : "K1 scale B");
  }
  if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;

  // K2 / K3 / K4 per residue group
  for (int g0 = 0; g0 < p.N; g0 += p.g) {
    const int gsize = std::min(p.g, p.N - g0);
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
      k23_pack<<<blocks_for(tiles, 1), kThreads, 0, stream>>>(pa);
      if (pa.rows_pad > pa.op.rows)
        k23_zero_rows<<<blocks_for(gsize * p.b * (pa.rows_pad - pa.op.rows) * p.k_hat / kVec),
                        kThreads, 0, stream>>>(pa.R, pa.op.rows, pa.rows_pad, p.b, p.k_hat,
                                               gsize);
      IMD_PROF(stream, x == 0 ? "K2 pack A" : "K3 pack B");
    }
    if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
    if (const imd_status_t st = k4_group(RA, RB, W, p.m, p.n_hat, p.k_hat, p.b, g0, gsize, stream);
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
