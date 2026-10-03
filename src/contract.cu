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

namespace imd::detail {
namespace {

using imd::device::c_tables;
using namespace imd::kernels;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
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

__global__ void __launch_bounds__(kThreads) k1_scale(ScaleArgs a) {
  __shared__ int s_max[kWarps];
  __shared__ int s_flag[kWarps];
  __shared__ u128 s_sum[kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int64_t total = a.b * a.op.rows;
  for (int64_t r = blockIdx.x; r < total; r += gridDim.x) {
    const int64_t lam = r / a.op.rows, i = r % a.op.rows;
    const double* X = a.op.X + (a.op.v_bat[lam] + a.op.v_row[i]);
    // pass 1: flag and gamma = max ilogb over finite nonzero entries
    int gmax = INT_MIN, flag = 0;
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
    for (int w = 0; w < kWarps; ++w) {
      gmax = max(gmax, s_max[w]);
      flag |= s_flag[w];
    }
    // pass 2: S = sum ceil(m_h^2 2^(2(e_h - gamma) + F)), exact (order-independent)
    u128 S = 0;
    if (gmax != INT_MIN) {
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
        for (int w = 0; w < kWarps; ++w) tot += s_sum[w];
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
// rho = r + rows * lambda (index_map.hpp pack_pivot); a tile is kTileRows rows
// rho = lo + radix * d, d in [d0, d0 + kTileRows), by kTileK consecutive k-bar.
//   1. row info: the gather base v_bat[lambda] + v_row[r], sigma, the output row; v_k[h];
//   2. read: raw binary64 bits into shared memory. A warp reads along d (along_rows: X's
//      smallest-|stride| mode is the row / batch mode at `radix`) or along k-bar, so global
//      reads are coalesced along that mode whichever axis of X is fastest;
//   3. write: each thread converts 16 consecutive k-bar of one row and stores them as one
//      16-byte vector per modulus of the group, K-contiguous; h in [k, k_hat) is zero.
// The tile is generic in what it emits: `PackEmitter` converts 16 values for one modulus
// (an Ozaki-I slice mode would emit slices here and reuse the gather unchanged).
// B rows n..n_hat-1 (zero) are written by k23_zero_rows.

constexpr int kTileRows = 64;  // d per tile
constexpr int kTileK = 64;     // k-bar per tile; kTileRows * kTileK / 16 == kThreads
constexpr int kVec = 16;       // int8 per 16-byte store
static_assert(kTileRows * (kTileK / kVec) == kThreads, "one 16-byte chunk per thread");

struct PackArgs {
  Operand op;
  int64_t rows_pad, b, k, k_hat;  // rows_pad = m (A) or n_hat (B)
  const int16_t* sigma;           // [lambda][rows] = [rho]
  int g0, gsize;
  bool trunc;
  int8_t* R;          // [s][lambda][rows_pad][k_hat]
  int64_t radix;      // rho = lo + radix * d
  bool along_rows;    // read phase: warp lanes along d (else along k-bar)
};

// §3.3 + §3.4 for one row's 16 consecutive k-bar: quantize once, then residues per modulus.
struct PackEmitter {
  uint64_t mu[kVec];
  int eps[kVec];
  bool neg[kVec];

  __device__ __forceinline__ void load(const uint64_t* bits, int sigma, bool trunc) {
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      quantize(bits[e], sigma, trunc, &mu[e], &eps[e]);  // non-finite / zero -> mu = 0
      neg[e] = (bits[e] & kSignBit) != 0;
    }
  }
  __device__ __forceinline__ uint4 emit(int l) const {
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const int8_t r = mu[e] == 0 ? int8_t{0} : residue(mu[e], neg[e], eps[e], l);
      w[e / 4] |= static_cast<uint32_t>(static_cast<uint8_t>(r)) << (8 * (e % 4));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
};

__global__ void __launch_bounds__(kThreads) k23_pack(PackArgs a) {
  __shared__ uint64_t tile[kTileRows][kTileK + 1];  // +1: conflict-free column access
  __shared__ int64_t s_base[kTileRows];             // v_bat[lambda] + v_row[r]
  __shared__ int64_t s_out[kTileRows];              // (lambda rows_pad + r) k_hat
  __shared__ int s_sigma[kTileRows];
  __shared__ bool s_valid[kTileRows];
  __shared__ int64_t s_vk[kTileK];
  const int tid = threadIdx.x;
  const int64_t rows = a.op.rows, D = a.b * rows / a.radix;
  const int64_t n_d = (D + kTileRows - 1) / kTileRows, n_h = (a.k_hat + kTileK - 1) / kTileK;
  const int64_t tiles = a.radix * n_d * n_h, plane = a.b * a.rows_pad * a.k_hat;
  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int64_t ht = t % n_h, rest = t / n_h, dt = rest % n_d, lo = rest / n_d;
    const int64_t h0 = ht * kTileK, d0 = dt * kTileRows;
    // 1. row and column info
    if (tid < kTileRows) {
      const int64_t d = d0 + tid;
      const bool valid = d < D;
      s_valid[tid] = valid;
      if (valid) {
        const int64_t rho = lo + a.radix * d, lam = rho / rows, r = rho % rows;
        s_base[tid] = a.op.v_bat[lam] + a.op.v_row[r];
        s_out[tid] = (lam * a.rows_pad + r) * a.k_hat;
        s_sigma[tid] = a.sigma[rho];
      }
    } else if (tid < kTileRows + kTileK) {
      const int64_t h = h0 + (tid - kTileRows);
      s_vk[tid - kTileRows] = h < a.k ? a.op.v_k[h] : 0;
    }
    __syncthreads();
    // 2. coalesced read into shared memory (zero outside the operand)
    const int64_t h_end = a.k - h0;  // valid k-bar in this tile: hl < h_end
    if (a.along_rows) {
      const int dl = tid % kTileRows;
      for (int hl = tid / kTileRows; hl < kTileK; hl += kThreads / kTileRows)
        tile[dl][hl] = s_valid[dl] && hl < h_end ? to_bits(a.op.X[s_base[dl] + s_vk[hl]]) : 0;
    } else {
      const int hl = tid % kTileK;
      for (int dl = tid / kTileK; dl < kTileRows; dl += kThreads / kTileK)
        tile[dl][hl] = s_valid[dl] && hl < h_end ? to_bits(a.op.X[s_base[dl] + s_vk[hl]]) : 0;
    }
    __syncthreads();
    // 3. convert and store 16 K-contiguous int8 per modulus
    const int dl = tid / (kTileK / kVec), hl = (tid % (kTileK / kVec)) * kVec;
    if (s_valid[dl] && h0 + hl < a.k_hat) {  // k_hat is a multiple of 16
      uint64_t bits[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) bits[e] = tile[dl][hl + e];
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

template <int L>
__global__ void __launch_bounds__(kThreads) k5_crt(OutArgs a) {
  const int64_t total = a.b * a.m * a.n, plane = a.b * a.m * a.n_hat;
  for (int64_t idx = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; idx < total;
       idx += int64_t{gridDim.x} * blockDim.x) {
    const int64_t j = idx % a.n, t = idx / a.n, i = t % a.m, lam = t / a.m;
    double D;
    if (a.fA[lam * a.m + i] != 0 || a.fB[lam * a.n + j] != 0) {
      D = native_dot(a, lam, i, j);
    } else {
      uint32_t mag[L];
      bool neg;
      crt<L>(a.W + ((lam * a.m + i) * a.n_hat + j), plane, a.N, mag, &neg);
      D = round_scaled<L>(mag, neg, -(int{a.sA[lam * a.m + i]} + int{a.sB[lam * a.n + j]}));
    }
    store_out(a, a.c_bat[lam] + a.c_row[i] + a.c_col[j], D);
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
  if (cudaMemcpyAsync(vec, p.vec_host, static_cast<size_t>(p.vec_count) * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    return IMD_STATUS_CUDA_ERROR;
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
    return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
  }
  if (p.path == Path::kDirect) {
    k_direct<<<out_blocks, kThreads, 0, stream>>>(out);
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
  sa.op = opA, sa.sigma = sA, sa.flag = fA;
  k1_scale<<<blocks_for(p.b * p.m, 1), kThreads, 0, stream>>>(sa);
  sa.op = opB, sa.sigma = sB, sa.flag = fB;
  k1_scale<<<blocks_for(p.b * p.n, 1), kThreads, 0, stream>>>(sa);
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
      const int64_t tiles =
          pa.radix * ((D + kTileRows - 1) / kTileRows) * ((p.k_hat + kTileK - 1) / kTileK);
      k23_pack<<<blocks_for(tiles, 1), kThreads, 0, stream>>>(pa);
      if (pa.rows_pad > pa.op.rows)
        k23_zero_rows<<<blocks_for(gsize * p.b * (pa.rows_pad - pa.op.rows) * p.k_hat / kVec),
                        kThreads, 0, stream>>>(pa.R, pa.op.rows, pa.rows_pad, p.b, p.k_hat,
                                               gsize);
    }
    if (cudaPeekAtLastError() != cudaSuccess) return IMD_STATUS_CUDA_ERROR;
    if (const imd_status_t st = k4_group(RA, RB, W, p.m, p.n_hat, p.k_hat, p.b, g0, gsize, stream);
        st != IMD_STATUS_SUCCESS)
      return st;
  }

  // K5
  out.sA = sA, out.sB = sB, out.fA = fA, out.fB = fB, out.W = W;
  switch (imd::constants::kTables.L32[p.N]) {
    case 1: k5_crt<1><<<out_blocks, kThreads, 0, stream>>>(out); break;
    case 2: k5_crt<2><<<out_blocks, kThreads, 0, stream>>>(out); break;
    case 3: k5_crt<3><<<out_blocks, kThreads, 0, stream>>>(out); break;
    case 4: k5_crt<4><<<out_blocks, kThreads, 0, stream>>>(out); break;
    case 5: k5_crt<5><<<out_blocks, kThreads, 0, stream>>>(out); break;
    default: return IMD_STATUS_CUDA_ERROR;  // unreachable: L32 in 1..5 for N in 2..20
  }
  return cudaPeekAtLastError() == cudaSuccess ? IMD_STATUS_SUCCESS : IMD_STATUS_CUDA_ERROR;
}

}  // namespace imd::detail
