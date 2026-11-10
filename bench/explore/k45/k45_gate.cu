// K45 feasibility gate (b) (immerdink-eval analysis/20261110-dataflow/plan.md §6 step 2, tracks/dataflow.md critique W4):
// can ONE warp per SM sub-partition (SMSP), capped at 96 registers, run immerdink's K5 CRT ('k5fin', sm_90 default) fast
// enough to live on its own warpgroup inside a fused K4+K5 kernel? Gate: warp IPC per SMSP >= 0.375 at r_lowk
// (m = n = 8192, k = 1024, N = 14) and >= 0.23 at t_df (m = n = 4096, k = 2048), where the IPC that counts is the
// reference K5's instruction count spread over the gate kernel's cycles (see k45_tab.py).
//
// What runs (N = 14, synthetic residue planes W = N x m x n bytes, W_l = hash mod p_l; sigma/tau in [40, 60]; no native rows):
//   ref      immerdink's k5f_crt_n<14, 4> VERBATIM (k5_extract.cuh, from src/contract.cu at the commit in K5_SOURCE), with
//            today's launch: 256-thread blocks, grid = ceil(m (n / 4) / 256) (host/analysis.hpp k5_launch, E = 4 for these sizes)
//   k45g E   the gate: grid = #SMs blocks of 128 threads (exactly one warp per SMSP; a 120 KiB dynamic shared-memory pad keeps it
//            at one block per SM, checked by %smid), grid-stride over the same items, <= 96 registers (__maxnreg__), the CRT
//            core = immerdink's k5f_elems VERBATIM (E = 1, 2, 4 outputs per thread)
//   k45g Epf the same with a software prefetch: the next item's N plane words are loaded before the current item's CRT
//            (k5f_elems_wv below = k5f_elems with its load loop moved out; otherwise the same statements)
//   k45g E4x2 two items per loop iteration (idx, idx + stride), both computed before either is stored: a remedy probe (2x ILP / loads)
// Validation: every gate output is compared bit for bit with the ref output (whole D); the ref itself against an exact host
// CRT (unsigned __int128) on a sample of outputs. Timing: interleaved repetitions (ref, gates, ref, gates, ...), medians.
// Modes: --mode all (validate + time; default) | ncu (validate only: every kernel launched once per shape, for ncu -k).
// Output: one JSON line per (shape, variant) on stdout and --out.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/numerics.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

#ifndef K45_MAXNREG
#define K45_MAXNREG 96
#endif
#ifndef K45_N
#define K45_N 14
#endif

namespace imd::detail {
namespace {
using imd::device::c_tables;
using namespace imd::kernels;
constexpr int kThreads = 256;   // = host/analysis.hpp kLaunchThreads at the K5 commit (the K5 block size)
#include "k5_extract.cuh"
}  // namespace
}  // namespace imd::detail

using imd::detail::OutArgs;
using imd::detail::kK5WarpSlots;
constexpr int kN = K45_N;
constexpr int kGateThreads = 128;   // one warp per SMSP

// ---- derived from immerdink (NOT verbatim): the load loop of k5f_elems, and k5f_elems without it --------------------------
namespace imd::detail {
namespace {
template <int N, int E>
__device__ __forceinline__ void k5f_load_wv(const OutArgs& a, uint32_t t, uint32_t j0, uint32_t (&wv)[N]) {
  const int64_t plane = a.b * a.m * a.n_hat;
  const uint8_t* W = a.W + int64_t{t} * a.n_hat + j0;
#pragma unroll
  for (int l = 0; l < N; ++l) {
    if constexpr (E == 4) wv[l] = __ldg(reinterpret_cast<const uint32_t*>(W + l * plane));
    else if constexpr (E == 2) wv[l] = __ldg(reinterpret_cast<const uint16_t*>(W + l * plane));
    else wv[l] = __ldg(W + l * plane);
  }
}
// k5f_elems = k5f_load_wv + k5f_crt_wv + k5f_fin_x (the same statements in the same order, split in three).
template <int N, int E>
__device__ __forceinline__ void k5f_crt_wv(const uint32_t (&wv)[N], uint32_t (&x)[E][imd::constants::kTables.L32[N]]) {
  constexpr int L = imd::constants::kTables.L32[N];
#pragma unroll
  for (int e = 0; e < E; ++e) {
#pragma unroll
    for (int w = 0; w < L; ++w) x[e][w] = 0;
    crt_f64_elem<N, L, 0, true>(wv, e, x[e]);
  }
}
template <int N, int E>
__device__ __forceinline__ void k5f_fin_x(const OutArgs& a, uint32_t t, uint32_t lam, uint32_t j0, bool vmeta,
                                          const uint32_t (&x)[E][imd::constants::kTables.L32[N]], double (&D)[E]) {
  constexpr int L = imd::constants::kTables.L32[N];
  const bool fa = a.fA[t] != 0;
  const int sa = a.sA[t];
  const int64_t bo = int64_t{lam} * a.n + j0;
  if constexpr (E == 4) {
    if (vmeta) {
      const uint32_t fb = __ldg(reinterpret_cast<const uint32_t*>(a.fB + bo));
      if (!fa && fb == 0) {
        const uint2 sb = __ldg(reinterpret_cast<const uint2*>(a.sB + bo));
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t w = e < 2 ? sb.x : sb.y;
          const int sbe = static_cast<int16_t>(static_cast<uint16_t>(w >> (16 * (e & 1))));
          D[e] = crt_round_fin<L, N>(x[e], -(sa + sbe));
        }
        return;
      }
    }
  }
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int64_t j = int64_t{j0} + e;
    if (j < a.n) {
      if (fa || a.fB[bo + e] != 0) {
        D[e] = native_dot(a, lam, t - lam * static_cast<uint32_t>(a.m), j);
      } else {
        D[e] = crt_round_fin<L, N>(x[e], -(sa + int{a.sB[bo + e]}));
      }
    }
  }
}
template <int N, int E>
__device__ __forceinline__ void k5f_elems_wv(const OutArgs& a, uint32_t t, uint32_t lam, uint32_t j0, bool vmeta, const uint32_t (&wv)[N],
                                             double (&D)[E]) {
  uint32_t x[E][imd::constants::kTables.L32[N]];
  k5f_crt_wv<N, E>(wv, x);
  k5f_fin_x<N, E>(a, t, lam, j0, vmeta, x, D);
}
}  // namespace
}  // namespace imd::detail

__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r; }

// The stores of one item: k5f_crt_n's store section (E = 4: through the warp's shared slots), unchanged.
template <int E>
__device__ __forceinline__ void gate_store(const OutArgs& a, bool a1, bool b0, int lane, double* wD, uint32_t nn, bool live, uint32_t jg,
                                           int64_t cbase, const double (&D)[E]) {
  using namespace imd::detail;
  if constexpr (E != 4) {
    if (live) {
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (E * jg + e < nn) k5f_store(a, a1, b0, cbase + a.c_col[E * jg + e], D[e]);
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int s = 4 * lane + e;
      wD[s + s / 16] = D[e];
    }
    __syncwarp();
    const uint32_t jg_first = __shfl_sync(0xffffffffu, jg, 0);
    const bool run = live && jg == jg_first + lane;
    if (__all_sync(0xffffffffu, run)) {
      const uint32_t j00 = 4 * (jg - lane);
      const bool n4 = (nn & 3) == 0;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int s = lane + 32 * r;
        const uint32_t j = j00 + s;
        if (n4 || j < nn) k5f_store(a, a1, b0, cbase + a.c_col[j], wD[s + s / 16]);
      }
    } else {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int s = lane + 32 * r, src = s / 4;
        const int64_t cb = __shfl_sync(0xffffffffu, cbase, src);
        const uint32_t j = 4 * __shfl_sync(0xffffffffu, jg, src) + (s & 3);
        const bool ok = __shfl_sync(0xffffffffu, live ? 1 : 0, src) != 0;
        if (ok && j < nn) k5f_store(a, a1, b0, cb + a.c_col[j], wD[s + s / 16]);
      }
    }
    __syncwarp();
  }
}

// The gate kernel: k5f_crt_n's loop and stores with kThreads -> kGateThreads, plus %smid and a clock probe (block 0, thread 0:
// clock64 and globaltimer at start and end -> the SM clock of this launch). MODE 0: k5f_crt_n's body (CRT core = k5f_elems);
// MODE 1: software prefetch of the next item's N plane words; MODE 2: two items per iteration (idx and idx + stride: both
// items' loads, then both CRTs in one basic block, then both finishes, then the stores: twice the independent chains and loads). __maxnreg__ cannot be combined with
// __launch_bounds__ (CUDA rule); __maxnreg__(96) alone = __launch_bounds__(128, 1) + the 96 cap (the launch fixes 128 threads).
template <int N, int E, int MODE>
__global__ void __maxnreg__(K45_MAXNREG) k45g_crt_n(OutArgs a, unsigned* smids, unsigned long long* clk) {
  using namespace imd::detail;
  __shared__ double sD[E == 4 ? (kGateThreads / 32) * kK5WarpSlots : 1];
  if (threadIdx.x == 0 && smids) smids[blockIdx.x] = smid();
  if (clk && blockIdx.x == 0 && threadIdx.x == 0) { clk[0] = clock64(); clk[1] = gtimer(); }   // nothing kept live across the loop
  const uint32_t m32 = static_cast<uint32_t>(a.m), nn = static_cast<uint32_t>(a.n);
  const uint32_t ng = (nn + E - 1) / E, total = static_cast<uint32_t>(a.b) * m32 * ng;
  const uint32_t stride = gridDim.x * kGateThreads;
  const int lane = threadIdx.x & 31;
  const bool a1 = k5f_alpha1(a), b0 = k5f_beta0(a), vmeta = k5f_vmeta(a);
  double* wD = sD + (E == 4 ? (threadIdx.x / 32) * kK5WarpSlots : 0);
  if constexpr (MODE == 2) {
    for (uint32_t base = blockIdx.x * kGateThreads; base < total; base += 2 * stride) {
      const uint32_t i0 = base + threadIdx.x, i1 = i0 + stride;
      const bool l0 = i0 < total, l1 = i1 < total;
      uint32_t jg0 = 0, jg1 = 0;
      int64_t cb0 = 0, cb1 = 0;
      double D0[E], D1[E];
#pragma unroll
      for (int e = 0; e < E; ++e) D0[e] = 0.0, D1[e] = 0.0;
      if (l0) {   // (i1 live implies i0 live) both items' loads, then both CRTs (one basic block), then both finishes
        constexpr int L = imd::constants::kTables.L32[N];
        const uint32_t t0 = i0 / ng, lam0 = t0 / m32, i1c = l1 ? i1 : i0, t1 = i1c / ng, lam1 = t1 / m32;
        jg0 = i0 - t0 * ng; jg1 = i1c - t1 * ng;
        uint32_t w0[N], w1[N], x0[E][L], x1[E][L];
        k5f_load_wv<N, E>(a, t0, E * jg0, w0);
        k5f_load_wv<N, E>(a, t1, E * jg1, w1);
        k5f_crt_wv<N, E>(w0, x0);
        k5f_crt_wv<N, E>(w1, x1);
        k5f_fin_x<N, E>(a, t0, lam0, E * jg0, vmeta, x0, D0);
        cb0 = a.c_bat[lam0] + a.c_row[t0 - lam0 * m32];
        if (l1) {
          k5f_fin_x<N, E>(a, t1, lam1, E * jg1, vmeta, x1, D1);
          cb1 = a.c_bat[lam1] + a.c_row[t1 - lam1 * m32];
        }
      }
      gate_store<E>(a, a1, b0, lane, wD, nn, l0, jg0, cb0, D0);
      gate_store<E>(a, a1, b0, lane, wD, nn, l1, jg1, cb1, D1);
    }
  } else {
    uint32_t wv[N];
    if constexpr (MODE == 1) {
      const uint32_t i0 = blockIdx.x * kGateThreads + threadIdx.x;
      if (i0 < total) { const uint32_t t0 = i0 / ng; k5f_load_wv<N, E>(a, t0, E * (i0 - t0 * ng), wv); }
    }
    for (uint32_t base = blockIdx.x * kGateThreads; base < total; base += stride) {
      const uint32_t idx = base + threadIdx.x;
      const bool live = idx < total;
      uint32_t jg = 0;
      int64_t cbase = 0;
      double D[E];
#pragma unroll
      for (int e = 0; e < E; ++e) D[e] = 0.0;
      if (live) {
        const uint32_t t = idx / ng, lam = t / m32;
        jg = idx - t * ng;
        if constexpr (MODE == 1) {
          const uint32_t nx = idx + stride, pidx = nx < total ? nx : idx;   // past the end: reload this item (valid address, value unused)
          const uint32_t pt = pidx / ng;
          uint32_t wn[N];
          k5f_load_wv<N, E>(a, pt, E * (pidx - pt * ng), wn);              // in flight during this item's CRT
          k5f_elems_wv<N, E>(a, t, lam, E * jg, vmeta, wv, D);
#pragma unroll
          for (int l = 0; l < N; ++l) wv[l] = wn[l];
        } else {
          k5f_elems<N, E>(a, t, lam, E * jg, vmeta, D);
        }
        cbase = a.c_bat[lam] + a.c_row[t - lam * m32];
      }
      gate_store<E>(a, a1, b0, lane, wD, nn, live, jg, cbase, D);
    }
  }
  if (clk && blockIdx.x == 0 && threadIdx.x == 0) { clk[2] = clock64(); clk[3] = gtimer(); }
}

// ---- helpers ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}
struct Moduli { uint32_t p[kN]; };
__global__ void fill_W(uint8_t* W, int64_t plane, Moduli mo) {   // W[l][t][j] = hash mod p_l (a valid residue)
  const int64_t total = plane * kN;
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < total; i += int64_t{gridDim.x} * blockDim.x) {
    const int l = static_cast<int>(i / plane);
    const int64_t r = i - l * plane;
    W[i] = static_cast<uint8_t>(hash3(static_cast<uint32_t>(r), static_cast<uint32_t>(r >> 32), l) % mo.p[l]);
  }
}
__global__ void cmp_bits(const double* x, const double* y, int64_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < n; i += int64_t{gridDim.x} * blockDim.x)
    c += __double_as_longlong(x[i]) != __double_as_longlong(y[i]);
  if (c) atomicAdd(bad, c);
}

// Host: the exact CRT of one output and D = RN(sign |z| 2^e) (round half even, normal range only).
using u128 = unsigned __int128;
static double u128_rn(u128 v) {
  if (v == 0) return 0.0;
  int b = 127;
  while (!((v >> b) & 1)) --b;
  if (b < 53) return static_cast<double>(static_cast<uint64_t>(v));
  const int sh = b - 52;
  u128 q = v >> sh, rem = v & ((u128(1) << sh) - 1), half = u128(1) << (sh - 1);
  if (rem > half || (rem == half && (q & 1))) ++q;
  return std::ldexp(static_cast<double>(static_cast<uint64_t>(q)), sh);
}
static double host_crt(const uint8_t* w, int e) {   // w[l] = residue mod p_l
  constexpr int L = imd::constants::kTables.L32[kN];
  static_assert(L <= 4, "host check uses 128-bit integers");
  u128 P = 0, H = 0;
  for (int i = L - 1; i >= 0; --i) { P = (P << 32) | imd::constants::kTables.P[kN][i]; H = (H << 32) | imd::constants::kTables.H[kN][i]; }
  u128 x = 0, M = 1;
  for (int l = 0; l < kN; ++l) {                    // Garner: x + M t = w_l (mod p_l)
    const uint32_t p = imd::constants::kTables.moduli[l];
    const uint32_t xm = static_cast<uint32_t>(x % p), mm = static_cast<uint32_t>(M % p);
    uint32_t t = 0;
    while ((xm + static_cast<uint64_t>(mm) * t) % p != w[l]) ++t;
    x += M * t;
    M *= p;
  }
  if (M != P) { std::printf("host check: product of moduli != P\n"); std::exit(1); }
  const bool neg = x > H;
  const u128 mag = neg ? P - x : x;
  const double d = std::ldexp(u128_rn(mag), e);
  return neg ? -d : d;
}

struct Shape { const char* name; int64_t m, n, k; };
struct Variant { std::string name; int E; int mode; bool gate; };   // mode: 0 plain, 1 prefetch, 2 two items per iteration

template <int E, int MODE>
static void launch_gate(const OutArgs& a, int grid, size_t dyn, unsigned* smids, unsigned long long* clk) {
  k45g_crt_n<kN, E, MODE><<<grid, kGateThreads, dyn>>>(a, smids, clk);
}
template <int E, int MODE>
static void set_dyn(size_t dyn) { CK(cudaFuncSetAttribute(k45g_crt_n<kN, E, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn))); }
template <int E, int MODE>
static int regs_of() { cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa, k45g_crt_n<kN, E, MODE>)); return fa.numRegs; }
// The gate instances (E, MODE): (1,0) (2,0) (2,1) (4,0) (4,1) (4,2).
static void launch_variant(const Variant& v, const OutArgs& a, int sms, size_t dyn, unsigned* smids, unsigned long long* clk) {
  if (!v.gate) {
    const int64_t work = a.b * a.m * ((a.n + 3) / 4);
    imd::detail::k5f_crt_n<kN, 4><<<static_cast<unsigned>((work + 255) / 256), 256>>>(a);
    return;
  }
  if (v.E == 1) launch_gate<1, 0>(a, sms, dyn, smids, clk);
  else if (v.E == 2 && v.mode == 1) launch_gate<2, 1>(a, sms, dyn, smids, clk);
  else if (v.E == 2) launch_gate<2, 0>(a, sms, dyn, smids, clk);
  else if (v.mode == 1) launch_gate<4, 1>(a, sms, dyn, smids, clk);
  else if (v.mode == 2) launch_gate<4, 2>(a, sms, dyn, smids, clk);
  else launch_gate<4, 0>(a, sms, dyn, smids, clk);
}
static int regs_variant(const Variant& v) {
  if (!v.gate) { cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa, imd::detail::k5f_crt_n<kN, 4>)); return fa.numRegs; }
  if (v.E == 1) return regs_of<1, 0>();
  if (v.E == 2) return v.mode == 1 ? regs_of<2, 1>() : regs_of<2, 0>();
  return v.mode == 1 ? regs_of<4, 1>() : v.mode == 2 ? regs_of<4, 2>() : regs_of<4, 0>();
}

int main(int argc, char** argv) {
  std::string mode = "all", out_path, shapes_arg = "r_lowk,t_df";
  int reps = 30, pad_kb = 120;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--mode") mode = v; else if (k == "--out") out_path = v; else if (k == "--reps") reps = std::atoi(v.c_str());
    else if (k == "--shapes") shapes_arg = v; else if (k == "--pad-kb") pad_kb = std::atoi(v.c_str());
    else { std::printf("unknown option %s\n", k.c_str()); return 2; }
  }
  const bool timing = mode == "all";
  if (mode != "all" && mode != "ncu") { std::printf("--mode all|ncu\n"); return 2; }
  FILE* out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "a");
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  std::printf("# device %s, %d SMs, cc %d.%d; N = %d, gate cap %d registers, pad %d KiB, reps %d, mode %s\n", prop.name, sms, prop.major, prop.minor,
              kN, K45_MAXNREG, pad_kb, reps, mode.c_str());
  const size_t dyn = size_t(pad_kb) * 1024;
  set_dyn<1, 0>(dyn); set_dyn<2, 0>(dyn); set_dyn<2, 1>(dyn); set_dyn<4, 0>(dyn); set_dyn<4, 1>(dyn); set_dyn<4, 2>(dyn);
  const std::vector<Variant> vars = {{"ref k5f_crt_n<14,4> 256thr", 4, 0, false}, {"k45g E1", 1, 0, true}, {"k45g E2", 2, 0, true},
                                     {"k45g E2pf", 2, 1, true}, {"k45g E4", 4, 0, true}, {"k45g E4pf", 4, 1, true}, {"k45g E4x2", 4, 2, true}};
  std::vector<Shape> shapes;
  for (const Shape& s : {Shape{"r_lowk", 8192, 8192, 1024}, Shape{"t_df", 4096, 4096, 2048}})
    if (("," + shapes_arg + ",").find("," + std::string(s.name) + ",") != std::string::npos) shapes.push_back(s);
  Moduli mo{};
  for (int l = 0; l < kN; ++l) mo.p[l] = imd::constants::kTables.moduli[l];
  int fails = 0;
  for (const Shape& s : shapes) {
    const int64_t m = s.m, n = s.n, plane = m * n;   // b = 1, n_hat = n (n % 16 == 0)
    uint8_t *W, *fA, *fB; int16_t *sA, *sB; int64_t *cbat, *crow, *ccol; double *Dref, *Dg; unsigned* smids; unsigned long long *bad, *clk;
    CK(cudaMalloc(&W, size_t(plane) * kN)); CK(cudaMalloc(&fA, m)); CK(cudaMalloc(&fB, n)); CK(cudaMalloc(&sA, m * 2)); CK(cudaMalloc(&sB, n * 2));
    CK(cudaMalloc(&cbat, 8)); CK(cudaMalloc(&crow, m * 8)); CK(cudaMalloc(&ccol, n * 8)); CK(cudaMalloc(&Dref, plane * 8)); CK(cudaMalloc(&Dg, plane * 8));
    CK(cudaMalloc(&smids, sms * 4)); CK(cudaMalloc(&bad, 8)); CK(cudaMalloc(&clk, 32));
    fill_W<<<4 * sms, 256>>>(W, plane, mo); CK(cudaGetLastError());
    CK(cudaMemset(fA, 0, m)); CK(cudaMemset(fB, 0, n));
    std::vector<int16_t> hsA(m), hsB(n); std::vector<int64_t> hrow(m), hcol(n);
    for (int64_t i = 0; i < m; ++i) hsA[i] = int16_t(40 + (i * 7919) % 21), hrow[i] = i * n;
    for (int64_t j = 0; j < n; ++j) hsB[j] = int16_t(40 + (j * 104729) % 21), hcol[j] = j;
    const int64_t zero = 0;
    CK(cudaMemcpy(sA, hsA.data(), m * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sB, hsB.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(crow, hrow.data(), m * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ccol, hcol.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cbat, &zero, 8, cudaMemcpyHostToDevice));
    OutArgs a{};
    a.A = nullptr; a.B = nullptr; a.C = Dref;
    a.a_row = a.a_k = a.a_bat = a.b_k = a.b_col = a.b_bat = nullptr;   // native_dot is never reached (fA = fB = 0)
    a.c_row = crow; a.c_col = ccol; a.c_bat = cbat;
    a.m = m; a.n = n; a.k = s.k; a.b = 1; a.n_hat = n;
    a.sA = sA; a.sB = sB; a.fA = fA; a.fB = fB; a.W = W; a.N = kN; a.alpha = 1.0; a.beta = 0.0;
    // reference + host check on a sample
    launch_variant(vars[0], a, sms, dyn, nullptr, nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    {
      const int S = 4096; int hbad = 0;
      std::vector<double> d(1);
      for (int q = 0; q < S; ++q) {
        const int64_t idx = (int64_t(q) * 2654435761LL + 12345) % plane, i = idx / n, j = idx % n;
        uint8_t w[kN];
        for (int l = 0; l < kN; ++l) CK(cudaMemcpy(&w[l], W + l * plane + idx, 1, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d.data(), Dref + idx, 8, cudaMemcpyDeviceToHost));
        const double h = host_crt(w, -(hsA[i] + hsB[j]));
        if (std::memcmp(&h, &d[0], 8) != 0) { if (hbad < 5) std::printf("host check %s (%lld,%lld): dev %.17g host %.17g\n", s.name, (long long)i, (long long)j, d[0], h); ++hbad; }
      }
      std::printf("%s ref vs exact host CRT: %d / %d mismatches\n", s.name, hbad, S);
      fails += hbad != 0;
      char line[256];
      std::snprintf(line, sizeof line, "{\"shape\":\"%s\",\"check\":\"ref_vs_host_crt\",\"samples\":%d,\"mismatches\":%d}\n", s.name, S, hbad);
      std::fputs(line, stdout); if (out) std::fputs(line, out);
    }
    // every gate variant once: bit-for-bit against the ref, one block per SM
    std::vector<int> ok(vars.size(), 1), uniq(vars.size(), 0);
    for (size_t v = 1; v < vars.size(); ++v) {
      a.C = Dg;
      CK(cudaMemset(Dg, 0xFF, plane * 8)); CK(cudaMemset(bad, 0, 8)); CK(cudaMemset(smids, 0xFF, sms * 4));
      launch_variant(vars[v], a, sms, dyn, smids, nullptr); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cmp_bits<<<4 * sms, 256>>>(Dg, Dref, plane, bad); CK(cudaDeviceSynchronize());
      unsigned long long hb = 0; CK(cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost));
      std::vector<unsigned> hs(sms); CK(cudaMemcpy(hs.data(), smids, sms * 4, cudaMemcpyDeviceToHost));
      std::sort(hs.begin(), hs.end()); uniq[v] = int(std::unique(hs.begin(), hs.end()) - hs.begin());
      ok[v] = hb == 0; fails += hb != 0;
      std::printf("%s %-12s bit mismatches vs ref: %llu; distinct SMs %d of %d blocks\n", s.name, vars[v].name.c_str(), hb, uniq[v], sms);
    }
    a.C = Dg;
    std::vector<std::vector<float>> ms(vars.size());
    std::vector<std::vector<double>> ghz(vars.size());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    if (timing) {
      for (int w = 0; w < 2; ++w) for (size_t v = 0; v < vars.size(); ++v) launch_variant(vars[v], a, sms, dyn, nullptr, nullptr);
      CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; ++r)
        for (size_t v = 0; v < vars.size(); ++v) {
          CK(cudaEventRecord(e0)); launch_variant(vars[v], a, sms, dyn, nullptr, vars[v].gate ? clk : nullptr); CK(cudaEventRecord(e1));
          CK(cudaEventSynchronize(e1)); float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms[v].push_back(t);
          if (vars[v].gate) {
            unsigned long long c[4]; CK(cudaMemcpy(c, clk, 32, cudaMemcpyDeviceToHost));
            if (c[3] > c[1]) ghz[v].push_back(double(c[2] - c[0]) / double(c[3] - c[1]));
          }
        }
    }
    const double flops = 2.0 * m * n * s.k;   // the shape's dense count: this stage's time as if it were the whole call
    for (size_t v = 0; v < vars.size(); ++v) {
      double med = 0, mn = 0, mx = 0, p10 = 0, p90 = 0, f = 0;
      if (!ms[v].empty()) {
        std::sort(ms[v].begin(), ms[v].end()); const size_t z = ms[v].size();
        med = z % 2 ? ms[v][z / 2] : 0.5 * (ms[v][z / 2 - 1] + ms[v][z / 2]); mn = ms[v].front(); mx = ms[v].back(); p10 = ms[v][z / 10]; p90 = ms[v][(9 * z) / 10];
      }
      if (!ghz[v].empty()) { std::sort(ghz[v].begin(), ghz[v].end()); f = ghz[v][ghz[v].size() / 2]; }
      char line[768];
      std::snprintf(line, sizeof line,
                    "{\"shape\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,\"N\":%d,\"variant\":\"%s\",\"E\":%d,\"mode\":%d,\"threads\":%d,\"grid\":%lld,"
                    "\"regs\":%d,\"bitexact_vs_ref\":%d,\"distinct_sms\":%d,\"reps\":%zu,\"tflops_eq_stage\":%.2f,\"ms_med\":%.4f,\"ms_min\":%.4f,"
                    "\"ms_p10\":%.4f,\"ms_p90\":%.4f,\"ms_max\":%.4f,\"sm_ghz_probe\":%.3f}\n",
                    s.name, (long long)m, (long long)n, (long long)s.k, kN, vars[v].name.c_str(), vars[v].E, vars[v].mode, vars[v].gate ? kGateThreads : 256,
                    vars[v].gate ? (long long)sms : (long long)((m * (n / 4) + 255) / 256), regs_variant(vars[v]), ok[v], vars[v].gate ? uniq[v] : 0, ms[v].size(),
                    med > 0 ? flops / (med * 1e9) : 0.0, med, mn, p10, p90, mx, f);
      std::fputs(line, stdout); if (out) std::fputs(line, out);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    for (void* p : {(void*)W, (void*)fA, (void*)fB, (void*)sA, (void*)sB, (void*)cbat, (void*)crow, (void*)ccol, (void*)Dref, (void*)Dg, (void*)smids, (void*)bad, (void*)clk}) CK(cudaFree(p));
  }
  if (out) std::fclose(out);
  std::printf("# %s\n", fails ? "VALIDATION FAILED" : "validation ok");
  return fails ? 1 : 0;
}
