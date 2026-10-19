// Rung 0 of the Hopper K4 test ladder (issue 17, docs/NOTES-17.md): a known-answer program for the layout assumptions that
// can only be checked on Hopper. Optional in the R1 campaign; nothing depends on it. Built for sm_90a only
// (tools/build_rung0.sh); it needs no GPU other than an H100 / H200 and runs for a fraction of a second.
//
// It runs the prototype kernels of k4_hopper_variants.def (CUTLASS 4.8's TMA + wgmma main loop, our mod-p -> u8 epilogue)
// on a small int8 problem with random data, in the fast-epilogue variants (the quad-transposed 128-bit stores that
// assume the wgmma accumulator layout) and the generic one (a byte per accumulator at CuTe's own coordinate), and
// compares the u8 planes with the exact CPU result, for several moduli (p = 256 included), in store mode and in
// add mode (a second K chunk added to the first). Reading the outcome:
//   generic OK, fast OK       the layout assumption holds on this GPU, and so does everything else in the path
//   generic OK, fast WRONG    the accumulator layout (tests/test_wgmma_layout.py) or the fast epilogue is wrong
//   generic WRONG             the main loop, the descriptors or the generic epilogue: look at issue 22's ladder
// Cases (all against exact CPU results): random data on 2 x 2 tiles of 128 x 256, K = 1024 (eight k-tiles: the pipeline wraps at 3-4
// stages), four batches with b = 2 batches per modulus and batch0 = 0 / 1 / 3 (the modulus of batch z is slot0 + (batch0 + z) / b), three
// disjoint modulus sets, p = 256 among them, store then add; and the extreme case A = B = -128 over a whole K chunk of 131008.
// Output: one JSON line per variant and mode on stdout, the summary on stderr; exit status 0 only if everything passed.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/kernel_hardware_info.h"
#include "k4/hopper/k4_hopper.cuh"

namespace {

using namespace imd::hopper;
namespace cg = cutlass::gemm;
using Coop = cg::KernelTmaWarpSpecializedCooperative;
using Ping = cg::KernelTmaWarpSpecializedPingpong;
using cute::Int;
using cute::Shape;
using _1 = Int<1>;
using _2 = Int<2>;

#define CK(x)                                                                                    \
  do {                                                                                           \
    const cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
      std::exit(2);                                                                              \
    }                                                                                            \
  } while (0)

constexpr int kM = 256, kN = 512, kK = 1024, kL = 4;  // 2 x 2 tiles of 128 x 256, eight k-tiles of 128, four batches
constexpr int kSplit = 448;                            // the two K chunks: [0, 448) (3.5 tiles: a half-zero last tile) and [448, 1024) (starts at phase 64 mod 128)
constexpr int kB = 2;                                  // batches per modulus: batch z (+ batch0) uses modulus slot0 + (batch0 + z) / kB

struct Problem {
  std::vector<int8_t> A, B;   // [z][m][k], [z][n][k]
  std::vector<int32_t> acc0;  // exact sums over k in [0, K/2) and over [K/2, K): [z][m][n]
  std::vector<int32_t> acc1;
};

Problem make_problem(uint64_t seed) {
  Problem P;
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> d(-127, 127);
  P.A.resize(size_t(kL) * kM * kK);
  P.B.resize(size_t(kL) * kN * kK);
  for (auto& x : P.A) x = int8_t(d(rng));
  for (auto& x : P.B) x = int8_t(d(rng));
  P.acc0.assign(size_t(kL) * kM * kN, 0);
  P.acc1.assign(size_t(kL) * kM * kN, 0);
  for (int z = 0; z < kL; ++z)
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kN; ++j) {
        int32_t s0 = 0, s1 = 0;
        const int8_t* a = &P.A[(size_t(z) * kM + i) * kK];
        const int8_t* b = &P.B[(size_t(z) * kN + j) * kK];
        for (int k = 0; k < kSplit; ++k) s0 += int32_t(a[k]) * b[k];
        for (int k = kSplit; k < kK; ++k) s1 += int32_t(a[k]) * b[k];
        P.acc0[(size_t(z) * kM + i) * kN + j] = s0;
        P.acc1[(size_t(z) * kM + i) * kN + j] = s1;
      }
  return P;
}

int mod_of(int64_t v, int p) { return int(((v % p) + p) % p); }

// 0: the card's SM count; > 0: the persistent scheduler is told there are this many SMs, so every CTA works through several tiles
// (the pipeline position carries over between tiles; with ping-pong, both warpgroups alternate, an odd number of tiles leaves one with fewer)
int g_sms = 0;

// Run one variant: store chunk 0, then (add = true) add chunk 1; compare with the CPU. Returns the number of wrong bytes.
template <class V>
long long run(const char* name, const Problem& P, int slot0, int batch0, bool add, int fill = 0xEE) {
  using StoreK = typename Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>::Kernel;
  using AddK = typename Variant<typename V::Tile, typename V::Cluster, typename V::Sched, true, V::kFast>::Kernel;
  using GemmS = cg::device::GemmUniversalAdapter<StoreK>;
  using GemmA = cg::device::GemmUniversalAdapter<AddK>;
  int8_t *dA, *dB;
  uint8_t* dW;
  const size_t wbytes = size_t(kL) * kM * kN;
  CK(cudaMalloc(&dA, P.A.size()));
  CK(cudaMalloc(&dB, P.B.size()));
  CK(cudaMalloc(&dW, wbytes));
  CK(cudaMemcpy(dA, P.A.data(), P.A.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, P.B.data(), P.B.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dW, fill, wbytes));
  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = g_sms ? g_sms : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  // strides by hand (this tree may lack cutlass/util): A (M, K, L) row-major, B (N, K, L) with K contiguous
  const auto sA = cute::make_stride(int64_t(kK), cute::Int<1>{}, int64_t(kM) * kK);
  const auto sB = cute::make_stride(int64_t(kK), cute::Int<1>{}, int64_t(kN) * kK);
  K4Args ea{};
  ea.W = dW, ea.ldw = kN, ea.plane_w = int64_t(kM) * kN, ea.slot0 = slot0, ea.b = kB, ea.batch0 = batch0;
  auto launch = [&](auto& gemm_tag, int k0, int kc, bool is_add) -> bool {
    using Gemm = std::remove_reference_t<decltype(gemm_tag)>;
    typename Gemm::Arguments args{};
    args.mode = cg::GemmUniversalMode::kGemm;
    args.problem_shape = {kM, kN, kc, kL};
    args.mainloop.ptr_A = dA + k0;
    args.mainloop.dA = sA;
    args.mainloop.ptr_B = dB + k0;
    args.mainloop.dB = sB;
    args.epilogue = ea;
    args.hw_info = hw;
    (void)is_add;
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "%s: can_implement failed\n", name);
      return false;
    }
    void* ws = nullptr;
    const size_t wsz = Gemm::get_workspace_size(args);
    if (wsz) CK(cudaMalloc(&ws, wsz));
    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess || gemm.run() != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "%s: initialize / run failed\n", name);
      return false;
    }
    CK(cudaDeviceSynchronize());
    if (ws) cudaFree(ws);
    return true;
  };
  GemmS gs;
  GemmA ga;
  bool ok = launch(gs, 0, kSplit, false);
  if (ok && add) ok = launch(ga, kSplit, kK - kSplit, true);
  std::vector<uint8_t> W(wbytes);
  CK(cudaMemcpy(W.data(), dW, wbytes, cudaMemcpyDeviceToHost));
  long long bad = ok ? 0 : (long long)wbytes;
  long long first_i = -1, first_j = -1, first_z = -1, first_want = 0, first_got = 0;
  if (ok)
    for (int z = 0; z < kL; ++z) {
      const int p = int(imd::constants::kTables.moduli[slot0 + (batch0 + z) / kB]);
      for (int i = 0; i < kM; ++i)
        for (int j = 0; j < kN; ++j) {
          const size_t k = (size_t(z) * kM + i) * kN + j;
          const int want = add ? mod_of(int64_t(mod_of(P.acc0[k], p)) + P.acc1[k], p) : mod_of(P.acc0[k], p);
          if (W[k] != want) {
            if (bad == 0) first_z = z, first_i = i, first_j = j, first_want = want, first_got = W[k];
            ++bad;
          }
        }
    }
  std::printf("{\"rung\": 0, \"variant\": \"%s\", \"mode\": \"%s\", \"sms\": %d, \"slot0\": %d, \"batch0\": %d, \"fill\": %d, \"wrong_bytes\": %lld, \"total_bytes\": %zu"
              ", \"first\": [%lld, %lld, %lld, %lld, %lld]}\n",
              name, add ? "add" : "store", g_sms, slot0, batch0, fill, bad, wbytes, first_z, first_i, first_j, first_want, first_got);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dW);
  return bad;
}

template <class T, class C, class S, bool F>
struct VT {
  using Tile = T;
  using Cluster = C;
  using Sched = S;
  static constexpr bool kFast = F;
};

}  // namespace

// The extreme case: A = B = -128 over a whole K chunk of 131008 (the largest the library issues): every sum is
// 131008 x 16384 = 2146435072, just inside int32 (the modulus index 3, p = 251, is past that modulus's certified range, epi_vmax[3] = 2047000250: this case is
// out of contract for it, kept because the Barrett reduction is exact there too, sampled offline), and with W added 2146435327. Store and add, 128 x 256 x 1.
template <class V>
long long run_extreme(const char* name, bool add) {
  constexpr int M = 128, N = 256, K = 131008;
  std::vector<int8_t> hA(size_t(M) * K, int8_t(-128)), hB(size_t(N) * K, int8_t(-128));
  int8_t *dA, *dB;
  uint8_t* dW;
  CK(cudaMalloc(&dA, hA.size()));
  CK(cudaMalloc(&dB, hB.size()));
  CK(cudaMalloc(&dW, size_t(M) * N));
  CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dW, 0xEE, size_t(M) * N));
  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = g_sms ? g_sms : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const auto sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M) * K);
  const auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(N) * K);
  K4Args ea{};
  ea.W = dW, ea.ldw = N, ea.plane_w = int64_t(M) * N, ea.slot0 = 3, ea.b = 1, ea.batch0 = 0;
  auto go = [&](auto tag, int k) {
    using Gemm = decltype(tag);
    typename Gemm::Arguments args{};
    args.mode = cg::GemmUniversalMode::kGemm;
    args.problem_shape = {M, N, k, 1};
    args.mainloop.ptr_A = dA, args.mainloop.dA = sA, args.mainloop.ptr_B = dB, args.mainloop.dB = sB;
    args.epilogue = ea;
    args.hw_info = hw;
    Gemm gemm;
    void* ws = nullptr;
    const size_t wsz = Gemm::get_workspace_size(args);
    if (wsz) CK(cudaMalloc(&ws, wsz));
    const bool ok = gemm.can_implement(args) == cutlass::Status::kSuccess && gemm.initialize(args, ws) == cutlass::Status::kSuccess &&
                    gemm.run() == cutlass::Status::kSuccess;
    CK(cudaDeviceSynchronize());
    if (ws) cudaFree(ws);
    return ok;
  };
  using StoreG = cg::device::GemmUniversalAdapter<typename Variant<typename V::Tile, typename V::Cluster, typename V::Sched, false, V::kFast>::Kernel>;
  using AddG = cg::device::GemmUniversalAdapter<typename Variant<typename V::Tile, typename V::Cluster, typename V::Sched, true, V::kFast>::Kernel>;
  bool ok = go(StoreG{}, K);
  if (ok && add) ok = go(AddG{}, K);
  std::vector<uint8_t> W(size_t(M) * N);
  CK(cudaMemcpy(W.data(), dW, W.size(), cudaMemcpyDeviceToHost));
  const int p = int(imd::constants::kTables.moduli[3]);
  const int64_t acc = int64_t(K) * 16384;
  const int want = add ? int(((acc % p) + p + (acc % p)) % p) : int((acc % p + p) % p);
  long long bad = ok ? 0 : (long long)W.size();
  if (ok)
    for (uint8_t w : W) bad += w != want;
  std::printf("{\"rung\": 0, \"case\": \"extreme\", \"variant\": \"%s\", \"mode\": \"%s\", \"wrong_bytes\": %lld, \"total_bytes\": %zu, \"want\": %d}\n",
              name, add ? "add" : "store", bad, W.size(), want);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dW);
  return bad;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop{};
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  if (prop.major != 9) {
    std::fprintf(stderr, "rung 0 needs a Hopper GPU (sm_90); this is sm_%d%d\n", prop.major, prop.minor);
    return 3;
  }
  const Problem P = make_problem(20261015);
  // modulus sets (slot0): disjoint, one of them holding p = 256 (the batches use slot0 + (batch0 + z) / b)
  int s256 = 0;
  for (int l = 0; l < imd::constants::kNMax; ++l)
    if (imd::constants::kTables.moduli[l] == 256u) s256 = l;
  const int slot0s[] = {s256 >= 2 ? s256 - 2 : 0, 6, 12};
  const int batch0s[] = {0, 1, 3};
  long long bad = 0;
  using T128x256 = Shape<Int<128>, Int<256>, Int<128>>;
  using T128x128 = Shape<Int<128>, Int<128>, Int<128>>;
  using T256x128 = Shape<Int<256>, Int<128>, Int<128>>;
  for (int si = 0; si < 3; ++si)
    for (bool add : {false, true}) {
      const int slot0 = slot0s[si], batch0 = batch0s[si];
      const int fill = si == 1 ? 0x11 : 0xEE;   // a fill that cannot be the expected byte everywhere
      bad += run<VT<T128x256, Shape<_1, _1, _1>, Coop, false>>("g_128x256x128_1x1_coop", P, slot0, batch0, add, fill);
      bad += run<VT<T128x256, Shape<_1, _1, _1>, Coop, true>>("f_128x256x128_1x1_coop", P, slot0, batch0, add, fill);
      bad += run<VT<T128x256, Shape<_1, _2, _1>, Coop, true>>("f_128x256x128_1x2_coop", P, slot0, batch0, add, fill);
      bad += run<VT<T128x128, Shape<_2, _1, _1>, Coop, false>>("g_128x128x128_2x1_coop", P, slot0, batch0, add, fill);
      bad += run<VT<T128x128, Shape<_2, _1, _1>, Coop, true>>("f_128x128x128_2x1_coop", P, slot0, batch0, add, fill);
      bad += run<VT<T128x128, Shape<_1, _1, _1>, Ping, false>>("g_128x128x128_1x1_ping", P, slot0, batch0, add, fill);
      bad += run<VT<T128x128, Shape<_1, _1, _1>, Ping, true>>("f_128x128x128_1x1_ping", P, slot0, batch0, add, fill);
    }
  // persistent loop: fewer SMs than tiles (16 to 32 tiles; 2 and 4 SMs keep the 2-CTA clusters whole, 3 gives odd tile counts per CTA)
  for (int sms : {2, 3, 4}) {
    g_sms = sms;
    for (bool add : {false, true}) {
      const int slot0 = slot0s[0], batch0 = batch0s[1];
      bad += run<VT<T128x256, Shape<_1, _1, _1>, Coop, true>>("f_128x256x128_1x1_coop", P, slot0, batch0, add);
      bad += run<VT<T128x128, Shape<_1, _1, _1>, Ping, true>>("f_128x128x128_1x1_ping", P, slot0, batch0, add);
      bad += run<VT<T128x128, Shape<_1, _1, _1>, Ping, false>>("g_128x128x128_1x1_ping", P, slot0, batch0, add);
      if (sms != 3) bad += run<VT<T128x128, Shape<_2, _1, _1>, Coop, true>>("f_128x128x128_2x1_coop", P, slot0, batch0, add);
    }
  }
  g_sms = 0;
  // the padded cluster grid (one M tile for a 2-CTA M cluster) comes last: if it hangs, everything above has reported
  for (int si = 0; si < 3; ++si)
    for (bool add : {false, true})
      bad += run<VT<T256x128, Shape<_2, _1, _1>, Coop, true>>("f_256x128x128_2x1_coop", P, slot0s[si], batch0s[si], add, si == 1 ? 0x11 : 0xEE);
  for (bool add : {false, true}) {
    bad += run_extreme<VT<T128x256, Shape<_1, _1, _1>, Coop, true>>("f_128x256x128_1x1_coop", add);
    bad += run_extreme<VT<T128x128, Shape<_1, _1, _1>, Ping, true>>("f_128x128x128_1x1_ping", add);
  }
  std::fprintf(stderr, bad ? "rung 0: %lld wrong bytes\n" : "rung 0: all variants exact (%lld wrong bytes)\n", bad);
  return bad ? 1 : 0;
}
