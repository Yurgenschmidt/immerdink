// imd_contract is stream-capture safe (DESIGN v1.2 §6): in each vec_mode, one call is
// captured into a CUDA graph on the caller's stream in cudaStreamCaptureModeGlobal (which
// makes any allocation or synchronization during capture, in any thread, an error), the
// graph is replayed twice, and each replay's C is bit-identical to an uncaptured call.
// Shapes: a matrix with three residue groups, a long-k matrix (v1.6 KL), a batched tensor,
// and an alpha = 0 call (beta-only K5), so K0 / K1 / KL / K2-K4 / K5 and the beta-only path
// are all captured.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "immerdink/immerdink.h"

namespace {

int g_failures = 0;
#define CHECK(x)                                                                         \
  do {                                                                                   \
    if (!(x)) {                                                                          \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                          \
      ++g_failures;                                                                      \
      return;                                                                            \
    }                                                                                    \
  } while (0)

std::vector<int64_t> dense(const std::vector<int64_t>& e) {
  std::vector<int64_t> s(e.size());
  int64_t acc = 1;
  for (size_t d = e.size(); d-- > 0;) s[d] = acc, acc *= e[d];
  return s;
}
int64_t count(const std::vector<int64_t>& e) {
  int64_t c = 1;
  for (int64_t x : e) c *= x;
  return c;
}
std::vector<double> values(int64_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<double> v(n);
  for (auto& x : v) x = g(rng) * std::exp(2.0 * g(rng));
  return v;
}

void run(const char* name, std::vector<int64_t> eA, std::vector<int32_t> lA, std::vector<int64_t> eB,
         std::vector<int32_t> lB, std::vector<int64_t> eC, std::vector<int32_t> lC, int N, int g,
         double alpha, int vec, imd_handle_t h) {
  const auto sA = dense(eA), sB = dense(eB), sC = dense(eC);
  const imd_tensor_t A{int32_t(eA.size()), eA.data(), sA.data(), lA.data()};
  const imd_tensor_t B{int32_t(eB.size()), eB.data(), sB.data(), lB.data()};
  const imd_tensor_t C{int32_t(eC.size()), eC.data(), sC.data(), lC.data()};
  const imd_options_t o{N, g, IMD_RN, IMD_EMULATE, vec, IMD_K4_TUNE_OFF};
  const int64_t nA = count(eA), nB = count(eB), nC = count(eC);
  const auto a = values(nA, 1), b = values(nB, 2), c0 = values(nC, 3);
  size_t wsb = 0;
  CHECK(imd_workspace_size(&A, &B, &C, &o, &wsb) == IMD_STATUS_SUCCESS);
  imd_plan_t plan = nullptr;
  CHECK(imd_plan_create(h, &A, &B, &C, &o, &plan) == IMD_STATUS_SUCCESS);
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  void* ws = nullptr;
  cudaStream_t s = nullptr;
  CHECK(cudaMalloc(&dA, nA * 8) == cudaSuccess && cudaMalloc(&dB, nB * 8) == cudaSuccess &&
        cudaMalloc(&dC, nC * 8) == cudaSuccess && cudaMalloc(&ws, wsb + 256) == cudaSuccess);
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(cudaMemcpy(dA, a.data(), nA * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaMemcpy(dB, b.data(), nB * 8, cudaMemcpyHostToDevice) == cudaSuccess);

  // reference: an uncaptured call
  std::vector<double> ref(nC), got(nC);
  CHECK(cudaMemcpy(dC, c0.data(), nC * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(imd_contract(plan, alpha, dA, dB, 0.5, dC, ws, wsb, s) == IMD_STATUS_SUCCESS);
  CHECK(cudaStreamSynchronize(s) == cudaSuccess);
  CHECK(cudaMemcpy(ref.data(), dC, nC * 8, cudaMemcpyDeviceToHost) == cudaSuccess);

  // capture one call (strictest mode), then replay it twice
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) == cudaSuccess);
  const imd_status_t st = imd_contract(plan, alpha, dA, dB, 0.5, dC, ws, wsb, s);
  CHECK(cudaStreamEndCapture(s, &graph) == cudaSuccess);
  CHECK(st == IMD_STATUS_SUCCESS);
  CHECK(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
  for (int replay = 0; replay < 2; ++replay) {
    // on s: the legacy stream is not ordered with a non-blocking stream
    CHECK(cudaMemcpyAsync(dC, c0.data(), nC * 8, cudaMemcpyHostToDevice, s) == cudaSuccess);
    CHECK(cudaMemsetAsync(ws, 0xA5, wsb, s) == cudaSuccess);  // the graph must rewrite what it reads
    CHECK(cudaGraphLaunch(exec, s) == cudaSuccess);
    if (const cudaError_t e = cudaStreamSynchronize(s); e != cudaSuccess)
      std::fprintf(stderr, "%s vec_mode %d replay %d: %s\n", name, vec, replay, cudaGetErrorString(e));
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaMemcpy(got.data(), dC, nC * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
    if (std::memcmp(got.data(), ref.data(), nC * 8) != 0) {
      std::fprintf(stderr, "FAIL %s vec_mode %d: replay %d differs from the uncaptured call\n",
                   name, vec, replay);
      ++g_failures;
    }
  }
  std::printf("  ok  %s, vec_mode %d: captured, replayed twice bit-exact\n", name, vec);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
  cudaFree(dA), cudaFree(dB), cudaFree(dC), cudaFree(ws);
  imd_plan_destroy(plan);
}

// v1.7 (DESIGN §6b.7): imd_pack + imd_contract_packed captured into ONE graph (the pack
// rewritten, then read, per replay) and replayed twice; each replay bit-identical to the
// uncaptured plain imd_contract (imd_pack_desc_for packs reproduce it bit for bit).
// `which`: bit 0 / 1 = pack the caller's A / B.
void run_packed(const char* name, std::vector<int64_t> eA, std::vector<int32_t> lA,
                std::vector<int64_t> eB, std::vector<int32_t> lB, std::vector<int64_t> eC,
                std::vector<int32_t> lC, int N, int g, int which, int vec, imd_handle_t h) {
  const auto sA = dense(eA), sB = dense(eB), sC = dense(eC);
  const imd_tensor_t A{int32_t(eA.size()), eA.data(), sA.data(), lA.data()};
  const imd_tensor_t B{int32_t(eB.size()), eB.data(), sB.data(), lB.data()};
  const imd_tensor_t C{int32_t(eC.size()), eC.data(), sC.data(), lC.data()};
  const imd_options_t o{N, g, IMD_RN, IMD_EMULATE, vec, IMD_K4_TUNE_OFF};
  const int64_t nA = count(eA), nB = count(eB), nC = count(eC);
  auto a = values(nA, 4), b = values(nB, 5);
  const auto c0 = values(nC, 6);
  a[nA / 3] = INFINITY;  // a flagged output: the native path reads the FP64 operand too
  imd_pack_desc_t desc[2];
  int32_t labels[2][8];
  imd_pack_t hp[2] = {nullptr, nullptr};
  size_t pb[2] = {0, 0}, pw[2] = {0, 0};
  for (int x = 0; x < 2; ++x) {
    if (!(which & (1 << x))) continue;
    std::memset(&desc[x], 0, sizeof desc[x]);
    desc[x].struct_size = sizeof desc[x];
    CHECK(imd_pack_desc_for(&A, &B, &C, &o, x, &desc[x], labels[x], 8) == IMD_STATUS_SUCCESS);
    CHECK(imd_pack_create(h, &desc[x], &hp[x]) == IMD_STATUS_SUCCESS);
    CHECK(imd_pack_bytes(hp[x], &pb[x], &pw[x]) == IMD_STATUS_SUCCESS);
  }
  const imd_packing_t packing = {hp[0] ? &desc[0] : nullptr, hp[1] ? &desc[1] : nullptr};
  size_t wsb = 0, wsp = 0;
  CHECK(imd_workspace_size(&A, &B, &C, &o, &wsb) == IMD_STATUS_SUCCESS);
  CHECK(imd_workspace_size_packed(&A, &B, &C, &o, &packing, &wsp) == IMD_STATUS_SUCCESS);
  imd_plan_t plain = nullptr, packed = nullptr;
  CHECK(imd_plan_create(h, &A, &B, &C, &o, &plain) == IMD_STATUS_SUCCESS);
  CHECK(imd_plan_create_packed(h, &A, &B, &C, &o, hp[0], hp[1], &packed) == IMD_STATUS_SUCCESS);
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  void *ws = nullptr, *dP[2] = {nullptr, nullptr}, *dPw[2] = {nullptr, nullptr};
  cudaStream_t s = nullptr;
  CHECK(cudaMalloc(&dA, nA * 8) == cudaSuccess && cudaMalloc(&dB, nB * 8) == cudaSuccess &&
        cudaMalloc(&dC, nC * 8) == cudaSuccess && cudaMalloc(&ws, wsb + 256) == cudaSuccess);
  for (int x = 0; x < 2; ++x)
    if (hp[x]) CHECK(cudaMalloc(&dP[x], pb[x]) == cudaSuccess && cudaMalloc(&dPw[x], pw[x] + 1) == cudaSuccess);
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(cudaMemcpy(dA, a.data(), nA * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaMemcpy(dB, b.data(), nB * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  // reference: the plain uncaptured call
  std::vector<double> ref(nC), got(nC);
  CHECK(cudaMemcpy(dC, c0.data(), nC * 8, cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(imd_contract(plain, 1.0, dA, dB, 0.5, dC, ws, wsb, s) == IMD_STATUS_SUCCESS);
  CHECK(cudaStreamSynchronize(s) == cudaSuccess);
  CHECK(cudaMemcpy(ref.data(), dC, nC * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
  // capture: the packs, then the packed contraction, on one stream (stream order is RAW)
  const imd_operand_t opA = {dA, dP[0], pb[0], hp[0]}, opB = {dB, dP[1], pb[1], hp[1]};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  imd_status_t st[3] = {IMD_STATUS_SUCCESS, IMD_STATUS_SUCCESS, IMD_STATUS_SUCCESS};
  CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) == cudaSuccess);
  for (int x = 0; x < 2; ++x)
    if (hp[x]) st[x] = imd_pack(hp[x], x == 0 ? dA : dB, dP[x], pb[x], dPw[x], pw[x], s);
  st[2] = imd_contract_packed(packed, 1.0, &opA, &opB, 0.5, dC, ws, wsp, s);
  CHECK(cudaStreamEndCapture(s, &graph) == cudaSuccess);
  CHECK(st[0] == IMD_STATUS_SUCCESS && st[1] == IMD_STATUS_SUCCESS && st[2] == IMD_STATUS_SUCCESS);
  CHECK(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
  for (int replay = 0; replay < 2; ++replay) {
    CHECK(cudaMemcpyAsync(dC, c0.data(), nC * 8, cudaMemcpyHostToDevice, s) == cudaSuccess);
    CHECK(cudaMemsetAsync(ws, 0xA5, wsp, s) == cudaSuccess);
    for (int x = 0; x < 2; ++x)  // the graph must rewrite every packed byte it reads
      if (hp[x]) CHECK(cudaMemsetAsync(dP[x], 0xA5, pb[x], s) == cudaSuccess);
    CHECK(cudaGraphLaunch(exec, s) == cudaSuccess);
    if (const cudaError_t e = cudaStreamSynchronize(s); e != cudaSuccess)
      std::fprintf(stderr, "%s vec_mode %d replay %d: %s\n", name, vec, replay, cudaGetErrorString(e));
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaMemcpy(got.data(), dC, nC * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
    if (std::memcmp(got.data(), ref.data(), nC * 8) != 0) {
      std::fprintf(stderr, "FAIL %s vec_mode %d: packed replay %d differs from the plain call\n",
                   name, vec, replay);
      ++g_failures;
    }
  }
  std::printf("  ok  %s, vec_mode %d: imd_pack + imd_contract_packed captured, replayed twice, "
              "bit-identical to imd_contract\n", name, vec);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);
  cudaFree(dA), cudaFree(dB), cudaFree(dC), cudaFree(ws);
  for (int x = 0; x < 2; ++x) cudaFree(dP[x]), cudaFree(dPw[x]), imd_pack_destroy(hp[x]);
  imd_plan_destroy(plain);
  imd_plan_destroy(packed);
}

}  // namespace

// Issue 23: IMD_TEST_SEQ / IMD_TEST_K5_CRT force the instruction-sequence set / the K5 CRT of every call (test hooks of the library built for the tests),
// so that the graph-captured k5fin kernels (seq 133 with the limb CRT) are captured and replayed too. IMD_TEST_REQUIRE_K5FIN=1: at least one k5fin launch.
extern "C" int imd_test_set_seq(int bits);
extern "C" int imd_test_set_k5_crt(int variant);
extern "C" int imd_test_k5_fin_counts(int reset, int64_t* out);

int main() {
  if (const char* v = std::getenv("IMD_TEST_SEQ"); v != nullptr && imd_test_set_seq(std::atoi(v)) != 0) {
    std::printf("FAIL: imd_test_set_seq(%s)\n", v);
    return 1;
  }
  if (const char* v = std::getenv("IMD_TEST_K5_CRT"); v != nullptr && imd_test_set_k5_crt(std::atoi(v)) != 0) {
    std::printf("FAIL: imd_test_set_k5_crt(%s)\n", v);
    return 1;
  }
  int64_t fin0[4] = {0, 0, 0, 0};
  imd_test_k5_fin_counts(1, fin0);
  imd_handle_t h = nullptr;
  if (imd_create(&h) != IMD_STATUS_SUCCESS) return 1;
  for (int vec : {IMD_VEC_GENERATE, IMD_VEC_RESIDENT}) {
    run("matrix 300x200x700 N=14 g=5", {300, 700}, {1, 3}, {700, 200}, {3, 2}, {300, 200}, {1, 2},
        14, 5, 1.0, vec, h);
    // v1.6 KL for both operands (k_hat = 5008 > 1024): A along k (two chunks), B along rows
    // (8-row tiles, ten chunks: re-read chunks), RB rows 1032..1039 zeroed by KL_B; g = 3. B has
    // 1032 rows (129 KL tiles: issue 21 klstarve keeps its KL on any device of < 258 SMs).
    run("matrix 200x1032x5000 N=8 g=3 (KL)", {200, 5000}, {1, 3}, {5000, 1032}, {3, 2}, {200, 1032},
        {1, 2}, 8, 3, 1.0, vec, h);
    // issue 21 klstarve: a 40-row B read along rows (5 KL tiles) takes K1 + K3 under capture.
    run("matrix 200x40x5000 N=8 g=3 (K1 + K3 for B)", {200, 5000}, {1, 3}, {5000, 40}, {3, 2}, {200, 40},
        {1, 2}, 8, 3, 1.0, vec, h);
    run("batched lik,lkj->ilj N=9", {4, 64, 200}, {4, 1, 3}, {4, 200, 48}, {4, 3, 2}, {64, 4, 48},
        {1, 4, 2}, 9, 0, -1.5, vec, h);
    run("alpha = 0 (beta-only K5)", {300, 700}, {1, 3}, {700, 200}, {3, 2}, {300, 200}, {1, 2}, 14,
        0, 0.0, vec, h);
    // v1.7 packed operands: A packed (KF on B), both packed (K0, K4, K5 only), B packed with
    // KL on A (k_hat > 1024), a batched tensor with A packed
    run_packed("packed A, matrix 300x200x700 N=14 g=5", {300, 700}, {1, 3}, {700, 200}, {3, 2},
               {300, 200}, {1, 2}, 14, 5, 1, vec, h);
    run_packed("packed A and B, matrix 300x200x700 N=14 g=5", {300, 700}, {1, 3}, {700, 200},
               {3, 2}, {300, 200}, {1, 2}, 14, 5, 3, vec, h);
    run_packed("packed B, matrix 200x40x5000 N=8 g=3 (KL on A)", {200, 5000}, {1, 3}, {5000, 40},
               {3, 2}, {200, 40}, {1, 2}, 8, 3, 2, vec, h);
    run_packed("packed A, batched lik,lkj->ilj N=9", {4, 64, 200}, {4, 1, 3}, {4, 200, 48},
               {4, 3, 2}, {64, 4, 48}, {1, 4, 2}, 9, 0, 1, vec, h);
  }
  imd_destroy(h);
  if (const char* r = std::getenv("IMD_TEST_REQUIRE_K5FIN"); r != nullptr && std::atoi(r) != 0) {
    int64_t fin[4] = {0, 0, 0, 0};
    imd_test_k5_fin_counts(0, fin);
    std::printf("k5fin launches (variant 0..3): %lld %lld %lld %lld\n", (long long)fin[0], (long long)fin[1], (long long)fin[2], (long long)fin[3]);
    if (fin[0] + fin[1] + fin[2] + fin[3] == 0) {
      std::printf("FAIL: no k5fin kernel launched\n");
      ++g_failures;
    }
  }
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
