// imd_contract is stream-capture safe (DESIGN v1.2 §6): in each vec_mode, one call is
// captured into a CUDA graph on the caller's stream in cudaStreamCaptureModeGlobal (which
// makes any allocation or synchronization during capture, in any thread, an error), the
// graph is replayed twice, and each replay's C is bit-identical to an uncaptured call.
// Shapes: a matrix with three residue groups, a batched tensor, and an alpha = 0 call
// (beta-only K5), so K0 / K1 / K2-K4 / K5 and the beta-only path are all captured.
#include <cmath>
#include <cstdint>
#include <cstdio>
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

}  // namespace

int main() {
  imd_handle_t h = nullptr;
  if (imd_create(&h) != IMD_STATUS_SUCCESS) return 1;
  for (int vec : {IMD_VEC_GENERATE, IMD_VEC_RESIDENT}) {
    run("matrix 300x200x700 N=14 g=5", {300, 700}, {1, 3}, {700, 200}, {3, 2}, {300, 200}, {1, 2},
        14, 5, 1.0, vec, h);
    run("batched lik,lkj->ilj N=9", {4, 64, 200}, {4, 1, 3}, {4, 200, 48}, {4, 3, 2}, {64, 4, 48},
        {1, 4, 2}, 9, 0, -1.5, vec, h);
    run("alpha = 0 (beta-only K5)", {300, 700}, {1, 3}, {700, 200}, {3, 2}, {300, 200}, {1, 2}, 14,
        0, 0.0, vec, h);
  }
  imd_destroy(h);
  std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
