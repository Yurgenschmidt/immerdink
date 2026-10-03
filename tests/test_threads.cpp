// Host-thread safety of the C API (DESIGN v1.2 §6, spec/HostAPI.tla).
//
// Phase 1 (first thing the process does with the library's GEMM path, so the threads make
// the process's first imd_contract calls concurrently): kThreads host threads start
// together and each, on its own stream with its own workspace and output buffers,
//   - contracts with two plans SHARED by all threads (one IMD_VEC_GENERATE, one
//     IMD_VEC_RESIDENT), created by the main thread before the threads start;
//   - creates its OWN plans concurrently with the other threads (both vec modes, a batched
//     tensor contraction) and contracts with them;
//   - destroys its own GENERATE plan right after imd_contract returns, before the stream is
//     synchronized (the v1.2 lifetime rule for generate plans);
//   - repeats kIters times, interleaving the plans.
// Phase 2: the main thread computes every result again, sequentially, on fresh plans and
// the legacy stream. Every phase-1 result must equal it bit for bit (results are a pure
// function of the inputs: DESIGN §1; the sequential path is checked against spec/oracle.py
// by the contract_* suites).
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <cuda_runtime.h>  // the typed cudaMalloc overload

#include "immerdink/immerdink.h"

namespace {

constexpr int kThreads = 8;
constexpr int kIters = 8;
std::atomic<int> g_failures{0};

void fail(const char* what, long a = 0, long b = 0) {
  if (g_failures.fetch_add(1) < 20) std::fprintf(stderr, "FAIL %s (%ld, %ld)\n", what, a, b);
}

bool ok(imd_status_t st, const char* what) {
  if (st == IMD_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "FAIL %s: %s\n", what, imd_status_string(st));
  g_failures.fetch_add(1);
  return false;
}
bool ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e));
  g_failures.fetch_add(1);
  return false;
}

// A contraction: dense row-major tensors, labels i=1 j=2 k=3 l=4.
struct Problem {
  std::vector<int64_t> eA, sA, eB, sB, eC, sC;
  std::vector<int32_t> lA, lB, lC;
  int N, g;
  int64_t nA, nB, nC;  // elements
  imd_tensor_t A() const { return {int32_t(eA.size()), eA.data(), sA.data(), lA.data()}; }
  imd_tensor_t B() const { return {int32_t(eB.size()), eB.data(), sB.data(), lB.data()}; }
  imd_tensor_t C() const { return {int32_t(eC.size()), eC.data(), sC.data(), lC.data()}; }
  imd_options_t opt(int vec) const { return {N, g, IMD_RN, IMD_EMULATE, vec}; }
};

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
Problem make(std::vector<int64_t> eA, std::vector<int32_t> lA, std::vector<int64_t> eB,
             std::vector<int32_t> lB, std::vector<int64_t> eC, std::vector<int32_t> lC, int N,
             int g) {
  Problem p;
  p.eA = eA, p.lA = lA, p.sA = dense(eA), p.nA = count(eA);
  p.eB = eB, p.lB = lB, p.sB = dense(eB), p.nB = count(eB);
  p.eC = eC, p.lC = lC, p.sC = dense(eC), p.nC = count(eC);
  p.N = N, p.g = g;
  return p;
}

// Inputs of one problem on the device (shared read-only by every thread) and C's initial value.
struct Data {
  double *A = nullptr, *B = nullptr;
  std::vector<double> C0;
};

std::vector<double> random_values(int64_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::vector<double> v(n);
  for (auto& x : v) x = uni(rng) * std::exp(2.0 * gauss(rng));
  if (n > 7) v[n / 7] = INFINITY;  // one flagged row / column: the native path too
  return v;
}

// One call: plan + C buffer + result. Runs imd_contract into dC on `stream`.
bool contract(imd_plan_t plan, const Data& d, double* dC, const Problem& p, void* ws, size_t wsb,
              cudaStream_t stream) {
  if (!ok(cudaMemcpyAsync(dC, d.C0.data(), p.nC * 8, cudaMemcpyHostToDevice, stream), "C upload"))
    return false;
  return ok(imd_contract(plan, 1.0, d.A, d.B, 0.5, dC, ws, wsb, stream), "imd_contract");
}

size_t ws_size(const Problem& p, int vec) {
  const imd_tensor_t A = p.A(), B = p.B(), C = p.C();
  const imd_options_t o = p.opt(vec);
  size_t b = 0;
  ok(imd_workspace_size(&A, &B, &C, &o, &b), "imd_workspace_size");
  return b;
}

imd_plan_t make_plan(imd_handle_t h, const Problem& p, int vec) {
  const imd_tensor_t A = p.A(), B = p.B(), C = p.C();
  const imd_options_t o = p.opt(vec);
  imd_plan_t plan = nullptr;
  ok(imd_plan_create(h, &A, &B, &C, &o, &plan), "imd_plan_create");
  return plan;
}

}  // namespace

int main() {
  // P0: shared plans (matrix; g = 2: seven residue groups, so seven K4 launches per call);
  // P1: per-thread plans (batched tensor, C permuted so lambda is not outermost).
  const Problem P[2] = {
      make({300, 700}, {1, 3}, {700, 200}, {3, 2}, {300, 200}, {1, 2}, 14, 2),
      make({4, 64, 200}, {4, 1, 3}, {4, 200, 48}, {4, 3, 2}, {64, 4, 48}, {1, 4, 2}, 9, 0),
  };
  Data D[2];
  for (int q = 0; q < 2; ++q) {
    const auto a = random_values(P[q].nA, 11 + q), b = random_values(P[q].nB, 21 + q);
    D[q].C0 = random_values(P[q].nC, 31 + q);
    if (!ok(cudaMalloc(&D[q].A, P[q].nA * 8), "malloc A") ||
        !ok(cudaMalloc(&D[q].B, P[q].nB * 8), "malloc B") ||
        !ok(cudaMemcpy(D[q].A, a.data(), P[q].nA * 8, cudaMemcpyHostToDevice), "upload A") ||
        !ok(cudaMemcpy(D[q].B, b.data(), P[q].nB * 8, cudaMemcpyHostToDevice), "upload B"))
      return 1;
  }
  imd_handle_t h = nullptr;
  if (!ok(imd_create(&h), "imd_create")) return 1;
  size_t wsb = 0;
  for (int q = 0; q < 2; ++q)
    for (int vec : {IMD_VEC_GENERATE, IMD_VEC_RESIDENT}) wsb = std::max(wsb, ws_size(P[q], vec));

  // Shared plans (main thread), then the threads: results[t][iter][slot] host copies, slot =
  // 0 shared GENERATE, 1 shared RESIDENT, 2 own GENERATE, 3 own RESIDENT.
  imd_plan_t shared[2] = {make_plan(h, P[0], IMD_VEC_GENERATE), make_plan(h, P[0], IMD_VEC_RESIDENT)};
  std::vector<std::vector<double>> results(kThreads * kIters * 4);
  std::barrier start(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      cudaStream_t s = nullptr;
      void* ws = nullptr;
      double* dC[4] = {};
      bool good = ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream") &&
                  ok(cudaMalloc(&ws, wsb), "malloc ws");
      for (int slot = 0; slot < 4 && good; ++slot)
        good = ok(cudaMalloc(&dC[slot], P[slot / 2].nC * 8), "malloc C");
      start.arrive_and_wait();  // every thread's first imd_plan_create / imd_contract together
      for (int it = 0; it < kIters && good; ++it) {
        // own plans, created concurrently by all threads
        imd_plan_t own_gen = make_plan(h, P[1], IMD_VEC_GENERATE);
        imd_plan_t own_res = make_plan(h, P[1], IMD_VEC_RESIDENT);
        const int order[4] = {(t + it) % 4, (t + it + 1) % 4, (t + it + 2) % 4, (t + it + 3) % 4};
        for (int slot : order) {
          imd_plan_t plan = slot == 0 ? shared[0] : slot == 1 ? shared[1] : slot == 2 ? own_gen : own_res;
          good = good && contract(plan, D[slot / 2], dC[slot], P[slot / 2], ws, wsb, s);
          if (slot == 2) {  // v1.2: a generate plan may go once its calls have returned
            imd_plan_destroy(own_gen);
            own_gen = nullptr;
          }
        }
        good = good && ok(cudaStreamSynchronize(s), "stream sync");
        imd_plan_destroy(own_res);  // resident: only after the GPU work
        for (int slot = 0; slot < 4 && good; ++slot) {
          auto& r = results[(t * kIters + it) * 4 + slot];
          r.resize(P[slot / 2].nC);
          good = ok(cudaMemcpy(r.data(), dC[slot], r.size() * 8, cudaMemcpyDeviceToHost), "download");
        }
      }
      for (double* c : dC) cudaFree(c);
      cudaFree(ws);
      if (s) cudaStreamDestroy(s);
      if (!good) fail("thread", t);
    });
  }
  for (auto& th : threads) th.join();
  for (imd_plan_t p : shared) imd_plan_destroy(p);

  // Phase 2: sequential reference on fresh plans, legacy stream.
  std::vector<double> ref[4];
  {
    void* ws = nullptr;
    double* dC = nullptr;
    if (!ok(cudaMalloc(&ws, wsb), "malloc ws") || !ok(cudaMalloc(&dC, P[1].nC * 8 + P[0].nC * 8), "C"))
      return 1;
    for (int slot = 0; slot < 4; ++slot) {
      const Problem& p = P[slot / 2];
      imd_plan_t plan = make_plan(h, p, slot % 2 == 0 ? IMD_VEC_GENERATE : IMD_VEC_RESIDENT);
      contract(plan, D[slot / 2], dC, p, ws, wsb, nullptr);
      ok(cudaDeviceSynchronize(), "sync");
      ref[slot].resize(p.nC);
      ok(cudaMemcpy(ref[slot].data(), dC, p.nC * 8, cudaMemcpyDeviceToHost), "download");
      imd_plan_destroy(plan);
    }
    cudaFree(dC);
    cudaFree(ws);
  }
  // generate and resident agree with each other ...
  for (int q = 0; q < 2; ++q)
    if (std::memcmp(ref[2 * q].data(), ref[2 * q + 1].data(), P[q].nC * 8) != 0)
      fail("vec modes differ (sequential)", q);
  // ... and every threaded result with the sequential one, bit for bit
  long checked = 0;
  for (int t = 0; t < kThreads; ++t)
    for (int it = 0; it < kIters; ++it)
      for (int slot = 0; slot < 4; ++slot) {
        const auto& r = results[(t * kIters + it) * 4 + slot];
        if (r.size() != ref[slot].size() || std::memcmp(r.data(), ref[slot].data(), r.size() * 8) != 0)
          fail("threaded result differs from the sequential one", t, slot);
        checked += long(r.size());
      }
  for (auto& d : D) cudaFree(d.A), cudaFree(d.B);
  imd_destroy(h);
  std::printf("%d threads x %d iterations x 4 calls (shared and own plans, both vec modes): "
              "%ld C elements bit-identical to the sequential run: %s\n",
              kThreads, kIters, checked, g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
