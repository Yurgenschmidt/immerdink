// K5 bimodality probe (docs/NOTES-K5-bimodal.md): n^3 ik,jk->ij through imd_contract with the
// profiling build's phase marks.
//   k5probe [nws=1] [nc=1] [reps=7] [ws_off_list=0] [n=4096] [N=14]
// Allocates A, B, nc C buffers, nws workspaces (+8 MiB slack each); for every (ws, C, off)
// prints the per-call K5 and K4 times, the device addresses (C, W region) and the median.
// Build against the bench's profiling library (bench/build.sh first), from the repo root:
//   nvcc -O2 -std=c++17 -arch=sm_89 -I include -I src bench/k5bimodal/k5probe.cu -o k5probe \
//     -L build/bench/immerdink -limmerdink -limmerdink_host -Xlinker -rpath,$PWD/build/bench/immerdink
// Run it through tools/run_bench.sh (timing) or nsys with --gpuctxsw=true (ctxstat.py).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "immerdink/immerdink.h"
#include "profile_hooks.h"

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CI(x) do { imd_status_t s_ = (x); if (s_ != IMD_STATUS_SUCCESS) { fprintf(stderr, "%s:%d imd %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

__global__ void fill(double* p, size_t n, unsigned long long seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    unsigned long long z = seed + i * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    p[i] = (double)(z >> 11) * 0x1.0p-53 - 0.5;
  }
}

static std::string sh(const char* cmd) {
  std::string r;
  FILE* f = popen(cmd, "r");
  if (!f) return r;
  char buf[256];
  while (fgets(buf, sizeof buf, f)) r += buf;
  pclose(f);
  while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
  return r;
}

int main(int argc, char** argv) {
  const int nws = argc > 1 ? atoi(argv[1]) : 1;
  const int nc = argc > 2 ? atoi(argv[2]) : 1;
  const int reps = argc > 3 ? atoi(argv[3]) : 7;
  std::vector<long long> offs;
  {
    std::string s = argc > 4 ? argv[4] : "0";
    size_t p = 0;
    while (p <= s.size()) {
      size_t q = s.find(',', p);
      if (q == std::string::npos) q = s.size();
      offs.push_back(strtoll(s.substr(p, q - p).c_str(), nullptr, 0));
      p = q + 1;
    }
  }
  const int64_t n = argc > 5 ? atoll(argv[5]) : 4096;
  const int N = argc > 6 ? atoi(argv[6]) : 14;
  int64_t ext[2] = {n, n}, sA[2] = {n, 1};
  int32_t la[2] = {0, 2}, lb[2] = {1, 2}, lc[2] = {0, 1};
  imd_tensor_t A{2, ext, sA, la}, B{2, ext, sA, lb}, C{2, ext, sA, lc};
  imd_options_t opt{N, 0, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
  imd_region_t reg[8];
  int32_t nreg = 0;
  size_t wsb = 0;
  CI(imd_workspace_layout(&A, &B, &C, &opt, reg, 8, &nreg, &wsb));
  imd_handle_t h;
  CI(imd_create(&h));
  imd_plan_t plan;
  CI(imd_plan_create(h, &A, &B, &C, &opt, &plan));
  cudaStream_t s;
  CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  double *dA, *dB;
  CU(cudaMalloc(&dA, n * n * 8));
  CU(cudaMalloc(&dB, n * n * 8));
  fill<<<1024, 256, 0, s>>>(dA, n * n, 1);
  fill<<<1024, 256, 0, s>>>(dB, n * n, 2);
  std::vector<double*> dC(nc);
  for (auto& c : dC) CU(cudaMalloc(&c, n * n * 8));
  const size_t slack = size_t{8} << 20;
  std::vector<char*> ws(nws);
  for (auto& w : ws) CU(cudaMalloc(&w, wsb + slack));
  CU(cudaStreamSynchronize(s));
  printf("ws_bytes %zu W_off %llu W_bytes %llu RA_off %llu RB_off %llu A %p B %p\n", wsb,
         (unsigned long long)reg[7].offset, (unsigned long long)reg[7].bytes,
         (unsigned long long)reg[5].offset, (unsigned long long)reg[6].offset, (void*)dA, (void*)dB);
  // warm
  for (int r = 0; r < 3; ++r) CI(imd_contract(plan, 1.0, dA, dB, 0.0, dC[0], ws[0], wsb, s));
  CU(cudaStreamSynchronize(s));
  imd_prof_enable(1);
  const char* tags[256];
  float ms[256];
  for (int iw = 0; iw < nws; ++iw)
    for (int ic = 0; ic < nc; ++ic)
      for (long long off : offs) {
        char* w = ws[iw] + off;
        std::vector<float> k5, k4, tot;
        for (int r = 0; r < reps; ++r) {
          imd_prof_reset();
          CI(imd_contract(plan, 1.0, dA, dB, 0.0, dC[ic], w, wsb, s));
          int cnt = imd_prof_read(256, tags, ms);
          float a4 = 0, a5 = 0, t = 0;
          for (int i = 0; i < cnt; ++i) {
            t += ms[i];
            if (!strncmp(tags[i], "K4", 2)) a4 += ms[i];
            if (!strcmp(tags[i], "K5 crt")) a5 += ms[i];
          }
          k5.push_back(a5), k4.push_back(a4), tot.push_back(t);
        }
        printf("  K5 seq:");
        for (float v : k5) printf(" %.3f", v);
        printf("\n  K4 seq:");
        for (float v : k4) printf(" %.3f", v);
        printf("\n");
        std::sort(k5.begin(), k5.end());
        std::sort(k4.begin(), k4.end());
        std::sort(tot.begin(), tot.end());
        const unsigned long long W = (unsigned long long)(w + reg[7].offset);
        const unsigned long long Cc = (unsigned long long)dC[ic];
        printf("ws%d c%d off %lld  ws %p W 0x%llx C 0x%llx  Wmod2M 0x%llx Cmod2M 0x%llx  W-C 0x%llx  "
               "K5 med %.4f min %.4f max %.4f  K4 %.3f tot %.3f\n",
               iw, ic, off, (void*)w, W, Cc, W & 0x1fffff, Cc & 0x1fffff, W - Cc, k5[reps / 2], k5[0],
               k5[reps - 1], k4[reps / 2], tot[reps / 2]);
        fflush(stdout);
      }
  imd_prof_enable(0);
  printf("clock %s\n", sh("nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader").c_str());
  return 0;
}
