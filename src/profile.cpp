// Phase marks for benchmarks (src/profile.hpp, src/profile_hooks.h). Compiled only with
// -DIMMERDINK_PROFILE=ON.
#include <cuda_runtime_api.h>

#include "profile.hpp"
#include "profile_hooks.h"

namespace {

constexpr int kPool = 4096;  // marks per profiled call (K4 launches dominate: chunks x groups)

struct Prof {
  bool enabled = false;
  bool created = false;
  int count = 0;
  cudaEvent_t ev[kPool];
  const char* tag[kPool];
};
Prof g_prof;

}  // namespace

namespace imd::detail {

void prof_mark(cudaStream_t stream, const char* tag) {
  if (!g_prof.enabled || g_prof.count >= kPool) return;
  if (cudaEventRecord(g_prof.ev[g_prof.count], stream) != cudaSuccess) return;
  g_prof.tag[g_prof.count++] = tag;
}

}  // namespace imd::detail

extern "C" {

IMD_API int imd_prof_enable(int on) {
  if (on && !g_prof.created) {
    for (int i = 0; i < kPool; ++i)
      if (cudaEventCreate(&g_prof.ev[i]) != cudaSuccess) return 1;
    g_prof.created = true;
  }
  g_prof.enabled = on != 0;
  g_prof.count = 0;
  return 0;
}

IMD_API void imd_prof_reset(void) { g_prof.count = 0; }

IMD_API int imd_prof_read(int max, const char** tags, float* ms) {
  if (g_prof.count < 2) return 0;
  if (cudaEventSynchronize(g_prof.ev[g_prof.count - 1]) != cudaSuccess) return -1;
  const int n = g_prof.count - 1;
  for (int i = 0; i < n && i < max; ++i) {
    float t = 0;
    if (cudaEventElapsedTime(&t, g_prof.ev[i], g_prof.ev[i + 1]) != cudaSuccess) return -1;
    tags[i] = g_prof.tag[i + 1];
    ms[i] = t;
  }
  return n;
}

}  // extern "C"
