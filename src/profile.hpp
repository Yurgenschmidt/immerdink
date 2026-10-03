// Opt-in per-phase timing for benchmarks (M4). Not part of the public API.
//
// Built only with -DIMMERDINK_PROFILE=ON (bench/CMakeLists.txt sets it). In a production
// build IMD_PROF expands to nothing, so the kernel sequence and host path are unchanged.
// In a profiling build each phase boundary calls prof_mark, which returns immediately
// unless a benchmark switched profiling on (imd_prof_enable); then it records a CUDA event
// on the stream with a static tag. The time of a tag is the event time since the previous
// mark. The bench reads the segments after synchronizing (src/profile_hooks.h).
// Process-global and unsynchronized: one benchmark thread only.
#pragma once

#include <cuda_runtime_api.h>

#if defined(IMMERDINK_PROFILE) && IMMERDINK_PROFILE
namespace imd::detail {
void prof_mark(cudaStream_t stream, const char* tag);
}  // namespace imd::detail
#  define IMD_PROF(stream, tag) ::imd::detail::prof_mark((stream), (tag))
#else
#  define IMD_PROF(stream, tag) ((void)0)
#endif
