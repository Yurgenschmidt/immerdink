/* Benchmark-only entry points of a libimmerdink built with -DIMMERDINK_PROFILE=ON
 * (src/profile.cpp). Not part of the public API (include/immerdink/immerdink.h).
 *
 *   imd_prof_enable(on)   on != 0: start recording phase marks (creates the event pool on
 *                         first use; clears earlier marks); 0: stop. Returns 0, or 1 if
 *                         the events could not be created.
 *   imd_prof_reset()      forget recorded marks (call before each profiled imd_contract).
 *   imd_prof_read(max, tags, ms)
 *                         synchronizes on the last mark, then writes up to `max` segments:
 *                         tags[i] = the phase that ended at mark i+1, ms[i] = its time.
 *                         Returns the number of segments (marks - 1), or -1 on error.
 *   imd_prof_plan_dims(plan, out[7])
 *                         post-swap GEMM view of a plan: m, n, k, b, n_hat, k_hat, path
 *                         (0 noop, 1 gemm, 2 direct). Returns 0.
 *   imd_prof_plan_k4(plan, &config)
 *                         the K4 configuration (IMD_K4_*) the plan runs, -1 without K4.
 *   imd_prof_set_k4_config(config)
 *                         plans created afterwards run K4 configuration `config`; -1
 *                         restores the plan-time rule / k4_tune. Returns 0, 1 if invalid.
 * Process-global, unsynchronized: one benchmark thread only. */
#ifndef IMMERDINK_PROFILE_HOOKS_H
#define IMMERDINK_PROFILE_HOOKS_H

#include <stdint.h>

#include "immerdink/immerdink.h"

#ifdef __cplusplus
extern "C" {
#endif

IMD_API int imd_prof_enable(int on);
IMD_API void imd_prof_reset(void);
IMD_API int imd_prof_read(int max, const char** tags, float* ms);
IMD_API int imd_prof_plan_dims(imd_plan_t plan, int64_t* out);
IMD_API int imd_prof_plan_k4(imd_plan_t plan, int32_t* config);
IMD_API int imd_prof_set_k4_config(int32_t config);

#ifdef __cplusplus
}
#endif

#endif
