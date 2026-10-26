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
 *   imd_prof_plan_k4_backend(plan, &backend)
 *                         issue 22: the K4 backend the plan tries first (0 mma.sync, 1 Hopper wgmma), -1 without K4.
 *   imd_prof_set_k4_backend(backend)
 *                         issue 22: plans created afterwards take backend 0 (mma.sync), 1 (wgmma, where the device class has it and
 *                         the build; 1 also lifts the profitability floor) or -1 (the device class's rule). Returns 0, 1 if invalid.
 *   imd_prof_k4_wgmma_counts(reset, out[2])
 *                         issue 22: K4 launches the Hopper backend served / declined since the last reset.
 *   imd_prof_set_k5_elems(elems)
 *                         later calls run K5 (j-bar fastest) with 1, 2 or 4 elements per
 *                         thread; 0 restores the launch rule (A5). Returns 0, 1 if invalid.
 *   imd_prof_set_k5_crt(variant)
 *                         issue 21: later calls run K5's CRT with the sums in binary64 limbs (1) or
 *                         the integer running CRT (0, default). Returns 0, 1 if invalid.
 *   imd_prof_set_seq(bits)
 *                         issue 21: later calls run the instruction-sequence set `bits` (1 E1 emitter, 2 Q1 quantize,
 *                         4 F1 finish; host/seq_config.hpp) instead of the device class's; -1 restores it.
 *                         Returns 0, 1 if invalid.
 *   imd_prof_plan_seq(plan, &bits)
 *                         issue 21: the sequence set the plan's launches run now (the forced one, else the device
 *                         class's). Returns 0, 1 for a null plan.
 *   imd_prof_get_k5_crt()
 *                         issue 21: K5's CRT variant in force (0 integer, 1 binary64 limbs).
 *   imd_prof_kernel_occupancy(kernel, out[4])
 *                         v1.10: CTAs per SM, registers, shared memory bytes and threads of
 *                         kernel IMD_PERF_KERNEL_* (immerdink_host.h). Returns 0, 1 on error.
 *   imd_prof_kernel_occupancy_w(kernel, w, out[4])
 *                         v1.16: the same for one pack-emitter word count w = 2, 3 of a
 *                         pack kernel (KF16, KF4, KL8, KL4, PACK); w = 0: the row above.
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
IMD_API int imd_prof_plan_k4_backend(imd_plan_t plan, int32_t* backend);
IMD_API int imd_prof_set_k4_backend(int32_t backend);
IMD_API int imd_prof_k4_wgmma_counts(int32_t reset, int64_t* out);
IMD_API int imd_prof_set_k5_elems(int32_t elems);
IMD_API int imd_prof_set_k5_crt(int32_t variant);
IMD_API int imd_prof_set_seq(int32_t bits);
IMD_API int imd_prof_plan_seq(imd_plan_t plan, int32_t* bits);
IMD_API int imd_prof_get_k5_crt(void);
IMD_API int imd_prof_kernel_occupancy(int32_t kernel, int32_t* out);
IMD_API int imd_prof_kernel_occupancy_w(int32_t kernel, int32_t w, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif
