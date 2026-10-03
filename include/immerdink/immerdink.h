/* immerdink: TBLIS-style FP64 tensor contractions via Ozaki scheme II on INT8 tensor cores.
 *
 *   C[I,J,L] <- alpha * sum_K A[I,K,L] * B[K,J,L] + beta * C[I,J,L]
 *
 * C ABI of docs/DESIGN.md §6 (ctypes-friendly: plain structs, int32/int64/pointers only).
 * Semantics, statuses and their precedence: DESIGN §5 and §6; workspace: DESIGN §4a. */
#ifndef IMMERDINK_IMMERDINK_H
#define IMMERDINK_IMMERDINK_H

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h> /* cudaStream_t */

/* The CUDA-free part of the ABI (libimmerdink_host, DESIGN §6a): statuses, options,
 * imd_tensor_t, imd_options_t, imd_workspace_size, imd_status_string and the host queries
 * (imd_workspace_layout, imd_plan_bytes, imd_lowering). */
#include "immerdink/immerdink_host.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct imd_handle_st* imd_handle_t;
typedef struct imd_plan_st* imd_plan_t;

/* Library context (DESIGN §6 lifecycle). */
IMD_API imd_status_t imd_create(imd_handle_t* handle);
IMD_API imd_status_t imd_destroy(imd_handle_t handle);

/* Lowering, legality (§5 rows 1-5), RN acceptance (§3.2), the K4 GEMM parameters and
 * (IMD_VEC_RESIDENT) the plan's device scatter vectors. Thread safety and lifetime: DESIGN
 * §6. With IMD_VEC_GENERATE a plan may be destroyed once every imd_contract using it has
 * returned; with IMD_VEC_RESIDENT synchronize those calls' streams first. */
IMD_API imd_status_t imd_plan_create(imd_handle_t handle, const imd_tensor_t* A,
                                     const imd_tensor_t* B, const imd_tensor_t* C,
                                     const imd_options_t* options, imd_plan_t* plan);
IMD_API imd_status_t imd_plan_destroy(imd_plan_t plan);

/* Device bytes the plan owns (DESIGN §4a, v1.2): 8 (2m + 2n + 2k + 3b) with
 * IMD_VEC_RESIDENT and a non-empty output, else 0. Equal to imd_plan_bytes (immerdink_host.h)
 * for the same arguments, which answers without a plan or a GPU. */
IMD_API imd_status_t imd_plan_device_bytes(imd_plan_t plan, size_t* bytes);

/* Enqueue the contraction on `stream`: no host synchronization, no allocation, no write to
 * the plan (one plan may serve several threads and streams at once). The workspace
 * (ws_bytes >= imd_workspace_size, 256-byte aligned) belongs to at most one in-flight
 * call. */
IMD_API imd_status_t imd_contract(imd_plan_t plan, double alpha, const double* A,
                                  const double* B, double beta, double* C, void* ws,
                                  size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_H */
