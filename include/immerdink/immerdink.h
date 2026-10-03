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

#if defined(_WIN32)
#  define IMD_API __declspec(dllexport)
#else
#  define IMD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  IMD_STATUS_SUCCESS = 0,
  IMD_STATUS_INVALID_VALUE = 1,
  IMD_STATUS_NOT_SUPPORTED = 2,
  IMD_STATUS_ALIASING = 3,
  IMD_STATUS_INSUFFICIENT_WORKSPACE = 4,
  IMD_STATUS_MISALIGNED = 5,
  IMD_STATUS_CUDA_ERROR = 6
} imd_status_t;

/* Options: rounding (§3.3) and mode (§4 DIRECT path). */
enum { IMD_RN = 0, IMD_TRUNC = 1 };
enum { IMD_EMULATE = 0, IMD_DIRECT = 1 };

/* A strided FP64 tensor view. extent, stride (in elements, any sign) and label each have
 * `rank` entries; the pointers may be NULL when rank = 0. Labels are arbitrary int32 ids;
 * equal ids across A, B and C denote the same index (§5). */
typedef struct {
  int32_t rank;
  const int64_t* extent;
  const int64_t* stride; /* elements */
  const int32_t* label;
} imd_tensor_t;

typedef struct {
  int32_t num_moduli;    /* N, 2..20                        */
  int32_t residue_group; /* g, 0 = N                         */
  int32_t rounding;      /* IMD_RN (default) | IMD_TRUNC     */
  int32_t mode;          /* IMD_EMULATE | IMD_DIRECT         */
} imd_options_t;

typedef struct imd_handle_st* imd_handle_t;
typedef struct imd_plan_st* imd_plan_t;

/* Library context. Lifecycle functions are not listed in DESIGN §6 (see docs/NOTES-M0.md). */
IMD_API imd_status_t imd_create(imd_handle_t* handle);
IMD_API imd_status_t imd_destroy(imd_handle_t handle);

/* Exact workspace bytes of DESIGN §4a. A pure host function: no GPU, no allocation. */
IMD_API imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                        const imd_tensor_t* C, const imd_options_t* options,
                                        size_t* bytes);

IMD_API imd_status_t imd_plan_create(imd_handle_t handle, const imd_tensor_t* A,
                                     const imd_tensor_t* B, const imd_tensor_t* C,
                                     const imd_options_t* options, imd_plan_t* plan);
IMD_API imd_status_t imd_plan_destroy(imd_plan_t plan);

/* Enqueue the contraction on `stream`: no host synchronization, no allocation. The
 * workspace (ws_bytes >= imd_workspace_size, 256-byte aligned) belongs to at most one
 * in-flight call. */
IMD_API imd_status_t imd_contract(imd_plan_t plan, double alpha, const double* A,
                                  const double* B, double beta, double* C, void* ws,
                                  size_t ws_bytes, cudaStream_t stream);

/* Static, human-readable name of a status ("IMD_STATUS_SUCCESS", ...). */
IMD_API const char* imd_status_string(imd_status_t status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_H */
