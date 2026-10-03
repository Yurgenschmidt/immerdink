/* immerdink host query API (docs/DESIGN.md §6a): libimmerdink_host, CUDA-free.
 *
 * The lowering (§5), the legality checks (§5 rows 1-5, §3.2 RN acceptance) and the memory
 * model (§4a) of immerdink, as pure host functions. They need no GPU, no CUDA runtime and
 * no driver; they allocate no device memory, have no global state and are thread-safe.
 * libimmerdink (the CUDA library) links this library and sizes its plans and workspaces
 * with exactly these functions, so the numbers reported here are the ones the kernels use.
 *
 * Plain C ABI (ctypes-friendly: int32/int64/uint64/pointers, no bitfields, no unions).
 * This header is also included by immerdink/immerdink.h, which adds the CUDA entry points.
 * Link: -limmerdink_host (alone, without CUDA), or -limmerdink (which needs it). */
#ifndef IMMERDINK_IMMERDINK_HOST_H
#define IMMERDINK_IMMERDINK_HOST_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define IMD_API __declspec(dllexport)
#else
#  define IMD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Version of the host query API (§6a "ABI stability"). It grows by one whenever a function
 * or a trailing struct field is added; nothing that exists changes meaning or layout. */
#define IMD_HOST_API_VERSION 2  /* 2: imd_lowering_t fused, kernel_launches (§6a v1.4) */

typedef enum {
  IMD_STATUS_SUCCESS = 0,
  IMD_STATUS_INVALID_VALUE = 1,
  IMD_STATUS_NOT_SUPPORTED = 2,
  IMD_STATUS_ALIASING = 3,
  IMD_STATUS_INSUFFICIENT_WORKSPACE = 4,
  IMD_STATUS_MISALIGNED = 5,
  IMD_STATUS_CUDA_ERROR = 6,
  IMD_STATUS_ALLOC_FAILED = 7 /* plan creation: host allocation, or the device allocation of
                                 IMD_VEC_RESIDENT, failed; host queries: host allocation */
} imd_status_t;

/* Options: rounding (§3.3) and mode (§4 DIRECT path). */
enum { IMD_RN = 0, IMD_TRUNC = 1 };
enum { IMD_EMULATE = 0, IMD_DIRECT = 1 };
/* Where the scatter vectors live (DESIGN v1.2 §4 K0, §4a, §6):
 *   IMD_VEC_GENERATE  (default) generated on the device by every imd_contract into the
 *                     workspace; the plan owns no device or pinned memory;
 *   IMD_VEC_RESIDENT  built once by imd_plan_create in plan-owned device memory
 *                     (imd_plan_device_bytes); no workspace vec region; the plan must
 *                     outlive the GPU work of its calls. */
enum { IMD_VEC_GENERATE = 0, IMD_VEC_RESIDENT = 1 };

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
  int32_t residue_group; /* g, 0 = N; the workspace/speed knob (§4a, §6a) */
  int32_t rounding;      /* IMD_RN (default) | IMD_TRUNC     */
  int32_t mode;          /* IMD_EMULATE | IMD_DIRECT         */
  int32_t vec_mode;      /* IMD_VEC_GENERATE (0) | IMD_VEC_RESIDENT (v1.2) */
} imd_options_t;

/* The host query API's version: IMD_HOST_API_VERSION of the library actually loaded. */
IMD_API uint32_t imd_host_api_version(void);

/* Exact workspace bytes of DESIGN §4a (the bytes imd_contract needs). SUCCESS,
 * INVALID_VALUE or NOT_SUPPORTED (§5 rows 1-5); never applies the §3.2 RN acceptance rule. */
IMD_API imd_status_t imd_workspace_size(const imd_tensor_t* A, const imd_tensor_t* B,
                                        const imd_tensor_t* C, const imd_options_t* options,
                                        size_t* bytes);

/* The §4a table as data. Region indices (§4a order): */
enum {
  IMD_REGION_SA = 0,  /* int16 shifts of A rows        */
  IMD_REGION_SB = 1,  /* int16 shifts of B columns     */
  IMD_REGION_FA = 2,  /* u8 non-finite flags, A rows   */
  IMD_REGION_FB = 3,  /* u8 non-finite flags, B cols   */
  IMD_REGION_VEC = 4, /* int64 scatter vectors         */
  IMD_REGION_RA = 5,  /* int8 residues of A, one group */
  IMD_REGION_RB = 6,  /* int8 residues of B, one group */
  IMD_REGION_W = 7,   /* u8 products, all N planes     */
  IMD_REGION_COUNT = 8
};
typedef struct {
  uint64_t offset; /* bytes from the workspace base; 256-aligned */
  uint64_t bytes;  /* 0: the region is absent for this call     */
} imd_region_t;

/* Same status as imd_workspace_size. On SUCCESS: regions[r] for r < min(capacity,
 * IMD_REGION_COUNT) (regions may be NULL when capacity = 0), *num_regions =
 * IMD_REGION_COUNT, *total = imd_workspace_size; num_regions and total may be NULL. */
IMD_API imd_status_t imd_workspace_layout(const imd_tensor_t* A, const imd_tensor_t* B,
                                          const imd_tensor_t* C, const imd_options_t* options,
                                          imd_region_t* regions, int32_t capacity,
                                          int32_t* num_regions, size_t* total);

/* Static name of a region index ("sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"), or NULL. */
IMD_API const char* imd_region_name(int32_t region);

/* What imd_plan_create would allocate for these arguments, without a plan or a GPU:
 *   device_bytes: imd_plan_device_bytes of the plan (§4a "plan device bytes");
 *   host_bytes:   the plan's host payload (§6a): its record, its copies of the tensor
 *                 descriptors, the lowering tables and the K4 launch parameters;
 *                 allocator overhead is not included.
 * Returns the plan-time status (what imd_plan_create returns barring CUDA and allocation
 * failures): §5 rows 1-5, then §3.2 RN acceptance (kappa >= T -> NOT_SUPPORTED). Either
 * pointer may be NULL; both are written only on SUCCESS. */
IMD_API imd_status_t imd_plan_bytes(const imd_tensor_t* A, const imd_tensor_t* B,
                                    const imd_tensor_t* C, const imd_options_t* options,
                                    size_t* device_bytes, size_t* host_bytes);

/* What a plan runs per imd_contract (§3.7, §4):
 *   NOOP     empty output (b m n = 0): nothing;
 *   EMULATE  the emulated pipeline K0-K5 (EMULATE mode, k >= 2);
 *   DIRECT   K0 and the native FP64 kernel (DIRECT mode, or K-empty: k = 1);
 *   SCALE    k = 0 (a zero-extent contracted label): K0 and one kernel, C <- beta C.
 * (A call with alpha = 0 also runs only K0 and the beta kernel; that is per call.) */
enum { IMD_DISPATCH_NOOP = 0, IMD_DISPATCH_EMULATE = 1, IMD_DISPATCH_DIRECT = 2,
       IMD_DISPATCH_SCALE = 3 };

/* The lowering summary (§5 GEMM view, §4 dispatch and launch structure, §4a sizes), the
 * input of a later performance model. Versioned by struct_size (§6a): the caller sets
 * struct_size = sizeof(imd_lowering_t); the library writes only the fields that fit. */
typedef struct {
  uint32_t struct_size;  /* in: sizeof(imd_lowering_t) as compiled by the caller */
  int32_t status;        /* the plan-time status, as returned                   */
  int32_t lowered;       /* 1: §5 and §4a succeeded (imd_workspace_size would return
                            SUCCESS); the fields below up to workspace_bytes are valid */
  int32_t dispatch;      /* IMD_DISPATCH_*                                         */
  int32_t swapped;       /* §5 step 8 role swap (GEMM computes C^T = B^T A^T)      */
  int32_t num_moduli;    /* N                                                      */
  int32_t residue_group; /* g, normalized to [1, N]                                */
  int32_t num_groups;    /* residue groups ceil(N / g) on EMULATE, else 0          */
  int64_t m, n, k, b;    /* post-swap GEMM view (§1.1, §5)                         */
  int64_t n_hat, k_hat;  /* ceil(n)_16, ceil(k)_16                                 */
  int64_t vec_entries;   /* int64 scatter-vector entries 2m + 2n + 2k + 3b (0 on NOOP) */
  int64_t k_chunks;      /* K4 K-chunks per group (§3.5) on EMULATE, else 0        */
  int64_t gemm_launches; /* K4 kernel launches per imd_contract on EMULATE, else 0 */
  uint64_t kappa;        /* ceil(sqrt(k)) for k >= 1, else 0 (§3.2)               */
  uint64_t workspace_bytes;   /* imd_workspace_size                               */
  uint64_t plan_device_bytes; /* imd_plan_bytes; valid only when status = SUCCESS  */
  uint64_t plan_host_bytes;   /* imd_plan_bytes; valid only when status = SUCCESS  */
  /* Version 2 (152 bytes; §4, §6a v1.4). Valid when lowered = 1, like k_chunks. */
  int32_t fused;           /* 1: EMULATE with the fused launch KF (K0 + K1 + the packs of
                              group 1), chosen by k_hat <= 512; else 0                */
  int32_t reserved;        /* 0                                                       */
  int64_t kernel_launches; /* kernel launches of one imd_contract with alpha != 0 (an
                              alpha = 0 call launches K0 (generate mode) + 1)          */
} imd_lowering_t;

/* Fills *out and returns the plan-time status (as imd_plan_bytes). INVALID_VALUE without
 * writing anything if out is NULL or out->struct_size < 136 (the version-1 size). With
 * lowered = 0 only struct_size, status and lowered are meaningful; the others are 0. */
IMD_API imd_status_t imd_lowering(const imd_tensor_t* A, const imd_tensor_t* B,
                                  const imd_tensor_t* C, const imd_options_t* options,
                                  imd_lowering_t* out);

/* Static, human-readable name of a status ("IMD_STATUS_SUCCESS", ...). */
IMD_API const char* imd_status_string(imd_status_t status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_HOST_H */
