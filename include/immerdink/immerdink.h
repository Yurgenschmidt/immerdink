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

/* Device status (issue 18, docs/NOTES-18.md). imd_device_info classifies the CURRENT device (the one
 * cudaGetDevice reports, as imd_plan_create does): the device class, which table set it gets and how
 * well it matches (imd_device_select), and what kind of code every kernel of the library runs as
 * there (cudaFuncGetAttributes on each: this loads them all, once per device, and the answer is
 * cached). info->status says whether plans can be made (IMD_PLAN_UNSUPPORTED: no) and how (NATIVE,
 * APPROXIMATE, RECOMPILED); failing_kernel and cuda_error say why not. SUCCESS, INVALID_VALUE
 * (NULL, or info->struct_size < sizeof(imd_device_info_t)), CUDA_ERROR (the device could not be
 * examined). The handle is only checked for NULL. */
IMD_API imd_status_t imd_device_info(imd_handle_t handle, imd_device_info_t* info);
/* The same for a plan: the device and table set it was made with, plus its K4 configuration
 * (info->k4_config). A plan is never UNSUPPORTED (imd_plan_create returns NOT_SUPPORTED instead).
 * A plan of an empty GEMM touches no CUDA: its info is zero (only struct_size set). Only imd_plan_create
 * checks the device: imd_pack_create and imd_contract_packed on an unusable device fail at launch with
 * the raw CUDA error. Both functions keep the caller's struct_size field. */
IMD_API imd_status_t imd_plan_info(imd_plan_t plan, imd_device_info_t* info);

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

/* ---- Packed operands (DESIGN v1.7 §6b) --------------------------------------------------
 * imd_pack writes one operand X, scaled and reduced to all N residue planes (the §6b.4
 * layout, immerdink_host.h), into caller-owned device memory; plans created against the
 * pack (imd_plan_create_packed) read it with imd_contract_packed instead of scaling and
 * packing X themselves, and adopt its canonical orders. */
typedef struct imd_pack_s* imd_pack_t; /* immutable host object: the key and the layout */

/* imd_pack_layout's statuses (immerdink_host.h); ALLOC_FAILED (the host object). Copies the
 * descriptor (O(rank)); no device work. The handle is immutable: any number of threads may
 * pass it to imd_pack, imd_plan_create_packed and imd_contract_packed concurrently; it may
 * be destroyed once every call that was passed it has returned (no GPU work reads it). */
IMD_API imd_status_t imd_pack_create(imd_handle_t handle, const imd_pack_desc_t* desc,
                                     imd_pack_t* pack);
IMD_API imd_status_t imd_pack_destroy(imd_pack_t pack);
/* The packed buffer's bytes (§6b.4) and the workspace imd_pack needs (either may be NULL).
 * pack_bytes is nondecreasing in N from host API 13 on (DESIGN §6a F2); pack_ws_bytes is not
 * covered by that promise. */
IMD_API imd_status_t imd_pack_bytes(imd_pack_t pack, size_t* pack_bytes, size_t* pack_ws_bytes);
/* Pack X (the FP64 operand the descriptor describes) into `packed` on `stream`. Checks, in
 * order: null pointers (pack; X and packed when pack_bytes > 0; ws when pack_ws_bytes > 0)
 * -> INVALID_VALUE; pack_bytes = 0 -> SUCCESS, nothing touched; ALIASING (packed vs X; ws vs
 * X or packed); MISALIGNED (X 8 bytes, packed and ws 256); packed_bytes < pack_bytes or
 * ws_bytes < pack_ws_bytes -> INSUFFICIENT_WORKSPACE. Then it only launches kernels (two):
 * no allocation, no synchronization, capture-safe. Every byte of [packed, packed +
 * pack_bytes) is written, and no byte beyond pack_bytes is (guaranteed from host API 13 on,
 * DESIGN §6a F3). The caller orders the packed buffer: contracts that read it after this
 * call (stream order or an event), a re-pack or free after them. */
IMD_API imd_status_t imd_pack(imd_pack_t pack, const double* X, void* packed,
                              size_t packed_bytes, void* ws, size_t ws_bytes,
                              cudaStream_t stream);

/* imd_plan_create with packed operands (NULL pack: that operand is packed in-line; both NULL:
 * imd_plan_create): its statuses, then PACK_MISMATCH (the pack does not fit the plan:
 * imd_lowering_packed's pack_mismatch says why), then the §3.2 RN acceptance. The plan
 * adopts the packs' canonical orders and copies their keys; it keeps no reference to the
 * handles. options.num_moduli and rounding must be the packs'. */
IMD_API imd_status_t imd_plan_create_packed(imd_handle_t handle, const imd_tensor_t* A,
                                            const imd_tensor_t* B, const imd_tensor_t* C,
                                            const imd_options_t* options, imd_pack_t packA,
                                            imd_pack_t packB, imd_plan_t* plan);

/* One operand of imd_contract_packed. */
typedef struct {
  const double* data;   /* the FP64 operand (always: K5's native path for flagged outputs,
                           DIRECT, and the in-line operand)                                 */
  const void* packed;   /* imd_pack's output for a packed slot of the plan, else NULL       */
  size_t packed_bytes;  /* bytes at `packed`                                                */
  imd_pack_t pack;      /* packed slot: a handle whose key equals the plan's (any handle of
                           the same pack descriptor); in-line slot: NULL                    */
} imd_operand_t;

/* imd_contract for a plan from imd_plan_create_packed. Checks, in order:
 *   1. A or B NULL -> INVALID_VALUE; an in-line slot with packed or pack non-NULL, or a
 *      packed slot whose pack is NULL or has another key -> PACK_MISMATCH;
 *   2. imd_contract's null rule with `data`, plus a packed slot's `packed` when it is read
 *      (alpha != 0, k != 0, the emulated GEMM, pack_bytes > 0) -> INVALID_VALUE;
 *   3. an empty output -> SUCCESS, nothing touched;
 *   4. ALIASING: C vs A.data, B.data and the packed buffers [packed, packed + pack_bytes);
 *      the workspace vs all of those and C (packed buffers may overlap each other and the
 *      FP64 operands: all are only read);
 *   5. MISALIGNED (packed buffers 256 bytes);
 *   6. ws_bytes < the packed plan's workspace size (imd_workspace_size_packed), or a
 *      non-NULL packed with packed_bytes < pack_bytes -> INSUFFICIENT_WORKSPACE.
 * Then as imd_contract (no allocation, no synchronization, capture-safe); the packed
 * buffers are only read, so any number of in-flight calls may read one. imd_contract on a
 * plan with a packed slot returns PACK_MISMATCH. */
IMD_API imd_status_t imd_contract_packed(imd_plan_t plan, double alpha, const imd_operand_t* A,
                                         const imd_operand_t* B, double beta, double* C,
                                         void* ws, size_t ws_bytes, cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_H */
