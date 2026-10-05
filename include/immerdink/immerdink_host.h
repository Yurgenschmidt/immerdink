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
#define IMD_HOST_API_VERSION 6  /* 2: imd_lowering_t fused, kernel_launches (§6a v1.4);
                                   3: K4 configurations, imd_options_t.k4_tune,
                                      imd_lowering_t sm_count, k4_config, k4_tune_bytes (v1.5);
                                   4: imd_lowering_t fused_large (v1.6);
                                   5: packed operands (§6b v1.7): IMD_STATUS_PACK_MISMATCH,
                                      imd_pack_layout, imd_pack_desc_for, imd_*_packed,
                                      imd_lowering_t packed, pack_mismatch;
                                   6: the performance model (§6c v1.8): imd_perf_* */

typedef enum {
  IMD_STATUS_SUCCESS = 0,
  IMD_STATUS_INVALID_VALUE = 1,
  IMD_STATUS_NOT_SUPPORTED = 2,
  IMD_STATUS_ALIASING = 3,
  IMD_STATUS_INSUFFICIENT_WORKSPACE = 4,
  IMD_STATUS_MISALIGNED = 5,
  IMD_STATUS_CUDA_ERROR = 6,
  IMD_STATUS_ALLOC_FAILED = 7, /* plan creation: host allocation, or the device allocation of
                                  IMD_VEC_RESIDENT, failed; host queries: host allocation */
  IMD_STATUS_PACK_MISMATCH = 8 /* v1.7 (§6b): a packed operand does not fit the plan (its
                                  compatibility key differs; imd_lowering_t pack_mismatch
                                  says why), or packed pointers / handles that do not match
                                  the plan's packed slots */
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

/* K4 tile configurations (DESIGN v1.5 §4): threadblock tile M x N (x 64), all bit-identical.
 * A plan takes the one chosen by the plan-time rule (imd_lowering_t k4_config), or, with
 * k4_tune = IMD_K4_TUNE_ON, the fastest one timed at plan creation. */
enum { IMD_K4_128x256 = 0, IMD_K4_128x128 = 1, IMD_K4_64x128 = 2, IMD_K4_64x64 = 3,
       IMD_K4_CONFIG_COUNT = 4 };
/* imd_options_t.k4_tune (v1.5): plan-time autotuning of the K4 configuration. OFF (default):
 * the rule. ON: imd_plan_create times every configuration on the plan's device (group 1's
 * K4 launches, a few short batches each, on temporary device memory of imd_lowering_t
 * k4_tune_bytes, freed before it returns) and keeps the fastest; with k4_tune_bytes = 0 it
 * keeps the rule's choice. */
enum { IMD_K4_TUNE_OFF = 0, IMD_K4_TUNE_ON = 1 };

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
  int32_t k4_tune;       /* IMD_K4_TUNE_OFF (0) | IMD_K4_TUNE_ON (v1.5)     */
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
                              group 1), chosen by k_hat <= 1024 (512 before v1.5)    */
  int32_t reserved;        /* 0                                                       */
  int64_t kernel_launches; /* kernel launches of one imd_contract with alpha != 0 (an
                              alpha = 0 call launches K0 (generate mode) + 1)          */
  /* Version 3 (168 bytes; §4, §6a v1.5). Valid when lowered = 1, like k_chunks. */
  int32_t sm_count;        /* IN as well as out: the SM count of the target device for the
                              K4 rule; 0 = 128 (the RTX 4090 it was fitted on). Out: the
                              count used                                                */
  int32_t k4_config;       /* IMD_K4_*: the plan-time rule's K4 configuration for sm_count
                              SMs (IMD_K4_128x256 when no emulated GEMM runs); a plan with
                              k4_tune = IMD_K4_TUNE_ON may run another one               */
  uint64_t k4_tune_bytes;  /* device bytes imd_plan_create allocates temporarily to time the
                              K4 configurations if k4_tune = IMD_K4_TUNE_ON (whatever this
                              call's option); 0: no tuning would run (no emulated GEMM, or
                              more than 256 MiB: the rule's choice is kept)               */
  /* Version 4 (176 bytes; §4, §6a v1.6). Valid when lowered = 1, like k_chunks. */
  int32_t fused_large;     /* bit 0 (1) / bit 1 (2): the GEMM's A / B (post-swap) is scaled and
                              packed for group 1 by the fused large launch KL (k_hat > 1024,
                              EMULATE); 0 otherwise. kernel_launches counts it            */
  int32_t reserved4;       /* 0                                                         */
  /* Version 5 (184 bytes; §6b v1.7). */
  int32_t packed;          /* bit 0 / bit 1: the CALLER's A / B is read from a packed operand
                              (imd_lowering_packed; 0 from imd_lowering)                    */
  int32_t pack_mismatch;   /* 0, or the IMD_PACK_MISMATCH_* reason of a PACK_MISMATCH status */
} imd_lowering_t;
enum { IMD_FUSED_LARGE_A = 1, IMD_FUSED_LARGE_B = 2 };

/* Fills *out and returns the plan-time status (as imd_plan_bytes). INVALID_VALUE without
 * writing anything if out is NULL or out->struct_size < 136 (the version-1 size). With
 * lowered = 0 only struct_size, status and lowered are meaningful; the others are 0.
 * out->sm_count is read (as an input) only when out->struct_size >= 168. */
IMD_API imd_status_t imd_lowering(const imd_tensor_t* A, const imd_tensor_t* B,
                                  const imd_tensor_t* C, const imd_options_t* options,
                                  imd_lowering_t* out);

/* ---- Packed operands (DESIGN v1.7 §6b, host API version 5) ------------------------------
 *
 * A packed operand X holds, per row, the shift sigma and the non-finite flag and the int8
 * residues of ALL N moduli over the whole canonical K, in the §6b.4 byte layout (format 1):
 *   sig   int16 sigma of row (lambda, rho) at index lambda r + rho
 *   flag  u8 flag, same index
 *   res   int8 residue of plane l, row (lambda, rho), k-bar at l b r k_hat + (lambda r + rho)
 *         k_hat + k-bar; zero K padding; a zeroed tail of (ceil(r)_16 - r) k_hat bytes
 * regions at 256-aligned offsets, every byte written (gaps 0). imd_pack (immerdink.h) writes
 * it; plans built against it read it instead of scaling and packing X themselves. The pack
 * fixes its canonical orders of K, L (batch) and R (X's rows); a plan adopts them.
 * Compatibility: a key of int64 words (format, N, rounding, b, r, k, then per class L, R, K
 * the fused modes (extent, stride in X) fused over X's strides alone), compared word for
 * word. */
#define IMD_PACK_FORMAT 1
enum { IMD_PACK_ORDER_AUTO = 0, IMD_PACK_ORDER_GIVEN = 1 };
/* imd_lowering_t.pack_mismatch: the first differing part, checked per packed operand (the
 * caller's A, then B) in this order, then ORDER, then the keys' extents and strides. */
enum {
  IMD_PACK_MISMATCH_NONE = 0,
  IMD_PACK_MISMATCH_FORMAT = 1,   /* another IMD_PACK_FORMAT                                  */
  IMD_PACK_MISMATCH_MODULI = 2,   /* the pack's N != options.num_moduli                       */
  IMD_PACK_MISMATCH_ROUNDING = 3, /* the pack's rounding != options.rounding                  */
  IMD_PACK_MISMATCH_CLASSES = 4,  /* the pack's labels do not map onto X's by mode position, or
                                     the plan's K / L / X-row classes differ from the pack's
                                     (incl. a K label only in the partner)                    */
  IMD_PACK_MISMATCH_EXTENTS = 5,  /* b, r, k or a fused mode's extent differs                 */
  IMD_PACK_MISMATCH_STRIDES = 6,  /* a fused mode's stride in X differs                       */
  IMD_PACK_MISMATCH_ORDER = 7     /* both operands packed and their K or L orders differ      */
};

/* What to pack (§6b.5). X's labels fall into K (contracted: not in C), L (batch: in C and in
 * the partner) and R (rows: the rest). AUTO: K and L listed in any order, R the rest, each
 * class ordered by (|stride| after the diagonal merge, first position in X's labels). GIVEN:
 * every label of X listed once, each class in canonical order (first fastest). */
typedef struct {
  uint32_t struct_size;          /* sizeof(imd_pack_desc_t) as the caller compiled it        */
  int32_t num_moduli;            /* N, 2..20 (no 0 = default: N is part of the key)          */
  int32_t rounding;              /* IMD_RN | IMD_TRUNC                                       */
  int32_t order;                 /* IMD_PACK_ORDER_AUTO | IMD_PACK_ORDER_GIVEN               */
  const imd_tensor_t* tensor;    /* X: extents, strides (elements), labels                   */
  int32_t num_contracted;        /* K                                                        */
  const int32_t* contracted;
  int32_t num_batch;             /* L                                                        */
  const int32_t* batch;
  int32_t num_rows;              /* GIVEN: R in order; AUTO: 0                               */
  const int32_t* rows;           /* AUTO: NULL                                               */
} imd_pack_desc_t;

/* The pack's sizes and layout (§6b.4). */
typedef struct {
  uint32_t struct_size;          /* in: sizeof(imd_pack_layout_t) as the caller compiled it  */
  int32_t status;                /* as returned                                              */
  int32_t format;                /* IMD_PACK_FORMAT                                          */
  int32_t num_moduli, rounding;  /* N, IMD_RN | IMD_TRUNC                                    */
  int32_t reserved;              /* 0                                                        */
  int64_t b, r, k, k_hat;        /* batches, rows, contraction length, ceil(k)_16            */
  uint64_t kappa;                /* ceil(sqrt(k)) (0 for k = 0)                              */
  imd_region_t sig, flag, res;   /* offsets and bytes in the packed buffer (0: empty pack)   */
  uint64_t pack_bytes;           /* §6b.4 total: the packed buffer's bytes                   */
  uint64_t pack_ws_bytes;        /* workspace imd_pack needs: 8 (r + k + b) aligned to 256
                                    (X's scatter vectors), 0 for an empty pack              */
  uint64_t key_digest;           /* FNV-1a of the key words: diagnostics only                */
} imd_pack_layout_t;

/* Which operands of a host query are packed: NULL member = that operand is packed in-line.
 * A NULL imd_packing_t* = neither. */
typedef struct {
  const imd_pack_desc_t* A;
  const imd_pack_desc_t* B;
} imd_packing_t;

/* Validate a pack descriptor and report its layout. SUCCESS; INVALID_VALUE (desc or out NULL,
 * a struct_size below this version's, N / rounding / order out of range, a listed label not
 * in X, listed twice or in two lists, GIVEN without every row label, AUTO with rows, a
 * repeated label of X with two extents, a negative extent, span / extent product / pack size
 * overflow); NOT_SUPPORTED (§6b.3: K empty after dropping extent-1 labels, or RN with k >= 2
 * and kappa >= T). With INVALID_VALUE only struct_size and status are written; with
 * NOT_SUPPORTED also the sizes and the layout (key_digest 0). A pack with b r k = 0 is valid
 * and has 0 bytes. */
IMD_API imd_status_t imd_pack_layout(const imd_pack_desc_t* desc, imd_pack_layout_t* out);

/* GIVEN descriptor of operand `operand` (0 = the caller's A, 1 = B) with exactly the classes
 * and canonical orders the unpacked plan of (A, B, C, options) uses: a pack made from it
 * reproduces that contraction bit for bit (flagged outputs included). out->struct_size must
 * be set (>= this version's size); out->tensor = A or B; the label lists are written to
 * labels[0, capacity) (K, then L, then R; out's pointers point into it). X's extent-1 labels
 * are listed after the canonical ones of their class (K: not in C; L: in C and the partner;
 * R: the rest). Statuses: imd_workspace_size's (§5 rows 1-5); INVALID_VALUE for operand not
 * 0 / 1, NULL out / labels with capacity > 0, or capacity below X's number of distinct
 * labels. A label of the plan's K that X does not carry (partner-only) is not listed: such
 * a plan rejects every pack of X (IMD_PACK_MISMATCH_CLASSES). */
IMD_API imd_status_t imd_pack_desc_for(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       int32_t operand, imd_pack_desc_t* out, int32_t* labels,
                                       int32_t capacity);

/* §4a with the regions sX, fX, RX of each packed operand's POST-swap role at 0 bytes (same
 * region order, bytes = 0 = absent). Statuses: imd_workspace_size's (§5 rows 1-5), then
 * each pack descriptor's (imd_pack_layout: INVALID_VALUE / NOT_SUPPORTED), then
 * PACK_MISMATCH (the plan-time compatibility check, no GPU needed). Never the §3.2 kappa
 * rule of the plan. */
IMD_API imd_status_t imd_workspace_size_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                               const imd_tensor_t* C,
                                               const imd_options_t* options,
                                               const imd_packing_t* packing, size_t* bytes);
IMD_API imd_status_t imd_workspace_layout_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                                 const imd_tensor_t* C,
                                                 const imd_options_t* options,
                                                 const imd_packing_t* packing,
                                                 imd_region_t* regions, int32_t capacity,
                                                 int32_t* num_regions, size_t* total);

/* imd_lowering of the packed plan (imd_plan_create_packed): the same statuses as
 * imd_workspace_size_packed, then the §3.2 RN acceptance (kappa >= T -> NOT_SUPPORTED, last).
 * workspace_bytes, fused, fused_large, kernel_launches and the plan bytes describe the packed
 * plan (no K1 or pack launch for a packed operand; KF / KL work on the in-line operand only);
 * version 5 adds packed (caller's operands) and pack_mismatch (the reason of PACK_MISMATCH,
 * with lowered = 1 and the summary fields valid as for NOT_SUPPORTED). */
IMD_API imd_status_t imd_lowering_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                         const imd_tensor_t* C, const imd_options_t* options,
                                         const imd_packing_t* packing, imd_lowering_t* out);

/* Static, human-readable name of a status ("IMD_STATUS_SUCCESS", ...). */
IMD_API const char* imd_status_string(imd_status_t status);

/* ---- Performance model (DESIGN v1.8 §6c, host API version 6) ------------------------------
 *
 * Predicted time of one imd_contract (imd_pack) per phase and in total, and a ranking of the
 * configurations a caller may choose (N, g, vec_mode, packed operands) inside a device-memory
 * budget, each with its exact footprint. Each phase time is linear in calibrated per-kernel
 * parameters:  t_p = sum_j param[class(p) * IMD_PERF_MAX_PARAMS + j] * x[p][j]  (ns; for the
 * K4 classes the max of the round-model terms j != 4 and the DRAM term j = 4), over
 * features x computed exactly from the plan's lowering (the table in §6c.1). A calibration is
 * a caller-owned plain struct (no handle, no allocation); NULL = the shipped one (RTX 4090,
 * sm_89). The calibration is read only by these functions, never by plan creation. */
#define IMD_PERF_CALIB_FORMAT 1     /* the calibration file format these functions read     */
#define IMD_PERF_MAX_PARAMS 6       /* parameters (features) per class                       */
#define IMD_PERF_PARAM_CAPACITY 96  /* imd_perf_calib_t.param slots: class * 6 + j            */
#define IMD_PERF_MAX_PHASES 16      /* imd_perf_t.phase capacity                              */

/* Kernel classes (one parameter set each; file key prefix in parentheses). */
enum {
  IMD_PERF_CLASS_K0 = 0,         /* (k0) K0 vectors, imd_pack's prep: launch, byte          */
  IMD_PERF_CLASS_KF = 1,         /* (kf) fused small launch: launch, byte, op, row, op_eps  */
  IMD_PERF_CLASS_KL = 2,         /* (kl) fused large launch, imd_pack: launch, byte, op,
                                    l2byte, op_eps                                          */
  IMD_PERF_CLASS_PACK = 3,       /* (pack) K2 / K3, groups 2..: launch, byte, op, op_eps    */
  IMD_PERF_CLASS_K4_128x256 = 4, /* (k4_128x256) K4: max(round model, byte): round, mac1,
                                    byte (slot 4), mac_ktail (slot 5, in the round model)   */
  IMD_PERF_CLASS_K4_128x128 = 5, /* (k4_128x128) round, mac1, mac2, byte, mac_ktail         */
  IMD_PERF_CLASS_K4_64x128 = 6,  /* (k4_64x128)  round, mac1, mac2, byte, mac_ktail         */
  IMD_PERF_CLASS_K4_64x64 = 7,   /* (k4_64x64)   round, mac1, mac2, mac3, byte, mac_ktail   */
  IMD_PERF_CLASS_K5 = 8,         /* (k5) CRT: launch, byte, op                              */
  IMD_PERF_CLASS_DIRECT = 9,     /* (direct) native FP64: launch, byte, flop                */
  IMD_PERF_CLASS_BETA = 10,      /* (beta) C <- beta C: launch, byte                        */
  IMD_PERF_CLASS_CALL = 11,      /* (call) per-call correction: call, launch (may be < 0)   */
  IMD_PERF_CLASS_COUNT = 12
};
/* Phases (imd_perf_phase_name: the tags of imd_bench's per-phase profile). */
enum {
  IMD_PERF_PHASE_K0 = 0,         /* "K0 vec"        */
  IMD_PERF_PHASE_KF = 1,         /* "KF fused"      */
  IMD_PERF_PHASE_KL_A = 2,       /* "KL A"          */
  IMD_PERF_PHASE_KL_B = 3,       /* "KL B"          */
  IMD_PERF_PHASE_PACK_A = 4,     /* "K2 pack A"     */
  IMD_PERF_PHASE_PACK_B = 5,     /* "K3 pack B"     */
  IMD_PERF_PHASE_K4 = 6,         /* "K4 gemm"       */
  IMD_PERF_PHASE_K5 = 7,         /* "K5 crt"        */
  IMD_PERF_PHASE_DIRECT = 8,     /* "DIRECT"        */
  IMD_PERF_PHASE_BETA = 9,       /* "K5 beta-only"  */
  IMD_PERF_PHASE_CALL = 10,      /* "call"          */
  IMD_PERF_PHASE_PACK_PREP = 11, /* "pack prep"     */
  IMD_PERF_PHASE_PACK_KL = 12,   /* "pack KL"       */
  IMD_PERF_PHASE_COUNT = 13
};
/* imd_perf_request_t.flags, imd_perf_rank_request_t.flags */
enum { IMD_PERF_BETA_NONZERO = 1, IMD_PERF_ALPHA_ZERO = 2 };

/* A calibration (§6c.2): the GPU it describes, the datasheet peaks of the bounds and the
 * parameters (ns per feature unit). Strings are NUL-terminated. */
typedef struct {
  uint32_t struct_size;           /* sizeof(imd_perf_calib_t) as the caller compiled it     */
  int32_t format;                 /* IMD_PERF_CALIB_FORMAT                                  */
  int32_t sm_count;               /* SMs of the calibrated GPU (the K4 rule and round model) */
  int32_t reserved;               /* 0                                                      */
  double peak_dram_bytes_per_s;   /* datasheet DRAM bandwidth                               */
  double peak_int8_ops_per_s;     /* datasheet dense int8 tensor-core ops                   */
  double peak_fp64_flops_per_s;   /* datasheet FP64 flops                                   */
  double l2_bytes;                /* L2 cache bytes: the K4 DRAM term counts a group's operand
                                     and plane bytes beyond it (§6c.1)                      */
  char gpu[64], date[32], version[64], clocks[128], notes[128];
  double param[IMD_PERF_PARAM_CAPACITY]; /* param[class * IMD_PERF_MAX_PARAMS + j]; 0 unused */
} imd_perf_calib_t;

/* The shipped calibration (RTX 4090, sm_89). out->struct_size must be set; SUCCESS. */
IMD_API imd_status_t imd_perf_calib_default(imd_perf_calib_t* out);
/* Parse calibration text (format 1: "key = value" lines, '#' comments; every parameter
 * required, no unknown or repeated key). INVALID_VALUE with *error_line = the 1-based line
 * (0: not tied to a line, e.g. a missing key); error_line may be NULL. */
IMD_API imd_status_t imd_perf_calib_parse(const char* text, size_t length, imd_perf_calib_t* out,
                                          int32_t* error_line);
/* imd_perf_calib_parse of a file's contents; an unreadable file is INVALID_VALUE, line 0. */
IMD_API imd_status_t imd_perf_calib_load(const char* path, imd_perf_calib_t* out,
                                         int32_t* error_line);
/* The canonical text of a calibration (every key; numbers in the shortest round-trip form,
 * so parse(format(c)) == c bit for bit). *length = its length without the NUL, always;
 * INSUFFICIENT_WORKSPACE if capacity <= *length (nothing useful written). */
IMD_API imd_status_t imd_perf_calib_format(const imd_perf_calib_t* calib, char* buf,
                                           size_t capacity, size_t* length);
/* Static names: a class's key prefix ("k0", ..., "call"), a parameter's name ("launch",
 * "byte", "op", "row", "l2byte", "round", "mac1".. "mac3", "flop", "call"; NULL for a slot
 * the class does not use) and a phase's name; NULL out of range. */
IMD_API const char* imd_perf_class_name(int32_t cls);
IMD_API const char* imd_perf_param_name(int32_t cls, int32_t j);
IMD_API const char* imd_perf_phase_name(int32_t phase);

/* The eps path (§6c.1, a property of the DATA, not of the plan). The packs (KF, KL, K2/K3,
 * imd_pack) quantize an element to mu 2^eps; when |a'| >= 2^53 (eps > 0) a residue costs one
 * more modular multiply, and a warp pays it for its whole tile (32 lanes x up to 16 elements)
 * if any element of the tile has eps > 0. eps_fraction = the fraction of the pack work that
 * takes that path. It is 0 whenever T < 2^53 (N <= 13: no |a'| reaches 2^53). The DEFAULT
 * (eps_fraction < 0) is a plan-only estimate for TYPICAL data, i.e. rows of comparable
 * entries such as the benchmark generator's (u - 1/2) exp(g / 2): with the margin
 * D = log2 T - log2(k) / 2 - 53 (T exact, not floor(log2 T)), the fraction of 512-element
 * tiles on the eps path rises from 0 at D = -2.5 to 1 at D = -1 (simulated: a tile's maximum
 * after the row's power-of-two scaling is about 1.6 bits above the row RMS margin; the
 * default is that linear ramp). Rows with one dominant entry have a fraction of about
 * 512 / k; the caller passes its own value then. Every prediction also reports eps_s, the
 * time of the eps terms at fraction 1, so a caller unsure of its data has the exact bracket
 * [time_s - f eps_s, time_s + (1 - f) eps_s] for the fraction f used (eps_fraction out). */

/* What to predict beyond the options. NULL = flags 0, k4_config -1, eps_fraction -1. */
typedef struct {
  uint32_t struct_size;  /* sizeof(imd_perf_request_t)                                      */
  int32_t flags;         /* IMD_PERF_BETA_NONZERO (C is read), IMD_PERF_ALPHA_ZERO          */
  int32_t k4_config;     /* -1: the plan-time rule for the calibration's sm_count; else IMD_K4_* */
  int32_t reserved;      /* 0                                                               */
  double eps_fraction;   /* < 0: the typical-data default; 0..1: the caller's (see above)   */
} imd_perf_request_t;

/* One phase: its features x (the model is sum_j param[cls * 6 + j] x[j]; K4: the max of the
 * terms j != 4 and the term j = 4; the op_eps features already carry the eps fraction), its
 * predicted time, and its MEMORY_MODEL §4 lower bound: bound_s = max(bytes / peak DRAM,
 * ops / peak). It has no struct_size: its layout (96 bytes) and IMD_PERF_MAX_PHASES are
 * FROZEN for host API version 6; a change would be a new struct and function names. */
typedef struct {
  int32_t phase;         /* IMD_PERF_PHASE_*                                                */
  int32_t cls;           /* IMD_PERF_CLASS_*                                                */
  int64_t launches;      /* kernel launches of the phase (0 for the call term)              */
  double time_s;         /* predicted                                                       */
  double bytes;          /* compulsory DRAM bytes                                           */
  double ops;            /* int8 ops (K4), FP64 flops (DIRECT), else 0                      */
  double bound_s;
  double x[IMD_PERF_MAX_PARAMS];
} imd_perf_phase_t;

typedef struct {
  uint32_t struct_size;  /* in: sizeof(imd_perf_t) as compiled by the caller                */
  int32_t status;        /* as returned; the fields below are filled only on SUCCESS        */
  int32_t dispatch;      /* IMD_DISPATCH_* (imd_perf_predict_pack: IMD_DISPATCH_EMULATE)    */
  int32_t k4_config;     /* the K4 configuration predicted (IMD_K4_128x256 without K4)      */
  int32_t sm_count;      /* the calibration's                                               */
  int32_t num_phases;    /* entries of phase[] in launch order (the call term last)         */
  double time_s;         /* predicted time: the sum of the phases                           */
  double bound_s;        /* sum of the phase bounds                                         */
  double flops_eq;       /* 2 b m n k (imd_pack: 0)                                         */
  double tflops_eq;      /* flops_eq / time_s / 1e12 (0 when time_s is 0)                   */
  double eps_s;          /* time of the eps-path terms at fraction 1 (the bracket above);
                            0 when T < 2^53 (no eps path)                                   */
  double eps_fraction;   /* the eps fraction used (the default estimate or the request's)  */
  int64_t kernel_launches;
  uint64_t workspace_bytes;   /* imd_workspace_size(_packed) (exact)                        */
  uint64_t plan_device_bytes; /* imd_plan_bytes (exact)                                     */
  uint64_t pack_bytes;        /* imd_perf_predict_pack: imd_pack_layout's (else 0)          */
  uint64_t pack_ws_bytes;     /* imd_perf_predict_pack: imd_pack's workspace (else 0)       */
  imd_perf_phase_t phase[IMD_PERF_MAX_PHASES];
} imd_perf_t;

/* Predicted imd_contract (packing != NULL: imd_contract_packed of that packed plan). Returns
 * the plan-time status (imd_lowering / imd_lowering_packed), INVALID_VALUE for out NULL,
 * a struct_size below version 1's, a request with an unknown flag or K4 configuration, or a
 * calibration of another format / with sm_count < 1. */
IMD_API imd_status_t imd_perf_predict(const imd_tensor_t* A, const imd_tensor_t* B,
                                      const imd_tensor_t* C, const imd_options_t* options,
                                      const imd_packing_t* packing,
                                      const imd_perf_request_t* request,
                                      const imd_perf_calib_t* calib, imd_perf_t* out);
/* Predicted imd_pack of a pack descriptor (imd_pack_layout's status); of the request only
 * eps_fraction is used (NULL: the default). */
IMD_API imd_status_t imd_perf_predict_pack(const imd_pack_desc_t* desc,
                                           const imd_perf_request_t* request,
                                           const imd_perf_calib_t* calib, imd_perf_t* out);

/* Ranking (§6c.3). Masks: */
enum { IMD_PERF_VEC_GENERATE = 1, IMD_PERF_VEC_RESIDENT = 2 };
enum { IMD_PERF_PACK_NONE = 1, IMD_PERF_PACK_A = 2, IMD_PERF_PACK_B = 4, IMD_PERF_PACK_AB = 8 };
typedef struct {
  uint32_t struct_size;          /* sizeof(imd_perf_rank_request_t)                          */
  uint32_t candidate_size;       /* sizeof(imd_perf_candidate_t) as compiled by the caller   */
  int32_t num_moduli_count;      /* >= 1                                                     */
  int32_t residue_group_count;   /* 0: every g in 1..N                                       */
  const int32_t* num_moduli;     /* N values, 2..20                                          */
  const int32_t* residue_groups; /* g values >= 1 (values > N are skipped for that N)        */
  int32_t vec_modes;             /* IMD_PERF_VEC_* mask; 0 = GENERATE                        */
  int32_t pack_modes;            /* IMD_PERF_PACK_* mask (the caller's operands); 0 = NONE   */
  int32_t reuse;                 /* contractions per imd_pack (amortization), >= 1           */
  int32_t flags;                 /* IMD_PERF_BETA_NONZERO                                    */
  uint64_t budget_bytes;         /* device memory for total_bytes; 0 = no budget             */
  double eps_fraction;           /* as imd_perf_request_t: < 0 the default, else 0..1        */
} imd_perf_rank_request_t;

typedef struct {
  int32_t num_moduli, residue_group, vec_mode; /* the candidate (g normalized to [1, N])      */
  int32_t packed;            /* bit 0 / bit 1: the caller's A / B packed (imd_pack_desc_for) */
  int32_t status;            /* plan-time status (imd_lowering / imd_lowering_packed)        */
  int32_t feasible;          /* 1: SUCCESS and total_bytes <= budget                         */
  int32_t k4_config;         /* predicted K4 configuration                                   */
  int32_t dispatch;          /* IMD_DISPATCH_*                                               */
  double time_s;             /* the rank key: contract_s + pack_s / reuse                    */
  double contract_s;         /* imd_perf_predict                                             */
  double pack_s;             /* sum of imd_perf_predict_pack over the packed operands        */
  double tflops_eq;          /* 2 b m n k / time_s / 1e12                                    */
  double eps_s;              /* eps-path time at fraction 1: contract + pack / reuse         */
  double eps_fraction;       /* the eps fraction used                                        */
  uint64_t workspace_bytes;  /* imd_workspace_size(_packed)                                  */
  uint64_t plan_device_bytes;/* imd_plan_bytes                                               */
  uint64_t pack_bytes;       /* sum of imd_pack_layout pack_bytes                            */
  uint64_t pack_ws_bytes;    /* sum of imd_pack_layout pack_ws_bytes                         */
  uint64_t total_bytes;      /* the four above (separate buffers)                            */
  int64_t kernel_launches;   /* per imd_contract                                             */
} imd_perf_candidate_t;

/* Every candidate of N x g x vec mode x pack mode (base gives rounding, mode, k4_tune), ranked:
 * feasible by time_s, then infeasible SUCCESS ones by time_s, then the others in enumeration
 * order (ties: total_bytes, N, g, vec_mode, packed). Writes min(capacity, n) candidates of
 * candidate_size bytes each and *num_candidates = n. INVALID_VALUE for a bad request or
 * argument; else SUCCESS with each candidate's own status. */
IMD_API imd_status_t imd_perf_rank(const imd_tensor_t* A, const imd_tensor_t* B,
                                   const imd_tensor_t* C, const imd_options_t* base,
                                   const imd_perf_rank_request_t* request,
                                   const imd_perf_calib_t* calib, imd_perf_candidate_t* out,
                                   int32_t capacity, int32_t* num_candidates);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_HOST_H */
