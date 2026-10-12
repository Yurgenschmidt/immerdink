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
#define IMD_HOST_API_VERSION 13 /* 2: imd_lowering_t fused, kernel_launches (§6a v1.4);
                                   3: K4 configurations, imd_options_t.k4_tune,
                                      imd_lowering_t sm_count, k4_config, k4_tune_bytes (v1.5);
                                   4: imd_lowering_t fused_large (v1.6);
                                   5: packed operands (§6b v1.7): IMD_STATUS_PACK_MISMATCH,
                                      imd_pack_layout, imd_pack_desc_for, imd_*_packed,
                                      imd_lowering_t packed, pack_mismatch;
                                   6: the performance model (§6c v1.8): imd_perf_*;
                                   7: performance model v1.10 (§6c): calibration format 2,
                                      occupancy (imd_perf_kernel_t, imd_perf_kernel_name),
                                      clock regimes, cache state, intervals, rank ties;
                                   8: accuracy-aware planning (§6f v1.12): imd_err_*,
                                      the requests' accuracy, err_* / infeasible fields;
                                   9: per-plan K alignment (§4b v1.13 stage 2):
                                      imd_lowering_t k_align (k_hat = ceil(k)_k_align),
                                      IMD_PACK_FORMAT_A16 / _A64, pack format 2;
                                  10: explicit calibration parse errors (§6c.3):
                                      imd_perf_calib_error_t, IMD_PERF_CALIB_ERR_*,
                                      imd_perf_calib_parse_ex / _load_ex / _default_ex,
                                      imd_perf_calib_error_name;
                                  11: clock-scaled byte energy (§6c.7 v1.15):
                                      imd_perf_calib_t energy_byte_f_pj, optional
                                      calibration key power.byte_f_pj; the eps path
                                      retired (§6c.1): eps_fraction ignored, eps_s /
                                      eps_fraction out 0, op_eps keys optional;
                                  12: performance model v1.17 (§6c): imd_perf_calib_t
                                      pack_l2_hit_bytes (key warm.pack_l2_bytes), optional
                                      keys kf.elem, pack.dram; param names KF 4 "elem",
                                      PACK 3 "dram", KL 4 NULL; op_eps keys retired;
                                  13: performance model v1.18 (§6c.8): imd_perf_calib_t
                                      clock_boost_mhz, power_act_w, power_act_gap_us,
                                      energy_dram_pj, warm_call_ns, warm_clean_byte_ns,
                                      pack_l2_planes (optional keys), IMD_PERF_CACHE_CLEAN;
                                      clock_mhz out up to the boost clock */

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
 * INVALID_VALUE or NOT_SUPPORTED (§5 rows 1-5); never applies the §3.2 RN acceptance rule.
 * Guaranteed from host API 13 on (§6a F1): with the descriptors, mode and vec_mode fixed, the
 * bytes are nondecreasing in N at every fixed g >= 1, in g in [1, N], and in N with g = 0,
 * where the query succeeds. The one status that depends on (N, g), INVALID_VALUE for a size
 * past uint64, occurs only at the largest N and g, so the N where it succeeds at a fixed g
 * form an interval. If it succeeds at (N_max, g_max), it succeeds at every (N, g) with
 * g <= N <= N_max and g <= g_max (g = 0 counts as N) with at most those bytes, so that one
 * workspace serves the plans at all of them (§6a C-ws). */
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
  int64_t n_hat, k_hat;  /* ceil(n)_16, ceil(k)_A with A = k_align (v6; host API < 9:
                            always ceil(k)_16). Read it; never recompute it (§4b)  */
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
  /* Version 6 (192 bytes; §4b v1.13 stage 2, host API 9). Valid when lowered = 1. */
  int32_t k_align;         /* the plan's resolved K alignment A (16 or 64): k_hat = ceil(k)_A,
                              the row stride and GEMM K extent of every residue row. AUTO:
                              64 iff k > 96. 16 for a lowering without residue rows (NOOP,
                              SCALE, DIRECT, K-empty); 0 when lowered = 0.
                              NOT the pack key: a DIRECT plan with k > 96 reports 16 here
                              but matches packs of format 2 (A64), the resolved A of its k
                              (AUTO, or the hook), exactly like the EMULATE plan of the same
                              tensors (§6b.4 word 1); compare a pack's format with
                              imd_pack_layout_t.format, never with this field             */
  int32_t reserved6;       /* 0                                                           */
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
 * residues of ALL N moduli over the whole canonical K, in the §6b.4 byte layout (format 1:
 * k_hat = ceil(k)_16; format 2, host API 9: k_hat = ceil(k)_64; a pack's format is that of
 * its k, 2 iff k > 96, §4b):
 *   sig   int16 sigma of row (lambda, rho) at index lambda r + rho
 *   flag  u8 flag, same index
 *   res   int8 residue of plane l, row (lambda, rho), k-bar at l b r k_hat + (lambda r + rho)
 *         k_hat + k-bar; zero K padding; a zeroed tail of (ceil(r)_16 - r) k_hat bytes
 * regions at 256-aligned offsets, every byte written (gaps 0). imd_pack (immerdink.h) writes
 * it; plans built against it read it instead of scaling and packing X themselves. The pack
 * fixes its canonical orders of K, L (batch) and R (X's rows); a plan adopts them.
 * Compatibility: a key of int64 words (format, N, rounding, b, r, k, then per class L, R, K
 * the fused modes (extent, stride in X) fused over X's strides alone), compared word for
 * word.
 * The pack bytes carry no header: a caller that persists them must persist the layout's
 * format and pack_bytes with them and re-pack when a fresh imd_pack_layout of the same
 * descriptor reports another format or size (bytes saved under host API <= 8 for k > 96 are
 * format 1 and NOT valid for a version-9 handle, which is format 2; nothing detects it).
 * Compare a saved format with a fresh layout's, never with IMD_PACK_FORMAT: packs of k <= 96
 * are format 1 in every version. */
#define IMD_PACK_FORMAT_A16 1  /* k_hat = ceil(k)_16 (k <= 96; every pack before host API 9) */
#define IMD_PACK_FORMAT_A64 2  /* k_hat = ceil(k)_64 (k > 96; host API 9)                     */
#define IMD_PACK_FORMAT 2      /* the newest format this library writes (1 before API 9)    */
enum { IMD_PACK_ORDER_AUTO = 0, IMD_PACK_ORDER_GIVEN = 1 };
/* imd_lowering_t.pack_mismatch: the first differing part. Per packed operand (the caller's A,
 * then B): an unknown format word (FORMAT), MODULI, ROUNDING, CLASSES (so A's wrong N is
 * MODULI even when B's format is unknown); then ORDER; then the keys' extents and strides of
 * both packs; then (host API 9) a format 1 vs 2 difference of either pack: the same tensor in
 * the other K alignment (in-process only through a test hook). */
enum {
  IMD_PACK_MISMATCH_NONE = 0,
  IMD_PACK_MISMATCH_FORMAT = 1,   /* an unknown format, or the pack's format (1 / 2) is not the
                                     plan's own (k alignment)                                 */
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
  int32_t format;                /* IMD_PACK_FORMAT_A16 (k <= 96) or _A64 (k > 96)           */
  int32_t num_moduli, rounding;  /* N, IMD_RN | IMD_TRUNC                                    */
  int32_t reserved;              /* 0                                                        */
  int64_t b, r, k, k_hat;        /* batches, rows, contraction length, ceil(k)_A (format's A) */
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
 * and has 0 bytes. Guaranteed from host API 13 on (§6a F2): with the rest of the descriptor
 * fixed, pack_bytes is nondecreasing in N where it is reported (a size past uint64 occurs
 * only at the largest N), so a buffer of the largest N's pack_bytes holds the pack at any
 * smaller N (§6a C-pack). pack_ws_bytes and the region offsets carry no such
 * promise (a caller sizing imd_pack's workspace once takes the maximum over its N). */
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
 * rule of the plan. From host API 13 on, F1 holds as for imd_workspace_size with the packed
 * roles fixed too (each pack descriptor at the query's N and rounding). One more status
 * depends on N: an RN pack is NOT_SUPPORTED at the smallest N, those where kappa >= T(N).
 * INVALID_VALUE also covers a pack's size past uint64, at the largest N. So if the query
 * succeeds at (N_max, g_max), every (N, g) below it (as for imd_workspace_size) either
 * succeeds with at most those bytes or is that NOT_SUPPORTED. The N where it succeeds at a
 * fixed g form an interval, not an up-set. */
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
#define IMD_PERF_CALIB_FORMAT 2     /* the newest calibration format (v7); formats 1 and 2 are
                                       read, and each is written as itself                  */
#define IMD_PERF_MAX_PARAMS 6       /* parameters (features) per class                       */
#define IMD_PERF_PARAM_CAPACITY 96  /* imd_perf_calib_t.param slots: class * 6 + j            */
#define IMD_PERF_PARAM_KL_DROP_SLOT (IMD_PERF_PARAM_CAPACITY - 1) /* param[] slot of warm.kl_drop
                                       (v1.19, 95): the last slot, past every class's slots     */
#define IMD_PERF_MAX_PHASES 16      /* imd_perf_t.phase capacity                              */

/* Kernel classes (one parameter set each; file key prefix in parentheses). */
enum {
  IMD_PERF_CLASS_K0 = 0,         /* (k0) K0 vectors, imd_pack's prep: launch, byte, chain   */
  IMD_PERF_CLASS_KF = 1,         /* (kf) fused small launch: launch, byte, op, row, elem
                                    (v12: the per-element chain), chain                     */
  IMD_PERF_CLASS_KL = 2,         /* (kl) fused large launch, imd_pack: launch, byte, op,
                                    l2byte, chain (v1.20, optional key kl.chain: the busiest
                                    SM's nc x g steps against the byte term), wave          */
  IMD_PERF_CLASS_PACK = 3,       /* (pack) K2 / K3, groups 2..: launch, byte, op, dram (v12:
                                    the bytes that are not L2-hit re-reads), chain, wave (the
                                    tile loads in sequence)                                  */
  IMD_PERF_CLASS_K4_128x256 = 4, /* (k4_128x256) K4: max(round model, byte): round, mac1,
                                    byte (slot 4), mac_ktail (slot 5, in the round model)   */
  IMD_PERF_CLASS_K4_128x128 = 5, /* (k4_128x128) round, mac1, mac2, byte, mac_ktail         */
  IMD_PERF_CLASS_K4_64x128 = 6,  /* (k4_64x128)  round, mac1, mac2, byte, mac_ktail         */
  IMD_PERF_CLASS_K4_64x64 = 7,   /* (k4_64x64)   round, mac1, mac2, mac3, byte, mac_ktail   */
  IMD_PERF_CLASS_K5 = 8,         /* (k5) CRT: launch, byte, op, chain, dram                 */
  IMD_PERF_CLASS_DIRECT = 9,     /* (direct) native FP64: launch, byte, flop, chain         */
  IMD_PERF_CLASS_BETA = 10,      /* (beta) C <- beta C: launch, byte, chain                 */
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

/* v7 (DESIGN v1.10 §6c.1): the kernels and variants whose occupancy (CTAs resident per SM)
 * the calibration measures; the waves and per-thread chain of a launch use it. */
#define IMD_PERF_KERNEL_CAPACITY 128
#define IMD_PERF_CLASS_CAPACITY 16  /* imd_perf_calib_t.band_class rows (>= IMD_PERF_CLASS_COUNT) */
enum {
  IMD_PERF_KERNEL_K0 = 0,         /* "k0"        k0_vec                                      */
  IMD_PERF_KERNEL_KF16 = 1,       /* "kf16"      KF, 16 values per thread                    */
  IMD_PERF_KERNEL_KF4 = 2,        /* "kf4"       KF, 4 values per thread                     */
  IMD_PERF_KERNEL_KL8 = 3,        /* "kl8"       KL, 8 values per thread (512 threads)       */
  IMD_PERF_KERNEL_KL4 = 4,        /* "kl4"       KL, 4 values per thread (1024 threads)      */
  IMD_PERF_KERNEL_PACK = 5,       /* "pack"      K2 / K3                                     */
  IMD_PERF_KERNEL_ZERO_ROWS = 6,  /* "zero_rows" K3's zero-rows launch                       */
  IMD_PERF_KERNEL_DIRECT = 7,     /* "direct"                                                */
  IMD_PERF_KERNEL_BETA = 8,       /* "beta"      K5 beta-only                                */
  IMD_PERF_KERNEL_PACK_PREP = 9,  /* "pack_prep" imd_pack's prep launch                      */
  IMD_PERF_KERNEL_K4 = 10,        /* + IMD_K4_*: "k4_128x256", ... (store-mod kernel)        */
  IMD_PERF_KERNEL_K5 = 14,        /* + 4 (N - 2) + v: v = 0 / 1 / 2 j-bar fastest with 1 / 2 / 4
                                     outputs per thread ("k5_n<N>_e<E>"), 3 lambda fastest
                                     ("k5b_n<N>"), N = 2..20                                 */
  IMD_PERF_KERNEL_COUNT = 90
};
typedef struct {
  int32_t resident;      /* CTAs per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor)     */
  int32_t regs;          /* registers per thread (cudaFuncGetAttributes)                    */
  int32_t smem_bytes;    /* shared memory per CTA (static + the launch's dynamic)           */
  int32_t threads;       /* threads per CTA of the launch                                   */
} imd_perf_kernel_t;
/* v7: the static key name of a kernel ("k0", "kf16", ..., "k5_n14_e4", "k5b_n14"); NULL out
 * of range. */
IMD_API const char* imd_perf_kernel_name(int32_t kernel);
/* v7 request fields (DESIGN v1.10 §6c.1). */
enum { IMD_PERF_REGIME_DEFAULT = 0,   /* SUSTAINED if the calibration has a power model     */
       IMD_PERF_REGIME_BURST = 1,     /* the calibration's burst state (effective clock
                                         power.clock_eff_mhz, below the boost readout)       */
       IMD_PERF_REGIME_SUSTAINED = 2  /* the clock at which the call's power meets the cap   */ };
enum { IMD_PERF_CACHE_COLD = 0,       /* the inputs come from DRAM                           */
       IMD_PERF_CACHE_WARM = 1,       /* the input was the previous call's output; modeled as
                                         REPEAT (provisional: one method, square GEMM pairs,
                                         N = 14 / 16). Recommended in that case. The v1.18
                                         pairs measure WARM 4.5-11.9 % below REPEAT on held-
                                         out shapes (NOTES-B2 §12; open).                     */
       IMD_PERF_CACHE_REPEAT = 2,     /* the calibration's state: the same call just ran (the
                                         inputs resident only if the whole footprint fits)   */
       IMD_PERF_CACHE_CLEAN = 3       /* v13 (§6c.8): the inputs evicted and the L2 clean (an
                                         L2 flushed by reads, e.g. imd_bench --l2 flush). COLD
                                         = evicted by other, dirty work (a call in a sequence) */ };
enum { IMD_PERF_ORDER_TIME = 0,       /* rank: by predicted time (the default; v6's order)   */
       IMD_PERF_ORDER_TIES_BY_FOOTPRINT = 1 /* rank: the ties with the best first, by footprint */ };
enum { IMD_PERF_BAND_SMALL = 0, IMD_PERF_BAND_MID = 1, IMD_PERF_BAND_LARGE = 2 }; /* predicted
                                         call < 100 us, < 1 ms, >= 1 ms                       */

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
  double param[IMD_PERF_PARAM_CAPACITY]; /* param[class * IMD_PERF_MAX_PARAMS + j]; 0 unused;
                             v1.19 (§6c.9): param[IMD_PERF_PARAM_KL_DROP_SLOT] = key
                             warm.kl_drop (optional, 0 when absent, in [0, 1]), the share of
                             the REPEAT / WARM saving that KL reads do not get. The file names
                             the key, so the text format does not depend on the slot. The
                             struct's layout and size are unchanged (the slot was unused), but
                             the slot is now meaningful: a hand-built imd_perf_calib_t MUST
                             zero `param` (memset the struct): the last slot is read, and a
                             nonzero one is a warm.kl_drop (outside [0, 1]: not a valid
                             calibration)                                                     */
  /* ---- v7 (format 2). A caller with a version-6 struct_size gets a true format-1 view of a
   *      format-2 calibration: the format-2 parameters zeroed, built-in occupancy, no power
   *      model, no cache term, no bands: a degraded model. Such callers should pass NULL. ---- */
  imd_perf_kernel_t kernel[IMD_PERF_KERNEL_CAPACITY]; /* [IMD_PERF_KERNEL_*]; 0 unused        */
  double clock_eff_mhz;   /* the effective SM clock of the burst-state parameters (NVML scale,
                             below the boost readout); 0: no clock scaling                  */
  double clock_min_mhz;   /* lowest SUSTAINED clock                                          */
  double power_cap_w;     /* board power limit; 0: no power model (SUSTAINED = BURST)        */
  double power_idle_w;    /* awake idle power                                                */
  double energy_tc_pj;    /* per int8 tensor-core op at clock_eff                            */
  double energy_alu_pj;   /* per residue / CRT limb op / FP64 flop at clock_eff              */
  double energy_byte_pj;  /* per DRAM byte                                                   */
  double energy_gamma;    /* energy per op scales with (f / clock_eff)^gamma                 */
  double warm_l2_bytes;   /* L2 bytes that can hold a call's inputs (COLD / WARM)            */
  double warm_l2byte_ns;  /* saving per input byte served from L2 instead of DRAM            */
  double band_call[3][2]; /* [IMD_PERF_BAND_*] {lo, hi} factors of the predicted call time  */
  double band_class[IMD_PERF_CLASS_CAPACITY][2]; /* [IMD_PERF_CLASS_*] {lo, hi} per phase    */
  double tie_ratio;       /* rank: a feasible candidate within tie_ratio x the best's time is a
                             tie (1 <= tie_ratio <= 1.25; 1: exact ties only)                */
  /* ---- v11 (DESIGN v1.15 §6c.7). A caller with a v7..v10 struct_size gets the folded view
   *      energy_byte_pj = byte_pj + byte_f_pj from the calibration functions (default /
   *      parse / load): a degraded, unvalidated view (never fitted; not an API-10 prediction),
   *      as the v6 view; such callers should pass NULL. The predictors do not fold: a struct
   *      of that size is read with energy_byte_f_pj = 0. ---- */
  double energy_byte_f_pj; /* per DRAM byte, scaled like the ops by (f / clock_eff)^gamma
                              (SM / L2-side energy of the traffic; key power.byte_f_pj,
                              optional, 0 when absent)                                       */
  /* ---- v12 (DESIGN v1.17 §6c.1, §6c.3). A caller with a v7..v11 struct_size is read
   *      (predictors, imd_perf_calib_format) with pack_l2_hit_bytes = warm_l2_bytes when
   *      pack.dram > 0, else 0 (no effect then). Exact for the shipped calibration (both
   *      sizes 0.8 x L2); for another file with pack.dram > 0 a degraded view (pass NULL). */
  double pack_l2_hit_bytes; /* L_hit: a K2 / K3 re-read of an operand hits L2 when the call's
                              group working set is at most this (key warm.pack_l2_bytes,
                              optional, 0 when absent: no hits)                             */
  /* ---- v13 (DESIGN v1.18 §6c.8). Optional keys, 0 when absent (0 predicts as API 12). A
   *      caller with a v7..v12 struct_size is read with these fields 0: for a file with the
   *      v1.18 keys (the shipped one) a degraded, unvalidated view; pass NULL. ---- */
  double clock_boost_mhz;   /* f_top: the top of the clock range (BURST; SUSTAINED solved on
                               [clock_min, f_top]; time flat above clock_eff). 0: clock_eff
                               (key power.clock_boost_mhz; 0 or >= clock_eff_mhz)            */
  double power_act_w;       /* busy-GPU power at clock_eff, x busy share x (f/f_ref)^(1+gamma)
                               (key power.act_w)                                              */
  double power_act_gap_us;  /* launch gap of the busy share b = 1 - launches x gap / t_burst
                               (key power.act_gap_us)                                         */
  double energy_dram_pj;    /* per byte of a call whose footprint exceeds warm_l2_bytes,
                               unscaled (key power.dram_pj)                                   */
  double warm_call_ns;      /* REPEAT / WARM: per-call saving when an input is resident
                               (key warm.call_ns)                                             */
  double warm_clean_byte_ns; /* CLEAN: credit per byte of min(written footprint, warm_l2_bytes)
                               on the call phase (key warm.clean_byte)                        */
  double pack_l2_planes;    /* 1: the PACK group working set counts one group's W planes;
                               0: the v1.17 set (key warm.pack_l2_planes, 0 or 1)             */
} imd_perf_calib_t;

/* The shipped calibration (RTX 4090, sm_89). out->struct_size must be set; SUCCESS. A
 * version-6-sized struct receives the degraded format-1 view (see above); pass NULL to the
 * imd_perf_* functions instead to use the full calibration (parsed once, no copy). */
IMD_API imd_status_t imd_perf_calib_default(imd_perf_calib_t* out);
/* Parse calibration text (format 1 or 2: "key = value" lines, '#' comments; every key of the
 * format required, no unknown or repeated key; format 2 adds the chain parameters and the
 * kernel., power., warm. and band. keys, DESIGN §6c.2; format 1 is migrated: built-in
 * occupancy, no power model, no cache term, no bands, chain parameters 0). INVALID_VALUE with *error_line = the 1-based line
 * (0: not tied to a line, e.g. a missing key); error_line may be NULL. */
IMD_API imd_status_t imd_perf_calib_parse(const char* text, size_t length, imd_perf_calib_t* out,
                                          int32_t* error_line);
/* imd_perf_calib_parse of a file's contents; an unreadable file is INVALID_VALUE, line 0. */
IMD_API imd_status_t imd_perf_calib_load(const char* path, imd_perf_calib_t* out,
                                         int32_t* error_line);
/* Host API 10 (DESIGN §6c.3): why a calibration was rejected. The _ex functions return the
 * status of the function without _ex and fill the caller's struct_size bytes of *err: on
 * SUCCESS line 0, reason IMD_PERF_CALIB_OK, key and message ""; else the first error in file
 * order (a missing key: line 0, the first missing key in canonical order). The format of the
 * calibration text is unchanged; the functions without _ex are wrappers. */
enum { IMD_PERF_CALIB_OK = 0,
       IMD_PERF_CALIB_ERR_ARGUMENT = 1,     /* NULL / undersized out or err, NULL text        */
       IMD_PERF_CALIB_ERR_FILE = 2,         /* load: cannot open or read, or >= 1 MiB         */
       IMD_PERF_CALIB_ERR_SYNTAX = 3,       /* a line not blank, '#' comment or key = value   */
       IMD_PERF_CALIB_ERR_UNKNOWN_KEY = 4,
       IMD_PERF_CALIB_ERR_REPEATED_KEY = 5,
       IMD_PERF_CALIB_ERR_VALUE = 6,        /* not a number, wrong count, out of range        */
       IMD_PERF_CALIB_ERR_TOO_LONG = 7,     /* a string value does not fit its field (+ NUL)  */
       IMD_PERF_CALIB_ERR_MISSING_KEY = 8,  /* a key the format requires                      */
       IMD_PERF_CALIB_ERR_FORMAT2_KEY = 9,  /* a format-2 key in a format-1 calibration       */
       IMD_PERF_CALIB_ERR_INCONSISTENT = 10 /* a cross-key rule: power clocks, tie_ratio      */ };
#define IMD_PERF_CALIB_ERR_COUNT 11
typedef struct {
  uint32_t struct_size;  /* sizeof(imd_perf_calib_error_t) as the caller compiled it (>= 16) */
  int32_t line;          /* 1-based line of the error; 0: not tied to a line                */
  int32_t reason;        /* IMD_PERF_CALIB_OK or IMD_PERF_CALIB_ERR_*                       */
  int32_t reserved;      /* 0                                                               */
  char key[64];          /* the offending or missing key (truncated), or ""; NUL-terminated */
  char message[192];     /* e.g. "line 9: notes: value is 133 bytes, the field holds 127"   */
} imd_perf_calib_error_t;
/* imd_perf_calib_parse with the reason; err may be NULL. */
IMD_API imd_status_t imd_perf_calib_parse_ex(const char* text, size_t length,
                                             imd_perf_calib_t* out, imd_perf_calib_error_t* err);
/* imd_perf_calib_load with the reason; err may be NULL. */
IMD_API imd_status_t imd_perf_calib_load_ex(const char* path, imd_perf_calib_t* out,
                                            imd_perf_calib_error_t* err);
/* imd_perf_calib_default with the reason: INVALID_VALUE with the shipped file's own parse
 * error (message prefixed "shipped calibration: ") if the embedded text did not parse, which
 * is also why every function given a NULL calibration then returns INVALID_VALUE. */
IMD_API imd_status_t imd_perf_calib_default_ex(imd_perf_calib_t* out,
                                               imd_perf_calib_error_t* err);
/* Static name of a reason ("IMD_PERF_CALIB_OK", "IMD_PERF_CALIB_ERR_TOO_LONG", ...); NULL out
 * of range. */
IMD_API const char* imd_perf_calib_error_name(int32_t reason);
/* The canonical text of a calibration (every key; numbers in the shortest round-trip form,
 * so parse(format(c)) == c bit for bit). *length = its length without the NUL, always;
 * INSUFFICIENT_WORKSPACE if capacity <= *length (nothing useful written). */
IMD_API imd_status_t imd_perf_calib_format(const imd_perf_calib_t* calib, char* buf,
                                           size_t capacity, size_t* length);
/* Static names: a class's key prefix ("k0", ..., "call"), a parameter's name ("launch",
 * "byte", "op", "row", "l2byte", "chain", "round", "mac1".. "mac3", "flop", "call"; v12:
 * "elem", "dram"; NULL for a slot the class does not use) and a phase's name; NULL out of
 * range. */
IMD_API const char* imd_perf_class_name(int32_t cls);
IMD_API const char* imd_perf_param_name(int32_t cls, int32_t j);
IMD_API const char* imd_perf_phase_name(int32_t phase);

/* The eps path (§6c.1) is RETIRED in host API 11 (DESIGN v1.15): up to v1.14 a pack residue of
 * an element with |a'| >= 2^53 cost one more modular multiply, so the pack time depended on the
 * data and the model took the fraction of pack work on that path as an input. The v1.14 pack
 * emitter (§3.4.1) has no such path: every residue costs the same for every eps. Since API 11
 * eps_fraction in the requests is validated as before (< 0, or 0..1; NaN or > 1 is
 * INVALID_VALUE) and then IGNORED; eps_s and eps_fraction in the results are always 0; the
 * op_eps features (x[4] of kf, x[3] of pack, and x[4] of kl unless the calibration has the
 * optional key kl.chain > 0, v1.20, DESIGN §6c.10) are always 0. The fields stay for the ABI. */

/* What to predict beyond the options. NULL = flags 0, k4_config -1, eps_fraction -1 (ignored). */
typedef struct {
  uint32_t struct_size;  /* sizeof(imd_perf_request_t)                                      */
  int32_t flags;         /* IMD_PERF_BETA_NONZERO (C is read), IMD_PERF_ALPHA_ZERO          */
  int32_t k4_config;     /* -1: the plan-time rule for the calibration's sm_count; else IMD_K4_* */
  int32_t reserved;      /* 0                                                               */
  double eps_fraction;   /* < 0 or 0..1, validated and ignored since API 11 (see above)      */
  /* ---- v7 (a struct_size of version 6: DEFAULT, COLD, no clock) ---- */
  int32_t regime;        /* IMD_PERF_REGIME_*                                               */
  int32_t cache;         /* IMD_PERF_CACHE_*                                                */
  double clock_mhz;      /* > 0: an NVML-scale SM clock for capped / sustained operation
                            (e.g. a measured settled clock; overrides the regime); values at
                            or above the top clock (v13: clock_boost_mhz if > 0, else
                            clock_eff_mhz) mean BURST; 0: none                              */
  /* ---- v8 (§6f; a struct_size of version 7: accuracy NULL) ---- */
  const struct imd_err_request_s* accuracy; /* metric, target and statistics of err_*;
                                               NULL: NORMWISE, no target                    */
} imd_perf_request_t;

/* One phase: its features x (the model is sum_j param[cls * 6 + j] x[j]; K4: the max of the
 * terms j != 4 and the term j = 4; KL with kl.chain > 0: the byte term j = 1 against the chain
 * term j = 4, plus the others; the op_eps features are 0 since API 11), its
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
  double eps_s;          /* 0 since API 11 (was: the eps-path terms at fraction 1)          */
  double eps_fraction;   /* 0 since API 11 (was: the eps fraction used)                     */
  int64_t kernel_launches;
  uint64_t workspace_bytes;   /* imd_workspace_size(_packed) (exact)                        */
  uint64_t plan_device_bytes; /* imd_plan_bytes (exact)                                     */
  uint64_t pack_bytes;        /* imd_perf_predict_pack: imd_pack_layout's (else 0)          */
  uint64_t pack_ws_bytes;     /* imd_perf_predict_pack: imd_pack's workspace (else 0)       */
  imd_perf_phase_t phase[IMD_PERF_MAX_PHASES];
  /* ---- v7 ---- */
  int32_t regime;        /* the regime used (DEFAULT resolved)                              */
  int32_t cache;         /* the cache state used                                            */
  double clock_mhz;      /* the SM clock assumed: at most clock_eff_mhz, v13 at most the top
                            clock (the power is evaluated there; the time uses
                            min(clock, clock_eff)); 0: no clock model                       */
  double power_w;        /* the call's average board power at that clock (0: no power model) */
  double warm_bytes;     /* input bytes served from L2 (WARM)                               */
  double time_lo_s;      /* [time_lo_s, time_hi_s]: an interval aimed at 80 % coverage (2-fold
                            CV on the hold-out; BURST / REPEAT residuals, applied unchanged in
                            every regime / cache state); = time_s without calibration bands */
  double time_hi_s;
  double phase_lo_s[IMD_PERF_MAX_PHASES]; /* the same per phase                             */
  double phase_hi_s[IMD_PERF_MAX_PHASES];
  /* ---- v8 (§6f) ---- */
  double err_bound;      /* rigorous error bound in the request's metric (+inf: none)       */
  double err_typical;    /* typical estimate (RN EMULATE; else = err_bound)                 */
  int32_t err_status;    /* IMD_ERR_UNCHECKED (no target) / IMD_ERR_MET / IMD_ERR_MISSED    */
  int32_t err_metric;    /* the metric used                                                 */
} imd_perf_t;

/* Predicted imd_contract (packing != NULL: imd_contract_packed of that packed plan). Returns
 * the plan-time status (imd_lowering / imd_lowering_packed), INVALID_VALUE for out NULL,
 * a struct_size below version 1's, a request with an unknown flag or K4 configuration, or a
 * calibration of another format (1 and 2 are valid) / with sm_count < 1, or a v7 request
 * with an unknown regime or cache or a negative or non-finite clock_mhz. v7: the prediction
 * at the request's regime / clock and cache state, with its band. */
IMD_API imd_status_t imd_perf_predict(const imd_tensor_t* A, const imd_tensor_t* B,
                                      const imd_tensor_t* C, const imd_options_t* options,
                                      const imd_packing_t* packing,
                                      const imd_perf_request_t* request,
                                      const imd_perf_calib_t* calib, imd_perf_t* out);
/* Predicted imd_pack of a pack descriptor (imd_pack_layout's status); of the request the
 * v7 fields are used (flags and k4_config are ignored; eps_fraction validated, ignored). */
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
  double eps_fraction;           /* as imd_perf_request_t: validated, ignored (API 11)       */
  /* ---- v7 (a struct_size of version 6: DEFAULT regime, COLD, no clock, ORDER_TIME) ---- */
  int32_t regime, cache;         /* as imd_perf_request_t                                     */
  int32_t order;                 /* IMD_PERF_ORDER_*                                          */
  int32_t reserved;              /* 0                                                         */
  double clock_mhz;              /* as imd_perf_request_t                                     */
  /* ---- v8 (§6f; a struct_size of version 7: accuracy NULL) ---- */
  const struct imd_err_request_s* accuracy; /* NULL: no error target (NORMWISE reported); a
                                               target > 0 makes the candidates whose estimate
                                               (by basis) exceeds it infeasible             */
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
  double eps_s;              /* 0 since API 11 (eps path retired)                            */
  double eps_fraction;       /* 0 since API 11                                               */
  uint64_t workspace_bytes;  /* imd_workspace_size(_packed)                                  */
  uint64_t plan_device_bytes;/* imd_plan_bytes                                               */
  uint64_t pack_bytes;       /* sum of imd_pack_layout pack_bytes                            */
  uint64_t pack_ws_bytes;    /* sum of imd_pack_layout pack_ws_bytes                         */
  uint64_t total_bytes;      /* the four above (separate buffers)                            */
  int64_t kernel_launches;   /* per imd_contract                                             */
  /* ---- v7 ---- */
  double time_lo_s;          /* interval of time_s: contract's + pack's / reuse              */
  double time_hi_s;
  double clock_mhz;          /* the contract's assumed SM clock                              */
  double power_w;            /* the contract's predicted average power                       */
  int32_t tie;               /* 1: feasible and time_s <= the best's x tie_ratio (any order) */
  int32_t raw_rank;          /* position in the v6 time order (0-based)                      */
  /* ---- v8 (§6f) ---- */
  double err_bound;          /* rigorous error bound in the request's metric (+inf: none)    */
  double err_typical;        /* typical estimate (RN EMULATE; else = err_bound)              */
  int32_t err_status;        /* IMD_ERR_UNCHECKED / IMD_ERR_MET / IMD_ERR_MISSED             */
  int32_t infeasible;        /* IMD_PERF_INFEASIBLE_* bits; feasible = (infeasible == 0)     */
} imd_perf_candidate_t;

/* v8: why a candidate is infeasible (imd_perf_candidate_t.infeasible). */
enum { IMD_PERF_INFEASIBLE_STATUS = 1,   /* its plan-time status is not SUCCESS               */
       IMD_PERF_INFEASIBLE_BUDGET = 2,   /* total_bytes > budget_bytes                        */
       IMD_PERF_INFEASIBLE_ACCURACY = 4  /* its error estimate (accuracy->basis) > target     */ };

/* Every candidate of N x g x vec mode x pack mode (base gives rounding, mode, k4_tune), ranked:
 * feasible by time_s, then infeasible SUCCESS ones by time_s, then the others in enumeration
 * order (ties: total_bytes, N, g, vec_mode, packed). v7: in every order, tie = 1 marks the
 * feasible candidates whose time_s <= the best's time_s x the calibration's tie_ratio; with
 * order IMD_PERF_ORDER_TIES_BY_FOOTPRINT (opt-in) they come first, by total_bytes (then
 * time_s, N, g, vec_mode, packed); raw_rank is every candidate's place in the time order. Writes
 * min(capacity, n) candidates of candidate_size bytes each and *num_candidates = n.
 * INVALID_VALUE for a bad request or argument; else SUCCESS with each candidate's own status. */
IMD_API imd_status_t imd_perf_rank(const imd_tensor_t* A, const imd_tensor_t* B,
                                   const imd_tensor_t* C, const imd_options_t* base,
                                   const imd_perf_rank_request_t* request,
                                   const imd_perf_calib_t* calib, imd_perf_candidate_t* out,
                                   int32_t capacity, int32_t* num_candidates);

/* ---- Accuracy-aware planning (§6f, host API version 8) -------------------------------------
 * The §3.8 error of a configuration from the caller's operand statistics; immerdink never
 * reads the data. Per output x_ij = sum_h a_ih b_hj of the GEMM view (a_i a row of A, b_j a
 * column of B, k the contracted length), D the emulated product (alpha = 1, beta = 0):
 *   E = 2 delta (nu_a + nu_b) s / d + 4 delta^2 k s^2 / d^2      (EMULATE; PROOFS L3a)
 *       delta = 1/2, d = T - kappa/2 (RN; kappa >= T: +inf) | delta = 1, d = T (TRUNC),
 *       s = sqrt(1 + k 2^-60), nu = min(l1_ratio, sqrt k) (0: sqrt k)
 *   NORMWISE    |D - x| / (||a_i|| ||b_j||) <= E (1 + u) + u                     (u = 2^-53)
 *   ELEMENTWISE |D - x| / |x|               <= E rho (1 + u) + u
 *   ABSOLUTE    |D - x|                     <= (E (1 + u) + u) norm_product
 *   DIRECT dispatch: gamma_k = k u / (1 - k u) in place of E (1 + u) + u (times rho /
 *   norm_product); NOOP / SCALE / k = 0: 0.
 * ABSOLUTE also adds 2^-1075 (EMULATE: a gradual underflow of D) or 2k 2^-1075 (DIRECT: the
 * fma chain's underflows); in C a little more (the subnormal rounding of the computation).
 * Guaranteed for finite inputs when the statistics are true bounds; NORMWISE / ELEMENTWISE
 * also need D (DIRECT: every partial sum) outside the subnormal range (PROOFS L3c).
 * The bound is on the product D. imd_contract writes C = beta C + RN(alpha D) (§3.7):
 *   alpha != 1 adds one rounding: a bound n becomes n (1 + u) + u per metric (ABSOLUTE: pass
 *     |alpha| times the norm_product);
 *   beta != 0 adds u |beta C + alpha D| (the fma), which no statistic here expresses.
 * The TYPICAL estimate (RN EMULATE only; NOT a guarantee) replaces E by
 *   min(E, s / d sqrt(8 ln(2 M / p)) + k s^2 / d^2), M = outputs, p = fail_prob:
 * Hoeffding's bound under independent zero-mean quantization errors, exceeded with
 * probability <= p per call; rows of many equal non-representable entries violate the model.
 * The C value of the bound is >= the exact real bound (double, times (1 + 2^-40)). */
enum { IMD_ERR_NORMWISE = 0, IMD_ERR_ELEMENTWISE = 1, IMD_ERR_ABSOLUTE = 2 };
enum { IMD_ERR_BASIS_BOUND = 0, IMD_ERR_BASIS_TYPICAL = 1 };
enum { IMD_ERR_UNCHECKED = 0, IMD_ERR_MET = 1, IMD_ERR_MISSED = 2 };

typedef struct imd_err_request_s {
  uint32_t struct_size;  /* sizeof(imd_err_request_t)                                       */
  int32_t metric;        /* IMD_ERR_NORMWISE / ELEMENTWISE / ABSOLUTE                       */
  double target;         /* >= 0; 0: no target (estimates reported, err_status UNCHECKED)   */
  double cond;           /* ELEMENTWISE: rho >= 1, upper bound on ||a_i|| ||b_j|| / |x_ij|   */
  double norm_product;   /* ABSOLUTE: > 0, upper bound on ||a_i|| ||b_j||                   */
  double l1_ratio_a;     /* 0 (unknown: sqrt k) or >= 1: upper bound on ||a_i||_1/||a_i||_2  */
  double l1_ratio_b;     /* the same for the columns b_j of B                               */
  int32_t basis;         /* IMD_ERR_BASIS_BOUND (default) / TYPICAL: what meets the target  */
  int32_t reserved;      /* must be 0 (INVALID_VALUE otherwise): reserved for flags     */
  double fail_prob;      /* TYPICAL: p in (0, 1); 0: 2^-40                                  */
} imd_err_request_t;

typedef struct {
  uint32_t struct_size;  /* in: sizeof(imd_err_t) as compiled by the caller                 */
  int32_t status;        /* as returned                                                     */
  int32_t num_moduli;    /* the N estimated (imd_err_min_moduli: the answer; 0: none)       */
  int32_t rounding;      /* IMD_RN / IMD_TRUNC                                              */
  int32_t dispatch;      /* IMD_DISPATCH_*                                                  */
  int32_t err_status;    /* IMD_ERR_UNCHECKED / MET / MISSED                                */
  int64_t k;             /* contracted length of the GEMM view                              */
  int64_t outputs;       /* M = b m n                                                       */
  double bound;          /* rigorous, in the request's metric (+inf: none)                  */
  double typical;        /* typical estimate, same metric                                   */
  double unit_bound;     /* E (EMULATE) or gamma_k (DIRECT): the coefficient of
                            ||a_i|| ||b_j|| before the metric                               */
  double unit_typical;   /* the typical one                                                 */
} imd_err_t;

/* The estimates of one configuration from scalars: num_moduli 2..20, rounding IMD_RN /
 * IMD_TRUNC, dispatch IMD_DISPATCH_*, k >= 0, outputs >= 0; request NULL: NORMWISE, no target.
 * INVALID_VALUE for a bad argument or request (a statistic missing for the metric, cond < 1,
 * l1_ratio in (0, 1), fail_prob outside [0, 1), a negative or non-finite target). */
IMD_API imd_status_t imd_err_estimate(int32_t num_moduli, int32_t rounding, int32_t dispatch,
                                      int64_t k, int64_t outputs,
                                      const imd_err_request_t* request, imd_err_t* out);
/* The smallest N in [n_lo, n_hi] (0 / 0: 2 / 20) whose estimate (request->basis) is <=
 * request->target (> 0), for this contraction with options' rounding and mode (N, g and
 * vec_mode do not matter): one lowering, then the closed form per N. SUCCESS: out at that N.
 * NOT_SUPPORTED with out->err_status MISSED: no N in the range meets the target (out at
 * n_hi, num_moduli 0). The lowering's own status (§5: INVALID_VALUE, or NOT_SUPPORTED for e.g.
 * a repeated or broadcast C label) is returned as such, with out zeroed (err_status
 * UNCHECKED): so err_status tells the two NOT_SUPPORTED cases apart. INVALID_VALUE for a bad
 * request or range. Only the lowering is checked, not the workspace of any N. DIRECT
 * dispatch: N does not matter; the answer is n_lo if met. */
IMD_API imd_status_t imd_err_min_moduli(const imd_tensor_t* A, const imd_tensor_t* B,
                                        const imd_tensor_t* C, const imd_options_t* options,
                                        int32_t n_lo, int32_t n_hi,
                                        const imd_err_request_t* request, imd_err_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IMMERDINK_IMMERDINK_HOST_H */
