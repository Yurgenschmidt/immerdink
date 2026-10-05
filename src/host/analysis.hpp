// Plan analysis (DESIGN §3.2, §4, §4a, §5, §6a): the single implementation of argument
// checks, lowering, legality, the RN acceptance rule, the workspace layout and the plan's
// sizes. libimmerdink_host's C ABI (host_api.cpp) and libimmerdink's imd_plan_create
// (api.cpp) both call these functions; neither has another copy.
#pragma once

#include <cstdint>

#include "export.hpp"
#include "immerdink/immerdink_host.h"
#include "index_map.hpp"
#include "k4_config.hpp"
#include "k4_launches.hpp"
#include "pack.hpp"
#include "plan_data.hpp"
#include "workspace.hpp"

namespace imd::detail {

// v1.7 (DESIGN §6b): the packed slots of a plan (analyze_packed).
struct PackedSlots {
  int32_t caller = 0;    // bit x: the caller's operand x (0 = A, 1 = B) is packed
  int32_t role = 0;      // bit x: the post-swap operand x is packed
  int32_t mismatch = 0;  // IMD_PACK_MISMATCH_* of a PACK_MISMATCH status
  std::vector<int64_t> key[2];  // slot keys per caller operand (SUCCESS; empty: in-line)
  PackLayout layout[2];         // per post-swap role: the pack's layout (SUCCESS)
};

struct Analysis {
  imd_options_t options{};  // residue_group normalized to [1, N]
  Lowered lowered;
  WorkspaceLayout layout;
  PackedSlots packed;       // v1.7: none unless analyze_packed found packs
};

// imd_workspace_size's status: null arguments and options (INVALID_VALUE), the §5 legality
// rows 1-5 in precedence order (index_map lower, then the mode-dependent GemmCoord part),
// then the §4a layout (uint64 / size_t overflow is row 5). Never the RN acceptance rule.
IMD_HOST_INTERNAL imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       Analysis* out);

// v1.7 (DESIGN §6b.5): imd_workspace_size's checks (analyze), then, for each caller operand
// with a pack (src[x] non-null), the plan-time compatibility check: PACK_MISMATCH (reason in
// out->packed.mismatch) if the pack's N or rounding differ from the options, if its labels
// do not map onto X's by mode position, if the plan's K, L or X-row classes differ from the
// pack's (a K label only in the partner included), or (both packed) if the two packs' K or L
// orders differ; then the plan ADOPTS the packs' orders of K, L and each packed operand's
// row class (adopt_order: fusion redone) and each slot key, computed from the adopted
// lowering restricted to X, must equal the pack's key (EXTENTS / STRIDES). The workspace
// layout omits sX, fX, RX of every packed post-swap role (also when the status is
// PACK_MISMATCH). With no pack it is analyze. Never the §3.2 kappa rule.
IMD_HOST_INTERNAL imd_status_t analyze_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                              const imd_tensor_t* C,
                                              const imd_options_t* options,
                                              const PackSource* const src[2], Analysis* out);

// The analysis of a host query with an imd_packing_t of pack DESCRIPTORS (host_api.cpp): no
// packing (or both members NULL) is analyze; otherwise analyze, then each descriptor's own
// status (pack_lower: imd_pack_layout's INVALID_VALUE / NOT_SUPPORTED), then analyze_packed.
IMD_HOST_INTERNAL imd_status_t analyze_with(const imd_tensor_t* A, const imd_tensor_t* B,
                                           const imd_tensor_t* C, const imd_options_t* options,
                                           const imd_packing_t* packing, Analysis* an);

// kappa >= T(N) (§3.2 RN acceptance), exactly.
IMD_HOST_INTERNAL bool kappa_at_least_T(uint64_t kappa, int N);

// The emulated GEMM runs (K1-K4: RN acceptance, K4 parameters): path kGemm with k >= 2.
// kGemm with k = 0 is the beta-only path (§3.7: K0 and one scale kernel).
inline bool emulates(const ExecPlan& e) { return e.path == Path::kGemm && e.k >= 2; }

// v1.4 (DESIGN §4 "KF"): the fused scale-and-pack launch is used iff the emulated GEMM runs
// and one pack tile row covers all of k_hat (k_hat <= kFusedKHatMax: a block owns whole
// rows). A pure function of the lowering: deterministic at plan time. v1.7 (§6b.5): KF
// scales and packs the in-line operands only, so a plan with both operands packed has none.
constexpr int64_t kFusedKHatMax = 1024;  // 512 until A3 (DESIGN v1.5)
inline bool fused_path(const ExecPlan& e) {
  return emulates(e) && e.k_hat <= kFusedKHatMax && e.packed != 3;
}

// v1.6 (DESIGN §4 "KL"): the operands scaled and packed (group 1) by the fused large launch
// KL, as a bit mask (bit 0: the GEMM's A, bit 1: its B, post-swap): every in-line (v1.7: not
// packed) operand of a plan with the emulated GEMM that does not take KF. A pure function of
// the lowering.
constexpr int32_t kFusedLargeA = 1, kFusedLargeB = 2;
inline int32_t fused_large(const ExecPlan& e) {
  if (!emulates(e) || fused_path(e)) return 0;
  return (kFusedLargeA | kFusedLargeB) & ~e.packed;
}

// v1.8 (DESIGN §4 KL, §6c): log2 of KL's tile length tk along k-bar. A KL tile holds 4096 FP64
// of th rows x tk: read along rows, th = 8 rows side by side (tk = 512); along k-bar, tk =
// pow2 >= k_hat in [64, 4096] (th = 4096 / tk). Rows longer than two tiles re-read their middle
// chunks from L2. contract.cu launch_kl and the performance model (perf.cpp) use this one rule.
inline int kl_tile_lg_k(bool along_rows, int64_t k_hat) {
  if (along_rows) return 9;
  int lg_tk = 6;
  while (lg_tk < 12 && (int64_t{1} << lg_tk) < k_hat) ++lg_tk;
  return lg_tk;
}

// Kernel launches of one imd_contract with alpha != 0 (DESIGN §4, §6a kernel_launches):
// NOOP 0; DIRECT / SCALE: K0 (generate mode) + 1; EMULATE: fused ? KF : K0 (generate) + one
// scale launch per operand (K1 of it, or v1.6 KL), then per residue group two packs (+ one
// launch zeroing the B rows n..n_hat-1 if n_hat > n), except group 1's packs of KF (both
// operands) or of the KL operands (KL_B zeroes those rows itself), the K4 launches, and K5.
// v1.7: a packed operand (e.packed) has no scale and no pack launch.
// `fused`, `fused_large` and `resident` are the plan's choices (fused_path, fused_large,
// vec_mode = IMD_VEC_RESIDENT); `k4_launches` is plan_k4_counts.
IMD_HOST_INTERNAL int64_t plan_launches(const ExecPlan& e, bool fused, int32_t fused_large,
                                        bool resident, int64_t k4_launches);

// kappa = ceil(sqrt(k)) exactly (k >= 1; 0 for k = 0), DESIGN §3.2.
IMD_HOST_INTERNAL uint64_t ceil_sqrt_u64(uint64_t k);

// The CUDA-free part of imd_plan_create after a successful analyze: §3.2 RN acceptance
// (kappa >= T with the emulated GEMM -> NOT_SUPPORTED), the dispatch path, the GEMM view,
// the §3.2 step 4 constants, the pack and K5 mappings, the K0 fused-mode descriptor and the
// plan's device bytes. Leaves sms, k4, vec_resident and vec_dev to libimmerdink. Returns
// SUCCESS, NOT_SUPPORTED, or INVALID_VALUE (more than kMaxVecModes fused modes:
// unreachable). The GEMM view (m, n, k, b, n_hat, k_hat, N, g, swapped, trunc), the path
// and the layout are set whatever the status.
IMD_HOST_INTERNAL imd_status_t plan_data(const Analysis& an, ExecPlan* e);

// The K4 launch counts of a plan (EMULATE path; zeros otherwise) under `lim`.
IMD_HOST_INTERNAL K4Counts plan_k4_counts(const ExecPlan& e, const LaunchLimits& lim);

// v1.7 (DESIGN §6b, imd_pack): the launch data of packing one operand, as an ExecPlan of
// one GEMM operand (post-swap A, rows m = r, g = N: every plane is "group 1"): path kGemm,
// m = r, n = 0, k, b, k_hat, N, the §3.2 step 4 constants of (N, k, rounding), the pack
// pivot of X (pack_pivot over X's R and L modes), and the fused-mode descriptor of X's
// classes R (as I), K and L with X's strides in slot A; vec_count = r + k + b (the rows, k
// and batch vectors only). Requires a SUCCESS pack_lower. Its device bytes are 0.
IMD_HOST_INTERNAL void pack_exec(const PackLowered& pl, ExecPlan* e);

// The plan's host payload (DESIGN §6a "plan host bytes"), for a plan built from these
// descriptors and this analysis and plan data, with the production K4 limits:
//   sizeof(imd_plan_st)
//   + (8 + 8 + 4) (rank_A + rank_B + rank_C)                 descriptor copies
//   + sizeof(LabelInfo) labels + 4 (labels in canonical orders)
//   + sizeof(FusedMode) (fused modes)                          lowering tables
//   + sizeof(VecMode) (fused modes)                            K0 descriptor, unless NOOP
//   + kK4PlanBytes + kK4StoreEntryBytes store_shapes + kK4AddEntryBytes add_shapes
//                                                              K4 parameters, EMULATE only
IMD_HOST_INTERNAL uint64_t plan_host_bytes(const imd_tensor_t& A, const imd_tensor_t& B,
                                           const imd_tensor_t& C, const Analysis& an,
                                           const ExecPlan& e);

// sizeof(K4Plan) and of one std::map entry of its store-mod / add-mod kernel parameter
// maps, for this build (gemm_k4.cu static_asserts these against the real types).
constexpr uint64_t kK4PlanBytes = 184;
constexpr uint64_t kK4StoreEntryBytes = 280;
constexpr uint64_t kK4AddEntryBytes = 280;

}  // namespace imd::detail
