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
#include "plan_data.hpp"
#include "workspace.hpp"

namespace imd::detail {

struct Analysis {
  imd_options_t options{};  // residue_group normalized to [1, N]
  Lowered lowered;
  WorkspaceLayout layout;
};

// imd_workspace_size's status: null arguments and options (INVALID_VALUE), the §5 legality
// rows 1-5 in precedence order (index_map lower, then the mode-dependent GemmCoord part),
// then the §4a layout (uint64 / size_t overflow is row 5). Never the RN acceptance rule.
IMD_HOST_INTERNAL imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       Analysis* out);

// The emulated GEMM runs (K1-K4: RN acceptance, K4 parameters): path kGemm with k >= 2.
// kGemm with k = 0 is the beta-only path (§3.7: K0 and one scale kernel).
inline bool emulates(const ExecPlan& e) { return e.path == Path::kGemm && e.k >= 2; }

// v1.4 (DESIGN §4 "KF"): the fused scale-and-pack launch is used iff the emulated GEMM runs
// and one pack tile row covers all of k_hat (k_hat <= kFusedKHatMax: a block owns whole
// rows). A pure function of the lowering: deterministic at plan time.
constexpr int64_t kFusedKHatMax = 1024;  // 512 until A3 (DESIGN v1.5)
inline bool fused_path(const ExecPlan& e) { return emulates(e) && e.k_hat <= kFusedKHatMax; }

// Kernel launches of one imd_contract with alpha != 0 (DESIGN §4, §6a kernel_launches):
// NOOP 0; DIRECT / SCALE: K0 (generate mode) + 1; EMULATE: fused ? KF : K0 (generate) + 2 K1,
// then per residue group not packed by KF two packs (+ one launch zeroing the B rows
// n..n_hat-1 if n_hat > n), the K4 launches, and K5. `fused` and `resident` are the plan's
// choices (fused_path, vec_mode = IMD_VEC_RESIDENT); `k4_launches` is plan_k4_counts.
IMD_HOST_INTERNAL int64_t plan_launches(const ExecPlan& e, bool fused, bool resident,
                                        int64_t k4_launches);

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
