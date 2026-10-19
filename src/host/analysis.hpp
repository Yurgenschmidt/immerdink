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
  // v1.13 §4b.5: the K alignment A, resolved ONCE per API call by analyze (k_align_of of the
  // lowered k: the test hook if set, else AUTO) and passed explicitly from then on: the
  // workspace layout, the slot keys' format, plan_data (ExecPlan::k_align).
  int32_t k_align = 0;
  Lowered lowered;
  WorkspaceLayout layout;
  PackedSlots packed;       // v1.7: none unless analyze_packed found packs
};

// imd_workspace_size's status: null arguments and options (INVALID_VALUE), the §5 legality
// rows 1-5 in precedence order (index_map lower, then the mode-dependent GemmCoord part),
// then the §4a layout (uint64 / size_t overflow is row 5). Never the RN acceptance rule.
// v1.13 §4b.5: the one resolution of A for a plan or sizing query (out->k_align =
// sel.of(k)). `sel` is the API call's one hook snapshot (k_align_sel(), taken at the call's
// entry and passed to every analysis of the call); the overload without it takes the
// snapshot itself (an entry point with this single analysis).
IMD_HOST_INTERNAL imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       const KAlignSel& sel, Analysis* out);
IMD_HOST_INTERNAL imd_status_t analyze(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, const imd_options_t* options,
                                       Analysis* out);

// v1.7 (DESIGN §6b.5): imd_workspace_size's checks (analyze), then, for each caller operand
// with a pack (src[x] non-null), the plan-time compatibility check: PACK_MISMATCH (reason in
// out->packed.mismatch), in this order (DESIGN v1.13 §4b.5 / §6b.5, the oracle's
// lower_packed, PackedOperand Reason): per pack (the caller's A, then B) an unknown format
// word (FORMAT), N or rounding differing from the options (MODULI, ROUNDING), labels that do
// not map onto X's by mode position or plan K, L or X-row classes other than the pack's (a K
// label only in the partner included: CLASSES); (both packed) the two packs' K or L orders
// differing (ORDER); then the plan ADOPTS the packs' orders of K, L and each packed
// operand's row class (adopt_order: fusion redone), and each slot key, computed from the
// adopted lowering restricted to X with the PLAN's own format word (of out->k_align), must
// equal the pack's key: the other words of both packs first (EXTENTS / STRIDES), then a
// format 1 vs 2 difference of either (FORMAT); after a match, the pack's k_hat must equal the
// plan's (FORMAT; a drift check). The workspace layout omits sX, fX, RX of every packed
// post-swap role (also when the status is PACK_MISMATCH). With no pack it is analyze. Never
// the §3.2 kappa rule.
IMD_HOST_INTERNAL imd_status_t analyze_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                              const imd_tensor_t* C,
                                              const imd_options_t* options,
                                              const KAlignSel& sel,
                                              const PackSource* const src[2], Analysis* out);
IMD_HOST_INTERNAL imd_status_t analyze_packed(const imd_tensor_t* A, const imd_tensor_t* B,
                                              const imd_tensor_t* C,
                                              const imd_options_t* options,
                                              const PackSource* const src[2], Analysis* out);

// analyze_packed after its analyze: `out` holds a SUCCESS analyze of (A, B, C, options); the
// pack checks, the adopted orders, the slot keys and the packed layout as described above.
IMD_HOST_INTERNAL imd_status_t match_packs(const imd_tensor_t* A, const imd_tensor_t* B,
                                           const PackSource* const src[2], Analysis* out);

// The analysis of a host query with an imd_packing_t of pack DESCRIPTORS (host_api.cpp): no
// packing (or both members NULL) is analyze; otherwise analyze, then each descriptor's own
// status (pack_lower: imd_pack_layout's INVALID_VALUE / NOT_SUPPORTED), then match_packs
// (analyze runs once). The plan and every descriptor resolve A from the same `sel`.
IMD_HOST_INTERNAL imd_status_t analyze_with(const imd_tensor_t* A, const imd_tensor_t* B,
                                           const imd_tensor_t* C, const imd_options_t* options,
                                           const KAlignSel& sel, const imd_packing_t* packing,
                                           Analysis* an);

// imd_pack_desc_for's body (host_api.cpp) with the caller's hook snapshot (imd_perf_rank
// builds every candidate's descriptors under its one snapshot).
IMD_HOST_INTERNAL imd_status_t pack_desc_for(const imd_tensor_t* A, const imd_tensor_t* B,
                                            const imd_tensor_t* C, const imd_options_t* options,
                                            const KAlignSel& sel, int32_t operand,
                                            imd_pack_desc_t* out, int32_t* labels,
                                            int32_t capacity);

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

// v1.8 (DESIGN §4, §6c): the moduli parts of a pack launch (KF, K2 / K3; gridDim.y): with
// fewer tiles than 2 per SM, the group's moduli are split over up to gsize block rows so that
// about 2 blocks per SM run (a launch choice only). contract.cu and the performance model use
// this one rule.
inline int64_t pack_moduli_parts(int64_t tiles, int gsize, int sms) {
  const int64_t want = 2 * int64_t{sms};
  if (tiles >= want) return 1;
  const int64_t parts = (want + tiles - 1) / tiles;
  return parts < gsize ? parts : gsize;
}

// v1.8 (DESIGN §4 KF, §6c): KF's launch shape. log2 of its tile length tk = pow2 >= k_hat
// (>= 16); a tile holds 2^lg_te FP64 (th = 2^(lg_te - lg_tk) rows): 4096 for the E = 16 kernel
// (16 values per thread), 1024 for the E = 4 one, which runs when the E = 16 grid would have
// fewer than kKfSmallBlocksPerSm blocks per SM even with every modulus in its own block row.
// The tiles of operand x: radix_x ceil((b rows_x / radix_x) / th) (none for a packed one).
// contract.cu launch_kf and the performance model (perf.cpp) use these rules.
constexpr int64_t kKfSmallBlocksPerSm = 16;
inline int kf_lg_tk(int64_t k_hat) {
  int lg_tk = 4;
  while ((int64_t{1} << lg_tk) < k_hat) ++lg_tk;
  return lg_tk;
}
inline int64_t kf_operand_tiles(int lg_te, int lg_tk, int64_t b, int64_t rows, int64_t radix) {
  const int64_t th = int64_t{1} << (lg_te - lg_tk);
  const int64_t D = b * rows / radix;
  return radix * ((D + th - 1) / th);
}
inline bool kf_small(int64_t tiles_4096, int gsize, int sms) {
  return tiles_4096 * gsize < kKfSmallBlocksPerSm * int64_t{sms};
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

// ---- Issue 21 (docs/NOTES-21.md): K1's L2 budgets are fractions of the card's L2 ----
// K1 (the row-scaling kernel) reads a row twice; the second pass hits L2 only while the rows in flight fit there. Two byte budgets
// decide how many rows it keeps in flight: the grid cap of a two-pass kernel (about 40 MiB of the RTX 4090's 72 MiB L2: 5/9) and
// the budget of the long-row variants (64 MiB: 8/9, "the L2 less what the rest of the GPU keeps there"). They were constants;
// they are now these fractions of the device's L2, which gives exactly the old values on the 4090 and 44.4 / 27.8 MiB (H100 and H200, 50 MiB: one GH100 die; R1 records
// the value the driver reports) and 113.8 / 71.1 MiB (RTX PRO 6000, 128 MiB). l2_bytes <= 0: the reference 72 MiB.
constexpr int64_t kReferenceL2Bytes = int64_t{72} << 20;
inline double k1_l2_budget_bytes(int64_t l2_bytes) {
  return static_cast<double>(l2_bytes > 0 ? l2_bytes : kReferenceL2Bytes) * 8.0 / 9.0;
}
inline double k1_l2_fit_bytes(int64_t l2_bytes) {
  return static_cast<double>(l2_bytes > 0 ? l2_bytes : kReferenceL2Bytes) * 5.0 / 9.0;
}

// ---- v1.10 (DESIGN §6c.1 "Waves and occupancy"): the launch shapes of the other kernels ----
// contract.cu launches with these rules and the performance model (perf.cpp) computes each
// launch's waves and per-thread chain from them: one implementation of every grid.

// The grid of a grid-stride kernel of 256-thread blocks (K0, DIRECT, beta-only, K5, the
// zero-rows and pack-prep launches; per_block = 1 for one block per tile): ceil(work /
// per_block) blocks, at least 1, at most 2^20 (grid-stride loops beyond).
constexpr int kLaunchThreads = 256;
constexpr int64_t kLaunchMaxBlocks = int64_t{1} << 20;
inline int64_t launch_blocks(int64_t work, int64_t per_block = kLaunchThreads) {
  const int64_t b = (work + per_block - 1) / per_block;
  return b < 1 ? 1 : b > kLaunchMaxBlocks ? kLaunchMaxBlocks : b;
}

// The pack tile (K2 / K3, KL, KF): 2^12 FP64 values of th rows x tk k-bar.
constexpr int kPackTileLgElems = 12;

// log2 of the K2 / K3 pack tile height: 64 rows read along k-bar; read along rows, the
// smallest power of two >= D in [8, 64] (D = b rows / radix).
inline int pack_tile_lg_rows(bool along_rows, int64_t D) {
  if (!along_rows) return 6;
  int lg = 3;
  while (lg < 6 && (int64_t{1} << lg) < D) ++lg;
  return lg;
}

// One K2 / K3 launch (k23_pack) of a residue group of gsize moduli: tiles of th x tk over the
// operand's radix x ceil(D / th) row tiles and ceil(k_hat / tk) k-bar tiles; grid
// (launch_blocks(tiles, 1), parts), 256 threads, 16 values per thread per tile.
struct PackLaunch {
  int lg_th = 6;
  int64_t tiles = 0, grid_x = 1, parts = 1;
};
inline PackLaunch pack_launch(bool along_rows, int64_t b, int64_t rows, int64_t radix,
                              int64_t k_hat, int gsize, int sms) {
  PackLaunch L;
  const int64_t D = b * rows / radix;
  L.lg_th = pack_tile_lg_rows(along_rows, D);
  const int64_t th = int64_t{1} << L.lg_th, tk = (int64_t{1} << kPackTileLgElems) / th;
  L.tiles = radix * ((D + th - 1) / th) * ((k_hat + tk - 1) / tk);
  L.grid_x = launch_blocks(L.tiles, 1);
  L.parts = pack_moduli_parts(L.tiles, gsize, sms);
  return L;
}

// One KL launch for an operand of `rows` rows per batch (DESIGN v1.6 §4): tile height th =
// 2^lg_th (lg_rows >= 0: imd_pack's or a test's choice, else kl_tile_lg_k), chunks per row nc =
// ceil(k_hat / tk), tiles = radix ceil(D / th), grid min(tiles, 2^20); rows held on chip (nc
// <= 2): 8 values per thread, 512 threads (2 CTAs per SM by launch bounds), else 4 values per
// thread, 1024 threads.
struct KlLaunch {
  int lg_th = 0, values = 8, threads = 512;
  int64_t nc = 1, tiles = 0, grid = 1;
};
inline KlLaunch kl_launch(bool along_rows, int32_t lg_rows, int64_t k_hat, int64_t b,
                          int64_t rows, int64_t radix) {
  KlLaunch L;
  L.lg_th = lg_rows >= 0 ? lg_rows : kPackTileLgElems - kl_tile_lg_k(along_rows, k_hat);
  const int64_t th = int64_t{1} << L.lg_th, tk = (int64_t{1} << kPackTileLgElems) / th;
  L.nc = (k_hat + tk - 1) / tk;
  const int64_t D = b * rows / radix;
  L.tiles = radix * ((D + th - 1) / th);
  L.grid = L.tiles < 1 ? 1 : L.tiles > kLaunchMaxBlocks ? kLaunchMaxBlocks : L.tiles;
  L.values = L.nc > 2 ? 4 : 8;
  L.threads = (1 << kPackTileLgElems) / L.values;
  return L;
}

// imd_pack's KL tile height (pack_exec's one-operand view, read along rows): th = 4096 /
// pow2(k_hat) rows side by side, 8 .. 64 (tk 512 .. 64); -1 (kl_tile_lg_k) read along k-bar.
inline int32_t pack_kl_lg_rows_for(bool along_rows, int64_t k_hat) {
  if (!along_rows) return -1;
  int lg_tk = 6;
  while (lg_tk < 9 && (int64_t{1} << lg_tk) < k_hat) ++lg_tk;
  return kPackTileLgElems - lg_tk;
}

// K5's launch (A5, NOTES-A5 §2): lambda-fastest plans (k5_batch_fast) take tiles of tl x tj =
// 1024 outputs (tl = pow2 >= b in [2, 32]), 4 per thread; the others 1, 2 or 4 consecutive
// j-bar per thread: 1 while b m n <= one_elem_per_sm / 2 per SM, 2 while b m n <= 8
// one_elem_per_sm per SM, else 4 (elems_override > 0: forced; contract.cu passes its test
// hooks, the model kK5OneElemPerSm and 0). variant: 0 / 1 / 2 = E 1 / 2 / 4, 3 lambda-fastest;
// work = items (E outputs each, grid of 256-thread blocks) or tiles (one block per tile).
constexpr int64_t kK5OneElemPerSm = 2048;
struct K5Launch {
  int variant = 2, elems = 4, lg_tl = 1;
  int64_t work = 0, grid = 1;
};
inline K5Launch k5_launch(bool batch_fast, int64_t b, int64_t m, int64_t n, int sms,
                          int64_t one_elem_per_sm = kK5OneElemPerSm, int elems_override = 0) {
  K5Launch L;
  if (batch_fast) {
    int lg_tl = 1;
    while (lg_tl < 5 && (int64_t{1} << lg_tl) < b) ++lg_tl;
    const int64_t tl = int64_t{1} << lg_tl, tj = 1024 / tl;
    L.variant = 3, L.elems = 4, L.lg_tl = lg_tl;
    L.work = m * ((b + tl - 1) / tl) * ((n + tj - 1) / tj);
    L.grid = launch_blocks(L.work, 1);
    return L;
  }
  const int64_t mn = b * m * n;
  const int e = elems_override > 0 ? elems_override
                : mn <= one_elem_per_sm / 2 * sms ? 1
                : mn <= 8 * one_elem_per_sm * sms ? 2 : 4;
  L.elems = e;
  L.variant = e == 1 ? 0 : e == 2 ? 1 : 2;
  L.work = b * m * ((n + e - 1) / e);
  L.grid = launch_blocks(L.work);
  return L;
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
// unreachable). The GEMM view (m, n, k, b, n_hat, k_hat, N, g, swapped, trunc), the path,
// k_align and the layout are set whatever the status. v1.13 §4b: k_align = an.k_align for a
// plan with residue rows (EMULATE, non-empty output, k >= 2), else 16 (NOOP, SCALE, DIRECT,
// K-empty, whatever the hook says); k_hat = ceil(k)_k_align.
IMD_HOST_INTERNAL imd_status_t plan_data(const Analysis& an, ExecPlan* e);

// The K4 launch counts of a plan (EMULATE path; zeros otherwise) under `lim`.
IMD_HOST_INTERNAL K4Counts plan_k4_counts(const ExecPlan& e, const LaunchLimits& lim);

// v1.7 (DESIGN §6b, imd_pack): the launch data of packing one operand, as an ExecPlan of
// one GEMM operand (post-swap A, rows m = r, g = N: every plane is "group 1"): path kGemm,
// m = r, n = 0, k, b, k_hat, N, the §3.2 step 4 constants of (N, k, rounding), the pack
// pivot of X (pack_pivot over X's R and L modes), and the fused-mode descriptor of X's
// classes R (as I), K and L with X's strides in slot A; vec_count = r + k + b (the rows, k
// and batch vectors only); k_align = the pack's. Requires a SUCCESS pack_lower. Its device
// bytes are 0.
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
