// IndexMap: label lowering and legality (docs/DESIGN.md §5).
//
// Steps 1-7 (extent agreement, diagonals, repeated-C rejection, extent-1 drop, I/J/K/L
// classification, single-operand labels, C-only labels), step 8 (the coalescing role swap,
// then the canonical label orders in the post-swap frame), step 9 (fusion), step 10 (int64
// additive scatter vectors, built from the fused modes) and the legality rows 1-5 of the
// §5 status table. Rows 6-7 (ALIASING, MISALIGNED) are execute-time checks (api.cpp).
//
// Fusion merges consecutive labels of a group without changing the canonical enumeration,
// so the scatter vectors (and every result) are the same with or without it; it shortens
// the mode lists that the pack's pivot choice (pack_pivot) works with.
#pragma once

#include <cstdint>
#include <vector>

#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

enum class LabelClass : uint8_t { I, J, K, L };

// One distinct label after steps 1-2 (diagonals merged: strides summed per tensor).
struct LabelInfo {
  int32_t label = 0;
  int64_t extent = 0;
  int count[3] = {0, 0, 0};       // occurrences in A, B, C
  int64_t stride[3] = {0, 0, 0};  // merged stride per tensor; 0 where absent (step 6)
  int first_pos = 0;              // first position in A.labels ++ B.labels ++ C.labels
  LabelClass cls = LabelClass::K; // steps 5-7 (extent-1 labels are dropped by step 4)
};

// One mode of a group after step 9: extent and stride per tensor (post-swap A, B, C; 0
// where the tensor does not carry the group or carries it with stride 0).
struct FusedMode {
  int64_t extent = 0;
  int64_t stride[3] = {0, 0, 0};
};

// The GEMM view of §1.1: b independent m x n x k GEMMs, after the role swap.
struct Lowered {
  std::vector<LabelInfo> labels;  // every distinct label, in first-occurrence order
  int64_t m = 0, n = 0, k = 0, b = 0;
  bool swapped = false;  // §5 step 8 coalescing: computing C^T = B^T A^T (A<->B, I<->J)
  // Indexed by LabelClass (I, J, K, L), post-swap frame, extent-1 labels dropped:
  std::vector<int32_t> order[4];    // step 8 canonical label order (first fastest)
  std::vector<FusedMode> fused[4];  // step 9 fused modes, in canonical order

  bool empty_output() const { return b == 0 || m == 0 || n == 0; }
};

// Steps 1-9 and legality rows 1-5 of DESIGN §5, in precedence order, except the
// mode-dependent GemmCoord part of row 5 (gemm_dims_ok). Pure host code; never touches
// tensor data. Returns SUCCESS, INVALID_VALUE or NOT_SUPPORTED; `out` is valid only on
// SUCCESS.
//
// Canonical orders (§5 step 8, post-swap frame, ties by first position in the caller's
// A.labels ++ B.labels ++ C.labels): I by (|s_C|, |s_A|, pos); J by (|s_C|, |s_B|, pos);
// L by (|s_C|, |s_A|, |s_B|, pos); K by (|s_big|, |s_other|, pos), big = A if m >= n.
// Fusion (§5 step 9): consecutive modes g, g' of a group merge iff s_X(g') = s_X(g) e(g)
// for every tensor X carrying the group (I: A, C; J: B, C; K: A, B; L: A, B, C), stride-0
// carriers included.
IMD_HOST_INTERNAL imd_status_t lower(const imd_tensor_t& A, const imd_tensor_t& B,
                                     const imd_tensor_t& C, Lowered* out);

// Descriptor sanity (rank >= 0, arrays non-null for rank > 0, extents >= 0), emptiness (a zero
// extent) and §5 row 5's span rule (sum_d |s_d| (e_d - 1) of a non-empty tensor fits int64);
// shared with the pack descriptor checks (pack.cpp).
bool descriptor_ok(const imd_tensor_t& t);
bool tensor_empty(const imd_tensor_t& t);
bool span_fits(const imd_tensor_t& t);

// The LabelInfo of a label (nullptr if absent).
IMD_HOST_INTERNAL const LabelInfo* find_label(const Lowered& lw, int32_t label);

// DESIGN v1.7 §6b.2: replace group g's canonical order by `order` (a permutation of
// lw->order[g]: the order a plan adopts from a pack) and redo step 9's fusion in it. Any
// order within a class is a valid GEMM view (spec IndexMap, adopted orders).
IMD_HOST_INTERNAL void adopt_order(Lowered* lw, int g, const std::vector<int32_t>& order);

// §5 row 5, GemmCoord part: m <= 2^31-1 and n_hat <= 2^31-1. DESIGN applies it in EMULATE
// mode with a non-empty output and k >= 2 only (when the emulated GEMM runs).
IMD_HOST_INTERNAL bool gemm_dims_ok(const Lowered& lw);

// Number of int64 entries of the §4a vec region: 2m + 2k + 2n + 3b.
IMD_HOST_INTERNAL int64_t scatter_vector_count(const Lowered& lw);

// §5 step 10 (scatter vectors) for a non-empty output. Writes the §4a vec region contents
// to out[0, scatter_vector_count): A: row (m), k (k), batch (b); B: k (k), col (n), batch
// (b); C: row (m), col (n), batch (b). Entry q of a group's vector is
// sum_t s_X(mode_t) * digit_t(q) over the group's fused modes (first fastest), in the
// post-swap frame.
IMD_HOST_INTERNAL void build_scatter_vectors(const Lowered& lw, int64_t* out);

// Gather tiling of the K2/K3 pack for operand X (0 = post-swap A, 1 = post-swap B). The
// pack enumerates the operand's rows as rho = r + rows * lambda (a mixed radix over the
// row group's fused modes, then L's) and k-bar. Reads are coalesced along X's
// smallest-|stride| mode: if that is a row or batch mode u whose |stride| is below the
// k-bar step |s_X(first K mode)|, the tile's rows are rho = lo + radix * d with radix =
// the product of the extents before u, and a warp reads along d; otherwise a warp reads
// along k-bar (radix 1). Zero strides are skipped (a broadcast read is one transaction).
struct PackPivot {
  bool along_rows = false;
  int64_t radix = 1;
};
IMD_HOST_INTERNAL PackPivot pack_pivot(const Lowered& lw, int X);

// K5 thread mapping (NOTES-M2M3 #4): true iff C's smallest-|stride| mode is a batch (L)
// mode rather than a J mode (after the §5 step 8 swap it is never in I). Canonical orders
// put each group's smallest |s_C| first, so the first fused modes are compared.
IMD_HOST_INTERNAL bool k5_batch_fastest(const Lowered& lw);

}  // namespace imd::detail
