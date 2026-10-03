// IndexMap: label lowering and legality (docs/DESIGN.md §5).
//
// Implemented: steps 1-7 (extent agreement, diagonals, repeated-C rejection, extent-1 drop,
// I/J/K/L classification, single-operand labels, C-only labels), step 8 (the coalescing
// role swap, then the canonical label orders in the post-swap frame), step 10 (int64
// additive scatter vectors), and the legality rows 1-5 of the §5 status table.
//
// TODO(M2): step 9 (fusion). It is a performance transformation only (it merges consecutive
// labels without changing the canonical enumeration), so results do not depend on it.
#pragma once

#include <cstdint>
#include <vector>

#include "immerdink/immerdink.h"

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

// The GEMM view of §1.1: b independent m x n x k GEMMs, after the role swap.
struct Lowered {
  std::vector<LabelInfo> labels;  // every distinct label, in first-occurrence order
  int64_t m = 0, n = 0, k = 0, b = 0;
  bool swapped = false;  // §5 step 8 coalescing: computing C^T = B^T A^T (A<->B, I<->J)

  bool empty_output() const { return b == 0 || m == 0 || n == 0; }
};

// Steps 1-8 and legality rows 1-5 of DESIGN §5, in precedence order, except the
// mode-dependent GemmCoord part of row 5 (gemm_dims_ok). Pure host code; never touches
// tensor data. Returns SUCCESS, INVALID_VALUE or NOT_SUPPORTED; `out` is valid only on
// SUCCESS.
imd_status_t lower(const imd_tensor_t& A, const imd_tensor_t& B, const imd_tensor_t& C,
                   Lowered* out);

// §5 row 5, GemmCoord part: m <= 2^31-1 and n_hat <= 2^31-1. DESIGN applies it in EMULATE
// mode with a non-empty output only.
bool gemm_dims_ok(const Lowered& lw);

// Number of int64 entries of the §4a vec region: 2m + 2k + 2n + 3b.
int64_t scatter_vector_count(const Lowered& lw);

// §5 step 8 (canonical orders) and step 10 (scatter vectors) for a non-empty output.
// Writes the §4a vec region contents to out[0, scatter_vector_count): A: row (m), k (k),
// batch (b); B: k (k), col (n), batch (b); C: row (m), col (n), batch (b). Entry q of a
// group's vector is sum_t s_X(label_t) * digit_t(q) over the group's canonical label order
// (first label fastest), in the post-swap frame.
//
// Canonical orders (§5 step 8, post-swap frame, ties by first position in the caller's
// A.labels ++ B.labels ++ C.labels): I by (|s_C|, |s_A|, pos); J by (|s_C|, |s_B|, pos);
// L by (|s_C|, |s_A|, |s_B|, pos); K by (|s_big|, |s_other|, pos), big = A if m >= n.
void build_scatter_vectors(const Lowered& lw, int64_t* out);

}  // namespace imd::detail
