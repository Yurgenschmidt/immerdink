// IndexMap: label lowering and legality (docs/DESIGN.md §5).
//
// M0 implements what the workspace size needs: steps 1-7 (extent agreement, diagonals,
// repeated-C rejection, extent-1 drop, I/J/K/L classification, single-operand labels,
// C-only labels), the coalescing role swap of step 8 (it decides which side is padded to
// n_hat, so it changes the workspace), and the legality rows 1-5 of the §5 status table.
//
// TODO(M2): the rest of step 8 (canonical label orders), step 9 (fusion) and step 10
// (int64 scatter vectors); see spec/IndexMap.tla `Lower`.
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

// Steps 1-8 (partial, see above) and legality rows 1-5 of DESIGN §5, in precedence
// order. Pure host code; never touches tensor data. Returns SUCCESS, INVALID_VALUE or
// NOT_SUPPORTED; `out` is valid only on SUCCESS.
imd_status_t lower(const imd_tensor_t& A, const imd_tensor_t& B, const imd_tensor_t& C,
                   Lowered* out);

}  // namespace imd::detail
