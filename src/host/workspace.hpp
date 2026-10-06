// Exact workspace layout (docs/DESIGN.md §4a; spec/params.py workspace_size /
// region_offsets; spec/Pipeline.tla LayoutExact).
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "export.hpp"

namespace imd::detail {

enum class Region : int { sA, sB, fA, fB, vec, RA, RB, W, count };

struct WorkspaceShape {
  int64_t m = 0, n = 0, k = 0, b = 0;  // lowered GEMM view (after the role swap)
  int num_moduli = 0;                  // N
  int residue_group = 0;               // g in [1, N] (0 already replaced by N)
  bool direct = false;                 // mode = DIRECT
  bool vec_resident = false;           // vec_mode = IMD_VEC_RESIDENT (v1.2): no vec region
  int32_t packed = 0;                  // v1.7 (§6b.6): bit 0 / 1 = the post-swap A / B is read
                                       // from a packed operand: no sX, fX, RX region
  int32_t k_align = 0;                 // v1.13 §4b: the plan's resolved A (16 or 64; analyze
                                       // passes it, never a default): k_hat = ceil(k)_A
};

struct WorkspaceLayout {
  uint64_t total = 0;                                         // workspace_size
  std::array<uint64_t, static_cast<int>(Region::count)> offset{};  // 256-aligned
  std::array<uint64_t, static_cast<int>(Region::count)> bytes{};   // 0 = region absent
};

// The §4a layout, or nullopt if a size or the total overflows uint64 (-> INVALID_VALUE).
//   b*m*n = 0                  -> total 0 (no regions)
//   DIRECT, k = 0, k = 1       -> vec only (k = 1 after lowering means K is empty)
//   otherwise                  -> sA sB fA fB vec RA RB W at cumulative 256-aligned offsets
//   vec_resident               -> the same without vec (DIRECT, k = 0, k = 1: no regions)
// k_align not 16 or 64 -> nullopt (a caller that forgot to pass the resolved A).
IMD_HOST_INTERNAL std::optional<WorkspaceLayout> workspace_layout(const WorkspaceShape& s);

}  // namespace imd::detail
