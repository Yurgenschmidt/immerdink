// Exact workspace layout (docs/DESIGN.md §4a; spec/params.py workspace_size /
// region_offsets; spec/Pipeline.tla LayoutExact).
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "export.hpp"

namespace imd::detail {

// PA, PB (v1.21 §6i.2, stage 2): IMD_HWGEMM's FP64 pack buffers; 0 bytes in the other modes.
enum class Region : int { sA, sB, fA, fB, vec, RA, RB, W, PA, PB, count };

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

// v1.21 §6i.2 (spec/IndexMap.tla HwCells): IMD_HWGEMM's K quantum, the pack buffers' k_hat =
// ceil(k)_16 (the GEMM's K tile in every §6i.7 configuration; not EMULATE's k_align).
constexpr int64_t kHwQ = 16;

// IMD_HWGEMM's workspace when the GEMM runs (non-empty output, k >= 2): vec iff need_vec and
// not vec_resident (8 (2m + 2n + 2k + 3b) bytes, as §4a), PA = 8 b m k_hat iff pack_a,
// PB = 8 b n k_hat iff pack_b (rows unpadded), at cumulative 256-aligned offsets in the
// region order; every EMULATE region absent. nullopt on uint64 overflow.
struct HwWorkspaceShape {
  int64_t m = 0, n = 0, k = 0, b = 0;  // lowered GEMM view (post-swap)
  bool need_vec = false, vec_resident = false, pack_a = false, pack_b = false;
};
IMD_HOST_INTERNAL std::optional<WorkspaceLayout> hw_workspace_layout(const HwWorkspaceShape& s);

// The §4a layout, or nullopt if a size or the total overflows uint64 (-> INVALID_VALUE).
//   b*m*n = 0                  -> total 0 (no regions)
//   DIRECT, k = 0, k = 1       -> vec only (k = 1 after lowering means K is empty)
//   otherwise                  -> sA sB fA fB vec RA RB W at cumulative 256-aligned offsets
//   vec_resident               -> the same without vec (DIRECT, k = 0, k = 1: no regions)
// k_align not 16 or 64 -> nullopt (a caller that forgot to pass the resolved A).
IMD_HOST_INTERNAL std::optional<WorkspaceLayout> workspace_layout(const WorkspaceShape& s);

}  // namespace imd::detail
