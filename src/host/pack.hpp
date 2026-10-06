// Packed operands (DESIGN v1.7 §6b, ROADMAP D1): the pack descriptor's lowering and
// statuses (§6b.3, §6b.5), the byte layout of formats 1 and 2 (§6b.4, v1.13 §4b: k_hat =
// ceil(k)_16 / ceil(k)_64, the pack's format is AUTO(k) of its k), the compatibility key (§6b.2)
// and the plan's view of a pack. CUDA-free; part of libimmerdink_host, the single source of
// truth of every pack size, key and status (libimmerdink's imd_pack_create / imd_pack /
// imd_plan_create_packed use exactly these functions).
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "export.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {

// v1.13 §4b: there is no single pack format any more. A pack's format is that of its resolved
// A (k_hat.hpp pack_format_of); a plan's slot key carries the plan's own format.
// The pack's label classes (§6b.1): K contracted, L batch, R rows.
enum : int { kPackK = 0, kPackL = 1, kPackR = 2 };
// Regions of the packed buffer (§6b.4), in layout order.
enum : int { kPackSig = 0, kPackFlag = 1, kPackRes = 2 };

// §6b.4: sig (2 b r), flag (b r), res (N b r k_hat + (ceil(r)_16 - r) k_hat) at 256-aligned
// offsets, k_hat = ceil(k)_A with A = k_align (16: format 1, 64: format 2; the caller passes
// the resolved A); total = the sum of the aligned sizes; all 0 when b r k = 0. nullopt when a
// size or the total exceeds uint64 (INVALID_VALUE) or k_align is not 16 or 64.
struct PackLayout {
  uint64_t offset[3] = {0, 0, 0}, bytes[3] = {0, 0, 0};
  uint64_t total = 0;
};
IMD_HOST_INTERNAL std::optional<PackLayout> pack_layout(int64_t b, int64_t r, int64_t k, int N,
                                                       int32_t k_align);

// imd_pack's workspace (an implementation choice, §6b.5 pack_ws_bytes): X's scatter vectors
// (rows r, k-bar k, batches b; int64): 8 (r + k + b) aligned to 256; 0 for an empty pack.
// nullopt on uint64 overflow.
IMD_HOST_INTERNAL std::optional<uint64_t> pack_ws_bytes(int64_t b, int64_t r, int64_t k);

// One fused mode of X in a pack class: extent and X's stride (elements).
struct PackMode {
  int64_t extent = 0, stride = 0;
};

// What imd_pack_layout / imd_pack_create derive from a descriptor.
struct PackLowered {
  bool lowered = false;  // the sizes and the layout are valid (SUCCESS or NOT_SUPPORTED)
  int N = 0;
  int32_t rounding = IMD_RN;
  int64_t b = 0, r = 0, k = 0, k_hat = 0;
  int32_t k_align = 0;  // v1.13 §4b: the pack's resolved A (k_align_of(k), once): 16 or 64
  int64_t format = 0;   // its format word (1 = A16, 2 = A64), key word 0
  uint64_t kappa = 0;  // ceil(sqrt(k)), 0 for k = 0
  PackLayout layout;
  uint64_t ws_bytes = 0;
  // SUCCESS only: per class (kPackK, kPackL, kPackR) the kept labels (extent != 1) in the
  // pack's canonical order (first fastest) and X's modes fused over X's strides alone.
  std::vector<int32_t> order[3];
  std::vector<PackMode> fused[3];
  std::vector<int64_t> key;  // §6b.2 compatibility key
};

// imd_pack_layout's status and lowering of a descriptor: INVALID_VALUE (argument, label-list
// and §5 rows 1 / 5 checks on X alone, layout overflow), then NOT_SUPPORTED (§6b.3: K empty
// after lowering; RN, k >= 2 and kappa >= T), else SUCCESS. AUTO orders each class by
// (|merged stride|, first position in X's labels); GIVEN takes the lists' order. Extent-1
// labels are dropped (§5 step 4); a repeated label of X is a diagonal (strides summed).
IMD_HOST_INTERNAL imd_status_t pack_lower(const imd_pack_desc_t& d, PackLowered* out);

// The §6b.2 key: FORMAT (the pack's own, or for a plan's slot key the plan's), N, rounding
// (IMD_RN 0 / IMD_TRUNC 1), b, r, k, then for the classes L, R, K (in that order) the number
// of fused modes followed by (extent, stride) per mode.
IMD_HOST_INTERNAL std::vector<int64_t> pack_key(int64_t format, int N, int32_t rounding,
                                                int64_t b, int64_t r,
                                                int64_t k, const std::vector<PackMode>& L,
                                                const std::vector<PackMode>& R,
                                                const std::vector<PackMode>& K);
// X's modes of one class fused over X's strides alone: consecutive modes merge iff
// s' = s e (§6b.2 key part 4).
IMD_HOST_INTERNAL void pack_fuse_append(std::vector<PackMode>& f, int64_t extent, int64_t stride);
// FNV-1a (64-bit) over the key words as little-endian int64 bytes (key_digest; diagnostics).
IMD_HOST_INTERNAL uint64_t key_digest(const std::vector<int64_t>& key);
// 0 if the keys are equal, else the IMD_PACK_MISMATCH_* reason of the first differing word
// after word 0 (by the slot key's structure): MODULI, ROUNDING, EXTENTS (b, r, k, mode counts
// and extents; one key a prefix of the other) or STRIDES; FORMAT only when word 0 is the one
// difference (v1.13 §4b.5: the format is compared last; oracle key_mismatch). An unknown
// format word is analyze_packed's earlier check.
IMD_HOST_INTERNAL int32_t key_mismatch(const std::vector<int64_t>& slot,
                                       const std::vector<int64_t>& pack);

// A pack as a plan sees it (imd_pack_t's copy, or a host query's descriptor): the labels of
// the pack's X by mode position (they map onto the plan's X by position) and its lowering.
struct PackSource {
  int32_t rank = 0;
  const int32_t* label = nullptr;
  const PackLowered* pl = nullptr;
};

}  // namespace imd::detail
