// Packed operands (DESIGN v1.7 §6b, ROADMAP D1): the pack descriptor's lowering and
// statuses (§6b.3, §6b.5), the format-1 byte layout (§6b.4), the compatibility key (§6b.2)
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

constexpr int64_t kPackFormat = IMD_PACK_FORMAT;
// The pack's label classes (§6b.1): K contracted, L batch, R rows.
enum : int { kPackK = 0, kPackL = 1, kPackR = 2 };
// Regions of the packed buffer (§6b.4), in layout order.
enum : int { kPackSig = 0, kPackFlag = 1, kPackRes = 2 };

// §6b.4 format 1: sig (2 b r), flag (b r), res (N b r k_hat + (ceil(r)_16 - r) k_hat) at
// 256-aligned offsets; total = the sum of the aligned sizes; all 0 when b r k = 0. nullopt
// when a size or the total exceeds uint64 (INVALID_VALUE).
struct PackLayout {
  uint64_t offset[3] = {0, 0, 0}, bytes[3] = {0, 0, 0};
  uint64_t total = 0;
};
IMD_HOST_INTERNAL std::optional<PackLayout> pack_layout(int64_t b, int64_t r, int64_t k, int N);

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

// The §6b.2 key: FORMAT, N, rounding (IMD_RN 0 / IMD_TRUNC 1), b, r, k, then for the classes
// L, R, K (in that order) the number of fused modes followed by (extent, stride) per mode.
IMD_HOST_INTERNAL std::vector<int64_t> pack_key(int N, int32_t rounding, int64_t b, int64_t r,
                                                int64_t k, const std::vector<PackMode>& L,
                                                const std::vector<PackMode>& R,
                                                const std::vector<PackMode>& K);
// X's modes of one class fused over X's strides alone: consecutive modes merge iff
// s' = s e (§6b.2 key part 4).
IMD_HOST_INTERNAL void pack_fuse_append(std::vector<PackMode>& f, int64_t extent, int64_t stride);
// FNV-1a (64-bit) over the key words as little-endian int64 bytes (key_digest; diagnostics).
IMD_HOST_INTERNAL uint64_t key_digest(const std::vector<int64_t>& key);
// 0 if the keys are equal, else the IMD_PACK_MISMATCH_* reason of the first differing word
// (by the slot key's structure): FORMAT, MODULI, ROUNDING, EXTENTS (b, r, k, mode counts and
// extents) or STRIDES.
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
