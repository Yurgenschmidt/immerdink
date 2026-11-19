#pragma once
// IMD_HWGEMM's plan-time rules shared by the device library (src/hwgemm.cu) and the GPU-free
// host library (the performance model, DESIGN §6i.5): one definition, so the prediction and the
// run pick the same tile and the same pack kernel. Header-only, no CUDA.

#include <cstdint>

namespace imd::detail {

// The GEMM tiles (src/hwgemm/): 128x128 and 64x128 need the Sm90 kernel family; Sm80 has 64x64.
enum HwTile : int32_t { kHw128x128 = 0, kHw64x128 = 1, kHw64x64 = 2, kHwTileCount = 3 };

// The provisional tile rule (NOTES-HWGEMM §4.3 gate, 6 shapes on H100; the per-shape rule is
// a deliverable of the tuning step): 64x64 when a 64-row tile already covers m or n, else
// 64x128 when 128x128 tiles would fill fewer than half the SMs, else 128x128. sm90 = false
// (the Sm80 kernel family: sm_80 / sm_89, or a binary without usable Sm90 kernels): always
// 64x64. m, n: the GEMM's own (stage 1 with a column-major C: after the C^T swap).
constexpr int32_t hw_tile_rule(int64_t m, int64_t n, int64_t b, bool sm90, int sms) {
  if (!sm90) return kHw64x64;
  if (m <= 64 || n <= 64) return kHw64x64;
  const int64_t ctas128 = (m + 127) / 128 * ((n + 127) / 128) * b;
  return 2 * ctas128 < sms ? kHw64x128 : kHw128x128;
}

// The FP64 pack's kernel (hwgemm_pack): the tiled smem-transpose pack when the source's fastest
// index runs along the packed rows (pack_pivot along_rows, radix) and the radix divides b·rows;
// else the element-wise pack.
constexpr bool hw_pack_tiled(bool along_rows, int64_t radix, int64_t b, int64_t rows) {
  return along_rows && radix >= 1 && (b * rows) % radix == 0;
}

}  // namespace imd::detail
