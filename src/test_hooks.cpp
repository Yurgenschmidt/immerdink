// Test-only entry points (not in the public header; used by tests/test_contract.py through
// ctypes). They never run unless a test calls them, and production behaviour is unchanged:
//
//   imd_test_set_limits(k_chunk, max_batch, max_n_tiles)
//       Override K4's launch limits (plan.hpp LaunchLimits) so the K-chunk loop, the batch
//       split and the n-hat (gridDim.y) split can be exercised on small shapes. 0 restores
//       a production value. k_chunk must be a multiple of 16 in [16, K_c'] (any such chunk
//       keeps every accumulator within the §3.5 bound); the others in [1, 65535].
//       Returns 0, or 1 (nothing changed) for an invalid argument.
//
//   imd_test_lowering(A, B, C, out, cap, count)
//       Dump the §5 lowering (index_map.hpp) as int64s, for comparison with
//       spec/oracle.py lower(): swapped, m, n, k, b; then per group I, J, K, L:
//       n_labels, labels..., n_modes, (extent, s_A, s_B, s_C) per fused mode; then the pack
//       pivots of the GEMM's A and B: along_rows, radix. Returns the lowering status;
//       *count = number of int64s (written only if <= cap).
#include <cstdint>
#include <vector>

#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "index_map.hpp"
#include "plan.hpp"

extern "C" {

IMD_API int imd_test_set_limits(int64_t k_chunk, int64_t max_batch, int64_t max_n_tiles) {
  const imd::detail::LaunchLimits def = imd::detail::default_launch_limits();
  if (k_chunk == 0) k_chunk = def.k_chunk;
  if (max_batch == 0) max_batch = def.max_batch;
  if (max_n_tiles == 0) max_n_tiles = def.max_n_tiles;
  if (k_chunk < 16 || k_chunk % 16 != 0 || k_chunk > imd::constants::kKcPrime) return 1;
  if (max_batch < 1 || max_batch > def.max_batch) return 1;
  if (max_n_tiles < 1 || max_n_tiles > def.max_n_tiles) return 1;
  imd::detail::launch_limits() = {k_chunk, max_batch, max_n_tiles};
  return 0;
}

IMD_API imd_status_t imd_test_lowering(const imd_tensor_t* A, const imd_tensor_t* B,
                                       const imd_tensor_t* C, int64_t* out, int64_t cap,
                                       int64_t* count) {
  if (A == nullptr || B == nullptr || C == nullptr || count == nullptr)
    return IMD_STATUS_INVALID_VALUE;
  try {
    imd::detail::Lowered lw;
    if (const imd_status_t st = imd::detail::lower(*A, *B, *C, &lw); st != IMD_STATUS_SUCCESS)
      return st;
    std::vector<int64_t> v = {lw.swapped ? 1 : 0, lw.m, lw.n, lw.k, lw.b};
    for (int g = 0; g < 4; ++g) {
      v.push_back(static_cast<int64_t>(lw.order[g].size()));
      for (const int32_t l : lw.order[g]) v.push_back(l);
      v.push_back(static_cast<int64_t>(lw.fused[g].size()));
      for (const auto& f : lw.fused[g]) {
        v.push_back(f.extent);
        for (int x = 0; x < 3; ++x) v.push_back(f.stride[x]);
      }
    }
    for (int x = 0; x < 2; ++x) {
      const imd::detail::PackPivot p = imd::detail::pack_pivot(lw, x);
      v.push_back(p.along_rows ? 1 : 0);
      v.push_back(p.radix);
    }
    *count = static_cast<int64_t>(v.size());
    if (out != nullptr && *count <= cap)
      for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
    return IMD_STATUS_SUCCESS;
  } catch (...) {
    return IMD_STATUS_ALLOC_FAILED;
  }
}

}  // extern "C"
