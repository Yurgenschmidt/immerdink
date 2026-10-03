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
//   imd_test_set_fused(mode)
//       mode -1: plans created afterwards take the v1.4 fused path KF when the plan-time
//       rule (analysis.hpp fused_path) says so (production); 0: never (K0, K1 and the packs
//       of group 1 run as separate launches). Results must not change. Returns 0, or 1 for
//       another mode.
//
//   imd_test_set_k5_one_elem(per_sm)
//       K5 (j-bar fastest) takes one element per thread while b m n <= per_sm x SMs, else
//       4 per thread with warp-transposed stores. per_sm -1 restores the production value;
//       0 always uses 4 per thread and also disables KF's 4-values-per-thread variant (no
//       small-problem launch variant anywhere). Applies to later imd_contract calls; results
//       must not change. Returns 0, or 1 for a negative value other than -1.
//
//   imd_test_plan_vec(plan, dev, count)
//       The plan's resident scatter vectors (IMD_VEC_RESIDENT, DESIGN v1.2): *dev = their
//       device address (NULL for a generate plan), *count = their int64 entries (the vec
//       region's length in either mode). Returns 0, or 1 for a null argument.
//
//   imd_test_lowering(A, B, C, out, cap, count)
//       Dump the §5 lowering (index_map.hpp) as int64s, for comparison with
//       spec/oracle.py lower(): swapped, m, n, k, b; then per group I, J, K, L:
//       n_labels, labels..., n_modes, (extent, s_A, s_B, s_C) per fused mode; then the pack
//       pivots of the GEMM's A and B: along_rows, radix. Returns the lowering status;
//       *count = number of int64s (written only if <= cap).
//
//   imd_test_plan_sizes(plan, out, cap, count)
//       What a real plan actually holds and uses, for comparison with libimmerdink_host's
//       queries (DESIGN §6a; tests/test_host.py): path, swapped, m, n, k, b, n_hat, k_hat,
//       N, g; workspace total; 8 x (offset, bytes) of its §4a layout; the plan device bytes
//       it allocated (0 for a generate plan); the K4 parameter sets (store, add) and the
//       K4 launches of one call; and its host payload, counted from the live containers:
//       sizeof(imd_plan_st) + descriptor copies + lowering tables + K0 descriptor + K4Plan
//       with its map entries; then (v1.4) whether it takes the fused path and its kernel
//       launches per imd_contract with alpha != 0. Returns 0, or 1 for a null argument;
//       *count as above.
#include <cstdint>
#include <vector>

#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "host/analysis.hpp"
#include "host/index_map.hpp"
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

IMD_API int imd_test_set_fused(int mode) {
  if (mode != -1 && mode != 0) return 1;
  imd::detail::fused_override() = mode;
  return 0;
}

IMD_API int imd_test_set_k5_one_elem(int64_t per_sm) {
  if (per_sm < -1) return 1;
  static const int64_t production = imd::detail::k5_one_elem_per_sm();
  imd::detail::k5_one_elem_per_sm() = per_sm == -1 ? production : per_sm;
  return 0;
}

IMD_API int imd_test_plan_vec(imd_plan_t plan, const int64_t** dev, int64_t* count) {
  const imd::detail::ExecPlan* e = imd::detail::exec_of(plan);
  if (e == nullptr || dev == nullptr || count == nullptr) return 1;
  *dev = e->vec_resident ? e->vec_dev : nullptr;
  *count = e->vec_count;
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

IMD_API int imd_test_plan_sizes(imd_plan_t plan, int64_t* out, int64_t cap, int64_t* count) {
  if (plan == nullptr || count == nullptr) return 1;
  const imd::detail::ExecPlan& e = plan->exec;
  std::vector<int64_t> v = {static_cast<int64_t>(e.path), e.swapped ? 1 : 0, e.m, e.n, e.k, e.b,
                            e.n_hat, e.k_hat, e.N, e.g, static_cast<int64_t>(e.layout.total)};
  for (size_t r = 0; r < e.layout.offset.size(); ++r) {
    v.push_back(static_cast<int64_t>(e.layout.offset[r]));
    v.push_back(static_cast<int64_t>(e.layout.bytes[r]));
  }
  v.push_back(e.vec_resident ? static_cast<int64_t>(e.device_bytes) : 0);
  imd::detail::K4Footprint k4;
  if (e.k4) k4 = imd::detail::k4_footprint(*e.k4, e.N, e.g);
  v.push_back(k4.store_shapes);
  v.push_back(k4.add_shapes);
  v.push_back(k4.launches);
  // The host payload, from the live plan (each container's size() times its element size).
  uint64_t host = sizeof(imd_plan_st);
  for (const imd::detail::TensorCopy* t : {&plan->A, &plan->B, &plan->C})
    host += t->extent.size() * sizeof(int64_t) + t->stride.size() * sizeof(int64_t) +
            t->label.size() * sizeof(int32_t);
  const imd::detail::Lowered& lw = plan->lowered;
  host += lw.labels.size() * sizeof(imd::detail::LabelInfo);
  for (int g = 0; g < 4; ++g)
    host += lw.order[g].size() * sizeof(int32_t) + lw.fused[g].size() * sizeof(imd::detail::FusedMode);
  host += e.vec_desc.modes.size() * sizeof(imd::detail::VecMode);
  host += k4.bytes;
  v.push_back(static_cast<int64_t>(host));
  v.push_back(e.fused ? 1 : 0);
  v.push_back(imd::detail::plan_launches(e, e.fused, e.vec_resident, k4.launches));
  *count = static_cast<int64_t>(v.size());
  if (out != nullptr && *count <= cap)
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
  return 0;
}

}  // extern "C"
