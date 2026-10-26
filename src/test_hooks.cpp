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
//       mode -1: plans created afterwards take the v1.4 fused path KF and the v1.6 fused
//       large path KL when the plan-time rules (analysis.hpp fused_path, fused_large) say so
//       (production); 0: never (K0, K1 and the packs of group 1 run as separate launches);
//       2: every plan with the emulated GEMM takes KL for both operands (never KF); 3: as 2,
//       with KL tiles of 64 rows (so rows of k_hat > 64 span several chunks: the register,
//       shared-memory and re-read chunks of KL all run on small shapes). Results must not
//       change. Returns 0, or 1 for another mode.
//
//   imd_test_set_k4_backend(backend) / imd_test_get_k4_backend() / imd_test_plan_k4_backend(plan) / imd_test_k4_wgmma_built() /
//   imd_test_k4_wgmma_available(handle) / imd_test_k4_wgmma_counts(reset, out[2]) / imd_test_k4_wgmma_dry_run(...)
//       issue 22: the Hopper (wgmma) K4 backend: force it or the mma.sync K4 for plans created afterwards, read back what a plan
//       tries first, whether the build and the device have it, the launches it served / declined, and a host-only dry run of one launch.
//
//   imd_test_set_k4_config(config)
//       config -1: plans created afterwards run the K4 configuration of the plan-time rule
//       (or k4_tune's; production); 0 .. IMD_K4_CONFIG_COUNT - 1: that configuration
//       (DESIGN v1.5 §4). Results must not change. Returns 0, or 1 for another value.
//
//   imd_test_set_k5_crt(variant)
//   imd_test_set_seq(bits)
//       issue 21: every later imd_contract runs the instruction-sequence set `bits` (1 E1 emitter, 2 Q1 quantize, 4 F1 finish;
//       host/seq_config.hpp), -1 the plan's (device class) set. Results must not change. Returns 0, or 1 for another value
//       (also for the Q1 bit without the E1 bit).
//   imd_test_get_seq()
//       the forced set (-1: none).
//   imd_test_plan_seq(plan)
//       the set the plan's launches run now: the forced one, else the plan's device-class default; -1 for a null plan.
//   imd_test_set_k5_one_elem(per_sm)
//       K5 (j-bar fastest) takes one element per thread while b m n <= per_sm x SMs, else
//       4 per thread with warp-transposed stores. per_sm -1 restores the production value;
//       0 always uses 4 per thread and also disables KF's 4-values-per-thread variant (no
//       small-problem launch variant anywhere). Applies to later imd_contract calls; results
//       must not change. Returns 0, or 1 for a negative value other than -1.
//
//   imd_test_k4_align_check(mode)
//       v1.13 §4b.5: mode 1 enables the check that every K4 operand pointer (call_of in
//       gemm_k4.cu) is = 0 mod the plan's K alignment A, for plans whose K-chunk is a multiple
//       of A, and resets its counts; 0 disables it; -1 only queries; 2 returns the launches
//       checked since the last reset (so a suite can show the check ran); 3 is mode 1 with a
//       mutation: the check sees the A operand pointer + 8 bytes (the launch is unchanged), so
//       every checked launch must count as a violation. Returns the violations counted since
//       the last reset, mode 2 the checked launches, -1 for an unknown mode (debug builds: on
//       by default, and a violation also fails the call, checked before anything is queued).
//       The K alignment itself is forced by libimmerdink_host's
//       imd_test_set_k_align (src/host/k_hat.cpp).
//
//   imd_test_crt_elem(N, variant, e, count, W, out)
//   imd_test_k5_launch_counts(reset, out)
//       Issue 21 (docs/NOTES-21.md): the first runs the real CRT device functions of contract.cu
//       (variant 0 integer running CRT, 1 binary64-limb CRT, 2..9 its compile-time mutations) on
//       `count` device residue vectors and writes x in [0, P) before crt_finish; the second
//       copies (and optionally zeroes) the counts of K5 launches per [limb CRT?][variant].
//       tests/test_crt_f64_unit.py, tests/test_crt_f64_gpu.py.
//
//   imd_test_set_pad_mutation(operand)
//       v1.13 §4b (hygiene): operand 0 / 1 makes the K2 / K3 pack of the post-swap GEMM
//       operand A / B store k-bar only below ceil(k)_16 (Pipeline neg_a64_pad_old_k3), so an
//       A = 64 plan leaves [ceil(k)_16, k_hat) of that operand unwritten while the other
//       operand's zeros keep the result bit-exact: the bug only initcheck on an unpoisoned
//       workspace can see (the initcheck mutation test). -1 restores production. Applies to
//       later imd_contract calls of plans packing that operand with K2 / K3 (imd_test_set_fused
//       (0), or groups after the first). Returns 0, or 1 for another value.
//
//   imd_test_pack_mismatch(A, B, C, options, packA, packB, reason)
//       v1.13 §4b.5 (hygiene): the status of plan creation's analysis of (A, B, C, options)
//       with these pack handles (NULL: in-line), as imd_plan_create_packed computes it, and
//       *reason = its IMD_PACK_MISMATCH_* (0 unless PACK_MISMATCH): the reason a
//       handle-based mismatch has, which imd_lowering_packed (descriptors, resolved with the
//       plan's own hook snapshot) cannot show. No GPU work, no allocation kept.
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
//       launches per imd_contract with alpha != 0; then (v1.5) its K4 configuration, the
//       configuration of its K4 parameters (both -1 without K4) and its SM count; then (v1.6)
//       its fused_large bit mask (the operands KL scales and packs); then (v1.7) its packed
//       slots (bit 0 / 1: the caller's A / B) and the pack bytes of the post-swap A and B
//       (0 for an in-line operand). The host payload includes the copied slot keys. Returns 0,
//       or 1 for a null argument; *count as above.
//
//   imd_test_kernel_occupancy_w(kernel, w, out)
//       v1.16 §3.4.1: the measured occupancy row (CTAs per SM, registers, static shared
//       memory bytes, threads) of kernel IMD_PERF_KERNEL_* (K4 excepted) for pack-emitter word
//       count w = 2, 3 (pack kernels KF16, KF4, KL8, KL4, PACK only), w = 0: the row the
//       performance model's occupancy query reports. Returns 0, or 1 for an invalid argument.
#include <cstdint>
#include <vector>

#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink.h"
#include "host/analysis.hpp"
#include "host/index_map.hpp"
#include "host/k4_wgmma.hpp"
#include "host/seq_config.hpp"
#include "k4/hopper/k4_wgmma_ops.hpp"
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
  if (mode < -1 || mode == 1 || mode > 3) return 1;
  imd::detail::fused_override() = mode;
  return 0;
}

IMD_API int imd_test_set_k4_config(int config) {
  if (config < -1 || config >= IMD_K4_CONFIG_COUNT) return 1;
  imd::detail::k4_config_override() = config;
  return 0;
}

// Issue 22 (docs/NOTES-22.md "Library integration (design)"): the K4 backend REQUEST of plans created afterwards (-1 the device class's rule, 0 mma.sync,
// 1 wgmma: the plan is then "wgmma, forced" = backend 2: every legal launch of that plan, the profitability floor lifted, for the plan's life; a
// device without the backend runs mma.sync whatever is asked). The request is read at plan creation only; resetting it later does not change plans.
IMD_API int imd_test_set_k4_backend(int backend) {
  if (!imd::detail::k4_backend_valid(backend)) return 1;
  imd::detail::k4_backend_override() = backend;
  return 0;
}
IMD_API int imd_test_get_k4_backend(void) { return imd::detail::k4_backend_override().load(); }
// The backend a plan tries first (0 mma.sync, 1 wgmma with the profitability floor, 2 wgmma forced; -1 for a null plan or a plan without K4).
IMD_API int imd_test_plan_k4_backend(imd_plan_t plan) {
  const imd::detail::ExecPlan* e = imd::detail::exec_of(plan);
  return e == nullptr || !e->k4 ? -1 : imd::detail::k4_backend_of(*e->k4);
}
// 1 if this build has the Hopper backend (IMMERDINK_K4_WGMMA), 0 if not; and whether it can run on the current device
// (device class, compute capability, probe) as imd_plan_create would see it.
IMD_API int imd_test_k4_wgmma_built(void) { return imd::detail::k4_wgmma_ops(0) != nullptr ? 1 : 0; }
IMD_API int imd_test_k4_wgmma_available(imd_handle_t handle) {
  imd_device_info_t info{};
  info.struct_size = sizeof info;
  if (imd_device_info(handle, &info) != IMD_STATUS_SUCCESS) return -1;
  return imd::detail::k4_wgmma_available(info.table_id, info.device.cc_major, info.device.cc_minor) ? 1 : 0;
}
// Launches the Hopper backend served / declined since the last reset, int64[2]; reset != 0 zeroes them.
IMD_API int imd_test_k4_wgmma_counts(int reset, int64_t* out) {
  if (out == nullptr) return 1;
  imd::detail::k4_wgmma_counts(reset != 0, out);
  return 0;
}
// The host half of a Hopper launch of a synthetic GEMM, on any device of a build with the backend (gemm_k4.cu k4_wgmma_dry_run). out[7] =
// variant (-1 not served), swizzle, raster, can_implement, persistent grid x / y / z. Returns 0, 1 if the backend is absent or can_implement refused.
IMD_API int imd_test_k4_wgmma_dry_run(int64_t m, int64_t nc, int64_t kc, int64_t zc, int64_t k_hat, int64_t n_hat, int64_t batch_a,
                                      int64_t batch_b, int sms, int clusters, int add, int force, int32_t* out) {
  if (out == nullptr) return 1;
  return imd::detail::k4_wgmma_dry_run(m, nc, kc, zc, k_hat, n_hat, batch_a, batch_b, sms, clusters, add != 0, force != 0, out) ? 0 : 1;
}

IMD_API int imd_test_set_k5_one_elem(int64_t per_sm) {
  if (per_sm < -1) return 1;
  static const int64_t production = imd::detail::k5_one_elem_per_sm();
  imd::detail::k5_one_elem_per_sm() = per_sm == -1 ? production : per_sm;
  return 0;
}

// A5: K5 (j-bar fastest) with 1, 2 or 4 elements per thread; 0 restores the launch rule.
IMD_API int imd_test_set_k5_elems(int elems) {
  if (elems != 0 && elems != 1 && elems != 2 && elems != 4) return 1;
  imd::detail::k5_elems_override() = elems;
  return 0;
}

// Issue 21: the instruction-sequence set of later calls (-1: the device class's).
IMD_API int imd_test_set_seq(int bits) {
  if (!imd::detail::seq_valid(bits)) return 1;
  imd::detail::seq_override() = bits;
  return 0;
}

IMD_API int imd_test_get_seq(void) { return imd::detail::seq_override().load(); }
IMD_API int imd_test_plan_seq(imd_plan_t plan) {
  const imd::detail::ExecPlan* e = imd::detail::exec_of(plan);
  return e == nullptr ? -1 : imd::detail::seq_effective(e->seq, imd::detail::seq_override().load());
}

// Issue 21 device-level unit test of the binary64-limb CRT: see plan.hpp test_crt_elem. W: device bytes [N][count], out: device u32 [count][5].
// variant 0 integer running CRT, 1 limb CRT, 2..9 its mutations. Returns 0, or 1 for an invalid argument or a launch error. The caller synchronizes.
IMD_API int imd_test_crt_elem(int N, int variant, int e, int64_t count, const uint8_t* W, uint32_t* out) {
  if (count > 0 && (W == nullptr || out == nullptr)) return 1;
  return imd::detail::test_crt_elem(N, variant, e, count, W, out) == cudaSuccess ? 0 : 1;
}

// K5 launches since the last reset, int64[8]: [limb CRT?][variant 0 / 1 / 2 = 1 / 2 / 4 elements per thread, 3 = batch fastest]; reset != 0 zeroes them.
IMD_API int imd_test_k5_launch_counts(int reset, int64_t* out) {
  if (out == nullptr) return 1;
  imd::detail::test_k5_launch_counts(reset != 0, out);
  return 0;
}

// Issue 21: K5's CRT with the sums in binary64 limbs (1) or the integer running CRT (0, the default).
IMD_API int imd_test_set_k5_crt(int variant) {
  if (variant != 0 && variant != 1) return 1;
  imd::detail::k5_crt_override() = variant;
  return 0;
}

IMD_API int imd_test_set_pad_mutation(int operand) {
  if (operand < -1 || operand > 1) return 1;
  imd::detail::pad_mutation().store(operand, std::memory_order_relaxed);
  return 0;
}

IMD_API imd_status_t imd_test_pack_mismatch(const imd_tensor_t* A, const imd_tensor_t* B,
                                            const imd_tensor_t* C, const imd_options_t* options,
                                            imd_pack_t packA, imd_pack_t packB,
                                            int32_t* reason) {
  if (reason == nullptr) return IMD_STATUS_INVALID_VALUE;
  *reason = 0;
  try {
    imd::detail::PackSource ps[2];
    const imd::detail::PackSource* src[2] = {nullptr, nullptr};
    const imd_pack_t pk[2] = {packA, packB};
    for (int x = 0; x < 2; ++x) {
      if (pk[x] == nullptr) continue;
      ps[x] = imd::detail::pack_source_of(pk[x]);
      src[x] = &ps[x];
    }
    imd::detail::Analysis an;
    const imd_status_t st = imd::detail::analyze_packed(A, B, C, options, src, &an);
    *reason = an.packed.mismatch;
    return st;
  } catch (...) {
    return IMD_STATUS_ALLOC_FAILED;
  }
}

IMD_API int64_t imd_test_k4_align_check(int mode) {
  if (mode < -1 || mode > 3) return -1;
  return imd::detail::k4_align_check(mode);
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
  host += (plan->slot_key[0].size() + plan->slot_key[1].size()) * sizeof(int64_t);  // v1.7
  v.push_back(static_cast<int64_t>(host));
  v.push_back(e.fused ? 1 : 0);
  v.push_back(imd::detail::plan_launches(e, e.fused, e.fused_large, e.vec_resident, k4.launches));
  // v1.5: the K4 configuration the plan runs (-1 without K4), its K4Plan's, and the SMs the
  // rule was evaluated for.
  v.push_back(e.k4 ? e.k4_config : -1);
  v.push_back(e.k4 ? k4.config : -1);
  v.push_back(e.sms);
  // v1.6: the operands of the fused large launch KL (bit 0: A, bit 1: B).
  v.push_back(e.fused_large);
  // v1.7: the packed operands (bit 0 / 1: the caller's A / B) and their pack bytes.
  v.push_back((plan->slot_key[0].empty() ? 0 : 1) | (plan->slot_key[1].empty() ? 0 : 2));
  v.push_back(static_cast<int64_t>(e.pack_bytes[0]));
  v.push_back(static_cast<int64_t>(e.pack_bytes[1]));
  *count = static_cast<int64_t>(v.size());
  if (out != nullptr && *count <= cap)
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
  return 0;
}

// v1.16 §3.4.1: the occupancy row of one W instantiation of a pack kernel (w = 0: the reported row).
IMD_API int imd_test_kernel_occupancy_w(int32_t kernel, int32_t w, int32_t* out) {
  if (out == nullptr || kernel < 0 || kernel >= IMD_PERF_KERNEL_COUNT ||
      (kernel >= IMD_PERF_KERNEL_K4 && kernel < IMD_PERF_KERNEL_K5))
    return 1;
  return imd::detail::kernel_occupancy_w(kernel, w, out) == cudaSuccess ? 0 : 1;
}

}  // extern "C"
