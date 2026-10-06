// The K extent and row stride of every residue row (DESIGN §4, v1.13 §4b): k_hat = ceil(k)_A.
// The single source of this rounding in libimmerdink_host (workspace sizing, the plan's GEMM
// view, the pack layout and the pack lowering); spec/params.py k_hat_of mirrors it.
//
// A, the plan's (or pack's) K alignment, is 16 or 64. k_align_of(k) is the ONE resolver
// (§4b.5 "One resolver, once per API call"): the test hook imd_test_set_k_align if set, else
// AUTO(k) = 64 iff k > kKPad64Min (96). It is called once per API call, in analyze (every
// sizing query, lowering and plan) and in pack_lower (every pack); from then on A is passed
// explicitly (WorkspaceShape::k_align, pack_layout's argument, ExecPlan::k_align,
// PackLowered::k_align). libimmerdink never calls it: its plans take A and k_hat from the
// host library's ExecPlan. spec/params.py k_align_of mirrors it.
#pragma once

#include <cstdint>

#include "checked.hpp"
#include "export.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::detail {

inline constexpr int32_t kKAlign16 = imd::constants::kKAlign;     // 16
inline constexpr int32_t kKAlign64 = imd::constants::kKAlignPad;  // 64

// AUTO(k) (§4b.3): a pure function of the logical (lowered) contraction length k.
inline constexpr int32_t k_align_auto(int64_t k) {
  return k > imd::constants::kKPad64Min ? kKAlign64 : kKAlign16;
}

// The test hook's value: 0 (AUTO), 16 or 64. Its state is one out-of-line std::atomic in
// k_hat.cpp (both libraries build with hidden visibility: an inline variable or a
// function-local static in this header would exist once per .so, and libimmerdink would never
// see the hook). The one test-only exception to §6a's "pure" (DESIGN §4b.5).
IMD_HOST_INTERNAL int32_t k_align_hook();

// One snapshot of the hook (DESIGN §4b.5, hygiene): taken ONCE at the entry of an API call
// (k_align_sel) and passed to every analysis of that call (analyze, pack_lower, analyze_with,
// the perf model's per-candidate analyses), so all of the call's A come from one hook value.
struct KAlignSel {
  int32_t forced = 0;  // the hook's value when taken: 0 (AUTO), 16 or 64
  constexpr int32_t of(int64_t k) const { return forced != 0 ? forced : k_align_auto(k); }
};

// Reads the hook (once): the selection of this API call.
IMD_HOST_INTERNAL KAlignSel k_align_sel();

// The resolver of an entry point with a single analysis: k_align_sel().of(k).
IMD_HOST_INTERNAL int32_t k_align_of(int64_t k);

// The pack format word of alignment a (§6b.4: 1 = A16, 2 = A64), and back (0: unknown format).
inline constexpr int64_t pack_format_of(int32_t a) {
  return a == kKAlign64 ? imd::constants::kPackFormatA64 : imd::constants::kPackFormatA16;
}
inline constexpr int32_t pack_format_align(int64_t format) {
  return format == imd::constants::kPackFormatA16   ? kKAlign16
         : format == imd::constants::kPackFormatA64 ? kKAlign64
                                                    : 0;
}

// ceil(k)_a; false when it does not fit uint64.
inline bool k_hat_of(uint64_t k, uint64_t a, uint64_t* k_hat) { return align_u64(k, a, k_hat); }

// ceil(k)_a for 0 <= k; false when it does not fit int64.
inline bool k_hat_of(int64_t k, int64_t a, int64_t* k_hat) {
  uint64_t u = 0;
  if (k < 0 || !align_u64(static_cast<uint64_t>(k), static_cast<uint64_t>(a), &u) ||
      u > static_cast<uint64_t>(INT64_MAX))
    return false;
  *k_hat = static_cast<int64_t>(u);
  return true;
}

}  // namespace imd::detail
