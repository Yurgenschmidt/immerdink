// The K-alignment resolver and its test hook (DESIGN v1.13 §4b.5; k_hat.hpp).
#include "k_hat.hpp"

#include <atomic>

namespace imd::detail {
namespace {

// 0 = AUTO; 16 or 64 = forced by imd_test_set_k_align. One variable in libimmerdink_host,
// read only through k_align_hook().
std::atomic<int32_t> g_k_align_hook{0};

}  // namespace

int32_t k_align_hook() { return g_k_align_hook.load(std::memory_order_relaxed); }

KAlignSel k_align_sel() { return KAlignSel{k_align_hook()}; }

int32_t k_align_of(int64_t k) { return k_align_sel().of(k); }

}  // namespace imd::detail

// Test-only export of libimmerdink_host (not in a public header; ctypes and the C tests bind
// it like libimmerdink's imd_test_* hooks). a = 0 restores AUTO; 16 or 64 forces A for every
// sizing query, lowering, plan and pack created afterwards in the process (libimmerdink plans
// follow: they take A from the host library's ExecPlan). Anything else returns 1 and changes
// nothing. Setting it between imd_pack_create and plan creation is the only way to make a pack
// and a plan disagree (PACK_MISMATCH FORMAT).
extern "C" IMD_HOST_INTERNAL int imd_test_set_k_align(int32_t a) {
  if (a != 0 && a != imd::detail::kKAlign16 && a != imd::detail::kKAlign64) return 1;
  imd::detail::g_k_align_hook.store(a, std::memory_order_relaxed);
  return 0;
}
