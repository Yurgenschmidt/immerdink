// DESIGN v1.13 §4b stage 2 inside libimmerdink_host (CPU only, links the host library alone):
// the K-alignment resolver and its test hook (imd_test_set_k_align), A resolved once per API
// call and passed explicitly (analyze -> ExecPlan::k_align, pack_lower -> PackLowered), the
// pack format word, and the plan-time PACK_MISMATCH reason order of analyze_packed (§4b.5,
// §6b.5; the same order as spec/oracle.py lower_packed and PackedOperand Reason):
//   an unknown format word first (before MODULI), then per pack MODULI, ROUNDING, CLASSES,
//   then ORDER, then the other key words of BOTH packs (EXTENTS, STRIDES), then a format 1 vs
//   2 difference of either (FORMAT), then the k_hat drift check (FORMAT).
// A pack and a plan of different A are made by changing the hook between pack_lower and
// analyze_packed, as imd_pack_create / imd_plan_create_packed would see it; the unreachable
// branches (an unknown format word, a k_hat that disagrees with an equal key) by editing a
// copy of the pack's lowering.
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "analysis.hpp"
#include "immerdink/immerdink_host.h"
#include "k_hat.hpp"
#include "pack.hpp"
#include "workspace.hpp"

extern "C" int imd_test_set_k_align(int32_t a);

namespace {

using namespace imd::detail;

int failures = 0, checks = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    ++checks;                                                       \
    if (!(c)) {                                                     \
      ++failures;                                                   \
      std::printf("FAIL line %d: %s\n", __LINE__, #c);              \
    }                                                               \
  } while (0)

struct Tensor {
  std::vector<int64_t> extent, stride;
  std::vector<int32_t> label;
  imd_tensor_t view() const {
    return {static_cast<int32_t>(extent.size()), const_cast<int64_t*>(extent.data()),
            const_cast<int64_t*>(stride.data()), const_cast<int32_t*>(label.data())};
  }
};

constexpr int32_t I = 1, J = 2, K = 3;

// C[i, j] = A[i, k] B[k, j], m = 3, n = 5; bstride_kfast: B stored k fastest (else j fastest).
struct Gemm {
  Tensor A, B, C;
  imd_tensor_t a, b, c;
  Gemm(int64_t k, bool b_kfast = false)
      : A{{3, k}, {k, 1}, {I, K}},
        B{{k, 5}, {b_kfast ? 1 : 5, b_kfast ? k : 1}, {K, J}},
        C{{3, 5}, {5, 1}, {I, J}} {
    a = A.view(), b = B.view(), c = C.view();
  }
};

imd_options_t opts(int N = 9, int32_t mode = IMD_EMULATE) {
  imd_options_t o{};
  o.num_moduli = N;
  o.residue_group = 0;
  o.rounding = IMD_RN;
  o.mode = mode;
  o.vec_mode = IMD_VEC_GENERATE;
  o.k4_tune = IMD_K4_TUNE_OFF;
  return o;
}

// The pack of one operand X (AUTO order, K = {k}), lowered under the current hook.
struct Pack {
  imd_tensor_t x;
  int32_t contracted[1] = {K};
  imd_pack_desc_t d{};
  PackLowered pl;
  imd_status_t st = IMD_STATUS_INVALID_VALUE;
  explicit Pack(const imd_tensor_t& t, int N = 9) : x(t) {
    d.struct_size = sizeof d;
    d.num_moduli = N;
    d.rounding = IMD_RN;
    d.order = IMD_PACK_ORDER_AUTO;
    d.tensor = &x;
    d.num_contracted = 1;
    d.contracted = contracted;
    st = pack_lower(d, &pl);
  }
  PackSource src() const { return {x.rank, x.label, &pl}; }
};

// analyze_packed's status and reason for packs (nullptr: in-line) of g's operands.
struct Res {
  imd_status_t st;
  int32_t reason;
};
Res plan_with(const Gemm& g, const PackLowered* pa, const PackLowered* pb,
              const imd_options_t& o = opts()) {
  PackSource s[2];
  const PackSource* src[2] = {nullptr, nullptr};
  if (pa != nullptr) s[0] = {g.a.rank, g.a.label, pa}, src[0] = &s[0];
  if (pb != nullptr) s[1] = {g.b.rank, g.b.label, pb}, src[1] = &s[1];
  Analysis an;
  const imd_status_t st = analyze_packed(&g.a, &g.b, &g.c, &o, src, &an);
  return {st, an.packed.mismatch};
}

void test_resolver() {
  CHECK(imd_test_set_k_align(7) == 1 && imd_test_set_k_align(32) == 1 &&
        imd_test_set_k_align(-16) == 1);
  CHECK(k_align_hook() == 0);
  CHECK(k_align_of(96) == 16 && k_align_of(97) == 64 && k_align_of(0) == 16 &&
        k_align_of(1) == 16 && k_align_of(int64_t{1} << 40) == 64);
  for (const int32_t h : {16, 64}) {
    CHECK(imd_test_set_k_align(h) == 0 && k_align_hook() == h);
    CHECK(k_align_of(5) == h && k_align_of(200) == h);
  }
  CHECK(imd_test_set_k_align(0) == 0 && k_align_hook() == 0 && k_align_of(200) == 64);
  CHECK(pack_format_of(16) == IMD_PACK_FORMAT_A16 && pack_format_of(64) == IMD_PACK_FORMAT_A64);
  CHECK(pack_format_align(1) == 16 && pack_format_align(2) == 64 && pack_format_align(3) == 0 &&
        pack_format_align(0) == 0);
  CHECK(IMD_PACK_FORMAT == IMD_PACK_FORMAT_A64 && IMD_HOST_API_VERSION >= 9);
  // A is passed explicitly: no layout without it.
  CHECK(!pack_layout(1, 1, 5, 9, 0) && !pack_layout(1, 1, 5, 9, 32) && pack_layout(1, 1, 5, 9, 16));
  WorkspaceShape s{3, 5, 5, 1, 9, 9, false, false, 0, 0};
  CHECK(!workspace_layout(s));
  s.k_align = 64;
  CHECK(workspace_layout(s) && workspace_layout(s)->bytes[static_cast<int>(Region::RA)] == 9 * 3 * 64);
}

// plan_data: ExecPlan::k_align and k_hat per path, AUTO and forced.
void test_plan_paths() {
  for (const int32_t h : {0, 16, 64}) {
    imd_test_set_k_align(h);
    for (const int64_t k : {5, 96, 97, 200}) {
      const int32_t a = h != 0 ? h : (k > 96 ? 64 : 16);
      Gemm g(k);
      for (const int32_t mode : {IMD_EMULATE, IMD_DIRECT}) {
        const imd_options_t o = opts(9, mode);
        Analysis an;
        ExecPlan e;
        CHECK(analyze(&g.a, &g.b, &g.c, &o, &an) == IMD_STATUS_SUCCESS && an.k_align == a);
        CHECK(plan_data(an, &e) == IMD_STATUS_SUCCESS);
        const int32_t want = mode == IMD_EMULATE ? a : 16;  // DIRECT: no residue rows
        CHECK(e.k_align == want && e.k_hat == (k + want - 1) / want * want);
        if (mode == IMD_EMULATE)
          CHECK(an.layout.bytes[static_cast<int>(Region::RA)] == uint64_t(9) * 3 * e.k_hat);
      }
    }
    // K-empty (no contracted label: k = 1), SCALE (k = 0) and NOOP: 16 whatever the hook.
    Tensor A{{3}, {1}, {I}}, B{{5}, {1}, {J}}, C{{3, 5}, {5, 1}, {I, J}};
    Tensor A0{{3, 0}, {1, 3}, {I, K}}, B0{{0, 5}, {5, 1}, {K, J}};
    Tensor An{{0, 7}, {7, 1}, {I, K}}, Bn{{7, 5}, {5, 1}, {K, J}}, Cn{{0, 5}, {5, 1}, {I, J}};
    const imd_tensor_t cases[3][3] = {{A.view(), B.view(), C.view()},
                                      {A0.view(), B0.view(), C.view()},
                                      {An.view(), Bn.view(), Cn.view()}};
    for (const auto& t : cases) {
      const imd_options_t o = opts();
      Analysis an;
      ExecPlan e;
      CHECK(analyze(&t[0], &t[1], &t[2], &o, &an) == IMD_STATUS_SUCCESS);
      CHECK(plan_data(an, &e) == IMD_STATUS_SUCCESS && e.k_align == 16 &&
            e.k_hat == (e.k + 15) / 16 * 16);
    }
  }
  imd_test_set_k_align(0);
}

void test_pack_lower() {
  for (const int32_t h : {0, 16, 64})
    for (const int64_t k : {2, 96, 97, 1000}) {
      imd_test_set_k_align(h);
      Gemm g(k);
      Pack p(g.a);
      const int32_t a = h != 0 ? h : (k > 96 ? 64 : 16);
      CHECK(p.st == IMD_STATUS_SUCCESS && p.pl.k_align == a && p.pl.format == pack_format_of(a) &&
            p.pl.key[0] == p.pl.format && p.pl.k_hat == (k + a - 1) / a * a);
      CHECK(p.pl.layout.bytes[kPackRes] == uint64_t(9) * 3 * p.pl.k_hat + 13 * p.pl.k_hat);
    }
  imd_test_set_k_align(0);
}

void test_reason_order() {
  // Same tensor, other alignment: FORMAT exactly when the pack's A differs from the plan's.
  for (const int64_t k : {5, 96, 97, 200})
    for (const int32_t hp : {0, 16, 64})
      for (const int32_t hq : {0, 16, 64}) {
        Gemm g(k);
        imd_test_set_k_align(hp);
        Pack pa(g.a), pb(g.b);
        imd_test_set_k_align(hq);
        const int32_t ap = hp ? hp : (k > 96 ? 64 : 16), aq = hq ? hq : (k > 96 ? 64 : 16);
        const int32_t want = ap == aq ? 0 : IMD_PACK_MISMATCH_FORMAT;
        const PackLowered* combos[3][2] = {{&pa.pl, nullptr}, {nullptr, &pb.pl}, {&pa.pl, &pb.pl}};
        for (const auto& xy : combos) {
          const Res r = plan_with(g, xy[0], xy[1]);
          CHECK(r.reason == want &&
                r.st == (want ? IMD_STATUS_PACK_MISMATCH : IMD_STATUS_SUCCESS));
        }
        // with N also differing: MODULI (the format is compared last)
        Pack pn(g.a, 8);
        CHECK(plan_with(g, &pn.pl, nullptr).reason == IMD_PACK_MISMATCH_MODULI);
      }
  imd_test_set_k_align(0);
  // A pack of k = 96 (format 1) offered to a plan of k = 97 (format 2): EXTENTS, not FORMAT.
  for (const auto& [kp, kq] : {std::pair{96, 97}, std::pair{97, 96}, std::pair{64, 200}}) {
    Gemm gp(kp), gq(kq);
    Pack pa(gp.a), pb(gp.b);
    CHECK(pa.pl.format != pack_format_of(k_align_of(kq)));
    CHECK(plan_with(gq, &pa.pl, nullptr).reason == IMD_PACK_MISMATCH_EXTENTS);
    CHECK(plan_with(gq, nullptr, &pb.pl).reason == IMD_PACK_MISMATCH_EXTENTS);
  }
  // Both packed: the other key words of BOTH packs before the FORMAT of either.
  for (const int64_t k : {97, 200}) {
    Gemm g(k), gs(k, true);  // gs: B with other strides (k fastest)
    imd_test_set_k_align(16);
    Pack fa(g.a), fb(g.b);   // format 1: the "other alignment" for k > 96
    imd_test_set_k_align(0);
    Pack ok_a(g.a), ok_b(g.b), sb(gs.b);
    CHECK(fa.pl.format == 1 && ok_a.pl.format == 2 && sb.pl.format == 2);
    CHECK(plan_with(g, &fa.pl, &sb.pl).reason == IMD_PACK_MISMATCH_STRIDES);  // A FORMAT, B STRIDES
    CHECK(plan_with(g, &ok_a.pl, &sb.pl).reason == IMD_PACK_MISMATCH_STRIDES);
    CHECK(plan_with(g, &fa.pl, &ok_b.pl).reason == IMD_PACK_MISMATCH_FORMAT);
    CHECK(plan_with(g, &ok_a.pl, &fb.pl).reason == IMD_PACK_MISMATCH_FORMAT);
    CHECK(plan_with(g, &ok_a.pl, &ok_b.pl).st == IMD_STATUS_SUCCESS);
  }
  // An unknown format word (a future or foreign key layout): FORMAT before MODULI.
  {
    Gemm g(200);
    Pack p(g.a, 8);  // N differs too
    PackLowered pl = p.pl;
    pl.key[0] = 3;
    CHECK(plan_with(g, &pl, nullptr).reason == IMD_PACK_MISMATCH_FORMAT);
    pl.key[0] = 0;
    CHECK(plan_with(g, &pl, nullptr).reason == IMD_PACK_MISMATCH_FORMAT);
    CHECK(plan_with(g, &p.pl, nullptr).reason == IMD_PACK_MISMATCH_MODULI);
  }
  // Drift check: an equal key with another k_hat is FORMAT (unreachable by construction).
  {
    Gemm g(200);
    Pack p(g.a);
    PackLowered pl = p.pl;
    CHECK(plan_with(g, &pl, nullptr).st == IMD_STATUS_SUCCESS);
    pl.k_hat += 64;
    CHECK(plan_with(g, &pl, nullptr).reason == IMD_PACK_MISMATCH_FORMAT);
  }
  // A DIRECT plan accepts a matching pack (key format = the resolved A, not the path's 16).
  {
    Gemm g(200);
    Pack p(g.a);
    CHECK(plan_with(g, &p.pl, nullptr, opts(9, IMD_DIRECT)).st == IMD_STATUS_SUCCESS);
  }
  // key_mismatch: word 0 last.
  {
    // format, N, rounding, b, r, k; L: 0 modes; R: 1 mode (3, 200); K: 1 mode (200, 1)
    const std::vector<int64_t> s = {2, 9, 0, 1, 3, 200, 0, 1, 3, 200, 1, 200, 1};
    std::vector<int64_t> p = s;
    p[0] = 1;
    CHECK(key_mismatch(s, p) == IMD_PACK_MISMATCH_FORMAT);
    p[12] = 7;
    CHECK(key_mismatch(s, p) == IMD_PACK_MISMATCH_STRIDES);
    p.pop_back();
    p.pop_back();
    CHECK(key_mismatch(s, p) == IMD_PACK_MISMATCH_EXTENTS);
    CHECK(key_mismatch(s, s) == IMD_PACK_MISMATCH_NONE);
  }
}

}  // namespace

int main() {
  test_resolver();
  test_plan_paths();
  test_pack_lower();
  test_reason_order();
  imd_test_set_k_align(0);
  if (failures) {
    std::printf("k alignment: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  std::printf("k alignment (DESIGN v1.13 §4b): %d checks passed\n", checks);
  return 0;
}
