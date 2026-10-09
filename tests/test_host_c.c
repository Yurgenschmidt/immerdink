/* The host query ABI (DESIGN §6a) from C, linked against libimmerdink_host only: no CUDA
 * header, library or device. The DESIGN §4a worked example (17 x 33 x 5, b = 3, N = 9,
 * g = 2; spec/params.py cert_workspace), the region table, plan bytes, the lowering summary,
 * the RN acceptance rule and the ABI-versioning rules. */
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "immerdink/immerdink_host.h"

/* libimmerdink_host's test-only K-alignment hook (DESIGN v1.13 §4b.5; not in a header). */
int imd_test_set_k_align(int32_t a);

static double dabs(double x) { return x < 0 ? -x : x; }

#ifdef CUDART_VERSION
#  error "immerdink_host.h must not pull in a CUDA header"
#endif

static int failures = 0;

#define CHECK(cond)                                         \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); \
      ++failures;                                           \
    }                                                       \
  } while (0)

int main(void) {
  /* C[i,j,l] = sum_k A[i,k,l] B[k,j,l]; labels i=1 j=2 k=3 l=4; dense, last mode fastest */
  const int64_t eA[3] = {17, 5, 3}, sA[3] = {15, 3, 1};
  const int64_t eB[3] = {5, 33, 3}, sB[3] = {99, 3, 1};
  const int64_t eC[3] = {17, 33, 3}, sC[3] = {99, 3, 1};
  const int32_t lA[3] = {1, 3, 4}, lB[3] = {3, 2, 4}, lC[3] = {1, 2, 4};
  const imd_tensor_t A = {3, eA, sA, lA}, B = {3, eB, sB, lB}, C = {3, eC, sC, lC};
  const imd_options_t opt = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
  const imd_options_t res = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_RESIDENT, IMD_K4_TUNE_OFF};

  CHECK(imd_host_api_version() == IMD_HOST_API_VERSION);
  CHECK(sizeof(imd_lowering_t) == 192 && IMD_HOST_API_VERSION >= 9);
  CHECK(offsetof(imd_lowering_t, k_align) == 184 && offsetof(imd_lowering_t, reserved6) == 188);
  CHECK(IMD_PACK_FORMAT_A16 == 1 && IMD_PACK_FORMAT_A64 == 2 && IMD_PACK_FORMAT == 2);
  CHECK(sizeof(imd_options_t) == 24);
  CHECK(sizeof(imd_pack_desc_t) == 72 && sizeof(imd_pack_layout_t) == 136);

  size_t bytes = 0;
  CHECK(imd_workspace_size(&A, &B, &C, &opt, &bytes) == IMD_STATUS_SUCCESS);
  CHECK(bytes == 30720);

  /* §4a table: sA 102, sB 198, fA 51, fB 99, vec 952, RA 2*3*17*16, RB 2*3*48*16,
   * W 9*3*17*48, at cumulative 256-aligned offsets. */
  imd_region_t reg[IMD_REGION_COUNT + 2];
  int32_t nreg = 0;
  size_t total = 0;
  memset(reg, 0xff, sizeof reg);
  CHECK(imd_workspace_layout(&A, &B, &C, &opt, reg, IMD_REGION_COUNT + 2, &nreg, &total) ==
        IMD_STATUS_SUCCESS);
  CHECK(nreg == IMD_REGION_COUNT && total == 30720);
  {
    const uint64_t want[IMD_REGION_COUNT] = {102, 198, 51, 99, 952, 1632, 4608, 22032};
    uint64_t off = 0;
    for (int r = 0; r < IMD_REGION_COUNT; ++r) {
      CHECK(reg[r].offset == off && reg[r].bytes == want[r]);
      off += (want[r] + 255) / 256 * 256;
    }
    CHECK(off == total);
    CHECK(reg[IMD_REGION_COUNT].offset == UINT64_MAX); /* nothing past the table */
  }
  CHECK(strcmp(imd_region_name(IMD_REGION_VEC), "vec") == 0);
  CHECK(strcmp(imd_region_name(IMD_REGION_W), "W") == 0);
  CHECK(imd_region_name(IMD_REGION_COUNT) == NULL && imd_region_name(-1) == NULL);
  CHECK(imd_workspace_layout(&A, &B, &C, &opt, NULL, 0, &nreg, &total) == IMD_STATUS_SUCCESS);
  CHECK(imd_workspace_layout(&A, &B, &C, &opt, NULL, 1, &nreg, &total) ==
        IMD_STATUS_INVALID_VALUE);
  /* RESIDENT: no vec region (30720 - 1024 = 29696). */
  CHECK(imd_workspace_layout(&A, &B, &C, &res, reg, IMD_REGION_COUNT, &nreg, &total) ==
        IMD_STATUS_SUCCESS);
  CHECK(total == 29696 && reg[IMD_REGION_VEC].bytes == 0);

  /* Plan bytes: 8 (34 + 66 + 10 + 9) = 952 device bytes when resident, else 0. */
  size_t dev = 1, host = 0;
  CHECK(imd_plan_bytes(&A, &B, &C, &opt, &dev, &host) == IMD_STATUS_SUCCESS);
  CHECK(dev == 0 && host > 0);
  CHECK(imd_plan_bytes(&A, &B, &C, &res, &dev, NULL) == IMD_STATUS_SUCCESS);
  CHECK(dev == 952);

  /* The lowering summary. */
  imd_lowering_t lw;
  memset(&lw, 0, sizeof lw);
  lw.struct_size = sizeof lw;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
  CHECK(lw.status == IMD_STATUS_SUCCESS && lw.lowered == 1);
  CHECK(lw.dispatch == IMD_DISPATCH_EMULATE && lw.swapped == 0);
  CHECK(lw.m == 17 && lw.n == 33 && lw.k == 5 && lw.b == 3);
  CHECK(lw.n_hat == 48 && lw.k_hat == 16 && lw.kappa == 3);
  CHECK(lw.num_moduli == 9 && lw.residue_group == 2 && lw.num_groups == 5);
  CHECK(lw.k_chunks == 1 && lw.gemm_launches == 5);
  CHECK(lw.vec_entries == 119 && lw.workspace_bytes == 30720);
  CHECK(lw.plan_device_bytes == 0 && lw.plan_host_bytes == host);
  /* v2: k_hat = 16 <= 1024: the fused launch KF; then groups 2..5 with two packs and the B
   * padding rows (n_hat > n) each, 5 K4 launches and K5: 1 + 4 * 3 + 5 + 1 = 19. */
  CHECK(lw.fused == 1 && lw.reserved == 0 && lw.kernel_launches == 19);
  /* v4: KF, so no KL. */
  CHECK(lw.fused_large == 0 && lw.reserved4 == 0);
  /* v6 (DESIGN v1.13 §4b): k = 5 <= 96: A = 16 (k_hat 16 above) */
  CHECK(lw.k_align == 16 && lw.reserved6 == 0);

  /* v3: sm_count 0 -> the reference 128 SMs; a configuration of the table; the autotuning
   * scratch of group 1 (g = 2, b = 3): RA 2*3*17*16 = 1632, RB 2*3*48*16 = 4608, W
   * 2*3*17*48 = 4896, each 256-aligned: 1792 + 4608 + 5120. */
  CHECK(lw.sm_count == 128);
  /* DESIGN §4 rule: 1 tile of every configuration per batch, 6 batches on 128 SMs: one
   * round each; 64x64 has the smallest estimate (1.39 us vs 2.94 for 128x256). */
  CHECK(lw.k4_config == IMD_K4_64x64);
  CHECK(lw.k4_tune_bytes == 11520);
  {
    imd_lowering_t l2;
    memset(&l2, 0, sizeof l2);
    l2.struct_size = sizeof l2;
    l2.sm_count = 46; /* an input: the rule for another device */
    CHECK(imd_lowering(&A, &B, &C, &opt, &l2) == IMD_STATUS_SUCCESS);
    CHECK(l2.sm_count == 46 && l2.k4_config == IMD_K4_64x64);
    CHECK(l2.m == 17 && l2.k4_tune_bytes == 11520);
    l2.sm_count = -1; /* invalid input: nothing written */
    l2.m = -7;
    CHECK(imd_lowering(&A, &B, &C, &opt, &l2) == IMD_STATUS_INVALID_VALUE);
    CHECK(l2.m == -7);
    /* k4_tune is validated like every option; ON changes no host query. */
    const imd_options_t tune = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_ON};
    const imd_options_t badt = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, 2};
    size_t b2 = 0;
    CHECK(imd_workspace_size(&A, &B, &C, &tune, &b2) == IMD_STATUS_SUCCESS && b2 == 30720);
    CHECK(imd_workspace_size(&A, &B, &C, &badt, &b2) == IMD_STATUS_INVALID_VALUE);
    memset(&l2, 0, sizeof l2);
    l2.struct_size = sizeof l2;
    CHECK(imd_lowering(&A, &B, &C, &tune, &l2) == IMD_STATUS_SUCCESS);
    CHECK(l2.k4_config == lw.k4_config && l2.k4_tune_bytes == 11520 && l2.plan_host_bytes == host);
  }

  /* A version-5 caller (struct_size 184) gets the v5 fields; the v6 tail is untouched. */
  lw.k_align = -5;
  lw.reserved6 = -5;
  lw.struct_size = 184;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
  CHECK(lw.m == 17 && lw.k_hat == 16 && lw.pack_mismatch == 0 && lw.k_align == -5 &&
        lw.reserved6 == -5);

  /* A version-3 caller (struct_size 168) gets the v3 fields; the v4 tail is untouched. */
  lw.fused_large = -5;
  lw.struct_size = 168;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
  CHECK(lw.m == 17 && lw.k4_tune_bytes == 11520 && lw.fused_large == -5);

  /* A version-2 caller (struct_size 152) gets the v2 fields; the v3 tail is untouched. */
  lw.sm_count = -5;
  lw.k4_config = -5;
  lw.k4_tune_bytes = 5;
  lw.struct_size = 152;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
  CHECK(lw.m == 17 && lw.kernel_launches == 19 && lw.sm_count == -5 && lw.k4_config == -5 &&
        lw.k4_tune_bytes == 5);

  /* A version-1 caller (struct_size 136) gets the v1 fields; the v2 tail is untouched. */
  lw.fused = -5;
  lw.kernel_launches = -5;
  lw.struct_size = 136;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
  CHECK(lw.m == 17 && lw.fused == -5 && lw.kernel_launches == -5);

  /* Versioning: struct_size below the version-1 size -> INVALID_VALUE, nothing written. */
  lw.struct_size = 8;
  lw.m = -7;
  CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_INVALID_VALUE);
  CHECK(lw.m == -7);
  CHECK(imd_lowering(&A, &B, &C, &opt, NULL) == IMD_STATUS_INVALID_VALUE);

  /* §3.2 RN acceptance: N = 2 (T = 180), k = 180^2 -> kappa = 180 >= T -> NOT_SUPPORTED at
   * plan time; the workspace query and the lowering still succeed; TRUNC is accepted. */
  {
    const int64_t k = 180 * 180;
    const int64_t ea[2] = {4, k}, sa[2] = {k, 1}, eb[2] = {k, 16}, sb[2] = {1, k};
    const int64_t ec[2] = {4, 16}, sc[2] = {16, 1};
    const int32_t la[2] = {1, 3}, lb[2] = {3, 2}, lc[2] = {1, 2};
    const imd_tensor_t a = {2, ea, sa, la}, b = {2, eb, sb, lb}, c = {2, ec, sc, lc};
    const imd_options_t rn = {2, 0, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
    const imd_options_t tr = {2, 0, IMD_TRUNC, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
    CHECK(imd_workspace_size(&a, &b, &c, &rn, &bytes) == IMD_STATUS_SUCCESS);
    CHECK(imd_plan_bytes(&a, &b, &c, &rn, &dev, &host) == IMD_STATUS_NOT_SUPPORTED);
    CHECK(imd_plan_bytes(&a, &b, &c, &tr, &dev, &host) == IMD_STATUS_SUCCESS);
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&a, &b, &c, &rn, &lw) == IMD_STATUS_NOT_SUPPORTED);
    CHECK(lw.status == IMD_STATUS_NOT_SUPPORTED && lw.lowered == 1 && lw.kappa == 180);
    CHECK(lw.workspace_bytes == bytes && lw.plan_host_bytes == 0);
    /* k_hat > 1024: no KF; v1.6: KL for A and B (both bits), so K0 + KL A + KL B + 1 K4 +
     * K5 (group 1's packs are KL's; n_hat = n: no zero rows). */
    CHECK(lw.fused == 0 && lw.fused_large == (IMD_FUSED_LARGE_A | IMD_FUSED_LARGE_B));
    CHECK(lw.kernel_launches == 5);
    /* v1.13 §4b: k = 32400 > 96 (= 16 mod 64): A = 64, k_hat 32448. v3 fields are valid with
     * lowered = 1, as k_chunks: 2*4*32448 + 2*16*32448 + 2*4*16, each 256-aligned. */
    CHECK(lw.k_align == 64 && lw.k_hat == 32448);
    CHECK(lw.k4_tune_bytes == 259584 + 1038336 + 256 && lw.sm_count == 128);
  }

  /* Illegal descriptors: the §5 status, lowered = 0. */
  {
    const int32_t lbad[3] = {1, 1, 4}; /* repeated label in C */
    const imd_tensor_t cbad = {3, eC, sC, lbad};
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&A, &B, &cbad, &opt, &lw) == IMD_STATUS_INVALID_VALUE);
    CHECK(lw.lowered == 0 && lw.m == 0);
    CHECK(imd_workspace_size(&A, &B, &C, NULL, &bytes) == IMD_STATUS_INVALID_VALUE);
  }

  CHECK(strcmp(imd_status_string(IMD_STATUS_NOT_SUPPORTED), "IMD_STATUS_NOT_SUPPORTED") == 0);

  /* v1.13 §4b.4 hand-computed A = 64 examples (spec/params.py cert_workspace /
   * cert_pack_layout). (m, n, k, b, N, g) = (1000, 3, 200, 2, 20, 7), AUTO: k_hat = 256; RA
   * 3584000 at 26112, RB 57344 at 3610112, W 640000 at 3667456; total 4307456. */
  {
    const int64_t ea[3] = {1000, 200, 2}, sa[3] = {400, 2, 1};
    const int64_t eb[3] = {200, 3, 2}, sb[3] = {6, 2, 1};
    const int64_t ec[3] = {1000, 3, 2}, sc[3] = {6, 2, 1};
    const imd_tensor_t a = {3, ea, sa, lA}, b = {3, eb, sb, lB}, c = {3, ec, sc, lC};
    const imd_options_t o = {20, 7, IMD_TRUNC, IMD_EMULATE, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
    imd_region_t r8[IMD_REGION_COUNT];
    int32_t n8 = 0;
    size_t t8 = 0;
    CHECK(imd_workspace_size(&a, &b, &c, &o, &bytes) == IMD_STATUS_SUCCESS && bytes == 4307456);
    CHECK(imd_workspace_layout(&a, &b, &c, &o, r8, IMD_REGION_COUNT, &n8, &t8) == 0 &&
          t8 == 4307456);
    CHECK(r8[IMD_REGION_RA].offset == 26112 && r8[IMD_REGION_RA].bytes == 3584000 &&
          r8[IMD_REGION_RB].offset == 3610112 && r8[IMD_REGION_RB].bytes == 57344 &&
          r8[IMD_REGION_W].offset == 3667456 && r8[IMD_REGION_W].bytes == 640000);
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&a, &b, &c, &o, &lw) == IMD_STATUS_SUCCESS);
    CHECK(lw.k == 200 && lw.k_hat == 256 && lw.k_align == 64 && lw.fused == 1);
    /* DIRECT: no residue rows, A = 16 whatever k */
    const imd_options_t d = {20, 7, IMD_TRUNC, IMD_DIRECT, IMD_VEC_GENERATE, IMD_K4_TUNE_OFF};
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&a, &b, &c, &d, &lw) == IMD_STATUS_SUCCESS);
    CHECK(lw.dispatch == IMD_DISPATCH_DIRECT && lw.k_align == 16 && lw.k_hat == 208);
  }
  /* pack (b, r, k, N) = (1, 1, 1000, 14), AUTO: format 2, k_hat 1024, res 14*1024 + 15*1024 =
   * 29696 at 512; total 30208. */
  {
    const int64_t ex[1] = {1000}, sx[1] = {1};
    const int32_t lx[1] = {3}, kx[1] = {3};
    const imd_tensor_t x = {1, ex, sx, lx};
    imd_pack_desc_t d;
    imd_pack_layout_t pl;
    memset(&d, 0, sizeof d);
    d.struct_size = sizeof d;
    d.num_moduli = 14, d.rounding = IMD_TRUNC, d.order = IMD_PACK_ORDER_AUTO, d.tensor = &x;
    d.num_contracted = 1, d.contracted = kx;
    memset(&pl, 0, sizeof pl);
    pl.struct_size = sizeof pl;
    CHECK(imd_pack_layout(&d, &pl) == IMD_STATUS_SUCCESS);
    CHECK(pl.format == IMD_PACK_FORMAT_A64 && pl.k_hat == 1024 && pl.res.offset == 512 &&
          pl.res.bytes == 29696 && pl.pack_bytes == 30208);
  }
  /* The test hook forces A (16 or 64; 0 = AUTO; anything else: 1, nothing changed). */
  CHECK(imd_test_set_k_align(32) == 1 && imd_test_set_k_align(64) == 0);
  {
    /* (17, 33, 5, 3, 9, 2) at A = 64: k_hat 64; RA 6528 at 2048, RB 18432 at 8704, W 22032 at
     * 27136; total 49408 (A = 16: 30720). KF's tile becomes tk = 64. */
    imd_region_t r8[IMD_REGION_COUNT];
    int32_t n8 = 0;
    size_t t8 = 0;
    CHECK(imd_workspace_size(&A, &B, &C, &opt, &bytes) == IMD_STATUS_SUCCESS && bytes == 49408);
    CHECK(imd_workspace_layout(&A, &B, &C, &opt, r8, IMD_REGION_COUNT, &n8, &t8) == 0 &&
          r8[IMD_REGION_RA].offset == 2048 && r8[IMD_REGION_RA].bytes == 6528 &&
          r8[IMD_REGION_RB].offset == 8704 && r8[IMD_REGION_RB].bytes == 18432 &&
          r8[IMD_REGION_W].offset == 27136 && r8[IMD_REGION_W].bytes == 22032);
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&A, &B, &C, &opt, &lw) == IMD_STATUS_SUCCESS);
    CHECK(lw.k_align == 64 && lw.k_hat == 64 && lw.fused == 1 && lw.kappa == 3);
    /* pack (3, 17, 5, 9) forced to format 2: res 29376 + 960 = 30336; total 30976 */
    imd_pack_desc_t pa;
    imd_pack_layout_t pl;
    int32_t la[8];
    memset(&pa, 0, sizeof pa);
    pa.struct_size = sizeof pa;
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 0, &pa, la, 8) == IMD_STATUS_SUCCESS);
    memset(&pl, 0, sizeof pl);
    pl.struct_size = sizeof pl;
    CHECK(imd_pack_layout(&pa, &pl) == IMD_STATUS_SUCCESS);
    CHECK(pl.format == IMD_PACK_FORMAT_A64 && pl.k_hat == 64 && pl.res.bytes == 30336 &&
          pl.pack_bytes == 30976);
  }
  CHECK(imd_test_set_k_align(16) == 0);
  {
    /* forced 16 at k = 200: k_hat 208 */
    const int64_t ea[2] = {4, 200}, sa[2] = {200, 1}, eb[2] = {200, 16}, sb[2] = {1, 200};
    const int64_t ec[2] = {4, 16}, sc[2] = {16, 1};
    const int32_t la[2] = {1, 3}, lb[2] = {3, 2}, lc[2] = {1, 2};
    const imd_tensor_t a = {2, ea, sa, la}, b = {2, eb, sb, lb}, c = {2, ec, sc, lc};
    memset(&lw, 0, sizeof lw);
    lw.struct_size = sizeof lw;
    CHECK(imd_lowering(&a, &b, &c, &opt, &lw) == IMD_STATUS_SUCCESS);
    CHECK(lw.k_align == 16 && lw.k_hat == 208);
  }
  CHECK(imd_test_set_k_align(0) == 0);
  CHECK(imd_workspace_size(&A, &B, &C, &opt, &bytes) == IMD_STATUS_SUCCESS && bytes == 30720);

  /* v1.7 (DESIGN §6b): packed operands of the same contraction. No swap (C's fastest label
   * l is a batch label): A is the GEMM's A with K = {k}, L = {l}, R = {i}; B: K, L, R = {j}. */
  {
    imd_pack_desc_t pa, pb;
    int32_t la[8], lb[8];
    memset(&pa, 0, sizeof pa);
    memset(&pb, 0, sizeof pb);
    pa.struct_size = pb.struct_size = sizeof pa;
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 0, &pa, la, 8) == IMD_STATUS_SUCCESS);
    CHECK(pa.tensor == &A && pa.num_moduli == 9 && pa.rounding == IMD_RN &&
          pa.order == IMD_PACK_ORDER_GIVEN);
    CHECK(pa.num_contracted == 1 && pa.contracted[0] == 3 && pa.num_batch == 1 &&
          pa.batch[0] == 4 && pa.num_rows == 1 && pa.rows[0] == 1);
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 1, &pb, lb, 8) == IMD_STATUS_SUCCESS);
    CHECK(pb.tensor == &B && pb.num_rows == 1 && pb.rows[0] == 2);
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 2, &pb, lb, 8) == IMD_STATUS_INVALID_VALUE);
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 0, &pb, lb, 2) == IMD_STATUS_INVALID_VALUE);

    /* §6b.4 (params.py cert_pack_layout): b 3, r 17, k 5, N 9: sig 102, flag 51, res 7584 */
    imd_pack_layout_t pl;
    memset(&pl, 0, sizeof pl);
    pl.struct_size = sizeof pl;
    CHECK(imd_pack_layout(&pa, &pl) == IMD_STATUS_SUCCESS && pl.status == IMD_STATUS_SUCCESS);
    CHECK(pl.format == IMD_PACK_FORMAT_A16 && pl.num_moduli == 9 && pl.rounding == IMD_RN);
    CHECK(pl.b == 3 && pl.r == 17 && pl.k == 5 && pl.k_hat == 16 && pl.kappa == 3);
    CHECK(pl.sig.offset == 0 && pl.sig.bytes == 102 && pl.flag.offset == 256 &&
          pl.flag.bytes == 51 && pl.res.offset == 512 && pl.res.bytes == 7584);
    CHECK(pl.pack_bytes == 8192 && pl.pack_ws_bytes == 256 && pl.key_digest != 0);
    /* AUTO with the same classes: the same orders (one label per class), the same key */
    {
      const int32_t k1[1] = {3}, l1[1] = {4};
      imd_pack_desc_t au = pa;
      imd_pack_layout_t pl2;
      au.order = IMD_PACK_ORDER_AUTO;
      au.num_contracted = 1, au.contracted = k1, au.num_batch = 1, au.batch = l1;
      au.num_rows = 0, au.rows = NULL;
      memset(&pl2, 0, sizeof pl2);
      pl2.struct_size = sizeof pl2;
      CHECK(imd_pack_layout(&au, &pl2) == IMD_STATUS_SUCCESS && pl2.key_digest == pl.key_digest);
      au.num_rows = 1, au.rows = l1; /* AUTO with rows */
      CHECK(imd_pack_layout(&au, &pl2) == IMD_STATUS_INVALID_VALUE && pl2.status == 1 && pl2.b == 0);
      au.num_rows = 0, au.rows = NULL;
      au.batch = k1; /* label 3 listed twice */
      CHECK(imd_pack_layout(&au, &pl2) == IMD_STATUS_INVALID_VALUE);
      au.batch = l1, au.num_contracted = 0, au.contracted = NULL; /* K empty */
      CHECK(imd_pack_layout(&au, &pl2) == IMD_STATUS_NOT_SUPPORTED && pl2.b == 3 && pl2.r == 85 &&
            pl2.k == 1 && pl2.key_digest == 0);
    }
    /* §6b.6: the workspace without sX, fX, RX (cert_pack_layout: 28416, 25600, 23296) */
    imd_packing_t pkA = {&pa, NULL}, pkB = {NULL, &pb}, pkAB = {&pa, &pb}, none = {NULL, NULL};
    CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, &pkA, &bytes) == 0 && bytes == 28416);
    CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, &pkB, &bytes) == 0 && bytes == 25600);
    CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, &pkAB, &bytes) == 0 && bytes == 23296);
    CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, &none, &bytes) == 0 && bytes == 30720);
    CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, NULL, &bytes) == 0 && bytes == 30720);
    CHECK(imd_workspace_layout_packed(&A, &B, &C, &opt, &pkA, reg, IMD_REGION_COUNT, &nreg,
                                      &total) == 0 && total == 28416);
    CHECK(reg[IMD_REGION_SA].bytes == 0 && reg[IMD_REGION_FA].bytes == 0 &&
          reg[IMD_REGION_RA].bytes == 0 && reg[IMD_REGION_SB].offset == 0 &&
          reg[IMD_REGION_W].offset == 6144);
    /* the packed plan: no pack launches for A in the 4 later groups (KF packs B only) */
    imd_lowering_t lp, lu;
    memset(&lp, 0, sizeof lp);
    memset(&lu, 0, sizeof lu);
    lp.struct_size = lu.struct_size = sizeof lp;
    CHECK(imd_lowering(&A, &B, &C, &opt, &lu) == IMD_STATUS_SUCCESS && lu.packed == 0);
    CHECK(imd_lowering_packed(&A, &B, &C, &opt, &pkA, &lp) == IMD_STATUS_SUCCESS);
    CHECK(lp.packed == 1 && lp.pack_mismatch == 0 && lp.workspace_bytes == 28416 && lp.fused == 1);
    CHECK(lp.kernel_launches == lu.kernel_launches - 4 && lp.plan_host_bytes > lu.plan_host_bytes);
    CHECK(imd_lowering_packed(&A, &B, &C, &opt, &pkAB, &lp) == IMD_STATUS_SUCCESS);
    CHECK(lp.packed == 3 && lp.fused == 0 && lp.fused_large == 0 &&
          lp.kernel_launches == 1 + lp.gemm_launches + 1); /* K0, K4, K5 */
    /* PACK_MISMATCH reasons: N, rounding (the summary stays valid, lowered = 1) */
    {
      imd_pack_desc_t bad = pa;
      imd_packing_t pk = {&bad, NULL};
      bad.num_moduli = 8;
      CHECK(imd_lowering_packed(&A, &B, &C, &opt, &pk, &lp) == IMD_STATUS_PACK_MISMATCH);
      CHECK(lp.status == IMD_STATUS_PACK_MISMATCH && lp.lowered == 1 && lp.m == 17 &&
            lp.pack_mismatch == IMD_PACK_MISMATCH_MODULI && lp.plan_host_bytes == 0);
      CHECK(imd_workspace_size_packed(&A, &B, &C, &opt, &pk, &bytes) == IMD_STATUS_PACK_MISMATCH);
      bad.num_moduli = 9, bad.rounding = IMD_TRUNC;
      CHECK(imd_lowering_packed(&A, &B, &C, &opt, &pk, &lp) == IMD_STATUS_PACK_MISMATCH &&
            lp.pack_mismatch == IMD_PACK_MISMATCH_ROUNDING);
      /* B's pack in A's slot: other classes (r 33 vs 17, j is not A's label) */
      pk.A = &pb;
      CHECK(imd_lowering_packed(&A, &B, &C, &opt, &pk, &lp) == IMD_STATUS_PACK_MISMATCH &&
            lp.pack_mismatch == IMD_PACK_MISMATCH_CLASSES);
    }
    CHECK(strcmp(imd_status_string(IMD_STATUS_PACK_MISMATCH), "IMD_STATUS_PACK_MISMATCH") == 0);
  }

  /* ---- v1.8 (DESIGN §6c): the performance model ---------------------------------------- */
  {
    CHECK(IMD_HOST_API_VERSION >= 7);
    /* v8 sizes (v7: 3672, 1952, 168, 88, 40; v6: 1232, 96, 1648, 128, 64, 24; the older
     * prefixes are unchanged); v11: imd_perf_calib_t 3680 (energy_byte_f_pj at 3672); v12:
     * 3688 (pack_l2_hit_bytes at 3680); v13 (DESIGN v1.18 §6c.8): 3744, seven doubles from
     * clock_boost_mhz at 3688 to pack_l2_planes at 3736 */
    CHECK(sizeof(imd_perf_calib_t) == 3744 && offsetof(imd_perf_calib_t, energy_byte_f_pj) == 3672 &&
          offsetof(imd_perf_calib_t, pack_l2_hit_bytes) == 3680 && IMD_HOST_API_VERSION >= 13);
    CHECK(offsetof(imd_perf_calib_t, clock_boost_mhz) == 3688 &&
          offsetof(imd_perf_calib_t, power_act_w) == 3696 &&
          offsetof(imd_perf_calib_t, power_act_gap_us) == 3704 &&
          offsetof(imd_perf_calib_t, energy_dram_pj) == 3712 &&
          offsetof(imd_perf_calib_t, warm_call_ns) == 3720 &&
          offsetof(imd_perf_calib_t, warm_clean_byte_ns) == 3728 &&
          offsetof(imd_perf_calib_t, pack_l2_planes) == 3736 && IMD_PERF_CACHE_CLEAN == 3);
    CHECK(strcmp(imd_perf_param_name(IMD_PERF_CLASS_KF, 4), "elem") == 0 &&
          strcmp(imd_perf_param_name(IMD_PERF_CLASS_PACK, 3), "dram") == 0 &&
          imd_perf_param_name(IMD_PERF_CLASS_KL, 4) == NULL);
    CHECK(sizeof(imd_perf_phase_t) == 96 &&
          sizeof(imd_perf_t) == 1976 && sizeof(imd_perf_candidate_t) == 192 &&
          sizeof(imd_perf_rank_request_t) == 96 && sizeof(imd_perf_request_t) == 48);
    CHECK(offsetof(imd_perf_t, err_bound) == 1952 && offsetof(imd_perf_candidate_t, err_bound) == 168 &&
          offsetof(imd_perf_rank_request_t, accuracy) == 88 &&
          offsetof(imd_perf_request_t, accuracy) == 40);
    CHECK(sizeof(imd_err_request_t) == 64 && sizeof(imd_err_t) == 72);
    CHECK(offsetof(imd_perf_calib_t, kernel) == 1232 && offsetof(imd_perf_t, regime) == 1648 &&
          offsetof(imd_perf_candidate_t, time_lo_s) == 128 &&
          offsetof(imd_perf_rank_request_t, regime) == 64 &&
          offsetof(imd_perf_request_t, regime) == 24);
    static imd_perf_calib_t cal, back;
    memset(&cal, 0, sizeof cal);
    cal.struct_size = sizeof cal;
    CHECK(imd_perf_calib_default(&cal) == IMD_STATUS_SUCCESS);
    CHECK((cal.format == 1 || cal.format == IMD_PERF_CALIB_FORMAT) && cal.sm_count == 128 &&
          cal.gpu[0] != '\0');
    CHECK(cal.peak_dram_bytes_per_s > 0 && cal.peak_int8_ops_per_s > 0);
    /* format -> parse: bit-exact */
    static char text[16384];
    size_t len = 0;
    CHECK(imd_perf_calib_format(&cal, text, sizeof text, &len) == IMD_STATUS_SUCCESS);
    CHECK(len > 0 && len < sizeof text && strlen(text) == len);
    {
      size_t need = 0;
      char small[8];
      CHECK(imd_perf_calib_format(&cal, small, sizeof small, &need) ==
                IMD_STATUS_INSUFFICIENT_WORKSPACE && need == len);
    }
    memset(&back, 0x5a, sizeof back);
    back.struct_size = sizeof back;
    int32_t line = -1;
    CHECK(imd_perf_calib_parse(text, len, &back, &line) == IMD_STATUS_SUCCESS && line == 0);
    CHECK(memcmp(back.param, cal.param, sizeof cal.param) == 0);
    CHECK(strcmp(back.gpu, cal.gpu) == 0 && strcmp(back.clocks, cal.clocks) == 0);
    CHECK(back.sm_count == cal.sm_count &&
          memcmp(&back.peak_dram_bytes_per_s, &cal.peak_dram_bytes_per_s, 4 * sizeof(double)) == 0);
    /* v1.19 (DESIGN §6c.9, host API 13, no struct growth): the optional warm.kl_drop lives in
     * param[IMD_PERF_PARAM_KL_DROP_SLOT], in [0, 1], written only when non-zero */
    {
      static char t0[16700], t1[16700], bad[16800];
      static imd_perf_calib_t kd, kb;
      const int kslot = IMD_PERF_PARAM_KL_DROP_SLOT;
      size_t n0 = 0, n1 = 0;
      int32_t el = -1;
      CHECK(kslot == IMD_PERF_PARAM_CAPACITY - 1 && kslot == 95 &&
            kslot >= IMD_PERF_CLASS_COUNT * IMD_PERF_MAX_PARAMS &&
            cal.format == IMD_PERF_CALIB_FORMAT);
      kd = cal;
      kd.param[kslot] = 0;
      CHECK(imd_perf_calib_format(&kd, t0, sizeof t0, &n0) == IMD_STATUS_SUCCESS &&
            strstr(t0, "warm.kl_drop") == NULL);
      kd.param[kslot] = 0.5;
      CHECK(imd_perf_calib_format(&kd, t1, sizeof t1, &n1) == IMD_STATUS_SUCCESS &&
            strstr(t1, "\nwarm.kl_drop = 0.5\n") != NULL && n1 > n0);
      memset(&kb, 0x5a, sizeof kb);
      kb.struct_size = sizeof kb;
      CHECK(imd_perf_calib_parse(t1, n1, &kb, &el) == IMD_STATUS_SUCCESS && el == 0 &&
            kb.param[kslot] == 0.5 && memcmp(kb.param, kd.param, sizeof kd.param) == 0);
      {
        const char* const bads[3] = {"1.5", "-0.25", "nan"};
        int i;
        for (i = 0; i < 3; ++i) {
          snprintf(bad, sizeof bad, "%s\nwarm.kl_drop = %s\n", t0, bads[i]);
          el = 0;
          CHECK(imd_perf_calib_parse(bad, strlen(bad), &kb, &el) == IMD_STATUS_INVALID_VALUE &&
                el > 0);
        }
      }
    }
    /* host API 11 (DESIGN v1.15 §6c.7): the optional power.byte_f_pj, and the folded view
     * energy_byte_pj = byte + byte_f of a v7..v10-sized struct (nothing written beyond it) */
    {
      static char t0[16600], t2[16600];
      static imd_perf_calib_t base, full, v10;
      const uint32_t v10_size = (uint32_t)offsetof(imd_perf_calib_t, energy_byte_f_pj);
      /* the shipped calibration with byte_f = 0 (the combined refit ships byte_f > 0): its
       * text has no key */
      base = cal;
      base.energy_byte_f_pj = 0;
      size_t n0 = 0;
      CHECK(imd_perf_calib_format(&base, t0, sizeof t0, &n0) == IMD_STATUS_SUCCESS &&
            strstr(t0, "power.byte_f_pj") == NULL);
      CHECK(cal.energy_byte_f_pj == 0 || strstr(text, "\npower.byte_f_pj = ") != NULL);
      const int n2 = snprintf(t2, sizeof t2, "%spower.byte_f_pj = 7.5\n", t0);
      full.struct_size = sizeof full;
      CHECK(imd_perf_calib_parse(t2, (size_t)n2, &full, &line) == IMD_STATUS_SUCCESS &&
            full.energy_byte_f_pj == 7.5 && full.energy_byte_pj == cal.energy_byte_pj);
      memset(&v10, 0x5a, sizeof v10);
      v10.struct_size = v10_size;
      CHECK(imd_perf_calib_parse(t2, (size_t)n2, &v10, &line) == IMD_STATUS_SUCCESS &&
            v10.struct_size == v10_size && v10.energy_byte_pj == cal.energy_byte_pj + 7.5);
      const unsigned char* tail = (const unsigned char*)&v10.energy_byte_f_pj;
      for (size_t i = 0; i < sizeof v10.energy_byte_f_pj; ++i) CHECK(tail[i] == 0x5a);
      memset(&v10, 0x5a, sizeof v10);
      v10.struct_size = v10_size;
      CHECK(imd_perf_calib_default(&v10) == IMD_STATUS_SUCCESS &&
            v10.energy_byte_pj == cal.energy_byte_pj + cal.energy_byte_f_pj && tail[0] == 0x5a);
      /* format: the key only when non-zero; round trip */
      static char t3[16600];
      size_t len3 = 0;
      CHECK(imd_perf_calib_format(&full, t3, sizeof t3, &len3) == IMD_STATUS_SUCCESS &&
            strstr(t3, "\npower.byte_f_pj = 7.5\n") != NULL);
    }
    /* host API 12 (DESIGN v1.17 §6c.1 / §6c.2): warm.pack_l2_bytes optional, written only when
     * non-zero; a v7..v11-sized struct is not written beyond its size; the retired op_eps
     * keys are accepted and discarded */
    {
      static char u0[16600], u2[16600];
      static imd_perf_calib_t base, full, v11;
      const uint32_t v11_size = (uint32_t)offsetof(imd_perf_calib_t, pack_l2_hit_bytes);
      base = cal;
      base.pack_l2_hit_bytes = 0;
      size_t n0 = 0;
      CHECK(imd_perf_calib_format(&base, u0, sizeof u0, &n0) == IMD_STATUS_SUCCESS &&
            strstr(u0, "warm.pack_l2_bytes") == NULL);
      const int n2 = snprintf(u2, sizeof u2, "%swarm.pack_l2_bytes = 6.25e7\nkf.op_eps = 3\n"
                              "kl.op_eps = 0\npack.op_eps = 1e-3\n", u0);
      full.struct_size = sizeof full;
      CHECK(imd_perf_calib_parse(u2, (size_t)n2, &full, &line) == IMD_STATUS_SUCCESS &&
            full.pack_l2_hit_bytes == 6.25e7 &&
            full.param[IMD_PERF_CLASS_KF * IMD_PERF_MAX_PARAMS + 4] ==
                cal.param[IMD_PERF_CLASS_KF * IMD_PERF_MAX_PARAMS + 4] &&
            full.param[IMD_PERF_CLASS_PACK * IMD_PERF_MAX_PARAMS + 3] ==
                cal.param[IMD_PERF_CLASS_PACK * IMD_PERF_MAX_PARAMS + 3]);
      memset(&v11, 0x5a, sizeof v11);
      v11.struct_size = v11_size;
      CHECK(imd_perf_calib_parse(u2, (size_t)n2, &v11, &line) == IMD_STATUS_SUCCESS &&
            v11.struct_size == v11_size);
      const unsigned char* tail = (const unsigned char*)&v11.pack_l2_hit_bytes;
      for (size_t i = 0; i < sizeof v11.pack_l2_hit_bytes; ++i) CHECK(tail[i] == 0x5a);
      static char u3[16600];
      size_t len3 = 0;
      CHECK(imd_perf_calib_format(&full, u3, sizeof u3, &len3) == IMD_STATUS_SUCCESS &&
            strstr(u3, "\nwarm.pack_l2_bytes = 62500000\n") != NULL &&
            strstr(u3, "op_eps") == NULL);
    }
    /* parse errors, with their lines */
    {
      static char bad[16600];
      int n;
      n = snprintf(bad, sizeof bad, "%sbogus.key = 1\n", text); /* unknown key: last line */
      int32_t lines = 0;
      for (size_t i = 0; i < len; ++i) lines += text[i] == '\n';
      CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_INVALID_VALUE &&
            line == lines + 1);
      n = snprintf(bad, sizeof bad, "%sk5.byte = 1\n", text); /* repeated key */
      CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_INVALID_VALUE &&
            line == lines + 1);
      n = snprintf(bad, sizeof bad, "%s", text);
      char* p = strstr(bad, "k5.byte = ");
      CHECK(p != NULL);
      if (p) { p[0] = '#'; } /* a missing parameter: line 0 */
      CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_INVALID_VALUE &&
            line == 0);
      n = snprintf(bad, sizeof bad, "%s", text);
      p = strstr(bad, "\nformat = ");
      if (p) p[10] = '3'; /* another format (1 and 2 are read) */
      CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_INVALID_VALUE &&
            line > 0);
      const char* bad_values[] = {"kl.byte = -1\n", "kl.byte = 1x\n", "kl.byte = nan\n",
                                  "kl.byte\n"};
      for (int i = 0; i < 4; ++i) {
        n = snprintf(bad, sizeof bad, "%s%s", bad_values[i], text);
        CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_INVALID_VALUE &&
              line == 1);
      }
      /* the call term is a signed correction; '#' after a number is a comment */
      n = snprintf(bad, sizeof bad, "%s", text);
      p = strstr(bad, "call.launch = ");
      if (p) { p[0] = '#'; }
      n = snprintf(bad + n, sizeof bad - (size_t)n, "call.launch = -2.5  # ns\n") + n;
      CHECK(imd_perf_calib_parse(bad, (size_t)n, &back, &line) == IMD_STATUS_SUCCESS &&
            back.param[IMD_PERF_CLASS_CALL * IMD_PERF_MAX_PARAMS + 1] == -2.5);
      CHECK(imd_perf_calib_load("/nonexistent/x.calib", &back, &line) == IMD_STATUS_INVALID_VALUE &&
            line == 0);
    }
    /* host API 10 (DESIGN §6c.3): every rejection with its reason, key and message (the _ex
     * functions), the same status and line as the functions without _ex. */
    {
      CHECK(IMD_HOST_API_VERSION >= 10 && sizeof(imd_perf_calib_error_t) == 272);
      static char bad[16800];
      static imd_perf_calib_t tmp;
      imd_perf_calib_error_t e;
      int32_t lines = 0;
      for (size_t i = 0; i < len; ++i) lines += text[i] == '\n';
      /* parse_ex vs parse: same status, same line; returns the reason */
#define PARSE_EX(txt, n, want_reason, want_line, want_key, want_msg)                          \
      do {                                                                                    \
        memset(&e, 0x5a, sizeof e);                                                           \
        e.struct_size = sizeof e;                                                             \
        tmp.struct_size = sizeof tmp;                                                         \
        int32_t l_old = -1;                                                                   \
        const imd_status_t s_old = imd_perf_calib_parse((txt), (n), &tmp, &l_old);            \
        const imd_status_t s_ex = imd_perf_calib_parse_ex((txt), (n), &tmp, &e);              \
        CHECK(s_old == s_ex && e.struct_size == sizeof e && e.reserved == 0);                 \
        CHECK(e.reason == (want_reason) && e.line == (want_line) && l_old == e.line);         \
        CHECK(strcmp(e.key, (want_key)) == 0 && strstr(e.message, (want_msg)) != NULL);       \
        CHECK((s_ex == IMD_STATUS_SUCCESS) == ((want_reason) == IMD_PERF_CALIB_OK));          \
      } while (0)
      PARSE_EX(text, len, IMD_PERF_CALIB_OK, 0, "", "");
      CHECK(e.message[0] == '\0');
      /* the notes line over its field (the failure that broke a shipped calibration):
       * 127 bytes fit, 128 do not */
      for (int over = 0; over < 2; ++over) {
        char notes[160];
        memset(notes, 'x', sizeof notes);
        notes[127 + over] = '\0';
        int n = snprintf(bad, sizeof bad, "notes = %s\n%s", notes, text);
        char* p = strstr(bad + 1, "\nnotes = ");
        CHECK(p != NULL);
        if (p) p[1] = '#'; /* the canonical notes line: commented out */
        if (over) {
          PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_TOO_LONG, 1, "notes",
                   "line 1: notes: value is 128 bytes, the field holds 127");
        } else {
          PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_OK, 0, "", "");
          CHECK(strlen(tmp.notes) == 127);
        }
      }
      {
        char gpu[80];
        memset(gpu, 'g', sizeof gpu);
        gpu[64] = '\0';
        int n = snprintf(bad, sizeof bad, "gpu = %s\n%s", gpu, text);
        PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_TOO_LONG, 1, "gpu", "the field holds 63");
      }
      int n = snprintf(bad, sizeof bad, "%sbogus.key = 1\n", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_UNKNOWN_KEY, lines + 1, "bogus.key",
               "unknown key");
      n = snprintf(bad, sizeof bad, "%sk5.byte = 1\n", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_REPEATED_KEY, lines + 1, "k5.byte",
               "repeated key (first on line ");
      n = snprintf(bad, sizeof bad, "%s", text);
      char* p = strstr(bad, "k5.byte = ");
      CHECK(p != NULL);
      if (p) p[0] = '#';
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_MISSING_KEY, 0, "k5.byte", "missing key");
      n = snprintf(bad, sizeof bad, "%s", text);
      p = strstr(bad, "\nkernel.k0 = ");
      CHECK(p != NULL);
      if (p) p[1] = '#';
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_MISSING_KEY, 0, "kernel.k0", "missing key");
      n = snprintf(bad, sizeof bad, "%s", text);
      p = strstr(bad, "\nformat = ");
      if (p) p[10] = '3';
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_VALUE, 3, "format", "(want 1 or 2)");
      const char* vals[] = {"kl.byte = -1\n", "kl.byte = 1x\n", "kl.byte = nan\n"};
      for (int i = 0; i < 3; ++i) {
        n = snprintf(bad, sizeof bad, "%s%s", vals[i], text);
        PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_VALUE, 1, "kl.byte", "invalid value");
      }
      n = snprintf(bad, sizeof bad, "kl.byte\n%s", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_SYNTAX, 1, "", "line 1: 'kl.byte' is not");
      n = snprintf(bad, sizeof bad, "kernel.k0 = 0 1 2 3\n%s", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_VALUE, 1, "kernel.k0", "resident >= 1");
      n = snprintf(bad, sizeof bad, "band.call_small = 2 1\n%s", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_VALUE, 1, "band.call_small", "lo <= hi");
      /* a non-printable key byte is shown as '?' */
      n = snprintf(bad, sizeof bad, "bo\x01gus = 1\n%s", text);
      PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_UNKNOWN_KEY, 1, "bo?gus", "unknown key");
      /* a format-2 key in a format-1 calibration: the first one's line */
      if (cal.format == 2) {
        n = snprintf(bad, sizeof bad, "%s", text);
        p = strstr(bad, "\nformat = 2");
        if (p) p[10] = '1';
        int32_t first_v2 = 0, l = 1;
        const char* q = bad;
        while (first_v2 == 0 && *q) {
          if (strncmp(q, "k0.chain", 8) == 0 || strncmp(q, "kernel.", 7) == 0 ||
              strncmp(q, "power.", 6) == 0)
            first_v2 = l;
          const char* nl = strchr(q, '\n');
          if (!nl) break;
          q = nl + 1;
          ++l;
        }
        CHECK(first_v2 > 0);
        memset(&e, 0, sizeof e);
        e.struct_size = sizeof e;
        tmp.struct_size = sizeof tmp;
        CHECK(imd_perf_calib_parse_ex(bad, (size_t)n, &tmp, &e) == IMD_STATUS_INVALID_VALUE);
        CHECK(e.reason == IMD_PERF_CALIB_ERR_FORMAT2_KEY && e.line > 0 && e.line <= first_v2 &&
              strstr(e.message, "a format-2 key in a format-1 calibration") != NULL);
        /* cross-key rules: line 0 */
        n = snprintf(bad, sizeof bad, "%s", text);
        p = strstr(bad, "\nrank.tie_ratio = ");
        CHECK(p != NULL);
        if (p) p[1] = '#';
        n = snprintf(bad + n, sizeof bad - (size_t)n, "rank.tie_ratio = 2\n") + n;
        PARSE_EX(bad, (size_t)n, IMD_PERF_CALIB_ERR_INCONSISTENT, 0, "rank.tie_ratio",
                 "must be in [1, 1.25]");
      }
      /* load_ex: a missing file; ARGUMENT */
      memset(&e, 0x5a, sizeof e);
      e.struct_size = sizeof e;
      CHECK(imd_perf_calib_load_ex("/nonexistent/x.calib", &tmp, &e) == IMD_STATUS_INVALID_VALUE &&
            e.reason == IMD_PERF_CALIB_ERR_FILE && e.line == 0 && e.key[0] == '\0' &&
            strstr(e.message, "cannot open '/nonexistent/x.calib'") != NULL);
      memset(&e, 0x5a, sizeof e);
      e.struct_size = sizeof e;
      CHECK(imd_perf_calib_parse_ex(text, len, NULL, &e) == IMD_STATUS_INVALID_VALUE &&
            e.reason == IMD_PERF_CALIB_ERR_ARGUMENT);
      CHECK(imd_perf_calib_load_ex(NULL, &tmp, &e) == IMD_STATUS_INVALID_VALUE &&
            e.reason == IMD_PERF_CALIB_ERR_ARGUMENT);
      /* err may be NULL; err->struct_size < 16: INVALID_VALUE, nothing written; 16: only the
       * first 16 bytes */
      CHECK(imd_perf_calib_parse_ex(text, len, &tmp, NULL) == IMD_STATUS_SUCCESS);
      memset(&e, 0x5a, sizeof e);
      e.struct_size = 12;
      CHECK(imd_perf_calib_parse_ex(text, len, &tmp, &e) == IMD_STATUS_INVALID_VALUE &&
            e.line == 0x5a5a5a5a);
      memset(&e, 0x5a, sizeof e);
      e.struct_size = 16;
      n = snprintf(bad, sizeof bad, "%sbogus.key = 1\n", text);
      CHECK(imd_perf_calib_parse_ex(bad, (size_t)n, &tmp, &e) == IMD_STATUS_INVALID_VALUE &&
            e.struct_size == 16 && e.line == lines + 1 &&
            e.reason == IMD_PERF_CALIB_ERR_UNKNOWN_KEY && e.reserved == 0 &&
            (unsigned char)e.key[0] == 0x5a);
      /* the shipped calibration parses: reason OK */
      memset(&e, 0x5a, sizeof e);
      e.struct_size = sizeof e;
      tmp.struct_size = sizeof tmp;
      CHECK(imd_perf_calib_default_ex(&tmp, &e) == IMD_STATUS_SUCCESS &&
            e.reason == IMD_PERF_CALIB_OK && e.line == 0 && e.message[0] == '\0' &&
            memcmp(tmp.param, cal.param, sizeof cal.param) == 0);
      CHECK(imd_perf_calib_default_ex(NULL, &e) == IMD_STATUS_INVALID_VALUE &&
            e.reason == IMD_PERF_CALIB_ERR_ARGUMENT);
      CHECK(strcmp(imd_perf_calib_error_name(IMD_PERF_CALIB_OK), "IMD_PERF_CALIB_OK") == 0 &&
            strcmp(imd_perf_calib_error_name(IMD_PERF_CALIB_ERR_TOO_LONG),
                   "IMD_PERF_CALIB_ERR_TOO_LONG") == 0 &&
            strcmp(imd_perf_calib_error_name(IMD_PERF_CALIB_ERR_INCONSISTENT),
                   "IMD_PERF_CALIB_ERR_INCONSISTENT") == 0 &&
            imd_perf_calib_error_name(IMD_PERF_CALIB_ERR_COUNT) == NULL &&
            imd_perf_calib_error_name(-1) == NULL);
#undef PARSE_EX
    }
    CHECK(strcmp(imd_perf_class_name(IMD_PERF_CLASS_KL), "kl") == 0);
    CHECK(strcmp(imd_perf_param_name(IMD_PERF_CLASS_K4_64x64, 3), "mac3") == 0);
    CHECK(strcmp(imd_perf_param_name(IMD_PERF_CLASS_K4_64x64, 4), "byte") == 0);
    CHECK(imd_perf_param_name(IMD_PERF_CLASS_K4_128x256, 2) == NULL);
    CHECK(strcmp(imd_perf_phase_name(IMD_PERF_PHASE_PACK_B), "K3 pack B") == 0);
    CHECK(imd_perf_phase_name(IMD_PERF_PHASE_COUNT) == NULL && imd_perf_class_name(-1) == NULL);

    /* predict: KF, the later groups' packs, K4, K5, the call term */
    static imd_perf_t pr;
    memset(&pr, 0, sizeof pr);
    pr.struct_size = sizeof pr;
    imd_lowering_t lw2;
    memset(&lw2, 0, sizeof lw2);
    lw2.struct_size = sizeof lw2;
    CHECK(imd_lowering(&A, &B, &C, &opt, &lw2) == IMD_STATUS_SUCCESS);
    CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, NULL, NULL, &pr) == IMD_STATUS_SUCCESS);
    CHECK(pr.status == IMD_STATUS_SUCCESS && pr.dispatch == IMD_DISPATCH_EMULATE);
    CHECK(pr.num_phases == 6 && pr.phase[0].phase == IMD_PERF_PHASE_KF &&
          pr.phase[1].phase == IMD_PERF_PHASE_PACK_A && pr.phase[2].phase == IMD_PERF_PHASE_PACK_B &&
          pr.phase[3].phase == IMD_PERF_PHASE_K4 && pr.phase[4].phase == IMD_PERF_PHASE_K5 &&
          pr.phase[5].phase == IMD_PERF_PHASE_CALL);
    CHECK(pr.phase[1].launches == 4 && pr.phase[2].launches == 8); /* B: + zero rows (48 > 33) */
    CHECK(pr.kernel_launches == lw2.kernel_launches && pr.k4_config == lw2.k4_config);
    CHECK(pr.workspace_bytes == 30720 && pr.plan_device_bytes == 0 && pr.sm_count == 128);
    {
      double sum = 0;
      for (int i = 0; i < pr.num_phases; ++i) sum += pr.phase[i].time_s;
      CHECK(pr.time_s > 0 && sum == pr.time_s);
      CHECK(pr.flops_eq == 2.0 * 17 * 33 * 5 * 3);
    }
    {
      imd_perf_request_t rq = {sizeof rq, IMD_PERF_ALPHA_ZERO, -1, 0, -1.0, 0, 0, 0.0, NULL};
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_SUCCESS);
      CHECK(pr.num_phases == 3 && pr.phase[0].phase == IMD_PERF_PHASE_K0 &&
            pr.phase[1].phase == IMD_PERF_PHASE_BETA && pr.kernel_launches == 2);
      rq.flags = 4;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_INVALID_VALUE);
      rq.flags = 0, rq.k4_config = IMD_K4_CONFIG_COUNT;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_INVALID_VALUE);
      rq.k4_config = -1, rq.eps_fraction = 1.5; /* a fraction above 1 */
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_INVALID_VALUE);
      /* host API 11 (DESIGN v1.15 §6c.1): the eps path is retired; a valid fraction is ignored
       * (eps_s and eps_fraction out are 0); NaN is still rejected */
      { volatile double z = 0.0; rq.eps_fraction = z / z; }
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_INVALID_VALUE);
      rq.eps_fraction = 1.0;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_SUCCESS &&
            pr.eps_fraction == 0.0 && pr.eps_s == 0.0);
      rq.eps_fraction = -1.0;
      back = cal;
      back.format = 3;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, NULL, &back, &pr) == IMD_STATUS_INVALID_VALUE);
      /* v7: a version-6-sized calibration, request and result are still accepted. The
       * calibration is then a TRUE format-1 view (the format-2 parameters zeroed: a degraded
       * model, the v6 formulas) and equals the parse of its own format-1 text; a bad regime
       * is not accepted */
      {
        static imd_perf_calib_t c6, c6b;
        static imd_perf_t p6, p7;
        static char t6[16384];
        size_t n6 = 0;
        memset(&c6, 0, sizeof c6);
        c6.struct_size = 1232;
        CHECK(imd_perf_calib_default(&c6) == IMD_STATUS_SUCCESS && c6.format == 1 &&
              c6.struct_size == 1232 && c6.kernel[0].resident == 0);
        CHECK(c6.param[IMD_PERF_CLASS_K5 * IMD_PERF_MAX_PARAMS + 4] == 0.0 &&  /* k5.dram */
              c6.param[IMD_PERF_CLASS_KL * IMD_PERF_MAX_PARAMS + 5] == 0.0);   /* kl.wave */
        CHECK(imd_perf_calib_format(&c6, t6, sizeof t6, &n6) == IMD_STATUS_SUCCESS &&
              strstr(t6, "format = 1") != NULL && strstr(t6, "k5.dram") == NULL);
        memset(&c6b, 0x5a, sizeof c6b);
        c6b.struct_size = 1232;
        CHECK(imd_perf_calib_parse(t6, n6, &c6b, NULL) == IMD_STATUS_SUCCESS &&
              memcmp(&c6, &c6b, 1232) == 0);
        memset(&p6, 0, sizeof p6);
        p6.struct_size = 1648;
        imd_perf_request_t r6 = {24, 0, -1, 0, -1.0, 99, 99, -5.0, NULL};  /* v7 fields ignored */
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r6, &c6, &p6) == IMD_STATUS_SUCCESS &&
              p6.time_s > 0 && p6.regime == 0 && p6.time_lo_s == 0);
        memset(&p7, 0, sizeof p7);
        p7.struct_size = sizeof p7;
        imd_perf_request_t r7 = {sizeof r7, 0, -1, 0, -1.0, IMD_PERF_REGIME_BURST,
                                 IMD_PERF_CACHE_COLD, 0.0, NULL};
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r7, &c6, &p7) == IMD_STATUS_SUCCESS &&
              p7.time_s == p6.time_s && p7.time_lo_s == p7.time_s);
        r7.regime = 7;
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r7, NULL, &p7) == IMD_STATUS_INVALID_VALUE);
        r7.regime = 0, r7.cache = IMD_PERF_CACHE_CLEAN;  /* v13 (DESIGN v1.18 §6c.8) */
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r7, NULL, &p7) == IMD_STATUS_SUCCESS &&
              p7.cache == IMD_PERF_CACHE_CLEAN && p7.time_s > 0);
        r7.cache = 4;
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r7, NULL, &p7) == IMD_STATUS_INVALID_VALUE);
        r7.cache = 0, r7.clock_mhz = -1.0;
        CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &r7, NULL, &p7) == IMD_STATUS_INVALID_VALUE);
        CHECK(strcmp(imd_perf_kernel_name(IMD_PERF_KERNEL_KF16), "kf16") == 0 &&
              strcmp(imd_perf_kernel_name(IMD_PERF_KERNEL_K5 + 4 * 12 + 2), "k5_n14_e4") == 0 &&
              imd_perf_kernel_name(IMD_PERF_KERNEL_COUNT) == NULL);
      }
      pr.struct_size = 100;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, NULL, &cal, &pr) == IMD_STATUS_INVALID_VALUE);
      pr.struct_size = sizeof pr;
    }

    /* predict_pack: imd_pack of A (prep, KL with every plane, the call term) */
    imd_pack_desc_t pa2;
    int32_t labs[3];
    memset(&pa2, 0, sizeof pa2);
    pa2.struct_size = sizeof pa2;
    CHECK(imd_pack_desc_for(&A, &B, &C, &opt, 0, &pa2, labs, 3) == IMD_STATUS_SUCCESS);
    imd_pack_layout_t pl;
    memset(&pl, 0, sizeof pl);
    pl.struct_size = sizeof pl;
    CHECK(imd_pack_layout(&pa2, &pl) == IMD_STATUS_SUCCESS);
    CHECK(imd_perf_predict_pack(&pa2, NULL, NULL, &pr) == IMD_STATUS_SUCCESS);
    CHECK(pr.num_phases == 3 && pr.phase[0].phase == IMD_PERF_PHASE_PACK_PREP &&
          pr.phase[1].phase == IMD_PERF_PHASE_PACK_KL && pr.pack_bytes == pl.pack_bytes &&
          pr.pack_ws_bytes == pl.pack_ws_bytes && pr.time_s > 0);

    /* rank: N = 9, every g, both vec modes, A packed or not: 36 candidates, all SUCCESS */
    const int32_t Ns[1] = {9};
    imd_perf_rank_request_t rr;
    memset(&rr, 0, sizeof rr);
    rr.struct_size = sizeof rr, rr.candidate_size = sizeof(imd_perf_candidate_t);
    rr.num_moduli_count = 1, rr.num_moduli = Ns;
    rr.vec_modes = IMD_PERF_VEC_GENERATE | IMD_PERF_VEC_RESIDENT;
    rr.pack_modes = IMD_PERF_PACK_NONE | IMD_PERF_PACK_A;
    rr.reuse = 4;
    rr.order = IMD_PERF_ORDER_TIME; /* v7: the v6 time order (ties by band: test_host.py) */
    static imd_perf_candidate_t cand[64];
    int32_t nc = 0;
    CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, NULL, 0, &nc) == IMD_STATUS_SUCCESS && nc == 36);
    CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_SUCCESS && nc == 36);
    for (int i = 0; i < nc; ++i) {
      const imd_perf_candidate_t* c = &cand[i];
      CHECK(c->status == IMD_STATUS_SUCCESS && c->feasible == 1);
      if (i > 0) CHECK(cand[i - 1].time_s <= c->time_s);
      CHECK(c->raw_rank == i && c->time_lo_s <= c->time_s && c->time_s <= c->time_hi_s);
      const imd_options_t o = {c->num_moduli, c->residue_group, IMD_RN, IMD_EMULATE, c->vec_mode,
                               IMD_K4_TUNE_OFF};
      size_t ws = 0, pdev = 0;
      if (c->packed == 0) {
        CHECK(imd_workspace_size(&A, &B, &C, &o, &ws) == 0 && ws == c->workspace_bytes);
        CHECK(c->pack_bytes == 0 && c->pack_s == 0 && c->time_s == c->contract_s);
      } else {
        const imd_packing_t pk = {&pa2, NULL};
        CHECK(imd_workspace_size_packed(&A, &B, &C, &o, &pk, &ws) == 0 && ws == c->workspace_bytes);
        CHECK(c->pack_bytes == pl.pack_bytes && c->pack_ws_bytes == pl.pack_ws_bytes);
        CHECK(c->time_s == c->contract_s + c->pack_s / 4);
      }
      CHECK(imd_plan_bytes(&A, &B, &C, &o, &pdev, NULL) == 0 && pdev == c->plan_device_bytes);
      CHECK(c->total_bytes == c->workspace_bytes + c->plan_device_bytes + c->pack_bytes +
                                  c->pack_ws_bytes);
    }
    /* a budget: the feasible ones first, every one within it */
    rr.budget_bytes = 28000; /* g = 1 unpacked fits (27648 B; resident 26624 + 952) */
    CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_SUCCESS);
    {
      int seen_infeasible = 0, feasible = 0;
      for (int i = 0; i < nc; ++i) {
        if (cand[i].feasible) {
          CHECK(!seen_infeasible && cand[i].total_bytes <= 28000);
          ++feasible;
        } else {
          seen_infeasible = 1;
          CHECK(cand[i].total_bytes > 28000);
        }
      }
      CHECK(feasible > 0 && feasible < nc);
    }
    rr.reuse = 0;
    CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_INVALID_VALUE);
    rr.reuse = 1, rr.num_moduli_count = 0;
    CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_INVALID_VALUE);

    /* ---- v8 (DESIGN §6f): accuracy-aware planning ---------------------------------------- */
    {
      imd_err_t er;
      memset(&er, 0, sizeof er);
      er.struct_size = sizeof er;
      /* N = 14, RN, k = 10^4, M = 10^6, NORMWISE (spec/params.py error_estimate) */
      CHECK(imd_err_estimate(14, IMD_RN, IMD_DISPATCH_EMULATE, 10000, 1000000, NULL, &er) ==
            IMD_STATUS_SUCCESS && er.status == 0 && er.err_status == IMD_ERR_UNCHECKED);
      CHECK(dabs(er.bound / 7.535111076066045e-15 - 1) < 1e-11 &&
            dabs(er.typical / 7.933485279084879e-16 - 1) < 1e-11 && er.typical <= er.bound);
      imd_err_request_t ar;
      memset(&ar, 0, sizeof ar);
      ar.struct_size = sizeof ar, ar.metric = IMD_ERR_ELEMENTWISE;  /* no cond: invalid */
      CHECK(imd_err_estimate(14, IMD_RN, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      ar.cond = 1e6, ar.target = 1e-9, ar.reserved = 1; /* reserved must be 0 */
      CHECK(imd_err_estimate(14, IMD_RN, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      ar.reserved = 0;
      CHECK(imd_err_estimate(14, IMD_RN, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_SUCCESS);
      CHECK(imd_err_estimate(1, IMD_RN, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      CHECK(imd_err_estimate(14, 7, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      /* DIRECT: gamma_k; NOOP: 0 */
      CHECK(imd_err_estimate(9, IMD_RN, IMD_DISPATCH_DIRECT, 4, 1, NULL, &er) == 0 &&
            er.bound >= 4 * 0x1p-53 && er.bound < 4.01 * 0x1p-53 && er.typical == er.bound);
      CHECK(imd_err_estimate(9, IMD_RN, IMD_DISPATCH_NOOP, 4, 0, NULL, &er) == 0 && er.bound == 0);
      /* min_moduli on this contraction == a scan of imd_err_estimate */
      imd_lowering_t lw8;
      memset(&lw8, 0, sizeof lw8);
      lw8.struct_size = sizeof lw8;
      CHECK(imd_lowering(&A, &B, &C, &opt, &lw8) == IMD_STATUS_SUCCESS);
      memset(&ar, 0, sizeof ar);
      ar.struct_size = sizeof ar, ar.target = 1e-12;
      int32_t want = 0;
      for (int32_t n = 2; n <= 20 && want == 0; ++n) {
        CHECK(imd_err_estimate(n, IMD_RN, lw8.dispatch, lw8.k, lw8.b * lw8.m * lw8.n, &ar, &er) == 0);
        if (er.bound <= 1e-12) want = n;
      }
      CHECK(want > 2 && imd_err_min_moduli(&A, &B, &C, &opt, 0, 0, &ar, &er) == IMD_STATUS_SUCCESS &&
            er.num_moduli == want && er.bound <= 1e-12 && er.err_status == IMD_ERR_MET);
      CHECK(imd_err_min_moduli(&A, &B, &C, &opt, 2, want - 1, &ar, &er) == IMD_STATUS_NOT_SUPPORTED &&
            er.num_moduli == 0 && er.bound > 1e-12);
      ar.target = 0;  /* min_moduli needs a target */
      CHECK(imd_err_min_moduli(&A, &B, &C, &opt, 0, 0, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      /* predict: the v8 fields; rank: candidates missing the target are infeasible */
      ar.target = 1e-12;
      imd_perf_request_t rq8 = {sizeof rq8, 0, -1, 0, -1.0, 0, 0, 0.0, &ar};
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq8, NULL, &pr) == IMD_STATUS_SUCCESS &&
            pr.err_status == (opt.num_moduli >= want ? IMD_ERR_MET : IMD_ERR_MISSED));
      const int32_t Ns8[3] = {want - 1, want, want + 1};
      const int32_t gs8[1] = {1};
      rr.num_moduli_count = 3, rr.num_moduli = Ns8, rr.budget_bytes = 0, rr.accuracy = &ar;
      rr.residue_group_count = 1, rr.residue_groups = gs8; /* 3 N x 2 vec x 2 packs = 12 */
      CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_SUCCESS);
      CHECK(cand[0].feasible && cand[0].num_moduli >= want);
      for (int i = 0; i < nc; ++i)
        CHECK(cand[i].feasible == (cand[i].num_moduli >= want) &&
              (cand[i].infeasible == 0) == (cand[i].num_moduli >= want) &&
              (cand[i].num_moduli >= want || cand[i].infeasible == IMD_PERF_INFEASIBLE_ACCURACY));
      rr.struct_size = 88; /* a v7-sized request: no accuracy, all feasible */
      CHECK(imd_perf_rank(&A, &B, &C, &opt, &rr, NULL, cand, 64, &nc) == IMD_STATUS_SUCCESS);
      for (int i = 0; i < nc; ++i) CHECK(cand[i].feasible == 1);
    }
  }
  printf("host C API: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
