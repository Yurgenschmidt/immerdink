/* The host query ABI (DESIGN §6a) from C, linked against libimmerdink_host only: no CUDA
 * header, library or device. The DESIGN §4a worked example (17 x 33 x 5, b = 3, N = 9,
 * g = 2; spec/params.py cert_workspace), the region table, plan bytes, the lowering summary,
 * the RN acceptance rule and the ABI-versioning rules. */
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "immerdink/immerdink_host.h"

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
  CHECK(sizeof(imd_lowering_t) == 184 && IMD_HOST_API_VERSION >= 5);
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
    /* v3 fields are valid with lowered = 1, as k_chunks: 259200 + 1036800 + 128, aligned. */
    CHECK(lw.k4_tune_bytes == 259328 + 1036800 + 256 && lw.sm_count == 128);
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
    CHECK(pl.format == IMD_PACK_FORMAT && pl.num_moduli == 9 && pl.rounding == IMD_RN);
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
     * prefixes are unchanged) */
    CHECK(sizeof(imd_perf_calib_t) == 3672 && sizeof(imd_perf_phase_t) == 96 &&
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
      /* N = 9: T < 2^53, so the eps fraction is 0 whatever the request */
      rq.eps_fraction = 1.0;
      CHECK(imd_perf_predict(&A, &B, &C, &opt, NULL, &rq, &cal, &pr) == IMD_STATUS_SUCCESS &&
            pr.eps_fraction == 0.0);
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
        r7.regime = 0, r7.cache = 3;
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
            dabs(er.typical / 6.702646867434198e-16 - 1) < 1e-11 && er.typical <= er.bound);
      imd_err_request_t ar;
      memset(&ar, 0, sizeof ar);
      ar.struct_size = sizeof ar, ar.metric = IMD_ERR_ELEMENTWISE;  /* no cond: invalid */
      CHECK(imd_err_estimate(14, IMD_RN, IMD_DISPATCH_EMULATE, 100, 1, &ar, &er) == IMD_STATUS_INVALID_VALUE);
      ar.cond = 1e6, ar.target = 1e-9;
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
