/* The host query ABI (DESIGN §6a) from C, linked against libimmerdink_host only: no CUDA
 * header, library or device. The DESIGN §4a worked example (17 x 33 x 5, b = 3, N = 9,
 * g = 2; spec/params.py cert_workspace), the region table, plan bytes, the lowering summary,
 * the RN acceptance rule and the ABI-versioning rules. */
#include <stdio.h>
#include <string.h>

#include "immerdink/immerdink_host.h"

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
  CHECK(sizeof(imd_lowering_t) == 176 && IMD_HOST_API_VERSION >= 4);
  CHECK(sizeof(imd_options_t) == 24);

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
  printf("host C API: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
