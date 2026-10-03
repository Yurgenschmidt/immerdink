/* The public header is valid C (C11) and the C ABI links and runs from C:
 * imd_workspace_size on the DESIGN §4a worked example (17 x 33 x 5, b = 3, N = 9, g = 2),
 * the plan's RN acceptance rule, and the contract skeleton's argument checks (no kernels
 * yet: M1). */
#include <stdio.h>

#include "immerdink/immerdink.h"

static int failures = 0;

static void expect(const char* what, imd_status_t got, imd_status_t want) {
  if (got != want) {
    fprintf(stderr, "FAIL %s: %s, expected %s\n", what, imd_status_string(got),
            imd_status_string(want));
    ++failures;
  }
}

/* C[i,j] = sum_k A[i,k] B[k,j] for N = 2 (T = 180) with the given k. */
static imd_status_t plan_n2(imd_handle_t h, int64_t k, int32_t rounding) {
  const int64_t ea[2] = {4, k}, sa[2] = {k, 1}, eb[2] = {k, 16}, sb[2] = {1, k};
  const int64_t ec[2] = {4, 16}, sc[2] = {16, 1};
  const int32_t la[2] = {1, 3}, lb[2] = {3, 2}, lc[2] = {1, 2};
  const imd_tensor_t A = {2, ea, sa, la}, B = {2, eb, sb, lb}, C = {2, ec, sc, lc};
  const imd_options_t o = {2, 0, rounding, IMD_EMULATE};
  imd_plan_t plan = NULL;
  const imd_status_t st = imd_plan_create(h, &A, &B, &C, &o, &plan);
  imd_plan_destroy(plan);
  return st;
}

int main(void) {
  /* C[i,j,l] = sum_k A[i,k,l] B[k,j,l]; labels i=1 j=2 k=3 l=4; dense, last mode fastest */
  const int64_t eA[3] = {17, 5, 3}, sA[3] = {15, 3, 1};
  const int64_t eB[3] = {5, 33, 3}, sB[3] = {99, 3, 1};
  const int64_t eC[3] = {17, 33, 3}, sC[3] = {99, 3, 1};
  const int32_t lA[3] = {1, 3, 4}, lB[3] = {3, 2, 4}, lC[3] = {1, 2, 4};
  const imd_tensor_t A = {3, eA, sA, lA}, B = {3, eB, sB, lB}, C = {3, eC, sC, lC};
  const imd_options_t opt = {9, 2, IMD_RN, IMD_EMULATE};
  size_t bytes = 0;

  expect("workspace_size", imd_workspace_size(&A, &B, &C, &opt, &bytes), IMD_STATUS_SUCCESS);
  if (bytes != 30720) { /* spec/params.py cert_workspace, hand-computed */
    fprintf(stderr, "FAIL workspace bytes %zu, expected 30720\n", bytes);
    ++failures;
  }

  imd_handle_t h = NULL;
  imd_plan_t plan = NULL;
  expect("create", imd_create(&h), IMD_STATUS_SUCCESS);
  expect("plan_create null handle", imd_plan_create(NULL, &A, &B, &C, &opt, &plan),
         IMD_STATUS_INVALID_VALUE);
  expect("plan_create", imd_plan_create(h, &A, &B, &C, &opt, &plan), IMD_STATUS_SUCCESS);
  expect("contract null plan", imd_contract(NULL, 1.0, NULL, NULL, 0.0, NULL, NULL, 0, 0),
         IMD_STATUS_INVALID_VALUE);
  expect("contract null C", imd_contract(plan, 1.0, NULL, NULL, 0.0, NULL, NULL, 0, 0),
         IMD_STATUS_INVALID_VALUE);
  {
    /* Host buffers stand in for device pointers: the argument checks never dereference. */
    static double a[17 * 5 * 3], b[5 * 33 * 3], c[17 * 33 * 3];
    static _Alignas(256) unsigned char ws[30720 + 256];
    expect("contract misaligned C",
           imd_contract(plan, 1.0, a, b, 0.0, (double*)((char*)c + 4), ws, 30720, 0),
           IMD_STATUS_MISALIGNED);
    expect("contract misaligned ws", imd_contract(plan, 1.0, a, b, 0.0, c, ws + 8, 30720, 0),
           IMD_STATUS_MISALIGNED);
    expect("contract small ws", imd_contract(plan, 1.0, a, b, 0.0, c, ws, 30719, 0),
           IMD_STATUS_INSUFFICIENT_WORKSPACE);
    expect("contract alpha=0 does not need A/B",
           imd_contract(plan, 0.0, NULL, NULL, 0.0, c, ws, 30720, 0), IMD_STATUS_NOT_SUPPORTED);
    expect("contract (kernels are M1)", imd_contract(plan, 1.0, a, b, 0.0, c, ws, 30720, 0),
           IMD_STATUS_NOT_SUPPORTED);
  }
  expect("plan_destroy", imd_plan_destroy(plan), IMD_STATUS_SUCCESS);

  /* RN acceptance (DESIGN §3.2): reject kappa = ceil(sqrt k) >= T; N = 2 has T = 180. */
  expect("RN kappa = T - 1", plan_n2(h, 179 * 179, IMD_RN), IMD_STATUS_SUCCESS);
  expect("RN kappa = T", plan_n2(h, 179 * 179 + 1, IMD_RN), IMD_STATUS_NOT_SUPPORTED);
  expect("TRUNC kappa = T", plan_n2(h, 179 * 179 + 1, IMD_TRUNC), IMD_STATUS_SUCCESS);

  expect("destroy", imd_destroy(h), IMD_STATUS_SUCCESS);
  printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
