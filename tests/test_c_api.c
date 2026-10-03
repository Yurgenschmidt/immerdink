/* The public header is valid C (C11) and the C ABI links and runs from C:
 * imd_workspace_size on the DESIGN §4a worked example (17 x 33 x 5, b = 3, N = 9, g = 2),
 * the plan's RN acceptance rule, the contract's argument checks in DESIGN §6 order, and one
 * contraction on device buffers. */
#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>

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
  const imd_options_t o = {2, 0, rounding, IMD_EMULATE, IMD_VEC_GENERATE};
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
  const imd_options_t opt = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE};
  size_t bytes = 0;

  expect("workspace_size", imd_workspace_size(&A, &B, &C, &opt, &bytes), IMD_STATUS_SUCCESS);
  if (bytes != 30720) { /* spec/params.py cert_workspace, hand-computed */
    fprintf(stderr, "FAIL workspace bytes %zu, expected 30720\n", bytes);
    ++failures;
  }

  imd_handle_t h = NULL;
  imd_plan_t plan = NULL;
  expect("create", imd_create(&h), IMD_STATUS_SUCCESS);
  {
    /* DESIGN v1.2 vec_mode: RESIDENT drops the vec region (30720 - 1024 = 29696, params.py
     * cert_workspace) and the plan owns exactly 8 (34 + 66 + 10 + 9) = 952 device bytes. */
    const imd_options_t res = {9, 2, IMD_RN, IMD_EMULATE, IMD_VEC_RESIDENT};
    const imd_options_t bad = {9, 2, IMD_RN, IMD_EMULATE, 2};
    size_t dev = 1;
    imd_plan_t pr = NULL;
    expect("workspace_size resident", imd_workspace_size(&A, &B, &C, &res, &bytes),
           IMD_STATUS_SUCCESS);
    if (bytes != 29696) {
      fprintf(stderr, "FAIL resident workspace bytes %zu, expected 29696\n", bytes);
      ++failures;
    }
    expect("workspace_size vec_mode 2", imd_workspace_size(&A, &B, &C, &bad, &bytes),
           IMD_STATUS_INVALID_VALUE);
    expect("plan_create vec_mode 2", imd_plan_create(h, &A, &B, &C, &bad, &pr),
           IMD_STATUS_INVALID_VALUE);
    expect("plan_create resident", imd_plan_create(h, &A, &B, &C, &res, &pr), IMD_STATUS_SUCCESS);
    expect("plan_device_bytes", imd_plan_device_bytes(pr, &dev), IMD_STATUS_SUCCESS);
    if (dev != 952) {
      fprintf(stderr, "FAIL resident plan device bytes %zu, expected 952\n", dev);
      ++failures;
    }
    expect("plan_device_bytes null", imd_plan_device_bytes(pr, NULL), IMD_STATUS_INVALID_VALUE);
    expect("plan_device_bytes null plan", imd_plan_device_bytes(NULL, &dev),
           IMD_STATUS_INVALID_VALUE);
    imd_plan_destroy(pr);
    pr = NULL;
    expect("plan_create generate", imd_plan_create(h, &A, &B, &C, &opt, &pr), IMD_STATUS_SUCCESS);
    expect("plan_device_bytes generate", imd_plan_device_bytes(pr, &dev), IMD_STATUS_SUCCESS);
    if (dev != 0) {
      fprintf(stderr, "FAIL generate plan device bytes %zu, expected 0\n", dev);
      ++failures;
    }
    imd_plan_destroy(pr);
  }
  expect("plan_create null handle", imd_plan_create(NULL, &A, &B, &C, &opt, &plan),
         IMD_STATUS_INVALID_VALUE);
  expect("plan_create", imd_plan_create(h, &A, &B, &C, &opt, &plan), IMD_STATUS_SUCCESS);
  expect("contract null plan", imd_contract(NULL, 1.0, NULL, NULL, 0.0, NULL, NULL, 0, 0),
         IMD_STATUS_INVALID_VALUE);
  expect("contract null C", imd_contract(plan, 1.0, NULL, NULL, 0.0, NULL, NULL, 0, 0),
         IMD_STATUS_INVALID_VALUE);
  {
    /* Device buffers: a full contraction runs (bit-exactness is tests/test_contract.py). */
    double *a = NULL, *b = NULL, *c = NULL;
    unsigned char* ws = NULL;
    if (cudaMalloc((void**)&a, sizeof(double) * 17 * 5 * 3) != cudaSuccess ||
        cudaMalloc((void**)&b, sizeof(double) * 5 * 33 * 3) != cudaSuccess ||
        cudaMalloc((void**)&c, sizeof(double) * 17 * 33 * 3) != cudaSuccess ||
        cudaMalloc((void**)&ws, 30720 + 256) != cudaSuccess) {
      fprintf(stderr, "FAIL cudaMalloc\n");
      return 1;
    }
    cudaMemset(a, 0, sizeof(double) * 17 * 5 * 3);
    cudaMemset(b, 0, sizeof(double) * 5 * 33 * 3);
    cudaMemset(c, 0, sizeof(double) * 17 * 33 * 3);
    expect("contract null A", imd_contract(plan, 1.0, NULL, b, 0.0, c, ws, 30720, 0),
           IMD_STATUS_INVALID_VALUE);
    expect("contract null ws", imd_contract(plan, 1.0, a, b, 0.0, c, NULL, 30720, 0),
           IMD_STATUS_INVALID_VALUE);
    expect("contract misaligned C",
           imd_contract(plan, 1.0, a, b, 0.0, (double*)((char*)c + 4), ws, 30720, 0),
           IMD_STATUS_MISALIGNED);
    expect("contract misaligned ws", imd_contract(plan, 1.0, a, b, 0.0, c, ws + 8, 30720, 0),
           IMD_STATUS_MISALIGNED);
    expect("contract small ws", imd_contract(plan, 1.0, a, b, 0.0, c, ws, 30719, 0),
           IMD_STATUS_INSUFFICIENT_WORKSPACE);
    expect("contract alpha=0 does not need A/B",
           imd_contract(plan, 0.0, NULL, NULL, 0.0, c, ws, 30720, 0), IMD_STATUS_SUCCESS);
    expect("contract", imd_contract(plan, 1.0, a, b, 0.5, c, ws, 30720, 0), IMD_STATUS_SUCCESS);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "FAIL contract kernels\n");
      ++failures;
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(ws);
  }
  expect("plan_destroy", imd_plan_destroy(plan), IMD_STATUS_SUCCESS);

  /* RN acceptance (DESIGN §3.2): reject kappa = ceil(sqrt k) >= T; N = 2 has T = 180. */
  expect("RN kappa = T - 1", plan_n2(h, 179 * 179, IMD_RN), IMD_STATUS_SUCCESS);
  expect("RN kappa = T", plan_n2(h, 179 * 179 + 1, IMD_RN), IMD_STATUS_NOT_SUPPORTED);
  expect("TRUNC kappa = T", plan_n2(h, 179 * 179 + 1, IMD_TRUNC), IMD_STATUS_SUCCESS);

  {
    /* Empty output (m = 0): SUCCESS with nothing touched, but C must still be non-null. */
    const int64_t e0[2] = {0, 5}, s0[2] = {5, 1}, ec0[2] = {0, 33}, sc0[2] = {33, 1};
    const int32_t l0[2] = {1, 3}, lc0[2] = {1, 2};
    const imd_tensor_t A0 = {2, e0, s0, l0}, C0 = {2, ec0, sc0, lc0};
    const int64_t eb0[2] = {5, 33}, sb0[2] = {33, 1};
    const int32_t lb0[2] = {3, 2};
    const imd_tensor_t B0 = {2, eb0, sb0, lb0};
    imd_plan_t p0 = NULL;
    double dummy = 0.0;
    expect("plan empty", imd_plan_create(h, &A0, &B0, &C0, &opt, &p0), IMD_STATUS_SUCCESS);
    expect("contract empty, null C", imd_contract(p0, 1.0, &dummy, &dummy, 0.0, NULL, NULL, 0, 0),
           IMD_STATUS_INVALID_VALUE);
    expect("contract empty", imd_contract(p0, 1.0, &dummy, &dummy, 0.0, &dummy, NULL, 0, 0),
           IMD_STATUS_SUCCESS);
    imd_plan_destroy(p0);
  }
  if (strcmp(imd_status_string(IMD_STATUS_ALLOC_FAILED), "IMD_STATUS_ALLOC_FAILED") != 0) {
    fprintf(stderr, "FAIL status string\n");
    ++failures;
  }

  expect("destroy", imd_destroy(h), IMD_STATUS_SUCCESS);
  printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
