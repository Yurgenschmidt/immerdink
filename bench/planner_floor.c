/* The C floor of the planner queries (DESIGN §6e): ns per call of each libimmerdink_host
 * entry point, timed in a tight single-threaded C loop, for comparison with the Python
 * wrappers (bench/planner_overhead.py drives it and prints both).
 *
 *   imd_planner_floor < cases.txt
 *
 * One case per input line (whitespace-separated integers after the name):
 *   name N g vec_mode  rA eA[rA] sA[rA] lA[rA]  rB eB.. sB.. lB..  rC eC.. sC.. lC..
 * One output line per case:
 *   name ws lowering plan_bytes pack_desc_for predict_null predict_calib predict_pack rank
 * in ns per call (median of 7 batches of ~4 ms each). predict_null passes calib = NULL (the
 * shipped calibration, parsed once by the library, v1.10); predict_calib passes a
 * calibration already loaded with imd_perf_calib_default; rank: N = {N}, every g,
 * GENERATE only, no packs, calibration given; predict_pack: imd_pack_desc_for(A)'s
 * descriptor. CPU only, no GPU, no allocation in the loops. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "immerdink/immerdink_host.h"

#define MAXR 16
typedef struct {
  int64_t e[MAXR], s[MAXR];
  int32_t l[MAXR];
  imd_tensor_t t;
} Tensor;

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

static int read_tensor(Tensor* x) {
  int r = 0;
  if (scanf("%d", &r) != 1 || r < 0 || r > MAXR) return 0;
  for (int i = 0; i < r; ++i)
    if (scanf("%lld", (long long*)&x->e[i]) != 1) return 0;
  for (int i = 0; i < r; ++i)
    if (scanf("%lld", (long long*)&x->s[i]) != 1) return 0;
  for (int i = 0; i < r; ++i)
    if (scanf("%d", &x->l[i]) != 1) return 0;
  x->t.rank = r;
  x->t.extent = x->e;
  x->t.stride = x->s;
  x->t.label = x->l;
  return 1;
}

/* The query being timed, as a switch so that every function runs through one loop. */
enum { Q_WS, Q_LOWERING, Q_PLAN_BYTES, Q_PACK_DESC, Q_PREDICT_NULL, Q_PREDICT_CALIB,
       Q_PREDICT_PACK, Q_RANK, Q_COUNT };

typedef struct {
  const imd_tensor_t *A, *B, *C;
  imd_options_t opt;
  imd_perf_calib_t* cal;
  imd_pack_desc_t pdesc;
  int32_t plabels[MAXR];
  imd_perf_rank_request_t rreq;
  int32_t rank_n;
  imd_perf_candidate_t cands[32];
  imd_lowering_t lw;
  imd_perf_t perf;
} Ctx;

static volatile int64_t sink;

static void run(Ctx* c, int q) {
  size_t a = 0, b = 0;
  int32_t n = 0;
  int st = 0;
  switch (q) {
    case Q_WS: st = imd_workspace_size(c->A, c->B, c->C, &c->opt, &a); break;
    case Q_LOWERING:
      c->lw.struct_size = sizeof c->lw;
      c->lw.sm_count = 0;
      st = imd_lowering(c->A, c->B, c->C, &c->opt, &c->lw);
      break;
    case Q_PLAN_BYTES: st = imd_plan_bytes(c->A, c->B, c->C, &c->opt, &a, &b); break;
    case Q_PACK_DESC: {
      imd_pack_desc_t d;
      memset(&d, 0, sizeof d);
      d.struct_size = sizeof d;
      st = imd_pack_desc_for(c->A, c->B, c->C, &c->opt, 0, &d, c->plabels, MAXR);
      break;
    }
    case Q_PREDICT_NULL:
      c->perf.struct_size = sizeof c->perf;
      st = imd_perf_predict(c->A, c->B, c->C, &c->opt, NULL, NULL, NULL, &c->perf);
      break;
    case Q_PREDICT_CALIB:
      c->perf.struct_size = sizeof c->perf;
      st = imd_perf_predict(c->A, c->B, c->C, &c->opt, NULL, NULL, c->cal, &c->perf);
      break;
    case Q_PREDICT_PACK:
      c->perf.struct_size = sizeof c->perf;
      st = imd_perf_predict_pack(&c->pdesc, NULL, c->cal, &c->perf);
      break;
    case Q_RANK:
      st = imd_perf_rank(c->A, c->B, c->C, &c->opt, &c->rreq, c->cal, c->cands, 32, &n);
      break;
  }
  sink += st + (int64_t)a + n;
}

static int cmp(const void* x, const void* y) {
  const double a = *(const double*)x, b = *(const double*)y;
  return a < b ? -1 : a > b;
}

static double time_query(Ctx* c, int q) {
  long iters = 1;
  for (;;) {  /* calibrate a batch of ~4 ms */
    const double t0 = now_ns();
    for (long i = 0; i < iters; ++i) run(c, q);
    const double dt = now_ns() - t0;
    if (dt > 4e6 || iters > (1L << 26)) break;
    iters *= dt < 1e5 ? 16 : 2;
  }
  double med[7];
  for (int r = 0; r < 7; ++r) {
    const double t0 = now_ns();
    for (long i = 0; i < iters; ++i) run(c, q);
    med[r] = (now_ns() - t0) / (double)iters;
  }
  qsort(med, 7, sizeof med[0], cmp);
  return med[3];
}

int main(void) {
  static imd_perf_calib_t cal;
  cal.struct_size = sizeof cal;
  if (imd_perf_calib_default(&cal) != IMD_STATUS_SUCCESS) {
    fprintf(stderr, "imd_perf_calib_default failed\n");
    return 1;
  }
  char name[128];
  while (scanf("%127s", name) == 1) {
    int N = 0, g = 0, vec = 0;
    static Tensor A, B, C;
    if (scanf("%d %d %d", &N, &g, &vec) != 3 || !read_tensor(&A) || !read_tensor(&B) ||
        !read_tensor(&C)) {
      fprintf(stderr, "bad case line at %s\n", name);
      return 2;
    }
    static Ctx c;
    memset(&c, 0, sizeof c);
    c.A = &A.t, c.B = &B.t, c.C = &C.t;
    c.opt = (imd_options_t){N, g, IMD_RN, IMD_EMULATE, vec, IMD_K4_TUNE_OFF};
    c.cal = &cal;
    c.pdesc.struct_size = sizeof c.pdesc;
    const int pst = imd_pack_desc_for(c.A, c.B, c.C, &c.opt, 0, &c.pdesc, c.plabels, MAXR);
    c.rank_n = N;
    c.rreq = (imd_perf_rank_request_t){sizeof c.rreq, sizeof c.cands[0], 1, 0, &c.rank_n, NULL,
                                       IMD_PERF_VEC_GENERATE, IMD_PERF_PACK_NONE, 1, 0, 0, -1.0,
                                       0, 0, 0, 0, 0.0};  /* v7: DEFAULT, COLD, ORDER_BAND */
    printf("%s", name);
    for (int q = 0; q < Q_COUNT; ++q) {
      if (q == Q_PREDICT_PACK && pst != IMD_STATUS_SUCCESS) {
        printf(" nan");
        continue;
      }
      printf(" %.1f", time_query(&c, q));
    }
    printf("\n");
    fflush(stdout);
  }
  return 0;
}
