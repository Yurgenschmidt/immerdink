#!/usr/bin/env python3
"""M5 accuracy study, part 1: CC-shaped contractions with stressing data vs DESIGN §3.8.

    tools/memcap.sh python3 bench/accuracy/contractions.py [--quick] [--out FILE]

For each fixture x distribution, the GEMM-view operands are generated, folded back into
the fixture's tensor layouts and contracted by imd_contract for N = 8..20, RN and TRUNC.
Checks per run:
  grid  a random grid of sampled rows x columns of the GEMM view (exact reference,
        exact shifts sigma/tau): max normwise error |D - x| / (||a_i|| ||b_j||), max
        elementwise error |D - x| / |x|, and max / median of |D - x| / bound(i, j).
        A ratio > 1 is a bound violation.
  full  every output: |D - D_native| <= bound + gamma_k sum|a||b| (a necessary condition
        for the bound, D_native = numpy FP64 matmul); the count of failures must be 0.
The native FP64 (numpy/OpenBLAS) error on the same grid is reported for scale.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

FIXTURES = {
    # name: (labA, labB, labC, extents, quick extents)
    "df_acQ": ("acQ", "bdQ", "acbd", dict(a=24, c=24, b=24, d=24, Q=600),
               dict(a=12, c=12, b=12, d=12, Q=200)),
    "df_ijdkQ": ("ijdkQ", "ldQ", "klij", dict(i=8, j=8, k=8, l=8, d=24, Q=150),
                 dict(i=4, j=4, k=4, l=4, d=12, Q=60)),
    "ring_kjbc": ("kjbc", "kiac", "abij", dict(k=12, i=12, j=12, a=48, b=48, c=48),
                  dict(k=6, i=6, j=6, a=20, b=20, c=20)),
    "T_bdck": ("bdck", "adij", "abcijk", dict(a=40, b=40, c=40, d=40, i=5, j=5, k=5),
               dict(a=16, b=16, c=16, d=16, i=3, j=3, k=3)),
}
DISTS = ["uniform", "decay_same", "decay_opp", "outliers", "cancel"]


def gen(dist, m, n, k, rng):
    """GEMM-view operands A (m x k), B (k x n)."""
    if dist == "uniform":
        return rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
    if dist in ("decay_same", "decay_opp"):
        # magnitudes 1e-1 .. 1e-10 along the contracted index (outermost K label varies
        # slowest), plus 3 decades along rows/columns; random signs
        h = np.arange(k) / max(k - 1, 1)
        da = 10.0 ** (-1 - 9 * h)
        db = da if dist == "decay_same" else da[::-1]
        ra = 10.0 ** (-3 * np.arange(m) / max(m - 1, 1))
        rb = 10.0 ** (-3 * np.arange(n) / max(n - 1, 1))
        A = rng.uniform(0.5, 1, (m, k)) * rng.choice([-1, 1], (m, k)) * da[None, :] * ra[:, None]
        B = rng.uniform(0.5, 1, (k, n)) * rng.choice([-1, 1], (k, n)) * db[:, None] * rb[None, :]
        return A, B
    if dist == "outliers":
        A = rng.standard_normal((m, k))
        B = rng.standard_normal((k, n))
        for _ in range(2):   # two outliers per row of A and per column of B, 1e3..1e8
            A[np.arange(m), rng.integers(0, k, m)] *= 10.0 ** rng.uniform(3, 8, m)
            B[rng.integers(0, k, n), np.arange(n)] *= 10.0 ** rng.uniform(3, 8, n)
        return A, B
    if dist == "cancel":
        # pairs h, h + k/2 cancel to relative 1e-6: x ~ 1e-6 sqrt(k), ||a|| ||b|| ~ k/3
        assert k % 2 == 0
        h2 = k // 2
        A = rng.uniform(-1, 1, (m, k))
        B = rng.uniform(-1, 1, (k, n))
        A[:, h2:] = A[:, :h2] * (1 + 1e-6 * rng.uniform(-1, 1, (m, h2)))
        B[h2:, :] = -B[:h2, :]
        return A, B
    raise ValueError(dist)


def run_case(runner, fx, dist, ext, Ns, rnds, nsamp, seed, log):
    labA, labB, labC = FIXTURES[fx][:3]
    V = cm.View(labA, labB, labC, ext)
    m, n, k = V.m, V.n, V.k
    rng = np.random.default_rng(seed)
    Am, Bm = gen(dist, m, n, k, rng)
    A, B = V.a_from_mat(Am), V.b_from_mat(Bm)
    t0 = time.time()

    # exact shifts' N-independent parts for every row / column
    sgA = [cm.s_gamma(Am[i]) for i in range(m)]
    sgB = [cm.s_gamma(Bm[:, j]) for j in range(n)]
    na, nb = cm.norms(Am, 1), cm.norms(Bm, 0)
    absAB = np.abs(Am) @ np.abs(Bm)
    gk = k * cm.U / (1 - k * cm.U)
    Dn = Am @ Bm

    # sampled grid with an exact reference
    ri = np.sort(rng.choice(m, min(m, nsamp), replace=False))
    cj = np.sort(rng.choice(n, min(n, max(nsamp, 4096 // min(m, nsamp))), replace=False))
    hi = np.empty((ri.size, cj.size))
    lo = np.empty_like(hi)
    for a, i in enumerate(ri):
        for b, j in enumerate(cj):
            hi[a, b], lo[a, b] = cm.exact_ref(Am[i], Bm[:, j])
    nab = na[ri][:, None] * nb[cj][None, :]
    xabs = np.abs(hi + lo)
    nz = xabs > 0
    def err_of(Dg):
        return np.abs((Dg - hi) - lo)
    en = err_of(Dn[np.ix_(ri, cj)])
    native = {"normwise": float(np.max(en / nab)),
              "elementwise": float(np.max(en[nz] / xabs[nz])) if nz.any() else 0.0}
    cancel = float(np.median(xabs / nab))
    log(f"# {fx}/{dist}: m={m} n={n} k={k}, grid {ri.size}x{cj.size}, "
        f"median |x|/(|a||b|) = {cancel:.2e}, setup {time.time() - t0:.1f}s; "
        f"native FP64 normwise {native['normwise']:.2e} elementwise {native['elementwise']:.2e}")

    recs = []
    for rnd in rnds:
        for N in Ns:
            st, C = runner.contract(labA, A, labB, B, labC, V.shapeC, N, rnd)
            rec = dict(fixture=fx, dist=dist, m=m, n=n, k=k, N=N, rounding=rnd, status=st,
                       grid=[int(ri.size), int(cj.size)], median_cancel=cancel, native=native)
            if st != "SUCCESS":
                recs.append(rec)
                log(f"  {rnd:5s} N={N:2d}: {st}")
                continue
            D = V.c_to_mat(C)
            plan = cm.o.Plan(N, k, rnd)
            sig = np.array([cm.sigma_of(s, plan) for s in sgA])
            tau = np.array([cm.sigma_of(s, plan) for s in sgB])
            # grid
            Dg = D[np.ix_(ri, cj)]
            e = err_of(Dg)
            bd = cm.bound(N, k, rnd, na[ri][:, None], nb[cj][None, :], sig[ri][:, None],
                          tau[cj][None, :], Dg)
            ratio = e / bd
            # full: necessary condition against native FP64
            bfull = cm.bound(N, k, rnd, na[:, None], nb[None, :], sig[:, None], tau[None, :],
                             D) + gk * absAB
            full_bad = int(np.count_nonzero(np.abs(D - Dn) > bfull * (1 + 1e-12)))
            rec.update(normwise=float(np.max(e / nab)),
                       elementwise=float(np.max(e[nz] / xabs[nz])) if nz.any() else 0.0,
                       ratio_max=float(np.max(ratio)), ratio_median=float(np.median(ratio)),
                       full_violations=full_bad, grid_violations=int(np.sum(ratio > 1)),
                       bound_rel=float(np.max(bd / nab)))
            recs.append(rec)
            flag = "  <-- VIOLATION" if rec["grid_violations"] or full_bad else ""
            log(f"  {rnd:5s} N={N:2d}: normwise {rec['normwise']:.2e} elementwise "
                f"{rec['elementwise']:.2e} obs/bound max {rec['ratio_max']:.2e} median "
                f"{rec['ratio_median']:.2e} full-check fails {full_bad}{flag}")
            if flag:
                a, b = np.unravel_index(np.argmax(ratio), ratio.shape)
                log(f"    reproducer: fixture {fx} dist {dist} seed {seed} row {ri[a]} col "
                    f"{cj[b]} D={Dg[a, b]!r} x={hi[a, b]!r}+{lo[a, b]!r} bound={bd[a, b]!r}")
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=None)
    ap.add_argument("--quick", action="store_true", help="small extents, few N (smoke test)")
    ap.add_argument("--fixtures", nargs="*", default=list(FIXTURES))
    ap.add_argument("--dists", nargs="*", default=DISTS)
    ap.add_argument("--N", nargs="*", type=int, default=None)
    ap.add_argument("--rows", type=int, default=64, help="sampled rows of the GEMM view")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None, help="JSONL output")
    args = ap.parse_args()
    Ns = args.N or ([8, 12, 16, 20] if args.quick else list(range(8, 21)))
    runner = cm.Runner(args.lib)
    out = open(args.out, "w") if args.out else None
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=cm.ROOT).stdout.strip()
    viol = 0

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    for fx in args.fixtures:
        ext = FIXTURES[fx][4] if args.quick else FIXTURES[fx][3]
        for di, dist in enumerate(args.dists):
            recs = run_case(runner, fx, dist, ext, Ns, ["RN", "TRUNC"], args.rows,
                            args.seed + 101 * di, log)
            for r in recs:
                viol += r.get("grid_violations", 0) + r.get("full_violations", 0)
                if out:
                    r["commit"] = commit
                    out.write(json.dumps(r) + "\n")
    runner.close()
    print(f"done in {time.time() - t0:.0f}s; bound violations: {viol}")
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
