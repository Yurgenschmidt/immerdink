#!/usr/bin/env python3
"""DESIGN v1.12 §6f on the GPU: the measured error against libimmerdink_host's error estimates.

    # phase 1 (GPU, ~1 min): contract every case, keep the sampled output grids
    QUIET_GATE=1 tools/run_quiet.sh python3 bench/accuracy/err_api.py --phase gpu --out DIR
    # phase 2 (CPU only): exact references, statistics, the C estimates, the report
    tools/memcap.sh python3 bench/accuracy/err_api.py --phase check --out DIR
    # both in one process (ctest accuracy_err_api_quick, under the GPU lock of tools/test.sh)
    python3 bench/accuracy/err_api.py --quick

Random cases: data families (phi = (u - 1/2) exp(phi g) for phi = 0.5..4, rows and columns
scaled by 2^U(-300, 300), uniform, same sign, M5's decay_opp and outliers, and the structured
"constant" rows the typical model excludes), k = 64..10^5 (log-uniform), N = 5..20, RN and
TRUNC, the GEMM `ik,kj->ij` and a two-label contraction `akp,bkp->ab`. For each case, on a
sampled grid of outputs with an essentially exact reference (common.exact_ref):

  rigorous  the measured NORMWISE / ELEMENTWISE / ABSOLUTE errors are <= imd_err_estimate's
            bound with the caller's statistics computed from the data (l1 ratios unknown and
            given; rho and norm_product from the norms; statistics inflated by 1e-9 for their
            own float rounding). Any violation fails the run (exit 1).
  typical   coverage of the typical estimate (Hoeffding, union bound over the grid's outputs)
            per family: the fraction of cases whose grid maximum is <= the estimate, and the
            max / median of measured / estimate. Not a pass/fail criterion (the "constant"
            family is expected to exceed it).

The bound math is the C library's (host.err_estimate); this script only measures.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402

sys.path.insert(0, os.path.join(cm.ROOT, "python"))
from immerdink import host  # noqa: E402

FAMILIES = ["phi0.5", "phi1", "phi2", "phi4", "scaled", "uniform", "same_sign", "decay_opp",
            "outliers", "constant"]
SPECS = {"gemm": ("ik", "kj", "ij"), "two_k": ("akp", "bkp", "ab")}
STAT_INFLATE = 1 + 1e-9


def make_cases(n, seed, quick):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)] if quick else rng.choice(FAMILIES)
        k = int(round(10 ** rng.uniform(math.log10(64), 3.0 if quick else 5.0)))
        k -= k % 4                                   # two_k: p = 4
        mn = int(min(512, max(16, 4e6 / k))) if not quick else 24
        spec = rng.choice(list(SPECS))
        rnd = "RN" if rng.random() < 0.8 else "TRUNC"
        N = rng.randint(5, 13) if rng.random() < 0.7 else rng.randint(14, 20)
        g = 16 if k > 20000 else 32
        out.append(dict(case=i, family=fam, k=k, m=mn, n=mn, spec=spec, rounding=rnd, N=N,
                        grid=min(g, mn), seed=seed * 100003 + i))
    return out


def gen(case):
    """GEMM-view operands A (m x k), B (k x n) of a case (deterministic in case['seed'])."""
    rng = np.random.default_rng(case["seed"])
    m, n, k, fam = case["m"], case["n"], case["k"], case["family"]
    if fam.startswith("phi") or fam == "scaled":
        phi = float(fam[3:]) if fam.startswith("phi") else 1.0
        A = (rng.random((m, k)) - 0.5) * np.exp(phi * rng.standard_normal((m, k)))
        B = (rng.random((k, n)) - 0.5) * np.exp(phi * rng.standard_normal((k, n)))
        if fam == "scaled":
            A *= np.exp2(rng.integers(-300, 300, m))[:, None]
            B *= np.exp2(rng.integers(-300, 300, n))[None, :]
        return A, B
    if fam == "uniform":
        return rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
    if fam == "same_sign":
        return rng.uniform(0.5, 1, (m, k)), rng.uniform(0.5, 1, (k, n))
    if fam == "constant":
        # every row of A (column of B) one value that is not representable at its scale: the
        # quantization errors are equal and add coherently (outside the typical model)
        a = 0.1 * (1 + rng.random(m))
        b = 0.3 * (1 + rng.random(n))
        return np.repeat(a[:, None], k, 1), np.repeat(b[None, :], k, 0)
    from contractions import gen as m5gen   # decay_opp, outliers (M5's generators)
    return m5gen(fam, m, n, k, rng)


def to_tensors(case, Am, Bm):
    """(labA, A, labB, B, labC, extC) of the case's spec from the GEMM-view matrices."""
    m, n, k = case["m"], case["n"], case["k"]
    if case["spec"] == "gemm":
        return "ik", Am, "kj", Bm, "ij", (m, n)
    p = 4
    A = np.ascontiguousarray(Am.reshape(m, k // p, p))            # a, k, p
    B = np.ascontiguousarray(Bm.T.reshape(n, k // p, p))          # b, k, p
    return "akp", A, "bkp", B, "ab", (m, n)


def grid_of(case):
    rng = np.random.default_rng(case["seed"] + 7)
    g = case["grid"]
    return (np.sort(rng.choice(case["m"], g, replace=False)),
            np.sort(rng.choice(case["n"], g, replace=False)))


def phase_gpu(cases, lib, outdir, log):
    runner = cm.Runner(lib)
    t0 = time.time()
    grids = {}
    try:
        for c in cases:
            Am, Bm = gen(c)
            la, A, lb, B, lc, ext = to_tensors(c, Am, Bm)
            st, D = runner.contract(la, A, lb, B, lc, ext, c["N"], c["rounding"])
            ri, cj = grid_of(c)
            grids[c["case"]] = (st, None if D is None else D[np.ix_(ri, cj)])
    finally:
        runner.close()
    log(f"# GPU phase: {len(cases)} contractions in {time.time() - t0:.1f} s")
    if outdir:
        np.savez_compressed(os.path.join(outdir, "grids.npz"),
                            **{f"c{i}": g[1] for i, g in grids.items() if g[1] is not None})
        with open(os.path.join(outdir, "status.json"), "w") as f:
            json.dump({str(i): g[0] for i, g in grids.items()}, f)
    return grids


def estimate(c, metric, outputs, **stats):
    return host.err_estimate(c["N"], c["k"], rounding=c["rounding"], dispatch="EMULATE",
                             outputs=outputs, accuracy=host.Accuracy(metric=metric, **stats))


def check_case(c, st, Dg, log):
    rec = dict(c, status=st)
    if st != "SUCCESS":
        return rec
    Am, Bm = gen(c)
    ri, cj = grid_of(c)
    na, nb = cm.norms(Am, 1), cm.norms(Bm, 0)
    l1a, l1b = np.sum(np.abs(Am), 1), np.sum(np.abs(Bm), 0)
    nu_a = float(np.max(np.where(na > 0, l1a / np.where(na > 0, na, 1), 1.0))) * STAT_INFLATE
    nu_b = float(np.max(np.where(nb > 0, l1b / np.where(nb > 0, nb, 1), 1.0))) * STAT_INFLATE
    hi = np.empty(Dg.shape)
    lo = np.empty(Dg.shape)
    for a, i in enumerate(ri):
        for b, j in enumerate(cj):
            hi[a, b], lo[a, b] = cm.exact_ref(Am[i], Bm[:, j])
    err = np.abs((Dg - hi) - lo)
    nab = na[ri][:, None] * nb[cj][None, :]
    xabs = np.abs(hi + lo)
    nz = (xabs > 0) & (nab > 0)
    normwise = float(np.max(np.where(nab > 0, err / np.where(nab > 0, nab, 1), 0)))
    absolute = float(np.max(err))
    npd = float(np.max(na) * np.max(nb)) * STAT_INFLATE
    rho = float(np.max(nab[nz] / xabs[nz])) * STAT_INFLATE if nz.any() else math.inf
    elementwise = float(np.max(err[nz] / xabs[nz])) if nz.any() else 0.0
    M = c["m"] * c["n"]
    Mg = Dg.size
    b_n = estimate(c, "NORMWISE", M)
    b_nu = estimate(c, "NORMWISE", M, l1_ratio_a=max(nu_a, 1.0), l1_ratio_b=max(nu_b, 1.0))
    b_a = estimate(c, "ABSOLUTE", M, norm_product=max(npd, 1e-300))
    viol = []
    tol = 1 + 1e-9   # the measured side's float norms
    if normwise > b_n.bound * tol:
        viol.append(("NORMWISE", normwise, b_n.bound))
    if normwise > b_nu.bound * tol:
        viol.append(("NORMWISE l1", normwise, b_nu.bound))
    if absolute > b_a.bound * tol and npd > 0:
        viol.append(("ABSOLUTE", absolute, b_a.bound))
    rec.update(normwise=normwise, absolute=absolute, elementwise=elementwise, rho=rho,
               nu_a=nu_a, nu_b=nu_b, sqrt_k=math.sqrt(c["k"]), bound_normwise=b_n.bound,
               bound_normwise_l1=b_nu.bound, bound_absolute=b_a.bound,
               ratio_bound=normwise / b_n.bound, ratio_bound_l1=normwise / b_nu.bound)
    if math.isfinite(rho):
        b_e = estimate(c, "ELEMENTWISE", M, cond=max(rho, 1.0))
        rec.update(bound_elementwise=b_e.bound)
        if elementwise > b_e.bound * tol:
            viol.append(("ELEMENTWISE", elementwise, b_e.bound))
    # typical: the union bound over the grid's outputs (the maximum measured is over the grid)
    t_n = estimate(c, "NORMWISE", Mg)
    rec.update(typical_normwise_grid=t_n.typical, ratio_typical=normwise / t_n.typical,
               typical_covered=bool(normwise <= t_n.typical),
               unit_bound=b_n.unit_bound, unit_typical_grid=t_n.unit_typical)
    rec["violations"] = viol
    if viol:
        log(f"  VIOLATION case {c['case']} {c['family']} N={c['N']} {c['rounding']} k={c['k']}: "
            f"{viol}")
    return rec


def report(recs, log):
    ok = [r for r in recs if r.get("status") == "SUCCESS"]
    nviol = sum(len(r["violations"]) for r in ok)
    log(f"\n{len(recs)} cases, {len(ok)} contracted; rigorous-bound violations: {nviol}")
    log(f"max measured / bound: NORMWISE {max(r['ratio_bound'] for r in ok):.3g}, with l1 "
        f"ratios {max(r['ratio_bound_l1'] for r in ok):.3g}")
    log("\ntypical estimate (RN; emulation regime = unit typical > 8u):")
    log(f"{'family':>10} {'cases':>5} {'covered':>8} {'max meas/typ':>13} {'median':>8} "
        f"{'emu cases':>9} {'emu covered':>11} {'emu max':>8}")
    for fam in FAMILIES:
        rs = [r for r in ok if r["family"] == fam and r["rounding"] == "RN"]
        if not rs:
            continue
        emu = [r for r in rs if r["unit_typical_grid"] > 8 * 2.0**-53]
        rt = sorted(r["ratio_typical"] for r in rs)
        cov = sum(r["typical_covered"] for r in rs)
        ecov = sum(r["typical_covered"] for r in emu)
        emax = max((r["ratio_typical"] for r in emu), default=float("nan"))
        log(f"{fam:>10} {len(rs):>5} {cov:>4}/{len(rs):<3} {rt[-1]:>13.3g} "
            f"{rt[len(rt) // 2]:>8.3g} {len(emu):>9} {ecov:>7}/{len(emu):<3} {emax:>8.3g}")
    return nviol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("gpu", "check", "both"), default="both")
    ap.add_argument("--lib", default=None, help="libimmerdink.so (GPU phase)")
    ap.add_argument("--host-lib", default=None, help="libimmerdink_host.so (check phase)")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="small cases (ctest)")
    ap.add_argument("--out", default=None, help="directory for grids.npz / records.jsonl")
    a = ap.parse_args()
    if a.host_lib:
        host.load(a.host_lib)
        host._loaded[None] = host._loaded[a.host_lib]
    n = 20 if a.quick and a.cases == 300 else a.cases
    cases = make_cases(n, a.seed, a.quick)

    def log(s):
        print(s, flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=cm.ROOT).stdout.strip()
    if a.phase in ("gpu", "both"):
        grids = phase_gpu(cases, a.lib, a.out, log)
        if a.phase == "gpu":
            return 0
    else:
        z = np.load(os.path.join(a.out, "grids.npz"))
        sts = json.load(open(os.path.join(a.out, "status.json")))
        grids = {c["case"]: (sts[str(c["case"])], z[f"c{c['case']}"]
                             if f"c{c['case']}" in z else None) for c in cases}
    t0 = time.time()
    recs = []
    for c in cases:
        st, Dg = grids[c["case"]]
        recs.append(check_case(c, st, Dg, log))
    log(f"# check phase: {time.time() - t0:.0f} s")
    if a.out:
        with open(os.path.join(a.out, "records.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(dict(r, commit=commit)) + "\n")
    return 1 if report(recs, log) else 0


if __name__ == "__main__":
    sys.exit(main())
