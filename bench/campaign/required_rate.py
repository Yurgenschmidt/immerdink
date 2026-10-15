#!/usr/bin/env python3
"""The required-rate table (issue 16): how fast must K4, the INT8 matrix-multiply kernel, run for immerdink to match
GEMMul8 fast, if the pack/CRT kernels around it take 1.0, 0.67 or 0.5 times today's time?

    required_rate.py JOBFOLDER [--factors 1,0.67,0.5] [--n 14] [--csv OUT]

JOBFOLDER holds sweep_imd.jsonl (immerdink records with measured phases) and compare.jsonl (the comparison sweep). Per case:

    T*       time of GEMMul8 fast at DGEMM-level accuracy (the parity reference)
    P        pack/CRT time today = the measured phases of the immerdink call at N moduli except K4
    R_today  K4 rate today = K4 operations / K4 phase time, in TOPS (INT8 operations actually executed: 2 N b m n_hat k_hat)
    R*(f)    required K4 rate = K4 operations / (T* - f P), the "required-rate budget" of the tracking issue

R*(f) is "n/a" when f P >= T* (no K4 speed can reach parity with that pack/CRT time). The phase marks add a little overhead,
so P and K4 time are the phase sums, not the whole-call median; the immerdink N is the --n of the sweep (14); where the
DGEMM-level immerdink configuration of the comparison sweep has another N the operations and P are scaled by N / n (flagged).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import summarize as sm  # noqa: E402

CASES = ["s256", "s512", "s_cc16", "s_batch", "sq1024", "sq2048", "sq4096", "sq8192", "r_lowk", "r_bigk", "r_skinny", "r_mm",
         "t_batch", "t_ikl", "t_df", "t_cc", "t_permout"]


def rows(job, factors=(1.0, 0.67, 0.5), n=14):
    sweep = [r for r in sm.load(os.path.join(job, "sweep_imd.jsonl")) if r.get("method") == "imd" and r.get("status") == "ok"]
    comp = sm.load(os.path.join(job, "compare.jsonl"))
    by_case = sm.cases(comp)
    out = []
    for case in CASES:
        row = {"case": case}
        rec = next((r for r in sweep if r["case"] == case and r.get("N") == n and r.get("phases")), None)
        fams = sm.case_families(by_case.get(case, []))
        ref, _, best = sm.dgemm_level(fams, "tflops_eq")
        g8 = best.get("gemmul8-fast")
        imd_best = best.get("imd")
        if rec is None or g8 is None:
            row["note"] = "no phases in sweep_imd.txt (add the case to the R1 sweep)" if rec is None else "no GEMMul8 fast at DGEMM-level accuracy"
            out.append(row)
            continue
        ph = rec["phases"]
        k4_ms = sum(v["ms"] for k, v in ph.items() if k.startswith("K4"))
        p_ms = sum(v["ms"] for k, v in ph.items() if not k.startswith("K4"))
        m, nn, k, b = rec["m"], rec["n"], rec["k"], rec.get("b", 1)
        ops = sm.phases(m, nn, k, b, n, rec.get("k_align"), rec.get("k_hat"))["K4 GEMM"][1]
        scale = 1.0
        nb = imd_best.get("N") if imd_best else None
        if nb and nb != n:
            scale = nb / n
            row["note"] = f"immerdink needs N={nb} at DGEMM-level accuracy: operations and pack/CRT scaled by {nb}/{n}"
        if case == "s_batch":
            row["note"] = "excluded from parity: GEMMul8 makes one call per batch slice"
        ops *= scale
        p_ms *= scale
        k4_ms *= scale
        t_star = g8["median_s"] * 1e3
        row.update(tstar_ms=t_star, p_ms=p_ms, k4_ms=k4_ms, k4_ops=ops, r_today=ops / (k4_ms * 1e-3) / 1e12,
                   t_imd_ms=rec["median_s"] * 1e3, g8_N=g8.get("N"), imd_N=nb)
        for f in factors:
            budget = t_star - f * p_ms
            row[f"r_req_{f:g}"] = ops / (budget * 1e-3) / 1e12 if budget > 0 else None
        out.append(row)
    return out


def markdown(rs, factors):
    head = "| case | T* GEMMul8 fast (ms) | pack/CRT P (ms) | K4 (ms) | K4 today (TOPS) | " + " | ".join(
        f"R* at {f:g} P (TOPS)" for f in factors) + " | R*/today at 0.67 P | note |"
    lines = [head, "|---|---|---|---|---|" + "---|" * len(factors) + "---|---|"]
    for r in rs:
        if "tstar_ms" not in r:
            lines.append(f"| {r['case']} | | | | | " + " | " * len(factors) + f"| | {r.get('note', '')} |")
            continue
        cells = [f"{r[f'r_req_{f:g}']:.3g}" if r[f"r_req_{f:g}"] else "n/a" for f in factors]
        mid = r.get("r_req_0.67")
        ratio = f"{mid / r['r_today']:.2f}" if mid else "n/a"
        lines.append(f"| {r['case']} | {r['tstar_ms']:.3g} | {r['p_ms']:.3g} | {r['k4_ms']:.3g} | {r['r_today']:.0f} | "
                     + " | ".join(cells) + f" | {ratio} | {r.get('note', '')} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job")
    p.add_argument("--factors", default="1,0.67,0.5")
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--csv")
    a = p.parse_args(argv)
    factors = [float(x) for x in a.factors.split(",")]
    rs = rows(a.job, factors, a.n)
    print(markdown(rs, factors))
    if a.csv:
        keys = sorted({k for r in rs for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
