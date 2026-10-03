#!/usr/bin/env python3
"""Markdown tables for docs/ACCURACY.md from bench/accuracy/results/contractions.jsonl.

    python3 bench/accuracy/summarize.py [contractions.jsonl]
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402,F401  (puts spec/ on sys.path)
import params  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DISTS = ["uniform", "decay_same", "decay_opp", "outliers", "cancel"]


def load(path):
    return [json.loads(line) for line in open(path)]


def table(recs, key, agg, Ns, fmt, rnds=("RN", "TRUNC"), extra=None):
    cell = defaultdict(list)
    for r in recs:
        if r["status"] == "SUCCESS":
            cell[(r["dist"], r["rounding"], r["N"])].append(r[key] if key else r)
    head = "| data | rnd | " + " | ".join(f"N={N}" for N in Ns) + " |"
    lines = [head, "|" + "---|" * (len(Ns) + 2)]
    for d in DISTS:
        for rnd in rnds:
            vals = [agg(cell[(d, rnd, N)]) if cell[(d, rnd, N)] else None for N in Ns]
            lines.append(f"| {d} | {rnd} | " +
                         " | ".join("—" if v is None else fmt(v) for v in vals) + " |")
    if extra:
        lines += extra
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results",
                                                             "contractions.jsonl")
    recs = load(path)
    Ns = sorted({r["N"] for r in recs})
    viol = sum(r.get("grid_violations", 0) + r.get("full_violations", 0) for r in recs)
    nruns = sum(r["status"] == "SUCCESS" for r in recs)
    grid = sum(r["grid"][0] * r["grid"][1] for r in recs if r["status"] == "SUCCESS")
    full = sum(r["m"] * r["n"] for r in recs if r["status"] == "SUCCESS")
    print(f"runs {nruns}, grid elements checked exactly {grid}, full-check elements {full}, "
          f"violations {viol}\n")

    print("### Max observed / bound (over the 4 fixtures, sampled grid)\n")
    print(table(recs, "ratio_max", max, Ns, lambda v: f"{v:.2f}"))
    print("\n### Median observed / bound (median over the 4 fixtures of the per-run median)\n")
    print(table(recs, "ratio_median", lambda v: sorted(v)[len(v) // 2], Ns,
                lambda v: f"{v:.0e}"))
    kmax = max(r["k"] for r in recs)
    g = [params.error_guidance(N, kmax, params.RN)["term1"] for N in Ns]
    extra = ["| *§3.8 RN term 1, k=%d* | | " % kmax + " | ".join(f"*{v:.0e}*" for v in g) + " |"]
    print("\n### Max normwise error |D − x| / (‖a‖‖b‖) (over the 4 fixtures)\n")
    print(table(recs, "normwise", max, Ns, lambda v: f"{v:.0e}", extra=extra))
    print("\n### Max elementwise error |D − x| / |x| (over the 4 fixtures)\n")
    print(table(recs, "elementwise", max, Ns, lambda v: f"{v:.0e}"))

    print("\n### Native FP64 (numpy/OpenBLAS) on the same grids, max over fixtures\n")
    print("| data | normwise | elementwise | median |x|/(‖a‖‖b‖) |\n|---|---|---|---|")
    for d in DISTS:
        rs = [r for r in recs if r["dist"] == d]
        print(f"| {d} | {max(r['native']['normwise'] for r in rs):.0e} | "
              f"{max(r['native']['elementwise'] for r in rs):.0e} | "
              f"{min(r['median_cancel'] for r in rs):.0e} – "
              f"{max(r['median_cancel'] for r in rs):.0e} |")

    print("\n### Smallest N for a normwise target (RN): guaranteed by §3.8 vs observed here\n")
    targets = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-15, 2**-53]
    ks = [1000, 10**4, 2**17]
    print("| target ε (×‖a‖‖b‖) | " + " | ".join(f"§3.8, k={k}" for k in ks) +
          " | observed max (all data, k ≤ %d) |\n|" % kmax + "---|" * (len(ks) + 2))
    for t in targets:
        def guaranteed(k):
            for N in range(params.N_MIN, params.N_MAX + 1):
                gd = params.error_guidance(N, k, params.RN)
                if gd["rn_feasible"] and gd["term1"] + gd["term2"] <= t:
                    return str(N)
            return ">20"
        obs = ">20"
        for N in Ns:
            if max(r["normwise"] for r in recs if r["status"] == "SUCCESS" and r["N"] == N and
                   r["rounding"] == "RN") <= t:
                obs = ("≤%d" % N) if N == Ns[0] else str(N)
                break
        lab = "u = 2^-53" if t == 2**-53 else f"{t:.0e}"
        print(f"| {lab} | " + " | ".join(guaranteed(k) for k in ks) + f" | {obs} |")

    print("\n### Per fixture, RN: max observed/bound and max normwise error at N = 8, 12, 16\n")
    fx = sorted({r["fixture"] for r in recs}, key=lambda f: [r["fixture"] for r in recs].index(f))
    print("| fixture | (m, n, k) | data | N=8 o/b | N=8 nw | N=12 o/b | N=12 nw | N=16 o/b | "
          "N=16 nw |\n|---|---|---|---|---|---|---|---|---|")
    for f in fx:
        for d in DISTS:
            row = {r["N"]: r for r in recs if r["fixture"] == f and r["dist"] == d and
                   r["rounding"] == "RN"}
            r0 = row[Ns[0]]
            print(f"| {f} | ({r0['m']}, {r0['n']}, {r0['k']}) | {d} | " + " | ".join(
                f"{row[N]['ratio_max']:.2f} | {row[N]['normwise']:.0e}" for N in (8, 12, 16)) +
                " |")


if __name__ == "__main__":
    main()
