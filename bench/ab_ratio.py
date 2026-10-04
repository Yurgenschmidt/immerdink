#!/usr/bin/env python3
"""Before/after vs GEMMul8 from a bench/run_ab.sh pair (A5): per (case, N) the median
event times of GEMMul8 (perm + GEMMul8 when the GEMM view needs permutations: the fair
comparison) and imd in each side's own process, the same-process ratios t(G8) / t(imd)
(> 1: imd faster), and after / before for imd. Markdown table on stdout.

    bench/ab_ratio.py before.jsonl after.jsonl
"""
import json
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "ok" or "method" not in r:
            continue
        out.setdefault((r["case"], r["N"]), {})[r["method"]] = r
    return out


def g8(m):
    return m.get("perm+gemmul8") or m.get("gemmul8")


def main():
    b, a = load(sys.argv[1]), load(sys.argv[2])
    print("| case | N | G8 µs (b / a) | imd before | imd after | after / before | before vs G8 | **after vs G8** |")
    print("|---|---|---|---|---|---|---|---|")
    for key in b:
        if key not in a or "imd" not in b[key] or "imd" not in a[key]:
            continue
        gb, ga = g8(b[key]), g8(a[key])
        ib, ia = b[key]["imd"]["median_s"] * 1e6, a[key]["imd"]["median_s"] * 1e6
        tgb = gb["median_s"] * 1e6 if gb else float("nan")
        tga = ga["median_s"] * 1e6 if ga else float("nan")
        perm = " (perm)" if gb and gb["method"].startswith("perm") else ""
        print(f"| {key[0]}{perm} | {key[1]} | {tgb:.1f} / {tga:.1f} | {ib:.1f} | {ia:.1f} | "
              f"{ia / ib:.3f} | {tgb / ib:.2f} | **{tga / ia:.2f}** |")


if __name__ == "__main__":
    main()
