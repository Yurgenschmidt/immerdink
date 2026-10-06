#!/usr/bin/env python3
"""Before/after of immerdink from a bench/run_ab.sh pair: per (case, method, variant, N) the
median event time in each side's own process and after / before (< 1: the after build is
faster). For duel-b records the in-process ratio t_a / t_b of each side is shown as well.
Markdown table on stdout.

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
        out.setdefault((r["case"], r["method"], r["variant"], r["N"]), r)
    return out


def main():
    b, a = load(sys.argv[1]), load(sys.argv[2])
    print("| case | method | variant | N | before µs | after µs | **after / before** | duel t_a / t_b (b → a) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, rb in b.items():
        ra = a.get(key)
        if ra is None:
            continue
        tb, ta = rb["median_s"] * 1e6, ra["median_s"] * 1e6
        duel = ""
        if "ratio_median" in rb and "ratio_median" in ra:
            duel = f"{rb['ratio_median']:.3f} → {ra['ratio_median']:.3f}"
        case, method, variant, N = key
        print(f"| {case} | {method} | {variant} | {N} | {tb:.1f} | {ta:.1f} | **{ta / tb:.3f}** | {duel} |")


if __name__ == "__main__":
    main()
