#!/usr/bin/env python3
"""Check that the shapes of a hold-out sweep are absent from a fit sweep, by extents.

    python3 bench/sweep_disjoint.py bench/sweep_mc.txt --against bench/sweep_calib.txt \
        [--report bench/sweep_perf_validate.txt bench/sweep_cache.txt ...] \
        [--shape 'ik,kj->ij|i=2048,k=2048,j=2048' ...]

Two shapes match when their GEMM views are equal: (m, n, k, b), or (n, m, k, b) with the
operands swapped (the same products, transposed), whatever the spec's labels and mode order.
The view is imd_bench's (bench/imd_bench.cu parse_case): L = labels of C in both operands,
I / J = labels of C in A / B only, K = labels of A and B not in C. Exit 1 if any case of the
checked sweep (or any --shape) matches a case of an --against sweep; matches in --report
sweeps are listed but do not fail. (C1 M-c, docs/NOTES-C1.md §4.3: every M-c shape is checked
absent from bench/sweep_calib.txt.)
"""

import argparse
import math
import sys


def parse_ext(s):
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if kv:
            lab, val = kv.split("=")
            out[lab.strip()] = int(val)
    return out


def gemm_view(spec, ext):
    ins, c = spec.replace(" ", "").split("->")
    a, b = ins.split(",")
    L = [x for x in c if x in a and x in b]
    I = [x for x in c if x in a and x not in b]
    J = [x for x in c if x in b and x not in a]
    K = [x for x in a if x not in c and x in b]
    prod = lambda labs: math.prod(ext[x] for x in labs)  # noqa: E731
    return prod(I), prod(J), prod(K), prod(L)


def cases(path):
    out = []
    for n, line in enumerate(open(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) < 3:
            continue
        name, spec, ext = f[0], f[1], parse_ext(f[2])
        out.append((name, spec, f[2], gemm_view(spec, ext), f"{path}:{n}"))
    return out


def key_set(v):
    m, n, k, b = v
    return {(m, n, k, b), (n, m, k, b)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sweep", nargs="?", help="the hold-out sweep to check")
    ap.add_argument("--against", nargs="+", required=True, help="fit sweeps: a match fails")
    ap.add_argument("--report", nargs="*", default=[], help="other sweeps: matches are listed")
    ap.add_argument("--shape", action="append", default=[], metavar="SPEC|EXT",
                    help="an extra shape to check, e.g. 'ik,kj->ij|i=2048,k=2048,j=2048'")
    a = ap.parse_args(argv)
    checked = cases(a.sweep) if a.sweep else []
    for s in a.shape:
        spec, ext = s.split("|")
        checked.append((f"--shape {spec} {ext}", spec, ext, gemm_view(spec, parse_ext(ext)),
                        "command line"))
    fits = {p: cases(p) for p in a.against}
    reps = {p: cases(p) for p in a.report}
    bad = 0
    for name, spec, ext, view, where in checked:
        hits = [(p, c) for p, cs in fits.items() for c in cs if view in key_set(c[3])]
        info = [(p, c) for p, cs in reps.items() for c in cs if view in key_set(c[3])]
        m, n, k, b = view
        status = "PRESENT in a fit sweep" if hits else "absent"
        print(f"{name:<34} {spec:<18} m={m} n={n} k={k} b={b}: {status}")
        for p, c in hits:
            print(f"    fit:    {c[0]} ({c[1]} {c[2]}) {c[4]}")
        for p, c in info:
            print(f"    report: {c[0]} ({c[1]} {c[2]}) {c[4]}")
        bad += bool(hits)
    print(f"{len(checked)} shapes checked against {', '.join(a.against)}: "
          f"{'all absent' if not bad else f'{bad} present'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
