#!/usr/bin/env python3
"""Check that the shapes of a hold-out sweep are absent from a fit sweep, by extents.

    python3 bench/sweep_disjoint.py bench/sweep_mc.txt --against bench/sweep_calib.txt \
        [--report bench/sweep_perf_validate.txt bench/sweep_cache.txt ...] \
        [--shape 'ik,kj->ij|i=2048,k=2048,j=2048' ...]
    python3 bench/sweep_disjoint.py --self-test      (CPU; ctest sweep_disjoint_selftest)

Two shapes match when their GEMM views are equal: (m, n, k, b), or (n, m, k, b) with the
operands swapped (the same products, transposed), whatever the spec's labels and mode order.
The view is imd_bench's (bench/imd_bench.cu parse_case): L = labels of C in both operands,
I / J = labels of C in A / B only, K = labels of A and B not in C. Exit 1 if any case of the
checked sweep (or any --shape) matches a case of an --against sweep; matches in --report
sweeps are listed but do not fail. (C1 M-c, docs/NOTES-C1.md §4.3: every M-c shape is checked
absent from bench/sweep_calib.txt.) The method, N list and data of a case are not compared:
two cases with one view match even when one is DIRECT or packed.
"""

import argparse
import contextlib
import io
import math
import os
import sys
import tempfile


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


def self_test():
    """The matching rule and the exit codes on made-up sweeps, then the committed invariant:
    the M-c hold-out (bench/sweep_mc.txt) shares no GEMM view with either fit sweep, and the
    Q24 fit cases (bench/sweep_calib_n.txt) none with M-c or the product hold-out
    (NOTES-C1 §5.9 MC-3 (4) re-runs M-c on this case list after a refit that includes them),
    and the MC-3a narrow-KL hold-out (bench/sweep_mc3a_holdout.txt, NOTES-C1 §5.11) none with
    either of those or the fit sweeps."""
    def view(s):
        spec, ext = s.split("|")
        return gemm_view(spec, parse_ext(ext))

    def match(s, t):
        return view(s) in key_set(view(t))

    jac = "abcd,cdij->abij|a=40,b=40,c=40,d=40,i=10,j=10"
    pairs = [  # (shape, shape, should match, what)
        (jac, "ik,kj->ij|i=1600,k=1600,j=100", True, "the same view under another spec"),
        ("ik,kj->ij|i=1024,k=1024,j=1024", "ik,jk->ij|i=1024,j=1024,k=1024", True,
         "a square under ik,kj and ik,jk (NOTES-C1 §5.9 MC-4)"),
        ("ik,kj->ij|i=1600,k=1600,j=100", "jk,ik->ij|i=1600,j=100,k=1600", True,
         "A and B swapped: (m, n) = (1600, 100) against (100, 1600)"),
        ("kjbc,kiac->abij|i=32,j=32,k=32,a=128,b=128,c=128", "ik,jk->ij|i=4096,j=4096,k=4096",
         True, "a tensor view equal to a square (v_t_cc and 4096^3)"),
        ("abk,kcd->cadb|a=64,b=64,c=64,d=64,k=4096", "ik,kj->ij|i=4096,k=4096,j=4096", True,
         "a permuted output equal to a square (v_t_perm and 4096^3)"),
        ("ikl,kjl->ijl|i=512,j=512,k=2048,l=8", "lik,ljk->lij|l=8,i=512,j=512,k=2048", True,
         "the batch label last or first"),
        ("ik,kj->ij|i=2000,k=2000,j=2000", "ik,jk->ij|i=2048,j=2048,k=2048", False,
         "a near size (2000^3, 2048^3)"),
        ("ik,kj->ij|i=4095,k=4096,j=4096", "ik,kj->ij|i=4096,k=4096,j=4096", False,
         "one extent off by one"),
        ("ik,kj->ij|i=1600,k=100,j=1600", jac, False, "k exchanged with n"),
        ("bik,bkj->bij|b=2,i=1024,k=1024,j=1024", "ik,kj->ij|i=2048,k=1024,j=1024", False,
         "a batch of 2 against one call of twice the rows"),
    ]
    fail = 0
    for s, t, want, what in pairs:
        for x, y in ((s, t), (t, s)):
            if match(x, y) != want:
                print(f"FAIL: {what}: {x} vs {y} {'should' if want else 'should not'} match")
                fail += 1
    if view(jac) != (1600, 100, 1600, 1):
        print(f"FAIL: the Jacobi view is {view(jac)}, not (1600, 100, 1600, 1)")
        fail += 1

    def run(argv):  # main's exit code, its output captured
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(argv)
        return rc, out.getvalue()

    with tempfile.TemporaryDirectory() as d:
        def sweep(name, lines):
            p = os.path.join(d, name)
            with open(p, "w") as f:
                f.write("# a comment | with | bars\n\n" + "\n".join(lines) + "\n")
            return p
        hold = sweep("hold.txt", ["h_jac | abcd,cdij->abij | a=40,b=40,c=40,d=40,i=10,j=10 "
                                  "| imd | 4,8 | --reps 3",
                                  "h_sq  | ik,kj->ij | i=2000,k=2000,j=2000 | imd | 4,8"])
        fit = sweep("fit.txt", ["f_jac | ik,kj->ij | i=1600,k=1600,j=100 | direct | 8"])
        other = sweep("other.txt", ["o_sq | ik,jk->ij | i=2048,j=2048,k=2048 | imd | 8"])
        both = sweep("both.txt", ["b_sq | ik,jk->ij | i=2000,j=2000,k=2000 | imd | 14"])
        checks = [  # (argv, exit code, text that must appear, what)
            ([hold, "--against", fit], 1, "fit:    f_jac", "a hold-out case in a fit sweep"),
            ([hold, "--against", other], 0, "all absent", "no case in the fit sweep"),
            ([hold, "--against", other, "--report", fit, both], 0, "report: b_sq",
             "matches in --report sweeps are listed and do not fail"),
            ([hold, "--against", other, both], 1, "1 present",
             "a match in any of several --against sweeps fails"),
            (["--against", other, "--shape", "ik,kj->ij|i=2048,k=2048,j=2048"], 1, "PRESENT",
             "a --shape in a fit sweep"),
            (["--against", fit, "--shape", "ik,kj->ij|i=2048,k=2048,j=2048"], 0, "all absent",
             "a --shape absent from the fit sweep"),
        ]
        for argv, want_rc, text, what in checks:
            rc, out = run(argv)
            if rc != want_rc or text not in out:
                print(f"FAIL: {what}: exit {rc} (want {want_rc}), '{text}' "
                      f"{'found' if text in out else 'missing'}\n{out}")
                fail += 1

    bench = os.path.dirname(os.path.abspath(__file__))
    mc, calib, calib_n, hold_v, mc3a = (os.path.join(bench, f) for f in (
        "sweep_mc.txt", "sweep_calib.txt", "sweep_calib_n.txt", "sweep_perf_validate.txt",
        "sweep_mc3a_holdout.txt"))
    for argv, what in (([mc, "--against", calib, calib_n], "M-c against the fit sweeps"),
                       ([calib_n, "--against", mc, hold_v], "the Q24 cases against M-c and "
                                                            "the product hold-out"),
                       ([mc3a, "--against", calib, calib_n, mc, hold_v],
                        "the MC-3a hold-out against the fit sweeps, M-c and the product "
                        "hold-out")):
        rc, out = run(argv)
        if rc != 0:
            print(f"FAIL: {what}: a shared GEMM view\n{out}")
            fail += 1
    n_checked = 2 * len(pairs) + 1 + len(checks) + 3
    print(f"sweep_disjoint self-test: {n_checked - fail} of {n_checked} checks pass")
    return 1 if fail else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sweep", nargs="?", help="the hold-out sweep to check")
    ap.add_argument("--against", nargs="+", help="fit sweeps: a match fails (required)")
    ap.add_argument("--report", nargs="*", default=[], help="other sweeps: matches are listed")
    ap.add_argument("--shape", action="append", default=[], metavar="SPEC|EXT",
                    help="an extra shape to check, e.g. 'ik,kj->ij|i=2048,k=2048,j=2048'")
    ap.add_argument("--self-test", action="store_true",
                    help="check the matching rule, the exit codes and the committed sweeps")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.against:
        ap.error("--against is required")
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
