#!/usr/bin/env python3
"""Smoke test of imd_bench's accuracy fields (method imd; docs/NOTES-BENCH-ACCURACY.md).

    python3 bench/test_accuracy_fields.py --bench build/bench/imd_bench [--keep]

Checks, on a small square GEMM at N = 6, 8, 11, 14 (the GPU is used under the caller's GPU lock):
  * the usage text names --accuracy; --accuracy 2 is refused;
  * one imd record per N, each with the acc_* fields (acc_ref, acc_n_sampled, acc_max_relerr_cw, acc_bits_cw, acc_max_rel_to_norm,
    acc_dgemm, acc_dgemm_max_relerr_cw, acc_vs_dgemm_max_cw, acc_vs_dgemm_n = m*n);
  * the componentwise error is finite, positive and strictly decreasing with N (6 > 8 > 11 > 14), the sampled and the
    full-matrix (versus native DGEMM) figures agree to a small factor, and N = 14 is below 1e-8;
  * --accuracy 0 drops the DGEMM fields but keeps the sampled ones, with the same C (c_hash).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

CASE = ["--name", "acc", "--spec", "ik,kj->ij", "--ext", "i=256,j=256,k=256"]
FAST = ["--reps", "2", "--warmup", "1", "--phase-reps", "1", "--max-seconds", "5", "--samples", "512"]
NS = [6, 8, 11, 14]


def run(bench, extra, out):
    return subprocess.run([bench] + CASE + FAST + extra + ["--out", out], capture_output=True, text=True, timeout=600)


def records(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    bad = []
    tmp = tempfile.mkdtemp(prefix="imd_acc_")

    u = subprocess.run([args.bench], capture_output=True, text=True, timeout=60)
    if u.returncode == 0 or "--accuracy" not in u.stderr:
        bad.append("the usage text does not name --accuracy (or exits 0)")
    if run(args.bench, ["--N", "8", "--accuracy", "2"], os.path.join(tmp, "bad.jsonl")).returncode == 0:
        bad.append("--accuracy 2 was accepted")

    out = os.path.join(tmp, "on.jsonl")
    r = run(args.bench, ["--N", ",".join(map(str, NS)), "--accuracy", "1"], out)
    recs = []
    if r.returncode != 0:
        bad.append(f"run failed: exit {r.returncode}: {r.stderr[-400:]}")
    else:
        recs = [x for x in records(out) if x.get("method") == "imd"]
    by_n = {x["N"]: x for x in recs}
    if sorted(by_n) != NS or len(recs) != len(NS):
        bad.append(f"expected one imd record per N {NS}, got {[x['N'] for x in recs]}")
    need = ["acc_ref", "acc_n_sampled", "acc_max_relerr_cw", "acc_bits_cw", "acc_max_relerr_elem", "acc_max_rel_to_norm", "acc_dgemm",
            "acc_dgemm_max_relerr_cw", "acc_vs_dgemm_max_cw", "acc_vs_dgemm_mean_cw", "acc_vs_dgemm_n", "max_cwerr"]
    cw = []
    for n in NS:
        x = by_n.get(n)
        if x is None:
            continue
        miss = [k for k in need if k not in x]
        if miss:
            bad.append(f"N={n}: missing {miss} (acc_dgemm={x.get('acc_dgemm')})")
            continue
        if x["acc_n_sampled"] != 512 or x["acc_vs_dgemm_n"] != 256 * 256 or x.get("acc_vs_dgemm_nonfinite") != 0:
            bad.append(f"N={n}: acc_n_sampled {x['acc_n_sampled']}, acc_vs_dgemm_n {x['acc_vs_dgemm_n']}")
        if not (x["acc_max_relerr_cw"] > 0 and x["acc_vs_dgemm_max_cw"] > 0):
            bad.append(f"N={n}: non-positive error {x['acc_max_relerr_cw']} {x['acc_vs_dgemm_max_cw']}")
        if x["acc_max_relerr_cw"] != x["max_cwerr"] or x["acc_max_rel_to_norm"] != x["max_normerr"]:
            bad.append(f"N={n}: acc_* disagrees with max_cwerr / max_normerr")
        # |C - dgemm| <= |C - ref| + |dgemm - ref|, over all entries versus 512 samples: same order of magnitude
        if x["acc_vs_dgemm_max_cw"] > 20 * (x["acc_max_relerr_cw"] + x["acc_dgemm_max_relerr_cw"]) + 1e-15:
            bad.append(f"N={n}: vs-DGEMM {x['acc_vs_dgemm_max_cw']:.3g} far above sampled {x['acc_max_relerr_cw']:.3g}")
        cw.append(x["acc_max_relerr_cw"])
    if len(cw) == len(NS):
        if not all(a > b for a, b in zip(cw, cw[1:])):
            bad.append(f"componentwise error not strictly decreasing with N {NS}: {cw}")
        if cw[-1] > 1e-8:
            bad.append(f"N=14 componentwise error {cw[-1]:.3g} above 1e-8")

    out0 = os.path.join(tmp, "off.jsonl")
    r = run(args.bench, ["--N", "8", "--accuracy", "0"], out0)
    if r.returncode != 0:
        bad.append(f"--accuracy 0 run failed: {r.stderr[-300:]}")
    else:
        x = [y for y in records(out0) if y.get("method") == "imd"][0]
        if "acc_max_relerr_cw" not in x or "acc_vs_dgemm_max_cw" in x or "acc_dgemm_max_relerr_cw" in x:
            bad.append("--accuracy 0 must keep the sampled fields and drop the DGEMM ones")
        if 8 in by_n and x.get("c_hash") != by_n[8].get("c_hash"):
            bad.append("c_hash differs with --accuracy 0")

    if args.keep:
        print("records in", tmp)
    if bad:
        for b in bad:
            print("FAIL ", b)
        return 1
    print("OK  cw error N=%s: %s" % (NS, ", ".join(f"{v:.2e}" for v in cw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
