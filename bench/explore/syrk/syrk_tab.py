#!/usr/bin/env python3
"""explore/syrk-big: tables from imd_bench JSONL (syrk_run.sh check / bench). Standard library only.

Per case (n, k) and N: every record with TFLOPS-eq first (dense count 2 n^2 k for every method), the median time, the SM
clock, the stage times (imd_syrk: pack / blocks / mirror; dsyrk: syrk / mirror), the profile sums when present, and the
accuracy fields. Then the correctness summary: imd_syrk's whole-C bit identity with dense immerdink, and whether the
cuBLAS emulation produced a C bit-identical to native DGEMM (= it fell back to DGEMM).
usage: syrk_tab.py file.jsonl [...]
"""
import json
import sys


def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return recs


def f(x, fmt="%.1f"):
    return fmt % x if isinstance(x, (int, float)) else "-"


def main():
    recs = load(sys.argv[1:])
    cases = []
    for r in recs:
        key = (r.get("case"), r.get("n"), r.get("k"))
        if key not in cases:
            cases.append(key)
    bad = 0
    for case in cases:
        rs = [r for r in recs if (r.get("case"), r.get("n"), r.get("k")) == case]
        print("\n== %s  n=%s k=%s  (TFLOPS-eq = 2 n^2 k / t)" % case)
        print("%-9s %-34s %3s %9s %10s %6s  %-44s %s" % ("method", "variant", "N", "TFLOPSeq", "median_ms", "MHz",
                                                          "stages ms", "accuracy"))
        for r in rs:
            if r.get("status") != "ok":
                print("%-9s %-34s %3s  %s" % (r.get("method"), r.get("variant", "")[:34], r.get("N"), r.get("status")))
                continue
            st = ""
            if r.get("method") == "imd_syrk":
                st = "pack %s blk %s mir %s" % (f(r.get("stage_pack_ms"), "%.2f"), f(r.get("stage_blocks_ms"), "%.2f"),
                                                f(r.get("stage_mirror_ms"), "%.2f"))
                if "prof_k4_ms" in r:
                    st += " | K4 %s K5 %s gap %s" % (f(r.get("prof_k4_ms")), f(r.get("prof_k5_ms")), f(r.get("prof_gap_ms")))
            elif r.get("method") == "dsyrk":
                st = "syrk %s mir %s" % (f(r.get("stage_syrk_ms"), "%.2f"), f(r.get("stage_mirror_ms"), "%.2f"))
            acc = "relmax %s" % f(r.get("max_relerr"), "%.2e")
            if "acc_vs_dgemm_max_rel" in r:
                acc += " vsDGEMM max %s fro %s" % (f(r.get("acc_vs_dgemm_max_rel"), "%.2e"),
                                                  f(r.get("acc_vs_dgemm_fro_rel"), "%.2e"))
            if "bit_identical_to_dense" in r:
                acc += " bitid=%s (%s diff)" % (r["bit_identical_to_dense"], r.get("syrk_vs_dense_mismatch"))
                bad += 0 if r["bit_identical_to_dense"] is True else 1
            print("%-9s %-34s %3s %9s %10s %6s  %-44s %s" % (
                r.get("method"), r.get("variant", "")[:34], r.get("N"), f(r.get("tflops_eq")),
                f(1e3 * r["median_s"], "%.2f") if isinstance(r.get("median_s"), (int, float)) else "-",
                f(r.get("sm_mhz_median"), "%.0f"), st, acc))
        dg = [r.get("c_hash") for r in rs if r.get("method") == "dgemm" and r.get("c_hash")]
        for r in rs:
            if r.get("method") == "emu" and r.get("c_hash") and r.get("c_hash") in dg:
                print("  WARNING: emu %s C is bit-identical to native DGEMM: cuBLAS fell back (workspace?)" % r.get("variant"))
    chk = [r for r in recs if "bit_identical_to_dense" in r]
    if chk:
        print("\nCHECK imd_syrk vs dense immerdink: %d configurations, %d not bit-identical -> %s" % (
            len(chk), bad, "PASS" if bad == 0 else "FAIL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
