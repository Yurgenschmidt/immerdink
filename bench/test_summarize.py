#!/usr/bin/env python3
"""Self-test of the comparison tables of bench/summarize.py (CPU only, synthetic records).

    python3 bench/test_summarize.py

Checks the DGEMM-level rule (the fastest configuration at or below DGEMM's error), the cuTENSOR emulation
verdict (identical-results test, hash fall back, inconclusive native), perm+ preference over bare GEMM
records, and that missing or failed methods never break the tables.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S  # noqa: E402


def rec(case, method, variant="", N=0, tf=1.0, ms=1.0, e=1e-16, **kw):
    r = dict(case=case, method=method, variant=variant, N=N, status="ok", m=1024, n=1024, k=1024, b=1,
             median_s=ms * 1e-3, tflops_eq=tf, max_normerr=e, max_relerr=e, nonfinite=0, spread_pct=1.0)
    r.update(kw)
    return r


def render(recs, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        S.compare_tables(recs, **kw)
        S.config_table(recs, **kw)
    return buf.getvalue()


def check(cond, what):
    if not cond:
        print("FAIL:", what)
        sys.exit(1)
    print("ok  :", what)


# --- DGEMM-level rule -------------------------------------------------------------------
recs = [
    rec("c", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=1.2, ms=1.7, e=4.0e-16),
    rec("c", "cutensor", "CUTENSOR_COMPUTE_DESC_64F", tf=1.2, ms=1.7, e=4.0e-16),
    rec("c", "imd", "RN g=N", N=13, tf=90, e=1.2e-15),        # fastest but above DGEMM's error: not eligible
    rec("c", "imd", "RN g=N", N=14, tf=80, e=1.3e-16),        # the best eligible one
    rec("c", "imd", "RN g=N", N=15, tf=70, e=1.0e-17),
    rec("c", "gemmul8", "INT8 fastmode", N=14, tf=50, e=2.9e-16, g8_mode="fast"),
    rec("c", "gemmul8", "INT8 accuratemode", N=14, tf=40, e=3.4e-16, g8_mode="accu"),
    rec("c", "emu", "fix47", N=47, tf=60, e=5.0e-15),         # fails DGEMM-level
    rec("c", "emu", "fix55", N=55, tf=45, e=2.7e-17),
    rec("c", "emu", "dyn", N=0, tf=30, e=2.3e-17),
]
fams = S.case_families(recs)
ref, src, best = S.dgemm_level(fams, "tflops_eq")
check(abs(ref - 4.0e-16) < 1e-30 and src == "cuBLAS DGEMM", "reference error is the DGEMM record's")
check(best["imd"]["N"] == 14, "immerdink: N=13 (faster, above DGEMM's error) is skipped, N=14 chosen")
check(best["emu-fixed"]["N"] == 55, "emulation: fix47 fails DGEMM-level, fix55 chosen")
check(best["gemmul8-fast"]["N"] == 14 and best["gemmul8-accu"]["N"] == 14, "GEMMul8 fast and accurate are separate families")
out = render(recs)
check("80" in out and "imd / best emulator" in out, "comparison table renders")
check("1.60×" in out, "immerdink 80 over the best eligible emulator (GEMMul8 fast, 50) is 1.60×")

# --- cuBLAS emulation that fell back to native (C hash equal to DGEMM's) -------------------------
recs3 = [
    rec("f", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=50, ms=1.0, e=4e-16, c_hash="aa"),
    rec("f", "imd", "RN g=N", N=14, tf=40, e=1e-16, c_hash="ii"),
    rec("f", "emu", "fix55", N=55, tf=48, e=4e-16, c_hash="aa"),     # same C as DGEMM: did not emulate
    rec("f", "emu", "dyn", N=0, tf=30, e=2e-17, c_hash="dd"),          # different C: counted
    rec("g", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=50, ms=1.0, e=4e-16, c_hash="aa"),
    rec("g", "imd", "RN g=N", N=14, tf=40, e=1e-16, c_hash="ii"),
    rec("g", "emu", "fix55", N=55, tf=48, e=4e-16, c_hash="aa"),
    rec("h", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=50, ms=1.0, e=4e-16),
    rec("h", "emu", "fix55", N=55, tf=48, e=4e-16),                    # no hashes: stays counted
]
f3 = S.case_families([r for r in recs3 if r["case"] == "f"])
check(f3["emu-fixed"][0].get("native_fallback") is True and not f3["emu-dyn"][0].get("native_fallback"),
      "emu record with the DGEMM record's c_hash is flagged native_fallback")
_, _, b3 = S.dgemm_level(f3, "tflops_eq")
check("emu-fixed" not in b3 and b3["emu-dyn"]["variant"] == "dyn", "fallback emu record is not a candidate for the best emulator")
out3 = render(recs3)
check("not engaged (1 of 1 configurations = native C)" in out3, "all-fallback emu family is shown as not engaged")
check("1.33×" in out3, "case f: imd 40 over the only real emulator (dyn 30) is 1.33x")
f3h = S.case_families([r for r in recs3 if r["case"] == "h"])
check(not f3h["emu-fixed"][0].get("native_fallback"), "no c_hash: not flagged")

# --- perm+ records win over the bare GEMM record ---------------------------------------------------
recs2 = [
    rec("t", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=1.5, e=3e-16, layout="gemm-view (operands pre-permuted)"),
    rec("t", "perm+dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=1.0, e=3e-16, layout="original"),
    rec("t", "imd", "RN g=N", N=14, tf=20, e=1e-16),
]
f2 = S.case_families(recs2)
check(len(f2["dgemm"]) == 1 and f2["dgemm"][0]["method"] == "perm+dgemm", "perm+dgemm replaces the bare gemm-view record")

# --- cuTENSOR emulation verdicts ------------------------------------------------------------------
nat = rec("c", "cutensor", tf=1.2, ms=1.7, e=4e-16, c_hash="aa", repeat_identical=True)
check(S.emu_verdict(nat, rec("c", "cutensor-emu", tf=1.2, ms=1.7, e=4e-16, c_hash="aa", identical_to_native=True))[0] == "no",
      "identical_to_native: not engaged")
check(S.emu_verdict(nat, rec("c", "cutensor-emu", tf=9, ms=0.3, e=1e-16, c_hash="bb", identical_to_native=False))[0] == "yes",
      "differing C and a different time: likely engaged")
check(S.emu_verdict(nat, rec("c", "cutensor-emu", tf=1.2, ms=1.69, e=4.1e-16, c_hash="bb", identical_to_native=False))[0] == "no",
      "differing C but native time and error (another native algorithm, the RTX 4090 t_cc case): not engaged?")
check(S.emu_verdict(nat, rec("c", "cutensor-emu", tf=1.2, ms=1.7, e=4e-16, c_hash="aa"))[0] == "no",
      "no identical_to_native but equal c_hash: not engaged")
check(S.emu_verdict(nat, rec("c", "cutensor-emu", tf=3.0, ms=0.5, e=4e-16, c_hash="bb"))[0] == "yes",
      "no identical_to_native, different c_hash and time, native reproducible: likely engaged")
nondet = rec("c", "cutensor", tf=1.2, ms=1.7, e=4e-16, c_hash="aa", repeat_identical=False)
check(S.emu_verdict(nondet, rec("c", "cutensor-emu", tf=1.2, ms=1.7, e=4e-16, c_hash="bb", identical_to_native=False))[0] == "maybe",
      "native not reproducible and emulated time/error match it: inconclusive")
check(S.emu_verdict(None, None)[0] == "none", "no emulated record")

# --- robustness: missing and failed methods ------------------------------------------------------------------
only_imd = [rec("c", "imd", "RN g=N", N=14, tf=80, e=1e-16)]
out = render(only_imd)
check("c" in out, "a case with immerdink alone renders (no DGEMM reference)")
failed = [
    rec("c", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", tf=1.2, e=4e-16),
    rec("c", "imd", "RN g=N", N=14, tf=80, e=1e-16),
    dict(case="c", method="emu", variant="dyn", N=0, status="error: needs CUDA 13 cuBLAS (x)"),
    dict(case="c", method="gemmul8", variant="INT8 fastmode", N=14, status="error: built without GEMMul8"),
]
out = render(failed)
check("needs CUDA 13 cuBLAS" in out and "built without GEMMul8" in out, "failed methods are listed")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S.compare_tables([], energy=False)
check("no records" in buf.getvalue(), "empty input")

# --- energy flavour -----------------------------------------------------------------------------------
en = [
    rec("e", "dgemm", "CUBLAS_COMPUTE_64F_PEDANTIC", bench_mode="energy", gflops_per_w=3.0, energy_j_per_call=0.1, e=4e-16),
    rec("e", "imd", "RN g=N", N=14, bench_mode="energy", gflops_per_w=300.0, energy_j_per_call=0.001, e=1e-16),
    rec("e", "perm+gemmul8", "INT8 fastmode", N=14, bench_mode="energy", gflops_per_w=150.0, energy_j_per_call=0.002, e=2e-16,
        g8_mode="fast"),
]
out = render(en, energy=True)
check("300" in out and "2.00" in out, "energy table: GFLOPS/W of immerdink and the ratio to the best emulator")
print("all ok")
