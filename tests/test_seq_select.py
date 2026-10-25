#!/usr/bin/env python3
"""Issue 21: the instruction-sequence set a plan runs (src/host/seq_config.hpp), read back through the test hooks.

    python3 tests/test_seq_select.py [--lib build/libimmerdink.so]

Pins the per-class defaults against this GPU's compute capability (sm_89: E1 = 1; sm_90: E1 = 1; anything else: 0),
checks that a forced set (imd_test_set_seq) is what a plan reports, that -1 restores the class default, and that the
invalid sets (the Q1 bit without the E1 bit, out of range) are rejected. A seq run that silently ran the default would
pass every bit-identity test without testing anything; this and the harness's IMD_TEST_SEQ read-back are the guard.
"""
import argparse
import ctypes as C
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

DEFAULTS = {(8, 9): 1, (9, 0): 1}   # keep equal to kSeqClasses in src/host/seq_config.hpp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--no-seq-twins", action="store_true", help="built with IMMERDINK_SEQ_TWINS=OFF: only E1 can be forced")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    major, minor = C.c_int(), C.c_int()
    dev = C.c_int()
    cuda.check(cuda.rt.cudaGetDevice(C.byref(dev)), "cudaGetDevice")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(major), 75, dev), "cc major")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(minor), 76, dev), "cc minor")
    want_default = DEFAULTS.get((major.value, minor.value), 0)
    T = {X: hz.make_tensor(e, hz.dense(e), l) for X, e, l in
         (("A", [16, 32], [1, 3]), ("B", [24, 32], [2, 3]), ("C", [16, 24], [1, 2]))}
    st, plan = lib.plan_create(T["A"], T["B"], T["C"], 8)
    assert st == "SUCCESS", st
    bad = []
    try:
        lib.set_seq(-1)
        if os.environ.get("IMD_TEST_SEQ") in (None, ""):
            if lib.plan_seq(plan) != want_default:
                bad.append(f"default set {lib.plan_seq(plan)}, expected {want_default} on sm_{major.value}{minor.value}")
        for bits in ((0, 1) if args.no_seq_twins else (0, 1, 3, 4, 5, 7)):
            lib.set_seq(bits)
            if lib.plan_seq(plan) != bits or lib.get_seq() != bits:
                bad.append(f"forced {bits}: the plan reports {lib.plan_seq(plan)}, the hook {lib.get_seq()}")
        for bits in ((2, 3, 4, 8, -2) if args.no_seq_twins else (2, 6, 8, -2)):
            if lib.lib.imd_test_set_seq(bits) == 0:
                bad.append(f"set {bits} (Q1 without E1, or out of range) was accepted")
        lib.set_seq(-1)
        if lib.get_seq() != -1:
            bad.append("-1 did not restore the class default")
    finally:
        lib.set_seq(int(os.environ["IMD_TEST_SEQ"]) if os.environ.get("IMD_TEST_SEQ") not in (None, "") else -1)
        lib.plan_destroy(plan)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  sm_{major.value}{minor.value}: default set {want_default}; forced sets read back; invalid sets rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
