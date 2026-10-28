#!/usr/bin/env python3
"""Issue 21: the instruction-sequence set a plan runs (src/host/seq_config.hpp), read back through the test hooks.

    python3 tests/test_seq_select.py [--lib build/libimmerdink.so]

Pins the per-class defaults against this GPU's compute capability (sm_89: E1 = 1; sm_90: E1 + F1 + limb CRT + klwarp + klload + k5fin = 189 (R2: 13, R4: 173, R5: 189); anything
else, sm_120 included: 0), checks that a forced set (imd_test_set_seq) is what a plan reports, that -1 restores the class default, and that the
invalid sets (the Q1 bit without the E1 bit, the KL bits 16 / 32 / 64 are valid in every combination (issue 23: all 48 sets are forced and read back), bit 128 (k5fin) without F1, bit 8 (the limb CRT is a class default only) and out of range) are rejected. The K5 CRT hook is tri-state (-1 follow
the plan's set, 0 integer, 1 limbs): what the plan's K5 runs (imd_test_plan_crt) is pinned for the default, each forced value, and a forced set 0..7 (integer unless the hook forces limbs). A seq run that silently ran the default would
pass every bit-identity test without testing anything; this and the harness's IMD_TEST_SEQ read-back are the guard.
"""
import argparse
import ctypes as C
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

LIMB_CRT = 8
DEFAULTS = {(8, 9): 1, (9, 0): 1 | 4 | LIMB_CRT | 16 | 32 | 128}   # keep equal to kSeqClasses in src/host/seq_config.hpp (tests/test_seq_table.cpp checks the same rows without a GPU)


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
    if args.no_seq_twins:
        want_default &= ~(2 | 4 | 16 | 32 | 128)   # the F1 / Q1 / k5fin kernels are not compiled: the class default is masked
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
        # every combination of the bits 1 E1, 2 Q1, 4 F1, 16 klwarp, 32 klload, 64 klshape, 128 k5fin that is a set (Q1 needs E1, 128 needs F1; bit 8 is a class
        # default only) is forced and read back (issue 23); the KL bits and k5fin are kernel twins like Q1 / F1, so a build without the twins forces E1 and 0 only
        mask = 1 if args.no_seq_twins else 1 | 2 | 4 | 16 | 32 | 64 | 128
        valid_sets = [b for b in range(256) if b & ~mask == 0 and (b & 2 == 0 or b & 1) and (b & 128 == 0 or b & 4)]
        for bits in valid_sets:
            lib.set_seq(bits)
            if lib.plan_seq(plan) != bits or lib.get_seq() != bits:
                bad.append(f"forced {bits}: the plan reports {lib.plan_seq(plan)}, the hook {lib.get_seq()}")
        # rejected: Q1 without E1, k5fin without F1, bit 8, the twins' bits in a build without them, out of range
        rejected = [2, 6, 8, 9, 10, 13, 15, 18, 24, 114, 128, 129, 130, 136, 140, 144, 145, 255, 256, -2]
        if args.no_seq_twins:
            rejected += [3, 4, 5, 7, 16, 17, 32, 64, 113, 127, 132, 133, 135, 245]
        else:
            rejected += [2 | 16, 2 | 32, 2 | 64, 2 | 112, 16 | 8, 127 | 8, 128 | 2 | 4, 128 | 8 | 4]
        lib.set_seq(1)
        for bits in rejected:
            if lib.lib.imd_test_set_seq(bits) == 0:
                bad.append(f"set {bits} (not a set in this build) was accepted")
            elif lib.get_seq() != 1:
                bad.append(f"the rejected set {bits} changed the forced set to {lib.get_seq()}")
        lib.set_seq(-1)
        if lib.get_seq() != -1:
            bad.append("-1 did not restore the class default")
        # the K5 CRT hook: -1 follows the set, 0 / 1 force; a forced seq has no limb-CRT bit
        for v in (2, -2):
            if lib.lib.imd_test_set_k5_crt(v) == 0:
                bad.append(f"set_k5_crt({v}) was accepted")
        lib.set_k5_crt(-1)
        if lib.get_k5_crt() != -1:
            bad.append(f"the K5 CRT hook did not read back -1: {lib.get_k5_crt()}")
        env_seq = os.environ.get("IMD_TEST_SEQ") not in (None, "")
        want_limb = 1 if want_default & LIMB_CRT else 0
        if not env_seq:
            lib.set_seq(-1)
            if lib.plan_crt(plan) != want_limb:
                bad.append(f"default CRT: the plan's K5 runs {lib.plan_crt(plan)}, expected {want_limb} on sm_{major.value}{minor.value}")
            lib.set_k5_crt(0)   # seq default, crt int: the integer CRT on every class
            if lib.plan_crt(plan) != 0 or lib.get_k5_crt() != 0:
                bad.append(f"crt int with the class default set: the plan's K5 runs {lib.plan_crt(plan)}, the hook reads {lib.get_k5_crt()}")
            lib.set_k5_crt(1)
            if lib.plan_crt(plan) != 1:
                bad.append("crt f64 with the class default set did not run the limb CRT")
        for bits in ((0, 1) if args.no_seq_twins else (0, 1, 5, 7, 17, 113, 133, 245)):   # forced sets without the limb-CRT bit: integer CRT unless the hook forces limbs
            lib.set_seq(bits)
            lib.set_k5_crt(-1)
            if lib.plan_crt(plan) != 0:
                bad.append(f"forced seq {bits}, crt following the set: the plan's K5 runs {lib.plan_crt(plan)}, expected the integer CRT")
            lib.set_k5_crt(1)
            if lib.plan_crt(plan) != 1 or lib.plan_seq(plan) != bits:
                bad.append(f"forced seq {bits}, crt f64: CRT {lib.plan_crt(plan)}, set {lib.plan_seq(plan)}")
            lib.set_k5_crt(0)
            if lib.plan_crt(plan) != 0:
                bad.append(f"forced seq {bits}, crt int: CRT {lib.plan_crt(plan)}")
    finally:
        lib.set_k5_crt(-1)
        lib.set_seq(int(os.environ["IMD_TEST_SEQ"]) if os.environ.get("IMD_TEST_SEQ") not in (None, "") else -1)
        lib.plan_destroy(plan)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  sm_{major.value}{minor.value}: default set {want_default} (limb CRT {want_default & LIMB_CRT != 0}); forced sets and the K5 CRT hook read back; invalid sets rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
