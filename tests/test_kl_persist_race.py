#!/usr/bin/env python3
"""Issue 25 (docs/NOTES-25.md): the klpersist KL kernel (seq bit 256) under compute-sanitizer racecheck and synccheck.

    python3 tests/test_kl_persist_race.py --lib build/libimmerdink.so --sanitizer <compute-sanitizer>

The persistent kernel's correctness rests on ordering that a hash test cannot always see: the cp.async prefetch of the next tile's chunk 0 must be waited for
(cp.async.wait_group) before the thread reads it, and the next tile's row info (written by th threads into the other slot) must sit behind a barrier before
any thread reads it. A dropped wait passed the bit-identity tests on the RTX 4090 (the copies land long before pass 2 / 3 end) but racecheck reports it
(32768 hazards per access: the mutation check of docs/NOTES-25.md). This runs the KL cases of tests/test_kl_bits.py that cover every prefetch path (16-byte
copies, k tails, unaligned rows, zero / non-finite rows, rows of one warp or less, two chunks, g = 4) with the four klpersist kernel variants (Kb 6 / 7,
E 8 / 16) and the grid forced to 1 and 3 CTAs (several tiles per CTA), racecheck and synccheck restricted to kl_scale_pack. PASS needs 0 hazards and 0 errors
(and the same C hash as the set without bit 256 is checked by test_kl_bits.py, not here: the sanitizers serialize, the numbers are the same).
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = "k4096_chunk1,k1000_tail,k8192_two_chunks,k70_small_rows,k17_n20,odd_pitch_A,g4_n14,specials"
SETS = (289, 305, 353, 369)   # E1 + klload + klpersist, + klwarp, + klshape, + both
GRIDS = (1, 3)


def child(lib_path, sets, grids):
    sys.path.insert(0, HERE)
    import harness as hz  # noqa: E402
    import test_kl_bits as K  # noqa: E402
    lib, cuda = hz.Library(lib_path), hz.Cuda()
    names = set(CASES.split(","))
    try:
        for case in K.cases(False):
            if case[0] not in names:
                continue
            for bits in sets:
                for grid in grids:
                    lib.set_kl_grid(grid)
                    _, seen, _ = K.run(lib, cuda, case, bits, case[7])
                    if not any(kb & 4 for (_, kb) in seen):
                        print(f"FAIL  {case[0]} seq {bits} grid {grid}: no klpersist launch ({seen})")
                        return 1
    finally:
        lib.set_kl_grid(0)
        lib.set_seq(-1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--sanitizer", help="compute-sanitizer binary (required unless --child)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args.lib, SETS, GRIDS)
    rc = 0
    for tool, ok_pat in (("racecheck", r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"), ("synccheck", r"ERROR SUMMARY: 0 errors")):
        cmd = [args.sanitizer, "--tool", tool, "--kernel-name", "kns=kl_scale_pack", sys.executable, os.path.abspath(__file__), "--lib", args.lib, "--child"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        good = r.returncode == 0 and re.search(ok_pat, out) is not None
        print(("PASS" if good else "FAIL") + f"  {tool}: klpersist sets {SETS}, grids {GRIDS}, {len(CASES.split(','))} cases")
        if not good:
            print("\n".join(out.splitlines()[-30:]))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
