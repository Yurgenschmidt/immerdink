#!/usr/bin/env python3
"""R4 integration: the instruction-sequence bits of issues 21 / 23 / 25 forced TOGETHER (E1 1, Q1 2, F1 4, klwarp 16, klload 32, klshape 64, k5fin 128, klpersist 256) give the
bits of the baseline.

    python3 tests/test_forced_sets.py [--lib build/libimmerdink.so] [--quick]

The R4 duel sides (docs/NOTES-R4-INTEGRATION.md): 5 (the sm_90 pack set without the limb-CRT bit), 21 / 37 / 69 (+ klwarp / klload / klshape), 53 / 101 / 85 / 117 (KL
pairs and triple), 133 (+ k5fin), and 149 / 165 / 181 / 197 / 245 (k5fin with the KL bits; 181 with crt=f64 is the R5 sm_90 default 189); issue 25 (R6, docs/NOTES-25.md): 293 / 421 (37 / 165 + klpersist: 421 with crt=f64 is the R4 sm_90
default 173 + klpersist), 309 / 437 (+ klwarp; 437 with crt=f64 is the R5 sm_90 default 189 + klpersist, the R6 duel B side) and 357 / 501 (+ klshape). The klpersist sets also run with the persistent grid forced to 3 CTAs (imd_test_set_kl_grid), so the
small cases walk several tiles per CTA and take the prefetch. Every set runs with the K5 CRT forced to the limbs (imd_test_set_k5_crt(1), the duel
side's `crt=f64`); the sets without bit 128 also run with the integer CRT (crt=int). Each contraction of the KL list (tests/test_kl_bits.py) must give ONE hash over all
configurations and equal to the baseline (set 0, integer CRT); the KL launch counter must show the kernel each operand's bits ask for, and the k5fin launch counter must be
nonzero for a set with bit 128 and the limb CRT over the list (k5f_fits can decline a shape, so per-case counts are not asserted) and zero everywhere else: a bit that silently
ran the baseline kernel would pass every bit-identity test.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import test_kl_bits as K  # noqa: E402

SETS = (5, 21, 37, 69, 53, 101, 85, 117, 133, 149, 165, 181, 197, 245, 293, 421, 309, 437, 357, 501)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-seq-twins", action="store_true", help="built with IMMERDINK_SEQ_TWINS=OFF: only checks that the sets are refused")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    if args.no_seq_twins:
        ok = all(lib.lib.imd_test_set_seq(b) != 0 for b in SETS)
        print("PASS  the combined sets are refused without the twins" if ok else "FAIL  a combined set was accepted without the twins")
        return 0 if ok else 1
    configs = [(s, 1, 0) for s in SETS] + [(s, 0, 0) for s in SETS if not s & 128]   # (set, crt, kl grid): crt 1 = limbs ('crt=f64'), 0 = integer
    configs += [(s, 1, 3) for s in SETS if s & 256]   # issue 25: the persistent KL with 3 CTAs (several tiles per CTA: the prefetch runs)
    bad, n, fin_seen = [], 0, {}
    try:
        for case in K.cases(args.quick):
            fmode = case[7]
            lib.set_k5_crt(0)
            ref, _, _ = K.run(lib, cuda, case, 0, fmode)   # the baseline: set 0, integer CRT
            for bits, crt, grid in configs:
                lib.set_k5_crt(crt)
                lib.set_kl_grid(grid)
                lib.k5_fin_counts(reset=True)
                h, seen, k_hat = K.run(lib, cuda, case, bits, fmode)
                fin = sum(lib.k5_fin_counts(reset=True).values())
                n += 1
                if h != ref:
                    bad.append(f"{case[0]} seq {bits} crt {'f64' if crt else 'int'}: hash {h}, baseline {ref}")
                want = K.expect(case, bits, fmode, k_hat)
                if seen != want:
                    bad.append(f"{case[0]} seq {bits} crt {'f64' if crt else 'int'}: KL launches {seen}, expected {want}")
                if fin and not (bits & 128 and crt):
                    bad.append(f"{case[0]} seq {bits} crt {'f64' if crt else 'int'}: {fin} k5fin launches where none are possible")
                if bits & 128 and crt:
                    fin_seen[bits] = fin_seen.get(bits, 0) + fin
        for bits in (s for s in SETS if s & 128):
            if not fin_seen.get(bits):
                bad.append(f"seq {bits} crt f64: no k5fin launch over the whole case list")
    finally:
        lib.set_seq(-1)
        lib.set_fused(-1)
        lib.set_k5_crt(-1)
        lib.set_kl_grid(0)
    if bad:
        print("FAIL")
        for b in bad[:60]:
            print("  " + b)
        return 1
    print(f"PASS  {n} runs: the combined sets {SETS} (crt f64; the sets without 128 also crt int) are bit-identical to the baseline; KL and k5fin launch counters as asked "
          f"(k5fin launches per set: {dict(sorted(fin_seen.items()))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
