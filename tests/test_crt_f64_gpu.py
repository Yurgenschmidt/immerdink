#!/usr/bin/env python3
"""Issue 21: K5's CRT with the sums in binary64 limbs gives the same bits as the integer running CRT.

    python3 tests/test_crt_f64_gpu.py [--lib build/libimmerdink.so]

Every case of tests/data/output_hashes.json (the committed hashes, so the f64 variant also reproduces the contract), then every N = 2..20 on a mid-size GEMM of
each input family, and batched / permuted-output shapes, are run with the integer CRT and with the binary64-limb CRT (imd_test_set_k5_crt); the output hashes must be
equal. The limb CRT uses the FP64 pipe: on an RTX 4090 it is slow but exact, which is what this checks. spec/test_crt_f64.py checks the algorithm against the oracle on the
residue vectors directly.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import test_output_hashes as T  # noqa: E402


def case(name, spec_a, spec_b, spec_c, N, family, rounding="RN", alpha=1.0, beta=0.0):
    return {"name": name, "A": spec_a, "B": spec_b, "C": spec_c, "N": N, "family": family, "rounding": rounding, "alpha": alpha, "beta": beta}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--full", action="store_true", help="also the large committed cases (default: tags small and mid)")
    ap.add_argument("--no-seq-twins", action="store_true", help="the library was built with IMMERDINK_SEQ_TWINS=OFF: no F1 / Q1 kernels")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    with open(T.DATA) as f:
        committed = [c for c in json.load(f)["cases"] if args.full or set(c["tags"]) & {"small", "mid"}]
    cases = [(c, c["c_hash"]) for c in committed]
    for N in range(2, 21):
        for fam in ("u", "pow2"):
            cases.append((case(f"n{N}_{fam}", {"extent": [48, 160], "label": [1, 3]}, {"extent": [56, 160], "label": [2, 3]},
                               {"extent": [48, 56], "label": [1, 2]}, N, fam, "TRUNC" if N % 3 == 0 else "RN"), None))
    cases.append((case("batched", {"extent": [24, 64, 6], "label": [1, 3, 4]}, {"extent": [40, 64, 6], "label": [2, 3, 4]},
                       {"extent": [24, 40, 6], "label": [1, 2, 4]}, 14, "pow2"), None))
    cases.append((case("alpha_beta", {"extent": [33, 70], "label": [1, 3]}, {"extent": [29, 70], "label": [2, 3]},
                       {"extent": [33, 29], "label": [1, 2]}, 12, "u", alpha=0.75, beta=-0.5), None))
    bad, n = [], 0
    # (crt variant, instruction-sequence set): the integer CRT with the baseline finish is the reference; the limb CRT, F1 (the CRT finish
    # through I2F.F64.U64, issue 21 sequence bit 4) and both together must give the same bits.
    combos = [(1, 0)] if args.no_seq_twins else [(1, 0), (0, 4), (1, 4), (0, 7)]
    try:
        for c, committed_hash in cases:
            lib.set_k5_crt(0)
            lib.set_seq(0)
            h0 = T.hash_c(T.run_gpu(lib, cuda, c))
            n += 1
            if committed_hash is not None and h0 != committed_hash:
                bad.append(f"{c['name']}: reference {h0}, committed {committed_hash}")
            for crt, seq in combos:
                lib.set_k5_crt(crt)
                lib.set_seq(seq)
                h1 = T.hash_c(T.run_gpu(lib, cuda, c))
                if h0 != h1:
                    bad.append(f"{c['name']}: integer CRT {h0}, crt={crt} seq={seq}: {h1}")
    finally:
        lib.set_k5_crt(0)
        lib.set_seq(-1)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  {n} cases: the binary64-limb CRT and the F1 finish, alone and together, are bit-identical to the integer CRT with the baseline finish (N = 2..20, both input families, RN and TRUNC, batched, alpha and beta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
