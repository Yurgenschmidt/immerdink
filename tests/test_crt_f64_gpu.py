#!/usr/bin/env python3
"""Issue 21: K5's CRT with the sums in binary64 limbs gives the same bits as the integer running CRT.

    python3 tests/test_crt_f64_gpu.py [--lib build/libimmerdink.so]

Every case of tests/data/output_hashes.json (the committed hashes, so the f64 variant also reproduces the contract), then every N = 2..20 on a mid-size GEMM of
each input family, and batched / permuted-output shapes, are run with the integer CRT and with the binary64-limb CRT (imd_test_set_k5_crt); the output hashes must be
equal. K5's element variants (1, 2, 4 per thread, batch fastest) are forced for N = 5, 8, 12, 16, 20, with the launch counter checked. The limb CRT uses the FP64 pipe: on an RTX 4090 it is slow but exact, which is what this checks. spec/test_crt_f64.py checks the algorithm against the oracle on the
residue vectors directly; tests/test_crt_f64_unit.py runs the device functions on the edge vectors.
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
    try:
        for c, committed_hash in cases:
            lib.set_k5_crt(0)
            h0 = T.hash_c(T.run_gpu(lib, cuda, c))
            lib.set_k5_crt(1)
            h1 = T.hash_c(T.run_gpu(lib, cuda, c))
            n += 1
            if h0 != h1 or (committed_hash is not None and h1 != committed_hash):
                bad.append(f"{c['name']}: integer CRT {h0}, limb CRT {h1}, committed {committed_hash}")
        # the K5 element variants, forced: 1, 2 and 4 elements per thread (j-bar fastest) and the batch-fastest kernel, for a few N, with tails (n and the
        # batch extent are not multiples of the per-thread element count / tile). The launch counter (imd_test_k5_launch_counts, recorded by launch_k5 at
        # the branch that picks the kernel) shows that the intended variant, with the intended CRT, ran: exactly one K5 launch per run.
        names = {0: "E=1", 1: "E=2", 2: "E=4", 3: "bfast"}
        for N in (5, 8, 12, 16, 20):
            shapes = []
            for variant, elems in ((0, 1), (1, 2), (2, 4)):
                shapes.append((variant, elems, case(f"var_{names[variant]}_n{N}", {"extent": [37, 130], "label": [1, 3]}, {"extent": [53, 130], "label": [2, 3]},
                                                    {"extent": [37, 53], "label": [1, 2]}, N, "pow2" if N % 2 else "u", "TRUNC" if N % 3 == 0 else "RN")))
            for bt in (6, 40):
                shapes.append((3, 0, case(f"var_bfast_b{bt}_n{N}", {"extent": [9, 33, bt], "label": [1, 3, 4]}, {"extent": [11, 33, bt], "label": [2, 3, 4]},
                                          {"extent": [9, 11, bt], "label": [1, 2, 4]}, N, "pow2" if N % 2 else "u"))
                          )
            for variant, elems, c in shapes:
                hs = []
                for crt in (0, 1):
                    lib.set_k5_crt(crt)
                    lib.set_k5_elems(elems)
                    lib.k5_launch_counts(reset=True)
                    hs.append(T.hash_c(T.run_gpu(lib, cuda, c)))
                    seen = lib.k5_launch_counts(reset=True)
                    if seen != {(crt, variant): 1}:
                        bad.append(f"{c['name']}: CRT {crt}: the K5 launches were {seen}, expected exactly one launch of variant {names[variant]} ({crt}, {variant})")
                n += 1
                if hs[0] != hs[1]:
                    bad.append(f"{c['name']}: integer CRT {hs[0]}, limb CRT {hs[1]}")
    finally:
        lib.set_k5_crt(0)
        lib.set_k5_elems(0)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  {n} cases: the binary64-limb CRT is bit-identical to the integer CRT (N = 2..20, both input families, RN and TRUNC, batched, alpha and beta; K5 variants E = 1, 2, 4 and bfast forced and confirmed for N = 5, 8, 12, 16, 20)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
