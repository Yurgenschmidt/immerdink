#!/usr/bin/env python3
"""Issue 21: K5's CRT with the sums in binary64 limbs gives the same bits as the integer running CRT.

    python3 tests/test_crt_f64_gpu.py [--lib build/libimmerdink.so]

Every case of tests/data/output_hashes.json (the committed hashes, so the f64 variant also reproduces the contract), then every N = 2..20 on a mid-size GEMM of
each input family, and batched / permuted-output shapes, are run with the integer CRT and with the binary64-limb CRT (imd_test_set_k5_crt); the output hashes must be
equal (also with the F1 finish, instruction-sequence bit 4). The K5 hook is tri-state (-1 follow the device class's set, 0 integer, 1 limbs): the default run, the "seq default, crt int" run
and a forced seq with the CRT following the set are also run, and the launch counter shows which CRT each took (sm_90 defaults to the limb CRT, every other class to the integer CRT). K5's element variants (1, 2, 4 per thread, batch fastest) are forced for N = 5, 8, 12, 16, 20, with the launch counter checked and F1 forced on each. The limb CRT uses the FP64 pipe: on an RTX 4090 it is slow but exact, which is what this checks. spec/test_crt_f64.py checks the algorithm against the oracle on the
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


LIMB_CRT_CLASSES = {(9, 0)}


def case(name, spec_a, spec_b, spec_c, N, family, rounding="RN", alpha=1.0, beta=0.0):
    return {"name": name, "A": spec_a, "B": spec_b, "C": spec_c, "N": N, "family": family, "rounding": rounding, "alpha": alpha, "beta": beta}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--full", action="store_true", help="also the large committed cases (default: tags small and mid)")
    ap.add_argument("--no-seq-twins", action="store_true", help="the library was built with IMMERDINK_SEQ_TWINS=OFF: no F1 / Q1 kernels")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    import ctypes as C
    major, minor, dev = C.c_int(), C.c_int(), C.c_int()
    cuda.check(cuda.rt.cudaGetDevice(C.byref(dev)), "cudaGetDevice")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(major), 75, dev), "cc major")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(minor), 76, dev), "cc minor")
    default_limb = 1 if (major.value, minor.value) in LIMB_CRT_CLASSES else 0   # keep equal to the kSeqLimbCrt rows of kSeqClasses (src/host/seq_config.hpp)
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
    # Issue 23: bit 128 (k5fin, the limb CRT + F1 kernels with fewer instructions) is added: (1, 132) is the kernel it replaces' combination (limb CRT, F1, bit 128), (1, 135) the whole stack,
    # (0, 132) the integer CRT with the bit set, where it must do nothing (the shipped K5 runs).
    combos = [(1, 0)] if args.no_seq_twins else [(1, 0), (0, 4), (1, 4), (0, 7), (1, 132), (1, 135), (0, 132)]
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
            for crt, seq in ((-1, -1), (0, -1), (-1, 5 if not args.no_seq_twins else 1)) + (() if args.no_seq_twins else ((-1, 133), (1, 133))):   # the class default, seq default + crt int, a forced seq with the CRT following the set
                lib.set_k5_crt(crt)
                lib.set_seq(seq)
                h1 = T.hash_c(T.run_gpu(lib, cuda, c))
                if h0 != h1:
                    bad.append(f"{c['name']}: integer CRT {h0}, crt={crt} seq={seq}: {h1}")
        # the K5 element variants, forced: 1, 2 and 4 elements per thread (j-bar fastest) and the batch-fastest kernel, for a few N, with tails (n and the
        # batch extent are not multiples of the per-thread element count / tile). The launch counter (imd_test_k5_launch_counts, recorded by launch_k5 at
        # the branch that picks the kernel) shows that the intended variant, with the intended CRT, ran: exactly one K5 launch per run. Every variant also
        # runs with the F1 finish forced (seq 4 and 7), so F1 is exercised on all four K5 kernels with both CRTs, against the integer CRT with the baseline finish.
        names = {0: "E=1", 1: "E=2", 2: "E=4", 3: "bfast"}
        vcombos = [(0, 0)] + combos + ([] if args.no_seq_twins else [(1, 7)])   # (the seq-set combos include the k5fin ones: 132, 135)
        # (crt hook, seq hook) -> the CRT the launch must take: forced values are themselves; -1 follows the class default's limb-CRT bit when the seq is not forced, and a forced seq has none (integer)
        dcombos = [(-1, -1), (0, -1), (-1, 0 if args.no_seq_twins else 5)]
        want = lambda crt, seq: default_limb if (crt == -1 and seq == -1) else (0 if crt == -1 else crt)
        for N in (5, 8, 12, 16, 20):
            shapes = []
            for variant, elems in ((0, 1), (1, 2), (2, 4)):
                shapes.append((variant, elems, case(f"var_{names[variant]}_n{N}", {"extent": [37, 130], "label": [1, 3]}, {"extent": [53, 130], "label": [2, 3]},
                                                    {"extent": [37, 53], "label": [1, 2]}, N, "pow2" if N % 2 else "u", "TRUNC" if N % 3 == 0 else "RN")))
            for bt in (6, 40):
                shapes.append((3, 0, case(f"var_bfast_b{bt}_n{N}", {"extent": [9, 33, bt], "label": [1, 3, 4]}, {"extent": [11, 33, bt], "label": [2, 3, 4]},
                                          {"extent": [9, 11, bt], "label": [1, 2, 4]}, N, "pow2" if N % 2 else "u"))
                          )
            # issue 23 (k5fin): n % 4 == 0 (the vector metadata loads), and rows of 32 / 33 / 128 j-bar groups of 4 (a warp inside one row, a warp across two rows), n % 4 != 0 with long rows
            for nn_, mm_ in ((64, 5), (132, 3), (512, 2), (130, 3)):
                shapes.append((2, 4, case(f"var_E=4_m{mm_}_n{nn_}_N{N}", {"extent": [mm_, 70], "label": [1, 3]}, {"extent": [nn_, 70], "label": [2, 3]},
                                          {"extent": [mm_, nn_], "label": [1, 2]}, N, "pow2" if N % 2 else "u", "TRUNC" if N % 3 == 0 else "RN")))
            for variant, elems, bt_, nn_ in ((0, 1, 1, 64), (1, 2, 1, 132)):
                shapes.append((variant, elems, case(f"var_{names[variant]}_n{nn_}_N{N}", {"extent": [6, 33], "label": [1, 3]}, {"extent": [nn_, 33], "label": [2, 3]},
                                                    {"extent": [6, nn_], "label": [1, 2]}, N, "u")))
            shapes.append((3, 0, case(f"var_bfast_b6_n64_N{N}", {"extent": [9, 33, 6], "label": [1, 3, 4]}, {"extent": [64, 33, 6], "label": [2, 3, 4]},
                                      {"extent": [9, 64, 6], "label": [1, 2, 4]}, N, "u")))
            for variant, elems, c in shapes:
                hs = []
                for crt, seq in vcombos:
                    lib.set_k5_crt(crt)
                    lib.set_seq(seq)
                    lib.set_k5_elems(elems)
                    lib.k5_launch_counts(reset=True)
                    lib.k5_fin_counts(reset=True)
                    hs.append(T.hash_c(T.run_gpu(lib, cuda, c)))
                    seen = lib.k5_launch_counts(reset=True)
                    if seen != {(crt, variant): 1}:
                        bad.append(f"{c['name']}: CRT {crt} seq {seq}: the K5 launches were {seen}, expected exactly one launch of variant {names[variant]} ({crt}, {variant})")
                    fin = lib.k5_fin_counts(reset=True)   # k5fin (bit 128) runs for the limb CRT + F1 + bit 128, nowhere else
                    want_fin = {variant: 1} if (crt == 1 and (seq & 132) == 132) else {}
                    if fin != want_fin:
                        bad.append(f"{c['name']}: CRT {crt} seq {seq}: the k5fin launches were {fin}, expected {want_fin}")
                for crt, seq in dcombos:
                    lib.set_k5_crt(crt)
                    lib.set_seq(seq)
                    lib.set_k5_elems(elems)
                    lib.k5_launch_counts(reset=True)
                    lib.k5_fin_counts(reset=True)
                    hs.append(T.hash_c(T.run_gpu(lib, cuda, c)))
                    seen = lib.k5_launch_counts(reset=True)
                    if lib.k5_fin_counts(reset=True):
                        bad.append(f"{c['name']}: crt hook {crt} seq hook {seq}: a k5fin kernel ran, but no class default has bit 128")
                    if seen != {(want(crt, seq), variant): 1}:
                        bad.append(f"{c['name']}: crt hook {crt} seq hook {seq}: the K5 launches were {seen}, expected exactly one launch of variant {names[variant]} ({want(crt, seq)}, {variant})")
                n += 1
                if len(set(hs)) != 1:
                    bad.append(f"{c['name']}: hashes over (crt, seq) {vcombos + dcombos}: {hs}")
    finally:
        lib.set_k5_crt(-1)
        lib.set_k5_elems(0)
        lib.set_seq(-1)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  {n} cases: the binary64-limb CRT and the F1 finish, alone and together, are bit-identical to the integer CRT with the baseline finish (N = 2..20, both input families, RN and TRUNC, batched, alpha and beta; K5 variants E = 1, 2, 4 and bfast forced and confirmed, also with F1, for N = 5, 8, 12, 16, 20)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
