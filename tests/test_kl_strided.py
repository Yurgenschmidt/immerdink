#!/usr/bin/env python3
"""Issue 21 "klstrided": the sm_90 class's KL rule gives the same bits as KL for both operands.

    python3 tests/test_kl_strided.py [--lib build/libimmerdink.so]

On the H100 / H200 class (analysis.hpp kKlStridedTable), an operand whose k is not the unit stride (KL reads it along rows: 8 rows
x 512 k per tile) and whose row takes nc = ceil(k_hat / 512) >= kKlStridedNcMin (64) chunks is scaled and packed by K1 + K2/K3 instead
of KL. imd_test_set_fused(6) applies that rule on any device, so this test runs on the RTX 4090 too. Per case the output hash with
mode 6 must equal the one with KL for both operands (mode 2), and the KL launch counter must show the expected number of KL
operands (a rule that silently kept KL would pass the hash check).
"""
import argparse
import sys

import harness as hz
import test_kl_bits as KB

tens = KB.tens


def cases():
    # (name, A, B, C, N, g, family, fmode-unused, KL operands expected under mode 6)
    k = 33000  # k_hat >= 32768: nc >= 64 for an operand read along rows
    return [
        ("A_strided_long", tens([k, 24], [3, 1]), tens([20, k], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "u", 6, 1),
        ("B_strided_long", tens([24, k], [1, 3]), tens([k, 20], [3, 2]), tens([24, 20], [1, 2]), 14, 0, "pow2", 6, 1),
        ("AB_strided_long", tens([k, 24], [3, 1]), tens([k, 20], [3, 2]), tens([24, 20], [1, 2]), 12, 3, "u", 6, 0),
        ("AB_contig_long", tens([24, k], [1, 3]), tens([20, k], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "u", 6, 2),
        ("A_strided_short", tens([16000, 24], [3, 1]), tens([20, 16000], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "u", 6, 2),
        ("A_strided_batched", tens([k, 9, 2], [3, 1, 4]), tens([7, 2, k], [2, 4, 3]), tens([9, 7, 2], [1, 2, 4]), 14, 0, "pow2", 6, 1),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    bad = []
    try:
        for case in cases():
            h2, seen2, _ = KB.run(lib, cuda, case[:8], -1, 2)   # case[8] is the expectation, not the "specials" flag
            h6, seen6, k_hat = KB.run(lib, cuda, case[:8], -1, 6)
            if h6 != h2:
                bad.append(f"{case[0]}: hash {h6} (klstrided rule), {h2} (KL both)")
            if sum(seen2.values()) != 2:
                bad.append(f"{case[0]}: mode 2 ran KL for {sum(seen2.values())} operands, expected 2")
            if sum(seen6.values()) != case[8]:
                bad.append(f"{case[0]} (k_hat {k_hat}): klstrided ran KL for {sum(seen6.values())} operands, expected {case[8]}")
    finally:
        lib.set_seq(-1)
        lib.set_fused(-1)
    if bad:
        print("FAIL")
        for b in bad:
            print("  " + b)
        return 1
    print(f"PASS  {len(cases())} cases: the klstrided rule is bit-identical to KL for both operands and drops KL where it should")
    return 0


if __name__ == "__main__":
    sys.exit(main())
