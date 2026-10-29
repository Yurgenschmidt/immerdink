#!/usr/bin/env python3
"""CPU-only check of the decision logic behind the per-W pack-occupancy check (tests/occupancy_known.py, used by tests/test_contract.py).

    python3 tests/test_occupancy_known.py

Mocked tables: the recorded Hopper pattern is KNOWN (passes, reported); a different Hopper pattern fails; sm_89 with differing W fails; sm_89 with equal W
passes; the known pattern on another kernel or another major fails; consistent tables on Hopper still pass as before.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occupancy_known import KNOWN_MESSAGE, occupancy_verdict  # noqa: E402

HOPPER_W2, HOPPER_W3 = [4, 64, 40768, 256], [2, 116, 40768, 256]


def rows(w0, w2, w3):
    return {0: list(w0), 2: list(w2), 3: list(w3)}


def main():
    bad = []

    def expect(label, got, want):
        if got[0] != want:
            bad.append(f"{label}: {got}, expected {want}")

    expect("Hopper pattern, W=0 shows W=3", occupancy_verdict("pack", rows(HOPPER_W3, HOPPER_W2, HOPPER_W3), 9), "known")
    expect("Hopper pattern, W=0 shows W=2", occupancy_verdict("pack", rows(HOPPER_W2, HOPPER_W2, HOPPER_W3), 9), "known")
    if occupancy_verdict("pack", rows(HOPPER_W3, HOPPER_W2, HOPPER_W3), 9)[1] != KNOWN_MESSAGE:
        bad.append("the known message changed")
    expect("different Hopper pattern (registers)", occupancy_verdict("pack", rows([2, 117, 40768, 256], [4, 64, 40768, 256], [2, 117, 40768, 256]), 9), "fail")
    expect("different Hopper pattern (CTAs)", occupancy_verdict("pack", rows([2, 116, 40768, 256], [3, 64, 40768, 256], [2, 116, 40768, 256]), 9), "fail")
    expect("different Hopper pattern (smem)", occupancy_verdict("pack", rows(HOPPER_W3, [4, 64, 40000, 256], HOPPER_W3), 9), "fail")
    expect("Hopper pattern, W=0 not a W row", occupancy_verdict("pack", rows([1, 116, 40768, 256], HOPPER_W2, HOPPER_W3), 9), "fail")
    expect("Hopper pattern on another kernel", occupancy_verdict("kl8", rows(HOPPER_W3, HOPPER_W2, HOPPER_W3), 9), "fail")
    expect("sm_89 with the Hopper pattern", occupancy_verdict("pack", rows(HOPPER_W3, HOPPER_W2, HOPPER_W3), 8), "fail")
    expect("sm_89 differing W", occupancy_verdict("pack", rows([2, 100, 40768, 256], [3, 64, 40768, 256], [2, 100, 40768, 256]), 8), "fail")
    expect("sm_120 with the Hopper pattern", occupancy_verdict("pack", rows(HOPPER_W3, HOPPER_W2, HOPPER_W3), 12), "fail")
    eq = [3, 80, 40768, 256]
    expect("sm_89 equal", occupancy_verdict("pack", rows(eq, eq, eq), 8), "ok")
    expect("sm_90 equal (the model's assumption holds)", occupancy_verdict("pack", rows(eq, eq, eq), 9), "ok")
    expect("missing row", occupancy_verdict("pack", {0: None, 2: eq, 3: eq}, 8), "fail")
    expect("W=0 not the most constrained", occupancy_verdict("kl8", rows([4, 64, 40768, 256], [3, 80, 40768, 256], [3, 80, 40768, 256]), 8), "fail")
    if bad:
        for b in bad:
            print("FAIL ", b)
        return 1
    print("PASS  the known Hopper pack-occupancy pattern is accepted only on sm_90 with exactly the recorded rows; every other table decides as before")
    return 0


if __name__ == "__main__":
    sys.exit(main())
