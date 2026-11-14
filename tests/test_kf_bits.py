#!/usr/bin/env python3
"""Issue 21 task 28: KF's fixed-part bits (16384 kff64: sigma from a binary64 square sum with klf64's filter; 32768 kfaff: k offsets as h * stride) give the same bits as the shipped KF.

    python3 tests/test_kf_bits.py [--lib build/libimmerdink.so] [--quick]

KF is the fused small-k pack (k_hat <= 1024: one launch scales and packs both operands, DESIGN v1.4 §4). Every combination of the two bits is forced
(imd_test_set_seq) on top of emitter sets with and without a KF twin (the twins exist for the sm_90 emitter sets only: pack S = 15 at W = 2, S = 1 at W = 3;
host/seq_config.hpp seq_kf_twin) on a list of contractions: k_hat from 16 to 1024 (rows of 1 .. 64 lanes, so rows within a warp and rows across two warps),
tails, batched, g > 1, N = 8 / 14 / 20 (W = 2 and W = 3), a transposed operand (k not the unit stride: kfaff with a stride other than 1), an odd row pitch, a K class of
two modes (kfaff must not be taken), and rows of special values (zero, Inf, NaN, subnormal, all non-finite, subnormal only: kff64 must fall back to the exact integers
for the whole tile), each with the production launch rule and with the small-problem variants off (KF with 16 values per thread). The output hash must be one value
per case and launch mode over ALL sets, and the launch counter (imd_test_kf_launch_counts) must show that the KF twin the set asks for ran (a forced bit that silently
ran the shipped KF would pass every bit-identity test; the counter is the guard). The threshold rows (square sums on and next to the sigma thresholds, where the
filter must decide to fall back) are test_contract.py --suite klf64 under the kff64 sets (ctest contract_klf64_seq*).
"""
import argparse
import ctypes as C
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import test_kl_bits as KB  # noqa: E402
import test_output_hashes as T  # noqa: E402

tens = KB.tens
KFF64, KFAFF = 16384, 32768


def cases(quick):
    cs = []
    cs.append(("k1024_two_warps", tens([37, 1024], [1, 3]), tens([29, 1024], [2, 3]), tens([37, 29], [1, 2]), 14, 0, "u"))
    cs.append(("k1000_tail", tens([41, 1000], [1, 3]), tens([35, 1000], [2, 3]), tens([41, 35], [1, 2]), 14, 0, "pow2"))
    cs.append(("k16_tiny", tens([300, 16], [1, 3]), tens([260, 16], [2, 3]), tens([300, 260], [1, 2]), 14, 0, "u"))
    cs.append(("k70", tens([50, 70], [1, 3]), tens([41, 70], [2, 3]), tens([50, 41], [1, 2]), 8, 0, "pow2"))
    cs.append(("k300_g4", tens([64, 300], [1, 3]), tens([48, 300], [2, 3]), tens([64, 48], [1, 2]), 14, 4, "u"))
    cs.append(("k500_n20", tens([23, 500], [1, 3]), tens([19, 500], [2, 3]), tens([23, 19], [1, 2]), 20, 0, "u"))
    cs.append(("batched_k200", tens([11, 200, 3], [1, 3, 4]), tens([13, 200, 3], [2, 3, 4]), tens([11, 13, 3], [1, 2, 4]), 12, 3, "pow2"))
    cs.append(("odd_pitch_A", tens([21, 611], [1, 3], [613, 1]), tens([17, 611], [2, 3]), tens([21, 17], [1, 2]), 14, 0, "u"))
    cs.append(("transposed_B", tens([31, 800], [1, 3]), tens([800, 27], [3, 2]), tens([31, 27], [1, 2]), 14, 0, "u"))
    # K class of two modes that do not fuse (their order differs between A and B): v_k from the table, kfaff not taken
    cs.append(("two_k_modes", tens([9, 12, 20], [1, 3, 5]), tens([20, 7, 12], [5, 2, 3]), tens([9, 7], [1, 2]), 14, 0, "pow2"))
    cs.append(("specials", tens([24, 900], [1, 3]), tens([20, 900], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "pow2", 2, "specials"))
    cs.append(("specials_k64", tens([24, 64], [1, 3]), tens([20, 64], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "u", 2, "specials"))
    if quick:
        cs = [c for c in cs if c[0] in ("k1024_two_warps", "k16_tiny", "k500_n20", "transposed_B", "two_k_modes", "specials")]
    # (name, A, B, C, N, g, family, [fmode placeholder, "specials"]): KB.make_data reads the specials flag by length
    return cs


def run(lib, cuda, case, bits, small_off):
    name, A, B, Cd, N, g, family = case[:7]
    specials = len(case) > 8
    T3 = {X: hz.make_tensor(t["extent"], t["stride"], t["label"]) for X, t in (("A", A), ("B", B), ("C", Cd))}
    lib.set_seq(bits)
    lib.set_k5_one_elem(0 if small_off else -1)
    st, plan = lib.plan_create(T3["A"], T3["B"], T3["C"], N, g)
    if st != "SUCCESS":
        raise RuntimeError(f"{name}: imd_plan_create {st}")
    st, ws_bytes = lib.workspace_size(T3["A"], T3["B"], T3["C"], N, g)
    bufs = {}
    try:
        for X, t, seed in (("A", A, 3), ("B", B, 6), ("C", Cd, 9)):
            host = KB.make_data(name, t, seed, family if X != "C" else "u", specials)
            base = cuda.malloc((host.size + t["off"]) * 8 + 64)
            addr = (base.value or 0) + t["off"] * 8
            cuda.upload_raw(addr, host)
            bufs[X] = (base, addr, host)
        ws = cuda.malloc(max(ws_bytes, 1))
        lib.kf_launch_counts(reset=True)
        st = lib.contract(plan, 1.0, C.c_void_p(bufs["A"][1]), C.c_void_p(bufs["B"][1]), 0.0, C.c_void_p(bufs["C"][1]), ws, ws_bytes)
        cuda.sync()
        seen = lib.kf_launch_counts(reset=True)
        cuda.free(ws)
        if st != "SUCCESS":
            raise RuntimeError(f"{name}: imd_contract {st}")
        out = np.empty(bufs["C"][2].size, dtype=np.float64)
        cuda.download_raw(out, bufs["C"][1])
        return T.hash_c(out), seen
    finally:
        for p, _, _ in bufs.values():
            cuda.free(p)
        lib.plan_destroy(plan)


def pack_s(bits, W):
    """host/seq_config.hpp seq_pack_s."""
    if not bits & 1:
        return 0
    s = 3 if (bits & 2) and W == 2 else 1
    s |= 4 if (bits & 1024) and W == 2 else 0
    s |= 8 if (bits & 1024) and (bits & 2048) and W == 2 else 0
    return s


def want_kf(case, bits):
    """The Kf value the KF launch must report: the set's KF bits where a twin exists (seq_kf_twin), else 0 (the shipped KF)."""
    W = 2 if case[4] <= 16 else 3
    S = pack_s(bits, W)
    twin = (W == 2 and S == 15) or (W == 3 and S == 1)
    return ((1 if bits & KFF64 else 0) | (2 if bits & KFAFF else 0)) if twin else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-seq-twins", action="store_true", help="built with IMMERDINK_SEQ_TWINS=OFF: the KF bits are not supported; checks only that they are refused")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    if args.no_seq_twins:
        ok = all(lib.lib.imd_test_set_seq(b) != 0 for b in (KFF64, KFAFF, 8119 | KFF64))
        print("PASS  the KF bits are refused without the twins" if ok else "FAIL  a KF bit was accepted without the twins")
        return 0 if ok else 1
    # emitter sets: 8119 = the sm_90 twin of the default (S = 15 at W = 2, S = 1 at W = 3: KF twins), 1 = E1 (S = 1 at W = 2: no twin), 0 = baseline
    emit_sets = (8119, 1) if args.quick else (8119, 3511, 1, 0)
    bad, n, twin_runs = [], 0, 0
    try:
        for case in cases(args.quick):
            for small_off in (False, True):
                ref = None
                for base, kf in itertools.product(emit_sets, (0, KFF64, KFAFF, KFF64 | KFAFF)):
                    bits = base | kf
                    h, seen = run(lib, cuda, case, bits, small_off)
                    n += 1
                    ref = ref or h
                    tag = f"{case[0]}{' (small off)' if small_off else ''} seq {bits}"
                    if h != ref:
                        bad.append(f"{tag}: hash {h}, reference {ref} (seq {emit_sets[0]})")
                    want = want_kf(case, bits)
                    got = {k for (_, k) in seen}
                    if sum(seen.values()) != 1 or got != {want}:
                        bad.append(f"{tag}: KF launches {seen}, expected one with Kf {want}")
                    twin_runs += want != 0
    finally:
        lib.set_seq(-1)
        lib.set_k5_one_elem(-1)
    if bad:
        print("FAIL")
        for b in bad[:60]:
            print("  " + b)
        return 1
    print(f"PASS  {n} runs ({twin_runs} on a KF twin): every combination of kff64 / kfaff over the emitter sets {emit_sets} is bit-identical per case, "
          f"with and without the small-problem variants, and each run launched the KF kernel the set asks for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
