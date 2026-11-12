#!/usr/bin/env python3
"""Issue 23: the KL fixed-part cuts (instruction-sequence bits 16 klwarp, 32 klload, 64 klshape; issue 21: 256 klf64, with klwarp; 512 kllong) give the same bits as the baseline KL kernel.

    python3 tests/test_kl_bits.py [--lib build/libimmerdink.so] [--quick]

Every combination of the three bits over the emitter sets {0 (baseline), 1 (E1), 5 (E1 + F1), 7 (E1 + Q1 + F1)} is forced (imd_test_set_seq) on a list of
contractions whose operands are read by KL (imd_test_set_fused(2) forces it for both operands; 3 = 64-row tiles, many chunks per row): odd shapes and tails, K not
a multiple of the vector width or of 16, rows of one chunk, two chunks and more than two chunks (the 4-values-per-thread kernel), g > 1 groups, batched, strided
rows (odd row pitch: row bases that are not 16-byte aligned), a transposed operand (the gather path of klload), a base pointer offset by one double (misaligned: klload
must not be taken, or fall back), zero rows, Inf, NaN and subnormals. The output hash must be one value per case over ALL sets, and the launch counter
(imd_test_kl_launch_counts) must show that each operand ran the kernel the set asks for: E = 4 / 8 / 16 values per thread and Kb (bit 0 klwarp, bit 1 klload,
which the launch takes only for a unit-stride k read along k-bar and a 16-byte aligned base: by bit 32 klload for at most two chunks, by bit 512 kllong for more). A bit that silently ran the baseline kernel
would pass every bit-identity test; the counter is the guard. The committed hashes, the oracle suites and the sequence-set readback are the other tests of these bits
(ctest *_klN, output_hashes_klN, seq_select).
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
import test_output_hashes as T  # noqa: E402


def ext_len(extent, stride):
    return 1 + sum((e - 1) * s for e, s in zip(extent, stride))


def tens(extent, label, stride=None, off=0, **kw):
    return {"extent": extent, "label": label, "stride": stride or hz.dense(extent), "off": off, **kw}


def cases(quick):
    cs = []
    # unit-stride k on both operands (labels: A [m k], B [n k]); k_hat 4096 = one chunk of 512 threads x 8
    cs.append(("k4096_chunk1", tens([5, 4096], [1, 3]), tens([7, 4096], [2, 3]), tens([5, 7], [1, 2]), 14, 0, "u", 2))
    cs.append(("k1000_tail", tens([37, 1000], [1, 3]), tens([29, 1000], [2, 3]), tens([37, 29], [1, 2]), 14, 0, "pow2", 2))
    cs.append(("k8192_two_chunks", tens([6, 8192], [1, 3]), tens([5, 8192], [2, 3]), tens([6, 5], [1, 2]), 12, 0, "pow2", 2))
    cs.append(("k8300_three_chunks", tens([3, 8300], [1, 3]), tens([4, 8300], [2, 3]), tens([3, 4], [1, 2]), 14, 0, "u", 2))
    cs.append(("k70_small_rows", tens([50, 70], [1, 3]), tens([41, 70], [2, 3]), tens([50, 41], [1, 2]), 8, 0, "pow2", 2))
    cs.append(("k17_n20", tens([23, 17], [1, 3]), tens([19, 17], [2, 3]), tens([23, 19], [1, 2]), 20, 0, "u", 2))
    cs.append(("g4_n14", tens([40, 1500], [1, 3]), tens([36, 1500], [2, 3]), tens([40, 36], [1, 2]), 14, 4, "pow2", 2))
    cs.append(("batched_g3", tens([11, 2100, 3], [1, 3, 4]), tens([13, 2100, 3], [2, 3, 4]), tens([11, 13, 3], [1, 2, 4]), 12, 3, "u", 2))
    # an odd row pitch: row bases at odd double offsets (not 16-byte aligned) in A, even pitch in B
    cs.append(("odd_pitch_A", tens([21, 1111], [1, 3], [1113, 1]), tens([17, 1111], [2, 3]), tens([21, 17], [1, 2]), 14, 0, "u", 2))
    # the base pointer one double off 16 bytes
    cs.append(("offset_ptr_A", tens([19, 2000], [1, 3], off=1), tens([23, 2000], [2, 3]), tens([19, 23], [1, 2]), 14, 0, "pow2", 2))
    cs.append(("offset_ptr_AB", tens([19, 2000], [1, 3], off=3), tens([23, 2000], [2, 3], off=1), tens([19, 23], [1, 2]), 10, 0, "u", 2))
    # a transposed operand: k is not the unit stride (the gather path)
    cs.append(("transposed_B", tens([31, 3000], [1, 3]), tens([3000, 27], [3, 2]), tens([31, 27], [1, 2]), 14, 0, "u", 2))
    cs.append(("transposed_AB", tens([3000, 31], [3, 1]), tens([3000, 27], [3, 2]), tens([31, 27], [1, 2]), 14, 0, "pow2", 2))
    # many chunks per row with 64-row tiles (fused mode 3), a one-chunk and a two-chunk row there too
    cs.append(("tiles64_k900", tens([30, 900], [1, 3]), tens([26, 900], [2, 3]), tens([30, 26], [1, 2]), 14, 0, "pow2", 3))
    cs.append(("tiles64_k100", tens([30, 100], [1, 3]), tens([26, 100], [2, 3]), tens([30, 26], [1, 2]), 14, 0, "u", 3))
    # special values: a zero row, Inf, NaN, subnormals, and a row of all non-finite
    cs.append(("specials", tens([24, 4100], [1, 3]), tens([20, 4100], [2, 3]), tens([24, 20], [1, 2]), 14, 0, "pow2", 2, "specials"))
    # issue 21 task 20 (klcs): long rows (nc = 4; with specials klf64 falls back to the exact passes) with the special values
    cs.append(("specials_long", tens([12, 12300], [1, 3]), tens([11, 12300], [2, 3]), tens([12, 11], [1, 2]), 14, 0, "pow2", 2, "specials"))
    if quick:
        cs = [c for c in cs if c[0] in ("k4096_chunk1", "k1000_tail", "k8192_two_chunks", "odd_pitch_A", "offset_ptr_A", "transposed_B", "specials", "g4_n14",
                                        "k8300_three_chunks", "specials_long")]
    return cs


def make_data(name, X, seed, family, specials):
    n = ext_len(X["extent"], X["stride"])
    host = T.fill(n, seed, family)
    if specials and X is not None and seed != 9:
        k = X["extent"][-1] if seed == 3 else X["extent"][-1]
        rows = X["extent"][0]
        s0 = X["stride"][0]
        host[3 * s0:3 * s0 + k] = 0.0                       # a zero row
        host[5 * s0 + 7] = np.inf
        host[6 * s0 + 9] = np.nan
        host[7 * s0 + 1] = 5e-324                           # subnormals
        host[7 * s0 + 2] = -2.2250738585072009e-308
        host[8 * s0:8 * s0 + k] = np.where(np.arange(k) % 2 == 0, np.inf, np.nan)   # all non-finite
        host[9 * s0:9 * s0 + k] = np.where(np.arange(k) % 3 == 0, 1e-310, 0.0)      # subnormal-only row
    return host


def run(lib, cuda, case, bits, fmode, counts_wanted=True):
    name, A, B, Cd, N, g, family, fm = case[:8]
    specials = len(case) > 8
    T3 = {X: hz.make_tensor(t["extent"], t["stride"], t["label"]) for X, t in (("A", A), ("B", B), ("C", Cd))}
    lib.set_seq(bits)
    lib.set_fused(fmode)
    try:
        st, plan = lib.plan_create(T3["A"], T3["B"], T3["C"], N, g)
    finally:
        lib.set_fused(-1)
    if st != "SUCCESS":
        raise RuntimeError(f"{name}: imd_plan_create {st}")
    st, ws_bytes = lib.workspace_size(T3["A"], T3["B"], T3["C"], N, g)
    k_hat = lib.plan_sizes(plan)[7]
    bufs = {}
    try:
        for X, t, seed in (("A", A, 3), ("B", B, 6), ("C", Cd, 9)):
            host = make_data(name, t, seed, family if X != "C" else "u", specials)
            base = cuda.malloc((host.size + t["off"]) * 8 + 64)
            addr = (base.value or 0) + t["off"] * 8
            cuda.upload_raw(addr, host)
            bufs[X] = (base, addr, host)
        ws = cuda.malloc(max(ws_bytes, 1))
        lib.kl_launch_counts(reset=True)
        st = lib.contract(plan, 1.0, C.c_void_p(bufs["A"][1]), C.c_void_p(bufs["B"][1]), 0.0, C.c_void_p(bufs["C"][1]), ws, ws_bytes)
        cuda.sync()
        seen = lib.kl_launch_counts(reset=True)
        cuda.free(ws)
        if st != "SUCCESS":
            raise RuntimeError(f"{name}: imd_contract {st}")
        out = np.empty(bufs["C"][2].size, dtype=np.float64)
        cuda.download_raw(out, bufs["C"][1])
        return T.hash_c(out), seen, k_hat
    finally:
        for p, _, _ in bufs.values():
            cuda.free(p)
        lib.plan_destroy(plan)


def expect(case, bits, fmode, k_hat):
    """The KL launches {(E, Kb): count} a set must produce: one per operand."""
    out = {}
    for X in (case[1], case[2]):
        unit = X["stride"][-1] == 1 and X["label"][-1] == 3   # k is the last mode, read along k-bar
        # chunks per row: tk = 4096 along k-bar (the whole tile), 512 when the rows are read side by side (k not unit-stride), 64 with the 64-row tiles of mode 3
        nc = -(-k_hat // 64) if fmode == 3 else max(1, -(-k_hat // (4096 if unit else 512)))
        aligned = (X["off"] % 2 == 0)
        direct = unit and aligned
        if nc > 2:   # kllong (512) takes the direct loads here, klload (32) only for the rows held on chip
            e, kb = 4, (1 if bits & 16 else 0) | (2 if (bits & 512) and direct else 0) | (4 if bits & 256 else 0)
            if kb == 7 and case[4] <= 16:   # klcs (4096): only on Kb 7 with E = 4, W = 2 (N <= 16)
                kb |= 8 if bits & 4096 else 0
        else:
            e = 16 if bits & 64 else 8
            kb = (1 if bits & 16 else 0) | (2 if (bits & 32) and direct else 0) | (4 if bits & 256 else 0)
        out[(e, kb)] = out.get((e, kb), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-seq-twins", action="store_true", help="built with IMMERDINK_SEQ_TWINS=OFF: the KL bits are not supported; checks only that they are refused")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("FAIL  the library has no test hooks")
        return 1
    if args.no_seq_twins:
        ok = all(lib.lib.imd_test_set_seq(b) != 0 for b in (16, 32, 64, 113))
        print("PASS  the KL bits are refused without the twins" if ok else "FAIL  a KL bit was accepted without the twins")
        return 0 if ok else 1
    bad, n = [], 0
    emit_sets = (1,) if args.quick else (0, 1, 5, 7)
    try:
        for case in cases(args.quick):
            fmode = case[7]
            ref = None
            # klcs / klph (issue 21 task 20) on the sets that take them (klwarp + klload + klf64 + kllong) and on one that does not (klwarp + kllong);
            # klph is a runtime delay (no Kb bit): the same kernel runs, the bits must not change
            extra = [(base, 27 | x) for base in emit_sets for x in (32, 64, 96)] + [(emit_sets[-1], 17 | 96)]
            for base, kl in list(itertools.product(emit_sets, range(32))) + extra:
                if kl & 8 and not kl & 1:
                    continue                                  # klf64 needs klwarp (the hook refuses it alone)
                bits = base | (16 if kl & 1 else 0) | (32 if kl & 2 else 0) | (64 if kl & 4 else 0) | (256 if kl & 8 else 0) | (512 if kl & 16 else 0) | \
                    (4096 if kl & 32 else 0) | (8192 if kl & 64 else 0)
                h, seen, k_hat = run(lib, cuda, case, bits, fmode)
                n += 1
                ref = ref or h
                if h != ref:
                    bad.append(f"{case[0]} seq {bits}: hash {h}, reference {ref} (seq {emit_sets[0]})")
                want = expect(case, bits, fmode, k_hat)
                if seen != want:
                    bad.append(f"{case[0]} seq {bits}: KL launches {seen}, expected {want}")
    finally:
        lib.set_seq(-1)
        lib.set_fused(-1)
    if bad:
        print("FAIL")
        for b in bad[:60]:
            print("  " + b)
        return 1
    print(f"PASS  {n} runs: every combination of klwarp / klload / klshape / klf64 / kllong (and klcs / klph on the long-row sets) over the emitter sets {emit_sets} is bit-identical, and each operand ran the kernel the set asks for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
