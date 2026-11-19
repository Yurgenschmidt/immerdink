#!/usr/bin/env python3
"""IMD_HWGEMM stage 2 workspace (DESIGN v1.21 §6i.2, issue #75): exact bytes, GPU-free.

    python3 tests/test_hw_workspace.py [--host-lib build/libimmerdink_host.so]

What this checks. When the hardware FP64 GEMM runs (mode HWGEMM, non-empty output, k >= 2),
the workspace holds only what that GEMM path needs (spec/IndexMap.tla HwCells):
  vec  the scatter vectors, 8 (2m + 2n + 2k + 3b) bytes, iff some operand is packed, C is
       stored through its vectors or the batch has several modes, and the vectors are not
       plan-resident (IMD_VEC_RESIDENT);
  PA   the FP64 pack of the GEMM's A, 8 b m k_hat bytes, iff A is not affine;
  PB   the FP64 pack of the GEMM's B, 8 b n k_hat bytes, iff B is not affine;
with k_hat = k rounded up to 16 (HWGEMM's own K tile, not EMULATE's k_align), regions placed
in that order at 256-byte aligned offsets, and every EMULATE region absent. "Affine" is the
§6i.2 rule: each of the operand's two index groups is one fused mode (or empty), the strides
are >= 0 and one is exactly 1.

The expectations are written out by hand per case (the operand that is not affine is built
that way on purpose), not recomputed with the library's own logic. Only libimmerdink_host is
loaded (no CUDA).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402

ALIGN = 256
HWQ = 16


def up(x, q):
    return (x + q - 1) // q * q


def expect(m, n, k, b, vec, pa, pb, slab=None):
    """{region: (offset, bytes)} and total for the HWGEMM layout of a lowered m, n, k, b.
    slab: the K-slab length L (§6i.3): the packs hold one slab, ceil(L)_16 columns; the
    vectors still describe all of k."""
    kp = k if slab is None else min(slab, k)
    sizes = [("vec", 8 * (2 * m + 2 * n + 2 * k + 3 * b) if vec else 0),
             ("PA", 8 * b * m * up(kp, HWQ) if pa else 0),
             ("PB", 8 * b * n * up(kp, HWQ) if pb else 0)]
    regs, off = {}, 0
    for name, nb in sizes:
        if nb:
            regs[name] = (off, nb)
            off += up(nb, ALIGN)
    return regs, off


# (name, spec, extents, strides (sA, sB, sC) or None, packs of the CALLER's (A, B), C scattered
# or multi-mode batch (vectors needed), vec_mode)
CASES = [
    ("affine", "ik,jk->ij", dict(i=5, j=6, k=7), None, (False, False), False, "GENERATE"),
    ("affine_k16", "ik,jk->ij", dict(i=5, j=6, k=16), None, (False, False), False, "GENERATE"),
    # A (5 x 7) at i*2 + k*10: no unit stride -> packed
    ("pack_a", "ik,jk->ij", dict(i=5, j=6, k=7), ([2, 10], None, None), (True, False), True,
     "GENERATE"),
    ("pack_a_k16", "ik,jk->ij", dict(i=5, j=6, k=16), ([2, 10], None, None), (True, False), True,
     "GENERATE"),
    ("pack_a_k17", "ik,jk->ij", dict(i=5, j=6, k=17), ([2, 10], None, None), (True, False), True,
     "GENERATE"),
    ("pack_b", "ik,jk->ij", dict(i=9, j=4, k=33), (None, [3, 12], None), (False, True), True,
     "GENERATE"),
    ("pack_both", "ik,jk->ij", dict(i=5, j=6, k=7), ([2, 10], [2, 12], None), (True, True), True,
     "GENERATE"),
    # K split over two modes the lowering cannot fuse for A (k stride 1, l stride 9 != 4)
    ("pack_a_two_k", "ikl,jkl->ij", dict(i=3, j=4, k=4, l=2), ([20, 1, 9], None, None),
     (True, False), True, "GENERATE"),
    # C (5 x 6) at i*14 + j*2: scattered store, operands affine
    ("c_scatter", "ik,jk->ij", dict(i=5, j=6, k=7), (None, None, [14, 2]), (False, False), True,
     "GENERATE"),
    # batch modes b, c that do not fuse in A (b stride 67 != 3 * 20): per-lambda offsets
    ("multi_batch", "bcik,bcjk->bcij", dict(b=2, c=3, i=4, j=5, k=5), ([67, 20, 5, 1], None, None),
     (False, False), True, "GENERATE"),
    ("one_batch", "bik,bjk->bij", dict(b=6, i=4, j=5, k=5), None, (False, False), False,
     "GENERATE"),
    # C "ji" (i fastest): the lowering swaps roles, so the caller's packed A is the GEMM's B
    ("pack_a_swapped", "ik,jk->ji", dict(i=5, j=6, k=7), ([2, 10], None, None), (True, False),
     True, "GENERATE"),
    ("pack_b_swapped", "ik,jk->ji", dict(i=9, j=4, k=33), (None, [3, 12], None), (False, True),
     True, "GENERATE"),
    ("pack_a_resident", "ik,jk->ij", dict(i=5, j=6, k=7), ([2, 10], None, None), (True, False),
     True, "RESIDENT"),
    ("scatter_resident", "ik,jk->ij", dict(i=5, j=6, k=7), (None, None, [14, 2]), (False, False),
     True, "RESIDENT"),
]

# K slabs (host API 16, DESIGN §6i.3): (name, spec, extents, strides, packs, vec, k_slab, L) with
# L the slab length the library must use: k_slab rounded up to 16, capped at k.
SLAB_CASES = [
    ("slab_both_16", "ik,jk->ij", dict(i=5, j=6, k=100), ([2, 10], [2, 12], None), (True, True),
     True, 16, 16),
    ("slab_both_round", "ik,jk->ij", dict(i=5, j=6, k=100), ([2, 10], [2, 12], None), (True, True),
     True, 17, 32),
    ("slab_a_1", "ik,jk->ij", dict(i=5, j=6, k=100), ([2, 10], None, None), (True, False), True, 1,
     16),
    ("slab_ge_k", "ik,jk->ij", dict(i=5, j=6, k=100), ([2, 10], None, None), (True, False), True,
     100, 100),
    ("slab_huge", "ik,jk->ij", dict(i=5, j=6, k=100), ([2, 10], None, None), (True, False), True,
     1 << 62, 100),
    ("slab_affine", "ik,jk->ij", dict(i=5, j=6, k=100), None, (False, False), False, 32, 32),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-lib", default=os.path.join(
        os.environ.get("IMMERDINK_BUILD_DIR", os.path.join(os.path.dirname(HERE), "build")),
        "libimmerdink_host.so"))
    args = ap.parse_args()
    lib = host.load(args.host_lib)

    fails = 0
    seen_swap = []
    for name, spec, ext, strides, packs, vec, vec_mode in CASES:
        kw = dict(strides=strides, N=8, mode="HWGEMM", vec_mode=vec_mode, lib=lib)
        try:
            lw = host.lowering(spec, ext, **kw)
            if lw.status != "SUCCESS" or lw.dispatch != "HWGEMM":
                raise AssertionError(f"lowering {lw.status} / {lw.dispatch}")
            seen_swap.append(lw.swapped)
            pa, pb = (packs[1], packs[0]) if lw.swapped else packs
            regs, total = expect(lw.m, lw.n, lw.k, lw.b, vec and vec_mode == "GENERATE", pa, pb)
            L = host.workspace_layout(spec, ext, **kw)
            ws = host.workspace_size(spec, ext, **kw)
            if L.regions != regs or L.total != total or ws != total or lw.workspace_bytes != total:
                raise AssertionError(f"layout {L.regions} total {L.total} size {ws} "
                                     f"lowering {lw.workspace_bytes}; want {regs} total {total}")
            want_vec = 2 * lw.m + 2 * lw.n + 2 * lw.k + 3 * lw.b if vec else 0
            if lw.vec_entries != want_vec:
                raise AssertionError(f"vec_entries {lw.vec_entries}, want {want_vec}")
            print(f"ok    {name:<18} m={lw.m} n={lw.n} k={lw.k} b={lw.b} swapped={int(lw.swapped)} "
                  f"workspace {total} B {sorted(regs)}")
        except (AssertionError, host.ImmerdinkError) as e:
            fails += 1
            print(f"FAIL  {name:<18} {e}")
    if not any(seen_swap):
        fails += 1
        print("FAIL  no case exercised the role swap")
    for name, spec, ext, strides, packs, vec, k_slab, L in SLAB_CASES:
        kw = dict(strides=strides, N=8, mode="HWGEMM", lib=lib)
        try:
            lw = host.lowering(spec, ext, k_slab=k_slab, **kw)
            if lw.status != "SUCCESS" or lw.dispatch != "HWGEMM" or lw.swapped:
                raise AssertionError(f"lowering {lw.status} / {lw.dispatch} / swap {lw.swapped}")
            regs, total = expect(lw.m, lw.n, lw.k, lw.b, vec, *packs, slab=L)
            lay = host.workspace_layout(spec, ext, k_slab=k_slab, **kw)
            ws = host.workspace_size(spec, ext, k_slab=k_slab, **kw)
            if lay.regions != regs or lay.total != total or ws != total or \
                    lw.workspace_bytes != total:
                raise AssertionError(f"layout {lay.regions} total {lay.total} size {ws}; want "
                                     f"{regs} total {total}")
            # launches: K0 (vectors needed) + S (packs + one GEMM chunk)
            S = -(-lw.k // L)
            want_l = (1 if vec else 0) + S * (sum(packs) + 1)
            if lw.kernel_launches != want_l:
                raise AssertionError(f"kernel_launches {lw.kernel_launches}, want {want_l}")
            print(f"ok    {name:<18} k={lw.k} k_slab={k_slab} L={L} S={S} workspace {total} B")
        except (AssertionError, host.ImmerdinkError) as e:
            fails += 1
            print(f"FAIL  {name:<18} {e}")
    # k_slab outside HWGEMM, or negative: INVALID_VALUE
    for mode, k_slab in (("EMULATE", 16), ("DIRECT", 16), ("HWGEMM", -16)):
        try:
            host.workspace_size("ik,jk->ij", dict(i=5, j=6, k=100), N=8, mode=mode, k_slab=k_slab,
                                lib=lib)
            fails += 1
            print(f"FAIL  k_slab {k_slab} in {mode}: accepted")
        except host.ImmerdinkError as e:
            if "INVALID_VALUE" not in str(e):
                fails += 1
                print(f"FAIL  k_slab {k_slab} in {mode}: {e}")
    n = len(CASES) + len(SLAB_CASES)
    print(f"{n - fails}/{n} cases ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
