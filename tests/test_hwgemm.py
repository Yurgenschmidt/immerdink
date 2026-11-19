#!/usr/bin/env python3
"""IMD_HWGEMM stages 1 and 2 (DESIGN v1.21 §6i, issue #75): bit-exact GPU tests against DIRECT.

    python3 tests/test_hwgemm.py [--lib build/libimmerdink.so] [--quick]

Why DIRECT is the oracle: an unslabbed HWGEMM tile starts its accumulator at +0 and adds one
fma per K element in ascending canonical order, which is exactly DIRECT's native_dot
(§6i.4; measured on sm_89, H100 and H200 by bench/hwgemm_bitid.cu). With the §3.7 output
scaling done by the same operations (store_out), every C element must match DIRECT bit for
bit (NaN == NaN regardless of payload), and the elements C's descriptor does not address
must stay untouched.

Each case builds strided A, B, C descriptors, fills them with mixed data (wide exponents,
signed zeros, subnormals, and in some cases Inf / NaN), runs a DIRECT plan and a HWGEMM plan
on identical inputs and compares the whole C buffers (the sign of a zero element may differ
when k is not a multiple of the K tile: §6i.4; those elements are counted and reported). It also checks the plan-level
contract: workspace 0 and dispatch HWGEMM for fully affine calls, DIRECT / SCALE dispatch for
K-empty and k = 0, and the batch / column chunking. Stage 2 (suite stage2): packed operands,
scattered C and multi-mode batches against DIRECT (zero signs exact when both operands are
packed: §6i.4's -0 pads); suite own: the own kernel against stage 1's CUTLASS kernel, bit for
bit.
"""

import argparse
import ctypes as C
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

HWGEMM = 2
DISPATCH = {0: "NOOP", 1: "EMULATE", 2: "DIRECT", 3: "SCALE", 4: "HWGEMM"}
TILES = {0: "128x128", 1: "64x128", 2: "64x64"}


class Fail(Exception):
    pass


def setup_hooks(lib):
    L = lib.lib
    L.imd_test_set_hw_tile.argtypes = [C.c_int32]
    L.imd_test_set_hw_tile.restype = C.c_int
    L.imd_test_plan_hw_tile.argtypes = [C.c_void_p, C.POINTER(C.c_int32)]
    L.imd_test_plan_hw_tile.restype = C.c_int
    L.imd_test_set_hw_own.argtypes = [C.c_int32]
    L.imd_test_set_hw_own.restype = None
    L.imd_test_set_hw_own_vec2.argtypes = [C.c_int32]
    L.imd_test_set_hw_own_vec2.restype = None


def plan_tile(lib, plan):
    t = C.c_int32(-9)
    assert lib.lib.imd_test_plan_hw_tile(plan, C.byref(t)) == 0
    return t.value


def device_cc(cuda):
    major = C.c_int(0)
    cuda.rt.cudaDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    cuda.check(cuda.rt.cudaDeviceGetAttribute(C.byref(major), 75, 0), "cc major")  # cudaDevAttrComputeCapabilityMajor
    return major.value


# ---------------------------------------------------------------------------------------
# Descriptors and data
# ---------------------------------------------------------------------------------------

class Desc:
    """A strided tensor: labels, extents, strides (elements, any sign), as an imd_tensor_t."""

    def __init__(self, labels, ext, stride):
        self.labels, self.extent, self.stride = list(labels), list(ext), list(stride)
        self.t = hz.make_tensor(self.extent, self.stride, self.labels)

    def empty(self):
        return any(e == 0 for e in self.extent)

    def span(self):
        """(length, base offset) of the smallest buffer holding every element."""
        if self.empty() or not self.extent:
            return 1, 0
        lo = sum(min(0, s * (e - 1)) for e, s in zip(self.extent, self.stride))
        hi = sum(max(0, s * (e - 1)) for e, s in zip(self.extent, self.stride))
        return hi - lo + 1, -lo


def strides(ext, order, neg=(), gap=None):
    """Dense strides with order[0] fastest; modes in `neg` get a negative sign; gap[d] adds
    padding before mode d."""
    st, acc = [0] * len(ext), 1
    for d in order:
        acc += (gap or {}).get(d, 0)
        st[d] = -acc if d in neg else acc
        acc *= max(ext[d], 1)
    return st


def mixed(rng, n, specials=False):
    """Doubles over a wide exponent range, with signed zeros and subnormals; optionally Inf /
    NaN (a few)."""
    mant = np.array([rng.uniform(1.0, 2.0) for _ in range(n)])
    expo = np.array([rng.randint(-60, 60) for _ in range(n)])
    sign = np.array([rng.choice((-1.0, 1.0)) for _ in range(n)])
    x = sign * np.ldexp(mant, expo)
    for _ in range(max(1, n // 50)):
        x[rng.randrange(n)] = rng.choice((0.0, -0.0, 5e-324, -2.5e-310, 1e-300))
    if specials:
        for v in (np.inf, -np.inf, np.nan):
            x[rng.randrange(n)] = v
    return x


ZERO_SIGN = {"elements": 0}   # zero elements whose sign differs (§6i.4 allowance), counted


def bit_mismatches(got, want, exact_zero=False):
    """Indices where got and want differ in bits, with NaN == NaN (any payload) and, per
    DESIGN §6i.4, +0 == -0: CUTLASS zero-fills a partial K tile with +0 * +0 products, and
    -0 + (+0) = +0, so a zero sum that DIRECT returns as -0 may come back as +0 when k is
    not a multiple of the kernel's K tile (every other value is unchanged by x + (+0))."""
    gb, wb = got.view(np.uint64), want.view(np.uint64)
    zs = (gb != wb) & (got == 0) & (want == 0)
    if exact_zero:  # both operands packed: the -0 / +0 K-tail pads keep the sign (§6i.4)
        zs[:] = False
    ZERO_SIGN["elements"] += int(np.count_nonzero(zs))
    return np.nonzero((gb != wb) & ~(np.isnan(got) & np.isnan(want)) & ~zs)[0]


# ---------------------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------------------

def run_case(lib, cuda, name, A, B, Cd, alphas_betas, rng, expect="HWGEMM", launches=None,
             specials=False, tiles=(-1,), c_fill_nan=False, verbose=False, affine=True,
             exact_zero=False, own=False):
    """Plans DIRECT and HWGEMM for (A, B, C); expect = HWGEMM / DIRECT / SCALE (dispatch of
    the HWGEMM plan) or NOT_SUPPORTED (plan creation). Compares every (alpha, beta).
    affine = False: a stage-2 call (packs, scattered C or a multi-mode batch: workspace may be
    nonzero). exact_zero: zero signs must match too. own: the reference is the same HWGEMM
    plan on stage 1's CUTLASS kernel and the tested run uses the own kernel (hook), bit for
    bit including zero signs."""
    st, ws = lib.workspace_size(A.t, B.t, Cd.t, 14, mode=HWGEMM)
    if expect == "NOT_SUPPORTED":
        if st != "NOT_SUPPORTED":
            raise Fail(f"{name}: workspace_size {st}, want NOT_SUPPORTED")
        st, plan = lib.plan_create(A.t, B.t, Cd.t, 14, mode=HWGEMM)
        if st != "NOT_SUPPORTED":
            raise Fail(f"{name}: plan_create {st}, want NOT_SUPPORTED")
        return 0
    if st != "SUCCESS":
        raise Fail(f"{name}: workspace_size {st}")
    if expect == "HWGEMM" and affine and ws != 0:
        raise Fail(f"{name}: HWGEMM workspace {ws} bytes, want 0 (fully affine)")
    st, wsd = lib.workspace_size(A.t, B.t, Cd.t, 14, mode=hz.DIRECT)
    assert st == "SUCCESS", st

    la, ba = A.span()
    lb, bb = B.span()
    lc, bc = Cd.span()
    hA, hB = mixed(rng, la, specials), mixed(rng, lb, specials)
    hC0 = np.full(lc, np.nan) if c_fill_nan else mixed(rng, lc)
    dA, dB, dC = cuda.malloc(8 * la), cuda.malloc(8 * lb), cuda.malloc(8 * lc)
    dW = cuda.malloc(max(ws, wsd, 256))
    cuda.upload(dA, hA)
    cuda.upload(dB, hB)
    off = lambda p, base: C.c_void_p(p.value + 8 * base)  # noqa: E731
    compared = 0
    try:
        st, pd = lib.plan_create(A.t, B.t, Cd.t, 14, mode=hz.DIRECT)
        assert st == "SUCCESS", f"{name}: DIRECT plan {st}"
        for tile in tiles:
            lib.lib.imd_test_set_hw_tile(tile)
            st, ph = lib.plan_create(A.t, B.t, Cd.t, 14, mode=HWGEMM)
            lib.lib.imd_test_set_hw_tile(-1)
            if st != "SUCCESS":
                raise Fail(f"{name}: HWGEMM plan {st}")
            try:
                got_tile = plan_tile(lib, ph)
                if expect == "HWGEMM" and tile >= 0 and got_tile != tile:
                    raise Fail(f"{name}: plan tile {got_tile}, want {tile}")
                if expect != "HWGEMM" and got_tile != -1:
                    raise Fail(f"{name}: {expect} plan reports tile {got_tile}")
                for alpha, beta in alphas_betas:
                    want, got = np.empty(lc), np.empty(lc)
                    runs = (((ph, want, ws, 0), (ph, got, ws, 1)) if own
                            else ((pd, want, wsd, 0), (ph, got, ws, 0)))
                    for plan, out, wsb, own_on in runs:
                        cuda.upload(dC, hC0)
                        lib.lib.imd_test_set_hw_own(own_on)
                        try:
                            s = lib.contract(plan, alpha, off(dA, ba), off(dB, bb), beta,
                                             off(dC, bc), dW if wsb else None, wsb)
                        finally:
                            lib.lib.imd_test_set_hw_own(0)
                        if s != "SUCCESS":
                            raise Fail(f"{name}: contract {s} (alpha {alpha}, beta {beta})")
                        cuda.sync()
                        cuda.download(out, dC)
                    bad = bit_mismatches(got, want, exact_zero=exact_zero or own)
                    if len(bad):
                        i = bad[0]
                        raise Fail(f"{name} tile {TILES.get(got_tile, got_tile)} alpha {alpha} "
                                   f"beta {beta}: {len(bad)} of {lc} elements differ, first at "
                                   f"{i}: {got[i]!r} vs {'stage 1' if own else 'DIRECT'} {want[i]!r}")
                    compared += lc
            finally:
                lib.plan_destroy(ph)
        lib.plan_destroy(pd)
    finally:
        for p in (dA, dB, dC, dW):
            cuda.free(p)
    if verbose:
        print(f"  ok  {name}: {compared} elements")
    return compared


# ---------------------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------------------

AB_FULL = [(1.0, 0.0), (1.0, 1.0), (-0.5, 0.0), (3.0, -2.0), (0.0, 0.5), (0.0, 0.0)]
AB_QUICK = [(1.0, 0.0), (3.0, -2.0), (0.0, 0.5)]


def gemm_case(i_, j_, k_, b_, lay_a, lay_b, lay_c, batch=False, gap=0, bneg=False):
    """An (optionally batched) GEMM with per-tensor layouts: 'r' = second index contiguous
    (A: K, B: J... see below), 'c' = first index contiguous. A is (i, k), B is (k, j), C is
    (i, j); a batch label l is outermost (negative in C when bneg)."""
    I, J, K, L = 1, 2, 3, 4

    def mk(labels, ext, lay, inner_last):
        # lay 'r': the LAST of the two matrix modes fastest; 'c': the first.
        order = [1, 0] if lay == "r" else [0, 1]
        if batch:
            labels, ext = [L] + labels, [b_] + ext
            order = [d + 1 for d in order] + [0]
        g = {order[1]: gap} if gap else None
        neg = (0,) if (batch and bneg and inner_last) else ()
        return Desc(labels, ext, strides(ext, order, neg=neg, gap=g))

    A = mk([I, K], [i_, k_], lay_a, False)
    B = mk([K, J], [k_, j_], lay_b, False)
    Cd = mk([I, J], [i_, j_], lay_c, True)
    return A, B, Cd


def suite_layouts(lib, cuda, rng, ab, quick, verbose):
    """Every A / B / C layout combination (the four CUTLASS layout pairs plus the C^T swap,
    and the lowering's own role swap), with odd shapes, padding and batches."""
    n = 0
    shapes = [(64, 64, 16, 1), (129, 65, 1000, 1), (200, 136, 333, 3)]
    if not quick:
        shapes += [(1, 70, 33, 2), (70, 1, 33, 2), (37, 41, 2, 5), (256, 256, 4097, 1)]
    for (i_, j_, k_, b_) in shapes:
        for la in "rc":
            for lb in "rc":
                for lc in "rc":
                    A, B, Cd = gemm_case(i_, j_, k_, b_, la, lb, lc, batch=b_ > 1,
                                         gap=3 if (i_ + j_) % 2 else 0, bneg=lc == "c")
                    n += run_case(lib, cuda, f"gemm {i_}x{j_}x{k_} b{b_} A{la} B{lb} C{lc}",
                                  A, B, Cd, ab, rng, verbose=verbose)
    return n


def suite_values(lib, cuda, rng, ab, quick, verbose):
    """Inf / NaN propagation; beta = 0 must not read C (C prefilled with NaN)."""
    n = 0
    A, B, Cd = gemm_case(96, 80, 257, 2, "r", "c", "r", batch=True)
    n += run_case(lib, cuda, "specials", A, B, Cd, ab, rng, specials=True, verbose=verbose)
    n += run_case(lib, cuda, "beta 0, C NaN", A, B, Cd, [(1.0, 0.0), (-2.0, 0.0), (0.0, 0.0)],
                  rng, c_fill_nan=True, verbose=verbose)
    return n


def suite_tensor(lib, cuda, rng, ab, quick, verbose):
    """Tensor-shaped affine calls: fused multi-label groups, broadcast (stride-0) batch, a
    C-only label, k = 1 (DIRECT) and k = 0 (SCALE) in HWGEMM mode, and non-affine calls
    (NOT_SUPPORTED in stage 1)."""
    I1, I2, J1, K1, K2, L = 1, 2, 3, 4, 5, 6
    n = 0
    # (i1 i2) fuse in A and C, (k1 k2) in A and B: one mode each -> affine
    A = Desc([K1, K2, I1, I2], [5, 7, 3, 11], strides([5, 7, 3, 11], [0, 1, 2, 3]))
    B = Desc([J1, K1, K2], [13, 5, 7], strides([13, 5, 7], [1, 2, 0]))
    Cd = Desc([I1, I2, J1], [3, 11, 13], strides([3, 11, 13], [2, 0, 1]))
    n += run_case(lib, cuda, "fused groups", A, B, Cd, ab, rng, verbose=verbose)
    # broadcast batch: A's batch stride 0
    sa = strides([9, 17], [1, 0])
    A = Desc([L, I1, K1], [4, 9, 17], [0] + sa)
    B = Desc([L, K1, J1], [4, 17, 6], strides([4, 17, 6], [2, 1, 0]))
    Cd = Desc([L, I1, J1], [4, 9, 6], strides([4, 9, 6], [2, 1, 0]))
    n += run_case(lib, cuda, "broadcast batch (A stride 0)", A, B, Cd, ab, rng, verbose=verbose)
    # K-empty: outer product -> DIRECT dispatch also in HWGEMM mode
    A = Desc([I1], [40], [1])
    B = Desc([J1], [30], [1])
    Cd = Desc([I1, J1], [40, 30], [30, 1])
    n += run_case(lib, cuda, "outer product (DIRECT)", A, B, Cd, ab, rng, expect="DIRECT",
                  verbose=verbose)
    # k = 0 -> SCALE
    A = Desc([I1, K1], [8, 0], [1, 8])
    B = Desc([K1, J1], [0, 5], [5, 1])
    Cd = Desc([I1, J1], [8, 5], [5, 1])
    n += run_case(lib, cuda, "k = 0 (SCALE)", A, B, Cd, ab, rng, expect="SCALE", verbose=verbose)
    # not affine: K split into two non-fusable modes in A (a gap) -> A packed -> stage 1 NO
    A = Desc([I1, K1, K2], [6, 4, 5], [1, 7, 40])
    B = Desc([K1, K2, J1], [4, 5, 3], strides([4, 5, 3], [0, 1, 2]))
    Cd = Desc([I1, J1], [6, 3], [3, 1])
    n += run_case(lib, cuda, "A not affine", A, B, Cd, ab, rng, affine=False, verbose=verbose)
    # not affine: negative K stride
    A = Desc([I1, K1], [6, 9], [9, -1])
    B = Desc([K1, J1], [9, 3], [3, 1])
    Cd = Desc([I1, J1], [6, 3], [3, 1])
    n += run_case(lib, cuda, "negative K stride", A, B, Cd, ab, rng, affine=False,
                  verbose=verbose)
    # not affine: two batch modes that do not fuse
    # (A's gap of 1 before mode L keeps (L, L+1) from fusing into one batch mode)
    A = Desc([L, L + 1, I1, K1], [2, 3, 4, 5], strides([2, 3, 4, 5], [3, 2, 1, 0], gap={0: 1}))
    B = Desc([L + 1, L, K1, J1], [3, 2, 5, 6], strides([3, 2, 5, 6], [3, 2, 0, 1]))
    Cd = Desc([L, L + 1, I1, J1], [2, 3, 4, 6], strides([2, 3, 4, 6], [3, 2, 1, 0]))
    n += run_case(lib, cuda, "multi-mode batch", A, B, Cd, ab, rng, affine=False,
                  verbose=verbose)
    return n


def suite_stage2(lib, cuda, rng, ab, quick, verbose, cc_major):
    """Stage 2 (§6i.2): packed operands (one or both; both: GEMM on k_hat with -0 / +0 pads,
    zero signs exact), scattered C, multi-mode batches, their combinations, every tile, with
    k on and off the K tile."""
    I1, J1, K1, K2, L = 1, 2, 3, 4, 6
    tiles = (0, 1, 2) if cc_major >= 9 else (2,)
    n = 0
    shapes = [(6, 3, 9, 1), (70, 33, 17, 1), (130, 200, 64, 2)]
    if not quick:
        shapes += [(257, 129, 1001, 1), (65, 66, 2, 3)]
    for (i_, j_, k_, b_) in shapes:
        ext = dict(i=i_, j=j_, k=k_)
        # A packed: no unit stride (i stride 2, k stride 2 i_ + 1)
        pa = [2, 2 * i_ + 1]
        # B packed: K stride 3, J stride 3 k_ + 2
        pb = [3, 3 * k_ + 2]
        # C scattered: i stride 2, j stride 2 i_ + 3
        pc = [2, 2 * i_ + 3]
        for what, sa, sb, sc in (("A packed", pa, None, None), ("B packed", None, pb, None),
                                 ("both packed", pa, pb, None), ("C scattered", None, None, pc),
                                 ("all three", pa, pb, pc)):
            def mk(labels, ex, st, fallback):
                if b_ > 1:  # batch outermost, stride past the matrix
                    span = 1 + sum(abs(x) * (e - 1) for x, e in zip(st or fallback, ex))
                    return Desc([L] + labels, [b_] + ex, [span + 5] + list(st or fallback))
                return Desc(labels, ex, list(st or fallback))
            A = mk([I1, K1], [i_, k_], [pa[0], pa[1]] if sa else None, [k_, 1])
            B = mk([K1, J1], [k_, j_], [pb[0], pb[1]] if sb else None, [j_, 1])
            Cd = mk([I1, J1], [i_, j_], [pc[0], pc[1]] if sc else None, [j_, 1])
            n += run_case(lib, cuda, f"{what} {i_}x{j_}x{k_} b{b_}", A, B, Cd, ab, rng,
                          affine=False, exact_zero=bool(sa and sb), tiles=tiles, verbose=verbose)
    # multi-mode batch (A in place, offsets from the vectors) with B packed and C scattered
    A = Desc([L, L + 1, I1, K1], [2, 3, 40, 21], strides([2, 3, 40, 21], [3, 2, 1, 0], gap={0: 1}))
    B = Desc([L + 1, L, K1, J1], [3, 2, 21, 35], [3 * 21 * 35 * 2 + 7, 21 * 35 + 1, 1, 21 + 2])
    Cd = Desc([L, L + 1, I1, J1], [2, 3, 40, 35], strides([2, 3, 40, 35], [2, 3, 1, 0]))
    n += run_case(lib, cuda, "multi-batch + B packed + C cols", A, B, Cd, ab, rng, affine=False,
                  tiles=tiles, verbose=verbose)
    # K split over two non-fusable modes in both operands (both packed), specials
    A = Desc([I1, K1, K2], [33, 4, 5], [1, 37, 160])
    B = Desc([K1, K2, J1], [4, 5, 19], [1, 9, 47])
    Cd = Desc([I1, J1], [33, 19], [19, 1])
    n += run_case(lib, cuda, "two K modes, both packed, specials", A, B, Cd, ab, rng,
                  affine=False, exact_zero=True, specials=True, verbose=verbose)
    n += run_case(lib, cuda, "both packed, beta 0, C NaN", A, B, Cd,
                  [(1.0, 0.0), (0.0, 0.0)], rng, affine=False, exact_zero=True,
                  c_fill_nan=True, verbose=verbose)
    # the tiled pack along a row mode that is not the first fused one (pack_pivot radix 7):
    # C orders I2 before I1, A has I1 fastest (stride 1) and K next (stride 5); not fusable
    I2 = 5
    A = Desc([I1, I2, K1], [5, 7, 9], [1, 48, 5])
    B = Desc([K1, J1], [9, 6], [6, 1])
    Cd = Desc([I1, I2, J1], [5, 7, 6], [42, 6, 1])
    n += run_case(lib, cuda, "A packed along its 2nd row mode (radix 7)", A, B, Cd, ab, rng,
                  affine=False, tiles=tiles, verbose=verbose)
    # multi-mode batch beyond one grid.z
    A = Desc([L, L + 1, I1, K1], [7, 10001, 2, 3], [1, 8, 7 * 10001 * 8, 7 * 10001 * 8 * 2 + 1])
    B = Desc([L, L + 1, K1, J1], [7, 10001, 3, 2], strides([7, 10001, 3, 2], [3, 2, 1, 0]))
    Cd = Desc([L, L + 1, I1, J1], [7, 10001, 2, 2], strides([7, 10001, 2, 2], [3, 2, 1, 0]))
    n += run_case(lib, cuda, "70007 batches, A packed", A, B, Cd, [(1.0, 0.0), (3.0, -2.0)], rng,
                  affine=False, verbose=verbose)
    return n


def run_slab_case(lib, cuda, name, A, B, Cd, kmode, k_slab, L, alphas_betas, rng, tiles=(-1,),
                  exact_zero=False, verbose=False):
    """K slabs (host API 16, DESIGN §6i.3): HWGEMM with k_slab against the slab chain written
    out with DIRECT, one plan per slab on the sub-descriptors of A and B (K mode `kmode`, one
    per operand: its index in A and in B) from K position h0 = s L: slab 0 with the caller's
    beta, the later slabs with beta = 1, alpha on each (alpha = 0: the beta-only call, once).
    Bit-identical (zero signs when exact_zero). Also: workspace_size(k_slab) <= unslabbed."""
    ia, ib = kmode
    k = A.extent[ia]
    st, ws = lib.workspace_size(A.t, B.t, Cd.t, 14, mode=HWGEMM, k_slab=k_slab)
    if st != "SUCCESS":
        raise Fail(f"{name}: workspace_size(k_slab {k_slab}) {st}")
    st, ws1 = lib.workspace_size(A.t, B.t, Cd.t, 14, mode=HWGEMM)
    assert st == "SUCCESS", st
    if ws > ws1:
        raise Fail(f"{name}: slabbed workspace {ws} > unslabbed {ws1}")
    slabs = []
    for h0 in range(0, k, L):
        ln = min(L, k - h0)
        ea, eb = list(A.extent), list(B.extent)
        ea[ia], eb[ib] = ln, ln
        sa, sb = Desc(A.labels, ea, A.stride), Desc(B.labels, eb, B.stride)
        st, wsd = lib.workspace_size(sa.t, sb.t, Cd.t, 14, mode=hz.DIRECT)
        assert st == "SUCCESS", st
        slabs.append((h0, sa, sb, wsd))
    la, ba = A.span()
    lb, bb = B.span()
    lc, bc = Cd.span()
    hA, hB, hC0 = mixed(rng, la), mixed(rng, lb), mixed(rng, lc)
    wmax = max([ws, 256] + [w for _, _, _, w in slabs])
    dA, dB, dC, dW = cuda.malloc(8 * la), cuda.malloc(8 * lb), cuda.malloc(8 * lc), cuda.malloc(wmax)
    cuda.upload(dA, hA)
    cuda.upload(dB, hB)
    off = lambda p, base: C.c_void_p(p.value + 8 * base)  # noqa: E731
    compared, plans = 0, []
    try:
        for h0, sa, sb, wsd in slabs:
            st, pd = lib.plan_create(sa.t, sb.t, Cd.t, 14, mode=hz.DIRECT)
            assert st == "SUCCESS", f"{name}: DIRECT slab plan {st}"
            plans.append((pd, h0, wsd))
        for tile in tiles:
            lib.lib.imd_test_set_hw_tile(tile)
            st, ph = lib.plan_create(A.t, B.t, Cd.t, 14, mode=HWGEMM, k_slab=k_slab)
            lib.lib.imd_test_set_hw_tile(-1)
            if st != "SUCCESS":
                raise Fail(f"{name}: HWGEMM plan (k_slab {k_slab}) {st}")
            try:
                for alpha, beta in alphas_betas:
                    want, got = np.empty(lc), np.empty(lc)
                    cuda.upload(dC, hC0)
                    for s, (pd, h0, wsd) in enumerate(plans if alpha != 0.0 else plans[:1]):
                        r = lib.contract(pd, alpha, off(dA, ba + h0 * A.stride[ia]),
                                         off(dB, bb + h0 * B.stride[ib]), beta if s == 0 else 1.0,
                                         off(dC, bc), dW if wsd else None, wsd)
                        assert r == "SUCCESS", f"{name}: DIRECT slab {s}: {r}"
                    cuda.sync()
                    cuda.download(want, dC)
                    cuda.upload(dC, hC0)
                    r = lib.contract(ph, alpha, off(dA, ba), off(dB, bb), beta, off(dC, bc),
                                     dW if ws else None, ws)
                    if r != "SUCCESS":
                        raise Fail(f"{name}: contract {r} (alpha {alpha}, beta {beta})")
                    cuda.sync()
                    cuda.download(got, dC)
                    bad = bit_mismatches(got, want, exact_zero=exact_zero)
                    if len(bad):
                        i = bad[0]
                        raise Fail(f"{name} k_slab {k_slab} tile {tile} alpha {alpha} beta {beta}: "
                                   f"{len(bad)} of {lc} elements differ, first at {i}: "
                                   f"{got[i]!r} vs DIRECT slab chain {want[i]!r}")
                    compared += lc
            finally:
                lib.plan_destroy(ph)
    finally:
        for pd, _, _ in plans:
            lib.plan_destroy(pd)
        for p in (dA, dB, dC, dW):
            cuda.free(p)
    if verbose:
        print(f"  ok  {name} (k_slab {k_slab}, {len(slabs)} slabs): {compared} elements")
    return compared


def suite_slabs(lib, cuda, rng, ab, quick, verbose, cc_major):
    """K slabs (§6i.3) on both paths: stage 1 (fully affine, one GemmUniversal per slab, A and
    B from the slab's first K position) and stage 2 (per slab the packs into the slab-sized
    PA / PB, then the own GEMM), against the DIRECT slab chain; k_slab rounded up to 16 and
    a last slab shorter than L."""
    I1, J1, K1, L = 1, 2, 3, 6
    tiles = (0, 2) if cc_major >= 9 else (2,)
    n = 0
    for (i_, j_, k_, b_) in [(70, 33, 100, 1), (130, 65, 257, 2)][: 1 if quick else 2]:
        def mk(labels, ex, st):
            if b_ > 1:
                span = 1 + sum(abs(x) * (e - 1) for x, e in zip(st, ex))
                return Desc([L] + labels, [b_] + ex, [span + 5] + list(st))
            return Desc(labels, ex, list(st))
        ka = 1 if b_ == 1 else 2  # index of K in A (I1, K1) / B (K1, J1), after the batch mode
        kb = 0 if b_ == 1 else 1
        aff_a, aff_b, aff_c = [k_, 1], [1, k_], [j_, 1]       # A row, B col (K fastest), C row
        pa, pb, pc = [2, 2 * i_ + 1], [3, 3 * k_ + 2], [2, 2 * i_ + 3]
        for what, sa, sb, sc, ez in (("affine", aff_a, aff_b, aff_c, False),
                                     ("A packed", pa, aff_b, aff_c, False),
                                     ("both packed", pa, pb, aff_c, True),
                                     ("both packed, C scattered", pa, pb, pc, True)):
            A = mk([I1, K1], [i_, k_], sa)
            B = mk([K1, J1], [k_, j_], sb)
            Cd = mk([I1, J1], [i_, j_], sc)
            for k_slab, Lr in ((16, 16), (40, 48)):
                n += run_slab_case(lib, cuda, f"slabs {what} {i_}x{j_}x{k_} b{b_}", A, B, Cd,
                                   (ka, kb), k_slab, Lr, ab, rng, tiles=tiles, exact_zero=ez,
                                   verbose=verbose)
    return n


def suite_own(lib, cuda, rng, ab, quick, verbose, cc_major):
    """The own kernel (hw_own.cuh) on fully affine calls equals stage 1's CUTLASS
    GemmUniversal bit for bit (zero signs included) on every tile and layout pair: the probe
    of the epilogue's accumulator coordinates."""
    tiles = (0, 1, 2) if cc_major >= 9 else (2,)
    n = 0
    for (i_, j_, k_, b_) in ((64, 64, 16, 1), (129, 65, 33, 1), (200, 136, 333, 3)):
        for la in "rc":
            for lb in "rc":
                for lc in "rc":
                    A, B, Cd = gemm_case(i_, j_, k_, b_, la, lb, lc, batch=b_ > 1)
                    n += run_case(lib, cuda, f"own {i_}x{j_}x{k_} b{b_} A{la} B{lb} C{lc}", A, B,
                                  Cd, ab, rng, tiles=tiles, own=True, verbose=verbose)
    # the element-wise epilogue store (the paired 16-byte store off: the timing knob)
    lib.lib.imd_test_set_hw_own_vec2(0)
    try:
        A, B, Cd = gemm_case(129, 65, 33, 1, "r", "r", "r", batch=False)
        n += run_case(lib, cuda, "own 129x65x33 element-wise C stores", A, B, Cd, ab, rng,
                      tiles=tiles, own=True, verbose=verbose)
    finally:
        lib.lib.imd_test_set_hw_own_vec2(1)
    return n


def suite_chunks(lib, cuda, rng, ab, quick, verbose):
    """Batches beyond one grid.z (65535) and columns beyond one grid.y chunk (65535 * 64)."""
    n = 0
    A, B, Cd = gemm_case(2, 3, 5, 70000, "r", "r", "r", batch=True)
    n += run_case(lib, cuda, "70000 batches", A, B, Cd, [(1.0, 0.0), (3.0, -2.0)], rng,
                  verbose=verbose)
    if not quick:
        # DIRECT's scatter vectors are 67 MB here, just over harness.plan_create's 64 MiB guard
        saved, lib.MAX_PLAN_VEC = lib.MAX_PLAN_VEC, 128 << 20
        try:
            A, B, Cd = gemm_case(1, 65535 * 64 + 100, 2, 1, "r", "r", "r")
            n += run_case(lib, cuda, "4.19M columns", A, B, Cd, [(1.0, 1.0)], rng,
                          verbose=verbose)
        finally:
            lib.MAX_PLAN_VEC = saved
    return n


def suite_tiles(lib, cuda, rng, ab, quick, verbose, cc_major):
    """Every tile the device has (sm_90+: three; sm_80 / sm_89: 64x64) is bit-identical."""
    tiles = (0, 1, 2) if cc_major >= 9 else (2,)
    n = 0
    for shape in ((130, 260, 77, 2), (300, 129, 1000, 1)):
        A, B, Cd = gemm_case(*shape, "c", "r", "r", batch=shape[3] > 1)
        n += run_case(lib, cuda, f"tiles {shape}", A, B, Cd, ab, rng, tiles=tiles, verbose=verbose)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=75)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    lib = hz.Library(args.lib)
    setup_hooks(lib)
    cuda = hz.Cuda()
    cc = device_cc(cuda)
    rng = random.Random(args.seed)
    ab = AB_QUICK if args.quick else AB_FULL
    total, failed = 0, []
    suites = [("layouts", suite_layouts), ("values", suite_values), ("tensor", suite_tensor),
              ("chunks", suite_chunks),
              ("tiles", lambda *a: suite_tiles(*a, cc_major=cc)),
              ("own", lambda *a: suite_own(*a, cc_major=cc)),
              ("stage2", lambda *a: suite_stage2(*a, cc_major=cc)),
              ("slabs", lambda *a: suite_slabs(*a, cc_major=cc))]
    for name, fn in suites:
        try:
            got = fn(lib, cuda, rng, ab, args.quick, args.verbose)
            total += got
            print(f"{name}: ok ({got} elements compared)")
        except Fail as e:
            failed.append(name)
            print(f"{name}: FAIL {e}")
    print(f"test_hwgemm: {'FAIL ' + ', '.join(failed) if failed else 'ok'} "
          f"(sm_{cc}x, {total} elements bit-identical to DIRECT up to the sign of zero; "
          f"{ZERO_SIGN['elements']} zero elements differ in sign)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
