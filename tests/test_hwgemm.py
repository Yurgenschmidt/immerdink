#!/usr/bin/env python3
"""IMD_HWGEMM stage 1 (DESIGN v1.21 §6i, issue #75): bit-exact GPU tests against DIRECT.

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
contract of stage 1: workspace 0 and dispatch HWGEMM for fully affine calls, NOT_SUPPORTED
for the others (packed operands, scattered C, multi-mode batch: the pack stage), DIRECT /
SCALE dispatch for K-empty and k = 0, and the batch / column chunking.
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


def bit_mismatches(got, want):
    """Indices where got and want differ in bits, with NaN == NaN (any payload) and, per
    DESIGN §6i.4, +0 == -0: CUTLASS zero-fills a partial K tile with +0 * +0 products, and
    -0 + (+0) = +0, so a zero sum that DIRECT returns as -0 may come back as +0 when k is
    not a multiple of the kernel's K tile (every other value is unchanged by x + (+0))."""
    gb, wb = got.view(np.uint64), want.view(np.uint64)
    zs = (gb != wb) & (got == 0) & (want == 0)
    ZERO_SIGN["elements"] += int(np.count_nonzero(zs))
    return np.nonzero((gb != wb) & ~(np.isnan(got) & np.isnan(want)) & ~zs)[0]


# ---------------------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------------------

def run_case(lib, cuda, name, A, B, Cd, alphas_betas, rng, expect="HWGEMM", launches=None,
             specials=False, tiles=(-1,), c_fill_nan=False, verbose=False):
    """Plans DIRECT and HWGEMM for (A, B, C); expect = HWGEMM / DIRECT / SCALE (dispatch of
    the HWGEMM plan) or NOT_SUPPORTED (plan creation). Compares every (alpha, beta)."""
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
    if expect == "HWGEMM" and ws != 0:
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
                    for plan, out, wsb in ((pd, want, wsd), (ph, got, ws)):
                        cuda.upload(dC, hC0)
                        s = lib.contract(plan, alpha, off(dA, ba), off(dB, bb), beta, off(dC, bc),
                                         dW if wsb else None, wsb)
                        if s != "SUCCESS":
                            raise Fail(f"{name}: contract {s} (alpha {alpha}, beta {beta})")
                        cuda.sync()
                        cuda.download(out, dC)
                    bad = bit_mismatches(got, want)
                    if len(bad):
                        i = bad[0]
                        raise Fail(f"{name} tile {TILES.get(got_tile, got_tile)} alpha {alpha} "
                                   f"beta {beta}: {len(bad)} of {lc} elements differ, first at "
                                   f"{i}: {got[i]!r} vs DIRECT {want[i]!r}")
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
    run_case(lib, cuda, "A not affine", A, B, Cd, ab, rng, expect="NOT_SUPPORTED")
    # not affine: negative K stride
    A = Desc([I1, K1], [6, 9], [9, -1])
    B = Desc([K1, J1], [9, 3], [3, 1])
    Cd = Desc([I1, J1], [6, 3], [3, 1])
    run_case(lib, cuda, "negative K stride", A, B, Cd, ab, rng, expect="NOT_SUPPORTED")
    # not affine: two batch modes that do not fuse
    # (A's gap of 1 before mode L keeps (L, L+1) from fusing into one batch mode)
    A = Desc([L, L + 1, I1, K1], [2, 3, 4, 5], strides([2, 3, 4, 5], [3, 2, 1, 0], gap={0: 1}))
    B = Desc([L + 1, L, K1, J1], [3, 2, 5, 6], strides([3, 2, 5, 6], [3, 2, 0, 1]))
    Cd = Desc([L, L + 1, I1, J1], [2, 3, 4, 6], strides([2, 3, 4, 6], [3, 2, 1, 0]))
    run_case(lib, cuda, "multi-mode batch", A, B, Cd, ab, rng, expect="NOT_SUPPORTED")
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
              ("tiles", lambda *a: suite_tiles(*a, cc_major=cc))]
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
