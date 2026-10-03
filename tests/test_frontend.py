#!/usr/bin/env python3
"""Tensor front end (PLAN M2) and robustness (PLAN M3) tests of imd_contract.

    python3 tests/test_frontend.py [--lib build/libimmerdink.so] [--shim libimd_alloc_shim.so]
                                   [--suite NAME ...] [-v]

Every GPU case goes through tests/test_contract.py's run_case: bit-exact vs
spec/oracle.py contract_tensor (whole C buffer, stage-by-stage workspace checks on the
emulated path), poisoned guard zones around the workspace that must stay untouched, and,
when the allocation shim is preloaded (--shim, tests/alloc_shim.c), no CUDA allocation
inside imd_contract.

Suites:
  lowering   the library's §5 lowering (src/test_hooks.cpp imd_test_lowering: swap, m n k
             b, canonical orders, fused modes) == spec/oracle.py lower(), and the plan
             status == the oracle's, on thousands of random descriptors (legal and not)
  legality   every §5 row 1-5 status, and the precedence between rows, by hand
  random     random legal contractions up to rank 6, bit-exact on the GPU
  fixtures   'acQ,bdQ->acbd', 'ijdkQ,ldQ->klij', 'kjbc,kiac->abij', the six output
             permutations of 'abd,dc->abc'
  robust     execute-time statuses (§6 order: null pointers, empty output, ALIASING,
             MISALIGNED, INSUFFICIENT_WORKSPACE), A == B aliasing, launch splits (n-hat
             over gridDim.y, batch count, K-chunks) and bitwise invariance across residue
             group sizes g and K-chunk lengths, degenerate / zero-size shapes
  highrank   rank 6-12 operands (up to ~15 distinct labels) with hard size guards
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_contract as tc  # noqa: E402
from test_contract import Fail, gen, inject, layout, run_case, scatter, buffer_for  # noqa: E402

o, hz, params = tc.o, tc.hz, tc.params
STATS = tc.STATS


def tdesc(d):
    return hz.make_tensor(d.extent, d.stride, d.label)


ORACLE_MAX_VEC = 2**20   # entries per oracle scatter vector (host memory guard)


class TooBig(Exception):
    pass


def oracle_status(A, B, C, mode=o.EMULATE):
    """(status, view) of spec/oracle.py lower(), or ("TOO_BIG", None) for a legal view
    whose scatter vectors would exceed ORACLE_MAX_VEC entries: the oracle materialises
    them as Python lists (a 2^31-row legality case would take tens of GB)."""
    real = o._radix_vec

    def guarded(modes, X):
        if math.prod(e for _, e, _ in modes) > ORACLE_MAX_VEC:
            raise TooBig()
        return real(modes, X)
    o._radix_vec = guarded
    try:
        return "SUCCESS", o.lower(A, B, C, mode)
    except TooBig:
        return "TOO_BIG", None
    except o.InvalidValue:
        return "INVALID_VALUE", None
    except o.NotSupported:
        return "NOT_SUPPORTED", None
    finally:
        o._radix_vec = real


def lowering_matches(lib, A, B, C, what):
    """The library's lowering equals the oracle's; returns the library's (or None). The
    lowering hook has no mode: the mode-dependent GemmCoord part of row 5 is checked by
    imd_plan_create / imd_workspace_size (suite legality), so compare with DIRECT here."""
    want_st, v = oracle_status(A, B, C, o.DIRECT)
    st, got = lib.lowering(tdesc(A), tdesc(B), tdesc(C))
    if want_st == "TOO_BIG" and st == "SUCCESS":
        return got                                     # legal; too big to compare in Python
    if st != want_st:
        raise Fail(f"{what}: lowering status {st}, oracle {want_st} "
                   f"(A {A.label} {A.extent} {A.stride}; B {B.label} {B.extent} {B.stride}; "
                   f"C {C.label} {C.extent} {C.stride})")
    if st != "SUCCESS":
        return None
    exp = {"swapped": v.swapped, "m": v.m, "n": v.n, "k": v.k, "b": v.b,
           "order": {g: list(v.order[g]) for g in "IJKL"},
           "fused": {g: [(e, st_["A"], st_["B"], st_["C"]) for _, e, st_ in v.fused[g]]
                     for g in "IJKL"}}
    for key in exp:
        if got[key] != exp[key]:
            raise Fail(f"{what}: lowering {key}: library {got[key]} oracle {exp[key]}")
    return got


# ---------------------------------------------------------------------------------------
# Random contractions
# ---------------------------------------------------------------------------------------

class Gen:
    """A random contraction with given label roles.

    roles: counts of I (A,C), J (B,C), K (A,B), L (A,B,C), A1 (A only, summed), B1 (B only),
    C1 (C only, extent 1 or 0), DA / DB (a label repeated in A / B: diagonal; needs a label
    of A / B to repeat). Extents from `ext` (a list to draw from). Each tensor's label order
    is a random permutation, its storage order an independent random permutation; A and B
    get negative (prob neg), zero (prob zero) and padded (prob pad) strides; C gets padded
    strides and negative signs with half that probability (C must pass §5 row 4, which any
    such layout does: strides grow by at least the extent product)."""

    def __init__(self, rng, roles, ext, neg=0.3, zero=0.05, pad=0.3, labels_from=1):
        self.rng = rng
        r = dict(I=0, J=0, K=0, L=0, A1=0, B1=0, C1=0, DA=0, DB=0)
        r.update(roles)
        lab = iter(range(labels_from, labels_from + 64))
        groups = {g: [next(lab) for _ in range(r[g])] for g in ("I", "J", "K", "L", "A1", "B1", "C1")}
        self.ext = {}
        for g, ls in groups.items():
            for l in ls:
                self.ext[l] = 1 if g == "C1" else rng.choice(ext)
        a = groups["I"] + groups["K"] + groups["L"] + groups["A1"]
        b = groups["J"] + groups["K"] + groups["L"] + groups["B1"]
        c = groups["I"] + groups["J"] + groups["L"] + groups["C1"]
        a += [rng.choice(a) for _ in range(r["DA"]) if a]
        b += [rng.choice(b) for _ in range(r["DB"]) if b]
        for x in (a, b, c):
            rng.shuffle(x)
        self.labels = (a, b, c)
        self.neg, self.zero, self.pad = neg, zero, pad

    def descs(self, aligned=False):
        """aligned: every tensor stores its modes in one common random label order, dense
        and positive (consecutive labels then fuse wherever every carrier agrees)."""
        rng = self.rng
        rank = list(self.ext)
        rng.shuffle(rank)
        out = []
        for t, lab in enumerate(self.labels):
            ext = [self.ext[l] for l in lab]
            order = list(range(len(lab)))
            rng.shuffle(order)
            if aligned:
                order.sort(key=lambda d: rank.index(lab[d]))
            neg = 0.0 if aligned else self.neg if t < 2 else self.neg / 2
            st = layout(rng, ext, order=order, neg=neg, pad=0.0 if aligned else self.pad)
            if t < 2:
                st = [0 if rng.random() < self.zero else s for s in st]
            out.append(o.Desc(ext, st, lab))
        return out


def view_size(v):
    return v.b * v.m * v.n * v.k


def run_random(lib, cuda, name, A, B, C, rng, verbose, N=None, mode=None, kinds=None):
    N = N or rng.choice(tc.NS)
    g = rng.choice((0, 1, rng.randint(1, N)))
    rounding = rng.choice(("RN", "TRUNC"))
    mode = mode or (o.DIRECT if rng.random() < 0.1 else o.EMULATE)
    kind = rng.choice(kinds or ("uniform", "phi", "wide", "ints"))
    TA, TB = gen(rng, A.extent, kind), gen(rng, B.extent, kind)
    if rng.random() < 0.2 and TA.size:
        inject(rng, TA, 0.02)
    alpha = rng.choice((1.0, 1.0, -0.5, 1.5, 0.0))
    beta = rng.choice((0.0, 0.0, 1.0, 0.25))
    TC = gen(rng, C.extent, "uniform")
    return run_case(lib, cuda, name, A, B, C, TA, TB, TC, alpha, beta, N, g, rounding,
                    mode=mode, rng=rng, verbose=verbose)


def rand_roles_small(rng):
    """Roles for a random contraction with every operand of rank <= 6.
    Rank <= 6 is a COVERAGE CHOICE for this fast random sweep (oracle cost, many cases),
    not a library limit: the library takes any rank (imd_tensor_t.rank is int32). Ranks
    6-12 are covered by suite highrank below, with explicit size guards."""
    while True:
        r = {g: rng.choice((0, 0, 1, 1, 2)) for g in ("I", "J", "K", "L")}
        r.update(A1=int(rng.random() < 0.15), B1=int(rng.random() < 0.15),
                 C1=int(rng.random() < 0.1), DA=int(rng.random() < 0.15),
                 DB=int(rng.random() < 0.1))
        ra = r["I"] + r["K"] + r["L"] + r["A1"] + r["DA"]
        rb = r["J"] + r["K"] + r["L"] + r["B1"] + r["DB"]
        rc = r["I"] + r["J"] + r["L"] + r["C1"]
        if max(ra, rb, rc) <= 6:
            return r


def suite_lowering(lib, cuda, verbose):
    """Library lowering == oracle lowering on random descriptors, legal and illegal."""
    rng = random.Random(606)
    n_ok = n_bad = n_swap = n_fused = 0
    for t in range(3000):
        G = Gen(rng, rand_roles_small(rng), (0, 1, 2, 2, 3, 3, 4) if t % 10 == 0 else (1, 2, 3, 4))
        A, B, C = G.descs(aligned=t % 3 == 0)
        mut = rng.random()
        if mut < 0.05 and C.extent:                     # break row 4 (C self-overlap)
            d = rng.randrange(len(C.extent))
            C.stride[d] = rng.choice((0, 1, -1))
        elif mut < 0.08 and A.extent:                   # break row 1 (extent mismatch)
            d = rng.randrange(len(A.extent))
            A.extent[d] += 1
        elif mut < 0.10:                                # repeated C label (row 2)
            if C.label:
                C.label.append(C.label[0]), C.extent.append(C.extent[0]), C.stride.append(10**6)
        elif mut < 0.12:                                # C-only label (row 3, or extent 0/1)
            C.label.append(99), C.extent.append(rng.choice((0, 1, 2))), C.stride.append(10**7)
        got = lowering_matches(lib, A, B, C, f"lowering[{t}]")
        if got is None:
            n_bad += 1
            st, _ = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), 9)
            want, _ = oracle_status(A, B, C)
            if st != want:
                raise Fail(f"lowering[{t}]: imd_workspace_size {st}, oracle {want}")
            continue
        n_ok += 1
        n_swap += got["swapped"]
        n_fused += sum(len(got["order"][g]) - len(got["fused"][g]) for g in "IJKL")
    if n_ok < 1500 or n_bad < 200 or n_swap < 300 or n_fused < 100:
        raise Fail(f"lowering coverage too thin: {n_ok} legal, {n_bad} illegal, {n_swap} swapped, "
                   f"{n_fused} fusions")
    if verbose:
        print(f"  ok  {n_ok} legal lowerings identical to the oracle ({n_swap} swapped, "
              f"{n_fused} label fusions), {n_bad} illegal with the oracle's status")


def suite_random(lib, cuda, verbose):
    """Random legal contractions, every operand of rank <= 6 (see rand_roles_small)."""
    rng = random.Random(707)
    done, t = 0, 0
    pivots = {"rows": 0, "rows_radix": 0, "k": 0}   # K2/K3 gather tiling (pack_pivot) seen
    while done < 150:
        t += 1
        G = Gen(rng, rand_roles_small(rng), (0, 1, 2, 3, 3, 4, 5) if t % 8 == 0 else (1, 2, 3, 4, 5))
        A, B, C = G.descs(aligned=t % 4 == 0)
        st, v = oracle_status(A, B, C)
        if st != "SUCCESS" or view_size(v) > 2**15:
            continue
        got = lowering_matches(lib, A, B, C, f"random[{t}]")
        if v.k >= 2 and not v.empty_output():
            for along, radix in got["pivot"]:
                pivots["rows" if along else "k"] += 1
                pivots["rows_radix"] += along and radix > 1
        run_random(lib, cuda, f"random[{t}] {A.label},{B.label}->{C.label}", A, B, C, rng, verbose)
        done += 1
    if min(pivots.values()) < 10:
        raise Fail(f"random: gather tilings too thin {pivots}")
    if verbose:
        print(f"  ok  {done} random contractions (rank <= 6); pack tilings {pivots}")


# ---------------------------------------------------------------------------------------
# Legality (plan time, §5 rows 1-5 and precedence)
# ---------------------------------------------------------------------------------------

def suite_legality(lib, cuda, verbose):
    D = o.Desc
    i, j, k, l, x, y = 1, 2, 3, 4, 5, 6
    big = 2**62
    cases = [
        # (name, A, B, C, N, mode, expected status; the oracle agrees where it models it)
        ("ok", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]), D([2, 4], [4, 1], [i, j]),
         "SUCCESS"),
        ("row 1 extent mismatch", D([2, 3], [3, 1], [i, k]), D([4, 4], [4, 1], [k, j]),
         D([2, 4], [4, 1], [i, j]), "INVALID_VALUE"),
        ("row 1 extent mismatch within A (diagonal)", D([2, 3], [3, 1], [i, i]),
         D([2, 4], [4, 1], [i, j]), D([4], [1], [j]), "INVALID_VALUE"),
        ("row 2 repeated C label", D([2, 3], [3, 1], [i, k]), D([3, 2], [2, 1], [k, j]),
         D([2, 2], [2, 1], [i, i]), "NOT_SUPPORTED"),
        ("row 2 repeated C label, extent 1", D([1, 3], [3, 1], [i, k]), D([3, 2], [2, 1], [k, j]),
         D([1, 1, 2], [2, 2, 1], [i, i, j]), "NOT_SUPPORTED"),
        ("row 2 repeated C label, extent 0", D([0, 3], [3, 1], [i, k]), D([3, 2], [2, 1], [k, j]),
         D([0, 0, 2], [2, 2, 1], [i, i, j]), "NOT_SUPPORTED"),
        ("row 3 C-only label extent 2", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4, 2], [8, 1, 16], [i, j, x]), "NOT_SUPPORTED"),
        ("C-only label extent 1 is fine", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4, 1], [4, 1, 77], [i, j, x]), "SUCCESS"),
        ("C-only label extent 0: empty C, no-op", D([2, 3], [3, 1], [i, k]),
         D([3, 4], [4, 1], [k, j]), D([2, 4, 0], [4, 1, 8], [i, j, x]), "SUCCESS"),
        ("row 4 C stride 0", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4], [0, 1], [i, j]), "INVALID_VALUE"),
        ("row 4 C overlapping modes", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4], [3, 1], [i, j]), "INVALID_VALUE"),
        ("row 4 C equal |strides| (+/-)", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4], [-1, 1], [i, j]), "INVALID_VALUE"),
        ("row 4 C passes with negative strides", D([2, 3], [3, 1], [i, k]),
         D([3, 4], [4, 1], [k, j]), D([2, 4], [-4, -1], [i, j]), "SUCCESS"),
        ("row 4 not applied to an empty C", D([0, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([0, 4], [0, 0], [i, j]), "SUCCESS"),
        ("row 4 not applied to A/B (stride 0, overlap)", D([2, 3], [0, 0], [i, k]),
         D([3, 4], [1, 1], [k, j]), D([2, 4], [4, 1], [i, j]), "SUCCESS"),
        ("row 4 sufficient, not necessary: rejected injective C", D([2, 3], [3, 1], [i, k]),
         D([3, 3], [3, 1], [k, j]), D([2, 3], [3, 2], [i, j]), "INVALID_VALUE"),
        ("row 5 span > int64 (A)", D([3, 3], [big, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([3, 4], [4, 1], [i, j]), "INVALID_VALUE"),
        ("row 5 merged diagonal stride > int64", D([2, 2], [big, big], [k, k]),
         D([2, 4], [4, 1], [k, j]), D([4], [1], [j]), "INVALID_VALUE"),
        ("row 5 extent product > int64", D([2**32, 2**32, 2], [0, 0, 1], [x, y, k]),
         D([2, 4], [4, 1], [k, j]), D([4], [1], [j]), "INVALID_VALUE"),
        ("row 5 b m k > int64", D([2**20, 2**20, 2**30], [0, 0, 0], [l, i, x]),
         D([2**20, 2], [0, 0], [l, j]), D([2**20, 2**20, 2], [2**21, 2, 1], [l, i, j]),
         "INVALID_VALUE"),
        ("row 5 GemmCoord m > 2^31-1 (EMULATE, k >= 2)", D([2**31, 2], [2, 1], [i, k]),
         D([2, 1], [1, 1], [k, j]), D([2**31, 1], [1, 1], [i, j]), "INVALID_VALUE"),
        ("row 5 GemmCoord not applied with k = 1 (K-empty)", D([2**31], [1], [i]),
         D([1], [1], [j]), D([2**31, 1], [1, 1], [i, j]), "SUCCESS"),
        ("row 5 GemmCoord not applied with k = 0", D([2**31, 0], [1, 1], [i, k]),
         D([0, 1], [1, 1], [k, j]), D([2**31, 1], [1, 1], [i, j]), "SUCCESS"),
        # precedence: the first failing row wins
        ("row 1 before row 2", D([2, 3], [3, 1], [i, k]), D([4, 2], [2, 1], [k, j]),
         D([2, 2], [2, 1], [i, i]), "INVALID_VALUE"),
        ("row 2 before row 3", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 2, 4, 2], [8, 8, 1, 16], [i, i, j, x]), "NOT_SUPPORTED"),
        ("row 3 before row 4", D([2, 3], [3, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([2, 4, 2], [0, 1, 16], [i, j, x]), "NOT_SUPPORTED"),
        ("row 4 before row 5", D([3, 3], [big, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([3, 4], [0, 1], [i, j]), "INVALID_VALUE"),
        ("row 2 before row 5", D([3, 3], [big, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([3, 3], [4, 1], [i, i]), "NOT_SUPPORTED"),
        ("row 3 before row 5 (overflow)", D([3, 3], [big, 1], [i, k]), D([3, 4], [4, 1], [k, j]),
         D([3, 4, 2], [4, 1, 12], [i, j, x]), "NOT_SUPPORTED"),
    ]
    for name, A, B, C, want in cases:
        st_o, _ = oracle_status(A, B, C)
        if st_o == "TOO_BIG" and want == "SUCCESS":
            st_o = want
        if st_o != want:
            raise Fail(f"legality '{name}': oracle {st_o}, expected {want}")
        st_w, _ = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), 9)
        st_p = st_w
        # imd_plan_create pins 8 (2m + 2n + 2k + 3b) bytes of scatter vectors: only for
        # small legal problems (the failing ones fail before any allocation).
        st_l, lw = lib.lowering(tdesc(A), tdesc(B), tdesc(C))
        if st_l != "SUCCESS" or 8 * (2 * (lw["m"] + lw["n"] + lw["k"]) + 3 * lw["b"]) <= \
                lib.MAX_PLAN_HOST:
            st_p, plan = lib.plan_create(tdesc(A), tdesc(B), tdesc(C), 9)
            if plan is not None:
                lib.plan_destroy(plan)
        if st_w != want or st_p != want:
            raise Fail(f"legality '{name}': workspace_size {st_w}, plan_create {st_p}, "
                       f"expected {want}")
        if want != "SUCCESS":
            lowering_matches(lib, A, B, C, f"legality '{name}'")
    # Mode-dependent row 5 parts and §4a overflow (library only; the oracle does not size).
    A, B, C = (D([2**31, 2], [2, 1], [i, k]), D([2, 1], [1, 1], [k, j]),
               D([2**31, 1], [1, 1], [i, j]))
    st, _ = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), 9, mode=hz.DIRECT)
    if st != "SUCCESS":
        raise Fail(f"GemmCoord applied in DIRECT mode: {st}")
    # W = N b m n_hat bytes > 2^64 while b m n < 2^63: m = n = 2^30, b = 4, k = 2, N = 20.
    A = D([2**30, 2, 4], [1, 2**30, 2**31], [i, k, l])
    B = D([2, 2**30, 4], [1, 2, 2**31], [k, j, l])
    C = D([2**30, 2**30, 4], [2**30, 1, 2**60], [i, j, l])
    for N in (2, 20):
        st, ws = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), N)
        want = "SUCCESS" if N * 2**62 < 2**64 else "INVALID_VALUE"   # W = N b m n_hat
        if st != want:
            raise Fail(f"workspace uint64 overflow, N={N}: {st} (expected {want})")
    # Malformed descriptors / options.
    good = (hz.make_tensor([2, 3], [3, 1], [i, k]), hz.make_tensor([3, 4], [4, 1], [k, j]),
            hz.make_tensor([2, 4], [4, 1], [i, j]))
    for name, tens, N, g, rnd, mode in (
            ("negative extent", (hz.make_tensor([-1, 3], [3, 1], [i, k]),) + good[1:], 9, 0, 0, 0),
            ("negative rank", (hz.Tensor(-1, None, None, None),) + good[1:], 9, 0, 0, 0),
            ("null arrays", (hz.Tensor(2, None, None, None),) + good[1:], 9, 0, 0, 0),
            ("N = 1", good, 1, 0, 0, 0), ("N = 21", good, 21, 0, 0, 0),
            ("g = N + 1", good, 9, 10, 0, 0), ("g = -1", good, 9, -1, 0, 0),
            ("rounding 2", good, 9, 0, 2, 0), ("mode 2", good, 9, 0, 0, 2)):
        st_w, _ = lib.workspace_size(*tens, N, g, rnd, mode)
        st_p, plan = lib.plan_create(*tens, N, g, rnd, mode)
        if plan is not None:
            lib.plan_destroy(plan)
        if st_w != "INVALID_VALUE" or st_p != "INVALID_VALUE":
            raise Fail(f"malformed '{name}': {st_w} / {st_p}")
    if verbose:
        print(f"  ok  {len(cases)} §5 legality / precedence cases + mode-dependent row 5, "
              f"§4a overflow and 9 malformed-argument cases")


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------

def einsum_descs(rng, spec, ext, neg=0.0, pad=0.0, orders=None):
    """Descriptors for an einsum string 'ab,bc->ac' (one letter per label, extents from
    `ext`); storage: last mode fastest unless orders[t] (a mode permutation) is given."""
    lhs, rhs = spec.split("->")
    out = []
    for t, s in enumerate(lhs.split(",") + [rhs]):
        e = [ext[ch] for ch in s]
        od = None if orders is None or orders[t] is None else orders[t]
        st = layout(rng, e, order=od, neg=neg if t < 2 else 0.0, pad=pad)
        out.append(o.Desc(e, st, [ord(ch) for ch in s]))
    return out


def suite_fixtures(lib, cuda, verbose):
    rng = random.Random(808)

    def run(spec, ext, N=9, **kw):
        A, B, C = einsum_descs(rng, spec, ext, **kw)
        lowering_matches(lib, A, B, C, spec)
        for rounding, alpha, beta in (("RN", 1.0, 0.0), ("TRUNC", -0.5, 1.0)):
            TA, TB = gen(rng, A.extent, "phi"), gen(rng, B.extent, "phi")
            run_case(lib, cuda, f"{spec} {rounding}", A, B, C, TA, TB, gen(rng, C.extent, "uniform"),
                     alpha, beta, N, 0, rounding, rng=rng, expect_kind="GEMM", verbose=verbose)

    run("acQ,bdQ->acbd", dict(a=5, c=4, b=3, d=6, Q=37), N=14)
    run("acQ,bdQ->acbd", dict(a=5, c=4, b=3, d=6, Q=37), N=8, neg=0.5, pad=0.5)
    run("ijdkQ,ldQ->klij", dict(i=3, j=4, d=5, k=2, Q=11, l=6), N=16)
    run("ijdkQ,ldQ->klij", dict(i=3, j=4, d=5, k=2, Q=11, l=6), N=9, neg=0.4, pad=0.4,
        orders=[[2, 0, 4, 1, 3], [1, 2, 0], None])
    run("kjbc,kiac->abij", dict(k=7, j=3, b=4, c=5, i=2, a=6), N=20)
    run("kjbc,kiac->abij", dict(k=7, j=3, b=4, c=5, i=2, a=6), N=3, neg=0.5, pad=0.3)
    # (T)-style: 'abd,dc->X' for all six permutations X of 'abc'
    for perm in ("abc", "acb", "bac", "bca", "cab", "cba"):
        run(f"abd,dc->{perm}", dict(a=4, b=5, d=19, c=6), N=14)


# ---------------------------------------------------------------------------------------
# Robustness: execute-time statuses, launch splits, invariance, degenerate shapes
# ---------------------------------------------------------------------------------------

def exec_status_cases(lib, cuda, verbose):
    """§6 imd_contract checks, in order: null pointers; empty output -> SUCCESS with
    nothing touched; ALIASING (§5 row 6); MISALIGNED (row 7); INSUFFICIENT_WORKSPACE. A
    failing call must not touch C."""
    D = o.Desc
    i, j, k = 1, 2, 3
    m, n, kk = 6, 5, 7
    A, B, C = D([m, kk], [kk, 1], [i, k]), D([kk, n], [n, 1], [k, j]), D([m, n], [n, 1], [i, j])
    st, need = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), 9)
    assert st == "SUCCESS" and need > 0
    st, plan = lib.plan_create(tdesc(A), tdesc(B), tdesc(C), 9)
    assert st == "SUCCESS"
    nA, nB, nC = m * kk * 8, kk * n * 8, m * n * 8
    arena_bytes = 1 << 20
    arena = cuda.malloc(arena_bytes)
    base = arena.value
    cmark = np.full(m * n, 42.0)
    n_checks = 0

    def put_c(pC):
        cuda.upload_raw(pC, cmark)

    def c_untouched(pC):
        got = np.empty(m * n)
        cuda.download_raw(got, pC)
        return np.array_equal(got, cmark)

    try:
        cuda.memset(arena, 0, arena_bytes)
        pA, pB, pC, pW = base + 1024, base + 4096, base + 8192, base + 65536
        lay = dict(A=pA, B=pB, C=pC, W=pW)

        def call(want, alpha=1.0, beta=0.0, p=None, ws_bytes=None, what=""):
            nonlocal n_checks
            q = dict(lay)
            q.update(p or {})
            if q["C"] is not None and q["C"] % 8 == 0 and q["C"] >= base:
                put_c(q["C"])
            st = lib.contract(plan, alpha, q["A"], q["B"], beta, q["C"], q["W"],
                              need if ws_bytes is None else ws_bytes)
            cuda.sync()
            if st != want:
                raise Fail(f"imd_contract {what}: {st}, expected {want}")
            if st != "SUCCESS" and q["C"] is not None and q["C"] % 8 == 0 and not c_untouched(q["C"]):
                raise Fail(f"imd_contract {what}: returned {st} but wrote C")
            n_checks += 1

        call("SUCCESS", what="baseline")
        # 1. null pointers
        call("INVALID_VALUE", p=dict(C=None), what="C null")
        call("INVALID_VALUE", p=dict(A=None), what="A null (alpha != 0, k != 0)")
        call("INVALID_VALUE", p=dict(B=None), what="B null")
        call("INVALID_VALUE", p=dict(W=None), what="ws null (workspace_size > 0)")
        call("SUCCESS", alpha=0.0, beta=2.0, p=dict(A=None, B=None), what="A, B null, alpha = 0")
        call("INVALID_VALUE", p=dict(C=None, A=pC), what="null before aliasing")
        # 3. ALIASING: byte intervals, conservative (also when alpha = 0)
        call("ALIASING", p=dict(C=pA + 8), what="C inside A")
        call("ALIASING", p=dict(C=pB - nC + 8), what="C overlaps B's first element")
        call("SUCCESS", p=dict(C=pA + nA), what="C adjacent to A (touching, no overlap)")
        call("ALIASING", alpha=0.0, beta=1.0, p=dict(C=pA), what="C == A, alpha = 0")
        call("ALIASING", p=dict(W=pA - 256), what="workspace overlaps A")
        call("ALIASING", p=dict(W=pB), what="workspace overlaps B")
        call("ALIASING", p=dict(W=(pC - need + 256) // 256 * 256), what="workspace end overlaps C")
        call("SUCCESS", p=dict(W=pW, C=pW + need), what="C right after [ws, ws + size)")
        call("SUCCESS", p=dict(B=pA), what="A == B (allowed; same buffer, shapes differ)")
        # 4. MISALIGNED (after ALIASING)
        call("MISALIGNED", p=dict(A=pA + 4), what="A 4-byte aligned")
        call("MISALIGNED", p=dict(B=pB + 2), what="B misaligned")
        call("MISALIGNED", p=dict(C=pC + 4), what="C misaligned")
        call("MISALIGNED", p=dict(W=pW + 8), what="ws 8-byte aligned")
        call("MISALIGNED", p=dict(W=pW + 128), what="ws 128-byte aligned")
        call("ALIASING", p=dict(C=pA + 4), what="aliasing before misaligned")
        # 5. INSUFFICIENT_WORKSPACE (after MISALIGNED)
        call("INSUFFICIENT_WORKSPACE", ws_bytes=need - 1, what="ws_bytes = size - 1")
        call("INSUFFICIENT_WORKSPACE", ws_bytes=0, what="ws_bytes = 0")
        call("MISALIGNED", p=dict(W=pW + 8), ws_bytes=0, what="misaligned before insufficient")
        call("SUCCESS", ws_bytes=need + 12345, what="ws_bytes > size")
    finally:
        lib.plan_destroy(plan)

    # 2. empty output: SUCCESS, nothing touched, even with aliasing / misaligned / null
    # A, B, ws and ws_bytes = 0 (workspace_size = 0).
    Ae, Be, Ce = D([0, kk], [kk, 1], [i, k]), D([kk, n], [n, 1], [k, j]), D([0, n], [n, 1], [i, j])
    st, need0 = lib.workspace_size(tdesc(Ae), tdesc(Be), tdesc(Ce), 9)
    st2, plan = lib.plan_create(tdesc(Ae), tdesc(Be), tdesc(Ce), 9)
    try:
        if st != "SUCCESS" or need0 != 0 or st2 != "SUCCESS":
            raise Fail(f"empty output: {st} {need0} {st2}")
        for what, args in (("C misaligned, null ws", (base, base, base + 3, None)),
                           ("aliased + misaligned", (base + 4, base + 4, base + 4, base + 4))):
            st = lib.contract(plan, 1.0, args[0], args[1], 0.0, args[2], args[3], 0)
            if st != "SUCCESS":
                raise Fail(f"empty output ({what}): {st}")
            n_checks += 1
        for what, args in (("C null", (base, base, None)), ("A null", (None, base, base))):
            st = lib.contract(plan, 1.0, args[0], args[1], 0.0, args[2], None, 0)
            if st != "INVALID_VALUE":
                raise Fail(f"empty output, {what}: {st} (null checks come first)")
        if lib.contract(plan, 0.0, None, None, 0.0, base, None, 0) != "SUCCESS":
            raise Fail("empty output, alpha = 0, null A, B, ws")
    finally:
        lib.plan_destroy(plan)
    # k = 0 through a B-only label of extent 0: A is non-empty but not read (null
    # allowed); the workspace is still needed (vec region); C vs A is still checked.
    Ak, Bk = D([m, kk], [kk, 1], [i, k]), D([kk, n, 0], [n, 1, 1], [k, j, 7])
    st, needk = lib.workspace_size(tdesc(Ak), tdesc(Bk), tdesc(C), 9)
    _, plan = lib.plan_create(tdesc(Ak), tdesc(Bk), tdesc(C), 9)
    try:
        put_c(base + 8192)
        st = lib.contract(plan, 1.0, None, None, 0.5, base + 8192, base + 65536, needk)
        cuda.sync()
        got = np.empty(m * n)
        cuda.download_raw(got, base + 8192)
        if st != "SUCCESS" or not np.array_equal(got, 0.5 * cmark):
            raise Fail(f"k = 0 with null A, B: {st}")
        st = lib.contract(plan, 1.0, base + 8192, base + 8192, 0.5, base + 8192, base + 65536, needk)
        if st != "ALIASING":
            raise Fail(f"k = 0: C == A must still be ALIASING (conservative), got {st}")
        n_checks += 2
    finally:
        lib.plan_destroy(plan)
        cuda.free(arena)
    if verbose:
        print(f"  ok  {n_checks} execute-time status checks in the §6 order")


def suite_robust(lib, cuda, verbose):
    rng = random.Random(909)
    exec_status_cases(lib, cuda, verbose)
    D = o.Desc
    i, j, k, l, x, y = 1, 2, 3, 4, 5, 6

    # A == B (same buffer, same descriptor): 'ki,kj->ij' = A^T A.
    m, kk = 9, 33
    A = D([kk, m], [m, 1], [k, i])
    B = D([kk, m], [m, 1], [k, j])
    C = D([m, m], [m, 1], [i, j])
    TA = gen(rng, (kk, m), "phi")
    bufA, _ = scatter(TA, A, None)
    want = np.array(o.contract_tensor(o.Tensor(bufA.tolist(), 0, A), o.Tensor(bufA.tolist(), 0, B),
                                      o.Tensor([0.0] * m * m, 0, C), 1.0, 0.0, 14))
    st, need = lib.workspace_size(tdesc(A), tdesc(B), tdesc(C), 14)
    _, plan = lib.plan_create(tdesc(A), tdesc(B), tdesc(C), 14)
    dA, dC, dW = cuda.malloc(bufA.nbytes), cuda.malloc(m * m * 8), cuda.malloc(need)
    try:
        cuda.upload(dA, bufA)
        st = lib.contract(plan, 1.0, dA.value, dA.value, 0.0, dC.value, dW.value, need)
        cuda.sync()
        got = np.empty(m * m)
        cuda.download(got, dC)
        if st != "SUCCESS" or len(tc.bit_mismatches(got, want)):
            raise Fail(f"A == B (A^T A): {st}, {len(tc.bit_mismatches(got, want))} mismatches")
    finally:
        for p in (dA, dC, dW):
            cuda.free(p)
        lib.plan_destroy(plan)
    if verbose:
        print("  ok  A == B (one buffer for both operands) bit-exact")

    # Peak device memory: cudaMemGetInfo bracketing of imd_contract (PLAN M3). Free device
    # memory must be the same before the call and after it completes; the caller's
    # workspace is the only memory used (the allocation shim and the guard zones of
    # run_case check the same more strictly). Another process using the GPU can move the
    # free count, so the check passes if any of 5 attempts brackets exactly; an allocation
    # by the library would move it in every attempt. A warm-up call first loads the
    # kernels (CUDA lazy loading takes device memory on a kernel's first launch).
    A, B, C = tc.matrix(random.Random(3), 300, 200, 500)
    tA, tB, tC = tdesc(A), tdesc(B), tdesc(C)
    _, need = lib.workspace_size(tA, tB, tC, 14)
    _, plan = lib.plan_create(tA, tB, tC, 14)
    sizes = [8 * buffer_for(d)[0] for d in (A, B, C)] + [need]
    bufs = [cuda.malloc(n_) for n_ in sizes]
    try:
        for p_, n_ in zip(bufs, sizes):
            cuda.memset(p_, 0, n_)
        args = (plan, 1.0, bufs[0].value, bufs[1].value, 0.0, bufs[2].value, bufs[3].value, need)
        if lib.contract(*args) != "SUCCESS":
            raise Fail("memory bracket: warm-up failed")
        cuda.sync()
        deltas = []
        for _ in range(5):
            f0, _ = cuda.mem_info()
            st = lib.contract(*args)
            cuda.sync()
            f1, _ = cuda.mem_info()
            deltas.append(f0 - f1)
            if st != "SUCCESS":
                raise Fail(f"memory bracket: {st}")
            if f0 == f1:
                break
        else:
            raise Fail(f"memory bracket: free device memory moved by {deltas} bytes")
    finally:
        for p_ in bufs:
            cuda.free(p_)
        lib.plan_destroy(plan)
    if verbose:
        print(f"  ok  cudaMemGetInfo bracket: no device memory beyond the {need}-byte workspace "
              f"(attempts: {len(deltas)})")

    # Launch splits via the test-only limits (production values restored after each). Every
    # run of one problem uses the same fill for C's unaddressed gaps (rng seed 77), so the
    # whole C buffers can be compared bit for bit.
    def with_limits(lim, fn):
        lib.set_limits(*lim)
        try:
            return fn()
        finally:
            lib.set_limits(0, 0, 0)

    def mat(m, n, kk, seed):
        r = random.Random(seed)
        A, B, C = tc.matrix(r, m, n, kk, neg=0.2, pad=0.2)
        TA, TB, TC = gen(r, (m, kk), "phi"), gen(r, (kk, n), "phi"), gen(r, (m, n), "uniform")
        inject(r, TA, 0.01)
        return A, B, C, TA, TB, TC

    # n-hat over gridDim.y: one 256-column tile per launch, n_hat = 608 -> 3 launches; also
    # combined with a batch split (max_batch 2 over g b = 3 * 4) and K-chunks of 16.
    A, B, C, TA, TB, TC = mat(5, 600, 40, 1)
    ref = run_case(lib, cuda, "n-hat split ref", A, B, C, TA, TB, TC, 1.0, 0.5, 9, 4, rng=random.Random(77),
                   expect_kind="GEMM", verbose=verbose)
    for lim in ((0, 0, 1), (16, 0, 1), (0, 2, 1), (32, 3, 2)):
        got = with_limits(lim, lambda: run_case(
            lib, cuda, f"n-hat split, limits (k_chunk, max_batch, max_n_tiles) = {lim}", A, B, C,
            TA, TB, TC, 1.0, 0.5, 9, 4, rng=random.Random(77), expect_kind="GEMM", verbose=verbose,
            chunk=lim[0] or o.K_C_ALIGNED))
        if not np.array_equal(got.view(np.uint64), ref.view(np.uint64)):
            raise Fail(f"launch limits {lim} changed the result")
    # Batched with b = 5 and a role swap: batch split down to 1 per launch.
    G = Gen(rng, dict(I=1, J=1, K=2, L=1), (5, 6, 7))
    A, B, C = G.descs()
    TA, TB, TC = gen(rng, A.extent, "wide"), gen(rng, B.extent, "wide"), gen(rng, C.extent, "uniform")
    ref = run_case(lib, cuda, "batch split ref", A, B, C, TA, TB, TC, -1.0, 0.0, 14, 3, rng=random.Random(77),
                   expect_kind="GEMM", verbose=verbose)
    for lim in ((0, 1, 0), (48, 4, 1)):
        got = with_limits(lim, lambda: run_case(
            lib, cuda, f"batch split, limits {lim}", A, B, C, TA, TB, TC, -1.0, 0.0, 14, 3,
            rng=random.Random(77), expect_kind="GEMM", verbose=verbose, chunk=lim[0] or o.K_C_ALIGNED))
        if not np.array_equal(got.view(np.uint64), ref.view(np.uint64)):
            raise Fail(f"launch limits {lim} changed the result")

    # Bitwise invariance across residue group sizes g (1..N) and K-chunk lengths.
    for N, rounding in ((9, "RN"), (16, "TRUNC")):
        A, B, C, TA, TB, TC = mat(13, 21, 133, 2 + N)
        outs = []
        for g in sorted({1, 2, 3, N // 2, N - 1, N}):
            outs.append(run_case(lib, cuda, f"g-invariance N={N} g={g}", A, B, C, TA, TB, TC,
                                 0.75, 0.0, N, g, rounding, rng=random.Random(77), expect_kind="GEMM",
                                 verbose=verbose))
        for ch in (16, 32, 64, 128):
            outs.append(with_limits((ch, 0, 0), lambda: run_case(
                lib, cuda, f"chunk-invariance N={N} k_chunk={ch}", A, B, C, TA, TB, TC, 0.75,
                0.0, N, 0, rounding, rng=random.Random(77), expect_kind="GEMM", verbose=verbose, chunk=ch)))
        if any(not np.array_equal(a.view(np.uint64), outs[0].view(np.uint64)) for a in outs):
            raise Fail(f"results differ across g / K-chunk (N={N})")
    if verbose:
        print("  ok  bitwise invariance across g and K-chunk lengths; launch splits")

    # Degenerate and zero-size shapes (§3.7, §5).
    def run(name, A, B, C, expect, N=9, alpha=1.0, beta=0.5, mode=o.EMULATE):
        run_case(lib, cuda, name, A, B, C, gen(rng, A.extent, "phi"), gen(rng, B.extent, "phi"),
                 gen(rng, C.extent, "uniform"), alpha, beta, N, 0, "RN", mode=mode, rng=rng,
                 expect_kind=expect, verbose=verbose)

    run("1x1x2", D([1, 2], [2, 1], [i, k]), D([2, 1], [1, 1], [k, j]), D([1, 1], [1, 1], [i, j]), "GEMM")
    run("dot 'k,k->' (rank-0 C)", D([50], [1], [k]), D([50], [-1], [k]), D([], [], []), "GEMM")
    run("',->' (all rank 0)", D([], [], []), D([], [], []), D([], [], []), "DIRECT")
    run("',j->j' (scalar A)", D([], [], []), D([7], [2], [j]), D([7], [1], [j]), "DIRECT")
    run("'k,kj->j' (m = 1)", D([30], [1], [k]), D([30, 7], [7, 1], [k, j]), D([7], [1], [j]), "GEMM")
    run("'ik,k->i' (n = 1, swap)", D([7, 30], [30, 1], [i, k]), D([30], [1], [k]), D([7], [1], [i]),
        "GEMM")
    run("skinny 4000x1x3", D([4000, 3], [3, 1], [i, k]), D([3], [1], [k]), D([4000], [1], [i]), "GEMM")
    run("k = 1 (K-empty)", D([4, 1], [1, 4], [i, k]), D([1, 5], [5, 1], [k, j]),
        D([4, 5], [5, 1], [i, j]), "DIRECT")
    run("k = 0", D([4, 0], [1, 4], [i, k]), D([0, 5], [5, 1], [k, j]), D([4, 5], [5, 1], [i, j]),
        "SCALE", beta=-3.0)
    run("A-only label of extent 0 (k = 0)", D([4, 3, 0], [3, 1, 1], [i, k, x]),
        D([3, 5], [5, 1], [k, j]), D([4, 5], [5, 1], [i, j]), "SCALE")
    run("B-only label summed", D([4, 3], [3, 1], [i, k]), D([3, 5, 6], [30, 6, 1], [k, j, y]),
        D([4, 5], [5, 1], [i, j]), "GEMM")
    run("m = 0", D([0, 3], [3, 1], [i, k]), D([3, 5], [5, 1], [k, j]), D([0, 5], [5, 1], [i, j]), "NOOP")
    run("n = 0", D([4, 3], [3, 1], [i, k]), D([3, 0], [1, 1], [k, j]), D([4, 0], [1, 1], [i, j]), "NOOP")
    run("b = 0", D([4, 3, 0], [3, 1, 1], [i, k, l]), D([3, 5, 0], [5, 1, 1], [k, j, l]),
        D([4, 5, 0], [5, 1, 1], [i, j, l]), "NOOP")
    run("C-only extent 0", D([4, 3], [3, 1], [i, k]), D([3, 5], [5, 1], [k, j]),
        D([4, 5, 0], [5, 1, 1], [i, j, x]), "NOOP")
    run("C-only extent 1", D([4, 3], [3, 1], [i, k]), D([3, 5], [5, 1], [k, j]),
        D([4, 1, 5], [5, 99, 1], [i, x, j]), "GEMM")
    run("all extent 1", D([1, 1], [5, 7], [i, k]), D([1, 1], [3, 3], [k, j]),
        D([1, 1], [1, 1], [i, j]), "DIRECT")
    run("stride-0 A (broadcast read)", D([4, 3], [0, 1], [i, k]), D([3, 5], [5, 1], [k, j]),
        D([4, 5], [5, 1], [i, j]), "GEMM")
    run("DIRECT mode, batched", D([3, 8, 2], [16, 1, 8], [i, k, l]), D([8, 4, 2], [4, 1, 32], [k, j, l]),
        D([3, 4, 2], [4, 1, 12], [i, j, l]), "DIRECT", mode=o.DIRECT)


# ---------------------------------------------------------------------------------------
# High rank (rank 6-12 per operand), bounded in size and time
# ---------------------------------------------------------------------------------------

HR_MAX_ELEMS = 2**15        # every tensor: logical elements (product of extents)
HR_MAX_BUFFER = 2**17       # every tensor: buffer elements incl. padding gaps
HR_MAX_MNK = 2**20          # b m n k of the lowered GEMM (the oracle is pure Python)
HR_MAX_WS = 64 * 2**20      # workspace bytes


def hr_guard(name, A, B, C, v, N, g):
    """Hard size guards, asserted before a case runs (so later edits cannot silently grow
    the suite past its memory / time budget)."""
    for X, d in (("A", A), ("B", B), ("C", C)):
        ne = math.prod(d.extent)
        assert ne <= HR_MAX_ELEMS, f"{name}: {X} has {ne} elements > {HR_MAX_ELEMS}"
        nb = buffer_for(d)[0]
        assert nb <= HR_MAX_BUFFER, f"{name}: {X} buffer {nb} > {HR_MAX_BUFFER}"
    mnk = v.b * v.m * v.n * max(v.k, 1)
    assert mnk <= HR_MAX_MNK, f"{name}: b m n k = {mnk} > {HR_MAX_MNK}"
    ws = 0 if v.empty_output() else params.workspace_size(v.m, v.n, v.k, v.b, N, g or N,
                                                         params.EMULATE)
    assert ws <= HR_MAX_WS, f"{name}: workspace {ws} > {HR_MAX_WS}"


def hr_case(rng, roles, ext, tries=50):
    """Descriptors for `roles`, re-drawing the layouts until fusion cannot collapse any
    group with >= 2 labels to a single mode (returns (A, B, C, view, collapsed?))."""
    G = Gen(rng, roles, ext, neg=0.3, zero=0.03, pad=0.3)
    for _ in range(tries):
        A, B, C = G.descs()
        st, v = oracle_status(A, B, C)
        assert st == "SUCCESS", (st, roles)
        if v.empty_output():
            return A, B, C, v, False
        if all(len(v.fused[g]) > 1 for g in "IJKL" if len(v.order[g]) >= 2):
            return A, B, C, v, False
    return A, B, C, v, True


def suite_highrank(lib, cuda, verbose):
    """Operands of rank 6-12 (up to ~15 distinct labels), extents mostly 2-3 with some 1s,
    all label roles, permuted label orders, padded / negative / zero strides in A and B and
    permuted C layouts that keep fusion from collapsing the groups."""
    rng = random.Random(1212)
    t0 = time.time()
    cases = [
        ("rank 8 x 8 -> 8", dict(I=3, J=3, K=3, L=2), (2, 3), {}),
        ("rank 8 x 8 -> 8 (b)", dict(I=2, J=3, K=3, L=3), (2, 2, 3, 1), {}),
        ("rank 10 x 9 -> 8 (A-only)", dict(I=3, J=3, K=4, L=2, A1=1), (2, 2, 2, 3), {}),
        ("rank 10 x 10 -> 6, diagonal in B", dict(I=2, J=2, K=4, L=2, B1=1, DB=1), (2, 2, 3), {}),
        ("rank 12 x 12 -> 8, all extents 2", dict(I=3, J=3, K=5, L=2, A1=1, B1=1, DA=1, DB=1),
         (2,), {}),
        ("rank 12 x 11 -> 9, all extents 2, DIRECT", dict(I=4, J=3, K=6, L=2), (2,),
         dict(mode=o.DIRECT)),
        ("rank 9 x 9 -> 10, K-empty (DIRECT dispatch)", dict(I=4, J=4, L=5), (2, 2, 1), {}),
        ("rank 8 x 8 -> 8, extent-0 K label (k = 0)", dict(I=3, J=3, K=3, L=2), (2, 3), dict(zero="K")),
        ("rank 8 x 8 -> 8, extent-0 L label (no-op)", dict(I=3, J=3, K=3, L=2), (2, 3), dict(zero="L")),
    ]
    # CCSD(T)-style rank-6 outputs: 'ijae,ekbc->X' for permuted X of 'abcijk'.
    ccsdt = []
    for perm in ("abcijk", "kjicba", "bacjik", "cijkab", "ikjbca"):
        ccsdt.append(perm)
    # Random fill-up: operand ranks 6-12.
    while len(cases) < 40:
        r = {g: rng.randint(1, 4) for g in ("I", "J", "K", "L")}
        r.update(A1=int(rng.random() < 0.3), B1=int(rng.random() < 0.3), C1=int(rng.random() < 0.3),
                 DA=int(rng.random() < 0.2), DB=int(rng.random() < 0.2))
        ra = r["I"] + r["K"] + r["L"] + r["A1"] + r["DA"]
        rb = r["J"] + r["K"] + r["L"] + r["B1"] + r["DB"]
        labels = sum(r[g] for g in ("I", "J", "K", "L", "A1", "B1", "C1"))
        if not (6 <= max(ra, rb) <= 12 and min(ra, rb) >= 4 and labels <= 14):
            continue
        cases.append((f"random ranks {ra} x {rb}", r, rng.choice(((2, 3), (2, 2, 3), (1, 2, 2, 3))), {}))

    n_run = n_collapsed = n_checked = 0
    max_rank = 0
    for idx, (name, roles, ext, opt) in enumerate(cases):
        name = f"highrank[{idx}] {name}"
        for _ in range(100):               # re-draw extents until within the size guards
            A, B, C, v, collapsed = hr_case(rng, roles, ext)
            if "zero" in opt:
                lab = v.order[opt["zero"]][0]
                for d in (A, B, C):
                    d.extent = [0 if q == lab else e for q, e in zip(d.label, d.extent)]
                _, v = oracle_status(A, B, C)
            if v.b * v.m * v.n * max(v.k, 1) <= HR_MAX_MNK // 4 and all(
                    math.prod(d.extent) <= HR_MAX_ELEMS for d in (A, B, C)) and all(
                    buffer_for(d)[0] <= HR_MAX_BUFFER for d in (A, B, C)):
                break
        else:
            raise Fail(f"{name}: no draw within the size guards")
        N = rng.choice((3, 9, 14, 20))
        g = rng.choice((0, 1, rng.randint(1, N)))
        hr_guard(name, A, B, C, v, N, g)
        got = lowering_matches(lib, A, B, C, name)
        max_rank = max(max_rank, len(A.label), len(B.label))
        if not v.empty_output():
            n_checked += 1
            # the library's fused group counts (not just the oracle's) stay > 1
            lib_collapsed = any(len(got["fused"][q]) <= 1 for q in "IJKL" if len(got["order"][q]) >= 2)
            n_collapsed += lib_collapsed or collapsed
        mode = opt.get("mode", o.EMULATE)
        kind = rng.choice(("uniform", "phi", "ints"))
        TA, TB = gen(rng, A.extent, kind), gen(rng, B.extent, kind)
        if rng.random() < 0.2 and TA.size:
            inject(rng, TA, 0.01)
        run_case(lib, cuda, f"{name} {A.label},{B.label}->{C.label}", A, B, C, TA, TB,
                 gen(rng, C.extent, "uniform"), rng.choice((1.0, -0.5)), rng.choice((0.0, 0.5)),
                 N, g, rng.choice(("RN", "TRUNC")), mode=mode, rng=rng, verbose=verbose)
        n_run += 1
    for idx, perm in enumerate(ccsdt):
        ext = dict(i=3, j=2, k=3, a=4, b=3, c=2, e=17)
        A, B, C = einsum_descs(rng, f"ijae,ekbc->{perm}", ext, neg=0.3, pad=0.3,
                               orders=[rng.sample(range(4), 4), rng.sample(range(4), 4), None])
        _, v = oracle_status(A, B, C)
        name = f"highrank (T) 'ijae,ekbc->{perm}'"
        hr_guard(name, A, B, C, v, 14, 0)
        lowering_matches(lib, A, B, C, name)
        for mode in (o.EMULATE, o.DIRECT):
            TA, TB = gen(rng, A.extent, "phi"), gen(rng, B.extent, "phi")
            run_case(lib, cuda, f"{name} {mode}", A, B, C, TA, TB, gen(rng, C.extent, "uniform"),
                     1.0, 0.0, 14, 0, "RN", mode=mode, rng=rng,
                     expect_kind="GEMM" if mode == o.EMULATE else "DIRECT", verbose=verbose)
            n_run += 1
    # Fusion must not have degenerated the suite into a low-rank one.
    if n_collapsed > n_checked // 10:
        raise Fail(f"highrank: fusion collapsed a group in {n_collapsed} of {n_checked} cases")
    if verbose:
        print(f"  ok  {n_run} high-rank contractions (max operand rank {max_rank}; fusion "
              f"collapsed a group in {n_collapsed}/{n_checked}) in {time.time() - t0:.1f}s")


SUITES = {"lowering": suite_lowering, "legality": suite_legality, "random": suite_random,
          "fixtures": suite_fixtures, "robust": suite_robust, "highrank": suite_highrank}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--shim", help="tests/alloc_shim.c library, already in LD_PRELOAD")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
    if not lib.has_test_hooks:
        print("SKIP  library built without test hooks")
        return 0
    if args.shim:
        lib.load_alloc_shim(args.shim)
    t0 = time.time()
    try:
        for s in args.suite or list(SUITES):
            t1 = time.time()
            SUITES[s](lib, cuda, args.verbose)
            print(f"PASS  {s:8s} {time.time() - t1:6.1f}s", flush=True)
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"{STATS['cases']} contractions bit-exact vs spec/oracle.py "
          f"({STATS['elements']} C elements; dispatch {STATS['kinds']}; "
          f"{STATS['stage_bytes']} workspace bytes stage-checked) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
