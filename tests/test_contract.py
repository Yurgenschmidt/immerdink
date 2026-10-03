#!/usr/bin/env python3
"""Bit-exact GPU vs oracle tests of imd_contract (PLAN M1, DESIGN §3-§5).

    python3 tests/test_contract.py [--lib build/libimmerdink.so] [--suite NAME ...]

Every case runs the library on the GPU and spec/oracle.py's contract_tensor on the same
input bits, then compares:
  * the whole C buffer bit for bit (NaN == NaN regardless of payload: DESIGN does not
    specify NaN payloads), including elements C's descriptor does not address (untouched);
  * for the emulated path, every intermediate the workspace holds after the call, stage
    by stage (DESIGN §4a regions, offsets from spec/params.py region_offsets):
      K0  vec      the int64 scatter vectors            == oracle lower().vec_region()
      K1  sA sB    shifts sigma / tau (int16)           == oracle scale_vector
          fA fB    non-finite flags                     == oracle flags
      K2  RA       int8 residues of the LAST group, h < k; zero padding h >= k
      K3  RB       likewise, rows j < n; zero rows j >= n
      K4  W        u8 planes of all N moduli, j < n;    zero for j >= n
      K5  C        (above)
Suites: stages (random matrices, stage checks), e2e (random shapes/options/data), tensor
(strided / permuted / batched / diagonal / K-empty / k = 0 / batch split), bigk (k > K_c').
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
sys.path.insert(0, HERE)
import oracle as o  # noqa: E402
import params  # noqa: E402
import harness as hz  # noqa: E402

RND = {"RN": hz.RN, "TRUNC": hz.TRUNC}
MODE = {o.EMULATE: hz.EMULATE, o.DIRECT: hz.DIRECT}
STATS = {"cases": 0, "elements": 0, "stage_bytes": 0, "kinds": {}}


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

def rand_double(rng, emin=-1080, emax=1030):
    """Finite double with a wide exponent spread, including subnormals and extremes."""
    e = rng.randint(emin, emax)
    m = rng.getrandbits(53) | (1 << 52) if rng.random() < 0.9 else rng.getrandbits(rng.randint(1, 53))
    v = o.round_dyadic(m if rng.random() < 0.5 else -m, e - 52)
    return math.copysign(1.7976931348623157e308, v) if math.isinf(v) else v


def gen(rng, shape, kind):
    n = int(np.prod(shape)) if shape else 1
    if kind == "uniform":
        v = [rng.uniform(-1, 1) for _ in range(n)]
    elif kind == "phi":       # DESIGN's test generator, phi = 4: (rand - 0.5) exp(phi randn)
        v = [(rng.random() - 0.5) * math.exp(4 * rng.gauss(0, 1)) for _ in range(n)]
    elif kind == "wide":
        v = [rand_double(rng) for _ in range(n)]
    elif kind == "huge":
        v = [rand_double(rng, 900, 1023) for _ in range(n)]
    elif kind == "tiny":
        v = [rand_double(rng, -1080, -1000) for _ in range(n)]
    elif kind == "ints":
        v = [float(rng.randint(-3, 3)) for _ in range(n)]
    else:
        raise ValueError(kind)
    return np.array(v, dtype=np.float64).reshape(shape)


def inject(rng, T, rate):
    """Sprinkle Inf / -Inf / NaN / -0 / zeros into T (in place)."""
    flat = T.reshape(-1)
    for _ in range(max(1, int(rate * flat.size))):
        flat[rng.randrange(flat.size)] = rng.choice((math.inf, -math.inf, math.nan, -0.0, 0.0))


# ---------------------------------------------------------------------------------------
# Strided buffers
# ---------------------------------------------------------------------------------------

def layout(rng, ext, labels=None, order=None, neg=0.0, pad=0.0):
    """Strides for extents `ext` with mode order[0] fastest (default: last mode fastest),
    random negative signs (probability neg) and random gaps (probability pad)."""
    r = len(ext)
    order = list(range(r - 1, -1, -1)) if order is None else order
    st, acc = [0] * r, 1
    for d in order:
        acc += rng.randint(1, 3) if rng.random() < pad else 0
        st[d] = acc * (-1 if rng.random() < neg else 1)
        acc *= max(ext[d], 1)
    return st


def buffer_for(desc):
    """(length, base) of the smallest buffer holding every element of desc."""
    if desc.empty() or not desc.extent:
        return 1, 0
    lo = sum(min(0, s * (e - 1)) for e, s in zip(desc.extent, desc.stride))
    hi = sum(max(0, s * (e - 1)) for e, s in zip(desc.extent, desc.stride))
    return hi - lo + 1, -lo


def scatter(T, desc, fill):
    """Flat buffer (numpy) and base for logical tensor T (shape = desc.extent); elements
    the descriptor does not address hold `fill` values (a buffer-sized array)."""
    length, base = buffer_for(desc)
    buf = np.array(fill[:length] if fill is not None else np.zeros(length), dtype=np.float64)
    if T is not None and not desc.empty() and desc.extent:
        idx = np.indices(desc.extent).reshape(len(desc.extent), -1)
        off = base + sum(np.int64(s) * idx[d] for d, s in enumerate(desc.stride))
        buf[off] = T.reshape(-1)
    elif T is not None and not desc.extent:
        buf[base] = T.reshape(-1)[0]
    return buf, base


# ---------------------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------------------

def bit_mismatches(got, want):
    gb, wb = got.view(np.uint64), want.view(np.uint64)
    return np.nonzero((gb != wb) & ~(np.isnan(got) & np.isnan(want)))[0]


GUARD = 4096   # poisoned bytes before and after the workspace; must stay untouched (M3)
MAX_HOST_BUFFER = 64 << 20    # bytes per operand buffer (keeps every test process small)
MAX_WORKSPACE = 512 << 20     # device workspace bytes per case


def run_case(lib, cuda, name, A, B, Cd, TA, TB, TC, alpha, beta, N, g=0, rounding="RN",
             mode=o.EMULATE, rng=None, expect_kind=None, verbose=False, chunk=o.K_C_ALIGNED):
    """Run one contraction on the GPU and in the oracle; raise Fail on any difference.
    The workspace sits between two GUARD-byte poisoned zones that must stay untouched (the
    library touches only [ws, ws + workspace_size)); with the allocation shim loaded,
    imd_contract must not allocate. `chunk` is the oracle's K-chunk length (the library's
    is set with lib.set_limits). Returns the GPU's C buffer."""
    rng = rng or random.Random(0)
    lenA, _ = buffer_for(A)
    lenB, _ = buffer_for(B)
    lenC, _ = buffer_for(Cd)
    for X, n_el in (("A", lenA), ("B", lenB), ("C", lenC)):   # host memory guard
        if 8 * n_el > MAX_HOST_BUFFER:
            raise Fail(f"{name}: {X} buffer of {8 * n_el} bytes > {MAX_HOST_BUFFER} (test guard)")
    bufA, baseA = scatter(TA, A, gen(rng, (lenA,), "uniform"))
    bufB, baseB = scatter(TB, B, gen(rng, (lenB,), "uniform"))
    bufC, baseC = scatter(TC, Cd, gen(rng, (lenC,), "uniform"))

    if lib.has_test_hooks:   # size guard before the (pure Python) oracle materialises it
        st, lw = lib.lowering(hz.make_tensor(A.extent, A.stride, A.label),
                              hz.make_tensor(B.extent, B.stride, B.label),
                              hz.make_tensor(Cd.extent, Cd.stride, Cd.label))
        if st == "SUCCESS" and (lw["b"] * lw["m"] * lw["n"] * max(lw["k"], 1) > 2**27 or
                                2 * (lw["m"] + lw["n"] + lw["k"]) + 3 * lw["b"] > 2**22):
            raise Fail(f"{name}: view {lw['b']}x{lw['m']}x{lw['n']}x{lw['k']} too big (test guard)")

    # oracle
    t0 = time.time()
    trace = {}
    want = np.array(o.contract_tensor(o.Tensor(bufA.tolist(), baseA, A),
                                      o.Tensor(bufB.tolist(), baseB, B),
                                      o.Tensor(bufC.tolist(), baseC, Cd), alpha, beta, N,
                                      rounding, g, mode, chunk=chunk, trace=trace),
                    dtype=np.float64)
    t_oracle = time.time() - t0
    view, kind = trace["view"], trace["kind"]
    if expect_kind is not None and kind != expect_kind:
        raise Fail(f"{name}: oracle dispatch {kind}, expected {expect_kind}")

    # library
    tA = hz.make_tensor(A.extent, A.stride, A.label)
    tB = hz.make_tensor(B.extent, B.stride, B.label)
    tC = hz.make_tensor(Cd.extent, Cd.stride, Cd.label)
    st, ws_bytes = lib.workspace_size(tA, tB, tC, N, g, RND[rounding], MODE[mode])
    if st != "SUCCESS":
        raise Fail(f"{name}: imd_workspace_size {st}")
    gg = N if g in (0, None) else g
    pmode = params.DIRECT if mode == o.DIRECT else params.EMULATE
    want_ws = 0 if view.empty_output() else params.workspace_size(view.m, view.n, view.k, view.b,
                                                                   N, gg, pmode)
    if ws_bytes != want_ws:
        raise Fail(f"{name}: workspace {ws_bytes} != params {want_ws}")
    if ws_bytes > MAX_WORKSPACE:
        raise Fail(f"{name}: workspace {ws_bytes} > {MAX_WORKSPACE} (test guard)")
    st, plan = lib.plan_create(tA, tB, tC, N, g, RND[rounding], MODE[mode])
    if st != "SUCCESS":
        raise Fail(f"{name}: imd_plan_create {st}")
    dA, dB, dC = cuda.malloc(bufA.nbytes), cuda.malloc(bufB.nbytes), cuda.malloc(bufC.nbytes)
    dW = cuda.malloc(ws_bytes + 2 * GUARD)
    try:
        cuda.upload(dA, bufA)
        cuda.upload(dB, bufB)
        cuda.upload(dC, bufC)
        cuda.memset(dW, 0xA5, ws_bytes + 2 * GUARD)   # poison: every byte read must be written
        pA = (dA.value or 0) + 8 * baseA
        pB = (dB.value or 0) + 8 * baseB
        pC = (dC.value or 0) + 8 * baseC
        pW = dW.value + GUARD
        allocs = lib.alloc_count()
        t0 = time.time()
        st = lib.contract(plan, alpha, pA, pB, beta, pC, pW, ws_bytes)
        if st != "SUCCESS":
            raise Fail(f"{name}: imd_contract {st}")
        cuda.sync()
        t_gpu = time.time() - t0
        if allocs is not None and lib.alloc_count() != allocs:
            raise Fail(f"{name}: imd_contract allocated ({lib.alloc_count() - allocs} calls)")
        got = np.empty_like(bufC)
        cuda.download(got, dC)
        wsg = np.empty(ws_bytes + 2 * GUARD, dtype=np.uint8)
        cuda.download(wsg, dW)
        if (wsg[:GUARD] != 0xA5).any() or (wsg[GUARD + ws_bytes:] != 0xA5).any():
            raise Fail(f"{name}: bytes outside [ws, ws + workspace_size) were written")
        ws = wsg[GUARD:GUARD + max(ws_bytes, 1)]
    finally:
        for p in (dA, dB, dC, dW):
            cuda.free(p)
        lib.plan_destroy(plan)

    if kind == "GEMM":
        check_stages(name, view, trace, ws, N, gg)
    bad = bit_mismatches(got, want)
    if len(bad):
        i = bad[0]
        raise Fail(f"{name}: {len(bad)}/{got.size} C elements differ; first at {i}: "
                   f"gpu {got[i]!r} ({got.view(np.uint64)[i]:#018x}) oracle {want[i]!r} "
                   f"({want.view(np.uint64)[i]:#018x})")
    STATS["cases"] += 1
    STATS["elements"] += got.size
    STATS["kinds"][kind] = STATS["kinds"].get(kind, 0) + 1
    if verbose:
        print(f"  ok  {name}: {kind} m={view.m} n={view.n} k={view.k} b={view.b} N={N} g={gg} "
              f"{rounding} a={alpha} b={beta} swap={view.swapped} "
              f"(oracle {t_oracle:.2f}s, gpu {t_gpu:.3f}s)", flush=True)
    return got


def check_stages(name, view, trace, ws, N, g):
    """Compare every workspace region with the oracle's intermediates (see module doc)."""
    m, n, k, b = view.m, view.n, view.k, view.b
    kh, nh = -(-k // 16) * 16, -(-n // 16) * 16
    regs = {r: (s, e) for r, s, e in params.region_offsets(m, n, k, b, N, g, params.EMULATE)}

    def reg(r, dtype, shape):
        s, e = regs[r]
        return ws[s:e].view(dtype).reshape(shape)

    def need(ok, what):
        if not ok:
            raise Fail(f"{name}: stage {what} differs from the oracle")

    bt = trace["batches"]
    need(np.array_equal(reg("vec", np.int64, (-1,)), np.array(view.vec_region(), np.int64)),
         "K0 vec (scatter vectors)")
    need(np.array_equal(reg("sA", np.int16, (b, m)), np.array([t["sigma"] for t in bt])), "K1 sigma")
    need(np.array_equal(reg("sB", np.int16, (b, n)), np.array([t["tau"] for t in bt])), "K1 tau")
    need(np.array_equal(reg("fA", np.uint8, (b, m)), np.array([t["fA"] for t in bt], np.uint8)),
         "K1 fA")
    need(np.array_equal(reg("fB", np.uint8, (b, n)), np.array([t["fB"] for t in bt], np.uint8)),
         "K1 fB")
    last = list(range(N))[(N - 1) // g * g:]           # the last residue group
    RA = reg("RA", np.int8, (g, b, m, kh))
    RB = reg("RB", np.int8, (g, b, nh, kh))
    for s, l in enumerate(last):
        need(np.array_equal(RA[s, :, :, :k], np.array([t["RA"][l] for t in bt], np.int8)),
             f"K2 RA (modulus {l})")
        need(np.array_equal(RB[s, :, :n, :k], np.array([t["RB"][l] for t in bt], np.int8)),
             f"K3 RB (modulus {l})")
    need(not RA[:len(last), :, :, k:].any() and not RB[:len(last), :, :, k:].any()
         and not RB[:len(last), :, n:, :].any(), "K2/K3 zero padding")
    W = reg("W", np.uint8, (N, b, m, nh))
    want_W = np.array([[t["W"][l] for t in bt] for l in range(N)], np.uint8)
    need(np.array_equal(W[:, :, :, :n], want_W), "K4 W planes")
    need(not W[:, :, :, n:].any(), "K4 W padding columns")
    STATS["stage_bytes"] += ws.size


# ---------------------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------------------

I, J, K, L, X, Y, Z = 1, 2, 3, 4, 5, 6, 7
NS = (2, 3, 8, 9, 14, 16, 20)


def matrix(rng, m, n, k, a_kfast=None, b_kfast=None, c_jfast=True, neg=0.0, pad=0.0):
    """'ik,kj->ij' descriptors with random (or given) storage orders."""
    a_kfast = rng.random() < 0.5 if a_kfast is None else a_kfast
    b_kfast = rng.random() < 0.5 if b_kfast is None else b_kfast
    A = o.Desc([m, k], layout(rng, [m, k], order=[1, 0] if a_kfast else [0, 1], neg=neg, pad=pad), [I, K])
    B = o.Desc([k, n], layout(rng, [k, n], order=[0, 1] if b_kfast else [1, 0], neg=neg, pad=pad), [K, J])
    C = o.Desc([m, n], layout(rng, [m, n], order=[1, 0] if c_jfast else [0, 1], neg=neg, pad=pad), [I, J])
    return A, B, C


def suite_stages(lib, cuda, verbose):
    """Random matrices with stage-by-stage checks; one group (g = N) so RA/RB hold all N."""
    rng = random.Random(101)
    for t in range(28):
        N = NS[t % len(NS)]
        rounding = ("RN", "TRUNC")[t % 2]
        m, n = rng.randint(1, 48), rng.randint(1, 48)
        k = rng.choice((2, 5, 16, 17, 40, 97, 255, 600))
        A, B, C = matrix(rng, m, n, k)
        kind = ("uniform", "phi", "wide", "ints")[t % 4]
        TA, TB = gen(rng, (m, k), kind), gen(rng, (k, n), kind)
        if t % 3 == 0:
            TA[rng.randrange(m)] = 0.0                       # zero row
            TB[:, rng.randrange(n)] = 0.0                    # zero column
        if t % 4 == 1:
            inject(rng, TA, 0.03)                            # non-finite rows
            inject(rng, TB, 0.03)
        run_case(lib, cuda, f"stages[{t}]", A, B, C, TA, TB, gen(rng, (m, n), "uniform"),
                 1.0, 0.0, N, 0, rounding, rng=rng, expect_kind="GEMM", verbose=verbose)


def suite_e2e(lib, cuda, verbose):
    """Random shapes, options, layouts and data; alpha/beta special cases."""
    rng = random.Random(202)
    alphas, betas = (1.0, 0.0, -2.5, 0.3, 3.0), (0.0, 1.0, 0.7, -1.25)
    for t in range(100):
        N = rng.choice(NS)
        rounding = rng.choice(("RN", "TRUNC"))
        g = rng.choice((0, 1, N, rng.randint(1, N)))
        if t % 4 == 3:   # PLAN M1 range: m, n, k in [1, 3000], within the oracle's budget
            k = int(math.exp(rng.uniform(0, math.log(3000))))
            mn = max(1, int(2e6 / (k * N)))
            m = min(3000, int(math.exp(rng.uniform(0, math.log(mn)))))
            n = min(3000, max(1, mn // m))
        else:
            m, n = rng.randint(1, 40), rng.randint(1, 40)
            k = rng.choice((1, 2, 3, 7, 15, 16, 17, 31, 33, 64, 100, 129, 257, 1000))
        A, B, C = matrix(rng, m, n, k, c_jfast=rng.random() < 0.7, neg=0.2, pad=0.2)
        kind = rng.choice(("uniform", "phi", "wide", "huge", "tiny", "ints"))
        TA, TB = gen(rng, (m, k), kind), gen(rng, (k, n), rng.choice((kind, "uniform", "wide")))
        TC = gen(rng, (m, n), "uniform")
        if rng.random() < 0.3:
            TA[rng.randrange(m)] = 0.0
        if rng.random() < 0.3:
            inject(rng, TA if rng.random() < 0.5 else TB, 0.02)
        alpha, beta = rng.choice(alphas), rng.choice(betas)
        if beta == 0.0 and rng.random() < 0.6:
            TC[:] = math.nan                                 # beta = 0: C must not be read
        mode = o.DIRECT if rng.random() < 0.1 else o.EMULATE
        run_case(lib, cuda, f"e2e[{t}]", A, B, C, TA, TB, TC, alpha, beta, N, g, rounding,
                 mode=mode, rng=rng, verbose=verbose)
    # explicit special cases on one shape
    m, n, k = 9, 13, 21
    A, B, C = matrix(rng, m, n, k)
    TA, TB, TC = gen(rng, (m, k), "phi"), gen(rng, (k, n), "phi"), gen(rng, (m, n), "phi")
    TA[2, 3], TB[4, 5], TB[0, 7] = math.inf, math.nan, -math.inf
    TA[5] = 0.0
    TCn = TC.copy()
    TCn[1, 1] = math.nan
    for alpha, beta, Cin, label in ((1.0, 0.0, np.full((m, n), math.nan), "beta=0, C=NaN"),
                                    (0.0, 0.0, np.full((m, n), math.nan), "alpha=0, beta=0"),
                                    (0.0, -3.0, TC, "alpha=0 (A, B not read)"),
                                    (1.0, 1.0, TCn, "alpha=1, beta=1, NaN in C"),
                                    (-0.75, 0.5, TC, "alpha, beta other"),
                                    (-0.0, 2.0, TC, "alpha=-0")):
        for rounding in ("RN", "TRUNC"):
            run_case(lib, cuda, f"special[{label}, {rounding}]", A, B, C, TA, TB, Cin, alpha,
                     beta, 14, 5, rounding, rng=rng, verbose=verbose)


def suite_tensor(lib, cuda, verbose):
    """Strided / permuted / batched tensor contractions through the full lowering."""
    rng = random.Random(303)

    def T(ext, lab, order=None, neg=0.0, pad=0.0):
        return o.Desc(ext, layout(rng, ext, order=order, neg=neg, pad=pad), lab)

    def run(name, A, B, C, N=9, rounding="RN", alpha=1.0, beta=0.0, kind="phi", nonfinite=False,
            expect=None, g=0, mode=o.EMULATE):
        TA, TB = gen(rng, A.extent, kind), gen(rng, B.extent, kind)
        if nonfinite:
            inject(rng, TA, 0.01)
        run_case(lib, cuda, name, A, B, C, TA, TB, gen(rng, C.extent, "uniform"), alpha, beta,
                 N, g, rounding, mode=mode, rng=rng, expect_kind=expect, verbose=verbose)

    m, n, k, b = 5, 7, 19, 3
    # 'ikl,kjl->ijl' (batched), dense and with B k-fastest
    run("ikl,kjl->ijl", T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]), T([m, n, b], [I, J, L]),
        N=8, expect="GEMM")
    run("ikl,kjl->ijl (B k-fastest, l outer in C)", T([m, k, b], [I, K, L]),
        T([k, n, b], [K, J, L], order=[0, 2, 1]), T([m, n, b], [I, J, L], order=[1, 0, 2]),
        N=16, rounding="TRUNC", alpha=-1.5, beta=0.25, nonfinite=True, expect="GEMM")
    # permuted output with i fastest in C: the role swap (C^T = B^T A^T)
    run("kai,bkj->jbai (swap)", T([k, 3, m], [K, X, I]), T([2, k, n], [Y, K, J]),
        T([n, 2, 3, m], [J, Y, X, I]), N=14, expect="GEMM")
    # DF-like 'acQ,bdQ->acbd'
    run("acQ,bdQ->acbd", T([3, 4, 23], [I, X, K]), T([2, 5, 23], [J, Y, K]),
        T([3, 4, 2, 5], [I, X, J, Y]), N=20, beta=1.0, expect="GEMM")
    # negative and padded strides everywhere, beta != 0 (gaps of C stay untouched)
    run("ikl,kjl->ijl (neg/pad strides)", T([m, k, b], [I, K, L], neg=0.5, pad=0.5),
        T([k, n, b], [K, J, L], neg=0.5, pad=0.5), T([m, n, b], [I, J, L], neg=0.5, pad=0.5),
        N=9, beta=0.5, expect="GEMM")
    # diagonal in A and an A-only label (summed, stride 0 in B), K split over two labels
    run("iik x,kj->ij (diagonal + A-only)", T([m, m, 4, 3], [I, I, K, X]), T([4, n], [K, J]),
        T([m, n], [I, J]), N=3, expect="GEMM")
    run("ik1j (extent-1 modes dropped)", T([m, 1, k], [I, Z, K]), T([k, 1, n], [K, Y, J]),
        T([m, 1, n], [I, Y, J]), N=2, rounding="TRUNC", expect="GEMM")
    # K-empty -> DIRECT dispatch: outer product and batched Hadamard
    run("i,j->ij (K-empty)", T([m], [I]), T([n], [J]), T([m, n], [I, J]), kind="wide",
        expect="DIRECT")
    run("il,jl->ijl (K-empty, batched)", T([m, 4], [I, L]), T([n, 4], [J, L]),
        T([m, n, 4], [I, J, L]), alpha=2.0, beta=-1.0, nonfinite=True, expect="DIRECT")
    run("ikl,kjl->ijl DIRECT mode", T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]),
        T([m, n, b], [I, J, L]), alpha=0.5, beta=1.0, nonfinite=True, mode=o.DIRECT,
        expect="DIRECT")
    # k = 0 (a K label of extent 0): C <- beta C, A and B not read
    run("k=0", T([m, 0], [I, K]), T([0, n], [K, J]), T([m, n], [I, J]), beta=-2.0,
        expect="SCALE")
    run("k=0, beta=0", T([m, 0], [I, K]), T([0, n], [K, J]), T([m, n], [I, J]), expect="SCALE")
    # empty output: no-op
    run("empty output", T([0, k], [I, K]), T([k, n], [K, J]), T([0, n], [I, J]), expect="NOOP")
    # §3.7 native path order (oracle A7): the canonical K order decides between -Inf and NaN.
    # K = {k1, k2} (extent 2 each); big = A (m >= n); K sorted by |s_A|: the label with the
    # smaller A stride is fastest. Values by (k1, k2): (0,0) = 0, (0,1) = -Inf, (1,0) = (1,1)
    # = MAX, B = 2: k2 fastest -> -Inf; k1 fastest -> +Inf - Inf = NaN.
    MAX = 1.7976931348623157e308
    TA = np.array([[[0.0, -math.inf], [MAX, MAX]]])          # [i, k1, k2]
    TB = np.full((2, 2, 1), 2.0)                              # [k1, k2, j]
    outs = []
    for name, sa, mode in (("k2 fastest", [4, 2, 1], o.EMULATE), ("k1 fastest", [4, 1, 2], o.EMULATE),
                           ("k1 fastest, DIRECT", [4, 1, 2], o.DIRECT)):
        A, B, C = o.Desc([1, 2, 2], sa, [I, X, Y]), o.Desc([2, 2, 1], [2, 1, 1], [X, Y, J]), \
            o.Desc([1, 1], [1, 1], [I, J])
        run_case(lib, cuda, f"native K order ({name})", A, B, C, TA, TB, np.zeros((1, 1)), 1.0,
                 0.0, 9, rng=rng, mode=mode, verbose=verbose)
        outs.append(o.contract_tensor(o.Tensor(scatter(TA, A, None)[0].tolist(), 0, A),
                                      o.Tensor(scatter(TB, B, None)[0].tolist(), 0, B),
                                      o.Tensor([0.0], 0, C), 1.0, 0.0, 9, mode=mode)[0])
    if not (outs[0] == -math.inf and math.isnan(outs[1]) and math.isnan(outs[2])):
        raise Fail(f"native K order case does not separate the orders: {outs}")
    # m = n = 1 with b = 3300 and g = N = 20: 66000 batches -> the K4 batch-count split
    run("kl,kl->l (b=3300, 66000 GEMM batches)", T([3, 3300], [K, L]), T([3, 3300], [K, L]),
        T([3300], [L]), N=20, kind="uniform", expect="GEMM")


def suite_bigk(lib, cuda, verbose):
    """k > K_c' = 131056: two and three K-chunks (store-mod, then add-mod)."""
    rng = random.Random(404)
    k = 131056 + 37
    A, B, C = matrix(rng, 2, 3, k, a_kfast=True, b_kfast=False)
    TA, TB = gen(rng, (2, k), "phi"), gen(rng, (k, 3), "uniform")
    TB[5, 1] = math.inf
    run_case(lib, cuda, "bigk 2 chunks (2x3, N=3 RN)", A, B, C, TA, TB, gen(rng, (2, 3), "uniform"),
             1.0, 0.5, 3, 0, "RN", rng=rng, expect_kind="GEMM", verbose=verbose)
    k = 2 * 131056 + 100
    A, B, C = matrix(rng, 2, 2, k)
    TA, TB = gen(rng, (2, k), "wide"), gen(rng, (k, 2), "phi")
    TA[1, 7] = math.nan
    run_case(lib, cuda, "bigk 3 chunks (2x2, N=9 g=4 RN)", A, B, C, TA, TB, np.zeros((2, 2)),
             -1.0, 0.0, 9, 4, "RN", rng=rng, expect_kind="GEMM", verbose=verbose)
    k = 2 * 131056 + 5
    A, B, C = matrix(rng, 1, 1, k)
    TA, TB = gen(rng, (1, k), "uniform"), gen(rng, (k, 1), "uniform")
    run_case(lib, cuda, "bigk 3 chunks (1x1, N=2 TRUNC)", A, B, C, TA, TB, np.zeros((1, 1)),
             1.0, 0.0, 2, 0, "TRUNC", rng=rng, expect_kind="GEMM", verbose=verbose)


def suite_api(lib, cuda, verbose):
    """Plan-time statuses on the contraction path."""
    def mat(m, n, k):
        return (hz.make_tensor([m, k], [k, 1], [I, K]), hz.make_tensor([k, n], [n, 1], [K, J]),
                hz.make_tensor([m, n], [n, 1], [I, J]))
    # RN acceptance (kappa >= T -> NOT_SUPPORTED) only when the emulated GEMM runs
    checks = [(mat(2, 2, 179 * 179), 2, hz.RN, hz.EMULATE, "SUCCESS"),
              (mat(2, 2, 179 * 179 + 1), 2, hz.RN, hz.EMULATE, "NOT_SUPPORTED"),
              (mat(2, 2, 179 * 179 + 1), 2, hz.TRUNC, hz.EMULATE, "SUCCESS"),
              (mat(2, 2, 179 * 179 + 1), 2, hz.RN, hz.DIRECT, "SUCCESS"),
              (mat(2**31, 1, 2), 9, hz.RN, hz.EMULATE, "INVALID_VALUE"),   # GemmCoord (row 5)
              (mat(0, 1, 2), 9, hz.RN, hz.EMULATE, "SUCCESS")]
    for (A, B, C), N, rnd, mode, want in checks:
        st, plan = lib.plan_create(A, B, C, N, 0, rnd, mode)
        if plan is not None:
            lib.plan_destroy(plan)
        if st != want:
            raise Fail(f"plan status {st}, expected {want}")
    if verbose:
        print(f"  ok  plan-time statuses ({len(checks)} cases)")


def suite_replay(lib, cuda, verbose):
    """Every matrix contraction made by spec/test_oracle.py's hand-built boundary tests
    (edges (5), non-finite (4), determinism (3), adversarial A1-A8, the phi x k x N x
    rounding sweep (1)+(2)) replayed on the GPU:
    shift-rule boundaries, subnormal double-rounding traps, the single beta fma, native-path
    order, chunk splits... Calls with a non-default chunk / rule / word width are skipped
    (not library options); k = 1 calls are padded to k = 2 with a zero (the library would
    dispatch K-empty to DIRECT; the padded problem keeps the emulated path). Inputs that
    must not be read (Poison) become NaN."""
    import test_oracle as to
    calls = []
    real = o.contract

    def record(A, B, C, alpha, beta, N, rounding="RN", g=None, chunk=o.K_C_ALIGNED, m=None,
               n=None, k=None, trace=None, _rule=None, _L_w=None):
        out = real(A, B, C, alpha, beta, N, rounding, g, chunk, m, n, k, trace, _rule, _L_w)
        if chunk == o.K_C_ALIGNED and _rule is None and _L_w is None:
            calls.append((A, B, C, alpha, beta, N, rounding, g, m, n, k))
        return out

    o.contract = record
    try:
        quiet = open(os.devnull, "w")
        stdout, sys.stdout = sys.stdout, quiet
        try:
            for fn in (to.test_edges, to.test_nonfinite, to.test_determinism, to.test_adversarial,
                       to.test_sweep):
                fn()
        finally:
            sys.stdout = stdout
            quiet.close()
    finally:
        o.contract = real

    def mat(X, r, c):
        try:
            return np.array([[X[i][j] for j in range(c)] for i in range(r)], dtype=np.float64).reshape(r, c)
        except (AssertionError, TypeError, IndexError):
            return np.full((r, c), math.nan)          # Poison / None: must not be read

    rng = random.Random(505)
    done = 0
    for t, (A, B, C, alpha, beta, N, rounding, g, m, n, k) in enumerate(calls):
        m = len(A) if m is None else m
        k = (len(A[0]) if m else len(B)) if k is None else k
        n = (len(B[0]) if k else len(C[0])) if n is None else n
        if m == 0 or n == 0:
            continue
        TA, TB, TC = mat(A, m, k), mat(B, k, n), mat(C, m, n)
        if k == 1:
            TA = np.concatenate([TA, np.zeros((m, 1))], axis=1)
            TB = np.concatenate([TB, np.zeros((1, n))], axis=0)
            k = 2
        Ad, Bd, Cd = matrix(rng, m, n, k, a_kfast=True, b_kfast=False)
        run_case(lib, cuda, f"replay[{t}]", Ad, Bd, Cd, TA, TB, TC, alpha, beta, N, g or 0,
                 rounding, rng=rng, verbose=False)
        done += 1
    if verbose:
        print(f"  ok  {done} of {len(calls)} recorded oracle-test contractions replayed bit-exact")


SUITES = {"api": suite_api, "replay": suite_replay, "stages": suite_stages, "e2e": suite_e2e, "tensor": suite_tensor,
          "bigk": suite_bigk}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--shim", help="tests/alloc_shim.c library, already in LD_PRELOAD: "
                                   "imd_contract must not allocate")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    lib, cuda = hz.Library(args.lib), hz.Cuda()
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
