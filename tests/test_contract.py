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
(strided / permuted / batched / diagonal / K-empty / k = 0 / batch split), bigk (k > K_c'),
k_align (DESIGN v1.13 §4b stage 2: every producer path at A = 16 and A = 64, forced through
libimmerdink_host's hook; results identical across A; the K4 operand-pointer check).

Environment (DESIGN v1.13 §4b.8): IMD_TEST_POISON=0 leaves every workspace a fresh,
unwritten cudaMalloc (for compute-sanitizer --tool initcheck: a 0xA5 memset counts as
initialised; the guard-zone check is skipped); IMD_TEST_QUICK=1 runs each case in the
production rule, unfused (K2/K3) and KL with 64-row tiles only (sanitizer runs).
IMD_TEST_PAD_MUTATION = none / A / B: the mutation of suite pad_mutation under
IMD_TEST_POISON=0 (tests/initcheck_pad_mutation.sh: the initcheck mutation test).
"""

import argparse
import contextlib
import ctypes as ct
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
import occupancy_known as oc  # noqa: E402

POISON = os.environ.get("IMD_TEST_POISON", "1") != "0"
QUICK = os.environ.get("IMD_TEST_QUICK", "0") == "1"
RND = {"RN": hz.RN, "TRUNC": hz.TRUNC}
MODE = {o.EMULATE: hz.EMULATE, o.DIRECT: hz.DIRECT}
STATS = {"cases": 0, "elements": 0, "stage_bytes": 0, "kinds": {}, "k4": {}}
K4_CONFIGS = 4   # IMD_K4_CONFIG_COUNT (DESIGN v1.5 §4)


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


def gen_eps(rng, shape, kind, kaxis):
    """DESIGN v1.14 §3.4.1: eps-heavy operands. Every vector along the contracted axis `kaxis`
    (a row of the GEMM view, scaled by its own sigma) is one of
      sub1      one nonzero subnormal m 2^-1074 (m of 1..52 bits) at a random position: the
                row's sigma puts it near T, so the device's eps reaches eps_max(N) - bits + 1
      sub       subnormals of up to 1..20 bits in every entry (about 20 % zero)
      dominant  (u - 1/2) exp(g / 2) 2^s (s per row in [-20, 20)) with one entry x 2^20
                (imd_bench --data dominant; eps up to bitlen(T) - 53 on the dominant entry)
      phi       gen()'s phi = 4 data
    and `mix` picks one of them per vector."""
    k = shape[kaxis]
    rest = [d for i, d in enumerate(shape) if i != kaxis]
    flat = np.zeros((int(np.prod(rest)) if rest else 1, k))
    for r in range(flat.shape[0]):
        kd = rng.choice(("sub1", "sub", "dominant", "phi")) if kind == "mix" else kind
        if kd == "sub1":
            L = rng.randint(1, 52)
            m = rng.getrandbits(L) | (1 << (L - 1))
            flat[r, rng.randrange(k)] = math.ldexp(m, -1074) * rng.choice((1, -1))
        elif kd == "sub":
            L = rng.randint(1, 20)
            for h in range(k):
                if rng.random() >= 0.2:
                    flat[r, h] = math.ldexp(rng.getrandbits(L) | 1, -1074) * rng.choice((1, -1))
        elif kd == "dominant":
            sc = math.ldexp(1.0, rng.randint(-20, 19))
            for h in range(k):
                flat[r, h] = (rng.random() - 0.5) * math.exp(0.5 * rng.gauss(0, 1)) * sc
            flat[r, rng.randrange(k)] *= 2.0**20
        else:
            flat[r] = gen(rng, (k,), "phi")
    X = flat.reshape(rest + [k])
    return np.moveaxis(X, -1, kaxis).copy()


def eps_reached(T, kaxis, N, rounding):
    """The device's eps (src/kernels/numerics.cuh quantize) of every nonzero quantized entry
    of T's vectors along kaxis, with the §3.2 rule's sigma (spec/params.py scale_row)."""
    out = []
    X = np.moveaxis(T, kaxis, -1).reshape(-1, T.shape[kaxis])
    for row in X:
        vals = [float(v) if math.isfinite(v) else 0.0 for v in row]
        sig, _, _ = params.scale_row(vals, N, rounding)
        out += [params._dev_eps(v, sig) for v in vals if params.quantize(v, sig, rounding)[0] != 0]
    return out


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

@contextlib.contextmanager
def k_align(lib, a):
    """DESIGN v1.13 §4b.5: A forced to 16 / 64 (0 / "AUTO": the rule) in the library (the hook
    imd_test_set_k_align) and in the spec side (params / oracle K_ALIGN_POLICY: every default
    k_hat_of, workspace_size, region_offsets, pack follows); AUTO restored on exit."""
    a = 0 if a in (0, "AUTO") else int(a)
    lib.set_k_align(a)
    params.K_ALIGN_POLICY = o.K_ALIGN_POLICY = "AUTO" if a == 0 else a
    try:
        yield
    finally:
        lib.set_k_align(0)
        params.K_ALIGN_POLICY = o.K_ALIGN_POLICY = "AUTO"


def k4_tune_bytes(view, N, g, kh=None):
    """DESIGN v1.5 §6 k4_tune: group 1's RA, RB and W planes, each 256-aligned, in one
    temporary allocation; 0 (no tuning) above 256 MiB. kh: the plan's k_hat (default: the
    policy's A for view.k, v1.13 §4b)."""
    kh = params.k_hat_of(view.k, params.k_align_of(view.k)) if kh is None else kh
    nh, gb = -(-view.n // 16) * 16, min(g, N) * view.b
    regs = [gb * view.m * kh, gb * nh * kh, gb * view.m * nh]
    total = sum(-(-r // 256) * 256 for r in regs)
    return 0 if max(regs) > 256 << 20 or total > 256 << 20 else total


def bit_mismatches(got, want):
    gb, wb = got.view(np.uint64), want.view(np.uint64)
    return np.nonzero((gb != wb) & ~(np.isnan(got) & np.isnan(want)))[0]


GUARD = 4096   # poisoned bytes before and after the workspace; must stay untouched (M3)
MAX_HOST_BUFFER = 64 << 20    # bytes per operand buffer (keeps every test process small)
MAX_WORKSPACE = 512 << 20     # device workspace bytes per case


def run_case(lib, cuda, name, A, B, Cd, TA, TB, TC, alpha, beta, N, g=0, rounding="RN",
             mode=o.EMULATE, rng=None, expect_kind=None, verbose=False, chunk=o.K_C_ALIGNED,
             k4_expect=None):
    """Run one contraction on the GPU and in the oracle; raise Fail on any difference.
    `k4_expect` (test hooks, GEMM): {"k_hat", "store", "add", "launches"} that every plan's K4
    footprint (imd_test_plan_sizes: k_hat, store-/add-mod shape keys, launches) must equal.
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

    # library, in both scatter-vector modes (DESIGN v1.2 vec_mode): each bit-exact vs the
    # oracle, stage-checked, and the two C buffers identical bit for bit (NaN payloads too)
    tA = hz.make_tensor(A.extent, A.stride, A.label)
    tB = hz.make_tensor(B.extent, B.stride, B.label)
    tC = hz.make_tensor(Cd.extent, Cd.stride, Cd.label)
    gg = N if g in (0, None) else g
    pmode = params.DIRECT if mode == o.DIRECT else params.EMULATE
    results = {}
    t_gpu = 0.0
    # v1.4: each vec mode with the plan-time fused-path rule (fmode -1) and, with the test
    # hooks, with the fused path disabled (fmode 0): every run bit-exact, all C identical.
    # fmode 1: the plan-time rule, but without the small-problem launch variants (K5 one
    # element per thread, KF four values per thread).
    # v1.6: fmode 2 forces the fused large launch KL (K1 + group 1's pack per operand) for
    # both operands of every emulated case, fmode 3 the same with 64-row KL tiles, so rows
    # longer than 64 span several chunks (registers, shared memory and re-read chunks).
    fmodes = ((-1, 0, 1) + ((2, 3) if kind == "GEMM" else ())) if lib.has_test_hooks else (-1,)
    if QUICK:
        fmodes = tuple(f for f in fmodes if f in (-1, 0, 3))
    # v1.5 (DESIGN §4 "K4 configurations"): k4 -1 is the plan-time rule; with the test hooks
    # every emulated case also runs once in each K4 configuration (forced) and once with the
    # k4_tune plan option: all bit-exact vs the oracle and bit-identical to each other.
    runs = [(vec, pvec, fm, -1) for vec, pvec in ((hz.VEC_GENERATE, params.VEC_GENERATE),
                                                 (hz.VEC_RESIDENT, params.VEC_RESIDENT))
            for fm in fmodes]
    if QUICK:
        runs = [r for r in runs if r[0] == hz.VEC_GENERATE]
    elif lib.has_test_hooks and kind == "GEMM":
        runs += [(hz.VEC_GENERATE, params.VEC_GENERATE, -1, c)
                 for c in list(range(K4_CONFIGS)) + ["tune"]]
    # v1.13 §4b: the plan's K alignment (the policy's A for view.k; DIRECT etc.: no residue rows)
    A_plan = params.k_align_of(view.k)
    kh_plan = params.k_hat_of(view.k, A_plan)
    for vec, pvec, fmode, k4 in runs:
        vname = f"{name} [{pvec}{ {-1: '', 0: ', unfused', 1: ', large launches', 2: ', KL', 3: ', KL 64-row tiles'}[fmode]}" \
                f"{'' if k4 == -1 else f', K4 {k4}'}]"
        st, ws_bytes = lib.workspace_size(tA, tB, tC, N, g, RND[rounding], MODE[mode], vec)
        if st != "SUCCESS":
            raise Fail(f"{vname}: imd_workspace_size {st}")
        want_ws = 0 if view.empty_output() else params.workspace_size(view.m, view.n, view.k, view.b,
                                                                       N, gg, pmode, pvec)
        if ws_bytes != want_ws:
            raise Fail(f"{vname}: workspace {ws_bytes} != params {want_ws}")
        if ws_bytes > MAX_WORKSPACE:
            raise Fail(f"{vname}: workspace {ws_bytes} > {MAX_WORKSPACE} (test guard)")
        allocs0 = lib.alloc_count()
        if lib.has_test_hooks:
            lib.set_fused(fmode if fmode in (0, 2, 3) else -1)
            lib.set_k4_config(k4 if isinstance(k4, int) else -1)
        try:
            st, plan = lib.plan_create(tA, tB, tC, N, g, RND[rounding], MODE[mode], vec,
                                       k4_tune=1 if k4 == "tune" else 0)
        finally:
            if lib.has_test_hooks:
                lib.set_fused(-1)
                lib.set_k4_config(-1)
        if st != "SUCCESS":
            raise Fail(f"{vname}: imd_plan_create {st}")
        # CUDA allocations of plan creation (shim): one device allocation for resident
        # vectors, none otherwise (no pinned memory in v1.2). The resident count also proves
        # the shim interposes the library's runtime calls. v1.5: k4_tune makes one temporary
        # allocation of k4_tune_bytes (freed before imd_plan_create returns) when nonzero.
        if allocs0 is not None:
            want_allocs = 1 if (pvec == params.VEC_RESIDENT and not view.empty_output()) else 0
            if k4 == "tune" and k4_tune_bytes(view, N, gg, kh_plan) > 0:
                want_allocs += 1
            if lib.alloc_count() - allocs0 != want_allocs:
                raise Fail(f"{vname}: imd_plan_create made {lib.alloc_count() - allocs0} CUDA "
                           f"allocations, expected {want_allocs}")
        if lib.has_test_hooks and kind == "GEMM":
            cfg, k4p_cfg, _ = lib.plan_k4(plan)
            if cfg != k4p_cfg or not 0 <= cfg < K4_CONFIGS or (isinstance(k4, int) and k4 >= 0
                                                              and cfg != k4):
                lib.plan_destroy(plan)
                raise Fail(f"{vname}: plan K4 configuration {cfg} / {k4p_cfg}, wanted {k4}")
            STATS["k4"][cfg] = STATS["k4"].get(cfg, 0) + 1
            got_kh = lib.plan_sizes(plan)[7]   # v1.13 §4b: the real plan's k_hat
            if got_kh != kh_plan:   # read before plan_destroy (review 2026-10-06 must-fix 1)
                lib.plan_destroy(plan)
                raise Fail(f"{vname}: plan k_hat {got_kh}, want {kh_plan}")
            if k4_expect is not None:
                v = lib.plan_sizes(plan)
                got_k4 = {"k_hat": v[7], "store": v[28], "add": v[29], "launches": v[30]}
                if got_k4 != k4_expect:
                    lib.plan_destroy(plan)
                    raise Fail(f"{vname}: plan K4 footprint {got_k4}, expected {k4_expect}")
        dA, dB, dC = cuda.malloc(bufA.nbytes), cuda.malloc(bufB.nbytes), cuda.malloc(bufC.nbytes)
        dW = cuda.malloc(ws_bytes + 2 * GUARD)
        try:
            st, dev_bytes = lib.plan_device_bytes(plan)
            want_dev = params.plan_device_bytes(view.m, view.n, view.k, view.b, pvec)
            if st != "SUCCESS" or dev_bytes != want_dev:
                raise Fail(f"{vname}: imd_plan_device_bytes {st} {dev_bytes} != params {want_dev}")
            plan_vec = None
            if lib.has_test_hooks:
                addr, cnt = lib.plan_vec(plan)
                if (addr is not None) != (want_dev > 0) or (addr and 8 * cnt != want_dev):
                    raise Fail(f"{vname}: resident vectors {addr} x {cnt} vs {want_dev} bytes")
                if addr:
                    plan_vec = np.empty(cnt, dtype=np.int64)
                    cuda.download_raw(plan_vec, addr)
            cuda.upload(dA, bufA)
            cuda.upload(dB, bufB)
            cuda.upload(dC, bufC)
            if POISON:   # poison: every byte read must be written (else initcheck's fresh memory)
                cuda.memset(dW, 0xA5, ws_bytes + 2 * GUARD)
            pA = (dA.value or 0) + 8 * baseA
            pB = (dB.value or 0) + 8 * baseB
            pC = (dC.value or 0) + 8 * baseC
            pW = dW.value + GUARD
            allocs = lib.alloc_count()
            launches0 = lib.launch_count()
            t0 = time.time()
            if lib.has_test_hooks:   # the unfused runs also take K5's 4- (KL: 2-) per-thread mapping
                lib.set_k5_mapping(fmode)
            try:
                st = lib.contract(plan, alpha, pA, pB, beta, pC, pW, ws_bytes)
            finally:
                if lib.has_test_hooks:
                    lib.reset_k5_mapping()
            if st != "SUCCESS":
                raise Fail(f"{vname}: imd_contract {st}")
            # v1.4: the kernel launches of the call are what the plan (and libimmerdink_host's
            # imd_lowering kernel_launches, tests/test_host.py) says.
            if launches0 is not None and lib.has_test_hooks:
                fused, kl = lib.plan_kernels(plan)
                large = lib.plan_fused_large(plan)
                if fmode == 0 and (fused or large):
                    raise Fail(f"{vname}: the fused paths were not disabled")
                if fmode in (2, 3) and (fused or large != 3):
                    raise Fail(f"{vname}: KL not forced (fused {fused}, fused_large {large})")
                emulated = mode == o.EMULATE and view.k > 1 and not view.empty_output()
                if fmode in (-1, 1) and large != (3 if emulated and not fused else 0):
                    raise Fail(f"{vname}: fused_large {large} against the plan-time rule")
                if alpha == 0.0 and not view.empty_output():
                    kl = (0 if pvec == params.VEC_RESIDENT else 1) + 1   # K0 + beta kernel
                if lib.launch_count() - launches0 != kl:
                    raise Fail(f"{vname}: imd_contract launched {lib.launch_count() - launches0} "
                               f"kernels, the plan says {kl}")
                STATS["fused"] = STATS.get("fused", 0) + (1 if fused else 0)
                STATS["kl"] = STATS.get("kl", 0) + (1 if large else 0)
            cuda.sync()
            t_gpu += time.time() - t0
            if allocs is not None and lib.alloc_count() != allocs:
                raise Fail(f"{vname}: imd_contract allocated ({lib.alloc_count() - allocs} calls)")
            got = np.empty_like(bufC)
            cuda.download(got, dC)
            wsg = np.empty(ws_bytes + 2 * GUARD, dtype=np.uint8)
            cuda.download(wsg, dW)
            if POISON and ((wsg[:GUARD] != 0xA5).any() or (wsg[GUARD + ws_bytes:] != 0xA5).any()):
                raise Fail(f"{vname}: bytes outside [ws, ws + workspace_size) were written")
            ws = wsg[GUARD:GUARD + max(ws_bytes, 1)]
        finally:
            for p in (dA, dB, dC, dW):
                cuda.free(p)
            lib.plan_destroy(plan)
        if plan_vec is not None and not np.array_equal(plan_vec, np.array(view.vec_region(), np.int64)):
            raise Fail(f"{vname}: resident scatter vectors differ from the oracle's")
        if kind == "GEMM":
            check_stages(vname, view, trace, ws, N, gg, pvec, A_plan)
        bad = bit_mismatches(got, want)
        if len(bad):
            i = bad[0]
            raise Fail(f"{vname}: {len(bad)}/{got.size} C elements differ; first at {i}: "
                       f"gpu {got[i]!r} ({got.view(np.uint64)[i]:#018x}) oracle {want[i]!r} "
                       f"({want.view(np.uint64)[i]:#018x})")
        results[(pvec, fmode, k4)] = got
    got = results[(params.VEC_GENERATE, -1, -1)]
    for key, other in results.items():
        if not np.array_equal(got.view(np.uint64), other.view(np.uint64)):
            raise Fail(f"{name}: C differs between runs {(params.VEC_GENERATE, -1, -1)} and {key}")
    STATS["cases"] += 1
    STATS["elements"] += got.size
    STATS["kinds"][kind] = STATS["kinds"].get(kind, 0) + 1
    if verbose:
        print(f"  ok  {name}: {kind} m={view.m} n={view.n} k={view.k} b={view.b} N={N} g={gg} "
              f"{rounding} a={alpha} b={beta} swap={view.swapped} "
              f"(oracle {t_oracle:.2f}s, gpu {t_gpu:.3f}s)", flush=True)
    return got


def check_stages(name, view, trace, ws, N, g, vec=params.VEC_GENERATE, A=None):
    """Compare every workspace region with the oracle's intermediates (see module doc). With
    vec = RESIDENT there is no vec region (run_case checks the plan's copy instead). A: the
    plan's K alignment (v1.13 §4b; None: the policy's for k); RA / RB rows are k_hat = ceil(k)_A
    and their K padding [k, k_hat) must be zero (whichever producer wrote the last group: KF,
    KL, K2 / K3)."""
    m, n, k, b = view.m, view.n, view.k, view.b
    A = params.k_align_of(k) if A is None else A
    kh, nh = params.k_hat_of(k, A), -(-n // 16) * 16
    regs = {r: (s, e) for r, s, e in params.region_offsets(m, n, k, b, N, g, params.EMULATE, vec,
                                                           k_align=A)}

    def reg(r, dtype, shape):
        s, e = regs[r]
        return ws[s:e].view(dtype).reshape(shape)

    def need(ok, what):
        if not ok:
            raise Fail(f"{name}: stage {what} differs from the oracle")

    bt = trace["batches"]
    if "vec" in regs:
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


def suite_every_n(lib, cuda, verbose):
    """A5: K5 is one kernel per N (src/contract.cu k5_crt_n, k5_crt_bfast<N>), so every N in
    [kNMin, kNMax] runs here: a matrix (C j-bar fastest) and a batched case with C's batch
    label fastest (K5's lambda-fastest kernel), with zero rows and non-finite entries; the
    usual runs of run_case add K5's other per-thread mappings (4 and, forced KL, 2)."""
    rng = random.Random(909)
    for N in range(params.N_MIN, params.N_MAX + 1):
        rounding = ("RN", "TRUNC")[N % 2]
        m, n = rng.randint(3, 40), rng.randint(3, 40)
        k = rng.choice((5, 16, 33, 129))
        A, B, C = matrix(rng, m, n, k, c_jfast=True)
        kind = ("phi", "wide", "uniform", "ints")[N % 4]
        TA, TB = gen(rng, (m, k), kind), gen(rng, (k, n), kind)
        TA[rng.randrange(m)] = 0.0
        inject(rng, TB, 0.02)
        run_case(lib, cuda, f"every_n[N={N}]", A, B, C, TA, TB, gen(rng, (m, n), "uniform"),
                 (1.0, -0.75)[N % 2], (0.0, 0.5)[N % 3 == 0], N, 0, rounding, rng=rng,
                 expect_kind="GEMM", verbose=verbose)
        mb, nb, kb, b = rng.randint(2, 9), rng.randint(2, 9), rng.choice((3, 17)), rng.randint(3, 6)
        A = o.Desc([mb, kb, b], layout(rng, [mb, kb, b]), [I, K, L])
        B = o.Desc([kb, nb, b], layout(rng, [kb, nb, b]), [K, J, L])
        C = o.Desc([mb, nb, b], layout(rng, [mb, nb, b], order=[2, 1, 0]), [I, J, L])  # l fastest
        TA, TB = gen(rng, A.extent, kind), gen(rng, B.extent, kind)
        inject(rng, TA, 0.02)
        run_case(lib, cuda, f"every_n[N={N}, batch fastest]", A, B, C, TA, TB,
                 gen(rng, C.extent, "uniform"), 1.0, 0.0, N, 0, rounding, rng=rng,
                 expect_kind="GEMM", verbose=verbose)


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
    """k > K_c' = 131008 (v1.13): two and three K-chunks (store-mod, then add-mod), including the
    windows k_hat in (j K_c', j 131056] that gained a chunk with v1.13."""
    KP = o.K_C_ALIGNED
    rng = random.Random(404)
    k = KP + 37
    A, B, C = matrix(rng, 2, 3, k, a_kfast=True, b_kfast=False)
    TA, TB = gen(rng, (2, k), "phi"), gen(rng, (k, 3), "uniform")
    TB[5, 1] = math.inf
    run_case(lib, cuda, "bigk 2 chunks (2x3, N=3 RN)", A, B, C, TA, TB, gen(rng, (2, 3), "uniform"),
             1.0, 0.5, 3, 0, "RN", rng=rng, expect_kind="GEMM", verbose=verbose)
    k = 2 * KP + 100
    A, B, C = matrix(rng, 2, 2, k)
    TA, TB = gen(rng, (2, k), "wide"), gen(rng, (k, 2), "phi")
    TA[1, 7] = math.nan
    run_case(lib, cuda, "bigk 3 chunks (2x2, N=9 g=4 RN)", A, B, C, TA, TB, np.zeros((2, 2)),
             -1.0, 0.0, 9, 4, "RN", rng=rng, expect_kind="GEMM", verbose=verbose)
    k = 2 * KP + 5
    A, B, C = matrix(rng, 1, 1, k)
    TA, TB = gen(rng, (1, k), "uniform"), gen(rng, (k, 1), "uniform")
    run_case(lib, cuda, "bigk 3 chunks (1x1, N=2 TRUNC)", A, B, C, TA, TB, np.zeros((1, 1)),
             1.0, 0.0, 2, 0, "TRUNC", rng=rng, expect_kind="GEMM", verbose=verbose)
    # v1.13 §4b windows: k_hat in (K_c', 131056] is two chunks (131008 + a 16..48 tail; one
    # chunk before v1.13), k_hat = K_c' exactly is one; k_hat in (2 K_c', 2 * 131056] is three.
    # Both operands both ways round (K-fast and not), every case bit-exact vs the oracle.
    # Stage 2 (must-fix 7): no multiple of 64 lies in the windows, so under AUTO these would
    # be 64-aligned tails; the windows are A = 16 rows, pinned through the hook.
    with k_align(lib, 16):
        bigk_windows(lib, cuda, rng, verbose)
    bigk_full_chunks(lib, cuda, rng, verbose)
    bigk_long_rows(lib, cuda, rng, verbose)


def bigk_windows(lib, cuda, rng, verbose):
    KP = o.K_C_ALIGNED
    for (m, n, k, N, g, rnd, a_kf, b_kf) in (
            (3, 2, KP, 8, 3, "RN", True, False),            # one chunk, 2047 whole K tiles
            (2, 3, KP + 1, 9, 4, "RN", False, True),        # k_hat = KP + 16: tail 16
            (3, 3, 131040, 3, 0, "TRUNC", True, True),      # tail 32
            (2, 2, 131050, 14, 5, "RN", False, False),      # k_hat = 131056: tail 48
            (1, 2, 131056, 2, 0, "TRUNC", True, False),
            (2, 1, 2 * KP + 48, 8, 0, "RN", False, True)):  # k_hat = 262064: three chunks
        A, B, C = matrix(rng, m, n, k, a_kfast=a_kf, b_kfast=b_kf)
        TA, TB = gen(rng, (m, k), "phi"), gen(rng, (k, n), "wide")
        if k % 2 == 0:
            TB[k - 1, 0] = -math.inf          # in the last chunk's tail
        chunks = -(-params.k_hat_of(k, 16) // KP)
        gg = N if g == 0 else g
        groups = [gg] * (N // gg) + ([N % gg] if N % gg else [])
        expect = {"k_hat": params.k_hat_of(k, 16), "store": len(set(groups)),
                  "add": len(set(groups)) * params.k4_add_chunk_keys(params.k_hat_of(k, 16)),
                  "launches": chunks * len(groups)}
        run_case(lib, cuda, f"bigk window k={k} ({chunks} chunks, {m}x{n}, N={N} g={g} {rnd})", A, B,
                 C, TA, TB, gen(rng, (m, n), "uniform"), 1.0, 0.25, N, g, rnd, rng=rng,
                 expect_kind="GEMM", verbose=verbose, k4_expect=expect)


def bigk_full_chunks(lib, cuda, rng, verbose):
    KP = o.K_C_ALIGNED
    # Review 2026-10-05: whole-chunk multiples of K_c'. k_hat = 2 K_c' is two full chunks, so
    # the add-mod launch has the store-mod kc (131008); k_hat = 3 K_c' is three full chunks,
    # and the add-mod shape of the last chunk dedupes with the full one (one add-mod shape
    # key). The chunk counts are asserted here (params.k_chunk_list) and on the plan
    # (launches = chunks x groups; m, n tiny so one batch part and one n part per chunk).
    for (m, n, nch, N, g, rnd, a_kf, b_kf) in (
            (2, 1, 2, 3, 0, "RN", True, False),             # k_hat = 262016
            (1, 2, 3, 4, 3, "TRUNC", False, True)):         # k_hat = 393024, groups 3 + 1
        k = nch * KP
        A_k = params.k_align_of(k)            # v1.13 §4b: k = 0 mod 64, k_hat = k for any A
        chunks = params.k_chunk_list(params.k_hat_of(k, A_k))
        if params.k_hat_of(k, A_k) != k or chunks != [(c * KP, KP) for c in range(nch)]:
            raise Fail(f"bigk k={k}: chunks {chunks}, expected {nch} full chunks of {KP}")
        gg = N if g == 0 else g
        groups = [gg] * (N // gg) + ([N % gg] if N % gg else [])
        expect = {"k_hat": k, "store": len(set(groups)), "add": len(set(groups)),
                  "launches": nch * len(groups)}
        A, B, C = matrix(rng, m, n, k, a_kfast=a_kf, b_kfast=b_kf)
        TA, TB = gen(rng, (m, k), "phi"), gen(rng, (k, n), "wide")
        TA[0, k - 3] = math.inf                # in the last full chunk
        run_case(lib, cuda, f"bigk k={k} ({nch} full chunks, {m}x{n}, N={N} g={g} {rnd})", A, B,
                 C, TA, TB, gen(rng, (m, n), "uniform"), 1.0, -0.5, N, g, rnd, rng=rng,
                 expect_kind="GEMM", verbose=verbose, k4_expect=expect)


def bigk_long_rows(lib, cuda, rng, verbose):
    # A4: long rows with more rows than K1's L2-capped grid (k = 32768: 160 rows in flight),
    # so K1 takes two 512-thread blocks per SM (contract.cu launch_k1).
    k = 32768
    A, B, C = matrix(rng, 170, 1, k, a_kfast=True, b_kfast=True)
    TA, TB = gen(rng, (170, k), "phi"), gen(rng, (k, 1), "uniform")
    run_case(lib, cuda, "long rows (170x1, k=32768, N=3 RN)", A, B, C, TA, TB,
             np.zeros((170, 1)), 1.0, 0.0, 3, 0, "RN", rng=rng, expect_kind="GEMM", verbose=verbose)
    # A3: rows too long for two per SM in L2 (2 x 128 x 8 k > 64 MiB): one 1024-thread block
    # per SM, grid-strided over more rows than blocks; pass 2 forward (128 x 320 KB fit).
    k = 40000
    A, B, C = matrix(rng, 130, 1, k, a_kfast=True, b_kfast=True)
    TA, TB = gen(rng, (130, k), "wide"), gen(rng, (k, 1), "uniform")
    run_case(lib, cuda, "longer rows (130x1, k=40000, N=2 TRUNC)", A, B, C, TA, TB,
             np.zeros((130, 1)), 1.0, 0.0, 2, 0, "TRUNC", rng=rng, expect_kind="GEMM",
             verbose=verbose)
    # ... and rows in flight beyond 64 MiB (60 x 1.2 MB): pass 2 backwards; two K-chunks.
    k = 150000
    A, B, C = matrix(rng, 60, 1, k, a_kfast=True, b_kfast=True)
    TA, TB = gen(rng, (60, k), "phi"), gen(rng, (k, 1), "uniform")
    TA[7, 1234] = math.nan
    global MAX_HOST_BUFFER
    saved, MAX_HOST_BUFFER = MAX_HOST_BUFFER, 80 << 20   # 72 MB of A: more than the budget
    try:
        run_case(lib, cuda, "very long rows (60x1, k=150000, N=3 TRUNC)", A, B, C, TA, TB,
                 np.zeros((60, 1)), 1.0, 0.0, 3, 0, "TRUNC", rng=rng, expect_kind="GEMM",
                 verbose=verbose)
    finally:
        MAX_HOST_BUFFER = saved


def suite_k_align(lib, cuda, verbose):
    """DESIGN v1.13 §4b.8 (stage 2): every producer path at A = 16 and A = 64 (forced through
    libimmerdink_host's hook imd_test_set_k_align), each run bit-exact vs the oracle on a
    0xA5-poisoned workspace with the K padding [k, k_hat) checked zero (run_case's fused
    modes: the rule (KF for k_hat <= 1024, else KL), unfused K2 / K3 with set_fused(0), KL
    forced and KL with 64-row tiles; every K4 configuration and k4_tune), and the results of
    A = 16 and A = 64 identical bit for bit. Also: the K-chunk limit at multiples of 64 and at
    an odd multiple of 16 (correctness only), and the K4 operand-pointer check (= 0 mod A)
    with no violation, shown to run (checked launches > 0) and to fire (a mutation)."""
    if not lib.has_test_hooks:
        raise Fail("suite k_align needs the test hooks")
    KP = o.K_C_ALIGNED
    lib.k4_align_check(1)
    cases = [  # (label, m, n, k, b, N, g, rounding, a_kfast, b_kfast, k_chunk)
        ("KF A16", 7, 9, 33, 1, 8, 0, "RN", True, False, 0),
        ("KF A16 edge", 6, 5, 96, 1, 9, 0, "TRUNC", False, True, 0),
        ("KF", 7, 9, 97, 1, 8, 0, "RN", True, True, 0),
        ("KF", 5, 11, 112, 2, 14, 0, "TRUNC", False, False, 0),
        ("KF", 9, 4, 600, 1, 9, 0, "RN", True, False, 0),
        ("KF", 3, 17, 1000, 2, 8, 0, "RN", False, True, 0),
        ("KF 1008", 4, 6, 1008, 1, 8, 0, "TRUNC", True, True, 0),
        ("KL", 5, 6, 1040, 1, 9, 0, "RN", True, False, 0),
        ("KL", 3, 18, 4080, 1, 8, 0, "TRUNC", False, True, 0),
        ("K2/K3 g<N", 7, 20, 97, 1, 9, 4, "RN", False, False, 0),
        ("K2/K3 g<N", 5, 3, 144, 2, 9, 2, "TRUNC", True, True, 0),
        ("K2/K3 g<N", 4, 33, 1040, 1, 9, 4, "RN", True, False, 0),
        ("small k", 4, 5, 1, 1, 8, 0, "RN", True, True, 0),
        ("small k", 4, 5, 16, 1, 8, 0, "RN", False, True, 0),
        ("small k", 4, 5, 17, 1, 8, 0, "TRUNC", True, False, 0),
        ("small k", 4, 5, 48, 1, 8, 0, "RN", True, True, 0),
        ("k_chunk 192", 6, 5, 1000, 1, 8, 3, "RN", True, False, 192),
        ("k_chunk 64", 3, 4, 300, 1, 3, 0, "TRUNC", False, True, 64),
        ("k_chunk 208 (16 odd)", 6, 5, 1000, 1, 8, 0, "RN", False, False, 208),
        ("two chunks", 2, 3, KP + 1, 1, 3, 0, "RN", True, False, 0),       # k_hat64 131072
        ("two chunks", 1, 2, 131072, 1, 2, 0, "TRUNC", False, True, 0),
        ("two chunks", 1, 2, 196608, 1, 3, 0, "RN", True, True, 0),        # 131008 + 65600
        ("two chunks", 1, 2, 196624, 1, 3, 0, "TRUNC", True, False, 0),
        ("three chunks", 1, 1, 262017, 1, 2, 0, "TRUNC", True, True, 0),   # k_hat64 = 2 K_c' + 64
    ]
    n = 0
    try:
        for t, (label, m, nn, k, b, N, g, rnd, a_kf, b_kf, kc) in enumerate(cases):
            seed = 5000 + t
            rng = random.Random(seed)
            if b == 1:
                A, B, Cd = matrix(rng, m, nn, k, a_kfast=a_kf, b_kfast=b_kf)
                TA, TB = gen(rng, (m, k), "phi"), gen(rng, (k, nn), "wide")
                TC = gen(rng, (m, nn), "uniform")
            else:
                A = o.Desc([m, k, b], layout(rng, [m, k, b]), [I, K, L])
                B = o.Desc([k, nn, b], layout(rng, [k, nn, b]), [K, J, L])
                Cd = o.Desc([m, nn, b], layout(rng, [m, nn, b]), [I, J, L])
                TA, TB = gen(rng, A.extent, "phi"), gen(rng, B.extent, "wide")
                TC = gen(rng, Cd.extent, "uniform")
            if k > 2:
                inject(rng, TA, 0.01)
            out = {}
            for a in (16, 64):
                if kc:
                    lib.set_limits(k_chunk=kc)
                try:
                    with k_align(lib, a):
                        out[a] = run_case(lib, cuda, f"k_align {label} k={k} {m}x{nn} b={b} N={N} "
                                          f"g={g} {rnd} [A={a}]", A, B, Cd, TA, TB, TC, 1.0, 0.5, N,
                                          g, rnd, rng=random.Random(seed), verbose=verbose,
                                          chunk=kc or KP,
                                          expect_kind="GEMM" if k > 1 else None)
                finally:
                    if kc:
                        lib.set_limits()
                n += 1
            if not np.array_equal(out[16].view(np.uint64), out[64].view(np.uint64)):
                raise Fail(f"k_align {label} k={k}: C differs between A = 16 and A = 64")
        k4_align_check_verdict(lib, "contract")
        # the check fires: one case per A under the mutation (mode 3)
        rng = random.Random(4999)
        A, B, Cd = matrix(rng, 7, 9, 200, a_kfast=True, b_kfast=False)
        TA, TB = gen(rng, (7, 200), "phi"), gen(rng, (200, 9), "wide")
        TC = gen(rng, (7, 9), "uniform")
        for a in (16, 64):
            with k_align(lib, a):
                k4_align_check_mutated(lib, lambda: run_case(
                    lib, cuda, f"k_align mutation [A={a}]", A, B, Cd, TA, TB, TC, 1.0, 0.5, 8, 0,
                    "RN", rng=random.Random(4999), verbose=verbose, expect_kind="GEMM"),
                    f"contract A={a}")
    finally:
        lib.k4_align_check(0)
    STATS["k_align_runs"] = n


def k4_align_check_verdict(lib, what):
    """After a suite under k4_align_check(1): no violation, and the check really ran (review
    2026-10-06 must-fix 3: a call site passing k_align = 0 would disable it silently)."""
    bad, checked = lib.k4_align_check(-1), lib.k4_align_check(2)
    if bad:
        raise Fail(f"{what}: K4 operand pointers not = 0 mod A: {bad} of {checked} launches")
    if checked <= 0:
        raise Fail(f"{what}: the K4 operand-pointer check ran on no launch")


def k4_align_check_mutated(lib, run, what):
    """The check fires: under mode 3 (the check sees the A pointer + 8 bytes, the launch is
    unchanged) every checked launch is a violation. run() is one contraction; release builds
    complete it (and it stays bit-exact), debug builds fail the call at the first violation.
    Leaves the check on (mode 1, counts reset)."""
    lib.k4_align_check(3)
    try:
        run()
        failed = None
    except Fail as e:
        failed = e
    bad, checked = lib.k4_align_check(-1), lib.k4_align_check(2)
    lib.k4_align_check(1)
    if checked <= 0 or bad != checked:
        raise Fail(f"{what}: mutated K4 pointer check: {bad} violations of {checked} launches "
                   f"(want all, > 0); {failed or 'call completed'}")


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
    # v1.16 §3.4.1: the pack kernels' occupancy per emitter word count W (2, 3). The model
    # keeps one row per kernel (the most constrained W); its features use the CTAs per SM, so
    # they must be equal for every W, and the reported row must be the most constrained one.
    names = {1: "kf16", 2: "kf4", 3: "kl8", 4: "kl4", 5: "pack"}   # IMD_PERF_KERNEL_*
    major, dev = ct.c_int(), ct.c_int()
    cuda.check(cuda.rt.cudaGetDevice(ct.byref(dev)), "cudaGetDevice")
    cuda.check(cuda.rt.cudaDeviceGetAttribute(ct.byref(major), 75, dev), "compute capability major")
    cc_major = major.value
    for kern, name in names.items():
        rows = {w: lib.kernel_occupancy_w(kern, w) for w in (0, 2, 3)}
        status, msg = oc.occupancy_verdict(name, rows, cc_major)
        if status == "fail":
            raise Fail(msg)
        if status == "known":
            print(f"known  {msg}")   # always printed: a divergence accepted until issue 23, not a pass of the model's assumption
            continue
        if verbose:
            print(f"  ok  occupancy {name}: {rows[0][0]} CTAs/SM at W = 2, 3; registers "
                  f"{rows[2][1]} / {rows[3][1]}")
    for kern, w in ((0, 2), (5, 1), (5, 4), (5, -1), (14, 2)):   # K0: no W; W 1 / 4 / -1; K5
        if lib.kernel_occupancy_w(kern, w) is not None:
            raise Fail(f"occupancy hook accepted kernel {kern} W {w}")


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
               n=None, k=None, trace=None, _rule=None, _L_w=None, **ext):
        out = real(A, B, C, alpha, beta, N, rounding, g, chunk, m, n, k, trace, _rule, _L_w, **ext)
        if chunk == o.K_C_ALIGNED and _rule is None and _L_w is None and not any(ext.values()):
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


def suite_pad_mutation(lib, cuda, verbose):
    """DESIGN v1.13 §4b.8 / hygiene: the initcheck mutation test for ONE-SIDED padding. The
    test hook imd_test_set_pad_mutation(x) makes the K2 / K3 pack of the post-swap GEMM operand
    x store k-bar only below ceil(k)_16 (Pipeline neg_a64_pad_old_k3), so with A = 64 the bytes
    [ceil(k)_16, k_hat) of x's residue rows are never written while the other operand's are
    zeros. The call stays BIT-EXACT (the zeros cancel the garbage): only initcheck on an
    unpoisoned workspace can see it (tests/initcheck_pad_mutation.sh, ctest
    contract_initcheck_pad). This suite runs the case: with the poisoned workspace (default) for
    no mutation, A and B, each bit-exact vs the oracle, and the workspace shows the mutation
    took effect (x's padding still 0xA5, the other operand's 0); with IMD_TEST_POISON=0 (the
    initcheck driver) only the mutation IMD_TEST_PAD_MUTATION = A / B / none, bit-exact."""
    if not lib.has_test_hooks:
        raise Fail("suite pad_mutation needs the test hooks")
    env = os.environ.get("IMD_TEST_PAD_MUTATION", "none")
    if env not in ("none", "A", "B"):
        raise Fail(f"IMD_TEST_PAD_MUTATION={env!r}: none, A or B")
    rng = random.Random(4242)
    m, n, k, N = 7, 9, 97, 8
    A, B, Cd = matrix(rng, m, n, k, a_kfast=True, b_kfast=True)
    TA, TB, TC = gen(rng, (m, k), "phi"), gen(rng, (k, n), "wide"), gen(rng, (m, n), "uniform")
    lenA, _ = buffer_for(A)
    lenB, _ = buffer_for(B)
    lenC, _ = buffer_for(Cd)
    bufA, baseA = scatter(TA, A, gen(rng, (lenA,), "uniform"))
    bufB, baseB = scatter(TB, B, gen(rng, (lenB,), "uniform"))
    bufC, baseC = scatter(TC, Cd, gen(rng, (lenC,), "uniform"))
    k16, k_hat = -(-k // 16) * 16, -(-k // 64) * 64    # 112, 128
    with k_align(lib, 64):
        want = np.array(o.contract_tensor(o.Tensor(bufA.tolist(), baseA, A),
                                          o.Tensor(bufB.tolist(), baseB, B),
                                          o.Tensor(bufC.tolist(), baseC, Cd), 1.0, 0.5, N, "RN",
                                          N, o.EMULATE), dtype=np.float64)
        tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (A, B, Cd))
        runs = ("none", "A", "B") if POISON else (env,)
        lib.set_fused(0)                   # K1 + K2 / K3 for every group (no KF, no KL)
        try:
            st, plan = lib.plan_create(tA, tB, tC, N, N)
        finally:
            lib.set_fused()
        if st != "SUCCESS":
            raise Fail(f"pad mutation: imd_plan_create {st}")
        try:
            v = lib.plan_sizes(plan)
            if v[1] != 0 or v[7] != k_hat or v[9] != N:
                raise Fail(f"pad mutation: plan swapped {v[1]}, k_hat {v[7]}, g {v[9]}")
            ws_bytes = v[10]
            off = {x: (v[11 + 2 * r], v[12 + 2 * r]) for x, r in ((0, 5), (1, 6))}  # RA, RB
            rows = {0: m, 1: -(-n // 16) * 16}     # rows held: m, n_hat (B's n.. are zero rows)
            real = {0: m, 1: n}
            for which in runs:
                x = {"none": -1, "A": 0, "B": 1}[which]
                dA, dB, dC = (cuda.malloc(b.nbytes) for b in (bufA, bufB, bufC))
                dW = cuda.malloc(ws_bytes)        # fresh; with POISON filled with 0xA5
                try:
                    cuda.upload(dA, bufA)
                    cuda.upload(dB, bufB)
                    cuda.upload(dC, bufC)
                    if POISON:
                        cuda.memset(dW, 0xA5, ws_bytes)
                    lib.set_pad_mutation(x)
                    try:
                        st = lib.contract(plan, 1.0, dA.value + 8 * baseA, dB.value + 8 * baseB,
                                          0.5, dC.value + 8 * baseC, dW.value, ws_bytes)
                    finally:
                        lib.set_pad_mutation(-1)
                    if st != "SUCCESS":
                        raise Fail(f"pad mutation {which}: imd_contract {st}")
                    cuda.sync()
                    got = np.empty_like(bufC)
                    cuda.download(got, dC)
                    if POISON:
                        ws = np.empty(ws_bytes, dtype=np.uint8)
                        cuda.download(ws, dW)
                finally:
                    for p in (dA, dB, dC, dW):
                        cuda.free(p)
                bad = bit_mismatches(got, want)
                if len(bad):
                    raise Fail(f"pad mutation {which}: {len(bad)} C elements differ from the "
                               "oracle (the other operand's zero padding should cancel it)")
                if POISON:
                    for y in (0, 1):
                        o_, nb = off[y]
                        R = ws[o_:o_ + nb].reshape(N, rows[y], k_hat)   # [s][row][k_hat], b = 1
                        pad = R[:, :real[y], k16:k_hat]
                        if (R[:, real[y]:, :] != 0).any():
                            raise Fail(f"pad mutation {which}: operand {y}'s zero rows")
                        if y == x and not (pad == 0xA5).all():
                            raise Fail(f"pad mutation {which}: operand {y}'s padding was written")
                        if y != x and (pad != 0).any():
                            raise Fail(f"pad mutation {which}: operand {y}'s padding not zero")
                if verbose:
                    print(f"  ok  pad mutation {which}: bit-exact"
                          f"{', padding [112, 128) unwritten on that side only' if POISON and x >= 0 else ''}")
        finally:
            lib.plan_destroy(plan)


def suite_eps(lib, cuda, verbose):
    """DESIGN v1.14 §3.4.1, the branch-free pack emitter on eps-heavy data (gen_eps): rows
    of subnormals scaled to eps in [32, 77] at N = 17-20 (both word selects of W = 3) and up
    to 62 at N = 16 (W = 2), eps > 0 at N <= 13 (subnormals; v1.16: eps up to eps_max - 3 at
    N = 5..8, N = 2), dominant entries, RN and TRUNC,
    g < N; small operands (KF E = 4, moduli parts, gridDim.y > 1), K2/K3, KL (forced, and by
    the rule for k_hat > 1024, E = 8 and, with 64-row tiles, E = 4): every run_case variant
    bit-exact vs the oracle. Fails if the data of the emulated cases (k >= 2; k = 1 is DIRECT)
    did not reach the intended eps."""
    rng = random.Random(1414)
    reach = {}
    shapes = ((9, 13, 1, "sub1", "sub1"), (17, 12, 5, "sub", "mix"), (24, 20, 33, "mix", "sub1"),
              (12, 30, 129, "dominant", "mix"), (20, 16, 300, "mix", "dominant"))
    cases = [(N, t, sh) for N in (2, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20)
             for t, sh in enumerate(shapes) if not (t == 4 and N in (2, 5, 7, 14, 15, 18, 19))]
    cases += [(16, 5, (6, 5, 1100, "mix", "sub1")), (20, 6, (5, 7, 1100, "sub1", "mix")),
              (17, 7, (4, 4, 2100, "mix", "sub"))]
    for N, t, (m, n, k, ka, kb) in cases:
        rounding = ("RN", "TRUNC")[(N + t) % 2]
        g = min((0, 1, 3, 0, 5, 0, 4, 0)[t], N)
        A, B, C = matrix(rng, m, n, k)
        TA, TB = gen_eps(rng, (m, k), ka, 1), gen_eps(rng, (k, n), kb, 0)
        if t == 3:
            inject(rng, TB, 0.02)
        # Only emulated cases count toward the reach check: k = 1 takes Path::kDirect (no pack
        # emitter runs), so its subnormal rows must not satisfy the thresholds (review S1).
        e = eps_reached(TA, 1, N, rounding) + eps_reached(TB, 0, N, rounding) if k >= 2 else []
        r = reach.setdefault(N, {"max": 0, "ge32": 0, "ge64": 0, "pos": 0})
        r["max"] = max([r["max"]] + e)
        r["ge32"] += sum(x >= 32 for x in e)
        r["ge64"] += sum(x >= 64 for x in e)
        r["pos"] += sum(x > 0 for x in e)
        run_case(lib, cuda, f"eps[N={N} {m}x{n}x{k} {ka}/{kb} g={g} {rounding}]", A, B, C, TA, TB,
                 gen(rng, (m, n), "uniform"), rng.choice((1.0, -0.75)), 0.0, N, g, rounding,
                 rng=rng, verbose=verbose)
    for N in sorted(reach):
        r = reach[N]
        if verbose:
            print(f"  eps reached N={N}: max {r['max']} (eps_max {params.constants(N)['eps_max']}), "
                  f"{r['pos']} entries eps > 0, {r['ge32']} >= 32, {r['ge64']} >= 64")
        if r["pos"] == 0 or (N >= 13 and r["ge32"] == 0) or (N >= 17 and r["ge64"] == 0) or \
                (N == 16 and r["max"] < 60) or (N >= 17 and r["max"] < params.constants(N)["eps_max"] - 3) or \
                (5 <= N <= 8 and r["max"] < params.constants(N)["eps_max"] - 3):
            raise Fail(f"suite eps: N={N} did not reach the intended eps ({r})")


def adversarial_vec(rng, plan, k, scale_exp=0):
    """Issue 21 (klf64, docs/NOTES-21-klf64.md): a length-k vector whose exact square sum S (§3.2 step 4) is a threshold
    of largest_u or within a few units of one, so that a binary64 estimate of S cannot decide u and the exact passes must
    run. Entries are y 2^(scale_exp - 30) with integers y < 2^31 (the largest, y0 >= 2^30, sets gamma = scale_exp), so S
    = sum y^2 exactly. Many equal small entries make the binary64 sum round many times. Returns None when no threshold
    is reachable."""
    c, R = plan.shift_lhs_mul, plan.shift_rhs
    y0 = rng.randrange(2 ** 30, 2 ** 31)
    S0 = y0 * y0
    cands = []
    for u in range(-80, 80):
        t = R // (c * 4 ** u) if u >= 0 else (R * 4 ** (-u)) // c   # the largest S with c S 4^u <= R
        if S0 < t < S0 + (k - 1) * (y0 * y0 - 1):
            cands.append(t)
    if not cands:
        return None
    r = rng.random()
    off = 0 if r < 0.3 else rng.choice((-1, 1)) if r < 0.7 else rng.choice((-1, 1)) * rng.randint(2, 2 ** rng.randint(2, 30))
    target = rng.choice(cands) + off
    ys = [y0]
    if rng.random() < 0.5:        # a bulk of equal entries (many roundings), the rest greedy
        cnt = rng.randint(k // 4, k - 8)
        v = math.isqrt((target - S0) // (cnt + 1))
        if 0 < v < y0:
            ys += [v] * cnt
    rest = target - sum(y * y for y in ys)
    while rest > 0 and len(ys) < k:
        y = min(math.isqrt(rest), y0)
        ys.append(y)
        rest -= y * y
    if rest != 0:
        return None
    vec = [math.ldexp(float(y), scale_exp - 30) * rng.choice((1, -1)) for y in ys]
    vec += [0.0] * (k - len(vec))
    rng.shuffle(vec)
    return vec


def klf64_operand(rng, plan, rows, k, kind_seed):
    """rows x k, each row one of: adversarial (most), random phi, zero, a lone 5e-324, subnormal-only, normal with
    subnormals, adversarial with an Inf / NaN, all non-finite."""
    X = np.zeros((rows, k))
    for r in range(rows):
        kind = rng.choice(("adv",) * 6 + ("phi", "zero", "min_sub", "sub_only", "mixed_sub", "adv_nonfinite", "nonfinite"))
        if kind in ("adv", "adv_nonfinite"):
            v = None
            for _ in range(50):
                v = adversarial_vec(rng, plan, k, rng.choice((0, 0, rng.randint(-1000, 990))))
                if v is not None:
                    break
            if v is None:
                raise Fail(f"suite klf64: no reachable threshold (N={plan.N}, k={k})")
            X[r] = v
            if kind == "adv_nonfinite":
                X[r, rng.randrange(k)] = rng.choice((math.inf, -math.inf, math.nan))
        elif kind == "phi":
            X[r] = gen(rng, (k,), "phi")
        elif kind == "min_sub":
            X[r, rng.randrange(k)] = rng.choice((5e-324, -5e-324))
        elif kind == "sub_only":
            for h in rng.sample(range(k), rng.randint(1, 40)):
                X[r, h] = math.ldexp(rng.getrandbits(rng.randint(1, 52)) | 1, -1074)
        elif kind == "mixed_sub":
            X[r] = gen(rng, (k,), "tiny")
            X[r, rng.randrange(k)] = math.ldexp(1.0, -1022) * rng.uniform(1, 2)
        elif kind == "nonfinite":
            X[r] = [rng.choice((math.inf, -math.inf, math.nan)) for _ in range(k)]
    return X


def suite_klf64(lib, cuda, verbose):
    """Issue 21 (klf64): operands whose rows' square sums sit on or next to the sigma thresholds (the filter must fall
    back to the exact passes there), mixed in tiles with ordinary, zero, subnormal and non-finite rows (the fallback is
    tile-wide). Bit-exact vs the oracle under any instruction-sequence set; meaningful with IMD_TEST_SEQ = a klf64 set."""
    rng = random.Random(2111)
    for (m, n, k, N, g, rnd, a_kf, b_kf) in (
            (12, 10, 4096, 14, 0, "RN", True, True),      # one chunk per row (the sm_90 sq4096 shape)
            (9, 7, 8192, 14, 0, "RN", True, False),       # two chunks per row; B read side by side
            (8, 6, 1500, 8, 0, "TRUNC", True, True),
            (6, 5, 9000, 20, 4, "RN", False, True),       # more than two chunks (4 values per thread)
            (7, 9, 2100, 12, 3, "TRUNC", False, False)):
        plan = o.Plan(N, k, rnd)
        A, B, C = matrix(rng, m, n, k, a_kfast=a_kf, b_kfast=b_kf)
        TA = klf64_operand(rng, plan, m, k, 0)
        TB = klf64_operand(rng, plan, n, k, 1).T.copy()
        run_case(lib, cuda, f"klf64 thresholds {m}x{n} k={k} N={N} g={g} {rnd}", A, B, C, TA, TB,
                 gen(rng, (m, n), "uniform"), 1.0, 0.5, N, g, rnd, rng=rng, expect_kind="GEMM", verbose=verbose)


SUITES = {"api": suite_api, "replay": suite_replay, "stages": suite_stages, "e2e": suite_e2e, "tensor": suite_tensor,
          "bigk": suite_bigk, "every_n": suite_every_n, "k_align": suite_k_align,
          "pad_mutation": suite_pad_mutation, "eps": suite_eps, "klf64": suite_klf64}


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
          f"{STATS['stage_bytes']} workspace bytes stage-checked; {STATS.get('fused', 0)} runs "
          f"on the fused path KF, {STATS.get('kl', 0)} with KL; K4 configurations run {dict(sorted(STATS['k4'].items()))}) "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
