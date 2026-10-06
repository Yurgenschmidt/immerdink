#!/usr/bin/env python3
"""Bit-exact GPU tests of packed operands (DESIGN v1.7 §6b, ROADMAP D1).

    python3 tests/test_packed.py [--lib build/libimmerdink.so] [--shim ...] [--suite NAME ...]

Every case packs one or both operands on the GPU (imd_pack) and in spec/oracle.py (pack),
compares EVERY byte of the packed buffer (sigma, flags, all N residue planes, zero K
padding, the zeroed tail and alignment gaps; guard zones around it and around imd_pack's
workspace untouched), then runs imd_plan_create_packed + imd_contract_packed and compares C
bit for bit with oracle.contract_tensor(packs=...) (which test_oracle shows equals the
unpacked contraction in the adopted order, and the plain one for imd_pack_desc_for packs).
Each case runs in both vec modes, with the fused paths by rule, off (unfused), without the
small-problem variants, KL forced for the in-line operand and KL with 64-row tiles, in every
K4 configuration and with k4_tune: all bit-identical; with the allocation shim, imd_pack and
imd_contract_packed make no CUDA allocation and launch exactly the kernels the plan says.
Suites:
  packed  random matrices (desc_for and AUTO packs, A / B / both, swapped roles, every N
          range, RN and TRUNC, g, alpha / beta, Inf / NaN) and tensor contractions (batched,
          permuted, the role swap, diagonals, X-only labels, extent-1 modes, DIRECT mode,
          alpha = 0, k = 0, empty outputs, the K4 batch split, native-path K order)
  reuse   one packed buffer read by several plans: other partners, C layouts, roles
  api     every PACK_MISMATCH reason; imd_contract on a packed plan; the operand and handle
          rules of imd_contract_packed; null / ALIASING / MISALIGNED / short-buffer statuses
          of imd_pack and imd_contract_packed
"""

import argparse
import ctypes as C
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
import test_contract as tc  # noqa: E402

Fail = tc.Fail
I, J, K, L, X, Y, Z = tc.I, tc.J, tc.K, tc.L, tc.X, tc.Y, tc.Z
NS = tc.NS
GUARD = tc.GUARD
STATS = {"cases": 0, "runs": 0, "pack_bytes": 0, "roles": set(), "orders_differ": 0,
         "flagged": 0, "kinds": {}, "kl": 0, "kf": 0}


def pack_descs(A, B, Cd, mode, N, rounding, which, order):
    """oracle.PackDesc of the caller's A / B per `which` ("A", "B", "AB"): imd_pack_desc_for's
    (order "desc_for") or AUTO with X's labels classified by membership."""
    out = [None, None]
    for x, (Xd, P) in enumerate(((A, B), (B, A))):
        if "AB"[x] not in which:
            continue
        if order == "desc_for":
            out[x] = o.pack_desc_for(A, B, Cd, mode, x, N, rounding)
        else:
            xl = list(dict.fromkeys(Xd.label))
            out[x] = o.PackDesc(Xd, N, rounding, o.ORDER_AUTO,
                                [lab for lab in xl if lab not in Cd.label],
                                [lab for lab in xl if lab in Cd.label and lab in P.label])
    return out


def packing_arg(descs):
    """An imd_packing_t (two imd_pack_desc_t pointers) and the structs it points to."""
    keep = [hz.make_pack_desc(d) if d is not None else None for d in descs]
    arr = (C.c_void_p * 2)(*[C.cast(C.pointer(d), C.c_void_p) if d is not None else None
                             for d in keep])
    return arr, keep


def device_pack(lib, cuda, od, data_want, pX, name, stream=None):
    """imd_pack_create + imd_pack on the GPU into a guarded buffer; every byte == the
    oracle's; no allocation; 2 launches (0 for an empty pack). Returns (handle, device
    allocation, pack_bytes); the packed bytes start at alloc + GUARD."""
    desc = hz.make_pack_desc(od)
    st, h = lib.pack_create(desc)
    if st != "SUCCESS":
        raise Fail(f"{name}: imd_pack_create {st}")
    pv = o.pack_lower(od)
    pb, pw = lib.pack_bytes(h)
    if pb != pv.pack_bytes or pw != pv.pack_ws or pb != len(data_want):
        raise Fail(f"{name}: imd_pack_bytes {pb}, {pw}; oracle {pv.pack_bytes}, {pv.pack_ws}")
    dP = cuda.malloc(pb + 2 * GUARD)
    dW = cuda.malloc(pw + 2 * GUARD)
    if tc.POISON:   # poison: every byte must be written (else initcheck's fresh memory)
        cuda.memset(dP, 0xA5, pb + 2 * GUARD)
        cuda.memset(dW, 0x5A, pw + 2 * GUARD)
    try:
        allocs, launches = lib.alloc_count(), lib.launch_count()
        st = lib.pack(h, pX, dP.value + GUARD, pb, dW.value + GUARD if pw else None, pw, stream)
        if st != "SUCCESS":
            raise Fail(f"{name}: imd_pack {st}")
        if allocs is not None and lib.alloc_count() != allocs:
            raise Fail(f"{name}: imd_pack allocated")
        if launches is not None and lib.launch_count() - launches != (2 if pb else 0):
            raise Fail(f"{name}: imd_pack launched {lib.launch_count() - launches} kernels")
        cuda.sync()
        got = np.empty(pb + 2 * GUARD, dtype=np.uint8)
        cuda.download(got, dP)
        wsg = np.empty(pw + 2 * GUARD, dtype=np.uint8)
        cuda.download(wsg, dW)
        if tc.POISON and ((got[:GUARD] != 0xA5).any() or (got[GUARD + pb:] != 0xA5).any()):
            raise Fail(f"{name}: imd_pack wrote outside the packed buffer")
        if tc.POISON and ((wsg[:GUARD] != 0x5A).any() or (wsg[GUARD + pw:] != 0x5A).any()):
            raise Fail(f"{name}: imd_pack wrote outside its workspace")
        want = np.frombuffer(bytes(data_want), dtype=np.uint8)
        bad = np.nonzero(got[GUARD:GUARD + pb] != want)[0]
        if len(bad):
            i = int(bad[0])
            regs = {nm: off for nm, (off, _) in pv.regs.items()}
            raise Fail(f"{name}: {len(bad)}/{pb} packed bytes differ; first at {i} (regions "
                       f"{regs}): gpu {got[GUARD + i]} oracle {want[i]}")
    except Exception:
        cuda.free(dP)
        lib.pack_destroy(h)
        raise
    finally:
        cuda.free(dW)
    STATS["pack_bytes"] += pb
    return h, dP, pb


def check_packed_bytes(cuda, dP, pb, data_want, name):
    got = np.empty(pb, dtype=np.uint8)
    cuda.download_raw(got, dP.value + GUARD)
    if not np.array_equal(got, np.frombuffer(bytes(data_want), dtype=np.uint8)):
        raise Fail(f"{name}: the packed buffer changed (contracts only read it)")


def run_packed_case(lib, cuda, name, A, B, Cd, TA, TB, TC, alpha, beta, N, g=0, rounding="RN",
                    mode=o.EMULATE, which="A", order="desc_for", rng=None, verbose=False,
                    full=True):
    """One packed contraction on the GPU vs the oracle (see the module doc). `full`: every
    vec mode, fused-path variant and K4 configuration (else the production rule only)."""
    rng = rng or random.Random(0)
    lens = [tc.buffer_for(d)[0] for d in (A, B, Cd)]
    for n_el in lens:
        if 8 * n_el > tc.MAX_HOST_BUFFER:
            raise Fail(f"{name}: buffer of {8 * n_el} bytes (test guard)")
    bufA, baseA = tc.scatter(TA, A, tc.gen(rng, (lens[0],), "uniform"))
    bufB, baseB = tc.scatter(TB, B, tc.gen(rng, (lens[1],), "uniform"))
    bufC, baseC = tc.scatter(TC, Cd, tc.gen(rng, (lens[2],), "uniform"))
    T = (o.Tensor(bufA.tolist(), baseA, A), o.Tensor(bufB.tolist(), baseB, B),
         o.Tensor(bufC.tolist(), baseC, Cd))
    ods = pack_descs(A, B, Cd, mode, N, rounding, which, order)
    datas = [o.pack(T[x], ods[x]) if ods[x] is not None else None for x in (0, 1)]
    trace = {}
    t0 = time.time()
    want = np.array(o.contract_tensor(*T, alpha, beta, N, rounding, g, mode, trace=trace,
                                      packs=tuple(o.PackedOperand(ods[x], datas[x])
                                                  if ods[x] is not None else None for x in (0, 1))),
                    dtype=np.float64)
    if order == "desc_for":    # imd_pack_desc_for packs reproduce the plain contraction
        plain = np.array(o.contract_tensor(*T, alpha, beta, N, rounding, g, mode), np.float64)
        if len(tc.bit_mismatches(want, plain)):
            raise Fail(f"{name}: oracle packed != plain with imd_pack_desc_for packs")
    t_oracle = time.time() - t0
    view, kind, info = trace["view"], trace["kind"], trace["packed"]
    roles = info["packed_roles"]
    STATS["roles"] |= roles
    STATS["orders_differ"] += order != "desc_for" and any(
        info["orders"][c] != o.lower(A, B, Cd, mode, build_vec=False).order[c] for c in info["orders"])
    STATS["flagged"] += kind == "GEMM" and any(any(bt["fA"]) or any(bt["fB"])
                                               for bt in trace["batches"])
    gg = N if g in (0, None) else g
    pmode = params.DIRECT if mode == o.DIRECT else params.EMULATE
    tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (A, B, Cd))
    dA, dB, dC = cuda.malloc(bufA.nbytes), cuda.malloc(bufB.nbytes), cuda.malloc(bufC.nbytes)
    cuda.upload(dA, bufA)
    cuda.upload(dB, bufB)
    pA, pB = (dA.value or 0) + 8 * baseA, (dB.value or 0) + 8 * baseB
    pC = (dC.value or 0) + 8 * baseC
    handles, dPs, pbs = [None, None], [None, None], [0, 0]
    results = {}
    try:
        for x in (0, 1):
            if ods[x] is not None:
                handles[x], dPs[x], pbs[x] = device_pack(lib, cuda, ods[x], datas[x], (pA, pB)[x],
                                                         f"{name} [pack {'AB'[x]}]")
        emulated = kind == "GEMM"
        fmodes = (-1, 0, 1, 2, 3) if (full and lib.has_test_hooks and emulated) else (-1,)
        if tc.QUICK:
            fmodes = tuple(f for f in fmodes if f in (-1, 0, 3))
        runs = [(vec, pvec, fm, -1) for vec, pvec in ((hz.VEC_GENERATE, params.VEC_GENERATE),
                                                       (hz.VEC_RESIDENT, params.VEC_RESIDENT))
                for fm in fmodes] if full else [(hz.VEC_GENERATE, params.VEC_GENERATE, -1, -1)]
        if tc.QUICK:
            runs = [r for r in runs if r[0] == hz.VEC_GENERATE]
        elif full and lib.has_test_hooks and emulated:
            runs += [(hz.VEC_GENERATE, params.VEC_GENERATE, -1, c)
                     for c in list(range(tc.K4_CONFIGS)) + ["tune"]]
        pk_arg, keep = packing_arg(ods)
        for vec, pvec, fmode, k4 in runs:
            vname = f"{name} [{pvec}, fused {fmode}, K4 {k4}]"
            ws_bytes = C.c_size_t(0)
            st = lib.lib.imd_workspace_size_packed(C.byref(tA), C.byref(tB), C.byref(tC),
                                                   C.byref(hz.Options(N, g, tc.RND[rounding],
                                                                      tc.MODE[mode], vec)),
                                                   pk_arg, C.byref(ws_bytes))
            ws_bytes = ws_bytes.value
            want_ws = 0 if view.empty_output() else params.workspace_size(
                view.m, view.n, view.k, view.b, N, gg, pmode, pvec, packed=roles)
            if hz.STATUS[st] != "SUCCESS" or ws_bytes != want_ws:
                raise Fail(f"{vname}: imd_workspace_size_packed {hz.STATUS[st]} {ws_bytes}, "
                           f"params {want_ws}")
            if lib.has_test_hooks:
                lib.set_fused(fmode if fmode in (0, 2, 3) else -1)
                lib.set_k4_config(k4 if isinstance(k4, int) else -1)
            allocs0 = lib.alloc_count()
            try:
                st, plan = lib.plan_create_packed(tA, tB, tC, N, handles[0], handles[1], g,
                                                  tc.RND[rounding], tc.MODE[mode], vec,
                                                  k4_tune=1 if k4 == "tune" else 0)
            finally:
                if lib.has_test_hooks:
                    lib.set_fused(-1)
                    lib.set_k4_config(-1)
            if st != "SUCCESS":
                raise Fail(f"{vname}: imd_plan_create_packed {st}")
            if allocs0 is not None:
                want_allocs = 1 if (pvec == params.VEC_RESIDENT and not view.empty_output()) else 0
                if k4 == "tune" and tc.k4_tune_bytes(view, N, gg) > 0:
                    want_allocs += 1
                if lib.alloc_count() - allocs0 != want_allocs:
                    lib.plan_destroy(plan)
                    raise Fail(f"{vname}: imd_plan_create_packed made "
                               f"{lib.alloc_count() - allocs0} allocations")
            dW = cuda.malloc(ws_bytes + 2 * GUARD)
            try:
                if lib.has_test_hooks:
                    bits, pbA, pbB = lib.plan_packed(plan)
                    want_bits = (1 if ods[0] is not None else 0) | (2 if ods[1] is not None else 0)
                    by_role = {info["roles"][x]: pbs[x] for x in (0, 1) if ods[x] is not None}
                    if bits != want_bits or (pbA, pbB) != (by_role.get("A", 0), by_role.get("B", 0)):
                        raise Fail(f"{vname}: plan packed slots {bits} {pbA} {pbB}")
                cuda.upload(dC, bufC)
                if tc.POISON:
                    cuda.memset(dW, 0xA5, ws_bytes + 2 * GUARD)
                opA = hz.Operand(pA, dPs[0].value + GUARD if dPs[0] else None, pbs[0], handles[0])
                opB = hz.Operand(pB, dPs[1].value + GUARD if dPs[1] else None, pbs[1], handles[1])
                allocs, launches0 = lib.alloc_count(), lib.launch_count()
                if lib.has_test_hooks:
                    lib.set_k5_mapping(fmode)
                try:
                    st = lib.contract_packed(plan, alpha, opA, opB, beta, pC, dW.value + GUARD,
                                             ws_bytes)
                finally:
                    if lib.has_test_hooks:
                        lib.reset_k5_mapping()
                if st != "SUCCESS":
                    raise Fail(f"{vname}: imd_contract_packed {st}")
                if launches0 is not None and lib.has_test_hooks:
                    fused, kl = lib.plan_kernels(plan)
                    large = lib.plan_fused_large(plan)
                    pbits = (1 if "A" in roles else 0) | (2 if "B" in roles else 0)
                    if large & pbits or (fused and pbits == 3):
                        raise Fail(f"{vname}: KF / KL planned for a packed operand")
                    if fmode in (2, 3) and (fused or large != (3 & ~pbits)):
                        raise Fail(f"{vname}: KL not forced on the in-line operand")
                    if alpha == 0.0 and not view.empty_output():
                        kl = (0 if pvec == params.VEC_RESIDENT else 1) + 1
                    if lib.launch_count() - launches0 != kl:
                        raise Fail(f"{vname}: imd_contract_packed launched "
                                   f"{lib.launch_count() - launches0} kernels, the plan says {kl}")
                    STATS["kf"] += bool(fused)
                    STATS["kl"] += bool(large)
                if allocs is not None and lib.alloc_count() != allocs:
                    raise Fail(f"{vname}: imd_contract_packed allocated")
                cuda.sync()
                got = np.empty_like(bufC)
                cuda.download(got, dC)
                wsg = np.empty(ws_bytes + 2 * GUARD, dtype=np.uint8)
                cuda.download(wsg, dW)
                if tc.POISON and ((wsg[:GUARD] != 0xA5).any() or
                                  (wsg[GUARD + ws_bytes:] != 0xA5).any()):
                    raise Fail(f"{vname}: bytes outside [ws, ws + workspace_size) were written")
            finally:
                cuda.free(dW)
                lib.plan_destroy(plan)
            bad = tc.bit_mismatches(got, want)
            if len(bad):
                i = bad[0]
                raise Fail(f"{vname}: {len(bad)}/{got.size} C elements differ; first at {i}: "
                           f"gpu {got[i]!r} oracle {want[i]!r}")
            results[(pvec, fmode, k4)] = got
            STATS["runs"] += 1
        for x in (0, 1):
            if ods[x] is not None:
                check_packed_bytes(cuda, dPs[x], pbs[x], datas[x], f"{name} [pack {'AB'[x]}]")
        del keep
    finally:
        for p in (dA, dB, dC) + tuple(d for d in dPs if d is not None):
            cuda.free(p)
        for h in handles:
            if h is not None:
                lib.pack_destroy(h)
    first = next(iter(results.values()))
    for key, other in results.items():
        if not np.array_equal(first.view(np.uint64), other.view(np.uint64)):
            raise Fail(f"{name}: C differs between runs ({key})")
    STATS["cases"] += 1
    STATS["kinds"][kind] = STATS["kinds"].get(kind, 0) + 1
    if verbose:
        print(f"  ok  {name}: {kind} m={view.m} n={view.n} k={view.k} b={view.b} N={N} g={gg} "
              f"{rounding} packed {sorted(roles)} ({order}) swap={view.swapped} "
              f"(oracle {t_oracle:.2f}s, {len(results)} runs)", flush=True)
    return first


# ---------------------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------------------

def suite_packed(lib, cuda, verbose):
    rng = random.Random(1701)
    # Random matrices: every packing, both orders, swapped roles (C i-fastest), Inf / NaN.
    for t in range(48):
        N = rng.choice(NS)
        rounding = rng.choice(("RN", "TRUNC"))
        g = rng.choice((0, 1, N, rng.randint(1, N)))
        m, n = rng.randint(1, 24), rng.randint(1, 24)
        k = rng.choice((2, 3, 7, 16, 17, 33, 100, 257, 600, 1030, 1100))
        A, B, Cd = tc.matrix(rng, m, n, k, c_jfast=rng.random() < 0.6, neg=0.2, pad=0.2)
        kind = rng.choice(("uniform", "phi", "wide", "ints"))
        TA, TB, TC = tc.gen(rng, (m, k), kind), tc.gen(rng, (k, n), kind), tc.gen(rng, (m, n), "uniform")
        if t % 3 == 1:
            tc.inject(rng, TA if rng.random() < 0.5 else TB, 0.03)
        if t % 5 == 2:
            TA[rng.randrange(m)] = 0.0
        alpha, beta = rng.choice(((1.0, 0.0), (-2.5, 0.7), (0.3, 1.0), (0.0, -1.25)))
        mode = o.DIRECT if t % 12 == 11 else o.EMULATE
        which = ("A", "B", "AB")[t % 3]
        order = ("desc_for", "auto")[(t // 3) % 2]
        run_packed_case(lib, cuda, f"packed[{t}] {which} {order}", A, B, Cd, TA, TB, TC, alpha,
                        beta, N, g, rounding, mode, which, order, rng, verbose, full=t % 4 != 3)

    def T(ext, lab, order=None, neg=0.0, pad=0.0):
        return o.Desc(ext, tc.layout(rng, ext, order=order, neg=neg, pad=pad), lab)

    def run(name, A, B, Cd, which="AB", order="desc_for", N=9, rounding="RN", alpha=1.0,
            beta=0.0, kind="phi", nonfinite=False, g=0, mode=o.EMULATE):
        TA, TB = tc.gen(rng, A.extent, kind), tc.gen(rng, B.extent, kind)
        if nonfinite:
            tc.inject(rng, TA, 0.02)
            tc.inject(rng, TB, 0.02)
        for w in ([which] if isinstance(which, str) else which):
            for od in ([order] if isinstance(order, str) else order):
                run_packed_case(lib, cuda, f"{name} [{w} {od}]", A, B, Cd, TA, TB,
                                tc.gen(rng, Cd.extent, "uniform"), alpha, beta, N, g, rounding,
                                mode, w, od, rng, verbose)

    m, n, k, b = 5, 7, 19, 3
    both = ("A", "B", "AB")
    run("ikl,kjl->ijl", T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]), T([m, n, b], [I, J, L]),
        which=both, order=("desc_for", "auto"), N=8, nonfinite=True)
    run("ikl,kjl->ijl (B k-fastest, l outer in C)", T([m, k, b], [I, K, L]),
        T([k, n, b], [K, J, L], order=[0, 2, 1]), T([m, n, b], [I, J, L], order=[1, 0, 2]),
        which=both, order="auto", N=16, rounding="TRUNC", alpha=-1.5, beta=0.25)
    run("kai,bkj->jbai (swap)", T([k, 3, m], [K, X, I]), T([2, k, n], [Y, K, J]),
        T([n, 2, 3, m], [J, Y, X, I]), which=both, order=("desc_for", "auto"), N=14)
    run("acQ,bdQ->acbd (DF)", T([3, 4, 23], [I, X, K]), T([2, 5, 23], [J, Y, K]),
        T([3, 4, 2, 5], [I, X, J, Y]), which=both, order="auto", N=20, beta=1.0)
    run("ikl,kjl->ijl (neg/pad strides)", T([m, k, b], [I, K, L], neg=0.5, pad=0.5),
        T([k, n, b], [K, J, L], neg=0.5, pad=0.5), T([m, n, b], [I, J, L], neg=0.5, pad=0.5),
        which=both, order=("desc_for", "auto"), N=9, beta=0.5)
    # diagonal + an A-only label: A's pack carries it in K (B has it with stride 0)
    run("iik x,kj->ij (diagonal + A-only)", T([m, m, 4, 3], [I, I, K, X]), T([4, n], [K, J]),
        T([m, n], [I, J]), which="A", order=("desc_for", "auto"), N=3)
    run("ik1j (extent-1 modes)", T([m, 1, k], [I, Z, K]), T([k, 1, n], [K, Y, J]),
        T([m, 1, n], [I, Y, J]), which=both, order=("desc_for", "auto"), N=2, rounding="TRUNC")
    # DIRECT mode, alpha = 0, k = 0 (a 0-byte pack) and an empty output ignore the pack
    run("ikl,kjl->ijl DIRECT mode", T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]),
        T([m, n, b], [I, J, L]), which="AB", alpha=0.5, beta=1.0, nonfinite=True, mode=o.DIRECT)
    run("alpha = 0", T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]), T([m, n, b], [I, J, L]),
        which="AB", alpha=0.0, beta=-2.0)
    run("k=0", T([m, 0], [I, K]), T([0, n], [K, J]), T([m, n], [I, J]), which="AB", beta=-2.0)
    run("empty output", T([0, k], [I, K]), T([k, n], [K, J]), T([0, n], [I, J]), which="AB")
    # rows longer than one KL chunk at the production rule (k_hat > 1024: KL in-line)
    run("long k (KL rule)", T([3, 2100], [I, K]), T([2100, 2], [K, J]), T([3, 2], [I, J]),
        which=both, order="auto", N=9, nonfinite=True)
    # m = n = 1, b = 3300, g = N = 20: 66000 batches (the K4 batch split) over pack planes
    run("kl,kl->l (b=3300)", T([3, 3300], [K, L]), T([3, 3300], [K, L]), T([3300], [L]),
        which="AB", order="auto", N=20, kind="uniform")
    # every residue-group size g = 1..N (a group is the slice of planes g0.. of the pack)
    A, B, Cd = T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]), T([m, n, b], [I, J, L])
    TA, TB, TC = tc.gen(rng, A.extent, "phi"), tc.gen(rng, B.extent, "phi"), tc.gen(rng, Cd.extent, "uniform")
    tc.inject(rng, TB, 0.02)
    for gg in range(1, 8):
        for w in ("A", "B", "AB"):
            run_packed_case(lib, cuda, f"g sweep g={gg} [{w}]", A, B, Cd, TA, TB, TC, 1.0, 0.5, 7,
                            gg, "TRUNC" if gg % 2 else "RN", o.EMULATE, w, "auto", rng, verbose,
                            full=False)
    # §3.7 native order through a pack (oracle A7): AUTO adopts X's K order (k1 fastest:
    # NaN), imd_pack_desc_for the plan's (k2 fastest: -Inf), as the plain call.
    MAX = 1.7976931348623157e308
    Xa = o.Desc([2, 2], [1, 2], [X, Y])
    Ya = o.Desc([2, 2, 4], [8, 1, 24], [X, Y, J])
    Ca = o.Desc([4], [1], [J])
    TX = np.array([[0.0, -math.inf], [MAX, MAX]])            # [x, y]: (0,1) = -Inf
    TY = np.zeros((2, 2, 4))
    TY[0, :, :], TY[1, :, :] = 1.0, 2.0
    out = {}
    for od in ("auto", "desc_for"):
        out[od] = run_packed_case(lib, cuda, f"native K order via a pack ({od})", Xa, Ya, Ca, TX,
                                  TY, np.zeros(4), 1.0, 0.0, 2, 0, "RN", o.EMULATE, "A", od, rng,
                                  verbose)
    if not (math.isnan(out["auto"][0]) and out["desc_for"][0] == -math.inf):
        raise Fail(f"native K order through a pack: {out}")
    if STATS["roles"] != {"A", "B"} or STATS["flagged"] < 5 or STATS["orders_differ"] < 3:
        raise Fail(f"coverage: {STATS}")


def suite_k_align(lib, cuda, verbose):
    """DESIGN v1.13 §4b.8 (stage 2): packs of format 1 and 2 (A forced to 16 / 64 through
    libimmerdink_host's hook) byte-exact vs oracle.pack(fmt) on a 0xA5-poisoned pack buffer,
    and packed A, packed B (the post-swap B role: the tail rows, b > 1, an n_hat split so
    j0 > 0) and both packed, bit-exact vs the oracle on every producer path and identical
    across A; PACK_MISMATCH when the hook changes between imd_pack_create and plan creation
    (both directions), and in imd_contract_packed for a handle of the other alignment; the K4
    operand-pointer check (= 0 mod A) with no violation."""
    if not lib.has_test_hooks:
        raise Fail("suite k_align needs the test hooks")
    rng = random.Random(1713)

    def T(ext, lab, order=None):
        return o.Desc(ext, tc.layout(rng, ext, order=order), lab)

    lib.k4_align_check(1)
    n_runs = 0
    try:
        shapes = [  # (label, A, B, C, N, g, rounding, which, n-tile limit)
            ("matrix k=97", T([5, 97], [I, K]), T([97, 7], [K, J]), T([5, 7], [I, J]), 9, 0,
             "RN", ("A", "B", "AB"), 0),
            ("matrix k=1000", T([4, 1000], [I, K]), T([1000, 6], [K, J]), T([4, 6], [I, J]), 8, 3,
             "TRUNC", ("A", "B", "AB"), 0),
            # C i-fastest: the role swap, the caller's A in the post-swap B role; n = 300 rows of
            # B role: n_hat 304 reads the tail; b = 2; 256-column n-hat parts: j0 = 256
            ("B role tail k=97 b=2", T([300, 97, 2], [I, K, L]), T([97, 3, 2], [K, J, L]),
             T([300, 3, 2], [I, J, L], order=[0, 1, 2]), 9, 0, "RN", ("A", "AB"), 1),
            ("B role tail k=1000 b=2", T([3, 1000, 2], [I, K, L]), T([1000, 300, 2], [K, J, L]),
             T([3, 300, 2], [I, J, L], order=[1, 0, 2]), 8, 0, "TRUNC", ("B", "AB"), 1),
            ("k=96 (format 1 either way)", T([6, 96], [I, K]), T([96, 5], [K, J]), T([6, 5], [I, J]),
             9, 0, "RN", ("AB",), 0),
        ]
        for label, A, B, Cd, N, g, rnd, whiches, ntiles in shapes:
            TA, TB = tc.gen(rng, A.extent, "phi"), tc.gen(rng, B.extent, "wide")
            tc.inject(rng, TA, 0.01)
            TC = tc.gen(rng, Cd.extent, "uniform")
            for which in whiches:
                out = {}
                for a in (16, 64):
                    if ntiles:
                        lib.set_limits(max_n_tiles=ntiles)
                    try:
                        with tc.k_align(lib, a):
                            for x, Xd in enumerate((A, B)):
                                if "AB"[x] in which:
                                    fmt = o.pack_lower(pack_descs(A, B, Cd, o.EMULATE, N, rnd,
                                                                  which, "auto")[x]).format
                                    if fmt != (2 if a == 64 else 1):
                                        raise Fail(f"{label}: pack format {fmt} at A = {a}")
                            out[a] = run_packed_case(lib, cuda, f"k_align {label} [{which}, A={a}]",
                                                     A, B, Cd, TA, TB, TC, 1.0, 0.5, N, g, rnd,
                                                     o.EMULATE, which, "auto", random.Random(7),
                                                     verbose)
                    finally:
                        if ntiles:
                            lib.set_limits()
                    n_runs += 1
                if not np.array_equal(out[16].view(np.uint64), out[64].view(np.uint64)):
                    raise Fail(f"{label} [{which}]: C differs between A = 16 and A = 64")
        bad = lib.k4_align_check(-1)
        if bad:
            raise Fail(f"K4 operand pointers not = 0 mod A: {bad} launches")
    finally:
        lib.k4_align_check(0)
    # FORMAT: a pack and a plan of different A (only through the hook)
    A, B, Cd = T([5, 200], [I, K]), T([200, 7], [K, J]), T([5, 7], [I, J])
    tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (A, B, Cd))
    n_mis = 0
    for pack_a, plan_a in ((16, 64), (64, 16), (16, 0), (0, 16), (64, 64), (0, 0)):
        ods = pack_descs(A, B, Cd, o.EMULATE, 9, "RN", "AB", "auto")
        with tc.k_align(lib, pack_a):
            hs = [lib.pack_create(hz.make_pack_desc(od)) for od in ods]
        if any(st != "SUCCESS" for st, _ in hs):
            raise Fail(f"imd_pack_create {hs}")
        try:
            want = "SUCCESS" if (pack_a or 64) == (plan_a or 64) else "PACK_MISMATCH"
            with tc.k_align(lib, plan_a):
                for pa, pb in ((hs[0][1], None), (None, hs[1][1]), (hs[0][1], hs[1][1])):
                    st, plan = lib.plan_create_packed(tA, tB, tC, 9, pa, pb)
                    if st == "SUCCESS":
                        lib.plan_destroy(plan)
                    if st != want:
                        raise Fail(f"pack A={pack_a}, plan A={plan_a}: {st}, want {want}")
                    n_mis += st == "PACK_MISMATCH"
            # a plan made against handles of one alignment, called with a handle of the other
            if want == "PACK_MISMATCH":
                with tc.k_align(lib, plan_a):
                    hp = lib.pack_create(hz.make_pack_desc(ods[0]))
                    st, plan = lib.plan_create_packed(tA, tB, tC, 9, hp[1], None)
                try:
                    if st != "SUCCESS":
                        raise Fail(f"plan of the handle's own alignment: {st}")
                    pb_ = lib.pack_bytes(hs[0][1])[0]
                    opA = hz.Operand(1 << 20, 1 << 21, pb_, hs[0][1])   # never dereferenced
                    opB = hz.Operand(1 << 22, None, 0, None)
                    st = lib.contract_packed(plan, 1.0, opA, opB, 0.0, 1 << 23, None, 0)
                    if st != "PACK_MISMATCH":
                        raise Fail(f"imd_contract_packed with a handle of A={pack_a}: {st}")
                    n_mis += 1
                finally:
                    lib.plan_destroy(plan)
                    lib.pack_destroy(hp[1])
        finally:
            for _, h in hs:
                lib.pack_destroy(h)
    if n_mis < 12:
        raise Fail(f"too few FORMAT mismatches: {n_mis}")
    STATS["k_align"] = (n_runs, n_mis)


def suite_reuse(lib, cuda, verbose):
    """One packed buffer of X[i, k, l] read by several plans (different partners, C layouts,
    batch / row splits, the post-swap A and B roles, g and vec_mode), each bit-exact vs the
    oracle; the buffer is never written by a contraction."""
    rng = random.Random(1702)
    Xd = o.Desc([5, 7, 3], [21, 3, 1], [I, K, L])
    TX = tc.gen(rng, (5, 7, 3), "phi")
    TX[1, 2, 0] = math.inf
    N, rnd = 9, "RN"
    od = o.PackDesc(Xd, N, rnd, o.ORDER_AUTO, [K], [L])
    lenX, baseX = tc.buffer_for(Xd)
    bufX, baseX = tc.scatter(TX, Xd, tc.gen(rng, (lenX,), "uniform"))
    oX = o.Tensor(bufX.tolist(), baseX, Xd)
    data = o.pack(oX, od)
    dX = cuda.malloc(bufX.nbytes)
    cuda.upload(dX, bufX)
    pX = dX.value + 8 * baseX
    h, dP, pb = device_pack(lib, cuda, od, data, pX, "reuse pack")
    partners = [(o.Desc([7, 4, 3], [12, 3, 1], [K, J, L]), o.Desc([5, 4, 3], [12, 3, 1], [I, J, L])),
                (o.Desc([7, 4, 3], [1, 21, 7], [K, J, L]), o.Desc([5, 4, 3], [1, 15, 5], [I, J, L])),
                (o.Desc([3, 7, 9], [63, 9, 1], [L, K, Y]), o.Desc([3, 9, 5], [45, 5, 1], [L, Y, I])),
                (o.Desc([2, 7, 3], [21, 3, 1], [J, K, L]), o.Desc([3, 5, 2], [1, 3, 15], [L, I, J]))]
    roles, n = set(), 0
    try:
        for t, (Bd, Cd) in enumerate(partners):
            TB = tc.gen(rng, Bd.extent, "wide")
            lenB, _ = tc.buffer_for(Bd)
            bufB, baseB = tc.scatter(TB, Bd, tc.gen(rng, (lenB,), "uniform"))
            lenC, _ = tc.buffer_for(Cd)
            bufC, baseC = tc.scatter(tc.gen(rng, Cd.extent, "uniform"), Cd,
                                     tc.gen(rng, (lenC,), "uniform"))
            oB, oC = o.Tensor(bufB.tolist(), baseB, Bd), o.Tensor(bufC.tolist(), baseC, Cd)
            dB, dC = cuda.malloc(bufB.nbytes), cuda.malloc(bufC.nbytes)
            cuda.upload(dB, bufB)
            pBd, pC = dB.value + 8 * baseB, dC.value + 8 * baseC
            try:
                for x_is_a in (True, False):     # X as the caller's A, then as the caller's B
                    for g, vec in ((3, hz.VEC_GENERATE), (9, hz.VEC_RESIDENT), (1, hz.VEC_GENERATE)):
                        tr = {}
                        pk = o.PackedOperand(od, data)
                        args = (oX, oB, oC) if x_is_a else (oB, oX, oC)
                        want = np.array(o.contract_tensor(*args, 1.0, 0.5, N, rnd, g, trace=tr,
                                                          packs=(pk, None) if x_is_a else (None, pk)))
                        roles.add(tr["packed"]["roles"][0 if x_is_a else 1])
                        dA_, dB_ = (Xd, Bd) if x_is_a else (Bd, Xd)
                        tA, tB = (hz.make_tensor(d.extent, d.stride, d.label) for d in (dA_, dB_))
                        tC = hz.make_tensor(Cd.extent, Cd.stride, Cd.label)
                        st, plan = lib.plan_create_packed(tA, tB, tC, N, h if x_is_a else None,
                                                          None if x_is_a else h, g, hz.RN,
                                                          hz.EMULATE, vec)
                        if st != "SUCCESS":
                            raise Fail(f"reuse[{t}]: imd_plan_create_packed {st}")
                        pk_arg, keep = packing_arg((od, None) if x_is_a else (None, od))
                        wsb = C.c_size_t(0)
                        assert lib.lib.imd_workspace_size_packed(
                            C.byref(tA), C.byref(tB), C.byref(tC),
                            C.byref(hz.Options(N, g, hz.RN, hz.EMULATE, vec)), pk_arg,
                            C.byref(wsb)) == 0
                        dW = cuda.malloc(wsb.value)
                        try:
                            cuda.upload(dC, bufC)
                            opX = hz.Operand(pX, dP.value + GUARD, pb, h)
                            opB = hz.Operand(pBd, None, 0, None)
                            st = lib.contract_packed(plan, 1.0, opX if x_is_a else opB,
                                                     opB if x_is_a else opX, 0.5, pC, dW, wsb.value)
                            if st != "SUCCESS":
                                raise Fail(f"reuse[{t}]: imd_contract_packed {st}")
                            cuda.sync()
                            got = np.empty_like(bufC)
                            cuda.download(got, dC)
                        finally:
                            cuda.free(dW)
                            lib.plan_destroy(plan)
                        if len(tc.bit_mismatches(got, want)):
                            raise Fail(f"reuse[{t}] {'A' if x_is_a else 'B'} g={g}: C differs")
                        n += 1
            finally:
                cuda.free(dB)
                cuda.free(dC)
        check_packed_bytes(cuda, dP, pb, data, "reuse")
    finally:
        cuda.free(dX)
        cuda.free(dP)
        lib.pack_destroy(h)
    if roles != {"A", "B"}:
        raise Fail(f"reuse: roles {roles}")
    if verbose:
        print(f"  ok  one packed buffer, {n} contractions ({len(partners)} partners / C layouts, "
              f"both roles, g and vec_mode), bit-exact; the buffer unchanged")


def suite_api(lib, cuda, verbose):
    rng = random.Random(1703)
    n = 0

    def status_of_plan(Ad, Bd, Cd, pa, pb, N=9, rnd=hz.RN):
        tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (Ad, Bd, Cd))
        st, plan = lib.plan_create_packed(tA, tB, tC, N, pa, pb, 0, rnd)
        if plan is not None:
            lib.plan_destroy(plan)
        return st

    def handle(od):
        st, h = lib.pack_create(hz.make_pack_desc(od))
        if st != "SUCCESS":
            raise Fail(f"pack_create {st}")
        return h

    Xd = o.Desc([5, 7, 3], [21, 3, 1], [I, K, L])
    Bd, Cd = o.Desc([7, 4, 3], [12, 3, 1], [K, J, L]), o.Desc([5, 4, 3], [12, 3, 1], [I, J, L])
    hand = []
    hX = handle(o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [K], [L]))
    hand.append(hX)
    try:
        # imd_pack_create statuses (imd_pack_layout's)
        for od, want in ((o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [K, 99], [L]), "INVALID_VALUE"),
                         (o.PackDesc(Xd, 21, "RN", o.ORDER_AUTO, [K], [L]), "INVALID_VALUE"),
                         (o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [], [L]), "NOT_SUPPORTED"),
                         (o.PackDesc(o.Desc([32042], [1], [K]), 2, "RN", o.ORDER_AUTO, [K]),
                          "NOT_SUPPORTED")):
            st, h = lib.pack_create(hz.make_pack_desc(od))
            if h is not None:
                lib.pack_destroy(h)
            if st != want:
                raise Fail(f"imd_pack_create {st}, expected {want}")
            n += 1
        # every PACK_MISMATCH reason at plan creation (status), and imd_lowering_packed's
        # reason for the same descriptors (libimmerdink_host: the same check)
        X2 = o.Desc([7, 2, 5], [10, 5, 1], [K, Y, I])     # K = {k, y}
        Y2 = o.Desc([7, 2, 4], [8, 4, 1], [K, Y, J])
        C2 = o.Desc([5, 4], [4, 1], [I, J])
        odX = o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [K], [L])
        cases = [
            ("ok", Xd, Bd, Cd, odX, None, 9, hz.RN, "NONE"),
            ("MODULI", Xd, Bd, Cd, odX, None, 8, hz.RN, "MODULI"),
            ("ROUNDING", Xd, Bd, Cd, odX, None, 9, hz.TRUNC, "ROUNDING"),
            ("CLASSES: l a row in the pack", Xd, Bd, Cd,
             o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [K], []), None, 9, hz.RN, "CLASSES"),
            ("CLASSES: partner-only K", Xd, o.Desc([7, 4, 3, 2], [24, 6, 2, 1], [K, J, L, Z]), Cd,
             odX, None, 9, hz.RN, "CLASSES"),
            ("CLASSES: another rank", o.Desc([5, 7, 3, 1], [21, 3, 1, 1], [I, K, L, Z]), Bd, Cd,
             odX, None, 9, hz.RN, "CLASSES"),
            ("STRIDES", o.Desc([5, 7, 3], [1, 5, 35], [I, K, L]), Bd, Cd, odX, None, 9, hz.RN,
             "STRIDES"),
            ("EXTENTS", o.Desc([6, 7, 3], [21, 3, 1], [I, K, L]), Bd,
             o.Desc([6, 4, 3], [12, 3, 1], [I, J, L]), odX, None, 9, hz.RN, "EXTENTS"),
            ("ORDER", X2, Y2, C2, o.PackDesc(X2, 9, "RN", o.ORDER_GIVEN, [K, Y], [], [I]),
             o.PackDesc(Y2, 9, "RN", o.ORDER_GIVEN, [Y, K], [], [J]), 9, hz.RN, "ORDER"),
            ("both, same order", X2, Y2, C2,
             o.PackDesc(X2, 9, "RN", o.ORDER_GIVEN, [K, Y], [], [I]),
             o.PackDesc(Y2, 9, "RN", o.ORDER_GIVEN, [K, Y], [], [J]), 9, hz.RN, "NONE"),
        ]
        reasons = ["NONE", "FORMAT", "MODULI", "ROUNDING", "CLASSES", "EXTENTS", "STRIDES", "ORDER"]
        f = lib.lib.imd_lowering_packed
        f.argtypes = [C.POINTER(hz.Tensor)] * 3 + [C.POINTER(hz.Options), C.c_void_p, C.c_void_p]
        f.restype = C.c_int
        for what, Ad, Bdd, Cdd, oa, ob, N, rnd, want in cases:
            pa = handle(oa) if oa is not None else None
            pb = handle(ob) if ob is not None else None
            hand += [h for h in (pa, pb) if h is not None]
            st = status_of_plan(Ad, Bdd, Cdd, pa, pb, N, rnd)
            if st != ("SUCCESS" if want == "NONE" else "PACK_MISMATCH"):
                raise Fail(f"plan {what}: {st}, expected {want}")
            raw = (C.c_uint8 * 184)()
            C.c_uint32.from_buffer(raw, 0).value = 184
            pk_arg, keep = packing_arg((oa, ob))
            tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (Ad, Bdd, Cdd))
            lst = f(C.byref(tA), C.byref(tB), C.byref(tC), C.byref(hz.Options(N, 0, rnd)),
                    pk_arg, raw)
            got = reasons[C.c_int32.from_buffer(raw, 180).value]
            if hz.STATUS[lst] != st or got != want:
                raise Fail(f"imd_lowering_packed {what}: {hz.STATUS[lst]} {got}, expected {want}")
            del keep
            n += 1
        # contract-time rules on one packed plan
        tA, tB, tC = (hz.make_tensor(d.extent, d.stride, d.label) for d in (Xd, Bd, Cd))
        st, plan = lib.plan_create_packed(tA, tB, tC, 9, hX, None)
        assert st == "SUCCESS"
        pb_, pw = lib.pack_bytes(hX)
        wsb = C.c_size_t(0)
        pk_arg, keep = packing_arg((o.PackDesc(Xd, 9, "RN", o.ORDER_AUTO, [K], [L]), None))
        assert lib.lib.imd_workspace_size_packed(C.byref(tA), C.byref(tB), C.byref(tC),
                                                 C.byref(hz.Options(9, 0, hz.RN, hz.EMULATE)),
                                                 pk_arg, C.byref(wsb)) == 0
        ws_bytes = wsb.value
        big = cuda.malloc(1 << 20)   # one arena: X, B, C, packed, ws at chosen offsets
        try:
            base = big.value
            cuda.memset(big, 0, 1 << 20)
            pX, pB, pC, pP, pW = base, base + 4096, base + 8192, base + 16384, base + 65536
            cuda.upload_raw(pX, tc.gen(rng, (105,), "uniform"))
            assert lib.pack(hX, pX, pP, pb_, pW, pw) == "SUCCESS"
            hX2 = handle(o.PackDesc(o.Desc([5, 7, 3], [21, 3, 1], [11, 12, 13]), 9, "RN",
                                    o.ORDER_AUTO, [12], [13]))       # same key, other labels
            hand.append(hX2)
            hO = handle(o.PackDesc(Xd, 9, "RN", o.ORDER_GIVEN, [K], [L], [I]))  # same key too
            hand.append(hO)
            hB = handle(o.PackDesc(Bd, 9, "RN", o.ORDER_AUTO, [K], [L]))       # another key
            hand.append(hB)
            OP = hz.Operand

            def call(opA, opB, alpha=1.0, Cp=pC, ws=pW, wsz=ws_bytes):
                return lib.contract_packed(plan, alpha, opA, opB, 0.0, Cp, ws, wsz)

            good = OP(pX, pP, pb_, hX)
            plainB = OP(pB, None, 0, None)
            checks = [
                ("ok", (good, plainB), {}, "SUCCESS"),
                ("same key, renamed labels", (OP(pX, pP, pb_, hX2), plainB), {}, "SUCCESS"),
                ("same key, GIVEN", (OP(pX, pP, pb_, hO), plainB), {}, "SUCCESS"),
                ("NULL operand", (None, plainB), {}, "INVALID_VALUE"),
                ("in-line slot with packed", (good, OP(pB, pP, pb_, None)), {}, "PACK_MISMATCH"),
                ("in-line slot with a handle", (good, OP(pB, None, 0, hB)), {}, "PACK_MISMATCH"),
                ("packed slot without a handle", (OP(pX, pP, pb_, None), plainB), {}, "PACK_MISMATCH"),
                ("another key", (OP(pX, pP, pb_, hB), plainB), {}, "PACK_MISMATCH"),
                ("packed NULL, read", (OP(pX, None, 0, hX), plainB), {}, "INVALID_VALUE"),
                ("packed NULL, alpha = 0", (OP(pX, None, 0, hX), plainB), {"alpha": 0.0}, "SUCCESS"),
                ("C NULL", (good, plainB), {"Cp": None}, "INVALID_VALUE"),
                ("C overlaps packed", (good, plainB), {"Cp": pP + 64}, "ALIASING"),
                ("ws overlaps packed", (good, plainB), {"ws": pP}, "ALIASING"),
                ("packed overlaps A (only read)", (OP(pX, pX, pb_, hX), plainB), {}, "SUCCESS"),
                ("packed misaligned", (OP(pX, pP + 16, pb_, hX), plainB), {}, "MISALIGNED"),
                ("packed short", (OP(pX, pP, pb_ - 1, hX), plainB), {}, "INSUFFICIENT_WORKSPACE"),
                ("ws short", (good, plainB), {"wsz": ws_bytes - 1}, "INSUFFICIENT_WORKSPACE"),
            ]
            for what, (a, b_), kw, want in checks:
                st = call(a, b_, **kw)
                if st != want:
                    raise Fail(f"imd_contract_packed {what}: {st}, expected {want}")
                cuda.sync()
                n += 1
            # imd_contract on a plan with a packed slot
            if lib.contract(plan, 1.0, pX, pB, 0.0, pC, pW, ws_bytes) != "PACK_MISMATCH":
                raise Fail("imd_contract on a packed plan")
            # imd_pack statuses
            pchecks = [
                ("ok", (pX, pP, pb_, pW, pw), "SUCCESS"),
                ("X NULL", (None, pP, pb_, pW, pw), "INVALID_VALUE"),
                ("packed NULL", (pX, None, pb_, pW, pw), "INVALID_VALUE"),
                ("ws NULL", (pX, pP, pb_, None, pw), "INVALID_VALUE"),
                ("packed overlaps X", (pX, pX, pb_, pW, pw), "ALIASING"),
                ("ws overlaps X", (pX, pP, pb_, pX, pw), "ALIASING"),
                ("ws overlaps packed", (pX, pP, pb_, pP + 256, pw), "ALIASING"),
                ("X misaligned", (pX + 4, pP, pb_, pW, pw), "MISALIGNED"),
                ("packed misaligned", (pX, pP + 16, pb_, pW, pw), "MISALIGNED"),
                ("ws misaligned", (pX, pP, pb_, pW + 16, pw), "MISALIGNED"),
                ("packed short", (pX, pP, pb_ - 1, pW, pw), "INSUFFICIENT_WORKSPACE"),
                ("ws short", (pX, pP, pb_, pW, pw - 1), "INSUFFICIENT_WORKSPACE"),
            ]
            for what, args, want in pchecks:
                st = lib.pack(hX, *args)
                if st != want:
                    raise Fail(f"imd_pack {what}: {st}, expected {want}")
                cuda.sync()
                n += 1
            # an empty pack: SUCCESS with every pointer NULL
            hE = handle(o.PackDesc(o.Desc([0, 7], [7, 1], [I, K]), 9, "RN", o.ORDER_AUTO, [K]))
            hand.append(hE)
            if lib.pack_bytes(hE) != (0, 0) or lib.pack(hE, None, None, 0, None, 0) != "SUCCESS":
                raise Fail("empty pack")
            n += 1
            del keep
        finally:
            cuda.free(big)
            lib.plan_destroy(plan)
    finally:
        for h in hand:
            lib.pack_destroy(h)
    if verbose:
        print(f"  ok  {n} status checks (pack creation, every PACK_MISMATCH reason, the "
              f"imd_contract_packed and imd_pack check orders)")


SUITES = {"packed": suite_packed, "reuse": suite_reuse, "api": suite_api,
          "k_align": suite_k_align}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=hz.default_lib())
    ap.add_argument("--shim", help="tests/alloc_shim.c, already in LD_PRELOAD")
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
    print(f"{STATS['cases']} packed contractions ({STATS['runs']} runs) bit-exact vs "
          f"spec/oracle.py; {STATS['pack_bytes']} packed bytes compared; roles "
          f"{sorted(STATS['roles'])}; dispatch {STATS['kinds']}; {STATS['kf']} runs with KF, "
          f"{STATS['kl']} with KL (in-line operand) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
