#!/usr/bin/env python3
"""libimmerdink_host's footprints across N and g (DESIGN v1.19 §6g / §6a; NOTES-C1 §4.1).

    python3 tests/test_footprint.py --host-lib build/libimmerdink_host.so [--random 10000] [-v]

Precision phases size one workspace and one pack buffer for every phase. Checked on the
library, every value equal to spec/params.py (cert_footprint_monotone certifies the formulas):
  - F1 (ABI from API 13 on): imd_workspace_size and imd_workspace_size_packed are
    nondecreasing in N at every fixed g >= 1, in g in [1, N], and in N with g = 0 (= N);
    C-ws: the (N_max, g_max) workspace covers every (N <= N_max, g <= g_max);
  - F2 (ABI): imd_pack_layout's pack_bytes is nondecreasing in N; C-pack;
  - I1 (internal): the sig and flag offsets and bytes, the res offset and pack_ws_bytes do
    not depend on N; res(N') is a prefix of res(N_max);
  - I2 (internal): imd_plan_bytes' device bytes (and imd_lowering_packed's plan device bytes)
    do not depend on N or g; RN acceptance (NOT_SUPPORTED) is exactly k > rn_feasible_kmax(N),
    an up-set in N;
  - I3 (internal): packing more roles never increases the workspace at any (N, g).
Grid: m, n in {1, 15, 16, 17, 255, 256, 257}, k in {2, 96, 97, 1024, 1025, 131008, 131009},
b in {1, 3} ("lik,lkj->lij"), N = 2..20, g = 0..N, both vec modes, the caller's packed roles
none / A / B / AB, the K alignment AUTO and A16 (libimmerdink_host's test hook); then random
legal contractions (tests/test_host.py random_descs: permuted and negative strides, the role
swap, diagonals) at every N with sampled g, a random vec mode, packed set and alignment
(16, 64 or AUTO). CPU only; no GPU, no CUDA.
"""

import argparse
import ctypes as C
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "spec"))
from immerdink import host  # noqa: E402
import params  # noqa: E402
from test_host import random_descs  # noqa: E402

Ns = range(params.N_MIN, params.N_MAX + 1)
ROLE_SETS = ((), ("A",), ("B",), ("A", "B"))     # the CALLER's packed operands
VECS = ((0, params.VEC_GENERATE), (1, params.VEC_RESIDENT))
STATS = {"ws": 0, "packs": 0, "plan_bytes": 0, "shapes": 0, "rn_not_supported": 0}


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def _tensor(extent, stride, label):
    r = len(extent)
    t = host._Tensor(r, (C.c_int64 * max(r, 1))(*extent), (C.c_int64 * max(r, 1))(*stride),
                     (C.c_int32 * max(r, 1))(*label))
    return t


class Case:
    """One contraction's descriptors as C structs, its lowering and the pack descriptors of
    both operands (imd_pack_desc_for), mutated in place per (N, g, rounding, vec): one C call
    per query, no struct rebuilt."""

    def __init__(self, h, descs, policy):
        self.h, self.L = h, h.lib
        self.t = [_tensor(*d) for d in descs]
        self.policy = policy
        self.o = host._Options(8, 0, 0, 0, 0, 0)
        lw = host._Lowering()
        lw.struct_size = C.sizeof(host._Lowering)
        st = self.L.imd_lowering(*self.args(), C.byref(lw))
        self.ok = bool(lw.lowered) and lw.dispatch == 1     # EMULATE: the N-dependent case
        if not self.ok:
            return
        self.m, self.n, self.k, self.b = lw.m, lw.n, lw.k, lw.b
        self.swapped, self.A = bool(lw.swapped), lw.k_align
        need(st in (0, 2), f"imd_lowering status {st}")
        need(self.A == params.k_align_of(self.k, policy), f"k_align {self.A} under {policy}")
        self.pd, self.keep = [None, None], []
        for x in (0, 1):
            cap = max(len(descs[x][0]), 1)
            labels = (C.c_int32 * cap)()
            d = host._PackDesc()
            d.struct_size = C.sizeof(host._PackDesc)
            st = self.L.imd_pack_desc_for(*self.args(), x, C.byref(d), labels, cap)
            need(st == 0, f"imd_pack_desc_for({x}) status {st}")
            self.pd[x] = d
            self.keep.append(labels)
        # an operand whose own K is empty (its partner's K labels reach it at stride 0) has
        # no pack: imd_pack_layout NOT_SUPPORTED at any N and rounding (§6b.3)
        self.packable = []
        for d in self.pd:
            d.num_moduli, d.rounding = params.N_MAX, 1
            r = host._PackLayout()
            r.struct_size = C.sizeof(host._PackLayout)
            st = self.L.imd_pack_layout(C.byref(d), C.byref(r))
            need(st in (0, 2), f"imd_pack_layout status {st}")
            self.packable.append(st == 0)
        self.packing = {}
        for roles in ROLE_SETS:
            if all(self.packable["AB".index(x)] for x in roles):
                self.packing[roles] = host._Packing(
                    C.pointer(self.pd[0]) if "A" in roles else None,
                    C.pointer(self.pd[1]) if "B" in roles else None)
        # a packed set the plan accepts (PACK_MISMATCH: a K label the packed operand does not
        # carry, §6b.10 #10, is a plan-time rejection at every N)
        self.set(params.N_MAX, 0, 1, 0)
        self.role_sets = tuple(rs for rs in self.packing if self.ws(rs)[0] == 0)

    def args(self):
        return [C.byref(x) for x in self.t] + [C.byref(self.o)]

    def post_swap(self, roles):
        """The post-swap roles (params' `packed`) of the caller's packed operands."""
        return tuple(sorted({("B" if r == "A" else "A") if self.swapped else r for r in roles}))

    def set(self, N, g, rnd, vec):
        self.o.num_moduli, self.o.residue_group, self.o.rounding, self.o.vec_mode = N, g, rnd, vec
        for d in self.pd:
            d.num_moduli, d.rounding = N, rnd

    def ws(self, roles):
        out = C.c_size_t(0)
        if roles:
            st = self.L.imd_workspace_size_packed(*self.args(), C.byref(self.packing[roles]),
                                                  C.byref(out))
        else:
            st = self.L.imd_workspace_size(*self.args(), C.byref(out))
        STATS["ws"] += 1
        return st, out.value


def rn_ok(N, k):
    return k <= params.rn_feasible_kmax(N)


def check_workspace(c, vec, pvec, roles, rnd, gsets, what):
    """F1 and C-ws for one (case, vec, packed set, rounding) over N with g in gsets(N); every
    value == params. Returns {(N, g): bytes} of the SUCCESS points."""
    t = {}
    packed = c.post_swap(roles)
    for N in Ns:
        for g in gsets(N):
            c.set(N, g, rnd, vec)
            st, ws = c.ws(roles)
            # with packed roles the pack descriptors' own status comes first (§6b.11): RN with
            # kappa >= T(N) is NOT_SUPPORTED; imd_workspace_size alone never applies the rule
            want_st = 2 if (roles and rnd == 0 and not rn_ok(N, c.k)) else 0
            need(st == want_st, f"{what}: N={N} g={g} status {st}, want {want_st}")
            if st:
                continue
            want = params.workspace_size(c.m, c.n, c.k, c.b, N, g, params.EMULATE, pvec, packed,
                                         c.A)
            need(ws == want, f"{what}: N={N} g={g} {ws} bytes, params {want}")
            t[(N, g)] = ws
    for (N, g), ws in t.items():
        if g == 0 and (N, N) in t:
            need(ws == t[(N, N)], f"{what}: N={N} g=0 {ws} != g=N {t[(N, N)]}")
        if (N, g + 1) in t and g >= 1:
            need(ws <= t[(N, g + 1)], f"F1 in g: {what}: N={N} g={g}")
        if (N + 1, g) in t:
            need(ws <= t[(N + 1, g)], f"F1 in N: {what}: N={N} g={g}")
    # the SUCCESS points of a packed RN query are an up-set in N (RN acceptance)
    okN = sorted({N for N, _ in t})
    need(okN == list(range(okN[0], params.N_MAX + 1)) if okN else True,
         f"{what}: SUCCESS N not an up-set: {okN}")
    # C-ws, directly: the maximum over every phase (N' <= N, 1 <= g' <= g; g' = 0 is g' = N')
    # is the (N, g) workspace itself (a 2-D prefix maximum over the SUCCESS points)
    pm = {}
    for (N, g) in sorted(t):
        if g == 0:
            continue
        pm[(N, g)] = max(t[(N, g)], pm.get((N - 1, g), 0), pm.get((N, g - 1), 0))
        need(pm[(N, g)] == t[(N, g)], f"C-ws: {what}: a phase below ({N},{g}) needs more")
    return t


def check_i3(tabs, what):
    for sub in ROLE_SETS:
        for sup in ROLE_SETS:
            if set(sub) < set(sup) and sub in tabs and sup in tabs:
                for key, ws in tabs[sup].items():
                    if key in tabs[sub]:
                        need(ws <= tabs[sub][key], f"I3: {what}: packed {sup} {ws} > {sub} "
                                                   f"{tabs[sub][key]} at (N, g) = {key}")


def check_packs(c, rnd, what):
    """F2, I1, C-pack for both operands' packs across N; every layout == params."""
    for x in (0, 1):
        if not c.packable[x]:
            continue
        d = c.pd[x]
        lays = {}
        for N in Ns:
            d.num_moduli, d.rounding = N, rnd
            r = host._PackLayout()
            r.struct_size = C.sizeof(host._PackLayout)
            st = c.L.imd_pack_layout(C.byref(d), C.byref(r))
            STATS["packs"] += 1
            want_st = 2 if (rnd == 0 and not rn_ok(N, r.k)) else 0
            need(st == want_st, f"{what} pack {'AB'[x]}: N={N} status {st}, want {want_st}")
            fmt = params.pack_format_of(params.k_align_of(r.k, c.policy))
            need(r.format == fmt, f"{what} pack {'AB'[x]}: format {r.format}, want {fmt}")
            regs, total = params.pack_layout(r.b, r.r, r.k, N, fmt)
            got = [("sig", r.sig.offset, r.sig.bytes), ("flag", r.flag.offset, r.flag.bytes),
                   ("res", r.res.offset, r.res.bytes)]
            need(got == regs and r.pack_bytes == total,
                 f"{what} pack {'AB'[x]} N={N}: {got} {r.pack_bytes}, params {regs} {total}")
            need(r.pack_ws_bytes == params.pack_ws_bytes(r.b, r.r, r.k), "pack_ws_bytes")
            if st == 0:
                lays[N] = (got, r.pack_bytes, r.pack_ws_bytes)
        if not lays:
            continue
        top_regs, top, top_ws = lays[max(lays)]
        for N, (regs, total, pws) in lays.items():
            need(regs[0] == top_regs[0] and regs[1] == top_regs[1],
                 f"I1: {what} pack {'AB'[x]}: sig / flag differ at N={N}")
            need(regs[2][1] == top_regs[2][1] and regs[2][2] <= top_regs[2][2],
                 f"I1: {what} pack {'AB'[x]}: res(N={N}) not a prefix of res(N_max)")
            need(pws == top_ws, f"I1: {what} pack {'AB'[x]}: pack_ws_bytes depends on N")
            need(total <= top, f"C-pack: {what} pack {'AB'[x]} N={N}")
            if N + 1 in lays:
                need(total <= lays[N + 1][1], f"F2: {what} pack {'AB'[x]}: N={N}")


def check_plan_bytes(c, vec, pvec, gsets, what):
    """I2: imd_plan_bytes' device bytes, RN, every (N, g): one value (params), NOT_SUPPORTED
    exactly where kappa >= T(N); and imd_lowering_packed's plan device bytes (A and B packed,
    TRUNC) one value."""
    dev_seen, host_seen = set(), set()
    for N in Ns:
        for g in gsets(N):
            c.set(N, g, 0, vec)
            dev, hb = C.c_size_t(0), C.c_size_t(0)
            st = c.L.imd_plan_bytes(*c.args(), C.byref(dev), C.byref(hb))
            STATS["plan_bytes"] += 1
            want_st = 0 if rn_ok(N, c.k) else 2
            need(st == want_st, f"{what}: imd_plan_bytes N={N} g={g} status {st}")
            if st:
                STATS["rn_not_supported"] += 1
                continue
            dev_seen.add(dev.value)
            host_seen.add(hb.value)
    if dev_seen:
        need(dev_seen == {params.plan_device_bytes(c.m, c.n, c.k, c.b, pvec)},
             f"I2: {what}: plan device bytes {sorted(dev_seen)} vary with (N, g) or != params")
    seen = set()
    for N in Ns:
        for g in (0, 1):
            c.set(N, g, 1, vec)
            lw = host._Lowering()
            lw.struct_size = C.sizeof(host._Lowering)
            st = c.L.imd_lowering_packed(*c.args(), C.byref(c.packing[c.role_sets[-1]]),
                                         C.byref(lw))
            need(st == 0, f"{what}: imd_lowering_packed N={N} status {st}")
            seen.add(lw.plan_device_bytes)
    need(len(seen) == 1, f"I2: {what}: packed plan device bytes vary with (N, g): {seen}")
    return len(host_seen)


def run_case(c, what, vecs, role_sets, gsets, rnds_packed=(1,)):
    tabs = {}
    for vec, pvec in vecs:
        tabs = {}
        for roles in role_sets:
            for rnd in ((0,) if not roles else rnds_packed):
                rname = "RN" if rnd == 0 else "TRUNC"
                t = check_workspace(c, vec, pvec, roles, rnd, gsets,
                                    f"{what} vec {pvec} packed {roles} {rname}")
                if rnd == 1 or not roles:
                    tabs[roles] = t
        check_i3(tabs, f"{what} vec {pvec}")
        check_plan_bytes(c, vec, pvec, gsets, f"{what} vec {pvec}")
    for rnd in rnds_packed:
        check_packs(c, rnd, what)
    STATS["shapes"] += 1


def suite_grid(h, verbose):
    i, j, k, lab = 1, 2, 3, 4
    ms = (1, 15, 16, 17, 255, 256, 257)
    ks = (2, 96, 97, 1024, 1025, 131008, 131009)
    t0, n = time.time(), 0
    for policy, hook in (("AUTO", 0), (params.K_ALIGN, 16)):
        host.test_set_k_align(hook, lib=h)
        try:
            for m in ms:
                for nn in ms:
                    for kk in ks:
                        for b in (1, 3):
                            e = ([b, m, kk], [b, kk, nn], [b, m, nn])
                            descs = [(x, host.dense_strides(x), l)
                                     for x, l in zip(e, ([lab, i, k], [lab, k, j], [lab, i, j]))]
                            c = Case(h, descs, policy)
                            need(c.ok and c.role_sets == ROLE_SETS,
                                 f"grid shape {e} not EMULATE, or an operand without a pack")
                            run_case(c, f"grid m={m} n={nn} k={kk} b={b} A={policy}", VECS,
                                     ROLE_SETS, lambda N: range(0, N + 1),
                                     rnds_packed=(1, 0) if kk in (131008, 131009) else (1,))
                            n += 1
        finally:
            host.test_set_k_align(0, lib=h)
    if verbose:
        print(f"  ok  grid: {n} (shape, alignment) cases, every (N, g), both vec modes, every "
              f"packed set ({time.time() - t0:.1f}s)", flush=True)


def suite_random(h, count, verbose):
    rng = random.Random(1907)
    t0, n, tried = time.time(), 0, 0
    while n < count:
        tried += 1
        descs = random_descs(rng, legal_bias=1.0)
        hook = rng.choice((0, 0, 16, 64))
        policy = "AUTO" if hook == 0 else hook
        host.test_set_k_align(hook, lib=h)
        try:
            c = Case(h, descs, policy)
            if not c.ok:
                continue
            if len(c.role_sets) < 2:
                continue
            vec = rng.choice(VECS)
            roles = rng.choice(c.role_sets[1:])
            gs = {}

            def gsets(N):
                if N not in gs:
                    gs[N] = sorted({0, 1, 2, rng.randint(1, N), N} & set(range(0, N + 1)))
                return gs[N]
            run_case(c, f"random[{n}] {descs} A={policy}", (vec,), ((), roles), gsets,
                     rnds_packed=(rng.choice((0, 1)),))
            n += 1
        finally:
            host.test_set_k_align(0, lib=h)
    if verbose:
        print(f"  ok  random: {n} legal EMULATE contractions ({tried} drawn), every N, sampled "
              f"g ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", default=None)
    ap.add_argument("--random", type=int, default=10000)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    h = host.load(a.host_lib)
    need(h.version >= 13, f"host API {h.version} < 13 (the footprint guarantees are from 13 on)")
    t0 = time.time()
    try:
        suite_grid(h, a.verbose)
        suite_random(h, a.random, a.verbose)
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  {STATS['shapes']} cases: {STATS['ws']} workspace sizes, {STATS['packs']} pack "
          f"layouts, {STATS['plan_bytes']} plan sizes ({STATS['rn_not_supported']} RN "
          f"NOT_SUPPORTED) == params; F1, F2, C-ws, C-pack, I1-I3 in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
