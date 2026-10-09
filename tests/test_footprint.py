#!/usr/bin/env python3
"""libimmerdink_host's footprints across N and g (DESIGN v1.19 §6g / §6a; NOTES-C1 §4.1).

    python3 tests/test_footprint.py --host-lib build/libimmerdink_host.so [--random 10000] [-v]

Precision phases size one workspace and one pack buffer for every phase. Checked on the
library, every value equal to spec/params.py (cert_footprint_monotone certifies the formulas):
  - F1 (ABI from API 13 on): imd_workspace_size and imd_workspace_size_packed are
    nondecreasing in N at every fixed g >= 1, in g in [1, N], and in N with g = 0 (= N), over
    the (N, g) where they succeed; every status == params (INVALID_VALUE: a size past uint64,
    which grows with N and g; NOT_SUPPORTED: an RN pack's kappa >= T(N)), so at each g the N
    form a NOT_SUPPORTED prefix, a SUCCESS interval and an INVALID_VALUE suffix;
    C-ws (the downward statement): if the query succeeds at (N_max, g_max), every
    (N <= N_max, g <= g_max) succeeds with at most those bytes or is an RN pack's
    NOT_SUPPORTED;
  - F2 (ABI): imd_pack_layout's pack_bytes is nondecreasing in N (the same status pattern);
    C-pack;
  - I1 (internal): the sig and flag offsets and bytes, the res offset and pack_ws_bytes do
    not depend on N; res(N') is a prefix of res(N_max);
  - I2 (internal): imd_plan_bytes' device bytes (and imd_lowering_packed's plan device bytes)
    do not depend on N or g; NOT_SUPPORTED (the RN acceptance rule) is exactly
    k > rn_feasible_kmax(N), so the accepted N are an up-set;
  - I3 (internal): packing more roles never increases the workspace at any (N, g).
Grid: m, n in {1, 15, 16, 17, 255, 256, 257}, k in {2, 96, 97, 1024, 1025, 131008, 131009},
b in {1, 3} ("lik,lkj->lij"), N = 2..20, g = 0..N, both vec modes, the caller's packed roles
none / A / B / AB, the K alignment AUTO and A16 (libimmerdink_host's test hook); then random
legal contractions (tests/test_host.py random_descs: permuted and negative strides, the role
swap, diagonals) at every N with sampled g, a random vec mode, packed set and alignment
(16, 64 or AUTO); then "ik,kj->ij" shapes whose sizes pass uint64 at large N or g (RA and a
pack's res; W), every (N, g), RN and TRUNC, where C-ws's downward statement is also checked
pair by pair. CPU only; no GPU, no CUDA.
"""

import argparse
import ctypes as C
import os
import random
import re
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
STATS = {"ws": 0, "packs": 0, "plan_bytes": 0, "shapes": 0, "rn_not_supported": 0, "pairs": 0}


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
    per query, no struct rebuilt. The N-independent queries (lowering, pack descriptors,
    packability, the packed sets the plan accepts) run at N = 8 (lowering) and N_max with
    g = 0, or all at `base` = (N, g) with TRUNC: a shape whose sizes pass uint64 at large N
    or g takes a small base."""

    def __init__(self, h, descs, policy, base=None):
        self.h, self.L = h, h.lib
        self.t = [_tensor(*d) for d in descs]
        self.policy = policy
        Nb, gb = base or (params.N_MAX, 0)
        self.o = (host._Options(8, 0, 0, 0, 0, 0) if base is None else
                  host._Options(Nb, gb, 1, 0, 0, 0))
        self.inline, self.packst = {}, {}   # status caches: (N, g, vec) in-line, (x, N, rnd) packs
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
        self.packable, self.pack_dims = [], []
        for d in self.pd:
            d.num_moduli, d.rounding = Nb, 1
            r = host._PackLayout()
            r.struct_size = C.sizeof(host._PackLayout)
            st = self.L.imd_pack_layout(C.byref(d), C.byref(r))
            need(st in (0, 2), f"imd_pack_layout status {st}")
            self.packable.append(st == 0)
            self.pack_dims.append((r.b, r.r, r.k, r.format))   # the pack's (b, r, k, format)
        self.packing = {}
        for roles in ROLE_SETS:
            if all(self.packable["AB".index(x)] for x in roles):
                self.packing[roles] = host._Packing(
                    C.pointer(self.pd[0]) if "A" in roles else None,
                    C.pointer(self.pd[1]) if "B" in roles else None)
        # a packed set the plan accepts (PACK_MISMATCH: a K label the packed operand does not
        # carry, §6b.10 #10, is a plan-time rejection at every N)
        self.set(Nb, gb, 1, 0)
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


SUCCESS, INVALID, NOT_SUPPORTED = 0, 1, 2
PATTERN = re.compile("N*S*I*")          # over increasing N: NOT_SUPPORTED, SUCCESS, INVALID_VALUE


def letters(sts):
    return "".join("SIN"[s] for s in sts)


def pack_status(c, x, N, rnd):
    """imd_pack_layout's status for the caller's operand x at (N, rnd), from params: a size
    past uint64 (INVALID_VALUE), then RN with kappa >= T(N) (NOT_SUPPORTED, §6b.3)."""
    key = (x, N, rnd)
    if key not in c.packst:
        b, r, k, fmt = c.pack_dims[x]
        try:
            params.pack_layout(b, r, k, N, fmt)
            c.packst[key] = NOT_SUPPORTED if (rnd == 0 and k >= 2 and not rn_ok(N, k)) else SUCCESS
        except OverflowError:
            c.packst[key] = INVALID
    return c.packst[key]


def ws_status(c, N, g, rnd, pvec, roles):
    """The workspace query's status at (N, g) from params, in the library's order: §5 row 5 on
    the in-line layout (every region and the total, even of a packed role, §6b.6), then each
    of the caller's packed operands' pack_status (A, then B). imd_workspace_size alone never
    applies the RN rule."""
    key = (N, g, pvec)
    if key not in c.inline:
        try:
            params.workspace_size(c.m, c.n, c.k, c.b, N, g, params.EMULATE, pvec, (), c.A)
            c.inline[key] = SUCCESS
        except OverflowError:
            c.inline[key] = INVALID
    if c.inline[key]:
        return c.inline[key]
    for x in (0, 1):
        if "AB"[x] in roles and (st := pack_status(c, x, N, rnd)):
            return st
    return SUCCESS


def check_workspace(c, vec, pvec, roles, rnd, gsets, what):
    """F1 and C-ws for one (case, vec, packed set, rounding) over N with g in gsets(N); every
    status and value == params. Returns ({(N, g): bytes} of the SUCCESS points, {(N, g):
    status} of every point)."""
    t, sts = {}, {}
    packed = c.post_swap(roles)
    for N in Ns:
        for g in gsets(N):
            c.set(N, g, rnd, vec)
            st, ws = c.ws(roles)
            want_st = ws_status(c, N, g, rnd, pvec, roles)
            need(st == want_st, f"{what}: N={N} g={g} status {st}, want {want_st}")
            sts[(N, g)] = st
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
    # at each g (g = 0 too), over increasing N: a NOT_SUPPORTED prefix (an RN pack's kappa), a
    # SUCCESS interval, an INVALID_VALUE suffix (a size past uint64): an interval, no up-set
    for g in sorted({g for _, g in sts}):
        seq = letters(sts[(N, g)] for N in Ns if (N, g) in sts)
        need(PATTERN.fullmatch(seq), f"{what}: g={g}: statuses over N {seq}")
    # C-ws's downward statement, by steps: below a SUCCESS (N, g >= 1), (N, g - 1) succeeds and
    # (N - 1, g) succeeds or is NOT_SUPPORTED; below a NOT_SUPPORTED, NOT_SUPPORTED (with the
    # byte steps above this gives every (N' <= N, g' <= g))
    for (N, g), st in sts.items():
        if g == 0:
            need((N, N) not in sts or sts[(N, N)] == st, f"{what}: N={N} g=0 status != g=N")
            continue
        if st == SUCCESS and g >= 2 and (N, g - 1) in sts:
            need(sts[(N, g - 1)] == SUCCESS, f"C-ws: {what}: ({N},{g}) succeeds, ({N},{g - 1}) "
                                             f"status {sts[(N, g - 1)]}")
        if st != INVALID and (N - 1, g) in sts:
            need(sts[(N - 1, g)] in ((SUCCESS, NOT_SUPPORTED) if st == SUCCESS else
                                     (NOT_SUPPORTED,)),
                 f"C-ws: {what}: ({N},{g}) status {st}, ({N - 1},{g}) {sts[(N - 1, g)]}")
    # C-ws, directly: the maximum over every phase (N' <= N, 1 <= g' <= g; g' = 0 is g' = N')
    # is the (N, g) workspace itself (a 2-D prefix maximum over the SUCCESS points)
    pm = {}
    for (N, g) in sorted(t):
        if g == 0:
            continue
        pm[(N, g)] = max(t[(N, g)], pm.get((N - 1, g), 0), pm.get((N, g - 1), 0))
        need(pm[(N, g)] == t[(N, g)], f"C-ws: {what}: a phase below ({N},{g}) needs more")
    return t, sts


def check_i3(tabs, what):
    for sub in ROLE_SETS:
        for sup in ROLE_SETS:
            if set(sub) < set(sup) and sub in tabs and sup in tabs:
                for key, ws in tabs[sup].items():
                    if key in tabs[sub]:
                        need(ws <= tabs[sub][key], f"I3: {what}: packed {sup} {ws} > {sub} "
                                                   f"{tabs[sub][key]} at (N, g) = {key}")


def check_packs(c, rnd, what):
    """F2, I1, C-pack for both operands' packs across N; every status and layout == params (a
    NOT_SUPPORTED prefix, then SUCCESS, then an INVALID_VALUE suffix over increasing N)."""
    for x in (0, 1):
        if not c.packable[x]:
            continue
        d = c.pd[x]
        lays, sts = {}, []
        for N in Ns:
            d.num_moduli, d.rounding = N, rnd
            r = host._PackLayout()
            r.struct_size = C.sizeof(host._PackLayout)
            st = c.L.imd_pack_layout(C.byref(d), C.byref(r))
            STATS["packs"] += 1
            want_st = pack_status(c, x, N, rnd)
            need(st == want_st, f"{what} pack {'AB'[x]}: N={N} status {st}, want {want_st}")
            sts.append(st)
            if st == INVALID:           # only struct_size and status are written
                continue
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
        need(PATTERN.fullmatch(letters(sts)),
             f"{what} pack {'AB'[x]}: statuses over N {letters(sts)}")
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
                t, _ = check_workspace(c, vec, pvec, roles, rnd, gsets,
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


# "ik,kj->ij" (m, k, n) whose sizes pass uint64 at large N or g (§5 row 5: INVALID_VALUE):
#   - m = 2^30, k = 2^31, n = 16: RA = g m k_hat (RB if swapped) passes 2^64 at g >= 8 (g = 0:
#     N >= 8), and so does the caller's A pack (res = N m k_hat) at N >= 8; RN rejects
#     k = 2^31 at the smallest N, so A packed with RN gives NOT_SUPPORTED, SUCCESS and
#     INVALID_VALUE over N at g = 0 (the re-review's counterexample to an up-set in N);
#   - m = n = 2^30, k = 2: W = N m n_hat passes 2^64 at N >= 16, at every g.
OVERFLOW_SHAPES = ((2**30, 2**31, 16), (2**30, 2, 2**30))


def runs(seq):
    """'NNSSI...' over Ns -> 'NOT_SUPPORTED N=2..3, SUCCESS 4..4, INVALID_VALUE 5..20'."""
    out, N0 = [], params.N_MIN
    for m in re.finditer(r"(.)\1*", seq):
        a, b = N0 + m.start(), N0 + m.end() - 1
        out.append(f"{dict(N='NOT_SUPPORTED', S='SUCCESS', I='INVALID_VALUE')[m.group(1)]} "
                   f"{'N=' if not out else ''}{a}..{b}")
    return ", ".join(out)


def check_downward(t, sts, c, roles, rnd, what):
    """C-ws's downward statement (DESIGN §6a F1), pair by pair: for every SUCCESS (N_max, g_max)
    and every (N, g) with 1 <= g <= N <= N_max and g <= g_max (g = 0 counts as N), (N, g)
    succeeds with at most ws(N_max, g_max) bytes, or is NOT_SUPPORTED by an RN pack's
    kappa >= T(N)."""
    pts = {(N, g or N): (st, t.get((N, g))) for (N, g), st in sts.items()}
    for (Nm, gm), (stm, wm) in pts.items():
        if stm != SUCCESS:
            continue
        for (N, g), (st, w) in pts.items():
            if N > Nm or g > gm:
                continue
            STATS["pairs"] += 1
            need((st == SUCCESS and w <= wm) or
                 (st == NOT_SUPPORTED and roles and rnd == 0 and not rn_ok(N, c.k)),
                 f"C-ws: {what}: ({Nm},{gm}) succeeds with {wm} bytes, ({N},{g}) status {st} "
                 f"({w} bytes)")


def suite_overflow(h, verbose):
    """F1, F2 and C-ws where a size passes uint64 (OVERFLOW_SHAPES): every status and value ==
    params at every (N, g), RN and TRUNC, both vec modes, every packed set; C-ws's downward
    statement on every pair; the statuses over N form the pattern the promise allows, and the
    re-review's counterexample is reproduced."""
    i, j, k = 1, 2, 3
    t0, seqs = time.time(), {}
    for s, (m, kk, nn) in enumerate(OVERFLOW_SHAPES):
        e = ([m, kk], [kk, nn], [m, nn])
        descs = [(x, host.dense_strides(x), l) for x, l in zip(e, ([i, k], [k, j], [i, j]))]
        c = Case(h, descs, "AUTO", base=(params.N_MIN, 1))
        need(c.ok and c.role_sets == ROLE_SETS,
             f"overflow shape {e}: not EMULATE, or a packed set refused at N = 2, g = 1")
        for vec, pvec in VECS:
            tabs = {}
            for roles in ROLE_SETS:
                for rnd in (0, 1):
                    what = (f"overflow m={m} k={kk} n={nn} vec {pvec} packed {roles} "
                            f"{'RN' if rnd == 0 else 'TRUNC'}")
                    t, sts = check_workspace(c, vec, pvec, roles, rnd, lambda N: range(0, N + 1),
                                             what)
                    check_downward(t, sts, c, roles, rnd, what)
                    for g in (0, 1):
                        seqs[(s, pvec, roles, rnd, g)] = letters(sts[(N, g)] for N in Ns)
                    if rnd == 1:
                        tabs[roles] = t
            check_i3(tabs, f"overflow m={m} k={kk} n={nn} vec {pvec}")
        for rnd in (0, 1):
            check_packs(c, rnd, f"overflow m={m} k={kk} n={nn}")
        STATS["shapes"] += 1
    G = params.VEC_GENERATE
    squeeze = lambda q: re.sub(r"(.)\1+", r"\1", q)  # noqa: E731
    for key, want, why in (((0, G, ("A",), 0, 0), "NSI", "A packed, RN, g = 0"),
                           ((0, G, ("A",), 0, 1), "NSI", "A packed, RN, g = 1"),
                           ((0, G, ("A",), 1, 0), "SI", "A packed, TRUNC, g = 0"),
                           ((0, G, ("A",), 1, 1), "SI", "A packed, TRUNC, g = 1"),
                           ((0, G, (), 1, 0), "SI", "plain, g = 0"),
                           ((0, G, (), 1, 1), "S", "plain, g = 1"),
                           ((1, G, (), 1, 1), "SI", "plain, g = 1")):
        need(squeeze(seqs[key]) == want,
             f"overflow shape {OVERFLOW_SHAPES[key[0]]} {why}: {runs(seqs[key])}, want {want}")
        if verbose:
            print(f"  ok  overflow m, k, n = {OVERFLOW_SHAPES[key[0]]}, {why}: {runs(seqs[key])}",
                  flush=True)
    if verbose:
        print(f"  ok  overflow: {len(OVERFLOW_SHAPES)} shapes, every (N, g), RN and TRUNC, both "
              f"vec modes, every packed set; {STATS['pairs']} C-ws pairs "
              f"({time.time() - t0:.1f}s)", flush=True)


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
        suite_overflow(h, a.verbose)
    except Fail as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  {STATS['shapes']} cases: {STATS['ws']} workspace sizes, {STATS['packs']} pack "
          f"layouts, {STATS['plan_bytes']} plan sizes ({STATS['rn_not_supported']} RN "
          f"NOT_SUPPORTED) == params, {len(OVERFLOW_SHAPES)} shapes past uint64 with "
          f"{STATS['pairs']} C-ws pairs; F1, F2, C-ws, C-pack, I1-I3 in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
