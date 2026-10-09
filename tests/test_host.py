#!/usr/bin/env python3
"""libimmerdink_host (DESIGN §6a) tests: the Python wrapper, the C ABI, the independent
spec (spec/params.py, spec/oracle.py) and real libimmerdink plans must agree exactly.

    python3 tests/test_host.py --host-lib build/libimmerdink_host.so --suite host
    python3 tests/test_host.py --host-lib ... --lib build/libimmerdink.so --suite plan

Suites:
  host  CPU only. On random descriptors (legal and illegal, every vec_mode, mode, rounding,
        g = 1..N): python/immerdink/host.py == direct C calls (own ctypes declarations) ==
        spec: status (oracle.lower + oracle.Plan, incl. the kappa >= T rule), m n k b swap,
        k_hat n_hat kappa, dispatch, groups, K4 chunks and launches (counted here from
        oracle.Plan.groups()), workspace bytes and regions (params.workspace_size /
        region_offsets), plan device bytes (params.plan_device_bytes). Plus the einsum front
        end and g_tradeoff.
  perf  CPU only. The performance model (DESIGN v1.8 §6c): the shipped calibration, format /
        parse round trips, every phase's features == an independent transcription of the
        §6c.1 table, K4 features x the A3 constants == the k4_select estimate, monotonicity,
        statuses, packed predictions, and imd_perf_rank's exact footprints, budget and order.
  plan  GPU. imd_plan_create on random descriptors (also K-chunked, batch-split and
        n-hat-split shapes): the plan's GEMM view, workspace layout, device bytes, K4
        parameter sets and launches, and host payload (test hook imd_test_plan_sizes) ==
        imd_lowering / imd_plan_bytes / imd_workspace_layout.
"""

import argparse
import itertools
import ctypes as C
import math
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
sys.path.insert(0, os.path.join(ROOT, "python"))
import oracle  # noqa: E402
import params  # noqa: E402
from immerdink import host  # noqa: E402

STATUS = host.STATUS
RN, TRUNC = "RN", "TRUNC"
# DESIGN v1.13 §4b: the K-alignment policy the C library runs and the spec side models.
# "AUTO" (the library's rule) unless a test forces A with set_k_align (the C hook
# imd_test_set_k_align and params / oracle.K_ALIGN_POLICY move together).
HOOK = "AUTO"


def set_k_align(a):
    """Force A (16 / 64) or restore AUTO ("AUTO" / 0) in the C library AND in params / oracle,
    so every spec default (k_hat_of, workspace_size, pack_layout, lower_packed) follows."""
    global HOOK
    a = "AUTO" if a in (0, "AUTO") else int(a)
    host.test_set_k_align(0 if a == "AUTO" else a)
    HOOK = params.K_ALIGN_POLICY = oracle.K_ALIGN_POLICY = a


def plan_k_align(k, dispatch):
    """The lowering's k_align (§4b.5): the resolved A of a plan with residue rows (EMULATE),
    16 for NOOP, SCALE, DIRECT and K-empty."""
    return params.k_align_of(k, HOOK) if dispatch == "EMULATE" else 16


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------------------
# Direct C access, declared independently of python/immerdink/host.py.

class CT(C.Structure):
    _fields_ = [("rank", C.c_int32), ("extent", C.c_void_p), ("stride", C.c_void_p),
                ("label", C.c_void_p)]


class CO(C.Structure):
    _fields_ = [("N", C.c_int32), ("g", C.c_int32), ("rnd", C.c_int32), ("mode", C.c_int32),
                ("vec", C.c_int32), ("k4_tune", C.c_int32)]


LOWERING_FIELDS = ["status", "lowered", "dispatch", "swapped", "num_moduli", "residue_group",
                   "num_groups", "m", "n", "k", "b", "n_hat", "k_hat", "vec_entries",
                   "k_chunks", "gemm_launches", "kappa", "workspace_bytes",
                   "plan_device_bytes", "plan_host_bytes", "fused", "kernel_launches",
                   "sm_count", "k4_config", "k4_tune_bytes", "fused_large", "packed",
                   "pack_mismatch", "k_align"]
# sizeof(imd_lowering_t) per host API version (DESIGN §6a; v5: §6b; v6: v1.13 §4b)
V1, V2, V3, V4, V5, V6 = 136, 152, 168, 176, 184, 192


class CPD(C.Structure):
    """imd_pack_desc_t (DESIGN v1.7 §6b.5), declared independently of the wrapper."""
    _fields_ = [("size", C.c_uint32), ("N", C.c_int32), ("rnd", C.c_int32), ("order", C.c_int32),
                ("tensor", C.c_void_p), ("nk", C.c_int32), ("k", C.c_void_p),
                ("nl", C.c_int32), ("l", C.c_void_p), ("nr", C.c_int32), ("r", C.c_void_p)]


class Direct:
    def __init__(self, path):
        self.lib = C.CDLL(path)
        self.keep = None

    def _args(self, descs, opts):
        """C arguments of one call; the arrays stay alive (self.keep) until the next."""
        arrays, ts = [], []
        for e, s, lab in descs:
            r = len(e)
            arrs = ((C.c_int64 * r)(*e), (C.c_int64 * r)(*s), (C.c_int32 * r)(*lab))
            arrays.append(arrs)
            ts.append(CT(r, *(C.cast(a, C.c_void_p) for a in arrs)))
        o = CO(*opts)
        self.keep = (arrays, ts, o)
        return [C.byref(t) for t in ts] + [C.byref(o)]

    def workspace(self, descs, opts):
        out = C.c_size_t(7)
        st = self.lib.imd_workspace_size(*self._args(descs, opts), C.byref(out))
        return st, out.value

    def layout(self, descs, opts):
        buf = (C.c_uint64 * 16)()
        cnt, tot = C.c_int32(0), C.c_size_t(0)
        st = self.lib.imd_workspace_layout(*self._args(descs, opts), buf, C.c_int32(8),
                                           C.byref(cnt), C.byref(tot))
        return st, tot.value, [(buf[2 * r], buf[2 * r + 1]) for r in range(8)]

    def plan_bytes(self, descs, opts):
        d, h = C.c_size_t(0), C.c_size_t(0)
        st = self.lib.imd_plan_bytes(*self._args(descs, opts), C.byref(d), C.byref(h))
        return st, d.value, h.value

    def _packing(self, packing):
        """An imd_packing_t of (PackSpec or None, PackSpec or None); arrays kept alive."""
        keep, ptrs = [], []
        for p in packing:
            if p is None:
                ptrs.append(None)
                continue
            X = p.tensor
            r = len(X.extent)
            arrs = ((C.c_int64 * max(r, 1))(*X.extent), (C.c_int64 * max(r, 1))(*X.stride),
                    (C.c_int32 * max(r, 1))(*X.label))
            t = CT(r, *(C.cast(a, C.c_void_p) for a in arrs))
            lists = [(C.c_int32 * max(len(x), 1))(*x) for x in (p.contracted, p.batch, p.rows)]
            d = CPD(C.sizeof(CPD), p.num_moduli, 0 if p.rounding == RN else 1,
                    0 if p.order == "AUTO" else 1, C.cast(C.pointer(t), C.c_void_p),
                    len(p.contracted), C.cast(lists[0], C.c_void_p) if p.contracted else None,
                    len(p.batch), C.cast(lists[1], C.c_void_p) if p.batch else None,
                    len(p.rows), C.cast(lists[2], C.c_void_p) if p.rows else None)
            keep.append((arrs, t, lists, d))
            ptrs.append(C.cast(C.pointer(d), C.c_void_p))
        pk = (C.c_void_p * 2)(*ptrs)
        return pk, keep

    def workspace_packed(self, descs, opts, packing):
        pk, keep = self._packing(packing)
        out = C.c_size_t(7)
        st = self.lib.imd_workspace_size_packed(*self._args(descs, opts), pk, C.byref(out))
        del keep
        return st, out.value

    def layout_packed(self, descs, opts, packing):
        pk, keep = self._packing(packing)
        buf = (C.c_uint64 * 16)()
        cnt, tot = C.c_int32(0), C.c_size_t(0)
        st = self.lib.imd_workspace_layout_packed(*self._args(descs, opts), pk, buf,
                                                  C.c_int32(8), C.byref(cnt), C.byref(tot))
        del keep
        return st, tot.value, [(buf[2 * r], buf[2 * r + 1]) for r in range(8)]

    def lowering(self, descs, opts, size=V6, sm_count=0, packing=None):
        """imd_lowering (imd_lowering_packed with `packing`) into a raw struct of `size`
        bytes (192: version 6; 184 / 176 / 168 / 152 / 136: a version-5 / -4 / -3 / -2 / -1 caller,
        whose struct must come back with its fields only). sm_count: the v3 input field
        (written only into a struct of version 3 or later)."""
        raw = (C.c_uint8 * (size + 16))(*([0xEE] * (size + 16)))
        C.c_uint32.from_buffer(raw, 0).value = size
        if size >= V3:
            C.c_int32.from_buffer(raw, V2).value = sm_count
        if packing is None:
            st = self.lib.imd_lowering(*self._args(descs, opts), raw)
        else:
            pk, keep = self._packing(packing)
            st = self.lib.imd_lowering_packed(*self._args(descs, opts), pk, raw)
            del keep
        if any(raw[i] != 0xEE for i in range(size, size + 16)):
            raise Fail("imd_lowering wrote past struct_size")
        i32 = [C.c_int32.from_buffer(raw, 4 * i).value for i in range(8)]
        i64 = [C.c_int64.from_buffer(raw, 32 + 8 * i).value for i in range(9)]
        u64 = [C.c_uint64.from_buffer(raw, 104 + 8 * i).value for i in range(4)]
        v2, v3, v4, v5, v6 = [], [], [], [], []
        if size >= V2:
            v2 = [C.c_int32.from_buffer(raw, 136).value, C.c_int64.from_buffer(raw, 144).value]
        if size >= V3:
            v3 = [C.c_int32.from_buffer(raw, 152).value, C.c_int32.from_buffer(raw, 156).value,
                  C.c_uint64.from_buffer(raw, 160).value]
        if size >= V4:
            v4 = [C.c_int32.from_buffer(raw, 168).value]
            if C.c_int32.from_buffer(raw, 172).value != 0:
                raise Fail("imd_lowering: reserved4 != 0")
        if size >= V5:
            v5 = [C.c_int32.from_buffer(raw, 176).value, C.c_int32.from_buffer(raw, 180).value]
        if size >= V6:
            v6 = [C.c_int32.from_buffer(raw, 184).value]
            if C.c_int32.from_buffer(raw, 188).value != 0:
                raise Fail("imd_lowering: reserved6 != 0")
        return st, dict(zip(LOWERING_FIELDS, i32[1:] + i64 + u64 + v2 + v3 + v4 + v5 + v6))


# ---------------------------------------------------------------------------------------
# Independent expectation from spec/

def spec_expect(descs, N, g, rnd, mode, vec):
    """(status, dict) from spec/oracle.py and spec/params.py alone."""
    A, B, Cd = (oracle.Desc(*d) for d in descs)
    omode = oracle.EMULATE if mode == "EMULATE" else oracle.DIRECT
    try:
        v = oracle.lower(A, B, Cd, mode=omode, build_vec=False)
    except oracle.InvalidValue:
        return "INVALID_VALUE", None
    except oracle.NotSupported:
        return "NOT_SUPPORTED", None
    pmode = params.EMULATE if mode == "EMULATE" else params.DIRECT
    pvec = params.VEC_GENERATE if vec == "GENERATE" else params.VEC_RESIDENT
    A_res = params.k_align_of(v.k, HOOK)   # v1.13 §4b.5: resolved once per call
    try:
        ws = params.workspace_size(v.m, v.n, v.k, v.b, N, g, pmode, pvec, k_align=A_res)
    except OverflowError:
        return "INVALID_VALUE", None
    m, n, k, b = v.m, v.n, v.k, v.b
    gg = N if g == 0 else g
    noop = b * m * n == 0
    dispatch = ("NOOP" if noop else "SCALE" if k == 0 else
                "DIRECT" if (mode == "DIRECT" or k == 1) else "EMULATE")
    k_align = plan_k_align(k, dispatch)
    k_hat, n_hat = params.k_hat_of(k, k_align), params.align_up(n, 16)
    exp = dict(m=m, n=n, k=k, b=b, swapped=bool(v.swapped), n_hat=n_hat, k_hat=k_hat,
               k_align=k_align,
               kappa=oracle.ceil_sqrt(k) if k > 0 else 0, workspace_bytes=ws,
               dispatch=dispatch, num_moduli=N, residue_group=gg,
               vec_entries=0 if noop else 2 * m + 2 * n + 2 * k + 3 * b,
               regions=params.region_offsets(m, n, k, b, N, g, pmode, pvec, k_align=A_res),
               plan_device_bytes=params.plan_device_bytes(m, n, k, b, pvec),
               num_groups=0, k_chunks=0, gemm_launches=0)
    status = "SUCCESS"
    if dispatch == "EMULATE":
        try:
            plan = oracle.Plan(N, k, rnd, g)
            groups = plan.groups()
        except oracle.NotSupported:
            status = "NOT_SUPPORTED"
            groups = [list(range(s, min(s + gg, N))) for s in range(0, N, gg)]
        chunks = -(-k_hat // params.K_C_PRIME)
        nparts = -(-n_hat // (65535 * 256))
        exp.update(num_groups=len(groups), k_chunks=chunks,
                   gemm_launches=sum(chunks * -(-len(grp) * b // 65535) * nparts
                                     for grp in groups))
    # v1.4 (DESIGN §4, §6a): the fused launch KF iff EMULATE and k_hat <= 1024; v1.6: else
    # KL for both operands (fused_large = 3) on EMULATE; kernel launches per call with
    # alpha != 0.
    k0 = 0 if vec == "RESIDENT" else 1
    fused = dispatch == "EMULATE" and k_hat <= 1024   # 512 until v1.5 (A3)
    fused_large = 3 if dispatch == "EMULATE" and not fused else 0
    if dispatch == "NOOP":
        launches = 0
    elif dispatch != "EMULATE":
        launches = k0 + 1                          # K0 + DIRECT / beta kernel
    else:
        zero_rows = 1 if n_hat > n else 0
        per_group = 2 + zero_rows                  # K2, K3 (+ B padding rows)
        group1 = 0 if fused else ((0 if fused_large & 1 else 1)
                                  + (0 if fused_large & 2 else 1 + zero_rows))
        launches = ((1 if fused else k0 + 2) + (exp["num_groups"] - 1) * per_group + group1
                    + exp["gemm_launches"] + 1)
    exp.update(fused=int(fused), kernel_launches=launches, fused_large=fused_large)
    # v1.5 (DESIGN §4, §6 k4_tune, §6a): the K4 rule for the reference 128 SMs and the
    # autotuning scratch bytes (both only with the emulated GEMM).
    emu = dispatch == "EMULATE"
    exp.update(sm_count=128,
               k4_config=spec_k4_config(m, n_hat, k_hat, b, N, gg, 128) if emu else 0,
               k4_tune_bytes=spec_k4_tune_bytes(m, n_hat, k_hat, b, gg) if emu else 0)
    return status, exp


def spec_k4_tune_bytes(m, n_hat, k_hat, b, g):
    """Group 1's RA, RB and W planes, each 256-aligned; 0 above 256 MiB (DESIGN v1.5 §6)."""
    regs = [g * b * m * k_hat, g * b * n_hat * k_hat, g * b * m * n_hat]
    total = sum(params.align_up(r, 256) for r in regs)
    return 0 if total > 256 << 20 else total


# DESIGN v1.5 §4 "K4 configurations": (tile_m, tile_n, CTAs per SM) and the model constants
# (round_ns, mac_fs by concurrency q = 1..CTAs per SM), transcribed from the DESIGN table.
K4_TILES = [(128, 256, 1), (128, 128, 2), (64, 128, 2), (64, 64, 3)]
K4_COSTS = [(2716, (418,)), (2550, (453, 456)), (1424, (505, 612)), (1333, (944, 1088, 802))]
K4_MARGIN = 50   # permille
SAT = (1 << 127) - 1


def spec_k4_estimate(c, m, n_hat, k_hat, gb, sms):
    """The estimate in fs: the busiest SM's j tiles in full rounds of `occ` concurrent CTAs
    and one round of the rest; every intermediate clamped at 2^127 - 1 (as the C++ does)."""
    tm, tn, occ = K4_TILES[c]
    rnd_ns, mac_fs = K4_COSTS[c]
    tiles = min(min(-(-m // tm) * -(-n_hat // tn), SAT) * gb, SAT)
    j = -(-tiles // sms)
    tile_macs = min(tm * tn * k_hat, SAT)

    def round_fs(q):
        return min(min(rnd_ns * 1000000, SAT) + min(min(tile_macs * q, SAT) * mac_fs[q - 1], SAT),
                   SAT)

    full, rest = divmod(j, occ)
    est = min(full * round_fs(occ), SAT) if full else 0
    return min(est + round_fs(rest), SAT) if rest else est


def spec_k4_config(m, n_hat, k_hat, b, N, g, sms):
    """DESIGN v1.5 §4 "K4 configurations": the selection rule, written independently of
    src/host/k4_config.cpp."""
    sms = sms if sms > 0 else 128
    gb = min(g, N) * b
    est = [spec_k4_estimate(c, m, n_hat, k_hat, gb, sms) for c in range(4)]
    best, best_est = 0, est[0]
    for c in range(1, 4):
        if min(est[c] * 1000, SAT) < min(est[0] * (1000 - K4_MARGIN), SAT) and est[c] < best_est:
            best, best_est = c, est[c]
    return best


# ---------------------------------------------------------------------------------------
# Random descriptors

def dense(ext, order, sign):
    st, acc = [0] * len(ext), 1
    for d in order:
        st[d] = acc
        acc *= max(ext[d], 1)
    return [s * g for s, g in zip(st, sign)]


def random_descs(rng, legal_bias=0.85):
    nlab = rng.randint(0, 7)
    labs = list(range(30, 30 + nlab))
    ext = {l: rng.choice([0, 1, 2, 3, 5, 16, 17, 40, 300] if rng.random() > 0.85
                         else [1, 2, 3, 5, 16, 17, 40]) for l in labs}
    mem = {l: rng.choice(("AB", "AC", "BC", "ABC", "ABC", "A", "B", "C", "AAB", "ABB"))
           for l in labs}
    descs = []
    for X in "ABC":
        lab = []
        for l in labs:
            lab += [l] * mem[l].count(X)
        rng.shuffle(lab)
        e = [ext[l] for l in lab]
        order = list(range(len(e)))
        rng.shuffle(order)
        st = dense(e, order, [rng.choice((1, -1)) for _ in e])
        if rng.random() > legal_bias and st:
            st[rng.randrange(len(st))] = rng.choice((0, 1, 2))
        descs.append((e, st, lab))
    return descs


def random_opts(rng, N=None, g=None, vec=None):
    N = N or rng.randint(2, 20)
    g = rng.randint(0, N) if g is None else g
    return (N, g, rng.choice((RN, RN, TRUNC)), rng.choice(("EMULATE", "EMULATE", "DIRECT")),
            vec or rng.choice(("GENERATE", "RESIDENT")))


def c_opts(N, g, rnd, mode, vec):
    return (N, g, 0 if rnd == RN else 1, 0 if mode == "EMULATE" else 1,
            0 if vec == "GENERATE" else 1)


# ---------------------------------------------------------------------------------------
# Suite host

def check_case(direct, descs, opts, what):
    N, g, rnd, mode, vec = opts
    co = c_opts(*opts)
    kw = dict(N=N, g=g, rounding=rnd, mode=mode, vec_mode=vec)
    want_st, exp = spec_expect(descs, N, g, rnd, mode, vec)

    # Python wrapper == direct C, for every query; a version-1 caller gets the same v1 fields.
    st_c, low_c = direct.lowering(descs, co)
    st_1, low_1 = direct.lowering(descs, co, size=V1)
    st_2, low_2 = direct.lowering(descs, co, size=V2)
    st_3, low_3 = direct.lowering(descs, co, size=V3)
    st_4, low_4 = direct.lowering(descs, co, size=V4)
    st_5, low_5 = direct.lowering(descs, co, size=V5)
    if st_5 != st_c or any(low_5[f] != low_c[f] for f in low_5) or "k_align" in low_5:
        raise Fail(f"{what}: a version-5 imd_lowering_t differs from version 6")
    if st_1 != st_c or any(low_1[f] != low_c[f] for f in low_1):
        raise Fail(f"{what}: a version-1 imd_lowering_t differs from version 5")
    if st_2 != st_c or any(low_2[f] != low_c[f] for f in low_2):
        raise Fail(f"{what}: a version-2 imd_lowering_t differs from version 5")
    if st_3 != st_c or any(low_3[f] != low_c[f] for f in low_3):
        raise Fail(f"{what}: a version-3 imd_lowering_t differs from version 5")
    if st_4 != st_c or any(low_4[f] != low_c[f] for f in low_4):
        raise Fail(f"{what}: a version-4 imd_lowering_t differs from version 5")
    if low_c["packed"] != 0 or low_c["pack_mismatch"] != 0:
        raise Fail(f"{what}: imd_lowering reports packed operands")
    # v1.5: k4_tune changes no host query; sm_count is the rule's input.
    st_t, low_t = direct.lowering(descs, co[:5] + (1,))
    if st_t != st_c or low_t != low_c:
        raise Fail(f"{what}: imd_lowering differs with k4_tune = ON")
    lw = host.lowering(descs, **kw)
    if lw.status != STATUS[st_c]:
        raise Fail(f"{what}: wrapper status {lw.status}, C {STATUS[st_c]}")
    for f in LOWERING_FIELDS[1:]:
        got = getattr(lw, f)
        want = low_c[f]
        if f == "dispatch":
            want = host.DISPATCH[want]
        if f == "pack_mismatch":
            want = host.PACK_MISMATCH[want]
        if got != want and not (isinstance(got, bool) and got == bool(want)):
            raise Fail(f"{what}: wrapper {f} = {got}, C {want}")
    ws_st, ws = direct.workspace(descs, co)
    lay_st, lay_tot, lay = direct.layout(descs, co)
    pb_st, pdev, phost = direct.plan_bytes(descs, co)
    if ws_st == 0:
        if host.workspace_size(descs, **kw) != ws:
            raise Fail(f"{what}: wrapper workspace_size differs")
        L = host.workspace_layout(descs, **kw)
        if L.total != lay_tot or L.regions != {host.REGIONS[r]: lay[r] for r in range(8)
                                               if lay[r][1]}:
            raise Fail(f"{what}: wrapper workspace_layout differs")
    if pb_st == 0 and host.plan_bytes(descs, **kw) != (pdev, phost):
        raise Fail(f"{what}: wrapper plan_bytes differs")
    if not (ws_st == lay_st) or (ws_st == 0 and lay_tot != ws):
        raise Fail(f"{what}: imd_workspace_layout disagrees with imd_workspace_size")

    # C == spec.
    workspace_status = want_st if exp is None else "SUCCESS"
    if STATUS[ws_st] != workspace_status:
        raise Fail(f"{what}: workspace status {STATUS[ws_st]}, spec {workspace_status}")
    if STATUS[st_c] != want_st or STATUS[pb_st] != want_st:
        raise Fail(f"{what}: plan-time status {STATUS[st_c]} / {STATUS[pb_st]}, spec {want_st}")
    if exp is None:
        if low_c["lowered"] != 0:
            raise Fail(f"{what}: lowered = 1 for an illegal descriptor")
        return want_st, None
    if low_c["lowered"] != 1:
        raise Fail(f"{what}: lowered = 0")
    for f in ("m", "n", "k", "b", "n_hat", "k_hat", "kappa", "workspace_bytes", "num_moduli",
              "residue_group", "vec_entries", "num_groups", "k_chunks", "gemm_launches",
              "fused", "kernel_launches", "sm_count", "k4_config", "k4_tune_bytes",
              "fused_large", "k_align"):
        if low_c[f] != exp[f]:
            raise Fail(f"{what}: {f} = {low_c[f]}, spec {exp[f]}")
    if exp["dispatch"] == "EMULATE":   # the rule for other devices (sm_count is an input)
        for sms in (16, 46, 82, 132, 170):
            st_s, low_s = direct.lowering(descs, co, sm_count=sms)
            want = spec_k4_config(exp["m"], exp["n_hat"], exp["k_hat"], exp["b"], N,
                                  exp["residue_group"], sms)
            if st_s != st_c or low_s["sm_count"] != sms or low_s["k4_config"] != want:
                raise Fail(f"{what}: sm_count {sms}: k4_config {low_s['k4_config']}, spec {want}")
    if bool(low_c["swapped"]) != exp["swapped"] or host.DISPATCH[low_c["dispatch"]] != exp["dispatch"]:
        raise Fail(f"{what}: swap / dispatch differ from the spec")
    regs = [(host.REGIONS[r], o, o + nb) for r, (o, nb) in enumerate(lay) if nb]
    if regs != [tuple(x) for x in exp["regions"]]:
        raise Fail(f"{what}: regions {regs}, spec {exp['regions']}")
    if want_st == "SUCCESS":
        if pdev != exp["plan_device_bytes"] or low_c["plan_device_bytes"] != pdev:
            raise Fail(f"{what}: plan device bytes {pdev}, spec {exp['plan_device_bytes']}")
        if phost == 0 or low_c["plan_host_bytes"] != phost:
            raise Fail(f"{what}: plan host bytes {phost} / {low_c['plan_host_bytes']}")
    return want_st, exp


# ---------------------------------------------------------------------------------------
# Packed operands (DESIGN v1.7 §6b): host queries vs the oracle and params.py

def oracle_pack(p):
    """oracle.PackDesc of a host.PackSpec."""
    return oracle.PackDesc(oracle.Desc(list(p.tensor.extent), list(p.tensor.stride),
                                       list(p.tensor.label)), p.num_moduli, p.rounding,
                           p.order, list(p.contracted), list(p.batch),
                           None if p.order == "AUTO" else list(p.rows))


def spec_pack_status(p):
    try:
        return "SUCCESS", oracle.pack_lower(oracle_pack(p))
    except oracle.InvalidValue:
        return "INVALID_VALUE", None
    except oracle.NotSupported:
        return "NOT_SUPPORTED", None


def check_pack_layout(p, what):
    """host.pack_layout == the oracle (pack_lower) == params.pack_layout / pack_ws_bytes."""
    want_st, pv = spec_pack_status(p)
    try:
        lay = host.pack_layout(p)
    except host.ImmerdinkError as e:
        if want_st != e.status:
            raise Fail(f"{what}: imd_pack_layout {e.status}, oracle {want_st}") from e
        return want_st, None
    if lay.status != want_st:
        raise Fail(f"{what}: imd_pack_layout {lay.status}, oracle {want_st}")
    A = params.k_align_of(lay.k, HOOK)        # v1.13 §4b: the pack's resolved A and format
    fmt = params.pack_format_of(A)
    regs, total = params.pack_layout(lay.b, lay.r, lay.k, p.num_moduli, fmt)
    want_regs = {n: (o, nb) for n, o, nb in regs} or {n: (0, 0) for n in ("sig", "flag", "res")}
    if lay.regions != want_regs or lay.pack_bytes != total or \
            lay.pack_ws_bytes != params.pack_ws_bytes(lay.b, lay.r, lay.k) or \
            lay.k_hat != params.k_hat_of(lay.k, A) or lay.format != fmt:
        raise Fail(f"{what}: pack layout {lay} vs params {regs} {total}")
    if want_st == "SUCCESS":
        if (lay.b, lay.r, lay.k, lay.kappa) != (pv.b, pv.r, pv.k, pv.kappa) or \
                lay.key_digest != pv.digest:
            raise Fail(f"{what}: pack sizes / key digest differ from the oracle")
    return want_st, pv


def spec_packed_expect(descs, opts, packing):
    """(status, reason, roles, view) of a packed plan from the oracle (lower_packed, Plan)."""
    N, g, rnd, mode, vec = opts
    A, B, Cd = (oracle.Desc(*d) for d in descs)
    omode = oracle.EMULATE if mode == "EMULATE" else oracle.DIRECT
    try:
        v, info = oracle.lower_packed(A, B, Cd, omode,
                                      tuple(oracle_pack(p) if p else None for p in packing), N, rnd,
                                      build_vec=False)
    except oracle.InvalidValue:
        return "INVALID_VALUE", 0, None, None
    except oracle.NotSupported:
        return "NOT_SUPPORTED", 0, None, None
    except oracle.PackMismatch as e:
        return "PACK_MISMATCH", e.reason, None, None
    if mode == "EMULATE" and not v.empty_output() and v.k > 1:
        try:
            oracle.Plan(N, v.k, rnd, g)
        except oracle.NotSupported:
            return "NOT_SUPPORTED", 0, info["packed_roles"], v
    return "SUCCESS", 0, info["packed_roles"], v


def check_packed(direct, descs, opts, rng, what):
    """For the caller's A and B: imd_pack_desc_for == oracle.pack_desc_for, the layouts, and
    the packed queries (lowering, workspace size / layout) == oracle + params, wrapper == C,
    for desc_for packs, AUTO packs, both packed and mismatching packs."""
    N, g, rnd, mode, vec = opts
    kw = dict(N=N, g=g, rounding=rnd, mode=mode, vec_mode=vec)
    co = c_opts(*opts)
    A, B, Cd = (oracle.Desc(*d) for d in descs)
    omode = oracle.EMULATE if mode == "EMULATE" else oracle.DIRECT
    try:
        v0 = oracle.lower(A, B, Cd, omode, build_vec=False)
    except (oracle.InvalidValue, oracle.NotSupported):
        return {}
    seen = {}
    specs = []
    for x in (0, 1):
        p = host.pack_desc_for(descs, operand=x, **kw)
        od = oracle.pack_desc_for(A, B, Cd, omode, x, N, rnd)
        if (list(p.contracted), list(p.batch), list(p.rows)) != (od.contracted, od.batch, od.rows):
            raise Fail(f"{what}: imd_pack_desc_for({x}) {p} vs oracle {od.__dict__}")
        st, _ = check_pack_layout(p, f"{what} pack {x}")
        specs.append(p if st == "SUCCESS" else None)
        X = descs[x]
        xl = list(dict.fromkeys(X[2]))
        auto = host.PackSpec(host.Desc(*map(tuple, X)), N, rnd, "AUTO",
                             tuple(lab for lab in xl if lab not in descs[2][2]),
                             tuple(lab for lab in xl if lab in descs[2][2] and lab in descs[1 - x][2]))
        check_pack_layout(auto, f"{what} auto pack {x}")
        specs.append(auto if spec_pack_status(auto)[0] == "SUCCESS" else None)
    pA, aA, pB, aB = specs
    variants = [(pA, None), (None, pB), (pA, pB), (aA, None), (None, aB), (aA, aB)]
    if pA is not None:   # mismatching packs: another N, the other rounding
        variants.append((host.PackSpec(pA.tensor, N + 1 if N < 20 else N - 1, rnd, "GIVEN",
                                       pA.contracted, pA.batch, pA.rows), None))
        variants.append((host.PackSpec(pA.tensor, N, TRUNC if rnd == RN else RN, "GIVEN",
                                       pA.contracted, pA.batch, pA.rows), pB))
        Xn = host.Desc(pA.tensor.extent, tuple(-s for s in pA.tensor.stride), pA.tensor.label)
        variants.append((host.PackSpec(Xn, N, rnd, "GIVEN", pA.contracted, pA.batch, pA.rows),
                         None))     # another tensor (negated strides): STRIDES
    for packing in variants:
        if packing == (None, None) or any(p is None for p in packing) and \
                not any(p is not None for p in packing):
            continue
        if all(p is None for p in packing):
            continue
        want_st, reason, roles, v = spec_packed_expect(descs, opts, packing)
        st, low = direct.lowering(descs, co, packing=packing)
        lw = host.lowering(descs, packing=packing, **kw)
        if STATUS[st] != want_st or lw.status != want_st:
            raise Fail(f"{what}: packed plan status {STATUS[st]} / {lw.status}, oracle {want_st} "
                       f"(packing {packing})")
        seen[want_st if not reason else host.PACK_MISMATCH[reason]] = 1
        for f in LOWERING_FIELDS[1:]:
            got, want = getattr(lw, f), low[f]
            if f == "dispatch":
                want = host.DISPATCH[want]
            if f == "pack_mismatch":
                want = host.PACK_MISMATCH[want]
            if got != want and not (isinstance(got, bool) and got == bool(want)):
                raise Fail(f"{what}: packed wrapper {f} = {got}, C {want}")
        if low["pack_mismatch"] != reason:
            raise Fail(f"{what}: pack_mismatch {low['pack_mismatch']}, oracle {reason}")
        want_bits = (1 if packing[0] else 0) | (2 if packing[1] else 0)
        if want_st in ("SUCCESS", "NOT_SUPPORTED", "PACK_MISMATCH") and low["lowered"] and \
                low["packed"] != want_bits:
            raise Fail(f"{what}: packed bits {low['packed']}, want {want_bits}")
        ws_st, ws = direct.workspace_packed(descs, co, packing)
        lay_st, lay_tot, lay = direct.layout_packed(descs, co, packing)
        ws_want = want_st if want_st != "NOT_SUPPORTED" or v is None else "SUCCESS"
        if STATUS[ws_st] != ws_want or STATUS[lay_st] != ws_want:
            raise Fail(f"{what}: packed workspace status {STATUS[ws_st]} / {STATUS[lay_st]}, "
                       f"want {ws_want}")
        if v is None:
            continue
        pmode = params.EMULATE if mode == "EMULATE" else params.DIRECT
        pvec = params.VEC_GENERATE if vec == "GENERATE" else params.VEC_RESIDENT
        A_res = params.k_align_of(v.k, HOOK)
        want_ws = params.workspace_size(v.m, v.n, v.k, v.b, N, g, pmode, pvec, packed=roles,
                                        k_align=A_res)
        want_regs = params.region_offsets(v.m, v.n, v.k, v.b, N, g, pmode, pvec, packed=roles,
                                          k_align=A_res)
        if ws != want_ws or lay_tot != want_ws or low["workspace_bytes"] != want_ws or \
                host.workspace_size(descs, packing=packing, **kw) != want_ws:
            raise Fail(f"{what}: packed workspace {ws} / {lay_tot}, params {want_ws}")
        if [(host.REGIONS[r], o, o + nb) for r, (o, nb) in enumerate(lay) if nb] != \
                [tuple(x) for x in want_regs]:
            raise Fail(f"{what}: packed regions {lay}, params {want_regs}")
        # launches and fused paths of the packed plan (DESIGN §6b.5)
        emu = mode == "EMULATE" and not v.empty_output() and v.k > 1
        if low["lowered"] and low["k_align"] != (A_res if emu else 16):
            raise Fail(f"{what}: packed k_align {low['k_align']}, want {A_res if emu else 16}")
        n_hat, k_hat = params.align_up(v.n, 16), params.k_hat_of(v.k, A_res)
        pbits = (1 if "A" in roles else 0) | (2 if "B" in roles else 0)
        fused = emu and k_hat <= 1024 and pbits != 3
        fl = (3 & ~pbits) if emu and not fused else 0
        if emu:
            k0 = 0 if vec == "RESIDENT" else 1
            zr = 1 if n_hat > v.n else 0
            pa, pb = (0 if pbits & 1 else 1), (0 if pbits & 2 else 1 + zr)
            head = 1 if fused else k0 + (0 if pbits & 1 else 1) + (0 if pbits & 2 else 1)
            group1 = 0 if fused else (0 if fl & 1 else pa) + (0 if fl & 2 else pb)
            want_l = head + (low["num_groups"] - 1) * (pa + pb) + group1 + low["gemm_launches"] + 1
            if low["kernel_launches"] != want_l or low["fused"] != int(fused) or \
                    low["fused_large"] != fl:
                raise Fail(f"{what}: packed launches {low['kernel_launches']} fused "
                           f"{low['fused']} / {low['fused_large']}, spec {want_l} {fused} {fl}")
        if want_st == "SUCCESS":
            un_st, un = direct.lowering(descs, co)
            for f in ("m", "n", "k", "b", "swapped", "k4_config", "gemm_launches", "k4_tune_bytes"):
                if un[f] != low[f]:
                    raise Fail(f"{what}: packed {f} {low[f]} != unpacked {un[f]}")
            if low["plan_host_bytes"] <= 0:
                raise Fail(f"{what}: packed plan host bytes 0")
    return seen


def suite_host(args):
    direct = Direct(args.host_lib)
    host.load(args.host_lib)  # the wrapper's default library: the one under test
    os.environ["IMMERDINK_HOST_LIB"] = args.host_lib
    rng = random.Random(20261004)
    by = {}
    # Random descriptors and options; every 4th also with packed operands (DESIGN §6b).
    packed_seen = {}
    for t in range(args.cases):
        descs, opts = random_descs(rng), random_opts(rng)
        st, _ = check_case(direct, descs, opts, f"random[{t}]")
        by[st] = by.get(st, 0) + 1
        if t % 4 == 0:
            for key in check_packed(direct, descs, opts, rng, f"random[{t}] packed"):
                packed_seen[key] = packed_seen.get(key, 0) + 1
    for need in ("SUCCESS", "MODULI", "ROUNDING", "CLASSES", "ORDER", "STRIDES"):
        if need not in packed_seen:
            raise Fail(f"packed host queries: no {need} case among {packed_seen}")
    # Every g = 1..N and every vec_mode on legal descriptors.
    n_sweep = 0
    for t in range(args.cases // 20):
        descs = random_descs(rng, legal_bias=1.0)
        N = rng.randint(2, 20)
        rnd, mode = rng.choice((RN, TRUNC)), rng.choice(("EMULATE", "DIRECT"))
        for g in range(1, N + 1):
            for vec in ("GENERATE", "RESIDENT"):
                check_case(direct, descs, (N, g, rnd, mode, vec), f"sweep[{t}] g={g} {vec}")
                n_sweep += 1
    # The kappa >= T boundary (RN): N = 2 (T = 180), k = 179^2 + 1 .. 180^2.
    for k in (179 * 179, 179 * 179 + 1, 180 * 180 - 1, 180 * 180, 180 * 180 + 1):
        descs = [([3, k], [k, 1], [1, 3]), ([k, 5], [5, 1], [3, 2]), ([3, 5], [5, 1], [1, 2])]
        for rnd in (RN, TRUNC):
            st, _ = check_case(direct, descs, (2, 0, rnd, "EMULATE", "GENERATE"), f"kappa k={k}")
            want = "NOT_SUPPORTED" if rnd == RN and math.isqrt(k - 1) + 1 >= 180 else "SUCCESS"
            if st != want:
                raise Fail(f"kappa k={k} {rnd}: {st}, expected {want}")
    # Multi-chunk K, batch split, n-hat split, large shapes (closed-form K4 counts).
    big = [(1, 2, 262200, 1), (3, 17, 131057, 2), (1, 1, 393168, 1), (2, 3, 5, 70000),
           (3, 2, 131056, 2), (1, 3, 131009, 1), (2, 1, 262112, 1),
           (1, 2**24 + 16, 2, 1), (5, 7, 300000, 40000)]
    for (m, n, k, b) in big:
        descs = [([m, k, b], [k * b, b, 1], [1, 3, 4]), ([k, n, b], [n * b, b, 1], [3, 2, 4]),
                 ([m, n, b], [n * b, b, 1], [1, 2, 4])]
        for g in (1, 3, 20):
            check_case(direct, descs, (20, g, TRUNC, "EMULATE", "GENERATE"), f"big {m},{n},{k},{b}")
    # v1.13 §4b K_c' = 131008: the lowering's K-chunks and K4 launches at the boundary k (one
    # more chunk exactly for k_hat in (j K_c', j 131056]), vs params and vs literals.
    want_chunks = {131008: 1, 131009: 2, 131050: 2, 131056: 2, 131057: 2, 131072: 2,
                   196608: 2, 262016: 2, 262017: 3, 262112: 3, 262113: 3, 393024: 3, 393025: 4}
    for k, chunks in want_chunks.items():
        descs = [([3, k], [k, 1], [1, 3]), ([k, 5], [5, 1], [3, 2]), ([3, 5], [5, 1], [1, 2])]
        for g in (1, 4, 9):
            st, _ = check_case(direct, descs, (9, g, TRUNC, "EMULATE", "GENERATE"), f"K_c' k={k} g={g}")
            lw = host.lowering(descs, N=9, g=g, rounding=TRUNC)
            ngroups = -(-9 // g)
            if st != "SUCCESS" or lw.k_chunks != chunks or lw.k_align != params.k_align_of(k) or \
                    lw.k_chunks != -(-params.k_hat_of(k, lw.k_align) // params.K_C_PRIME) or \
                    lw.gemm_launches != chunks * ngroups:
                raise Fail(f"K_c' boundary k={k} g={g}: {st}, k_chunks {lw.k_chunks} (want {chunks}), "
                           f"gemm_launches {lw.gemm_launches} (want {chunks * ngroups})")
    # The einsum front end == explicit descriptors; g_tradeoff == params for every g.
    spec, ext = "kjbc,kiac->abij", dict(a=24, b=20, c=16, i=12, j=10, k=8)
    descs = host.contraction(spec, ext)
    cc = [(list(d.extent), list(d.stride), list(d.label)) for d in descs]
    if cc[0] != ([8, 10, 20, 16], [3200, 320, 16, 1], [ord(x) for x in "kjbc"]):
        raise Fail(f"contraction(): A = {cc[0]}")
    lw = host.lowering(spec, ext, N=14)
    if (lw.m, lw.n, lw.k, lw.b) != (24 * 12, 20 * 10, 16 * 8, 1) and \
            (lw.n, lw.m, lw.k, lw.b) != (24 * 12, 20 * 10, 16 * 8, 1):
        raise Fail(f"einsum lowering {lw}")
    for row in host.g_tradeoff(spec, ext, N=14):
        want = params.workspace_size(lw.m, lw.n, lw.k, lw.b, 14, row["g"])
        if row["workspace"] != want:
            raise Fail(f"g_tradeoff g={row['g']}: {row['workspace']}, params {want}")
    a_cm = host.Desc((8, 10, 20, 16), (1, 8, 80, 1600), descs[0].label)
    if host.workspace_size(spec, ext, strides=(a_cm.stride, None, None), N=9) != \
            host.workspace_size((a_cm, descs[1], descs[2]), N=9):
        raise Fail("einsum strides")
    try:
        host.workspace_size(spec, dict(ext, b=-1), N=9)
        raise Fail("negative extent accepted")
    except host.ImmerdinkError as e:
        if e.status != "INVALID_VALUE":
            raise Fail(f"negative extent: {e.status}") from e
    n_kal = suite_k_align(direct, rng, max(args.cases // 10, 40))
    if args.verbose:
        print(f"  ok  {args.cases} random cases {by}, {n_sweep} g/vec sweep cases, kappa "
              f"boundary, {len(big)} K4-split shapes, {len(want_chunks)} K_c' boundary k, einsum front end, g_tradeoff; packed "
              f"queries {packed_seen}; k alignment (v1.13 §4b): {n_kal}")


# ---------------------------------------------------------------------------------------
# DESIGN v1.13 §4b stage 2: the per-plan K alignment in the host queries

K_ALIGN_BOUNDARY = (1, 16, 17, 64, 65, 95, 96, 97, 112, 113, 128, 129, 144, 145, 1023, 1024,
                    1025, 1040, 4080, 131008, 131009, 131072, 196608, 196624, 262017, 392961,
                    393008, 393009)


def gemm_descs(m, n, k, b=1):
    """C[i,j,l] = A[i,k,l] B[k,j,l], dense, last mode fastest (l fastest: no role swap)."""
    return [([m, k, b], [k * b, b, 1], [1, 3, 4]), ([k, n, b], [n * b, b, 1], [3, 2, 4]),
            ([m, n, b], [n * b, b, 1], [1, 2, 4])]


def suite_k_align(direct, rng, n_random):
    """Every host query under AUTO and with the hook at 16 and 64 vs params / oracle at the same
    policy: random cases and packed queries, the boundary k list, the k_align of every
    dispatch, the add-key merge (393008), the k4_tune cap, a k = 96 pack against a k = 97 plan
    (EXTENTS), the v5 / v6 struct mirror, and accuracy / kappa independent of A."""
    counts = {}
    co = lambda o: c_opts(*o)  # noqa: E731
    if (host._Lowering.k_align.offset, host._Lowering.reserved6.offset,
            C.sizeof(host._Lowering)) != (184, 188, V6):
        raise Fail("host.py imd_lowering_t v6 mirror")
    if (host.PACK_FORMAT_A16, host.PACK_FORMAT_A64, host.PACK_FORMAT) != (1, 2, 2) or \
            params.PACK_FORMAT != 2 or oracle.PACK_FORMAT != 2:
        raise Fail("pack format constants")
    try:
        host.test_set_k_align(32)
        raise Fail("hook accepted 32")
    except ValueError:
        pass
    try:
        for hook in (16, 64, "AUTO"):
            set_k_align(hook)
            for t in range(n_random):
                descs, opts = random_descs(rng, legal_bias=1.0), random_opts(rng)
                st, _ = check_case(direct, descs, opts, f"hook {hook} random[{t}]")
                counts[st] = counts.get(st, 0) + 1
                if t % 3 == 0:
                    for key in check_packed(direct, descs, opts, rng, f"hook {hook} random[{t}] packed"):
                        counts["packed " + key] = counts.get("packed " + key, 0) + 1
            for k in K_ALIGN_BOUNDARY:
                for (m, n, b, g) in ((3, 5, 1, 1), (17, 33, 2, 4)):
                    if k > 4096 and (m, n, b) != (3, 5, 1):
                        continue
                    descs = gemm_descs(m, n, k, b)
                    for opts in ((9, g, TRUNC, "EMULATE", "GENERATE"),
                                 (9, g, TRUNC, "DIRECT", "RESIDENT")):
                        st, exp = check_case(direct, descs, opts, f"hook {hook} k={k} {opts[3]}")
                        want = plan_k_align(k, "EMULATE" if opts[3] == "EMULATE" and k > 1 else "DIRECT")
                        if st != "SUCCESS" or exp["k_align"] != want or \
                                host.lowering(descs, N=9, g=g, rounding=TRUNC,
                                              mode=opts[3]).k_align != want:
                            raise Fail(f"hook {hook} k={k}: {st}, k_align {exp and exp['k_align']}")
                        counts["boundary"] = counts.get("boundary", 0) + 1
            # K-empty, SCALE and NOOP: k_align 16 whatever the hook
            for descs, disp in (([([3], [1], [1]), ([5], [1], [2]), ([3, 5], [5, 1], [1, 2])], "DIRECT"),
                                (gemm_descs(3, 5, 0), "SCALE"), (gemm_descs(0, 5, 200), "NOOP")):
                st, exp = check_case(direct, descs, (9, 0, RN, "EMULATE", "GENERATE"),
                                     f"hook {hook} {disp}")
                lw = host.lowering(descs, N=9)
                if lw.dispatch != disp or lw.k_align != 16 or exp["k_align"] != 16:
                    raise Fail(f"hook {hook} {disp}: k_align {lw.k_align}")
            # accuracy and kappa use the logical k: identical across A
            acc = host.Accuracy(metric="NORMWISE", target=1e-12)
            for k in (97, 1000, 4080):
                p = host.predict(gemm_descs(64, 48, k), N=14, accuracy=acc)
                r = host.rank(gemm_descs(64, 48, k), N=[8, 14], accuracy=acc)
                lw = host.lowering(gemm_descs(64, 48, k), N=14)
                key = (p.err_bound, p.err_typical, p.err_status, lw.kappa,
                       tuple((c.num_moduli, c.err_bound, c.err_typical, c.feasible) for c in r))
                prev = counts.setdefault(("acc", k), key)
                if prev != key:
                    raise Fail(f"hook {hook} k={k}: accuracy / kappa differ across A: {key} vs {prev}")
    finally:
        set_k_align("AUTO")
    # add-key merge (DESIGN §4b.3): k = 393008 has one add-mod shape key fewer at A = 64;
    # 393009 and 262016 do not differ. Plan host bytes per add key: kK4AddEntryBytes = 280.
    for k, diff in ((393008, 280), (392961, 280), (393009, 0), (392960, 0), (262016, 0)):
        hb = {}
        for hook in (16, 64):
            set_k_align(hook)
            hb[hook] = host.plan_bytes(gemm_descs(3, 5, k), N=9, rounding=TRUNC)[1]
            if hook == 64 and params.k4_add_chunk_keys(params.k_hat_of(k, 16)) - \
                    params.k4_add_chunk_keys(params.k_hat_of(k, 64)) != diff // 280:
                raise Fail(f"params add keys at k={k}")
        set_k_align("AUTO")
        if hb[16] - hb[64] != diff:
            raise Fail(f"k={k}: plan host bytes A16 {hb[16]} - A64 {hb[64]} != {diff}")
    # the k4_tune cap: post-swap (m, n, k, b, g) = (2000000, 1 or 2 (n_hat 16), 97, 1, 1):
    # 256001792 B at A = 16, over 256 MiB (0) at A = 64 (j fastest in C: no swap)
    tdescs = [([2000000, 97], [97, 1], [1, 3]), ([97, 2], [2, 1], [3, 2]),
              ([2000000, 2], [2, 1], [1, 2])]
    tune = {}
    for hook in (16, 64, "AUTO"):
        set_k_align(hook)
        lw = host.lowering(tdescs, N=2, g=1, rounding=TRUNC)
        tune[hook] = lw.k4_tune_bytes
        if (lw.m, lw.n_hat) != (2000000, 16) or \
                lw.k4_tune_bytes != params.k4_tune_bytes(lw.m, lw.n, lw.k_hat, lw.b, 1):
            raise Fail(f"k4_tune cap view {lw}")
    set_k_align("AUTO")
    if tune != {16: 256001792, 64: 0, "AUTO": 0}:
        raise Fail(f"k4_tune cap: {tune}")
    # a pack of k = 96 (format 1) offered to a plan of k = 97 (format 2): EXTENTS, not FORMAT
    for kp, kq in ((96, 97), (97, 96)):
        X = host.Desc((3, kp, 1), (kp, 1, 1), (1, 3, 4))
        pk = host.PackSpec(X, 9, RN, "AUTO", (3,), (4,))
        if host.pack_layout(pk).format != (2 if kp > 96 else 1):
            raise Fail(f"pack format at k={kp}")
        lw = host.lowering(gemm_descs(3, 5, kq), N=9, packing=(pk, None))
        st, low = direct.lowering(gemm_descs(3, 5, kq), co((9, 0, RN, "EMULATE", "GENERATE")),
                                  packing=(pk, None))
        if lw.status != "PACK_MISMATCH" or lw.pack_mismatch != "EXTENTS" or \
                low["pack_mismatch"] != 5:
            raise Fail(f"k={kp} pack, k={kq} plan: {lw.status} {lw.pack_mismatch}")
    counts["acc"] = len([k for k in counts if isinstance(k, tuple)])
    return {k: v for k, v in counts.items() if not isinstance(k, tuple)}


# ---------------------------------------------------------------------------------------
# Suite perf (DESIGN v1.8 §6c): the performance model, CPU only

L32 = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]  # §2: CRT limbs per N
MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191,
          181, 179, 173]  # DESIGN §2
K_C_PRIME = params.K_C_PRIME  # §3.5 131008 (v1.13)


def spec_k4_features(c, m, n_hat, k_hat, b, N, g, sms, k_chunks, l2, tail_mod=64):
    """K4 features (rounds, MACs at q = 1..3 concurrent CTAs) over the groups and K-chunks,
    written from DESIGN §6c.1 independently of src/host/perf.cpp."""
    tm, tn, occ = K4_TILES[c]
    x = [0.0] * 6
    groups = [g] * (N // g) + ([N % g] if N % g else [])
    last = k_hat - (k_chunks - 1) * K_C_PRIME
    chunks = [K_C_PRIME] * (k_chunks - 1) + [last]
    for gs in groups:
        for ci, kc in enumerate(chunks):
            tiles = -(-m // tm) * -(-n_hat // tn) * gs * b
            j = -(-tiles // sms)
            full, rest = divmod(j, occ)
            # v1.13 §4b.6: a chunk's MMA work is ceil(kc)_64 (CUTLASS runs the residue tile as a
            # full zero-filled 64-wide tile), the same for A = 16 and A = 64
            mma = -(-kc // 64) * 64
            x[0] += full + (1 if rest else 0)
            x[occ] += full * occ * tm * tn * mma
            if rest:
                x[rest] += rest * tm * tn * mma
            # format 1: a partial 64-wide K tile; format 2 (v1.10): the chunk's residue rows
            # start mid-sector (row stride k_hat, chunk offset ci K_c'): the MACs again
            slow = kc % 64 if tail_mod == 64 else (k_hat % 32 or (ci * K_C_PRIME) % 32)
            if slow:
                x[5] += (full * occ + rest) * tm * tn * mma
    for gs in groups:  # the DRAM term: a group's operands and W planes beyond L2
        x[4] += max(0, gs * b * (m + n_hat) * k_hat + gs * b * m * n_hat * (2 * k_chunks - 1)
                    - l2)
    return x


def spec_launch_blocks(work, per=256):
    return min(max(-(-work // per), 1), 1 << 20)


def spec_chain(gx, gy, units, u, resident, sms):
    """The busiest SM's per-thread chain of one launch (DESIGN v1.10 §6c.1 "Waves and
    occupancy"): ceil(B / (S r)) waves x ceil(units / grid_x) grid-stride steps x u."""
    if units <= 0 or gx <= 0:
        return 0
    return -(-(gx * gy) // (sms * resident)) * -(-units // gx) * u


def spec_parts(tiles, gsize, sms):
    return 1 if tiles >= 2 * sms else min(gsize, -(-2 * sms // tiles))


def spec_kf_chain(rows, k_hat, g, sms, r16=2, r4=4, elem=False):
    """KF's busiest-SM chain (DESIGN §6c.1), for operands read along k-bar (radix 1): tiles of
    4096 values (E = 16, r16 blocks per SM) or, below 16 blocks per SM even with all moduli
    split, 1024 (E = 4, r4 per SM); moduli parts when tiles < 2 SMs. elem (v1.17, kf.elem):
    the per-element chain, E steps per tile instead of E ceil(g / parts)."""
    tk = 16
    while tk < k_hat:
        tk *= 2
    tiles = sum(-(-r // (4096 // tk)) for r in rows)
    small = tiles * g < 16 * sms
    if small:
        tiles = sum(-(-r // max(1024 // tk, 1)) for r in rows)
    parts = spec_parts(tiles, g, sms)
    return spec_chain(spec_launch_blocks(tiles, 1), parts, tiles,
                      (4 if small else 16) * (1 if elem else -(-g // parts)),
                      r4 if small else r16, sms)


def spec_phases(lw, vec, along_rows, l2, beta=False, occ=None, sms=128, tail_mod=64, l_hit=0.0,
                planes=False):
    """[(name, launches, x)] of one imd_contract (DESIGN §6c.1 table) from the lowering
    summary and the operands' read direction (along rows: KL tiles of 512). occ: kernel name ->
    CTAs per SM (the calibration's occupancy table); the chains assume radix 1 (operands read
    along k-bar) and j-bar-fastest K5: callers compare them only for such cases. l_hit (v1.17):
    the calibration's warm.pack_l2_bytes (the PACK hit rule; both operands in-line); planes
    (v1.18, DESIGN §6c.8): S counts one group's W planes, g b m n_hat (2 k_chunks - 1)."""
    m, n, k, b, nh, kh = lw.m, lw.n, lw.k, lw.b, lw.n_hat, lw.k_hat
    N, g = lw.num_moduli, lw.residue_group
    V = lw.vec_entries
    res = vec == "RESIDENT"
    outb = 8 * b * m * n * (2 if beta else 1)
    occ = occ or {}
    r = lambda name: occ.get(name, 2)  # noqa: E731

    def grid_chain(name, outputs, u):
        return spec_chain(spec_launch_blocks(outputs), 1, -(-outputs // 256), u, r(name), sms)

    k0 = ("K0 vec", 1, [1, 8 * V, grid_chain("k0", V, 1), 0, 0, 0])
    ph = []
    if lw.dispatch in ("DIRECT", "SCALE"):
        if not res:
            ph.append(k0)
        if lw.dispatch == "SCALE":
            ph.append(("K5 beta-only", 1, [1, outb, grid_chain("beta", b * m * n, 1), 0, 0, 0]))
        else:
            ph.append(("DIRECT", 1, [1, 8 * b * (m * k + n * k) + outb, 2 * b * m * n * k,
                                     grid_chain("direct", b * m * n, k), 0, 0]))
        return ph + [("call", 0, [1, lw.kernel_launches, 0, 0, 0, 0])]
    rows, pad = [b * m, b * n], [b * m, b * nh]
    # v1.15 (DESIGN §6c.1 "The eps path — retired"): no eps fraction; the op_eps slots are 0
    # and the chains carry no (1 + f) factor (wide = 0 keeps the v1.10 transcription's shape)
    wide = 0.0
    if not lw.fused and not res:
        ph.append(k0)
    if lw.fused:
        byts = (0 if res else 8 * V) + sum(8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x]
                                           for x in (0, 1))
        ops = g * (rows[0] + rows[1]) * k
        # v1.17: slot 4 = the per-element chain (kf.elem)
        ph.append(("KF fused", 1, [1, byts, ops, rows[0] + rows[1],
                                   spec_kf_chain(rows, kh, g, sms, r("kf16"), r("kf4"), elem=True),
                                   (1 + wide) * spec_kf_chain(rows, kh, g, sms, r("kf16"),
                                                              r("kf4"))]))
    else:
        for x, name in ((0, "KL A"), (1, "KL B")):
            lg_tk = 9 if along_rows[x] else min(12, max(6, (kh - 1).bit_length()))
            tk = 1 << lg_tk
            th = 4096 // tk
            nc = -(-kh // tk)
            tiles = -(-rows[x] // th)
            values = 4 if nc > 2 else 8
            wave = spec_chain(min(tiles, 1 << 20), 1, tiles, nc,
                              r("kl4" if values == 4 else "kl8"), sms)
            ph.append((name, 1, [1, 8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x],
                                 g * rows[x] * k, 16 * rows[x] * max(0, kh - 2 * tk),
                                 wide * g * rows[x] * k, wave]))
    G = lw.num_groups
    sizes = [g] * (N // g) + ([N % g] if N % g else [])
    # v1.17: the PACK hit rule: the group working set (both operands, one group's residues)
    S = sum(8 * rows[y] * k + g * pad[y] * kh for y in (0, 1))
    if planes:
        S += g * b * m * nh * (2 * lw.k_chunks - 1)
    hit = 0 < l_hit and S <= l_hit
    for x, name in ((0, "K2 pack A"), (1, "K3 pack B")):
        if G > 1:
            per = 2 if (x == 1 and nh > n) else 1
            chain = wave = 0
            for gs in sizes[1:]:  # group 1 is KF's / KL's
                tiles = -(-rows[x] // 64) * -(-kh // 64)
                parts = spec_parts(tiles, gs, sms)
                chain += spec_chain(spec_launch_blocks(tiles, 1), parts, tiles,
                                    16 * -(-gs // parts), r("pack"), sms)
                wave += spec_chain(spec_launch_blocks(tiles, 1), parts, tiles, 1, r("pack"), sms)
            byts = (G - 1) * 8 * rows[x] * k + (N - g) * pad[x] * kh
            # v1.17: slot 3 = the bytes that are not L2-hit re-reads (group 1 is KF's / KL's,
            # so every pack read is a re-read)
            ph.append((name, (G - 1) * per, [(G - 1) * per, byts, (N - g) * rows[x] * k,
                                             byts - ((G - 1) * 8 * rows[x] * k if hit else 0),
                                             (1 + wide) * chain, wave]))
    ph.append(("K4 gemm", lw.gemm_launches,
               spec_k4_features(lw.k4_config, m, nh, kh, b, N, g, sms, lw.k_chunks, l2,
                                tail_mod)))
    mn = b * m * n
    e = 1 if mn <= 1024 * sms else 2 if mn <= 8 * 2048 * sms else 4
    work = b * m * -(-n // e)
    k5c = spec_chain(spec_launch_blocks(work), 1, -(-work // 256), e * N * L32[N],
                     r(f"k5_n{N}_e{e}"), sms)
    ph.append(("K5 crt", 1, [1, N * b * m * nh + 3 * b * (m + n) + outb, N * L32[N] * b * m * n,
                             k5c, max(0, N * b * m * nh - l2), 0]))
    return ph + [("call", 0, [1, lw.kernel_launches, 0, 0, 0, 0])]


# The slot of each phase's v1.10 chain feature (compared only when spec_phases' assumptions
# hold: radix 1, j-bar-fastest K5).
CHAIN_SLOT = {"K0 vec": 2, "KF fused": 5, "KL A": 5, "KL B": 5, "K2 pack A": 4, "K3 pack B": 4,
              "K5 crt": 3, "DIRECT": 3, "K5 beta-only": 2}


def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


# DESIGN v1.10 §6c.1: the feature slots whose time scales with the SM clock (an independent
# transcription of "ops, chains, L2 traffic, K4's round model; not DRAM bytes, launches, call").
SPEC_CLOCK_BOUND = {
    # v1.17: not kf.elem (slot 4), pack.dram (slot 3); kl slot 4 is unused (retired op_eps)
    "k0": (2,), "kf": (2, 3, 5), "kl": (2, 3), "pack": (2, 4), "k5": (2, 3),  # not pack.wave
    "direct": (2, 3), "beta": (2,), "call": ()}


def synthetic_v2(cal, names, rng):
    """A format-2 calibration: the shipped parameters, random chain parameters, a power model
    (cap 450 W), a cache term and bands."""
    params = dict(cal.params)
    for key in sorted(host.FORMAT2_PARAMS):  # sorted: independent of the string hash seed
        params[key] = rng.choice((0.0, rng.random() * 0.05))
    extra = {f"kernel.{n}": (v[0] + rng.randint(0, 2), 64, 4096, v[3])
             for n, v in cal.kernels.items()}
    extra.update({"power.clock_eff_mhz": 2800.0, "power.clock_min_mhz": 1400.0,
                  "power.cap_w": 450.0, "power.idle_w": 60.0, "power.tc_pj": 0.7,
                  "power.alu_pj": 5.0, "power.byte_pj": 20.0, "power.gamma": 2.0,
                  "warm.l2_bytes": 3.0e7, "warm.l2byte": 3e-4, "rank.tie_ratio": 1.08,
                  "band.call_small": (0.85, 1.2), "band.call_mid": (0.9, 1.12),
                  "band.call_large": (0.95, 1.06)})
    for c in host.PERF_CLASSES:
        extra[f"band.{c}"] = (1.0, 1.0) if c == "call" else (0.8, 1.25)
    return host.Calibration.from_values(cal.meta, params, extra=extra), params, extra


def phase_ns(params, names, q, scale=1.0):
    """A phase's time (ns) from its features and the parameters, the clock-bound terms times
    scale (DESIGN §6c.1; K4: max of the round model and the DRAM term)."""
    if q.cls.startswith("k4_"):
        rounds = sum(params.get(f"{q.cls}.{n}", 0.0) * x for j, (n, x) in
                     enumerate(zip(names[q.cls], q.x)) if n and j != 4)
        return max(rounds * scale, params.get(f"{q.cls}.byte", 0.0) * q.x[4])
    return sum(params.get(f"{q.cls}.{n}", 0.0) * x * (scale if j in SPEC_CLOCK_BOUND[q.cls] else 1)
               for j, (n, x) in enumerate(zip(names[q.cls], q.x)) if n)


def suite_perf_v110(cal, names, rng):
    count = 0
    shapes = [("ik,jk->ij", dict(i=64, j=64, k=64)), ("ik,jk->ij", dict(i=300, j=200, k=1000)),
              ("ik,jk->ij", dict(i=2048, j=1000, k=3000)), ("ik,jk->ij", dict(i=1000, j=64, k=50000)),
              ("ik,jk->ij", dict(i=8192, j=8192, k=8192)), ("lik,ljk->lij", dict(l=6, i=100, j=60, k=1500)),
              ("i,j->ij", dict(i=300, j=200))]
    # (1) a format-1 file (the v6 shipped one) predicts exactly the v6 formulas: no chain terms,
    # no clock scaling, no cache term, zero-width bands, and is written back as format 1.
    v6_path = os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib")
    v6 = host.Calibration.load(v6_path)
    if v6.file_format != 1 or host.Calibration.parse(v6.format()).format() != v6.format() or \
            "format = 1" not in v6.format() or "kernel." in v6.format():
        raise Fail("format-1 round trip")
    for spec, ext in shapes:
        for N, g in ((8, 0), (16, 5)):
            for regime in ("DEFAULT", "SUSTAINED"):
                for cache in ("COLD", "WARM"):
                    p = host.predict(spec, ext, N=N, g=g, regime=regime, cache=cache, calib=v6)
                    if p.status != "SUCCESS":
                        continue
                    for q in p.phases:
                        if not close(q.time_s, 1e-9 * phase_ns(v6.params, names, q), 1e-12) or \
                                q.lo_s != q.time_s or q.hi_s != q.time_s:
                            raise Fail(f"format 1 is not v6: {spec} {ext} {q}")
                    want_regime = "BURST" if regime == "DEFAULT" else regime
                    if p.regime != want_regime or p.clock_mhz != 0 or p.power_w != 0 or \
                            p.time_lo_s != p.time_s or p.time_hi_s != p.time_s:
                        raise Fail(f"format 1 v7 fields: {p}")
                    count += 1
    # (2) format 2: round trip, parse errors
    syn, params, extra = synthetic_v2(cal, names, rng)
    text = syn.format()
    again = host.Calibration.parse(text)
    if again.file_format != 2 or again.format() != text or again.params != syn.params or \
            again.extra != syn.extra or "format = 2" not in text:
        raise Fail("format-2 round trip")
    lines = text.splitlines()
    bad = {
        "missing kernel line": "\n".join(l for l in lines if not l.startswith("kernel.kf4 ")),
        "band p10 > p90": text.replace("band.call_mid = 0.9 1.12", "band.call_mid = 1.2 1.1"),
        "resident 0": "\n".join(("kernel.k0 = 0 64 4096 256" if l.startswith("kernel.k0 ") else l)
                                for l in lines),
        "format-2 key in format 1": v6.format() + "power.cap_w = 1\n",
        "chain in format 1": v6.format() + "k0.chain = 1\n",
        "missing chain in format 2": "\n".join(l for l in lines if not l.startswith("k5.chain")),
        "three numbers in a band": text.replace("band.call_mid = 0.9 1.12", "band.call_mid = 0.9 1 2"),
    }
    for why, t in bad.items():
        try:
            host.Calibration.parse(t)
            raise Fail(f"accepted a calibration with {why}")
        except ValueError:
            pass
    # (3) clock regimes, (4) cache states, (5) bands
    for spec, ext in shapes:
        for N, g in ((8, 0), (14, 4), (16, 0)):
            kw = dict(N=N, g=g, calib=syn)
            pb = host.predict(spec, ext, regime="BURST", cache="COLD", **kw)
            if pb.status != "SUCCESS" or not pb.phases:
                continue
            if pb.clock_mhz != 2800.0:
                raise Fail(f"BURST clock {pb.clock_mhz}")
            pr = host.predict(spec, ext, clock_mhz=2800.0, cache="COLD", **kw)
            if not close(pr.time_s, pb.time_s):
                raise Fail("clock_mhz = f_ref differs from BURST")
            p2 = host.predict(spec, ext, clock_mhz=1400.0, cache="COLD", **kw)
            for q, q2 in zip(pb.phases, p2.phases):  # clock-bound terms double
                if not close(q2.time_s, 1e-9 * phase_ns(params, names, q, 2.0), 1e-9) or \
                        not close(q.time_s, 1e-9 * phase_ns(params, names, q), 1e-9):
                    raise Fail(f"clock scaling {spec} {ext} {q.name}: {q2.time_s} vs "
                               f"{1e-9 * phase_ns(params, names, q, 2.0)}")
            if p2.time_s < pb.time_s * (1 - 1e-12):
                raise Fail("a lower clock predicts a shorter time")
            p3 = host.predict(spec, ext, clock_mhz=5600.0, cache="COLD", **kw)  # clamped: BURST
            if p3.clock_mhz != 2800.0 or not close(p3.time_s, pb.time_s):
                raise Fail(f"clock above the effective clock is not BURST: {p3.clock_mhz}")
            ps = host.predict(spec, ext, regime="SUSTAINED", cache="COLD", **kw)
            pd = host.predict(spec, ext, cache="COLD", **kw)
            if pd.regime != "SUSTAINED" or not close(pd.time_s, ps.time_s):
                raise Fail("DEFAULT is not SUSTAINED with a power model")
            if ps.time_s < pb.time_s * (1 - 1e-12) or ps.clock_mhz > 2800.0:
                raise Fail("SUSTAINED faster than BURST")
            if pb.power_w <= 450.0:
                if ps.clock_mhz != 2800.0 or not close(ps.time_s, pb.time_s):
                    raise Fail("SUSTAINED below the cap differs from BURST")
            elif ps.clock_mhz > 1400.0:
                # v1.17: the bisection stops at 0.05 MHz, so the solved clock brackets the cap
                # (a 1e-9 equality held only by chance for flat P(f); with the random v1.17
                # terms it does not)
                lo, hi = (host.predict(spec, ext, clock_mhz=ps.clock_mhz + d, cache="COLD",
                                       **kw).power_w for d in (-0.05, 0.05))
                if not (lo <= 450.0 <= hi and abs(ps.power_w - 450.0) <= hi - lo):
                    raise Fail(f"SUSTAINED power {ps.power_w} W at {ps.clock_mhz} MHz "
                               f"({lo} .. {hi} W)")
            cold, rep, warm = (host.predict(spec, ext, regime="BURST", cache=c, **kw).time_s
                               for c in ("COLD", "REPEAT", "WARM"))
            if not (cold >= rep * (1 - 1e-12) and close(rep, warm)):  # WARM = REPEAT
                raise Fail(f"cache order {cold} {rep} {warm}")
            lo, hi = extra[("band.call_small", "band.call_mid", "band.call_large")[
                0 if pb.time_s < 1e-4 else 1 if pb.time_s < 1e-3 else 2]]
            if not (close(pb.time_lo_s, lo * pb.time_s) and close(pb.time_hi_s, hi * pb.time_s)):
                raise Fail("call band")
            for q in pb.phases:
                blo, bhi = extra[f"band.{q.cls}"]
                if not (close(q.lo_s, blo * q.time_s) and close(q.hi_s, bhi * q.time_s)):
                    raise Fail("phase band")
            count += 1
    # (6) rank: ties by footprint, raw_rank = the v6 time order, ORDER_TIME = that order
    for spec, ext in (("ik,jk->ij", dict(i=2048, j=1000, k=3000)),
                      ("kjbc,kiac->abij", dict(a=24, b=20, c=16, i=12, j=10, k=8))):
        kw = dict(N=[8, 14], vec_modes=("GENERATE", "RESIDENT"), packs=("NONE", "A", "B", "AB"),
                  reuse=3, calib=syn)
        by_time = host.rank(spec, ext, **kw)  # the default order is TIME
        if by_time != host.rank(spec, ext, order="TIME", **kw):
            raise Fail("rank default order is not TIME")
        banded = host.rank(spec, ext, order="TIES_BY_FOOTPRINT", **kw)
        if [c.tie for c in sorted(banded, key=lambda c: c.raw_rank)] != [c.tie for c in by_time]:
            raise Fail("tie differs between the orders")
        if [c.raw_rank for c in by_time] != list(range(len(by_time))):
            raise Fail("raw_rank of the time order")
        if sorted(c.raw_rank for c in banded) != list(range(len(banded))):
            raise Fail("raw_rank is not a permutation")
        if [(c.num_moduli, c.residue_group, c.vec_mode, c.packed) for c in
                sorted(banded, key=lambda c: c.raw_rank)] != \
                [(c.num_moduli, c.residue_group, c.vec_mode, c.packed) for c in by_time]:
            raise Fail("raw_rank does not reproduce the time order")
        best = by_time[0]
        ties = [c for c in banded if c.tie]
        want = [c for c in by_time if c.feasible and c.status == "SUCCESS" and
                c.time_s <= best.time_s * 1.08]
        if len(ties) != len(want) or banded[:len(ties)] != ties or not best.tie:
            raise Fail("rank ties")
        for a_, b_ in zip(ties, ties[1:]):
            if a_.total_bytes > b_.total_bytes:
                raise Fail("ties not ordered by footprint")
        if banded[len(ties):] != [c for c in by_time if not c.tie]:
            raise Fail("rank: the non-tied candidates keep the time order")
        for c in banded:
            if c.status == "SUCCESS" and not (c.time_lo_s <= c.time_s <= c.time_hi_s):
                raise Fail("candidate band")
        count += 1
    # (7) occupancy table names
    kn = host.kernel_names()
    if len(kn) != 90 or kn[0] != "k0" or kn[14] != "k5_n2_e1" or kn[-1] != "k5b_n20" or \
            len(set(kn)) != len(kn):
        raise Fail(f"kernel names {kn[:3]} ... {kn[-3:]}")
    return count


def _sized_calib(fn, size, *args):
    """Call a C calibration function (host API 10 _ex form) with a struct whose struct_size is
    `size` (a caller compiled against an older API): the bytes beyond it are poisoned and must
    stay untouched. Returns (status, raw, error)."""
    h = host.load()
    raw = host._PerfCalib()
    C.memset(C.byref(raw), 0x5A, C.sizeof(raw))
    raw.struct_size = size
    e = host._PerfCalibError()
    e.struct_size = C.sizeof(e)
    st = getattr(h.lib, fn)(*args, C.byref(raw), C.byref(e))
    tail = bytes(raw)[size:]
    if tail != b"\x5a" * len(tail):
        raise Fail(f"{fn} wrote beyond struct_size {size}")
    return st, raw, e


V10_CALIB_SIZE = 3672  # sizeof(imd_perf_calib_t) of host API 7..10 (v9 and v10: the same)


def suite_perf_v115(cal):
    """DESIGN v1.15 §6c.7 (host API 11): the optional power.byte_f_pj, the folded v7..v10 view,
    identity at byte_f = 0, monotonicity of the SUSTAINED clock."""
    count = 0
    shapes = [("ik,jk->ij", dict(i=512, j=512, k=512)), ("ik,jk->ij", dict(i=1500, j=96, k=40000)),
              ("ik,jk->ij", dict(i=4096, j=4096, k=4096)), ("ik,jk->ij", dict(i=8192, j=8192, k=8192)),
              ("ik,jk->ij", dict(i=12288, j=384, k=1040)),
              ("acQ,bdQ->acbd", dict(a=64, b=64, c=64, d=64, Q=2048))]
    regimes = (("BURST", 0.0), ("SUSTAINED", 0.0), ("DEFAULT", 0.0), ("DEFAULT", 2200.0))

    def preds(c):
        return [host.predict(sp, ex, N=N, regime=rg, clock_mhz=ck, cache=cc, calib=c)
                for sp, ex in shapes for N in (8, 14, 16) for rg, ck in regimes
                for cc in ("COLD", "REPEAT")]

    # The shipped calibration of the combined refit (NOTES-B2 §10) carries power.byte_f_pj:
    # it formats with the key and parses back bit-exactly; the checks below use its byte_f = 0
    # variant (the calibration without the key) as their base.
    # v1.17: and without the PACK hit rule (warm.pack_l2_bytes = 0), which the v7..v10 view
    # below cannot hold either (suite_perf_v117 checks that key).
    ship = cal
    if ship.power["power.byte_f_pj"] != 0.0 or ship.warm["warm.pack_l2_bytes"] != 0.0 or \
            any(ship.extra[kk] != 0.0 for kk in V118_KEYS):
        ts = ship.format()
        if bytes(host.Calibration.parse(ts)._raw) != bytes(ship._raw):
            raise Fail("the shipped calibration does not round trip")
        if ship.power["power.byte_f_pj"] != 0.0 and "power.byte_f_pj = " not in ts:
            raise Fail("the shipped power.byte_f_pj is not written")
        e0 = dict(ship.extra)
        e0["power.byte_f_pj"] = 0.0
        e0["warm.pack_l2_bytes"] = 0.0
        e0.update({kk: 0.0 for kk in V118_KEYS})  # v1.18: the v7..v10 view has none of them
        cal = host.Calibration.from_values(ship.meta, ship.params, extra=e0)
        count += 1
    # (1) without byte_f the text has no key; an explicit 0 is the same calibration, bit for
    # bit, formats to the same text and predicts identically (every regime, cache, shape).
    t0 = cal.format()
    if "power.byte_f_pj" in t0 or cal.power["power.byte_f_pj"] != 0.0:
        raise Fail("a calibration with byte_f = 0 writes power.byte_f_pj")
    c0 = host.Calibration.parse(t0.replace("power.gamma = ", "power.byte_f_pj = 0\npower.gamma = "))
    if bytes(c0._raw) != bytes(cal._raw) or c0.format() != t0:
        raise Fail("power.byte_f_pj = 0 is not the calibration without the key")
    p_ship = preds(cal) if ship is not cal else preds(None)
    if preds(c0) != p_ship or preds(cal) != p_ship:
        raise Fail("byte_f = 0 does not predict bit-identically to the shipped calibration")
    count += len(p_ship)
    # (2) round trip with the key (format -> parse bit-exact; written after power.byte_pj)
    ex = dict(cal.extra)
    ex.update({"power.byte_pj": 100.0, "power.byte_f_pj": 235.5})
    c2 = host.Calibration.from_values(cal.meta, cal.params, extra=ex)
    t2 = c2.format()
    if "power.byte_pj = 100\npower.byte_f_pj = 235.5\n" not in t2:
        raise Fail("power.byte_f_pj is not written after power.byte_pj")
    back = host.Calibration.parse(t2)
    if bytes(back._raw) != bytes(c2._raw) or back.format() != t2 or \
            back.power["power.byte_f_pj"] != 235.5:
        raise Fail("power.byte_f_pj round trip")
    # (3) parse errors: format 1, negative, repeated, not a number
    v6 = host.Calibration.load(os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib"))
    bad = {"format-2 key in format 1": (v6.format() + "power.byte_f_pj = 1\n", "FORMAT2_KEY"),
           "negative": (t0 + "power.byte_f_pj = -1\n", "VALUE"),
           "not a number": (t0 + "power.byte_f_pj = x\n", "VALUE"),
           "repeated": (t2 + "power.byte_f_pj = 1\n", "REPEATED_KEY")}
    for why, (t, reason) in bad.items():
        try:
            host.Calibration.parse(t)
            raise Fail(f"accepted power.byte_f_pj: {why}")
        except host.CalibrationError as e:
            if e.reason != reason or e.key != "power.byte_f_pj":
                raise Fail(f"power.byte_f_pj {why}: {e.reason} {e.key}")
        count += 1
    # (4) the folded view of a v7..v10-sized struct: energy_byte_pj = byte + byte_f, the rest
    # the same bytes; parse, load and default (byte_f = 0: the full calibration's prefix).
    h = host.load()
    data = t2.encode()
    want = host._PerfCalib.from_buffer_copy(bytes(c2._raw))
    want.energy_byte_pj = 100.0 + 235.5
    want.struct_size = V10_CALIB_SIZE
    st, v10, _ = _sized_calib("imd_perf_calib_parse_ex", V10_CALIB_SIZE, data, len(data))
    if st != 0 or bytes(v10)[:V10_CALIB_SIZE] != bytes(want)[:V10_CALIB_SIZE]:
        raise Fail("folded v10 view (parse)")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.calib")
        with open(path, "w") as f:
            f.write(t2)
        st, v10l, _ = _sized_calib("imd_perf_calib_load_ex", V10_CALIB_SIZE, os.fsencode(path))
        if st != 0 or bytes(v10l)[:V10_CALIB_SIZE] != bytes(want)[:V10_CALIB_SIZE]:
            raise Fail("folded v10 view (load)")
    st, v10d, _ = _sized_calib("imd_perf_calib_default_ex", V10_CALIB_SIZE)
    full = host._PerfCalib.from_buffer_copy(bytes(ship._raw))
    full.struct_size = V10_CALIB_SIZE
    full.energy_byte_pj = ship.power["power.byte_pj"] + ship.power["power.byte_f_pj"]
    if st != 0 or bytes(v10d)[:V10_CALIB_SIZE] != bytes(full)[:V10_CALIB_SIZE]:
        raise Fail("v10 view of the shipped calibration differs from its folded view")
    # its predictions = the unscaled form with e_byte = byte + byte_f (a file without the key);
    # read with byte_f = 0 although the poisoned tail is beyond its size
    ex_fold = dict(ex)
    ex_fold.update({"power.byte_pj": 100.0 + 235.5, "power.byte_f_pj": 0.0})
    # v1.17 review (§6c.3): a v7..v11-sized struct reads L_hit = warm.l2_bytes when pack.dram > 0
    if cal.params["pack.dram"] > 0:
        ex_fold["warm.pack_l2_bytes"] = ex_fold["warm.l2_bytes"]
    c_fold = host.Calibration.from_values(cal.meta, cal.params, extra=ex_fold)
    p_fold = preds(c_fold)
    if preds(host.Calibration(v10, h)) != p_fold:
        raise Fail("the v10 view does not predict as the folded (unscaled) calibration")
    if preds(c2) == p_fold:
        raise Fail("byte_f > 0 predicts as the folded calibration (no capped shape?)")
    # imd_perf_calib_format of a v10-sized struct: read with byte_f = 0 (no key written)
    if host.Calibration(v10, h).format() != c_fold.format():
        raise Fail("format of a v10-sized struct")
    count += len(p_fold)
    # (5) monotonicity of the SUSTAINED clock (gamma = 1; f_s < f_ref: the scaled term is the
    # smaller one there): moving byte energy into byte_f never lowers the clock, raising byte_f
    # never raises it, and the 2026-10-05 refit pair lowers it for tensor-core-heavy calls.
    def mk(tc, byte, byte_f):
        e = dict(cal.extra)
        e.update({"power.tc_pj": tc, "power.byte_pj": byte, "power.byte_f_pj": byte_f,
                  "power.gamma": 1.0})
        return host.Calibration.from_values(cal.meta, cal.params, extra=e)

    tc0, b0 = cal.power["power.tc_pj"], cal.power["power.byte_pj"]
    fref, cmin = cal.power["power.clock_eff_mhz"], cal.power["power.clock_min_mhz"]
    for sp, ext in shapes:
        for N in (8, 14, 16):
            def sus(c):
                return host.predict(sp, ext, N=N, regime="SUSTAINED", cache="COLD", calib=c)
            moved = [sus(mk(tc0, b0 * (1 - a), b0 * a)) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
            for x, y in zip(moved, moved[1:]):
                if y.clock_mhz < x.clock_mhz or y.time_s > x.time_s * (1 + 1e-12):
                    raise Fail(f"moving e_byte to e_byte^f lowered the clock: {sp} {ext} N={N}")
            raised = [sus(mk(tc0, b0, bf)) for bf in (0.0, 50.0, 150.0, 400.0, 1000.0)]
            for x, y in zip(raised, raised[1:]):
                if y.clock_mhz > x.clock_mhz or y.time_s < x.time_s * (1 - 1e-12):
                    raise Fail(f"raising e_byte^f raised the clock: {sp} {ext} N={N}")
            # a capped solve brackets the cap within the bisection's 0.05 MHz
            for bf in (150.0, 1000.0):
                c = mk(tc0, b0, bf)
                q = sus(c)
                if cmin < q.clock_mhz < fref:
                    lo, hi = (host.predict(sp, ext, N=N, clock_mhz=q.clock_mhz + d, cache="COLD",
                                           calib=c).power_w for d in (-0.05, 0.05))
                    if not lo <= 450.0 <= hi:
                        raise Fail(f"SUSTAINED clock {q.clock_mhz} MHz does not meet the cap "
                                   f"({lo} .. {hi} W) with byte_f {bf}")
            count += 10
    for ext in (dict(i=8192, j=8192, k=8192), dict(i=16384, j=16384, k=16384)):
        old = host.predict("ik,jk->ij", ext, N=14, regime="SUSTAINED", calib=mk(0.6498, 335.0, 0.0))
        new = host.predict("ik,jk->ij", ext, N=14, regime="SUSTAINED", calib=mk(0.691, 0.0, 296.0))
        if not (old.clock_mhz < fref and new.clock_mhz < old.clock_mhz and new.time_s > old.time_s):
            raise Fail(f"refit pair: clock {old.clock_mhz} -> {new.clock_mhz} at {ext}")
        count += 1
    return count


V11_CALIB_SIZE = 3680  # sizeof(imd_perf_calib_t) of host API 11


def suite_perf_v117(cal):
    """DESIGN v1.17 §6c.1 / §6c.2 (host API 12): kf.elem, pack.dram and warm.pack_l2_bytes
    optional (absent = 0, written only when non-zero, format-2 keys); the new terms are
    additive (with their parameters 0 the features change but no time does); pack.dram with
    L_hit 0 charges every byte; a v11-sized struct reads L_hit as warm.l2_bytes when
    pack.dram > 0, else 0 (review 2026-10-06: exact for a v1.17-style file; v1.18 review: for
    the v1.18 shipped file exactly the documented degraded view)."""
    count = 0
    shapes = [("ik,jk->ij", dict(i=1024, j=1024, k=1024)), ("ik,jk->ij", dict(i=1472, j=1472, k=1472)),
              ("ik,jk->ij", dict(i=1792, j=1792, k=1792)), ("ik,jk->ij", dict(i=1024, j=96, k=4096)),
              ("ik,jk->ij", dict(i=4096, j=96, k=4096)), ("kjbc,kiac->abij",
                                                           dict(a=48, b=48, c=48, i=12, j=12, k=12)),
              ("acQ,bdQ->acbd", dict(a=64, b=64, c=64, d=64, Q=2048))]

    def preds(c, gs=(0, 2, 4)):
        return [host.predict(sp, ex, N=N, g=g, regime=rg, cache=cc, calib=c)
                for sp, ex in shapes for N in (8, 14, 20) for g in gs
                for rg in ("BURST", "SUSTAINED") for cc in ("COLD", "REPEAT")]

    keys = ("kf.elem", "pack.dram", "warm.pack_l2_bytes")
    # (1) the base: the shipped calibration without the three (an API-11 calibration)
    params = dict(cal.params, **{"kf.elem": 0.0, "pack.dram": 0.0})
    extra = dict(cal.extra, **{"warm.pack_l2_bytes": 0.0})
    extra.update({kk: 0.0 for kk in V118_KEYS})  # v1.18: the v11 view has none of them
    base = host.Calibration.from_values(cal.meta, params, extra=extra)
    tb = base.format()
    if any(f"\n{kk} = " in tb for kk in keys):
        raise Fail("v1.17 keys written although 0")
    stripped = "\n".join(l for l in tb.split("\n") if l.split(" = ")[0] not in keys)
    if bytes(host.Calibration.parse(stripped)._raw) != bytes(base._raw):
        raise Fail("v1.17 keys: absent is not 0")
    p0 = preds(base)
    # (2) additive: L_hit > 0 (and its features) without pack.dram, kf.elem features without
    # kf.elem: the same times, bit for bit
    lh = host.Calibration.from_values(cal.meta, params,
                                      extra=dict(extra, **{"warm.pack_l2_bytes": 6.04e7}))
    p1 = preds(lh)
    if [(q.time_s, q.time_lo_s, q.time_hi_s, q.clock_mhz) for q in p1] != \
            [(q.time_s, q.time_lo_s, q.time_hi_s, q.clock_mhz) for q in p0]:
        raise Fail("L_hit without pack.dram changes a prediction")
    hits = sum(1 for p in p1 for q in p.phases if q.cls == "pack" and q.x[3] < q.x[1])
    elems = sum(1 for p in p1 for q in p.phases if q.cls == "kf" and q.x[4] > 0)
    if not hits or not elems:
        raise Fail(f"v1.17 features not exercised: {hits} hit pack phases, {elems} KF elem")
    count += len(p0)
    # (3) pack.dram with L_hit 0: x3 = x1 in every pack phase; with L_hit, x3 = x1 - re-reads
    # exactly when S <= L_hit, and a hit never makes a call slower than no hit
    pd = dict(params, **{"pack.dram": 4e-4, "kf.elem": 300.0})
    c_nohit = host.Calibration.from_values(cal.meta, pd, extra=extra)
    c_hit = host.Calibration.from_values(cal.meta, pd, extra=dict(extra, **{"warm.pack_l2_bytes": 6.04e7}))
    for a, b in zip(preds(c_nohit), preds(c_hit)):
        if a.status != "SUCCESS":
            continue
        for q in a.phases:
            if q.cls == "pack" and q.x[3] != q.x[1]:
                raise Fail("pack.dram feature with L_hit 0 is not every byte")
        if b.time_s > a.time_s * (1 + 1e-12):
            raise Fail("an L2 hit made a call slower")
        count += 1
    # (4) round trip with the keys, written only when non-zero, format-1 / negative / repeated
    t2 = c_hit.format()
    for kk in keys:
        if f"\n{kk} = " not in t2:
            raise Fail(f"{kk} not written")
    if bytes(host.Calibration.parse(t2)._raw) != bytes(c_hit._raw) or \
            host.Calibration.parse(t2).format() != t2:
        raise Fail("v1.17 keys round trip")
    v6 = host.Calibration.load(os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib"))
    for kk in keys:
        bad = {"format-2 key in format 1": (v6.format() + f"{kk} = 1\n", "FORMAT2_KEY"),
               "negative": (tb + f"{kk} = -1\n", "VALUE"),
               "repeated": (t2 + f"{kk} = 1\n", "REPEATED_KEY")}
        for why, (t, reason) in bad.items():
            try:
                host.Calibration.parse(t)
                raise Fail(f"accepted {kk}: {why}")
            except host.CalibrationError as e:
                if e.reason != reason or e.key != kk:
                    raise Fail(f"{kk} {why}: {e.reason} {e.key}")
            count += 1
    # (5) a v11-sized struct: parse / default leave the field alone, and a struct of that size
    # predicts and formats with L_hit = warm.l2_bytes when pack.dram > 0, else 0 (DESIGN §6c.3
    # [review 2026-10-06])
    data = t2.encode()
    st, v11, _ = _sized_calib("imd_perf_calib_parse_ex", V11_CALIB_SIZE, data, len(data))
    want = host._PerfCalib.from_buffer_copy(bytes(c_hit._raw))
    want.struct_size = V11_CALIB_SIZE
    if st != 0 or bytes(v11)[:V11_CALIB_SIZE] != bytes(want)[:V11_CALIB_SIZE]:
        raise Fail("v11 view (parse)")
    lw = c_hit.warm["warm.l2_bytes"]
    if not (lw > 0 and lw != 6.04e7):
        raise Fail(f"v11 view: the test needs warm.l2_bytes > 0 and != L_hit ({lw})")
    c_lw = host.Calibration.from_values(cal.meta, pd, extra=dict(extra, **{"warm.pack_l2_bytes": lw}))
    v11c = host.Calibration(v11, host.load())
    if preds(v11c) != preds(c_lw) or preds(v11c) == preds(c_nohit):
        raise Fail("a v11-sized struct with pack.dram > 0 does not predict with L_hit = warm.l2_bytes")
    if v11c.format() != c_lw.format():
        raise Fail("a v11-sized struct with pack.dram > 0 does not format with L_hit = warm.l2_bytes")
    count += 2
    # pack.dram = 0 (every file before v1.17): L_hit 0, the text gains no key
    data0 = tb.encode()
    st, v11b, _ = _sized_calib("imd_perf_calib_parse_ex", V11_CALIB_SIZE, data0, len(data0))
    v11bc = host.Calibration(v11b, host.load())
    if st != 0 or preds(v11bc) != p0 or v11bc.format() != tb:
        raise Fail("a v11-sized struct with pack.dram = 0 is not the calibration without the keys")
    count += 1
    # v1.18 review: the fallback is exact for a v1.17-style file (L_hit = warm.l2_bytes, no
    # v1.18 keys): its v11 view predicts as the full calibration
    if cal.params["pack.dram"] > 0:
        c117 = host.Calibration.from_values(cal.meta, cal.params, extra=dict(cal.extra, **dict(
            {kk: 0.0 for kk in V118_KEYS}, **{"warm.pack_l2_bytes": cal.warm["warm.l2_bytes"]})))
        t117 = c117.format().encode()
        st, v11e, _ = _sized_calib("imd_perf_calib_parse_ex", V11_CALIB_SIZE, t117, len(t117))
        if st != 0 or preds(host.Calibration(v11e, host.load())) != preds(c117):
            raise Fail("a v1.17-style file's v11 view does not predict as the file")
        count += 1
    # the shipped calibration's v11 default view: exactly the documented degraded view (the
    # v1.18 fields 0, L_hit = warm.l2_bytes when pack.dram > 0); since v1.18 not NULL's model
    st, v11d, _ = _sized_calib("imd_perf_calib_default_ex", V11_CALIB_SIZE)
    if st != 0:
        raise Fail("v11 view (default)")
    deg = host.Calibration.from_values(cal.meta, cal.params, extra=dict(cal.extra, **dict(
        {kk: 0.0 for kk in V118_KEYS},
        **{"warm.pack_l2_bytes": cal.warm["warm.l2_bytes"] if cal.params["pack.dram"] > 0 else 0.0})))
    if preds(host.Calibration(v11d, host.load())) != preds(deg):
        raise Fail("the shipped calibration's v11 view is not the documented degraded view")
    count += 1
    return count


V118_KEYS = ("power.clock_boost_mhz", "power.act_w", "power.act_gap_us", "power.dram_pj",
             "warm.call_ns", "warm.clean_byte", "warm.pack_l2_planes")
V12_CALIB_SIZE = 3688  # sizeof(imd_perf_calib_t) of host API 12


def suite_perf_v118(cal):
    """DESIGN v1.18 §6c.8 (host API 13): the seven optional keys (absent = 0 predicts
    bit-identically), their grammar, the cache terms (CLEAN credit on the call phase, the
    per-call REPEAT saving only when res > 0), the S rule with K4's planes, the boost clock
    (BURST at f_top with the f_ref time), the busy power and DRAM energy (monotone P(f), the
    capped solve brackets the cap) and the v12-sized view (the fields read as 0)."""
    count = 0
    shapes = [("ik,jk->ij", dict(i=256, j=256, k=256)), ("ik,jk->ij", dict(i=1024, j=1024, k=1024)),
              ("ik,jk->ij", dict(i=2048, j=2048, k=2048)), ("ik,jk->ij", dict(i=4096, j=4096, k=512)),
              ("ik,jk->ij", dict(i=1500, j=96, k=40000)), ("ik,jk->ij", dict(i=8192, j=8192, k=8192))]
    caches = ("COLD", "REPEAT", "WARM", "CLEAN")
    regs = (("BURST", 0.0), ("SUSTAINED", 0.0), ("DEFAULT", 2200.0))

    def preds(c, gs=(0, 4)):
        return [host.predict(sp, ex, N=N, g=g, regime=rg, clock_mhz=ck, cache=cc, calib=c)
                for sp, ex in shapes for N in (8, 14) for g in gs for rg, ck in regs
                for cc in caches]

    # (1) the base: the shipped calibration without the seven keys (an API-12 calibration)
    extra0 = dict(cal.extra, **{kk: 0.0 for kk in V118_KEYS})
    base = host.Calibration.from_values(cal.meta, cal.params, extra=extra0)
    tb = base.format()
    if any(f"\n{kk} = " in tb for kk in V118_KEYS):
        raise Fail("v1.18 keys written although 0")
    stripped = "\n".join(l for l in tb.split("\n") if l.split(" = ")[0] not in V118_KEYS)
    explicit = tb + "".join(f"{kk} = 0\n" for kk in V118_KEYS)
    if bytes(host.Calibration.parse(stripped)._raw) != bytes(base._raw) or \
            bytes(host.Calibration.parse(explicit)._raw) != bytes(base._raw):
        raise Fail("v1.18 keys: absent / explicit 0 is not the calibration without them")
    p0 = preds(base)
    if preds(host.Calibration.parse(explicit)) != p0:
        raise Fail("v1.18 keys explicitly 0 do not predict bit-identically")
    for a, b in zip(p0[0::4], p0[3::4]):  # COLD vs CLEAN with omega = 0
        if repr(a) != repr(b).replace("cache='CLEAN'", "cache='COLD'"):
            raise Fail("CLEAN without warm.clean_byte differs from COLD")
    count += len(p0)
    # (2) grammar: optional format-2 keys, >= 0, planes 0 / 1, boost 0 or >= clock_eff
    vals = {"power.clock_boost_mhz": 2790.0, "power.act_w": 160.0, "power.act_gap_us": 1.0,
            "power.dram_pj": 80.0, "warm.call_ns": 1400.0, "warm.clean_byte": 1.9e-4,
            "warm.pack_l2_planes": 1.0}
    full = host.Calibration.from_values(cal.meta, cal.params, extra=dict(extra0, **vals))
    tf = full.format()
    for kk in V118_KEYS:
        if f"\n{kk} = " not in tf:
            raise Fail(f"{kk} not written")
    back = host.Calibration.parse(tf)
    if bytes(back._raw) != bytes(full._raw) or back.format() != tf:
        raise Fail("v1.18 keys round trip")
    if any(full.extra[kk] != v for kk, v in vals.items()):
        raise Fail("v1.18 keys: Calibration.power / .warm")
    v6 = host.Calibration.load(os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib"))
    fe = full.power["power.clock_eff_mhz"]
    for kk in V118_KEYS:
        bad = {"format-2 key in format 1": (v6.format() + f"{kk} = 1\n", "FORMAT2_KEY", kk),
               "negative": (tb + f"{kk} = -1\n", "VALUE", kk),
               "repeated": (tf + f"{kk} = 1\n", "REPEATED_KEY", kk)}
        if kk == "warm.pack_l2_planes":
            bad["not 0 or 1"] = (tb + f"{kk} = 0.5\n", "VALUE", kk)
        if kk == "power.clock_boost_mhz":
            bad["below clock_eff"] = (tb + f"{kk} = {fe - 1}\n", "INCONSISTENT", kk)
        for why, (t, reason, key) in bad.items():
            try:
                host.Calibration.parse(t)
                raise Fail(f"accepted {kk}: {why}")
            except host.CalibrationError as e:
                if e.reason != reason or e.key != key:
                    raise Fail(f"{kk} {why}: {e.reason} {e.key}")
            count += 1
    # (3) the cache terms: REPEAT = COLD - (Delta res + c [res > 0]) (unclamped: small Delta, c),
    # CLEAN = COLD - omega min(F - I, L_w) on the call phase only
    lw, dl, cc, om = 6.0e7, 1e-4, 300.0, 2e-4
    cw = host.Calibration.from_values(cal.meta, cal.params, extra=dict(
        extra0, **{"warm.l2_bytes": lw, "warm.l2byte": dl, "warm.call_ns": cc,
                   "warm.clean_byte": om}))
    n_res = [0, 0]
    for sp, ex in shapes:
        for N in (8, 14):
            pc, pr, pk = (host.predict(sp, ex, N=N, regime="BURST", cache=st, calib=cw)
                          for st in ("COLD", "REPEAT", "CLEAN"))
            if pc.status != "SUCCESS":
                continue
            res = pr.warm_bytes
            n_res[res > 0] += 1
            want = pc.time_s - 1e-9 * (dl * res + (cc if res > 0 else 0.0))
            if not close(pr.time_s, want, 1e-9) or (res == 0 and pr.time_s != pc.time_s):
                raise Fail(f"REPEAT cache term {sp} {ex} N={N}: {pr.time_s} vs {want}")
            out = 8 * ex["i"] * ex["j"]  # F - I = workspace + the output (no packed buffers)
            credit = 1e-9 * om * min(pc.workspace_bytes + out, lw)
            for a, b in zip(pc.phases, pk.phases):
                if a.name == "call":
                    if not close(b.time_s, a.time_s - credit, 1e-9):
                        raise Fail(f"CLEAN credit {sp} {ex} N={N}: {b.time_s} vs {a.time_s - credit}")
                elif a.time_s != b.time_s:
                    raise Fail(f"CLEAN changed phase {a.name}")
            if pk.warm_bytes != 0 or pk.cache != "CLEAN":
                raise Fail("CLEAN reports resident bytes")
            count += 1
    if not (n_res[0] and n_res[1]):
        raise Fail(f"cache terms not exercised on both sides: {n_res}")
    # (4) the S rule with planes: 4096^2 x 512 at g = 4 hits by the v1.17 proxy, misses with planes
    ext = dict(i=4096, j=4096, k=512)
    for pl, want_hit in ((0.0, True), (1.0, False)):
        c = host.Calibration.from_values(cal.meta, cal.params, extra=dict(
            extra0, **{"warm.pack_l2_bytes": 8.3e7, "warm.pack_l2_planes": pl}))
        p = host.predict("ik,jk->ij", ext, N=14, g=4, calib=c)
        hit = any(q.cls == "pack" and q.x[3] < q.x[1] for q in p.phases)
        if hit != want_hit:
            raise Fail(f"S rule with planes={pl}: hit {hit}")
        count += 1
    # (5) power: boost clock, busy power, DRAM energy
    pw = dict(extra0, **{"power.clock_eff_mhz": 2500.0, "power.clock_min_mhz": 1500.0,
                         "power.cap_w": 450.0, "power.idle_w": 60.0, "power.gamma": 2.5})
    c_nb = host.Calibration.from_values(cal.meta, cal.params, extra=pw)
    c_b = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{
        "power.clock_boost_mhz": 2790.0, "power.act_w": 160.0, "power.act_gap_us": 1.0,
        "power.dram_pj": 80.0}))
    c_b0 = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{
        "power.clock_boost_mhz": 2790.0}))
    n_cap = [0, 0]
    for sp, ex in shapes:
        for N in (8, 14):
            kw = dict(N=N, cache="COLD")
            bnb = host.predict(sp, ex, regime="BURST", calib=c_nb, **kw)
            if bnb.status != "SUCCESS":
                continue
            bb = host.predict(sp, ex, regime="BURST", calib=c_b, **kw)
            if bb.clock_mhz != 2790.0 or bb.time_s != bnb.time_s or bnb.clock_mhz != 2500.0:
                raise Fail(f"BURST with a boost: clock {bb.clock_mhz}, time {bb.time_s} vs {bnb.time_s}")
            hi = host.predict(sp, ex, clock_mhz=3500.0, calib=c_b, **kw)
            mid = host.predict(sp, ex, clock_mhz=2600.0, calib=c_b, **kw)
            if hi.clock_mhz != 2790.0 or hi.time_s != bb.time_s or mid.time_s != bb.time_s or \
                    mid.clock_mhz != 2600.0 or not mid.power_w <= hi.power_w:
                raise Fail("clock_mhz above f_ref: time not flat or not clamped to the boost")
            # P(f) monotone in f; the busy and DRAM terms only add power
            prev = None
            for f in (1500.0, 1900.0, 2300.0, 2500.0, 2650.0, 2790.0):
                q = host.predict(sp, ex, clock_mhz=f, calib=c_b, **kw)
                q0 = host.predict(sp, ex, clock_mhz=f, calib=c_b0, **kw)
                if prev is not None and q.power_w < prev * (1 - 1e-12):
                    raise Fail(f"P(f) not monotone: {sp} {ex} N={N} at {f} MHz")
                if q.power_w < q0.power_w * (1 - 1e-12) or \
                        q.power_w - q0.power_w > 160.0 * (f / 2500.0) ** 3.5 + 1e-9 + \
                        80.0 * 1e-12 * sum(ph.bytes for ph in q.phases) / q.time_s:
                    raise Fail(f"busy / DRAM power out of range at {f} MHz: {q.power_w} vs {q0.power_w}")
                prev = q.power_w
            ps = host.predict(sp, ex, regime="SUSTAINED", calib=c_b, **kw)
            if ps.time_s < bb.time_s * (1 - 1e-12) or ps.clock_mhz > 2790.0:
                raise Fail("SUSTAINED faster than BURST with a boost")
            if bb.power_w <= 450.0:
                n_cap[0] += 1
                if ps.clock_mhz != 2790.0 or ps.time_s != bb.time_s:
                    raise Fail("SUSTAINED below the cap is not BURST with a boost")
            elif ps.clock_mhz > 1500.0:
                n_cap[1] += 1
                lo_, hi_ = (host.predict(sp, ex, clock_mhz=ps.clock_mhz + d, calib=c_b, **kw).power_w
                            for d in (-0.05, 0.05))
                if not lo_ <= 450.0 <= hi_:
                    raise Fail(f"SUSTAINED with busy power: {ps.clock_mhz} MHz ({lo_} .. {hi_} W)")
            count += 1
    if not (n_cap[0] and n_cap[1]):
        raise Fail(f"boost / cap not exercised on both sides: {n_cap}")
    # a call whose footprint fits warm.l2_bytes pays no DRAM energy
    small = ("ik,jk->ij", dict(i=256, j=256, k=256))
    cd = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{
        "power.dram_pj": 500.0, "warm.l2_bytes": 6.0e7}))
    c0 = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{"warm.l2_bytes": 6.0e7}))
    if host.predict(*small, N=8, regime="BURST", calib=cd).power_w != \
            host.predict(*small, N=8, regime="BURST", calib=c0).power_w:
        raise Fail("DRAM energy charged to a call that fits the L2")
    big = ("ik,jk->ij", dict(i=8192, j=8192, k=8192))
    if not host.predict(*big, N=8, regime="BURST", calib=cd).power_w > \
            host.predict(*big, N=8, regime="BURST", calib=c0).power_w:
        raise Fail("DRAM energy not charged to a call beyond the L2")
    count += 2
    # (5b) v1.18 review: the power divides by the request's cache-state time (energy / t of the
    # state, so (P - idle) t is the same in every state at a given clock), and the SUSTAINED
    # solve of a REPEAT / CLEAN request brackets the cap with that state's power
    c_e = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{
        "warm.call_ns": 1200.0, "warm.clean_byte": 1.7e-4}))
    c_ec = host.Calibration.from_values(cal.meta, cal.params, extra=dict(pw, **{
        "warm.call_ns": 1200.0, "warm.clean_byte": 1.7e-4, "power.clock_boost_mhz": 2790.0,
        "power.act_w": 160.0, "power.act_gap_us": 1.0, "power.dram_pj": 80.0}))
    n_faster = 0
    for sp, ex in shapes:
        for N in (8, 14):
            pc = host.predict(sp, ex, N=N, clock_mhz=2400.0, cache="COLD", calib=c_e)
            if pc.status != "SUCCESS":
                continue
            for cc in ("REPEAT", "WARM", "CLEAN"):
                pq = host.predict(sp, ex, N=N, clock_mhz=2400.0, cache=cc, calib=c_e)
                ec, eq = (pc.power_w - 60.0) * pc.time_s, (pq.power_w - 60.0) * pq.time_s
                if abs(eq - ec) > 1e-9 * ec:
                    raise Fail(f"{cc} power not energy / its own time: {sp} {ex} N={N}")
                if pq.time_s < pc.time_s * (1 - 1e-9):
                    n_faster += 1
                    if not pq.power_w > pc.power_w:
                        raise Fail(f"a faster {cc} call does not draw more power")
                ps = host.predict(sp, ex, N=N, regime="SUSTAINED", cache=cc, calib=c_ec)
                if 1500.0 < ps.clock_mhz < 2790.0:
                    lo_, hi_ = (host.predict(sp, ex, N=N, clock_mhz=ps.clock_mhz + d, cache=cc,
                                             calib=c_ec).power_w for d in (-0.05, 0.05))
                    if not lo_ <= 450.0 <= hi_:
                        raise Fail(f"SUSTAINED {cc}: {ps.clock_mhz} MHz ({lo_} .. {hi_} W)")
                count += 1
    if not n_faster:
        raise Fail("no REPEAT / CLEAN call faster than COLD: the state-time check is vacuous")
    # (6) a v12-sized struct: parse leaves the tail alone; it predicts as the calibration without
    # the seven keys
    data = tf.encode()
    st, v12, _ = _sized_calib("imd_perf_calib_parse_ex", V12_CALIB_SIZE, data, len(data))
    want = host._PerfCalib.from_buffer_copy(bytes(full._raw))
    want.struct_size = V12_CALIB_SIZE
    if st != 0 or bytes(v12)[:V12_CALIB_SIZE] != bytes(want)[:V12_CALIB_SIZE]:
        raise Fail("v12 view (parse)")
    no118 = host.Calibration.from_values(cal.meta, cal.params, extra=extra0)
    if preds(host.Calibration(v12, host.load())) != preds(no118):
        raise Fail("a v12-sized struct does not predict as the calibration without the v1.18 keys")
    if host.Calibration(v12, host.load()).format() != no118.format():
        raise Fail("a v12-sized struct formats with v1.18 keys")
    count += 1
    return count


V119_KEY = "warm.kl_drop"


def suite_perf_v119(cal):
    """DESIGN v1.19 §6c.9 (host API 13, no struct growth): the optional key warm.kl_drop, the
    share of the REPEAT / WARM saving that KL reads do not get (stored in the first parameter
    slot past the classes). Absent or 0 predicts bit-identically; format 2 only, a number in
    [0, 1], written only when non-zero, round trip; KL phases keep (1 - d) of the saving, every
    other phase, COLD and CLEAN are unchanged; d = 1 puts a KL read at its COLD time."""
    count = 0
    shapes = [("ik,kj->ij", dict(i=1600, k=1600, j=100)),     # KL, narrow n, inputs resident
              ("abcd,cdij->abij", dict(a=40, b=40, c=40, d=40, i=10, j=10)),  # the Jacobi view
              ("ik,jk->ij", dict(i=1024, j=1024, k=1024)),   # KF, inputs resident
              ("ik,jk->ij", dict(i=256, j=256, k=256)),
              ("ik,jk->ij", dict(i=2048, j=2048, k=2048)),   # KL, nothing resident
              ("ik,jk->ij", dict(i=1500, j=96, k=40000))]
    caches = ("COLD", "REPEAT", "WARM", "CLEAN")

    def preds(c):
        return [host.predict(sp, ex, N=N, g=g, regime=rg, cache=cc, calib=c)
                for sp, ex in shapes for N in (4, 8, 14) for g in (0, 4)
                for rg in ("BURST", "SUSTAINED") for cc in caches]

    # (1) absent == explicit 0 == the calibration without the key, bit for bit
    extra0 = dict(cal.extra, **{V119_KEY: 0.0})
    base = host.Calibration.from_values(cal.meta, cal.params, extra=extra0)
    tb = base.format()
    if f"\n{V119_KEY} = " in tb:
        raise Fail("warm.kl_drop written although 0")
    explicit = host.Calibration.parse(tb + f"{V119_KEY} = 0\n")
    if bytes(explicit._raw) != bytes(base._raw) or explicit.format() != tb:
        raise Fail("warm.kl_drop: explicit 0 is not the calibration without it")
    p0 = preds(base)
    if preds(explicit) != p0:
        raise Fail("warm.kl_drop explicitly 0 does not predict bit-identically")
    count += len(p0) + 1
    # (2) grammar: optional, format 2 only, a number in [0, 1], not repeated
    d1 = 0.35
    full = host.Calibration.from_values(cal.meta, cal.params, extra=dict(extra0, **{V119_KEY: d1}))
    tf = full.format()
    if f"\n{V119_KEY} = " not in tf or full.warm[V119_KEY] != d1 or full.extra[V119_KEY] != d1:
        raise Fail("warm.kl_drop not written or not in Calibration.warm / .extra")
    back = host.Calibration.parse(tf)
    if bytes(back._raw) != bytes(full._raw) or back.format() != tf:
        raise Fail("warm.kl_drop round trip")
    v6 = host.Calibration.load(os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib"))
    for why, (t, reason) in {"format-2 key in format 1": (v6.format() + f"{V119_KEY} = 0.5\n",
                                                          "FORMAT2_KEY"),
                             "negative": (tb + f"{V119_KEY} = -0.1\n", "VALUE"),
                             "above 1": (tb + f"{V119_KEY} = 1.01\n", "VALUE"),
                             "not a number": (tb + f"{V119_KEY} = nan\n", "VALUE"),
                             "repeated": (tf + f"{V119_KEY} = 0.5\n", "REPEATED_KEY")}.items():
        try:
            host.Calibration.parse(t)
            raise Fail(f"accepted warm.kl_drop: {why}")
        except host.CalibrationError as e:
            if e.reason != reason or e.key != V119_KEY:
                raise Fail(f"warm.kl_drop {why}: {e.reason} {e.key}")
        count += 1
    for ok in (0.0, 1.0, 0.75):
        host.Calibration.parse(tb + f"{V119_KEY} = {ok}\n")
        count += 1
    # (3) the rule: with a small Delta nothing is clamped, so a KL phase's REPEAT time is
    # COLD - (1 - d) (COLD - REPEAT at d = 0); every other phase, COLD and CLEAN are unchanged
    lw, dl = 6.0e7, 1e-4
    cw = host.Calibration.from_values(cal.meta, cal.params, extra=dict(
        extra0, **{"warm.l2_bytes": lw, "warm.l2byte": dl}))
    n_seen = {"kl": 0, "kf": 0, "kl_none": 0}
    for d in (0.25, 0.8, 1.0):
        cd = host.Calibration.from_values(cal.meta, cal.params, extra=dict(
            extra0, **{"warm.l2_bytes": lw, "warm.l2byte": dl, V119_KEY: d}))
        for sp, ex in shapes:
            for N in (4, 8, 14):
                for g in (0, 4):
                    kw = dict(N=N, g=g, regime="BURST")
                    c0 = {st: host.predict(sp, ex, cache=st, calib=cw, **kw) for st in caches}
                    cx = {st: host.predict(sp, ex, cache=st, calib=cd, **kw) for st in caches}
                    if c0["COLD"].status != "SUCCESS":
                        continue
                    if repr(cx["COLD"]) != repr(c0["COLD"]) or repr(cx["CLEAN"]) != repr(c0["CLEAN"]):
                        raise Fail(f"warm.kl_drop changed COLD / CLEAN: {sp} {ex} N={N} g={g}")
                    if [q.time_s for q in cx["WARM"].phases] != [q.time_s for q in cx["REPEAT"].phases]:
                        raise Fail("WARM is not REPEAT with warm.kl_drop")
                    for a, b, r in zip(c0["COLD"].phases, c0["REPEAT"].phases, cx["REPEAT"].phases):
                        if a.cls == "kl":
                            want = a.time_s - (1 - d) * (a.time_s - b.time_s)
                            if not close(r.time_s, want, 1e-12):
                                raise Fail(f"KL phase {r.name}: {r.time_s} vs {want} (d={d})")
                            n_seen["kl" if a.time_s != b.time_s else "kl_none"] += 1
                            if d == 1.0 and r.time_s != a.time_s:
                                raise Fail("d = 1: a KL read keeps part of the saving")
                        else:
                            if r.time_s != b.time_s:
                                raise Fail(f"warm.kl_drop changed the non-KL phase {r.name}")
                            n_seen["kf"] += a.cls == "kf" and a.time_s != b.time_s
                    count += 1
    if not (n_seen["kl"] and n_seen["kf"] and n_seen["kl_none"]):
        raise Fail(f"warm.kl_drop: the rule is not exercised on all sides: {n_seen}")
    return count


def suite_calib_errors(cal, lib_path):
    """Host API 10 (DESIGN §6c.3): every calibration rejection as a CalibrationError with its
    line, reason, key and message; the shipped calibration loads (no error kept); the
    generation-time checks of bench/perf_fit.py (check_meta, fit_text, --check) on a notes
    value one byte over its field and on other malformed metadata."""
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import perf_fit
    n = 0
    text = cal.format()
    lines = text.split("\n")
    nl = len(lines)            # the line after the last ("...\n" ends with an empty item)
    ln = {l.split(" = ")[0]: i + 1 for i, l in enumerate(lines) if " = " in l}
    notes128 = "n" * 128

    def drop(key):
        return "\n".join(l for l in lines if not l.startswith(key + " = "))

    def subst(key, value):
        return "\n".join((f"{key} = {value}" if l.startswith(key + " = ") else l) for l in lines)

    cases = [  # (text, reason, line, key, message part)
        (subst("notes", notes128), "TOO_LONG", ln["notes"], "notes",
         "notes: value is 128 bytes, the field holds 127"),
        (subst("gpu", "g" * 64), "TOO_LONG", ln["gpu"], "gpu", "the field holds 63"),
        (subst("clocks", "§" * 64), "TOO_LONG", ln["clocks"], "clocks",
         "value is 128 bytes"),    # UTF-8 bytes, not characters
        (text + "bogus.key = 1\n", "UNKNOWN_KEY", nl, "bogus.key", "unknown key"),
        (text + "k5.byte = 1\n", "REPEATED_KEY", nl, "k5.byte",
         f"repeated key (first on line {ln['k5.byte']})"),
        (drop("k5.byte"), "MISSING_KEY", 0, "k5.byte", "missing key"),
        (drop("band.call_mid"), "MISSING_KEY", 0, "band.call_mid", "missing key"),
        (drop("gpu"), "MISSING_KEY", 0, "gpu", "missing key"),
        (subst("format", "3"), "VALUE", ln["format"], "format", "(want 1 or 2)"),
        (subst("kl.byte", "-1"), "VALUE", ln["kl.byte"], "kl.byte", "invalid value '-1'"),
        (subst("sm_count", "0"), "VALUE", ln["sm_count"], "sm_count", "[1, 1000000]"),
        (subst("band.call_mid", "1.2 1.1"), "VALUE", ln["band.call_mid"], "band.call_mid",
         "lo <= hi"),
        ("kl.byte\n" + text, "SYNTAX", 1, "", "'kl.byte' is not"),
        (subst("rank.tie_ratio", "2"), "INCONSISTENT", 0, "rank.tie_ratio", "[1, 1.25]"),
    ]
    if cal.power["power.cap_w"] > 0:
        cases.append((subst("power.clock_min_mhz", "1e9"), "INCONSISTENT", 0,
                      "power.clock_min_mhz", "power.cap_w > 0 needs"))
    if cal.file_format == 2:
        cases.append((subst("format", "1"), "FORMAT2_KEY", None, None,
                      "a format-2 key in a format-1 calibration"))
    for t, reason, line, key, msg in cases:
        try:
            host.Calibration.parse(t)
            raise Fail(f"accepted a calibration ({reason}, {key})")
        except host.CalibrationError as e:
            if (e.reason != reason or (line is not None and e.line != line) or
                    (key is not None and e.key != key) or msg not in e.message or
                    not isinstance(e, ValueError) or msg not in str(e)):
                raise Fail(f"{reason} {key}: got {e.reason} line {e.line} key {e.key!r}: "
                           f"{e.message!r}")
        n += 1
    # 127 bytes fit
    if host.Calibration.parse(subst("notes", "n" * 127)).meta["notes"] != "n" * 127:
        raise Fail("a 127-byte notes value")
    # load: a missing file, the committed over-long test vector (line 9)
    try:
        host.Calibration.load("/nonexistent/x.calib")
        raise Fail("loaded a missing file")
    except host.CalibrationError as e:
        if e.reason != "FILE" or e.line != 0 or "cannot open" not in e.message:
            raise Fail(f"missing file: {e}")
    vec = os.path.join(ROOT, "tests/data/perf_calib_notes_too_long.calib")
    try:
        host.Calibration.load(vec)
        raise Fail("loaded the over-long notes vector")
    except host.CalibrationError as e:
        if (e.reason, e.line, e.key) != ("TOO_LONG", 9, "notes"):
            raise Fail(f"over-long notes vector: {e}")
    # the shipped calibration parses: nothing kept, predict with calib=None works
    h = host.load(lib_path)
    if h.shipped_calib_error is not None:
        raise Fail(f"shipped calibration: {h.shipped_calib_error}")
    h.check_shipped_calib()
    n += 4
    # perf_fit: the generation-time checks
    good = dict(cal.meta)
    if perf_fit.check_meta(good):
        raise Fail(f"check_meta on the shipped metadata: {perf_fit.check_meta(good)}")
    bad_meta = {"notes": notes128, "gpu": "a\nb", "version": " v", "clocks": "§" * 64,
                "date": "2026-10-06 "}
    probs = perf_fit.check_meta(bad_meta)
    for want in ("notes: 128 bytes, the calibration field holds 127", "gpu: contains a line break",
                 "version: leading or trailing blank", "clocks: 128 bytes",
                 "date: leading or trailing blank"):
        if not any(want in p for p in probs):
            raise Fail(f"check_meta missed {want!r}: {probs}")
    cut = perf_fit.fit_text("§" * 100, "notes")      # 200 bytes -> 63 characters, 126 bytes
    if cut != "§" * 63 or perf_fit.fit_text("short", "notes") != "short":
        raise Fail(f"fit_text: {cut!r}")
    if perf_fit.check_meta({"notes": perf_fit.fit_text("x" * 500 + " ", "notes")}):
        raise Fail("fit_text output does not pass check_meta")
    with tempfile.TemporaryDirectory() as d:
        ok_path = os.path.join(d, "ok.calib")
        with open(ok_path, "w") as f:
            f.write(text)
        if perf_fit.check_calibration_file(ok_path, lib_path):
            raise Fail(f"--check on a good file: {perf_fit.check_calibration_file(ok_path)}")
        probs = perf_fit.check_calibration_file(vec, lib_path)
        if (len(probs) != 2 or ":9: notes: 128 bytes" not in probs[0] or
                "[TOO_LONG]" not in probs[1]):
            raise Fail(f"--check on the over-long vector: {probs}")
        for argv, code in ((["--check", ok_path], 0), (["--check", ok_path, vec], 1)):
            try:
                perf_fit.main(argv + ["--lib", lib_path])
                raise Fail(f"perf_fit {argv}: no exit")
            except SystemExit as e:
                if e.code != code:
                    raise Fail(f"perf_fit {argv}: exit {e.code}, want {code}")
    return n + 8


def suite_perf(args):
    host.load(args.host_lib)
    os.environ["IMMERDINK_HOST_LIB"] = args.host_lib
    # --- calibration: shipped file == default; format -> parse is bit-exact ---------------
    cal = host.Calibration.default()
    shipped = host.Calibration.load(os.path.join(ROOT, "src/host/calib/rtx4090_sm89.calib"))
    if cal.params != shipped.params or cal.meta != shipped.meta:
        raise Fail("the shipped calibration file differs from imd_perf_calib_default")
    if cal.meta["sm_count"] != 128 or "4090" not in cal.meta["gpu"]:
        raise Fail(f"shipped calibration: {cal.meta}")
    again = host.Calibration.parse(cal.format())
    if again.params != cal.params or again.meta != cal.meta or again.format() != cal.format():
        raise Fail("format -> parse round trip")
    names = host.param_names()
    if sorted(cal.params) != sorted(f"{c}.{n}" for c, ns in names.items() for n in ns
                                    if n and (cal.file_format == 2 or
                                              f"{c}.{n}" not in host.FORMAT2_PARAMS)):
        raise Fail("params keys")
    rng = random.Random(6)
    vals = {k: rng.choice((0.0, rng.random() * 10 ** rng.randint(-6, 4))) for k in cal.params}
    vals["call.launch"] = -1.5
    cv = host.Calibration.from_values(cal.meta, vals, extra=cal.extra or None)
    if cv.params != vals or host.Calibration.parse(cv.format()).params != vals:
        raise Fail("from_values / round trip of arbitrary values")
    for bad in ("format = 1\n", cal.format().replace("k5.op =", "k5.ops ="),
                cal.format() + "gpu = again\n", cal.format().replace("kl.byte = ", "kl.byte = -")):
        try:
            host.Calibration.parse(bad)
            raise Fail(f"accepted an invalid calibration: {bad[-40:]!r}")
        except ValueError:
            pass
    n_err = suite_calib_errors(cal, args.host_lib)  # host API 10 (DESIGN §6c.3)
    if args.verbose:
        print(f"  ok  calibration errors: {n_err} checks (reasons, lines, keys, messages; "
              f"perf_fit generation checks and --check)")
    # --- features == the independent transcription; K4 == the A3 estimate ----------------
    cases = [  # spec, extents, along_rows (A, B) of the post-swap operands
        ("ik,jk->ij", dict(i=64, j=64, k=64), (False, False)),
        ("ik,jk->ij", dict(i=300, j=200, k=1000), (False, False)),
        ("ik,jk->ij", dict(i=2048, j=1000, k=3000), (False, False)),
        ("ik,jk->ij", dict(i=1000, j=64, k=50000), (False, False)),
        ("ki,kj->ij", dict(i=512, j=384, k=5000), (True, True)),
        ("ik,jk->ij", dict(i=5, j=7, k=300000), (False, False)),
        ("lik,ljk->lij", dict(l=6, i=100, j=60, k=1500), (False, False)),
        ("ikl,kjl->ijl", dict(i=128, j=96, k=2500, l=4), (True, True)),
        ("i,j->ij", dict(i=300, j=200), (False, False)),
    ]
    n_feat = 0
    occ = {n: v[0] for n, v in cal.kernels.items()}
    # v1.17: the PACK hit rule on both sides (the shipped L_hit, none, and 60.4 MB: the cases'
    # group working sets range from < 1 MB to ~110 MB)
    # v1.18: and with S counting K4's W planes (warm.pack_l2_planes), at 83 MB too
    hit_cals = [(cal, cal.warm["warm.pack_l2_bytes"] if cal.file_format == 2 else 0.0,
                 cal.file_format == 2 and cal.warm["warm.pack_l2_planes"] == 1.0)]
    if cal.file_format == 2:
        for lh, pl in ((0.0, 0.0), (6.04e7, 0.0), (6.04e7, 1.0), (8.3e7, 1.0)):
            hit_cals.append((host.Calibration.from_values(
                cal.meta, cal.params, extra=dict(cal.extra, **{"warm.pack_l2_bytes": lh,
                                                               "warm.pack_l2_planes": pl})),
                lh, pl == 1.0))
    n_hit = [0, 0]
    for (hcal, lh, pl), (spec, ext, along) in itertools.product(hit_cals, cases):
        for N, g in ((8, 0), (14, 4), (20, 7), (3, 1)):
            for vec in ("GENERATE", "RESIDENT"):
                for mode in ("EMULATE", "DIRECT"):
                    lw = host.lowering(spec, ext, N=N, g=g, vec_mode=vec, mode=mode)
                    p = host.predict(spec, ext, N=N, g=g, vec_mode=vec, mode=mode, calib=hcal)
                    if p.status != lw.status:
                        raise Fail(f"{spec} {ext}: status {p.status} vs lowering {lw.status}")
                    if p.status != "SUCCESS":
                        continue
                    want = spec_phases(lw, vec, along, cal.meta["l2_bytes"], occ=occ,
                                       tail_mod=64 if cal.file_format == 1 else 32, l_hit=lh,
                                       planes=pl)
                    for q in p.phases:
                        if q.cls == "pack":
                            n_hit[q.x[3] < q.x[1]] += 1
                    got = [(q.name, q.launches, list(q.x)) for q in p.phases]
                    if along != (False, False):  # the chains' transcription assumes radix 1
                        for ph in want + got:
                            if ph[0] in CHAIN_SLOT and ph[0] != "KF fused":
                                ph[2][CHAIN_SLOT[ph[0]]] = 0.0
                                if ph[0] in ("K2 pack A", "K3 pack B"):
                                    ph[2][5] = 0.0  # pack.wave
                    ok = len(want) == len(got) and all(
                        a[0] == b[0] and a[1] == b[1] and all(close(u, v) for u, v in zip(a[2], b[2]))
                        for a, b in zip(want, got))
                    if not ok:
                        raise Fail(f"features {spec} {ext} N={N} g={g} {vec} {mode}:\n  "
                                   f"got  {got}\n  want {want}")
                    if p.kernel_launches != lw.kernel_launches or p.k4_config != lw.k4_config or \
                            p.workspace_bytes != lw.workspace_bytes or \
                            p.plan_device_bytes != lw.plan_device_bytes:
                        raise Fail(f"summary {spec} {ext}: {p} vs {lw}")
                    if not close(p.time_s, sum(q.time_s for q in p.phases)):
                        raise Fail("time_s != sum of phases")
                    if not close(p.tflops_eq, p.flops_eq / p.time_s * 1e-12):
                        raise Fail("tflops_eq")
                    n_feat += 1
    if len(hit_cals) > 1 and not (n_hit[0] and n_hit[1]):
        raise Fail(f"PACK hit rule not exercised on both sides: {n_hit}")
    # K4 features x the A3 constants == the k4_select estimate (one chunk; G equal groups)
    a3 = {"k4_128x256.round": 2716, "k4_128x256.mac1": 418e-6,
          "k4_128x128.round": 2550, "k4_128x128.mac1": 453e-6, "k4_128x128.mac2": 456e-6,
          "k4_64x128.round": 1424, "k4_64x128.mac1": 505e-6, "k4_64x128.mac2": 612e-6,
          "k4_64x64.round": 1333, "k4_64x64.mac1": 944e-6, "k4_64x64.mac2": 1088e-6,
          "k4_64x64.mac3": 802e-6}
    for (m, n, k, b) in ((64, 64, 64, 1), (1000, 64, 50000, 1), (4096, 4096, 4096, 1),
                         (300, 200, 1000, 7), (16384, 512, 4096, 1)):
        spec, ext = "lik,ljk->lij", dict(l=b, i=m, j=n, k=k)
        lw = host.lowering(spec, ext, N=14, g=7)
        for c in range(4):
            p = host.predict(spec, ext, N=14, g=7, k4_config=c)
            q = p.phase("K4 gemm")
            ns = sum(a3.get(f"{q.cls}.{nm}", 0.0) * x for nm, x in zip(names[q.cls][:4], q.x) if nm)
            est = 2 * spec_k4_estimate(c, lw.m, lw.n_hat, lw.k_hat, 7 * lw.b, 128) / 1e6
            if q.cls != host.PERF_CLASSES[4 + c] or not close(ns, est, 1e-9):
                raise Fail(f"K4 {m},{n},{k},{b} config {c}: {ns} ns vs A3 estimate {est} ns")
    # --- monotonicity: a bigger shape never takes less time (same K4 configuration, path) -
    # Against the shipped (fitted) calibration and a synthetic one with large v1.17 terms
    # (kf.elem, pack.dram, L_hit 60.4 MB) and the retired op_eps keys in its text (v1.15 /
    # v1.17: accepted and discarded; every request eps fraction predicts the same time).
    heavy = dict(cal.params)
    for key in ("kf.elem", "pack.dram"):
        heavy[key] = 50 * (heavy[key] + 1e-4)
    hx = dict(cal.extra or {}, **{"warm.pack_l2_bytes": 6.04e7})
    mono = 0
    for which, cb in (("shipped", None), ("heavy v1.17", host.Calibration.from_values(
            cal.meta, heavy, extra=hx))):
        for t in range(120):
            # n a multiple of 16: doubling n then keeps n_hat = n (no zero-rows launch appears
            # or disappears; one launch less can be a real saving)
            m, n = rng.randint(1, 4000), 16 * rng.randint(1, 250)
            k = rng.choice((rng.randint(2, 1000), rng.randint(1100, 60000)))
            N = rng.randint(2, 20)
            g = rng.randint(1, N)
            for eps in (0.0, 1.0, None):
                base = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=N, g=g, eps_fraction=eps,
                                    calib=cb)
                if base.status != "SUCCESS":
                    continue
                for ext in (dict(i=2 * m, j=n, k=k), dict(i=m, j=2 * n, k=k)) + \
                        ((dict(i=m, j=n, k=2 * k),) if k > 1024 else ()):
                    big = host.predict("ik,jk->ij", ext, N=N, g=g, k4_config=base.k4_config,
                                       eps_fraction=eps, calib=cb)
                    if big.status != "SUCCESS":
                        continue
                    floor = base.time_s
                    if big.time_s < floor * (1 - 1e-12):
                        raise Fail(f"monotonicity ({which}, eps {eps}): {ext} N={N} g={g}: "
                                   f"{big.time_s} < {floor} ({m},{n},{k})")
                    mono += 1
            more_n = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=min(N + 1, 20), g=0, calib=cb)
            same = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=N, g=0,
                                k4_config=more_n.k4_config, calib=cb)
            if more_n.status == same.status == "SUCCESS" and more_n.time_s < same.time_s:
                raise Fail(f"monotonicity in N ({which}): {m},{n},{k} N={N}")
    # v1.15 / v1.17 (DESIGN §6c.1 "The eps path — retired", §6c.2, §6c.4): every request
    # fraction predicts bit-identically, eps_s / eps_fraction are 0, and the retired op_eps
    # keys in a calibration's text are accepted and have no effect (not stored, not written).
    opk = ("kf.op_eps", "kl.op_eps", "pack.op_eps")

    def with_op_eps(c, vals=(5e-4, 7e-4, 9e-4)):
        return host.Calibration.parse(c.format() + "".join(
            f"{kk} = {v!r}\n" for kk, v in zip(opk, vals)))
    zc = host.Calibration.parse(cal.format())
    hv = with_op_eps(cal)
    if bytes(hv._raw) != bytes(zc._raw) or any(f"{kk} = " in hv.format() for kk in opk):
        raise Fail("retired op_eps keys: stored or written")
    n_eps = 0
    for N in range(2, 21):
        for k in (16, 64, 300, 1024, 4096, 50000):
            ext = dict(i=256, j=192, k=k)
            for rg in ("BURST", "SUSTAINED"):
                ref = host.predict("ik,jk->ij", ext, N=N, regime=rg, calib=zc)
                if ref.status != "SUCCESS":
                    continue
                for f in (None, -1.0, 0.0, 0.5, 1.0):
                    for cb in (None, hv, zc):
                        p = host.predict("ik,jk->ij", ext, N=N, eps_fraction=f, regime=rg, calib=cb)
                        if repr(p) != repr(ref):
                            raise Fail(f"eps retired: N={N} k={k} {rg} f={f}: prediction differs")
                        if p.eps_s != 0 or p.eps_fraction != 0:
                            raise Fail(f"eps retired: eps_s {p.eps_s} / eps_fraction {p.eps_fraction}")
                        for q in p.phases:
                            if q.cls == "kl" and q.x[4] != 0:
                                raise Fail(f"eps retired: kl slot 4 feature {q.x[4]}")
                        n_eps += 1
            pk = host.pack_desc_for("ik,jk->ij", ext, operand=0, N=N)
            pp = [host.predict_pack(pk, eps_fraction=f, regime="BURST", calib=zc)
                  for f in (None, 0.0, 1.0)]
            if pp[0].status == "SUCCESS" and (repr(pp[1]) != repr(pp[0]) or repr(pp[2]) != repr(pp[0])
                                               or pp[0].eps_s != 0):
                raise Fail(f"eps retired: predict_pack N={N} k={k}")
    rk = [host.rank("ik,jk->ij", dict(i=512, j=512, k=2048), N=[14, 16], packs=("NONE", "A"),
                    eps_fraction=f, calib=hv) for f in (None, 0.0, 1.0)]
    if repr(rk[0]) != repr(rk[1]) or repr(rk[0]) != repr(rk[2]) or any(c.eps_s != 0 or c.eps_fraction != 0 for c in rk[0]):
        raise Fail("eps retired: rank depends on eps_fraction or reports eps terms")
    for bad in (1.5, float("nan")):
        try:
            st = host.predict("ik,jk->ij", dict(i=8, j=8, k=8), N=16, eps_fraction=bad).status
        except Exception:
            continue
        if st == "INVALID_VALUE":
            continue
        raise Fail(f"eps retired: eps_fraction {bad} accepted (still validated)")
    # the retired keys in format 1 too (the v6 test vector), and the v1.15 shipped file
    # (with its op_eps keys absent) still loads
    v6 = host.Calibration.load(os.path.join(ROOT, "tests/data/perf_calib_v6_format1.calib"))
    if bytes(with_op_eps(v6)._raw) != bytes(v6._raw):
        raise Fail("retired op_eps keys in format 1")
    for bad, why in (("kf.op_eps = -1\n", "negative"), ("kf.op_eps = 1\nkf.op_eps = 1\n", "repeated")):
        try:
            host.Calibration.parse(cal.format() + bad)
            raise Fail(f"retired op_eps key accepted ({why})")
        except host.CalibrationError:
            pass
    n_eps += 1
    # DESIGN v1.15 §6c.2 "Not backward-readable" (review 2026-10-06): the shipped file omits
    # the op_eps keys (MISSING_KEY in API <= 10) and carries keys an older library rejects
    # (v1.15: power.byte_f_pj; v1.18: the busy-power / boost / cache keys, unknown in API <= 12);
    # if either changes, the format note in §6c.2 must be revisited.
    with open(os.path.join(ROOT, "src/host/calib/rtx4090_sm89.calib")) as f:
        shipped_keys = {l.split(" = ")[0].strip() for l in f if " = " in l and not l.startswith("#")}
    if shipped_keys & set(opk) or not shipped_keys & ({"power.byte_f_pj"} | set(V118_KEYS)):
        raise Fail("shipped calibration: op_eps / optional keys differ from DESIGN §6c.2's "
                   "backward-readability note")
    n_eps += 1
    print(f"  ok  eps retired (v1.15 / v1.17):{n_eps} predictions independent of eps_fraction; "
          "retired op_eps keys accepted and discarded")
    # --- statuses: kappa >= T (N = 2, T = 180) gives the plan-time status and no time -------
    p = host.predict("ik,jk->ij", dict(i=3, j=5, k=180 * 180), N=2)
    if p.status != "NOT_SUPPORTED" or p.time_s != 0 or p.phases:
        raise Fail(f"kappa >= T: {p}")
    p = host.predict("ik,jk->ij", dict(i=3, j=0, k=5), N=8)
    if p.status != "SUCCESS" or p.time_s != 0 or p.phases or p.dispatch != "NOOP":
        raise Fail(f"empty output: {p}")
    if host.predict("ik,jk->ij", dict(i=3, j=5, k=7), N=8, k4_config=9).status != "INVALID_VALUE":
        raise Fail("k4_config 9 accepted")
    # --- packed predictions: no KF / KL share or pack launches for a packed operand --------
    spec, ext = "ik,jk->ij", dict(i=2048, j=1000, k=3000)
    pa = host.pack_desc_for(spec, ext, operand=0, N=14)
    p = host.predict(spec, ext, N=14, packing=(pa, None))
    if p.phase("KL A") is not None or p.phase("KL B") is None or p.phase("K2 pack A"):
        raise Fail(f"packed A: {[q.name for q in p.phases]}")
    pp = host.predict_pack(pa)
    lay = host.pack_layout(pa)
    if [q.name for q in pp.phases] != ["pack prep", "pack KL", "call"] or \
            pp.pack_bytes != lay.pack_bytes or pp.pack_ws_bytes != lay.pack_ws_bytes:
        raise Fail(f"predict_pack: {pp}")
    # --- rank: exact footprints, budget, order --------------------------------------------
    n_rank = 0
    for spec, ext in (("ik,jk->ij", dict(i=2048, j=1000, k=3000)),
                      ("kjbc,kiac->abij", dict(a=24, b=20, c=16, i=12, j=10, k=8)),
                      ("lik,ljk->lij", dict(l=6, i=100, j=60, k=1500))):
        Ns = [8, 14]
        allc = host.rank(spec, ext, N=Ns, vec_modes=("GENERATE", "RESIDENT"),
                         packs=("NONE", "A", "B", "AB"), reuse=3)
        if len(allc) != sum(N for N in Ns) * 2 * 4:
            raise Fail(f"rank: {len(allc)} candidates")
        foot = sorted(c.total_bytes for c in allc if c.status == "SUCCESS")
        budget = foot[len(foot) // 3]
        ranked = host.rank(spec, ext, N=Ns, vec_modes=("GENERATE", "RESIDENT"),
                           packs=("NONE", "A", "B", "AB"), reuse=3, budget=budget,
                           order="TIME")  # the v6 order (ORDER_BAND: suite_perf_v110)
        seen_bad = False
        for i, c in enumerate(ranked):
            pk = [host.pack_desc_for(spec, ext, operand=x, N=c.num_moduli)
                  if c.packed in (("A", "AB"), ("B", "AB"))[x] else None for x in (0, 1)]
            kw = dict(N=c.num_moduli, g=c.residue_group, vec_mode=c.vec_mode)
            if c.status != "SUCCESS":
                seen_bad = True
                continue
            if seen_bad:
                raise Fail("rank: a SUCCESS candidate after a failed one")
            ws = host.workspace_size(spec, ext, packing=tuple(pk) if any(pk) else None, **kw)
            dev, _ = host.plan_bytes(spec, ext, **kw)
            pb = sum(host.pack_layout(x).pack_bytes for x in pk if x)
            pw = sum(host.pack_layout(x).pack_ws_bytes for x in pk if x)
            if (c.workspace_bytes, c.plan_device_bytes, c.pack_bytes, c.pack_ws_bytes,
                    c.total_bytes) != (ws, dev, pb, pw, ws + dev + pb + pw):
                raise Fail(f"rank footprint {c} vs {(ws, dev, pb, pw)}")
            if c.feasible != (c.total_bytes <= budget):
                raise Fail(f"rank feasibility {c}")
            ps = sum(host.predict_pack(x).time_s for x in pk if x)
            pc = host.predict(spec, ext, packing=tuple(pk) if any(pk) else None, **kw).time_s
            if not (close(c.contract_s, pc) and close(c.pack_s, ps) and
                    close(c.time_s, pc + ps / 3)):
                raise Fail(f"rank times {c}: contract {pc}, pack {ps}")
            if i > 0 and ranked[i - 1].status == "SUCCESS":
                prev = ranked[i - 1]
                if prev.feasible < c.feasible or (prev.feasible == c.feasible and
                                                  prev.time_s > c.time_s):
                    raise Fail(f"rank order: {prev} before {c}")
            n_rank += 1
    n_v110 = suite_perf_v110(cal, names, rng)  # DESIGN v1.10 (host API 7, format 2)
    if args.verbose:
        print(f"  ok  v1.10: {n_v110} checks (format 1 = v6, format 2, regimes, cache, bands, "
              f"rank ties)")
    n_v115 = suite_perf_v115(cal)  # DESIGN v1.15 §6c.7 (host API 11)
    if args.verbose:
        print(f"  ok  v1.15: {n_v115} checks (power.byte_f_pj optional, round trip, errors, "
              f"folded v10 view, identity at 0, clock monotonicity)")
    n_v117 = suite_perf_v117(cal)  # DESIGN v1.17 §6c.1 / §6c.2 (host API 12)
    if args.verbose:
        print(f"  ok  v1.17: {n_v117} checks (kf.elem / pack.dram / warm.pack_l2_bytes optional "
              f"and additive, round trip, errors, v11 view)")
    n_v118 = suite_perf_v118(cal)  # DESIGN v1.18 §6c.8 (host API 13)
    if args.verbose:
        print(f"  ok  v1.18: {n_v118} checks (seven optional keys, grammar, CLEAN / REPEAT cache "
              f"terms, S with planes, boost clock, busy power, DRAM energy, v12 view)")
    n_v119 = suite_perf_v119(cal)  # DESIGN v1.19 §6c.9 (host API 13, warm.kl_drop)
    if args.verbose:
        print(f"  ok  v1.19: {n_v119} checks (warm.kl_drop optional, grammar, KL share of the "
              f"REPEAT saving, COLD / CLEAN / non-KL phases unchanged)")
    if args.verbose:
        print(f"  ok  calibration round trip, {n_feat} feature transcriptions, K4 == A3 "
              f"estimate, {mono} monotonicity pairs, statuses, packed, {n_rank} ranked "
              f"candidates with exact footprints")


# ---------------------------------------------------------------------------------------
# Suite plan (GPU)

def suite_plan(args):
    sys.path.insert(0, HERE)
    import harness
    lib = harness.Library(args.lib)
    if not lib.has_test_hooks:
        raise Fail("libimmerdink was built without test hooks")
    f = lib.lib.imd_test_plan_sizes
    f.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.c_int64, C.POINTER(C.c_int64)]
    f.restype = C.c_int
    lib.set_limits()  # production launch limits (the host queries use them)
    direct = Direct(args.host_lib)
    host.load(args.host_lib)
    os.environ["IMMERDINK_HOST_LIB"] = args.host_lib
    hdl = C.c_void_p()
    assert lib.lib.imd_create(C.byref(hdl)) == 0
    rng = random.Random(20261005)
    n_plans, n_k4, n_tuned = 0, 0, 0
    path_of = {0: "NOOP", 1: "EMULATE", 2: "DIRECT"}  # imd::detail::Path order (k = 0: SCALE)

    def one(descs, opts, what, tune=0):
        nonlocal n_plans, n_k4, n_tuned
        co = c_opts(*opts) + (tune,)
        st, low = direct.lowering(descs, co)
        T = [harness.make_tensor(*d) for d in descs]
        plan = C.c_void_p()
        pst = lib.lib.imd_plan_create(hdl, *[C.byref(t) for t in T],
                                      C.byref(harness.Options(*co)), C.byref(plan))
        if pst != st:
            lib.lib.imd_plan_destroy(plan)
            raise Fail(f"{what}: imd_plan_create {STATUS[pst]}, imd_lowering {STATUS[st]}")
        if pst != 0:
            return
        try:
            cnt = C.c_int64(0)
            buf = (C.c_int64 * 64)()
            assert f(plan, buf, 64, C.byref(cnt)) == 0 and cnt.value <= 64
            v = list(buf)[:cnt.value]
            path, sw, m, n, k, b, nh, kh, N, g, total = v[:11]
            regs = [(v[11 + 2 * r], v[12 + 2 * r]) for r in range(8)]
            dev, store, add, launches, hbytes, fused, kernels = v[27:34]
            cfg, k4p_cfg, sms, fused_large = v[34:38]
            _, ltot, lay = direct.layout(descs, co)
            _, pdev, phost = direct.plan_bytes(descs, co)
            # v1.5: the plan runs the rule's K4 configuration for its device's SM count (any
            # configuration with k4_tune), and holds that configuration's parameters.
            _, low_s = direct.lowering(descs, co, sm_count=sms)
            if path == 1 and k > 0:
                ok = cfg == k4p_cfg and 0 <= cfg < 4 and (tune or cfg == low_s["k4_config"])
                n_tuned += bool(tune and low_s["k4_tune_bytes"])
            else:
                ok = cfg == k4p_cfg == -1
            if not ok or (sms and low_s["sm_count"] != sms):   # a NOOP plan reads no SM count
                raise Fail(f"{what}: plan K4 configuration {cfg} / {k4p_cfg} on {sms} SMs, "
                           f"host query {low_s['k4_config']} (k4_tune {tune})")
            dsz = C.c_size_t(0)
            assert lib.lib.imd_plan_device_bytes(plan, C.byref(dsz)) == 0
            got = dict(dispatch="SCALE" if k == 0 and path else path_of[path], swapped=sw, m=m, n=n, k=k, b=b, n_hat=nh,
                       k_hat=kh, num_moduli=N, residue_group=g, workspace_bytes=total,
                       gemm_launches=launches, plan_device_bytes=dev, plan_host_bytes=hbytes,
                       fused=fused, kernel_launches=kernels, fused_large=fused_large)
            want = dict(low, dispatch=host.DISPATCH[low["dispatch"]])
            for key, val in got.items():
                if val != want[key]:
                    raise Fail(f"{what}: plan {key} = {val}, host query {want[key]}")
            if regs != lay or total != ltot or dev != pdev or hbytes != phost or dsz.value != dev:
                raise Fail(f"{what}: plan layout / bytes differ from the host queries")
            n_plans += 1
            n_k4 += path == 1
            del store, add  # counted in the host payload (hbytes)
        finally:
            lib.lib.imd_plan_destroy(plan)

    n_packed = {}

    def one_packed(descs, opts, what):
        """v1.7 (DESIGN §6b): real packed plans (imd_pack_create + imd_plan_create_packed)
        == imd_lowering_packed / imd_workspace_layout_packed: status, view, layout, launches,
        fused paths, packed slots, pack bytes and the host payload (incl. the copied keys)."""
        N, g, rnd, mode, vec = opts
        A, B, Cd = (oracle.Desc(*d) for d in descs)
        omode = oracle.EMULATE if mode == "EMULATE" else oracle.DIRECT
        try:
            oracle.lower(A, B, Cd, omode, build_vec=False)
        except (oracle.InvalidValue, oracle.NotSupported):
            return
        co = c_opts(*opts)
        for which in ((0,), (1,), (0, 1)):
            for order in ("desc_for", "auto"):
                specs = [None, None]
                for x in which:
                    if order == "desc_for":
                        specs[x] = host.pack_desc_for(descs, operand=x, N=N, g=g, rounding=rnd,
                                                      mode=mode, vec_mode=vec)
                    else:
                        X = descs[x]
                        xl = list(dict.fromkeys(X[2]))
                        specs[x] = host.PackSpec(host.Desc(*map(tuple, X)), N, rnd, "AUTO",
                                                 tuple(lab for lab in xl if lab not in descs[2][2]),
                                                 tuple(lab for lab in xl if lab in descs[2][2]
                                                       and lab in descs[1 - x][2]))
                if any(s is not None and spec_pack_status(s)[0] != "SUCCESS" for s in specs):
                    continue
                handles = [None, None]
                try:
                    for x in which:
                        hst, handles[x] = lib.pack_create(harness.make_pack_desc(oracle_pack(specs[x])))
                        if hst != "SUCCESS":
                            raise Fail(f"{what}: imd_pack_create {hst}")
                    st, low = direct.lowering(descs, co, packing=specs)
                    T = [harness.make_tensor(*d) for d in descs]
                    plan = C.c_void_p()
                    pst = lib.lib.imd_plan_create_packed(hdl, *[C.byref(t) for t in T],
                                                         C.byref(harness.Options(*co, 0)),
                                                         handles[0], handles[1], C.byref(plan))
                    if pst != st:
                        lib.lib.imd_plan_destroy(plan)
                        raise Fail(f"{what} packed {which} {order}: imd_plan_create_packed "
                                   f"{STATUS[pst]}, imd_lowering_packed {STATUS[st]}")
                    n_packed[STATUS[st]] = n_packed.get(STATUS[st], 0) + 1
                    if pst != 0:
                        continue
                    try:
                        cnt = C.c_int64(0)
                        buf = (C.c_int64 * 64)()
                        assert f(plan, buf, 64, C.byref(cnt)) == 0 and cnt.value <= 64
                        v = list(buf)[:cnt.value]
                        path, sw, m, n, k, b, nh, kh, NN, gg, total = v[:11]
                        regs = [(v[11 + 2 * r], v[12 + 2 * r]) for r in range(8)]
                        dev, store, add, launches, hbytes, fused, kernels = v[27:34]
                        fused_large, slots, pbA, pbB = v[37:41]
                        _, ltot, lay = direct.layout_packed(descs, co, specs)
                        got = dict(swapped=sw, m=m, n=n, k=k, b=b, n_hat=nh, k_hat=kh,
                                   workspace_bytes=total, gemm_launches=launches,
                                   plan_device_bytes=dev, plan_host_bytes=hbytes, fused=fused,
                                   kernel_launches=kernels, fused_large=fused_large, packed=slots)
                        for key, val in got.items():
                            if val != low[key]:
                                raise Fail(f"{what} packed {which} {order}: plan {key} = {val}, "
                                           f"imd_lowering_packed {low[key]}")
                        if regs != lay or total != ltot:
                            raise Fail(f"{what} packed: plan layout differs from the host query")
                        roles = [(x == 0) != bool(sw) for x in (0, 1)]   # True: post-swap A
                        want_pb = [0, 0]
                        for x in which:
                            want_pb[0 if roles[x] else 1] = host.pack_layout(specs[x]).pack_bytes
                        if [pbA, pbB] != want_pb:
                            raise Fail(f"{what} packed: plan pack bytes {pbA, pbB}, {want_pb}")
                    finally:
                        lib.lib.imd_plan_destroy(plan)
                finally:
                    for h in handles:
                        if h is not None:
                            lib.pack_destroy(h)

    for t in range(args.cases):
        descs = random_descs(rng)
        # v1.5: every 8th plan with k4_tune (its sizes must still equal the host queries).
        one(descs, random_opts(rng), f"random[{t}]", tune=int(t % 8 == 7))
        if t % 4 == 0:
            one_packed(descs, random_opts(rng), f"random[{t}]")
    for t in range(max(1, args.cases // 40)):
        descs = random_descs(rng, legal_bias=1.0)
        N = rng.randint(2, 20)
        for g in range(1, N + 1):
            for vec in ("GENERATE", "RESIDENT"):
                one(descs, (N, g, TRUNC, "EMULATE", vec), f"sweep[{t}] g={g} {vec}")
    # K-chunked, batch-split and n-hat-split plans (no data is touched by plan creation).
    for (m, n, k, b) in [(1, 2, 262200, 1), (3, 17, 131057, 2), (2, 3, 5, 70000),
                         (3, 2, 131056, 2), (1, 3, 131009, 1), (2, 1, 262112, 1),
                         (1, 2**24 + 16, 2, 1), (5, 7, 300000, 40000)]:
        descs = [([m, k, b], [k * b, b, 1], [1, 3, 4]), ([k, n, b], [n * b, b, 1], [3, 2, 4]),
                 ([m, n, b], [n * b, b, 1], [1, 2, 4])]
        for g in (1, 3, 20):
            one(descs, (20, g, TRUNC, "EMULATE", "GENERATE"), f"big {m},{n},{k},{b} g={g}")
    lib.lib.imd_destroy(hdl)
    if args.verbose:
        print(f"  ok  {n_plans} plans ({n_k4} emulated, {n_tuned} autotuned) sized exactly as the "
              f"host queries say, K4 configurations as the rule says; packed plans {n_packed}")
    if n_packed.get("SUCCESS", 0) < 50:
        raise Fail(f"too few packed plans: {n_packed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", required=True)
    ap.add_argument("--lib", help="libimmerdink.so (suite plan)")
    ap.add_argument("--suite", choices=("host", "perf", "plan"), required=True)
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.cases is None:
        args.cases = 4000 if args.suite == "host" else 400
    try:
        {"host": suite_host, "perf": suite_perf, "plan": suite_plan}[args.suite](args)
    except Fail as e:
        print(f"FAIL {e}")
        return 1
    print(f"PASS {args.suite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
