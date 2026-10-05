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
import ctypes as C
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "spec"))
sys.path.insert(0, os.path.join(ROOT, "python"))
import oracle  # noqa: E402
import params  # noqa: E402
from immerdink import host  # noqa: E402

STATUS = host.STATUS
RN, TRUNC = "RN", "TRUNC"


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
                   "pack_mismatch"]
# sizeof(imd_lowering_t) per host API version (DESIGN §6a; v5: §6b)
V1, V2, V3, V4, V5 = 136, 152, 168, 176, 184


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

    def lowering(self, descs, opts, size=V5, sm_count=0, packing=None):
        """imd_lowering (imd_lowering_packed with `packing`) into a raw struct of `size`
        bytes (184: version 5; 176 / 168 / 152 / 136: a version-4 / -3 / -2 / -1 caller,
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
        v2, v3, v4, v5 = [], [], [], []
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
        return st, dict(zip(LOWERING_FIELDS, i32[1:] + i64 + u64 + v2 + v3 + v4 + v5))


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
    try:
        ws = params.workspace_size(v.m, v.n, v.k, v.b, N, g, pmode, pvec)
    except OverflowError:
        return "INVALID_VALUE", None
    m, n, k, b = v.m, v.n, v.k, v.b
    gg = N if g == 0 else g
    k_hat, n_hat = params.align_up(k, 16), params.align_up(n, 16)
    noop = b * m * n == 0
    dispatch = ("NOOP" if noop else "SCALE" if k == 0 else
                "DIRECT" if (mode == "DIRECT" or k == 1) else "EMULATE")
    exp = dict(m=m, n=n, k=k, b=b, swapped=bool(v.swapped), n_hat=n_hat, k_hat=k_hat,
               kappa=oracle.ceil_sqrt(k) if k > 0 else 0, workspace_bytes=ws,
               dispatch=dispatch, num_moduli=N, residue_group=gg,
               vec_entries=0 if noop else 2 * m + 2 * n + 2 * k + 3 * b,
               regions=params.region_offsets(m, n, k, b, N, g, pmode, pvec),
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
              "fused_large"):
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
    regs, total = params.pack_layout(lay.b, lay.r, lay.k, p.num_moduli)
    want_regs = {n: (o, nb) for n, o, nb in regs} or {n: (0, 0) for n in ("sig", "flag", "res")}
    if lay.regions != want_regs or lay.pack_bytes != total or \
            lay.pack_ws_bytes != params.pack_ws_bytes(lay.b, lay.r, lay.k) or \
            lay.k_hat != params.align_up(lay.k, 16) or lay.file_format != 1:
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
        want_ws = params.workspace_size(v.m, v.n, v.k, v.b, N, g, pmode, pvec, packed=roles)
        want_regs = params.region_offsets(v.m, v.n, v.k, v.b, N, g, pmode, pvec, packed=roles)
        if ws != want_ws or lay_tot != want_ws or low["workspace_bytes"] != want_ws or \
                host.workspace_size(descs, packing=packing, **kw) != want_ws:
            raise Fail(f"{what}: packed workspace {ws} / {lay_tot}, params {want_ws}")
        if [(host.REGIONS[r], o, o + nb) for r, (o, nb) in enumerate(lay) if nb] != \
                [tuple(x) for x in want_regs]:
            raise Fail(f"{what}: packed regions {lay}, params {want_regs}")
        # launches and fused paths of the packed plan (DESIGN §6b.5)
        n_hat, k_hat = params.align_up(v.n, 16), params.align_up(v.k, 16)
        emu = mode == "EMULATE" and not v.empty_output() and v.k > 1
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
           (1, 2**24 + 16, 2, 1), (5, 7, 300000, 40000)]
    for (m, n, k, b) in big:
        descs = [([m, k, b], [k * b, b, 1], [1, 3, 4]), ([k, n, b], [n * b, b, 1], [3, 2, 4]),
                 ([m, n, b], [n * b, b, 1], [1, 2, 4])]
        for g in (1, 3, 20):
            check_case(direct, descs, (20, g, TRUNC, "EMULATE", "GENERATE"), f"big {m},{n},{k},{b}")
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
    if args.verbose:
        print(f"  ok  {args.cases} random cases {by}, {n_sweep} g/vec sweep cases, kappa "
              f"boundary, {len(big)} K4-split shapes, einsum front end, g_tradeoff; packed "
              f"queries {packed_seen}")


# ---------------------------------------------------------------------------------------
# Suite perf (DESIGN v1.8 §6c): the performance model, CPU only

L32 = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]  # §2: CRT limbs per N
MODULI = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191,
          181, 179, 173]  # DESIGN §2
K_C_PRIME = 131056


def spec_k4_features(c, m, n_hat, k_hat, b, N, g, sms, k_chunks, l2):
    """K4 features (rounds, MACs at q = 1..3 concurrent CTAs) over the groups and K-chunks,
    written from DESIGN §6c.1 independently of src/host/perf.cpp."""
    tm, tn, occ = K4_TILES[c]
    x = [0.0] * 6
    groups = [g] * (N // g) + ([N % g] if N % g else [])
    last = k_hat - (k_chunks - 1) * K_C_PRIME
    chunks = [K_C_PRIME] * (k_chunks - 1) + [last]
    for gs in groups:
        for kc in chunks:
            tiles = -(-m // tm) * -(-n_hat // tn) * gs * b
            j = -(-tiles // sms)
            full, rest = divmod(j, occ)
            x[0] += full + (1 if rest else 0)
            x[occ] += full * occ * tm * tn * kc
            if rest:
                x[rest] += rest * tm * tn * kc
            if kc % 64:  # a partial 64-wide K tile: the chunk's MACs again (mac_ktail)
                x[5] += (full * occ + rest) * tm * tn * kc
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


def spec_kf_chain(rows, k_hat, g, sms, r16=2, r4=4):
    """KF's busiest-SM chain (DESIGN §6c.1), for operands read along k-bar (radix 1): tiles of
    4096 values (E = 16, r16 blocks per SM) or, below 16 blocks per SM even with all moduli
    split, 1024 (E = 4, r4 per SM); moduli parts when tiles < 2 SMs."""
    tk = 16
    while tk < k_hat:
        tk *= 2
    tiles = sum(-(-r // (4096 // tk)) for r in rows)
    small = tiles * g < 16 * sms
    if small:
        tiles = sum(-(-r // max(1024 // tk, 1)) for r in rows)
    parts = spec_parts(tiles, g, sms)
    return spec_chain(spec_launch_blocks(tiles, 1), parts, tiles,
                      (4 if small else 16) * -(-g // parts), r4 if small else r16, sms)


def spec_phases(lw, vec, along_rows, l2, beta=False, occ=None, sms=128):
    """[(name, launches, x)] of one imd_contract (DESIGN §6c.1 table) from the lowering
    summary and the operands' read direction (along rows: KL tiles of 512). occ: kernel name ->
    CTAs per SM (the calibration's occupancy table); the chains assume radix 1 (operands read
    along k-bar) and j-bar-fastest K5: callers compare them only for such cases."""
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
    # the typical-data eps fraction (immerdink_host.h): exact log2 T, the margin ramp
    lt = math.log2(math.isqrt((math.prod(MODULI[:N]) - 1) // 2))
    wide = 0.0 if lt < 53 else min(1.0, max(0.0, (lt - 0.5 * math.log2(k) - 53 + 2.5) / 1.5))
    if not lw.fused and not res:
        ph.append(k0)
    if lw.fused:
        byts = (0 if res else 8 * V) + sum(8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x]
                                           for x in (0, 1))
        ops = g * (rows[0] + rows[1]) * k
        ph.append(("KF fused", 1, [1, byts, ops, rows[0] + rows[1], wide * ops,
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
            chain = spec_chain(min(tiles, 1 << 20), 1, tiles, nc * values * g,
                               r("kl4" if values == 4 else "kl8"), sms)
            ph.append((name, 1, [1, 8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x],
                                 g * rows[x] * k, 16 * rows[x] * max(0, kh - 2 * tk),
                                 wide * g * rows[x] * k, (1 + wide) * chain]))
    G = lw.num_groups
    sizes = [g] * (N // g) + ([N % g] if N % g else [])
    for x, name in ((0, "K2 pack A"), (1, "K3 pack B")):
        if G > 1:
            per = 2 if (x == 1 and nh > n) else 1
            chain = 0
            for gs in sizes[1:]:  # group 1 is KF's / KL's
                tiles = -(-rows[x] // 64) * -(-kh // 64)
                parts = spec_parts(tiles, gs, sms)
                chain += spec_chain(spec_launch_blocks(tiles, 1), parts, tiles,
                                    16 * -(-gs // parts), r("pack"), sms)
            ph.append((name, (G - 1) * per, [(G - 1) * per, (G - 1) * 8 * rows[x] * k
                                             + (N - g) * pad[x] * kh, (N - g) * rows[x] * k,
                                             wide * (N - g) * rows[x] * k, (1 + wide) * chain,
                                             0]))
    ph.append(("K4 gemm", lw.gemm_launches,
               spec_k4_features(lw.k4_config, m, nh, kh, b, N, g, sms, lw.k_chunks, l2)))
    mn = b * m * n
    e = 1 if mn <= 1024 * sms else 2 if mn <= 8 * 2048 * sms else 4
    work = b * m * -(-n // e)
    k5c = spec_chain(spec_launch_blocks(work), 1, -(-work // 256), e * N * L32[N],
                     r(f"k5_n{N}_e{e}"), sms)
    ph.append(("K5 crt", 1, [1, N * b * m * nh + 3 * b * (m + n) + outb, N * L32[N] * b * m * n,
                             k5c, 0, 0]))
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
    "k0": (2,), "kf": (2, 3, 4, 5), "kl": (2, 3, 4, 5), "pack": (2, 3, 4), "k5": (2, 3),
    "direct": (2, 3), "beta": (2,), "call": ()}


def synthetic_v2(cal, names, rng):
    """A format-2 calibration: the shipped parameters, random chain parameters, a power model
    (cap 450 W), a cache term and bands."""
    params = dict(cal.params)
    for key in host.FORMAT2_PARAMS:
        params[key] = rng.choice((0.0, rng.random() * 0.05))
    extra = {f"kernel.{n}": (v[0] + rng.randint(0, 2), 64, 4096, v[3])
             for n, v in cal.kernels.items()}
    extra.update({"power.clock_ref_mhz": 2800.0, "power.clock_min_mhz": 1400.0,
                  "power.cap_w": 450.0, "power.idle_w": 60.0, "power.tc_pj": 0.7,
                  "power.alu_pj": 5.0, "power.byte_pj": 20.0, "power.gamma": 2.0,
                  "warm.l2_bytes": 3.0e7, "warm.l2byte": 3e-4,
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
                                q.p10_s != q.time_s or q.p90_s != q.time_s:
                            raise Fail(f"format 1 is not v6: {spec} {ext} {q}")
                    want_regime = "BURST" if regime == "DEFAULT" else regime
                    if p.regime != want_regime or p.clock_mhz != 0 or p.power_w != 0 or \
                            p.time_p10_s != p.time_s or p.time_p90_s != p.time_s:
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
            pb = host.predict(spec, ext, regime="BURST", cache="REPEAT", **kw)
            if pb.status != "SUCCESS" or not pb.phases:
                continue
            if pb.clock_mhz != 2800.0:
                raise Fail(f"BURST clock {pb.clock_mhz}")
            pr = host.predict(spec, ext, clock_mhz=2800.0, cache="REPEAT", **kw)
            if not close(pr.time_s, pb.time_s):
                raise Fail("clock_mhz = f_ref differs from BURST")
            p2 = host.predict(spec, ext, clock_mhz=5600.0, cache="REPEAT", **kw)
            for q, q2 in zip(pb.phases, p2.phases):  # clock-bound terms halve
                if not close(q2.time_s, 1e-9 * phase_ns(params, names, q, 0.5), 1e-9) or \
                        not close(q.time_s, 1e-9 * phase_ns(params, names, q), 1e-9):
                    raise Fail(f"clock scaling {spec} {ext} {q.name}: {q2.time_s} vs "
                               f"{1e-9 * phase_ns(params, names, q, 0.5)}")
            if p2.time_s > pb.time_s * (1 + 1e-12):
                raise Fail("a higher clock predicts a longer time")
            ps = host.predict(spec, ext, regime="SUSTAINED", cache="REPEAT", **kw)
            pd = host.predict(spec, ext, cache="REPEAT", **kw)
            if pd.regime != "SUSTAINED" or not close(pd.time_s, ps.time_s):
                raise Fail("DEFAULT is not SUSTAINED with a power model")
            if ps.time_s < pb.time_s * (1 - 1e-12) or ps.clock_mhz > 2800.0:
                raise Fail("SUSTAINED faster than BURST")
            if pb.power_w <= 450.0:
                if ps.clock_mhz != 2800.0 or not close(ps.time_s, pb.time_s):
                    raise Fail("SUSTAINED below the cap differs from BURST")
            elif ps.clock_mhz > 1400.0 and not close(ps.power_w, 450.0, 1e-9):
                raise Fail(f"SUSTAINED power {ps.power_w} W at {ps.clock_mhz} MHz")
            cold, rep, warm = (host.predict(spec, ext, regime="BURST", cache=c, **kw).time_s
                               for c in ("COLD", "REPEAT", "WARM"))
            if not (cold >= rep * (1 - 1e-12) and rep >= warm * (1 - 1e-12)):
                raise Fail(f"cache order {cold} {rep} {warm}")
            lo, hi = extra[("band.call_small", "band.call_mid", "band.call_large")[
                0 if pb.time_s < 1e-4 else 1 if pb.time_s < 1e-3 else 2]]
            if not (close(pb.time_p10_s, lo * pb.time_s) and close(pb.time_p90_s, hi * pb.time_s)):
                raise Fail("call band")
            for q in pb.phases:
                blo, bhi = extra[f"band.{q.cls}"]
                if not (close(q.p10_s, blo * q.time_s) and close(q.p90_s, bhi * q.time_s)):
                    raise Fail("phase band")
            count += 1
    # (6) rank: ties by footprint, raw_rank = the v6 time order, ORDER_TIME = that order
    for spec, ext in (("ik,jk->ij", dict(i=2048, j=1000, k=3000)),
                      ("kjbc,kiac->abij", dict(a=24, b=20, c=16, i=12, j=10, k=8))):
        kw = dict(N=[8, 14], vec_modes=("GENERATE", "RESIDENT"), packs=("NONE", "A", "B", "AB"),
                  reuse=3, calib=syn)
        by_time = host.rank(spec, ext, order="TIME", **kw)
        banded = host.rank(spec, ext, **kw)
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
                c.time_p10_s <= best.time_p90_s]
        if len(ties) != len(want) or banded[:len(ties)] != ties or not best.tie:
            raise Fail("rank ties")
        for a_, b_ in zip(ties, ties[1:]):
            if a_.total_bytes > b_.total_bytes:
                raise Fail("ties not ordered by footprint")
        if banded[len(ties):] != [c for c in by_time if not c.tie]:
            raise Fail("rank: the non-tied candidates keep the time order")
        for c in banded:
            if c.status == "SUCCESS" and not (c.time_p10_s <= c.time_s <= c.time_p90_s):
                raise Fail("candidate band")
        count += 1
    # (7) occupancy table names
    kn = host.kernel_names()
    if len(kn) != 90 or kn[0] != "k0" or kn[14] != "k5_n2_e1" or kn[-1] != "k5b_n20" or \
            len(set(kn)) != len(kn):
        raise Fail(f"kernel names {kn[:3]} ... {kn[-3:]}")
    return count


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
    cv = host.Calibration.from_values(cal.meta, vals)
    if cv.params != vals or host.Calibration.parse(cv.format()).params != vals:
        raise Fail("from_values / round trip of arbitrary values")
    for bad in ("format = 1\n", cal.format().replace("k5.op =", "k5.ops ="),
                cal.format() + "gpu = again\n", cal.format().replace("kl.byte = ", "kl.byte = -")):
        try:
            host.Calibration.parse(bad)
            raise Fail(f"accepted an invalid calibration: {bad[-40:]!r}")
        except ValueError:
            pass
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
    for spec, ext, along in cases:
        for N, g in ((8, 0), (14, 4), (20, 7), (3, 1)):
            for vec in ("GENERATE", "RESIDENT"):
                for mode in ("EMULATE", "DIRECT"):
                    lw = host.lowering(spec, ext, N=N, g=g, vec_mode=vec, mode=mode)
                    p = host.predict(spec, ext, N=N, g=g, vec_mode=vec, mode=mode)
                    if p.status != lw.status:
                        raise Fail(f"{spec} {ext}: status {p.status} vs lowering {lw.status}")
                    if p.status != "SUCCESS":
                        continue
                    want = spec_phases(lw, vec, along, cal.meta["l2_bytes"], occ=occ)
                    got = [(q.name, q.launches, list(q.x)) for q in p.phases]
                    if along != (False, False):  # the chains' transcription assumes radix 1
                        for ph in want + got:
                            if ph[0] in CHAIN_SLOT and ph[0] != "KF fused":
                                ph[2][CHAIN_SLOT[ph[0]]] = 0.0
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
    # Against the shipped (fitted) calibration and a synthetic one with a large eps-path cost.
    # With a fixed eps fraction the time never drops; with the default fraction it may drop
    # when k doubles (the typical-data estimate falls from 1 to 0 as k grows, DESIGN §6c.1),
    # but never below the time without the eps terms.
    heavy = dict(cal.params)
    for key in ("kf.op_eps", "kl.op_eps", "pack.op_eps"):
        heavy[key] = 50 * (heavy[key] + 1e-4)
    mono = 0
    for which, cb in (("shipped", None), ("heavy eps", host.Calibration.from_values(cal.meta, heavy))):
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
                    floor = base.time_s - (base.eps_fraction * base.eps_s if eps is None else 0)
                    if big.time_s < floor * (1 - 1e-12):
                        raise Fail(f"monotonicity ({which}, eps {eps}): {ext} N={N} g={g}: "
                                   f"{big.time_s} < {floor} ({m},{n},{k})")
                    mono += 1
            more_n = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=min(N + 1, 20), g=0, calib=cb)
            same = host.predict("ik,jk->ij", dict(i=m, j=n, k=k), N=N, g=0,
                                k4_config=more_n.k4_config, calib=cb)
            if more_n.status == same.status == "SUCCESS" and more_n.time_s < same.time_s:
                raise Fail(f"monotonicity in N ({which}): {m},{n},{k} N={N}")
    # the eps bracket: time at fraction f = time at 0 + f eps_s; the default ramp
    for N, k in ((14, 300), (15, 4096), (16, 50000), (17, 2048), (13, 64)):
        ext = dict(i=256, j=256, k=k)
        p0, p1 = (host.predict("ik,jk->ij", ext, N=N, eps_fraction=f) for f in (0.0, 1.0))
        pd = host.predict("ik,jk->ij", ext, N=N)
        lt = math.log2(math.isqrt((math.prod(MODULI[:N]) - 1) // 2))
        want = 0.0 if lt < 53 else min(1.0, max(0.0, (lt - 0.5 * math.log2(k) - 53 + 2.5) / 1.5))
        if not close(pd.eps_fraction, want, 1e-9) or \
                not close(pd.time_s, p0.time_s + pd.eps_fraction * pd.eps_s) or \
                not close(p1.time_s, p0.time_s + p0.eps_s) or (lt < 53 and p1.eps_fraction != 0):
            raise Fail(f"eps bracket N={N} k={k}: {pd.eps_fraction} vs {want}")
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
                           packs=("NONE", "A", "B", "AB"), reuse=3, budget=budget)
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
