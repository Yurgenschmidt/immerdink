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
                ("vec", C.c_int32)]


LOWERING_FIELDS = ["status", "lowered", "dispatch", "swapped", "num_moduli", "residue_group",
                   "num_groups", "m", "n", "k", "b", "n_hat", "k_hat", "vec_entries",
                   "k_chunks", "gemm_launches", "kappa", "workspace_bytes",
                   "plan_device_bytes", "plan_host_bytes"]


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

    def lowering(self, descs, opts):
        raw = (C.c_uint8 * 136)()
        C.c_uint32.from_buffer(raw, 0).value = 136
        st = self.lib.imd_lowering(*self._args(descs, opts), raw)
        i32 = [C.c_int32.from_buffer(raw, 4 * i).value for i in range(8)]
        i64 = [C.c_int64.from_buffer(raw, 32 + 8 * i).value for i in range(9)]
        u64 = [C.c_uint64.from_buffer(raw, 104 + 8 * i).value for i in range(4)]
        return st, dict(zip(LOWERING_FIELDS, i32[1:] + i64 + u64))


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
    return status, exp


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

    # Python wrapper == direct C, for every query.
    st_c, low_c = direct.lowering(descs, co)
    lw = host.lowering(descs, **kw)
    if lw.status != STATUS[st_c]:
        raise Fail(f"{what}: wrapper status {lw.status}, C {STATUS[st_c]}")
    for f in LOWERING_FIELDS[1:]:
        got = getattr(lw, f)
        want = low_c[f]
        if f == "dispatch":
            want = host.DISPATCH[want]
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
              "residue_group", "vec_entries", "num_groups", "k_chunks", "gemm_launches"):
        if low_c[f] != exp[f]:
            raise Fail(f"{what}: {f} = {low_c[f]}, spec {exp[f]}")
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


def suite_host(args):
    direct = Direct(args.host_lib)
    host.load(args.host_lib)  # the wrapper's default library: the one under test
    os.environ["IMMERDINK_HOST_LIB"] = args.host_lib
    rng = random.Random(20261004)
    by = {}
    # Random descriptors and options.
    for t in range(args.cases):
        descs, opts = random_descs(rng), random_opts(rng)
        st, _ = check_case(direct, descs, opts, f"random[{t}]")
        by[st] = by.get(st, 0) + 1
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
              f"boundary, {len(big)} K4-split shapes, einsum front end, g_tradeoff")


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
    hdl = C.c_void_p()
    assert lib.lib.imd_create(C.byref(hdl)) == 0
    rng = random.Random(20261005)
    n_plans, n_k4 = 0, 0
    path_of = {0: "NOOP", 1: "EMULATE", 2: "DIRECT"}  # imd::detail::Path order (k = 0: SCALE)

    def one(descs, opts, what):
        nonlocal n_plans, n_k4
        co = c_opts(*opts)
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
            dev, store, add, launches, hbytes = v[27:32]
            _, ltot, lay = direct.layout(descs, co)
            _, pdev, phost = direct.plan_bytes(descs, co)
            dsz = C.c_size_t(0)
            assert lib.lib.imd_plan_device_bytes(plan, C.byref(dsz)) == 0
            got = dict(dispatch="SCALE" if k == 0 and path else path_of[path], swapped=sw, m=m, n=n, k=k, b=b, n_hat=nh,
                       k_hat=kh, num_moduli=N, residue_group=g, workspace_bytes=total,
                       gemm_launches=launches, plan_device_bytes=dev, plan_host_bytes=hbytes)
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

    for t in range(args.cases):
        descs = random_descs(rng)
        one(descs, random_opts(rng), f"random[{t}]")
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
        print(f"  ok  {n_plans} plans ({n_k4} emulated) sized exactly as the host queries say")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", required=True)
    ap.add_argument("--lib", help="libimmerdink.so (suite plan)")
    ap.add_argument("--suite", choices=("host", "plan"), required=True)
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.cases is None:
        args.cases = 4000 if args.suite == "host" else 400
    try:
        (suite_host if args.suite == "host" else suite_plan)(args)
    except Fail as e:
        print(f"FAIL {e}")
        return 1
    print(f"PASS {args.suite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
