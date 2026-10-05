#!/usr/bin/env python3
"""python/immerdink/planner.py (DESIGN §6e) against python/immerdink/host.py. CPU only.

    python3 tests/test_planner.py --host-lib build/libimmerdink_host.so [--cases 3000] [-v]

The fast paths must return BIT-IDENTICAL results to host.py on random descriptors (legal and
illegal; every vec_mode, mode, rounding, g; packed operands, mismatched packs) and random
requests (beta, alpha_zero, k4_config, eps_fraction, synthetic calibrations):
  - Query.workspace_size / plan_bytes / pack_desc_for: equal values, or the same status raised;
  - Query.lowering / predict / predict_pack: .to_*() == host.py's dataclass (floats compared
    by their bits) and every lazy attribute == the dataclass attribute;
  - Query.rank: Ranking.to_list() == host.rank (order, footprints, statuses, bits), one C
    call (counted); Ranking.best(b) on a budget-free ranking == rank(budget=b)[0];
  - calib=None (the cached imd_perf_calib_default passed explicitly) == host.py's NULL;
  - Planner: one Query per contraction (spec + extents + strides, raw descriptors, Layouts),
    bounded; Memo: hits return the stored object, bounded FIFO, unhashable arguments bypass;
  - threads: one Planner + Memo (small, so it evicts) and one Query shared by 8 threads give
    exactly the serial results; the memo never exceeds its bound;
  - argument errors: ValueError for bad option names, ImmerdinkError statuses as host.py.
"""

import argparse
import dataclasses
import os
import random
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from immerdink import host, planner  # noqa: E402
from immerdink.device import Layout  # noqa: E402
from test_host import random_descs, random_opts  # noqa: E402


class Fail(Exception):
    pass


def bits(x):
    """A comparable form with floats by their bits (0.0 != -0.0, NaN == NaN)."""
    if isinstance(x, float):
        return ("f", x.hex())
    if dataclasses.is_dataclass(x):
        return (type(x).__name__,) + tuple(bits(getattr(x, f.name)) for f in dataclasses.fields(x))
    if isinstance(x, (tuple, list)):
        return tuple(bits(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, bits(v)) for k, v in x.items()))
    return x


def same(a, b, what):
    if bits(a) != bits(b):
        raise Fail(f"{what}:\n  fast {a!r}\n  host {b!r}")


def same_view(view, dc, what):
    """Every lazy attribute of a view == the host.py dataclass's."""
    for f in dataclasses.fields(dc):
        if bits(getattr(view, f.name)) != bits(getattr(dc, f.name)):
            raise Fail(f"{what}: .{f.name} {getattr(view, f.name)!r} != {getattr(dc, f.name)!r}")


def outcome(fn):
    """('ok', value) or ('err', status) of a call."""
    try:
        return ("ok", fn())
    except host.ImmerdinkError as e:
        return ("err", e.status)


def to_desc(d):
    return host.Desc(*map(tuple, d))


def random_calib(rng, base):
    params = {k: v * rng.uniform(0.5, 2.0) for k, v in base.params.items()}
    return host.Calibration.from_values(base.meta, params, lib=base._lib)


def random_packing(rng, descs, opts, lib):
    """None, or (PackSpec or None) x 2 from pack_desc_for (sometimes for another N or
    rounding: a PACK_MISMATCH)."""
    if rng.random() < 0.6:
        return None
    N, g, rnd, mode, vec = opts
    out = []
    for x in (0, 1):
        if rng.random() < 0.4:
            out.append(None)
            continue
        Np = N if rng.random() < 0.85 else rng.randint(2, 20)
        try:
            out.append(host.pack_desc_for(descs, operand=x, N=Np, g=g, rounding=rnd, mode=mode,
                                          vec_mode=vec, lib=lib))
        except host.ImmerdinkError:
            out.append(None)
    return tuple(out)


# ------------------------------------------------------------------------------------------

def check_single(rng, lib, cals, cases, verbose):
    n_ok = n_err = 0
    for i in range(cases):
        descs = tuple(to_desc(d) for d in random_descs(rng))
        opts = random_opts(rng)
        N, g, rnd, mode, vec = opts
        kw = dict(N=N, g=g, rounding=rnd, mode=mode, vec_mode=vec, lib=lib)
        q = planner.Query(descs, lib=lib)
        fo = dict(rounding=rnd, mode=mode)
        what = f"case {i} {descs} {opts}"
        # sizes
        a = outcome(lambda: q.workspace_size(N, g, vec, **fo))
        b = outcome(lambda: host.workspace_size(descs, **kw))
        same(a, b, what + " workspace_size")
        n_ok += a[0] == "ok"
        n_err += a[0] == "err"
        same(outcome(lambda: q.plan_bytes(N, g, vec, **fo)),
             outcome(lambda: host.plan_bytes(descs, **kw)), what + " plan_bytes")
        sms = rng.choice((0, 0, 46, 128, 132))
        lw = q.lowering(N, g, vec, sm_count=sms, **fo)
        hl = host.lowering(descs, sm_count=sms, **kw)
        same(lw.to_lowering(), hl, what + " lowering")
        same_view(lw, hl, what + " lowering view")
        for x in (0, 1):
            same(outcome(lambda: q.pack_desc_for(x, N, g, vec, **fo)),
                 outcome(lambda: host.pack_desc_for(descs, operand=x, **kw)),
                 what + f" pack_desc_for({x})")
        # packed queries
        packing = random_packing(rng, descs, opts, lib)
        if packing is not None:
            same(outcome(lambda: q.workspace_size(N, g, vec, packing=packing, **fo)),
                 outcome(lambda: host.workspace_size(descs, packing=packing, **kw)),
                 what + f" workspace_size_packed {packing}")
            lw = q.lowering(N, g, vec, packing=packing, **fo)
            same(lw.to_lowering(), host.lowering(descs, packing=packing, **kw),
                 what + " lowering_packed")
        # predictions: default request and calibration (host.py passes NULL), random ones
        p = q.predict(N, g, vec, packing=packing, **fo)
        hp = host.predict(descs, packing=packing, **kw)
        same(p.to_prediction(), hp, what + " predict")
        same_view(p, hp, what + " predict view")
        if hp.phases:
            same(p.phase(hp.phases[-1].name), hp.phase(hp.phases[-1].name), what + " phase")
        req = dict(beta=rng.random() < 0.5, alpha_zero=rng.random() < 0.2,
                   k4_config=rng.choice((None, None, 0, 1, 2, 3)),
                   eps_fraction=rng.choice((None, 0.0, 1.0, rng.random())))
        cal = rng.choice(cals)
        hp = host.predict(descs, packing=packing, calib=cal, **req, **kw)
        same(q.predict(N, g, vec, packing=packing, calib=cal, **req, **fo).to_prediction(), hp,
             what + f" predict {req}")
        same(q.predict(N, g, vec, packing=packing, calib=cal, request=planner.Request(**req),
                       **fo).to_prediction(), hp, what + f" predict Request {req}")
        if packing is not None:
            for pk in packing:
                if pk is not None:
                    eps = rng.choice((None, 0.5))
                    same(q.predict_pack(pk, eps_fraction=eps, calib=cal).to_prediction(),
                         host.predict_pack(pk, eps_fraction=eps, calib=cal, lib=lib),
                         what + " predict_pack")
    if verbose:
        print(f"  single queries: {cases} cases ({n_ok} sized, {n_err} with an error status)")
    if n_ok < cases // 4:
        raise Fail(f"too few legal cases ({n_ok})")


def random_rank_args(rng):
    Ns = rng.sample(range(2, 21), rng.randint(1, 3))
    if rng.random() < 0.03:
        Ns.append(rng.choice((1, 21)))           # INVALID_VALUE
    N = Ns[0] if len(Ns) == 1 and rng.random() < 0.5 else Ns
    g = None if rng.random() < 0.5 else sorted(rng.sample(range(1, 21), rng.randint(1, 4)))
    if g is not None and rng.random() < 0.03:
        g.append(0)                              # INVALID_VALUE
    vm = tuple(rng.sample(("GENERATE", "RESIDENT"), rng.randint(1, 2)))
    pk = tuple(rng.sample(("NONE", "A", "B", "AB"), rng.randint(1, 4)))
    return dict(N=N, g=g, vec_modes=vm, packs=pk, reuse=rng.randint(1, 4),
                budget=rng.choice((0, 0, 2**20, 2**26, 2**30)), beta=rng.random() < 0.5,
                eps_fraction=rng.choice((None, 0.0, 0.7)))


def check_rank(rng, lib, cals, cases, verbose):
    total = 0
    fns = planner._state(lib)[0]
    real, calls = fns.perf_rank, [0]

    def counted(*a):
        calls[0] += 1
        return real(*a)
    for i in range(cases):
        descs = tuple(to_desc(d) for d in random_descs(rng, legal_bias=0.95))
        args = random_rank_args(rng)
        rnd, mode = rng.choice(("RN", "RN", "TRUNC")), rng.choice(("EMULATE", "EMULATE", "DIRECT"))
        cal = rng.choice(cals + [None])
        q = planner.Query(descs, lib=lib)
        what = f"rank case {i} {descs} {args} {rnd} {mode}"
        fns.perf_rank, calls[0] = counted, 0
        try:
            a = outcome(lambda: q.rank(rounding=rnd, mode=mode, calib=cal, **args))
        finally:
            fns.perf_rank = real
        if calls[0] != 1:
            raise Fail(f"{what}: {calls[0]} imd_perf_rank calls (one expected)")
        b = outcome(lambda: host.rank(descs, rounding=rnd, mode=mode, calib=cal, lib=lib, **args))
        if a[0] != b[0]:
            raise Fail(f"{what}: {a} vs {b}")
        if a[0] == "err":
            same(a, b, what)
            continue
        r, hr = a[1], b[1]
        same(r.to_list(), hr, what)
        if len(r) != len(hr):
            raise Fail(what + " len")
        total += len(r)
        if hr:
            same_view(r[0], hr[0], what + " [0]")
            same_view(r[-1], hr[-1], what + " [-1]")
            same([c.to_candidate() for c in r[1:3]], hr[1:3], what + " slice")
        # best(budget) on the budget-free ranking == rank(budget)[0] (when one fits)
        if args["budget"] == 0 and hr:
            for b in (2**20, 2**26, 2**30):
                hb = host.rank(descs, rounding=rnd, mode=mode, calib=cal, lib=lib,
                               **dict(args, budget=b))
                best = r.best(b)
                first = hb[0] if hb and hb[0].feasible else None
                if (best is None) != (first is None):
                    raise Fail(f"{what}: best({b}) {best} vs {first}")
                if best is not None:
                    same(best.to_candidate(), dataclasses.replace(first, feasible=True),
                         what + f" best({b})")
    if verbose:
        print(f"  rank: {cases} cases, {total} candidates")


def check_planner_memo(rng, lib, verbose):
    spec, ext = "kjbc,kiac->abij", dict(a=64, b=64, c=64, i=32, j=32, k=32)
    pl = planner.Planner(lib=lib, max_queries=4)
    q = pl.query(spec, ext)
    if pl.query(spec, dict(ext)) is not q or pl.query(spec, dict(ext, z=5)) is not q:
        raise Fail("Planner.query: not one Query per contraction")
    if pl.query(spec, dict(ext, a=65)) is q:
        raise Fail("Planner.query: other extents, same Query")
    st = ([4096 * 64, 64 * 64, 64, 1], None, None)
    qs = pl.query(spec, dict(ext, a=64), strides=st)
    if qs is q or pl.query(spec, ext, strides=st) is not qs:
        raise Fail("Planner.query: strides not in the key")
    same(qs.workspace_size(14), host.workspace_size(spec, ext, strides=st, N=14, lib=lib),
         "strided workspace_size")
    descs = host.contraction(spec, ext)
    qd = pl.query(descs)
    if pl.query(tuple((d.extent, d.stride, d.label) for d in descs)) is not qd:
        raise Fail("Planner.query: raw descriptors not normalized")
    for x in range(10):
        pl.query(spec, dict(ext, i=x + 1))
    if len(pl) > 4:
        raise Fail(f"Planner bound: {len(pl)} > 4")
    lays = [Layout(d.extent, d.stride) for d in descs]
    ql = pl.query_arrays(spec, *lays)
    if pl.query_arrays(spec, *lays) is not ql:
        raise Fail("query_arrays not cached")
    same(ql.workspace_size(14, 7), host.workspace_size(spec, ext, N=14, g=7, lib=lib),
         "query_arrays")
    same(planner.Query.from_layouts(spec, *lays, lib=lib).predict(14).to_prediction(),
         host.predict(spec, ext, N=14, lib=lib), "from_layouts")
    # Planner signatures == host.py
    for N in (8, 14, 17):
        same(pl.predict(spec, ext, N=N, beta=True).to_prediction(),
             host.predict(spec, ext, N=N, beta=True, lib=lib), "Planner.predict")
        same(pl.rank(spec, ext, N=N, budget=2**28).to_list(),
             host.rank(spec, ext, N=N, budget=2**28, lib=lib), "Planner.rank")
        same(pl.lowering(spec, ext, N=N).to_lowering(), host.lowering(spec, ext, N=N, lib=lib),
             "Planner.lowering")
        same(pl.plan_bytes(spec, ext, N=N), host.plan_bytes(spec, ext, N=N, lib=lib),
             "Planner.plan_bytes")
        same(pl.pack_desc_for(spec, ext, operand=1, N=N),
             host.pack_desc_for(spec, ext, operand=1, N=N, lib=lib), "Planner.pack_desc_for")
    # Memo
    memo = planner.Memo(8)
    pm = planner.Planner(lib=lib, memo=memo)
    a = pm.predict(spec, ext, N=14)
    if pm.predict(spec, ext, N=14) is not a or memo.hits != 1:
        raise Fail("Memo: no hit for an identical query")
    if pm.predict(spec, ext, N=14, beta=True) is a:
        raise Fail("Memo: a different query hit")
    r = pm.rank(spec, ext, N=[8, 14])
    if pm.rank(spec, ext, N=[8, 14]).to_list() != r.to_list():   # unhashable N list: bypass
        raise Fail("Memo: unhashable bypass")
    for N in range(2, 21):
        pm.workspace_size(spec, ext, N=N)
    if len(memo) > 8:
        raise Fail(f"Memo bound: {len(memo)} > 8")
    if pm.workspace_size(spec, ext, N=20) != host.workspace_size(spec, ext, N=20, lib=lib):
        raise Fail("Memo: value")
    memo.clear()
    if len(memo) or memo.hits:
        raise Fail("Memo.clear")
    # immutable views, argument errors
    p = pl.predict(spec, ext, N=14)
    for obj, attr in ((p, "time_s"), (p, "foo"), (pl.lowering(spec, ext, N=14), "m")):
        try:
            setattr(obj, attr, 1)
        except AttributeError:
            pass
        else:
            raise Fail(f"view attribute {attr} is writable")
    for bad in (dict(vec_mode="X"), dict(rounding="RZ"), dict(mode="FAST")):
        try:
            q.predict(14, **bad)
        except ValueError:
            pass
        else:
            raise Fail(f"bad option {bad} accepted")
    for fn in (lambda: q.rank([1, 14]), lambda: q.rank(14, g=[0]),
               lambda: q.workspace_size(25), lambda: q.pack_desc_for(2, 14)):
        if outcome(fn)[0] != "err":
            raise Fail("error status not raised")
    same(outcome(lambda: q.workspace_size(25)),
         outcome(lambda: host.workspace_size(spec, ext, N=25, lib=lib)), "N = 25")
    if q.predict(25).status != host.predict(spec, ext, N=25, lib=lib).status:
        raise Fail("predict status N = 25")
    for fn in (lambda: planner.Request(nonsense=1), lambda: q.rank(14, nonsense=1),
               lambda: planner.Request(flags=1)):
        try:
            fn()
        except TypeError:
            pass
        else:
            raise Fail("an unknown (or reserved) request field was accepted")
    if planner.Request(beta=True) != planner.Request(beta=True) or \
            hash(planner.Request(eps_fraction=0.5)) != hash(planner.Request(eps_fraction=0.5)):
        raise Fail("Request equality")
    if planner.default_calibration(lib) is not planner.default_calibration(lib):
        raise Fail("default_calibration not shared")
    if verbose:
        print("  Planner / Memo / views / errors: ok")


def check_threads(rng, lib, verbose, threads=8, per_thread=300):
    shapes = [("kjbc,kiac->abij", dict(a=64, b=64, c=64, i=32, j=32, k=32)),
              ("Qia,Qjb->iajb", dict(Q=256, i=16, j=16, a=64, b=64)),
              ("ik,kj->ij", dict(i=512, j=512, k=512)), ("ik,kj->ij", dict(i=16, j=16, k=16)),
              ("abef,efij->abij", dict(a=48, b=48, e=48, f=48, i=16, j=16))]
    queries = []
    for _ in range(threads * per_thread):
        spec, ext = rng.choice(shapes)
        N = rng.choice((8, 14, 17))
        g = rng.choice((0, 1))
        vec = rng.choice(("GENERATE", "RESIDENT"))
        kind = rng.choice(("ws", "p", "p", "lw", "r"))
        queries.append((kind, spec, ext, N, g, vec))

    def expect(qr):
        kind, spec, ext, N, g, vec = qr
        if kind == "ws":
            return bits(outcome(lambda: host.workspace_size(spec, ext, N=N, g=g, vec_mode=vec,
                                                            lib=lib)))
        if kind == "p":
            return bits(host.predict(spec, ext, N=N, g=g, vec_mode=vec, lib=lib))
        if kind == "lw":
            return bits(host.lowering(spec, ext, N=N, g=g, vec_mode=vec, lib=lib))
        return bits(host.rank(spec, ext, N=N, budget=2**28, lib=lib))

    def fast(pl, qr):
        kind, spec, ext, N, g, vec = qr
        if kind == "ws":
            return bits(outcome(lambda: pl.workspace_size(spec, ext, N=N, g=g, vec_mode=vec)))
        if kind == "p":
            return bits(pl.predict(spec, ext, N=N, g=g, vec_mode=vec).to_prediction())
        if kind == "lw":
            return bits(pl.lowering(spec, ext, N=N, g=g, vec_mode=vec).to_lowering())
        return bits(pl.rank(spec, ext, N=N, budget=2**28).to_list())

    want = [expect(qr) for qr in queries]
    memo = planner.Memo(32)
    shared = [planner.Planner(lib=lib, memo=memo, max_queries=3), planner.Planner(lib=lib)]
    errors = []
    over = []

    def worker(t):
        try:
            for j in range(t, len(queries), threads):
                for pl in shared:
                    got = fast(pl, queries[j])
                    if got != want[j]:
                        errors.append((queries[j], got, want[j]))
                        return
                if len(memo) > memo.maxsize:
                    over.append(len(memo))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise Fail(f"threads: {errors[0]}")
    if over or len(memo) > memo.maxsize:
        raise Fail(f"threads: memo over its bound {over}")
    if verbose:
        print(f"  threads: {threads} x {per_thread} queries on a shared Planner + Memo "
              f"({memo.hits} hits) and a shared Planner: identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", required=True)
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lib = host.load(a.host_lib)
    rng = random.Random(a.seed)
    base = host.Calibration.default(lib)
    cals = [base, random_calib(rng, base), random_calib(rng, base)]
    try:
        check_single(rng, lib, cals, a.cases, a.verbose)
        check_rank(rng, lib, cals, max(a.cases // 5, 20), a.verbose)
        check_planner_memo(rng, lib, a.verbose)
        check_threads(rng, lib, a.verbose)
    except Fail as e:
        print(f"FAIL {e}")
        return 1
    print("PASS planner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
