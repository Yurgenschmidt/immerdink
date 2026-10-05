#!/usr/bin/env python3
"""Accuracy-aware planning (DESIGN v1.12 §6f, host API 8) against spec/params.py. CPU only.

    python3 tests/test_accuracy.py --host-lib build/libimmerdink_host.so [--cases 20000] [-v]

  - imd_err_estimate vs params.error_estimate (exact rational enclosure) on random (N, k,
    rounding, dispatch, outputs, metric, statistics, basis, fail_prob): the rigorous bound and
    its unit coefficient lie in [exact, exact (1 + 2^-39)] (never below the real bound); the
    typical estimate agrees to 1e-11 relative and is <= the bound; +inf exactly where params
    has no bound; err_status = (estimate by basis <= target); INVALID_VALUE exactly where
    params raises;
  - imd_err_min_moduli == a scan of imd_err_estimate and of params over N (both bases, ranges,
    random descriptors and einsum contractions), NOT_SUPPORTED when none meets;
  - predict / rank: err_* equal imd_err_estimate at the lowering's (N, k, dispatch, b m n);
    feasibility = SUCCESS and within the budget and the estimate meets the target, with the
    infeasible bits; the order is v6's (feasible by time, infeasible SUCCESS by time, the
    rest); the first candidate is the fastest one meeting target and budget; a v7-sized rank
    request and a v7-sized prediction behave as v7 (no accuracy, no err fields written).
"""

import argparse
import ctypes as C
import math
import os
import random
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "spec"))
import params as P  # noqa: E402
from immerdink import host  # noqa: E402
from test_host import random_descs  # noqa: E402

RN_TRUNC = ("RN", "TRUNC")
INFLATE_TOL = 1 + Fraction(1, 2**39)


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def raw_estimate(h, N, k, rnd, disp, outputs, acc):
    r = host._Err()
    r.struct_size = C.sizeof(host._Err)
    st = h.lib.imd_err_estimate(N, host._ROUNDING[rnd], host.DISPATCH.index(disp), k, outputs,
                                host._accuracy_ptr(acc), C.byref(r))
    return st, r


def params_estimate(N, k, rnd, disp, outputs, acc):
    acc = acc or host.Accuracy()
    try:
        return P.error_estimate(N, k, rnd, disp, outputs, host.ERR_METRICS[acc.metric],
                                cond=acc.cond, norm_product=acc.norm_product,
                                l1_ratio_a=acc.l1_ratio_a, l1_ratio_b=acc.l1_ratio_b,
                                fail_prob=acc.fail_prob, target=acc.target)
    except ValueError:
        return None


def check_enclosed(c, enc, what):
    if enc is None:
        need(c == math.inf, f"{what}: C {c!r}, params: no bound")
        return
    lo, hi = enc
    need(math.isfinite(c) and lo <= Fraction(c) <= hi * INFLATE_TOL,
         f"{what}: C {c!r} outside [{float(lo)!r}, {float(hi * INFLATE_TOL)!r}]")


def check_typical(c, ref, what):
    if math.isinf(ref) or ref == 0:
        need(c == ref, f"{what}: typical {c!r} vs {ref!r}")
    else:
        need(abs(c / ref - 1) <= 1e-11, f"{what}: typical {c!r} vs {ref!r}")


def random_acc(rng):
    """A random Accuracy (or None), sometimes with statistics the C must reject."""
    if rng.random() < 0.1:
        return None
    metric = rng.choice(list(host.ERR_METRICS))
    bad = rng.random() < 0.08
    cond = rng.choice((1.0, 3.5, 1e3, 1e9)) if metric == "ELEMENTWISE" else 0.0
    npd = 10.0**rng.uniform(-30, 30) if metric == "ABSOLUTE" else 0.0
    l1a = rng.choice((0.0, 0.0, 1.0, rng.uniform(1, 50), 1e12))
    l1b = rng.choice((0.0, 0.0, 1.0, rng.uniform(1, 50)))
    target = rng.choice((0.0, 0.0, 10.0**rng.uniform(-20, -2)))
    fp = rng.choice((0.0, 0.0, 10.0**rng.uniform(-12, -1)))
    if bad:
        which = rng.randrange(6)
        if which == 0:
            cond, metric = rng.choice((0.0, 0.5, math.inf)), "ELEMENTWISE"
        elif which == 1:
            npd, metric = rng.choice((0.0, -1.0, math.nan)), "ABSOLUTE"
        elif which == 2:
            l1a = rng.choice((0.5, -1.0, math.nan))
        elif which == 3:
            target = rng.choice((-1e-9, math.inf, math.nan))
        elif which == 4:
            fp = rng.choice((1.0, -0.1, 2.0))
        else:
            l1b = 0.999
    return host.Accuracy(metric=metric, target=target, cond=cond, norm_product=npd,
                         l1_ratio_a=l1a, l1_ratio_b=l1b,
                         basis=rng.choice(list(host.ERR_BASES)), fail_prob=fp)


def check_estimates(h, rng, cases, verbose):
    n_inf = n_inv = n_ok = 0
    for i in range(cases):
        N = rng.randint(2, 20)
        rnd = rng.choice(RN_TRUNC)
        disp = rng.choice(("EMULATE",) * 6 + ("DIRECT", "DIRECT", "NOOP", "SCALE"))
        k = rng.choice((0, 1, 2, 3)) if rng.random() < 0.05 else int(2**rng.uniform(1, 44))
        if rng.random() < 0.05:
            k = 2**rng.randint(50, 62)                 # DIRECT k u >= 1/2, huge kappa
        outputs = rng.choice((0, 1, int(2**rng.uniform(0, 40))))
        acc = random_acc(rng)
        st, r = raw_estimate(h, N, k, rnd, disp, outputs, acc)
        ref = params_estimate(N, k, rnd, disp, outputs, acc)
        what = f"case {i}: N={N} k={k} {rnd} {disp} M={outputs} {acc}"
        if ref is None:
            need(st == 1, f"{what}: params rejects, C status {st}")
            n_inv += 1
            continue
        need(st == 0 and r.status == 0, f"{what}: C status {st}")
        need((r.num_moduli, r.k, r.outputs, r.dispatch) == (N, k, outputs, host.DISPATCH.index(disp)),
             what + " echo")
        check_enclosed(r.bound, ref["bound"], what + " bound")
        check_enclosed(r.unit_bound, ref["unit_bound"], what + " unit_bound")
        check_typical(r.typical, ref["typical"], what)
        check_typical(r.unit_typical, ref["unit_typical"], what + " unit")
        need(r.typical <= r.bound and r.unit_typical <= r.unit_bound, what + " typical > bound")
        tgt = acc.target if acc else 0.0
        if tgt > 0:
            v = r.typical if acc.basis == "TYPICAL" else r.bound
            need(r.err_status == (1 if v <= tgt else 2), what + f" err_status {r.err_status}")
        else:
            need(r.err_status == 0, what + " err_status without target")
        n_inf += r.bound == math.inf
        n_ok += 1
    # argument checks params does not see
    for args in ((1, 10, "RN", "EMULATE", 1), (21, 10, "RN", "EMULATE", 1),
                 (8, -1, "RN", "EMULATE", 1), (8, 10, "RN", "EMULATE", -1)):
        need(raw_estimate(h, *args, None)[0] == 1, f"INVALID_VALUE expected for {args}")
    bad = host._Err()
    bad.struct_size = C.sizeof(host._Err) - 8
    need(h.lib.imd_err_estimate(8, 0, 1, 10, 1, None, C.byref(bad)) == 1, "short imd_err_t")
    short = host.Accuracy(target=1e-9)
    short._c.struct_size = 56
    need(raw_estimate(h, 8, 10, "RN", "EMULATE", 1, short)[0] == 1, "short imd_err_request_t")
    if verbose:
        print(f"  imd_err_estimate: {cases} cases ({n_ok} valid, {n_inf} without a bound, "
              f"{n_inv} invalid requests) vs params.error_estimate")
    need(n_ok > cases // 2 and n_inv > cases // 50 and n_inf > 0, "case mix")


SPECS = [("ik,kj->ij", "ijk"), ("kjbc,kiac->abij", "abcijk"), ("Qia,Qjb->iajb", "Qabij"),
         ("bk,k->b", "bk"), ("ij,ij->ij", "ij"), ("bik,bkj->bij", "bijk")]


def random_contraction(rng):
    if rng.random() < 0.5:
        return tuple(host.Desc(*map(tuple, d)) for d in random_descs(rng, legal_bias=0.95)), None
    spec, labs = rng.choice(SPECS)
    ext = {l: rng.choice((1, 2, 7, 16, 64, 300)) for l in labs}
    if "k" in ext and rng.random() < 0.3:
        ext["k"] = rng.choice((3000, 70000, 2**20))
    return spec, ext


def check_min_moduli(h, rng, cases, verbose):
    found = none = errs = 0
    for i in range(cases):
        spec, ext = random_contraction(rng)
        rnd, mode = rng.choice(RN_TRUNC), rng.choice(("EMULATE",) * 3 + ("DIRECT",))
        acc = host.Accuracy(metric=rng.choice(list(host.ERR_METRICS)),
                            target=10.0**rng.uniform(-18, -3), cond=rng.choice((1.0, 1e4)),
                            norm_product=rng.choice((1.0, 1e-6)),
                            l1_ratio_a=rng.choice((0.0, 3.0)),
                            basis=rng.choice(list(host.ERR_BASES)))
        n_lo, n_hi = rng.choice(((0, 0), (0, 0), (5, 11), (2, 2), (14, 20)))
        e = host.min_moduli(spec, ext, accuracy=acc, rounding=rnd, mode=mode, n_lo=n_lo,
                            n_hi=n_hi, lib=h)
        what = f"min_moduli case {i}: {spec} {ext} {rnd} {mode} {acc} [{n_lo}, {n_hi}]"
        lw = host.lowering(spec, ext, N=20, g=1, rounding=rnd, mode=mode, lib=h)
        if lw.status != "SUCCESS":
            need(e.status == lw.status and e.num_moduli == 0, f"{what}: {e.status} vs {lw.status}")
            errs += 1
            continue
        M = lw.b * lw.m * lw.n
        lo, hi = n_lo or 2, n_hi or 20
        want = 0
        for N in range(lo, hi + 1):
            x = host.err_estimate(N, lw.k, rounding=rnd, dispatch=lw.dispatch, outputs=M,
                                  accuracy=acc, lib=h)
            ref = params_estimate(N, lw.k, rnd, lw.dispatch, M, acc)
            pv = ref["typical"] if acc.basis == "TYPICAL" else (
                math.inf if ref["bound"] is None else float(ref["bound"][1]))
            if x.err_status == "MET":
                # params agrees up to the stated tolerance (a target between the two is a tie)
                need(pv <= acc.target * (1 + 1e-10), f"{what}: N={N} C meets, params {pv!r}")
                want = N
                break
            need(not (pv * (1 + 1e-10) < acc.target), f"{what}: N={N} params meets, C not")
        if want:
            need(e.status == "SUCCESS" and e.num_moduli == want and e.err_status == "MET",
                 f"{what}: {e} want N={want}")
            found += 1
        else:
            need(e.status == "NOT_SUPPORTED" and e.num_moduli == 0 and e.err_status == "MISSED",
                 f"{what}: {e} want none")
            none += 1
        need(e.k == lw.k and e.outputs == M and e.dispatch == lw.dispatch, what + " echo")
    # a request without a target / an empty range: INVALID_VALUE
    spec, ext = "ik,kj->ij", dict(i=8, j=8, k=100)
    for acc, rng_ in ((host.Accuracy(), (0, 0)), (host.Accuracy(target=1e-9), (12, 8)),
                      (host.Accuracy(target=1e-9), (1, 8)), (host.Accuracy(target=1e-9), (2, 21))):
        e = host.min_moduli(spec, ext, accuracy=acc, n_lo=rng_[0], n_hi=rng_[1], lib=h)
        need(e.status == "INVALID_VALUE", f"min_moduli {acc} {rng_}: {e.status}")
    if verbose:
        print(f"  imd_err_min_moduli: {cases} cases ({found} found, {none} none, {errs} with a "
              "lowering status) == scan of imd_err_estimate and params")


def check_predict_rank(h, rng, cases, verbose):
    total = 0
    for i in range(cases):
        spec, ext = random_contraction(rng)
        rnd, mode = rng.choice(RN_TRUNC), rng.choice(("EMULATE",) * 4 + ("DIRECT",))
        acc = host.Accuracy(metric=rng.choice(list(host.ERR_METRICS)),
                            target=rng.choice((0.0, 10.0**rng.uniform(-17, -6))),
                            cond=rng.choice((1.0, 1e5)), norm_product=rng.choice((1.0, 2.5)),
                            basis=rng.choice(list(host.ERR_BASES)))
        Ns = sorted(rng.sample(range(2, 21), rng.randint(1, 5)))
        budget = rng.choice((0, 0, 2**22, 2**28))
        what = f"rank case {i}: {spec} {ext} {rnd} {mode} {acc} N={Ns} budget={budget}"
        lw = host.lowering(spec, ext, N=20, g=1, rounding=rnd, mode=mode, lib=h)
        try:
            cands = host.rank(spec, ext, N=Ns, g=[1, 2], vec_modes=("GENERATE", "RESIDENT"),
                              packs=("NONE", "A"), budget=budget, rounding=rnd, mode=mode,
                              accuracy=acc, lib=h)
        except host.ImmerdinkError as e:
            need(lw.status != "SUCCESS", f"{what}: raised {e} on a legal contraction")
            continue
        M = lw.b * lw.m * lw.n
        seen = 0   # 0: feasible block, 1: infeasible SUCCESS, 2: the rest
        for c in cands:
            if c.status == "SUCCESS":
                x = host.err_estimate(c.num_moduli, lw.k, rounding=rnd, dispatch=c.dispatch,
                                      outputs=M, accuracy=acc, lib=h)
                need((c.err_bound, c.err_typical, c.err_status) ==
                     (x.bound, x.typical, x.err_status), f"{what}: {c} vs {x}")
                bits = (2 if budget and c.total_bytes > budget else 0) | \
                    (4 if x.err_status == "MISSED" else 0)
                need(c.infeasible == bits and c.feasible == (bits == 0),
                     f"{what}: infeasible {c.infeasible} want {bits}")
                grp = 0 if c.feasible else 1
            else:
                need(c.infeasible == 1 and not c.feasible and c.err_bound == math.inf,
                     f"{what}: non-SUCCESS candidate {c}")
                grp = 2
            need(grp >= seen, f"{what}: order (group {grp} after {seen})")
            seen = grp
        ok = [c for c in cands if c.feasible]
        if ok:
            need(cands[0] is ok[0] and all(ok[0].time_s <= c.time_s for c in ok), what + " best")
        need(all(a.time_s <= b.time_s for a, b in zip(ok, ok[1:])), what + " time order")
        # predict: the same estimates as the candidate of that configuration
        if lw.status == "SUCCESS":
            N = rng.choice(Ns)
            p = host.predict(spec, ext, N=N, rounding=rnd, mode=mode, accuracy=acc, lib=h)
            if p.status == "SUCCESS":
                x = host.err_estimate(N, lw.k, rounding=rnd, dispatch=p.dispatch, outputs=M,
                                      accuracy=acc, lib=h)
                need((p.err_bound, p.err_typical, p.err_status, p.err_metric) ==
                     (x.bound, x.typical, x.err_status, acc.metric), f"{what}: predict {p}")
            p0 = host.predict(spec, ext, N=N, rounding=rnd, mode=mode, lib=h)
            if p0.status == "SUCCESS":
                x0 = host.err_estimate(N, lw.k, rounding=rnd, dispatch=p0.dispatch, outputs=M,
                                       lib=h)
                need((p0.err_bound, p0.err_status, p0.err_metric) ==
                     (x0.bound, "UNCHECKED", "NORMWISE"), f"{what}: predict without accuracy")
        total += len(cands)
    check_v7_sizes(h)
    if verbose:
        print(f"  predict / rank: {cases} cases, {total} candidates; v7-sized structs as v7")


def check_v7_sizes(h):
    """A v7-sized rank request ignores accuracy (all budget-feasible); a v7-sized imd_perf_t
    gets no err fields; an invalid accuracy is INVALID_VALUE for v8 requests only."""
    spec, ext = "ik,kj->ij", dict(i=64, j=64, k=4096)
    h_, call = host._prep(spec, ext, None, 8, 0, "RN", "EMULATE", "GENERATE", h)
    acc = host.Accuracy(target=1e-30)                       # no N meets it
    narr = (C.c_int32 * 2)(8, 14)
    for size, want_feasible in ((88, True), (96, False)):
        req = host._RankRequest(size, C.sizeof(host._Candidate), 2, 0,
                                C.cast(narr, C.POINTER(C.c_int32)), None, 0, 0, 1, 0, 0, -1.0,
                                0, 0, 0, 0, 0.0, acc._ptr)
        out = (host._Candidate * 64)()
        n = C.c_int32(0)
        need(h.lib.imd_perf_rank(*call.args(), C.byref(req), None, out, 64, C.byref(n)) == 0,
             f"rank size {size}")
        need(all(bool(c.feasible) == want_feasible for c in out[:n.value]), f"v{size} feasibility")
    bad = host.Accuracy(metric="ABSOLUTE")                  # no norm_product: invalid
    for size, want in ((40, 0), (48, 1)):
        req = host._PerfRequest(size, 0, -1, 0, -1.0, 0, 0, 0.0, bad._ptr)
        r = host._Perf()
        r.struct_size = 1952
        r.err_bound = 12345.0
        st = h.lib.imd_perf_predict(*call.args(), None, C.byref(req), None, C.byref(r))
        need(st == want, f"predict request size {size}: status {st}")
        if st == 0:
            need(r.err_bound == 12345.0, "a v7-sized imd_perf_t got err fields")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host-lib", required=True)
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    h = host.load(a.host_lib)
    rng = random.Random(a.seed)
    try:
        check_estimates(h, rng, a.cases, a.verbose)
        check_min_moduli(h, rng, max(a.cases // 20, 50), a.verbose)
        check_predict_rank(h, rng, max(a.cases // 50, 30), a.verbose)
    except Fail as e:
        print(f"FAIL {e}")
        return 1
    print("PASS accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
