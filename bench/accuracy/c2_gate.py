#!/usr/bin/env python3
"""M-a, the combined C2 study (NOTES-C1 §4.3, protocol IR1): K-equilibration (oracle.kweights'
rule, guarded), K-segmentation and the envelope rule against fast mode, at realistic sizes,
in exact integer emulation (bench/accuracy/kw_engine.py, bit-identical to spec/oracle.py).
CPU only, single-threaded; no GPU.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 \\
      tools/memcap.sh python3 bench/accuracy/c2_gate.py TIER [options] --out DIR

TIER: selftest | pilot | S | L | real | multi | batch | b6 (options: --families, --seeds,
--sizes, --dstar, --real-dir, --b6). Records: one JSON line per run in DIR/<tier>.jsonl (raw
records belong in immerdink-eval results/c1/m_a/, untracked); c2_summary.py turns them into
tables. Wrap the timeout INSIDE memcap (tools/memcap.sh timeout 600 python3 ...): memcap runs
the command in a systemd scope that a timeout around memcap.sh does not stop.

Per run (§4.3 "Engine"):
  1. full operands, exact, over every row and column: the exponent extrema per shared K
     index, Delta, the median, w and the clamp counts per scheme (and a round-trip check
     that every scaled entry is exact: gate (iii)); sigma of every row of A, A~, B, B~ at
     every N; gain_row_min, gain_col_min, G and the guard at N_ref in {8, 14} (a run whose
     guard differs between them is flagged); the worst pair W = min_i log2(||a_i|| /
     ||a~_i||) + min_j log2(||b_j|| / ||b~_j||) (FP64 norms: a diagnostic); percentiles of
     sigma~ - sigma;
  2. sampled outputs, 64 x 64 (all outputs when m = n <= 128): up to 4 rows argmin
     (sigma~ - sigma) at N_ref = 14 and up to 4 argmin log2(||a|| / ||a~||) per K-eq scheme,
     then seeded random rows up to 64; columns the same way. On the sample: exact x and the
     emulated D of every scheme at every N = 4..20 (RN) and N in {8, 14} (TRUNC), with the
     CRT precondition sum|a'||b'| <= H asserted on every output of every scheme;
  3. per target, the smallest N whose sampled max-elementwise (1e-6 .. 1e-12) and
     max-normwise (relative to the PLAIN norms; 1e-8 .. 1e-15) error meets it; realized
     regressions of guard-accepted K-eq (outputs whose error exceeds fast's at the same N by
     more than 2x, against the floor u|x|); the L5d request (what min_moduli picks for the
     weighted call from the plain statistics) against the measured minimum; [M] time and
     TFLOPS-eq from imd_perf_predict of this worktree's library with every cost term.

Schemes: fast; keq_d<d> (K-eq, deadband d); kseg<S> (K split into S contiguous bands in
canonical order, each its own sigma / tau and plan k, accumulated in FP64 as beta = 1 does);
keq_d<d>+kseg2; env (Appendix B's envelope rule, FP64-evaluated with outward rounding, never
above the exact rule); sc25 (exact-integer [SC25], m = n = 128 only: it needs a full GEMM).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kw_engine as ke  # noqa: E402
import c2_fixtures as fx  # noqa: E402

o = ke.o
ROOT = ke.ROOT
U = ke.U

NS = list(range(4, 21))                       # RN: every N = 4..20
NS_TRUNC = [8, 14]
N_REF = (8, 14)
EW_TARGETS = [1e-6, 1e-8, 1e-10, 1e-12]       # max elementwise
NW_TARGETS = [1e-8, 1e-10, 1e-12, 1e-15]      # max normwise, relative to the plain norms
GATE_EW, GATE_NW = (1e-8, 1e-10), (1e-10, 1e-12)
SAMPLE = 64
ITERS = 16                                    # the gate's run length (helper amortization)


# ---------------------------------------------------------------------------------------
# Library, provenance, performance model
# ---------------------------------------------------------------------------------------

def load_host():
    """immerdink.host loaded from this worktree's python/ (by file: the package __init__
    imports the device module) on this worktree's build/libimmerdink_host.so."""
    path = os.path.join(ROOT, "python", "immerdink", "host.py")
    spec = importlib.util.spec_from_file_location("imd_host_c2", path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    lib = os.path.join(ROOT, "build", "libimmerdink_host.so")
    h = host.load(lib)
    return host, h


def provenance(h):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    with open(h.path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return dict(commit=commit, lib=os.path.relpath(h.path, ROOT), lib_sha256=sha,
                host_api=int(h.version))


class Perf:
    """[M] predictions (imd_perf_predict: DEFAULT regime, COLD, GENERATE, g = N, beta = 0,
    in-line operands) and the [E] bandwidth-based costs of the caller-side pieces, from the
    shipped calibration: an FP64 operand stream at kl.byte ns per DRAM byte (KL's fitted
    rate for reading FP64 operands) and pack.launch ns per launch."""

    def __init__(self, host, h):
        self.host, self.h, self.cache = host, h, {}
        cal = host.Calibration.default(h)
        p = cal.params() if callable(cal.params) else cal.params
        self.byte_ns, self.launch_ns = p["kl.byte"], p["pack.launch"]

    def predict(self, spec, ext, N, beta=False):
        key = (spec, tuple(sorted(ext.items())), N, beta)
        if key not in self.cache:
            r = self.host.predict(spec, ext, N=N, beta=beta, lib=self.h)
            assert r.status == "SUCCESS", (spec, ext, N, r.status)
            self.cache[key] = (r.time_s, r.flops_eq)
        return self.cache[key]

    def stream_s(self, elems, passes):
        """passes x 8 B x elems at the FP64 operand rate."""
        return passes * 8 * elems * self.byte_ns * 1e-9

    def min_moduli(self, m, n, k, acc):
        r = self.host.min_moduli("ik,kj->ij", dict(i=m, k=k, j=n), accuracy=acc, lib=self.h)
        return r.num_moduli if r.status == "SUCCESS" else None


def kw_costs(perf, mA, mB, scenario):
    """Per-call [E] cost (s) of K-eq's caller-side pieces for operands of mA and mB elements
    (NOTES-C1 §2.6, §4.5): the helper with the guard (two reads of A and B, 4 launches) and the
    unfused pre-scaling (16 B per element per changed operand, 1 launch each). scenario:
    "invariant" (both operands fixed over the 16-iteration run: helper and scaled copies made
    once, amortized), "A" / "B" / "both" (that operand rebuilt every iteration: the helper
    re-runs per call, the rebuilt operands are re-scaled per call)."""
    helper = perf.stream_s(mA + mB, 2) + 4 * perf.launch_ns * 1e-9
    pre = {"A": perf.stream_s(mA, 2) + perf.launch_ns * 1e-9,
           "B": perf.stream_s(mB, 2) + perf.launch_ns * 1e-9}
    if scenario == "invariant":
        return (helper + pre["A"] + pre["B"]) / ITERS
    changed = ["A", "B"] if scenario == "both" else [scenario]
    return helper + sum(pre[x] for x in changed)


# ---------------------------------------------------------------------------------------
# Schemes on the sample
# ---------------------------------------------------------------------------------------

def plan_of(N, k, rnd, cache={}):
    key = (N, k, rnd)
    if key not in cache:
        cache[key] = o.Plan(N, k, rnd)
    return cache[key]


def keys():
    return [(N, "RN") for N in NS] + [(N, "TRUNC") for N in NS_TRUNC]


def segmented(As, Bs, S, N, rnd, ratios, cache):
    """K-segmentation (NOTES-C1 §2.7, L7): S contiguous bands, each its own sigma / tau at
    its own plan k, D_s = RN(z_s 2^-(sigma_s + tau_s)), C = RN(C + D_s) in band order.
    cache: the bands' operands and (N-independent) row statistics, per scheme."""
    k = As.shape[1]
    if not cache:
        for band in np.array_split(np.arange(k), S):
            a, b = np.ascontiguousarray(As[:, band]), np.ascontiguousarray(Bs[:, band])
            cache.append((a, ke.RowStats(a), b, ke.RowStats(b)))
    C = None
    for a, sta, b, stb in cache:
        pl = plan_of(N, a.shape[1], rnd)
        sa = ke.shifts(a, sta, pl)
        sb = ke.shifts(b, stb, pl)
        D, r = ke.emulate(a, b, sa, sb, pl, rnd)
        ratios.append(r)
        C = D if C is None else C + D
    return C


class Sample:
    """The sampled rows / columns and their exact products."""

    def __init__(self, A, Bt, rows, cols, na, nb):
        self.rows, self.cols = rows, cols
        self.As, self.Bs = np.ascontiguousarray(A[rows]), np.ascontiguousarray(Bt[cols])
        assert np.isfinite(self.As).all() and np.isfinite(self.Bs).all()
        self.hi, self.lo, _, _ = ke.exact_ref(self.As, self.Bs)
        self.na, self.nb = na[rows], nb[cols]
        xa = np.abs(self.hi)
        with np.errstate(divide="ignore"):
            self.rho = np.where(xa > 0, self.na[:, None] * self.nb[None, :] / xa, np.inf)


RHO_MUST = 1e8     # diagnostic: elementwise over the outputs with rho_ij <= 1e8 only


def evaluate(sample, D):
    """(E, max normwise, max elementwise, max elementwise over rho <= RHO_MUST). The last is a
    labelled diagnostic for operands with structural near-zero outputs (a molecule's point
    group): DESIGN §6f.1 measures an elementwise target over the outputs that must meet it."""
    E, nw, ew = ke.errors(D, sample.hi, sample.lo, sample.na, sample.nb)
    fin = ~np.isnan(ew)
    must = fin & (sample.rho <= RHO_MUST)
    return E, float(nw.max()), float(ew[fin].max()) if fin.any() else 0.0, \
        float(ew[must].max()) if must.any() else 0.0


def nmin(curve, target):
    """Smallest N (RN) whose max error <= target; None if no N <= 20 meets it."""
    for N in NS:
        if N in curve and curve[N] <= target:
            return N
    return None


# ---------------------------------------------------------------------------------------
# One GEMM-view case
# ---------------------------------------------------------------------------------------

def select(n_all, adv, rng, size):
    """Adversarial indices first (deduplicated, in order), then seeded random up to size."""
    if n_all <= size:
        return np.arange(n_all)
    out = list(dict.fromkeys(int(i) for i in adv))[:size // 2]
    rest = np.setdiff1d(np.arange(n_all), out)
    out += rng.choice(rest, size - len(out), replace=False).tolist()
    return np.array(sorted(out))


def keq_full(A, Bt, stA, stB, na, nb, sigF, tauF, w, info):
    """Step 1 for one weight vector: weighted shifts at every N, the guard at N_ref, the
    worst pair, percentiles, exactness of the scaling (gate iii)."""
    Aw, va = ke.scale(A, w, 1)
    stAw = ke.RowStats(Aw)
    sigW = {key: ke.shifts(Aw, stAw, plan_of(key[0], A.shape[1], key[1])) for key in keys()}
    naw = ke.norms(Aw)
    del Aw
    Bw, vb = ke.scale(Bt, w, -1)
    stBw = ke.RowStats(Bw)
    tauW = {key: ke.shifts(Bw, stBw, plan_of(key[0], A.shape[1], key[1])) for key in keys()}
    nbw = ke.norms(Bw)
    del Bw
    r = dict(info, w_hist=np.unique(w, return_counts=True)[1].size, fp64_violations=va + vb)
    r.pop("w0", None)
    for N in N_REF:
        ga = ke.gain_min(sigF[(N, "RN")], sigW[(N, "RN")], stA)
        gb = ke.gain_min(tauF[(N, "RN")], tauW[(N, "RN")], stB)
        G = None if ga is None or gb is None else ga + gb
        r[f"G{N}"], r[f"gain_row_min{N}"], r[f"gain_col_min{N}"] = G, ga, gb
        r[f"guard{N}"] = 1 if G is None or G >= o.KW_GUARD_MARGIN else 0
    r["guard_flag"] = r["guard8"] != r["guard14"]
    okA, okB = ~stA.zero, ~stB.zero
    la, lb = np.log2(na[okA] / naw[okA]), np.log2(nb[okB] / nbw[okB])
    r["worst_pair_bits"] = float(la.min() + lb.min())
    r["median_pair_bits"] = float(np.median(la) + np.median(lb))
    dA = (sigW[(14, "RN")] - sigF[(14, "RN")])[okA]
    dB = (tauW[(14, "RN")] - tauF[(14, "RN")])[okB]
    q = [0, 1, 5, 50, 95, 100]
    r["dsig_pct_rows"] = np.percentile(dA, q).tolist()
    r["dsig_pct_cols"] = np.percentile(dB, q).tolist()
    def adversarial(s1, s0, ok, n0, n1):
        """Up to 4 argmin (sigma~ - sigma) at N_ref = 14, then up to 4 argmin log2(||x|| / ||x~||)."""
        by_gain = np.argsort(s1 - s0 + np.where(ok, 0, 10 ** 6), kind="stable")[:4]
        by_norm = np.argsort(np.where(ok, la_full(n0, n1), np.inf), kind="stable")[:4]
        return list(by_gain) + list(by_norm)
    adv_r = adversarial(sigW[(14, "RN")], sigF[(14, "RN")], okA, na, naw)
    adv_c = adversarial(tauW[(14, "RN")], tauF[(14, "RN")], okB, nb, nbw)
    return r, sigW, tauW, adv_r, adv_c


def la_full(n0, n1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(np.where(n1 > 0, n0 / n1, 1.0))


def sc25_shifts(A, Bt, plan):
    """Exact-integer [SC25] (NOTES-C1 §2.5): Abar = ceil(|a| 2^(5 - gamma)) <= 64, Cbar =
    Abar Bbar^T exact, d_i largest with 4^d max_j Cbar_ij <= H; sigma = 5 - gamma + d."""
    def bar(X):
        g = ke.RowStats(X).gamma
        return np.ceil(np.ldexp(np.abs(X), (5 - g)[:, None])), g
    Ab, ga = bar(A)
    Bb, gb = bar(Bt)
    Cb = Ab @ Bb.T                                      # exact: <= 2^12 k < 2^53
    d = np.array([o.largest_u(int(r), plan.H) for r in Cb.max(axis=1)])
    e = np.array([o.largest_u(int(c), plan.H) for c in Cb.max(axis=0)])
    assert d.min() >= 0 and e.min() >= 0
    return 5 - ga + d, 5 - gb + e


def run_case(A, Bt, meta, cfg, perf, stale_w=None):
    """One GEMM-view case (NOTES-C1 §4.3 steps 1-4). Returns (record, {d: w})."""
    t0 = time.process_time()
    m, k = A.shape
    n = Bt.shape[0]
    rec = dict(meta, m=m, n=n, k=k)
    stA, stB = ke.RowStats(A), ke.RowStats(Bt)
    assert not stA.flag.any() and not stB.flag.any()
    na, nb = ke.norms(A), ke.norms(Bt)
    fb = []
    sigF = {key: ke.shifts(A, stA, plan_of(key[0], k, key[1]), fb) for key in keys()}
    tauF = {key: ke.shifts(Bt, stB, plan_of(key[0], k, key[1]), fb) for key in keys()}
    extA, extB = ke.extrema([A]), ke.extrema([Bt])
    keq, ws, adv_r, adv_c = {}, {}, [], []
    for d in cfg["deadbands"]:
        w, info = ke.weights(extA, extB, d)
        r, sigW, tauW, ar, ac = keq_full(A, Bt, stA, stB, na, nb, sigF, tauF, w, info)
        keq[d] = (r, sigW, tauW, w)
        ws[d] = w
        adv_r += ar
        adv_c += ac
    rec["keq"] = {f"d{d}": v[0] for d, v in keq.items()}
    rng = np.random.default_rng([meta.get("seed", 0), 4242])
    size = cfg.get("sample", SAMPLE)
    rows = select(m, adv_r, rng, size)
    cols = select(n, adv_c, rng, size)
    S = Sample(A, Bt, rows, cols, na, nb)
    rec["sample"] = [int(rows.size), int(cols.size)]

    # stale weights (seed s - 1's w on this data), step 1 + sample
    stale = {}
    if stale_w:
        for d, w in stale_w.items():
            if w.size != k or d not in cfg.get("stale_d", []):
                continue
            r, sigW, tauW, _, _ = keq_full(A, Bt, stA, stB, na, nb, sigF, tauF, w,
                                           dict(stale=True))
            keq[f"stale{d}"] = (r, sigW, tauW, w)
            stale[f"d{d}"] = r
    rec["stale"] = stale

    # every scheme on the sample
    schemes = {}
    ratios = []
    Ecache = {}
    env_vecs = {}
    segc, wsamp = {}, {}
    for key in keys():
        N, rnd = key
        pl = plan_of(N, k, rnd)
        out = {}
        D, r = ke.emulate(S.As, S.Bs, sigF[key][rows], tauF[key][cols], pl, rnd)
        ratios.append(r)
        ev = evaluate(S, D)
        Ecache[("fast", key)], out["fast"] = ev[0], ev[1:]
        for d, (r_, sigW, tauW, w) in keq.items():
            if d not in wsamp:
                wsamp[d] = (np.ldexp(S.As, w[None, :]), np.ldexp(S.Bs, -w[None, :]))
            Aws, Bws = wsamp[d]
            D, rr = ke.emulate(Aws, Bws, sigW[key][rows], tauW[key][cols], pl, rnd)
            ratios.append(rr)
            ev = evaluate(S, D)
            Ecache[(f"keq_d{d}", key)], out[f"keq_d{d}"] = ev[0], ev[1:]
            if isinstance(d, int) and d in cfg["kseg_keq"]:
                D = segmented(Aws, Bws, 2, N, rnd, ratios, segc.setdefault(f"keq{d}", []))
                out[f"keq_d{d}+kseg2"] = evaluate(S, D)[1:]
        for Sg in cfg["kseg"]:
            D = segmented(S.As, S.Bs, Sg, N, rnd, ratios, segc.setdefault(f"S{Sg}", []))
            out[f"kseg{Sg}"] = evaluate(S, D)[1:]
        if cfg.get("env"):
            s = ke.envelope_s(pl)
            EA = ke.envelope_vec(A, sigF[key], pl, s)
            EB = ke.envelope_vec(Bt, tauF[key], pl, s)
            uA, uB = ke.u_of(sigF[key], stA), ke.u_of(tauF[key], stB)
            dA = ke.envelope_d(A, sigF[key], uA, stA, EB, rows, pl, s)
            dB = ke.envelope_d(Bt, tauF[key], uB, stB, EA, cols, pl, s)
            D, rr = ke.emulate(S.As, S.Bs, sigF[key][rows] + dA, tauF[key][cols] + dB, pl, rnd)
            ratios.append(rr)
            out["env"] = evaluate(S, D)[1:]
            env_vecs[key] = (float(np.median(dA)), float(np.median(dB)))
        if cfg.get("sc25"):
            sa, sb = sc25_shifts(A, Bt, pl)
            D, rr = ke.emulate(S.As, S.Bs, sa[rows], sb[cols], pl, rnd)
            ratios.append(rr)
            out["sc25"] = evaluate(S, D)[1:]
        for name, v in out.items():
            schemes.setdefault(name, {}).setdefault(rnd, {})[N] = list(v)
    rec["crt_max_ratio"] = max(x for x in ratios if x is not None)
    rec["fallback_rows"] = len(fb)
    if env_vecs:
        rec["env_median_d"] = {f"{N}{r[0]}": v for (N, r), v in env_vecs.items()}
    rec["curves"] = {s: {r: {str(N): v for N, v in c.items()} for r, c in rr.items()}
                     for s, rr in schemes.items()}
    rec["nmin"] = {}
    for s, rr in schemes.items():
        rn = rr["RN"]
        rec["nmin"][s] = dict(ew={str(t): nmin({N: v[1] for N, v in rn.items()}, t)
                                  for t in EW_TARGETS},
                              nw={str(t): nmin({N: v[0] for N, v in rn.items()}, t)
                                  for t in NW_TARGETS},
                              ew8={str(t): nmin({N: v[2] for N, v in rn.items()}, t)
                                   for t in EW_TARGETS})

    # realized regressions of guard-accepted K-eq. (1) As pre-registered: sampled outputs
    # whose error exceeds fast's at the same N by > 2x (floor u|x|), count and max ratio.
    # Per-output errors at one N are dominated by independent rounding noise (two
    # half-normal errors whose scales differ 2x still give ~16 % of outputs above 2x), so
    # two noise-robust views are recorded next to it: (2) systematic: outputs whose MEDIAN
    # ratio over the N where fast's error is above 4 u|x| exceeds 2; (3) metric: the N at
    # which the scheme's sampled max error (elementwise or normwise) exceeds fast's by > 2x,
    # among the N where fast's own error in that metric is below 1 (not saturated).
    xa = np.abs(S.hi)
    rec["regress"] = {}
    for d in keq:
        r_ = keq[d][0]
        if r_["guard14"] != 1:
            continue
        cnt, mx, byN, R = 0, 0.0, {}, []
        for key in keys():
            Ef, Ew = Ecache[("fast", key)], Ecache[(f"keq_d{d}", key)]
            ratio = Ew / np.maximum(Ef, np.maximum(U * xa, 1e-300))
            c = int(np.sum(ratio > 2))
            cnt += c
            mx = max(mx, float(ratio.max()))
            if c:
                byN[f"{key[0]}{key[1][0]}"] = c
            if key[1] == "RN":
                R.append(np.where(Ef > 4 * U * xa, ratio, np.nan))
        R = np.array(R)
        enough = np.sum(~np.isnan(R), axis=0) >= 3
        with np.errstate(all="ignore"):
            medr = np.nanmedian(np.where(enough[None], R, np.nan), axis=0)
        sysc = int(np.sum(enough & (medr > 2)))
        cf, cw = schemes["fast"]["RN"], schemes[f"keq_d{d}"]["RN"]
        metric = sorted({N for N in NS for j in (0, 1)
                         if cf[N][j] < 1 and cw[N][j] > 2 * max(cf[N][j], U)})
        rec["regress"][f"d{d}"] = dict(count=cnt, max_ratio=mx, by_N=byN,
                                       outputs=int(S.hi.size * len(keys())),
                                       systematic=sysc, systematic_outputs=int(enough.sum()),
                                       systematic_max=float(np.nanmax(medr)) if enough.any() else None,
                                       metric_N=metric)

    # L5d: the weighted call's rigorous request from the plain statistics
    rec["l5d"] = l5d(perf, S, m, n, k, keq, rec)
    rec["perf"] = perf_block(perf, meta, m, n, k)
    rec["cpu_s"] = round(time.process_time() - t0, 2)
    rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return rec, ws


def l5d(perf, S, m, n, k, keq, rec):
    """min_moduli (BOUND) for the plain call (cond = the sample's max rho, l1 unknown) and,
    per K-eq scheme with its G at N_ref = 14, for the weighted call through L5d:
    cond' = max(1, RN(ldexp(cond, 2 - G) (1 + 2^-19))) (ELEMENTWISE) and
    t' = RN(ldexp(t, G - 2) (1 - 2^-19)) (NORMWISE relative to the plain norms)."""
    host = perf.host
    fin = np.isfinite(S.rho)
    cond = float(S.rho[fin].max()) if fin.any() else math.inf
    out = dict(cond_sample=cond, fast={}, keq={})
    for t in EW_TARGETS:
        acc = host.Accuracy(metric="ELEMENTWISE", target=t, cond=cond)
        out["fast"][f"ew{t}"] = perf.min_moduli(m, n, k, acc) if math.isfinite(cond) else None
    for t in NW_TARGETS:
        out["fast"][f"nw{t}"] = perf.min_moduli(m, n, k, host.Accuracy(metric="NORMWISE", target=t))
    for d, (r, *_rest) in keq.items():
        G = r.get("G14")
        if G is None or not math.isfinite(cond):
            continue
        res = {}
        c2 = max(1.0, math.ldexp(cond, 2 - G) * (1 + 2.0 ** -19))
        for t in EW_TARGETS:
            res[f"ew{t}"] = perf.min_moduli(m, n, k, host.Accuracy(metric="ELEMENTWISE",
                                                                    target=t, cond=c2))
        for t in NW_TARGETS:
            t2 = math.ldexp(t, G - 2) * (1 - 2.0 ** -19)
            res[f"nw{t}"] = perf.min_moduli(m, n, k, host.Accuracy(metric="NORMWISE",
                                                                    target=t2))
        out["keq"][f"d{d}"] = res
    return out


def perf_block(perf, meta, m, n, k):
    """[M] time and TFLOPS-eq per N of the call (its own spec when known, else the GEMM view
    ik,kj->ij), and the K-seg S = 2 composition (two calls of k/2, the second with beta = 1)
    on the GEMM view, as a ratio applied to the spec's time."""
    spec = meta.get("spec", "ik,kj->ij")
    ext = meta.get("ext") or dict(i=m, k=k, j=n)
    out = {}
    for N in NS:
        t, f = perf.predict(spec, ext, N)
        tg, _ = perf.predict("ik,kj->ij", dict(i=m, k=k, j=n), N)
        k1 = k // 2
        ts = perf.predict("ik,kj->ij", dict(i=m, k=k - k1, j=n), N)[0] + \
            perf.predict("ik,kj->ij", dict(i=m, k=k1, j=n), N, beta=True)[0]
        out[str(N)] = dict(time_s=t, tflops_eq=f / t * 1e-12, flops_eq=f,
                           kseg2_time_s=t * ts / tg)
    return out


# ---------------------------------------------------------------------------------------
# Several partners (IR6) and batches (IR16)
# ---------------------------------------------------------------------------------------

def run_multi(X, Ys, meta, cfg, perf, x_role="A"):
    """X pre-scaled by 2^w once, partners Y_p by 2^-w; w from the union extrema; G = min_p
    G_p (each pair at N_ref). x_role: X's role in every pair ("A": rows, "B": columns).
    Compared with each pair's own single-partner w. Sampled outputs per pair."""
    t0 = time.process_time()
    k = X.shape[1]
    rec = dict(meta, k=k, x_rows=X.shape[0], partners=[Y.shape[0] for Y in Ys])
    stX, nX = ke.RowStats(X), ke.norms(X)
    stY = [ke.RowStats(Y) for Y in Ys]
    nY = [ke.norms(Y) for Y in Ys]
    extX = ke.extrema([X])
    extYs = [ke.extrema([Y]) for Y in Ys]
    extU = ke.extrema(Ys)
    d = cfg["dstar"]
    variants = {"shared": ke.weights(extX, extU, d)[0]}
    for p in range(len(Ys)):
        variants[f"own{p}"] = ke.weights(extX, extYs[p], d)[0]
    sig = {key: ke.shifts(X, stX, plan_of(key[0], k, key[1])) for key in keys()}
    sigY = [{key: ke.shifts(Y, s, plan_of(key[0], k, key[1])) for key in keys()}
            for Y, s in zip(Ys, stY)]
    W = {}
    for name, w in variants.items():
        Xw, v1 = ke.scale(X, w, 1)
        stXw = ke.RowStats(Xw)
        sXw = {key: ke.shifts(Xw, stXw, plan_of(key[0], k, key[1])) for key in keys()}
        del Xw
        sYw, viol = [], v1
        for Y in Ys:
            Yw, v2 = ke.scale(Y, w, -1)
            stYw = ke.RowStats(Yw)
            sYw.append({key: ke.shifts(Yw, stYw, plan_of(key[0], k, key[1])) for key in keys()})
            viol += v2
            del Yw
        Gp = []
        for p in range(len(Ys)):
            gx = ke.gain_min(sig[(14, "RN")], sXw[(14, "RN")], stX)
            gy = ke.gain_min(sigY[p][(14, "RN")], sYw[p][(14, "RN")], stY[p])
            Gp.append(None if gx is None or gy is None else gx + gy)
        dG = [g for g in Gp if g is not None]
        W[name] = (w, sXw, sYw, dict(G_pairs=Gp, G=min(dG) if dG else None,
                                     guard=1 if not dG or min(dG) >= 3 else 0,
                                     fp64_violations=viol, num_weighted=int((w != 0).sum())))
    rec["weights"] = {nm: v[3] for nm, v in W.items()}
    rng = np.random.default_rng([meta.get("seed", 0), 4343])
    pairs = []
    for p, Y in enumerate(Ys):
        rx = select(X.shape[0], [], rng, SAMPLE)
        ry = select(Y.shape[0], [], rng, SAMPLE)
        if x_role == "A":
            Sm = Sample(X, Y, rx, ry, nX, nY[p])
        else:
            Sm = Sample(Y, X, ry, rx, nY[p], nX)
        curves = {}
        for key in keys():
            N, rnd = key
            pl = plan_of(N, k, rnd)
            sx, sy = sig[key][rx], sigY[p][key][ry]
            out = {}
            D, _ = (ke.emulate(Sm.As, Sm.Bs, sx, sy, pl, rnd) if x_role == "A" else
                    ke.emulate(Sm.As, Sm.Bs, sy, sx, pl, rnd))
            out["fast"] = evaluate(Sm, D)[1:]
            for nm in ("shared", f"own{p}"):
                w, sXw, sYw, _ = W[nm]
                Xs = np.ldexp(X[rx], w[None, :])
                Ysm = np.ldexp(Y[ry], -w[None, :])
                if x_role == "A":
                    D, _ = ke.emulate(Xs, Ysm, sXw[key][rx], sYw[p][key][ry], pl, rnd)
                else:
                    D, _ = ke.emulate(Ysm, Xs, sYw[p][key][ry], sXw[key][rx], pl, rnd)
                out["own" if nm.startswith("own") else nm] = evaluate(Sm, D)[1:]
            for nm, v in out.items():
                curves.setdefault(nm, {})[N if rnd == "RN" else f"T{N}"] = v
        pr = dict(nmin={nm: dict(ew={str(t): nmin({N: v[1] for N, v in c.items()
                                                    if isinstance(N, int)}, t) for t in EW_TARGETS},
                                 nw={str(t): nmin({N: v[0] for N, v in c.items()
                                                    if isinstance(N, int)}, t) for t in NW_TARGETS},
                                 ew8={str(t): nmin({N: v[2] for N, v in c.items()
                                                    if isinstance(N, int)}, t) for t in EW_TARGETS})
                        for nm, c in curves.items()},
                  curves={nm: {str(N): v for N, v in c.items()} for nm, c in curves.items()})
        pairs.append(pr)
    rec["pairs"] = pairs
    rec["cpu_s"] = round(time.process_time() - t0, 2)
    rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return rec


def run_batched(As, Bs, meta, cfg, perf, per_batch=16):
    """IR16: batches share one w (Q12 a: extrema over every lambda; the guard over every
    (lambda, row)) against per-lambda w (Q12 b). Sampled outputs: per_batch rows x columns of
    every batch, max errors over all batches."""
    t0 = time.process_time()
    b, k = len(As), As[0].shape[1]
    d = cfg["dstar"]
    rec = dict(meta, b=b, m=As[0].shape[0], n=Bs[0].shape[0], k=k)
    shared, _ = ke.weights(ke.extrema(As), ke.extrema(Bs), d)
    st = [(ke.RowStats(A), ke.RowStats(B)) for A, B in zip(As, Bs)]
    nrm = [(ke.norms(A), ke.norms(B)) for A, B in zip(As, Bs)]
    own = [ke.weights(ke.extrema([A]), ke.extrema([B]), d)[0] for A, B in zip(As, Bs)]
    rng = np.random.default_rng([meta.get("seed", 0), 4444])
    curves = {"fast": {}, "shared": {}, "per_batch": {}}
    gains = {"shared": [], "per_batch": []}
    viol = 0
    for lam in range(b):
        A, B = As[lam], Bs[lam]
        (sA, sB), (na, nb) = st[lam], nrm[lam]
        rows = select(A.shape[0], [], rng, per_batch)
        cols = select(B.shape[0], [], rng, per_batch)
        Sm = Sample(A, B, rows, cols, na, nb)
        sig = {key: ke.shifts(A, sA, plan_of(key[0], k, key[1])) for key in keys()}
        tau = {key: ke.shifts(B, sB, plan_of(key[0], k, key[1])) for key in keys()}
        wsh = {}
        for nm, w in (("shared", shared), ("per_batch", own[lam])):
            Aw, v1 = ke.scale(A, w, 1)
            Bw, v2 = ke.scale(B, w, -1)
            viol += v1 + v2
            stAw, stBw = ke.RowStats(Aw), ke.RowStats(Bw)
            sAw = {key: ke.shifts(Aw, stAw, plan_of(key[0], k, key[1])) for key in keys()}
            sBw = {key: ke.shifts(Bw, stBw, plan_of(key[0], k, key[1])) for key in keys()}
            gains[nm].append((ke.gain_min(sig[(14, "RN")], sAw[(14, "RN")], sA),
                              ke.gain_min(tau[(14, "RN")], sBw[(14, "RN")], sB)))
            wsh[nm] = (w, sAw, sBw)
        for key in keys():
            N, rnd = key
            pl = plan_of(N, k, rnd)
            D, _ = ke.emulate(Sm.As, Sm.Bs, sig[key][rows], tau[key][cols], pl, rnd)
            res = {"fast": evaluate(Sm, D)[1:]}
            for nm, (w, sAw, sBw) in wsh.items():
                D, _ = ke.emulate(np.ldexp(Sm.As, w[None, :]), np.ldexp(Sm.Bs, -w[None, :]),
                                  sAw[key][rows], sBw[key][cols], pl, rnd)
                res[nm] = evaluate(Sm, D)[1:]
            kk = N if rnd == "RN" else f"T{N}"
            for nm, v in res.items():
                old = curves[nm].get(kk, [0.0] * len(v))
                curves[nm][kk] = [max(a, b) for a, b in zip(old, v)]

    def guard_of(gl, shared_rows):
        if shared_rows:          # one guard over every (lambda, row) and (lambda, column)
            ga = [g[0] for g in gl if g[0] is not None]
            gb = [g[1] for g in gl if g[1] is not None]
            G = min(ga) + min(gb) if ga and gb else None
            return dict(G=G, guard=1 if G is None or G >= 3 else 0)
        Gs = [None if g[0] is None or g[1] is None else g[0] + g[1] for g in gl]
        dG = [g for g in Gs if g is not None]
        return dict(G_min=min(dG) if dG else None, fails=sum(g < 3 for g in dG),
                    guard=1 if all(g >= 3 for g in dG) else 0)
    rec["guard"] = {"shared": guard_of(gains["shared"], True),
                    "per_batch": guard_of(gains["per_batch"], False)}
    rec["fp64_violations"] = viol
    rec["nmin"] = {nm: dict(ew={str(t): nmin({N: v[1] for N, v in c.items() if isinstance(N, int)}, t)
                                for t in EW_TARGETS},
                            nw={str(t): nmin({N: v[0] for N, v in c.items() if isinstance(N, int)}, t)
                                for t in NW_TARGETS},
                            ew8={str(t): nmin({N: v[2] for N, v in c.items() if isinstance(N, int)}, t)
                                 for t in EW_TARGETS}) for nm, c in curves.items()}
    rec["curves"] = {nm: {str(N): v for N, v in c.items()} for nm, c in curves.items()}
    rec["cpu_s"] = round(time.process_time() - t0, 2)
    return rec


# ---------------------------------------------------------------------------------------
# B6 estimate (IR11): output-selective refinement
# ---------------------------------------------------------------------------------------

def b6_estimate(perf, A, Bt, x_full, targets=(1e-8, 1e-10), label="exact", rho_cap=None):
    """Per row rho_i = max_j ||a_i|| ||b_j|| / |x_ij| over FULL outputs (x_full exact or FP64:
    `label`); N(rho) = min_moduli ELEMENTWISE (BOUND) at the call's (m, n, k); N_hi = N(max
    rho_i). For every N_lo below N_hi: rows needing more than N_lo are re-run at N_hi as a
    gathered dense m' x k x n call; [M] t(N_lo, full) + t(N_hi, m') against t(N_hi, full)."""
    host = perf.host
    m, k = A.shape
    n = Bt.shape[0]
    na, nb = ke.norms(A), ke.norms(Bt)
    with np.errstate(divide="ignore"):
        rho = np.where(np.abs(x_full) > 0, na[:, None] * nb[None, :] / np.abs(x_full), np.inf)
    excluded = 0.0
    if rho_cap is not None:          # outputs with rho > cap need not meet the target
        excluded = float(np.mean(rho > rho_cap))
        rho = np.where(rho > rho_cap, 1.0, rho)
    rho_i = rho.max(axis=1)
    out = dict(label=label, rows=m, rho_max=float(rho_i.max()), rho_cap=rho_cap,
               excluded_outputs=excluded, rho_median_row=float(np.median(rho_i)), targets={})
    memo = {}

    def Nof(r):
        if not math.isfinite(r):
            return None
        key = float(r)
        if key not in memo:
            memo[key] = perf.min_moduli(m, n, k, host.Accuracy(metric="ELEMENTWISE", target=t,
                                                               cond=max(1.0, r)))
        return memo[key]
    for t in targets:
        memo.clear()
        Ns = np.array([Nof(r) or 99 for r in rho_i])
        N_hi = int(Ns.max())
        if N_hi > 20:
            out["targets"][str(t)] = dict(N_hi=None, unreachable_rows=int((Ns > 20).sum()))
            continue
        t_hi = perf.predict("ik,kj->ij", dict(i=m, k=k, j=n), N_hi)[0]
        best = (1.0, N_hi, 0.0)
        for N_lo in range(int(Ns.min()), N_hi):
            mp = int((Ns > N_lo).sum())
            tt = perf.predict("ik,kj->ij", dict(i=m, k=k, j=n), N_lo)[0] + \
                (perf.predict("ik,kj->ij", dict(i=mp, k=k, j=n), N_hi)[0] if mp else 0.0)
            if tt / t_hi < best[0]:
                best = (tt / t_hi, N_lo, mp / m)
        hist = {int(N): int(c) for N, c in zip(*np.unique(Ns, return_counts=True))}
        out["targets"][str(t)] = dict(N_hi=N_hi, best_ratio=best[0], N_lo=best[1],
                                      rows_refined=best[2], rows_per_N=hist)
    return out


# ---------------------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------------------

TIER_S_SIZES = [(512, 3600), (512, 16384), (128, 600), (128, 3600)]   # 512 first: d*
TIER_L_SIZES = [(2048, 3600), (2048, 16384), (4096, 3600)]
TIER_S_FAMS = ["uniform", "phi0.5", "phi2", "phi4", "decay_same", "decay_opp", "opp_nonsep",
               "outliers", "cancel", "same_sign", "local_df", "mixed", "cc_df", "cc_ppl",
               "cc_ring"]
TIER_L_FAMS = ["outliers", "phi2", "phi4", "decay_opp", "opp_nonsep", "uniform", "cc_df",
               "cc_ppl", "cc_ring"]


def fixture(fam, tier, m, k, seed):
    """(A, Bt, meta) of a family at a tier size."""
    if fam.startswith("cc_"):
        size = fx.CC_SIZES[(tier, m, k)].get(fam)
        if size is None:
            return None
        A, Bt, info = fx.cc(fam, size, seed)
        o_, v_, _ = size
        ext = {"cc_df": dict(a=v_, c=v_, b=v_, d=v_, Q=A.shape[1]),
               "cc_ppl": dict(i=o_, j=o_, c=v_, d=v_, a=v_, b=v_),
               "cc_ring": dict(i=o_, l=o_, a=v_, d=v_, k=o_, c=v_)}[fam]
        return A, Bt, dict(info, ext=ext, cc_size=list(size))
    A, Bt = fx.synthetic(fam, m, m, k, seed)
    return A, Bt, {}


def tier_cfg(tier, m, dstar):
    """Schemes per tier (§4.3). Tier S runs m = 512 first (it chooses d*: every K-eq(d) +
    K-seg 2 there), then m = 128 with d* known; stale weights for d = 3 (m = 512) or d*."""
    if tier == "S":
        return dict(deadbands=[0, 2, 3, 5], kseg=[2, 4],
                    kseg_keq=[0, 2, 3, 5] if m == 512 else [dstar],
                    stale_d=[3] if m == 512 else [dstar], env=True, sc25=(m <= 128),
                    sample=SAMPLE if m > 128 else 128)
    return dict(deadbands=sorted({0, dstar}), kseg=[2], kseg_keq=[dstar], stale_d=[dstar],
                env=True, sc25=False, sample=SAMPLE)


def run_tier(args, perf, prov, log):
    sizes = TIER_S_SIZES if args.tier == "S" else TIER_L_SIZES
    if args.sizes:
        sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes]
    fams = args.families or (TIER_S_FAMS if args.tier == "S" else TIER_L_FAMS)
    nseeds = {"S": {s: 5 for s in TIER_S_SIZES},
              "L": {(2048, 3600): 3, (2048, 16384): 3, (4096, 3600): 2}}[args.tier]
    for (m, k) in sizes:
        for fam in fams:
            prev = None
            seeds = args.seeds if args.seeds is not None else list(range(nseeds.get((m, k), 1)))
            for seed in seeds:
                f = fixture(fam, args.tier, m, k, seed)
                if f is None:
                    continue
                A, Bt, extra = f
                meta = dict(prov, tier=args.tier, family=fam, size=[m, k], seed=seed, **extra)
                rec, ws = run_case(A, Bt, meta, tier_cfg(args.tier, m, args.dstar), perf, prev)
                if args.b6 and fam in ("outliers", "opp_nonsep", "phi4") and m <= 128:
                    x_full = exact_full(A, Bt)
                    rec["b6"] = b6_estimate(perf, A, Bt, x_full, label="exact")
                prev = ws
                log(rec)
                del A, Bt


def exact_full(A, Bt):
    """Exact x (as hi) of every output (small m, n only)."""
    return ke.exact_ref(A, Bt)[0]


def run_real(args, perf, prov, log):
    """R1-R7 (NOTES-C1 §4.3, IR19) on the psi4 operands in --real-dir."""
    d = args.real_dir
    jobs = args.families or ["R1:benzene_ccpvtz", "R2:benzene_ccpvdz", "R2b:benzene_ccpvdz",
                             "R3:benzene_ccpvtz", "R4:benzene_ccpvtz", "R5:benzene_ccpvtz",
                             "R6:benzene_ccpvtz", "R7:benzene_ccpvtz", "R2:water_ccpv5z",
                             "R1:water_ccpv5z"]
    cache = {}
    for job in jobs:
        which, mol = job.split(":")
        if mol not in cache:
            cache.clear()
            cache[mol] = fx.load_real(os.path.join(d, mol + ".npz"))
        R = cache[mol]
        meta = dict(prov, tier="real", family=which, molecule=mol, seed=0)
        cfg = tier_cfg("L", 4096, args.dstar)
        cfg["dstar"] = args.dstar
        if which == "R6":
            X, Ys, info = fx.real_multi(R)
            rec = run_multi(X, Ys, dict(meta, **info), cfg, perf, x_role="B")
        elif which == "R7":
            As, Bs = fx.real_batched(R)
            rec = run_batched(As, Bs, meta, cfg, perf)
        else:
            A, Bt, info = fx.real_contractions(R, which)
            rec, _ = run_case(A, Bt, dict(meta, **info), cfg, perf)
            if args.b6 and A.shape[0] * Bt.shape[0] <= 5e7:
                x = np.vstack([A[i0:i0 + 256] @ Bt.T for i0 in range(0, A.shape[0], 256)])
                rec["b6"] = b6_estimate(perf, A, Bt, x, label="FP64 x (estimate)")
                del x
            del A, Bt
        log(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tier", choices=["selftest", "pilot", "S", "L", "real", "multi", "batch",
                                     "b6"])
    ap.add_argument("--out", required=False, default=None, help="directory for <tier>.jsonl")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--sizes", nargs="*", default=None, help="MxK, e.g. 2048x3600")
    ap.add_argument("--dstar", type=int, default=3)
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--b6", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()
    t0 = time.process_time()
    ke.self_test(verbose=True)
    if args.tier == "selftest":
        return 0
    host, h = load_host()
    prov = provenance(h)
    assert prov["host_api"] >= 13, prov
    perf = Perf(host, h)
    out = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out = open(os.path.join(args.out, f"{args.tier}{args.tag}.jsonl"), "a")

    def log(rec):
        line = json.dumps(rec, default=lambda v: v.tolist() if hasattr(v, "tolist") else
                          (int(v) if isinstance(v, np.integer) else str(v)))
        if out:
            out.write(line + "\n")
            out.flush()
        short = {k_: rec.get(k_) for k_ in ("tier", "family", "molecule", "size", "seed", "cpu_s",
                                             "peak_rss_mb")}
        nm = rec.get("nmin", {})
        print(json.dumps(short), "| ew1e-10:",
              {s: v["ew"].get("1e-10") for s, v in nm.items()} if nm else "", flush=True)

    if args.tier == "pilot":
        args.tier, args.families, args.sizes, args.seeds = "L", ["outliers"], ["4096x3600"], [0]
        run_tier(args, perf, prov, log)
    elif args.tier in ("S", "L"):
        run_tier(args, perf, prov, log)
    elif args.tier == "real":
        run_real(args, perf, prov, log)
    elif args.tier == "multi":
        cfg = dict(dstar=args.dstar)
        for (m, k) in [tuple(int(v) for v in s.split("x")) for s in (args.sizes or ["2048x3600"])]:
            for seed in (args.seeds or [0]):
                X, Ys = fx.multi3(m, m, k, seed)
                log(run_multi(X, Ys, dict(prov, tier="multi", family="multi3", size=[m, k],
                                          seed=seed), cfg, perf, x_role="A"))
    elif args.tier == "b6":
        # IR11 on the real operands: full FP64 outputs (an estimate), every output and only the
        # outputs with rho <= RHO_MUST (structural near-zeros of the point group excluded)
        cache = {}
        for job in args.families:
            which, mol = job.split(":")
            if mol not in cache:
                cache.clear()
                cache[mol] = fx.load_real(os.path.join(args.real_dir, mol + ".npz"))
            A, Bt, info = fx.real_contractions(cache[mol], which)
            x = np.vstack([A[i0:i0 + 256] @ Bt.T for i0 in range(0, A.shape[0], 256)])
            rec = dict(prov, tier="b6", family=which, molecule=mol, m=A.shape[0], n=Bt.shape[0],
                       k=A.shape[1])
            rec["b6"] = b6_estimate(perf, A, Bt, x, label="FP64 x (estimate)")
            rec["b6_must"] = b6_estimate(perf, A, Bt, x, label="FP64 x (estimate)",
                                         rho_cap=RHO_MUST)
            log(rec)
            del A, Bt, x
    elif args.tier == "batch":
        cfg = dict(dstar=args.dstar)
        for seed in (args.seeds or [0]):
            As, Bs = fx.batch_mixed(16, 256, 256, 3600, seed)
            log(run_batched(As, Bs, dict(prov, tier="batch", family="batch_mixed", seed=seed),
                            cfg, perf))
    print(f"done: {time.process_time() - t0:.1f} CPU-s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
