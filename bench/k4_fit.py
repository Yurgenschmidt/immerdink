#!/usr/bin/env python3
"""The K4 tile-selection fit and its adoption rule (issue 20, docs/NOTES-20.md).

K4 has four tile shapes. The planner (src/host/k4_config.cpp k4_select) predicts each one's time from a cost
table fitted on the RTX 4090 (docs/NOTES-A3.md) and takes a smaller tile only if it is predicted 5% faster than
128x256. This module is the fit script that note describes and that was never committed: it reproduces the shipped
constants from the A3 data, fits a table for another card from a forced-tile sweep, and decides whether to adopt it.

  k4_fit.py fit        RECORDS... [--sms N]                fit a cost table; print it and its relative error
  k4_fit.py decide     RECORDS... [--sms N]                the adoption rule (below) on those records
  k4_fit.py identify   [--sms N] [--noise 0.03]            how well a set of shapes pins each constant down
  k4_fit.py occupancy  [--card h100|...]                   CTAs per SM of each tile on a card, from shared memory and registers
  k4_fit.py a3-check   A3_NSYS A3_PHASES                   reproduce the shipped table from the A3 data

Records: JSON lines, either the nsys format of the A3 data ({"case", "N", "cfg", "k4_ms", "view"}) or imd_bench records of a
sweep with a forced tile (`--k4 0,1,2,3`: "k4_config", "phases" {"K4 gemm": {"ms"}}, "phase_view", "case", "N").

The model (DESIGN v1.5 section 4, as k4_config.cpp): over one group's launch a tile configuration c with tile tm x tn runs
tiles = ceil(m / tm) ceil(n_hat / tn) g b tiles; the busiest SM gets j = ceil(tiles / SMs); it runs floor(j / o) rounds of o
concurrent CTAs (o = CTAs per SM) and one round of the j mod o left. A round of q CTAs costs
round_ns + q tm tn k_hat mac_fs[q]. The constants are fitted by linear least squares in the RELATIVE error, per configuration.

The adoption rule (fixed before the data, issue 20): per card, adopt the refit table only if, on the held-out cases (every 4th case
by name, HOLDOUT_EVERY), (a) the geometric mean of the measured time of the refit's pick over the current table's pick is at most
1 / 1.015; (b) no held-out (case, N) is more than 3% slower with the refit's pick; and (c) the mean leave-one-case-out regret of the
refit rule (time of its pick over the best tile's time, minus 1), over all cases, is at most 2%. Otherwise keep the table.
"""
import argparse
import json
import math
import re
import sys

import numpy as np

SAT = (1 << 127) - 1
HOLDOUT_EVERY = 4
GAIN_MIN = 0.015          # (a)
SLOWER_MAX = 0.03         # (b)
REGRET_MAX = 0.02         # (c)
MARGIN_PERMILLE = 50      # a smaller tile must beat 128x256 by 5% (k4_config.hpp kK4MarginPermille)

# K4ConfigDesc (src/host/k4_config.hpp): tile_m, tile_n, stages, CTAs per SM the cost model assumes (4090 figures)
CONFIGS = [dict(name="128x256", tm=128, tn=256, stages=3, ctas=1, threads=256, regs=248),
           dict(name="128x128", tm=128, tn=128, stages=3, ctas=2, threads=128, regs=None),
           dict(name="64x128", tm=64, tn=128, stages=4, ctas=2, threads=128, regs=None),
           dict(name="64x64", tm=64, tn=64, stages=4, ctas=3, threads=64, regs=None)]
# the shipped table (src/host/k4_config.hpp kK4Costs): (round_ns, (mac_fs[1], ..., mac_fs[ctas]))
SHIPPED = [(2716, (418,)), (2550, (453, 456)), (1424, (505, 612)), (1333, (944, 1088, 802))]


class Table:
    def __init__(self, costs, ctas=None, margin=MARGIN_PERMILLE):
        self.costs = [(int(r), tuple(int(x) for x in m)) for r, m in costs]
        self.ctas = tuple(ctas) if ctas else tuple(c["ctas"] for c in CONFIGS)
        self.margin = margin

    def __repr__(self):
        return "Table(" + ", ".join(f"{r}|{'/'.join(map(str, m))}" for r, m in self.costs) + ")"

    def as_dict(self):
        return {"costs": [[r, list(m)] for r, m in self.costs], "ctas": list(self.ctas), "margin_permille": self.margin}


SHIPPED_TABLE = Table(SHIPPED)


# ----------------------------------------------------------------------------------------------- the model, exactly
def rounds(table, c, m, nh, kh, gb, sms):
    """(full rounds of o CTAs, CTAs in the last round, MACs of a tile) as k4_rounds."""
    cfg = CONFIGS[c]
    tiles = -(-m // cfg["tm"]) * -(-nh // cfg["tn"]) * gb
    j = -(-tiles // sms)
    o = table.ctas[c]
    return j // o, j % o, cfg["tm"] * cfg["tn"] * kh


def estimate_fs(table, c, m, nh, kh, gb, sms):
    """k4_estimate_fs: integer femtoseconds, saturating at 2^127 - 1."""
    full, rest, macs = rounds(table, c, m, nh, kh, gb, sms)
    rnd, mac = table.costs[c]
    o = table.ctas[c]

    def round_fs(q):
        return min(rnd * 10**6 + macs * q * mac[q - 1], SAT)

    est = min(full * round_fs(o), SAT) if full else 0
    return min(est + round_fs(rest), SAT) if rest else est


def pick(table, m, nh, kh, gb, sms):
    """k4_select: the configuration the planner takes."""
    est = [estimate_fs(table, c, m, nh, kh, gb, sms) for c in range(4)]
    best, best_est = 0, est[0]
    for c in range(1, 4):
        if min(est[c] * 1000, SAT) < min(est[0] * (1000 - table.margin), SAT) and est[c] < best_est:
            best, best_est = c, est[c]
    return best


# ----------------------------------------------------------------------------------------------------- the data
MIN_MS_MARKS = 0.10   # a phase mark (CUDA events around the K4 launch) shorter than this is too coarse to fit or judge on (jitter of a few us is 5% of 0.1 ms)


class Obs:
    __slots__ = ("case", "N", "cfg", "ms", "m", "nh", "kh", "b", "src")

    def __init__(self, case, N, cfg, ms, m, nh, kh, b, src="nsys"):
        self.case, self.N, self.cfg, self.ms, self.m, self.nh, self.kh, self.b = case, N, cfg, ms, m, nh, kh, b
        self.src = src                  # "nsys" (kernel medians) or "marks" (the bench's K4 phase marks)

    @property
    def gb(self):
        return self.N * self.b          # g = N: one group, one launch (the sweeps run with g = N)


_VIEW = re.compile(r"(\d+)x(\d+)x(\d+)(?: b=(\d+))? n_hat=(\d+) k_hat=(\d+)")


def parse_view(v):
    a = _VIEW.match(v)
    if not a:
        raise ValueError(f"cannot read the GEMM view {v!r}")
    m, _n, _k, b, nh, kh = a.groups()
    return int(m), int(nh), int(kh), int(b or 1)


def load_records(paths):
    """Observations from JSON-lines files (see obs_from_dicts)."""
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return obs_from_dicts(recs)


def obs_from_dicts(recs):
    """Observations from parsed records (A3 nsys format or imd_bench records with a forced tile); of several records of one
    (case, N, tile) the first is kept."""
    out, seen = [], set()
    for r in recs:
        src = "nsys"
        if "k4_ms" in r:
            cfg, ms, view = r["cfg"], r["k4_ms"], r["view"]
        elif "phases" in r and "K4 gemm" in r.get("phases", {}) and r.get("status", "ok") == "ok":
            cfg, ms, view = r.get("k4_config"), r["phases"]["K4 gemm"]["ms"], r["phase_view"]
            src = "marks"
        else:
            continue
        if cfg is None or cfg < 0 or ms is None or ms <= 0:
            continue
        key = (r["case"], r["N"], cfg)
        if key in seen:
            continue
        seen.add(key)
        m, nh, kh, b = parse_view(view)
        out.append(Obs(r["case"], r["N"], cfg, ms, m, nh, kh, b, src))
    return out


# ------------------------------------------------------------------------------------------------------- the fit
def _nnls(A, y, iters=200):
    """Non-negative least squares (Lawson-Hanson, small sizes)."""
    n = A.shape[1]
    x = np.zeros(n)
    P = []
    w = A.T @ (y - A @ x)
    for _ in range(iters):
        if not (w[[i for i in range(n) if i not in P]] > 1e-12).any() if len(P) < n else True:
            break
        cand = [i for i in range(n) if i not in P]
        P.append(max(cand, key=lambda i: w[i]))
        while True:
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], y, rcond=None)[0]
            if (s[P] > 0).all():
                x = s
                break
            neg = [i for i in P if s[i] <= 0]
            alpha = min((x[i] / (x[i] - s[i]) for i in neg if x[i] != s[i]), default=1.0)
            x = x + alpha * (s - x)
            P = [i for i in P if x[i] > 1e-12]
        w = A.T @ (y - A @ x)
    return x


def fit_config(obs, c, sms, ctas):
    """(round_ns, [mac_fs]) and the relative residuals of configuration c on its observations."""
    rows, y = [], []
    for o in (x for x in obs if x.cfg == c):
        full, rest, macs = rounds(Table(SHIPPED, ctas=ctas), c, o.m, o.nh, o.kh, o.gb, sms)
        feat = [0.0] * (1 + ctas[c])
        feat[0] = full + (1 if rest else 0)
        feat[ctas[c]] += full * ctas[c] * macs
        if rest:
            feat[rest] += rest * macs
        rows.append(feat)
        y.append(o.ms * 1e-3)
    if len(y) < 1 + ctas[c]:
        return None
    X, y = np.array(rows), np.array(y)
    scale = np.array([1e-9] + [1e-15] * ctas[c])
    A = (X * scale) / y[:, None]
    p = _nnls(A, np.ones(len(y)))
    res = A @ p - 1.0
    return p, res


def fit_table(obs, sms=128, ctas=None):
    """A Table fitted on `obs` (all four tiles needed)."""
    ctas = tuple(ctas) if ctas else tuple(c["ctas"] for c in CONFIGS)
    costs, rel = [], []
    for c in range(4):
        r = fit_config(obs, c, sms, ctas)
        if r is None:
            return None, None
        p, res = r
        costs.append((round(p[0]), tuple(round(x) for x in p[1:])))
        rel.append(res)
    return Table(costs, ctas), rel


def identify(shapes, sms=128, ctas=None, noise=0.03, Ns=(8, 14), table=SHIPPED_TABLE):
    """Relative standard error of each constant if every (shape, N, tile) is timed once with `noise` relative noise: the
    sampling error of the relative least-squares fit, sqrt(diag(noise^2 (A^T A)^-1)) / constant. A constant that no shape
    exercises is reported as inf (it cannot be fitted)."""
    ctas = tuple(ctas) if ctas else table.ctas
    out = []
    for c in range(4):
        rnd, mac = table.costs[c]
        p0 = np.array([rnd] + list(mac), float)
        sc = np.array([1e-9] + [1e-15] * ctas[c])
        A = []
        for (m, nh, kh, b) in shapes:
            for N in Ns:
                full, rest, macs = rounds(table, c, m, nh, kh, N * b, sms)
                feat = np.zeros(1 + ctas[c])
                feat[0] = full + (1 if rest else 0)
                feat[ctas[c]] += full * ctas[c] * macs
                if rest:
                    feat[rest] += rest * macs
                t = float(feat @ (p0 * sc))
                if t * 1e3 < MIN_MS_MARKS:       # decide() drops a mark this short: it cannot help identify a constant
                    continue
                A.append(feat * sc / t)
        if not A:                       # every row is below the mark cut: nothing identifies anything
            out.append([math.inf] * (1 + ctas[c]))
            continue
        A = np.array(A)
        exercised = (A != 0).any(axis=0)
        sd = []
        if exercised.all():
            cov = noise ** 2 * np.linalg.inv(A.T @ A)
            sd = [float(math.sqrt(cov[i, i]) / p0[i]) for i in range(len(p0))]
        else:
            sub = A[:, exercised]
            cov = noise ** 2 * np.linalg.inv(sub.T @ sub) if exercised.any() else None
            k = 0
            for i in range(len(p0)):
                if exercised[i]:
                    sd.append(float(math.sqrt(cov[k, k]) / p0[i])); k += 1
                else:
                    sd.append(math.inf)
        out.append(sd)
    return out


# ------------------------------------------------------------------------------------ the adoption rule (issue 20)
def usable(obs):
    """The observations every command fits and judges on: phase marks shorter than MIN_MS_MARKS are dropped."""
    return [o for o in obs if o.src != "marks" or o.ms >= MIN_MS_MARKS]


def view_of(o):
    return (o.m, o.nh, o.kh, o.b)


def table_ok(t, ctas):
    """A fitted table is usable only if every constant is positive (a zero would make that tile look free) and no tile needs more
    concurrent CTAs than K4Cost holds (3)."""
    if any(n > 3 or n < 1 for n in ctas):
        return False
    return all(r > 0 and len(m) == ctas[c] and all(x > 0 for x in m) for c, (r, m) in enumerate(t.costs))


def by_case(obs):
    d = {}
    for o in obs:
        d.setdefault((o.case, o.N), {})[o.cfg] = o
    return {k: v for k, v in d.items() if set(v) == {0, 1, 2, 3}}


def holdout_cases(cases):
    names = sorted(set(cases))
    return {n for i, n in enumerate(names) if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}


def holdout_views(views):
    """Every 4th distinct GEMM view (m, n_hat, k_hat, b) in sorted order: cases that are the same view under different names
    (s_df and x1k_256, t_ikl and t_batch) fall on the same side."""
    vs = sorted(set(views))
    return {v for i, v in enumerate(vs) if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}


def decide(obs, sms=128, ctas=None, old=SHIPPED_TABLE):
    """The adoption verdict: {"verdict": "ADOPT" | "KEEP" | "INCONCLUSIVE", "reason", numbers, "table"}. Phase marks below
    MIN_MS_MARKS are not used (too coarse); a (case, N) with any tile missing is not used."""
    obs = usable(obs)
    groups = by_case(obs)
    cases = {k[0] for k in groups}
    views = {view_of(g[0]) for g in groups.values()}
    if len(cases) < 8:
        return {"verdict": "INCONCLUSIVE", "reason": f"only {len(cases)} cases with all four tiles (need 8)", "table": None}
    held_views = holdout_views(views)
    held = {k[0] for k, g in groups.items() if view_of(g[0]) in held_views}     # cases on a held-out view
    train = [o for o in obs if view_of(o) not in held_views and (o.case, o.N) in groups]
    new, rel = fit_table(train, sms, ctas)
    if new is None:
        return {"verdict": "INCONCLUSIVE", "reason": "too few training observations to fit every constant", "table": None}
    ctas = new.ctas

    def t_of(g, c):
        return g[c].ms

    ratios, worst, rows = [], 1.0, []
    for (case, N), g in sorted(groups.items()):
        if view_of(g[0]) not in held_views:
            continue
        o0 = g[0]
        c_old = pick(old, o0.m, o0.nh, o0.kh, o0.gb, sms)
        c_new = pick(new, o0.m, o0.nh, o0.kh, o0.gb, sms)
        r = t_of(g, c_new) / t_of(g, c_old)
        ratios.append(r)
        worst = max(worst, r)
        rows.append((case, N, c_old, c_new, r))
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 1.0
    # leave-one-case-out regret over all cases
    regrets = []
    for view in sorted(views):
        tr = [o for o in obs if view_of(o) != view and (o.case, o.N) in groups]
        t, _ = fit_table(tr, sms, ctas)
        if t is None:
            continue
        for (cs, N), g in groups.items():
            if view_of(g[0]) != view:
                continue
            o0 = g[0]
            c = pick(t, o0.m, o0.nh, o0.kh, o0.gb, sms)
            regrets.append(t_of(g, c) / min(t_of(g, k) for k in range(4)) - 1.0)
    regret = sum(regrets) / len(regrets) if regrets else math.inf
    old_regret = sum((g[pick(old, g[0].m, g[0].nh, g[0].kh, g[0].gb, sms)].ms / min(g[k].ms for k in range(4)) - 1.0)
                     for g in groups.values()) / len(groups)
    gain = 1.0 / gm - 1.0
    # what the adopted table is: the fit on every usable observation (the training split only validated the procedure)
    final, _ = fit_table([o for o in obs if (o.case, o.N) in groups], sms, ctas)
    if final is None:
        return {"verdict": "INCONCLUSIVE", "reason": "too few observations to fit every constant on all the data", "table": None}
    ok = gain >= GAIN_MIN and worst - 1.0 <= SLOWER_MAX and regret <= REGRET_MAX
    reason = (f"held-out gain {100 * gain:.2f}% (need {100 * GAIN_MIN:g}), worst held-out slowdown {100 * (worst - 1):.2f}% "
              f"(at most {100 * SLOWER_MAX:g}), leave-one-out regret {100 * regret:.2f}% (at most {100 * REGRET_MAX:g})")
    # A refit that is not a usable table (a constant of zero: this data cannot identify it, typically because the short phase marks were cut) can never be
    # adopted. If the rule would have adopted it the verdict is INCONCLUSIVE; if the rule says keep, the table is kept either way.
    unusable = not (table_ok(new, ctas) and table_ok(final, ctas))
    if unusable:
        reason += "; the fit has a constant that is not positive (not identified by this data): " + repr(final)
    verdict = ("INCONCLUSIVE" if unusable else "ADOPT") if ok else "KEEP"
    unidentified = final.as_dict() if unusable else None      # never offered as "table": a consumer that skips the verdict must not ship it
    if unusable:
        final = None
    return {"verdict": verdict, "reason": reason, "gain": gain, "worst_slowdown": worst - 1.0,
            "regret": regret, "held_out_views": sorted(held_views), "held_out_cases": sorted(held), "held_out_rows": rows,
            "old_table_regret": old_regret, "table": final.as_dict() if final is not None else None, "unidentified_fit": unidentified,
            "validated_on_training_split": new.as_dict()}


# ------------------------------------------------------------------------------- occupancy of each tile on a card
# per-SM limits (CUDA programming guide table 'Technical specifications per compute capability'; the shared-memory figure
# is what a block may use out of the SM's, 1 KiB per block is reserved by the driver)
CARDS = {
    "rtx4090": dict(sm="8.9", sms=128, smem=100 * 1024, regs=65536, threads=1536, blocks=24, reserved=1024),
    "h100": dict(sm="9.0", sms=132, smem=228 * 1024, regs=65536, threads=2048, blocks=32, reserved=1024),
    "h200": dict(sm="9.0", sms=132, smem=228 * 1024, regs=65536, threads=2048, blocks=32, reserved=1024),
    "rtxpro6000": dict(sm="12.0", sms=188, smem=100 * 1024, regs=65536, threads=1536, blocks=24, reserved=1024),
}
# registers per thread of each K4 configuration (the larger of its store-mod and add-mod kernels), from cuobjdump --dump-resource-usage of the sm_89 build (2026-10-15)
# an sm_90 / sm_120 build may differ: `occupancy --regs` overrides.
REGS_89 = [248, 252, 168, 168]


def smem_bytes(c):
    cfg = CONFIGS[c]
    return cfg["stages"] * (cfg["tm"] + cfg["tn"]) * 64


def occupancy(card, c, regs=None):
    """CTAs per SM of tile c on `card`: the minimum of the shared-memory, register, thread and block limits."""
    k = CARDS[card]
    cfg = CONFIGS[c]
    r = (regs or REGS_89)[c]
    by_smem = k["smem"] // (smem_bytes(c) + k["reserved"])
    # registers are allocated per warp in units of 256 registers
    per_warp = -(-r * 32 // 256) * 256
    warps = cfg["threads"] // 32
    by_regs = k["regs"] // (per_warp * warps)
    by_threads = k["threads"] // cfg["threads"]
    return max(0, min(by_smem, by_regs, by_threads, k["blocks"]))


# ----------------------------------------------------------------------------------------------------------- CLI
def _shapes_17():
    def c16(x):
        return -(-x // 16) * 16

    def ca(k):
        return -(-k // 64) * 64 if k > 96 else c16(k)
    raw = [(256, 256, 256, 1), (512, 512, 512, 1), (1536, 1536, 1536, 1), (64, 64, 64, 256), (1024, 1024, 1024, 1),
           (2048, 2048, 2048, 1), (4096, 4096, 4096, 1), (8192, 8192, 8192, 1), (8192, 8192, 1024, 1),
           (2048, 2048, 32768, 1), (16384, 512, 4096, 1), (1000, 64, 50000, 1), (4096, 4096, 2048, 1),
           (4096, 4096, 4096, 1), (4096, 4096, 4096, 1), (1024, 1024, 1024, 16), (1024, 1024, 1024, 16)]
    return [(m, c16(n), ca(k), b) for (m, n, k, b) in raw]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "decide"):
        p = s.add_parser(name)
        p.add_argument("records", nargs="+")
        p.add_argument("--sms", type=int, default=128)
        p.add_argument("--ctas", help="CTAs per SM per tile, comma list (default 1,2,2,3)")
        p.add_argument("--exclude", action="append", default=[], help="case names to drop")
    p = s.add_parser("identify")
    p.add_argument("--sms", type=int, default=128)
    p.add_argument("--noise", type=float, default=0.03)
    p = s.add_parser("occupancy")
    p.add_argument("--card", choices=sorted(CARDS))
    p.add_argument("--regs", help="registers per thread of each tile, comma list")
    p = s.add_parser("a3-check")
    p.add_argument("nsys")
    p.add_argument("phases")
    a = ap.parse_args(argv)

    if a.cmd in ("fit", "decide"):
        obs = usable([o for o in load_records(a.records) if o.case not in a.exclude])
        ctas = [int(x) for x in a.ctas.split(",")] if a.ctas else None
        if a.cmd == "fit":
            t, rel = fit_table(obs, a.sms, ctas)
            if t is None:
                print("not enough observations for every tile", file=sys.stderr)
                return 1
            print(json.dumps({"table": t.as_dict(), "n": [len(r) for r in rel],
                              "rms_relative_error": [float(np.sqrt(np.mean(r ** 2))) for r in rel]}))
        else:
            print(json.dumps(decide(obs, a.sms, ctas), default=float))
    elif a.cmd == "identify":
        for c, sd in enumerate(identify(_shapes_17(), a.sms, noise=a.noise)):
            print(CONFIGS[c]["name"], ["inf" if math.isinf(x) else f"{100 * x:.1f}%" for x in sd])
    elif a.cmd == "occupancy":
        regs = [int(x) for x in a.regs.split(",")] if a.regs else None
        for card in ([a.card] if a.card else sorted(CARDS)):
            print(card, {CONFIGS[c]["name"]: (occupancy(card, c, regs), CONFIGS[c]["ctas"]) for c in range(4)})
    elif a.cmd == "a3-check":
        obs = [o for o in load_records([a.nsys, a.phases]) if o.case != "s_df"]
        t, rel = fit_table(obs, 128)
        print(repr(t), "n =", [len(r) for r in rel])
        return 0 if t.costs == SHIPPED_TABLE.costs else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
