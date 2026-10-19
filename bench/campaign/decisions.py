"""Decisions: the formulas that turn measured records into a verdict, written before the data exist.

A decisions file (TOML, pinned by sha256 in the manifest) has one [[decision]] table per question:

    [[decision]]
    id = "wgmma-go"   kind = "threshold"   requires = ["ceil.int8"]
    exp = "ceil.int8"  file = "ceil.jsonl"  where = {method = "ceil"}  field = "tops"
    aggregate = "max"  go_at_least = 1200.0  nogo_below = 900.0

Kinds (each a pure function of the records; a verdict is one of VERDICTS):
  threshold  one number from the records against two limits: GO / NO-GO / INCONCLUSIVE
  paired     interleaved A/B records (duel-b rows, or `method` = xduel-imd for immerdink against a baseline, with pair_t_a /
             pair_t_b arrays; a = reference, b = variant, ratio > 1: b faster): per case a
             faster / tie / slower class by the tie rule (stats.py); ADOPT / KEEP / INCONCLUSIVE
  parity     the parity gate over comparison records (bench sweep_compare): per case the ratio of the best DGEMM-level
             configuration of two families (default immerdink / GEMMul8 fast, TFLOPS-eq), PASS when the geometric mean
             over the cases (some excluded, e.g. the one-call-per-slice batch case) reaches the limit and no case is
             below the floor once its measured spread is added; else FAIL
  tile_refit forced-tile sweep records (imd_bench --k4 0,1,2,3, the K4 phase mark): refit the planner's K4 tile cost table
             (bench/k4_fit.py) and decide by the rule fixed for issue 20 whether the card adopts it: ADOPT / KEEP /
             INCONCLUSIVE (held-out gain >= 1.5%, nothing > 3% slower, leave-one-out regret <= 2%). The card's SM count is
             read from the records. The refitted table is in the details.
  rate_projection  the wgmma go / no-go of issue 16 (NOTES-16, docs/CAMPAIGN.md): from K4's measured rate, the best stock INT8 GEMM
             rate on K4's launch shape and the interleaved GEMMul8 time, project the immerdink time with faster K4 and
             pack/CRT and compare it with GEMMul8; GO / NO-GO / INCONCLUSIVE (role confirm: PASS / FAIL on the rate gain)
  guard      records against a stored baseline file: PASS / FLAGGED by the regression guard (field: a time, e.g. median_s;
             a slowdown is new / base - 1; records with samples_s use the robust spread)

`requires` lists the experiments whose data the verdict stands on. A damaged one makes the verdict
INVALID (the analyzer decides that, analyze.py); one that simply did not run (failed, skipped for time)
makes it INCONCLUSIVE.
"""
import hashlib
import json
import math
import os
import re
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I

import stats

VERDICTS = ("ADOPT", "KEEP", "GO", "NO-GO", "PASS", "FAIL", "FLAGGED", "INCONCLUSIVE", "INVALID")   # rate_projection: GO / NO-GO / INCONCLUSIVE, or PASS / FAIL in its confirm role
AGGREGATES = ("max", "min", "median", "geomean", "mean")
PLACEHOLDER = re.compile(r"\b(TBD|TODO|FIXME|XXX)\b|\?\?")

# kind -> {param: type or tuple of types}; "requires" and "id" are common to all kinds
REQUIRED = {
    "threshold": {"exp": str, "file": str, "field": str, "aggregate": str,
                  "go_at_least": (int, float), "nogo_below": (int, float)},
    "paired": {"exp": str, "file": str},
    "guard": {"exp": str, "file": str, "baseline": str, "baseline_sha256": str, "key": list, "field": str},
    "parity": {"exp": str, "file": str, "geomean_at_least": (int, float), "case_floor": (int, float)},
    "tile_refit": {"exp": str, "file": str},
    "rate_projection": {"exp": str, "file": str, "xduel_exp": str, "xduel_file": str, "ceiling_exp": str, "ceiling_file": str,
                        "cases": list, "gain_go": (int, float), "pack_factor_go": (int, float), "pack_factor_opt": (int, float),
                        "window": (int, float), "mma_margin": (int, float),
                        # every parameter of the rule is in the (hash-pinned) decisions file: no default in this code can change it
                        "n": int, "g8_n": int, "s_min": (int, float), "s_max": (int, float), "u_floor": (int, float),
                        "u_noisy": (int, float), "noisy_spread_pct": (int, float)},
}
RATE_CASES = ("sq4096", "sq8192")   # the cases of the wgmma rule, registered in docs/NOTES-16.md
OPTIONAL = {
    "threshold": {"where": dict, "classes": list},
    "paired": {"where": dict, "method": str, "min_faster": int, "noisy_spread_pct": (int, float),
               "noisy_min_effect": (int, float), "conf": (int, float), "classes": list},
    "guard": {"where": dict, "floor": (int, float), "k": (int, float), "classes": list,
              "spread_field": str},
    "parity": {"a_family": str, "b_family": str, "exclude": list, "classes": list},
    "tile_refit": {"where": dict, "classes": list, "exclude": list},
    "rate_projection": {"role": str, "confirm_gain": (int, float), "classes": list},
}


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    d = tomllib.loads(raw.decode("utf-8"))
    return d.get("decision", []), hashlib.sha256(raw).hexdigest()


def lint(path, experiments):
    """Findings for a decisions file; `experiments` maps id -> manifest entry."""
    from manifest import Finding
    F = []
    try:
        decs, _ = load(path)
    except Exception as e:  # tomllib.TOMLDecodeError and OSError
        return [Finding("error", path, f"cannot read: {e}")]
    if not decs:
        F.append(Finding("error", path, "no [[decision]] table"))
    seen = set()
    for i, d in enumerate(decs):
        w = f"decision {d.get('id', i)}"
        if not isinstance(d.get("id"), str) or d["id"] in seen:
            F.append(Finding("error", w, "needs a unique string id"))
        seen.add(d.get("id"))
        for k, v in d.items():
            if isinstance(v, str) and PLACEHOLDER.search(v):
                F.append(Finding("error", w, f"placeholder text in {k}: {v!r}"))
        kind = d.get("kind")
        if kind not in REQUIRED:
            F.append(Finding("error", w, f"kind must be one of {tuple(REQUIRED)}"))
            continue
        for p, t in REQUIRED[kind].items():
            if p not in d:
                F.append(Finding("error", w, f"missing {p}"))
            elif not isinstance(d[p], t) or isinstance(d[p], bool):
                F.append(Finding("error", w, f"{p} has the wrong type"))
        allowed = set(REQUIRED[kind]) | set(OPTIONAL[kind]) | {"id", "kind", "requires", "note"}
        for k in d:
            if k not in allowed:
                F.append(Finding("error", w, f"unknown parameter {k}"))
        req = d.get("requires")
        if not (isinstance(req, list) and req and all(isinstance(r, str) for r in req)):
            F.append(Finding("error", w, "needs requires = [experiment ids]"))
        else:
            for r in req:
                if r not in experiments:
                    F.append(Finding("error", w, f"requires unknown experiment {r!r}"))
        if d.get("exp") not in experiments:
            F.append(Finding("error", w, f"exp {d.get('exp')!r} is not in the manifest"))
        elif isinstance(req, list) and d.get("exp") not in req:
            F.append(Finding("error", w, "exp must be listed in requires"))
        if kind == "threshold":
            if d.get("aggregate") not in AGGREGATES:
                F.append(Finding("error", w, f"aggregate must be one of {AGGREGATES}"))
            if isinstance(d.get("go_at_least"), (int, float)) and isinstance(d.get("nogo_below"), (int, float)) \
                    and d["nogo_below"] > d["go_at_least"]:
                F.append(Finding("error", w, "nogo_below must not exceed go_at_least"))
        if kind == "rate_projection":
            for key in ("xduel_exp", "ceiling_exp"):
                if d.get(key) not in experiments:
                    F.append(Finding("error", w, f"{key} {d.get(key)!r} is not in the manifest"))
                elif isinstance(req, list) and d.get(key) not in req:
                    F.append(Finding("error", w, f"{key} must be listed in requires"))
            if isinstance(d.get("cases"), list) and sorted(d["cases"]) != sorted(RATE_CASES):
                F.append(Finding("error", w, f"cases must be exactly {list(RATE_CASES)}, the registered ones"))
            if d.get("role", "binding") not in ("binding", "confirm"):
                F.append(Finding("error", w, "role must be binding or confirm"))
            if d.get("role") == "confirm" and "confirm_gain" not in d:
                F.append(Finding("error", w, "role confirm needs confirm_gain"))
            if isinstance(d.get("pack_factor_opt"), (int, float)) and isinstance(d.get("pack_factor_go"), (int, float)) \
                    and d["pack_factor_opt"] > d["pack_factor_go"]:
                F.append(Finding("error", w, "pack_factor_opt must not exceed pack_factor_go"))
        if kind == "guard" and isinstance(d.get("baseline"), str):
            bp = os.path.join(os.path.dirname(os.path.abspath(path)), d["baseline"])
            if not os.path.exists(bp):
                F.append(Finding("error", w, f"baseline file {d['baseline']} does not exist"))
            elif isinstance(d.get("baseline_sha256"), str):
                from manifest import sha256_file
                if sha256_file(bp) != d["baseline_sha256"]:
                    F.append(Finding("error", w, "baseline_sha256 does not match the baseline file"))
    return F


def applies(dec, cls):
    cl = dec.get("classes")
    return cl is None or cls in cl


def select(records, where):
    return [r for r in records if all(r.get(k) == v for k, v in (where or {}).items())]


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _aggregate(vals, how):
    if how == "max":
        return max(vals)
    if how == "min":
        return min(vals)
    if how == "median":
        return stats.median(vals)
    if how == "mean":
        return sum(vals) / len(vals)
    return math.exp(sum(math.log(v) for v in vals) / len(vals))   # geomean (positive values only)


def evaluate(dec, ctx):
    """Verdict of one decision. `ctx.jsonl(exp, file)` returns the parsed records, ctx.base_dir the decisions folder.
    Returns {"verdict", "reason", "details"}."""
    return _EVAL[dec["kind"]](dec, ctx)


def _threshold(dec, ctx):
    recs = select(ctx.jsonl(dec["exp"], dec["file"]), dec.get("where"))
    vals = [v for v in (_num(r.get(dec["field"])) for r in recs
                        if r.get("status", "ok") in ("ok", None)) if v is not None]
    if not vals:
        return {"verdict": "INCONCLUSIVE", "reason": f"no usable {dec['field']} value in {dec['file']}", "details": {"records": len(recs)}}
    if dec["aggregate"] == "geomean" and min(vals) <= 0:
        return {"verdict": "INCONCLUSIVE", "reason": "geomean of a non-positive value", "details": {}}
    v = _aggregate(vals, dec["aggregate"])
    det = {"value": v, "n": len(vals), "go_at_least": dec["go_at_least"], "nogo_below": dec["nogo_below"]}
    if v >= dec["go_at_least"]:
        return {"verdict": "GO", "reason": f"{dec['aggregate']}({dec['field']}) = {v:.6g} >= {dec['go_at_least']:g}", "details": det}
    if v < dec["nogo_below"]:
        return {"verdict": "NO-GO", "reason": f"{dec['aggregate']}({dec['field']}) = {v:.6g} < {dec['nogo_below']:g}", "details": det}
    return {"verdict": "INCONCLUSIVE", "reason": f"{v:.6g} lies between {dec['nogo_below']:g} and {dec['go_at_least']:g}", "details": det}


def _paired(dec, ctx):
    recs = select(ctx.jsonl(dec["exp"], dec["file"]), dec.get("where"))
    rows = [r for r in recs if r.get("method") == dec.get("method", "duel-b") and r.get("status", "ok") == "ok"]
    if not rows:
        return {"verdict": "INCONCLUSIVE", "reason": "no duel-b record", "details": {}}
    per = []
    for r in rows:
        ta, tb = r.get("pair_t_a"), r.get("pair_t_b")
        if not (isinstance(ta, list) and isinstance(tb, list) and len(ta) == len(tb)):
            per.append({"case": r.get("case"), "class": "inconclusive", "why": "no pair samples (run with --keep-samples)"})
            continue
        p = stats.paired_ratio(ta, tb, dec.get("conf", 0.95))
        cls = stats.classify_pair(p, _num(r.get("spread_pct")) or 0.0, dec.get("noisy_spread_pct", stats.NOISY_SPREAD_PCT),
                                  dec.get("noisy_min_effect", stats.NOISY_MIN_EFFECT))
        per.append({"case": r.get("case"), "class": cls, "ratio": p.ratio, "lo": p.lo, "hi": p.hi, "n": p.n,
                    "bit_identical": r.get("bit_identical")})
    n = {c: sum(x["class"] == c for x in per) for c in ("faster", "tie", "slower", "inconclusive")}
    det = {"cases": per, "counts": n}
    if any(x.get("bit_identical") is False for x in per):
        return {"verdict": "KEEP", "reason": "a variant is not bit-identical to its reference", "details": det}
    if n["inconclusive"]:
        return {"verdict": "INCONCLUSIVE", "reason": f"{n['inconclusive']} case(s) without an interval", "details": det}
    mf = dec.get("min_faster", 1)
    if n["faster"] >= mf and n["slower"] == 0:
        return {"verdict": "ADOPT", "reason": f"{n['faster']} case(s) faster, none slower, {n['tie']} tie(s)", "details": det}
    if n["faster"] == 0:
        return {"verdict": "KEEP", "reason": f"no case faster ({n['slower']} slower, {n['tie']} tie)", "details": det}
    return {"verdict": "INCONCLUSIVE", "reason": f"mixed: {n['faster']} faster, {n['slower']} slower", "details": det}


def _guard(dec, ctx):
    recs = select(ctx.jsonl(dec["exp"], dec["file"]), dec.get("where"))
    base = select(ctx.base_jsonl(dec["baseline"]), dec.get("where"))
    key = dec["key"]
    f = dec["field"]
    sp = dec.get("spread_field", "spread_pct")

    def idx(rs):
        return {tuple(r.get(k) for k in key): r for r in rs if r.get("status", "ok") == "ok" and _num(r.get(f)) is not None}

    def noise(r):      # the robust spread when the record carries its repetitions, else the bench's range spread
        s = r.get("samples_s")
        if isinstance(s, list) and len(s) >= stats.MIN_PAIRS and f == "median_s":
            return stats.robust_spread_pct(s)
        return _num(r.get(sp)) or 0.0

    new, old = idx(recs), idx(base)
    flagged, checked = [], 0
    for k, r in new.items():
        b = old.get(k)
        if b is None:
            continue
        checked += 1
        fl, slow, tol = stats.guard(b[f], noise(b), r[f], noise(r),
                                    dec.get("floor", stats.GUARD_FLOOR), dec.get("k", stats.GUARD_K))
        if fl:
            flagged.append({"key": list(k), "slowdown": slow, "tolerance": tol})
    if checked == 0:
        return {"verdict": "INCONCLUSIVE", "reason": "no record matches the baseline", "details": {}}
    det = {"checked": checked, "flagged": flagged}
    if flagged:
        return {"verdict": "FLAGGED", "reason": f"{len(flagged)} of {checked} slower than max(floor, k x spread)", "details": det}
    return {"verdict": "PASS", "reason": f"{checked} compared, none beyond the guard", "details": det}


CASE_CLASS = {"s256": "small", "s512": "small", "s_cc16": "small", "s_batch": "small", "sq1024": "mid squares",
              "sq2048": "mid squares", "sq4096": "large squares", "sq8192": "large squares", "r_lowk": "rectangular",
              "r_bigk": "rectangular", "r_skinny": "rectangular", "r_mm": "rectangular", "t_df": "tensor contractions",
              "t_cc": "tensor contractions", "t_permout": "tensor contractions", "t_batch": "tensor contractions",
              "t_ikl": "tensor contractions"}


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _parity(dec, ctx):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # bench/summarize.py
    import summarize as sm
    recs = [r for r in select(ctx.jsonl(dec["exp"], dec["file"]), dec.get("where")) if r.get("bench_mode") != "energy"]
    fa, fb = dec.get("a_family", "imd"), dec.get("b_family", "gemmul8-fast")
    excl = set(dec.get("exclude", []))
    rows = []
    for name, rs_ in sm.cases(recs).items():
        fams = sm.case_families(rs_)
        _, _, best = sm.dgemm_level(fams, "tflops_eq")
        ra, rb = best.get(fa), best.get(fb)
        row = {"case": name, "class": CASE_CLASS.get(name, "other"), "excluded": name in excl}
        if ra is None or rb is None:
            row["note"] = f"no DGEMM-level record for {fa if ra is None else fb}"
        else:
            spread = max(_num(ra.get("spread_pct")) or 0.0, _num(rb.get("spread_pct")) or 0.0)
            ratio = ra["tflops_eq"] / rb["tflops_eq"]
            row.update(ratio=ratio, spread_pct=spread, ratio_plus_spread=ratio * (1.0 + spread / 100.0),
                       a=ra["tflops_eq"], b=rb["tflops_eq"], a_cfg=ra.get("N"), b_cfg=rb.get("N"))
        rows.append(row)
    counted = [r for r in rows if not r["excluded"]]
    missing = [r["case"] for r in counted if "ratio" not in r]
    if not counted or missing:
        return {"verdict": "INCONCLUSIVE", "reason": "no comparison for: " + ", ".join(missing or ["any case"]), "details": {"cases": rows}}
    gm = _geomean([r["ratio"] for r in counted])
    gm_all = _geomean([r["ratio"] for r in rows if "ratio" in r])
    by_class = {}
    for r in counted:
        by_class.setdefault(r["class"], []).append(r["ratio"])
    low = [r["case"] for r in counted if r["ratio_plus_spread"] < dec["case_floor"]]
    det = {"cases": rows, "geomean": gm, "geomean_all_cases": gm_all, "below_floor": low,
           "class_geomean": {k: _geomean(v) for k, v in by_class.items()}}
    ok = gm >= dec["geomean_at_least"] and not low
    why = f"geomean {gm:.3f} over {len(counted)} cases (limit {dec['geomean_at_least']:g})"
    if low:
        why += f"; below {dec['case_floor']:g} even with spread: {', '.join(low)}"
    return {"verdict": "PASS" if ok else "FAIL", "reason": why, "details": det}


def _tile_refit(dec, ctx):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # bench/k4_fit.py
    try:
        import k4_fit
    except ImportError as e:     # numpy missing on the analysis machine
        return {"verdict": "INCONCLUSIVE", "reason": f"the tile fit needs numpy: {e}", "details": {}}
    recs = [r for r in select(ctx.jsonl(dec["exp"], dec["file"]), dec.get("where")) if r.get("case") not in dec.get("exclude", [])]
    sms = {int(r["sm_count"]) for r in recs if r.get("sm_count")}
    if len(sms) != 1:
        return {"verdict": "INCONCLUSIVE", "reason": f"records from {len(sms)} SM counts (need exactly one card)", "details": {}}
    d = k4_fit.decide(k4_fit.obs_from_dicts(recs), sms.pop())
    return {"verdict": d["verdict"], "reason": d["reason"], "details": {k: v for k, v in d.items() if k not in ("verdict", "reason")}}


def _rate_projection(dec, ctx):
    """The wgmma go / no-go rule (issue 16, reviewed by the independent reviewer 2026-10-15). Per case (sq4096, sq8192), N = 14:
         ops = 2 N b m n_hat k_hat, K4 and P (= all other phases) from the immerdink phase marks, R_K4 = ops / K4
         s = T_imd / (P + K4) (the whole-call anchor; outside [s_min, s_max] the measurement is not trusted)
         Q(R, f) = s (f P + ops / R) / T*      (> window: the projected immerdink is slower than GEMMul8 fast by more than the window)
       T_imd and T* come from the interleaved xduel records (immerdink and GEMMul8 fast), R_stock is the best bit-exact stock
       CUTLASS Hopper rate on K4's launch shape (m, n_hat, k_hat, batch N b; int8 output preferred, but only for a variant that also ran
       bit-exact with int32 output), R_ref = max(best exact stock rate of any output type, cuBLASLt),
       u the noise haircut: max(u_floor, the spreads of GEMMul8, immerdink, the stock and cuBLASLt records), or max(u_noisy, ...) if any
       spread is above noisy_spread_pct; a missing spread, or u >= 1, makes the case INCONCLUSIVE. The xduel-imd record is the one written
       right after the GEMMul8 record it was paired with. Every parameter is in the decisions file; `cases` must be sq4096 and sq8192.
         A  R_stock / R_K4 >= gain_go                       B  Q(R_stock (1 - u), pack_factor_go) <= window
         C  mma_margin R_mma_peak < the rate at which Q(R, pack_factor_go) = window (mma.sync tuning alone cannot reach it)
         O  Q(R_ref (1 + u), pack_factor_opt) <= window     (the optimistic case: if even this fails, wgmma cannot help)
       GO: A, B, C at every case with all inputs valid. NO-GO: O fails at some case. Else INCONCLUSIVE.
       role confirm (H200): PASS if R_stock / R_K4 >= confirm_gain at every case, FAIL if below, INCONCLUSIVE if data are missing."""
    n_mod = dec["n"]
    if sorted(dec["cases"]) != sorted(RATE_CASES):
        return {"verdict": "INCONCLUSIVE", "reason": f"cases are not the registered {list(RATE_CASES)}", "details": {"cases": []}}
    sweep = ctx.jsonl(dec["exp"], dec["file"])
    xd = ctx.jsonl(dec["xduel_exp"], dec["xduel_file"])
    ce = ctx.jsonl(dec["ceiling_exp"], dec["ceiling_file"])
    role = dec.get("role", "binding")
    s_lo, s_hi = dec["s_min"], dec["s_max"]
    rows, why_inconclusive = [], []
    for case in dec["cases"]:
        row = {"case": case}
        rows.append(row)
        rec = next((r for r in sweep if r.get("case") == case and r.get("method") == "imd" and r.get("N") == n_mod
                    and r.get("phases") and r.get("status", "ok") == "ok"), None)
        g8 = next((r for r in xd if r.get("case") == case and r.get("method") in ("gemmul8", "perm+gemmul8") and r.get("g8_mode") == "fast"
                   and r.get("N") == dec["g8_n"] and r.get("status", "ok") == "ok" and _num(r.get("median_s"))), None)
        # the xduel-imd record is written right after the baseline record it was paired with: take the one that follows this GEMMul8 record
        im = None
        if g8 is not None:
            for r in xd[xd.index(g8) + 1:]:
                if r.get("case") == case and r.get("method") == "xduel-imd":
                    im = r if r.get("status", "ok") == "ok" and _num(r.get("median_s")) else None
                    break
        if rec is None or g8 is None or im is None:
            row["note"] = "missing: " + ", ".join(n for n, x in (("immerdink phase record", rec), ("GEMMul8 fast xduel record", g8),
                                                                   ("xduel-imd record", im)) if x is None)
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        ph = rec["phases"]
        if not all(_num(v.get("ms")) is not None for v in ph.values()):
            row["note"] = "a phase mark is not a number"
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        k4 = sum(v["ms"] for k, v in ph.items() if k.startswith("K4")) * 1e-3
        pk = sum(v["ms"] for k, v in ph.items() if not k.startswith("K4")) * 1e-3
        m, n, k, b = rec["m"], rec["n"], rec["k"], rec.get("b", 1)
        nh, kh = -(-n // 16) * 16, rec.get("k_hat") or k
        ops = 2.0 * n_mod * b * m * nh * kh
        scale = (im.get("N") or n_mod) / n_mod         # immerdink at another N for DGEMM-level accuracy: ops, P, K4 scale
        ops, k4, pk = ops * scale, k4 * scale, pk * scale
        t_imd, t_star = im["median_s"], g8["median_s"]
        if not (k4 > 0 and pk > 0 and ops > 0 and t_imd > 0 and t_star > 0):
            row["note"] = "a time or an operation count is not positive"
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        r_k4 = ops / k4 / 1e12
        s_ = t_imd / (pk + k4)
        sp3 = [_num(g8.get("spread_pct")), _num(im.get("spread_pct")), _num(rec.get("spread_pct"))]
        spreads = [x for x in sp3 if x is not None]
        shape = (m, nh, kh, int(n_mod * b))
        match = [r for r in ce if r.get("status") == "ok" and (r.get("m"), r.get("n"), r.get("k"), r.get("batch", 1)) == shape
                 and r.get("exact") is True and _num(r.get("tops"))]
        stock = [r for r in match if str(r.get("variant", "")).startswith("hopper:")]
        # int8 output is what K4 stores, but its exactness check is weak (almost every output saturates): an int8 rate counts only if the same
        # variant and shape also ran bit-exact with int32 output
        i32_variants = {r["variant"] for r in stock if r.get("out") != "i8"}
        pref = [r for r in stock if r.get("out") == "i8" and r["variant"] in i32_variants] or [r for r in stock if r.get("out") != "i8"]
        r_stock = max((r["tops"] for r in pref), default=None)
        r_stock_any = max((r["tops"] for r in stock), default=None)      # best exact stock rate of any output type: the optimistic case's reference
        lt = [r for r in ce if r.get("status") == "ok" and r.get("variant") == "cublaslt" and (r.get("m"), r.get("n"), r.get("k")) == shape[:3]
              and r.get("exact") is True and _num(r.get("tops"))]
        r_lt = max((r["tops"] for r in lt), default=None)
        r_mma = max((r["tops"] for r in ce if r.get("variant") == "mma_peak" and r.get("status") == "ok" and _num(r.get("tops"))), default=None)
        if any(r.get("variant") == "mma_peak_check" and r.get("exact") is False for r in ce):
            r_mma = None                                                 # the closed-form check of the synthetic peak failed
        sp_of = lambda recs, top: [_num(r.get("spread_pct")) for r in recs if r["tops"] == top][:1]
        sp3 += sp_of(pref, r_stock) + sp_of(lt, r_lt)
        if any(x is None for x in sp3):
            row["note"] = "a spread is missing: the noise haircut cannot be computed"
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        spreads = sp3
        noisy = any(x > dec["noisy_spread_pct"] for x in spreads)
        u = max([dec["u_noisy"] if noisy else dec["u_floor"]] + [x / 100.0 for x in spreads])
        if u >= 1.0:
            row["note"] = f"a spread of {100 * u:.0f}% or more: the haircut would make the rate zero or negative"
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        row.update(ops=ops, k4_ms=k4 * 1e3, p_ms=pk * 1e3, r_k4=r_k4, t_imd_ms=t_imd * 1e3, tstar_ms=t_star * 1e3, s=s_, u=u,
                   r_stock=r_stock, r_cublaslt=r_lt, r_mma=r_mma, stock_out="i8" if pref and pref[0].get("out") == "i8" else "i32",
                   stock_configs=len(stock))
        if not (s_lo <= s_ <= s_hi):
            row["note"] = f"whole-call anchor s = {s_:.3f} outside [{s_lo}, {s_hi}]: phase marks and call disagree"
            why_inconclusive.append(f"{case}: {row['note']}")
            continue
        Q = lambda R, f: s_ * (f * pk + ops / (R * 1e12)) / t_star
        w = dec["window"]
        pg, po = dec["pack_factor_go"], dec["pack_factor_opt"]
        if r_stock is not None:
            row["gain"] = r_stock / r_k4
            row["A"] = row["gain"] >= dec["gain_go"]
            row["q_go"] = Q(r_stock * (1 - u), pg)
            row["B"] = row["q_go"] <= w
        den = w * t_star / s_ - pg * pk
        row["r_needed"] = ops / den / 1e12 if den > 0 else None      # None: no K4 rate reaches the window with this pack/CRT time
        if r_mma is not None:
            row["C"] = row["r_needed"] is None or dec["mma_margin"] * r_mma < row["r_needed"]
        r_ref = max([x for x in (r_stock_any, r_lt) if x is not None], default=None)
        if r_ref is not None:
            row["q_opt"] = Q(r_ref * (1 + u), po)
            row["O"] = row["q_opt"] <= w
    det = {"cases": rows}
    if role == "confirm":
        g = [r.get("gain") for r in rows]
        if why_inconclusive or any(x is None for x in g):
            return {"verdict": "INCONCLUSIVE", "reason": "; ".join(why_inconclusive) or "no stock rate", "details": det}
        low = [r["case"] for r in rows if r["gain"] < dec["confirm_gain"]]
        return {"verdict": "FAIL" if low else "PASS", "details": det,
                "reason": ("gain below %g at %s" % (dec["confirm_gain"], ", ".join(low))) if low else
                          "stock / K4 gain " + ", ".join(f"{r['case']} {r['gain']:.2f}" for r in rows)}
    # a case whose optimistic projection fails decides NO-GO even if the other case is missing
    failing_o = [r["case"] for r in rows if r.get("O") is False]
    if failing_o:
        return {"verdict": "NO-GO", "details": det,
                "reason": "even the best demonstrated rate with pack/CRT at %g of today's time misses the %g window at %s" %
                          (dec["pack_factor_opt"], dec["window"], ", ".join(failing_o))}
    if why_inconclusive:
        return {"verdict": "INCONCLUSIVE", "reason": "; ".join(why_inconclusive), "details": det}
    if any(r.get(k) is None for r in rows for k in ("A", "B", "C", "O")):
        return {"verdict": "INCONCLUSIVE", "reason": "a stock Hopper rate, cuBLASLt or the mma.sync peak is missing", "details": det}
    if all(r["A"] and r["B"] and r["C"] for r in rows):
        return {"verdict": "GO", "details": det,
                "reason": "gain %s; projected within %g of GEMMul8 fast with pack/CRT at %g" %
                          (", ".join(f"{r['case']} {r['gain']:.2f}" for r in rows), dec["window"], dec["pack_factor_go"])}
    why = [f"{r['case']}: " + ", ".join(k for k in ("A", "B", "C") if not r[k]) + " fail" for r in rows if not (r["A"] and r["B"] and r["C"])]
    return {"verdict": "INCONCLUSIVE", "reason": "O holds, GO conditions do not: " + "; ".join(why), "details": det}


_EVAL = {"threshold": _threshold, "paired": _paired, "guard": _guard, "parity": _parity, "rate_projection": _rate_projection,
         "tile_refit": _tile_refit}
