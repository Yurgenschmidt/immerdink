"""Decisions: the formulas that turn measured records into a verdict, written before the data exist.

A decisions file (TOML, pinned by sha256 in the manifest) has one [[decision]] table per question:

    [[decision]]
    id = "wgmma-go"   kind = "threshold"   requires = ["ceil.int8"]
    exp = "ceil.int8"  file = "ceil.jsonl"  where = {method = "ceil"}  field = "tops"
    aggregate = "max"  go_at_least = 1200.0  nogo_below = 900.0

Kinds (each a pure function of the records; a verdict is one of VERDICTS):
  threshold  one number from the records against two limits: GO / NO-GO / INCONCLUSIVE
  paired     interleaved A/B records (duel-b rows with pair_t_a / pair_t_b arrays): per case a
             faster / tie / slower class by the tie rule (stats.py); ADOPT / KEEP / INCONCLUSIVE
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

VERDICTS = ("ADOPT", "KEEP", "GO", "NO-GO", "PASS", "FLAGGED", "INCONCLUSIVE", "INVALID")
AGGREGATES = ("max", "min", "median", "geomean", "mean")
PLACEHOLDER = re.compile(r"\b(TBD|TODO|FIXME|XXX)\b|\?\?")

# kind -> {param: type or tuple of types}; "requires" and "id" are common to all kinds
REQUIRED = {
    "threshold": {"exp": str, "file": str, "field": str, "aggregate": str,
                  "go_at_least": (int, float), "nogo_below": (int, float)},
    "paired": {"exp": str, "file": str},
    "guard": {"exp": str, "file": str, "baseline": str, "baseline_sha256": str, "key": list, "field": str},
}
OPTIONAL = {
    "threshold": {"where": dict, "classes": list},
    "paired": {"where": dict, "min_faster": int, "noisy_spread_pct": (int, float),
               "noisy_min_effect": (int, float), "conf": (int, float), "classes": list},
    "guard": {"where": dict, "floor": (int, float), "k": (int, float), "classes": list,
              "spread_field": str},
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
    rows = [r for r in recs if r.get("method") == "duel-b" and r.get("status", "ok") == "ok"]
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


_EVAL = {"threshold": _threshold, "paired": _paired, "guard": _guard}
