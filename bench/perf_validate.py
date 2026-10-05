#!/usr/bin/env python3
"""Validate the performance model (DESIGN v1.8 / v1.10 §6c.5) against measured imd_bench records.

    python3 bench/perf_validate.py validate.jsonl [...] [--calib file.calib] [--lib ...]
                                   [--per-call] [--phases] [--markdown] [--burst]
                                   [--bands out.calib] [--sustained sustained.jsonl ...]

For every timed call of the records (method imd, and the plain / packed contract and imd_pack of
method packed) it compares the predicted total (imd_perf_predict with the given calibration,
default: the shipped one) with the measured timed median, and, with --phases, every predicted
phase with the measured phase median. The error distribution is printed per size class of the
measured call (< 100 us, 100 us - 1 ms, >= 1 ms) against the §6c.5 targets; TFLOPS-eq first,
time alongside.

v1.10: each record is predicted in the cache state it was measured in (REPEAT: the bench's
repeat calls; COLD for --l2 flush) at its measured SM clock (clock-conditioned; --burst: at the
calibration's reference clock). The prediction bands (§6c.1) are checked: the share of calls
inside their [p10, p90] band, by 2-fold cross-validation over the cases (bands from one half,
coverage on the other); --bands out.calib writes the calibration with the bands of the whole
hold-out set (format 2). --sustained: the SUSTAINED prediction of method-sustained windows
(power model as calibrated; the leave-one-out check is perf_fit's).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402
import perf_fit  # noqa: E402

CLASSES = [("< 100 us", 0, 100e-6, 0.15, 0.30), ("100 us - 1 ms", 100e-6, 1e-3, 0.10, 0.20),
           (">= 1 ms", 1e-3, float("inf"), 0.05, 0.15)]
BAND_KEYS = ("band.call_small", "band.call_mid", "band.call_large")


def evaluate(samples, calib, lib=None):
    """Per sample: (label, measured s, predicted s, flops, [(phase, measured s, predicted s,
    class)], case, p10 s, p90 s), the prediction from the C library (imd_perf_predict with
    `calib`)."""
    rows = []
    for s in samples:
        p = s.again(calib)
        ph = [(q.name, s.measured[q.name], q.time_s, q.cls) for q in p.phases
              if q.name in s.measured]
        rows.append((s.label, s.timed_s, p.time_s, p.flops_eq, ph, s.case, p.time_p10_s,
                     p.time_p90_s))
    return rows


def summarize(rows, markdown=False):
    ok = True
    out = []
    hdr = ("| measured call | calls | median |err| | p90 |err| | max |err| | target (median / p90) | met |\n"
           "|---|---|---|---|---|---|---|") if markdown else \
        f"{'measured call':<15} {'calls':>5} {'median':>8} {'p90':>8} {'max':>8}   target"
    out.append(hdr)
    for name, lo, hi, tmed, tp90 in CLASSES:
        e = [abs(r[2] - r[1]) / r[1] for r in rows if lo <= r[1] < hi]
        if not e:
            continue
        med, p90, mx = float(np.median(e)), float(np.percentile(e, 90)), max(e)
        met = med <= tmed and p90 <= tp90
        ok = ok and met
        if markdown:
            out.append(f"| {name} | {len(e)} | {100 * med:.1f} % | {100 * p90:.1f} % | "
                       f"{100 * mx:.1f} % | {100 * tmed:.0f} % / {100 * tp90:.0f} % | "
                       f"{'yes' if met else 'NO'} |")
        else:
            out.append(f"{name:<15} {len(e):>5} {100 * med:>7.1f}% {100 * p90:>7.1f}% "
                       f"{100 * mx:>7.1f}%   {100 * tmed:.0f}% / {100 * tp90:.0f}% "
                       f"{'met' if met else 'MISSED'}")
    return ok, "\n".join(out)


def phase_summary(rows, markdown=False):
    per = {}
    for r in rows:
        if r[1] < 1e-3:
            continue
        for name, mt, pt, _ in r[4]:
            if mt > 0:
                per.setdefault(name, []).append(abs(pt - mt) / mt)
    out = ["| phase (calls >= 1 ms) | n | median |err| | p90 |err| |\n|---|---|---|---|"
           if markdown else f"{'phase (calls >= 1 ms)':<22} {'n':>4} {'median':>8} {'p90':>8}"]
    for name in host.PERF_PHASES:
        if name not in per:
            continue
        e = per[name]
        med, p90 = float(np.median(e)), float(np.percentile(e, 90))
        out.append(f"| {name} | {len(e)} | {100 * med:.1f} % | {100 * p90:.1f} % |" if markdown
                   else f"{name:<22} {len(e):>4} {100 * med:>7.1f}% {100 * p90:>7.1f}%")
    return "\n".join(out)


# ---- v1.10: prediction bands ---------------------------------------------------------------

def size_class(t):
    return 0 if t < 100e-6 else 1 if t < 1e-3 else 2


def band_values(rows, lo_q=10, hi_q=90, min_n=5):
    """{band key: (p10, p90)} of measured / predicted: per size class of the PREDICTED call, per
    kernel class for the phases; a class with fewer than min_n values takes the pooled band."""
    calls = [[], [], []]
    phases = {}
    for r in rows:
        if r[2] > 0:
            calls[size_class(r[2])].append(r[1] / r[2])
        for _, mt, pt, cls in r[4]:
            if pt > 0 and mt > 0 and cls != "call":
                phases.setdefault(cls, []).append(mt / pt)
    pooled = [x for c in calls for x in c]
    pooled_ph = [x for v in phases.values() for x in v]

    def band(v, fallback):
        v = v if len(v) >= min_n else fallback
        if not v:
            return (1.0, 1.0)
        lo, hi = float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))
        lo = max(lo, 1e-6)
        return (min(lo, 1.0), max(hi, 1.0))  # the band contains the prediction itself

    out = {k: band(calls[i], pooled) for i, k in enumerate(BAND_KEYS)}
    for cls in host.PERF_CLASSES:
        out[f"band.{cls}"] = (1.0, 1.0) if cls == "call" else band(phases.get(cls, []), pooled_ph)
    return out


def coverage(rows, bands):
    """Share of calls whose measured time lies in [p10, p90] x predicted under `bands`, overall
    and per size class of the measured call."""
    inside = [[], [], []]
    for r in rows:
        lo, hi = bands[BAND_KEYS[size_class(r[2])]]
        inside[size_class(r[1])].append(lo * r[2] <= r[1] <= hi * r[2])
    flat = [x for c in inside for x in c]
    return (float(np.mean(flat)) if flat else float("nan"),
            [float(np.mean(c)) if c else float("nan") for c in inside], [len(c) for c in inside])


def cross_validated_coverage(rows, lo_q=10, hi_q=90, splits=20):
    """2-fold cross-validation over the cases, repeated over `splits` seeded random halvings:
    bands (quantiles lo_q / hi_q) from one half, coverage on the other, both ways, pooled."""
    cases = sorted({r[5] for r in rows})
    res = []
    for seed in range(splits):
        perm = list(np.random.default_rng(seed).permutation(cases))
        half = [set(perm[0::2]), set(perm[1::2])]
        for i in (0, 1):
            b = band_values([r for r in rows if r[5] in half[i]], lo_q, hi_q)
            for r in rows:
                if r[5] in half[i]:
                    continue
                lo, hi = b[BAND_KEYS[size_class(r[2])]]
                res.append((size_class(r[1]), lo * r[2] <= r[1] <= hi * r[2]))
    per = [[x for c, x in res if c == k] for k in range(3)]
    return (float(np.mean([x for _, x in res])) if res else float("nan"),
            [float(np.mean(c)) if c else float("nan") for c in per],
            [len(c) // (2 * splits) * 2 for c in per])


# The quantile pairs tried for the stored [p10, p90] band (DESIGN §6c.1): the narrowest whose
# cross-validated coverage reaches 80 % (in-sample p10 / p90 of a few hundred calls under-cover
# new cases: the residuals of one case are correlated).
QUANTILE_PAIRS = ((10, 90), (7.5, 92.5), (5, 95), (2.5, 97.5), (1, 99))


def choose_quantiles(rows, target=0.80):
    for lo, hi in QUANTILE_PAIRS:
        cov = cross_validated_coverage(rows, lo, hi)
        if cov[0] >= target:
            return lo, hi, cov
    return lo, hi, cov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+")
    ap.add_argument("--calib", default=None, help="calibration file (default: the shipped one)")
    ap.add_argument("--lib", default=None, help="libimmerdink_host.so")
    ap.add_argument("--per-call", action="store_true", help="print every call")
    ap.add_argument("--phases", action="store_true", help="per-phase errors (calls >= 1 ms)")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--burst", action="store_true",
                    help="predict at the reference clock, not the records' measured clocks")
    ap.add_argument("--bands", default=None, metavar="OUT",
                    help="write the calibration with the hold-out bands to OUT (format 2)")
    ap.add_argument("--sustained", nargs="*", default=[],
                    help="method-sustained records: SUSTAINED predictions vs measured")
    a = ap.parse_args(argv)
    perf_fit.USE_RECORD_CLOCK = not a.burst
    records = []
    for f in a.records:
        records += [json.loads(line) for line in open(f) if line.strip()]
    calib = host.Calibration.load(a.calib, a.lib) if a.calib else host.Calibration.default(a.lib)
    rows = evaluate(perf_fit.samples_from_records(records, a.lib, calib), calib, a.lib)
    if a.per_call:
        print(f"{'call':<44} {'TFLOPS-eq meas':>14} {'pred':>8} {'time meas':>11} {'pred':>11} {'err':>7}")
        for r in sorted(rows, key=lambda r: r[1]):
            lab, m, p, fl = r[:4]
            tm, tp = (fl / m * 1e-12, fl / p * 1e-12) if fl > 0 else (float("nan"),) * 2
            print(f"{lab[:44]:<44} {tm:>14.2f} {tp:>8.2f} {1e3 * m:>9.4f}ms {1e3 * p:>9.4f}ms "
                  f"{100 * (p - m) / m:>+6.1f}%")
    ok, table = summarize(rows, a.markdown)
    print(table)
    sub = [r for r in rows if re.search(r"N=1[567]\b", r[0])]
    if sub:  # the operating range of ehrenfest (informational)
        print("N = 15..17 only:")
        print(summarize(sub, a.markdown)[1])
    if a.phases:
        print(phase_summary(rows, a.markdown))
    if calib.file_format == 2:
        cov, per, n = coverage(rows, calib.bands)
        print(f"band coverage (this calibration's bands, in-sample if they came from these "
              f"records): {100 * cov:.0f} % (< 100 us {100 * per[0]:.0f} % of {n[0]}, "
              f"100 us - 1 ms {100 * per[1]:.0f} % of {n[1]}, >= 1 ms {100 * per[2]:.0f} % "
              f"of {n[2]})")
    for lo_q, hi_q in QUANTILE_PAIRS[:3]:
        cov, per, n = cross_validated_coverage(rows, lo_q, hi_q)
        print(f"band coverage of the hold-out quantiles {lo_q} / {hi_q}, 2-fold cross-validated "
              f"over the cases x 20 (target 80 %, 70-90 % per class): {100 * cov:.0f} % (< 100 us "
              f"{100 * per[0]:.0f} %, 100 us - 1 ms {100 * per[1]:.0f} %, >= 1 ms "
              f"{100 * per[2]:.0f} %; calls {n[0]} / {n[1]} / {n[2]})")
    if a.bands:
        lo_q, hi_q, cov = choose_quantiles(rows)
        print(f"stored band = the hold-out quantiles {lo_q} / {hi_q} (cross-validated coverage "
              f"{100 * cov[0]:.0f} %)")
        b = band_values(rows, lo_q, hi_q)
        print("bands: " + ", ".join(f"{k[5:]} {lo:.3f}-{hi:.3f}" for k, (lo, hi) in b.items()))
        extra = calib.extra if calib.file_format == 2 else perf_fit.seed_extra(a.lib)
        extra = dict(extra, **b)
        params = dict(calib.params)
        for k in host.FORMAT2_PARAMS:  # a format-1 calibration: its chain parameters are 0
            params.setdefault(k, 0.0)
        cal2 = host.Calibration.from_values(calib.meta, params, a.lib, extra)
        with open(a.bands, "w") as f:
            f.write(cal2.format())
        print(f"wrote {a.bands}")
    if a.sustained:
        recs = []
        for f in a.sustained:
            recs += [json.loads(line) for line in open(f) if line.strip()]
        srows = perf_fit.sustained_rows(recs, calib, a.lib)
        print(f"{'sustained window':<40} {'TFLOPS-eq meas':>14} {'pred':>7} {'t meas':>10} "
              f"{'t pred':>10} {'err':>7} {'f meas':>7} {'f pred':>7}")
        errs = []
        for r in srows:
            p = host.predict(r["spec"], r["ext"], N=r["N"], g=r["g"], vec_mode=r["vec"],
                             regime="SUSTAINED", cache="REPEAT", calib=calib, lib=a.lib)
            errs.append(abs(p.time_s - r["t"]) / r["t"])
            print(f"{r['label'][:40]:<40} {p.flops_eq / r['t'] * 1e-12:>14.2f} "
                  f"{p.tflops_eq:>7.2f} {1e3 * r['t']:>8.3f}ms {1e3 * p.time_s:>8.3f}ms "
                  f"{100 * (p.time_s - r['t']) / r['t']:>+6.1f}% {r['f']:>7.0f} {p.clock_mhz:>7.0f}")
        if errs:
            print(f"sustained: median |err| {100 * float(np.median(errs)):.1f} %, max "
                  f"{100 * max(errs):.1f} % over {len(errs)} windows")
    print(f"calibration: {calib.meta['gpu']} ({calib.meta['date']}, {calib.meta['version']}, "
          f"format {calib.file_format})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
