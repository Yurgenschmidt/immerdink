#!/usr/bin/env python3
"""Validate the performance model (DESIGN v1.8 §6c.5) against measured imd_bench records.

    python3 bench/perf_validate.py validate.jsonl [...] [--calib file.calib] [--lib ...]
                                   [--per-call] [--phases] [--markdown]

For every timed call of the records (method imd, and the plain / packed contract and imd_pack of
method packed) it compares the predicted total (imd_perf_predict with the given calibration,
default: the shipped one) with the measured timed median, and, with --phases, every predicted
phase with the measured phase median. The error distribution is printed per size class of the
measured call (< 100 us, 100 us - 1 ms, >= 1 ms) against the §6c.5 targets; TFLOPS-eq first,
time alongside.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402
import perf_fit  # noqa: E402

CLASSES = [("< 100 us", 0, 100e-6, 0.15, 0.30), ("100 us - 1 ms", 100e-6, 1e-3, 0.10, 0.20),
           (">= 1 ms", 1e-3, float("inf"), 0.05, 0.15)]


def evaluate(samples, calib, lib=None):
    """Per sample: (label, measured s, predicted s, flops, [(phase, measured s, predicted s)]),
    the prediction from the C library (imd_perf_predict with `calib`)."""
    rows = []
    for s in samples:
        p = s.again(calib)
        ph = [(q.name, s.measured[q.name], q.time_s) for q in p.phases if q.name in s.measured]
        rows.append((s.label, s.timed_s, p.time_s, p.flops_eq, ph))
    return rows


def summarize(rows, markdown=False):
    ok = True
    out = []
    hdr = ("| measured call | calls | median |err| | p90 |err| | max |err| | target (median / p90) | met |\n"
           "|---|---|---|---|---|---|---|") if markdown else \
        f"{'measured call':<15} {'calls':>5} {'median':>8} {'p90':>8} {'max':>8}   target"
    out.append(hdr)
    for name, lo, hi, tmed, tp90 in CLASSES:
        e = [abs(p - m) / m for _, m, p, _, _ in rows if lo <= m < hi]
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
    for _, m, _, _, ph in rows:
        if m < 1e-3:
            continue
        for name, mt, pt in ph:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+")
    ap.add_argument("--calib", default=None, help="calibration file (default: the shipped one)")
    ap.add_argument("--lib", default=None, help="libimmerdink_host.so")
    ap.add_argument("--per-call", action="store_true", help="print every call")
    ap.add_argument("--phases", action="store_true", help="per-phase errors (calls >= 1 ms)")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    records = []
    for f in a.records:
        records += [json.loads(line) for line in open(f) if line.strip()]
    calib = host.Calibration.load(a.calib, a.lib) if a.calib else host.Calibration.default(a.lib)
    rows = evaluate(perf_fit.samples_from_records(records, a.lib), calib, a.lib)
    if a.per_call:
        print(f"{'call':<44} {'TFLOPS-eq meas':>14} {'pred':>8} {'time meas':>11} {'pred':>11} {'err':>7}")
        for lab, m, p, fl, _ in sorted(rows, key=lambda r: r[1]):
            tm, tp = (fl / m * 1e-12, fl / p * 1e-12) if fl > 0 else (float("nan"),) * 2
            print(f"{lab[:44]:<44} {tm:>14.2f} {tp:>8.2f} {1e3 * m:>9.4f}ms {1e3 * p:>9.4f}ms "
                  f"{100 * (p - m) / m:>+6.1f}%")
    ok, table = summarize(rows, a.markdown)
    print(table)
    if a.phases:
        print(phase_summary(rows, a.markdown))
    print(f"calibration: {calib.meta['gpu']} ({calib.meta['date']}, {calib.meta['version']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
