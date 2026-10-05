#!/usr/bin/env python3
"""Fit the performance-model calibration (DESIGN v1.8 §6c.2) from imd_bench records.

    python3 bench/perf_fit.py calib.jsonl [more.jsonl ...] -o out.calib [--lib libimmerdink_host.so]

Input: the JSON lines of a calibration sweep (bench/sweep_calib.txt through bench/run_sweep.sh,
or tools/calibrate.sh), made by the profiling build of imd_bench: per record the timed median of
imd_contract (median_s) and the median per-phase times ("phases"; packed records:
plain_/packed_/pack_phases, contract_packed_ms, pack_ms).

The features of every phase come from libimmerdink_host (imd_perf_predict through
python/immerdink/host.py): no feature formula lives here. Per kernel class the parameters
minimize the relative error sum(((theta . x - t) / t)^2) subject to theta >= 0 (exact NNLS by
enumerating the active sets of <= 4 parameters); then the call term (sign-free least squares of
(t_timed - sum of the predicted phases) / t_timed on (1, kernel_launches)). BETA (not measured
by the sweep) takes K5's launch and byte parameters. numpy only.
"""

import argparse
import datetime
import itertools
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402

CALL = "call"


# ------------------------------------------------------------------------------------------
# Records -> (predicted phases with features, measured times)

def parse_ext(s):
    return {kv.split("=")[0]: int(kv.split("=")[1]) for kv in s.split(",")}


def variant_options(rec):
    """N, g, vec_mode, mode, k4 override of an imd record (from its variant string)."""
    v = rec.get("variant", "")
    words = v.split()
    k4 = None
    for w in words:
        if w.startswith("k4=") and w[3:].isdigit():
            k4 = int(w[3:])
    if any(w == "k4=tune" for w in words):
        k4 = rec.get("k4_config")  # the tuned plan's configuration
    return dict(N=int(rec["N"]), g=int(rec.get("g", 0) or 0),
                vec_mode="RESIDENT" if "vec=resident" in v else "GENERATE",
                mode="DIRECT" if "mode=direct" in v else "EMULATE", k4_config=k4)


class Sample:
    """One measured call: its predicted phases (features) and the measured times."""

    def __init__(self, label, pred, measured_phases, timed_s, again):
        self.label = label            # "case variant N"
        self.pred = pred              # host.Prediction (features; times under some calibration)
        self.measured = measured_phases  # phase name -> seconds (phase marks)
        self.timed_s = timed_s        # timed median of the whole call (no marks)
        self.again = again            # calib -> host.Prediction (the C model, that calibration)


def samples_from_records(records, lib=None):
    out = []
    for r in records:
        if r.get("status") != "ok":
            continue
        meth = r.get("method")
        spec, ext = r.get("spec"), r.get("ext")
        if not spec or not ext:
            continue
        ext = parse_ext(ext)
        if meth == "imd" and "phases" in r:
            o = variant_options(r)
            def again(calib, spec=spec, ext=ext, o=o):
                return host.predict(spec, ext, N=o["N"], g=o["g"], vec_mode=o["vec_mode"],
                                    mode=o["mode"], k4_config=o["k4_config"], calib=calib,
                                    lib=lib)
            p = again(None)
            if p.status != "SUCCESS":
                continue
            meas = {k: v["ms"] * 1e-3 for k, v in r["phases"].items() if v.get("ms") is not None}
            out.append(Sample(f"{r['case']} {r['variant']} N={r['N']}", p, meas, r["median_s"],
                              again))
        elif meth == "packed" and "packed_phases" in r:
            which = 0 if r.get("pack") == "A" else 1
            N = int(r["N"])
            pk = host.pack_desc_for(spec, ext, operand=which, N=N, lib=lib)
            packing = (pk, None) if which == 0 else (None, pk)
            fns = {"plain_": lambda c, spec=spec, ext=ext, N=N: host.predict(
                       spec, ext, N=N, calib=c, lib=lib),
                   "packed_": lambda c, spec=spec, ext=ext, N=N, packing=packing: host.predict(
                       spec, ext, N=N, packing=packing, calib=c, lib=lib),
                   "pack_": lambda c, pk=pk: host.predict_pack(pk, calib=c, lib=lib)}
            lab = f"{r['case']} pack {r.get('pack')} N={N}"
            for pre, timed in (("plain_", r["contract_ms"] * 1e-3),
                               ("packed_", r["contract_packed_ms"] * 1e-3),
                               ("pack_", r["pack_ms"] * 1e-3)):
                pred = fns[pre](None)
                ph = r.get(pre + "phases")
                if ph is None or pred.status != "SUCCESS":
                    continue
                meas = {k: v["ms"] * 1e-3 for k, v in ph.items() if v.get("ms") is not None}
                out.append(Sample(f"{lab} {pre.rstrip('_')}", pred, meas, timed, fns[pre]))
    return out


# ------------------------------------------------------------------------------------------
# Fit

def nnls_relative(X, t):
    """argmin_{theta >= 0} sum(((X theta - t) / t)^2), exactly, by enumerating active sets
    (columns <= 4). Columns that are all zero get 0."""
    W = X / t[:, None]
    y = np.ones_like(t)
    ncol = X.shape[1]
    used = [j for j in range(ncol) if np.any(X[:, j] != 0)]
    scale = np.array([np.abs(W[:, j]).max() if j in used else 1.0 for j in range(ncol)])
    best, best_res = np.zeros(ncol), float(np.sum(y ** 2))
    for r in range(1, len(used) + 1):
        for S in itertools.combinations(used, r):
            A = W[:, S] / scale[list(S)]
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            if np.any(sol < 0):
                continue
            res = float(np.sum((A @ sol - y) ** 2))
            if res < best_res - 1e-15:
                theta = np.zeros(ncol)
                theta[list(S)] = sol / scale[list(S)]
                best, best_res = theta, res
    return best


def lstsq_relative(X, t):
    W = X / t[:, None]
    sol, *_ = np.linalg.lstsq(W, np.ones_like(t), rcond=None)
    return sol


def phase_rows(samples):
    """class -> list of (x, measured seconds, label)."""
    rows = {}
    for s in samples:
        for p in s.pred.phases:
            if p.cls == CALL or p.name not in s.measured:
                continue
            t = s.measured[p.name]
            if t <= 0:
                continue
            rows.setdefault(p.cls, []).append((np.array(p.x), t, f"{s.label} {p.name}"))
    return rows


K4_BYTE = 4  # K4 classes: t = max(sum of the round-model terms j < 4, the DRAM term j = 4)


def is_k4(cls):
    return cls.startswith("k4_")


def predicted_ns(params, names, cls, x):
    """The model of one phase (DESIGN §6c.1; the C is perf.cpp perf_phase_seconds): the sum of
    the terms, for K4 the max of the round model and the DRAM term."""
    terms = [params.get(f"{cls}.{n}", 0.0) * x[j] if n else 0.0 for j, n in enumerate(names[cls])]
    if is_k4(cls):
        return max(sum(terms[:K4_BYTE]), terms[K4_BYTE])
    return sum(terms)


def fit_k4(X, t, byte_col):
    """max(round model, DRAM term) (a roofline): for each candidate DRAM rate (bandwidths from
    100 GB/s to 2 TB/s, or none) the round model is fitted (NNLS, relative) on the phases the
    DRAM term does not explain (its time below 90 % of the measured), and the rate with the
    smallest total relative error of the max is kept."""
    comp = [j for j in range(X.shape[1]) if j != byte_col]
    xb = X[:, byte_col]
    best, best_err = None, float("inf")
    for bw in [None] + list(np.geomspace(100e9, 2000e9, 61)):
        rate = 0.0 if bw is None else 1.0 / bw
        sel = rate * xb < 0.9 * t
        if sel.sum() < len(comp):
            continue
        th = np.zeros(X.shape[1])
        th[comp] = nnls_relative(X[sel][:, comp], t[sel])
        th[byte_col] = rate
        err = float(np.sum(((np.maximum(X[:, comp] @ th[comp], rate * xb) - t) / t) ** 2))
        if err < best_err:
            best, best_err = th, err
    return best


def fit(samples, lib=None, verbose=True, worst=0):
    names = host.param_names(lib)
    rows = phase_rows(samples)
    params = {}
    report = []
    for cls in host.PERF_CLASSES:
        if cls in (CALL, "beta"):
            continue
        used = [j for j, n in enumerate(names[cls]) if n]
        data = rows.get(cls, [])
        if not data:
            report.append((cls, 0, None, None))
            continue
        X = np.array([d[0][used] for d in data])
        t = np.array([d[1] for d in data])
        if is_k4(cls):
            theta = fit_k4(X, t, used.index(K4_BYTE))
        else:
            theta = nnls_relative(X, t)
        for j, v in zip(used, theta):
            params[f"{cls}.{names[cls][j]}"] = float(v) * 1e9  # s -> ns per unit
        pred = np.array([predicted_ns({f"{cls}.{names[cls][j]}": v for j, v in zip(used, theta)},
                                      names, cls, d[0]) for d in data])
        err = np.abs(pred - t) / t
        report.append((cls, len(data), float(np.median(err)), float(np.percentile(err, 90))))
        if worst:
            for i in np.argsort(-err)[:worst]:
                print(f"    {cls:<11} {100 * (pred[i] - t[i]) / t[i]:+7.1f}%  "
                      f"{1e3 * t[i]:9.4f} ms  {data[i][2]}")
    # Classes the sweep does not exercise (documented stand-ins).
    for cls in host.PERF_CLASSES:
        for n in names[cls]:
            if n and f"{cls}.{n}" not in params and cls != CALL:
                params[f"{cls}.{n}"] = 0.0
    params["beta.launch"] = params.get("k5.launch", 0.0)
    params["beta.byte"] = params.get("k5.byte", 0.0)
    # The call term: (t_timed - sum of the predicted phases) on (1, kernel_launches).
    Xc, rc, tc = [], [], []
    for s in samples:
        tp = sum(predicted_ns(params, names, p.cls, p.x) for p in s.pred.phases if p.cls != CALL)
        Xc.append([1.0, float(s.pred.kernel_launches)])
        rc.append(s.timed_s - tp * 1e-9)
        tc.append(s.timed_s)
    Xc, rc, tc = np.array(Xc), np.array(rc), np.array(tc)
    sol, *_ = np.linalg.lstsq(Xc / tc[:, None], rc / tc, rcond=None)
    params["call.call"], params["call.launch"] = float(sol[0] * 1e9), float(sol[1] * 1e9)
    if verbose:
        print(f"{'class':<12} {'phases':>6} {'median|err|':>12} {'p90|err|':>9}")
        for cls, n, med, p90 in report:
            if med is None:
                print(f"{cls:<12} {n:>6}   (no data: parameters 0)")
            else:
                print(f"{cls:<12} {n:>6} {100 * med:>11.1f}% {100 * p90:>8.1f}%")
    return params


def clock_summary(records):
    clocks = sorted({int(r["gpu_after"].split()[0]) for r in records
                     if r.get("status") == "ok" and r.get("gpu_after", "").split()[:1]
                     and r["gpu_after"].split()[0].isdigit()})
    states = [r["clock_state"] for r in records if "clock_state" in r]
    locked = any(s.get("applications_clocks_setting") == "Active" for s in states)
    mx = next((r.get("gpu_max_clocks") for r in records if r.get("gpu_max_clocks")), "?")
    pw = sorted({s.get("power_w") for s in states if s.get("power_w")}, key=float)
    span = f"SM {clocks[0]}-{clocks[-1]} MHz after calls" if clocks else "SM ?"
    return (f"{'locked' if locked else 'stock, not locked'}; {span}; max {mx}; "
            f"post-warm-up power {pw[0] if pw else '?'}-{pw[-1] if pw else '?'} W")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+", help="JSON lines of a calibration sweep")
    ap.add_argument("-o", "--out", required=True, help="calibration file to write")
    ap.add_argument("--lib", default=None, help="libimmerdink_host.so")
    ap.add_argument("--version", default=None, help="version string (default: the commit)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--worst", type=int, default=0, help="print the N worst phases per class")
    ap.add_argument("--l2-bytes", type=float, default=72 * 2**20,
                    help="the GPU's L2 cache bytes (default: the RTX 4090's 72 MiB)")
    a = ap.parse_args(argv)
    records = []
    for f in a.records:
        records += [json.loads(line) for line in open(f) if line.strip()]
    samples = samples_from_records(records, a.lib)
    if not samples:
        sys.exit("no usable records (profiling build, method imd / packed, status ok)")
    params = fit(samples, a.lib, worst=a.worst)
    ok = [r for r in records if r.get("status") == "ok" and r.get("gpu")]
    commit = next((r.get("commit") for r in ok if r.get("commit")), "unknown")
    meta = {"gpu": ok[0]["gpu"], "sm_count": int(ok[0]["sm_count"]),
            "date": datetime.date.today().isoformat(),
            "version": a.version or f"immerdink {commit}, host API {host.API_VERSION}",
            "clocks": clock_summary(records)[:127],
            "notes": (a.notes or f"fit on {len(samples)} calls ({len(ok)} records) by "
                      f"bench/perf_fit.py")[:127],
            "peak_dram_bytes_per_s": 1008e9, "peak_int8_ops_per_s": 660.6e12,
            "peak_fp64_flops_per_s": 1.29e12, "l2_bytes": a.l2_bytes}
    if "4090" not in meta["gpu"]:
        print("note: the datasheet peaks are the RTX 4090's (edit the peak_* lines) and the "
              "fit used --l2-bytes for this GPU's L2",
              file=sys.stderr)
    cal = host.Calibration.from_values(meta, params, a.lib)
    with open(a.out, "w") as f:
        f.write(cal.format())
    # The fit's own error on the timed totals (in-sample).
    errs = []
    for s in samples:
        tp = sum(predicted_ns(params, host.param_names(a.lib), p.cls, p.x) for p in s.pred.phases)
        errs.append(abs(tp * 1e-9 - s.timed_s) / s.timed_s)
    print(f"in-sample total: median |err| {100 * statistics.median(errs):.1f}%, "
          f"p90 {100 * float(np.percentile(errs, 90)):.1f}%, max {100 * max(errs):.1f}% "
          f"over {len(errs)} calls")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
