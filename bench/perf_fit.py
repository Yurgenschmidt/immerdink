#!/usr/bin/env python3
"""Fit the performance-model calibration (DESIGN v1.8 §6c.2) from imd_bench records.

    python3 bench/perf_fit.py calib.jsonl [more.jsonl ...] -o out.calib [--lib libimmerdink_host.so]

Input: the JSON lines of a calibration sweep (bench/sweep_calib.txt through bench/run_sweep.sh,
or tools/calibrate.sh), made by the profiling build of imd_bench: per record the timed median of
imd_contract (median_s) and the median per-phase times ("phases"; packed records:
plain_/packed_/pack_phases, contract_packed_ms, pack_ms).

The features of every phase come from libimmerdink_host (imd_perf_predict through
python/immerdink/host.py) under a SEED calibration of the measured device: its sm_count and
l2_bytes from the records (imd_bench records them; --sm-count / --l2-bytes override), so the
K4 configuration, the rounds and the DRAM term are the measured GPU's. No feature formula lives
here. Per kernel class the parameters minimize the relative error sum(((theta . x - t) / t)^2)
subject to theta >= 0 (exact NNLS by enumerating the active sets of <= 5 parameters); for the K4
classes the roofline max(round model, DRAM term) with a scan over the DRAM rate (--k4-shared-dram:
one rate for the four configurations). Then the call term: sign-free least squares of
(t_timed - the C model's phases) / t_timed on (1, kernel_launches), where the phases are
re-predicted by the C library with the fitted parameters (the C does the sum / max). BETA (not
measured by the sweep) takes K5's launch and byte parameters. Records with --data dominant
(one dominant entry per row) are predicted with eps fraction min(1, 512 / k); the others with
the typical-data default. Datasheet peaks: the RTX 4090's, else --peak-dram / --peak-int8 /
--peak-fp64 are required. numpy only.
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


def record_eps(rec, ext):
    """The eps fraction of a record's data: None (the typical-data default) for the normal
    generator; for one dominant entry per row, the tiles holding it: min(1, 512 / k)."""
    if rec.get("data") != "dominant":
        return None
    lw = host.lowering(rec["spec"], ext, N=int(rec.get("N", 8)))
    return min(1.0, 512.0 / max(lw.k, 1))


def samples_from_records(records, lib=None, seed=None):
    """seed: the calibration whose sm_count / l2_bytes the features use (the measured GPU)."""
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
            eps = record_eps(r, ext)

            def again(calib, spec=spec, ext=ext, o=o, eps=eps):
                return host.predict(spec, ext, N=o["N"], g=o["g"], vec_mode=o["vec_mode"],
                                    mode=o["mode"], k4_config=o["k4_config"], eps_fraction=eps,
                                    calib=calib, lib=lib)
            p = again(seed)
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
            eps = record_eps(r, ext)
            fns = {"plain_": lambda c, spec=spec, ext=ext, N=N, eps=eps: host.predict(
                       spec, ext, N=N, eps_fraction=eps, calib=c, lib=lib),
                   "packed_": lambda c, spec=spec, ext=ext, N=N, packing=packing, eps=eps:
                       host.predict(spec, ext, N=N, packing=packing, eps_fraction=eps, calib=c,
                                    lib=lib),
                   "pack_": lambda c, pk=pk, eps=eps: host.predict_pack(
                       pk, eps_fraction=eps, calib=c, lib=lib)}
            lab = f"{r['case']} pack {r.get('pack')} N={N}"
            for pre, timed in (("plain_", r["contract_ms"] * 1e-3),
                               ("packed_", r["contract_packed_ms"] * 1e-3),
                               ("pack_", r["pack_ms"] * 1e-3)):
                pred = fns[pre](seed)
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


def fit_k4(X, t, byte_col, rates=None):
    """max(round model, DRAM term) (a roofline): for each candidate DRAM rate (bandwidths from
    100 GB/s to 2 TB/s, or none) the round model is fitted (NNLS, relative) on the phases the
    DRAM term does not explain (its time below 90 % of the measured), and the rate with the
    smallest total relative error of the max is kept."""
    comp = [j for j in range(X.shape[1]) if j != byte_col]
    xb = X[:, byte_col]
    best, best_err = None, float("inf")
    if rates is None:
        rates = [0.0] + [1.0 / bw for bw in np.geomspace(100e9, 2000e9, 61)]
    for rate in rates:
        sel = rate * xb < 0.9 * t
        if sel.sum() < len(comp):
            continue
        th = np.zeros(X.shape[1])
        th[comp] = nnls_relative(X[sel][:, comp], t[sel])
        th[byte_col] = rate
        err = float(np.sum(((np.maximum(X[:, comp] @ th[comp], rate * xb) - t) / t) ** 2))
        if err < best_err:
            best, best_err = th, err
    return best, best_err


def fit(samples, seed, lib=None, verbose=True, worst=0, k4_shared_dram=False):
    names = host.param_names(lib)
    rows = phase_rows(samples)
    params = {}
    report = []
    shared = None
    if k4_shared_dram:  # one DRAM rate for the four K4 configurations: the best total
        best = None
        for rate in [0.0] + [1.0 / bw for bw in np.geomspace(100e9, 2000e9, 61)]:
            tot = 0.0
            for cls in host.PERF_CLASSES[4:8]:
                data = rows.get(cls, [])
                if not data:
                    continue
                used = [j for j, n in enumerate(names[cls]) if n]
                X = np.array([d[0][used] for d in data])
                t = np.array([d[1] for d in data])
                th = fit_k4(X, t, used.index(K4_BYTE), [rate])[0]
                if th is None:
                    tot = float("inf")
                    break
                tot += fit_k4(X, t, used.index(K4_BYTE), [rate])[1]
            if best is None or tot < best[1]:
                best = (rate, tot)
        shared = best[0]
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
            theta = fit_k4(X, t, used.index(K4_BYTE), None if shared is None else [shared])[0]
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
    # The call term: (t_timed - the C model's phases with these parameters, call term 0) on
    # (1, kernel_launches).
    params["call.call"], params["call.launch"] = 0.0, 0.0
    trial = host.Calibration.from_values(seed.meta, params, lib)
    Xc, rc, tc = [], [], []
    for s in samples:
        p = s.again(trial)
        Xc.append([1.0, float(p.kernel_launches)])
        rc.append(s.timed_s - p.time_s)
        tc.append(s.timed_s)
    Xc, rc, tc = np.array(Xc), np.array(rc), np.array(tc)
    sol, *_ = np.linalg.lstsq(Xc / tc[:, None], rc / tc, rcond=None)
    params["call.call"], params["call.launch"] = float(sol[0] * 1e9), float(sol[1] * 1e9)
    if verbose:
        rates = {c: params.get(f"{c}.byte", 0.0) for c in host.PERF_CLASSES[4:8]}
        print("K4 DRAM rates (GB/s): " + ", ".join(
            f"{c} {1 / v:.0f}" if v > 0 else f"{c} none" for c, v in rates.items()))
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
    ap.add_argument("--sm-count", type=int, default=None, help="override the records' sm_count")
    ap.add_argument("--l2-bytes", type=float, default=None,
                    help="override the records' l2_bytes (needed for records without it; "
                         "an RTX 4090 defaults to 72 MiB)")
    ap.add_argument("--peak-dram", type=float, default=None, help="datasheet DRAM bytes/s")
    ap.add_argument("--peak-int8", type=float, default=None, help="datasheet int8 ops/s")
    ap.add_argument("--peak-fp64", type=float, default=None, help="datasheet FP64 flops/s")
    ap.add_argument("--k4-shared-dram", action="store_true",
                    help="one DRAM rate for the four K4 configurations")
    a = ap.parse_args(argv)
    records = []
    for f in a.records:
        records += [json.loads(line) for line in open(f) if line.strip()]
    ok = [r for r in records if r.get("status") == "ok" and r.get("gpu")]
    if not ok:
        sys.exit("no usable records")
    gpus = {r["gpu"] for r in ok}
    if len(gpus) != 1:
        sys.exit(f"records of several GPUs: {sorted(gpus)}")
    gpu = ok[0]["gpu"]
    is4090 = "RTX 4090" in gpu
    sms = a.sm_count or int(ok[0]["sm_count"])
    l2 = a.l2_bytes or next((float(r["l2_bytes"]) for r in ok if r.get("l2_bytes")), None)
    if l2 is None:
        if not is4090:
            sys.exit("the records have no l2_bytes: pass --l2-bytes")
        l2 = 72 * 2**20
    peaks = {"peak_dram_bytes_per_s": a.peak_dram, "peak_int8_ops_per_s": a.peak_int8,
             "peak_fp64_flops_per_s": a.peak_fp64}
    for k, v in zip(peaks, (1008e9, 660.6e12, 1.29e12)):  # RTX 4090 datasheet
        if peaks[k] is None:
            if not is4090:
                sys.exit(f"{gpu}: pass --peak-dram, --peak-int8 and --peak-fp64 (datasheet)")
            peaks[k] = v
    commit = next((r.get("commit") for r in ok if r.get("commit")), "unknown")
    meta = {"gpu": gpu, "sm_count": sms, "date": datetime.date.today().isoformat(),
            "version": a.version or f"immerdink {commit}, host API {host.API_VERSION}",
            "clocks": clock_summary(records)[:127], **peaks, "l2_bytes": l2}
    # The seed: the measured device's sm_count and L2 with the shipped parameters (features
    # depend only on sm_count and l2_bytes).
    seed = host.Calibration.from_values(meta, host.Calibration.default(a.lib).params, a.lib)
    samples = samples_from_records(records, a.lib, seed)
    if not samples:
        sys.exit("no usable records (profiling build, method imd / packed, status ok)")
    params = fit(samples, seed, a.lib, worst=a.worst, k4_shared_dram=a.k4_shared_dram)
    meta["notes"] = (a.notes or f"fit on {len(samples)} calls ({len(ok)} records) by "
                     f"bench/perf_fit.py")[:127]
    cal = host.Calibration.from_values(meta, params, a.lib)
    with open(a.out, "w") as f:
        f.write(cal.format())
    # The fit's own error on the timed totals (in-sample, the C model).
    errs = [abs(s.again(cal).time_s - s.timed_s) / s.timed_s for s in samples]
    print(f"in-sample total: median |err| {100 * statistics.median(errs):.1f}%, "
          f"p90 {100 * float(np.percentile(errs, 90)):.1f}%, max {100 * max(errs):.1f}% "
          f"over {len(errs)} calls")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
