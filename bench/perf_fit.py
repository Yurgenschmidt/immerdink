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
measured by the sweep) takes K5's launch and byte parameters. The effective L2 of the K4
DRAM term is scanned (a fraction of the recorded L2; --no-l2-scan keeps the nominal size).
v1.15 (DESIGN §6c.1 "The eps path — retired"): every record is predicted the same way whatever
its data (the v1.14 pack emitter has no eps path; the op_eps features are 0, so the op_eps
parameters fit to 0 and are not written). Datasheet peaks: the RTX 4090's, else --peak-dram / --peak-int8 /
--peak-fp64 are required. numpy only.

v1.10 (DESIGN §6c, format 2): the method-occupancy record gives the occupancy table (else the
built-in one); each record is predicted at its measured SM clock (sm_mhz_median of the per-call
probe, else gpu_after) in the REPEAT cache state, and the clock-bound features are scaled to the
reference clock (the median NVML clock of the calls < 1 ms; with --clock-normalize); the chain features
are fitted like the others (--no-chain drops them, --kernel kf4=4 overrides an occupancy for an
A/B). Method-sustained records fit the power model (NNLS on the windows' mean power, a scan of
gamma; a leave-one-case-out check is printed; v1.15 §6c.7: by default the clock-scaled form,
power.byte_f_pj fitted with power.byte_pj held at --byte-pj, gamma 1; --power-form unscaled is
the v1.10 fit), method-pair records the cache term (L_w scan,
Delta by least squares; Delta = 0 when the measured effect is below 3 %). --format1 writes a v1.8
file. Bands are added by bench/perf_validate.py --bands from the hold-out set.

Host API 10 (DESIGN §6c.3): the metadata strings are checked against the calibration's field
sizes before anything is written (check_meta: UTF-8 bytes, NUL included, no line break, no
leading or trailing blank); a --version or --notes that does not fit is an error, the generated
clocks, version and notes are cut to fit at a UTF-8 boundary (an empty --version / --notes
means the generated value, as before host API 10). The written file is re-loaded and must
format back to the same text.

    python3 bench/perf_fit.py --check FILE [FILE ...]

checks existing calibration files (every string line, then the C parser with its reason) and
exits 1 listing every problem.
"""

import argparse
import datetime
import itertools
import json
import math
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from immerdink import host  # noqa: E402

CALL = "call"


# ------------------------------------------------------------------------------------------
# Generation-time checks (host API 10, DESIGN §6c.3)

def fit_text(text, field):
    """`text` cut to the calibration string field's capacity (UTF-8 bytes, NUL excluded) at a
    character boundary: for generated metadata only (a caller's value is checked instead)."""
    cap = host.CALIB_STRING_FIELDS[field] - 1
    b = text.encode()
    if len(b) <= cap:
        return text
    return b[:cap].decode(errors="ignore").rstrip(" \t")


def check_meta(meta):
    """The problems of a calibration's metadata strings: [] if every value can be written and
    parsed back unchanged (fits its field, no line break, no leading / trailing blank)."""
    probs = []
    for k, cap in host.CALIB_STRING_FIELDS.items():
        if k not in meta:
            continue
        v = str(meta[k])
        n = len(v.encode())
        if n > cap - 1:
            probs.append(f"{k}: {n} bytes, the calibration field holds {cap - 1}")
        if "\n" in v or "\r" in v:
            probs.append(f"{k}: contains a line break")
        if v != v.strip(" \t"):
            probs.append(f"{k}: leading or trailing blank (the parser would trim it)")
    return probs


def check_calibration_file(path, lib=None):
    """The problems of an existing calibration file: every string line against its field
    (all of them, by line), then the C parser's first error with its reason."""
    probs = []
    try:
        text = open(path, "rb").read()
    except OSError as e:
        return [f"{path}: {e}"]
    for i, raw in enumerate(text.split(b"\n"), 1):
        line = raw.strip(b" \t\r")
        if not line or line.startswith(b"#") or b"=" not in line:
            continue
        key, val = line.split(b"=", 1)
        key = key.strip(b" \t").decode(errors="replace")
        cap = host.CALIB_STRING_FIELDS.get(key)
        val = val.strip(b" \t\r")
        if cap is not None and len(val) > cap - 1:
            probs.append(f"{path}:{i}: {key}: {len(val)} bytes, the calibration field holds "
                         f"{cap - 1}")
    try:
        host.Calibration.load(path, lib)
    except host.CalibrationError as e:
        probs.append(str(e))
    return probs


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

    def __init__(self, label, pred, measured_phases, timed_s, again, clock=0.0, case=""):
        self.label = label            # "case variant N"
        self.pred = pred              # host.Prediction (features; times under some calibration)
        self.measured = measured_phases  # phase name -> seconds (phase marks)
        self.timed_s = timed_s        # timed median of the whole call (no marks)
        self.again = again            # (calib, state=None) -> host.Prediction (the C model,
                                      # that calibration, at this record's SM clock and cache
                                      # state, or `state`)
        self.clock = clock            # v1.10: the record's SM clock (MHz; 0: unknown)
        self.case = case


# v1.10 (DESIGN §6c.1 "Clock regimes"): the feature slots whose time scales with the SM clock,
# per class (perf.cpp kClockBound; host_perf checks the two agree through predictions).
CLOCK_BOUND = {
    "k0": (2,), "kf": (2, 3, 4, 5), "kl": (2, 3, 4), "pack": (2, 3, 4),  # not wave (slot 5): memory latency
    "k4_128x256": (0, 1, 2, 3, 5), "k4_128x128": (0, 1, 2, 3, 5), "k4_64x128": (0, 1, 2, 3, 5),
    "k4_64x64": (0, 1, 2, 3, 5), "k5": (2, 3), "direct": (2, 3), "beta": (2,), "call": ()}


def record_clock(rec):
    """The SM clock of a record's timed calls (MHz): NVML's right after the timed loop
    (gpu_after; the same source as the sustained windows' clock, so the power model and the
    time model share one clock scale), else 0. The per-call probe (sm_mhz_median, imd_bench
    --clock-probe) is not used: one warp timing %globaltimer reads ~11 % below NVML on the
    4090 with a ~28 % spread between calls (globaltimer granularity); it is kept in the records
    as a relative indicator (NOTES-B2 §6)."""
    g = rec.get("gpu_after", "").split()
    return float(g[0]) if g and g[0].isdigit() else 0.0


# Whether samples are predicted at their record's clock (v1.10 clock normalization; perf_fit
# --no-clock and perf_validate --burst switch it off).
USE_RECORD_CLOCK = True


def _aligned(fn, a, lib):
    """fn under the record's K alignment (DESIGN v1.13 §4b.6): libimmerdink_host's test hook
    set to a (0 = AUTO) around the call, AUTO restored after."""
    def wrapped(*args, **kw):
        if a == 0:
            return fn(*args, **kw)
        host.test_set_k_align(a, lib=lib)
        try:
            return fn(*args, **kw)
        finally:
            host.test_set_k_align(0, lib=lib)
    return wrapped


def record_k_align(rec):
    """The K-alignment hook a record ran under: its k_align_hook (imd_bench, v1.13 stage 2:
    0 = AUTO), else 16 (records from before stage 2 ran every plan at A = 16)."""
    return int(rec.get("k_align_hook", 16))


def samples_from_records(records, lib=None, seed=None):
    """seed: the calibration whose sm_count / l2_bytes the features use (the measured GPU)."""
    out = []
    for r in records:
        if r.get("status") != "ok":
            continue
        ka = record_k_align(r)
        meth = r.get("method")
        spec, ext = r.get("spec"), r.get("ext")
        if not spec or not ext:
            continue
        ext = parse_ext(ext)
        clk = record_clock(r) if USE_RECORD_CLOCK else 0.0
        cache = "COLD" if "l2=flush" in r.get("variant", "") else "REPEAT"
        if meth == "imd" and "phases" in r:
            o = variant_options(r)

            def again(calib, state=None, spec=spec, ext=ext, o=o, clk=clk, cache=cache):
                return host.predict(spec, ext, N=o["N"], g=o["g"], vec_mode=o["vec_mode"],
                                    mode=o["mode"], k4_config=o["k4_config"],
                                    regime="BURST", cache=state or cache, clock_mhz=clk,
                                    calib=calib, lib=lib)
            again = _aligned(again, ka, lib)
            p = again(seed)
            if p.status != "SUCCESS":
                continue
            meas = {k: v["ms"] * 1e-3 for k, v in r["phases"].items() if v.get("ms") is not None}
            out.append(Sample(f"{r['case']} {r['variant']} N={r['N']}", p, meas, r["median_s"],
                              again, clk, r["case"]))
        elif meth == "packed" and "packed_phases" in r:
            which = 0 if r.get("pack") == "A" else 1
            N = int(r["N"])
            pk = host.pack_desc_for(spec, ext, operand=which, N=N, lib=lib)   # no sizes: A-free
            packing = (pk, None) if which == 0 else (None, pk)
            # method packed times imd_pack, the plain and the packed contract interleaved: each
            # finds the L2 holding the other calls' traffic, i.e. COLD for its own inputs
            kw = dict(regime="BURST", clock_mhz=clk, lib=lib)
            fns = {"plain_": lambda c, st=None, spec=spec, ext=ext, N=N, kw=kw:
                       host.predict(spec, ext, N=N, calib=c,
                                    cache=st or "COLD", **kw),
                   "packed_": lambda c, st=None, spec=spec, ext=ext, N=N, packing=packing,
                   kw=kw: host.predict(spec, ext, N=N, packing=packing, calib=c,
                                       cache=st or "COLD", **kw),
                   "pack_": lambda c, st=None, pk=pk, kw=kw: host.predict_pack(
                       pk, calib=c, cache=st or "COLD", **kw)}
            fns = {key: _aligned(f, ka, lib) for key, f in fns.items()}
            lab = f"{r['case']} pack {r.get('pack')} N={N}"
            for pre, timed in (("plain_", r["contract_ms"] * 1e-3),
                               ("packed_", r["contract_packed_ms"] * 1e-3),
                               ("pack_", r["pack_ms"] * 1e-3)):
                pred = fns[pre](seed)
                ph = r.get(pre + "phases")
                if ph is None or pred.status != "SUCCESS":
                    continue
                meas = {k: v["ms"] * 1e-3 for k, v in ph.items() if v.get("ms") is not None}
                out.append(Sample(f"{lab} {pre.rstrip('_')}", pred, meas, timed, fns[pre],
                                  clk, r["case"]))
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


def phase_rows(samples, fref=0.0):
    """class -> list of (x, measured seconds, label). v1.10: with a reference clock fref > 0,
    the clock-bound features of a record measured at clock f are scaled by fref / f, so the
    fitted parameters are the reference clock's (DESIGN §6c.1)."""
    rows = {}
    for s in samples:
        for p in s.pred.phases:
            if p.cls == CALL or p.name not in s.measured:
                continue
            t = s.measured[p.name]
            if t <= 0:
                continue
            x = np.array(p.x)
            if fref > 0 and s.clock > 0:
                for j in CLOCK_BOUND[p.cls]:
                    x[j] *= fref / s.clock
            rows.setdefault(p.cls, []).append((x, t, f"{s.label} {p.name}"))
    return rows


K4_BYTE = 4  # K4 classes: t = max(sum of the round-model terms j != 4, the DRAM term j = 4)


def is_k4(cls):
    return cls.startswith("k4_")


def predicted_ns(params, names, cls, x):
    """The model of one phase (DESIGN §6c.1; the C is perf.cpp perf_phase_seconds): the sum of
    the terms, for K4 the max of the round model and the DRAM term."""
    terms = [params.get(f"{cls}.{n}", 0.0) * x[j] if n else 0.0 for j, n in enumerate(names[cls])]
    if is_k4(cls):
        return max(sum(terms) - terms[K4_BYTE], terms[K4_BYTE])
    return sum(terms)


def k4_round_monotone(cls_names, used):
    """The K4 MAC parameters reparametrized so that a round of q concurrent CTAs never costs
    less than a round of q - 1 (2026-10-06, NOTES-B2 §8): with c_q = q mac_q (the cost per
    tile-MAC of a q-CTA round), fit d_p >= 0 with c_q = d_1 + ... + d_q. Returns T with
    theta = T d over the class's used columns (identity outside the mac columns): fitting X T
    keeps NNLS and gives c_1 <= c_2 <= c_3. Without it NNLS may set mac2 = 0 next to mac1 > 0
    (64x64, stage-2 refit), and the predicted time then drops when a shape grows from one to
    two CTAs per SM (host_perf monotonicity)."""
    T = np.eye(len(used))
    macs = [(int(cls_names[j][3:]), i) for i, j in enumerate(used)
            if cls_names[j] and re.fullmatch(r"mac\d", cls_names[j])]
    for q, i in macs:
        T[i, :] = 0.0
        for p, ip in macs:
            if p <= q:
                T[i, ip] = 1.0 / q
    return T


def fit_k4(X, t, byte_col, rates=None, T=None):
    """max(round model, DRAM term) (a roofline): for each candidate DRAM rate (bandwidths from
    100 GB/s to 2 TB/s, or none) the round model is fitted (NNLS, relative) on the phases the
    DRAM term does not explain (its time below 90 % of the measured), and the rate with the
    smallest total relative error of the max is kept. T (k4_round_monotone): fit in d with
    theta = T d."""
    if T is not None:
        th, err = fit_k4(X @ T, t, byte_col, rates)
        return (None if th is None else T @ th), err
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


def fit(samples, seed, lib=None, verbose=True, worst=0, k4_shared_dram=False, fref=0.0,
        drop=()):
    """drop: parameter keys forced to 0 (e.g. the v1.10 chain features, for an A/B)."""
    names = host.param_names(lib)
    rows = phase_rows(samples, fref)
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
                T = k4_round_monotone(names[cls], used)
                th, e = fit_k4(X, t, used.index(K4_BYTE), [rate], T)
                if th is None:
                    tot = float("inf")
                    break
                tot += e
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
        for j in [used.index(j) for j in used if f"{cls}.{names[cls][j]}" in drop]:
            X[:, j] = 0.0  # a dropped feature: its parameter fits to 0
        if is_k4(cls):
            theta = fit_k4(X, t, used.index(K4_BYTE), None if shared is None else [shared],
                           k4_round_monotone(names[cls], used))[0]
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
    if seed.file_format == 1:  # a format-1 calibration has none of the v1.10 chain parameters
        for k in NEW_CHAINS:
            params.pop(k, None)
    trial = host.Calibration.from_values(seed.meta, params, lib, seed.extra or None)
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


def k4_error(samples, lib, shared, fref=0.0):
    """Sum over the K4 classes of the fitted relative squared error (the L2 scan's criterion)."""
    names = host.param_names(lib)
    rows = phase_rows(samples, fref)
    tot = 0.0
    for cls in host.PERF_CLASSES[4:8]:
        data = rows.get(cls, [])
        if not data:
            continue
        used = [j for j, n in enumerate(names[cls]) if n]
        X = np.array([d[0][used] for d in data])
        t = np.array([d[1] for d in data])
        tot += fit_k4(X, t, used.index(K4_BYTE), None, k4_round_monotone(names[cls], used))[1]
    return tot


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


NEW_CHAINS = tuple(sorted(host.FORMAT2_PARAMS))  # the v1.10 chain parameters (format 2)


def occupancy_from_records(records, lib=None):
    """kernel name -> (resident, regs, smem, threads) of the last method-occupancy record, or
    None (then the built-in table of a format-1 calibration is used)."""
    occ = [r for r in records if r.get("method") == "occupancy" and r.get("status") == "ok"]
    if not occ:
        return None
    return {k: tuple(int(x) for x in v) for k, v in occ[-1]["kernels"].items()}


def reference_clock(records):
    """The reference SM clock (MHz) of the time model: the median clock of the uncapped timed
    records (calls < 1 ms; the large ones run at the power cap even in a burst)."""
    clk = [record_clock(r) for r in records if r.get("status") == "ok" and
           r.get("method") in ("imd", "packed") and 0 < r.get("median_s", 1) < 1e-3]
    clk = [c for c in clk if c > 0]
    return float(statistics.median(clk)) if clk else 0.0


def seed_extra(lib, kernels=None, fref=0.0, base=None):
    """The format-2 keys of a seed calibration: the occupancy table (measured, else the
    built-in one), the reference clock without a power model, no cache term, bands 1."""
    ref = host.Calibration.default(lib)
    ex = {f"kernel.{n}": v for n, v in ref.kernels.items()}
    if kernels:
        for n, v in kernels.items():
            ex[f"kernel.{n}"] = v
    ex.update({"power.clock_eff_mhz": fref, "power.clock_min_mhz": fref, "power.cap_w": 0.0,
               "power.idle_w": 0.0, "power.tc_pj": 0.0, "power.alu_pj": 0.0,
               "power.byte_pj": 0.0, "power.gamma": 2.0,
               "warm.l2_bytes": 0.0, "warm.l2byte": 0.0, "rank.tie_ratio": 1.0})
    ex.update({k: (1.0, 1.0) for k in ref.bands})
    if base:
        ex.update(base)
    return ex


def base_params(lib):
    p = dict(host.Calibration.default(lib).params)
    for k in NEW_CHAINS:
        p.setdefault(k, 0.0)
    return p


# ---- v1.10: the power model (DESIGN §6c.1 "Clock regimes") ----------------------------------

def sustained_rows(records, cal, lib=None):
    """Per method-sustained record: (case, N, measured s per call, mean power W, mean SM clock,
    tensor-core ops, ALU ops, DRAM bytes, power limit W, label). The op and byte counts come
    from the C model's phases (perf.cpp call_power uses the same selection)."""
    out = []
    for r in records:
        if r.get("method") != "sustained" or r.get("status") != "ok":
            continue
        ext = parse_ext(r["ext"])
        v = r.get("variant", "")
        p = host.predict(r["spec"], ext, N=int(r["N"]), g=int(r.get("g", 0) or 0),
                         vec_mode="RESIDENT" if "vec=resident" in v else "GENERATE",
                         regime="BURST", cache="REPEAT", calib=cal, lib=lib)
        if p.status != "SUCCESS":
            continue
        tc = sum(q.ops for q in p.phases if q.cls.startswith("k4_"))
        alu = sum(q.x[2] for q in p.phases if q.cls in ("kf", "kl", "pack", "k5", "direct"))
        D = sum(q.bytes for q in p.phases)
        out.append(dict(case=r["case"], N=int(r["N"]), t=float(r["median_s"]),
                        P=float(r["power_w_mean"]), f=float(r["sm_mhz_mean"]), tc=tc, alu=alu,
                        D=D, cap=float(r.get("power_limit_w") or 450.0),
                        label=f"{r['case']} {v} N={r['N']}", spec=r["spec"], ext=ext,
                        g=int(r.get("g", 0) or 0), vec="RESIDENT" if "vec=resident" in v else "GENERATE"))
    return out


POWER_FORMS = ("scaled", "unscaled")


def fit_power(rows, fref, gammas=None, idle=60.0, form="scaled", byte_pj=0.0):
    """The power model over the sustained windows at their measured clock f, time t and power
    P (NNLS of P - idle, relative in the dynamic power; the gamma of the smallest power error).
    The awake idle is held fixed (`idle`): with most windows pinned at the cap, a free
    intercept absorbs the power (it fitted 366 W on the post-BIOS data).
      form "scaled" (v1.15, DESIGN §6c.7, the default): P = idle + ((e_tc tc + e_alu alu +
        e_byte^f D) (f / fref)^gamma + e_byte D) 1e-12 / t with e_byte held fixed at `byte_pj`
        (the two byte columns are collinear on the windows: 0, or a stated prior); fitted
        e_tc, e_alu, e_byte^f; gammas default (1,).
      form "unscaled" (v1.10): P = idle + ((e_tc tc + e_alu alu) (f / fref)^gamma + e_byte D)
        1e-12 / t; fitted e_tc, e_alu, e_byte; gammas default (1, 2, 3).
    Returns the power.* keys (cap = the records' enforced limit, clock_min = 0.8 x the lowest
    measured; power.byte_f_pj only for the scaled form)."""
    if form not in POWER_FORMS:
        raise ValueError(f"power form {form!r}: one of {POWER_FORMS}")
    scaled = form == "scaled"
    if gammas is None:
        gammas = (1.0,) if scaled else (1.0, 2.0, 3.0)
    best = None
    P = np.array([r["P"] for r in rows])
    fixed = np.array([byte_pj * r["D"] * 1e-12 / r["t"] for r in rows]) if scaled else 0.0
    for gm in gammas:
        X = np.array([[r["tc"] * (r["f"] / fref) ** gm * 1e-12 / r["t"],
                       r["alu"] * (r["f"] / fref) ** gm * 1e-12 / r["t"],
                       r["D"] * ((r["f"] / fref) ** gm if scaled else 1.0) * 1e-12 / r["t"]]
                      for r in rows])
        th = nnls_relative(X, np.maximum(P - idle - fixed, 1e-3))  # relative in the dynamic power
        err = float(np.sum(((idle + fixed + X @ th - P) / P) ** 2))
        if best is None or err < best[0]:
            best = (err, gm, th)
    _, gm, th = best
    out = {"power.clock_eff_mhz": fref, "power.clock_min_mhz": 0.8 * min(r["f"] for r in rows),
           "power.cap_w": float(np.median([r["cap"] for r in rows])), "power.idle_w": float(idle),
           "power.tc_pj": float(th[0]), "power.alu_pj": float(th[1]),
           "power.byte_pj": float(byte_pj) if scaled else float(th[2]), "power.gamma": gm}
    if scaled:
        out["power.byte_f_pj"] = float(th[2])
    return out


def power_validation(rows, meta, params, extra, fref, lib=None, verbose=True, idle=60.0,
                     **power_kw):
    """Leave-one-case-out: the power model fitted without the window's case (fit_power with
    power_kw: form, byte_pj, gammas), then the window's SUSTAINED prediction (time, clock)
    against the measured. Returns [(label, t meas, t pred, f meas, f pred, P meas, P pred)]."""
    out = []
    for case in sorted({r["case"] for r in rows}):
        train = [r for r in rows if r["case"] != case]
        if len(train) < 4:
            continue
        pw = fit_power(train, fref, idle=idle, **power_kw)
        cal = host.Calibration.from_values(meta, params, lib, dict(extra, **pw))
        for r in rows:
            if r["case"] != case:
                continue
            p = host.predict(r["spec"], r["ext"], N=r["N"], g=r["g"], vec_mode=r["vec"],
                             regime="SUSTAINED", cache="REPEAT", calib=cal, lib=lib)
            out.append((r["label"], r["t"], p.time_s, r["f"], p.clock_mhz, r["P"], p.power_w))
    if verbose and out:
        print(f"{'sustained window (leave-one-case-out)':<44} {'TFLOPS-eq':>9} {'t meas':>10} "
              f"{'t pred':>10} {'err':>7} {'f meas':>7} {'f pred':>7} {'P meas':>7} {'P pred':>7}")
        for lab, tm, tp, fm, fp, pm, pp in out:
            print(f"{lab[:44]:<44} {'':>9} {1e3 * tm:>8.3f}ms {1e3 * tp:>8.3f}ms "
                  f"{100 * (tp - tm) / tm:>+6.1f}% {fm:>7.0f} {fp:>7.0f} {pm:>7.1f} {pp:>7.1f}")
    return out


# ---- v1.10: the cache term (DESIGN §6c.1 "Cache state") ------------------------------------

def pair_cases(records):
    """(case, spec, ext, N) -> {"cold"|"warm"|"repeat": measured median s} of method pair."""
    out = {}
    for r in records:
        if r.get("method") != "pair" or r.get("status") != "ok":
            continue
        key = (r["case"], r["spec"], r["ext"], int(r["N"]))
        out.setdefault(key, {})[r["cache"]] = float(r["median_s"])
    return {k: v for k, v in out.items() if len(v) == 3}


def fit_warm(cases, meta, params, extra, l2, lib=None, min_effect=0.03, verbose=True):
    """L_w (a scan over fractions of the nominal L2) and Delta >= 0 (least squares, relative):
    measured t_state - t_repeat against the model's (res_repeat - res_state) bytes, which the C
    model gives as its COLD minus REPEAT time at a unit Delta. Delta = 0 when
    no measured |COLD - REPEAT| or |WARM - REPEAT| reaches min_effect of the call."""
    effects = []
    for (case, spec, ext, N), t in sorted(cases.items()):
        effects.append((case, N, t["cold"] / t["repeat"] - 1, t["warm"] / t["repeat"] - 1, t))
    if verbose:
        print(f"{'cache pair case':<20} {'N':>3} {'repeat':>10} {'cold vs repeat':>15} "
              f"{'warm vs repeat':>15}")
        for case, N, ec, ew, t in effects:
            print(f"{case:<20} {N:>3} {1e3 * t['repeat']:>8.4f}ms {100 * ec:>+14.1f}% "
                  f"{100 * ew:>+14.1f}%")
    big = max((max(abs(ec), abs(ew)) for _, _, ec, ew, _ in effects), default=0.0)
    if big < min_effect:
        if verbose:
            print(f"cache effect below {100 * min_effect:.0f} % everywhere: warm.l2byte = 0")
        return {"warm.l2_bytes": 0.0, "warm.l2byte": 0.0}, effects
    best = None
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        unit = 1e-6  # ns per byte: small, so no REPEAT phase is clamped at 0
        ex = dict(extra, **{"warm.l2_bytes": frac * l2, "warm.l2byte": unit})
        cal = host.Calibration.from_values(meta, params, lib, ex)
        X, y = [], []
        for (case, spec, ext, N), t in cases.items():
            e = parse_ext(ext)
            pr = host.predict(spec, e, N=N, cache="REPEAT", regime="BURST", calib=cal, lib=lib)
            for st in ("COLD",):  # WARM is modeled as REPEAT (perf.cpp)
                d = host.predict(spec, e, N=N, cache=st, regime="BURST", calib=cal,
                                 lib=lib).time_s - pr.time_s
                X.append(d / unit / t["repeat"])
                y.append((t[st.lower()] - t["repeat"]) / t["repeat"])
        X, y = np.array(X), np.array(y)
        delta = max(0.0, float(X @ y) / float(X @ X)) if float(X @ X) > 0 else 0.0
        err = float(np.sum((delta * X - y) ** 2))
        if best is None or err < best[0]:
            best = (err, frac, delta)
    _, frac, delta = best
    if verbose:
        print(f"cache term: L_w = {frac:.1f} x L2, {delta:.3g} ns per byte "
              f"({1 / delta if delta > 0 else float('inf'):.3g} B/ns saved)")
    return {"warm.l2_bytes": frac * l2, "warm.l2byte": delta}, effects


def main(argv=None):
    global USE_RECORD_CLOCK
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="*", help="JSON lines of a calibration sweep (with --check: "
                    "calibration files to check)")
    ap.add_argument("-o", "--out", help="calibration file to write")
    ap.add_argument("--check", action="store_true",
                    help="check the given calibration files (string fields, the C parser) and "
                         "exit: 1 if any has a problem")
    ap.add_argument("--lib", default=None, help="libimmerdink_host.so")
    ap.add_argument("--version", default=None,
                    help="version string (default or \"\": the commit, cut to fit)")
    ap.add_argument("--notes", default=None,
                    help="notes string (default or \"\": the fit's sample counts, cut to fit)")
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
    ap.add_argument("--no-l2-scan", action="store_true",
                    help="use the nominal L2 (no scan of the effective L2 fraction)")
    # v1.10
    ap.add_argument("--clock-normalize", action="store_true",
                    help="scale each record's clock-bound features by f_ref / its NVML clock "
                         "(off by default: on the post-BIOS 4090 data it did not improve the "
                         "hold-out, NOTES-B2 §6)")
    ap.add_argument("--no-clock", action="store_true",
                    help="no reference clock at all (no power model, SUSTAINED = BURST)")
    ap.add_argument("--idle-w", type=float, default=60.0,
                    help="awake-idle board power held fixed in the power fit (W; the 4090: about "
                         "60 W, evaluation workspace findings/energy-20261004.md)")
    ap.add_argument("--clock-ref", type=float, default=None, help="reference SM clock (MHz)")
    # v1.15 (DESIGN §6c.7)
    ap.add_argument("--power-form", choices=POWER_FORMS, default="scaled",
                    help="power model: 'scaled' (v1.15 default: clock-scaled byte energy "
                         "power.byte_f_pj fitted, power.byte_pj held at --byte-pj) or "
                         "'unscaled' (the v1.10 fit of power.byte_pj, no byte_f key)")
    ap.add_argument("--byte-pj", type=float, default=0.0,
                    help="scaled form: the fixed unscaled byte energy (pJ per DRAM byte; 0, or "
                         "a stated prior), written as power.byte_pj")
    ap.add_argument("--power-gammas", default=None,
                    help="comma-separated gammas scanned by the power residual (default: 1 for "
                         "the scaled form, 1,2,3 for the unscaled form)")
    ap.add_argument("--no-chain", action="store_true",
                    help="drop the v1.10 chain features of K0, KL, PACK, K5, DIRECT, BETA")
    ap.add_argument("--kernel", action="append", default=[], metavar="NAME=R",
                    help="override a kernel's CTAs per SM (e.g. kf4=4, the v1.8 built-in)")
    ap.add_argument("--format1", action="store_true",
                    help="write a format-1 file (v1.8 features only; implies --no-clock, "
                         "--no-chain)")
    a = ap.parse_args(argv)
    if a.check:
        probs = [p for f in a.records for p in check_calibration_file(f, a.lib)]
        for p in probs:
            print(p, file=sys.stderr)
        if not a.records:
            ap.error("--check needs calibration files")
        print(f"{len(a.records)} calibration file(s): "
              f"{'ok' if not probs else str(len(probs)) + ' problem(s)'}")
        sys.exit(1 if probs else 0)
    if not a.records or not a.out:
        ap.error("records and -o/--out are required (or --check FILE...)")
    if a.format1:
        a.no_clock = a.no_chain = True
    if a.byte_pj < 0 or not math.isfinite(a.byte_pj):
        ap.error("--byte-pj must be a finite number >= 0")
    if a.byte_pj != 0 and a.power_form != "scaled":
        ap.error("--byte-pj is the scaled form's fixed term")
    power_kw = dict(form=a.power_form, byte_pj=a.byte_pj,
                    gammas=tuple(float(x) for x in a.power_gammas.split(","))
                    if a.power_gammas else None)
    USE_RECORD_CLOCK = a.clock_normalize and not a.no_clock
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
            "version": a.version or fit_text(f"immerdink {commit}, host API {host.API_VERSION}",
                                             "version"),
            "clocks": fit_text(clock_summary(records), "clocks"), **peaks, "l2_bytes": l2}
    if probs := check_meta(meta):  # before the fit: a caller's --version or the records' gpu
        sys.exit("calibration metadata cannot be written (DESIGN §6c.3):\n  " + "\n  ".join(probs))
    fref = 0.0 if a.no_clock else (a.clock_ref or reference_clock(records))
    fnorm = fref if USE_RECORD_CLOCK else 0.0  # the clock normalization of the fit
    kernels = occupancy_from_records(records, a.lib) or {}
    for kv in a.kernel:
        n, r = kv.split("=")
        old = kernels.get(n) or host.Calibration.default(a.lib).kernels[n]
        kernels[n] = (int(r),) + tuple(old[1:])
    extra = None if a.format1 else seed_extra(a.lib, kernels, fref)
    if extra is not None:
        print(f"reference clock {fref:.0f} MHz; occupancy: " +
              ", ".join(f"{n} {extra['kernel.' + n][0]}" for n in
                        ("k0", "kf16", "kf4", "kl8", "kl4", "pack", "k5_n14_e4", "k5b_n14")))
    params0 = base_params(a.lib) if extra is not None else host.Calibration.default(a.lib).params
    drop = NEW_CHAINS if a.no_chain else ()
    # The seed: the measured device's sm_count and L2 with the shipped parameters (features
    # depend only on sm_count, l2_bytes and the occupancy table).
    seed = host.Calibration.from_values(meta, params0, a.lib, extra)
    samples = samples_from_records(records, a.lib, seed)
    if not samples:
        sys.exit("no usable records (profiling build, method imd / packed, status ok)")
    if not a.no_l2_scan:
        # The effective L2 of K4's DRAM term: the measured device's L2 is shared with W, the
        # other operand and the packs' traffic, so a group's working set streams from DRAM well
        # before it reaches the nominal size. Scan a fraction of the nominal L2 and keep the one
        # with the smallest K4 error (the K4 features are the only ones that depend on it).
        best = None
        for frac in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            m2 = dict(meta, l2_bytes=max(frac * l2, 1.0))  # l2_bytes > 0
            sd = host.Calibration.from_values(m2, params0, a.lib, extra)
            sm = samples_from_records(records, a.lib, sd)
            err = k4_error(sm, a.lib, a.k4_shared_dram, fnorm)
            print(f"  effective L2 {frac:.2f} x {l2 / 2**20:.0f} MiB: K4 relative SSE {err:.3f}")
            if best is None or err < best[0] - 1e-9:
                best = (err, frac, sd, sm)
        _, frac, seed, samples = best
        meta["l2_bytes"] = max(frac * l2, 1.0)
        print(f"effective L2: {frac:.2f} of the nominal {l2 / 2**20:.0f} MiB")
    # The cache term first (DESIGN §6c.1 "Cache state"): Delta and L_w from the COLD / REPEAT
    # pairs; then each REPEAT phase measurement gets its resident-input saving added back, so
    # the parameters are COLD costs (a REPEAT prediction subtracts the saving again).
    cases = pair_cases(records) if extra is not None else {}
    if cases:
        wm, _ = fit_warm(cases, meta, params0, extra, l2, a.lib)
        extra.update(wm)
        if wm["warm.l2byte"] > 0:
            calw = host.Calibration.from_values(meta, params0, a.lib, extra)
            added = []
            for s in samples:
                if "l2=flush" in s.label or " pack " in s.label:
                    continue  # measured COLD already (flushed, or method packed's interleaving)
                cold, rep = s.again(calw, "COLD"), s.again(calw, "REPEAT")
                for qc, qr in zip(cold.phases, rep.phases):
                    if qc.name in s.measured and qc.time_s != qr.time_s:
                        s.measured[qc.name] += qc.time_s - qr.time_s
                added.append((cold.time_s - rep.time_s) / s.timed_s)
            print(f"cache term: REPEAT phases raised to COLD by {100 * np.median(added):.1f} % "
                  f"(median), {100 * max(added):.1f} % (max) of the call")
        seed = host.Calibration.from_values(meta, seed.params, a.lib, extra)
    params = fit(samples, seed, a.lib, worst=a.worst, k4_shared_dram=a.k4_shared_dram,
                 fref=fnorm, drop=drop)
    if extra is not None:
        rows = sustained_rows(records, host.Calibration.from_values(meta, params, a.lib, extra),
                              a.lib)
        if len(rows) >= 4 and fref > 0:
            # The time model's effective clock (DESIGN §6c.1): the burst calibration's large calls
            # already ran below the boost clock (the limiter acts within a timed loop), so the
            # clock that SUSTAINED scales from is scanned (leave-one-case-out median error of
            # the sustained windows), from the boost clock down.
            best = None
            for fr in [fref - 25 * i for i in range(0, 17)]:
                out = power_validation(rows, meta, params, extra, fr, a.lib, verbose=False,
                                       idle=a.idle_w, **power_kw)
                err = float(np.median([abs(tp - tm) / tm for _, tm, tp, *_ in out]))
                print(f"  time-model clock {fr:.0f} MHz: sustained LOO median |err| {100 * err:.2f} %")
                if best is None or err < best[0] - 1e-12:
                    best = (err, fr)
            fr = best[1] if a.clock_ref is None else fref
            pw = fit_power(rows, fr, idle=a.idle_w, **power_kw)
            print(f"power model ({a.power_form} form" +
                  (f", power.byte_pj held at {a.byte_pj:g} pJ/B" if a.power_form == "scaled" else "")
                  + "): " + ", ".join(f"{k} {v:.4g}" for k, v in pw.items()))
            power_validation(rows, meta, params, extra, fr, a.lib, idle=a.idle_w, **power_kw)
            extra.update(pw)
    meta["notes"] = a.notes or fit_text(  # --notes "" (like no --notes): the generated notes
        f"fit on {len(samples)} calls ({len(ok)} records) by bench/perf_fit.py", "notes")
    if probs := check_meta(meta):  # a caller's --notes is never cut silently
        sys.exit("calibration metadata cannot be written (DESIGN §6c.3):\n  " + "\n  ".join(probs))
    cal = host.Calibration.from_values(meta, params, a.lib, extra)
    text = cal.format()
    with open(a.out, "w") as f:
        f.write(text)
    # What was written must load and format back to the same text (DESIGN §6c.3).
    try:
        back = host.Calibration.load(a.out, a.lib).format()
    except host.CalibrationError as e:
        sys.exit(f"the written calibration does not load: {e}")
    if back != text:
        sys.exit(f"{a.out}: the written calibration does not re-load to the same text")
    # The fit's own error on the timed totals (in-sample, the C model).
    errs = [abs(s.again(cal).time_s - s.timed_s) / s.timed_s for s in samples]
    print(f"in-sample total: median |err| {100 * statistics.median(errs):.1f}%, "
          f"p90 {100 * float(np.percentile(errs, 90)):.1f}%, max {100 * max(errs):.1f}% "
          f"over {len(errs)} calls")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
