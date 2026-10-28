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
its data (the v1.14 pack emitter has no eps path; v1.17: the op_eps keys are retired). Datasheet peaks: the RTX 4090's, else --peak-dram / --peak-int8 /
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

v1.17 (DESIGN §6c.1, host API 12): KF's per-element chain (kf.elem) and the PACK bytes that are
not L2-hit re-reads (pack.dram) are fitted like the other features; L_hit (warm.pack_l2_bytes)
is --pack-l2-frac (0.8) x the records' nominal L2, set in the seed so the features of the fit
are the prediction's (the PACK error over 0.6-1.0 x L2 is printed as a diagnostic); the
effective clock is the vertex of a parabola through the LOO scan near its argmin (--fref-rule),
and the nested leave-one-out error of that choice is printed; the phases of --l2 flush records
(REPEAT state: imd_bench flushes before the timed calls only) are raised to COLD.

v1.18 (DESIGN §6c.8, host API 13): the cache fit adds the per-call REPEAT saving (warm.call_ns;
since perf-minor-noc not fitted by default, --cache-terms delta,c,omega fits it)
and the clean-L2 credit (warm.clean_byte): method pair's COLD and the totals of --l2 flush
records are the CLEAN state (an L2 flushed by reads), L_w is scanned over 0.30-1.20 x L2; the
PACK set counts one group's W planes (warm.pack_l2_planes = 1, --pack-l2-frac 1.1); the power
model's default form "busy" adds the busy-GPU power (power.act_w, launch gap --act-gap-us) and
the DRAM energy (power.dram_pj), solves the clock up to the boost clock (power.clock_boost_mhz:
the median clock of the windows that never touched the power cap) with gamma 2.5 as a prior;
the f_ref scan goes down to 2255 MHz on the 4090. The busy share and the DRAM bytes of a window
come from the C model (a probe calibration), not from a formula here.

v1.19 (DESIGN §6c.9, NOTES-C1 §5.9 MC-3a, host API 13): the optional key warm.kl_drop, the share
of the REPEAT / WARM saving that KL reads do not get, is fitted after the cache term, against the
KL phase marks (fit_kl_drop: a scan of d in 0 ... 1 that refits the KL class on the phases raised
to COLD with the share (1 - d) of the model's saving and scores the REPEAT-state prediction of
every KL phase against its mark). The default is --kl-drop 0 (absent, the v1.18 rule), so a routine
recalibration never ships a nonzero warm.kl_drop; --kl-drop scan runs the scan, --kl-drop D fixes it.

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
    # v1.17: kf.elem (slot 4) and pack.dram (slot 3) are not clock-bound (tile-load latency,
    # DRAM bytes); v1.20: kl slot 4 is kl.chain (clock-bound, the busiest SM's steps)
    "k0": (2,), "kf": (2, 3, 5), "kl": (2, 3, 4), "pack": (2, 4),  # not wave (slot 5): memory latency
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
        # v1.18 (DESIGN §6c.8): a flushed record's total is the CLEAN state (the flush reads its
        # buffer, so the L2 holds no dirty lines; its phases are REPEAT state, v1.17)
        cache = "CLEAN" if "l2=flush" in r.get("variant", "") else "REPEAT"
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


KL_CHAIN = 4  # v1.20 (DESIGN §6c.10): kl.chain, the slot of the retired kl.op_eps


def fit_slots(names, cls):
    """The slots a plain NNLS fit of the class uses: every named slot but kl.chain (v1.20), which
    has its own fit step (fit_kl_chain) and stays absent in a routine recalibration."""
    return [j for j, n in enumerate(names[cls]) if n and f"{cls}.{n}" != "kl.chain"]


def nnls_weighted(X, y, w):
    """argmin_{theta >= 0} sum((w (X theta - y))^2), exactly, by enumerating active sets (columns
    <= 5); nnls_relative is w = 1 / y with y the time. Columns that are all zero get 0."""
    A0 = X * w[:, None]
    yw = y * w
    ncol = X.shape[1]
    used = [j for j in range(ncol) if np.any(X[:, j] != 0)]
    scale = np.array([np.abs(A0[:, j]).max() if j in used else 1.0 for j in range(ncol)])
    best, best_res = np.zeros(ncol), float(np.sum(yw ** 2))
    for r in range(1, len(used) + 1):
        for S in itertools.combinations(used, r):
            A = A0[:, S] / scale[list(S)]
            sol, *_ = np.linalg.lstsq(A, yw, rcond=None)
            if np.any(sol < 0):
                continue
            res = float(np.sum((A @ sol - yw) ** 2))
            if res < best_res - 1e-15:
                theta = np.zeros(ncol)
                theta[list(S)] = sol / scale[list(S)]
                best, best_res = theta, res
    return best


KL_BASE = (0, 2, 3, 5)  # kl slots outside the max: launch, op, l2byte, wave


def kl_chain_scan(X, t, byte_rates, chains):
    """v1.20 (DESIGN §6c.10): for each (byte rate B, chain C) the max arm r = max(B bytes, C s) is
    taken off the time and (launch, op, l2byte, wave) are fitted (exact NNLS, relative weights of
    the time itself) on the remainder; returns the full 6-slot theta (seconds per unit) and the
    relative SSE of the smallest, over the grid."""
    w = 1.0 / t
    best = (float("inf"), None)
    for B in byte_rates:
        for C in chains:
            r = np.maximum(B * X[:, 1], C * X[:, 4])
            th = np.zeros(X.shape[1])
            th[list(KL_BASE)] = nnls_weighted(X[:, KL_BASE], t - r, w)
            th[1], th[4] = B, C
            sse = float(np.sum(((np.maximum(th[1] * X[:, 1], th[4] * X[:, 4]) + X[:, list(KL_BASE)]
                                 @ th[list(KL_BASE)] - t) / t) ** 2))
            if sse < best[0]:
                best = (sse, th)
    return best[1], best[0]


def kl_chain_grid(b0, n_b=31, n_c=61):
    """Byte rates around the plain fit's (0.5 x ... 2 x) and chains 10 ps ... 2 us per unit."""
    return (b0 * np.geomspace(0.5, 2.0, n_b), np.concatenate([[0.0], np.geomspace(1e-11, 2e-6, n_c)]))


def fit_kl_chain(X, t, cases, verbose=True):
    """v1.20 (DESIGN §6c.10, NOTES-C1 §5.13): the KL class with the chain arm. X: the KL phases'
    features (6 slots; x[4] = s), t: the marks (seconds, raised to COLD), cases: the shape of each
    phase (the folds). The plain NNLS fit without the chain and the scan with it are each scored
    by leave-one-shape-out relative SSE; the chain is kept only if its held-out error is smaller
    (P4), else kl.chain = 0 (absent). Returns (theta in seconds per unit, 6 slots; used: bool)."""
    used = [0, 1, 2, 3, 5]  # launch, byte, op, l2byte, wave (the plain fit of the class)
    plain = np.zeros(X.shape[1])
    plain[used] = nnls_relative(X[:, used], t)
    b0 = max(plain[1], 1e-13)
    full, sse_full = kl_chain_scan(X, t, *kl_chain_grid(b0))
    cs = sorted(set(cases))
    cv_plain = cv_chain = 0.0
    bs, chs = kl_chain_grid(b0, 11, 31)
    for c in cs:
        tr = np.array([k != c for k in cases])
        te = ~tr
        if tr.sum() < 6:
            continue
        thp = np.zeros(X.shape[1])
        thp[used] = nnls_relative(X[tr][:, used], t[tr])
        cv_plain += float(np.sum(((X[te] @ thp - t[te]) / t[te]) ** 2))
        thc, _ = kl_chain_scan(X[tr], t[tr], bs, chs)
        pr = (np.maximum(thc[1] * X[te, 1], thc[4] * X[te, 4]) +
              X[te][:, list(KL_BASE)] @ thc[list(KL_BASE)])
        cv_chain += float(np.sum(((pr - t[te]) / t[te]) ** 2))
    keep = cv_chain < cv_plain and full[4] > 0
    if verbose:
        sse_plain = float(np.sum(((X @ plain - t) / t) ** 2))
        print(f"KL chain (kl.chain): {len(t)} KL phases, {len(cs)} shapes; relative SSE plain "
              f"{sse_plain:.3f}, with chain {sse_full:.3f}; leave-one-shape-out plain {cv_plain:.3f}, "
              f"chain {cv_chain:.3f} -> {'kept' if keep else 'absent'}")
        if keep:
            print(f"  fitted: byte {full[1] * 1e9:.4g} ns/B, chain {full[4] * 1e9:.4g} ns/unit, "
                  f"launch {full[0] * 1e9:.4g}, op {full[2] * 1e9:.4g}, l2byte {full[3] * 1e9:.4g}, "
                  f"wave {full[5] * 1e9:.4g} ns")
            bb = full[1]
            prof = [(C, kl_chain_scan(X, t, [bb], [C])[1]) for C in np.geomspace(1e-9, 2e-6, 13)]
            print("  objective profile in C (ns/unit: relative SSE): " +
                  ", ".join(f"{C * 1e9:.3g}: {v:.3f}" for C, v in prof))
    return (full if keep else plain), keep


def is_k4(cls):
    return cls.startswith("k4_")


def predicted_ns(params, names, cls, x):
    """The model of one phase (DESIGN §6c.1; the C is perf.cpp perf_phase_seconds): the sum of
    the terms, for K4 the max of the round model and the DRAM term."""
    terms = [params.get(f"{cls}.{n}", 0.0) * x[j] if n else 0.0 for j, n in enumerate(names[cls])]
    if is_k4(cls):
        return max(sum(terms) - terms[K4_BYTE], terms[K4_BYTE])
    if cls == "kl" and params.get("kl.chain", 0.0) > 0:  # v1.20 (§6c.10): byte arm vs chain arm
        return sum(terms) - terms[1] - terms[KL_CHAIN] + max(terms[1], terms[KL_CHAIN])
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
        drop=(), kl_chain=False):
    """drop: parameter keys forced to 0 (e.g. the v1.10 chain features, for an A/B). kl_chain
    (v1.20, opt-in): the KL class with the chain arm (fit_kl_chain; the samples' features must
    come from a seed with kl.chain > 0, see main)."""
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
                used = fit_slots(names, cls)
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
        used = fit_slots(names, cls)
        data = rows.get(cls, [])
        if not data:
            report.append((cls, 0, None, None))
            continue
        X = np.array([d[0][used] for d in data])
        t = np.array([d[1] for d in data])
        for j in [used.index(j) for j in used if f"{cls}.{names[cls][j]}" in drop]:
            X[:, j] = 0.0  # a dropped feature: its parameter fits to 0
        kl_full = None
        if is_k4(cls):
            theta = fit_k4(X, t, used.index(K4_BYTE), None if shared is None else [shared],
                           k4_round_monotone(names[cls], used))[0]
        elif cls == "kl" and kl_chain:
            Xf = np.array([d[0] for d in data])
            kl_full, _ = fit_kl_chain(Xf, t, [d[2].split()[0] for d in data], verbose)
            theta = kl_full[used]
        else:
            theta = nnls_relative(X, t)
        for j, v in zip(used, theta):
            params[f"{cls}.{names[cls][j]}"] = float(v) * 1e9  # s -> ns per unit
        if kl_full is not None:
            params["kl.chain"] = float(kl_full[KL_CHAIN]) * 1e9
        pdict = {f"{cls}.{names[cls][j]}": v for j, v in zip(used, theta)}
        if kl_full is not None:
            pdict["kl.chain"] = float(kl_full[KL_CHAIN])
        pred = np.array([predicted_ns(pdict, names, cls, d[0]) for d in data])
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
    if seed.file_format == 1:  # a format-1 calibration has none of the format-2 parameters
        for k in host.FORMAT2_PARAMS:
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
        used = fit_slots(names, cls)
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


# the v1.10 chain parameters (format 2; --no-chain drops them); v1.17's kf.elem / pack.dram
# are format-2 parameters but not chains of v1.10
NEW_CHAINS = tuple(sorted(host.FORMAT2_PARAMS - {"kf.elem", "pack.dram", "kl.chain"}))


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


def seed_extra(lib, kernels=None, fref=0.0, base=None, pack_l2=0.0, planes=False):
    """The format-2 keys of a seed calibration: the occupancy table (measured, else the
    built-in one), the reference clock without a power model, no cache term, bands 1, and
    (v1.17) L_hit of the PACK hit rule (pack_l2 bytes; it shapes the features, so the seed
    carries it from the start); v1.18: planes = the PACK set counts K4's W planes
    (warm.pack_l2_planes)."""
    ref = host.Calibration.default(lib)
    ex = {f"kernel.{n}": v for n, v in ref.kernels.items()}
    if kernels:
        for n, v in kernels.items():
            ex[f"kernel.{n}"] = v
    ex.update({"power.clock_eff_mhz": fref, "power.clock_min_mhz": fref, "power.cap_w": 0.0,
               "power.idle_w": 0.0, "power.tc_pj": 0.0, "power.alu_pj": 0.0,
               "power.byte_pj": 0.0, "power.gamma": 2.0,
               "warm.l2_bytes": 0.0, "warm.l2byte": 0.0, "rank.tie_ratio": 1.0,
               "warm.pack_l2_bytes": float(pack_l2),
               "warm.pack_l2_planes": 1.0 if planes else 0.0, "warm.call_ns": 0.0,
               "warm.clean_byte": 0.0, "power.act_w": 0.0, "power.act_gap_us": 0.0,
               "power.dram_pj": 0.0, "power.clock_boost_mhz": 0.0})
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

def window_capped(r):
    """v1.18: whether a sustained window touched the power cap (clock-event reason 0x4) in any
    trace sample (else in its post-window gpu_after reasons)."""
    tr = r.get("trace") or []
    if tr:
        return any(int(row[5]) & 0x4 for row in tr if len(row) > 5)
    g = r.get("gpu_after", "").replace(",", " ").split()
    try:
        return bool(int(g[-1], 16) & 0x4)
    except (IndexError, ValueError):
        return True


def power_probe(cal, gap_us, lib=None):
    """v1.18: a calibration whose power_w is one C-model quantity per call: the busy share b
    (act_w = 1 W, launch gap gap_us; regime BURST at f = f_ref) or the DRAM bytes x 1e-12 / t
    (dram_pj = 1). perf_fit reads the window features from it, so no formula lives here."""
    ex = dict(cal.extra)
    ex.update({"power.clock_eff_mhz": 2500.0, "power.clock_min_mhz": 2500.0, "power.cap_w": 1e9,
               "power.idle_w": 0.0, "power.tc_pj": 0.0, "power.alu_pj": 0.0,
               "power.byte_pj": 0.0, "power.byte_f_pj": 0.0, "power.gamma": 1.0,
               "power.clock_boost_mhz": 0.0, "power.act_w": 0.0, "power.act_gap_us": 0.0,
               "power.dram_pj": 0.0})
    busy = host.Calibration.from_values(cal.meta, cal.params, lib, dict(
        ex, **{"power.act_w": 1.0, "power.act_gap_us": float(gap_us)}))
    dram = host.Calibration.from_values(cal.meta, cal.params, lib, dict(ex, **{"power.dram_pj": 1.0}))
    return busy, dram


def sustained_rows(records, cal, lib=None, gap_us=1.0):
    """Per method-sustained record: (case, N, measured s per call, mean power W, mean SM clock,
    tensor-core ops, ALU ops, DRAM bytes, power limit W, label). The op and byte counts come
    from the C model's phases (perf.cpp call_power uses the same selection). v1.18: the mode
    (DIRECT windows: " mode=direct" in the variant), whether the window touched the cap, and
    from the C model the busy share b at the launch gap gap_us and the DRAM-energy bytes."""
    out = []
    busy_cal, dram_cal = power_probe(cal, gap_us, lib)
    for r in records:
        if r.get("method") != "sustained" or r.get("status") != "ok":
            continue
        ext = parse_ext(r["ext"])
        v = r.get("variant", "")
        kw = dict(N=int(r["N"]), g=int(r.get("g", 0) or 0),
                  vec_mode="RESIDENT" if "vec=resident" in v else "GENERATE",
                  mode="DIRECT" if "mode=direct" in v else "EMULATE", lib=lib)
        p = host.predict(r["spec"], ext, regime="BURST", cache="REPEAT", calib=cal, **kw)
        if p.status != "SUCCESS":
            continue
        tc = sum(q.ops for q in p.phases if q.cls.startswith("k4_"))
        alu = sum(q.x[2] for q in p.phases if q.cls in ("kf", "kl", "pack", "k5", "direct"))
        D = sum(q.bytes for q in p.phases)
        pb = host.predict(r["spec"], ext, regime="BURST", cache="COLD", calib=busy_cal, **kw)
        pd = host.predict(r["spec"], ext, regime="BURST", cache="COLD", calib=dram_cal, **kw)
        out.append(dict(case=r["case"], N=int(r["N"]), t=float(r["median_s"]),
                        P=float(r["power_w_mean"]), f=float(r["sm_mhz_mean"]), tc=tc, alu=alu,
                        D=D, cap=float(r.get("power_limit_w") or 450.0),
                        label=f"{r['case']} {v} N={r['N']}", spec=r["spec"], ext=ext,
                        g=kw["g"], vec=kw["vec_mode"], mode=kw["mode"],
                        busy=float(pb.power_w), Ddram=float(pd.power_w) * pd.time_s * 1e12,
                        capped=window_capped(r)))
    return out


def boost_clock(rows):
    """v1.18: the boost clock of the power model's solve: the median NVML clock of the sustained
    windows that never touched the power cap (0 if there is none)."""
    f = [r["f"] for r in rows if not r["capped"]]
    return float(statistics.median(f)) if f else 0.0


def fold_power_kw(power_kw, train):
    """v1.18 review (DESIGN §6c.8): the fit_power arguments of one fold. With boost_auto (no
    --clock-boost-mhz) the boost clock is the training windows' own boost_clock, so a left-out
    uncapped window never sets the clock it is predicted at; a given boost stays fixed."""
    kw = dict(power_kw)
    if kw.pop("boost_auto", False):
        kw["boost"] = boost_clock(train)
    return kw


POWER_FORMS = ("busy", "scaled", "unscaled")


def fit_power(rows, fref, gammas=None, idle=60.0, form="busy", byte_pj=0.0, gap_us=1.0,
              boost=0.0):
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
      form "busy" (v1.18, DESIGN §6c.8, the default): the scaled form + P_act b (f / fref)^(1 +
        gamma) + e_dram D_dram 1e-12 / t (b and D_dram from the C model, sustained_rows); fitted
        e_tc, e_alu, e_byte^f, P_act, e_dram; the launch gap gap_us and the boost clock `boost`
        (0: none; then the solve stops at fref) are written as given; gammas default (2.5,), a
        prior.
    Returns the power.* keys (cap = the records' enforced limit, clock_min = 0.8 x the lowest
    measured; power.byte_f_pj only for the scaled / busy forms, the v1.18 keys only for busy)."""
    if form not in POWER_FORMS:
        raise ValueError(f"power form {form!r}: one of {POWER_FORMS}")
    scaled = form in ("scaled", "busy")
    busy = form == "busy"
    if gammas is None:
        gammas = (2.5,) if busy else (1.0,) if scaled else (1.0, 2.0, 3.0)
    best = None
    P = np.array([r["P"] for r in rows])
    fixed = np.array([byte_pj * r["D"] * 1e-12 / r["t"] for r in rows]) if scaled else 0.0
    for gm in gammas:
        X = np.array([[r["tc"] * (r["f"] / fref) ** gm * 1e-12 / r["t"],
                       r["alu"] * (r["f"] / fref) ** gm * 1e-12 / r["t"],
                       r["D"] * ((r["f"] / fref) ** gm if scaled else 1.0) * 1e-12 / r["t"]] +
                      ([r["busy"] * (r["f"] / fref) ** (1 + gm), r["Ddram"] * 1e-12 / r["t"]]
                       if busy else [])
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
    out.update({"power.act_w": float(th[3]) if busy else 0.0,
                "power.act_gap_us": float(gap_us) if busy and th[3] > 0 else 0.0,
                "power.dram_pj": float(th[4]) if busy else 0.0,
                "power.clock_boost_mhz": float(boost) if busy and boost >= fref else 0.0})
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
        pw = fit_power(train, fref, idle=idle, **fold_power_kw(power_kw, train))
        cal = host.Calibration.from_values(meta, params, lib, dict(extra, **pw))
        for r in rows:
            if r["case"] != case:
                continue
            p = host.predict(r["spec"], r["ext"], N=r["N"], g=r["g"], vec_mode=r["vec"],
                             mode=r.get("mode", "EMULATE"), regime="SUSTAINED", cache="REPEAT",
                             calib=cal, lib=lib)
            out.append((r["label"], r["t"], p.time_s, r["f"], p.clock_mhz, r["P"], p.power_w))
    if verbose and out:
        print(f"{'sustained window (leave-one-case-out)':<44} {'TFLOPS-eq':>9} {'t meas':>10} "
              f"{'t pred':>10} {'err':>7} {'f meas':>7} {'f pred':>7} {'P meas':>7} {'P pred':>7}")
        for lab, tm, tp, fm, fp, pm, pp in out:
            print(f"{lab[:44]:<44} {'':>9} {1e3 * tm:>8.3f}ms {1e3 * tp:>8.3f}ms "
                  f"{100 * (tp - tm) / tm:>+6.1f}% {fm:>7.0f} {fp:>7.0f} {pm:>7.1f} {pp:>7.1f}")
    return out


def fref_scan(rows, meta, params, extra, fref, lib, idle, power_kw, step=25.0, count=23):
    """[(f, LOO median |t err|)] over the candidates fref, fref - step, ... (DESIGN §6c.1;
    v1.18: 23 candidates, down to 2255 MHz from the 4090's 2805)."""
    out = []
    for fr in [fref - step * i for i in range(count)]:
        val = power_validation(rows, meta, params, extra, fr, lib, verbose=False, idle=idle,
                               **power_kw)
        if val:
            out.append((fr, float(np.median([abs(tp - tm) / tm for _, tm, tp, *_ in val]))))
    return out


def fref_select(scan, rule="vertex", half_window=100.0):
    """v1.17 (DESIGN §6c.1): the effective clock from a LOO scan. "argmin": the candidate with
    the smallest error (v1.10). "vertex": the vertex of the least-squares parabola through the
    candidates within +-half_window of the argmin, clamped to them; the argmin if there are
    fewer than 5 such candidates or the parabola is not convex."""
    f0 = min(scan, key=lambda q: q[1])[0]
    if rule == "argmin":
        return f0
    pts = [(f, e) for f, e in scan if abs(f - f0) <= half_window + 1e-9]
    if len(pts) < 5:
        return f0
    f = np.array([q[0] for q in pts]) - f0
    e = np.array([q[1] for q in pts])
    a, b, _ = np.polyfit(f, e, 2)
    if not a > 0:
        return f0
    return float(round(f0 + float(np.clip(-b / (2 * a), f.min(), f.max()))))  # whole MHz


def nested_fref_validation(rows, meta, params, extra, fref, lib, idle, power_kw, rule):
    """v1.17: the honest figure of the f_ref choice. Per case: the scan and the rule on the
    other cases (their own leave-one-out), the power fit on them at that f_ref, then the case's
    SUSTAINED prediction. [(label, t meas, t pred, f meas, f pred, P meas, P pred, fold f_ref)]."""
    out = []
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        train = [r for r in rows if r["case"] != case]
        if len({r["case"] for r in train}) < 5:
            continue
        scan = fref_scan(train, meta, params, extra, fref, lib, idle, power_kw)
        if not scan:
            continue
        fr = fref_select(scan, rule)
        pw = fit_power(train, fr, idle=idle, **fold_power_kw(power_kw, train))
        cal = host.Calibration.from_values(meta, params, lib, dict(extra, **pw))
        for r in rows:
            if r["case"] == case:
                p = host.predict(r["spec"], r["ext"], N=r["N"], g=r["g"], vec_mode=r["vec"],
                                 mode=r.get("mode", "EMULATE"), regime="SUSTAINED",
                                 cache="REPEAT", calib=cal, lib=lib)
                out.append((r["label"], r["t"], p.time_s, r["f"], p.clock_mhz, r["P"],
                            p.power_w, fr))
    return out


def class_error(samples, cls, lib=None, fref=0.0):
    """(phases, median, p90, relative SSE) of one non-K4 class's NNLS fit on these samples."""
    names = host.param_names(lib)
    data = phase_rows(samples, fref).get(cls, [])
    if not data:
        return 0, 0.0, 0.0, 0.0
    used = fit_slots(names, cls)
    X = np.array([d[0][used] for d in data])
    t = np.array([d[1] for d in data])
    th = nnls_relative(X, t)
    err = np.abs(X @ th - t) / t
    return len(data), float(np.median(err)), float(np.percentile(err, 90)), float(np.sum(err ** 2))


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


def fit_warm(cases, meta, params, extra, l2, lib=None, min_effect=0.03, verbose=True,
             form="clean", terms=("delta", "c", "omega")):
    """L_w (a scan over fractions of the nominal L2) and Delta >= 0 (least squares, relative):
    measured t_state - t_repeat against the model's (res_repeat - res_state) bytes, which the C
    model gives as its COLD minus REPEAT time at a unit Delta. Delta = 0 when
    no measured |COLD - REPEAT| or |WARM - REPEAT| reaches min_effect of the call.
    v1.18 (DESIGN §6c.8), form "clean" (the default): method pair's COLD is the CLEAN state;
    L_w over 0.30-1.20 x L2 (steps of 0.05) and (Delta, c, omega) >= 0 by exact NNLS of the
    measured (CLEAN - REPEAT) / REPEAT against the C model's columns (COLD - REPEAT) at unit
    Delta, at unit c, and (CLEAN - COLD) at unit omega. form "v110": the v1.10 fit."""
    if form == "clean":
        return fit_warm_clean(cases, meta, params, extra, l2, lib, min_effect, verbose, terms)
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


def pair_effects(cases, verbose=True):
    effects = []
    for (case, spec, ext, N), t in sorted(cases.items()):
        effects.append((case, N, t["cold"] / t["repeat"] - 1, t["warm"] / t["repeat"] - 1, t))
    if verbose:
        print(f"{'cache pair case':<20} {'N':>3} {'repeat':>10} {'clean vs repeat':>16} "
              f"{'warm vs repeat':>15}")
        for case, N, ec, ew, t in effects:
            print(f"{case:<20} {N:>3} {1e3 * t['repeat']:>8.4f}ms {100 * ec:>+15.1f}% "
                  f"{100 * ew:>+14.1f}%")
    return effects


def cache_columns(cases, meta, params, extra, lw, lib=None):
    """v1.18: per pair case, the C model's (COLD - REPEAT) at unit Delta (ns/B), at unit c (ns)
    and (CLEAN - COLD) at unit omega (ns/B), relative to the measured REPEAT time, at L_w = lw
    bytes; and the measured (CLEAN - REPEAT) / REPEAT."""
    units = (1e-6, 1.0, 1e-6)
    cals = []
    for i, key in enumerate(("warm.l2byte", "warm.call_ns", "warm.clean_byte")):
        ex = dict(extra, **{"warm.l2_bytes": float(lw), "warm.l2byte": 0.0, "warm.call_ns": 0.0,
                            "warm.clean_byte": 0.0})
        ex[key] = units[i]
        cals.append(host.Calibration.from_values(meta, params, lib, ex))
    A, y, keys = [], [], []
    for (case, spec, ext, N), t in sorted(cases.items()):
        e = parse_ext(ext)

        def pr(cal, st):
            return host.predict(spec, e, N=N, cache=st, regime="BURST", calib=cal, lib=lib).time_s
        row = [(pr(cals[0], "COLD") - pr(cals[0], "REPEAT")) / units[0],
               (pr(cals[1], "COLD") - pr(cals[1], "REPEAT")) / units[1],
               (pr(cals[2], "CLEAN") - pr(cals[2], "COLD")) / units[2]]
        A.append([v / t["repeat"] for v in row])
        y.append((t["cold"] - t["repeat"]) / t["repeat"])
        keys.append((case, N))
    return np.array(A), np.array(y), keys


def nnls_plain(A, y):
    """argmin_{theta >= 0} ||A theta - y||^2 by enumerating the active sets (few columns)."""
    best = (float(np.sum(y ** 2)), np.zeros(A.shape[1]))
    for r in range(1, A.shape[1] + 1):
        for S in itertools.combinations(range(A.shape[1]), r):
            sol, *_ = np.linalg.lstsq(A[:, S], y, rcond=None)
            if np.any(sol < 0):
                continue
            th = np.zeros(A.shape[1])
            th[list(S)] = sol
            err = float(np.sum((A @ th - y) ** 2))
            if err < best[0] - 1e-18:
                best = (err, th)
    return best


def fit_warm_clean(cases, meta, params, extra, l2, lib=None, min_effect=0.03, verbose=True,
                   terms=("delta", "c", "omega")):
    """v1.18 (DESIGN §6c.8): see fit_warm. terms: the fitted subset of (delta, c, omega); the
    others are 0 (an A/B of a term, perf_fit --cache-terms)."""
    keep = [i for i, n in enumerate(("delta", "c", "omega")) if n in terms]
    effects = pair_effects(cases, verbose)
    big = max((max(abs(ec), abs(ew)) for _, _, ec, ew, _ in effects), default=0.0)
    zero = {"warm.l2_bytes": 0.0, "warm.l2byte": 0.0, "warm.call_ns": 0.0, "warm.clean_byte": 0.0}
    if big < min_effect:
        if verbose:
            print(f"cache effect below {100 * min_effect:.0f} % everywhere: no cache term")
        return zero, effects
    best = None
    for frac in np.round(np.arange(0.30, 1.2001, 0.05), 2):
        A, y, _ = cache_columns(cases, meta, params, extra, frac * l2, lib)
        err, th_k = nnls_plain(A[:, keep], y)
        th = np.zeros(3)
        th[keep] = th_k
        if best is None or err < best[0] - 1e-15:
            best = (err, float(frac), th)
    err, frac, (delta, c, om) = best
    if verbose:
        print(f"cache term: L_w = {frac:.2f} x L2, Delta {delta:.4g} ns per byte, c {c:.4g} ns per "
              f"call, omega {om:.4g} ns per byte (relative SSE {err:.4f} over {len(cases)} pairs)")
    return {"warm.l2_bytes": frac * l2, "warm.l2byte": float(delta), "warm.call_ns": float(c),
            "warm.clean_byte": float(om)}, effects


def kl_drop_columns(samples, cal0, fref=0.0):
    """v1.19 (§6c.9): per KL phase of a measured (REPEAT-state) sample, its features x (the clock
    normalization of the fit applied), the measured time m and the model's REPEAT saving
    credit = COLD - REPEAT of that phase under cal0 (warm.kl_drop 0). Method packed's samples are
    COLD already and skipped, as the raising of the fit skips them."""
    X, m, cu, lab = [], [], [], []
    for s in samples:
        if " pack " in s.label:
            continue
        cold, rep = s.again(cal0, "COLD"), s.again(cal0, "REPEAT")
        for qc, qr in zip(cold.phases, rep.phases):
            if qc.cls != "kl" or qc.name not in s.measured or s.measured[qc.name] <= 0:
                continue
            x = np.array(qr.x)
            if fref > 0 and s.clock > 0:
                for j in CLOCK_BOUND["kl"]:
                    x[j] *= fref / s.clock
            X.append(x)
            m.append(s.measured[qc.name])
            cu.append(qc.time_s - qr.time_s)
            lab.append(f"{s.label} {qc.name}")
    return np.array(X), np.array(m), np.array(cu), lab


def fit_kl_drop(samples, cal0, names, fref=0.0, grid=None, verbose=True):
    """v1.19 (§6c.9): the share d of the REPEAT / WARM saving that KL reads do not get
    (warm.kl_drop). For each d of the grid the KL class is refitted (exact NNLS, relative) on the
    phase marks raised to COLD with (1 - d) x the model's saving, and the REPEAT-state
    prediction theta . x - (1 - d) credit of every KL phase is scored against its mark (relative
    squared error); the d with the smallest sum is kept (0 when no KL phase has a saving)."""
    X, m, cu, lab = kl_drop_columns(samples, cal0, fref)
    if len(m) == 0 or not np.any(cu > 0):
        if verbose:
            print("KL credit: no KL phase with a modeled saving: warm.kl_drop = 0")
        return 0.0
    used = [j for j, n in enumerate(names["kl"]) if n]
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2) if grid is None else grid
    best = None
    has = cu > 0
    if verbose:
        print(f"KL credit scan (warm.kl_drop): {len(m)} KL phases, {int(has.sum())} with a saving")
        print(f"  {'d':>5} {'rel SSE':>9} {'median|err|':>12} {'p90|err|':>9}  (REPEAT-state KL phases; "
              f"with saving: median, p90)")
    for d in grid:
        th = np.zeros(X.shape[1])
        th[used] = nnls_relative(X[:, used], m + (1 - d) * cu)
        e = (X @ th - (1 - d) * cu - m) / m
        sse = float(np.sum(e ** 2))
        if verbose:
            print(f"  {d:5.2f} {sse:9.3f} {100 * np.median(np.abs(e)):11.1f}% "
                  f"{100 * np.percentile(np.abs(e), 90):8.1f}%  "
                  f"{100 * np.median(np.abs(e[has])):6.1f}% {100 * np.percentile(np.abs(e[has]), 90):6.1f}%")
        if best is None or sse < best[0] - 1e-12:
            best = (sse, float(d))
    if verbose:
        print(f"KL credit: warm.kl_drop = {best[1]:.2f} (relative SSE {best[0]:.3f})")
    return best[1]


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
    ap.add_argument("--power-form", choices=POWER_FORMS, default="busy",
                    help="power model: 'busy' (v1.18 default: the scaled form + busy-GPU power "
                         "power.act_w, DRAM energy power.dram_pj and the boost clock; gamma 2.5), "
                         "'scaled' (v1.15: clock-scaled byte energy power.byte_f_pj fitted, "
                         "power.byte_pj held at --byte-pj) or 'unscaled' (the v1.10 fit of "
                         "power.byte_pj, no byte_f key)")
    ap.add_argument("--act-gap-us", type=float, default=1.0,
                    help="busy form: the launch gap of the busy share (us per launch; a prior)")
    ap.add_argument("--clock-boost-mhz", type=float, default=None,
                    help="busy form: the boost clock of the solve (default: the median clock of "
                         "the sustained windows that never touched the power cap; 0: none)")
    ap.add_argument("--cache-form", choices=("clean", "v110"), default="clean",
                    help="cache fit: 'clean' (v1.18: Delta, per-call c, clean credit omega; L_w "
                         "0.30-1.20 x L2) or 'v110' (Delta only, L_w 0.1-1.0 x L2)")
    ap.add_argument("--cache-terms", default="delta,omega",
                    help="clean cache fit: the fitted subset of delta,c,omega (the others 0). "
                         "Default delta,omega: the per-call REPEAT saving c is not identifiable "
                         "from the pair records (NOTES-B2 §12 \"Decision: c dropped\"); "
                         "delta,c,omega is the perf-minor fit")
    ap.add_argument("--kl-drop", default="0",
                    help="v1.19: warm.kl_drop, the share of the REPEAT / WARM saving that KL reads "
                         "do not get: a number in [0, 1] (the default 0: the key is absent, the "
                         "v1.18 rule, so a routine recalibration cannot ship a nonzero key) or "
                         "'scan' (opt in: fitted against the KL phase marks)")
    ap.add_argument("--kl-chain", action="store_true",
                    help="v1.20 (DESIGN §6c.10): fit the KL class with the chain arm "
                         "max(byte arm, chain arm) and write kl.chain if its leave-one-shape-out "
                         "error beats the plain fit (opt in: a routine recalibration never ships "
                         "it); format 2 only")
    ap.add_argument("--byte-pj", type=float, default=0.0,
                    help="scaled form: the fixed unscaled byte energy (pJ per DRAM byte; 0, or "
                         "a stated prior), written as power.byte_pj")
    ap.add_argument("--power-gammas", default=None,
                    help="comma-separated gammas scanned by the power residual (default: 2.5, a "
                         "prior, for the busy form, 1 for the scaled form, 1,2,3 for the "
                         "unscaled form)")
    ap.add_argument("--no-chain", action="store_true",
                    help="drop the v1.10 chain features of K0, KL, PACK, K5, DIRECT, BETA")
    ap.add_argument("--drop", action="append", default=[], metavar="CLASS.FEATURE",
                    help="force a parameter to 0 (its feature column dropped from the fit; for "
                         "an A/B of a feature, e.g. --drop kf.elem)")
    ap.add_argument("--kernel", action="append", default=[], metavar="NAME=R",
                    help="override a kernel's CTAs per SM (e.g. kf4=4, the v1.8 built-in)")
    # v1.17 (DESIGN §6c.1, §6c.2)
    ap.add_argument("--pack-l2-frac", type=float, default=None,
                    help="L_hit of the PACK hit rule (warm.pack_l2_bytes) as a fraction of the "
                         "records' nominal L2 (default 1.1 with the W planes in S, v1.18: 83 MB "
                         "on the RTX 4090; 0.8 with --no-pack-planes; 0: no hit rule)")
    ap.add_argument("--no-pack-planes", action="store_true",
                    help="the v1.17 PACK set S without K4's W planes (warm.pack_l2_planes 0)")
    ap.add_argument("--fref-rule", choices=("vertex", "argmin"), default="vertex",
                    help="effective clock from the sustained LOO scan: the vertex of a parabola "
                         "within +-100 MHz of the argmin (v1.17 default) or the argmin (v1.10)")
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
    if a.kl_drop != "scan":
        try:
            a.kl_drop = float(a.kl_drop)
        except ValueError:
            ap.error("--kl-drop must be 'scan' or a number in [0, 1]")
        if not 0 <= a.kl_drop <= 1:
            ap.error("--kl-drop must be 'scan' or a number in [0, 1]")
    if a.byte_pj < 0 or not math.isfinite(a.byte_pj):
        ap.error("--byte-pj must be a finite number >= 0")
    if a.byte_pj != 0 and a.power_form == "unscaled":
        ap.error("--byte-pj is the scaled / busy forms' fixed term")
    if a.pack_l2_frac is None:
        a.pack_l2_frac = 0.8 if a.no_pack_planes else 1.1
    power_kw = dict(form=a.power_form, byte_pj=a.byte_pj,
                    gammas=tuple(float(x) for x in a.power_gammas.split(","))
                    if a.power_gammas else None)
    if a.power_form == "busy":
        power_kw["gap_us"] = a.act_gap_us
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
    if not 0 <= a.pack_l2_frac <= 4:
        ap.error("--pack-l2-frac must be in [0, 4]")
    extra = None if a.format1 else seed_extra(a.lib, kernels, fref, pack_l2=a.pack_l2_frac * l2,
                                              planes=not a.no_pack_planes)
    if extra is not None:
        print(f"reference clock {fref:.0f} MHz; occupancy: " +
              ", ".join(f"{n} {extra['kernel.' + n][0]}" for n in
                        ("k0", "kf16", "kf4", "kl8", "kl4", "pack", "k5_n14_e4", "k5b_n14")))
    params0 = base_params(a.lib) if extra is not None else host.Calibration.default(a.lib).params
    if a.kl_chain:
        if extra is None:
            ap.error("--kl-chain needs calibration format 2")
        # the features of the seed's samples: x[4] of a KL phase is filled only for a seed with
        # kl.chain > 0; this value makes the chain arm negligible (never binding) for them
        params0 = dict(params0, **{"kl.chain": 1e-9})
    drop = (NEW_CHAINS if a.no_chain else ()) + tuple(a.drop)
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
        wm, _ = fit_warm(cases, meta, params0, extra, l2, a.lib, form=a.cache_form,
                         terms=tuple(a.cache_terms.split(",")))
        extra.update(wm)
        if wm["warm.l2byte"] > 0 and not a.format1:
            # v1.19 (§6c.9): the KL share of the saving, against the KL phase marks, with the
            # cache term fitted above (Delta, L_w) and the KL class refitted at each d
            if a.kl_drop == "scan":
                cal0 = host.Calibration.from_values(meta, params0, a.lib, extra)
                kd = fit_kl_drop(samples, cal0, host.param_names(a.lib), fnorm)
            else:
                kd = a.kl_drop
            if kd > 0:
                extra["warm.kl_drop"] = kd
        if wm["warm.l2byte"] > 0 or wm.get("warm.call_ns", 0) > 0:
            calw = host.Calibration.from_values(meta, params0, a.lib, extra)
            added = []
            for s in samples:
                # v1.17: a flushed record's phases are REPEAT state (imd_bench flushes before
                # the timed calls only, not the phase reps), so they are raised like the others
                if " pack " in s.label:
                    continue  # measured COLD already (method packed's interleaving)
                cold, rep = s.again(calw, "COLD"), s.again(calw, "REPEAT")
                for qc, qr in zip(cold.phases, rep.phases):
                    if qc.name in s.measured and qc.time_s != qr.time_s:
                        s.measured[qc.name] += qc.time_s - qr.time_s
                added.append((cold.time_s - rep.time_s) / s.timed_s)
            print(f"cache term: REPEAT phases raised to COLD by {100 * np.median(added):.1f} % "
                  f"(median), {100 * max(added):.1f} % (max) of the call")
        seed = host.Calibration.from_values(meta, seed.params, a.lib, extra)
    if extra is not None and extra["warm.pack_l2_bytes"] > 0:
        # v1.17 diagnostic (no selection): the PACK class's fitted error against L_hit
        print(f"PACK L2-hit rule: L_hit = {a.pack_l2_frac:.2f} x {l2 / 2**20:.0f} MiB = "
              f"{extra['warm.pack_l2_bytes'] / 1e6:.1f} MB, S "
              f"{'without' if a.no_pack_planes else 'with'} K4's W planes; PACK fit error by "
              f"L_hit (diagnostic):")
        for frac in (0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3):
            sd = host.Calibration.from_values(meta, seed.params, a.lib,
                                              dict(extra, **{"warm.pack_l2_bytes": frac * l2}))
            n, med, p90, sse = class_error(samples_from_records(records, a.lib, sd), "pack",
                                           a.lib, fnorm)
            print(f"  L_hit {frac:.2f} x L2 ({frac * l2 / 1e6:5.1f} MB): {n} phases, median "
                  f"{100 * med:.1f} %, p90 {100 * p90:.1f} %, relative SSE {sse:.3f}")
    params = fit(samples, seed, a.lib, worst=a.worst, k4_shared_dram=a.k4_shared_dram,
                 fref=fnorm, drop=drop, kl_chain=a.kl_chain)
    if extra is not None:
        rows = sustained_rows(records, host.Calibration.from_values(meta, params, a.lib, extra),
                              a.lib, gap_us=a.act_gap_us)
        if a.power_form == "busy":
            boost = boost_clock(rows) if a.clock_boost_mhz is None else a.clock_boost_mhz
            power_kw["boost"] = boost
            # v1.18 review: the LOO / nested folds recompute it from their training windows
            power_kw["boost_auto"] = a.clock_boost_mhz is None
            uncapped = sum(1 for r in rows if not r["capped"])
            print(f"boost clock of the power solve: {boost:.0f} MHz ("
                  f"{'given' if a.clock_boost_mhz is not None else 'median of the ' + str(uncapped) + ' uncapped windows; recomputed in every fold'})")
            if a.clock_boost_mhz is None and uncapped == 0:
                print("WARNING: no sustained window below the power cap: no boost clock (the "
                      "solve stops at f_ref) while the busy power P_act is still fitted; pass "
                      "--clock-boost-mhz or add uncapped windows (DESIGN §6c.8)", file=sys.stderr)
        if len(rows) >= 4 and fref > 0:
            # The time model's effective clock (DESIGN §6c.1): the burst calibration's large calls
            # already ran below the boost clock (the limiter acts within a timed loop), so the
            # clock that SUSTAINED scales from is scanned (leave-one-case-out median error of
            # the sustained windows), from the boost clock down.
            scan = fref_scan(rows, meta, params, extra, fref, a.lib, a.idle_w, power_kw)
            for f_, err in scan:
                print(f"  time-model clock {f_:.0f} MHz: sustained LOO median |err| {100 * err:.2f} %")
            fr = fref_select(scan, a.fref_rule) if a.clock_ref is None else fref
            print(f"time-model clock (f_ref, rule {a.fref_rule if a.clock_ref is None else 'given'}): "
                  f"{fr:.1f} MHz (scan argmin {min(scan, key=lambda q: q[1])[0]:.0f} MHz)")
            if a.clock_ref is None:
                nest = nested_fref_validation(rows, meta, params, extra, fref, a.lib, a.idle_w,
                                              power_kw, a.fref_rule)
                if nest:
                    errs = [abs(tp - tm) / tm for _, tm, tp, *_ in nest]
                    print(f"nested leave-one-case-out (f_ref chosen inside each fold, rule "
                          f"{a.fref_rule}): median |t err| {100 * float(np.median(errs)):.2f} %, "
                          f"folds' f_ref {min(q[-1] for q in nest):.0f}-{max(q[-1] for q in nest):.0f} MHz")
                    for lab, tm, tp, fm, fp, pm, pp, ff in nest:
                        print(f"    {lab[:44]:<44} {100 * (tp - tm) / tm:>+6.1f}%  f {fm:.0f} / "
                              f"{fp:.0f} MHz  P {pm:.0f} / {pp:.0f} W  (fold f_ref {ff:.0f})")
            pw = fit_power(rows, fr, idle=a.idle_w, **fold_power_kw(power_kw, rows))
            print(f"power model ({a.power_form} form" +
                  (f", power.byte_pj held at {a.byte_pj:g} pJ/B" if a.power_form != "unscaled" else "")
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
