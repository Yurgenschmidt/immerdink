#!/usr/bin/env python3
"""Re-fit (and validate) the profitability rule of the Hopper wgmma K4 backend (src/host/k4_wgmma.hpp kK4WgmmaProfit, docs/NOTES-22.md
"Profitability rule").

Input: one or more floor_report.jsonl files of the campaign's `duel.floor` experiment (rows `floor_row`: case, m, n, k, ratio_forced,
median_s_mma, median_s_forced; N = 14), as   card=path   arguments. The batch per K4 launch g is read from the case name
(`..g1_..` = 1, `..g4_..` = 4, else 14 = one launch).

    python3 -I tools/fit_k4_wgmma_profit.py h100=.../floor_report.jsonl h200=.../floor_report.jsonl [--sms 132] [--check]

The model (per K4 launch, T = ceil(m/128) * ceil(nc/256) * zc output tiles of the 128 x 256 CTA tile, kt = ceil(kc/128) k steps):
    delta_us = f + (T / sms) * (a + b * kt)        wgmma launch time minus mma.sync launch time (negative = wgmma faster)
    fitted by weighted least squares on the whole-call difference  median_s_forced - median_s_mma  (the other phases of the call
    cancel), summed over the call's launches, weights 1 / max(|delta|, 8 us)  (relative error, floored: the boundary rows count).
The decision (made in the header, restated and checked here with the header's constants):
    serve a launch iff  m, nc >= min_mn, kc >= min_kc (the measured envelope), T >= min_fill * sms  and  -delta_us >= max(min_gain_us, min_gain_rel * mma_us)
    with  mma_us = mma_wave_k_us * (T / sms) * kt  (an estimate of the mma.sync K4 time that ignores its per-wave fixed cost).
Prints the fit, the per-shape table, leave-one-shape-out, leave-one-family-out and card-transfer validation. --check exits 1 if the
header's constants differ from the fit by more than 5 % or a validation criterion fails.
"""
import json
import math
import re
import sys

import numpy as np

# The header's constants (keep in step with kK4WgmmaProfit[IMD_TABLE_SM90]).
HEADER = dict(launch_us=2.40, wave_us=-0.57, wave_k_us=-0.51, mma_wave_k_us=1.13, min_gain_us=0.75, min_gain_rel=0.10, min_fill=0.5, min_mn=512, min_kc=256)
N_MODULI = 14


def cdiv(a, b):
    return -(-a // b)


def launches(g, n=N_MODULI):
    z = []
    while n > 0:
        z.append(min(g, n))
        n -= g
    return z


def load(specs):
    rows = []
    for spec in specs:
        card, path = spec.split("=", 1)
        for line in open(path):
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("method") != "floor_row":
                continue
            mg = re.search(r"g(\d+)_n", r["case"])
            g = int(mg.group(1)) if mg else N_MODULI
            rows.append(dict(card=card, case=r["case"], m=r["m"], n=r["n"], k=r["k"], g=g, ratio=r["ratio_forced"],
                             dt=1e6 * (r["median_s_forced"] - r["median_s_mma"]), tm=1e6 * r["median_s_mma"]))
    return rows


def launch_tiles(r):
    kt = cdiv(r["k"], 128)
    return [(cdiv(r["m"], 128) * cdiv(r["n"], 256) * zc, kt) for zc in launches(r["g"])]


def features(r, sms):
    L = tw = tk = 0.0
    for T, kt in launch_tiles(r):
        L += 1
        tw += T / sms
        tk += T / sms * kt
    return [L, tw, tk]


def fit(rows, sms):
    X = np.array([features(r, sms) for r in rows])
    y = np.array([r["dt"] for r in rows])
    w = 1.0 / np.maximum(np.abs(y), 8.0)
    return np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]


def delta_us(co, T, kt, sms):
    return co[0] + (T / sms) * (co[1] + co[2] * kt)


def serves(co, hp, r, sms):
    """The header's rule with the constants (f, a, b) = co and the margins of `hp`: every launch of the call must be served."""
    if min(r["m"], r["n"]) < hp["min_mn"] or r["k"] < hp["min_kc"]:
        return False
    for T, kt in launch_tiles(r):
        if T < hp["min_fill"] * sms:
            return False
        mma = hp["mma_wave_k_us"] * (T / sms) * kt
        if -delta_us(co, T, kt, sms) < max(hp["min_gain_us"], hp["min_gain_rel"] * mma):
            return False
    return True


def main(argv):
    sms = 132
    check = False
    specs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--sms":
            sms = int(argv[i + 1])
            i += 2
        elif argv[i] == "--check":
            check = True
            i += 1
        else:
            specs.append(argv[i])
            i += 1
    if not specs:
        print(__doc__)
        return 2
    rows = load(specs)
    cases = sorted(set(r["case"] for r in rows))
    co = fit(rows, sms)
    print(f"{len(rows)} rows ({len(cases)} shapes x {len(set(r['card'] for r in rows))} cards), sms {sms}")
    print(f"fit: launch_us {co[0]:.3f}  wave_us {co[1]:.3f}  wave_k_us {co[2]:.3f}   (header: {HEADER['launch_us']} {HEADER['wave_us']} {HEADER['wave_k_us']})")
    hco = [HEADER["launch_us"], HEADER["wave_us"], HEADER["wave_k_us"]]
    ok = True
    for c, h in zip(co, hco):
        if abs(c - h) > 0.05 * abs(h) + 0.02:
            print(f"  constant differs from the header by more than 5 %: fit {c:.3f} header {h}")
            ok = False
    print("\nper shape (header constants; ratio = measured whole-call forced wgmma / mma.sync, min over cards):")
    print(f"  {'case':22s} {'T':>6s} {'kt':>3s} {'dhat/launch':>11s} {'meas dt/call':>12s} {'fit dt/call':>11s} {'ratio':>6s}  serve  loo")
    unsafe = missed = 0
    for cs in cases:
        rr = [r for r in rows if r["case"] == cs]
        r0 = rr[0]
        lt = launch_tiles(r0)
        mn = min(x["ratio"] for x in rr)
        s_full = serves(hco, HEADER, r0, sms)
        s_loo = serves(fit([r for r in rows if r["case"] != cs], sms), HEADER, r0, sms)
        dh = delta_us(hco, lt[0][0], lt[0][1], sms)
        fitdt = sum(delta_us(hco, T, kt, sms) for T, kt in lt)
        mdt = np.mean([x["dt"] for x in rr])
        flag = ""
        if s_full and mn < 1.02:
            flag += " UNSAFE"
            unsafe += 1
        if s_loo and mn < 1.02:
            flag += " LOO-UNSAFE"
            unsafe += 1
        if (not s_full) and mn >= 1.05:
            flag += " missed"
            missed += 1
        print(f"  {cs:22s} {lt[0][0]:6d} {lt[0][1]:3d} {dh:11.2f} {mdt:12.1f} {fitdt:11.1f} {mn:6.3f}  {int(s_full)}      {int(s_loo)}{flag}")
    print(f"\nleave-one-shape-out (both cards of the shape held out): unsafe {unsafe}, missed wins >= 5 % {missed} (a LOO-UNSAFE / UNSAFE flag fails the check)")
    fams = {
        "g=1 and g=4": lambda r: r["g"] < N_MODULI,
        "small (all of m,n,k <= 1024, one launch)": lambda r: r["g"] == N_MODULI and max(r["m"], r["n"], r["k"]) <= 1024,
        "rectangles": lambda r: r["case"].startswith("r"),
        "k <= 512": lambda r: r["k"] <= 512,
        "squares": lambda r: r["case"].startswith("sq"),
    }
    print("\nleave-one-family-out:")
    for name, f in fams.items():
        tr = [r for r in rows if not f(r)]
        te = [r for r in rows if f(r)]
        if len(tr) < 6 or not te:
            continue
        c2 = fit(tr, sms)
        bad = sorted(set(r["case"] for r in te if serves(c2, HEADER, r, sms) and r["ratio"] < 1.02))
        miss = sorted(set(r["case"] for r in te if not serves(c2, HEADER, r, sms) and r["ratio"] >= 1.05))
        print(f"  out {name:42s} fit {np.round(c2, 2)}  unsafe {bad}  missed {miss}")
        if bad:
            ok = False
    cards = sorted(set(r["card"] for r in rows))
    if len(cards) > 1:
        print("\ncard transfer (fit on one card, decide the other):")
        for a in cards:
            for b in cards:
                if a == b:
                    continue
                c2 = fit([r for r in rows if r["card"] == a], sms)
                bad = [r["case"] for r in rows if r["card"] == b and serves(c2, HEADER, r, sms) and r["ratio"] < 1.02]
                miss = [r["case"] for r in rows if r["card"] == b and not serves(c2, HEADER, r, sms) and r["ratio"] >= 1.05]
                print(f"  fit {a} -> {b}: {np.round(c2, 3)} unsafe {bad} missed {miss}")
                if bad:
                    ok = False
    # with the header's constants, every card's own measurement: nothing served below 1.02
    for r in rows:
        if serves(hco, HEADER, r, sms) and r["ratio"] < 1.02:
            print(f"UNSAFE on {r['card']}: {r['case']} ratio {r['ratio']}")
            ok = False
    if unsafe:
        ok = False
    print("\nCHECK", "PASS" if ok else "FAIL")
    return 0 if (ok or not check) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
