#!/usr/bin/env python3
"""Score the KL phases of imd records against the model (issue #7, docs/NOTES-C1.md §5.13).

    python3 bench/perf_score_kl.py records.jsonl [more.jsonl ...] [--calib file.calib] [--lib ...]
        [--ratio-calib file.calib] [--rows]

Per KL phase mark (KL A, KL B, pack KL; every call length, unlike perf_validate --phases, which
drops calls under 1 ms): the REPEAT-state prediction against the mark, signed and unsigned
relative error, grouped by
  narrow multi-pass : KL B / pack KL with wave >= 3 whose launch is under 128 CTAs (waves = 1, so
                      wave = nc); the cases of the sweeps' nc / x / g / side groups, by name
  other KL          : every other KL phase
and, for the narrow multi-pass group, by nc bin {3}, {4}, {>= 5}, g = N / g < N, and (with
--ratio-calib, a calibration with kl.chain) the chain arm / byte arm of that calibration, bins
[0, 0.5), [0.5, 2], (2, inf). A bin with fewer than 3 distinct shapes is printed "not scored".
Exit 0; the bars are read by hand against NOTES-C1 §5.13 (P1, P2, P5). CPU only.
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

KL_PHASES = ("KL A", "KL B", "pack KL")
# launches over 128 CTAs (waves > 1) or not narrow: by case name (sweep_klb_*.txt geometry)
WIDE = re.compile(r"^[hk]_(w\d+|x(3072|4096|6144|8192))$")
MULTI = re.compile(r"^[hk]_(nc\d+|x\d+|g|w\d+|anmp|batch|tensor|pack)$")


def narrow_mp(case, phase, wave):
    if phase.name not in ("KL B", "pack KL") or wave < 3:
        return False
    if WIDE.match(case):
        return False
    return bool(MULTI.match(case)) or case in ("q_kl1", "q_kl2", "q_kl4") or case.startswith("m_jac")


def rows_of(records, calib, ratio_calib, lib):
    out = []
    for s in perf_fit.samples_from_records(records, lib, calib):
        rep = s.again(calib, "REPEAT")
        rep_r = s.again(ratio_calib, "REPEAT") if ratio_calib else None
        for i, p in enumerate(rep.phases):
            if p.cls != "kl" or p.name not in KL_PHASES or p.name not in s.measured:
                continue
            m = s.measured[p.name]
            if m <= 0:
                continue
            ratio = None
            if rep_r is not None:
                th = ratio_calib.params
                byte = th["kl.byte"] * p.x[1]
                chain = th.get("kl.chain", 0.0) * rep_r.phases[i].x[4]
                ratio = chain / byte if byte > 0 else None
            out.append(dict(case=s.case, label=s.label, phase=p.name, wave=int(p.x[5]), bytes=p.x[1],
                            m=m, pred=p.time_s, err=(p.time_s - m) / m, ratio=ratio,
                            mp=narrow_mp(s.case, p, int(p.x[5]))))
    return out


def stats(rows):
    e = np.array([r["err"] for r in rows])
    shapes = len({r["case"] for r in rows})
    if len(rows) == 0:
        return "n=0"
    if shapes < 3:
        return f"n={len(rows)} shapes={shapes}  not scored (< 3 shapes)  median signed {100 * np.median(e):+.1f} %"
    return (f"n={len(rows)} shapes={shapes}  median|err| {100 * np.median(abs(e)):.1f} %  "
            f"p90|err| {100 * np.percentile(abs(e), 90):.1f} %  median signed {100 * np.median(e):+.1f} %")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--ratio-calib", default=None, help="calibration with kl.chain for the arm ratio bins")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--rows", action="store_true", help="print every row")
    a = ap.parse_args(argv)
    calib = host.Calibration.load(a.calib, a.lib) if a.calib else host.Calibration.default(a.lib)
    rcal = host.Calibration.load(a.ratio_calib, a.lib) if a.ratio_calib else None
    recs = []
    for f in a.records:
        recs += [r for r in map(json.loads, (l for l in open(f) if l.strip()))
                 if r.get("method") in ("imd", "packed") and r.get("status") == "ok"]
    rows = rows_of(recs, calib, rcal, a.lib)
    if a.rows:
        for r in rows:
            print(f"{r['label'][:44]:44} {r['phase']:8} wave={r['wave']:3} m={r['m'] * 1e6:7.1f} "
                  f"pred={r['pred'] * 1e6:7.1f} err={100 * r['err']:+6.1f} %"
                  + (f" ratio={r['ratio']:.2f}" if r["ratio"] is not None else "") + (" MP" if r["mp"] else ""))
    mp = [r for r in rows if r["mp"]]
    print("narrow multi-pass KL B :", stats(mp))
    print("other KL phases        :", stats([r for r in rows if not r["mp"]]))
    for name, lo, hi in (("nc 3", 3, 3), ("nc 4", 4, 4), ("nc >= 5", 5, 99)):
        print(f"  {name:<20}:", stats([r for r in mp if lo <= r["wave"] <= hi]))
    if rcal is not None:
        for name, lo, hi in (("arm ratio < 0.5", -1, 0.5), ("arm ratio 0.5 .. 2", 0.5, 2.0001), ("arm ratio > 2", 2.0001, 1e9)):
            print(f"  {name:<20}:", stats([r for r in mp if r["ratio"] is not None and lo <= r["ratio"] < hi]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
